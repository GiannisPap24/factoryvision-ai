"""Bounded scalar reports for internally generated probes; no filesystem or network I/O."""

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, fields
from enum import StrEnum

from factoryvision.inference.reporting import spreadsheet_cell
from factoryvision.robustness.assessment import (
    Crossing,
    FrozenThreshold,
    MapStability,
    MeasurementStatus,
    Reason,
    ScoreStability,
    ThresholdRule,
)
from factoryvision.robustness.contracts import (
    MAX_REPORT_BYTES,
    PROTOCOL_ID,
    PROTOCOL_VERSION,
    Applicability,
    Axis,
    Budgets,
    Category,
    Classification,
    Diagnostic,
    DiagnosticStatus,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
    _require,
    _tiles,
    _unique,
)

REPORT_KIND = "synthetic-robustness-diagnostics"
GENERATOR = "controlled-probes-v1"
ADAPTER = "analytical-rgb-mean-v1"


class RowStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class RunReason(StrEnum):
    NONE = "none"
    NO_METRICS = "no_applicable_metrics"
    ADAPTER_ERROR = "adapter_error"
    INVALID_OUTPUT = "invalid_output"
    TRANSFORM_ERROR = "transform_error"
    GENERATION_ERROR = "generation_error"
    ASSESSMENT_ERROR = "assessment_error"
    BASELINE_UNAVAILABLE = "baseline_unavailable"
    CANCELLED = "cancelled"


def units_for(plan, parent):
    return _tiles(parent) if plan.category is Category.METALPARTS else 1


def schedule(plan):
    """Explicit parent order; identity repetitions first, then each listed condition once."""
    for parent in plan.parents:
        ordinal = tuple(ParentID).index(parent.parent_id) + 1
        for index, spec in enumerate(plan.perturbations):
            for repetition in range(1, (plan.baseline_repetitions if index == 0 else 1) + 1):
                yield parent, ordinal, index, repetition, spec, units_for(plan, parent)


@dataclass(frozen=True, slots=True)
class Observation:
    case: int
    condition: int
    repetition: int
    reserved: int
    attempted: int
    status: RowStatus
    reason: RunReason
    score: ScoreStability | None = None
    map: MapStability | None = None

    def __post_init__(self):
        for value, low, high in (
            (self.case, 1, 4),
            (self.condition, 0, 45),
            (self.repetition, 1, 256),
            (self.reserved, 1, 128),
            (self.attempted, 0, self.reserved),
        ):
            _require(type(value) is int and low <= value <= high)
        _require(type(self.status) is RowStatus and type(self.reason) is RunReason)
        if self.status in (RowStatus.MEASURED, RowStatus.NOT_APPLICABLE):
            _require(self.attempted == self.reserved and type(self.score) is ScoreStability)
            self.score.__post_init__()
            if self.map is not None:
                _require(type(self.map) is MapStability)
                self.map.__post_init__()
            for metric in (self.score, self.map):
                if metric is not None:
                    _require(metric.classification is Classification.SOFTWARE_DIAGNOSTIC)
                    _require(
                        type(metric.protocol_version) is int
                        and metric.protocol_version == PROTOCOL_VERSION
                    )
            measured = self.score.status is MeasurementStatus.MEASURED or (
                self.map is not None and self.map.status is MeasurementStatus.MEASURED
            )
            _require((self.status is RowStatus.MEASURED) == measured)
            _require(self.reason is (RunReason.NONE if measured else RunReason.NO_METRICS))
        else:
            _require(self.score is None and self.map is None)
            if self.status is RowStatus.NOT_EVALUATED:
                _require(
                    self.attempted == 0
                    and self.reason in (RunReason.CANCELLED, RunReason.BASELINE_UNAVAILABLE)
                )
            else:
                _require(
                    self.reason
                    in (
                        RunReason.ADAPTER_ERROR,
                        RunReason.INVALID_OUTPUT,
                        RunReason.TRANSFORM_ERROR,
                        RunReason.GENERATION_ERROR,
                        RunReason.ASSESSMENT_ERROR,
                        RunReason.CANCELLED,
                    )
                )
                if self.reason in (RunReason.TRANSFORM_ERROR, RunReason.GENERATION_ERROR):
                    _require(self.attempted == 0)
                else:
                    _require(self.attempted > 0)
                if self.reason is RunReason.ASSESSMENT_ERROR:
                    _require(self.attempted == self.reserved)


@dataclass(frozen=True, slots=True)
class SyntheticReport:
    plan: Plan
    rows: tuple[Observation, ...]
    threshold: FrozenThreshold | None = None

    def __post_init__(self):
        _require(type(self.plan) is Plan)
        self.plan.__post_init__()
        if self.threshold is not None:
            _require(type(self.threshold) is FrozenThreshold)
            self.threshold.__post_init__()
        expected = tuple(schedule(self.plan))
        _require(type(self.rows) is tuple and len(self.rows) == len(expected))
        cancelled = False
        blocked = set()
        for row, (parent, ordinal, index, repeat, spec, units) in zip(
            self.rows, expected, strict=True
        ):
            _require(type(row) is Observation)
            row.__post_init__()
            _require(
                (row.case, row.condition, row.repetition, row.reserved)
                == (ordinal, index, repeat, units)
            )
            if cancelled:
                _require(
                    row.status is RowStatus.NOT_EVALUATED and row.reason is RunReason.CANCELLED
                )
            elif ordinal in blocked:
                _require(
                    row.status is RowStatus.NOT_EVALUATED
                    and row.reason in (RunReason.BASELINE_UNAVAILABLE, RunReason.CANCELLED)
                )
            else:
                _require(row.reason is not RunReason.BASELINE_UNAVAILABLE)
            cancelled |= row.reason is RunReason.CANCELLED
            if index == 0 and repeat == 1 and row.status is RowStatus.FAILED:
                blocked.add(ordinal)
            if row.status is not RowStatus.MEASURED:
                status = DiagnosticStatus(row.status.value)
                applicability = (
                    Applicability.NOT_APPLICABLE
                    if status is DiagnosticStatus.NOT_APPLICABLE
                    else Applicability.CONDITIONAL
                )
                Diagnostic(parent.parent_id, spec, applicability, status)
            if row.score is not None and row.score.status is MeasurementStatus.MEASURED:
                _require((row.score.crossing is None) == (self.threshold is None))
        _require(self.attempted_units <= self.plan.planned_inference_calls)

    @property
    def attempted_units(self):
        return sum(row.attempted for row in self.rows)


COLUMNS = (
    "report_kind",
    "schema_version",
    "protocol_id",
    "protocol_version",
    "provenance",
    "classification",
    "generator",
    "adapter",
    "category",
    "max_parents",
    "max_inference_calls",
    "max_source_pixels",
    "max_native_tiles",
    "max_report_bytes",
    "baseline_repetitions",
    "threshold_value",
    "threshold_rule",
    "planned_units",
    "attempted_units",
    "measured_rows",
    "failed_rows",
    "not_applicable_rows",
    "not_evaluated_rows",
    "case",
    "width",
    "height",
    "condition",
    "repetition",
    "kind",
    "severity",
    "axis",
    "seed",
    "status",
    "reason",
    "reserved",
    "attempted",
    "score_status",
    "score_reason",
    "signed",
    "absolute",
    "relative",
    "crossing",
    "map_status",
    "map_reason",
    "coverage",
    "mae",
    "rmse",
    "maximum_absolute",
)
INTEGER_COLUMNS = frozenset(
    (
        "schema_version",
        "protocol_version",
        "max_parents",
        "max_inference_calls",
        "max_source_pixels",
        "max_native_tiles",
        "max_report_bytes",
        "baseline_repetitions",
        "planned_units",
        "attempted_units",
        "measured_rows",
        "failed_rows",
        "not_applicable_rows",
        "not_evaluated_rows",
        "case",
        "width",
        "height",
        "condition",
        "repetition",
        "seed",
        "reserved",
        "attempted",
    )
)
FLOAT_COLUMNS = frozenset(
    (
        "threshold_value",
        "signed",
        "absolute",
        "relative",
        "coverage",
        "mae",
        "rmse",
        "maximum_absolute",
    )
)


def _records(report):
    counts = Counter(row.status.value for row in report.rows)
    result = []
    for row, (parent, _, _, _, spec, _) in zip(report.rows, schedule(report.plan), strict=True):
        score, maps = row.score, row.map
        record = dict(
            report_kind=REPORT_KIND,
            schema_version=1,
            protocol_id=PROTOCOL_ID,
            protocol_version=PROTOCOL_VERSION,
            provenance=report.plan.provenance,
            classification=report.plan.classification,
            generator=GENERATOR,
            adapter=ADAPTER,
            category=report.plan.category,
            **{f.name: getattr(report.plan.budgets, f.name) for f in fields(Budgets)},
            baseline_repetitions=report.plan.baseline_repetitions,
            threshold_value=None if report.threshold is None else report.threshold.value,
            threshold_rule=None if report.threshold is None else report.threshold.rule,
            planned_units=report.plan.planned_inference_calls,
            attempted_units=report.attempted_units,
            **{s.value + "_rows": counts[s.value] for s in RowStatus},
            case=row.case,
            width=parent.width,
            height=parent.height,
            condition=row.condition,
            repetition=row.repetition,
            kind=spec.kind,
            severity=spec.severity,
            axis=spec.axis,
            seed=spec.seed,
            status=row.status,
            reason=row.reason,
            reserved=row.reserved,
            attempted=row.attempted,
            score_status=None if score is None else score.status,
            score_reason=None if score is None else score.reason,
            signed=None if score is None else score.signed,
            absolute=None if score is None else score.absolute,
            relative=None if score is None else score.relative,
            crossing=None if score is None else score.crossing,
            map_status=maps.status
            if maps is not None
            else (MeasurementStatus.NOT_APPLICABLE if score is not None else None),
            map_reason=maps.reason
            if maps is not None
            else ("unavailable_output" if score is not None else None),
            coverage=None if maps is None else maps.coverage,
            mae=None if maps is None else maps.mae,
            rmse=None if maps is None else maps.rmse,
            maximum_absolute=None if maps is None else maps.maximum_absolute,
        )
        _require(set(record) == set(COLUMNS))
        result.append({key: record[key] for key in COLUMNS})
    return result


def _json(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cell(value):
    text = "null" if value is None else str(value)
    text = spreadsheet_cell(text)
    _require(len(text) <= 64)
    return text


def _csv(records):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in records:
        writer.writerow(_cell(row[column]) for column in COLUMNS)
    return buffer.getvalue().encode("utf-8")


def report_bound(plan: Plan) -> int:
    """Conservative byte bound for both encodings before generating any pixels.

    Every validated scalar's protected CSV cell is <=64 ASCII characters. The JSON
    bound quotes every scalar, including numbers/null, so it is conservative too.
    """
    _require(type(plan) is Plan)
    plan.__post_init__()
    count = sum(1 for _ in schedule(plan))
    worst = {c: "x" * 64 for c in COLUMNS}
    json_bound = (
        len(_json({"kind": REPORT_KIND, "schema_version": 1, "rows": []}))
        + count * len(_json(worst))
        + count
        - 1
    )
    csv_bound = len(_csv([])) + count * (len(_csv([worst])) - len(_csv([])))
    return max(json_bound, csv_bound)


def _bounded(raw, budget):
    _require(type(raw) is bytes and len(raw) <= min(budget, MAX_REPORT_BYTES))
    return raw


def json_bytes(report: SyntheticReport) -> bytes:
    _require(type(report) is SyntheticReport)
    report.__post_init__()
    records = _records(report)
    for row in records:
        for value in row.values():
            _cell(value)
    return _bounded(
        _json({"kind": REPORT_KIND, "schema_version": 1, "rows": records}),
        report.plan.budgets.max_report_bytes,
    )


def csv_bytes(report: SyntheticReport) -> bytes:
    _require(type(report) is SyntheticReport)
    report.__post_init__()
    return _bounded(_csv(_records(report)), report.plan.budgets.max_report_bytes)


def _parse_records(records):
    _require(type(records) is list and 1 <= len(records) <= 256)
    for row in records:
        _require(type(row) is dict and set(row) == set(COLUMNS))
        for value in row.values():
            _require(value is None or type(value) in (str, int, float))
    first = records[0]
    parents, conditions = {}, {}
    for row in records:
        ordinal = row["case"]
        _require(type(ordinal) is int and 1 <= ordinal <= 4)
        parents.setdefault(
            ordinal, Parent(tuple(ParentID)[ordinal - 1], row["width"], row["height"])
        )
        if ordinal == first["case"]:
            index = row["condition"]
            _require(type(index) is int and 0 <= index <= 45)
            conditions.setdefault(
                index,
                Perturbation(
                    Kind(row["kind"]),
                    row["severity"],
                    None if row["axis"] is None else Axis(row["axis"]),
                    row["seed"],
                ),
            )
    _require(tuple(conditions) == tuple(range(len(conditions))))
    plan = Plan(
        Category(first["category"]),
        tuple(parents.values()),
        tuple(conditions.values()),
        Budgets(**{f.name: first[f.name] for f in fields(Budgets)}),
        first["baseline_repetitions"],
    )
    threshold = (
        None
        if first["threshold_value"] is None
        else FrozenThreshold(first["threshold_value"], ThresholdRule(first["threshold_rule"]))
    )
    observations = []
    for r in records:
        status, reason = RowStatus(r["status"]), RunReason(r["reason"])
        score = maps = None
        if status in (RowStatus.MEASURED, RowStatus.NOT_APPLICABLE):
            score = ScoreStability(
                MeasurementStatus(r["score_status"]),
                Reason(r["score_reason"]),
                r["signed"],
                r["absolute"],
                r["relative"],
                None if r["crossing"] is None else Crossing(r["crossing"]),
            )
            if r["map_reason"] != "unavailable_output":
                maps = MapStability(
                    MeasurementStatus(r["map_status"]),
                    Reason(r["map_reason"]),
                    r["coverage"],
                    r["mae"],
                    r["rmse"],
                    r["maximum_absolute"],
                )
        observations.append(
            Observation(
                r["case"],
                r["condition"],
                r["repetition"],
                r["reserved"],
                r["attempted"],
                status,
                reason,
                score,
                maps,
            )
        )
    report = SyntheticReport(plan, tuple(observations), threshold)
    # Reconstruct every fixed field, status, count, repeated header and parameter.
    _require(_json(_records(report)) == _json(records))
    return report


def from_json(raw: bytes) -> SyntheticReport:
    try:
        _bounded(raw, MAX_REPORT_BYTES)
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
        _require(type(doc) is dict and set(doc) == {"kind", "schema_version", "rows"})
        _require(
            doc["kind"] == REPORT_KIND
            and type(doc["schema_version"]) is int
            and doc["schema_version"] == 1
        )
        report = _parse_records(doc["rows"])
        _bounded(raw, report.plan.budgets.max_report_bytes)
        return report
    except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
        raise RobustnessInvalid() from None


def from_csv(raw: bytes) -> SyntheticReport:
    try:
        _bounded(raw, MAX_REPORT_BYTES)
        reader = csv.reader(io.StringIO(raw.decode("utf-8"), newline=""), strict=True)
        _require(tuple(next(reader)) == COLUMNS)
        records = []
        for cells in reader:
            _require(len(records) < 256 and len(cells) == len(COLUMNS))
            row = {}
            for key, cell in zip(COLUMNS, cells, strict=True):
                _require(len(cell) <= 64)
                text = cell.removeprefix("[text] ")
                if text == "null":
                    value = None
                elif key in INTEGER_COLUMNS:
                    value = int(text)
                elif key in FLOAT_COLUMNS or key == "severity":
                    value = (
                        int(text) if key == "severity" and row["kind"] == Kind.JPEG else float(text)
                    )
                else:
                    value = text
                _require(_cell(value) == cell)
                row[key] = value
            records.append(row)
        report = _parse_records(records)
        _bounded(raw, report.plan.budgets.max_report_bytes)
        return report
    except (
        ValueError,
        TypeError,
        KeyError,
        StopIteration,
        csv.Error,
        RecursionError,
        OverflowError,
    ):
        raise RobustnessInvalid() from None
