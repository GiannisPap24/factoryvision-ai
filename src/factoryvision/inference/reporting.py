"""Versioned scalar-only reports. No inference, Streamlit, output files or network."""

import csv
import hashlib
import io
import json
import math
import re
import statistics
import unicodedata
from dataclasses import asdict, dataclass
from importlib.resources import files
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factoryvision.inference.batch import BatchReport

SCHEMA_VERSION = 1
LIMITATIONS = (
    "Research prototype; not certified industrial acceptance/rejection or PASS/FAIL.",
    "Anomaly scores and heatmap intensities are not defect probabilities.",
    "Low scores do not establish defect absence; different models use different scales.",
    "Cashew: historical VisA evaluation; the official set is now observed. Feature-memory "
    "and student-teacher independent defect accuracy is not established.",
    "MetalParts: exploratory source-domain diagnostics, not independent test performance; "
    "no validated binary threshold. RobustAD target-domain evaluation: NOT CLEARED.",
    "PCB1: BLOCKED. PCB2: INCONCLUSIVE. No new official evaluation is represented here.",
)
TIMING_SCOPE = (
    "Batch duration includes verification, decoding, inference and display preparation; "
    "excludes upload, initial preview, reporting/export generation and browser rendering."
)
ERRORS = {
    None: None,
    "input_rejected": "Input rejected before inference.",
    "inference_failed": "Image inference failed; no score available.",
    "execution_unavailable": "Not inspected because model artifacts or CPU execution failed.",
}
STATUSES = ("inspected", "rejected", "failed", "skipped")


class ReportInvalid(ValueError):
    """Fixed public message; never echo rejected input or internal exceptions."""

    def __init__(self):
        super().__init__("Report validation failed. Clear the batch and run it again.")


def require(condition):
    if not condition:
        raise ReportInvalid()


def number(value):
    try:
        require(type(value) in (int, float) and math.isfinite(value) and value >= 0)
    except OverflowError as exc:
        raise ReportInvalid() from exc


def digest(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    category: str
    name: str
    score_label: str
    registered_artifact_sha256: str
    catalog_sha256: str
    frozen_threshold: float | None
    scientific_status: str


def model_info(category: str, model_id: str) -> ModelInfo:
    """Read installed approved metadata only, never model files or dataset payloads."""
    try:
        require(type(category) is str and type(model_id) is str)
        if category == "cashew":
            raw = files("factoryvision.inference").joinpath("cashew_artifacts.json").read_bytes()
            spec = next(m for m in json.loads(raw)["models"] if m["id"] == model_id)
            name, label = spec["name"], spec["score_label"]
            sha, threshold = spec["assets"]["weights"]["sha256"], spec["threshold"]
            status = "Historical Cashew research; frozen normal-calibration threshold only."
        else:
            require(category == "metalparts")
            require(model_id in ("metalparts_autoencoder", "metalparts_feature_memory"))
            feature = model_id == "metalparts_feature_memory"
            filename = (
                "metalparts_feature_artifacts.json" if feature else "metalparts_artifacts.json"
            )
            raw = files("factoryvision.inference").joinpath(filename).read_bytes()
            sha = json.loads(raw)["assets"]["memory" if feature else "checkpoint"]["sha256"]
            threshold = None
            name = "Experimental MetalParts " + ("feature memory" if feature else "autoencoder")
            label = (
                "Maximum descriptor RMS nearest-neighbor distance (not a probability)"
                if feature
                else "Native RGB reconstruction MSE (continuous; not a probability)"
            )
            status = "Exploratory source-domain research; independent evaluation NOT CLEARED."
        digest(sha)
        if threshold is not None:
            number(threshold)
        require(all(type(s) is str and 0 < len(s) < 160 for s in (name, label)))
        return ModelInfo(
            model_id, category, name, label, sha, hashlib.sha256(raw).hexdigest(), threshold, status
        )
    except (OSError, KeyError, StopIteration, TypeError, ValueError) as exc:
        raise ReportInvalid() from exc


@dataclass(frozen=True)
class ReportRow:
    image_id: str
    status: str
    score: float | None
    threshold: float | None
    research_anomaly_flag: bool | None
    inference_ms: float | None
    adapter_total_ms: float | None
    artifact_sha256: str | None
    error_code: str | None

    @property
    def interpretation(self):
        if self.status != "inspected":
            return "Not inspected; no anomaly decision."
        if self.threshold is None:
            return "Continuous score; no validated binary threshold."
        if self.research_anomaly_flag:
            return "Above frozen research threshold; review anomaly map."
        return "At/below frozen research threshold; does not establish defect absence."


@dataclass(frozen=True)
class Counts:
    selected: int
    accepted: int
    rejected: int
    inspected: int
    failed: int
    skipped: int


def counts(rows: tuple[ReportRow, ...]) -> Counts:
    totals = {s: sum(r.status == s for r in rows) for s in STATUSES}
    return Counts(len(rows), len(rows) - totals["rejected"], **totals)


@dataclass(frozen=True)
class ResearchReport:
    schema_version: int
    batch_id: str
    model: ModelInfo
    execution_status: str
    summary: Counts
    batch_duration_ms: float
    rows: tuple[ReportRow, ...]

    def __post_init__(self):
        self.validate()

    def validate(self):
        require(type(self.schema_version) is int and self.schema_version == SCHEMA_VERSION)
        digest(self.batch_id)
        require(type(self.model) is ModelInfo)
        require(self.model == model_info(self.model.category, self.model.model_id))
        require(type(self.rows) is tuple and 1 <= len(self.rows) <= 20)
        require(all(type(r) is ReportRow for r in self.rows))
        require(type(self.summary) is Counts)
        require(all(type(v) is int and v >= 0 for v in asdict(self.summary).values()))
        require(self.summary == counts(self.rows))
        number(self.batch_duration_ms)
        require(self.execution_status in ("completed", "completed_with_issues", "stopped"))
        expected = (
            "completed_with_issues"
            if any(r.status != "inspected" for r in self.rows)
            else "completed"
        )
        if self.execution_status != "stopped":
            require(self.execution_status == expected and not self.summary.skipped)
        else:
            require(self.summary.failed + self.summary.skipped > 0)
        skipped = False
        for index, row in enumerate(self.rows, 1):
            require(row.image_id == f"Image {index:03d}" and row.status in STATUSES)
            if row.status == "inspected":
                require(not skipped and row.error_code is None)
                number(row.score)
                number(row.inference_ms)
                number(row.adapter_total_ms)
                require(row.adapter_total_ms >= row.inference_ms)
                require(row.artifact_sha256 == self.model.registered_artifact_sha256)
                require(row.threshold == self.model.frozen_threshold)
                if row.threshold is None:
                    require(row.research_anomaly_flag is None)
                else:
                    number(row.threshold)
                    require(type(row.research_anomaly_flag) is bool)
                    require(row.research_anomaly_flag == (row.score > row.threshold))
            else:
                require(
                    all(
                        getattr(row, k) is None
                        for k in (
                            "score",
                            "threshold",
                            "research_anomaly_flag",
                            "inference_ms",
                            "adapter_total_ms",
                            "artifact_sha256",
                        )
                    )
                )
                require(
                    row.error_code
                    == {
                        "rejected": "input_rejected",
                        "failed": "inference_failed",
                        "skipped": "execution_unavailable",
                    }[row.status]
                )
                skipped = skipped or row.status == "skipped"
        require(sum(r.adapter_total_ms or 0 for r in self.rows) <= self.batch_duration_ms + 1e-6)

    @property
    def report_id(self):
        return "fvr1-" + hashlib.sha256(_json(self._document())).hexdigest()

    def _document(self):
        self.validate()
        values = [r.inference_ms for r in self.rows if r.status == "inspected"]
        return {
            "schema_version": self.schema_version,
            "producer": "FactoryVision AI",
            "batch_id": self.batch_id,
            "category": self.model.category,
            "model": {
                **asdict(self.model),
                "artifact_version": "sha256:" + self.model.registered_artifact_sha256,
                "observed_artifact_sha256": self.model.registered_artifact_sha256
                if values
                else None,
            },
            "execution_status": self.execution_status,
            "summary": asdict(self.summary),
            "timing": {
                "batch_duration_ms": self.batch_duration_ms,
                "inference_sum_ms": sum(values) if values else None,
                "inference_median_ms": statistics.median(values) if values else None,
                "inference_min_ms": min(values) if values else None,
                "inference_max_ms": max(values) if values else None,
                "scope": TIMING_SCOPE,
            },
            "rows": [
                {
                    **asdict(r),
                    "interpretation": r.interpretation,
                    "error_message": ERRORS[r.error_code],
                }
                for r in self.rows
            ],
            "limitations": list(LIMITATIONS),
        }


def create_report(batch: "BatchReport") -> ResearchReport:
    """Project the completed snapshot. Ignore free-form names/errors and all image data."""
    try:
        require(type(batch.schema_version) is int and batch.schema_version == 1)
        require(type(batch.rows) is tuple and type(batch.fatal_error) is str)
        model = model_info(batch.category, batch.model_id)
        rows = []
        for row in batch.rows:
            require(row.status in STATUSES and type(row.error) is str)
            digest(row.input_sha256)
            inspected = row.status == "inspected"
            require((not row.error) if inspected else bool(row.error))
            if not inspected:
                require(row.detail is None and row.artifact_sha256 == "")
            rows.append(
                ReportRow(
                    row.image_id,
                    row.status,
                    row.score,
                    row.threshold,
                    row.above_threshold,
                    row.inference_ms,
                    row.total_ms,
                    row.artifact_sha256 if inspected else None,
                    {
                        "inspected": None,
                        "rejected": "input_rejected",
                        "failed": "inference_failed",
                        "skipped": "execution_unavailable",
                    }[row.status],
                )
            )
        rows = tuple(rows)
        status = (
            "stopped"
            if batch.fatal_error
            else (
                "completed_with_issues"
                if any(r.status != "inspected" for r in rows)
                else "completed"
            )
        )
        return ResearchReport(
            1, batch.batch_id, model, status, counts(rows), batch.elapsed_ms, rows
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ReportInvalid() from exc


def _json(document):
    return (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode(
        "utf-8"
    )


def json_bytes(report: ResearchReport) -> bytes:
    document = report._document()
    return _json({"report_id": report.report_id, **document})


CSV_COLUMNS = (
    "schema_version",
    "report_id",
    "batch_id",
    "category",
    "model_id",
    "model_name",
    "artifact_version",
    "registered_artifact_sha256",
    "catalog_sha256",
    "scientific_status",
    "execution_status",
    "selected",
    "accepted",
    "rejected",
    "inspected",
    "failed",
    "skipped",
    "batch_duration_ms",
    "image_id",
    "status",
    "score_label",
    "score",
    "threshold",
    "research_anomaly_flag",
    "interpretation",
    "inference_ms",
    "adapter_total_ms",
    "observed_artifact_sha256",
    "error_code",
    "error_message",
    "limitations",
    "timing_scope",
)


def spreadsheet_cell(value: str) -> str:
    """Defense in depth: mark risky text literally; do not rely on CSV quoting."""
    require(type(value) is str)
    normalized = unicodedata.normalize("NFKC", value)
    risky = normalized.lstrip().startswith(("=", "+", "-", "@"))
    if risky or any(unicodedata.category(c).startswith("C") for c in value):
        return "[text] " + value
    return value


def csv_bytes(report: ResearchReport) -> bytes:
    document = report._document()
    report_id = report.report_id
    model, summary = document["model"], document["summary"]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for row in document["rows"]:
        record = {
            **model,
            **summary,
            **row,
            "schema_version": report.schema_version,
            "report_id": report_id,
            "batch_id": report.batch_id,
            "model_name": model["name"],
            "execution_status": report.execution_status,
            "batch_duration_ms": report.batch_duration_ms,
            "observed_artifact_sha256": row["artifact_sha256"],
            "limitations": " | ".join(LIMITATIONS),
            "timing_scope": TIMING_SCOPE,
        }
        values = []
        for key in CSV_COLUMNS:
            value = record[key]
            if value is None:
                value = "null"
            elif type(value) is bool:
                value = "true" if value else "false"
            elif type(value) in (float, int):
                value = repr(value)
            values.append(spreadsheet_cell(value))
        writer.writerow(values)
    return buffer.getvalue().encode("utf-8")
