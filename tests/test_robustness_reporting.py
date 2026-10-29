"""Scalar-only canonical reports, strict readers and privacy rejection paths."""

import csv
import io
import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest
from test_robustness_runner import plan

from factoryvision.robustness.contracts import Axis, Kind, Perturbation, RobustnessInvalid
from factoryvision.robustness.reporting import (
    COLUMNS,
    RowStatus,
    RunReason,
    _cell,
    csv_bytes,
    from_csv,
    from_json,
    json_bytes,
    report_bound,
)
from factoryvision.robustness.runner import run


def test_json_csv_round_trips_and_scalar_only_allowlist():
    p = plan(
        perturbations=(
            Perturbation(Kind.IDENTITY),
            Perturbation(Kind.BRIGHTNESS, 0.9),
            Perturbation(Kind.TRANSLATION, -0.01, Axis.X),
            Perturbation(Kind.NOISE, 2 / 255, seed=42),
        )
    )
    report = run(p)
    for encode, decode in ((json_bytes, from_json), (csv_bytes, from_csv)):
        raw = encode(report)
        assert decode(raw) == report and encode(decode(raw)) == raw
        assert len(raw) <= report_bound(p) <= 2097152
    document = json.loads(json_bytes(report))
    for row in document["rows"]:
        assert set(row) == set(COLUMNS)
        assert all(value is None or type(value) in (str, int, float) for value in row.values())
    assert document["schema_version"] == 1
    assert all(r["protocol_version"] == 2 for r in document["rows"])
    with pytest.raises(FrozenInstanceError):
        report.rows = ()


@pytest.mark.parametrize("encode", [json_bytes, csv_bytes])
def test_exact_final_utf8_byte_limit_and_one_byte_less(encode):
    report = run(plan())
    cap = len(encode(report))
    for _ in range(4):
        report = replace(
            report,
            plan=replace(report.plan, budgets=replace(report.plan.budgets, max_report_bytes=cap)),
        )
        cap = len(encode(report))
    assert len(encode(report)) == report.plan.budgets.max_report_bytes
    too_small = replace(
        report,
        plan=replace(report.plan, budgets=replace(report.plan.budgets, max_report_bytes=cap - 1)),
    )
    with pytest.raises(RobustnessInvalid):
        encode(too_small)


@pytest.mark.parametrize(
    "field",
    [
        "filename",
        "path",
        "pixels",
        "heatmap",
        "notes",
        "artifact_sha256",
        "credentials",
        "model_id",
    ],
)
def test_private_or_historical_fields_are_rejected_without_disclosure(field):
    doc = json.loads(json_bytes(run(plan())))
    secret = "C:" + "\\".join(("", "Users", "fabricated-private-person", "image"))
    doc["rows"][0][field] = secret
    with pytest.raises(RobustnessInvalid) as caught:
        from_json(json.dumps(doc).encode())
    assert str(caught.value) == str(RobustnessInvalid())


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol_version": 1},
        {"schema_version": True},
        {"provenance": "real_dataset"},
        {"classification": "HISTORICAL"},
        {"generator": "caller_asserted_synthetic"},
        {"adapter": "real_model"},
        {"attempted_units": False},
        {"attempted_units": 0},
        {"failed_rows": 99},
        {"reserved": True},
        {"signed": float("nan")},
        {"coverage": float("inf")},
        {"status": "failed"},
        {"case": 5},
        {"case": True},
        {"reason": "=malicious"},
        {"score_status": "not_evaluated"},
    ],
)
def test_invalid_schema_counts_status_or_nonfinite_numbers(changes):
    doc = json.loads(json_bytes(run(plan())))
    doc["rows"][0].update(changes)
    with pytest.raises(RobustnessInvalid):
        from_json(json.dumps(doc).encode())


def test_duplicate_keys_missing_rows_wrong_order_and_history_mixing():
    report = run(plan())
    raw = json_bytes(report)
    duplicate = raw.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1', 1)
    with pytest.raises(RobustnessInvalid):
        from_json(duplicate)
    for operation in ("remove", "reverse", "extra"):
        doc = json.loads(raw)
        if operation == "remove":
            doc["rows"].pop()
        elif operation == "reverse":
            doc["rows"].reverse()
        else:
            doc["private_metadata"] = {}
        with pytest.raises(RobustnessInvalid):
            from_json(json.dumps(doc).encode())
    with pytest.raises(RobustnessInvalid):
        from_json(b'{"schema_version":1,"evidence_id":"historical"}')


@pytest.mark.parametrize("cell", ["=1+1", "+CMD", "-1+1", "@SUM(A1)", "\t=1", "\uff1d1+1"])
def test_spreadsheet_defense_and_formula_rejection(cell):
    assert _cell(cell).startswith("[text] ")
    rows = list(csv.reader(io.StringIO(csv_bytes(run(plan())).decode())))
    rows[1][COLUMNS.index("reason")] = cell
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    with pytest.raises(RobustnessInvalid):
        from_csv(buffer.getvalue().encode())


def test_negative_numbers_are_protected_but_round_trip_as_numbers():
    report = run(
        plan(perturbations=(Perturbation(Kind.IDENTITY), Perturbation(Kind.BRIGHTNESS, 0.9)))
    )
    raw = csv_bytes(report)
    assert b"[text] -" in raw
    assert from_csv(raw) == report
    assert from_csv(raw).rows[-1].score.signed < 0
    with pytest.raises(RobustnessInvalid):
        from_csv(raw.replace(b"[text] -", b"-"))


@pytest.mark.parametrize(
    "raw",
    [b"", b"\xff", b"[]", b"{}", b"null", b" " * 2097153],
    ids=["empty", "encoding", "array", "object", "null", "oversized"],
)
def test_malformed_input_is_bounded_and_sanitized(raw):
    for decode in (from_json, from_csv):
        with pytest.raises(RobustnessInvalid) as caught:
            decode(raw)
        assert str(caught.value) == str(RobustnessInvalid())


def test_csv_duplicate_columns_unknown_columns_and_bad_cells():
    rows = list(csv.reader(io.StringIO(csv_bytes(run(plan())).decode())))
    for change in ("duplicate", "unknown", "missing", "nonfinite"):
        modified = [list(r) for r in rows]
        if change == "duplicate":
            modified[0][1] = modified[0][0]
        elif change == "unknown":
            modified[0][0] = "private_field"
        elif change == "missing":
            modified[1].pop()
        else:
            modified[1][COLUMNS.index("signed")] = "nan"
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerows(modified)
        with pytest.raises(RobustnessInvalid):
            from_csv(buffer.getvalue().encode())


def test_forged_status_order_and_counters_fail_validation():
    report = run(plan())
    with pytest.raises(RobustnessInvalid):
        replace(report, rows=report.rows[1:])
    with pytest.raises(RobustnessInvalid):
        replace(report.rows[0], attempted=2)
    with pytest.raises(RobustnessInvalid):
        replace(report.rows[0], status=RowStatus.FAILED, reason=RunReason.ADAPTER_ERROR)
    with pytest.raises(RobustnessInvalid):
        json_bytes(object())


def test_execution_cannot_mutate_or_export_as_historical_evidence():
    from factoryvision.inference import model_registry as registry

    before = {name: registry.source_bytes(name) for name in registry.SOURCES}
    evidence = registry.experiment_index()
    report = run(plan())
    assert len(before) == 9 and before == {
        name: registry.source_bytes(name) for name in registry.SOURCES
    }
    assert evidence == registry.experiment_index()
    assert evidence[-2].status == "BLOCKED" and evidence[-1].status == "INCONCLUSIVE"
    with pytest.raises(registry.RegistryInvalid):
        registry.json_bytes(report)
    with pytest.raises(RobustnessInvalid):
        from_json(json.dumps(asdict(evidence[0])).encode())
