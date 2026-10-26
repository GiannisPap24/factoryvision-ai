"""Scalar-only synthetic report contracts. No trained artifacts or datasets needed."""

import csv
import io
import json
from dataclasses import replace

import pytest

from factoryvision.inference import reporting as rp
from factoryvision.inference.batch import BatchReport, BatchRow


def snapshot(category="cashew", model_id="autoencoder"):
    info = rp.model_info(category, model_id)
    score = 0.12345678901234567
    row = BatchRow(
        "Image 001",
        "a" * 64,
        "inspected",
        model_name="private ignored name",
        score_label="private ignored label",
        score=score,
        threshold=info.frozen_threshold,
        above_threshold=None if info.frozen_threshold is None else score > info.frozen_threshold,
        inference_ms=2.125,
        total_ms=3.5,
        artifact_sha256=info.registered_artifact_sha256,
    )
    return BatchReport("b" * 64, category, model_id, (row,), 4.5)


@pytest.fixture
def source():
    return snapshot()


@pytest.mark.parametrize(
    "category,model",
    [
        ("cashew", "autoencoder"),
        ("cashew", "spatial_mean"),
        ("cashew", "feature_memory"),
        ("cashew", "student_teacher"),
        ("metalparts", "metalparts_autoencoder"),
        ("metalparts", "metalparts_feature_memory"),
    ],
)
def test_six_models_exact_contract(category, model):
    source = snapshot(category, model)
    report = rp.create_report(source)
    assert report.rows[0].score == source.rows[0].score
    assert report.rows[0].threshold == source.rows[0].threshold
    assert report.rows[0].artifact_sha256 == source.rows[0].artifact_sha256
    document = json.loads(rp.json_bytes(report))
    assert document["schema_version"] == 1 and document["report_id"].startswith("fvr1-")
    assert document["model"]["artifact_version"].startswith("sha256:")
    assert document["summary"] == dict(
        selected=1, accepted=1, inspected=1, rejected=0, failed=0, skipped=0
    )
    assert document["timing"]["inference_median_ms"] == 2.125
    assert "not a probability" in report.model.score_label or category == "cashew"
    if category == "metalparts":
        assert document["rows"][0]["threshold"] is None
        assert document["rows"][0]["research_anomaly_flag"] is None
        assert "no validated" in report.rows[0].interpretation


def test_partial_and_stopped_failures_are_sanitized(source):
    rows = source.rows + (
        BatchRow("Image 002", "c" * 64, "rejected", "=SECRET C:\\private\\image.png\ntraceback"),
        BatchRow("Image 003", "d" * 64, "failed", "@secret-token"),
        BatchRow("Image 004", "e" * 64, "skipped", "private checkpoint path"),
    )
    report = rp.create_report(replace(source, rows=rows, fatal_error="private CPU traceback"))
    assert report.execution_status == "stopped"
    assert report.summary == rp.Counts(4, 3, 1, 1, 1, 1)
    text = rp.json_bytes(report).decode()
    assert "SECRET" not in text and "private" not in text and "traceback" not in text
    assert [r.status for r in report.rows] == ["inspected", "rejected", "failed", "skipped"]
    for row in report.rows[1:]:
        assert row.score is None and row.inference_ms is None and row.threshold is None
    partial = rp.create_report(replace(source, rows=rows[:3]))
    assert partial.execution_status == "completed_with_issues"
    failed = rp.create_report(replace(source, rows=(replace(rows[2], image_id="Image 001"),)))
    assert json.loads(rp.json_bytes(failed))["model"]["observed_artifact_sha256"] is None
    assert json.loads(rp.json_bytes(failed))["timing"]["inference_sum_ms"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "1", 10**400, None])
def test_invalid_numeric_values(source, value):
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, elapsed_ms=value))
    for field in ("score", "inference_ms", "total_ms"):
        with pytest.raises(rp.ReportInvalid):
            rp.create_report(replace(source, rows=(replace(source.rows[0], **{field: value}),)))


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"batch_id": "../secret"},
        {"rows": ()},
        {"category": "pcb1"},
        {"model_id": "metalparts_autoencoder"},
        {"fatal_error": False},
        {"elapsed_ms": 1},
        {"rows": []},
    ],
)
def test_invalid_batch_metadata(source, changes):
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, **changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"image_id": "=SUM(1,2)"},
        {"status": "passed"},
        {"input_sha256": "private"},
        {"error": "unexpected success error"},
        {"threshold": 0.5},
        {"above_threshold": None},
        {"above_threshold": False},
        {"above_threshold": 1},
        {"artifact_sha256": "f" * 64},
        {"total_ms": 1},
    ],
)
def test_invalid_success_row(source, changes):
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, rows=(replace(source.rows[0], **changes),)))


def test_missing_scores_thresholds_and_corrupted_reports(source, monkeypatch):
    report = rp.create_report(source)
    with pytest.raises(rp.ReportInvalid):
        replace(report, summary=replace(report.summary, accepted=0))
    with pytest.raises(rp.ReportInvalid):
        replace(report, summary=replace(report.summary, selected=True))
    with pytest.raises(rp.ReportInvalid):
        replace(report, model=replace(report.model, name="../private"))
    with pytest.raises(rp.ReportInvalid):
        replace(report, execution_status="stopped")
    with pytest.raises(rp.ReportInvalid):
        replace(report, rows=list(report.rows))
    object.__setattr__(report, "batch_duration_ms", float("nan"))
    for export in (rp.csv_bytes, rp.json_bytes):
        with pytest.raises(rp.ReportInvalid):
            export(report)
    metal = snapshot("metalparts", "metalparts_autoencoder")
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(metal, rows=(replace(metal.rows[0], threshold=0.5),)))
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, rows=(replace(source.rows[0], threshold=None),)))
    monkeypatch.setattr(rp, "files", lambda *a: None)
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(source)


def test_rejected_invalid_states_and_order(source):
    row = BatchRow("Image 001", "a" * 64, "rejected", "invalid image")
    for changes in (
        {"score": 0.1},
        {"error": ""},
        {"detail": b"pixels"},
        {"artifact_sha256": "a" * 64},
    ):
        with pytest.raises(rp.ReportInvalid):
            rp.create_report(replace(source, rows=(replace(row, **changes),)))
    skipped = replace(row, status="skipped")
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, rows=(skipped,)))
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(
            replace(
                source,
                rows=(skipped, replace(source.rows[0], image_id="Image 002")),
                fatal_error="stop",
            )
        )
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(replace(source, rows=source.rows * 2))
    with pytest.raises(rp.ReportInvalid):
        rp.create_report(None)


def test_csv_json_roundtrip_stability_and_no_persistence(source, tmp_path):
    report = rp.create_report(source)
    before = list(tmp_path.rglob("*"))
    csv_blob = rp.csv_bytes(report)
    json_blob = rp.json_bytes(report)
    assert csv_blob == rp.csv_bytes(report) == rp.csv_bytes(rp.create_report(source))
    assert json_blob == rp.json_bytes(report) == rp.json_bytes(rp.create_report(source))
    assert before == list(tmp_path.rglob("*"))
    reader = csv.DictReader(io.StringIO(csv_blob.decode("utf8"), newline=""))
    rows = list(reader)
    assert tuple(reader.fieldnames) == rp.CSV_COLUMNS
    assert float(rows[0]["score"]) == source.rows[0].score
    assert rows[0]["error_code"] == "null" and rows[0]["research_anomaly_flag"] == "true"
    assert json.loads(json_blob)["rows"][0]["error_code"] is None
    other = rp.create_report(replace(source, elapsed_ms=5.5))
    assert other.report_id != report.report_id
    assert not any(
        word in json_blob + csv_blob
        for word in (b"input_sha256", b"payload", b'"preview"', b"PNG", b"EXIF", b"private ignored")
    )
    low = rp.create_report(
        replace(source, rows=(replace(source.rows[0], score=0.0, above_threshold=False),))
    )
    assert "does not establish" in low.rows[0].interpretation


@pytest.mark.parametrize(
    "value",
    [
        "=1+1",
        "+cmd",
        "-1+2",
        "@SUM(A1)",
        "\t=1",
        "\r=1",
        "\n=1",
        "  =1",
        "\x00x",
        "\ufeff=1",
        "＝1",
        "＋1",
        "－1",
        "＠x",
    ],
)
def test_spreadsheet_formula_controls_and_unicode(value):
    cell = rp.spreadsheet_cell(value)
    assert cell.startswith("[text] ")
    buffer = io.StringIO(newline="")
    csv.writer(buffer, quoting=csv.QUOTE_ALL).writerow([cell])
    assert next(csv.reader(io.StringIO(buffer.getvalue(), newline=""))) == [cell]


def test_quotes_unicode_newlines_and_delimiters():
    value = 'Ελληνικά, "quoted"; 日本語\nnext line'
    cell = rp.spreadsheet_cell(value)
    buffer = io.StringIO(newline="")
    csv.writer(buffer, quoting=csv.QUOTE_ALL).writerow([cell, "safe"])
    assert next(csv.reader(io.StringIO(buffer.getvalue(), newline=""))) == [
        "[text] " + value,
        "safe",
    ]
    assert rp.spreadsheet_cell('Ελληνικά, "quoted"') == 'Ελληνικά, "quoted"'
