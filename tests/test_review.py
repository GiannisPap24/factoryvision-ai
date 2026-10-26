"""Offline human observations: immutable evidence, bounded history and safe exports."""

import csv
import io
import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
from test_reporting import snapshot

from factoryvision.demo import build_demo
from factoryvision.inference import review as rv
from factoryvision.inference.batch import BatchRow
from factoryvision.inference.reporting import create_report

TIME = "2026-01-01T00:00:00.000000Z"


def records(category="cashew", model="autoencoder"):
    return rv.batch_records(create_report(snapshot(category, model)))


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
def test_single_batch_six_model_evidence(category, model):
    source = snapshot(category, model)
    row = source.rows[0]
    single = rv.single_records(
        (
            SimpleNamespace(
                model_id=model,
                score=row.score,
                threshold=row.threshold,
                above_threshold=row.above_threshold,
                artifact_sha256=row.artifact_sha256,
            ),
        ),
        category,
    )
    batch = rv.batch_records(create_report(source))
    assert (
        replace(single[0].evidence, inspection_id=batch[0].evidence.inspection_id)
        == batch[0].evidence
    )
    changed = rv.revise(single[0], rv.Decision.DISAGREES_WITH_RESULT, "Check lighting", TIME)
    assert changed.evidence is single[0].evidence
    assert changed.evidence.score == row.score
    assert changed.evidence.research_flag == row.above_threshold
    assert single[0].history == ()
    if category == "metalparts":
        assert changed.evidence.threshold is changed.evidence.research_flag is None
    with pytest.raises(FrozenInstanceError):
        changed.evidence.score = 0


def test_pending_revision_idempotency_and_reset():
    original = records()[0]
    assert original.decision == rv.Decision.NOT_REVIEWED
    assert original.comment == "" and original.reviewed_at_utc is None
    assert rv.revise(original, rv.Decision.NOT_REVIEWED, "", TIME) is original
    current = rv.revise(original, rv.Decision.AGREES_WITH_RESULT, "Useful overlay", TIME)
    assert rv.revise(current, current.decision, current.comment, rv.utc_now()) is current
    changed = rv.revise(current, rv.Decision.NEEDS_FURTHER_REVIEW, "Check edge", TIME)
    pending = rv.revise(changed, rv.Decision.NOT_REVIEWED, "", TIME)
    assert len(pending.history) == 3 and pending.decision == rv.Decision.NOT_REVIEWED
    assert pending.history[:2] == changed.history
    assert pending.reviewed_at_utc == TIME
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(changed, rv.Decision.AGREES_WITH_RESULT, "", "2025-01-01T00:00:00.000000Z")
    rv.timestamp_valid(rv.utc_now())


@pytest.mark.parametrize("decision", ["AGREES_WITH_RESULT", "PASS", True, None, 1])
def test_decision_type_rejected(decision):
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(records()[0], decision, "", TIME)


@pytest.mark.parametrize(
    "note",
    [
        None,
        1,
        "x" * 501,
        "new\nline",
        "tab\t",
        "\x00",
        "\u202e",
        "\ud800",
        r"C:\private\image.png",
        "C:/private/image.png",
        "/var/factoryvision-fixture/pic",
        "/.private",
        "/ path with spaces",
        r"\\host\private",
        "file:private",
        "person@example.com",
        "ghp_example",
        "sk-example",
        "github_pat_example",
        "AKIA" + "1" * 16,  # Fabricated secret-shape input, never an actual credential.
        "token=example",
        "api key: example",
        "Bearer example",
        "PRIVATE KEY",
        "Ｃ：／private",
    ],
)
def test_note_privacy_validation_sanitized(note):
    with pytest.raises(rv.ReviewInvalid) as error:
        rv.revise(records()[0], rv.Decision.NEEDS_FURTHER_REVIEW, note, TIME)
    assert str(error.value) == str(rv.ReviewInvalid())


@pytest.mark.parametrize("note", ["", "x" * 500, "Έλεγχος ακμής", 'Edge, "check"', "=SUM(1,2)"])
def test_valid_notes(note):
    assert rv.revise(records()[0], rv.Decision.AGREES_WITH_RESULT, note, TIME).comment == note


@pytest.mark.parametrize(
    "stamp",
    [
        None,
        "yesterday",
        "2026-02-30T00:00:00.000000Z",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00.000000+02:00",
    ],
)
def test_timestamp_rejected(stamp):
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(records()[0], rv.Decision.AGREES_WITH_RESULT, "", stamp)


def test_limits_and_invalid_records():
    current = records()[0]
    for n in range(rv.MAX_REVISIONS):
        current = rv.revise(current, rv.Decision.AGREES_WITH_RESULT, str(n), TIME)
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(current, rv.Decision.AGREES_WITH_RESULT, "one too many", TIME)
    for changes in (
        {"schema_version": True},
        {"schema_version": 2},
        {"history": []},
        {"history": (None,)},
        {"reviewer_provenance": "verified"},
        {"evidence": None},
    ):
        with pytest.raises(rv.ReviewInvalid):
            replace(current, **changes)
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(None, rv.Decision.NOT_REVIEWED, "", TIME)
    with pytest.raises(rv.ReviewInvalid):
        rv.revise(current, rv.Decision.NOT_REVIEWED, "must be empty", TIME)
    for invalid in (None, (), [], [None], tuple([current] * 21), (None,), (current, current)):
        with pytest.raises(rv.ReviewInvalid):
            rv.document(invalid)
    with pytest.raises(rv.ReviewInvalid):
        rv.document(records() + records())
    with pytest.raises(rv.ReviewInvalid):
        rv.single_records((), "cashew")
    with pytest.raises(rv.ReviewInvalid):
        rv.batch_records(None)
    with pytest.raises(rv.ReviewInvalid):
        rv.demo_records(None)


@pytest.mark.parametrize(
    "changes",
    [
        {"inspection_id": None},
        {"inspection_id": "../secret"},
        {"image_id": "secret.png"},
        {"image_id": None},
        {"image_id": "Image 021"},
        {"category": "pcb1"},
        {"model_id": "unknown"},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": -1},
        {"score": True},
        {"score": 10**400},
        {"threshold": 0},
        {"research_flag": None},
        {"research_flag": False},
        {"research_flag": 1},
        {"artifact_sha256": "a" * 64},
        {"origin": "ground_truth"},
    ],
)
def test_evidence_tampering_rejected(changes):
    with pytest.raises(rv.ReviewInvalid):
        replace(records()[0].evidence, **changes)


def test_batch_isolation_rejections_and_no_source_mutation():
    source = snapshot()
    source = replace(
        source,
        elapsed_ms=10.0,
        rows=(
            source.rows[0],
            replace(source.rows[0], image_id="Image 002"),
            BatchRow("Image 003", "c" * 64, "rejected", "private filename"),
        ),
    )
    before = create_report(source)
    pending = rv.batch_records(before)
    changed = (rv.revise(pending[0], rv.Decision.DISAGREES_WITH_RESULT, "Check", TIME), pending[1])
    assert len(changed) == 2 and changed[1].history == ()
    assert changed[0].evidence.inspection_id == changed[1].evidence.inspection_id
    assert create_report(source) == before
    invalid = replace(before)
    object.__setattr__(invalid, "schema_version", 2)
    with pytest.raises(rv.ReviewInvalid):
        rv.batch_records(invalid)


def test_exports_determinism_history_privacy_and_csv_formulas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pending = records()[0]
    reviewed = rv.revise(pending, rv.Decision.AGREES_WITH_RESULT, "=SUM(1,2)", TIME)
    reviewed = rv.revise(reviewed, rv.Decision.NEEDS_FURTHER_REVIEW, "More light", TIME)
    group = (reviewed, replace(pending, evidence=replace(pending.evidence, image_id="Image 002")))
    first = rv.json_bytes(group)
    assert first == rv.json_bytes(group)
    assert rv.csv_bytes(group) == rv.csv_bytes(group)
    data = json.loads(first)
    assert data["schema"] == rv.SCHEMA
    assert data["records"][0]["history"][0]["comment"] == "=SUM(1,2)"
    assert data["records"][1]["reviewed_at_utc"] is None
    rows = list(csv.DictReader(io.StringIO(rv.csv_bytes(group).decode())))
    assert rows[0]["comment"].startswith("[text] ") and rows[0]["revision"] == "1"
    assert rows[1]["comment"] == "More light" and rows[1]["revision"] == "2"
    assert rows[2]["revision"] == "0" and rows[2]["reviewed_at_utc"] == ""
    for forbidden in (b"private", b"filename", b"EXIF", b"a" * 64, b"b" * 64, b"base64"):
        assert forbidden not in first and forbidden not in rv.csv_bytes(group)
    assert list(tmp_path.iterdir()) == []


def test_demo_separation_and_null_predictions():
    demo = rv.demo_records(build_demo().report)
    assert len(demo) == 2
    for record in demo:
        assert record.evidence.origin == "synthetic_demo"
        assert (
            record.evidence.score is record.evidence.model_id is record.evidence.threshold is None
        )
        assert record.evidence.research_flag is record.evidence.artifact_sha256 is None
    assert "synthetic_demo" in rv.json_bytes(demo).decode()
    for changes in ({"category": "cashew"}, {"score": 1}):
        with pytest.raises(rv.ReviewInvalid):
            replace(demo[0].evidence, **changes)
    real = records()[0]
    aligned_demo = replace(
        demo[0], evidence=replace(demo[0].evidence, inspection_id=real.evidence.inspection_id)
    )
    with pytest.raises(rv.ReviewInvalid):
        rv.document((real, aligned_demo))
    metal = records("metalparts", "metalparts_autoencoder")[0]
    metal = replace(
        metal, evidence=replace(metal.evidence, inspection_id=real.evidence.inspection_id)
    )
    with pytest.raises(rv.ReviewInvalid):
        rv.document((real, metal))
    with pytest.raises(rv.ReviewInvalid):
        replace(metal.evidence, research_flag=True)
