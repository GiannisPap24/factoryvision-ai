"""Synthetic adversarial metadata: sanitized errors and isolation from evidence."""

import builtins
import json
import subprocess
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

import pytest

from factoryvision.robustness.contracts import (
    Applicability,
    Category,
    Diagnostic,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
    from_json,
    json_bytes,
)


def metadata(record_type=Plan):
    identity = Perturbation(Kind.IDENTITY)
    record = (
        Plan(Category.CASHEW, (Parent(ParentID.FIRST, 256, 256),), (identity,))
        if record_type is Plan
        else Diagnostic(ParentID.FIRST, identity, Applicability.CONDITIONAL)
    )
    return json.loads(json_bytes(record))


@pytest.mark.parametrize("record_type", [Plan, Diagnostic])
@pytest.mark.parametrize(
    "field,value",
    [
        ("protocol_id", "https://example.invalid/private"),
        ("provenance", "official_test"),
        ("provenance", "human_review"),
        ("classification", "validated_evidence"),
        ("schema_version", True),
        ("protocol_version", 1.0),
        ("source", "../private.json"),
        ("metrics", {"auroc": 1.0}),
        ("score", 0.5),
        ("ood_confidence", 0.99),
        ("artifact_version", "sha256:" + "0" * 64),
    ],
)
def test_metadata_cannot_be_promoted_to_historical_or_real_data_evidence(record_type, field, value):
    data = metadata(record_type)
    data[field] = value
    with pytest.raises(RobustnessInvalid):
        from_json(json.dumps(data).encode(), record_type)


@pytest.mark.parametrize("record_type", [Plan, Diagnostic])
def test_unsafe_identifiers_are_rejected_without_disclosure(record_type):
    fake_path = "C:" + "\\".join(("", "Users", "fabricated-person", "private"))
    for identifier in (
        fake_path,
        "../private.png",
        "https://example.invalid/input.png",
        "file:///private",
        "case-001\nprivate",
        "someone" + "@example.invalid",
        "case-005",
    ):
        data = metadata(record_type)
        target = data["parents"][0] if record_type is Plan else data
        target["parent_id"] = identifier
        with pytest.raises(RobustnessInvalid) as caught:
            from_json(json.dumps(data).encode(), record_type)
        assert str(caught.value) == str(RobustnessInvalid())
        rendered = "".join(traceback.format_exception(caught.value))
        assert identifier not in rendered
        assert "fabricated-person" not in rendered


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b"\xff",
        b"[]",
        b"null",
        b"true",
        b"1",
        b"{}",
        b"",
        b" " * 65537,
        b"[" * 2000 + b"]" * 2000,
        "not bytes",
        None,
        b'{"private-property": malformed}',
    ],
    ids=[
        "syntax",
        "encoding",
        "array",
        "null",
        "bool",
        "integer",
        "empty-object",
        "empty",
        "oversized",
        "too-deep",
        "string",
        "none",
        "private-key",
    ],
)
def test_malformed_bounded_input_has_fixed_public_error(raw):
    with pytest.raises(RobustnessInvalid) as caught:
        from_json(raw, Plan)
    assert str(caught.value) == str(RobustnessInvalid())
    assert "private-property" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "0.5"])
def test_untrusted_json_numbers_fail_closed(value):
    data = metadata()
    data["perturbations"].append(
        {"kind": "gaussian_blur", "severity": value, "axis": None, "seed": None}
    )
    with pytest.raises(RobustnessInvalid):
        from_json(json.dumps(data).encode(), Plan)


def test_unknown_record_types_and_report_caps_fail_closed():
    raw = json.dumps(metadata()).encode()
    for record_type in (dict, object, None):
        with pytest.raises(RobustnessInvalid):
            from_json(raw, record_type)
    for value in (None, {}, raw):
        with pytest.raises(RobustnessInvalid):
            json_bytes(value)
    for cap in (0, True, 2097153):
        with pytest.raises(RobustnessInvalid):
            json_bytes(from_json(raw, Plan), max_bytes=cap)


def test_import_does_not_load_ml_image_or_application_modules():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import factoryvision.robustness.contracts; "
            "assert not any(m.split('.')[0] in ('torch', 'numpy', 'PIL', 'streamlit') "
            "or m.startswith('factoryvision.inference') for m in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == result.stderr == ""


def test_round_trip_has_no_filesystem_io(monkeypatch):
    raw = json.dumps(metadata()).encode()

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata contracts must not access files")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "rglob", forbidden)
        assert from_json(json_bytes(from_json(raw, Plan)), Plan).planned_inference_calls == 3


def test_historical_evidence_and_all_nine_raw_resources_remain_unchanged():
    from factoryvision.inference import model_registry as registry

    before = {name: registry.source_bytes(name) for name in registry.SOURCES}
    evidence = registry.experiment_index()
    assert len(before) == 9
    for record_type in (Plan, Diagnostic):
        raw = json.dumps(metadata(record_type)).encode()
        assert from_json(json_bytes(from_json(raw, record_type)), record_type)
        # The diagnostic parser cannot ingest historical evidence, either.
        with pytest.raises(RobustnessInvalid):
            from_json(json.dumps(asdict(evidence[0])).encode(), record_type)
    assert before == {name: registry.source_bytes(name) for name in registry.SOURCES}
    assert evidence == registry.experiment_index()
    assert evidence[-2].status == "BLOCKED"
    assert evidence[-1].status == "INCONCLUSIVE"
    assert evidence[4].status == evidence[5].status == "EXPLORATORY"
    assert all("NOT CLEARED" in row.evaluation_protocol for row in evidence[4:6])
