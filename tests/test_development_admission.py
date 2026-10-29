"""Fabricated proposals only: metadata never clears a sample or grants execution."""

import json
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from factoryvision.robustness.contracts import (
    Category,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
)
from factoryvision.robustness.contracts import (
    from_json as synthetic_from_json,
)
from factoryvision.robustness.contracts import (
    json_bytes as synthetic_json,
)
from factoryvision.robustness.development_admission import (
    Admission,
    AdmissionInvalid,
    Authorization,
    DevelopmentAdmission,
    Eligibility,
    ExecutionDenied,
    Review,
    from_json,
    json_bytes,
    require_execution,
)


@pytest.mark.parametrize(
    "review,state",
    [
        (Review.UNREVIEWED, Eligibility.NEEDS_PROVENANCE),
        (Review.WITHHELD, Eligibility.NOT_CLEARED),
        (Review.ADVERSE, Eligibility.BLOCKED),
        (Review.DOCUMENTED_CANDIDATE, Eligibility.ELIGIBLE_IN_PRINCIPLE),
    ],
)
def test_eligibility_never_implies_sample_artifact_or_execution_clearance(review, state):
    record = DevelopmentAdmission(review=review)
    assert record.eligibility is state
    assert record.sample_admission is record.artifact_admission is Admission.NOT_VERIFIED
    assert record.execution is Authorization.NOT_AUTHORIZED
    assert record.prior_exposure == "validation_used_for_model_selection"
    assert from_json(json_bytes(record)) == record
    with pytest.raises(ExecutionDenied, match="not authorized"):
        require_execution(from_json(json_bytes(record)))
    with pytest.raises(FrozenInstanceError):
        record.execution = "authorized"


@pytest.mark.parametrize(
    "change",
    [
        {"category": "pcb1"},
        {"category": "pcb2"},
        {"category": "metalparts"},
        {"model_id": "spatial_mean"},
        {"model_id": "student_teacher"},
        {"model_id": "feature_memory"},
        {"model_id": "metalparts_autoencoder"},
        {"official_split": "test"},
        {"partition": "test"},
        {"partition": "train"},
        {"partition": "calibration"},
        {"label": "anomaly"},
        {"dataset": "robustad"},
        {"release": "unknown"},
        {"revision": "0" * 40},
        {"licence_record": "unknown"},
        {"prior_exposure": "unseen"},
        {"purpose": "independent_evaluation"},
        {"purpose": "threshold_tuning"},
        {"purpose": "ood_validation"},
        {"schema_version": True},
        {"protocol_version": 2},
        {"numerical_protocol_version": 1},
        {"numerical_protocol_version": 2.0},
        {"schema_version": float("nan")},
        {"schema_version": float("inf")},
        {"sample_admission": "verified"},
        {"artifact_admission": "verified"},
        {"execution": "authorized"},
        {"review": "documented_candidate_only"},
        {"protocol_id": "factoryvision-synthetic-robustness"},
        {"kind": "historical"},
    ],
)
def test_prohibited_or_incomplete_provenance_is_rejected(change):
    with pytest.raises(AdmissionInvalid):
        replace(DevelopmentAdmission(), **change)


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "url",
        "sample_id",
        "sample_sha256",
        "artifact_sha256",
        "pixels",
        "notes",
        "token",
        "trusted_synthetic",
        "environment",
        "verified",
    ],
)
def test_unknown_private_fields_and_caller_proof_rejected(field):
    data = json.loads(json_bytes(DevelopmentAdmission()))
    data[field] = "fabricated-private-value"
    with pytest.raises(AdmissionInvalid) as caught:
        from_json(json.dumps(data).encode())
    assert str(caught.value) == "Invalid development admission metadata."


def test_all_fields_required_and_redundant_status_verified():
    original = json.loads(json_bytes(DevelopmentAdmission()))
    for field in original:
        data = dict(original)
        del data[field]
        with pytest.raises(AdmissionInvalid):
            from_json(json.dumps(data).encode())
    for field, value in (
        ("eligibility", "eligible_in_principle"),
        ("review", None),
        ("execution", True),
        ("sample_admission", "cleared"),
    ):
        with pytest.raises(AdmissionInvalid):
            from_json(json.dumps({**original, field: value}).encode())


@pytest.mark.parametrize(
    "raw",
    [b"", b"{}", b"[]", b"null", b"\xff", b" " * 16385, b'{"schema_version":1,"schema_version":1}'],
)
def test_untrusted_json_is_bounded_and_sanitized(raw):
    with pytest.raises(AdmissionInvalid):
        from_json(raw)


def test_synthetic_and_historical_schema_isolation():
    plan = Plan(Category.CASHEW, (Parent(ParentID.FIRST, 2, 2),), (Perturbation(Kind.IDENTITY),))
    with pytest.raises(AdmissionInvalid):
        from_json(synthetic_json(plan))
    with pytest.raises(ValueError):
        synthetic_from_json(json_bytes(DevelopmentAdmission()), Plan)
    with pytest.raises(AdmissionInvalid):
        from_json(b'{"schema_version":1,"evidence_id":"historical-cashew"}')
    with pytest.raises(AdmissionInvalid):
        json_bytes(plan)


def test_no_io_and_forged_records_cannot_bypass_gate(monkeypatch):
    import builtins
    from pathlib import Path

    def forbidden(*args, **kwargs):
        pytest.fail("Admission must not access the filesystem")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    record = DevelopmentAdmission(review=Review.DOCUMENTED_CANDIDATE)
    assert from_json(json_bytes(record)) == record
    with pytest.raises(ExecutionDenied):
        require_execution(record)
    object.__setattr__(record, "execution", "authorized")
    for action in (require_execution, json_bytes):
        with pytest.raises(AdmissionInvalid):
            action(record)
    # There is no private membership/hash field to mistake for proof.
    assert not any("sha256" in key or "path" in key for key in asdict(DevelopmentAdmission()))
