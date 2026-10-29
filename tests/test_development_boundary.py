"""Analytical InspectionResult doubles only; no neural forward or real assets."""

import builtins
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from factoryvision.inference import service, uploads
from factoryvision.inference.reporting import ModelInfo
from factoryvision.inference.service import InspectionResult
from factoryvision.robustness import development_boundary as boundary
from factoryvision.robustness.assessment import ThresholdRule
from factoryvision.robustness.development_admission import AdmissionInvalid
from factoryvision.robustness.development_boundary import (
    CompatibilityObservation,
    InterfaceSpec,
    MapSpace,
    ProbeReason,
    ProbeStatus,
    SyntheticInspectionDouble,
    exercise_synthetic_interface,
    validate_inspection,
)
from factoryvision.robustness.reporting import json_bytes as synthetic_report_json


def spec():
    # Deliberately fabricated artifact/catalog hashes and threshold; never authenticates files.
    return InterfaceSpec(
        ModelInfo(
            "autoencoder",
            "cashew",
            "Convolutional autoencoder",
            "Mean RGB squared reconstruction error",
            "a" * 64,
            "b" * 64,
            0.5,
            "Historical Cashew research; frozen normal-calibration threshold only.",
        )
    )


def fixture_result():
    pixels = np.full((10, 10, 3), 128, dtype=np.uint8)
    return pixels, SyntheticInspectionDouble(spec()).evaluate(pixels)


@pytest.fixture(autouse=True)
def no_models_or_decoders(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No model, checkpoint, image decoder or service execution is permitted")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", forbidden)
    monkeypatch.setattr(service.InspectionService, "__init__", forbidden)
    monkeypatch.setattr(uploads, "decode_upload", forbidden)
    monkeypatch.setattr(service, "decode_upload", forbidden)
    monkeypatch.setattr(service, "preprocess", forbidden)


def test_internal_fixture_ordering_and_frozen_threshold_crossing(monkeypatch):
    original = SyntheticInspectionDouble.evaluate
    seen = []

    def evaluate(self, pixels):
        seen.append(pixels.copy())
        assert not pixels.flags.writeable
        return original(self, pixels)

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert len(rows) == sum(r.attempted for r in rows) == 5
    assert [r.ordinal for r in rows] == [1, 2, 3, 4, 5]
    for i in (1, 2):
        np.testing.assert_array_equal(seen[0], seen[i])
        assert rows[i].score.absolute == rows[i].spatial.mae == 0
    np.testing.assert_array_equal(seen[3][0, 0], [58, 115, 173])
    np.testing.assert_array_equal(seen[4][0, 0], [70, 141, 211])
    assert rows[3].score.crossing.value == "down"
    assert all(
        r.kind == "synthetic_service_interface_probe"
        and r.model_execution == "not_run"
        and r.map_space is MapSpace.PREPROCESSED_128
        for r in rows
    )
    assert rows == exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert len(seen) == 10


def test_model_score_is_preserved_not_recomputed_from_spatial_or_display(monkeypatch):
    original = SyntheticInspectionDouble.evaluate
    scores = iter([0.5, 0.5, 0.5, 0.4, 0.7])

    def evaluate(self, pixels):
        result = original(self, pixels)
        score = next(scores)
        return replace(
            result,
            score=score,
            above_threshold=score > 0.5,
            spatial=np.full((128, 128), 0.2, dtype=np.float32),
        )

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert rows[-1].score.signed == pytest.approx(0.2)
    assert rows[-1].spatial.mae == 0.0
    assert rows[0].score.crossing.value == "none"  # Strict greater-than at equality.
    assert rows[-1].score.crossing.value == "up"


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "spatial_mean"},
        {"model_name": "private"},
        {"score_label": "display"},
        {"artifact_sha256": "c" * 64},
        {"threshold": 0.6},
        {"threshold": True},
        {"above_threshold": 1},
        {"above_threshold": False},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": -1.0},
        {"score": True},
        {"spatial": np.zeros((10, 10), dtype=np.float32)},
        {"spatial": np.zeros((128, 128), dtype=np.uint8)},
        {"spatial": np.full((128, 128), np.nan)},
        {"spatial": np.full((128, 128), -1.0)},
        {"original": np.zeros((10, 10, 3), dtype=np.uint8)},
        {"reference": np.zeros((128, 128, 3))},
        {"reconstruction": np.full((128, 128, 3), np.inf)},
        {"total_ms": float("inf")},
        {"inference_ms": True},
    ],
)
def test_malformed_inspection_results_rejected_without_values(change):
    pixels, result = fixture_result()
    with pytest.raises(AdmissionInvalid) as caught:
        validate_inspection(replace(result, **change), spec(), pixels)
    assert str(caught.value) == "Invalid development admission metadata."


@pytest.mark.parametrize("component", ["score", "map", "both"])
def test_missing_components_remain_unavailable(monkeypatch, component):
    original = SyntheticInspectionDouble.evaluate

    def evaluate(self, pixels):
        result = original(self, pixels)
        return replace(
            result,
            score=None if component in ("score", "both") else result.score,
            above_threshold=None if component in ("score", "both") else result.above_threshold,
            spatial=None if component in ("map", "both") else result.spatial,
        )

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert sum(r.attempted for r in rows) == 5
    if component == "both":
        assert all(
            r.status is ProbeStatus.NOT_APPLICABLE and r.score.signed is None and r.spatial is None
            for r in rows
        )
    elif component == "map":
        assert all(r.spatial is None for r in rows)
    else:
        assert all(r.score.signed is None and r.spatial.mae is not None for r in rows)


def test_failure_consumes_attempt_sanitizes_and_blocks_missing_baseline(monkeypatch):
    calls = []

    def fail(self, pixels):
        calls.append(1)
        raise RuntimeError("fabricated-private-path-and-token")

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", fail)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert len(calls) == sum(r.attempted for r in rows) == 1
    assert rows[0].reason is ProbeReason.DOUBLE_ERROR
    assert all(r.reason is ProbeReason.BASELINE_UNAVAILABLE for r in rows[1:])
    assert "fabricated-private" not in repr(rows)
    assert all(r.score is r.spatial is None for r in rows)


def test_later_failure_has_no_retry_or_anchor_replacement(monkeypatch):
    original = SyntheticInspectionDouble.evaluate
    calls = 0

    def evaluate(self, pixels):
        nonlocal calls
        calls += 1
        return None if calls == 2 else original(self, pixels)

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert calls == sum(r.attempted for r in rows) == 5
    assert rows[1].reason is ProbeReason.INVALID_OUTPUT
    assert rows[2].score.absolute == 0.0 and rows[-1].score.signed > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("category", "pcb1"),
        ("category", "pcb2"),
        ("category", "metalparts"),
        ("model_id", "feature_memory"),
        ("frozen_threshold", None),
        ("frozen_threshold", float("nan")),
        ("frozen_threshold", True),
        ("registered_artifact_sha256", "unverified"),
    ],
)
def test_interface_model_compatibility(field, value):
    with pytest.raises(AdmissionInvalid):
        InterfaceSpec(replace(spec().model, **{field: value}))


def test_no_supplied_arrays_callback_adapter_or_geometry():
    double = SyntheticInspectionDouble(spec())
    with pytest.raises(TypeError):
        exercise_synthetic_interface(double, pixels=np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(TypeError):
        exercise_synthetic_interface(double, rotation=1.0)
    with pytest.raises(AdmissionInvalid):
        exercise_synthetic_interface(lambda pixels: pixels)

    class Subclass(SyntheticInspectionDouble):
        pass

    with pytest.raises(AdmissionInvalid):
        exercise_synthetic_interface(Subclass(spec()))
    with pytest.raises(AdmissionInvalid):
        replace(spec(), map_space="native")
    with pytest.raises(AdmissionInvalid):
        replace(spec(), threshold_rule=ThresholdRule.GREATER_EQUAL)


def test_arrays_detached_and_probe_has_only_scalar_diagnostics():
    pixels, result = fixture_result()
    assert type(result) is InspectionResult
    output = validate_inspection(result, spec(), pixels)
    result.spatial[:] = 7
    assert not output.spatial.flags.writeable and output.spatial.max() < 1
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    encoded = json.dumps([asdict(r) for r in rows], allow_nan=False)
    assert all(word not in encoded for word in ("sha256", "filename", "original", "pixels"))
    with pytest.raises(ValueError):
        synthetic_report_json(rows)
    with pytest.raises(AdmissionInvalid):
        replace(rows[0], model_execution="real_model")
    with pytest.raises(AdmissionInvalid):
        replace(rows[0], kind="historical")
    with pytest.raises(AdmissionInvalid):
        CompatibilityObservation(True, ProbeStatus.FAILED, ProbeReason.DOUBLE_ERROR, 1)


def test_probe_has_no_filesystem_or_decode_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("No filesystem access during interface exercise")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    assert len(exercise_synthetic_interface(SyntheticInspectionDouble(spec()))) == 5


def test_importing_foundation_does_not_load_services_or_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import sys
import factoryvision.robustness.development_admission
import factoryvision.robustness.development_boundary
assert 'torch' not in sys.modules
assert 'factoryvision.inference.service' not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stderr == ""


def test_preparation_and_assessment_failures_are_sanitized_and_accounted(monkeypatch):
    original = boundary.perturb
    calls = 0

    def transform(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fabricated-private-details")
        return original(*args, **kwargs)

    monkeypatch.setattr(boundary, "perturb", transform)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert sum(r.attempted for r in rows) == 4
    assert rows[1].reason is ProbeReason.PREPARATION_ERROR and rows[1].attempted == 0
    assert rows[2].score.absolute == 0

    def fail(*args, **kwargs):
        raise RuntimeError("fabricated-private-details")

    monkeypatch.setattr(boundary, "score_stability", fail)
    rows = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))
    assert rows[0].reason is ProbeReason.ASSESSMENT_ERROR
    assert sum(r.attempted for r in rows) == 1
    assert "fabricated-private" not in repr(rows)
    monkeypatch.setattr(boundary, "_generate", fail)
    with pytest.raises(AdmissionInvalid):
        exercise_synthetic_interface(SyntheticInspectionDouble(spec()))


@pytest.mark.parametrize(
    "change",
    [
        {"status": "measured"},
        {"attempted": True},
        {"attempted": 0},
        {"schema_version": True},
        {"protocol_version": 2},
        {"numerical_protocol_version": 1},
        {"protocol_id": "historical"},
        {"reason": ProbeReason.DOUBLE_ERROR},
        {"score": None},
    ],
)
def test_invalid_probe_status_and_schema_rejected(change):
    row = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))[0]
    with pytest.raises(AdmissionInvalid):
        replace(row, **change)


def test_probe_cannot_export_as_history_and_validated_metrics_keep_classification():
    from factoryvision.inference import model_registry

    row = exercise_synthetic_interface(SyntheticInspectionDouble(spec()))[0]
    with pytest.raises(model_registry.RegistryInvalid):
        model_registry.json_bytes(row)
    object.__setattr__(row.score, "classification", "historical")
    with pytest.raises(AdmissionInvalid):
        row.__post_init__()
