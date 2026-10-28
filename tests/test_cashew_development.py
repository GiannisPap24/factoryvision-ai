"""Fabricated declarations, internally generated RGB and explicit doubles only."""

import builtins
import io
import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from factoryvision.inference import model_registry, service, uploads
from factoryvision.inference.reporting import ModelInfo
from factoryvision.robustness import cashew_development as cd
from factoryvision.robustness import reporting, windows_isolation
from factoryvision.robustness.assessment import Crossing
from factoryvision.robustness.development_admission import (
    AdmissionInvalid,
    DevelopmentAdmission,
    ExecutionDenied,
    Review,
)
from factoryvision.robustness.development_boundary import (
    InterfaceSpec,
    ProbeStatus,
    SyntheticInspectionDouble,
)
from factoryvision.robustness.execution_ledger import Case, LedgerInvalid, Outcome


def spec():
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
    )  # Fabricated identities/threshold, never artifact or sample verification.


def preparation():
    return cd.CashewSyntheticPreparation(cd.SyntheticWorkerDouble(spec()))


@pytest.fixture(autouse=True)
def no_real_access(monkeypatch):
    violations = []

    def forbidden(*args, **kwargs):
        violations.append(True)
        raise AssertionError("Real model/asset access or process launch forbidden")

    for target, name in (
        (torch, "load"),
        (torch.nn.Module, "_call_impl"),
        (service.InspectionService, "__init__"),
        (service.InspectionService, "inspect"),
        (service.InspectionService, "_load"),
        (service.InspectionService, "_verify"),
        (service, "verified_asset"),
        (service, "load_checkpoint"),
        (windows_isolation.SyntheticSupervisor, "run"),
        (Path, "rglob"),
        (Path, "glob"),
    ):
        monkeypatch.setattr(target, name, forbidden)
    # Keep the real decoder and all its validation, with fabricated empty blocklists.
    # No source-image identity metadata or asset file is needed by these fixtures.
    monkeypatch.setattr(uploads, "protected_hashes", lambda: (frozenset(), frozenset()))
    yield
    assert not violations  # A swallowed exception must not hide forbidden access.


def test_exact_plan_lossless_transport_and_existing_preprocess_order(monkeypatch):
    probe = preparation()
    original_evaluate = SyntheticInspectionDouble.evaluate
    original_inspect = cd.SyntheticWorkerDouble.inspect
    original_generate = cd._generate
    seen, prepared, originals, payloads = [], [], [], []

    def generate(parent):
        pixels = original_generate(parent)
        originals.append(pixels)
        return pixels

    def inspect(self, payload):
        payloads.append(payload)
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(io.BytesIO(payload)) as image:
            assert image.mode == "RGB" and image.size == (10, 8)
            assert not image.getexif() and not image.info
        return original_inspect(self, payload)

    def evaluate(self, pixels):
        assert pixels.shape == (8, 10, 3) and pixels.dtype == np.uint8
        assert not pixels.flags.writeable
        seen.append(pixels.copy())
        # Actual existing preprocessing, applied only to owned synthetic RGB.
        # No NN, model construction, checkpoint loader or alternative preprocessing.
        with Image.fromarray(pixels) as image:
            prepared.append(service.preprocess(image))
        return original_evaluate(self, pixels)

    monkeypatch.setattr(cd, "_generate", generate)
    monkeypatch.setattr(cd.SyntheticWorkerDouble, "inspect", inspect)
    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    report = probe.run()
    assert report.reason is cd.StopReason.COMPLETE
    assert len(seen) == len(payloads) == len(report.entries) == 5
    assert len(originals) == 1 and not originals[0].flags.writeable
    for i in range(3):
        np.testing.assert_array_equal(seen[i], originals[0])
        assert report.observations[i].score.absolute == report.observations[i].spatial.mae == 0
    for i, factor in ((3, 0.9), (4, 1.1)):
        expected = np.rint(np.clip(originals[0].astype(np.float64) * factor, 0, 255)).astype(
            np.uint8
        )
        np.testing.assert_array_equal(seen[i], expected)
        with Image.fromarray(expected) as image:
            assert torch.equal(prepared[i], service.preprocess(image))
    assert all(
        value.shape == (1, 3, 128, 128) and value.dtype == torch.float32 for value in prepared
    )
    assert [e.case for e in report.entries] == list(Case)
    assert all(e.outcome is Outcome.SUCCEEDED and e.charged == 1 for e in report.entries)
    assert probe._worker._state == "closed"


def test_lifecycle_reservation_dispatch_and_single_session(monkeypatch):
    probe = preparation()
    events = []
    for name in ("start_suspended", "verify", "resume", "inspect", "close"):
        original = getattr(cd.SyntheticWorkerDouble, name)

        def observe(self, *args, method=original, label=name):
            events.append(label)
            entry = probe._ledger.entries[-1]
            if label in ("start_suspended", "verify"):
                assert entry.outcome is Outcome.RESERVED and entry.charged == 0
            if label in ("resume", "inspect"):
                assert entry.outcome is Outcome.DISPATCHED and entry.charged == 1
            return method(self, *args)

        monkeypatch.setattr(cd.SyntheticWorkerDouble, name, observe)
    probe.run()
    assert events == ["start_suspended", "verify", "resume", *["inspect"] * 5, "close"]
    with pytest.raises(AdmissionInvalid):
        probe.run()
    with pytest.raises(LedgerInvalid):
        probe._ledger.reserve(Case.FIRST)
    assert events.count("inspect") == 5 and events.count("start_suspended") == 1


@pytest.mark.parametrize(
    "model", ["spatial_mean", "feature_memory", "student_teacher", "metalparts_autoencoder"]
)
def test_unsupported_models_rejected(model):
    with pytest.raises(AdmissionInvalid):
        cd.SyntheticWorkerDouble(InterfaceSpec(replace(spec().model, model_id=model)))


def test_no_real_service_callback_subclass_pixels_paths_or_override():
    class Subclass(cd.SyntheticWorkerDouble):
        pass

    for worker in (object.__new__(service.InspectionService), lambda: 1, Subclass(spec())):
        with pytest.raises(AdmissionInvalid):
            cd.CashewSyntheticPreparation(worker)
    for kwargs in (
        {"pixels": np.zeros((2, 2, 3))},
        {"path": "fabricated"},
        {"authorized": True},
        {"conditions": ()},
    ):
        with pytest.raises(TypeError):
            preparation().run(**kwargs)


def test_scores_and_raw_maps_not_recomputed_or_normalized(monkeypatch):
    original = SyntheticInspectionDouble.evaluate
    scores = iter([0.5, 0.5, 0.5, 0.4, 0.7])
    maps = iter([2.0, 2.0, 2.0, 4.0, 8.0])

    def evaluate(self, pixels):
        result = original(self, pixels)
        score = next(scores)
        return replace(
            result,
            score=score,
            above_threshold=score > 0.5,
            spatial=np.full((128, 128), next(maps)),
        )

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    report = preparation().run()
    assert report.threshold.value == 0.5
    assert report.observations[0].score.crossing is Crossing.NONE
    assert report.observations[3].score.crossing is Crossing.NONE  # equality baseline is not above
    assert report.observations[4].score.crossing is Crossing.UP
    assert report.observations[-1].score.signed == pytest.approx(0.2)
    assert report.observations[-1].spatial.mae == report.observations[-1].spatial.rmse == 6.0
    assert report.observations[-1].spatial.maximum_absolute == 6.0


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "feature_memory"},
        {"threshold": 0.7},
        {"above_threshold": "true"},
        {"artifact_sha256": "c" * 64},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": -1.0},
        {"spatial": np.zeros((10, 8))},
        {"spatial": np.zeros((128, 128), dtype=np.uint8)},
        {"spatial": np.full((128, 128), np.nan)},
        {"spatial": np.full((128, 128), -1.0)},
        {"original": np.zeros((8, 10, 3), dtype=np.uint8)},
        {"total_ms": float("inf")},
    ],
)
def test_invalid_results_abort_charge_and_remain_sanitized(monkeypatch, change):
    original = SyntheticInspectionDouble.evaluate
    monkeypatch.setattr(
        SyntheticInspectionDouble, "evaluate", lambda self, p: replace(original(self, p), **change)
    )
    report = preparation().run()
    assert report.reason is cd.StopReason.INVALID_OUTPUT
    assert len(report.entries) == 1 and report.entries[0].outcome is Outcome.FAILED
    assert report.entries[0].charged == 1 and report.observations == ()
    assert "sha256" not in cd.json_bytes(report).decode()


@pytest.mark.parametrize("missing", ["score", "map", "both", "result"])
def test_unavailable_results_do_not_fabricate_metrics(monkeypatch, missing):
    original = SyntheticInspectionDouble.evaluate

    def evaluate(self, pixels):
        result = original(self, pixels)
        if missing == "result":
            return None
        return replace(
            result,
            score=None if missing in ("score", "both") else result.score,
            above_threshold=None if missing in ("score", "both") else result.above_threshold,
            spatial=None if missing in ("map", "both") else result.spatial,
        )

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    report = preparation().run()
    if missing == "result":
        assert report.reason is cd.StopReason.INVALID_OUTPUT and not report.observations
    else:
        assert len(report.observations) == 5
        if missing == "both":
            assert all(row.status is ProbeStatus.NOT_APPLICABLE for row in report.observations)
        if missing in ("score", "both"):
            assert all(row.score.signed is None for row in report.observations)
        if missing in ("map", "both"):
            assert all(row.spatial is None for row in report.observations)


@pytest.mark.parametrize("ordinal", [1, 2])
@pytest.mark.parametrize(
    "error,outcome,reason",
    [
        (TimeoutError, Outcome.TIMED_OUT, cd.StopReason.TIMED_OUT),
        (InterruptedError, Outcome.CANCELLED, cd.StopReason.CANCELLED),
        (RuntimeError, Outcome.UNKNOWN, cd.StopReason.UNKNOWN),
        (KeyboardInterrupt, Outcome.UNKNOWN, cd.StopReason.UNKNOWN),
    ],
)
def test_interrupted_worker_abort_only_no_refund_or_replay(
    monkeypatch, ordinal, error, outcome, reason
):
    original = cd.SyntheticWorkerDouble.inspect
    calls = []

    def inspect(self, payload):
        calls.append(1)
        if len(calls) == ordinal:
            raise error("fabricated-private-error")
        return original(self, payload)

    monkeypatch.setattr(cd.SyntheticWorkerDouble, "inspect", inspect)
    probe = preparation()
    report = probe.run()
    assert report.reason is reason and report.entries[-1].outcome is outcome
    assert len(calls) == sum(e.charged for e in report.entries) == ordinal
    assert len(report.observations) == ordinal - 1
    with pytest.raises(AdmissionInvalid):
        probe.run()
    assert len(calls) == ordinal
    assert "fabricated-private" not in cd.json_bytes(report).decode()


@pytest.mark.parametrize(
    "stage,reason,charged",
    [
        ("start_suspended", cd.StopReason.ISOLATION, 0),
        ("verify", cd.StopReason.ISOLATION, 0),
        ("resume", cd.StopReason.UNKNOWN, 1),
    ],
)
def test_control_failures_before_resume_fail_closed(monkeypatch, stage, reason, charged):
    calls = []

    def fail(*args):
        raise RuntimeError("fabricated-private-control-error")

    monkeypatch.setattr(cd.SyntheticWorkerDouble, stage, fail)
    monkeypatch.setattr(cd.SyntheticWorkerDouble, "inspect", lambda *args: calls.append(1))
    report = preparation().run()
    assert not calls and report.reason is reason and report.entries[0].charged == charged
    assert report.entries[0].outcome is (Outcome.UNKNOWN if charged else Outcome.NOT_STARTED)


def test_unverified_lifecycle_never_resumes(monkeypatch):
    monkeypatch.setattr(cd.SyntheticWorkerDouble, "verify", lambda self: False)
    report = preparation().run()
    assert report.reason is cd.StopReason.ISOLATION and report.entries[0].charged == 0


def test_preparation_failure_retains_reservation_without_dispatch(monkeypatch):
    original = cd._png
    calls = []

    def encode(pixels):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("fabricated-private-encoder-error")
        return original(pixels)

    monkeypatch.setattr(cd, "_png", encode)
    report = preparation().run()
    assert report.reason is cd.StopReason.PREPARATION
    assert len(report.entries) == 2 and sum(e.charged for e in report.entries) == 1
    assert report.entries[-1].outcome is Outcome.NOT_STARTED
    data = json.loads(cd.json_bytes(report))
    assert (data["reserved"], data["charged"], data["unreserved"], data["not_dispatched"]) == (
        2,
        1,
        3,
        4,
    )


@pytest.mark.parametrize("after_dispatch", [False, True])
def test_cancellation_retains_capacity_and_stops(monkeypatch, after_dispatch):
    event = threading.Event()
    original = cd.SyntheticWorkerDouble.inspect
    if not after_dispatch:
        event.set()
    else:

        def inspect(self, payload):
            result = original(self, payload)
            event.set()
            return result

        monkeypatch.setattr(cd.SyntheticWorkerDouble, "inspect", inspect)
    report = preparation().run(cancel=event)
    assert report.reason is cd.StopReason.CANCELLED and len(report.entries) == 1
    assert report.entries[0].charged == int(after_dispatch) and not report.observations


def test_cleanup_uncertainty_blocks_reuse_without_refund(monkeypatch):
    monkeypatch.setattr(cd.SyntheticWorkerDouble, "close", lambda self: False)
    probe = preparation()
    report = probe.run()
    assert report.reason is cd.StopReason.CLEANUP and sum(e.charged for e in report.entries) == 5
    assert json.loads(cd.json_bytes(report))["run_status"] == "unknown"
    with pytest.raises(AdmissionInvalid):
        probe.run()


@pytest.mark.parametrize("review", list(Review))
def test_admission_never_allows_real_binding(review):
    with pytest.raises(ExecutionDenied):
        cd.prepare_development(DevelopmentAdmission(review=review))


def test_no_hidden_binding_even_if_gate_returns(monkeypatch):
    monkeypatch.setattr(cd, "require_execution", lambda record: None)
    with pytest.raises(ExecutionDenied):
        cd.prepare_development(DevelopmentAdmission())


def test_no_filesystem_access_during_fixture_or_gate(monkeypatch):
    violations = []

    def forbidden(*args, **kwargs):
        violations.append(True)
        raise AssertionError("No filesystem access in integration fixture")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    assert preparation().run().reason is cd.StopReason.COMPLETE
    with pytest.raises(ExecutionDenied):
        cd.prepare_development(DevelopmentAdmission())
    assert not violations


def test_scalar_report_label_bounds_and_schema_separation():
    report = preparation().run()
    raw = cd.json_bytes(report)
    data = json.loads(raw)
    assert len(raw) <= 16384
    assert data["kind"] == "synthetic_cashew_integration_probe"
    assert data["model_execution"] == "not_run" and data["isolation"] == "lifecycle_double_only"
    assert data["charged"] == data["reserved"] == data["planned_attempts"] == 5
    assert data["conditions"] == [
        {"kind": "identity", "severity": None},
        {"kind": "identity", "severity": None},
        {"kind": "identity", "severity": None},
        {"kind": "brightness", "severity": 0.9},
        {"kind": "brightness", "severity": 1.1},
    ]
    assert not any(
        word in raw.decode()
        for word in ("sha256", "pixels", "filename", "environment", "reconstruction")
    )
    for exporter in (reporting.json_bytes, reporting.csv_bytes):
        with pytest.raises(ValueError):
            exporter(report)
    with pytest.raises(model_registry.RegistryInvalid):
        model_registry.json_bytes(report)
    for change in (
        {"kind": "real_development"},
        {"model_execution": "real_model"},
        {"isolation": "verified_job"},
        {"schema_version": True},
    ):
        forged = preparation().run()
        for key, value in change.items():
            object.__setattr__(forged, key, value)
        with pytest.raises(AdmissionInvalid):
            cd.json_bytes(forged)


def test_invalid_report_accounting_and_worker_metadata_changes_rejected():
    report = preparation().run()
    with pytest.raises(AdmissionInvalid):
        replace(report, entries=report.entries[:-1])
    with pytest.raises(AdmissionInvalid):
        replace(report, observations=())
    with pytest.raises(AdmissionInvalid):
        replace(report, reason=cd.StopReason.UNKNOWN)
    probe = preparation()
    probe._worker.spec = replace(spec(), model=replace(spec().model, frozen_threshold=0.6))
    with pytest.raises(AdmissionInvalid):
        probe.run()


def test_no_replacement_baseline_after_missing_initial_outputs(monkeypatch):
    original = SyntheticInspectionDouble.evaluate
    count = 0

    def evaluate(self, pixels):
        nonlocal count
        count += 1
        result = original(self, pixels)
        return (
            replace(result, score=None, above_threshold=None, spatial=None)
            if count == 1
            else result
        )

    monkeypatch.setattr(SyntheticInspectionDouble, "evaluate", evaluate)
    report = preparation().run()
    assert count == 5
    assert all(row.status is ProbeStatus.NOT_APPLICABLE for row in report.observations)
    assert all(row.score.signed is None and row.spatial is None for row in report.observations)


def test_replaced_inner_double_cannot_execute_callback():
    probe = preparation()
    calls = []

    class Unapproved:
        def evaluate(self, pixels):
            calls.append(1)

    probe._worker._double = Unapproved()
    report = probe.run()
    assert not calls and report.reason is cd.StopReason.UNKNOWN
    assert report.entries[0].charged == 1
