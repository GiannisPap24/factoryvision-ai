"""Cashew integration preparation, exercised only with internal synthetic pixels/doubles.

No real-service binding, asset root, process launcher, callback, execution grant or
real-image entry point exists here. The lifecycle double does NOT provide OS isolation.
"""

import io
import threading
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from PIL import Image

from factoryvision.inference.uploads import decode_upload, readonly
from factoryvision.robustness.assessment import (
    FrozenThreshold,
    GridConvention,
    MeasurementStatus,
    ThresholdRule,
    map_stability,
    score_stability,
)
from factoryvision.robustness.contracts import Category, Kind, Parent, ParentID, Perturbation, Plan
from factoryvision.robustness.development_admission import (
    ExecutionDenied,
    require,
    require_execution,
)
from factoryvision.robustness.development_boundary import (
    CompatibilityObservation,
    InterfaceSpec,
    ProbeReason,
    ProbeStatus,
    SyntheticInspectionDouble,
    validate_inspection,
)
from factoryvision.robustness.execution_ledger import Case, Entry, ExecutionLedger, Outcome
from factoryvision.robustness.perturbations import _rgb, perturb
from factoryvision.robustness.reporting import _bounded, _json, schedule
from factoryvision.robustness.runner import _generate


def prepare_development(admission):
    """Gate-only preparation: never constructs a service or worker, even after gate changes."""
    require_execution(admission)
    raise ExecutionDenied()  # No real binding exists in this phase.


def _plan():
    return Plan(
        Category.CASHEW,
        (Parent(ParentID.SECOND, 10, 8),),
        (
            Perturbation(Kind.IDENTITY),
            Perturbation(Kind.BRIGHTNESS, 0.9),
            Perturbation(Kind.BRIGHTNESS, 1.1),
        ),
    )


def _png(pixels):
    """Fresh RGB PNG in memory, no resize/normalization or carried image metadata."""
    _rgb(pixels)
    with io.BytesIO() as buffer:
        with Image.frombytes("RGB", (pixels.shape[1], pixels.shape[0]), pixels.tobytes()) as image:
            image.save(buffer, format="PNG", compress_level=1)
        payload = buffer.getvalue()
    require(0 < len(payload) <= 4096)  # Fixed tiny internal parent only.
    return payload


class SyntheticWorkerDouble:
    """Single simulated session around the existing analytical InspectionResult double.

    State transitions model the future seam only: no Job Object, watchdog, resource
    enforcement or actual preprocessing/model execution is represented by this double.
    """

    def __init__(self, spec: InterfaceSpec):
        require(type(spec) is InterfaceSpec)
        spec.__post_init__()
        self.spec = spec
        self._double = SyntheticInspectionDouble(spec)
        self._state = "new"

    def start_suspended(self):
        require(self._state == "new")
        self._state = "suspended"

    def verify(self):
        require(self._state == "suspended")
        self._state = "verified"
        return True

    def resume(self):
        require(self._state == "verified")
        self._state = "running"

    def inspect(self, payload):
        require(self._state == "running")
        require(type(self._double) is SyntheticInspectionDouble and self._double.spec == self.spec)
        require(type(payload) is bytes and 0 < len(payload) <= 4096)
        require(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        with decode_upload(payload) as image:
            return self._double.evaluate(readonly(image))

    def close(self):
        self._state = "closed"
        return True


class StopReason(StrEnum):
    COMPLETE = "complete"
    PREPARATION = "preparation_failed"
    ISOLATION = "simulated_isolation_failed"
    INVALID_OUTPUT = "invalid_output"
    ASSESSMENT = "assessment_failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    UNKNOWN = "execution_unknown"
    CLEANUP = "cleanup_unverified"


@dataclass(frozen=True, slots=True)
class SyntheticIntegrationReport:
    entries: tuple[Entry, ...]
    observations: tuple[CompatibilityObservation, ...]
    threshold: FrozenThreshold
    reason: StopReason
    kind: str = field(default="synthetic_cashew_integration_probe", init=False)
    model_id: str = field(default="autoencoder", init=False)
    model_execution: str = field(default="not_run", init=False)
    isolation: str = field(default="lifecycle_double_only", init=False)
    schema_version: int = field(default=1, init=False)
    numerical_protocol_version: int = field(default=2, init=False)

    def __post_init__(self):
        for key, value in (
            ("kind", "synthetic_cashew_integration_probe"),
            ("model_id", "autoencoder"),
            ("model_execution", "not_run"),
            ("isolation", "lifecycle_double_only"),
            ("schema_version", 1),
            ("numerical_protocol_version", 2),
        ):
            require(type(getattr(self, key)) is type(value) and getattr(self, key) == value)
        require(type(self.threshold) is FrozenThreshold)
        self.threshold.__post_init__()
        require(self.threshold.rule is ThresholdRule.GREATER and self.threshold.value >= 0)
        require(type(self.reason) is StopReason)
        require(type(self.entries) is tuple and 1 <= len(self.entries) <= 5)
        require(type(self.observations) is tuple)
        for index, entry in enumerate(self.entries):
            require(type(entry) is Entry)
            entry.__post_init__()
            require(entry.case is tuple(Case)[index])
            require(entry.outcome not in (Outcome.RESERVED, Outcome.DISPATCHED))
            if index < len(self.entries) - 1:
                require(entry.outcome is Outcome.SUCCEEDED)
        successful = sum(e.outcome is Outcome.SUCCEEDED for e in self.entries)
        require(len(self.observations) == successful)
        for ordinal, row in enumerate(self.observations, 1):
            require(type(row) is CompatibilityObservation)
            row.__post_init__()
            require(
                row.ordinal == ordinal
                and row.status in (ProbeStatus.MEASURED, ProbeStatus.NOT_APPLICABLE)
            )
        last = self.entries[-1]
        if self.reason is StopReason.COMPLETE:
            require(successful == 5)
        elif self.reason is not StopReason.CLEANUP:
            expected = {
                StopReason.PREPARATION: Outcome.NOT_STARTED,
                StopReason.ISOLATION: Outcome.NOT_STARTED,
                StopReason.INVALID_OUTPUT: Outcome.FAILED,
                StopReason.ASSESSMENT: Outcome.FAILED,
                StopReason.CANCELLED: Outcome.CANCELLED,
                StopReason.TIMED_OUT: Outcome.TIMED_OUT,
                StopReason.UNKNOWN: Outcome.UNKNOWN,
            }[self.reason]
            require(last.outcome is expected)
        if successful == len(self.entries):
            require(successful == 5)


def json_bytes(report: SyntheticIntegrationReport):
    """Separate synthetic scalar envelope. No development/historical report or file write."""
    require(type(report) is SyntheticIntegrationReport)
    report.__post_init__()
    data = asdict(report)
    data.update(
        planned_attempts=5,
        reserved=len(report.entries),
        charged=sum(e.charged for e in report.entries),
        unreserved=5 - len(report.entries),
        not_dispatched=5 - sum(e.charged for e in report.entries),
        run_status="complete"
        if report.reason is StopReason.COMPLETE
        else (
            "unknown" if report.reason in (StopReason.UNKNOWN, StopReason.CLEANUP) else "aborted"
        ),
        recovery="abort_only",
        conditions=[
            {"kind": item[4].kind, "severity": item[4].severity} for item in schedule(_plan())
        ],
    )
    return _bounded(_json(data), 16384)


class CashewSyntheticPreparation:
    """Single-use integration exercise; exact five-condition plan, no supplied pixels.

    A new instance is a separate explicitly initiated synthetic test, never automatic
    recovery of an interrupted run. No real worker entry point is installed here.
    """

    def __init__(self, worker: SyntheticWorkerDouble):
        require(type(worker) is SyntheticWorkerDouble and worker._state == "new")
        require(type(worker.spec) is InterfaceSpec)
        worker.spec.__post_init__()
        self._worker = worker
        self._spec = worker.spec
        self._ledger = ExecutionLedger(5)
        self._used = False
        self._lock = threading.Lock()

    def run(self, *, cancel=None):
        require(cancel is None or type(cancel) is threading.Event)
        require(self._lock.acquire(blocking=False))
        try:
            require(not self._used and self._worker._state == "new")
            require(self._worker.spec == self._spec)
            self._spec.__post_init__()
            self._used = True
            return self._run(cancel)
        finally:
            self._lock.release()

    def _run(self, cancel):
        plan = _plan()
        require(plan.planned_inference_calls == 5)
        threshold = FrozenThreshold(self._spec.model.frozen_threshold, ThresholdRule.GREATER)
        original = baseline = None
        observations = []
        reason = StopReason.COMPLETE
        try:
            for ordinal, item in enumerate(schedule(plan), 1):
                self._ledger.reserve(tuple(Case)[ordinal - 1])
                outcome, reason = Outcome.NOT_STARTED, StopReason.PREPARATION
                try:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError()
                    parent, _, _, _, condition, _ = item
                    if original is None:
                        original = _generate(parent)
                    transformed = perturb(original, condition, plan=plan, parent=parent)
                    payload = _png(transformed.pixels)
                    if ordinal == 1:
                        reason = StopReason.ISOLATION
                        self._worker.start_suspended()
                        require(self._worker.verify() is True)
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError()
                    self._ledger.dispatch()  # Before first resume AND every inspect request.
                    outcome, reason = Outcome.UNKNOWN, StopReason.UNKNOWN
                    if ordinal == 1:
                        self._worker.resume()
                    result = self._worker.inspect(payload)
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError()
                    outcome, reason = Outcome.FAILED, StopReason.INVALID_OUTPUT
                    output = validate_inspection(result, self._spec, transformed.pixels)
                    reason = StopReason.ASSESSMENT
                    anchor = output if ordinal == 1 else baseline
                    score = score_stability(anchor.score, output.score, threshold=threshold)
                    spatial = None
                    if anchor.spatial is not None and output.spatial is not None:
                        spatial = map_stability(
                            anchor.spatial,
                            output.spatial,
                            perturbation=condition,
                            image_size=(128, 128),
                            reference_grid=GridConvention.NATIVE,
                            candidate_grid=GridConvention.NATIVE,
                        )  # Native to the preprocessing grid, not the original source image.
                    measured = score.status is MeasurementStatus.MEASURED or spatial is not None
                    observations.append(
                        CompatibilityObservation(
                            ordinal,
                            ProbeStatus.MEASURED if measured else ProbeStatus.NOT_APPLICABLE,
                            ProbeReason.NONE if measured else ProbeReason.UNAVAILABLE,
                            1,
                            score,
                            spatial,
                        )
                    )
                    if ordinal == 1:
                        baseline = output
                    outcome, reason = Outcome.SUCCEEDED, StopReason.COMPLETE
                except InterruptedError:
                    outcome, reason = Outcome.CANCELLED, StopReason.CANCELLED
                except TimeoutError:
                    outcome, reason = Outcome.TIMED_OUT, StopReason.TIMED_OUT
                except BaseException:
                    pass  # Retain the conservative stage outcome; never expose exception text.
                self._ledger.finish(outcome)
                if outcome is not Outcome.SUCCEEDED:
                    break  # Abort-only, including known failures; no retry or replacement baseline.
        finally:
            try:
                require(self._worker.close() is True)
            except BaseException:
                reason = StopReason.CLEANUP
        return SyntheticIntegrationReport(
            self._ledger.entries, tuple(observations), threshold, reason
        )
