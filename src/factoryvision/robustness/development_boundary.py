"""Unexecuted service compatibility boundary, exercised only by an internal synthetic double.

No service instance, model loader, supplied image, path or callback is accepted by the probe.
InspectionResult is imported lazily for type compatibility, never to construct a service.
"""

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from factoryvision.inference.reporting import ModelInfo
from factoryvision.inference.uploads import readonly
from factoryvision.robustness.assessment import (
    FrozenThreshold,
    GridConvention,
    MapStability,
    MeasurementStatus,
    ScoreStability,
    ThresholdRule,
    map_stability,
    score_stability,
)
from factoryvision.robustness.contracts import (
    Category,
    Classification,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
)
from factoryvision.robustness.development_admission import AdmissionInvalid, require
from factoryvision.robustness.perturbations import _rgb, perturb
from factoryvision.robustness.runner import _generate


class MapSpace(StrEnum):
    PREPROCESSED_128 = "cashew_preprocessed_128_grid"


@dataclass(frozen=True, slots=True)
class InterfaceSpec:
    """Declared ModelInfo for compatibility checks, NOT verification of a model artifact."""

    model: ModelInfo
    map_space: MapSpace = MapSpace.PREPROCESSED_128
    threshold_rule: ThresholdRule = ThresholdRule.GREATER

    def __post_init__(self):
        require(type(self.model) is ModelInfo)
        m = self.model
        for value, expected in (
            (m.model_id, "autoencoder"),
            (m.category, "cashew"),
            (m.name, "Convolutional autoencoder"),
            (m.score_label, "Mean RGB squared reconstruction error"),
            (
                m.scientific_status,
                "Historical Cashew research; frozen normal-calibration threshold only.",
            ),
        ):
            require(type(value) is str and value == expected)
        for value in (m.registered_artifact_sha256, m.catalog_sha256):
            require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None)
        require(
            type(m.frozen_threshold) is float
            and math.isfinite(m.frozen_threshold)
            and m.frozen_threshold >= 0
        )
        require(type(self.map_space) is MapSpace)
        require(self.threshold_rule is ThresholdRule.GREATER)


@dataclass(frozen=True, slots=True, eq=False)
class InterfaceOutput:
    """Unverified in-memory numerical view, not an execution/admission receipt."""

    score: float | None
    spatial: np.ndarray | None
    map_space: MapSpace = MapSpace.PREPROCESSED_128

    def __post_init__(self):
        require(type(self.map_space) is MapSpace)
        if self.score is not None:
            require(type(self.score) is float and math.isfinite(self.score) and self.score >= 0)
        if self.spatial is not None:
            require(type(self.spatial) is np.ndarray and self.spatial.shape == (128, 128))
            require(self.spatial.dtype in (np.float32, np.float64))
            require(np.isfinite(self.spatial).all() and (self.spatial >= 0).all())
            object.__setattr__(self, "spatial", readonly(self.spatial))


def validate_inspection(result, spec: InterfaceSpec, original: np.ndarray) -> InterfaceOutput:
    """Validate an existing result without executing, authenticating or exporting it.

    A float display-normalized map cannot be recognized from its bytes alone. The future
    verified service binding must establish raw-map origin. No geometry mapping is asserted.
    """
    from factoryvision.inference.service import InspectionResult

    try:
        require(type(spec) is InterfaceSpec)
        spec.__post_init__()
        _rgb(original)
        require(type(result) is InspectionResult)
        m = spec.model
        for value, expected in (
            (result.model_id, m.model_id),
            (result.model_name, m.name),
            (result.score_label, m.score_label),
            (result.artifact_sha256, m.registered_artifact_sha256),
        ):
            require(type(value) is str and value == expected)
        require(type(result.threshold) is float and result.threshold == m.frozen_threshold)
        output = InterfaceOutput(result.score, result.spatial)
        if output.score is None:
            require(result.above_threshold is None)
        else:
            require(type(result.above_threshold) is bool)
            require(result.above_threshold == (output.score > m.frozen_threshold))
        _rgb(result.original)
        require(np.array_equal(original, result.original))
        require(result.reference is None)
        if result.reconstruction is not None:
            value = result.reconstruction
            require(type(value) is np.ndarray and value.shape == (128, 128, 3))
            require(value.dtype in (np.float32, np.float64) and np.isfinite(value).all())
            require((value >= 0).all() and (value <= 1).all())
        for value in (
            result.preprocessing_ms,
            result.inference_ms,
            result.artifact_check_load_ms,
            result.total_ms,
        ):
            require(type(value) is float and math.isfinite(value) and value >= 0)
        return output
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise AdmissionInvalid() from None


@dataclass(frozen=True, slots=True)
class SyntheticInspectionDouble:
    """Analytical interface fixture, not an autoencoder or a real-model adapter."""

    spec: InterfaceSpec

    def __post_init__(self):
        require(type(self.spec) is InterfaceSpec)
        self.spec.__post_init__()

    def evaluate(self, pixels):
        from factoryvision.inference.service import InspectionResult

        _rgb(pixels)
        m = self.spec.model
        # Constant raw map makes the interface probe analytically transparent. No NN/resize.
        value = float(np.mean(pixels, dtype=np.float64) / 255)
        spatial = np.full((128, 128), value, dtype=np.float32)
        score = float(spatial.mean())
        return InspectionResult(
            m.model_id,
            m.name,
            m.score_label,
            score,
            m.frozen_threshold,
            score > m.frozen_threshold,
            spatial,
            readonly(pixels),
            None,
            None,
            0.0,
            0.0,
            0.0,
            0.0,
            m.registered_artifact_sha256,
        )


class ProbeStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"


class ProbeReason(StrEnum):
    NONE = "none"
    UNAVAILABLE = "unavailable_output"
    PREPARATION_ERROR = "preparation_error"
    DOUBLE_ERROR = "synthetic_double_error"
    INVALID_OUTPUT = "invalid_output"
    ASSESSMENT_ERROR = "assessment_error"
    BASELINE_UNAVAILABLE = "baseline_unavailable"


@dataclass(frozen=True, slots=True)
class CompatibilityObservation:
    ordinal: int
    status: ProbeStatus
    reason: ProbeReason
    attempted: int
    score: ScoreStability | None = None
    spatial: MapStability | None = None
    kind: str = "synthetic_service_interface_probe"
    model_execution: str = "not_run"
    map_space: MapSpace = MapSpace.PREPROCESSED_128
    protocol_id: str = "factoryvision-synthetic-interface-probe"
    protocol_version: int = 1
    schema_version: int = 1
    numerical_protocol_version: int = 2

    def __post_init__(self):
        require(type(self.ordinal) is int and 1 <= self.ordinal <= 5)
        require(type(self.attempted) is int and self.attempted in (0, 1))
        require(type(self.status) is ProbeStatus and type(self.reason) is ProbeReason)
        require(type(self.kind) is str and self.kind == "synthetic_service_interface_probe")
        require(type(self.model_execution) is str and self.model_execution == "not_run")
        require(type(self.map_space) is MapSpace)
        require(
            type(self.protocol_id) is str
            and self.protocol_id == "factoryvision-synthetic-interface-probe"
        )
        for value, expected in (
            (self.protocol_version, 1),
            (self.schema_version, 1),
            (self.numerical_protocol_version, 2),
        ):
            require(type(value) is int and value == expected)
        if self.status in (ProbeStatus.MEASURED, ProbeStatus.NOT_APPLICABLE):
            require(self.attempted == 1 and type(self.score) is ScoreStability)
            self.score.__post_init__()
            if self.spatial is not None:
                require(type(self.spatial) is MapStability)
                self.spatial.__post_init__()
            for metric in (self.score, self.spatial):
                if metric is not None:
                    require(metric.classification is Classification.SOFTWARE_DIAGNOSTIC)
                    require(type(metric.protocol_version) is int and metric.protocol_version == 2)
            measured = self.score.status is MeasurementStatus.MEASURED or (
                self.spatial is not None and self.spatial.status is MeasurementStatus.MEASURED
            )
            require((self.status is ProbeStatus.MEASURED) == measured)
            require(self.reason is (ProbeReason.NONE if measured else ProbeReason.UNAVAILABLE))
        else:
            require(self.score is None and self.spatial is None)
            if self.status is ProbeStatus.NOT_EVALUATED:
                require(self.attempted == 0 and self.reason is ProbeReason.BASELINE_UNAVAILABLE)
            else:
                require(
                    self.reason
                    in (
                        ProbeReason.PREPARATION_ERROR,
                        ProbeReason.DOUBLE_ERROR,
                        ProbeReason.INVALID_OUTPUT,
                        ProbeReason.ASSESSMENT_ERROR,
                    )
                )
                require(
                    self.attempted == (0 if self.reason is ProbeReason.PREPARATION_ERROR else 1)
                )


def exercise_synthetic_interface(
    double: SyntheticInspectionDouble,
) -> tuple[CompatibilityObservation, ...]:
    """Fixed five-attempt fixture; no admission argument and no real-execution path.

    Numeric comparison uses the 128-square preprocessing frame as the comparison domain.
    NATIVE below means native to THAT frame, never native to the original RGB image.
    Only identity/brightness are used; no source-to-map alignment or mask projection occurs.
    """
    require(type(double) is SyntheticInspectionDouble)
    double.__post_init__()
    parent = Parent(ParentID.FIRST, 10, 10)
    specs = (
        Perturbation(Kind.IDENTITY),
        Perturbation(Kind.BRIGHTNESS, 0.9),
        Perturbation(Kind.BRIGHTNESS, 1.1),
    )
    plan = Plan(Category.CASHEW, (parent,), specs)
    try:
        original = _generate(parent)
    except Exception:
        raise AdmissionInvalid() from None
    baseline = None
    rows = []
    threshold = FrozenThreshold(double.spec.model.frozen_threshold, ThresholdRule.GREATER)
    for ordinal, spec in enumerate((specs[0], specs[0], specs[0], *specs[1:]), 1):
        if ordinal > 1 and baseline is None:
            rows.append(
                CompatibilityObservation(
                    ordinal, ProbeStatus.NOT_EVALUATED, ProbeReason.BASELINE_UNAVAILABLE, 0
                )
            )
            continue
        reason = ProbeReason.PREPARATION_ERROR
        attempted = 0
        try:
            transformed = perturb(original, spec, plan=plan, parent=parent)
            reason = ProbeReason.DOUBLE_ERROR
            attempted = 1
            result = double.evaluate(transformed.pixels)  # One attempt; never retried.
            reason = ProbeReason.INVALID_OUTPUT
            output = validate_inspection(result, double.spec, transformed.pixels)
            reason = ProbeReason.ASSESSMENT_ERROR
            anchor = output if ordinal == 1 else baseline
            score = score_stability(anchor.score, output.score, threshold=threshold)
            spatial = None
            if anchor.spatial is not None and output.spatial is not None:
                spatial = map_stability(
                    anchor.spatial,
                    output.spatial,
                    perturbation=spec,
                    image_size=(128, 128),
                    reference_grid=GridConvention.NATIVE,
                    candidate_grid=GridConvention.NATIVE,
                )
            measured = score.status is MeasurementStatus.MEASURED or spatial is not None
            rows.append(
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
        except Exception:
            rows.append(CompatibilityObservation(ordinal, ProbeStatus.FAILED, reason, attempted))
    return tuple(rows)
