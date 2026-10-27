"""Model-independent, synthetic numerical diagnostics; no inference or public exports."""

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from factoryvision.robustness.contracts import (
    PROTOCOL_VERSION,
    Classification,
    Kind,
    Perturbation,
    _require,
)
from factoryvision.robustness.perturbations import (
    BLOCK_ROWS,
    GEOMETRIC,
    Geometry,
    _bilinear,
    _shape,
    _spec,
)


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"


class Reason(StrEnum):
    NONE = "none"
    MISSING_SCORE = "missing_score"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    UNSUPPORTED_ALIGNMENT = "unsupported_alignment"


class ThresholdRule(StrEnum):
    GREATER = "greater"
    GREATER_EQUAL = "greater_equal"


class Crossing(StrEnum):
    NONE = "none"
    UP = "up"
    DOWN = "down"


class GridConvention(StrEnum):
    NATIVE = "native_pixel_centers"
    HALF_PIXEL = "explicit_half_pixel_resize"
    UNSUPPORTED = "unsupported"


def _finite(value):
    _require(type(value) is float and math.isfinite(value))


@dataclass(frozen=True, slots=True)
class FrozenThreshold:
    value: float
    rule: ThresholdRule

    def __post_init__(self):
        _finite(self.value)
        _require(type(self.rule) is ThresholdRule)


@dataclass(frozen=True, slots=True)
class ScoreStability:
    status: MeasurementStatus
    reason: Reason
    signed: float | None
    absolute: float | None
    relative: float | None
    crossing: Crossing | None
    classification: Classification = field(default=Classification.SOFTWARE_DIAGNOSTIC, init=False)
    protocol_version: int = field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self):
        _require(type(self.status) is MeasurementStatus and type(self.reason) is Reason)
        if self.status is MeasurementStatus.NOT_APPLICABLE:
            _require(self.reason is Reason.MISSING_SCORE)
            _require(
                all(v is None for v in (self.signed, self.absolute, self.relative, self.crossing))
            )
        else:
            _require(self.reason is Reason.NONE)
            _finite(self.signed)
            _finite(self.absolute)
            _require(self.absolute == abs(self.signed))
            if self.relative is not None:
                _finite(self.relative)
            _require(self.crossing is None or type(self.crossing) is Crossing)


@dataclass(frozen=True, slots=True)
class MapStability:
    status: MeasurementStatus
    reason: Reason
    coverage: float | None
    mae: float | None
    rmse: float | None
    maximum_absolute: float | None
    classification: Classification = field(default=Classification.SOFTWARE_DIAGNOSTIC, init=False)
    protocol_version: int = field(default=PROTOCOL_VERSION, init=False)

    def __post_init__(self):
        _require(type(self.status) is MeasurementStatus and type(self.reason) is Reason)
        if self.coverage is not None:
            _finite(self.coverage)
            _require(0 <= self.coverage <= 1)
        if self.status is MeasurementStatus.NOT_APPLICABLE:
            _require(all(v is None for v in (self.mae, self.rmse, self.maximum_absolute)))
            _require(
                (self.reason is Reason.UNSUPPORTED_ALIGNMENT and self.coverage is None)
                or (
                    self.reason is Reason.INSUFFICIENT_COVERAGE
                    and self.coverage is not None
                    and self.coverage < 0.8
                )
            )
        else:
            _require(
                self.reason is Reason.NONE and self.coverage is not None and self.coverage >= 0.8
            )
            for value in (self.mae, self.rmse, self.maximum_absolute):
                _finite(value)
                _require(value >= 0)
            _require(self.mae <= self.rmse or math.isclose(self.mae, self.rmse, rel_tol=1e-12))
            _require(
                self.rmse <= self.maximum_absolute
                or math.isclose(self.rmse, self.maximum_absolute, rel_tol=1e-12)
            )


def score_stability(
    reference: float | None, candidate: float | None, *, threshold: FrozenThreshold | None = None
) -> ScoreStability:
    """Candidate minus reference; explicit comparator required for a frozen threshold."""
    for value in (reference, candidate):
        if value is not None:
            _finite(value)
    if threshold is not None:
        _require(type(threshold) is FrozenThreshold)
        threshold.__post_init__()
    if reference is None or candidate is None:
        return ScoreStability(
            MeasurementStatus.NOT_APPLICABLE, Reason.MISSING_SCORE, None, None, None, None
        )
    signed = candidate - reference
    _finite(signed)
    relative = signed / abs(reference) if reference != 0 else None
    if relative is not None and not math.isfinite(relative):
        relative = None
    crossing = None
    if threshold is not None:
        before, after = (
            [v > threshold.value for v in (reference, candidate)]
            if threshold.rule is ThresholdRule.GREATER
            else [v >= threshold.value for v in (reference, candidate)]
        )
        crossing = Crossing.NONE if before == after else Crossing.UP if after else Crossing.DOWN
    return ScoreStability(
        MeasurementStatus.MEASURED, Reason.NONE, signed, abs(signed), relative, crossing
    )


def _map(array):
    _require(type(array) is np.ndarray and array.ndim == 2)
    _require(array.dtype in (np.float32, np.float64))
    _shape(*array.shape)
    _require(np.isfinite(array).all())


def _mask(mask, shape):
    if mask is not None:
        _require(type(mask) is np.ndarray and mask.dtype == np.bool_ and mask.shape == shape)


def _grid_scale(grid, shape, image_size):
    if grid is GridConvention.NATIVE:
        return (1.0, 1.0) if shape == (image_size[1], image_size[0]) else None
    if grid is GridConvention.HALF_PIXEL:
        return image_size[0] / shape[1], image_size[1] / shape[0]
    return None


def map_stability(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    perturbation: Perturbation,
    image_size: tuple[int, int],
    reference_grid: GridConvention,
    candidate_grid: GridConvention,
    reference_valid: np.ndarray | None = None,
    candidate_valid: np.ndarray | None = None,
) -> MapStability:
    """Align raw synthetic maps onto the reference grid with conservative support masks.

    Grid declarations are explicit mathematical assumptions, never inferred from a model name.
    Geometry requires a candidate validity mask on its raw-map grid. Missing/unsupported
    correspondence returns not_applicable; malformed or nonfinite data is rejected.
    """
    _map(reference)
    _map(candidate)
    _spec(perturbation)
    _require(type(image_size) is tuple and len(image_size) == 2)
    _shape(image_size[1], image_size[0])
    _require(type(reference_grid) is GridConvention and type(candidate_grid) is GridConvention)
    _mask(reference_valid, reference.shape)
    _mask(candidate_valid, candidate.shape)
    ref_scale = _grid_scale(reference_grid, reference.shape, image_size)
    cand_scale = _grid_scale(candidate_grid, candidate.shape, image_size)
    if (
        ref_scale is None
        or cand_scale is None
        or (perturbation.kind in GEOMETRIC and candidate_valid is None)
    ):
        return MapStability(
            MeasurementStatus.NOT_APPLICABLE, Reason.UNSUPPORTED_ALIGNMENT, None, None, None, None
        )
    geometry = Geometry(
        *image_size, perturbation if perturbation.kind in GEOMETRIC else Perturbation(Kind.IDENTITY)
    )
    h, w = reference.shape
    aligned = np.empty(reference.shape, dtype=np.float64)
    valid = np.empty(reference.shape, dtype=bool)
    same_grid = (
        perturbation.kind not in GEOMETRIC
        and reference.shape == candidate.shape
        and reference_grid is candidate_grid
    )
    for start in range(0, h, BLOCK_ROWS):
        stop = min(h, start + BLOCK_ROWS)
        if same_grid:
            # No coordinate round trip: preserve exact identity on fractional resize grids.
            aligned[start:stop] = candidate[start:stop]
            valid[start:stop] = True
            if reference_valid is not None:
                valid[start:stop] &= reference_valid[start:stop]
            if candidate_valid is not None:
                valid[start:stop] &= candidate_valid[start:stop]
            continue
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(start, stop, dtype=np.float64))
        x, y = (u + 0.5) * ref_scale[0] - 0.5, (v + 0.5) * ref_scale[1] - 0.5
        tx, ty = geometry._xy(x, y)
        cu, cv = (tx + 0.5) / cand_scale[0] - 0.5, (ty + 0.5) / cand_scale[1] - 0.5
        values, supported = _bilinear(candidate, cu, cv, mask=candidate_valid)
        supported &= (x >= 0) & (x <= image_size[0] - 1) & (y >= 0) & (y <= image_size[1] - 1)
        supported &= (tx >= 0) & (tx <= image_size[0] - 1) & (ty >= 0) & (ty <= image_size[1] - 1)
        if reference_valid is not None:
            supported &= reference_valid[start:stop]
        aligned[start:stop], valid[start:stop] = values, supported
    count = int(np.count_nonzero(valid))
    coverage = float(count / valid.size)
    if count * 5 < valid.size * 4:
        return MapStability(
            MeasurementStatus.NOT_APPLICABLE,
            Reason.INSUFFICIENT_COVERAGE,
            coverage,
            None,
            None,
            None,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        difference = np.abs(aligned[valid] - reference[valid].astype(np.float64))
    _require(np.isfinite(difference).all())
    maximum = float(difference.max())
    # Scale before squaring/summing: finite extreme magnitudes need not overflow RMSE.
    if maximum:
        difference /= maximum
        mae = float(difference.mean()) * maximum
        rmse = math.sqrt(float(np.mean(difference * difference))) * maximum
    else:
        mae = rmse = 0.0
    return MapStability(MeasurementStatus.MEASURED, Reason.NONE, coverage, mae, rmse, maximum)
