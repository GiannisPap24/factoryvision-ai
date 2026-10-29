"""Analytical scalar/raw-map fixtures; measured diagnostics are never research evidence."""

import builtins
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from factoryvision.robustness.assessment import (
    Crossing,
    FrozenThreshold,
    GridConvention,
    MapStability,
    MeasurementStatus,
    Reason,
    ScoreStability,
    ThresholdRule,
    map_stability,
    score_stability,
)
from factoryvision.robustness.contracts import (
    Axis,
    Kind,
    Perturbation,
    RobustnessInvalid,
    json_bytes,
)
from factoryvision.robustness.perturbations import Geometry


def compare(reference, candidate, **kwargs):
    values = dict(
        perturbation=Perturbation(Kind.IDENTITY),
        image_size=(reference.shape[1], reference.shape[0]),
        reference_grid=GridConvention.NATIVE,
        candidate_grid=GridConvention.NATIVE,
    )
    return map_stability(reference, candidate, **(values | kwargs))


def test_exact_score_differences_relative_and_frozen_comparator_boundaries():
    result = score_stability(2.0, 5.0)
    assert (result.signed, result.absolute, result.relative, result.crossing) == (
        3.0,
        3.0,
        1.5,
        None,
    )
    assert result.status is MeasurementStatus.MEASURED
    assert score_stability(5.0, 2.0).signed == -3.0
    assert score_stability(-2.0, -1.0).relative == 0.5
    assert score_stability(0.0, 0.0).absolute == 0.0
    assert score_stability(0.0, 1.0).relative is None
    assert score_stability(5e-324, 1.0).relative is None
    gt = FrozenThreshold(1.0, ThresholdRule.GREATER)
    ge = FrozenThreshold(1.0, ThresholdRule.GREATER_EQUAL)
    assert score_stability(0.0, 1.0, threshold=gt).crossing is Crossing.NONE
    assert score_stability(0.0, 1.0, threshold=ge).crossing is Crossing.UP
    assert score_stability(2.0, 1.0, threshold=gt).crossing is Crossing.DOWN
    assert score_stability(2.0, 1.0, threshold=ge).crossing is Crossing.NONE


@pytest.mark.parametrize("scores", [(None, None), (None, 1.0), (1.0, None)])
def test_missing_scores_are_not_applicable_not_zero(scores):
    result = score_stability(*scores)
    assert (
        result.status is MeasurementStatus.NOT_APPLICABLE and result.reason is Reason.MISSING_SCORE
    )
    assert result.signed is result.absolute is result.relative is result.crossing is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, 1, "private"])
def test_invalid_score_or_threshold_has_fixed_error(value):
    for action in (
        lambda: score_stability(value, 1.0),
        lambda: score_stability(None, value),
        lambda: FrozenThreshold(value, ThresholdRule.GREATER),
    ):
        with pytest.raises(RobustnessInvalid) as caught:
            action()
        assert str(caught.value) == str(RobustnessInvalid())
    with pytest.raises(RobustnessInvalid):
        FrozenThreshold(1.0, "greater")


def test_analytical_map_metrics_identical_constant_zero_and_extreme_magnitudes():
    ref = np.zeros((2, 2), dtype=np.float64)
    candidate = np.array([[0.0, 1.0], [2.0, 3.0]])
    result = compare(ref, candidate)
    assert result.mae == 1.5 and result.rmse == pytest.approx(np.sqrt(3.5))
    assert result.maximum_absolute == 3.0 and result.coverage == 1.0
    for array in (ref, candidate, np.full((2, 2), 7.0)):
        identical = compare(array, array)
        assert identical.mae == identical.rmse == identical.maximum_absolute == 0.0
    large = compare(ref, np.full((2, 2), 1e200))
    assert large.mae == large.rmse == large.maximum_absolute == 1e200
    with pytest.raises(RobustnessInvalid):
        compare(np.full((2, 2), -1e308), np.full((2, 2), 1e308))
    with pytest.raises(RobustnessInvalid):
        score_stability(-1e308, 1e308)


@pytest.mark.parametrize("count", [0, 79, 80, 81, 100])
def test_exact_eighty_percent_coverage_gate(count):
    ref = np.zeros((10, 10))
    mask = (np.arange(100) < count).reshape(10, 10)
    result = compare(ref, np.ones((10, 10)), candidate_valid=mask)
    assert result.coverage == count / 100
    if count < 80:
        assert result.status is MeasurementStatus.NOT_APPLICABLE
        assert result.reason is Reason.INSUFFICIENT_COVERAGE
        assert result.mae is result.rmse is result.maximum_absolute is None
    else:
        assert result.status is MeasurementStatus.MEASURED
        assert result.mae == result.rmse == result.maximum_absolute == 1.0


@pytest.mark.parametrize(
    "spec",
    [
        Perturbation(Kind.ROTATION, -3.0),
        Perturbation(Kind.ROTATION, 3.0),
        Perturbation(Kind.TRANSLATION, 0.02, Axis.X),
        Perturbation(Kind.TRANSLATION, -0.01, Axis.Y),
    ],
)
def test_geometry_alignment_cancels_analytical_ramp_instead_of_comparing_unaligned_maps(spec):
    h, w = 80, 100
    y, x = np.indices((h, w), dtype=float)
    reference = x + 2 * y
    geometry = Geometry(w, h, spec)
    points = np.column_stack((x.ravel(), y.ravel()))
    inverse = geometry.points(points, inverse=True).reshape(h, w, 2)
    sx, sy = inverse[..., 0], inverse[..., 1]
    candidate = sx + 2 * sy
    mask = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    result = compare(reference, candidate, perturbation=spec, candidate_valid=mask)
    assert result.status is MeasurementStatus.MEASURED and result.coverage >= 0.8
    assert result.maximum_absolute < 1e-12
    assert np.abs(candidate - reference).mean() > 0.1


def test_explicit_coarse_half_pixel_grids_and_geometry_do_not_assume_native_resolution():
    # Both maps are on a 40x50 grid, with centers at 2*u+0.5 in a 100x80 image.
    h, w = 40, 50
    y, x = np.indices((h, w), dtype=float)
    reference = 2 * x + 0.5 + 2 * (2 * y + 0.5)
    spec = Perturbation(Kind.TRANSLATION, 0.02, Axis.X)
    candidate = reference - 2.0
    mask = np.ones((h, w), dtype=bool)
    mask[:, 0] = False
    result = compare(
        reference,
        candidate,
        image_size=(100, 80),
        perturbation=spec,
        reference_grid=GridConvention.HALF_PIXEL,
        candidate_grid=GridConvention.HALF_PIXEL,
        candidate_valid=mask,
    )
    assert result.status is MeasurementStatus.MEASURED
    assert result.mae == result.rmse == result.maximum_absolute == 0.0
    unsupported = compare(reference, candidate, image_size=(100, 80), candidate_valid=mask)
    assert unsupported.reason is Reason.UNSUPPORTED_ALIGNMENT


def test_masks_require_all_contributing_samples_and_are_not_nearest_neighbor():
    reference = np.zeros((10, 10))
    mask = np.ones((10, 10), dtype=bool)
    mask[:, 4] = False
    result = compare(
        reference,
        reference,
        perturbation=Perturbation(Kind.TRANSLATION, 0.01, Axis.X),
        candidate_valid=mask,
    )
    # Interpolation touches column 4 for source columns 3 and 4; last column is outside.
    assert result.coverage == 0.7 and result.reason is Reason.INSUFFICIENT_COVERAGE


def test_unsupported_alignment_and_missing_geometry_mask_are_explicit():
    a = np.zeros((10, 10))
    for changes in (
        {"candidate_grid": GridConvention.UNSUPPORTED},
        {"perturbation": Perturbation(Kind.ROTATION, 1.0)},
    ):
        result = compare(a, a, **changes)
        assert result.reason is Reason.UNSUPPORTED_ALIGNMENT and result.coverage is None
        assert result.mae is None
    result = compare(a, a, reference_valid=np.zeros((10, 10), dtype=bool))
    assert result.coverage == 0.0 and result.reason is Reason.INSUFFICIENT_COVERAGE


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((2, 2, 3)),
        np.zeros((0, 2)),
        np.zeros((2, 2), dtype=np.uint8),
        np.full((2, 2), np.nan),
        np.full((2, 2), np.inf),
        np.zeros((2, 2), dtype=object),
    ],
    ids=["heatmap", "empty", "uint8", "nan", "inf", "object"],
)
def test_malformed_or_display_map_rejected(array):
    with pytest.raises(RobustnessInvalid):
        compare(np.zeros((2, 2)), array)


def test_invalid_masks_geometry_and_grid_declarations_are_rejected():
    a = np.zeros((2, 2))
    for changes in (
        {"candidate_valid": np.ones((2, 2))},
        {"reference_valid": np.ones((1, 2), dtype=bool)},
        {"candidate_valid": np.zeros((0, 2), dtype=bool)},
        {"image_size": (True, 2)},
        {"image_size": [2, 2]},
        {"candidate_grid": "native_pixel_centers"},
        {"perturbation": "flip"},
    ):
        with pytest.raises(RobustnessInvalid):
            compare(a, a, **changes)
    with pytest.raises(RobustnessInvalid):
        Geometry(2, 2, Perturbation(Kind.BLUR, 0.5))


def test_typed_results_are_immutable_diagnostic_only_and_not_public_exports():
    score = score_stability(1.0, 2.0)
    metrics = compare(np.zeros((2, 2)), np.ones((2, 2)))
    for result in (score, metrics):
        assert result.classification.value == "software_diagnostic" and result.protocol_version == 2
        with pytest.raises(FrozenInstanceError):
            result.status = MeasurementStatus.NOT_APPLICABLE
        with pytest.raises(RobustnessInvalid):
            json_bytes(result)
    for action in (
        lambda: replace(score, absolute=99.0),
        lambda: replace(score, status="HISTORICAL"),
        lambda: replace(metrics, coverage=0.79),
        lambda: replace(metrics, rmse=float("nan")),
        lambda: replace(metrics, mae=2.0),
        lambda: MapStability(MeasurementStatus.NOT_APPLICABLE, Reason.NONE, None, None, None, None),
        lambda: ScoreStability(
            MeasurementStatus.NOT_APPLICABLE, Reason.MISSING_SCORE, 0.0, None, None, None
        ),
    ):
        with pytest.raises(RobustnessInvalid):
            action()


def test_metrics_are_memory_only_and_do_not_mutate_arrays(monkeypatch):
    reference, candidate = np.zeros((10, 10)), np.ones((10, 10))
    before = candidate.copy()
    mask = np.ones((10, 10), dtype=bool)
    reference.setflags(write=False)
    candidate.setflags(write=False)
    mask.setflags(write=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Numerical assessment must not access files")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        assert compare(reference, candidate, candidate_valid=mask).mae == 1.0
        assert score_stability(0.0, 1.0).absolute == 1.0
    np.testing.assert_array_equal(candidate, before)
    assert mask.all()


def test_different_declared_grids_compare_analytical_values_without_guessing_resize():
    y, x = np.indices((4, 4), dtype=float)
    candidate = x + 2 * y
    ry, rx = np.indices((2, 2), dtype=float)
    reference = (2 * rx + 0.5) + 2 * (2 * ry + 0.5)
    result = compare(
        reference, candidate, image_size=(4, 4), reference_grid=GridConvention.HALF_PIXEL
    )
    assert result.coverage == 1.0 and result.maximum_absolute == 0.0


def test_float32_maps_and_single_pixel_grid_have_defined_behavior():
    ref = np.zeros((1, 1), dtype=np.float32)
    candidate = np.full((1, 1), 3, dtype=np.float32)
    result = compare(ref, candidate)
    assert result.coverage == 1.0 and result.mae == result.rmse == result.maximum_absolute == 3.0


@pytest.mark.parametrize("image_size", [(101, 103), (7, 11)])
def test_same_fractional_grid_has_exact_identity_without_resampling(image_size):
    reference = np.arange(29 * 31, dtype=float).reshape(29, 31)
    result = compare(
        reference,
        reference,
        image_size=image_size,
        reference_grid=GridConvention.HALF_PIXEL,
        candidate_grid=GridConvention.HALF_PIXEL,
    )
    assert result.coverage == 1.0 and result.mae == result.rmse == result.maximum_absolute == 0.0
    result = compare(
        reference,
        reference + 1.25,
        image_size=image_size,
        reference_grid=GridConvention.HALF_PIXEL,
        candidate_grid=GridConvention.HALF_PIXEL,
        perturbation=Perturbation(Kind.BRIGHTNESS, 1.1),
    )
    assert result.coverage == 1.0 and result.mae == result.rmse == result.maximum_absolute == 1.25
