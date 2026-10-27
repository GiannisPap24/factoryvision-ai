"""Analytical pixels and original synthetic geometry only; no models or datasets."""

import builtins
import io
import math
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_robustness_contracts import candidates

from factoryvision.robustness import perturbations as module
from factoryvision.robustness.contracts import (
    Axis,
    Budgets,
    Category,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
    from_json,
    json_bytes,
)
from factoryvision.robustness.perturbations import Geometry, TransformResult, perturb


def apply(rgb, spec, parent_id=ParentID.FIRST):
    parent = Parent(parent_id, rgb.shape[1], rgb.shape[0])
    specifications = (
        (Perturbation(Kind.IDENTITY),)
        if spec.kind is Kind.IDENTITY
        else (
            Perturbation(Kind.IDENTITY),
            spec,
        )
    )
    plan = Plan(Category.CASHEW, (parent,), specifications)
    return perturb(rgb, spec, plan=plan, parent=parent)


@pytest.mark.parametrize(
    "spec", candidates(), ids=lambda p: f"{p.kind}-{p.severity}-{p.axis}-{p.seed}"
)
def test_every_approved_condition_is_repeatable_detached_and_shape_preserving(spec):
    original = np.arange(17 * 19 * 3, dtype=np.uint16).reshape(17, 19, 3).astype(np.uint8)
    before = original.copy()
    first, second = apply(original, spec), apply(original, spec)
    np.testing.assert_array_equal(first.pixels, second.pixels)
    np.testing.assert_array_equal(first.valid, second.valid)
    np.testing.assert_array_equal(original, before)
    assert first.pixels.shape == original.shape and first.pixels.dtype == np.uint8
    assert first.valid.shape == original.shape[:2] and first.valid.dtype == bool
    assert 0 <= first.coverage <= 1
    assert not np.shares_memory(first.pixels, original)
    assert not first.pixels.flags.writeable and not first.valid.flags.writeable
    with pytest.raises(FrozenInstanceError):
        first.pixels = original
    with pytest.raises(ValueError):
        first.pixels[0, 0, 0] = 0


def test_identity_preserves_bytes_even_for_strided_readonly_input():
    rgb = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)[::-1, ::-1]
    rgb.setflags(write=False)
    result = apply(rgb, Perturbation(Kind.IDENTITY))
    assert result.pixels.tobytes() == rgb.tobytes()
    assert result.valid.all() and result.coverage == 1.0


def test_brightness_clipping_and_ties_to_even_oracle():
    rgb = np.array([[[5, 15, 255], [1, 0, 125]]], dtype=np.uint8)
    actual = apply(rgb, Perturbation(Kind.BRIGHTNESS, 0.9)).pixels
    np.testing.assert_array_equal(actual, [[[4, 14, 230], [1, 0, 112]]])
    actual = apply(np.array([[[0, 128, 255]]], dtype=np.uint8), Perturbation(Kind.BRIGHTNESS, 1.35))
    np.testing.assert_array_equal(actual.pixels, [[[0, 173, 255]]])


def test_contrast_uses_fixed_midpoint_not_image_or_channel_means():
    rgb = np.array([[[0, 50, 100], [150, 200, 255]]], dtype=np.uint8)
    np.testing.assert_array_equal(
        apply(rgb, Perturbation(Kind.CONTRAST, 0.6)).pixels,
        [[[51, 81, 111], [141, 171, 204]]],
    )
    np.testing.assert_array_equal(
        apply(rgb, Perturbation(Kind.CONTRAST, 1.4)).pixels,
        [[[0, 19, 89], [159, 229, 255]]],
    )
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    assert (apply(black, Perturbation(Kind.CONTRAST, 0.6)).pixels == 51).all()


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_blur_matches_independent_2d_kernel_and_edge_replication(sigma):
    rgb = np.zeros((5, 7, 3), dtype=np.uint8)
    rgb[0, 0] = (255, 120, 30)
    rgb[2, 3] = (50, 100, 200)
    radius = math.ceil(3 * sigma)
    weights = [math.exp(-0.5 * (i / sigma) ** 2) for i in range(-radius, radius + 1)]
    total = sum(weights) ** 2
    expected = np.zeros_like(rgb)
    for y in range(5):
        for x in range(7):
            pixel = np.zeros(3)
            for dy, wy in zip(range(-radius, radius + 1), weights, strict=True):
                for dx, wx in zip(range(-radius, radius + 1), weights, strict=True):
                    pixel += rgb[min(4, max(0, y + dy)), min(6, max(0, x + dx))] * (wy * wx / total)
            expected[y, x] = np.rint(pixel)
    np.testing.assert_array_equal(apply(rgb, Perturbation(Kind.BLUR, sigma)).pixels, expected)
    constant = np.full((1, 1, 3), 137, dtype=np.uint8)
    np.testing.assert_array_equal(apply(constant, Perturbation(Kind.BLUR, sigma)).pixels, constant)


def test_noise_uses_explicit_case_replicate_derivation_and_preserves_global_rng(monkeypatch):
    rgb = np.full((67, 9, 3), 128, dtype=np.uint8)
    spec = Perturbation(Kind.NOISE, 5 / 255, seed=1729)
    state = np.random.get_state()
    first = apply(rgb, spec, ParentID.SECOND)
    # Protocol oracle: seed 1729, case ordinal 2, replicate ordinal 1.
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([1729, 2, 1])))
    expected = np.rint(
        np.clip(
            (rgb.astype(float) / 255 + rng.standard_normal(rgb.shape) * (5 / 255)) * 255, 0, 255
        )
    ).astype(np.uint8)
    np.testing.assert_array_equal(first.pixels, expected)
    monkeypatch.setattr(module, "BLOCK_ROWS", 3)
    np.testing.assert_array_equal(first.pixels, apply(rgb, spec, ParentID.SECOND).pixels)
    assert not np.array_equal(first.pixels, apply(rgb, spec, ParentID.FIRST).pixels)
    assert not np.array_equal(
        first.pixels, apply(rgb, replace(spec, seed=42), ParentID.SECOND).pixels
    )
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])


def test_jpeg_encoder_is_fixed_in_memory_and_metadata_free(monkeypatch):
    real_save = Image.Image.save
    seen = []

    def checked_save(image, target, **parameters):
        assert isinstance(target, io.BytesIO)
        assert not image.info and not image.getexif()
        assert parameters == dict(
            format="JPEG",
            quality=80,
            subsampling=0,
            progressive=False,
            optimize=False,
            exif=b"",
            icc_profile=None,
        )
        real_save(image, target, **parameters)
        with Image.open(io.BytesIO(target.getvalue())) as decoded:
            assert not decoded.getexif()
            assert not any(k in decoded.info for k in ("exif", "icc_profile", "comment"))
            seen.append(decoded.size)

    monkeypatch.setattr(Image.Image, "save", checked_save)
    result = apply(np.full((7, 9, 3), 80, dtype=np.uint8), Perturbation(Kind.JPEG, 80))
    assert seen == [(9, 7)] and result.valid.all()


@pytest.mark.parametrize("axis", list(Axis))
@pytest.mark.parametrize("fraction", [-0.04, -0.02, -0.01, 0.01, 0.02, 0.04])
def test_translation_sign_axes_and_exact_integer_displacement(axis, fraction):
    rgb = np.arange(100 * 100 * 3).reshape(100, 100, 3).astype(np.uint8)
    result = apply(rgb, Perturbation(Kind.TRANSLATION, fraction, axis))
    shift = int(fraction * 100)
    dim = 1 if axis is Axis.X else 0
    rolled = np.roll(rgb, shift, axis=dim)
    np.testing.assert_array_equal(result.pixels[result.valid], rolled[result.valid])
    assert (result.pixels[~result.valid] == 128).all()
    assert result.coverage == (100 - abs(shift)) / 100


def test_subpixel_translation_uses_bilinear_pixels_and_conservative_mask():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    rgb[:, 5:] = 100
    result = apply(rgb, Perturbation(Kind.TRANSLATION, 0.01, Axis.X))
    assert (result.pixels[:, 5] == 90).all()
    assert (result.pixels[:, 0] == 13).all()  # 0.1 * neutral 128, not wraparound.
    assert not result.valid[:, 0].any() and result.valid[:, 1:].all()


@pytest.mark.parametrize("angle", [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
def test_rotation_forward_inverse_and_asymmetric_linear_ramp(angle):
    spec = Perturbation(Kind.ROTATION, angle)
    geometry = Geometry(31, 21, spec)
    points = np.array([[15.0, 10.0], [20.0, 10.0], [8.0, 4.0]])
    mapped = geometry.points(points)
    np.testing.assert_allclose(geometry.points(mapped, inverse=True), points, atol=1e-13)
    np.testing.assert_allclose(mapped[0], points[0], atol=1e-13)
    assert (mapped[1, 1] < 10) == (angle > 0)
    y, x = np.indices((21, 31))
    rgb = np.stack((x + 2 * y, 3 * x + y, x + y), axis=-1).astype(np.uint8)
    result = apply(rgb, spec)
    output_points = np.column_stack((x.ravel(), y.ravel())).astype(float)
    source_points = geometry.points(output_points, inverse=True)
    sx, sy = source_points.T
    expected = np.rint(np.stack((sx + 2 * sy, 3 * sx + sy, sx + sy), axis=-1)).reshape(rgb.shape)
    np.testing.assert_array_equal(result.pixels[result.valid], expected[result.valid])
    assert 0.8 <= result.coverage < 1


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=float),
        np.full((2, 2, 3), np.nan),
        np.full((2, 2, 3), -1, dtype=np.int16),
        np.zeros((2, 2, 3), dtype=object),
        [[0, 0, 0]],
    ],
    ids=["2d", "rgba", "empty", "float", "nan", "negative", "object", "list"],
)
def test_invalid_rgb_rejected(invalid):
    parent = Parent(ParentID.FIRST, 2, 2)
    plan = Plan(Category.CASHEW, (parent,), (Perturbation(Kind.IDENTITY),))
    with pytest.raises(RobustnessInvalid):
        perturb(invalid, Perturbation(Kind.IDENTITY), plan=plan, parent=parent)


def test_admission_rejects_unplanned_conditions_dimensions_old_protocol_and_oversized_views():
    parent = Parent(ParentID.FIRST, 2, 2)
    plan = Plan(Category.CASHEW, (parent,), (Perturbation(Kind.IDENTITY),))
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(RobustnessInvalid):
        perturb(rgb, Perturbation(Kind.BLUR, 0.5), plan=plan, parent=parent)
    with pytest.raises(RobustnessInvalid):
        perturb(rgb[:1], Perturbation(Kind.IDENTITY), plan=plan, parent=parent)
    with pytest.raises(RobustnessInvalid):
        perturb(
            rgb,
            Perturbation(Kind.IDENTITY),
            plan=plan,
            parent=replace(parent, parent_id=ParentID.SECOND),
        )
    with pytest.raises(RobustnessInvalid):
        from_json(json_bytes(plan).replace(b'"protocol_version":2', b'"protocol_version":1'), Plan)
    # Strided views test limits before any large allocation.
    for shape in ((4097, 1, 3), (2048, 2049, 3)):
        too_large = np.broadcast_to(np.uint8(0), shape)
        with pytest.raises(RobustnessInvalid):
            TransformResult(too_large, np.broadcast_to(True, shape[:2]))
    assert plan.budgets == Budgets()


def test_original_demo_geometry_can_be_reused_without_new_generator():
    from factoryvision.demo import build_demo

    for payload in build_demo().images:
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.array(image)
        original = rgb.copy()
        assert apply(rgb, Perturbation(Kind.ROTATION, 1.0)).coverage >= 0.8
        np.testing.assert_array_equal(rgb, original)


def test_transforms_do_not_access_files_or_network(monkeypatch):
    Image.init()  # Resolve installed codec imports before forbidding runtime I/O.
    rgb = np.full((7, 11, 3), 100, dtype=np.uint8)

    def forbidden(*args, **kwargs):
        pytest.fail("Synthetic transforms must remain memory-only")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "rglob", forbidden)
        for spec in candidates():
            assert apply(rgb, spec).pixels.shape == rgb.shape
    # Network is blocked by the existing autouse fixture.


def test_fresh_process_replay_ignores_python_hash_seed_and_imports_no_model_services():
    script = """
import hashlib, sys
import numpy as np
from factoryvision.robustness.contracts import Category, Kind, Parent, ParentID, Perturbation, Plan
from factoryvision.robustness.perturbations import perturb
from factoryvision.robustness.assessment import score_stability
p = Parent(ParentID.THIRD, 7, 9)
s = Perturbation(Kind.NOISE, 10 / 255, seed=2026)
plan = Plan(Category.CASHEW, (p,), (Perturbation(Kind.IDENTITY), s))
result = perturb(np.full((9, 7, 3), 128, dtype=np.uint8), s, plan=plan, parent=p)
assert score_stability(0.0, 1.0).absolute == 1.0
assert not any(m.split('.')[0] in ('torch', 'torchvision', 'streamlit') for m in sys.modules)
assert not any(m in sys.modules for m in ('factoryvision.inference.service',
    'factoryvision.inference.metalparts', 'factoryvision.inference.model_registry'))
print(hashlib.sha256(result.pixels.tobytes()).hexdigest())
"""
    outputs = []
    for seed in ("17", "91"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        )
        assert result.stderr == ""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] and len(outputs[0].strip()) == 64


def test_jpeg_failure_does_not_disclose_exception_details(monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("fabricated-private-detail")

    monkeypatch.setattr(Image.Image, "save", broken)
    with pytest.raises(RobustnessInvalid) as caught:
        apply(np.zeros((2, 2, 3), dtype=np.uint8), Perturbation(Kind.JPEG, 95))
    assert str(caught.value) == str(RobustnessInvalid())
    assert caught.value.__suppress_context__


def test_geometry_points_and_result_masks_reject_malformed_values():
    geometry = Geometry(2, 2, Perturbation(Kind.IDENTITY))
    for points in (
        np.zeros((2, 3)),
        np.zeros((0, 2)),
        np.ones((2, 2), dtype=int),
        np.full((2, 2), np.nan),
        np.full((2, 2), 8193.0),
        [[0.0, 1.0]],
    ):
        with pytest.raises(RobustnessInvalid):
            geometry.points(points)
    with pytest.raises(RobustnessInvalid):
        geometry.points(np.zeros((1, 2)), inverse=1)
    for mask in (np.zeros((2, 2)), np.zeros((1, 2), dtype=bool)):
        with pytest.raises(RobustnessInvalid):
            TransformResult(np.zeros((2, 2, 3), dtype=np.uint8), mask)
