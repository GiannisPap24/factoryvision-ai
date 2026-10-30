"""Owned fabricated records; accepted 5D-1 renderer is an independent test oracle."""

import io
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from factoryvision.inference import batch, visualization
from factoryvision.inference.service import InspectionResult


def accepted_panels(result):
    """Frozen visualization.py algorithm at ceb06ba; do not track candidate edits."""
    height, width = result.original.shape[:2]
    scale = min(1, 1024 / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    original = Image.fromarray(result.original).resize(size, Image.Resampling.BILINEAR)
    spatial = np.asarray(Image.fromarray(result.spatial).resize(size, Image.Resampling.BILINEAR))
    ceiling = float(result.spatial.max())
    intensity = np.clip(spatial / ceiling, 0, 1) if ceiling > 0 else np.zeros_like(spatial)
    stops = np.array(
        [[255, 247, 188], [254, 196, 79], [217, 95, 14], [153, 52, 4]], dtype=np.float32
    )
    rgb = np.stack(
        [np.interp(intensity, [0, 1 / 3, 2 / 3, 1], stops[:, c]) for c in range(3)], axis=-1
    )
    heatmap = Image.fromarray(np.round(rgb).astype(np.uint8))
    overlay = Image.blend(original, heatmap, 0.45)
    output = {"original": original, "heatmap": heatmap, "overlay": overlay, "ceiling": ceiling}
    for name in ("reconstruction", "reference"):
        array = getattr(result, name)
        output[name] = (
            None
            if array is None
            else Image.fromarray((np.clip(array, 0, 1) * 255).round().astype(np.uint8)).resize(
                size, Image.Resampling.BILINEAR
            )
        )
    return output


@pytest.fixture(autouse=True)
def no_models_or_files(monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Display must not load models, use network or write files")

    monkeypatch.setattr(torch, "load", denied)
    monkeypatch.setattr(torch.nn.Module, "__init__", denied)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", denied)
    import builtins
    import socket

    original = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            denied()
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(socket, "socket", denied)


def record(height=128, width=128, pattern="ramp", dtype=np.float32, auxiliary=False):
    rng = np.random.default_rng(27)
    original = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    spatial = {
        "zero": lambda: np.zeros((17, 23), dtype=dtype),
        "constant": lambda: np.full((17, 23), 0.4, dtype=dtype),
        "negative": lambda: np.full((17, 23), -1, dtype=dtype),
        "ramp": lambda: np.linspace(-0.2, 2, 391, dtype=dtype).reshape(17, 23),
        "random": lambda: rng.random((height, width)).astype(dtype),
        "tiny": lambda: np.full((17, 23), np.nextafter(dtype(0), dtype(1)), dtype=dtype),
    }[pattern]()
    aux = rng.uniform(-0.1, 1.1, (13, 19, 3)).astype(np.float32) if auxiliary else None
    for array in (original, spatial, aux):
        if array is not None:
            array.setflags(write=False)
    return SimpleNamespace(original=original, spatial=spatial, reconstruction=aux, reference=aux)


def equivalent(result):
    before = {name: None if value is None else value.copy() for name, value in vars(result).items()}
    # Float64 subnormals below Pillow's float32 range already warn in the baseline.
    # Compare their diagnostics explicitly as well as their resulting pixels.
    with warnings.catch_warnings(record=True) as candidate_warnings:
        warnings.simplefilter("always")
        actual = visualization.panels(result)
    with warnings.catch_warnings(record=True) as baseline_warnings:
        warnings.simplefilter("always")
        expected = accepted_panels(result)
    assert {(w.category, str(w.message)) for w in candidate_warnings} == {
        (w.category, str(w.message)) for w in baseline_warnings
    }
    assert actual["ceiling"] == expected["ceiling"] == float(result.spatial.max())
    for name in ("original", "heatmap", "overlay", "reconstruction", "reference"):
        if expected[name] is None:
            assert actual[name] is None
            continue
        assert actual[name].size == expected[name].size
        assert actual[name].mode == expected[name].mode == "RGB"
        assert actual[name].tobytes() == expected[name].tobytes()
        assert actual[name].info == {}
        for side in (128, 384):
            encoded = batch.png(actual[name], side)
            assert encoded == batch.png(expected[name], side)
            with Image.open(io.BytesIO(encoded)) as decoded:
                assert decoded.info == {} and not decoded.getexif()
                assert max(decoded.size) <= side
    for name, value in before.items():
        if value is not None:
            assert np.array_equal(getattr(result, name), value)


@pytest.mark.parametrize("shape", [(128, 128), (512, 512), (61, 99), (1041, 23)])
@pytest.mark.parametrize("pattern", ["zero", "constant", "negative", "ramp", "random", "tiny"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_exact_pixels_previews_geometry_and_immutability(shape, pattern, dtype):
    equivalent(record(*shape, pattern, dtype, auxiliary=True))


def test_palette_and_rounding_boundaries():
    palette = np.array([[255, 247, 188], [254, 196, 79], [217, 95, 14], [153, 52, 4]])
    values = [0, 1 / 3, 2 / 3, 1]
    for segment in range(3):
        for channel in range(3):
            high, low = palette[segment, channel], palette[segment + 1, channel]
            for integer in range(low, high):
                point = np.float32(segment / 3 + (integer + 0.5 - high) / (low - high) / 3)
                values.extend(
                    (np.nextafter(point, np.float32(0)), point, np.nextafter(point, np.float32(1)))
                )
    result = record(64, 64)
    result.spatial = np.resize(np.array(values, dtype=np.float32), (64, 64))
    equivalent(result)
    result = record(1, 4)
    result.spatial = np.array([[0, 1 / 3, 2 / 3, 1]], dtype=np.float32)
    assert np.array_equal(np.asarray(visualization.panels(result)["heatmap"])[0], palette)
    result.spatial = np.zeros((1, 4), dtype=np.float32)
    assert np.all(np.asarray(visualization.panels(result)["heatmap"]) == palette[0])


def test_noncontiguous_inputs_and_missing_auxiliary():
    result = record(64, 96, "random")
    result.original = result.original[:, ::-1]
    result.spatial = result.spatial[:, ::-1]
    equivalent(result)


@pytest.mark.parametrize(
    "model_id",
    [
        "autoencoder",
        "spatial_mean",
        "feature_memory",
        "student_teacher",
        "metalparts_autoencoder",
        "metalparts_feature_memory",
    ],
)
def test_batch_compact_preserves_scores_and_png_content(model_id):
    source = record(512, 512, "random", auxiliary=True)
    category = "metalparts" if model_id.startswith("metalparts_") else "cashew"
    result = InspectionResult(
        model_id,
        "Synthetic",
        "Synthetic score",
        0.25,
        None if category == "metalparts" else 0.5,
        None if category == "metalparts" else False,
        source.spatial,
        source.original,
        source.reconstruction,
        source.reference,
        1.0,
        2.0,
        3.0,
        6.0,
        "a" * 64,
    )
    row = batch.compact(batch.BatchInput("Image 001", "b" * 64, b""), result, category, model_id)
    expected = accepted_panels(source)
    assert row.score == result.score and row.threshold == result.threshold
    assert row.above_threshold == result.above_threshold
    assert row.artifact_sha256 == result.artifact_sha256
    for name in ("original", "heatmap", "overlay", "reconstruction", "reference"):
        assert getattr(row.detail, name) == batch.png(expected[name], 384)


@pytest.mark.parametrize(
    "model_id",
    [
        "autoencoder",
        "spatial_mean",
        "feature_memory",
        "student_teacher",
        "metalparts_autoencoder",
        "metalparts_feature_memory",
    ],
)
@pytest.mark.parametrize("auxiliary", ["none", "reconstruction", "reference"])
def test_single_inspection_presents_exact_accepted_images(monkeypatch, model_id, auxiliary):
    from factoryvision.app import metalparts_ui, ui

    images = []

    class Surface:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def columns(self, n):
            return [self] * n

        def tabs(self, labels):
            return [self] * len(labels)

        def image(self, image, **kwargs):
            images.append(image)

        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    surface = Surface()
    monkeypatch.setattr(ui, "st", surface)
    monkeypatch.setattr(metalparts_ui, "st", surface)
    monkeypatch.setattr(ui.presentation, "st", surface)
    monkeypatch.setattr(
        metalparts_ui, "MetalPartsSource", lambda: SimpleNamespace(revision="synthetic")
    )
    source = record(96, 128, "random", auxiliary=True)
    source.reconstruction = source.reconstruction if auxiliary == "reconstruction" else None
    source.reference = source.reference if auxiliary == "reference" else None
    result = InspectionResult(
        model_id,
        "Synthetic",
        "Synthetic score",
        0.25,
        None,
        None,
        source.spatial,
        source.original,
        source.reconstruction,
        source.reference,
        1.0,
        2.0,
        3.0,
        6.0,
        "a" * 64,
    )
    module = metalparts_ui if model_id.startswith("metalparts_") else ui
    module.show_result(result)
    expected = accepted_panels(source)
    names = ["original", "heatmap", "overlay"]
    if auxiliary == "reconstruction" or (auxiliary == "reference" and module is ui):
        names.append(auxiliary)
    assert len(images) == len(names)
    for image, name in zip(images, names, strict=True):
        assert image.tobytes() == expected[name].tobytes()
        assert image.size == expected[name].size


def test_profile_is_scalar_bounded_and_restores_trace(monkeypatch):
    import sys

    from factoryvision import performance_display_profile as profile

    original = sys.gettrace()
    groups = profile.prepare()
    result = profile.measure(record(32, 32), groups)
    assert set(result) == set(profile.LABELS)
    assert all(type(v) is float and 0 <= v < 10000 for v in result.values())
    assert sys.gettrace() is original

    def failed(record):
        raise ValueError("Controlled profiling failure")

    monkeypatch.setattr(profile, "panels", failed)
    with pytest.raises(ValueError, match="Controlled"):
        profile.measure(record(32, 32), groups)
    assert sys.gettrace() is original


@pytest.mark.parametrize("damage", ["extra", "nan", "negative", "bool"])
def test_profile_wire_rejects_untrusted_metadata(damage):
    from test_performance_baseline import result

    from factoryvision import performance_baseline as b
    from factoryvision.performance_display_profile import LABELS

    data = result()
    profile = dict.fromkeys(LABELS, 1.0)
    data["display_profile_ms"] = profile
    b.validate_worker(data, "result", 128, 1)
    if damage == "extra":
        profile["source_filename"] = "unapproved"
    else:
        profile["geometry"] = {"nan": float("nan"), "negative": -1, "bool": True}[damage]
    with pytest.raises(b.InvalidBenchmark):
        b.validate_worker(data, "result", 128, 1)
