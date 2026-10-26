"""Synthetic-only inference contracts; no downloaded images or model weights."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, PngImagePlugin

from factoryvision.inference import service as mod
from factoryvision.inference import uploads
from factoryvision.inference.service import (
    MODEL_IDS,
    InspectionService,
    ModelUnavailable,
    preprocess,
    threshold_flag,
    verified_asset,
)
from factoryvision.inference.visualization import panels


def encoded(mode="RGB", size=(80, 40), color=100, fmt="PNG", **kwargs):
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


class Tiny(torch.nn.Module):
    def __init__(self, kind="autoencoder"):
        super().__init__()
        self.kind = kind

    def forward(self, x):
        assert not self.training and not torch.is_grad_enabled()
        if self.kind in ("autoencoder", "spatial_mean"):
            return x * 0.5
        return self.score(x)

    def score(self, x):
        assert not self.training and not torch.is_grad_enabled()
        return torch.ones(1), torch.ones(1, 128, 128)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    engine = InspectionService(tmp_path)
    monkeypatch.setattr(engine, "_load", lambda key: (Tiny(key).eval(), engine.specs[key]))
    return engine


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P"])
def test_decode_rgb_and_strip_metadata(mode):
    meta = PngImagePlugin.PngInfo()
    meta.add_text("private", "DO NOT RETAIN")
    color = (100, 100, 100, 255) if mode == "RGBA" else 100
    result = uploads.decode_upload(encoded(mode, color=color, pnginfo=meta))
    assert result.mode == "RGB" and result.size == (80, 40) and result.info == {}
    tensor = preprocess(result)
    assert tensor.shape == (1, 3, 128, 128) and tensor.dtype == torch.float32
    assert tensor.min() >= 0 and tensor.max() <= 1


def test_jpeg_normalization():
    tensor = preprocess(uploads.decode_upload(encoded(color=(255, 0, 0), fmt="JPEG")))
    assert tensor[0, 0].mean() > 0.99 and tensor[0, 1:].max() < 0.01


def test_upload_never_invokes_unsupported_format_parser(monkeypatch):
    Image.init()

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported parser must not receive uploaded bytes")

    monkeypatch.setitem(Image.OPEN, "XBM", (forbidden, lambda prefix: True))
    with pytest.raises(uploads.InputRejected):
        uploads.decode_upload(b"#define fabricated_width 1\n#define fabricated_height 1\n")
    assert uploads.decode_upload(encoded()).size == (80, 40)


@pytest.mark.parametrize(
    "payload",
    [b"", b"broken", None, b"x" * (8 * 1024**2 + 1)],
    ids=["empty", "corrupt", "wrong-type", "oversized"],
)
def test_invalid_bytes(payload):
    with pytest.raises(uploads.InputRejected):
        uploads.decode_upload(payload)


@pytest.mark.parametrize(
    "options",
    [
        {"fmt": "GIF"},
        {"size": (4097, 1)},
        {"size": (4001, 4000)},
        {"mode": "RGBA", "color": (1, 2, 3, 0)},
    ],
)
def test_reject_unsupported_geometry_alpha(options):
    with pytest.raises(uploads.InputRejected):
        uploads.decode_upload(encoded(**options))


def test_corrupt_truncated_orientation_and_animation():
    with pytest.raises(uploads.InputRejected):
        uploads.decode_upload(encoded()[:-20])
    exif = Image.Exif()
    exif[274] = 6
    with pytest.raises(uploads.InputRejected, match="upright"):
        uploads.decode_upload(encoded(exif=exif))
    animated = encoded(save_all=True, append_images=[Image.new("RGB", (80, 40), "red")])
    with pytest.raises(uploads.InputRejected, match="single image"):
        uploads.decode_upload(animated)


@pytest.mark.parametrize("limit", [1000, 2000])
def test_bombs(monkeypatch, limit):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", limit)
    with pytest.raises(uploads.InputRejected, match="unsafe"):
        uploads.decode_upload(encoded())


def test_official_fingerprints_reject_without_real_test_access(monkeypatch):
    payload = encoded()
    monkeypatch.setattr(
        uploads, "protected_hashes", lambda: ({hashlib.sha256(payload).hexdigest()}, set())
    )
    with pytest.raises(uploads.InputRejected, match="official test"):
        uploads.decode_upload(payload)
    image = Image.open(io.BytesIO(payload))
    sha = hashlib.sha256(b"RGB:80:40:" + image.tobytes()).hexdigest()
    monkeypatch.setattr(uploads, "protected_hashes", lambda: (set(), {sha}))
    with pytest.raises(uploads.InputRejected, match="decoded pixels"):
        uploads.decode_upload(payload)


@pytest.mark.parametrize(
    "score,threshold,expected", [(1, 1, False), (1.1, 1, True), (0, 1, False), (1, None, None)]
)
def test_threshold(score, threshold, expected):
    assert threshold_flag(score, threshold) is expected


@pytest.mark.parametrize(
    "score,threshold", [(float("nan"), 1), (-1, 1), (1, -1), (1, float("inf"))]
)
def test_nonfinite_threshold_score(score, threshold):
    with pytest.raises(ModelUnavailable):
        threshold_flag(score, threshold)


@pytest.mark.parametrize("key", MODEL_IDS)
def test_results_and_alignment(engine, key, tmp_path):
    before = list(tmp_path.rglob("*"))
    result = engine.inspect(encoded(), key)
    second = engine.inspect(encoded(), key)
    assert result.score == second.score
    assert np.array_equal(result.spatial, second.spatial)
    assert result.spatial.shape == (128, 128)
    assert not result.spatial.flags.writeable and not result.original.flags.writeable
    assert (result.reconstruction is not None) == (key == "autoencoder")
    assert (result.reference is not None) == (key == "spatial_mean")
    view = panels(result)
    assert view["original"].size == view["overlay"].size == view["heatmap"].size == (80, 40)
    assert result.total_ms >= result.inference_ms >= 0
    zero = panels(replace(result, spatial=np.zeros((128, 128), dtype=np.float32)))
    assert zero["ceiling"] == 0 and zero["heatmap"].getpixel((0, 0)) == (255, 247, 188)
    assert list(tmp_path.rglob("*")) == before


def test_preview_bounds(engine):
    result = engine.inspect(encoded(size=(2048, 1024)), "autoencoder")
    assert panels(result)["original"].size == (1024, 512)


def test_multiple_selection(engine):
    keys = ["student_teacher", "autoencoder"]
    assert [r.model_id for r in engine.inspect_many(encoded(), keys)] == keys
    for invalid in ([], ["oops"], ["autoencoder", "autoencoder"]):
        with pytest.raises(ModelUnavailable):
            engine.inspect_many(encoded(), invalid)


@pytest.mark.parametrize(
    "output",
    [
        (torch.zeros(2), torch.zeros(1, 128, 128)),
        (torch.zeros(1), torch.zeros(1, 16, 16)),
        (torch.zeros(1), torch.full((1, 128, 128), float("nan"))),
        (torch.zeros(1), -torch.ones(1, 128, 128)),
    ],
)
def test_bad_outputs(engine, monkeypatch, output):
    monkeypatch.setattr(engine, "_load", lambda key: (lambda x: output, engine.specs[key]))
    with pytest.raises(ModelUnavailable, match="invalid output"):
        engine.inspect(encoded(), "feature_memory")


def test_hash_path_guards(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"synthetic")
    entry = {"path": p.name, "bytes": 9, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    assert verified_asset(tmp_path, entry) == p
    for change in ({"path": "../outside"}, {"bytes": 8}, {"sha256": "0" * 64}, {"path": "absent"}):
        with pytest.raises(ModelUnavailable):
            verified_asset(tmp_path, {**entry, **change})


def test_link_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "is_symlink", lambda self: True)
    with pytest.raises(ModelUnavailable, match="Linked"):
        verified_asset(tmp_path, {"path": "x"})


def test_missing_models_and_catalog(tmp_path, monkeypatch):
    engine = InspectionService(tmp_path)
    assert all(not row["available"] for row in engine.statuses())
    with pytest.raises(ModelUnavailable):
        engine._verify("bad")
    registry = mod.catalog()
    registry["category"] = "pcb1"
    monkeypatch.setattr(mod, "catalog", lambda: registry)
    with pytest.raises(ModelUnavailable, match="catalog"):
        InspectionService(tmp_path)


@pytest.fixture
def verified_engine(tmp_path, monkeypatch, manifest):
    engine = InspectionService(tmp_path)
    engine.registry["fit_ids"] = sorted(
        s.sample_id for s in manifest.samples if s.partition.value == "train"
    )
    engine.registry["manifest_sha256"] = mod.manifest_sha256(manifest)
    monkeypatch.setattr(mod, "load_manifest", lambda path: manifest)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "threshold": engine.specs["autoencoder"]["threshold"],
                "comparison": "score > threshold",
                "interpolation": "linear",
                "quantile": 0.95,
                "count": 68,
            }
        )
    )
    monkeypatch.setattr(mod, "verified_asset", lambda root, entry: calibration)
    return engine, calibration


def test_compatibility_validation(verified_engine, monkeypatch, manifest):
    engine, path = verified_engine
    # Synthetic manifest category is deliberately different from the supported category.
    fake = SimpleNamespace(
        samples=[
            SimpleNamespace(category="cashew", sample_id=s.sample_id, partition=s.partition)
            for s in manifest.samples
        ]
    )
    monkeypatch.setattr(mod, "load_manifest", lambda p: fake)
    monkeypatch.setattr(mod, "manifest_sha256", lambda m: engine.registry["manifest_sha256"])
    assert engine._verify("autoencoder")[0]["id"] == "autoencoder"
    assert engine.statuses()[0]["available"]
    engine.registry["fit_ids"] = []
    with pytest.raises(ModelUnavailable, match="membership"):
        engine._verify("autoencoder")
    engine.registry["fit_ids"] = sorted(
        s.sample_id for s in manifest.samples if s.partition.value == "train"
    )
    path.write_text("{}")
    with pytest.raises(ModelUnavailable, match="unavailable"):
        engine._verify("autoencoder")


@pytest.mark.parametrize("key", MODEL_IDS)
def test_safe_adapter_dispatch_and_cache(tmp_path, monkeypatch, key):
    engine = InspectionService(tmp_path)
    spec = engine.specs[key]
    weights = tmp_path / "tiny.pt"
    torch.save(torch.zeros(3, 128, 128), weights)
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "manifest_sha256": engine.registry["manifest_sha256"],
                "baseline_sha256": spec["assets"]["weights"]["sha256"],
            }
        )
    )
    paths = {"weights": weights, "metadata": meta, "teacher": weights}
    monkeypatch.setattr(engine, "_verify", lambda k: (spec, paths))
    monkeypatch.setattr(mod, "load_checkpoint", lambda *a: (Tiny(), {}))
    monkeypatch.setattr(mod, "load_memory", lambda *a: (torch.zeros(1), {}))
    monkeypatch.setattr(mod, "load_student", lambda *a: (Tiny(), {}))
    monkeypatch.setattr(mod, "load_extractor", lambda *a: Tiny())
    monkeypatch.setattr(mod, "FeatureMemory", lambda *a: Tiny("feature_memory"))
    monkeypatch.setattr(mod, "SpatialTeacher", lambda *a: Tiny())
    monkeypatch.setattr(mod, "StudentTeacher", lambda *a: Tiny("student_teacher"))
    model, _ = engine._load(key)
    assert not model.training and engine._load(key)[0] is model
    assert engine.inspect(encoded(), key).spatial.shape == (128, 128)


def test_adapter_load_failure(tmp_path, monkeypatch):
    engine = InspectionService(tmp_path)
    monkeypatch.setattr(
        engine, "_verify", lambda k: (engine.specs[k], {"weights": tmp_path / "missing"})
    )
    with pytest.raises(ModelUnavailable, match="safely loaded"):
        engine._load("autoencoder")
