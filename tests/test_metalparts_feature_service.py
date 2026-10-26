"""Synthetic bank/service safety and method-specific inference; no real data."""

import copy
import io
import json

import pytest
import torch
from PIL import Image
from test_metalparts_memory import Extractor

from factoryvision.data.metalparts import MetalPartsSource
from factoryvision.inference import metalparts_feature as module
from factoryvision.inference.service import ModelUnavailable
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.metalparts_memory import identity, save_memory


@pytest.fixture
def engine(tmp_path, monkeypatch):
    expected = identity(MetalPartsSource(), "a" * 64, "b" * 64, ["fabricated-tile"])
    save_memory(tmp_path, torch.zeros(4, 384), expected)
    (tmp_path / "weights.pth").write_bytes(b"synthetic placeholder; never deserialized")
    assets = {}
    for key, name in [
        ("memory", "memory.pt"),
        ("metadata", "memory.json"),
        ("weights", "weights.pth"),
    ]:
        path = tmp_path / name
        assets[key] = {"path": name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    monkeypatch.setattr(module, "catalog", lambda: {"identity": expected, "assets": assets})
    monkeypatch.setattr(module, "load_extractor", lambda path: Extractor())
    return module.MetalPartsFeatureService(tmp_path)


def payload():
    output = io.BytesIO()
    Image.new("RGB", (301, 279), (60, 90, 120)).save(output, format="PNG")
    return output.getvalue()


def test_safe_service_reload_and_category(engine):
    assert engine.status()["available"]
    first = engine.inspect(payload())
    second = engine.inspect(payload())
    assert first.score == second.score > 0
    assert first.spatial.shape == (279, 301) and not first.spatial.flags.writeable
    assert first.reconstruction is None and first.threshold is None
    assert first.model_id == "metalparts_feature_memory"
    with pytest.raises(ModelUnavailable, match="only"):
        engine.inspect(payload(), "cashew")
    (engine.root / "memory.pt").write_bytes(b"corrupted after cache load")
    assert not engine.status()["available"]
    with pytest.raises(ModelUnavailable):
        engine.inspect(payload())


def test_missing_corrupt_identity_and_load(engine, monkeypatch):
    registry = copy.deepcopy(engine.registry)
    registry["identity"]["category"] = "cashew"
    monkeypatch.setattr(module, "catalog", lambda: registry)
    with pytest.raises(ModelUnavailable, match="catalog"):
        module.MetalPartsFeatureService(engine.root)
    meta = engine.root / "memory.json"
    original = meta.read_bytes()
    meta.write_text(json.dumps({"identity": {}}))
    entry = engine.registry["assets"]["metadata"]
    entry.update(bytes=meta.stat().st_size, sha256=file_sha256(meta))
    assert not engine.status()["available"]
    meta.write_bytes(original)
    entry.update(bytes=meta.stat().st_size, sha256=file_sha256(meta))
    monkeypatch.setattr(module, "load_memory", lambda *a: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ModelUnavailable, match="could not"):
        engine.inspect(payload())
    meta.unlink()
    assert not engine.status()["available"]


def test_invalid_model_map(engine, monkeypatch):
    monkeypatch.setattr(
        module, "native_memory_map", lambda *a: (_ for _ in ()).throw(ValueError("map"))
    )
    with pytest.raises(ModelUnavailable, match="map"):
        engine.inspect(payload())
