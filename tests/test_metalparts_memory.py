"""Fabricated normal parents and features only; no downloaded weights or source payloads."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.models import resnet18

from factoryvision.data.metalparts import SYNTHETIC_REVISION, MetalPartsSample, MetalPartsSource
from factoryvision.data.metalparts_development import DevelopmentSample
from factoryvision.domain.contracts import Label, Partition
from factoryvision.evaluation.metalparts import PixelRanking
from factoryvision.evaluation.metalparts_comparison import DistancePixelRanking
from factoryvision.inference.uploads import InputRejected
from factoryvision.models import metalparts_memory as module
from factoryvision.models.feature_memory import FrozenResNet18, nearest_distances
from factoryvision.models.metalparts_memory import (
    FittingTiles,
    ResourceBudget,
    descriptor_indices,
    fit_native_memory,
    native_memory_map,
)
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.metalparts_memory import identity, load_memory, save_memory


@pytest.fixture
def tiles(tmp_path):
    source = MetalPartsSource(SYNTHETIC_REVISION)
    sample = MetalPartsSample(
        source, "train", "normal/fabricated.png", Label.NORMAL, None, 301, 279
    )
    path = tmp_path / sample.image_path
    path.parent.mkdir(parents=True)
    Image.fromarray(np.random.default_rng(4).integers(0, 256, (279, 301, 3), dtype=np.uint8)).save(
        path
    )
    record = DevelopmentSample(sample, file_sha256(path), "a" * 64)
    return FittingTiles(tmp_path, (record,))


class Extractor(torch.nn.Module):
    def forward(self, image):
        return torch.nn.functional.avg_pool2d(image.mean(1, keepdim=True), 8).repeat(1, 384, 1, 1)


def test_deterministic_fit_traceability(tiles):
    model, trace = fit_native_memory(Extractor(), tiles, lambda: None)
    replay, other = fit_native_memory(Extractor(), tiles, lambda: None)
    assert torch.equal(model.bank, replay.bank) and trace == other
    assert len(model.bank) == len(tiles.ids) * 4 == 16
    assert [r["tile_id"] for r in trace] == tiles.ids
    assert {tuple(r["xy"]) for r in trace} == {(0, 0), (45, 0), (0, 23), (45, 23)}
    assert all(
        torch.equal(descriptor_indices(r["tile_id"]), torch.tensor(r["descriptor_indices"]))
        for r in trace
    )
    assert all(not p.requires_grad for p in model.parameters())
    assert nearest_distances(model.bank[:1], model.bank).max() < 1e-3
    assert nearest_distances(model.bank[:1] + 3, model.bank).min() > 1


def test_frozen_real_architecture(tiles):
    torch.manual_seed(3)
    extractor = FrozenResNet18(resnet18(weights=None))
    before = parameter_sha256(extractor)
    extractor.train()
    model, _ = fit_native_memory(extractor, tiles, lambda: None)
    assert not extractor.training and not any(p.requires_grad for p in extractor.parameters())
    assert parameter_sha256(extractor) == before
    assert torch.isfinite(model.bank).all()


def test_fit_isolation_and_budget(tiles, tmp_path):
    sample = tiles.dataset.samples[0]
    for records in [(), (replace(sample, partition=Partition.VALIDATION),), (object(),)]:
        with pytest.raises(ValueError):
            FittingTiles(tmp_path, records)
    with pytest.raises(ValueError):
        fit_native_memory(Extractor(), [], lambda: None)
    with pytest.raises(ValueError):
        descriptor_indices("")
    with pytest.raises(TimeoutError):
        fit_native_memory(Extractor(), tiles, ResourceBudget(-1, lambda: 0))
    with pytest.raises(TimeoutError):
        ResourceBudget(10, lambda: 101, 100)()
    unfrozen = torch.nn.Conv2d(3, 384, 1)
    with pytest.raises(ValueError, match="frozen"):
        fit_native_memory(unfrozen, tiles, lambda: None)
    with pytest.raises(ValueError, match="geometry"):
        fit_native_memory(torch.nn.Identity(), tiles, lambda: None)


class TileMap(torch.nn.Module):
    def forward(self, image):
        # Constant map reproduces itself across every overlap/edge.
        return torch.tensor([2.0]), torch.full((1, 128, 128), 0.75)


def test_native_map_coverage_and_maximum_rule():
    spatial, score = native_memory_map(TileMap(), Image.new("RGB", (301, 279)))
    assert spatial.shape == (279, 301) and (spatial == 0.75).all() and score == 2
    # Image maximum comes from descriptors, not from an interpolated native map maximum.
    assert score > float(spatial.max())
    with pytest.raises(InputRejected):
        native_memory_map(TileMap(), Image.new("RGB", (255, 279)))
    with pytest.raises(InputRejected):
        native_memory_map(TileMap(), Image.new("L", (301, 279)))


def test_invalid_map_or_uncovered(monkeypatch):
    class Bad(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([float("nan")]), torch.zeros((1, 128, 128))

    with pytest.raises(ValueError, match="Invalid"):
        native_memory_map(Bad(), Image.new("RGB", (256, 256)))
    monkeypatch.setattr(module, "coordinates", lambda *a, **k: [])
    with pytest.raises(ValueError, match="uncovered"):
        native_memory_map(TileMap(), Image.new("RGB", (256, 256)))


def test_same_quantization_including_above_one():
    rng = np.random.default_rng(19)
    mask = rng.integers(0, 2, (13, 27), dtype=np.uint8)
    values = rng.random(mask.shape, dtype=np.float32)
    old, new = PixelRanking(), DistancePixelRanking()
    old.add(values, mask)
    new.add(values, mask)
    assert np.array_equal(old.positive, new.positive) and np.array_equal(old.negative, new.negative)
    for key in ("auroc", "average_precision", "positive", "negative", "prevalence"):
        assert old.result()[key] == new.result()[key]
    values = values * 2
    new = DistancePixelRanking()
    new.add(values, mask)
    quantized = np.floor(values.astype(np.float64) * 65535).ravel()
    assert len(new.positive) > 65536
    assert new.result()["auroc"] == pytest.approx(roc_auc_score(mask.ravel(), quantized))
    assert new.result()["average_precision"] == pytest.approx(
        average_precision_score(mask.ravel(), quantized)
    )
    assert DistancePixelRanking().result()["auroc"] is None
    with pytest.raises(MemoryError):
        new.add(np.full((2, 2), 17.0), np.zeros((2, 2)))
    for bad in (float("nan"), -1):
        with pytest.raises(ValueError):
            new.add(np.full((2, 2), bad), np.zeros((2, 2)))


def test_artifact_replay_identity_and_corruption(tiles, tmp_path):
    expected = identity(MetalPartsSource(SYNTHETIC_REVISION), "b" * 64, "c" * 64, tiles.ids)
    model, _ = fit_native_memory(Extractor(), tiles, lambda: None)
    output = tmp_path / "artifact"
    output.mkdir()
    saved = save_memory(output, model.bank, expected)
    loaded, meta = load_memory(output, expected)
    assert saved == meta and torch.equal(loaded, model.bank)
    with pytest.raises(FileExistsError):
        save_memory(output, model.bank, expected)
    with pytest.raises(ValueError):
        load_memory(output, {**expected, "category": "cashew"})
    with pytest.raises(ValueError):
        load_memory(output, {**expected, "threshold": 0.5})
    with pytest.raises(ValueError):
        load_memory(output, {**expected, "split_sha256": "a" * 64})
    (output / "memory.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        load_memory(output, expected)
    (output / "memory.pt").unlink()
    with pytest.raises(ValueError):
        load_memory(output, expected)


def test_invalid_artifact_contract(tmp_path):
    with pytest.raises(ValueError):
        identity("cashew", "a" * 64, "b" * 64, ["tile"])
    with pytest.raises(ValueError):
        identity(MetalPartsSource(), "wrong", "b" * 64, ["tile"])
    with pytest.raises(ValueError):
        identity(MetalPartsSource(), "a" * 64, "b" * 64, [f"{i:05}" for i in range(2049)])
    expected = identity(MetalPartsSource(SYNTHETIC_REVISION), "a" * 64, "b" * 64, ["tile"])
    with pytest.raises(ValueError):
        save_memory(tmp_path, torch.zeros((3, 384)), expected)


def test_native_orientation_and_edge_signal():
    class ImageMap(torch.nn.Module):
        def forward(self, image):
            spatial = image[:, 0]
            return spatial.flatten(1).max(1).values, spatial

    array = np.zeros((300, 400, 3), dtype=np.uint8)
    array[:] = np.linspace(0, 255, 400, dtype=np.uint8)[None, :, None]
    spatial, score = native_memory_map(ImageMap(), Image.fromarray(array))
    assert spatial.shape == (300, 400) and spatial[150, 350] > spatial[150, 50]
    np.testing.assert_allclose(spatial[100], spatial[200], atol=1e-6)
    assert spatial[:, -1].min() > 0.98 and spatial[:, 0].max() < 0.02 and score > 0.99


def test_experiment_synthetic_end_to_end(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts/metalparts_feature_experiment.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("feature_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for name, path in {
        "ROOT": tmp_path,
        "SOURCE": tmp_path / "source",
        "REPORT": tmp_path / "report",
        "RUN": tmp_path / "runs/feature",
        "NORMAL": tmp_path / "runs/normal",
        "COMPARISON": tmp_path / "runs/comparison",
        "OLD": tmp_path / "runs/old",
        "SEAL": tmp_path / "seal.json",
        "WEIGHTS": tmp_path / "weights.pth",
    }.items():
        monkeypatch.setattr(runner, name, path)
    runner.OLD.mkdir(parents=True)
    runner.SEAL.write_text("{}")
    runner.WEIGHTS.write_bytes(b"fabricated")
    rows, verified, old_scores = [], {}, []
    ae_hist = PixelRanking()
    for name, anomaly, partition in [
        ("fit", 0, "train"),
        ("normal", 0, "validation"),
        ("defect", 1, "source_anomaly"),
    ]:
        item = MetalPartsSample(
            MetalPartsSource(),
            "train",
            f"{'anomaly' if anomaly else 'normal'}/{name}.png",
            Label.ANOMALY if anomaly else Label.NORMAL,
            f"MetalParts/metal_parts_data_dir_train/masks/{name}.png" if anomaly else None,
            301,
            279,
        )
        path = runner.SOURCE / item.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (301, 279), (100, 80, 40)).save(path)
        verified[item.image_path] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        mask = np.zeros((279, 301), dtype=np.uint8)
        if anomaly:
            mask[100:110, 100:110] = 1
            path = runner.SOURCE / item.mask_path
            path.parent.mkdir(parents=True)
            Image.fromarray(mask * 255).save(path)
            verified[item.mask_path] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        row = {
            "sample_id": item.sample_id,
            "file_name": item.file_name,
            "label": anomaly,
            "mask_path": item.mask_path,
            "width": 301,
            "height": 279,
            "image_path": item.image_path,
            "partition": partition,
            "image_sha256": verified[item.image_path]["sha256"],
            "pixel_sha256": "a" * 64,
        }
        rows.append(row)
        if partition != "train":
            array = np.full((279, 301), 0.05, dtype=np.float32)
            np.save(runner.OLD / (item.sample_id + ".npy"), array)
            ae_hist.add(array, mask)
            old_scores.append({"sample_id": item.sample_id, "map": item.sample_id + ".npy"})
    expected_ids = FittingTiles(runner.SOURCE, runner.fitting_records(rows)).ids
    sealed = {
        "identity": identity(MetalPartsSource(), "a" * 64, "b" * 64, expected_ids),
        "inventory": rows,
        "selected_ids": [r["sample_id"] for r in rows if r["partition"] != "train"],
        "preview_ids": [rows[-1]["sample_id"]],
        "fitting_parent_ids": [rows[0]["sample_id"]],
    }
    (runner.OLD / "scores.json").write_text(json.dumps(old_scores))
    (runner.OLD / "metrics.json").write_text(
        json.dumps(
            {
                "pixel": ae_hist.result(),
                "image": {},
                "latency_seconds": {},
                "peak_working_set_bytes": 0,
            }
        )
    )
    monkeypatch.setattr(runner, "context", lambda: (sealed, verified))
    monkeypatch.setattr(runner, "load_extractor", lambda path: Extractor())
    # Exercise guard construction separately; do not leave process-wide hooks in pytest.
    monkeypatch.setattr(runner.sys, "addaudithook", lambda guard: None)
    runner.fit()
    assert runner.read(runner.RUN / "fit.json")["exact_determinism"]
    runner.normal()
    runner.compare()
    result = runner.read(runner.COMPARISON / "metrics.json")
    assert result["ae_cached_pixel_metrics_exact"]
    assert result["feature_memory"]["image"]["positive"] == 1
    assert result["feature_memory"]["pixel"]["positive"] == 100
    assert len(result["preview_assets"]) == 1
    with pytest.raises(FileExistsError):
        runner.compare()
