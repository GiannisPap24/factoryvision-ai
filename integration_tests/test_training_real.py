"""Explicit completed-experiment verification; never trains or evaluates test images."""

import json
from pathlib import Path

import torch

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.reconstruction import calibrate_normal, reconstruction_errors
from factoryvision.training.checkpoint import file_sha256, load_checkpoint
from factoryvision.training.engine import score_normals, seed_everything

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cashew-ae-128-seed42"


def read(name):
    return json.loads((RUN / name).read_text())


def test_full_run_evidence_and_parameter_updates():
    config, result, history = read("experiment.json"), read("result.json"), read("history.json")
    assert result["completed"] and result["epochs"] == 12 and result["optimizer_steps"] == 240
    assert config["initial_parameter_sha256"] != result["final_parameter_sha256"]
    assert result["checkpoint_sha256"] == file_sha256(RUN / "best.pt")
    assert all(r["train_samples"] == 315 and r["validation_samples"] == 67 for r in history)
    assert all(r["parameters_changed"] and r["optimizer_steps"] == 20 for r in history)
    assert all(
        torch.isfinite(torch.tensor([r["train_loss"], r["validation_loss"]])).all() for r in history
    )
    assert result["best_epoch"] == min(history, key=lambda r: r["validation_loss"])["epoch"]
    assert not result["official_test_scored"]
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    assert set(read("last-training-order.json")) == {
        s.sample_id for s in manifest.samples if s.partition == Partition.TRAIN
    }


def test_real_reload_validation_and_spatial_shapes():
    seed_everything(42, 4)
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    model, _ = load_checkpoint(RUN, manifest_sha256(manifest), 128)
    data = ManifestDataset(
        manifest, ROOT / "data/visa", Partition.VALIDATION, ImageSettings(128, 128)
    )
    loader = make_loader(data, 16)
    assert score_normals(model, loader, "validation") == read("best-validation.json")
    batch = next(iter(loader))
    with torch.inference_mode():
        scores, maps = reconstruction_errors(batch["image"], model(batch["image"]))
    assert scores.shape == (16,) and maps.shape == (16, 128, 128)
    assert maps.device.type == "cpu" and torch.isfinite(maps).all()


def test_calibration_uses_exact_normal_partition_and_fixed_rule():
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    record = read("calibration-scores.json")
    assert record["sample_ids"] == sorted(
        s.sample_id for s in manifest.samples if s.partition == Partition.CALIBRATION
    )
    assert len(record["scores"]) == 68
    assert calibrate_normal(torch.tensor(record["scores"], dtype=torch.float64), 0.95) == read(
        "calibration.json"
    )
    assert set(record["sample_ids"]).isdisjoint(read("last-training-order.json"))


def test_exported_metadata_contains_no_absolute_local_paths():
    for path in RUN.glob("*.json"):
        text = path.read_text()
        assert str(ROOT) not in text and "C:\\" not in text and "C:/" not in text
        assert "GiannisPap24" not in text and "@users.noreply" not in text
