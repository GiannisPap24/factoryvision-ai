"""Explicit opt-in normal-only PCB1 artifact checks; never collected by default discovery."""

import json
import sys
from pathlib import Path

import pytest
import torch

from factoryvision.data.category import audit_normals, normal_members
from factoryvision.data.development import DevelopmentDataset, DevelopmentReadGuard
from factoryvision.data.images import make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import Partition
from factoryvision.models.pcb1 import FAMILIES, load_family, predict
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256, seed_everything

ROOT = Path(__file__).resolve().parents[1]


def test_official_normal_inventory_and_split_only():
    official = load_manifest(ROOT / "data/manifests/pcb1-official.json")
    derived = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    assert derive_splits(official, 42) == derived
    root = ROOT / "data/pcb1-development"
    guard = DevelopmentReadGuard(root, normal_members(derived))
    sys.addaudithook(guard)
    enriched, audit = audit_normals(official, root)
    assert (
        enriched == official and audit["normal_images"] == 904 and audit["test_images_decoded"] == 0
    )
    assert not audit["byte_duplicate_groups"] and not audit["pixel_duplicate_groups"]
    assert len(guard.opened) == 904
    assert not (ROOT / "reports/pcb1-opening").exists()
    assert not (ROOT / "data/pcb1-test-quarantine").exists()


@pytest.mark.parametrize("family", FAMILIES)
def test_all_normal_scores_reload_without_recalibration(family):
    seed_everything(42, 4)
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    fingerprint = manifest_sha256(manifest)
    root = ROOT / "data/pcb1-development"
    guard = DevelopmentReadGuard(root, normal_members(manifest))
    sys.addaudithook(guard)
    fit_ids = [s.sample_id for s in manifest.samples if s.partition == Partition.TRAIN]
    directory = ROOT / "runs" / f"pcb1-{family.replace('_', '-')}-v1"
    model = load_family(
        directory, family, fingerprint, fit_ids, ROOT / "data/pretrained/resnet18-f37072fd.pth"
    )
    before = parameter_sha256(model)
    recorded = json.loads((directory / "normal-scores.json").read_text())
    threshold_sha = file_sha256(directory / "calibration.json")
    for partition in ("validation", "calibration"):
        dataset = DevelopmentDataset(manifest, root, Partition(partition))
        scores = []
        for batch in make_loader(dataset, 8):
            s, _ = predict(model, family, batch["image"])
            scores.extend(s.tolist())
        torch.testing.assert_close(
            torch.tensor(scores), torch.tensor(recorded[partition]["scores"]), atol=1e-6, rtol=1e-5
        )
        assert [s.sample_id for s in dataset.samples] == recorded[partition]["sample_ids"]
    assert parameter_sha256(model) == before
    assert file_sha256(directory / "calibration.json") == threshold_sha
    assert len(guard.opened) == 272
