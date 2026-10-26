"""Explicit 2C normal development checks. No official Cashew test image/mask reads."""

import json
from pathlib import Path

import numpy as np
import torch

from factoryvision.data.development import DevelopmentDataset
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development
from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.models.feature_memory import FeatureMemory, load_extractor
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.feature_artifacts import load_memory

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cashew-feature-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_historical_artifacts_unchanged_without_recomputing_test_metrics():
    for milestone in ("2a", "2b"):
        evidence = read(ROOT / f"docs/evidence/milestone-{milestone}-verification.json")
        for name, record in evidence["artifacts"].items():
            assert file_sha256(ROOT / name) == record["sha256"], name
    old = read(ROOT / "runs/cashew-eval-v1/frozen-artifacts.json")
    for name, sha in old["execution_sources"].items():
        assert file_sha256(ROOT / name) == sha, name


def test_recipe_sources_artifacts_and_permitted_read_logs():
    for name, sha in read(RUN / "source-snapshot.json").items():
        assert file_sha256(ROOT / name) == sha, name
    for name, sha in read(RUN / "evaluate-hashes.json").items():
        assert file_sha256(RUN / name) == sha, name
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    allowed = {s.image_path for s in manifest.samples if s.partition != Partition.TEST}
    assert len(allowed) == 450
    for name in ("fit-reads.json", "evaluate-reads.json"):
        report = read(RUN / name)
        assert set(report["opened_paths"]) == allowed
        assert report["unique_files"] == 450 and report["official_test_files"] == 0
    assert (
        file_sha256(ROOT / "docs/protocols/feature-memory-development-v1.json")
        == read(RUN / "fit.json")["recipe_sha256"]
    )
    for name in ("fit.json", "development.json"):
        result = read(RUN / name)
        assert result["backbone_state_before"] == result["backbone_state_after"]
        assert result["peak_working_set_bytes"] < 2 * 1024**3


def test_all_permitted_normal_images_hash_decode_and_geometry():
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    audit = {r["sample_id"]: r for r in read(ROOT / "data/metadata/cashew-audit.json")["records"]}
    for partition, count in (
        (Partition.TRAIN, 315),
        (Partition.VALIDATION, 67),
        (Partition.CALIBRATION, 68),
    ):
        dataset = DevelopmentDataset(manifest, ROOT / "data/visa", partition)
        assert len(dataset) == count
        for i, s in enumerate(dataset.samples):
            assert (
                file_sha256(ROOT / "data/visa" / s.image_path) == audit[s.sample_id]["image_sha256"]
            )
            sample = dataset[i]
            assert sample["image"].shape == (3, 128, 128) and sample["label"] == 0
            assert sample["original_size"].tolist() == [1176, 1274]  # Height, width.
            assert not sample["mask_available"] and sample["mask"].sum() == 0


def test_memory_membership_real_reload_and_normal_calibration():
    torch.set_num_threads(4)
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    ids = {
        p.value: sorted(s.sample_id for s in manifest.samples if s.partition == p)
        for p in (Partition.TRAIN, Partition.VALIDATION, Partition.CALIBRATION)
    }
    recipe = file_sha256(ROOT / "docs/protocols/feature-memory-development-v1.json")
    bank, meta = load_memory(RUN, manifest_sha256(manifest), recipe, ids["train"])
    assert bank.shape == (5040, 384) and meta["fit_ids"] == ids["train"]
    selected = read(RUN / "patch-selection.json")
    assert [r["sample_id"] for r in selected] == ids["train"]
    assert all(len(r["patch_indices"]) == len(set(r["patch_indices"])) == 16 for r in selected)
    records = read(RUN / "normal-scores.json")
    for p in ("validation", "calibration"):
        assert records[p]["sample_ids"] == ids[p]
    assert calibrate_development(records["calibration"], ids["calibration"]) == read(
        RUN / "calibration.json"
    )
    model = FeatureMemory(load_extractor(ROOT / "data/pretrained/resnet18-f37072fd.pth"), bank)
    assert parameter_sha256(model.extractor) == read(RUN / "fit.json")["backbone_state_after"]
    sample = DevelopmentDataset(manifest, ROOT / "data/visa", Partition.VALIDATION)[0]
    score, spatial = model(sample["image"][None])
    assert abs(float(score[0]) - records["validation"]["scores"][0]) < 1e-6
    assert spatial.shape == (1, 128, 128)


def test_synthetic_saved_maps_metrics_and_identity_replay():
    rows = read(RUN / "synthetic-probes.json")
    normal = read(RUN / "normal-scores.json")
    assert [r["source_id"] for r in rows] == normal["validation"]["sample_ids"][:16]
    assert all(r["synthetic"] for r in rows)
    with np.load(RUN / "synthetic-maps.npz", allow_pickle=False) as a:
        assert a["maps"].shape == a["masks"].shape == (16, 128, 128)
        assert a["masks"].sum() == 4096 and np.isin(a["masks"], (0, 1)).all()
        assert (
            binary_metrics(a["masks"].ravel(), a["maps"].ravel())
            == read(RUN / "development.json")["synthetic"]["pixel_metrics"]
        )
