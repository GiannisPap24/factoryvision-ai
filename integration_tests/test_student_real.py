"""Explicit permitted-normal and saved student evidence checks; no official test metrics."""

import json
from pathlib import Path

import numpy as np
import torch

from factoryvision.data.development import DevelopmentDataset
from factoryvision.data.images import make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development
from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.models.feature_memory import load_extractor
from factoryvision.models.student_teacher import SpatialTeacher, StudentTeacher
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.student_artifacts import load_student
from factoryvision.training.student_engine import epoch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cashew-student-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_preserved_prior_artifacts_and_frozen_student_sources():
    for milestone in ("2a", "2b", "2c"):
        for name, r in read(ROOT / f"docs/evidence/milestone-{milestone}-verification.json")[
            "artifacts"
        ].items():
            assert file_sha256(ROOT / name) == r["sha256"]
    for name, sha in read(RUN / "experiment.json")["source_hashes"].items():
        assert file_sha256(ROOT / name) == sha
    for name, sha in read(RUN / "evaluate-hashes.json").items():
        assert file_sha256(RUN / name) == sha


def test_exact_training_membership_real_updates_and_selection():
    experiment = read(RUN / "experiment.json")
    history = read(RUN / "history.json")
    result = read(RUN / "result.json")
    assert len(experiment["fit_ids"]) == 315 and len(experiment["validation_ids"]) == 67
    assert not set(experiment["fit_ids"]) & set(experiment["validation_ids"])
    for row in history:
        assert sorted(row["training_ids"]) == experiment["fit_ids"]
        assert row["validation_ids"] == experiment["validation_ids"]
        assert np.isfinite([row["train_loss"], row["validation_loss"]]).all()
        assert row["updates"] == 40
    best = min(history, key=lambda r: r["validation_loss"])
    assert result["best_epoch"] == best["epoch"] and result["updates"] == sum(
        r["updates"] for r in history
    )
    assert result["student_initial_sha256"] != result["student_best_sha256"]
    assert result["teacher_before"] == result["teacher_after"]
    assert result["reload_score_max_abs"] <= 1e-6 and result["reload_map_max_abs"] <= 1e-6
    assert result["peak_working_set_bytes"] < 2 * 1024**3 and result["elapsed_seconds"] < 600


def test_permitted_read_logs_and_calibration_isolation():
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    allowed = {s.image_path for s in manifest.samples if s.partition != Partition.TEST}
    for name in ("train-reads.json", "evaluate-reads.json"):
        r = read(RUN / name)
        assert (
            set(r["opened_paths"]) == allowed
            and r["unique_files"] == 450
            and r["official_test_files"] == 0
        )
    records = read(RUN / "normal-scores.json")
    ids = sorted(s.sample_id for s in manifest.samples if s.partition == Partition.CALIBRATION)
    assert calibrate_development(records["calibration"], ids) == read(RUN / "calibration.json")


def test_actual_installed_weights_reload_all_normal_validation_loss():
    torch.set_num_threads(4)
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    experiment = read(RUN / "experiment.json")
    student, meta = load_student(
        RUN, manifest_sha256(manifest), experiment["recipe_sha256"], experiment["fit_ids"]
    )
    model = StudentTeacher(
        SpatialTeacher(load_extractor(ROOT / "data/pretrained/resnet18-f37072fd.pth")), student
    )
    assert parameter_sha256(student) == read(RUN / "result.json")["student_best_sha256"]
    ds = DevelopmentDataset(manifest, ROOT / "data/visa", Partition.VALIDATION)
    val = epoch(model, make_loader(ds, 8), experiment["validation_ids"], "validation")
    assert abs(val["loss"] - meta["validation_loss"]) < 1e-6
    x = next(iter(make_loader(ds, 8)))["image"]
    score, spatial = model.score(x)
    saved = torch.load(RUN / "best-probe.pt", weights_only=True, map_location="cpu")
    torch.testing.assert_close(score, saved["score"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(spatial, saved["map"], atol=1e-6, rtol=1e-5)


def test_saved_synthetic_maps_and_state_integrity():
    result = read(RUN / "development.json")
    assert result["state_before"] == result["state_after"]
    rows = read(RUN / "synthetic-probes.json")
    assert all(r["synthetic"] for r in rows)
    assert [r["source_id"] for r in rows] == read(RUN / "normal-scores.json")["validation"][
        "sample_ids"
    ][:16]
    with np.load(RUN / "synthetic-maps.npz", allow_pickle=False) as a:
        assert a["maps"].shape == a["masks"].shape == (16, 128, 128)
        assert a["masks"].sum() == 4096
        assert (
            binary_metrics(a["masks"].ravel(), a["maps"].ravel())
            == result["synthetic"]["pixel_metrics"]
        )
