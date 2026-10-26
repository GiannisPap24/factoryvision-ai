"""Explicit replay checks on frozen real outputs; no new fitting or test inference."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from factoryvision.data.manifest import load_manifest
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.pipeline import summarize
from factoryvision.evaluation.protocol import PROTOCOL_SHA256, load_protocol, verify_model_artifacts
from factoryvision.training.checkpoint import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/cashew-eval-v1"


def read(name):
    return json.loads((RUN / name).read_text())


def test_freeze_precedes_test_predictions_and_artifacts_unchanged():
    protocol = load_protocol(ROOT / "docs/protocols/cashew-evaluation-v1.json")
    verify_model_artifacts(ROOT, protocol)
    frozen = read("frozen-artifacts.json")
    started = read("evaluation-started.json")
    assert frozen["protocol_sha256"] == started["protocol_sha256"] == PROTOCOL_SHA256
    assert (
        datetime.fromisoformat(protocol["frozen_at_utc"])
        < datetime.fromisoformat(frozen["frozen_at_utc"])
        < datetime.fromisoformat(started["at_utc"])
    )
    assert started["frozen_artifacts_sha256"] == file_sha256(RUN / "frozen-artifacts.json")
    for name, digest in read("artifact-hashes.json").items():
        assert file_sha256(RUN / name) == digest, name
    for name, digest in frozen["execution_sources"].items():
        assert file_sha256(ROOT / name) == digest, name


def test_baseline_fit_and_calibration_membership_are_isolated():
    m = load_manifest(ROOT / "data/manifests/cashew-seed42.json")

    def expected(partition):
        return sorted(s.sample_id for s in m.samples if s.partition == partition)

    assert read("baseline-fit-ids.json") == expected(Partition.TRAIN)
    assert read("baseline-calibration-scores.json")["sample_ids"] == expected(Partition.CALIBRATION)
    assert set(read("baseline-fit-ids.json")).isdisjoint(expected(Partition.TEST))
    assert len(read("baseline-fit-ids.json")) == 315
    assert read("baseline-calibration.json")["count"] == 68


def test_saved_image_and_pixel_metrics_replay_exactly():
    rows = read("per-image.json")
    report = read("metrics.json")
    with np.load(RUN / "spatial-maps.npz", allow_pickle=False) as arrays:
        masks = arrays["masks"]
        assert masks.shape == (150, 128, 128) and masks.sum() == 27672
        assert np.isin(masks, (0, 1)).all()
        for name in ("autoencoder", "spatial_mean"):
            saved = report["models"][name]
            assert summarize(rows[name], arrays[name], masks, saved["image"]["threshold"]) == saved
            np.testing.assert_allclose(
                arrays[name].mean(axis=(1, 2)),
                [r["score"] for r in rows[name]],
                rtol=1e-6,
                atol=1e-8,
            )
            positives = np.array([r["score"] for r in rows[name] if r["label"] == 1])
            negatives = np.array([r["score"] for r in rows[name] if r["label"] == 0])
            independent_auc = (positives[:, None] > negatives).mean() + 0.5 * (
                positives[:, None] == negatives
            ).mean()
            assert abs(independent_auc - saved["image"]["auroc"]) < 1e-12


def test_exact_test_membership_labels_and_predeclared_examples():
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    expected = sorted(
        (s for s in manifest.samples if s.partition == Partition.TEST), key=lambda s: s.sample_id
    )
    rows = read("per-image.json")
    for records in rows.values():
        assert [r["sample_id"] for r in records] == [s.sample_id for s in expected]
        assert [r["label"] for r in records] == [int(s.label.value == "anomaly") for s in expected]
        assert sum(r["label"] for r in records) == 100
    selected = [s.sample_id for s in expected if s.label.value == "anomaly"][:4] + [
        s.sample_id for s in expected if s.label.value == "normal"
    ][:2]
    assert {p.stem.removeprefix("example-") for p in RUN.glob("example-*.png")} == set(selected)
