"""Synthetic frozen manifests and metrics; no real data, checkpoint or test-set access."""

import json

import numpy as np
import pytest

from factoryvision.evaluation.confirmatory import (
    authorize_opening,
    pooled_metrics,
    summarize,
    verified_seal,
)
from factoryvision.models.pcb1 import FAMILIES, PROTOCOL_SHA
from factoryvision.training.checkpoint import file_sha256


@pytest.fixture
def sealed(tmp_path):
    membership = {
        "train": ["pcb1-a"],
        "validation": ["pcb1-b"],
        "calibration": ["pcb1-c", "pcb1-d"],
        "test": ["pcb1-e"],
    }
    files, models = {}, {}
    for family in FAMILIES:
        directory = f"runs/{family}"
        root = tmp_path / directory
        root.mkdir(parents=True)
        values = {
            "best.pt": "synthetic bytes only",
            "checkpoint.json": {},
            "development.json": {},
            "pcb1-provenance.json": dict(
                family=family,
                smoke=False,
                manifest_sha256="a" * 64,
                fit_ids=membership["train"],
                calibration_ids=membership["calibration"],
            ),
            "calibration.json": dict(
                threshold=0.5, comparison="score > threshold", quantile=0.95, interpolation="linear"
            ),
            "normal-scores.json": dict(
                calibration=dict(
                    partition="calibration", sample_ids=membership["calibration"], scores=[0.1, 0.5]
                )
            ),
        }
        for name, value in values.items():
            path = root / name
            path.write_text(json.dumps(value))
            files[f"{directory}/{name}"] = file_sha256(path)
        models[family] = dict(directory=directory, threshold=0.5)
    seal = dict(
        schema=1,
        category="pcb1",
        status="READY FOR BLINDED PCB1 TEST",
        protocol_sha256=PROTOCOL_SHA,
        manifest_sha256="a" * 64,
        membership=membership,
        files=files,
        models=models,
    )
    path = tmp_path / "seal.json"
    path.write_text(json.dumps(seal))
    return tmp_path, path, seal


def test_seal_valid_then_tamper(sealed):
    root, path, seal = sealed
    assert verified_seal(root, path, file_sha256(path)) == seal
    with pytest.raises(ValueError, match="identity"):
        verified_seal(root, path, "0" * 64)
    (root / next(iter(seal["files"]))).write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        verified_seal(root, path, file_sha256(path))


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda s: s.update(status="BLOCKED"), "unready"),
        (lambda s: s.update(files={}), "Empty"),
        (lambda s: s["models"]["autoencoder"].update(directory="elsewhere"), "location"),
        (lambda s: s["files"].pop("runs/autoencoder/best.pt"), "Required"),
        (lambda s: s["models"]["autoencoder"].update(threshold=0.9), "provenance"),
        (lambda s: s["membership"].update(test=s["membership"]["train"]), "overlap"),
    ],
)
def test_seal_contract_rejections(sealed, change, error):
    root, path, seal = sealed
    change(seal)
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError, match=error):
        verified_seal(root, path, file_sha256(path))


def test_separate_bound_authorization(tmp_path):
    path = tmp_path / "approval.json"
    path.write_text("{}")
    with pytest.raises(PermissionError):
        authorize_opening(path, "a" * 64)
    path.write_text(
        json.dumps(
            dict(
                decision="OWNER_AUTHORIZES_PCB1_HELDOUT_OPENING",
                seal_sha256="a" * 64,
                milestone="2E2",
            )
        )
    )
    assert authorize_opening(path, "a" * 64)["milestone"] == "2E2"
    with pytest.raises(PermissionError):
        authorize_opening(path, "b" * 64)


def test_exact_pooled_pixels_and_edges():
    assert pooled_metrics([0, 1], [0.1, 0.9])["average_precision"] == 1
    assert pooled_metrics([0, 0], [0.1, 0.9])["auroc"] is None
    # The historical 3M cap must not silently truncate PCB1's full pooled pixels.
    y = np.zeros(3_000_001, dtype=np.uint8)
    s = np.zeros_like(y, dtype=np.float32)
    assert pooled_metrics(y, s)["pixels"] == 3_000_001
    for y, s in (([], []), ([0], [float("nan")]), ([2], [0.1]), ([0, 1], [0.1])):
        with pytest.raises(ValueError):
            pooled_metrics(y, s)


def test_four_family_metrics_and_determinism():
    masks = np.zeros((2, 128, 128), dtype=np.uint8)
    masks[1, 4:8, 4:8] = 1
    maps = {f: masks.astype(np.float32) for f in FAMILIES}
    scores = {f: [0.1, 0.9] for f in FAMILIES}
    thresholds = {f: 0.5 for f in FAMILIES}
    report = summarize([0, 1], masks, scores, maps, thresholds)
    assert report == summarize([0, 1], masks, scores, maps, thresholds)
    assert set(report["descriptive_pixel_ap_ranks"].values()) == {1}
    assert report["models"]["autoencoder"]["image"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert report["paired_image_intervals"]["autoencoder"]["auroc"]["ci95_percentile"] == [0.0, 0.0]
    masks[:] = 0
    report = summarize([0, 0], masks, scores, maps, thresholds)
    assert report["paired_image_intervals"]["autoencoder"] is None
    assert report["models"]["autoencoder"]["image"]["recall"] is None
    assert report["models"]["autoencoder"]["image"]["fp"] == 1
    with pytest.raises(ValueError):
        summarize([0], masks, scores, maps, thresholds)
    maps["autoencoder"] = np.zeros((2, 16, 16))
    with pytest.raises(ValueError, match="geometry"):
        summarize([0, 0], masks, scores, maps, thresholds)
