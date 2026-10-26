"""Synthetic exact metric, isolation, alignment, freeze and report checks; no real images."""

import hashlib
import json

import numpy as np
import pytest
import torch

from factoryvision.evaluation import protocol
from factoryvision.evaluation.metrics import binary_metrics, paired_bootstrap
from factoryvision.evaluation.pipeline import evaluate_batches, summarize
from factoryvision.evaluation.reconstruction import calibrate_normal
from factoryvision.models.spatial_mean import SpatialMean, fit_spatial_mean
from factoryvision.training.engine import score_normals


def batch(partition="test"):
    image = torch.stack((torch.zeros(3, 4, 4), torch.ones(3, 4, 4)))
    return {
        "image": image,
        "mask": torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4))),
        "label": torch.tensor([0, 1] if partition == "test" else [0, 0]),
        "partition": [partition] * 2,
        "sample_id": ["synthetic-a", "synthetic-b"],
    }


def test_known_ranking_and_confusion():
    r = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.4)
    assert r["auroc"] == 0.75 and r["average_precision"] == pytest.approx(5 / 6)
    assert r["confusion_matrix"] == [[2, 0], [1, 1]]
    assert r["precision"] == 1 and r["recall"] == 0.5 and r["f1"] == pytest.approx(2 / 3)
    assert r["normal_fpr"] == 0
    assert binary_metrics([0, 1], [0, 1])["auroc"] == 1
    assert binary_metrics([0, 1], [1, 0])["auroc"] == 0
    tied = binary_metrics([0, 1], [0.5, 0.5], 0.5)
    assert tied["auroc"] == 0.5 and tied["average_precision"] == 0.5
    assert tied["precision"] is None and tied["recall"] == 0 and tied["f1"] == 0


def test_undefined_metrics_are_null_and_json_safe():
    r = binary_metrics([0, 0], [0, 0], 0)
    assert all(r[k] is None for k in ("auroc", "average_precision", "precision", "recall", "f1"))
    assert r["normal_fpr"] == 0
    r = binary_metrics([1, 1], [1, 1], 0)
    assert r["auroc"] is None and r["average_precision"] is None and r["normal_fpr"] is None
    assert r["precision"] == r["recall"] == r["f1"] == 1
    assert "NaN" not in json.dumps(r, allow_nan=False)


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([], []),
        ([0], [0, 1]),
        ([[0]], [[0]]),
        ([2], [0]),
        ([0], [np.nan]),
        ([1], [np.inf]),
        ([float("nan")], [0]),
    ],
)
def test_invalid_metric_inputs(labels, scores):
    with pytest.raises(ValueError):
        binary_metrics(labels, scores)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, True, "0"])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.0, 1.0], threshold)


def test_pixel_capacity_limit():
    with pytest.raises(ValueError):
        binary_metrics(np.zeros(3_000_001, dtype=np.uint8), np.zeros(3_000_001, dtype=np.float32))


def test_paired_bootstrap_known_differences_and_repeatability():
    y = [0, 0, 1, 1]
    perfect = [0, 0, 1, 1]
    reverse = [1, 1, 0, 0]
    r = paired_bootstrap(y, perfect, reverse, repeats=12)
    assert r["auroc"] == {"ae_minus_mean": 1.0, "ci95_percentile": [1.0, 1.0]}
    assert r == paired_bootstrap(y, perfect, reverse, repeats=12)
    assert paired_bootstrap(y, perfect, perfect, repeats=12)["average_precision"][
        "ci95_percentile"
    ] == [0.0, 0.0]
    with pytest.raises(ValueError):
        paired_bootstrap([0, 0], [0, 1], [1, 0], 12)
    with pytest.raises(ValueError):
        paired_bootstrap(y, perfect, reverse, 1)


def test_training_only_mean_exact_reference_and_calibration():
    b = batch("train")
    model, ids = fit_spatial_mean([b])
    assert ids == b["sample_id"] and len(list(model.parameters())) == 0
    torch.testing.assert_close(model.reference, torch.full((3, 4, 4), 0.5), rtol=0, atol=0)
    b["partition"] = ["calibration"] * 2
    scores = score_normals(model, [b], "calibration")
    assert scores["scores"] == [0.25, 0.25]
    assert calibrate_normal(torch.tensor(scores["scores"]))["threshold"] == 0.25
    assert calibrate_normal(torch.tensor(scores["scores"]))["exceedances"] == 0
    torch.testing.assert_close(model(b["image"])[0], model.reference)


@pytest.mark.parametrize("partition", ["test", "validation", "calibration"])
def test_baseline_refuses_nontraining_partition(partition):
    with pytest.raises(ValueError):
        fit_spatial_mean([batch(partition)])


def test_baseline_refuses_anomalies_order_duplicates_and_geometry():
    b = batch("train")
    b["label"][0] = 1
    with pytest.raises(ValueError):
        fit_spatial_mean([b])
    b = batch("train")
    b["sample_id"].reverse()
    with pytest.raises(ValueError):
        fit_spatial_mean([b])
    b = batch("train")
    b["sample_id"] = ["same", "same"]
    with pytest.raises(ValueError):
        fit_spatial_mean([b])
    b = batch("train")
    b["sample_id"] = ["one"]
    with pytest.raises(ValueError):
        fit_spatial_mean([b])
    with pytest.raises(ValueError):
        fit_spatial_mean([])
    b = batch("train")
    c = batch("train")
    c["image"] = torch.zeros(2, 3, 2, 2)
    with pytest.raises(ValueError):
        fit_spatial_mean([b, c])


@pytest.mark.parametrize(
    "reference",
    [
        torch.zeros(1, 4, 4),
        torch.zeros(3, 0, 0),
        torch.zeros(3, 4, 4, dtype=torch.float64),
        torch.full((3, 4, 4), float("nan")),
        torch.full((3, 4, 4), -1.0),
        torch.full((3, 4, 4), 2.0),
    ],
)
def test_invalid_reference(reference):
    with pytest.raises(ValueError):
        SpatialMean(reference)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(1, 1, 4, 4),
        torch.zeros(0, 3, 4, 4),
        torch.zeros(1, 3, 2, 2),
        torch.full((1, 3, 4, 4), float("nan")),
        torch.zeros(1, 3, 4, 4, dtype=torch.float64),
        torch.full((1, 3, 4, 4), 2.0),
    ],
)
def test_invalid_baseline_image(image):
    with pytest.raises(ValueError):
        SpatialMean(torch.zeros(3, 4, 4))(image)


def test_pipeline_exact_map_alignment_order_and_report():
    b = batch()
    model = SpatialMean(torch.zeros(3, 4, 4))
    model.train()
    rows, maps, masks = evaluate_batches(model, [b], b["sample_id"], 0.5)
    assert not model.training and maps.shape == masks.shape == (2, 4, 4)
    assert maps.dtype == np.float32 and masks.dtype == np.uint8
    assert np.array_equal(maps, masks)
    report = summarize(rows, maps, masks, 0.5)
    assert report["image"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert report["pixel"]["auroc"] == report["pixel"]["average_precision"] == 1
    assert report["anomaly_masks_empty_after_resize"] == 0
    assert [r["score"] for r in rows] == [0.0, 1.0]
    assert [r["prediction"] for r in rows] == [0, 1]
    # Compare Python float64 threshold, without first rounding it into a float32 tensor.
    below_one = 1.0 - 1e-10
    assert evaluate_batches(model, [b], b["sample_id"], below_one)[0][1]["prediction"] == 1
    assert evaluate_batches(model, [b], b["sample_id"], 1.0)[0][1]["prediction"] == 0
    assert json.loads(json.dumps(report)) == report


@pytest.mark.parametrize(
    "case",
    [
        "partition",
        "labels",
        "mask_shape",
        "mask_nonfinite",
        "mask_fraction",
        "normal_mask",
        "ids_count",
    ],
)
def test_invalid_evaluation_batch(case):
    b = batch()
    model = SpatialMean(torch.zeros(3, 4, 4))
    if case == "partition":
        b["partition"] = ["train"] * 2
    elif case == "labels":
        b["label"][0] = 2
    elif case == "mask_shape":
        b["mask"] = torch.zeros(2, 1, 3, 4)
    elif case == "mask_nonfinite":
        b["mask"][1, 0, 0, 0] = float("nan")
    elif case == "mask_fraction":
        b["mask"][1, 0, 0, 0] = 0.5
    elif case == "normal_mask":
        b["mask"][0, 0, 0, 0] = 1
    else:
        b["sample_id"] = ["one"]
    with pytest.raises(ValueError):
        evaluate_batches(model, [b], ["synthetic-a", "synthetic-b"], 0.5)


def test_pipeline_order_capacity_and_summary_rejection():
    b = batch()
    model = SpatialMean(torch.zeros(3, 4, 4))
    for ids in ([], ["same", "same"], ["b", "a"]):
        with pytest.raises(ValueError):
            evaluate_batches(model, [b], ids, 0.5)
    with pytest.raises(ValueError):
        evaluate_batches(model, [], b["sample_id"], 0.5)
    with pytest.raises(ValueError):
        evaluate_batches(model, [b], ["a", "b"], 0.5)
    with pytest.raises(ValueError):
        evaluate_batches(model, [b], b["sample_id"], 0.5, max_pixels=1)
    rows, maps, masks = evaluate_batches(model, [b], b["sample_id"], 0.5)
    with pytest.raises(ValueError):
        summarize(rows, maps, masks[:1], 0.5)
    with pytest.raises(ValueError):
        summarize(rows, maps, masks.astype(float) + 0.5, 0.5)
    rows[0]["prediction"] = 1
    with pytest.raises(ValueError):
        summarize(rows, maps, masks, 0.5)


def test_protocol_hash_and_model_artifact_identity(tmp_path, monkeypatch):
    p = tmp_path / "protocol.json"
    p.write_text('{"schema":1}')
    monkeypatch.setattr(protocol, "PROTOCOL_SHA256", hashlib.sha256(p.read_bytes()).hexdigest())
    assert protocol.load_protocol(p) == {"schema": 1}
    p.write_text("{}")
    with pytest.raises(ValueError):
        protocol.load_protocol(p)
    names = ["best.pt", "checkpoint.json", "calibration.json", "calibration-scores.json"]
    directory = tmp_path / "runs/cashew-ae-128-seed42"
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"synthetic fixture, not real weights")
    digest = hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest()
    value = {
        "autoencoder": {
            "checkpoint": "runs/cashew-ae-128-seed42/best.pt",
            "sha256": digest,
            "metadata_sha256": digest,
            "calibration_sha256": digest,
            "calibration_scores_sha256": digest,
        }
    }
    protocol.verify_model_artifacts(tmp_path, value)
    (directory / "best.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError):
        protocol.verify_model_artifacts(tmp_path, value)
