"""Normal development scoring/calibration only; no real anomaly performance estimate."""

import torch

from factoryvision.evaluation.reconstruction import calibrate_normal
from factoryvision.models.feature_memory import checked_ids, require_development_batch


@torch.inference_mode()
def score_development(model, loader, partition, expected_ids):
    if partition not in ("validation", "calibration"):
        raise ValueError("Development scoring permits validation/calibration normals only.")
    checked_ids(expected_ids)
    model.eval()
    ids, scores = [], []
    for batch in loader:
        require_development_batch(batch, partition, expected_ids, len(ids))
        values, _ = model(batch["image"])
        if values.shape != (len(batch["image"]),) or not torch.isfinite(values).all():
            raise ValueError("Invalid image scores.")
        ids.extend(batch["sample_id"])
        scores.extend(values.tolist())
    if ids != expected_ids:
        raise ValueError("Incomplete development membership.")
    return {"partition": partition, "sample_ids": ids, "scores": scores}


def calibrate_development(record, expected_ids):
    checked_ids(expected_ids)
    if record["partition"] != "calibration" or record["sample_ids"] != expected_ids:
        raise ValueError("Threshold must use the exact approved normal calibration membership.")
    if len(record["scores"]) != len(expected_ids):
        raise ValueError("Calibration score count mismatch.")
    return calibrate_normal(torch.tensor(record["scores"], dtype=torch.float64))
