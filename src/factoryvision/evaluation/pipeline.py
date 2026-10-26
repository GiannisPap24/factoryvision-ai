"""Bounded deterministic held-out inference, kept separate from fitting and calibration."""

import numpy as np
import torch

from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.evaluation.reconstruction import reconstruction_errors


def evaluate_batches(
    model, loader, expected_ids: list[str], threshold: float, max_pixels=3_000_000
):
    if not expected_ids or expected_ids != sorted(set(expected_ids)):
        raise ValueError("Expected IDs must be nonempty, unique and ordered.")
    model.eval()
    rows = []
    maps = []
    masks = []
    pixels = 0
    with torch.inference_mode():
        for batch in loader:
            image, mask, label = batch["image"], batch["mask"], batch["label"]
            n = image.shape[0]
            if (
                set(batch["partition"]) != {"test"}
                or len(batch["partition"]) != n
                or len(batch["sample_id"]) != n
                or label.shape != (n,)
                or not ((label == 0) | (label == 1)).all()
                or mask.shape != (n, 1, image.shape[2], image.shape[3])
                or not torch.isfinite(mask).all()
                or not ((mask == 0) | (mask == 1)).all()
                or (mask[label == 0] != 0).any()
            ):
                raise ValueError("Invalid test membership, labels or aligned binary masks.")
            pixels += mask.numel()
            if pixels > max_pixels:
                raise ValueError("Pixel metric memory bound exceeded.")
            reconstruction = model(image)
            scores, spatial = reconstruction_errors(image, reconstruction)  # Debug evaluation here.
            for i, sample_id in enumerate(batch["sample_id"]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "label": int(label[i]),
                        "score": float(scores[i]),
                        "prediction": int(float(scores[i]) > threshold),
                        "positive_mask_pixels": int(mask[i].sum()),
                    }
                )
            maps.append(spatial.cpu().numpy().copy())
            masks.append(mask[:, 0].cpu().numpy().astype(np.uint8))
    if [r["sample_id"] for r in rows] != expected_ids:
        raise ValueError("Observed test identities/order differ from frozen membership.")
    return rows, np.concatenate(maps), np.concatenate(masks)


def summarize(rows, maps, masks, threshold):
    if (
        maps.shape != masks.shape
        or maps.ndim != 3
        or len(rows) != maps.shape[0]
        or not np.isin(masks, (0, 1)).all()
    ):
        raise ValueError("Rows/maps/masks must align exactly.")
    image = binary_metrics([r["label"] for r in rows], [r["score"] for r in rows], threshold)
    if any(r["prediction"] != int(r["score"] > threshold) for r in rows):
        raise ValueError("Saved predictions disagree with frozen threshold.")
    return {
        "image": image,
        "pixel": binary_metrics(masks.ravel(), maps.ravel()),
        "anomaly_masks_empty_after_resize": sum(
            r["label"] == 1 and r["positive_mask_pixels"] == 0 for r in rows
        ),
    }
