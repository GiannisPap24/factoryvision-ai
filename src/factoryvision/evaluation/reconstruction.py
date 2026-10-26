"""Raw squared RGB reconstruction errors; no test metrics or heatmap fitting."""

import torch


def reconstruction_errors(image: torch.Tensor, reconstruction: torch.Tensor):
    if (
        image.shape != reconstruction.shape
        or image.ndim != 4
        or image.shape[1] != 3
        or image.numel() == 0
    ):
        raise ValueError("Matching nonempty RGB batches required.")
    if not torch.isfinite(image).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("Non-finite reconstruction.")
    spatial = (image - reconstruction).square().mean(dim=1)
    return spatial.mean(dim=(1, 2)), spatial


def calibrate_normal(scores: torch.Tensor, quantile: float = 0.95) -> dict:
    if (
        scores.ndim != 1
        or scores.numel() < 2
        or not torch.isfinite(scores).all()
        or (scores < 0).any()
        or type(quantile) not in (float, int)
        or not 0 < quantile <= 1
    ):
        raise ValueError("Need >=2 finite nonnegative normal scores and a valid quantile.")
    values = scores.to(dtype=torch.float64, device="cpu")
    threshold = torch.quantile(values, quantile, interpolation="linear").item()
    return {
        "quantile": quantile,
        "interpolation": "linear",
        "comparison": "score > threshold",
        "threshold": threshold,
        "count": values.numel(),
        "exceedances": int((values > threshold).sum()),
        "scope": "normal image-level calibration only; no pixel threshold or recall estimate",
    }
