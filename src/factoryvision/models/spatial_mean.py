"""Spatial mean RGB reference fitted exclusively to ordered normal training images."""

import torch
from torch import nn

from factoryvision.training.engine import require_normal_batch


class SpatialMean(nn.Module):
    def __init__(self, reference: torch.Tensor):
        super().__init__()
        if (
            reference.ndim != 3
            or reference.shape[0] != 3
            or reference.numel() == 0
            or reference.dtype != torch.float32
            or reference.device.type != "cpu"
            or not torch.isfinite(reference).all()
            or reference.min() < 0
            or reference.max() > 1
        ):
            raise ValueError("Reference must be finite float32 CPU RGB [3,H,W] in [0,1].")
        self.register_buffer("reference", reference.detach().clone())

    def forward(self, image):
        if (
            image.ndim != 4
            or tuple(image.shape[1:]) != tuple(self.reference.shape)
            or image.shape[0] == 0
            or image.dtype != torch.float32
            or image.device.type != "cpu"
            or not torch.isfinite(image).all()
            or image.min() < 0
            or image.max() > 1
        ):
            raise ValueError(
                "Input must match reference RGB geometry, dtype, range and CPU device."
            )
        return self.reference.unsqueeze(0).expand_as(image)


def fit_spatial_mean(loader):
    total = None
    ids = []
    count = 0
    for batch in loader:
        require_normal_batch(batch, "train")
        for image in batch["image"]:
            SpatialMean(image)  # Validate each fitting image before accumulation.
            if total is not None and image.shape != total.shape:
                raise ValueError("Training images must share one geometry.")
            total = image.double() if total is None else total + image.double()
            count += 1
        ids.extend(batch["sample_id"])
    if not ids or ids != sorted(set(ids)) or len(ids) != count:
        raise ValueError("Fitting requires ordered unique nonempty normal training IDs.")
    return SpatialMean((total / len(ids)).float()), ids
