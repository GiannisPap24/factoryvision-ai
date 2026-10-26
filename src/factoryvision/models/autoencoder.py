"""Compact CPU RGB autoencoder; architecture identifier is part of checkpoint compatibility."""

import torch
from torch import nn

ARCHITECTURE = "conv-ae-v1"
PREPROCESSING = "rgb-float01-bilinear-aa-square-v1"


class ConvAutoencoder(nn.Module):
    def __init__(self, resolution: int = 128):
        super().__init__()
        if type(resolution) is not int or not 32 <= resolution <= 256 or resolution % 16:
            raise ValueError("Resolution must be a multiple of 16 in [32,256].")
        self.resolution = resolution
        encoder = []
        for a, b in zip((3, 8, 16, 32), (8, 16, 32, 16), strict=True):
            encoder.extend((nn.Conv2d(a, b, 4, 2, 1), nn.ReLU()))
        decoder = []
        for a, b in zip((16, 32, 16, 8), (32, 16, 8, 3), strict=True):
            decoder.extend(
                (nn.ConvTranspose2d(a, b, 4, 2, 1), nn.Sigmoid() if b == 3 else nn.ReLU())
            )
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if (
            image.ndim != 4
            or image.shape[0] < 1
            or tuple(image.shape[1:]) != (3, self.resolution, self.resolution)
            or image.dtype != torch.float32
            or image.device.type != "cpu"
        ):
            raise ValueError("Expected nonempty float32 CPU [B,3,R,R] input.")
        if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
            raise ValueError("RGB inputs must be finite and in [0,1].")
        return self.decoder(self.encoder(image))
