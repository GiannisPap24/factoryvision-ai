"""Explicit real frozen model inference on a synthetic image; no dataset reads."""

import io
from pathlib import Path

import torch
from PIL import Image

from factoryvision.inference.service import InspectionService

torch.set_num_threads(4)
buffer = io.BytesIO()
Image.new("RGB", (192, 128), (100, 80, 40)).save(buffer, format="PNG")
result = InspectionService(Path.cwd()).inspect(buffer.getvalue(), "autoencoder")
print(f"Synthetic debug probe: score={result.score:.8f}; map={result.spatial.shape}")
