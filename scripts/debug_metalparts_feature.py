"""Frozen MetalParts checkpoint on fabricated pixels; no training or dataset access."""

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from factoryvision.inference.metalparts_feature import MetalPartsFeatureService

torch.set_num_threads(4)
pixels = np.zeros((320, 384, 3), dtype=np.uint8)
pixels[64:256, 96:288] = (180, 190, 200)
buffer = io.BytesIO()
Image.fromarray(pixels).save(buffer, format="PNG")
engine = MetalPartsFeatureService(Path(__file__).resolve().parents[1])
result = engine.inspect(buffer.getvalue())
assert result.spatial.shape == (320, 384)
assert result.threshold is None and result.above_threshold is None
print(f"MetalParts feature synthetic inference: score={result.score:.8f}; no dataset opened")
