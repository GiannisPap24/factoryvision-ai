"""Explicit synthetic batch against frozen local artifacts; no datasets or writes."""

import io
from pathlib import Path

import torch
from PIL import Image

from factoryvision.inference.batch import BatchService, prepare
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.service import InspectionService

torch.set_num_threads(4)
root = Path(__file__).resolve().parents[1]
buffer = io.BytesIO()
Image.new("RGB", (384, 320), (100, 80, 40)).save(buffer, format="PNG")
engine = BatchService(
    InspectionService(root), MetalPartsService(root), MetalPartsFeatureService(root)
)
plan = prepare("metalparts", "metalparts_feature_memory", [buffer.getvalue()] * 2)
report = engine.run(plan)
assert report.count("inspected") == 2
assert all(row.threshold is None for row in report.rows)
print(f"Synthetic batch verified: {len(report.rows)} images; {report.elapsed_ms:.2f} ms")
