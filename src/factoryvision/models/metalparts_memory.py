"""One bounded native-patch MetalParts memory; reuse frozen ResNet18 and exact NN."""

import hashlib
import time

import numpy as np
import torch
from torch.nn import functional as F

from factoryvision.data.metalparts_development import (
    DevelopmentSample,
    NativePatchDataset,
    PatchConfig,
    coordinates,
)
from factoryvision.domain.contracts import Partition
from factoryvision.inference.service import preprocess
from factoryvision.inference.uploads import InputRejected
from factoryvision.models.feature_memory import (
    DIMENSIONS,
    MAX_BANK,
    FeatureMemory,
    checked_features,
    checked_ids,
)

RECIPE = "metalparts-native256-stride192-resnet18-rgb128-layer23-rmsnn-max-v1"
CONFIG = PatchConfig()
PER_TILE = 4


class ResourceBudget:
    def __init__(self, seconds, memory, limit=8 * 1024**3):
        self.deadline = time.perf_counter() + seconds
        self.memory, self.limit = memory, limit

    def __call__(self):
        if time.perf_counter() > self.deadline or self.memory() > self.limit:
            raise TimeoutError("MetalParts feature-memory time or memory budget exceeded.")


class FittingTiles:
    """Only declared source-training fitting normals, never silently filter other partitions."""

    def __init__(self, root, samples):
        if not samples or any(
            not isinstance(s, DevelopmentSample) or s.partition != Partition.TRAIN for s in samples
        ):
            raise ValueError("Memory fitting requires only verified normal fitting parents.")
        self.dataset = NativePatchDataset(root, samples, Partition.TRAIN, CONFIG)
        if len({s.sample.source for s in samples}) != 1:
            raise ValueError("Mixed fitting source identities.")
        self.ids = [
            f"{sample.sample_id}-y{y:05d}-x{x:05d}" for sample, x, y in self.dataset.patches
        ]
        checked_ids(self.ids)
        if len(self.ids) * PER_TILE > MAX_BANK:
            raise ValueError("Fitting membership exceeds the fixed memory budget.")

    def __iter__(self):
        for tile_id, (sample, x, y) in zip(self.ids, self.dataset.patches, strict=True):
            image = self.dataset._image(sample)
            crop = image.crop((x, y, x + CONFIG.size, y + CONFIG.size))
            yield tile_id, sample.sample_id, (x, y), preprocess(crop)


def descriptor_indices(tile_id):
    if not isinstance(tile_id, str) or not tile_id:
        raise ValueError("A stable tile ID is required.")
    seed = int.from_bytes(hashlib.sha256(f"42|{tile_id}".encode()).digest()[:8], "big")
    return (
        torch.randperm(256, generator=torch.Generator().manual_seed(seed))[:PER_TILE].sort().values
    )


@torch.inference_mode()
def fit_native_memory(extractor, tiles, budget):
    if not isinstance(tiles, FittingTiles):
        raise ValueError("Require an explicitly verified fitting-tile source.")
    extractor.eval()
    if any(p.requires_grad for p in extractor.parameters()):
        raise ValueError("Feature extractor must remain frozen.")
    chunks, trace = [], []
    for tile_id, parent_id, xy, image in tiles:
        budget()
        feature = extractor(image)
        if feature.shape != (1, DIMENSIONS, 16, 16):
            raise ValueError("Unexpected MetalParts fitting feature geometry.")
        candidates = feature[0].permute(1, 2, 0).reshape(256, DIMENSIONS)
        checked_features(candidates)
        indices = descriptor_indices(tile_id)
        chunks.append(candidates[indices].clone())
        trace.append(
            {
                "tile_id": tile_id,
                "parent_id": parent_id,
                "xy": list(xy),
                "descriptor_indices": indices.tolist(),
            }
        )
    bank = torch.cat(chunks)
    checked_features(bank, bank=True)
    if len(bank) != len(tiles.ids) * PER_TILE:
        raise ValueError("Incomplete fitting-tile membership.")
    return FeatureMemory(extractor, bank), trace


@torch.inference_mode()
def native_memory_map(model, image, budget=lambda: None):
    """Full tile coverage; native mean distance map, maximum pre-interpolation descriptor score."""
    if image.mode != "RGB" or min(image.size) < 256 or max(image.size) > 4096:
        raise InputRejected("MetalParts requires RGB images from 256 to 4096 pixels per side.")
    total = np.zeros((image.height, image.width), dtype=np.float32)
    count = np.zeros_like(total)
    maximum = 0.0
    model.eval()
    for x, y in coordinates(image.width, image.height, CONFIG, full=True):
        budget()
        tensor = preprocess(image.crop((x, y, x + 256, y + 256)))
        score, low = model(tensor)
        if (
            score.shape != (1,)
            or low.shape != (1, 128, 128)
            or not torch.isfinite(score).all()
            or not torch.isfinite(low).all()
            or score.min() < 0
            or low.min() < 0
        ):
            raise ValueError("Invalid feature-memory score or spatial map.")
        native = F.interpolate(low[:, None], (256, 256), mode="bilinear", align_corners=False)[
            0, 0
        ].numpy()
        total[y : y + 256, x : x + 256] += native
        count[y : y + 256, x : x + 256] += 1
        maximum = max(maximum, float(score[0]))
    if not (count > 0).all():
        raise ValueError("Feature-memory map has uncovered native pixels.")
    return total / count, maximum
