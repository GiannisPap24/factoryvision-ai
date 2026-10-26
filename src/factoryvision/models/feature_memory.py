"""Small frozen ResNet18 feature memory; deliberately not a PatchCore reproduction."""

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

from factoryvision.training.checkpoint import checked_directory, file_sha256

WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
RECIPE = "resnet18-layer23-rgb128-imagenet-avg3-rmsnn-max-v1"
DIMENSIONS = 384
MAX_BANK = 8192
PATCHES_PER_IMAGE = 16


def checked_rgb(image):
    if (
        not isinstance(image, torch.Tensor)
        or image.dtype != torch.float32
        or image.device.type != "cpu"
        or image.ndim != 4
        or not 1 <= image.shape[0] <= 16
        or image.shape[1:] != (3, 128, 128)
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
    ):
        raise ValueError("Require finite CPU float32 [B,3,128,128] RGB in [0,1], B<=16.")


class FrozenResNet18(nn.Module):
    """Explicit layer2/3 extraction; injected random backbones are for synthetic tests only."""

    def __init__(self, backbone):
        super().__init__()
        self.stem = nn.Sequential(*list(backbone.children())[:5])
        self.layer2, self.layer3 = backbone.layer2, backbone.layer3
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Parent .train() cannot update BatchNorm statistics or enable gradient fitting.
        return super().train(False)

    @torch.inference_mode()
    def forward(self, image):
        checked_rgb(image)
        low = self.layer2(self.stem((image - self.mean) / self.std))
        high = self.layer3(low)
        low = F.avg_pool2d(low, 3, stride=1, padding=1, count_include_pad=True)
        high = F.avg_pool2d(high, 3, stride=1, padding=1, count_include_pad=True)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((low, high), dim=1)


def load_extractor(path: Path, expected_sha: str = WEIGHTS_SHA256):
    checked_directory(path.parent)
    if (
        path.is_symlink()
        or not path.is_file()
        or not 1 <= path.stat().st_size <= 100_000_000
        or file_sha256(path) != expected_sha
    ):
        raise ValueError("Missing, linked, oversized or corrupt pretrained weights.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    backbone = resnet18(weights=None)  # Never download implicitly.
    expected = backbone.state_dict()
    legacy_keys = {k for k in expected if not k.endswith(".num_batches_tracked")}
    if not isinstance(state, dict) or state.keys() not in (expected.keys(), legacy_keys):
        raise ValueError("ResNet18 state keys mismatch.")
    for name, tensor in state.items():
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected[name].shape
            or tensor.dtype != expected[name].dtype
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError("Invalid pretrained state tensor.")
    # Official V1 predates BatchNorm counters. PyTorch's strict loader supplies zeros;
    # every learned parameter/running mean/variance is still required and checked.
    backbone.load_state_dict(state, strict=True)
    return FrozenResNet18(backbone)


def checked_features(features, *, bank=False):
    limit = MAX_BANK if bank else 4096
    if (
        not isinstance(features, torch.Tensor)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or features.ndim != 2
        or not 1 <= features.shape[0] <= limit
        or features.shape[1] != DIMENSIONS
        or not torch.isfinite(features).all()
        or features.abs().max() > 1e6
    ):
        raise ValueError("Require bounded finite float32 CPU [N,384] features.")


def nearest_distances(query, bank, query_chunk=256, bank_chunk=512):
    """Exact RMS Euclidean 1-NN with two-axis blocks; no full pairwise matrix."""
    checked_features(query)
    checked_features(bank, bank=True)
    if any(type(n) is not int or not 1 <= n <= 512 for n in (query_chunk, bank_chunk)):
        raise ValueError("Distance chunks must be integers in [1,512].")
    result = []
    for q in query.split(query_chunk):
        best = torch.full((len(q),), float("inf"))
        for b in bank.split(bank_chunk):
            squared = q.square().sum(1)[:, None] + b.square().sum(1)[None, :] - 2 * q @ b.T
            best = torch.minimum(best, squared.clamp_min(0).min(1).values)
        result.append((best / DIMENSIONS).sqrt())
    return torch.cat(result)


class FeatureMemory(nn.Module):
    def __init__(self, extractor, bank):
        super().__init__()
        checked_features(bank, bank=True)
        self.extractor = extractor
        self.register_buffer("bank", bank.detach().clone())
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, image):
        checked_rgb(image)
        feature = self.extractor(image)
        if feature.shape != (len(image), DIMENSIONS, 16, 16):
            raise ValueError("Extractor must return [B,384,16,16].")
        query = feature.permute(0, 2, 3, 1).reshape(-1, DIMENSIONS)
        patch = nearest_distances(query, self.bank).reshape(len(image), 16, 16)
        score = patch.flatten(1).max(1).values  # Debug feature scoring here.
        spatial = F.interpolate(patch[:, None], (128, 128), mode="bilinear", align_corners=False)
        return score, spatial[:, 0]


def require_development_batch(batch, partition, expected_ids, offset):
    checked_rgb(batch["image"])
    n = len(batch["image"])
    if (
        partition not in ("train", "validation", "calibration")
        or batch["partition"] != [partition] * n
        or batch["label"].shape != (n,)
        or not (batch["label"] == 0).all()
        or batch["sample_id"] != expected_ids[offset : offset + n]
    ):
        raise ValueError("Require exact approved normal development membership and order.")


def checked_ids(ids):
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(s, str) and s for s in ids)
        or ids != sorted(set(ids))
    ):
        raise ValueError("Require sorted unique nonempty sample identities.")


@torch.inference_mode()
def fit_memory(extractor, loader, expected_ids):
    checked_ids(expected_ids)
    if len(expected_ids) * PATCHES_PER_IMAGE > MAX_BANK:
        raise ValueError("Declared training membership exceeds the memory-bank cap.")
    extractor.eval()
    chunks, selection, offset = [], [], 0
    for batch in loader:
        require_development_batch(batch, "train", expected_ids, offset)
        features = extractor(batch["image"])
        if features.shape != (len(batch["image"]), DIMENSIONS, 16, 16):
            raise ValueError("Invalid fitting feature geometry.")
        for sample_id, feature in zip(batch["sample_id"], features, strict=True):
            patches = feature.permute(1, 2, 0).reshape(256, DIMENSIONS)
            checked_features(patches)
            seed = int.from_bytes(hashlib.sha256(f"42|{sample_id}".encode()).digest()[:8], "big")
            indices = (
                torch.randperm(256, generator=torch.Generator().manual_seed(seed))[
                    :PATCHES_PER_IMAGE
                ]
                .sort()
                .values
            )
            chunks.append(patches[indices].clone())
            selection.append({"sample_id": sample_id, "patch_indices": indices.tolist()})
        offset += len(batch["image"])
    if offset != len(expected_ids):
        raise ValueError("Incomplete fitting membership.")
    return FeatureMemory(extractor, torch.cat(chunks)), selection
