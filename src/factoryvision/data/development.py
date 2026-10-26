"""Development-only image access; official test records are never decoded by this adapter."""

import hashlib
import os
from pathlib import Path

import torch

from factoryvision.data.images import ImageSettings, ManifestDataset
from factoryvision.domain.contracts import Label, OfficialSplit, Partition


class DevelopmentReadGuard:
    """Python audit hook: allowlisted development paths only inside the real image root."""

    def __init__(self, root, samples):
        self.root = Path(root).resolve()
        if any(
            s.partition == Partition.TEST
            or s.label != Label.NORMAL
            or s.official_split != OfficialSplit.TRAIN
            or s.mask_path
            for s in samples
        ):
            raise ValueError("Read allowlist requires normal development records only.")
        self.allowed = {(self.root / s.image_path).resolve() for s in samples}
        if not self.allowed or any(not p.is_relative_to(self.root) for p in self.allowed):
            raise ValueError("Empty or escaping development read allowlist.")
        self.opened = set()

    def __call__(self, event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(self.root):
                if path not in self.allowed:
                    raise PermissionError("Official test/unapproved dataset file read prohibited.")
                self.opened.add(path.relative_to(self.root).as_posix())


class DevelopmentDataset(ManifestDataset):
    def __init__(self, manifest, root, partition):
        if partition not in (Partition.TRAIN, Partition.VALIDATION, Partition.CALIBRATION):
            raise ValueError("Official test data is prohibited for feature development.")
        super().__init__(manifest, root, partition, ImageSettings(128, 128))
        if any(
            s.label != Label.NORMAL or s.official_split != OfficialSplit.TRAIN or s.mask_path
            for s in self.samples
        ):
            raise ValueError(
                "Development must contain only official-training normals without masks."
            )


def synthetic_defect(image, sample_id):
    """Deterministic 16x16 magenta/green checkerboard on a normal validation image."""
    if (
        image.shape != (3, 128, 128)
        or image.dtype != torch.float32
        or image.device.type != "cpu"
        or not torch.isfinite(image).all()
        or image.min() < 0
        or image.max() > 1
        or not isinstance(sample_id, str)
        or not sample_id
    ):
        raise ValueError("Require finite float32 RGB128 and a source identity.")
    digest = hashlib.sha256(f"pseudo-v1|{sample_id}".encode()).digest()
    y, x = 40 + digest[0] % 33, 40 + digest[1] % 33
    checker = ((torch.arange(16)[:, None] // 4 + torch.arange(16)[None, :] // 4) % 2).float()
    changed = image.clone()
    changed[:, y : y + 16, x : x + 16] = torch.stack((checker, 1 - checker, checker))
    mask = torch.zeros(128, 128, dtype=torch.uint8)
    mask[y : y + 16, x : x + 16] = 1
    return changed, mask
