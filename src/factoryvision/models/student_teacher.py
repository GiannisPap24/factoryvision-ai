"""Compact feature distillation; EfficientAD-inspired, not an algorithm reproduction."""

import torch
from torch import nn
from torch.nn import functional as F

from factoryvision.models.feature_memory import checked_rgb

ARCHITECTURE = "compact-st-layer23-unitl2-v1"
EPSILON = 1e-6


class SpatialTeacher(nn.Module):
    """Reuse verified ResNet18 modules; expose raw layer2/layer3 features."""

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, image):
        checked_rgb(image)
        e = self.extractor
        low = e.layer2(e.stem((image - e.mean) / e.std))
        return low, e.layer3(low)


class CompactStudent(nn.Module):
    """Four stride-two convolutions, two linear 1x1 heads; no BatchNorm/dropout."""

    def __init__(self, seed=42):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Student seed must be an integer in [0,2**32).")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1),
                nn.ReLU(),
            )
            self.high = nn.Sequential(nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU())
            self.low_head = nn.Conv2d(64, 128, 1)
            self.high_head = nn.Conv2d(96, 256, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image):
        checked_rgb(image)
        low = self.stem((image - self.mean) / self.std)
        return self.low_head(low), self.high_head(self.high(low))


def discrepancy(teacher, student):
    """Mean of channel-summed squared unit-vector errors, aligned to the 16x16 grid."""
    if not isinstance(teacher, tuple) or not isinstance(student, tuple):
        raise ValueError("Require two feature tuples.")
    if len(teacher) != 2 or len(student) != 2:
        raise ValueError("Require exactly layer2 and layer3 features.")
    maps = []
    batch = None
    for target, prediction, shape in zip(
        teacher, student, ((128, 16, 16), (256, 8, 8)), strict=True
    ):
        for tensor in (target, prediction):
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tensor.shape[1:] != shape
                or not 1 <= tensor.shape[0] <= 16
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError("Invalid finite CPU feature geometry/type.")
        batch = target.shape[0] if batch is None else batch
        if target.shape != prediction.shape or target.shape[0] != batch:
            raise ValueError("Feature batch sizes differ.")
        delta = F.normalize(target.detach(), dim=1, eps=EPSILON) - F.normalize(
            prediction, dim=1, eps=EPSILON
        )
        local = delta.square().sum(1, keepdim=True)
        maps.append(F.interpolate(local, (16, 16), mode="bilinear", align_corners=False))
    return (maps[0] + maps[1])[:, 0] / 2


class StudentTeacher(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher, self.student = teacher, student
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, image):
        checked_rgb(image)
        with torch.no_grad():
            target = self.teacher(image)
        return discrepancy(target, self.student(image))

    @torch.inference_mode()
    def score(self, image):
        self.eval()
        patch = self(image)
        return patch.flatten(1).max(1).values, F.interpolate(
            patch[:, None], (128, 128), mode="bilinear", align_corners=False
        )[:, 0]
