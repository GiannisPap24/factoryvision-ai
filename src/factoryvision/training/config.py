"""Strict, bounded experiment configuration without machine-specific paths."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    resolution: int = 128
    batch_size: int = 16
    learning_rate: float = 0.001
    seed: int = 42
    max_epochs: int = 12
    patience: int = 3
    min_delta: float = 0.00001
    threads: int = 4
    max_seconds: int = 1800
    calibration_quantile: float = 0.95

    def __post_init__(self):
        bounds = {
            "resolution": (32, 256),
            "batch_size": (1, 32),
            "seed": (0, 2**32 - 1),
            "max_epochs": (1, 12),
            "patience": (1, 12),
            "threads": (1, 8),
            "max_seconds": (1, 3600),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid {name}.")
        if self.resolution % 16:
            raise ValueError("Resolution must be divisible by 16.")
        for name, low, high in (
            ("learning_rate", 0, 0.1),
            ("min_delta", 0, 0.01),
            ("calibration_quantile", 0, 1),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or not low <= value <= high
                or (name != "min_delta" and value == 0)
            ):
                raise ValueError(f"Invalid {name}.")

    def to_dict(self) -> dict:
        return asdict(self)
