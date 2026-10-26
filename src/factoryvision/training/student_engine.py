"""Bounded student updates with exact development membership checks."""

import math
import time
from dataclasses import dataclass

import torch

from factoryvision.models.feature_memory import checked_ids, require_development_batch


@dataclass(frozen=True)
class StudentConfig:
    seed: int = 42
    batch_size: int = 8
    epochs: int = 12
    patience: int = 3
    learning_rate: float = 0.001
    threads: int = 4
    seconds: int = 600

    def __post_init__(self):
        for value, limit in (
            (self.batch_size, 16),
            (self.epochs, 12),
            (self.patience, 12),
            (self.threads, 4),
            (self.seconds, 1800),
        ):
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError("Invalid bounded student configuration.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid seed.")
        if type(self.learning_rate) is not float or not 0 < self.learning_rate <= 0.01:
            raise ValueError("Invalid learning rate.")


def step(model, optimizer, batch):
    ids = batch["sample_id"]
    require_development_batch(batch, "train", ids, 0)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch["image"]).mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite student loss.")
    loss.backward()
    if any(p.grad is not None for p in model.teacher.parameters()):
        raise ValueError("Teacher received gradients.")
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.student.parameters()):
        raise ValueError("Missing/nonfinite student gradients.")
    optimizer.step()  # Debug: inspect genuine student gradients and update here.
    if any(not torch.isfinite(p).all() for p in model.student.parameters()):
        raise ValueError("Nonfinite student update.")
    return float(loss.detach())


def epoch(model, loader, expected_ids, partition, *, optimizer=None, deadline=math.inf):
    checked_ids(expected_ids)
    if partition not in ("train", "validation") or (optimizer is not None) != (
        partition == "train"
    ):
        raise ValueError("Only training updates or normal validation select checkpoints.")
    model.train(partition == "train")
    seen, total, updates = [], 0.0, 0
    for batch in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("Student time budget exceeded.")
        ids = batch["sample_id"]
        if len(set(ids)) != len(ids) or set(ids) & set(seen) or not set(ids) <= set(expected_ids):
            raise ValueError("Unexpected/repeated development identities.")
        required = ids if partition == "train" else expected_ids
        require_development_batch(
            batch, partition, required, 0 if partition == "train" else len(seen)
        )
        if optimizer is not None:
            loss = step(model, optimizer, batch)
            updates += 1
        else:
            with torch.inference_mode():
                loss = float(model(batch["image"]).mean())
        if not math.isfinite(loss):
            raise ValueError("Nonfinite epoch loss.")
        total += loss * len(ids)
        seen.extend(ids)
    if sorted(seen) != expected_ids:
        raise ValueError("Incomplete development epoch.")
    return {"loss": total / len(seen), "samples": len(seen), "updates": updates, "sample_ids": seen}


@torch.inference_mode()
def score_normals(model, loader, expected_ids, partition):
    checked_ids(expected_ids)
    if partition not in ("validation", "calibration"):
        raise ValueError("Score normal validation/calibration only.")
    model.eval()
    ids, scores = [], []
    for batch in loader:
        require_development_batch(batch, partition, expected_ids, len(ids))
        score, _ = model.score(batch["image"])
        if score.shape != (len(batch["sample_id"]),) or not torch.isfinite(score).all():
            raise ValueError("Invalid student scores.")
        ids.extend(batch["sample_id"])
        scores.extend(score.tolist())
    if ids != expected_ids:
        raise ValueError("Incomplete scoring membership.")
    return {"partition": partition, "sample_ids": ids, "scores": scores}
