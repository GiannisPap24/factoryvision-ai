"""Finite checked normal-only CPU updates and sample-weighted validation."""

import hashlib
import random
import time

import numpy as np
import torch

from factoryvision.evaluation.reconstruction import reconstruction_errors


def seed_everything(seed: int, threads: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def parameter_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require_normal_batch(batch: dict, partition: str):
    if (
        partition not in ("train", "validation", "calibration")
        or set(batch["partition"]) != {partition}
        or batch["label"].shape != (batch["image"].shape[0],)
        or not (batch["label"] == 0).all()
    ):
        raise ValueError("Only the explicit normal development partition is permitted.")


def train_step(model, optimizer, batch: dict) -> float:
    require_normal_batch(batch, "train")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    reconstruction = model(batch["image"])
    errors, _ = reconstruction_errors(batch["image"], reconstruction)
    loss = errors.mean()
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss.")
    loss.backward()  # Debug breakpoint: inspect a real loss and gradients on the next line.
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("Missing or non-finite gradients.")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Non-finite optimizer result.")
    return loss.item()


def train_epoch(model, optimizer, loader, *, deadline: float) -> dict:
    total, count, steps = 0.0, 0, 0
    ids = []
    for batch in loader:
        if time.perf_counter() > deadline:
            raise TimeoutError("Training time budget exceeded; partial artifacts retained.")
        loss = train_step(model, optimizer, batch)
        size = batch["image"].shape[0]
        total += loss * size
        count += size
        steps += 1
        ids.extend(batch["sample_id"])
    if not count or len(set(ids)) != count:
        raise ValueError("Training epoch must contain nonempty unique sample IDs.")
    return {"loss": total / count, "samples": count, "steps": steps, "sample_ids": ids}


def score_normals(model, loader, partition: str) -> dict:
    model.eval()
    ids, scores = [], []
    with torch.inference_mode():
        for batch in loader:
            require_normal_batch(batch, partition)
            error, _ = reconstruction_errors(batch["image"], model(batch["image"]))
            ids.extend(batch["sample_id"])
            scores.extend(error.tolist())
    if not ids or ids != sorted(set(ids)):
        raise ValueError("Scoring requires nonempty unique sorted sample IDs.")
    return {"sample_ids": ids, "scores": scores, "mean": sum(scores) / len(scores)}


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.stale = float("inf"), 0, 0

    def update(self, loss: float, epoch: int) -> tuple[bool, bool]:
        if not np.isfinite(loss):
            raise ValueError("Non-finite validation loss.")
        # Save every strict best; min_delta controls stopping, not checkpoint ranking.
        improved = loss < self.best
        significant = loss < self.best - self.min_delta
        self.stale = 0 if significant else self.stale + 1
        if improved:
            self.best, self.best_epoch = loss, epoch
        return improved, self.stale >= self.patience
