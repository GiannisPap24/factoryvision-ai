"""Atomic local state_dict artifacts with explicit compatibility and integrity checks."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import torch

from factoryvision.models.autoencoder import ARCHITECTURE, PREPROCESSING, ConvAutoencoder


def checked_directory(directory: Path):
    for node in (directory, *directory.parents):
        if node.is_symlink() or node.is_junction():
            raise ValueError("Linked artifact directories are forbidden.")
    if not directory.is_dir():
        raise ValueError("Artifact directory must exist.")


def create_run(project: Path, name: str) -> Path:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
        raise ValueError("Use a portable run identifier, not a path.")
    checked_directory(project)
    base = project / "runs"
    base.mkdir(exist_ok=True)
    checked_directory(base)
    target = base / name
    target.mkdir()  # Refuse reuse of a completed or partial experiment.
    return target


def atomic_write(directory: Path, name: str, writer):
    checked_directory(directory)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.(json|pt)", name):
        raise ValueError("Invalid artifact filename.")
    target = directory / name
    if target.is_symlink() or target.is_junction():
        raise ValueError("Linked artifact target forbidden.")
    temporary = directory / f".{name}.{uuid.uuid4().hex}.partial"
    try:
        with temporary.open("xb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(directory: Path, name: str, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(directory, name, lambda stream: stream.write(data))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_checkpoint(directory: Path, model, manifest_hash: str, epoch: int, loss: float):
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
        raise ValueError("Invalid dataset fingerprint.")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    atomic_write(directory, "best.pt", lambda stream: torch.save(state, stream))
    metadata = {
        "schema": 1,
        "architecture": ARCHITECTURE,
        "resolution": model.resolution,
        "preprocessing": PREPROCESSING,
        "manifest_sha256": manifest_hash,
        "epoch": epoch,
        "validation_loss": loss,
        "checkpoint_sha256": file_sha256(directory / "best.pt"),
    }
    write_json(directory, "checkpoint.json", metadata)
    return metadata


def load_checkpoint(directory: Path, manifest_hash: str, resolution: int):
    checked_directory(directory)
    metadata_path, weights = directory / "checkpoint.json", directory / "best.pt"
    for path, limit in ((metadata_path, 65536), (weights, 16 * 1024**2)):
        if (
            path.is_symlink()
            or path.is_junction()
            or not path.is_file()
            or path.stat().st_size > limit
        ):
            raise ValueError("Missing, linked or oversized checkpoint artifact.")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "architecture": ARCHITECTURE,
        "resolution": resolution,
        "preprocessing": PREPROCESSING,
        "manifest_sha256": manifest_hash,
    }
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Checkpoint model/dataset/preprocessing compatibility mismatch.")
    if file_sha256(weights) != meta.get("checkpoint_sha256"):
        raise ValueError("Checkpoint hash mismatch.")
    state = torch.load(weights, map_location="cpu", weights_only=True)
    model = ConvAutoencoder(resolution)
    reference = model.state_dict()
    if not isinstance(state, dict) or state.keys() != reference.keys():
        raise ValueError("Checkpoint state keys mismatch.")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference[key].shape
            or value.dtype != reference[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid checkpoint tensor.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta
