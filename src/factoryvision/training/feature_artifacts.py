"""Hash-bound, weights-only memory artifacts; backbone weights remain a separate input."""

import json
import re

import torch

from factoryvision.models.feature_memory import (
    RECIPE,
    WEIGHTS_SHA256,
    checked_features,
    checked_ids,
)
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    file_sha256,
    write_json,
)


def save_memory(directory, bank, fit_ids, manifest_sha, recipe_sha):
    checked_directory(directory)
    checked_features(bank, bank=True)
    checked_ids(fit_ids)
    if bank.shape[0] != len(fit_ids) * 16:
        raise ValueError("Memory count differs from 16 patches per training image.")
    if not all(re.fullmatch(r"[0-9a-f]{64}", x) for x in (manifest_sha, recipe_sha)):
        raise ValueError("Invalid provenance hashes.")
    if any((directory / n).exists() for n in ("memory.pt", "memory.json")):
        raise ValueError("Refuse to overwrite an existing feature memory.")
    atomic_write(directory, "memory.pt", lambda f: torch.save(bank.detach().cpu().clone(), f))
    metadata = {
        "schema": 1,
        "recipe": RECIPE,
        "recipe_sha256": recipe_sha,
        "manifest_sha256": manifest_sha,
        "weights_sha256": WEIGHTS_SHA256,
        "fit_ids": fit_ids,
        "patch_count": len(bank),
        "memory_sha256": file_sha256(directory / "memory.pt"),
    }
    write_json(directory, "memory.json", metadata)
    return metadata


def load_memory(directory, manifest_sha, recipe_sha, fit_ids):
    checked_directory(directory)
    for name, limit in (("memory.json", 262144), ("memory.pt", 16 * 1024**2)):
        p = directory / name
        if p.is_symlink() or not p.is_file() or p.stat().st_size > limit:
            raise ValueError("Missing, linked or oversized feature artifact.")
    meta = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
    expected = {
        "schema": 1,
        "recipe": RECIPE,
        "recipe_sha256": recipe_sha,
        "manifest_sha256": manifest_sha,
        "weights_sha256": WEIGHTS_SHA256,
        "fit_ids": fit_ids,
        "memory_sha256": file_sha256(directory / "memory.pt"),
    }
    checked_ids(fit_ids)
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Feature artifact provenance/hash mismatch.")
    bank = torch.load(directory / "memory.pt", map_location="cpu", weights_only=True)
    checked_features(bank, bank=True)
    if len(bank) != len(fit_ids) * 16 or meta.get("patch_count") != len(bank):
        raise ValueError("Feature memory membership/count mismatch.")
    return bank, meta
