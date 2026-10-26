"""Separate MetalParts bank identity; never accept Cashew examples or arbitrary weights."""

import json

import torch

from factoryvision.data.metalparts import MetalPartsSource
from factoryvision.models.feature_memory import WEIGHTS_SHA256, checked_features, checked_ids
from factoryvision.models.metalparts_memory import PER_TILE, RECIPE
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    file_sha256,
    write_json,
)


def identity(source, split_sha, protocol_sha, tile_ids):
    checked_ids(tile_ids)
    if not isinstance(source, MetalPartsSource):
        raise ValueError("Require MetalParts source identity.")
    if any(
        len(s) != 64 or any(c not in "0123456789abcdef" for c in s)
        for s in (split_sha, protocol_sha)
    ):
        raise ValueError("Invalid split/protocol SHA-256.")
    if len(tile_ids) * PER_TILE > 8192:
        raise ValueError("Memory size exceeds the frozen cap.")
    return {
        "schema": 1,
        "recipe": RECIPE,
        "dataset": source.dataset_id,
        "category": "metalparts",
        "revision": source.revision,
        "weights_sha256": WEIGHTS_SHA256,
        "split_sha256": split_sha,
        "protocol_sha256": protocol_sha,
        "fit_tile_ids": tile_ids,
        "descriptors_per_tile": PER_TILE,
        "threshold": None,
    }


def validate_identity(value):
    expected = identity(
        MetalPartsSource(value["revision"]),
        value["split_sha256"],
        value["protocol_sha256"],
        value["fit_tile_ids"],
    )
    if value != expected:
        raise ValueError("Incompatible MetalParts memory category/recipe/provenance.")


def save_memory(directory, bank, expected):
    validate_identity(expected)
    checked_directory(directory)
    checked_features(bank, bank=True)
    if len(bank) != len(expected["fit_tile_ids"]) * PER_TILE:
        raise ValueError("Memory membership/count mismatch.")
    if any((directory / name).exists() for name in ("memory.pt", "memory.json")):
        raise FileExistsError("Refuse to replace an existing MetalParts memory.")
    atomic_write(
        directory, "memory.pt", lambda stream: torch.save(bank.detach().cpu().clone(), stream)
    )
    meta = {
        "identity": expected,
        "memory_sha256": file_sha256(directory / "memory.pt"),
        "shape": list(bank.shape),
    }
    write_json(directory, "memory.json", meta)
    return meta


def load_memory(directory, expected):
    validate_identity(expected)
    checked_directory(directory)
    for name, limit in [("memory.json", 512 * 1024), ("memory.pt", 16 * 1024**2)]:
        path = directory / name
        if (
            not path.is_file()
            or path.is_symlink()
            or path.is_junction()
            or path.stat().st_size > limit
        ):
            raise ValueError("Missing, linked or oversized MetalParts memory artifact.")
    meta = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
    if meta.get("identity") != expected or meta.get("memory_sha256") != file_sha256(
        directory / "memory.pt"
    ):
        raise ValueError("MetalParts memory identity/hash mismatch.")
    bank = torch.load(directory / "memory.pt", map_location="cpu", weights_only=True)
    checked_features(bank, bank=True)
    if (
        list(bank.shape) != meta.get("shape")
        or len(bank) != len(expected["fit_tile_ids"]) * PER_TILE
    ):
        raise ValueError("MetalParts memory dimensions/membership mismatch.")
    return bank, meta
