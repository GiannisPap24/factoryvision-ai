"""Model-free, explicit local integrity evidence for the MetalParts provider."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np

from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.images import decode_image
from factoryvision.data.leakage import fingerprints, screen_pairs
from factoryvision.data.metalparts import REVISION, MetalPartsManifest
from factoryvision.domain.errors import reject
from factoryvision.domain.paths import check_relative_path


def verify_source_files(root: Path, inventory: list[dict], revision: str) -> dict:
    """Verify a saved official tree against local bytes; caller retains tree provenance."""
    if revision != REVISION:
        reject("SOURCE_INVALID", "revision", "Inventory must belong to the pinned revision.")
    expected = set()
    records = {}
    for entry in inventory:
        reference = entry["path"]
        check_relative_path(reference, "inventory.path")
        if not reference.startswith("MetalParts/") or reference.casefold() in expected:
            reject("INVENTORY_INVALID", "path", "Require unique MetalParts-only references.")
        expected.add(reference.casefold())
        content = resolve_reference(root, reference).read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        if "lfs" in entry:
            actual, wanted = sha256, entry["lfs"]["oid"]
        else:
            actual = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            wanted = entry["oid"]
        if len(content) != entry["size"] or actual != wanted:
            reject("SOURCE_HASH_MISMATCH", reference, "File differs from published tree entry.")
        records[reference] = {"bytes": len(content), "sha256": sha256}
    actual_paths = {
        p.relative_to(root).as_posix().casefold()
        for p in (root / "MetalParts").rglob("*")
        if p.is_file()
    }
    if not records or expected != actual_paths:
        reject("INVENTORY_MISMATCH", "MetalParts", "Published and actual file inventories differ.")
    hashes = defaultdict(list)
    for reference, record in records.items():
        hashes[record["sha256"]].append(reference)
    return {
        "files": records,
        "bytes": sum(r["bytes"] for r in records.values()),
        "all_file_byte_duplicate_groups": [
            {"sha256": digest, "paths": paths}
            for digest, paths in sorted(hashes.items())
            if len(paths) > 1
        ],
    }


def _groups(records: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record)
    return [
        {
            "value": value,
            "ids": [r["id"] for r in group],
            "domains": sorted({r["partition"] for r in group}),
            "train_test": "train" in {r["partition"] for r in group}
            and len({r["partition"] for r in group}) > 1,
        }
        for value, group in sorted(groups.items())
        if len(group) > 1
    ]


def audit_metalparts(root: Path, manifest: MetalPartsManifest) -> dict:
    """Native pixels and fixed existing fingerprints only; no learned features or tuning."""
    records, screens = [], []
    dimensions, partitions, mask_modes, mask_values = Counter(), Counter(), Counter(), set()
    for sample in manifest.samples:
        image = decode_image(root, sample.image_path)
        dimensions[f"{image.width}x{image.height}"] += 1
        partitions[f"{sample.source_domain}/{sample.label.value}"] += 1
        pixel_header = f"RGB:{image.width}:{image.height}:".encode()
        record = {
            "id": sample.sample_id,
            "path": sample.image_path,
            "partition": sample.source_domain,
            "basename": PurePosixPath(sample.image_path).name,
            "byte_sha256": hashlib.sha256(
                resolve_reference(root, sample.image_path).read_bytes()
            ).hexdigest(),
            "pixel_sha256": hashlib.sha256(pixel_header + image.tobytes()).hexdigest(),
        }
        records.append(record)
        dhash, phash, rgb = fingerprints(image)
        screens.append(
            {
                "id": sample.sample_id,
                "partition": sample.source_domain,
                "dhash": dhash,
                "phash": phash,
                "rgb": rgb,
            }
        )
        if sample.mask_path:
            mask = decode_image(root, sample.mask_path, mask=True)
            mask_modes[mask.mode] += 1
            mask_values.update(int(x) for x in np.unique(np.asarray(mask)))
    byte_groups = _groups(records, "byte_sha256")
    pixel_groups = _groups(records, "pixel_sha256")
    basename_groups = [g for g in _groups(records, "basename") if len(g["domains"]) > 1]
    near = screen_pairs(screens)
    train_test_near = [c for c in near["candidates"] if "train" in c["partitions"]]
    blockers = []
    if any(g["train_test"] for g in byte_groups + pixel_groups):
        blockers.append("Confirmed cross-train/test image byte or native RGB pixel identity.")
    if train_test_near or any(g["train_test"] for g in basename_groups):
        blockers.append("Unresolved cross-train/test similarity or source-filename correspondence.")
    return {
        "manifest_sha256": manifest.sha256,
        "revision": manifest.source.revision,
        "images": len(records),
        "masks": sum(mask_modes.values()),
        "dimensions": dict(dimensions),
        "partitions": dict(partitions),
        "mask_modes": dict(mask_modes),
        "mask_values": sorted(mask_values),
        "byte_duplicate_groups": byte_groups,
        "pixel_duplicate_groups": pixel_groups,
        "cross_domain_basename_groups": basename_groups,
        "near_duplicate_screen": near,
        "train_test_near_candidates": len(train_test_near),
        "confirmed_related_captures": None,
        "capture_provenance": (
            "No publisher physical-item/group IDs in JSONL; filenames are hints only."
        ),
        "records": records,
        "blockers": blockers,
        "verdict": "BLOCKED — DATASET INTEGRITY OR ACQUISITION ISSUE"
        if blockers
        else "READY FOR METALPARTS TRAINING PLANNING",
        "limitations": (
            "Hash/similarity checks cannot prove physical-item independence. "
            "No model inference, mask-based tuning or statistical independence claim."
        ),
    }


def write_json(path: Path, value: dict) -> None:
    """Refuse overwriting earlier evidence."""
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2)
        target.write("\n")
