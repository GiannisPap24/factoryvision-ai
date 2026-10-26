"""Model-free, conservative pre-training integrity gate; never repairs a split."""

import hashlib
import io
import json
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from factoryvision.data.archive import inspect_archive
from factoryvision.data.integrity_groups import ImageFingerprint, group_evidence
from factoryvision.data.leakage import fingerprints
from factoryvision.domain.validation import require_valid_manifest


class AuditIncomplete(ValueError):
    """Insufficient technical evidence; cannot pass."""


class AuditDefect(ValueError):
    """Observed material source/prepared-data defect."""


@dataclass(frozen=True)
class Limits:
    images: int = 2000
    pairs: int = 2_000_000
    candidates: int = 250_000
    required_reviews: int = 200
    selected_bytes: int = 512 * 1024**2
    encoded_bytes: int = 16 * 1024**2
    image_pixels: int = 16_000_000
    process_bytes: int = 2 * 1024**3
    seconds: int = 600


DEFAULT_LIMITS = Limits()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_protocol(path, expected_sha):
    if file_hash(path) != expected_sha:
        raise AuditIncomplete("Frozen protocol identity mismatch.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["method"] != "pcb2-integrity-v1":
        raise AuditIncomplete("Unsupported integrity protocol.")
    return protocol


def extract_category_images(archive_path, destination, manifest, archive_sha, access, budget):
    """Validate all headers, copy only declared category images; masks stay in archive."""
    require_valid_manifest(manifest)
    categories = {s.category for s in manifest.samples}
    if len(categories) != 1:
        raise AuditDefect("Require exactly one category.")
    if file_hash(archive_path) != archive_sha:
        raise AuditIncomplete("Source archive identity mismatch.")
    members = inspect_archive(archive_path)
    category = next(iter(categories))
    wanted = {s.image_path for s in manifest.samples}
    declared_masks = {s.mask_path for s in manifest.samples if s.mask_path}
    images = {
        m.name for m in members if m.isfile() and m.name.startswith(f"{category}/Data/Images/")
    }
    masks = {m.name for m in members if m.isfile() and m.name.startswith(f"{category}/Data/Masks/")}
    if images != wanted or masks != declared_masks:
        raise AuditDefect("Missing/extra category image or mask reference in archive headers.")
    selected = sorted((m for m in members if m.name in wanted), key=lambda m: m.name)
    if len(selected) > Limits().images or sum(m.size for m in selected) > Limits().selected_bytes:
        raise AuditIncomplete("Extraction resource limit exceeded.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Dedicated extraction root already exists.")
    if destination.parent.resolve(strict=True) != destination.parent.absolute():
        raise AuditDefect("Extraction parent traverses links.")
    destination.mkdir()
    resolved = destination.resolve(strict=True)
    inventory = []
    with tarfile.open(archive_path, "r:") as archive:
        for member in selected:
            budget()
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(resolved):
                raise AuditDefect("Extraction escaped root.")
            access("archive_image_payload", member.name)
            digest = hashlib.sha256()
            with archive.extractfile(member) as source, target.open("xb") as output:
                for block in iter(lambda: source.read(1024**2), b""):
                    digest.update(block)
                    output.write(block)
            if target.stat().st_size != member.size or file_hash(target) != digest.hexdigest():
                raise AuditDefect("Prepared file differs from source member.")
            inventory.append(
                {"path": member.name, "bytes": member.size, "source_sha256": digest.hexdigest()}
            )
    return {
        "images": inventory,
        "mask_headers": sorted(masks),
        "archive_headers_checked": len(members),
        "mask_payloads_accessed": 0,
    }


def describe_image(sample_id, partition, payload, limits=DEFAULT_LIMITS):
    if len(payload) > limits.encoded_bytes:
        raise AuditIncomplete("Encoded image resource limit.")
    try:
        with Image.open(io.BytesIO(payload)) as original:
            if original.width * original.height > limits.image_pixels:
                raise AuditIncomplete("Decoded image resource limit.")
            if original.format not in {"JPEG", "PNG"} or getattr(original, "n_frames", 1) != 1:
                raise AuditDefect("Unsupported image format or frame count.")
            if original.getexif().get(274, 1) != 1:
                raise AuditDefect("Nontrivial image orientation requires review.")
            image = original.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise AuditDefect("Corrupt or unreadable image.") from exc
    dh, ph, rgb = fingerprints(image)
    pixel_sha = hashlib.sha256(
        f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    record = ImageFingerprint(
        sample_id, partition, hashlib.sha256(payload).hexdigest(), pixel_sha, dh, ph
    )
    full = np.asarray(image.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    box = (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4)
    crop = image.crop(box).resize((64, 64), Image.Resampling.LANCZOS)
    gray = np.asarray(crop.convert("L"), dtype=np.float64).ravel()
    gray -= gray.mean()
    return record, {
        "rgb": rgb,
        "full": full,
        "crop": np.asarray(crop, dtype=np.float32) / 255,
        "gray": gray,
        "width": image.width,
        "height": image.height,
    }


def scan_all_pairs(records, features, budget=lambda: None, limits=DEFAULT_LIMITS):
    """All three relations: train/train, train/test, test/test; labels never enter."""
    ordered = sorted(records, key=lambda r: r.sample_id)
    if len(ordered) > limits.images or len(ordered) * (len(ordered) - 1) // 2 > limits.pairs:
        raise AuditIncomplete("Pair/record resource limit.")
    if len({r.sample_id for r in ordered}) != len(ordered) or set(features) != {
        r.sample_id for r in ordered
    }:
        raise AuditDefect("Duplicate identity or incomplete image features.")
    candidates, comparisons = [], 0
    for i, a in enumerate(ordered):
        af = features[a.sample_id]
        for b in ordered[i + 1 :]:
            if comparisons % 1000 == 0:
                try:
                    budget()
                except AuditIncomplete as exc:
                    return {
                        "complete": False,
                        "reason": str(exc),
                        "comparisons": comparisons,
                        "candidates": candidates,
                        "required": [],
                    }
            comparisons += 1
            bf = features[b.sample_id]
            dh, ph = (a.dhash ^ b.dhash).bit_count(), (a.phash ^ b.phash).bit_count()
            rgb = float(np.sqrt(np.mean((af["rgb"] - bf["rgb"]) ** 2)))
            if dh > 4 and ph > 4 and rgb > 0.02:
                continue
            full = float(np.sqrt(np.mean((af["full"] - bf["full"]) ** 2)))
            crop = float(np.sqrt(np.mean((af["crop"] - bf["crop"]) ** 2)))
            denominator = float(np.linalg.norm(af["gray"]) * np.linalg.norm(bf["gray"]))
            corr = float(np.dot(af["gray"], bf["gray"]) / denominator) if denominator else None
            candidate = {
                "a": a.sample_id,
                "b": b.sample_id,
                "relation": "/".join(sorted((a.partition, b.partition))),
                "dhash_distance": dh,
                "phash_distance": ph,
                "rgb32_rms": rgb,
                "full128_rms": full,
                "crop64_rms": crop,
                "crop_correlation": corr,
                "strong": (full <= 0.015 and crop <= 0.025) or (corr is not None and corr >= 0.995),
                "exact": a.byte_sha256 == b.byte_sha256 or a.pixel_sha256 == b.pixel_sha256,
            }
            candidates.append(candidate)
            if len(candidates) > limits.candidates:
                return {
                    "complete": False,
                    "reason": "candidate cap",
                    "comparisons": comparisons,
                    "candidates": candidates,
                    "required": [],
                }
    required = {}
    by_relation = defaultdict(list)
    for p in candidates:
        by_relation[p["relation"]].append(p)
        if p["strong"] and not p["exact"]:
            required[(p["a"], p["b"])] = p
    for group in by_relation.values():
        for key in (
            lambda p: (p["full128_rms"], p["a"], p["b"]),
            lambda p: (p["crop64_rms"], p["a"], p["b"]),
            lambda p: hashlib.sha256(f"{p['a']}|{p['b']}".encode()).hexdigest(),
        ):
            for p in sorted(group, key=key)[:8]:
                if not p["exact"]:
                    required[(p["a"], p["b"])] = p
    return {
        "complete": True,
        "reason": None,
        "comparisons": comparisons,
        "candidates": candidates,
        "required": [required[k] for k in sorted(required)],
    }


def gate_decision(records, screen, *, decisions=(), source_verified=True, defects=()):
    """No similarity-only confirmation. Unresolved mandatory review cannot pass."""
    required = {(p["a"], p["b"]) for p in screen["required"]}
    identities = {record.sample_id for record in records}
    if any(a == b or a not in identities or b not in identities for a, b in required):
        raise ValueError("Invalid mandatory review identities.")
    supplied, confirmed = {}, []
    for decision in decisions:
        pair = tuple(sorted((decision["a"], decision["b"])))
        if pair not in required or pair in supplied:
            raise ValueError("Unexpected or repeated review decision.")
        if decision["decision"] not in {"confirmed", "rejected", "unresolved"}:
            raise ValueError("Invalid review disposition.")
        if decision["decision"] != "unresolved" and (
            decision.get("basis") not in {"independent_provenance", "authorized_visual_review"}
            or not decision.get("evidence")
        ):
            raise ValueError("Independent evidence required; hash proximity is insufficient.")
        supplied[pair] = decision["decision"]
        if decision["decision"] == "confirmed":
            confirmed.append(pair)
    groups = group_evidence(records, confirmed_pairs=confirmed)
    unresolved = [
        list(p) for p in sorted(required) if supplied.get(p, "unresolved") == "unresolved"
    ]
    if not source_verified:
        verdict, reason = "INCONCLUSIVE", "Source/protocol identity not verified."
    elif defects or groups["cross_partition_groups"]:
        verdict, reason = "BLOCKED", "Material defect or confirmed official train/test overlap."
    elif (
        not records
        or not screen["complete"]
        or screen["comparisons"] != len(records) * (len(records) - 1) // 2
        or len(required) > Limits().required_reviews
        or unresolved
    ):
        verdict, reason = "INCONCLUSIVE", "Incomplete scan or unresolved required duplicate review."
    else:
        verdict, reason = "PASS", "Completed registered checks; no confirmed cross-split overlap."
    return {
        "verdict": verdict,
        "reason": reason,
        "groups": groups,
        "unresolved_required_pairs": unresolved,
        "defects": list(defects),
        "physical_item_independence": "unverified",
        "sample_exclusions": 0,
    }
