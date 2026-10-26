"""Category-general official metadata and allowlisted normal extraction; no new image loader."""

import hashlib
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from factoryvision.data.archive import MAX_SELECTED_BYTES, inspect_archive
from factoryvision.data.images import decode_image
from factoryvision.data.visa import ATTRIBUTION, read_csv
from factoryvision.domain.contracts import (
    VISA_CATEGORIES,
    VISA_REVISION,
    VISA_SOURCE,
    VISA_VERSION,
    DatasetIdentity,
    Label,
    Manifest,
    OfficialSplit,
    Partition,
    Sample,
    Source,
    SourceKind,
)
from factoryvision.domain.validation import require_valid_manifest
from factoryvision.training.checkpoint import checked_directory, file_sha256


def load_category(path, category):
    if category not in VISA_CATEGORIES:
        raise ValueError("Unknown VisA category.")
    samples = []
    for row in read_csv(path, ["object", "split", "label", "image", "mask"]):
        if row["object"] != category:
            continue
        label, official = Label(row["label"]), OfficialSplit(row["split"])
        samples.append(
            Sample(
                f"visa-{category}-{label.value}-{Path(row['image']).stem.lower()}",
                "visa",
                VISA_VERSION,
                category,
                row["image"],
                label,
                row["mask"] or None,
                official,
                Partition(official.value),
                None,
                None,
            )
        )
    manifest = Manifest(
        1,
        DatasetIdentity(
            "visa",
            VISA_VERSION,
            Source(
                SourceKind.VISA,
                VISA_SOURCE,
                VISA_REVISION,
                "CC-BY-4.0",
                ATTRIBUTION,
            ),
        ),
        tuple(sorted(samples, key=lambda s: s.sample_id)),
        None,
    )
    require_valid_manifest(manifest)
    return manifest


def normal_members(manifest):
    require_valid_manifest(manifest)
    selected = [s for s in manifest.samples if s.official_split == OfficialSplit.TRAIN]
    if not selected or any(s.label != Label.NORMAL or s.mask_path for s in selected):
        raise ValueError("Require nonempty official normal training membership.")
    return selected


def extract_normals(archive_path, destination, manifest, archive_sha):
    """Hash whole archive and inspect headers; extract payloads of training normals only."""
    checked_directory(destination.parent)
    if file_sha256(archive_path) != archive_sha:
        raise ValueError("Archive identity mismatch.")
    members = inspect_archive(archive_path)
    wanted = {s.image_path for s in normal_members(manifest)}
    selected = [m for m in members if m.name in wanted and m.isfile()]
    if {m.name for m in selected} != wanted or sum(m.size for m in selected) > MAX_SELECTED_BYTES:
        raise ValueError("Missing/oversized normal-only archive selection.")
    destination.mkdir()  # Exclusive: never mix with a prior extraction.
    resolved = destination.resolve()
    with tarfile.open(archive_path, "r:") as archive:
        for member in selected:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(resolved):
                raise ValueError("Extraction path escaped.")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Missing regular payload.")
            with source, target.open("xb") as out:
                shutil.copyfileobj(source, out, length=1024**2)
            if target.stat().st_size != member.size:
                raise ValueError("Truncated normal payload.")
    return {
        "archive_sha256": archive_sha,
        "inspected_members": len(members),
        "selected_files": len(selected),
        "selected_bytes": sum(m.size for m in selected),
        "selected_paths": sorted(wanted),
        "heldout_payloads_extracted": 0,
    }


def audit_normals(manifest, root):
    """Enrich only training normals; retain untouched test metadata byte-for-byte in meaning."""
    normals = normal_members(manifest)
    expected = {s.image_path for s in normals}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != expected:
        raise ValueError("Normal extraction inventory differs or includes unapproved assets.")
    records, enriched, dimensions = [], {}, Counter()
    by_bytes, by_pixels = defaultdict(list), defaultdict(list)
    for s in normals:
        image = decode_image(root, s.image_path)
        byte_sha = file_sha256(root / s.image_path)
        pixel_sha = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        enriched[s.sample_id] = replace(s, content_sha256=byte_sha, group_id=f"pixel-{pixel_sha}")
        by_bytes[byte_sha].append(s.sample_id)
        by_pixels[pixel_sha].append(s.sample_id)
        dimensions[f"{image.width}x{image.height}"] += 1
        records.append(
            {
                "sample_id": s.sample_id,
                "image_path": s.image_path,
                "image_sha256": byte_sha,
                "decoded_rgb_sha256": pixel_sha,
                "width": image.width,
                "height": image.height,
            }
        )
    output = replace(
        manifest, samples=tuple(enriched.get(s.sample_id, s) for s in manifest.samples)
    )
    require_valid_manifest(output)
    return output, {
        "normal_images": len(normals),
        "dimensions": dict(dimensions),
        "records": records,
        "byte_duplicate_groups": [v for v in by_bytes.values() if len(v) > 1],
        "pixel_duplicate_groups": [v for v in by_pixels.values() if len(v) > 1],
        "test_images_decoded": 0,
        "masks_decoded": 0,
    }
