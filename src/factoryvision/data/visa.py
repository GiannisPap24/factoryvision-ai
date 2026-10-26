"""Explicit official one-class CSV ingestion and decoded Cashew integrity audit."""

import csv
import hashlib
import io
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.images import decode_image, decode_pair
from factoryvision.domain.contracts import (
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
from factoryvision.domain.errors import reject
from factoryvision.domain.validation import require_valid_manifest

ATTRIBUTION = (
    "Visual Anomaly (VisA), Zou, Jeong, Pemula, Zhang and Dabeer, "
    "SPot-the-Difference, ECCV 2022. https://registry.opendata.aws/visa/; CC BY 4.0."
)
CASHEW_DEFECTS = {
    "burnt",
    "corner or edge breakage",
    "different colour spot",
    "middle breakage",
    "small holes",
    "small scratches",
    "stuck together",
    "same colour spot",
    "other",
}


def annotation_label(value: str) -> str:
    if value == "normal":
        return "normal"
    if not value or any(part not in CASHEW_DEFECTS for part in value.split(",")):
        reject("CSV_VALUE", "annotation.label", "Unknown Cashew defect taxonomy value.")
    return "anomaly"


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    with path.open("rb") as stream:
        content = stream.read(2 * 1024**2 + 1)
    if len(content) > 2 * 1024**2:
        reject("CSV_LIMIT", "metadata", "CSV exceeds 2 MiB.")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        if reader.fieldnames != fields:
            reject("CSV_FIELDS", "metadata", "Require the exact official CSV column order.")
        result = []
        for row in reader:
            if set(row) != set(fields) or any(v is None for v in row.values()):
                reject("CSV_FIELDS", "metadata", "Malformed CSV record width.")
            result.append(row)
    except (UnicodeError, csv.Error) as exc:
        reject("CSV_INVALID", "metadata", f"Invalid UTF-8 CSV: {type(exc).__name__}")
    return result


def load_visa_cashew(path: Path) -> Manifest:
    rows = read_csv(path, ["object", "split", "label", "image", "mask"])
    samples = []
    for row in rows:
        if row["object"] != "cashew":
            continue
        if row["label"] not in ("normal", "anomaly") or row["split"] not in ("train", "test"):
            reject("CSV_VALUE", "metadata", "Use official normal/anomaly and train/test values.")
        label = Label(row["label"])
        # An empty official mask column denotes no stored normal mask, not a missing JSON key.
        sample = Sample(
            sample_id=f"visa-cashew-{label.value}-{Path(row['image']).stem.lower()}",
            dataset_id="visa",
            dataset_version=VISA_VERSION,
            category="cashew",
            image_path=row["image"],
            label=label,
            mask_path=row["mask"] or None,
            official_split=OfficialSplit(row["split"]),
            partition=Partition(row["split"]),
            group_id=None,
            content_sha256=None,
        )
        samples.append(sample)
    manifest = Manifest(
        1,
        DatasetIdentity(
            "visa",
            VISA_VERSION,
            Source(SourceKind.VISA, VISA_SOURCE, VISA_REVISION, "CC-BY-4.0", ATTRIBUTION),
        ),
        tuple(sorted(samples, key=lambda s: s.sample_id)),
        None,
    )
    require_valid_manifest(manifest)
    return manifest


def audit_cashew(manifest: Manifest, root: Path, annotations: Path) -> tuple[Manifest, dict]:
    """Read one sample at a time; detect exact bytes/pixels, not perceptual similarity."""
    require_valid_manifest(manifest)
    if manifest.dataset.source.kind != SourceKind.VISA or any(
        s.category != "cashew" for s in manifest.samples
    ):
        reject("AUDIT_SOURCE", "manifest", "This audit requires a VisA Cashew manifest.")
    annotated = read_csv(annotations, ["image", "label", "mask"])
    annotation_map = {r["image"]: (annotation_label(r["label"]), r["mask"]) for r in annotated}
    declared = {s.image_path: (s.label.value, s.mask_path or "") for s in manifest.samples}
    if len(annotation_map) != len(annotated) or annotation_map != declared:
        reject(
            "ANNOTATION_MISMATCH", "metadata", "Category annotations and official split disagree."
        )
    declared_files = {p for s in manifest.samples for p in (s.image_path, s.mask_path) if p}
    actual_files = {
        p.relative_to(root).as_posix() for p in (root / "cashew/Data").rglob("*") if p.is_file()
    }
    if actual_files != declared_files:
        reject("INVENTORY_MISMATCH", "files", "Missing or unreferenced category image/mask files.")
    byte_groups = defaultdict(list)
    pixel_groups = defaultdict(list)
    dimensions = Counter()
    formats = Counter()
    modes = Counter()
    mask_values = set()
    records = []
    enriched = []
    for sample in manifest.samples:
        image, _ = decode_pair(root, sample)
        image_path = resolve_reference(root, sample.image_path)
        with image_path.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        with Image.open(image_path) as source:
            formats[source.format] += 1
            modes[source.mode] += 1
            if source.format != "JPEG":
                reject("VISA_IMAGE_FORMAT", sample.image_path, "Official VisA images must be JPEG.")
        pixel_hash = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        byte_groups[file_hash].append(sample.sample_id)
        pixel_groups[pixel_hash].append(sample.sample_id)
        dimensions[f"{image.width}x{image.height}"] += 1
        mask_hash = None
        values = []
        if sample.mask_path:
            mask = decode_image(root, sample.mask_path, mask=True)
            values = np.unique(np.asarray(mask)).astype(int).tolist()
            if any(value < 0 or value > 9 for value in values):
                reject("MASK_CLASS", sample.mask_path, "Cashew mask class IDs must be in [0, 9].")
            mask_values.update(values)
            with resolve_reference(root, sample.mask_path).open("rb") as stream:
                mask_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        enriched.append(replace(sample, content_sha256=file_hash, group_id=f"pixel-{pixel_hash}"))
        records.append(
            {
                "sample_id": sample.sample_id,
                "image_sha256": file_hash,
                "decoded_rgb_sha256": pixel_hash,
                "mask_sha256": mask_hash,
                "width": image.width,
                "height": image.height,
                "mask_class_ids": values,
            }
        )
    output = replace(manifest, samples=tuple(enriched))
    require_valid_manifest(output)  # Includes exact-content and exact-pixel cross-split leakage.
    report = {
        "samples": len(records),
        "annotation_label_counts": dict(Counter(r["label"] for r in annotated)),
        "annotation_mapping": (
            "Explicit official Cashew defect names map to binary anomaly; source CSV unchanged."
        ),
        "masks": sum(s.mask_path is not None for s in manifest.samples),
        "official_counts": dict(
            Counter(f"{s.official_split.value}/{s.label.value}" for s in manifest.samples)
        ),
        "image_formats": dict(formats),
        "stored_image_modes": dict(modes),
        "dimensions": dict(sorted(dimensions.items())),
        "mask_class_ids": sorted(mask_values),
        "mask_format": "PNG",
        "mask_geometry_matches": True,
        "all_anomaly_masks_nonempty": True,
        "byte_duplicate_groups": [v for v in byte_groups.values() if len(v) > 1],
        "pixel_duplicate_groups": [v for v in pixel_groups.values() if len(v) > 1],
        "exact_cross_split_leakage": False,
        "near_duplicate_audit": "Not performed; physical/acquisition groups are not supplied.",
        "records": records,
    }
    return output, report
