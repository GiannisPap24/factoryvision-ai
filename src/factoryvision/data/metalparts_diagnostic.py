"""Explicit source-train real-anomaly diagnostics; never use the all-domain loader."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from factoryvision.data.images import decode_image
from factoryvision.data.metalparts import MetalPartsSample, MetalPartsSource, _unique_object
from factoryvision.data.metalparts_development import TRAIN_METADATA, verified_bytes
from factoryvision.domain.contracts import Label
from factoryvision.domain.paths import check_relative_path

PREFIX = "MetalParts/metal_parts_data_dir_train/"


class DiagnosticReadGuard:
    """Separate 3E allowlist; the stricter normal-only 3D guard remains unchanged."""

    def __init__(self, root, references):
        self.root = Path(root).resolve()
        references = tuple(references)
        for ref in references:
            check_relative_path(ref, "allowlist")
            if not ref.startswith(PREFIX):
                raise ValueError("Only source-training references may be allowlisted.")
        self.allowed = {(self.root / ref).resolve() for ref in references}
        if not self.allowed or any(not p.is_relative_to(self.root) for p in self.allowed):
            raise ValueError("Empty or escaping diagnostic allowlist.")
        self.opened = set()

    def __call__(self, event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(self.root):
                if path not in self.allowed:
                    raise PermissionError("Official test or unapproved source access prohibited.")
                self.opened.add(path.relative_to(self.root).as_posix())


def source_pair(root, sample, verified):
    if sample.source_domain != "train":
        raise ValueError("Source diagnostics prohibit official test images and masks.")
    verified_bytes(root, sample.image_path, verified[sample.image_path])
    image = decode_image(root, sample.image_path)
    if image.size != (sample.width, sample.height):
        raise ValueError("Source image geometry drift.")
    mask = np.zeros((image.height, image.width), dtype=np.uint8)
    if sample.mask_path is not None:
        verified_bytes(root, sample.mask_path, verified[sample.mask_path])
        original = decode_image(root, sample.mask_path, mask=True)
        values = np.asarray(original)
        if original.size != image.size or not np.isin(values, [0, 255]).all() or not values.any():
            raise ValueError("Source anomaly masks must align, be nonempty and contain only 0/255.")
        mask = (values > 0).astype(np.uint8)
    return image, mask


def audit_source(root, verified, development, source=None):
    """Hash/decode source training inventory before scoring; preserve exact 3D assignments."""
    source = source if source is not None else MetalPartsSource()
    raw = verified_bytes(root, TRAIN_METADATA, verified[TRAIN_METADATA])
    rows = [
        json.loads(line, object_pairs_hook=_unique_object) for line in raw.decode().splitlines()
    ]
    old = {r["sample_id"]: r for r in development["records"]}
    if len(old) != len(development["records"]):
        raise ValueError("Duplicate development records.")
    samples, inventory, seen, found_normals = [], [], set(), set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"file_name", "label", "mask"}
            or type(row["label"]) is not int
            or row["label"] not in (0, 1)
        ):
            raise ValueError("Invalid source-training metadata.")
        sample = MetalPartsSample(
            source,
            "train",
            row["file_name"],
            Label.NORMAL if row["label"] == 0 else Label.ANOMALY,
            row["mask"],
            1,
            1,
        )
        if sample.image_path.casefold() in seen:
            raise ValueError("Duplicate source reference.")
        seen.add(sample.image_path.casefold())
        verified_bytes(root, sample.image_path, verified[sample.image_path])
        image = decode_image(root, sample.image_path)
        sample = replace(sample, width=image.width, height=image.height)
        image, mask = source_pair(root, sample, verified)
        pixel = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        partition = "source_anomaly"
        if sample.label == Label.NORMAL:
            prior = old.get(sample.sample_id)
            if (
                prior is None
                or prior["image_path"] != sample.image_path
                or prior["content_sha256"] != verified[sample.image_path]["sha256"]
                or prior["pixel_sha256"] != pixel
                or (prior["width"], prior["height"]) != image.size
                or prior["partition"] not in ("train", "validation")
            ):
                raise ValueError("Normal membership or content differs from the 3D split.")
            found_normals.add(sample.sample_id)
            partition = prior["partition"]
        elif sample.sample_id in old:
            raise ValueError("Anomaly appears in normal training/validation inventory.")
        entry = {
            "sample_id": sample.sample_id,
            "image_path": sample.image_path,
            "file_name": sample.file_name,
            "label": row["label"],
            "mask_path": sample.mask_path,
            "partition": partition,
            "width": image.width,
            "height": image.height,
            "image_sha256": verified[sample.image_path]["sha256"],
            "pixel_sha256": pixel,
            "mask_sha256": verified[sample.mask_path]["sha256"] if sample.mask_path else None,
            "positive_pixels": int(mask.sum()),
            "physical_group": None,
        }
        inventory.append(entry)
        if partition != "train":
            samples.append(sample)
    if found_normals != set(old):
        raise ValueError("Missing normal development members.")
    if {s.sample_id for s in samples if s.label == Label.ANOMALY} != {
        r["sample_id"] for r in development["excluded"]
    }:
        raise ValueError("Source anomalies differ from the 3D excluded inventory.")
    return tuple(sorted(samples, key=lambda s: s.sample_id)), sorted(
        inventory, key=lambda r: r["sample_id"]
    )
