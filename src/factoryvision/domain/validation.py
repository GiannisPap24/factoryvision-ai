"""Pure manifest-level checks; no file reads, downloads or metadata repairs."""

import hashlib
import json
from collections import defaultdict
from pathlib import PurePosixPath

from factoryvision.domain.contracts import Label, Manifest, OfficialSplit, Partition, SourceKind
from factoryvision.domain.errors import ContractError, Issue


def assignments_sha256(manifest: Manifest) -> str:
    """Hash all identity-to-partition assignments in canonical order."""
    assignments = sorted((sample.sample_id, sample.partition.value) for sample in manifest.samples)
    payload = json.dumps(assignments, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_manifest(manifest: Manifest) -> tuple[Issue, ...]:
    """Return sorted, actionable violations without changing the immutable input."""
    issues: set[Issue] = set()
    ids: dict[str, list] = defaultdict(list)
    paths: dict[str, list] = defaultdict(list)
    groups: dict[str, list] = defaultdict(list)
    hashes: dict[str, list] = defaultdict(list)

    def add(code: str, field: str, message: str) -> None:
        issues.add(Issue(code, field, message))

    for sample in sorted(manifest.samples, key=lambda item: (item.sample_id, item.image_path)):
        field = f"samples.{sample.sample_id}"
        ids[sample.sample_id].append(sample)
        paths[sample.image_path.casefold()].append(sample)
        if sample.mask_path is not None:
            paths[sample.mask_path.casefold()].append(sample)
        if sample.group_id is not None:
            groups[sample.group_id].append(sample)
        if sample.content_sha256 is not None:
            hashes[sample.content_sha256].append(sample)
        if (sample.dataset_id, sample.dataset_version) != (
            manifest.dataset.dataset_id,
            manifest.dataset.version,
        ):
            add(
                "SAMPLE_DATASET_MISMATCH", field, "Match the manifest dataset identity and version."
            )
        if (sample.official_split == OfficialSplit.TEST) != (sample.partition == Partition.TEST):
            add("OFFICIAL_TEST_CHANGED", field, "Official test membership must remain unchanged.")
        if sample.official_split == OfficialSplit.TRAIN and sample.label != Label.NORMAL:
            add("ONE_CLASS_LABEL", field, "Official one-class training records must be normal.")
        if sample.partition != Partition.TEST and sample.label != Label.NORMAL:
            add(
                "DEVELOPMENT_ANOMALY", field, "Fit, validation and calibration accept normals only."
            )
        if manifest.dataset.source.kind == SourceKind.VISA:
            expected_image = (
                sample.category,
                "Data",
                "Images",
                "Normal" if sample.label == Label.NORMAL else "Anomaly",
            )
            image = PurePosixPath(sample.image_path)
            if image.parts[:-1] != expected_image or image.suffix.lower() != ".jpg":
                add(
                    "VISA_IMAGE_LAYOUT", field, "Match the category/Data/Images/label/*.JPG layout."
                )
            if sample.mask_path is not None:
                mask = PurePosixPath(sample.mask_path)
                if mask.parts[:-1] != (sample.category, "Data", "Masks", "Anomaly"):
                    add("VISA_MASK_LAYOUT", field, "Match category/Data/Masks/Anomaly/*.png.")
    for sample_id, members in ids.items():
        if len(members) > 1:
            add(
                "DUPLICATE_ID", f"samples.{sample_id}", "Use one record per stable sample identity."
            )
            if len({member.partition for member in members}) > 1:
                add(
                    "IDENTITY_LEAKAGE", f"samples.{sample_id}", "Identity occurs across partitions."
                )
    for path, members in paths.items():
        if len(members) > 1:
            add(
                "DUPLICATE_REFERENCE",
                f"references.{path}",
                "Image/mask references must be unique, including Windows case aliases.",
            )
            if len({member.partition for member in members}) > 1:
                add("REFERENCE_LEAKAGE", f"references.{path}", "File reference crosses partitions.")
    for group_id, members in groups.items():
        if len({member.partition for member in members}) > 1:
            add("GROUP_LEAKAGE", f"groups.{group_id}", "Keep the entire group in one partition.")
        if len({member.category for member in members}) > 1:
            add(
                "GROUP_CATEGORY_MISMATCH",
                f"groups.{group_id}",
                "Group IDs are dataset-wide and must belong to one category.",
            )
    for digest, members in hashes.items():
        if len({member.partition for member in members}) > 1:
            add(
                "CONTENT_LEAKAGE",
                f"content.{digest}",
                "Known identical content crosses partitions.",
            )
    derived = any(
        sample.partition in (Partition.VALIDATION, Partition.CALIBRATION)
        for sample in manifest.samples
    )
    if derived and manifest.split_provenance is None:
        add(
            "SPLIT_PROVENANCE_REQUIRED",
            "split_provenance",
            "Derived development partitions require recorded split provenance.",
        )
    if (
        manifest.split_provenance is not None
        and manifest.split_provenance.assignments_sha256 != assignments_sha256(manifest)
    ):
        add(
            "SPLIT_HASH_MISMATCH",
            "split_provenance.assignments_sha256",
            "Assignment hash does not match this manifest.",
        )
    return tuple(sorted(issues))


def require_valid_manifest(manifest: Manifest) -> None:
    if issues := validate_manifest(manifest):
        raise ContractError(issues)
