"""Pinned RobustAD source adapter; preserves supervised source rows without fitting.

VisA schema v1 is intentionally unchanged. This provider envelope reuses its labels,
official split enum, errors, path checks and bounded decoder, but is not a training
Manifest. Physical-item IDs are not supplied by the publisher's JSONL metadata.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.images import decode_pair
from factoryvision.domain.contracts import Label, OfficialSplit, require_enum
from factoryvision.domain.errors import reject
from factoryvision.domain.paths import check_relative_path

REVISION = "82c642b2d18be60f2dab104f8dd4dea1f4ccfacb"
REPOSITORY = "https://huggingface.co/datasets/AmazonScience/RobustAD"
DOMAINS = ("train", "test0", "test1", "test2", "test3", "test4", "test5", "test6")
SYNTHETIC_REVISION = "synthetic-metalparts-v1"


@dataclass(frozen=True, slots=True)
class MetalPartsSource:
    revision: str = REVISION

    def __post_init__(self) -> None:
        if self.revision not in (REVISION, SYNTHETIC_REVISION):
            reject("SOURCE_INVALID", "revision", "Use the pinned or explicit synthetic revision.")

    @property
    def dataset_id(self) -> str:
        return "robustad" if self.revision == REVISION else "factoryvision-synthetic-metalparts"

    @property
    def uri(self) -> str:
        return (
            f"{REPOSITORY}/tree/{REVISION}/MetalParts"
            if self.revision == REVISION
            else "urn:factoryvision:synthetic:metalparts-v1"
        )

    @property
    def license_id(self) -> str:
        return "CC-BY-4.0" if self.revision == REVISION else "LicenseRef-FactoryVision-Synthetic"

    @property
    def attribution(self) -> str:
        return (
            "Amazon Science; Latha Pemula, Dongqing Zhang, Onkar Dabeer; Robust AD, CVPRW 2025"
            if self.revision == REVISION
            else "FactoryVision fabricated offline fixture; not RobustAD observations"
        )


@dataclass(frozen=True, slots=True)
class MetalPartsSample:
    source: MetalPartsSource
    source_domain: str
    file_name: str
    label: Label
    mask_path: str | None
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.source, MetalPartsSource):
            reject("TYPE_INVALID", "source", "Use MetalPartsSource.")
        if self.source_domain not in DOMAINS:
            reject("DOMAIN_INVALID", "source_domain", "Preserve an official source domain.")
        require_enum(self.label, Label, "label")
        check_relative_path(self.file_name, "file_name")
        parts = PurePosixPath(self.file_name).parts
        if (
            len(parts) != 2
            or parts[0] != self.label.value
            or PurePosixPath(self.file_name).suffix.lower() not in (".jpg", ".jpeg", ".png")
        ):
            reject("SOURCE_LAYOUT", "file_name", "Require label directory and image filename.")
        if (self.label == Label.NORMAL) != (self.mask_path is None):
            reject("MASK_LABEL_MISMATCH", "mask", "Normal requires null; anomaly requires mask.")
        if self.mask_path is not None:
            check_relative_path(self.mask_path, "mask")
            expected = f"{self.directory}/masks/{PurePosixPath(self.file_name).stem}.png"
            if self.mask_path != expected:
                reject("MASK_REFERENCE", "mask", "Require same source domain and image stem.")
        if any(type(n) is not int or n < 1 for n in (self.width, self.height)):
            reject("IMAGE_DIMENSIONS", "dimensions", "Require positive integer native dimensions.")

    @property
    def directory(self) -> str:
        return f"MetalParts/metal_parts_data_dir_{self.source_domain}"

    @property
    def image_path(self) -> str:
        return f"{self.directory}/{self.file_name}"

    @property
    def metadata_path(self) -> str:
        return f"{self.directory}/metadata.jsonl"

    @property
    def category(self) -> str:
        return "metalparts"

    @property
    def sample_id(self) -> str:
        value = f"{self.source.dataset_id}\n{self.source.revision}\n{self.image_path}"
        return "metalparts-" + hashlib.sha256(value.encode()).hexdigest()

    @property
    def official_split(self) -> OfficialSplit:
        return OfficialSplit.TRAIN if self.source_domain == "train" else OfficialSplit.TEST

    @property
    def group_id(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class MetalPartsManifest:
    source: MetalPartsSource
    samples: tuple[MetalPartsSample, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, MetalPartsSource):
            reject("TYPE_INVALID", "source", "Use MetalPartsSource.")
        if not isinstance(self.samples, tuple) or not self.samples:
            reject("SAMPLES_INVALID", "samples", "Require a nonempty immutable tuple.")
        seen = set()
        for sample in self.samples:
            if not isinstance(sample, MetalPartsSample) or sample.source != self.source:
                reject("DATASET_SOURCE_MISMATCH", "samples", "All rows must share source identity.")
            key = sample.image_path.casefold()
            if key in seen:
                reject("SAMPLE_DUPLICATE", "samples", "Duplicate image reference, ignoring case.")
            seen.add(key)
        if {sample.source_domain for sample in self.samples} != set(DOMAINS):
            reject("DOMAIN_MISSING", "samples", "All eight source domains must remain present.")

    @property
    def training_normals(self) -> tuple[MetalPartsSample, ...]:
        """Eligibility only: does not derive fitting/validation/calibration assignments."""
        return tuple(
            s
            for s in self.samples
            if s.official_split == OfficialSplit.TRAIN and s.label == Label.NORMAL
        )

    def to_dict(self) -> dict:
        return {
            "schema": "factoryvision.robustad-metalparts.source-v1",
            "dataset_id": self.source.dataset_id,
            "category": "metalparts",
            "source": {
                **asdict(self.source),
                "uri": self.source.uri,
                "license_id": self.source.license_id,
                "attribution": self.source.attribution,
            },
            "samples": [
                {
                    **{k: v for k, v in asdict(s).items() if k != "source"},
                    "sample_id": s.sample_id,
                    "image_path": s.image_path,
                    "metadata_path": s.metadata_path,
                    "official_split": s.official_split.value,
                    "group_id": s.group_id,
                }
                for s in sorted(self.samples, key=lambda item: item.sample_id)
            ],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            reject("METADATA_INVALID", key, "Duplicate JSON key.")
        result[key] = value
    return result


DEFAULT_SOURCE = MetalPartsSource()


def load_metalparts(root: Path, source: MetalPartsSource = DEFAULT_SOURCE) -> MetalPartsManifest:
    """Explicit local decode/validation only; no network, transformations or model calls."""
    samples = []
    expected = set()
    for domain in DOMAINS:
        reference = f"MetalParts/metal_parts_data_dir_{domain}/metadata.jsonl"
        expected.add(reference)
        path = resolve_reference(root, reference)
        if path.stat().st_size > 4 * 1024**2:
            reject("METADATA_TOO_LARGE", reference, "JSONL must be at most 4 MiB per domain.")
        try:
            rows = [
                json.loads(line, object_pairs_hook=_unique_object)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        except (UnicodeError, json.JSONDecodeError) as exc:
            reject("METADATA_INVALID", reference, f"Malformed JSONL: {type(exc).__name__}.")
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"file_name", "label", "mask"}:
                reject("METADATA_FIELDS", reference, "Require exactly file_name, label, mask.")
            if type(row["label"]) is not int or row["label"] not in (0, 1):
                reject("LABEL_INVALID", reference, "Require integer 0 or 1, never bool.")
            label = Label.NORMAL if row["label"] == 0 else Label.ANOMALY
            provisional = MetalPartsSample(
                source, domain, row["file_name"], label, row["mask"], 1, 1
            )
            image, _ = decode_pair(root, provisional)
            sample = MetalPartsSample(
                source, domain, row["file_name"], label, row["mask"], image.width, image.height
            )
            samples.append(sample)
            expected.add(sample.image_path)
            if sample.mask_path:
                expected.add(sample.mask_path)
    manifest = MetalPartsManifest(source, tuple(samples))
    actual = {
        p.relative_to(root).as_posix() for p in (root / "MetalParts").rglob("*") if p.is_file()
    }
    if actual != expected:
        reject("INVENTORY_MISMATCH", "MetalParts", "Source files and metadata references differ.")
    return manifest
