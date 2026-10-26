"""Schema v1 types. VisA official fields and internal conventions stay distinct."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from factoryvision.domain.errors import reject
from factoryvision.domain.paths import check_relative_path

VISA_VERSION = "VisA_20220922"
VISA_REVISION = "2a692ab575001cbde74d402d897a7286086c6199"
VISA_SOURCE = (
    f"https://raw.githubusercontent.com/amazon-science/spot-diff/{VISA_REVISION}/split_csv/1cls.csv"
)
VISA_CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)


class SourceKind(StrEnum):
    VISA = "visa"
    SYNTHETIC = "synthetic"


class Label(StrEnum):
    NORMAL = "normal"
    ANOMALY = "anomaly"


class OfficialSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


def require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        reject("TEXT_INVALID", field, "Provide nonempty text without surrounding whitespace.")


def require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", value):
        reject("ID_INVALID", field, "Use 1–128 lowercase ASCII letters/digits, dots, _ or -.")


def require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        reject("HASH_INVALID", field, "Provide a lowercase SHA-256 hex digest or explicit null.")


def require_enum(value: object, enum_type: type[StrEnum], field: str) -> None:
    if not isinstance(value, enum_type):
        reject("ENUM_INVALID", field, f"Use a {enum_type.__name__} value.")


@dataclass(frozen=True, slots=True)
class Source:
    kind: SourceKind
    uri: str
    revision: str
    license_id: str
    attribution: str

    def __post_init__(self) -> None:
        require_enum(self.kind, SourceKind, "source.kind")
        for field in ("uri", "revision", "license_id", "attribution"):
            require_text(getattr(self, field), f"source.{field}")
        if self.kind == SourceKind.VISA:
            if (self.uri, self.revision, self.license_id) != (
                VISA_SOURCE,
                VISA_REVISION,
                "CC-BY-4.0",
            ):
                reject("SOURCE_INVALID", "source", "Use the pinned VisA revision, URI and license.")
        elif (self.uri, self.revision, self.license_id) != (
            "urn:factoryvision:synthetic:metadata-v1",
            "synthetic-v1",
            "LicenseRef-FactoryVision-Synthetic",
        ):
            reject("SOURCE_INVALID", "source", "Use the declared synthetic-v1 provenance tuple.")


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    dataset_id: str
    version: str
    source: Source

    def __post_init__(self) -> None:
        require_identifier(self.dataset_id, "dataset.dataset_id")
        require_text(self.version, "dataset.version")
        if not isinstance(self.source, Source):
            reject("TYPE_INVALID", "dataset.source", "Provide an immutable Source.")
        expected = (
            ("visa", VISA_VERSION)
            if self.source.kind == SourceKind.VISA
            else ("factoryvision-synthetic", "synthetic-v1")
        )
        if (self.dataset_id, self.version) != expected:
            reject("DATASET_SOURCE_MISMATCH", "dataset", "Dataset identity must match its source.")


@dataclass(frozen=True, slots=True)
class Sample:
    sample_id: str
    dataset_id: str
    dataset_version: str
    category: str
    image_path: str
    label: Label
    mask_path: str | None
    official_split: OfficialSplit
    partition: Partition
    group_id: str | None
    content_sha256: str | None

    def __post_init__(self) -> None:
        require_identifier(self.sample_id, "sample.sample_id")
        require_identifier(self.dataset_id, "sample.dataset_id")
        require_text(self.dataset_version, "sample.dataset_version")
        if not isinstance(self.category, str) or self.category not in VISA_CATEGORIES:
            reject("CATEGORY_UNSUPPORTED", "sample.category", "Use an official VisA category ID.")
        require_enum(self.label, Label, "sample.label")
        require_enum(self.official_split, OfficialSplit, "sample.official_split")
        require_enum(self.partition, Partition, "sample.partition")
        check_relative_path(self.image_path, "sample.image_path")
        if self.mask_path is not None:
            check_relative_path(self.mask_path, "sample.mask_path")
            if PurePosixPath(self.mask_path).suffix.lower() != ".png":
                reject(
                    "MASK_FORMAT", "sample.mask_path", "Anomaly mask references must end in .png."
                )
        if (self.label == Label.NORMAL) != (self.mask_path is None):
            reject(
                "MASK_LABEL_MISMATCH",
                "sample.mask_path",
                "Normal records require null; anomaly records require a mask reference.",
            )
        if self.group_id is not None:
            require_identifier(self.group_id, "sample.group_id")
        if self.content_sha256 is not None:
            require_sha256(self.content_sha256, "sample.content_sha256")


@dataclass(frozen=True, slots=True)
class SplitProvenance:
    algorithm: str
    seed: int
    parent_manifest_sha256: str
    assignments_sha256: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256-group-v1":
            reject("SPLIT_ALGORITHM", "split_provenance.algorithm", "Use sha256-group-v1.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            reject("SEED_INVALID", "split_provenance.seed", "Use an integer in [0, 2**32).")
        require_sha256(self.parent_manifest_sha256, "split_provenance.parent_manifest_sha256")
        require_sha256(self.assignments_sha256, "split_provenance.assignments_sha256")


@dataclass(frozen=True, slots=True)
class Manifest:
    schema_version: int
    dataset: DatasetIdentity
    samples: tuple[Sample, ...]
    split_provenance: SplitProvenance | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            reject(
                "SCHEMA_VERSION", "schema_version", "Only integer schema_version 1 is supported."
            )
        if not isinstance(self.dataset, DatasetIdentity):
            reject("TYPE_INVALID", "dataset", "Provide an immutable DatasetIdentity.")
        if not isinstance(self.samples, tuple) or not self.samples:
            reject("SAMPLES_INVALID", "samples", "Provide a nonempty tuple of Sample records.")
        if any(not isinstance(sample, Sample) for sample in self.samples):
            reject("TYPE_INVALID", "samples", "Each record must be an immutable Sample.")
        if self.split_provenance is not None and not isinstance(
            self.split_provenance, SplitProvenance
        ):
            reject("TYPE_INVALID", "split_provenance", "Provide SplitProvenance or explicit null.")
