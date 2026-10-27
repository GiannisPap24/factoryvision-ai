"""Strict, I/O-free metadata for robustness-v1. This module cannot run experiments."""

import json
import math
from dataclasses import asdict, dataclass, fields
from enum import StrEnum

PROTOCOL_ID = "factoryvision-synthetic-robustness"
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_METADATA_BYTES = 65536


class RobustnessInvalid(ValueError):
    def __init__(self):
        super().__init__("Invalid robustness metadata. Use the approved synthetic protocol.")


def _require(condition):
    if not condition:
        raise RobustnessInvalid()


def _integer(value, maximum):
    _require(type(value) is int and 1 <= value <= maximum)


class Provenance(StrEnum):
    OWNED_SYNTHETIC = "owned_synthetic"


class Classification(StrEnum):
    SOFTWARE_DIAGNOSTIC = "software_diagnostic"


class Category(StrEnum):
    CASHEW = "cashew"
    METALPARTS = "metalparts"


class ParentID(StrEnum):
    FIRST = "case-001"
    SECOND = "case-002"
    THIRD = "case-003"
    FOURTH = "case-004"


class Kind(StrEnum):
    IDENTITY = "identity"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    BLUR = "gaussian_blur"
    NOISE = "gaussian_noise"
    JPEG = "jpeg"
    ROTATION = "rotation"
    TRANSLATION = "translation"


class Axis(StrEnum):
    X = "x"
    Y = "y"


class Applicability(StrEnum):
    APPLICABLE = "applicable"
    CONDITIONAL = "conditional"
    NOT_APPLICABLE = "not_applicable"


class DiagnosticStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Perturbation:
    kind: Kind
    severity: float | int | None = None
    axis: Axis | None = None
    seed: int | None = None

    def __post_init__(self):
        _require(type(self.kind) is Kind)
        if self.kind is Kind.IDENTITY:
            _require(self.severity is None and self.axis is None and self.seed is None)
            return
        _require(type(self.severity) is (int if self.kind is Kind.JPEG else float))
        if self.kind is not Kind.JPEG:
            _require(math.isfinite(self.severity))
        allowed = {
            Kind.BRIGHTNESS: (0.90, 1.10, 0.80, 1.20, 0.65, 1.35),
            Kind.CONTRAST: (0.90, 1.10, 0.80, 1.20, 0.60, 1.40),
            Kind.BLUR: (0.5, 1.0, 2.0),
            Kind.NOISE: (2 / 255, 5 / 255, 10 / 255),
            Kind.JPEG: (95, 80, 60),
            Kind.ROTATION: (-1.0, 1.0, -3.0, 3.0, -5.0, 5.0),
            Kind.TRANSLATION: (-0.01, 0.01, -0.02, 0.02, -0.04, 0.04),
        }
        _require(self.severity in allowed[self.kind])
        if self.kind is Kind.TRANSLATION:
            _require(type(self.axis) is Axis)
        else:
            _require(self.axis is None)
        if self.kind is Kind.NOISE:
            _require(type(self.seed) is int and self.seed in (42, 1729, 2026))
        else:
            _require(self.seed is None)


@dataclass(frozen=True, slots=True)
class Budgets:
    max_parents: int = 4
    max_inference_calls: int = 256
    max_source_pixels: int = 4194304
    max_native_tiles: int = 128
    max_report_bytes: int = MAX_REPORT_BYTES

    def __post_init__(self):
        for value, limit in (
            (self.max_parents, 4),
            (self.max_inference_calls, 256),
            (self.max_source_pixels, 4194304),
            (self.max_native_tiles, 128),
            (self.max_report_bytes, MAX_REPORT_BYTES),
        ):
            _integer(value, limit)


@dataclass(frozen=True, slots=True)
class Parent:
    parent_id: ParentID
    width: int
    height: int

    def __post_init__(self):
        _require(type(self.parent_id) is ParentID)
        _integer(self.width, 4096)
        _integer(self.height, 4096)


def _identity(protocol_id, protocol_version, schema_version, provenance, classification):
    _require(type(protocol_id) is str and protocol_id == PROTOCOL_ID)
    _require(type(protocol_version) is int and protocol_version == 1)
    _require(type(schema_version) is int and schema_version == 1)
    _require(type(provenance) is Provenance)
    _require(type(classification) is Classification)


def _tiles(parent):
    # Budget arithmetic only: native 256-pixel windows, stride 192, edge coverage.
    return (1 + (parent.width - 256 + 191) // 192) * (1 + (parent.height - 256 + 191) // 192)


@dataclass(frozen=True, slots=True)
class Plan:
    category: Category
    parents: tuple[Parent, ...]
    perturbations: tuple[Perturbation, ...]
    budgets: Budgets = Budgets()
    baseline_repetitions: int = 3
    protocol_id: str = PROTOCOL_ID
    protocol_version: int = 1
    schema_version: int = 1
    provenance: Provenance = Provenance.OWNED_SYNTHETIC
    classification: Classification = Classification.SOFTWARE_DIAGNOSTIC

    def __post_init__(self):
        _identity(
            self.protocol_id,
            self.protocol_version,
            self.schema_version,
            self.provenance,
            self.classification,
        )
        _require(type(self.category) is Category and type(self.budgets) is Budgets)
        self.budgets.__post_init__()
        _require(type(self.parents) is tuple and 1 <= len(self.parents) <= self.budgets.max_parents)
        _require(type(self.perturbations) is tuple and 1 <= len(self.perturbations) <= 46)
        for parent in self.parents:
            _require(type(parent) is Parent)
            parent.__post_init__()
        for perturbation in self.perturbations:
            _require(type(perturbation) is Perturbation)
            perturbation.__post_init__()
        _require(len({p.parent_id for p in self.parents}) == len(self.parents))
        _require(len(set(self.perturbations)) == len(self.perturbations))
        _require(self.perturbations[0].kind is Kind.IDENTITY)
        _integer(self.baseline_repetitions, 256)
        _require(sum(p.width * p.height for p in self.parents) <= self.budgets.max_source_pixels)
        if self.category is Category.METALPARTS:
            for parent in self.parents:
                _require(parent.width >= 256 and parent.height >= 256)
                _require(_tiles(parent) <= self.budgets.max_native_tiles)
        _require(self.planned_inference_calls <= self.budgets.max_inference_calls)

    @property
    def planned_inference_calls(self) -> int:
        """Upper bound including every tile, baseline repeat and explicit noise replicate."""
        units = sum(_tiles(p) if self.category is Category.METALPARTS else 1 for p in self.parents)
        return units * (self.baseline_repetitions + len(self.perturbations) - 1)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """Metadata-only outcome boundary. Measured values require a later reviewed schema."""

    parent_id: ParentID
    perturbation: Perturbation
    applicability: Applicability
    status: DiagnosticStatus = DiagnosticStatus.NOT_EVALUATED
    protocol_id: str = PROTOCOL_ID
    protocol_version: int = 1
    schema_version: int = 1
    provenance: Provenance = Provenance.OWNED_SYNTHETIC
    classification: Classification = Classification.SOFTWARE_DIAGNOSTIC

    def __post_init__(self):
        _identity(
            self.protocol_id,
            self.protocol_version,
            self.schema_version,
            self.provenance,
            self.classification,
        )
        _require(type(self.parent_id) is ParentID and type(self.perturbation) is Perturbation)
        self.perturbation.__post_init__()
        _require(
            type(self.applicability) is Applicability and type(self.status) is DiagnosticStatus
        )
        _require(
            (self.applicability is Applicability.NOT_APPLICABLE)
            == (self.status is DiagnosticStatus.NOT_APPLICABLE)
        )


def _exact(data, record_type):
    _require(type(data) is dict and set(data) == {field.name for field in fields(record_type)})


def _enum(enum_type, value):
    _require(type(value) is str)
    return enum_type(value)


def _perturbation(data):
    _exact(data, Perturbation)
    return Perturbation(
        _enum(Kind, data["kind"]),
        data["severity"],
        None if data["axis"] is None else _enum(Axis, data["axis"]),
        data["seed"],
    )


def _decode(data, record_type):
    _require(record_type is Plan or record_type is Diagnostic)
    _exact(data, record_type)
    values = dict(data)
    values["provenance"] = _enum(Provenance, data["provenance"])
    values["classification"] = _enum(Classification, data["classification"])
    if record_type is Plan:
        _require(type(data["parents"]) is list and 1 <= len(data["parents"]) <= 4)
        _require(type(data["perturbations"]) is list and 1 <= len(data["perturbations"]) <= 46)
        _exact(data["budgets"], Budgets)
        values["budgets"] = Budgets(**data["budgets"])
        values["category"] = _enum(Category, data["category"])
        parents = []
        for item in data["parents"]:
            _exact(item, Parent)
            parents.append(
                Parent(_enum(ParentID, item["parent_id"]), item["width"], item["height"])
            )
        values["parents"] = tuple(parents)
        values["perturbations"] = tuple(_perturbation(p) for p in data["perturbations"])
    else:
        values["parent_id"] = _enum(ParentID, data["parent_id"])
        values["perturbation"] = _perturbation(data["perturbation"])
        values["applicability"] = _enum(Applicability, data["applicability"])
        values["status"] = _enum(DiagnosticStatus, data["status"])
    return record_type(**values)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result)
        result[key] = value
    return result


def from_json(raw: bytes, record_type: type[Plan] | type[Diagnostic]) -> Plan | Diagnostic:
    """Bounded, exact JSON schema; sanitize parse errors as well as validation errors."""
    try:
        _require(type(raw) is bytes and len(raw) <= MAX_METADATA_BYTES)
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
        return _decode(data, record_type)
    except (ValueError, TypeError, RecursionError):
        raise RobustnessInvalid() from None


def json_bytes(record: Plan | Diagnostic, *, max_bytes: int = MAX_REPORT_BYTES) -> bytes:
    """Canonical UTF-8 scalar metadata only; no file writes, source paths or metric arrays."""
    try:
        _integer(max_bytes, MAX_REPORT_BYTES)
        _require(type(record) in (Plan, Diagnostic))
        record.__post_init__()
        raw = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)
        payload = raw.encode("utf-8")
        _require(len(payload) <= min(max_bytes, MAX_METADATA_BYTES))
        from_json(payload, type(record))
        return payload
    except (ValueError, TypeError, RecursionError):
        raise RobustnessInvalid() from None
