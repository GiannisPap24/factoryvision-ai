"""Strict JSON boundary. Missing keys, nulls and unknown keys are distinct."""

import hashlib
import json
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path

from factoryvision.config import ValidationSettings
from factoryvision.domain.contracts import (
    DatasetIdentity,
    Label,
    Manifest,
    OfficialSplit,
    Partition,
    Sample,
    Source,
    SourceKind,
    SplitProvenance,
)
from factoryvision.domain.errors import ContractError, reject
from factoryvision.domain.validation import require_valid_manifest


def _object(value: object, fields: set[str], location: str) -> dict:
    if not isinstance(value, dict):
        reject("TYPE_INVALID", location, "Expected a JSON object.")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing:
        reject("FIELD_MISSING", location, f"Required fields: {', '.join(sorted(missing))}")
    if unknown:
        reject("FIELD_UNKNOWN", location, f"Unknown fields: {', '.join(sorted(unknown))}")
    return value


def _enum(enum_type: type[StrEnum], value: object, field: str) -> StrEnum:
    try:
        return enum_type(value)
    except (ValueError, TypeError):
        reject("ENUM_INVALID", field, f"Expected one of: {', '.join(enum_type)}")


def manifest_from_dict(value: object) -> Manifest:
    root = _object(value, {"schema_version", "dataset", "samples", "split_provenance"}, "manifest")
    identity = _object(root["dataset"], {"dataset_id", "version", "source"}, "dataset")
    source = _object(
        identity["source"], {"kind", "uri", "revision", "license_id", "attribution"}, "source"
    )
    dataset = DatasetIdentity(
        identity["dataset_id"],
        identity["version"],
        Source(
            _enum(SourceKind, source["kind"], "source.kind"),
            source["uri"],
            source["revision"],
            source["license_id"],
            source["attribution"],
        ),
    )
    if not isinstance(root["samples"], list):
        reject("TYPE_INVALID", "samples", "Expected a JSON array.")
    samples = []
    for index, raw in enumerate(root["samples"]):
        fields = set(Sample.__dataclass_fields__)
        record = _object(raw, fields, f"samples[{index}]")
        values = dict(record)
        for key, enum_type in (
            ("label", Label),
            ("official_split", OfficialSplit),
            ("partition", Partition),
        ):
            values[key] = _enum(enum_type, values[key], f"samples[{index}].{key}")
        samples.append(Sample(**values))
    provenance = root["split_provenance"]
    if provenance is not None:
        provenance = SplitProvenance(
            **_object(provenance, set(SplitProvenance.__dataclass_fields__), "split_provenance")
        )
    manifest = Manifest(root["schema_version"], dataset, tuple(samples), provenance)
    require_valid_manifest(manifest)
    return manifest


def manifest_to_dict(manifest: Manifest) -> dict:
    """Canonical ordering also makes output insensitive to input record order."""
    result = asdict(manifest)
    result["samples"] = sorted(result["samples"], key=lambda sample: sample["sample_id"])
    return result


def manifest_sha256(manifest: Manifest) -> str:
    payload = json.dumps(manifest_to_dict(manifest), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            reject("JSON_DUPLICATE_KEY", key, "JSON object keys must be unique.")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    reject("JSON_INVALID", "manifest", f"Non-finite JSON constant is not allowed: {value}")


def load_manifest(path: Path, settings: ValidationSettings | None = None) -> Manifest:
    settings = settings if settings is not None else ValidationSettings()
    try:
        with path.open("rb") as stream:
            content = stream.read(settings.max_manifest_bytes + 1)
    except OSError as exc:
        reject("MANIFEST_IO", "manifest", f"Cannot read manifest: {type(exc).__name__}")
    if len(content) > settings.max_manifest_bytes:
        reject("MANIFEST_TOO_LARGE", "manifest", "Manifest exceeds the configured byte limit.")
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_unique_pairs,
            parse_constant=_invalid_constant,
        )
    except ContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        reject("JSON_INVALID", "manifest", f"Invalid UTF-8 JSON: {type(exc).__name__}")
    return manifest_from_dict(value)
