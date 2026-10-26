"""Boundary tests: strict types, immutability, missing/null and source integrity."""

import copy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from factoryvision.config import ValidationSettings
from factoryvision.data.manifest import manifest_from_dict, manifest_to_dict
from factoryvision.domain.contracts import (
    VISA_REVISION,
    VISA_SOURCE,
    VISA_VERSION,
    DatasetIdentity,
    Label,
    Manifest,
    OfficialSplit,
    Partition,
    Source,
    SourceKind,
    SplitProvenance,
)
from factoryvision.domain.errors import ContractError, Issue


def codes(error):
    return {issue.code for issue in error.value.issues}


def test_valid_normal_anomaly_and_explicit_null(manifest):
    assert manifest.samples[0].label is Label.NORMAL
    assert manifest.samples[0].mask_path is None
    assert manifest.samples[-1].label is Label.ANOMALY
    assert manifest.samples[-1].mask_path.endswith(".png")
    assert manifest.samples[0].content_sha256 is None
    assert manifest.samples[0].official_split is OfficialSplit.TRAIN


def test_input_and_contract_immutability(payload):
    original = copy.deepcopy(payload)
    manifest = manifest_from_dict(payload)
    assert payload == original
    payload["samples"][0]["sample_id"] = "changed"
    assert manifest.samples[0].sample_id != "changed"
    with pytest.raises(FrozenInstanceError):
        manifest.samples[0].sample_id = "changed"
    with pytest.raises(FrozenInstanceError):
        manifest.dataset.source.uri = "changed"
    with pytest.raises(FrozenInstanceError):
        manifest.samples = ()
    serialized = manifest_to_dict(manifest)
    serialized["samples"][0]["sample_id"] = "mutated"
    assert manifest_to_dict(manifest) != serialized


@pytest.mark.parametrize("key", ["mask_path", "group_id", "content_sha256", "sample_id"])
def test_missing_is_not_null(payload, key):
    del payload["samples"][0][key]
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert codes(exc) == {"FIELD_MISSING"}


@pytest.mark.parametrize(
    "value", ["", " ", " bad", "bad ", "UPPER", "../x", "a" * 129, None, 1, []]
)
def test_invalid_identifiers(payload, value):
    payload["samples"][0]["sample_id"] = value
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert "ID_INVALID" in codes(exc)


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("category", "unknown", "CATEGORY_UNSUPPORTED"),
        ("category", [], "CATEGORY_UNSUPPORTED"),
        ("label", "defective", "ENUM_INVALID"),
        ("label", None, "ENUM_INVALID"),
        ("label", {}, "ENUM_INVALID"),
        ("official_split", "validation", "ENUM_INVALID"),
        ("partition", "unknown", "ENUM_INVALID"),
        ("mask_path", "", "PATH_INVALID"),
        ("mask_path", "mask.jpg", "MASK_FORMAT"),
        ("mask_path", "mask.png", "MASK_LABEL_MISMATCH"),
        ("dataset_version", "", "TEXT_INVALID"),
        ("dataset_version", " synthetic-v1", "TEXT_INVALID"),
        ("group_id", "", "ID_INVALID"),
        ("content_sha256", "", "HASH_INVALID"),
        ("content_sha256", "A" * 64, "HASH_INVALID"),
        ("content_sha256", 1, "HASH_INVALID"),
    ],
)
def test_invalid_sample_fields(payload, key, value, code):
    payload["samples"][0][key] = value
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert code in codes(exc)


def test_anomaly_without_mask(payload):
    payload["samples"][-1]["mask_path"] = None
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert codes(exc) == {"MASK_LABEL_MISMATCH"}


@pytest.mark.parametrize(
    ("location", "key", "value", "code"),
    [
        ("root", "schema_version", 2, "SCHEMA_VERSION"),
        ("root", "schema_version", True, "SCHEMA_VERSION"),
        ("root", "samples", None, "TYPE_INVALID"),
        ("root", "samples", [], "SAMPLES_INVALID"),
        ("root", "dataset", [], "TYPE_INVALID"),
        ("root", "split_provenance", {}, "FIELD_MISSING"),
        ("source", "kind", "unknown", "ENUM_INVALID"),
        ("source", "attribution", "", "TEXT_INVALID"),
        ("source", "attribution", None, "TEXT_INVALID"),
        ("source", "uri", "https://invented.example", "SOURCE_INVALID"),
        ("source", "revision", "synthetic-v2", "SOURCE_INVALID"),
        ("source", "license_id", "CC-BY-4.0", "SOURCE_INVALID"),
        ("dataset", "version", "synthetic-v2", "DATASET_SOURCE_MISMATCH"),
        ("dataset", "dataset_id", "visa", "DATASET_SOURCE_MISMATCH"),
    ],
)
def test_invalid_manifest_source_fields(payload, location, key, value, code):
    target = payload if location == "root" else payload["dataset"]
    if location == "source":
        target = target["source"]
    target[key] = value
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert code in codes(exc)


@pytest.mark.parametrize("location", ["root", "dataset", "source", "sample"])
def test_unknown_fields_rejected(payload, location):
    target = {
        "root": payload,
        "dataset": payload["dataset"],
        "source": payload["dataset"]["source"],
        "sample": payload["samples"][0],
    }[location]
    target["invented"] = 1
    with pytest.raises(ContractError) as exc:
        manifest_from_dict(payload)
    assert codes(exc) == {"FIELD_UNKNOWN"}


def test_valid_official_identity_is_metadata_only():
    source = Source(SourceKind.VISA, VISA_SOURCE, VISA_REVISION, "CC-BY-4.0", "VisA authors")
    assert DatasetIdentity("visa", VISA_VERSION, source).source == source
    with pytest.raises(ContractError, match="SOURCE_INVALID"):
        replace(source, revision="0" * 40)
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        DatasetIdentity("visa", VISA_VERSION, {})


def test_direct_constructor_strict_types(manifest):
    with pytest.raises(ContractError, match="ENUM_INVALID"):
        replace(manifest.samples[0], label="normal")
    with pytest.raises(ContractError, match="ENUM_INVALID"):
        replace(manifest.dataset.source, kind="synthetic")
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        replace(manifest, dataset={})
    with pytest.raises(ContractError, match="SAMPLES_INVALID"):
        replace(manifest, samples=list(manifest.samples))
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        replace(manifest, samples=("bad",))
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        replace(manifest, split_provenance={})
    assert isinstance(manifest, Manifest)


@pytest.mark.parametrize("seed", [-1, 2**32, True, "42", None])
def test_provenance_seed(seed):
    with pytest.raises(ContractError, match="SEED_INVALID"):
        SplitProvenance("sha256-group-v1", seed, "a" * 64, "b" * 64)


def test_provenance_algorithm():
    with pytest.raises(ContractError, match="SPLIT_ALGORITHM"):
        SplitProvenance("random", 42, "a" * 64, "b" * 64)


def test_config_valid_and_invalid(tmp_path):
    assert ValidationSettings(tmp_path).dataset_root == tmp_path
    assert ValidationSettings().dataset_root is None
    for value in (0, -1, True, "100"):
        with pytest.raises(ContractError, match="CONFIG_INVALID"):
            ValidationSettings(max_manifest_bytes=value)
    for root in (Path("relative"), "/absolute-but-string"):
        with pytest.raises(ContractError, match="CONFIG_INVALID"):
            ValidationSettings(dataset_root=root)


def test_issue_sorting_and_error_text():
    error = ContractError((Issue("Z", "field", "last"), Issue("A", "field", "first")))
    assert str(error) == "A [field]: first; Z [field]: last"
    assert error.issues[0].code == "A"


def test_nullable_group_and_known_hash(manifest):
    sample = replace(manifest.samples[0], group_id=None, content_sha256="a" * 64)
    assert sample.group_id is None
    assert sample.partition is Partition.TRAIN
