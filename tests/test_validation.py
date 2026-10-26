"""Manifest-wide contradictions and leakage must be rejected."""

from dataclasses import replace

import pytest

from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import (
    VISA_REVISION,
    VISA_SOURCE,
    VISA_VERSION,
    DatasetIdentity,
    Label,
    OfficialSplit,
    Partition,
    Source,
    SourceKind,
)
from factoryvision.domain.errors import ContractError
from factoryvision.domain.validation import require_valid_manifest, validate_manifest


def issue_codes(manifest):
    return {issue.code for issue in validate_manifest(manifest)}


def test_repeated_validation_and_order_invariance(manifest):
    assert validate_manifest(manifest) == ()
    assert validate_manifest(manifest) == validate_manifest(manifest)
    duplicate = replace(manifest, samples=manifest.samples + (manifest.samples[0],))
    reversed_duplicate = replace(duplicate, samples=tuple(reversed(duplicate.samples)))
    assert validate_manifest(duplicate) == validate_manifest(reversed_duplicate)
    with pytest.raises(ContractError, match="DUPLICATE_ID"):
        require_valid_manifest(duplicate)


def test_duplicate_identity_across_partitions(manifest):
    original = manifest.samples[0]
    copy = replace(original, partition=Partition.TEST, official_split=OfficialSplit.TEST)
    broken = replace(manifest, samples=manifest.samples + (copy,))
    assert {
        "DUPLICATE_ID",
        "IDENTITY_LEAKAGE",
        "REFERENCE_LEAKAGE",
        "DUPLICATE_REFERENCE",
        "GROUP_LEAKAGE",
    } <= issue_codes(broken)


def test_case_insensitive_duplicate_reference(manifest):
    other = replace(manifest.samples[1], image_path=manifest.samples[0].image_path.upper())
    assert "DUPLICATE_REFERENCE" in issue_codes(
        replace(manifest, samples=(manifest.samples[0], other))
    )


def test_shared_mask_or_image_mask_alias(manifest):
    anomaly = manifest.samples[-1]
    other = replace(anomaly, sample_id="other", image_path="other.JPG")
    assert "DUPLICATE_REFERENCE" in issue_codes(replace(manifest, samples=(anomaly, other)))
    aliased = replace(anomaly, image_path=anomaly.mask_path)
    assert "DUPLICATE_REFERENCE" in issue_codes(replace(manifest, samples=(aliased,)))


@pytest.mark.parametrize("field", ["dataset_id", "dataset_version"])
def test_sample_identity_mismatch(manifest, field):
    sample = replace(manifest.samples[0], **{field: "wrong"})
    assert "SAMPLE_DATASET_MISMATCH" in issue_codes(replace(manifest, samples=(sample,)))


@pytest.mark.parametrize(
    ("official", "partition"),
    [
        (OfficialSplit.TEST, Partition.TRAIN),
        (OfficialSplit.TRAIN, Partition.TEST),
    ],
)
def test_official_test_is_immutable(manifest, official, partition):
    sample = replace(manifest.samples[0], official_split=official, partition=partition)
    assert "OFFICIAL_TEST_CHANGED" in issue_codes(replace(manifest, samples=(sample,)))


def test_anomaly_not_allowed_in_development(manifest):
    sample = replace(
        manifest.samples[-1], official_split=OfficialSplit.TRAIN, partition=Partition.TRAIN
    )
    codes = issue_codes(replace(manifest, samples=(sample,)))
    assert {"ONE_CLASS_LABEL", "DEVELOPMENT_ANOMALY"} <= codes


def test_group_category_contradiction(manifest):
    other = replace(manifest.samples[1], category="pcb1", group_id=manifest.samples[0].group_id)
    assert "GROUP_CATEGORY_MISMATCH" in issue_codes(
        replace(manifest, samples=(manifest.samples[0], other))
    )


def test_known_content_hash_leakage(manifest):
    train = replace(manifest.samples[0], content_sha256="a" * 64)
    test = replace(manifest.samples[-1], content_sha256="a" * 64)
    assert "CONTENT_LEAKAGE" in issue_codes(replace(manifest, samples=(train, test)))
    same_partition = replace(manifest.samples[1], content_sha256="a" * 64)
    assert "CONTENT_LEAKAGE" not in issue_codes(replace(manifest, samples=(train, same_partition)))


def test_missing_and_wrong_split_provenance(manifest):
    derived = derive_splits(manifest)
    assert validate_manifest(derived) == ()
    assert "SPLIT_PROVENANCE_REQUIRED" in issue_codes(replace(derived, split_provenance=None))
    bad = replace(derived.split_provenance, assignments_sha256="0" * 64)
    assert "SPLIT_HASH_MISMATCH" in issue_codes(replace(derived, split_provenance=bad))


def test_official_visa_layout_on_fabricated_metadata(manifest):
    source = Source(SourceKind.VISA, VISA_SOURCE, VISA_REVISION, "CC-BY-4.0", "VisA authors")
    identity = DatasetIdentity("visa", VISA_VERSION, source)
    normal = replace(
        manifest.samples[0],
        dataset_id="visa",
        dataset_version=VISA_VERSION,
        image_path="cashew/Data/Images/Normal/fabricated-not-a-real-file.JPG",
    )
    anomaly = replace(
        manifest.samples[-1],
        dataset_id="visa",
        dataset_version=VISA_VERSION,
        image_path="cashew/Data/Images/Anomaly/fabricated-not-a-real-file.JPG",
        mask_path="cashew/Data/Masks/Anomaly/fabricated-not-a-real-file.png",
    )
    candidate = replace(manifest, dataset=identity, samples=(normal, anomaly))
    assert validate_manifest(candidate) == ()
    bad = replace(normal, image_path="pcb1/Data/Images/Normal/example.JPG")
    assert "VISA_IMAGE_LAYOUT" in issue_codes(replace(candidate, samples=(bad,)))
    bad = replace(normal, image_path="cashew/Data/Images/Normal/example.png")
    assert "VISA_IMAGE_LAYOUT" in issue_codes(replace(candidate, samples=(bad,)))
    bad = replace(anomaly, mask_path="cashew/Data/Masks/Normal/example.png")
    assert "VISA_MASK_LAYOUT" in issue_codes(replace(candidate, samples=(bad,)))
    assert anomaly.label is Label.ANOMALY
