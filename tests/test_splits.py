"""Stable splits, real grouping and preserved official tests."""

import random
from dataclasses import replace

import pytest

from factoryvision.data.manifest import manifest_sha256, manifest_to_dict
from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import OfficialSplit, Partition
from factoryvision.domain.errors import ContractError
from factoryvision.domain.validation import validate_manifest


def test_deterministic_order_and_provenance(manifest):
    original = manifest_to_dict(manifest)
    shuffled = list(manifest.samples)
    random.Random(123).shuffle(shuffled)
    reordered = replace(manifest, samples=tuple(shuffled))
    output = derive_splits(manifest)
    assert output == derive_splits(reordered)
    assert manifest_sha256(manifest) == manifest_sha256(reordered)
    assert output == derive_splits(manifest)
    assert output.split_provenance.parent_manifest_sha256 == manifest_sha256(manifest)
    assert manifest_to_dict(manifest) == original
    assert output.samples == tuple(sorted(output.samples, key=lambda sample: sample.sample_id))
    assert validate_manifest(output) == ()
    assert [s for s in output.samples if s.official_split == OfficialSplit.TEST] == sorted(
        [s for s in manifest.samples if s.official_split == OfficialSplit.TEST],
        key=lambda sample: sample.sample_id,
    )
    assert output != derive_splits(manifest, seed=43)


def test_groups_never_split(manifest):
    extra = replace(manifest.samples[0], sample_id="extra-view", image_path="cashew/extra.JPG")
    candidate = replace(manifest, samples=manifest.samples + (extra,))
    output = derive_splits(candidate)
    members = [s for s in output.samples if s.group_id == extra.group_id]
    assert len(members) == 2
    assert len({s.partition for s in members}) == 1


@pytest.mark.parametrize("count", [3, 4, 10])
def test_nonempty_partitions_for_small_data(manifest, count):
    candidate = replace(manifest, samples=manifest.samples[:count])
    output = derive_splits(candidate)
    assert {s.partition for s in output.samples} == {
        Partition.TRAIN,
        Partition.VALIDATION,
        Partition.CALIBRATION,
    }


def test_ungrouped_and_multiple_categories(manifest):
    first = tuple(replace(s, group_id=None) for s in manifest.samples[:3])
    second = tuple(
        replace(
            s,
            sample_id=f"pcb-{s.sample_id}",
            category="pcb1",
            image_path=f"pcb1/{s.sample_id}.JPG",
            group_id=f"pcb-{s.group_id}",
        )
        for s in manifest.samples[:3]
    )
    output = derive_splits(replace(manifest, samples=first + second))
    assert len(output.samples) == 6


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed(manifest, seed):
    with pytest.raises(ContractError, match="SEED_INVALID"):
        derive_splits(manifest, seed)


def test_no_training_too_few_and_already_derived(manifest):
    with pytest.raises(ContractError, match="SPLIT_NO_TRAINING"):
        derive_splits(replace(manifest, samples=(manifest.samples[-1],)))
    with pytest.raises(ContractError, match="SPLIT_TOO_FEW_GROUPS"):
        derive_splits(replace(manifest, samples=manifest.samples[:2]))
    with pytest.raises(ContractError, match="SPLIT_ALREADY_DERIVED"):
        derive_splits(derive_splits(manifest))


def test_known_content_cannot_be_split_despite_missing_group(manifest):
    candidate = replace(
        manifest, samples=tuple(replace(s, content_sha256="a" * 64) for s in manifest.samples[:3])
    )
    with pytest.raises(ContractError, match="CONTENT_LEAKAGE"):
        derive_splits(candidate)
