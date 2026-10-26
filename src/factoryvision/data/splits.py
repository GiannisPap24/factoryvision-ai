"""Versioned group splitting. No file reads, image hashes or model fitting."""

import hashlib
from collections import defaultdict
from dataclasses import replace

from factoryvision.data.manifest import manifest_sha256
from factoryvision.domain.contracts import Manifest, OfficialSplit, Partition, SplitProvenance
from factoryvision.domain.errors import reject
from factoryvision.domain.validation import assignments_sha256, require_valid_manifest


def derive_splits(manifest: Manifest, seed: int = 42) -> Manifest:
    """Assign normal training groups 70/15/remainder; preserve the official test.

    Ratios apply to groups, not images. At least three independent groups per
    category are required. Explicit groups are kept intact; ungrouped samples
    count as independent only provisionally, pending a future duplicate audit.
    """
    require_valid_manifest(manifest)
    if type(seed) is not int or not 0 <= seed < 2**32:
        reject("SEED_INVALID", "seed", "Use an integer in [0, 2**32).")
    if manifest.split_provenance is not None:
        reject("SPLIT_ALREADY_DERIVED", "split_provenance", "Split the original parent manifest.")
    categories: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    result = []
    for sample in manifest.samples:
        if sample.official_split == OfficialSplit.TEST:
            result.append(sample)
        else:
            group = f"group:{sample.group_id}" if sample.group_id else f"sample:{sample.sample_id}"
            categories[sample.category][group].append(sample)
    if not categories:
        reject(
            "SPLIT_NO_TRAINING", "samples", "Provide normal training groups before deriving splits."
        )
    for category, groups in sorted(categories.items()):
        keys = sorted(
            groups,
            key=lambda group: (
                hashlib.sha256(f"{seed}|{category}|{group}".encode()).hexdigest(),
                group,
            ),
        )
        count = len(keys)
        if count < 3:
            reject(
                "SPLIT_TOO_FEW_GROUPS", category, "Need at least three independent normal groups."
            )
        train_count = min(max(1, count * 70 // 100), count - 2)
        validation_count = max(1, count * 15 // 100)
        for index, key in enumerate(keys):
            if index < train_count:
                partition = Partition.TRAIN
            elif index < train_count + validation_count:
                partition = Partition.VALIDATION
            else:
                partition = Partition.CALIBRATION
            result.extend(replace(sample, partition=partition) for sample in groups[key])
    output = replace(manifest, samples=tuple(sorted(result, key=lambda sample: sample.sample_id)))
    provenance = SplitProvenance(
        "sha256-group-v1", seed, manifest_sha256(manifest), assignments_sha256(output)
    )
    output = replace(output, split_provenance=provenance)
    require_valid_manifest(output)
    return output
