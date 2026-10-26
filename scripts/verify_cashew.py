"""Explicit real-data audit and reproducible manifest preparation; never downloads."""

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from factoryvision.data.manifest import manifest_sha256, manifest_to_dict
from factoryvision.data.splits import derive_splits
from factoryvision.data.visa import audit_cashew, load_visa_cashew
from factoryvision.domain.contracts import OfficialSplit


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "data/visa"
    metadata = root / "data/metadata/1cls.csv"
    acquisition = json.loads((root / "data/raw/acquisition.json").read_text(encoding="utf-8"))
    if (
        hashlib.sha256(metadata.read_bytes()).hexdigest()
        != acquisition["metadata"]["1cls.csv"]["sha256"]
    ):
        raise ValueError("Official split CSV differs from acquisition evidence.")
    started = time.perf_counter()
    original = load_visa_cashew(metadata)
    parent, audit = audit_cashew(original, dataset_root, dataset_root / "cashew/image_anno.csv")
    derived = derive_splits(parent, seed=42)
    assert derived == derive_splits(parent, seed=42)
    assert tuple(s for s in parent.samples if s.official_split == OfficialSplit.TEST) == tuple(
        s for s in derived.samples if s.official_split == OfficialSplit.TEST
    )
    destination = root / "data/manifests"
    destination.mkdir(exist_ok=True)
    for name, manifest in [("cashew-official.json", parent), ("cashew-seed42.json", derived)]:
        (destination / name).write_text(
            json.dumps(manifest_to_dict(manifest), indent=2) + "\n", encoding="utf-8"
        )
    audit.update(
        parent_manifest_sha256=manifest_sha256(parent),
        derived_manifest_sha256=manifest_sha256(derived),
        partitions=dict(Counter(s.partition.value for s in derived.samples)),
        split_seed=42,
        split_algorithm="sha256-group-v1",
        official_test_preserved=True,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    (root / "data/metadata/cashew-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in audit.items() if k not in ("records", "dimensions")}, indent=2)
    )
    print(f"Unique image dimensions: {len(audit['dimensions'])}")


if __name__ == "__main__":
    main()
