"""Explicit pretraining integrity gate and near-duplicate candidate report. No training."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from factoryvision.data.images import decode_image
from factoryvision.data.leakage import fingerprints, screen_pairs
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.splits import derive_splits
from factoryvision.data.visa import audit_cashew, load_visa_cashew
from factoryvision.domain.contracts import Label, Partition


def main():
    root = Path(__file__).resolve().parents[1]
    parent = load_manifest(root / "data/manifests/cashew-official.json")
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    actual, audit = audit_cashew(
        load_visa_cashew(root / "data/metadata/1cls.csv"),
        root / "data/visa",
        root / "data/visa/cashew/image_anno.csv",
    )
    previous = json.loads((root / "data/metadata/cashew-audit.json").read_text())
    assert actual == parent and audit["records"] == previous["records"]
    assert derive_splits(actual, 42) == manifest
    assert Counter(s.partition.value for s in manifest.samples) == {
        "train": 315,
        "validation": 67,
        "calibration": 68,
        "test": 150,
    }
    assert all(s.label == Label.NORMAL for s in manifest.samples if s.partition != Partition.TEST)
    records = []
    for sample in manifest.samples:
        dh, ph, rgb = fingerprints(decode_image(root / "data/visa", sample.image_path))
        records.append(
            {
                "id": sample.sample_id,
                "partition": sample.partition.value,
                "dhash": dh,
                "phash": ph,
                "rgb": rgb,
            }
        )
    result = screen_pairs(records)
    result.update(
        manifest_sha256=manifest_sha256(manifest),
        parent_sha256=manifest_sha256(parent),
        exact_integrity_passed=True,
        original_masks_unchanged=True,
        official_test_preserved=True,
        fingerprint_recipe=(
            "LANCZOS; 9x8 horizontal dHash; 32x32 DCT top8x8 without DC, median bits; RGB32 RMS"
        ),
        fingerprints=[{k: v for k, v in r.items() if k != "rgb"} for r in records],
    )
    result["fingerprints_sha256"] = hashlib.sha256(
        json.dumps(result["fingerprints"], sort_keys=True).encode()
    ).hexdigest()
    target = root / "reports/cashew-leakage-screen.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("fingerprints", "candidates")}, indent=2
        )
    )
    print(f"Candidates requiring review: {len(result['candidates'])}")
    print(json.dumps(result["candidates"][:12], indent=2))


if __name__ == "__main__":
    main()
