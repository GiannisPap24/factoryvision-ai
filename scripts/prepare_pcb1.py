"""One-shot PCB1 normal-only preparation. Official test payloads stay in the archive."""

import json
import sys
from pathlib import Path

from factoryvision.data.category import audit_normals, extract_normals, load_category
from factoryvision.data.development import DevelopmentReadGuard
from factoryvision.data.manifest import manifest_sha256, manifest_to_dict
from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import OfficialSplit
from factoryvision.training.checkpoint import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "c7b59ae7d88e1ec32009067cae7f018e7ad2f5106f2737d6985563fe81ef38bc"


def main():
    protocol = ROOT / "docs/protocols/pcb1-confirmatory-v3.json"
    if file_sha256(protocol) != PROTOCOL_SHA:
        raise ValueError("Frozen PCB1 methodology changed.")
    recipe = json.loads(protocol.read_text())
    csv = ROOT / "data/metadata/1cls.csv"
    if file_sha256(csv) != recipe["source"]["split_sha256"]:
        raise ValueError("Official split changed.")
    manifest = load_category(csv, "pcb1")
    output = ROOT / "data/pcb1-development"
    normal = [s for s in manifest.samples if s.official_split == OfficialSplit.TRAIN]
    guard = DevelopmentReadGuard(output, normal)
    sys.addaudithook(guard)
    extraction = extract_normals(
        ROOT / "data/raw/VisA_20220922.tar", output, manifest, recipe["source"]["archive_sha256"]
    )
    official, audit = audit_normals(manifest, output)
    derived = derive_splits(official, 42)
    for s, original in zip(derived.samples, official.samples, strict=True):
        if original.official_split == OfficialSplit.TEST:
            assert s == original
    write_json(ROOT / "data/manifests", "pcb1-official.json", manifest_to_dict(official))
    write_json(ROOT / "data/manifests", "pcb1-seed42.json", manifest_to_dict(derived))
    report = {
        "protocol_sha256": PROTOCOL_SHA,
        "extraction": extraction,
        "audit": audit,
        "source_csv_sha256": file_sha256(csv),
        "manifest_sha256": manifest_sha256(derived),
        "parent_sha256": manifest_sha256(official),
        "partitions": {
            p: sum(s.partition.value == p for s in derived.samples)
            for p in ("train", "validation", "calibration", "test")
        },
        "test_membership_unchanged": True,
        "test_labels_used_for_selection": False,
        "normal_files_opened": len(guard.opened),
        "additional_download_bytes": 0,
    }
    write_json(ROOT / "data/metadata", "pcb1-preparation.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("audit", "extraction")}, indent=2)
    )


if __name__ == "__main__":
    main()
