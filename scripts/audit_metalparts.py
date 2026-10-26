"""Explicit model-free MetalParts audit. Never downloads, trains or scores a model."""

import argparse
import hashlib
import json
from pathlib import Path

from factoryvision.data.metalparts import REVISION, load_metalparts
from factoryvision.data.metalparts_audit import audit_metalparts, verify_source_files, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Pinned revision directory")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Evidence must be outside the source dataset directory")
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.root / "source-repository.json").read_text(encoding="utf-8"))
    if (
        source["sha"] != REVISION
        or source["id"] != "AmazonScience/RobustAD"
        or source["cardData"]["license"] != "cc-by-4.0"
    ):
        parser.error("Source repository/revision mismatch")
    inventory_path = args.root / "source-inventory.json"
    verified = verify_source_files(
        args.root, json.loads(inventory_path.read_text(encoding="utf-8")), source["sha"]
    )
    write_json(
        args.output / "source-verification.json",
        {
            **verified,
            "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
            "revision": REVISION,
        },
    )
    manifest = load_metalparts(args.root)
    write_json(args.output / "manifest.json", manifest.to_dict())
    report = audit_metalparts(args.root, manifest)
    write_json(args.output / "audit.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("images", "masks", "dimensions", "blockers", "verdict")},
            ensure_ascii=True,
        )
    )
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
