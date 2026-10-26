"""Explicit 2E2-R only: read already-opened PCB1 images, never masks or model outputs.

Writes new forensic evidence exclusively; preserves original opening and frozen seal.
Supplemental diagnostics are post-incident and cannot authorize model evaluation.
"""

import csv
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from tarfile import open as open_tar

import numpy as np
from PIL import Image

from factoryvision.data.integrity_groups import (
    fingerprint_bytes,
    group_evidence,
    perceptual_pairs,
    reject_cross_split,
)
from factoryvision.data.manifest import load_manifest
from factoryvision.evaluation.confirmatory import verified_seal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pcb1-incident-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, record):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    started = time.perf_counter()
    before = json.loads((OUT / "preservation-before.json").read_text())
    assert all(sha(ROOT / p) == v["sha256"] for p, v in before["original_files"].items())
    seal_path = ROOT / "docs/evidence/pcb1-evaluation-seal.json"
    seal = verified_seal(ROOT, seal_path, sha(seal_path))
    write(
        "audit-started.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "script_sha256": sha(Path(__file__)),
            "scope": "PCB1 images only; no masks/models/new category payloads",
        },
    )
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    opening = ROOT / "reports/pcb1-opening"
    original = json.loads((opening / "integrity.json").read_text())
    screen = json.loads((opening / "screen.json").read_text())
    review = json.loads((opening / "review.json").read_text())
    records, access = [], []
    samples = {sample.sample_id: sample for sample in manifest.samples}
    for sample in manifest.samples:
        base = "pcb1-test-quarantine" if sample.partition.value == "test" else "pcb1-development"
        path = ROOT / "data" / base / sample.image_path
        record = fingerprint_bytes(sample.sample_id, sample.partition.value, path.read_bytes())
        records.append(record)
        access.append(
            {
                "id": sample.sample_id,
                "partition": sample.partition.value,
                "path": path.relative_to(ROOT).as_posix(),
                "byte_sha256": record.byte_sha256,
                "pixel_sha256": record.pixel_sha256,
            }
        )
    by_id = {record.sample_id: record for record in records}
    for row in original["records"]:
        assert row["image_sha256"] == by_id[row["sample_id"]].byte_sha256
        assert row["pixel_sha256"] == by_id[row["sample_id"]].pixel_sha256
    indexed = perceptual_pairs(records)
    expected = {
        (p["a"], p["b"])
        for p in screen["refined_candidates"]
        if p["dhash_distance"] <= 4 or p["phash_distance"] <= 4
    }
    assert {(p["a"], p["b"]) for p in indexed["pairs"]} == expected
    pair = ("visa-pcb1-normal-0187", "visa-pcb1-normal-0188")
    assert review["confirmed_leakage"] is True
    required = {(p["a"], p["b"]) for p in review["reviewed_pairs"]}
    unresolved = [
        (p["a"], p["b"]) for p in screen["refined_candidates"] if (p["a"], p["b"]) not in required
    ]
    groups = group_evidence(records, confirmed_pairs=[pair], unresolved_pairs=unresolved)
    try:
        reject_cross_split(groups)
    except ValueError as exc:
        gate = str(exc)
    else:
        raise AssertionError("Incident must remain blocked.")
    archive_members = {}
    wanted = {samples[key].image_path for key in pair}
    with open_tar(ROOT / "data/raw/VisA_20220922.tar", "r:") as archive:
        for member in archive:
            if member.name in wanted:
                payload = archive.extractfile(member).read()
                archive_members[member.name] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
    assert set(archive_members) == wanted
    for key in pair:
        assert archive_members[samples[key].image_path]["sha256"] == by_id[key].byte_sha256
    with (ROOT / "data/metadata/1cls.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    original_rows = [row for row in rows if row["image"] in wanted]
    assert len(original_rows) == 2
    assert {row["split"] for row in original_rows} == {"train", "test"}
    metadata_counts = {
        category: dict(
            Counter(f"{r['split']}/{r['label']}" for r in rows if r["object"] == category)
        )
        for category in ("pcb2", "pcb3")
    }
    arrays, geometry = [], []
    for key in pair:
        path = next(r["path"] for r in access if r["id"] == key)
        with Image.open(ROOT / path) as image:
            geometry.append(list(image.size))
            arrays.append(np.asarray(image.convert("RGB"), dtype=np.float32) / 255)
    a, b = arrays
    diagnostic = {
        "stage": "post-incident forensic; no evaluation scores",
        "dimensions": geometry,
        "same_dimensions": a.shape == b.shape,
        "identical_rgb_pixels": bool(np.array_equal(a, b)),
        "fraction_changed_rgb_pixels": float(np.mean(np.any(a != b, axis=-1))),
        "rgb_rms_original_resolution": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_rgb_difference_b_minus_a": [float(v) for v in (b - a).mean(axis=(0, 1))],
    }
    regions = {
        "full": (slice(None), slice(None)),
        "top_left_background": (slice(0, 150), slice(0, 150)),
        "bottom_right_background": (slice(-150, None), slice(-150, None)),
        "center_object": (slice(250, 800), slice(150, 1150)),
    }
    diagnostic["regional_rgb_pearson"] = {
        name: float(np.corrcoef(a[region].ravel(), b[region].ravel())[0, 1])
        for name, region in regions.items()
    }
    partition_counts = Counter()
    for candidate in screen["refined_candidates"]:
        partition_counts["/".join(sorted(candidate["partitions"]))] += 1
    write("fingerprints.json", [asdict(r) for r in records])
    write("image-access.json", access)
    write("groups.json", groups)
    write("indexed-screen.json", indexed)
    write(
        "source-pair.json",
        {
            "archive_members": archive_members,
            "official_csv_rows": original_rows,
            "upstream_overlap_predates_preprocessing": True,
            "supplemental_diagnostics": diagnostic,
        },
    )
    write(
        "summary.json",
        {
            "verdict": "BLOCKED - CONFIRMATORY EVALUATION NOT COMPLETED",
            "images_mechanically_read": len(records),
            "partition_counts": dict(Counter(r.partition for r in records)),
            "masks_opened": 0,
            "new_category_images_opened": 0,
            "model_predictions": 0,
            "original_cross_partition_comparisons": screen["cross_partition_comparisons"],
            "original_candidate_count": screen["candidate_count"],
            "strong_pairs": [p for p in screen["refined_candidates"] if p["strong"]],
            "original_reviewed_pairs": len(required),
            "unreviewed_broad_candidates": len(unresolved),
            "candidate_partition_counts": dict(sorted(partition_counts.items())),
            "indexed_comparisons": indexed["comparisons"],
            "indexed_candidates": len(indexed["pairs"]),
            "indexed_matches_original_hamming_candidates": True,
            "unconfirmed_candidate_sample_count": len({key for p in unresolved for key in p}),
            "confirmed_cross_partition_samples": groups["affected_confirmed_samples"],
            "rejection": gate,
            "new_category_metadata_only": metadata_counts,
            "four_families_fit_includes_0188": all(
                pair[1] in seal["membership"]["train"] for _ in seal["models"]
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "scope_limit": (
                "Cross-partition near-duplicate screen only; within-partition near-duplicates "
                "and physical item provenance are not exhaustively adjudicated."
            ),
        },
    )
    assert all(sha(ROOT / p) == v["sha256"] for p, v in before["original_files"].items())
    print(
        "Read-only incident audit complete; original files unchanged; evaluation remains blocked."
    )


if __name__ == "__main__":
    main()
