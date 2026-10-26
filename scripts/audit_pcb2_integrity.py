"""Explicit 2F0 mechanical PCB2 integrity audit. No models, masks or image previews."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import time
from collections import Counter
from ctypes import wintypes
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from factoryvision.data.category import load_category
from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.manifest import manifest_to_dict
from factoryvision.data.pretraining_integrity import (
    AuditDefect,
    AuditIncomplete,
    Limits,
    describe_image,
    extract_category_images,
    file_hash,
    gate_decision,
    read_protocol,
    scan_all_pairs,
)
from factoryvision.domain.errors import ContractError

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/pcb2-integrity-v1"
DATA = ROOT / "data/pcb2-integrity-v1"


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    info = Counters()
    info.cb = ctypes.sizeof(info)
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(wintypes.HANDLE(-1), ctypes.byref(info), info.cb):
        raise AuditIncomplete("Cannot measure process memory.")
    return info.peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    protocol_path = ROOT / "docs/protocols/pcb2-integrity-v1.json"
    protocol = read_protocol(protocol_path, args.protocol_sha)
    if protocol["limits"] != asdict(Limits()):
        raise AuditIncomplete("Protocol resource constants differ from implementation.")
    for name, digest in protocol["execution_files"].items():
        if file_hash(ROOT / name) != digest:
            raise AuditIncomplete(f"Execution identity mismatch: {name}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != protocol["recovery_checkpoint"]:
        raise AuditIncomplete("Recovery checkpoint differs.")
    if file_hash(ROOT / "data/metadata/1cls.csv") != protocol["source"]["split_sha256"]:
        raise AuditIncomplete("Official split identity differs.")
    manifest = load_category(ROOT / "data/metadata/1cls.csv", "pcb2")
    OUT.mkdir()
    write(
        "started.json",
        {
            "utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": args.protocol_sha,
            "checkpoint": head,
            "authorization": "Owner 2F0: PCB2 mechanical integrity only",
        },
    )
    write("manifest.json", manifest_to_dict(manifest))
    started, cpu = time.perf_counter(), time.process_time()
    limits = Limits()

    def budget():
        if time.perf_counter() - started > limits.seconds or peak_memory() > limits.process_bytes:
            raise AuditIncomplete("Time or memory limit reached; no clean verdict.")

    records, features, inventory = [], {}, []
    screen = {
        "complete": False,
        "reason": "not scanned",
        "comparisons": 0,
        "candidates": [],
        "required": [],
    }
    extraction = None
    failure = None
    defects = []
    source_verified = False
    by_path = {s.image_path: s for s in manifest.samples}
    try:
        source_verified = (
            file_hash(ROOT / "data/raw/VisA_20220922.tar") == protocol["source"]["archive_sha256"]
        )
        with (OUT / "access.jsonl").open("x", encoding="utf-8") as log:

            def access(phase, path):
                sample = by_path[path]
                log.write(
                    json.dumps(
                        {
                            "utc": datetime.now(UTC).isoformat(),
                            "phase": phase,
                            "path": path,
                            "official_split": sample.official_split.value,
                        }
                    )
                    + "\n"
                )
                log.flush()

            extraction = extract_category_images(
                ROOT / "data/raw/VisA_20220922.tar",
                DATA,
                manifest,
                protocol["source"]["archive_sha256"],
                access,
                budget,
            )
            write("extraction.json", extraction)
            expected = {r["path"]: r["source_sha256"] for r in extraction["images"]}
            for sample in sorted(
                manifest.samples, key=lambda s: hashlib.sha256(s.image_path.encode()).hexdigest()
            ):
                budget()
                audit_id = "pcb2-" + hashlib.sha256(sample.image_path.encode()).hexdigest()[:24]
                path = resolve_reference(DATA, sample.image_path)
                access("mechanical_decode", sample.image_path)
                record, feature = describe_image(
                    audit_id, sample.partition.value, path.read_bytes()
                )
                if record.byte_sha256 != expected[sample.image_path]:
                    raise AuditDefect("Source/prepared content changed before decode.")
                records.append(record)
                features[audit_id] = feature
                inventory.append(
                    {
                        **asdict(record),
                        "path": sample.image_path,
                        "sample_id": sample.sample_id,
                        "label": sample.label.value,
                        "width": feature["width"],
                        "height": feature["height"],
                    }
                )
            write("inventory.json", inventory)
            screen = scan_all_pairs(records, features, budget)
            budget()
            if any(file_hash(DATA / r["path"]) != r["source_sha256"] for r in extraction["images"]):
                raise AuditDefect("Prepared source changed during scan.")
            content_labels = {}
            for row in inventory:
                content_labels.setdefault(row["pixel_sha256"], set()).add(row["label"])
            if any(len(labels) > 1 for labels in content_labels.values()):
                defects.append("Identical decoded image content has contradictory official labels.")
    except (AuditIncomplete, AuditDefect, ContractError, OSError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, (AuditDefect, ContractError)):
            defects.append(str(exc))
        screen["complete"] = False
        screen["reason"] = str(exc)
        if not (OUT / "inventory.json").exists():
            write("inventory.json", inventory)
    decision = gate_decision(records, screen, source_verified=source_verified, defects=defects)
    # No non-exact image pair is visually or automatically adjudicated in this milestone.
    summary = {
        **decision,
        "protocol_sha256": args.protocol_sha,
        "failure": failure,
        "official_counts": dict(
            Counter(f"{s.official_split.value}/{s.label.value}" for s in manifest.samples)
        ),
        "decoded_images": len(records),
        "dimensions": dict(Counter(f"{r['width']}x{r['height']}" for r in inventory)),
        "decoded_test_images": sum(r.partition == "test" for r in records),
        "mask_payloads_accessed": 0,
        "images_visually_displayed": 0,
        "model_predictions": 0,
        "pcb3_payloads_accessed": 0,
        "scan_complete": screen["complete"],
        "pair_comparisons": screen["comparisons"],
        "candidate_count": len(screen["candidates"]),
        "strong_count": sum(p["strong"] for p in screen["candidates"]),
        "required_pair_count": len(screen["required"]),
        "candidate_relations": dict(Counter(p["relation"] for p in screen["candidates"])),
        "required_relations": dict(Counter(p["relation"] for p in screen["required"])),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu,
        "process_peak_bytes": peak_memory(),
        "no_preparation_or_model_change": True,
    }
    write("screen.json", screen)
    write("summary.json", summary)
    write("evidence.json", {p.name: file_hash(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "verdict",
                    "reason",
                    "decoded_images",
                    "candidate_count",
                    "strong_count",
                    "required_pair_count",
                    "wall_seconds",
                    "process_peak_bytes",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
