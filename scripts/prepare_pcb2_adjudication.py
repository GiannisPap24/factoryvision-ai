"""Explicit one-shot 2F0-A evidence/package preparation; no human decisions invented."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from audit_pcb2_integrity import peak_memory

from factoryvision.data.adjudication import (
    adjudicate,
    bind_queue,
    blind_manifest,
    decode_verified,
    opaque_id,
    pair_png,
    structural_evidence,
)
from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.pretraining_integrity import file_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/protocols/pcb2-adjudication-v1.json"
OUT = "reports/pcb2-adjudication-v1"
DATA = "data/pcb2-integrity-v1"
KEY = "reports/pcb2-adjudication-preflight-v1/blinding-key.bin"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(root, expected_sha):
    if file_hash(root / PROTOCOL) != expected_sha:
        raise ValueError("Frozen adjudication protocol mismatch.")
    protocol = read(root / PROTOCOL)
    if protocol["method"] != "pcb2-adjudication-v1":
        raise ValueError("Unsupported method.")
    for name, digest in protocol["bound_files"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"Bound evidence changed: {name}")
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        != protocol["checkpoint"]
    ):
        raise ValueError("Checkpoint mismatch.")
    secret = (root / KEY).read_bytes()
    if hashlib.sha256(secret).hexdigest() != protocol["blinding_key_sha256"]:
        raise ValueError("Blinding-key mismatch.")
    inventory = read(root / "reports/pcb2-integrity-v1/inventory.json")
    required = read(root / "reports/pcb2-integrity-v1/screen.json")["required"]
    queue = bind_queue(required, inventory, secret)
    if (
        len(queue) != protocol["queue_count"]
        or sum(p["cross_partition"] for p in queue) != protocol["cross_count"]
    ):
        raise ValueError("Queue membership/count mismatch.")
    by_id = {opaque_id(r["path"]): r for r in inventory}
    selected = {p[k] for p in queue for k in ("a", "b")}
    # Check all required source bytes before creating any review pixels.
    for key in sorted(selected):
        row = by_id[key]
        if file_hash(resolve_reference(root / DATA, row["path"])) != row["byte_sha256"]:
            raise ValueError("Prepared source identity changed.")
    out = root / OUT
    out.mkdir()
    restricted, blind = out / "restricted", out / "reviewer"
    restricted.mkdir()
    blind.mkdir()
    start, cpu = time.perf_counter(), time.process_time()
    write(
        out / "started.json",
        {"utc": datetime.now(UTC).isoformat(), "protocol_sha256": expected_sha},
    )
    write(restricted / "queue.json", queue)
    write(restricted / "inventory.json", [by_id[k] for k in sorted(selected)])
    evidence, decisions, generated = [], [], []
    total_bytes = 0
    failure = None
    result = adjudicate(queue, inventory, [])
    with (restricted / "access.jsonl").open("x", encoding="utf-8") as log:
        try:
            for pair in queue:
                if time.perf_counter() - start > 600 or peak_memory() > 2 * 1024**3:
                    raise ValueError("Resource cap: review incomplete.")
                images = []
                for side in ("left", "right"):
                    row = by_id[pair[side]]
                    log.write(
                        json.dumps(
                            {
                                "utc": datetime.now(UTC).isoformat(),
                                "pair_id": pair["pair_id"],
                                "sample_id": row["sample_id"],
                                "path": row["path"],
                                "partition": row["partition"],
                                "purpose": "mechanical_ssim_and_review_render",
                                "visually_displayed": False,
                            }
                        )
                        + "\n"
                    )
                    log.flush()
                    path = resolve_reference(root / DATA, row["path"])
                    images.append(decode_verified(path.read_bytes(), row))
                measurements = structural_evidence(*images)
                png = pair_png(*images)
                total_bytes += len(png)
                if total_bytes > 512 * 1024**2:
                    raise ValueError("Review-image byte cap exceeded.")
                target = blind / (pair["pair_id"] + ".png")
                with target.open("xb") as stream:
                    stream.write(png)
                generated.append(
                    {"pair_id": pair["pair_id"], "sha256": file_hash(target), "bytes": len(png)}
                )
                a, b = by_id[pair["a"]], by_id[pair["b"]]
                exact = any(a[k] == b[k] for k in ("byte_sha256", "pixel_sha256"))
                decision = {
                    "pair_id": pair["pair_id"],
                    "decision": "CONFIRMED_DUPLICATE" if exact else "UNRESOLVED",
                    "evidence": "Verified identical original content."
                    if exact
                    else (
                        "Original byte and pixel hashes differ. Supplemental SSIM/translation "
                        "do not establish capture or physical-item identity. Independent blinded "
                        "review or acquisition provenance is unavailable."
                    ),
                    "reviewer_type": "technical_automation",
                    "reviewer_id": None,
                    "independent_human_review": False,
                    "confidence": "high" if exact else "insufficient",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
                decisions.append(decision)
                evidence.append(
                    {
                        **pair,
                        "technical": measurements,
                        "source_a": a,
                        "source_b": b,
                        "decision": decision,
                    }
                )
                result = adjudicate(queue, inventory, decisions)
                if result["verdict"] == "BLOCKED":
                    break
        except (ValueError, OSError) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            result = adjudicate(queue, inventory, decisions, provenance_valid=False)
    rendered = [p for p in queue if p["pair_id"] in {r["pair_id"] for r in generated}]
    write(blind / "pairs.json", blind_manifest(rendered))
    write(
        blind / "response-template.json",
        [
            {
                "pair_id": p["pair_id"],
                "decision": None,
                "evidence": "",
                "landmarks": [],
                "confidence": None,
                "reviewer_id": None,
                "timestamp": None,
            }
            for p in blind_manifest(rendered)
        ],
    )
    (blind / "README.txt").write_text(protocol["reviewer_instructions"], encoding="utf-8")
    # Avoid leaking private processing priority through creation/modification timestamps.
    for path in blind.iterdir():
        os.utime(path, (946684800, 946684800))
    # Hand off only this archive: filesystem birth times on Windows are not anonymized.
    with zipfile.ZipFile(
        out / "reviewer-bundle.zip", "x", compression=zipfile.ZIP_STORED
    ) as bundle:
        for path in sorted(blind.iterdir(), key=lambda p: p.name):
            info = zipfile.ZipInfo(path.name, date_time=(2000, 1, 1, 0, 0, 0))
            bundle.writestr(info, path.read_bytes())
    write(restricted / "pair-evidence.json", evidence)
    write(restricted / "decisions.json", decisions)
    write(restricted / "rendered.json", generated)
    summary = {
        **result,
        "failure": failure,
        "technical_pairs_processed": len(evidence),
        "review_images_generated": len(generated),
        "review_image_bytes": total_bytes,
        "processed_relations": dict(Counter(p["relation"] for p in evidence)),
        "human_reviews": 0,
        "agent_visual_reviews": 0,
        "mask_accesses": 0,
        "model_predictions": 0,
        "pcb3_payload_accesses": 0,
        "wall_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
        "peak_bytes": peak_memory(),
    }
    write(out / "summary.json", summary)
    write(
        out / "evidence.json",
        {
            p.relative_to(out).as_posix(): file_hash(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"groups", "unresolved"}}, indent=2
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-sha", required=True)
    prepare(ROOT, parser.parse_args().protocol_sha)
