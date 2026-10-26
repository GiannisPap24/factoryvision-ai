"""Explicit source-only 3E audit/seal and one frozen exploratory inference pass."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from train_cashew import peak_memory

from factoryvision.data.metalparts import REVISION, MetalPartsSample, MetalPartsSource
from factoryvision.data.metalparts_development import DevelopmentSample, PatchConfig
from factoryvision.data.metalparts_diagnostic import (
    PREFIX,
    DiagnosticReadGuard,
    audit_source,
    source_pair,
)
from factoryvision.domain.contracts import Label, Partition
from factoryvision.evaluation.metalparts import PixelRanking, ranking_metrics
from factoryvision.inference.metalparts import (
    DIAGNOSTIC_LABEL,
    MetalPartsService,
    reconstruct_native,
)
from factoryvision.inference.service import InspectionResult
from factoryvision.inference.visualization import panels
from factoryvision.training.checkpoint import create_run, file_sha256, write_json
from factoryvision.training.metalparts import native_diagnostic

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/metalparts-3e-v1"
RUN = ROOT / "runs/metalparts-source-diagnostic-v1"
PROTOCOL = ROOT / "docs/metalparts-development-diagnostic-protocol.md"
SOURCE_EVIDENCE = ROOT / "reports/robustad-3c-v1/integrity/source-verification-final.json"
DEVELOPMENT = ROOT / "reports/metalparts-3d-v1/development.json"
MANIFEST = REPORT / "diagnostic-manifest.json"


def source_inputs():
    if (
        file_sha256(SOURCE_EVIDENCE)
        != "84f2716eaf3afe766d34e569d4fb917abcd383201ffc5a8dc5f5ca5908b672fb"
    ):
        raise ValueError("3C source evidence drift.")
    if (
        file_sha256(DEVELOPMENT)
        != "cb3221c593e7205d7ea3da7bda3cbad28a208bbdf81dc054b326bcf473ecf69c"
    ):
        raise ValueError("3D split evidence drift.")
    root = ROOT / "data/raw/robustad" / REVISION
    verified = json.loads(SOURCE_EVIDENCE.read_text())["files"]
    guard = DiagnosticReadGuard(root, [p for p in verified if p.startswith(PREFIX)])
    sys.addaudithook(guard)
    return root, verified, guard


def seal():
    if MANIFEST.exists():
        raise FileExistsError("Diagnostic manifest already sealed.")
    root, verified, guard = source_inputs()
    development = json.loads(DEVELOPMENT.read_text())
    samples, inventory = audit_source(root, verified, development)
    assert len(samples) == 87 and len(inventory) == 231
    assert sum(r["partition"] == "train" for r in inventory) == 144
    assert sum(r["partition"] == "validation" for r in inventory) == 36
    assert sum(r["label"] for r in inventory) == 51
    engine = MetalPartsService(ROOT)
    assert engine.status()["available"]
    # Preserve former all-domain findings as metadata only, never open its images/masks.
    audit_path = ROOT / "reports/robustad-3c-v1/integrity/audit.json"
    audit = json.loads(audit_path.read_text())
    selected = [s.sample_id for s in samples]
    previews = [s.sample_id for s in samples if s.label == Label.NORMAL][:1]
    previews += [s.sample_id for s in samples if s.label == Label.ANOMALY][:3]
    code = [p for p in (ROOT / "src/factoryvision").rglob("*.py") if "app" not in p.parts]
    code += [
        Path(__file__),
        ROOT / "scripts/train_cashew.py",
        PROTOCOL,
        ROOT / "src/factoryvision/inference/metalparts_artifacts.json",
        ROOT / "src/factoryvision/inference/metalparts_protected_inputs.json",
        SOURCE_EVIDENCE,
        DEVELOPMENT,
        audit_path,
    ]
    manifest = {
        "label": DIAGNOSTIC_LABEL,
        "checkpoint_3d": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "created_before_anomaly_scoring": True,
        "revision": REVISION,
        "model": engine.registry,
        "protocol_sha256": file_sha256(PROTOCOL),
        "files": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in code},
        "inventory": inventory,
        "selected_ids": selected,
        "preview_ids": previews,
        "source_reads": sorted(guard.opened),
        "historical_relationships": {
            k: audit[k]
            for k in (
                "cross_domain_basename_groups",
                "train_test_near_candidates",
                "confirmed_related_captures",
                "capture_provenance",
                "limitations",
            )
        },
        "threshold": None,
    }
    write_json(REPORT, MANIFEST.name, manifest)
    print(
        json.dumps(
            {
                "source_records": len(inventory),
                "selected": len(samples),
                "source_files_opened": len(guard.opened),
                "manifest_sha256": file_sha256(MANIFEST),
            }
        )
    )


def evaluate():
    manifest = json.loads(MANIFEST.read_text())
    for name, expected in manifest["files"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Sealed execution/input drift: {name}")
    engine = MetalPartsService(ROOT)
    if engine.registry != manifest["model"]:
        raise ValueError("Model catalog drift.")
    root, verified, guard = source_inputs()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = engine._load()
    create_run(ROOT, RUN.name)
    write_json(
        RUN, "started.json", {"manifest_sha256": file_sha256(MANIFEST), "label": DIAGNOSTIC_LABEL}
    )
    selected = [r for r in manifest["inventory"] if r["sample_id"] in manifest["selected_ids"]]
    histogram, results, preview_assets = PixelRanking(), [], []
    start, cpu_start = time.perf_counter(), time.process_time()
    replayed = False
    for record in selected:
        sample = MetalPartsSample(
            MetalPartsSource(),
            "train",
            record["file_name"],
            Label.ANOMALY if record["label"] else Label.NORMAL,
            record["mask_path"],
            record["width"],
            record["height"],
        )
        image, mask = source_pair(root, sample, verified)
        tick = time.perf_counter()
        reconstructed, spatial, score = reconstruct_native(model, image)
        seconds = time.perf_counter() - tick
        if not replayed and sample.label == Label.NORMAL:
            previous = native_diagnostic(
                model,
                root,
                DevelopmentSample(
                    sample, record["image_sha256"], record["pixel_sha256"], Partition.VALIDATION
                ),
                PatchConfig(),
            )
            for a, b in zip(previous, (reconstructed, spatial, score), strict=True):
                np.testing.assert_array_equal(a, b)
            replayed = True
        histogram.add(spatial, mask)
        map_name = sample.sample_id + ".npy"
        np.save(RUN / map_name, spatial, allow_pickle=False)
        results.append(
            {
                "label": DIAGNOSTIC_LABEL,
                "sample_id": sample.sample_id,
                "anomaly": record["label"],
                "score": score,
                "seconds": seconds,
                "map": map_name,
                "map_sha256": file_sha256(RUN / map_name),
                "positive_pixels": int(mask.sum()),
                "native_pixels": int(mask.size),
            }
        )
        if sample.sample_id in manifest["preview_ids"]:
            result = InspectionResult(
                "metalparts_autoencoder",
                "Experimental MetalParts",
                "Native MSE",
                score,
                None,
                None,
                spatial,
                np.asarray(image),
                reconstructed,
                None,
                0,
                seconds * 1000,
                0,
                seconds * 1000,
                engine.registry["assets"]["checkpoint"]["sha256"],
            )
            view = panels(result)
            assets = {}
            for name in ("original", "reconstruction", "heatmap", "overlay"):
                path = RUN / f"{sample.sample_id}-{name}.png"
                view[name].save(path)
                assets[name] = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            path = RUN / f"{sample.sample_id}-mask.png"
            Image.fromarray(mask * 255).resize(
                view["original"].size, Image.Resampling.NEAREST
            ).save(path)
            assets["mask"] = {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            preview_assets.append(
                {
                    "sample_id": sample.sample_id,
                    "anomaly": record["label"],
                    "score": score,
                    "ceiling": view["ceiling"],
                    "assets": assets,
                }
            )
        print(
            json.dumps({"completed": len(results), "of": len(selected), "seconds": seconds}),
            flush=True,
        )
    metrics = {
        "label": DIAGNOSTIC_LABEL,
        "manifest_sha256": file_sha256(MANIFEST),
        "image": ranking_metrics([r["anomaly"] for r in results], [r["score"] for r in results]),
        "pixel": histogram.result(),
        "distributions": {},
        "threshold": None,
        "latency_seconds": {},
        "total_seconds": time.perf_counter() - start,
        "process_cpu_seconds": time.process_time() - cpu_start,
        "peak_working_set_bytes": peak_memory(),
        "exact_3d_native_replay": replayed,
        "source_reads": sorted(guard.opened),
        "preview_assets": preview_assets,
    }
    for label, name in [(0, "validation_normal"), (1, "source_anomaly")]:
        scores = [r["score"] for r in results if r["anomaly"] == label]
        metrics["distributions"][name] = dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                np.quantile(scores, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                strict=True,
            )
        )
    metrics["latency_seconds"] = dict(
        zip(
            ["min", "median", "p95", "max"],
            np.quantile([r["seconds"] for r in results], [0, 0.5, 0.95, 1]).tolist(),
            strict=True,
        )
    )
    np.savez(RUN / "pixel-histograms.npz", positive=histogram.positive, negative=histogram.negative)
    write_json(RUN, "scores.json", results)
    write_json(RUN, "metrics.json", metrics)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in ("label", "image", "pixel", "total_seconds", "peak_working_set_bytes")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "evaluate"))
    args = parser.parse_args()
    seal() if args.mode == "seal" else evaluate()
