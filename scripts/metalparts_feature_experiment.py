"""One sealed normal fit, normal validation and exploratory source comparison; no targets."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from train_cashew import peak_memory

from factoryvision.data.metalparts import REVISION, MetalPartsSample, MetalPartsSource
from factoryvision.data.metalparts_development import DevelopmentSample
from factoryvision.data.metalparts_diagnostic import DiagnosticReadGuard, source_pair
from factoryvision.domain.contracts import Label, Partition
from factoryvision.evaluation.metalparts import ranking_metrics
from factoryvision.evaluation.metalparts_comparison import LABEL, DistancePixelRanking
from factoryvision.inference.service import InspectionResult
from factoryvision.inference.visualization import panels
from factoryvision.models.feature_memory import WEIGHTS_SHA256, FeatureMemory, load_extractor
from factoryvision.models.metalparts_memory import (
    FittingTiles,
    ResourceBudget,
    fit_native_memory,
    native_memory_map,
)
from factoryvision.training.checkpoint import create_run, file_sha256, write_json
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.metalparts_memory import identity, load_memory, save_memory

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/metalparts-3f-v1"
PROTOCOL = ROOT / "docs/metalparts-feature-memory-protocol.md"
SEAL = REPORT / "seal.json"
RUN = ROOT / "runs/metalparts-feature-v1"
NORMAL = ROOT / "runs/metalparts-feature-normal-v1"
COMPARISON = ROOT / "runs/metalparts-feature-comparison-v1"
OLD = ROOT / "runs/metalparts-source-diagnostic-v1"
WEIGHTS = ROOT / "data/pretrained/resnet18-f37072fd.pth"
SOURCE = ROOT / "data/raw/robustad" / REVISION
OLD_MANIFEST = ROOT / "reports/metalparts-3e-v1/diagnostic-manifest.json"
VERIFICATION = ROOT / "reports/robustad-3c-v1/integrity/source-verification-final.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample(row):
    return MetalPartsSample(
        MetalPartsSource(),
        "train",
        row["file_name"],
        Label.ANOMALY if row["label"] else Label.NORMAL,
        row["mask_path"],
        row["width"],
        row["height"],
    )


def fitting_records(inventory):
    return tuple(
        DevelopmentSample(sample(row), row["image_sha256"], row["pixel_sha256"], Partition.TRAIN)
        for row in inventory
        if row["partition"] == "train"
    )


def seal():
    if SEAL.exists():
        raise FileExistsError("Feature experiment already sealed.")
    if (
        file_sha256(OLD_MANIFEST)
        != "9811dc9222b6241fb284bbfe27b8286905d26b398e7c38c37e2dfc603dcc48d4"
    ):
        raise ValueError("3E manifest drift.")
    manifest = read(OLD_MANIFEST)
    records = fitting_records(manifest["inventory"])
    if (
        len(records) != 144
        or sum(r["partition"] == "validation" for r in manifest["inventory"]) != 36
    ):
        raise ValueError("Accepted fitting/validation inventory mismatch.")
    tiles = FittingTiles(SOURCE, records)
    if len(tiles.ids) != 1728 or file_sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError("Tile inventory or trusted pretrained identity mismatch.")
    protocol_sha = file_sha256(PROTOCOL)
    expected = identity(
        MetalPartsSource(), manifest["model"]["identity"]["split_sha256"], protocol_sha, tiles.ids
    )
    code = [p for p in (ROOT / "src/factoryvision").rglob("*.py") if "app" not in p.parts]
    code += [
        Path(__file__),
        ROOT / "scripts/train_cashew.py",
        PROTOCOL,
        WEIGHTS,
        OLD_MANIFEST,
        VERIFICATION,
        ROOT / "reports/metalparts-3d-v1/development.json",
        OLD / "scores.json",
        OLD / "metrics.json",
        OLD / "pixel-histograms.npz",
    ]
    old_scores = read(OLD / "scores.json")
    code += [OLD / r["map"] for r in old_scores]
    payload = {
        "label": LABEL,
        "checkpoint_3e": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "identity": expected,
        "inventory": manifest["inventory"],
        "selected_ids": manifest["selected_ids"],
        "preview_ids": manifest["preview_ids"],
        "files": {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in code},
        "fitting_parent_ids": [r.sample_id for r in records],
        "fit_tiles": len(tiles.ids),
        "planned_descriptors": len(tiles.ids) * 4,
        "threshold": None,
    }
    write_json(REPORT, "seal.json", payload)
    print(
        json.dumps(
            {
                "seal_sha256": file_sha256(SEAL),
                "fit_parents": len(records),
                "fit_tiles": len(tiles.ids),
                "bank_descriptors": len(tiles.ids) * 4,
            }
        )
    )


def context():
    sealed = read(SEAL)
    for name, digest in sealed["files"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Sealed source/input drift: {name}")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return sealed, read(VERIFICATION)["files"]


def guard_for(rows):
    references = [p for r in rows for p in (r["image_path"], r["mask_path"]) if p]
    guard = DiagnosticReadGuard(SOURCE, references)
    sys.addaudithook(guard)
    return guard


def fit():
    sealed, _ = context()
    rows = [r for r in sealed["inventory"] if r["partition"] == "train"]
    guard = guard_for(rows)
    tiles = FittingTiles(SOURCE, fitting_records(rows))
    if tiles.ids != sealed["identity"]["fit_tile_ids"]:
        raise ValueError("Fitting tile identity drift.")
    create_run(ROOT, RUN.name)
    write_json(RUN, "started.json", {"seal_sha256": file_sha256(SEAL), "label": LABEL})
    budget = ResourceBudget(900, peak_memory)
    start, cpu = time.perf_counter(), time.process_time()
    extractor = load_extractor(WEIGHTS)
    before = parameter_sha256(extractor)
    model, trace = fit_native_memory(extractor, tiles, budget)
    fit_seconds = time.perf_counter() - start
    replay, replayed_trace = fit_native_memory(extractor, tiles, budget)
    if (
        not torch.equal(model.bank, replay.bank)
        or trace != replayed_trace
        or parameter_sha256(extractor) != before
    ):
        raise ValueError("Determinism or frozen-backbone verification failed.")
    meta = save_memory(RUN, model.bank, sealed["identity"])
    loaded, _ = load_memory(RUN, sealed["identity"])
    if not torch.equal(model.bank, loaded):
        raise ValueError("Bank reload mismatch.")
    write_json(RUN, "selection.json", trace)
    write_json(
        RUN,
        "fit.json",
        {
            "label": LABEL,
            "fit_seconds": fit_seconds,
            "total_with_replay_seconds": time.perf_counter() - start,
            "cpu_seconds": time.process_time() - cpu,
            "peak_working_set_bytes": peak_memory(),
            "backbone_before": before,
            "backbone_after": parameter_sha256(extractor),
            "bank_sha256": meta["memory_sha256"],
            "bank_bytes": (RUN / "memory.pt").stat().st_size,
            "tensor_shape": list(model.bank.shape),
            "fitting_parent_ids": sealed["fitting_parent_ids"],
            "exact_determinism": True,
            "exact_reload": True,
            "source_reads": sorted(guard.opened),
            "gradient_training": False,
            "threshold": None,
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in read(RUN / "fit.json").items()
                if k not in ("source_reads", "fitting_parent_ids")
            }
        )
    )


def loaded_model(sealed):
    bank, meta = load_memory(RUN, sealed["identity"])
    return FeatureMemory(load_extractor(WEIGHTS), bank), meta


def write_score(destination, row, spatial, score, seconds):
    path = destination / (row["sample_id"] + ".npy")
    np.save(path, spatial, allow_pickle=False)
    return {
        "sample_id": row["sample_id"],
        "anomaly": row["label"],
        "score": score,
        "seconds": seconds,
        "map": path.relative_to(ROOT).as_posix(),
        "map_sha256": file_sha256(path),
        "label": LABEL,
    }


def normal():
    sealed, verified = context()
    rows = [r for r in sealed["inventory"] if r["partition"] == "validation"]
    guard = guard_for(rows)
    create_run(ROOT, NORMAL.name)
    write_json(NORMAL, "started.json", {"seal_sha256": file_sha256(SEAL), "label": LABEL})
    model, meta = loaded_model(sealed)
    budget = ResourceBudget(900, peak_memory)
    results = []
    for i, row in enumerate(rows):
        image, _ = source_pair(SOURCE, sample(row), verified)
        start = time.perf_counter()
        spatial, score = native_memory_map(model, image, budget)
        seconds = time.perf_counter() - start
        if i == 0:
            independently_loaded, _ = loaded_model(sealed)
            other, other_score = native_memory_map(independently_loaded, image, budget)
            if not np.array_equal(other, spatial) or score != other_score:
                raise ValueError("Normal independent reload differs.")
        results.append(write_score(NORMAL, row, spatial, score, seconds))
        print(json.dumps({"normal_completed": len(results), "seconds": seconds}), flush=True)
    write_json(NORMAL, "scores.json", results)
    write_json(
        NORMAL,
        "diagnostics.json",
        {
            "label": LABEL,
            "bank_sha256": meta["memory_sha256"],
            "peak_working_set_bytes": peak_memory(),
            "exact_first_reload": True,
            "distribution": np.quantile(
                [r["score"] for r in results], [0, 0.25, 0.5, 0.75, 1]
            ).tolist(),
            "source_reads": sorted(guard.opened),
            "threshold": None,
        },
    )


def compare():
    sealed, verified = context()
    normals = {r["sample_id"]: r for r in read(NORMAL / "scores.json")}
    expected_normal = {
        r["sample_id"] for r in sealed["inventory"] if r["partition"] == "validation"
    }
    if set(normals) != expected_normal or read(NORMAL / "diagnostics.json")[
        "bank_sha256"
    ] != file_sha256(RUN / "memory.pt"):
        raise ValueError("Normal cache identity mismatch.")
    rows_by_id = {r["sample_id"]: r for r in sealed["inventory"]}
    rows = [rows_by_id[key] for key in sealed["selected_ids"]]
    guard = guard_for(rows)
    model, meta = loaded_model(sealed)
    create_run(ROOT, COMPARISON.name)
    write_json(
        COMPARISON,
        "started.json",
        {"seal_sha256": file_sha256(SEAL), "bank_sha256": meta["memory_sha256"], "label": LABEL},
    )
    budget = ResourceBudget(1800, peak_memory)
    start, cpu = time.perf_counter(), time.process_time()
    histogram, ae_hist = DistancePixelRanking(), DistancePixelRanking()
    results, previews, per_image = [], [], []
    old_scores = {r["sample_id"]: r for r in read(OLD / "scores.json")}
    for row in rows:
        budget()
        image, mask = source_pair(SOURCE, sample(row), verified)
        if row["sample_id"] in normals:
            result = normals[row["sample_id"]]
            if file_sha256(ROOT / result["map"]) != result["map_sha256"]:
                raise ValueError("Normal cache map drift.")
            spatial = np.load(ROOT / result["map"], allow_pickle=False)
        else:
            tick = time.perf_counter()
            spatial, score = native_memory_map(model, image, budget)
            result = write_score(COMPARISON, row, spatial, score, time.perf_counter() - tick)
        histogram.add(spatial, mask)
        old = old_scores[row["sample_id"]]
        ae_map = np.load(OLD / old["map"], allow_pickle=False)
        ae_hist.add(ae_map, mask)
        individual = DistancePixelRanking()
        individual.add(spatial, mask)
        per_image.append(
            {
                "sample_id": row["sample_id"],
                "anomaly": row["label"],
                "pixel": individual.result(),
                "peak_inside_mask": bool(mask[np.unravel_index(np.argmax(spatial), spatial.shape)]),
            }
        )
        results.append(result)
        if row["sample_id"] in sealed["preview_ids"]:
            view = panels(
                InspectionResult(
                    "metalparts_feature_memory",
                    "Feature-Memory Detector",
                    "Maximum descriptor RMS distance",
                    result["score"],
                    None,
                    None,
                    spatial,
                    np.asarray(image),
                    None,
                    None,
                    0,
                    result["seconds"] * 1000,
                    0,
                    result["seconds"] * 1000,
                    meta["memory_sha256"],
                )
            )
            assets = {}
            for key in ("original", "heatmap", "overlay"):
                path = COMPARISON / f"{row['sample_id']}-{key}.png"
                view[key].save(path)
                assets[key] = {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            previews.append(
                {
                    "sample_id": row["sample_id"],
                    "anomaly": row["label"],
                    "score": result["score"],
                    "ceiling": view["ceiling"],
                    "assets": assets,
                }
            )
        print(
            json.dumps(
                {
                    "comparison_completed": len(results),
                    "of": len(rows),
                    "cached_normal": not row["label"],
                    "seconds": result["seconds"],
                }
            ),
            flush=True,
        )
    old_metrics = read(OLD / "metrics.json")
    for key in ("positive", "negative", "auroc", "average_precision", "prevalence"):
        if ae_hist.result()[key] != old_metrics["pixel"][key]:
            raise ValueError("Common quantization does not reproduce original AE metrics.")
    feature = {
        "image": ranking_metrics([r["anomaly"] for r in results], [r["score"] for r in results]),
        "pixel": histogram.result(),
        "latency_seconds": dict(
            zip(
                ["min", "median", "p95", "max"],
                np.quantile([r["seconds"] for r in results], [0, 0.5, 0.95, 1]).tolist(),
                strict=True,
            )
        ),
        "peak_working_set_bytes": peak_memory(),
        "bank_bytes": (RUN / "memory.pt").stat().st_size,
        "backbone_bytes": WEIGHTS.stat().st_size,
        "bank_sha256": meta["memory_sha256"],
        "score_distributions": {},
    }
    for value, name in [(0, "validation_normal"), (1, "source_anomaly")]:
        feature["score_distributions"][name] = dict(
            zip(
                ["min", "q25", "median", "q75", "max"],
                np.quantile(
                    [r["score"] for r in results if r["anomaly"] == value], [0, 0.25, 0.5, 0.75, 1]
                ).tolist(),
                strict=True,
            )
        )
    metrics = {
        "label": LABEL,
        "seal_sha256": file_sha256(SEAL),
        "autoencoder": {
            k: old_metrics[k]
            for k in ("image", "pixel", "latency_seconds", "peak_working_set_bytes")
        },
        "feature_memory": feature,
        "ae_cached_pixel_metrics_exact": True,
        "total_comparison_seconds": time.perf_counter() - start,
        "comparison_cpu_seconds": time.process_time() - cpu,
        "source_reads": sorted(guard.opened),
        "preview_assets": previews,
        "threshold": None,
    }
    write_json(COMPARISON, "scores.json", results)
    write_json(COMPARISON, "per-image-localization.json", per_image)
    write_json(COMPARISON, "metrics.json", metrics)
    np.savez(
        COMPARISON / "pixel-histograms.npz",
        positive=histogram.positive,
        negative=histogram.negative,
    )
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k not in ("preview_assets", "source_reads")},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "fit", "normal", "compare"))
    args = parser.parse_args()
    {"seal": seal, "fit": fit, "normal": normal, "compare": compare}[args.mode]()
