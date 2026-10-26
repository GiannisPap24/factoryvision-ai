"""Explicit fit or development-only evaluation. Never constructs an official test loader."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from train_cashew import peak_memory

from factoryvision.data.development import (
    DevelopmentDataset,
    DevelopmentReadGuard,
    synthetic_defect,
)
from factoryvision.data.images import make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development, score_development
from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.models.feature_memory import FeatureMemory, fit_memory, load_extractor
from factoryvision.training.checkpoint import create_run, file_sha256, write_json
from factoryvision.training.engine import parameter_sha256, seed_everything
from factoryvision.training.feature_artifacts import load_memory, save_memory

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "docs/protocols/feature-memory-development-v1.json"
RECIPE_SHA = "62ebced7c38f87a73a31099895de9120dc2320a0906bf3ed7263523f13b656fa"
RUN_ID = "cashew-feature-v1"


def timed(function, repeats=30, warmups=5):
    samples = []
    with torch.inference_mode():
        for i in range(repeats + warmups):
            start = time.perf_counter_ns()
            function()
            if i >= warmups:
                samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "repeats": repeats,
        "warmups": warmups,
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
    }


def distribution(scores):
    return {
        k: float(v)
        for k, v in zip(
            ("min", "q05", "median", "q95", "max"),
            np.quantile(scores, [0, 0.05, 0.5, 0.95, 1]),
            strict=True,
        )
    }


def bounded(loader, seconds):
    start = time.perf_counter()
    for batch in loader:
        if time.perf_counter() - start > seconds or peak_memory() > 2 * 1024**3:
            raise RuntimeError("Development resource budget exceeded.")
        yield batch


def sources():
    return {
        p.relative_to(ROOT).as_posix(): file_sha256(p)
        for p in [*sorted((ROOT / "src").rglob("*.py")), Path(__file__)]
    }


def figure(path, original, changed, mask, spatial, scale):
    heat = (spatial / scale).clamp(0, 1)
    panels = [
        original,
        changed,
        torch.stack((heat, torch.zeros_like(heat), 1 - heat)),
        mask[None].expand(3, -1, -1).float(),
    ]
    canvas = Image.new("RGB", (1024, 294), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (4, 4), "SYNTHETIC CHECKERBOARD ONLY - not a real defect performance estimate", fill="black"
    )
    for j, (title, panel) in enumerate(
        zip(
            [
                "Normal validation source",
                "Synthetic pseudo-defect",
                f"Feature RMS map 0..{scale:.4f}",
                "Synthetic mask",
            ],
            panels,
            strict=True,
        )
    ):
        array = (panel.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        canvas.paste(
            Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST), (j * 256, 38)
        )
        draw.text((j * 256 + 4, 22), title, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate"])
    args = parser.parse_args()
    if file_sha256(RECIPE_PATH) != RECIPE_SHA:
        raise ValueError("Frozen development recipe changed.")
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    fingerprint = manifest_sha256(manifest)
    if fingerprint != recipe["manifest_sha256"]:
        raise ValueError("Accepted manifest changed.")
    development = [s for s in manifest.samples if s.partition != Partition.TEST]
    guard = DevelopmentReadGuard(ROOT / "data/visa", development)
    sys.addaudithook(guard)
    # Audit only permitted normal files; no image or mask from official test is opened.
    audit = json.loads((ROOT / "data/metadata/cashew-audit.json").read_text())
    hashes = {r["sample_id"]: r["image_sha256"] for r in audit["records"]}
    for s in development:
        if file_sha256(ROOT / "data/visa" / s.image_path) != hashes[s.sample_id]:
            raise ValueError("Development image identity changed.")
    datasets = {
        p.value: DevelopmentDataset(manifest, ROOT / "data/visa", p)
        for p in (Partition.TRAIN, Partition.VALIDATION, Partition.CALIBRATION)
    }
    ids = {p: [s.sample_id for s in ds.samples] for p, ds in datasets.items()}
    if {p: len(v) for p, v in ids.items()} != {"train": 315, "validation": 67, "calibration": 68}:
        raise ValueError("Development membership changed.")
    seed_everything(42, 4)
    extractor = load_extractor(ROOT / "data/pretrained/resnet18-f37072fd.pth")
    before = parameter_sha256(extractor)
    directory = ROOT / "runs" / RUN_ID
    if args.action == "fit":
        directory = create_run(ROOT, RUN_ID)
        start = time.perf_counter()
        write_json(directory, "recipe.json", recipe)
        write_json(directory, "source-snapshot.json", sources())
        model, selection = fit_memory(
            extractor, bounded(make_loader(datasets["train"], 8), 600), ids["train"]
        )
        elapsed = time.perf_counter() - start
        save_memory(directory, model.bank, ids["train"], fingerprint, RECIPE_SHA)
        write_json(directory, "patch-selection.json", selection)
        write_json(
            directory,
            "fit.json",
            {
                "seconds": elapsed,
                "samples": 315,
                "candidate_patches": 315 * 256,
                "selected_patches": len(model.bank),
                "memory_tensor_bytes": model.bank.numel() * 4,
                "frozen_parameters": sum(p.numel() for p in extractor.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "backbone_state_before": before,
                "backbone_state_after": parameter_sha256(extractor),
                "peak_working_set_bytes": peak_memory(),
                "recipe_sha256": RECIPE_SHA,
                "lock_sha256": file_sha256(ROOT / "uv.lock"),
            },
        )
    else:
        if json.loads((directory / "source-snapshot.json").read_text()) != sources():
            raise ValueError("Execution sources changed after fit.")
        with (directory / "development-started.json").open("x", encoding="utf-8") as f:
            json.dump(
                {"scope": "normal development and synthetic only", "recipe_sha256": RECIPE_SHA}, f
            )
        bank, meta = load_memory(directory, fingerprint, RECIPE_SHA, ids["train"])
        model = FeatureMemory(extractor, bank)
        start = time.perf_counter()
        records = {
            p: score_development(model, bounded(make_loader(datasets[p], 8), 300), p, ids[p])
            for p in ("validation", "calibration")
        }
        calibration = calibrate_development(records["calibration"], ids["calibration"])
        write_json(directory, "calibration.json", calibration)
        write_json(directory, "normal-scores.json", records)
        masks, maps, pseudo = [], [], []
        for i in range(16):
            sample = datasets["validation"][i]
            changed, mask = synthetic_defect(sample["image"], sample["sample_id"])
            score, spatial = model(changed[None])
            peak = int(spatial[0].argmax())
            peak_y, peak_x = divmod(peak, 128)
            pseudo.append(
                {
                    "source_id": sample["sample_id"],
                    "synthetic": True,
                    "normal_score": records["validation"]["scores"][i],
                    "synthetic_score": float(score[0]),
                    "threshold_exceeded": float(score[0]) > calibration["threshold"],
                    "peak_inside_mask": bool(mask[peak_y, peak_x]),
                }
            )
            masks.append(mask.numpy())
            maps.append(spatial[0].numpy())
            if i < 4:
                figure(
                    directory / f"synthetic-example-{i:02d}.png",
                    sample["image"],
                    changed,
                    mask,
                    spatial[0],
                    calibration["threshold"],
                )
        write_json(directory, "synthetic-probes.json", pseudo)
        np.savez_compressed(
            directory / "synthetic-maps.npz", maps=np.stack(maps), masks=np.stack(masks)
        )
        x = datasets["validation"][0]["image"][None]
        feature_time = timed(lambda: extractor(x))
        total_time = timed(lambda: model(x))
        from factoryvision.models.feature_memory import nearest_distances

        query = extractor(x).permute(0, 2, 3, 1).reshape(-1, 384)
        search_time = timed(lambda: nearest_distances(query, bank))
        decode_time = timed(lambda: datasets["validation"][0], warmups=0)
        result = {
            "scope": "exploratory normal-development and synthetic-only evidence",
            "official_test_accessed": False,
            "recipe_sha256": RECIPE_SHA,
            "normal": {
                p: {
                    "count": len(r["scores"]),
                    "distribution": distribution(r["scores"]),
                    "exceedances": sum(s > calibration["threshold"] for s in r["scores"]),
                    "exceedance_fraction": sum(s > calibration["threshold"] for s in r["scores"])
                    / len(r["scores"]),
                }
                for p, r in records.items()
            },
            "calibration": calibration,
            "synthetic": {
                "count": 16,
                "pixel_metrics": binary_metrics(np.stack(masks).ravel(), np.stack(maps).ravel()),
                "score_increased": sum(r["synthetic_score"] > r["normal_score"] for r in pseudo),
                "threshold_exceeded": sum(r["threshold_exceeded"] for r in pseudo),
                "peak_inside_mask": sum(r["peak_inside_mask"] for r in pseudo),
                "median_score_change": float(
                    np.median([r["synthetic_score"] - r["normal_score"] for r in pseudo])
                ),
                "warning": "Synthetic evidence only; no historical model superiority claim.",
            },
            "latency": {
                "feature_extraction": feature_time,
                "nearest_search": search_time,
                "preloaded_total": total_time,
                "decode_resize": decode_time,
            },
            "elapsed_seconds": time.perf_counter() - start,
            "peak_working_set_bytes": peak_memory(),
            "backbone_state_before": before,
            "backbone_state_after": parameter_sha256(extractor),
            "memory_artifact_sha256": meta["memory_sha256"],
        }
        write_json(directory, "development.json", result)
        print(json.dumps(result, indent=2))
    if before != parameter_sha256(extractor) or peak_memory() > 2 * 1024**3:
        raise RuntimeError("Frozen backbone or memory budget violated.")
    write_json(
        directory,
        f"{args.action}-reads.json",
        {
            "scope": "450 permitted normal development files",
            "opened_paths": sorted(guard.opened),
            "unique_files": len(guard.opened),
            "official_test_files": 0,
        },
    )
    write_json(
        directory,
        f"{args.action}-hashes.json",
        {
            p.name: file_sha256(p)
            for p in directory.iterdir()
            if p.is_file() and not p.name.endswith("-hashes.json")
        },
    )
    print(f"Completed {args.action}: {directory.name}; official test image/mask reads: 0")


if __name__ == "__main__":
    main()
