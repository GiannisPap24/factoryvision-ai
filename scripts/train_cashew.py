"""Explicit, bounded CPU experiment. Never scores official test images."""

import argparse
import ctypes
import json
import platform
import subprocess
import time
from ctypes import wintypes
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.reconstruction import calibrate_normal
from factoryvision.models.autoencoder import ARCHITECTURE, PREPROCESSING, ConvAutoencoder
from factoryvision.training.checkpoint import (
    create_run,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from factoryvision.training.config import TrainingConfig
from factoryvision.training.engine import (
    EarlyStopping,
    parameter_sha256,
    score_normals,
    seed_everything,
    train_epoch,
)


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
        raise OSError("Cannot measure process memory.")
    return info.peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--smoke", action="store_true", help="Two real training batches, one epoch."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = TrainingConfig(max_epochs=1 if args.smoke else 12)
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    fingerprint = manifest_sha256(manifest)
    gate = json.loads((root / "reports/cashew-leakage-review.json").read_text())
    if gate["manifest_sha256"] != fingerprint or gate["confirmed_leakage"]:
        raise ValueError("Dataset gate is incompatible or confirmed leakage requires review.")
    if gate["screen_sha256"] != file_sha256(root / "reports/cashew-leakage-screen.json"):
        raise ValueError("Leakage screen changed after review.")
    # Identity verification only; official test images never enter the model or score pipeline.
    for sample in manifest.samples:
        if file_sha256(root / "data/visa" / sample.image_path) != sample.content_sha256:
            raise ValueError("Image bytes differ from the accepted manifest.")
    directory = create_run(root, args.run_id)
    seed_everything(config.seed, config.threads)
    settings = ImageSettings(config.resolution, config.resolution)
    datasets = {
        p: ManifestDataset(manifest, root / "data/visa", p, settings)
        for p in (Partition.TRAIN, Partition.VALIDATION, Partition.CALIBRATION)
    }
    train = datasets[Partition.TRAIN]
    validation = datasets[Partition.VALIDATION]
    if args.smoke:
        train, validation = Subset(train, range(32)), Subset(validation, range(16))
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    validation_loader = DataLoader(
        validation,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    model = ConvAutoencoder(config.resolution)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_hash = parameter_sha256(model)
    source_files = sorted([*root.glob("src/**/*.py"), Path(__file__)])
    metadata = {
        "configuration": config.to_dict(),
        "mode": "smoke" if args.smoke else "full",
        "architecture": ARCHITECTURE,
        "preprocessing": PREPROCESSING,
        "parameters": sum(p.numel() for p in model.parameters()),
        "latent_shape": [16, config.resolution // 16, config.resolution // 16],
        "loss": "mean squared RGB reconstruction error over all channels and pixels",
        "optimizer": "Adam",
        "device": "cpu",
        "workers": 0,
        "training_shuffle": True,
        "validation_shuffle": False,
        "manifest_sha256": fingerprint,
        "parent_sha256": manifest.split_provenance.parent_manifest_sha256,
        "assignments_sha256": manifest.split_provenance.assignments_sha256,
        "leakage_screen_sha256": gate["screen_sha256"],
        "leakage_review_sha256": file_sha256(root / "reports/cashew-leakage-review.json"),
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.system(),
            "machine": platform.machine(),
        },
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "git_checkpoint": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {p.relative_to(root).as_posix(): file_sha256(p) for p in source_files},
        "initial_parameter_sha256": initial_hash,
        "calibration_rule_predeclared": "q=.95, linear interpolation, score > threshold",
        "test_metrics": "NOT YET EVALUATED",
    }
    write_json(directory, "experiment.json", metadata)
    started, cpu_started = time.perf_counter(), time.process_time()
    history = []
    stop = EarlyStopping(config.patience, config.min_delta)
    try:
        initial_validation = score_normals(model, validation_loader, "validation")
        write_json(directory, "initial-validation.json", initial_validation)
        for epoch in range(1, config.max_epochs + 1):
            epoch_started = time.perf_counter()
            fitted = train_epoch(
                model, optimizer, train_loader, deadline=started + config.max_seconds
            )
            validated = score_normals(model, validation_loader, "validation")
            changed = parameter_sha256(model) != initial_hash
            if not changed:
                raise ValueError("Optimizer did not change parameters.")
            improved, stopped = stop.update(validated["mean"], epoch)
            row = {
                "epoch": epoch,
                "train_loss": fitted["loss"],
                "validation_loss": validated["mean"],
                "train_samples": fitted["samples"],
                "validation_samples": len(validated["scores"]),
                "optimizer_steps": fitted["steps"],
                "parameters_changed": changed,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "peak_working_set_bytes": peak_memory(),
            }
            history.append(row)
            write_json(directory, "history.json", history)
            write_json(directory, "last-training-order.json", fitted["sample_ids"])
            print(json.dumps(row), flush=True)
            if row["peak_working_set_bytes"] > 2 * 1024**3:
                raise MemoryError("Process peak exceeded 2 GiB experiment budget.")
            if improved:
                save_checkpoint(directory, model, fingerprint, epoch, validated["mean"])
                write_json(directory, "best-validation.json", validated)
            if stopped:
                break
        training_seconds = time.perf_counter() - started
        training_cpu_seconds = time.process_time() - cpu_started
        reloaded, checkpoint = load_checkpoint(directory, fingerprint, config.resolution)
        repeated = score_normals(reloaded, validation_loader, "validation")
        expected = json.loads((directory / "best-validation.json").read_text())
        if repeated != expected:
            raise ValueError("Reloaded checkpoint validation outputs differ.")
        if not args.smoke:
            calibration = score_normals(
                reloaded,
                make_loader(datasets[Partition.CALIBRATION], config.batch_size, config.seed),
                "calibration",
            )
            threshold = calibrate_normal(
                torch.tensor(calibration["scores"], dtype=torch.float64),
                config.calibration_quantile,
            )
            write_json(directory, "calibration-scores.json", calibration)
            write_json(directory, "calibration.json", threshold)
        result = {
            "completed": True,
            "training_seconds": training_seconds,
            "training_cpu_seconds": training_cpu_seconds,
            "total_seconds_including_reload_calibration": time.perf_counter() - started,
            "peak_working_set_bytes": peak_memory(),
            "best_epoch": stop.best_epoch,
            "best_validation_loss": stop.best,
            "initial_validation_loss": initial_validation["mean"],
            "early_stopped": stopped,
            "epochs": len(history),
            "optimizer_steps": sum(r["optimizer_steps"] for r in history),
            "final_parameter_sha256": parameter_sha256(model),
            "best_parameter_sha256": parameter_sha256(reloaded),
            "parameters_changed": parameter_sha256(model) != initial_hash,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "reload_validation_exactly_equal": True,
            "official_test_scored": False,
        }
        write_json(directory, "result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        write_json(
            directory,
            "failure.json",
            {
                "type": type(exc).__name__,
                "message": "Experiment failed; inspect the local console traceback.",
                "completed_epochs": len(history),
            },
        )
        raise


if __name__ == "__main__":
    main()
