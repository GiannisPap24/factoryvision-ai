"""Explicit 3D source-normal preparation, smoke and one bounded pilot; no official test reads."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from train_cashew import peak_memory

from factoryvision.data.metalparts import REVISION, MetalPartsSource
from factoryvision.data.metalparts_development import (
    TRAIN_METADATA,
    NativePatchDataset,
    PatchConfig,
    load_training_normals,
    split_normals,
)
from factoryvision.models.autoencoder import ConvAutoencoder
from factoryvision.training.checkpoint import create_run, file_sha256, write_json
from factoryvision.training.engine import (
    parameter_sha256,
    score_normals,
    seed_everything,
    train_step,
)
from factoryvision.training.metalparts import (
    SourceReadGuard,
    artifact_identity,
    fit,
    load_model,
    native_diagnostic,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/metalparts-3d-v1"
PROTOCOL = ROOT / "docs/metalparts-exploratory-training-protocol.md"
CONFIG = PatchConfig()


def prepared():
    root = ROOT / "data/raw/robustad" / REVISION
    evidence = ROOT / "reports/robustad-3c-v1/integrity/source-verification-final.json"
    verified = json.loads(evidence.read_text())["files"]
    allowed = [
        p
        for p in verified
        if p == TRAIN_METADATA or p.startswith("MetalParts/metal_parts_data_dir_train/normal/")
    ]
    guard = SourceReadGuard(root, allowed)
    sys.addaudithook(guard)
    samples, excluded = load_training_normals(root, verified, MetalPartsSource())
    split, groups = split_normals(samples)
    inventory = [
        {
            "sample_id": s.sample_id,
            "image_path": s.image_path,
            "partition": s.partition.value,
            "content_sha256": s.content_sha256,
            "pixel_sha256": s.pixel_sha256,
            "width": s.sample.width,
            "height": s.sample.height,
            "group": groups[s.sample_id],
        }
        for s in split
    ]
    digest = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    return (
        root,
        split,
        guard,
        {
            "revision": REVISION,
            "metadata_sha256": verified[TRAIN_METADATA]["sha256"],
            "source_verification_sha256": file_sha256(evidence),
            "normal_count": len(samples),
            "anomaly_count": len(excluded),
            "excluded": excluded,
            "records": inventory,
            "split_sha256": digest,
            "group_count": len(set(groups.values())),
            "validation_independence": "potentially dependent; physical-item relationships unknown",
            "threshold": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "smoke", "train"))
    args = parser.parse_args()
    REPORT.mkdir(exist_ok=True)
    start = time.perf_counter()
    root, split, guard, evidence = prepared()
    if args.mode == "prepare":
        target = REPORT / "development.json"
        with target.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    **evidence,
                    "opened": sorted(guard.opened),
                    "seconds": time.perf_counter() - start,
                },
                stream,
                indent=2,
            )
        print(
            json.dumps(
                {
                    k: evidence[k]
                    for k in ("normal_count", "anomaly_count", "group_count", "split_sha256")
                }
            )
        )
        return
    sealed = json.loads((REPORT / "seal.json").read_text())
    if (
        file_sha256(PROTOCOL) != sealed["protocol_sha256"]
        or evidence["split_sha256"] != sealed["split_sha256"]
    ):
        raise ValueError("Protocol or source-development split changed after sealing.")
    for name, digest in sealed["code"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError("Execution source changed after sealing.")
    if args.mode == "train":
        smoke = json.loads((ROOT / "runs/metalparts-ae-smoke-v1/result.json").read_text())
        if not smoke["reload_equal"] or not smoke["parameters_changed"] or smoke["seconds"] > 900:
            raise ValueError("Smoke feasibility gate failed.")
    directory = create_run(ROOT, f"metalparts-ae-{'smoke' if args.mode == 'smoke' else 'pilot'}-v1")
    selected = split
    if args.mode == "smoke":
        selected = tuple(
            [s for s in split if s.partition.value == "train"][:4]
            + [s for s in split if s.partition.value == "validation"][:2]
        )
    from factoryvision.domain.contracts import Partition

    train = NativePatchDataset(root, selected, Partition.TRAIN, CONFIG)
    validation = NativePatchDataset(root, selected, Partition.VALIDATION, CONFIG)
    seed_everything(42, 4)
    model = ConvAutoencoder(256)
    initial = parameter_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    identity = artifact_identity(
        MetalPartsSource(), evidence["split_sha256"], sealed["protocol_sha256"], CONFIG
    )
    identity["purpose"] = args.mode
    write_json(
        directory,
        "configuration.json",
        {
            "identity": identity,
            "seed": 42,
            "batch_size": 8,
            "epochs": 1 if args.mode == "smoke" else 4,
            "learning_rate": 0.001,
            "threads": 4,
            "initial_parameter_sha256": initial,
            "source_seal": sealed,
            "development": evidence,
        },
    )
    write_json(
        directory,
        "patches.json",
        {
            "train": [train[i]["sample_id"] for i in range(len(train))],
            "validation": [validation[i]["sample_id"] for i in range(len(validation))],
            "config": asdict(CONFIG),
        },
    )

    def loader(epoch):
        return DataLoader(
            train, batch_size=8, sampler=train.shuffled_indices(42 + epoch), num_workers=0
        )

    val_loader = DataLoader(validation, batch_size=8, shuffle=False, num_workers=0)
    probe = next(iter(loader(0)))
    if args.mode == "smoke":
        loss = train_step(model, optimizer, probe)
        assert parameter_sha256(model) != initial
        write_json(
            directory,
            "single-batch.json",
            {
                "loss": loss,
                "parameters_changed": True,
                "finite_gradients": all(
                    p.grad is not None and torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                ),
            },
        )
    began = time.perf_counter()
    before, history = fit(
        model,
        optimizer,
        loader,
        val_loader,
        directory,
        identity,
        1 if args.mode == "smoke" else 4,
        began + (900 if args.mode == "smoke" else 3600),
        peak_memory,
    )
    loaded, meta = load_model(directory, identity)
    reference = json.loads((directory / "selected-validation.json").read_text())
    replay = score_normals(loaded, val_loader, "validation")
    assert replay == reference
    loaded_again, _ = load_model(directory, identity)
    with torch.inference_mode():
        equality = torch.equal(loaded(probe["image"]), loaded_again(probe["image"]))
    assert equality
    reconstruction, spatial, score = native_diagnostic(loaded, root, validation.samples[0], CONFIG)
    Image.fromarray(np.clip(reconstruction * 255, 0, 255).astype(np.uint8)).save(
        directory / "normal-reconstruction.png"
    )
    np.save(directory / "normal-error-map.npy", spatial)
    result = {
        "status": "EXPERIMENTAL MODEL TRAINED ON VERIFIED NORMAL SOURCE DATA",
        "identity": identity,
        "parameters": sum(p.numel() for p in model.parameters()),
        "parameters_changed": parameter_sha256(model) != initial,
        "initial_validation_loss": before["mean"],
        "epochs": history,
        "selected_epoch": meta["epoch"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "reload_equal": equality,
        "selected_validation_replay_equal": replay == reference,
        "seconds": time.perf_counter() - began,
        "total_process_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time(),
        "peak_working_set_bytes": peak_memory(),
        "train_parents": len(train.samples),
        "validation_parents": len(validation.samples),
        "train_patches": len(train),
        "validation_patches": len(validation),
        "normal_native_score": score,
        "native_map_shape": list(spatial.shape),
        "diagnostic_parent": validation.samples[0].sample_id,
        "opened_source_files": sorted(guard.opened),
        "test_images_opened": 0,
        "masks_opened": 0,
        "threshold": None,
    }
    assert result["parameters_changed"]
    write_json(directory, "result.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "selected_epoch",
                    "seconds",
                    "peak_working_set_bytes",
                    "reload_equal",
                    "checkpoint_sha256",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
