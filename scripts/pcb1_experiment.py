"""Explicit PCB1 normal-only smoke, fit and calibration. Never constructs test loaders."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from feature_experiment import distribution, timed
from student_experiment import train as train_student
from torch.utils.data import DataLoader, Subset
from train_cashew import peak_memory

from factoryvision.data.category import normal_members
from factoryvision.data.development import DevelopmentDataset, DevelopmentReadGuard
from factoryvision.data.images import make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development
from factoryvision.models.autoencoder import ConvAutoencoder
from factoryvision.models.feature_memory import load_extractor, require_development_batch
from factoryvision.models.pcb1 import (
    FAMILIES,
    PROTOCOL_SHA,
    fit_capped_memory,
    load_family,
    predict,
    save_reference,
)
from factoryvision.models.spatial_mean import fit_spatial_mean
from factoryvision.models.student_teacher import CompactStudent, SpatialTeacher, StudentTeacher
from factoryvision.training.checkpoint import (
    create_run,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from factoryvision.training.engine import (
    EarlyStopping,
    parameter_sha256,
    score_normals,
    seed_everything,
    train_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "data/pretrained/resnet18-f37072fd.pth"


def context():
    if file_sha256(ROOT / "docs/protocols/pcb1-confirmatory-v3.json") != PROTOCOL_SHA:
        raise ValueError("Frozen protocol changed.")
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    fingerprint = manifest_sha256(manifest)
    gate = json.loads((ROOT / "reports/pcb1-normal-review.json").read_text())
    if (
        gate["manifest_sha256"] != fingerprint
        or gate["protocol_sha256"] != PROTOCOL_SHA
        or gate["confirmed_leakage"]
        or not gate["review_complete"]
        or gate["screen_sha256"] != file_sha256(ROOT / "reports/pcb1-normal-screen.json")
    ):
        raise ValueError("Normal integrity review failed.")
    samples = normal_members(manifest)
    if any(s.category != "pcb1" for s in manifest.samples):
        raise ValueError("Category contamination.")
    root = ROOT / "data/pcb1-development"
    guard = DevelopmentReadGuard(root, samples)
    sys.addaudithook(guard)
    for sample in samples:
        if file_sha256(root / sample.image_path) != sample.content_sha256:
            raise ValueError("Normal data identity changed.")
    datasets = {
        p: DevelopmentDataset(manifest, root, Partition(p))
        for p in ("train", "validation", "calibration")
    }
    ids = {p: [s.sample_id for s in ds.samples] for p, ds in datasets.items()}
    return fingerprint, datasets, ids, guard


def bounded(loader, deadline):
    for batch in loader:
        if time.perf_counter() > deadline or peak_memory() > 2 * 1024**3:
            raise RuntimeError("PCB1 resource budget exceeded.")
        yield batch


def train_ae(directory, datasets, ids, fingerprint, smoke):
    model = ConvAutoencoder(128)
    initial_hash = parameter_sha256(model)
    used = {
        p: ids[p][: 32 if p == "train" else 16] if smoke else ids[p]
        for p in ("train", "validation")
    }
    tr = DataLoader(
        Subset(datasets["train"], range(len(used["train"]))),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
        num_workers=0,
    )
    va = DataLoader(
        Subset(datasets["validation"], range(len(used["validation"]))), batch_size=16, num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    stop = EarlyStopping(3, 1e-5)
    start, cpu = time.perf_counter(), time.process_time()
    deadline = start + 1800
    initial = score_normals(model, bounded(va, deadline), "validation")
    history = []
    for number in range(1, 2 if smoke else 13):
        fitted = train_epoch(model, optimizer, bounded(tr, deadline), deadline=deadline)
        if sorted(fitted["sample_ids"]) != used["train"]:
            raise ValueError("AE fitting membership mismatch.")
        validated = score_normals(model, bounded(va, deadline), "validation")
        if validated["sample_ids"] != used["validation"]:
            raise ValueError("AE validation membership mismatch.")
        improved, stopped = stop.update(validated["mean"], number)
        history.append(
            dict(
                epoch=number,
                train_loss=fitted["loss"],
                validation_loss=validated["mean"],
                updates=fitted["steps"],
                training_ids=fitted["sample_ids"],
                validation_ids=used["validation"],
                elapsed_seconds=time.perf_counter() - start,
            )
        )
        if improved:
            save_checkpoint(directory, model, fingerprint, number, validated["mean"])
            write_json(directory, "best-validation.json", validated)
        write_json(directory, "history.json", history)
        print(
            json.dumps({k: v for k, v in history[-1].items() if not k.endswith("_ids")}), flush=True
        )
        if stopped:
            break
    elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu
    restored, meta = load_checkpoint(directory, fingerprint, 128)
    if score_normals(restored, va, "validation") != json.loads(
        (directory / "best-validation.json").read_text()
    ):
        raise ValueError("AE reload differs.")
    if parameter_sha256(restored) == initial_hash:
        raise ValueError("AE did not learn.")
    write_json(
        directory,
        "result.json",
        dict(
            epochs=len(history),
            updates=sum(x["updates"] for x in history),
            elapsed_seconds=elapsed,
            cpu_seconds=cpu_elapsed,
            peak_working_set_bytes=peak_memory(),
            initial_validation_loss=initial["mean"],
            best_epoch=meta["epoch"],
            best_validation_loss=meta["validation_loss"],
            initial_sha256=initial_hash,
            best_sha256=parameter_sha256(restored),
            reload_exact=True,
            smoke=smoke,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "fit", "calibrate"))
    parser.add_argument("family", choices=FAMILIES)
    args = parser.parse_args()
    fingerprint, datasets, ids, guard = context()
    seed_everything(42, 4)
    smoke = args.mode == "smoke"
    name = f"pcb1-{args.family.replace('_', '-')}-{'smoke' if smoke else 'v1'}"
    directory = ROOT / "runs" / name
    if args.mode != "calibrate":
        directory = create_run(ROOT, name)
        write_json(
            directory,
            "pcb1-provenance.json",
            dict(
                family=args.family,
                protocol_sha256=PROTOCOL_SHA,
                manifest_sha256=fingerprint,
                fit_ids=ids["train"],
                validation_ids=ids["validation"],
                calibration_ids=ids["calibration"],
                smoke=smoke,
                lock_sha256=file_sha256(ROOT / "uv.lock"),
                source_hashes={
                    p.relative_to(ROOT).as_posix(): file_sha256(p)
                    for p in [*sorted((ROOT / "src").rglob("*.py")), Path(__file__)]
                },
            ),
        )
        if args.family == "autoencoder":
            train_ae(directory, datasets, ids, fingerprint, smoke)
        elif args.family == "student_teacher":
            model = StudentTeacher(SpatialTeacher(load_extractor(WEIGHTS)), CompactStudent(42))
            train_student(directory, model, datasets, ids, fingerprint, smoke)
        else:
            if smoke:
                raise ValueError("Smoke is defined only for trainable families.")
            start, cpu = time.perf_counter(), time.process_time()
            loader = bounded(make_loader(datasets["train"], 8), start + 600)
            selection = None
            if args.family == "spatial_mean":
                model, fitted_ids = fit_spatial_mean(loader)
                if fitted_ids != ids["train"]:
                    raise ValueError("Mean fitting membership mismatch.")
            else:
                model, selection = fit_capped_memory(load_extractor(WEIGHTS), loader, ids["train"])
            save_reference(directory, args.family, model, fingerprint, ids["train"], selection)
            write_json(
                directory,
                "result.json",
                dict(
                    elapsed_seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu,
                    peak_working_set_bytes=peak_memory(),
                    fit_count=len(ids["train"]),
                    memory_rows=len(model.bank) if selection else None,
                ),
            )
        return
    with (directory / "calibration-started.json").open("x") as f:
        json.dump(dict(protocol_sha256=PROTOCOL_SHA, manifest_sha256=fingerprint), f)
    model = load_family(directory, args.family, fingerprint, ids["train"], WEIGHTS)
    before = parameter_sha256(model)
    records, maps_max = {}, 0.0
    for partition in ("validation", "calibration"):
        record = dict(partition=partition, sample_ids=[], scores=[])
        for batch in bounded(make_loader(datasets[partition], 8), time.perf_counter() + 600):
            require_development_batch(batch, partition, ids[partition], len(record["sample_ids"]))
            score, spatial = predict(model, args.family, batch["image"])
            record["sample_ids"].extend(batch["sample_id"])
            record["scores"].extend(score.tolist())
            if partition == "calibration":
                maps_max = max(maps_max, float(spatial.max()))
        if record["sample_ids"] != ids[partition]:
            raise ValueError("Incomplete calibration/development scoring.")
        records[partition] = record
    calibration = calibrate_development(records["calibration"], ids["calibration"])
    write_json(directory, "normal-scores.json", records)
    write_json(directory, "calibration.json", calibration)
    probe = datasets["validation"][0]["image"][None]
    original = predict(model, args.family, probe)
    restored = load_family(directory, args.family, fingerprint, ids["train"], WEIGHTS)
    repeated = predict(restored, args.family, probe)
    for a, b in zip(original, repeated, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    if parameter_sha256(model) != before:
        raise ValueError("Calibration mutated model.")
    report = dict(
        normal_validation_false_alarms=sum(
            s > calibration["threshold"] for s in records["validation"]["scores"]
        ),
        distributions={p: distribution(r["scores"]) for p, r in records.items()},
        threshold=calibration["threshold"],
        display_map_max=maps_max,
        reload_score_max_abs=float((original[0] - repeated[0]).abs().max()),
        reload_map_max_abs=float((original[1] - repeated[1]).abs().max()),
        latency=timed(lambda: predict(model, args.family, probe)),
        warm_decode=timed(lambda: datasets["validation"][0]),
        peak_working_set_bytes=peak_memory(),
        normal_files_opened=len(guard.opened),
        heldout_payloads_accessed=0,
    )
    write_json(directory, "development.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
