"""Explicit one-shot bounded student smoke/train/development evaluation. No test loader."""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torchvision
from feature_experiment import distribution, figure, timed
from torch.utils.data import DataLoader, Subset
from train_cashew import peak_memory

from factoryvision.data.development import (
    DevelopmentDataset,
    DevelopmentReadGuard,
    synthetic_defect,
)
from factoryvision.data.images import make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development
from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.models.feature_memory import load_extractor
from factoryvision.models.student_teacher import CompactStudent, SpatialTeacher, StudentTeacher
from factoryvision.training.checkpoint import atomic_write, create_run, file_sha256, write_json
from factoryvision.training.engine import EarlyStopping, parameter_sha256, seed_everything
from factoryvision.training.student_artifacts import load_student, save_student
from factoryvision.training.student_engine import StudentConfig, epoch, score_normals

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "docs/protocols/student-development-v1.json"
RECIPE_SHA = "fb5fd19e04aaa737de16b0483ca3f218af50aaf47707549dbc7c0345a42b7779"
FULL_RUN = "cashew-student-v1"


def sources():
    paths = [
        *sorted((ROOT / "src").rglob("*.py")),
        Path(__file__),
        ROOT / "scripts/feature_experiment.py",
        ROOT / "scripts/train_cashew.py",
    ]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in paths}


def bounded(loader):
    for batch in loader:
        if peak_memory() > 2 * 1024**3:
            raise MemoryError("Student observed process peak exceeds 2 GiB.")
        yield batch


def train(directory, model, datasets, ids, fingerprint, smoke):
    config = StudentConfig(epochs=1 if smoke else 12)
    used = {
        p: values[: 16 if p == "train" else 8] if smoke else values
        for p, values in ids.items()
        if p in ("train", "validation")
    }
    if set(used["train"]) & set(used["validation"]):
        raise ValueError("Training/validation overlap.")
    train_data = Subset(datasets["train"], range(len(used["train"])))
    validation_data = Subset(datasets["validation"], range(len(used["validation"])))
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(validation_data, batch_size=config.batch_size, num_workers=0)
    teacher_before, student_before = (
        parameter_sha256(model.teacher),
        parameter_sha256(model.student),
    )
    write_json(
        directory,
        "experiment.json",
        {
            "config": asdict(config),
            "smoke": smoke,
            "recipe_sha256": RECIPE_SHA,
            "manifest_sha256": fingerprint,
            "fit_ids": used["train"],
            "validation_ids": used["validation"],
            "teacher_parameters": sum(p.numel() for p in model.teacher.parameters()),
            "student_parameters": sum(p.numel() for p in model.student.parameters()),
            "teacher_initial_sha256": teacher_before,
            "student_initial_sha256": student_before,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "numpy": np.__version__,
            },
            "lock_sha256": file_sha256(ROOT / "uv.lock"),
            "source_hashes": sources(),
        },
    )
    optimizer = torch.optim.Adam(model.student.parameters(), lr=config.learning_rate)
    stopping = EarlyStopping(config.patience, 0.0)
    start, cpu_start = time.perf_counter(), time.process_time()
    deadline = start + config.seconds
    initial = epoch(model, bounded(val_loader), used["validation"], "validation", deadline=deadline)
    history, updates = [], 0
    probe = next(iter(val_loader))["image"]
    for number in range(1, config.epochs + 1):
        tr = epoch(
            model,
            bounded(train_loader),
            used["train"],
            "train",
            optimizer=optimizer,
            deadline=deadline,
        )
        val = epoch(model, bounded(val_loader), used["validation"], "validation", deadline=deadline)
        improved, stop = stopping.update(val["loss"], number)
        updates += tr["updates"]
        row = {
            "epoch": number,
            "train_loss": tr["loss"],
            "validation_loss": val["loss"],
            "updates": tr["updates"],
            "training_ids": tr["sample_ids"],
            "validation_ids": val["sample_ids"],
            "elapsed_seconds": time.perf_counter() - start,
        }
        history.append(row)
        if improved:
            save_student(
                directory,
                model.student,
                fingerprint,
                RECIPE_SHA,
                used["train"],
                number,
                val["loss"],
            )
            score, spatial = model.score(probe)
            atomic_write(
                directory,
                "best-probe.pt",
                lambda f, score=score, spatial=spatial: torch.save(
                    {"score": score, "map": spatial}, f
                ),
            )
        write_json(directory, "history.json", history)
        if parameter_sha256(model.teacher) != teacher_before:
            raise RuntimeError("Teacher changed during student training.")
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_ids")}), flush=True)
        if stop:
            break
    elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
    final_student = parameter_sha256(model.student)
    restored, meta = load_student(directory, fingerprint, RECIPE_SHA, used["train"])
    model.student = restored
    reference = torch.load(directory / "best-probe.pt", weights_only=True, map_location="cpu")
    score, spatial = model.score(probe)
    torch.testing.assert_close(score, reference["score"], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(spatial, reference["map"], atol=1e-6, rtol=1e-5)
    if final_student == student_before or parameter_sha256(restored) == student_before:
        raise RuntimeError("Student did not change.")
    result = {
        "initial_validation_loss": initial["loss"],
        "best_epoch": meta["epoch"],
        "best_validation_loss": meta["validation_loss"],
        "epochs": len(history),
        "updates": updates,
        "elapsed_seconds": elapsed,
        "cpu_seconds": cpu,
        "mean_logical_cores": cpu / elapsed,
        "peak_working_set_bytes": peak_memory(),
        "student_initial_sha256": student_before,
        "student_final_sha256": final_student,
        "student_best_sha256": parameter_sha256(restored),
        "teacher_before": teacher_before,
        "teacher_after": parameter_sha256(model.teacher),
        "reload_score_max_abs": float((score - reference["score"]).abs().max()),
        "reload_map_max_abs": float((spatial - reference["map"]).abs().max()),
        "checkpoint_bytes": (directory / "best.pt").stat().st_size,
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "smoke": smoke,
    }
    write_json(directory, "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate(directory, model, datasets, ids, fingerprint):
    if read(directory / "experiment.json")["source_hashes"] != sources():
        raise ValueError("Frozen execution sources changed after training.")
    with (directory / "development-started.json").open("x") as f:
        json.dump({"recipe_sha256": RECIPE_SHA, "scope": "normal and synthetic only"}, f)
    model.student, _ = load_student(directory, fingerprint, RECIPE_SHA, ids["train"])
    before = parameter_sha256(model)
    start = time.perf_counter()
    records = {
        p: score_normals(model, bounded(make_loader(datasets[p], 8)), ids[p], p)
        for p in ("validation", "calibration")
    }
    calibration = calibrate_development(records["calibration"], ids["calibration"])
    write_json(directory, "calibration.json", calibration)
    write_json(directory, "normal-scores.json", records)
    masks, maps, rows = [], [], []
    for i in range(16):
        sample = datasets["validation"][i]
        changed, mask = synthetic_defect(sample["image"], sample["sample_id"])
        score, spatial = model.score(changed[None])
        y, x = divmod(int(spatial[0].argmax()), 128)
        rows.append(
            {
                "source_id": sample["sample_id"],
                "synthetic": True,
                "normal_score": records["validation"]["scores"][i],
                "synthetic_score": float(score[0]),
                "threshold_exceeded": float(score[0]) > calibration["threshold"],
                "peak_inside_mask": bool(mask[y, x]),
            }
        )
        masks.append(mask.numpy())
        maps.append(spatial[0].numpy())
        if i < 4:
            figure(
                directory / f"synthetic-example-{i:02}.png",
                sample["image"],
                changed,
                mask,
                spatial[0],
                calibration["threshold"],
            )
    write_json(directory, "synthetic-probes.json", rows)
    np.savez_compressed(
        directory / "synthetic-maps.npz", maps=np.stack(maps), masks=np.stack(masks)
    )
    x = datasets["validation"][0]["image"][None]
    batch = next(iter(make_loader(datasets["validation"], 8)))["image"]
    result = {
        "scope": "normal development and artificial diagnostics; no real defect accuracy",
        "normal": {
            p: {
                "count": len(r["scores"]),
                "distribution": distribution(r["scores"]),
                "exceedances": sum(s > calibration["threshold"] for s in r["scores"]),
            }
            for p, r in records.items()
        },
        "calibration": calibration,
        "synthetic": {
            "count": 16,
            "score_increased": sum(r["synthetic_score"] > r["normal_score"] for r in rows),
            "threshold_exceeded": sum(r["threshold_exceeded"] for r in rows),
            "peak_inside_mask": sum(r["peak_inside_mask"] for r in rows),
            "median_score_change": float(
                np.median([r["synthetic_score"] - r["normal_score"] for r in rows])
            ),
            "pixel_metrics": binary_metrics(np.stack(masks).ravel(), np.stack(maps).ravel()),
        },
        "latency": {
            "batch1": timed(lambda: model.score(x)),
            "batch8": timed(lambda: model.score(batch)),
            "decode_resize": timed(lambda: datasets["validation"][0], warmups=0),
        },
        "elapsed_seconds": time.perf_counter() - start,
        "peak_working_set_bytes": peak_memory(),
        "state_before": before,
        "state_after": parameter_sha256(model),
    }
    if before != result["state_after"]:
        raise RuntimeError("Evaluation changed model state.")
    write_json(directory, "development.json", result)
    print(json.dumps(result, indent=2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["smoke", "train", "evaluate"])
    parser.add_argument("--run-id", help="Fresh smoke/debug run only; full experiment ID is fixed")
    args = parser.parse_args()
    if args.run_id and args.action != "smoke":
        parser.error("Custom IDs are restricted to explicit smoke/debug runs.")
    if file_sha256(RECIPE) != RECIPE_SHA:
        raise ValueError("Frozen student recipe changed.")
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    fingerprint = manifest_sha256(manifest)
    if fingerprint != read(RECIPE)["manifest_sha256"]:
        raise ValueError("Accepted manifest changed.")
    normals = [s for s in manifest.samples if s.partition != Partition.TEST]
    guard = DevelopmentReadGuard(ROOT / "data/visa", normals)
    sys.addaudithook(guard)
    audit = {
        r["sample_id"]: r["image_sha256"]
        for r in read(ROOT / "data/metadata/cashew-audit.json")["records"]
    }
    for sample in normals:
        if file_sha256(ROOT / "data/visa" / sample.image_path) != audit[sample.sample_id]:
            raise ValueError("Normal source identity changed.")
    datasets = {
        p.value: DevelopmentDataset(manifest, ROOT / "data/visa", p)
        for p in (Partition.TRAIN, Partition.VALIDATION, Partition.CALIBRATION)
    }
    ids = {p: [s.sample_id for s in ds.samples] for p, ds in datasets.items()}
    if {p: len(v) for p, v in ids.items()} != {"train": 315, "validation": 67, "calibration": 68}:
        raise ValueError("Development membership changed.")
    seed_everything(42, 4)
    model = StudentTeacher(
        SpatialTeacher(load_extractor(ROOT / "data/pretrained/resnet18-f37072fd.pth")),
        CompactStudent(42),
    )
    if args.action == "evaluate":
        directory = ROOT / "runs" / FULL_RUN
        evaluate(directory, model, datasets, ids, fingerprint)
    else:
        if args.action == "train":
            smoke = read(ROOT / "runs/cashew-student-smoke-v1/result.json")
            if (
                smoke["updates"] != 2
                or smoke["teacher_before"] != smoke["teacher_after"]
                or smoke["peak_working_set_bytes"] > 2 * 1024**3
            ):
                raise ValueError("Smoke feasibility gate failed.")
        directory = create_run(
            ROOT, args.run_id or ("cashew-student-smoke-v1" if args.action == "smoke" else FULL_RUN)
        )
        write_json(directory, "recipe.json", read(RECIPE))
        train(directory, model, datasets, ids, fingerprint, args.action == "smoke")
    if peak_memory() > 2 * 1024**3:
        raise MemoryError("Observed memory budget exceeded.")
    write_json(
        directory,
        f"{args.action}-reads.json",
        {
            "unique_files": len(guard.opened),
            "official_test_files": 0,
            "opened_paths": sorted(guard.opened),
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


if __name__ == "__main__":
    main()
