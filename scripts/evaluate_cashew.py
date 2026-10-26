"""One frozen held-out evaluation. Existing score artifacts prevent accidental re-execution."""

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
import torch
from PIL import Image, ImageDraw
from threadpoolctl import threadpool_limits
from train_cashew import peak_memory

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.metrics import paired_bootstrap
from factoryvision.evaluation.pipeline import evaluate_batches, summarize
from factoryvision.evaluation.protocol import PROTOCOL_SHA256, load_protocol, verify_model_artifacts
from factoryvision.evaluation.reconstruction import reconstruction_errors
from factoryvision.models.spatial_mean import SpatialMean
from factoryvision.training.checkpoint import file_sha256, load_checkpoint, write_json
from factoryvision.training.engine import seed_everything


def latency(model, image):
    elapsed = []
    with torch.inference_mode():
        for i in range(105):
            started = time.perf_counter_ns()
            reconstruction_errors(image, model(image))
            if i >= 5:
                elapsed.append((time.perf_counter_ns() - started) / 1e6)
    return {
        "repetitions": 100,
        "warmups": 5,
        "median_ms": float(np.median(elapsed)),
        "p95_ms": float(np.quantile(elapsed, 0.95)),
        "batch_size": 1,
        "scope": "preloaded normal validation RGB, forward+scores, no decode",
    }


def examples(model, dataset, directory):
    selected = [i for i, s in enumerate(dataset.samples) if s.label.value == "anomaly"][:4] + [
        i for i, s in enumerate(dataset.samples) if s.label.value == "normal"
    ][:2]
    for i in selected:
        sample = dataset[i]
        x = sample["image"].unsqueeze(0)
        with torch.inference_mode():
            reconstructed = model(x)
            score, error = reconstruction_errors(x, reconstructed)
        heat = (error[0] / 0.05).clamp(0, 1)
        color = torch.stack((heat, torch.zeros_like(heat), 1 - heat))
        panels = [sample["image"], reconstructed[0], color, sample["mask"].repeat(3, 1, 1)]
        canvas = Image.new("RGB", (1024, 294), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (4, 4),
            f"Example only: {sample['sample_id']} | AE score {float(score[0]):.7f}",
            fill="black",
        )
        for j, (title, tensor) in enumerate(
            zip(
                (
                    "Resized original RGB",
                    "AE reconstruction",
                    "Error 0 blue to 0.05 red",
                    "Binary GT (normal: derived zeros)",
                ),
                panels,
                strict=True,
            )
        ):
            array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            canvas.paste(
                Image.fromarray(array).resize((256, 256), Image.Resampling.NEAREST), (j * 256, 38)
            )
            draw.text((j * 256 + 2, 22), title, fill="black")
        canvas.save(directory / f"example-{sample['sample_id']}.png")


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root / "runs/cashew-eval-v1"
    protocol = load_protocol(root / "docs/protocols/cashew-evaluation-v1.json")
    verify_model_artifacts(root, protocol)
    frozen = json.loads((directory / "frozen-artifacts.json").read_text())
    if frozen["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Frozen artifact protocol mismatch.")
    if file_sha256(root / "uv.lock") != frozen["lock_sha256"] or sklearn.__version__ != "1.9.1":
        raise ValueError("Frozen metric environment changed.")
    for name, digest in frozen["execution_sources"].items():
        if file_sha256(root / name) != digest:
            raise ValueError("Execution source changed after freeze.")
    for path, digest in (
        (directory / "baseline-mean.pt", frozen["baseline_sha256"]),
        (directory / "baseline-calibration.json", frozen["baseline_calibration_sha256"]),
        (root / "reports/leakage-2b-review.json", frozen["leakage_review_sha256"]),
        (root / "data/metadata/cashew-audit.json", frozen["audit_sha256"]),
    ):
        if file_sha256(path) != digest:
            raise ValueError("Frozen gate/baseline artifact changed.")
    audit = json.loads((root / "data/metadata/cashew-audit.json").read_text())
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    if manifest_sha256(manifest) != frozen["manifest_sha256"]:
        raise ValueError("Manifest changed.")
    by_id = {s.sample_id: s for s in manifest.samples}
    for record in audit["records"]:
        sample = by_id[record["sample_id"]]
        for relative, expected in (
            (sample.image_path, record["image_sha256"]),
            (sample.mask_path, record["mask_sha256"]),
        ):
            if relative and file_sha256(root / "data/visa" / relative) != expected:
                raise ValueError("Image/mask bytes changed after gate.")
    seed_everything(42, 4)
    ae, _ = load_checkpoint(root / "runs/cashew-ae-128-seed42", manifest_sha256(manifest), 128)
    mean = SpatialMean(
        torch.load(directory / "baseline-mean.pt", map_location="cpu", weights_only=True)
    )
    dataset = ManifestDataset(manifest, root / "data/visa", Partition.TEST, ImageSettings(128, 128))
    expected_ids = [s.sample_id for s in dataset.samples]
    thresholds = {
        "autoencoder": protocol["autoencoder"]["threshold"],
        "spatial_mean": frozen["baseline_threshold"],
    }
    # Exclusive marker is written before the first held-out model prediction.
    with (directory / "evaluation-started.json").open("x", encoding="utf-8") as f:
        json.dump(
            {
                "at_utc": datetime.now(UTC).isoformat(),
                "protocol_sha256": PROTOCOL_SHA256,
                "frozen_artifacts_sha256": file_sha256(directory / "frozen-artifacts.json"),
            },
            f,
        )
    started = time.perf_counter()
    results = {}
    rows = {}
    maps = {}
    truth = None
    try:
        with threadpool_limits(limits=4):
            for name, model in (("autoencoder", ae), ("spatial_mean", mean)):
                records, spatial, masks = evaluate_batches(
                    model, make_loader(dataset, 16), expected_ids, thresholds[name]
                )
                rows[name] = records
                maps[name] = spatial
                if truth is not None and not np.array_equal(truth, masks):
                    raise ValueError("Models used different ground-truth pixels.")
                truth = masks
                results[name] = summarize(records, spatial, masks, thresholds[name])
                if peak_memory() > 2 * 1024**3:
                    raise MemoryError("Evaluation process exceeded 2GiB.")
            uncertainty = paired_bootstrap(
                [r["label"] for r in rows["autoencoder"]],
                [r["score"] for r in rows["autoencoder"]],
                [r["score"] for r in rows["spatial_mean"]],
            )
        write_json(directory, "per-image.json", rows)
        np.savez_compressed(
            directory / "spatial-maps.npz",
            autoencoder=maps["autoencoder"],
            spatial_mean=maps["spatial_mean"],
            masks=truth,
        )
        validation = ManifestDataset(
            manifest, root / "data/visa", Partition.VALIDATION, ImageSettings(128, 128)
        )
        sample = validation[0]["image"].unsqueeze(0)
        timings = {"autoencoder": latency(ae, sample), "spatial_mean": latency(mean, sample)}
        decoding = []
        for _ in range(30):
            t = time.perf_counter_ns()
            validation[0]
            decoding.append((time.perf_counter_ns() - t) / 1e6)
        timings["decode_resize"] = {
            "repetitions": 30,
            "median_ms": float(np.median(decoding)),
            "p95_ms": float(np.quantile(decoding, 0.95)),
            "scope": "same normal validation file, warm cache, includes mask derivation",
        }
        write_json(directory, "latency.json", timings)
        examples(ae, dataset, directory)
        report = {
            "protocol_sha256": PROTOCOL_SHA256,
            "models": results,
            "paired_bootstrap": uncertainty,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": peak_memory(),
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "scikit_learn": sklearn.__version__,
                "threads": 4,
                "device": "cpu",
            },
            "complexity": {
                "autoencoder": {
                    "parameters": 37763,
                    "file_bytes": (root / protocol["autoencoder"]["checkpoint"]).stat().st_size,
                },
                "spatial_mean": {
                    "learned_reference_values": 49152,
                    "trainable_parameters": 0,
                    "reference_tensor_bytes": mean.reference.numel() * 4,
                    "file_bytes": (directory / "baseline-mean.pt").stat().st_size,
                },
            },
            "test_set_now_observed": True,
            "aupro": "Not computed; predeclared deferral",
            "pixel_threshold_metrics": "Not computed; no calibrated pixel threshold",
        }
        write_json(directory, "metrics.json", report)
        write_json(
            directory,
            "artifact-hashes.json",
            {
                p.name: file_sha256(p)
                for p in directory.iterdir()
                if p.is_file() and p.name != "artifact-hashes.json"
            },
        )
        print(json.dumps(report, indent=2))
        print(json.dumps(timings, indent=2))
    except Exception as exc:
        write_json(
            directory,
            "failure.json",
            {
                "type": type(exc).__name__,
                "note": "Inspect local traceback; partial evidence retained.",
            },
        )
        raise


if __name__ == "__main__":
    main()
