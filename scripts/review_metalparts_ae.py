"""Post-hoc inspection of saved 3E source maps; never changes scores or runs a model."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from factoryvision.data.metalparts import REVISION, MetalPartsSample, MetalPartsSource
from factoryvision.data.metalparts_diagnostic import DiagnosticReadGuard, source_pair
from factoryvision.domain.contracts import Label
from factoryvision.evaluation.metalparts import PixelRanking
from factoryvision.inference.service import InspectionResult
from factoryvision.inference.visualization import panels
from factoryvision.training.checkpoint import file_sha256, write_json


def main():
    project = Path(__file__).resolve().parents[1]
    output = project / "reports/metalparts-3f-v1/ae-review"
    output.mkdir()  # Retain original and any partial review.
    root = project / "data/raw/robustad" / REVISION
    original = project / "runs/metalparts-source-diagnostic-v1"
    manifest = json.loads(
        (project / "reports/metalparts-3e-v1/diagnostic-manifest.json").read_text()
    )
    verified = json.loads(
        (project / "reports/robustad-3c-v1/integrity/source-verification-final.json").read_text()
    )["files"]
    rows = {r["sample_id"]: r for r in manifest["inventory"]}
    scores = json.loads((original / "scores.json").read_text())
    selected = [rows[r["sample_id"]] for r in scores]
    allowed = [p for row in selected for p in (row["image_path"], row["mask_path"]) if p]
    guard = DiagnosticReadGuard(root, allowed)
    sys.addaudithook(guard)
    results = []
    sums = dict(border_pixels=0, all_pixels=0, border_error=0.0, all_error=0.0)
    for row, score in zip(selected, scores, strict=True):
        path = original / score["map"]
        if file_sha256(path) != score["map_sha256"]:
            raise ValueError("Saved AE map drift.")
        spatial = np.load(path, allow_pickle=False)
        if float(spatial.mean()) != score["score"]:
            raise ValueError("Saved AE score/map mismatch.")
        sample = MetalPartsSample(
            MetalPartsSource(),
            "train",
            row["file_name"],
            Label.ANOMALY if row["label"] else Label.NORMAL,
            row["mask_path"],
            row["width"],
            row["height"],
        )
        _, mask = source_pair(root, sample, verified)
        border = np.ones(spatial.shape, dtype=bool)
        border[16:-16, 16:-16] = False
        sums["border_pixels"] += int(border.sum())
        sums["all_pixels"] += int(spatial.size)
        sums["border_error"] += float(spatial[border].sum(dtype=np.float64))
        sums["all_error"] += float(spatial.sum(dtype=np.float64))
        hist = PixelRanking()
        hist.add(spatial, mask)
        peak = np.unravel_index(np.argmax(spatial), spatial.shape)
        results.append(
            {
                "sample_id": row["sample_id"],
                "anomaly": row["label"],
                "pixel_metrics": hist.result(),
                "peak_inside_mask": bool(mask[peak]),
                "mean_defect_error": float(spatial[mask.astype(bool)].mean())
                if mask.any()
                else None,
                "mean_background_error": float(spatial[~mask.astype(bool)].mean()),
                "border_error_fraction": float(
                    spatial[border].sum(dtype=np.float64) / spatial.sum(dtype=np.float64)
                ),
            }
        )
    anomalies = sorted(
        [r for r in results if r["anomaly"]],
        key=lambda r: (r["pixel_metrics"]["average_precision"], r["sample_id"]),
    )
    for title, record in [("lowest-pixel-ap", anomalies[0]), ("highest-pixel-ap", anomalies[-1])]:
        row = rows[record["sample_id"]]
        sample = MetalPartsSample(
            MetalPartsSource(),
            "train",
            row["file_name"],
            Label.ANOMALY,
            row["mask_path"],
            row["width"],
            row["height"],
        )
        image, mask = source_pair(root, sample, verified)
        score = next(s for s in scores if s["sample_id"] == record["sample_id"])
        spatial = np.load(original / score["map"], allow_pickle=False)
        result = InspectionResult(
            "autoencoder",
            "Frozen AE",
            "MSE",
            score["score"],
            None,
            None,
            spatial,
            np.asarray(image),
            None,
            None,
            0,
            0,
            0,
            0,
            "",
        )
        view = panels(result)
        for name in ("original", "heatmap", "overlay"):
            view[name].save(output / f"{title}-{name}.png")
        Image.fromarray(mask * 255).resize(view["original"].size, Image.Resampling.NEAREST).save(
            output / f"{title}-mask.png"
        )
    baseline = json.loads((original / "metrics.json").read_text())
    summary = {
        "label": "POST-HOC SAVED-AE SOURCE EVIDENCE REVIEW; NOT AN INDEPENDENT EVALUATION",
        "no_model_forward_passes": True,
        "boundary_band_pixels": 16,
        "boundary": sums,
        "border_pixel_fraction": sums["border_pixels"] / sums["all_pixels"],
        "border_error_fraction": sums["border_error"] / sums["all_error"],
        "constant_score_ap_reference": {k: baseline[k]["prevalence"] for k in ("image", "pixel")},
        "posthoc_examples": {"lowest_pixel_ap": anomalies[0], "highest_pixel_ap": anomalies[-1]},
        "anomaly_peaks_inside_mask": sum(r["peak_inside_mask"] for r in anomalies),
        "source_reads": sorted(guard.opened),
        "records": results,
    }
    write_json(output, "review.json", summary)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("records", "source_reads")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
