"""Render saved synthetic maps with correct discrepancy units; no model inference."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from factoryvision.data.development import (
    DevelopmentDataset,
    DevelopmentReadGuard,
    synthetic_defect,
)
from factoryvision.data.manifest import load_manifest
from factoryvision.domain.contracts import Partition
from factoryvision.training.checkpoint import file_sha256, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/cashew-student-v1"
    output = run / "illustrations"
    output.mkdir()  # Never overwrite the original diagnostic evidence.
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    dataset = DevelopmentDataset(manifest, root / "data/visa", Partition.VALIDATION)
    guard = DevelopmentReadGuard(root / "data/visa", dataset.samples)
    sys.addaudithook(guard)
    threshold = json.loads((run / "calibration.json").read_text())["threshold"]
    with np.load(run / "synthetic-maps.npz", allow_pickle=False) as stored:
        for i in range(4):
            sample = dataset[i]
            changed, mask = synthetic_defect(sample["image"], sample["sample_id"])
            assert np.array_equal(mask.numpy(), stored["masks"][i])
            heat = torch.from_numpy(stored["maps"][i] / threshold).clamp(0, 1)
            panels = [
                sample["image"],
                changed,
                torch.stack((heat, torch.zeros_like(heat), 1 - heat)),
                mask[None].expand(3, -1, -1).float(),
            ]
            canvas = Image.new("RGB", (1024, 300), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                "SYNTHETIC ONLY - student-teacher diagnostic; no real defect accuracy",
                fill="black",
            )
            titles = [
                "Normal validation source",
                "Artificial checkerboard",
                f"Squared unit-feature error 0..{threshold:.4f}",
                "Artificial mask",
            ]
            for j, (title, panel) in enumerate(zip(titles, panels, strict=True)):
                rgb = (panel.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                canvas.paste(
                    Image.fromarray(rgb).resize((256, 256), Image.Resampling.NEAREST), (j * 256, 44)
                )
                draw.text((j * 256 + 4, 24), title, fill="black")
            canvas.save(output / f"student-example-{i:02}.png")
    write_json(
        output,
        "provenance.json",
        {
            "source_maps_sha256": file_sha256(run / "synthetic-maps.npz"),
            "render_source_sha256": file_sha256(Path(__file__)),
            "change": (
                "Correct inherited feature-memory RMS caption; "
                "same saved maps and fixed display scale."
            ),
            "originals_preserved": True,
            "model_inference": False,
            "normal_files_opened": len(guard.opened),
            "official_test_files": 0,
            "figures": {p.name: file_sha256(p) for p in output.glob("*.png")},
        },
    )


if __name__ == "__main__":
    main()
