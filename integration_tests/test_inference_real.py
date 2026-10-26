"""Explicit 3A inference only: two Cashew validation normals, no evaluations or masks."""

import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch

from factoryvision.data.development import DevelopmentReadGuard
from factoryvision.data.manifest import load_manifest
from factoryvision.domain.contracts import Partition
from factoryvision.inference.service import MODEL_IDS, InspectionService
from factoryvision.training.engine import parameter_sha256


def test_frozen_cashew_inference_only(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    samples = sorted(
        (s for s in manifest.samples if s.partition == Partition.VALIDATION),
        key=lambda s: s.sample_id,
    )[:2]
    guard = DevelopmentReadGuard(root / "data/visa", samples)
    sys.addaudithook(guard)

    def prohibit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if (
                path.is_relative_to(root / "data")
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
                and path not in guard.allowed
            ):
                raise PermissionError("Non-allowlisted image/mask access prohibited in 3A")

    sys.addaudithook(prohibit)

    def no_network(*args, **kwargs):
        raise AssertionError("No network permitted in real inference check")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    engine = InspectionService(root)
    assert all(s["available"] for s in engine.statuses())
    observations = []
    for sample in samples:
        payload = (root / "data/visa" / sample.image_path).read_bytes()
        for key in MODEL_IDS:
            result = engine.inspect(payload, key)
            model = engine._models[key]
            before = parameter_sha256(model)
            again = engine.inspect(payload, key)
            assert parameter_sha256(model) == before
            assert result.score == again.score and np.array_equal(result.spatial, again.spatial)
            assert not model.training and all(not p.requires_grad for p in model.parameters())
            observations.append(
                {
                    "sample_id": sample.sample_id,
                    "model": key,
                    "score": result.score,
                    "threshold": result.threshold,
                    "first_inference_ms": result.inference_ms,
                    "repeat_inference_ms": again.inference_ms,
                    "first_total_ms": result.total_ms,
                    "repeat_total_ms": again.total_ms,
                    "artifact_sha256": result.artifact_sha256,
                }
            )
    assert guard.opened == {s.image_path for s in samples}
    report = root / "reports/application-3a-verification-v1"
    report.mkdir(exist_ok=True)
    (report / "real-inference.json").write_text(
        json.dumps(
            {
                "observations": observations,
                "opened_development_paths": sorted(guard.opened),
                "official_test_files": 0,
                "purpose": "Functionality/determinism only; no accuracy evaluation or calibration.",
            },
            indent=2,
        )
        + "\n"
    )
