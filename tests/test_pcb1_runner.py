"""The deferred opening/evaluation flow is exercised on fabricated archive payloads only."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from test_category import category  # noqa: F401 - shared synthetic fixture

from factoryvision.data.category import audit_normals, extract_normals, load_category
from factoryvision.data.splits import derive_splits
from factoryvision.models.pcb1 import FAMILIES
from factoryvision.training.checkpoint import file_sha256


def test_blinded_runner_synthetic_only(category, monkeypatch):  # noqa: F811 - pytest fixture
    archive, split, normal_root = category
    project = archive.parent
    manifest = load_category(split, "pcb1")
    extract_normals(archive, normal_root, manifest, file_sha256(archive))
    manifest = derive_splits(audit_normals(manifest, normal_root)[0], 42)
    (project / "data/raw").mkdir(parents=True)
    shutil.copyfile(archive, project / "data/raw/VisA_20220922.tar")
    shutil.copytree(normal_root, project / "data/pcb1-development")
    (project / "reports").mkdir()
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "fabricated_blinded_runner", scripts / "pcb1_blinded.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", project)
    monkeypatch.setattr(runner, "OPENING", project / "reports/opening")
    monkeypatch.setattr(runner, "QUARANTINE", project / "data/quarantine")
    seal = dict(
        manifest_sha256="a" * 64,
        membership={
            p: [s.sample_id for s in manifest.samples if s.partition.value == p]
            for p in ("train", "validation", "calibration", "test")
        },
        models={
            f: dict(directory=f"runs/{f}", threshold=0.5, display_map_max=1.0) for f in FAMILIES
        },
    )
    runner.open_test(seal, "b" * 64, manifest)
    assert not (runner.OPENING / "evaluation").exists()
    with pytest.raises(FileExistsError):
        runner.open_test(seal, "b" * 64, manifest)
    integrity = json.loads((runner.OPENING / "integrity.json").read_text())
    assert integrity["model_predictions"] == 0 and not integrity["duplicates"]
    review = project / "review.json"
    review.write_text("{}")
    with pytest.raises(ValueError, match="review gate"):
        runner.evaluate(seal, "b" * 64, manifest, review)
    review.write_text(
        json.dumps(
            dict(
                integrity_sha256=file_sha256(runner.OPENING / "integrity.json"),
                confirmed_leakage=False,
                review_complete=True,
                reviewed_pairs=integrity["required_pairs"],
            )
        )
    )
    monkeypatch.setattr(runner, "load_family", lambda *a: None)
    monkeypatch.setattr(
        runner, "predict", lambda model, family, images: (images.mean((1, 2, 3)), images.mean(1))
    )
    runner.evaluate(seal, "b" * 64, manifest, review)
    evidence = json.loads((runner.OPENING / "evaluation/evidence.json").read_text())
    for name, sha in evidence.items():
        assert file_sha256(runner.OPENING / "evaluation" / name) == sha
    with np.load(runner.OPENING / "evaluation/autoencoder-raw.npz", allow_pickle=False) as raw:
        assert raw["sample_ids"].tolist() == seal["membership"]["test"]
        assert not raw["masks"].any()
    with pytest.raises(FileExistsError):
        runner.evaluate(seal, "b" * 64, manifest, review)
    relative = next(iter(integrity["extracted_files"]))
    (runner.QUARANTINE / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.evaluate(seal, "b" * 64, manifest, review)
