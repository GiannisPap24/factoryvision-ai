"""Exercise the real audit orchestration with a tiny synthetic archive only."""

import importlib.util
import io
import json
import shutil
import sys
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_pretraining_integrity import fixture_archive, png

from factoryvision.data.pretraining_integrity import Limits, file_hash


@pytest.mark.parametrize(
    "case,expected",
    [
        ("exact", "BLOCKED"),
        ("ambiguous", "INCONCLUSIVE"),
        ("corrupt", "BLOCKED"),
        ("different", "PASS"),
    ],
)
def test_runner_model_free_exclusive_and_source_preserving(tmp_path, monkeypatch, case, expected):
    archive, _ = fixture_archive(tmp_path)
    payloads = [png("red"), png("red")]
    if case == "ambiguous":
        payloads[1] = png("blue")
    elif case == "corrupt":
        payloads[1] = b"not an image"
    elif case == "different":
        for i in range(2):
            pixels = np.random.default_rng(i).integers(0, 256, (32, 32, 3), dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            payloads[i] = buffer.getvalue()
    with tarfile.open(archive, "w") as stream:
        for i, payload in enumerate(payloads):
            member = tarfile.TarInfo(f"pcb2/Data/Images/Normal/{i:04}.JPG")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    for folder in ("data/raw", "data/metadata", "reports", "docs/protocols"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(archive, tmp_path / "data/raw/VisA_20220922.tar")
    shutil.copyfile(tmp_path / "split.csv", tmp_path / "data/metadata/1cls.csv")
    protocol = {
        "method": "pcb2-integrity-v1",
        "limits": asdict(Limits()),
        "execution_files": {},
        "recovery_checkpoint": "synthetic-head",
        "source": {
            "archive_sha256": file_hash(archive),
            "split_sha256": file_hash(tmp_path / "split.csv"),
        },
    }
    p = tmp_path / "docs/protocols/pcb2-integrity-v1.json"
    p.write_text(json.dumps(protocol))
    spec = importlib.util.spec_from_file_location(
        "synthetic_pcb2_runner",
        Path(__file__).resolve().parents[1] / "scripts/audit_pcb2_integrity.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "OUT", tmp_path / "reports/audit")
    monkeypatch.setattr(runner, "DATA", tmp_path / "data/prepared")
    monkeypatch.setattr(runner, "peak_memory", lambda: 1024)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: "synthetic-head")
    monkeypatch.setattr(sys, "argv", ["audit", "--protocol-sha", file_hash(p)])
    runner.main()
    result = json.loads((runner.OUT / "summary.json").read_text())
    assert result["verdict"] == expected
    assert result["model_predictions"] == result["mask_payloads_accessed"] == 0
    assert result["images_visually_displayed"] == 0
    assert (
        file_hash(tmp_path / "data/raw/VisA_20220922.tar") == protocol["source"]["archive_sha256"]
    )
    evidence = json.loads((runner.OUT / "evidence.json").read_text())
    for name, digest in evidence.items():
        assert file_hash(runner.OUT / name) == digest
    with pytest.raises(FileExistsError):
        runner.main()
