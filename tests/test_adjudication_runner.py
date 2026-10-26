"""Review orchestration on fabricated images; never accesses project datasets."""

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
from test_adjudication import fixture

from factoryvision.data.pretraining_integrity import file_hash


def runner_fixture(tmp_path, monkeypatch, colors=("red", "blue", "green")):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "synthetic_adjudication_runner", scripts / "prepare_pcb2_adjudication.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "synthetic-head")
    monkeypatch.setattr(runner, "peak_memory", lambda: 1000)
    rows, required, payloads = fixture(colors)
    for folder in (
        "reports/pcb2-integrity-v1",
        "reports/pcb2-adjudication-preflight-v1",
        "docs/protocols",
    ):
        (tmp_path / folder).mkdir(parents=True)
    for row, payload in zip(rows, payloads, strict=True):
        path = tmp_path / runner.DATA / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for name, value in (("inventory.json", rows), ("screen.json", {"required": required})):
        runner.write(tmp_path / "reports/pcb2-integrity-v1" / name, value)
    (tmp_path / runner.KEY).write_bytes(b"s" * 32)
    p = tmp_path / runner.PROTOCOL
    protocol = {
        "method": "pcb2-adjudication-v1",
        "checkpoint": "synthetic-head",
        "bound_files": {
            "reports/pcb2-integrity-v1/screen.json": file_hash(
                tmp_path / "reports/pcb2-integrity-v1/screen.json"
            )
        },
        "blinding_key_sha256": file_hash(tmp_path / runner.KEY),
        "queue_count": 3,
        "cross_count": 2,
        "reviewer_instructions": "Synthetic review only; no labels, filenames or scores.",
    }
    runner.write(p, protocol)
    return runner, rows, p


@pytest.mark.parametrize("case", ["unresolved", "exact", "resource"])
def test_package_blinding_stop_and_immutability(tmp_path, monkeypatch, case):
    runner, rows, protocol = runner_fixture(
        tmp_path,
        monkeypatch,
        ("red", "red", "green") if case == "exact" else ("red", "blue", "green"),
    )
    before = {p: file_hash(p) for p in tmp_path.rglob("*") if p.is_file()}
    if case == "resource":
        monkeypatch.setattr(runner, "peak_memory", lambda: 3 * 1024**3)
    result = runner.prepare(tmp_path, file_hash(protocol))
    assert result["verdict"] == ("BLOCKED" if case == "exact" else "INCONCLUSIVE")
    assert all(file_hash(p) == h for p, h in before.items())
    assert result["human_reviews"] == result["model_predictions"] == result["mask_accesses"] == 0
    out = tmp_path / runner.OUT
    evidence = runner.read(out / "evidence.json")
    assert all(file_hash(out / p) == h for p, h in evidence.items())
    private = runner.read(out / "restricted/pair-evidence.json")
    if case == "unresolved":
        assert len(private) == 3
        assert [p["cross_partition"] for p in private] == [True, True, False]
    elif case == "exact":
        assert len(private) < 3 and all(p["cross_partition"] for p in private)
    else:
        assert result["failure"] and not private
    with zipfile.ZipFile(out / "reviewer-bundle.zip") as bundle:
        assert bundle.namelist() == sorted(bundle.namelist())
        assert all(i.date_time == (2000, 1, 1, 0, 0, 0) for i in bundle.infolist())
        for name in bundle.namelist():
            assert "/" not in name
            if name.endswith(".json"):
                content = bundle.read(name).decode()
                assert all(
                    row["sample_id"] not in content and row["path"] not in content for row in rows
                )
                assert "relation" not in content and "cross_partition" not in content
    with pytest.raises(FileExistsError):
        runner.prepare(tmp_path, file_hash(protocol))


@pytest.mark.parametrize("case", ["sha", "method", "bound", "head", "key", "count", "source"])
def test_identity_failures_precede_review_pixels(tmp_path, monkeypatch, case):
    runner, rows, p = runner_fixture(tmp_path, monkeypatch)
    data = runner.read(p)
    expected = file_hash(p)
    if case == "sha":
        expected = "0" * 64
    elif case == "method":
        data["method"] = "unknown"
    elif case == "bound":
        data["bound_files"] = {"reports/pcb2-integrity-v1/screen.json": "0" * 64}
    elif case == "head":
        data["checkpoint"] = "other"
    elif case == "key":
        data["blinding_key_sha256"] = "0" * 64
    elif case == "count":
        data["queue_count"] = 99
    else:
        (tmp_path / runner.DATA / rows[0]["path"]).write_bytes(b"changed synthetic source")
    p.write_text(json.dumps(data))
    if case != "sha":
        expected = file_hash(p)
    with pytest.raises(ValueError):
        runner.prepare(tmp_path, expected)
    assert not (tmp_path / runner.OUT).exists()
