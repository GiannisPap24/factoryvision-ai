"""CI orchestration doubles and tiny scalar fixtures; never recursively run pytest."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci_check.py"
spec = importlib.util.spec_from_file_location("ci_check", SCRIPT)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def evidence(covered=95, possible=100):
    return {
        "meta": {"branch_coverage": True},
        "totals": {
            "covered_lines": covered - 1,
            "num_statements": possible - 1,
            "covered_branches": 1,
            "num_branches": 1,
            # Deliberately rounded: acceptance must use counts, not this display.
            "percent_covered_display": "95.00",
        },
    }


@pytest.mark.parametrize(
    "covered,possible,accepted",
    [(9471, 10000, False), (94999, 100000, False), (95, 100, True), (100, 100, True)],
)
def test_coverage_gate_real_process_exit(tmp_path, covered, possible, accepted):
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps(evidence(covered, possible)), encoding="utf-8")
    # Only the scalar gate is executed in this child: no tests, model or CI stages.
    code = (
        "import runpy,sys;from pathlib import Path;"
        "runpy.run_path(sys.argv[1])['require_coverage'](Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SCRIPT), str(report)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (result.returncode == 0) is accepted
    assert ("PASS: exact combined coverage" in result.stdout) is accepted
    if not accepted:
        assert "below 95%" in result.stderr


@pytest.mark.parametrize("damage", ["missing", "malformed", "unreadable", "empty", "schema"])
def test_coverage_evidence_fails_closed(tmp_path, monkeypatch, damage):
    report = tmp_path / "coverage.json"
    if damage != "missing":
        report.write_text(json.dumps(evidence()), encoding="utf-8")
    if damage == "malformed":
        report.write_text("{", encoding="utf-8")
    elif damage == "empty":
        report.write_text("", encoding="utf-8")
    elif damage == "schema":
        report.write_text("{}", encoding="utf-8")
    elif damage == "unreadable":

        def denied(*args, **kwargs):
            raise PermissionError("Controlled unreadable evidence")

        monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(SystemExit, match="missing, unreadable or invalid") as failure:
        ci.require_coverage(report)
    assert failure.value.code != 0


@pytest.mark.parametrize("damage", ["bool", "negative", "overflow", "float", "zero", "no_branch"])
def test_invalid_coverage_counts_fail_closed(tmp_path, damage):
    data = evidence()
    if damage == "no_branch":
        data["meta"]["branch_coverage"] = False
    elif damage == "zero":
        data["totals"] = dict.fromkeys(data["totals"], 0)
    else:
        data["totals"]["covered_lines"] = {
            "bool": True,
            "negative": -1,
            "overflow": 1000,
            "float": 94.0,
        }[damage]
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid combined coverage"):
        ci.require_coverage(report)


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    calls = []
    control = {"fail_stage": None, "write_evidence": True}
    monkeypatch.setattr(ci, "__file__", str(tmp_path / "scripts/ci_check.py"))
    monkeypatch.setattr(ci.shutil, "which", lambda _: "reviewed-uv")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        monkeypatch.delenv(key, raising=False)

    def fake_run(command, *, cwd, env, check):
        assert check is True and env["UV_OFFLINE"] == "1"
        calls.append(command)
        stage = (
            "pytest"
            if command[1:3] == ["-m", "pytest"]
            else " ".join(Path(str(value)).name for value in command[1:4])
        )
        if control["fail_stage"] and control["fail_stage"] in stage:
            raise subprocess.CalledProcessError(7, command)
        if stage == "pytest" and control["write_evidence"]:
            target = Path(next(x.split("json:", 1)[1] for x in command if "json:" in x))
            assert target.parent.is_dir() and not target.exists()
            target.write_text(json.dumps(evidence()), encoding="utf-8")
        if command[1:2] == ["build"]:
            output = Path(command[-1])
            output.mkdir(parents=True)
            with zipfile.ZipFile(output / "factoryvision_ai-fixture.whl", "w") as archive:
                archive.writestr("fixture.txt", "synthetic")
            with tarfile.open(output / "factoryvision_ai-fixture.tar.gz", "w:gz") as archive:
                info = tarfile.TarInfo("factoryvision_ai-fixture/fixture.txt")
                info.size = 9
                archive.addfile(info, io.BytesIO(b"synthetic"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    return calls, control, tmp_path


def test_one_positive_allowlist_and_all_non_test_stages(orchestration):
    calls, _, root = orchestration
    ci.main()
    pytest_calls = [c for c in calls if c[1:3] == ["-m", "pytest"]]
    assert len(pytest_calls) == 1
    command = pytest_calls[0]
    assert command[-2:] == ["tests/", "integration_tests/test_performance_isolation.py"]
    assert command[3:5] == ["-c", "pyproject.toml"]
    assert {"--cov=factoryvision", "--cov-branch", "--cov-config=pyproject.toml"} <= set(command)
    assert "--cov-report=term-missing" in command
    assert not {"integration_tests", "integration_tests/", ".", "--cov-append"} & set(command)
    assert not any("*" in arg for arg in command)
    assert len(list(root.glob(".tmp/ci-*/coverage.json"))) == 1
    assert calls[1][1:4] == ["-m", "ruff", "check"]
    assert calls[2][1:5] == ["-m", "ruff", "format", "--check"]
    assert calls[3][1:4] == ["lock", "--check", "--offline"]
    assert calls[4][1:3] == ["pip", "check"]
    assert calls[5][1:4] == ["build", "--offline", "--no-build-isolation"]
    assert all(Path(c[1]).name == "verify_distribution.py" for c in calls[6:8])
    assert calls[8][1:3] == ["sync", "--locked"] and "--no-install-project" in calls[8]
    assert calls[9][1:3] == ["pip", "install"] and "--no-deps" in calls[9]
    assert calls[10][1:3] == ["pip", "check"]
    assert Path(calls[11][1]).name == "smoke_installed.py"
    assert len(calls) == 12


@pytest.mark.parametrize(
    "stage", ["pytest", "ruff", "build", "verify_distribution", "smoke_installed"]
)
def test_nonzero_subprocess_stops_ci(orchestration, stage):
    calls, control, _ = orchestration
    control["fail_stage"] = stage
    with pytest.raises(subprocess.CalledProcessError) as failure:
        ci.main()
    assert failure.value.returncode == 7
    assert calls[-1] == failure.value.cmd
    assert len([c for c in calls if c[1:3] == ["-m", "pytest"]]) == 1


def test_no_stale_evidence_or_non_test_stages_after_missing_report(orchestration):
    calls, control, root = orchestration
    control["write_evidence"] = False
    (root / "coverage.json").write_text(json.dumps(evidence()), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing, unreadable or invalid"):
        ci.main()
    assert len(calls) == 1


@pytest.mark.parametrize("key", ["PYTEST_ADDOPTS", "PYTEST_PLUGINS"])
def test_environment_cannot_expand_selection(orchestration, monkeypatch, key):
    calls, _, _ = orchestration
    monkeypatch.setenv(key, "unapproved")
    with pytest.raises(SystemExit, match="environment overrides"):
        ci.main()
    assert calls == []


def test_coverage_configuration_preserves_floor_and_measurement():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["coverage"]["report"]["fail_under"] == 95
    assert config["tool"]["coverage"]["report"]["precision"] >= 2
    assert config["tool"]["coverage"]["run"] == {"branch": True, "source": ["factoryvision"]}
