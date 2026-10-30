"""Windows source-only CI equivalent; bootstrap dependencies before invoking offline."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path


def require_coverage(report):
    """Enforce the 95% combined floor from this run's unrounded integer counts."""
    try:
        evidence = json.loads(report.read_text(encoding="utf-8"))
        totals = evidence["totals"]
        counts = [
            totals[key]
            for key in ("covered_lines", "num_statements", "covered_branches", "num_branches")
        ]
        if evidence["meta"]["branch_coverage"] is not True or any(
            type(value) is not int or value < 0 for value in counts
        ):
            raise ValueError
        lines, statements, branches, possible_branches = counts
        if not (0 <= lines <= statements and 0 <= branches <= possible_branches):
            raise ValueError
        covered, possible = lines + branches, statements + possible_branches
        if statements == 0 or possible == 0:
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise SystemExit(
            "FAIL: missing, unreadable or invalid combined coverage evidence."
        ) from None
    # Report precision can still round 94.999% to 95.00%; never use it for acceptance.
    if 100 * covered < 95 * possible:
        raise SystemExit(f"FAIL: exact combined coverage {covered}/{possible} is below 95%.")
    print(
        f"PASS: exact combined coverage {covered}/{possible} "
        f"({100 * covered / possible:.8f}%) >= 95%.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args()
    uv = shutil.which(args.uv)
    if not uv:
        raise SystemExit("uv must already be installed (0.13.5 or later).")
    if sys.platform != "win32" or sys.version_info[:3] != (3, 14, 10):
        raise SystemExit("Verified baseline: Windows AMD64, Python 3.14.10.")
    if any(os.environ.get(key) for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")):
        raise SystemExit("CI requires pytest options/plugins without environment overrides.")
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / ("ci-" + uuid.uuid4().hex)
    temporary = root / ".tmp" / ("ci-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    env = {
        **os.environ,
        "UV_OFFLINE": "1",
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "PYTHONNOUSERSITE": "1",
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(root / ".uv-cache")),
    }

    def run(command, cwd=root, extra=None):
        subprocess.run(command, cwd=cwd, env={**env, **(extra or {})}, check=True)

    coverage_report = temporary / "coverage.json"
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            "pyproject.toml",
            "--cov=factoryvision",
            "--cov-branch",
            "--cov-config=pyproject.toml",
            "--cov-report=term-missing",
            f"--cov-report=json:{coverage_report}",
            f"--junitxml={temporary / 'tests.xml'}",
            "tests/",
            "integration_tests/test_performance_isolation.py",
        ]
    )
    require_coverage(coverage_report)
    targets = ["src", "tests", "integration_tests", "scripts"]
    run([sys.executable, "-m", "ruff", "check", *targets])
    run([sys.executable, "-m", "ruff", "format", "--check", *targets])
    run([uv, "lock", "--check", "--offline", "--python", sys.executable])
    run([uv, "pip", "check", "--python", sys.executable])
    run(
        [
            uv,
            "build",
            "--offline",
            "--no-build-isolation",
            "--python",
            sys.executable,
            "--out-dir",
            str(output),
        ]
    )
    for archive in output.glob("factoryvision_ai-*"):
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as package:
                names = package.namelist()
        else:
            with tarfile.open(archive) as package:
                names = [member.name.split("/", 1)[-1] for member in package.getmembers()]
        if any(
            name.startswith(("data/", "runs/", "reports/", ".tmp/", ".tools/", ".venv/"))
            or name.lower().endswith((".pt", ".pth", ".png", ".jpg", ".jpeg", ".env"))
            for name in names
        ):
            raise SystemExit("Unexpected research artifact in distribution.")
        print(f"Distribution inspected: {archive.name}, {len(names)} members", flush=True)
        run([sys.executable, str(root / "scripts/verify_distribution.py"), str(archive)])
    isolated = temporary / "installed"
    python = isolated / "Scripts/python.exe"
    run(
        [
            uv,
            "sync",
            "--locked",
            "--extra",
            "app",
            "--no-dev",
            "--no-install-project",
            "--python",
            sys.executable,
        ],
        extra={"UV_PROJECT_ENVIRONMENT": str(isolated)},
    )
    wheel = next(output.glob("*.whl"))
    run([uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)])
    run([uv, "pip", "check", "--python", str(python)])
    empty = temporary / "empty"
    empty.mkdir()
    run([str(python), str(root / "scripts/smoke_installed.py")], cwd=empty)
    print("PASS: offline source, distributions and fresh installed application.", flush=True)


if __name__ == "__main__":
    main()
