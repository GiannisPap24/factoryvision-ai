"""Run the same offline checks from VS Code and a terminal; never install packages."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report-dir", type=Path, default=Path("reports"))
args = parser.parse_args()
reports = (root / args.report_dir).resolve()
if not reports.is_relative_to((root / "reports").resolve()):
    parser.error("--report-dir must stay inside project reports/")
reports.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "UV_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
uv = root / ".tools/uv-0.13.5/uv.exe"
commands = [
    [
        sys.executable,
        "-m",
        "pytest",
        "--cov=factoryvision",
        "--cov-branch",
        "--cov-report=term-missing",
        f"--cov-report=json:{reports / 'coverage.json'}",
        f"--cov-report=html:{reports / 'htmlcov'}",
    ],
    [sys.executable, "-m", "ruff", "check", "src", "tests", "integration_tests", "scripts"],
    [
        sys.executable,
        "-m",
        "ruff",
        "format",
        "--check",
        "src",
        "tests",
        "integration_tests",
        "scripts",
    ],
    [str(uv), "lock", "--check", "--offline", "--python", sys.executable],
    [str(uv), "pip", "check", "--python", sys.executable],
]
results = []
for command in commands:
    print(f"Running: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=root, env=env, check=False)
    results.append({"command": command, "exit_code": result.returncode})
(reports / "quality.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
passed = all(result["exit_code"] == 0 for result in results)
print("Quality checks passed." if passed else "Quality checks failed.", flush=True)
raise SystemExit(0 if passed else 1)
