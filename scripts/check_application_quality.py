"""Current offline quality checks without overwriting historical research reports."""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report = root / "reports" / ("application-quality-" + uuid.uuid4().hex)
report.mkdir(exist_ok=False)
env = {**os.environ, "UV_OFFLINE": "1", "PYTHONNOUSERSITE": "1"}
uv = str(root / ".tools/uv-0.13.5/uv.exe")
targets = ["src", "tests", "integration_tests", "scripts"]
commands = [
    [
        sys.executable,
        "-m",
        "pytest",
        "--cov=factoryvision",
        "--cov-branch",
        "--cov-report=term-missing",
        f"--cov-report=json:{report / 'coverage.json'}",
    ],
    [sys.executable, "-m", "ruff", "check", *targets],
    [sys.executable, "-m", "ruff", "format", "--check", *targets],
    [uv, "lock", "--check", "--offline", "--python", sys.executable],
    [uv, "pip", "check", "--python", sys.executable],
]
results = []
for command in commands:
    print("Running:", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=root, env=env, check=False)
    results.append({"command": command, "exit_code": result.returncode})
(report / "quality.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
raise SystemExit(0 if all(r["exit_code"] == 0 for r in results) else 1)
