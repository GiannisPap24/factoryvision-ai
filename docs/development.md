# Development

The supported baseline is Windows AMD64, Python 3.14.10 and uv 0.13.5.
Start with [local setup](local-demo.md#first-installation). Keep the development
dependencies (the default sync includes them) and select `.venv-31410` in VS Code.
Workspace settings recommend the official OpenAI Codex extension, Python and Ruff.
Tasks and debugging are explicit; opening the folder starts no application or experiment.

## Offline acceptance

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
$env:UV_OFFLINE = '1'
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

See [CI details](ci.md). Initial dependency installation needs upstream access;
subsequent checks use the populated cache. The command tests fabricated inputs,
coverage, lint, format, lock, dependency compatibility, source-only distributions
and an installed-wheel application from an empty directory. It writes only local
ignored verification environments and builds. Review those before deleting them.

## Useful explicit commands

```powershell
.venv-31410/Scripts/python.exe -m factoryvision validate-manifest tests/fixtures/synthetic-manifest.json --json
.venv-31410/Scripts/python.exe -m pytest tests/test_app.py tests/test_catalog_demo.py tests/test_reporting.py
.venv-31410/Scripts/python.exe scripts/debug_catalog_demo.py
.venv-31410/Scripts/python.exe -m factoryvision.app
```

The module form is the verified CLI entry point. The generated console executable
may be blocked by local Windows application-control policy; do not bypass policy.
Use **FactoryVision: debug catalog and synthetic demo** for a dataset-free debugger
session. Six definitions, three demo validation rows and null inference scores are expected.

## Research boundaries

`integration_tests/` and research/debugging scripts are opt-in and can read real
assets or train models. They are not part of source-only acceptance. Read each
entry point and [reproduction prerequisites](reproduction.md) before use; do not
run the unrestricted integration suite or reopen completed official evaluations.
New experiments require separately prepared data, approvals and fresh run identities.

Keep source, tests, public documentation and `uv.lock` trackable. Datasets, runs,
private reports, environments, caches, screenshots and weights remain excluded.
Review staged content before committing. Public publication requires the separate
[owner decision and procedure](publication.md).
