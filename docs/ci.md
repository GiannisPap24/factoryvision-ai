# Source-only Windows verification

**Runtime:** Windows AMD64, Python 3.14.10, uv 0.13.5.
Owned source is MIT licensed. This source-only candidate contains no datasets or weights.

Hosted GitHub Actions: **NOT RUN — REQUIRES REMOTE REPOSITORY**.
Local equivalence does not prove hosted execution.

## Fresh source setup

Follow [verified Python and uv setup](local-demo.md#first-installation), including
the development dependencies. Then run from the source root:

```powershell
$env:UV_PROJECT_ENVIRONMENT = Join-Path (Get-Location) '.venv-31410'
.venv-31410/Scripts/python.exe scripts/ci_check.py --uv .tools/uv-0.13.5/uv.exe
```

## Workflow

`.github/workflows/verify.yml` uses normal push and pull-request triggers,
`windows-2025`, a 30-minute timeout and cancellation of superseded runs.
Permissions are `contents: read`; checkout does not persist credentials.
No privileged PR trigger, secret inputs, deployment, upload steps or shared cache
are configured. uv's local download cache is confined to the disposable checkout;
it is not restored from or saved to GitHub Actions cache.

| Action | Verified upstream release | Immutable commit |
|---|---|---|
| actions/checkout | v7.0.1 | `3d3c42e5aac5ba805825da76410c181273ba90b1` |
| astral-sh/setup-uv | v10.3.2 | `07a047e608db84b3d8c06cd9a7c7dcc9c355c18e` |

Action tags and commit objects were verified against upstream in R3. uv remains
pinned to 0.13.5. CI downloads the official Python.org 3.14.10 full-runtime ZIP,
checks its release-manifest SHA-256 and valid PSF Authenticode signer, then verifies
the exact version. It installs locked dependencies into `.venv-31410`. Automatic
Python downloads remain disabled; no uv-managed runtime availability is assumed.
Local verification can reuse that exact verified runtime without redownloading it.

`scripts/ci_check.py` runs the complete default fabricated-data suite with branch
coverage (95% floor), Ruff lint/format, offline lock consistency, dependency checks,
wheel/sdist builds, forbidden-member/content privacy scans and packaged-document link checks. It then creates a fresh runtime
environment, installs the wheel without the source package, checks dependencies
and launches AppTest from an empty working directory. That smoke asserts the import
comes from `site-packages`, six missing models, catalog presentation, explicit
synthetic generation/exports, resets and unavailable batch behavior. Network access
is denied except Windows asyncio's same-process loopback socket pair.

Generated environments and builds remain in unique ignored `.tmp/ci-*` and
`dist/ci-*` directories; checks never overwrite historical reports. No automatic
cleanup touches research directories. CI stdout contains runner diagnostics; no
private research artifacts are supplied to the job or uploaded separately.

## Validation and remaining gates

Workflow syntax/expressions were checked with upstream `actionlint` 1.7.12. Its
Windows archive SHA-256 was verified against upstream release checksums and asset
digest before local execution. The private 4E acceptance record contains actual local-equivalent results and
source-only installation evidence. It is not distributed with public source.

Full SHA pinning reduces mutable-action risk; it does not establish that every
upstream dependency is safe. Hosted runner images and bootstrap registries remain
external dependencies. Review updates and lock changes explicitly. Fresh hosted
installation, runner behavior and PR restrictions must be observed after a future
owner-approved remote exists; no hosted support is claimed now.

Sources: [GitHub secure use](https://docs.github.com/en/actions/reference/security/secure-use),
[Windows 2025 runner image](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md),
[setup-uv](https://github.com/astral-sh/setup-uv),
[uv GitHub integration](https://docs.astral.sh/uv/guides/integration/github/),
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
