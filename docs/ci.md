# Source-only Windows verification

**Runtime:** Windows AMD64, Python 3.14.10, uv 0.13.5.
Owned source is MIT licensed. This source-only release contains no datasets or weights.

Accepted public main: **1,448 passed, 97.61% combined coverage**, at
`4c3987e669360e36930ffe8d4317dd040bc0f47c` (5C merge, PR #3).
[Run 40795664976](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40795664976)
also passed Windows packaging/privacy checks and fresh-install smoke. This is
baseline evidence, not a CI result for unpublished 5D or launch-readiness changes.

Historical initial publication: **PASSED on 2026-10-26**.
[Run 40376270944](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40376270944)
verified root commit `691507dd409177219b4ba39afedfaaedfe440c06`: **711 tests passed,
98.09% combined statement and branch coverage**, plus lint/format, lock/dependency, distribution and installed-app checks.
The local audit had 710 passed and one Windows symlink-permission skip; the hosted
runner passed that test too. [Later workflow runs](https://github.com/GiannisPap24/factoryvision-ai/actions)
have their own status; this result applies to the linked run.

## Interpreting coverage

The suite enables branch measurement with `--cov-branch`. Coverage.py's total
combines executed statements and branch outcomes:

`(covered statements + covered branch outcomes) / (all statements + all branch outcomes)`.

Both the current **97.61%** and historical **98.09%** are combined totals, not branch-only coverage. The
configured 95% floor applies to the combined report. Test counts can differ locally
when Windows does not grant symlink-creation privileges; the skip is reported
explicitly and must not be counted as a pass. Historical CI links identify the
exact commit tested; use the workflow badge or current runs for the latest state.

The authoritative runner selects exactly `tests/` and
`integration_tests/test_performance_isolation.py` in one pytest invocation. It
never collects the entire integration directory. Pytest option/plugin environment
overrides are rejected before collection. No real-data integration tests are part
of this gate.

Coverage reports use two decimal places, retaining `fail_under = 95`. Precision
alone can still round a value just below 95% upward. After pytest succeeds, the
runner checks the same run's JSON statement/branch counts using the exact integer
comparison `100 * covered >= 95 * possible`. Missing, unreadable, malformed or
empty evidence fails closed; a nonzero pytest result stops CI immediately. The
JSON report and JUnit selection record use a fresh ignored `.tmp/ci-*` directory,
so an old coverage report cannot satisfy the current gate. No second test run or
coverage exclusion is introduced.

This repairs the PR #5 discrepancy in
[run 40932934406](https://github.com/GiannisPap24/factoryvision-ai/actions/runs/40932934406):
1,653 passing tests and 94.71% actual coverage produced a green job because of
integer rounding, while the approved performance isolation module was omitted.
That run does not satisfy the 95% acceptance floor. See the official coverage.py
[precision/fail-under configuration](https://coverage.readthedocs.io/en/latest/config.html#report-fail-under)
and [JSON reporting](https://coverage.readthedocs.io/en/latest/commands/cmd_json.html).

## Fresh source setup

The launch-readiness candidate constrains its app extra to pandas 3.0.6 and
Streamlit 1.66.1. See the [Windows compatibility rationale](local-demo.md#windows-application-dependency-compatibility),
including the unresolved earlier import timeout and limits of SAC compatibility
evidence. Acceptance must use a fresh environment created from the candidate
lockfile, followed by a fresh installed-wheel smoke. The earlier manually
overridden experiment and the public-main counts above are separate evidence;
they do not establish full acceptance of this candidate or validate pandas 3.1.0.

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

`scripts/ci_check.py` runs the explicit fabricated-data and synthetic performance
isolation allowlist above with combined branch/statement coverage (exact 95% floor),
Ruff lint/format, offline lock consistency, dependency checks,
wheel/sdist builds, forbidden-member/content privacy scans and packaged-document link checks. It then creates a fresh runtime
environment, installs the wheel without the source package, checks dependencies
and launches AppTest from an empty working directory. That smoke asserts the import
comes from `site-packages`, six missing models, catalog presentation, explicit
synthetic generation/exports, resets and unavailable batch behavior. Network access
is denied except Windows asyncio's same-process loopback socket pair.

The merged 5C verification also checks the installed analytical robustness runner, denied
real-development admission and five requests through the fixed synthetic Windows
IPC worker. These run from the empty directory using packaged modules, without
datasets or model artifacts. The full suite includes actual Windows isolation,
crash-containment and IPC tests; skips must be reported separately from passes.
The initial publication counts do not describe the later 5C merge. See the
[current scope and runnable examples](protocols/robustness-source-only.md).

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
installation and runner behavior passed in the linked initial push run. Fork-PR
behavior was not exercised by that push run and remains a separate verification scope.

Sources: [GitHub secure use](https://docs.github.com/en/actions/reference/security/secure-use),
[Windows 2025 runner image](https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-Readme.md),
[setup-uv](https://github.com/astral-sh/setup-uv),
[uv GitHub integration](https://docs.astral.sh/uv/guides/integration/github/),
[Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
