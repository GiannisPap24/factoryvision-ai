# Synthetic CPU pipeline baseline V1 (5D-1)

> Historical milestone record. Its local status and next-step statements describe
> that phase; see the [5D integration summary](milestone-5d-integration.md) for
> the later checkpoints and the closed synthetic GPU feasibility decision.

Classification: `synthetic_cpu_pipeline_benchmark`.
**Model execution: `not_run`.** These are software-pipeline diagnostics,
not neural-model latency or model-quality evidence.

## Scope and architecture

The local branch `feat/performance-5d-v1` starts at accepted main
`4c3987e669360e36930ffe8d4317dd040bc0f47c`. No commit or publication
is authorized by this milestone.

Three new modules separate contracts/statistics, a fixed synthetic worker,
and a Windows supervisor. Existing upload decoding, preprocessing, readonly
arrays, analytical diagnostics, display preparation, PNG preview encoding,
Windows Job Object primitives and execution ledgers are reused unchanged.

The worker generates owned RGB8 geometry at 128 x 128 and 512 x 512:
background RGB (18, 32, 48), with the existing fourth synthetic parent's
central rectangle RGB (45, 212, 191). PNG encoding is lossless and in memory;
no EXIF, ICC, input file, dataset, checkpoint or model is used. The worker
receives only the fixed case and repetition identifiers. Images and maps
never cross IPC.

## Protocol and timing definitions

- Six sequential independent workers, interleaved 128/512 for repetitions 1-3.
- Each worker: five discarded warmups and 30 measured repetitions.
  Total: 30 warmups, 180 measured repetitions, 210 pipeline executions.
- Four torch intra-op threads, one inter-op thread; child-local OpenBLAS=1,
  OMP=4 and MKL=4. No CPU affinity, priority or Windows power-policy changes.
- Fixed stage order: generation/PNG; safe decode/validation; existing
  128 x 128 float32 preprocessing; readonly RGB copy and tensor clone;
  analytical RGB mean with an owned float32 display map; heatmap and overlay
  plus three PNG previews capped at 384 pixels; scalar JSON.
- `perf_counter_ns` marks stage boundaries. Seven stage intervals sum to
  the measured pipeline interval. Mark/list bookkeeping is included in the
  following interval; final observation-tuple construction, reference cleanup,
  IPC and aggregation are outside that interval. The original image and
  analytical map retain their native case dimensions.
- Startup-to-ready wall time includes process creation and imports; worker
  import time starts after the minimal stdlib bootstrap. These are reported
  separately from warm pipeline measurements. The resource reader is resolved
  before timing; the unchanged decoder still rereads its two public packaged
  exclusion JSON resources on every pipeline call.
- Percentiles sort samples and linearly interpolate at index `(n-1)*p`.
  SD is sample standard deviation (`n-1` denominator), null for one sample.
  Throughput is sample count divided by the sum of pipeline times, excluding
  startup, warmups, IPC and teardown.
- Per-worker statistics use 30 samples. Pooled statistics use all 90 samples
  only when all three workers for that case complete. Percentiles are never
  averaged; samples within one process are correlated.

## Containment and reporting

Each worker starts suspended, with only two explicitly inherited pipe handles.
The dedicated job has a one-process limit, kill-on-close, and read-back verified
2 GiB process and job committed-memory limits. Assignment and membership are
verified before charging its reservation and resuming it. The deadline is
10 seconds from the start of that worker's setup. Existing bounded cleanup
has up to six seconds of separate grace; the reader-thread join has at most
two further seconds. These are watchdog bounds, not hard real-time guarantees.
Workers run sequentially, with no retries or resume API. The single-use benchmark
object also prevents reuse after uncertain cleanup.

The two ledgers each reserve three fixed workers. A reservation means one
35-repetition synthetic worker, never permission for real-model inference.
No frozen 5C ledger ceiling or 5C PNG/4 KiB IPC contract is changed. The new
scalar IPC has two strictly validated length-prefixed messages per worker,
each at most 16 KiB, read in chunks no larger than 4 KiB; EOF is required
after the result. Malformed, extra, duplicate-key or nonfinite data is rejected.
The final stdout/in-memory report is bounded to 128 KiB. Unknown measurements
remain null; model loading, artifact verification and neural forward remain
`not_run`. Source hashes use role labels, not filenames.

Native memory counters distinguish process peak commit, job peak commit and
maximum *observed* working set sampled about every 10 ms while waiting.
Working set is not guaranteed peak RSS or a commit cap. Peak counters can also
reflect rejected reservations, so their schema does not pretend the limit is
an observation bound. CPU time sums job user and kernel time in 100 ns units,
converted to milliseconds; on failures it is the pre-cleanup snapshot.
Full wall time includes setup, work and cleanup. These are whole-worker
observations including imports, not per-stage allocation measurements.

The child uses the existing interpreter with `-I -S -B`, fixed code/dependency
roots and a minimal process-local environment. No shell or arbitrary launch
argument is exposed. During imports the audit hook denies filesystem writes,
subprocesses, pickle class loading and socket operations, except the installed
library's local hostname lookup; no hostname enters a report. A process-local
temp/cache hint avoids tempfile directory probing. The import-only Inductor
`mkdir` request for the already-existing runtime directory is rejected before
the OS call with `FileExistsError`, which its `exist_ok` check accepts.
No cache directory or file is created.

After imports, reads are restricted to the decoder's two public exclusion JSON
resources; directory discovery, socket operations and subprocesses are denied.
Torch checkpoint loaders, hub downloads, module construction and module calls
are disabled in that child. Installed trusted library code is still imported:
these application guards and Job Objects are not an OS security boundary
against hostile Python/native code or an actor who can modify the installation.

## Observed baseline (2026-10-29)

AMD Ryzen 7 9700X, 16 logical CPUs, Windows AMD64 build 26300.
Python 3.14.10; torch 2.13.0+cpu; torchvision 0.28.0+cpu;
NumPy 2.5.4; Pillow 13.0.0.

All six workers completed: zero failures, timeouts or cancellations in the
measurement run; all exits, empty jobs, closed handles and reader joins verified.
The scalar report was 26857 bytes. Values below are rounded milliseconds;
native report values retain full measured precision.

### 128 x 128: 90 timed samples

| Stage | Min | p50 | p95 | Max | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Generation + PNG | 0.148 | 0.154 | 0.192 | 0.228 | 0.158 | 0.015 |
| Decode + validation | 0.983 | 1.011 | 1.141 | 1.346 | 1.026 | 0.059 |
| Preprocessing | 0.080 | 0.085 | 0.103 | 0.126 | 0.088 | 0.008 |
| Readonly + tensor copy | 0.018 | 0.019 | 0.025 | 0.035 | 0.020 | 0.003 |
| Analytical RGB mean | 0.132 | 0.135 | 0.165 | 0.218 | 0.139 | 0.015 |
| Heatmap + previews | 0.526 | 0.536 | 0.606 | 0.781 | 0.550 | 0.051 |
| Scalar JSON | 0.013 | 0.013 | 0.015 | 0.029 | 0.014 | 0.002 |
| Total pipeline | 1.911 | 1.962 | 2.094 | 2.633 | 1.994 | 0.134 |

Warm throughput: 501.38 images/s.

### 512 x 512: 90 timed samples

| Stage | Min | p50 | p95 | Max | Mean | Sample SD |
|---|---:|---:|---:|---:|---:|---:|
| Generation + PNG | 1.820 | 1.861 | 2.086 | 2.806 | 1.913 | 0.188 |
| Decode + validation | 2.124 | 2.227 | 2.600 | 2.747 | 2.263 | 0.127 |
| Preprocessing | 0.618 | 0.643 | 0.735 | 0.768 | 0.651 | 0.034 |
| Readonly + tensor copy | 0.113 | 0.120 | 0.152 | 0.289 | 0.126 | 0.021 |
| Analytical RGB mean | 2.194 | 2.345 | 2.798 | 3.409 | 2.392 | 0.216 |
| Heatmap + previews | 11.222 | 11.502 | 12.208 | 14.883 | 11.627 | 0.528 |
| Scalar JSON | 0.021 | 0.024 | 0.028 | 0.032 | 0.024 | 0.002 |
| Total pipeline | 18.315 | 18.751 | 20.616 | 23.137 | 18.996 | 0.851 |

Warm throughput: 52.64 images/s.

### Between-process observations

| Case / worker | Pipeline p50 ms | p95 ms | Mean ms | Sample SD ms | Images/s |
|---|---:|---:|---:|---:|---:|
| 128 / 1 | 1.967 | 2.577 | 2.054 | 0.212 | 486.90 |
| 512 / 1 | 18.632 | 19.161 | 18.812 | 0.723 | 53.16 |
| 128 / 2 | 1.959 | 2.057 | 1.964 | 0.049 | 509.26 |
| 512 / 2 | 18.738 | 19.139 | 18.808 | 0.216 | 53.17 |
| 128 / 3 | 1.958 | 2.040 | 1.966 | 0.041 | 508.64 |
| 512 / 3 | 18.881 | 22.215 | 19.366 | 1.199 | 51.64 |

### Whole-worker resources

| Case / worker | Import ms | Start to ready ms | Full wall ms | CPU ms | Process commit MiB | Job commit MiB | Sampled working set MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 / 1 | 1408.13 | 1448.50 | 1858.08 | 2609.38 | 303.81 | 304.95 | 293.52 |
| 512 / 1 | 1383.57 | 1419.76 | 2420.41 | 4968.75 | 324.45 | 325.60 | 313.94 |
| 128 / 2 | 1368.77 | 1405.03 | 1798.63 | 2578.13 | 304.31 | 305.45 | 293.86 |
| 512 / 2 | 1372.70 | 1408.88 | 2412.61 | 4906.25 | 324.73 | 325.88 | 314.11 |
| 128 / 3 | 1365.36 | 1401.16 | 1799.33 | 2625.00 | 303.73 | 304.88 | 293.37 |
| 512 / 3 | 1373.36 | 1409.86 | 2417.19 | 5015.63 | 325.67 | 326.80 | 314.02 |

Source SHA-256 at this measurement:

| Role | SHA-256 |
|---|---|
| contracts | `ef14a691248fefb0f34ff772efbee9a1d192e40ab00adac8d361c93d2f18c55f` |
| supervisor | `60ef4e892fd35b5571e6f3340b1e597dfca5ef7534a161c7b927d498d1d0055c` |
| worker | `32070f6eb0f0bca308e8e7e8ec0158ae578c97b5d11e8f103e6dfbdd998799b9` |

## Focused acceptance

Run from the public staging checkout using the existing environment:

```powershell
.venv-31410/Scripts/python.exe -B -m factoryvision.performance_supervisor
.venv-31410/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -s tests/test_performance_baseline.py integration_tests/test_performance_isolation.py
```

The directly affected regression command was:

```powershell
.venv-31410/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_robustness_execution_ledger.py tests/test_robustness_reporting.py tests/test_windows_isolation.py::test_proposed_2gib_setting_query_only_no_large_allocation tests/test_windows_isolation.py::test_kill_on_close_fallback_confirms_exit_but_does_not_hide_cleanup_fault tests/test_windows_isolation.py::test_structures_match_windows_64bit_abi tests/test_cashew_isolated_ipc.py::test_five_real_ipc_attempts_determinism_and_existing_diagnostics
```

The benchmark command writes scalar stdout only. The test command additionally
prints a bounded scalar evidence summary. It does not persist benchmark output.

Final new tests: **71 passed in 38.90 seconds**. Coverage includes exact inputs,
immutability, preprocessing equivalence, tensor/map ownership, display behavior,
controlled-clock stage timings, counts, percentiles, pooling, size limits,
metadata rejection, native memory semantics, six independent processes,
pre-launch and active cancellation, the actual 10-second timeout, malformed/
oversized/trailing output, forbidden writes/model construction, assignment
failure, verified launch ordering, cleanup and no retry/reuse.

Direct regressions: **67 passed in 2.07 seconds**, covering the existing execution
ledger and scalar reporting, Windows ABI/2 GiB readback/kill-on-close fallback,
and the existing five-case synthetic IPC determinism test. All remained
synthetic; no real neural inference was performed. Ruff lint and formatting
checks passed for the five new Python files. Both tracked `git diff --check`
and individual untracked-file whitespace checks passed. The full historical suite,
packaging, installed-app smoke test and CI were intentionally deferred to
the integrated 5D acceptance gate.

## Security, privacy and scientific review

The final scope is three source modules, two test files and this document.
No production source is edited. The review inspected all six new text files,
including their import, process, IPC, error-handling and reporting paths.
No credentials, personal emails, private paths, real image files, weights,
datasets, private reports or executable binaries are introduced. Generic
privacy-negative test strings are deliberate rejection fixtures.

The preflight discovered and reused shell/Git, the existing Python environment,
pytest/Ruff, Job Object support and the installed Plugin Management and Security
Best Practices skills. GitHub/CI and browser/computer-use capabilities were
available but unnecessary for this offline milestone. Browser/Chrome connection
was not exercised or claimed. No capability installation, account connection,
elevation, dependency/lockfile change or OS-security change was needed. The
security skill has no reference specific to a local Windows Python worker;
the review therefore uses the actual code, existing isolation contract and
the focused tests rather than claiming a framework-specific audit.

All nine scientific source resources were verified against their reviewed
SHA-256 pins and accepted-main bytes. Frozen thresholds, score definitions, artifact
identities, research metrics, Model Registry, Human Review and 5C protocols
remain unchanged. Scientific classifications remain:

- PCB1 BLOCKED; PCB2 INCONCLUSIVE.
- Cashew official test already observed; real Cashew development pilot BLOCKED.
- MetalParts exploratory-only; RobustAD target evaluation NOT CLEARED.
- Human Review is not verified ground truth.

The private development repository is neither read nor modified. Existing
VS Code windows, sessions and terminals are left undisturbed. Nothing is
staged, committed, pushed, merged or released.

## Limits and next decision

This is one machine/session with highly compressible, two-colour synthetic
geometry and warm caches. No thread-count comparison, confidence interval,
cold-file I/O, real-image distribution, model latency, GPU speedup, ONNX parity,
OOD accuracy or production readiness is established. Polling, audit guards,
shared desktop activity, CPU frequency changes and allocator/import behavior
affect results. Failed startup diagnostics during development were resolved
without widening memory/deadline limits; they are not included in the accepted
timing sample.

The first proposed optimization target is 512-pixel display/preview preparation,
about 61.2% of its measured mean pipeline time. At 128 pixels, decode/validation
accounts for about 51.4%; profiling the repeated public-resource parsing is a secondary
candidate, with protected-input filtering semantics preserved. Neither change
has been implemented.

**5D-1: GO for owner review. 5D-2: NEEDS DECISION / separate authorization.**
Any later optimization needs explicit scope and numerical equivalence checks.
A full CI-equivalent run, packaging/privacy scan and fresh installed-application
test remain required at integrated 5D acceptance. This milestone stops here.
