# 5D-2: Display profiling and a CPU allocation candidate

> Historical milestone record. Its local status and next-step statements describe
> that phase; see the [5D integration summary](milestone-5d-integration.md) for
> the later checkpoints and the closed synthetic GPU feasibility decision.

**Technical equivalence checks: PASS. Latency benefit: INCONCLUSIVE.**
The candidate is retained locally for owner review; 5D-2 is not committed.
No research model, dataset, checkpoint, GPU or conversion runtime was used.

## Checkpoint and preflight

The owner-approved 5D-1 checkpoint is
`ceb06ba3e973a140d954d645831d954a7f581a46`, with parent
`4c3987e669360e36930ffe8d4317dd040bc0f47c`.
It adds exactly the six reviewed 5D-1 files. Author and committer use the
verified GitHub noreply identity. The tree was clean immediately afterward.
The branch remains `feat/performance-5d-v1`; remote main was verified at the
accepted parent. No branch switch, reset, pull, rebase or publication occurred.

The mandatory preflight inspected available tools and the installed Plugin
Management and Security Best Practices instructions. Existing Git, Python
3.14.10, pytest 9.1.1, Ruff 0.17.2 and stdlib timing/tracing were sufficient.
No installation, external account connection, elevation or dependency change
was required. Browser/Computer Use skills and callable CUA were available;
no browser operation or Chrome connection verification was needed or claimed.
The security skill has no local Windows-worker-specific reference, so review
used the actual fixed worker/isolation code and focused tests.

The unchanged 5D-1 source hashes and recorded 71 tests / 67 regressions /
six successful workers were rechecked against their accepted report.
All six complete files and the staged diff were reviewed before the single
local checkpoint commit. Historical acceptance tests were not rerun merely
to create that checkpoint.

## Actual application path

Single Cashew and MetalParts inspection presentation both call the shared
`inference.visualization.panels`. Batch `compact` calls the same function,
then `batch.png` for original, heatmap, overlay and optional auxiliary panels.
The 5D-1 display stage includes shared panels plus three 384-pixel PNG previews;
single inspection uses the PIL panels directly. It does not include browser
layout, Streamlit transport or any neural forward pass.

The common pipeline reprojects original RGB and the spatial map to at most
1024 pixels per side, divides by the original raw-map maximum, clips display
intensity, interpolates the unchanged four palette stops, rounds to RGB8 and
blends with alpha 0.45. Reconstruction/reference use the original clip/round/
bilinear resize behavior. Preview encoding copies each PIL image, applies
the existing bilinear thumbnail operation, then encodes PNG in memory.

## Bounded profiling evidence

A private finite diagnostic mode in the existing worker attributes elapsed
time to the actual two shared functions. It maps only their known statements
to fixed labels, using AST inspection before the worker's read guard.
It neither executes source strings nor accepts a module, command or path.
The existing job, deadline, memory, allowlist, accounting and cleanup remain
unchanged. The mode adds ten instrumented display calls after the ordinary
5 warmups + 30 pipeline calls. Normal A/B runs do not enable it.

The tracer is restored in a finally block. Only thirteen validated finite,
nonnegative scalar mean-ms fields cross IPC; no source paths, line numbers,
pixels or metadata are exported. The existing 16 KiB frame cap is unchanged.
No raw profiling/benchmark files are automatically written.

Exploratory 512-pixel attribution, ten display calls per profile:

| Operation group | Baseline mean ms | Candidate mean ms |
|---|---:|---:|
| Geometry | 0.009 | 0.008 |
| Original RGB conversion + resizing | 0.347 | 0.352 |
| Spatial resizing + array conversion | 0.485 | 0.508 |
| Maximum + normalization | 0.270 | 0.292 |
| Palette allocation | 0.012 | 0.011 |
| Color interpolation + stacking | 4.167 | 4.479 |
| Final RGB conversion | 1.196 | 0.214 |
| Overlay blend | 0.525 | 0.548 |
| Output / auxiliary handling | 0.027 | 0.027 |
| Preview copies | 0.319 | 0.342 |
| Preview resizing | 2.887 | 2.918 |
| PNG encoding | 1.821 | 1.815 |
| Buffer/output | 0.008 | 0.007 |

Line attribution includes instrumentation and child-call overhead. It is a
hotspot diagnostic, not the uninstrumented A/B result. In the candidate,
per-channel rounding/casting moves into the interpolation/stacking group:
compare those two RGB groups together (5.363 versus 4.693 ms), not their
individual labels as if operation boundaries were unchanged. Allocation cost
is included in these groups, not separately estimated as allocator latency.

## One production change

Only the shared renderer changes production behavior internally: round and
cast each interpolated channel to uint8 before stacking, instead of stacking
three float64 channels then rounding/casting the whole RGB array.

The interpolation remains float64; no extra quantization, approximate palette,
lookup table, cache, resampling shortcut or score calculation is introduced.
The elementwise round/cast operations commute with channel stacking.
At 512 x 512, the stacked RGB allocation becomes 0.75 MiB instead of 6 MiB;
the large whole-RGB rounded temporary is also avoided. Observed job/process
commit improves, without an observed memory regression.

Palette construction was only about 0.012 ms; it was deliberately left alone.
Preview resampling/encoding, normalization, input protection and every model
interface are unchanged.

## Exactness evidence

The independent test oracle retains the accepted checkpoint renderer.
Its function-body AST was verified identical to checkpoint source.
Tests compare exact PIL pixels, dimensions, ceiling and deterministic PNG
bytes; no numerical tolerance was introduced.

Coverage includes float32/float64 maps, zero/positive constant/negative/tiny
maps, ramps and deterministic random maps, palette endpoints, neighboring
float32 values around half-rounding boundaries, noncontiguous arrays,
rectangular inputs and the 1024-pixel cap. Original/map/auxiliary immutability,
reconstruction/reference, absent auxiliaries, PNG preview bounds and metadata
stripping are checked.

Float64 subnormals below Pillow's float32 range already produce invalid-divide/
cast diagnostics in the baseline. Tests explicitly compare warning categories
and messages and exact resulting pixels. Warning multiplicity under an
always-warn filter can differ because casting now occurs per channel.
No new tolerance or suppression was added to production code.

Fabricated records for all six model identifiers exercise batch compaction
and both single-inspection presentation functions. Captured image payloads
match the oracle exactly; model scores, thresholds, flags and artifact
identities remain unchanged. The UI tests use a presentation double, not a
browser render. Model construction/calls/loading, socket creation and Python
file writes are denied by the focused test fixture; real datasets are never
used. The isolated worker additionally enforces its accepted read/write guards.

## Controlled A/B result

Both arms used the same Ryzen 7 9700X / Windows AMD64 host, Python 3.14.10,
torch 2.13.0+cpu, torchvision 0.28.0+cpu, NumPy 2.5.4 and Pillow 13.0.0.
Each arm used three independent workers per image size, five warmups and
30 timed repetitions per worker: 90 samples per size per arm.
Thread settings, fixed images, stage boundaries and isolation were identical.
The harness source hashes are identical across A and B; only renderer source
differs. The final comparison has 12 successful workers, zero failures and
zero timeouts, all with verified cleanup.

Baseline A ran before the production edit; candidate B ran afterward.
This is sequential A-then-B, not randomized or interleaved treatment assignment.
No fastest-run selection or repeated search for a desired result was used.
The historical 5D-1 medians are context only, not the comparator for B.

Percentiles use linear interpolation at (n-1)*p. SD is sample SD.
All latency values below are ms.

| Size / stage | Baseline p50 | Candidate p50 | Change | Baseline p95 | Candidate p95 | Baseline mean +/- SD | Candidate mean +/- SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 / display_preview | 0.537 | 0.536 | -0.20% | 0.770 | 0.761 | 0.563 +/- 0.074 | 0.558 +/- 0.061 |
| 128 / pipeline | 1.957 | 1.969 | 0.59% | 2.544 | 2.447 | 2.031 +/- 0.180 | 2.019 +/- 0.150 |
| 512 / display_preview | 11.310 | 10.570 | -6.55% | 14.577 | 13.443 | 11.661 +/- 1.016 | 10.895 +/- 0.958 |
| 512 / pipeline | 18.563 | 17.760 | -4.32% | 22.626 | 21.743 | 19.098 +/- 1.399 | 18.293 +/- 1.442 |

| Size / worker | Baseline display mean +/- SD | Candidate display mean +/- SD | Baseline pipeline mean +/- SD | Candidate pipeline mean +/- SD |
|---|---:|---:|---:|---:|
| 128 / 1 | 0.599 +/- 0.104 | 0.544 +/- 0.019 | 2.130 +/- 0.266 | 1.986 +/- 0.042 |
| 512 / 1 | 11.565 +/- 0.777 | 10.627 +/- 0.568 | 18.927 +/- 1.054 | 17.808 +/- 0.729 |
| 128 / 2 | 0.555 +/- 0.059 | 0.587 +/- 0.099 | 2.000 +/- 0.105 | 2.103 +/- 0.233 |
| 512 / 2 | 11.615 +/- 1.067 | 11.074 +/- 1.206 | 19.021 +/- 1.461 | 18.644 +/- 1.737 |
| 128 / 3 | 0.534 +/- 0.006 | 0.543 +/- 0.014 | 1.964 +/- 0.036 | 1.968 +/- 0.046 |
| 512 / 3 | 11.803 +/- 1.179 | 10.985 +/- 0.960 | 19.347 +/- 1.631 | 18.428 +/- 1.562 |

Warm throughput (startup/warmups/teardown excluded):

| Size | Baseline images/s | Candidate images/s |
|---|---:|---:|
| 128 | 492.29 | 495.27 |
| 512 | 52.36 | 54.66 |

At 512 pixels the observed display median decreased 6.55%, and full-pipeline
median decreased 4.32%. However, mean differences (0.766 and 0.805 ms) are
smaller than the pooled per-sample SDs (about 0.96-1.02 and 1.40-1.44 ms).
Per-worker means all trend lower, but three processes per arm and sequential
ordering do not resolve the uncertainty. **Latency benefit is INCONCLUSIVE**.
At 128 pixels, differences are also within observed variation; pipeline median
increased 0.59%. No general speedup is claimed.

## Resources

| Size / metric | Baseline min-max | Candidate min-max |
|---|---:|---:|
| 128 / peak_process_commit (MiB) | 302.67-304.61 | 302.65-303.73 |
| 128 / peak_job_commit (MiB) | 303.82-305.74 | 303.79-304.87 |
| 128 / observed_working_set (MiB) | 292.37-294.13 | 292.29-293.16 |
| 128 / startup_to_ready_ms | 1412.41-1418.48 | 1410.02-1422.23 |
| 128 / import_ms | 1376.48-1382.60 | 1374.39-1386.90 |
| 128 / wall_ms | 1814.47-1879.45 | 1820.87-1830.61 |
| 128 / cpu_ms | 2578.13-2718.75 | 2593.75-2640.63 |
| 512 / peak_process_commit (MiB) | 324.46-324.88 | 316.53-316.94 |
| 512 / peak_job_commit (MiB) | 325.59-326.02 | 317.67-318.07 |
| 512 / observed_working_set (MiB) | 311.54-314.86 | 305.95-306.48 |
| 512 / startup_to_ready_ms | 1410.84-1416.88 | 1407.26-1418.11 |
| 512 / import_ms | 1375.15-1378.72 | 1371.40-1381.94 |
| 512 / wall_ms | 2412.04-2438.90 | 2368.95-2407.84 |
| 512 / cpu_ms | 4953.13-5000.00 | 4781.25-4921.88 |

Memory columns are bytes converted to MiB; time columns are ms.
Process/job figures are committed-memory peaks, not RSS. Working set is sampled,
not guaranteed peak RSS. CPU time sums job user/kernel time, so it may exceed
wall time. Counters include imports and warmups, not solely display work.

At 512 pixels, mean peak process commit decreased by about 7.96 MiB and mean
peak job commit by about 7.98 MiB. Every candidate worker's peak commit was
below every baseline worker's at that size. At 128 pixels, memory differences
were small. No accepted memory/deadline limit was increased.

Source provenance:

| Component | SHA-256 |
|---|---|
| Accepted renderer | `f00cdbb8bf3dff771df2f3df70e5fa3b5d174778e32a931f75296f83afba8db9` |
| Candidate renderer | `7e0531af849fcc3f6ef759ef4eb0f466229176d5f3968a819fdff95231d37377` |
| Fixed profiler | `2a884de25d35d39c7cbafa0bd41d53093260b919d3bfd905202fb42bf32d7553` |
| A/B harness: contracts | `61b8a3a4a27b1440211ecffc24bfe338589b758e47a2b92f89723e0d62d6a14b` |
| A/B harness: worker | `9db40d4f020c24a0f55fd499538f6594ba3538bd6f1659770bd4d717d0354395` |
| A/B harness: supervisor | `f9a0f9cfc1109d7bdf63f44a9848009b9ecb3075764f5edc414772bbb5432cef` |

## Focused validation

Commands run using the existing checkout environment:

```powershell
.venv-31410/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_display_equivalence.py tests/test_performance_baseline.py
.venv-31410/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -s integration_tests/test_performance_isolation.py -k 'not six_independent'
.venv-31410/Scripts/python.exe -B -m pytest -q -p no:cacheprovider tests/test_reporting.py tests/test_robustness_perturbations.py tests/test_robustness_reporting.py tests/test_inference.py::test_decode_rgb_and_strip_metadata tests/test_inference.py::test_jpeg_normalization tests/test_batch.py::test_validation_and_ordering
```

- 137 unit/presentation/benchmark-contract tests passed in 4.56 s.
- 13 Windows integration tests passed in 28.71 s; one full six-worker test
  deselected because both final A/B arms had already exercised that schedule.
- 189 relevant existing regressions passed in 2.58 s.
- Total: 339 passing tests, plus the two successful six-worker measurement arms.
- Ruff lint and formatting checks passed for all seven affected Python files.
- Whitespace checks cover tracked changes and each new file.

The affected regression areas are shared display pixels, single-inspection
presentation, batch compaction/previews, scalar reporting, synthetic numerical
contracts and worker IPC/isolation. The full historical suite, packaging,
fresh installation and CI remain for integrated 5D acceptance.

## Security/privacy and scientific review

The local 5D-2 inventory comprises five modified tracked files and three new
files: shared visualization, three performance modules, the integration test,
the new fixed profiler, independent equivalence tests and this document.
All are text source/documentation. The final review covers complete diffs and
untracked contents, fixed imports, scalar schema, process ownership, trace
restoration and source reads. No material security/privacy issue was identified.
No credential, personal email, machine-specific path, private report, real
image/weight artifact or unexpected executable is included.

The profiler reads only known installed source for attribution; results never
include source contents or paths. It runs in the same fixed worker without a
new arbitrary launch interface. The ordinary benchmark path has no tracing.
No dependency, lockfile, licence, OS security policy or 5C IPC contract changed.
MIT and existing third-party notices are preserved. Application-level guards
and Job Objects still rely on trusted installed Python/native code.

All nine scientific resources match their accepted bytes and SHA-256 pins.
Model architectures, scores, preprocessing, thresholds, registry, historical
evidence and robustness protocols remain unchanged.

PCB1 remains BLOCKED; PCB2 INCONCLUSIVE; Cashew official test already observed;
MetalParts exploratory-only; RobustAD independent target NOT CLEARED; Human
Review is not verified ground truth; the real Cashew development pilot BLOCKED.

The private development repository was not read or modified. Existing VS Code
windows and sessions were not disturbed. There is only the authorized local
5D-1 checkpoint; 5D-2 changes remain unstaged/uncommitted. No push, PR, merge,
release, ONNX/GPU installation or 5D-3 work occurred.

## Recommendation

**NEEDS DECISION:** the candidate provides exact checked display output and
a clear reduction in temporary storage/observed 512-pixel commit, but latency
improvement remains inconclusive under the owner's variability rule.
Review whether to retain this allocation reduction on that basis; do not
present the observed medians as a validated speedup.

Remaining hypotheses are preview resizing/PNG work and the per-channel
interpolation path. Neither is changed further here. Any later investigation
must preserve pixel/PNG semantics and the isolation budget. No real-model
performance or industrial readiness conclusion follows from these measurements.
