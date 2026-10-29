# Synthetic Cashew isolated IPC: integration record

Milestone 5C-4C2A is synthetic-only. No real sample/artifact is cleared and actual
model execution remains denied. The private repository is not an asset source.

## Minimal design, before implementation

Reuse the existing Windows backend for job creation, committed-memory controls,
assignment, membership checks, resume, observation and termination. Add only a
fixed IPC launch path and a single-use five-request session. The existing probe
runner and its fixed stdlib worker retain their behavior.

Use two anonymous pipes with STARTUPINFOEX and an explicit two-handle inheritance
list: child input-read and output-write (also stderr). Parent endpoints and the
dedicated Job Object remain non-inheritable. No unrestricted inheritance fallback.
See [Microsoft handle-list rules](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute)
and [pipe endpoint ownership](https://learn.microsoft.com/en-us/windows/win32/procthread/creating-a-child-process-with-redirected-input-and-output).

One fixed application module uses existing analytical-double mathematics and V2
diagnostics. A small extraction of the double's numerical output avoids importing
the neural inference service in the child. Equivalence needs direct tests.

Frames use unsigned 32-bit network-order lengths with separate 4096-byte JSON and
4096-byte PNG limits. Only the internal 10x8 RGB8 synthetic parent and its fixed
five conditions are accepted. The worker independently checks decoded pixels
against that generated sequence. Maps remain in the worker; responses are scalars.
No externally supplied image/path, executable/module, threshold or callback enters
the public session API. No automatic writes, retries or real execution grants.

The parent owns reservations and dispatch accounting. A bounded I/O helper thread
keeps synchronous pipe calls out of the watchdog loop; termination of the owned
child and verified helper completion precede pipe closure. Unknown cleanup is
fail-closed, never success. A terminated session is not restarted.

## Repository and checkpoint

The authorized public-staging workspace, origin
`https://github.com/GiannisPap24/factoryvision-ai.git`, branch
`feat/robustness-5c-v1`, initial HEAD `c35b073861cadaa7552e7ef6e7b74e9ca7d3a054`
and exact four-file inventory matched the owner's request. Nothing was staged.
No additional repository AGENTS.md was found; supplied restrictions were retained.

All four 5C-4C1 files and their complete staged diff were reviewed directly. The
previous 409-test acceptance and final 52-test rerun were reused for the unchanged
checkpoint. Author/committer were verified GitHub noreply identities. Privacy
patterns and the nine scientific resources were checked again before committing.

The only new commit is the local 5C-4C1 checkpoint:
`2a1b959a858a2689ede995bb9d3b643f4444668f`.
Parent: `c35b073861cadaa7552e7ef6e7b74e9ca7d3a054`. Its exact inventory:

- `docs/roadmap.md`
- `docs/protocols/cashew-development-preparation-v1.md`
- `src/factoryvision/robustness/cashew_development.py`
- `tests/test_cashew_development.py`

The tree was clean immediately after that checkpoint. Current HEAD remains that
commit. Current 5C-4C2A inventory is exactly eight uncommitted files, nothing staged:

| Status | File | Purpose |
|---|---|---|
| Modified | `docs/roadmap.md` | Record local synthetic integration status |
| Modified | `src/factoryvision/robustness/development_boundary.py` | Extract the existing analytical output, retain InspectionResult wrapper behavior |
| Modified | `src/factoryvision/robustness/windows_isolation.py` | Share the accepted owned-job cleanup algorithm |
| New | `docs/protocols/cashew-isolated-ipc-v1.md` | Design, protocol, acceptance and audit record |
| New | `src/factoryvision/robustness/cashew_isolated_ipc.py` | Fixed allowlisted launch, watchdog, accounting and scalar report |
| New | `src/factoryvision/robustness/synthetic_ipc.py` | Narrow frame/schema, synthetic input and numerical contracts |
| New | `src/factoryvision/robustness/synthetic_ipc_worker.py` | Fixed analytical child entry and finite test faults |
| New | `tests/test_cashew_isolated_ipc.py` | 74 focused tests, including actual Windows IPC |

## Worker and reuse

`IsolatedCashewProbe.run()` accepts only existing Limits and an optional cancellation
Event. It is single-use, including after failure; no caller-supplied pixels, filename,
root, service, callback, command, module or execution-authorization flag is accepted.

The fixed child uses the current base Python interpreter with `-I -S -B`, no shell,
no venv redirector, no site/.pth execution, and only the installed project/dependency
roots derived from already loaded code. The environment contains SYSTEMROOT and two
fixed single-thread numerical-library settings; host secrets are not inherited.
Pillow codec registration occurs before the worker's post-import I/O guard. It is
import initialization, not image/model evaluation or an uncharged warmup.

The child imports neither PyTorch nor the inference service. A post-import audit
guard denies file opening/discovery, socket operations, subprocess execution and
neural-service imports during the exercise. Only source/runtime/dependency imports
occur before this guard. There is no model loader, artifact verification, dataset
discovery or service invocation in the worker.

`SyntheticInspectionDouble.evaluate_output()` contains the unchanged RGB-mean and
constant 128x128 raw-map calculation. Its existing `evaluate()` wrapper still returns
InspectionResult, preserving scores, strict threshold comparison and mutable result
map behavior. Direct equivalence and existing interface tests pass. No real inference
service or preprocessing implementation changed. Existing V2 perturbation equations,
assessment functions, synthetic runner and report schemas remain unchanged.

## Exact IPC contract

Both directions use `uint32` big-endian byte length followed by exactly that many
bytes. JSON payloads must be 1..4096 UTF-8 bytes. An inspect request consists of a
JSON frame followed by an independently length-prefixed 1..4096-byte PNG frame.
These are separate from the historical 2 MiB scalar-report cap. The new optional
in-memory scalar report has a tighter 16 KiB cap and writes nothing automatically.

Every JSON message carries exactly `protocol_version: 1`, `schema_version: 1`,
`kind: synthetic_service_interface_probe`, `model_execution: not_run`, `type` and
integer `seq`, plus the specific allowed fields below:

| Message | Sequence | Additional fields |
|---|---|---|
| ready | 0 | status=ready |
| inspect | 1 through 5, strictly increasing | none; one following PNG frame |
| result | same pending request | status=success, input_verified, baseline_score, score, threshold, above_threshold, signed, absolute, relative, crossing, mae, rmse, maximum_absolute |
| close | 6 | none; this is shutdown, not an inference attempt |
| done | 6 | status=complete; must be followed by EOF and process exit code zero |
| error | current sequence | status=failed, code=invalid_request; never accepted as a successful result |

Unknown/missing fields, duplicate JSON keys, wrong types including boolean sequence
numbers, nested objects/arrays, nonfinite numbers, invalid versions/identity, oversized
or truncated frames, wrong/stale/duplicate sequences and trailing terminal bytes
are rejected. Length validation precedes body reads/allocations. Error objects cannot
override supervisor status; their contents are not exported as diagnostics.

The PNG gate validates signature, chunk lengths/CRC/order, exact 10x8 RGB8 IHDR,
and only IHDR/IDAT/IEND chunks before Pillow decoding. Bounded zlib expansion requires
exactly 248 filtered bytes, capped at 249 during validation. Metadata chunks, JPEG,
large advertised dimensions, extra streams and trailing bytes are rejected. Decoded
pixels must exactly equal the independently generated synthetic condition in the
child. This verifies synthetic transport, not real sample admission or provenance.

## Windows ownership and watchdog evidence

The existing backend still creates the unnamed non-inheritable dedicated job, sets
and queries process/job committed-memory limits, active-process limit one and
KILL_ON_JOB_CLOSE, assigns the suspended child, verifies membership and resumes it.
Its previous stdlib probe launch remains unchanged with handle inheritance disabled.
Only its cleanup body was moved into a shared backend method; the termination,
bounded waits, job-empty query, final job-close fallback and handle-close order remain.

The new path adds STARTUPINFOEX, EXTENDED_STARTUPINFO_PRESENT and
PROC_THREAD_ATTRIBUTE_HANDLE_LIST containing exactly the child input-read and
output-write handles. Stderr uses that same output handle; exceptions are sanitized.
The two parent endpoints and dedicated job are verified non-inheritable. Parent
copies of child endpoints close after creation. Attribute-list failure has no fallback.

Actual tests intercept the real Windows calls and verify the two-element allowlist,
fixed command, flags, pre-resume reservation/dispatch, suspended state, assignment and
membership order. A deliberately inheritable unrelated event does not enter the
child. Remote-handle duplication checks the suspended child's handle table: absence
requires ERROR_INVALID_HANDLE; reused numeric slots must refer to a different object,
using [Microsoft CompareObjectHandles](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-compareobjecthandles).
The job handle likewise is not inherited. Tests close only newly owned handles.

The current execution context supports actual nested assignment: parent and child
both had inherited-job association, and membership in the new dedicated job was
verified before resume. No parent-job settings, breakaway policy or Windows security
settings changed. This evidence is specific to Windows build 26300 / Python 3.14.10.

One helper thread performs each synchronous pipe exchange while the supervisor polls
its completion/cancellation/deadline at 10 ms intervals. The watchdog starts before
job setup. On failure, owned-child/job termination and the existing bounded cleanup
occur before the helper is joined (up to a separate two seconds) and pipes are closed.
The existing cleanup has at most three separate two-second waits. Cleanup grace is
not part of the configured execution deadline. Unverified exit/join/handle cleanup
cannot report success; the object stays unusable. No broad process lookup/termination
or unrelated application control is used.

## Five-attempt and failure evidence

Each successful session uses one persistent worker and exactly five analytical
evaluations: identity three times, brightness 0.90, brightness 1.10. Every transform
starts from the unchanged internally generated parent. Worker-local raw maps are
compared with the first baseline using the existing V2 functions on the 128x128 grid;
there is no display normalization or score reconstruction from an alternative model.

The synthetic subprocess diagnostics exactly match the prior in-process analytical
exercise and an independently initiated repeat session:

| Condition | Signed score difference | Raw-map MAE |
|---|---:|---:|
| Identity 1, 2, 3 | 0.0 | 0.0 |
| Brightness 0.90 | -0.050049155950546265 | 0.050049006938934326 |
| Brightness 1.10 | 0.042042434215545654 | 0.04204249382019043 |

These are software-test values, not Cashew neural-network predictions. Fabricated
threshold 0.5 and metadata do not authenticate a model or replace its frozen threshold.

The supervisor alone reserves and charges. First charge occurs before resume
(including startup/handshake uncertainty); subsequent charges precede request writes.
Success yields reserved=charged=succeeded=5. A sixth reservation and reuse of the
same session are rejected before another launch. Workers supply no authoritative
attempt count. Shutdown is not an evaluation. There are no retries or refunds.

Actual process tests cover hang, partial response, malformed JSON, oversized length,
unexpected exit, stale and duplicate responses, trailing terminal bytes, cancellation
before/during communication, rejected malformed requests and setup failures. Partial
acknowledgement/exit preserves UNKNOWN; timeout/cancellation preserve charged counts.
All tested real fault paths confirm worker exit, empty job, completed I/O helper and
owned-handle closure. An injected cleanup-acknowledgement uncertainty overrides global
success with UNKNOWN without erasing already received results or refunding charges.

Short fault-test deadlines are 0.6 seconds; default normal-session deadline is five
seconds, with 256 MiB process and job committed-memory ceilings. One early complete
probe observed about 26.1 MiB peak job commit, 25.0 MiB peak process commit and 34.0 MiB
working set in about 0.144 seconds. These counters are distinct; working set is
observational, not a cap. CPU execution time was not measured. No large resource
stress was run. Proposed real-model four-thread, 2 GiB and 300-second settings remain
unvalidated and are not certified by these synthetic tests.

The in-memory ledger does not survive supervisor death. Existing disposable-supervisor
crash tests were rerun because shared cleanup moved; they still verify kill-on-close
and abort-only recovery. No new claim of durable recovery or a real-model crash test
is made. A crash or uncertain execution requires STOP and separate owner review.

## Exact validation commands and results

The combined focused suite passed **498 tests, zero failed, zero skipped**, in 8.09
seconds: 74 new IPC tests, 409 prior directly related tests, and 15 crash tests.
After final scalar-report consistency checks, the 74 new tests passed again in 5.17
seconds, zero failed/skipped. Final lint, format and diff checks pass. The final narrow
handle-evidence test passed (1 test in 1.82 seconds), checking ERROR_INVALID_HANDLE
explicitly and the final startup wrapper that suppresses import-failure tracebacks.

```powershell
.venv-31410/Scripts/python.exe -m pytest -q tests/test_cashew_isolated_ipc.py tests/test_cashew_development.py tests/test_development_admission.py tests/test_development_boundary.py tests/test_robustness_perturbations.py tests/test_robustness_assessment.py tests/test_robustness_runner.py tests/test_robustness_reporting.py tests/test_windows_isolation.py tests/test_windows_crash_containment.py tests/test_robustness_execution_ledger.py --tb=short
.venv-31410/Scripts/python.exe -m pytest -q tests/test_cashew_isolated_ipc.py --tb=short
.venv-31410/Scripts/python.exe -m pytest -q tests/test_cashew_isolated_ipc.py::test_launch_order_actual_allowlist_and_noninherited_job --tb=short
.venv-31410/Scripts/python.exe -m ruff check src/factoryvision/robustness/synthetic_ipc.py src/factoryvision/robustness/synthetic_ipc_worker.py src/factoryvision/robustness/cashew_isolated_ipc.py src/factoryvision/robustness/windows_isolation.py src/factoryvision/robustness/development_boundary.py tests/test_cashew_isolated_ipc.py
.venv-31410/Scripts/python.exe -m ruff format --check src/factoryvision/robustness/synthetic_ipc.py src/factoryvision/robustness/synthetic_ipc_worker.py src/factoryvision/robustness/cashew_isolated_ipc.py src/factoryvision/robustness/windows_isolation.py src/factoryvision/robustness/development_boundary.py tests/test_cashew_isolated_ipc.py
git diff --check
```

The interface regression initially caught changed map mutability after extraction;
the original wrapper behavior was restored rather than weakening the test. Parent
test guards reject actual model construction/loading/forward and filesystem discovery.
Actual successful child runs also pass the post-import file/network denial guard.

This is focused local acceptance, not full historical-suite/coverage, hosted CI,
fresh packaging/install or distribution privacy acceptance. Those remain necessary
at the eventual integrated publication gate. No browser or VS Code interaction was
needed or performed in this phase.

## Security/privacy and scientific verdict

Targeted review used the security-best-practices skill and official Microsoft handle
documentation. That skill has no desktop Python/Win32-specific reference; the actual
API contracts and repository implementation were reviewed directly. No material
security/privacy issue remains identified within this bounded synthetic scope.

All new/modified contents were directly inspected. Final checks cover exact inventory,
no staged work, secrets/private-key/token/email/absolute-path patterns, no unexpected
binary/data/weight files, no unsafe deserializers or dynamic IPC execution, and all
nine pinned scientific-resource hashes and baseline bytes. Dependency metadata,
MIT licence/third-party attribution, real inference/training, Model Registry and
Human Review are unchanged. Frozen V1/V2 protocols, metrics and thresholds are unchanged.

No real dataset image, checkpoint, weight or private research artifact was read.
No private development repository access, GitHub write, automatic report persistence,
external upload or Windows security change occurred. Existing VS Code sessions
were undisturbed. Raw pixels/maps, private identifiers/hashes, paths, EXIF/ICC,
environment values and tracebacks never enter the output report.

PCB1 remains BLOCKED; PCB2 INCONCLUSIVE; Cashew official test already observed;
MetalParts exploratory-only; RobustAD target NOT CLEARED. Human Review remains
distinct from verified ground truth. Sample/artifact admission is NOT_VERIFIED,
execution NOT_AUTHORIZED, and require_execution() remains unchanged and always denies.

## Remaining blockers and recommendation

GO for owner review of this local synthetic-only IPC implementation. It is not a
general OS security sandbox: source/dependency code and the in-process supervisor
are trusted; the post-import guard is defense in depth, not hostile-code containment.
Creation and assignment are not atomic: a supervisor crash during the initial
pre-assignment interval is outside the dedicated job's protection; the child is still
suspended and cannot execute the worker. No claim of universal launch-context support
is made.

NEEDS DECISION for a separately scoped real worker, real-input numerical boundary
and evidence/admission work. This worker deliberately rejects all pixels outside
the exact synthetic fixture; there is no hidden real-model activation flag.

BLOCKED for the actual Cashew pilot. It still requires exact authorized read-only
sample/artifact locations, VisA source/split/validation membership and prior exposure,
sample-byte/decoded-pixel verification, checkpoint/calibration SHA-256 verification,
licensing/attribution, a reviewed real-service binding, resource validation and explicit
execution authorization. The private repository must never be an implicit source.

No 5C-4C2A implementation commit, push, PR, merge, release or real pilot was performed.
Stop after this report and await owner review.
