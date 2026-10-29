# Supervisor crash containment: synthetic acceptance record

Milestone 5C-4B2B tests unexpected supervisor death without changing production
isolation, ledger, admission or model execution code. All processes are disposable
synthetic fixtures. No real samples, images, checkpoints or weights were accessed.

## Repository and checkpoint

Verified branch: `feat/robustness-5c-v1`; origin:
`https://github.com/GiannisPap24/factoryvision-ai.git`. Starting HEAD:
`4a8e897da5699be019e6597f34a195d29c501c48`.
The initial tree contained precisely one modified and five untracked approved
5C-4B2A files, nothing staged. Their complete contents were inspected; source/test
hashes matched the previously accepted version. The 515 passing focused tests
were reused without rerunning the complete set. All nine scientific resources
matched their pins and the accepted main baseline byte-for-byte.

The only new commit is the local 5C-4B2A checkpoint:
`9f07e5e6334eeb99a2f1cc9c7b7f7b374f14c1e1`, parent
`4a8e897da5699be019e6597f34a195d29c501c48`. Author and committer were verified
GitHub noreply identities. The staged bytes matched the inspected files and the
post-commit tree was clean. Exact checkpoint inventory:

- `docs/roadmap.md`
- `docs/protocols/windows-isolation-v1.md`
- `src/factoryvision/robustness/execution_ledger.py`
- `src/factoryvision/robustness/windows_isolation.py`
- `tests/test_robustness_execution_ledger.py`
- `tests/test_windows_isolation.py`

5C-4B2B remains uncommitted. Its exact four-file inventory is:

- `docs/roadmap.md` (modified)
- `docs/protocols/windows-crash-containment-v1.md` (new)
- `tests/_windows_crash_fixture.py` (new)
- `tests/test_windows_crash_containment.py` (new)

## Controller, supervisor and handle ownership

The test controller launches only the fixed helper in a separate hidden, suspended
base interpreter with `-I -S -B`. No external executable, script, callback or dataset
path is accepted by the harness API. Only SYSTEMROOT is carried into its environment.
The helper imports the existing stdlib-only isolation and ledger modules from the
known source location; no filesystem discovery or data import occurs.

The disposable helper runs the existing `SyntheticSupervisor.run` implementation.
Test-local observers verify the real allocation/assignment/dispatch/resume sequence.
Only the worker body is replaced in that disposable interpreter: a fixed
`time.sleep(8.0)` fixture. No production API or stored source is modified by these
observers. The worker uses the unchanged suspended launcher with inheritance FALSE.

The supervisor alone owns the unnamed worker Job Object handle. Readback verifies
the existing job/process memory limits, one-active-process limit and kill-on-close
flag. GetHandleInformation verifies non-inheritance of the job, worker process and
worker thread handles. Both supervisor and controller are checked to be OUTSIDE
the dedicated worker job before the worker resumes; only the new worker is inside.

The controller-to-supervisor launch has an explicitly justified two-handle allowlist
using STARTUPINFOEX and PROC_THREAD_ATTRIBUTE_HANDLE_LIST:

1. An anonymous readiness-pipe write handle.
2. A controller process handle with PROCESS_QUERY_INFORMATION only, so the helper
   can verify controller exclusion from the worker job. It has no terminate,
   process-mutation or process-handle-duplication rights.

That launch requires inheritance TRUE, as documented by Microsoft. Only the two
listed handles are inherited. Their inherit flags are cleared in the disposable
supervisor before launching the worker. The controller's read pipe and all returned
process/thread handles are noninheritable. No inherited parent-job policy is changed.
The worker launch itself remains inheritance FALSE.

Readiness is a strict 40-byte message with magic, bounded numeric identity, eight
successful checks and the exact expected reservation/dispatch counts. It contains
no strings, paths, environment values, model metadata or arbitrary serialized objects.
Runtime process IDs/handle values remain ephemeral IPC; public test results omit them.

After validating readiness, the controller duplicates only the worker PROCESS
handle from its verified supervisor process handle. It validates the returned
process identity and uses the handle for wait/query/emergency cleanup. The controller
never obtains or duplicates the worker JOB handle, so it cannot accidentally keep
the job alive after supervisor death. No PID lookup, OpenProcess/OpenJobObject or
process-name termination is used. Tests verify both duplicated handles are process
handles and the exact access masks; all six controller-owned handles are closed.

## Actual crash evidence and bounds

Environment: Windows build 26300, 64-bit Python 3.14.10, current command-runner
context. Both the disposable supervisor and its worker inherited an existing job;
assignment of the worker into the dedicated nested job succeeded.

The controller confirms readiness, verifies the worker is still alive after a
75 ms wait, and calls TerminateProcess ONLY on its verified, newly created supervisor
handle. The supervisor exits with the controller's fixed exit code 93. This bypasses
the supervisor's Python finally/cleanup. The worker then exits through OS job cleanup.
The initial direct observation measured approximately 1.9 ms from the termination
request through confirmation of both exits. This is one observed latency, not an SLA.

The crash test limits each supervisor/worker exit wait to one second and requires
combined latency below two seconds. This is earlier than either the supervisor's
five-second watchdog or the worker's eight-second natural completion. Readiness
waiting is bounded to three seconds; emergency cleanup waits are at most two seconds
per owned process. Only the controller's own returned/duplicated handles can be used
for emergency termination. A failed verification stays BLOCKED even if fallback
cleanup subsequently succeeds; there is no automatic retry or replacement run.

The controller remained alive and its parent-job-membership observation was unchanged.
No unrelated handle is a termination target, and no inherited job handle is opened,
closed or modified. Normal crash verification confirms both owned processes exited.
A mismatched-readiness-identity test also confirms both exits during failure cleanup.
A simulated supervisor-creation failure closes all three pre-launch controller handles.
Malformed readiness errors expose only the fixed sanitized failure message.

No optional worker-created descendant was launched: the accepted one-active-process
limit is retained. The only downstream process is the verified worker, whose exit
is confirmed. This does not certify a multi-process model architecture. Isolation
limits were not weakened merely to exercise a deeper process tree.

## Ledger interruption and future recovery policy

The disposable supervisor verifies RESERVED/zero charged before worker creation and
DISPATCHED/one charged before ResumeThread. Readiness conveys that observed state
before the controller terminates the supervisor. After the crash, the test result
retains the last observed reservation and charge, marks execution outcome UNKNOWN,
and marks recovery BLOCKED. Process exit is never interpreted as zero work performed.

The in-memory ledger itself is destroyed with the supervisor. The controller's
observation is not restoration or durable persistence of that ledger. A focused
living-ledger test verifies that a dispatched attempt without acknowledgement blocks
another reservation; explicitly marking UNKNOWN preserves consumed capacity.
Existing regressions cover failures, timeouts, cancellation, exact five-attempt
capacity, sixth-attempt rejection before process creation and absence of retries.

Future real-model integration MUST use this fail-closed recovery policy:

- Unexpected supervisor death, uncertain execution history or unverified cleanup
  invalidates continuation of that run. Stop further dispatch and mark the run UNKNOWN.
- Do not restart/replay a condition, reset the budget, create a replacement worker
  or treat a fresh in-memory ledger as proof that no attempts occurred.
- Retain any independently observed reservations/charges; if the full history is
  unavailable, do not invent a lower count or assume remaining budget is available.
- Confirm owned-process cleanup, preserve available evidence, and require explicit
  owner review before a separately scoped new run can be considered.
- If the outer controller/app also crashes, startup cannot presume an interrupted
  run is resumable. Recovery/ownership must be designed and accepted before real use.

This policy is documented, not a new persistent tracking implementation. There is
no database, run-resume endpoint, automatic recovery path or real-execution token.
Durable recovery remains outside the approved scope. Existing real admission always
rejects execution, so no real run can currently bypass this future integration gate.

## Resource evidence and launch-context limits

| Property | Verified evidence | Remaining limit |
|---|---|---|
| Nested assignment | Actual inherited-job child assigned before resume | Different app/debugger launch context needs its own check |
| Worker containment | Membership verified, supervisor/controller excluded | Job Objects are not a filesystem/network sandbox |
| Supervisor crash | Actual TerminateProcess; worker exits without supervisor cleanup | No real model or arbitrary descendants tested |
| Job/process committed memory | Existing separate 64 MiB rejection and 256 MiB positive-control tests pass | Not an enforced RSS guarantee |
| Proposed 2 GiB | Existing exact OS set/readback test passes with tiny worker | No allocation stress or real-model capacity certification |
| Proposed 300 seconds | Existing short monotonic watchdog tests pass | 300-second real workload untested |
| Four CPU threads | Proposed only | Not enforced or validated by these fixtures |
| Observed consumption | Crash-exit latency measured; finite sleeping worker | No new memory/CPU profile from the crash case; prior synthetic counters remain separate |

Synthetic containment is not evidence of anomaly accuracy, OOD performance, dataset
independence or industrial reliability. Real operating limits need separately
authorized measurements in the intended launch context.

## Focused verification

The new crash tests first passed 15/15. After import-order lint correction, the
combined focused command below passed **128 tests, zero failed, zero skipped** in
1.58 seconds: 15 new tests, 56 existing isolation/ledger tests and 57 admission tests.
The previously accepted 515-test evidence was reused; it was not rerun wholesale.

```powershell
.venv-31410/Scripts/python.exe -m pytest -q tests/test_windows_crash_containment.py --tb=short
.venv-31410/Scripts/python.exe -m pytest -q tests/test_windows_crash_containment.py tests/test_windows_isolation.py tests/test_robustness_execution_ledger.py tests/test_development_admission.py --tb=short
.venv-31410/Scripts/python.exe -m ruff check tests/_windows_crash_fixture.py tests/test_windows_crash_containment.py
.venv-31410/Scripts/python.exe -m ruff format --check tests/_windows_crash_fixture.py tests/test_windows_crash_containment.py
git diff --check
```

Lint, formatting and diff checks pass. Actual Windows cases have explicit non-Windows
skips, which are not accepted as containment evidence. No full CI-equivalent run,
distribution rebuild, fresh install or hosted CI is claimed in this bounded phase;
these remain required at integrated acceptance/publication.

## Security, integrity and recommendation

Every changed/untracked file was inspected. No credentials, personal emails, private
keys, machine-specific absolute paths, private identifiers or data/model payloads
were introduced. The fixed worker imports only time; the disposable supervisor checks
that Torch, NumPy, PIL and FactoryVision inference modules were not loaded. The harness
performs no network calls and writes no output files. Only known source/runtime files
are imported; no ignored dataset directories are discovered or inspected.

All nine scientific resources remain byte-identical to their pins and accepted main.
Production source, frozen V1/V2 protocols, Model Registry, historical metrics,
thresholds, inference/training, Human Review, reporting semantics, MIT attribution
and dependencies are unchanged. Sample/artifact admission remains NOT_VERIFIED and
execution NOT_AUTHORIZED. PCB1 BLOCKED, PCB2 INCONCLUSIVE, Cashew official test already
observed, MetalParts exploratory-only and RobustAD target NOT CLEARED are preserved.

No Windows security settings/privileges were changed. The original private repository,
existing VS Code windows/sessions and debugger configurations were untouched. No
push, PR, merge, release or publication occurred. Current 5C-4B2B work is uncommitted.

GO for owner acceptance of the final synthetic isolation gate in this launch context.
NEEDS DECISION for a separately authorized integration scope, including provenance,
sample/artifact review, recovery ownership, context verification and real resource
measurements. Actual model execution and the Cashew pilot remain BLOCKED.

## Official references reviewed

- [Job Objects and kill-on-close](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
- [Explicit inherited-handle allowlist](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute)
- [DuplicateHandle ownership and rights](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-duplicatehandle)
- [GetHandleInformation inheritance flags](https://learn.microsoft.com/en-us/windows/win32/api/handleapi/nf-handleapi-gethandleinformation)
