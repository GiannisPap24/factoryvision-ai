# Synthetic Windows isolation and execution ledger

Milestone 5C-4B2A implements a local, synthetic-only resource-control harness.
It does not authorize sample access, model loading, inference or the Cashew pilot.
No interface in this module accepts an admission override or real-model adapter.
The development admission gate remains unconditionally denied.

## Checkpoint and scope

The starting branch was `feat/robustness-5c-v1`, HEAD
`b471627a06b43e7289346aab8cbd9d1bb7b56164`, with exactly one modified and five
untracked 5C-4B1 files, nothing staged. Origin was the owner's public
`GiannisPap24/factoryvision-ai` repository. No other repository was accessed.

The reviewed six-file 5C-4B1 checkpoint is
`4a8e897da5699be019e6597f34a195d29c501c48`, parent
`b471627a06b43e7289346aab8cbd9d1bb7b56164`. Author and committer used the configured
GitHub noreply identity. Its exact inventory is:

- `docs/roadmap.md`
- `docs/protocols/development-admission-v1.md`
- `src/factoryvision/robustness/development_admission.py`
- `src/factoryvision/robustness/development_boundary.py`
- `tests/test_development_admission.py`
- `tests/test_development_boundary.py`

All staged bytes matched the directly inspected files. The accepted unchanged
459-test evidence (including 108 new foundation tests) was reused for checkpoint
review. Scientific resources matched all nine pins and the accepted main baseline.
The checkpoint left a clean tree. It was the only commit in this phase.

The subsequent, uncommitted 5C-4B2A inventory is exactly:

- `docs/roadmap.md` (modified)
- `docs/protocols/windows-isolation-v1.md` (new)
- `src/factoryvision/robustness/windows_isolation.py` (new)
- `src/factoryvision/robustness/execution_ledger.py` (new)
- `tests/test_windows_isolation.py` (new)
- `tests/test_robustness_execution_ledger.py` (new)

## Mechanism and ownership

`SyntheticSupervisor` serializes calls and reserves one of five exact synthetic
case identifiers before any process creation. Each reservation gets a fresh,
unnamed, noninheritable Job Object and at most one active worker. This is not a
persistent five-inference model worker; that integration is outside this phase.

The supervisor sets job and process committed-memory caps, active-process limit
one, and kill-on-job-close. It reads the limits back and verifies the job is empty.
It creates the child with `CREATE_SUSPENDED | CREATE_NO_WINDOW |
CREATE_UNICODE_ENVIRONMENT`, with handle inheritance disabled and no breakaway.
It records whether both supervisor and suspended child were already in jobs.
It assigns only its newly created child, verifies membership in the exact new job
and active count one, then charges dispatch before `ResumeThread`. Resume must
return an original suspend count of exactly one. No code runs before this gate.

Execution is limited to the source literal in this module and an exact Probe enum.
The interpreter is the running Python installation's base executable with
`-I -S -B`; callers cannot supply executable paths, scripts, shell commands,
callbacks, image paths or worker counts. The Windows virtual-environment redirector
initially returned exit 101 under the one-process job. Direct base-interpreter
launch resolved that startup issue without relaxing the one-process limit.

Only SYSTEMROOT is carried into the child environment, to support Windows runtime
loading; user environment secrets are not inherited or exported. The working
directory is the interpreter base directory. Standard-library/runtime loading is
necessary, but the fixed worker performs no dataset discovery, application imports,
image decoding, neural forward, network call or output-file write.

The only IPC is an allowlisted integer exit status from the fixed synthetic probe.
There are no pipes carrying arbitrary strings, pickle, shared queues or locks.
No raw child output or exception text is captured or exported. Unexpected codes
become a fixed invalid-output reason; unexpected supervisor errors are sanitized.

Only newly owned handles can be terminated or closed. No OpenProcess/OpenJobObject,
process-name termination, inherited-job termination, privilege escalation, policy
changes or assignment of the supervisor/editor/terminal exists. If assignment fails,
only the newly created suspended child is terminated through its returned handle.

## Actual Windows evidence

Tests ran on 64-bit Windows build 26300 with the existing Python 3.14.10 runtime.
A direct successful probe observed `parent_in_job=True`, `child_inherited_job=True`,
assignment and exact membership verified, then successful resume and completion.
Thus nested assignment was demonstrated in this command-runner context, not inferred
from API availability. No nested-job compatibility blocker was observed here.
Other launch contexts, including an application debugger, require their own gate.

The suspended-order test deliberately waits 75 ms before assignment: the success
worker cannot finish. It checks create/assign/verify/resume ordering and actual
ResumeThread suspend count. All three owned handles are closed exactly once.
Fault-injected creation, limit-setting, process-creation, assignment and membership
failures never resume a child. These are controlled faults, not claims that the
environment actually rejected normal assignment.

| Resource/control | Evidence | Scope/limitation |
|---|---|---|
| Job committed memory | 80 MiB VirtualAlloc denied at 64 MiB job / 256 MiB process cap | Actual OS rejection; small synthetic allocation only |
| Process committed memory | Same allocation denied at 64 MiB process / 256 MiB job cap | Separate process-limit test |
| Positive allocation control | Same allocation succeeds at 256 MiB / 256 MiB | Freed immediately after a short observation window |
| Proposed 2 GiB caps | Set and queried exactly; tiny success worker completes | No 2 GiB allocation or real-model capacity validation |
| Working set | Sampled via K32GetProcessMemoryInfo | Observed resident memory, not an enforced RSS ceiling |
| Wall-clock deadline | Noncooperative sleeping worker terminated at short deadlines | Supervisor monotonic watchdog; no CPU stress loop |
| CPU-time limit | None set | Not equivalent to elapsed deadline |
| Four CPU threads | Not implemented/tested | Proposed future model setting only |
| One active worker | Limit set/read back; sequential supervisor | No real persistent-worker integration |

The job-denial experiment returned allocation failure while Windows peak counters
could exceed the configured small cap. These raw counters are retained separately;
they are not interpreted as sustained successful allocation or a resident-memory
bound. Enforcement evidence is the rejected VirtualAlloc paired with the positive
control, not a claim that every peak counter must be below the configured ceiling.

The watchdog starts before job/worker setup and checks again before dispatch.
Synthetic deadlines are constrained to 0.05-10 seconds. Polling waits are 10 ms;
the process is terminated through its dedicated job, followed by confirmed exit
and an active-process-count zero query. Windows scheduling and synchronous API
calls prevent an exact real-time exit guarantee. Cleanup waits are separately
bounded to two seconds each, at most six seconds of waiting; elapsed result time
includes cleanup. The proposed 300-second pilot deadline remains unvalidated.

Success, controlled exception, malformed status, unexpected exit, timeout and
cancellation all confirm child exit and cleanup in normal tests. Injecting a
TerminateJobObject failure exercises kill-on-close fallback: exit is confirmed,
but the unobserved final job-empty state is explicitly `cleanup_unverified` and
outcome UNKNOWN. A simulated lost CloseHandle acknowledgement also stays UNKNOWN,
while remaining handles are closed. Either condition permanently blocks further
calls on that supervisor. The tests leave no owned child running.

Kill-on-close was exercised; abrupt death of a separate supervisor process and
crash recovery of its ledger were not tested. No existing host process was killed.
The Job Object is resource/lifecycle containment, not a filesystem/network sandbox.

## Accounting and admission

The in-memory ledger is supervisor-owned, with exact enums, strict integer budgets
1-5, frozen snapshots and validated transitions. No worker supplies counts.
Reservation consumes capacity permanently even if setup fails. Dispatch is charged
before resume. A failed/unknown resume acknowledgement therefore remains charged;
success, exception, timeout, cancellation and unknown outcomes never refund it.
An assignment failure is NOT_STARTED, charged zero, reservation retained.
Duplicate case identifiers, concurrent reservations, invalid transitions and a
sixth reservation are rejected before another process can be created. There is no
automatic retry, reset, external deserialization or real-execution token.

This ledger does not survive supervisor death and does not coordinate independent
supervisor instances. It is not a durable audit journal or a hostile-code security
boundary. Future app integration needs its own ownership/crash-recovery policy.

Sample/artifact states remain NOT_VERIFIED; execution remains NOT_AUTHORIZED.
Cashew validation exposure remains explicit. Eligibility is not authorization.
All existing six registrations, historical evidence, inference services, thresholds,
Human Review, V1/V2 protocols and nine scientific resources are unchanged.
PCB1 remains BLOCKED, PCB2 INCONCLUSIVE, Cashew official test already observed,
MetalParts exploratory source-domain only, RobustAD target NOT CLEARED.

## Verification and acceptance

The exact focused/regression command, run from the authorized repository:

```powershell
.venv-31410/Scripts/python.exe -m pytest -q tests/test_windows_isolation.py tests/test_robustness_execution_ledger.py tests/test_development_admission.py tests/test_development_boundary.py tests/test_robustness_contracts.py tests/test_robustness_security.py tests/test_robustness_perturbations.py tests/test_robustness_assessment.py tests/test_robustness_runner.py tests/test_robustness_reporting.py tests/test_model_registry.py tests/test_registry_checkout.py tests/test_distribution_privacy.py --tb=short
.venv-31410/Scripts/python.exe -m pytest -q tests/test_windows_isolation.py tests/test_robustness_execution_ledger.py --tb=short
.venv-31410/Scripts/python.exe -m ruff check src/factoryvision/robustness tests/test_windows_isolation.py tests/test_robustness_execution_ledger.py
.venv-31410/Scripts/python.exe -m ruff format --check src/factoryvision/robustness tests/test_windows_isolation.py tests/test_robustness_execution_ledger.py
git diff --check
```

Final results: 515 passed, zero failed, zero skipped: 56 new isolation/ledger tests
plus 459 existing focused regressions (6.08 seconds). After a lint-only correction
to a test context manager, all 56 new tests passed again (1.36 seconds), with zero
failures/skips. Ruff lint/format (12 files) and diff checks pass.
Windows-only cases have explicit skips elsewhere; skips are never enforcement proof.
No full historical suite, fresh distribution build/install or hosted CI is claimed.
Those remain required at the separately approved integrated acceptance gate.

The entire six-file diff, including untracked contents, was inspected. Credential,
private-key, personal-email and machine-path scans found no introduced information.
No datasets, images, checkpoints, weights or research artifacts were accessed or
added. All nine resource hashes and bytes still match accepted main. MIT licensing,
third-party attribution and dependencies are untouched. No publication or external
write occurred. The private repository and existing VS Code sessions were untouched.

Recommendation: GO for owner review of this synthetic isolation foundation.
NEEDS DECISION for a separately scoped integration phase. Actual development
execution and the Cashew pilot remain BLOCKED. The uncommitted implementation
does not expand admission, certify model operating limits or establish OOD or
industrial reliability. No push, PR, merge, release or next phase was performed.

## Official API references

- [Job objects and kill-on-close](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
- [Assignment and nested jobs](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject)
- [Committed-memory limits](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_extended_limit_information)
- [Suspended process creation flags](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags)
