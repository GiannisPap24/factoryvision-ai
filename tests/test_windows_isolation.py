"""Actual Windows synthetic jobs; fault injection never targets an existing process/job."""

import ast
import ctypes as c
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from factoryvision.robustness import windows_isolation as wi
from factoryvision.robustness.execution_ledger import Case, LedgerInvalid, Outcome

windows = pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows Job Objects required")


def clean(result):
    assert result.exit_confirmed and result.job_empty_confirmed and result.handles_closed
    assert (
        result.kind == "synthetic_windows_isolation_probe" and result.model_execution == "not_run"
    )


@windows
@pytest.mark.parametrize(
    "probe,outcome,reason",
    [
        (wi.Probe.SUCCESS, Outcome.SUCCEEDED, "completed"),
        (wi.Probe.EXCEPTION, Outcome.FAILED, "worker_exception"),
        (wi.Probe.INVALID, Outcome.FAILED, "invalid_output"),
        (wi.Probe.EXIT, Outcome.FAILED, "unexpected_exit"),
        (wi.Probe.HANG, Outcome.TIMED_OUT, "wall_clock_deadline"),
    ],
)
def test_actual_worker_lifecycle(probe, outcome, reason):
    result = wi.SyntheticSupervisor().run(
        Case.FIRST,
        probe,
        limits=wi.Limits(deadline_seconds=0.25 if probe is wi.Probe.HANG else 2.0),
    )
    clean(result)
    assert (result.outcome, result.reason) == (outcome, reason)
    assert (
        result.assigned and result.membership_verified and result.resumed and result.limits_verified
    )
    assert result.reserved == result.charged == 1
    assert result.elapsed_seconds < 3.0
    if probe is wi.Probe.HANG:
        assert result.elapsed_seconds >= 0.25
        assert result.observed_working_set > 0


@windows
def test_nested_assignment_suspended_order_and_all_owned_handles_closed(monkeypatch):
    events, handles, closed = [], [], []
    create_job, create_worker = wi._Win32.create_job, wi._Win32.create_worker
    assign, member, resume, close = (
        wi._Win32.assign,
        wi._Win32.member,
        wi._Win32.resume,
        wi._Win32.close,
    )

    def job(self):
        handle = create_job(self)
        handles.append(handle)
        return handle

    def worker(self, probe):
        result = create_worker(self, probe)
        handles.extend([result.process, result.thread])
        # A success worker cannot complete before assignment, even with a deliberate wait.
        assert not self.exited(result.process, 75)
        events.append("suspended")
        return result

    def assignment(self, job, process):
        assert events == ["suspended"]
        assign(self, job, process)
        events.append("assigned")

    def membership(self, process, job):
        flag = member(self, process, job)
        if job:
            assert flag and events == ["suspended", "assigned"]
            events.append("verified")
        return flag

    def resumed(self, thread):
        assert events == ["suspended", "assigned", "verified"]
        resume(self, thread)  # Checks the actual suspend count returned by ResumeThread is one.
        events.append("resumed")

    def closure(self, handle):
        assert handle in handles and handle not in closed
        closed.append(handle)
        return close(self, handle)

    for name, method in [
        ("create_job", job),
        ("create_worker", worker),
        ("assign", assignment),
        ("member", membership),
        ("resume", resumed),
        ("close", closure),
    ]:
        monkeypatch.setattr(wi._Win32, name, method)
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.SUCCESS)
    clean(result)
    assert result.outcome is Outcome.SUCCEEDED
    assert len(closed) == 3 and set(closed) == set(handles)
    assert events == ["suspended", "assigned", "verified", "resumed"]
    # Parent association is observed, never changed; nested support is proved if true.
    api = wi._Win32()
    assert result.parent_in_job == api.member(api.k.GetCurrentProcess(), None)
    # The returned child observation distinguishes actual nesting from a parent's
    # possible silent-breakaway policy; a parent-in-job flag alone is insufficient.
    assert type(result.child_inherited_job) is bool


@windows
@pytest.mark.parametrize(
    "limits,reason",
    [
        (wi.Limits(job_mib=64), "allocation_denied"),
        (wi.Limits(process_mib=64), "allocation_denied"),
        (wi.Limits(), "allocation_succeeded"),
    ],
)
def test_actual_memory_denial_and_same_allocation_control(limits, reason):
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.MEMORY, limits=limits)
    clean(result)
    assert result.outcome is Outcome.SUCCEEDED and result.reason == reason
    assert result.limits_verified and result.peak_job_commit > 0
    # Peak counters may include rejected reservations; do NOT assert peak <= configured cap.


@windows
def test_proposed_2gib_setting_query_only_no_large_allocation():
    result = wi.SyntheticSupervisor().run(
        Case.FIRST, wi.Probe.SUCCESS, limits=wi.Limits(job_mib=2048, process_mib=2048)
    )
    clean(result)
    assert result.outcome is Outcome.SUCCEEDED and result.limits_verified
    assert result.peak_job_commit < 64 * wi.MIB


@windows
@pytest.mark.parametrize("before_dispatch", [True, False])
def test_cancellation_and_no_refunds(monkeypatch, before_dispatch):
    signal = threading.Event()
    if before_dispatch:
        signal.set()
    else:
        resume = wi._Win32.resume

        def cancel_after_resume(self, thread):
            resume(self, thread)
            signal.set()

        monkeypatch.setattr(wi._Win32, "resume", cancel_after_resume)
    supervisor = wi.SyntheticSupervisor(1)
    result = supervisor.run(Case.FIRST, wi.Probe.HANG, cancel=signal)
    clean(result)
    assert result.outcome is Outcome.CANCELLED
    assert result.reserved == 1 and result.charged == int(not before_dispatch)
    with pytest.raises(LedgerInvalid):
        supervisor.run(Case.SECOND, wi.Probe.SUCCESS)


@windows
def test_actual_five_attempt_boundary_no_sixth_process(monkeypatch):
    supervisor = wi.SyntheticSupervisor()
    modes = [wi.Probe.SUCCESS, wi.Probe.EXCEPTION, wi.Probe.HANG, wi.Probe.INVALID, wi.Probe.EXIT]
    for ordinal, (case, mode) in enumerate(zip(Case, modes, strict=True), 1):
        result = supervisor.run(case, mode, limits=wi.Limits(deadline_seconds=0.2))
        clean(result)
        assert result.reserved == result.charged == ordinal
    created = []
    monkeypatch.setattr(wi._Win32, "create_job", lambda *args: created.append(1))
    with pytest.raises(LedgerInvalid):
        supervisor.run(Case.FIRST, wi.Probe.SUCCESS)
    assert not created and len(supervisor.entries) == 5


def test_busy_supervisor_rejects_concurrent_reservations():
    supervisor = wi.SyntheticSupervisor()
    with supervisor._lock, pytest.raises(ValueError, match="already running"):
        supervisor.run(Case.FIRST, wi.Probe.SUCCESS)
    assert supervisor.entries == ()


@windows
@pytest.mark.parametrize(
    "method,stage,charged",
    [
        ("create_job", "create_job", 0),
        ("configure", "set_limits", 0),
        ("create_worker", "create_worker", 0),
        ("assign", "assign_job", 0),
        ("resume", "resume", 1),
    ],
)
def test_fail_closed_api_faults_no_uncontained_execution(monkeypatch, method, stage, charged):
    resumed = []
    if method != "resume":
        monkeypatch.setattr(wi._Win32, "resume", lambda *args: resumed.append(True))

    def fail(*args):
        raise wi._Failure(stage, 5)

    monkeypatch.setattr(wi._Win32, method, fail)
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.SUCCESS)
    clean(result)
    assert result.reason == stage + ":access_denied"
    assert result.charged == charged and result.reserved == 1
    assert result.outcome is (Outcome.UNKNOWN if charged else Outcome.NOT_STARTED)
    assert not result.resumed and not resumed


@windows
def test_membership_failure_terminates_only_owned_assigned_worker(monkeypatch):
    member = wi._Win32.member
    monkeypatch.setattr(wi._Win32, "member", lambda self, p, j: False if j else member(self, p, j))
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.SUCCESS)
    clean(result)
    assert result.assigned and not result.resumed and result.charged == 0
    assert result.reason == "assignment_verification:failed"


@windows
def test_limit_readback_mismatch_prevents_worker_creation(monkeypatch):
    query = wi._Win32.query
    calls = []

    def mismatch(self, job, cls, struct):
        value = query(self, job, cls, struct)
        if cls == 9:
            value.job_memory = 0
        return value

    monkeypatch.setattr(wi._Win32, "query", mismatch)
    monkeypatch.setattr(wi._Win32, "create_worker", lambda *args: calls.append(1))
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.SUCCESS)
    clean(result)
    assert result.reason == "limit_verification:failed" and not calls and result.charged == 0


@windows
def test_unexpected_supervisor_exception_sanitized_with_reservation_retained(monkeypatch):
    def fail(*args):
        raise RuntimeError("fabricated-sensitive-environment")

    monkeypatch.setattr(wi._Win32, "resume", fail)
    supervisor = wi.SyntheticSupervisor()
    result = supervisor.run(Case.FIRST, wi.Probe.SUCCESS)
    clean(result)
    assert result.outcome is Outcome.UNKNOWN and result.charged == 1
    assert result.reason == "supervisor_failure"
    assert "sensitive" not in repr(asdict(result))
    assert supervisor.entries[0].outcome is Outcome.UNKNOWN


@windows
def test_deadline_includes_suspended_setup(monkeypatch):
    assign = wi._Win32.assign

    def delayed(self, job, process):
        assign(self, job, process)
        time.sleep(0.07)

    monkeypatch.setattr(wi._Win32, "assign", delayed)
    result = wi.SyntheticSupervisor().run(
        Case.FIRST, wi.Probe.SUCCESS, limits=wi.Limits(deadline_seconds=0.05)
    )
    clean(result)
    assert result.outcome is Outcome.TIMED_OUT and result.charged == 0 and not result.resumed


@windows
def test_kill_on_close_fallback_confirms_exit_but_does_not_hide_cleanup_fault(monkeypatch):
    def fail(*args):
        raise wi._Failure("terminate_job", 5)

    monkeypatch.setattr(wi._Win32, "terminate", fail)
    supervisor = wi.SyntheticSupervisor()
    result = supervisor.run(Case.FIRST, wi.Probe.HANG, limits=wi.Limits(deadline_seconds=0.1))
    assert result.exit_confirmed and result.handles_closed
    assert not result.job_empty_confirmed  # No query evidence after closing the job handle.
    assert result.outcome is Outcome.UNKNOWN and result.reason == "cleanup_unverified"
    assert result.reserved == result.charged == 1
    with pytest.raises(ValueError, match="execution blocked"):
        supervisor.run(Case.SECOND, wi.Probe.SUCCESS)
    assert len(supervisor.entries) == 1


@windows
def test_close_error_is_sanitized_and_other_owned_handles_still_closed(monkeypatch):
    close = wi._Win32.close
    closed = []

    def lost_acknowledgement(self, handle):
        assert close(self, handle)
        closed.append(handle)
        if len(closed) == 1:
            raise RuntimeError("fabricated-sensitive-close-error")
        return True

    monkeypatch.setattr(wi._Win32, "close", lost_acknowledgement)
    supervisor = wi.SyntheticSupervisor()
    result = supervisor.run(Case.FIRST, wi.Probe.SUCCESS)
    assert result.exit_confirmed and result.job_empty_confirmed
    assert not result.handles_closed and len(closed) == 3
    assert result.reason == "cleanup_unverified" and result.outcome is Outcome.UNKNOWN
    assert "sensitive" not in repr(result)
    with pytest.raises(ValueError, match="execution blocked"):
        supervisor.run(Case.SECOND, wi.Probe.SUCCESS)


@windows
def test_structures_match_windows_64bit_abi():
    if c.sizeof(c.c_void_p) != 8:
        pytest.skip("64-bit ABI check only")
    assert c.sizeof(wi._BasicLimits) == 64
    assert c.sizeof(wi._Extended) == 144
    assert c.sizeof(wi._Startup) == 104
    assert c.sizeof(wi._Process) == 24
    assert c.sizeof(wi._Memory) == 72


@pytest.mark.parametrize(
    "change",
    [
        {"job_mib": 63},
        {"job_mib": 2049},
        {"process_mib": True},
        {"job_mib": 64.0},
        {"deadline_seconds": 0.0},
        {"deadline_seconds": float("nan")},
        {"deadline_seconds": float("inf")},
        {"deadline_seconds": 300.0},
        {"deadline_seconds": 1},
    ],
)
def test_limits_reject_invalid_or_real_pilot_requests(change):
    with pytest.raises(ValueError):
        wi.Limits(**change)


def test_no_external_command_callback_path_or_worker_counts():
    supervisor = wi.SyntheticSupervisor()
    for kwargs in ({"command": "anything"}, {"path": "anything"}, {"worker": lambda: 1}):
        with pytest.raises(TypeError):
            supervisor.run(Case.FIRST, wi.Probe.SUCCESS, **kwargs)
    for probe in ("success", lambda: 1, None):
        with pytest.raises(ValueError):
            supervisor.run(Case.FIRST, probe)
    with pytest.raises(ValueError):
        supervisor.run(Case.FIRST, wi.Probe.SUCCESS, cancel=lambda: False)
    assert supervisor.entries == ()


def test_fixed_worker_has_no_network_filesystem_or_model_imports():
    tree = ast.parse(wi._WORKER)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"ctypes", "os", "sys", "time"}
    calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert calls == {
        "main",
        "RuntimeError",
        "os._exit",
        "time.sleep",
        "c.WinDLL",
        "k.VirtualAlloc",
        "k.VirtualFree",
        "c.get_last_error",
        "sys.exit",
    }
    source = Path(wi.__file__).read_text(encoding="utf-8")
    assert "OpenProcess" not in source and "OpenJobObject" not in source
    assert wi._CREATE_FLAGS == 0x08000404  # No breakaway or inherited handles.
    assert "factoryvision.inference" not in source


def test_unsupported_platform_is_not_isolation_success(monkeypatch):
    monkeypatch.setattr(wi.sys, "platform", "unsupported")
    result = wi.SyntheticSupervisor().run(Case.FIRST, wi.Probe.SUCCESS)
    assert result.outcome is Outcome.NOT_STARTED and result.reason == "platform:failed"
    assert not result.assigned and not result.resumed and result.charged == 0
