"""Focused disposable-supervisor crash acceptance; production runner remains unchanged."""

import ast
import sys

import _windows_crash_fixture as fixture
import pytest

from factoryvision.robustness.execution_ledger import Case, ExecutionLedger, LedgerInvalid, Outcome

windows = pytest.mark.skipif(
    sys.platform != "win32", reason="Actual Windows crash containment required"
)


@windows
def test_actual_supervisor_death_kills_worker_without_controller_job_handle(monkeypatch):
    close = fixture.CrashAPI.close
    duplicate = fixture.CrashAPI.duplicate
    closed, accesses = [], []

    def closing(self, handle):
        value = close(self, handle)
        assert value
        closed.append(handle)
        return value

    def duplicated(self, source, handle, access, inherit=False):
        value = duplicate(self, source, handle, access, inherit)
        # Every duplicate is a PROCESS handle, never a job handle retaining ownership.
        assert self.k.GetProcessId(value) > 0
        accesses.append((access, inherit))
        return value

    monkeypatch.setattr(fixture.CrashAPI, "close", closing)
    monkeypatch.setattr(fixture.CrashAPI, "duplicate", duplicated)
    result = fixture.run_crash_probe()
    assert accesses == [(0x400, True), (0x100401, False)]
    assert len(closed) == 6  # Two pipe ends, controller query handle, process/thread, worker view.
    assert all(
        result[key]
        for key in (
            "supervisor_exited",
            "worker_exited",
            "controller_alive",
            "parent_membership_unchanged",
            "owned_handles_closed",
        )
    )
    assert result["crash_exit_seconds"] < 2.0
    assert result["last_observed_reserved"] == result["last_observed_charged"] == 1
    assert result["execution_outcome"] == "unknown" and result["recovery"] == "blocked"
    assert not any(key in result for key in ("pid", "handle", "path", "job", "command"))


@windows
def test_setup_failure_closes_all_controller_handles_and_sanitizes(monkeypatch):
    initialize, close = fixture.CrashAPI.__init__, fixture.CrashAPI.close
    closed = []

    def fail_create(self):
        initialize(self)
        self.k.CreateProcessW = lambda *args: 0

    def closing(self, handle):
        value = close(self, handle)
        closed.append(value)
        return value

    monkeypatch.setattr(fixture.CrashAPI, "__init__", fail_create)
    monkeypatch.setattr(fixture.CrashAPI, "close", closing)
    with pytest.raises(fixture.CrashBlocked) as caught:
        fixture.run_crash_probe()
    assert closed == [True] * 3
    assert (
        str(caught.value) == "Synthetic crash containment could not be verified; execution blocked."
    )


@windows
def test_bad_ready_identity_cleans_up_created_processes_without_retry(monkeypatch):
    decode = fixture.decode_ready
    original_init = fixture.CrashAPI.__init__
    exit_checks, starts = [], []
    exited = fixture.CrashAPI.exited

    def mismatched(raw):
        handle, pid, parent, inherited = decode(raw)
        return handle, pid + 1, parent, inherited

    def initialize(self):
        original_init(self)
        create = self.k.CreateProcessW

        def counted(*args):
            starts.append(1)
            return create(*args)

        self.k.CreateProcessW = counted

    def waited(self, handle, milliseconds=0):
        value = exited(self, handle, milliseconds)
        if milliseconds == 2000:
            exit_checks.append(value)
        return value

    monkeypatch.setattr(fixture, "decode_ready", mismatched)
    monkeypatch.setattr(fixture.CrashAPI, "__init__", initialize)
    monkeypatch.setattr(fixture.CrashAPI, "exited", waited)
    with pytest.raises(fixture.CrashBlocked):
        fixture.run_crash_probe()
    assert starts == [1] and exit_checks == [True, True]


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"private-output",
        b"x" * 41,
        fixture._PACKET.pack(b"WRONG001", 1, 2, 255, 1, 1, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 0, 2, 255, 1, 1, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 1, 0, 255, 1, 1, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 1, 2, 254, 1, 1, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 1, 2, 255, 0, 1, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 1, 2, 255, 1, 0, 1, 1),
        fixture._PACKET.pack(fixture._MAGIC, 1, 2, 255, 1, 1, 2, 1),
    ],
)
def test_ready_message_bounds_and_sanitization(raw):
    with pytest.raises(fixture.CrashBlocked) as caught:
        fixture.decode_ready(raw)
    assert "private-output" not in str(caught.value)


def test_interrupted_dispatch_stays_pending_until_explicit_unknown_no_retry():
    ledger = ExecutionLedger(1)
    ledger.reserve(Case.FIRST)
    ledger.dispatch()
    # No completion acknowledgement: count is retained, a new reservation is forbidden.
    assert ledger.entries[0].outcome is Outcome.DISPATCHED
    with pytest.raises(LedgerInvalid):
        ledger.reserve(Case.SECOND)
    ledger.finish(Outcome.UNKNOWN)
    assert ledger.reserved == ledger.charged == 1
    with pytest.raises(LedgerInvalid):
        ledger.reserve(Case.SECOND)
    # This is a living supervisor's memory, NOT recovery of a dead process's ledger.


def test_bounded_fixture_has_only_sleep_no_model_file_or_network_access():
    tree = ast.parse(fixture._FINITE_WORKER)
    assert [n.names[0].name for n in tree.body if isinstance(n, ast.Import)] == ["time"]
    calls = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert calls == ["time.sleep(8.0)"]
    assert fixture._PACKET.size == 40
