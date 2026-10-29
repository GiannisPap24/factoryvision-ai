"""Single-use Windows synthetic IPC exercise; no real-service or supplied-image entry."""

import ctypes as c
import math
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes as w
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from factoryvision.robustness import synthetic_ipc as wire
from factoryvision.robustness import windows_isolation as wi
from factoryvision.robustness.development_boundary import CompatibilityObservation, ProbeStatus
from factoryvision.robustness.execution_ledger import Case, Entry, ExecutionLedger, Outcome
from factoryvision.robustness.reporting import _bounded, _json
from factoryvision.robustness.synthetic_ipc_worker import FAULTS

_LIMITS = wi.Limits(job_mib=256, process_mib=256, deadline_seconds=5.0)


class _StartupEx(c.Structure):
    _fields_ = [("startup", wi._Startup), ("attributes", c.c_void_p)]


class _IpcWin32(wi._Win32):
    """Only new primitive: a fixed suspended worker with two allowlisted pipe handles."""

    _fault = "none"  # Private, finite synthetic fault seam; never an IPC command/public API.

    def __init__(self):
        super().__init__()
        h, d, p = w.HANDLE, w.DWORD, c.c_void_p
        for name, args, result in (
            ("CreatePipe", [p, p, p, d], w.BOOL),
            ("SetHandleInformation", [h, d, d], w.BOOL),
            ("GetHandleInformation", [h, p], w.BOOL),
            ("InitializeProcThreadAttributeList", [p, d, d, p], w.BOOL),
            ("UpdateProcThreadAttribute", [p, d, c.c_size_t, p, c.c_size_t, p, p], w.BOOL),
            ("DeleteProcThreadAttributeList", [p], None),
            ("ReadFile", [h, p, d, p, p], w.BOOL),
            ("WriteFile", [h, p, d, p, p], w.BOOL),
        ):
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result
        self.handles = []
        self.pipe_closed = True
        self.allowlist_verified = False
        self.created_process = wi._Process()

    def inheritable(self, handle):
        flags = w.DWORD()
        self.check(self.k.GetHandleInformation(handle, c.byref(flags)), "handle_flags")
        return bool(flags.value & 1)

    def pipes(self):
        for _ in range(2):
            read, write = w.HANDLE(), w.HANDLE()
            self.check(self.k.CreatePipe(c.byref(read), c.byref(write), None, 65536), "pipe")
            self.handles.extend((read.value, write.value))
        self.child_read, self.parent_write, self.parent_read, self.child_write = self.handles

    def close_pipe(self, handle):
        if handle in self.handles:
            try:
                closed = self.close(handle)
            except BaseException:
                closed = False
            self.pipe_closed &= closed
            if closed:
                self.handles.remove(handle)

    def launch(self, job):
        wire.check(type(self._fault) is str and self._fault in FAULTS)
        self.pipes()
        startup, process = _StartupEx(), wi._Process()
        self.created_process = process  # Retain ownership even if post-create setup fails.
        startup.startup.cb = c.sizeof(startup)
        startup.startup.flags = 0x100  # STARTF_USESTDHANDLES
        startup.startup.stdin = self.child_read
        startup.startup.stdout = startup.startup.stderr = self.child_write
        size = c.c_size_t()
        self.k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        wire.check(0 < size.value <= 65536)
        storage = c.create_string_buffer(size.value)
        self.check(
            self.k.InitializeProcThreadAttributeList(storage, 1, 0, c.byref(size)), "attributes"
        )
        startup.attributes = c.cast(storage, c.c_void_p)
        allowed = (w.HANDLE * 2)(self.child_read, self.child_write)
        try:
            for handle in allowed:
                self.check(self.k.SetHandleInformation(handle, 1, 1), "inherit_pipe")
            wire.check(not self.inheritable(job))
            wire.check(
                not self.inheritable(self.parent_read) and not self.inheritable(self.parent_write)
            )
            wire.check(all(self.inheritable(h) for h in allowed) and job not in allowed)
            self.check(
                self.k.UpdateProcThreadAttribute(
                    storage, 0, 0x20002, allowed, c.sizeof(allowed), None, None
                ),
                "handle_allowlist",
            )
            self.allowlist_verified = True
            # No site startup/.pth, shell, venv redirector, user module or environment.
            # Only this installed source root and the existing NumPy/Pillow dependency root.
            roots = [
                str(Path(__file__).resolve().parents[2]),
                str(Path(np.__file__).resolve().parents[1]),
            ]
            bootstrap = (
                "import sys; sys.path[:0]=" + repr(roots) + "\ntry:\n"
                " from factoryvision.robustness.synthetic_ipc_worker import main\n"
                " code=main()\nexcept BaseException: code=31\nsys.exit(code)"
            )
            executable = sys._base_executable
            command = c.create_unicode_buffer(
                subprocess.list2cmdline(
                    [executable, "-I", "-S", "-B", "-c", bootstrap, self._fault]
                )
            )
            environment = c.create_unicode_buffer(
                "SYSTEMROOT="
                + os.environ.get("SYSTEMROOT", "")
                + "\0OPENBLAS_NUM_THREADS=1\0OMP_NUM_THREADS=1\0\0"
            )
            self.check(
                self.k.CreateProcessW(
                    executable,
                    command,
                    None,
                    None,
                    True,
                    wi._CREATE_FLAGS | 0x80000,
                    environment,
                    sys.base_prefix,
                    c.byref(startup),
                    c.byref(process),
                ),
                "create_ipc_worker",
            )
            return process  # Still suspended; caller must assign/verify before resume.
        finally:
            self.k.DeleteProcThreadAttributeList(storage)
            # Parent copies of the child endpoints must close even on launch failure.
            for handle in (self.child_read, self.child_write):
                self.close_pipe(handle)

    def read(self, count):
        wire.check(0 < count <= wire.MAX_JSON)
        buffer, read = c.create_string_buffer(count), w.DWORD()
        self.check(self.k.ReadFile(self.parent_read, buffer, count, c.byref(read), None), "read")
        return buffer.raw[: read.value]

    def write(self, payload):
        wire.check(type(payload) is bytes and 0 < len(payload) <= wire.MAX_JSON + wire.MAX_PNG + 8)
        offset = 0
        while offset < len(payload):
            block = payload[offset:]
            buffer, written = c.create_string_buffer(block), w.DWORD()
            self.check(
                self.k.WriteFile(self.parent_write, buffer, len(block), c.byref(written), None),
                "write",
            )
            wire.check(0 < written.value <= len(block))
            offset += written.value

    def require_eof(self):
        buffer, read = c.create_string_buffer(1), w.DWORD()
        ok = self.k.ReadFile(self.parent_read, buffer, 1, c.byref(read), None)
        wire.check((not ok and c.get_last_error() == 109) or (ok and read.value == 0))


@dataclass(frozen=True, slots=True)
class IpcReport:
    entries: tuple[Entry, ...]
    observations: tuple[CompatibilityObservation, ...]
    isolation: wi.Result
    allowlist_verified: bool
    io_joined: bool


def json_bytes(report):
    wire.check(type(report) is IpcReport and type(report.entries) is tuple)
    wire.check(1 <= len(report.entries) <= 5 and type(report.observations) is tuple)
    for index, entry in enumerate(report.entries):
        wire.check(type(entry) is Entry and entry.case is tuple(Case)[index])
        entry.__post_init__()
        wire.check(entry.outcome not in (Outcome.RESERVED, Outcome.DISPATCHED))
        if index < len(report.entries) - 1:
            wire.check(entry.outcome is Outcome.SUCCEEDED)
    wire.check(
        len(report.observations) == sum(e.outcome is Outcome.SUCCEEDED for e in report.entries)
    )
    for i, row in enumerate(report.observations, 1):
        wire.check(type(row) is CompatibilityObservation and row.ordinal == i)
        row.__post_init__()
        wire.check(row.status is ProbeStatus.MEASURED and row.spatial is not None)
    info = report.isolation
    wire.check(type(info) is wi.Result)
    wire.check(all(type(value) is str for value in (info.kind, info.model_execution, info.reason)))
    wire.check(
        info.kind == "synthetic_windows_isolation_probe" and info.model_execution == "not_run"
    )
    wire.check(info.case is Case.FIRST and type(info.outcome) is Outcome)
    for value in (
        info.parent_in_job,
        info.child_inherited_job,
        info.assigned,
        info.membership_verified,
        info.resumed,
        info.exit_confirmed,
        info.job_empty_confirmed,
        info.handles_closed,
        info.limits_verified,
    ):
        wire.check(type(value) is bool)
    for value in (
        info.reserved,
        info.charged,
        info.peak_job_commit,
        info.peak_process_commit,
        info.observed_working_set,
    ):
        wire.check(type(value) is int and 0 <= value <= 2**40)
    wire.check(
        type(info.elapsed_seconds) is float
        and math.isfinite(info.elapsed_seconds)
        and info.elapsed_seconds >= 0
    )
    wire.check(
        type(info) is wi.Result
        and info.reason
        in (
            "complete",
            "setup_failed",
            "invalid_response",
            "execution_unknown",
            "cancelled",
            "wall_clock_deadline",
            "cleanup_unverified",
        )
    )
    wire.check(
        info.reserved == len(report.entries)
        and info.charged == sum(e.charged for e in report.entries)
    )
    wire.check(type(report.allowlist_verified) is bool and type(report.io_joined) is bool)
    expected = {
        "complete": Outcome.SUCCEEDED,
        "setup_failed": Outcome.NOT_STARTED,
        "invalid_response": Outcome.FAILED,
        "execution_unknown": Outcome.UNKNOWN,
        "cancelled": Outcome.CANCELLED,
        "wall_clock_deadline": Outcome.TIMED_OUT,
        "cleanup_unverified": Outcome.UNKNOWN if info.charged else Outcome.NOT_STARTED,
    }[info.reason]
    wire.check(info.outcome is expected)
    if info.reason != "cleanup_unverified":
        wire.check(report.entries[-1].outcome is expected)
    if info.reason == "complete":
        wire.check(len(report.observations) == 5 and info.outcome is Outcome.SUCCEEDED)
        wire.check(
            all(
                (
                    info.assigned,
                    info.membership_verified,
                    info.resumed,
                    info.limits_verified,
                    info.exit_confirmed,
                    info.job_empty_confirmed,
                    info.handles_closed,
                    report.allowlist_verified,
                    report.io_joined,
                )
            )
        )
    return _bounded(
        _json(
            {
                "kind": "synthetic_service_interface_probe",
                "model_execution": "not_run",
                "protocol_version": 1,
                "schema_version": 1,
                "numerical_protocol_version": 2,
                "recovery": "abort_only",
                "planned_attempts": 5,
                "run_status": "complete"
                if info.reason == "complete"
                else ("unknown" if info.outcome is Outcome.UNKNOWN else "aborted"),
                **asdict(report),
            }
        ),
        16384,
    )


class IsolatedCashewProbe:
    """One persistent synthetic worker. No reset, input path, executable or model argument."""

    def __init__(self):
        self._ledger = ExecutionLedger(5)
        self._used = False
        self._lock = threading.Lock()

    @property
    def entries(self):
        return self._ledger.entries

    def run(self, *, limits=_LIMITS, cancel=None):
        wire.check(type(limits) is wi.Limits)
        limits.__post_init__()
        wire.check(cancel is None or type(cancel) is threading.Event)
        wire.check(self._lock.acquire(blocking=False))
        try:
            wire.check(not self._used)
            self._used = True
            self._ledger.reserve(Case.FIRST)
            return self._run(limits, cancel)
        finally:
            self._lock.release()

    def _run(self, limits, cancel):
        start = time.monotonic()
        deadline = start + limits.deadline_seconds
        api = job = None
        process = wi._Process()
        parent = inherited = assigned = membership = resumed = verified = False
        exited = empty = closed = joined = True
        peak_job = peak_process = working_set = 0
        thread = None
        rows = []
        baseline = None
        outcome, reason = Outcome.NOT_STARTED, "setup_failed"

        def guard():
            if cancel is not None and cancel.is_set():
                raise InterruptedError()
            if time.monotonic() >= deadline:
                raise TimeoutError()

        def exchange(payload=None, *, terminal=False):
            nonlocal thread, working_set
            result = []
            done = threading.Event()

            def io():
                try:
                    if payload is not None:
                        api.write(payload)
                    result.append(wire.decode(wire.read_frame(api.read, wire.MAX_JSON)))
                    if terminal:
                        api.require_eof()
                except BaseException as error:
                    result[:] = [error]
                finally:
                    done.set()

            guard()
            thread = threading.Thread(target=io, daemon=True, name="synthetic-ipc")
            thread.start()
            while not done.wait(0.01):
                guard()
                working_set = max(working_set, api.working_set(process.process))
                # Let EOF propagate through the reader; the deadline still bounds completion.
            thread.join()
            guard()
            if isinstance(result[0], BaseException):
                raise result[0]
            return result[0]

        try:
            guard()
            api = _IpcWin32()
            parent = api.member(api.k.GetCurrentProcess(), None)
            job = api.create_job()
            api.configure(job, limits)
            verified = True
            guard()
            process = api.launch(job)
            inherited = api.member(process.process, None)
            wire.check(not api.exited(process.process))
            api.assign(job, process.process)
            assigned = True
            membership = api.member(process.process, job)
            wire.check(membership and api.query(job, 1, wi._Accounting).active == 1)
            wire.check(api.allowlist_verified and api.pipe_closed and not api.inheritable(job))
            guard()
            # Conservative first charge before resume, including failed import/handshake.
            self._ledger.dispatch()
            outcome, reason = Outcome.UNKNOWN, "execution_unknown"
            api.resume(process.thread)
            resumed = True
            ready = exchange()
            wire.validate(ready, "ready", 0, ("status",))
            wire.check(ready["status"] == "ready")
            for seq, (_, pixels) in enumerate(wire.inputs(), 1):
                if seq > 1:
                    self._ledger.reserve(tuple(Case)[seq - 1])
                    outcome, reason = Outcome.NOT_STARTED, "setup_failed"
                packet = wire.request(seq, pixels)
                guard()
                if seq > 1:
                    self._ledger.dispatch()
                outcome, reason = Outcome.UNKNOWN, "execution_unknown"
                data = exchange(packet)
                outcome, reason = Outcome.FAILED, "invalid_response"
                row = wire.observation(data, seq, baseline)
                if seq == 1:
                    baseline = data["baseline_score"]
                # Final attempt succeeds only after terminal ack and normal process exit.
                if seq == 5:
                    outcome, reason = Outcome.UNKNOWN, "execution_unknown"
                    done = exchange(wire.encode(wire.envelope("close", 6)), terminal=True)
                    wire.validate(done, "done", 6, ("status",))
                    wire.check(done["status"] == "complete")
                    while not api.exited(process.process, 10):
                        guard()
                    wire.check(api.exit_code(process.process) == 0)
                    guard()
                rows.append(row)
                self._ledger.finish(Outcome.SUCCEEDED)
                outcome, reason = Outcome.SUCCEEDED, "complete"
        except InterruptedError:
            outcome, reason = Outcome.CANCELLED, "cancelled"
        except TimeoutError:
            outcome, reason = Outcome.TIMED_OUT, "wall_clock_deadline"
        except wire.TruncatedFrame:
            outcome = Outcome.UNKNOWN if self._ledger.entries[-1].charged else Outcome.NOT_STARTED
            reason = "execution_unknown" if self._ledger.entries[-1].charged else "setup_failed"
        except wire.ProtocolError:
            reason = "invalid_response" if self._ledger.entries[-1].charged else "setup_failed"
            outcome = Outcome.FAILED if self._ledger.entries[-1].charged else Outcome.NOT_STARTED
        except BaseException:
            pass  # Preserve conservative stage outcome; never expose exception contents.
        finally:
            if api is not None:
                if not process.process:
                    process = api.created_process
                exited, empty, closed, peak_job, peak_process = api.cleanup(job, process, assigned)
                if thread is not None and thread.ident is not None:
                    thread.join(2.0)  # Owned child is dead/pipe peer closed before this wait.
                    joined = not thread.is_alive()
                if joined:
                    for handle in tuple(api.handles):
                        api.close_pipe(handle)
                closed = closed and api.pipe_closed and not api.handles and joined
            if not (exited and empty and closed and joined):
                reason = "cleanup_unverified"
                outcome = (
                    Outcome.UNKNOWN if self._ledger.entries[-1].charged else Outcome.NOT_STARTED
                )
            if self._ledger.entries[-1].outcome in (Outcome.RESERVED, Outcome.DISPATCHED):
                self._ledger.finish(outcome)
        info = wi.Result(
            Case.FIRST,
            outcome,
            reason,
            parent,
            inherited,
            assigned,
            membership,
            resumed,
            exited,
            empty,
            closed,
            verified,
            self._ledger.reserved,
            self._ledger.charged,
            time.monotonic() - start,
            peak_job,
            peak_process,
            working_set,
        )
        report = IpcReport(
            self.entries, tuple(rows), info, bool(api and api.allowlist_verified), joined
        )
        json_bytes(report)
        return report
