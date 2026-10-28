"""Bounded Windows synthetic probes only; never a model executor or security sandbox.

Every process/job handle is newly created here. No arbitrary command, executable,
script, path, callback, inherited handle or worker-supplied accounting is accepted.
"""

import ctypes as c
import math
import os
import subprocess
import sys
import threading
import time
from ctypes import wintypes as w
from dataclasses import dataclass
from enum import StrEnum

from factoryvision.robustness.execution_ledger import Case, ExecutionLedger, Outcome

MIB = 1024 * 1024
_FLAGS = 0x2000 | 0x100 | 0x200 | 0x8  # Kill-on-close, process/job commit, active process.
_CREATE_FLAGS = 0x4 | 0x08000000 | 0x400  # Suspended, no window, Unicode environment.


class Probe(StrEnum):
    SUCCESS = "success"
    EXCEPTION = "exception"
    INVALID = "invalid"
    HANG = "hang"
    EXIT = "exit"
    MEMORY = "memory"


# Fixed trusted source, executed with -I -S -B. Scalar IPC is the process exit code.
# The sole allocation is 80 MiB virtual commit, never a 2 GiB stress allocation.
_WORKER = """
import ctypes as c
import os
import sys
import time
def main():
    mode = sys.argv[1]
    if mode == 'success':
        return 10
    if mode == 'exception':
        raise RuntimeError('synthetic exception')
    if mode == 'invalid':
        return 31
    if mode == 'exit':
        os._exit(97)
    if mode == 'hang':
        while True:
            time.sleep(0.1)
    if mode == 'memory':
        k = c.WinDLL('kernel32', use_last_error=True)
        k.VirtualAlloc.argtypes = [c.c_void_p, c.c_size_t, c.c_ulong, c.c_ulong]
        k.VirtualAlloc.restype = c.c_void_p
        k.VirtualFree.argtypes = [c.c_void_p, c.c_size_t, c.c_ulong]
        k.VirtualFree.restype = c.c_int
        address = k.VirtualAlloc(None, 80 * 1024 * 1024, 0x3000, 4)
        if not address:
            return 12 if c.get_last_error() in (8, 1455) else 14
        time.sleep(0.05)
        return 13 if k.VirtualFree(address, 0, 0x8000) else 14
    return 31
try:
    code = main()
except Exception:
    code = 20
sys.exit(code)
"""


@dataclass(frozen=True, slots=True)
class Limits:
    job_mib: int = 256
    process_mib: int = 256
    deadline_seconds: float = 2.0

    def __post_init__(self):
        if (
            any(type(v) is not int or not 64 <= v <= 2048 for v in (self.job_mib, self.process_mib))
            or type(self.deadline_seconds) is not float
            or not math.isfinite(self.deadline_seconds)
            or not 0.05 <= self.deadline_seconds <= 10.0
        ):
            raise ValueError("Invalid synthetic isolation limits.")


_DEFAULT_LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class Result:
    case: Case
    outcome: Outcome
    reason: str
    parent_in_job: bool
    child_inherited_job: bool
    assigned: bool
    membership_verified: bool
    resumed: bool
    exit_confirmed: bool
    job_empty_confirmed: bool
    handles_closed: bool
    limits_verified: bool
    reserved: int
    charged: int
    elapsed_seconds: float
    peak_job_commit: int
    peak_process_commit: int
    observed_working_set: int
    kind: str = "synthetic_windows_isolation_probe"
    model_execution: str = "not_run"


class _Failure(Exception):
    def __init__(self, stage, code=0):
        category = {5: "access_denied", 8: "resource_limit", 87: "unsupported"}.get(code, "failed")
        self.reason = stage + ":" + category
        super().__init__(self.reason)


class _BasicLimits(c.Structure):
    _fields_ = [
        ("process_time", c.c_longlong),
        ("job_time", c.c_longlong),
        ("flags", w.DWORD),
        ("min_ws", c.c_size_t),
        ("max_ws", c.c_size_t),
        ("active", w.DWORD),
        ("affinity", c.c_size_t),
        ("priority", w.DWORD),
        ("scheduling", w.DWORD),
    ]


class _Extended(c.Structure):
    _fields_ = [
        ("basic", _BasicLimits),
        ("io", c.c_ulonglong * 6),
        ("process_memory", c.c_size_t),
        ("job_memory", c.c_size_t),
        ("peak_process", c.c_size_t),
        ("peak_job", c.c_size_t),
    ]


class _Accounting(c.Structure):
    _fields_ = [
        ("times", c.c_longlong * 4),
        ("page_faults", w.DWORD),
        ("total", w.DWORD),
        ("active", w.DWORD),
        ("terminated", w.DWORD),
    ]


class _Memory(c.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("faults", w.DWORD),
        ("peak_ws", c.c_size_t),
        ("ws", c.c_size_t),
        ("other", c.c_size_t * 6),
    ]


class _Startup(c.Structure):
    _fields_ = [
        ("cb", w.DWORD),
        ("reserved", w.LPWSTR),
        ("desktop", w.LPWSTR),
        ("title", w.LPWSTR),
        ("x", w.DWORD),
        ("y", w.DWORD),
        ("xsize", w.DWORD),
        ("ysize", w.DWORD),
        ("xchars", w.DWORD),
        ("ychars", w.DWORD),
        ("fill", w.DWORD),
        ("flags", w.DWORD),
        ("show", w.WORD),
        ("reserved2_size", w.WORD),
        ("reserved2", c.c_void_p),
        ("stdin", w.HANDLE),
        ("stdout", w.HANDLE),
        ("stderr", w.HANDLE),
    ]


class _Process(c.Structure):
    _fields_ = [
        ("process", w.HANDLE),
        ("thread", w.HANDLE),
        ("pid", w.DWORD),
        ("tid", w.DWORD),
    ]


class _Win32:
    """Private owned-handle backend. No lookup/open of existing processes/jobs."""

    def __init__(self):
        if sys.platform != "win32":
            raise _Failure("platform")
        self.k = c.WinDLL("kernel32", use_last_error=True)
        h, d, p = w.HANDLE, w.DWORD, c.c_void_p
        signatures = {
            "CreateJobObjectW": ([p, w.LPCWSTR], h),
            "SetInformationJobObject": ([h, c.c_int, p, d], w.BOOL),
            "QueryInformationJobObject": ([h, c.c_int, p, d, p], w.BOOL),
            "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, p, p, w.BOOL, d, p, w.LPCWSTR, p, p], w.BOOL),
            "AssignProcessToJobObject": ([h, h], w.BOOL),
            "IsProcessInJob": ([h, h, p], w.BOOL),
            "GetCurrentProcess": ([], h),
            "ResumeThread": ([h], d),
            "WaitForSingleObject": ([h, d], d),
            "GetExitCodeProcess": ([h, p], w.BOOL),
            "TerminateJobObject": ([h, w.UINT], w.BOOL),
            "TerminateProcess": ([h, w.UINT], w.BOOL),
            "CloseHandle": ([h], w.BOOL),
            "K32GetProcessMemoryInfo": ([h, p, d], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result

    @staticmethod
    def check(value, stage):
        if not value:
            raise _Failure(stage, c.get_last_error())
        return value

    def member(self, process, job):
        flag = w.BOOL()
        self.check(self.k.IsProcessInJob(process, job, c.byref(flag)), "membership")
        return bool(flag.value)

    def create_job(self):
        return self.check(self.k.CreateJobObjectW(None, None), "create_job")

    def query(self, job, cls, structure):
        value = structure()
        self.check(
            self.k.QueryInformationJobObject(job, cls, c.byref(value), c.sizeof(value), None),
            "query_job",
        )
        return value

    def configure(self, job, limits):
        info = _Extended()
        info.basic.flags, info.basic.active = _FLAGS, 1
        info.process_memory = limits.process_mib * MIB
        info.job_memory = limits.job_mib * MIB
        self.check(
            self.k.SetInformationJobObject(job, 9, c.byref(info), c.sizeof(info)), "set_limits"
        )
        actual = self.query(job, 9, _Extended)
        if (
            actual.basic.flags != _FLAGS
            or actual.basic.active != 1
            or actual.process_memory != info.process_memory
            or actual.job_memory != info.job_memory
            or actual.basic.process_time
            or actual.basic.job_time
        ):
            raise _Failure("limit_verification")
        if self.query(job, 1, _Accounting).active != 0:
            raise _Failure("initial_job_not_empty")

    def create_worker(self, probe):
        startup, process = _Startup(), _Process()
        startup.cb = c.sizeof(startup)
        # Windows venv python.exe is a redirector that may spawn another process.
        # Use this running interpreter's base executable directly, stdlib only.
        executable = sys._base_executable
        # Explicit executable, no shell, no user flags, no inherited handles/environment.
        command = c.create_unicode_buffer(
            subprocess.list2cmdline(
                [
                    executable,
                    "-I",
                    "-S",
                    "-B",
                    "-c",
                    _WORKER,
                    probe.value,
                ]
            )
        )
        environment = c.create_unicode_buffer(
            "SYSTEMROOT=" + os.environ.get("SYSTEMROOT", "") + "\0\0"
        )
        self.check(
            self.k.CreateProcessW(
                executable,
                command,
                None,
                None,
                False,
                _CREATE_FLAGS,
                environment,
                sys.base_prefix,
                c.byref(startup),
                c.byref(process),
            ),
            "create_worker",
        )
        return process

    def assign(self, job, process):
        self.check(self.k.AssignProcessToJobObject(job, process), "assign_job")

    def resume(self, thread):
        if self.k.ResumeThread(thread) != 1:
            raise _Failure("resume")

    def exited(self, process, milliseconds=0):
        value = self.k.WaitForSingleObject(process, milliseconds)
        if value not in (0, 258):
            raise _Failure("wait")
        return value == 0

    def exit_code(self, process):
        value = w.DWORD()
        self.check(self.k.GetExitCodeProcess(process, c.byref(value)), "exit_code")
        return value.value

    def working_set(self, process):
        value = _Memory()
        value.cb = c.sizeof(value)
        # Observational only, never used to claim an RSS cap.
        return value.ws if self.k.K32GetProcessMemoryInfo(process, c.byref(value), value.cb) else 0

    def terminate(self, job, process, assigned):
        if assigned:
            self.check(self.k.TerminateJobObject(job, 90), "terminate_job")
        elif process:
            self.check(self.k.TerminateProcess(process, 90), "terminate_suspended_child")

    def close(self, handle):
        return bool(self.k.CloseHandle(handle))


class SyntheticSupervisor:
    """Sequential, bounded test harness. A fresh dedicated job for each reservation.

    Never attaches to real admission/services. Limits apply to synthetic processes only.
    In-process callers are trusted; monkeypatching Python is not a security boundary.
    """

    def __init__(self, budget=5):
        self._ledger = ExecutionLedger(budget)
        self._lock = threading.Lock()
        self._blocked = False

    @property
    def entries(self):
        return self._ledger.entries

    def run(self, case: Case, probe: Probe, *, limits=_DEFAULT_LIMITS, cancel=None):
        if type(probe) is not Probe or type(limits) is not Limits:
            raise ValueError("Invalid synthetic isolation request.")
        limits.__post_init__()
        if cancel is not None and type(cancel) is not threading.Event:
            raise ValueError("Invalid synthetic cancellation signal.")
        if not self._lock.acquire(blocking=False):
            raise ValueError("Synthetic supervisor is already running.")
        try:
            if self._blocked:
                raise ValueError("Synthetic supervisor cleanup is unverified; execution blocked.")
            self._ledger.reserve(case)
            return self._run(case, probe, limits, cancel)
        finally:
            self._lock.release()

    def _run(self, case, probe, limits, cancel):
        start = time.monotonic()
        deadline = start + limits.deadline_seconds
        api = job = None
        process = _Process()
        parent = inherited = assigned = membership = resumed = verified = False
        exit_confirmed = empty = closed = True
        peak_job = peak_process = working_set = 0
        outcome, reason = Outcome.NOT_STARTED, "not_started"

        def stop_reason():
            if cancel is not None and cancel.is_set():
                return Outcome.CANCELLED, "cancelled"
            if time.monotonic() >= deadline:
                return Outcome.TIMED_OUT, "wall_clock_deadline"
            return None

        def close_owned(handle):
            try:
                return api.close(handle)
            except BaseException:
                return False

        try:
            api = _Win32()
            parent = api.member(api.k.GetCurrentProcess(), None)  # Query only.
            job = api.create_job()
            empty = False
            api.configure(job, limits)
            verified = True
            stop = stop_reason()
            if stop:
                outcome, reason = stop
            else:
                process = api.create_worker(probe)
                exit_confirmed = False
                inherited = api.member(process.process, None)
                if api.exited(process.process):
                    raise _Failure("suspended_child_exited")
                api.assign(job, process.process)
                assigned = True
                membership = api.member(process.process, job)
                if not membership or api.query(job, 1, _Accounting).active != 1:
                    raise _Failure("assignment_verification")
                stop = stop_reason()
                if stop:
                    outcome, reason = stop
                else:
                    self._ledger.dispatch()  # Charge BEFORE ResumeThread; never refunded.
                    outcome, reason = Outcome.UNKNOWN, "dispatch_unacknowledged"
                    api.resume(process.thread)
                    resumed = True
                    while True:
                        stop = stop_reason()
                        if stop:
                            outcome, reason = stop
                            break
                        working_set = max(working_set, api.working_set(process.process))
                        if api.exited(process.process, 10):
                            code = api.exit_code(process.process)
                            accepted = (
                                {10: "completed"}
                                if probe is Probe.SUCCESS
                                else (
                                    {12: "allocation_denied", 13: "allocation_succeeded"}
                                    if probe is Probe.MEMORY
                                    else {}
                                )
                            )
                            if code in accepted:
                                outcome, reason = Outcome.SUCCEEDED, accepted[code]
                            else:
                                outcome = Outcome.FAILED
                                reason = {
                                    20: "worker_exception",
                                    31: "invalid_output",
                                    97: "unexpected_exit",
                                }.get(code, "invalid_output")
                            break
        except _Failure as failure:
            reason = failure.reason
        except BaseException:
            # No host/child exception text, environment, paths or raw child output.
            reason = "supervisor_failure"
        finally:
            # Cleanup is bounded separately: each wait is <= 2s, at most 6s total.
            # This grace is not hidden in the execution deadline or claimed as real time.
            try:
                if process.process:
                    if not api.exited(process.process):
                        api.terminate(job, process.process, assigned)
                    exit_confirmed = api.exited(process.process, 2000)
                if job:
                    until = time.monotonic() + 2.0
                    while True:
                        empty = api.query(job, 1, _Accounting).active == 0
                        if empty or time.monotonic() >= until:
                            break
                        time.sleep(0.005)
                    info = api.query(job, 9, _Extended)
                    peak_job, peak_process = info.peak_job, info.peak_process
            except BaseException:
                exit_confirmed = not bool(process.process)
                empty = False
            finally:
                # Close our job FIRST: kill-on-close is a final fallback if termination failed.
                if job:
                    closed = close_owned(job) and closed
                if process.process and not exit_confirmed:
                    try:
                        exit_confirmed = api.exited(process.process, 2000)
                    except BaseException:
                        exit_confirmed = False
                for handle in (process.thread, process.process):
                    if handle:
                        closed = close_owned(handle) and closed
            if not (exit_confirmed and empty and closed):
                self._blocked = True
                reason = "cleanup_unverified"
                outcome = (
                    Outcome.UNKNOWN if self._ledger.entries[-1].charged else Outcome.NOT_STARTED
                )
            self._ledger.finish(outcome)
        return Result(
            case,
            outcome,
            reason,
            parent,
            inherited,
            assigned,
            membership,
            resumed,
            exit_confirmed,
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
