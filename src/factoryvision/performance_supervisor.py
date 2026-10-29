"""Single-use, six-worker synthetic benchmark. No configurable executable or workload."""

import ctypes as c
import os
import platform
import subprocess
import sys
import threading
import time
from ctypes import wintypes as w
from pathlib import Path

import numpy as np

from factoryvision import performance_baseline as b
from factoryvision.robustness import windows_isolation as wi
from factoryvision.robustness.cashew_isolated_ipc import _IpcWin32, _StartupEx
from factoryvision.robustness.execution_ledger import Case, ExecutionLedger, Outcome

LIMITS = wi.Limits(job_mib=2048, process_mib=2048, deadline_seconds=10.0)
_FAULTS = (
    "none",
    "hang",
    "exit",
    "malformed",
    "oversized",
    "trailing",
    "deny_io",
    "deny_model",
    "profile_display",
)


class _Api(_IpcWin32):
    """Reuse owned handles, pipe I/O and cleanup, with a separately fixed launch target."""

    _fault = "none"

    def __init__(self, side, repetition):
        b.case(side, repetition)
        super().__init__()
        self.side, self.repetition = side, repetition

    def launch(self, job):
        b.require(type(self._fault) is str and self._fault in _FAULTS)
        self.pipes()
        startup, process = _StartupEx(), wi._Process()
        self.created_process = process
        startup.startup.cb = c.sizeof(startup)
        startup.startup.flags = 0x100
        startup.startup.stdin = self.child_read
        startup.startup.stdout = startup.startup.stderr = self.child_write
        size = c.c_size_t()
        self.k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        b.require(0 < size.value <= 65536)
        storage = c.create_string_buffer(size.value)
        self.check(
            self.k.InitializeProcThreadAttributeList(storage, 1, 0, c.byref(size)), "attributes"
        )
        startup.attributes = c.cast(storage, c.c_void_p)
        allowed = (w.HANDLE * 2)(self.child_read, self.child_write)
        try:
            for handle in allowed:
                self.check(self.k.SetHandleInformation(handle, 1, 1), "inherit_pipe")
            b.require(
                not self.inheritable(job)
                and not self.inheritable(self.parent_read)
                and not self.inheritable(self.parent_write)
            )
            b.require(all(self.inheritable(h) for h in allowed) and job not in allowed)
            self.check(
                self.k.UpdateProcThreadAttribute(
                    storage, 0, 0x20002, allowed, c.sizeof(allowed), None, None
                ),
                "handle_allowlist",
            )
            self.allowlist_verified = True
            roots = [
                str(Path(__file__).resolve().parents[1]),
                str(Path(np.__file__).resolve().parents[1]),
            ]
            # Only fixed installed code/dependencies are importable; no site/.pth or shell.
            # Before imports, deny network, subprocesses, deserialization and disk writes.
            bootstrap = """import sys, time, os, tempfile
started=time.perf_counter()
sys.path[:0]=ROOTS
# Avoid tempfile's write-based directory discovery during torch import.
# This is a process-local hint only; all writes remain denied.
tempfile.tempdir=sys.base_prefix
os.environ['TORCHINDUCTOR_CACHE_DIR']=sys.base_prefix
def audit(event,args):
 # Inductor checks its cache directory at import, even without model execution.
 # Reject that mkdir before the OS call; makedirs(exist_ok=True) accepts the
 # existing runtime directory. No cache directory or cache file is created.
 if event == 'os.mkdir' and os.path.normcase(os.path.abspath(args[0])) == (
     os.path.normcase(sys.base_prefix)):
  raise FileExistsError('Existing runtime directory')
 if event == 'open':
  path,mode,flags=args
  writes=os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND
  if (mode and any(x in mode for x in 'wax+')) or flags & writes:
   raise RuntimeError('Write denied')
 if event in ('subprocess.Popen','os.system','pickle.find_class','os.mkdir',
              'os.remove','os.rename','os.rmdir') or (
     event.startswith('socket.') and event != 'socket.gethostname'):
  raise RuntimeError('Operation denied')
sys.addaudithook(audit)
try:
 from factoryvision.performance_worker import main
 code=main(started)
except BaseException:
 code=31
sys.exit(code)
""".replace("ROOTS", repr(roots))
            executable = sys._base_executable
            command = c.create_unicode_buffer(
                subprocess.list2cmdline(
                    [
                        executable,
                        "-I",
                        "-S",
                        "-B",
                        "-c",
                        bootstrap,
                        str(self.side),
                        str(self.repetition),
                        self._fault,
                    ]
                )
            )
            environment = c.create_unicode_buffer(
                "SYSTEMROOT="
                + os.environ.get("SYSTEMROOT", "")
                + "\0OPENBLAS_NUM_THREADS=1\0OMP_NUM_THREADS=4\0MKL_NUM_THREADS=4\0\0"
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
                "create_performance_worker",
            )
            return process
        finally:
            self.k.DeleteProcThreadAttributeList(storage)
            for handle in (self.child_read, self.child_write):
                self.close_pipe(handle)


class Benchmark:
    """Trusted in-process caller, fixed schedule, abort-only. No retry or resume API."""

    def __init__(self):
        self._used = False
        self._lock = threading.Lock()
        # Each case has three worker reservations; no change to the 5C five-unit ceiling.
        # One reservation here means one 35-pipeline worker, never a real-model grant.
        self._ledgers = {side: ExecutionLedger(3) for side in b.SIDES}

    def run(self, *, cancel=None):
        b.require(cancel is None or type(cancel) is threading.Event)
        b.require(self._lock.acquire(blocking=False))
        try:
            b.require(not self._used)
            self._used = True
            report = dict(
                kind=b.KIND,
                schema_version=1,
                model_execution="not_run",
                baseline_commit=b.BASELINE,
                source_sha256=b.source_identity(),
                protocol=dict(
                    warmups=b.WARMUPS,
                    timed=b.SAMPLES,
                    processes_per_case=b.PROCESSES,
                    threads=b.THREADS,
                    process_mib=2048,
                    job_mib=2048,
                    deadline_seconds=10.0,
                    percentile="linear_(n-1)*p",
                    recovery="abort_only",
                    artifact_verification="not_run",
                    model_loading="not_run",
                    neural_forward="not_run",
                ),
                host=dict(
                    architecture=platform.machine(),
                    logical_cpus=os.cpu_count(),
                    windows_build=platform.version(),
                ),
                attempts=[],
            )
            for side, repetition in b.SCHEDULE:
                row = self._one(side, repetition, cancel)
                report["attempts"].append(row)
                if row["outcome"] != "complete":
                    break
            b.report_bytes(report)
            return report
        finally:
            self._lock.release()

    def _one(self, side, repetition, cancel):
        ledger = self._ledgers[side]
        ledger.reserve(tuple(Case)[repetition - 1])
        started = time.perf_counter()
        deadline = started + LIMITS.deadline_seconds
        api = job = thread = None
        process = wi._Process()
        row = dict(
            side=side,
            repetition=repetition,
            outcome="setup_failed",
            reserved=1,
            charged=0,
            startup_to_ready_ms=None,
            wall_ms=0.0,
            cpu_ms=None,
            peak_process_commit=None,
            peak_job_commit=None,
            observed_working_set=None,
            assigned=False,
            membership_verified=False,
            resumed=False,
            limits_verified=False,
            allowlist_verified=False,
            exit_confirmed=True,
            job_empty_confirmed=True,
            handles_closed=True,
            io_joined=True,
            ready=None,
            result=None,
        )
        working_set = 0

        def guard():
            if cancel is not None and cancel.is_set():
                raise InterruptedError()
            if time.perf_counter() >= deadline:
                raise TimeoutError()

        def receive(kind, terminal=False):
            nonlocal thread, working_set
            result, done = [], threading.Event()

            def read():
                try:
                    data = b.read_message(api.read)
                    b.validate_worker(data, kind, side, repetition)
                    if terminal:
                        api.require_eof()
                    result.append(data)
                except BaseException:
                    result.append(None)
                finally:
                    done.set()

            guard()
            thread = threading.Thread(target=read, daemon=True, name="performance-scalars")
            thread.start()
            while not done.wait(0.01):
                guard()
                working_set = max(working_set, api.working_set(process.process))
            thread.join()
            guard()
            b.require(len(result) == 1 and result[0] is not None)
            return result[0]

        try:
            guard()
            api = _Api(side, repetition)
            job = api.create_job()
            api.configure(job, LIMITS)
            row["limits_verified"] = True
            guard()
            process = api.launch(job)
            b.require(not api.exited(process.process))
            api.assign(job, process.process)
            row["assigned"] = True
            row["membership_verified"] = api.member(process.process, job)
            b.require(row["membership_verified"] and api.query(job, 1, wi._Accounting).active == 1)
            row["allowlist_verified"] = api.allowlist_verified
            b.require(api.allowlist_verified and api.pipe_closed and not api.inheritable(job))
            guard()
            ledger.dispatch()
            row["charged"], row["outcome"] = 1, "worker_failed"
            api.resume(process.thread)
            row["resumed"] = True
            row["ready"] = receive("ready")
            row["startup_to_ready_ms"] = (time.perf_counter() - started) * 1000
            row["result"] = receive("result", terminal=True)
            while not api.exited(process.process, 10):
                guard()
            b.require(api.exit_code(process.process) == 0)
            guard()
            row["outcome"] = "complete"
        except InterruptedError:
            row["outcome"] = "cancelled"
        except TimeoutError:
            row["outcome"] = "timeout"
        except b.InvalidBenchmark:
            row["outcome"] = "invalid_output" if row["charged"] else "setup_failed"
        except BaseException:
            pass  # Fixed categories only, no filenames or raw exception text.
        finally:
            if api is not None:
                if not process.process:
                    process = api.created_process
                if job:
                    try:
                        accounting = api.query(job, 1, wi._Accounting)
                        row["cpu_ms"] = (accounting.times[0] + accounting.times[1]) / 10000
                    except BaseException:
                        pass
                exited, empty, closed, peak_job, peak_process = api.cleanup(
                    job, process, row["assigned"]
                )
                if thread is not None and thread.ident is not None:
                    thread.join(2.0)
                    row["io_joined"] = not thread.is_alive()
                if row["io_joined"]:
                    for handle in tuple(api.handles):
                        api.close_pipe(handle)
                row.update(
                    exit_confirmed=exited,
                    job_empty_confirmed=empty,
                    handles_closed=closed
                    and api.pipe_closed
                    and not api.handles
                    and row["io_joined"],
                    peak_process_commit=peak_process or None,
                    peak_job_commit=peak_job or None,
                    observed_working_set=working_set or None,
                )
            if not all(
                row[k]
                for k in ("exit_confirmed", "job_empty_confirmed", "handles_closed", "io_joined")
            ):
                row["outcome"] = "cleanup_unverified"
            if row["outcome"] != "complete":
                row["result"] = None
            ledger.finish(
                {
                    "complete": Outcome.SUCCEEDED,
                    "cancelled": Outcome.CANCELLED,
                    "timeout": Outcome.TIMED_OUT,
                }.get(row["outcome"], Outcome.FAILED if row["charged"] else Outcome.NOT_STARTED)
            )
            row["wall_ms"] = (time.perf_counter() - started) * 1000
        return row


def main():
    b.require(len(sys.argv) == 1)
    report = Benchmark().run()
    sys.stdout.buffer.write(b.report_bytes(report) + b"\n")
    return (
        0 if len(report["attempts"]) == 6 and report["attempts"][-1]["outcome"] == "complete" else 1
    )


if __name__ == "__main__":
    try:
        code = main()
    except BaseException:
        print(
            '{"kind":"synthetic_cpu_pipeline_benchmark","model_execution":"not_run",'
            '"status":"aborted","reason":"invalid_execution"}'
        )
        code = 1
    sys.exit(code)
