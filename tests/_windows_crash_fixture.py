"""Disposable crash harness; no production command API, model imports or file outputs.

Only the fixed helper can be launched. Two explicitly allowlisted handles enter
the disposable supervisor: readiness pipe and query-only controller process handle.
The production worker launcher continues to use bInheritHandles=FALSE.
"""

import ctypes as c
import os
import struct
import subprocess
import sys
import time
from ctypes import wintypes as w
from pathlib import Path

# Direct isolated execution imports only these known stdlib-only project modules.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from factoryvision.robustness import windows_isolation as wi
from factoryvision.robustness.execution_ledger import Case, Outcome

_PACKET = struct.Struct("<8sQIIIIII")
_MAGIC = b"FVCRASH1"
_FINITE_WORKER = "import time\ntime.sleep(8.0)\n"


class CrashBlocked(RuntimeError):
    def __init__(self):
        super().__init__("Synthetic crash containment could not be verified; execution blocked.")


def require(condition):
    if not condition:
        raise CrashBlocked()


class _Security(c.Structure):
    _fields_ = [("size", w.DWORD), ("descriptor", c.c_void_p), ("inherit", w.BOOL)]


class _StartupEx(c.Structure):
    _fields_ = [("startup", wi._Startup), ("attributes", c.c_void_p)]


class CrashAPI(wi._Win32):
    def __init__(self):
        super().__init__()
        h, p, d, size = w.HANDLE, c.c_void_p, w.DWORD, c.c_size_t
        signatures = {
            "GetHandleInformation": ([h, p], w.BOOL),
            "SetHandleInformation": ([h, d, d], w.BOOL),
            "GetProcessId": ([h], d),
            "DuplicateHandle": ([h, h, h, p, d, w.BOOL, d], w.BOOL),
            "CreatePipe": ([p, p, p, d], w.BOOL),
            "PeekNamedPipe": ([h, p, d, p, p, p], w.BOOL),
            "ReadFile": ([h, p, d, p, p], w.BOOL),
            "WriteFile": ([h, p, d, p, p], w.BOOL),
            "InitializeProcThreadAttributeList": ([p, d, d, p], w.BOOL),
            "UpdateProcThreadAttribute": ([p, d, size, p, size, p, p], w.BOOL),
            "DeleteProcThreadAttributeList": ([p], None),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, result

    def flags(self, handle):
        flags = w.DWORD()
        require(self.k.GetHandleInformation(handle, c.byref(flags)))
        return flags.value

    def noninheritable(self, handle):
        require(self.k.SetHandleInformation(handle, 1, 0))
        require(self.flags(handle) == 0)

    def duplicate(self, source_process, source_handle, access, inherit=False):
        target = w.HANDLE()
        require(
            self.k.DuplicateHandle(
                source_process,
                source_handle,
                self.k.GetCurrentProcess(),
                c.byref(target),
                access,
                inherit,
                0,
            )
        )
        return target.value


def decode_ready(raw):
    """Fixed-size internal IPC only; no text, paths, arbitrary counters or object decoding."""
    require(type(raw) is bytes and len(raw) == _PACKET.size)
    magic, handle, pid, checks, reserved, charged, parent, inherited = _PACKET.unpack(raw)
    require(magic == _MAGIC and 0 < handle < 2**63 and pid > 0)
    require(checks == 255 and reserved == charged == 1)
    require(parent in (0, 1) and inherited in (0, 1))
    return handle, pid, bool(parent), bool(inherited)


def disposable_supervisor(pipe, controller):
    """Called only in a disposable interpreter, never in the test controller."""
    api = CrashAPI()
    require(api.flags(pipe) == api.flags(controller) == 1)
    api.noninheritable(pipe)
    api.noninheritable(controller)
    supervisor = wi.SyntheticSupervisor(1)
    parent_in_job = api.member(api.k.GetCurrentProcess(), None)

    class ObservedAPI(CrashAPI):
        def create_job(self):
            self.owned_job = super().create_job()
            return self.owned_job

        def create_worker(self, probe):
            # Reservation is authoritative BEFORE creation, not derived from the worker.
            require(supervisor.entries[0].outcome is Outcome.RESERVED)
            require(supervisor.entries[0].charged == 0)
            self.owned_worker = super().create_worker(probe)
            return self.owned_worker

        def assign(self, job, process):
            # At this point the real supervisor already owns both returned handles,
            # so a failed observation cannot strand an unassigned suspended child.
            require(process == self.owned_worker.process)
            self.inherited_job = self.member(process, None)
            super().assign(job, process)

        def resume(self, thread):
            job, worker = self.owned_job, self.owned_worker
            require(thread == worker.thread)
            entry = supervisor.entries[0]
            require(entry.outcome is Outcome.DISPATCHED and entry.charged == 1)
            info = self.query(job, 9, wi._Extended)
            checks = [
                self.flags(job) == 0,
                self.flags(worker.process) == 0,
                self.flags(worker.thread) == 0,
                not self.member(self.k.GetCurrentProcess(), job),
                not self.member(controller, job),
                self.member(worker.process, job),
                info.basic.flags == wi._FLAGS and bool(info.basic.flags & 0x2000),
                info.basic.active == 1 and self.query(job, 1, wi._Accounting).active == 1,
            ]
            require(all(checks))
            require(
                not any(
                    name == "torch"
                    or name == "numpy"
                    or name == "PIL"
                    or name.startswith("factoryvision.inference")
                    for name in sys.modules
                )
            )
            require(info.job_memory == info.process_memory == 256 * wi.MIB)
            require(not self.exited(worker.process))
            super().resume(thread)
            raw = _PACKET.pack(
                _MAGIC,
                worker.process,
                worker.pid,
                255,
                1,
                entry.charged,
                int(parent_in_job),
                int(self.inherited_job),
            )
            written = w.DWORD()
            require(self.k.WriteFile(pipe, raw, len(raw), c.byref(written), None))
            require(written.value == len(raw))

    # Test-local fixed fixture replaces only the worker body, not creation, flags,
    # assignment, limit verification, dispatch, resume, watchdog or cleanup logic.
    wi._WORKER = _FINITE_WORKER
    wi._Win32 = ObservedAPI
    try:
        supervisor.run(Case.FIRST, wi.Probe.HANG, limits=wi.Limits(deadline_seconds=5.0))
    finally:
        require(api.close(pipe))
        require(api.close(controller))
    # Reaching here means the controller did not perform the intended crash in time.
    raise CrashBlocked()


def run_crash_probe():
    """One crash only. No retries/recovery. Returns no PIDs, handles or paths.

    The controller never acquires a worker Job Object handle. Its only duplicated
    remote handle is the known worker process handle, for wait/query and emergency
    cleanup. It cannot keep the job alive after supervisor termination.
    """
    api = CrashAPI()
    own = []
    process = wi._Process()
    worker = None
    attributes = None
    initialized = False
    parent_before = api.member(api.k.GetCurrentProcess(), None)
    result = None
    cleanup_ok = True

    def keep(handle):
        own.append(handle)
        return handle

    def release(handle):
        require(api.close(handle))
        own.remove(handle)

    try:
        read, write = w.HANDLE(), w.HANDLE()
        security = _Security(c.sizeof(_Security), None, True)
        require(api.k.CreatePipe(c.byref(read), c.byref(write), c.byref(security), 4096))
        reader, writer = keep(read.value), keep(write.value)
        api.noninheritable(reader)
        # Query-only; no controller termination, mutation or duplication rights.
        controller = keep(
            api.duplicate(
                api.k.GetCurrentProcess(),
                api.k.GetCurrentProcess(),
                0x400,
                True,
            )
        )
        require(api.flags(writer) == api.flags(controller) == 1)
        size = c.c_size_t()
        api.k.InitializeProcThreadAttributeList(None, 1, 0, c.byref(size))
        require(0 < size.value <= 4096)
        attributes = c.create_string_buffer(size.value)
        require(api.k.InitializeProcThreadAttributeList(attributes, 1, 0, c.byref(size)))
        initialized = True
        handles = (w.HANDLE * 2)(writer, controller)
        require(
            api.k.UpdateProcThreadAttribute(
                attributes,
                0,
                0x20002,
                handles,
                c.sizeof(handles),
                None,
                None,
            )
        )  # PROC_THREAD_ATTRIBUTE_HANDLE_LIST, exactly two handles.
        startup = _StartupEx()
        startup.startup.cb = c.sizeof(startup)
        startup.attributes = c.cast(attributes, c.c_void_p)
        command = c.create_unicode_buffer(
            subprocess.list2cmdline(
                [
                    sys._base_executable,
                    "-I",
                    "-S",
                    "-B",
                    str(Path(__file__).absolute()),
                    str(writer),
                    str(controller),
                ]
            )
        )
        environment = c.create_unicode_buffer(
            "SYSTEMROOT=" + os.environ.get("SYSTEMROOT", "") + "\0\0"
        )
        require(
            api.k.CreateProcessW(
                sys._base_executable,
                command,
                None,
                None,
                True,
                wi._CREATE_FLAGS | 0x80000,
                environment,
                sys.base_prefix,
                c.byref(startup),
                c.byref(process),
            )
        )  # EXTENDED_STARTUPINFO_PRESENT; no breakaway.
        keep(process.process)
        keep(process.thread)
        require(api.k.GetProcessId(process.process) == process.pid)
        require(api.flags(process.process) == api.flags(process.thread) == 0)
        release(writer)
        release(controller)
        api.resume(process.thread)
        release(process.thread)

        # Readiness is bounded, fixed-size and never exposes child diagnostics.
        until = time.monotonic() + 3.0
        while True:
            available = w.DWORD()
            require(api.k.PeekNamedPipe(reader, None, 0, None, c.byref(available), None))
            require(available.value <= _PACKET.size)
            if available.value == _PACKET.size:
                break
            require(not api.exited(process.process) and time.monotonic() < until)
            time.sleep(0.005)
        buffer, received = c.create_string_buffer(_PACKET.size), w.DWORD()
        require(api.k.ReadFile(reader, buffer, _PACKET.size, c.byref(received), None))
        require(received.value == _PACKET.size)
        remote_handle, pid, supervisor_parent, worker_inherited = decode_ready(buffer.raw)
        # Duplication uses our verified supervisor handle, never OpenProcess/PID lookup.
        worker = keep(api.duplicate(process.process, remote_handle, 0x100401))
        require(api.k.GetProcessId(worker) == pid and api.flags(worker) == 0)
        require(not api.exited(worker, 75) and not api.exited(process.process))
        crash_start = time.monotonic()
        require(api.k.TerminateProcess(process.process, 93))
        require(api.exited(process.process, 1000) and api.exit_code(process.process) == 93)
        # This must beat BOTH the worker's 8s natural exit and supervisor's 5s watchdog.
        require(api.exited(worker, 1000))
        latency = time.monotonic() - crash_start
        require(latency < 2.0)
        require(api.member(api.k.GetCurrentProcess(), None) == parent_before)
        result = {
            "supervisor_exited": True,
            "worker_exited": True,
            "controller_alive": True,
            "parent_membership_unchanged": True,
            "supervisor_inherited_job": supervisor_parent,
            "worker_inherited_job": worker_inherited,
            "crash_exit_seconds": latency,
            "last_observed_reserved": 1,
            "last_observed_charged": 1,
            "execution_outcome": "unknown",
            "recovery": "blocked",
        }
    except BaseException:
        raise CrashBlocked() from None
    finally:
        # Emergency cleanup can target only returned/duplicated owned handles.
        # It never converts a worker-survival failure into a successful result.
        for handle in (process.process, worker):
            if handle:
                try:
                    if not api.exited(handle):
                        require(api.k.TerminateProcess(handle, 94))
                    require(api.exited(handle, 2000))
                except BaseException:
                    cleanup_ok = False
        if initialized:
            api.k.DeleteProcThreadAttributeList(attributes)
        for handle in reversed(own):
            try:
                cleanup_ok = api.close(handle) and cleanup_ok
            except BaseException:
                cleanup_ok = False
        if not cleanup_ok:
            raise CrashBlocked() from None
    require(result is not None)
    result["owned_handles_closed"] = True
    return result


if __name__ == "__main__":
    try:
        require(len(sys.argv) == 3)
        require(all(value.isascii() and value.isdigit() for value in sys.argv[1:]))
        disposable_supervisor(int(sys.argv[1]), int(sys.argv[2]))
    except BaseException:
        # No traceback, filesystem path, environment or arbitrary child output.
        sys.exit(52)
