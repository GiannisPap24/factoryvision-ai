"""Real Windows subprocesses, tiny owned synthetic images, no model/data access."""

import ctypes as c
import io
import json
import struct
import sys
import threading
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from factoryvision.inference import service, uploads
from factoryvision.robustness import cashew_development as cd
from factoryvision.robustness import cashew_isolated_ipc as ipc
from factoryvision.robustness import synthetic_ipc as wire
from factoryvision.robustness import synthetic_ipc_worker as worker
from factoryvision.robustness import windows_isolation as wi
from factoryvision.robustness.development_admission import DevelopmentAdmission, ExecutionDenied
from factoryvision.robustness.development_boundary import SyntheticInspectionDouble
from factoryvision.robustness.execution_ledger import Case, LedgerInvalid, Outcome

windows = pytest.mark.skipif(sys.platform != "win32", reason="Actual Windows IPC required")


@pytest.fixture(autouse=True)
def no_real_access(monkeypatch):
    violations = []

    def denied(*args, **kwargs):
        violations.append(True)
        raise AssertionError("No actual model or data access")

    for target, name in (
        (torch, "load"),
        (torch.nn.Module, "_call_impl"),
        (service.InspectionService, "__init__"),
        (service.InspectionService, "inspect"),
        (service, "load_checkpoint"),
        (service, "verified_asset"),
        (Path, "glob"),
        (Path, "rglob"),
    ):
        monkeypatch.setattr(target, name, denied)
    monkeypatch.setattr(uploads, "protected_hashes", lambda: (frozenset(), frozenset()))
    yield
    assert not violations


def clean(report):
    assert report.io_joined
    assert report.isolation.exit_confirmed
    assert report.isolation.job_empty_confirmed
    assert report.isolation.handles_closed


def test_extracted_double_exactly_preserves_existing_inspection_result():
    double = SyntheticInspectionDouble(wire.spec())
    for _, pixels in wire.inputs():
        output = double.evaluate_output(pixels)
        result = double.evaluate(pixels)
        assert result.score == output.score
        assert result.above_threshold == (output.score > 0.5)
        np.testing.assert_array_equal(result.spatial, output.spatial)
        np.testing.assert_array_equal(result.original, pixels)


@windows
def test_five_real_ipc_attempts_determinism_and_existing_diagnostics():
    first = ipc.IsolatedCashewProbe()
    report = first.run()
    clean(report)
    assert report.isolation.reason == "complete"
    assert report.isolation.reserved == report.isolation.charged == 5
    assert all(e.outcome is Outcome.SUCCEEDED for e in first.entries)
    assert report.allowlist_verified and report.isolation.membership_verified
    assert report.isolation.limits_verified and report.isolation.resumed
    assert report.isolation.peak_job_commit > 0
    expected = cd.CashewSyntheticPreparation(cd.SyntheticWorkerDouble(wire.spec())).run()
    assert report.observations == expected.observations
    repeat = ipc.IsolatedCashewProbe().run()
    clean(repeat)
    assert repeat.observations == report.observations
    assert report.observations[3].score.signed == -0.050049155950546265
    assert report.observations[4].spatial.mae == 0.04204249382019043
    with pytest.raises(wire.ProtocolError):
        first.run()
    with pytest.raises(LedgerInvalid):
        first._ledger.reserve(Case.FIRST)
    raw = ipc.json_bytes(report)
    assert len(raw) < 16384
    data = json.loads(raw)
    assert data["model_execution"] == "not_run" and data["recovery"] == "abort_only"
    assert data["kind"] == "synthetic_service_interface_probe"
    assert not any(s in raw.decode() for s in ("sha256", "pixels", "filename", "traceback"))


@windows
def test_launch_order_actual_allowlist_and_noninherited_job(monkeypatch):
    api = ipc._IpcWin32()
    # An unrelated inheritable event must not leak through the explicit handle list.
    api.k.CreateEventW.argtypes = [c.c_void_p, wi.w.BOOL, wi.w.BOOL, wi.w.LPCWSTR]
    api.k.CreateEventW.restype = wi.w.HANDLE
    sentinel = api.k.CreateEventW(None, True, False, None)
    assert sentinel
    assert api.k.SetHandleInformation(sentinel, 1, 1)
    events, allowed_sets, created, closed, failures = [], [], [], [], []
    probe = ipc.IsolatedCashewProbe()
    launch, assign, member, resume = (
        ipc._IpcWin32.launch,
        wi._Win32.assign,
        wi._Win32.member,
        wi._Win32.resume,
    )
    original_init, original_close = ipc._IpcWin32.__init__, wi._Win32.close

    def init(self):
        original_init(self)
        update, create = self.k.UpdateProcThreadAttribute, self.k.CreateProcessW

        def attribute(*args):
            assert args[2] == 0x20002
            handles = tuple(c.cast(args[3], c.POINTER(wi.w.HANDLE * 2)).contents)
            assert handles == (self.child_read, self.child_write)
            assert sentinel not in handles
            allowed_sets.append(handles)
            return update(*args)

        def process(*args):
            assert args[4] is True and args[5] == wi._CREATE_FLAGS | 0x80000
            assert args[0] == sys._base_executable
            command = args[1].value
            assert "-I -S -B -c" in command and "synthetic_ipc_worker import main" in command
            assert "shell" not in command and "torch" not in command
            assert len(allowed_sets) == 1
            return create(*args)

        self.k.UpdateProcThreadAttribute = attribute
        self.k.CreateProcessW = process

    def started(self, job):
        assert probe.entries[-1].outcome is Outcome.RESERVED
        result = launch(self, job)
        assert not self.exited(result.process, 50)
        assert not self.inheritable(job)
        assert not self.inheritable(self.parent_write) and not self.inheritable(self.parent_read)
        created.append(result.pid)
        # Duplicate from our still-suspended child for identity comparison only.
        self.k.DuplicateHandle.argtypes = [
            wi.w.HANDLE,
            wi.w.HANDLE,
            wi.w.HANDLE,
            c.c_void_p,
            wi.w.DWORD,
            wi.w.BOOL,
            wi.w.DWORD,
        ]
        self.k.DuplicateHandle.restype = wi.w.BOOL
        compare = c.WinDLL("kernelbase", use_last_error=True).CompareObjectHandles
        compare.argtypes = [wi.w.HANDLE, wi.w.HANDLE]
        compare.restype = wi.w.BOOL
        for handle in (sentinel, job):
            copied = wi.w.HANDLE()
            if self.k.DuplicateHandle(
                result.process, handle, self.k.GetCurrentProcess(), c.byref(copied), 0, False, 2
            ):
                try:
                    assert not compare(handle, copied)
                finally:
                    assert original_close(self, copied)
            else:
                assert c.get_last_error() == 6  # ERROR_INVALID_HANDLE, not an access failure.
        events.append("suspended")
        return result

    def assigned(self, job, process):
        assert events == ["suspended"]
        assign(self, job, process)
        events.append("assigned")

    def membership(self, process, job):
        value = member(self, process, job)
        if job:
            assert value and events == ["suspended", "assigned"]
            events.append("verified")
        return value

    def resumed(self, thread):
        assert events == ["suspended", "assigned", "verified"]
        assert probe.entries[-1].outcome is Outcome.DISPATCHED
        resume(self, thread)
        events.append("resumed")

    def closure(self, handle):
        closed.append(handle)
        return original_close(self, handle)

    for cls, name, method in (
        (ipc._IpcWin32, "__init__", init),
        (ipc._IpcWin32, "launch", started),
        (wi._Win32, "assign", assigned),
        (wi._Win32, "member", membership),
        (wi._Win32, "resume", resumed),
        (wi._Win32, "close", closure),
    ):

        def checked(*args, _method=method, _name=name, **kwargs):
            try:
                return _method(*args, **kwargs)
            except BaseException as error:
                failures.append((_name, error))
                raise

        monkeypatch.setattr(cls, name, checked)
    try:
        report = probe.run()
        assert not failures, failures
        clean(report)
        assert report.isolation.reason == "complete"
        assert events == ["suspended", "assigned", "verified", "resumed"] and len(created) == 1
        assert len(closed) == 7  # Four pipe, process, primary thread, dedicated job handles.
        assert sentinel not in closed and api.inheritable(sentinel)
    finally:
        assert original_close(api, sentinel)


@windows
@pytest.mark.parametrize(
    "fault,outcome,charged",
    [
        ("hang", Outcome.TIMED_OUT, 2),
        ("partial", Outcome.UNKNOWN, 2),
        ("malformed", Outcome.FAILED, 2),
        ("oversized", Outcome.FAILED, 2),
        ("exit", Outcome.UNKNOWN, 2),
        ("stale", Outcome.FAILED, 2),
        ("duplicate", Outcome.FAILED, 3),
        ("trailing", Outcome.FAILED, 5),
    ],
)
def test_actual_faults_abort_without_refund_retry_or_children(monkeypatch, fault, outcome, charged):
    monkeypatch.setattr(ipc._IpcWin32, "_fault", fault)
    probe = ipc.IsolatedCashewProbe()
    report = probe.run(limits=wi.Limits(deadline_seconds=0.6))
    clean(report)
    assert report.entries[-1].outcome is outcome
    assert len(report.entries) == sum(e.charged for e in report.entries) == charged
    assert len(report.observations) == charged - 1
    assert report.isolation.elapsed_seconds < 3.0
    with pytest.raises(wire.ProtocolError):
        probe.run()


@windows
@pytest.mark.parametrize("phase", ["before", "during"])
def test_actual_cancellation(monkeypatch, phase):
    event = threading.Event()
    write = ipc._IpcWin32.write
    if phase == "before":
        event.set()
    else:
        monkeypatch.setattr(ipc._IpcWin32, "_fault", "hang")
        count = 0

        def cancel(self, packet):
            nonlocal count
            count += 1
            write(self, packet)
            if count == 2:
                event.set()

        monkeypatch.setattr(ipc._IpcWin32, "write", cancel)
    report = ipc.IsolatedCashewProbe().run(cancel=event)
    clean(report)
    assert report.isolation.outcome is Outcome.CANCELLED
    assert report.isolation.charged == (0 if phase == "before" else 2)


@windows
@pytest.mark.parametrize("stage", ["configure", "assign", "resume"])
def test_launch_failure_preserves_cleanup_and_accounting(monkeypatch, stage):
    def fail(*args):
        raise RuntimeError("synthetic-private-error")

    monkeypatch.setattr(wi._Win32, stage, fail)
    report = ipc.IsolatedCashewProbe().run()
    clean(report)
    assert report.isolation.charged == (1 if stage == "resume" else 0)
    assert report.isolation.outcome is (
        Outcome.UNKNOWN if stage == "resume" else Outcome.NOT_STARTED
    )
    assert b"private-error" not in ipc.json_bytes(report)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b'{"x":[]}',
        b"[]",
        b"{",
        b"\xff",
        b'{"x":null}',
        b"x" * 4097,
        b"[" * 2000 + b"]" * 2000,
    ],
)
def test_invalid_json_rejected(raw):
    with pytest.raises(wire.ProtocolError):
        wire.decode(raw)


@pytest.mark.parametrize(
    "data",
    [
        wire.envelope("inspect", True),
        wire.envelope("inspect", 0),
        wire.envelope("inspect", 6),
        wire.envelope("wrong", 1),
        {**wire.envelope("inspect", 1), "schema_version": 2},
        {**wire.envelope("inspect", 1), "protocol_version": True},
        {**wire.envelope("inspect", 1), "kind": "real_model"},
        {**wire.envelope("inspect", 1), "model_execution": "run"},
        {**wire.envelope("inspect", 1), "path": "fabricated.png"},
        {**wire.envelope("inspect", 1), "url": "https://example.invalid"},
        {k: v for k, v in wire.envelope("inspect", 1).items() if k != "seq"},
    ],
)
def test_request_schema_and_order(data):
    with pytest.raises(wire.ProtocolError):
        wire.WorkerSession().inspect(data, b"")


@pytest.mark.parametrize(
    "raw",
    [b"", b"\x00", b"\x00\x00\x00\x00", struct.pack("!I", 4097), struct.pack("!I", 4) + b"ab"],
)
def test_frame_bounds_and_truncation(raw):
    with pytest.raises(wire.ProtocolError):
        wire.read_frame(io.BytesIO(raw).read, wire.MAX_JSON)


def test_frame_fragmentation_and_no_oversized_body_read():
    stream = io.BytesIO(wire.frame(b"abc", 10))
    assert wire.read_frame(lambda n: stream.read(min(n, 1)), 10) == b"abc"
    calls = []

    def read(n):
        calls.append(n)
        return struct.pack("!I", 2**32 - 1)

    with pytest.raises(wire.ProtocolError):
        wire.read_frame(read, wire.MAX_JSON)
    assert calls == [4]


def chunk(kind, body):
    return struct.pack("!I", len(body)) + kind + body + struct.pack("!I", zlib.crc32(kind + body))


def test_png_dimensions_compression_metadata_and_pixel_preservation():
    for _, pixels in wire.inputs():
        before = pixels.tobytes()
        payload = wire._png(pixels)
        np.testing.assert_array_equal(wire.decode_png(payload), pixels)
        assert not pixels.flags.writeable and pixels.tobytes() == before
    valid = wire._png(next(wire.inputs())[1])
    huge = (
        valid[:8]
        + chunk(b"IHDR", struct.pack("!IIBBBBB", 100000, 100000, 8, 2, 0, 0, 0))
        + valid[33:]
    )
    bomb = valid[:33] + chunk(b"IDAT", zlib.compress(b"0" * 10000)) + chunk(b"IEND", b"")
    metadata = valid[:33] + chunk(b"tEXt", b"private\x00note") + valid[33:]
    for payload in (
        huge,
        bomb,
        metadata,
        valid + b"trailing",
        valid[:-1],
        b"x" * 4097,
        b"\xff\xd8JPEG",
    ):
        with pytest.raises(wire.ProtocolError):
            wire.decode_png(payload)


def test_only_exact_internal_synthetic_parent_and_five_calls():
    session = wire.WorkerSession()
    for seq, (_, pixels) in enumerate(wire.inputs(), 1):
        result = session.inspect(wire.envelope("inspect", seq), wire._png(pixels))
        assert result["input_verified"] is True
    with pytest.raises(wire.ProtocolError):
        session.inspect(wire.envelope("inspect", 6), wire._png(pixels))
    with pytest.raises(wire.ProtocolError):
        wire.WorkerSession().inspect(
            wire.envelope("inspect", 1), wire._png(np.zeros((8, 10, 3), dtype=np.uint8))
        )


@pytest.mark.parametrize(
    "change",
    [
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": True},
        {"score": -1.0},
        {"mae": -1.0},
        {"rmse": float("inf")},
        {"seq": 2},
        {"above_threshold": True},
        {"threshold": 0.4},
        {"input_verified": False},
        {"status": "failed"},
        {"maximum_absolute": 0.3},
        {"unexpected": 1},
    ],
)
def test_scalar_output_validation(change):
    session = wire.WorkerSession()
    data = session.inspect(wire.envelope("inspect", 1), wire._png(next(wire.inputs())[1]))
    with pytest.raises(ValueError):
        wire.observation({**data, **change}, 1, None)


@pytest.mark.parametrize(
    "event,args",
    [
        ("open", ()),
        ("os.scandir", ()),
        ("subprocess.Popen", ()),
        ("socket.connect", ()),
        ("import", ("torch",)),
        ("import", ("factoryvision.inference.service",)),
    ],
)
def test_worker_post_import_guard(event, args):
    with pytest.raises(RuntimeError):
        worker._deny_io(event, args)


def test_no_real_admission_or_arbitrary_entry_api():
    with pytest.raises(ExecutionDenied):
        cd.prepare_development(DevelopmentAdmission())
    for arg in ("path", "pixels", "executable", "module", "callback", "authorized"):
        with pytest.raises(TypeError):
            ipc.IsolatedCashewProbe().run(**{arg: "fabricated"})
    probe = ipc.IsolatedCashewProbe()
    with probe._lock, pytest.raises(wire.ProtocolError):
        probe.run()
    assert not probe.entries


@windows
def test_safe_report_rejects_forged_metadata():
    report = ipc.IsolatedCashewProbe().run()
    for change in (
        {"reason": "private"},
        {"model_execution": "run"},
        {"kind": "real"},
        {"elapsed_seconds": float("nan")},
        {"observed_working_set": True},
    ):
        with pytest.raises(ValueError):
            ipc.json_bytes(replace(report, isolation=replace(report.isolation, **change)))


@windows
def test_cleanup_uncertainty_overrides_success_and_blocks_reuse(monkeypatch):
    original = ipc._IpcWin32.close_pipe

    def uncertain(self, handle):
        original(self, handle)
        if handle == self.parent_read:
            self.pipe_closed = False  # Actual handle closed; inject uncertain acknowledgement.

    monkeypatch.setattr(ipc._IpcWin32, "close_pipe", uncertain)
    probe = ipc.IsolatedCashewProbe()
    report = probe.run()
    assert report.isolation.reason == "cleanup_unverified"
    assert report.isolation.outcome is Outcome.UNKNOWN
    assert report.isolation.charged == 5 and len(report.observations) == 5
    assert report.isolation.exit_confirmed and report.isolation.job_empty_confirmed
    assert not report.isolation.handles_closed
    assert json.loads(ipc.json_bytes(report))["run_status"] == "unknown"
    with pytest.raises(wire.ProtocolError):
        probe.run()


@windows
@pytest.mark.parametrize("damage", ["version", "unknown", "duplicate_key", "short", "oversized"])
def test_worker_rejects_actual_bad_requests_without_second_dispatch(monkeypatch, damage):
    write = ipc._IpcWin32.write
    calls = []

    def damaged(self, packet):
        calls.append(1)
        if damage == "short":
            packet = b"\x00\x00\x00\x20{"  # Waits until parent watchdog terminates the child.
        elif damage == "oversized":
            packet = struct.pack("!I", wire.MAX_JSON + 1)
        elif damage == "duplicate_key":
            packet = wire.frame(b'{"seq":1,"seq":1}', wire.MAX_JSON)
        else:
            data = wire.envelope("inspect", 1)
            if damage == "version":
                data["protocol_version"] = 9
            else:
                data["module"] = "fabricated"
            packet = wire.encode(data)
        write(self, packet)

    monkeypatch.setattr(ipc._IpcWin32, "write", damaged)
    report = ipc.IsolatedCashewProbe().run(limits=wi.Limits(deadline_seconds=0.6))
    clean(report)
    assert len(calls) == report.isolation.charged == report.isolation.reserved == 1
    assert report.isolation.outcome is (Outcome.TIMED_OUT if damage == "short" else Outcome.FAILED)


@windows
def test_attribute_failure_has_no_insecure_fallback_or_worker(monkeypatch):
    original = ipc._IpcWin32.__init__
    starts = []

    def init(self):
        original(self)
        self.k.UpdateProcThreadAttribute = lambda *args: 0
        self.k.CreateProcessW = lambda *args: starts.append(1)

    monkeypatch.setattr(ipc._IpcWin32, "__init__", init)
    report = ipc.IsolatedCashewProbe().run()
    clean(report)
    assert not starts and report.isolation.charged == 0 and not report.isolation.resumed
