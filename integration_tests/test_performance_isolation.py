"""Opt-in Windows subprocess acceptance. Fixed synthetic work only, no neural inference."""

import json
import sys
import threading

import pytest

from factoryvision import performance_baseline as b
from factoryvision import performance_supervisor as supervisor

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows Job Objects required")


def clean(row):
    assert all(
        row[k] for k in ("exit_confirmed", "job_empty_confirmed", "handles_closed", "io_joined")
    )


def test_six_independent_workers_and_actual_baseline(monkeypatch):
    launched = []
    original = supervisor._Api.launch

    def launch(api, job):
        process = original(api, job)
        launched.append((api.side, api.repetition, process.pid))
        return process

    monkeypatch.setattr(supervisor._Api, "launch", launch)
    benchmark = supervisor.Benchmark()
    report = benchmark.run()
    parsed = json.loads(b.report_bytes(report))
    # A resource/deadline failure requires owner review, never automatic limit expansion.
    assert parsed["status"] == "complete", parsed
    assert [(s, r) for s, r, _ in launched] == list(b.SCHEDULE)
    assert len({pid for _, _, pid in launched}) == 6
    for row in report["attempts"]:
        clean(row)
        assert row["charged"] == 1 and row["membership_verified"]
        assert len(row["result"]["samples_ms"]) == 30
        assert row["cpu_ms"] >= 0
        assert row["peak_process_commit"] > 0 and row["peak_job_commit"] > 0
    assert all(r["stages_ms"]["pipeline"]["n"] == 90 for r in parsed["pooled"])
    with pytest.raises(b.InvalidBenchmark):
        benchmark.run()
    # Scalar evidence only; pytest captures unless the owner asks for stdout (-s).
    print(
        json.dumps(
            {
                "pooled": parsed["pooled"],
                "workers": [
                    dict(
                        {k: v for k, v in row.items() if k not in ("result", "ready")},
                        import_ms=row["ready"]["import_ms"],
                    )
                    for row in report["attempts"]
                ],
                "runtime": report["attempts"][0]["ready"],
                "per_process": parsed["summaries"],
                "source_sha256": report["source_sha256"],
                "report_bytes": len(b.report_bytes(report)),
            },
            sort_keys=True,
        )
    )


def test_cancel_before_launch_no_charge(monkeypatch):
    def forbidden(*args):
        pytest.fail("Cancelled benchmark launched a process")

    monkeypatch.setattr(supervisor._Api, "launch", forbidden)
    cancel = threading.Event()
    cancel.set()
    report = supervisor.Benchmark().run(cancel=cancel)
    assert len(report["attempts"]) == 1
    row = report["attempts"][0]
    assert row["outcome"] == "cancelled" and row["charged"] == 0
    assert row["result"] is None and row["peak_job_commit"] is None
    clean(row)


@pytest.mark.parametrize(
    "fault", ["exit", "malformed", "oversized", "trailing", "deny_io", "deny_model"]
)
def test_bad_worker_aborts_without_retry_and_cleans(monkeypatch, fault):
    monkeypatch.setattr(supervisor._Api, "_fault", fault)
    report = supervisor.Benchmark().run()
    assert len(report["attempts"]) == 1
    row = report["attempts"][0]
    assert row["charged"] == 1 and row["outcome"] != "complete"
    assert row["result"] is None
    clean(row)


def test_hang_deadline_and_cleanup(monkeypatch):
    monkeypatch.setattr(supervisor._Api, "_fault", "hang")
    report = supervisor.Benchmark().run()
    row = report["attempts"][0]
    assert len(report["attempts"]) == 1 and row["outcome"] == "timeout"
    assert row["ready"] is not None and row["charged"] == 1
    clean(row)


def test_cancel_after_ready_aborts_one_owned_worker(monkeypatch):
    cancel = threading.Event()
    validate = b.validate_worker

    def ready_then_cancel(data, kind, side, repetition):
        value = validate(data, kind, side, repetition)
        if kind == "ready":
            cancel.set()
        return value

    monkeypatch.setattr(b, "validate_worker", ready_then_cancel)
    monkeypatch.setattr(supervisor._Api, "_fault", "hang")
    benchmark = supervisor.Benchmark()
    report = benchmark.run(cancel=cancel)
    row = report["attempts"][0]
    assert cancel.is_set() and len(report["attempts"]) == 1
    assert row["outcome"] == "cancelled" and row["charged"] == 1 and row["resumed"]
    clean(row)
    with pytest.raises(b.InvalidBenchmark):
        benchmark.run()


def test_assignment_failure_never_resumes_suspended_child(monkeypatch):
    def fail(*args):
        raise OSError("Controlled assignment failure")

    def no_resume(*args):
        pytest.fail("Unassigned worker resumed")

    monkeypatch.setattr(supervisor._Api, "assign", fail)
    monkeypatch.setattr(supervisor._Api, "resume", no_resume)
    report = supervisor.Benchmark().run()
    row = report["attempts"][0]
    assert len(report["attempts"]) == 1 and row["outcome"] == "setup_failed"
    assert row["charged"] == 0 and not row["resumed"] and row["ready"] is None
    clean(row)


def test_unverified_cleanup_blocks_reuse(monkeypatch):
    cleanup = supervisor._Api.cleanup

    def uncertain(api, job, process, assigned):
        exited, empty, closed, peak_job, peak_process = cleanup(api, job, process, assigned)
        assert exited and empty and closed  # Actual owned process is safely gone.
        return exited, False, closed, peak_job, peak_process

    monkeypatch.setattr(supervisor._Api, "cleanup", uncertain)
    monkeypatch.setattr(supervisor._Api, "_fault", "exit")
    benchmark = supervisor.Benchmark()
    report = benchmark.run()
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["outcome"] == "cleanup_unverified"
    with pytest.raises(b.InvalidBenchmark):
        benchmark.run()


def test_suspended_assignment_allowlist_and_membership_order(monkeypatch):
    events = []
    launch, assign, member, resume = (
        supervisor._Api.launch,
        supervisor._Api.assign,
        supervisor._Api.member,
        supervisor._Api.resume,
    )

    def launched(api, job):
        process = launch(api, job)
        assert api.allowlist_verified and not api.inheritable(job)
        assert api.pipe_closed and not api.exited(process.process, 50)
        events.append("suspended")
        return process

    def assigned(api, job, process):
        assert events == ["suspended"]
        assign(api, job, process)
        events.append("assigned")

    def verified(api, process, job):
        flag = member(api, process, job)
        assert flag and events == ["suspended", "assigned"]
        events.append("verified")
        return flag

    def resumed(api, thread):
        assert events == ["suspended", "assigned", "verified"]
        resume(api, thread)
        events.append("resumed")

    for name, method in (
        ("launch", launched),
        ("assign", assigned),
        ("member", verified),
        ("resume", resumed),
    ):
        monkeypatch.setattr(supervisor._Api, name, method)
    monkeypatch.setattr(supervisor._Api, "_fault", "exit")
    row = supervisor.Benchmark().run()["attempts"][0]
    assert events == ["suspended", "assigned", "verified", "resumed"]
    clean(row)


def test_fixed_display_profiling_in_existing_isolation(monkeypatch):
    from factoryvision.performance_display_profile import LABELS

    monkeypatch.setattr(supervisor._Api, "_fault", "profile_display")
    row = supervisor.Benchmark()._one(512, 1, None)
    assert row["outcome"] == "complete" and row["charged"] == 1
    clean(row)
    profile = row["result"]["display_profile_ms"]
    assert set(profile) == set(LABELS) and all(v >= 0 for v in profile.values())
    assert len(row["result"]["samples_ms"]) == 30
    print(json.dumps({"display_profile_ms": profile}, sort_keys=True))
