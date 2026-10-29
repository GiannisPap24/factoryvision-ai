"""Offline fixed synthetic protocol, numerical semantics and statistics."""

import copy
import io
import json
import math
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tf

from factoryvision import performance_baseline as b
from factoryvision import performance_worker as worker
from factoryvision.inference.uploads import decode_upload
from factoryvision.inference.visualization import panels


def result():
    return dict(
        b.envelope("result", 128, 1),
        warmups=5,
        samples_ms=[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 28.0] for _ in range(30)],
    )


def ready():
    return dict(
        b.envelope("ready", 128, 1),
        import_ms=10.0,
        runtime=dict(
            python="3.14.10",
            torch="2.13.0+cpu",
            torchvision="0.28.0+cpu",
            numpy="2.5.4",
            pillow="13.0.0",
            intraop=4,
            interop=1,
        ),
    )


def report():
    return dict(
        kind=b.KIND,
        schema_version=1,
        model_execution="not_run",
        baseline_commit=b.BASELINE,
        source_sha256={name: "a" * 64 for name in ("contracts", "worker", "supervisor")},
        protocol=dict(
            warmups=5,
            timed=30,
            processes_per_case=3,
            threads=4,
            process_mib=2048,
            job_mib=2048,
            deadline_seconds=10.0,
            percentile="linear_(n-1)*p",
            recovery="abort_only",
            artifact_verification="not_run",
            model_loading="not_run",
            neural_forward="not_run",
        ),
        host=dict(architecture="AMD64", logical_cpus=16, windows_build="10.0.26300"),
        attempts=[
            dict(
                side=128,
                repetition=1,
                outcome="complete",
                reserved=1,
                charged=1,
                startup_to_ready_ms=15.0,
                wall_ms=1000.0,
                cpu_ms=50.0,
                peak_process_commit=100,
                peak_job_commit=100,
                observed_working_set=None,
                assigned=True,
                membership_verified=True,
                resumed=True,
                limits_verified=True,
                allowlist_verified=True,
                exit_confirmed=True,
                job_empty_confirmed=True,
                handles_closed=True,
                io_joined=True,
                ready=ready(),
                result=result(),
            )
        ],
    )


def test_exact_configuration():
    assert (b.WARMUPS, b.SAMPLES, b.PROCESSES, b.THREADS) == (5, 30, 3, 4)
    assert b.SCHEDULE == ((128, 1), (512, 1), (128, 2), (512, 2), (128, 3), (512, 3))
    assert len(b.STAGES) == 8 and b.STAGES[-1] == "pipeline"


@pytest.mark.parametrize("side", [128, 512])
def test_owned_geometry_pixels_and_metadata(side):
    pixels, payload = worker.generate(side)
    assert np.array_equal(pixels, worker.generate(side)[0])
    assert payload == worker.generate(side)[1]
    assert pixels.shape == (side, side, 3) and pixels.dtype == np.uint8
    assert not pixels.flags.writeable
    assert tuple(pixels[0, 0]) == (18, 32, 48)
    assert tuple(pixels[side // 2, side // 2]) == (45, 212, 191)
    decoded = decode_upload(payload)
    assert np.array_equal(np.asarray(decoded), pixels)
    assert not decoded.info and not decoded.getexif()
    with Image.open(io.BytesIO(payload)) as image:
        assert not image.info and image.format == "PNG"


@pytest.mark.parametrize("side", [128, 512])
def test_pipeline_semantics_and_ownership(side, monkeypatch):
    def denied(*args, **kwargs):
        pytest.fail("Neural/model loading operation attempted")

    monkeypatch.setattr(torch, "load", denied)
    monkeypatch.setattr(torch.nn.Module, "__init__", denied)
    ticks = iter(range(0, 9_000_000, 1_000_000))
    row, observations = worker.pipeline(side, clock=lambda: next(ticks))
    assert row == [1.0] * 7 + [7.0]
    pixels, image, tensor, original, copied, analytical, raw_map, view, scalar = observations
    assert np.array_equal(pixels, original)
    assert not original.flags.writeable and not raw_map.flags.writeable
    assert not np.shares_memory(original, pixels)
    assert not np.shares_memory(raw_map, analytical.raw_map)
    expected = (
        tf.pil_to_tensor(
            tf.resize(image, [128, 128], interpolation=InterpolationMode.BILINEAR, antialias=True)
        )
        .float()
        .div(255)
        .unsqueeze(0)
    )
    assert tensor.dtype == torch.float32 and tensor.shape == (1, 3, 128, 128)
    assert torch.equal(tensor, expected) and torch.equal(copied, tensor)
    assert copied.data_ptr() != tensor.data_ptr()
    copied.zero_()
    assert torch.equal(tensor, expected)
    assert analytical.raw_map.dtype == np.float64 and raw_map.dtype == np.float32
    assert raw_map.shape == (side, side) and np.isfinite(raw_map).all()
    assert analytical.score == pytest.approx(np.mean(original, dtype=np.float64) / 255)
    other = panels(
        SimpleNamespace(original=original, spatial=raw_map, reconstruction=None, reference=None)
    )
    for key in ("original", "heatmap", "overlay"):
        assert view[key].tobytes() == other[key].tobytes()
    assert scalar == worker.pipeline(side)[1][-1]
    assert json.loads(scalar)["model_execution"] == "not_run"


def test_repetition_counts_warmups_excluded(monkeypatch):
    calls = []

    def pipeline(side):
        calls.append(side)
        return [float(len(calls))] * 8, None

    monkeypatch.setattr(worker, "pipeline", pipeline)
    rows = worker.measure(128)
    assert calls == [128] * 35
    assert len(rows) == 30 and rows[0][0] == 6 and rows[-1][0] == 35


def test_percentiles_and_sample_deviation():
    values = [0.0, 10.0, 20.0, 30.0]
    stats = b.distribution(values)
    assert stats["p50"] == 15 and stats["p95"] == pytest.approx(28.5)
    assert stats["sample_stdev"] == pytest.approx(math.sqrt(500 / 3))
    assert b.distribution([1.0])["sample_stdev"] is None
    assert b.summarize(result())["throughput_images_s"] == pytest.approx(1000 / 28)


@pytest.mark.parametrize("value", [True, "1", None, -1, float("nan"), float("inf"), 10001])
def test_reject_bad_measurements(value):
    data = result()
    data["samples_ms"][0][0] = value
    with pytest.raises(b.InvalidBenchmark):
        b.validate_worker(data, "result", 128, 1)
    with pytest.raises(b.InvalidBenchmark):
        b.distribution([value])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(warmups=4),
        lambda d: d.update(warmups=True),
        lambda d: d.update(side=512),
        lambda d: d.update(path="secret"),
        lambda d: d["samples_ms"].pop(),
        lambda d: d["samples_ms"][0].pop(),
        lambda d: d["samples_ms"][0].__setitem__(7, 0),
        lambda d: d["samples_ms"][0].__setitem__(7, 1),
        lambda d: d.update(model_execution="run"),
    ],
)
def test_strict_worker_schema(mutate):
    data = result()
    mutate(data)
    with pytest.raises(b.InvalidBenchmark):
        b.validate_worker(data, "result", 128, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["runtime"].update(hostname="private"),
        lambda d: d["runtime"].update(torch="C:/private"),
        lambda d: d["runtime"].update(intraop=8),
        lambda d: d["runtime"].update(interop=True),
        lambda d: d.update(import_ms=-1),
    ],
)
def test_safe_runtime_fields(mutate):
    data = ready()
    mutate(data)
    with pytest.raises(b.InvalidBenchmark):
        b.validate_worker(data, "ready", 128, 1)


def test_fragmented_bounded_scalar_transport():
    stream = io.BytesIO(b.frame(result()))
    assert b.read_message(lambda n: stream.read(min(n, 2))) == result()
    with pytest.raises(b.InvalidBenchmark):
        b.encoded({"x": "x" * b.MAX_FRAME})
    with pytest.raises(b.InvalidBenchmark):
        b.read_message(io.BytesIO(struct.pack("!I", b.MAX_FRAME + 1)).read)


@pytest.mark.parametrize("payload", [b"", b"{", b'{"a":1,"a":2}', b'{"x":NaN}', b"\xff"])
def test_bad_transport(payload):
    with pytest.raises(b.InvalidBenchmark):
        b.read_message(io.BytesIO(struct.pack("!I", len(payload)) + payload).read)


def test_report_is_bounded_and_has_no_input_payloads():
    data = report()
    raw = b.report_bytes(data)
    assert raw == b.report_bytes(copy.deepcopy(data)) and len(raw) < b.MAX_REPORT
    parsed = json.loads(raw)
    assert parsed["status"] == "aborted" and parsed["not_started_workers"] == 5
    assert parsed["attempts"][0]["observed_working_set"] is None
    assert all(term not in raw for term in (b"pixels", b"data:image", b"traceback", b"hostname"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(secret="no"),
        lambda d: d.update(schema_version=True),
        lambda d: d["protocol"].update(deadline_seconds=11.0),
        lambda d: d["attempts"][0].update(charged=0),
        lambda d: d["attempts"][0].update(handles_closed=False),
        lambda d: d["attempts"][0].update(peak_job_commit=-1),
        lambda d: d["attempts"][0].update(peak_process_commit=0),
        lambda d: d["attempts"][0].update(cpu_ms=float("nan")),
        lambda d: d["attempts"][0].update(outcome="failed/private/path"),
        lambda d: d["host"].update(windows_build="user@private"),
    ],
)
def test_report_rejects_untrusted_metadata(mutate):
    data = report()
    mutate(data)
    with pytest.raises(b.InvalidBenchmark):
        b.report_bytes(data)


@pytest.mark.parametrize(
    "side,repeat", [(True, 1), (127, 1), (4096, 1), (128, 0), (128, 4), (128, True)]
)
def test_no_arbitrary_workload(side, repeat):
    with pytest.raises(b.InvalidBenchmark):
        b.case(side, repeat)


def test_worker_read_guard_denies_writes_network_and_unapproved_reads():
    audit = worker._guard_reads()
    for event, args in (
        ("open", ("private.pth", "rb", 0)),
        ("open", ("anything", "wb", 1)),
        ("socket.connect", ()),
        ("subprocess.Popen", ()),
        ("pickle.find_class", ()),
        ("os.scandir", ()),
    ):
        with pytest.raises(RuntimeError, match="denied"):
            audit(event, args)


def test_counter_observations_are_not_memory_caps():
    data = report()
    data["attempts"][0]["observed_working_set"] = 3 * 1024**3
    data["attempts"][0]["peak_job_commit"] = 3 * 1024**3
    assert json.loads(b.report_bytes(data))["attempts"][0]["peak_job_commit"] == 3 * 1024**3


def test_source_identity_has_roles_without_paths_or_filenames():
    values = b.source_identity()
    assert set(values) == {"contracts", "worker", "supervisor"}
    assert len(set(values.values())) == 3
    assert all(len(value) == 64 for value in values.values())


def test_pooled_statistics_use_samples_not_means_of_percentiles():
    data = report()
    first = data["attempts"][0]
    data["attempts"] = []
    for i, (side, repetition) in enumerate(b.SCHEDULE):
        row = copy.deepcopy(first)
        row.update(side=side, repetition=repetition)
        for key in ("ready", "result"):
            row[key].update(side=side, repetition=repetition)
        row["result"]["samples_ms"] = [[float(i + 1)] * 7 + [float(7 * (i + 1))]] * 30
        data["attempts"].append(row)
    parsed = json.loads(b.report_bytes(data))
    assert parsed["status"] == "complete"
    assert parsed["failure_count"] == parsed["timeout_count"] == 0
    assert parsed["pooled"][0]["stages_ms"]["pipeline"]["p50"] == 21
    assert parsed["pooled"][0]["stages_ms"]["pipeline"]["n"] == 90
    assert parsed["pooled"][0]["throughput_images_s"] == pytest.approx(1000 / 21)
    raw = b.report_bytes(data)
    for term in (b".py", b"C:\\", b"@", b"hostname", b"environment", b"pixels", b"traceback"):
        assert term not in raw


def test_size_boundaries_exact_and_overflow():
    # JSON string adds two quote bytes, with no escaping for this alphabet.
    assert len(b.encoded("x" * (b.MAX_FRAME - 2))) == b.MAX_FRAME
    with pytest.raises(b.InvalidBenchmark):
        b.encoded("x" * (b.MAX_FRAME - 1))
    assert len(b.encoded("x" * (b.MAX_REPORT - 2), b.MAX_REPORT)) == b.MAX_REPORT
    with pytest.raises(b.InvalidBenchmark):
        b.encoded("x" * (b.MAX_REPORT - 1), b.MAX_REPORT)


def test_only_public_decoder_resources_allowed_readonly():
    from pathlib import Path

    audit = worker._guard_reads()
    root = Path(worker.__file__).resolve().parent / "inference"
    for name in ("protected_inputs.json", "metalparts_protected_inputs.json"):
        path = str(root / name)
        audit("open", (path, "r", 0))
        with pytest.raises(RuntimeError):
            audit("open", (path, "w", 1))


def test_report_preserves_unknown_failed_measurements():
    data = report()
    row = data["attempts"][0]
    row.update(outcome="worker_failed", result=None, cpu_ms=None, observed_working_set=None)
    parsed = json.loads(b.report_bytes(data))
    assert parsed["failure_count"] == 1 and parsed["pooled"] == []
    assert parsed["attempts"][0]["cpu_ms"] is None
