"""Fixed synthetic benchmark contracts/statistics. No model or subprocess entry points."""

import hashlib
import json
import math
import re
import statistics
import struct

KIND = "synthetic_cpu_pipeline_benchmark"
BASELINE = "4c3987e669360e36930ffe8d4317dd040bc0f47c"
SIDES = (128, 512)
WARMUPS = 5
SAMPLES = 30
PROCESSES = 3
THREADS = 4
STAGES = (
    "generation_png",
    "decode_validation",
    "preprocessing",
    "readonly_tensor_copy",
    "analytical",
    "display_preview",
    "scalar_json",
    "pipeline",
)
MAX_FRAME = 16384
MAX_REPORT = 131072
SCHEDULE = tuple((side, repetition) for repetition in range(1, 4) for side in SIDES)
OUTCOMES = (
    "complete",
    "cancelled",
    "timeout",
    "invalid_output",
    "worker_failed",
    "setup_failed",
    "cleanup_unverified",
)


class InvalidBenchmark(ValueError):
    def __init__(self):
        super().__init__("Invalid synthetic benchmark record.")


def require(condition):
    if not condition:
        raise InvalidBenchmark()


def number(value, maximum=10000):
    require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum)


def case(side, repetition):
    require(type(side) is int and side in SIDES)
    require(type(repetition) is int and 1 <= repetition <= PROCESSES)


def encoded(value, limit=MAX_FRAME):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise InvalidBenchmark() from None
    require(0 < len(raw) <= limit)
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result)
        result[key] = value
    return result


def read_message(read):
    # New scalar-only protocol. The frozen 5C PNG/frame contract is not used or changed.
    def exact(size):
        result = bytearray()
        while len(result) < size:
            block = read(min(4096, size - len(result)))
            require(type(block) is bytes and 0 < len(block) <= size - len(result))
            result.extend(block)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    require(0 < size <= MAX_FRAME)
    try:
        return json.loads(
            exact(size).decode(), object_pairs_hook=_pairs, parse_constant=lambda _: require(False)
        )
    except (ValueError, UnicodeError, RecursionError):
        raise InvalidBenchmark() from None


def frame(value):
    raw = encoded(value)
    return struct.pack("!I", len(raw)) + raw


def envelope(kind, side, repetition):
    case(side, repetition)
    return dict(
        kind=KIND,
        schema_version=1,
        model_execution="not_run",
        type=kind,
        side=side,
        repetition=repetition,
    )


def validate_worker(data, kind, side, repetition):
    base = envelope(kind, side, repetition)
    extra = {"import_ms", "runtime"} if kind == "ready" else {"warmups", "samples_ms"}
    if kind == "result" and type(data) is dict and "display_profile_ms" in data:
        extra.add("display_profile_ms")
        profile = data["display_profile_ms"]
        require(
            type(profile) is dict
            and set(profile)
            == {
                "geometry",
                "original_rgb_resize",
                "spatial_resize",
                "normalization",
                "palette",
                "color_interpolation",
                "rgb_conversion",
                "overlay_blend",
                "output_auxiliary",
                "preview_copy",
                "preview_resize",
                "png_encoding",
                "buffer_output",
            }
        )
        for value in profile.values():
            number(value)
    require(kind in ("ready", "result") and type(data) is dict and set(data) == set(base) | extra)
    require(all(type(data[k]) is type(v) and data[k] == v for k, v in base.items()))
    if kind == "ready":
        number(data["import_ms"])
        runtime = data["runtime"]
        keys = {"python", "torch", "torchvision", "numpy", "pillow", "intraop", "interop"}
        require(type(runtime) is dict and set(runtime) == keys)
        for key in keys - {"intraop", "interop"}:
            require(
                type(runtime[key]) is str and re.fullmatch(r"[0-9][0-9a-z.+-]{0,39}", runtime[key])
            )
        require(type(runtime["intraop"]) is int and runtime["intraop"] == THREADS)
        require(type(runtime["interop"]) is int and runtime["interop"] == 1)
    else:
        require(type(data["warmups"]) is int and data["warmups"] == WARMUPS)
        rows = data["samples_ms"]
        require(type(rows) is list and len(rows) == SAMPLES)
        for row in rows:
            require(type(row) is list and len(row) == len(STAGES))
            for value in row:
                number(value)
            require(row[-1] > 0 and sum(row[:-1]) <= row[-1] + 1e-6)
    return data


def distribution(values):
    require(type(values) in (list, tuple) and 1 <= len(values) <= SAMPLES * PROCESSES)
    for value in values:
        number(value)
    ordered = sorted(values)

    def percentile(p):
        position = (len(ordered) - 1) * p
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return dict(
        n=len(values),
        min=min(values),
        max=max(values),
        p50=percentile(0.5),
        p95=percentile(0.95),
        mean=statistics.mean(values),
        sample_stdev=statistics.stdev(values) if len(values) > 1 else None,
    )


def summarize(data):
    validate_worker(data, "result", data["side"], data["repetition"])
    rows = data["samples_ms"]
    return dict(
        stages_ms={name: distribution([row[i] for row in rows]) for i, name in enumerate(STAGES)},
        throughput_images_s=SAMPLES * 1000 / sum(row[-1] for row in rows),
    )


def scalar_payload(side, score, ceiling, preview_bytes):
    case(side, 1)
    number(score, 1)
    number(ceiling, 1)
    require(type(preview_bytes) is int and 0 < preview_bytes <= 4 * 1024**2)
    return encoded(
        dict(
            kind=KIND,
            model_execution="not_run",
            side=side,
            analytical_rgb_mean=score,
            display_ceiling=ceiling,
            preview_bytes=preview_bytes,
        )
    )


def source_identity():
    # Only our three fixed source files; no Git subprocess or repository discovery.
    from pathlib import Path

    parent = Path(__file__).resolve().parent
    return {
        role: hashlib.sha256((parent / name).read_bytes()).hexdigest()
        for role, name in {
            "contracts": "performance_baseline.py",
            "worker": "performance_worker.py",
            "supervisor": "performance_supervisor.py",
        }.items()
    }


def report_bytes(report):
    require(
        type(report) is dict
        and set(report)
        == {
            "kind",
            "schema_version",
            "model_execution",
            "baseline_commit",
            "source_sha256",
            "protocol",
            "host",
            "attempts",
        }
    )
    require(
        report["kind"] == KIND
        and report["schema_version"] == 1
        and type(report["schema_version"]) is int
        and report["model_execution"] == "not_run"
        and report["baseline_commit"] == BASELINE
    )
    protocol = dict(
        warmups=WARMUPS,
        timed=SAMPLES,
        processes_per_case=PROCESSES,
        threads=THREADS,
        process_mib=2048,
        job_mib=2048,
        deadline_seconds=10.0,
        percentile="linear_(n-1)*p",
        recovery="abort_only",
        artifact_verification="not_run",
        model_loading="not_run",
        neural_forward="not_run",
    )
    require(
        report["protocol"] == protocol
        and all(type(report["protocol"][k]) is type(v) for k, v in protocol.items())
    )
    sources = report["source_sha256"]
    require(type(sources) is dict and set(sources) == {"contracts", "worker", "supervisor"})
    require(all(type(v) is str and re.fullmatch("[0-9a-f]{64}", v) for v in sources.values()))
    host = report["host"]
    require(type(host) is dict and set(host) == {"architecture", "logical_cpus", "windows_build"})
    require(
        host["architecture"] == "AMD64"
        and type(host["logical_cpus"]) is int
        and 1 <= host["logical_cpus"] <= 1024
    )
    require(
        type(host["windows_build"]) is str and re.fullmatch(r"[0-9.]{1,32}", host["windows_build"])
    )
    attempts = report["attempts"]
    require(type(attempts) is list and 1 <= len(attempts) <= 6)
    for index, row in enumerate(attempts):
        require(
            type(row) is dict
            and set(row)
            == {
                "side",
                "repetition",
                "outcome",
                "reserved",
                "charged",
                "startup_to_ready_ms",
                "wall_ms",
                "cpu_ms",
                "peak_process_commit",
                "peak_job_commit",
                "observed_working_set",
                "assigned",
                "membership_verified",
                "resumed",
                "limits_verified",
                "allowlist_verified",
                "exit_confirmed",
                "job_empty_confirmed",
                "handles_closed",
                "io_joined",
                "ready",
                "result",
            }
        )
        case(row["side"], row["repetition"])
        require((row["side"], row["repetition"]) == SCHEDULE[index])
        require(
            row["outcome"] in OUTCOMES and type(row["reserved"]) is int and row["reserved"] == 1
        )
        require(type(row["charged"]) is int and row["charged"] in (0, 1))
        if index < len(attempts) - 1:
            require(row["outcome"] == "complete")
        for key in (
            "assigned",
            "membership_verified",
            "resumed",
            "limits_verified",
            "allowlist_verified",
            "exit_confirmed",
            "job_empty_confirmed",
            "handles_closed",
            "io_joined",
        ):
            require(type(row[key]) is bool)
        number(row["wall_ms"], 120000)
        for key in ("startup_to_ready_ms", "cpu_ms"):
            if row[key] is not None:
                number(row[key], 120000)
        for key in ("peak_process_commit", "peak_job_commit", "observed_working_set"):
            if row[key] is not None:
                # Native SIZE_T observations, not enforcement limits. Rejected reservations
                # can affect commit peaks; sampled working set is not capped by commit.
                require(type(row[key]) is int and 0 < row[key] <= 2**64 - 1)
        if row["resumed"]:
            require(row["charged"] == 1 and row["membership_verified"])
        if row["membership_verified"]:
            require(row["assigned"])
        if row["charged"]:
            require(row["limits_verified"] and row["allowlist_verified"])
        if row["ready"] is not None:
            validate_worker(row["ready"], "ready", row["side"], row["repetition"])
        if row["result"] is not None:
            validate_worker(row["result"], "result", row["side"], row["repetition"])
        if row["outcome"] == "complete":
            require(row["charged"] == 1 and row["ready"] is not None and row["result"] is not None)
            require(
                all(
                    row[k]
                    for k in (
                        "assigned",
                        "membership_verified",
                        "resumed",
                        "limits_verified",
                        "allowlist_verified",
                        "exit_confirmed",
                        "job_empty_confirmed",
                        "handles_closed",
                        "io_joined",
                    )
                )
            )
            require(
                all(
                    row[k] is not None
                    for k in (
                        "cpu_ms",
                        "startup_to_ready_ms",
                        "peak_process_commit",
                        "peak_job_commit",
                    )
                )
            )
        else:
            require(row["result"] is None)
    output = dict(report)
    output["status"] = (
        "complete" if len(attempts) == 6 and attempts[-1]["outcome"] == "complete" else "aborted"
    )
    output["failure_count"] = sum(
        r["outcome"] not in ("complete", "cancelled", "timeout") for r in attempts
    )
    output["timeout_count"] = sum(r["outcome"] == "timeout" for r in attempts)
    output["cancelled_count"] = sum(r["outcome"] == "cancelled" for r in attempts)
    output["not_started_workers"] = 6 - len(attempts)
    output["summaries"] = [
        dict(side=r["side"], repetition=r["repetition"], **summarize(r["result"]))
        for r in attempts
        if r["result"] is not None
    ]
    output["pooled"] = []
    for side in SIDES:
        rows = [r["result"] for r in attempts if r["side"] == side and r["result"] is not None]
        if len(rows) == 3:
            samples = [s for row in rows for s in row["samples_ms"]]
            output["pooled"].append(
                dict(
                    side=side,
                    throughput_images_s=len(samples) * 1000 / sum(r[-1] for r in samples),
                    stages_ms={
                        name: distribution([r[i] for r in samples]) for i, name in enumerate(STAGES)
                    },
                )
            )
    return encoded(output, MAX_REPORT)
