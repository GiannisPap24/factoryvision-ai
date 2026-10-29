"""Fixed synthetic wire contract; bounded bytes, no paths, executable data or persistence."""

import io
import json
import math
import struct
import zlib

import numpy as np
from PIL import Image

from factoryvision.inference.reporting import ModelInfo
from factoryvision.inference.uploads import readonly
from factoryvision.robustness.assessment import (
    FrozenThreshold,
    GridConvention,
    MapStability,
    MeasurementStatus,
    Reason,
    ThresholdRule,
    map_stability,
    score_stability,
)
from factoryvision.robustness.cashew_development import _plan, _png
from factoryvision.robustness.contracts import _unique
from factoryvision.robustness.development_boundary import (
    CompatibilityObservation,
    InterfaceSpec,
    ProbeReason,
    ProbeStatus,
    SyntheticInspectionDouble,
)
from factoryvision.robustness.perturbations import perturb
from factoryvision.robustness.reporting import _json, schedule
from factoryvision.robustness.runner import _generate

MAX_JSON = 4096
MAX_PNG = 4096
IDENTITY = "synthetic_service_interface_probe"
BASE = dict(protocol_version=1, schema_version=1, kind=IDENTITY, model_execution="not_run")


class ProtocolError(ValueError):
    def __init__(self):
        super().__init__("Invalid synthetic IPC.")


class TruncatedFrame(ProtocolError):
    """No complete acknowledgement; dispatched execution remains uncertain."""


def check(value):
    if not value:
        raise ProtocolError()


def envelope(kind, seq, **values):
    return {**BASE, "type": kind, "seq": seq, **values}


def frame(payload, limit):
    check(type(payload) is bytes and 0 < len(payload) <= limit)
    return struct.pack("!I", len(payload)) + payload


def encode(message):
    return frame(_json(message), MAX_JSON)


def read_frame(read, limit):
    def exact(count):
        parts = bytearray()
        while len(parts) < count:
            part = read(count - len(parts))
            if part == b"":
                raise TruncatedFrame()
            check(type(part) is bytes and 0 < len(part) <= count - len(parts))
            parts.extend(part)
        return bytes(parts)

    size = struct.unpack("!I", exact(4))[0]
    check(0 < size <= limit)  # Before reading or allocating the advertised body.
    return exact(size)


def decode(raw):
    try:
        check(type(raw) is bytes and 0 < len(raw) <= MAX_JSON)
        data = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError()),
        )
        check(type(data) is dict)
        # Flat scalar protocol: bounded depth and no executable/object graphs.
        check(all(type(v) in (str, int, float, bool) for v in data.values()))
        check(all(not isinstance(v, float) or math.isfinite(v) for v in data.values()))
        return data
    except (ValueError, TypeError, OverflowError, RecursionError):
        raise ProtocolError() from None


def validate(data, kind, seq, extra=()):
    check(type(data) is dict and set(data) == set(BASE) | {"type", "seq"} | set(extra))
    for key, value in {**BASE, "type": kind, "seq": seq}.items():
        check(type(data[key]) is type(value) and data[key] == value)


def spec():
    return InterfaceSpec(
        ModelInfo(
            "autoencoder",
            "cashew",
            "Convolutional autoencoder",
            "Mean RGB squared reconstruction error",
            "a" * 64,
            "b" * 64,
            0.5,
            "Historical Cashew research; frozen normal-calibration threshold only.",
        )
    )  # Fabricated metadata only; never read a catalog/artifact.


def inputs():
    plan = _plan()
    original = _generate(plan.parents[0])
    for item in schedule(plan):
        condition = item[4]
        pixels = perturb(original, condition, plan=plan, parent=item[0]).pixels
        yield condition, pixels


def request(seq, pixels):
    check(type(seq) is int and 1 <= seq <= 5)
    return encode(envelope("inspect", seq)) + frame(_png(pixels), MAX_PNG)


def decode_png(payload):
    """Only tiny RGB8 PNG, IHDR checked before Pillow and only critical chunks accepted."""
    check(type(payload) is bytes and 0 < len(payload) <= MAX_PNG)
    check(payload[:8] == b"\x89PNG\r\n\x1a\n")
    offset, chunks, compressed = 8, [], bytearray()
    while offset < len(payload):
        check(offset + 12 <= len(payload))
        size = struct.unpack("!I", payload[offset : offset + 4])[0]
        check(size <= MAX_PNG and offset + 12 + size <= len(payload))
        kind = payload[offset + 4 : offset + 8]
        body = payload[offset + 8 : offset + 8 + size]
        crc = struct.unpack("!I", payload[offset + 8 + size : offset + 12 + size])[0]
        check(zlib.crc32(kind + body) == crc)
        check(kind in (b"IHDR", b"IDAT", b"IEND"))
        if not chunks:
            check(kind == b"IHDR" and body == struct.pack("!IIBBBBB", 10, 8, 8, 2, 0, 0, 0))
        else:
            check(kind != b"IHDR" and chunks[-1] != b"IEND")
        if kind == b"IEND":
            check(size == 0)
        if kind == b"IDAT":
            compressed.extend(body)
        chunks.append(kind)
        offset += 12 + size
    check(len(chunks) >= 3 and chunks[-1] == b"IEND" and b"IDAT" in chunks)
    inflater = zlib.decompressobj()
    try:
        decoded = inflater.decompress(bytes(compressed), 249)
    except zlib.error:
        raise ProtocolError() from None
    check(
        len(decoded) == 248
        and inflater.eof
        and not inflater.unconsumed_tail
        and not inflater.unused_data
    )  # Eight filter bytes plus 240 RGB bytes.
    with Image.open(io.BytesIO(payload), formats=("PNG",)) as image:
        check(image.size == (10, 8) and image.mode == "RGB" and not image.info)
        return readonly(np.array(image, dtype=np.uint8))


def metrics(seq, reference, output, condition):
    threshold = FrozenThreshold(0.5, ThresholdRule.GREATER)
    score = score_stability(reference.score, output.score, threshold=threshold)
    spatial = map_stability(
        reference.spatial,
        output.spatial,
        perturbation=condition,
        image_size=(128, 128),
        reference_grid=GridConvention.NATIVE,
        candidate_grid=GridConvention.NATIVE,
    )
    return envelope(
        "result",
        seq,
        status="success",
        input_verified=True,
        baseline_score=reference.score,
        score=output.score,
        threshold=0.5,
        above_threshold=output.score > 0.5,
        signed=score.signed,
        absolute=score.absolute,
        relative=score.relative,
        crossing=score.crossing.value,
        mae=spatial.mae,
        rmse=spatial.rmse,
        maximum_absolute=spatial.maximum_absolute,
    )


RESULT_FIELDS = (
    "status",
    "input_verified",
    "baseline_score",
    "score",
    "threshold",
    "above_threshold",
    "signed",
    "absolute",
    "relative",
    "crossing",
    "mae",
    "rmse",
    "maximum_absolute",
)


def observation(data, seq, baseline):
    validate(data, "result", seq, RESULT_FIELDS)
    check(data["status"] == "success" and type(data["status"]) is str)
    check(data["input_verified"] is True)
    for key in (
        "baseline_score",
        "score",
        "threshold",
        "signed",
        "absolute",
        "relative",
        "mae",
        "rmse",
        "maximum_absolute",
    ):
        check(type(data[key]) is float and math.isfinite(data[key]) and abs(data[key]) <= 2)
    check(0 <= data["score"] <= 1 and 0 < data["baseline_score"] <= 1)
    check(data["baseline_score"] == (data["score"] if seq == 1 else baseline))
    check(data["threshold"] == 0.5 and type(data["above_threshold"]) is bool)
    check(data["above_threshold"] == (data["score"] > 0.5))
    score = score_stability(
        data["baseline_score"], data["score"], threshold=FrozenThreshold(0.5, ThresholdRule.GREATER)
    )
    for key in ("signed", "absolute", "relative", "crossing"):
        check(data[key] == getattr(score, key))
    spatial = MapStability(
        MeasurementStatus.MEASURED,
        Reason.NONE,
        1.0,
        data["mae"],
        data["rmse"],
        data["maximum_absolute"],
    )
    if seq <= 3:
        check(data["score"] == data["baseline_score"] and spatial.maximum_absolute == 0.0)
    return CompatibilityObservation(seq, ProbeStatus.MEASURED, ProbeReason.NONE, 1, score, spatial)


class WorkerSession:
    """Fixed five-request analytical session. Maps never cross the IPC boundary."""

    def __init__(self):
        self._inputs = iter(inputs())
        self._seq = 1
        self._baseline = None
        self._double = SyntheticInspectionDouble(spec())

    def inspect(self, data, payload):
        check(self._seq <= 5)
        validate(data, "inspect", self._seq)
        pixels = decode_png(payload)
        condition, expected = next(self._inputs)
        check(np.array_equal(pixels, expected))  # Only internally reproducible synthetic input.
        output = self._double.evaluate_output(pixels)
        if self._seq == 1:
            self._baseline = output
        result = metrics(self._seq, self._baseline, output, condition)
        observation(result, self._seq, self._baseline.score)
        self._seq += 1
        return result
