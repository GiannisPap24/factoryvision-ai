"""Bounded sequential orchestration over the existing, authoritative adapters."""

import hashlib
import io
import json
import math
import statistics
import time
from dataclasses import dataclass, field

from PIL import Image

from factoryvision.inference.service import MODEL_IDS, ModelUnavailable
from factoryvision.inference.uploads import MAX_BYTES, InputRejected, decode_upload
from factoryvision.inference.visualization import panels

MAX_IMAGES = 20
MAX_TOTAL_BYTES = 32 * 1024**2
DISPLAY_SIDE = 384
METAL_MODELS = ("metalparts_autoencoder", "metalparts_feature_memory")


def models(category):
    if category == "cashew":
        return MODEL_IDS
    if category == "metalparts":
        return METAL_MODELS
    raise InputRejected("Choose Cashew or experimental MetalParts.")


def check_budget(sizes, limit=MAX_IMAGES):
    if type(limit) is not int or not 1 <= limit <= MAX_IMAGES:
        raise InputRejected("Batch limit must be an integer between 1 and 20.")
    if len(sizes) > limit:
        raise InputRejected(f"Select at most {limit} images. This selection was not processed.")
    if any(type(n) is not int or n < 0 for n in sizes):
        raise InputRejected("Invalid upload size.")
    if sum(sizes) > MAX_TOTAL_BYTES:
        raise InputRejected("Total upload budget is 32 MiB. This selection was not processed.")


def identity(category, model_id, payloads, limit=MAX_IMAGES):
    if model_id not in models(category):
        raise InputRejected("The selected model does not support this category.")
    if any(not isinstance(p, bytes) for p in payloads):
        raise InputRejected("Batch inputs must be uploaded image bytes, never filesystem paths.")
    check_budget([len(p) for p in payloads], limit)
    record = [1, category, model_id, limit, [hashlib.sha256(p).hexdigest() for p in payloads]]
    return hashlib.sha256(json.dumps(record, separators=(",", ":")).encode()).hexdigest()


def png(image, side):
    image = image.copy()
    image.thumbnail((side, side), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class BatchInput:
    image_id: str
    sha256: str
    payload: bytes = field(repr=False)
    error: str = ""
    preview: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class BatchPlan:
    batch_id: str
    category: str
    model_id: str
    limit: int
    items: tuple[BatchInput, ...]

    @property
    def accepted(self):
        return sum(not item.error for item in self.items)


def prepare(category, model_id, payloads, limit=MAX_IMAGES):
    """Validate and preview one decoded image at a time; never load or run a model."""
    batch_id = identity(category, model_id, payloads, limit)
    items = []
    for index, payload in enumerate(payloads, 1):
        error, preview = "", None
        try:
            if len(payload) > MAX_BYTES:
                raise InputRejected("Image exceeds the 8 MiB per-file limit.")
            image = decode_upload(payload)
            if category == "metalparts" and min(image.size) < 256:
                raise InputRejected("MetalParts requires at least 256 pixels on each side.")
            preview = png(image, 128)
            del image
        except InputRejected as exc:
            error = str(exc)
        items.append(
            BatchInput(
                f"Image {index:03d}", hashlib.sha256(payload).hexdigest(), payload, error, preview
            )
        )
    return BatchPlan(batch_id, category, model_id, limit, tuple(items))


@dataclass(frozen=True)
class Detail:
    original: bytes = field(repr=False)
    heatmap: bytes = field(repr=False)
    overlay: bytes = field(repr=False)
    ceiling: float
    reconstruction: bytes | None = field(default=None, repr=False)
    reference: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class BatchRow:
    image_id: str
    input_sha256: str
    status: str
    error: str = ""
    model_name: str = ""
    score_label: str = ""
    score: float | None = None
    threshold: float | None = None
    above_threshold: bool | None = None
    inference_ms: float | None = None
    total_ms: float | None = None
    artifact_sha256: str = ""
    detail: Detail | None = field(default=None, repr=False)

    @property
    def interpretation(self):
        if self.status != "inspected":
            return "Not inspected"
        if self.above_threshold is None:
            return "Continuous score; no validated threshold"
        if self.above_threshold:
            return "Above frozen research threshold; review anomaly map"
        return "At/below frozen research threshold; does not establish defect absence"


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    category: str
    model_id: str
    rows: tuple[BatchRow, ...]
    elapsed_ms: float
    fatal_error: str = ""
    schema_version: int = 1

    def count(self, status):
        return sum(row.status == status for row in self.rows)

    @property
    def median_inference_ms(self):
        values = [r.inference_ms for r in self.rows if r.status == "inspected"]
        return statistics.median(values) if values else None


def compact(item, result, category, model_id):
    if (
        result.model_id != model_id
        or not math.isfinite(result.score)
        or (
            category == "metalparts"
            and (result.threshold is not None or result.above_threshold is not None)
        )
    ):
        raise ModelUnavailable("Incompatible model output.")
    view = panels(result)
    images = {
        name: None if view[name] is None else png(view[name], DISPLAY_SIDE)
        for name in ("original", "heatmap", "overlay", "reconstruction", "reference")
    }
    return BatchRow(
        item.image_id,
        item.sha256,
        "inspected",
        model_name=result.model_name,
        score_label=result.score_label,
        score=result.score,
        threshold=result.threshold,
        above_threshold=result.above_threshold,
        inference_ms=result.inference_ms,
        total_ms=result.total_ms,
        artifact_sha256=result.artifact_sha256,
        detail=Detail(ceiling=view["ceiling"], **images),
    )


class BatchService:
    """Dispatch only; preprocessing, loading and scoring stay in the original services."""

    def __init__(self, cashew, metal, feature):
        self.cashew, self.metal, self.feature = cashew, metal, feature

    def status(self, category, model_id):
        if model_id not in models(category):
            raise InputRejected("The selected model does not support this category.")
        if category == "cashew":
            return next(
                (s for s in self.cashew.statuses() if s["id"] == model_id),
                {"available": False, "reason": "Model not registered."},
            )
        return (self.metal if model_id == METAL_MODELS[0] else self.feature).status()

    def _one(self, item, category, model_id):
        if category == "cashew":
            result = self.cashew.inspect(item.payload, model_id)
        else:
            engine = self.metal if model_id == METAL_MODELS[0] else self.feature
            result = engine.inspect(item.payload)
        # This frame releases the full-resolution result before the next image starts.
        return compact(item, result, category, model_id)

    def run(self, plan, progress=None):
        expected = identity(
            plan.category, plan.model_id, [i.payload for i in plan.items], plan.limit
        )
        if expected != plan.batch_id:
            raise InputRejected("Batch inputs changed; select and validate them again.")
        start = time.perf_counter()
        rows, fatal = [], ""
        try:
            if not self.status(plan.category, plan.model_id)["available"]:
                raise ModelUnavailable("Unavailable")
        except (ModelUnavailable, RuntimeError, MemoryError):
            fatal = "Model artifacts or CPU environment unavailable. No further images inspected."
        for index, item in enumerate(plan.items, 1):
            if item.error:
                row = BatchRow(item.image_id, item.sha256, "rejected", item.error)
            elif fatal:
                row = BatchRow(item.image_id, item.sha256, "skipped", fatal)
            else:
                try:
                    row = self._one(item, plan.category, plan.model_id)
                except InputRejected as exc:
                    row = BatchRow(item.image_id, item.sha256, "failed", str(exc))
                except (ModelUnavailable, RuntimeError, MemoryError):
                    fatal = (
                        "Model artifacts or CPU execution failed. Remaining images were skipped."
                    )
                    row = BatchRow(item.image_id, item.sha256, "failed", fatal)
            rows.append(row)
            if progress is not None:
                progress(index, len(plan.items), row.status)
        return BatchReport(
            plan.batch_id,
            plan.category,
            plan.model_id,
            tuple(rows),
            (time.perf_counter() - start) * 1000,
            fatal,
        )
