"""Sequential synthetic-only runner. No input files, model services or persistence."""

import math
from dataclasses import dataclass
from threading import Event

import numpy as np

from factoryvision.inference.uploads import readonly
from factoryvision.robustness.assessment import (
    FrozenThreshold,
    GridConvention,
    MeasurementStatus,
    _finite,
    _map,
    map_stability,
    score_stability,
)
from factoryvision.robustness.contracts import Category, Parent, ParentID, Plan, _require
from factoryvision.robustness.perturbations import BLOCK_ROWS, _rgb, _shape, perturb
from factoryvision.robustness.reporting import (
    Observation,
    RowStatus,
    RunReason,
    SyntheticReport,
    csv_bytes,
    json_bytes,
    report_bound,
    schedule,
    units_for,
)


@dataclass(frozen=True, slots=True, eq=False)
class AnalyticalOutput:
    score: float | None
    raw_map: np.ndarray | None
    grid: GridConvention = GridConvention.NATIVE


@dataclass(frozen=True, slots=True)
class AnalyticalAdapter:
    """One call is one analytical RGB-mean image/tile evaluation, never model inference."""

    def evaluate(self, pixels: np.ndarray) -> AnalyticalOutput:
        _rgb(pixels)
        raw_map = np.mean(pixels, axis=2, dtype=np.float64) / 255.0
        return AnalyticalOutput(float(raw_map.mean()), raw_map)


def _generate(parent):
    """Four fixed original probes selected only by the non-identifying case ordinal."""
    _require(type(parent) is Parent)
    parent.__post_init__()
    h, w = parent.height, parent.width
    _shape(h, w)  # Check before allocating, even if called directly during tests.
    pixels = np.empty((h, w, 3), dtype=np.uint8)
    if parent.parent_id is ParentID.FIRST:
        pixels[:] = (64, 128, 192)
    elif parent.parent_id is ParentID.SECOND:
        x = np.rint(np.linspace(0, 255, w)).astype(np.uint16)
        y = np.rint(np.linspace(0, 255, h)).astype(np.uint16)
        pixels[:, :, 0], pixels[:, :, 1] = x[None, :], y[:, None]
        for start in range(0, h, BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, h)
            pixels[start:stop, :, 2] = (x[None, :] + y[start:stop, None]) // 2
    else:
        # Original demo colors/rectangle concept, generated directly without its file decoder.
        pixels[:] = (18, 32, 48)
        if parent.parent_id is ParentID.THIRD:
            for start in range(0, h, BLOCK_ROWS):
                stop = min(start + BLOCK_ROWS, h)
                bright = (
                    (np.arange(start, stop)[:, None] // 8 + np.arange(w)[None, :] // 8) % 2
                ) == 1
                pixels[start:stop][bright] = (45, 212, 191)
        else:
            pixels[h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)] = (
                45,
                212,
                191,
            )
    return readonly(pixels)


def _positions(size):
    positions = list(range(0, size - 256 + 1, 192))
    if positions[-1] != size - 256:
        positions.append(size - 256)
    return positions


def _windows(plan, parent):
    if plan.category is Category.CASHEW:
        yield 0, 0, parent.height, parent.width
    else:
        for y in _positions(parent.height):
            for x in _positions(parent.width):
                yield y, x, 256, 256


class _Failure(Exception):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason.value)


@dataclass
class _Counter:
    limit: int
    used: int = 0

    def charge(self):
        _require(self.used < self.limit)
        self.used += 1


def _validate_output(output, shape):
    _require(type(output) is AnalyticalOutput and type(output.grid) is GridConvention)
    if output.score is not None:
        _finite(output.score)
    raw = output.raw_map
    if raw is not None:
        _map(raw)
        if output.grid is GridConvention.NATIVE:
            _require(raw.shape == shape)
        raw = readonly(raw)
    return AnalyticalOutput(output.score, raw, output.grid)


def _evaluate(pixels, plan, parent, adapter, counter, cancel):
    reserved, before = units_for(plan, parent), counter.used
    tiled = plan.category is Category.METALPARTS
    sums = np.zeros(pixels.shape[:2], dtype=np.float64) if tiled else None
    weights = np.zeros(pixels.shape[:2], dtype=np.uint8) if tiled else None
    scores = []
    complete_maps = True
    for y, x, h, w in _windows(plan, parent):
        if cancel.is_set():
            raise _Failure(RunReason.CANCELLED)
        _require(counter.used - before < reserved)
        counter.charge()  # Reservation is consumed before invoking even a failing adapter.
        try:
            output = adapter.evaluate(pixels[y : y + h, x : x + w])
        except KeyboardInterrupt:
            raise _Failure(RunReason.CANCELLED) from None
        except Exception:
            raise _Failure(RunReason.ADAPTER_ERROR) from None
        if cancel.is_set():
            raise _Failure(RunReason.CANCELLED)
        try:
            output = _validate_output(output, (h, w))
        except Exception:
            raise _Failure(RunReason.INVALID_OUTPUT) from None
        if not tiled:
            return output
        scores.append(output.score)
        if output.raw_map is None or output.grid is not GridConvention.NATIVE:
            complete_maps = False
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                sums[y : y + h, x : x + w] += output.raw_map
            weights[y : y + h, x : x + w] += 1
        del output
    _require(counter.used - before == reserved)
    try:
        score = None if any(s is None for s in scores) else math.fsum(s / reserved for s in scores)
        raw_map = None
        if complete_maps:
            _require((weights > 0).all())
            sums /= weights
            raw_map = sums
        return _validate_output(AnalyticalOutput(score, raw_map), pixels.shape[:2])
    except Exception:
        raise _Failure(RunReason.INVALID_OUTPUT) from None


def preflight(
    plan: Plan, *, adapter: AnalyticalAdapter, threshold: FrozenThreshold | None = None
) -> int:
    """Validate the whole plan, exact adapter, independently counted units and report bound."""
    _require(type(plan) is Plan and type(adapter) is AnalyticalAdapter)
    plan.__post_init__()
    if threshold is not None:
        _require(type(threshold) is FrozenThreshold)
        threshold.__post_init__()
    for parent in plan.parents:
        _require(sum(1 for _ in _windows(plan, parent)) == units_for(plan, parent))
    bound = report_bound(plan)
    _require(bound <= plan.budgets.max_report_bytes)
    return bound


def run(
    plan: Plan,
    *,
    adapter: AnalyticalAdapter | None = None,
    threshold: FrozenThreshold | None = None,
    cancel: Event | None = None,
) -> SyntheticReport:
    """Generate one original parent at a time and compute scalar diagnostic rows only.

    Public callers cannot supply arrays, generators, source claims, paths or arbitrary adapters.
    Tests can explicitly replace the approved adapter's method in trusted test code. This
    restriction is not a sandbox against code that can monkeypatch the running process.
    """
    adapter = AnalyticalAdapter() if adapter is None else adapter
    preflight(plan, adapter=adapter, threshold=threshold)
    _require(cancel is None or type(cancel) is Event)
    cancel = Event() if cancel is None else cancel
    counter = _Counter(plan.planned_inference_calls)
    rows = []
    current = original = baseline = None
    cancelled = False
    for parent, ordinal, index, repeat, spec, units in schedule(plan):
        cancelled |= cancel.is_set()
        if cancelled:
            rows.append(
                Observation(
                    ordinal, index, repeat, units, 0, RowStatus.NOT_EVALUATED, RunReason.CANCELLED
                )
            )
            continue
        first = index == 0 and repeat == 1
        if current != parent:
            original = baseline = None
            current = parent
        if not first and baseline is None:
            rows.append(
                Observation(
                    ordinal,
                    index,
                    repeat,
                    units,
                    0,
                    RowStatus.NOT_EVALUATED,
                    RunReason.BASELINE_UNAVAILABLE,
                )
            )
            continue
        before = counter.used
        transformed = output = None
        stage = RunReason.GENERATION_ERROR if first else RunReason.TRANSFORM_ERROR
        try:
            if first:
                original = _generate(parent)
            stage = RunReason.TRANSFORM_ERROR
            transformed = perturb(original, spec, plan=plan, parent=parent)
            output = _evaluate(transformed.pixels, plan, parent, adapter, counter, cancel)
            stage = RunReason.ASSESSMENT_ERROR
            anchor = output if first else baseline
            score = score_stability(anchor.score, output.score, threshold=threshold)
            maps = None
            if anchor.raw_map is not None and output.raw_map is not None:
                maps = map_stability(
                    anchor.raw_map,
                    output.raw_map,
                    perturbation=spec,
                    image_size=(parent.width, parent.height),
                    reference_grid=anchor.grid,
                    candidate_grid=output.grid,
                    candidate_valid=transformed.valid
                    if output.grid is GridConvention.NATIVE
                    else None,
                )
            measured = score.status is MeasurementStatus.MEASURED or (
                maps is not None and maps.status is MeasurementStatus.MEASURED
            )
            rows.append(
                Observation(
                    ordinal,
                    index,
                    repeat,
                    units,
                    counter.used - before,
                    RowStatus.MEASURED if measured else RowStatus.NOT_APPLICABLE,
                    RunReason.NONE if measured else RunReason.NO_METRICS,
                    score,
                    maps,
                )
            )
            if first:
                baseline = output
        except (_Failure, KeyboardInterrupt) as exc:
            reason = exc.reason if type(exc) is _Failure else RunReason.CANCELLED
            cancelled |= reason is RunReason.CANCELLED
            status = (
                RowStatus.NOT_EVALUATED
                if cancelled and counter.used == before
                else RowStatus.FAILED
            )
            rows.append(
                Observation(ordinal, index, repeat, units, counter.used - before, status, reason)
            )
        except Exception:
            reason = (
                RunReason.INVALID_OUTPUT
                if counter.used > before and stage is not RunReason.ASSESSMENT_ERROR
                else stage
            )
            rows.append(
                Observation(
                    ordinal, index, repeat, units, counter.used - before, RowStatus.FAILED, reason
                )
            )
        finally:
            transformed = output = None
    original = baseline = None
    report = SyntheticReport(plan, tuple(rows), threshold)
    _require(report.attempted_units == counter.used)
    # Verify BOTH final byte encodings; no partial/truncated success is returned on overflow.
    json_bytes(report)
    csv_bytes(report)
    return report
