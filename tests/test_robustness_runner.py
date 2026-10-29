"""Synthetic integration probes; no real models, datasets or evidence imports."""

import builtins
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from PIL import Image

from factoryvision.robustness import runner
from factoryvision.robustness.assessment import FrozenThreshold, GridConvention, ThresholdRule
from factoryvision.robustness.contracts import (
    Axis,
    Budgets,
    Category,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
)
from factoryvision.robustness.reporting import (
    RowStatus,
    RunReason,
    csv_bytes,
    json_bytes,
    report_bound,
)
from factoryvision.robustness.runner import AnalyticalAdapter, AnalyticalOutput, preflight, run


def plan(**changes):
    value = Plan(
        Category.CASHEW,
        (Parent(ParentID.FIRST, 10, 10),),
        (Perturbation(Kind.IDENTITY), Perturbation(Kind.BRIGHTNESS, 1.1)),
    )
    return replace(value, **changes)


def test_end_to_end_known_values_and_explicit_order():
    p = plan(parents=(Parent(ParentID.FOURTH, 10, 10), Parent(ParentID.FIRST, 10, 10)))
    first, second = run(p), run(p)
    assert first == second and json_bytes(first) == json_bytes(second)
    assert csv_bytes(first) == csv_bytes(second)
    assert [(r.case, r.condition, r.repetition) for r in first.rows] == [
        (4, 0, 1),
        (4, 0, 2),
        (4, 0, 3),
        (4, 1, 1),
        (1, 0, 1),
        (1, 0, 2),
        (1, 0, 3),
        (1, 1, 1),
    ]
    assert first.attempted_units == p.planned_inference_calls == 8
    for row in first.rows:
        assert row.status is RowStatus.MEASURED
        if row.condition == 0:
            assert row.score.absolute == row.map.mae == row.map.rmse == 0.0
    # RGB (64,128,192) -> (70,141,211), so mean intensity drift is 38/(3*255).
    assert first.rows[-1].score.signed == pytest.approx(38 / 765)
    assert first.rows[-1].map.mae == pytest.approx(38 / 765)


def test_source_is_internal_generated_once_per_parent_and_never_chained(monkeypatch):
    original_generate, original_perturb = runner._generate, runner.perturb
    generated, observed = [], []

    def generate(parent):
        value = original_generate(parent)
        generated.append(value)
        return value

    def transform(rgb, spec, **kwargs):
        observed.append(rgb)
        assert not rgb.flags.writeable
        return original_perturb(rgb, spec, **kwargs)

    monkeypatch.setattr(runner, "_generate", generate)
    monkeypatch.setattr(runner, "perturb", transform)
    run(plan())
    assert len(generated) == 1 and all(a is generated[0] for a in observed)
    np.testing.assert_array_equal(generated[0][0, 0], [64, 128, 192])
    with pytest.raises(TypeError):
        run(plan(), images=[np.zeros((10, 10, 3), dtype=np.uint8)])
    with pytest.raises(RobustnessInvalid):
        run(plan(), adapter=lambda pixels: pixels)


def test_noise_replicates_and_parent_cases_are_replayable():
    p = plan(
        parents=(Parent(ParentID.SECOND, 17, 19), Parent(ParentID.THIRD, 17, 19)),
        perturbations=(
            Perturbation(Kind.IDENTITY),
            *(Perturbation(Kind.NOISE, 5 / 255, seed=s) for s in (42, 1729, 2026)),
        ),
    )
    a, b = run(p), run(p)
    assert json_bytes(a) == json_bytes(b)
    assert len({row.map.mae for row in a.rows if row.condition > 0}) > 1


def test_fresh_process_replay_and_no_model_imports():
    script = """
import hashlib,sys
from factoryvision.robustness.contracts import *
from factoryvision.robustness.runner import run
from factoryvision.robustness.reporting import json_bytes,csv_bytes
p=Plan(Category.CASHEW,(Parent(ParentID.THIRD,7,9),),(Perturbation(Kind.IDENTITY),Perturbation(Kind.NOISE,5/255,seed=42)))
r=run(p)
assert not any(m.split('.')[0] in ('torch','torchvision','streamlit') for m in sys.modules)
assert 'factoryvision.inference.model_registry' not in sys.modules
print(hashlib.sha256(json_bytes(r)+csv_bytes(r)).hexdigest())
"""
    outputs = []
    for seed in ("11", "99"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env=dict(os.environ, PYTHONHASHSEED=seed),
        )
        assert result.stderr == ""
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


def test_exact_256_attempts_and_no_extra_condition():
    p = plan(
        parents=(Parent(ParentID.FIRST, 1, 1),),
        perturbations=(Perturbation(Kind.IDENTITY),),
        baseline_repetitions=256,
    )
    assert run(p).attempted_units == 256
    with pytest.raises(RobustnessInvalid):
        replace(p, perturbations=(*p.perturbations, Perturbation(Kind.JPEG, 95)))


def test_native_units_are_real_analytical_tile_attempts_not_claimed_model_inference(monkeypatch):
    calls = []
    original = AnalyticalAdapter.evaluate

    def evaluate(self, pixels):
        calls.append(pixels.shape)
        return original(self, pixels)

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", evaluate)
    p = plan(category=Category.METALPARTS, parents=(Parent(ParentID.FIRST, 257, 256),))
    result = run(p)
    assert result.attempted_units == 8 and calls == [(256, 256, 3)] * 8
    assert all(r.reserved == r.attempted == 2 for r in result.rows)
    assert result.rows[-1].score.signed == pytest.approx(38 / 765)
    assert result.rows[-1].map.mae == pytest.approx(38 / 765)


def test_preflight_pixel_tile_and_report_boundaries_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Rejected or cancelled plans must not allocate parents")

    monkeypatch.setattr(runner, "_generate", forbidden)
    p = plan(parents=tuple(Parent(case, 1024, 1024) for case in ParentID))
    event = Event()
    event.set()
    assert run(p, cancel=event).attempted_units == 0
    with pytest.raises(RobustnessInvalid):
        replace(p, parents=(*p.parents[:-1], Parent(ParentID.FOURTH, 1024, 1025)))
    native = plan(
        category=Category.METALPARTS,
        parents=(Parent(ParentID.FIRST, 1409, 2945),),
        perturbations=(Perturbation(Kind.IDENTITY),),
        baseline_repetitions=2,
    )
    assert preflight(native, adapter=AnalyticalAdapter()) <= 2097152
    assert sum(r.reserved for r in run(native, cancel=event).rows) == 256
    with pytest.raises(RobustnessInvalid):
        replace(native, budgets=Budgets(max_native_tiles=127))
    minimum = report_bound(p)
    assert (
        preflight(
            replace(p, budgets=Budgets(max_report_bytes=minimum)), adapter=AnalyticalAdapter()
        )
        == minimum
    )
    with pytest.raises(RobustnessInvalid):
        run(replace(p, budgets=Budgets(max_report_bytes=minimum - 1)))
    bad = plan()
    object.__setattr__(bad, "protocol_version", 1)
    with pytest.raises(RobustnessInvalid):
        run(bad)


@pytest.mark.parametrize(
    "output",
    [
        None,
        AnalyticalOutput(float("nan"), None),
        AnalyticalOutput(None, np.full((10, 10), np.inf)),
        AnalyticalOutput(True, None),
        AnalyticalOutput(None, np.zeros((1, 1))),
        AnalyticalOutput(None, np.zeros((10, 10), dtype=np.uint8)),
    ],
    ids=["wrong-type", "nan-score", "inf-map", "bool-score", "wrong-grid-shape", "uint8-map"],
)
def test_invalid_output_consumes_attempt_and_does_not_replace_first_baseline(monkeypatch, output):
    monkeypatch.setattr(AnalyticalAdapter, "evaluate", lambda self, pixels: output)
    result = run(plan())
    assert result.attempted_units == 1
    assert result.rows[0].reason is RunReason.INVALID_OUTPUT
    assert all(
        r.reason is RunReason.BASELINE_UNAVAILABLE and r.attempted == 0 for r in result.rows[1:]
    )
    assert all(r.score is r.map is None for r in result.rows)


def test_adapter_exception_on_later_native_tile_keeps_consumed_reservation(monkeypatch):
    calls = 0
    original = AnalyticalAdapter.evaluate

    def evaluate(self, pixels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fabricated-private-detail")
        return original(self, pixels)

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", evaluate)
    p = plan(category=Category.METALPARTS, parents=(Parent(ParentID.FIRST, 257, 256),))
    result = run(p)
    assert calls == result.attempted_units == 2
    assert result.rows[0].reason is RunReason.ADAPTER_ERROR
    assert sum(r.reserved for r in result.rows) == 8
    assert b"fabricated-private-detail" not in json_bytes(result)


def test_later_failure_does_not_reset_cumulative_counter_or_retry(monkeypatch):
    calls = 0
    original = AnalyticalAdapter.evaluate

    def evaluate(self, pixels):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("hidden")
        return original(self, pixels)

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", evaluate)
    result = run(plan())
    assert calls == result.attempted_units == 4
    assert result.rows[1].status is RowStatus.FAILED
    assert result.rows[2].status is result.rows[3].status is RowStatus.MEASURED


@pytest.mark.parametrize("component", ["score", "map", "both", "unsupported"])
def test_unavailable_outputs_and_geometry_compatibility_are_explicit(monkeypatch, component):
    def evaluate(self, pixels):
        return AnalyticalOutput(
            None if component in ("score", "both") else 0.5,
            None if component in ("map", "both") else np.zeros(pixels.shape[:2]),
            GridConvention.UNSUPPORTED if component == "unsupported" else GridConvention.NATIVE,
        )

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", evaluate)
    p = plan(perturbations=(Perturbation(Kind.IDENTITY), Perturbation(Kind.ROTATION, 1.0)))
    result = run(p)
    assert result.attempted_units == 4
    if component == "both":
        assert all(r.status is RowStatus.NOT_APPLICABLE for r in result.rows)
    elif component == "map":
        assert all(r.map is None for r in result.rows)
    elif component == "score":
        assert all(r.score.reason.value == "missing_score" for r in result.rows)
    else:
        assert all(r.map.reason.value == "unsupported_alignment" for r in result.rows)


def test_cancellation_is_latched_and_counted_before_next_attempt(monkeypatch):
    event = Event()
    original = AnalyticalAdapter.evaluate

    def evaluate(self, pixels):
        event.set()
        return original(self, pixels)

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", evaluate)
    result = run(plan(), cancel=event)
    assert result.attempted_units == 1
    assert (
        result.rows[0].status is RowStatus.FAILED and result.rows[0].reason is RunReason.CANCELLED
    )
    assert all(r.status is RowStatus.NOT_EVALUATED for r in result.rows[1:])
    event.clear()
    assert result.attempted_units == 1


def test_threshold_and_geometry_use_existing_assessment_semantics():
    p = plan(
        perturbations=(
            Perturbation(Kind.IDENTITY),
            Perturbation(Kind.BRIGHTNESS, 1.1),
            Perturbation(Kind.TRANSLATION, 0.01, Axis.X),
        )
    )
    result = run(p, threshold=FrozenThreshold(0.53, ThresholdRule.GREATER))
    assert result.rows[3].score.crossing.value == "up"
    assert result.rows[4].map.coverage == 0.8  # Conservative forward + inverse support.


def test_generation_failure_and_memory_only_execution(monkeypatch):
    Image.init()
    p = plan()

    def forbidden(*args, **kwargs):
        raise AssertionError("No file access")

    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "rglob", forbidden)
        assert run(p).attempted_units == 4
    monkeypatch.setattr(runner, "_generate", forbidden)
    result = run(p)
    assert result.attempted_units == 0 and result.rows[0].reason is RunReason.GENERATION_ERROR


def test_runtime_unit_deviation_cannot_exceed_preflight_reservation(monkeypatch):
    original = runner._windows
    invocations = 0

    def deviating(plan, parent):
        nonlocal invocations
        invocations += 1
        yield from original(plan, parent)
        if invocations > 1:
            yield 0, 0, 256, 256

    monkeypatch.setattr(runner, "_windows", deviating)
    result = run(plan(category=Category.METALPARTS, parents=(Parent(ParentID.FIRST, 257, 256),)))
    assert result.attempted_units == 2
    assert result.rows[0].reason is RunReason.INVALID_OUTPUT
    assert all(r.attempted == 0 for r in result.rows[1:])


def test_keyboard_interrupt_consumes_attempt_and_cancels_remaining_rows(monkeypatch):
    def interrupt(self, pixels):
        raise KeyboardInterrupt()

    monkeypatch.setattr(AnalyticalAdapter, "evaluate", interrupt)
    result = run(plan())
    assert result.attempted_units == 1 and result.rows[0].reason is RunReason.CANCELLED
    assert all(r.reason is RunReason.CANCELLED for r in result.rows)


def test_failed_transform_does_not_consume_an_adapter_unit_or_replace_anchor(monkeypatch):
    original = runner.perturb
    calls = 0

    def transform(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("hidden fixture detail")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "perturb", transform)
    result = run(plan())
    assert result.attempted_units == 3
    assert result.rows[1].reason is RunReason.TRANSFORM_ERROR
    assert result.rows[-1].score.signed == pytest.approx(38 / 765)
