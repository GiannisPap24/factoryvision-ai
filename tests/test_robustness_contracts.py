"""Offline protocol admission and exact schema tests; no image or model execution."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from factoryvision.robustness.contracts import (
    Applicability,
    Axis,
    Budgets,
    Category,
    Diagnostic,
    DiagnosticStatus,
    Kind,
    Parent,
    ParentID,
    Perturbation,
    Plan,
    RobustnessInvalid,
    from_json,
    json_bytes,
)


def plan(**changes):
    return replace(
        Plan(Category.CASHEW, (Parent(ParentID.FIRST, 256, 256),), (Perturbation(Kind.IDENTITY),)),
        **changes,
    )


def candidates():
    result = [Perturbation(Kind.IDENTITY)]
    for kind, severities in (
        (Kind.BRIGHTNESS, (0.90, 1.10, 0.80, 1.20, 0.65, 1.35)),
        (Kind.CONTRAST, (0.90, 1.10, 0.80, 1.20, 0.60, 1.40)),
        (Kind.BLUR, (0.5, 1.0, 2.0)),
        (Kind.JPEG, (95, 80, 60)),
        (Kind.ROTATION, (-1.0, 1.0, -3.0, 3.0, -5.0, 5.0)),
    ):
        result.extend(Perturbation(kind, severity) for severity in severities)
    result.extend(
        Perturbation(Kind.NOISE, sigma / 255, seed=seed)
        for sigma in (2, 5, 10)
        for seed in (42, 1729, 2026)
    )
    result.extend(
        Perturbation(Kind.TRANSLATION, fraction, axis=axis)
        for fraction in (-0.01, 0.01, -0.02, 0.02, -0.04, 0.04)
        for axis in Axis
    )
    return tuple(result)


def test_all_frozen_candidates_and_pixel_budget_are_admitted_explicitly():
    record = plan(
        parents=tuple(Parent(parent_id, 1024, 1024) for parent_id in ParentID),
        perturbations=candidates(),
    )
    assert len(record.perturbations) == 46
    assert record.planned_inference_calls == 192
    assert sum(p.width * p.height for p in record.parents) == 4194304
    assert from_json(json_bytes(record), Plan) == record
    # The sum, not each parent's individual size, exceeds the budget here.
    with pytest.raises(RobustnessInvalid):
        replace(record, parents=(Parent(ParentID.FIRST, 1024, 1025), *record.parents[1:]))


def test_pixels_at_limit_and_one_pixel_over():
    record = plan(parents=(Parent(ParentID.FIRST, 2048, 2048),))
    assert from_json(json_bytes(record), Plan) == record
    with pytest.raises(RobustnessInvalid):
        replace(record, parents=(*record.parents, Parent(ParentID.SECOND, 1, 1)))


def test_inference_limit_includes_baselines_and_noise_replicates():
    record = plan(baseline_repetitions=256)
    assert record.planned_inference_calls == 256
    with pytest.raises(RobustnessInvalid):
        replace(
            record,
            perturbations=(*record.perturbations, Perturbation(Kind.NOISE, 2 / 255, seed=42)),
        )
    noise = tuple(Perturbation(Kind.NOISE, 2 / 255, seed=s) for s in (42, 1729, 2026))
    record = plan(perturbations=(Perturbation(Kind.IDENTITY), *noise))
    assert record.planned_inference_calls == 6
    with pytest.raises(RobustnessInvalid):
        replace(record, budgets=Budgets(max_inference_calls=5))


def test_native_tile_limit_and_every_tile_counts_as_an_inference_unit():
    # 8 * 16 edge-covering tiles; within the aggregate source-pixel limit.
    record = plan(
        category=Category.METALPARTS,
        parents=(Parent(ParentID.FIRST, 1409, 2945),),
        baseline_repetitions=2,
    )
    assert record.planned_inference_calls == 256
    for changes in (
        {"budgets": Budgets(max_native_tiles=127)},
        {"baseline_repetitions": 3},
        {"parents": (Parent(ParentID.FIRST, 255, 256),)},
    ):
        with pytest.raises(RobustnessInvalid):
            replace(record, **changes)
    boundary = plan(
        category=Category.METALPARTS,
        budgets=Budgets(max_native_tiles=1),
    )
    assert boundary.planned_inference_calls == 3
    with pytest.raises(RobustnessInvalid):
        replace(boundary, parents=(Parent(ParentID.FIRST, 257, 256),))


@pytest.mark.parametrize(
    "field,maximum",
    [
        ("max_parents", 4),
        ("max_inference_calls", 256),
        ("max_source_pixels", 4194304),
        ("max_native_tiles", 128),
        ("max_report_bytes", 2097152),
    ],
)
def test_budget_types_and_boundaries(field, maximum):
    for valid in (1, maximum):
        assert getattr(Budgets(**{field: valid}), field) == valid
    for invalid in (0, -1, maximum + 1, True, 1.0, "1", None):
        with pytest.raises(RobustnessInvalid):
            Budgets(**{field: invalid})


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "cashew"},
        {"category": "pcb1"},
        {"category": "pcb2"},
        {"parents": []},
        {"parents": ()},
        {"perturbations": []},
        {"perturbations": ()},
        {"baseline_repetitions": True},
        {"baseline_repetitions": 0},
        {"baseline_repetitions": 257},
        {"baseline_repetitions": 3.0},
        {"budgets": {}},
        {"protocol_id": "other"},
        {"protocol_version": True},
        {"protocol_version": 2},
        {"schema_version": True},
        {"schema_version": 2},
        {"provenance": "owned_synthetic"},
        {"classification": "software_diagnostic"},
    ],
)
def test_plan_rejects_wrong_types_and_unsupported_scope(changes):
    with pytest.raises(RobustnessInvalid):
        plan(**changes)


def test_plan_rejects_duplicates_missing_identity_and_reduced_parent_budget():
    record = plan()
    for changes in (
        {"parents": record.parents * 2},
        {"perturbations": record.perturbations * 2},
        {"perturbations": (Perturbation(Kind.BLUR, 0.5),)},
        {"perturbations": (Perturbation(Kind.BLUR, 0.5), Perturbation(Kind.IDENTITY))},
        {
            "parents": (*record.parents, Parent(ParentID.SECOND, 256, 256)),
            "budgets": Budgets(max_parents=1),
        },
        {"parents": ("case-001",)},
        {"perturbations": ("identity",)},
    ):
        with pytest.raises(RobustnessInvalid):
            replace(record, **changes)


@pytest.mark.parametrize("dimension", [True, 1.0, "256", 0, -1, 4097, None])
def test_parent_dimensions_are_bounded_integer_pixels(dimension):
    for width, height in ((dimension, 256), (256, dimension)):
        with pytest.raises(RobustnessInvalid):
            Parent(ParentID.FIRST, width, height)


@pytest.mark.parametrize(
    "arguments",
    [
        ("flip",),
        (Kind.IDENTITY, 1.0),
        (Kind.BRIGHTNESS, 1.0),
        (Kind.CONTRAST, 0.65),
        (Kind.BLUR, 1),
        (Kind.BLUR, True),
        (Kind.NOISE, 2 / 255),
        (Kind.NOISE, 2 / 255, None, True),
        (Kind.NOISE, 2 / 255, None, 7),
        (Kind.NOISE, 0.02, None, 42),
        (Kind.JPEG, 95.0),
        (Kind.JPEG, True),
        (Kind.JPEG, 100),
        (Kind.JPEG, 10**1000),
        (Kind.ROTATION, 90.0),
        (Kind.ROTATION, 1.0, None, 42),
        (Kind.TRANSLATION, 0.01),
        (Kind.TRANSLATION, 1.0, Axis.X),
        (Kind.TRANSLATION, 0.01, "x"),
        (Kind.BLUR, 0.5, Axis.X),
        (Kind.BLUR, float("nan")),
        (Kind.BLUR, float("inf")),
        (Kind.BLUR, -float("inf")),
    ],
)
def test_unsupported_transform_or_severity_is_rejected(arguments):
    with pytest.raises(RobustnessInvalid):
        Perturbation(*arguments)


def test_diagnostic_applicability_state_matrix_and_no_success_claim():
    for applicability in Applicability:
        for status in DiagnosticStatus:
            valid = (applicability is Applicability.NOT_APPLICABLE) == (
                status is DiagnosticStatus.NOT_APPLICABLE
            )
            args = (ParentID.FIRST, Perturbation(Kind.IDENTITY), applicability, status)
            if valid:
                record = Diagnostic(*args)
                assert from_json(json_bytes(record), Diagnostic) == record
            else:
                with pytest.raises(RobustnessInvalid):
                    Diagnostic(*args)
    record = Diagnostic(ParentID.FIRST, Perturbation(Kind.IDENTITY), Applicability.APPLICABLE)
    for field, value in (
        ("status", "success"),
        ("status", "HISTORICAL"),
        ("status", "EXPLORATORY"),
        ("provenance", "real_dataset"),
        ("classification", "ground_truth"),
        ("applicability", "applicable"),
    ):
        with pytest.raises(RobustnessInvalid):
            replace(record, **{field: value})


def test_exact_schema_missing_unknown_and_duplicate_keys_at_each_level():
    payload = json_bytes(plan())
    for location in ((), ("budgets",), ("parents", 0), ("perturbations", 0)):
        for operation in ("extra", "missing"):
            data = json.loads(payload)
            target = data
            for key in location:
                target = target[key]
            if operation == "extra":
                target["unexpected"] = None
            else:
                target.pop(next(iter(target)))
            with pytest.raises(RobustnessInvalid):
                from_json(json.dumps(data).encode(), Plan)
    for raw in (
        payload.replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1'),
        payload.replace(b'"width":256', b'"width":256,"width":256'),
    ):
        with pytest.raises(RobustnessInvalid):
            from_json(raw, Plan)


def test_immutable_canonical_round_trip_without_mutable_container_aliases():
    record = plan(perturbations=candidates())
    payload = json_bytes(record)
    data = json.loads(payload)
    reordered = json.dumps(data, indent=4, sort_keys=False).encode()
    assert json_bytes(from_json(reordered, Plan)) == payload == json_bytes(record)
    data["parents"][0]["width"] = 1
    assert record.parents[0].width == 256
    for obj, field, value in (
        (record, "schema_version", 2),
        (record.parents[0], "width", 1),
        (record.budgets, "max_parents", 1),
        (record.perturbations[0], "seed", 42),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, value)
    assert not hasattr(record, "__dict__")
    assert json_bytes(record, max_bytes=len(payload)) == payload
    with pytest.raises(RobustnessInvalid):
        json_bytes(record, max_bytes=len(payload) - 1)


def test_export_revalidates_even_forced_nested_mutation():
    record = plan(budgets=Budgets())
    object.__setattr__(record.budgets, "max_source_pixels", True)
    with pytest.raises(RobustnessInvalid):
        json_bytes(record)
