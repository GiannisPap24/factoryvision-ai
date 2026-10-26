"""Synthetic batch contracts: no real datasets, weights, persistence or network."""

import io
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image
from test_inference import Tiny, encoded

from factoryvision.inference import batch
from factoryvision.inference.service import MODEL_IDS, InspectionService, ModelUnavailable
from factoryvision.inference.uploads import InputRejected


@pytest.fixture
def services(tmp_path, monkeypatch):
    cashew = InspectionService(tmp_path)
    monkeypatch.setattr(cashew, "_load", lambda key: (Tiny(key).eval(), cashew.specs[key]))
    monkeypatch.setattr(
        cashew, "statuses", lambda: [{"id": k, "available": True} for k in MODEL_IDS]
    )

    def metal(model_id, payload):
        return replace(
            cashew.inspect(payload, "autoencoder"),
            model_id=model_id,
            threshold=None,
            above_threshold=None,
        )

    ae = SimpleNamespace(
        status=lambda: {"available": True},
        inspect=lambda p: metal(batch.METAL_MODELS[0], p),
    )
    feature = SimpleNamespace(
        status=lambda: {"available": True},
        inspect=lambda p: replace(metal(batch.METAL_MODELS[1], p), reconstruction=None),
    )
    return batch.BatchService(cashew, ae, feature)


@pytest.mark.parametrize("limit", [0, 21, True, 1.5])
def test_invalid_limit(limit):
    with pytest.raises(InputRejected):
        batch.check_budget([], limit)


def test_count_and_total_budget_before_decode(monkeypatch):
    monkeypatch.setattr(batch, "decode_upload", lambda p: pytest.fail("Must reject before decode"))
    with pytest.raises(InputRejected, match="at most"):
        batch.prepare("cashew", "autoencoder", [b"x"] * 21)
    with pytest.raises(InputRejected, match="at most"):
        batch.prepare("cashew", "autoencoder", [b"x"] * 2, limit=1)
    with pytest.raises(InputRejected, match="32 MiB"):
        batch.prepare("cashew", "autoencoder", [b"x" * (8 * 1024**2)] * 5)
    with pytest.raises(InputRejected, match="Invalid upload"):
        batch.check_budget([-1])
    batch.check_budget([8 * 1024**2] * 4)


@pytest.mark.parametrize(
    "category,model,payloads",
    [
        ("pcb1", "autoencoder", []),
        ("cashew", "metalparts_autoencoder", []),
        ("metalparts", "student_teacher", []),
        ("cashew", "autoencoder", ["../private.png"]),
    ],
)
def test_category_registration_and_paths(category, model, payloads):
    with pytest.raises(InputRejected):
        batch.prepare(category, model, payloads)


def test_validation_and_ordering():
    first, second = encoded(color=50), encoded(color=60)
    payloads = [second, b"bad", first, first, b"x" * (8 * 1024**2 + 1)]
    plan = batch.prepare("cashew", "autoencoder", payloads)
    assert plan.accepted == 3
    assert [i.image_id for i in plan.items] == [f"Image {i:03d}" for i in range(1, 6)]
    assert [i.payload for i in plan.items] == payloads
    assert plan.items[1].preview is None and plan.items[4].error
    assert plan.batch_id == batch.prepare("cashew", "autoencoder", payloads).batch_id
    assert plan.batch_id != batch.identity("cashew", "autoencoder", payloads[::-1])
    assert plan.batch_id != batch.identity("cashew", "spatial_mean", payloads)
    assert batch.prepare("metalparts", batch.METAL_MODELS[0], [first]).accepted == 0
    assert batch.prepare("cashew", "autoencoder", []).accepted == 0


@pytest.mark.parametrize("model_id", [*MODEL_IDS, *batch.METAL_MODELS])
def test_numeric_adapter_reuse_and_display_semantics(services, model_id):
    category = "cashew" if model_id in MODEL_IDS else "metalparts"
    payload = encoded(size=(384, 320))
    plan = batch.prepare(category, model_id, [payload])
    if category == "cashew":
        expected = services.cashew.inspect(payload, model_id)
    else:
        engine = services.metal if model_id == batch.METAL_MODELS[0] else services.feature
        expected = engine.inspect(payload)
    events = []
    report = services.run(plan, lambda *event: events.append(event))
    row = report.rows[0]
    assert row.score == expected.score and row.threshold == expected.threshold
    assert row.artifact_sha256 == expected.artifact_sha256
    assert row.above_threshold == expected.above_threshold
    assert row.detail.ceiling == float(expected.spatial.max())
    assert (row.detail.reconstruction is None) == (expected.reconstruction is None)
    assert (row.detail.reference is None) == (expected.reference is None)
    assert report.count("inspected") == 1 and report.elapsed_ms > 0
    assert report.median_inference_ms == row.inference_ms
    assert events == [(1, 1, "inspected")]
    for data in (row.detail.original, row.detail.overlay, row.detail.heatmap):
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (384, 320)
    if category == "metalparts":
        assert row.threshold is None and "no validated threshold" in row.interpretation
    else:
        assert "research threshold" in row.interpretation
    assert "defect absence" in replace(row, above_threshold=False).interpretation


def test_individual_failure_continues_and_counts(services, monkeypatch):
    payload = encoded()
    plan = batch.prepare("cashew", "autoencoder", [payload, payload, b"bad", payload])
    original = services.cashew.inspect
    calls = []

    def inspect(data, model):
        calls.append(data)
        if len(calls) == 2:
            raise InputRejected("Synthetic changed input")
        return original(data, model)

    monkeypatch.setattr(services.cashew, "inspect", inspect)
    events = []
    report = services.run(plan, lambda *args: events.append(args))
    assert [r.status for r in report.rows] == ["inspected", "failed", "rejected", "inspected"]
    assert len(calls) == 3 and not report.fatal_error
    assert [e[0] for e in events] == [1, 2, 3, 4]
    assert report.rows[1].score is None and report.rows[1].interpretation == "Not inspected"


@pytest.mark.parametrize("error", [ModelUnavailable, RuntimeError, MemoryError])
def test_fatal_stops_remaining_and_hides_private_errors(services, monkeypatch, error):
    plan = batch.prepare("cashew", "autoencoder", [encoded(), b"bad", encoded()])
    calls = []

    def fail(*args):
        calls.append(1)
        raise error("PRIVATE FILE PATH")

    monkeypatch.setattr(services.cashew, "inspect", fail)
    report = services.run(plan)
    assert len(calls) == 1 and report.fatal_error and report.median_inference_ms is None
    assert [r.status for r in report.rows] == ["failed", "rejected", "skipped"]
    assert "PRIVATE" not in repr(report)


def test_missing_model_never_infers(services, monkeypatch):
    plan = batch.prepare("metalparts", batch.METAL_MODELS[0], [encoded(size=(256, 256))])
    monkeypatch.setattr(services.metal, "status", lambda: {"available": False})
    monkeypatch.setattr(services.metal, "inspect", lambda p: pytest.fail("No artifact"))
    report = services.run(plan)
    assert report.count("skipped") == 1 and report.fatal_error
    with pytest.raises(InputRejected):
        services.status("metalparts", "autoencoder")
    with pytest.raises(InputRejected, match="changed"):
        services.run(replace(plan, batch_id="changed"))


def test_full_maps_released_and_no_persistence(services, monkeypatch, tmp_path):
    original = services.cashew.inspect
    refs = []

    def inspect(*args):
        assert all(ref() is None for ref in refs), "Previous native map retained"
        result = original(*args)
        refs.extend([weakref.ref(result.original), weakref.ref(result.spatial)])
        return result

    monkeypatch.setattr(services.cashew, "inspect", inspect)
    before = list(tmp_path.rglob("*"))
    plan = batch.prepare("cashew", "autoencoder", [encoded(size=(1200, 900))] * 20)
    report = services.run(plan)
    assert report.count("inspected") == 20 and all(ref() is None for ref in refs)
    assert before == list(tmp_path.rglob("*"))
    for row in report.rows:
        with Image.open(io.BytesIO(row.detail.original)) as image:
            assert image.size == (384, 288)
    assert sum(len(r.detail.original) for r in report.rows) < 20 * 384 * 384 * 3


def test_incompatible_output_stops(services, monkeypatch):
    original = services.metal.inspect
    monkeypatch.setattr(services.metal, "inspect", lambda p: replace(original(p), threshold=0.5))
    report = services.run(
        batch.prepare("metalparts", batch.METAL_MODELS[0], [encoded(size=(256, 256))])
    )
    assert report.fatal_error and report.count("failed") == 1
