"""Review lifecycle through the real app; synthetic models, offline and no checkpoint files."""

import io
from dataclasses import replace

import pytest
from test_app import app as app
from test_app import batch_open, encoded
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_reporting import snapshot

from factoryvision.app import review_ui, ui
from factoryvision.inference import review as rv
from factoryvision.inference.batch import BatchRow


def button(at, label):
    return next(b for b in at.button if b.label == label)


def save(at, scope="single", decision=rv.Decision.NEEDS_FURTHER_REVIEW, note="Check edge", index=0):
    at.selectbox(key=f"fv_review_{scope}_decision_{index}").set_value(decision)
    at.text_area(key=f"fv_review_{scope}_note_{index}").set_value(note)
    button(at, "Save human review").click().run()
    assert not at.exception
    return at.session_state[f"fv_review_{scope}_bundle"]


def single_open(app, monkeypatch):
    at, engine = app
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run().sidebar.radio[0].set_value("Image inspection").run()
    at.checkbox[0].check().run()
    button(at, "Run inspection").click().run()
    assert not at.exception and not at.error
    return at, engine


def test_single_edit_export_and_explicit_rerun(app, monkeypatch):
    at, engine = single_open(app, monkeypatch)
    evidence = at.session_state["results"]
    pending = at.session_state["fv_review_single_bundle"]
    first = save(at)
    assert first.records[0].evidence is pending.records[0].evidence
    second = save(at, decision=rv.Decision.DISAGREES_WITH_RESULT, note="Check again")
    assert len(second.records[0].history) == 2
    assert second.records[0].history[0] == first.records[0].history[0]
    assert at.session_state["results"] is evidence
    payload = rv.json_bytes(second.records)
    # Ordinary UI reruns and export rendering must not infer or train again.
    monkeypatch.setattr(engine, "inspect_many", lambda *a: pytest.fail("Unexpected inference"))
    at.run()
    assert rv.json_bytes(at.session_state["fv_review_single_bundle"].records) == payload
    assert len(at.get("download_button")) == 2
    save(at, decision=rv.Decision.NOT_REVIEWED, note="")
    assert (
        at.session_state["fv_review_single_bundle"].records[0].decision == rv.Decision.NOT_REVIEWED
    )
    button(at, "Clear image and results").click().run()
    assert "fv_review_single_bundle" not in at.session_state


@pytest.mark.parametrize("change", ["image", "model", "category", "navigation", "consent", "rerun"])
def test_single_stale_state_is_invalidated(app, monkeypatch, change):
    at, _ = single_open(app, monkeypatch)
    old = save(at)
    if change == "image":
        monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded(color=150)))
        at.run()
    elif change == "model":
        at.multiselect[0].set_value(["spatial_mean"]).run()
    elif change == "category":
        at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    elif change == "navigation":
        at.sidebar.radio[0].set_value("Compare").run()
    elif change == "consent":
        at.checkbox[0].uncheck().run()
    else:
        button(at, "Run inspection").click().run()
        fresh = at.session_state["fv_review_single_bundle"]
        assert fresh.records[0].evidence.inspection_id != old.records[0].evidence.inspection_id
        assert fresh.records[0].history == ()
        return
    assert not at.exception and "fv_review_single_bundle" not in at.session_state
    assert not at.get("download_button")


def test_invalid_comment_preserves_prior_observation_and_exports(app, monkeypatch):
    at, _ = single_open(app, monkeypatch)
    good = save(at)
    invalid = save(at, note="C:/private/image.png")
    assert at.error and at.error[0].value == str(rv.ReviewInvalid())
    assert invalid is good
    assert "private" not in rv.json_bytes(invalid.records).decode()


def test_batch_row_isolation_and_reports_keep_review(app, monkeypatch):
    at, engine = batch_open(app, monkeypatch, [encoded(), b"bad", encoded(color=150)])
    at.checkbox[0].check().run()
    button(at, "Run batch inspection").click().run()
    original = at.session_state["batch_report"]
    first = save(at, "batch")
    assert len(first.records) == 2
    assert [r.evidence.image_id for r in first.records] == ["Image 001", "Image 003"]
    assert first.records[1].history == ()
    monkeypatch.setattr(engine, "inspect", lambda *a: pytest.fail("Unexpected inference"))
    at.selectbox(key="fv_review_batch_selection").set_value(1).run()
    assert at.text_area[0].value == ""
    second = save(at, "batch", rv.Decision.AGREES_WITH_RESULT, "Aligned", 1)
    assert second.records[0] is first.records[0]
    assert at.session_state["batch_report"] is original
    at.sidebar.radio[0].set_value("Reports").run()
    assert at.session_state["fv_review_batch_bundle"] is second
    at.sidebar.radio[0].set_value("Batch inspection").run()
    assert at.session_state["fv_review_batch_bundle"] is second
    button(at, "Clear batch").click().run()
    assert "fv_review_batch_bundle" not in at.session_state


@pytest.mark.parametrize("change", ["model", "category", "image", "consent"])
def test_batch_review_invalidates_with_inputs(app, monkeypatch, change):
    images = [encoded()]
    at, _ = batch_open(app, monkeypatch, images)
    at.checkbox[0].check().run()
    button(at, "Run batch inspection").click().run()
    save(at, "batch")
    if change == "model":
        at.selectbox(key="batch_model_cashew").set_value("spatial_mean").run()
    elif change == "category":
        at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    elif change == "image":
        images[:] = [encoded(color=150)]
        at.run()
    else:
        at.checkbox[0].uncheck().run()
    assert not at.exception and "fv_review_batch_bundle" not in at.session_state


def test_source_only_demo_no_training_or_dataset_access(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_app import entry

    from factoryvision.inference.batch import BatchService
    from factoryvision.inference.service import InspectionService

    def forbidden(*args, **kwargs):
        pytest.fail("No model, training or dataset access is authorized by human review")

    monkeypatch.setattr(InspectionService, "_load", forbidden)
    monkeypatch.setattr(InspectionService, "inspect", forbidden)
    monkeypatch.setattr(BatchService, "run", forbidden)
    engine = InspectionService(tmp_path)
    at = AppTest.from_function(entry, args=(engine,), default_timeout=10).run()
    at.sidebar.radio[0].set_value("Image inspection").run()
    assert button(at, "Run inspection").disabled and not at.get("download_button")
    at.sidebar.radio[0].set_value("Synthetic demo").run()
    button(at, "Generate synthetic demo").click().run()
    first = save(at, "demo", note="Synthetic validation observation")
    assert all(r.evidence.origin == "synthetic_demo" for r in first.records)
    assert all(r.evidence.score is None for r in first.records)
    button(at, "Generate synthetic demo").click().run()
    assert at.session_state["fv_review_demo_bundle"].records[0].history == ()
    button(at, "Clear synthetic demo").click().run()
    assert "fv_review_demo_bundle" not in at.session_state
    assert not at.exception and list(tmp_path.iterdir()) == []


def isolated_review(source):
    from factoryvision.app.review_ui import render

    render("batch", source)


def test_failed_or_corrupt_result_cannot_be_reviewed():
    from streamlit.testing.v1 import AppTest

    failed = replace(snapshot(), rows=(BatchRow("Image 001", "a" * 64, "failed", "hidden"),))
    at = AppTest.from_function(isolated_review, args=(failed,)).run()
    assert at.info and not at.get("download_button") and not at.exception
    at = AppTest.from_function(isolated_review, args=(None,)).run()
    assert "fv_review_batch_bundle" not in at.session_state
    at = AppTest.from_function(isolated_review, args=("private invalid source",)).run()
    assert "private" not in at.error[0].value and not at.exception
    assert "fv_review_batch_bundle" not in at.session_state


def test_bad_cached_review_fails_closed(app, monkeypatch):
    at, _ = single_open(app, monkeypatch)
    bundle = at.session_state["fv_review_single_bundle"]
    at.session_state["fv_review_single_bundle"] = review_ui.Bundle(bundle.source, (None,))
    at.run()
    assert at.error and not at.get("download_button")
    assert "fv_review_single_bundle" not in at.session_state


def test_compare_four_cashew_reviews_are_independent(app, monkeypatch):
    from factoryvision.inference.service import MODEL_IDS

    at, _ = app
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run().sidebar.radio[0].set_value("Compare").run()
    at.multiselect[0].set_value(list(MODEL_IDS)).run()
    at.checkbox[0].check().run()
    button(at, "Run inspection").click().run()
    bundle = save(at)
    assert len(bundle.records) == 4
    assert all(not r.history for r in bundle.records[1:])
    at.selectbox(key="fv_review_single_selection").set_value(3).run()
    changed = save(at, index=3, decision=rv.Decision.AGREES_WITH_RESULT)
    assert changed.records[:3] == bundle.records[:3]
    assert len(at.metric) == 12


def test_compare_two_metalparts_reviews_keep_null_flags(app, tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_app import metal_entry
    from test_metalparts_inference import Zero, payload

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.metalparts import MetalPartsService
    from factoryvision.inference.reporting import model_info

    metal = MetalPartsService(tmp_path)
    monkeypatch.setattr(metal, "status", lambda: dict(available=True, reason="Synthetic fixture"))
    monkeypatch.setattr(metal, "_load", lambda: Zero())

    class Feature:
        def status(self):
            return metal.status()

        def inspect(self, data):
            info = model_info("metalparts", "metalparts_feature_memory")
            return replace(
                metal.inspect(data),
                model_id=info.model_id,
                artifact_sha256=info.registered_artifact_sha256,
                reconstruction=None,
            )

    monkeypatch.setattr(metalparts_ui, "feature_service", lambda root: Feature())
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(payload()[1]))
    at = AppTest.from_function(metal_entry, args=(app[1], metal), default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    at.sidebar.radio[0].set_value("Compare").run()
    at.multiselect[0].set_value(["Convolutional Autoencoder", "Feature-Memory Detector"]).run()
    at.checkbox[0].check().run()
    button(at, "Run MetalParts inspection").click().run()
    bundle = save(at)
    assert len(bundle.records) == 2 and not at.error
    at.selectbox(key="fv_review_single_selection").set_value(1).run()
    changed = save(at, index=1)
    assert changed.records[0] is bundle.records[0]
    assert all(r.evidence.threshold is r.evidence.research_flag is None for r in changed.records)
    at.multiselect[0].set_value(["Feature-Memory Detector"]).run()
    assert "fv_review_single_bundle" not in at.session_state
