"""Workbench source-only contracts through native widgets; synthetic pixels only."""

from dataclasses import replace

import pytest
from test_app import app as app
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import PAGES, navigation_warnings
from test_launch_readiness import source_app as source_app
from test_review_app import button, save

from factoryvision.demo import build_demo


@pytest.mark.parametrize(
    ("category", "page"),
    [
        ("Cashew", "Compare"),
        ("MetalParts (experimental)", "Image inspection"),
        ("MetalParts (experimental)", "Compare"),
    ],
)
@pytest.mark.parametrize("availability", ["available", "unavailable", "mixed"])
def test_model_readiness_precedes_preview_and_setup(app, monkeypatch, category, page, availability):
    from streamlit.testing.v1 import AppTest
    from test_app import metal_entry

    from factoryvision.app import metalparts_ui

    _, engine = app
    status_calls = []

    def available(model_id):
        return availability == "available" or (
            availability == "mixed" and model_id in {"autoencoder", "feature"}
        )

    monkeypatch.setattr(
        engine,
        "statuses",
        lambda: [
            dict(
                id=key, name=spec["name"], available=available(key), reason="Synthetic availability"
            )
            for key, spec in engine.specs.items()
        ],
    )

    class Metal:
        root = engine.root

        def __init__(self, model_id):
            self.model_id = model_id

        def status(self):
            status_calls.append(self.model_id)
            return dict(available=available(self.model_id), reason="Synthetic availability")

    monkeypatch.setattr(metalparts_ui, "feature_service", lambda root: Metal("feature"))
    at = AppTest.from_function(metal_entry, args=(engine, Metal("metal")), default_timeout=10)
    at.run().selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value(page).run()
    run = "Run inspection" if category == "Cashew" else "Run MetalParts inspection"

    def check(expected):
        assert not at.exception
        notices = [c for c in at.caption if c.value.startswith("Inference unavailable")]
        assert len(notices) == int(expected)
        if expected:
            elements = list(at.main)
            preview = next(m for m in at.markdown if m.value == "**Input preview**")
            setup = next(h for h in at.subheader if h.value.startswith("Inspection "))
            assert elements.index(notices[0]) < min(elements.index(preview), elements.index(setup))

    unavailable = availability == "unavailable" or (
        availability == "mixed" and category != "Cashew"
    )
    check(unavailable)
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    at.checkbox[0].check().run()
    assert at.image and button(at, run).disabled == unavailable
    check(unavailable)
    if category != "Cashew":
        status_calls.clear()
        at.run()
        assert status_calls == ["metal"]
        if availability == "mixed":
            at.multiselect[0].set_value(["Feature-Memory Detector"]).run()
            check(False)
            assert not button(at, run).disabled
            if page == "Compare":
                at.multiselect[0].set_value(
                    ["Convolutional Autoencoder", "Feature-Memory Detector"]
                ).run()
                check(True)
                assert button(at, run).disabled
        at.multiselect[0].set_value([]).run()
        check(False)
        assert button(at, run).disabled


@pytest.fixture
def fabricated_inspection_app(app, monkeypatch):
    """Native Inspect/Compare controls with fabricated records, no model execution."""
    import numpy as np
    from streamlit.testing.v1 import AppTest
    from test_app import metal_entry

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.reporting import model_info
    from factoryvision.inference.service import InspectionResult

    _, engine = app
    calls = []

    def result(payload, model_id):
        calls.append(model_id)
        category = "metalparts" if model_id.startswith("metalparts") else "cashew"
        info = model_info(category, model_id)
        score = 0.12345678901234566
        return InspectionResult(
            model_id,
            info.name,
            info.score_label,
            score,
            info.frozen_threshold,
            None if info.frozen_threshold is None else score > info.frozen_threshold,
            np.zeros((256, 256), dtype=np.float32),
            np.zeros((256, 256, 3), dtype=np.uint8),
            None,
            None,
            0.01,
            0.01,
            0.01,
            0.03,
            info.registered_artifact_sha256,
        )

    class Metal:
        root = engine.root

        def __init__(self, model_id):
            self.model_id = model_id

        def status(self):
            return dict(available=True, reason="Fabricated result adapter")

        def inspect(self, payload):
            return result(payload, self.model_id)

    monkeypatch.setattr(engine, "inspect", result)
    monkeypatch.setattr(
        metalparts_ui, "feature_service", lambda root: Metal("metalparts_feature_memory")
    )
    at = AppTest.from_function(
        metal_entry, args=(engine, Metal("metalparts_autoencoder")), default_timeout=10
    ).run()
    return at, calls


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
@pytest.mark.parametrize("page", ["Image inspection", "Compare"])
def test_rerun_notice_precedes_run_and_preserves_explicit_review_lifecycle(
    fabricated_inspection_app, category, page
):
    from test_review_app import review_widget

    from factoryvision.app.review_ui import RERUN_NOTICE
    from factoryvision.inference import review

    at, calls = fabricated_inspection_app
    at.selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value(page).run()
    run = "Run inspection" if category == "Cashew" else "Run MetalParts inspection"
    clear = (
        "Clear image and results" if category == "Cashew" else "Clear MetalParts image and results"
    )

    def notice_before_run():
        notices = [w for w in at.warning if w.value == RERUN_NOTICE]
        assert len(notices) == 1
        elements = list(at.main)
        assert elements.index(notices[0]) < elements.index(button(at, run))

    assert not any(w.value == RERUN_NOTICE for w in at.warning)
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    at.checkbox[0].check().run()
    assert not any(w.value == RERUN_NOTICE for w in at.warning)
    button(at, run).click().run()
    # This must pass on the first result render, before any review is saved or rerun occurs.
    notice_before_run()
    assert len(calls) == 1 and not at.exception
    pending = at.session_state["fv_review_single_bundle"]
    assert all(not r.history for r in pending.records)
    saved = save(at, note="Synthetic history to export")
    notice_before_run()
    assert len(at.get("download_button")) == 2
    exports = (review.json_bytes(saved.records), review.csv_bytes(saved.records))
    at.run()
    assert at.session_state["fv_review_single_bundle"] is saved
    assert (review.json_bytes(saved.records), review.csv_bytes(saved.records)) == exports
    assert len(calls) == 1
    notice_before_run()
    button(at, run).click().run()
    fresh = at.session_state["fv_review_single_bundle"]
    assert fresh.source is not saved.source and fresh.generation != saved.generation
    assert len(calls) == 2 and not at.exception
    assert all(not r.history and not r.comment for r in fresh.records)
    assert review_widget(at, "selectbox", "Human decision").value == review.Decision.NOT_REVIEWED
    assert review_widget(at, "text_area", "Review note (optional)").value == ""
    assert b"Synthetic history to export" not in review.json_bytes(fresh.records)
    notice_before_run()
    button(at, clear).click().run()
    assert not any(w.value == RERUN_NOTICE for w in at.warning)
    assert "fv_review_single_bundle" not in at.session_state
    assert button(at, run).disabled and len(calls) == 2


@pytest.mark.parametrize(
    "score,threshold",
    [
        (0.0, 0.0),
        (0.12345678901234566, 0.0031531278509646654),
        (0.0031531278509646654, 0.0031531278509646654),
        (0.003153127850964666, 0.0031531278509646654),
        (1e-20, 1e-21),
        (1e20, 1e21),
        (0.12345678901234566, None),
    ],
)
def test_precision_disclosure_preserves_finite_values_and_compact_summary(app, score, threshold):
    """Fabricated boundary values exercise display only, without changing frozen metadata."""
    from streamlit.testing.v1 import AppTest
    from test_app import result_entry
    from test_inference import encoded

    _, engine = app
    result = replace(
        engine.inspect(encoded(), "autoencoder"),
        score=score,
        threshold=threshold,
        above_threshold=None if threshold is None else score > threshold,
    )
    spatial = result.spatial.tobytes()
    original = result.original.tobytes()
    at = AppTest.from_function(result_entry, args=(result,)).run()
    assert not at.exception
    disclosure = next(
        e for e in at.expander if e.label == "Score precision, timing and model identity"
    )
    text = next(m.value for m in disclosure.markdown if m.value.startswith("Raw score: "))
    shown_score, shown_threshold = text.removeprefix("Raw score: ").split(" · threshold: ")
    assert float(shown_score) == score
    if threshold is None:
        assert shown_threshold == "None"
        assert any("no binary decision" in w.value for w in at.warning)
    else:
        assert float(shown_threshold) == threshold
        assert (float(shown_score) > float(shown_threshold)) == result.above_threshold
    assert [(m.label, m.value) for m in at.metric[:2]] == [
        ("Anomaly score", f"{score:.8f}"),
        ("Frozen threshold", "Unavailable" if threshold is None else f"{threshold:.8f}"),
    ]
    assert any("scores are not probabilities" in c.value for c in at.caption)
    at.run()
    assert not at.exception
    assert result.score == score and result.threshold == threshold
    assert result.spatial.tobytes() == spatial and result.original.tobytes() == original


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_content_navigation_rerun_preserves_selection_results_and_saved_review(
    source_app, category
):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value("Synthetic demo").run()
    button(at, "Generate synthetic demo").click().run()
    saved = save(at, "demo", note="Synthetic navigation retention check")
    generated = at.session_state["synthetic_demo"]
    at.run()
    assert not at.exception
    assert at.main.selectbox[0].value == category
    assert at.main.radio("workspace_page").value == "Synthetic demo"
    assert at.session_state["synthetic_demo"] is generated
    assert at.session_state["fv_review_demo_bundle"] is saved
    assert len(navigation_warnings(at)) == 1
    assert not at.sidebar.radio and not at.sidebar.selectbox
    # An intentional transition still clears the same session scope.
    at.main.radio("workspace_page").set_value("Overview").run()
    assert "synthetic_demo" not in at.session_state
    assert "fv_review_demo_bundle" not in at.session_state


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_all_workspaces_keep_native_router_and_source_only_state(source_app, category):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    for page in PAGES:
        at.radio("workspace_page").set_value(page).run()
        assert not at.exception
        assert (
            len([r for r in at.radio if r.key == "workspace_page"]) == 1
            and len(at.radio("workspace_page").options) == 8
        )
        assert not at.sidebar.radio and not at.sidebar.selectbox and not at.sidebar.button
        assert len([s for s in at.main.selectbox if s.label == "Category"]) == 1
        assert at.main.radio("workspace_page").value == page
        assert not navigation_warnings(at)
        assert "synthetic_demo" not in at.session_state and "results" not in at.session_state
        if page == "Overview":
            assert button(at, "Open synthetic demo").proto.type == "primary"
            assert not at.metric


@pytest.mark.parametrize("page", ["Image inspection", "Compare"])
@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_preview_without_models_then_clear_and_navigation(source_app, category, page):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value(page).run()
    metal = category != "Cashew"
    run = "Run MetalParts inspection" if metal else "Run inspection"
    clear = "Clear MetalParts image and results" if metal else "Clear image and results"
    assert button(at, run).disabled
    assert not at.image
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert not at.exception and at.image
    assert button(at, run).disabled and navigation_warnings(at)
    at.checkbox[0].check().run()
    assert button(at, run).disabled
    button(at, clear).click().run()
    assert not at.exception and not at.image
    assert at.file_uploader[0].value is None and not at.checkbox[0].value
    assert not navigation_warnings(at)
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[1], "image/png")).run()
    at.radio("workspace_page").set_value(
        "Compare" if page == "Image inspection" else "Image inspection"
    ).run()
    assert at.file_uploader[0].value is None and not at.checkbox[0].value
    assert not navigation_warnings(at) and not at.exception


def test_comparison_view_changes_only_presentation(app, monkeypatch):
    """Existing synthetic adapters: switching views cannot request another inference."""
    from test_inference import encoded

    from factoryvision.inference import review

    at, engine = app
    calls = []
    inspect_many = engine.inspect_many

    def run(payload, selected):
        calls.append(tuple(selected))
        return inspect_many(payload, selected)

    monkeypatch.setattr(engine, "inspect_many", run)
    at.run().radio("workspace_page").set_value("Compare").run()
    at.multiselect[0].set_value(["spatial_mean", "feature_memory"]).run()
    at.file_uploader[0].set_value(("synthetic.png", encoded(), "image/png")).run()
    at.checkbox[0].check().run()
    button(at, "Run inspection").click().run()
    results = at.session_state["results"]
    review_records = at.session_state["fv_review_single_bundle"].records
    review_exports = (review.csv_bytes(review_records), review.json_bytes(review_records))
    arrays = [result.spatial.copy() for result in results]
    selector = next(r for r in at.radio if r.label == "View for all models")
    assert selector.value == "Overlay"
    for label in selector.options:
        next(r for r in at.radio if r.label == selector.label).set_value(label).run()
        assert not at.exception
        assert at.session_state["results"] is results
        assert [result.model_id for result in results] == ["spatial_mean", "feature_memory"]
        assert all(
            (result.spatial == before).all() for result, before in zip(results, arrays, strict=True)
        )
        assert len(at.image) == (2 if label == "Reconstruction / reference" else 3)
        disclosures = [m.value for m in at.markdown if m.value.startswith("Raw score: ")]
        assert len(disclosures) == len(results)
        for text, result in zip(disclosures, results, strict=True):
            score_text, threshold_text = text.removeprefix("Raw score: ").split(" · threshold: ")
            assert float(score_text) == result.score
            assert float(threshold_text) == result.threshold
        records = at.session_state["fv_review_single_bundle"].records
        assert (review.csv_bytes(records), review.json_bytes(records)) == review_exports
    assert calls == [("spatial_mean", "feature_memory")]
    at.checkbox[0].uncheck().run()
    assert "results" not in at.session_state
    assert not any(r.label == selector.label for r in at.radio)
    assert button(at, "Run inspection").disabled


@pytest.mark.parametrize("payload", [b"", b"invalid synthetic bytes"])
@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_rejected_upload_has_no_stale_preview(source_app, category, payload):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value("Image inspection").run()
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert at.image
    at.file_uploader[0].set_value(("rejected.png", payload, "image/png")).run()
    assert not at.exception and not at.image and at.error
    expected = "nonempty" if not payload else "corrupt"
    assert any(expected in error.value for error in at.error)
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))


def test_catalog_groups_keep_all_cards_and_mixed_readiness(source_app, monkeypatch):
    at, engine = source_app
    at.run().radio("workspace_page").set_value("Model Catalog").run()
    assert sum("All unavailable locally" in item.value for item in at.markdown) == 2
    cards = [b for b in at.get("download_button") if b.label == "Download model card"]
    assert len(cards) == 6
    assert len(at.selectbox(key="catalog_model_cashew").options) == 4
    monkeypatch.setattr(
        engine,
        "statuses",
        lambda: [
            {"id": key, "available": key == "autoencoder", "reason": "Synthetic status fixture"}
            for key in ("autoencoder", "spatial_mean", "feature_memory", "student_teacher")
        ],
    )
    at.run()
    assert not at.exception
    assert sum("All unavailable locally" in item.value for item in at.markdown) == 1
    assert any(item.value == "Verified files" for item in at.caption)
    assert sum(item.value == "Unavailable" for item in at.caption) == 3
    at.selectbox(key="catalog_model_cashew").set_value("spatial_mean").run()
    assert any("Real inference is disabled" in item.value for item in at.error)
    at.selectbox[0].set_value("MetalParts (experimental)").run()
    assert len(at.selectbox(key="catalog_model_metalparts").options) == 2
    assert len([b for b in at.get("download_button") if b.label == "Download model card"]) == 6


def test_demo_validation_display_and_pending_exports_preserve_contract(source_app):
    from test_review_app import save

    from factoryvision.demo import csv_bytes, json_bytes

    warning = (
        "Session-only reviews: Generating a new demo or clearing the current demo removes "
        "saved Human Review entries and unsaved edits. "
        "Save any edits and export your reviews before continuing."
    )
    at, _ = source_app
    at.run().radio("workspace_page").set_value("Synthetic demo").run()
    assert any(item.value == warning for item in at.warning)
    assert not at.image and not at.get("download_button")
    button(at, "Generate synthetic demo").click().run()
    assert any(item.value == warning for item in at.warning)
    demo, csv_export, json_export = at.session_state["synthetic_demo"]
    assert len(at.image) == 2
    assert list(at.table[0].value["Validation"]) == ["Valid", "Valid", "Rejected"]
    assert list(at.table[0].value["Score"]) == ["Not computed"] * 3
    assert all(
        row.score is row.threshold is row.research_anomaly_flag is None for row in demo.report.rows
    )
    assert csv_export == csv_bytes(build_demo().report)
    assert json_export == json_bytes(build_demo().report)
    assert any("Exports include the two pending records" in item.value for item in at.info)
    assert len(at.get("download_button")) == 4
    save(at, scope="demo", note="Synthetic geometry checked")
    assert any(item.value == warning for item in at.warning)
    assert not any("No saved review yet" in item.value for item in at.info)
    button(at, "Generate synthetic demo").click().run()
    assert all(not r.history for r in at.session_state["fv_review_demo_bundle"].records)
    assert any("No saved review yet" in item.value for item in at.info)
    assert any(item.value == warning for item in at.warning)
    save(at, scope="demo", note="Synthetic review before clearing")
    button(at, "Clear synthetic demo").click().run()
    assert any(item.value == warning for item in at.warning)
    assert not at.image and not at.get("download_button") and not at.exception
    assert "fv_review_demo_bundle" not in at.session_state
