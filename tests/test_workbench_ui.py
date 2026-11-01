"""Workbench source-only contracts through native widgets; synthetic pixels only."""

import pytest
from test_app import app as app
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import PAGES, navigation_warnings
from test_launch_readiness import source_app as source_app
from test_review_app import button, save

from factoryvision.demo import build_demo


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
        "saved Human Review entries. Export your reviews before continuing."
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
