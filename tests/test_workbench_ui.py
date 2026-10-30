"""Workbench source-only contracts through native widgets; synthetic pixels only."""

import pytest
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import PAGES
from test_launch_readiness import source_app as source_app
from test_review_app import button

from factoryvision.demo import build_demo


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_all_workspaces_keep_native_router_and_source_only_state(source_app, category):
    at, _ = source_app
    at.run().sidebar.selectbox[0].set_value(category).run()
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception
        assert len(at.sidebar.radio) == 1 and len(at.sidebar.radio[0].options) == 8
        assert not at.sidebar.warning
        assert "synthetic_demo" not in at.session_state and "results" not in at.session_state
        if page == "Overview":
            assert button(at, "Open synthetic demo").proto.type == "primary"
            assert not at.metric


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_preview_without_models_then_clear_and_navigation(source_app, category):
    at, _ = source_app
    at.run().sidebar.selectbox[0].set_value(category).run()
    at.sidebar.radio[0].set_value("Image inspection").run()
    metal = category != "Cashew"
    run = "Run MetalParts inspection" if metal else "Run inspection"
    clear = "Clear MetalParts image and results" if metal else "Clear image and results"
    assert button(at, run).disabled
    assert not at.image
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert not at.exception and at.image
    assert button(at, run).disabled and at.sidebar.warning
    at.checkbox[0].check().run()
    assert button(at, run).disabled
    button(at, clear).click().run()
    assert not at.exception and not at.image
    assert at.file_uploader[0].value is None and not at.checkbox[0].value
    assert not at.sidebar.warning
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[1], "image/png")).run()
    at.sidebar.radio[0].set_value("Compare").run()
    assert at.file_uploader[0].value is None and not at.checkbox[0].value
    assert not at.sidebar.warning and not at.exception


@pytest.mark.parametrize("payload", [b"", b"invalid synthetic bytes"])
@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_rejected_upload_has_no_stale_preview(source_app, category, payload):
    at, _ = source_app
    at.run().sidebar.selectbox[0].set_value(category).run()
    at.sidebar.radio[0].set_value("Image inspection").run()
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert at.image
    at.file_uploader[0].set_value(("rejected.png", payload, "image/png")).run()
    assert not at.exception and not at.image and at.error
    expected = "nonempty" if not payload else "corrupt"
    assert any(expected in error.value for error in at.error)
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))


def test_catalog_groups_keep_all_cards_and_mixed_readiness(source_app, monkeypatch):
    at, engine = source_app
    at.run().sidebar.radio[0].set_value("Model Catalog").run()
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
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert len(at.selectbox(key="catalog_model_metalparts").options) == 2
    assert len([b for b in at.get("download_button") if b.label == "Download model card"]) == 6


def test_demo_validation_display_and_pending_exports_preserve_contract(source_app):
    from test_review_app import save

    from factoryvision.demo import csv_bytes, json_bytes

    at, _ = source_app
    at.run().sidebar.radio[0].set_value("Synthetic demo").run()
    assert not at.image and not at.get("download_button")
    button(at, "Generate synthetic demo").click().run()
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
    assert not any("No saved review yet" in item.value for item in at.info)
    button(at, "Generate synthetic demo").click().run()
    assert all(not r.history for r in at.session_state["fv_review_demo_bundle"].records)
    assert any("No saved review yet" in item.value for item in at.info)
    button(at, "Clear synthetic demo").click().run()
    assert not at.image and not at.get("download_button") and not at.exception
    assert "fv_review_demo_bundle" not in at.session_state
