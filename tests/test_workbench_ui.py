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
