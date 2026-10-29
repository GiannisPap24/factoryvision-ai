"""Source-only first use and navigation notices; no model execution or persistence."""

from types import SimpleNamespace

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from test_app import entry
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_review_app import button, save

from factoryvision.app.presentation import navigation_notice
from factoryvision.inference.batch import BatchService
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.service import InspectionService

PAGES = (
    "Overview",
    "Image inspection",
    "Compare",
    "Batch inspection",
    "Reports",
    "Model Catalog",
    "Research results",
    "Synthetic demo",
)


@pytest.fixture
def source_app(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Source-only readiness checks must not load or execute research models.")

    for cls in (InspectionService, MetalPartsService, MetalPartsFeatureService):
        monkeypatch.setattr(cls, "_load", forbidden)
        monkeypatch.setattr(cls, "inspect", forbidden)
    monkeypatch.setattr(BatchService, "run", forbidden)
    engine = InspectionService(tmp_path)
    return AppTest.from_function(entry, args=(engine,), default_timeout=10), engine


@pytest.mark.parametrize("ready", [False, True])
def test_first_action_tracks_verified_availability(source_app, monkeypatch, ready):
    at, engine = source_app
    if ready:
        monkeypatch.setattr(
            engine,
            "statuses",
            lambda: [
                {"id": "autoencoder", "name": "Synthetic readiness fixture", "available": True}
            ],
        )
    at.run()
    assert not at.exception and not at.sidebar.warning
    label = "Start inspection" if ready else "Open synthetic demo"
    assert button(at, label).proto.type == "primary"
    if not ready:
        assert not any(b.label in ("Start inspection", "Compare models") for b in at.button)
        assert any("no neural inference" in item.value for item in at.info)
        button(at, "Open Model Catalog").click().run()
        assert at.sidebar.radio[0].value == "Model Catalog" and len(at.table[0].value) == 6
        at.sidebar.radio[0].set_value("Overview").run()
    button(at, label).click().run()
    assert at.sidebar.radio[0].value == ("Image inspection" if ready else "Synthetic demo")
    assert not at.exception and "results" not in at.session_state


@pytest.mark.parametrize("page", PAGES)
def test_empty_source_routes_do_not_warn_of_nonexistent_work(source_app, page):
    at, _ = source_app
    at.run().sidebar.radio[0].set_value(page).run()
    assert not at.exception and not at.sidebar.warning
    assert "results" not in at.session_state and "synthetic_demo" not in at.session_state


@pytest.mark.parametrize("page", PAGES)
def test_saved_demo_review_warned_before_each_workspace_transition(source_app, page):
    at, _ = source_app
    at.run()
    button(at, "Open synthetic demo").click().run()
    button(at, "Generate synthetic demo").click().run()
    # Warning is present on the same run that first produces data, before navigation.
    assert "synthetic demo" in at.sidebar.warning[0].value
    assert "unsaved edits" in at.sidebar.warning[0].value
    saved = save(at, "demo", note="Synthetic navigation check")
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception
    if page == "Synthetic demo":
        assert at.session_state["fv_review_demo_bundle"] is saved
    else:
        assert "synthetic_demo" not in at.session_state
        assert "fv_review_demo_bundle" not in at.session_state
        assert not at.sidebar.warning


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_category_change_clears_review_demo_and_consent(source_app, category):
    at, _ = source_app
    at.run()
    at.sidebar.selectbox[0].set_value(category).run()
    button(at, "Open synthetic demo").click().run()
    button(at, "Generate synthetic demo").click().run()
    save(at, "demo")
    assert "category" in at.sidebar.warning[0].value
    target = "Cashew" if category != "Cashew" else "MetalParts (experimental)"
    at.sidebar.selectbox[0].set_value(target).run()
    assert not at.exception and not at.sidebar.warning
    assert "synthetic_demo" not in at.session_state
    assert "fv_review_demo_bundle" not in at.session_state
    assert not at.session_state["cashew_ack"] and not at.session_state["metal_ack"]


@pytest.mark.parametrize(
    "key", ["results", "metal_result", "metal_results", "upload_0", "fv_review_single_bundle"]
)
def test_notice_identifies_single_result_or_review_loss_without_reading_content(key):
    state = {key: object()}
    before = state.copy()
    notice = navigation_notice(state)
    assert "current image, results and Human Review notes" in notice
    assert "Save and export" in notice and state == before


def test_batch_notice_distinguishes_retained_pair_from_destructive_navigation():
    assert navigation_notice({"batch_plan": SimpleNamespace(items=())}) is None
    for state in ({"batch_plan": SimpleNamespace(items=(object(),))}, {"batch_report": object()}):
        notice = navigation_notice(state)
        assert "leaving Batch Inspection and Reports" in notice
        assert "retained between those two workspaces" in notice
        assert "unsaved edits may be lost" in notice
