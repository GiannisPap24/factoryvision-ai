"""Native shell order and retained session safeguards; not browser focus verification."""

import pytest
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import PAGES, navigation_warnings
from test_launch_readiness import source_app as source_app
from test_review_app import button

from factoryvision.demo import build_demo


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
@pytest.mark.parametrize("page", ["Image inspection", "Compare"])
def test_empty_inspection_setup_shortcut_precedes_single_uploader(source_app, category, page):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value(page).run()
    assert not at.exception and len(at.file_uploader) == 1
    link = next(
        m
        for m in at.markdown
        if m.value == "[Go to Inspection setup and upload](#inspection-setup)"
    )
    heading = next(h for h in at.subheader if h.value == "Inspection setup")
    assert heading.proto.anchor == "inspection-setup"
    elements = list(at.main)
    assert elements.index(link) < elements.index(heading) < elements.index(at.file_uploader[0])
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert not at.exception and at.image
    assert not any(m.value == link.value for m in at.markdown)


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_unavailable_batch_prerequisites_precede_upload_and_reports_offer_catalog(
    source_app, category
):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value("Batch inspection").run()
    notice = next(i for i in at.info if i.value.startswith("Selected model unavailable."))
    elements = list(at.main)
    assert elements.index(notice) < elements.index(at.file_uploader[0])
    assert button(at, "Run batch inspection").disabled
    at.selectbox("workspace_page").set_value("Reports").run()
    assert any("verified local model" in i.value for i in at.info)
    assert any("does not create a research report" in c.value for c in at.caption)
    button(at, "Check models in Catalog").click().run()
    assert not at.exception and at.selectbox("workspace_page").value == "Model Catalog"


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_shared_header_and_route_context_keep_one_native_navigation(source_app, category):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    for page in PAGES:
        at.selectbox("workspace_page").set_value(page).run()
        assert not at.exception
        headers = [
            element
            for element in at.get("html")
            if '<header class="fv-workbench-header">' in element.proto.body
        ]
        assert len(headers) == 1 and len(at.title) == 1
        elements = list(at.main)
        assert elements.index(headers[0]) < elements.index(at.selectbox[0])
        assert elements.index(at.selectbox[0]) < elements.index(at.title[0])
        scope = "Both categories" if page == "Model Catalog" else category
        assert any(c.value.startswith(scope + " · ") for c in at.caption)
        assert len([s for s in at.selectbox if s.key == "workspace_page"]) == 1
        assert at.main.selectbox("workspace_page").value == page
        assert len(at.main.selectbox("workspace_page").options) == 8
        assert not at.sidebar.radio and not at.sidebar.selectbox
        assert any("PCB1: BLOCKED · PCB2: INCONCLUSIVE" in c.value for c in at.caption)


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_full_width_loss_notice_precedes_navigation_and_inspect_keeps_image_first(
    source_app, category
):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value("Image inspection").run()
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    assert not at.exception
    notice = navigation_warnings(at)
    assert len(notice) == 1
    elements = list(at.main)
    assert elements.index(notice[0]) < elements.index(at.selectbox[0])
    assert elements.index(at.image[0]) < elements.index(at.file_uploader[0])
    assert at.image[0].proto.imgs[0].alt == "Input image"
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))
    before = at.image[0].proto.imgs[0]
    image_identity = (before.url, before.caption, before.alt)
    at.run()
    after = at.image[0].proto.imgs[0]
    assert (after.url, after.caption, after.alt) == image_identity
    clear = (
        "Clear image and results" if category == "Cashew" else "Clear MetalParts image and results"
    )
    button(at, clear).click().run()
    assert not at.exception and not at.image and not navigation_warnings(at)
    assert not at.checkbox[0].value and at.file_uploader[0].value is None


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
@pytest.mark.parametrize("page", ["Image inspection", "Compare"])
def test_expected_missing_models_are_informational_but_invalid_inputs_remain_errors(
    source_app, category, page
):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value(page).run()
    assert not at.exception and not at.error
    assert any("inference is disabled" in message.value for message in at.info)
    assert len(at.file_uploader) == 1
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))
    if category != "Cashew":
        assert any("no binary threshold" in message.value for message in at.caption)
        assert any(e.label == "Artifact verification details" for e in at.expander)
    at.file_uploader[0].set_value(("invalid.png", b"not an image", "image/png")).run()
    assert not at.exception and at.error and not at.image
    at.file_uploader[0].set_value(("synthetic.png", build_demo().images[0], "image/png")).run()
    at.checkbox[0].check().run()
    assert not at.exception and not at.error and len(at.image) == 4
    assert at.image[0].proto.imgs[0].alt == "Input image"
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))
    assert "results" not in at.session_state and "metal_results" not in at.session_state


@pytest.mark.parametrize(
    ("action", "page"),
    [
        ("Open Inspect", "Image inspection"),
        ("Open Compare", "Compare"),
        ("Open Batch", "Batch inspection"),
    ],
)
def test_overview_workflow_entries_allow_preview_without_model_execution(source_app, action, page):
    at, _ = source_app
    at.run()
    button(at, action).click().run()
    assert not at.exception and at.selectbox("workspace_page").value == page
    assert len(at.file_uploader) == 1
    assert all(b.disabled for b in at.button if b.label.startswith("Run "))
    assert "results" not in at.session_state and "batch_report" not in at.session_state


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_overview_guides_source_only_start_without_creating_results(source_app, category):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    assert not at.exception and not at.error and not at.warning
    assert button(at, "Open synthetic demo").proto.type == "primary"
    assert button(at, "Open Model Catalog").proto.type == "secondary"
    assert any(
        "0 of " in item.value and "local methods ready" in item.value for item in at.markdown
    )
    assert any("food safety" in item.value for item in at.markdown)
    assert any("NOT CLEARED" in item.value for item in at.caption)
    assert "synthetic_demo" not in at.session_state
    assert len(at.image) == 8
    button(at, "Open synthetic demo").click().run()
    assert not at.exception and at.selectbox("workspace_page").value == "Synthetic demo"
    assert "synthetic_demo" not in at.session_state
    button(at, "Generate synthetic demo").click().run()
    assert not at.exception and "synthetic_demo" in at.session_state
    assert "results" not in at.session_state and "metal_results" not in at.session_state
