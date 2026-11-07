"""F7 semantic contracts through native images; only owned synthetic inputs."""

import pytest
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import source_app as source_app
from test_review_app import button

from factoryvision.demo import build_demo


def test_nested_detail_cannot_drive_outer_viewer_css_state():
    """Guard the shipped selectors; actual native four-state behavior needs a browser.

    A nested details element retains open when its ancestor closes. The verified
    Streamlit structure gives each keyed disclosure its own direct stExpander
    child. Every open-state rule must select that owner's details, so the hidden
    inner state cannot hide previews or leave cards spanning their grids.
    """
    import re
    from importlib.resources import files

    css = files("factoryvision.app").joinpath("workbench.css").read_text(encoding="utf-8")
    rules = [
        (selector, body)
        for selector, body in re.findall(r"([^{}]+)\{([^{}]+)\}", css)
        if "details[open]" in selector
    ]
    assert len(rules) == 4
    owner_state = '[class*="st-key-fv-expand-"] > [data-testid="stExpander"] > details[open]'
    for selector, _ in rules:
        assert "details[open]" not in selector.replace(owner_state, "OUTER_OPEN")
    for context in (
        "st-key-fv-image-",
        "st-key-fv-sample-strip",
        "st-key-fv-inspection-layout",
        "st-key-fv-comparison-grid",
    ):
        assert any(context in selector for selector, _ in rules)
    spans = next(selector for selector, _ in rules if "st-key-fv-sample-strip" in selector)
    assert "st-key-fv-batch-thumbnails" in spans


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
@pytest.mark.parametrize(
    "page", ["Overview", "Synthetic demo", "Image inspection", "Compare", "Batch inspection"]
)
def test_meaningful_images_are_named_without_upload_filenames(source_app, category, page):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value(page).run()
    if page == "Synthetic demo":
        button(at, "Generate synthetic demo").click().run()
    elif page in {"Image inspection", "Compare", "Batch inspection"}:
        # Filenames are deliberately unsuitable as accessible descriptions.
        upload = ("private-upload-name.png", build_demo().images[0], "image/png")
        at.file_uploader[0].set_value([upload] if page == "Batch inspection" else upload).run()
    assert not at.exception
    images = [image for element in at.image for image in element.proto.imgs]
    assert images
    assert all(image.alt.strip() and image.caption for image in images)
    assert all("private-upload-name" not in image.alt for image in images)
    assert all(image.alt != image.caption for image in images)
    before = [(image.url, image.caption, image.alt) for image in images]
    at.run()
    assert [(i.url, i.caption, i.alt) for e in at.image for i in e.proto.imgs] == before
    # Native ignore-mode disclosures are client-only. AppTest sees their rendered
    # content; real-browser checks own collapse/AX/focus behavior.
    assert any(e.label.endswith("Complete image") for e in at.expander)
    assert len(images) % 4 == 0
    for index in range(0, len(images), 4):
        preview, whole, context, detail = images[index : index + 4]
        assert [i.alt for i in (whole, context, detail)] == [
            prefix + preview.alt
            for prefix in (
                "Complete view: ",
                "Context for magnified detail: ",
                "Magnified detail: ",
            )
        ]
        assert all(i.caption == preview.caption for i in (whole, context, detail))
        assert whole.url == context.url == detail.url
        # Overview's96px thumbnail is deliberately a smaller native rendition.
        if page != "Overview":
            assert whole.url == preview.url
    assert not any(
        at.session_state.get(key) for key in ("results", "metal_results", "batch_report")
    )


@pytest.mark.parametrize("kind", ["pil", "png", "jpeg", "batch", "demo", "large"])
def test_image_view_preserves_accepted_full_transport(monkeypatch, kind):
    """Compare full bytes, not media URLs (large media may use partial hashing)."""
    import io
    from types import SimpleNamespace

    import numpy as np
    from PIL import Image
    from streamlit.elements.lib import image_utils
    from streamlit.elements.lib.layout_utils import LayoutConfig

    from factoryvision.app import presentation
    from factoryvision.inference import batch

    size = (1003, 1501) if kind == "large" else ((173, 91) if kind == "pil" else (91, 173))
    pixels = np.random.default_rng(8).integers(0, 256, (*size, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    image = source
    if kind in {"png", "jpeg", "large"}:
        buffer = io.BytesIO()
        source.save(buffer, format="JPEG" if kind == "jpeg" else "PNG")
        image = buffer.getvalue()
    elif kind == "batch":
        image = batch.png(source, 384)
    elif kind == "demo":
        image = build_demo().images[0]
    before = image.tobytes() if isinstance(image, Image.Image) else image
    transports = []

    def add(data, mime, *args, **kwargs):
        transports.append((bytes(data), mime))
        return "/media/synthetic"

    monkeypatch.setattr(image_utils.runtime, "exists", lambda: True)
    monkeypatch.setattr(
        image_utils.runtime,
        "get_instance",
        lambda: SimpleNamespace(media_file_mgr=SimpleNamespace(add=add)),
    )
    monkeypatch.setattr(image_utils.caching, "save_media_data", lambda *args, **kwargs: None)

    def encode(value, width):
        image_utils.image_to_url(
            value, LayoutConfig(width=width), False, "RGB", "auto", "synthetic"
        )

    # Frozen original dialog contract, preserving input type and native defaults.
    encode(image, "stretch")
    expected = transports.pop()
    preview_width = 96 if kind == "demo" else "stretch"
    encode(image, preview_width)
    expected_preview = transports.pop()
    semantics = []

    class Surface:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def container(self, *args, **kwargs):
            return self

        def expander(self, *args, **kwargs):
            return self

        def caption(self, *args, **kwargs):
            pass

        def image(self, value, *, caption, alt, width):
            assert value is image
            semantics.append((caption, alt))
            encode(value, width)

    monkeypatch.setattr(presentation, "st", Surface())
    presentation.image_view(
        image, caption="Synthetic caption", alt="Synthetic image", key="test", width=preview_width
    )
    assert transports == [expected_preview, expected, expected, expected]
    assert len({alt for _, alt in semantics}) == 4
    assert all(caption == "Synthetic caption" for caption, _ in semantics)
    decoded = Image.open(io.BytesIO(expected[0]))
    for encoded, mime in transports[1:]:
        actual = Image.open(io.BytesIO(encoded))
        assert mime == expected[1]
        assert (actual.mode, actual.size, actual.tobytes()) == (
            decoded.mode,
            decoded.size,
            decoded.tobytes(),
        )
    assert (image.tobytes() if isinstance(image, Image.Image) else image) == before
    assert np.array_equal(np.asarray(source), pixels)
