"""F7 semantic contracts through native images; only owned synthetic inputs."""

import pytest
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import source_app as source_app
from test_review_app import button

from factoryvision.demo import build_demo


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
@pytest.mark.parametrize(
    "page", ["Overview", "Synthetic demo", "Image inspection", "Compare", "Batch inspection"]
)
def test_meaningful_images_are_named_without_upload_filenames(source_app, category, page):
    at, _ = source_app
    at.run().selectbox[0].set_value(category).run()
    at.radio("workspace_page").set_value(page).run()
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
