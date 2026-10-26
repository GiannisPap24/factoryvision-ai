"""Catalog presentation through existing artifact checks, without model inference."""

from importlib.resources import files
from pathlib import PurePosixPath

import streamlit as st

from factoryvision.app import presentation
from factoryvision.inference.model_catalog import catalog_entries


def render(coordinator, category):
    entries = catalog_entries(coordinator)
    st.caption("File verification only. Models load on explicit inference; nothing is downloaded.")
    with st.expander("Registry summary"):
        presentation.table(
            [
                {
                    "Model": e.definition.identity.name,
                    "Category": e.definition.identity.category,
                    "Family": e.definition.family,
                    "Local artifacts": "Verified files" if e.available else "Unavailable",
                }
                for e in entries
            ],
            "catalog",
        )
    for offset in range(0, len(entries), 2):
        for column, entry in zip(st.columns(2), entries[offset : offset + 2], strict=True):
            definition, info = entry.definition, entry.definition.identity
            with column.container(border=True, key="fv-card-catalog-" + info.model_id):
                st.subheader(info.name)
                st.caption(info.category.upper() + " · " + definition.family)
                st.markdown("**" + ("Verified files" if entry.available else "Unavailable") + "**")
                st.write(definition.learning)
                st.caption("Outputs: " + ", ".join(definition.outputs))
                st.caption("Local runtime: Python 3.14.10 · PyTorch CPU · Pillow")
                st.caption(definition.evaluation)
                card_name = PurePosixPath(definition.documentation).name
                st.download_button(
                    "Download model card",
                    files("factoryvision.app").joinpath("model_cards", card_name).read_bytes(),
                    file_name=card_name,
                    mime="text/markdown",
                    key="model-card-" + info.model_id,
                    on_click="ignore",
                )
    choices = {
        e.definition.identity.model_id: e
        for e in entries
        if e.definition.identity.category == category
    }
    selected = st.selectbox(
        "Catalog model",
        list(choices),
        format_func=lambda key: choices[key].definition.identity.name,
        key="catalog_model_" + category,
    )
    entry = choices[selected]
    definition, info = entry.definition, entry.definition.identity
    st.subheader(info.name)
    st.caption("Stable model ID: " + info.model_id + " · Category: " + info.category)
    st.write(definition.family + " · " + definition.learning)
    st.warning(definition.evaluation)
    if entry.available:
        st.success(
            "Registered artifact files verified. Runtime loading is checked again on "
            "explicit inference; file verification alone is not a prediction."
        )
    else:
        st.error("Registered artifacts unavailable or incompatible. Real inference is disabled.")
        presentation.setup_help()
    st.write("Preprocessing: " + definition.preprocessing)
    st.write("Outputs: " + ", ".join(definition.outputs))
    st.write("Score: " + info.score_label)
    st.write(
        "No validated binary threshold."
        if info.frozen_threshold is None
        else f"Frozen research threshold: {info.frozen_threshold:.17g}; score > threshold."
    )
    st.caption("Expected artifact SHA-256")
    st.code(info.registered_artifact_sha256, language=None)
    st.caption("Verified local artifact SHA-256")
    st.code(entry.verified_artifact_sha256 or "Unavailable", language=None)
    if definition.pretrained_sha256:
        st.write("Frozen torchvision ResNet18 IMAGENET1K_V1 component; not newly trained here.")
        st.code(definition.pretrained_sha256, language=None)
        st.link_button("Pretrained component source (external)", definition.pretrained_url)
    else:
        st.write("No pretrained third-party component.")
    st.write("Model card in the source distribution: " + definition.documentation)
    with st.expander("Scientific limitations", expanded=False):
        for limitation in definition.limitations:
            st.write(limitation)
