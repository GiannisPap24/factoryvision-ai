"""Explicit synthetic validation demonstration with separate state and export namespace."""

import streamlit as st

from factoryvision.app import presentation, review_ui
from factoryvision.demo import NOTICE, build_demo, csv_bytes, json_bytes


def clear():
    review_ui.clear("demo")
    st.session_state.pop("synthetic_demo", None)


def generate():
    demo = build_demo()
    st.session_state["synthetic_demo"] = (demo, csv_bytes(demo.report), json_bytes(demo.report))


def render():
    st.warning(NOTICE)
    if "synthetic_demo" in st.session_state:
        st.warning(
            "Session-only reviews: Generating a new demo or clearing the current demo removes "
            "saved Human Review entries and unsaved edits. "
            "Save any edits and export your reviews before continuing."
        )
    with st.container(horizontal=True, gap="small", key="fv-demo-actions"):
        st.button("Generate synthetic demo", type="primary", on_click=generate)
        st.button(
            "Clear synthetic demo",
            on_click=clear,
            disabled="synthetic_demo" not in st.session_state,
        )
    bundle = st.session_state.get("synthetic_demo")
    if bundle is None:
        with st.container(border=True, key="fv-card-demo-empty", gap="small"):
            st.subheader("A complete workflow, without model weights")
            st.write(
                "Generate two original geometric images and one deliberately invalid input, "
                "then practice validation, Human Review and exports. No model or weights required."
            )
            st.info(
                "No demonstration has run. Generate explicitly; no files are written automatically."
            )
        return
    demo, csv_export, json_export = bundle
    st.markdown("[Review & exports](#demo-review-exports)")
    with st.container(key="fv-demo-images"):
        for column, image, row, alt in zip(
            st.columns(2),
            demo.images,
            demo.report.rows[:2],
            presentation.SYNTHETIC_IMAGE_ALTS,
            strict=True,
        ):
            with column.container(border=True, gap="xsmall"):
                presentation.image_view(
                    image,
                    caption=row.image_id + " · Generated geometry",
                    width="stretch",
                    alt=f"{row.image_id}: {alt}",
                    key="demo-" + row.image_id,
                )
                st.caption(f"{row.width} × {row.height} RGB · Valid input")
    summary = st.container(horizontal=True, key="fv-demo-summary", gap="medium")
    with summary.container(border=True, key="fv-card-demo-validation", gap="small"):
        st.subheader("Input validation")
        st.caption("2 accepted · 1 rejected · Inference not executed")
        presentation.table(
            [
                {
                    "Input": row.image_id,
                    "Validation": "Valid" if row.validation_status == "demo_valid" else "Rejected",
                    "Score": "Not computed",
                }
                for row in demo.report.rows
            ],
            "demo",
        )
        st.caption(
            "Demo 003 contains deliberately invalid bytes. Scores, thresholds and research "
            "flags remain null; validation is not a defect decision."
        )
    with summary.container(border=True, key="fv-card-demo-exports", gap="small"):
        st.subheader("Demo exports", anchor="demo-review-exports")
        st.caption("Validation records only · synthetic-workflow-v1")
        st.download_button(
            "Download synthetic demo CSV",
            csv_export,
            file_name=demo.report.demo_id + ".csv",
            mime="text/csv; charset=utf-8",
            on_click="ignore",
            key="demo_csv",
            width="stretch",
        )
        st.download_button(
            "Download synthetic demo JSON",
            json_export,
            file_name=demo.report.demo_id + ".json",
            mime="application/json",
            on_click="ignore",
            key="demo_json",
            width="stretch",
        )
        st.caption(
            "Synthetic warning retained; no model identity. These are separate from "
            "Quality Control Research Reports."
        )
        st.caption(
            "Category selection does not make these geometric examples category-matched "
            "research data."
        )
        with st.expander("Demo document identity"):
            st.code(demo.report.demo_id, language=None)
    with st.expander("Human Review & review exports"):
        review_ui.render("demo", demo.report)
