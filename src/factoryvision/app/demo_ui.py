"""Explicit synthetic validation demonstration with separate state and export namespace."""

from dataclasses import asdict

import streamlit as st

from factoryvision.app import presentation, review_ui
from factoryvision.demo import NOTICE, build_demo, csv_bytes, json_bytes


def clear():
    review_ui.clear("demo")
    st.session_state.pop("synthetic_demo", None)


def render():
    st.warning(NOTICE)
    st.write(
        "Generate two original geometric images and one deliberately invalid input. "
        "This demonstrates validation and example exports. No research model, weights, "
        "training, anomaly scores or defect decisions are involved."
    )
    st.caption(
        "Category selection does not turn geometric examples into category-matched "
        "research data. This demonstration is independent of the research models."
    )
    actions = st.columns([1, 1, 2])
    if actions[1].button("Clear synthetic demo"):
        clear()
    if actions[0].button("Generate synthetic demo", type="primary"):
        demo = build_demo()
        st.session_state["synthetic_demo"] = (demo, csv_bytes(demo.report), json_bytes(demo.report))
    bundle = st.session_state.get("synthetic_demo")
    if bundle is None:
        st.info(
            "No demonstration has run. Generate explicitly; no files are written automatically."
        )
        return
    demo, csv_export, json_export = bundle
    st.subheader("Synthetic validation example")
    st.caption(NOTICE)
    for column, image in zip(st.columns(2), demo.images, strict=True):
        column.image(
            image, caption="Self-generated geometry; not an inspected product", width="content"
        )
    presentation.table([asdict(row) for row in demo.report.rows], "demo")
    st.write(
        "Two inputs pass image validation; one is rejected. All scores and thresholds "
        "are null because no inference was executed."
    )
    st.caption("Separate demo schema: synthetic-workflow-v1 · " + demo.report.demo_id)
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download synthetic demo CSV",
        csv_export,
        file_name=demo.report.demo_id + ".csv",
        mime="text/csv; charset=utf-8",
        on_click="ignore",
        key="demo_csv",
    )
    downloads[1].download_button(
        "Download synthetic demo JSON",
        json_export,
        file_name=demo.report.demo_id + ".json",
        mime="application/json",
        on_click="ignore",
        key="demo_json",
    )
    st.caption(
        "These exports carry the synthetic warning and no model identity. They cannot "
        "be used as schema-v1 Quality Control Research Reports."
    )
    review_ui.render("demo", demo.report)
