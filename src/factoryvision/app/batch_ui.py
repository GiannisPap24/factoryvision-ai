"""Explicit local batch workflow. Session state contains one bounded batch only."""

import streamlit as st

from factoryvision.app import presentation, report_ui, review_ui
from factoryvision.inference.batch import (
    MAX_IMAGES,
    BatchService,
    check_budget,
    identity,
    models,
    prepare,
)
from factoryvision.inference.uploads import InputRejected


def reset():
    review_ui.clear("batch")
    for key in (
        "batch_plan",
        "batch_report",
        "batch_signature",
        "batch_started",
        "batch_detail",
        "batch_exports",
        "batch_archive_view",
    ):
        st.session_state.pop(key, None)


def clear():
    reset()
    generation = st.session_state.get("batch_upload_generation", 0)
    st.session_state.pop(f"batch_upload_{generation}", None)
    st.session_state["batch_upload_generation"] = generation + 1
    st.session_state["batch_ack"] = False


def detail(row):
    st.subheader(row.image_id + " · " + row.model_name)
    presentation.result_images(
        {
            "original": row.detail.original,
            "heatmap": row.detail.heatmap,
            "overlay": row.detail.overlay,
            "reconstruction": row.detail.reconstruction,
            "reference": row.detail.reference,
        },
        key="batch",
    )
    st.write(row.interpretation)
    st.caption(row.score_label + " · not a defect probability")
    st.write(
        f"Score: {row.score!r} · Frozen threshold: {row.threshold!r}"
        f" · CPU inference: {row.inference_ms:.2f} ms"
    )
    presentation.legend(row.detail.ceiling)
    st.caption(
        "Display preview: at most 384 pixels per side. Full-resolution maps "
        "are released after each image."
    )
    with st.expander("Verified artifact identity"):
        st.code(row.artifact_sha256, language=None)


def results(report):
    st.warning(
        "Session only — export before leaving. Batch Inspection and Reports retain this batch "
        "and saved reviews. Leaving this pair, changing category or clearing removes them. "
        "Refresh, session expiry or a server restart may also lose them."
    )
    summary, evidence, human = st.tabs(["Summary & exports", "Image evidence", "Human review"])
    with summary:
        valid = report_ui.render(report)
    if not valid:
        review_ui.clear("batch")
        st.error(
            "Batch evidence and Human Review unavailable: "
            "report validation failed. Clear and retry."
        )
        return
    good = {row.image_id: row for row in report.rows if row.status == "inspected"}
    with evidence:
        if good:
            chosen = st.selectbox("Inspect individual result", list(good), key="batch_detail")
            detail(good[chosen])
        else:
            st.info("No successful inspection. Rejected, failed and skipped inputs have no maps.")
    with human:
        st.caption(
            "Review successful predictions only. Human observations are separate from the "
            "research report; export both if you need results and review history."
        )
        review_ui.render("batch", report)


def render(category, cashew, metal, feature):
    category = "cashew" if category == "Cashew" else "metalparts"
    engine = BatchService(cashew, metal, feature)
    names = {k: cashew.specs[k]["name"] for k in models("cashew")}
    names.update(
        metalparts_autoencoder="Experimental MetalParts autoencoder",
        metalparts_feature_memory="Experimental MetalParts feature memory",
    )
    if st.session_state.get("batch_archive_view") and "batch_report" in st.session_state:
        st.info(
            "Batch snapshot retained. Open Reports to review its status or export; "
            "inference will not repeat."
        )
        st.button("Open quality report", on_click=presentation.navigate, args=("Reports",))
        if st.button("Clear batch"):
            clear()
            st.rerun()
        results(st.session_state["batch_report"])
        return
    setup, input_column = st.columns([1, 1.25], gap="medium")
    inputs = input_column.container(border=True, key="fv-batch-inputs", gap="xsmall")
    with setup.container(border=True, key="fv-batch-setup", gap="xsmall"):
        st.subheader("Batch setup")
        st.caption("One compatible model · Explicit execution · Sequential CPU")
        with st.expander("Permitted inputs and processing limits"):
            st.warning(
                "Research demonstration only. No industrial acceptance/rejection decisions. "
                "Use permitted category-matched images; official test images and transformed "
                "copies are prohibited. MetalParts has no validated binary threshold."
            )
            st.caption(
                "One model · Sequential CPU · JPEG/PNG · 8 MiB per image · 32 MiB total · "
                "4096 pixels per side / 16M pixels per image. "
                "MetalParts minimum: 256 pixels per side. "
                "Uploader order is preserved; filenames are not included in results."
            )
        settings = st.columns([2, 1])
        selected = settings[0].selectbox(
            "Batch model", models(category), format_func=names.get, key=f"batch_model_{category}"
        )
        limit = settings[1].number_input(
            "Batch image limit", min_value=1, max_value=MAX_IMAGES, value=20, step=1
        )
        if st.button("Clear batch"):
            clear()
        uploads = st.file_uploader(
            "Choose batch images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            max_upload_size=8,
            key=f"batch_upload_{st.session_state.get('batch_upload_generation', 0)}",
        )
        if "batch_budget_error" in st.session_state:
            st.error(st.session_state.pop("batch_budget_error"))
        try:
            check_budget([u.size for u in uploads], int(limit))
        except InputRejected as exc:
            clear()
            st.session_state["batch_budget_error"] = str(exc) + " Upload selection cleared."
            st.rerun()
        acknowledged = st.checkbox(
            "I confirm these are permitted images for the selected category, "
            "not official test images.",
            key="batch_ack",
        )
        payloads = tuple(upload.getvalue() for upload in uploads)
        batch_id = identity(category, selected, payloads, int(limit))
        signature = (batch_id, acknowledged)
        if st.session_state.get("batch_signature") != signature:
            reset()
            st.session_state["batch_signature"] = signature
        if "batch_plan" not in st.session_state:
            try:
                st.session_state["batch_plan"] = prepare(category, selected, payloads, int(limit))
            except (MemoryError, RuntimeError):
                st.error(
                    "Batch preview could not finish. Clear the batch and choose smaller images."
                )
                return
        plan = st.session_state["batch_plan"]
        with inputs:
            st.subheader("Input status")
            for column, label, value in zip(
                st.columns(3),
                ("Selected", "Validated inputs", "Rejected inputs"),
                (len(plan.items), plan.accepted, len(plan.items) - plan.accepted),
                strict=True,
            ):
                column.metric(label, value, border=True)
            if not plan.items:
                st.info("No images selected. Add permitted JPEG/PNG files in Batch setup.")
            if plan.items:
                with st.expander("Selected batch preview", expanded=True):
                    presentation.table(
                        [
                            {"Image": i.image_id, "Validation": i.error or "Accepted"}
                            for i in plan.items
                        ],
                        "batch-inputs",
                    )
                    for offset in range(0, len(plan.items), 4):
                        for column, item in zip(
                            st.columns(4), plan.items[offset : offset + 4], strict=False
                        ):
                            if item.preview:
                                column.image(
                                    item.preview,
                                    caption=item.image_id,
                                    width="content",
                                    alt=f"{item.image_id} validated input preview",
                                )
        status = engine.status(category, selected)
        if not status["available"]:
            st.error("Selected model unavailable. Restore its original verified local artifacts.")
            presentation.setup_help()
        started = st.session_state.get("batch_started") == batch_id
        if started and "batch_report" not in st.session_state:
            st.error(
                "The previous run did not complete. Clear before trying again; no automatic resume."
            )
        if (
            st.button(
                "Run batch inspection",
                type="primary",
                width="stretch",
                disabled=not (plan.accepted and acknowledged and status["available"]) or started,
            )
            and not started
        ):
            st.session_state["batch_started"] = batch_id
            progress = st.progress(0, text="Starting sequential CPU inspection…")

            def update(done, total, state):
                progress.progress(done / total, text=f"{done}/{total} · {state}")

            try:
                report = engine.run(plan, update)
                # Publish one complete immutable report, never a partially mutated result list.
                st.session_state["batch_report"] = report
                st.rerun()
            except Exception:
                # Render the terminal controls from the existing no-replay guard immediately.
                st.rerun()
    if "batch_report" in st.session_state:
        st.button("Open quality report", on_click=presentation.navigate, args=("Reports",))
        results(st.session_state["batch_report"])
    st.caption(
        "No background jobs, cancellation or resumption. A completed batch is not rerun "
        "on reruns. Clear images/results when finished; no secure memory erasure is claimed."
    )
