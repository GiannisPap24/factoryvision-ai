"""Local inspection interface. All scoring lives in factoryvision.inference."""

import hashlib
from pathlib import Path

import streamlit as st
import torch

from factoryvision.app import batch_ui, catalog_ui, demo_ui, metalparts_ui, presentation
from factoryvision.inference.batch import BatchService
from factoryvision.inference.service import InspectionService, ModelUnavailable
from factoryvision.inference.uploads import InputRejected, decode_upload
from factoryvision.inference.visualization import panels


@st.cache_resource(max_entries=1, show_spinner=False)
def service():
    torch.set_num_threads(4)
    return InspectionService(Path.cwd())


def research():
    st.subheader("Recorded research evidence")
    st.caption("Historical Cashew evaluation • 50 normal + 100 anomalous images • Milestone 2B")
    st.dataframe(
        {
            "Method": ["Autoencoder", "Spatial mean"],
            "Image AUROC": [0.937600, 0.736800],
            "Image AP": [0.943219, 0.841216],
            "Recall at frozen threshold": [0.52, 0.15],
            "False positives / 50 normal": [3, 0],
            "Pixel AUROC": [0.818119, 0.895301],
            "Pixel AP": [0.080936, 0.267302],
        }
    )
    st.markdown(
        "AUROC measures ranking across thresholds; AP summarizes precision and recall. "
        "Recall here uses the frozen normal-calibration threshold. Pixel metrics measure "
        "localization across image pixels, where background imbalance matters. "
        "The autoencoder missed 48 of 100 anomalies; spatial mean missed 85. "
        "Good image ranking does not imply precise defect localization."
    )
    st.info(
        "Feature-memory and student–teacher: official Cashew defect metrics are not established. "
        "Their development diagnostics do not establish comparable test accuracy."
    )
    st.warning(
        "PCB1: BLOCKED — confirmed source overlap. PCB2: INCONCLUSIVE — "
        "56 similarity pairs remain unresolved; no independent human adjudication."
    )
    st.caption(
        "Source: docs/model-card.md and runs/cashew-eval-v1/metrics.json. "
        "These figures are recorded evidence, not new evaluation results."
    )


def show_result(result):
    view = panels(result)
    with st.container(border=True, key="fv-card-result-" + result.model_id):
        st.subheader(result.model_name)
        from factoryvision.inference.model_catalog import METHODS

        st.caption(METHODS[result.model_id][0])
        columns = st.columns(3)
        columns[0].metric("Anomaly score", f"{result.score:.8f}")
        columns[1].metric(
            "Frozen threshold",
            "Unavailable" if result.threshold is None else f"{result.threshold:.8f}",
        )
        columns[2].metric("CPU inference", f"{result.inference_ms:.1f} ms")
        st.caption(result.score_label + " · scores are not probabilities")
        if result.above_threshold is None:
            st.warning("No valid frozen threshold: no binary decision is available.")
        elif result.above_threshold:
            st.warning("Above frozen threshold — elevated anomaly score; inspect the output.")
        else:
            st.info("At or below frozen threshold — this does not establish defect absence.")
        map_tab, overlay_tab, reconstruction_tab = st.tabs(
            ["Anomaly map", "Overlay", "Reconstruction / reference"]
        )
        with map_tab:
            left, right = st.columns(2)
            left.image(view["original"], caption="Original RGB", width="stretch")
            right.image(view["heatmap"], caption="Relative anomaly intensity", width="stretch")
        with overlay_tab:
            st.image(view["overlay"], caption="45% heatmap overlay", width="stretch")
        with reconstruction_tab:
            if view["reconstruction"] is not None:
                st.image(
                    view["reconstruction"], caption="Autoencoder reconstruction", width="stretch"
                )
            elif view["reference"] is not None:
                st.image(
                    view["reference"], caption="Frozen spatial mean reference", width="stretch"
                )
            else:
                st.info("This model produces anomaly features, not an RGB reconstruction.")
        presentation.legend(view["ceiling"])
        with st.expander("Score precision, timing and model identity"):
            st.write(f"Raw score: {result.score:.10g} · threshold: {result.threshold}")
            st.caption(
                f"Preprocess {result.preprocessing_ms:.1f} ms · artifact verification/loading "
                f"{result.artifact_check_load_ms:.1f} ms · total {result.total_ms:.1f} ms"
            )
            st.code(result.artifact_sha256, language=None)


def inspect(engine, statuses, compare=False):
    if not compare and not any(row["available"] for row in statuses):
        st.caption("Inference unavailable · No verified local model artifacts")
    controls, output, preview_slot = presentation.inspection_layout(compare)
    preview = None
    with controls:
        st.subheader("Inspection controls" if compare else "Inspection setup")
        st.caption("Category: Cashew · Change category in navigation")
        if compare:
            st.caption("1 · Select model   2 · Upload   3 · Run")
        available = {s["id"]: s["name"] for s in statuses if s["available"]}
        chosen = st.multiselect(
            "Models",
            options=list(available),
            default=list(available)[:1],
            format_func=available.get,
            max_selections=None if compare else 1,
            key="compare_models" if compare else "inspect_model",
        )
        upload = st.file_uploader(
            "Choose a Cashew image",
            type=["png", "jpg", "jpeg"],
            max_upload_size=8,
            key=f"upload_{st.session_state.get('upload_generation', 0)}",
        )
        acknowledged = st.checkbox(
            "I confirm this is a permitted Cashew image, not an official test image.",
            key="cashew_ack",
        )
        if upload is None and compare:
            st.info("Choose a model and upload an image to begin. Inference runs only on request.")
        payload = upload.getvalue() if upload is not None else None
        identity = (
            hashlib.sha256(payload).hexdigest() if payload else None,
            tuple(chosen),
            compare,
            acknowledged,
        )
        if st.session_state.get("request_identity") != identity:
            st.session_state.pop("results", None)
            st.session_state["request_identity"] = identity
        valid = False
        if payload is not None:
            try:
                preview = decode_upload(payload)
                preview.thumbnail((640, 640))
                if compare:
                    with st.expander("Input preview"):
                        st.image(preview, caption="Original RGB before inference", width="content")
                valid = True
            except InputRejected as exc:
                st.error(str(exc))
        st.button("Clear image and results", on_click=presentation.clear_inspection)
        if st.button(
            "Run inspection",
            type="primary",
            width="content" if compare else "stretch",
            disabled=not (valid and chosen and acknowledged),
        ):
            st.session_state.pop("results", None)
            try:
                with st.spinner("Verifying frozen artifacts and running CPU inference…"):
                    st.session_state["results"] = engine.inspect_many(payload, chosen)
            except (InputRejected, ModelUnavailable) as exc:
                st.error(str(exc))
            except (RuntimeError, MemoryError):
                presentation.inference_failure()
        if not available:
            st.error(
                "No verified local model artifacts are available. "
                "Preview remains available; inference is disabled. "
                "See Model Catalog for prerequisites."
            )
            presentation.setup_help()
        with st.expander("Permitted inputs and limits"):
            st.write(
                "Cashew research models: category and distribution are not detected "
                "automatically. Official test images, including PCB1/PCB2, are prohibited."
            )
            st.caption(
                "JPEG / PNG · ≤8 MiB · ≤4096 pixels per side · ≤16M pixels · "
                "opaque, upright, single-frame. Uploads stay in process memory."
            )
        st.caption(
            "Models are displayed in selection order. Their raw scores use different scales; "
            "they cannot be used to rank models. Changing the image or selection clears results."
            if compare
            else "Preview only until an explicit run. "
            "Changing image, model or consent clears results."
        )
    if preview_slot is not None:
        with preview_slot.container():
            presentation.input_viewer(preview if valid else None)
    with output:
        results = st.session_state.get("results", ())
        if not results and compare:
            st.subheader("Spatial evidence")
            st.info(
                "No inspection yet. Select a model, upload a permitted image and run inspection."
            )
        for offset in range(0, len(results), 2):
            row = results[offset : offset + 2]
            for column, result in zip(st.columns(len(row)), row, strict=True):
                with column:
                    show_result(result)
        from factoryvision.app import review_ui

        review_ui.render("single", results or None, "cashew")


def render(engine, metal_engine=None):
    st.set_page_config(
        page_title="FactoryVision AI · Industrial Precision",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    presentation.styling()
    st.sidebar.header("FactoryVision AI")
    st.sidebar.caption("Industrial Precision")
    navigation_notice = st.sidebar.empty()
    category = st.sidebar.selectbox("Category", ["Cashew", "MetalParts (experimental)"])
    labels = {
        "Overview": "Overview",
        "Image inspection": "Inspect",
        "Compare": "Compare",
        "Batch inspection": "Batch Inspection",
        "Reports": "Reports",
        "Model Catalog": "Model Catalog",
        "Research results": "Research",
        "Synthetic demo": "Demo",
    }
    with st.sidebar.container(key="fv-nav"):
        page = st.radio("Workspace", list(labels), format_func=labels.get, key="workspace_page")
    presentation.switch_category(category)
    previous = st.session_state.get("active_page")
    if previous != page:
        if {previous, page} <= {"Batch inspection", "Reports"}:
            if page == "Reports" and "batch_report" in st.session_state:
                st.session_state["batch_archive_view"] = True
        else:
            batch_ui.clear()
            st.session_state.pop("batch_archive_view", None)
        presentation.clear_inspection()
        demo_ui.clear()
        st.session_state["active_page"] = page
    st.sidebar.divider()
    st.sidebar.markdown("**Research prototype**")
    st.sidebar.caption("Local CPU · No cloud inference")
    st.sidebar.caption("PCB1: BLOCKED · PCB2: INCONCLUSIVE")
    titles = {
        "Overview": (
            "Overview",
            "Explore the workflow. Understand availability. Keep results local.",
        ),
        "Image inspection": (
            "Inspect an image",
            "One image. One compatible model. Aligned spatial evidence.",
        ),
        "Compare": (
            "Compare models",
            "One input, distinct methods. Compare maps, not raw score scales.",
        ),
        "Batch inspection": (
            "Batch inspection",
            "Validate inputs, run sequential CPU inspection and review every outcome.",
        ),
        "Reports": (
            "Quality reports",
            "Review the completed batch. Export validated results without rerunning inference.",
        ),
        "Model Catalog": (
            "Model Catalog",
            "Six frozen methods. Transparent provenance and local availability.",
        ),
        "Research results": (
            "Research evidence",
            "Recorded experiments, methodology and limits. No new evaluation.",
        ),
        "Synthetic demo": (
            "Synthetic workflow demonstration",
            "Original geometry. Validation only. No model inference.",
        ),
    }
    with st.container(key="fv-shell"):
        presentation.header()
        st.title(titles[page][0])
        st.caption(titles[page][1])
        statuses = engine.statuses() if category == "Cashew" else []
        if page == "Overview":
            if category != "Cashew":
                metal = (
                    metal_engine if metal_engine is not None else metalparts_ui.service(engine.root)
                )
                statuses = [
                    dict(metal.status(), name="Experimental MetalParts autoencoder"),
                    dict(
                        metalparts_ui.feature_service(engine.root).status(),
                        name="Experimental MetalParts feature memory",
                    ),
                ]
            presentation.overview(category, statuses)
        elif page == "Synthetic demo":
            demo_ui.render()
        elif page == "Model Catalog":
            catalog_ui.render(
                BatchService(
                    engine,
                    metal_engine
                    if metal_engine is not None
                    else metalparts_ui.service(engine.root),
                    metalparts_ui.feature_service(engine.root),
                ),
                "cashew" if category == "Cashew" else "metalparts",
            )
        elif page == "Batch inspection":
            batch_ui.render(
                category,
                engine,
                metal_engine if metal_engine is not None else metalparts_ui.service(engine.root),
                metalparts_ui.feature_service(engine.root),
            )
        elif page == "Reports":
            source = st.session_state.get("batch_report")
            if source is None:
                st.info(
                    "No completed batch report. Run Batch Inspection to create a research report."
                )
                st.button(
                    "Open Batch Inspection",
                    on_click=presentation.navigate,
                    args=("Batch inspection",),
                )
            else:
                batch_ui.results(source)
                st.button("Clear report and batch", on_click=batch_ui.clear)
        elif category == "MetalParts (experimental)":
            metal = metal_engine if metal_engine is not None else metalparts_ui.service(engine.root)
            if page in ("Image inspection", "Compare"):
                metalparts_ui.inspect(metal, compare=page == "Compare")
            else:
                metalparts_ui.research(metal.root)
        elif page in ("Image inspection", "Compare"):
            inspect(engine, statuses, compare=page == "Compare")
        else:
            research()
        with st.expander("Research and session boundaries"):
            st.write(
                "Research demonstration only. Not industrially validated. Do not use outputs "
                "for safety, food quality or automated acceptance decisions. High scores can "
                "reflect lighting or geometry; low scores do not establish defect absence."
            )
            st.caption(
                "Session results stay in memory until cleared or the session expires. "
                "Clearing is not a secure memory-erasure guarantee. No cloud inference "
                "or application telemetry."
            )
    # Fill above navigation after rendering so newly created results are covered immediately.
    notice = presentation.navigation_notice(st.session_state)
    if notice:
        navigation_notice.warning("Before navigating: " + notice)


if __name__ == "__main__":
    render(service())
