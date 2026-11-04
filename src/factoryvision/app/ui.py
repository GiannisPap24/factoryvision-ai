"""Local inspection interface. All scoring lives in factoryvision.inference."""

import hashlib
from pathlib import Path

import streamlit as st
import torch

from factoryvision.app import batch_ui, catalog_ui, demo_ui, metalparts_ui, presentation, review_ui
from factoryvision.inference.batch import BatchService
from factoryvision.inference.service import InspectionService, ModelUnavailable
from factoryvision.inference.uploads import InputRejected, decode_upload
from factoryvision.inference.visualization import panels


@st.cache_resource(max_entries=1, show_spinner=False)
def service():
    torch.set_num_threads(4)
    return InspectionService(Path.cwd())


def research():
    st.subheader("Recorded Cashew evidence")
    with st.container(border=True, key="fv-card-research-status", gap="small"):
        st.markdown("**HISTORICAL · Official test now observed**")
        st.write(
            "One frozen evaluation · Milestone 2B · Model card dated 24 October 2026. "
            "150 images: 100 anomalous and 50 normal. These are recorded results, "
            "not a new evaluation or evidence of production readiness."
        )
        st.warning(
            "Autoencoder missed 48 of 100 anomalies; spatial mean missed 85. "
            "Zero observed false positives for spatial mean does not establish zero risk."
        )
    with st.container(border=True, key="fv-card-research-ranking", gap="small"):
        st.subheader("Ranking and localization")
        st.caption("Published precision · AUROC and AP are not accuracy or defect probabilities.")
        presentation.table(
            {
                "Metric": ["Image AUROC", "Image AP", "Pixel AUROC", "Pixel AP"],
                "Autoencoder": ["0.937600", "0.943219", "0.818119", "0.080936"],
                "Spatial mean": ["0.736800", "0.841216", "0.895301", "0.267302"],
            },
            "research-cashew-ranking",
        )
        st.write(
            "The autoencoder ranks images better on this test; spatial mean ranks pixels "
            "better. Good image ranking does not imply precise defect localization or "
            "overall neural-model superiority."
        )
        with st.expander("How to read these metrics"):
            st.write(
                "AUROC measures ranking across thresholds. AP is non-interpolated average "
                "precision, not trapezoidal PR area. Image prevalence is 66.7% anomalous; "
                "AP and precision do not transfer directly to a low-defect production line."
            )
            st.write(
                "Pixel metrics pool 2,457,600 resized pixels, including normal images, "
                "with 27,672 positives. Correlated pixels and background imbalance limit "
                "interpretation. No AUPRO or thresholded pixel metrics are claimed."
            )
    with st.container(border=True, key="fv-card-research-threshold", gap="small"):
        st.subheader("Outcomes at frozen thresholds")
        presentation.table(
            {
                "Metric": [
                    "TN / FP / FN / TP",
                    "Precision",
                    "Recall",
                    "F1",
                    "Normal false-positive rate",
                ],
                "Autoencoder": ["47 / 3 / 48 / 52", "0.945455", "0.520000", "0.670968", "0.060000"],
                "Spatial mean": [
                    "50 / 0 / 85 / 15",
                    "1.000000",
                    "0.150000",
                    "0.260870",
                    "0.000000",
                ],
            },
            "research-cashew-threshold",
        )
        st.caption(
            "TN: true negative · FP: false positive · FN: false negative · TP: true positive. "
            "Observed false positives: 3 versus 0 among 50 normals."
        )
        with st.expander("Exact thresholds and methodology"):
            st.write("Autoencoder: **0.0031531278509646654**")
            st.write("Spatial mean: **0.017185416119173163**")
            st.write(
                "315 normal fitting images; 67 normal validation images; 68 separate normal "
                "calibration images. Linear 95th percentile; strict score > threshold. "
                "These empirical thresholds do not guarantee a production false-alarm rate."
            )
            st.write(
                "Both methods use 128 × 128 RGB with bilinear antialiased resizing and "
                "raw mean squared RGB residuals. Masks use nearest-neighbor resizing. "
                "Identical test membership; protocol, artifacts and thresholds frozen before "
                "prediction. No test-driven postprocessing, retraining or recalibration."
            )
    with st.expander("Uncertainty and study limitations", expanded=True):
        st.write(
            "One category, one seed and one observed test set. Physical-item and acquisition "
            "independence is not established; near-duplicate uncertainty remains. Square "
            "resizing may hide small defects; residuals can emphasize edges and background. "
            "No camera, factory-shift or industrial-readiness validation is established."
        )
        st.caption(
            "AE-minus-mean image AUROC difference: 0.2008 (conditional paired-bootstrap "
            "95% interval 0.110195 to 0.291805); AP difference: 0.102003 "
            "(0.032783 to 0.167381). The 1,000 image-stratified resamples assume image "
            "independence; physical-item/acquisition independence is unverified."
        )
    st.subheader("Development evidence and evaluation gates")
    st.info(
        "DEVELOPMENT ONLY · Feature-memory and student–teacher: official Cashew defect "
        "metrics are not established. Artificial-probe diagnostics do not establish "
        "real-defect accuracy and cannot be compared with the official-test results above."
    )
    st.warning(
        "PCB1: BLOCKED — confirmed source overlap. PCB2: INCONCLUSIVE — "
        "56 similarity pairs remain unresolved; no independent human adjudication."
    )
    with st.expander("Sources and historical provenance"):
        st.write(
            "Packaged authority: model_cards/model-card.md (Milestone 2B). "
            "Source-distribution counterpart: docs/model-card.md. Historical run reference: "
            "runs/cashew-eval-v1/metrics.json; research assets are not included in this package."
        )
        st.caption(
            "Historical Cashew runtime: Python 3.12.15, torch 2.13.0+cpu, Windows AMD64. "
            "This provenance is separate from the current application runtime."
        )
        st.write(
            "Development sources: model_cards/feature-model-card.md and "
            "model_cards/student-model-card.md. Open Model Catalog for frozen artifact "
            "identities, model cards and the verified evidence index."
        )


def show_result(result, display=None):
    view = panels(result)
    with st.container(border=True, key="fv-card-result-" + result.model_id):
        st.subheader(result.model_name)
        from factoryvision.inference.model_catalog import METHODS

        st.caption(METHODS[result.model_id][0])
        presentation.result_images(view, display, key=result.model_id)
        presentation.legend(view["ceiling"])
        st.markdown("**Model diagnostics**")
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
        with st.expander("Score precision, timing and model identity"):
            st.write(f"Raw score: {result.score!r} · threshold: {result.threshold!r}")
            st.caption(
                f"Preprocess {result.preprocessing_ms:.1f} ms · artifact verification/loading "
                f"{result.artifact_check_load_ms:.1f} ms · total {result.total_ms:.1f} ms"
            )
            st.code(result.artifact_sha256, language=None)


def inspect(engine, statuses, compare=False):
    if not any(row["available"] for row in statuses):
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
                preview.thumbnail((1024, 1024))
                valid = True
            except InputRejected as exc:
                st.error(str(exc))
        st.button("Clear image and results", on_click=presentation.clear_inspection)
        rerun_notice = st.empty()
        if st.button(
            "Run inspection",
            type="primary",
            width="stretch",
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
        if compare and st.session_state.get("results"):
            st.markdown("[View comparison results](#comparison-results)")
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
    results = st.session_state.get("results", ())
    # Fill the pre-Run slot after execution so even the first result warns before another Run.
    if results:
        rerun_notice.warning(review_ui.RERUN_NOTICE)
    with preview_slot.container():
        if results and not compare:
            show_result(results[0])
        else:
            presentation.input_viewer(preview if valid else None)
    with output:
        if compare:
            if results:
                display = presentation.comparison_view()
                for offset in range(0, len(results), 2):
                    row = results[offset : offset + 2]
                    for column, result in zip(st.columns(len(row)), row, strict=True):
                        with column:
                            show_result(result, display)
            else:
                st.info(
                    "No model output yet. Upload a permitted image and explicitly run inspection."
                )
        review_ui.render("single", results or None, "cashew")


def render(engine, metal_engine=None):
    st.set_page_config(
        page_title="FactoryVision AI · Industrial Precision",
        page_icon="◈",
        layout="wide",
    )
    presentation.styling()
    # Keep navigation in document flow: collapsed native sidebars retain focusable widgets.
    layout = st.container(horizontal=True, gap="medium", key="fv-layout")
    navigation = layout.container(key="fv-nav")
    content = layout.container(key="fv-shell")
    navigation.subheader("FactoryVision AI")
    navigation.caption("Industrial Precision")
    navigation_notice = navigation.empty()
    category = navigation.selectbox("Category", ["Cashew", "MetalParts (experimental)"])
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
    page = navigation.radio(
        "Workspace", list(labels), format_func=labels.get, key="workspace_page", horizontal=True
    )
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
    navigation.divider()
    navigation.markdown("**Research prototype**")
    navigation.caption("Local CPU · No cloud inference")
    navigation.caption("PCB1: BLOCKED · PCB2: INCONCLUSIVE")
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
            "Synthetic Demo",
            "Original geometry. Validation only. No model inference.",
        ),
    }
    with content:
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
