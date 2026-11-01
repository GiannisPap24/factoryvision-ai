"""Experimental category presentation; no fitting, hidden inference or binary decisions."""

import hashlib
import json
from importlib.resources import files

import streamlit as st
import torch

from factoryvision.app import presentation, review_ui
from factoryvision.data.metalparts import MetalPartsSource
from factoryvision.inference.metalparts import DIAGNOSTIC_LABEL, MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.service import ModelUnavailable, verified_asset
from factoryvision.inference.uploads import InputRejected, decode_upload
from factoryvision.inference.visualization import panels


@st.cache_resource(max_entries=1, show_spinner=False)
def service(root):
    torch.set_num_threads(4)
    return MetalPartsService(root)


@st.cache_resource(max_entries=1, show_spinner=False)
def feature_service(root):
    torch.set_num_threads(4)
    return MetalPartsFeatureService(root)


def show_result(result, display=None):
    feature = result.model_id == "metalparts_feature_memory"
    st.subheader("Feature-Memory Detector" if feature else "Convolutional Autoencoder")
    view = panels(result)
    presentation.result_images(view, display, reference=False, key=result.model_id)
    presentation.legend(view["ceiling"])
    st.markdown("**Model diagnostics**")
    left, right = st.columns(2)
    left.metric(
        "Maximum feature distance" if feature else "Continuous reconstruction score",
        f"{result.score:.8f}",
    )
    right.metric("CPU inference", f"{result.inference_ms:.1f} ms")
    st.caption(
        result.score_label + ". No validated threshold or binary decision. "
        "Raw scores have different units across methods and cannot be compared directly."
    )
    st.caption(
        "Native map; display reduced to at most 1024 pixels per side. Blur, edges "
        "and tile seams can produce error unrelated to defects."
    )
    st.caption(
        f"Preprocess {result.preprocessing_ms:.1f} ms · artifact verification/loading "
        f"{result.artifact_check_load_ms:.1f} ms · total {result.total_ms:.1f} ms"
    )
    with st.expander("Verified MetalParts provenance"):
        st.write(
            "RobustAD MetalParts · frozen ResNet18 layer2/3 · 6,912 normal-source descriptors"
            if feature
            else "RobustAD MetalParts · normal-source conv-ae-v1 · native patches 256 · stride 192"
        )
        st.code(result.artifact_sha256, language=None)
        st.caption(f"Dataset revision: {MetalPartsSource().revision}")


def inspect(engine, compare=False):
    controls, output, preview_slot = presentation.inspection_layout(compare)
    image = None
    with controls:
        st.subheader("Inspection controls" if compare else "Inspection setup")
        st.caption("Category: MetalParts (experimental) · Change category in navigation")
        st.warning(
            "Experimental source-domain models. Autoencoder trained; feature memory fitted with "
            "a frozen backbone. Independent defect performance is unestablished; "
            "no binary threshold."
        )
        options = {
            "Convolutional Autoencoder": engine,
            "Feature-Memory Detector": feature_service(engine.root),
        }
        selected = st.multiselect(
            "MetalParts models",
            list(options),
            default=["Convolutional Autoencoder"],
            key="metal_models" if compare else "metal_single_model",
            max_selections=None if compare else 1,
        )
        st.caption(
            "Compare compatible methods using separately labelled map scales."
            if compare
            else "Select one model, upload a permitted image and run inspection."
        )
        upload = st.file_uploader(
            "Choose a MetalParts image",
            type=["png", "jpg", "jpeg"],
            max_upload_size=8,
            key=f"metal_upload_{st.session_state.get('metal_upload_generation', 0)}",
        )
        acknowledge = st.checkbox(
            "I confirm this is a permitted MetalParts image, not an official test image.",
            key="metal_ack",
        )
        payload = upload.getvalue() if upload is not None else None
        identity = (
            hashlib.sha256(payload).hexdigest() if payload else None,
            acknowledge,
            tuple(selected),
            compare,
        )
        if st.session_state.get("metal_request") != identity:
            st.session_state.pop("metal_result", None)
            st.session_state.pop("metal_results", None)
            st.session_state["metal_request"] = identity
        valid = False
        if payload is not None:
            try:
                image = decode_upload(payload)
                if min(image.size) < 256:
                    raise InputRejected(
                        "MetalParts requires at least 256 pixels on each side; "
                        "no automatic upscaling."
                    )
                image.thumbnail((1024, 1024))
                valid = True
            except InputRejected as exc:
                st.error(str(exc))
        statuses = [options[name].status() for name in selected]
        st.button("Clear MetalParts image and results", on_click=presentation.clear_inspection)
        if st.button(
            "Run MetalParts inspection",
            type="primary",
            width="stretch",
            disabled=not (
                valid and acknowledge and selected and all(s["available"] for s in statuses)
            ),
        ):
            st.session_state.pop("metal_result", None)
            st.session_state.pop("metal_results", None)
            try:
                with st.spinner(
                    "Verifying frozen MetalParts artifacts and scoring native patches…"
                ):
                    results = tuple(options[name].inspect(payload) for name in selected)
                    st.session_state["metal_results"] = results
                    st.session_state["metal_result"] = results[0]
            except (InputRejected, ModelUnavailable) as exc:
                st.error(str(exc))
            except (RuntimeError, MemoryError):
                presentation.inference_failure()
        if compare and st.session_state.get("metal_results"):
            st.markdown("[View comparison results](#comparison-results)")
        for status in statuses:
            if not status["available"]:
                st.error(status["reason"])
        if any(not status["available"] for status in statuses):
            presentation.setup_help()
        with st.expander("Permitted inputs and limits"):
            st.caption(
                "JPEG / PNG · ≤8 MiB · 256–4096 pixels per side · ≤16M pixels · "
                "opaque, upright, single-frame. Uploads remain in memory."
            )
            st.caption(
                "Category is not detected automatically. Official test0–test6 images and "
                "transformed copies are prohibited."
            )
    results = st.session_state.get("metal_results", ())
    with preview_slot.container():
        if results and not compare:
            show_result(results[0])
        else:
            presentation.input_viewer(image if valid else None)
    with output:
        if compare:
            if results:
                display = presentation.comparison_view()
                for column, result in zip(st.columns(len(results)), results, strict=True):
                    with column.container(border=True, key="fv-card-result-" + result.model_id):
                        show_result(result, display)
            else:
                st.info(
                    "No model output yet. Upload a permitted image and explicitly run inspection."
                )
        review_ui.render("single", results or None, "metalparts")


def research(root):
    comparison(root)
    with st.expander("Earlier autoencoder training and development diagnostics"):
        development_history(root)
    st.caption("PCB1: BLOCKED · PCB2: INCONCLUSIVE · No independent target evaluation is cleared.")


def development_history(root):
    st.subheader("MetalParts source-development observations")
    st.warning(DIAGNOSTIC_LABEL)
    source = MetalPartsSource()
    st.markdown(f"[AmazonScience/RobustAD source]({source.uri}) · **{source.license_id}**")
    st.caption(source.attribution)
    st.caption(f"Pinned revision: {source.revision}")
    st.write(
        "The frozen model learned from 144 normal source images, with 36 normal "
        "images used for checkpoint selection. Four CPU epochs, 864 optimizer "
        "updates, native 256-pixel patches; Adam 0.001, batch eight, seed 42."
    )
    try:
        summary = json.loads(
            files("factoryvision.inference").joinpath("metalparts_research.json").read_text()
        )
    except (OSError, ValueError):
        st.info("Local research summary is unavailable.")
        return
    st.caption(f"Untrained validation MSE: {summary['training']['initial_validation_loss']}")
    st.caption(
        "Recorded normal reconstruction losses, not detection accuracy. "
        "Scroll tables for full precision."
    )
    presentation.table(
        [
            {
                "Epoch": r["epoch"],
                "Fitting MSE": str(r["train_loss"]),
                "Validation MSE": str(r["validation_loss"]),
            }
            for r in summary["training"]["epochs"]
        ],
        "research-metal-training",
    )
    metrics = summary["metrics"]
    presentation.table(
        [
            {
                "Level": name,
                "AUROC": str(metrics[name]["auroc"]),
                "Average precision": str(metrics[name]["average_precision"]),
                "Positive": metrics[name]["positive"],
                "Negative": metrics[name]["negative"],
                "Prevalence": str(metrics[name]["prevalence"]),
            }
            for name in ("image", "pixel")
        ],
        "research-metal-development",
    )
    st.caption(
        "Pixel metrics include every native pixel using 65,536 fixed bins: exact for "
        "quantized scores, approximate for unquantized rankings. No threshold was "
        "optimized."
    )
    st.warning(
        "The 36 comparison normals already influenced model selection. Source "
        "anomalies may depict related physical objects. Images and pixels are "
        "dependent; these descriptive results do not establish shifted-domain "
        "performance. Independent MetalParts evaluation remains NOT CLEARED."
    )
    st.write(
        "Reconstruction error measures mismatch with a normal-trained model. High "
        "error may reflect defects, lighting, geometry, blur or tile seams. A "
        "normal-trained autoencoder can also reconstruct defects. There is no "
        "validated binary threshold."
    )
    st.caption(
        "Four examples were selected by stable sample ID before scoring (one "
        "validation normal and three source anomalies). Third-party pixels remain "
        "local and are absent from the package."
    )
    for example in summary["previews"]:
        with st.expander(
            ("Source anomaly" if example["anomaly"] else "Validation normal")
            + " · "
            + example["sample_id"][-12:]
        ):
            st.caption(
                f"Continuous score {example['score']:.8f} · "
                f"relative heatmap ceiling {example['ceiling']:.6g}"
            )
            try:
                assets = {
                    key: verified_asset(root, entry) for key, entry in example["assets"].items()
                }
                for label in ("original", "reconstruction", "heatmap", "overlay", "mask"):
                    st.image(assets[label].read_bytes(), caption=label.title(), width="stretch")
            except (OSError, ModelUnavailable):
                st.info(
                    "This local development visualization is unavailable or failed verification."
                )


def render(page, engine):
    if page == "Image inspection":
        inspect(engine)
    elif page == "Research results":
        research(engine.root)
    else:
        st.subheader("Experimental MetalParts inspection")
        st.warning(
            "Experimental source-domain models. Autoencoder trained; feature memory fitted "
            "with a frozen backbone. Independent defect performance is unestablished."
        )
        status = engine.status()
        (st.success if status["available"] else st.error)(status["reason"])
        feature_status = feature_service(engine.root).status()
        (st.success if feature_status["available"] else st.error)(feature_status["reason"])
        presentation.steps()
        if not (status["available"] and feature_status["available"]):
            presentation.setup_help()
        st.write(
            "Open Image inspection, select models and upload a permitted image. Choose the "
            "Convolutional Autoencoder, Feature-Memory Detector or both, then press "
            "Run MetalParts inspection. Both methods cover the full native image."
        )
        st.info(
            "Continuous method-specific scores only. No validated defect threshold. "
            "Cashew models remain available through the category selector; raw scores "
            "are not comparable across categories."
        )
        st.caption(
            "Independent MetalParts evaluation: NOT CLEARED · PCB1: BLOCKED · PCB2: INCONCLUSIVE"
        )


def comparison(root):
    st.subheader("Recorded MetalParts comparison")
    try:
        summary = json.loads(
            files("factoryvision.inference")
            .joinpath("metalparts_comparison.json")
            .read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        st.info("Local feature-memory comparison is unavailable.")
        return
    with st.container(border=True, key="fv-card-research-status", gap="small"):
        st.markdown("**EXPLORATORY · Independent target evaluation NOT CLEARED**")
        st.warning(summary["label"])
        st.write(
            "87 source-domain images: 36 validation normals and 51 previously observed "
            "anomalies. The normals already influenced checkpoint selection. Physical-item "
            "and capture independence is unresolved. No binary threshold exists."
        )
    with st.container(border=True, key="fv-card-research-ranking", gap="small"):
        st.subheader("Observed ranking and localization")
        st.caption("Historical values · AP means average precision · No independent-winner claim.")
        presentation.table(
            {
                "Metric": ["Image AUROC", "Image AP", "Pixel AUROC", "Pixel AP"],
                **{
                    label: [
                        str(summary[method][level][metric])
                        for level in ("image", "pixel")
                        for metric in ("auroc", "average_precision")
                    ]
                    for method, label in (
                        ("autoencoder", "Autoencoder"),
                        ("feature_memory", "Feature memory"),
                    )
                },
            },
            "research-metal-ranking",
        )
        st.caption("Full recorded precision. On narrow screens, scroll the table horizontally.")
        st.write(
            "Feature memory ranks this observed set better in aggregate, but localization "
            "AP is lower on 10 of 51 anomalous images. Both methods place their maximum "
            "outside the defect mask on 22 images. "
            "Aggregate rankings do not remove shared failures."
        )
    with st.expander("Methodology and interpretation", expanded=True):
        st.write(
            "Both methods use the same 144 normal fitting images. The autoencoder learns "
            "normal reconstruction; feature memory uses a frozen ResNet18 backbone and "
            "6,912 normal-source descriptors, with no gradient training. Raw scores use "
            "different units and aggregation; they cannot directly rank methods."
        )
        st.write(
            "AUROC measures ranking across thresholds; AP summarizes precision and recall. "
            "All 349,870,500 native pixels are included, with 84,903 positives. Pixel "
            "localization has severe background imbalance. Positive prevalence: 58.62% "
            "of images, 0.02427% of pixels (rounded constant-score AP references)."
        )
        st.caption(
            "Native pixel rankings use 65,536 bins with fixed width 1/65535, no clipping "
            "or per-image normalization. Exact for quantized scores; approximate for "
            "unquantized rankings. Quantization tie frequency differs across methods."
        )
    with st.expander("Uncertainty and scientific limits", expanded=True):
        st.warning(
            "Source images may share physical objects. Normals influenced AE checkpoint "
            "selection; anomalies were already observed. Unresolved official train/test "
            "similarity candidates remain. Independent target evaluation: NOT CLEARED."
        )
        st.write(
            "Dependent images and pixels do not support independent confidence intervals. "
            "No official test0–test6 payload was opened for this comparison. Source-domain "
            "ranking does not establish shifted-domain performance. Blur, lighting, edges "
            "and tile seams can dominate scores; defects can also be reconstructed well."
        )
    with st.expander("Provenance and recorded runtime"):
        source = MetalPartsSource()
        st.markdown(f"[AmazonScience/RobustAD source]({source.uri}) · **{source.license_id}**")
        st.caption(source.attribution)
        st.caption(f"Pinned revision: {source.revision}")
        st.write(
            "Sources: metalparts_comparison.json (Milestone 3F), metalparts_research.json "
            "(earlier 3E diagnostics), model_cards/metalparts-model-card.md and "
            "model_cards/metalparts-feature-model-card.md. Open Model Catalog for frozen "
            "artifact, split and protocol identities."
        )
        st.warning(
            "Separate historical runs; feature comparison overlapped local verification "
            "work. These timings are not a controlled speed comparison or live-line throughput."
        )
        presentation.table(
            [
                {
                    "Method": label,
                    "Median CPU seconds": str(summary[method]["latency_seconds"]["median"]),
                    "p95 CPU seconds": str(summary[method]["latency_seconds"]["p95"]),
                }
                for method, label in (
                    ("autoencoder", "Autoencoder"),
                    ("feature_memory", "Feature memory"),
                )
            ],
            "research-metal-timing",
        )
    st.subheader("Historical visual diagnostics")
    st.info(
        "Research images are not included in the source package. Local examples appear "
        "only when their existing assets pass verification; no model output is generated here."
    )
    st.caption(
        "Below are the same four examples selected before scoring. Method-specific "
        "heatmap ceilings use different units; raw values are not directly comparable."
    )
    historical = json.loads(
        files("factoryvision.inference").joinpath("metalparts_research.json").read_text()
    )
    originals = {r["sample_id"]: r for r in historical["previews"]}
    for example in summary["preview_assets"]:
        with st.expander("Comparison example · " + example["sample_id"][-12:]):
            old = originals[example["sample_id"]]
            left, right = st.columns(2)
            try:
                for column, record, label, units in [
                    (left, old, "Autoencoder", "MSE"),
                    (right, example, "Feature memory", "descriptor RMS distance"),
                ]:
                    column.caption(f"{label} · heatmap 0–{record['ceiling']:.6g} {units}")
                    path = verified_asset(root, record["assets"]["overlay"])
                    column.image(path.read_bytes(), caption=label, width="stretch")
            except (OSError, ModelUnavailable):
                st.info("Local comparison image unavailable or failed verification.")
