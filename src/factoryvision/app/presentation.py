"""Shared local presentation and session lifecycle; no model or dataset operations."""

from importlib.resources import files

import streamlit as st

from factoryvision.app.typography import font_styles

SYNTHETIC_IMAGE_ALTS = (
    "Teal rectangle on a dark background",
    "Yellow circle on a dark background",
)


def styling():
    # Packaged, constant CSS/font bytes only; no user content or JavaScript.
    st.html(
        font_styles()
        + files("factoryvision.app").joinpath("workbench.css").read_text(encoding="utf-8")
    )


def header():
    with st.container(key="fv-app-header"):
        st.html(
            '<header class="fv-workbench-header">'
            "<strong>FactoryVision AI</strong><span>Industrial Precision</span>"
            "<small>Research · Local CPU</small></header>"
        )


def page_header(title, description, category):
    with st.container(key="fv-page-header", gap="xxsmall"):
        st.title(title)
        st.caption(f"{category} · {description}")


@st.dialog("Expanded image", width="large", on_dismiss="ignore")
def _expanded_image(image, caption, alt):
    # Native modal owns trapping, Escape/Close and delayed focus restoration.
    # Streamlit disables its image fullscreen toolbar inside dialogs.
    with st.container(key="fv-expanded-image", gap="small"):
        st.image(image, caption=caption, alt=alt, width="stretch")
        st.caption("Press Escape or Close to return to the image control.")


def image_view(image, *, caption, alt, key, width="stretch", action=None):
    """Retain native image semantics with a focus-restoring expanded view."""
    with st.container(key="fv-image-" + key, gap="xsmall", width=width):
        st.image(image, caption=caption, alt=alt, width=width)
        if st.button(
            action or "Expand: " + alt,
            key="fv-expand-" + key,
            help="Open a larger image. Escape or Close returns focus here.",
            type="tertiary",
        ):
            _expanded_image(image, caption, alt)


def input_viewer(image=None, *, refined=False):
    """Display already validated RGB pixels; never call an inference service."""
    if image is None:
        st.markdown("[Go to Inspection setup and upload](#inspection-setup)")
    with st.container(key="fv-input-viewer"):
        st.markdown("**Input preview**")
        if image is None and refined:
            st.html(
                '<div class="fv-empty-viewer fv-empty-inspect">'
                '<p class="fv-viewer-kicker">IMAGE WORKSPACE</p>'
                "<h2>Start with an image</h2>"
                "<p>Upload a permitted PNG or JPEG in Inspection setup. "
                "Your validated image will appear here.</p>"
                '<p class="fv-viewer-note">Preview works without model files. '
                "No inference runs until you choose Run inspection.</p></div>"
            )
        elif image is None:
            st.html(
                '<div class="fv-empty-viewer"><h2>Your image belongs here.</h2>'
                "<p>Choose a permitted image in Inspection setup. A validated preview "
                "is separate from inference.</p><p>No model output, score or anomaly map.</p></div>"
            )
        else:
            image_view(
                image,
                caption="Validated RGB · preview only",
                alt="Input image",
                key="input-preview",
            )
        st.caption("Original RGB before inference · Session only")


def inspection_layout(compare):
    # One native tree for both workflows; styling never changes widget identity.
    layout = st.container(horizontal=True, key="fv-inspection-layout", gap="medium")
    viewer = layout.container(key="fv-inspection-evidence", gap="xsmall")
    controls = layout.container(border=True, key="fv-inspection-setup", gap="small")
    # A transient empty placeholder unmounts the opener during a button rerun,
    # leaving the native dialog without a connected element to restore focus to.
    preview = viewer.container(key="fv-inspection-preview", gap="xsmall")
    return controls, st.container(key="fv-inspection-output"), preview


VIEW_LABELS = ("Original", "Heatmap", "Overlay", "Reconstruction / reference")


def result_images(view, display=None, *, reference=True, key="result"):
    """Present existing display pixels; no scoring, normalization or image edits."""

    def show(label):
        if label == "Reconstruction / reference":
            if view["reconstruction"] is not None:
                image_view(
                    view["reconstruction"],
                    caption="Reconstructed RGB",
                    alt="Model RGB reconstruction",
                    key=key + "-reconstruction",
                )
            elif reference and view["reference"] is not None:
                image_view(
                    view["reference"],
                    caption="Frozen spatial mean reference",
                    alt="Frozen spatial reference image",
                    key=key + "-reference",
                )
            else:
                st.html(
                    '<div class="fv-unavailable-spatial">'
                    "<strong>No RGB reconstruction</strong>"
                    "<p>This model produces anomaly features, not an RGB reconstruction.</p>"
                    "</div>"
                )
                st.caption("Reconstruction unavailable for this method")
        else:
            name, caption, alt = {
                "Original": ("original", "Original RGB", "Input image"),
                "Heatmap": ("heatmap", "Relative anomaly intensity", "Relative anomaly map"),
                "Overlay": (
                    "overlay",
                    "45% heatmap overlay",
                    "Input image with relative anomaly map",
                ),
            }[label]
            image_view(view[name], caption=caption, alt=alt, key=key + "-" + name)

    with st.container(key="fv-spatial-viewer-" + key, gap="xsmall"):
        if display is None:
            for tab, label in zip(st.tabs(VIEW_LABELS), VIEW_LABELS, strict=True):
                with tab:
                    show(label)
        else:
            show(display)


def comparison_view():
    with st.container(key="fv-comparison-toolbar", gap="xsmall"):
        st.subheader("Side-by-side evidence", anchor="comparison-results")
        st.caption(
            "Same input · Selection order · Each map has its own relative scale. "
            "Raw scores use different units and cannot rank models."
        )
        return st.radio("View for all models", VIEW_LABELS, index=2, horizontal=True)


def comparison_panels(results, render_result):
    """Keep complete method panels in selection order, with at most two per row."""
    display = comparison_view()
    with st.container(horizontal=True, key="fv-comparison-grid", gap="medium"):
        for result in results:
            with st.container(key="fv-comparison-method-" + result.model_id):
                render_result(result, display)


def diagnostics(values, key):
    """Native metrics with enough room for full values at every panel width."""
    with st.container(border=True, key="fv-diagnostics-" + key, gap="xsmall"):
        st.markdown("**Model diagnostics**")
        with st.container(horizontal=True, key="fv-metrics-" + key, gap="small"):
            for index, (label, value) in enumerate(values):
                with st.container(key=f"fv-metric-{key}-{index}"):
                    st.metric(label, value)


def table(data, key):
    with st.container(key="fv-table-" + key):
        st.table(data)


def legend(ceiling):
    st.html(
        '<div class="fv-legend" aria-hidden="true">'
        '<span style="background:#FFF7BC"></span>'
        '<span style="background:#FEC44F"></span>'
        '<span style="background:#D95F0E"></span>'
        '<span style="background:#993404"></span></div>'
    )
    st.caption(
        f"Pale yellow 0 → amber → dark orange {ceiling:.6g} raw map intensity. "
        "Per-image maximum normalization; not a defect probability. "
        "Image geometry is preserved."
    )


def navigate(page):
    st.session_state["workspace_page"] = page


def clear_inspection():
    from factoryvision.app import review_ui

    review_ui.clear("single")
    for key in (
        "results",
        "request_identity",
        "metal_result",
        "metal_results",
        "metal_request",
        "cashew_ack",
        "metal_ack",
    ):
        st.session_state.pop(key, None)
    st.session_state["cashew_ack"] = False
    st.session_state["metal_ack"] = False
    for prefix in ("upload", "metal_upload"):
        generation = st.session_state.get(prefix + "_generation", 0)
        st.session_state.pop(f"{prefix}_{generation}", None)
        st.session_state[prefix + "_generation"] = generation + 1


def overview(category, statuses):
    ready = sum(row["available"] for row in statuses)
    layout = st.container(horizontal=True, key="fv-overview-layout", gap="medium")
    with layout.container(border=True, key="fv-card-start", gap="small"):
        st.caption("LOCAL INSPECTION WORKBENCH · " + category)
        st.subheader("From image to evidence")
        st.write("Try validation, Human Review and exports with original synthetic examples.")
        with st.container(horizontal=True, gap="small", key="fv-overview-actions"):
            st.button(
                "Open synthetic demo",
                type="primary",
                on_click=navigate,
                args=("Synthetic demo",),
            )
            st.button("Open Model Catalog", on_click=navigate, args=("Model Catalog",))
        if ready:
            with st.container(horizontal=True, gap="small"):
                st.button(
                    "Start inspection",
                    on_click=navigate,
                    args=("Image inspection",),
                )
                st.button("Compare models", on_click=navigate, args=("Compare",))
        if not ready:
            st.caption(
                "Missing weights are expected in a fresh checkout. The demo performs no "
                "neural inference and produces no anomaly scores."
            )
        else:
            st.caption(f"{ready} verified local methods available for {category}.")
        from factoryvision.demo import build_demo

        # Original thumbnails only. Do not create a Demo session or review/export bundle.
        with st.container(horizontal=True, gap="small", key="fv-sample-strip"):
            for number, image in enumerate(build_demo().images, 1):
                image_view(
                    image,
                    caption=f"Synthetic example {number}",
                    width=96,
                    alt=SYNTHETIC_IMAGE_ALTS[number - 1],
                    key=f"overview-{number}",
                    action=f"Expand {number}",
                )
        st.caption("Original geometry · No dataset or model weights required")
    with layout.container(border=True, key="fv-card-workflow", gap="small"):
        st.subheader("Local model readiness")
        st.markdown(f"**{ready} of {len(statuses)} local methods ready · {category}**")
        st.caption(
            "Verified artifacts are required for inference. Synthetic Demo is always available."
        )
        with st.expander("Local model readiness", expanded=bool(ready)):
            st.caption(
                "Six registered methods across two categories. Registration is not readiness."
            )
            availability(statuses)
        st.markdown("**Source-only installation**")
        st.write(
            "Preview permitted images, explore model provenance and review recorded research. "
            "Model weights and datasets are separate local assets."
        )
    with st.container(horizontal=True, key="fv-workflow-grid", gap="medium"):
        for key, title, description, action, page in (
            (
                "inspect",
                "Inspect",
                "One image, one method. Preview before explicit inference.",
                "Open Inspect",
                "Image inspection",
            ),
            (
                "compare",
                "Compare",
                "One input, multiple methods. Review independently scaled maps.",
                "Open Compare",
                "Compare",
            ),
            (
                "batch",
                "Batch Inspection",
                "Validate a bounded selection. Review every outcome.",
                "Open Batch",
                "Batch inspection",
            ),
        ):
            with st.container(border=True, key="fv-card-workflow-" + key, gap="xsmall"):
                st.subheader(title)
                st.caption(description)
                st.button(action, on_click=navigate, args=(page,), width="stretch")
    with st.expander("First-session guide"):
        for number, title, body in (
            ("01", "Generate examples", "Open Demo, then generate the synthetic inputs."),
            ("02", "Review validation", "See valid and rejected inputs. No defect prediction."),
            (
                "03",
                "Try review and export",
                "Practice Human Review and download synthetic JSON/CSV.",
            ),
        ):
            with st.container(key="fv-workflow-step-" + number, gap="xxsmall"):
                st.markdown(f"**{number} · {title}**")
                st.write(body)
    with st.container(key="fv-overview-boundaries", gap="xsmall"):
        st.markdown("**Research boundaries**")
        st.write(
            "Research demonstration. Not validated for industrial acceptance, "
            "food safety or automated product-quality decisions."
        )
        st.caption("PCB1: BLOCKED · PCB2: INCONCLUSIVE · RobustAD: NOT CLEARED")
        st.caption(
            "MetalParts: exploratory-only. Cashew official test already observed. "
            "Human Review is not verified ground truth. Real Cashew pilot: BLOCKED."
        )


def setup_help():
    with st.expander("Model files missing? Local setup guide", expanded=False):
        st.write(
            "The source package includes the interface and recorded research summaries. "
            "Datasets, trained checkpoints and pretrained weights are separate local assets. "
            "An unavailable model cannot run; this app never downloads or trains automatically."
        )
        st.markdown(
            "**Existing research workspace:** launch from its root so the approved relative "
            "artifact paths resolve. Restore the verified artifacts if missing or damaged. "
            "Renaming another checkpoint will not pass the identity checks.\n\n"
            "**Fresh checkout:** read `docs/local-demo.md` for installation and "
            "`docs/reproduction.md` for licensed-data prerequisites and guarded training commands. "
            "New experiments require their own provenance review and explicit model registration; "
            "they do not replace the frozen demo artifacts automatically."
        )


def availability(statuses):
    table(
        [
            {
                "Model": row["name"],
                "Local availability": "Ready" if row["available"] else "Unavailable",
            }
            for row in statuses
        ],
        "availability",
    )
    if any(not row["available"] for row in statuses):
        st.caption(
            "Unavailable methods need their exact verified local artifacts. "
            "Registration describes a method; it does not make it ready for inference."
        )
        with st.expander("Artifact verification details"):
            for row in statuses:
                if not row["available"]:
                    st.write(f"{row['name']}: {row['reason']}")
        setup_help()


def navigation_notice(state):
    """Describe existing cleanup without retaining data or reading review content."""
    if state.get("synthetic_demo") is not None:
        return (
            "Changing workspace or category clears the synthetic demo and its Human Review "
            "notes, including unsaved edits. Download any exports first."
        )
    plan = state.get("batch_plan")
    if state.get("batch_report") is not None or (plan is not None and plan.items):
        return (
            "Changing category or leaving Batch Inspection and Reports clears batch inputs, "
            "results and Human Review notes. A completed batch and saved reviews are retained "
            "between those two workspaces. Save and export notes first; unsaved edits may be lost."
        )
    uploads = any(
        state.get(f"{prefix}_{state.get(prefix + '_generation', 0)}")
        for prefix in ("upload", "metal_upload", "batch_upload")
    )
    if (
        uploads
        or any(state.get(key) is not None for key in ("results", "metal_result", "metal_results"))
        or any(key.startswith("fv_review_") for key in state)
    ):
        return (
            "Changing workspace or category clears the current image, results and Human Review "
            "notes, including unsaved edits. Save and export notes before navigating."
        )
    return None


def steps():
    for column, title, body in zip(
        st.columns(3),
        ("01 · Choose", "02 · Inspect", "03 · Compare"),
        (
            "Select a category and compatible local models.",
            "Upload a permitted image, then explicitly run CPU inference.",
            "Review maps, model-specific scores and measured latency.",
        ),
        strict=True,
    ):
        with column.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(body)


def switch_category(category):
    if st.session_state.get("active_category") == category:
        return
    from factoryvision.app import review_ui

    review_ui.clear()
    st.session_state["batch_ack"] = False
    for key in (
        "results",
        "request_identity",
        "metal_result",
        "metal_results",
        "metal_request",
        "batch_plan",
        "batch_report",
        "batch_exports",
        "synthetic_demo",
        "batch_signature",
        "batch_started",
        "batch_detail",
        "batch_budget_error",
        "batch_archive_view",
        "cashew_ack",
        "metal_ack",
    ):
        st.session_state.pop(key, None)
    st.session_state["cashew_ack"] = False
    st.session_state["metal_ack"] = False
    # New widget identities release references to previous category uploads.
    for prefix in ("upload", "metal_upload", "batch_upload"):
        generation = st.session_state.get(f"{prefix}_generation", 0)
        st.session_state.pop(f"{prefix}_{generation}", None)
        st.session_state[f"{prefix}_generation"] = generation + 1
    st.session_state["active_category"] = category


def inference_failure():
    # Do not render exception text: tensor/backend errors may contain private paths.
    st.error(
        "CPU inference could not finish. Clear the image and retry with a smaller supported "
        "image or one model at a time. If it persists, restart the local app and verify the "
        "original model artifacts. No result was retained for this attempt."
    )
