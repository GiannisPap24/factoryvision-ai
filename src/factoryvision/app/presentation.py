"""Shared local presentation and session lifecycle; no model or dataset operations."""

import streamlit as st


def styling():
    # Static CSS scoped to our documented st.container(key=...) classes only.
    # No untrusted interpolation, external assets or Streamlit internal selectors.
    st.html("""<style>
    .st-key-fv-shell {max-width: 1440px; margin: 0 auto;}
    .st-key-fv-shell h1 {letter-spacing: -.035em; line-height: 1.15;}
    .st-key-fv-shell h2, .st-key-fv-shell h3 {letter-spacing: -.02em;}
    .st-key-fv-nav label {padding: .4rem .5rem; border-radius: .375rem;}
    .st-key-fv-nav label:has(input:checked) {background: #1E40AF;}
    .st-key-fv-nav label:focus-within {outline: 3px solid #60A5FA; outline-offset: 2px;}
    .st-key-fv-shell [class*="st-key-fv-card-"] {background: #FFFFFF; border-radius: .5rem;}
    .st-key-fv-shell img {border-radius: .5rem;}
    .st-key-fv-shell label p {white-space: normal; overflow: visible; text-overflow: clip;}
    .st-key-fv-shell [class*="st-key-fv-table"] {overflow-x: auto;}
    .st-key-fv-shell table {font-size: .9rem;}
    .st-key-fv-shell button:focus-visible, .st-key-fv-nav input:focus-visible {
        outline: 3px solid #2563EB; outline-offset: 3px;
    }
    .st-key-fv-shell .fv-legend {height: .65rem; display: flex; border-radius: .25rem;
        overflow: hidden; max-width: 22rem; margin: .5rem 0;}
    .st-key-fv-shell .fv-legend span {flex: 1;}
    @media (max-width: 600px) {
        .st-key-fv-shell h1 {font-size: 1.7rem;}
        .st-key-fv-shell table {font-size: .8rem;}
    }
    </style>""")


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
    with st.container(horizontal=True, gap="small"):
        st.button("Start inspection", type="primary", on_click=navigate, args=("Image inspection",))
        st.button("Compare models", on_click=navigate, args=("Compare",))
        st.button("Open demo", on_click=navigate, args=("Synthetic demo",))
    for column, label, value in zip(
        st.columns(4),
        (
            "Object categories",
            "Registered methods",
            "Ready for " + category.split(" ")[0],
            "Execution",
        ),
        (2, 6, ready, "Local CPU"),
        strict=True,
    ):
        with column.container(key="fv-card-" + label.replace(" ", "-")):
            st.metric(label, value, border=True)
    left, right = st.columns([1.6, 1], gap="medium")
    with left.container(border=True, key="fv-card-workflow"):
        st.subheader("Your inspection workflow")
        for number, title, body in (
            ("01", "Select an image", "Choose a permitted image and a compatible model."),
            ("02", "Run a frozen model", "Start CPU inference explicitly. Inputs stay local."),
            ("03", "Review spatial evidence", "Inspect aligned maps and model-specific scores."),
        ):
            st.markdown(f"**{number} · {title}**")
            st.caption(body)
    with right.container(border=True, key="fv-card-status"):
        st.subheader("Research status")
        st.markdown(
            "**Cashew** · Historical evaluation\n\n"
            "**MetalParts** · Exploratory only\n\n"
            "**PCB1: BLOCKED** · Source overlap\n\n"
            "**PCB2: INCONCLUSIVE** · Adjudication pending"
        )
    with st.container(border=True, key="fv-card-readiness"):
        st.subheader("Local model readiness")
        availability(statuses)
    st.warning(
        "Research demonstration. Not validated for industrial acceptance, "
        "food safety or automated product-quality decisions."
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
        st.error(
            "Some registered models are unavailable. Restore verified local artifacts to inspect."
        )
        with st.expander("Artifact verification details"):
            for row in statuses:
                if not row["available"]:
                    st.write(f"{row['name']}: {row['reason']}")
        setup_help()


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
