"""Source-only registry details within the existing Model Catalog workspace."""

from dataclasses import asdict

import streamlit as st

from factoryvision.app import presentation
from factoryvision.inference.model_registry import STATUSES, experiment_index, json_bytes


def render_provenance(record):
    with st.expander("Version and recorded provenance", expanded=False):
        st.caption("Registry schema 1 · Artifact version is content-addressed")
        st.code(record.artifact_version, language=None)
        experiment = record.experiment
        st.text("Historical run ID: " + (experiment.run_id or "Unavailable"))
        st.text("Dataset: " + experiment.dataset + " / " + experiment.category)
        st.text("Dataset revision: " + (experiment.dataset_revision or "Unavailable"))
        st.text("Partition identity: " + experiment.partition_scope)
        st.code(experiment.partition_sha256 or "Unavailable", language=None)
        st.text(experiment.configuration_provenance)
        st.text("Canonical configuration fingerprint: Unavailable")
        st.text("Recipe file SHA-256: " + (experiment.recipe_file_sha256 or "Unavailable"))
        st.text("Protocol file SHA-256: " + (experiment.protocol_sha256 or "Unavailable"))
        st.text("Recorded execution: " + (experiment.execution_provenance or "Unavailable"))
        st.text("Execution timestamp: Unavailable; documentation dates are not run timestamps.")
        if record.catalog.definition.pretrained_sha256:
            st.text(
                "Component family: Frozen pretrained feature extractor. "
                "This component is distinct from the fitted memory or trained student."
            )
        st.download_button(
            "Download registry metadata (JSON)",
            json_bytes(record),
            file_name="factoryvision-registry-" + experiment.model_id + ".json",
            mime="application/json",
            on_click="ignore",
        )


def render_history():
    st.subheader("Historical experiment evidence")
    st.caption(
        "Six model summaries and two integrity reviews. Evidence remains available "
        "when local checkpoints are absent. No experiment is executed here."
    )
    st.info(
        "Metrics retain their original dataset and protocol. This is not a cross-dataset "
        "leaderboard or a production-readiness ranking. Human reviews are observations, "
        "not verified ground-truth labels."
    )
    category = st.selectbox(
        "Evidence category",
        ("All", "cashew", "metalparts", "pcb1", "pcb2"),
        key="registry_category",
    )
    status = st.selectbox("Evidence status", ("All", *STATUSES), key="registry_status")
    rows = tuple(
        e
        for e in experiment_index()
        if (category == "All" or e.category == category) and (status == "All" or e.status == status)
    )
    if not rows:
        st.info("No evidence matches these filters.")
        return
    options = {e.evidence_id: e for e in rows}
    selected = st.selectbox("Experiment evidence", tuple(options), key="registry_evidence")
    experiment = options[selected]
    st.text("Scientific status: " + experiment.status)
    st.text(experiment.evaluation_protocol)
    st.caption("Evidence ID identifies this summary; a run ID is shown only when recorded.")
    st.text("Recorded run ID: " + (experiment.run_id or "Unavailable"))
    st.text(experiment.partition_scope)
    if experiment.metrics:
        presentation.table(
            [
                {
                    "Metric": m.name,
                    "Value": str(m.value),
                    "Scope": m.scope,
                    "Evidence source": m.evidence.resource,
                    "Source location": m.locator,
                }
                for m in experiment.metrics
            ],
            "registry_metrics",
        )
        st.caption(
            "Values are transcribed from accepted package evidence without recomputation. "
            "Cashew values retain the model card's published precision. Other endpoints "
            "and limitations remain in the associated model cards."
        )
    else:
        st.warning(experiment.missing_metrics_reason)
    with st.expander("Evidence sources and fingerprints"):
        for source in experiment.sources:
            st.text(source.resource)
            st.code(source.sha256, language=None)
        st.json(asdict(experiment), expanded=False)
