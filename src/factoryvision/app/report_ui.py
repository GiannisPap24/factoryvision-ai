"""Session-local presentation of validated scalar reports and in-memory exports."""

from dataclasses import dataclass, field

import streamlit as st

from factoryvision.app import presentation
from factoryvision.inference.batch import BatchReport
from factoryvision.inference.reporting import (
    LIMITATIONS,
    TIMING_SCOPE,
    ReportInvalid,
    ResearchReport,
    create_report,
    csv_bytes,
    json_bytes,
)


@dataclass(frozen=True)
class Exports:
    source: BatchReport = field(repr=False)
    report: ResearchReport
    csv: bytes = field(repr=False)
    json: bytes = field(repr=False)


def render(source: BatchReport) -> bool:
    try:
        bundle = st.session_state.get("batch_exports")
        if not isinstance(bundle, Exports) or bundle.source is not source:
            report = create_report(source)
            bundle = Exports(source, report, csv_bytes(report), json_bytes(report))
            st.session_state["batch_exports"] = bundle
        report = bundle.report
        report.validate()
    except ReportInvalid:
        st.session_state.pop("batch_exports", None)
        st.error("Report validation failed. Downloads are unavailable; clear the batch and retry.")
        return False
    st.subheader("Quality Control Research Report")
    st.caption("Research inspection summary · Schema v1 · No industrial PASS/FAIL decision")
    if report.execution_status == "completed":
        st.success("Batch complete. Inspection does not establish product quality.")
    elif report.execution_status == "stopped":
        st.error("Batch stopped. Failed and skipped images have no anomaly decision.")
    else:
        st.warning("Batch complete with issues. Review rejected and failed images below.")
    st.write(f"Category: {report.model.category} · Model: {report.model.name}")
    st.caption("Report: " + report.report_id)
    count = report.summary
    for column, label, value in zip(
        st.columns(4),
        ("Inspected", "Rejected / failed", "Skipped", "Batch processing"),
        (
            count.inspected,
            count.rejected + count.failed,
            count.skipped,
            f"{report.batch_duration_ms / 1000:.3f} s",
        ),
        strict=True,
    ):
        column.metric(label, value, border=True)
    st.caption(
        f"Selected: {count.selected} · Accepted inputs: {count.accepted} · "
        f"Rejected: {count.rejected} · Failed: {count.failed}. "
        "Accepted means upload validation passed, not product quality."
    )
    values = [r.inference_ms for r in report.rows if r.status == "inspected"]
    if values:
        import statistics

        st.caption(
            f"Model inference per successful image: median {statistics.median(values):.2f} ms · "
            f"min {min(values):.2f} ms · max {max(values):.2f} ms"
        )
    else:
        st.caption("Model inference timing: unavailable; no successful images.")
    st.caption(TIMING_SCOPE)
    st.write(report.model.scientific_status)
    with st.expander("Model provenance and score semantics"):
        st.write(report.model.score_label)
        st.write("Artifact version: SHA-256 of the registered frozen artifact")
        st.code(report.model.registered_artifact_sha256, language=None)
        st.caption(
            "Artifact identity observed in successful results."
            if values
            else "Registered identity only; no successful inference verified an artifact."
        )
        st.caption("Installed catalog SHA-256: " + report.model.catalog_sha256)
        st.write(
            "No validated binary threshold."
            if report.model.frozen_threshold is None
            else f"Frozen research threshold: {report.model.frozen_threshold:.17g} "
            "(score > threshold)"
        )
    presentation.table(
        [
            {
                "Image": row.image_id,
                "Status": row.status,
                "Score": row.score,
                "Frozen threshold": row.threshold,
                "Research anomaly flag": row.research_anomaly_flag,
                "Interpretation": row.interpretation,
                "Model inference ms": row.inference_ms,
                "Error category": row.error_code,
            }
            for row in report.rows
        ],
        "report-results",
    )
    st.caption("Scores retain full precision in CSV/JSON. null means unavailable, never zero.")
    with st.expander("Scientific limitations", expanded=False):
        for limitation in LIMITATIONS:
            st.write(limitation)
    left, right = st.columns(2)
    filename = "factoryvision-" + report.report_id
    left.download_button(
        "Download research CSV",
        bundle.csv,
        file_name=filename + ".csv",
        mime="text/csv; charset=utf-8",
        key="csv-" + report.report_id,
        on_click="ignore",
    )
    right.download_button(
        "Download research JSON",
        bundle.json,
        file_name=filename + ".json",
        mime="application/json",
        key="json-" + report.report_id,
        on_click="ignore",
    )
    st.caption(
        "Downloads contain safe image IDs, scalar results and research provenance only. "
        "No filenames, images or EXIF. Exporting does not rerun inference. "
        "The app writes no report files automatically; your browser saves only on request."
    )
    return True
