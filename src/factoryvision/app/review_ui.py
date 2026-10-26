"""Session-only review controls shared by inspection, batch and synthetic demo."""

from dataclasses import dataclass, field

import streamlit as st

from factoryvision.inference import review
from factoryvision.inference.reporting import ReportInvalid, create_report


@dataclass(frozen=True)
class Bundle:
    source: object = field(repr=False)
    records: tuple[review.ReviewRecord, ...]


def clear(scope=None):
    prefix = "fv_review_" + (scope + "_" if scope else "")
    for key in list(st.session_state):
        if key.startswith(prefix):
            st.session_state.pop(key, None)


def render(scope, source, category=None):
    if source is None:
        clear(scope)
        return
    prefix = "fv_review_" + scope + "_"
    try:
        bundle = st.session_state.get(prefix + "bundle")
        if not isinstance(bundle, Bundle) or bundle.source is not source:
            clear(scope)
            if scope == "batch":
                records = review.batch_records(create_report(source))
            elif scope == "demo":
                records = review.demo_records(source)
            else:
                records = review.single_records(source, category)
            bundle = Bundle(source, records)
            st.session_state[prefix + "bundle"] = bundle
        if not bundle.records:
            st.info("No successful result is available for human review.")
            return
        review.document(bundle.records)
    except (review.ReviewInvalid, ReportInvalid, ValueError, AttributeError, TypeError):
        clear(scope)
        st.error("Human review unavailable: result validation failed. Clear and retry.")
        return

    st.subheader("Human review" if scope != "demo" else "Synthetic review practice")
    st.caption(review.NOTICE)
    if scope == "demo":
        st.info(
            "Simulated workflow only. Review the validation example; no model prediction exists."
        )
    st.caption(
        "Session-only notes. Explicit exports include notes and revision history. "
        "Do not enter private information. Common paths/contact details/credentials are rejected; "
        "this is not comprehensive sensitive-data detection."
    )
    index = st.selectbox(
        "Result to review",
        range(len(bundle.records)),
        format_func=lambda i: (
            bundle.records[i].evidence.image_id
            + " · "
            + (bundle.records[i].evidence.model_id or "Synthetic validation")
        ),
        key=prefix + "selection",
    )
    record = bundle.records[index]
    evidence = record.evidence
    score = "Not applicable (synthetic validation)" if evidence.score is None else evidence.score
    flag = "Not defined" if evidence.research_flag is None else str(evidence.research_flag)
    st.write(f"Original score: {score} · Research flag: {flag}")
    st.caption(
        "Agreement concerns the displayed evidence; it is not a defect label or product acceptance."
    )
    with st.form(prefix + "form_" + str(index)):
        decision = st.selectbox(
            "Human decision",
            list(review.Decision),
            index=list(review.Decision).index(record.decision),
            key=prefix + f"decision_{index}",
        )
        note = st.text_area(
            "Review note (optional)",
            value=record.comment,
            max_chars=review.MAX_COMMENT,
            key=prefix + f"note_{index}",
        )
        saved = st.form_submit_button("Save human review")
    if saved:
        try:
            changed = review.revise(record, decision, note, review.utc_now())
            records = bundle.records[:index] + (changed,) + bundle.records[index + 1 :]
            bundle = Bundle(source, records)
            st.session_state[prefix + "bundle"] = bundle
        except review.ReviewInvalid as exc:
            st.error(str(exc))
    reviewed = sum(r.decision != review.Decision.NOT_REVIEWED for r in bundle.records)
    st.write(f"Reviewed: {reviewed} · Pending: {len(bundle.records) - reviewed}")
    st.table(
        [
            {
                "Image": r.evidence.image_id,
                "Model": r.evidence.model_id or "Synthetic",
                "Human observation": r.decision.value,
                "Revision": len(r.history),
                "UTC time": r.reviewed_at_utc or "Not reviewed",
            }
            for r in bundle.records
        ]
    )
    st.caption(
        "Set NOT_REVIEWED with an empty note to return to pending. "
        "Earlier revisions remain until clearing."
    )
    name = ("synthetic-" if scope == "demo" else "") + "human-review-" + evidence.inspection_id
    columns = st.columns(2)
    columns[0].download_button(
        "Download human review JSON",
        review.json_bytes(bundle.records),
        file_name=name + ".json",
        mime="application/json",
        key=prefix + "json",
        on_click="ignore",
    )
    columns[1].download_button(
        "Download human review CSV",
        review.csv_bytes(bundle.records),
        file_name=name + ".csv",
        mime="text/csv; charset=utf-8",
        key=prefix + "csv",
        on_click="ignore",
    )
