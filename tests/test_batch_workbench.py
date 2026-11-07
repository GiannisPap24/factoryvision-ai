"""UX-4 presentation on the real lifecycle with fabricated offline inputs."""

from dataclasses import replace

import pytest
from test_app import app as app
from test_app import batch_open, encoded
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_launch_readiness import source_app as source_app
from test_review_app import button, save
from test_workbench_ui import fabricated_inspection_app as fabricated_inspection_app


@pytest.mark.parametrize("category", ["cashew", "metalparts"])
@pytest.mark.parametrize(
    "score",
    [0.0, 0.12345678901234566, 0.0031531278509646654, 0.003153127850964666, 1e-20, 1e20],
)
def test_batch_and_reports_precision_keeps_snapshot_and_exports(source_app, category, score):
    """Fabricated completed snapshots; source-only services cannot run inference."""
    from test_reporting import snapshot

    from factoryvision.demo import build_demo
    from factoryvision.inference import reporting
    from factoryvision.inference.batch import Detail

    model_id = "autoencoder" if category == "cashew" else "metalparts_autoencoder"
    source = snapshot(category, model_id)
    row = source.rows[0]
    png = build_demo().images[0]
    row = replace(
        row,
        score=score,
        above_threshold=None if row.threshold is None else score > row.threshold,
        model_name="Fabricated precision fixture",
        score_label="Synthetic display check",
        detail=Detail(png, png, png, 1.0),
    )
    source = replace(source, rows=(row,))
    report = reporting.create_report(source)
    expected_exports = (reporting.csv_bytes(report), reporting.json_bytes(report))
    at, _ = source_app
    at.run().selectbox[0].set_value(
        "Cashew" if category == "cashew" else "MetalParts (experimental)"
    ).run()
    at.selectbox("workspace_page").set_value("Batch inspection").run()
    at.session_state["batch_report"] = source
    at.session_state["batch_archive_view"] = True
    at.run()
    for page in ("Batch inspection", "Reports", "Batch inspection"):
        at.selectbox("workspace_page").set_value(page).run()
        assert not at.exception
        text = next(m.value for m in at.markdown if m.value.startswith("Score: "))
        score_text, remainder = text.removeprefix("Score: ").split(" · Frozen threshold: ")
        threshold_text, timing = remainder.split(" · CPU inference: ")
        assert float(score_text) == row.score
        if row.threshold is None:
            assert threshold_text == "None" and row.above_threshold is None
            assert any("no validated threshold" in m.value for m in at.markdown)
        else:
            assert float(threshold_text) == row.threshold
            assert (float(score_text) > float(threshold_text)) == row.above_threshold
        assert timing == "2.12 ms"
        assert any("not a defect probability" in c.value for c in at.caption)
        assert at.session_state["batch_report"] is source
        outcomes = next(table.value for table in at.table if "Score" in table.value.columns)
        shown = outcomes.iloc[0]
        assert isinstance(shown["Score"], str)
        assert float(shown["Score"]) == row.score
        assert float(shown["Model inference ms"]) == row.inference_ms
        if row.threshold is None:
            assert shown["Frozen threshold"] == "null"
        else:
            assert float(shown["Frozen threshold"]) == row.threshold
        exports = at.session_state["batch_exports"]
        assert exports.source is source
        assert (exports.csv, exports.json) == expected_exports
        assert row.detail.original == row.detail.heatmap == row.detail.overlay == png


def test_stopped_batch_keeps_success_evidence_review_and_exports(app, monkeypatch):
    at, engine = batch_open(app, monkeypatch, [encoded(), encoded(color=150), encoded(color=200)])
    original = engine.inspect
    calls = []

    def inspect(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("Controlled synthetic stop")
        return original(*args)

    monkeypatch.setattr(engine, "inspect", inspect)
    at.checkbox(key="batch_ack").check().run()
    button(at, "Run batch inspection").click().run()
    assert not at.exception and len(calls) == 2
    source = at.session_state["batch_report"]
    assert [r.status for r in source.rows] == ["inspected", "failed", "skipped"]
    assert [t.label for t in at.tabs] == [
        "Summary & exports",
        "Image evidence",
        "Original",
        "Heatmap",
        "Overlay",
        "Reconstruction / reference",
        "Human review",
    ]
    assert at.selectbox(key="batch_detail").options == ["Image 001"]
    assert [(m.label, m.value) for m in at.metric[-4:]] == [
        ("Inspected", "1"),
        ("Rejected", "0"),
        ("Failed", "1"),
        ("Skipped", "1"),
    ]
    assert any("Session only" in w.value for w in at.warning)
    assert any("pending records" in i.value for i in at.info)
    bundle = at.session_state["batch_exports"]
    reviews = save(at, scope="batch")
    assert any(e.label.endswith("Complete image") for e in at.expander)
    at.run()
    assert not at.exception and len(calls) == 2
    assert at.session_state["batch_exports"] is bundle
    assert at.session_state["fv_review_batch_bundle"] is reviews
    at.selectbox("workspace_page").set_value("Reports").run()
    at.selectbox("workspace_page").set_value("Batch inspection").run()
    assert not at.exception and len(calls) == 2
    assert at.session_state["batch_report"] is source
    assert at.session_state["batch_exports"] is bundle
    assert at.session_state["fv_review_batch_bundle"] is reviews
    assert any("Batch snapshot retained" in i.value for i in at.info)
    assert not any("Completed batch retained" in s.value for s in at.success)
    assert len(at.get("download_button")) == 4


def test_invalid_report_feedback_is_outside_result_tabs(app, monkeypatch):
    at, _ = batch_open(app, monkeypatch, [encoded()])
    at.checkbox(key="batch_ack").check().run()
    button(at, "Run batch inspection").click().run()
    at.session_state["batch_report"] = replace(at.session_state["batch_report"], elapsed_ms=-1)
    at.run()
    assert not at.exception and not at.get("download_button")
    assert "fv_review_batch_bundle" not in at.session_state
    assert any("Batch evidence and Human Review unavailable" in e.value for e in at.error)
    assert all(
        "Batch evidence and Human Review unavailable" not in e.value
        for tab in at.tabs
        for e in tab.error
    )


@pytest.mark.parametrize("category", ["Cashew", "MetalParts (experimental)"])
def test_interrupted_terminal_state_is_immediate_and_clear_allows_one_new_run(
    fabricated_inspection_app, category, monkeypatch
):
    from factoryvision.demo import build_demo
    from factoryvision.inference.batch import BatchService

    at, _ = fabricated_inspection_app
    at.selectbox[0].set_value(category).run()
    at.selectbox("workspace_page").set_value("Batch inspection").run()
    payload = build_demo().images[0]
    uploads = [(f"synthetic-{i}.png", payload, "image/png") for i in range(3)]
    uploads.append(("synthetic-invalid.png", b"invalid", "image/png"))
    at.file_uploader[0].set_value(uploads).run()
    at.checkbox(key="batch_ack").check().run()
    original = BatchService.run
    calls = []
    interrupt = True

    def execute(self, plan, progress=None):
        calls.append(plan.batch_id)
        if interrupt:
            raise ValueError("Synthetic backend detail must not be displayed")
        return original(self, plan, progress)

    monkeypatch.setattr(BatchService, "run", execute)
    button(at, "Run batch inspection").click().run()
    assert not at.exception and len(calls) == 1
    assert button(at, "Run batch inspection").disabled
    assert "batch_report" not in at.session_state and "batch_exports" not in at.session_state
    marker = at.session_state["batch_started"]
    errors = [e.value for e in at.error]
    assert any("previous run" in e for e in errors)
    assert all("Synthetic backend detail" not in e for e in errors)
    for _ in range(2):
        at.run()
        assert button(at, "Run batch inspection").disabled and len(calls) == 1
        assert at.session_state["batch_started"] == marker
        assert [e.value for e in at.error] == errors
    button(at, "Clear batch").click().run()
    assert "batch_started" not in at.session_state and not at.checkbox(key="batch_ack").value
    interrupt = False
    at.file_uploader[0].set_value(uploads).run()
    at.checkbox(key="batch_ack").check().run()
    button(at, "Run batch inspection").click().run()
    assert not at.exception and len(calls) == 2
    report = at.session_state["batch_report"]
    assert [r.status for r in report.rows] == ["inspected", "inspected", "inspected", "rejected"]
    saved = save(at, scope="batch", note="Synthetic retained completed review")
    at.selectbox("workspace_page").set_value("Reports").run()
    at.selectbox("workspace_page").set_value("Batch inspection").run()
    assert len(calls) == 2 and at.session_state["batch_report"] is report
    assert at.session_state["fv_review_batch_bundle"] is saved
