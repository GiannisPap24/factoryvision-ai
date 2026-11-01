"""UX-4 presentation on the real lifecycle with fabricated offline inputs."""

from dataclasses import replace

from test_app import app as app
from test_app import batch_open, encoded
from test_app import local_event_loop_ipc as local_event_loop_ipc
from test_review_app import button, save


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
    at.sidebar.radio[0].set_value("Reports").run()
    at.sidebar.radio[0].set_value("Batch inspection").run()
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
