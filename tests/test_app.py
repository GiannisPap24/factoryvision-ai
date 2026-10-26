"""Presentation interactions through Streamlit's test runner; synthetic inputs only."""

import io
import socket
import tempfile
from dataclasses import replace

import pytest
import torch
from test_inference import Tiny, encoded

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from factoryvision.app import ui
from factoryvision.inference.service import MODEL_IDS, InspectionService, ModelUnavailable

ORIGINAL_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def local_event_loop_ipc(monkeypatch, tmp_path, no_network):
    torch.set_num_threads(1)

    # Windows asyncio needs a connected self-pipe. Only this in-process socket pair
    # may connect, to its freshly bound loopback listener; app connections remain blocked.
    def pair(*args, **kwargs):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            client = socket.socket()
            ORIGINAL_CONNECT(client, listener.getsockname())
            server, _ = listener.accept()
        return server, client

    monkeypatch.setattr(socket, "socketpair", pair)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    with pytest.raises(AssertionError, match="Network access"):
        socket.create_connection(("127.0.0.1", 8501))


def entry(engine):
    from factoryvision.app.ui import render

    render(engine)


def metal_entry(engine, metal_engine):
    from factoryvision.app.ui import render

    render(engine, metal_engine)


def test_metalparts_selection_explicit_run_and_clear(app, tmp_path, monkeypatch):
    from test_metalparts_inference import Zero, payload

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.metalparts import MetalPartsService

    _, cashew = app
    metal = MetalPartsService(tmp_path)
    monkeypatch.setattr(
        metal, "status", lambda: {"available": True, "reason": "Synthetic verified"}
    )
    calls = []

    def load():
        calls.append("explicit inference")
        return Zero()

    monkeypatch.setattr(metal, "_load", load)
    at = AppTest.from_function(metal_entry, args=(cashew, metal), default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert not at.exception and not calls
    at.sidebar.radio[0].set_value("Image inspection").run()
    assert at.button[1].disabled
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(payload()[1]))
    at.run().checkbox[0].check().run()
    assert not calls and not at.button[1].disabled
    at.button[1].click().run()
    assert not at.exception and len(calls) == 1 and len(at.metric) == 2
    assert at.session_state["metal_result"].threshold is None
    at.sidebar.selectbox[0].set_value("Cashew").run()
    assert "metal_result" not in at.session_state
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    at.checkbox[0].check().run().button[1].click().run()
    at.button[0].click().run()
    assert "metal_result" not in at.session_state


def test_metalparts_errors_and_research(app, tmp_path, monkeypatch):
    from test_metalparts_inference import payload

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.metalparts import MetalPartsService

    _, cashew = app
    metal = MetalPartsService(tmp_path)
    at = AppTest.from_function(metal_entry, args=(cashew, metal), default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert at.error and not at.exception
    at.sidebar.radio[0].set_value("Image inspection").run()
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(b"bad"))
    at.run()
    assert at.error and at.button[1].disabled
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run()
    assert any("256" in err.value for err in at.error)
    monkeypatch.setattr(metal, "status", lambda: {"available": True, "reason": "Synthetic"})
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(payload()[1]))
    at.run().checkbox[0].check().run().button[1].click().run()
    assert at.error and "metal_result" not in at.session_state
    at.sidebar.radio[0].set_value("Research results").run()
    assert not at.exception and any("NOT INDEPENDENT" in w.value for w in at.warning)


@pytest.fixture
def app(tmp_path, monkeypatch):
    engine = InspectionService(tmp_path)
    monkeypatch.setattr(
        engine,
        "statuses",
        lambda: [
            {
                "id": key,
                "name": engine.specs[key]["name"],
                "available": True,
                "reason": "Synthetic test",
            }
            for key in MODEL_IDS
        ],
    )
    monkeypatch.setattr(engine, "_load", lambda key: (Tiny(key).eval(), engine.specs[key]))
    return AppTest.from_function(entry, args=(engine,), default_timeout=10), engine


def test_navigation_and_explicit_inference(app, monkeypatch):
    at, engine = app
    at.run()
    assert not at.exception and at.table
    at.sidebar.radio[0].set_value("Research results").run()
    assert not at.exception and "Recorded" in at.subheader[0].value
    at.sidebar.radio[0].set_value("Compare").run()
    assert at.button[1].disabled and "results" not in at.session_state
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run()
    at.checkbox[0].check().run()
    assert not at.button[1].disabled and "results" not in at.session_state
    at.multiselect[0].set_value(list(MODEL_IDS)).run()
    assert "results" not in at.session_state
    at.button[1].click().run()
    assert not at.exception and len(at.session_state["results"]) == 4
    assert len(at.metric) == 12
    at.multiselect[0].set_value(["autoencoder"]).run()
    assert "results" not in at.session_state
    at.button[1].click().run()
    at.button[0].click().run()
    assert "results" not in at.session_state


def test_input_and_model_failures(app, monkeypatch):
    at, engine = app
    at.run().sidebar.radio[0].set_value("Image inspection").run()
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(b"bad"))
    at.run()
    assert at.error and at.button[1].disabled
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run().checkbox[0].check().run()

    def fail(*args):
        raise ModelUnavailable("Synthetic missing model")

    monkeypatch.setattr(engine, "inspect_many", fail)
    at.button[1].click().run()
    assert "Synthetic missing" in at.error[0].value and "results" not in at.session_state
    monkeypatch.setattr(engine, "statuses", lambda: [])
    at.run()
    assert any("No verified" in e.value for e in at.error)


def result_entry(result):
    from factoryvision.app.ui import show_result

    show_result(result)


def test_missing_threshold_and_low_score(app):
    _, engine = app
    result = engine.inspect(encoded(), "autoencoder")
    for threshold, flag in ((None, None), (1.0, False)):
        at = AppTest.from_function(
            result_entry, args=(replace(result, threshold=threshold, above_threshold=flag),)
        ).run()
        assert not at.exception and len(at.metric) == 3


def test_launcher_flags(monkeypatch):
    import sys

    import streamlit.web.cli

    from factoryvision.app.__main__ import main

    seen = []
    monkeypatch.setattr(streamlit.web.cli, "main", lambda: seen.extend(sys.argv))
    monkeypatch.setattr(sys, "argv", [])
    main()
    assert "--server.address=127.0.0.1" in seen
    assert "--browser.gatherUsageStats=false" in seen
    assert "--server.enableXsrfProtection=true" in seen


def test_metalparts_comparison_selection(app, tmp_path, monkeypatch):
    from test_metalparts_inference import Zero, payload

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.metalparts import MetalPartsService

    _, cashew = app
    metal = MetalPartsService(tmp_path)
    monkeypatch.setattr(metal, "status", lambda: {"available": True, "reason": "Synthetic"})
    monkeypatch.setattr(metal, "_load", lambda: Zero())
    calls = []

    class Feature:
        def status(self):
            return {"available": True, "reason": "Synthetic"}

        def inspect(self, data):
            calls.append("feature")
            return replace(
                metal.inspect(data), model_id="metalparts_feature_memory", reconstruction=None
            )

    monkeypatch.setattr(metalparts_ui, "feature_service", lambda root: Feature())
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(payload()[1]))
    at = AppTest.from_function(metal_entry, args=(cashew, metal), default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    at.sidebar.radio[0].set_value("Compare").run()
    at.multiselect[0].set_value(["Convolutional Autoencoder", "Feature-Memory Detector"]).run()
    at.checkbox[0].check().run()
    assert not calls
    at.button[1].click().run()
    assert not at.exception and len(at.metric) == 4 and calls == ["feature"]
    assert len(at.session_state["metal_results"]) == 2
    at.multiselect[0].set_value(["Feature-Memory Detector"]).run()
    assert "metal_results" not in at.session_state
    at.button[1].click().run()
    assert len(at.metric) == 2
    at.multiselect[0].set_value([]).run()
    assert at.button[1].disabled and "metal_result" not in at.session_state


@pytest.mark.parametrize("error", [RuntimeError, MemoryError])
def test_cpu_failure_is_actionable_and_does_not_expose_paths(app, monkeypatch, error):
    at, engine = app
    at.run().sidebar.radio[0].set_value("Image inspection").run()
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run().checkbox[0].check().run().button[1].click().run()
    assert "results" in at.session_state

    def fail(*args):
        raise error("private-machine-path must not be displayed")

    monkeypatch.setattr(engine, "inspect_many", fail)
    at.button[1].click().run()
    assert not at.exception and "results" not in at.session_state
    assert any("smaller supported image" in item.value for item in at.error)
    assert all("private-machine-path" not in item.value for item in at.error)


def test_category_switch_releases_upload_and_consent(app, monkeypatch):
    at, _ = app
    at.run().sidebar.radio[0].set_value("Image inspection").run()
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run().checkbox[0].check().run().button[1].click().run()
    generation = at.session_state["upload_generation"]
    at.session_state[f"upload_{generation}"] = "synthetic retained upload"
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert not at.exception and "results" not in at.session_state
    assert f"upload_{generation}" not in at.session_state
    at.sidebar.selectbox[0].set_value("Cashew").run()
    assert not at.checkbox[0].value and at.button[1].disabled
    assert "results" not in at.session_state


def test_fresh_checkout_explains_missing_models_without_loading(tmp_path, monkeypatch):
    engine = InspectionService(tmp_path)

    def forbidden(*args):
        raise AssertionError("Navigation must never load or infer")

    monkeypatch.setattr(engine, "_load", forbidden)
    at = AppTest.from_function(entry, args=(engine,), default_timeout=10).run()
    assert not at.exception
    assert any("Model files missing" in item.label for item in at.expander)
    assert any("docs/reproduction.md" in item.value for item in at.markdown)
    at.sidebar.radio[0].set_value("Image inspection").run()
    assert at.button[1].disabled and at.error and not at.exception


def test_metalparts_runtime_failure_is_contained(app, tmp_path, monkeypatch):
    from test_metalparts_inference import payload

    from factoryvision.app import metalparts_ui
    from factoryvision.inference.metalparts import MetalPartsService

    _, cashew = app
    metal = MetalPartsService(tmp_path)
    monkeypatch.setattr(metal, "status", lambda: {"available": True, "reason": "Synthetic"})

    def fail(*args):
        raise RuntimeError("private-machine-path")

    monkeypatch.setattr(metal, "inspect", fail)
    at = AppTest.from_function(metal_entry, args=(cashew, metal), default_timeout=10).run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    at.sidebar.radio[0].set_value("Image inspection").run()
    monkeypatch.setattr(metalparts_ui.st, "file_uploader", lambda *a, **k: io.BytesIO(payload()[1]))
    at.run().checkbox[0].check().run().button[1].click().run()
    assert not at.exception and "metal_results" not in at.session_state
    assert any("CPU inference could not finish" in item.value for item in at.error)


class BatchUpload(io.BytesIO):
    @property
    def size(self):
        return len(self.getvalue())


def batch_open(app, monkeypatch, payloads):
    from factoryvision.app import batch_ui

    at, engine = app
    at.run().sidebar.radio[0].set_value("Batch inspection").run()
    monkeypatch.setattr(
        batch_ui.st, "file_uploader", lambda *a, **k: [BatchUpload(p) for p in payloads]
    )
    at.run()
    return at, engine


def test_batch_explicit_once_details_and_input_resets(app, monkeypatch):
    data = [encoded(), b"invalid", encoded(color=150)]
    at, engine = batch_open(app, monkeypatch, data)
    original = engine.inspect
    calls = []

    def inspect(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(engine, "inspect", inspect)
    assert not at.exception and at.button[1].disabled and not calls
    assert at.session_state["batch_plan"].accepted == 2
    at.checkbox[0].check().run()
    assert not calls
    at.button[1].click().run()
    assert not at.exception and len(calls) == 2
    report = at.session_state["batch_report"]
    assert report.count("inspected") == 2 and report.count("rejected") == 1
    bundle = at.session_state["batch_exports"]
    assert bundle.source is report and len(at.get("download_button")) == 2
    assert bundle.report.summary.rejected == 1
    at.run().selectbox(key="batch_detail").set_value("Image 003").run()
    assert at.session_state["batch_exports"] is bundle
    assert at.session_state["batch_exports"].csv == bundle.csv
    assert len(calls) == 2 and at.button[1].disabled
    at.selectbox(key="batch_model_cashew").set_value("spatial_mean").run()
    assert "batch_report" not in at.session_state and len(calls) == 2
    assert "batch_exports" not in at.session_state and not at.get("download_button")
    at.button[1].click().run()
    assert len(calls) == 4 and not at.exception
    data.reverse()
    at.run()
    assert "batch_report" not in at.session_state
    assert "batch_exports" not in at.session_state
    at.button[1].click().run()
    at.number_input[0].set_value(3).run()
    assert "batch_report" not in at.session_state
    assert "batch_exports" not in at.session_state
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert not at.exception and "batch_report" not in at.session_state
    assert "batch_exports" not in at.session_state
    assert not at.checkbox[0].value
    at.sidebar.radio[0].set_value("Image inspection").run()
    assert "batch_plan" not in at.session_state
    assert "batch_exports" not in at.session_state


def test_batch_missing_failed_interrupted_and_clear(app, monkeypatch):
    from factoryvision.inference.batch import BatchService

    at, engine = batch_open(app, monkeypatch, [encoded()])
    monkeypatch.setattr(engine, "statuses", lambda: [])
    at.checkbox[0].check().run()
    assert at.button[1].disabled and at.error and not at.exception
    monkeypatch.setattr(BatchService, "status", lambda *a: {"available": True})

    def fail(*args):
        raise ValueError("private exception detail")

    monkeypatch.setattr(BatchService, "run", fail)
    at.run().button[1].click().run()
    assert not at.exception and "batch_report" not in at.session_state
    at.run()
    assert at.button[1].disabled and any("previous run" in e.value for e in at.error)
    assert all("private exception" not in e.value for e in at.error)
    at.button[0].click().run()
    assert "batch_started" not in at.session_state and not at.checkbox[0].value


def test_batch_budget_and_preview_failure(app, monkeypatch):
    from factoryvision.app import batch_ui

    at, _ = batch_open(app, monkeypatch, [])
    seen = []

    def uploads(*args, **kwargs):
        seen.append(kwargs["key"])
        return [BatchUpload(encoded())] * 21 if len(seen) == 1 else []

    monkeypatch.setattr(batch_ui.st, "file_uploader", uploads)
    at.run()
    assert not at.exception and any("at most 20" in e.value for e in at.error)
    assert seen[0] != seen[1] and at.button[1].disabled

    def fail(*args):
        raise MemoryError("private")

    monkeypatch.setattr(batch_ui, "prepare", fail)
    del at.session_state["batch_plan"]
    at.run()
    assert not at.exception and any("preview could not finish" in e.value for e in at.error)


def test_batch_fatal_report_remains_reviewable(app, monkeypatch):
    at, engine = batch_open(app, monkeypatch, [encoded(), encoded(color=200)])

    def fail(*args):
        raise ModelUnavailable("private artifact path")

    monkeypatch.setattr(engine, "inspect", fail)
    at.checkbox[0].check().run().button[1].click().run()
    assert not at.exception and at.button[1].disabled
    report = at.session_state["batch_report"]
    assert report.count("failed") == 1 and report.count("skipped") == 1
    assert any("Batch stopped" in e.value for e in at.error)
    assert len(at.get("download_button")) == 2
    bundle = at.session_state["batch_exports"]
    assert bundle.report.summary.failed == 1 and bundle.report.summary.skipped == 1
    assert b'"score": null' in bundle.json


def test_invalid_report_disables_exports_and_clear_releases_bundle(app, monkeypatch):
    from dataclasses import replace

    at, _ = batch_open(app, monkeypatch, [encoded()])
    at.checkbox[0].check().run().button[1].click().run()
    assert "batch_exports" in at.session_state
    at.session_state["batch_report"] = replace(
        at.session_state["batch_report"], elapsed_ms=float("nan")
    )
    at.run()
    assert not at.exception and "batch_exports" not in at.session_state
    assert not at.get("download_button")
    assert any("Report validation failed" in item.value for item in at.error)
    at.button[0].click().run()
    assert "batch_report" not in at.session_state and "batch_exports" not in at.session_state


def test_fresh_catalog_and_demo_never_infer_or_mix_report_state(tmp_path, monkeypatch):
    from factoryvision.demo import NOTICE
    from factoryvision.inference.batch import BatchService

    def forbidden(*args, **kwargs):
        raise AssertionError("Navigation and synthetic demo must never infer")

    engine = InspectionService(tmp_path)
    monkeypatch.setattr(engine, "_load", forbidden)
    monkeypatch.setattr(BatchService, "run", forbidden)
    at = AppTest.from_function(entry, args=(engine,), default_timeout=10).run()
    at.sidebar.radio[0].set_value("Model Catalog").run()
    assert not at.exception and at.error and len(at.table[0].value) == 6
    assert any("Unavailable" in x.value for x in at.code)
    for model in MODEL_IDS:
        at.selectbox(key="catalog_model_cashew").set_value(model).run()
        assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic demo").run()
    assert not at.exception and "synthetic_demo" not in at.session_state
    assert not at.get("download_button") and any(x.value == NOTICE for x in at.warning)
    next(b for b in at.button if b.label == "Generate synthetic demo").click().run()
    bundle = at.session_state["synthetic_demo"]
    assert not at.exception and len(at.get("download_button")) == 2
    assert all(key not in at.session_state for key in ("batch_report", "batch_exports", "results"))
    at.run()
    assert at.session_state["synthetic_demo"] is bundle
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert "synthetic_demo" not in at.session_state and not at.get("download_button")
    next(b for b in at.button if b.label == "Generate synthetic demo").click().run()
    next(b for b in at.button if b.label == "Clear synthetic demo").click().run()
    assert "synthetic_demo" not in at.session_state
    next(b for b in at.button if b.label == "Generate synthetic demo").click().run()
    at.sidebar.radio[0].set_value("Model Catalog").run()
    assert "synthetic_demo" not in at.session_state and at.error and not at.exception
    at.selectbox(key="catalog_model_metalparts").set_value("metalparts_feature_memory").run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Batch inspection").run()
    assert at.button[1].disabled and not at.get("download_button") and not at.exception


def test_catalog_verified_status_uses_existing_adapter(app):
    at, _ = app
    at.run().sidebar.radio[0].set_value("Model Catalog").run()
    assert not at.exception and at.success
    assert not any(x.value == "Unavailable" for x in at.code)


def test_reports_navigation_preserves_exports_without_inference(app, monkeypatch):
    at, engine = batch_open(app, monkeypatch, [encoded(), b"invalid", encoded(color=150)])
    calls = []
    original = engine.inspect

    def inspect(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(engine, "inspect", inspect)
    at.checkbox[0].check().run().button[1].click().run()
    source = at.session_state["batch_report"]
    exports = at.session_state["batch_exports"]
    assert len(calls) == 2
    at.sidebar.radio[0].set_value("Reports").run()
    assert not at.exception and at.session_state["batch_report"] is source
    assert at.session_state["batch_exports"] is exports
    assert len(at.get("download_button")) == 2 and len(calls) == 2
    at.run()
    assert at.session_state["batch_exports"].csv == exports.csv and len(calls) == 2
    at.sidebar.radio[0].set_value("Batch inspection").run()
    assert not at.exception and at.session_state["batch_report"] is source
    assert not any(b.label == "Run batch inspection" for b in at.button)
    assert len(calls) == 2
    next(b for b in at.button if b.label == "Clear batch").click().run()
    assert "batch_report" not in at.session_state and "batch_exports" not in at.session_state
    assert any(b.label == "Run batch inspection" for b in at.button)


def test_report_is_invalidated_on_category_or_unrelated_navigation(app, monkeypatch):
    at, _ = batch_open(app, monkeypatch, [encoded()])
    at.checkbox[0].check().run().button[1].click().run()
    at.sidebar.radio[0].set_value("Reports").run()
    at.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
    assert not at.exception and "batch_report" not in at.session_state
    assert "batch_exports" not in at.session_state and not at.get("download_button")
    next(b for b in at.button if b.label == "Open Batch Inspection").click().run()
    assert at.sidebar.radio[0].value == "Batch inspection"
    at.sidebar.radio[0].set_value("Overview").run()
    assert "batch_plan" not in at.session_state


def test_inspect_compare_transition_requires_new_explicit_run(app, monkeypatch):
    at, engine = app
    monkeypatch.setattr(ui.st, "file_uploader", lambda *a, **k: io.BytesIO(encoded()))
    at.run()
    next(b for b in at.button if b.label == "Start inspection").click().run()
    assert at.sidebar.radio[0].value == "Image inspection"
    assert at.multiselect[0].proto.max_selections == 1
    at.checkbox[0].check().run().button[1].click().run()
    assert len(at.session_state["results"]) == 1
    at.sidebar.radio[0].set_value("Compare").run()
    assert "results" not in at.session_state and not at.checkbox[0].value
    at.multiselect[0].set_value(list(MODEL_IDS)).run()
    assert "results" not in at.session_state
    at.checkbox[0].check().run().button[1].click().run()
    assert len(at.session_state["results"]) == 4
    at.sidebar.radio[0].set_value("Image inspection").run()
    assert "results" not in at.session_state and len(at.multiselect[0].value) == 1


def test_overview_actions_reports_empty_and_model_cards(app):
    at, _ = app
    at.run()
    next(b for b in at.button if b.label == "Compare models").click().run()
    assert at.sidebar.radio[0].value == "Compare" and not at.exception
    at.sidebar.radio[0].set_value("Overview").run()
    next(b for b in at.button if b.label == "Open demo").click().run()
    assert (
        at.sidebar.radio[0].value == "Synthetic demo" and "synthetic_demo" not in at.session_state
    )
    at.sidebar.radio[0].set_value("Reports").run()
    assert not at.exception and not at.get("download_button")
    at.sidebar.radio[0].set_value("Model Catalog").run()
    cards = [b for b in at.get("download_button") if b.label == "Download model card"]
    assert len(cards) == 6 and not at.exception
