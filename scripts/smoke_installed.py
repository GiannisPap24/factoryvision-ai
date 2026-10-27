"""Run from an empty directory with the installed wheel and app extra; no research assets."""

import json
import socket
from pathlib import Path

from streamlit.testing.v1 import AppTest

import factoryvision
from factoryvision.app import ui
from factoryvision.demo import NOTICE, build_demo, csv_bytes, json_bytes
from factoryvision.inference.batch import BatchService
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.model_catalog import catalog_entries
from factoryvision.inference.model_registry import experiment_index, registry_records
from factoryvision.inference.review import Decision
from factoryvision.inference.review import json_bytes as review_json
from factoryvision.inference.service import InspectionService

assert "site-packages" in str(factoryvision.__file__)
assert not list(Path.cwd().iterdir()), "Smoke working directory must be empty."
connect = socket.socket.connect


def deny(*args, **kwargs):
    raise AssertionError("Installed smoke must not connect to a network.")


def local_pair(*args, **kwargs):
    # Windows asyncio IPC only: same-process listener, not an app connection.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.socket()
        connect(client, listener.getsockname())
        server, _ = listener.accept()
    return server, client


socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.socketpair = local_pair
engine = BatchService(
    InspectionService(Path.cwd()),
    MetalPartsService(Path.cwd()),
    MetalPartsFeatureService(Path.cwd()),
)
entries = catalog_entries(engine)
assert len(entries) == 6 and not any(row.available for row in entries)
registry = registry_records(engine)
assert len(registry) == 6 and all(not row.catalog.available for row in registry)
assert [row.status for row in experiment_index()[-2:]] == ["BLOCKED", "INCONCLUSIVE"]
demo = build_demo()
assert json.loads(json_bytes(demo.report))["inference_executed"] is False
assert csv_bytes(demo.report) == csv_bytes(build_demo().report)
app = AppTest.from_file(ui.__file__, default_timeout=15).run()
assert not app.exception
app.sidebar.radio[0].set_value("Model Catalog").run()
assert not app.exception and app.error and app.table
app.selectbox(key="registry_category").set_value("pcb1").run()
assert not app.exception and any("BLOCKED" in text.value for text in app.text)
app.sidebar.radio[0].set_value("Synthetic demo").run()
assert not app.exception and any(NOTICE in warning.value for warning in app.warning)
next(button for button in app.button if button.label == "Generate synthetic demo").click().run()
assert not app.exception and len(app.get("download_button")) == 4
assert app.session_state["synthetic_demo"][2] == json_bytes(demo.report)
app.selectbox(key="fv_review_demo_decision_0").set_value(Decision.NEEDS_FURTHER_REVIEW)
app.text_area(key="fv_review_demo_note_0").set_value("Installed synthetic review")
next(button for button in app.button if button.label == "Save human review").click().run()
assert not app.exception
reviews = app.session_state["fv_review_demo_bundle"].records
assert reviews[0].decision == Decision.NEEDS_FURTHER_REVIEW
assert reviews[0].evidence.score is None and reviews[1].history == ()
assert review_json(reviews) == review_json(reviews)
app.sidebar.selectbox[0].set_value("MetalParts (experimental)").run()
assert "synthetic_demo" not in app.session_state
assert "fv_review_demo_bundle" not in app.session_state
app.sidebar.radio[0].set_value("Batch inspection").run()
assert not app.exception and not app.get("download_button")
print(
    "PASS: installed six-model catalog, missing artifacts, demo, human review, exports and resets."
)
