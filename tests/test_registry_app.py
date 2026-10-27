"""Registry UI shares the existing app and never needs inference or private assets."""

from streamlit.testing.v1 import AppTest
from test_app import entry
from test_app import local_event_loop_ipc as local_event_loop_ipc

from factoryvision.app import catalog_ui
from factoryvision.inference.model_registry import RegistryInvalid
from factoryvision.inference.service import InspectionService


def test_evidence_filters_and_source_only_details(tmp_path):
    at = AppTest.from_function(entry, args=(InspectionService(tmp_path),), default_timeout=15).run()
    at.sidebar.radio[0].set_value("Model Catalog").run()
    assert not at.exception and len(at.table[0].value) == 6
    assert len(at.selectbox(key="registry_evidence").options) == 8
    assert len(at.table[1].value) == 4
    assert at.table[1].value["Value"].tolist() == ["0.9376", "0.943219", "0.818119", "0.080936"]
    assert any(b.label == "Download registry metadata (JSON)" for b in at.get("download_button"))
    for category, expected in [
        ("pcb1", "BLOCKED"),
        ("pcb2", "INCONCLUSIVE"),
        ("metalparts", "EXPLORATORY"),
    ]:
        at.selectbox(key="registry_category").set_value(category).run()
        assert not at.exception and any(expected in t.value for t in at.text)
    at.selectbox(key="registry_status").set_value("HISTORICAL").run()
    assert not at.exception and any("No evidence matches" in t.value for t in at.info)
    at.selectbox(key="registry_category").set_value("All").run()
    assert len(at.selectbox(key="registry_evidence").options) == 2
    at.selectbox(key="registry_status").set_value("DEVELOPMENT ONLY").run()
    assert any("Official real-defect evaluation unavailable" in t.value for t in at.warning)


def test_tampered_registry_has_bounded_ui_error(tmp_path, monkeypatch):
    def reject(*args):
        raise RegistryInvalid()

    monkeypatch.setattr(catalog_ui, "registry_records", reject)
    at = AppTest.from_function(entry, args=(InspectionService(tmp_path),), default_timeout=15).run()
    at.sidebar.radio[0].set_value("Model Catalog").run()
    assert not at.exception and any("could not be verified" in t.value for t in at.error)
    assert not at.get("download_button")
