"""Catalog projections and strictly separate synthetic examples; no research assets."""

import csv
import io
import json
from dataclasses import replace

import pytest

from factoryvision import demo
from factoryvision.inference.batch import BatchService, models
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.model_catalog import CatalogEntry, catalog_entries, definitions
from factoryvision.inference.reporting import ReportInvalid, create_report, model_info
from factoryvision.inference.service import InspectionService


def test_catalog_uses_exact_existing_registrations_and_provenance(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Catalog must not load or infer")

    for cls in (InspectionService, MetalPartsService, MetalPartsFeatureService):
        monkeypatch.setattr(cls, "_load", forbidden)
        monkeypatch.setattr(cls, "inspect", forbidden)
    engine = BatchService(
        InspectionService(tmp_path), MetalPartsService(tmp_path), MetalPartsFeatureService(tmp_path)
    )
    rows = catalog_entries(engine)
    assert len(rows) == 6 and all(not r.available for r in rows)
    assert all(r.verified_artifact_sha256 is None for r in rows)
    assert tuple(r.definition.identity.model_id for r in rows) == (
        *models("cashew"),
        *models("metalparts"),
    )
    for row in rows:
        d = row.definition
        assert d.identity == model_info(d.identity.category, d.identity.model_id)
        assert d.documentation.startswith("docs/") and d.documentation.endswith(".md")
        assert "C:" not in str(d) and "User" not in str(d)
        assert d.outputs and d.preprocessing and d.evaluation and d.limitations
        if d.identity.category == "metalparts":
            assert d.identity.frozen_threshold is None
        if "feature_memory" in d.identity.model_id:
            assert d.family == "Frozen-feature memory" and d.pretrained_sha256
        if d.identity.model_id == "spatial_mean":
            assert d.family == "Fitted statistical method" and d.pretrained_sha256 is None
    monkeypatch.setattr(engine, "status", lambda *args: {"available": True})
    ready = catalog_entries(engine)
    assert all(
        r.verified_artifact_sha256 == r.definition.identity.registered_artifact_sha256
        for r in ready
    )
    with pytest.raises(ValueError):
        CatalogEntry(definitions()[0], True, None)
    with pytest.raises(ValueError):
        CatalogEntry(definitions()[0], "true", definitions()[0].identity.registered_artifact_sha256)


def test_demo_is_deterministic_and_cannot_masquerade_as_research(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Demo must not call inference")

    monkeypatch.setattr(InspectionService, "inspect", forbidden)
    monkeypatch.setattr(BatchService, "run", forbidden)
    before = list(tmp_path.iterdir())
    a, b = demo.build_demo(), demo.build_demo()
    assert a == b and len(a.images) == 2
    assert demo.json_bytes(a.report) == demo.json_bytes(b.report)
    assert demo.csv_bytes(a.report) == demo.csv_bytes(b.report)
    doc = json.loads(demo.json_bytes(a.report))
    assert doc["notice"] == demo.NOTICE and doc["schema"] == "synthetic-workflow-v1"
    assert not doc["inference_executed"] and doc["model_id"] is None
    assert doc["artifact_sha256"] is None
    assert [r["validation_status"] for r in doc["rows"]] == [
        "demo_valid",
        "demo_valid",
        "demo_rejected",
    ]
    rows = list(csv.DictReader(io.StringIO(demo.csv_bytes(a.report).decode("utf8"))))
    assert len(rows) == 3 and all(r["notice"] == demo.NOTICE for r in rows)
    assert all(r["score"] == "null" and r["inference_executed"] == "false" for r in rows)
    assert all(r["score"] is None and r["threshold"] is None for r in doc["rows"])
    assert a.images[0] not in demo.json_bytes(a.report)
    assert before == list(tmp_path.iterdir())
    with pytest.raises(ReportInvalid):
        create_report(a.report)


@pytest.mark.parametrize(
    "change",
    [
        {"image_id": "../private"},
        {"validation_status": "inspected"},
        {"width": 12},
        {"score": 0.9},
        {"threshold": 1.0},
        {"research_anomaly_flag": True},
    ],
)
def test_demo_export_rejects_fabricated_inference_or_identity(change):
    source = demo.build_demo().report
    altered = replace(source, rows=(replace(source.rows[0], **change), *source.rows[1:]))
    for serializer in (demo.json_bytes, demo.csv_bytes):
        with pytest.raises(ValueError, match="Invalid synthetic"):
            serializer(altered)


def test_demo_invalid_collection():
    for rows in ((), [], (object(), object(), object())):
        with pytest.raises(ValueError):
            demo.json_bytes(demo.DemoReport(rows))
