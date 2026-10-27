"""Historical evidence adapters: offline, immutable, source-only and fail closed."""

import io
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from factoryvision.inference import model_registry as registry
from factoryvision.inference.batch import BatchService
from factoryvision.inference.metalparts import MetalPartsService
from factoryvision.inference.metalparts_feature import MetalPartsFeatureService
from factoryvision.inference.model_catalog import definitions
from factoryvision.inference.service import InspectionService


def coordinator(root):
    return BatchService(
        InspectionService(root), MetalPartsService(root), MetalPartsFeatureService(root)
    )


def test_source_only_registry_deterministic_no_load_no_discovery(tmp_path, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("Registry must not load, infer or discover files")

    for cls in (InspectionService, MetalPartsService, MetalPartsFeatureService):
        monkeypatch.setattr(cls, "_load", forbidden)
        monkeypatch.setattr(cls, "inspect", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(Path, "rglob", forbidden)
    before = {name: registry.source_bytes(name) for name in registry.SOURCES}
    engine = coordinator(tmp_path)
    records = registry.registry_records(engine)
    assert records == registry.registry_records(engine)
    assert len(records) == 6 and len({r.experiment.model_id for r in records}) == 6
    assert tuple(r.catalog.definition for r in records) == definitions()
    assert all(
        not r.catalog.available and r.catalog.verified_artifact_sha256 is None for r in records
    )
    assert all(r.artifact_version == "sha256:" + r.experiment.artifact_sha256 for r in records)
    assert list(tmp_path.iterdir()) == []
    assert before == {name: registry.source_bytes(name) for name in registry.SOURCES}
    for record in records:
        payload = registry.json_bytes(record)
        assert payload == registry.json_bytes(record)
        assert b"C:" not in payload and b"User" not in payload
        assert json.loads(payload)["schema_version"] == 1
        with pytest.raises(FrozenInstanceError):
            record.schema_version = 2
    monkeypatch.setattr(engine, "status", lambda *a: {"available": True})
    ready = registry.registry_records(engine)
    assert all(r.catalog.verified_artifact_sha256 == r.experiment.artifact_sha256 for r in ready)
    assert [r.experiment for r in ready] == [r.experiment for r in records]


def test_historical_scope_precision_and_missing_evidence():
    rows = registry.experiment_index()
    ae, mean, feature, student, metal, memory, pcb1, pcb2 = rows
    assert len({r.evidence_id for r in rows}) == 8
    assert [m.value for m in ae.metrics] == [0.937600, 0.943219, 0.818119, 0.080936]
    assert [m.value for m in mean.metrics] == [0.736800, 0.841216, 0.895301, 0.267302]
    assert ae.run_id == "cashew-ae-128-seed42" and mean.run_id == "cashew-eval-v1"
    assert ae.partition_sha256 == "80bac86ae2fc1b0ad7e0e046597fb4a3a1e1f4f92ce556d5dddbfdd37d51f84c"
    assert "3.12.15" in ae.execution_provenance and "3.14" not in ae.execution_provenance
    assert ae.configuration_fingerprint is None and ae.recipe_file_sha256 is None
    assert (
        feature.recipe_file_sha256
        == "62ebced7c38f87a73a31099895de9120dc2320a0906bf3ed7263523f13b656fa"
    )
    assert (
        student.recipe_file_sha256
        == "fb5fd19e04aaa737de16b0483ca3f218af50aaf47707549dbc7c0345a42b7779"
    )
    assert all(not r.metrics and r.missing_metrics_reason for r in (feature, student, pcb1, pcb2))
    assert pcb1.status == "BLOCKED" and pcb2.status == "INCONCLUSIVE"
    assert pcb1.model_id is None and pcb2.artifact_sha256 is None
    assert all(r.executed_at is None and r.configuration_fingerprint is None for r in rows)
    assert metal.status == memory.status == "EXPLORATORY"
    assert metal.dataset == memory.dataset == "RobustAD"
    assert metal.partition_sha256 == memory.partition_sha256
    assert metal.dataset_revision == "82c642b2d18be60f2dab104f8dd4dea1f4ccfacb"
    assert metal.metrics[0].value == 0.5969498910675382
    assert memory.metrics[0].value == 0.8017429193899782
    assert all("65536 bins" in r.metrics[2].scope for r in (metal, memory))
    assert all("NOT CLEARED" in r.evaluation_protocol for r in (metal, memory))
    assert metal.execution_provenance is None  # Never substitute today's runtime.


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"evidence_id": "../private"},
        {"category": "metalparts"},
        {"dataset": "Unknown"},
        {"run_id": "invented"},
        {"executed_at": "2026-10-26"},
        {"artifact_sha256": "0" * 64},
        {"partition_sha256": "0" * 64},
        {"configuration_fingerprint": "0" * 64},
        {"status": "PRODUCTION READY"},
        {"metrics": []},
        {"sources": ()},
    ],
)
def test_experiment_rejects_invented_provenance(changes):
    with pytest.raises(registry.RegistryInvalid):
        replace(registry.experiment_index()[0], **changes)


def test_metric_mutation_and_model_version_relationships(tmp_path):
    record = registry.registry_records(coordinator(tmp_path))[0]
    experiment = record.experiment
    wrong = replace(experiment.metrics[0], value=0.0)
    with pytest.raises(registry.RegistryInvalid):
        replace(experiment, metrics=(wrong, *experiment.metrics[1:]))
    for changes in (
        {"schema_version": True},
        {"artifact_version": "sha256:" + "0" * 64},
        {"experiment": registry.experiment_index()[1]},
        {"catalog": object()},
        {"experiment": object()},
        {
            "catalog": replace(
                record.catalog, definition=replace(record.catalog.definition, family="Invented")
            )
        },
    ):
        with pytest.raises(registry.RegistryInvalid):
            replace(record, **changes)
    with pytest.raises(registry.RegistryInvalid):
        registry.json_bytes(experiment)
    for changes in (
        {"value": float("nan")},
        {"value": True},
        {"value": 1.1},
        {"evidence": object()},
        {"scope": ""},
        {"name": "accuracy"},
    ):
        with pytest.raises(registry.RegistryInvalid):
            replace(experiment.metrics[0], **changes)


@pytest.mark.parametrize(
    "name",
    [
        "../secret",
        "C:/private/file",
        "/etc/passwd",
        "file:///private",
        "https://example.com",
        None,
        [],
        "model_cards/../../secret",
    ],
)
def test_resource_allowlist_rejects_paths_before_io(name, monkeypatch):
    monkeypatch.setattr(registry, "files", lambda *a: pytest.fail("Must reject before file I/O"))
    with pytest.raises(registry.RegistryInvalid):
        registry.source_bytes(name)
    with pytest.raises(registry.RegistryInvalid):
        registry.EvidenceReference(name, "0" * 64)


@pytest.mark.parametrize(
    "payload",
    [b"{}", b"{invalid", b"<script>alert(1)</script>", b"x" * 262145],
    ids=["changed-json", "malformed", "script", "oversized"],
)
def test_tampered_package_fails_before_catalog_or_display(payload, monkeypatch):
    class Resource:
        def joinpath(self, *parts):
            return self

        def open(self, mode):
            return io.BytesIO(payload)

    monkeypatch.setattr(registry, "files", lambda *a: Resource())
    monkeypatch.setattr(registry, "definitions", lambda: pytest.fail("Unverified catalog read"))
    with pytest.raises(registry.RegistryInvalid):
        registry.experiment_index()


def test_missing_resource_does_not_disclose_path(monkeypatch):
    def missing(*args):
        raise OSError("private-path-secret")

    monkeypatch.setattr(registry, "files", missing)
    with pytest.raises(registry.RegistryInvalid) as failure:
        registry.source_bytes("cashew_artifacts.json")
    assert "private" not in str(failure.value)
    with pytest.raises(registry.RegistryInvalid):
        registry.EvidenceReference("cashew_artifacts.json", "0" * 64)


@pytest.mark.parametrize("name", registry.SOURCES)
def test_line_ending_tampering_is_not_normalized(name, monkeypatch):
    original = registry.source_bytes(name)
    changed = (
        original.replace(b"\r\n", b"\n")
        if b"\r\n" in original
        else original.replace(b"\n", b"\r\n")
    )
    assert changed != original

    class Resource:
        def joinpath(self, *parts):
            return self

        def open(self, mode):
            return io.BytesIO(changed)

    monkeypatch.setattr(registry, "files", lambda *a: Resource())
    with pytest.raises(registry.RegistryInvalid):
        registry.source_bytes(name)
