"""Read-only, versioned projections of accepted catalog and historical evidence.

No registration store, filesystem discovery, model deserialization or network I/O.
Pins bind adapters to reviewed package sources; they are not publisher signatures.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from importlib.resources import files
from pathlib import PurePosixPath

from factoryvision.inference.model_catalog import CatalogEntry, catalog_entries, definitions

# Package-relative resources only. Manifest paths are evidence, never read targets.
SOURCES = {
    "cashew_artifacts.json": "bc5027a41e5edaaa88337a559a3187c5144f8a35c4e7a964c8f7f98e289d0bd1",
    "metalparts_artifacts.json": "fd906518221411adb3f8275055191e7b2c3adcea43570974d903d745d820ca3c",
    "metalparts_feature_artifacts.json": (
        "f46ca4854607ace7898f4497842e799c6de113b164d5aaedf809a4063a5bfad3"
    ),
    "metalparts_comparison.json": (
        "be53dc22192df598abc7591dc814caab8918b7b964952a5f3127a80f92854d0f"
    ),
    "model_cards/model-card.md": "4e0b757caeacd2f6ecbbef108046d7a8ec4f52ab8d8f9c3016ff4b9ad7fc7e11",
    "model_cards/feature-model-card.md": (
        "8ccf8622a22ac000ea951bff39c687d256651e1badfeff3ece0726329d052a5b"
    ),
    "model_cards/student-model-card.md": (
        "782444abcc45599aa6ae852df65aed171992884164ef04b4bdeb72106f2f3e6b"
    ),
    "model_cards/metalparts-model-card.md": (
        "96e12065198148a9aff0419cdd7118bafbc97bb5825eef732458aee784ca5d71"
    ),
    "model_cards/metalparts-feature-model-card.md": (
        "2db9192c3d8c17c946701307742d85cc22939837902701ed8c6e2270d6a89642"
    ),
}
STATUSES = ("HISTORICAL", "DEVELOPMENT ONLY", "EXPLORATORY", "BLOCKED", "INCONCLUSIVE")


class RegistryInvalid(ValueError):
    def __init__(self):
        super().__init__("Registry evidence could not be verified. Use an intact source package.")


def require(condition):
    if not condition:
        raise RegistryInvalid()


def source_bytes(resource: str) -> bytes:
    """Bounded read of an exact allowlisted installed resource, never a supplied path."""
    require(type(resource) is str and resource in SOURCES)
    package = "factoryvision.app" if resource.startswith("model_cards/") else __package__
    try:
        with files(package).joinpath(resource).open("rb") as handle:
            raw = handle.read(262145)
        require(len(raw) <= 262144 and hashlib.sha256(raw).hexdigest() == SOURCES[resource])
        return raw
    except OSError:
        raise RegistryInvalid() from None


def document(resource):
    return json.loads(source_bytes(resource))


@dataclass(frozen=True)
class EvidenceReference:
    resource: str
    sha256: str

    def __post_init__(self):
        require(type(self.resource) is str and self.resource in SOURCES)
        require(type(self.sha256) is str and self.sha256 == SOURCES[self.resource])


def reference(resource):
    source_bytes(resource)
    return EvidenceReference(resource, SOURCES[resource])


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    scope: str
    evidence: EvidenceReference
    locator: str

    def __post_init__(self):
        require(self.name in ("auroc", "average_precision"))
        require(type(self.value) is float and math.isfinite(self.value) and 0 <= self.value <= 1)
        require(type(self.evidence) is EvidenceReference)
        require(all(type(s) is str and 0 < len(s) <= 600 for s in (self.scope, self.locator)))


@dataclass(frozen=True)
class ExperimentEvidence:
    schema_version: int
    evidence_id: str
    model_id: str | None
    artifact_sha256: str | None
    dataset: str
    category: str
    dataset_revision: str | None
    run_id: str | None
    partition_sha256: str | None
    partition_scope: str
    configuration_fingerprint: str | None
    recipe_file_sha256: str | None
    protocol_sha256: str | None
    configuration_provenance: str
    learning: str | None
    execution_provenance: str | None
    executed_at: str | None
    evaluation_protocol: str
    status: str
    metrics: tuple[Metric, ...]
    missing_metrics_reason: str | None
    sources: tuple[EvidenceReference, ...]

    def __post_init__(self):
        # V1 accepts only exact projections of reviewed historical evidence. This
        # also rejects mutated metrics, dataset/model pairs and fabricated timestamps.
        expected = _payload(self.evidence_id)
        for field in fields(self):
            actual, wanted = getattr(self, field.name), expected[field.name]
            require(type(actual) is type(wanted) and actual == wanted)


def _cashew_metrics(model_id, card):
    text = source_bytes(card).decode("utf-8")
    column = 2 if model_id == "autoencoder" else 3
    result = []
    for label, name, scope in (
        ("Image AUROC", "auroc", "image: 150 official images; 100 anomalous, 50 normal"),
        ("Image Average Precision", "average_precision", "image: same 150 official images"),
        ("Pixel AUROC", "auroc", "pixel: 2457600 resized pixels; 27672 positives"),
        ("Pixel Average Precision", "average_precision", "pixel: same pooled resized pixels"),
    ):
        line = next(line for line in text.splitlines() if line.startswith(f"| {label} |"))
        result.append(Metric(name, float(line.split("|")[column]), scope, reference(card), label))
    return tuple(result)


def _metal_metrics(model_id):
    resource = "metalparts_comparison.json"
    key = model_id.removeprefix("metalparts_")
    data = document(resource)[key]
    result = []
    for level in ("image", "pixel"):
        group = data[level]
        scope = f"{level}: {group['positive']} positives, {group['negative']} negatives."
        if level == "pixel":
            scope += f" {group['bins']} bins; " + group["interpretation"]
            scope += " " + group.get("quantization", group.get("score_quantization", ""))
        for metric in ("auroc", "average_precision"):
            result.append(
                Metric(metric, group[metric], scope, reference(resource), f"{key}.{level}.{metric}")
            )
    return tuple(result)


def _payload(evidence_id):
    require(type(evidence_id) is str)
    # Validate package manifests before invoking the existing catalog projection.
    for name in (
        "cashew_artifacts.json",
        "metalparts_artifacts.json",
        "metalparts_feature_artifacts.json",
    ):
        source_bytes(name)
    known = {"history-" + d.identity.model_id: d for d in definitions()}
    require(evidence_id in (*known, "integrity-pcb1", "integrity-pcb2"))
    payload = dict(
        schema_version=1,
        evidence_id=evidence_id,
        model_id=None,
        artifact_sha256=None,
        dataset="VisA",
        category="",
        dataset_revision=None,
        run_id=None,
        partition_sha256=None,
        partition_scope="Unavailable in packaged integrity summary",
        configuration_fingerprint=None,
        recipe_file_sha256=None,
        protocol_sha256=None,
        configuration_provenance="Unavailable",
        learning=None,
        execution_provenance=None,
        executed_at=None,
        evaluation_protocol="Integrity review; no accepted evaluation metrics",
        status="",
        metrics=(),
        missing_metrics_reason=None,
        sources=(),
    )
    if evidence_id.startswith("integrity-"):
        category = evidence_id.removeprefix("integrity-")
        payload.update(
            category=category,
            status="BLOCKED" if category == "pcb1" else "INCONCLUSIVE",
            missing_metrics_reason=(
                "Source overlap blocks confirmatory evaluation."
                if category == "pcb1"
                else "56 unresolved pairs; no independent human adjudication."
            ),
            sources=(reference("model_cards/model-card.md"),),
        )
        return payload

    definition = known[evidence_id]
    info = definition.identity
    card = "model_cards/" + PurePosixPath(definition.documentation).name
    payload.update(
        model_id=info.model_id,
        artifact_sha256=info.registered_artifact_sha256,
        category=info.category,
        learning=definition.learning,
        evaluation_protocol=definition.evaluation,
        configuration_provenance="Recorded method and preprocessing in the associated model card; "
        "canonical configuration fingerprint unavailable.",
    )
    if info.category == "cashew":
        resource = "cashew_artifacts.json"
        data = document(resource)
        spec = next(m for m in data["models"] if m["id"] == info.model_id)
        payload.update(
            run_id=PurePosixPath(spec["run"]).name,
            partition_sha256=data["manifest_sha256"],
            partition_scope="Canonical dataset/split manifest; 315 fitting normals, "
            "67 validation normals, 68 calibration normals; official test now observed.",
            recipe_file_sha256=spec["assets"].get("recipe", {}).get("sha256"),
            execution_provenance="Historical model card: Ryzen 7 9700X; Python 3.12.15; "
            "four Torch CPU threads. This is not the current application's runtime.",
            status="HISTORICAL"
            if info.model_id in ("autoencoder", "spatial_mean")
            else "DEVELOPMENT ONLY",
        )
        if payload["status"] == "HISTORICAL":
            payload["metrics"] = _cashew_metrics(info.model_id, card)
        else:
            payload["missing_metrics_reason"] = (
                "Official real-defect evaluation unavailable. Normal calibration and artificial "
                "probe diagnostics remain in the model card; they are not real-defect accuracy."
            )
    else:
        resource = (
            "metalparts_feature_artifacts.json"
            if "feature" in info.model_id
            else "metalparts_artifacts.json"
        )
        data = document(resource)
        identity = data["identity"]
        asset = data["assets"]["memory" if "feature" in info.model_id else "checkpoint"]
        payload.update(
            dataset="RobustAD",
            dataset_revision=identity["revision"],
            run_id=PurePosixPath(asset["path"]).parent.name,
            partition_sha256=identity["split_sha256"],
            partition_scope="Accepted source-normal fitting split; evaluation uses 51 source "
            "anomalies and 36 previously used validation normals. Target evaluation NOT CLEARED.",
            protocol_sha256=identity["protocol_sha256"],
            configuration_provenance="Recorded recipe: "
            + identity["recipe"]
            + "; protocol file SHA-256 is separate from an unavailable canonical config hash.",
            status="EXPLORATORY",
            metrics=_metal_metrics(info.model_id),
        )
    payload["sources"] = (reference(resource), reference(card))
    if info.category == "metalparts":
        payload["sources"] += (reference("metalparts_comparison.json"),)
    return payload


def experiment_index() -> tuple[ExperimentEvidence, ...]:
    """Six model evidence summaries and two integrity summaries, not eight trained runs."""
    for resource in SOURCES:
        source_bytes(resource)
    ids = tuple("history-" + d.identity.model_id for d in definitions())
    return tuple(
        ExperimentEvidence(**_payload(key)) for key in (*ids, "integrity-pcb1", "integrity-pcb2")
    )


@dataclass(frozen=True)
class RegistryRecord:
    schema_version: int
    catalog: CatalogEntry
    artifact_version: str
    experiment: ExperimentEvidence

    def __post_init__(self):
        require(type(self.schema_version) is int and self.schema_version == 1)
        require(type(self.catalog) is CatalogEntry)
        require(self.catalog.definition in definitions())
        self.catalog.__post_init__()
        require(type(self.experiment) is ExperimentEvidence)
        self.experiment.__post_init__()
        info = self.catalog.definition.identity
        require(
            self.experiment.model_id == info.model_id and self.experiment.category == info.category
        )
        require(self.experiment.artifact_sha256 == info.registered_artifact_sha256)
        require(self.artifact_version == "sha256:" + info.registered_artifact_sha256)


def registry_records(coordinator) -> tuple[RegistryRecord, ...]:
    experiments = {e.model_id: e for e in experiment_index() if e.model_id is not None}
    return tuple(
        RegistryRecord(
            1,
            entry,
            "sha256:" + entry.definition.identity.registered_artifact_sha256,
            experiments[entry.definition.identity.model_id],
        )
        for entry in catalog_entries(coordinator)
    )


def json_bytes(record: RegistryRecord) -> bytes:
    """Explicit scalar metadata export only; no local paths, datasets or inference output."""
    require(type(record) is RegistryRecord)
    record.__post_init__()
    return json.dumps(
        asdict(record), ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8")
