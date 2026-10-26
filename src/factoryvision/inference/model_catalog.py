"""Typed presentation of existing registrations; never loads or registers a model."""

from dataclasses import dataclass

from factoryvision.inference.batch import models
from factoryvision.inference.reporting import LIMITATIONS, ModelInfo, model_info
from factoryvision.models.feature_memory import WEIGHTS_SHA256, WEIGHTS_URL

# Descriptive annotations only. IDs, names, thresholds and artifact identities remain
# authoritative in the existing services/package manifests, projected by model_info.
METHODS = {
    "autoencoder": (
        "Trained neural network",
        "Normal-only convolutional reconstruction training",
        "Historical official Cashew evaluation; test set now observed",
        "docs/model-card.md",
    ),
    "spatial_mean": (
        "Fitted statistical method",
        "Mean normal RGB reference; no gradient training",
        "Historical official Cashew evaluation; test set now observed",
        "docs/model-card.md",
    ),
    "feature_memory": (
        "Frozen-feature memory",
        "Frozen pretrained ResNet18; fitted normal descriptor bank",
        "Development diagnostics only; independent defect accuracy not established",
        "docs/feature-model-card.md",
    ),
    "student_teacher": (
        "Trained neural network",
        "Compact student trained against frozen pretrained teacher",
        "Development diagnostics only; independent defect accuracy not established",
        "docs/student-model-card.md",
    ),
    "metalparts_autoencoder": (
        "Trained neural network",
        "Normal source-domain convolutional reconstruction training",
        "Exploratory source-domain diagnostics; independent target evaluation NOT CLEARED",
        "docs/metalparts-model-card.md",
    ),
    "metalparts_feature_memory": (
        "Frozen-feature memory",
        "Frozen pretrained ResNet18; fitted source-normal descriptor bank",
        "Exploratory source-domain diagnostics; independent target evaluation NOT CLEARED",
        "docs/metalparts-feature-model-card.md",
    ),
}


@dataclass(frozen=True)
class Definition:
    identity: ModelInfo
    family: str
    learning: str
    preprocessing: str
    outputs: tuple[str, ...]
    evaluation: str
    documentation: str
    pretrained_sha256: str | None
    pretrained_url: str | None
    limitations: tuple[str, ...] = LIMITATIONS


@dataclass(frozen=True)
class CatalogEntry:
    definition: Definition
    available: bool
    verified_artifact_sha256: str | None

    def __post_init__(self):
        expected = self.definition.identity.registered_artifact_sha256 if self.available else None
        if type(self.available) is not bool or self.verified_artifact_sha256 != expected:
            raise ValueError("Invalid catalog availability state.")


def definitions() -> tuple[Definition, ...]:
    """All six existing registrations with supplemental method documentation."""
    result = []
    for category in ("cashew", "metalparts"):
        for key in models(category):
            info = model_info(category, key)
            family, learning, evaluation, documentation = METHODS[key]
            pretrained = key in ("feature_memory", "student_teacher", "metalparts_feature_memory")
            preprocessing = (
                "RGB; bilinear antialiased resize to 128 x 128; CPU float32 in [0,1]."
                if category == "cashew"
                else "Native RGB; 256 x 256 tiles, stride 192 including final boundary tiles; "
                "uniform overlap averaging; CPU float32 in [0,1]."
            )
            if pretrained:
                preprocessing += (
                    " Frozen ResNet18 uses ImageNet channel normalization."
                    if category == "cashew"
                    else " Each tile is resized to 128 x 128 for frozen ResNet18 with "
                    "ImageNet channel normalization; distances mapped to native coordinates."
                )
            outputs = ("continuous anomaly score", "spatial anomaly map", "display overlay")
            if "autoencoder" in key:
                outputs += ("RGB reconstruction",)
            if key == "spatial_mean":
                outputs += ("RGB reference",)
            if info.frozen_threshold is not None:
                outputs += ("frozen research threshold flag",)
            result.append(
                Definition(
                    info,
                    family,
                    learning,
                    preprocessing,
                    outputs,
                    evaluation,
                    documentation,
                    WEIGHTS_SHA256 if pretrained else None,
                    WEIGHTS_URL if pretrained else None,
                )
            )
    return tuple(result)


def catalog_entries(coordinator) -> tuple[CatalogEntry, ...]:
    """Reuse original file/metadata verification. Never deserialize or infer here."""
    entries = []
    for definition in definitions():
        info = definition.identity
        available = coordinator.status(info.category, info.model_id)["available"]
        entries.append(
            CatalogEntry(
                definition, available, info.registered_artifact_sha256 if available else None
            )
        )
    return tuple(entries)
