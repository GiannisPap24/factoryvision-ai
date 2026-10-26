"""CPU Cashew inference through original model implementations and frozen artifacts."""

import json
import math
import threading
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.reconstruction import reconstruction_errors
from factoryvision.inference.uploads import decode_upload, readonly
from factoryvision.models.feature_memory import FeatureMemory, load_extractor
from factoryvision.models.spatial_mean import SpatialMean
from factoryvision.models.student_teacher import SpatialTeacher, StudentTeacher
from factoryvision.training.checkpoint import checked_directory, file_sha256, load_checkpoint
from factoryvision.training.feature_artifacts import load_memory
from factoryvision.training.student_artifacts import load_student

MODEL_IDS = ("autoencoder", "spatial_mean", "feature_memory", "student_teacher")


class ModelUnavailable(ValueError):
    """Artifact availability/compatibility error safe for presentation."""


@dataclass(frozen=True)
class InspectionResult:
    model_id: str
    model_name: str
    score_label: str
    score: float
    threshold: float | None
    above_threshold: bool | None
    spatial: np.ndarray
    original: np.ndarray
    reconstruction: np.ndarray | None
    reference: np.ndarray | None
    preprocessing_ms: float
    inference_ms: float
    artifact_check_load_ms: float
    total_ms: float
    artifact_sha256: str


def catalog():
    return json.loads(
        files("factoryvision.inference").joinpath("cashew_artifacts.json").read_text()
    )


def preprocess(image):
    resized = TF.resize(image, [128, 128], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.pil_to_tensor(resized).to(dtype=torch.float32).div(255).unsqueeze(0)


def threshold_flag(score, threshold):
    if not math.isfinite(score) or score < 0:
        raise ModelUnavailable("The model returned an invalid image score.")
    if threshold is None:
        return None
    if not math.isfinite(threshold) or threshold < 0:
        raise ModelUnavailable("The frozen threshold is invalid.")
    return score > threshold


def verified_asset(root, entry):
    """Only internal pinned paths; reject aliases/links before hashing or deserialization."""
    path = root / entry["path"]
    if not path.resolve().is_relative_to(root.resolve()):
        raise ModelUnavailable("The artifact path escapes the project.")
    for node in (path, *path.parents):
        if node.is_symlink() or node.is_junction():
            raise ModelUnavailable("Linked artifact paths are not accepted.")
        if node == root:
            break
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ModelUnavailable("A required frozen artifact is missing or has changed size.")
    if file_sha256(path) != entry["sha256"]:
        raise ModelUnavailable("A frozen artifact failed its SHA-256 check.")
    return path


class InspectionService:
    """One process-local model cache; uploaded bytes/results are never cached here."""

    def __init__(self, project: Path):
        self.root = Path(project).absolute()
        self.registry = catalog()
        self.specs = {s["id"]: s for s in self.registry["models"]}
        if (
            tuple(self.specs) != MODEL_IDS
            or self.registry["category"] != "cashew"
            or self.registry["resolution"] != 128
        ):
            raise ModelUnavailable("Unsupported installed artifact catalog.")
        self._models = {}
        self._lock = threading.RLock()

    def _verify(self, model_id):
        if model_id not in MODEL_IDS:
            raise ModelUnavailable("Choose one of the four supported Cashew models.")
        try:
            checked_directory(self.root)
            manifest_path = verified_asset(self.root, self.registry["manifest"])
            manifest = load_manifest(manifest_path)
            if manifest_sha256(manifest) != self.registry["manifest_sha256"] or {
                s.category for s in manifest.samples
            } != {"cashew"}:
                raise ModelUnavailable(
                    "The artifact dataset is incompatible with Cashew inference."
                )
            fit_ids = sorted(
                s.sample_id for s in manifest.samples if s.partition == Partition.TRAIN
            )
            if fit_ids != self.registry["fit_ids"]:
                raise ModelUnavailable(
                    "The normal fitting membership does not match the frozen model."
                )
            spec = self.specs[model_id]
            paths = {k: verified_asset(self.root, v) for k, v in spec["assets"].items()}
            calibration = json.loads(paths["calibration"].read_text(encoding="utf-8"))
            if (
                calibration["threshold"] != spec["threshold"]
                or calibration["comparison"] != "score > threshold"
                or calibration["interpolation"] != "linear"
                or calibration["quantile"] != 0.95
                or calibration["count"] != 68
            ):
                raise ModelUnavailable("The calibration record differs from its frozen contract.")
            threshold_flag(0, spec["threshold"])
            return spec, paths
        except (OSError, ValueError, KeyError) as exc:
            if isinstance(exc, ModelUnavailable):
                raise
            raise ModelUnavailable(
                "Required local Cashew artifacts are unavailable or incompatible."
            ) from exc

    def statuses(self):
        result = []
        for key in MODEL_IDS:
            try:
                self._verify(key)
                reason, available = "Frozen artifact files verified", True
            except ModelUnavailable as exc:
                reason, available = str(exc), False
            result.append(
                {
                    "id": key,
                    "name": self.specs[key]["name"],
                    "available": available,
                    "reason": reason,
                }
            )
        return result

    def _load(self, model_id):
        spec, paths = self._verify(model_id)  # Also recheck files on every explicit run.
        if model_id in self._models:
            return self._models[model_id], spec
        directory = paths["weights"].parent
        manifest_sha = self.registry["manifest_sha256"]
        ids = self.registry["fit_ids"]
        try:
            if model_id == "autoencoder":
                model, _ = load_checkpoint(directory, manifest_sha, 128)
            elif model_id == "spatial_mean":
                meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
                if (
                    meta["manifest_sha256"] != manifest_sha
                    or meta["baseline_sha256"] != spec["assets"]["weights"]["sha256"]
                ):
                    raise ValueError("Spatial reference provenance mismatch.")
                model = SpatialMean(
                    torch.load(paths["weights"], map_location="cpu", weights_only=True)
                )
                if model.reference.shape != (3, 128, 128):
                    raise ValueError("Spatial reference shape mismatch.")
            elif model_id == "feature_memory":
                bank, _ = load_memory(
                    directory, manifest_sha, spec["assets"]["recipe"]["sha256"], ids
                )
                model = FeatureMemory(load_extractor(paths["teacher"]), bank)
            else:
                student, _ = load_student(
                    directory, manifest_sha, spec["assets"]["recipe"]["sha256"], ids
                )
                model = StudentTeacher(SpatialTeacher(load_extractor(paths["teacher"])), student)
            model.eval().requires_grad_(False)
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            raise ModelUnavailable(
                "The frozen model could not be safely loaded; inference was not run."
            ) from exc
        self._models[model_id] = model
        return model, spec

    def inspect(self, payload: bytes, model_id: str) -> InspectionResult:
        start = time.perf_counter()
        image = decode_upload(payload)
        x = preprocess(image)
        prepared = time.perf_counter()
        with self._lock, torch.inference_mode():
            model, spec = self._load(model_id)
            loaded = time.perf_counter()
            reconstruction = reference = None
            try:
                if model_id in ("autoencoder", "spatial_mean"):
                    output = model(x)
                    scores, spatial = reconstruction_errors(x, output)
                    display = readonly(output[0].permute(1, 2, 0).numpy())
                    if model_id == "autoencoder":
                        reconstruction = display
                    else:
                        reference = display
                elif model_id == "student_teacher":
                    scores, spatial = model.score(x)
                else:
                    scores, spatial = model(x)
                if (
                    scores.shape != (1,)
                    or spatial.shape != (1, 128, 128)
                    or not torch.isfinite(spatial).all()
                    or (spatial < 0).any()
                ):
                    raise ValueError("Invalid score/map geometry or values.")
                score = float(scores[0])
                flag = threshold_flag(score, spec["threshold"])
                finished = time.perf_counter()  # Real breakpoint: validated model result.
            except (ValueError, RuntimeError, TypeError) as exc:
                raise ModelUnavailable(
                    "Inference failed or produced an invalid output. No result is shown."
                ) from exc
        return InspectionResult(
            model_id,
            spec["name"],
            spec["score_label"],
            score,
            spec["threshold"],
            flag,
            readonly(spatial[0].numpy()),
            readonly(image),
            reconstruction,
            reference,
            (prepared - start) * 1000,
            (finished - loaded) * 1000,
            (loaded - prepared) * 1000,
            (finished - start) * 1000,
            spec["assets"]["weights"]["sha256"],
        )

    def inspect_many(self, payload, model_ids):
        if (
            not model_ids
            or len(model_ids) != len(set(model_ids))
            or any(k not in MODEL_IDS for k in model_ids)
        ):
            raise ModelUnavailable("Choose one or more distinct supported Cashew models.")
        # Explicit caller order; never sort/rank incompatible raw scores.
        return tuple(self.inspect(payload, key) for key in model_ids)
