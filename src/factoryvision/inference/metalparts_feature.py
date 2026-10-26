"""Package-pinned normal-source MetalParts memory; continuous scores only."""

import json
import threading
import time
from importlib.resources import files
from pathlib import Path

import numpy as np

from factoryvision.data.metalparts import REVISION
from factoryvision.inference.service import InspectionResult, ModelUnavailable, verified_asset
from factoryvision.inference.uploads import decode_upload, readonly
from factoryvision.models.feature_memory import FeatureMemory, load_extractor
from factoryvision.models.metalparts_memory import native_memory_map
from factoryvision.training.metalparts_memory import load_memory, validate_identity


def catalog():
    return json.loads(
        files("factoryvision.inference")
        .joinpath("metalparts_feature_artifacts.json")
        .read_text(encoding="utf-8")
    )


class MetalPartsFeatureService:
    def __init__(self, project: Path):
        self.root = Path(project).absolute()
        self.registry = catalog()
        self._model = None
        self._lock = threading.RLock()
        try:
            validate_identity(self.registry["identity"])
            if self.registry["identity"]["revision"] != REVISION:
                raise ValueError("revision")
        except (KeyError, ValueError) as exc:
            raise ModelUnavailable("Unsupported MetalParts memory catalog or category.") from exc

    def _verify(self):
        try:
            paths = {k: verified_asset(self.root, v) for k, v in self.registry["assets"].items()}
            meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if meta["identity"] != self.registry["identity"]:
                raise ValueError("identity")
            if paths["memory"].parent != paths["metadata"].parent:
                raise ValueError("location")
            return paths
        except (OSError, ValueError, KeyError) as exc:
            raise ModelUnavailable(
                "MetalParts feature artifacts are missing or incompatible."
            ) from exc

    def status(self):
        try:
            self._verify()
            return {"available": True, "reason": "Frozen MetalParts feature artifacts verified"}
        except ModelUnavailable as exc:
            return {"available": False, "reason": str(exc)}

    def _load(self):
        paths = self._verify()
        if self._model is None:
            try:
                bank, _ = load_memory(paths["memory"].parent, self.registry["identity"])
                self._model = FeatureMemory(load_extractor(paths["weights"]), bank).eval()
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                raise ModelUnavailable("The frozen MetalParts memory could not be loaded.") from exc
        return self._model

    def inspect(self, payload: bytes, category="metalparts"):
        if category != "metalparts":
            raise ModelUnavailable("This memory supports only experimental MetalParts.")
        start = time.perf_counter()
        image = decode_upload(payload)
        preprocessing = time.perf_counter()
        with self._lock:
            model = self._load()
            loaded = time.perf_counter()
            try:
                spatial, score = native_memory_map(model, image)
            except ValueError as exc:
                raise ModelUnavailable(str(exc)) from exc
        end = time.perf_counter()
        return InspectionResult(
            model_id="metalparts_feature_memory",
            model_name="Experimental MetalParts feature memory",
            score_label="Maximum descriptor RMS nearest-neighbor distance (not a probability)",
            score=score,
            threshold=None,
            above_threshold=None,
            spatial=readonly(spatial),
            original=readonly(np.asarray(image)),
            reconstruction=None,
            reference=None,
            preprocessing_ms=(preprocessing - start) * 1000,
            inference_ms=(end - loaded) * 1000,
            artifact_check_load_ms=(loaded - preprocessing) * 1000,
            total_ms=(end - start) * 1000,
            artifact_sha256=self.registry["assets"]["memory"]["sha256"],
        )
