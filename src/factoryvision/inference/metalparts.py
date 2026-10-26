"""Frozen native MetalParts inference, independent of Cashew score/threshold contracts."""

import json
import threading
import time
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from factoryvision.data.metalparts import REVISION
from factoryvision.data.metalparts_development import PatchConfig, coordinates
from factoryvision.inference.service import InspectionResult, ModelUnavailable, verified_asset
from factoryvision.inference.uploads import InputRejected, decode_upload, readonly
from factoryvision.training.metalparts import load_model, validate_identity

STATUS = (
    "Experimental MetalParts model — trained on normal source-domain images. "
    "Independent defect detection performance has not been established."
)
DIAGNOSTIC_LABEL = "EXPLORATORY SOURCE-DOMAIN DIAGNOSTICS — NOT INDEPENDENT TEST PERFORMANCE"
CONFIG = PatchConfig()


def catalog():
    return json.loads(
        files("factoryvision.inference").joinpath("metalparts_artifacts.json").read_text()
    )


@torch.inference_mode()
def reconstruct_native(model, image):
    """Batch one, row-major, uniform RGB overlap average, then channel-mean squared error.

    Matches the accepted 3D native diagnostic exactly; no resizing, map smoothing,
    per-image normalization, mask input or sparse selected-patch approximation.
    """
    if image.mode != "RGB" or min(image.size) < CONFIG.size or max(image.size) > 4096:
        raise InputRejected("MetalParts requires RGB images from 256 to 4096 pixels per side.")
    pixels = np.asarray(image).astype(np.float32) / 255
    total = np.zeros_like(pixels)
    count = np.zeros(pixels.shape[:2], dtype=np.float32)
    model.eval()
    for x, y in coordinates(image.width, image.height, CONFIG, full=True):
        patch = torch.from_numpy(pixels[y : y + 256, x : x + 256].copy()).permute(2, 0, 1)[None]
        reconstructed = model(patch)[0].permute(1, 2, 0).numpy()
        if (
            reconstructed.shape != (256, 256, 3)
            or not np.isfinite(reconstructed).all()
            or reconstructed.min() < 0
            or reconstructed.max() > 1
        ):
            raise ModelUnavailable("Invalid MetalParts reconstruction.")
        total[y : y + 256, x : x + 256] += reconstructed
        count[y : y + 256, x : x + 256] += 1
    if not (count > 0).all():
        raise ModelUnavailable("Native reconstruction has uncovered pixels.")
    reconstruction = total / count[..., None]
    spatial = ((pixels - reconstruction) ** 2).mean(axis=2)
    return reconstruction, spatial, float(spatial.mean())


class MetalPartsService:
    """Only a package-pinned local checkpoint; no uploaded models or retained images."""

    def __init__(self, project: Path):
        self.root = Path(project).absolute()
        self.registry = catalog()
        self._model = None
        self._lock = threading.RLock()
        identity = self.registry["identity"]
        validate_identity(identity)
        if (
            identity["dataset"] != "robustad"
            or identity["revision"] != REVISION
            or identity.get("purpose") != "train"
            or identity["patches"] != {"size": 256, "stride": 192, "limit": 12}
        ):
            raise ModelUnavailable("Unsupported MetalParts catalog or category.")

    def _verify(self):
        try:
            paths = {k: verified_asset(self.root, v) for k, v in self.registry["assets"].items()}
            meta = json.loads(paths["metadata"].read_text())
            if meta["identity"] != self.registry["identity"]:
                raise ValueError("identity")
            if paths["checkpoint"].parent != paths["metadata"].parent:
                raise ValueError("location")
            return paths
        except (OSError, ValueError, KeyError) as exc:
            raise ModelUnavailable(
                "MetalParts frozen artifacts are missing or incompatible."
            ) from exc

    def status(self):
        try:
            self._verify()
            return {"available": True, "reason": "Frozen MetalParts artifact files verified"}
        except ModelUnavailable as exc:
            return {"available": False, "reason": str(exc)}

    def _load(self):
        paths = self._verify()
        if self._model is None:
            try:
                self._model, _ = load_model(paths["checkpoint"].parent, self.registry["identity"])
                self._model.requires_grad_(False)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                raise ModelUnavailable(
                    "The frozen MetalParts checkpoint could not be loaded."
                ) from exc
        return self._model

    def inspect(self, payload: bytes, category="metalparts"):
        if category != "metalparts":
            raise ModelUnavailable("This checkpoint supports only experimental MetalParts.")
        start = time.perf_counter()
        image = decode_upload(payload)
        preprocessing = time.perf_counter()
        with self._lock:
            model = self._load()
            loaded = time.perf_counter()
            reconstruction, spatial, score = reconstruct_native(model, image)
        end = time.perf_counter()
        return InspectionResult(
            model_id="metalparts_autoencoder",
            model_name="Experimental MetalParts autoencoder",
            score_label="Native RGB reconstruction MSE (continuous; not a probability)",
            score=score,
            threshold=None,
            above_threshold=None,
            spatial=readonly(spatial),
            original=readonly(np.asarray(image)),
            reconstruction=readonly(reconstruction),
            reference=None,
            preprocessing_ms=(preprocessing - start) * 1000,
            inference_ms=(end - loaded) * 1000,
            artifact_check_load_ms=(loaded - preprocessing) * 1000,
            total_ms=(end - start) * 1000,
            artifact_sha256=self.registry["assets"]["checkpoint"]["sha256"],
        )
