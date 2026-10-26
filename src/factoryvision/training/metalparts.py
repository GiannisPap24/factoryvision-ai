"""MetalParts-specific artifacts and native normal diagnostics; shared AE/optimizer engine."""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from factoryvision.data.images import decode_image
from factoryvision.data.metalparts import MetalPartsSource
from factoryvision.data.metalparts_development import PatchConfig, coordinates
from factoryvision.domain.contracts import require_sha256
from factoryvision.domain.paths import check_relative_path
from factoryvision.models.autoencoder import ARCHITECTURE, ConvAutoencoder
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    file_sha256,
    write_json,
)
from factoryvision.training.engine import score_normals, train_epoch

RECIPE = "metalparts-native-patches-ae-v1"


class SourceReadGuard:
    """Python open-event guard for an explicit metadata/normal-image allowlist."""

    def __init__(self, root: Path, references):
        self.root = root.resolve()
        for reference in references:
            check_relative_path(reference, "allowlist")
            if (
                reference != "MetalParts/metal_parts_data_dir_train/metadata.jsonl"
                and not reference.startswith("MetalParts/metal_parts_data_dir_train/normal/")
            ):
                raise ValueError(
                    "Only source-train metadata and normal image paths may be allowlisted."
                )
        self.allowed = {(self.root / p).resolve() for p in references}
        if not self.allowed or any(not p.is_relative_to(self.root) for p in self.allowed):
            raise ValueError("Empty or escaping source allowlist.")
        self.opened = set()

    def __call__(self, event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if path.is_relative_to(self.root):
                if path not in self.allowed:
                    raise PermissionError(
                        "Unapproved source read: test/anomaly/mask access prohibited."
                    )
                self.opened.add(path.relative_to(self.root).as_posix())


def artifact_identity(source, split_hash, protocol_hash, config: PatchConfig):
    if not isinstance(source, MetalPartsSource):
        raise ValueError("Require MetalParts source identity.")
    require_sha256(split_hash, "split_hash")
    require_sha256(protocol_hash, "protocol_hash")
    return {
        "schema": 1,
        "dataset": source.dataset_id,
        "category": "metalparts",
        "revision": source.revision,
        "architecture": ARCHITECTURE,
        "recipe": RECIPE,
        "patches": asdict(config),
        "split_sha256": split_hash,
        "protocol_sha256": protocol_hash,
        "threshold": None,
        "status": "EXPERIMENTAL; independent evaluation NOT CLEARED",
    }


def validate_identity(identity):
    expected = artifact_identity(
        MetalPartsSource(identity["revision"]),
        identity["split_sha256"],
        identity["protocol_sha256"],
        PatchConfig(**identity["patches"]),
    )
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("Invalid MetalParts artifact identity or unauthorized calibration.")


def save_model(directory, model, identity, epoch, loss):
    validate_identity(identity)
    if model.resolution != identity["patches"]["size"] or identity["threshold"] is not None:
        raise ValueError("Geometry mismatch or unauthorized calibration.")
    atomic_write(directory, "best.pt", lambda stream: torch.save(model.state_dict(), stream))
    meta = {
        "identity": identity,
        "epoch": epoch,
        "validation_loss": loss,
        "checkpoint_sha256": file_sha256(directory / "best.pt"),
    }
    write_json(directory, "checkpoint.json", meta)
    return meta


def load_model(directory, identity):
    validate_identity(identity)
    checked_directory(directory)
    for name, limit in (("checkpoint.json", 65536), ("best.pt", 16 * 1024**2)):
        p = directory / name
        if not p.is_file() or p.is_symlink() or p.is_junction() or p.stat().st_size > limit:
            raise ValueError("Missing, linked or oversized MetalParts artifact.")
    meta = json.loads((directory / "checkpoint.json").read_text())
    if (
        meta.get("identity") != identity
        or identity.get("threshold") is not None
        or identity.get("recipe") != RECIPE
    ):
        raise ValueError("MetalParts source/split/recipe compatibility mismatch.")
    if file_sha256(directory / "best.pt") != meta.get("checkpoint_sha256"):
        raise ValueError("Checkpoint hash mismatch.")
    model = ConvAutoencoder(identity["patches"]["size"])
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    reference = model.state_dict()
    if not isinstance(state, dict) or state.keys() != reference.keys():
        raise ValueError("Checkpoint tensor keys mismatch.")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference[key].shape
            or value.dtype != reference[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid checkpoint tensor.")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, meta


class BudgetLoader:
    """Shared engine sees a normal loader; check wall time and measured memory each batch."""

    def __init__(self, loader, deadline, memory, limit=8 * 1024**3):
        self.loader, self.deadline, self.memory, self.limit = loader, deadline, memory, limit

    def __iter__(self):
        for batch in self.loader:
            if time.perf_counter() > self.deadline or self.memory() > self.limit:
                raise TimeoutError("Development time or resident-memory budget exceeded.")
            yield batch


def fit(
    model,
    optimizer,
    training_loader,
    validation_loader,
    directory,
    identity,
    epochs,
    deadline,
    memory,
):
    if type(epochs) is not int or not 1 <= epochs <= 12:
        raise ValueError("One pilot permits 1–12 epochs.")
    before = score_normals(model, BudgetLoader(validation_loader, deadline, memory), "validation")
    history, best = [], float("inf")
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        training = train_epoch(
            model,
            optimizer,
            BudgetLoader(training_loader(epoch), deadline, memory),
            deadline=deadline,
        )
        validation = score_normals(
            model, BudgetLoader(validation_loader, deadline, memory), "validation"
        )
        if validation["mean"] < best:
            best = validation["mean"]
            save_model(directory, model, identity, epoch, best)
            write_json(directory, "selected-validation.json", validation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": training["loss"],
                "validation_loss": validation["mean"],
                "steps": training["steps"],
                "patches": training["samples"],
                "seconds": time.perf_counter() - start,
                "peak_working_set_bytes": memory(),
            }
        )
        write_json(
            directory,
            "history.json",
            {"initial_validation_loss": before["mean"], "epochs": history},
        )
        print(json.dumps(history[-1]), flush=True)
    return before, history


@torch.inference_mode()
def native_diagnostic(model, root, sample, config):
    """Full native tiling of one permitted development parent, uniform overlap averaging."""
    from factoryvision.data.metalparts_development import DevelopmentSample, verified_bytes

    if not isinstance(sample, DevelopmentSample):
        raise ValueError("Require a verified normal development record.")
    path = root / sample.image_path
    verified_bytes(
        root, sample.image_path, {"bytes": path.stat().st_size, "sha256": sample.content_sha256}
    )
    image = decode_image(root, sample.image_path)
    if image.size != (sample.sample.width, sample.sample.height):
        raise ValueError("Native geometry mismatch.")
    pixels = np.asarray(image).astype(np.float32) / 255
    total = np.zeros_like(pixels)
    count = np.zeros(pixels.shape[:2], dtype=np.float32)
    model.eval()
    for x, y in coordinates(image.width, image.height, config, full=True):
        patch = torch.from_numpy(pixels[y : y + config.size, x : x + config.size].copy()).permute(
            2, 0, 1
        )[None]
        reconstructed = model(patch)[0].permute(1, 2, 0).numpy()
        if reconstructed.shape != patch.shape[2:] + (3,) or not np.isfinite(reconstructed).all():
            raise ValueError("Invalid reconstruction geometry or values.")
        total[y : y + config.size, x : x + config.size] += reconstructed
        count[y : y + config.size, x : x + config.size] += 1
    if not (count > 0).all():
        raise ValueError("Native reconstruction has uncovered pixels.")
    reconstruction = total / count[..., None]
    spatial = ((pixels - reconstruction) ** 2).mean(axis=2)
    return reconstruction, spatial, float(spatial.mean())
