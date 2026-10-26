"""Category-specific artifact adapters for the four unchanged accepted architectures."""

import hashlib
import json

import torch

from factoryvision.evaluation.reconstruction import reconstruction_errors
from factoryvision.models.feature_memory import (
    DIMENSIONS,
    MAX_BANK,
    FeatureMemory,
    checked_features,
    checked_ids,
    checked_rgb,
    load_extractor,
    require_development_batch,
)
from factoryvision.models.spatial_mean import SpatialMean
from factoryvision.models.student_teacher import SpatialTeacher, StudentTeacher
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    file_sha256,
    load_checkpoint,
    write_json,
)
from factoryvision.training.student_artifacts import load_student

FAMILIES = ("autoencoder", "spatial_mean", "feature_memory", "student_teacher")
PROTOCOL_SHA = "c7b59ae7d88e1ec32009067cae7f018e7ad2f5106f2737d6985563fe81ef38bc"
STUDENT_SHA = "fb5fd19e04aaa737de16b0483ca3f218af50aaf47707549dbc7c0345a42b7779"


@torch.inference_mode()
def fit_capped_memory(extractor, loader, expected_ids):
    checked_ids(expected_ids)
    k = min(16, MAX_BANK // len(expected_ids))
    if k < 1:
        raise ValueError("Fitting membership exceeds memory cap.")
    chunks, selected, offset = [], [], 0
    for batch in loader:
        require_development_batch(batch, "train", expected_ids, offset)
        features = extractor(batch["image"])
        if features.shape != (len(batch["image"]), DIMENSIONS, 16, 16):
            raise ValueError("Invalid feature geometry.")
        for identity, feature in zip(batch["sample_id"], features, strict=True):
            patches = feature.permute(1, 2, 0).reshape(256, DIMENSIONS)
            checked_features(patches)
            seed = int.from_bytes(hashlib.sha256(f"42|{identity}".encode()).digest()[:8], "big")
            indices = (
                torch.randperm(256, generator=torch.Generator().manual_seed(seed))[:k].sort().values
            )
            chunks.append(patches[indices].clone())
            selected.append({"sample_id": identity, "patch_indices": indices.tolist()})
        offset += len(batch["image"])
    if offset != len(expected_ids):
        raise ValueError("Incomplete fitting membership.")
    return FeatureMemory(extractor, torch.cat(chunks)), selected


def save_reference(directory, family, model, manifest_sha, fit_ids, selection=None):
    checked_ids(fit_ids)
    if family not in ("spatial_mean", "feature_memory"):
        raise ValueError("Unsupported reference family.")
    tensor = model.reference if family == "spatial_mean" else model.bank
    atomic_write(
        directory, "best.pt", lambda f: torch.save({"reference": tensor.detach().clone()}, f)
    )
    write_json(
        directory,
        "checkpoint.json",
        dict(
            schema=1,
            family=family,
            manifest_sha256=manifest_sha,
            protocol_sha256=PROTOCOL_SHA,
            fit_ids=fit_ids,
            selection=selection,
            checkpoint_sha256=file_sha256(directory / "best.pt"),
        ),
    )


def load_family(directory, family, manifest_sha, fit_ids, weights):
    checked_ids(fit_ids)
    if family == "autoencoder":
        return load_checkpoint(directory, manifest_sha, 128)[0]
    if family == "student_teacher":
        student, _ = load_student(directory, manifest_sha, STUDENT_SHA, fit_ids)
        return StudentTeacher(SpatialTeacher(load_extractor(weights)), student).eval()
    if family not in ("spatial_mean", "feature_memory"):
        raise ValueError("Unknown model family.")
    checked_directory(directory)
    for name, limit in (("checkpoint.json", 2_000_000), ("best.pt", 16_000_000)):
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise ValueError("Missing, linked or oversized reference artifact.")
    meta = json.loads((directory / "checkpoint.json").read_text())
    expected = dict(
        schema=1,
        family=family,
        manifest_sha256=manifest_sha,
        protocol_sha256=PROTOCOL_SHA,
        fit_ids=fit_ids,
        checkpoint_sha256=file_sha256(directory / "best.pt"),
    )
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Reference compatibility/hash mismatch.")
    state = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    if (
        not isinstance(state, dict)
        or set(state) != {"reference"}
        or not isinstance(state["reference"], torch.Tensor)
    ):
        raise ValueError("Invalid reference state.")
    tensor = state["reference"]
    if family == "spatial_mean":
        if tensor.shape != (3, 128, 128) or meta["selection"] is not None:
            raise ValueError("Invalid spatial reference geometry.")
        return SpatialMean(tensor).eval()
    k = min(16, MAX_BANK // len(fit_ids))
    expected_selection = []
    for identity in fit_ids:
        seed = int.from_bytes(hashlib.sha256(f"42|{identity}".encode()).digest()[:8], "big")
        idx = torch.randperm(256, generator=torch.Generator().manual_seed(seed))[:k].sort().values
        expected_selection.append(dict(sample_id=identity, patch_indices=idx.tolist()))
    if tensor.shape != (len(fit_ids) * k, DIMENSIONS) or meta["selection"] != expected_selection:
        raise ValueError("Invalid memory membership/geometry.")
    return FeatureMemory(load_extractor(weights), tensor)


@torch.inference_mode()
def predict(model, family, image):
    checked_rgb(image)
    model.eval()
    if family in ("autoencoder", "spatial_mean"):
        scores, maps = reconstruction_errors(image, model(image))
    elif family == "feature_memory":
        scores, maps = model(image)
    elif family == "student_teacher":
        scores, maps = model.score(image)
    else:
        raise ValueError("Unknown model family.")
    if (
        scores.shape != (len(image),)
        or maps.shape != (len(image), 128, 128)
        or not torch.isfinite(scores).all()
        or not torch.isfinite(maps).all()
        or (scores < 0).any()
        or (maps < 0).any()
    ):
        raise ValueError("Invalid anomaly outputs.")
    return scores, maps
