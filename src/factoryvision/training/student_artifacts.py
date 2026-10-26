"""Student-only state dictionaries bound to teacher, dataset and frozen recipe."""

import json
import math
import re

import torch

from factoryvision.models.feature_memory import WEIGHTS_SHA256, checked_ids
from factoryvision.models.student_teacher import ARCHITECTURE, CompactStudent
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    file_sha256,
    write_json,
)


def provenance(manifest_sha, recipe_sha, fit_ids):
    checked_ids(fit_ids)
    if not all(
        isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in (manifest_sha, recipe_sha)
    ):
        raise ValueError("Invalid student provenance hashes.")
    return {
        "schema": 1,
        "architecture": ARCHITECTURE,
        "manifest_sha256": manifest_sha,
        "recipe_sha256": recipe_sha,
        "teacher_sha256": WEIGHTS_SHA256,
        "fit_ids": fit_ids,
    }


def save_student(directory, student, manifest_sha, recipe_sha, fit_ids, epoch, loss):
    checked_directory(directory)
    meta = provenance(manifest_sha, recipe_sha, fit_ids)
    if type(epoch) is not int or not 1 <= epoch <= 12 or not math.isfinite(loss) or loss < 0:
        raise ValueError("Invalid checkpoint selection metadata.")
    state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Nonfinite student state.")
    atomic_write(directory, "best.pt", lambda f: torch.save(state, f))
    meta.update(
        epoch=epoch, validation_loss=loss, checkpoint_sha256=file_sha256(directory / "best.pt")
    )
    write_json(directory, "checkpoint.json", meta)
    return meta


def load_student(directory, manifest_sha, recipe_sha, fit_ids):
    checked_directory(directory)
    expected = provenance(manifest_sha, recipe_sha, fit_ids)
    for name, limit in (("checkpoint.json", 262144), ("best.pt", 8 * 1024**2)):
        path = directory / name
        if (
            path.is_symlink()
            or path.is_junction()
            or not path.is_file()
            or path.stat().st_size > limit
        ):
            raise ValueError("Missing/linked/oversized student artifact.")
    meta = json.loads((directory / "checkpoint.json").read_text(encoding="utf-8"))
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Student teacher/dataset/recipe compatibility mismatch.")
    if file_sha256(directory / "best.pt") != meta.get("checkpoint_sha256"):
        raise ValueError("Student checkpoint hash mismatch.")
    if (
        type(meta.get("epoch")) is not int
        or not 1 <= meta["epoch"] <= 12
        or not isinstance(meta.get("validation_loss"), (int, float))
        or not math.isfinite(meta["validation_loss"])
        or meta["validation_loss"] < 0
    ):
        raise ValueError("Invalid checkpoint selection metadata.")
    state = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    student = CompactStudent()
    reference = student.state_dict()
    if not isinstance(state, dict) or state.keys() != reference.keys():
        raise ValueError("Student state keys mismatch.")
    for key, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference[key].shape
            or value.dtype != reference[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Invalid student state tensor.")
    student.load_state_dict(state, strict=True)
    student.eval()
    return student, meta
