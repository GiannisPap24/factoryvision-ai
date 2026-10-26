"""Evaluation-only four-family metrics, immutable seal validation and opening authorization."""

import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from factoryvision.domain.paths import check_relative_path
from factoryvision.evaluation.metrics import binary_metrics, paired_bootstrap
from factoryvision.models.pcb1 import FAMILIES, PROTOCOL_SHA
from factoryvision.training.checkpoint import checked_directory, file_sha256


def verified_seal(root, seal_path, expected_sha):
    checked_directory(root)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha) or file_sha256(seal_path) != expected_sha:
        raise ValueError("Evaluation seal identity mismatch.")
    seal = json.loads(seal_path.read_text())
    if (
        seal.get("schema") != 1
        or seal.get("category") != "pcb1"
        or seal.get("status") != "READY FOR BLINDED PCB1 TEST"
        or seal.get("protocol_sha256") != PROTOCOL_SHA
        or set(seal.get("models", {})) != set(FAMILIES)
    ):
        raise ValueError("Incompatible/unready confirmatory seal.")
    files = seal.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty sealed inventory.")
    for relative, sha in files.items():
        check_relative_path(relative, "sealed.path")
        path = root / relative
        checked_directory(path.parent)
        if path.is_symlink() or not path.is_file() or file_sha256(path) != sha:
            raise ValueError(f"Sealed file identity changed: {relative}")
    for family, model in seal["models"].items():
        directory = model["directory"]
        check_relative_path(directory, "model.directory")
        if PurePosixPath(directory).parts[:1] != ("runs",):
            raise ValueError("Invalid model location.")
        for name in (
            "best.pt",
            "checkpoint.json",
            "pcb1-provenance.json",
            "calibration.json",
            "normal-scores.json",
            "development.json",
        ):
            if f"{directory}/{name}" not in files:
                raise ValueError("Required model artifact missing from seal.")
        provenance = json.loads((root / directory / "pcb1-provenance.json").read_text())
        calibration = json.loads((root / directory / "calibration.json").read_text())
        record = json.loads((root / directory / "normal-scores.json").read_text())["calibration"]
        if (
            provenance["family"] != family
            or provenance["smoke"]
            or provenance["manifest_sha256"] != seal["manifest_sha256"]
            or provenance["fit_ids"] != seal["membership"]["train"]
            or provenance["calibration_ids"] != seal["membership"]["calibration"]
            or record["partition"] != "calibration"
            or record["sample_ids"] != seal["membership"]["calibration"]
            or calibration["threshold"] != model["threshold"]
            or calibration["comparison"] != "score > threshold"
            or calibration["quantile"] != 0.95
            or calibration["interpolation"] != "linear"
            or not np.isfinite(model["threshold"])
            or model["threshold"] < 0
        ):
            raise ValueError("Frozen calibration/model provenance mismatch.")
    groups = [set(seal["membership"][p]) for p in ("train", "validation", "calibration", "test")]
    if any(not g for g in groups) or any(
        a & b for i, a in enumerate(groups) for b in groups[i + 1 :]
    ):
        raise ValueError("Sealed membership overlap/empty partition.")
    return seal


def authorize_opening(approval_path, seal_sha):
    approval = json.loads(Path(approval_path).read_text())
    if (
        approval.get("decision") != "OWNER_AUTHORIZES_PCB1_HELDOUT_OPENING"
        or approval.get("seal_sha256") != seal_sha
        or approval.get("milestone") != "2E2"
    ):
        raise PermissionError("Separate owner authorization bound to this exact seal is required.")
    return approval


def pooled_metrics(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores)
    if (
        y.ndim != 1
        or y.shape != s.shape
        or not 1 <= y.size <= 8_000_000
        or not np.isin(y, (0, 1)).all()
        or not np.isfinite(s).all()
    ):
        raise ValueError("Invalid pooled pixel arrays or resource cap.")
    both = bool(y.min() != y.max())
    return dict(
        pixels=int(y.size),
        positives=int(y.sum()),
        auroc=float(roc_auc_score(y, s)) if both else None,
        average_precision=float(average_precision_score(y, s)) if both else None,
    )


def summarize(labels, masks, scores, maps, thresholds):
    """Exact pooled pixels and paired image bootstrap; no fit or calibration operation."""
    y = np.asarray(labels)
    masks = np.asarray(masks)
    if (
        y.ndim != 1
        or not 1 <= len(y) <= 488
        or masks.shape != (len(y), 128, 128)
        or not np.isin(y, (0, 1)).all()
        or not np.isin(masks, (0, 1)).all()
        or (masks[y == 0] != 0).any()
        or any(set(d) != set(FAMILIES) for d in (scores, maps, thresholds))
    ):
        raise ValueError("Invalid four-family evaluation inputs.")
    results = {}
    for family in FAMILIES:
        spatial = np.asarray(maps[family])
        if spatial.shape != masks.shape:
            raise ValueError("Prediction/mask geometry differs.")
        results[family] = dict(
            image=binary_metrics(y, scores[family], thresholds[family]),
            pixel=pooled_metrics(masks.ravel(), spatial.ravel()),
        )
    intervals, differences = {}, {}
    primary = results["student_teacher"]["pixel"]["average_precision"]
    for family in FAMILIES[:-1]:
        comparator = results[family]["pixel"]["average_precision"]
        differences[family] = (
            primary - comparator if primary is not None and comparator is not None else None
        )
        if len(np.unique(y)) == 2:
            interval = paired_bootstrap(
                y, scores["student_teacher"], scores[family], repeats=1000, seed=42
            )
            for metric in ("auroc", "average_precision"):
                interval[metric]["student_minus_comparator"] = interval[metric].pop("ae_minus_mean")
            intervals[family] = interval
        else:
            intervals[family] = None
    aps = {f: r["pixel"]["average_precision"] for f, r in results.items()}
    ranks = {
        f: 1 + sum(v is not None and v > a for v in aps.values()) if a is not None else None
        for f, a in aps.items()
    }
    return dict(
        models=results,
        descriptive_pixel_ap_ranks=ranks,
        student_minus_comparator_pixel_ap=differences,
        paired_image_intervals=intervals,
        primary="student_teacher pooled pixel AP",
        pixel_intervals=None,
        independence="Unverified physical-item independence; image intervals conditional.",
    )
