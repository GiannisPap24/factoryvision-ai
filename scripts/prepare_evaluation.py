"""Fit/calibrate the declared statistical baseline and seal artifacts before test inference."""

import json
from datetime import UTC, datetime
from pathlib import Path

import torch

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.protocol import PROTOCOL_SHA256, load_protocol, verify_model_artifacts
from factoryvision.evaluation.reconstruction import calibrate_normal
from factoryvision.models.spatial_mean import fit_spatial_mean
from factoryvision.training.checkpoint import atomic_write, create_run, file_sha256, write_json
from factoryvision.training.engine import score_normals, seed_everything


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = load_protocol(root / "docs/protocols/cashew-evaluation-v1.json")
    verify_model_artifacts(root, protocol)
    review = json.loads((root / "reports/leakage-2b-review.json").read_text())
    if (
        review["confirmed_leakage"]
        or not review["evaluation_permitted"]
        or review["protocol_sha256"] != PROTOCOL_SHA256
        or review["screen_sha256"] != file_sha256(root / "reports/leakage-2b-screen.json")
    ):
        raise ValueError("Integrity/leakage gate not accepted.")
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    if manifest_sha256(manifest) != protocol["dataset"]["manifest_sha256"]:
        raise ValueError("Frozen manifest changed.")
    directory = create_run(root, "cashew-eval-v1")
    seed_everything(42, 4)

    def loader(p):
        return make_loader(
            ManifestDataset(manifest, root / "data/visa", p, ImageSettings(128, 128)), 16
        )

    baseline, fit_ids = fit_spatial_mean(loader(Partition.TRAIN))
    assert len(fit_ids) == 315
    calibration = score_normals(baseline, loader(Partition.CALIBRATION), "calibration")
    assert len(calibration["scores"]) == 68
    threshold = calibrate_normal(torch.tensor(calibration["scores"], dtype=torch.float64), 0.95)
    atomic_write(directory, "baseline-mean.pt", lambda f: torch.save(baseline.reference, f))
    write_json(directory, "baseline-fit-ids.json", fit_ids)
    write_json(directory, "baseline-calibration-scores.json", calibration)
    write_json(directory, "baseline-calibration.json", threshold)
    sources = [
        *root.glob("src/**/*.py"),
        root / "scripts/prepare_evaluation.py",
        root / "scripts/evaluate_cashew.py",
        root / "scripts/review_leakage_2b.py",
    ]
    frozen = {
        "protocol_sha256": PROTOCOL_SHA256,
        "lock_sha256": file_sha256(root / "uv.lock"),
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "official_test_scoring_started": False,
        "manifest_sha256": manifest_sha256(manifest),
        "autoencoder_sha256": protocol["autoencoder"]["sha256"],
        "baseline_sha256": file_sha256(directory / "baseline-mean.pt"),
        "baseline_threshold": threshold["threshold"],
        "baseline_calibration_sha256": file_sha256(directory / "baseline-calibration.json"),
        "audit_sha256": file_sha256(root / "data/metadata/cashew-audit.json"),
        "leakage_review_sha256": file_sha256(root / "reports/leakage-2b-review.json"),
        "execution_sources": {p.relative_to(root).as_posix(): file_sha256(p) for p in sources},
    }
    # Exclusive creation: this seal cannot be replaced by a later invocation.
    with (directory / "frozen-artifacts.json").open("x", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2, allow_nan=False)
        f.write("\n")
    write_json(
        directory,
        "source-snapshot.json",
        {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sources},
    )
    print(
        json.dumps(
            {
                "frozen_artifacts_sha256": file_sha256(directory / "frozen-artifacts.json"),
                "baseline_threshold": threshold,
                "test_predictions_computed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
