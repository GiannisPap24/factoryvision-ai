"""Hash-anchored v1 protocol; changes require a separately named predeclared protocol."""

import json

from factoryvision.training.checkpoint import file_sha256

PROTOCOL_SHA256 = "3b89b8447f00e478ab5d7793a50ca4cb92f3d0804bd6bee9fada564618adaa81"


def load_protocol(path):
    if file_sha256(path) != PROTOCOL_SHA256:
        raise ValueError("Frozen evaluation protocol hash mismatch.")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_model_artifacts(root, protocol):
    ae = protocol["autoencoder"]
    pairs = {
        ae["checkpoint"]: ae["sha256"],
        "runs/cashew-ae-128-seed42/checkpoint.json": ae["metadata_sha256"],
        "runs/cashew-ae-128-seed42/calibration.json": ae["calibration_sha256"],
        "runs/cashew-ae-128-seed42/calibration-scores.json": ae["calibration_scores_sha256"],
    }
    for name, digest in pairs.items():
        if file_sha256(root / name) != digest:
            raise ValueError("Frozen model/calibration artifact changed.")
