"""VS Code breakpoint probe on one permitted normal validation image; no fitting."""

import json

from pcb1_experiment import ROOT, WEIGHTS, context

from factoryvision.models.pcb1 import load_family, predict
from factoryvision.training.engine import seed_everything

seed_everything(42, 4)
fingerprint, datasets, ids, guard = context()
model = load_family(
    ROOT / "runs/pcb1-autoencoder-v1", "autoencoder", fingerprint, ids["train"], WEIGHTS
)
sample = datasets["validation"][0]
scores, maps = predict(model, "autoencoder", sample["image"][None])
result = {  # Set a real breakpoint here after inference.
    "sample_id": sample["sample_id"],
    "partition": sample["partition"],
    "score": float(scores[0]),
    "map_shape": list(maps.shape),
    "test_images_opened": 0,
}
print(json.dumps(result, indent=2))
