"""Debug the actual frozen feature model on fabricated RGB; no real image access."""

import json
from pathlib import Path

import torch

from factoryvision.models.feature_memory import FeatureMemory, load_extractor
from factoryvision.training.feature_artifacts import load_memory


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/cashew-feature-v1"
    meta = json.loads((run / "memory.json").read_text())
    bank, _ = load_memory(run, meta["manifest_sha256"], meta["recipe_sha256"], meta["fit_ids"])
    torch.set_num_threads(4)
    model = FeatureMemory(load_extractor(root / "data/pretrained/resnet18-f37072fd.pth"), bank)
    image = torch.zeros(1, 3, 128, 128)
    image[:, :, 48:64, 48:64] = 1
    score, spatial = model(image)
    assert score.shape == (1,) and spatial.shape == (1, 128, 128)
    assert torch.isfinite(spatial).all()
    print("Fabricated RGB debugger probe only; no real images or performance claim.")
    print({"score": score.tolist(), "map_shape": list(spatial.shape), "bank": list(bank.shape)})


if __name__ == "__main__":
    main()
