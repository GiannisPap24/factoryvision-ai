"""Synthetic evaluation breakpoint target; never reads the official test set."""

import torch

from factoryvision.evaluation.pipeline import evaluate_batches, summarize
from factoryvision.models.spatial_mean import SpatialMean


def main():
    torch.set_num_threads(4)
    model = SpatialMean(torch.zeros(3, 4, 4))
    batch = {
        "image": torch.stack((torch.zeros(3, 4, 4), torch.ones(3, 4, 4))),
        "mask": torch.stack((torch.zeros(1, 4, 4), torch.ones(1, 4, 4))),
        "label": torch.tensor([0, 1]),
        "partition": ["test", "test"],
        "sample_id": ["synthetic-a", "synthetic-b"],
    }
    rows, maps, masks = evaluate_batches(model, [batch], batch["sample_id"], 0.5)
    result = summarize(rows, maps, masks, 0.5)
    assert result["image"]["confusion_matrix"] == [[1, 0], [0, 1]]
    assert result["pixel"]["auroc"] == 1
    print(result)


if __name__ == "__main__":
    main()
