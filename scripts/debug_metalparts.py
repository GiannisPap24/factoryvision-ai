"""Synthetic 256-pixel real-gradient debugger probe; no dataset or run artifacts."""

import torch

from factoryvision.models.autoencoder import ConvAutoencoder
from factoryvision.training.engine import parameter_sha256, seed_everything, train_step

seed_everything(42, 4)
model = ConvAutoencoder(256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch = {
    "image": torch.linspace(0, 1, 3 * 256 * 256).reshape(1, 3, 256, 256),
    "label": torch.zeros(1, dtype=torch.int64),
    "partition": ["train"],
    "sample_id": ["synthetic-metalparts-debug"],
}
before = parameter_sha256(model)
loss = train_step(model, optimizer, batch)
assert parameter_sha256(model) != before
print(f"Synthetic MetalParts geometry: loss={loss:.8f}; real update verified; no dataset opened")
