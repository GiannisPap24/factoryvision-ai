"""Offline synthetic membership, capped fitting, family identity and immutable reload."""

import json

import pytest
import torch
from torch import nn

from factoryvision.models import pcb1
from factoryvision.models.spatial_mean import SpatialMean


class Extractor(nn.Module):
    def forward(self, images):
        return (
            torch.arange(384 * 16 * 16, dtype=torch.float32)
            .reshape(1, 384, 16, 16)
            .expand(len(images), -1, -1, -1)
            / 10000
        )


def batch(ids):
    return dict(
        image=torch.zeros(len(ids), 3, 128, 128),
        label=torch.zeros(len(ids)),
        partition=["train"] * len(ids),
        sample_id=ids,
    )


def test_capped_bank_every_image_and_repeatability(monkeypatch, tmp_path):
    monkeypatch.setattr(pcb1, "MAX_BANK", 24)
    ids = ["pcb1-0", "pcb1-1"]
    model, selection = pcb1.fit_capped_memory(Extractor(), [batch(ids)], ids)
    other, repeated = pcb1.fit_capped_memory(Extractor(), [batch(ids)], ids)
    assert model.bank.shape == (24, 384)
    assert selection == repeated and torch.equal(model.bank, other.bank)
    pcb1.save_reference(tmp_path, "feature_memory", model, "a" * 64, ids, selection)
    monkeypatch.setattr(pcb1, "load_extractor", lambda _: Extractor())
    reloaded = pcb1.load_family(tmp_path, "feature_memory", "a" * 64, ids, None)
    assert torch.equal(reloaded.bank, model.bank)
    score, spatial = pcb1.predict(reloaded, "feature_memory", torch.zeros(1, 3, 128, 128))
    assert score.shape == (1,) and spatial.shape == (1, 128, 128)
    meta = json.loads((tmp_path / "checkpoint.json").read_text())
    meta["selection"][0]["patch_indices"][0] += 1
    (tmp_path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="membership"):
        pcb1.load_family(tmp_path, "feature_memory", "a" * 64, ids, None)


def test_fitting_rejections(monkeypatch):
    ids = ["pcb1-0"]
    with pytest.raises(ValueError, match="Incomplete"):
        pcb1.fit_capped_memory(Extractor(), [], ids)
    wrong = batch(ids)
    wrong["partition"] = ["test"]
    with pytest.raises(ValueError, match="approved"):
        pcb1.fit_capped_memory(Extractor(), [wrong], ids)
    with pytest.raises(ValueError, match="geometry"):
        pcb1.fit_capped_memory(nn.Identity(), [batch(ids)], ids)
    monkeypatch.setattr(pcb1, "MAX_BANK", 0)
    with pytest.raises(ValueError, match="cap"):
        pcb1.fit_capped_memory(Extractor(), [], ids)


def test_reference_identity_tampering_and_scoring(tmp_path):
    ids = ["pcb1-0"]
    model = SpatialMean(torch.zeros(3, 128, 128))
    pcb1.save_reference(tmp_path, "spatial_mean", model, "a" * 64, ids)
    loaded = pcb1.load_family(tmp_path, "spatial_mean", "a" * 64, ids, None)
    scores, maps = pcb1.predict(loaded, "spatial_mean", torch.ones(1, 3, 128, 128))
    assert scores.tolist() == [1.0] and maps.min() == 1
    for family, sha, identities in (
        ("spatial_mean", "b" * 64, ids),
        ("feature_memory", "a" * 64, ids),
        ("spatial_mean", "a" * 64, ["cashew-0"]),
    ):
        with pytest.raises(ValueError, match="compatibility"):
            pcb1.load_family(tmp_path, family, sha, identities, None)
    with (tmp_path / "best.pt").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        pcb1.load_family(tmp_path, "spatial_mean", "a" * 64, ids, None)
    with pytest.raises(ValueError, match="Unknown"):
        pcb1.predict(model, "other", torch.ones(1, 3, 128, 128))
    with pytest.raises(ValueError, match="Unknown"):
        pcb1.load_family(tmp_path, "other", "a" * 64, ids, None)
    with pytest.raises(ValueError, match="Unsupported"):
        pcb1.save_reference(tmp_path, "other", model, "a" * 64, ids)


def test_architecture_dispatch_and_bad_outputs(monkeypatch):
    model = SpatialMean(torch.zeros(3, 128, 128))
    monkeypatch.setattr(pcb1, "load_checkpoint", lambda *a: (model, {}))
    assert pcb1.load_family(None, "autoencoder", "a" * 64, ["pcb1-0"], None) is model
    assert pcb1.predict(model, "autoencoder", torch.ones(1, 3, 128, 128))[0].item() == 1

    class FakeStudent(nn.Module):
        def score(self, image):
            return torch.zeros(len(image)), torch.zeros(len(image), 128, 128)

    monkeypatch.setattr(pcb1, "load_student", lambda *a: (nn.Identity(), {}))
    monkeypatch.setattr(pcb1, "load_extractor", lambda *a: nn.Identity())
    monkeypatch.setattr(pcb1, "SpatialTeacher", lambda x: x)
    monkeypatch.setattr(pcb1, "StudentTeacher", lambda *a: FakeStudent())
    restored = pcb1.load_family(None, "student_teacher", "a" * 64, ["pcb1-0"], None)
    assert pcb1.predict(restored, "student_teacher", torch.zeros(1, 3, 128, 128))[0].item() == 0

    class Bad(nn.Module):
        def forward(self, x):
            return torch.tensor([float("nan")]), torch.zeros(1, 128, 128)

    with pytest.raises(ValueError, match="outputs"):
        pcb1.predict(Bad(), "feature_memory", torch.zeros(1, 3, 128, 128))


@pytest.mark.parametrize(
    "state,error", [({}, "state"), ({"reference": torch.zeros(3, 16, 16)}, "geometry")]
)
def test_malformed_reference_state(tmp_path, state, error):
    model = SpatialMean(torch.zeros(3, 128, 128))
    ids = ["pcb1-0"]
    pcb1.save_reference(tmp_path, "spatial_mean", model, "a" * 64, ids)
    torch.save(state, tmp_path / "best.pt")
    meta = json.loads((tmp_path / "checkpoint.json").read_text())
    meta["checkpoint_sha256"] = pcb1.file_sha256(tmp_path / "best.pt")
    (tmp_path / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match=error):
        pcb1.load_family(tmp_path, "spatial_mean", "a" * 64, ids, None)
    (tmp_path / "best.pt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        pcb1.load_family(tmp_path, "spatial_mean", "a" * 64, ids, None)
