"""Offline synthetic feature-memory tests; never require real data or pretrained downloads."""

import json

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from factoryvision.data.development import (
    DevelopmentDataset,
    DevelopmentReadGuard,
    synthetic_defect,
)
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.feature_development import calibrate_development, score_development
from factoryvision.models.feature_memory import (
    FeatureMemory,
    FrozenResNet18,
    checked_features,
    checked_ids,
    checked_rgb,
    fit_memory,
    load_extractor,
    nearest_distances,
    require_development_batch,
)
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.feature_artifacts import load_memory, save_memory


class TinyFeatures(nn.Module):
    def forward(self, image):
        value = image.mean((1, 2, 3))[:, None, None, None]
        return value.expand(-1, 384, 16, 16)


def batch(partition="train"):
    return {
        "image": torch.stack((torch.zeros(3, 128, 128), torch.ones(3, 128, 128))),
        "partition": [partition] * 2,
        "label": torch.zeros(2, dtype=torch.int64),
        "sample_id": ["synthetic-a", "synthetic-b"],
    }


@pytest.fixture(scope="module", autouse=True)
def bounded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def test_resnet_shapes_frozen_gradients_and_batchnorm():
    torch.manual_seed(42)
    model = FrozenResNet18(resnet18(weights=None))
    before = parameter_sha256(model)
    model.train()
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(not m.training for m in model.modules())
    features = model(batch()["image"])
    assert features.shape == (2, 384, 16, 16) and not features.requires_grad
    assert before == parameter_sha256(model)
    torch.testing.assert_close(features, model(batch()["image"]), atol=0, rtol=0)


@pytest.mark.parametrize(
    "value",
    [
        None,
        torch.zeros(1, 3, 64, 64),
        torch.zeros(0, 3, 128, 128),
        torch.zeros(17, 3, 128, 128),
        torch.ones(1, 3, 128, 128, dtype=torch.float64),
        torch.full((1, 3, 128, 128), float("nan")),
        torch.full((1, 3, 128, 128), -1.0),
        torch.full((1, 3, 128, 128), 2.0),
    ],
)
def test_invalid_rgb(value):
    with pytest.raises(ValueError):
        checked_rgb(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        torch.zeros(0, 384),
        torch.zeros(1, 385),
        torch.zeros(1, 384, 1),
        torch.zeros(8193, 384),
        torch.zeros(1, 384, dtype=torch.float64),
        torch.full((1, 384), float("inf")),
        torch.full((1, 384), 1e7),
    ],
)
def test_invalid_features(value):
    with pytest.raises(ValueError):
        checked_features(value, bank=True)


def test_distance_exact_known_answers_and_chunk_equivalence():
    bank = torch.stack((torch.zeros(384), torch.ones(384)))
    query = torch.stack((torch.full((384,), 0.25), torch.full((384,), 2.0)))
    expected = torch.tensor([0.25, 1.0])
    for qc, bc in [(1, 1), (2, 2), (256, 512)]:
        torch.testing.assert_close(nearest_distances(query, bank, qc, bc), expected, rtol=0, atol=0)
    torch.manual_seed(5)
    q, b = torch.rand(7, 384), torch.rand(11, 384)
    direct = ((q[:, None].double() - b[None].double()).square().mean(2)).min(1).values.sqrt()
    torch.testing.assert_close(nearest_distances(q, b, 3, 4).double(), direct, atol=1e-6, rtol=1e-6)
    with pytest.raises(ValueError):
        nearest_distances(q, b, 513, 1)
    with pytest.raises(ValueError):
        nearest_distances(q, b, 1, False)


def test_fit_deterministic_patch_selection_membership_and_batching():
    b = batch()
    model, selected = fit_memory(TinyFeatures(), [b], b["sample_id"])
    assert model.bank.shape == (32, 384)
    assert len(selected) == 2
    assert all(len(set(x["patch_indices"])) == 16 for x in selected)
    singles = [{k: v[i : i + 1] for k, v in b.items()} for i in range(2)]
    again, selection = fit_memory(TinyFeatures(), singles, b["sample_id"])
    assert selection == selected
    torch.testing.assert_close(model.bank, again.bank, rtol=0, atol=0)
    score, spatial = model(torch.full((1, 3, 128, 128), 0.25))
    torch.testing.assert_close(score, torch.tensor([0.25]), rtol=0, atol=0)
    torch.testing.assert_close(spatial, torch.full((1, 128, 128), 0.25), rtol=0, atol=1e-7)


def test_spatial_alignment_and_max_image_scoring():
    class Spatial(nn.Module):
        def forward(self, image):
            output = torch.zeros(len(image), 384, 16, 16)
            output[:, :, 3, 5] = 1
            return output

    model = FeatureMemory(Spatial(), torch.zeros(1, 384))
    score, spatial = model(torch.zeros(1, 3, 128, 128))
    assert score.item() == 1
    y, x = divmod(int(spatial[0].argmax()), 128)
    assert 24 <= y < 32 and 40 <= x < 48
    assert spatial[0, 0, 0] == 0 and spatial.max() <= 1


@pytest.mark.parametrize("partition", ["validation", "calibration", "test"])
def test_fit_refuses_other_partitions_before_extraction(partition):
    class Forbidden(nn.Module):
        def forward(self, image):
            raise AssertionError("Must reject before feature extraction")

    b = batch(partition)
    with pytest.raises(ValueError):
        fit_memory(Forbidden(), [b], b["sample_id"])


@pytest.mark.parametrize("ids", [[], ["b", "a"], ["a", "a"], [1], [""]])
def test_invalid_ids(ids):
    with pytest.raises(ValueError):
        checked_ids(ids)


def test_fit_bounds_and_bad_batch_geometry():
    b = batch()
    with pytest.raises(ValueError):
        fit_memory(TinyFeatures(), [b], [f"id-{i:04d}" for i in range(513)])
    with pytest.raises(ValueError):
        fit_memory(TinyFeatures(), [], b["sample_id"])
    b["label"][0] = 1
    with pytest.raises(ValueError):
        fit_memory(TinyFeatures(), [b], b["sample_id"])
    b = batch()
    with pytest.raises(ValueError):
        require_development_batch(b, "test", b["sample_id"], 0)
    with pytest.raises(ValueError):
        require_development_batch(b, "train", ["wrong", "ids"], 0)

    class Wrong(nn.Module):
        def forward(self, image):
            return torch.zeros(len(image), 3, 16, 16)

    with pytest.raises(ValueError):
        fit_memory(Wrong(), [b], b["sample_id"])
    with pytest.raises(ValueError):
        FeatureMemory(Wrong(), torch.zeros(1, 384))(b["image"])


def test_normal_scoring_and_calibration_isolation():
    model = FeatureMemory(TinyFeatures(), torch.zeros(1, 384))
    b = batch("calibration")
    record = score_development(model, [b], "calibration", b["sample_id"])
    assert record["scores"] == [0, 1]
    calibration = calibrate_development(record, b["sample_id"])
    assert calibration["threshold"] == pytest.approx(0.95) and calibration["exceedances"] == 1
    b["partition"] = ["validation"] * 2
    validation = score_development(model, [b], "validation", b["sample_id"])
    with pytest.raises(ValueError):
        calibrate_development(validation, b["sample_id"])
    with pytest.raises(ValueError):
        calibrate_development(record, ["wrong"])
    with pytest.raises(ValueError):
        calibrate_development({**record, "scores": []}, b["sample_id"])
    for partition in ("test", "train"):
        with pytest.raises(ValueError):
            score_development(model, [b], partition, b["sample_id"])
    with pytest.raises(ValueError):
        score_development(model, [], "validation", b["sample_id"])

    class BadScores(nn.Module):
        def forward(self, image):
            return torch.tensor([float("nan")] * len(image)), None

    with pytest.raises(ValueError):
        score_development(BadScores(), [b], "validation", b["sample_id"])


def test_development_adapter_never_decodes_test_images(manifest, tmp_path, monkeypatch):
    # Metadata-only synthetic manifest: constructor can select train without opening images.
    from factoryvision.data import images

    opened = []

    def decode(root, sample):
        opened.append(sample.sample_id)
        raise RuntimeError("synthetic decoder spy")

    monkeypatch.setattr(images, "decode_pair", decode)
    with pytest.raises(ValueError):
        DevelopmentDataset(manifest, tmp_path, Partition.TEST)
    assert not opened
    train = DevelopmentDataset(manifest, tmp_path, Partition.TRAIN)
    with pytest.raises(RuntimeError, match="decoder spy"):
        train[0]
    assert opened == [train.samples[0].sample_id]


def test_synthetic_defect_repeatability_geometry_and_invalid_input():
    original = torch.full((3, 128, 128), 0.5)
    image, mask = synthetic_defect(original, "normal-development-a")
    second, truth = synthetic_defect(original, "normal-development-a")
    assert torch.equal(image, second) and torch.equal(mask, truth)
    assert mask.sum() == 256 and torch.equal((image != original).any(0), mask.bool())
    assert original.eq(0.5).all() and image.min() == 0 and image.max() == 1
    with pytest.raises(ValueError):
        synthetic_defect(original, "")
    with pytest.raises(ValueError):
        synthetic_defect(torch.zeros(3, 4, 4), "id")


def test_pretrained_weights_missing_corrupt_and_strict_state(tmp_path):
    path = tmp_path / "synthetic-resnet.pt"
    with pytest.raises(ValueError):
        load_extractor(path)
    path.write_bytes(b"corrupt synthetic fixture")
    with pytest.raises(ValueError):
        load_extractor(path)
    torch.manual_seed(42)
    state = resnet18(weights=None).state_dict()
    torch.save(state, path)
    model = load_extractor(path, file_sha256(path))
    assert model(torch.zeros(1, 3, 128, 128)).shape == (1, 384, 16, 16)
    legacy = {k: v for k, v in state.items() if not k.endswith(".num_batches_tracked")}
    torch.save(legacy, path)
    restored = load_extractor(path, file_sha256(path))
    assert parameter_sha256(restored) == parameter_sha256(model)
    legacy.pop("conv1.weight")
    torch.save(legacy, path)
    with pytest.raises(ValueError, match="keys"):
        load_extractor(path, file_sha256(path))
    torch.save({}, path)
    with pytest.raises(ValueError, match="keys"):
        load_extractor(path, file_sha256(path))
    state["conv1.weight"][0, 0, 0, 0] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError, match="tensor"):
        load_extractor(path, file_sha256(path))


def test_memory_artifact_roundtrip_provenance_and_corruption(tmp_path):
    bank = torch.zeros(16, 384)
    ids = ["synthetic-a"]
    meta = save_memory(tmp_path, bank, ids, "a" * 64, "b" * 64)
    result, loaded = load_memory(tmp_path, "a" * 64, "b" * 64, ids)
    assert loaded == meta and torch.equal(result, bank)
    with pytest.raises(ValueError):
        save_memory(tmp_path, bank, ids, "a" * 64, "b" * 64)
    with pytest.raises(ValueError):
        load_memory(tmp_path, "c" * 64, "b" * 64, ids)
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "c" * 64, ids)
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ["synthetic-other"])
    (tmp_path / "memory.pt").write_bytes(b"bad")
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ids)
    with pytest.raises(ValueError):
        load_memory(tmp_path / "absent", "a" * 64, "b" * 64, ids)
    torch.save(torch.zeros(15, 384), tmp_path / "memory.pt")
    meta["memory_sha256"] = file_sha256(tmp_path / "memory.pt")
    (tmp_path / "memory.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ids)
    (tmp_path / "memory.json").unlink()
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ids)
    with pytest.raises(ValueError):
        save_memory(tmp_path, bank, ids, "bad", "b" * 64)
    with pytest.raises(ValueError):
        save_memory(tmp_path, bank[:1], ids, "a" * 64, "b" * 64)


def test_read_guard_allowlist_and_test_file_denial(manifest, tmp_path):
    samples = [s for s in manifest.samples if s.partition == Partition.TRAIN]
    guard = DevelopmentReadGuard(tmp_path, samples)
    permitted = tmp_path / samples[0].image_path
    guard("open", (str(permitted), "r", 0))
    guard("open", (str(permitted).encode(), "r", 0))
    guard("open", (tmp_path.parent / "outside.txt", "r", 0))
    guard("open", (123, "r", 0))
    guard("other-event", ())
    assert guard.opened == {samples[0].image_path}
    with pytest.raises(PermissionError):
        guard("open", (str(tmp_path / "test-only.jpg"), "r", 0))
    with pytest.raises(ValueError):
        DevelopmentReadGuard(tmp_path, list(manifest.samples))
    with pytest.raises(ValueError):
        DevelopmentReadGuard(tmp_path, [])


def test_development_adapter_rejects_bad_selected_metadata(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from factoryvision.data.images import ManifestDataset

    def bad_init(self, *args):
        self.samples = [SimpleNamespace(label="anomaly")]

    monkeypatch.setattr(ManifestDataset, "__init__", bad_init)
    with pytest.raises(ValueError):
        DevelopmentDataset(None, tmp_path, Partition.TRAIN)


def test_nonfinite_extracted_features_and_query_bound():
    class Invalid(nn.Module):
        def forward(self, image):
            return torch.full((len(image), 384, 16, 16), float("nan"))

    b = batch()
    with pytest.raises(ValueError):
        fit_memory(Invalid(), [b], b["sample_id"])
    with pytest.raises(ValueError):
        checked_features(torch.zeros(4097, 384))


def test_memory_metadata_patch_count_and_invalid_bank(tmp_path):
    ids = ["synthetic-a"]
    meta = save_memory(tmp_path, torch.zeros(16, 384), ids, "a" * 64, "b" * 64)
    meta["patch_count"] = 15
    (tmp_path / "memory.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ids)
    torch.save(torch.full((16, 384), float("nan")), tmp_path / "memory.pt")
    meta["memory_sha256"] = file_sha256(tmp_path / "memory.pt")
    (tmp_path / "memory.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_memory(tmp_path, "a" * 64, "b" * 64, ids)
