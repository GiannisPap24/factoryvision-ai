"""Tiny synthetic CPU model, isolation, scoring and artifact tests; no real data."""

import json
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from factoryvision.data.leakage import fingerprints, screen_pairs
from factoryvision.evaluation.reconstruction import calibrate_normal, reconstruction_errors
from factoryvision.models.autoencoder import ConvAutoencoder
from factoryvision.training.checkpoint import (
    atomic_write,
    checked_directory,
    create_run,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from factoryvision.training.config import TrainingConfig
from factoryvision.training.engine import (
    EarlyStopping,
    parameter_sha256,
    require_normal_batch,
    score_normals,
    seed_everything,
    train_epoch,
    train_step,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    seed_everything(42, 1)


def batch(partition="train", size=2):
    return {
        "image": torch.rand(size, 3, 32, 32),
        "label": torch.zeros(size, dtype=torch.int64),
        "partition": [partition] * size,
        "sample_id": [f"synthetic-{i}" for i in range(size)],
    }


@pytest.mark.parametrize("size,resolution", [(1, 32), (2, 64), (3, 128)])
def test_model_forward_backward_shapes(size, resolution):
    model = ConvAutoencoder(resolution)
    x = torch.rand(size, 3, resolution, resolution)
    out = model(x)
    assert out.shape == x.shape and out.min() >= 0 and out.max() <= 1
    assert model.encoder(x).shape == (size, 16, resolution // 16, resolution // 16)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("value", [True, 0, 31, 33, 512, 128.0])
def test_model_invalid_resolution(value):
    with pytest.raises(ValueError):
        ConvAutoencoder(value)


@pytest.mark.parametrize(
    "x",
    [
        torch.zeros(1, 1, 32, 32),
        torch.zeros(0, 3, 32, 32),
        torch.zeros(1, 3, 16, 16),
        torch.zeros(1, 3, 32, 32, dtype=torch.float64),
        torch.full((1, 3, 32, 32), float("nan")),
        torch.full((1, 3, 32, 32), -1.0),
        torch.full((1, 3, 32, 32), 2.0),
    ],
)
def test_model_invalid_inputs(x):
    with pytest.raises(ValueError):
        ConvAutoencoder(32)(x)


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", True),
        ("batch_size", 0),
        ("max_epochs", 13),
        ("resolution", 48.0),
        ("resolution", 33),
        ("seed", -1),
        ("patience", 0),
        ("threads", 9),
        ("max_seconds", 0),
        ("learning_rate", float("nan")),
        ("learning_rate", 0),
        ("min_delta", -0.01),
        ("min_delta", -1e-16),
        ("calibration_quantile", 2),
        ("learning_rate", True),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        replace(TrainingConfig(), **{field: value})


def test_config_and_repeatable_initialization():
    assert TrainingConfig().to_dict()["resolution"] == 128
    assert TrainingConfig(min_delta=0).min_delta == 0
    first = parameter_sha256(ConvAutoencoder(32))
    seed_everything(42, 1)
    assert parameter_sha256(ConvAutoencoder(32)) == first


def test_actual_updates_and_modes():
    model = ConvAutoencoder(32)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    initial = parameter_sha256(model)
    b = batch()
    loss = train_step(model, opt, b)
    assert np.isfinite(loss) and parameter_sha256(model) != initial and model.training
    b["partition"] = ["validation"] * 2
    result = score_normals(model, [b], "validation")
    assert not model.training and len(result["scores"]) == 2
    assert score_normals(model, [b], "validation") == result
    assert all(not p.requires_grad or p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("partition", ["test", "validation", "calibration"])
def test_training_partition_rejection(partition):
    model = ConvAutoencoder(32)
    with pytest.raises(ValueError):
        train_step(model, torch.optim.Adam(model.parameters()), batch(partition))


def test_normal_batch_rejects_anomaly_and_bad_labels():
    b = batch()
    b["label"][0] = 1
    with pytest.raises(ValueError):
        require_normal_batch(b, "train")
    b["label"] = torch.zeros(3)
    with pytest.raises(ValueError):
        require_normal_batch(b, "train")


def test_train_epoch_weighting_and_deadline():
    model = ConvAutoencoder(32)
    opt = torch.optim.Adam(model.parameters())
    a = batch(size=2)
    b = batch(size=1)
    b["sample_id"] = ["synthetic-2"]
    result = train_epoch(model, opt, [a, b], deadline=time.perf_counter() + 30)
    assert result["samples"] == 3 and result["steps"] == 2 and len(result["sample_ids"]) == 3
    with pytest.raises(TimeoutError):
        train_epoch(model, opt, [a], deadline=0)
    with pytest.raises(ValueError):
        train_epoch(model, opt, [], deadline=0)
    with pytest.raises(ValueError):
        train_epoch(model, opt, [a, a], deadline=time.perf_counter() + 30)


def test_scoring_rejects_empty_duplicate_or_unsorted_ids():
    model = ConvAutoencoder(32)
    with pytest.raises(ValueError):
        score_normals(model, [], "validation")
    b = batch("validation")
    b["sample_id"].reverse()
    with pytest.raises(ValueError):
        score_normals(model, [b], "validation")
    b["sample_id"] = ["same", "same"]
    with pytest.raises(ValueError):
        score_normals(model, [b], "validation")


def test_nonfinite_training_loss_gradients_and_optimizer():
    model = ConvAutoencoder(32)
    opt = torch.optim.Adam(model.parameters())
    handle = model.register_forward_hook(lambda m, i, o: o * float("nan"))
    with pytest.raises(ValueError, match="Non-finite reconstruction"):
        train_step(model, opt, batch())
    handle.remove()
    handle = model.register_forward_hook(lambda m, i, o: o + 1e30)
    with pytest.raises(ValueError, match="Non-finite training loss"):
        train_step(model, opt, batch())
    handle.remove()
    handle = next(model.parameters()).register_hook(lambda g: g * float("inf"))
    with pytest.raises(ValueError, match="gradients"):
        train_step(model, opt, batch())
    handle.remove()

    class BadOptimizer:
        def zero_grad(self, **kw):
            model.zero_grad(**kw)

        def step(self):
            with torch.no_grad():
                next(model.parameters()).fill_(float("nan"))

    with pytest.raises(ValueError, match="optimizer"):
        train_step(model, BadOptimizer(), batch())


def test_reconstruction_exact_semantics():
    x = torch.zeros(2, 3, 4, 5)
    y = x.clone()
    y[:, 0] = 1
    scores, maps = reconstruction_errors(x, y)
    assert maps.shape == (2, 4, 5) and scores.shape == (2,)
    torch.testing.assert_close(maps, torch.full((2, 4, 5), 1 / 3))
    torch.testing.assert_close(scores, torch.full((2,), 1 / 3))
    with pytest.raises(ValueError):
        reconstruction_errors(x, y[:1])
    with pytest.raises(ValueError):
        reconstruction_errors(x[:, :1], y[:, :1])
    with pytest.raises(ValueError):
        reconstruction_errors(x[:0], y[:0])
    with pytest.raises(ValueError):
        reconstruction_errors(x, y * float("inf"))


def test_calibration_predeclared_quantile():
    r = calibrate_normal(torch.arange(20, dtype=torch.float64))
    assert r["threshold"] == pytest.approx(18.05) and r["exceedances"] == 1
    assert calibrate_normal(torch.ones(68))["exceedances"] == 0
    assert calibrate_normal(torch.arange(20.0).flip(0)) == r


@pytest.mark.parametrize(
    "scores,q",
    [
        (torch.ones(1), 0.95),
        (torch.ones(2, 2), 0.95),
        (torch.tensor([0.0, float("nan")]), 0.95),
        (torch.tensor([-1.0, 0.0]), 0.95),
        (torch.ones(2), 0),
        (torch.ones(2), float("nan")),
        (torch.ones(2), True),
    ],
)
def test_invalid_calibration(scores, q):
    with pytest.raises(ValueError):
        calibrate_normal(scores, q)


def test_stopping_selects_strict_best_and_counts_patience():
    s = EarlyStopping(2, 0.01)
    assert s.update(0.5, 1) == (True, False)
    assert s.update(0.495, 2) == (True, False)
    assert s.update(0.496, 3) == (False, True)
    assert s.best_epoch == 2
    assert s.update(0.4, 4) == (True, False)
    with pytest.raises(ValueError):
        s.update(float("nan"), 5)


def test_safe_checkpoint_roundtrip(tmp_path):
    d = create_run(tmp_path, "toy")
    model = ConvAutoencoder(32)
    meta = save_checkpoint(d, model, "a" * 64, 1, 0.2)
    assert meta["checkpoint_sha256"] == file_sha256(d / "best.pt")
    loaded, _ = load_checkpoint(d, "a" * 64, 32)
    x = batch()["image"]
    torch.testing.assert_close(model(x), loaded(x), rtol=0, atol=0)
    assert not loaded.training
    with pytest.raises(ValueError):
        load_checkpoint(d, "b" * 64, 32)
    with pytest.raises(ValueError):
        load_checkpoint(d, "a" * 64, 64)
    with pytest.raises(ValueError):
        save_checkpoint(d, model, "bad", 1, 0.2)
    (d / "best.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(d, "a" * 64, 32)


@pytest.mark.parametrize("kind", ["keys", "shape", "dtype", "nonfinite", "object"])
def test_checkpoint_rejects_invalid_state(tmp_path, kind):
    d = create_run(tmp_path, "toy")
    model = ConvAutoencoder(32)
    save_checkpoint(d, model, "a" * 64, 1, 0.2)
    state = model.state_dict()
    key = next(iter(state))
    if kind == "keys":
        state.pop(key)
    elif kind == "shape":
        state[key] = torch.zeros(1)
    elif kind == "dtype":
        state[key] = state[key].double()
    elif kind == "nonfinite":
        state[key].fill_(float("nan"))
    else:
        state = []
    torch.save(state, d / "best.pt")
    meta = json.loads((d / "checkpoint.json").read_text())
    meta["checkpoint_sha256"] = file_sha256(d / "best.pt")
    write_json(d, "checkpoint.json", meta)
    with pytest.raises(ValueError):
        load_checkpoint(d, "a" * 64, 32)


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "has space", "", "UPPER"])
def test_artifact_path_rejection(tmp_path, name):
    with pytest.raises(ValueError):
        create_run(tmp_path, name)


def test_atomic_failure_preserves_previous_and_cleans_temp(tmp_path):
    write_json(tmp_path, "result.json", {"old": True})

    def broken(stream):
        stream.write(b"partial")
        raise OSError("synthetic failure")

    with pytest.raises(OSError):
        atomic_write(tmp_path, "result.json", broken)
    assert json.loads((tmp_path / "result.json").read_text()) == {"old": True}
    assert list(tmp_path.iterdir()) == [tmp_path / "result.json"]
    with pytest.raises(ValueError):
        atomic_write(tmp_path, "../evil.json", broken)
    with pytest.raises(ValueError):
        checked_directory(tmp_path / "missing")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path, "a" * 64, 32)
    d = create_run(tmp_path, "once")
    with pytest.raises(FileExistsError):
        create_run(tmp_path, "once")
    assert d.is_dir()


def test_fingerprint_sensitivity_and_candidates():
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    dh, ph, rgb = fingerprints(image)
    again = fingerprints(image.copy())
    assert (dh, ph) == again[:2] and np.array_equal(rgb, again[2])
    assert fingerprints(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))[:2] != (dh, ph)
    records = [
        {"id": "a", "partition": "train", "dhash": dh, "phash": ph, "rgb": rgb},
        {"id": "b", "partition": "test", "dhash": dh, "phash": ph, "rgb": rgb},
        {
            "id": "c",
            "partition": "test",
            "dhash": dh ^ ((1 << 64) - 1),
            "phash": ph ^ ((1 << 63) - 1),
            "rgb": np.ones_like(rgb),
        },
    ]
    result = screen_pairs(records)
    assert result["cross_partition_comparisons"] == 2
    assert len(result["candidates"]) == 1 and result["candidates"][0]["rgb_rms"] == 0


def test_artifacts_reject_actual_junctions(tmp_path):
    import _winapi

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    _winapi.CreateJunction(str(real), str(linked))
    with pytest.raises(ValueError, match="Linked"):
        create_run(linked, "escape")
    _winapi.CreateJunction(str(real), str(tmp_path / "result.json"))
    with pytest.raises(ValueError, match="Linked"):
        write_json(tmp_path, "result.json", {"bad": True})
    assert not list(real.iterdir())
