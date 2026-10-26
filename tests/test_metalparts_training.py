"""Fabricated normal sources only; real gradient updates do not require real datasets."""

import json
import time
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from factoryvision.data.metalparts import SYNTHETIC_REVISION, MetalPartsSource
from factoryvision.data.metalparts_development import (
    TRAIN_METADATA,
    NativePatchDataset,
    PatchConfig,
    coordinates,
    load_training_normals,
    split_normals,
    verified_bytes,
)
from factoryvision.domain.contracts import Label, Partition
from factoryvision.models.autoencoder import ConvAutoencoder
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import (
    parameter_sha256,
    score_normals,
    seed_everything,
    train_step,
)
from factoryvision.training.metalparts import (
    BudgetLoader,
    SourceReadGuard,
    artifact_identity,
    fit,
    load_model,
    native_diagnostic,
    save_model,
)


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "MetalParts/metal_parts_data_dir_train"
    (directory / "normal").mkdir(parents=True)
    rows, verified = [], {}
    rng = np.random.default_rng(18)
    for i in range(10):
        path = directory / f"normal/fabricated-{i}.png"
        Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(path)
        rows.append({"file_name": f"normal/fabricated-{i}.png", "label": 0, "mask": None})
        verified[path.relative_to(tmp_path).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    # Deliberately absent anomaly payload and mask: the loader must not open either.
    rows.append(
        {
            "file_name": "anomaly/fabricated.png",
            "label": 1,
            "mask": "MetalParts/metal_parts_data_dir_train/masks/fabricated.png",
        }
    )
    path = tmp_path / TRAIN_METADATA
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    verified[TRAIN_METADATA] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return tmp_path, verified, MetalPartsSource(SYNTHETIC_REVISION)


def changed_metadata(source, rows):
    root, verified, _ = source
    p = root / TRAIN_METADATA
    p.write_text(rows, encoding="utf-8")
    verified[TRAIN_METADATA] = {"bytes": p.stat().st_size, "sha256": file_sha256(p)}


def test_metadata_split_traceability(source):
    root, _, _ = source
    normals, excluded = load_training_normals(*source)
    assert len(normals) == 10 and len(excluded) == 1
    assert all(s.label == Label.NORMAL and s.mask_path is None for s in normals)
    split, groups = split_normals(normals)
    assert split_normals(tuple(reversed(normals))) == (split, groups)
    assert sum(s.partition == Partition.TRAIN for s in split) == 8
    config = PatchConfig(32, 24, 3)
    train = NativePatchDataset(root, split, Partition.TRAIN, config)
    validation = NativePatchDataset(root, split, Partition.VALIDATION, config)
    assert len(train) == 24 and len(validation) == 6
    assert {s.sample_id for s in train.samples}.isdisjoint(s.sample_id for s in validation.samples)
    item = train[0]
    assert item["image"].shape == (3, 32, 32) and item["image"].dtype == torch.float32
    assert 0 <= item["image"].min() <= item["image"].max() <= 1
    assert item["sample_id"].startswith(item["parent_id"])
    assert train.shuffled_indices(42) == train.shuffled_indices(42)
    assert sorted(train.shuffled_indices(42)) == list(range(len(train)))
    assert [validation[i]["sample_id"] for i in range(len(validation))] == sorted(
        validation[i]["sample_id"] for i in range(len(validation))
    )


@pytest.mark.parametrize(
    "text,match",
    [
        ('{"file_name":"normal/a.png","label":true,"mask":null}', "LABEL_INVALID"),
        ('{"file_name":"normal/a.png","label":2,"mask":null}', "LABEL_INVALID"),
        ('{"file_name":"normal/a.png","label":1,"mask":null}', "SOURCE_LAYOUT"),
        ('{"file_name":"../test0/a.png","label":0,"mask":null}', "PATH_UNSAFE"),
        ('{"label":0}', "METADATA_FIELDS"),
        ("[]", "METADATA_FIELDS"),
        ("{", "METADATA_INVALID"),
        ('{"label":0,"label":1}', "METADATA_INVALID"),
        ("", "SAMPLES_INVALID"),
    ],
)
def test_reject_source_metadata(source, text, match):
    changed_metadata(source, text)
    with pytest.raises(ValueError, match=match):
        load_training_normals(*source)


def test_corruption_and_duplicates(source):
    root, verified, _ = source
    text = (root / TRAIN_METADATA).read_text()
    changed_metadata(source, text + "\n" + text.splitlines()[0])
    with pytest.raises(ValueError, match="SAMPLE_DUPLICATE"):
        load_training_normals(*source)
    changed_metadata(source, text)
    reference = next(p for p in verified if p.endswith(".png"))
    expected = verified.pop(reference)
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        load_training_normals(*source)
    verified[reference] = expected
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        verified_bytes(root, reference, {**expected, "bytes": 1})
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        verified_bytes(root, reference, {**expected, "sha256": "0" * 64})


def test_exact_and_confirmed_groups_stay_together(source):
    normals, _ = load_training_normals(*source)
    related = ((normals[0].sample_id, normals[1].sample_id),)
    altered = (normals[0], replace(normals[1], pixel_sha256=normals[0].pixel_sha256), *normals[2:])
    for items, links in ((normals, related), (altered, ())):
        split, groups = split_normals(items, related_pairs=links)
        assert groups[items[0].sample_id] == groups[items[1].sample_id]
        assert split[0].partition == split[1].partition
    with pytest.raises(ValueError, match="unknown"):
        split_normals(normals, related_pairs=(("absent", normals[0].sample_id),))
    with pytest.raises(ValueError, match="separate"):
        split_normals(altered[:2])
    with pytest.raises(ValueError, match="Duplicate"):
        split_normals((normals[0], normals[0]))
    with pytest.raises(ValueError, match="seed"):
        split_normals(normals, True)


def test_test_and_calibration_exclusion_and_guard(source):
    root, verified, _ = source
    normals, _ = load_training_normals(*source)
    for partition in (Partition.TEST, Partition.CALIBRATION, "train"):
        with pytest.raises(ValueError):
            replace(normals[0], partition=partition)
        with pytest.raises(ValueError):
            NativePatchDataset(root, normals, partition)
    with pytest.raises(ValueError):
        replace(normals[0], sample=replace(normals[0].sample, source_domain="test0"))
    for refs in ([], ["../outside"], ["MetalParts/metal_parts_data_dir_test0/normal/a.png"]):
        with pytest.raises(ValueError):
            SourceReadGuard(root, refs)
    guard = SourceReadGuard(root, list(verified))
    guard("open", (str(root / TRAIN_METADATA),))
    assert guard.opened == {TRAIN_METADATA}
    guard("other", ())
    guard("open", (123,))
    guard("open", (str(root.parent / "unrelated"),))
    with pytest.raises(PermissionError):
        guard("open", (str(root / "MetalParts/metal_parts_data_dir_test0/masks/absent.png"),))


@pytest.mark.parametrize(
    "kwargs",
    [{"size": True}, {"size": 48}, {"stride": 0}, {"stride": 300}, {"limit": 0}, {"limit": 129}],
)
def test_patch_config_bounds(kwargs):
    with pytest.raises(ValueError):
        PatchConfig(**kwargs)


def test_patch_edges_and_limits():
    config = PatchConfig()
    assert len(coordinates(2681, 1500, config)) == 12
    full = coordinates(2681, 1500, config, full=True)
    assert len(full) == 112 and full[-1] == (2425, 1244)
    assert coordinates(256, 256, config) == ((0, 0),)
    with pytest.raises(ValueError):
        coordinates(255, 256, config)
    with pytest.raises(ValueError, match="Too many"):
        coordinates(500, 500, PatchConfig(32, 1, 2))


def test_real_update_checkpoint_and_native_geometry(source, tmp_path):
    root, _, origin = source
    split, _ = split_normals(load_training_normals(*source)[0])
    config = PatchConfig(32, 24, 3)
    train = NativePatchDataset(root, split, Partition.TRAIN, config)
    val = NativePatchDataset(root, split, Partition.VALIDATION, config)
    seed_everything(42, 2)
    model = ConvAutoencoder(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    before = parameter_sha256(model)
    batch = next(iter(DataLoader(train, batch_size=4)))
    assert np.isfinite(train_step(model, optimizer, batch))
    assert before != parameter_sha256(model)
    identity = artifact_identity(origin, "a" * 64, "b" * 64, config)
    run = tmp_path / "artifact"
    run.mkdir()
    _, history = fit(
        model,
        optimizer,
        lambda _: DataLoader(train, batch_size=4),
        DataLoader(val, batch_size=4),
        run,
        identity,
        2,
        time.perf_counter() + 30,
        lambda: 100,
    )
    assert len(history) == 2
    loaded, meta = load_model(run, identity)
    assert score_normals(loaded, DataLoader(val, batch_size=4), "validation") == json.loads(
        (run / "selected-validation.json").read_text()
    )
    reconstruction, spatial, score = native_diagnostic(
        torch.nn.Identity(), root, val.samples[0], config
    )
    assert reconstruction.shape == (48, 64, 3) and spatial.shape == (48, 64) and score == 0
    # A nonuniform native pattern reconstructs exactly even through overlapping borders.
    np.testing.assert_array_equal(
        reconstruction,
        np.asarray(Image.open(root / val.samples[0].image_path)).astype(np.float32) / 255,
    )
    with pytest.raises(ValueError, match="compatibility"):
        load_model(run, {**identity, "split_sha256": "c" * 64})
    with pytest.raises(ValueError, match="calibration"):
        load_model(run, {**identity, "threshold": 0.5})
    with pytest.raises(ValueError):
        load_model(run, {**identity, "dataset": "visa"})
    (run / "best.pt").write_bytes(b"fabricated corruption")
    with pytest.raises(ValueError, match="hash"):
        load_model(run, identity)
    assert meta["identity"]["threshold"] is None


def test_budget_and_bad_artifacts(source, tmp_path):
    origin = source[2]
    config = PatchConfig(32, 24, 3)
    identity = artifact_identity(origin, "a" * 64, "b" * 64, config)
    for deadline, memory in ((time.perf_counter() - 1, 0), (time.perf_counter() + 10, 100)):
        with pytest.raises(TimeoutError):
            list(BudgetLoader([{}], deadline, lambda value=memory: value, limit=10))
    with pytest.raises(ValueError):
        fit(None, None, None, None, tmp_path, identity, 13, 0, lambda: 0)
    with pytest.raises(ValueError):
        artifact_identity(None, "a" * 64, "b" * 64, config)
    with pytest.raises(ValueError, match="Geometry"):
        save_model(tmp_path, ConvAutoencoder(64), identity, 1, 0.1)
    with pytest.raises(ValueError, match="Missing"):
        load_model(tmp_path, identity)
    model = ConvAutoencoder(32)
    save_model(tmp_path, model, identity, 1, 0.1)
    for state, message in [
        ({}, "keys"),
        ({**model.state_dict(), "encoder.0.weight": torch.tensor(float("nan"))}, "tensor"),
    ]:
        torch.save(state, tmp_path / "best.pt")
        meta = json.loads((tmp_path / "checkpoint.json").read_text())
        meta["checkpoint_sha256"] = file_sha256(tmp_path / "best.pt")
        (tmp_path / "checkpoint.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=message):
            load_model(tmp_path, identity)


def test_dataset_missing_and_changed_geometry(source):
    root, _, _ = source
    samples = load_training_normals(*source)[0]
    config = PatchConfig(32, 24, 3)
    with pytest.raises(ValueError):
        NativePatchDataset(root, (), Partition.TRAIN, config)
    changed = replace(samples[0], sample=replace(samples[0].sample, width=65))
    ds = NativePatchDataset(root, (changed,), Partition.TRAIN, config)
    with pytest.raises(ValueError, match="geometry"):
        ds[0]
    with pytest.raises(ValueError, match="geometry"):
        native_diagnostic(torch.nn.Identity(), root, changed, config)
    with pytest.raises(ValueError, match="verified"):
        native_diagnostic(torch.nn.Identity(), root, None, config)
