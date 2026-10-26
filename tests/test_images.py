"""Offline decoding and real CPU DataLoader tests using locally fabricated pixels."""

from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from factoryvision.data.images import (
    ImageSettings,
    ManifestDataset,
    decode_image,
    decode_pair,
    make_loader,
)
from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import Partition
from factoryvision.domain.errors import ContractError


@pytest.fixture
def image_root(tmp_path, manifest):
    for index, sample in enumerate(manifest.samples):
        path = tmp_path / sample.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((12, 16, 3), dtype=np.uint8)
        pixels[:, :8] = [index * 10, 40, 200]
        pixels[:, 8:] = [200, 90, index * 10]
        Image.fromarray(pixels).save(path)
        if sample.mask_path:
            path = tmp_path / sample.mask_path
            path.parent.mkdir(parents=True, exist_ok=True)
            mask = np.zeros((12, 16), dtype=np.uint8)
            mask[2:8, 3:10] = 3
            mask[8:10, 10:12] = 9
            Image.fromarray(mask).save(path)
    return tmp_path


def test_rgb_masks_and_stored_geometry(manifest, image_root):
    image, mask = decode_pair(image_root, manifest.samples[-1])
    assert image.mode == "RGB" and image.size == (16, 12)
    assert set(np.unique(mask)) == {0, 255}
    _, normal = decode_pair(image_root, manifest.samples[0])
    assert not np.asarray(normal).any()


def test_tensor_transform_and_mask_nearest(manifest, image_root):
    dataset = ManifestDataset(manifest, image_root, Partition.TEST, ImageSettings(6, 8))
    anomaly_index = next(i for i, s in enumerate(dataset.samples) if s.mask_path is not None)
    sample = dataset[anomaly_index]
    assert sample["image"].shape == (3, 6, 8)
    assert sample["mask"].shape == (1, 6, 8)
    assert sample["image"].dtype == sample["mask"].dtype == torch.float32
    assert sample["image"].device.type == "cpu"
    assert 0 <= sample["image"].min() <= sample["image"].max() <= 1
    assert set(sample["mask"].unique().tolist()) <= {0.0, 1.0}
    _, mask = decode_pair(image_root, dataset.samples[anomaly_index])
    expected = np.asarray(mask.resize((8, 6), Image.Resampling.NEAREST)) / 255
    np.testing.assert_array_equal(sample["mask"][0].numpy(), expected)
    assert sample["label"] == 1 and sample["mask_available"] is True
    assert sample["original_size"].tolist() == [12, 16]
    normal_index = 1 - anomaly_index
    assert dataset[normal_index]["label"] == 0
    assert dataset[normal_index]["mask_available"] is False


def test_repeated_dataloader_order_and_isolation(manifest, image_root):
    derived = derive_splits(manifest)
    outputs = {}
    for partition in Partition:
        dataset = ManifestDataset(derived, image_root, partition, ImageSettings(8, 8))
        reordered = replace(derived, samples=tuple(reversed(derived.samples)))
        assert dataset.samples == ManifestDataset(reordered, image_root, partition).samples
        first = list(make_loader(dataset, batch_size=2))
        second = list(make_loader(dataset, batch_size=2))
        ids = []
        for a, b in zip(first, second, strict=True):
            assert a["sample_id"] == b["sample_id"]
            torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
            torch.testing.assert_close(a["mask"], b["mask"], rtol=0, atol=0)
            assert a["label"].dtype == torch.int64
            assert a["mask_available"].dtype == torch.bool
            assert a["image"].shape[1:] == (3, 8, 8)
            assert a["mask"].shape[1:] == (1, 8, 8)
            ids.extend(a["sample_id"])
            if partition != Partition.TEST:
                assert a["label"].sum().item() == 0
        outputs[partition] = set(ids)
    assert sum(map(len, outputs.values())) == len(set.union(*outputs.values())) == 12


@pytest.mark.parametrize("size", [(0, 1), (1, 2049), (True, 4), (4.5, 3)])
def test_resolution_bounds(size):
    with pytest.raises(ContractError, match="IMAGE_CONFIG"):
        ImageSettings(*size)


def test_dataset_invalid_selection_and_metadata(manifest, image_root):
    with pytest.raises(ContractError, match="ENUM_INVALID"):
        ManifestDataset(manifest, image_root, "train")
    with pytest.raises(ContractError, match="PARTITION_EMPTY"):
        ManifestDataset(manifest, image_root, Partition.CALIBRATION)
    with pytest.raises(ContractError, match="IMAGE_CONFIG"):
        ManifestDataset(manifest, image_root, Partition.TRAIN, "bad")
    invalid = replace(manifest, samples=manifest.samples + (manifest.samples[0],))
    with pytest.raises(ContractError, match="DUPLICATE_ID"):
        ManifestDataset(invalid, image_root, Partition.TRAIN)


def test_missing_and_corrupt_images(manifest, image_root):
    sample = manifest.samples[0]
    path = image_root / sample.image_path
    path.write_bytes(b"not an image")
    with pytest.raises(ContractError, match="IMAGE_DECODE"):
        decode_pair(image_root, sample)
    path.unlink()
    with pytest.raises(ContractError, match="FILE_MISSING"):
        decode_pair(image_root, sample)


def test_image_limits_formats_and_orientation(tmp_path, monkeypatch):
    from factoryvision.data import images

    path = tmp_path / "sample.png"
    Image.new("L", (3, 2), 127).save(path)
    assert decode_image(tmp_path, path.name).mode == "RGB"
    monkeypatch.setattr(images, "MAX_PIXELS", 5)
    with pytest.raises(ContractError, match="IMAGE_TOO_LARGE"):
        decode_image(tmp_path, path.name)
    monkeypatch.setattr(images, "MAX_PIXELS", 20_000_000)
    monkeypatch.setattr(images, "MAX_FILE_BYTES", 1)
    with pytest.raises(ContractError, match="IMAGE_TOO_LARGE"):
        decode_image(tmp_path, path.name)
    monkeypatch.setattr(images, "MAX_FILE_BYTES", 32 * 1024**2)
    Image.new("RGB", (3, 2)).save(path, format="GIF")
    with pytest.raises(ContractError, match="IMAGE_FORMAT"):
        decode_image(tmp_path, path.name)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (3, 2)).save(path, format="JPEG", exif=exif)
    with pytest.raises(ContractError, match="IMAGE_ORIENTATION"):
        decode_image(tmp_path, path.name)


@pytest.mark.parametrize(
    "mode,size,value,code",
    [
        ("RGB", (16, 12), 1, "MASK_MODE"),
        ("L", (8, 8), 1, "MASK_DIMENSIONS"),
        ("L", (16, 12), 0, "MASK_EMPTY"),
    ],
)
def test_invalid_mask_geometry_and_content(manifest, image_root, mode, size, value, code):
    sample = manifest.samples[-1]
    Image.new(mode, size, value).save(image_root / sample.mask_path)
    with pytest.raises(ContractError, match=code):
        decode_pair(image_root, sample)


def test_missing_mask(manifest, image_root):
    sample = manifest.samples[-1]
    (image_root / sample.mask_path).unlink()
    with pytest.raises(ContractError, match="FILE_MISSING"):
        decode_pair(image_root, sample)


@pytest.mark.parametrize("batch_size", [0, 65, True, 1.5])
def test_invalid_loader_batch(manifest, image_root, batch_size):
    dataset = ManifestDataset(manifest, image_root, Partition.TRAIN)
    with pytest.raises(ContractError, match="LOADER_CONFIG"):
        make_loader(dataset, batch_size)


def test_loader_seed_and_memory_bounds(manifest, image_root):
    dataset = ManifestDataset(manifest, image_root, Partition.TRAIN, ImageSettings(2048, 2048))
    with pytest.raises(ContractError, match="LOADER_CONFIG"):
        make_loader(dataset, 64)
    for seed in (-1, True, 2**32):
        with pytest.raises(ContractError, match="SEED_INVALID"):
            make_loader(dataset, 1, seed)
