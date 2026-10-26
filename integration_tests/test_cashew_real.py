"""Explicit real-data tests. Excluded from default pytest testpaths; no downloads."""

from collections import Counter
from pathlib import Path

import pytest
import torch

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.splits import derive_splits
from factoryvision.data.visa import audit_cashew, load_visa_cashew
from factoryvision.domain.contracts import OfficialSplit, Partition

ROOT = Path(__file__).resolve().parents[1]


def test_real_inventory_decodes_and_matches_frozen_parent():
    source = load_visa_cashew(ROOT / "data/metadata/1cls.csv")
    parent, report = audit_cashew(
        source, ROOT / "data/visa", ROOT / "data/visa/cashew/image_anno.csv"
    )
    frozen = load_manifest(ROOT / "data/manifests/cashew-official.json")
    assert parent == frozen
    assert report["samples"] == 600 and report["masks"] == 100
    assert report["official_counts"] == {
        "train/normal": 450,
        "test/normal": 50,
        "test/anomaly": 100,
    }
    assert not report["byte_duplicate_groups"] and not report["pixel_duplicate_groups"]


def test_frozen_split_preserves_official_test_and_provenance():
    parent = load_manifest(ROOT / "data/manifests/cashew-official.json")
    derived = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    assert derive_splits(parent, 42) == derived
    assert derived.split_provenance.parent_manifest_sha256 == manifest_sha256(parent)
    assert Counter(s.partition.value for s in derived.samples) == {
        "train": 315,
        "validation": 67,
        "calibration": 68,
        "test": 150,
    }
    assert [s for s in parent.samples if s.official_split == OfficialSplit.TEST] == [
        s for s in derived.samples if s.official_split == OfficialSplit.TEST
    ]


@pytest.mark.parametrize("partition", list(Partition))
def test_real_cpu_dataloader_batch(partition):
    manifest = load_manifest(ROOT / "data/manifests/cashew-seed42.json")
    dataset = ManifestDataset(manifest, ROOT / "data/visa", partition, ImageSettings(128, 128))
    a = next(iter(make_loader(dataset, batch_size=2)))
    b = next(iter(make_loader(dataset, batch_size=2)))
    assert a["image"].shape == (2, 3, 128, 128) and a["mask"].shape == (2, 1, 128, 128)
    assert a["image"].dtype == a["mask"].dtype == torch.float32
    assert a["image"].device.type == "cpu"
    assert a["label"].dtype == torch.int64
    assert a["sample_id"] == b["sample_id"]
    torch.testing.assert_close(a["image"], b["image"], rtol=0, atol=0)
    assert set(a["mask"].unique().tolist()) <= {0.0, 1.0}
