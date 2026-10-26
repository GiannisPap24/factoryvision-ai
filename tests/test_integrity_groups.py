import io
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

from factoryvision.data.integrity_groups import (
    ImageFingerprint,
    fingerprint_bytes,
    group_evidence,
    perceptual_pairs,
    reject_cross_split,
)


def record(name, partition="train", dhash=0, phash=0):
    import hashlib

    digest = hashlib.sha256(name.encode()).hexdigest()
    return ImageFingerprint(name, partition, digest, digest, dhash, phash)


def test_exact_pixels_across_different_encodings_and_no_mutation():
    pixels = np.random.default_rng(42).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    first, second = io.BytesIO(), io.BytesIO()
    image = Image.fromarray(pixels)
    image.save(first, format="PNG")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("synthetic", "different container, same pixels")
    image.save(second, format="PNG", pnginfo=metadata)
    a = fingerprint_bytes("a", "train", first.getvalue())
    b = fingerprint_bytes("b", "test", second.getvalue())
    c = fingerprint_bytes("c", "calibration", first.getvalue())
    assert a.byte_sha256 != b.byte_sha256
    assert a.pixel_sha256 == b.pixel_sha256
    records = (a, b, c)
    result = group_evidence(records)
    assert result["exact_groups"]["byte_sha256"] == [["a", "c"]]
    assert result["exact_groups"]["pixel_sha256"] == [["a", "b", "c"]]
    assert result["affected_confirmed_samples"] == ["a", "b", "c"]
    with pytest.raises(ValueError, match="prohibited"):
        reject_cross_split(result)
    assert result == group_evidence(tuple(reversed(records)))
    assert records == (a, b, c)


def test_transitive_confirmed_groups_and_unresolved_not_merged():
    records = [record("a"), record("b", "validation"), record("c", "test"), record("d")]
    result = group_evidence(
        records, confirmed_pairs=[("a", "b"), ("b", "c")], unresolved_pairs=[("c", "d")]
    )
    assert result["related_groups"][0]["members"] == ["a", "b", "c"]
    assert result["unresolved_cross_partition_pairs"] == [["c", "d"]]
    assert result["affected_confirmed_samples"] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        reject_cross_split(group_evidence(records, unresolved_pairs=[("a", "c")]))
    reject_cross_split(group_evidence([record("a"), record("b")], confirmed_pairs=[("a", "b")]))
    reject_cross_split(group_evidence([]))


def test_index_matches_exhaustive_hamming_with_boundary_and_order():
    rng = np.random.default_rng(17)
    records = [
        record(
            str(i),
            ("train", "test")[i % 2],
            int(rng.integers(0, 2**63)),
            int(rng.integers(0, 2**63)),
        )
        for i in range(80)
    ]
    records += [
        record("edge-a", "train", 0, 2**63 - 1),
        record("edge-b", "test", (1 << 63) | 7, 0),
        record("edge-c", "test", 31, 2**63 - 32),
    ]
    expected = {
        (a.sample_id, b.sample_id)
        for a, b in combinations(sorted(records, key=lambda r: r.sample_id), 2)
        if a.partition != b.partition
        and ((a.dhash ^ b.dhash).bit_count() <= 4 or (a.phash ^ b.phash).bit_count() <= 4)
    }
    report = perceptual_pairs(records)
    assert {(p["a"], p["b"]) for p in report["pairs"]} == expected
    assert report == perceptual_pairs(list(reversed(records)))


@pytest.mark.parametrize(
    "changes",
    [
        {"sample_id": " "},
        {"partition": "unknown"},
        {"byte_sha256": "g" * 64},
        {"pixel_sha256": "x"},
        {"dhash": -1},
        {"phash": 2**63},
        {"dhash": True},
    ],
)
def test_invalid_fingerprints(changes):
    with pytest.raises(ValueError):
        replace(record("a"), **changes)


def test_resource_caps_and_invalid_reviews(monkeypatch):
    records = [record("a"), record("b", "test"), record("c", "test")]
    with pytest.raises(ValueError, match="Comparison cap"):
        perceptual_pairs(records, max_pairs=1)
    with pytest.raises(ValueError, match="Positive"):
        perceptual_pairs(records, max_pairs=0)
    with pytest.raises(ValueError, match="Record limit"):
        group_evidence([record("a")] * 2001)
    with pytest.raises(ValueError, match="Duplicate"):
        perceptual_pairs([record("a"), record("a")])
    for pairs in ([("a", "z")], [("a", "a")]):
        with pytest.raises(ValueError, match="identity"):
            group_evidence(records, confirmed_pairs=pairs)
    with pytest.raises(ValueError, match="Conflicting"):
        group_evidence(records, confirmed_pairs=[("a", "b")], unresolved_pairs=[("b", "a")])
    with pytest.raises(ValueError, match="byte limit"):
        fingerprint_bytes("a", "train", b"x" * (16 * 1024 * 1024 + 1))
    original = Image.new("RGB", (4001, 4000))
    monkeypatch.setattr(Image, "open", lambda _: original)
    with pytest.raises(ValueError, match="pixel limit"):
        fingerprint_bytes("a", "train", b"")
