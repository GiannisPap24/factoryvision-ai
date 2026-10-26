"""Entirely fabricated bytes/metadata; no real VisA or model access."""

import io
import json
import tarfile
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from factoryvision.data.category import load_category
from factoryvision.data.pretraining_integrity import (
    AuditDefect,
    AuditIncomplete,
    Limits,
    describe_image,
    extract_category_images,
    file_hash,
    gate_decision,
    read_protocol,
    scan_all_pairs,
)
from factoryvision.domain.errors import ContractError


def png(color):
    stream = io.BytesIO()
    Image.new("RGB", (16, 12), color).save(stream, format="PNG")
    return stream.getvalue()


def prepared(colors, partitions):
    records, features = [], {}
    for i, (color, partition) in enumerate(zip(colors, partitions, strict=True)):
        record, feature = describe_image(f"s{i}", partition, png(color))
        records.append(record)
        features[record.sample_id] = feature
    return records, features


def test_confirmed_exact_and_within_partition_groups_preserve_inputs():
    records, features = prepared(["red"] * 3, ["train", "train", "test"])
    before = tuple(records)
    screen = scan_all_pairs(records, features)
    result = gate_decision(records, screen)
    assert screen["comparisons"] == 3 and screen["required"] == []
    assert result["verdict"] == "BLOCKED"
    assert result["groups"]["affected_confirmed_samples"] == ["s0", "s1", "s2"]
    assert tuple(records) == before and result["sample_exclusions"] == 0
    for partition in ("train", "test"):
        same = [replace(r, partition=partition) for r in records]
        assert gate_decision(same, scan_all_pairs(same, features))["verdict"] == "PASS"


def test_hash_false_positive_is_not_confirmation_and_requires_evidence():
    records, features = prepared(["black", "white"], ["train", "test"])
    screen = scan_all_pairs(records, features)
    assert screen["candidates"][0]["dhash_distance"] == 0
    assert not screen["candidates"][0]["strong"]
    result = gate_decision(records, screen)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["groups"]["confirmed_pairs"] == []
    decision = {
        "a": "s0",
        "b": "s1",
        "decision": "rejected",
        "basis": "independent_provenance",
        "evidence": "Synthetic different objects",
    }
    assert gate_decision(records, screen, decisions=[decision])["verdict"] == "PASS"
    confirmed = {**decision, "decision": "confirmed", "evidence": "Synthetic capture identity"}
    assert gate_decision(records, screen, decisions=[confirmed])["verdict"] == "BLOCKED"
    with pytest.raises(ValueError, match="Independent"):
        gate_decision(records, screen, decisions=[{**confirmed, "basis": "perceptual_hash"}])


def test_similarity_refinement_determinism_and_ambiguous_strong_pair():
    rng = np.random.default_rng(42)
    a = rng.integers(0, 200, (32, 32, 3), dtype=np.uint8)
    records, features = [], {}
    for i, pixels in enumerate((a, a + 1, rng.integers(0, 256, a.shape, dtype=np.uint8))):
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="PNG")
        record, feature = describe_image(f"r{i}", ("train", "test", "test")[i], stream.getvalue())
        records.append(record)
        features[record.sample_id] = feature
    first = scan_all_pairs(records, features)
    assert first == scan_all_pairs(records[::-1], features)
    assert first["candidates"][0]["strong"]
    assert first["candidates"][0]["crop_correlation"] > 0.995
    assert gate_decision(records, first)["verdict"] == "INCONCLUSIVE"


def test_pass_with_complete_no_candidates_and_fail_closed_states():
    records, features = prepared(["black", "white"], ["train", "test"])
    records[1] = replace(records[1], dhash=2**64 - 1, phash=2**63 - 1)
    empty = scan_all_pairs(records, features)
    assert not empty["candidates"] and gate_decision(records, empty)["verdict"] == "PASS"
    assert gate_decision(records, {**empty, "complete": False})["verdict"] == "INCONCLUSIVE"
    assert gate_decision(records, empty, source_verified=False)["verdict"] == "INCONCLUSIVE"
    assert gate_decision(records, empty, defects=["contradictory metadata"])["verdict"] == "BLOCKED"


def test_protocol_identity_and_unsupported_method(tmp_path):
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps({"method": "pcb2-integrity-v1"}))
    sha = file_hash(p)
    assert read_protocol(p, sha)["method"] == "pcb2-integrity-v1"
    p.write_text(json.dumps({"method": "changed"}))
    with pytest.raises(AuditIncomplete, match="identity"):
        read_protocol(p, sha)
    with pytest.raises(AuditIncomplete, match="Unsupported"):
        read_protocol(p, file_hash(p))


def test_caps_incomplete_scan_bad_reviews_and_corrupt_images():
    records, features = prepared(["red", "blue", "green"], ["train", "test", "test"])
    limited = scan_all_pairs(records, features, limits=replace(Limits(), candidates=1))
    assert not limited["complete"] and gate_decision(records, limited)["verdict"] == "INCONCLUSIVE"
    with pytest.raises(AuditIncomplete, match="Pair"):
        scan_all_pairs(records, features, limits=replace(Limits(), pairs=1))
    with pytest.raises(AuditDefect, match="incomplete"):
        scan_all_pairs(records, {})
    with pytest.raises(AuditDefect, match="Duplicate"):
        scan_all_pairs(records + [records[0]], features)
    screen = scan_all_pairs(records, features)
    pair = {"a": "s0", "b": "s1", "decision": "unresolved"}
    assert gate_decision(records, screen, decisions=[pair])["verdict"] == "INCONCLUSIVE"
    for decisions in ([{**pair, "b": "missing"}], [pair, pair], [{**pair, "decision": "invented"}]):
        with pytest.raises(ValueError):
            gate_decision(records, screen, decisions=decisions)
    with pytest.raises(AuditDefect, match="Corrupt"):
        describe_image("x", "train", b"broken")
    for limits in (replace(Limits(), encoded_bytes=1), replace(Limits(), image_pixels=1)):
        with pytest.raises(AuditIncomplete):
            describe_image("x", "train", png("red"), limits)
    bad = io.BytesIO()
    Image.new("RGB", (2, 2)).save(bad, format="BMP")
    with pytest.raises(AuditDefect, match="format"):
        describe_image("x", "train", bad.getvalue())
    bad = io.BytesIO()
    image = Image.new("RGB", (2, 2))
    exif = image.getexif()
    exif[274] = 3
    image.save(bad, format="JPEG", exif=exif)
    with pytest.raises(AuditDefect, match="orientation"):
        describe_image("x", "train", bad.getvalue())


def fixture_archive(tmp_path, *, omit=False, unsafe=False):
    archive = tmp_path / "images.tar"
    split = tmp_path / "split.csv"
    split.write_text(
        "object,split,label,image,mask\npcb2,train,normal,pcb2/Data/Images/Normal/0000.JPG,\npcb2,test,normal,pcb2/Data/Images/Normal/0001.JPG,\n"
    )
    with tarfile.open(archive, "w") as stream:
        for i in range(1 if omit else 2):
            payload = png("red")
            member = tarfile.TarInfo(f"pcb2/Data/Images/Normal/{i:04}.JPG")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
        if unsafe:
            member = tarfile.TarInfo("../escape")
            member.size = 0
            stream.addfile(member, io.BytesIO())
    return archive, load_category(split, "pcb2")


def test_source_preserving_allowlisted_extraction(tmp_path):
    archive, manifest = fixture_archive(tmp_path)
    before = file_hash(archive)
    events = []
    result = extract_category_images(
        archive,
        tmp_path / "new",
        manifest,
        before,
        lambda phase, path: events.append((phase, path)),
        lambda: None,
    )
    assert len(result["images"]) == 2 and result["mask_payloads_accessed"] == 0
    assert file_hash(archive) == before and len(events) == 2
    for row in result["images"]:
        assert file_hash(tmp_path / "new" / row["path"]) == row["source_sha256"]
    with pytest.raises(FileExistsError):
        extract_category_images(
            archive, tmp_path / "new", manifest, before, lambda *_: None, lambda: None
        )
    with pytest.raises(AuditIncomplete, match="identity"):
        extract_category_images(
            archive, tmp_path / "other", manifest, "0" * 64, lambda *_: None, lambda: None
        )


@pytest.mark.parametrize("kind", ["missing", "unsafe"])
def test_missing_and_unsafe_archive_fail_before_extraction(tmp_path, kind):
    archive, manifest = fixture_archive(tmp_path, omit=kind == "missing", unsafe=kind == "unsafe")
    with pytest.raises((AuditDefect, ContractError)):
        extract_category_images(
            archive, tmp_path / "new", manifest, file_hash(archive), lambda *_: None, lambda: None
        )
    assert not (tmp_path / "new").exists()


def test_budget_interruption_and_unknown_review_cannot_pass():
    records, features = prepared(["red", "blue"], ["train", "test"])

    def exhausted():
        raise AuditIncomplete("Synthetic resource exhaustion")

    partial = scan_all_pairs(records, features, exhausted)
    assert partial["comparisons"] == 0 and not partial["complete"]
    assert gate_decision(records, partial)["verdict"] == "INCONCLUSIVE"
    empty = {"complete": True, "comparisons": 0, "required": []}
    assert gate_decision([], empty)["verdict"] == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="identities"):
        gate_decision(records, {**partial, "required": [{"a": "s0", "b": "unknown"}]})


def test_extraction_resource_cap_and_mixed_categories(tmp_path, monkeypatch):
    from factoryvision.data import pretraining_integrity as module

    archive, manifest = fixture_archive(tmp_path)
    first, second = manifest.samples
    mixed = replace(
        manifest,
        samples=(
            first,
            replace(
                second, category="pcb3", image_path=second.image_path.replace("pcb2/", "pcb3/")
            ),
        ),
    )
    with pytest.raises(AuditDefect, match="one category"):
        extract_category_images(
            archive, tmp_path / "new", mixed, file_hash(archive), lambda *_: None, lambda: None
        )
    monkeypatch.setattr(module, "Limits", lambda: replace(Limits(), selected_bytes=1))
    with pytest.raises(AuditIncomplete, match="resource"):
        extract_category_images(
            archive, tmp_path / "new", manifest, file_hash(archive), lambda *_: None, lambda: None
        )
    assert not (tmp_path / "new").exists()
