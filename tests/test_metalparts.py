"""Fabricated offline MetalParts-shaped fixtures, never real dataset observations."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from PIL import Image

from factoryvision.data.metalparts import (
    DOMAINS,
    REVISION,
    SYNTHETIC_REVISION,
    MetalPartsManifest,
    MetalPartsSource,
    load_metalparts,
)
from factoryvision.data.metalparts_audit import audit_metalparts, verify_source_files, write_json
from factoryvision.domain.contracts import Label, OfficialSplit
from factoryvision.domain.errors import ContractError


@pytest.fixture
def fabricated(tmp_path):
    rng = np.random.default_rng(19)
    for domain in DOMAINS:
        directory = tmp_path / "MetalParts" / f"metal_parts_data_dir_{domain}"
        rows = []
        for label in ("normal", "anomaly"):
            name = f"fabricated_{domain}_{label}"
            path = directory / label / f"{name}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (40, 48, 3), dtype=np.uint8)).save(path)
            mask = None
            if label == "anomaly":
                target = directory / "masks" / f"{name}.png"
                target.parent.mkdir(exist_ok=True)
                pixels = np.zeros((40, 48), dtype=np.uint8)
                pixels[10:12, 15:17] = 255
                Image.fromarray(pixels).save(target)
                mask = target.relative_to(tmp_path).as_posix()
            rows.append(
                {"file_name": f"{label}/{name}.png", "label": int(label == "anomaly"), "mask": mask}
            )
        (directory / "metadata.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
    return tmp_path


def load(root):
    return load_metalparts(root, MetalPartsSource(SYNTHETIC_REVISION))


def metadata(root):
    return root / "MetalParts/metal_parts_data_dir_train/metadata.jsonl"


def edit_row(root, updates, index=0):
    path = metadata(root)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[index].update(updates)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def inventory(root):
    result = []
    for i, path in enumerate(sorted((root / "MetalParts").rglob("*"))):
        if path.is_file():
            content = path.read_bytes()
            row = {"path": path.relative_to(root).as_posix(), "size": len(content)}
            if i % 2:
                row["lfs"] = {"oid": hashlib.sha256(content).hexdigest()}
            else:
                row["oid"] = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
            result.append(row)
    return result


def test_preserves_anomalous_training_source_and_domains(fabricated):
    manifest = load(fabricated)
    assert len(manifest.samples) == 16
    assert len(manifest.training_normals) == 1
    assert sum(s.official_split == OfficialSplit.TRAIN for s in manifest.samples) == 2
    assert {s.source_domain for s in manifest.samples} == set(DOMAINS)
    assert all(
        (s.width, s.height, s.category, s.group_id) == (48, 40, "metalparts", None)
        for s in manifest.samples
    )
    assert len({s.sample_id for s in manifest.samples}) == 16
    assert load(fabricated).sha256 == manifest.sha256
    exported = manifest.to_dict()
    assert exported["source"]["license_id"] == "LicenseRef-FactoryVision-Synthetic"
    assert exported["source"]["uri"].startswith("urn:")
    assert "fabricated" in exported["source"]["attribution"]
    with pytest.raises(FrozenInstanceError):
        manifest.samples[0].width = 4
    official = MetalPartsSource()
    assert official.dataset_id == "robustad"
    assert official.license_id == "CC-BY-4.0"
    assert REVISION in official.uri
    assert "Amazon Science" in official.attribution
    assert replace(manifest.samples[0], source=official).sample_id != manifest.samples[0].sample_id
    with pytest.raises(ContractError, match="SOURCE_INVALID"):
        MetalPartsSource("main")


@pytest.mark.parametrize(
    "updates,index,code",
    [
        ({"label": True}, 0, "LABEL_INVALID"),
        ({"label": 2}, 0, "LABEL_INVALID"),
        ({"mask": ""}, 0, "MASK_LABEL_MISMATCH"),
        ({"mask": None}, 1, "MASK_LABEL_MISMATCH"),
        ({"mask": "MetalParts/metal_parts_data_dir_test0/masks/wrong.png"}, 1, "MASK_REFERENCE"),
        ({"mask": "../../outside.png"}, 1, "PATH_UNSAFE"),
        ({"file_name": "../outside.png"}, 0, "PATH_UNSAFE"),
        ({"file_name": "C:/outside.png"}, 0, "PATH_UNSAFE"),
        ({"file_name": "anomaly/false.png"}, 0, "SOURCE_LAYOUT"),
        ({"file_name": "normal/a/b.png"}, 0, "SOURCE_LAYOUT"),
        ({"file_name": "normal/a.gif"}, 0, "SOURCE_LAYOUT"),
        ({"file_name": "normal/missing.png"}, 0, "FILE_MISSING"),
        ({"extra": 3}, 0, "METADATA_FIELDS"),
    ],
)
def test_metadata_rejected(fabricated, updates, index, code):
    edit_row(fabricated, updates, index)
    with pytest.raises(ContractError, match=code):
        load(fabricated)


@pytest.mark.parametrize(
    "text,code",
    [
        ("{", "METADATA_INVALID"),
        ("[]", "METADATA_FIELDS"),
        ('{"label":0,"label":1}', "METADATA_INVALID"),
        ("\n", "METADATA_INVALID"),
        ("", "DOMAIN_MISSING"),
    ],
)
def test_malformed_jsonl(fabricated, text, code):
    metadata(fabricated).write_text(text, encoding="utf-8")
    with pytest.raises(ContractError, match=code):
        load(fabricated)


def test_metadata_encoding_and_size(fabricated):
    metadata(fabricated).write_bytes(b"\xff")
    with pytest.raises(ContractError, match="METADATA_INVALID"):
        load(fabricated)
    metadata(fabricated).write_bytes(b" " * (4 * 1024**2 + 1))
    with pytest.raises(ContractError, match="METADATA_TOO_LARGE"):
        load(fabricated)


@pytest.mark.parametrize(
    "size,mode,value,code",
    [
        ((47, 40), "L", 255, "MASK_DIMENSIONS"),
        ((48, 40), "L", 0, "MASK_EMPTY"),
        ((48, 40), "RGB", 255, "MASK_MODE"),
    ],
)
def test_mask_geometry_and_semantics(fabricated, size, mode, value, code):
    row = json.loads(metadata(fabricated).read_text().splitlines()[1])
    Image.new(mode, size, value).save(fabricated / row["mask"])
    with pytest.raises(ContractError, match=code):
        load(fabricated)


def test_duplicate_and_orphan(fabricated):
    original = metadata(fabricated).read_text()
    metadata(fabricated).write_text(original + "\n" + original.splitlines()[0])
    with pytest.raises(ContractError, match="SAMPLE_DUPLICATE"):
        load(fabricated)
    metadata(fabricated).write_text(original)
    (metadata(fabricated).parent / "orphan.txt").write_text("synthetic")
    with pytest.raises(ContractError, match="INVENTORY_MISMATCH"):
        load(fabricated)


def test_typed_envelope_guards(fabricated):
    manifest = load(fabricated)
    sample = manifest.samples[0]
    for updates, code in [
        ({"width": True}, "IMAGE_DIMENSIONS"),
        ({"height": 0}, "IMAGE_DIMENSIONS"),
        ({"source": None}, "TYPE_INVALID"),
        ({"source_domain": "test7"}, "DOMAIN_INVALID"),
        ({"label": "normal"}, "ENUM_INVALID"),
    ]:
        with pytest.raises(ContractError, match=code):
            replace(sample, **updates)
    for source, samples, code in [
        (None, manifest.samples, "TYPE_INVALID"),
        (manifest.source, [], "SAMPLES_INVALID"),
        (MetalPartsSource(), manifest.samples, "DATASET_SOURCE_MISMATCH"),
        (manifest.source, (None,), "DATASET_SOURCE_MISMATCH"),
        (manifest.source, (sample,), "DOMAIN_MISSING"),
    ]:
        with pytest.raises(ContractError, match=code):
            MetalPartsManifest(source, samples)


def test_inventory_hashes_and_corruption(fabricated):
    entries = inventory(fabricated)
    verified = verify_source_files(fabricated, entries, REVISION)
    assert len(verified["files"]) == 32  # 16 fabricated images, 8 masks, 8 JSONL.
    assert verified["bytes"] == sum(e["size"] for e in entries)
    assert len(verified["all_file_byte_duplicate_groups"]) == 1
    assert len(verified["all_file_byte_duplicate_groups"][0]["paths"]) == 8
    assert all("/masks/" in p for p in verified["all_file_byte_duplicate_groups"][0]["paths"])
    images = sorted((fabricated / "MetalParts").glob("*/normal/*.png"))
    images[1].write_bytes(images[0].read_bytes())
    repeated = verify_source_files(fabricated, inventory(fabricated), REVISION)
    assert sorted(len(g["paths"]) for g in repeated["all_file_byte_duplicate_groups"]) == [2, 8]


def test_inventory_failures(fabricated):
    entries = inventory(fabricated)
    for rows, revision, code in [
        (entries, "main", "SOURCE_INVALID"),
        (entries + [entries[0]], REVISION, "INVENTORY_INVALID"),
        ([{"path": "Other/a.png"}], REVISION, "INVENTORY_INVALID"),
        (entries[:-1], REVISION, "INVENTORY_MISMATCH"),
        ([], REVISION, "INVENTORY_MISMATCH"),
    ]:
        with pytest.raises(ContractError, match=code):
            verify_source_files(fabricated, rows, revision)
    (fabricated / entries[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ContractError, match="SOURCE_HASH_MISMATCH"):
        verify_source_files(fabricated, entries, REVISION)


def test_fixed_screen_is_candidates_not_confirmed_identity(fabricated):
    manifest = load(fabricated)
    train = manifest.training_normals[0]
    test = next(
        s for s in manifest.samples if s.source_domain == "test0" and s.label == Label.NORMAL
    )
    # Same pixels, different encoding bytes: pixel identity must still block.
    Image.open(fabricated / train.image_path).save(fabricated / test.image_path, compress_level=0)
    report = audit_metalparts(fabricated, manifest)
    assert report["images"] == 16 and report["masks"] == 8
    assert report["dimensions"] == {"48x40": 16}
    assert report["mask_values"] == [0, 255]
    assert report["pixel_duplicate_groups"][0]["train_test"]
    assert not report["byte_duplicate_groups"]
    assert report["train_test_near_candidates"] >= 1
    assert report["confirmed_related_captures"] is None
    assert report["verdict"].startswith("BLOCKED")
    assert len(report["blockers"]) == 2
    (fabricated / test.image_path).write_bytes((fabricated / train.image_path).read_bytes())
    assert audit_metalparts(fabricated, manifest)["byte_duplicate_groups"][0]["train_test"]


def test_clear_synthetic_screen_and_evidence_no_overwrite(fabricated, monkeypatch):
    monkeypatch.setattr(
        "factoryvision.data.metalparts_audit.screen_pairs", lambda _: {"candidates": []}
    )
    report = audit_metalparts(fabricated, load(fabricated))
    assert report["verdict"] == "READY FOR METALPARTS TRAINING PLANNING"
    target = fabricated / "synthetic-evidence.json"
    write_json(target, report)
    with pytest.raises(FileExistsError):
        write_json(target, report)


def test_filename_correspondence_is_unresolved_not_proven_duplicate(fabricated, monkeypatch):
    manifest = load(fabricated)
    train = manifest.training_normals[0]
    test = next(
        s for s in manifest.samples if s.source_domain == "test0" and s.label == Label.NORMAL
    )
    renamed = replace(test, file_name=train.file_name)
    (fabricated / test.image_path).rename(fabricated / renamed.image_path)
    samples = tuple(renamed if s == test else s for s in manifest.samples)
    monkeypatch.setattr(
        "factoryvision.data.metalparts_audit.screen_pairs", lambda _: {"candidates": []}
    )
    result = audit_metalparts(fabricated, MetalPartsManifest(manifest.source, samples))
    assert len(result["cross_domain_basename_groups"]) == 1
    assert result["byte_duplicate_groups"] == result["pixel_duplicate_groups"] == []
    assert result["confirmed_related_captures"] is None
    assert result["verdict"].startswith("BLOCKED")


@pytest.mark.parametrize("kind,code", [("image", "IMAGE_DECODE"), ("mask", "FILE_MISSING")])
def test_corrupt_image_and_missing_mask(fabricated, kind, code):
    rows = [json.loads(line) for line in metadata(fabricated).read_text().splitlines()]
    if kind == "image":
        (metadata(fabricated).parent / rows[0]["file_name"]).write_bytes(b"fabricated bad encoding")
    else:
        (fabricated / rows[1]["mask"]).unlink()
    with pytest.raises(ContractError, match=code):
        load(fabricated)
