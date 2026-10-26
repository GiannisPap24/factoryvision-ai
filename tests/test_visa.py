"""Fabricated VisA-shaped CSVs/images; no real dataset files or network access."""

import csv
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from factoryvision.data.visa import annotation_label, audit_cashew, load_visa_cashew, read_csv
from factoryvision.domain.errors import ContractError


def write_rows(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fabricated_visa(tmp_path):
    rows = []
    annotations = []
    for index in range(5):
        anomaly = index == 4
        label = "anomaly" if anomaly else "normal"
        kind = "Anomaly" if anomaly else "Normal"
        image = f"cashew/Data/Images/{kind}/fabricated-{index}.JPG"
        mask = f"cashew/Data/Masks/Anomaly/fabricated-{index}.png" if anomaly else ""
        path = tmp_path / image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (index * 40, 50, 90)).save(path)
        if mask:
            path = tmp_path / mask
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.zeros((12, 16), dtype=np.uint8)
            pixels[2:8, 4:10] = 1
            Image.fromarray(pixels).save(path)
        rows.append(
            dict(
                object="cashew",
                split="train" if index < 3 else "test",
                label=label,
                image=image,
                mask=mask,
            )
        )
        annotations.append(dict(image=image, label="burnt" if anomaly else "normal", mask=mask))
    split_path = tmp_path / "1cls.csv"
    annotation_path = tmp_path / "image_anno.csv"
    write_rows(split_path, ["object", "split", "label", "image", "mask"], rows)
    write_rows(annotation_path, ["image", "label", "mask"], annotations)
    return tmp_path, split_path, annotation_path


def test_official_shape_ingestion_and_audit(fabricated_visa):
    root, split, annotations = fabricated_visa
    original = load_visa_cashew(split)
    enriched, report = audit_cashew(original, root, annotations)
    assert report["samples"] == 5 and report["masks"] == 1
    assert report["image_formats"] == {"JPEG": 5}
    assert report["mask_class_ids"] == [0, 1]
    assert report["byte_duplicate_groups"] == report["pixel_duplicate_groups"] == []
    assert all(s.content_sha256 and s.group_id for s in enriched.samples)
    assert all(s.content_sha256 is None for s in original.samples)
    assert audit_cashew(original, root, annotations) == (enriched, report)


@pytest.mark.parametrize(
    "content,code",
    [
        (b"wrong,header\n", "CSV_FIELDS"),
        (b"image,label,mask\nx,normal\n", "CSV_FIELDS"),
        (b"image,label,mask\nx,normal,,extra\n", "CSV_FIELDS"),
        (b"\xff", "CSV_INVALID"),
        (b'image,label,mask\n"unterminated', "CSV_INVALID"),
        (b"x" * (2 * 1024**2 + 1), "CSV_LIMIT"),
    ],
    ids=["header", "missing-column", "extra-column", "utf8", "quoting", "size-limit"],
)
def test_csv_strict_boundary(tmp_path, content, code):
    path = tmp_path / "bad.csv"
    path.write_bytes(content)
    with pytest.raises(ContractError, match=code):
        read_csv(path, ["image", "label", "mask"])


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("label", "unknown", "CSV_VALUE"),
        ("split", "validation", "CSV_VALUE"),
        ("image", "../bad.JPG", "PATH_UNSAFE"),
        ("mask", "missing.png", "MASK_LABEL_MISMATCH"),
    ],
)
def test_invalid_official_records(fabricated_visa, field, value, code):
    _, split, _ = fabricated_visa
    fields = ["object", "split", "label", "image", "mask"]
    rows = read_csv(split, fields)
    rows[0][field] = value
    write_rows(split, fields, rows)
    with pytest.raises(ContractError, match=code):
        load_visa_cashew(split)


def test_other_category_is_not_loaded(fabricated_visa):
    _, split, _ = fabricated_visa
    fields = ["object", "split", "label", "image", "mask"]
    rows = read_csv(split, fields)
    rows.append({**rows[0], "object": "pcb1"})
    write_rows(split, fields, rows)
    assert len(load_visa_cashew(split).samples) == 5


def test_annotation_taxonomy():
    assert annotation_label("normal") == "normal"
    assert annotation_label("burnt,small holes") == "anomaly"
    for label in ("", "anomaly", "normal,burnt", "unknown"):
        with pytest.raises(ContractError, match="CSV_VALUE"):
            annotation_label(label)


def test_annotation_mismatch_and_duplicate(fabricated_visa):
    root, split, annotations = fabricated_visa
    manifest = load_visa_cashew(split)
    fields = ["image", "label", "mask"]
    rows = read_csv(annotations, fields)
    write_rows(annotations, fields, rows + [rows[0]])
    with pytest.raises(ContractError, match="ANNOTATION_MISMATCH"):
        audit_cashew(manifest, root, annotations)
    write_rows(annotations, fields, rows[:-1])
    with pytest.raises(ContractError, match="ANNOTATION_MISMATCH"):
        audit_cashew(manifest, root, annotations)


def test_extra_missing_and_invalid_files(fabricated_visa):
    root, split, annotations = fabricated_visa
    manifest = load_visa_cashew(split)
    extra = root / "cashew/Data/unreferenced.bin"
    extra.write_bytes(b"unexpected")
    with pytest.raises(ContractError, match="INVENTORY_MISMATCH"):
        audit_cashew(manifest, root, annotations)
    extra.unlink()
    sample = next(s for s in manifest.samples if s.mask_path)
    Image.new("L", (16, 12), 255).save(root / sample.mask_path)
    with pytest.raises(ContractError, match="MASK_CLASS"):
        audit_cashew(manifest, root, annotations)


def test_wrong_source_and_disguised_format(manifest, fabricated_visa):
    root, split, annotations = fabricated_visa
    with pytest.raises(ContractError, match="AUDIT_SOURCE"):
        audit_cashew(manifest, root, annotations)
    official = load_visa_cashew(split)
    path = root / official.samples[0].image_path
    Image.new("RGB", (16, 12), "red").save(path, format="PNG")
    with pytest.raises(ContractError, match="VISA_IMAGE_FORMAT"):
        audit_cashew(official, root, annotations)


def test_actual_content_duplicate_train_test_rejected(fabricated_visa):
    root, split, annotations = fabricated_visa
    manifest = load_visa_cashew(split)
    train = next(s for s in manifest.samples if s.official_split.value == "train")
    test = next(s for s in manifest.samples if s.official_split.value == "test" and not s.mask_path)
    (root / test.image_path).write_bytes((root / train.image_path).read_bytes())
    with pytest.raises(ContractError, match="CONTENT_LEAKAGE|GROUP_LEAKAGE"):
        audit_cashew(manifest, root, annotations)


def test_same_partition_duplicates_form_one_group(fabricated_visa):
    root, split, annotations = fabricated_visa
    manifest = load_visa_cashew(split)
    train = [s for s in manifest.samples if s.official_split.value == "train"]
    (root / train[1].image_path).write_bytes((root / train[0].image_path).read_bytes())
    enriched, report = audit_cashew(manifest, root, annotations)
    assert len(report["byte_duplicate_groups"]) == len(report["pixel_duplicate_groups"]) == 1
    groups = {
        s.group_id
        for s in enriched.samples
        if s.sample_id in (train[0].sample_id, train[1].sample_id)
    }
    assert len(groups) == 1


def test_non_cashew_source_is_rejected(fabricated_visa):
    root, split, annotations = fabricated_visa
    manifest = load_visa_cashew(split)
    sample = manifest.samples[0]
    changed = replace(
        sample,
        category="pcb1",
        image_path=sample.image_path.replace("cashew/", "pcb1/"),
        mask_path=sample.mask_path.replace("cashew/", "pcb1/") if sample.mask_path else None,
    )
    candidate = replace(manifest, samples=(changed,))
    with pytest.raises(ContractError, match="AUDIT_SOURCE"):
        audit_cashew(candidate, root, annotations)
