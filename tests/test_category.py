"""Synthetic category payloads prove that held-out archive members stay unopened."""

import csv
import io
import tarfile
from dataclasses import replace

import pytest
from PIL import Image

from factoryvision.data.category import (
    audit_normals,
    extract_normals,
    load_category,
    normal_members,
)
from factoryvision.data.normal_review import review_normals, review_samples
from factoryvision.data.splits import derive_splits
from factoryvision.domain.contracts import OfficialSplit
from factoryvision.training.checkpoint import file_sha256


@pytest.fixture
def category(tmp_path):
    rows = []
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, "w") as tar:
        for index in range(12):
            train = index < 10
            path = f"pcb1/Data/Images/Normal/{index:04}.JPG"
            rows.append(
                dict(
                    object="pcb1",
                    split="train" if train else "test",
                    label="normal",
                    image=path,
                    mask="",
                )
            )
            buffer = io.BytesIO()
            Image.new("RGB", (16, 12), (index * 20, 50, 90)).save(buffer, format="JPEG")
            info = tarfile.TarInfo(path)
            info.size = len(buffer.getvalue())
            tar.addfile(info, io.BytesIO(buffer.getvalue()))
    split = tmp_path / "split.csv"
    with split.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return archive, split, tmp_path / "normal-only"


def test_only_normals_opened_and_official_membership_preserved(category, monkeypatch):
    archive, split, root = category
    manifest = load_category(split, "pcb1")
    opened = []
    original = tarfile.TarFile.extractfile

    def tracked(self, member):
        opened.append(member.name)
        return original(self, member)

    monkeypatch.setattr(tarfile.TarFile, "extractfile", tracked)
    report = extract_normals(archive, root, manifest, file_sha256(archive))
    assert set(opened) == {s.image_path for s in normal_members(manifest)}
    assert report["selected_files"] == 10 and report["heldout_payloads_extracted"] == 0
    enriched, audit = audit_normals(manifest, root)
    derived = derive_splits(enriched, 42)
    assert audit["normal_images"] == 10 and audit["test_images_decoded"] == 0
    assert audit["byte_duplicate_groups"] == audit["pixel_duplicate_groups"] == []
    assert [s for s in derived.samples if s.official_split == OfficialSplit.TEST] == [
        s for s in manifest.samples if s.official_split == OfficialSplit.TEST
    ]
    groups = {}
    for s in derived.samples:
        if s.group_id:
            groups.setdefault(s.group_id, set()).add(s.partition)
    assert all(len(partitions) == 1 for partitions in groups.values())
    assert derived == derive_splits(enriched, 42)
    with pytest.raises(FileExistsError):
        extract_normals(archive, root, manifest, file_sha256(archive))
    (root / "unapproved.txt").write_text("extra")
    with pytest.raises(ValueError, match="inventory"):
        audit_normals(manifest, root)


def test_reject_identity_missing_and_category(category):
    archive, split, root = category
    with pytest.raises(ValueError, match="category"):
        load_category(split, "unknown")
    manifest = load_category(split, "pcb1")
    with pytest.raises(ValueError, match="identity"):
        extract_normals(archive, root, manifest, "0" * 64)
    changed = replace(
        manifest,
        samples=tuple(
            replace(s, image_path=s.image_path.replace("0000", "absent")) for s in manifest.samples
        ),
    )
    with pytest.raises(ValueError, match="Missing"):
        extract_normals(archive, root, changed, file_sha256(archive))
    assert not root.exists()


def test_exact_duplicates_are_grouped(category):
    archive, split, root = category
    manifest = load_category(split, "pcb1")
    extract_normals(archive, root, manifest, file_sha256(archive))
    normals = normal_members(manifest)
    (root / normals[1].image_path).write_bytes((root / normals[0].image_path).read_bytes())
    enriched, audit = audit_normals(manifest, root)
    assert len(audit["byte_duplicate_groups"]) == len(audit["pixel_duplicate_groups"]) == 1
    derived = derive_splits(enriched, 42)
    assert derived.samples[0].partition == derived.samples[1].partition


def test_normal_only_refinement_is_deterministic(category):
    archive, split, root = category
    manifest = load_category(split, "pcb1")
    extract_normals(archive, root, manifest, file_sha256(archive))
    enriched, _ = audit_normals(manifest, root)
    derived = derive_splits(enriched, 42)
    first = review_normals(derived, root)
    assert first == review_normals(derived, root)
    assert first["heldout_images_decoded"] == 0
    assert first["candidate_count"] > 0
    assert all(c["crop_correlation"] is None for c in first["refined_candidates"])
    assert len(first["selected_for_visual_review"]) <= 24
    import numpy as np

    image = Image.fromarray(np.tile(np.arange(16, dtype=np.uint8), (12, 1))).convert("RGB")
    samples = [
        replace(derived.samples[0], partition=derived.samples[1].partition),
        replace(derived.samples[1], partition=derived.samples[-1].partition),
    ]
    result = review_samples(samples, lambda _: image)
    assert result["strong_count"] == 1
