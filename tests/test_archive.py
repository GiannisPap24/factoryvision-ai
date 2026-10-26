"""Fabricated tar archives exercise preflight and selective extraction offline."""

import io
import sys
import tarfile

import pytest

from factoryvision.data.archive import extract_cashew, inspect_archive, select_cashew
from factoryvision.domain.errors import ContractError


def write_tar(path, entries):
    with tarfile.open(path, "w") as archive:
        for name, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "../outside"
            member.size = 3 if kind == tarfile.REGTYPE else 0
            archive.addfile(member, io.BytesIO(b"abc") if member.isfile() else None)
    return path


def test_selective_extract_and_no_overwrite(tmp_path):
    path = write_tar(
        tmp_path / "safe.tar",
        [
            ("cashew", tarfile.DIRTYPE),
            ("cashew/image_anno.csv", tarfile.REGTYPE),
            ("cashew/Data/Images/Normal/synthetic.JPG", tarfile.REGTYPE),
            ("cashew/Data/Images/Anomaly/synthetic.JPG", tarfile.REGTYPE),
            ("cashew/Data/Masks/Anomaly/synthetic.png", tarfile.REGTYPE),
            ("pcb1/Data/Images/Normal/deferred.JPG", tarfile.REGTYPE),
        ],
    )
    result = extract_cashew(path, tmp_path / "output")
    assert result["inspected_members"] == 6 and result["selected_files"] == 4
    assert not (tmp_path / "output/pcb1").exists()
    assert (tmp_path / "output/cashew/image_anno.csv").read_bytes() == b"abc"
    with pytest.raises(ContractError, match="EXTRACTION_EXISTS"):
        extract_cashew(path, tmp_path / "output")


@pytest.mark.parametrize(
    "name,kind,code",
    [
        ("/absolute", tarfile.REGTYPE, "PATH_UNSAFE"),
        ("../outside", tarfile.REGTYPE, "PATH_UNSAFE"),
        ("C:/outside", tarfile.REGTYPE, "PATH_UNSAFE"),
        ("cashew/link", tarfile.SYMTYPE, "ARCHIVE_TYPE"),
        ("cashew/hard", tarfile.LNKTYPE, "ARCHIVE_TYPE"),
        ("cashew/fifo", tarfile.FIFOTYPE, "ARCHIVE_TYPE"),
        ("unknown/file", tarfile.REGTYPE, "ARCHIVE_STRUCTURE"),
    ],
)
def test_reject_unsafe_members_before_writing(tmp_path, name, kind, code):
    path = write_tar(tmp_path / "unsafe.tar", [(name, kind)])
    with pytest.raises(ContractError, match=code):
        extract_cashew(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_duplicates_file_ancestor_and_structure(tmp_path):
    for entries, code in [
        ([("cashew/a", tarfile.REGTYPE), ("cashew/A", tarfile.REGTYPE)], "ARCHIVE_DUPLICATE"),
        ([("cashew/a", tarfile.REGTYPE), ("cashew/a/b", tarfile.REGTYPE)], "ARCHIVE_STRUCTURE"),
        ([("cashew/unexpected.exe", tarfile.REGTYPE)], "ARCHIVE_STRUCTURE"),
        ([("pcb1/Data/Images/Normal/deferred.JPG", tarfile.REGTYPE)], "ARCHIVE_LIMIT"),
    ]:
        path = write_tar(tmp_path / "bad.tar", entries)
        with pytest.raises(ContractError, match=code):
            select_cashew(inspect_archive(path))


@pytest.mark.parametrize(
    "limit,value",
    [
        ("MAX_MEMBERS", 0),
        ("MAX_MEMBER_BYTES", 2),
        ("MAX_EXPANDED_BYTES", 2),
        ("MAX_SELECTED_BYTES", 2),
    ],
)
def test_resource_limits(tmp_path, monkeypatch, limit, value):
    from factoryvision.data import archive

    monkeypatch.setattr(archive, limit, value)
    path = write_tar(tmp_path / "limited.tar", [("cashew/image_anno.csv", tarfile.REGTYPE)])
    with pytest.raises(ContractError, match="ARCHIVE_LIMIT"):
        extract_cashew(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Real Windows junction check")
def test_extraction_parent_junction_rejected(tmp_path):
    import _winapi

    outside = tmp_path / "outside"
    outside.mkdir()
    _winapi.CreateJunction(str(outside), str(tmp_path / "linked"))
    path = write_tar(tmp_path / "safe.tar", [("cashew/image_anno.csv", tarfile.REGTYPE)])
    with pytest.raises(ContractError, match="PATH_ESCAPE"):
        extract_cashew(path, tmp_path / "linked/output")
    assert not (outside / "output").exists()
