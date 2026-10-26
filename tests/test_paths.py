"""Lexical checks are portable; actual symlink/junction resolution is opt-in."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from factoryvision.data.filesystem import resolve_reference, validate_files
from factoryvision.domain.errors import ContractError
from factoryvision.domain.paths import check_relative_path


@pytest.mark.parametrize(
    "reference",
    [
        "/etc/passwd",
        "C:/outside.JPG",
        "C:relative.JPG",
        r"\\server\share\a.JPG",
        "//server/share/a.JPG",
        r"cashew\image.JPG",
        "../image.JPG",
        "cashew/../image.JPG",
        "./image.JPG",
        "cashew//image.JPG",
        "cashew/",
        "cashew/file:stream.JPG",
        "cashew/NUL.JPG",
        "cashew/con",
        "cashew/COM1.JPG",
        "cashew/LPT9.JPG",
        "cashew/COM¹.JPG",
        "cashew/file. ",
        "cashew/file.",
        " cashew/file.JPG",
        "cashew/a?.JPG",
        "cashew/a*.JPG",
        "cashew/a|b.JPG",
        "cashew/a<b.JPG",
        "cashew/a>b.JPG",
        'cashew/a"b.JPG',
        "cashew/a\x00.JPG",
        "cashew/a\n.JPG",
    ],
)
def test_unsafe_reference_rejected(reference):
    with pytest.raises(ContractError, match="PATH_UNSAFE"):
        check_relative_path(reference, "image_path")


@pytest.mark.parametrize("reference", ["", None, 42, []])
def test_malformed_reference(reference):
    with pytest.raises(ContractError, match="PATH_INVALID"):
        check_relative_path(reference, "image_path")


def test_safe_reference():
    assert check_relative_path("cashew/Data/Images/Normal/fabricated.JPG", "image") is None


@pytest.mark.filesystem
@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction integration test")
def test_windows_junction_escape(tmp_path):
    import _winapi

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "synthetic.JPG").write_bytes(b"fabricated")
    _winapi.CreateJunction(str(outside), str(root / "escape"))
    with pytest.raises(ContractError, match="PATH_ESCAPE"):
        resolve_reference(root, "escape/synthetic.JPG")
    with pytest.raises(ContractError, match="PATH_ESCAPE"):
        resolve_reference(root, "escape/missing.JPG")


@pytest.mark.filesystem
def test_existing_files_and_missing_root(tmp_path, manifest):
    for sample in manifest.samples:
        for reference in (sample.image_path, sample.mask_path):
            if reference is not None:
                file = tmp_path / reference
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"synthetic placeholder; deliberately not an image")
    assert len(validate_files(manifest, tmp_path)) == len(manifest.samples) + 1
    assert resolve_reference(tmp_path, manifest.samples[0].image_path).is_file()
    with pytest.raises(ContractError, match="FILE_MISSING"):
        resolve_reference(tmp_path, "missing.JPG")
    with pytest.raises(ContractError, match="FILE_MISSING"):
        resolve_reference(tmp_path, "cashew")
    with pytest.raises(ContractError, match="PATH_RESOLUTION"):
        resolve_reference(tmp_path / "missing-root", "image.JPG")
    with pytest.raises(ContractError, match="ROOT_INVALID"):
        resolve_reference(tmp_path / manifest.samples[0].image_path, "image.JPG")


@pytest.mark.filesystem
def test_symlink_escape_and_internal_symlink(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "sample.JPG").write_bytes(b"fabricated")
    (root / "inside.JPG").write_bytes(b"fabricated")
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
        (root / "alias.JPG").symlink_to(root / "inside.JPG")
    except OSError as exc:
        pytest.skip(f"Host does not permit symlink creation: {exc}")
    with pytest.raises(ContractError, match="PATH_ESCAPE"):
        resolve_reference(root, "escape/sample.JPG")
    with pytest.raises(ContractError, match="PATH_ESCAPE"):
        resolve_reference(root, "escape/nonexistent.JPG")
    assert resolve_reference(root, "alias.JPG") == (root / "inside.JPG").resolve()


@pytest.mark.filesystem
def test_resolution_loop_or_os_error_is_actionable(tmp_path):
    for exception in (OSError("blocked"), RuntimeError("loop")):
        with (
            patch.object(Path, "resolve", side_effect=exception),
            pytest.raises(ContractError, match="PATH_RESOLUTION"),
        ):
            resolve_reference(tmp_path, "image.JPG")
