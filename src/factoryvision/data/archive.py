"""Inspect every tar member before selectively copying regular Cashew files."""

import shutil
import tarfile
from pathlib import Path, PurePosixPath

from factoryvision.domain.contracts import VISA_CATEGORIES
from factoryvision.domain.errors import reject
from factoryvision.domain.paths import check_relative_path

MAX_MEMBERS = 50_000
MAX_MEMBER_BYTES = 32 * 1024**2
MAX_EXPANDED_BYTES = 3 * 1024**3
MAX_SELECTED_BYTES = 512 * 1024**2


def _expected_structure(member: tarfile.TarInfo) -> bool:
    parts = PurePosixPath(member.name).parts
    if parts == ("LICENSE-DATASET",):
        return member.isfile()
    if parts[0] == "split_csv":
        return (len(parts) == 1 and member.isdir()) or (
            len(parts) == 2
            and member.isfile()
            and parts[1] in ("1cls.csv", "2cls_fewshot.csv", "2cls_highshot.csv")
        )
    if parts[0] not in VISA_CATEGORIES:
        return False
    tail = parts[1:]
    if member.isdir():
        return tail in (
            (),
            ("Data",),
            ("Data", "Images"),
            ("Data", "Masks"),
            ("Data", "Images", "Normal"),
            ("Data", "Images", "Anomaly"),
            ("Data", "Masks", "Anomaly"),
        )
    return (
        tail == ("image_anno.csv",)
        or (
            len(tail) == 4
            and tail[:2] == ("Data", "Images")
            and tail[2] in ("Normal", "Anomaly")
            and tail[3].endswith(".JPG")
        )
        or (
            len(tail) == 4 and tail[:3] == ("Data", "Masks", "Anomaly") and tail[3].endswith(".png")
        )
    )


def inspect_archive(path: Path) -> tuple[tarfile.TarInfo, ...]:
    members = []
    seen = set()
    expanded = 0
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            if len(members) >= MAX_MEMBERS:
                reject("ARCHIVE_LIMIT", "members", "Too many archive entries.")
            if not member.isdir() and not member.isfile():
                reject("ARCHIVE_TYPE", member.name, "Only regular files/directories are allowed.")
            if member.sparse is not None:
                reject("ARCHIVE_TYPE", member.name, "Sparse archive entries are unsupported.")
            name = member.name.rstrip("/") if member.isdir() else member.name
            check_relative_path(name, "archive.member")
            if name.casefold() in seen:
                reject("ARCHIVE_DUPLICATE", name, "Duplicate or case-aliased archive member.")
            seen.add(name.casefold())
            if member.size < 0 or member.size > MAX_MEMBER_BYTES:
                reject("ARCHIVE_LIMIT", name, "Member exceeds the declared byte bound.")
            expanded += member.size
            if expanded > MAX_EXPANDED_BYTES:
                reject("ARCHIVE_LIMIT", "bytes", "Archive expansion exceeds the byte bound.")
            members.append(member)
    # A file must never be the ancestor of another member, even outside Cashew.
    regular = {m.name.casefold() for m in members if m.isfile()}
    for member in members:
        if any(str(p).casefold() in regular for p in PurePosixPath(member.name).parents):
            reject("ARCHIVE_STRUCTURE", member.name, "A regular file is used as a directory.")
    for member in members:
        if not _expected_structure(member):
            reject("ARCHIVE_STRUCTURE", member.name, "Unexpected VisA archive structure.")
    return tuple(members)


def select_cashew(members: tuple[tarfile.TarInfo, ...]) -> tuple[tarfile.TarInfo, ...]:
    selected = []
    for member in members:
        parts = PurePosixPath(member.name).parts
        if member.isfile() and member.name in ("LICENSE-DATASET", "split_csv/1cls.csv"):
            selected.append(member)
            continue
        if parts[0] != "cashew" or member.isdir():
            continue
        permitted = (
            parts == ("cashew", "image_anno.csv")
            or (
                len(parts) == 5
                and parts[:3] == ("cashew", "Data", "Images")
                and parts[3] in ("Normal", "Anomaly")
                and parts[4].endswith(".JPG")
            )
            or (
                len(parts) == 5
                and parts[:4] == ("cashew", "Data", "Masks", "Anomaly")
                and parts[4].endswith(".png")
            )
        )
        if not permitted:
            reject("ARCHIVE_STRUCTURE", member.name, "Unexpected Cashew file structure.")
        selected.append(member)
    if not selected or sum(m.size for m in selected) > MAX_SELECTED_BYTES:
        reject("ARCHIVE_LIMIT", "cashew", "Require a nonempty selection no larger than 512 MiB.")
    return tuple(selected)


def extract_cashew(path: Path, destination: Path) -> dict:
    """Require a new destination; copy bytes only, never apply archive modes/links."""
    members = inspect_archive(path)
    selected = select_cashew(members)
    if destination.exists() or destination.is_symlink():
        reject("EXTRACTION_EXISTS", "destination", "Use a new dedicated extraction directory.")
    if destination.parent.resolve(strict=True) != destination.parent.absolute():
        reject("PATH_ESCAPE", "destination", "Extraction parent must not traverse links.")
    destination.mkdir()
    resolved = destination.resolve(strict=True)
    with tarfile.open(path, "r:") as archive:
        for member in selected:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(resolved):
                reject("PATH_ESCAPE", member.name, "Extraction target escaped the root.")
            source = archive.extractfile(member)
            if source is None:
                reject("ARCHIVE_TYPE", member.name, "Missing regular-file payload.")
            with source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024**2)
            if target.stat().st_size != member.size:
                reject("ARCHIVE_TRUNCATED", member.name, "Extracted byte count differs.")
    return {
        "inspected_members": len(members),
        "selected_files": len(selected),
        "selected_bytes": sum(m.size for m in selected),
        "selected_paths": sorted(m.name for m in selected),
    }
