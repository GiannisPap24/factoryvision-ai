"""Explicit local selective extraction after archive preflight; never downloads."""

import hashlib
import json
import subprocess
from pathlib import Path

from factoryvision.data.archive import extract_cashew


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    archive = root / "data/raw/VisA_20220922.tar"
    record = json.loads((root / "data/raw/acquisition.json").read_text(encoding="utf-8"))
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive.stat().st_size != record["size_bytes"] or digest != record["sha256"]:
        raise ValueError("Archive differs from its recorded acquisition.")
    destination = root / "data/visa"
    subprocess.run(["git", "check-ignore", "-q", str(destination / "cashew")], cwd=root, check=True)
    result = extract_cashew(archive, destination)
    for name, relative in (
        ("1cls.csv", "split_csv/1cls.csv"),
        ("LICENSE-DATASET", "LICENSE-DATASET"),
    ):
        content = (destination / relative).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["metadata"][name]["sha256"]:
            raise ValueError(f"Archive {name} differs from pinned official repository metadata.")
    result["pinned_metadata_matches_archive"] = True
    path = root / "data/metadata/extraction.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "selected_paths"}, indent=2))


if __name__ == "__main__":
    main()
