"""Explicit official archive acquisition. Never called by tests or folder-open tasks."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from factoryvision.domain.contracts import VISA_REVISION, VISA_SOURCE

ARCHIVE_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
EXPECTED_BYTES = 1929840640
MAX_BYTES = 2684354560  # Owner's 2.5 GiB ceiling.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "data/raw"
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination.absolute():
        raise ValueError("Acquisition destination must not traverse symlinks or junctions.")
    archive = destination / "VisA_20220922.tar"
    subprocess.run(["git", "check-ignore", "-q", str(archive)], cwd=root, check=True)
    if archive.exists():
        raise FileExistsError("Archive already exists; verify its saved acquisition record.")
    with urllib.request.urlopen(
        urllib.request.Request(ARCHIVE_URL, method="HEAD"), timeout=30
    ) as r:
        size = int(r.headers["Content-Length"])
        etag = r.headers.get("ETag")
        if r.url != ARCHIVE_URL or size != EXPECTED_BYTES or size > MAX_BYTES:
            raise ValueError(
                "Official archive location or documented size changed; review required."
            )
    if shutil.disk_usage(destination).free < size + 2 * 1024**3:
        raise ValueError("Insufficient disk budget for acquisition and selective extraction.")
    started = time.perf_counter()
    digest = hashlib.sha256()
    total = 0
    partial = archive.with_suffix(".tar.partial")
    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response, partial.open("xb") as output:
        if response.url != ARCHIVE_URL or int(response.headers["Content-Length"]) != size:
            raise ValueError("GET response differs from the reviewed source/size.")
        while chunk := response.read(4 * 1024**2):
            total += len(chunk)
            if total > size or total > MAX_BYTES:
                raise ValueError("Download exceeds declared or authorized size.")
            output.write(chunk)
            digest.update(chunk)
            if total % (128 * 1024**2) == 0:
                print(f"Downloaded {total / 1024**2:.0f} / {size / 1024**2:.0f} MiB", flush=True)
    if total != size:
        raise ValueError("Incomplete archive; partial file retained for diagnosis.")
    partial.rename(archive)
    record = {
        "source_url": ARCHIVE_URL,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
        "size_bytes": total,
        "sha256": digest.hexdigest(),
        "etag": etag,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "independently_authenticated_checksum": False,
        "checksum_note": (
            "Locally computed SHA-256; no authoritative SHA-256 found. "
            "Multipart ETag is not an MD5 checksum."
        ),
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "registry": "https://registry.opendata.aws/visa/",
        "metadata_revision": VISA_REVISION,
    }
    (destination / "acquisition.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    metadata = root / "data/metadata"
    metadata.mkdir(exist_ok=True)
    for name, url in (
        ("1cls.csv", VISA_SOURCE),
        (
            "LICENSE-DATASET",
            f"https://raw.githubusercontent.com/amazon-science/spot-diff/{VISA_REVISION}/LICENSE-DATASET",
        ),
        (
            "NOTICE",
            f"https://raw.githubusercontent.com/amazon-science/spot-diff/{VISA_REVISION}/NOTICE",
        ),
    ):
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read(2 * 1024**2 + 1)
        if len(content) > 2 * 1024**2:
            raise ValueError("Metadata exceeds the bounded download limit.")
        with (metadata / name).open("xb") as output:
            output.write(content)
        record.setdefault("metadata", {})[name] = {
            "url": url,
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    (destination / "acquisition.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
