"""Inspect distribution members and text without extracting or publishing them."""

import argparse
import json
import posixpath
import re
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

PRIVATE_PATH = re.compile(r"(?:[A-Z]:[/\\]+Users[/\\]+|/(?:home|Users)/)[^\s/\\]+", re.I)
SECRET = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|\bAKIA[A-Z0-9]{16}\b"
)
FORBIDDEN_ROOTS = {".git", "data", "runs", "reports", ".tmp", ".tools", ".uv-cache"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".png", ".jpg", ".jpeg", ".zip", ".exe", ".dll"}


def inspect_distribution(archive: Path) -> dict:
    """Return sanitized findings; never print matched private content."""
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as package:
            contents = {n: package.read(n) for n in package.namelist() if not n.endswith("/")}
    else:
        with tarfile.open(archive) as package:
            contents = {
                member.name.split("/", 1)[-1]: package.extractfile(member).read()
                for member in package.getmembers()
                if member.isfile()
            }
    findings = []
    for name, payload in contents.items():
        path = Path(name)
        first = name.split("/", 1)[0]
        if (
            first in FORBIDDEN_ROOTS
            or first.startswith(".venv")
            or path.name in {"AGENTS.md", ".env"}
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
            or ".." in path.parts
            or name.startswith(("/", "\\"))
        ):
            findings.append({"member": name, "rule": "forbidden-member"})
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"member": name, "rule": "unexpected-binary"})
            continue
        normalized = text.replace("\\\\", "\\")
        for rule, pattern in [("private-path", PRIVATE_PATH), ("secret-pattern", SECRET)]:
            for number, line in enumerate(normalized.splitlines(), 1):
                if pattern.search(line):
                    findings.append({"member": name, "line": number, "rule": rule})
        if path.suffix == ".md":
            for url in re.findall(r"\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", text):
                parsed = urlsplit(url.strip("<>"))
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target = posixpath.normpath(
                    posixpath.join(posixpath.dirname(name), unquote(parsed.path))
                )
                if target not in contents:
                    findings.append({"member": name, "rule": "missing-doc-link", "target": target})
    return {"archive": archive.name, "members": len(contents), "findings": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    args = parser.parse_args()
    results = [inspect_distribution(path) for path in args.archives]
    print(json.dumps(results, indent=2))
    raise SystemExit(1 if any(r["findings"] for r in results) else 0)


if __name__ == "__main__":
    main()
