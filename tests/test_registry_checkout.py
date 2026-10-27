"""Exercise real Git checkout filters offline without changing the source repository."""

import hashlib
import subprocess
from pathlib import Path

import pytest

from factoryvision.inference.model_registry import SOURCES, source_bytes


@pytest.mark.parametrize("autocrlf", ["false", "true"])
def test_git_checkout_preserves_pinned_resource_bytes(tmp_path, autocrlf):
    def git(*args, payload=None):
        return subprocess.run(
            ["git", "-c", f"core.autocrlf={autocrlf}", *args],
            cwd=tmp_path,
            input=payload,
            capture_output=True,
            check=True,
        ).stdout

    git("init", "--quiet")
    attributes = Path(__file__).resolve().parents[1] / ".gitattributes"
    contents = {".gitattributes": attributes.read_bytes()}
    expected = {}
    for resource, digest in SOURCES.items():
        package = "app" if resource.startswith("model_cards/") else "inference"
        path = f"src/factoryvision/{package}/{resource}"
        contents[path] = source_bytes(resource)
        expected[path] = digest
    # Write raw blobs directly: git add would apply clean filters and could hide
    # an unwanted normalization of the historical CRLF resources.
    for path, raw in contents.items():
        oid = git("hash-object", "-w", "--stdin", payload=raw).decode().strip()
        git("update-index", "--add", "--cacheinfo", f"100644,{oid},{path}")
    git("checkout-index", "--all")
    for path, digest in expected.items():
        checked_out = (tmp_path / path).read_bytes()
        assert checked_out == contents[path], path
        assert hashlib.sha256(checked_out).hexdigest() == digest, path
        attrs = git("check-attr", "text", "eol", "--", path).decode()
        if b"\r\n" in contents[path]:
            assert ": text: unset" in attrs and ": eol: unspecified" in attrs
        else:
            assert ": text: set" in attrs and ": eol: lf" in attrs
    # Clean filters must preserve the blobs as well as checkout/smudge filters.
    git("add", "--", *expected)
    for path in expected:
        assert git("show", f":{path}") == contents[path], path
