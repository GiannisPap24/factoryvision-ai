"""Adversarial distribution checks use only fabricated content."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "distribution_check", Path(__file__).parents[1] / "scripts/verify_distribution.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def archive_at(tmp_path, members, kind):
    path = tmp_path / ("synthetic.whl" if kind == "wheel" else "synthetic.tar.gz")
    if kind == "wheel":
        with zipfile.ZipFile(path, "w") as package:
            for name, data in members.items():
                package.writestr(name, data)
    else:
        with tarfile.open(path, "w:gz") as package:
            for name, data in members.items():
                info = tarfile.TarInfo("synthetic/" + name)
                info.size = len(data)
                package.addfile(info, io.BytesIO(data))
    return path


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("escaped", [False, True])
def test_private_paths_rejected_without_value_disclosure(tmp_path, kind, escaped):
    value = "C:" + "\\".join(("", "Users", "fabricated-private-user", "work"))
    data = value.replace("\\", "\\\\") if escaped else value
    result = module.inspect_distribution(
        archive_at(tmp_path, {"evidence.json": data.encode()}, kind)
    )
    assert result["findings"] == [{"member": "evidence.json", "line": 1, "rule": "private-path"}]
    assert "fabricated-private-user" not in str(result)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_forbidden_artifacts_secret_and_broken_link(tmp_path, kind):
    result = module.inspect_distribution(
        archive_at(
            tmp_path,
            {
                "AGENTS.md": b"private instructions",
                "runs/model.pt": b"not a real model",
                "README.md": b"[Missing](missing.md)",
                "token.txt": ("ghp" + "_" + "A" * 25).encode(),
            },
            kind,
        )
    )
    assert {r["rule"] for r in result["findings"]} == {
        "forbidden-member",
        "missing-doc-link",
        "secret-pattern",
    }


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_portable_documentation_and_resolved_links(tmp_path, kind):
    result = module.inspect_distribution(
        archive_at(
            tmp_path,
            {"README.md": b"${USERPROFILE} [Guide](docs/guide.md)", "docs/guide.md": b"Synthetic"},
            kind,
        )
    )
    assert result["findings"] == []
