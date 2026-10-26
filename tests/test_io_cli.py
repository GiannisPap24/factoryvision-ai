"""Offline CLI, strict JSON parsing and immutable serialization."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from factoryvision.cli import main
from factoryvision.config import ValidationSettings
from factoryvision.data.manifest import load_manifest, manifest_from_dict, manifest_to_dict
from factoryvision.domain.errors import ContractError

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic-manifest.json"


def test_roundtrip(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_to_dict(manifest)), encoding="utf-8")
    output = load_manifest(path)
    assert manifest_to_dict(output) == manifest_to_dict(manifest)


@pytest.mark.parametrize(
    ("content", "code"),
    [
        (b"{", "JSON_INVALID"),
        (b'{"schema_version": 1, "schema_version": 2}', "JSON_DUPLICATE_KEY"),
        (b'{"x": NaN}', "JSON_INVALID"),
        (b'{"x": Infinity}', "JSON_INVALID"),
        (b"\xff", "JSON_INVALID"),
        (b'{"schema_version":' + b"9" * 5000 + b"}", "JSON_INVALID"),
        (b"[" * 1500 + b"]" * 1500, "TYPE_INVALID"),
    ],
)
def test_invalid_json_or_root_type(tmp_path, content, code):
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(ContractError) as exc:
        load_manifest(path)
    assert {issue.code for issue in exc.value.issues} == {code}


def test_manifest_size_and_missing_file(tmp_path):
    path = tmp_path / "big.json"
    path.write_text("{} ", encoding="utf-8")
    with pytest.raises(ContractError, match="MANIFEST_TOO_LARGE"):
        load_manifest(path, ValidationSettings(max_manifest_bytes=2))
    with pytest.raises(ContractError, match="MANIFEST_IO"):
        load_manifest(tmp_path / "absent.json")


@pytest.mark.parametrize("value", [None, [], "bad", 1])
def test_non_object_json(value):
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        manifest_from_dict(value)


def test_invalid_record_object(payload):
    payload["samples"][0] = "not-an-object"
    with pytest.raises(ContractError, match="TYPE_INVALID"):
        manifest_from_dict(payload)


def test_cli_valid_text_and_json(capsys):
    assert main(["validate-manifest", str(FIXTURE)]) == 0
    assert "Valid manifest: 12 samples; files_checked=False" in capsys.readouterr().out
    assert main(["validate-manifest", str(FIXTURE), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["files_checked"] is False


def test_cli_invalid_text_json_and_file_checks(tmp_path, capsys):
    assert main(["validate-manifest", str(tmp_path / "missing")]) == 2
    assert "MANIFEST_IO" in capsys.readouterr().out
    assert main(["validate-manifest", str(tmp_path / "missing"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False
    assert main(["validate-manifest", str(FIXTURE), "--dataset-root", str(tmp_path)]) == 2
    assert "FILE_MISSING" in capsys.readouterr().out


def test_cli_valid_files(tmp_path, manifest, capsys):
    for sample in manifest.samples:
        for reference in (sample.image_path, sample.mask_path):
            if reference is not None:
                target = tmp_path / reference
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"synthetic placeholder")
    assert main(["validate-manifest", str(FIXTURE), "--dataset-root", str(tmp_path)]) == 0
    assert "files_checked=True" in capsys.readouterr().out


def test_real_module_entrypoint():
    process = subprocess.run(
        [sys.executable, "-m", "factoryvision", "validate-manifest", str(FIXTURE), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["valid"] is True


def test_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
