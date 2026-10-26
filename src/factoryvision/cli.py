"""Offline validation command; no implicit network access or output files."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from factoryvision.config import ValidationSettings
from factoryvision.data.filesystem import validate_files
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.errors import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="factoryvision")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser(
        "validate-manifest", help="Validate a metadata manifest offline."
    )
    validate.add_argument("manifest", type=Path)
    validate.add_argument(
        "--dataset-root", type=Path, help="Also resolve existing image/mask files."
    )
    validate.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    try:
        settings = ValidationSettings(
            dataset_root=args.dataset_root.absolute() if args.dataset_root is not None else None
        )
        manifest = load_manifest(args.manifest, settings)
        if settings.dataset_root is not None:
            validate_files(manifest, settings.dataset_root)
        result = {
            "valid": True,
            "samples": len(manifest.samples),
            "manifest_sha256": manifest_sha256(manifest),
            "files_checked": settings.dataset_root is not None,
        }
        exit_code = 0
    except ContractError as exc:
        result = {"valid": False, "errors": [asdict(issue) for issue in exc.issues]}
        exit_code = 2
    if args.json_output:
        print(json.dumps(result, sort_keys=True))
    elif exit_code == 0:
        print(
            f"Valid manifest: {result['samples']} samples; files_checked={result['files_checked']}"
        )
    else:
        for issue in result["errors"]:
            print(f"{issue['code']} [{issue['field']}]: {issue['message']}")
    return exit_code
