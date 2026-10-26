"""Create one exclusive portable pre-test seal after accepted normal-only verification."""

import json
from datetime import UTC, datetime
from pathlib import Path

from factoryvision.data.category import normal_members
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.models.pcb1 import FAMILIES, PROTOCOL_SHA
from factoryvision.training.checkpoint import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = json.loads((ROOT / "reports/pcb1-verification.json").read_text())
    if not all(
        evidence[k] is True
        for k in (
            "offline_quality",
            "normal_checks",
            "fresh_wheel",
            "vscode_debug",
            "historical_artifacts_unchanged",
        )
    ):
        raise ValueError("Pre-test verification incomplete.")
    if (ROOT / "reports/pcb1-opening").exists() or (ROOT / "data/pcb1-test-quarantine").exists():
        raise ValueError("Official test was already opened.")
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    if file_sha256(ROOT / "docs/protocols/pcb1-confirmatory-v3.json") != PROTOCOL_SHA:
        raise ValueError("Methodology changed.")
    paths = [
        *ROOT.glob("src/**/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("docs/protocols/*"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        ROOT / ".vscode/tasks.json",
        ROOT / ".vscode/launch.json",
        ROOT / "data/raw/VisA_20220922.tar",
        ROOT / "data/raw/acquisition.json",
        ROOT / "data/metadata/1cls.csv",
        ROOT / "data/metadata/pcb1-preparation.json",
        ROOT / "data/manifests/pcb1-official.json",
        ROOT / "data/manifests/pcb1-seed42.json",
        ROOT / "data/pretrained/resnet18-f37072fd.pth",
        ROOT / "reports/pcb1-normal-screen.json",
        ROOT / "reports/pcb1-normal-review.json",
        ROOT / "reports/pcb1-verification.json",
    ]
    paths += [ROOT / "data/pcb1-development" / s.image_path for s in normal_members(manifest)]
    models = {}
    for family in FAMILIES:
        directory = ROOT / "runs" / f"pcb1-{family.replace('_', '-')}-v1"
        development = json.loads((directory / "development.json").read_text())
        calibration = json.loads((directory / "calibration.json").read_text())
        if (
            development["heldout_payloads_accessed"]
            or development["reload_score_max_abs"] > 1e-6
            or development["reload_map_max_abs"] > 1e-6
        ):
            raise ValueError("Reload/blinding gate failed.")
        models[family] = dict(
            directory=directory.relative_to(ROOT).as_posix(),
            checkpoint_sha256=file_sha256(directory / "best.pt"),
            threshold=calibration["threshold"],
            calibration_sha256=file_sha256(directory / "calibration.json"),
            display_map_max=development["display_map_max"],
        )
        paths += [p for p in directory.iterdir() if p.is_file()]
    seal = dict(
        schema=1,
        category="pcb1",
        status="READY FOR BLINDED PCB1 TEST",
        sealed_at_utc=datetime.now(UTC).isoformat(),
        checkpoint="960b76d57c834c0b8f73d19538c5a7a013c92f26",
        protocol_sha256=PROTOCOL_SHA,
        manifest_sha256=manifest_sha256(manifest),
        membership={
            p: [s.sample_id for s in manifest.samples if s.partition.value == p]
            for p in ("train", "validation", "calibration", "test")
        },
        models=models,
        files={p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(set(paths))},
        owner_authorization="NOT GRANTED FOR 2E2; separate explicit approval required",
        official_test_payloads_accessed=0,
        no_postseal_fit_or_threshold_changes=True,
    )
    path = ROOT / "docs/evidence/pcb1-evaluation-seal.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(seal, stream, indent=2, allow_nan=False)
        stream.write("\n")
    sha = file_sha256(path)
    with path.with_suffix(".sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{sha}  pcb1-evaluation-seal.json\n")
    print(json.dumps(dict(seal_sha256=sha, files=len(seal["files"])), indent=2))


if __name__ == "__main__":
    main()
