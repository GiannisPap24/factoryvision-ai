"""A reproducible debugger target using fabricated metadata only."""

from pathlib import Path

from factoryvision.data.manifest import load_manifest
from factoryvision.domain.validation import validate_manifest

fixture_path = Path(__file__).resolve().parents[1] / "tests/fixtures/synthetic-manifest.json"
manifest = load_manifest(fixture_path)
issues = validate_manifest(manifest)  # Set the verification breakpoint here.
print(f"Validated {len(manifest.samples)} synthetic records; issues={len(issues)}")
