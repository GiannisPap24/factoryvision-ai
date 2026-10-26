"""Opt-in filesystem verification; never decodes or reads image content."""

from pathlib import Path

from factoryvision.domain.contracts import Manifest
from factoryvision.domain.errors import reject
from factoryvision.domain.paths import check_relative_path
from factoryvision.domain.validation import require_valid_manifest


def resolve_reference(root: Path, reference: str) -> Path:
    """Resolve an existing file under root, rejecting symlink/junction escapes.

    This is a validation-time check, not an atomic secure-open primitive. Do not
    use it to authorize later reads from a concurrently untrusted directory.
    """
    check_relative_path(reference, "reference")
    try:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            reject("ROOT_INVALID", "dataset_root", "Dataset root must be an existing directory.")
        candidate = (resolved_root / reference).resolve()
        if not candidate.is_relative_to(resolved_root):
            reject("PATH_ESCAPE", "reference", "Resolved reference escapes the dataset root.")
        if not candidate.is_file():
            reject("FILE_MISSING", "reference", f"Expected an existing file: {reference}")
    except (OSError, RuntimeError) as exc:
        reject(
            "PATH_RESOLUTION", "reference", f"Cannot resolve reference safely: {type(exc).__name__}"
        )
    return candidate


def validate_files(manifest: Manifest, root: Path) -> tuple[Path, ...]:
    require_valid_manifest(manifest)
    return tuple(
        resolve_reference(root, reference)
        for sample in sorted(manifest.samples, key=lambda item: item.sample_id)
        for reference in (sample.image_path, sample.mask_path)
        if reference is not None
    )
