"""Portable reference validation, independent of filesystem access."""

from pathlib import PureWindowsPath

from factoryvision.domain.errors import reject


def check_relative_path(value: str, field: str) -> None:
    """Require unambiguous POSIX-style relative references, safe on Windows too."""
    if not isinstance(value, str) or not value:
        reject("PATH_INVALID", field, "Provide a nonempty relative path string.")
    if value.startswith("/") or PureWindowsPath(value).drive or "\\" in value:
        reject("PATH_UNSAFE", field, "Use a relative path with forward slashes; no drive or UNC.")
    for part in value.split("/"):
        if (
            part in ("", ".", "..")
            or part != part.strip()
            or part.endswith(".")
            or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
            or PureWindowsPath(part).is_reserved()
        ):
            reject(
                "PATH_UNSAFE", field, "Remove traversal, empty components or Windows-unsafe names."
            )
