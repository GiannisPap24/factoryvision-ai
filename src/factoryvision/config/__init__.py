"""Small validated settings; no environment-variable or global configuration reads."""

from dataclasses import dataclass
from pathlib import Path

from factoryvision.domain.errors import reject


@dataclass(frozen=True, slots=True)
class ValidationSettings:
    dataset_root: Path | None = None
    max_manifest_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        if type(self.max_manifest_bytes) is not int or self.max_manifest_bytes <= 0:
            reject("CONFIG_INVALID", "max_manifest_bytes", "Use a positive integer byte limit.")
        if self.dataset_root is not None and (
            not isinstance(self.dataset_root, Path) or not self.dataset_root.is_absolute()
        ):
            reject("CONFIG_INVALID", "dataset_root", "Use an absolute Path or explicit null.")
