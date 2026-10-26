"""Bounded RGB/mask decoding and deterministic CPU datasets; no augmentation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from factoryvision.data.filesystem import resolve_reference
from factoryvision.domain.contracts import Label, Manifest, Partition
from factoryvision.domain.errors import reject
from factoryvision.domain.validation import require_valid_manifest

MAX_FILE_BYTES = 32 * 1024**2
MAX_PIXELS = 20_000_000


@dataclass(frozen=True, slots=True)
class ImageSettings:
    height: int = 256
    width: int = 256

    def __post_init__(self) -> None:
        if any(type(n) is not int or not 1 <= n <= 2048 for n in (self.height, self.width)):
            reject("IMAGE_CONFIG", "resolution", "Use integer dimensions in [1, 2048].")


def decode_image(root: Path, reference: str, *, mask: bool = False) -> Image.Image:
    """Decode stored pixel geometry, without EXIF rotation; masks retain class IDs."""
    path = resolve_reference(root, reference)
    if path.stat().st_size > MAX_FILE_BYTES:
        reject("IMAGE_TOO_LARGE", reference, "Encoded file exceeds 32 MiB.")
    try:
        with Image.open(path) as source:
            if source.format not in ({"PNG"} if mask else {"JPEG", "PNG"}):
                reject("IMAGE_FORMAT", reference, "Use JPEG/PNG images and PNG masks.")
            if source.width * source.height > MAX_PIXELS or getattr(source, "n_frames", 1) != 1:
                reject("IMAGE_TOO_LARGE", reference, "Require one frame and at most 20M pixels.")
            if source.getexif().get(274, 1) != 1:
                reject("IMAGE_ORIENTATION", reference, "Review nontrivial EXIF orientation.")
            source.load()
            if mask:
                if source.mode not in ("1", "L", "P", "I", "I;16"):
                    reject("MASK_MODE", reference, "Require a single-channel class-index mask.")
                return source.copy()
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        reject("IMAGE_DECODE", reference, f"Image decoding failed: {type(exc).__name__}")


def decode_pair(root: Path, sample) -> tuple[Image.Image, Image.Image]:
    image = decode_image(root, sample.image_path)
    if sample.mask_path is None:
        # A declared normal has no stored mask; this zero target is derived, not an annotation.
        mask = Image.new("L", image.size, 0)
    else:
        original = decode_image(root, sample.mask_path, mask=True)
        if original.size != image.size:
            reject("MASK_DIMENSIONS", sample.mask_path, "Mask must match stored image geometry.")
        binary = np.asarray(original) > 0
        if not binary.any():
            reject(
                "MASK_EMPTY", sample.mask_path, "Anomaly annotation contains no positive pixels."
            )
        mask = Image.fromarray(binary.astype(np.uint8) * 255)
    return image, mask


class ManifestDataset(Dataset):
    """A single explicit partition in stable sample-ID order, decoded on demand."""

    def __init__(
        self,
        manifest: Manifest,
        root: Path,
        partition: Partition,
        settings: ImageSettings | None = None,
    ) -> None:
        require_valid_manifest(manifest)
        if not isinstance(partition, Partition):
            reject("ENUM_INVALID", "partition", "Use an explicit Partition value.")
        self.settings = settings if settings is not None else ImageSettings()
        if not isinstance(self.settings, ImageSettings):
            reject("IMAGE_CONFIG", "settings", "Use ImageSettings.")
        self.root = root
        self.samples = tuple(
            sorted(
                (s for s in manifest.samples if s.partition == partition), key=lambda s: s.sample_id
            )
        )
        if not self.samples:
            reject("PARTITION_EMPTY", "partition", "Selected partition has no records.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image, mask = decode_pair(self.root, sample)
        original_size = torch.tensor([image.height, image.width], dtype=torch.int64)
        size = [self.settings.height, self.settings.width]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return {
            "image": TF.pil_to_tensor(image).to(dtype=torch.float32).div(255),
            "mask": TF.pil_to_tensor(mask).to(dtype=torch.float32).div(255),
            "label": int(sample.label == Label.ANOMALY),
            "sample_id": sample.sample_id,
            "partition": sample.partition.value,
            "mask_available": sample.mask_path is not None,
            "original_size": original_size,
        }


def make_loader(dataset: ManifestDataset, batch_size: int = 4, seed: int = 42) -> DataLoader:
    """No shuffle, workers, pinned memory or prefetch; bounded CPU batches."""
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        reject("LOADER_CONFIG", "batch_size", "Use an integer batch size in [1, 64].")
    if type(seed) is not int or not 0 <= seed < 2**32:
        reject("SEED_INVALID", "seed", "Use an integer in [0, 2**32).")
    if batch_size * 4 * dataset.settings.height * dataset.settings.width * 4 > 512 * 1024**2:
        reject("LOADER_CONFIG", "batch_size", "Image/mask batch would exceed 512 MiB.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
