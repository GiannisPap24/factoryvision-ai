"""Normal-source-only MetalParts development; never load the eight-domain adapter here."""

import hashlib
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from factoryvision.data.filesystem import resolve_reference
from factoryvision.data.images import decode_image
from factoryvision.data.metalparts import MetalPartsSample, MetalPartsSource, _unique_object
from factoryvision.domain.contracts import Label, OfficialSplit, Partition, require_sha256
from factoryvision.domain.errors import reject

TRAIN_METADATA = "MetalParts/metal_parts_data_dir_train/metadata.jsonl"


@dataclass(frozen=True, slots=True)
class DevelopmentSample:
    sample: MetalPartsSample
    content_sha256: str
    pixel_sha256: str
    partition: Partition = Partition.TRAIN

    def __post_init__(self):
        if (
            not isinstance(self.sample, MetalPartsSample)
            or self.sample.official_split != OfficialSplit.TRAIN
            or self.sample.label != Label.NORMAL
            or self.sample.mask_path is not None
            or self.partition not in (Partition.TRAIN, Partition.VALIDATION)
            or not isinstance(self.partition, Partition)
        ):
            reject(
                "DEVELOPMENT_ONLY", "sample", "Require source-train normals; no calibration/test."
            )
        require_sha256(self.content_sha256, "content_sha256")
        require_sha256(self.pixel_sha256, "pixel_sha256")

    @property
    def image_path(self):
        return self.sample.image_path

    @property
    def sample_id(self):
        return self.sample.sample_id

    @property
    def label(self):
        return self.sample.label

    @property
    def mask_path(self):
        return self.sample.mask_path

    @property
    def official_split(self):
        return self.sample.official_split


def verified_bytes(root: Path, reference: str, expected: dict) -> bytes:
    path = resolve_reference(root, reference)
    if path.stat().st_size > 32 * 1024**2 or path.stat().st_size != expected["bytes"]:
        reject("SOURCE_HASH_MISMATCH", reference, "Size differs from verified source inventory.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected["sha256"]:
        reject("SOURCE_HASH_MISMATCH", reference, "Bytes differ from verified source inventory.")
    return content


def load_training_normals(root: Path, verified: dict, source: MetalPartsSource):
    """Read train JSONL and label-0 images only; label-1 references are checked, never decoded."""
    raw = verified_bytes(root, TRAIN_METADATA, verified[TRAIN_METADATA])
    try:
        rows = [
            json.loads(line, object_pairs_hook=_unique_object) for line in raw.decode().splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError):
        reject("METADATA_INVALID", TRAIN_METADATA, "Malformed training JSONL.")
    samples, excluded, seen = [], [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"file_name", "label", "mask"}:
            reject("METADATA_FIELDS", TRAIN_METADATA, "Require exactly file_name, label, mask.")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            reject("LABEL_INVALID", TRAIN_METADATA, "Require integer 0 or 1.")
        label = Label.NORMAL if row["label"] == 0 else Label.ANOMALY
        sample = MetalPartsSample(source, "train", row["file_name"], label, row["mask"], 1, 1)
        if sample.image_path.casefold() in seen:
            reject("SAMPLE_DUPLICATE", sample.image_path, "Duplicate full path, ignoring case.")
        seen.add(sample.image_path.casefold())
        if label == Label.ANOMALY:
            excluded.append(
                {
                    "sample_id": sample.sample_id,
                    "path": sample.image_path,
                    "reason": "label-1 source anomaly",
                }
            )
            continue
        if sample.image_path not in verified:
            reject("SOURCE_HASH_MISMATCH", sample.image_path, "Missing verified source identity.")
        content = verified_bytes(root, sample.image_path, verified[sample.image_path])
        image = decode_image(root, sample.image_path)
        sample = replace(sample, width=image.width, height=image.height)
        pixels = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        samples.append(DevelopmentSample(sample, hashlib.sha256(content).hexdigest(), pixels))
    if not samples:
        reject("SAMPLES_INVALID", TRAIN_METADATA, "No eligible normal training images.")
    return tuple(sorted(samples, key=lambda s: s.sample_id)), excluded


def split_normals(samples: tuple[DevelopmentSample, ...], seed: int = 42, related_pairs=()):
    """80/20 groups: exact bytes/pixels plus explicitly confirmed relations, never heuristics."""
    if type(seed) is not int or not 0 <= seed < 2**32 or len(samples) < 2:
        raise ValueError("Require a valid seed and at least two normal samples.")
    ids = {s.sample_id: s for s in samples}
    if len(ids) != len(samples) or len({s.sample.source for s in samples}) != 1:
        raise ValueError("Duplicate IDs or mixed source identities.")
    parent = {key: key for key in ids}

    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key

    def join(a, b):
        if a not in ids or b not in ids:
            raise ValueError("Confirmed relationship references unknown sample.")
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for field in ("content_sha256", "pixel_sha256"):
        seen = {}
        for sample in samples:
            value = getattr(sample, field)
            if value in seen:
                join(sample.sample_id, seen[value])
            seen[value] = sample.sample_id
    for a, b in related_pairs:
        join(a, b)
    groups = sorted(
        {find(key) for key in ids}, key=lambda g: hashlib.sha256(f"{seed}|{g}".encode()).hexdigest()
    )
    if len(groups) < 2:
        raise ValueError("Cannot separate development groups.")
    validation = set(groups[: max(1, (len(groups) + 4) // 5)])
    result = tuple(
        replace(
            s,
            partition=Partition.VALIDATION if find(s.sample_id) in validation else Partition.TRAIN,
        )
        for s in sorted(samples, key=lambda s: s.sample_id)
    )
    return result, {key: find(key) for key in sorted(ids)}


@dataclass(frozen=True, slots=True)
class PatchConfig:
    size: int = 256
    stride: int = 192
    limit: int = 12

    def __post_init__(self):
        if (
            type(self.size) is not int
            or self.size not in (32, 64, 128, 256)
            or type(self.stride) is not int
            or not 1 <= self.stride <= self.size
            or type(self.limit) is not int
            or not 1 <= self.limit <= 128
        ):
            raise ValueError("Invalid bounded native-patch configuration.")


def coordinates(width: int, height: int, config: PatchConfig, *, full=False):
    if any(type(n) is not int or n < config.size or n > 10000 for n in (width, height)):
        raise ValueError("Native dimensions must fit an unresized patch and stay bounded.")

    def axis(length):
        positions = list(range(0, length - config.size + 1, config.stride))
        if positions[-1] != length - config.size:
            positions.append(length - config.size)
        return positions

    xs, ys = axis(width), axis(height)
    if len(xs) * len(ys) > 4096:
        raise ValueError("Too many patches; increase stride.")
    grid = tuple((x, y) for y in ys for x in xs)
    if full or len(grid) <= config.limit:
        return grid
    return tuple(grid[i] for i in np.linspace(0, len(grid) - 1, config.limit, dtype=int))


DEFAULT_PATCH_CONFIG = PatchConfig()


class NativePatchDataset(Dataset):
    """Two-parent RGB cache, native crops, stable identities; no masks or rescaling."""

    def __init__(self, root, samples, partition, config=DEFAULT_PATCH_CONFIG):
        if partition not in (Partition.TRAIN, Partition.VALIDATION) or not isinstance(
            partition, Partition
        ):
            raise ValueError("Only fitting or validation is allowed.")
        self.root, self.config, self.partition = root, config, partition
        self.samples = tuple(
            sorted((s for s in samples if s.partition == partition), key=lambda s: s.sample_id)
        )
        if not self.samples or len({s.sample_id for s in self.samples}) != len(self.samples):
            raise ValueError("Require unique nonempty development membership.")
        self.patches = tuple(
            (s, x, y)
            for s in self.samples
            for x, y in coordinates(s.sample.width, s.sample.height, config)
        )
        self._image = lru_cache(maxsize=2)(self._decode)

    def _decode(self, sample):
        verified_bytes(
            self.root,
            sample.image_path,
            {
                "bytes": resolve_reference(self.root, sample.image_path).stat().st_size,
                "sha256": sample.content_sha256,
            },
        )
        image = decode_image(self.root, sample.image_path)
        if image.size != (sample.sample.width, sample.sample.height):
            raise ValueError("Native geometry changed.")
        return image

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        sample, x, y = self.patches[index]
        crop = self._image(sample).crop((x, y, x + self.config.size, y + self.config.size))
        tensor = torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float().div(255)
        return {
            "image": tensor,
            "label": 0,
            "partition": self.partition.value,
            "sample_id": f"{sample.sample_id}-y{y:05d}-x{x:05d}",
            "parent_id": sample.sample_id,
            "xy": torch.tensor([x, y]),
        }

    def shuffled_indices(self, seed):
        generator = torch.Generator().manual_seed(seed)
        groups = {}
        for i, (sample, _, _) in enumerate(self.patches):
            groups.setdefault(sample.sample_id, []).append(i)
        keys = list(groups)
        return [
            groups[keys[k]][j]
            for k in torch.randperm(len(keys), generator=generator).tolist()
            for j in torch.randperm(len(groups[keys[k]]), generator=generator).tolist()
        ]
