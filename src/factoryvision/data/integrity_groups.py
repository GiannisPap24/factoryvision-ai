"""Bounded, model-free integrity evidence; never removes or reassigns samples.

Post-incident utility, separate from the unchanged preregistered PCB1 screen.
Perceptual proximity proposes review pairs, never establishes object identity.
"""

import hashlib
import io
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image

from factoryvision.data.leakage import fingerprints

PARTITIONS = frozenset(("train", "validation", "calibration", "test"))
MAX_RECORDS = 2000


@dataclass(frozen=True)
class ImageFingerprint:
    sample_id: str
    partition: str
    byte_sha256: str
    pixel_sha256: str
    dhash: int
    phash: int

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("Nonempty sample identity required.")
        if self.partition not in PARTITIONS:
            raise ValueError("Unknown partition.")
        for digest in (self.byte_sha256, self.pixel_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid SHA-256 identity.")
        for value, bits in ((self.dhash, 64), (self.phash, 63)):
            if type(value) is not int or not 0 <= value < 2**bits:
                raise ValueError("Invalid perceptual hash.")


def fingerprint_bytes(sample_id: str, partition: str, payload: bytes) -> ImageFingerprint:
    """Decode one bounded image, discard pixels afterward; no masks or labels accepted."""
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Image byte limit exceeded.")
    with Image.open(io.BytesIO(payload)) as original:
        if original.width * original.height > 16_000_000:
            raise ValueError("Image pixel limit exceeded.")
        image = original.convert("RGB")
        dhash, phash, _ = fingerprints(image)
        pixels = f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
    return ImageFingerprint(
        sample_id,
        partition,
        hashlib.sha256(payload).hexdigest(),
        hashlib.sha256(pixels).hexdigest(),
        dhash,
        phash,
    )


def _records(records):
    if len(records) > MAX_RECORDS:
        raise ValueError("Record limit exceeded.")
    ordered = sorted(records, key=lambda r: r.sample_id)
    if len({r.sample_id for r in ordered}) != len(ordered):
        raise ValueError("Duplicate sample identity.")
    return ordered


def perceptual_pairs(records, *, max_pairs=1_000_000):
    """Cross-partition dHash<=4 OR pHash<=4 via five-band candidate indexing.

    <=4 changed bits must leave one of five disjoint bands unchanged. This is
    lossless for the stated Hamming rule, not for arbitrary image transforms.
    Worst-case comparisons remain quadratic, with an explicit fail-closed cap.
    The original RGB32 RMS rule is intentionally not claimed by this utility.
    """
    ordered = _records(records)
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("Positive comparison cap required.")
    buckets, seen, pairs = defaultdict(list), set(), []
    for i, record in enumerate(ordered):
        candidates = set()
        keys = []
        for field in ("dhash", "phash"):
            value = getattr(record, field)
            for band, start in enumerate((0, 13, 26, 39, 52)):
                key = (field, band, (value >> start) & 8191)
                candidates.update(buckets[key])
                keys.append(key)
        for j in sorted(candidates):
            other = ordered[j]
            if other.partition == record.partition:
                continue
            seen.add((j, i))
            if len(seen) > max_pairs:
                raise ValueError("Comparison cap exceeded; audit incomplete.")
            dh = (other.dhash ^ record.dhash).bit_count()
            ph = (other.phash ^ record.phash).bit_count()
            if dh <= 4 or ph <= 4:
                pairs.append(
                    {
                        "a": other.sample_id,
                        "b": record.sample_id,
                        "dhash_distance": dh,
                        "phash_distance": ph,
                    }
                )
        for key in keys:
            buckets[key].append(i)
    return {
        "comparisons": len(seen),
        "pairs": sorted(pairs, key=lambda p: (p["a"], p["b"])),
        "rule": "dHash64<=4 OR pHash63<=4; cross-partition; review required",
    }


def group_evidence(records, *, confirmed_pairs=(), unresolved_pairs=()):
    """Group exact/explicitly confirmed edges only; keep uncertain edges separate."""
    ordered = _records(records)
    by_id = {r.sample_id: r for r in ordered}
    parent = {key: key for key in by_id}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a, b):
        left, right = sorted((find(a), find(b)))
        parent[right] = left

    def checked_pairs(pairs):
        result = set()
        for a, b in pairs:
            if a == b or a not in by_id or b not in by_id:
                raise ValueError("Unknown or self-related sample identity.")
            result.add(tuple(sorted((a, b))))
        return sorted(result)

    confirmed = checked_pairs(confirmed_pairs)
    unresolved = checked_pairs(unresolved_pairs)
    if set(confirmed) & set(unresolved):
        raise ValueError("Conflicting review decisions.")
    exact = {}
    for field in ("byte_sha256", "pixel_sha256"):
        buckets = defaultdict(list)
        for record in ordered:
            buckets[getattr(record, field)].append(record.sample_id)
        groups = sorted(group for group in buckets.values() if len(group) > 1)
        exact[field] = groups
        for group in groups:
            for member in group[1:]:
                union(group[0], member)
    for a, b in confirmed:
        union(a, b)
    groups = defaultdict(list)
    for key in sorted(by_id):
        groups[find(key)].append(key)
    related = [
        {"members": members, "partitions": sorted({by_id[k].partition for k in members})}
        for members in sorted(groups.values())
        if len(members) > 1
    ]
    cross = [group for group in related if len(group["partitions"]) > 1]
    unresolved_cross = [[a, b] for a, b in unresolved if by_id[a].partition != by_id[b].partition]
    return {
        "exact_groups": exact,
        "confirmed_pairs": [list(p) for p in confirmed],
        "related_groups": related,
        "cross_partition_groups": cross,
        "affected_confirmed_samples": sorted({k for g in cross for k in g["members"]}),
        "unresolved_cross_partition_pairs": unresolved_cross,
        "physical_item_independence": "not established by image fingerprints",
    }


def reject_cross_split(report):
    """Mandatory explicit rejection, without modifying evidence or sample membership."""
    if report["cross_partition_groups"] or report["unresolved_cross_partition_pairs"]:
        raise ValueError("Cross-split overlap or unresolved review; fitting/evaluation prohibited.")
