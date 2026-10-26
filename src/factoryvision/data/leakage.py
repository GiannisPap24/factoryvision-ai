"""Heuristic near-duplicate screening; fingerprints do not identify physical items."""

import numpy as np
from PIL import Image


def fingerprints(image: Image.Image) -> tuple[int, int, np.ndarray]:
    """64-bit horizontal dHash, 63 AC-bit DCT pHash, and 32x32 RGB thumbnail."""
    gray = image.convert("L")
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float64)
    difference = small[:, 1:] > small[:, :-1]
    values = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    indices = np.arange(32)
    basis = np.cos(np.pi * (2 * indices[None, :] + 1) * indices[:, None] / 64)
    coefficients = (basis @ values @ basis.T)[:8, :8].ravel()[1:]
    perceptual = coefficients > np.median(coefficients)

    def bits(array):
        return sum(int(bit) << i for i, bit in enumerate(array.ravel()))

    rgb = np.asarray(image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS))
    return bits(difference), bits(perceptual), rgb.astype(np.float32) / 255


def screen_pairs(records: list[dict]) -> dict:
    """Flag cross-partition dHash<=4 OR pHash<=4 OR RGB RMS<=0.02; no exclusions."""
    candidates = []
    comparisons = 0
    for i, a in enumerate(records):
        for b in records[i + 1 :]:
            if a["partition"] == b["partition"]:
                continue
            comparisons += 1
            dh = (a["dhash"] ^ b["dhash"]).bit_count()
            ph = (a["phash"] ^ b["phash"]).bit_count()
            rms = float(np.sqrt(np.mean((a["rgb"] - b["rgb"]) ** 2)))
            if dh <= 4 or ph <= 4 or rms <= 0.02:
                candidates.append(
                    {
                        "a": a["id"],
                        "b": b["id"],
                        "partitions": [a["partition"], b["partition"]],
                        "dhash_distance": dh,
                        "phash_distance": ph,
                        "rgb_rms": rms,
                    }
                )
    candidates.sort(key=lambda c: (c["rgb_rms"], c["a"], c["b"]))
    return {
        "cross_partition_comparisons": comparisons,
        "candidates": candidates,
        "rule": "dHash64 <= 4 OR pHash63 <= 4 OR RGB32 RMS <= 0.02",
        "limitations": "Heuristic, alignment sensitive, no proof of physical-item independence.",
    }
