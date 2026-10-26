"""Predeclared normal-only similarity refinement; screening cannot prove independence."""

import hashlib

import numpy as np
from PIL import Image

from factoryvision.data.category import normal_members
from factoryvision.data.images import decode_image
from factoryvision.data.leakage import fingerprints, screen_pairs


def review_normals(manifest, root):
    return review_samples(normal_members(manifest), lambda s: decode_image(root, s.image_path))


def review_samples(samples, decode):
    """Shared frozen screen; caller explicitly controls which payloads may be decoded."""
    cache, records = {}, []
    for sample in samples:
        image = decode(sample)
        dh, ph, rgb = fingerprints(image)
        records.append(
            dict(id=sample.sample_id, partition=sample.partition.value, dhash=dh, phash=ph, rgb=rgb)
        )
        full = (
            np.asarray(image.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        )
        box = (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4)
        crop = image.crop(box).resize((64, 64), Image.Resampling.LANCZOS)
        gray = np.asarray(crop.convert("L"), dtype=np.float64).ravel()
        gray -= gray.mean()
        cache[sample.sample_id] = full, np.asarray(crop, dtype=np.float32) / 255, gray
    screen = screen_pairs(records)
    refined = []
    for candidate in screen["candidates"]:
        af, ac, ag = cache[candidate["a"]]
        bf, bc, bg = cache[candidate["b"]]
        full = float(np.sqrt(np.mean((af - bf) ** 2)))
        crop = float(np.sqrt(np.mean((ac - bc) ** 2)))
        denominator = float(np.linalg.norm(ag) * np.linalg.norm(bg))
        corr = float(np.dot(ag, bg) / denominator) if denominator else None
        refined.append(
            {
                **candidate,
                "full128_rms": full,
                "crop64_rms": crop,
                "crop_correlation": corr,
                "strong": (full <= 0.015 and crop <= 0.025) or (corr is not None and corr >= 0.995),
            }
        )
    selected = (
        sorted(refined, key=lambda c: (c["full128_rms"], c["a"], c["b"]))[:8]
        + sorted(refined, key=lambda c: (c["crop64_rms"], c["a"], c["b"]))[:8]
        + sorted(refined, key=lambda c: hashlib.sha256(f"{c['a']}|{c['b']}".encode()).hexdigest())[
            :8
        ]
        + [c for c in refined if c["strong"]]
    )
    return {
        "cross_partition_comparisons": screen["cross_partition_comparisons"],
        "candidate_count": len(refined),
        "strong_count": sum(c["strong"] for c in refined),
        "selected_for_visual_review": list({(c["a"], c["b"]): c for c in selected}.values()),
        "refined_candidates": refined,
        "heldout_images_decoded": sum(s.partition.value == "test" for s in samples),
        "labels_or_model_scores_used": False,
        "physical_independence": "unverified",
    }
