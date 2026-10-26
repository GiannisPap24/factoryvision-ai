"""Execute the predeclared label-independent integrity/similarity procedure; no model use."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from factoryvision.data.images import decode_image
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.splits import derive_splits
from factoryvision.data.visa import audit_cashew, load_visa_cashew
from factoryvision.training.checkpoint import file_sha256


def main():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "docs/protocols/cashew-evaluation-v1.json").read_text())
    previous = json.loads((root / "data/metadata/cashew-audit.json").read_text())
    actual, audit = audit_cashew(
        load_visa_cashew(root / "data/metadata/1cls.csv"),
        root / "data/visa",
        root / "data/visa/cashew/image_anno.csv",
    )
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    assert actual == load_manifest(root / "data/manifests/cashew-official.json")
    assert audit["records"] == previous["records"]
    assert derive_splits(actual, 42) == manifest
    assert manifest_sha256(manifest) == protocol["dataset"]["manifest_sha256"]
    screen_path = root / protocol["leakage_review"]["base_screen"]
    assert file_sha256(screen_path) == protocol["leakage_review"]["base_screen_sha256"]
    candidates = json.loads(screen_path.read_text())["candidates"]
    cache, paths = {}, {}
    for s in manifest.samples:
        image = decode_image(root / "data/visa", s.image_path)
        full = (
            np.asarray(image.resize((128, 128), Image.Resampling.LANCZOS), dtype=np.float32) / 255
        )
        box = (image.width // 4, image.height // 4, 3 * image.width // 4, 3 * image.height // 4)
        crop = image.crop(box).resize((64, 64), Image.Resampling.LANCZOS)
        rgb = np.asarray(crop, dtype=np.float32) / 255
        gray = np.asarray(crop.convert("L"), dtype=np.float64).ravel()
        gray -= gray.mean()
        cache[s.sample_id] = full, rgb, gray
        paths[s.sample_id] = s.image_path
    refined = []
    for c in candidates:
        af, ac, ag = cache[c["a"]]
        bf, bc, bg = cache[c["b"]]
        full = float(np.sqrt(np.mean((af - bf) ** 2)))
        crop = float(np.sqrt(np.mean((ac - bc) ** 2)))
        corr = float(np.dot(ag, bg) / (np.linalg.norm(ag) * np.linalg.norm(bg)))
        refined.append(
            {
                **c,
                "full128_rms": full,
                "crop64_rms": crop,
                "crop_correlation": corr,
                "strong": (full <= 0.015 and crop <= 0.025) or corr >= 0.995,
            }
        )

    def order(c):
        return hashlib.sha256(f"{c['a']}|{c['b']}".encode()).hexdigest()

    selected = (
        sorted(refined, key=lambda c: (c["full128_rms"], c["a"], c["b"]))[:8]
        + sorted(refined, key=lambda c: (c["crop64_rms"], c["a"], c["b"]))[:8]
        + sorted(refined, key=order)[:8]
        + [c for c in refined if c["strong"]]
    )
    selected = list({(c["a"], c["b"]): c for c in selected}.values())
    result = {
        "protocol_sha256": file_sha256(root / "docs/protocols/cashew-evaluation-v1.json"),
        "manifest_sha256": manifest_sha256(manifest),
        "exact_integrity_passed": True,
        "original_masks_unchanged": True,
        "official_test_unchanged": True,
        "exact_byte_duplicate_groups": audit["byte_duplicate_groups"],
        "exact_pixel_duplicate_groups": audit["pixel_duplicate_groups"],
        "candidate_count": len(refined),
        "strong_count": sum(c["strong"] for c in refined),
        "selected_for_visual_review": selected,
        "refined_candidates": refined,
        "labels_and_model_scores_used": False,
    }
    (root / "reports/leakage-2b-screen.json").write_text(json.dumps(result, indent=2) + "\n")
    for page in range((len(selected) + 7) // 8):
        canvas = Image.new("RGB", (1040, 1120), "white")
        draw = ImageDraw.Draw(canvas)
        for j, c in enumerate(selected[page * 8 : (page + 1) * 8]):
            x, y = (j % 2) * 520, (j // 2) * 280
            for k, key in enumerate(("a", "b")):
                image = decode_image(root / "data/visa", paths[c[key]])
                image.thumbnail((250, 235))
                canvas.paste(image, (x + k * 260, y + 25))
                draw.text((x + k * 260, y + 4), c[key].replace("visa-cashew-", ""), fill="black")
            draw.text(
                (x, y + 260),
                f"full={c['full128_rms']:.4f} crop={c['crop64_rms']:.4f} "
                f"corr={c['crop_correlation']:.4f}",
                fill="black",
            )
        canvas.save(root / f"reports/leakage-2b-review-{page + 1}.jpg")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("refined_candidates", "selected_for_visual_review")
            },
            indent=2,
        )
    )
    print(f"Visual pairs: {len(selected)}; pages: {(len(selected) + 7) // 8}")


if __name__ == "__main__":
    main()
