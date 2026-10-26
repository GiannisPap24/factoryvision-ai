"""Screen only approved PCB1 development normals before any fitting."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from factoryvision.data.category import normal_members
from factoryvision.data.development import DevelopmentReadGuard
from factoryvision.data.images import decode_image
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.normal_review import review_normals
from factoryvision.training.checkpoint import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    root = ROOT / "data/pcb1-development"
    samples = normal_members(manifest)
    guard = DevelopmentReadGuard(root, samples)
    sys.addaudithook(guard)
    result = review_normals(manifest, root)
    result.update(
        protocol_sha256=file_sha256(ROOT / "docs/protocols/pcb1-confirmatory-v3.json"),
        manifest_sha256=manifest_sha256(manifest),
    )
    write_json(ROOT / "reports", "pcb1-normal-screen.json", result)
    paths = {s.sample_id: s.image_path for s in samples}
    selected = result["selected_for_visual_review"]
    for page in range((len(selected) + 7) // 8):
        canvas = Image.new("RGB", (1040, 1120), "white")
        draw = ImageDraw.Draw(canvas)
        for j, pair in enumerate(selected[page * 8 : (page + 1) * 8]):
            x, y = (j % 2) * 520, (j // 2) * 280
            for k, key in enumerate(("a", "b")):
                image = decode_image(root, paths[pair[key]])
                image.thumbnail((250, 235))
                canvas.paste(image, (x + k * 260, y + 25))
                draw.text((x + k * 260, y + 4), pair[key], fill="black")
            draw.text(
                (x, y + 260),
                f"full={pair['full128_rms']:.4f} crop={pair['crop64_rms']:.4f} "
                f"corr={pair['crop_correlation']}",
                fill="black",
            )
        canvas.save(ROOT / f"reports/pcb1-normal-review-{page + 1}.jpg")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("selected_for_visual_review", "refined_candidates")
            },
            indent=2,
        )
    )
    print(f"Required visual review: {len(selected)} pairs / {(len(selected) + 7) // 8} pages")


if __name__ == "__main__":
    main()
