"""Deferred 2E2 only. Verify is read-only; open/evaluate require separate owner authorization."""

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from train_cashew import peak_memory

from factoryvision.data.archive import MAX_SELECTED_BYTES, inspect_archive
from factoryvision.data.images import (
    ImageSettings,
    ManifestDataset,
    decode_image,
    decode_pair,
    make_loader,
)
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.data.normal_review import review_samples
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.confirmatory import authorize_opening, summarize, verified_seal
from factoryvision.models.pcb1 import FAMILIES, load_family, predict
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import seed_everything

ROOT = Path(__file__).resolve().parents[1]
OPENING = ROOT / "reports/pcb1-opening"
QUARANTINE = ROOT / "data/pcb1-test-quarantine"


def exclusive_json(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def examples(out, dataset, labels, maps, seal):
    """Only first four anomalous/first two normal IDs, after authorized evaluation."""
    chosen = [
        i
        for label, count in ((1, 4), (0, 2))
        for i in [j for j, y in enumerate(labels) if y == label][:count]
    ]
    for i in chosen:
        sample = dataset[i]
        canvas = Image.new("RGB", (768, 154), "white")
        draw = ImageDraw.Draw(canvas)
        panels = [
            ("RGB", (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)),
            ("mask", np.repeat(sample["mask"].numpy().transpose(1, 2, 0), 3, axis=2) * 255),
        ]
        for family in FAMILIES:
            scale = seal["models"][family]["display_map_max"]
            heat = np.clip(maps[family][i] / max(scale, 1e-12), 0, 1)
            panels.append((family, np.stack((heat, np.zeros_like(heat), 1 - heat), axis=-1) * 255))
        for j, (name, pixels) in enumerate(panels):
            canvas.paste(Image.fromarray(pixels.astype(np.uint8)), (j * 128, 22))
            draw.text((j * 128, 2), name, fill="black")
        canvas.save(out / f"example-{sample['sample_id']}.png")


def open_test(seal, seal_sha, manifest):
    """Authorization is checked by main BEFORE this function; no predictions here."""
    OPENING.mkdir()  # Exclusive marker precedes any held-out payload access.
    exclusive_json(
        OPENING / "started.json",
        dict(seal_sha256=seal_sha, state="OPENED; metrics prohibited until integrity review"),
    )
    wanted = {
        p
        for s in manifest.samples
        if s.partition == Partition.TEST
        for p in (s.image_path, s.mask_path)
        if p
    }
    archive_path = ROOT / "data/raw/VisA_20220922.tar"
    selected = [m for m in inspect_archive(archive_path) if m.isfile() and m.name in wanted]
    if {m.name for m in selected} != wanted or sum(m.size for m in selected) > MAX_SELECTED_BYTES:
        raise ValueError("Held-out extraction inventory/cap mismatch.")
    QUARANTINE.mkdir()
    with tarfile.open(archive_path, "r:") as archive:
        for member in selected:
            path = QUARANTINE / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            if path.stat().st_size != member.size:
                raise ValueError("Truncated held-out payload.")
    rows, exact, pixels = [], {}, {}

    def decode(sample):
        root = QUARANTINE if sample.partition == Partition.TEST else ROOT / "data/pcb1-development"
        return decode_image(root, sample.image_path)

    for sample in manifest.samples:
        root = QUARANTINE if sample.partition == Partition.TEST else ROOT / "data/pcb1-development"
        image, mask = decode_pair(root, sample)
        byte_sha = file_sha256(root / sample.image_path)
        pixel_sha = hashlib.sha256(
            f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        for groups, digest in ((exact, byte_sha), (pixels, pixel_sha)):
            groups.setdefault(digest, []).append(
                dict(id=sample.sample_id, partition=sample.partition.value)
            )
        if sample.partition == Partition.TEST:
            rows.append(
                dict(
                    sample_id=sample.sample_id,
                    image_path=sample.image_path,
                    image_sha256=byte_sha,
                    pixel_sha256=pixel_sha,
                    mask_path=sample.mask_path,
                    mask_sha256=file_sha256(root / sample.mask_path) if sample.mask_path else None,
                    disappearing_mask=bool(
                        sample.mask_path
                        and not np.asarray(mask.resize((128, 128), Image.Resampling.NEAREST)).any()
                    ),
                )
            )
    duplicates = [g for groups in (exact, pixels) for g in groups.values() if len(g) > 1]
    screen = review_samples(manifest.samples, decode)
    exclusive_json(OPENING / "screen.json", screen)
    paths = {s.sample_id: s for s in manifest.samples}
    selected = screen["selected_for_visual_review"]
    for page in range((len(selected) + 7) // 8):
        canvas = Image.new("RGB", (1040, 1120), "white")
        draw = ImageDraw.Draw(canvas)
        for j, pair in enumerate(selected[page * 8 : (page + 1) * 8]):
            x, y = (j % 2) * 520, (j // 2) * 280
            for k, key in enumerate(("a", "b")):
                image = decode(paths[pair[key]])
                image.thumbnail((250, 235))
                canvas.paste(image, (x + k * 260, y + 25))
                draw.text((x + k * 260, y + 4), pair[key], fill="black")
        canvas.save(OPENING / f"review-{page + 1}.png")
    exclusive_json(
        OPENING / "integrity.json",
        dict(
            seal_sha256=seal_sha,
            records=rows,
            duplicates=duplicates,
            screen_sha256=file_sha256(OPENING / "screen.json"),
            required_pairs=[dict(a=p["a"], b=p["b"]) for p in selected],
            extracted_files={p: file_sha256(QUARANTINE / p) for p in sorted(wanted)},
            verdict="BLOCKED" if duplicates else "PENDING REQUIRED VISUAL REVIEW",
            model_predictions=0,
        ),
    )
    print(
        "Opening complete. No predictions made. "
        "Review every required pair before separate evaluate command."
    )


def evaluate(seal, seal_sha, manifest, review_path):
    integrity_path = OPENING / "integrity.json"
    integrity = json.loads(integrity_path.read_text())
    review = json.loads(review_path.read_text())
    if (
        integrity["seal_sha256"] != seal_sha
        or integrity["duplicates"]
        or integrity["screen_sha256"] != file_sha256(OPENING / "screen.json")
        or review.get("integrity_sha256") != file_sha256(integrity_path)
        or review.get("confirmed_leakage") is not False
        or review.get("review_complete") is not True
        or review.get("reviewed_pairs") != integrity["required_pairs"]
    ):
        raise ValueError("Opening integrity/required visual review gate not accepted.")
    for path, sha in integrity["extracted_files"].items():
        if file_sha256(QUARANTINE / path) != sha:
            raise ValueError("Held-out payload changed after integrity review.")
    out = OPENING / "evaluation"
    out.mkdir()  # Never overwrite/re-run a partial or completed evaluation.
    exclusive_json(
        out / "started.json", dict(seal_sha256=seal_sha, review_sha256=file_sha256(review_path))
    )
    seed_everything(42, 4)
    dataset = ManifestDataset(manifest, QUARANTINE, Partition.TEST, ImageSettings(128, 128))
    if len(dataset) > 488 or [s.sample_id for s in dataset.samples] != seal["membership"]["test"]:
        raise ValueError("Frozen official membership mismatch/cap exceeded.")
    labels, masks, identities = [], [], []
    for batch in make_loader(dataset, 8):
        labels.extend(batch["label"].tolist())
        masks.extend(batch["mask"][:, 0].numpy().astype(np.uint8))
        identities.extend(batch["sample_id"])
    masks = np.asarray(masks)
    scores, maps, resources = {}, {}, {}
    for family in FAMILIES:
        entry = seal["models"][family]
        model = load_family(
            ROOT / entry["directory"],
            family,
            seal["manifest_sha256"],
            seal["membership"]["train"],
            ROOT / "data/pretrained/resnet18-f37072fd.pth",
        )
        values, spatial, latencies = [], [], []
        start, cpu = time.perf_counter(), time.process_time()
        for batch in make_loader(dataset, 8):
            if peak_memory() > 4 * 1024**3:
                raise MemoryError("Evaluation peak exceeded 4GiB.")
            began = time.perf_counter()
            s, m = predict(model, family, batch["image"])
            latencies.append(dict(count=len(s), seconds=time.perf_counter() - began))
            values.extend(s.tolist())
            spatial.extend(m.numpy())
        scores[family] = values
        maps[family] = np.asarray(spatial)
        resources[family] = dict(
            wall_seconds=time.perf_counter() - start,
            cpu_seconds=time.process_time() - cpu,
            peak_bytes=peak_memory(),
            preloaded_batches=latencies,
        )
        with (out / f"{family}-raw.npz").open("xb") as f:
            np.savez_compressed(
                f,
                sample_ids=np.asarray(identities),
                labels=np.asarray(labels),
                masks=masks,
                scores=np.asarray(values),
                maps=maps[family],
            )
        exclusive_json(
            out / f"{family}-images.json",
            [
                dict(
                    sample_id=i,
                    label=y,
                    score=s,
                    threshold=entry["threshold"],
                    flagged=s > entry["threshold"],
                )
                for i, y, s in zip(identities, labels, values, strict=True)
            ],
        )
    report = summarize(
        labels, masks, scores, maps, {f: seal["models"][f]["threshold"] for f in FAMILIES}
    )
    report.update(
        resources=resources,
        seal_sha256=seal_sha,
        disappearing_masks=[r["sample_id"] for r in integrity["records"] if r["disappearing_mask"]],
    )
    exclusive_json(out / "metrics.json", report)
    examples(out, dataset, labels, maps, seal)
    exclusive_json(
        out / "evidence.json",
        {p.name: file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "open", "evaluate"))
    parser.add_argument(
        "--seal", type=Path, default=ROOT / "docs/evidence/pcb1-evaluation-seal.json"
    )
    parser.add_argument("--seal-sha", required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    seal = verified_seal(ROOT, args.seal, args.seal_sha)
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    if manifest_sha256(manifest) != seal["manifest_sha256"]:
        raise ValueError("Dataset identity mismatch.")
    if args.mode == "verify":
        print("Frozen PCB1 seal verified. No held-out payload opened.")
        return
    if args.approval is None:
        raise PermissionError("Explicit separate owner approval file required.")
    authorize_opening(args.approval, args.seal_sha)
    if args.mode == "open":
        open_test(seal, args.seal_sha, manifest)
    else:
        if args.review is None:
            raise ValueError("Completed integrity review required.")
        evaluate(seal, args.seal_sha, manifest, args.review)


if __name__ == "__main__":
    main()
