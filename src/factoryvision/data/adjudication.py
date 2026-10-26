"""Blinded integrity evidence, never model evaluation or automatic object identification."""

import hashlib
import hmac
import io
import re
from datetime import datetime

import numpy as np
from PIL import Image

from factoryvision.data.integrity_groups import ImageFingerprint, group_evidence

OUTCOMES = frozenset(
    ("CONFIRMED_DUPLICATE", "CONFIRMED_RELATED", "CONFIRMED_DISTINCT", "UNRESOLVED")
)


def opaque_id(path):
    return "pcb2-" + hashlib.sha256(path.encode()).hexdigest()[:24]


def bind_queue(required, inventory, secret):
    """Join only the sealed queue; no similarity discovery or label-based ordering."""
    if len(secret) != 32 or not required or len(required) > 200:
        raise ValueError("Invalid blinding key or queue size.")
    records = {opaque_id(r["path"]): r for r in inventory}
    if len(records) != len(inventory):
        raise ValueError("Duplicate inventory identity.")
    result, seen = [], set()
    for number, pair in enumerate(required, 1):
        a, b = pair["a"], pair["b"]
        if a >= b or (a, b) in seen or a not in records or b not in records:
            raise ValueError("Unknown, repeated or unordered queue pair.")
        seen.add((a, b))
        first, second = records[a], records[b]
        relation = "/".join(sorted((first["partition"], second["partition"])))
        if relation != pair["relation"] or relation not in {
            "test/train",
            "train/train",
            "test/test",
        }:
            raise ValueError("Queue relation does not match provenance.")
        digest = hmac.new(secret, f"{a}|{b}".encode(), "sha256").hexdigest()
        left, right = (a, b) if int(digest[-1], 16) % 2 == 0 else (b, a)
        result.append(
            {
                "pair_id": "P-" + digest[:24],
                "original_queue_row": number,
                "a": a,
                "b": b,
                "left": left,
                "right": right,
                "cross_partition": relation == "test/train",
                "relation": relation,
                "original_measurements": dict(pair),
            }
        )
    if len({r["pair_id"] for r in result}) != len(result):
        raise ValueError("Anonymous identity collision.")
    return sorted(result, key=lambda r: (not r["cross_partition"], r["pair_id"]))


def blind_manifest(queue):
    """Allowlist fields; no paths, hashes, prior scores, labels, relations or priority."""
    return [
        {"pair_id": p["pair_id"], "image": p["pair_id"] + ".png"}
        for p in sorted(queue, key=lambda p: p["pair_id"])
    ]


def decode_verified(payload, record):
    if len(payload) > 16 * 1024**2 or hashlib.sha256(payload).hexdigest() != record["byte_sha256"]:
        raise ValueError("Source bytes changed or exceed limit.")
    with Image.open(io.BytesIO(payload)) as original:
        if (
            original.width * original.height > 16_000_000
            or original.format not in {"JPEG", "PNG"}
            or getattr(original, "n_frames", 1) != 1
            or original.getexif().get(274, 1) != 1
        ):
            raise ValueError("Unsupported image dimensions, format or orientation.")
        image = original.convert("RGB")
    digest = hashlib.sha256(
        f"RGB:{image.width}:{image.height}:".encode() + image.tobytes()
    ).hexdigest()
    if digest != record["pixel_sha256"] or list(image.size) != [record["width"], record["height"]]:
        raise ValueError("Decoded pixels or dimensions changed.")
    return image


def pair_png(left, right):
    """Native RGB pixels, left then right with 16px gray gutter; no inherited metadata."""
    canvas = Image.new("RGB", (left.width + 16 + right.width, max(left.height, right.height)), 128)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + 16, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


def _gray(image):
    return (
        np.asarray(image.convert("L").resize((256, 256), Image.Resampling.LANCZOS), np.float64)
        / 255
    )


def _blur(array):
    x = np.arange(-5, 6, dtype=np.float64)
    weights = np.exp(-(x * x) / (2 * 1.5**2))
    weights /= weights.sum()
    horizontal = np.einsum(
        "ijk,k->ij", np.lib.stride_tricks.sliding_window_view(array, 11, axis=1), weights
    )
    return np.einsum(
        "ijk,k->ij", np.lib.stride_tricks.sliding_window_view(horizontal, 11, axis=0), weights
    )


def ssim(a, b):
    """Mean valid-window SSIM, Gaussian11 sigma1.5, population moments, L=1."""
    if (
        a.shape != b.shape
        or a.ndim != 2
        or min(a.shape) < 11
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or min(a.min(), b.min()) < 0
        or max(a.max(), b.max()) > 1
    ):
        raise ValueError("SSIM requires equal finite grayscale arrays in [0,1].")
    ma, mb = _blur(a), _blur(b)
    va, vb = np.maximum(0, _blur(a * a) - ma * ma), np.maximum(0, _blur(b * b) - mb * mb)
    covariance = _blur(a * b) - ma * mb
    score = ((2 * ma * mb + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2)
    )
    return float(np.mean(score))


def structural_evidence(left, right):
    """One bounded translation estimate; no matching threshold or identity decision."""
    a, b = _gray(left), _gray(right)
    raw = ssim(a, b)
    ac, bc = a - a.mean(), b - b.mean()
    result = {
        "raw_ssim": raw,
        "translation_dy_dx": None,
        "aligned_ssim": None,
        "status": "degenerate",
    }
    if np.linalg.norm(ac) < 1e-12 or np.linalg.norm(bc) < 1e-12:
        return result
    spectrum = np.fft.fft2(ac) * np.conj(np.fft.fft2(bc))
    phase = spectrum / np.maximum(np.abs(spectrum), 1e-12)
    surface = np.fft.ifft2(phase).real
    location = np.unravel_index(np.argmax(surface), surface.shape)
    shift = [int(v if v <= 128 else v - 256) for v in location]
    result["translation_dy_dx"] = shift
    if max(map(abs, shift)) > 16:
        return {**result, "status": "translation_outside_bound"}
    ay, ax = (slice(max(v, 0), min(256 + v, 256)) for v in shift)
    by, bx = (slice(max(-v, 0), min(256 - v, 256)) for v in shift)
    return {**result, "aligned_ssim": ssim(a[ay, ax], b[by, bx]), "status": "translation_only"}


def adjudicate(queue, inventory, decisions, *, humans=None, provenance_valid=True):
    """Validate attestations; neither this function nor metrics authenticate human identity."""
    humans = humans or {}
    by_pair = {p["pair_id"]: p for p in queue}
    if not by_pair or len(by_pair) != len(queue):
        raise ValueError("Empty or repeated review queue.")
    by_id = {opaque_id(r["path"]): r for r in inventory}
    accepted, confirmed, distinct = {}, [], []
    for decision in decisions:
        pair_id = decision["pair_id"]
        if pair_id not in by_pair or pair_id in accepted:
            raise ValueError("Unknown or conflicting pair decision.")
        outcome = decision["decision"]
        if outcome not in OUTCOMES:
            raise ValueError("Unsupported classification.")
        stamp = datetime.fromisoformat(decision["timestamp"])
        if stamp.tzinfo is None or not decision["evidence"].strip():
            raise ValueError("Timezone and concrete evidence required.")
        pair = by_pair[pair_id]
        a, b = by_id[pair["a"]], by_id[pair["b"]]
        exact = any(a[k] == b[k] for k in ("byte_sha256", "pixel_sha256"))
        human = decision["reviewer_type"] == "independent_human"
        if decision["reviewer_type"] not in {"technical_automation", "independent_human"}:
            raise ValueError("Unsupported reviewer provenance.")
        if decision["independent_human_review"] is not human:
            raise ValueError("False human-review assertion.")
        if human:
            registry = humans.get(decision.get("reviewer_id"), {})
            if (
                registry.get("independent") is not True
                or registry.get("blinded") is not True
                or not registry.get("attestation_sha256")
                or not re.fullmatch("[0-9a-f]{64}", registry["attestation_sha256"])
            ):
                raise ValueError("Independent blinded reviewer attestation required.")
        resolved = outcome != "UNRESOLVED"
        if resolved and decision["confidence"] != "high":
            raise ValueError("High confidence required for a resolved decision.")
        if not resolved and decision["confidence"] != "insufficient":
            raise ValueError("Unresolved decisions require insufficient confidence.")
        if exact and outcome in {"CONFIRMED_DISTINCT", "CONFIRMED_RELATED"}:
            raise ValueError("Exact content contradicts classification.")
        if (
            resolved
            and not (outcome == "CONFIRMED_DUPLICATE" and exact)
            and (
                not human
                or decision.get("basis")
                not in {
                    "capture_provenance",
                    "derived_capture",
                    "object_specific_markings",
                    "physical_distinction",
                }
                or len(decision.get("landmarks", [])) < 2
                or any(not isinstance(x, str) or not x.strip() for x in decision["landmarks"])
            )
        ):
            raise ValueError("Independent evidence beyond similarity required.")
        accepted[pair_id] = decision
        if outcome in {"CONFIRMED_DUPLICATE", "CONFIRMED_RELATED"}:
            confirmed.append((pair["a"], pair["b"]))
        elif outcome == "CONFIRMED_DISTINCT":
            distinct.append((pair["a"], pair["b"]))
    records = [
        ImageFingerprint(
            k, r["partition"], r["byte_sha256"], r["pixel_sha256"], r["dhash"], r["phash"]
        )
        for k, r in by_id.items()
    ]
    unresolved = [
        p
        for p in queue
        if p["pair_id"] not in accepted or accepted[p["pair_id"]]["decision"] == "UNRESOLVED"
    ]
    groups = group_evidence(
        records,
        confirmed_pairs=confirmed,
        unresolved_pairs=[
            (p["a"], p["b"]) for p in unresolved if (p["a"], p["b"]) not in confirmed
        ],
    )
    for group in groups["related_groups"]:
        if any(a in group["members"] and b in group["members"] for a, b in distinct):
            raise ValueError("Distinct decision conflicts with a confirmed connected group.")
    verdict = "INCONCLUSIVE"
    if provenance_valid and groups["cross_partition_groups"]:
        verdict = "BLOCKED"
    elif provenance_valid and not unresolved:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "groups": groups,
        "unresolved": [p["pair_id"] for p in unresolved],
        "independent_human_decisions": sum(
            d["independent_human_review"] for d in accepted.values()
        ),
        "accepted_decisions": len(accepted),
        "provenance_valid": provenance_valid,
    }
