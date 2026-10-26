"""Fabricated images and explicit fictional reviewer attestations only."""

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from factoryvision.data.adjudication import (
    adjudicate,
    bind_queue,
    blind_manifest,
    decode_verified,
    opaque_id,
    pair_png,
    ssim,
    structural_evidence,
)
from factoryvision.data.pretraining_integrity import describe_image


def fixture(colors=("red", "blue", "green"), partitions=("train", "test", "test")):
    rows, payloads = [], []
    for i, (color, partition) in enumerate(zip(colors, partitions, strict=True)):
        image = Image.new("RGB", (32, 32), color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()
        record, _ = describe_image(str(i), partition, payload)
        rows.append(
            {
                "path": f"pcb2/Data/Images/Normal/{i}.png",
                "sample_id": f"synthetic-normal-{i}",
                "label": "normal",
                "partition": partition,
                "width": 32,
                "height": 32,
                "byte_sha256": record.byte_sha256,
                "pixel_sha256": record.pixel_sha256,
                "dhash": record.dhash,
                "phash": record.phash,
            }
        )
        payloads.append(payload)
    required = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = sorted((opaque_id(rows[i]["path"]), opaque_id(rows[j]["path"])))
            required.append(
                {
                    "a": a,
                    "b": b,
                    "relation": "/".join(sorted((rows[i]["partition"], rows[j]["partition"]))),
                }
            )
    required.sort(key=lambda p: (p["a"], p["b"]))
    return rows, required, payloads


def decision(pair, outcome="UNRESOLVED", human=False):
    return {
        "pair_id": pair["pair_id"],
        "decision": outcome,
        "evidence": "Fabricated test-only capture evidence.",
        "reviewer_type": "independent_human" if human else "technical_automation",
        "reviewer_id": "fictional-test-reviewer" if human else None,
        "independent_human_review": human,
        "confidence": "insufficient" if outcome == "UNRESOLVED" else "high",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "basis": "capture_provenance",
        "landmarks": ["fabricated serial A", "fabricated capture ledger B"],
    }


HUMANS = {
    "fictional-test-reviewer": {
        "independent": True,
        "blinded": True,
        "attestation_sha256": "a" * 64,
    }
}


def test_anonymity_completeness_priority_and_determinism():
    rows, required, _ = fixture()
    queue = bind_queue(required, rows, b"s" * 32)
    assert queue == bind_queue(required, rows[::-1], b"s" * 32)
    assert [p["cross_partition"] for p in queue] == [True, True, False]
    assert {p["original_queue_row"] for p in queue} == {1, 2, 3}
    visible = blind_manifest(queue)
    assert all(set(r) == {"pair_id", "image"} for r in visible)
    assert not any(
        word in str(visible).lower()
        for word in ("train", "test", "normal", "anomaly", "pcb2", "relation", "sha256")
    )
    changed = [{**r, "label": "anomaly", "sample_id": "hidden"} for r in rows]
    assert blind_manifest(bind_queue(required, changed, b"s" * 32)) == visible
    assert blind_manifest(bind_queue(required, rows, b"x" * 32)) != visible


def test_invalid_queue_and_relations():
    rows, required, _ = fixture()
    for q, inv, key in [
        ([], rows, b"s" * 32),
        (required, rows, b"x"),
        (required, rows + [rows[0]], b"s" * 32),
        (required + [required[0]], rows, b"s" * 32),
        ([{**required[0], "a": "unknown"}], rows, b"s" * 32),
        ([{**required[0], "relation": "invented"}], rows, b"s" * 32),
    ]:
        with pytest.raises(ValueError):
            bind_queue(q, inv, key)


def test_verified_decode_and_metadata_free_native_pair():
    rows, _, payloads = fixture()
    before = list(payloads)
    left = decode_verified(payloads[0], rows[0])
    right = decode_verified(payloads[1], rows[1])
    left.info["private"] = "normal/train"
    result = pair_png(left, right)
    with Image.open(io.BytesIO(result)) as image:
        assert image.size == (80, 32) and not image.info
        assert image.crop((0, 0, 32, 32)).tobytes() == left.tobytes()
        assert image.crop((48, 0, 80, 32)).tobytes() == right.tobytes()
    assert payloads == before
    with pytest.raises(ValueError, match="bytes"):
        decode_verified(payloads[1], rows[0])
    with pytest.raises(ValueError, match="pixels"):
        decode_verified(payloads[0], {**rows[0], "width": 99})
    bad = io.BytesIO()
    left.save(bad, format="BMP")
    with pytest.raises(ValueError, match="Unsupported"):
        decode_verified(
            bad.getvalue(), {**rows[0], "byte_sha256": hashlib.sha256(bad.getvalue()).hexdigest()}
        )


def test_ssim_known_answers_and_translation_recovery():
    zero, one = np.zeros((32, 32)), np.ones((32, 32))
    assert ssim(zero, zero) == pytest.approx(1)
    assert ssim(zero, one) == pytest.approx(0.0001 / 1.0001)
    rng = np.random.default_rng(18)
    a = rng.integers(0, 256, (256, 256), dtype=np.uint8)
    first = Image.fromarray(a)
    moved = Image.fromarray(np.roll(a, (4, -3), axis=(0, 1)))
    evidence = structural_evidence(first, moved)
    assert evidence["translation_dy_dx"] == [-4, 3]
    assert evidence["aligned_ssim"] == pytest.approx(1)
    assert evidence["raw_ssim"] < 0.1
    assert (
        structural_evidence(first, Image.fromarray(np.roll(a, 30, axis=0)))["status"]
        == "translation_outside_bound"
    )
    assert (
        structural_evidence(Image.new("L", (32, 32)), Image.new("L", (32, 32), 255))["status"]
        == "degenerate"
    )
    for b in (np.ones((4, 4)), one * np.nan, one * 2, one * -1):
        with pytest.raises(ValueError):
            ssim(zero, b)


def test_unresolved_cannot_pass_and_missing_human_not_invented():
    rows, required, _ = fixture()
    queue = bind_queue(required, rows, b"s" * 32)
    result = adjudicate(queue, rows, [decision(p) for p in queue])
    assert result["verdict"] == "INCONCLUSIVE"
    assert len(result["unresolved"]) == 3 and result["independent_human_decisions"] == 0
    assert len(result["groups"]["unresolved_cross_partition_pairs"]) == 2
    assert adjudicate(queue, rows, [])["verdict"] == "INCONCLUSIVE"


def test_independent_resolutions_and_cross_partition_stop():
    rows, required, _ = fixture()
    queue = bind_queue(required, rows, b"s" * 32)
    decisions = [decision(p, "CONFIRMED_DISTINCT", True) for p in queue]
    assert adjudicate(queue, rows, decisions, humans=HUMANS)["verdict"] == "PASS"
    assert (
        adjudicate(queue, rows, decisions, humans=HUMANS, provenance_valid=False)["verdict"]
        == "INCONCLUSIVE"
    )
    for outcome in ("CONFIRMED_DUPLICATE", "CONFIRMED_RELATED"):
        d = decision(queue[0], outcome, True)
        assert adjudicate(queue, rows, [d], humans=HUMANS)["verdict"] == "BLOCKED"


@pytest.mark.parametrize(
    "change",
    [
        {"decision": "invented"},
        {"pair_id": "unknown"},
        {"timestamp": "2026-01-01"},
        {"evidence": " "},
        {"reviewer_type": "AI-human"},
        {"independent_human_review": True},
        {"confidence": "high"},
        {"decision": "CONFIRMED_DISTINCT", "confidence": "high"},
    ],
)
def test_reject_unsupported_and_untraceable_decisions(change):
    rows, required, _ = fixture()
    queue = bind_queue(required, rows, b"s" * 32)
    with pytest.raises(ValueError):
        adjudicate(queue, rows, [{**decision(queue[0]), **change}])


def test_reviewer_provenance_conflicts_and_graph_consistency():
    rows, required, _ = fixture()
    queue = bind_queue(required, rows, b"s" * 32)
    d = decision(queue[0], "CONFIRMED_DISTINCT", True)
    for ds, humans in [
        ([d], {}),
        ([{**d, "confidence": "low"}], HUMANS),
        ([{**d, "basis": "high_ssim"}], HUMANS),
        ([{**d, "landmarks": [""]}], HUMANS),
        ([d, d], HUMANS),
    ]:
        with pytest.raises(ValueError):
            adjudicate(queue, rows, ds, humans=humans)
    decisions = [decision(p, "CONFIRMED_RELATED", True) for p in queue]
    decisions[-1] = decision(queue[-1], "CONFIRMED_DISTINCT", True)
    with pytest.raises(ValueError, match="connected group"):
        adjudicate(queue, rows, decisions, humans=HUMANS)
    with pytest.raises(ValueError, match="queue"):
        adjudicate([], rows, [])
    with pytest.raises(ValueError, match="queue"):
        adjudicate(queue + [queue[0]], rows, [])


def test_exact_confirmation_and_contradiction():
    rows, required, _ = fixture(("red", "red"), ("train", "test"))
    queue = bind_queue(required, rows, b"s" * 32)
    assert (
        adjudicate(queue, rows, [decision(queue[0], "CONFIRMED_DUPLICATE")])["verdict"] == "BLOCKED"
    )
    for outcome in ("CONFIRMED_DISTINCT", "CONFIRMED_RELATED"):
        with pytest.raises(ValueError, match="Exact"):
            adjudicate(queue, rows, [decision(queue[0], outcome, True)], humans=HUMANS)
