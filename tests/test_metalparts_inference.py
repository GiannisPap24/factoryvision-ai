"""Synthetic-only inference, source restrictions, metric ties and native geometry."""

import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

from factoryvision.data.metalparts import SYNTHETIC_REVISION, MetalPartsSample, MetalPartsSource
from factoryvision.data.metalparts_development import TRAIN_METADATA, DevelopmentSample, PatchConfig
from factoryvision.data.metalparts_diagnostic import DiagnosticReadGuard, audit_source, source_pair
from factoryvision.domain.contracts import Label, Partition
from factoryvision.evaluation.metalparts import PixelRanking, ranking_metrics
from factoryvision.inference import metalparts as module
from factoryvision.inference.metalparts import MetalPartsService, reconstruct_native
from factoryvision.inference.service import ModelUnavailable
from factoryvision.inference.uploads import InputRejected, protected_hashes
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.metalparts import native_diagnostic


def payload(size=(301, 279)):
    image = Image.fromarray(
        np.random.default_rng(9).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return image, buffer.getvalue()


class Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_native_edges_overlap_alignment():
    image, _ = payload()
    reconstruction, spatial, score = reconstruct_native(torch.nn.Identity(), image)
    np.testing.assert_allclose(
        reconstruction, np.asarray(image).astype(np.float32) / 255, atol=1e-7
    )
    assert spatial.shape == (279, 301) and score < 1e-14
    reconstruction, spatial, score = reconstruct_native(Zero(), image)
    expected = ((np.asarray(image).astype(np.float32) / 255) ** 2).mean(axis=2)
    np.testing.assert_array_equal(spatial, expected)
    assert score == float(expected.mean()) and not reconstruction.any()
    # Known local pixel signal is at its original coordinate, including trailing edges.
    pixels = np.zeros((277, 299, 3), dtype=np.uint8)
    pixels[276, 298] = 255
    _, spatial, _ = reconstruct_native(Zero(), Image.fromarray(pixels))
    assert spatial[276, 298] == 1 and np.count_nonzero(spatial) == 1


@pytest.mark.parametrize(
    "image",
    [Image.new("RGB", (255, 256)), Image.new("L", (256, 256)), Image.new("RGB", (4097, 256))],
)
def test_geometry_rejected(image):
    with pytest.raises(InputRejected):
        reconstruct_native(Zero(), image)


def test_invalid_model_and_coverage(monkeypatch):
    class Bad(torch.nn.Module):
        def forward(self, x):
            return x * float("nan")

    with pytest.raises(ModelUnavailable, match="Invalid"):
        reconstruct_native(Bad(), Image.new("RGB", (256, 256)))
    monkeypatch.setattr(module, "coordinates", lambda *a, **k: [])
    with pytest.raises(ModelUnavailable, match="uncovered"):
        reconstruct_native(Zero(), Image.new("RGB", (256, 256)))


def test_exact_agreement_with_accepted_3d(tmp_path):
    image, _ = payload()
    source = MetalPartsSource(SYNTHETIC_REVISION)
    sample = MetalPartsSample(source, "train", "normal/fake.png", Label.NORMAL, None, *image.size)
    path = tmp_path / sample.image_path
    path.parent.mkdir(parents=True)
    image.save(path)
    record = DevelopmentSample(sample, file_sha256(path), "a" * 64, Partition.VALIDATION)
    expected = native_diagnostic(Zero(), tmp_path, record, PatchConfig())
    actual = reconstruct_native(Zero(), image)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_metric_correctness_ties_and_histogram():
    rng = np.random.default_rng(7)
    scores = rng.choice([0.0, 0.001, 0.2, 0.5, 1.0], size=(57, 83)).astype(np.float32)
    mask = rng.integers(0, 2, size=scores.shape, dtype=np.uint8)
    exact = ranking_metrics(mask.ravel(), scores.ravel())
    assert exact["auroc"] == pytest.approx(roc_auc_score(mask.ravel(), scores.ravel()))
    assert exact["average_precision"] == pytest.approx(
        average_precision_score(mask.ravel(), scores.ravel())
    )
    hist = PixelRanking()
    hist.add(scores[:20], mask[:20])
    hist.add(scores[20:], mask[20:])
    quantized = np.floor(scores.astype(np.float64) * 65535)
    got = hist.result()
    assert got["auroc"] == pytest.approx(roc_auc_score(mask.ravel(), quantized.ravel()))
    assert got["average_precision"] == pytest.approx(
        average_precision_score(mask.ravel(), quantized.ravel())
    )
    assert got["positive"] + got["negative"] == mask.size
    assert ranking_metrics([0, 1], [0, 1])["auroc"] == 1
    assert ranking_metrics([0, 1], [1, 0])["auroc"] == 0


@pytest.mark.parametrize("labels,scores", [([], []), ([0, 0], [1, 2]), ([1, 1], [1, 2])])
def test_undefined_metrics(labels, scores):
    result = ranking_metrics(labels, scores)
    assert result["auroc"] is None and result["average_precision"] is None and result["reason"]
    assert PixelRanking().result()["auroc"] is None


@pytest.mark.parametrize(
    "labels,scores", [([0], [float("nan")]), ([2], [0]), ([0, 1], [0]), ([[0]], [[0]])]
)
def test_invalid_ranking(labels, scores):
    with pytest.raises(ValueError):
        ranking_metrics(labels, scores)


@pytest.mark.parametrize(
    "score,mask",
    [
        (np.zeros((2, 2)), np.ones((1, 2))),
        (np.zeros((2, 2)), np.full((2, 2), 2)),
        (np.full((2, 2), np.nan), np.zeros((2, 2))),
        (np.full((2, 2), -1), np.zeros((2, 2))),
        (np.full((2, 2), 1.1), np.zeros((2, 2))),
    ],
)
def test_invalid_pixel_metrics(score, mask):
    with pytest.raises(ValueError):
        PixelRanking().add(score, mask)


def test_service_missing_category_catalog(tmp_path, monkeypatch):
    engine = MetalPartsService(tmp_path)
    assert not engine.status()["available"]
    with pytest.raises(ModelUnavailable):
        engine.inspect(b"irrelevant", category="cashew")
    with pytest.raises(ModelUnavailable):
        engine._load()
    original = module.catalog()
    for field, value in [
        ("dataset", "visa"),
        ("purpose", "smoke"),
        ("patches", {"size": 128, "stride": 96, "limit": 12}),
    ]:
        mutated = json.loads(json.dumps(original))
        mutated["identity"][field] = value
        monkeypatch.setattr(module, "catalog", lambda mutated=mutated: mutated)
        with pytest.raises(ValueError):
            MetalPartsService(tmp_path)


def test_service_inference_no_decision_persistence(tmp_path, monkeypatch):
    engine = MetalPartsService(tmp_path)
    monkeypatch.setattr(engine, "_load", lambda: Zero())
    _, data = payload()
    result = engine.inspect(data)
    assert result.threshold is None and result.above_threshold is None
    assert result.model_id == "metalparts_autoencoder" and result.spatial.shape == (279, 301)
    assert not result.spatial.flags.writeable and result.total_ms >= result.inference_ms > 0
    assert list(tmp_path.iterdir()) == []


def test_protected_target_hashes_added():
    byte, pixel = protected_hashes()
    assert len(byte) == 1721 and len(pixel) == 1721


def test_verified_synthetic_checkpoint_cache_and_drift(tmp_path, monkeypatch):
    from factoryvision.models.autoencoder import ConvAutoencoder
    from factoryvision.training.metalparts import save_model

    registry = module.catalog()
    run = tmp_path / "runs/fabricated"
    run.mkdir(parents=True)
    save_model(run, ConvAutoencoder(256), registry["identity"], 1, 0.1)
    for key, name in [("checkpoint", "best.pt"), ("metadata", "checkpoint.json")]:
        path = run / name
        registry["assets"][key] = {
            "path": path.relative_to(tmp_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    monkeypatch.setattr(module, "catalog", lambda: registry)
    engine = MetalPartsService(tmp_path)
    assert engine.status()["available"]
    first = engine._load()
    assert engine._load() is first and not any(p.requires_grad for p in first.parameters())
    result = engine.inspect(payload((256, 256))[1])
    assert result.score >= 0 and result.threshold is None
    (run / "best.pt").write_bytes(b"corrupt")
    assert not engine.status()["available"]
    with pytest.raises(ModelUnavailable):
        engine._load()


def test_artifact_metadata_and_load_failures(tmp_path, monkeypatch):
    engine = MetalPartsService(tmp_path)
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"identity": {"category": "cashew"}}))
    monkeypatch.setattr(module, "verified_asset", lambda *a: path)
    assert not engine.status()["available"]
    monkeypatch.setattr(engine, "_verify", lambda: {"checkpoint": path})

    def fail(*args):
        raise ValueError("fabricated incompatible tensors")

    monkeypatch.setattr(module, "load_model", fail)
    with pytest.raises(ModelUnavailable, match="could not be loaded"):
        engine._load()


@pytest.fixture
def source_fixture(tmp_path):
    source = MetalPartsSource(SYNTHETIC_REVISION)
    verified, old, excluded, rows = {}, [], [], []
    for name, label, partition in [
        ("fit", 0, "train"),
        ("val", 0, "validation"),
        ("anomaly", 1, "source_anomaly"),
    ]:
        file_name = f"{'anomaly' if label else 'normal'}/{name}.png"
        mask_path = f"MetalParts/metal_parts_data_dir_train/masks/{name}.png" if label else None
        sample = MetalPartsSample(
            source,
            "train",
            file_name,
            Label.ANOMALY if label else Label.NORMAL,
            mask_path,
            301,
            279,
        )
        path = tmp_path / sample.image_path
        path.parent.mkdir(parents=True, exist_ok=True)
        image, _ = payload()
        image.save(path)
        verified[sample.image_path] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        if label:
            path = tmp_path / mask_path
            path.parent.mkdir(parents=True)
            Image.new("L", image.size, 255).save(path)
            verified[mask_path] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
            excluded.append({"sample_id": sample.sample_id})
        else:
            old.append(
                {
                    "sample_id": sample.sample_id,
                    "image_path": sample.image_path,
                    "partition": partition,
                    "width": 301,
                    "height": 279,
                    "content_sha256": verified[sample.image_path]["sha256"],
                    "pixel_sha256": hashlib.sha256(b"RGB:301:279:" + image.tobytes()).hexdigest(),
                }
            )
        rows.append({"file_name": file_name, "label": label, "mask": mask_path})
    path = tmp_path / TRAIN_METADATA
    path.write_bytes(("\n".join(json.dumps(r) for r in rows)).encode())
    verified[TRAIN_METADATA] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return tmp_path, verified, {"records": old, "excluded": excluded}, source


def test_source_audit_selection_and_guard(source_fixture):
    root, verified, development, source = source_fixture
    samples, inventory = audit_source(*source_fixture)
    assert len(samples) == 2 and len(inventory) == 3
    assert sum(s.label == Label.ANOMALY for s in samples) == 1
    guard = DiagnosticReadGuard(root, verified)
    guard("open", (str(root / TRAIN_METADATA), "r", 0))
    assert guard.opened == {TRAIN_METADATA}
    guard("other", ())
    with pytest.raises(PermissionError):
        guard("open", (str(root / "MetalParts/metal_parts_data_dir_test0/normal/x.png"), "r", 0))
    with pytest.raises(ValueError):
        DiagnosticReadGuard(root, ["MetalParts/metal_parts_data_dir_test0/normal/x.png"])
    with pytest.raises(ValueError):
        DiagnosticReadGuard(root, [])
    test = replace(next(s for s in samples if s.label == Label.NORMAL), source_domain="test0")
    with pytest.raises(ValueError, match="official test"):
        source_pair(root, test, verified)
    development["records"][0]["partition"] = "test"
    with pytest.raises(ValueError, match="membership"):
        audit_source(*source_fixture)


def test_invalid_source_masks_and_inventory(source_fixture):
    root, verified, development, source = source_fixture
    samples, _ = audit_source(*source_fixture)
    anomaly = next(s for s in samples if s.label == Label.ANOMALY)
    path = root / anomaly.mask_path
    for size, value in [((301, 279), 0), ((301, 279), 7), ((300, 279), 255)]:
        Image.new("L", size, value).save(path)
        verified[anomaly.mask_path] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        with pytest.raises(ValueError, match="masks"):
            source_pair(root, anomaly, verified)
    with pytest.raises(ValueError, match="geometry"):
        source_pair(root, replace(anomaly, width=300), verified)


@pytest.mark.parametrize("mutation", ["duplicate", "extra", "excluded", "anomaly_member"])
def test_source_membership_drift(source_fixture, mutation):
    root, verified, development, source = source_fixture
    if mutation == "duplicate":
        development["records"].append(development["records"][0])
    elif mutation == "extra":
        development["records"].append({**development["records"][0], "sample_id": "extra"})
    elif mutation == "excluded":
        development["excluded"] = []
    else:
        development["records"].append({"sample_id": development["excluded"][0]["sample_id"]})
    with pytest.raises(ValueError):
        audit_source(root, verified, development, source)


@pytest.mark.parametrize("mutation", ["fields", "duplicate"])
def test_source_metadata_drift(source_fixture, mutation):
    root, verified, development, source = source_fixture
    path = root / TRAIN_METADATA
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "fields":
        rows[0]["label"] = True
    else:
        rows.append(rows[0])
    path.write_text("\n".join(json.dumps(row) for row in rows))
    verified[TRAIN_METADATA] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    with pytest.raises(ValueError):
        audit_source(root, verified, development, source)


def test_explicit_runner_synthetic_end_to_end(source_fixture, monkeypatch):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).parents[1] / "scripts/metalparts_diagnostic.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("diagnostic_runner", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root, verified, _, _ = source_fixture
    samples, inventory = audit_source(*source_fixture)
    registry = {"assets": {"checkpoint": {"sha256": "a" * 64}}}
    manifest = {
        "files": {},
        "model": registry,
        "inventory": inventory,
        "selected_ids": [s.sample_id for s in samples],
        "preview_ids": [],
    }
    # Production IDs are computed with its source identity; preselect both synthetic paths.
    manifest["preview_ids"] = [replace(s, source=MetalPartsSource()).sample_id for s in samples]
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "RUN", root / "runs/fabricated-diagnostic")
    monkeypatch.setattr(runner, "MANIFEST", path)
    guard = DiagnosticReadGuard(root, verified)
    monkeypatch.setattr(runner, "source_inputs", lambda: (root, verified, guard))

    class Engine:
        def __init__(self, root):
            self.registry = registry

        def _load(self):
            return Zero()

    monkeypatch.setattr(runner, "MetalPartsService", Engine)
    runner.evaluate()
    result = json.loads((runner.RUN / "metrics.json").read_text())
    assert result["image"]["positive"] == result["image"]["negative"] == 1
    assert result["pixel"]["positive"] == result["pixel"]["negative"] == 301 * 279
    assert result["exact_3d_native_replay"] and len(result["preview_assets"]) == 2
    with pytest.raises(FileExistsError):
        runner.evaluate()
    manifest["files"] = {"manifest.json": "a" * 64}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="drift"):
        runner.evaluate()
