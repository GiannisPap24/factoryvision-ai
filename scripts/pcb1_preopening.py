"""Read-only candidate audit and frozen normal benchmark, before the first PCB1 opening."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from feature_experiment import timed
from train_cashew import peak_memory

from factoryvision.data.development import DevelopmentDataset, DevelopmentReadGuard
from factoryvision.data.manifest import load_manifest, manifest_sha256
from factoryvision.domain.contracts import Partition
from factoryvision.evaluation.confirmatory import pooled_metrics, verified_seal
from factoryvision.evaluation.metrics import binary_metrics
from factoryvision.models.pcb1 import FAMILIES, load_family, predict
from factoryvision.training.checkpoint import file_sha256
from factoryvision.training.engine import parameter_sha256, seed_everything

ROOT = Path(__file__).resolve().parents[1]
SEAL_SHA = "8b8782d73761695ce8d75f612b59cabfec3fc25c717dd4467ac69bada50f2b30"


def main():
    for name in ("reports/pcb1-opening", "data/pcb1-test-quarantine"):
        if (ROOT / name).exists():
            raise ValueError("Pre-opening audit must precede all official test access.")
    seal = verified_seal(ROOT, ROOT / "docs/evidence/pcb1-evaluation-seal.json", SEAL_SHA)
    manifest = load_manifest(ROOT / "data/manifests/pcb1-seed42.json")
    if manifest_sha256(manifest) != seal["manifest_sha256"]:
        raise ValueError("Canonical dataset identity mismatch.")
    allowed = [s for s in manifest.samples if s.partition != Partition.TEST]
    root = ROOT / "data/pcb1-development"
    guard = DevelopmentReadGuard(root, allowed)
    sys.addaudithook(guard)
    seed_everything(42, 4)
    dataset = DevelopmentDataset(manifest, root, Partition.VALIDATION)
    sample = dataset[0]
    probe = sample["image"][None]
    models = {}
    for family in FAMILIES:
        entry = seal["models"][family]
        model = load_family(
            ROOT / entry["directory"],
            family,
            seal["manifest_sha256"],
            seal["membership"]["train"],
            ROOT / "data/pretrained/resnet18-f37072fd.pth",
        )
        before = parameter_sha256(model)
        original = predict(model, family, probe)
        repeated = predict(model, family, probe)
        for a, b in zip(original, repeated, strict=True):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        latency = timed(
            lambda model=model, family=family: predict(model, family, probe), repeats=30, warmups=5
        )
        if parameter_sha256(model) != before:
            raise ValueError("Inference changed candidate state.")
        models[family] = dict(
            **entry,
            normal_probe_score=float(original[0][0]),
            state_sha256=before,
            inference_latency=latency,
            process_peak_bytes=peak_memory(),
            parameters=sum(p.numel() for p in model.parameters()),
            fitted_reference_values=model.reference.numel() if family == "spatial_mean" else None,
            bank_bytes=model.bank.numel() * model.bank.element_size()
            if family == "feature_memory"
            else None,
            artifact_bytes=(ROOT / entry["directory"] / "best.pt").stat().st_size,
            state_unchanged=True,
        )
    decode = timed(lambda: dataset[0], repeats=30, warmups=5)
    # Known orientation/equality: equality is negative; higher scores are positive.
    known = binary_metrics([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9], 0.5)
    assert known["confusion_matrix"] == [[2, 0], [1, 1]]
    assert known["auroc"] == 0.875
    # Full declared pooled-pixel cap on synthetic arrays; no real anomaly assets involved.
    labels = np.zeros(8_000_000, dtype=np.uint8)
    labels[::100] = 1
    scores = labels.astype(np.float32)
    pooled = pooled_metrics(labels, scores)
    assert pooled["auroc"] == pooled["average_precision"] == 1.0
    if peak_memory() > 4 * 1024**3:
        raise MemoryError("Synthetic maximum-size metric feasibility exceeds frozen 4GiB cap.")
    verified_seal(ROOT, ROOT / "docs/evidence/pcb1-evaluation-seal.json", SEAL_SHA)
    protocol = json.loads((ROOT / "docs/protocols/pcb1-confirmatory-v3.json").read_text())
    report = dict(
        schema=1,
        status="PASS — APPROVED TO OPEN ONCE, SUBJECT TO POST-OPENING INTEGRITY GATE",
        created_at_utc=datetime.now(UTC).isoformat(),
        checkpoint=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        seal_sha256=SEAL_SHA,
        protocol_sha256=seal["protocol_sha256"],
        manifest_sha256=seal["manifest_sha256"],
        frozen_configuration=protocol,
        source_hashes={
            p: file_sha256(ROOT / p)
            for p in (
                "scripts/pcb1_blinded.py",
                "scripts/pcb1_preopening.py",
                "src/factoryvision/evaluation/confirmatory.py",
                "tests/test_pcb1_runner.py",
            )
        },
        models=models,
        normal_sample_id=sample["sample_id"],
        warm_decode=decode,
        benchmark="first normal validation image, batch1, 5warmups/30repeats; frozen v3 rule",
        benchmark_peak_scope="cumulative process peak, not independently isolated model allocation",
        synthetic_maximum_pooled_pixels=pooled,
        maximum_audit_process_peak_bytes=peak_memory(),
        equality_and_orientation=known,
        official_test_payloads_accessed=0,
        python=platform.python_version(),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        numpy=np.__version__,
        cpu="AMD Ryzen 7 9700X; 8 physical/16 logical; Torch threads4",
        lock_sha256=file_sha256(ROOT / "uv.lock"),
        protocol_or_candidate_changes=False,
        complete_execution_log="reports/pcb1-2e2-execution.log",
        prior_checks="396 offline passes, one symlink-permission skip; 5 normal-only passes",
    )
    target = ROOT / "reports/pcb1-2e2-preopening.json"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                status=report["status"],
                audit_sha256=file_sha256(target),
                maximum_audit_process_peak_bytes=peak_memory(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
