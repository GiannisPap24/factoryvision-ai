"""Real Cashew batch/debug target; no model, optimization or dataset download."""

import ctypes
import json
import sys
import time
from ctypes import wintypes
from pathlib import Path

import torch

from factoryvision.data.images import ImageSettings, ManifestDataset, make_loader
from factoryvision.data.manifest import load_manifest
from factoryvision.domain.contracts import Partition


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(2)
    manifest = load_manifest(root / "data/manifests/cashew-seed42.json")
    dataset = ManifestDataset(manifest, root / "data/visa", Partition.TRAIN, ImageSettings())
    loader = make_loader(dataset, batch_size=4)
    started = time.perf_counter()
    batch = next(iter(loader))
    assert batch["image"].shape == (4, 3, 256, 256)  # Verification breakpoint: inspect batch.
    assert batch["mask"].shape == (4, 1, 256, 256)
    assert batch["label"].sum().item() == 0
    report = {
        "python": sys.executable,
        "torch": torch.__version__,
        "device": str(batch["image"].device),
        "image_shape": list(batch["image"].shape),
        "mask_shape": list(batch["mask"].shape),
        "image_dtype": str(batch["image"].dtype),
        "mask_dtype": str(batch["mask"].dtype),
        "label_dtype": str(batch["label"].dtype),
        "sample_ids": batch["sample_id"],
        "workers": loader.num_workers,
        "cpu_threads": torch.get_num_threads(),
        "batch_tensor_bytes": sum(
            tensor.nelement() * tensor.element_size()
            for tensor in batch.values()
            if isinstance(tensor, torch.Tensor)
        ),
        "first_batch_seconds": round(time.perf_counter() - started, 4),
    }
    if sys.platform == "win32":

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working_set",
                    "working_set",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        info = MemoryCounters()
        info.cb = ctypes.sizeof(info)
        query = ctypes.windll.psapi.GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        query.restype = wintypes.BOOL
        if query(wintypes.HANDLE(-1), ctypes.byref(info), info.cb):
            report["process_peak_working_set_bytes"] = info.peak_working_set
    (root / "reports/dataloader-smoke.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
