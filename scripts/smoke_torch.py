"""Explicit CPU/native-library check; no model, training loop or network access."""

import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "pillow")
        },
        "passed": False,
    }
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        import torchvision

        assert torch.version.cuda is None
        assert torch.version.hip is None
        assert not torch.cuda.is_available()
        torch.set_num_threads(2)
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, device="cpu")
        loss = x.square().sum()
        loss.backward()
        torch.testing.assert_close(x.grad, torch.tensor([2.0, 4.0, 6.0]))
        torch.testing.assert_close(torch.from_numpy(np.array([1, 2])), torch.tensor([1, 2]))
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
        assert kept.tolist() == [0]
        output = root / ".tmp/pytorch-smoke/tensor.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensor": x.detach()}, output)
        loaded = torch.load(output, map_location="cpu", weights_only=True)
        torch.testing.assert_close(loaded["tensor"], x.detach())
        report.update(
            passed=True,
            torch_version=torch.__version__,
            torchvision_version=torchvision.__version__,
            cuda_build=torch.version.cuda,
            hip_build=torch.version.hip,
            device=str(x.device),
            gradient=x.grad.tolist(),
            native_torchvision_nms=kept.tolist(),
            serialization=".tmp/pytorch-smoke/tensor.pt",
        )
    except Exception as exc:
        report.update(
            error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc()
        )
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    path = root / "reports/pytorch-smoke.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
