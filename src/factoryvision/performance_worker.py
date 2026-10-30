"""One fixed synthetic worker; all images are generated here, never received or loaded."""

import io
import os
import sys
import time
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, PngImagePlugin

from factoryvision import performance_baseline as contract
from factoryvision.inference.batch import png
from factoryvision.inference.service import preprocess
from factoryvision.inference.uploads import decode_upload, readonly
from factoryvision.inference.visualization import panels
from factoryvision.robustness.contracts import Parent, ParentID
from factoryvision.robustness.runner import AnalyticalAdapter, _generate

# Finite internal integration-test seams; never user options or wire commands.
_FAULTS = (
    "none",
    "hang",
    "exit",
    "malformed",
    "oversized",
    "trailing",
    "deny_io",
    "deny_model",
    "profile_display",
)


def generate(side):
    contract.case(side, 1)
    pixels = _generate(Parent(ParentID.FOURTH, side, side))
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return pixels, buffer.getvalue()


def pipeline(side, clock=time.perf_counter_ns):
    contract.case(side, 1)
    start = previous = clock()
    elapsed = []

    def mark():
        nonlocal previous
        now = clock()
        elapsed.append((now - previous) / 1e6)
        previous = now

    pixels, payload = generate(side)
    mark()
    image = decode_upload(payload)
    mark()
    tensor = preprocess(image)
    mark()
    original = readonly(np.asarray(image))
    copied = tensor.clone()
    mark()
    analytical = AnalyticalAdapter().evaluate(original)
    raw_map = readonly(analytical.raw_map.astype(np.float32))
    mark()
    view = panels(
        SimpleNamespace(original=original, spatial=raw_map, reconstruction=None, reference=None)
    )
    previews = tuple(png(view[key], 384) for key in ("original", "heatmap", "overlay"))
    mark()
    scalar = contract.scalar_payload(
        side, analytical.score, view["ceiling"], sum(map(len, previews))
    )
    mark()
    elapsed.append((previous - start) / 1e6)
    # Test-visible observations stay in-process; only elapsed scalars cross IPC.
    return elapsed, (pixels, image, tensor, original, copied, analytical, raw_map, view, scalar)


def measure(side):
    for _ in range(contract.WARMUPS):
        pipeline(side)
    samples = []
    for _ in range(contract.SAMPLES):
        row, observations = pipeline(side)
        samples.append(row)
        del observations
    return samples


def _deny(*args, **kwargs):
    raise RuntimeError("Synthetic worker operation denied")


def _guard_reads():
    # The unchanged decoder rereads two PUBLIC packaged exclusion resources per call.
    root = Path(__file__).resolve().parent / "inference"
    allowed = {
        str((root / name).resolve())
        for name in ("protected_inputs.json", "metalparts_protected_inputs.json")
    }

    def audit(event, args):
        if event == "open":
            path, mode, flags = args
            if (
                type(path) is not str
                or str(Path(path).resolve()) not in allowed
                or (mode is not None and mode not in ("r", "rb"))
                or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                _deny()
        elif event in (
            "os.listdir",
            "os.scandir",
            "os.system",
            "subprocess.Popen",
            "pickle.find_class",
        ) or event.startswith("socket."):
            _deny()

    return audit


def main(import_start):
    contract.require(len(sys.argv) == 4)
    side, repetition, fault = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    contract.case(side, repetition)
    contract.require(fault in _FAULTS)
    contract.require(PngImagePlugin is not None)
    Image.preinit()
    # Resolve the package resource reader while imports are still permitted.
    # The measured decoder retains its normal per-call public-resource reads.
    files("factoryvision.inference")
    if fault == "profile_display":
        from factoryvision import performance_display_profile

        groups = performance_display_profile.prepare()
    torch.set_num_threads(contract.THREADS)
    torch.set_num_interop_threads(1)
    # Reuse tensor preprocessing without permitting models, checkpoints or downloads.
    torch.load = torch.serialization.load = torch.jit.load = _deny
    torch.nn.Module.__init__ = torch.nn.Module._call_impl = _deny
    torch.hub.load = torch.hub.load_state_dict_from_url = torch.hub.download_url_to_file = _deny
    runtime = dict(
        python=".".join(map(str, sys.version_info[:3])),
        torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),
        numpy=np.__version__,
        pillow=PIL.__version__,
        intraop=torch.get_num_threads(),
        interop=torch.get_num_interop_threads(),
    )
    sys.addaudithook(_guard_reads())
    writer = sys.stdout.buffer

    def send(value):
        writer.write(contract.frame(value))
        writer.flush()

    send(
        dict(
            contract.envelope("ready", side, repetition),
            runtime=runtime,
            import_ms=(time.perf_counter() - import_start) * 1000,
        )
    )
    if fault == "hang":
        while True:
            time.sleep(0.05)
    if fault == "exit":
        return 31
    if fault == "deny_io":
        with open("unapproved.synthetic", "wb"):
            pass
    if fault == "deny_model":
        torch.nn.Linear(1, 1)
    if fault in ("malformed", "oversized"):
        writer.write(b"\xff\xff\xff\xff" if fault == "oversized" else b"\x00\x00\x00\x01{")
        writer.flush()
        return 31
    result = dict(
        contract.envelope("result", side, repetition),
        warmups=contract.WARMUPS,
        samples_ms=measure(side),
    )
    if fault == "profile_display":
        original, _ = generate(side)
        raw = readonly(AnalyticalAdapter().evaluate(original).raw_map.astype(np.float32))
        record = SimpleNamespace(
            original=original, spatial=raw, reconstruction=None, reference=None
        )
        result["display_profile_ms"] = performance_display_profile.measure(record, groups)
    contract.validate_worker(result, "result", side, repetition)
    send(result)
    if fault == "trailing":
        send(result)
    return 0
