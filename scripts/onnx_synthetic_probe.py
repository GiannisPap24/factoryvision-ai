"""Opt-in, random-weight CPU operator probe; never application model loading.

Run only in the reviewed disposable environment using the bounded launcher in
docs/onnx-synthetic-feasibility.md. Optional ONNX imports occur only in run().
"""

import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

if __name__ == "__main__":
    _root = Path(__file__).resolve().parents[1]
    _environment = _root / ".tmp/onnx-synthetic-5d3/env"
    if Path(sys.prefix).resolve() != _environment.resolve() or len(sys.argv) != 1:
        raise SystemExit("Use the reviewed disposable environment; no input arguments accepted.")
    sys.path.extend([str(_root / "src"), str(_root / ".venv-31410/Lib/site-packages")])

import numpy as np
import torch
from torch import nn

from factoryvision.models.autoencoder import ARCHITECTURE, ConvAutoencoder

SEED = 20261029
SHAPE = (1, 3, 128, 128)
OPSET = 18
ATOL = 1e-6
RTOL = 1e-5
WEIGHT_CLASS = "randomly_initialized_synthetic_weights"
OPERATORS = {"Conv": 4, "ConvTranspose": 4, "Relu": 7, "Sigmoid": 1}


class TensorOnly(nn.Module):
    """Export adapter for these exact modules, NOT an input validation boundary."""

    def __init__(self, original):
        super().__init__()
        if type(original) is not ConvAutoencoder or original.resolution != 128:
            raise ValueError("Only the original 128-pixel ConvAutoencoder is supported.")
        self.encoder = original.encoder
        self.decoder = original.decoder

    def forward(self, image):
        return self.decoder(self.encoder(image))


def build_pair():
    # CPU generator only; do not initialize a GPU or perturb the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(SEED)
        original = ConvAutoencoder(128).cpu().float().eval().requires_grad_(False)
    return original, TensorOnly(original).eval()


def inputs():
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    grid = torch.arange(128)
    checker = ((grid[:, None] + grid[None, :]) % 2).float()
    return {
        "zeros": torch.zeros(SHAPE),
        "ones": torch.ones(SHAPE),
        "ramp": torch.linspace(0, 1, 128).reshape(1, 1, 1, 128).expand(SHAPE).contiguous(),
        "checker": checker.reshape(1, 1, 128, 128).expand(SHAPE).contiguous(),
        "random": torch.rand(SHAPE, generator=generator),
    }


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("ascii"))
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def messages(message):
    """Traverse protobuf messages, including tensor attributes, without loading data."""
    yield message
    for field, value in message.ListFields():
        if field.type == field.TYPE_MESSAGE:
            for child in value if field.is_repeated else [value]:
                yield from messages(child)


def remove_debug_metadata(graph):
    # Exporter's stack traces can contain local paths. Strip only annotation fields.
    count = 0
    for message in messages(graph):
        for name in ("doc_string", "metadata_props"):
            if name in message.DESCRIPTOR.fields_by_name and getattr(message, name):
                message.ClearField(name)
                count += 1
    return count


def validate_graph(graph, onnx):
    """Narrow allowlist for our in-memory generated graph, not a model-file API."""

    def require(condition):
        if not condition:
            raise ValueError("Graph violates the fixed synthetic export contract.")

    require(graph.ByteSize() <= 1024 * 1024)
    require(not graph.functions and not graph.training_info)
    require([(o.domain, o.version) for o in graph.opset_import] == [("", OPSET)])
    require(not graph.graph.sparse_initializer)
    require(len(graph.graph.initializer) == 16)
    for tensor in graph.graph.initializer:
        require(tensor.data_type == onnx.TensorProto.FLOAT)
    for items, name in ((graph.graph.input, "image"), (graph.graph.output, "reconstruction")):
        require(len(items) == 1 and items[0].name == name)
        tensor_type = items[0].type.tensor_type
        require(tensor_type.elem_type == onnx.TensorProto.FLOAT)
        require(tuple(d.dim_value for d in tensor_type.shape.dim) == SHAPE)
        require(all(not d.dim_param for d in tensor_type.shape.dim))
    require(Counter(node.op_type for node in graph.graph.node) == OPERATORS)
    require(all(not node.domain for node in graph.graph.node))
    for message in messages(graph):
        if isinstance(message, onnx.TensorProto):
            require(
                not message.external_data and message.data_location != onnx.TensorProto.EXTERNAL
            )
        if isinstance(message, onnx.AttributeProto):
            require(
                message.type in (onnx.AttributeProto.INT, onnx.AttributeProto.INTS)
                or (
                    message.type == onnx.AttributeProto.STRING
                    and message.name == "auto_pad"
                    and message.s == b"NOTSET"
                )
            )
        for field, value in message.ListFields():
            if field.name in ("doc_string", "metadata_props"):
                require(not value)
            if field.type == field.TYPE_STRING:
                for item in value if field.is_repeated else [value]:
                    require(not re.search(r"[A-Za-z]:[\\/]|^[/\\]|https?://|@", item))
    onnx.checker.check_model(graph, full_check=True)


def parity(original, wrapper, session):
    results = {}
    with torch.inference_mode():
        for name, image in inputs().items():
            reference = original(image).numpy()
            wrapped = wrapper(image).numpy()
            if not np.array_equal(reference, wrapped):
                raise ValueError("Original and export adapter differ.")
            if not np.array_equal(reference, original(image).numpy()):
                raise ValueError("Original output is not repeatable.")
            actual = session.run(["reconstruction"], {"image": image.numpy()})[0]
            repeated = session.run(["reconstruction"], {"image": image.numpy()})[0]
            if actual.shape != SHAPE or actual.dtype != np.float32 or not np.isfinite(actual).all():
                raise ValueError("Invalid runtime output contract.")
            if not np.array_equal(actual, repeated):
                raise ValueError("Runtime output is not repeatable.")
            error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
            meaningful = np.abs(reference) >= 1e-3
            measurements = {
                "max_absolute_error": float(error.max()),
                "mean_absolute_error": float(error.mean()),
                "max_relative_error_abs_reference_ge_1e_3": (
                    float((error[meaningful] / np.abs(reference[meaningful])).max())
                    if meaningful.any()
                    else None
                ),
                "wrapper_exact": True,
                "pytorch_repeat_exact": True,
                "runtime_repeat_exact": True,
                "within_fixed_tolerance": bool(
                    np.allclose(actual, reference, atol=ATOL, rtol=RTOL)
                ),
            }
            results[name] = measurements
    return results


def audit_policy(event, args, read_roots, write_root):
    """Python audit boundary; native OS activity is not comprehensively traced."""
    if (event.startswith("socket.") and event != "socket.gethostname") or event in (
        "subprocess.Popen",
        "os.system",
        "os.startfile",
        "os.exec",
        "os.spawn",
        "os.chdir",
        "pickle.find_class",
    ):
        raise PermissionError("Probe network/process/deserialization operation forbidden.")
    if event in ("os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown"):
        raise PermissionError("Probe filesystem mutation is forbidden.")
    if event not in (
        "open",
        "os.mkdir",
        "os.remove",
        "os.rmdir",
        "os.rename",
        "os.listdir",
        "os.scandir",
    ):
        return
    if event == "open" and isinstance(args[0], int):
        return
    paths = args[:2] if event == "os.rename" else args[:1]
    for raw in paths:
        path = Path(raw or ".").resolve()
        write = event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename")
        if event == "open":
            write = bool(
                args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
        roots = (write_root,) if write else (*read_roots, write_root)
        if not any(path.is_relative_to(root) for root in roots):
            raise PermissionError("Probe filesystem access outside approved roots.")
        if path.suffix.lower() in (".pt", ".pth", ".ckpt", ".safetensors"):
            raise PermissionError("Research-weight access is forbidden.")


def run(output, read_roots):
    """Caller supplies only the launcher's fixed, newly created ignored directory."""
    output = output.resolve()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for name in ("TEMP", "TMP", "TORCHINDUCTOR_CACHE_DIR", "TORCH_COMPILE_DEBUG_DIR", "TORCH_HOME"):
        os.environ[name] = str(output)
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    sys.addaudithook(lambda event, args: audit_policy(event, args, read_roots, output))

    import onnx
    import onnxruntime as ort
    import onnxscript

    ort.disable_telemetry_events()
    original, wrapper = build_pair()
    with warnings.catch_warnings(record=True) as caught, torch.inference_mode():
        warnings.simplefilter("always")
        program = torch.onnx.export(
            wrapper,
            (inputs()["ramp"],),
            dynamo=True,
            verbose=False,
            opset_version=OPSET,
            input_names=["image"],
            output_names=["reconstruction"],
            external_data=False,
            report=False,
            verify=False,
            dump_exported_program=False,
            artifacts_dir=str(output),
        )
    graph = program.model_proto
    stripped = remove_debug_metadata(graph)
    validate_graph(graph, onnx)
    raw = graph.SerializeToString()
    # Never load a user-supplied graph. Runtime consumes these validated in-memory bytes.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = False
    session = ort.InferenceSession(raw, sess_options=options, providers=["CPUExecutionProvider"])
    session.disable_fallback()
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError("Unexpected runtime provider.")
    results = parity(original, wrapper, session)
    report = {
        "classification": WEIGHT_CLASS,
        "architecture": ARCHITECTURE,
        "seed": SEED,
        "parameter_sha256": parameter_digest(original),
        "shape": SHAPE,
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxscript": onnxscript.__version__,
        "onnxruntime": ort.__version__,
        "opset": OPSET,
        "ir_version": graph.ir_version,
        "operators": dict(Counter(node.op_type for node in graph.graph.node)),
        "graph_bytes": len(raw),
        "graph_sha256": hashlib.sha256(raw).hexdigest(),
        "annotation_fields_removed": stripped,
        "external_data_references": 0,
        "warning_categories": [type(w.message).__name__ for w in caught],
        "providers": session.get_providers(),
        "atol": ATOL,
        "rtol": RTOL,
        "comparisons": results,
        "pass": all(item["within_fixed_tolerance"] for item in results.values()),
    }
    # Full warning text can contain local paths; keep it only in ignored local evidence.
    (output / "warnings.txt").write_text(
        "\n".join(str(w.message) for w in caught), encoding="utf-8"
    )
    (output / "synthetic.onnx").write_bytes(raw)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise ValueError("Fixed float32 tolerance failed; do not relax it retrospectively.")


def launch():
    """Fresh child, fixed roots, minimal environment, 60-second wall-clock bound."""
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[1]
    experiment = root / ".tmp/onnx-synthetic-5d3"
    for path in (experiment, *experiment.parents):
        if path.is_symlink() or path.is_junction():
            raise ValueError("Linked experiment roots are forbidden.")
    output = Path(tempfile.mkdtemp(prefix="run-", dir=experiment))
    read_roots = [
        root / "src",
        root / "scripts",
        root / ".venv-31410/Lib/site-packages",
        experiment / "env/Lib/site-packages",
        Path(sys.base_prefix),
    ]
    code = (
        "import sys,runpy;from pathlib import Path;"
        "sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8');"
        f"sys.path.extend({[str(read_roots[0]), str(read_roots[2])]!r});"
        f"m=runpy.run_path({str(Path(__file__).resolve())!r});"
        f"m['run'](Path({str(output)!r}),tuple(map(Path,{[str(p) for p in read_roots]!r})))"
    )
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
    env.update(
        TEMP=str(output),
        TMP=str(output),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    (output / "process-stdout.txt").write_text(result.stdout, encoding="utf-8")
    (output / "process-stderr.txt").write_text(result.stderr, encoding="utf-8")
    print(result.stdout)
    if result.returncode:
        raise SystemExit("Probe failed; review ignored process-stderr.txt before any further run.")


if __name__ == "__main__":
    launch()
