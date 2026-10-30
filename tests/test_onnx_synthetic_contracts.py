"""Offline probe contracts; optional ONNX checks never add default dependencies."""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from factoryvision.models.autoencoder import ConvAutoencoder

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/onnx_synthetic_probe.py"
spec = importlib.util.spec_from_file_location("onnx_synthetic_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
HAS_ONNX = importlib.util.find_spec("onnx") is not None


@pytest.fixture(autouse=True)
def small_cpu(monkeypatch):
    torch.set_num_threads(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint/model downloads are not part of the synthetic probe.")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", forbidden)


def test_original_modules_deterministic_parameters_and_rng_preserved():
    before = torch.random.get_rng_state().clone()
    original, wrapper = probe.build_pair()
    second, _ = probe.build_pair()
    assert torch.equal(before, torch.random.get_rng_state())
    assert type(original) is ConvAutoencoder and original.resolution == 128
    assert wrapper.encoder is original.encoder and wrapper.decoder is original.decoder
    assert list(wrapper.parameters()) == list(original.parameters())
    assert not original.training and not wrapper.training
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in original.parameters())
    assert probe.parameter_digest(original) == probe.parameter_digest(second)
    assert probe.WEIGHT_CLASS == "randomly_initialized_synthetic_weights"


@pytest.mark.parametrize("name", ["zeros", "ones", "ramp", "checker", "random"])
def test_exact_wrapper_equivalence_and_repeatable_inputs(name):
    original, wrapper = probe.build_pair()
    image = probe.inputs()[name]
    assert image.shape == probe.SHAPE and image.dtype == torch.float32
    assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
    assert torch.equal(image, probe.inputs()[name])
    with torch.inference_mode():
        expected = original(image)
        assert torch.equal(expected, wrapper(image))
        assert torch.equal(expected, original(image))


@pytest.mark.parametrize(
    "case",
    ["rank", "empty", "channels", "resolution", "dtype", "device", "nan", "inf", "low", "high"],
)
def test_original_validation_still_rejects_invalid_input(case):
    image = torch.zeros(probe.SHAPE)
    if case == "rank":
        image = image[0]
    elif case == "empty":
        image = image[:0]
    elif case == "channels":
        image = image[:, :1]
    elif case == "resolution":
        image = image[:, :, :64]
    elif case == "dtype":
        image = image.double()
    elif case == "device":
        image = torch.empty(probe.SHAPE, device="meta")  # No GPU creation.
    else:
        image.fill_({"nan": float("nan"), "inf": float("inf"), "low": -0.1, "high": 1.1}[case])
    original, _ = probe.build_pair()
    with pytest.raises(ValueError):
        original(image)


def test_wrapper_rejects_different_architecture():
    class Subclass(ConvAutoencoder):
        pass

    for model in (ConvAutoencoder(32), torch.nn.Identity(), Subclass(128)):
        with pytest.raises(ValueError):
            probe.TensorOnly(model)


@pytest.mark.parametrize(
    "event",
    [
        "socket.connect",
        "socket.getaddrinfo",
        "socket.sendto",
        "subprocess.Popen",
        "os.system",
        "pickle.find_class",
    ],
)
def test_audit_denies_network_process_and_deserialization(event, tmp_path):
    with pytest.raises(PermissionError):
        probe.audit_policy(event, (), (tmp_path,), tmp_path)


@pytest.mark.parametrize(
    "event", ["open", "os.mkdir", "os.remove", "os.rmdir", "os.rename", "os.listdir", "os.scandir"]
)
def test_audit_denies_filesystem_outside_roots(event, tmp_path):
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    args = (str(outside), "r", os.O_RDONLY) if event == "open" else (str(outside), str(inside))
    with pytest.raises(PermissionError):
        probe.audit_policy(event, args, (inside,), inside)


def test_audit_read_only_dependencies_and_no_weights(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    probe.audit_policy("open", (str(source / "safe.py"), "r", os.O_RDONLY), (source,), output)
    probe.audit_policy("open", (str(output / "result.json"), "w", os.O_WRONLY), (source,), output)
    for path, flag in ((source / "safe.py", os.O_WRONLY), (source / "best.pt", os.O_RDONLY)):
        with pytest.raises(PermissionError):
            probe.audit_policy("open", (str(path), None, flag), (source,), output)


@pytest.mark.parametrize("event", ["os.link", "os.symlink", "os.truncate", "os.chmod", "os.chown"])
def test_audit_denies_link_and_metadata_mutation(event, tmp_path):
    with pytest.raises(PermissionError):
        probe.audit_policy(event, (str(tmp_path),), (tmp_path,), tmp_path)


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "repeat", "tolerance"])
def test_parity_detects_invalid_runtime_results(fault):
    original, wrapper = probe.build_pair()

    class Session:
        calls = 0

        def run(self, names, feed):
            self.calls += 1
            actual = original(torch.from_numpy(feed["image"])).numpy().copy()
            if fault == "shape":
                return [actual[0]]
            if fault == "dtype":
                return [actual.astype(np.float64)]
            if fault == "nan":
                actual.flat[0] = np.nan
            if fault == "repeat" and self.calls % 2 == 0:
                actual += 0.1
            if fault == "tolerance":
                actual += 0.01
            return [actual]

    if fault == "tolerance":
        assert all(
            not item["within_fixed_tolerance"]
            for item in probe.parity(original, wrapper, Session()).values()
        )
    else:
        with pytest.raises(ValueError):
            probe.parity(original, wrapper, Session())


def test_default_import_has_no_onnx_requirement():
    if not HAS_ONNX:
        assert "onnx" not in sys.modules and "onnxruntime" not in sys.modules


@pytest.fixture(scope="module")
def own_graph():
    if not HAS_ONNX:
        pytest.skip("Optional: run only in the isolated ONNX environment.")
    import onnx

    _, wrapper = probe.build_pair()
    with torch.inference_mode():
        graph = torch.onnx.export(
            wrapper,
            (probe.inputs()["ramp"],),
            dynamo=True,
            verbose=False,
            opset_version=probe.OPSET,
            external_data=False,
            input_names=["image"],
            output_names=["reconstruction"],
        ).model_proto
    probe.remove_debug_metadata(graph)
    return onnx, graph


def test_optional_structural_validation_and_cpu_parity(own_graph):
    import onnxruntime as ort

    onnx, graph = own_graph
    probe.validate_graph(graph, onnx)
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        graph.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    session.disable_fallback()
    assert session.get_providers() == ["CPUExecutionProvider"]
    original, wrapper = probe.build_pair()
    assert all(
        item["within_fixed_tolerance"] for item in probe.parity(original, wrapper, session).values()
    )


@pytest.mark.parametrize(
    "fault",
    [
        "external",
        "custom_domain",
        "custom_op",
        "shape",
        "dtype",
        "path",
        "metadata",
        "function",
        "training",
        "opset",
        "tensor_attribute",
        "string_attribute",
    ],
)
def test_optional_unsafe_generated_graph_rejected(own_graph, fault):
    onnx, original = own_graph
    graph = onnx.ModelProto()
    graph.CopyFrom(original)
    if fault == "external":
        tensor = graph.graph.initializer[0]
        tensor.data_location = onnx.TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="unapproved.bin")
    elif fault == "custom_domain":
        graph.graph.node[0].domain = "unapproved"
    elif fault == "custom_op":
        graph.graph.node[0].op_type = "Unapproved"
    elif fault == "shape":
        graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    elif fault == "dtype":
        graph.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    elif fault == "path":
        graph.graph.name = "/unapproved/local/path"
    elif fault == "metadata":
        graph.metadata_props.add(key="unapproved", value="annotation")
    elif fault == "function":
        graph.functions.add(name="Unapproved")
    elif fault == "training":
        graph.training_info.add()
    elif fault == "opset":
        graph.opset_import[0].version += 1
    else:
        attr = graph.graph.node[0].attribute.add(name="unapproved")
        attr.type = (
            onnx.AttributeProto.TENSOR
            if fault == "tensor_attribute"
            else onnx.AttributeProto.STRING
        )
    with pytest.raises(ValueError):
        probe.validate_graph(graph, onnx)
