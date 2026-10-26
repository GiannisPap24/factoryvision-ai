"""Offline fabricated-data student distillation contracts and numerical checks."""

import json
import time

import pytest
import torch
from torch import nn
from torchvision.models import resnet18

from factoryvision.data.development import (
    DevelopmentDataset,
    DevelopmentReadGuard,
    synthetic_defect,
)
from factoryvision.domain.contracts import Partition
from factoryvision.models.feature_memory import FrozenResNet18
from factoryvision.models.student_teacher import (
    CompactStudent,
    SpatialTeacher,
    StudentTeacher,
    discrepancy,
)
from factoryvision.training.checkpoint import create_run, file_sha256
from factoryvision.training.engine import parameter_sha256
from factoryvision.training.student_artifacts import load_student, save_student
from factoryvision.training.student_engine import StudentConfig, epoch, score_normals, step


@pytest.fixture(scope="module", autouse=True)
def cpu():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, image):
        return (
            self.weight.expand(len(image), 128, 16, 16),
            self.weight.expand(len(image), 256, 8, 8),
        )


def model():
    return StudentTeacher(FakeTeacher(), CompactStudent())


def batch(partition="train"):
    return {
        "image": torch.rand(2, 3, 128, 128, generator=torch.Generator().manual_seed(4)),
        "partition": [partition] * 2,
        "label": torch.zeros(2, dtype=torch.int64),
        "sample_id": ["synthetic-a", "synthetic-b"],
    }


def test_real_teacher_shapes_freeze_and_student_independence():
    teacher = SpatialTeacher(FrozenResNet18(resnet18(weights=None)))
    before = parameter_sha256(teacher)
    m = StudentTeacher(teacher, CompactStudent())
    initial = parameter_sha256(m.student)
    m.train()
    assert all(not x.training for x in teacher.modules())
    target = teacher(batch()["image"])
    assert [list(x.shape) for x in target] == [[2, 128, 16, 16], [2, 256, 8, 8]]
    assert all(not x.requires_grad for x in target)
    step(m, torch.optim.Adam(m.student.parameters(), lr=0.001), batch())
    assert parameter_sha256(teacher) == before
    assert parameter_sha256(m.student) != initial
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.student.parameters())
    assert parameter_sha256(CompactStudent()) == parameter_sha256(CompactStudent())
    assert parameter_sha256(CompactStudent(1)) != parameter_sha256(CompactStudent(2))


def test_discrepancy_known_values_gradients_and_map_geometry():
    a = (torch.ones(2, 128, 16, 16), torch.ones(2, 256, 8, 8))
    b = tuple((-x).requires_grad_() for x in a)
    d = discrepancy(a, b)
    torch.testing.assert_close(d, torch.full((2, 16, 16), 4.0))
    d.mean().backward()
    assert all(x.grad is not None for x in b)
    torch.testing.assert_close(discrepancy(a, a), torch.zeros(2, 16, 16))
    zero = tuple(torch.zeros_like(x) for x in a)
    assert torch.isfinite(discrepancy(zero, zero)).all()
    m = model()
    x = batch()["image"]
    patch = m(x)
    score, spatial = m.score(x)
    torch.testing.assert_close(score, patch.flatten(1).max(1).values)
    assert spatial.shape == (2, 128, 128)
    torch.testing.assert_close(m.score(x[:1])[0], score[:1], atol=1e-6, rtol=0)
    altered, mask = synthetic_defect(x[0], "synthetic-a")
    assert mask.sum() == 256
    assert torch.isfinite(m.score(altered[None])[1]).all()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        (torch.zeros(1),),
        (torch.zeros(1), torch.zeros(1)),
        (torch.zeros(1, 128, 16, 16, dtype=torch.float64), torch.zeros(1, 256, 8, 8)),
        (torch.full((1, 128, 16, 16), float("nan")), torch.zeros(1, 256, 8, 8)),
        (torch.zeros(2, 128, 16, 16), torch.zeros(1, 256, 8, 8)),
        (torch.zeros(0, 128, 16, 16), torch.zeros(0, 256, 8, 8)),
    ],
)
def test_bad_features(value):
    with pytest.raises(ValueError):
        discrepancy(value, (torch.zeros(1, 128, 16, 16), torch.zeros(1, 256, 8, 8)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 13},
        {"epochs": True},
        {"batch_size": 17},
        {"patience": 0},
        {"threads": 5},
        {"seconds": 0},
        {"seed": -1},
        {"seed": True},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.0},
        {"learning_rate": 1},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        StudentConfig(**kwargs)


def test_default_config_and_invalid_student_input():
    assert StudentConfig().epochs == 12
    with pytest.raises(ValueError):
        CompactStudent(-1)
    with pytest.raises(ValueError):
        CompactStudent()(torch.zeros(1, 3, 64, 64))


def test_epoch_membership_order_selection_and_finite_learning():
    m = model()
    b = batch()
    ids = b["sample_id"]
    opt = torch.optim.Adam(m.student.parameters(), lr=0.001)
    initial = parameter_sha256(m.student)
    result = epoch(m, [b], ids, "train", optimizer=opt)
    assert result["updates"] == 1 and result["samples"] == 2
    assert parameter_sha256(m.student) != initial
    val = batch("validation")
    assert epoch(m, [val], ids, "validation")["updates"] == 0
    assert len(score_normals(m, [val], ids, "validation")["scores"]) == 2
    for p, optimizer in [
        ("test", None),
        ("calibration", None),
        ("train", None),
        ("validation", opt),
    ]:
        with pytest.raises(ValueError):
            epoch(m, [b], ids, p, optimizer=optimizer)
    for loader, expected in [
        ([b, b], ids),
        ([], ids),
        ([b], ["unknown"]),
        ([b], ids + ["synthetic-c"]),
    ]:
        with pytest.raises(ValueError):
            epoch(m, loader, expected, "train", optimizer=opt)
    with pytest.raises(TimeoutError):
        epoch(m, [b], ids, "train", optimizer=opt, deadline=time.perf_counter() - 1)
    reversed_val = {**val, "sample_id": list(reversed(ids))}
    with pytest.raises(ValueError):
        epoch(m, [reversed_val], ids, "validation")
    with pytest.raises(ValueError):
        step(m, opt, batch("test"))
    for p, loader in [("test", [b]), ("validation", [])]:
        with pytest.raises(ValueError):
            score_normals(m, loader, ids, p)


def test_nonfinite_loss_gradient_and_optimizer_fail_closed():
    m = model()
    b = batch()
    opt = torch.optim.Adam(m.student.parameters(), lr=0.001)
    original = m.forward
    m.forward = lambda x: original(x) * float("nan")
    with pytest.raises(ValueError, match="loss"):
        step(m, opt, b)
    with pytest.raises(ValueError, match="loss"):
        epoch(m, [batch("validation")], b["sample_id"], "validation")
    m.forward = original
    h = next(m.student.parameters()).register_hook(lambda x: x * float("nan"))
    with pytest.raises(ValueError, match="gradients"):
        step(m, opt, b)
    h.remove()
    next(m.teacher.parameters()).grad = torch.ones(1)
    with pytest.raises(ValueError, match="Teacher"):
        step(m, opt, b)
    next(m.teacher.parameters()).grad = None
    opt.step = lambda: next(m.student.parameters()).data.fill_(float("nan"))
    with pytest.raises(ValueError, match="update"):
        step(m, opt, b)


def test_invalid_scores():
    m = model()
    m.score = lambda x: (torch.full((2,), float("nan")), None)
    with pytest.raises(ValueError):
        score_normals(m, [batch("validation")], batch()["sample_id"], "validation")


def artifact(tmp_path):
    directory = create_run(tmp_path, "student")
    s = CompactStudent()
    ids = ["synthetic-a"]
    save_student(directory, s, "a" * 64, "b" * 64, ids, 1, 0.2)
    return directory, s, ids


def test_checkpoint_roundtrip_and_path_safety(tmp_path):
    d, s, ids = artifact(tmp_path)
    restored, meta = load_student(d, "a" * 64, "b" * 64, ids)
    assert parameter_sha256(restored) == parameter_sha256(s) and meta["epoch"] == 1
    for x, y in zip(restored(batch()["image"]), s(batch()["image"]), strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        create_run(tmp_path, "student")
    with pytest.raises(ValueError):
        create_run(tmp_path, "../escape")
    for manifest, recipe, fit in [
        ("c" * 64, "b" * 64, ids),
        ("a" * 64, "c" * 64, ids),
        ("a" * 64, "b" * 64, ["other"]),
    ]:
        with pytest.raises(ValueError):
            load_student(d, manifest, recipe, fit)
    with pytest.raises(ValueError):
        save_student(d, s, "bad", "b" * 64, ids, 1, 0.1)
    with pytest.raises(ValueError):
        save_student(d, s, "a" * 64, "b" * 64, ids, 13, 0.1)
    next(s.parameters()).data.fill_(float("inf"))
    with pytest.raises(ValueError):
        save_student(d, s, "a" * 64, "b" * 64, ids, 1, 0.1)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "hash",
        "teacher",
        "epoch",
        "loss",
        "keys",
        "shape",
        "dtype",
        "nonfinite",
        "oversize",
    ],
)
def test_bad_artifacts(tmp_path, change):
    d, s, ids = artifact(tmp_path)
    p = d / "best.pt"
    meta = json.loads((d / "checkpoint.json").read_text())
    if change == "missing":
        p.unlink()
    elif change == "hash":
        p.write_bytes(b"corrupt")
    elif change == "oversize":
        with p.open("wb") as f:
            f.truncate(8 * 1024**2 + 1)
    elif change in ("teacher", "epoch", "loss"):
        meta[{"teacher": "teacher_sha256", "epoch": "epoch", "loss": "validation_loss"}[change]] = {
            "teacher": "x",
            "epoch": 0,
            "loss": -1,
        }[change]
    else:
        state = s.state_dict()
        key = next(iter(state))
        if change == "keys":
            state.pop(key)
        if change == "shape":
            state[key] = torch.zeros(1)
        if change == "dtype":
            state[key] = state[key].double()
        if change == "nonfinite":
            state[key].fill_(float("nan"))
        torch.save(state, p)
        meta["checkpoint_sha256"] = file_sha256(p)
    (d / "checkpoint.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        load_student(d, "a" * 64, "b" * 64, ids)


def test_test_partition_rejected_before_any_decode(tmp_path):
    with pytest.raises(ValueError):
        DevelopmentDataset(None, tmp_path, Partition.TEST)
    # Audit-hook rejection is checked directly, without installing a global hook in pytest.
    from types import SimpleNamespace

    from factoryvision.domain.contracts import Label, OfficialSplit

    s = SimpleNamespace(
        partition=Partition.TRAIN,
        label=Label.NORMAL,
        official_split=OfficialSplit.TRAIN,
        mask_path=None,
        image_path="normal.png",
    )
    guard = DevelopmentReadGuard(tmp_path, [s])
    guard("open", (str(tmp_path / "normal.png"), "r"))
    with pytest.raises(PermissionError):
        guard("open", (str(tmp_path / "test.png"), "r"))
