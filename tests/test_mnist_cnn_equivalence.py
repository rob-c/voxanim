"""The CNN program on screen and the numbers on screen are the same thing: examples/mnist_cnn.py,
run in PyTorch from voxanim.study_course_cnn's initial weights on the course's first batch,
takes the same SGD step as the numpy twin (loss, every gradient and every new weight, to
1e-5). Its print lines, and the errors of the broken CNN programs of
examples/blocks_parts.py, write exactly the text the twin keeps for the animations.

The PyTorch side runs the script's own lines (voxanim.blocks.code_lines). Skipped when torch
is not installed (it is not a dependency of voxanim) or MNIST is not downloaded."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from conftest import needs_mnist
from test_mnist_blocks_equivalence import TOL, close, namespace, run_script

from voxanim import study_course as S
from voxanim import study_course_cnn as C

torch = pytest.importorskip("torch", reason="torch is not installed: the equivalence of the "
                                            "CNN script and its numpy twin is checked with it")


def cnn_model(argv=()):
    ns = run_script(("cnn-model", "cnn-loss", "cnn-optimizer"), namespace(argv))
    assert tuple(ns["model"].state_dict()) == C.KEYS        # the same names, in order
    ns["model"].load_state_dict({k: torch.tensor(v) for k, v in C.init_params().items()})
    return ns


@needs_mnist
def test_one_sgd_step_of_the_cnn_matches_the_numpy_twin():
    ns = cnn_model()
    b = S.batch(0)
    ns["x"], ns["y"] = torch.tensor(b["x"]), torch.tensor(b["y"])
    run_script(("cnn-step",), ns)
    model = ns["model"]
    assert abs(float(ns["loss"].item()) - C.losses()[0]) < TOL
    want = C.grads(0)
    for k, p in model.named_parameters():
        assert tuple(p.grad.shape) == want[k].shape
        close(p.grad.numpy(), want[k])
    after = C.params(1)
    for k, v in model.state_dict().items():
        close(v.numpy(), after[k])


@needs_mnist
def test_every_block_gives_the_twins_numbers_and_shapes():
    ns = cnn_model()
    model = ns["model"]
    x = torch.tensor(S.batch(0)["x"])
    a = C.forward(C.init_params(), S.batch(0)["x"])
    keys = ("z1", "h1", "p1", "z2", "h2", "p2", "flat", "logits")
    shapes = dict(C.shapes())
    with torch.no_grad():
        h = x
        for block, k, (name, shape) in zip(model, keys, C.shapes()[1:], strict=True):
            h = block(h)
            assert type(block).__name__ == name and tuple(h.shape) == shape
            close(h.numpy(), a[k])
    assert shapes["input"] == (64, 1, 28, 28) and shapes["Linear"] == (64, 10)


def test_torch_initialises_the_cnn_in_the_same_range():
    torch.manual_seed(0)
    ns = run_script(("cnn-model",), namespace())
    ours = C.init_params()
    for k, v in ns["model"].state_dict().items():
        w = ours[k.replace("bias", "weight")]
        bound = 1 / np.sqrt(np.prod(w.shape[1:]))
        assert tuple(v.shape) == ours[k].shape
        assert float(v.abs().max()) <= bound and np.abs(ours[k]).max() <= bound
        if v.numel() >= 72:                         # enough numbers to come near the bound
            assert float(v.abs().max()) > 0.9 * bound and np.abs(ours[k]).max() > 0.9 * bound
    assert sum(p.numel() for p in ns["model"].parameters()) == C.count() == 5258


def test_what_the_cnn_script_prints_is_recorded():
    ns = run_script(("cnn-model", "cnn-loss", "cnn-optimizer"), namespace())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_script(("cnn-print-model",), ns)
    assert out.getvalue().rstrip("\n") == C.PRINTED_MODEL
    ns["x"] = torch.randn(64, 1, 28, 28)
    ns["y"] = torch.randint(0, 10, (64,))
    run_script(("cnn-train-mode", "cnn-step"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_script(("cnn-grad",), ns)
    assert out.getvalue().strip() == C.PRINTED_GRAD


def test_the_broken_cnn_programs_fail_as_recorded():
    """Each broken program of examples/blocks_parts.py, run by torch on a batch shaped like
    the digits', raises exactly the error the twin keeps; the fixed line gives [1, 10]."""
    torch.manual_seed(0)
    nn = torch.nn
    ns = dict(torch=torch, nn=nn, x=torch.randn(64, 1, 28, 28),
              cnn=nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
                                nn.Flatten(), nn.Linear(400, 10)))
    for name, message in C.ERRORS.items():
        with pytest.raises(RuntimeError) as error:
            got = run_script((name,), dict(ns))
            got["model"](got["x"])                  # the in_features model fails when run
        assert str(error.value) == message, name
    got = run_script(("cnn-fix-no-batch",), dict(ns))
    assert tuple(got["scores"].shape) == C.FIXED_SHAPES["cnn-fix-no-batch"]
