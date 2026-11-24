"""The program on screen and the numbers on screen are the same thing: examples/mnist_blocks.py,
run in PyTorch from voxanim.study_course's initial weights on its first batch, takes the same
SGD step and the same Adam step as the numpy twin (loss, every gradient, every new weight
and Adam's m and v, to 1e-5).

The PyTorch side runs the script's own lines: the regions "model", "loss", "optimizer" and
"step" are read with voxanim.blocks.code_lines and executed as they are. Skipped when torch
is not installed (it is not a dependency of voxanim) or MNIST is not downloaded."""

from __future__ import annotations

import types

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks
from voxanim import study_course as S

torch = pytest.importorskip("torch", reason="torch is not installed: the equivalence of the "
                                            "script and its numpy twin is checked with it")

TOL = 1e-5


def run_script(regions, ns):
    """Execute the script's regions, in order, in the namespace ``ns``."""
    for name in regions:
        exec("\n".join(blocks.code_lines(name)), ns)        # the script's own lines
    return ns


def namespace(argv=()):
    """What the script's imports and device region give its later lines."""
    return dict(torch=torch, nn=torch.nn, SGD=torch.optim.SGD, Adam=torch.optim.Adam,
                sys=types.SimpleNamespace(argv=["mnist_blocks.py", *argv]), device="cpu")


def one_step(argv=()):
    """The script's model with the twin's initial weights, one step of its loop on the
    twin's first batch: ``(loss, grads, new weights, optimizer)``."""
    ns = run_script(("model", "loss", "optimizer"), namespace(argv))
    model = ns["model"]
    keys = list(model.state_dict())
    assert tuple(keys) == S.KEYS                    # the same names, in the same order
    model.load_state_dict({k: torch.tensor(v) for k, v in S.init_params().items()})
    b = S.batch(0)
    ns["x"], ns["y"] = torch.tensor(b["x"]), torch.tensor(b["y"])
    run_script(("step",), ns)
    grads = {k: p.grad.detach().numpy().copy() for k, p in model.named_parameters()}
    new = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    return float(ns["loss"].item()), grads, new, ns["optimizer"]


def close(a, b):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=0,
                               atol=TOL)


@needs_mnist
def test_one_sgd_step_matches_the_numpy_twin():
    loss, grads, new, optimizer = one_step()
    assert type(optimizer).__name__ == "SGD" and optimizer.defaults["lr"] == S.LR
    assert abs(loss - S.losses("sgd")[0]) < TOL
    want = S.grads(0)
    for k in S.KEYS:
        assert grads[k].shape == want[k].shape                  # .grad: the weight's shape
        close(grads[k], want[k])
    after = S.params(1)
    for k in S.KEYS:
        close(new[k], after[k])
    # and the step really moved the weights, by lr times the gradient
    assert max(np.abs(after[k] - S.init_params()[k]).max() for k in S.KEYS) > 1e-3


@needs_mnist
def test_one_adam_step_matches_the_numpy_twin():
    loss, grads, new, optimizer = one_step(("adam",))
    assert type(optimizer).__name__ == "Adam"
    d = optimizer.defaults
    assert (d["lr"], tuple(d["betas"]), d["eps"]) == (S.ADAM_LR, S.BETAS, S.EPS)
    assert abs(loss - S.losses("adam")[0]) < TOL
    for k in S.KEYS:
        close(grads[k], S.grads(0, "adam")[k])
        close(new[k], S.params(1, "adam")[k])
    state = S.adam_state(1)
    model_params = dict(zip(S.KEYS, optimizer.param_groups[0]["params"]))
    for k in S.KEYS:
        st = optimizer.state[model_params[k]]
        close(st["exp_avg"].numpy(), state["m"][k])
        close(st["exp_avg_sq"].numpy(), state["v"][k])


@needs_mnist
def test_the_untrained_model_scores_the_batch_as_the_twin_does():
    ns = run_script(("model", "loss"), namespace())
    model = ns["model"]
    model.load_state_dict({k: torch.tensor(v) for k, v in S.init_params().items()})
    b = S.batch(0)
    with torch.no_grad():
        logits = model(torch.tensor(b["x"])).numpy()
    a = S.activations(0)
    close(logits, a["logits"])
    assert a["x"].shape == (64, 784) and logits.shape == (64, 10)


def test_torch_initialises_linear_layers_in_the_same_range():
    """The twin's initialisation is PyTorch's: weights and biases within 1/sqrt(fan_in)."""
    torch.manual_seed(0)
    ns = run_script(("model",), namespace())
    ours = S.init_params()
    for k, v in ns["model"].state_dict().items():
        fan_in = v.shape[-1] if k.endswith("weight") else ours[k.replace("bias", "weight")].shape[1]
        bound = 1 / np.sqrt(fan_in)
        assert float(v.abs().max()) <= bound and np.abs(ours[k]).max() <= bound
        assert float(v.abs().max()) > 0.9 * bound and np.abs(ours[k]).max() > 0.9 * bound
        assert tuple(v.shape) == ours[k].shape


def test_what_the_script_prints_is_recorded():
    """print(model) and print(model[1].weight.grad.shape), run by torch, write exactly the
    text voxanim.study_course keeps for the animations; model.train() is its own region."""
    import contextlib
    import io

    ns = run_script(("model", "loss", "optimizer"), namespace())
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_script(("print-model",), ns)
    assert out.getvalue().rstrip("\n") == S.PRINTED_MODEL
    ns["x"] = torch.randn(64, 1, 28, 28)
    ns["y"] = torch.randint(0, 10, (64,))
    run_script(("train-mode", "step"), ns)
    assert ns["model"].training
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        run_script(("grad",), ns)
    assert out.getvalue().strip() == S.PRINTED_GRAD
    assert tuple(ns["model"][1].weight.grad.shape) == S.params(0)["1.weight"].shape
