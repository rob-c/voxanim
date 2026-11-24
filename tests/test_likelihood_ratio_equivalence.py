"""examples/likelihood_ratio.py's own lines, in PyTorch, against voxanim.study_likelihood_ratio:
the made-up events are the same; from the study's starting weights, on the study's first batch,
one training step gives the same loss, gradients and Adam step (both parts); the exact ratio
and the weights s / (1 - s) the script computes are the study's."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_likelihood_ratio as L

torch = pytest.importorskip("torch", reason="torch is not installed: the likelihood-ratio "
                                            "script is checked with it")


@pytest.fixture(scope="module")
def script():
    """The script's names (its functions; ``main`` not run)."""
    return runpy.run_path(str(blocks.EXAMPLES / "likelihood_ratio.py"), run_name="test")


def run(region, ns):
    exec("\n".join(blocks.code_lines(region)), ns)
    return ns


def load(net, P):
    net.load_state_dict({k: torch.tensor(v) for k, v in P.items()})


class Batch:
    """``torch`` for the region ``lr-step``: its ``randint`` gives the study's batch."""

    def __init__(self, idx):
        self.idx = idx

    def randint(self, high, size):
        assert size == (L.BATCH,)
        assert int(self.idx.max()) < high
        return torch.tensor(self.idx)


def test_the_settings_on_screen_are_the_studys():
    assert "lr=3e-4" in blocks.code_line("lr-train", "Adam(") and L.LR == 3e-4
    assert "(%d,)" % L.BATCH in blocks.code_line("lr-step", "randint")
    source = (blocks.EXAMPLES / "likelihood_ratio.py").read_text()
    assert "train(net, X, Y, %d)" % L.STEPS_ONE in source
    assert "train(judge, X, Y, %d)" % L.STEPS_TWO in blocks.code_lines("lr-judge")
    assert "        nn.Linear(32, 32), nn.ReLU()," in blocks.code_lines("lr-model")
    assert L.HIDDEN == 32


def test_the_events_are_the_studys(script):
    ns = run("lr-hypotheses", dict(script))
    ns["net"] = ns["classifier"](1)
    x0, x1, f0, f1 = L.hypotheses()
    np.testing.assert_array_equal(ns["x0"], x0)
    np.testing.assert_array_equal(ns["x1"], x1)
    X, Y = L.tensors(x0, x1)
    np.testing.assert_array_equal(ns["X"].numpy(), X)
    np.testing.assert_array_equal(ns["Y"].numpy(), Y)
    run("lr-power", ns)                                # fresh events from the same generator
    np.testing.assert_array_equal(ns["f0"], f0)
    np.testing.assert_array_equal(ns["f1"], f1)
    two = run("lr-samples", dict(script))
    sim, data, _new, _seen = L.samples()
    np.testing.assert_array_equal(two["sim"], sim)
    np.testing.assert_array_equal(two["data"], data)
    np.testing.assert_array_equal(np.asarray(two["cov"]), np.asarray(L.DATA_COV))


@pytest.mark.parametrize("part", (1, 2))
def test_one_training_step(script, part):
    if part == 1:
        r, (a, b, *_rest), seed, inputs = L.run_one(), L.hypotheses(), L.SEED_ONE, 1
    else:
        r, (a, b, *_rest), seed, inputs = L.run_two(), L.samples(), L.SEED_TWO, 2
    X, Y = L.tensors(a, b)
    idx = L.batch_indices(seed, 1, len(X))[0]
    ns = dict(script)
    net = ns["classifier"](inputs)
    assert list(net.state_dict()) == list(L.KEYS)
    start = L.weights(r, "start")
    np.testing.assert_array_equal(start["0.weight"], L.init(inputs, seed)["0.weight"])
    load(net, start)
    ns.update(net=net, X=torch.tensor(X), Y=torch.tensor(Y), torch=Batch(idx),
              opt=torch.optim.Adam(net.parameters(), lr=L.LR))
    run("lr-step", ns)
    assert abs(ns["loss"].item() - float(r["loss0"])) < 1e-5
    grads = L.weights(r, "grad0")
    for k, p in net.named_parameters():
        np.testing.assert_allclose(p.grad.numpy(), grads[k], rtol=1e-4, atol=1e-6)
    after = L.weights(r, "step1")
    for k, p in net.named_parameters():
        g = grads[k]
        sure = np.abs(g) > 1e-4 * max(float(np.abs(g).max()), 1e-12)   # Adam's sign step
        np.testing.assert_allclose(p.detach().numpy()[sure], after[k][sure], rtol=0, atol=1e-5)


def test_the_exact_ratio_and_the_weights_are_the_studys(script):
    ns = dict(script)
    net = ns["classifier"](1)
    load(net, L.weights(L.run_one(), "final"))
    ns["net"] = net
    run("lr-ratio", ns)
    g = ns["g"].numpy()[:, 0]
    np.testing.assert_allclose(g, L.GRID, atol=1e-6)
    np.testing.assert_allclose(ns["exact"].numpy()[:, 0], np.exp(L.log_ratio(L.GRID)),
                               rtol=2e-5)
    np.testing.assert_allclose(np.log(ns["learned"].numpy()[:, 0]), L.run_one()["grid_logit"],
                               atol=2e-4)
    two = run("lr-samples", dict(script))
    judge = ns["classifier"](2)
    load(judge, L.weights(L.run_two(), "final"))
    two["judge"] = judge
    run("lr-weights", two)
    np.testing.assert_array_equal(two["new"], L.samples()[2])
    w = np.exp(L.run_two()["new_logit"])
    np.testing.assert_allclose(two["w"], w, rtol=2e-4)
    run("lr-check", two)
    run("lr-sum", two)
    h = L.two()["hists"]
    np.testing.assert_array_equal(two["x_data"], h["x"]["data"])
    np.testing.assert_array_equal(two["y_data"], h["y"]["data"])
    np.testing.assert_array_equal(two["sum_data"], h["x + y"]["data"])
    np.testing.assert_allclose(two["x_sim"], h["x"]["weighted"], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(two["sum_sim"], h["x + y"]["weighted"], rtol=1e-3, atol=1e-3)
    run("lr-neff", two)
    assert abs(two["n_eff"] - L.two()["n_eff"]) < 1.0
    assert abs(two["big"] - L.two()["w_max"]) < 1e-3 * L.two()["w_max"]
