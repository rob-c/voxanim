"""examples/jets_as_sets.py's own lines, in PyTorch, from voxanim.study_jets's starting weights
on its first batch: both models' loss, gradients and first Adam step match the numpy twin (to
1e-5); the made-up jets and the shuffle are the script's, number for number; and the measuring
lines, run on the twin's trained weights, give the twin's answers."""

from __future__ import annotations

import runpy

import numpy as np
import pytest
from test_mnist_blocks_equivalence import close

from voxanim import blocks
from voxanim import study_jets as J

torch = pytest.importorskip("torch", reason="torch is not installed: the jets script is "
                                            "checked with it")
SCRIPT = blocks.EXAMPLES / "jets_as_sets.py"


def run(regions, ns):
    for r in regions:
        exec("\n".join(blocks.code_lines(r)), ns)
    return ns


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(SCRIPT), run_name="test")


def models(script):
    """The models as the script's regions on screen build them, in a fresh namespace."""
    ns = dict(torch=torch, nn=torch.nn, F=torch.nn.functional, bce=script["bce"])
    run(("js-mlp", "js-deepsets", "js-sum"), ns)
    assert list(ns["mlp"].state_dict()) == list(J.MLP_KEYS)
    assert list(ns["phi"].state_dict()) == list(J.PHI_KEYS)
    assert list(ns["rho"].state_dict()) == list(J.RHO_KEYS)
    ns["flat_mlp"] = lambda x, mask: ns["mlp"](x)
    return ns


def load(module, P):
    module.load_state_dict({k: torch.tensor(v) for k, v in P.items()})


def test_the_settings_and_the_made_up_jets_are_the_scripts(script):
    assert (script["MAX"], script["N_TRAIN"], script["N_TEST"]) == (J.MAX, J.N_TRAIN, J.N_TEST)
    assert (script["STEPS"], script["BATCH"], script["LR"]) == (J.STEPS, J.BATCH, J.LR)
    X, M, y = script["make_jets"](300, J.TRAIN_SEED)
    tx, tm, ty = (a[:300] for a in J.train_set())
    np.testing.assert_array_equal(X, tx)
    np.testing.assert_array_equal(M, tm)
    np.testing.assert_array_equal(y, ty)
    Xt, Mt, _yt, Xs = J.test_set()
    np.testing.assert_array_equal(script["make_jets"](J.N_TEST, J.TEST_SEED)[0], Xt)
    np.testing.assert_array_equal(script["shuffled"](Xt, Mt, np.random.default_rng(
        J.SHUFFLE_SEED)), Xs)
    pick = np.random.default_rng(J.BATCH_SEED)
    for step in range(3):
        np.testing.assert_array_equal(pick.integers(0, J.N_TRAIN, J.BATCH), J.batch_indices(step))


def test_the_on_screen_lines_count_and_pad(script):
    ns = models(script)
    assert sum(p.numel() for p in ns["mlp"].parameters()) == J.results()["count"]["mlp"]
    assert (sum(p.numel() for p in ns["phi"].parameters())
            + sum(p.numel() for p in ns["rho"].parameters())) == J.results()["count"]["ds"]
    ns = run(("js-pad",), dict(np=np, MAX=J.MAX))
    p = np.array([[0.2, 0.1, 0.0], [0.7, 0.0, 0.1], [0.1, -0.3, 0.2]])
    x, mask = ns["as_list"](p)
    assert x.shape == (20, 3) and mask.tolist() == [True] * 3 + [False] * 17
    np.testing.assert_array_equal(x[:3, 0], np.float32([0.7, 0.2, 0.1]))   # hardest first
    assert not x[3:].any()


def one_step(script, which):
    ns = models(script)
    start = J.init()
    load(ns["mlp"], start["mlp"])
    load(ns["phi"], start["phi"])
    load(ns["rho"], start["rho"])
    X, M, y = J.train_set()
    if which == "mlp":
        model, params = ns["flat_mlp"], list(ns["mlp"].parameters())
    else:
        model, params = ns["deep_sets"], [*ns["phi"].parameters(), *ns["rho"].parameters()]
    ns.update(model=model, X=torch.tensor(X), M=torch.tensor(M), y=torch.tensor(y),
              i=torch.as_tensor(J.batch_indices(0)), opt=torch.optim.Adam(params, lr=J.LR))
    # the step's lines, in two parts, so the gradients can be read before the step
    lines = blocks.code_lines("js-step")
    assert lines[-1] == "opt.step()"
    exec("\n".join(lines[:-1]), ns)
    return ns


def check_step(ns, module, prefix):
    for k, p in module.named_parameters():
        close(p.grad.numpy(), J.weights(prefix + "G0")[k])
    ns["opt"].step()
    for k, v in module.state_dict().items():
        # Adam's first step moves a weight by about lr whatever its gradient's size: where the
        # gradient is next to nothing rounding decides, so compare the rest
        g = J.weights(prefix + "G0")[k]
        real = np.abs(g) > 1e-6
        close(v.numpy()[real], J.weights(prefix + "1")[k][real])
        assert np.abs(v.numpy() - J.weights(prefix + "1")[k]).max() <= 2 * J.LR


def test_one_mlp_step(script):
    ns = one_step(script, "mlp")
    assert abs(ns["loss"].item() - J.run()["loss_mlp"][0]) < 1e-5
    check_step(ns, ns["mlp"], "mlp")


def test_one_deep_sets_step(script):
    ns = one_step(script, "ds")
    assert abs(ns["loss"].item() - J.run()["loss_ds"][0]) < 1e-5
    # the step touches both phi and rho; the step is taken once, after both are checked
    for k, p in ns["phi"].named_parameters():
        close(p.grad.numpy(), J.weights("phiG0")[k])
    check_step(ns, ns["rho"], "rho")
    for k, v in ns["phi"].state_dict().items():
        real = np.abs(J.weights("phiG0")[k]) > 1e-6
        close(v.numpy()[real], J.weights("phi1")[k][real])


def test_the_measuring_lines_give_the_twins_answers(script):
    ns = models(script)
    load(ns["mlp"], J.weights("mlp"))
    load(ns["phi"], J.weights("phi"))
    load(ns["rho"], J.weights("rho"))
    Xt, Mt, yt, Xs = J.test_set()
    p = J.probs()
    T = torch.as_tensor
    for model, key in ((ns["flat_mlp"], "mlp"), (ns["deep_sets"], "ds")):
        run_ns = dict(torch=torch, model=model, X=T(Xt), M=T(Mt), y=T(yt), Xs=T(Xs))
        run(("js-measure",), run_ns)
        close(run_ns["a"].numpy(), p[key])
        close(run_ns["b"].numpy(), p[key + "_shuffled"])
        assert abs(run_ns["right"].item() - J.results()["right"][key]) < 1e-6
        moved = run_ns["moved"].numpy()
        if key == "ds":
            assert moved.max() < 1e-6                   # float rounding only
        else:
            assert abs(moved.mean() - J.results()["mlp"]["mean_moved"]) < 1e-5


def test_sorted_again_is_the_same_list(script):
    Xt, Mt, _yt, Xs = J.test_set()
    ns = dict(script, Xs=Xs, Mt=Mt, Xt=Xt)
    exec("\n".join(blocks.code_lines("js-again")).replace("print(", "same = ("), ns)
    assert bool(ns["same"])
