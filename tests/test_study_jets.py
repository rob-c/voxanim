"""voxanim.study_jets: the made-up jets, the two models' arithmetic, and the measured claims.
No dataset is needed: the jets are made up by the script's recipe."""

from __future__ import annotations

import numpy as np

from voxanim import study_jets as J


# ---------------------------------------------------------------------- the made-up jets
def test_the_jets_are_padded_sorted_lists_with_a_mask():
    X, M, y = J.make_jets(500, 11)
    assert X.shape == (500, 20, 3) and X.dtype == np.float32 and M.shape == (500, 20)
    n = M.sum(1)
    assert n.min() >= 6 and n.max() <= 20
    assert 0.4 < y.mean() < 0.6                                  # half and half
    for x, m in zip(X, M, strict=True):
        k = int(m.sum())
        assert m[:k].all() and not m[k:].any()                   # real rows first, pads last
        assert not x[k:].any()                                   # pads are zeros
        z, de, dp = x[:k, 0], x[:k, 1], x[:k, 2]
        assert (np.diff(z) <= 0).all()                           # sorted by z, hardest first
        assert abs(z.sum() - 1) < 1e-5                           # momentum fractions add to 1
        # centred (z-weighted) and turned: the spread runs along eta, hardest on the + side
        assert abs(z @ de) < 1e-5 and abs(z @ dp) < 1e-5
        assert z @ de ** 2 >= z @ dp ** 2 - 1e-6
        assert abs(z @ (de * dp)) < 1e-5
        assert de[0] >= -1e-6


def test_two_prongs_are_wider_than_one():
    X, _M, y = J.make_jets(2000, 12)
    width = (X[..., 0] * np.hypot(X[..., 1], X[..., 2])).sum(1)
    assert np.median(width[y == 1]) > np.median(width[y == 0])


def test_the_shuffle_keeps_each_jets_particles_and_its_pads():
    X, M, _y = J.make_jets(200, 13)
    Xs = J.shuffled(X, M, np.random.default_rng(0))
    assert not np.array_equal(Xs, X)
    for x, xs, m in zip(X, Xs, M, strict=True):
        k = int(m.sum())
        assert sorted(map(tuple, x[:k])) == sorted(map(tuple, xs[:k]))
        assert not xs[k:].any()
    np.testing.assert_array_equal(J.sorted_again(Xs, M), X)     # sorting undoes it exactly


# ---------------------------------------------------------------------- the models
def test_the_models_have_the_scripts_sizes():
    P = J.init()
    assert tuple(P["mlp"]) == J.MLP_KEYS and tuple(P["phi"]) == J.PHI_KEYS
    assert P["mlp"]["1.weight"].shape == (64, 60) and P["mlp"]["5.weight"].shape == (1, 64)
    assert P["phi"]["0.weight"].shape == (32, 3) and P["rho"]["2.weight"].shape == (1, 32)
    assert J.count(P["mlp"]) == 60 * 64 + 64 + 64 * 64 + 64 + 64 + 1 == 8129
    assert J.count(P["phi"]) + J.count(P["rho"]) == (3 * 32 + 32 + 32 * 32 + 32) + (
        32 * 32 + 32 + 32 + 1) == 2273


def test_deep_sets_cannot_see_the_order_in_exact_arithmetic():
    # in float64 the sums of these few numbers are exact whatever the order
    P = {k: {n: v.astype(np.float64) for n, v in p.items()} for k, p in J.init(3).items()}
    X, M, _y = J.make_jets(100, 14)
    Xs = J.shuffled(X, M, np.random.default_rng(1)).astype(np.float64)
    a = J.ds_logits(P["phi"], P["rho"], X.astype(np.float64), M)
    b = J.ds_logits(P["phi"], P["rho"], Xs, M)
    assert np.abs(a - b).max() < 1e-12
    # without the mask, each pad would add phi(0), which is not 0
    phi0 = np.maximum(J._two(P["phi"], np.zeros((1, 3)))[0], 0)
    assert np.abs(phi0).sum() > 0
    # the MLP sees the order
    m = J.mlp_logits(P["mlp"], X.astype(np.float64)) - J.mlp_logits(P["mlp"], Xs)
    assert np.abs(m).max() > 1e-3


def _numeric(f, P, key, idx, h=1e-6):
    old = P[key][idx]
    P[key][idx] = old + h
    up = f()
    P[key][idx] = old - h
    down = f()
    P[key][idx] = old
    return (up - down) / (2 * h)


def test_the_gradients_are_the_slopes():
    X, M, y = (a[:16] for a in J.make_jets(16, 15))
    X = X.astype(np.float64)
    P = {k: {n: v.astype(np.float64) for n, v in p.items()} for k, p in J.init(4).items()}
    mlp, phi, rho = P["mlp"], P["phi"], P["rho"]

    def bce(logit):
        return float((np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit))))
                     .mean())
    _l, G = J.mlp_grads(mlp, X, y)
    for key, idx in (("1.weight", (3, 7)), ("3.bias", (5,)), ("5.weight", (0, 9))):
        num = _numeric(lambda: bce(J.mlp_logits(mlp, X)), mlp, key, idx)
        assert abs(num - G[key][idx]) < 1e-5
    _l, Gp, Gr = J.ds_grads(phi, rho, X, M, y)
    for P_, G_, key, idx in ((phi, Gp, "0.weight", (4, 1)), (phi, Gp, "2.bias", (6,)),
                             (rho, Gr, "0.weight", (2, 3)), (rho, Gr, "2.bias", (0,))):
        num = _numeric(lambda: bce(J.ds_logits(phi, rho, X, M)), P_, key, idx)
        assert abs(num - G_[key][idx]) < 1e-5


# ---------------------------------------------------------------------- the measured claims
def test_the_measured_claims():
    r = J.results()
    right = r["right"]
    assert right["mlp"] > 0.85 and right["ds"] > 0.85               # both learn the task
    assert right["mlp_shuffled"] < right["mlp"] - 0.05               # shuffled: the MLP drops
    assert right["ds_shuffled"] == right["ds"]                       # Deep Sets: not at all
    assert right["mlp_again"] == right["mlp"]                        # sorted again: back
    assert r["mlp_again_same"] == r["n_test"] == r["sorted_again_same"] == 4000
    assert r["ds"]["changed"] == 0 and r["ds"]["max_moved"] < 1e-6  # rounding only
    assert r["mlp"]["changed"] > 0.1 * r["n_test"]
    assert r["mlp"]["median_moved"] < 0.1 < r["mlp"]["mean_moved"]   # most move little
    assert 0 < r["ds"]["bitwise_same"] < r["n_test"]                 # float32: last bits move
    for m in J.MODELS:
        first, last = r["loss"][m]
        assert 0.68 < first < 0.72 and last < 0.3                    # from ln 2 down
    assert r["train_right"]["mlp"] - right["mlp"] < 0.05             # not badly overfitted
    assert len(J.run()["loss_mlp"]) == len(J.run()["loss_ds"]) == J.STEPS
