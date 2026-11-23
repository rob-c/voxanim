"""The experiment behind animation 177 (voxanim.study_bridge_loss): the identities between
losses and likelihoods hold exactly on cases with known answers, the gradients the
models are trained with are right, and what the study measures on MNIST holds as a
relative claim (a retrain elsewhere moves the last digits)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import study_bridge_loss as S

NAME = "every-loss-is-a-likelihood"


def points(seed=0, n=9):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, n))
    s = rng.uniform(0.05, 0.2, n)
    return x, 0.3 + 0.8 * x + rng.normal(0.0, s), s


# ---------------------------------------------------------------------- Gaussian errors
def test_minus_twice_log_likelihood_is_chi_square_plus_a_constant():
    x, y, s = points()
    a = np.linspace(-1.0, 1.0, 7)
    b = np.linspace(0.0, 2.0, 7)
    const = np.log(2 * np.pi * s ** 2).sum()
    np.testing.assert_allclose(-2 * S.gauss_loglik(a, b, x, y, s), S.chi2(a, b, x, y, s) + const,
                               rtol=1e-12, atol=1e-10)


def test_the_weighted_line_is_exact_on_a_line_and_minimises_chi_square():
    x, _, s = points(1)
    (a, b), _ = S.weighted_line(x, 0.25 - 1.5 * x, s)
    assert np.isclose(a, 0.25) and np.isclose(b, -1.5)
    x, y, s = points(2)
    (a, b), cov = S.weighted_line(x, y, s)
    best = S.chi2(a, b, x, y, s)
    for da, db in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
        assert S.chi2(a + da, b + db, x, y, s) > best
    assert np.isclose(S.best_a(b, x, y, s), a)         # the profile passes through the fit
    # the profile of -ln L over b has curvature 1 / sigma_b^2: its width is the error
    def nll(v):
        return -float(S.gauss_loglik(S.best_a(v, x, y, s), v, x, y, s))
    sb = np.sqrt(cov[1, 1])
    assert np.isclose(S.curvature(nll, b, sb * 0.01), 1 / sb ** 2, rtol=1e-5)
    # and -ln L rises by one half one error away
    assert np.isclose(nll(b + sb) - nll(b), 0.5, rtol=1e-9)


def test_with_one_sigma_chi_square_is_n_over_sigma_squared_times_the_mse():
    x, y, _ = points(3, 40)
    for sigma in (0.1, 0.37):
        for a, b in ((0.0, 1.0), (0.4, 0.6)):
            assert np.isclose(S.chi2(a, b, x, y, np.full(len(x), sigma)),
                              len(x) * S.mse(a, b, x, y) / sigma ** 2)


def test_bins_and_half_inks():
    x = np.array([0.03, 0.03, 0.05, 0.05, 0.07, 0.5, 0.0])
    y = np.array([1.0, 3.0, 2.0, 4.0, 9.0, 9.0, 9.0])
    bx, by, bs, bn = S.bin_means(x, y, edges=np.array([0.02, 0.04, 0.06, 0.08]), min_bin=2)
    np.testing.assert_allclose(bx, [0.03, 0.05])           # the lone 0.07 is left out
    np.testing.assert_allclose(by, [2.0, 3.0])
    np.testing.assert_allclose(bs, [1.0, 1.0])               # sd sqrt 2, over sqrt 2
    np.testing.assert_array_equal(bn, [2, 2])
    img = np.zeros((1, 28, 28))
    img[0, :, :14] = 1.0
    img[0, :14, 14:] = 0.5
    left, right = S.half_inks(img.reshape(1, 784))
    assert np.isclose(left[0], 1.0) and np.isclose(right[0], 0.25)


def test_curvature_of_a_parabola():
    assert np.isclose(S.curvature(lambda v: 3.0 * (v - 1.0) ** 2, 0.4, 1e-3), 6.0)


# ---------------------------------------------------------------------- one of ten, one of two
def test_two_classes_of_a_softmax_are_a_sigmoid():
    z = np.linspace(-30.0, 30.0, 61)
    t = (np.arange(61) % 3 == 0).astype(float)
    lp = S.log_softmax(np.stack([np.zeros_like(z), z], 1))
    cat = -lp[np.arange(61), t.astype(int)].mean()
    assert np.isclose(S.binary_cross_entropy(z, t), cat, rtol=1e-12)
    np.testing.assert_allclose(np.exp(S.log_sigmoid(z)), 1 / (1 + np.exp(-z)), rtol=1e-12)
    assert np.isfinite(S.log_sigmoid(np.array([-800.0, 800.0]))).all()


def test_cross_entropy_is_minus_the_mean_log_of_a_product():
    rng = np.random.default_rng(4)
    z = rng.normal(0, 2, (50, 10))
    y = rng.integers(0, 10, 50)
    p = np.exp(S.log_softmax(z))[np.arange(50), y]
    loss, _, _ = S.softmax_loss(np.eye(10), np.zeros(10), z, y)
    assert np.isclose(loss, -np.log(np.prod(p)) / 50)
    # a uniform guess scores ln 10
    assert np.isclose(S.softmax_loss(np.zeros((10, 10)), np.zeros(10), z, y)[0], np.log(10))


def test_gradients():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, (16, 12))
    y = rng.integers(0, 10, 16)
    P = dict(W=rng.normal(0, 0.3, (12, 10)), c=rng.normal(0, 0.3, 10))
    _, gW, gc = S.softmax_loss(P["W"], P["c"], x, y)
    fd_check(lambda Q: S.softmax_loss(Q["W"], Q["c"], x, y)[0], P, dict(W=gW, c=gc), ("W", "c"))
    t = (rng.uniform(size=16) > 0.5).astype(float)
    P = dict(w=rng.normal(0, 0.5, 12), c=np.array(0.2))
    _, gw, gc = S.logistic_loss(P["w"], P["c"], x, t)
    fd_check(lambda Q: S.logistic_loss(Q["w"], Q["c"], x, t)[0], P,
             dict(w=gw, c=np.array(gc)), ("w", "c"))


# ---------------------------------------------------------------------- measured on MNIST
@pytest.fixture(scope="module")
def study():
    return S.STUDIES[NAME]().validate()


@needs_mnist
def test_the_gaussian_fit(study):
    g = study.metadata["gauss"]
    assert g["argmin"] == g["argmax"]                       # one slope, both ways
    assert g["b_chi2_min"] == g["b_likelihood_max"] and np.isclose(g["b_chi2_min"], g["b"])
    assert g["constant_ptp"] < 1e-9                         # the same constant at every slope
    assert np.isclose(g["constant"], g["sum_ln_2pi_sigma2"])
    assert np.isclose(g["loglik_max"], -(g["chi2_min"] + g["constant"]) / 2)
    assert np.isclose(g["curvature_sigma"], g["sigma_b"], rtol=1e-4)
    # the curvature's error describes the scatter of the slope over disjoint samples
    assert abs(g["spread"] / g["mean_sigma_b"] - 1) < 0.25
    assert g["samples"] == 120 and g["points"] >= 7 and g["digits_used"] > 0.9 * g["n_digits"]
    assert 0.5 < g["median_chi2_per_dof"] < 2.5             # a line describes the bins
    assert 0.6 < g["b"] < 1.0 and abs(g["b"] - g["mean_b"]) < 3 * g["spread"]
    p0, p1 = study.panels[0].data, study.panels[1].data
    assert p1["values"].sum() <= 120 and p1["values"].sum() >= 115
    # the panel's two curves differ by the constant everywhere
    np.testing.assert_allclose(p0["ys"][1] - p0["ys"][0], g["constant"], atol=1e-9)


@needs_mnist
def test_least_squares_on_every_digit(study):
    one, g = study.metadata["one_sigma"], study.metadata["gauss"]
    assert one["n"] == 500 and one["n_test"] == 10000
    assert np.isclose(one["chi2"], one["n"])                # sigma = sqrt(MSE): chi2 = N
    assert np.isclose(one["sigma"] ** 2, one["mse_train"])
    assert abs(one["b"] - g["b"]) < 2 * g["sigma_b"]        # the two lines agree
    assert 0.7 < one["mse_test"] / one["mse_train"] < 1.4   # held out: about the same
    assert one["nll_test"] > one["nll_train"]
    d = study.panels[2].data
    x, y = d["xy"].T
    q = np.linalg.lstsq(np.stack([np.ones(500), x], 1), y, rcond=None)[0]
    np.testing.assert_allclose(q, d["line"])


@needs_mnist
def test_the_classifiers(study):
    m = study.metadata
    ten, two = m["softmax"], m["logistic"]
    assert ten["n_test"] == 10000 and ten["parameters"] == 7850
    assert np.isclose(ten["cross_entropy"], -ten["log_likelihood"] / ten["n_test"])
    assert np.isclose(ten["log10_product"], ten["log_likelihood"] / np.log(10))
    assert ten["cross_entropy"] < ten["uniform"] / 5 and ten["accuracy"] > 0.9
    assert ten["train_cross_entropy"] < ten["uniform"] / 5
    assert np.isclose(sum(ten["shown_probabilities"]), 1.0)
    assert np.isclose(ten["shown_loss"], -np.log(ten["shown_probabilities"][ten["shown_label"]]))
    assert two["n_test"] == 1991 and two["nines"] == 1009 and two["n_train"] == 11791
    assert np.isclose(two["bce"], -two["log_likelihood"] / two["n_test"])
    assert two["bce"] < two["uniform"] / 4 and two["accuracy"] > 0.93
    # the ten-class model read on two logits is a model of the same kind, about as good
    assert 0.6 < two["ten_class_bce"] / two["bce"] < 1.6
    h = study.panels[3].data
    assert np.isclose(h["ys"][1][0], np.log(10))            # every digit 1 in 10 at the start
    assert h["ys"][1][-1] < h["ys"][1][0] / 5
    assert np.isclose(h["logistic_ys"][1][0], np.log(2))
    run = S.softmax_run()
    lp = S.log_softmax(run["test_logits"])
    assert np.isclose(np.exp(lp).sum(1), 1.0).all()
