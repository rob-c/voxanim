"""The experiment behind animation 176 (voxanim.study_bridge_kl): the three quantities on
distributions with known answers, the identity CE = H + KL, the Gaussian fits on known
distributions, and every measured claim of the study as a relative one."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_bridge_kl as K
from voxanim import study_info as I


# ---------------------------------------------------------------------- no data needed
def test_entropy_kl_and_cross_entropy_on_a_known_pair():
    p, q = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    lq = np.log(q)
    assert np.isclose(K.entropy_terms(p).sum(), np.log(2))
    assert np.isclose(K.kl_terms(p, lq).sum(), 0.5 * np.log(25 / 9))
    assert np.isclose(K.cross_entropy_terms(p, lq).sum(), -0.5 * np.log(0.09))
    assert np.isclose(K.reverse_kl(lq, p), 0.9 * np.log(1.8) + 0.1 * np.log(0.2))
    # a class the target gives nothing costs nothing in H and KL, and makes KL(q || p) infinite
    p0 = np.array([1.0, 0.0])
    assert K.entropy_terms(p0).sum() == 0.0
    assert np.isclose(K.kl_terms(p0, lq).sum(), -np.log(0.9))
    assert np.isinf(K.reverse_kl(lq, p0))


def test_cross_entropy_is_entropy_plus_kl_term_by_term():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(10) * 0.3, size=200)
    p[:20, 3] = 0.0                                   # some targets with empty classes
    p /= p.sum(1, keepdims=True)
    lq = I.log_softmax(rng.normal(0, 3, (200, 10)))
    h, k, c = K.entropy_terms(p), K.kl_terms(p, lq), K.cross_entropy_terms(p, lq)
    np.testing.assert_allclose(h + k, c, atol=1e-12)
    assert (k.sum(1) >= -1e-12).all()                 # Gibbs: KL is never negative ...
    assert (k < 0).any()                              # ... though single terms can be
    assert np.allclose(K.kl_terms(p, np.log(np.maximum(p, 1e-300))).sum(1), 0.0)
    H, _, CE = K._split(p, lq)
    assert (CE >= H - 1e-12).all()                    # no model beats the target's entropy


def test_smoothed_targets():
    y = np.array([7, 0, 3])
    p = K.smoothed(y, 0.1)
    np.testing.assert_allclose(p.sum(1), 1.0)
    assert np.allclose(p[np.arange(3), y], 0.91) and np.isclose(p[0, 0], 0.01)
    h = K.entropy_terms(p[0]).sum()
    assert np.isclose(h, -0.91 * np.log(0.91) - 9 * 0.01 * np.log(0.01))


def test_logs_and_normalised_gaussians():
    a = np.array([[1.0, 2.0, 3.0], [-800.0, -801.0, -799.0]])
    np.testing.assert_allclose(K.logsumexp(a[:1]), np.log(np.exp(a[0]).sum()))
    assert np.isfinite(K.logsumexp(a)).all()
    grid = np.linspace(-10, 10, 401)
    lq = K.gaussian_log_q(grid, np.array([0.0, 2.0]), np.array([1.0, 0.5]))
    np.testing.assert_allclose(np.exp(lq).sum(1), 1.0)
    q = np.exp(lq[1])
    assert np.isclose((q * grid).sum(), 2.0) and np.isclose(np.sqrt((q * (grid - 2) ** 2).sum()),
                                                            0.5, rtol=1e-3)
    v = np.random.default_rng(1).normal(0, 1, 300)
    lp = K.kde_log_p(grid, v, 0.3)
    assert np.isclose(K.logsumexp(lp), 0.0) and np.isfinite(lp).all()
    assert K.forward_kl(lp, lp) == pytest.approx(0.0, abs=1e-12)
    assert K.reverse_kl_grid(lp, lp) == pytest.approx(0.0, abs=1e-12)


def _mixture():
    """Two well-separated humps: 600 points at 0 (width 0.5), 400 at 8 (width 1.5)."""
    rng = np.random.default_rng(2)
    v = np.concatenate([rng.normal(0, 0.5, 600), rng.normal(8, 1.5, 400)])
    grid = np.linspace(v.mean() - 8 * v.std(), v.mean() + 8 * v.std(), 1201)
    return v, grid, K.kde_log_p(grid, v)


def test_forward_kl_matches_the_moments_and_reverse_kl_picks_a_hump():
    v, grid, lp = _mixture()
    p = np.exp(lp)
    mean = (p * grid).sum()
    sd = np.sqrt((p * (grid - mean) ** 2).sum())
    mu, s, kl = K.fit_gaussian(grid, lp, "forward", v)
    assert mu == pytest.approx(mean, abs=1e-3 * sd) and s == pytest.approx(sd, rel=1e-3)
    assert kl == pytest.approx(K.forward_kl(lp, K.gaussian_log_q(grid, mu, s)))
    mu_r, s_r, kl_r = K.fit_gaussian(grid, lp, "reverse", v)
    assert abs(mu_r) < 0.3 and s_r < 1.0              # on the narrow, heavier hump
    for dm, ds in ((0.05, 1.0), (-0.05, 1.0), (0.0, 1.05), (0.0, 0.95)):
        worse = K.reverse_kl_grid(lp, K.gaussian_log_q(grid, mu_r + dm, s_r * ds))
        assert worse > kl_r                           # a minimum, not just a grid point


# ---------------------------------------------------------------------- measured on MNIST
@pytest.fixture(scope="module")
def study():
    return K.kl_divergence().validate()


@needs_mnist
def test_the_study_says_what_was_run(study):
    m = study.metadata
    assert len(study.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]
    assert m["evidence"]["exact"] and m["evidence"]["measured"]
    assert m["n"] == 10000 and m["eps"] == K.EPS and m["teacher_temperature"] == K.TEACHER_T
    assert m["teacher_seed"] != m["model_seed"]


@needs_mnist
def test_the_parts_add_up_for_every_target(study):
    rows = study.metadata["rows"]
    p, lq, lq0, _ = K.targets()
    for t in K.TARGETS:
        r = rows[t]
        assert r["identity_gap"] < 1e-9
        assert r["entropy"] + r["kl"] == pytest.approx(r["cross_entropy"], abs=1e-9)
        H, KL, _ = K._split(p[t], lq)
        assert np.isclose(KL.mean(), r["kl"]) and np.isclose(H.mean(), r["entropy"])
        # the untrained network: the same target entropy, much more KL
        H0, _, _ = K._split(p[t], lq0)
        assert np.isclose(H0.mean(), r["entropy"])
        assert r["untrained_kl"] > 2 * r["kl"]
    assert rows["one-hot"]["entropy"] == 0.0
    eps = K.EPS
    exact = -(1 - eps + eps / 10) * np.log(1 - eps + eps / 10) - 9 * eps / 10 * np.log(eps / 10)
    assert rows["smoothed"]["entropy"] == pytest.approx(exact, abs=1e-12)
    assert rows["teacher"]["entropy"] > 0.1
    # the one-hot loss is 175's mean test loss, all of it KL
    loss = I.logits_loss_and_nats().metadata
    assert rows["one-hot"]["cross_entropy"] == pytest.approx(loss["mean_nats"], rel=1e-9)
    # a soft target costs the model more than the labels it was trained on
    assert rows["smoothed"]["kl"] > rows["teacher"]["kl"] > rows["one-hot"]["kl"]


@needs_mnist
def test_one_digit_term_by_term(study):
    m = study.metadata
    sh, d = m["shown"], study.panels[0].data
    loss = I.logits_loss_and_nats().metadata
    assert sh["digit"] == loss["shown_digit"] and sh["label"] == loss["shown_label"]
    assert sh["onehot_loss"] == pytest.approx(loss["shown_loss_nats"])
    np.testing.assert_allclose(d["entropy"] + d["kl"], d["values"], atol=1e-12)
    assert sh["entropy"] + sh["kl"] == pytest.approx(sh["cross_entropy"])
    assert sh["entropy"] == pytest.approx(m["rows"]["smoothed"]["entropy"])  # same for every digit
    k = np.asarray(d["kl"])
    q, p = np.asarray(d["q"]), np.asarray(d["p"])
    assert ((k < 0) == (q > p)).all()                 # a KL term is negative where q > p
    assert sh["wrong_share"] > 0.5                    # most of it from the wrong digits


@needs_mnist
def test_kl_is_not_symmetric(study):
    rows = study.metadata["rows"]
    assert rows["one-hot"]["reverse_kl"] is None and rows["one-hot"]["reverse_infinite"] == 10000
    for t in ("smoothed", "teacher"):
        r = rows[t]
        assert r["reverse_infinite"] == 0 and r["reverse_kl"] > 0
        assert r["kl"] > 1.5 * r["reverse_kl"]
    sh = study.metadata["shown"]
    assert sh["kl"] > 2 * sh["reverse_kl"]


@needs_mnist
def test_two_humps_one_bell(study):
    hm = study.metadata["humps"]
    f, b = hm["forward"], hm["reverse"]
    # the forward fit is the distribution's own mean and spread
    assert f["mu"] == pytest.approx(hm["p_mean"], abs=1e-3 * hm["p_sd"])
    assert f["sd"] == pytest.approx(hm["p_sd"], rel=1e-3)
    # the humps are two: the valley lies between the classes, each side holds about half
    lo, hi = hm["class_mean"]["1"], hm["class_mean"]["0"]
    assert lo < hm["valley"] < hi and 0.3 < hm["p_mass_right"] < 0.7
    # forward covers both humps, centred where p is thin; reverse sits on the 1s
    assert f["p_at_centre"] < 0.1 and 0.3 < f["mass_right"] < 0.7
    assert abs(b["mu"] - lo) < hm["class_sd"]["1"] and b["mass_right"] < 0.01
    assert b["sd"] < f["sd"] / 3
    assert b["p_at_centre"] > 0.5
    # each fit wins on its own measure, and loses badly on the other
    assert f["kl_pq"] < b["kl_pq"] / 10 and b["kl_qp"] < f["kl_qp"] / 5
    hb = K.humps()
    grid, lp = hb["grid"], hb["lp"]
    for fn, fit in ((K.forward_kl, f), (K.reverse_kl_grid, b)):
        best = fn(lp, K.gaussian_log_q(grid, fit["mu"], fit["sd"]))
        for dm, ds in ((0.05, 1.0), (-0.05, 1.0), (0.0, 1.03), (0.0, 0.97)):
            assert fn(lp, K.gaussian_log_q(grid, fit["mu"] + dm, fit["sd"] * ds)) > best
    # the curves drawn are densities whose area is one
    d = study.panels[3].data
    dx = float(d["x"][1] - d["x"][0])
    np.testing.assert_allclose(np.asarray(d["ys"]).sum(1) * dx, 1.0, atol=1e-3)


@needs_mnist
def test_the_teacher_is_another_run(study):
    m = study.metadata
    assert m["teacher_accuracy"] > 0.95 and m["accuracy"] > 0.95
    z = I.test_scores()[0]
    assert not np.allclose(K.teacher_logits(), z)
