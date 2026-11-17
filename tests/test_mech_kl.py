"""Animations 170-171 (the KL term of a VAE; DEC's centres through training): what
they draw is what was measured, and what was measured supports what they say."""

import os

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import deep_models as dm
from voxanim import layout, presets, qa
from voxanim import mech_kl as M
from voxanim import study_families as F
from voxanim import study_kl as S
from voxanim.mech_families import TEN
from voxanim.next_catalog import build_study

NAMES = ("vae-kl-term", "dec-centroids-training")
every = pytest.mark.parametrize("name", [pytest.param(n, marks=needs_mnist) for n in NAMES])
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_kl", name))
    return _built[name]


def shown(anim, k, u):
    """Every string on screen ``u`` of the way through phase ``k``, with the live subtitle."""
    a, b = anim.spans[k]
    scene = anim.scene(float(a + (b - a) * u))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def has(strings, part):
    return any(part in s for s in strings)


# ---------------------------------------------------------------------- both
@every
def test_studies_say_what_was_run(name):
    _anim, study = built(name)
    meta = study.metadata
    assert set(S.STUDIES) == set(NAMES) == set(M.ANIMATIONS)
    assert meta["dataset"] == "MNIST" and len(meta["seeds"]) >= 3
    assert meta["drawn_seed"] == S.DRAWN == S.SEEDS[0]            # fixed in advance: the first
    assert meta["data"]["train_x_sha256"] != meta["data"]["test_x_sha256"]
    assert os.path.exists(meta["artifact"]) and meta["limitations"] and meta["designs_tried"]
    assert meta["evidence"]["measured"] and meta["evidence"]["exact"] and meta["measures"]
    assert len(study.panels) == 4 and study.validate() is study


@every
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.duration <= 66.0
    assert qa.check(name, samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))

    def key(sc):
        return sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)

    assert key(first) == key(last)
    for k in range(len(anim.PHASES)):                    # a subtitle is two lines at most
        lines = layout.wrap(anim.subtitle_for(k), layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(s, layout.SUB_SIZE) for s in lines) <= 1 - 2 * layout.SAFE
        # every phase says what kind of evidence it shows
        assert any(s in (M.MEASURED, M.EXACT) for s in shown(anim, k, 0.5))


@every
def test_every_latent_picture_keeps_the_ten_digit_colours_and_a_legend(name, monkeypatch):
    anim, _study = built(name)
    calls = []
    real = M.scatter

    def spy(z, inks, *args, **kwargs):
        calls.append((len(z), np.unique(np.asarray(inks), axis=0)))
        return real(z, inks, *args, **kwargs)

    monkeypatch.setattr(M, "scatter", spy)
    clouds = {"vae-kl-term": (2, 3, 4, 5), "dec-centroids-training": (0, 1, 2, 3, 4, 5)}[name]
    for k in clouds:
        calls.clear()
        strings = shown(anim, k, 0.7)
        assert calls, "phase %d draws no cloud" % (k + 1)
        for n, inks in calls:                            # all 2,000 codes, in the ten colours
            assert n == F.TEST_N and len(inks) == 10
            assert {tuple(np.round(c, 4)) for c in inks} == {
                tuple(np.round(np.asarray(c, np.float32), 4)) for c in TEN}
        assert all(str(digit) in strings for digit in range(10))       # the legend


def test_scatter_draws_each_point_in_its_colour_later_ones_on_top():
    red, blue = np.array([1.0, 0, 0], np.float32), np.array([0, 0, 1.0], np.float32)
    lim = (-1.0, 1.0, -1.0, 1.0)
    tex = M.scatter(np.array([[0.0, 0.0], [0.5, 0.5], [0.0, 0.0], [9.0, 9.0]]),
                    np.array([red, red, blue, red]), lim, 50, r=3.0)
    assert tex.shape == (100, 100, 3)
    np.testing.assert_allclose(tex[50, 50], blue, atol=1e-6)     # the later point covers
    np.testing.assert_allclose(tex[25, 75], red, atol=1e-6)      # y is up
    np.testing.assert_allclose(tex[5, 5], M.PAPER, atol=1e-6)
    assert 20 < (np.abs(tex - M.PAPER).sum(-1) > 0.05).sum() < 400    # two dots, nothing more
    empty = M.scatter(np.zeros((0, 2)), np.zeros((0, 3)), lim, 50)
    np.testing.assert_allclose(empty, np.broadcast_to(M.PAPER, empty.shape), atol=1e-6)


def test_region_borders_lie_between_the_centres():
    mu = np.array([[-1.0, 0.0], [1.0, 0.0]])
    tex = M.regions(mu, (-2.0, 2.0, -2.0, 2.0), 40)
    edge = np.abs(tex - M.PAPER).sum(-1) > 0.01
    rows, cols = np.nonzero(edge)
    assert tex.shape == (80, 80, 3) and len(rows) >= 80
    assert np.abs(cols - 39.5).max() <= 2                       # the line x = 0, top to bottom


def test_stacked_histogram_counts_every_code_once():
    rng = np.random.default_rng(0)
    v, lab = rng.normal(size=500), rng.integers(0, 10, 500)
    tex = M.stacked(v, lab, -4.0, 4.0, 16, 200, 500.0)
    filled = (np.abs(tex - M.PAPER).sum(-1) > 0.01).sum(0)       # coloured rows per bin
    np.testing.assert_allclose(filled, np.histogram(v, np.linspace(-4, 4, 17))[0] * 200 / 500,
                               atol=1.0)


# ---------------------------------------------------------------------- 170: functions
def test_bell_functions_against_known_values():
    np.testing.assert_allclose(S.phi([0.0, 1.0, 1.959964, -1.0]),
                               [0.5, 0.841345, 0.975, 0.158655], atol=1e-6)
    mid, wide = S.kl_pieces(0.0, 1.0)
    assert mid == 0.0 and wide == 0.0                           # zero only at mu = 0, sd = 1
    assert all(sum(S.kl_pieces(m, s)) > 0 for m, s in ((0.5, 1.0), (0.0, 0.5), (0.0, 2.0)))
    # against the integral of q ln(q / p)
    x = np.linspace(-5, 6, 200001)
    q, p = M.normal(x, 0.7, 0.3), M.normal(x)
    assert sum(S.kl_pieces(0.7, 0.3)) == pytest.approx(
        float(np.sum(q * np.log(q / p)) * (x[1] - x[0])), abs=1e-6)
    np.testing.assert_allclose(S.RING_SHARE, [0.3935, 0.8647, 0.9889], atol=1e-4)


def test_gaussianity_reads_a_bell_as_a_bell():
    z = np.random.default_rng(3).normal(size=(20000, 2))
    g = S.gaussianity(z)
    assert g["fit"] < 0.002 and g["ks"].max() < 0.02 and g["shape"].max() < 0.02
    np.testing.assert_allclose(g["rings"], S.RING_SHARE, atol=0.01)
    moved = S.gaussianity(z * [2.0, 0.5] + [1.0, 0.0])           # the best-fit Gaussian's KL
    assert moved["fit"] == pytest.approx(0.5 * (4 + 0.25 + 1 - 2 - np.log(4 * 0.25)), abs=0.03)
    assert moved["ks"].max() > 0.15 and moved["shape"].max() < 0.02      # same shape, moved
    two = np.concatenate([z[:10000] * 0.3 - [1.5, 0], z[10000:] * 0.3 + [1.5, 0]])
    assert S.gaussianity(two)["shape"][0] > 0.1                  # two lumps are not a bell


@needs_mnist
def test_weighted_gradients_are_the_models_and_right():
    x = F.data()["x"][:6]
    rng = np.random.default_rng(0)
    P = dm.vae_init(2, rng)
    eps = rng.normal(size=(6, 2))
    f = dm.vae_forward(P, x, eps)
    one = S.beta_grads(P, f, 1.0)
    for k, g in dm.vae_backward(P, f).items():                   # weight 1 is the model's own
        np.testing.assert_array_equal(one[k], g)

    def loss(Q, beta):
        _l, bce, kl = dm.vae_loss(dm.vae_forward(Q, x, eps))
        return bce + beta * kl

    g = S.beta_grads(P, f, 3.0)
    for name, idx in (("Wmu", (5, 1)), ("blv", (0,)), ("We1", (3, 7)), ("Wd0", (1, 2))):
        up, down = ({k: v.copy() for k, v in P.items()} for _ in range(2))
        up[name][idx] += 1e-5
        down[name][idx] -= 1e-5
        assert g[name][idx] == pytest.approx((loss(up, 3.0) - loss(down, 3.0)) / 2e-5,
                                             rel=1e-4, abs=1e-6)
    plain = S.beta_grads(P, dm.vae_forward(P, x, eps * 0.0), 0.0)        # the autoencoder
    assert not plain["Wlv"].any() and not plain["blv"].any()


# ---------------------------------------------------------------------- 170: the runs
@needs_mnist
def test_checkpoints_and_runs_are_what_the_study_says():
    d = S.kl_arrays()
    assert S.CHECKS[0] == 0 and list(S.CHECKS) == sorted(set(S.CHECKS))
    assert sum(c < S.EPOCH for c in S.CHECKS) >= 5 and S.CHECKS[-1] == 20 * S.EPOCH
    assert np.array_equal(d["labels"], F.data()["yt"])
    for run in S.RUNS:
        assert d[run + "_mu"].shape == (len(S.SEEDS), len(S.CHECKS), F.TEST_N, 2)
    # one start for all four runs of a seed: the same weights give the same first cloud
    for run in S.RUNS[1:]:
        np.testing.assert_array_equal(d[run + "_mu"][:, 0], d["ae_mu"][:, 0])
    # the recorded KL term is the model's own
    P = S.kl_params("vae")
    f = dm.vae_forward(P, F.data()["xt"], np.zeros((F.TEST_N, 2)))
    drawn = S.SEEDS.index(S.DRAWN)
    np.testing.assert_allclose(f["mu"], d["vae_mu"][drawn, -1], atol=2e-3)
    assert dm.vae_loss(f)[2] == pytest.approx(d["vae_kl"][drawn, -1].sum(), abs=0.02)
    # and no held-out digit is a training digit
    assert F.nearest(F.data()["xt"][:300], F.data()["x"])[0].min() > 0.5


@needs_mnist
def test_the_cloud_leaves_the_bell_then_settles_into_it_and_only_with_the_kl_term():
    d = S.kl_arrays()
    vae, ae = np.asarray(d["vae_g_fit"], float), np.asarray(d["ae_g_fit"], float)
    early = [i for i, c in enumerate(S.CHECKS) if c <= 80]
    for s in range(len(S.SEEDS)):
        assert vae[s, early].max() > 2.5 > 10 * vae[s, -1]        # burst apart, then settled
        assert vae[s, -1] < 0.2 < vae[s, 0]                        # closer than at the start
        assert np.all(np.diff(vae[s, 6:]) < 0.02)                  # falling pass after pass
        assert ae[s, -1] > 5.0 > 25 * vae[s, -1]                   # no KL term: far, and
        assert ae[s, -1] > ae[s, 6]                                # still moving away
    rings, bell = np.asarray(d["vae_g_rings"], float)[:, -1], np.array(S.RING_SHARE)
    assert np.abs(rings - bell).max() < 0.13                       # near 39 / 86 / 99 %,
    assert np.all(rings[:, 1] < bell[1] - 0.05)                    # but not a bell: too few
    assert np.all(np.asarray(d["ae_g_rings"], float)[:, -1, 2] < 0.75)
    # ... and the lumps of the digit classes stay: even centred and scaled to unit
    # covariance the cloud is further from a bell than 2,000 draws of one would be
    # (their KS distance exceeds 1.36 / sqrt(2000) = 0.030 one time in twenty)
    shape = np.asarray(d["vae_g_shape"], float).mean(2)
    assert np.all(shape[:, -1] > 0.035) and shape[:, -1].mean() > shape[:, 0].mean() + 0.01
    # the KL term itself rises from nothing and stays up: the blobs never reach the prior
    kl = np.asarray(d["vae_kl"], float).sum(2)
    assert np.all(kl[:, 0] < 0.5) and np.all(kl[:, -1] > 5.0)
    # the autoencoder rebuilds no better (within 3 nats of about 144)
    assert abs(d["ae_bce"][:, -1].mean() - d["vae_bce"][:, -1].mean()) < 3.0


@needs_mnist
def test_the_weight_trades_the_two_terms():
    d = S.kl_arrays()

    def last(key):
        return np.asarray(d[key], float)[:, -1]

    kl = {run: last(run + "_kl").sum(1) for run in S.RUNS}
    spread = {run: np.exp(0.5 * last(run + "_lv")).mean((1, 2)) for run in S.RUNS}
    assert S.BETA == {"ae": 0.0, "low": 0.1, "vae": 1.0, "high": 10.0}
    assert np.all(kl["low"] > kl["vae"] + 3) and np.all(kl["vae"] > kl["high"] + 3)
    assert np.all(last("low_g_fit") > 10 * last("vae_g_fit"))
    assert np.all(last("high_bce") > last("vae_bce") + 10)        # rebuilding pays for it
    assert np.all(spread["high"] > 3 * spread["vae"])              # wider blobs
    assert np.all(d["high_sharp"] < d["vae_sharp"] - 0.05)         # blurrier digits
    assert np.all(last("low_bce") < last("vae_bce") + 1)


@needs_mnist
def test_the_trained_blobs_sit_where_the_two_terms_balance():
    _anim, study = built("vae-kl-term")
    scan = study.metadata["pull_scan"]
    c, w = np.array(scan["centre"]), np.array(scan["width"])
    s, ws = np.array(scan["centre_scale"]), np.array(scan["width_scale"])
    assert 0.9 <= s[c.sum(1).argmin()] <= 1.1 and 0.7 <= ws[w.sum(1).argmin()] <= 1.5
    home = c[list(s).index(1.0)]
    assert c[0, 1] < home[1] - 1 and c[0, 0] > home[0] + 100      # to the middle: KL down,
    assert w[-1, 1] < home[1] - 2 and w[-1, 0] > home[0] + 20     # wider: KL down, both pay
    d = S.kl_arrays()
    assert home[0] == pytest.approx(d["vae_bce"][S.SEEDS.index(S.DRAWN), -1], abs=0.5)


# ---------------------------------------------------------------------- 170: on screen
@needs_mnist
def test_kl_numbers_on_screen_are_the_measured_ones():
    anim, study = built("vae-kl-term")
    t = study.metadata["table"]
    d = S.kl_arrays()
    drawn = S.SEEDS.index(S.DRAWN)
    # the tracked digit's blob and the formula on it
    np.testing.assert_allclose(anim.mu0, d["vae_mu"][drawn, -1, S.TRACK], atol=2e-3)
    j = anim.j0
    mid, wide = S.kl_pieces(anim.mu0[j], anim.sd0[j])
    strings = shown(anim, 1, 0.36)
    assert has(strings, "μ = %.2f      σ = %.2f" % (anim.mu0[j], anim.sd0[j]))
    assert has(strings, "½ μ²  =  %.2f" % mid) and has(strings, "KL  =  %.2f nats" % (mid + wide))
    assert has(strings, "= %.1f nats" % anim.kl0)
    assert has(shown(anim, 1, 0.86), "KL  =  0.00 nats")            # slid onto the prior
    # the training phase ends on the final checkpoint's numbers
    strings = shown(anim, 3, 0.97)
    assert has(strings, "reconstruction term: %.0f nats" % d["vae_bce"][:, -1].mean())
    assert has(strings, "KL term: %.1f nats" % d["vae_kl"][:, -1].sum(1).mean())
    assert has(strings, "cloud's distance from the bell: %.2f" % d["vae_g_fit"][:, -1].mean())
    assert has(strings, "%.0f%%   %.0f%%   %.0f%%" % tuple(
        100 * d["vae_g_rings"][drawn, -1]))
    assert has(shown(anim, 3, 0.04), "update 0") and has(strings, "20 passes (3,120 updates)")
    # the contrast and the three weights
    strings = shown(anim, 4, 0.9)
    for run in ("vae", "ae"):
        assert has(strings, "distance from the bell: %.2f" % t[run]["fit"]["mean"])
    strings = shown(anim, 5, 0.95)
    for run in ("low", "vae", "high"):
        assert has(strings, "reconstruction %.0f" % t[run]["reconstruction"]["mean"])
        assert has(strings, "KL term %.1f" % t[run]["kl"]["mean"])
        assert has(strings, "sharpness %.2f" % t[run]["sharpness"]["mean"])
    assert has([anim.subtitle_for(3)], "%.2f after 20 passes" % t["vae"]["fit"]["mean"])


# ---------------------------------------------------------------------- 171: functions
def test_cluster_measures_against_brute_force():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 4, 300)
    assert S.nmi(a, a) == pytest.approx(1.0) and S.nmi(a, (a + 1) % 4) == pytest.approx(1.0)
    assert S.nmi(a, rng.integers(0, 4, 300)) < 0.05
    z = rng.normal(size=(60, 2))
    lab = rng.integers(0, 3, 60)
    brute = []
    for i in range(60):
        dist = np.sqrt(((z - z[i]) ** 2).sum(1))
        own = (lab == lab[i]) & (np.arange(60) != i)
        other = min(dist[lab == c].mean() for c in range(3) if c != lab[i])
        brute.append((other - dist[own].mean()) / max(other, dist[own].mean()))
    assert S.silhouette(z, lab, k=3) == pytest.approx(np.mean(brute), abs=1e-9)
    tight = np.concatenate([rng.normal(size=(30, 2)) * 0.01, rng.normal(size=(30, 2)) * 0.01 + 5])
    assert S.silhouette(tight, np.repeat([0, 1], 30), k=2) > 0.99
    conf = S.confusion(np.array([0, 0, 1, 2]), np.array([3, 3, 4, 3]), k=3)
    assert conf.sum() == 4 and conf[0, 3] == 2 and conf[1, 4] == 1
    assert S.shared_digits([[5, 5, 0], [9, 1, 0]]) == [[0, 1], [0]]


@needs_mnist
def test_kmeans_trace_is_the_models_kmeans_step_by_step():
    z = S.dec_run("dec")["z"][0]
    trace, within = S.kmeans_trace(z, S.K, np.random.default_rng(4))
    np.testing.assert_array_equal(trace[-1], dm.kmeans(z, S.K, np.random.default_rng(4)))
    assert trace.shape == (S.KM_ITERS + 1, S.K, 2) and np.all(np.diff(within) <= 1e-12)
    lab = ((z[:, None] - trace[3][None]) ** 2).sum(-1).argmin(1)     # one round by hand
    np.testing.assert_allclose(trace[4], [z[lab == j].mean(0) for j in range(S.K)])


# ---------------------------------------------------------------------- 171: the runs
@needs_mnist
def test_dec_runs_are_the_generative_families_runs_recorded_more_often():
    if not os.path.exists(F.cache_file("models", F.SETTINGS)):
        pytest.skip("the generative-families models are not trained")
    labels = F.data()["yt"]
    for seed in S.SEEDS:
        for kind in S.KINDS:
            run = S.dec_run(kind, 2, seed)
            cloud = F.trace(kind, 2, seed, "cloud")
            np.testing.assert_allclose(run["z"][0][:F.CLOUD], cloud[0], atol=1e-4)
            np.testing.assert_allclose(run["z"][-1][:F.CLOUD], cloud[-1], atol=2e-3)
            np.testing.assert_allclose(run["mu"][-1], F.trace(kind, 2, seed, "centres")[-1],
                                       atol=2e-3)
            assert run["conf"].sum((1, 2)).tolist() == [F.TEST_N] * len(S.DEC_CHECKS)
            np.testing.assert_array_equal(run["conf"][-1].sum(0), np.bincount(labels))
    assert list(S.DEC_CHECKS) == sorted(set(S.DEC_CHECKS)) and S.DEC_CHECKS[0] == 0
    assert sum(c <= 150 for c in S.DEC_CHECKS) >= 6                # more of them early


@needs_mnist
def test_dec_separates_more_than_it_purifies():
    _anim, study = built("dec-centroids-training")
    for code in S.DEC_CODES:
        for kind in S.KINDS:
            runs = [S.dec_run(kind, code, s) for s in S.SEEDS]
            for r in runs:
                assert r["sil"][-1] > r["sil"][0] + 0.25           # tighter and further apart
                assert r["maxq"][-1] > r["maxq"][0] + 0.15         # surer assignments
                assert r["moved"][0] == 0 and np.all(np.diff(r["moved"]) > 0)
                assert -0.02 < r["acc"][-1] - r["acc"][0] < 0.12   # labels: a little, at most
                assert r["acc"][-1] < 0.65 and r["nmi"][-1] < 0.65
            gain = np.mean([r["acc"][-1] - r["acc"][0] for r in runs])
            assert 0.0 < gain < 0.1
    for s in S.SEEDS:                                    # the decoder: left behind, or kept
        dec, idec = S.dec_run("dec", 2, s), S.dec_run("idec", 2, s)
        assert dec["mse"][0] == idec["mse"][0]
        assert dec["mse"][-1] > dec["mse"][0] + 0.02 and idec["mse"][-1] < idec["mse"][0] + 0.01
    t = study.metadata["table"]
    # ten numbers are not clearly better here, so they are reported, not drawn
    assert abs(t["c10"]["dec"]["acc"]["end"]["mean"] - t["c2"]["dec"]["acc"]["end"]["mean"]) < 0.05
    st = study.metadata["starts"]
    assert len(st["accuracy_end"]) == 1 + S.EXTRA_STARTS
    spread = max(st["accuracy_end"]) - min(st["accuracy_end"])
    seeds = t["c2"]["dec"]["acc"]["end"]
    assert spread < 0.06 < seeds["max"] - seeds["min"]             # the autoencoder matters more
    # digits still share clusters at the end of the drawn run
    assert sum(len(g) >= 2 for g in study.metadata["shared_digits"]) >= 3
    assert np.array_equal(study.panels[3].data["values"], S.dec_run("dec")["conf"][-1])


# ---------------------------------------------------------------------- 171: on screen
@needs_mnist
def test_dec_numbers_on_screen_are_the_measured_ones():
    anim, study = built("dec-centroids-training")
    run, st = S.dec_run("dec"), S.dec_start()
    # k-means: the centres drawn are the trace's, the last ones DEC's start
    np.testing.assert_array_equal(anim.km[-1], run["mu"][0])
    assert has(shown(anim, 1, 0.97), "digits %.2f" % st["km_acc"][-1])
    assert has(shown(anim, 1, 0.97), "nearest centre: %.2f" % st["km_within"][-1])
    # q and p of the three codes: the model's kernel, the target from the training
    # codes' cluster sizes
    q, _k = dm.soft_assign(run["z"][0], run["mu"][0])
    assert [int(anim.labels[i]) for i in anim.pick] == [d for d, _v in anim.PICKS]
    np.testing.assert_allclose(anim.q3, q[anim.pick])
    np.testing.assert_allclose(anim.q3.sum(1), 1.0)
    w = q[anim.pick] ** 2 / run["freq0"]
    np.testing.assert_allclose(anim.p3, w / w.sum(1, keepdims=True))
    assert np.all(anim.p3.max(1) > anim.q3.max(1))                # the target is sharper
    assert run["freq0"].sum() == pytest.approx(F.TRAIN_N)
    strings = shown(anim, 2, 0.97)
    for r in range(3):
        assert has(strings, "largest q %.2f" % anim.q3[r].max())
        assert has(strings, "largest p %.2f" % anim.p3[r].max())
    assert has(strings, "= %.3f" % anim.kl_a)
    assert anim.kl_a == pytest.approx(float((anim.p3[0] * np.log(anim.p3[0] / anim.q3[0])).sum()))
    # the training phase ends on the drawn run's final numbers
    strings = shown(anim, 3, 0.97)
    assert has(strings, "DEC update 1,500")
    assert has(strings, "accuracy %.2f   (was %.2f)" % (run["acc"][-1], run["acc"][0]))
    assert has(strings, "silhouette %.2f   (was %.2f)" % (run["sil"][-1], run["sil"][0]))
    assert has(strings, "centres moved %.2f" % run["moved"][-1])
    assert has(shown(anim, 3, 0.04), "DEC update 0")
    # separated against pure, and what it depends on
    strings = shown(anim, 4, 0.95)
    assert has(strings, "silhouette %.2f, from %.2f" % (run["sil"][-1], run["sil"][0]))
    assert has(strings, "accuracy %.2f, from %.2f" % (run["acc"][-1], run["acc"][0]))
    assert sorted(anim.conf.sum(1)) == sorted(run["conf"][-1].sum(1))
    t = study.metadata["table"]["c2"]
    sub = anim.subtitle_for(3)
    for k in ("sil", "acc"):
        assert "%.2f to %.2f" % (t["dec"][k]["start"]["mean"], t["dec"][k]["end"]["mean"]) in sub
    strings = shown(anim, 5, 0.95)
    assert has(strings, "rebuilding error %.3f" % t["dec"]["mse"]["end"]["mean"])
    assert has(strings, "rebuilding error %.3f" % t["idec"]["mse"]["end"]["mean"])
    assert has(strings, "%.2f" % t["dec"]["acc"]["end"]["mean"])
