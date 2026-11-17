"""Animations 154-156 (voxanim.mech_phys_adv) and their experiments (voxanim.study_phys_adv):
the layout is clean, the loop closes, every phase says what kind of evidence it shows, the
numbers on screen are the study's, each study's claim holds as a relative claim with a
margin over seeds or pseudo-experiments, and the recorded arrays are what they say.

The Gaussian-process background needs no data and is built here from an empty cache; the
other two train on MNIST and are skipped without it.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import data_marks, needs_mnist

from voxanim import cache, gp, layout, presets, qa
from voxanim import mech_phys_adv as M
from voxanim import study_phys_adv as S
from voxanim.next_catalog import MNIST, NO_DATA, build_study

AE, GAN, GPB = "autoencoder-anomaly-detection", "gan-mode-collapse", "gp-background-fit"
NAMES = (AE, GAN, GPB)
EACH = [pytest.param(n, marks=data_marks(n)) for n in NAMES]
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_phys_adv", name))
    return _built[name]


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, k, at=0.9):
    """Every string on screen ``at`` of the way through phase ``k``."""
    a, b = anim.spans[k]
    return [x.s for x in anim.scene(min(a + (b - a) * at, b - anim.FADE - 0.05)).texts]


def has(strings, part):
    return any(part in s for s in strings)


# ---------------------------------------------------------------------- all three
def test_registered_with_what_they_need():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)
    assert {AE, GAN} <= MNIST and GPB in NO_DATA


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    anim, _ = built(name)
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for a, b in anim.spans:                           # every phase draws things, not only text
        assert len(anim.scene((a + b) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    anim, _ = built(name)
    for sub in subtitles(anim):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2 and sub
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(anim.PHASES)):                 # what kind of evidence, in every phase
        for at in (0.3, 0.9):
            assert any(t.startswith(M.TAG_STARTS) for t in texts(anim, k, at))


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    _, s = built(name)
    assert len(s.panels) == 4 and s.validate() is s
    assert len(s.metadata["designs_tried"]) >= 2 and s.metadata["limitations"]
    assert s.metadata["evidence"]


# ---------------------------------------------------------------------- shared helpers
def test_helpers_on_known_inputs():
    assert np.isclose(S.entropy_bits(np.full(10, 0.1)), np.log2(10))
    assert S.entropy_bits([0, 1, 0]) == 0.0 and S.entropy_bits([0.5, 0.5]) == 1.0
    assert S.auc([0.0, 1.0], [2.0, 3.0]) == 1.0 and S.auc([2.0, 3.0], [0.0, 1.0]) == 0.0
    assert S.auc([1.0, 1.0], [1.0]) == 0.5 and S.auc([0.0, 2.0], [1.0]) == 0.5
    rng = np.random.default_rng(0)
    seen, unseen = rng.normal(0, 1, 4000), rng.normal(1, 1, 3000)
    curve = S.roc(seen, unseen)
    assert curve[0] < 0.05 and curve[-1] > 0.99 and (np.diff(curve) >= 0).all()
    area = float(((curve[1:] + curve[:-1]) / 2 * np.diff(S.ROC_GRID)).sum())
    assert abs(area - S.auc(seen, unseen)) < 0.005          # the curve and the pair count agree
    assert abs(S.auc(seen, unseen) - 0.7602) < 0.02         # Phi(1 / sqrt 2), the exact area
    assert np.isclose(S.rebuild_error(np.ones(784), np.zeros(784)), 1.0)


# ---------------------------------------------------------------------- 154
@needs_mnist
def test_anomaly_data_is_what_it_says():
    _, s = built(AE)
    m = s.metadata
    x, _y, xt, yt, meta = S.ae_data()
    assert meta["train"]["x_sha256"] != meta["held_out"]["x_sha256"]
    assert meta["train"]["official_partition"] == "train"
    assert meta["held_out"]["official_partition"] == "test"
    assert len(x) == S.AE_TRAIN and len(xt) == S.AE_TEST and m["split_sizes"] == [20000, 5000]
    assert sum(m["removed_from_training"]) == S.AE_TRAIN
    d = S.ae_arrays()
    assert d["err"].shape == (len(S.AE_CODES), 10, len(S.AE_SEEDS), S.AE_TEST)
    p0, p1, p2, p3 = (p.data for p in s.panels)
    left = m["featured"]
    seen_id, unseen_id = (int(v) for v in p0["ids"])
    assert yt[seen_id] != left and yt[unseen_id] == left
    assert np.allclose(p0["values"][0].ravel(), xt[seen_id])
    for k in (0, 1):                                  # the error on screen is the pictures'
        got = S.rebuild_error(p0["values"][2 * k + 1].ravel(), p0["values"][2 * k].ravel())
        assert abs(got - p0["errors"][k]) < 1e-5
    seen, unseen = np.asarray(p1["seen"]), np.asarray(p1["unseen"])
    assert len(seen) == (yt != left).sum() and len(unseen) == (yt == left).sum()
    assert abs(np.median(seen) - p0["errors"][0]) < 1e-6      # the typical digits are medians
    assert abs(np.median(unseen) - p0["errors"][1]) < 1e-3
    assert np.isclose(S.auc(seen, unseen), p2["auc"][left, 0])
    counts = [int(v) for v in p1["counts"]]
    assert counts[1] == (seen > p1["threshold"]).sum() and counts[3] / counts[2] == m["flagged"]
    assert abs(m["false_alarm"] - (1 - S.AE_KEEP)) < 0.002
    assert np.allclose(p1["values"].sum(1), 1.0)
    assert S.AE_CODES[int(np.argmax(p3["mean"]))] == m["code_shown"]


@needs_mnist
def test_anomaly_claims():
    _, s = built(AE)
    m = s.metadata
    area = np.asarray(s.panels[3].data["seeds"])              # [code, removed class, seed]
    main, last = S.AE_CODES.index(m["code_shown"]), len(S.AE_CODES) - 1
    per_class = area[main]
    easy, hard = m["easiest"], m["hardest"]
    # the unseen class is found: every seed, for the featured class
    assert per_class[m["featured"]].min() > 0.85
    # some classes are much easier anomalies than others: more than seeds differ
    assert per_class[easy].min() > per_class[hard].max() + 0.3
    assert per_class.std(1).max() < 0.05
    # a code that is too large rebuilds everything: the area falls, for every seed
    by_seed = area.mean(1)                                    # [code, seed]
    assert by_seed[main].min() > by_seed[last].max() + 0.1
    seen, unseen = (np.asarray(s.panels[3].data[k]) for k in ("seen_mse", "unseen_mse"))
    assert (np.diff(seen) < 0).all() and (np.diff(unseen) < 0).all()
    assert unseen[main] / seen[main] > unseen[last] / seen[last] + 0.1


@needs_mnist
def test_anomaly_numbers_on_screen():
    anim, s = built(AE)
    m, sub = s.metadata, subtitles(anim)
    p0 = s.panels[0].data
    assert "error %.3f" % p0["errors"][0] in sub[1] and "%.3f. That" % p0["errors"][1] in sub[1]
    assert "%d%% of the unseen" % round(100 * m["flagged"]) in sub[2]
    assert "area under it is %.2f" % s.panels[2].data["auc"][m["featured"], 0] in sub[2]
    by = m["auc_by_class"]
    assert "from %.2f (the %ds) down to %.2f (the %ds)" % (
        by[str(m["easiest"])]["mean"], m["easiest"], by[str(m["hardest"])]["mean"],
        m["hardest"]) in sub[3]
    assert "with %d numbers the mean area is %.2f, with 128 it is %.2f" % (
        m["code_shown"], m["auc_by_code"][str(m["code_shown"])], m["auc_by_code"]["128"]) in sub[4]
    assert has(texts(anim, 1), "error = %.3f" % p0["errors"][1])
    counts = [int(v) for v in s.panels[1].data["counts"]]
    assert has(texts(anim, 2), "%d of %d unseen" % (counts[3], counts[2]))
    assert has(texts(anim, 3), "the hardest: an unseen %d" % m["hardest"])
    assert has(texts(anim, 4), "code 128") and has(texts(anim, 0), "code: %d numbers"
                                                   % m["code_shown"])


# ---------------------------------------------------------------------- 155
@needs_mnist
def test_collapse_data_is_what_it_says():
    _, s = built(GAN)
    m = s.metadata
    p0, p1, p2, _p3 = (p.data for p in s.panels)
    n_set, n_seed, n_snap = len(S.GAN_SETTINGS), len(S.GAN_SEEDS), len(S.gan_steps())
    hist = np.asarray(p1["history"])
    assert hist.shape == (n_set, n_seed, n_snap, 10) and np.allclose(hist.sum(3), 1.0)
    assert np.asarray(p0["history"]).shape == (n_set, n_seed, n_snap, S.GAN_KEPT, 784)
    assert list(p0["steps"]) == list(range(0, S.GAN_STEPS + 1, S.GAN_EVERY))
    H = np.asarray(p2["seeds"])
    assert np.isclose(H[1, 2, 7], S.entropy_bits(hist[1, 2, 7]))
    assert np.array_equal(p2["top"], hist.argmax(3))
    assert np.allclose(p2["end"], H[:, :, -S.GAN_LAST:].mean(2))
    assert np.array_equal(p2["collapsed"], np.asarray(p2["end"]) < S.GAN_COLLAPSED)
    assert m["data"]["train_x_sha256"] != m["data"]["test_x_sha256"]
    assert np.isclose(np.sum(p1["values"][0]), 1.0) and m["real_entropy"] > 3.25
    assert abs(np.asarray(p1["real_labels"]) - np.asarray(p1["values"][0])).max() < 0.02
    G, D = S.gan_init(True, np.random.default_rng(0))
    assert G["W3"].shape == (S.GAN_CODE + 10, 64) and D["Wa"].shape == (794, 256)
    G, D = S.gan_init(False, np.random.default_rng(0))
    assert G["W3"].shape == (S.GAN_CODE, 64) and D["Wa"].shape == (784, 256)
    assert S.gan_onehot([3]).tolist() == [[0, 0, 0, S.GAN_LABEL, 0, 0, 0, 0, 0, 0]]
    z = np.random.default_rng(1).normal(size=(4, S.GAN_CODE))
    assert S.gan_generate(G, z)[-1].shape == (4, 784)


@needs_mnist
def test_collapse_claims():
    _, s = built(GAN)
    m = s.metadata
    bal, heavy, bal_l, heavy_l = m["settings"]
    e, c, a = m["entropy_end"], m["covered_end"], m["asked_end"]
    # the balanced generator covers the digits on every seed; the generator-heavy one does not
    assert e[bal]["min"] > e[heavy]["max"] + 0.3
    assert c[bal]["min"] >= 7 and c[heavy]["mean"] < c[bal]["mean"] - 3
    assert m["collapsed"][bal] == 0 and m["collapsed"][heavy] >= 3
    # the label is obeyed; a generator that is not told cannot obey it
    assert a[bal_l]["min"] > 0.6 and a[bal_l]["min"] > a[bal]["max"] + 0.4
    assert a[bal]["max"] < 0.2 and a[heavy]["max"] < 0.2
    assert c[bal_l]["min"] >= 7
    # and in the collapsing schedule the label brings the digits back, on every seed
    assert c[heavy_l]["min"] > c[heavy]["max"] + 3
    assert e[heavy_l]["min"] > e[heavy]["max"] + 1.0 and m["collapsed"][heavy_l] == 0
    assert a[heavy_l]["min"] > a[heavy]["max"] + 0.4


@needs_mnist
def test_collapse_numbers_on_screen():
    anim, s = built(GAN)
    m, sub = s.metadata, subtitles(anim)
    H = np.asarray(s.panels[2].data["seeds"])
    bal, heavy, bal_l, heavy_l = m["settings"]
    assert "entropy %.2f of a possible 3.32 bits" % H[0, 0, -1] in sub[0]
    assert "one digit: entropy %.2f bits" % H[1, 0, -1] in sub[1]
    assert "%d%% of its samples" % round(100 * s.panels[1].data["values"][2].max()) in sub[1]
    assert "ends at %.1f bits and the generator-heavy one at %.1f: %d of its 5 runs" % (
        m["entropy_end"][bal]["mean"], m["entropy_end"][heavy]["mean"],
        m["collapsed"][heavy]) in sub[2]
    assert "reads %d%% of the labelled samples" % round(100 * m["asked_end"][bal_l]["mean"]) \
        in sub[4]
    assert has(texts(anim, 0), "=  %.2f bits" % H[0, 0, -1])
    assert has(texts(anim, 1), "after %s updates" % f"{S.GAN_STEPS:,}")
    assert has(texts(anim, 2), "generator-heavy: %d of 5 runs" % m["collapsed"][heavy])
    assert has(texts(anim, 4), "%.1f" % m["covered_end"][heavy_l]["mean"])
    assert "from %.1f to %.1f of 10" % (m["covered_end"][heavy]["mean"],
                                        m["covered_end"][heavy_l]["mean"]) in sub[4]
    assert has(texts(anim, 3), "%d%% and %d%%" % (round(100 * m["asked_end"][bal_l]["mean"]),
                                                  round(100 * m["asked_end"][heavy_l]["mean"])))


# ---------------------------------------------------------------------- 156
def test_background_builds_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    s = S.gp_background_fit().validate()              # offline (conftest): a download would raise
    M.GPBackground(s).scene(3.0)
    assert [p.name.split("-")[3] for p in tmp_path.iterdir()] == ["gp"]
    assert s.metadata["dataset"].startswith("none") and s.metadata["evidence"]["exact"]


def test_background_pieces_are_what_they_say():
    b, t, win = S.gp_truth(), S.gp_template(), S.gp_window()
    assert np.isclose(S.gp_truth(100.0), 9300.0) and np.isclose(t.sum(), 1.0)
    assert win.sum() == 4 and np.allclose(S.gp_weights([S.GP_M0]) @ t, 1.0)
    # the fixed function is the maximum-likelihood exponential: an exact exponential is
    # returned as it is, and at the fit the two score equations of Poisson counts hold
    line = 500.0 * np.exp(-(S.GP_M - 100.0) / 80.0)
    assert np.allclose(S.fit_exponential(line)[0], line, rtol=1e-9)
    y0, y1 = S.gp_toys()
    fit = S.fit_exponential(y0[:50])
    assert np.abs((y0[:50] - fit).sum(1)).max() < 1e-6
    assert np.abs(((y0[:50] - fit) * S.GP_M).sum(1)).max() < 1e-3
    # the smoother is the Gaussian process's posterior mean and spread (voxanim.gp)
    ell = 60.0
    H, sd, L = S.gp_smoother(ell)
    z = ((y0[0] - fit[0]) / np.sqrt(fit[0]))
    post = gp.posterior(gp.RBF(variance=S.GP_AMP ** 2, lengthscale=ell), S.GP_M, z, S.GP_M, 1.0)
    assert np.allclose(H @ z, post.mu, atol=1e-6) and np.allclose(sd, post.sd, atol=1e-6)
    ev = gp.log_marginal(gp.RBF(variance=S.GP_AMP ** 2, lengthscale=ell), S.GP_M, z, 1.0)
    assert np.isclose(S.gp_evidence(z, L)[0], ev.value, atol=1e-5)
    # the pseudo-experiments are Poisson about the stated truth, and seeded
    assert y0.shape == (S.GP_TOYS, 80) and (y0 == np.round(y0)).all()
    assert np.abs((y0.mean(0) - b) / np.sqrt(b / S.GP_TOYS)).max() < 4.5
    assert np.abs((y0.var(0) - b) / b).max() < 0.12
    assert abs((y1 - y0).sum(1).mean() - S.GP_S) < 1.0 and ((y1 - y0) >= 0).all()
    s = build_study("study_phys_adv", GPB)
    assert np.array_equal(s.panels[0].data["ys"][0], y0[0])
    assert np.array_equal(s.panels[0].data["with_signal"], y1[0])
    assert np.allclose(s.panels[0].data["ys"][1], b)
    assert abs(S.GP_S / s.metadata["window_error"] - 5) < 0.25


def test_background_claims():
    _, s = built(GPB)
    m = s.metadata
    p2, p3 = s.panels[2].data, s.panels[3].data
    ells = np.asarray(p2["x"])
    spurious, share = (np.asarray(v) for v in p2["ys"])
    exact, sem = np.asarray(p2["exact"]), np.asarray(p2["sem"])
    e, good, short = m["exponential"], m["gp_good"], m["gp_short"]
    # pseudo-experiments and the exact (no-fluctuation) calculation agree
    assert np.abs(spurious - exact[0]).max() < 5 * sem[0].max() + 0.3
    assert np.abs(share - exact[1]).max() < 0.01
    assert abs(e["spurious"] - e["spurious_exact"]) < 5 * e["spurious_error"]
    # the fixed function leaves a spurious signal; the Gaussian process does not
    assert abs(e["spurious"]) > 20 * e["spurious_error"]
    assert abs(good["spurious"]) < 0.25 * abs(e["spurious"])
    scan = np.abs(np.asarray(p3["exact"]) / np.asarray(p3["error"]))
    assert scan[0].max() > 5 and scan[1].max() < 1 and scan[1].mean() < 0.1 * scan[0].mean()
    # the catch: a short length scale absorbs the resonance, a longer one less
    assert short["share"] < 0.1 and 0.5 < good["share"] < e["share"] - 0.1
    assert (np.diff(share[ells >= S.GP_SHORT]) > 0).all()
    assert share[-1] > 0.85
    # and a process that is too stiff has a spurious signal of its own
    assert np.abs(spurious[ells >= 160]).max() > 0.5 * abs(e["spurious"])
    assert np.abs(spurious[ells <= m["ell_good"]]).max() < 0.1 * abs(e["spurious"])
    assert S.GP_SIGMA < S.GP_SHORT < m["ell_good"]
    assert sum(v[0] for v in m["evidence_pick"].values()) == S.GP_TOYS


def test_background_numbers_on_screen():
    anim, s = built(GPB)
    m, sub = s.metadata, subtitles(anim)
    e, good, short = m["exponential"], m["gp_good"], m["gp_short"]
    assert "98,000 events in 80 bins" in sub[0] and 97500 < m["total_events"] < 98500
    assert "averages %s events" % M.signed(e["spurious"]) in sub[1]
    assert "from %s to %s events" % (M.signed(e["spurious"]),
                                     M.signed(good["spurious"], 1)) in sub[2]
    assert "ℓ = %g GeV" % m["ell_good"] in sub[2]
    assert "keeps %d%% of the signal at ℓ = %g GeV and %d%% at %g; the exponential keeps %d%%" % (
        round(100 * short["share"]), m["ell_short"], round(100 * good["share"]), m["ell_good"],
        round(100 * e["share"])) in sub[4]
    c = int(s.panels[1].data["centre"])
    n, f = s.panels[0].data["ys"][0][c], s.panels[0].data["ys"][2][c]
    assert has(texts(anim, 1, 0.3), "N = %d,  fit = %.1f" % (n, f))
    assert has(texts(anim, 1, 0.3), "= %s" % M.signed((n - f) / np.sqrt(f), 2))
    assert np.isclose((n - f) / np.sqrt(f), s.panels[1].data["ys"][0][c])
    assert has(texts(anim, 3), "left in the window: %s" % M.signed(m["drawn"]["short"]))
    assert has(texts(anim, 0), "9000 · exp(−(m − 100) / 50)")
