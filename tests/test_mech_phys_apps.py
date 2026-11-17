"""Animations 150-153 (voxanim.mech_phys_apps) and their experiments
(voxanim.study_phys_apps): the layout is clean, the loop closes, every phase says what
kind of evidence it shows and the first says that digits stand in for events, the
numbers on screen are the study's, each study's claim holds as a relative claim with a
margin over seeds, and the recorded arrays are what they say they are."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_phys_apps as M
from voxanim import study_phys_apps as S
from voxanim.next_catalog import build_study

NET, SCORE, TREES, CHECK = ("network-regression", "classifier-score-as-variable",
                            "boosted-regression-trees", "overtraining-check")
EACH = [pytest.param(name, marks=needs_mnist) for name in (NET, SCORE, TREES, CHECK)]


def study(name):
    return build_study("study_phys_apps", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts]


# ---------------------------------------------------------------------- all four
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        assert any(t in (M.MEASURED, M.TOYS) for t in texts(a, k))
    # no physics data: said on screen in one plain line, with the physics counterpart
    assert M.STAND_IN in texts(a, 0)
    assert "stand in for events" in subtitles(a)[0] and "analysis" in subtitles(a)[0]


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3
    assert m["designs_tried"] and m["limitations"] and m["evidence"]["measured"]
    assert "default_rng" in m["where_the_seed_enters"]
    assert "stand in" in m["stand_in"] and "analysis" in m["physics_counterpart"]


# ---------------------------------------------------------------------- no data needed
def test_rotation_is_anticlockwise_and_bilinear():
    rng = np.random.default_rng(0)
    img = rng.random((2, 28, 28))
    assert np.allclose(S.rotate(img, 0.0), img)
    assert np.allclose(S.rotate(img, 90.0), np.rot90(img, 1, (1, 2)))    # anticlockwise
    assert np.allclose(S.rotate(img, [90.0, -90.0])[1], np.rot90(img[1], -1))
    dot = np.zeros((1, 28, 28))
    dot[0, 13, 20] = 1.0                              # to the right of the centre
    up = S.rotate(dot, 30.0)[0]                       # anticlockwise: it goes up the picture
    rows, cols = np.indices((28, 28))
    assert (up * rows).sum() / up.sum() < 11.0 and (up * cols).sum() / up.sum() < 20.0
    flat = S.rotate(np.ones((1, 28, 28)), 17.0)[0]    # the four weights of a pixel sum to one
    assert np.allclose(flat[8:20, 8:20], 1.0) and flat.max() <= 1.0 + 1e-12
    corner = np.zeros((1, 28, 28))
    corner[0, 0, 0] = 1.0
    assert S.rotate(corner, 45.0).sum() == 0.0        # turned out of the picture: blank


def test_cells_and_tilt():
    img = np.arange(784.0).reshape(1, 28, 28)
    c = S.cells(img, 7)
    assert c.shape == (1, 16) and np.isclose(c[0, 0], img[0, :7, :7].mean())
    assert S.cells(img, 4).shape == (1, 49) == (1, S.REG_CELLS)
    bar = np.zeros((1, 28, 28))
    bar[0, 4:24, 13:15] = 1.0                         # an upright bar
    assert abs(S.tilt(bar)[0]) < 1e-9
    for angle in (-30.0, 12.0, 40.0):                 # turning the bar turns its tilt
        assert abs(S.tilt(S.rotate(bar, angle))[0] - angle) < 1.0


@pytest.mark.parametrize("loss", ["square", "logistic"])
def test_backpropagation_matches_finite_differences(loss):
    rng = np.random.default_rng(3)
    P = S.net_init(rng, (5, 4, 3, 1))
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    x = rng.normal(size=(7, 5))
    y = rng.normal(size=7) if loss == "square" else (rng.random(7) < 0.5) * 1.0
    fd_check(lambda Q: S.net_loss(Q, x, y, loss), P, S.net_grads(P, x, y, loss), list(P))
    assert S.net_output(P, x).shape == (7,)           # one unit, no squashing


def test_training_lowers_the_loss_and_weight_decay_shrinks_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 0.3
    runs = {}
    for decay in (0.0, 2.0):
        P = S.net_init(np.random.default_rng(2), (3, 16, 1))
        first = S.net_loss(P, x, y, "square")
        seen = S.train_net(P, x, y, "square", np.random.default_rng(4), 300, 20, 0.01,
                           (0, 150, 300), lambda Q: S.net_loss(Q, x, y, "square"), decay)
        assert len(seen) == 3 and seen[0] == first and seen[-1] < 0.3 * first
        runs[decay] = float(np.abs(P["W0"]).sum()), seen[-1]
    assert runs[2.0][0] < 0.9 * runs[0.0][0]          # smaller weights ...
    assert runs[0.0][1] < 0.1 * first                 # ... and without it the fit is close


def test_a_tree_finds_a_step_and_boosting_fits_the_residuals():
    x = np.linspace(-89.0, 89.0, 400)
    y = np.where(x < 20.0, -10.0, 30.0)
    xb = S.binned(x, -90.0, 90.0, 18)[:, None]
    assert xb.min() == 0 and xb.max() == 17
    feature, cut, leaf = S.fit_tree(xb, y, 1, 18)
    assert feature[0] == 0 and S.threshold(cut[0], -90.0, 90.0, 18) == 20.0
    assert np.allclose(leaf, [-10.0, 30.0])
    assert np.allclose(S.tree_predict(xb, (feature, cut, leaf)), y)
    start, trees, fitted, (watch,) = S.boost(xb, y, 1, 0.5, 6, (xb,), 18)
    assert np.isclose(start, y.mean()) and len(trees) == 6
    assert np.allclose(fitted, watch)
    # each tree is the residual's step; half of it is added: the error halves every time
    err = np.sqrt(((fitted - y) ** 2).mean(1))
    assert np.allclose(err[1:] / err[:-1], 0.5)
    assert np.allclose(fitted[1], start + 0.5 * S.tree_predict(xb, trees[0]))
    # two features, depth two: the second level splits on the other feature
    rng = np.random.default_rng(0)
    two = rng.integers(0, 16, (500, 2))
    target = 3.0 * (two[:, 0] > 7) + 1.0 * (two[:, 1] > 3)
    feature, cut, leaf = S.fit_tree(two, target, 2)
    assert feature.tolist() == [0, 1, 1] and cut.tolist() == [7, 3, 3]
    assert np.allclose(leaf, [0.0, 1.0, 3.0, 4.0])
    flat = S.fit_tree(two, np.ones(500), 2)           # nothing to gain: everything goes left
    assert np.allclose(S.tree_predict(two, flat), 1.0)


def test_roc_area_and_efficiencies():
    assert S.auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and S.auc([0.1], [0.5, 0.9]) == 0.0
    assert S.auc([0.5, 0.5], [0.5]) == 0.5            # ties count a half
    assert np.isclose(S.auc([0.3, 0.7], [0.5, 0.1]), 0.75)
    eff_s, eff_b = S.efficiencies([0.2, 0.6, 0.9, 0.95], [0.1, 0.3, 0.7], [0.0, 0.5, 0.92])
    assert np.allclose(eff_s, [1.0, 0.75, 0.25]) and np.allclose(eff_b, [1.0, 1 / 3, 0.0])
    assert np.isclose(S.significance(100.0, 300.0), 5.0)
    table = S.cut_table(np.array([0.2, 0.6, 0.9, 0.95]), np.array([0.1, 0.3, 0.7]), 50)
    half = int(np.flatnonzero(S.CUTS == 0.5)[0])
    assert np.isclose(table["purity"][half], 750.0 / (750.0 + 50000.0 / 3))


def test_kolmogorov_smirnov_distance_and_its_chance():
    assert S.ks_distance([1, 2, 3], [1, 2, 3])[0] == 0.0
    assert S.ks_distance([1, 2, 3], [7, 8])[0] == 1.0
    d, at = S.ks_distance([0.1, 0.4, 0.7, 0.9], [0.5, 0.6, 0.8, 0.95])
    assert np.isclose(d, 0.5) and at == 0.4          # two of four below, none of the others
    # Kolmogorov's distribution at its textbook points (5%, 10% and 1%)
    assert np.allclose(S.kolmogorov_tail([1.3581, 1.2238, 1.6276]), [0.05, 0.10, 0.01],
                       atol=2e-4)
    assert S.kolmogorov_tail(0.1)[0] == 1.0 and S.kolmogorov_tail(3.0)[0] < 1e-6
    assert np.isclose(S.ks_critical(100, 400), 1.3581 * np.sqrt(500 / 40000), atol=1e-4)
    assert np.isclose(S.ks_p_value(S.ks_critical(100, 400), 100, 400), 0.05, atol=1e-6)
    # and against samples of one distribution: about 5% exceed the 5% distance
    rng = np.random.default_rng(5)
    crit = S.ks_critical(100, 2000)
    over = np.mean([S.ks_distance(rng.random(100), rng.random(2000))[0] > crit
                    for _ in range(2000)])
    assert 0.025 < over < 0.065


def test_the_mixture_fit_recovers_an_exact_mixture():
    sig = np.array([0.05, 0.05, 0.1, 0.2, 0.6])
    bkg = np.array([0.5, 0.25, 0.15, 0.07, 0.03])
    for f in (0.05, 0.2, 0.5):
        counts = 1000.0 * (f * sig + (1 - f) * bkg)
        fit, sigma = S.fit_fraction(counts, sig, bkg)
        assert np.isclose(fit[0], f, atol=1e-9)
        grid = np.array([f - 0.02, f, f + 0.02])
        lnl = S.log_likelihood(counts, sig, bkg, grid)
        assert lnl[1] > lnl[0] and lnl[1] > lnl[2]
        # one sigma: where ln L is one half lower
        drop = lnl[1] - S.log_likelihood(counts, sig, bkg, [f + sigma[0]])[0]
        assert 0.4 < drop < 0.6
    only, _ = S.fit_fraction(np.array([1000.0 * bkg, 1000.0 * sig]), sig, bkg)
    assert np.allclose(only, [0.0, 1.0], atol=1e-9)
    shape = S.shapes(np.array([0.01, 0.02, 0.99]), 4)
    assert np.isclose(shape.sum(), 1.0) and shape.min() > 0


def test_labels_and_the_stretched_axis():
    assert M.deg(-3.04) == "−3.0°" and M.deg(12.34) == "+12.3°" and M.deg(-40.0, False) == "−40°"
    assert M.plain(2.84) == "2.8" and M.plain(-15.42) == "(−15.4)"
    z = M.stretch(np.array([0.0, 1e-9, 0.25, 0.5, 0.75, 1.0]))
    assert np.all(np.diff(z) >= 0) and z[3] == 0.0 and z[0] == -M.STRETCH and z[-1] == M.STRETCH
    assert np.isclose(S.sigmoid(z[2]), 0.25)


# ---------------------------------------------------------------------- 150: regression
@needs_mnist
def test_regression_events_are_real_turned_and_disjoint():
    te, tr = S.reg_heldout(), S.reg_train(0)
    assert len(te["angle"]) == S.REG_TEST and len(tr["angle"]) == S.REG_TRAIN
    assert len(np.unique(te["index"])) == S.REG_TEST and len(np.unique(tr["index"])) == S.REG_TRAIN
    assert np.abs(te["angle"]).max() <= S.REG_RANGE and np.abs(tr["angle"]).max() <= S.REG_RANGE
    assert abs(te["angle"].mean()) < 1.5 and 25.0 < te["angle"].std() < 27.0     # uniform
    from voxanim import mnist
    xtr, _, xte, yte = mnist.load()                   # training file against test file
    assert np.array_equal(te["original"][5], xte[te["index"][5]].reshape(28, 28))
    assert np.array_equal(te["label"], yte[te["index"]])
    again = S.rotate(xte[te["index"][:3]], te["angle"][:3])
    assert np.array_equal(again, te["image"][:3])
    assert np.allclose(te["x"][:3], S.cells(again, S.REG_BLOCK))
    assert np.allclose(tr["x"][:2], S.cells(S.rotate(xtr[tr["index"][:2]], tr["angle"][:2]), 4))
    assert not np.array_equal(S.reg_train(1)["index"], tr["index"])


@needs_mnist
def test_regression_numbers_on_screen_are_the_study():
    s, a = study(NET), anim(NET)
    m, subs = s.metadata, subtitles(a)
    r, e, ex = m["resolution"], m["edge_bias"], m["example"]
    assert "σ = %.1f° (%.1f to %.1f" % (r["mean"], r["min"], r["max"]) in subs[3]
    assert "%.1f°" % m["guess_zero_rms"] in subs[3]
    assert "from %.1f° to %.1f°" % (m["before_training"]["rms"], m["held_rms"]) in subs[2]
    assert M.deg(e["lowest_bin"]) in subs[4] and M.deg(e["highest_bin"]) in subs[4]
    assert "ŷ = " + M.deg(ex["after"]) in subs[1] and "%.1f" % ex["loss_after"] in subs[1]
    # the worked example: the terms drawn add up to the prediction, and its loss
    after = a.ex[1]
    assert np.isclose(after["terms"].sum() + after["bias"], after["prediction"])
    assert np.isclose(after["prediction"], ex["after"]) and np.isclose(after["truth"],
                                                                        ex["truth"])
    assert np.isclose(after["loss"], (ex["after"] - ex["truth"]) ** 2)
    assert np.isclose(after["slope"], 2 * (ex["after"] - ex["truth"]))
    shown = texts(a, 1, 0.9)
    assert "ŷ = " + M.deg(ex["after"]) in shown
    assert "= (%s − %s)² = %.1f" % (M.plain(ex["after"]), M.plain(ex["truth"]),
                                    ex["loss_after"]) in shown
    before = texts(a, 1, 0.4)
    assert "ŷ = " + M.deg(ex["before"]) in before and "before any training" in before
    # the prediction recorded during training is the one recomputed from the saved weights
    d = S.net_data()
    assert np.isclose(d["held"][0, -1, ex["event"]], ex["after"], atol=1e-3)
    assert np.isclose(d["held"][0, 0, ex["event"]], ex["before"], atol=1e-3)
    assert ex["event"] == S.typical_event() < S.NET_SCATTER


@needs_mnist
def test_regression_resolves_the_angle_and_is_pulled_to_the_middle():
    s = study(NET)
    m = s.metadata
    held, train = s.panels[0].data["per_seed"], s.panels[0].data["per_seed_train"]
    pred = s.panels[1].data["xy"][:, 1]
    assert np.allclose(pred, s.panels[1].data["snapshots"][-1])
    # every seed: far better than always answering 0, and than its untrained self
    assert max(m["resolution"]["per_seed"]) < 0.5 * m["guess_zero_rms"]
    assert held[:, -1].max() < 0.5 * held[:, 0].min()
    assert train[:, -1].mean() < held[:, -1].mean()   # held-out is the harder number
    # the residual distribution narrows; its final width and mean are the resolution and bias
    p2 = s.panels[2].data
    assert np.allclose(p2["values"].sum(1), 1.0) and p2["width"][-1] < 0.5 * p2["width"][0]
    res = S.net_data()["held"][0, -1] - S.reg_heldout()["angle"]
    assert np.isclose(res.std(), p2["width"][-1]) and np.isclose(res.mean(), p2["mean"][-1])
    assert np.isclose(m["resolution"]["per_seed"][0], res.std())
    # bias against the true value: positive at the low end, negative at the high end, in
    # every seed, by more than the bias in the middle
    bias, width = s.panels[3].data["bias"], s.panels[3].data["width"]
    assert bias[:, 0].min() > 3.0 and bias[:, -1].max() < -3.0
    assert bias[:, 0].min() > np.abs(bias[:, 4]).max() and -bias[:, -1].max() > np.abs(
        bias[:, 4]).max()
    assert width.mean(0).max() < 1.3 * width.mean(0).min()        # the resolution is flat
    b, w = S.resolution(S.net_data()["held"][1, -1], S.reg_heldout()["angle"])
    assert np.allclose(b, bias[1]) and np.allclose(w, width[1])


# ---------------------------------------------------------------------- 152: boosted trees
@needs_mnist
def test_trees_numbers_on_screen_are_the_study():
    s, a = study(TREES), anim(TREES)
    m, subs = s.metadata, subtitles(a)
    one, best, net = m["one_feature"], m["best"], m["network"]
    assert "from %.1f° to %.1f° to %.1f°" % tuple(one["train_rms"][:3]) in subs[2]
    assert "lowest after %d trees (%.1f°)" % (one["held_lowest_at"], one["held_lowest"]) in subs[3]
    assert "rate %g and depth %d, reaches %.1f°" % (best["rate"], best["depth"],
                                                    best["held"]) in subs[5]
    assert "%.1f° (%d seeds each)" % (net["held"], len(m["seeds"])) in subs[5]
    table = texts(a, 5, 0.9)
    for v in s.panels[2].data["values"]:
        assert "%.1f" % v in table
    assert "%.1f°" % best["held"] in table and "%.1f°" % net["held"] in table
    # the tree drawn is the first tree fitted: its questions and leaves
    drawn = texts(a, 1, 0.55)
    p0 = s.panels[0].data
    for cut in p0["cuts"][0]:
        assert "tilt < %s?" % M.deg(cut, False) in drawn
    for leaf in p0["leaves"][0]:
        assert M.deg(leaf) in drawn


@needs_mnist
def test_the_staircase_is_boosting_refitted_from_the_points():
    p0 = study(TREES).panels[0].data
    x, y = p0["xy"].T
    tr = S.reg_train(0)
    assert np.allclose(x, tr["tilt"][:S.ONE_N]) and np.allclose(y, tr["angle"][:S.ONE_N])
    assert np.corrcoef(x, y)[0, 1] > 0.3              # the feature does say something
    xb = S.binned(x, *S.ONE_RANGE, S.ONE_BINS)[:, None]
    grid = S.binned(p0["grid"], *S.ONE_RANGE, S.ONE_BINS)[:, None]
    start, _, fitted, (steps,) = S.boost(xb, y, S.ONE_DEPTH, S.ONE_RATE, S.ONE_TREES,
                                         (grid,), S.ONE_BINS)
    assert np.isclose(start, p0["start"]) and np.allclose(steps, p0["steps"])
    assert np.allclose(y - fitted, p0["residuals"])
    # tree m + 1 is fitted to the residuals after m trees, and added at the learning rate
    second = S.fit_tree(xb, p0["residuals"][1], S.ONE_DEPTH, S.ONE_BINS)
    assert np.allclose(S.tree_predict(grid, second), p0["alone"][1])
    assert np.allclose(p0["steps"][2], p0["steps"][1] + S.ONE_RATE * p0["alone"][1])
    assert np.allclose(p0["leaves"][1], second[2])
    # a leaf is the mean residual of its events
    left = x < p0["cuts"][0][0]
    lowest = left & (x < p0["cuts"][0][1])
    assert np.isclose(p0["leaves"][0][0], (y - start)[lowest].mean())
    assert np.all(np.diff(p0["train_rms"]) <= 1e-9)   # the training error can only fall
    assert p0["held_rms"].argmin() < S.ONE_TREES and p0["held_rms"].min() < p0["held_rms"][0] - 3


@needs_mnist
def test_trees_overfit_at_full_rate_and_trail_the_network():
    s = study(TREES)
    m, p1, p2 = s.metadata, s.panels[1].data, s.panels[2].data
    curves = p2["all_curves"]                         # rate, depth, trees: mean of seeds
    rates, depths = m["rates"], m["depths"]
    full, slow, deep = rates.index(1.0), rates.index(0.03), len(depths) - 1
    assert curves.shape == (len(rates), len(depths), S.BRT_TREES + 1)
    assert np.allclose(curves[:, :, 0], curves[0, 0, 0])          # all start from the mean
    assert np.allclose(p1["ys"], curves[:, depths.index(m["main"]["depth"])])
    # a small learning rate is still improving at the end; at rate 1 deep trees turn up early
    assert curves[slow].argmin(-1).min() == S.BRT_TREES
    assert curves[full, deep].argmin() < 100
    assert curves[full, deep, -1] > curves[full, deep].min() + 0.5
    per_seed = S.brt_data()["held"][full, deep]
    assert np.all(per_seed[:, -1] > per_seed.min(1) + 0.5)        # in every seed
    # depth and rate matter: the best setting beats the worst by a wide margin
    assert p2["values"].max() > p2["values"].min() + 4.0
    # training error is below held-out error for the best setting
    best, net, one = m["best"], m["network"], m["one_feature"]
    assert best["train"] < best["held"] - 1.0
    # same events as the network: its numbers here are the regression study's
    reg = study(NET).panels[0].data["per_seed"][:len(m["seeds"]), -1]
    assert np.allclose(net["per_seed"], reg)
    # every seed: the network is ahead of the best trees, and both far ahead of one feature
    assert max(net["per_seed"]) < min(best["per_seed"]) - 0.3
    assert best["held"] < one["held_lowest"] - 5.0 < curves[0, 0, 0] - 8.0
    # the comparison by true angle is of the same held-out events
    p3 = s.panels[3].data
    truth = S.reg_heldout()["angle"]
    i, j = rates.index(best["rate"]), depths.index(best["depth"])
    per = [S.resolution(S.brt_data()["final"][i, j, k], truth) for k in range(len(m["seeds"]))]
    assert np.allclose(np.mean([w for _, w in per], 0), p3["ys"][0])
    assert np.allclose([b for b, _ in per], p3["trees_bias"])
    assert np.allclose(p3["ys"][1], study(NET).panels[3].data["width"][:len(m["seeds"])].mean(0))


@needs_mnist
def test_one_boosted_run_repeats():
    d = S.brt_data()
    i, j = S.BRT_RATES.index(0.3), S.BRT_DEPTHS.index(1)
    train, held, final = S.brt_run(1, 0.3, 1)
    assert np.allclose(train, d["train"][i, j, 1]) and np.allclose(held, d["held"][i, j, 1])
    assert np.allclose(final, d["final"][i, j, 1], atol=1e-4)
    truth = S.reg_heldout()["angle"]
    assert np.isclose(held[-1], np.sqrt(((final - truth) ** 2).mean()))


# ---------------------------------------------------------------------- 151: the score
@needs_mnist
def test_signal_and_background_are_real_and_disjoint():
    from voxanim import mnist
    _, ytr, _, yte = mnist.load()
    d = S.cls_data()
    for g, digit in enumerate((S.CLS_SIGNAL, S.CLS_BACKGROUND)):
        total = int((ytr == digit).sum() + (yte == digit).sum())
        held = int((d["held_y"] == 1 - g).sum())
        assert len(d["pool"][g]) == S.CLS_POOL and held == total - S.CLS_POOL
    assert d["held_x"].shape[1] == 16 and len(d["held_x"]) == len(d["held_y"])
    x, y, v, vy = S.cls_draw(100, 0, 50)
    assert x.shape == (200, 16) and v.shape == (100, 16) and y.sum() == 100 and vy.sum() == 50
    rows = {r.tobytes() for r in x}
    assert not rows & {r.tobytes() for r in v}        # validation is never trained on
    assert not np.array_equal(S.cls_draw(100, 1)[0], x)
    m = study(SCORE).metadata
    assert m["heldout"] == dict(signal=int(d["held_y"].sum()),
                                background=int((1 - d["held_y"]).sum()))


@needs_mnist
def test_score_numbers_on_screen_are_the_study():
    s, a = study(SCORE), anim(SCORE)
    m, subs = s.metadata, subtitles(a)
    c, b, f = m["cut_at_half"], m["best_cut"], m["fit"]
    assert "keeps %s of the signal" % M.pct(c["signal_efficiency"][0]) in subs[2]
    assert "rejects %s of the background" % M.pct(c["background_rejection"][0]) in subs[2]
    assert "%.3f (%.3f to %.3f" % (m["auc"]["mean"], m["auc"]["min"], m["auc"]["max"]) in subs[3]
    assert "moves up to %.2f: %s pure" % (b["imbalanced"][0],
                                          M.pct(b["purity_imbalanced"][0])) in subs[4]
    k = S.FIT_SHOWN
    seed0 = S.score_data()["fit_fitted"][0, k]        # the classifier whose scores are drawn
    assert "give %.3f ± %.3f" % (seed0.mean(), seed0.std()) in subs[5]
    assert "%s samples" % M.commas(f["toys"]) in subs[5] and len(seed0) == f["toys"]
    # the cut's arithmetic, in events, from the held-out scores themselves
    d, y = S.score_data(), S.cls_data()["held_y"]
    score = S.sigmoid(d["held"][0])
    kept_s, kept_b = int((score[y == 1] > 0.5).sum()), int((score[y == 0] > 0.5).sum())
    n_s, n_b = m["heldout"]["signal"], m["heldout"]["background"]
    assert np.isclose(kept_s / n_s, c["signal_efficiency"][0])
    assert np.isclose(1 - kept_b / n_b, c["background_rejection"][0])
    shown = texts(a, 2, 0.4)
    assert "= %s of %s kept = %s" % (M.commas(kept_s), M.commas(n_s),
                                     M.pct(kept_s / n_s)) in shown
    assert "= %s of %s removed = %s" % (M.commas(n_b - kept_b), M.commas(n_b),
                                        M.pct(1 - kept_b / n_b)) in shown
    # the histograms are those scores, and the example scores are the first held-out events
    hist = s.panels[0].data
    assert np.allclose(hist["values"].sum(1), 1.0)
    assert np.allclose(hist["values"][0] * n_s, np.histogram(score[y == 1], hist["edges"])[0])
    assert np.allclose(hist["example_scores"][0], score[:S.CLS_PICTURES])
    assert np.allclose(hist["example_scores"][1], score[n_s:n_s + S.CLS_PICTURES])
    # the fit drawn is the first pseudo-experiment, refitted here
    p3 = s.panels[3].data
    fit, sigma = S.fit_fraction(p3["first"][k], p3["signal_shape"], p3["background_shape"])
    assert np.isclose(fit[0], p3["fitted_shown"][0]) and np.isclose(a.fit, fit[0])
    assert np.isclose(sigma[0], p3["error_shown"][0]) and p3["first"][k].sum() == S.FIT_EVENTS
    assert "f = %.3f ± %.3f" % (fit[0], sigma[0]) in texts(a, 5, 0.9)
    assert M.TOYS in texts(a, 5) and M.MEASURED in texts(a, 4)


@needs_mnist
def test_the_score_separates_overlaps_and_the_best_cut_moves():
    s = study(SCORE)
    m = s.metadata
    r = S.score_summary()
    # the classes separate, on held-out digits, in every seed, yet overlap
    assert r["auc"].min() > 0.9 and r["auc"].max() < 0.99
    assert m["auc"]["train"] < m["auc"]["mean"] + 0.02            # no overtraining to speak of
    eff_s, eff_b = s.panels[1].data["per_seed_signal"], s.panels[1].data["per_seed_background"]
    assert np.all(np.diff(eff_s, axis=1) <= 0) and np.all(np.diff(eff_b, axis=1) <= 0)
    assert np.all(eff_s[:, 1:-1] > eff_b[:, 1:-1])                # the ROC is above the diagonal
    # the ROC area is the area under efficiency against efficiency
    area = -np.trapezoid(eff_s[0], eff_b[0]) if hasattr(np, "trapezoid") else -np.trapz(
        eff_s[0], eff_b[0])
    assert abs(area - r["auc"][0]) < 0.01
    b, c = m["best_cut"], m["cut_at_half"]
    for k in range(len(m["seeds"])):                  # in every seed
        assert b["imbalanced"][k] > b["even"][k] + 0.25           # the best cut moves up
        assert c["purity_imbalanced"][k] < c["purity_even"][k] - 0.5
        assert b["purity_imbalanced"][k] > c["purity_imbalanced"][k] + 0.3
        assert b["z_imbalanced"][k] > b["z_imbalanced_at_half"][k] * 1.3
        assert b["signal_efficiency_imbalanced"][k] < c["signal_efficiency"][k] - 0.1
    z = s.panels[2].data["ys"][1]
    assert np.isclose(z.max(), b["z_imbalanced"][0])
    assert np.isclose(S.CUTS[int(z.argmax())], b["imbalanced"][0])


@needs_mnist
def test_the_shape_fit_recovers_the_fraction():
    m = study(SCORE).metadata
    f = m["fit"]
    true = np.array(f["fractions"])
    mean, spread, counted = np.array(f["mean"]), np.array(f["spread"]), np.array(
        f["counted_spread"])
    assert f["toys"] >= 2000
    assert np.abs(mean - true).max() < 0.012 and f["worst_offset"] < 0.02
    assert np.all(np.abs(mean - true) < np.abs(np.array(f["counted_mean"]) - true))
    # the spread of the fitted fraction is what the likelihood says it is ...
    assert np.allclose(spread, f["likelihood_error"], rtol=0.06)
    # ... narrower than counting above a cut, wider than if every event's class were known
    assert np.all(spread < 0.95 * counted) and np.all(spread > 1.1 * np.array(
        f["perfect_spread"]))
    d = S.score_data()
    assert d["fit_fitted"].shape == (len(m["seeds"]), len(true), f["toys"])
    assert np.all(d["fit_first"].sum(-1) == f["events"])
    for k in range(len(m["seeds"])):                  # in every seed's classifier
        assert np.abs(d["fit_fitted"][k].mean(-1) - true).max() < 0.02
        assert np.all(d["fit_fitted"][k].std(-1) < d["fit_counted"][k].std(-1))
    # the samples really hold about the stated fraction of signal events
    assert np.abs(d["fit_first_signal"] / f["events"] - true).max() < 0.06


# ---------------------------------------------------------------------- 153: overtraining
@needs_mnist
def test_overtraining_numbers_on_screen_are_the_study():
    s, a = study(CHECK), anim(CHECK)
    m, subs = s.metadata, subtitles(a)
    end, stop, decay = m["rows"]
    i, j = m["sizes"].index(m["small"]), m["sizes"].index(m["large"])
    ks, gap, auc = m["ks"], m["auc_gap"], m["heldout_auc"]
    assert "D = %.2f" % m["example"]["distance"] in subs[2]
    assert "%.2f apart" % m["critical"][i] in subs[2]
    assert "from %.2f at %d a class to %.2f" % (ks[end]["mean"][0], m["sizes"][0],
                                                ks[end]["mean"][j]) in subs[3]
    assert "from %.3f to %.3f" % (gap[end]["mean"][0], gap[end]["mean"][j]) in subs[3]
    assert "from %.2f to %.2f; weight decay to %.2f" % (
        ks[end]["mean"][i], ks[stop]["mean"][i], ks[decay]["mean"][i]) in subs[4]
    assert "%.3f against %.3f" % (auc[decay]["mean"][j], auc[end]["mean"][j]) in subs[4]
    # the distance drawn is the largest gap of the two curves drawn
    p0 = s.panels[0].data
    d, at = S.ks_distance(p0["example_train"], p0["example_held"])
    assert np.isclose(d, m["example"]["distance"]) and at == m["example"]["at"]
    assert len(p0["example_train"]) == m["small"]
    assert "D = %.2f" % d in texts(a, 2, 0.9)
    assert np.isclose(m["example"]["p_value"], S.ks_p_value(d, m["small"], len(p0["example_held"])))
    # stretching the axis does not change the distance
    assert np.isclose(S.ks_distance(M.stretch(p0["example_train"]),
                                    M.stretch(p0["example_held"]))[0], d, atol=0.02)
    # the curves' ends are the bars' numbers
    p1, p2, p3 = (p.data for p in s.panels[1:])
    assert np.isclose(p1["ys"][0][-1], p2["ys"][0][i]) and np.isclose(p1["ys"][1][-1],
                                                                       p3["ys"][0][i])
    assert np.allclose(p2["ys"][:, i], [ks[n]["mean"][i] for n in m["rows"]])
    assert np.allclose(p2["critical"], m["critical"])
    assert np.isclose(m["critical"][i], np.mean([S.ks_critical(m["small"], n) for n in
                                                 S.ot_summary()["n_held"]]))


@needs_mnist
def test_training_scores_pile_up_only_on_the_small_sample():
    s = study(CHECK)
    p0 = s.panels[0].data
    hist = p0["values"].reshape(2, 2, 2, -1)          # case, class, set, bin
    assert np.allclose(hist.sum(-1), 1.0) and np.allclose(p0["during"][-1], hist[0])
    ends = hist[..., 0] + hist[..., -1]               # the share in the two end bins
    assert ends[0, :, 0].min() > 0.97                 # small sample: training at 0 or 1 ...
    assert ends[0, :, 1].max() < ends[0, :, 0].min() - 0.05       # ... held-out not
    assert hist[0, 0, 0, -1] > 0.97 and hist[0, 1, 0, 0] > 0.97   # each at its own end
    assert hist[0, 0, 1, 0] > 0.03                    # held-out signal sure of the wrong answer
    assert np.abs(hist[1, :, 0] - hist[1, :, 1]).max() < 0.05     # large sample: they agree
    assert np.abs(hist[0, :, 0] - hist[0, :, 1]).max() > 0.15
    # during training the pile grows
    during = p0["during"]
    pile = during[:, 0, 0, -1]
    assert pile[0] < 0.6 and pile[-1] > pile[0] + 0.3


@needs_mnist
def test_distance_and_gap_fall_with_sample_size_and_the_remedies_help():
    s = study(CHECK)
    m = s.metadata
    p1, p2, p3 = (p.data for p in s.panels[1:])
    ks, gap, auc = p2["per_seed"], p3["per_seed"], p3["held_per_seed"]   # row, size, seed
    i, j = m["sizes"].index(m["small"]), m["sizes"].index(m["large"])
    crit = np.array(m["critical"])
    assert ks.shape == (3, len(m["sizes"]), len(m["seeds"]))
    # trained to the end: in every seed, the smallest sample is far from held-out and the
    # largest close; the small one beyond what chance allows
    assert ks[0, 0].min() > ks[0, j].max() + 0.2 and gap[0, 0].min() > gap[0, j].max() + 0.05
    assert np.all(np.diff(ks[0].mean(-1)) < 0) and np.all(np.diff(gap[0].mean(-1)) < 0)
    assert ks[0, i].min() > crit[i] + 0.03 and ks[0, j].max() < 0.06
    assert np.all(np.diff(crit) < 0) and 0.25 < crit[0] < 0.30
    # during training on the small sample both grow
    assert p1["ys"][0][-1] > p1["ys"][0][0] + 0.1 and p1["ys"][1][-1] > p1["ys"][1][0] + 0.05
    # while the held-out ROC area does not get better after the early stop
    assert p1["held_auc"][-1] < p1["held_auc"].max() + 1e-9
    assert p1["valid"][-1] > 2 * p1["valid"].min()    # and the validation loss turns up
    # the remedies, at the small sample, mean of seeds: distance and gap are both lower
    for row in (1, 2):
        assert ks[row, i].mean() < ks[0, i].mean() - 0.07
        assert gap[row, i].mean() < gap[0, i].mean() - 0.015
        assert ks[row, i].mean() < crit[i] + 0.01     # back to about what chance allows
    # early stopping does not cost held-out ROC area; this weight decay does at large sizes
    assert auc[1, i].mean() > auc[0, i].mean() - 0.005
    assert auc[2, j].mean() < auc[0, j].mean() - 0.02
    assert np.all(auc[0].mean(-1)[1:] > auc[0].mean(-1)[:-1])     # more events, better network


@needs_mnist
def test_one_overtraining_run_repeats():
    d = S.ot_data()
    i = S.OT_SIZES.index(50)
    again = S.ot_run(50, 2, 0.0)
    assert np.allclose(again["ks"], d["ks"][0, i, 2]) and np.allclose(again["auc"],
                                                                      d["auc"][0, i, 2])
    assert np.allclose(again["hist"], d["hist"][0, i, 2])
    assert again["ks"].shape == (len(S.OT_SNAPS), 2) and S.OT_SNAPS[-1] == S.OT_UPDATES
