"""Animations 165-169 (voxanim.mech_boost) and their experiments (voxanim.study_boost):
the pieces of the boosting code are right when checked against brute force (the gain
formula, the histogram split against the exact one, the subtraction trick, the default
direction for missing values, pruning, AdaBoost's weights); the layout is clean, the
loop closes, every phase says what kind of evidence it shows and the first that digits
stand in for events; the numbers on screen are the study's; and each study's claim
holds as a relative claim with a margin over seeds."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_boost as M
from voxanim import study_boost as S
from voxanim.next_catalog import build_study

ADA, GB, GAIN, HIST, ROUND = ("adaboost", "boosting-shrinkage-and-early-stopping",
                              "xgboost-split-gain", "xgboost-histograms-and-missing-values",
                              "xgboost-one-round")
NAMES = (ADA, GB, GAIN, HIST, ROUND)
EACH = [pytest.param(name, marks=needs_mnist) for name in NAMES]
XGB = [pytest.param(name, marks=needs_mnist) for name in (GAIN, HIST, ROUND)]


def study(name):
    return build_study("study_boost", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts]


def node(seed=0, n=40, f=3, levels=6, missing=0.0):
    """A small random node: binned features, slopes and curvatures."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, levels, (n, f)).astype(float)
    if missing:
        x[rng.random((n, f)) < missing] = np.nan
    edges = S.value_edges(x)
    bins = S.width(edges)
    y = (rng.random(n) < 0.5) * 1.0
    g, h = S.slopes(rng.normal(0, 1.5, n), y)
    return x, S.to_bins(x, edges, bins), bins, g, h, y


def brute_leaf(G, H, lam):
    """The lowest value of ``G w + (H + lambda) w^2 / 2`` found by trying many w."""
    w = np.linspace(-40.0, 40.0, 1600001)
    return float((G * w + 0.5 * (H + lam) * w * w).min())


# ---------------------------------------------------------------------- the pieces
def test_slopes_are_the_derivatives_of_the_log_loss():
    score = np.array([-2.0, -0.3, 0.0, 0.7, 3.0])
    for y in (0.0, 1.0):
        g, h = S.slopes(score, np.full(5, y))
        e = 1e-4
        up, mid, down = (S.event_loss(score + d, y) for d in (e, 0.0, -e))
        assert np.allclose(g, (up - down) / (2 * e), atol=1e-7)
        assert np.allclose(h, (up - 2 * mid + down) / e ** 2, atol=1e-5)
    assert np.isclose(S.log_loss(np.zeros(4), np.array([0, 1, 1, 0.0])), np.log(2))
    assert S.error_rate(np.array([1.0, -1.0, 2.0]), np.array([1, 1, 0.0])) == 2 / 3
    assert S.roc_area(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0.0])) == 1.0


def test_leaf_value_and_score_are_the_bottom_of_the_parabola():
    for G, H, lam in ((-3.2, 4.1, 0.0), (1.7, 0.6, 1.0), (0.4, 2.0, 10.0)):
        w = S.leaf_value(G, H, lam)
        assert np.isclose(S.approx(G, H, lam, w), -0.5 * S.leaf_score(G, H, lam))
        assert S.approx(G, H, lam, w) <= min(S.approx(G, H, lam, w + 1e-3),
                                             S.approx(G, H, lam, w - 1e-3))
        assert np.isclose(brute_leaf(G, H, lam), -0.5 * S.leaf_score(G, H, lam), atol=1e-6)
    assert abs(S.leaf_value(1.0, 1.0, 10.0)) < abs(S.leaf_value(1.0, 1.0, 0.0))   # lambda shrinks


@pytest.mark.parametrize("lam", [0.0, 1.0, 5.0])
def test_gain_formula_against_brute_force_on_small_nodes(lam):
    for seed in range(4):
        _, xb, bins, g, h, _ = node(seed, n=14, f=2, levels=5)
        rows = np.arange(len(g))
        parent = brute_leaf(g.sum(), h.sum(), lam)
        best = (-np.inf, None)
        table = S.gains(S.histograms(xb, g, h, rows, bins), lam)
        for f in range(xb.shape[1]):
            for cut in range(bins - 1):
                left = xb[:, f] <= cut
                if left.all() or not left.any():
                    assert table[f, cut, 0] == -np.inf       # an empty side: not a split
                    continue
                brute = parent - brute_leaf(g[left].sum(), h[left].sum(), lam) \
                    - brute_leaf(g[~left].sum(), h[~left].sum(), lam)
                formula = S.split_gain(g[left].sum(), h[left].sum(), g[~left].sum(),
                                       h[~left].sum(), lam)
                assert np.isclose(formula, brute, atol=1e-5)
                assert np.isclose(table[f, cut, 0], brute, atol=1e-5)
                if brute > best[0] + 1e-9:
                    best = (brute, (f, cut))
        got = S.scan(S.histograms(xb, g, h, rows, bins), lam)
        assert np.isclose(got[0], best[0], atol=1e-5) and got[1:3] == best[1]
    assert np.isclose(S.split_gain(1.0, 2.0, -3.0, 1.0, 1.0, gamma=0.5),
                      S.split_gain(1.0, 2.0, -3.0, 1.0, 1.0) - 0.5)


def test_histogram_split_is_the_exact_split_when_every_value_has_its_own_bin():
    rng = np.random.default_rng(7)
    n = 240
    for seed in range(5):
        x = np.round(rng.normal(size=(n, 4)), 1 if seed % 2 else 3)
        edges = S.value_edges(x)
        bins = S.width(edges)
        xb = S.to_bins(x, edges, bins)
        assert all(len(np.unique(xb[:, j])) == len(np.unique(x[:, j])) for j in range(4))
        y = (rng.random(n) < 0.5) * 1.0
        g, h = S.slopes(rng.normal(size=n), y)
        rows = np.arange(n)
        for lam, mcw in ((0.0, 0.0), (1.0, 0.0), (1.0, 0.8)):
            hist = S.scan(S.histograms(xb, g, h, rows, bins), lam, mcw)
            exact = S.exact_split(xb, g, h, rows, lam, mcw)
            assert np.isclose(hist[0], exact[0]) and hist[1:3] == exact[1:3]
            sort = S.exact_split(x, g, h, rows, lam, mcw)          # on the raw values too
            assert np.isclose(sort[0], exact[0]) and sort[1] == exact[1]
            assert np.array_equal(x[:, sort[1]] <= sort[2], xb[:, exact[1]] <= exact[2])
        # whole trees (nodes large enough that no two cuts tie for the best)
        a = S.grow(xb, g, h, bins, 2, 1.0, method="hist")
        b = S.grow(xb, g, h, bins, 2, 1.0, method="exact")
        for key in ("feature", "cut", "left", "right", "N"):
            assert np.array_equal(a[key], b[key])
        assert np.allclose(a["value"], b["value"]) and np.allclose(a["gain"], b["gain"])
    # fewer bins: a coarser search, never a better split
    coarse = S.quantile_edges(x, 4)
    cb = S.to_bins(x, coarse, S.width(coarse))
    assert S.scan(S.histograms(cb, g, h, rows, S.width(coarse)))[0] <= exact_gain(x, g, h)


def exact_gain(x, g, h):
    return S.exact_split(x, g, h, np.arange(len(g)))[0] + 1e-12


def test_subtraction_trick_gives_the_same_histograms_and_the_same_tree():
    for seed in range(3):
        _, xb, bins, g, h, _ = node(seed, n=80, f=4, levels=9, missing=0.15 * (seed == 2))
        rows = np.arange(len(g))
        left = rows[xb[:, 0] <= 3]
        right = rows[xb[:, 0] > 3]
        parent = S.histograms(xb, g, h, rows, bins)
        assert np.allclose(parent - S.histograms(xb, g, h, left, bins),
                           S.histograms(xb, g, h, right, bins))
        assert parent[2].sum(1).tolist() == [len(rows)] * xb.shape[1]     # every event once
        with_, without = {}, {}
        a = S.grow(xb, g, h, bins, 4, 1.0, subtract=True, stats=with_)
        b = S.grow(xb, g, h, bins, 4, 1.0, subtract=False, stats=without)
        for key in a:
            assert np.allclose(a[key], b[key])
        assert with_["added"] < without["added"] and with_["scored"] == without["scored"]
        assert without["added"] == sum(n * xb.shape[1] for n, lv in zip(b["N"], b["level"])
                                       if lv < 4 and n > 1)


def test_default_direction_for_missing_values_is_chosen_by_gain():
    # ten events; the three without a reading pull the way the high readings do
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, np.nan, np.nan, np.nan])[:, None]
    g = np.array([1.0, 1.1, 0.9, 1.0, -1.0, -1.2, -0.9, -1.0, -1.1, -0.8])
    h = np.full(10, 0.25)
    edges = S.value_edges(x)
    bins = S.width(edges)
    xb = S.to_bins(x, edges, bins)
    assert (xb[7:, 0] == bins).all() and xb[:7, 0].max() == bins - 1
    rows = np.arange(10)
    hist = S.histograms(xb, g, h, rows, bins)
    assert hist[2, 0, -1] == 3 and np.isclose(hist[0, 0, -1], g[7:].sum())
    gain, f, cut, miss_left = S.scan(hist, 1.0)
    assert (f, cut, miss_left) == (0, 3, False)              # after 0.4, missing to the right
    both = S.gains(hist, 1.0)[0, 3]
    left = S.split_gain(g[[0, 1, 2, 3, 7, 8, 9]].sum(), 7 * 0.25, g[4:7].sum(), 3 * 0.25, 1.0)
    right = S.split_gain(g[:4].sum(), 1.0, g[4:].sum(), 1.5, 1.0)
    assert np.allclose(both, [left, right]) and right > left and np.isclose(gain, right)
    assert S.scan(S.histograms(xb, -g * np.r_[np.ones(7), -np.ones(3)], h, rows, bins),
                  1.0)[3] is True                            # pulled the other way: left
    tree = S.grow(xb, g, h, bins, 1, 1.0)
    assert np.array_equal(S.leaf_of(tree, xb, bins) == tree["right"][0], np.r_[[False] * 4,
                                                                               [True] * 6])
    # a forced side: no choice is made, and the gain cannot be larger
    for side in (0, 1):
        forced = S.scan(hist, 1.0, force=side)
        assert forced[3] is (side == 0) and forced[0] <= gain + 1e-12
    # nothing missing at the node: one split per cut, and it is recorded as "left"
    whole = S.histograms(xb, g, h, rows[:7], bins)
    assert np.isinf(S.gains(whole, 1.0)[..., 1]).all() and S.scan(whole, 1.0)[3] is True
    # a split may put the missing events alone on one side
    lone = S.gains(hist, 0.0)[0, bins - 1]
    assert lone[0] == -np.inf and np.isfinite(lone[1])


def test_growth_pruning_and_the_restraints():
    _, xb, bins, g, h, _ = node(3, n=300, f=5, levels=12)
    full = S.grow(xb, g, h, bins, 4, 1.0)
    inner, leaves = S.reached(full)
    assert len(leaves) == len(inner) + 1 == S.n_leaves(full)
    assert (full["gain"][inner] > 0).all() and (full["level"] <= 4).all()
    for k in inner:                                   # a node's events are its children's
        a, b = full["left"][k], full["right"][k]
        assert full["N"][k] == full["N"][a] + full["N"][b]
        assert np.isclose(full["G"][k], full["G"][a] + full["G"][b])
    assert np.allclose(full["value"], -full["G"] / (full["H"] + 1.0))
    assert np.allclose(S.predict(full, xb, bins), full["value"][S.leaf_of(full, xb, bins)])
    counts = np.bincount(S.leaf_of(full, xb, bins), minlength=len(full["N"]))
    assert np.array_equal(counts[leaves], full["N"][leaves]) and counts[inner].sum() == 0
    # gamma: only splits whose children are leaves and whose gain is below it are undone
    gamma = float(np.median(full["gain"][inner]))
    cut, removed = S.prune(full, gamma)
    assert 0 < removed < len(inner) and S.n_leaves(cut) == len(leaves) - removed
    for k in S.reached(cut)[0]:
        kids = (cut["left"][k], cut["right"][k])
        assert cut["gain"][k] >= gamma or any(cut["left"][c] >= 0 for c in kids)
    assert S.n_leaves(S.prune(full, np.inf)[0]) == 1 and S.prune(full, 0.0)[1] == 0
    # min_child_weight: no leaf with less curvature; a larger value, fewer leaves
    held = S.grow(xb, g, h, bins, 4, 1.0, mcw=3.0)
    leaves_h = S.reached(held)[1]
    assert (held["H"][leaves_h] >= 3.0 - 1e-9).all() and len(leaves_h) < len(leaves)
    # the two orders of growth: the same first split, the same number of leaves
    level = S.grow(xb, g, h, bins, 3, 1.0, leaves=8, policy="level")
    leaf = S.grow(xb, g, h, bins, 7, 1.0, leaves=8, policy="leaf")
    assert S.n_leaves(level) == S.n_leaves(leaf) == 8 and level["level"].max() == 3
    assert level["feature"][0] == leaf["feature"][0] and level["cut"][0] == leaf["cut"][0]
    order = leaf["order"][leaf["order"] >= 0]
    assert sorted(order) == list(range(7))
    chosen = leaf["gain"][np.argsort(np.where(leaf["order"] >= 0, leaf["order"], 99))[:7]]
    assert chosen[0] == leaf["gain"][0] and leaf["level"].max() >= level["level"].max()
    # columns: a tree only asks about the cells it was given
    some = S.grow(xb, g, h, bins, 3, 1.0, cols=np.array([1, 4]))
    assert set(some["feature"][some["left"] >= 0]) <= {1, 4}


def test_quantile_bins_and_missing_slot():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 2))
    x[:40, 1] = np.nan
    edges = S.quantile_edges(x, 8)
    assert [len(e) for e in edges] == [7, 7] and S.width(edges) == 8
    xb = S.to_bins(x, edges, 8)
    assert np.all(np.abs(np.bincount(xb[:, 0], minlength=8) - 100) <= 1)      # equal shares
    assert (xb[:40, 1] == 8).all() and xb[40:, 1].max() == 7
    assert np.all(x[40:, 1][xb[40:, 1] == 0] < edges[1][0])                  # below the first cut
    flat = S.quantile_edges(np.r_[np.zeros(60), np.linspace(0.1, 1, 40)][:, None], 8)
    assert len(flat[0]) == 3 and flat[0].min() > 0           # repeated cuts are dropped
    assert len(S.quantile_edges(np.zeros((50, 1)), 8)[0]) == 0
    assert S.cell_name(0) == "A1" and S.cell_name(6) == "B3" and S.cell_name(15) == "D4"


def test_boosting_lowers_the_loss_and_plain_boosting_is_least_squares_on_the_gradient():
    x, xb, bins, _, _, y = node(5, n=400, f=4, levels=10)
    y = ((x[:, 0] > 4) ^ (x[:, 1] > 6)) * 1.0                 # two cells must be combined
    r = S.boost(xb, y, bins, 30, rate=0.3, depth=2, lam=1.0, keep=True, watch=((xb, y),))
    assert np.all(np.diff(r["loss"][0]) < 1e-12) and r["loss"][0][-1] < 0.1
    assert np.allclose(r["loss"][0], r["loss"][1]) and r["error"][0][-1] == 0.0
    assert np.allclose(r["scores"][0][1], 0.3 * S.predict(r["trees"][0], xb, bins), atol=1e-6)
    stumps = S.boost(xb, y, bins, 30, rate=0.3, depth=1, lam=1.0)
    assert stumps["loss"][0][-1] > 0.4                        # one cell at a time cannot
    # plain: splits by least squares on -g (h = 1), leaves by one Newton step
    plain = S.boost(xb, y, bins, 3, rate=0.5, depth=2, second=False, keep=True)
    score = np.zeros(len(y))
    for tree in plain["trees"]:
        g, h = S.slopes(score, y)
        ls = S.grow(xb, g, np.ones(len(y)), bins, 2)
        assert np.array_equal(tree["feature"], ls["feature"])
        assert np.array_equal(tree["cut"], ls["cut"])
        for k in S.reached(tree)[1]:
            rows = S.members(tree, xb, bins)[k]
            assert np.isclose(tree["value"][k], -g[rows].sum() / h[rows].sum())
        score = score + 0.5 * S.predict(tree, xb, bins)
    # before any tree every h is 1/4: the two kinds of split are the same
    first = S.boost(xb, y, bins, 1, depth=3, keep=True)["trees"][0]
    assert np.array_equal(first["feature"], S.boost(xb, y, bins, 1, depth=3, second=False,
                                                    keep=True)["trees"][0]["feature"])
    # subsampling is seeded
    a = S.boost(xb, y, bins, 5, subsample=0.5, colsample=0.5, seed=3)["loss"]
    assert np.array_equal(a, S.boost(xb, y, bins, 5, subsample=0.5, colsample=0.5,
                                     seed=3)["loss"])
    assert not np.array_equal(a, S.boost(xb, y, bins, 5, subsample=0.5, colsample=0.5,
                                         seed=4)["loss"])


def test_early_stopping_rule():
    valid = [1.0, 0.8, 0.7, 0.71, 0.72, 0.69, 0.7, 0.71, 0.72, 0.73, 0.5]
    assert S.stop_round(valid, 2) == (2, 4)                   # two rounds without a new low
    assert S.stop_round(valid, 3) == (5, 8)
    assert S.stop_round(valid, 50) == (10, 10)                # never stopped: the best seen


def test_adaboost_weights_and_say():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[:, None]
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1.0])            # one event out of place
    edges = S.value_edges(x)
    bins = S.width(edges)
    xb = S.to_bins(x, edges, bins)
    sy = 2 * y - 1
    j, cut, sign, err = S.stump(xb, sy, np.full(10, 0.1), bins)
    wrong = S.stump_answer(xb, j, cut, sign) != sy
    assert sign == 1.0 and wrong.sum() == 1 and np.isclose(err, 0.1)
    for c in range(bins - 1):                                 # no stump does better
        for s in (1.0, -1.0):
            assert (S.stump_answer(xb, 0, c, s) != sy).mean() >= 0.1 - 1e-12
    r = S.adaboost(xb, y, bins, 3, (xb,), (0, 1, 2, 3))
    alpha = 0.5 * np.log(0.9 / 0.1)
    assert np.isclose(r["alpha"][0], alpha) and np.isclose(r["err"][0], 0.1)
    w0, w1 = r["weights"][0], r["weights"][1]
    assert np.allclose(w0, 0.1) and np.allclose(r["weights"].sum(1), 1.0)
    assert np.allclose(w1[wrong] / w1[~wrong][0], np.exp(2 * alpha))      # e^alpha over e^-alpha
    assert np.isclose(w1[wrong].sum(), 0.5)           # the last stump is now no better than a coin
    assert np.isclose(r["top_share"][1], w1.max()) and r["top_share"][0] == pytest.approx(0.1)
    assert np.allclose(r["votes"][0], r["votes"][1])
    assert np.allclose(r["votes"][0][1], alpha * S.stump_answer(xb, j, cut, sign), atol=1e-6)
    # the second stump is fitted to the weights: its weighted error is what it reports
    a2 = S.stump_answer(xb, r["feature"][1], r["cut"][1], r["sign"][1])
    assert np.isclose(r["err"][1], w1[a2 != sy].sum())
    assert ((r["votes"][0][3] > 0) == (y > 0.5)).all()        # three stumps get all ten right


def test_tree_layout_and_labels():
    left, right = np.array([1, 3, -1, -1, -1]), np.array([2, 4, -1, -1, -1])
    at = M.tree_layout(left, right)
    assert [at[k][1] for k in range(5)] == [0, 1, 1, 2, 2]
    assert at[3][0] < at[4][0] < at[2][0] and np.isclose(at[1][0], (at[3][0] + at[4][0]) / 2)
    assert np.isclose(at[0][0], (at[1][0] + at[2][0]) / 2)
    assert M.f2(-0.456) == "−0.46" and M.signed(0.5) == "+0.50" and M.pct(0.1234) == "12.3%"
    assert M.question("B3", 0.4423) == "B3 < 0.44?" and M.question("A4", 0.0013) == "A4 < 0.001?"
    assert M.stage(0.0, 5) == (0, 0.0) and M.stage(1.0, 5)[0] == 4 and M.commas(45748) == "45,748"


def test_nothing_is_imported_from_a_boosting_library():
    for module in (S, M):
        source = inspect.getsource(module)
        for name in ("import xgboost", "import lightgbm", "import sklearn", "from sklearn",
                     "from xgboost", "import scipy"):
            assert name not in source
    assert "numpy" in M.OWN and M.OWN == S.OWN


# ---------------------------------------------------------------------- all five
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert a.t_done >= 45.0 and a.duration <= 66.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3
    for t in np.linspace(0.0, a.duration, 90):        # no instant fails to draw
        a.scene(float(t))


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        assert any(t in M.TAGS for t in texts(a, k))
    # no physics data: said on screen in one plain line, with the physics counterpart
    assert M.STAND_IN in texts(a, 0)
    assert "stand in" in subtitles(a)[0] and "analysis" in subtitles(a)[0]


@pytest.mark.parametrize("name", XGB)
def test_xgboost_animations_say_whose_implementation_they_draw(name):
    a, m = anim(name), study(name).metadata
    assert m["own"] == S.OWN and "written out in NumPy" in m["source"]
    assert sum(M.OWN in texts(a, k) for k in range(len(a.PHASES))) >= 3
    assert "library" in S.EVIDENCE["timing"] and S.OWN in study(name).subtitle


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3
    assert m["designs_tried"] and m["limitations"] and m["evidence"]["measured"]
    assert "default_rng" in m["where_the_seed_enters"]
    assert "stand in" in m["stand_in"] and "analysis" in m["physics_counterpart"].lower()
    assert m["two_cells"] == [S.cell_name(j) for j in S.best_pair()]


@needs_mnist
def test_events_are_real_digits_read_out_as_cells_and_the_sets_are_disjoint():
    from voxanim import mnist

    xtr, ytr, _, _ = mnist.load()
    d = S.data()
    assert [len(p) for p in d["pool"]] == [S.POOL, S.POOL]
    assert (ytr[d["pool_id"][0]] == S.SIGNAL).all() and (ytr[d["pool_id"][1]] == 9).all()
    assert len(np.unique(np.concatenate(d["pool_id"]))) == 2 * S.POOL
    assert np.allclose(d["pool"][0][:3], S.cells(xtr[d["pool_id"][0][:3]], S.BLOCK))
    assert d["pool"][0].shape[1] == S.CELLS == 16
    n4, n9 = (ytr == 4).sum() - S.POOL + d["held_from_test"][0], \
        (ytr == 9).sum() - S.POOL + d["held_from_test"][1]
    assert len(d["held_y"]) == n4 + n9 and d["held_y"].sum() == n4
    for seed in (0, 1):
        tr = S.draw(seed)
        assert len(tr["y"]) == 2 * S.TRAIN and len(tr["vy"]) == 2 * S.VALID
        assert (ytr[tr["id"]] == np.where(tr["y"] > 0.5, 4, 9)).all()
        assert np.allclose(tr["x"][[0, -1]], S.cells(xtr[tr["id"][[0, -1]]], S.BLOCK))
        both = np.concatenate([tr["x"], tr["vx"]])              # training and validation
        assert len(np.unique(both.round(12), axis=0)) == len(both)
    assert not np.array_equal(S.draw(0)["id"], S.draw(1)["id"])
    assert np.array_equal(S.pictures(tr["id"][:2]).reshape(2, -1), xtr[tr["id"][:2]])
    q = S.prep(0)
    assert q["bins"] <= S.BINS and q["xb"].max() < q["bins"] and q["held"][0].shape[1] == 16


# ---------------------------------------------------------------------- 165: AdaBoost
@needs_mnist
def test_adaboost_numbers_on_screen_are_the_study():
    s, a = study(ADA), anim(ADA)
    m, subs = s.metadata, subtitles(a)
    two, f = m["two"], m["two"]["first"]
    assert "%d of the %d events wrong: err = %.3f" % (f[0]["wrong"], two["events"],
                                                    f[0]["err"]) in subs[1]
    assert "= %.2f" % f[0]["alpha"] in subs[1] and "α = %.2f" % f[1]["alpha"] in subs[2]
    assert M.pct(m["held_error"]["last"]) in subs[4] and M.pct(m["train_error"]["last"]) in subs[4]
    assert M.pct(m["weights"]["top_share_end"], 0) in subs[5]
    shown = texts(a, 1, 0.9)
    assert "err = %d × 1/%d = %s" % (f[0]["wrong"], two["events"], M.f3(f[0]["err"])) in shown
    assert "%s < %s?" % (f[0]["cell"], M.f2(f[0]["threshold"])) in shown
    assert M.pct(two["held_error"][-1]) in texts(a, 3, 0.97)
    p = s.panels[0].data
    for k, row in enumerate(f):                       # the say is half the log of the odds
        assert np.isclose(row["alpha"], 0.5 * np.log((1 - row["err"]) / row["err"]))
        assert row["wrong"] == p["wrong"][k].sum()
        assert np.isclose(row["err"], p["weights"][k][p["wrong"][k]].sum())
    assert np.isclose(f[0]["err"], f[0]["wrong"] / two["events"])
    w0, w1 = p["weights"][0], p["weights"][1]
    assert np.allclose(w0, 1 / two["events"]) and np.allclose(p["weights"].sum(1), 1.0)
    assert np.allclose(w1[p["wrong"][0]] / w1[~p["wrong"][0]][0], np.exp(2 * f[0]["alpha"]))
    # the first stump drawn on the plane is the stump: the events on its wrong side
    j = int(p["feature"][0])
    above = p["xy"][:, j] >= p["threshold"][0]
    assert np.array_equal(above == (p["sign"][0] > 0), (p["classes"] > 0.5) ^ p["wrong"][0])
    assert p["planes"].shape == (len(S.ADA_TWO_SHOWN), S.ADA_SIDE, S.ADA_SIDE)


@needs_mnist
def test_adaboost_lowers_the_error_and_piles_weight_on_a_few_events():
    s = study(ADA)
    m = s.metadata
    train, held = s.panels[1].data["per_seed"]
    assert (held[:, -1] < held[:, 1] - 0.10).all()            # every seed, by ten points
    assert (train[:, -1] < held[:, -1]).all() and train[:, -1].min() > 0.05   # never fitted
    top = s.panels[1].data["top_share"]
    assert np.allclose(top[:, 0], 0.01) and (top[:, -1] > 5 * top[:, 0]).all()
    w = m["weights"]
    assert min(w["heaviest"]) > 5.0                           # times the average weight
    assert max(w["heaviest_margin"]) < 0.0                    # the vote still gets them wrong
    # the margins: fewer wrong, but half of them stay near a tie
    counts = s.panels[2].data["counts"][0]
    centres = s.panels[2].data["centres"]
    wrong = (counts * (centres < 0)).sum(1) / counts.sum(1)
    assert wrong[-1] < wrong[0] - 0.10 and m["margin"]["median"][-1] < 0.2
    assert np.isclose(s.panels[2].data["ys"][:, -1], 1.0).all()
    assert len(s.panels[3].data["values"]) == S.ADA_HEAVY


# ---------------------------------------------------------------------- 166: settings
@needs_mnist
def test_settings_numbers_on_screen_are_the_study():
    s, a = study(GB), anim(GB)
    m, subs = s.metadata, subtitles(a)
    one, e = m["one"], m["early_stopping"]["seed0"]
    assert "from %.3f to %.3f" % tuple(one["train_loss"]) in subs[2]
    assert "%.3f" % m["rates"]["rate 0.1"]["lowest"]["mean"] in subs[3]
    assert "here %d" % e["chosen"] in subs[5] and "%.3f" % e["held_end"] in subs[5]
    assert "%.3f" % m["depths"]["depth 1"]["lowest"]["mean"] in subs[4]
    p = s.panels[0].data
    # the round that is drawn: every score moves by the rate times its leaf's value
    assert np.allclose(p["after"], p["xy"][:, 0] + one["rate"] * p["tree_value"][p["leaf"]],
                       atol=1e-5)
    leaves = np.flatnonzero(p["tree_left"] < 0)
    assert np.allclose(p["tree_value"][leaves], -p["tree_G"][leaves] / p["tree_H"][leaves])
    assert np.allclose(p["mean_pull"], -p["tree_G"] / p["tree_N"])
    assert one["train_loss"][1] < one["train_loss"][0]
    shown = texts(a, 1, 0.95)
    for k in leaves:
        assert M.signed(p["tree_value"][k]) in shown and "%d" % p["tree_N"][k] in shown
    assert "held-out loss with those: %s" % M.f3(e["held_there"]) in texts(a, 5, 0.95)


@needs_mnist
def test_small_rates_need_more_trees_and_end_lower_and_early_stopping_finds_the_turn():
    s = study(GB)
    m = s.metadata
    low, at = s.panels[1].data["lowest"], s.panels[1].data["lowest_at"]      # rate, seed
    assert list(m["rates"]) == ["rate 1", "rate 0.3", "rate 0.1", "rate 0.03"]
    assert (low[2] < low[0] - 0.03).all() and (low[1] < low[0] - 0.03).all()
    assert low[2].mean() < low[1].mean() and abs(low[3].mean() - low[2].mean()) < 0.005
    assert (at[0] < at[1]).all() and (at[1] < at[2]).all() and (at[2] < at[3]).all()
    d = s.panels[3].data
    assert (d["at_stop"] <= d["lowest"] + 0.012).all()        # close to the best there was
    assert (d["end"][:3] > d["at_stop"][:3] + 0.1).all()      # and far below training on
    assert (d["stopped_at"] >= d["stop"]).all()
    assert (d["stopped_at"] - d["stop"] <= S.GB_PATIENCE).all()
    curve = d["ys"]                                           # seed 0, rate 0.1
    assert curve[0, -1] < 0.05 < curve[2, -1] and np.argmin(curve[1]) == m[
        "early_stopping"]["seed0"]["chosen"]
    # depth: one cell at a time is clearly worse; half the rows is no worse than all
    dep, sub = m["depths"], m["subsamples"]
    assert dep["depth 1"]["lowest"]["min"] > dep["depth 3"]["lowest"]["max"] + 0.04
    assert sub["50% of the rows"]["lowest"]["mean"] < sub["100% of the rows"]["lowest"][
        "mean"] + 0.002
    assert sub["10% of the rows"]["lowest"]["mean"] > sub["100% of the rows"]["lowest"][
        "mean"] + 0.02


# ---------------------------------------------------------------------- 167: the gain
@needs_mnist
def test_gain_numbers_on_screen_are_the_study():
    s, a = study(GAIN), anim(GAIN)
    m, subs = s.metadata, subtitles(a)
    e, hd, nd = m["one_event"], m["handful"], m["node"]
    assert "g = %s and h = %.2f" % (M.f2(e["g"]), e["h"]) in subs[0]
    assert "= %s" % M.f2(hd["value"]) in subs[1] and M.f3(hd["score"]) in subs[1]
    assert "gain %.2f" % max(hd["gain"]) in subs[2]
    assert np.isclose(e["g"], e["p"] - 1.0) and np.isclose(e["h"], e["p"] * (1 - e["p"]))
    p = s.panels[1].data
    g, h, y = p["xy"][:, 1], p["h"], p["classes"]
    prob = S.sigmoid(p["score"])
    assert np.allclose(g, prob - y) and np.allclose(h, prob * (1 - prob))
    assert np.all(np.diff(p["xy"][:, 0]) >= 0)                # in the order of the cell
    # the handful: sums, the leaf, and every cut's gain against brute force
    assert np.isclose(hd["G"], g.sum()) and np.isclose(hd["H"], h.sum())
    assert np.isclose(hd["value"], -g.sum() / (h.sum() + hd["lam"]))
    assert np.isclose(hd["score"], brute_leaf(g.sum(), h.sum(), hd["lam"]), atol=1e-6)
    for k in range(len(g) - 1):
        brute = brute_leaf(g.sum(), h.sum(), hd["lam"]) \
            - brute_leaf(g[:k + 1].sum(), h[:k + 1].sum(), hd["lam"]) \
            - brute_leaf(g[k + 1:].sum(), h[k + 1:].sum(), hd["lam"])
        assert np.isclose(p["gain"][k], brute, atol=1e-5)
    # the parabola is close to the true change of the loss near its bottom
    near = np.abs(p["w"] - hd["value"]) < 0.3
    assert np.abs(p["exact"] - p["approx"])[near].max() < 0.01
    assert abs(hd["exact_minimum_at"] - hd["value"]) <= 0.05
    shown = texts(a, 2, 0.97)
    assert "gain = %s" % M.f2(max(hd["gain"])) in shown
    assert "GL = %s   HL = %s" % (M.f2(hd["GL"]), M.f2(hd["HL"])) in shown
    assert "= %s / %s = %s" % (M.f2(-hd["G"]), M.f2(hd["H"] + hd["lam"]),
                               M.f2(hd["value"])) in texts(a, 1, 0.97)
    # the small node: lambda shrinks both leaf values and the gain
    assert np.all(np.diff(np.abs(nd["left_value"])) < 0)
    assert np.all(np.diff(np.abs(nd["right_value"])) < 0) and np.all(np.diff(nd["best_gain"]) < 0)
    assert nd["best_gain"][1] < nd["gamma"] and 0 < nd["allowed_by_mcw"] < nd["thresholds"]
    assert "%d of %d thresholds" % (nd["thresholds"] - nd["allowed_by_mcw"],
                                    nd["thresholds"]) in subs[3]


@needs_mnist
def test_second_order_changes_splits_but_not_the_held_out_loss_here():
    s = study(GAIN)
    m = s.metadata
    di, cp, sw = m["disagree"], m["compare"], m["sweeps"]
    # the node where they disagree: each criterion prefers its own split
    assert di["second_scores"][0] > di["second_scores"][1]
    assert di["plain_scores"][1] > di["plain_scores"][0]
    assert di["share_by_round"]["first"] == 0.0               # all h equal: no difference
    assert di["share_by_round"]["rounds_31_to_100"] > 0.3
    share = s.panels[3].data["share"]
    assert share[:2].max() == 0.0 and share[30:].mean() > share[2:10].mean()
    # held-out loss: no difference beyond the spread of the seeds
    names = cp["names"]
    diff = cp["plain_minus_second"]
    assert abs(diff["mean"]) < 0.005 and diff["min"] < 0 < diff["max"]
    assert abs(cp["held"][names[0]]["mean"] - cp["held"][names[2]]["mean"]) < 0.01
    # gamma: smaller trees, and no better at any value tried
    leaves, held = sw["gamma"]["leaves"], sw["gamma"]["held"]
    assert np.all(np.diff(leaves) < 0) and leaves[-1] < leaves[0] / 4
    assert min(held[1:]) > held[0] - 0.002 and held[-1] > held[0] + 0.02
    # lambda and min_child_weight: mild; fewer leaves with min_child_weight
    assert max(sw["lam"]["held"]) - min(sw["lam"]["held"]) < 0.03
    assert np.all(np.diff(sw["mcw"]["leaves"]) < 0)
    assert sw["gamma"]["held_end"][2] < sw["gamma"]["held_end"][0]      # it does curb the end


# ---------------------------------------------------------------------- 168: histograms
@needs_mnist
def test_histogram_numbers_on_screen_are_the_study():
    s, a = study(HIST), anim(HIST)
    m, subs = s.metadata, subtitles(a)
    ti, mi, gr = m["timing"], m["missing"], m["growth"]
    assert M.commas(ti["scored_per_tree"]["exact"]) in subs[0]
    assert M.commas(ti["scored_per_tree"]["32 bins"]) in subs[1]
    assert M.commas(ti["added_per_tree"]["32 bins"]) in subs[2]
    assert "gain %.2f against %.2f" % (mi["drawn"]["gain_left"],
                                       mi["drawn"]["gain_right"]) in subs[3]
    assert "%.3f" % mi["held"]["random"]["learnt"]["mean"] in subs[4]
    assert "%.3f and %.3f" % (gr["level"]["mean"], gr["leaf"]["mean"]) in subs[5]
    assert "vary" in subs[6] and M.TIMED in texts(a, 6)
    p = s.panels[0].data
    # the events dropped into the bins add up to the histograms drawn
    assert np.allclose(np.bincount(p["drop_bin"], weights=p["drop_g"], minlength=8), p["values"])
    assert np.allclose(np.bincount(p["drop_bin"], weights=p["drop_h"], minlength=8), p["H"])
    assert np.array_equal(np.searchsorted(p["edges"], p["drop_x"], side="right"), p["drop_bin"])
    G, H = p["values"], p["H"]
    for k in range(7):                                        # the scan, cut by cut
        assert np.isclose(p["gain"][k], S.split_gain(G[:k + 1].sum(), H[:k + 1].sum(),
                                                     G[k + 1:].sum(), H[k + 1:].sum(), 1.0))
    assert "gain %s" % M.f2(p["gain"].max()) in texts(a, 1, 0.97)
    # the subtraction, in every bin, for G, H and the counts
    assert np.allclose(p["sub_parent"] - p["sub_left"], p["sub_right"])
    assert p["sub_parent"][2].sum() == p["sub_n"][0] == 2 * S.TRAIN
    assert p["sub_left"][2].sum() == p["sub_n"][1] and m["subtraction"]["holds_exactly"]
    # the split with holes: the better side is kept
    d = mi["drawn"]
    assert d["sent_left"] == (d["gain_left"] > d["gain_right"])
    hist, gain = s.panels[1].data["miss_hist"], s.panels[1].data["miss_gain"]
    assert hist[2, -1] == d["missing"] and np.isclose(gain.max(), max(d["gain_left"],
                                                                      d["gain_right"]))
    assert 0.25 < d["missing"] / (2 * S.TRAIN) < 0.35         # the rule: 30% at random
    for which, t in (("level", gr["level_depth"]), ("leaf", gr["leaf_depth"])):
        tree = {k: s.panels[3].data["%s_%s" % (which, k)] for k in ("left", "right", "level")}
        assert (tree["left"] < 0).sum() == gr["leaves"] and tree["level"].max() == t


@needs_mnist
def test_histograms_are_as_good_as_exact_and_a_learnt_side_helps_with_random_holes():
    s = study(HIST)
    m = s.metadata
    b, ti, held = m["by_bins"], m["timing"], m["missing"]["held"]
    assert abs(b["32 bins"]["mean"] - b["exact"]["mean"]) < 0.01
    assert b["2 bins"]["min"] > b["32 bins"]["max"] + 0.1
    means = [b["%d bins" % k]["mean"] for k in S.HIST_BINS]
    assert means[0] > means[1] > means[2] > means[3] > means[4]
    # counts are reproducible; times are this machine's: only that they are times
    scored = ti["scored_per_tree"]
    assert scored["32 bins"] < scored["256 bins"] < scored["exact"]
    assert ti["added_per_tree"]["32 bins"] < 0.6 * ti["added_without_subtraction"]
    assert ti["sorted_per_tree"] == ti["added_without_subtraction"] > 0
    t = s.panels[2].data["timing"]
    assert t.shape == (len(S.HIST_BINS) + 1, 5) and (t[:, 0] > 0).all() and "machine" in ti["note"]
    # missing readings
    rnd, floor = held["random"], held["floor"]
    assert rnd["learnt"]["mean"] < min(rnd["left"]["mean"], rnd["right"]["mean"],
                                       rnd["mean"]["mean"]) - 0.01
    assert rnd["learnt"]["max"] < rnd["left"]["mean"] and rnd["learnt"]["max"] < rnd[
        "right"]["mean"]
    assert floor["mean"]["min"] > floor["learnt"]["max"] + 0.03       # filling in is worst
    assert abs(floor["learnt"]["mean"] - floor["left"]["mean"]) < 0.01
    assert m["missing"]["sent_left"]["floor"] > m["missing"]["sent_left"]["random"]
    assert m["missing"]["complete"]["mean"] < floor["left"]["mean"] - 0.05
    # level-wise against leaf-wise: no difference beyond the seeds
    gap = m["growth"]["level_minus_leaf"]
    assert abs(gap["mean"]) < 0.005 and gap["min"] < 0 < gap["max"]


# ---------------------------------------------------------------------- 169: one round
@needs_mnist
def test_one_round_numbers_on_screen_are_the_study():
    s, a = study(ROUND), anim(ROUND)
    m, subs = s.metadata, subtitles(a)
    r, rv = m["round"], m["rivals"]
    assert "%s < %.3f, with gain %.0f" % (r["root_cell"], r["root_threshold"],
                                         r["root_gain"]) in subs[1]
    assert "leaving %d leaves" % r["leaves_after"] in subs[2]
    assert "from %.3f to %.3f" % tuple(r["train_loss"][:2]) in subs[3]
    for name in S.RIVALS:
        assert M.pct(rv[name]["error"]["mean"]) in subs[6]
    p = s.panels[0].data
    t0, t1, t2 = p["table"]
    # the table: p from F, g and h from p; the step; and round two from where it ended
    for t in (t0, t1, t2):
        assert np.allclose(t[1], S.sigmoid(t[0]))
        assert np.allclose(t[2], t[1] - (np.array(p["labels"]) == "4"))
        assert np.allclose(t[3], t[1] * (1 - t[1]))
    assert np.allclose(t0[0], 0.0) and np.allclose(np.abs(t0[2]), 0.5)
    assert np.allclose(t1[0], t0[0] + r["rate"] * t0[4])
    assert np.allclose(t2[0], t1[0] + r["rate"] * t1[4])
    # the tree: leaf weights from the sums, the pruned split the weakest, below gamma
    G, H, value = p["tree_G"], p["tree_H"], p["tree_value"]
    assert np.allclose(value[1:], -G[1:] / (H[1:] + r["lam"])) and abs(value[0]) < 1e-9
    inner = np.flatnonzero(p["tree_left"] >= 0)
    assert len(r["pruned_nodes"]) == 1 and r["leaves_after"] == r["leaves_before"] - 1
    k = r["pruned_nodes"][0]
    assert p["tree_gain"][k] == p["tree_gain"][inner].min() < r["gamma"]
    assert np.sort(p["tree_gain"][inner])[1] > r["gamma"] and not p["kept"][k]
    assert set(t0[5].astype(int)) <= set(np.flatnonzero(p["tree_left"] < 0)) | {k}
    assert np.allclose(t0[4], value[t0[5].astype(int)])
    # the root: the best cut of the best cell, from the histograms that are drawn
    assert np.isclose(p["best_by_feature"].max(), r["root_gain"])
    assert np.isclose(p["root_gain"].max(), r["root_gain"])
    hist = p["root_hist"]
    assert hist[2].sum() == 2 * S.TRAIN and np.isclose(hist[0].sum(), 0.0)
    assert r["train_loss"][0] > r["train_loss"][1] > r["train_loss"][2]
    assert np.isclose(r["train_loss"][0], np.log(2))
    shown = texts(a, 3, 0.95)
    assert M.signed(t1[0][0]) in shown and "F + %g × w" % r["rate"] in shown
    assert "%s to %s" % (M.pct(rv["one tree"]["error"]["min"]),
                         M.pct(rv["one tree"]["error"]["max"])) in texts(a, 6, 0.95)


@needs_mnist
def test_the_knobs_and_the_five_methods():
    s = study(ROUND)
    m = s.metadata
    lo, kn, rv = m["loose"], m["knobs"], m["rivals"]
    assert set(kn) == set(S.KNOB_NAMES.values()) and len(kn) == 7
    for name in ("learning rate", "max depth", "lambda", "gamma", "min child weight"):
        assert kn[name]["end"]["max"] < lo["end"]["min"] - 0.2        # each restrains it alone
    assert kn["subsample"]["end"]["mean"] > lo["end"]["mean"] - 0.05   # this one does not
    assert kn["colsample"]["end"]["mean"] < lo["end"]["mean"] - 0.05
    assert kn["gamma"]["leaves"] < lo["leaves"] / 5 and kn["max depth"]["leaves"] <= 4
    assert (s.panels[2].data["stopped"] <= s.panels[2].data["values"] + 0.002).all()
    # many rounds: subsampling changes little
    mn = m["many"]
    assert abs(mn["held_lowest"]["all"] - mn["held_lowest"]["sampled"]) < 0.01
    assert mn["train_end"]["all"] < mn["held_end"]["all"] - 0.1
    # the five: one tree worst, boosting best, and the last gap small but in every seed
    err = {n: rv[n]["error"] for n in S.RIVALS}
    assert err["one tree"]["min"] > err["AdaBoost"]["max"] + 0.02
    assert err["AdaBoost"]["min"] > err["forest"]["max"] + 0.01
    assert err["forest"]["min"] > err["gradient boosting"]["max"] + 0.005
    gap = m["boosting_gap"]["gradient_minus_xgboost_error"]
    assert gap["min"] > 0 and gap["max"] < 0.015
    auc = s.panels[3].data["auc_per_seed"]                    # method, seed
    assert (auc[4] > auc[3]).all() and (auc[3] > auc[1]).all() and (auc[1] > auc[2]).all()
    assert (auc[2] > auc[0]).all() and auc.min() > 0.8
    assert len(m["rival_settings"]) == 5 and all(len(v["picked"]) == len(m["seeds"])
                                                 for v in rv.values())
