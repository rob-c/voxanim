"""Animations 160-164 (voxanim.mech_trees) and their experiments (voxanim.study_trees):
the layout is clean, the loop closes, every phase says what kind of evidence it shows
and the first says that digits stand in for events, the numbers on screen are the
study's, each study's claim holds as a relative claim with a margin over seeds, and the
trees do what their arithmetic says."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_trees as M
from voxanim import study_trees as S
from voxanim.mech_phys_apps import STAND_IN, pct
from voxanim.next_catalog import build_study

SPLITS, PRUNING, UNSTABLE, BAGGING, FOREST = (
    "tree-splits-and-impurity", "tree-pruning-and-depth", "tree-instability", "bagging",
    "random-forest")
EACH = [pytest.param(name, marks=needs_mnist)
        for name in (SPLITS, PRUNING, UNSTABLE, BAGGING, FOREST)]


def study(name):
    return build_study("study_trees", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts]


def step_data(n=200, seed=0):
    """Two features; the label is 1 where the first is above 0.6, with no noise."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, 2))
    return x, (x[:, 0] > 0.6).astype(int)


# ---------------------------------------------------------------------- all five
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_every_frame_of_a_coarse_pass_builds(name):
    a = anim(name)                                    # a frame between the sampled stills
    for t in np.arange(0.0, a.duration, 0.37):        # once crashed an earlier animation
        assert a.scene(float(t)).texts is not None


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
    assert STAND_IN in texts(a, 0)                    # no physics data: said in one line
    assert "stand in for events" in subtitles(a)[0] and "analysis" in subtitles(a)[0]


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name).validate()
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3
    assert m["designs_tried"] and m["limitations"] and m["evidence"]["measured"]
    assert "default_rng" in m["where_the_seed_enters"]
    assert "stand in" in m["stand_in"] and "analysis" in m["physics_counterpart"]
    assert "pixels" in m["features"]["two"] and "49 cells" in m["features"]["cells"]


# ---------------------------------------------------------------------- no data needed
def test_impurities_and_the_gain_of_a_split():
    p = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(S.impurity(p), [0.0, 0.375, 0.5, 0.0])                 # 1 - p^2 - q^2
    assert np.allclose(S.impurity(p), 1 - p ** 2 - (1 - p) ** 2)
    assert np.allclose(S.impurity(p, "entropy"), [0.0, 0.8112781, 1.0, 0.0])  # bits
    assert np.allclose(S.impurity(p, "error"), [0.0, 0.25, 0.5, 0.0])
    # the textbook case: 40 + 40 into (30, 10) and (10, 30), or into (20, 40) and (20, 0)
    first = [float(S.split_gain(80, 40, 40, 30, c)) for c in S.CRITERIA]
    second = [float(S.split_gain(80, 40, 60, 20, c)) for c in S.CRITERIA]
    assert np.isclose(first[2], 0.25) and np.isclose(second[2], 0.25)         # error: a tie
    assert second[0] > first[0] + 0.03 and second[1] > first[1] + 0.1         # not for these
    assert float(S.split_gain(10, 5, 0, 0)) == 0.0                            # an empty side
    w = S.worked(12, 6, 5, 5)
    assert w["counts"] == [(6, 6), (5, 0), (1, 6)]
    assert np.isclose(w["gini"]["right"], 1 - (1 / 7) ** 2 - (6 / 7) ** 2)
    assert np.isclose(w["gini"]["gain"], 0.5 - 7 / 12 * w["gini"]["right"])
    assert np.isclose(w["entropy"]["gain"], float(S.split_gain(12, 6, 5, 5, "entropy")))


def test_thresholds_are_midpoints_and_bins_follow_them():
    x = np.array([[0.1, 5.0], [0.3, 5.0], [0.3, 7.0], [0.9, 6.0]])
    cuts = S.midpoints(x)
    assert np.allclose(cuts[0], [0.2, 0.6]) and np.allclose(cuts[1], [5.5, 6.5])
    xb = S.binned(x, cuts)
    assert xb[:, 0].tolist() == [0, 1, 1, 2] and xb[:, 1].tolist() == [0, 0, 2, 1]
    assert np.array_equal(S.binned(x, cuts) <= 0, x <= cuts[:, 0])            # left: value <= cut
    one = S.midpoints(np.array([[1.0, 2.0], [1.0, 3.0]]))                     # a constant column
    assert one[0, 0] == S.PAD and one[1, 0] == 2.5
    assert S.midpoints(np.linspace(0, 1, 50)[:, None], most=7).shape == (1, 7)


def test_a_tree_finds_a_step_and_grows_to_purity():
    x, y = step_data()
    tree, cuts = S.fit_two(x, y)
    assert tree["feature"][0] == 0 and len(tree["feature"]) == 3              # one split does it
    v = cuts[0, tree["cut"][0]]
    assert x[x[:, 0] <= 0.6, 0].max() <= v <= x[x[:, 0] > 0.6, 0].min()
    assert np.isclose(tree["gain"][0], float(S.impurity(y.mean())))           # all of it removed
    xb = S.binned(x, cuts)
    assert S.error(tree, xb, y) == 0.0 and S.leaf_count(tree) == 2
    assert np.array_equal(S.vote(tree, xb), y)
    # noise on the labels: it still grows until no training event is wrong
    rng = np.random.default_rng(1)
    noisy = np.where(rng.random(len(y)) < 0.2, 1 - y, y)
    deep, c2 = S.fit_two(x, noisy)
    xb2 = S.binned(x, c2)
    assert S.error(deep, xb2, noisy) == 0.0 and S.leaf_count(deep) > 20
    assert np.all(deep["left"][deep["feature"] >= 0] > np.flatnonzero(deep["feature"] >= 0))
    # counts add up at every node, and cutting at a depth is the same tree grown to it
    for t in np.flatnonzero(deep["feature"] >= 0):
        a, b = deep["left"][t], deep["right"][t]
        assert deep["n"][t] == deep["n"][a] + deep["n"][b]
        assert deep["n1"][t] == deep["n1"][a] + deep["n1"][b]
    short = S.grow(xb2, noisy, c2.shape[1] + 1, max_depth=3)
    assert np.array_equal(S.vote(short, xb2), S.vote(deep, xb2, max_depth=3))
    assert S.leaf_count(short) == S.leaf_count(deep, max_depth=3) <= 8
    # the controls hold: leaf size, and a weighted event counts as that many
    small = S.grow(xb2, noisy, c2.shape[1] + 1, min_leaf=15)
    assert small["n"][small["feature"] < 0].min() >= 15
    w = np.random.default_rng(2).integers(0, 3, len(y))
    twice = np.repeat(np.arange(len(y)), w)
    a = S.grow(xb2, noisy, c2.shape[1] + 1, w=w)
    b = S.grow(xb2[twice], noisy[twice], c2.shape[1] + 1)
    assert all(np.array_equal(a[k], b[k]) for k in ("feature", "cut", "n", "n1"))
    # the plane is divided into as many rectangles as leaves, covering it once
    r = S.regions(deep, c2, ((0.0, 1.0), (0.0, 1.0)))
    assert len(r) == S.leaf_count(deep)
    assert np.isclose(((r[:, 2] - r[:, 1]) * (r[:, 4] - r[:, 3])).sum(), 1.0)


def test_pruning_cuts_the_weakest_link_first():
    x, y = step_data(300, 3)
    rng = np.random.default_rng(4)
    noisy = np.where(rng.random(len(y)) < 0.15, 1 - y, y)
    tree, cuts = S.fit_two(x, noisy)
    xb = S.binned(x, cuts)
    alphas, internals, links = S.pruning(tree)
    assert alphas[0] == 0.0 and np.all(np.diff(alphas) > 0)
    leaves = [S.leaf_count(tree, internal=i) for i in internals]
    errors = [S.error(tree, xb, noisy, internal=i) for i in internals]
    assert leaves[-1] == 1 and np.all(np.diff(leaves) < 0) and np.all(np.diff(errors) > 0)
    assert errors[0] == 0.0 and links[0] == -1
    # each tree is the cheapest of the sequence at its own alpha, and at the next
    cost = np.array(errors)[None] + alphas[:, None] * np.array(leaves)[None]
    for k in range(len(alphas)):
        assert cost[k, k] <= cost[k].min() + 1e-12
        if k:
            assert np.isclose(cost[k, k], cost[k, k - 1])                      # where they tie
    # the first cut: its g is the smallest of all, worked out from the counts
    wrong, below, n_leaves = S.link_arithmetic(tree, internals[0], links[1])
    assert np.isclose((wrong - below) / len(y) / (n_leaves - 1), alphas[1])
    for t in np.flatnonzero(internals[0]):
        w, b, n = S.link_arithmetic(tree, internals[0], t)
        assert (w - b) / len(y) / (n - 1) >= alphas[1] - 1e-12
    assert S.pruned_at(alphas, alphas[2]) == 2 and S.pruned_at(alphas, alphas[2] - 1e-9) == 1
    assert S.pruned_at(alphas, 1.0) == len(alphas) - 1


def test_a_random_handful_is_drawn_at_every_node_and_repeats():
    x, y = step_data(300, 5)
    x = np.column_stack([x, np.zeros(len(x)), np.random.default_rng(6).random((len(x), 3))])
    cuts = S.midpoints(x)
    xb, bins = S.binned(x, cuts), cuts.shape[1] + 1
    a = S.grow(xb, y, bins, mtry=2, rng=np.random.default_rng(7))
    b = S.grow(xb, y, bins, mtry=2, rng=np.random.default_rng(7))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    asks = np.flatnonzero(a["feature"] >= 0)
    for t in asks:
        drawn = a["tried"][t][a["tried"][t] >= 0]
        assert len(drawn) >= 2 and len(set(drawn)) == len(drawn)
        assert a["feature"][t] in drawn and a["feature"][t] != 2               # never the blank
    assert S.error(a, xb, y) == 0.0
    # with every feature tried it is the plain tree
    full = S.grow(xb, y, bins, mtry=6, rng=np.random.default_rng(7))
    plain = S.grow(xb, y, bins)
    assert np.array_equal(full["feature"], plain["feature"])
    w = S.bootstrap(np.random.default_rng(8), 1000)
    assert w.sum() == 1000 and abs((w > 0).mean() - (1 - 1 / np.e)) < 0.04


def test_disagreement_and_the_variance_formula():
    votes = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]])
    pairs = [(0, 1), (0, 2), (1, 2)]
    assert np.isclose(S.disagreement(votes),
                      np.mean([np.mean(votes[a] != votes[b]) for a, b in pairs]))
    assert S.disagreement(np.ones((5, 9), int)) == 0.0
    # answers that share a part (the sample) and have a part of their own (the resample)
    rng = np.random.default_rng(0)
    samples, events, trees = 400, 30, S.BAG_STEPS[-1]
    lean = rng.random((samples, 1, events))                      # the same for a bag's trees
    answers = (rng.random((samples, trees, events)) < 0.2 + 0.6 * lean).astype(float)
    steps = np.array(S.BAG_STEPS)
    shares = np.cumsum(answers, 1)[:, steps - 1] / steps[None, :, None]
    v = S.vote_variance(shares)
    assert np.allclose(v["measured"], v["formula"], rtol=0.12)
    assert np.isclose(v["sigma2"], 0.25, atol=0.01)              # mean answer 0.5: p (1 - p)
    assert np.isclose(v["cov"], 0.36 / 12, atol=0.004)           # variance of 0.6 * uniform
    assert np.isclose(v["formula"][-1], v["cov"] + (v["sigma2"] - v["cov"]) / trees)
    assert v["formula"][-1] > v["cov"] > 0                       # the floor is not reached


def test_labels_and_layout_of_a_drawn_tree():
    assert M.question(1, 4.7644) == "spread ≤ 4.76?" and M.question(0, 0.25) == "top ≤ 0.25?"
    assert M.fours(1, 6, "no: ") == "no: 1 four, 6 nines"
    assert M.gini_line("no:", 1, 6, 0.2449) == "no: 1 − (1/7)² − (6/7)² = 0.245"
    assert M.exact(2000) == "exact arithmetic on 2,000 events" and M.is_evidence(M.MEASURED)
    assert M.is_evidence(M.TRAINING) and not M.is_evidence("measured elsewhere")
    assert M.cell_name(10) == "row 2, column 4"
    x, y = step_data(60, 9)
    y = np.where(np.random.default_rng(1).random(60) < 0.3, 1 - y, y)
    tree, _ = S.fit_two(x, y)
    pos = M.tree_layout(tree, ((0.1, 0.9), (0.3, 0.6)))
    assert set(pos) == set(range(len(tree["feature"]))) and pos[0][1] == 0.3
    for t in np.flatnonzero(tree["feature"] >= 0):                # between its children
        a, b = pos[int(tree["left"][t])], pos[int(tree["right"][t])]
        assert a[0] < pos[int(t)][0] < b[0] and a[1] > pos[int(t)][1]
    assert M.parents(tree)[0] == -1 and M.parents(tree)[tree["left"][0]] == 0


# ---------------------------------------------------------------------- the events
@needs_mnist
def test_events_are_real_two_classes_and_disjoint():
    from voxanim import mnist

    e = S.events()
    xtr, ytr, xte, yte = mnist.load()
    assert len(set(e["pool_index"]) & set(e["held_index"])) == 0
    assert len(e["pool_y"]) == 2 * S.POOL and e["pool_y"].sum() == S.POOL
    assert e["pool_index"].max() < len(xtr)                       # training file only
    assert np.all(ytr[e["pool_index"]] == np.where(e["pool_y"] == 1, S.SIGNAL, S.BACKGROUND))
    test = e["held_index"] >= len(xtr)
    assert test.sum() == np.isin(yte, (S.SIGNAL, S.BACKGROUND)).sum()
    assert np.all(yte[e["held_index"][test] - len(xtr)]
                  == np.where(e["held_y"][test] == 1, S.SIGNAL, S.BACKGROUND))
    img = S.picture(e["held_index"][[0, -1]])
    assert np.allclose(img[1].ravel(), xte[e["held_index"][-1] - len(xtr)])
    assert np.allclose(e["held_two"][0], S.two_features(img[:1])[0])
    assert np.allclose(e["held_cells"][0], img[0].reshape(7, 4, 7, 4).mean((1, 3)).ravel())
    assert np.isclose(e["held_two"][0, 0], img[0][5:10, 10:18].mean())
    bar = np.zeros((1, 28, 28))
    bar[0, 4:24, 13:15] = 1.0                                      # rows 4..23, evenly inked
    assert np.isclose(S.spread(bar)[0], np.std(np.arange(4, 24)))
    rows, extra = S.draw(50, 3, 20)
    assert len(set(rows) | set(extra)) == 140 and e["pool_y"][rows].sum() == 50
    # both features say something, neither says everything
    for f in (0, 1):
        t, c = S.fit_two(e["pool_two"][:, [f, f]], e["pool_y"], max_depth=1)
        err = S.error(t, S.binned(e["held_two"][:, [f, f]], c), e["held_y"])
        assert 0.10 < err < 0.25


# ---------------------------------------------------------------------- 160: one split
@needs_mnist
def test_splits_numbers_on_screen_are_the_study():
    s, a = study(SPLITS), anim(SPLITS)
    m, subs = s.metadata, subtitles(a)
    h, n, b = m["handful"], m["node"], m["blind"]
    assert "from %.3f to %.3f, a gain of %.3f" % (h["gini"]["node"], h["gini"]["after"],
                                                  h["gini"]["gain"]) in subs[1]
    drawn = texts(a, 1, 0.95)
    (a1, a0), (b1, b0), (c1, c0) = h["counts"]
    assert M.fours(a1, a0, "the node: ") in drawn and M.fours(b1, b0, "yes: ") in drawn
    assert M.fours(c1, c0, "no: ") in drawn and M.question(h["feature"], h["threshold"]) in drawn
    assert M.gini_line("no:", c1, c0, h["gini"]["right"]) in drawn
    assert "gain = %.3f − %.3f = %.3f" % (h["gini"]["node"], h["gini"]["after"],
                                          h["gini"]["gain"]) in drawn
    assert any("gain %.3f" % h["entropy"]["gain"] in t for t in drawn)
    assert "gain %.3f" % h["gini"]["gain"] in subs[2]
    assert "Gini gain %.3f" % n["gini"]["gain"] in subs[3] and "%.3f" % n["best_other"] in subs[3]
    assert any("Gini gain %.3f" % n["gini"]["gain"] in t for t in texts(a, 3, 0.9))
    assert "(%d wrong before, %d + %d after)" % tuple(b["wrong"]) in subs[4]
    last = texts(a, 4, 0.95)
    for v in m["criteria"].values():
        assert "%s, %d leaves" % (pct(v["held"]["mean"], 1), round(v["leaves"]["mean"])) in last
    assert M.fours(*b["counts"][0], "further down the tree: ") in last


@needs_mnist
def test_the_scan_is_the_split_a_tree_takes():
    s = study(SPLITS)
    m, p0, p1 = s.metadata, s.panels[0].data, s.panels[1].data
    x, y = p0["xy"], p0["classes"]
    e = S.events()
    rows, seed = S.handful()
    assert seed == m["handful"]["seed"] and np.allclose(x, e["pool_two"][rows])
    assert y.sum() == S.HANDFUL == len(y) - y.sum()
    # every gain on screen, from the events themselves
    for f in range(2):
        for c, v in enumerate(p0["cuts"][f]):
            left = x[:, f] <= v
            assert left.sum() == p0["left"][f, c] and y[left].sum() == p0["left1"][f, c]
            for k, name in enumerate(S.CRITERIA):
                assert np.isclose(p0["gains"][k, f, c],
                                  S.split_gain(len(y), y.sum(), left.sum(), y[left].sum(), name))
    h = m["handful"]
    assert 0 < h["gini"]["right"] < 0.5 or 0 < h["gini"]["left"] < 0.5        # a mixed side
    assert np.isclose(h["gini"]["gain"], p0["gains"][0].max())
    tree, cuts = S.fit_two(x, y, max_depth=1)                                 # what a tree does
    assert tree["feature"][0] == h["feature"]
    assert np.isclose(cuts[h["feature"], tree["cut"][0]], h["threshold"])
    # the large node: the curve's peak is the root a tree grown on those thresholds takes
    xs, ys, _ = S.sample("two", S.CELLS_N, 0)
    root = S.grow(S.binned(xs, p1["grid"]), ys, S.NODE_GRID + 1, max_depth=1)
    n = m["node"]
    assert root["feature"][0] == n["feature"]
    assert np.isclose(p1["grid"][n["feature"], root["cut"][0]], n["threshold"])
    assert np.isclose(root["gain"][0], n["gini"]["gain"]) and n["gini"]["gain"] > n["best_other"]
    assert n["counts"][1][0] + n["counts"][2][0] == S.CELLS_N
    assert m["roots"]["same_in_every_seed"]


@needs_mnist
def test_the_error_sees_nothing_where_gini_does_and_its_tree_stops_early():
    s = study(SPLITS)
    m, p2 = s.metadata, s.panels[2].data
    b = m["blind"]
    (a1, a0), (b1, b0), (c1, c0) = b["counts"]
    assert a1 + a0 >= S.BLIND_MIN and a1 == b1 + c1 and a0 == b0 + c0
    assert b["wrong"][0] == b["wrong"][1] + b["wrong"][2] == min(a1, a0)      # no gain at all
    assert abs(b["error_best"]) < 1e-9 and b["gini"]["gain"] > 0.01
    assert b["entropy"]["gain"] > 0.01
    per = p2["per_seed"]                                                      # seed, criterion
    gini, entropy, error = per.T
    assert np.all(error > gini + 0.015) and np.all(error > entropy + 0.015)   # in every seed
    assert abs(gini.mean() - entropy.mean()) < 0.01                           # those two: alike
    assert np.all(p2["train"][:, :2] == 0.0) and np.all(p2["train"][:, 2] > 0.05)
    assert np.all(p2["leaves"][:, 2] < 0.6 * p2["leaves"][:, 0])
    same = p2["same"]
    assert np.all(same[:, 0] > 0.5) and np.all(same[:, 2] > 0.1)
    run, share_, asked = S.criterion_run(1)
    assert np.allclose(run[:, 0], per[1]) and np.allclose(share_, same[1])
    assert asked == p2["asked"][1]


# ---------------------------------------------------------------------- 161: holding back
@needs_mnist
def test_pruning_numbers_on_screen_are_the_study():
    s, a = study(PRUNING), anim(PRUNING)
    m, subs = s.metadata, subtitles(a)
    g, c = m["grown"], m["chosen"]["alpha"]
    assert "into %d pieces" % g["leaves"] in subs[0] and pct(g["held"], 1) in subs[0]
    assert "α = %g: %d leaves, held-out error %s against %s" % (
        c["setting"], c["leaves"], pct(c["held"], 1), pct(g["held"], 1)) in subs[4]
    end = texts(a, 0, 0.95)
    assert "depth %d" % g["depth"] in end and "%d leaves" % g["leaves"] in end
    assert "held-out %s" % pct(g["held"], 1) in end and "training 0.0%" in end
    seq = m["sequence"]
    pruned = texts(a, 3, 0.5)
    k = next(i for i, v in enumerate(seq["alphas"]) if "α = %.4f" % v in pruned)
    assert "%d leaves" % seq["leaves"][k] in pruned
    wrong, below, leaves = seq["cuts"][k]
    assert "next: (%d − %d) / %d ÷ (%d − 1) = %.4f" % (wrong, below, 2 * S.TWO_N, leaves,
                                                       seq["alphas"][k + 1]) in pruned
    chosen = texts(a, 4, 0.95)
    assert "chosen: α = %g" % c["setting"] in chosen
    assert "%d leaves: %s" % (c["leaves"], pct(c["held"], 1)) in chosen
    assert sum("|" in t for t in chosen) == c["leaves"]                       # its leaves, drawn
    o = m["over_seeds"]
    assert pct(o["two_depth"]["end"], 1) in subs[5] and pct(o["cells_alpha"]["best"], 1) in subs[5]


@needs_mnist
def test_the_drawn_tree_is_regrown_from_its_events():
    s = study(PRUNING)
    m, p0, p1, p2 = s.metadata, s.panels[0].data, s.panels[1].data, s.panels[2].data
    e = S.events()
    x, y, rows = S.sample("two", S.TWO_N, 0)
    assert np.allclose(p0["xy"], x) and np.allclose(p0["xy"], e["pool_two"][rows])
    tree, cuts = S.fit_two(x, y)
    kept = S.tree_from(p0, "tree_")
    assert all(np.array_equal(tree[k], kept[k]) for k in kept) and np.allclose(cuts, p0["cuts"])
    xb, hb = S.binned(x, cuts), S.binned(e["held_two"], cuts)
    assert S.error(tree, xb, y) == 0.0 == m["grown"]["train"]
    assert np.isclose(S.error(tree, hb, e["held_y"]), m["grown"]["held"])
    d = m["shown"]["depth"]
    assert np.isclose(p1["ys"][1][d - 1], S.error(tree, hb, e["held_y"], max_depth=d))
    assert p1["leaves"][d - 1] == S.leaf_count(tree, max_depth=d) <= 2 ** d
    assert np.all(np.diff(p1["ys"][0]) <= 1e-12)                  # training error only falls
    alphas, internals, links = S.pruning(tree)
    assert np.allclose(alphas, p2["alphas"]) and np.array_equal(internals, p2["internals"])
    assert np.array_equal(links, p2["links"])
    assert np.all(np.diff(p2["seq"][:, 2]) < 0) and p2["seq"][-1, 2] == 1
    small = S.tree_from(p0, "leaf_tree_")
    assert small["n"][small["feature"] < 0].min() >= m["shown"]["leaf"]
    gains = S.tree_from(p0, "gain_tree_")
    asks = gains["feature"] >= 0
    assert np.all((gains["gain"] * gains["n"])[asks] / gains["n"][0] >= m["shown"]["gain"])
    # folds: every training event held back exactly once, both classes in each
    fold = S.folds(y, 0)
    assert np.bincount(fold).tolist() == [2 * S.TWO_N // S.FOLDS] * S.FOLDS
    assert all(y[fold == k].sum() == S.TWO_N // S.FOLDS for k in range(S.FOLDS))
    cv = S.cross_validate(x, y, 0)
    assert np.allclose(cv["alpha"], p2["ys"][0]) and np.allclose(cv["depth"], p2["cv_depth"])
    k, _ = S.choose(cv["alpha"], S.ALPHAS)
    assert S.ALPHAS[k] == m["chosen"]["alpha"]["setting"] and cv["alpha"][k] == cv["alpha"].min()


@needs_mnist
def test_held_back_trees_do_better_on_two_features_and_no_worse_on_cells():
    s = study(PRUNING)
    m, p3 = s.metadata, s.panels[3].data
    g, c = m["grown"], m["chosen"]
    # the drawn tree: every control, set by cross-validation, beats growing to the end
    for name in ("depth", "leaf", "gain", "alpha"):
        assert c[name]["held"] < g["held"] - 0.02 and c[name]["leaves"] < g["leaves"] / 2
    two = p3["two_depth"]                              # seed, depth, (train, held, leaves)
    assert np.all(two[:, -1, 0] == 0.0)                # grown to the end: no training error
    best = two[:, :, 1].mean(0).argmin()
    assert 1 <= best + 1 <= 6
    assert np.all(two[:, -1, 1] > two[:, best, 1] + 0.02)          # it turns up, every seed
    for name in ("leaf", "gain", "alpha"):
        a = p3["two_" + name]
        k = a[:, :, 1].mean(0).argmin()
        assert k > 0 and np.all(a[:, 0, 1] > a[:, k, 1] + 0.02)
        assert np.allclose(a[:, 0], two[:, -1])        # the first setting is no control
    # 49 cells: no turn-up with depth at this size; pruning helps a little
    cells = p3["cells_depth"][:, :, 1].mean(0)
    assert cells[-1] - cells.min() < 0.01 < cells[0] - cells[-1]
    alpha = p3["cells_alpha"][:, :, 1]
    assert 0.002 < (alpha[:, 0] - alpha[:, 1]).mean() < 0.02
    o = m["over_seeds"]
    assert np.isclose(o["two_depth"]["end"], two[:, -1, 1].mean())
    assert np.isclose(o["cells_alpha"]["best"], alpha.mean(0).min())
    run = S.sweep_run("two", 2)
    assert np.allclose(run["depth"], two[2]) and np.allclose(run["alpha"], p3["two_alpha"][2])


# ---------------------------------------------------------------------- 162: instability
@needs_mnist
def test_instability_numbers_on_screen_are_the_study():
    s, a = study(UNSTABLE), anim(UNSTABLE)
    m, subs = s.metadata, subtitles(a)
    first = [t["threshold"][0] for t in m["shown"]]
    assert "(%.2f to %.2f)" % (min(first), max(first)) in subs[1]
    four = texts(a, 1, 0.9)
    for t in m["shown"]:
        for f, v in zip(t["feature"], t["threshold"]):
            assert M.question(f, v) in four
        assert "%d leaves" % t["leaves"] in four
    assert "On %s of held-out events" % pct(m["split_share"]) in subs[2]
    votes = s.panels[1].data["values"]
    drawn = [t for t in texts(a, 2, 0.9) if t in ("4", "9")]
    assert drawn.count("9") == int((1 - votes).sum())
    assert drawn.count("4") == int(votes.sum()) + 1    # and the 4 on the plane's axis
    d = m["disagreement"]
    for kind in S.KINDS:
        assert pct(d[kind]["end"]["mean"], 1) in subs[3]
    bars = texts(a, 3, 0.95)
    for kind in S.KINDS:
        for v in m["swapped"][kind]["changed"]:
            assert pct(v, 1) in bars
    p = m["parts"]["cells"]
    assert "falls from %.3f to %.3f" % (p["bias2"][0], p["bias2"][-1]) in subs[4]
    one = m["one_event"]
    last = texts(a, 4, 0.95)
    for row in (one["shallow"], one["deep"]):
        assert "bias² = (1 − %.2f)² = %.2f" % (row["mean"], row["bias2"]) in last
        assert "variance around it = %.2f" % row["variance"] in last
    assert "variance: what bagging removes" in last and "bias: what boosting removes" in last


@needs_mnist
def test_resampled_trees_differ_and_deep_trees_trade_bias_for_variance():
    s = study(UNSTABLE)
    m, p0, p1, p2, p3 = (s.metadata, *(p.data for p in s.panels))
    x, y, _ = S.sample("two", S.TWO_N, 0)
    assert np.allclose(p0["xy"], x)
    w = p0["weights"]
    assert np.all(w.sum(1) == len(y)) and np.all(np.abs((w > 0).mean(1) - 0.632) < 0.04)
    tree, cuts = S.fit_two(x, y, w[1])                 # a drawn tree, from its resample
    kept = S.tree_from(p0, "tree1_")
    assert all(np.array_equal(tree[k], kept[k]) for k in kept)
    e = S.events()
    assert np.array_equal(S.vote(tree, S.binned(p1["listed_xy"], cuts)), p1["values"][1])
    assert np.all(p1["values"].min(0) != p1["values"].max(0))     # listed: they disagree
    assert sorted(p1["labels"]) == [0] * S.INST_EVENTS + [1] * S.INST_EVENTS
    assert 0.05 < m["split_share"] < 0.5
    leaves = [t["leaves"] for t in m["shown"]]
    assert len({round(t["threshold"][1], 3) for t in m["shown"]}) == len(leaves)
    # how often two trees disagree: more when grown to the end, in every seed, on the cells
    cells = p2["per_seed"][1]
    assert np.all(cells[:, -1] > cells[:, 0] + 0.05)
    for k, kind in enumerate(S.KINDS):
        end = p2["per_seed"][k][:, -1]
        assert np.all(end > 0.08) and np.isclose(end.mean(),
                                                 m["disagreement"][kind]["end"]["mean"])
        sw = p2["swapped"][k].mean(0)
        assert np.all(np.diff(sw) > 0) and sw[0] < 0.03 < sw[-1]   # more swapped, more changed
        assert m["roots"][kind]["same_feature"] > 0.8              # the root mostly stays
    # two trees disagree more often than either is wrong more than the other
    assert m["disagreement"]["cells"]["end"]["mean"] > m["disagreement"]["cells"][
        "single_error_end"]
    run = S.resample_run("two", 1)
    assert np.allclose(run[0], p2["per_seed"][0][1]) and np.allclose(run[3], p2["swapped"][0][1])
    # bias and variance add up to the squared error, exactly, and move opposite ways
    for parts in (p3["ys"], p3["two"]):
        mse, bias2, var = parts
        assert np.allclose(mse, bias2 + var) and np.all(np.diff(var[2:]) > 0)
        assert var[-1] > var[0] + 0.04
    mse, bias2, var = p3["ys"]
    assert bias2[-1] < bias2[0] - 0.1 and np.all(np.diff(bias2) < 0)
    assert var[-1] > bias2[-1]                         # grown to the end: mostly variance
    assert p3["together"][-1] < p3["single"][-1] - 0.03            # which a vote would remove
    one = m["one_event"]
    assert one["label"] == 1 and np.isclose(one["deep"]["mse"],
                                            one["deep"]["bias2"] + one["deep"]["variance"])
    assert one["deep"]["variance"] > one["shallow"]["variance"] + 0.1
    assert set(np.unique(p3["one"][1])) <= {0.0, 1.0}  # pure leaves answer 0 or 1
    assert e["held_y"][S.instability_data()["listed"][0]] == 1


# ---------------------------------------------------------------------- 163: bagging
@needs_mnist
def test_bagging_numbers_on_screen_are_the_study():
    s, a = study(BAGGING), anim(BAGGING)
    m, subs = s.metadata, subtitles(a)
    pk = m["picked"]
    assert "here %d of %d come up" % (pk["unique"], pk["events"]) in subs[0]
    assert "%d are left out" % pk["left_out"] in subs[0]
    counts = s.panels[0].data["counts"]
    drawn = texts(a, 0, 0.95)
    for k in range(int(counts.max()) + 1):
        assert drawn.count("×%d" % k) == int((counts == k).sum())
    assert any(pct(pk["expected_share"]) in t and pct(1 - 1 / np.e, 1) in t for t in drawn)
    assert pct(m["plane"]["together"], 1) in subs[2]
    held = m["held"]
    for b in (1, 10, 100):
        assert pct(held[b]["mean"], 1) in subs[3]
    v = m["variance"]
    assert "ρ = %.3f and σ² = %.3f" % (v["rho"], v["sigma2"]) in subs[4]
    var = texts(a, 4, 0.95)
    assert "correlation ρ = %.3f" % v["rho"] in var
    assert "never below ρσ² = %.4f, however many trees" % v["floor"] in var
    assert "%s after 100 trees against %s held-out" % (pct(m["oob"][100]["mean"], 1),
                                                       pct(held[100]["mean"], 1)) in subs[5]


@needs_mnist
def test_the_bootstrap_and_the_vote_over_the_plane():
    s = study(BAGGING)
    m, p0, p1 = s.metadata, s.panels[0].data, s.panels[1].data
    n = len(p0["counts"])
    assert n == 2 * S.BAG_PICKED and p0["classes"].sum() == S.BAG_PICKED
    assert np.array_equal(np.bincount(p0["picks"], minlength=n), p0["counts"])
    assert p0["counts"].sum() == n and (p0["counts"] == 0).sum() == m["picked"]["left_out"] > 0
    assert np.isclose(m["picked"]["expected_share"], 1 - (1 - 1 / n) ** n)
    us = m["unique_share"]
    assert abs(us["measured"] - us["limit"]) < 0.005 and np.isclose(us["limit"], 1 - np.exp(-1))
    # the maps are the trees' answers: the first, regrown from its resample
    x, y, _ = S.sample("two", S.TWO_N, 0)
    w = p1["weights"]
    assert np.allclose(p1["xy"], x) and np.all(w.sum(1) == len(y)) and len(w) == S.BAG_TREES
    tree, cuts = S.fit_two(x, y, w[0])
    grid = S.vote(tree, S.binned(S.plane_grid(), cuts)).reshape(S.BAG_GRID, S.BAG_GRID)
    assert np.array_equal(grid, p1["maps"][0]) and np.array_equal(p1["values"][0], p1["maps"][0])
    assert np.allclose(p1["values"][2], p1["maps"][:3].mean(0))    # the vote of three
    assert S.plane_grid()[0, 1] > S.plane_grid()[-1, 1]            # row 0 is the top
    # single trees say 0 or 1; a hundred are divided over a band, and smoother
    last = p1["values"][-1]
    assert set(np.unique(p1["maps"])) == {0.0, 1.0}
    assert np.mean((last > 0.1) & (last < 0.9)) > 0.05
    steps = np.abs(np.diff(last, axis=1))              # a tree's edge is a step of 1
    assert steps[steps > 0].mean() < 0.4 and np.abs(np.diff(p1["maps"][0], axis=1)).max() == 1
    assert p1["together"][-1] < p1["single"].mean() - 0.015
    assert np.isclose(p1["together"][0], p1["single"][0])


@needs_mnist
def test_more_trees_help_then_flatten_as_the_formula_says():
    s = study(BAGGING)
    m, p2, p3 = s.metadata, s.panels[2].data, s.panels[3].data
    held, oob = p2["per_seed"]                         # sample, step
    steps = list(S.BAG_STEPS)
    assert np.all(held[:, -1] < held[:, 0] - 0.035)    # in every one of the 20 samples
    assert np.all(held[:, -1] < p2["single"].mean(1) - 0.04)
    gain_late = held[:, steps.index(30)].mean() - held[:, -1].mean()
    gain_early = held[:, 0].mean() - held[:, steps.index(10)].mean()
    assert 0 < gain_late < 0.15 * gain_early           # it flattens
    v = m["variance"]
    assert np.allclose(p3["ys"][0], p3["ys"][1], rtol=0.08) and v["worst_ratio"] < 0.08
    assert 0.02 < v["rho"] < 0.3 and np.isclose(v["floor"], v["rho"] * v["sigma2"])
    assert p3["ys"][0][-1] > v["floor"] > 0.02 * v["sigma2"]       # not to zero
    assert p3["ys"][0][0] > 10 * p3["ys"][0][-1]
    # out-of-bag: near the held-out error at 100 trees, above it early, when few trees judge
    gap = m["oob_minus_held"]
    assert abs(gap["mean"]) < 0.01 and gap["mean_abs"] < 0.012
    k = steps.index(10)
    assert oob[:, k].mean() > held[:, k].mean() + 0.01
    lag = m["held_at_share_of_trees"]
    assert abs(lag["held"] - oob[:, -1].mean()) < abs(held[:, -1].mean() - oob[:, -1].mean())
    run = S.bag_run(3)
    assert np.allclose(run["held"], held[3]) and np.allclose(run["oob"], oob[3])
    assert np.allclose(run["single"], p2["single"][3]) and abs(run["unique"].mean() - 0.632) < 0.01


# ---------------------------------------------------------------------- 164: random forest
@needs_mnist
def test_forest_numbers_on_screen_are_the_study():
    s, a = study(FOREST), anim(FOREST)
    m, subs = s.metadata, subtitles(a)
    p0 = s.panels[0].data
    for j in range(S.RF_NODES):                        # each node's handful and its choice
        drawn = texts(a, 0, (j + 0.9) / S.RF_NODES * 0.92 + 0.04)
        for c, g in zip(p0["values"][j], p0["gains"][j]):
            assert M.cell_name(int(c)) in drawn and "%.3f" % g in drawn
        thr = (p0["cut"][j] + 1) / S.CELL_BINS
        assert "taken: is the cell in %s ≤ %.2f?" % (M.cell_name(int(p0["taken"][j])),
                                                     thr) in drawn
    assert "they start from %d different cells" % m["first_cells"][S.RF_MTRY] in subs[1]
    best, f = m["best"], m["forest"]
    assert "%s of held-out events wrong with %d cells tried" % (pct(best["held"], 1),
                                                                best["mtry"]) in subs[2]
    assert pct(f[1]["mean"], 1) in subs[2] and pct(f[49]["mean"], 1) in subs[2]
    bars = texts(a, 2, 0.95)
    for v in (m["single"][49]["mean"], f[1]["mean"], f[49]["mean"], m["usual"], best["held"]):
        assert pct(v, 1) in bars
    imp = m["importance"]
    assert "from %s to %s" % (pct(imp["base"], 1),
                              pct(imp["base"] + imp["top_rise"][0], 1)) in subs[3]
    assert "%s to %s" % (pct(imp["base"], 1), pct(imp["base"] + imp["top_rise"][0], 1)) in texts(
        a, 3, 0.95)
    assert "%d cells share %s of the gain" % (imp["quiet_cells"],
                                              pct(imp["quiet_gain"])) in subs[4]


@needs_mnist
def test_a_handful_makes_trees_less_alike_and_the_forest_better():
    s = study(FOREST)
    m, p0, p1 = s.metadata, s.panels[0].data, s.panels[1].data
    rows = p1["per_seed"]                              # seed, cells tried, (forest, single, ...)
    forest, single, alike = rows[:, :, 0], rows[:, :, 1], rows[:, :, 2]
    one, bag = S.MTRY.index(1), S.MTRY.index(49)
    best = forest.mean(0).argmin()
    assert 0 < best < len(S.MTRY) - 1 and S.MTRY[best] == m["best"]["mtry"]
    assert np.all(forest[:, best] < forest[:, one] - 0.005)        # in every seed
    assert np.all(forest[:, best] < forest[:, bag] - 0.002)
    assert np.all(np.diff(alike, axis=1) > 0) and np.all(np.diff(single, axis=1) < 0)
    assert np.all(alike[:, bag] > 2 * alike[:, one]) and np.all(single[:, one] > 0.2)
    assert np.all(forest < single - 0.04)              # any forest beats its trees
    assert np.all(forest[:, :] <= rows[:, :, 3] + 0.002)           # and 100 trees beat 10
    # every bagged tree starts from the same cell; a handful scatters the start
    first = p1["first"]
    counts = [len(np.unique(row)) for row in first]
    assert counts[bag] == 1 and counts[S.MTRY.index(S.RF_MTRY)] >= 10 and counts[one] >= 25
    assert np.all(np.diff(counts) <= 0)
    # the handful drawn for the root: the cell taken is the best of those offered
    for j in range(S.RF_NODES):
        cells_, gains = p0["values"][j].astype(int), p0["gains"][j]
        assert len(set(cells_)) == S.RF_MTRY and p0["taken"][j] == cells_[gains.argmax()]
    assert p0["counts"][0, 0] == 2 * S.CELLS_N
    assert p0["counts"][0, 1] + p0["counts"][0, 2] == p0["counts"][0, 0]
    # bagging is the forest with all 49 tried: the same trees as the bagging study's
    res = S.bag_run(2)
    assert np.isclose(res["held"][-1], forest[2, bag]) and np.isclose(res["alike"], alike[2, bag])
    assert np.isclose(study(BAGGING).panels[2].data["per_seed"][0][2, -1], forest[2, bag])


@needs_mnist
def test_importance_by_gain_and_by_permutation():
    s = study(FOREST)
    m, p2, p3 = s.metadata, s.panels[2].data, s.panels[3].data
    gain, perm = p2["values"], p3["values"]
    assert np.isclose(gain.sum(), 1.0) and np.all(gain >= 0)
    assert np.allclose(p2["per_seed"].sum(1), 1.0)
    assert gain[0].max() < 0.002 and np.abs(perm[0]).max() < 0.001    # the blank top row
    imp = m["importance"]
    top = imp["top"][0]
    assert top == gain.argmax() == perm.argmax()       # they agree on the first
    r, c = divmod(top, S.SIDE)
    assert S.TOP_ROWS[0] // S.BLOCK <= r <= (S.TOP_ROWS[1] - 1) // S.BLOCK        # top centre
    assert S.TOP_COLS[0] // S.BLOCK <= c <= (S.TOP_COLS[1] - 1) // S.BLOCK
    assert np.all(p3["per_seed"][:, top] > 0.05)       # in every seed: five points and more
    assert imp["top_share_permutation"] > imp["top_share_gain"] + 0.1
    # ... and part below it: cells with a real share of the gain that the forest can do without
    assert imp["quiet_cells"] >= 5 and imp["quiet_gain"] > 0.1
    quiet = (gain.ravel() > 0.01) & (perm.ravel() < 0.001)
    assert quiet.sum() == imp["quiet_cells"] and np.isclose(gain.ravel()[quiet].sum(),
                                                            imp["quiet_gain"])
    assert np.all(p3["per_seed"][:, quiet].mean(0) < 0.001)
    assert imp["rank_correlation"] < imp["correlation"]
    assert np.isclose(np.clip(perm, 0, None).ravel()[top] / np.clip(perm, 0, None).sum(),
                      imp["top_share_permutation"])
    # shuffling a cell no tree asks about changes nothing; the function repeats
    x, y = step_data(300, 2)
    cells_ = np.column_stack([x, np.random.default_rng(3).random((300, S.CELLS - 2))])
    xb = S.binned(cells_, S.CELL_CUTS)
    trees = [S.grow(xb, y, S.CELL_BINS, w=S.bootstrap(np.random.default_rng(b), 300))
             for b in range(5)]
    rise, base = S.permutation_importance(trees, xb, y, np.random.default_rng(0))
    assert base < 0.02 and rise[0] > 0.2 and np.all(rise[2:] < 0.02)
    g = S.gain_importance(trees)
    assert g.argmax() == 0 and np.isclose(g.sum(), 1.0)
