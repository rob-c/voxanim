"""The experiment behind animation 178 (voxanim.study_bridge_cut): the figure of merit, the
threshold scan and its Gini gain, ROC curves and their areas, and the histogram likelihood
ratio on cases with known answers; then every measured claim as a relative claim (CI
builds the tree on another machine, where the last digits may differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_bridge_cut as B
from voxanim import study_trees as T

# ---------------------------------------------------------------------- one cut


def test_significance_of_known_counts():
    assert B.significance(9.0, 0.0) == pytest.approx(3.0)
    assert B.significance(30.0, 70.0) == pytest.approx(3.0)
    assert B.significance(0.0, 0.0) == 0.0 and B.significance(0.0, 5.0) == 0.0
    s, b = np.array([1.0, 4.0]), np.array([3.0, 5.0])
    np.testing.assert_allclose(B.significance(s, b), [0.5, 4 / 3])


def test_kept_counts_each_side():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = np.array([1, 1, 0, 1, 0, 0])
    s, b = B.kept(x, y, np.array([0.5, 2.5, 4.5, 9.0]))
    assert s.tolist() == [0, 2, 3, 3] and b.tolist() == [0, 0, 1, 3]
    s, b = B.kept(x, y, np.array([0.5, 2.5, 4.5, 9.0]), below=False)
    assert s.tolist() == [3, 1, 0, 0] and b.tolist() == [3, 3, 2, 0]
    assert B.signal_side(x, y) and not B.signal_side(-x, y)


def test_a_clean_gap_is_chosen_by_both_scores():
    x = np.r_[np.arange(5.0), np.arange(5.0) + 10]          # signal low, background high
    y = np.r_[np.ones(5, int), np.zeros(5, int)]
    sc = B.scan(x, y)
    assert sc["below"] and sc["fom_best"] == sc["gain_best"]
    k = sc["gain_best"]
    assert sc["cuts"][k] == pytest.approx(7.0)               # midway across the gap
    assert sc["gain"][k] == pytest.approx(0.5)               # Gini 0.5 -> two pure halves
    assert sc["fom"][k] == pytest.approx(np.sqrt(5.0))       # s = 5, b = 0


def test_the_scan_gain_is_the_tree_gain_and_weights_only_scale():
    rng = np.random.default_rng(4)
    x = np.r_[rng.normal(0, 1, 300), rng.normal(1.2, 1, 300)]
    y = np.r_[np.ones(300, int), np.zeros(300, int)]
    sc = B.scan(x, y)
    l1, l0 = B.kept(x, y, sc["cuts"], True)
    np.testing.assert_allclose(sc["gain"], T.split_gain(600, 300, l1 + l0, l1, "gini"))
    tree, cuts = T.fit_two(x[:, None], y, max_depth=1)        # the root: the same cut
    assert cuts[0, tree["cut"][0]] == pytest.approx(sc["cuts"][sc["gain_best"]])
    assert tree["gain"][0] == pytest.approx(sc["gain"].max())
    # the same weight on every event changes neither choice
    w = B.scan(x, y, weights=(0.25, 0.25))
    assert w["gain_best"] == sc["gain_best"] and w["fom_best"] == sc["fom_best"]
    np.testing.assert_allclose(w["fom"], 0.5 * sc["fom"])
    # rarer signal: the figure of merit wants a purer, tighter cut
    rare = B.scan(x, y, weights=(0.05, 1.0))
    assert rare["cuts"][rare["fom_best"]] < sc["cuts"][sc["fom_best"]]


# ---------------------------------------------------------------------- ROC


def _pairs(ss, sb):
    """The chance that a signal score beats a background one, ties half, by brute force."""
    d = np.asarray(ss)[:, None] - np.asarray(sb)[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def test_roc_and_area_on_known_scores():
    e, k = B.roc([3.0, 4.0], [1.0, 2.0])
    assert B.area(e, k) == pytest.approx(1.0)
    assert e.tolist() == [0, 0.5, 1, 1, 1] and k.tolist() == [0, 0, 0, 0.5, 1]
    assert B.area(*B.roc([1.0, 2.0], [3.0, 4.0])) == pytest.approx(0.0)
    e, k = B.roc(np.ones(5), np.ones(7))                     # one tie: a straight line
    assert e.tolist() == [0, 1] and k.tolist() == [0, 1]
    assert B.area(e, k) == pytest.approx(0.5)
    rng = np.random.default_rng(5)
    ss, sb = rng.integers(0, 6, 200), rng.integers(0, 4, 150)
    e, k = B.roc(ss, sb)
    assert B.area(e, k) == pytest.approx(_pairs(ss, sb))
    assert np.all(np.diff(e) >= 0) and np.all(np.diff(k) >= 0)
    # any rising function of the score keeps the same events: the same curve
    e2, k2 = B.roc(np.exp(ss / 2.0), np.exp(sb / 2.0))
    np.testing.assert_allclose(e2, e)
    np.testing.assert_allclose(k2, k)


def test_background_at_an_efficiency_follows_the_straight_pieces():
    e, k = np.array([0.0, 0.5, 0.5, 1.0]), np.array([0.0, 0.1, 0.3, 1.0])
    assert B.at_efficiency(e, k, 0.25) == pytest.approx(0.05)
    assert B.at_efficiency(e, k, 0.5) == pytest.approx(0.1)   # the lowest on a vertical
    assert B.at_efficiency(e, k, 0.75) == pytest.approx(0.65)


def test_the_histogram_ratio_divides_shares_cell_by_cell():
    rng = np.random.default_rng(6)
    x = np.r_[rng.normal(0, 1, (400, 2)), rng.normal(1, 1, (600, 2))]
    y = np.r_[np.ones(400, int), np.zeros(600, int)]
    edges, ratio, hs, hb = B.ratio_table(x, y, 5, smooth=0.0)
    assert hs.sum() == 400 and hb.sum() == 600
    assert np.all(np.abs(np.r_[hs + hb].sum(1) - 200) <= 1)   # equal-count bins of feature 0
    ok = hb > 0
    np.testing.assert_allclose(ratio[ok], (hs[ok] / 400) / (hb[ok] / 600))
    score = B.ratio_score(x, edges, ratio)
    r, c = B.ratio_cells(x, edges)
    np.testing.assert_allclose(score, ratio[r, c])
    _, smooth, _, _ = B.ratio_table(x, y, 5)
    assert np.all(np.isfinite(smooth)) and np.all(smooth > 0)


# ---------------------------------------------------------------------- measured


@pytest.fixture(scope="module")
def study():
    return B.from_a_cut_to_a_tree().validate()


@needs_mnist
def test_events_and_counts(study):
    m = study.metadata
    e = T.events()
    assert not set(e["pool_index"]) & set(e["held_index"])        # held out means held out
    assert m["n_train"] == m["train_signal"] + m["train_background"] == 8000
    assert m["n_held"] == m["held_signal"] + m["held_background"] == len(e["held_y"])
    root = m["root"]
    assert sum(root["left"]) + sum(root["right"]) == m["n_train"]
    assert root["left"][0] + root["right"][0] == m["train_signal"]
    assert len(study.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]


@needs_mnist
def test_the_stump_is_the_gini_cut_and_close_to_the_figure_of_merit(study):
    m = study.metadata
    one, rare = m["one_feature"], m["rare"]
    assert m["stump_is_gini_cut"] and m["root"]["feature"] == B.FEATURE
    assert m["root"]["gain"] == pytest.approx(one["by_gini"]["gain"])
    assert one["by_gini"]["gain"] >= one["by_fom"]["gain"]
    assert one["by_fom"]["fom"] >= one["by_gini"]["fom"]
    # on an even mix the two scores choose nearly the same cut ...
    assert one["apart"] < 0.1
    assert abs(one["by_fom"]["held"]["efficiency"] - one["by_gini"]["held"]["efficiency"]) \
        < 0.03
    # ... and the root's feature is the better one by Gini
    assert one["by_gini"]["gain"] > m["other_feature"]["by_gini"]["gain"]
    # with the 4s rare both move to a much tighter cut, close together
    assert rare["by_fom"]["cut"] < one["by_gini"]["cut"] - 0.1
    assert abs(rare["by_fom"]["cut"] - rare["by_gini"]["cut"]) < 0.1
    assert rare["by_fom"]["held"]["background"] < 0.5 * one["by_fom"]["held"]["background"]


@needs_mnist
def test_deeper_fits_training_but_held_out_turns(study):
    m = study.metadata
    D = m["depths"]
    train = np.array([d["train_area"] for d in D])
    heldout = np.array([d["held_area"] for d in D])
    assert np.all(np.diff(train) >= -1e-12) and train[-1] == pytest.approx(1.0)
    assert all(d["leaves"] <= 2 ** d["depth"] for d in D)
    best = m["best_depth"]
    assert 3 <= best <= 8 and heldout[best - 1] == heldout.max()
    assert heldout[best - 1] > heldout[0] + 0.05                    # far better than a stump
    assert heldout[-1] < heldout[best - 1] - 0.05                   # grown out: worse again
    assert train[-1] - heldout[-1] > 0.1


@needs_mnist
def test_the_tree_approaches_the_likelihood_ratio(study):
    m = study.metadata
    r = m["ratio"]
    best = m["depths"][m["best_depth"] - 1]
    assert r["held_area"] > m["cut_area"] + 0.02                    # two features beat one
    assert abs(r["held_area"] - best["held_area"]) < 0.01           # the tree comes close
    assert r["held_area"] >= best["held_area"] - 0.002               # and not past it
    stump = m["stump_held"]["background"]
    assert best["background_at_stump"] < 0.5 * stump
    assert r["background_at_stump"] < 0.5 * stump
    # the stump is one point of the one-feature curve
    assert abs(m["cut_background_at_stump"] - stump) < 0.01
    # the ratio's area hardly depends on the number of bins from 6 to 16
    sweep = dict(zip(r["sweep"]["bins"], r["sweep"]["held_area"]))
    mid = [sweep[b] for b in (6, 8, 10, 12, 16)]
    assert max(mid) - min(mid) < 0.005 and sweep[40] < max(mid)


@needs_mnist
def test_the_cached_run_reloads_the_same():
    a = B.from_a_cut_to_a_tree().metadata
    B.run.cache_clear()
    b = B.from_a_cut_to_a_tree().metadata
    assert a["best_area"] == b["best_area"] and a["root"] == b["root"]
