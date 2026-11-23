"""Animations 172-173 (voxanim.mech_info_trees): entropy and Gini in a tree's nodes, and
entropy followed down a whole tree. The layout is clean, the loop closes, every phase says
what kind of evidence it shows, the numbers on screen are the study's
(voxanim.study_info), and the arithmetic the animations rest on holds."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_info_trees as M
from voxanim import study_info as S
from voxanim.mech_phys_apps import commas, pct
from voxanim.next_catalog import build_study

HAND, DOWN = "entropy-and-gini-by-hand", "entropy-down-a-tree"
EACH = [pytest.param(name, marks=needs_mnist) for name in (HAND, DOWN)]
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_info", name))
    return _built[name]


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts]


def has(strings, part):
    return any(part in s for s in strings)


# ---------------------------------------------------------------------- both
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a, _ = built(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_every_frame_of_a_coarse_pass_builds(name):
    a, _ = built(name)
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a, _ = built(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))


# ---------------------------------------------------------------------- no data needed
def test_code_words_are_a_prefix_code_with_the_lengths_given():
    counts = np.array([5.0, 9, 12, 13, 16, 45, 0])
    lengths = S.huffman_lengths(counts)
    words = M.code_words(lengths)
    assert [len(w) for w in words] == [int(v) for v in lengths] and words[-1] == ""
    live = [w for w in words if w]
    for i, a in enumerate(live):
        for j, b in enumerate(live):
            assert i == j or not b.startswith(a)          # no codeword begins another
    assert sum(2.0 ** -len(w) for w in live) == 1.0       # and none could be shorter
    assert M.code_words([1, 2, 2]) == ["0", "10", "11"]
    p = counts / counts.sum()
    mean = float((p * lengths).sum())
    assert S.entropy(counts) <= mean < S.entropy(counts) + 1


def test_pairs_and_the_questions_of_a_node():
    counts = np.array([30.0, 50, 20, 0, 0, 5, 0, 0, 10, 0])
    share, err = S.pairs_differ(counts, n=4000, seed=3)
    run, a, b = M.running_share(counts, n=4000, seed=3)
    assert run[-1] == share and len(a) == len(b) == 12
    assert abs(share - S.gini(counts)) < 4 * err
    assert M.ask(5) == "ink ≤ 0.188?" and M.threshold(5) == 6 / 32
    assert M.cell_name(23) == "row 4, column 3"
    assert M.exact(10000) == "exact arithmetic on the counts of 10,000 training digits"
    assert M.is_evidence(M.MEASURED % "10") and M.is_evidence(M.TRAINING % "10")
    assert not M.is_evidence("measured elsewhere")
    inks = [M.entropy_ink(h) for h in (0.0, 1.0, 3.0)]
    assert inks[0][1] > inks[1][1] > inks[2][1]             # deeper magenta: less green


def test_a_tree_s_depth_map_and_layout():
    rng = np.random.default_rng(0)
    xb = rng.integers(0, S.BINS, (400, 3))
    y = (xb[:, 0] // 4 + xb[:, 1] // 11) % S.CLASSES
    tree = S.grow_multi(xb, y, max_depth=5)
    H = S.impurity_of(tree["counts"], "entropy")
    depths = range(6)
    m = M.depth_map(tree, xb, depths)
    for d in depths:                                      # a row's mean: the entropy left
        assert np.isclose(m[d].mean(), H[S.path_nodes(tree, xb, d)].mean())
    rank = M.preorder(tree)
    assert rank[0] == 0 and sorted(rank) == list(range(len(rank)))
    pos = M.node_layout(tree, 2, 0.1, 0.9)
    for t in np.flatnonzero((tree["feature"] >= 0) & (tree["depth"] < 2)):
        a, b = pos[int(tree["left"][t])], pos[int(tree["right"][t])]
        assert a < pos[int(t)] < b


# ---------------------------------------------------------------------- 172: by hand
@needs_mnist
def test_by_hand_numbers_on_screen_are_the_study():
    a, s = built(HAND)
    m, subs = s.metadata, subtitles(a)
    root, left, right, deep = m["nodes"]
    one = texts(a, 0, 0.95)
    for c, p in zip(root["counts"], root["shares"]):
        assert commas(c) in one and "%.3f" % p in one
    two = texts(a, 1, 0.95)
    assert all("%.3f" % v in two for v in root["entropy_terms"])
    assert "%.3f" % root["entropy"] in two and "%.3f" % m["max_entropy"] in two
    assert "%.3f bits here" % root["entropy"] in subs[1]
    three = texts(a, 2, 0.97)
    assert all("%.3f" % v in three for v in root["gini_terms"])
    assert "share that differ: %.4f ± %.4f; Gini %.4f" % (
        root["pairs_differ"], root["pairs_error"], root["gini"]) in three
    assert "%.4f ± %.4f differ" % (root["pairs_differ"], root["pairs_error"]) in subs[2]
    four = texts(a, 3, 0.95)
    words = M.code_words(root["huffman_lengths"])
    assert all(len(w) for w in words) and has(four, "a Huffman code")
    for v in (root["entropy"], root["huffman_mean"], deep["entropy"], deep["huffman_mean"]):
        assert "%.3f" % v in four
    five = texts(a, 4, 0.95)
    assert "yes: %s digits" % commas(left["n"]) in five
    assert "no: %s digits" % commas(right["n"]) in five
    assert has(five, "= %.3f bits" % m["weighted_children_entropy"])
    assert has(five, "= %.3f bits" % m["first_gain"]) and M.ask(root["cut"]) in five
    assert "%.3f   %.3f" % (deep["entropy"], deep["gini"]) in five
    assert "gain of %.3f" % m["first_gain"] in subs[4]
    six = texts(a, 5, 0.95)
    for v in m["nats_vs_gini"]:
        assert "%.3f" % v["nats"] in six and "%.3f" % v["gini"] in six


@needs_mnist
def test_by_hand_study_holds():
    _, s = built(HAND)
    m = s.metadata
    root, left, right, deep = m["nodes"]
    assert root["n"] == left["n"] + right["n"] == m["n_train"]
    assert np.allclose(np.add(left["counts"], right["counts"]), root["counts"])
    assert root["entropy"] <= m["max_entropy"] and root["gini"] <= m["max_gini"]
    assert np.isclose(m["first_gain"], root["entropy"] - m["weighted_children_entropy"])
    for n in m["nodes"]:
        assert np.isclose(sum(n["entropy_terms"]), n["entropy"])
        assert np.isclose(sum(n["gini_terms"]), n["gini"])
        assert n["entropy"] <= n["huffman_mean"] < n["entropy"] + 1
        assert abs(n["pairs_differ"] - n["gini"]) < 4 * n["pairs_error"]
        assert n["entropy_nats"] >= n["gini"]
    assert max(deep["shares"]) > 0.9 and deep["huffman_mean"] >= 1.0


# ---------------------------------------------------------------------- 173: down a tree
@needs_mnist
def test_down_a_tree_numbers_on_screen_are_the_study():
    a, s = built(DOWN)
    m, subs = s.metadata, subtitles(a)
    D = m["depths"]
    one = texts(a, 0, 0.95)
    assert "%s nodes," % commas(m["nodes"]) in one and "%s leaves" % commas(m["leaves"]) in one
    assert "%.2f" % D[0]["entropy"] in one and "%.2f bits" % S.LOG2_10 in one
    two = texts(a, 1, 0.95)
    assert "gains %.3f = %.3f − %.3f left" % (m["total_gain"], D[0]["entropy"],
                                              D[-1]["entropy"]) in two
    assert "add up to %.3f bits" % m["total_gain"] in subs[1]
    three = texts(a, 2, 0.97)
    for d in (0, 1, 5, 10, 14):                        # entropy × ln 2 is the training loss
        r = D[d]
        assert "%.4f" % r["entropy"] in three and "%.4f" % r["train_nats_unsmoothed"] in three
        assert "%.4f" % (r["entropy"] * np.log(2)) == "%.4f" % r["train_nats_unsmoothed"]
    assert "%.3f at the root" % D[0]["train_nats_unsmoothed"] in subs[2]
    four = texts(a, 3, 0.97)
    assert commas(D[-1]["test_zero_probability"]) in four
    assert "gives %s held-out digits probability 0" % commas(D[-1]["test_zero_probability"]) \
        in subs[3]
    five = texts(a, 4, 0.97)
    best = D[m["best_depth"]]
    assert "%.3f" % best["test_nats"] in five
    assert "root: %.3f nats; best: %.3f at depth %d" % (D[0]["test_nats"], best["test_nats"],
                                                        best["depth"]) in five
    assert "%s at depth %d, %s at %d" % (pct(best["test_accuracy"]), best["depth"],
                                         pct(D[-1]["test_accuracy"]), D[-1]["depth"]) in five
    assert "to %.3f at depth %d (%s right)" % (best["test_nats"], best["depth"],
                                               pct(best["test_accuracy"])) in subs[4]


@needs_mnist
def test_down_a_tree_study_holds():
    a, s = built(DOWN)
    m = s.metadata
    D = m["depths"]
    H = np.array([r["entropy"] for r in D])
    test = np.array([r["test_nats"] for r in D])
    assert np.all(np.diff(H) < 0)                             # training entropy keeps falling
    assert np.allclose(m["gain_by_depth"], -np.diff(H))       # each depth's gains: what goes
    assert np.isclose(m["total_gain"], H[0] - H[-1])
    assert np.allclose(H * np.log(2), [r["train_nats_unsmoothed"] for r in D])
    best = m["best_depth"]
    assert 0 < best < len(D) - 1 and test[best] == test.min() and test[-1] > test[best]
    assert abs(test[0] - np.log(S.CLASSES)) < 0.01           # the root: about ln 10
    assert D[-1]["test_zero_probability"] > 0 == D[0]["test_zero_probability"]
    # the drawn depth map averages to the entropy left, and the zero example is real
    assert np.allclose(a.map.mean(1), H)
    assert a.zero_counts[a.zero_label] == 0 and a.zero_counts.sum() > 0
