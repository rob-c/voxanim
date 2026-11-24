"""The numbers behind animation 279 (voxanim.study_blocks_zj): the followed jet, its list and
its shuffle, phi on each row and the sum, and the numbers the screen writes. Everything comes
from voxanim.study_jets (the numpy twin of examples/jets_as_sets.py). No dataset is needed:
the jets are made up."""

from __future__ import annotations

import numpy as np

from voxanim import study_blocks_zj as Z
from voxanim import study_jets as J
from voxanim.next_catalog import build_study


def meta():
    return build_study("study_blocks_zj", "jets-as-sets").metadata


def test_the_numbers_are_written_as_the_screen_writes_them():
    assert Z.pct(0.902) == "90.2%" and Z.pct(0.92375) == "92.4%"
    assert Z.p_text(0.7218) == "0.72" and Z.p_text(0.0113) == "0.01"
    assert Z.tiny(3.84e-7) == "0.0000004" and Z.tiny(1e-6) == "0.000001" and Z.tiny(0) == "0"


def test_the_study_has_four_panels_and_its_code():
    s = build_study("study_blocks_zj", "jets-as-sets")
    assert len(s.panels) == 4 and s.title == "Jets as sets of particles"
    m = s.metadata
    assert m["code_sum"][-1] == "    return rho(h.sum(1))"
    assert m["code_pad"][0] == "def as_list(p):"


def test_the_followed_jets_are_chosen_by_the_rule():
    w, o = Z.followed()
    X, M, y, _Xs = J.test_set()
    p = J.probs()
    assert y[w] == 1 and y[o] == 0
    assert M[w].sum() == M[o].sum() and 11 <= M[w].sum() <= 14
    for i in (w, o):
        assert np.abs(X[i, :, 1:]).max() < Z.REACH                # inside the panel
        assert (p["mlp"][i] > 0.5) == y[i] and (p["ds"][i] > 0.5) == y[i]
    flip = (p["mlp"] > 0.5) != (p["mlp_shuffled"] > 0.5)
    assert flip[w] and not flip[o]
    moved = np.abs(p["mlp"] - p["mlp_shuffled"])
    middle = np.median(moved[flip])
    two = np.where((np.abs(X[:, :, 1:]).max(axis=(1, 2)) < Z.REACH) & flip & (y == 1)
                   & ((p["mlp"] > 0.5) == y) & ((p["ds"] > 0.5) == y)
                   & (M.sum(1) >= 11) & (M.sum(1) <= 14))[0]
    assert abs(moved[w] - middle) == np.abs(moved[two] - middle).min()


def test_the_followed_jet_its_shuffle_and_its_sum():
    m = meta()
    f = m["followed"]
    rows, shuffled = np.asarray(f["rows"]), np.asarray(f["shuffled"])
    n = f["n"]
    assert f["pads"] == 20 - n and sum(f["mask"]) == n
    assert sorted(f["perm"]) == list(range(n))
    np.testing.assert_array_equal(shuffled[:n], rows[f["perm"]])  # slot s holds row perm[s]
    assert not np.array_equal(shuffled, rows)
    phi = np.asarray(m["phi_rows"])
    assert not phi[n:].any() and phi[:n].any()                     # the pads' rows: 0
    assert np.abs(np.asarray(m["phi_pad"])).sum() > 0               # phi(0) itself is not 0
    np.testing.assert_allclose(phi.sum(0), m["phi_sum"], atol=1e-5)
    assert m["phi_sum_moved"] < 1e-5                               # the same sum, shuffled
    assert f["p_mlp_again"] == f["p_mlp"]
    assert f["ds_moved"] < 1e-6 and m["p_text"]["ds"] == m["p_text"]["ds_shuffled"]
    assert m["p_text"]["mlp"] != m["p_text"]["mlp_shuffled"]


def test_what_the_screen_says():
    m = meta()
    r = J.results()
    assert m["right_text"] == {k: Z.pct(v) for k, v in r["right"].items()}
    assert m["right_text"]["ds"] == m["right_text"]["ds_shuffled"]
    assert m["right_text"]["mlp"] == m["right_text"]["mlp_again"] != m["right_text"][
        "mlp_shuffled"]
    assert m["changed"] == r["mlp"]["changed"] and m["changed_ds"] == 0
    assert m["count_text"] == {"mlp": "8,129", "ds": "2,273"}
    assert float(m["ds_bound_text"]) >= m["ds_max_moved"]
    assert m["median_moved"] < 0.1                                  # "most move little"
