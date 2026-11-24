"""The numbers behind animations 272 and 278 (voxanim.study_blocks_zf): the ensembles of SGD runs
against HESSE, and the Asimov dataset against toy experiments. Everything comes from
voxanim.study_ensembles and voxanim.study_asimov (the numpy twins of examples/ensembles.py and
examples/asimov.py); the claims shown on screen are checked against them here. No dataset is
needed: the data are made up."""

from __future__ import annotations

import numpy as np

from voxanim import blocks as B
from voxanim import study_asimov as A
from voxanim import study_blocks_zf as Z
from voxanim import study_ensembles as E
from voxanim.next_catalog import build_study


def ens():
    return build_study("study_blocks_zf", "error-bars-on-a-network")


def asi():
    return build_study("study_blocks_zf", "asimov-dataset")


def test_the_numbers_are_written_as_the_screen_writes_them():
    assert Z.two(0.14708) == "0.15" and Z.three(0.017259) == "0.017"
    assert Z.signed(-0.8219) == "−0.82" and Z.signed(0.5) == "0.50"
    assert [Z.num(v) for v in (33.0, 9.2, 0.8, 10.0)] == ["33", "9.2", "0.8", "10"]


def test_both_studies_have_four_panels_and_their_titles():
    for s, title in ((ens(), "Error bars on a network"), (asi(), "The Asimov dataset")):
        assert s.title == title and len(s.panels) == 4
        kinds = {p.kind for p in s.panels}
        assert kinds <= {"curve", "bars", "matrix", "points", "images"}
        for p in s.panels:
            if p.kind == "points":
                assert np.asarray(p.data["xy"]).shape[1] == 2


# ---------------------------------------------------------------------- 272
def test_ensembles_metadata_is_the_studys():
    m = ens().metadata
    c = E.compare()
    np.testing.assert_allclose(m["seeds"], c["seeds"])
    np.testing.assert_allclose(m["boots"], c["boots"])
    assert m["seeds_text"] == ["0.017", "0.016", "0.021"] and m["seeds_round"] == "0.02"
    assert m["boots_text"] == ["0.15", "0.14", "0.11"]
    assert m["hesse_text"] == ["0.18", "0.21", "0.11"]
    assert m["toys_text"] == ["0.18", "0.22", "0.11"]
    assert m["corr_boots_text"] == "−0.82" and m["corr_hesse_text"] == "−0.81"
    assert m["first_text"] == ["5.15", "2.44", "1.02"]
    assert (m["resample_distinct"], m["resample_out"], m["resample_twice"]) == (28, 12, 11)
    assert m["resample_out"] + m["resample_distinct"] == 40
    assert sum(m["resample"]) == 40 and max(m["resample"]) == m["resample_most"] == 3
    assert m["printed_seeds"] == "[0.017 0.016 0.021]" and m["printed_hesse"] == "[0.18 0.21 0.11]"
    assert 0.3 < m["short_tau"] < 0.4                   # "about a third short in τ"
    assert m["runs"] == 100 and m["steps"] == 3000 and m["batch"] == 8 and m["lr"] == 0.005


def test_every_answer_is_inside_the_plane_drawn():
    m = ens().metadata
    pts = np.concatenate([m["seeds"], m["boots"]])[:, 1:]
    (t0, t1), (b0, b1) = m["tau_range"], m["b_range"]
    assert (pts[:, 0] > t0).all() and (pts[:, 0] < t1).all()
    assert (pts[:, 1] > b0).all() and (pts[:, 1] < b1).all()
    for ell in (m["hesse_ellipse"], m["boots_ellipse"]):
        e = np.asarray(ell)
        assert (e[:, 0] > t0).all() and (e[:, 0] < t1).all()
        assert (e[:, 1] > b0).all() and (e[:, 1] < b1).all()


def test_ensembles_code_is_the_scripts():
    m = ens().metadata
    for key, region in (("code_train", "en-train"), ("code_seeds", "en-seeds"),
                        ("code_bootstrap", "en-bootstrap"), ("code_spread", "en-spread"),
                        ("code_hesse", "en-hesse")):
        assert m[key] == list(B.code_lines(region))
        assert B.code_file(region) == "examples/ensembles.py"
        assert len(m[key]) <= B.MAX_LINES and max(map(len, m[key])) <= B.MAX_CHARS
    assert "for seed in range(100):" in m["code_seeds"]
    assert "    points = draw.integers(0, 40, 40)" in m["code_bootstrap"]


# ---------------------------------------------------------------------- 278
def test_asimov_metadata_is_the_studys():
    m = asi().metadata
    near, far = A.case("b"), A.case("b/10")
    assert m["expected_text"] == ["33", "28", "20", "11"]
    assert m["expected_text_less"] == ["6", "10", "9.2", "3.8"]
    assert (m["median_text"], m["asimov_text"], m["simple_text"]) == ("2.89", "2.93", "3.16")
    assert (m["median_text_less"], m["asimov_text_less"], m["simple_text_less"]) == (
        "6.74", "6.77", "9.98")
    assert m["first"] == [35, 33, 17, 14] and m["first_z_text"] == "3.38"
    assert m["first_mu_text"] == "1.19"
    assert m["median"] == near["median"] and m["asimov_z_less"] == far["asimov_z"]
    assert sum(m["hist"]) == sum(m["hist_less"]) == A.TOYS          # every toy in the chart
    assert m["toys_text"] == "10,000"


def test_asimov_code_is_the_scripts():
    m = asi().metadata
    for key, region in (("code_counts", "as-counts"), ("code_z", "as-z"),
                        ("code_toys", "as-toys"), ("code_asimov", "as-asimov"),
                        ("code_simple", "as-simple"), ("code_less", "as-less")):
        assert m[key] == list(B.code_lines(region))
        assert B.code_file(region) == "examples/asimov.py"
        assert len(m[key]) <= B.MAX_LINES and max(map(len, m[key])) <= B.MAX_CHARS
    assert m["code_asimov"][0] == "asimov = s + b"
