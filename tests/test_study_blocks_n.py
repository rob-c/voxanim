"""The numbers behind animations 224-226 (voxanim.study_blocks_n): the places of a 3 x 3 stamp
and how often each pixel is read, without padding and with padding=1, the ring of zeros
against the paper's -0.42, and "same" against "valid" for k = 3, 5, 7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_n as S
from voxanim import study_course as course
from voxanim import study_padding as pad

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue_titles():
    return {item["number"]: item["title"] for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_are_the_catalogues():
    titles = catalogue_titles()
    assert (titles[224], titles[225], titles[226]) == (
        "Why maps shrink", "A ring of padding", "Same and valid")
    assert titles[223] == "CNN mistakes" and titles[227] == "What goes in the border"


def test_the_stamps_settings_are_read_from_the_script():
    assert S.conv_settings("pad-none") == (1, 8, 3, 0)
    assert S.conv_settings("pad-one") == (1, 8, 3, 1)
    assert S.conv_settings("pad-same") == (1, 8, "k", "same")     # over two lines


@pytest.mark.parametrize("side,k,padding", [(28, 3, 0), (28, 3, 1), (28, 5, 0), (28, 5, 2),
                                            (10, 3, 0), (7, 7, 3)])
def test_places_counted_one_by_one_are_the_coverage(side, k, padding):
    n = pad.out_side(side, k, padding)
    places = S.places(side, k, padding)
    assert len(places) == n * n and places[0] == (0, 0) and places[-1] == (n - 1, n - 1)
    full = S.coverage_after(len(places), side, k, padding)
    np.testing.assert_array_equal(full, pad.coverage(side, k, padding))
    # every pixel: the places that read it are as many as its coverage
    for r in range(side):
        for c in (0, side // 2, side - 1):
            assert len(S.covering((r, c), side, k, padding)) == full[r, c]
    # the map fills as the stamp sweeps: never more than at the end
    half = S.coverage_after(len(places) // 2, side, k, padding)
    assert (half <= full).all()
    reads = (len(places) // 2) * k * k
    if padding == 0:
        assert half.sum() == reads
    else:                                  # the top row's places read the ring too
        assert half.sum() < reads


def test_corner_edge_and_middle():
    assert S.CORNER == (0, 27) and S.EDGE[1] == 27 and 2 <= S.EDGE[0] <= 25
    cov0, cov1 = pad.coverage(28, 3, 0), pad.coverage(28, 3, 1)
    assert [cov0[p] for _n, p in S.SPOTS] == [1, 3, 9]
    assert [cov1[p] for _n, p in S.SPOTS] == [4, 6, 9]
    # padding does not make the border read as often as the middle
    assert cov1[S.CORNER] < cov1[S.EDGE] < cov1[S.MIDDLE] == 9
    assert S.read_counts(cov0) == {1: 4, 2: 8, 3: 96, 4: 4, 6: 96, 9: 576}
    assert S.read_counts(cov1) == {4: 4, 6: 104, 9: 676}
    assert S.covering(S.CORNER, padding=0) == [(0, 25)]
    assert S.covering(S.CORNER, padding=1) == [(0, 26), (0, 27), (1, 26), (1, 27)]


def test_same_and_valid():
    s = S.same_and_valid().validate()
    rows = s.metadata["rows"]
    assert [(r["k"], r["pad"], r["same"], r["valid"]) for r in rows] == [
        (3, 1, 28, 26), (5, 2, 28, 24), (7, 3, 28, 22)]
    assert [r["pad_sum"] for r in rows] == ["(3 − 1) ÷ 2 = 1", "(5 − 1) ÷ 2 = 2",
                                            "(7 − 1) ÷ 2 = 3"]
    assert [r["valid_sum"] for r in rows] == ["28 − 2 = 26", "28 − 4 = 24", "28 − 6 = 22"]
    assert rows[2]["printed"] == ["7 torch.Size([64, 8, 28, 28])",
                                  "7 torch.Size([64, 8, 22, 22])"]
    np.testing.assert_array_equal(s.panels[3].data["values"],
                                  [[3, 1, 28, 26], [5, 2, 28, 24], [7, 3, 28, 22]])


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_maps_shrink():
    s = S.maps_shrink().validate()
    m = s.metadata
    assert (m["across"], m["places"]) == (26, 676)
    assert (m["across_sum"], m["places_sum"]) == ("28 − 2 = 26", "26 × 26 = 676")
    assert (m["corner"], m["edge"], m["middle"]) == (1, 3, 9)
    assert m["spots"]["edge"]["places"] == [[22, 25], [23, 25], [24, 25]]
    assert len(m["spots"]["middle"]["places"]) == 9
    assert m["shape_change"] == "[64, 1, 28, 28] → [64, 8, 26, 26]"
    assert m["fewer"] == 784 - 576 and m["label"] == 5
    np.testing.assert_array_equal(s.panels[0].data["values"], course.digit(0)[0][0])
    np.testing.assert_array_equal(s.panels[1].data["values"], pad.coverage())


@needs_mnist
def test_a_ring():
    s = S.a_ring().validate()
    m = s.metadata
    padded = s.panels[0].data["values"]
    assert padded.shape == (30, 30) and m["big_sum"] == "1 + 28 + 1 = 30"
    ring = np.ones((30, 30), bool)
    ring[1:-1, 1:-1] = False
    assert (padded[ring] == 0).all() and m["ring_cells"] == 116
    np.testing.assert_array_equal(padded[1:-1, 1:-1], course.digit(0)[0][0])
    # the paper after Normalize is -0.42, so the ring of 0 is not paper
    assert m["paper_value"] == pytest.approx(pad.BG) and round(pad.BG, 4) == -0.4242
    assert padded[1:-1, 1:-1].min() == pytest.approx(pad.BG, abs=1e-6)
    assert m["ring_value"] > m["paper_value"] and 0.1 < m["ring_shade"] < 0.15
    assert m["bg_sum"] == "(0 − 0.1307) ÷ 0.3081 = −0.42"
    # the magnified corner is the top right, where the last place across lands
    assert np.allclose(m["corner3"], [[0, 0, 0], [pad.BG, pad.BG, 0], [pad.BG, pad.BG, 0]])
    assert (m["across"], m["places"], m["across_sum"]) == (28, 784, "30 − 2 = 28")
    assert (m["corner"], m["edge"], m["middle"]) == (4, 6, 9)
    assert m["shape_change"] == "[64, 1, 28, 28] → [64, 8, 28, 28]"
    np.testing.assert_array_equal(s.panels[1].data["values"], pad.coverage(pad=1))
