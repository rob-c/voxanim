"""The numbers behind animations 236-237 (voxanim.study_blocks_s): an IVF index that looks
in fewer places, and a retrained encoder that needs a new store."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_s as S
from voxanim import study_course as course
from voxanim import study_vectordb as D


# ---------------------------------------------------------------------- no data needed
def test_percentages():
    assert S.pct(0.99539) == "99.5%" and S.pct(0.0808570) == "8.1%"
    assert S.pct(0.0759) == "7.6%" and S.pct(0.10030) == "10.0%"
    assert S.pct(0.99997) == "100.0%"


# ---------------------------------------------------------------------- MNIST
@needs_mnist
def test_the_query_followed_is_picked_by_its_rule():
    t = S.example_query()
    assert t == 1
    for earlier in range(t):        # every earlier test digit fails the rule
        assert (S.visit(earlier, 1)["found"] == D.K
                or S.visit(earlier, S.MARKED)["found"] < D.K)


@needs_mnist
def test_visit_is_the_scripts_ivf_search():
    iv = D.ivf()
    v = D.store(0)["vectors"]
    for t in (0, 1, 2):
        q = D.query(t, 0)
        for nprobe in (1, 2, 4):
            s = S.visit(t, nprobe)
            near = np.argsort(-(iv["centres"] @ q))[:nprobe]
            assert sorted(s["near"]) == sorted(int(j) for j in near)
            assert s["scanned"] == int(np.isin(iv["home"], near).sum())
            assert s["scanned"] == int(iv["list_sizes"][near].sum())
        # all 64 lists: everything scanned, every one of the 10 nearest found
        assert S.visit(t, 64)["scanned"] == len(v) and S.visit(t, 64)["found"] == D.K


@needs_mnist
def test_ivf_index():
    s = S.ivf_index().validate()
    m = s.metadata
    assert (m["lists"], m["stored"], m["smallest"], m["biggest"]) == (64, 60_000, 89, 4_761)
    assert sum(m["sizes"]) == 60_000 and m["sizes"][m["biggest_list"]] == 4_761
    assert (m["query"], m["query_label"]) == (1, 2)
    steps = m["steps"]
    assert [x["scanned"] for x in steps] == [1_923, 3_874, 5_119]
    assert [x["found"] for x in steps] == [7, 8, 10]
    assert [x["share_text"] for x in steps] == ["3.2%", "6.5%", "8.5%"]
    assert steps[-1]["near"] == m["order"][:4]
    near = np.asarray(m["near"])
    assert np.allclose(near, D.ivf()["centres"] @ D.query(1, 0), atol=1e-6)
    assert m["order"] == [int(j) for j in np.argsort(-near)[:4]]
    assert m["nprobe"] == [1, 2, 4, 8, 16, 32, 64]
    assert m["recall_text"][:4] == ["80.9%", "95.7%", "99.5%", "99.9%"]
    assert m["scanned_text"] == ["2.8%", "5.1%", "8.1%", "11.9%", "22.0%", "46.6%", "100.0%"]
    assert (m["marked_recall_text"], m["marked_scanned_text"]) == ("99.5%", "8.1%")
    # more lists: never less found, always more scanned
    assert np.all(np.diff(m["scanned"]) > 0) and m["scanned"][-1] == 1.0
    assert np.all(np.diff(m["recall"][:4]) > 0)
    assert any("not a time" in x or "neither count is a time" in x for x in m["limitations"])


@needs_mnist
def test_chance_is_the_label_frequencies():
    y, yt = course.data()[1], course.data()[3]
    pick = np.random.default_rng(0).integers(0, len(y), (200_000,))
    q = np.random.default_rng(1).integers(0, len(yt), (200_000,))
    assert S.chance() == pytest.approx(float((y[pick] == yt[q]).mean()), abs=0.003)
    assert S.chance() == pytest.approx(0.1003, abs=1e-4)


@needs_mnist
def test_versions():
    s = S.versions().validate()
    m = s.metadata
    assert (m["query"], m["label"], m["dims"]) == (0, 7, 32)
    q, q2 = np.asarray(m["q"]), np.asarray(m["q2"])
    assert np.isclose(np.linalg.norm(q), 1, atol=1e-5) and np.isclose(np.linalg.norm(q2), 1,
                                                                      atol=1e-5)
    assert m["dot_text"] == "0.12" and m["cross_text"] == "0.08" and m["self_dot_text"] == "1.00"
    assert m["mixed"]["labels"] == [3, 3, 3, 3, 3]
    assert m["again"]["labels"] == [7] * 5 and m["first"]["labels"] == [7] * 5
    # the found digits are the stored digits with those ids
    x = course.data()[0]
    for f in (m["mixed"], m["again"]):
        for i, img in zip(f["ids"], f["images"], strict=True):
            assert np.array_equal(np.asarray(img, np.float32), x[i].reshape(28, 28))
    assert m["mixed"]["ids"] == [int(i) for i in D.search(0, 5, seed=1, store_seed=0)["ids"]]
    assert m["rate_text"] == dict(same="96.4%", mixed="7.6%", chance="10.0%",
                                  reembedded="96.6%")
    assert m["right"] == 759 and m["right_text"] == "759" and m["queries"] == 10_000
    assert m["rate"]["mixed"] < m["rate"]["chance"] < m["rate"]["same"]
    assert m["mixed_top10_text"] == "9.8%"
    assert any("7s" in x for x in m["limitations"])
