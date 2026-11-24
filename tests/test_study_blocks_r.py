"""The numbers behind animations 234-235 (voxanim.study_blocks_r): the pairs, the [64, 64]
grid and its diagonal as training goes, the store, one search, and the quality of search
over every test digit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_r as R
from voxanim import study_course as course
from voxanim import study_vectordb as vdb

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue_titles():
    return {item["number"]: item["title"] for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_are_the_catalogues():
    titles = catalogue_titles()
    assert (titles[234], titles[235]) == ("Training vectors for search", "Store, then search")
    assert titles[236] == "An index: look in fewer places"


def test_sorting_by_digit_keeps_the_pairs_on_the_diagonal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 12)
    g = rng.uniform(-1, 1, (12, 12))
    s, order = R.by_digit(g, y)
    assert (np.diff(y[order]) >= 0).all()
    assert np.allclose(np.diag(s), np.diag(g)[order])          # each row keeps its pair
    assert np.allclose(np.sort(s, None), np.sort(g, None))
    m = R.split_means(g, y)
    diag = np.eye(12, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~diag
    assert np.isclose(m["pos"], g[diag].mean()) and np.isclose(m["neg"], g[~diag].mean())
    # the other cells split into the same digit and other digits, weighted by their counts
    n_same, n_other = same.sum(), (~diag & ~same).sum()
    assert np.isclose(m["neg"], (m["same"] * n_same + m["other"] * n_other) / (n_same + n_other))


def test_the_printed_line_is_torchs_and_fits_the_slot():
    text = R.printed([38743, 53843, 24372, 50115, 1833], [7, 7, 7, 7, 7])
    assert text == "tensor([38743, 53843, 24372, 50115,  1833]) tensor([7, 7, 7, 7, 7])"
    lines = R.wrapped(text).split("\n")
    assert len(lines) == 2 and all(len("→ " + s) <= 41 for s in lines)
    assert "".join(s.strip() + " " for s in lines).split() == text.replace(",  ", ", ").split()
    torch = pytest.importorskip("torch", reason="torch is not installed")
    ids, labels = torch.tensor([38743, 53843, 24372, 50115, 1833]), torch.tensor([7] * 5)
    assert text == "%s %s" % (ids, labels)


def test_formats():
    assert R.two(0.927) == "0.93" and R.two(-0.004) == "0.00" and R.pct(0.96009) == "96.0%"


# ---------------------------------------------------------------------- 234
@needs_mnist
def test_training():
    s = R.training().validate()
    m = s.metadata
    assert len(s.panels) == 4 and s.title == "Training vectors for search"
    assert m["parameters"] == 104_608 == 784 * 128 + 128 + 128 * 32 + 32
    # the pair: two drawings of one digit, two different training digits
    b = vdb.batch(0)
    y = course.data()[1]
    assert m["pair_label"] == int(b["y"][0]) == int(y[m["pair_i"]]) == int(y[m["pair_j"]]) == 5
    assert m["pair_i"] != m["pair_j"]
    assert not np.allclose(m["x"], m["p"])
    # 32 numbers each, length 1; their dot product is the grid's corner
    assert len(m["a"]) == len(m["b"]) == 32
    assert np.isclose(m["a_norm"], 1.0, atol=1e-5) and np.isclose(m["b_norm"], 1.0, atol=1e-5)
    snap = m["snap"]
    assert np.isclose(m["dot"], snap[0]["grid"][0][0], atol=1e-5) and m["dot_text"] == "0.61"
    # the grids, their diagonals and the loss: the measured headline
    assert [(snap[s]["pos_text"], snap[s]["neg_text"], snap[s]["loss_text"]) for s in R.SNAPS] \
        == [("0.59", "0.49", "4.19"), ("0.85", "0.43", "2.58"), ("0.93", "0.36", "2.13")]
    assert (m["first_loss_text"], m["last_100_text"], m["chance_text"]) == ("4.19", "2.20",
                                                                            "4.16")
    for st in R.SNAPS:
        g = np.asarray(snap[st]["grid"])
        assert g.shape == (64, 64) and g.max() <= 1.0 + 1e-5 and g.min() >= -1.0 - 1e-5
        assert sum(snap[st]["counts"]) == 64
    # the same digit off the diagonal: almost the pairs' score at the end
    end = snap[937]
    assert (end["same_text"], end["other_text"]) == ("0.92", "0.29")
    assert end["same"] > 3 * end["other"] and end["pos"] - end["same"] < 0.01
    s937 = np.asarray(end["sorted"])
    assert np.allclose(np.diag(s937), np.diag(end["grid"])[end["order"]])
    assert any("false negative" in t for t in m["limitations"])


# ---------------------------------------------------------------------- 235
@needs_mnist
def test_store_and_search():
    s = R.store_and_search().validate()
    m = s.metadata
    assert len(s.panels) == 4 and s.title == "Store, then search"
    assert m["store_shape_text"] == "[60000, 32]" and m["stored_text"] == "60,000"
    assert m["rows"] == [0, 1, 2, 3, 4, 59999]
    assert m["row_labels"] == [int(v) for v in course.data()[1][m["rows"]]] == [5, 0, 4, 1, 9, 8]
    st = vdb.store(0)
    assert np.allclose(m["row_vectors"], st["vectors"][m["rows"]])
    assert np.allclose(np.linalg.norm(st["vectors"], axis=1), 1.0, atol=1e-5)
    # the query: the script's XT[:1], a 7
    assert m["query"] == 0 and m["query_label"] == 7 == int(course.data()[3][0])
    assert np.isclose(m["q_norm"], 1.0, atol=1e-5)
    scores = st["vectors"] @ np.asarray(m["q"], np.float32)
    assert np.allclose(m["row_scores"], scores[m["rows"]], atol=1e-6)
    assert m["row_score_texts"] == ["0.22", "0.23", "0.58", "0.36", "0.38", "0.10"]
    # the top 5: the 5 highest of the 60,000 scores, all 7s
    assert m["top"] == [38743, 53843, 24372, 50115, 1833]
    assert sorted(scores, reverse=True)[:5] == pytest.approx(m["top_scores"])
    assert m["top_labels"] == [7] * 5 and m["top_score_texts"] == ["0.999"] * 5
    assert np.allclose(m["top_images"][0], course.data()[0][38743].reshape(28, 28))
    assert m["printed"] == "tensor([38743, 53843, 24372, 50115,  1833]) tensor([7, 7, 7, 7, 7])"
    assert m["same_label_mean"] > 0.9 > 0.3 > m["other_label_mean"]
    # quality over every test digit: 32 numbers about as good as 784 pixels
    q = m["quality_texts"]
    assert (q["trained"]["top1"], q["trained"]["top10"]) == ("96.4%", "96.0%")
    assert (q["pixels"]["top1"], q["pixels"]["top10"]) == ("97.0%", "95.4%")
    assert (q["untrained"]["top1"], q["untrained"]["top10"]) == ("87.0%", "81.9%")
    assert m["quality"]["pixels"]["top1"] > m["quality"]["trained"]["top1"]   # honest: top 1
    assert m["smaller"] == 24.5
