"""The numbers behind animations 227-229 (voxanim.study_blocks_o): F.pad's four modes, deep
stacks of stamps with and without padding, and the course's CNN with padding=1, measured."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_o as O
from voxanim import study_padding as pad

NUMPY = {"reflect": "reflect", "replicate": "edge", "circular": "wrap"}


# ---------------------------------------------------------------------- no data needed
@pytest.mark.parametrize("mode", sorted(NUMPY))
def test_where_each_border_number_comes_from(mode):
    a = np.random.default_rng(0).normal(size=(8, 8))
    big = np.pad(a, 2, mode=NUMPY[mode])
    for r in range(12):
        for c in range(12):
            assert big[r, c] == a[O.source(mode, r), O.source(mode, c)]


def test_the_modes_in_words():
    # reflect mirrors about the edge, which it does not repeat; replicate repeats the edge;
    # circular takes the other side
    assert [O.source("reflect", i) for i in (0, 1, 2, 10, 11)] == [2, 1, 0, 6, 5]
    assert [O.source("replicate", i) for i in (0, 1, 2, 10, 11)] == [0, 0, 0, 7, 7]
    assert [O.source("circular", i) for i in (0, 1, 2, 10, 11)] == [6, 7, 0, 0, 1]
    assert O.source("constant", 0) is None and O.source("constant", 5) == 3


def test_the_sources_are_torchs():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    F = torch.nn.functional
    a = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    for mode in NUMPY:
        big = F.pad(a, [2] * 4, mode=mode)[0, 0].numpy()
        for r in range(12):
            for c in range(12):
                assert big[r, c] == a[0, 0, O.source(mode, r), O.source(mode, c)].item()
    assert (F.pad(a, [2] * 4, mode="constant")[0, 0, :2] == 0).all()


def test_the_readout():
    block = np.pad(np.arange(64, dtype=float).reshape(8, 8) / 10, 2, mode="reflect")
    assert O.readout(block, "reflect", row=2) == "0.20  0.10 | 0  0.10  0.20"
    assert O.number(0.0) == "0" and O.number(-0.4242) == "−0.42"


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_the_modes():
    s = O.modes_study().validate()
    m = s.metadata
    assert (m["label"], m["box"], m["side"], m["padded"]) == (5, [5, 12, 8, 15], 8, 12)
    assert np.array_equal(np.asarray(m["part"]), pad.part())
    assert m["modes"] == ["constant", "reflect", "replicate", "circular"]
    blocks = {k: np.asarray(v) for k, v in m["blocks"].items()}
    assert all(b.shape == (12, 12) for b in blocks.values())
    ring = np.ones((12, 12), bool)
    ring[2:10, 2:10] = False
    assert (blocks["constant"][ring] == 0).all()                 # 0, not the paper
    assert m["bg_text"] == "−0.42" and m["zero_text"] == "0"
    r, c = m["paper_tile"]
    assert blocks["constant"][r, c] == np.float32(pad.BG)
    assert m["arrows"] == {"reflect": [[[6, 3], [6, 1]], [[6, 4], [6, 0]]],
                           "replicate": [[[6, 2], [6, 1]], [[6, 2], [6, 0]]],
                           "circular": [[[6, 9], [6, 1]], [[6, 8], [6, 0]]]}
    for mode, pairs in m["arrows"].items():
        for (sr, sc), (tr, tc) in pairs:
            assert blocks[mode][sr, sc] == blocks[mode][tr, tc]
    assert m["readout"] == {"constant": "0  0 | 0.59  1.56  0.94",
                            "reflect": "0.94  1.56 | 0.59  1.56  0.94",
                            "replicate": "0.59  0.59 | 0.59  1.56  0.94",
                            "circular": "−0.28  −0.42 | 0.59  1.56  ...  −0.28  −0.42"}
    assert m["copied"] == {"reflect": ["1.56", "0.94"], "replicate": ["0.59", "0.59"],
                           "circular": ["−0.42", "−0.28"]}
    assert m["printed"]["replicate"] == "replicate torch.Size([1, 1, 12, 12])"
    assert m["part_text"] == "[1, 1, 8, 8]"


@needs_mnist
def test_the_deep_stacks():
    s = O.deep_study().validate()
    m = s.metadata
    assert m["valid"] == list(range(28, 3, -2)) and m["same"] == [28] * 13
    assert m["printed_deep"] == "torch.Size([64, 1, 4, 4])"
    assert m["printed_one"] == "padding=1: torch.Size([64, 8, 28, 28])"
    assert m["side_walk"] == [28, 28, 14, 14, 7]
    assert [x.shape[0] for x in (s.panels[2].data[k] for k in ("values", "conv1", "pool1",
                                                                 "conv2", "pool2"))] == \
        m["side_walk"]
    assert (m["in_features"], m["plain_in_features"]) == (784, 400)
    assert (m["in_features_sum"], m["plain_sum"]) == ("16 × 7 × 7 = 784", "16 × 5 × 5 = 400")
    assert len(m["plain_flat"]) == 400 and np.asarray(s.panels[3].data["values"]).size == 784
    assert [st["sum"] for st in m["steps"]] == ["28 + 2 − 2 = 28", "28 ÷ 2 = 14",
                                                "14 + 2 − 2 = 14", "14 ÷ 2 = 7"]
    assert m["shape_texts"] == ["[64, 1, 28, 28]", "[64, 8, 28, 28]", "[64, 8, 14, 14]",
                                "[64, 16, 14, 14]", "[64, 16, 7, 7]", "[64, 784]", "[64, 10]"]
    assert m["linear_code"] == "    nn.Linear(16 * 7 * 7, 10),"


@needs_mnist
def test_the_cnn_with_padding():
    s = O.in_the_cnn().validate()
    m = s.metadata
    assert (m["plain_params"], m["padded_params"]) == (5_258, 9_098)
    assert m["plain_counts"] == [80, 1_168, 4_010] and m["padded_counts"] == [80, 1_168, 7_850]
    # padding adds no numbers to the stamps: all the extra are in the last layer
    assert m["extra"] == 3_840 == m["padded_counts"][2] - m["plain_counts"][2]
    assert (m["plain_in"], m["padded_in"]) == (400, 784)
    assert (m["plain_accuracy_text"], m["padded_accuracy_text"]) == ("96.4%", "97.6%")
    assert (m["plain_missed"], m["padded_missed"]) == (364, 237)
    assert (m["printed_plain"], m["printed_padded"]) == ("0.9636", "0.9763")
    assert sum(m["per_digit_plain"]) == 364 and sum(m["per_digit_padded"]) == 237
    assert m["eval_steps"][0] == 0 and m["eval_steps"][-1] == 938
    assert m["plain_shifted_text"][-1] == "18.5%" and m["padded_shifted_text"][-1] == "16.9%"
    # moved digits: both fall alike, the padded one no better
    plain, padded = np.asarray(m["plain_shifted"]), np.asarray(m["padded_shifted"])
    assert plain[0] - plain[-1] > 0.75 and padded[0] - padded[-1] > 0.75
    assert np.abs(plain - padded).max() < 0.02
    assert padded[-1] < plain[-1]
    img = np.asarray(m["moved_images"])
    assert img.shape == (4, 28, 28) and m["moved"] == [0, 2, 4, 6]
    assert np.allclose(img[3][:, :6], m["bg"])                    # the gap is paper
    assert np.array_equal(img[3][:, 6:], img[0][:, :-6])
