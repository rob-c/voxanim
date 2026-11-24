"""The numbers behind animations 215-217 (voxanim.study_blocks_k): the CNN's blocks, shapes,
print(model) and counts; the side walk to 16 x 5 x 5 = 400 and torch's error when it is
784; and train_set[0] through the trained CNN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_k as S
from voxanim import study_course as course
from voxanim import study_course_cnn as cnn
from voxanim.study_blocks_b import unwrap

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue_titles():
    found = {}

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("number"), int) and "title" in o:
                found[o["number"]] = o["title"]
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(json.loads(CATALOGUE.read_text()))
    return found


# ---------------------------------------------------------------------- no data needed
def test_the_titles_are_the_catalogues():
    titles = catalogue_titles()
    assert (titles[215], titles[216], titles[217]) == (
        "The CNN, block by block", "Where the 400 comes from", "One digit through the CNN")
    assert titles[214] == "Common mistakes" and titles[218] == "A shared stamp's gradient"


def test_print_model_is_wrapped_without_losing_anything():
    pieces = S.wrap_printed()
    assert unwrap(pieces) == cnn.PRINTED_MODEL
    assert all(len(s) <= S.WRAP for s, _ in pieces)
    assert pieces[0][0] == "Sequential(" and pieces[-1][0] == ")"
    blocks = [b for _, b in pieces if b >= 0]
    assert sorted(set(blocks)) == list(range(8)) and blocks == sorted(blocks)
    # never broken inside a setting: (3, 3) and (1, 1) stay whole
    assert "      kernel_size=(3, 3)," in [s for s, _ in pieces]
    assert not any(s.rstrip().endswith("(3,") for s, _ in pieces)
    first, second = S.pages(pieces)
    assert first + second == pieces and len(first) <= 11 and len(second) <= 11
    assert {b for _, b in first} == {-1, 0, 1, 2, 3} and {b for _, b in second} == {-1, 4, 5,
                                                                                    6, 7}


def test_the_model_lines_are_the_scripts():
    lines, names, at = S.model_blocks()
    assert names == ["nn.Conv2d(1, 8, 3)", "nn.ReLU()", "nn.MaxPool2d(2)",
                     "nn.Conv2d(8, 16, 3)", "nn.ReLU()", "nn.MaxPool2d(2)", "nn.Flatten()",
                     "nn.Linear(16 * 5 * 5, 10)"]
    assert at == [1, 1, 2, 3, 3, 4, 5, 6] and lines[0] == "model = nn.Sequential("
    assert tuple(S.kind(n) for n in names) == cnn.BLOCKS
    w = np.arange(2 * 3 * 3 * 3).reshape(2, 3, 3, 3)
    pic = S.stamps(w)
    assert pic.shape == (6, 9)
    np.testing.assert_array_equal(pic[3:6, 6:9], w[1, 2])         # map 1's stamp on input 2


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_the_model_counts_shapes_and_prints():
    s = S.the_model().validate()
    m = s.metadata
    assert m["counts"] == [80, 0, 0, 1_168, 0, 0, 0, 4_010]
    assert m["total"] == 5_258 == cnn.count() and m["mlp_total"] == 52_650 == course.count()
    assert (m["total_weights"], m["total_biases"]) == (5_224, 34)
    assert m["sums"] == ["8 × 1 × 3 × 3 + 8 = 80", "16 × 8 × 3 × 3 + 16 = 1,168",
                         "10 × 400 + 10 = 4,010"]
    assert m["input_text"] == "[64, 1, 28, 28]"
    assert m["shape_texts"] == ["[64, 8, 26, 26]", "[64, 8, 26, 26]", "[64, 8, 13, 13]",
                                "[64, 16, 11, 11]", "[64, 16, 11, 11]", "[64, 16, 5, 5]",
                                "[64, 400]", "[64, 10]"]
    assert m["per_digit"] == [784, 5_408, 5_408, 1_352, 1_936, 1_936, 400, 400, 10]
    assert m["side_walk"] == [28, 26, 13, 11, 5]
    assert "\n".join(m["printed_lines"]).replace(",\n      ", ", ") == cnn.PRINTED_MODEL
    assert m["pages"][0] + m["pages"][1] == m["printed_lines"]
    assert m["page_blocks"] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert np.asarray(s.panels[3].data["values"]).shape == (64, 10)
    P = cnn.params(cnn.EPOCH_STEPS)
    np.testing.assert_array_equal(s.panels[2].data["values"], S.stamps(P["0.weight"]))


@needs_mnist
def test_the_side_walk_makes_400_and_784_is_an_error():
    s = S.the_shapes().validate()
    m = s.metadata
    assert m["side_walk"] == [28, 26, 13, 11, 5] == list(cnn.side_walk())
    assert [st["sum"] for st in m["steps"]] == ["28 − 2 = 26", "26 ÷ 2 = 13", "13 − 2 = 11",
                                                "11 ÷ 2 = 5.5 → 5"]
    assert [st["op"] for st in m["steps"]] == ["−2", "÷2", "−2", "÷2"]
    assert m["in_features"] == 400 == cnn.IN_FEATURES == 16 * 5 * 5
    assert m["in_features_sum"] == "16 × 5 × 5 = 400" and m["dropped"] == 21
    assert m["linear_code"].strip() == "nn.Linear(16 * 5 * 5, 10),"
    assert m["wrong_code"].strip() == "nn.Linear(784, 10),"
    assert (m["rows"], m["got"], m["wants"], m["outs"]) == (64, 400, 784, 10)
    assert " ".join(m["error_lines"]) == m["error"] == cnn.ERRORS["cnn-mistake-in-features"]
    assert all(len(s) <= 39 for s in m["error_lines"])
    assert m["transitions"] == ["[64, 1, 28, 28] → [64, 8, 26, 26]",
                                "[64, 8, 26, 26] → [64, 8, 13, 13]",
                                "[64, 8, 13, 13] → [64, 16, 11, 11]",
                                "[64, 16, 11, 11] → [64, 16, 5, 5]",
                                "[64, 16, 5, 5] → [64, 400]", "[64, 400] → [64, 10]"]
    maps = s.panels[1].data
    assert [np.shape(maps[k]) for k in ("values", "conv1", "pool1", "conv2", "pool2")] == [
        (28, 28), (26, 26), (13, 13), (11, 11), (5, 5)]
    # the squares are the same digit's real maps: pooled is MaxPool2d of the map before
    a = cnn.activations(cnn.EPOCH_STEPS, S.DIGIT)
    c1, c2 = m["maps_used"]
    np.testing.assert_array_equal(maps["pool1"], a["p1"][0, c1])
    np.testing.assert_array_equal(maps["pool2"], a["p2"][0, c2])
    h = maps["conv2"]
    np.testing.assert_array_equal(maps["pool2"], h[:10, :10].reshape(5, 2, 5, 2).max((1, 3)))
    sixteen = np.asarray(s.panels[2].data["values"])
    assert sixteen.shape == (16, 5, 5) and sixteen.size == 400
    assert m["batch_printed"] == "torch.Size([64, 1, 28, 28])\ntorch.Size([64])"


@needs_mnist
def test_one_digit_through_the_trained_cnn():
    s = S.forward_one().validate()
    m = s.metadata
    assert (m["digit"], m["label"], m["said"]) == (0, 5, 5)
    assert m["second"] == 3 and round(m["score"], 2) == 10.16
    assert round(m["second_score"], 2) == 8.22 and 0.8 < m["prob"] < 0.9
    scores = np.asarray(s.panels[3].data["values"])
    assert scores.argmax() == m["label"] and len(scores) == 10
    P = cnn.params(cnn.EPOCH_STEPS)
    flat = np.asarray(s.panels[3].data["flat"])
    np.testing.assert_allclose(flat @ P["7.weight"].T + P["7.bias"], scores, atol=1e-4)
    p2 = np.asarray(s.panels[2].data["pooled"])
    np.testing.assert_array_equal(p2.reshape(-1), flat)
    h1 = np.asarray(s.panels[1].data["values"])
    p1 = np.asarray(s.panels[1].data["pooled"])
    assert h1.shape == (8, 26, 26) and p1.shape == (8, 13, 13) and (h1 >= 0).all()
    np.testing.assert_array_equal(p1, h1.reshape(8, 13, 2, 13, 2).max((2, 4)))
    assert m["shape_texts"] == {"x": "[1, 1, 28, 28]", "h1": "[1, 8, 26, 26]",
                                "p1": "[1, 8, 13, 13]", "h2": "[1, 16, 11, 11]",
                                "p2": "[1, 16, 5, 5]", "flat": "[1, 400]",
                                "logits": "[1, 10]"}
    assert m["printed"] == "torch.Size([1, 28, 28]) 5"
    x, y = course.digit(0)
    np.testing.assert_array_equal(s.panels[0].data["values"], x[0])
    assert y == 5
    assert 0.5 < m["zero_share"]["h1"] < 0.7
