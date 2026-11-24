"""The numbers behind animations 188-190 (voxanim.study_blocks_b): one output of nn.Linear
added up by hand, what nn.ReLU sets to 0, and the stack's shapes, print(model) and counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_b as S
from voxanim import study_course as course

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
def test_print_model_is_wrapped_without_losing_anything():
    pieces = S.wrap_printed()
    assert S.unwrap(pieces) == course.PRINTED_MODEL
    assert all(len(s) <= S.WRAP for s, _ in pieces)
    assert pieces[0][0] == "Sequential(" and pieces[-1][0] == ")"
    # every block's lines, in order, each block once
    blocks = [k for _, k in pieces if k >= 0]
    assert sorted(set(blocks)) == list(range(6)) and blocks == sorted(blocks)
    toy = S.wrap_printed("(0): a, b, c", width=9, indent=2)
    assert toy == [("(0): a,", 0), ("  b, c", 0)]
    assert S.unwrap(toy, indent=2) == "(0): a, b, c"


def test_the_link_back_is_the_counting_animation():
    titles = catalogue_titles()
    assert titles[183] == S.COUNTED
    assert (titles[188], titles[189], titles[190]) == (
        "nn.Linear: a grid of weights", "nn.ReLU: negatives become 0",
        "A model is a stack of blocks")


def test_the_model_lines_are_the_scripts():
    lines, at = S.model_lines()
    assert [lines[i].strip() for i in at] == [
        "nn.Flatten(),", "nn.Linear(784, 64),", "nn.ReLU(),", "nn.Linear(64, 32),",
        "nn.ReLU(),", "nn.Linear(32, 10),"]


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_one_output_is_the_products_added_up_plus_the_bias():
    s = S.a_linear_layer().validate()
    m = s.metadata
    P = course.params(course.EPOCH_STEPS)
    x = course.forward(P, course.first_batch()["x"])["x"][0]
    j = m["output"]
    W, b = P["1.weight"].astype(np.float64), P["1.bias"].astype(np.float64)
    np.testing.assert_allclose(m["added"], (W[j] * x).sum(), rtol=1e-6)
    np.testing.assert_allclose(m["added"] + m["bias"], m["out"], atol=1e-4)
    np.testing.assert_allclose(np.asarray(s.panels[3].data["values"]), x @ W.T + b, atol=1e-4)
    assert m["out"] == max(s.panels[3].data["values"])            # the biggest output
    run = np.asarray(s.panels[2].data["ys"][0])
    assert len(run) == 784 and np.isclose(run[-1], m["added"])
    # the numbers on screen add up as shown, to the places shown
    assert round(m["added"], 2) + round(m["bias"], 2) == round(m["out"], 2)
    for p in m["pairs"]:
        assert np.isclose(p["x"] * p["w"], p["product"], rtol=1e-5)
        assert x[p["at"]] == np.float32(p["x"]) and p["at"] == 28 * p["row"] + p["col"]
        assert abs(round(p["x"], 2) * round(p["w"], 3) - round(p["product"], 2)) < 0.005
    products = sorted(m["pairs"], key=lambda p: -p["product"])
    assert products == m["pairs"]
    assert (m["weights"], m["biases"], m["numbers"]) == (50_176, 64, 50_240)
    assert m["batch_in"] == [64, 784] and m["batch_out"] == [64, 64]
    assert m["label"] == 0 and m["model"][m["linear_line"]].strip() == "nn.Linear(784, 64),"


@needs_mnist
def test_relu_sets_the_negative_numbers_to_0():
    s = S.relu().validate()
    m = s.metadata
    z = np.asarray(s.panels[0].data["values"])
    h = np.asarray(s.panels[1].data["values"])
    np.testing.assert_array_equal(h, np.maximum(z, 0))
    assert m["zeroed"] + m["kept"] == 64 and m["zeroed"] == int((z <= 0).sum())
    assert m["after"] == [max(v, 0.0) for v in m["before"]]
    assert m["window"][0] == m["output"] and len(m["window"]) == S.WINDOW
    assert any(v < 0 for v in m["before"]) and any(v > 0 for v in m["before"])
    lin = S.a_linear_layer().metadata
    assert m["output"] == lin["output"] and np.isclose(m["before"][0], lin["out"])
    batch = np.asarray(s.panels[2].data["values"])
    zero = np.asarray(s.panels[2].data["zero"], bool)
    assert batch.shape == (64, 64) and (batch[zero] == 0).all() and (batch[~zero] > 0).all()
    assert m["batch_zeroed"] == int(zero.sum()) == round(m["batch_share"] * 4096)
    assert m["batch_numbers"] == 4096 and m["train_digits"] == 60_000
    # measured: about half of the first hidden numbers, fewer at the second ReLU
    assert 0.4 < m["batch_share"] < 0.6 and 0.4 < m["train_share"] < 0.6
    assert m["second_train_share"] < m["train_share"] and m["weights"] == 0


@needs_mnist
def test_the_stack_counts_shapes_and_prints():
    s = S.a_model_is_a_stack().validate()
    m = s.metadata
    assert m["names"] == ["nn.Flatten()", "nn.Linear(784, 64)", "nn.ReLU()",
                          "nn.Linear(64, 32)", "nn.ReLU()", "nn.Linear(32, 10)"]
    assert m["counts"] == [0, 50_240, 0, 2_080, 0, 330]
    assert m["total"] == 52_650 == course.count() == S.COUNTED_TOTAL
    assert (m["total_weights"], m["total_biases"]) == (52_544, 106)
    assert m["shapes"] == [[64, 784], [64, 64], [64, 64], [64, 32], [64, 32], [64, 10]]
    assert m["input_text"] == "[64, 1, 28, 28]" and m["sizes"] == [784, 64, 32, 10]
    assert m["printed"] == course.PRINTED_MODEL
    assert "\n".join(m["printed_lines"]).replace(",\n      ", ", ") == course.PRINTED_MODEL
    assert m["count_code"][-1] == "print(sum(p.numel() for p in params))"
    assert m["printed_sum"] == "52650"
    assert np.asarray(s.panels[3].data["values"]).shape == (64, 10)
