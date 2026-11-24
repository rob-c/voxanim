"""The counts behind animation 183 (voxanim.study_count_nets): the formulas on layers with
known answers, both models' totals and running totals, each PyTorch line's arguments
against its shape, and every count against the parameter arrays of the page's own models."""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import mnist
from voxanim import study_count_nets as S


def args(code):
    """The numbers a PyTorch line is called with, products worked out: ``nn.Linear(4 * 13 *
    13, 10)`` -> ``[676, 10]``; ``kernel_size=3`` -> 3."""
    inside = code[code.index("(") + 1:code.rindex(")")]
    out = []
    for a in filter(None, (x.strip() for x in inside.split(","))):
        a = a.split("=")[-1]
        assert re.fullmatch(r"[\d *]+", a), code
        out.append(int(np.prod([int(v) for v in a.split("*")])))
    return out


# ---------------------------------------------------------------------- no data needed
def test_a_dense_layer_is_inputs_times_outputs_plus_one_bias_each():
    assert S.linear(1, 1)["total"] == 2
    t = S.linear(3, 2)
    assert (t["weights"], t["biases"], t["total"], t["code"]) == (6, 2, 8, "nn.Linear(3, 2)")
    a = S.linear(784, 64)
    assert (a["weights"], a["biases"], a["total"]) == (50_176, 64, 50_240)


def test_a_convolution_does_not_depend_on_the_picture():
    c = S.conv(1, 4, 3)
    assert (c["weights"], c["biases"], c["total"]) == (36, 4, 40)
    assert c["code"] == "nn.Conv2d(1, 4, kernel_size=3)" and c["shape"] == [4, 1, 3, 3]
    assert S.conv(3, 8, 5)["total"] == 8 * 3 * 5 * 5 + 8
    small, big = S.cnn_layers(side=28), S.cnn_layers(side=84)
    assert small[0] == big[0]                                  # the filters: the same 40
    assert small[-1]["inputs"] == 676 and big[-1]["inputs"] == 4 * 41 * 41   # the dense grows
    assert S.pooled_side(28) == 13 and S.pooled_side(84) == 41


def test_the_dnn_counts_and_running_totals():
    d = S.dnn_layers()
    assert [x["code"] for x in d] == ["nn.Flatten()", "nn.Linear(784, 64)", "nn.ReLU()",
                                      "nn.Linear(64, 32)", "nn.ReLU()", "nn.Linear(32, 10)"]
    assert [x["total"] for x in d] == [0, 50_240, 0, 2_080, 0, 330]
    assert [x["running"] for x in d] == [0, 50_240, 50_240, 52_320, 52_320, 52_650]
    assert sum(x["total"] for x in d) == 52_650 == d[-1]["running"]
    assert S.biggest(d) == 1


def test_the_cnn_counts_and_running_totals():
    c = S.cnn_layers()
    assert [x["code"] for x in c] == ["nn.Conv2d(1, 4, kernel_size=3)", "nn.ReLU()",
                                      "nn.MaxPool2d(2)", "nn.Flatten()",
                                      "nn.Linear(4 * 13 * 13, 10)"]
    assert [x["total"] for x in c] == [40, 0, 0, 0, 6_770]
    assert [x["running"] for x in c] == [40, 40, 40, 40, 6_810]
    assert c[-1]["weights"] == 6_760 and S.biggest(c) == 4


def test_each_code_line_matches_its_shape():
    for layer in S.dnn_layers() + S.cnn_layers() + [S.linear(3, 2), S.linear(1, 1)]:
        a = args(layer["code"])
        if layer["kind"] == "Linear":
            assert a == layer["shape"] == [layer["inputs"], layer["outputs"]]
            assert layer["weights"] == a[0] * a[1] and layer["biases"] == a[1]
        elif layer["kind"] == "Conv2d":
            c_in, c_out, k = a
            assert layer["shape"] == [c_out, c_in, k, k]
            assert layer["weights"] == c_out * c_in * k * k and layer["biases"] == c_out
        else:
            assert layer["total"] == 0 and layer["shape"] == []
            assert a == ([S.POOL] if layer["kind"] == "MaxPool2d" else [])


def test_the_constants_are_the_page_models():
    assert S.SIZES == (784, 64, 32, 10) == tuple(mnist.SIZES)
    assert (S.FILTERS, S.KERNEL, S.POOL) == (4, 3, 2)
    assert S.cnn_layers()[-1]["inputs"] == mnist.CNN_DENSE_IN


def test_the_check_catches_a_wrong_count():
    layers = [S.linear(3, 2)]
    good = [[np.zeros((3, 2)), np.zeros(2)]]
    assert S.check(layers, good, "toy") == [[6, 2]]
    assert S.numel(good) == 8
    with pytest.raises(ValueError):
        S.check(layers, [[np.zeros((3, 2)), np.zeros(3)]], "toy")


def test_the_same_filters_read_any_picture():
    rng = np.random.default_rng(0)
    Wc, bc = rng.normal(size=(3, 3, 4)), rng.normal(size=4)
    for side in (28, 84):
        img = rng.random((side, side))
        maps = S.filter_maps(Wc, bc, img)
        assert maps.shape == (side - 2, side - 2, 4)
        r, c, f = side // 2, side // 3, 2
        assert np.isclose(maps[r, c, f], (img[r:r + 3, c:c + 3] * Wc[..., f]).sum() + bc[f])
    sizes = S.any_size(Wc, bc)
    assert [s["parameters"] for s in sizes] == [40, 40]
    assert [s["map_side"] for s in sizes] == [26, 82]
    assert [s["dense_inputs"] for s in sizes] == [676, 4 * 41 * 41]


# ---------------------------------------------------------------------- the page's models
@needs_mnist
def test_every_count_is_the_size_of_a_real_parameter_array():
    trained, _ = mnist.train()
    fresh = mnist._init(np.random.default_rng(1))
    for arrays in (trained, fresh):
        lin = [x for x in S.dnn_layers() if x["total"]]
        for (W, b), layer in zip(arrays, lin):
            assert W.shape == tuple(layer["shape"])
            assert (W.size, b.size) == (layer["weights"], layer["biases"])
        assert S.numel(arrays) == 52_650
    (Wc, bc), (Wd, bd) = mnist.cnn_train()[0]
    conv, dense = S.cnn_layers()[0], S.cnn_layers()[-1]
    assert (Wc.size, bc.size) == (conv["weights"], conv["biases"]) == (36, 4)
    assert Wd.shape == (dense["inputs"], dense["outputs"]) and bd.size == dense["biases"]
    assert S.numel(mnist.cnn_train()[0]) == 6_810


@needs_mnist
def test_the_study():
    s = S.counting_weights().validate()
    m = s.metadata
    assert len(s.panels) == 4
    assert m["dnn"]["total"] == 52_650 and m["cnn"]["total"] == 6_810
    assert m["dnn"]["running"] == [50_240, 52_320, 52_650]
    assert m["cnn"]["running"] == [40, 6_810]
    assert m["dnn"]["array_sizes"] == [[50_176, 64], [2_048, 32], [320, 10]]
    assert m["cnn"]["array_sizes"] == [[36, 4], [6_760, 10]]
    assert m["toy"]["total"] == 8 and m["one"]["total"] == 2
    assert m["code"] == "sum(p.numel() for p in model.parameters())"
    assert m["ratio"] == pytest.approx(52_650 / 6_810)
    assert m["shares"]["dnn"] == pytest.approx(50_240 / 52_650)
    assert m["shares"]["cnn"] == pytest.approx(6_770 / 6_810)
    assert [k["parameters"] for k in m["cnn"]["any_size"]] == [40, 40]
    dnn, cnn, filters, totals = (p.data for p in s.panels)
    assert dnn["values"].sum() == 52_650 and dnn["running"][-1] == 52_650
    assert dnn["labels"] == [x["code"] for x in m["dnn"]["layers"]]
    assert cnn["values"].sum() == 6_810 and cnn["labels"][0] == "nn.Conv2d(1, 4, kernel_size=3)"
    Wc = mnist.cnn_train()[0][0][0]
    assert filters["values"].shape == (4, 3, 3)
    np.testing.assert_array_equal(filters["values"][1], Wc[..., 1])
    np.testing.assert_array_equal(totals["values"], [52_650, 6_810])
