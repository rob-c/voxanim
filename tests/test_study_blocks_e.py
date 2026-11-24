"""The numbers behind animations 197-199 (voxanim.study_blocks_e): softmax and the loss on
the trained model's logits, every block's shapes and weight count worked out from its
settings, and the data pipeline's ToTensor, Normalize and batches."""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_e as E
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_softmax_is_positive_and_adds_up_to_one():
    p = E.softmax([2.0, 1.0, -3.0])
    assert (p > 0).all() and math.isclose(p.sum(), 1.0)
    assert np.allclose(E.softmax([1000.0, 1000.0]), [0.5, 0.5])        # no overflow


def test_the_double_softmax_floor():
    """A one-hot output, softmaxed again, gives the right digit e / (e + 9): the loss can
    never be below ln(1 + 9/e)."""
    floor = E.double_softmax_floor()
    assert math.isclose(floor, -math.log(E.softmax(np.eye(10)[3])[3]))
    assert math.isclose(floor, math.log(1 + 9 / math.e)) and round(floor, 2) == 1.46
    for p in (np.full(10, 0.1), E.softmax(np.arange(10.0))):
        assert -np.log(E.softmax(p)).max() > floor - 1e-12


def test_the_blocks_settings_are_read_from_the_code():
    assert E.block_call("model", "Linear(784, 64)")[1:] == ("Linear", [784, 64], {})
    assert E.block_call("conv", "Conv2d")[:3] == ("nn.Conv2d(1, 4, 3)", "Conv2d", [1, 4, 3])
    text, name, args, kw = E.block_call("attention", "MultiheadAttention")
    assert (name, args, kw) == ("MultiheadAttention", [16, 2], {"batch_first": True})
    assert text == "nn.MultiheadAttention(16, 2, batch_first=True)"       # two lines joined
    assert E.block_call("softmax", ".softmax(")[:2] == ("logits.softmax(dim=1)", "softmax")
    made = E.made_up_inputs()
    assert made["x"] == [64, 1, 28, 28] and made["h"] == [64, 64] and made["y"] == [64]
    assert made["q"] == [64, 5, 16] and made["logits"] == [64, 10]


def test_shapes_and_counts_by_pytorchs_rules():
    assert E.output_shape("Conv2d", [1, 4, 3], {}, [64, 1, 28, 28]) == [64, 4, 26, 26]
    assert E.output_shape("MaxPool2d", [2], {}, [64, 4, 26, 26]) == [64, 4, 13, 13]
    assert E.output_shape("Embedding", [10, 16], {}, [64]) == [64, 16]
    assert E.output_shape("Flatten", [], {}, [64, 1, 28, 28]) == [64, 784]
    with pytest.raises(ValueError):
        E.output_shape("Linear", [784, 64], {}, [64, 1, 28, 28])
    assert E.weights("Linear", [784, 64], {}) == (50_240, "784 × 64 + 64")
    assert E.weights("Conv2d", [1, 4, 3], {})[0] == 40
    assert E.weights("BatchNorm1d", [64], {})[0] == E.weights("LayerNorm", [64], {})[0] == 128
    assert E.weights("Embedding", [10, 16], {})[0] == 160
    assert E.weights("MultiheadAttention", [16, 2], {})[0] == 1088
    assert E.weights("ReLU", [], {}) == (0, "0")


def test_the_recap_follows_the_course_and_the_parts():
    s = E.which_is_which().validate()
    m = s.metadata
    b = {x["name"]: x for x in m["blocks"]}
    assert [x["name"] for x in m["blocks"]] == [r[0] for r in E.RECAP] and m["total"] == 11
    assert b["Flatten"]["shape_out"] == b["Linear"]["shape_in"] == [64, 784]
    assert b["Linear"]["shape_out"] == b["ReLU"]["shape_in"] == [64, 64]
    assert b["MaxPool2d"]["shape_in"] == b["Conv2d"]["shape_out"] == [64, 4, 26, 26]
    assert b["MultiheadAttention"]["shape_out_text"] == "[64, 5, 16]"
    assert m["learning"] == 6 and sum(x["learns"] for x in m["blocks"]) == 6
    assert sorted(m["groups"][0] + m["groups"][1] + m["groups"][2] + m["groups"][3]) == list(
        range(11))
    assert max(len(g) for g in m["groups"]) <= 4                     # 4 blocks on screen at most
    # the Linear block's count is the course model's first layer
    P = course.init_params()
    assert b["Linear"]["weights"] == P["1.weight"].size + P["1.bias"].size


def test_the_recap_matches_torch():
    torch = pytest.importorskip("torch")
    from torch import nn
    m = E.which_is_which().metadata
    for b in m["blocks"]:
        if b["name"] == "softmax":
            out = torch.randn(*b["shape_in"]).softmax(dim=1)
            assert list(out.shape) == b["shape_out"] and b["weights"] == 0
            continue
        block = getattr(nn, b["name"])(*b["args"], **b["kwargs"])
        assert sum(p.numel() for p in block.parameters()) == b["weights"], b["name"]
        if b["name"] == "Embedding":
            x = torch.randint(0, b["args"][0], tuple(b["shape_in"]))
        else:
            x = torch.randn(*b["shape_in"])
        out = block(x, x, x)[0] if b["name"] == "MultiheadAttention" else block(x)
        assert list(out.shape) == b["shape_out"], b["name"]


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_logits_softmax_and_the_loss():
    s = E.logits_and_softmax().validate()
    m = s.metadata
    z, p = np.asarray(m["logits"]), np.asarray(m["probs"])
    y = m["label"]
    assert len(z) == 10 and math.isclose(p.sum(), 1.0) and (p > 0).all()
    assert np.allclose(p, np.exp(z) / np.exp(z).sum())
    assert m["prediction"] == y == int(np.argmax(p)) and p[y] < E.SURE
    # the digit is the first one so chosen
    P = course.params(course.EPOCH_STEPS)
    xt, yt = course.data()[2:]
    q = E.softmax(course.forward(P, xt[:m["index"] + 1])["logits"])
    right = (q.argmax(1) == yt[:m["index"] + 1]) & (q[np.arange(len(q)), yt[:m["index"] + 1]]
                                                     < E.SURE)
    assert right[-1] and not right[:-1].any()
    # CrossEntropyLoss on the logits is -ln of the right digit's share
    assert math.isclose(m["loss"], -math.log(p[y]), rel_tol=1e-12)
    assert abs(m["loss"] - m["loss_check"]) < 1e-5
    assert math.isclose(m["cumulative"][-1], 1.0)
    # softmax twice: flat, and above the floor
    t = np.asarray(m["twice"])
    assert t.max() < m["top_twice"] and m["loss_twice"] > m["floor"] > m["loss"]
    assert math.isclose(m["loss_twice"], -math.log(t[y]))
    assert m["accuracy_twice"] < m["accuracy"] and m["batch_shape"] == [64, 10]
    assert (round(m["accuracy"], 4), round(m["accuracy_twice"], 4)) == (0.9494, 0.8495)


@needs_mnist
def test_the_data_pipeline():
    s = E.data_pipeline().validate()
    m = s.metadata
    raw, t, n = (np.asarray(m[k], float) for k in ("raw", "tensor", "normalised"))
    assert np.allclose(t, raw / 255, atol=1e-7)                       # ToTensor
    assert np.allclose(n, (t - 0.1307) / 0.3081, atol=1e-5)           # Normalize
    x, _y = course.digit(0)
    r, c = m["zoom"]
    np.testing.assert_array_equal(n, x[0, r:r + E.PATCH, c:c + E.PATCH])
    i, j = m["pixel"]
    assert m["pixel_raw"] == raw[i, j] and 0 < m["pixel_raw"] < 255
    assert math.isclose(m["pixel_norm"], n[i, j], abs_tol=1e-6)
    assert round(m["mean"], 4) == m["script_mean"] and round(m["std"], 4) == m["script_std"]
    assert m["pixels"] == 60_000 * 784 and sum(m["hist"]) == m["pixels"]
    assert (m["steps"], m["last_batch"], m["full_batches"]) == (938, 32, 937)
    assert m["full_batches"] * 64 + m["last_batch"] == m["train"] == 60_000
    assert m["epoch_sum"] == "937 × 64 + 32 = 60,000"
    assert m["first"] != m["again"] and m["first"] == [int(v) for v in course.order(0)[:8]]
    assert m["belt"] == [0, 1, 2, 937] and m["belt_sizes"] == [64, 64, 64, 32]
