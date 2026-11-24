"""The numbers behind animations 185-187 (voxanim.study_blocks_a): a tensor's shape and
values, the first batch, and where nn.Flatten puts each pixel."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_a as A
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_flattening_is_row_after_row():
    assert A.flat_index(0, 0) == 0 and A.flat_index(0, 27) == 27
    assert A.flat_index(1, 0) == 28 and A.flat_index(27, 27) == 783
    a = np.arange(784).reshape(28, 28)
    for r, c in ((14, 15), (3, 9), (27, 0)):
        assert a.reshape(-1)[A.flat_index(r, c)] == a[r, c]


def test_the_window_with_the_most_different_numbers():
    img = np.zeros((6, 6))
    img[3:6, 2:5] = np.arange(9).reshape(3, 3)
    assert A.zoom_window(img, 3) == (3, 2)


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_a_tensor_is_784_numbers_shaped_1_28_28():
    s = A.a_tensor().validate()
    m = s.metadata
    assert m["shape"] == [1, 28, 28] and m["numbers"] == 784 == 1 * 28 * 28
    assert m["label"] == 5 and m["printed"] == "torch.Size([1, 28, 28]) 5"
    assert m["inked"] + m["blank"] == 784
    x, _y = course.digit(0)
    r, c = m["zoom"]
    np.testing.assert_array_equal(np.asarray(m["patch"]), x[0, r:r + A.ZOOM, c:c + A.ZOOM])
    assert len(np.unique(np.round(m["patch"], 4))) >= 6          # paper, ink and between
    assert np.isclose(m["low"], m["paper"]) and m["high"] > 2.8
    assert m["code"] == ["x, y = train_set[0]", "print(x.shape, y)"]


@needs_mnist
def test_a_batch_is_64_digits_and_64_labels():
    s = A.a_batch().validate()
    m = s.metadata
    assert m["shape"] == [64, 1, 28, 28] and m["label_shape"] == [64]
    assert m["numbers"] == 64 * 784 == 50_176
    assert sum(m["counts"]) == 64 and len(m["counts"]) == 10
    _x, y = course.data()[:2]
    assert m["labels"] == [int(v) for v in y[m["indices"]]]
    assert m["counts"][m["most"]] == max(m["counts"]) and min(m["counts"]) > 0
    assert m["printed"] == "torch.Size([64, 1, 28, 28]) torch.Size([64])"
    assert (m["steps"], m["last_batch"]) == (938, 32)


@needs_mnist
def test_flatten_keeps_every_number_in_row_order():
    s = A.flatten().validate()
    m = s.metadata
    img = np.asarray(s.panels[0].data["values"][0])
    strip = np.asarray(s.panels[1].data["values"][0])
    np.testing.assert_array_equal(strip, img.reshape(-1))
    r, c = m["pixel"]
    assert m["place"] == r * 28 + c and strip[m["place"]] == img[r, c] == m["value"]
    assert m["value"] > 2.0                                     # an inked pixel
    assert m["batch_in"] == [64, 1, 28, 28] and m["batch_out"] == [64, 784]
    flat = np.asarray(s.panels[2].data["values"])
    np.testing.assert_allclose(flat, course.first_batch()["x"].reshape(64, -1))
    assert m["weights"] == 0 and m["model"][m["flatten_line"]].strip() == "nn.Flatten(),"
