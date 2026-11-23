"""The experiment of animation 179 (voxanim.study_bridge_local): its measures on cases with
known answers, the counts that must hold exactly, the CNN's gradients, and what it claims
about MNIST as relative statements (CI retrains; last digits differ)."""

from __future__ import annotations

import numpy as np
from conftest import fd_check, needs_mnist

from voxanim import study_bridge_local as B
from voxanim import study_info as I
from voxanim import study_vision as V

NAME = "dense-layer-learns-local"


# ---------------------------------------------------------------------- no data needed
def test_correlation_by_distance_on_known_pictures():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(784, 64))
    c = B.correlation_by_distance(noise)
    assert c.shape == B.DISTANCES.shape and np.abs(c).max() < 0.02    # independent: about 0
    # a smooth picture: every unit a slow ramp across columns plus one down the rows
    yy, xx = np.mgrid[0:28, 0:28]
    ramps = np.stack([np.cos(2 * np.pi * (xx + k) / 56) + np.sin(2 * np.pi * (yy - k) / 56)
                      for k in range(16)]).reshape(16, -1).T
    s = B.correlation_by_distance(ramps)
    assert s[0] > 0.95 and np.all(np.diff(s) < 0)                     # near: alike; falls
    # a checkerboard: neighbours opposite, two apart the same
    board = np.tile(((yy + xx) % 2 * 2 - 1).reshape(-1, 1), (1, 4)).astype(float)
    board += 1e-3 * rng.normal(size=board.shape)
    b = B.correlation_by_distance(board, [1, 2])
    assert b[0] < -0.99 and b[1] > 0.99


def test_pairs_cover_rows_and_columns():
    W = np.arange(784.0).reshape(784, 1)
    a, b = B.pairs_at(W, 3)
    assert len(a) == len(b) == 2 * 28 * 25
    assert set(np.unique(b - a)) == {3.0, 84.0}                       # 3 right, or 3 rows down
    # one pixel to the right is study 174's neighbour measure
    a1, b1 = B.pairs_at(W, 1)
    h0, h1 = I.neighbour_pairs(W)
    np.testing.assert_array_equal(a1[:len(h0)], h0)
    np.testing.assert_array_equal(b1[:len(h1)], h1)


def test_falls_to_interpolates():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    assert B.falls_to(d, np.array([0.4, 0.3, 0.1, 0.0])) == 2.5       # 0.2 between 0.3, 0.1
    assert B.falls_to(d, np.array([0.4, 0.2, 0.1, 0.0])) == 2.0
    assert np.isnan(B.falls_to(d, np.array([0.4, 0.39, 0.38, 0.37])))


def test_weight_counts_are_exact():
    c = B.weight_counts()
    assert c["dense"] == 784 * 64 == 50176 and c["conv"] == 9 * 64 == 576
    assert c["out_side"] == 26 and c["outputs"] == 64 * 26 * 26 == 43264
    assert c["dense_same_outputs"] == 784 * 43264 == 33918976
    assert c["local"] == 9 * 43264 == 389376
    assert c["dense"] // c["conv"] == 87
    # the dense layer of the classifier is exactly the one counted
    assert I.SIZES[0] * I.SIZES[1] == c["dense"]
    small = B.weight_counts(units=2, kernel=3, side=5)
    assert small == dict(dense=50, conv=18, outputs=18, out_side=3, dense_same_outputs=450,
                         local=162, pixels=25, units=2, kernel=3)


def test_shift_moves_and_drops():
    rng = np.random.default_rng(1)
    x = rng.random((3, 784))
    img = x.reshape(3, 28, 28)
    r = B.shift(x, 2, (0, 1)).reshape(3, 28, 28)
    np.testing.assert_array_equal(r[:, :, 2:], img[:, :, :-2])
    assert (r[:, :, :2] == 0).all()
    d = B.shift(x, 3, (-1, 0)).reshape(3, 28, 28)                     # up
    np.testing.assert_array_equal(d[:, :-3], img[:, 3:])
    assert (d[:, -3:] == 0).all()
    np.testing.assert_array_equal(B.shift(x, 0, (1, 0)), x)
    # there and back again keeps everything that did not leave the frame
    back = B.shift(B.shift(x, 4, (0, -1)), 4, (0, 1)).reshape(3, 28, 28)
    np.testing.assert_array_equal(back[:, :, 4:], img[:, :, 4:])


def test_shifted_accuracy_and_ink():
    x = np.zeros((4, 784))
    x[:, 14 * 28 + 14] = 1.0                                          # one lit pixel, centred
    y = np.array([0, 1, 0, 1])

    def centre(v):                                                    # 1 when still centred
        return (v.reshape(-1, 28, 28)[:, 14, 14] > 0).astype(int)
    acc = B.shifted_accuracy(centre, x, y)
    assert acc.shape == (len(B.SHIFTS), len(B.DIRECTIONS))
    assert np.all(acc[0] == 0.5) and np.all(acc[1:] == 0.5)
    np.testing.assert_allclose(B.ink_kept(x), 1.0)
    edge = np.zeros((1, 784))
    edge[0, 14 * 28 + 27] = 1.0                                       # on the right edge
    assert np.isclose(B.ink_kept(edge)[1], 0.75)                      # gone moving right only


def test_cnn_gradients():
    rng = np.random.default_rng(2)
    p = V.cnn_init(rng, (2, 3), False, False)
    x = rng.random((3, 784))
    y = np.array([1, 4, 7])
    _, G = B.cnn_loss_grads(p, x, y)
    assert set(G) == set(p)
    fd_check(lambda P: B.cnn_loss_grads(P, x, y)[0], p, G, sorted(p))


def test_a_convolution_moves_with_the_picture():
    rng = np.random.default_rng(3)
    p = V.cnn_init(rng, (4, 4), False, False)
    x = rng.random((5, 784))
    for k, direction in ((2, (0, 1)), (3, (1, 0)), (1, (0, -1))):
        diff, scale = B.equivariance(p, x, k, direction)
        assert diff < 1e-12 * scale and scale > 0.1


# ---------------------------------------------------------------------- MNIST
@needs_mnist
def test_study_says_what_was_run():
    s = B.STUDIES[NAME]().validate()
    m = s.metadata
    assert len(s.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]
    assert m["evidence"]["measured"] and m["evidence"]["exact"]
    assert m["comes_from"] == "The entropy of a network's weights"
    assert m["leads_to"] == "Convolution, 3×3 kernel"
    assert m["updates"] == B.EPOCHS * (60000 // B.BATCH) == 2340
    assert m["mlp_parameters"] == 52650
    assert m["cnn_parameters"] == 9 * 8 + 8 + 72 * 16 + 16 + 5 * 5 * 16 * 10 + 10 == 5258


@needs_mnist
def test_trained_weights_are_local():
    m = B.STUDIES[NAME]().metadata
    before, after = (np.array(m["correlation"][w]) for w in ("before", "after"))
    assert np.abs(before).max() < 0.02                                # noise at every distance
    assert after[0] > 0.35 and after[0] > after[1] > after[2] > after[3]
    assert abs(after[-5:]).max() < 0.2 * after[0]                     # far apart: little left
    assert 1.0 < m["half_distance"] < 4.0
    # one pixel apart along rows and columns is close to study 174's rows-only measure
    h = m["horizontal_neighbours"]
    assert abs(h["after"] - after[0]) < 0.1 and abs(h["before"]) < 0.02
    np.testing.assert_allclose(h["after"], I.measures(I.params("after"))[2])


@needs_mnist
def test_shifted_digits():
    m = B.STUDIES[NAME]().metadata
    dense, cnn = (np.array(m["accuracy"][k]) for k in ("dense", "cnn"))
    assert dense[0] > 0.95 and cnn[0] > 0.95                          # both good in place
    assert np.all(np.diff(dense) < 0)                                 # moving costs the dense
    assert dense[-1] < dense[0] - 0.3
    # what was measured: the CNN keeps more at every shift past zero
    assert np.all(cnn[1:] > dense[1:])
    assert cnn[-1] > dense[-1] + 0.1
    assert m["ink_kept"][0] == 1.0 and m["ink_kept"][-1] > 0.98       # the ink stays in frame
    eq = m["equivariance"]
    assert eq["digits"] == B.EQUIVARIANCE_DIGITS
    assert eq["largest_difference"] < 1e-9 * eq["largest_value"]
    # the curves drawn are the four directions' mean
    for k in ("dense", "cnn"):
        np.testing.assert_allclose(np.mean(m["accuracy_by_direction"][k], 1), m["accuracy"][k])
    # the dense classifier scored here is study 174's
    assert np.isclose(m["accuracy"]["dense"][0], I.weight_entropy().metadata["accuracy"][
        "after"][0])
