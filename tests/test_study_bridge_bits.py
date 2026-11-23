"""The experiment behind animation 182 (voxanim.study_bridge_bits): the formats and grids
on cases with known answers, the entropy counter, the block scales against
study_hardware's, and every measured claim as a relative claim (CI retrains the
classifier elsewhere, where the last digits differ)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_bridge_bits as B
from voxanim import study_hardware as H
from voxanim import study_info as S

# ---------------------------------------------------------------------- exact


def test_bits_stored_are_the_formats_widths():
    assert [B.width(f) for f in B.FORMATS] == [32, 16, 16, 8, 4, 8.25, 4.25]
    assert B.width("FP4 scaled") - B.width("FP4") == B.SCALE_BITS / B.BLOCK == 0.25


def test_entropy_of_known_codes():
    assert B.code_entropy([3, 3, 3]) == (0.0, 1)
    bits, n = B.code_entropy(np.arange(16))
    assert n == 16 and bits == pytest.approx(4.0)
    bits, n = B.code_entropy([0, 0, 1, 2])
    assert n == 3 and bits == pytest.approx(1.5)
    assert B.code_entropy([-0.0, 0.0, 1.0, 1.0]) == (pytest.approx(1.0), 2)   # −0 is 0
    rng = np.random.default_rng(0)
    for _ in range(50):                               # never more than log2 of what is used
        c = rng.integers(0, rng.integers(1, 40), rng.integers(1, 300))
        bits, n = B.code_entropy(c)
        assert -1e-12 <= bits <= np.log2(n) + 1e-12 and bits <= np.log2(len(c)) + 1e-12


def test_integer_grid_has_its_levels_and_rounds_to_nearest():
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.1, (64, 8))
    for b in B.INT_BITS:
        v, c = B.integer_grid(b, W)
        top = 2 ** (b - 1) - 1
        assert np.all(c == np.round(c)) and np.abs(c).max() == top
        step = np.abs(W).max() / top
        np.testing.assert_allclose(v, c * step)
        assert np.abs(v - W).max() <= step / 2 + 1e-12
        assert len(np.unique(c)) <= 2 ** b - 1


def test_formats_round_onto_their_grids_and_are_idempotent():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.2, 4000)
    grid = H.small_grid("FP4")
    assert list(grid) == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    for f in ("FP16", "BF16", "FP8", "FP4"):
        v, c, s = B.rounded(f, x)
        np.testing.assert_array_equal(v, c)
        np.testing.assert_array_equal(s, 1.0)
        np.testing.assert_array_equal(H.to_format(f, v), v)
    v = B.rounded("FP4", x)[0]
    assert set(np.abs(v)) <= set(grid)
    assert np.all(v[np.abs(x) <= 0.25] == 0) and np.all(v[np.abs(x) > 0.25] != 0)


def test_block_scales_are_study_hardwares():
    rng = np.random.default_rng(3)
    W = rng.normal(0, 0.1, (96, 5)) * rng.uniform(0.01, 3, (96, 1))
    for f in ("FP8", "FP4"):
        v, c, s = B.rounded(f + " scaled", W)
        np.testing.assert_array_equal(v, H.to_format_scaled(f, W))
        np.testing.assert_array_equal(v, c * s)
        assert set(np.abs(c.ravel())) <= set(H.small_grid(f))
        assert np.all(np.log2(s) == np.round(np.log2(s)))         # powers of two
        for k in range(0, 96, B.BLOCK):                           # one per block and output
            assert np.all(s[k:k + B.BLOCK] == s[k])
            top = np.abs(c[k:k + B.BLOCK]).max(0)
            assert np.all(top >= H.SMALL[f][3] / 2)               # the largest near the top


# ---------------------------------------------------------------------- measured


@pytest.fixture(scope="module")
def meta():
    return B.bits_in_a_weight().validate().metadata


@needs_mnist
def test_study_says_what_was_run(meta):
    s = B.bits_in_a_weight()
    assert len(s.panels) == 4 and meta["dataset"] == "MNIST" and meta["limitations"]
    assert meta["evidence"]["measured"] and meta["evidence"]["exact"]
    assert meta["weights"] == 784 * 64 + 64 * 32 + 32 * 10 == 52544
    assert meta["histogram_174"] == pytest.approx(S.histogram_entropy(S.flat(S.params())))


@needs_mnist
def test_the_pipeline_scores_as_study_info():
    xt, yt = B.test_data()
    acc, loss = B.score(S.params("after"), xt, yt)
    _, lp, y, _ = S.test_scores()
    assert acc == pytest.approx(float((lp.argmax(1) == y).mean()))
    assert loss == pytest.approx(float(-lp[np.arange(len(y)), y].mean()), rel=1e-5)


@needs_mnist
def test_every_format_carries_fewer_bits_than_it_stores(meta):
    f, n = meta["formats"], meta["weights"]
    for name in B.FORMATS:
        assert f[name]["entropy"] < f[name]["stored"]
        assert f[name]["entropy"] <= np.log2(f[name]["distinct"]) + 1e-9
    # float32 keeps (nearly) every value different: the count is at its ceiling
    assert f["FP32"]["distinct"] > 0.99 * n
    assert f["FP32"]["entropy"] == pytest.approx(np.log2(n), abs=0.01)
    # narrower formats, fewer values and fewer bits
    assert f["FP32"]["entropy"] > f["FP16"]["entropy"] > f["BF16"]["entropy"] > \
        f["FP8"]["entropy"] > f["FP4 scaled"]["entropy"] > f["FP4"]["entropy"]
    assert f["FP8"]["distinct"] < 2 ** 8 and f["FP4 scaled"]["distinct"] <= 15
    assert f["FP4 scaled"]["entropy"] > 0.8 * 4                  # the scale uses the codes
    for name in ("FP8 scaled", "FP4 scaled"):                     # scales are cheap to send
        assert f[name]["scale_entropy"] < B.SCALE_BITS / B.BLOCK


@needs_mnist
def test_accuracy_holds_to_fp8_and_plain_fp4_collapses(meta):
    f, base = meta["formats"], meta["float32"]["accuracy"]
    assert base > 0.95
    for name in ("FP32", "FP16", "BF16", "FP8", "FP8 scaled"):
        assert f[name]["accuracy"] >= base - B.HOLD
    assert f["FP4"]["zeros"] > 0.9 and f["FP4"]["accuracy"] < 0.5
    assert f["FP4 scaled"]["zeros"] < 0.2 and f["FP4 scaled"]["accuracy"] > base - 0.02
    assert f["FP4"]["entropy"] < 0.5 < 3.0 < f["FP4 scaled"]["entropy"]
    assert meta["block_zero_scaled"] < meta["block_zero_fp4"]


@needs_mnist
def test_integer_grids_hold_until_few_bits_are_left(meta):
    ints, base = meta["ints"], meta["float32"]["accuracy"]
    es = [ints[str(b)]["entropy"] for b in B.INT_BITS]
    assert all(a < b for a, b in itertools.pairwise(es))               # more bits, more carried
    for b in B.INT_BITS:
        assert ints[str(b)]["entropy"] < b
    fewest = meta["fewest_int_bits"]
    assert 3 <= fewest <= 6
    for b in B.INT_BITS:
        held = ints[str(b)]["accuracy"] >= base - B.HOLD
        assert held == (b >= fewest) or b < meta["below_int_bits"]
    assert ints["8"]["accuracy"] >= base - B.HOLD
    assert ints["2"]["accuracy"] < 0.5 < ints["3"]["accuracy"]
    assert meta["fewest_entropy"] == ints[str(fewest)]["entropy"]
    assert 2.5 < meta["fewest_entropy"] < 5.0
