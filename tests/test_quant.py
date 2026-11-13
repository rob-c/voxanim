"""Quantization and precision (``voxanim.quant``) and the efficiency animations.

The quantizers, calibration, the straight-through estimator (against its
definition: the rounded loss is flat almost everywhere, so a finite-difference
check of the weight gradient is impossible and is not attempted; the bias
gradients, which rounding does not touch, are checked against finite
differences), bfloat16 rounding against an integer reference, the underflow
example, and every claim the subtitles make, as relative claims (trained
numbers differ in the last bits between machines).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import quant as Q
from voxanim.mnist import CLASSES, _backward, _forward, _init

# ------------------------------------------------------------------ quantizers


def _values(n=5000, seed=1):
    return np.random.default_rng(seed).normal(0, 0.2, (n, 8))


@pytest.mark.parametrize("bits", Q.BITS)
@pytest.mark.parametrize("symmetric", [True, False])
def test_quantizer_levels_and_round_trip(bits, symmetric):
    """At most 2^bits codes, the ones the quantizer promises; quantizing a
    quantized tensor changes nothing; the error is at most half a step."""
    x = _values()
    q = Q.calibrate(x, bits, symmetric)
    codes = q.quantize(x)
    assert codes.min() >= q.qmin and codes.max() <= q.qmax
    assert q.levels == (2 ** bits - 1 if symmetric else 2 ** bits)
    xq = q.fake(x)
    assert len(np.unique(xq)) <= q.levels
    assert np.array_equal(q.fake(xq), xq)
    assert np.abs(xq - x).max() <= 0.5 * float(q.scale.max()) + 1e-12
    if symmetric:
        g = q.grid()
        assert np.allclose(g, -g[::-1]) and 0.0 in g           # a symmetric grid through 0
    else:
        assert q.fake(np.zeros((1, 8)))[0, 0] == 0.0           # 0 is exact
    # min-max: the largest magnitude maps to the last code
    i = np.unravel_index(np.argmax(np.abs(x)), x.shape)
    assert codes[i] in (q.qmin, q.qmax)


def test_per_channel_scales():
    """One scale per column, each column's own largest weight at the last code."""
    x = _values()
    x[:, 3] *= 5.0
    q = Q.calibrate(x, 4, axis=1)
    assert q.scale.shape == (1, 8) and q.per_channel
    assert q.scale[0, 3] > 4 * q.scale[0, 0]
    assert np.allclose(np.abs(q.fake(x)).max(0), np.abs(x).max(0))
    per_tensor = Q.calibrate(x, 4)
    assert ((q.fake(x) - x) ** 2).sum() < ((per_tensor.fake(x) - x) ** 2).sum()
    with pytest.raises(ValueError):
        q.grid()


def test_calibration_clip_never_worse():
    """The MSE-calibrated clip has no more squared error than min-max, and at
    2 bits a lot less."""
    x = _values()
    for bits in Q.BITS:
        mm = ((Q.calibrate(x, bits).fake(x) - x) ** 2).sum()
        q, clip = Q.calibrate_mse(x, bits)
        mse = ((q.fake(x) - x) ** 2).sum()
        assert mse <= mm + 1e-12 and 0.1 <= float(clip) <= 1.0
        if bits == 2:
            assert mse < 0.5 * mm
    q, clip = Q.calibrate_mse(x, 2, axis=1)
    assert clip.shape == (1, 8)


def test_model_bytes():
    params = [[np.zeros((784, 64)), np.zeros(64)], [np.zeros((64, 10)), np.zeros(10)]]
    n_w = 784 * 64 + 64 * 10
    assert Q.model_bytes(params, 32) == 4 * (n_w + 74)
    assert Q.model_bytes(params, 8) == n_w + 4 * 74 + 4 * 2
    assert Q.model_bytes(params, 2, per_channel=True) == n_w // 4 + 4 * 74 + 4 * (64 + 10)


# ------------------------------------------------------------------ the STE


def _small_net(seed=0):
    rng = np.random.default_rng(seed)
    sizes = (12, 8, 6, CLASSES)
    P = {"%s%d" % (name, k): a for k, layer in enumerate(_init(rng, sizes))
         for name, a in zip("Wb", layer, strict=True)}
    x = rng.random((16, sizes[0]))
    y = np.eye(CLASSES)[rng.integers(0, CLASSES, 16)]
    return P, x, y


def test_ste_matches_its_definition():
    """``dL/dW`` is the float network's gradient at the rounded weights, masked
    where the weight saturates; the mask is exactly the clipping range."""
    P, x, y = _small_net()
    qs = [Q.calibrate_mse(P["W%d" % k], 2)[0] for k in range(3)]
    g, gq, _p = Q.qat_forward_backward(P, qs, x, y)
    params = [[qs[k].fake(P["W%d" % k]), P["b%d" % k]] for k in range(3)]
    acts, pre, p = _forward(params, x)
    ref, _d, _da = _backward(params, acts, pre, p, y)
    for k in range(3):
        W = P["W%d" % k]
        mask = qs[k].ste_mask(W)
        inside = np.abs(W) <= (qs[k].qmax + 0.5) * qs[k].scale.item()
        assert np.array_equal(mask, inside.astype(float))
        assert 0 < mask.mean() < 1                         # some weights do saturate
        assert np.allclose(gq[k], ref[k][0])
        assert np.allclose(g["W%d" % k], ref[k][0] * mask)
        assert np.allclose(g["b%d" % k], ref[k][1])
    # the gradient through the STE is not zero, although the true one is
    assert np.abs(g["W0"]).max() > 0


def test_bias_gradients_by_finite_differences():
    """Rounding does not touch the biases, so their gradients are exact."""
    P, x, y = _small_net()
    qs = [Q.calibrate_mse(P["W%d" % k], 4)[0] for k in range(3)]
    g, _gq, _p = Q.qat_forward_backward(P, qs, x, y)
    fd_check(lambda PP: Q.qat_loss(PP, qs, x, y), P, g, ["b0", "b1", "b2"])


def test_rounded_loss_is_flat():
    """Why finite differences cannot check the weight gradient: a tiny move of
    a weight does not change the quantized loss at all."""
    P, x, y = _small_net()
    qs = [Q.calibrate_mse(P["W%d" % k], 2)[0] for k in range(3)]
    base = Q.qat_loss(P, qs, x, y)
    P2 = {k: v.copy() for k, v in P.items()}
    P2["W0"][0, 0] += 1e-6
    assert Q.qat_loss(P2, qs, x, y) == base


# ------------------------------------------------------------------ number formats


def _bf16_reference(v):
    """bfloat16 rounding of a float32 by integer arithmetic on its bit pattern."""
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    upper, lower = u >> 16, u & 0xFFFF
    if lower > 0x8000 or (lower == 0x8000 and upper & 1):
        upper += 1
    return struct.unpack("<f", struct.pack("<I", upper << 16))[0]


def test_bf16_rounding_against_reference():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 1, 2000), rng.normal(0, 1e-30, 100),
                        10.0 ** rng.uniform(-40, 38, 500) * rng.choice([-1, 1], 500),
                        [1 + 2 ** -8, 1 + 3 * 2 ** -8, -1 - 2 ** -8, 0.0, -0.0, 0.425468802]]
                       ).astype(np.float32)
    got = Q.round_bf16(x)
    ref = np.array([_bf16_reference(float(v)) for v in x], np.float32)
    assert np.array_equal(got, ref)
    assert np.all(got.view(np.uint32) & 0xFFFF == 0)             # 7 mantissa bits
    normal = np.abs(x) >= 2.0 ** -126
    assert np.all(np.abs(got - x)[normal] <= 2.0 ** -8 * np.abs(x)[normal])   # half an ulp
    assert float(Q.round_bf16(np.float32(1 + 2 ** -8))) == 1.0   # ties to even
    assert float(Q.round_bf16(np.float32(1 + 3 * 2 ** -8))) == 1 + 2 ** -6
    assert np.isinf(Q.round_bf16(np.float32(3.4e38))) and np.isnan(Q.round_bf16(np.float32("nan")))
    assert float(Q.round_to(0.425468802, "bfloat16")) == 0.42578125
    assert float(Q.round_to(0.425468802, "float16")) == pytest.approx(0.425537109, abs=1e-9)


def test_format_info_and_bits():
    for fmt, (s, e, m) in Q.FIELDS.items():
        info = Q.format_info(fmt)
        assert (info["sign"], info["exponent"], info["mantissa"]) == (s, e, m)
        sign, exp, man = Q.bit_fields(1.0, fmt)
        assert sign == "0" and man == "0" * m
        assert int(exp, 2) == 2 ** (e - 1) - 1                  # the bias: 1.0 = 2^0
        assert "".join(Q.bit_fields(-1.0, fmt))[0] == "1"
    assert Q.format_info("float16")["tiny"] == 2.0 ** -14
    assert Q.format_info("float16")["smallest_subnormal"] == 2.0 ** -24
    assert Q.format_info("bfloat16")["tiny"] == 2.0 ** -126
    assert Q.format_info("bfloat16")["max"] == pytest.approx(3.3895e38, rel=1e-4)


def test_ulp_and_representable_grid():
    assert Q.ulp(1.0, "float32") == 2.0 ** -23
    assert Q.ulp(1.0, "float16") == 2.0 ** -10
    assert Q.ulp(1.0, "bfloat16") == 2.0 ** -7
    assert Q.ulp(1e-5, "float16") == 2.0 ** -24                  # subnormal: the step freezes
    assert Q.ulp(1e-3, "float16") == 2.0 ** -20
    assert Q.ulp(1e-3, "float16") / 1e-3 < 2 * Q.ulp(1e-5, "float16") / 1e-5
    pts = Q.representable(0.99, 1.01, "bfloat16")
    assert len(pts) == 4 and np.all(np.diff(pts) > 0) and 1.0 in pts
    assert np.array_equal(Q.round_to(pts, "bfloat16"), pts.astype(np.float32))
    assert Q.representable(0.99, 1.01, "float32") is None        # too many to draw
    assert len(Q.representable(0.99e-5, 1.01e-5, "float16")) == 3
    assert Q.next_up(-1.0, "bfloat16") == -1 + 2.0 ** -8


def test_classify_and_loss_scaling():
    g = np.array([1e-9, 1e-6, 1e-3, -1e-9, 0.0])
    zero, sub, normal = Q.classify(g, "float16")
    assert (zero, sub, normal) == (0.5, 0.25, 0.25)
    assert Q.classify(g, "bfloat16") == (0.0, 0.0, 1.0)
    assert Q.classify(g * 2.0 ** 16, "float16")[0] == 0.0
    back = Q.loss_scaled(g, 2.0 ** 16, "float16")
    assert np.all(back[g != 0] != 0)
    assert np.allclose(back, g, rtol=2.0 ** -10)


# ------------------------------------------------------------------ on the real models


@pytest.fixture(scope="module")
def mlp():
    params, acc, data = Q.trained_mlp()
    return params, acc, data


@needs_mnist
def test_ptq_claims(mlp):
    """int8 is free, int4 nearly, 2 bits collapses per tensor and comes back
    with per-channel and calibrated scales; the bytes shrink as promised."""
    params, acc, (xtr, _ytr, xte, yte) = mlp
    tab = Q.ptq_accuracies(params, xte, yte)
    assert acc > 0.95
    assert abs(tab[8, "per-tensor"] - acc) < 0.003
    assert acc - 0.015 < tab[4, "per-tensor"] < acc + 0.003
    assert tab[2, "per-tensor"] < 0.3                           # near chance
    assert tab[2, "per-channel"] > tab[2, "per-tensor"]
    assert tab[2, "calibrated"] > tab[2, "per-channel"] + 0.2 and tab[2, "calibrated"] > 0.6
    b = {bits: Q.model_bytes(params, bits) for bits in (32, 8, 4, 2)}
    assert b[32] / 4.2 < b[8] < b[32] / 3.9 and b[4] < b[32] / 7.8 and b[2] < b[32] / 15
    ranges = Q.activation_ranges(params, xtr[:Q.CALIB_IMAGES])
    pq, _ = Q.quantize_params(params, 8, **Q.SCHEMES["calibrated"])
    a88 = float((Q.forward_quantized(pq, xte, 8, ranges).argmax(1) == yte).mean())
    assert abs(a88 - acc) < 0.005


@needs_mnist
def test_qat_claims(mlp):
    """QAT wins back most of what PTQ lost at 2 bits, the first steps flip a
    few weights, and the float weights do not gather at the levels."""
    _params, acc, _data = mlp
    r2, r4 = Q.qat_train(2), Q.qat_train(4)
    assert r2.acc_float == pytest.approx(acc)
    lost, won = acc - r2.acc[0], r2.acc[-1] - r2.acc[0]
    assert lost > 0.1 and won > 0.8 * lost and r2.acc[-1] > 0.94
    assert abs(r4.acc[-1] - acc) < 0.01 and r4.acc[0] > acc - 0.015
    q = r2.quantizer
    assert np.array_equal(r2.steps[:11], np.arange(11))
    flips = [(q.quantize(r2.W[i]) != q.quantize(r2.W[i + 1])).sum() for i in range(5)]
    assert all(0 < f < 0.05 * r2.W[0].size for f in flips)
    u = r2.W[-1] / q.scale.item()
    frac = np.abs(u - np.round(u))
    near_level, near_boundary = (frac < 0.1).mean(), (frac > 0.4).mean()
    assert near_boundary > 1.2 * near_level                     # oscillation, not clustering
    assert r2.W[-1].std() > r2.W[0].std()                       # they spread
    assert r2.grad_q.shape == r2.W[0].shape and np.abs(r2.grad_q).max() > 0


@needs_mnist
def test_underflow_example():
    u = Q.underflow_gradient()
    g = u["g"]
    assert g.shape == (100, 100) and u["rms"] == pytest.approx(np.sqrt(np.mean(g ** 2)))
    zero, sub, normal = Q.classify(g, "float16")
    assert zero > 0.5 and normal == 0.0 and zero + sub == pytest.approx(1.0)
    assert Q.classify(g, "bfloat16") == (0.0, 0.0, 1.0)
    zero, sub, normal = Q.classify(g * Q.LOSS_SCALE, "float16")
    assert zero == 0.0 and normal > 0.9                        # scaled: none lost, most normal
    back = Q.loss_scaled(g, Q.LOSS_SCALE, "float16")
    nz = g != 0
    assert np.all(back[nz] != 0)
    assert np.median(np.abs(back - g)[nz] / np.abs(g[nz])) < 1e-3


@needs_mnist
def test_precision_claims():
    """Which precisions train: all but float16 alone, which diverges at once."""
    runs = {label: r for (label, _f, _m), r in zip(Q.PRECISION_RUNS, Q.precision_runs(),
                                                   strict=True)}
    f32, f16, bf16, mixed = (runs[k] for k in ("float32", "float16", "bfloat16",
                                                "float16 + float32 master weights"))
    assert f32.diverged == -1 and f32.acc > 0.94
    assert 0 <= f16.diverged < 5 and f16.acc < 0.2
    assert bf16.diverged == -1 and f32.acc - 0.015 < bf16.acc <= f32.acc + 0.005
    assert mixed.diverged == -1 and abs(mixed.acc - f32.acc) < 0.005
    assert float(Q.round_to(1e-8, "float16")) == 0.0            # Adam's epsilon


# ------------------------------------------------------------------ the animations


@needs_mnist
@pytest.mark.parametrize("name", ["quantization", "quantization-aware-training",
                                  "mixed-precision"])
def test_animations_build(name):
    from voxanim import presets
    anim = presets.build(name)
    assert 40 <= anim.duration <= 60
    seen = set()
    for t in np.linspace(0, anim.t_done, 25):
        anim.scene(float(t))
        seen.add(anim.subtitle)
        assert anim.subtitle
    assert len(seen) >= len(anim.PHASES)


@needs_mnist
def test_quantization_table_matches_the_model(mlp):
    from voxanim.quant_anim import Quantization
    params, acc, (_xtr, _ytr, xte, yte) = mlp
    anim = Quantization()
    assert anim.acc == acc
    for s in anim.stages:
        assert s["acc"] == Q.ptq_accuracies(params, xte, yte, (s["bits"],),
                                            {s["scheme"]: Q.SCHEMES[s["scheme"]]})[
            s["bits"], s["scheme"]]
        assert np.allclose(s["Wq"], s["q"].fake(anim.W))
