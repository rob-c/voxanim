"""Convolution arithmetic and the conv animations' schedule."""

from __future__ import annotations

import itertools

import numpy as np
from conftest import CONV_PRESETS

from voxanim import presets
from voxanim.conv import ConvSpec, ConvStage, conv_forward


def test_conv_forward_matches_einsum():
    rng = np.random.default_rng(0)
    for spec in (ConvSpec(), ConvSpec(padding=(1, 1, 1, 1), stride=(2, 2)),
                 ConvSpec(groups=2), ConvSpec(groups=8, filters=8),
                 ConvSpec(in_hw=(12, 1), kernel=(3, 1), filters=4)):
        x = rng.normal(size=spec.in_hw + (spec.in_ch,))
        w = rng.normal(size=(spec.filters,) + spec.kernel + (spec.group_ch,))
        got = conv_forward(spec, x, w)[..., 0]
        t, b, l, r = spec.padding
        xp = np.pad(x, ((t, b), (l, r), (0, 0)))
        (kh, kw), (sh, sw) = spec.kernel, spec.stride
        Ho, Wo = spec.out_hw
        want = np.zeros((Ho, Wo, spec.filters))
        for f in range(spec.filters):
            g = f // spec.per_group
            xs = xp[..., g * spec.group_ch:(g + 1) * spec.group_ch]
            for i in range(Ho):
                for j in range(Wo):
                    want[i, j, f] = np.einsum(
                        "abc,abc->", xs[i * sh:i * sh + kh, j * sw:j * sw + kw], w[f])
        assert np.allclose(got, want), spec


def _brute(spec, x, w):
    """Tap-by-tap reference, written independently of conv_forward."""
    H, W, Cin, D = x.shape
    a, b = spec.input_dilation
    t, _bo, l, _r = spec.padding
    Hp, Wp = spec.padded_hw
    xp = np.zeros((Hp, Wp, Cin, spec.padded_d))
    for i in range(H):
        for j in range(W):
            xp[t + i * a, l + j * b, :, spec.pad_d[0]:spec.pad_d[0] + D] = x[i, j]
    (kh, kw), (sh, sw), (dh, dw) = spec.kernel, spec.stride, spec.dilation
    Ho, Wo = spec.out_hw
    y = np.zeros((Ho, Wo, spec.out_ch, spec.out_d))
    for i, j, k in np.ndindex(Ho, Wo, spec.out_d):
        taps = [(i * sh + p * dh, j * sw + q * dw, k * spec.stride_d + r)
                for p in range(kh) for q in range(kw) for r in range(spec.kernel_d)]
        if spec.pool:
            vals = np.array([xp[u, v, :, z] for u, v, z in taps])
            y[i, j, :, k] = vals.max(0) if spec.pool == "max" else vals.mean(0)
            continue
        for f in range(spec.filters):
            g = f // spec.per_group
            for (u, v, z), (p, q, r) in zip(taps, np.ndindex(kh, kw, spec.kernel_d)):
                for c in range(spec.group_ch):
                    y[i, j, f, k] += xp[u, v, g * spec.group_ch + c, z] * w[f, p, q, c, r]
    return y


def test_conv_forward_variants_match_brute_force():
    rng = np.random.default_rng(1)
    for spec in (ConvSpec(in_hw=(9, 9), in_ch=2, filters=2, dilation=(2, 2)),
                 ConvSpec(in_hw=(4, 4), in_ch=2, filters=2, input_dilation=(2, 2),
                          padding=(2, 2, 2, 2)),
                 ConvSpec(in_hw=(5, 5), in_d=5, in_ch=2, filters=2, kernel_d=3,
                          stride=(2, 1), stride_d=2),
                 ConvSpec(in_hw=(8, 8), in_ch=3, kernel=(2, 2), stride=(2, 2), pool="max"),
                 ConvSpec(in_hw=(8, 8), in_ch=3, kernel=(2, 2), stride=(2, 2), pool="avg")):
        x = rng.normal(size=spec.in_hw + (spec.in_ch, spec.in_d))
        w = rng.normal(size=(spec.filters,) + spec.kernel + (spec.group_ch, spec.kernel_d))
        got = conv_forward(spec, x, None if spec.pool else w)
        assert got.shape == spec.out_hw + (spec.out_ch, spec.out_d), spec
        assert np.allclose(got, _brute(spec, x, w)), spec
    # by hand: ones everywhere -> every output counts the taps it read
    s = ConvSpec(in_hw=(4, 4), in_d=4, in_ch=1, filters=1, kernel_d=3)
    y = conv_forward(s, np.ones((4, 4, 1, 4)), np.ones((1, 3, 3, 1, 3)))
    assert y.shape == (2, 2, 1, 2) and np.all(y == 27)


def test_presets_cover_every_output_cell_once():
    for name in CONV_PRESETS:
        anim = presets.get(name).build()
        for st in anim.stages:
            if not isinstance(st, ConvStage):
                anim.scene(st.t_end + 0.01)
                assert st.output.present.all(), name       # a combine step
                continue
            s = st.spec
            Ho, Wo = s.out_hw
            cells = [(x.i, x.j, x.k, x.f) for x in st.steps]
            assert len(cells) == len(set(cells)) == Ho * Wo * s.out_d * s.passes, name
            starts = [x.t for x in st.steps]
            assert all(b > a for a, b in itertools.pairwise(starts)), name
            # every output cell is on screen once the stage has finished
            anim.scene(st.t_end + 0.01)
            assert st.output.present.all(), name


def test_serpentine_order():
    s = ConvSpec(in_hw=(3, 5), kernel=(1, 3), order="serpentine")
    assert s.steps()[:6] == [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 1, 0), (1, 0, 0)]
