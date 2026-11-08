"""Colours and palettes.

Colours are float RGB triples in ``[0, 1]`` (sRGB, not linearised -- the look
is tuned in display space, like the originals).  Every palette colour is the
*base* colour of a voxel, i.e. what its fully-lit right-hand face renders as;
the shading rig in :mod:`voxanim.sprites` derives the other faces from it.

The defaults were sampled from the animatedai.github.io conv2d GIFs.
"""

from __future__ import annotations

import colorsys

import numpy as np


def rgb(value):
    """Accept '#rrggbb', (r, g, b) in 0..255, or (r, g, b) in 0..1."""
    if isinstance(value, str):
        s = value.lstrip("#")
        return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0
    arr = np.asarray(value, np.float32)
    return arr / 255.0 if arr.max() > 1.0 else arr


def mix(a, b, t):
    a, b = rgb(a), rgb(b)
    return a + (b - a) * np.float32(t)


def lighten(c, t):
    return mix(c, (1.0, 1.0, 1.0), t)


def darken(c, t):
    return rgb(c) * np.float32(1.0 - t)


def hex_of(c):
    c = np.clip(np.round(rgb(c) * 255), 0, 255).astype(int)
    return "#%02x%02x%02x" % tuple(c)


def shift_lightness(c, amount):
    """HLS lightness shift; keeps hue, used to derive group tints."""
    h, l, s = colorsys.rgb_to_hls(*rgb(c))
    return np.array(colorsys.hls_to_rgb(h, min(1.0, max(0.0, l + amount)), s), np.float32)


# ------------------------------------------------------------------ palette

INPUT = rgb("#449696")            # teal input volume
PAD = rgb("#8a8a8a")              # zero-padding voxels
BACKGROUND = rgb("#dfdfdf")
CAGE = rgb("#4a4a4a")             # output wireframe
OUTLINE = rgb("#ffffff")          # receptive-field frame
TEXT = rgb("#a0a0a0")

# Filter colours in the order the originals use them.
FILTERS = [rgb(h) for h in (
    "#d84e4e",   # red
    "#a1814b",   # ochre
    "#5f9a45",   # olive green
    "#3d9e45",   # green
    "#429696",   # teal
    "#6370eb",   # blue
    "#8952eb",   # violet
    "#be3c98",   # magenta
)]


def filter_colors(n):
    """``n`` distinct filter colours: the original 8, then evenly spaced hues."""
    if n <= len(FILTERS):
        return FILTERS[:n]
    out = list(FILTERS)
    for k in range(n - len(FILTERS)):
        h = (k * 0.618034) % 1.0
        out.append(np.array(colorsys.hls_to_rgb(h, 0.52, 0.55), np.float32))
    return out


def shades(base, n, lo=-0.20, hi=0.16):
    """``n`` lightness steps of one hue (the depthwise animation's red ramp)."""
    if n == 1:
        return [rgb(base)]
    return [shift_lightness(base, lo + (hi - lo) * k / (n - 1)) for k in range(n)]


def value_color(base, v, gain=0.62):
    """Modulate ``base`` by a signed value in [-1, 1]: brighter = larger."""
    v = float(np.clip(v, -1.0, 1.0))
    base = rgb(base)
    if v >= 0:
        return mix(base, (1.0, 1.0, 1.0), gain * 0.75 * v)
    return mix(base, base * 0.28, gain * -v)


# ------------------------------------------------------------------ value styles
#
# How a number in [-1, 1] becomes a voxel colour.  "hue" keeps the role's own
# colour (filter / input tint) and changes its lightness; "diverging" is a
# signed map -- warm for positive, cool for negative, neutral grey at zero --
# for weights and responses whose sign matters (edge filters); "gray" is an
# image in [0, 1].

POSITIVE = rgb("#e8622f")
NEGATIVE = rgb("#2f7ae8")
NEUTRAL = rgb("#cfd2d4")
GRAY_LO = rgb("#1d1f21")
GRAY_HI = rgb("#f3f3f1")


def diverging(v, pos=POSITIVE, neg=NEGATIVE, neutral=NEUTRAL):
    v = float(np.clip(v, -1.0, 1.0))
    a = abs(v) ** 0.75
    return mix(neutral, pos if v >= 0 else neg, a)


def gray(v):
    return mix(GRAY_LO, GRAY_HI, float(np.clip(v, 0.0, 1.0)))


def styled(style, base, v):
    """Colour for value ``v`` drawn in ``style`` (base = the role's own colour)."""
    if style == "diverging":
        return diverging(v)
    if style == "gray":
        return gray(v)
    if style == "relu":
        # after ReLU: 0 stays dark, positive values light up in the base hue
        if v <= 0:
            return rgb("#2b2f33")
        return mix(rgb(base) * 0.45, lighten(base, 0.6), float(np.clip(v, 0, 1)) ** 0.6)
    return value_color(base, v)
