"""Scalar helpers for timelines: clamping, remapping and easing curves.

Every function takes plain floats so timeline code can stay readable:

    u = ease_in_out(remap(t, t0, t1))
"""

from __future__ import annotations

import math


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def lerp(a, b, t):
    return a + (b - a) * t


def remap(x, x0, x1):
    """Progress of ``x`` through ``[x0, x1]``, clamped to ``[0, 1]``."""
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    return clamp((x - x0) / (x1 - x0))


def smoothstep(x0, x1, x):
    t = remap(x, x0, x1)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t):
    """Cubic in/out -- the default for anything that travels."""
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    t = clamp(t)
    return t * t * t


def ease_out_back(t, overshoot=1.4):
    """Arrives slightly past the target and settles -- used for 'pops'."""
    t = clamp(t)
    c3 = overshoot + 1
    return 1 + c3 * (t - 1) ** 3 + overshoot * (t - 1) ** 2


def pulse(t, t0, rise, fall):
    """0 -> 1 -> 0 envelope starting at ``t0``: linear rise, exponential-ish fall."""
    if t < t0:
        return 0.0
    if t < t0 + rise:
        return (t - t0) / rise if rise > 0 else 1.0
    return max(0.0, 1.0 - ease_out((t - t0 - rise) / fall)) if fall > 0 else 0.0


def soft_band(x, lo, hi, soft):
    """1 inside ``[lo, hi]``, smoothly falling to 0 over ``soft`` outside it."""
    if hi <= lo:
        return 0.0
    a = smoothstep(lo - soft, lo + soft, x) if soft > 0 else float(x >= lo)
    b = 1.0 - (smoothstep(hi - soft, hi + soft, x) if soft > 0 else float(x > hi))
    return a * b


def exp_decay(n, scale):
    return math.exp(-n / scale) if scale > 0 else 0.0
