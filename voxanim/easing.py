"""Scalar helpers for timelines: clamping, remapping and easing curves.

Every function takes plain floats so timeline code can stay readable:

    u = ease_in_out(remap(t, t0, t1))

Mind the argument order: :func:`remap` takes the value *first*
(``remap(x, x0, x1)``, like the rest of this module), while
:func:`smoothstep` takes it *last* (``smoothstep(x0, x1, x)``, the GLSL
convention).
"""

from __future__ import annotations


def clamp(x, lo=0.0, hi=1.0):
    """``x`` limited to ``[lo, hi]``."""
    return lo if x < lo else (hi if x > hi else x)


def lerp(a, b, t):
    """``a`` at ``t = 0`` to ``b`` at ``t = 1`` (not clamped)."""
    return a + (b - a) * t


def remap(x, x0, x1):
    """Progress of ``x`` through ``[x0, x1]``, clamped to ``[0, 1]``.

    Value first: ``remap(x, x0, x1)`` (compare :func:`smoothstep`).  An empty
    interval is a step at ``x1``."""
    if x1 <= x0:
        return 1.0 if x >= x1 else 0.0
    return clamp((x - x0) / (x1 - x0))


def smoothstep(x0, x1, x):
    """Hermite ``3t^2 - 2t^3`` of :func:`remap` ``(x, x0, x1)``.

    Value *last*: ``smoothstep(x0, x1, x)`` (GLSL order), unlike
    :func:`remap`."""
    t = remap(x, x0, x1)
    return t * t * (3.0 - 2.0 * t)


def ease_in_out(t):
    """Cubic in/out -- the default for anything that travels."""
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out(t):
    """Cubic ease-out: fast start, gentle arrival."""
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in(t):
    """Cubic ease-in: gentle start, fast arrival."""
    t = clamp(t)
    return t * t * t


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
