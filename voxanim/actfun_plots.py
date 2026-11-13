"""Small plots of an activation function shared by the ``activation-*``
animations: ``f`` and ``f'`` over ``z``, and ``f'`` over the histogram of a
real layer's pre-activations.  All of them draw into a :class:`~voxanim.diagram.Diagram`
inside a box of layout fractions ``((x0, x1), (top, bottom))``.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .deepnet import act, act_grad
from .diagram import GRAD
from .easing import remap, smoothstep

Z_RANGE = (-4.5, 4.5)                     # the plots' z axis
Y_RANGES = {"sigmoid": (-0.3, 1.3), "tanh": (-1.3, 1.3)}
Y_DEFAULT = (-1.3, 4.3)                   # the rectifiers
SLOPE_RANGE = (-0.2, 1.3)                 # f' plots: GELU's and SiLU's slopes top 1
N_CURVE = 121                             # points on a curve
PANEL = (0.955, 0.955, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
AXIS = (0.5, 0.52, 0.54)
HIST_INK = (0.72, 0.74, 0.76)             # the histogram bars under a slope curve
FLAT_INK = (0.84, 0.70, 0.70)             # the bars of units on a flat part of the activation
LINE_W, SLOPE_W = 2.0, 1.6


def ramp_colors(v, lo_hi, dark, lit):
    """``(n, 3)`` colours of ``|v|`` on a log scale, linearly from ``dark`` at
    ``10 ** lo`` (and below) to ``lit`` at ``10 ** hi``; a non-finite value is dark."""
    lo, hi = lo_hi
    v = np.asarray(v, np.float64).ravel()
    a = np.clip((np.log10(np.abs(v) + 1e-300) - lo) / (hi - lo), 0.0, 1.0)
    a[~np.isfinite(v)] = 0.0
    dark, lit = np.asarray(dark, np.float32), np.asarray(lit, np.float32)
    return (dark[None] + (lit - dark)[None] * a[:, None]).astype(np.float32)


def y_range(key):
    """The plot's y range for activation ``key``."""
    return Y_RANGES.get(key, Y_DEFAULT)


def zs():
    """The z samples of a curve, skipping 0 exactly (ReLU' jumps there)."""
    z = np.linspace(*Z_RANGE, N_CURVE)
    return z[z != 0.0]


def point(anim, box, z, v, yr, lift=0.3):
    """World point of ``(z, v)`` in the plot ``box`` whose y range is ``yr``."""
    (x0, x1), (y0, y1) = box
    fx = x0 + (x1 - x0) * (z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
    fy = y1 - (y1 - y0) * (v - yr[0]) / (yr[1] - yr[0])
    return anim.P(fx, fy, lift)


def panel(anim, sc, box, op, yr):
    """The plot's background, its axes through 0 and nothing else."""
    (x0, x1), (y0, y1) = box
    a, b = anim.P(x0, y0), anim.P(x1, y1)
    w, h = anim.px(b) - anim.px(a)
    anim.rect(sc, (a + b) / 2, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0, radius=4,
              lift=0.1)
    anim.polyline(sc, [point(anim, box, z, 0.0, yr, 0.2) for z in Z_RANGE], 1.0, AXIS, 0.6 * op,
                  bias=0.1)
    anim.polyline(sc, [point(anim, box, 0.0, v, yr, 0.2) for v in yr], 1.0, AXIS, 0.6 * op,
                  bias=0.1)


def curve(anim, sc, box, values, yr, u, ink, width, op):
    """The first fraction ``u`` of the curve ``values`` over :func:`zs`."""
    z = zs()
    m = int(np.ceil(len(z) * u))
    if m > 1:
        pts = [point(anim, box, zi, float(np.clip(v, *yr)), yr) for zi, v in
               zip(z[:m], values[:m])]
        anim.polyline(sc, pts, width, ink, 0.95 * op, bias=0.2)


def function_plot(anim, sc, box, a, u, op, sub=""):
    """Activation ``a`` (an :class:`~voxanim.actfun.Act`) in ``box``: ``f`` in its
    ink, drawn over the first half of ``u``, then ``f'`` in the gradient
    colour; its label (and ``sub`` under it) below the plot."""
    if op <= 0.01:
        return
    yr = y_range(a.key)
    panel(anim, sc, box, op, yr)
    z = zs()
    curve(anim, sc, box, act(a.key, z), yr, remap(u, 0.0, 0.55), a.ink, LINE_W, op)
    curve(anim, sc, box, act_grad(a.key, z), yr, remap(u, 0.5, 1.0), GRAD, SLOPE_W, op)
    (x0, x1), (_y0, y1) = box
    anim.label(sc, anim.P((x0 + x1) / 2, y1), a.label, layout.MIN_TEXT, anchor="mt", dy=6,
               ink=a.ink, opacity=op)
    if sub:
        anim.label(sc, anim.P((x0 + x1) / 2, y1), sub, layout.MIN_TEXT, anchor="mt", dy=24,
                   ink=layout.NOTE_INK, opacity=op)
    anim.label(sc, anim.P(x1, y1), "f′", layout.MIN_TEXT, anchor="rb", dx=-3, dy=-4,
               ink=tuple(float(v) for v in GRAD * 0.8), opacity=op * smoothstep(0.5, 0.7, u))


def slope_plot(anim, sc, box, a, hist, flat, op, u=1.0, full=0.35):
    """``f'`` of activation ``a`` over the histogram ``hist`` (fractions per
    bin over :data:`Z_RANGE`) of a real layer's pre-activations; bins whose
    centre lies where ``|f'| <`` the flat threshold are tinted (``flat`` marks
    them).  A bin holding ``full`` of the values fills the plot's height."""
    if op <= 0.01:
        return
    yr = SLOPE_RANGE
    panel(anim, sc, box, op, yr)
    (x0, x1), (y0, y1) = box
    n = len(hist)
    edges = np.linspace(Z_RANGE[0], Z_RANGE[1], n + 1)
    shown = int(np.ceil(n * smoothstep(0.0, 0.6, u)))
    for b in range(shown):
        h = min(1.0, hist[b] / full)
        if h <= 0:
            continue
        fx0 = x0 + (x1 - x0) * (edges[b] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
        fx1 = x0 + (x1 - x0) * (edges[b + 1] - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
        top = y1 - (y1 - y0) * h
        p0, p1 = anim.P(fx0, top), anim.P(fx1, y1)
        w, hh = anim.px(p1) - anim.px(p0)
        anim.rect(sc, (p0 + p1) / 2, max(w - 1.0, 1.0), hh, FLAT_INK if flat[b] else HIST_INK,
                  0.9 * op, radius=0.0, lift=0.15)
    curve(anim, sc, box, act_grad(a.key, zs()), yr, remap(u, 0.3, 1.0), a.ink, LINE_W, op)
    anim.label(sc, anim.P((x0 + x1) / 2, y1), a.label, layout.MIN_TEXT, anchor="mt", dy=6,
               ink=a.ink, opacity=op)
