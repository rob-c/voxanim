"""Drawing helpers shared by the regularisation animations (all flat
:class:`~voxanim.diagram.Diagram` s): the family's inks, signed weights as
small images, many small images packed into one texture, and contour lines
of a sampled surface.

Everything here is drawing only; the numbers come from :mod:`voxanim.regul`
and :mod:`voxanim.regul_ae`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .diagram import draw_image
from .netanim import div_colors


def _ink(h):
    return tuple(float(v) for v in C.rgb(h))


NONE_INK = (0.40, 0.42, 0.46)            # no regulariser: grey
L2_INK = _ink("#2f8a86")                 # L2 / weight decay: teal (as regul.REGS)
L1_INK = _ink("#3f55c8")                 # L1: blue (as regul.REGS)
EN_INK = _ink("#7a46d6")                 # elastic net: violet
SLOPE_INK = _ink("#8a6a2e")              # L1 by its subgradient: brown
ZERO = np.array([0.985, 0.985, 0.97], np.float32)   # an exactly-zero weight: blank paper
GAP_PAPER = (0.86, 0.865, 0.87)          # between the cells of a mosaic
SIDE = 28


def signed_texture(v, vmax, side=SIDE, mark_zero=True):
    """``(side * side,)`` signed values as a ``(side, side, 3)`` texture on the
    diverging scale (orange +, blue -, grey near 0); with ``mark_zero`` an
    entry that is *exactly* 0 is blank paper, so L1's zeros read apart from
    merely small weights."""
    v = np.asarray(v, np.float64).ravel()
    col = div_colors(v, vmax)
    if mark_zero:
        col[v == 0] = ZERO
    return col.reshape(side, side, 3)


def mosaic(textures, cols, gap=1, paper=GAP_PAPER):
    """Equal-sized ``(h, w, 3)`` textures packed row-major into one texture,
    ``cols`` per row, ``gap`` texels of ``paper`` between them (and none
    around the outside)."""
    textures = [np.asarray(t, np.float32) for t in textures]
    h, w = textures[0].shape[:2]
    rows = -(-len(textures) // cols)
    out = np.empty((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), np.float32)
    out[:] = np.asarray(paper, np.float32)
    for i, t in enumerate(textures):
        r, c = divmod(i, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    return out


def weight_mosaic(W, cols, vmax=None, mark_zero=True, units=None, gap=1):
    """The columns of ``W`` ``(784, units)`` (each unit's 28 x 28 weights) as one
    mosaic texture, every unit on the same colour scale (``vmax``, default the
    largest ``|W|``); ``units`` picks and orders the columns shown."""
    W = np.asarray(W, np.float64)
    idx = range(W.shape[1]) if units is None else units
    vmax = float(np.abs(W).max()) if vmax is None else vmax
    return mosaic([signed_texture(W[:, j], vmax, mark_zero=mark_zero) for j in idx], cols, gap)


def draw_mosaic(anim, sc, c, tex, cell_px, op, lift=0.3):
    """``tex`` drawn at world point ``c`` with each texel ``cell_px`` px wide."""
    h, w = tex.shape[:2]
    draw_image(anim, sc, c, tex, w * cell_px, h * cell_px, op, grid=0.0, lift=lift)


# ------------------------------------------------------------------ contours

# marching squares: for each of the 16 corner patterns, which cell edges the
# line joins (edges 0 bottom, 1 right, 2 top, 3 left; corners counter-clockwise
# from (i, j)).  The two saddle patterns pick one of their two readings.
_CASES = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 2), (0, 1)],
          6: [(0, 2)], 7: [(3, 2)], 8: [(2, 3)], 9: [(0, 2)], 10: [(0, 3), (1, 2)],
          11: [(1, 2)], 12: [(1, 3)], 13: [(0, 1)], 14: [(3, 0)]}


def contour_segments(F, xs, ys, level):
    """Line segments ``[((x0, y0), (x1, y1)), ...]`` of the contour ``F == level``
    of the surface ``F`` sampled at ``F[r, c] = f(xs[c], ys[r])``."""
    F = np.asarray(F, np.float64)
    above = level < F
    segs = []
    for r in range(F.shape[0] - 1):
        for c in range(F.shape[1] - 1):
            k = (int(above[r, c]) | int(above[r, c + 1]) << 1 | int(above[r + 1, c + 1]) << 2
                 | int(above[r + 1, c]) << 3)
            if k in (0, 15):
                continue
            corners = ((xs[c], ys[r], F[r, c]), (xs[c + 1], ys[r], F[r, c + 1]),
                       (xs[c + 1], ys[r + 1], F[r + 1, c + 1]), (xs[c], ys[r + 1], F[r + 1, c]))

            def edge(e, corners=corners):
                (xa, ya, fa), (xb, yb, fb) = corners[e], corners[(e + 1) % 4]
                u = (level - fa) / (fb - fa) if fb != fa else 0.5
                return (xa + (xb - xa) * u, ya + (yb - ya) * u)
            for a, b in _CASES[k]:
                segs.append((edge(a), edge(b)))
    return segs
