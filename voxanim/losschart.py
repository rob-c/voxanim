"""A loss-against-steps chart drawn in the scene, shared by the training-deep-
networks animations.

The chart lives in a box of *layout fractions* (as everything placed with
``cam.frac``); ``place(fx, fy)`` maps a layout fraction to the world point a
line goes through (``cam.frac`` for a :class:`~voxanim.netanim.NetAnimation`,
``Diagram.P`` for a flat diagram), and text goes through the camera's
composition zoom so it lines up with the lines.  Either axis may be
logarithmic; curves are drawn up to a step, so the chart draws itself as the
recorded run plays.

Also here: :func:`pow10`, the ``8×10⁻²¹`` typography of a magnitude read-out.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .text import TIMES

AXIS = (0.5, 0.52, 0.54)                 # the axes' colour
TICK = 0.006                             # tick length, frame fraction
_SUPER = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sup(n):
    """Superscript digits (and minus) for ``n``: ``sup(-21) == "⁻²¹"``."""
    return str(n).translate(_SUPER)


def pow10(v, digits=0):
    """``v`` as ``m×10ⁿ`` with ``digits`` decimals in the mantissa (``8×10⁻²¹``);
    values in ``[0.01, 1000)`` are written plainly."""
    v = float(v)
    if v == 0:
        return "0"
    if 0.01 <= abs(v) < 1000:
        return "%.*g" % (3, v) if digits == 0 else "%.*f" % (digits, v)
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    if digits == 0 and round(m) == 10:
        m, e = 1.0, e + 1
    return "%.*f%s10%s" % (digits, m, TIMES, sup(e))


def frame_xy(cam, fx, fy):
    """Layout fraction -> frame fraction (the camera's composition zoom)."""
    k = cam.zoom
    return 0.5 + (fx - 0.5) * k, 0.5 + (fy - 0.5) * k


class LossChart:
    """A chart in the layout box ``((x0, x1), (top, bottom))`` with ``x`` over
    ``x_range`` and ``y`` over ``y_range`` (log scales when asked)."""

    def __init__(self, box, x_range, y_range, xlog=False, ylog=False):
        (self.x0, self.x1), (self.y0, self.y1) = box
        self.x_range, self.y_range = x_range, y_range
        self.xlog, self.ylog = xlog, ylog

    def fx(self, x):
        """Layout x of a step."""
        a, b = self.x_range
        if self.xlog:
            a, b, x = np.log10(a), np.log10(b), np.log10(max(x, 1e-300))
        return self.x0 + (self.x1 - self.x0) * float(np.clip((x - a) / (b - a), 0, 1))

    def fy(self, y):
        """Layout y of a loss."""
        a, b = self.y_range
        if self.ylog:
            a, b, y = np.log10(a), np.log10(b), np.log10(max(y, 1e-300))
        return self.y1 - (self.y1 - self.y0) * float(np.clip((y - a) / (b - a), 0, 1))

    def axes(self, sc, cam, place, op, xlabel="", ylabel="", xticks=(), yticks=(),
             size=layout.MIN_TEXT):
        """The two axes with their labels and tick labels."""
        if op <= 0.01:
            return
        x0, x1, y0, y1 = self.x0, self.x1, self.y0, self.y1
        sc.line(place(x0, y1), place(x1, y1), width=1.0, color=AXIS, opacity=0.7 * op, bias=0.0)
        sc.line(place(x0, y1), place(x0, y0), width=1.0, color=AXIS, opacity=0.7 * op, bias=0.0)
        for x, label in xticks:
            f = self.fx(x)
            sc.line(place(f, y1), place(f, y1 + TICK), width=1.0, color=AXIS, opacity=0.7 * op,
                    bias=0.0)
            sc.text(*frame_xy(cam, f, y1 + TICK + 0.006), label, size, "regular", layout.NOTE_INK,
                    op, "mt")
        for y, label in yticks:
            f = self.fy(y)
            sc.line(place(x0 - TICK, f), place(x0, f), width=1.0, color=AXIS, opacity=0.7 * op,
                    bias=0.0)
            sc.text(*frame_xy(cam, x0 - TICK - 0.006, f), label, size, "regular", layout.NOTE_INK,
                    op, "rm")
        if xlabel:
            sc.text(*frame_xy(cam, (x0 + x1) / 2, y1 + TICK + 0.046), xlabel, size, "regular",
                    layout.CAPTION_INK, op, "mt")
        if ylabel:
            sc.text(*frame_xy(cam, x0, y0 - 0.014), ylabel, size, "regular", layout.CAPTION_INK,
                    op, "lb")

    def curve(self, sc, place, steps, values, upto, ink, op, width=1.3, dot=True, every=1):
        """``values`` against ``steps`` drawn up to step ``upto`` (interpolated),
        ending in a glowing voxel unless ``dot`` is False; ``every`` thins the
        points drawn."""
        if op <= 0.01 or upto < steps[0]:
            return None
        steps, values = np.asarray(steps, float), np.asarray(values, float)
        k = int(np.searchsorted(steps, upto, side="right"))
        xs = np.append(steps[:k:every], min(upto, steps[-1]))
        ys = np.append(values[:k:every], np.interp(min(upto, steps[-1]), steps, values))
        prev = None
        for x, y in zip(xs, ys):
            p = place(self.fx(x), self.fy(y))
            if prev is not None:
                sc.line(prev, p, width=width, color=ink, opacity=0.9 * op, bias=0.0)
            prev = p
        if dot and prev is not None:
            sc.voxel(prev, ink, op, glow=0.3, shadow=False, reflect=False, size=0.6)
        return prev

    def mark(self, sc, place, x, ink, op, dashes=12, width=0.9):
        """A dashed vertical line at step ``x`` (an early-stopping point)."""
        if op <= 0.01:
            return
        f = self.fx(x)
        a, b = place(f, self.y1), place(f, self.y0)
        for i in range(dashes):
            u0, u1 = i / dashes, (i + 0.5) / dashes
            sc.line(a + (b - a) * u0, a + (b - a) * u1, width=width, color=ink, opacity=0.8 * op,
                    bias=0.0)

    def legend(self, sc, cam, entries, op, at, step=0.042, size=layout.MIN_TEXT, anchor="lm"):
        """``entries`` ``[(text, ink), ...]`` stacked from layout point ``at``."""
        if op <= 0.01:
            return
        fx, fy = at
        for n, (text, ink) in enumerate(entries):
            sc.text(*frame_xy(cam, fx, fy + n * step), text, size, "regular", ink, op, anchor)
