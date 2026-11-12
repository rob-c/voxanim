"""The drawing the five Gaussian-process animations share.

:class:`GPDiagram` is a :class:`~voxanim.diagram.Diagram` with a 2-D plot of
functions of ``x`` in the diagram plane (its :class:`Plot`; an animation can
add a second one, the acquisition panel of Bayesian optimisation):

* :meth:`GPDiagram.axes` -- the frame, the zero line and the tick labels,
* :meth:`GPDiagram.band` -- the ``mu +- 2 sd`` band as four stacked translucent
  layers (``+-0.5, 1, 1.5, 2`` standard deviations: densest at the mean), one
  shaded quad whose shader reads the mean and standard deviation on the grid,
* :meth:`GPDiagram.curve` -- a function as a polyline (dashed for a truth
  that the model does not know), clipped to the plot,
* :meth:`GPDiagram.points` -- observations as dark disks,
* :meth:`GPDiagram.heat` -- a kernel matrix on a sheet of voxels, the signed
  ``div_colors`` scale (orange positive, blue negative, grey zero),

plus :class:`Beats`, a small timeline builder, and the palette
(:data:`MEAN`, :data:`SAMPLES`, :data:`OBS`, :data:`TRUTH`, :data:`ACQ`),
so the five look like one family.  All of it draws numbers computed by
:mod:`voxanim.gp`; nothing here is a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .diagram import DATA, EDGE, GRAD, Diagram
from .netanim import div_colors
from .text import num

# ------------------------------------------------------------------ palette

MEAN = DATA                                     # the posterior mean (teal)
BAND = C.rgb("#2f9d93")                         # the +-2 sd band (translucent teal)
SAMPLES = [C.rgb("#b2691f"), C.rgb("#6c5bd2"), C.rgb("#5f9a45"), C.rgb("#b84a8a")]
OBS = C.rgb("#2a2d31")                          # observations
TRUTH = (0.40, 0.42, 0.44)                      # the true function (dashed grey)
ACQ = GRAD                                      # the acquisition function (magenta)
FOCUS = (0.12, 0.36, 0.66)                      # the thing in play
PANEL = (0.975, 0.975, 0.97)                    # the plot's backdrop
PANEL_EDGE = (0.62, 0.64, 0.66)
BAND_LEVELS = (0.5, 1.0, 1.5, 2.0)              # the stacked layers, in standard deviations
BAND_OPACITY = 0.17                             # each layer's opacity (they add up)
TICK_PX = 5.0                                   # tick length, px
PLOT = (0.05, 0.31, 0.60, 0.93)                 # default plot box, layout fractions
LIFT_BAND, LIFT_CURVE, LIFT_POINT = 0.2, 0.5, 1.0   # draw order inside the plot


def _ink(c):
    """A text colour from a drawing colour, dark enough to read on the backdrop."""
    return tuple(float(v) for v in C.darken(c, 0.3))


MEAN_INK, ACQ_INK = _ink(MEAN), _ink(ACQ)


class Beats:
    """A timeline built beat by beat: ``add(name, seconds, gap)`` appends a
    phase after the previous one; ``Tm`` is the ``{name: (start, end)}`` the
    :class:`~voxanim.netanim.NetAnimation` reads."""

    def __init__(self, start=0.2):
        self.Tm, self.t = {}, start

    def add(self, name, seconds, gap=0.3):
        """Append phase ``name`` of ``seconds``; the next starts ``gap`` later.
        Returns ``(start, end)``."""
        self.Tm[name] = (self.t, self.t + seconds)
        self.t += seconds + gap
        return self.Tm[name]

    def pause(self, seconds):
        """Leave a gap before the next phase."""
        self.t += seconds


@dataclass
class Plot:
    """A plot's geometry: world centre ``c``, size ``w x h`` px, and the data
    ranges ``xr``, ``yr`` it shows."""

    c: np.ndarray
    w: float
    h: float
    xr: tuple
    yr: tuple

    def inside(self, y):
        """True where ``y`` is within the plot's range."""
        return (np.asarray(y) >= self.yr[0]) & (np.asarray(y) <= self.yr[1])


class GPDiagram(Diagram):
    """A diagram with a plot of functions of ``x``: ``xr`` and ``yr`` are the
    ranges shown in the box ``plot`` (layout fractions ``x0, y0, x1, y1``).
    Every drawing method takes ``plot=None`` for the main plot or another
    :class:`Plot` made with :meth:`make_plot`."""

    def __init__(self, title, xr, yr, plot=PLOT, azimuth=10.0, elevation=12.0):
        super().__init__(title, "", azimuth=azimuth, elevation=elevation)
        self.plot = self.make_plot(plot, xr, yr)
        self.xr, self.yr = self.plot.xr, self.plot.yr

    def make_plot(self, box, xr, yr):
        """A :class:`Plot` in the layout ``box`` ``(x0, y0, x1, y1)`` showing ``xr`` x ``yr``."""
        x0, y0, x1, y1 = box
        c = self.P((x0 + x1) / 2, (y0 + y1) / 2)
        w, h = (self.px(self.P(x1, y1)) - self.px(self.P(x0, y0))).tolist()
        return Plot(c, w, h, (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])))

    # ------------------------------------------------------------------ coordinates
    def pt(self, x, y, lift=0.0, plot=None):
        """World point of ``(x, y)`` in data units (not clipped)."""
        p = plot or self.plot
        u = (float(x) - p.xr[0]) / (p.xr[1] - p.xr[0]) - 0.5
        v = (float(y) - p.yr[0]) / (p.yr[1] - p.yr[0]) - 0.5
        return self.shift(p.c, u * p.w, -v * p.h) + self.view * lift

    def inside(self, y, plot=None):
        """True where ``y`` is within the plot's range."""
        return (plot or self.plot).inside(y)

    # ------------------------------------------------------------------ the plot itself
    def axes(self, sc, op, xlabel="x", ylabel="f(x)", xticks=None, yticks=None, zero=True,
             plot=None):
        """The plot's panel, its zero line and tick labels (``xticks``/``yticks``
        default to the ends of the ranges and 0)."""
        if op <= 0.01:
            return
        p = plot or self.plot
        xr, yr = p.xr, p.yr
        self.rect(sc, p.c, p.w + 2, p.h + 2, PANEL, op, border=PANEL_EDGE, border_w=1.0,
                  radius=2, lift=0.0)
        if zero and yr[0] < 0 < yr[1]:
            sc.line(self.pt(xr[0], 0, 0.1, p), self.pt(xr[1], 0, 0.1, p), width=1.0,
                    color=EDGE, opacity=0.45 * op, bias=0.0)
        xticks = [xr[0], (xr[0] + xr[1]) / 2, xr[1]] if xticks is None else xticks
        yticks = [yr[0], 0.0, yr[1]] if yticks is None else yticks
        for x in xticks:
            q = self.pt(x, yr[0], plot=p)
            sc.line(q, self.shift(q, 0, TICK_PX), width=1.0, color=EDGE, opacity=0.8 * op,
                    bias=0.0)
            self.label(sc, q, num(x, 0), layout.MIN_TEXT, anchor="mt", dy=TICK_PX + 3,
                       opacity=op, ink=layout.NOTE_INK)
        for y in yticks:
            if not yr[0] <= y <= yr[1]:
                continue
            q = self.pt(xr[0], y, plot=p)
            sc.line(q, self.shift(q, -TICK_PX, 0), width=1.0, color=EDGE, opacity=0.8 * op,
                    bias=0.0)
            self.label(sc, q, num(y, 0), layout.MIN_TEXT, anchor="rm", dx=-(TICK_PX + 4),
                       opacity=op, ink=layout.NOTE_INK)
        if xlabel:
            self.label(sc, self.pt(xr[1], yr[0], plot=p), xlabel, layout.MIN_TEXT,
                       anchor="lb", dx=9, dy=2, opacity=op, ink=layout.NOTE_INK)
        if ylabel:
            self.label(sc, self.pt(xr[0], yr[1], plot=p), ylabel, layout.MIN_TEXT,
                       anchor="lt", dx=6, dy=4, opacity=op, ink=layout.NOTE_INK)

    def band(self, sc, xs, mu, sd, op, color=BAND, levels=BAND_LEVELS, lift=LIFT_BAND,
             plot=None):
        """The ``mu +- k sd`` band, for ``k`` in ``levels``, as stacked translucent
        layers: one quad over the whole plot whose shader interpolates ``mu``
        and ``sd`` on the grid ``xs`` and stacks :data:`BAND_OPACITY` per layer."""
        if op <= 0.01:
            return
        p = plot or self.plot
        xs, mu, sd = (np.asarray(v, np.float64) for v in (xs, mu, sd))
        c = p.c + self.view * lift
        M, w, h = self.Mr, p.w, p.h
        xr, yr = p.xr, p.yr
        color = np.asarray(color, np.float32)

        def shader(P):
            q = (P - c) @ M.T                           # px from the plot centre
            x = xr[0] + (q[..., 0] / w + 0.5) * (xr[1] - xr[0])
            y = yr[1] - (q[..., 1] / h + 0.5) * (yr[1] - yr[0])
            m, s = np.interp(x, xs, mu), np.interp(x, xs, sd)
            # one output pixel, in data units of y, for anti-aliased edges
            pf = (yr[1] - yr[0]) / h
            if q.ndim == 3 and q.shape[0] > 1:
                pf *= max(float(np.hypot(*(q[0, 1] - q[0, 0]))), 1e-6)
            d = np.abs(y - m)
            a = np.zeros(q.shape[:-1], np.float32)
            for k in levels:
                a += np.clip((k * s - d) / pf + 0.5, 0.0, 1.0)
            alpha = np.clip(a * BAND_OPACITY, 0.0, 1.0) * op
            return np.broadcast_to(color, q.shape[:-1] + (3,)), alpha, 0.0
        self._billboard(sc, c, w / 2, h / 2, shader)

    def fill(self, sc, xs, ys, op, color, lift=LIFT_BAND, plot=None):
        """The area between ``y = 0`` and the curve ``ys`` (an acquisition
        function), one translucent quad like :meth:`band`."""
        if op <= 0.01:
            return
        p = plot or self.plot
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        c = p.c + self.view * lift
        M, w, h = self.Mr, p.w, p.h
        xr, yr = p.xr, p.yr
        color = np.asarray(color, np.float32)

        def shader(P):
            q = (P - c) @ M.T
            x = xr[0] + (q[..., 0] / w + 0.5) * (xr[1] - xr[0])
            y = yr[1] - (q[..., 1] / h + 0.5) * (yr[1] - yr[0])
            top = np.interp(x, xs, ys)
            pf = (yr[1] - yr[0]) / h
            if q.ndim == 3 and q.shape[0] > 1:
                pf *= max(float(np.hypot(*(q[0, 1] - q[0, 0]))), 1e-6)
            a = np.clip((top - y) / pf + 0.5, 0.0, 1.0) * np.clip(y / pf + 0.5, 0.0, 1.0)
            return np.broadcast_to(color, q.shape[:-1] + (3,)), a * op, 0.0
        self._billboard(sc, c, w / 2, h / 2, shader)

    def curve(self, sc, xs, ys, color, width=2.0, op=1.0, dashed=False, lift=LIFT_CURVE,
              upto=None, plot=None):
        """``ys`` against ``xs`` as a polyline, clipped to the plot's range.
        ``dashed`` draws every other segment; ``upto`` (0..1) draws only the
        first part of the curve, with its end interpolated."""
        if op <= 0.01:
            return
        p = plot or self.plot
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        if upto is not None:
            n = len(xs) - 1
            k = int(np.floor(upto * n))
            if k < n:
                f = upto * n - k
                xs = np.concatenate([xs[:k + 1], [xs[k] + (xs[k + 1] - xs[k]) * f]])
                ys = np.concatenate([ys[:k + 1], [ys[k] + (ys[k + 1] - ys[k]) * f]])
            if len(xs) < 2:
                return
        for a, b in _clipped(xs, ys, p.yr):
            pts = [self.pt(x, y, lift, p) for x, y in zip(a, b)]
            for i in range(len(pts) - 1):
                if dashed and i % 2:
                    continue
                sc.line(pts[i], pts[i + 1], width=width, color=color, opacity=op, bias=0.2)

    def points(self, sc, X, y, op, color=OBS, r=4.2, lift=LIFT_POINT, each=None, plot=None):
        """Observations as disks; ``each`` gives a per-point opacity."""
        p = plot or self.plot
        for i, (x, v) in enumerate(zip(X, y)):
            o = op if each is None else op * each[i]
            if o > 0.01 and p.inside(v):
                self.disk(sc, self.pt(x, v, lift, p), r, color, o, outline=0.3, lift=0.0)

    def marker(self, sc, x, y, op, color=FOCUS, r=6.0, lift=LIFT_POINT + 0.2, plot=None):
        """A ringed marker (the point in play)."""
        p = plot or self.plot
        if op > 0.01 and p.inside(y):
            self.disk(sc, self.pt(x, y, lift, p), r, (1.0, 1.0, 1.0), op, ring=color,
                      ring_w=2.2, ring_op=1.0, lift=0.0)

    def vline(self, sc, x, op, color=FOCUS, width=1.2, lift=LIFT_CURVE, plot=None):
        """A vertical guide at ``x`` across the plot."""
        p = plot or self.plot
        if op > 0.01:
            sc.line(self.pt(x, p.yr[0], lift, p), self.pt(x, p.yr[1], lift, p), width=width,
                    color=color, opacity=op, bias=0.2)

    # ------------------------------------------------------------------ kernel matrices
    @staticmethod
    def heat(sheet, K, vmax, mask=None, glow=None):
        """Fill ``sheet`` with the matrix ``K`` on the signed colour scale (``vmax``
        at full colour); ``mask`` hides cells, ``glow`` (n,) lights them."""
        K = np.asarray(K, np.float64).ravel()
        sheet.fill(div_colors(K, vmax), glow)
        if mask is not None:
            sheet.show(np.asarray(mask).ravel())

    def sheet_label(self, sc, sheet, text, op, above=False, size=layout.MIN_TEXT, ink=None):
        """A caption under (or above) a sheet, through the caption helper so it
        clears the voxels."""
        self.caption(sc, sheet, text, op, size=size, above=above, ink=ink)


def _clipped(xs, ys, yr):
    """The runs of a curve inside the range ``yr``, with the crossing points
    added, as ``[(xs, ys)]``."""
    lo, hi = yr
    runs, cur_x, cur_y = [], [], []

    def cross(i, level):
        f = (level - ys[i]) / (ys[i + 1] - ys[i])
        return xs[i] + (xs[i + 1] - xs[i]) * f

    for i in range(len(xs)):
        if lo <= ys[i] <= hi:
            if not cur_x and i > 0:                    # entering: start at the crossing
                level = lo if ys[i - 1] < lo else hi
                cur_x.append(cross(i - 1, level))
                cur_y.append(level)
            cur_x.append(xs[i])
            cur_y.append(ys[i])
        elif cur_x:                                    # leaving: end at the crossing
            level = lo if ys[i] < lo else hi
            cur_x.append(cross(i - 1, level))
            cur_y.append(level)
            runs.append((cur_x, cur_y))
            cur_x, cur_y = [], []
    if len(cur_x) > 1:
        runs.append((cur_x, cur_y))
    return [r for r in runs if len(r[0]) > 1]


def stagger(u, j, n, width=0.5):
    """Progress of item ``j`` of ``n`` when they start one after another and each
    takes a fraction ``width`` of the whole."""
    span = 1.0 - width
    start = span * j / max(1, n - 1) if n > 1 else 0.0
    return float(np.clip((u - start) / width, 0.0, 1.0))


def log_lerp(a, b, u):
    """``a`` to ``b`` geometrically (for positive quantities such as a lengthscale)."""
    return float(a * (b / a) ** u)
