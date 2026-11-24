"""The kit the 100 "next" animations are drawn with: a mechanism on screen, then
what it does to the numbers.

Each of those animations is a :class:`Mechanism`: a flat :class:`~voxanim.diagram.Diagram`
in a few named phases, every phase its own drawing method and its own live
subtitle. The measurements come from the animation's *study*
(:mod:`voxanim.next_catalog` builds it: real training runs, GPT-2 probes and
exact counts, with their provenance); this module only draws. What it adds to
``Diagram`` is the vocabulary those animations share:

* things: :meth:`Mechanism.box`, :meth:`tokens`, :meth:`grid` (a matrix whose
  cells can be revealed, ringed and dimmed), :meth:`digit`,
* connections: :meth:`arc` and :meth:`arrow`, drawn progressively, and
  :meth:`pulse` (a value travelling along one),
* results: :meth:`chart` / :meth:`plot` (a :class:`~voxanim.losschart.LossChart`
  with round ticks) and :meth:`hbar`,
* text: :meth:`text`, :meth:`lines`.

Positions are layout fractions of the frame (0..1 across, 0..1 down; the
title block ends at about 0.26 and the title-safe area at 0.95); sizes are
reference pixels (the frame is 960 x 540).

The colours are the conv2d animations': :data:`INKS` starts red, blue, green.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .diagram import Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart

# The family's colours, as the conv2d animations use them for their filters
# (:data:`voxanim.color.FILTERS`): red, blue, green, then the rest.
INKS = [C.rgb(c) for c in (
    "#d84e4e",   # red
    "#4f60ee",   # blue
    "#3d9e45",   # green
    "#e8912a",   # amber
    "#8952eb",   # violet
    "#1f9e94",   # teal
    "#be3c98",   # magenta
    "#a1814b",   # ochre
)]
RED, BLUE, GREEN, AMBER, VIOLET, TEAL, MAGENTA, OCHRE = INKS
AXIS = (0.42, 0.44, 0.46)
GRID = (0.74, 0.75, 0.76)
PAPER = C.rgb("#f6f6f3")
TRACK = (0.78, 0.79, 0.80)
DIM = C.rgb("#c9cbcd")            # something present but not taking part
DIGIT_INK = C.rgb("#22252a")
WHITE = (1.0, 1.0, 1.0)
TEXT = layout.MIN_TEXT


def nice_ticks(lo, hi, target=4):
    """Round tick values covering ``lo..hi`` (at most ``target + 2`` of them)."""
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return [lo]
    raw = span / target
    mag = 10.0 ** np.floor(np.log10(raw))
    step = min((1, 2, 2.5, 5, 10), key=lambda m: abs(m * mag - raw)) * mag
    first = np.ceil(lo / step - 1e-9) * step
    ticks = [float(first + k * step) for k in range(int((hi - first) / step + 1e-9) + 1)]
    return ticks or [lo]


def fit(text, size, width):
    """``text`` in as many lines as it needs to stay within ``width`` (a
    fraction of the frame): :func:`layout.wrap` stops at two."""
    lines, line = [], ""
    for word in " ".join(str(text).split()).split(" "):
        trial = (line + " " + word).strip()
        if line and layout.text_width(trial, size) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def number(v):
    """A measured value as a short label."""
    v = float(v)
    if v == 0:
        return "0"
    if 1e6 <= abs(v) < 1e15:
        # 537M rather than 5.4e+08: an axis has room for four characters, not eleven
        for scale, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(v) >= scale:
                return "%.3g%s" % (v / scale, unit)
    if 10000 <= abs(v) < 1e6:
        return "{:,.0f}".format(float("%.3g" % v))      # 130,000 rather than 1.3e+05
    if abs(v) >= 1e15 or abs(v) < 0.001:
        return "%.2g" % v
    if abs(v) >= 100:
        return "%.0f" % v
    return ("%.3g" % v)


def heat(q, hot=RED, cold=BLUE):
    """Colours for values scaled to ``[-1, 1]``: paper at zero, ``hot`` for
    positive and ``cold`` for negative, deepening with magnitude. The square
    root lifts small values off the paper."""
    q = np.asarray(q, np.float32)[..., None]
    a = np.sqrt(np.abs(q))
    target = np.where(q >= 0, np.asarray(hot, np.float32), np.asarray(cold, np.float32))
    deep = target * 0.90
    col = PAPER + (target - PAPER) * np.minimum(a / 0.85, 1.0)
    return col + (deep - target) * np.clip((a - 0.85) / 0.15, 0.0, 1.0)


def soft(ink, t=0.55):
    """``ink`` moved towards the paper: a fill behind text in that ink."""
    return tuple(float(v) for v in C.mix(ink, PAPER, t))


def dark(ink, t=0.2):
    """``ink`` darkened enough to read as text on the grey background."""
    return tuple(float(v) * (1 - t) for v in ink)


class Mechanism(Diagram):
    """An animation in named phases, each drawn by ``phase_<n>`` (1-based).

    A subclass sets :attr:`TITLE` and :attr:`PHASES` (``(name, seconds)``
    pairs), reads what it needs from ``self.study`` in :meth:`setup`, and
    defines ``phase_1(sc, u, op)`` ... (``u``: 0..1 through that phase;
    ``op``: its opacity, which every draw call must be given) and
    :meth:`subtitle_for`. Phases crossfade; the loop closes on an empty frame.
    """

    TITLE = ""
    PHASES: tuple = ()
    FADE = 0.5
    # How far towards the viewer arrows (:meth:`arc`, :meth:`arrow`) are lifted, as a box's
    # ``lift``: lines and boxes are painted back to front, so an arrow on the paper (0)
    # goes under any box lifted above it (the Blocks series lifts them over everything).
    ARROW_LIFT = 0.0

    def __init__(self, study=None):
        super().__init__(self.TITLE or (study.title if study is not None else ""), "", 10.0)
        self.study = study
        self.hold, self.reset = 2.5, 2.0
        self.setup()
        t, self.spans = 0.0, []
        for _name, seconds in self.PHASES:
            self.spans.append((t, t + seconds))
            t += seconds
        self.finish(t)

    # ------------------------------------------------------------------ to override
    def setup(self):
        """Read the study; compute everything the phases draw."""

    def subtitle_for(self, k):
        """The live subtitle of phase ``k`` (0-based): what was measured."""
        return ""

    # ------------------------------------------------------------------ the frame
    def phase_at(self, t):
        """Index of the phase running at ``t``."""
        return int(min(max(np.searchsorted([a for a, _ in self.spans], t, side="right") - 1, 0),
                       len(self.spans) - 1))

    def draw(self, sc, t, reset):
        k = 0 if reset > 0 else self.phase_at(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        self.subtitle = self.subtitle_for(k)
        last = len(self.spans) - 1
        for i, (a, b) in enumerate(self.spans):
            op = smoothstep(a, a + self.FADE, t) * back
            if i < last:
                op *= 1.0 - smoothstep(b - self.FADE, b, t)
            if op > 0.01:
                u = float(np.clip((t - a) / (b - a), 0.0, 1.0))
                getattr(self, "phase_%d" % (i + 1))(sc, u, op)
        self.step_indicator(sc, k, [name for name, _ in self.PHASES])

    # ------------------------------------------------------------------ geometry
    def span(self, dfx=0.0, dfy=0.0):
        """A layout-fraction extent as reference px ``(dx, dy)``."""
        cam = self.cam
        return dfx * cam.zoom * cam.width, dfy * cam.zoom * cam.height

    def frac(self, dx=0.0, dy=0.0):
        """Reference px as a layout-fraction extent: the inverse of :meth:`span`."""
        cam = self.cam
        return dx / (cam.zoom * cam.width), dy / (cam.zoom * cam.height)

    # ------------------------------------------------------------------ text
    def text(self, sc, fx, fy, s, op, size=TEXT, anchor="mm", ink=None, strong=False):
        """``s`` at layout ``(fx, fy)``."""
        self.label(sc, self.P(float(fx), float(fy)), str(s), size, anchor=anchor, opacity=op,
                   ink=layout.STRONG_INK if strong else ink)

    def lines(self, sc, fx, fy, rows, op, size=TEXT, step=0.042, anchor="lm", width=None):
        """Rows of text (``str`` or ``(str, ink)``) stacked down from ``(fx, fy)``;
        ``width`` (frame fraction) wraps each. Returns the layout y after the last."""
        y = fy
        for row in rows:
            s, ink = row if isinstance(row, tuple) else (row, None)
            for line in (fit(s, size, width) if width else [s]):
                self.text(sc, fx, y, line, op, size, anchor, ink)
                y += step
        return y

    # ------------------------------------------------------------------ things
    def box(self, sc, fx, fy, w, h, fill, op, text=None, ink=None, border=None, size=TEXT,
            radius=6.0, glow=0.0, fill_op=1.0):
        """A rounded ``w x h`` px box centred at ``(fx, fy)``, with ``text`` in it."""
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, w, h, fill, op, border=border, radius=radius, glow=glow,
                  fill_op=fill_op)
        if text is not None:
            self.label(sc, c, str(text), size, opacity=op,
                       ink=layout.STRONG_INK if ink is None else ink)

    def tokens(self, sc, words, fx0, fx1, fy, op, inks=None, w=None, h=30.0, active=None,
               size=TEXT):
        """A row of token boxes spread from ``fx0`` to ``fx1``; returns their
        centre x fractions. ``inks``: one colour per token (default paper);
        ``active``: indices drawn at full strength, the others dimmed."""
        n = len(words)
        xs = [fx0 + (fx1 - fx0) * (i + 0.5) / n for i in range(n)]
        w = w if w is not None else min(self.span((fx1 - fx0) / n)[0] - 6.0, 64.0)
        for i, (word, x) in enumerate(zip(words, xs)):
            ink = None if inks is None else inks[i % len(inks)]
            on = active is None or i in active
            fill = PAPER if ink is None else soft(ink)
            self.box(sc, x, fy, w, h, fill, op * (1.0 if on else 0.35), word,
                     ink=None if ink is None else dark(ink, 0.35),
                     border=AXIS if ink is None else ink, size=size)
        return xs

    def grid(self, sc, values, fx, fy, w, h, op, reveal=1.0, vmax=None, ring=(), ring_ink=None,
             frame=True, lines=0.16, hot=RED, cold=BLUE, mask=False):
        """A matrix as a ``w x h`` px heatmap centred at ``(fx, fy)``: paper at
        zero, ``hot`` positive, ``cold`` negative. ``reveal`` (0..1) shows
        that fraction of its rows; ``ring``: ``(row, col)`` cells outlined.
        Returns ``cell(row, col) -> (fx, fy)``.

        ``mask``: the grid is a plain yes/no mask (who may read whom), not a map with a
        scale. Dark mode keeps a heat map as it is in light mode, since its caption reads
        "a deeper cell: more"; a mask it turns with the page, the allowed cells vivid on
        dark paper, where kept it stood as a white panel on the dark page."""
        a = np.atleast_2d(np.asarray(values, float))
        rows, cols = a.shape
        vmax = vmax if vmax is not None else max(float(np.abs(a).max()), 1e-10)
        rgb = heat(np.clip(a / vmax, -1, 1), hot, cold)
        shown = int(np.ceil(rows * float(np.clip(reveal, 0, 1)) - 1e-9))
        rgb[shown:] = PAPER * 0.97
        c = self.P(float(fx), float(fy))
        if frame:
            self.rect(sc, c, w + 8, h + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5,
                      lift=0.1)
        draw_image(self, sc, c, rgb.astype(np.float32), w, h, op,
                   grid=lines if min(w / cols, h / rows) >= 9 else 0,
                   keep_theme=False if mask else None)
        dw, dh = self.frac(w, h)

        def cell(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / cols, fy - dh / 2 + (r + 0.5) * dh / rows
        for r, k in ring:
            x, y = cell(r, k)
            ink = AMBER if ring_ink is None else ring_ink
            self.rect(sc, self.P(x, y), w / cols + 2, h / rows + 2, ink, op, border=ink,
                      border_w=2.2, radius=2, lift=0.7, fill_op=0.0)
        return cell

    def digit(self, sc, img, fx, fy, size, op, frame=None, ink=DIGIT_INK):
        """An image in [0, 1] (flat or square) as dark ink on pale paper, ``size``
        px wide, optionally framed in the colour ``frame``."""
        v = np.asarray(img, float)
        if v.ndim == 1:
            side = int(round(np.sqrt(v.size)))
            v = v.reshape(side, side)
        v = np.clip(v, 0, 1)[..., None]
        c = self.P(float(fx), float(fy))
        if frame is not None:
            self.rect(sc, c, size + 8, size + 8, frame, op, radius=5, lift=0.1)
        draw_image(self, sc, c, (PAPER * (1 - v) + np.asarray(ink) * v).astype(np.float32),
                   size, size, op, grid=0)

    # ------------------------------------------------------------------ connections
    def _curve_points(self, a, b, bend, n=24):
        (ax, ay), (bx, by) = a, b
        mx, my = (ax + bx) / 2, (ay + by) / 2 - bend
        s = np.linspace(0.0, 1.0, n)
        x = (1 - s) ** 2 * ax + 2 * s * (1 - s) * mx + s ** 2 * bx
        y = (1 - s) ** 2 * ay + 2 * s * (1 - s) * my + s ** 2 * by
        return x, y

    def arc(self, sc, a, b, ink, op, u=1.0, bend=0.08, width=2.2, head=True, lift=None):
        """A curved connection from layout point ``a`` to ``b`` (bulging up by
        ``bend``; negative: down), drawn as far as fraction ``u``, with an
        arrowhead at its end; ``lift``: towards the viewer (default
        :attr:`ARROW_LIFT`; 0 puts it on the paper, behind every box)."""
        if op <= 0.01 or u <= 0.0:
            return
        x, y = self._curve_points(a, b, bend)
        m = max(2, int(np.ceil(len(x) * min(u, 1.0))))
        pts = [self.P(float(px), float(py)) for px, py in zip(x[:m], y[:m])]
        pts = [p + self.view * (self.ARROW_LIFT if lift is None else lift) for p in pts]
        self.polyline(sc, pts, width, ink, op, bias=0.3)
        if head and m >= 2:
            self._head(sc, pts[-2], pts[-1], ink, op, width)

    def arrow(self, sc, a, b, ink, op, u=1.0, width=2.2, head=True, lift=None):
        """A straight connection from ``a`` to ``b`` (layout points), as :meth:`arc`."""
        if lift is None:            # a subclass's own arc() need not know lift
            self.arc(sc, a, b, ink, op, u, 0.0, width, head)
        else:
            self.arc(sc, a, b, ink, op, u, 0.0, width, head, lift=lift)

    def _head(self, sc, p, q, ink, op, width):
        sp, sq = self.px(p), self.px(q)
        d = sq - sp
        n = float(np.hypot(*d))
        if n < 1e-6:
            return
        d = d / n
        size = 5.0 + 1.6 * width
        for sign in (1, -1):
            wing = -d * size + sign * np.array([-d[1], d[0]]) * size * 0.5
            self.polyline(sc, [q, self.shift(q, wing[0], wing[1])], width, ink, op, bias=0.3)

    def pulse(self, sc, a, b, u, ink, op=1.0, bend=0.0, r=4.5):
        """A lit value travelling from ``a`` to ``b`` (layout points) along the
        :meth:`arc` of the same ``bend``; ``u`` in 0..1 (nothing outside it)."""
        if not 0.0 < u < 1.0 or op <= 0.01:
            return
        x, y = self._curve_points(a, b, bend, n=41)
        k = ease_in_out(u) * 40
        i = int(min(k, 39))
        f = k - i
        fade = min(1.0, u / 0.12, (1 - u) / 0.12)
        self.disk(sc, self.P(float(x[i] + (x[i + 1] - x[i]) * f),
                             float(y[i] + (y[i + 1] - y[i]) * f)),
                  r, ink, op * fade, glow=0.6, lift=1.6, outline=0.0)

    # ------------------------------------------------------------------ results
    def chart(self, box, xs, ys, pad=0.06, xlog=False, ylog=False):
        """A chart in the layout box ``((x0, x1), (top, bottom))`` scaled to
        hold every value of ``xs`` and ``ys``."""
        xs, ys = np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()
        lo, hi = float(ys.min()), float(ys.max())
        if hi - lo < 1e-12:
            lo, hi = lo - 0.5, hi + 0.5
        if not ylog:
            lo, hi = lo - (hi - lo) * pad, hi + (hi - lo) * pad
        return LossChart(box, (float(xs.min()), float(xs.max())), (lo, hi), xlog=xlog, ylog=ylog)

    def axes(self, sc, chart, op, xlabel="", ylabel="", xticks=None, yticks=None, grid=True):
        """``chart``'s axes with round ticks (or the ``(value, label)`` ticks given)."""
        if xticks is None:
            xticks = [(v, number(v)) for v in nice_ticks(*chart.x_range, 4)]
        if yticks is None:
            yticks = [(v, number(v)) for v in nice_ticks(*chart.y_range, 4)]
        if grid:
            for v, _ in yticks:
                y = chart.fy(v)
                self.polyline(sc, [self.P(chart.x0, y), self.P(chart.x1, y)], 1.0, GRID,
                              0.55 * op, bias=0.0)
        chart.axes(sc, self.cam, self.P, op * self._cap, xlabel, ylabel, xticks, yticks)

    def plot(self, sc, chart, xs, ys, ink, op, u=1.0, width=1.9, dot=True):
        """``ys`` against ``xs`` on ``chart``, drawn as far as fraction ``u`` of
        the x range, ending in a lit point. Returns that point's layout
        ``(fx, fy)`` (None before the curve starts)."""
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if op <= 0.01 or len(xs) == 0:
            return None
        upto = xs[0] + (xs[-1] - xs[0]) * float(np.clip(u, 0, 1))
        every = max(1, len(xs) // 200)
        chart.curve(sc, self.P, xs, ys, upto, ink, op, width=width, dot=False, every=every)
        end = chart.fx(upto), chart.fy(float(np.interp(upto, xs, ys)))
        if dot:
            self.disk(sc, self.P(*end), 5.5, ink, op, glow=0.45, lift=1.4, outline=0.0)
        return end

    def hbar(self, sc, fx, fy, length, ink, op, label="", value="", u=1.0, thick=20.0,
             track=None):
        """A horizontal bar growing right from ``(fx, fy)`` to ``length`` px (at
        ``u`` = 1), its ``label`` to the left and ``value`` at its end; ``track``
        (px) draws the full scale behind it."""
        at = self.P(float(fx), float(fy))
        if track:
            self.rect(sc, self.shift(at, track / 2, 0), track, thick, TRACK, 0.5 * op,
                      radius=4, lift=0.3)
        w = max(float(length) * smoothstep(0.0, 1.0, u), 0.1)
        self.rect(sc, self.shift(at, w / 2, 0), w, thick, ink, op, radius=min(4.0, w / 2))
        if label:
            self.label(sc, self.shift(at, -10, 0), str(label), TEXT, anchor="rm", opacity=op)
        if value != "":
            self.label(sc, self.shift(at, w + 8, 0), str(value), TEXT, anchor="lm", opacity=op,
                       ink=layout.STRONG_INK)
