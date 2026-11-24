"""Animations 275-276 of "Blocks and code", "From simulation to a result" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_zh`: made-up simulated events
(``examples/weights_and_scale_factors.py``), weighted, and corrected to made-up data.

* :class:`EventWeights` (``event-weights``): three simulated samples, a signal and two
  backgrounds, as a stacked histogram of their events; each event's weight, cross-section x
  luminosity / events made; the histogram weighted, each sample adding up to what is
  expected; the weight in the loss (a classifier trained with and without it, its score
  drawn over the histogram); in an efficiency; and negative weights, drawn below the axis,
  with the effective number of events they leave.
* :class:`ScaleFactors` (``scale-factors``): the cut chosen on simulation (it keeps 80% of
  the simulated signal); a control region of signal-like probes, in simulation and in the
  made-up data, which spread wider; how many pass the cut in each; the scale factor, their
  ratio, with its error; the signal region's count, simulation against data, before and
  after the simulated signal is scaled.

The alphabet: signal red, bkg 1 blue, bkg 2 teal (as in "The overtraining check"); the data
black dots with error bars; the loss, and the score trained with the weights, violet; the
score trained without them grey, dashed. Histograms are bars on a chart (bins of x), a
negative weight's bar hangs below the zero line.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    INK,
    LOSS,
    QUIET,
    SIZE,
    Blocks,
    Code,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, BLUE, RED, TEAL, dark, soft

ON = "made-up data"
INKS = (RED, BLUE, TEAL)                   # signal, bkg 1, bkg 2
TEXT_INKS = tuple(dark(c, 0.3) for c in INKS)
STACK = (1, 2, 0)                          # bottom to top: bkg 1, bkg 2, signal
LOSS_INK = dark(LOSS, 0.3)
PLAIN = (0.55, 0.57, 0.59)                 # the score trained without weights
DATA_INK = INK

# ---------------------------------------------------------------------- the layout
CHART = ((0.135, 0.465), (0.435, 0.645))   # the chart: (x0, x1), (top, bottom)
NOTE_Y = 0.368                             # a note over the chart, on the right
ROW_Y = (0.715, 0.76, 0.805, 0.85)         # the rows under the chart
CHIP_X, NAME_X = 0.07, 0.083               # a row's colour chip and its first word
COLS = (0.225, 0.31, 0.375, 0.465)         # 275's table: sigma, N, w, sum w (right edges)
KEY = (0.15, 0.163, (0.455, 0.491, 0.527))  # the key in the chart: chip x, name x, rows y


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Sim(Blocks):
    """What the two share: time, the chart, histograms, rows of text and their chips."""

    STATION = None

    def setup(self):
        self.meta = self.study.metadata

    def station_for(self, k):
        return None

    # ------------------------------------------------------------------ time
    def time_of(self, k, u):
        a, b = self.spans[k]
        return float(a + (b - a) * u)

    def since(self, t0, width=0.4):
        return smoothstep(t0, t0 + width, getattr(self, "_t", 0.0))

    def at_u(self, k, u, width=0.4):
        """0 before ``u`` of phase ``k``, 1 from ``width`` s later (by the clock)."""
        return self.since(self.time_of(k, u), width)

    def prog(self, k, u0, u1):
        t0, t1 = self.time_of(k, u0), self.time_of(k, u1)
        return ease_in_out(float(np.clip((getattr(self, "_t", 0.0) - t0) / (t1 - t0), 0, 1)))

    def now(self):
        return self.where(getattr(self, "_t", 0.0))

    # ------------------------------------------------------------------ the chart
    def frame(self, sc, ch, op, xticks, yticks, ylabel=""):
        """The chart's two axes with ticks ``((value, label), ...)`` (those in range); no grid
        lines, so that the key and the labels inside the chart sit clear."""
        if op <= 0.01:
            return
        lo, hi = ch.y_range
        ys = [(v, s) for v, s in yticks if lo - 1e-9 <= v <= hi + 1e-9]
        self.axes(sc, ch, op, ylabel, xticks=[], yticks=ys, grid=False)
        for v, s in xticks:
            f = ch.fx(v)
            self.polyline(sc, [self.P(f, ch.y1), self.P(f, ch.y1 + 0.006)], 1.0, AXIS,
                          0.7 * op, bias=0.0)
            self.text(sc, f, ch.y1 + 0.026, s, op, SIZE, ink=layout.NOTE_INK)
        if ylabel:
            self.text(sc, ch.x0, ch.y0 - 0.02, ylabel, op, SIZE, "lm", ink=layout.CAPTION_INK)
        if lo < 0 < hi:                    # the zero line, when the chart goes below it
            y = ch.fy(0.0)
            self.polyline(sc, [self.P(ch.x0, y, 0.6), self.P(ch.x1, y, 0.6)], 1.4, AXIS,
                          0.9 * op, bias=0.2)

    def axes(self, sc, ch, op, ylabel="", xticks=None, yticks=None, grid=True):
        super().axes(sc, ch, op, "", "", xticks=xticks, yticks=yticks, grid=grid)

    def bars(self, sc, ch, edges, values, ink, op, base=None, fill=0.85, lift=0.4):
        """A histogram: a bar per bin from ``base`` (default 0) by ``values`` (up, or down when
        negative); returns the tops."""
        values = np.asarray(values, float)
        base = np.zeros_like(values) if base is None else np.asarray(base, float)
        tops = base + values
        if op <= 0.01:
            return tops
        for i, (b0, b1) in enumerate(zip(base, tops, strict=True)):
            y0, y1 = ch.fy(min(b0, b1)), ch.fy(max(b0, b1))
            if y0 - y1 < 0.0012:
                continue
            x0, x1 = ch.fx(float(edges[i])), ch.fx(float(edges[i + 1]))
            w, h = self.span(x1 - x0, y0 - y1)
            self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), max(w - 1.2, 1.0), h, ink, op,
                      radius=1.0, lift=lift, fill_op=fill)
        return tops

    def outline(self, sc, ch, edges, values, ink, op, width=2.0):
        """A histogram's outline, a step line in front of the bars."""
        if op <= 0.01:
            return
        pts = []
        for i, v in enumerate(values):
            y = ch.fy(float(v))
            pts += [(ch.fx(float(edges[i])), y), (ch.fx(float(edges[i + 1])), y)]
        self.polyline(sc, [self.P(x, y, self.ARROW_LIFT) for x, y in pts], width, ink, op,
                      bias=0.3)

    def vline(self, sc, ch, x, ink, op, label=None, n=14, width=2.0):
        """A dashed vertical line at ``x`` over the chart (the cut), named at its top."""
        if op <= 0.01:
            return
        f = ch.fx(x)
        for i in range(n):
            a = ch.y1 + (ch.y0 - ch.y1) * i / n
            b = ch.y1 + (ch.y0 - ch.y1) * (i + 0.55) / n
            self.polyline(sc, [self.P(f, a, self.ARROW_LIFT), self.P(f, b, self.ARROW_LIFT)],
                          width, ink, op, bias=0.3)
        if label:                          # to its left, at the top: clear of the bars
            self.text(sc, f - 0.006, ch.y0 + 0.015, label, op, SIZE, "rm", ink=ink)

    def shade(self, sc, ch, x0, op, ink=AXIS):
        """The part of the chart right of ``x0`` (what passes the cut), lightly filled."""
        if op <= 0.01:
            return
        a = ch.fx(x0)
        w, h = self.span(ch.x1 - a, ch.y1 - ch.y0)
        self.rect(sc, self.P((a + ch.x1) / 2, (ch.y0 + ch.y1) / 2), w, h, soft(ink, 0.5), op,
                  radius=0.0, lift=0.05, fill_op=0.3)

    def chip(self, sc, fx, fy, ink, op, kind="box"):
        """A row's colour chip: a box (a sample), a dot (data), a line or a dashed line."""
        if op <= 0.01:
            return
        if kind == "box":
            self.rect(sc, self.P(fx, fy), 12.0, 12.0, ink, op, radius=2.5, lift=0.4,
                      fill_op=0.85)
        elif kind == "dot":
            self.disk(sc, self.P(fx, fy), 4.0, ink, op, lift=1.5, outline=0.1)
        else:
            dx = self.frac(8.0)[0]
            parts = ((-1, -0.25), (0.25, 1)) if kind == "dashed" else ((-1, 1),)
            for a, b in parts:
                self.polyline(sc, [self.P(fx + a * dx, fy, 1.4), self.P(fx + b * dx, fy, 1.4)],
                              2.4, ink, op, bias=0.3)

    def row(self, sc, i, text, op, ink=INK, chip=None, chip_kind="box", strong=False,
            size=SIZE):
        """Row ``i`` under the chart: an optional chip, then ``text``."""
        if op <= 0.01 or not text:
            return
        y = ROW_Y[i]
        if chip is not None:
            self.chip(sc, CHIP_X, y, chip, op, chip_kind)
        self.text(sc, NAME_X, y, text, op, size, "lm", ink=None if strong else ink,
                  strong=strong)

    def key(self, sc, op, dim=(1.0, 1.0, 1.0)):
        """The samples' key, in the chart's top left corner (each entry faded by ``dim``)."""
        cx, nx, ys = KEY
        for j, y in enumerate(ys):
            a = op * (0.25 + 0.75 * dim[j])
            self.chip(sc, cx, y, INKS[j], a)
            self.text(sc, nx, y, self.meta["names"][j], a, SIZE, "lm", ink=TEXT_INKS[j])

    def errbar(self, sc, fx, fy0, fy1, ink, op, cap=5.0, width=1.8):
        """A vertical error bar from layout y ``fy0`` to ``fy1`` at ``fx``, with caps."""
        if op <= 0.01:
            return
        dx = self.frac(cap)[0]
        lift = self.ARROW_LIFT
        self.polyline(sc, [self.P(fx, fy0, lift), self.P(fx, fy1, lift)], width, ink, op,
                      bias=0.3)
        for y in (fy0, fy1):
            self.polyline(sc, [self.P(fx - dx, y, lift), self.P(fx + dx, y, lift)], width, ink,
                          op, bias=0.3)

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the title)."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths, strict=True)):
            ink = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", ink, 1.0, "la")
            x += w + gap


# ====================================================================== 275
class EventWeights(Sim):
    """The samples counted; a weight each; the histogram weighted; the weights in the loss;
    in an efficiency; negative weights."""

    TITLE = "Event weights"
    PHASES = (("samples", 10.0), ("a weight", 11.0), ("weighted", 11.0), ("the loss", 11.0),
              ("efficiency", 8.5), ("negative", 9.5))
    LAST = "The overtraining check"
    NEXT = "Simulation is not data: scale factors"
    RANGES = ((0.0, 18000.0), (0.0, 9000.0), (-2000.0, 6500.0))   # the chart's y, in turn
    YTICKS = tuple((v, f"{v:,}") for v in (0, 5000, 10000, 15000))
    GROW = ((0.08, 0.2), (0.26, 0.38), (0.44, 0.56))   # phase 1: bkg 1, bkg 2, signal grow
    MORPH = (0.1, 0.45)          # phase 3: counted to weighted
    RESCALE = (0.66, 0.84)       # phase 3: the chart's y range halves
    FADE_OTHERS = (0.05, 0.2)    # phase 6: signal and bkg 1 leave
    LOWER = (0.2, 0.34)          # phase 6: the chart makes room below zero
    SPLIT = (0.37, 0.52)         # phase 6: bkg 2's bars split into + and -

    def setup(self):
        super().setup()
        m = self.meta
        self.edges = np.asarray(m["edges"])
        self.counts = np.asarray(m["counts"])
        self.weighted = np.asarray(m["weighted"])
        self.grid = np.asarray(m["grid"])
        self.CODE = (
            Code("ws-samples", run=((0.0, 1), (0.08, 3), (0.26, 4), (0.44, 2))),
            Code(("ws-samples", "ws-weight"), run=((0.0, 0), (0.3, 6), (0.5, 7))),
            Code("ws-hist", run=((0.0, 2), (0.5, 4)), out=((0.5, m["printed_hist"]),)),
            Code("ws-loss", run=((0.0, 0), (0.12, 1), (0.25, 3), (0.45, 4), (0.55, 5))),
            Code(("ws-cut", "ws-eff"), run=((0.0, 0), (0.3, 1), (0.42, 2), (0.62, 3)),
                 out=((0.62, m["printed_eff"]),)),
            Code("ws-negative", run=((0.0, 0), (0.2, 2), (0.4, 3), (0.72, 4))))
        self.EVIDENCE = (
            measured("the samples' histograms", ON),
            exact(m["worked"]),
            exact("each sample's weights add up to σ × L"),
            measured("trained twice, %s steps each" % format(m["steps"], ","), ON),
            measured("background passing the cut", ON),
            exact("(12,000)² ÷ (10,000 × 2²) = 3,600"))

    def station_for(self, k):
        return "loss" if k == 3 else None

    def heading_for(self, k, u):
        return ("three made-up simulated samples",
                "a weight: w = σ × L ÷ N",
                "weighted: each adds up to σ × L",
                "in the loss: cross-entropy × w",
                "in an efficiency: weighted shares",
                "negative weights: fewer events' worth")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Made-up simulated events of one variable x: a signal and two backgrounds, made "
            "%s, %s and %s at a time." % tuple(m["n_text"]),
            "Each event counts for σ × L ÷ N real ones: cross-section times luminosity (%g "
            "fb⁻¹), over the events made." % m["L"],
            "Weighted, each sample adds up to σ × L, the events expected: the signal is "
            "%s of them, not %s." % (m["share_weighted_text"], m["share_unweighted_text"]),
            "In the loss each event's cross-entropy counts w times. The mean score is then "
            "%s, the real share, not %s." % (m["mean_weighted_text"],
                                              m["mean_unweighted_text"]),
            "Background passing the cut x > %s: counted, %s; weighted, %s, since bkg 2, "
            "which passes more, counts more." % (m["cut_text"], m["eff_counted_text"],
                                                 m["eff_weighted_text"]),
            "Some generators make negative weights: 2,000 of bkg 2's events count −2. The sum "
            "is still 12,000, but worth only %s events." % m["n_eff_text"],
        ][k]

    # ------------------------------------------------------------------ the state at t
    def y_range(self):
        r0, r1, r2 = (np.array(r) for r in self.RANGES)
        a = self.prog(2, *self.RESCALE)
        b = self.prog(5, *self.LOWER)
        lo, hi = r0 + (r1 - r0) * a
        if b > 0:
            lo, hi = r1 + (r2 - r1) * b
        return float(lo), float(hi)

    def chart(self):
        return LossChart(CHART, (0.0, 1.6), self.y_range())

    def heights(self):
        """Each sample's bar heights now: grown (phase 1), counted, then weighted."""
        k, _u = self.now()
        g = [self.prog(0, *self.GROW[j]) if k == 0 else 1.0 for j in range(3)]
        order = {1: 0, 2: 1, 0: 2}          # which grows when: bkg 1, bkg 2, signal
        m = self.prog(2, *self.MORPH)
        out = []
        for j in range(3):
            h = self.counts[j] + (self.weighted[j] - self.counts[j]) * m
            out.append(h * g[order[j]])
        return out

    # ------------------------------------------------------------------ steady
    def steady(self, sc, k, u, op):
        m = self.meta
        ch = self.chart()
        xticks = [(0.0, "0"), (0.5, "0.5"), (1.0, "1"), (1.5, "1.5")]
        self.frame(sc, ch, op, xticks, self.YTICKS, "events per bin of x")
        self.text(sc, ch.x1 + 0.004, ch.y1 + 0.026, "x", op, SIZE, "lm", ink=layout.NOTE_INK)
        hs = self.heights()
        others = 1.0 - self.prog(5, *self.FADE_OTHERS)     # signal and bkg 1, in phase 6
        split = self.prog(5, *self.SPLIT)
        neg = m["neg"]
        b1 = self.bars(sc, ch, self.edges, hs[1], INKS[1], op * others)
        if split > 0:
            # bkg 2 with negative weights: + up from zero, - down from zero
            top2 = hs[2] + (np.asarray(neg["hist_pos"]) - hs[2]) * split
            self.bars(sc, ch, self.edges, top2, INKS[2], op)
            self.bars(sc, ch, self.edges, np.asarray(neg["hist_neg"]) * split, INKS[2], op,
                      fill=0.4)
        else:
            top2 = self.bars(sc, ch, self.edges, hs[2], INKS[2], op, base=b1 * others)
        self.bars(sc, ch, self.edges, hs[0], INKS[0], op * others, base=top2)
        # the key: from the loss on (the table names the samples before); in phase 6 only
        # bkg 2 stays
        self.key(sc, op * self.kept(3, 5), dim=(others, others, 1.0))
        # the table: phases 1-3
        self.table(sc, k, u, op * self.kept(0, 2))
        # the score curves: phase 4
        c = op * self.kept(3, 3)
        if c > 0.01:
            self.score_curve(sc, ch, m["score_unweighted"], PLAIN, c * self.at_u(3, 0.3),
                             dashed=True)
            self.score_curve(sc, ch, m["score_weighted"], LOSS, c * self.at_u(3, 0.55))
            self.text(sc, ch.x1, ch.y0 - 0.02, "score, 0 to 1", c * self.at_u(3, 0.3), SIZE,
                      "rm", ink=LOSS_INK)

    def score_curve(self, sc, ch, ys, ink, op, dashed=False):
        if op <= 0.01:
            return
        pts = [(ch.fx(x), ch.y1 + (ch.y0 - ch.y1) * float(s)) for x, s in zip(self.grid, ys,
                                                                             strict=True)]
        if not dashed:
            self.polyline(sc, [self.P(x, y, self.ARROW_LIFT) for x, y in pts], 2.6, ink, op,
                          bias=0.3)
            return
        for i in range(0, len(pts) - 4, 8):
            self.polyline(sc, [self.P(x, y, self.ARROW_LIFT) for x, y in pts[i:i + 5]], 2.2,
                          ink, op, bias=0.3)

    def table(self, sc, k, u, op):
        """Phases 1-3: a row per sample (its chip and name), and the columns as they come:
        events made; cross-section and weight; the sum of the weights."""
        if op <= 0.01:
            return
        m = self.meta
        order = {1: 0, 2: 1, 0: 2}
        cols_on = (self.at_u(1, 0.1), self.at_u(0, 0.05), self.at_u(1, 0.5), self.at_u(2, 0.5))
        heads = ("σ (fb)", "N made", "w", "Σw")
        for c, (x, h) in enumerate(zip(COLS, heads, strict=True)):
            self.text(sc, x, ROW_Y[0], h, op * cols_on[c], SIZE, "rm", ink=QUIET)
        self.text(sc, NAME_X, ROW_Y[0], "L = %g fb⁻¹" % m["L"], op * self.at_u(1, 0.1), SIZE,
                  "lm", ink=QUIET)
        for j in range(3):
            y = ROW_Y[1 + j]
            a = op * self.at_u(0, self.GROW[order[j]][0])
            self.chip(sc, CHIP_X, y, INKS[j], a)
            self.text(sc, NAME_X, y, m["names"][j], a, SIZE, "lm", ink=TEXT_INKS[j])
            vals = (m["sigma_text"][j], m["n_text"][j], m["weights_text"][j], m["sums_text"][j])
            for c, (x, v) in enumerate(zip(COLS, vals, strict=True)):
                w = cols_on[c] if c != 1 else 1.0
                if c == 3:
                    w *= self.at_u(2, 0.5 + 0.06 * j)
                self.text(sc, x, y, v, a * w, SIZE, "rm", ink=INK, strong=c >= 2)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, CHART[0][1], NOTE_Y, "%s simulated events" % format(m["n_total"], ","),
                  on(u, 0.64, op), SIZE, "rm", strong=True)

    def phase_2(self, sc, u, op):
        self.text(sc, CHART[0][1], NOTE_Y, "signal: " + self.meta["worked"], on(u, 0.6, op),
                  SIZE, "rm", strong=True)

    def phase_3(self, sc, u, op):
        m = self.meta
        self.text(sc, CHART[0][1], NOTE_Y, "%s expected in all" % format(
            int(m["expected_total"]), ","), on(u, 0.72, op), SIZE, "rm", strong=True)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.row(sc, 0, "loss = Σ w × cross-entropy ÷ Σ w", on(u, 0.08, op), ink=LOSS_INK)
        self.row(sc, 1, "no weights: mean score " + m["mean_unweighted_text"], on(u, 0.3, op),
                 ink=INK, chip=PLAIN, chip_kind="dashed")
        self.row(sc, 2, "weights: mean score " + m["mean_weighted_text"], on(u, 0.55, op),
                 ink=LOSS_INK, chip=LOSS, chip_kind="line")
        self.row(sc, 3, "= signal's share: %s made, %s expected" % (
            m["share_unweighted_text"].replace(".0%", "%"), m["share_weighted_text"]),
            on(u, 0.75, op), ink=INK)

    def phase_5(self, sc, u, op):
        m = self.meta
        ch = self.chart()
        self.shade(sc, ch, m["cut"], on(u, 0.18, op))
        self.vline(sc, ch, m["cut"], INK, on(u, 0.08, op), label="cut")
        self.row(sc, 0, "cut: x > " + m["cut_text"], on(u, 0.08, op), strong=True)
        self.row(sc, 1, "background passing the cut:", on(u, 0.3, op))
        self.row(sc, 2, m["eff_counted_row"], on(u, 0.42, op))
        self.row(sc, 3, m["eff_weighted_row"], on(u, 0.62, op), strong=True)

    def phase_6(self, sc, u, op):
        m = self.meta
        ch = self.chart()
        if u > self.SPLIT[1]:
            self.outline(sc, ch, self.edges, m["neg"]["net"], INK, on(u, 0.58, op))
        rows = m["neg_rows"]
        self.row(sc, 0, "bkg 2, made with negative weights:", on(u, 0.2, op), ink=TEXT_INKS[2],
                 chip=INKS[2])
        self.row(sc, 1, rows[0], on(u, 0.4, op))
        self.row(sc, 2, "the outline: the sum, %s as before" % format(round(m["neg"]["total"]),
                                                                      ","), on(u, 0.58, op))
        self.row(sc, 3, "%s of %s" % (rows[2], format(self.meta["n"][2], ",")),
                 on(u, 0.7, op), strong=True)


# ====================================================================== 276
class ScaleFactors(Sim):
    """The cut; the control region's probes; how many pass; the scale factor; the signal
    region before and after."""

    TITLE = "Simulation is not data: scale factors"
    PHASES = (("the cut", 9.0), ("control", 10.0), ("passing", 10.0), ("factor", 11.0),
              ("signal region", 10.0), ("corrected", 11.0))
    LAST = "Event weights"
    NEXT = "Unfolding: running the smearing backwards"
    X_RANGE = (0.5, 1.5)
    SIGNAL_Y = (0.0, 4200.0)
    SHARE_Y = (0.0, 0.21)
    SR_Y = (4400.0, 5300.0)       # the signal region's count: a zoomed number line
    SR_X = (0.25, 0.42)           # its two columns: simulation, data
    SWAP = (0.04, 0.2)            # phase 2: the signal's histogram gives way to the probes
    TO_SR = (0.08, 0.2)           # phase 5: the chart gives way to the signal region
    MOVE = (0.2, 0.45)            # phase 6: the simulation's count comes down

    def setup(self):
        super().setup()
        m = self.meta
        self.sig_edges = np.asarray(m["signal_edges"])
        self.probe_edges = np.asarray(m["probe_edges"])
        self.CODE = (
            Code("ws-cut", run=((0.0, 0),)),
            Code(("ws-probes", "ws-eff-cr"), run=((0.0, 0), (0.45, 1))),
            Code(("ws-probes", "ws-eff-cr"), run=((0.0, 2), (0.45, 3))),
            Code("ws-sf", run=((0.0, 0), (0.45, 1))),
            Code(("ws-passing", "ws-correct"), run=((0.0, 2), (0.15, 3), (0.3, 4), (0.5, 6))),
            Code(("ws-passing", "ws-correct"), run=((0.0, 5),)))
        self.EVIDENCE = (
            measured("the cut keeps %s of %s" % (format(m["signal_kept"], ","),
                                                 format(m["signal_n"], ",")), ON),
            measured("the probes' histograms", ON),
            measured("%s and %s probes pass" % (format(m["pass_mc"], ","),
                                                format(m["pass_data"], ",")), ON),
            exact("%.4f ÷ %.4f = %s" % (m["eff_data"], m["eff_mc"], m["sf_text"])),
            measured("the count above the cut", ON),
            measured("%s ± %s against %s ± %s" % (
                m["corrected_text"], m["corrected_err_text"], m["observed_text"],
                m["data_err_text"]), ON))

    def heading_for(self, k, u):
        return ("a cut chosen on simulation",
                "a control region: signal-like probes",
                "how many pass: simulation and data",
                "the scale factor: data ÷ simulation",
                "the signal region: simulation is too high",
                "the simulated signal × the scale factor")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Last time's simulated signal. The cut x > %s is chosen on simulation: it keeps "
            "80%% of it, the 80%% working point." % m["cut_text"],
            "A control region of objects known to be signal-like (tag and probe). Made up: "
            "in the data they spread wider, %g not 0.10." % m["data_spread"],
            "Above the cut: %s%% of the simulated probes, but %s%% of the data's. The cut "
            "keeps fewer in data." % ("%.1f" % (100 * m["eff_mc"]), "%.1f" % (100 * m["eff_data"])),
            "The scale factor is ε_data ÷ ε_MC = %s, ± %s from counting %s and %s probes." % (
                m["sf_text"], m["sf_err_text"], format(m["n_mc"], ","),
                format(m["n_data"], ",")),
            "Above the cut, the weighted simulation predicts %s events; the made-up data "
            "have %s ± %s." % (m["predicted_text"], m["observed_text"], m["data_err_text"]),
            "Scaled by %s, the simulated signal gives %s ± %s: within one error of the data. "
            "It fixes the count, not the shape." % (m["sf_text"], m["corrected_text"],
                                                   m["corrected_err_text"]),
        ][k]

    # ------------------------------------------------------------------ the charts
    def x_chart(self, y_range):
        return LossChart(CHART, self.X_RANGE, y_range)

    def sr_chart(self):
        return LossChart(CHART, (0.0, 1.0), self.SR_Y)

    # ------------------------------------------------------------------ steady
    def steady(self, sc, k, u, op):
        # the chart of x: phases 1-4, leaving as phase 5 starts; then the signal region (one
        # out, then the other in: the picture is never empty)
        c = op * (1.0 - self.prog(4, 0.0, self.TO_SR[0]))
        if c > 0.01:
            self.x_chart_now(sc, k, c)
        s = op * self.prog(4, *self.TO_SR)
        if s > 0.01:
            self.signal_region(sc, s)
        # the rows: the efficiencies from phase 3 on, the scale factor from phase 4
        self.rows(sc, k, u, op)

    def x_chart_now(self, sc, k, op):
        m = self.meta
        swap = self.prog(1, *self.SWAP)
        xticks = [(0.7, "0.7"), (1.0, "1"), (1.3, "1.3")]
        ch = self.x_chart(self.SIGNAL_Y)
        self.text(sc, ch.x1 + 0.004, ch.y1 + 0.026, "x", op, SIZE, "lm", ink=layout.NOTE_INK)
        if swap < 0.5:                     # one chart out, then the other in
            ch = self.x_chart(self.SIGNAL_Y)
            a = op * (1 - smoothstep(0.0, 0.5, swap))
            self.frame(sc, ch, a, xticks, [(0.0, "0"), (2000.0, "2,000"), (4000.0, "4,000")],
                       "simulated signal, events per bin")
            self.bars(sc, ch, self.sig_edges, m["signal_counts"], INKS[0], a)
        if swap > 0.5:
            ch = self.x_chart(self.SHARE_Y)
            a = op * smoothstep(0.5, 1.0, swap)
            self.frame(sc, ch, a, xticks, [(0.0, "0"), (0.1, "10%"), (0.2, "20%")],
                       "probes, share per bin of x")
            grow = self.prog(1, 0.2, 0.36)
            self.bars(sc, ch, self.probe_edges, np.asarray(m["probe_mc_shares"]) * grow,
                      INKS[0], a, fill=0.55)
            d = a * self.at_u(1, 0.45)
            if d > 0.01:
                e = self.probe_edges
                for i, (v, err) in enumerate(zip(m["probe_data_shares"], m["probe_data_err"],
                                                 strict=True)):
                    if v <= 0:
                        continue
                    fx = ch.fx(float(e[i] + e[i + 1]) / 2)
                    self.errbar(sc, fx, ch.fy(v - err), ch.fy(v + err), DATA_INK, d, cap=2.5,
                                width=1.3)
                    self.disk(sc, self.P(fx, ch.fy(v)), 3.0, DATA_INK, d, lift=1.6, outline=0.1)
        # the cut, all through: phase 1, then again from phase 3
        ch = self.x_chart(self.SIGNAL_Y)
        cut = op * (self.at_u(0, 0.3) * (1 - self.prog(1, 0.0, 0.08))
                    + self.at_u(2, 0.08) * (k >= 2))
        self.shade(sc, ch, m["cut"], cut)
        self.vline(sc, ch, m["cut"], INK, cut, label="cut")

    def signal_region(self, sc, op):
        """The count above the cut as a zoomed number line: the simulation's prediction (a
        red square, coming down in phase 6 to its corrected value, a ghost left where it
        was) and the data's (a black dot), each with its error bar."""
        m = self.meta
        sr = m["sr"]
        ch = self.sr_chart()
        ticks = [(v, f"{v:,}") for v in (4500, 4750, 5000, 5250)]
        self.axes(sc, ch, op, "", xticks=[], yticks=ticks, grid=False)
        self.text(sc, ch.x0, ch.y0 - 0.02, "events above the cut", op, SIZE, "lm",
                  ink=layout.CAPTION_INK)
        xs, xd = self.SR_X
        mv = self.prog(5, *self.MOVE)
        top = sr["predicted"] + (sr["corrected"] - sr["predicted"]) * mv
        err = sr["predicted_err"] + (sr["corrected_err"] - sr["predicted_err"]) * mv
        a = op * self.at_u(4, 0.2)
        self.text(sc, xs, ch.y1 + 0.03, "simulation", a, SIZE, ink=TEXT_INKS[0])
        if mv > 0:
            self.disk(sc, self.P(xs, ch.fy(sr["predicted"])), 5.0, soft(RED, 0.4), op * 0.8,
                      lift=1.5, outline=0.0)
        self.errbar(sc, xs, ch.fy(top - err), ch.fy(top + err), RED, a)
        self.rect(sc, self.P(xs, ch.fy(top)), 9.0, 9.0, RED, a, border=dark(RED, 0.3),
                  border_w=1.2, radius=2.0, lift=1.6)
        label = m["predicted_text"] if mv < 0.5 else "%s ± %s" % (m["corrected_text"],
                                                                 m["corrected_err_text"])
        self.text(sc, xs - 0.018, ch.fy(top), label, a, SIZE, "rm", ink=TEXT_INKS[0])
        if mv > 0.5:
            self.text(sc, xs - 0.018, ch.fy(sr["predicted"]), "was " + m["predicted_text"],
                      op * self.at_u(5, self.MOVE[1]), SIZE, "rm", ink=QUIET)
        d = op * self.at_u(4, 0.5)
        self.text(sc, xd, ch.y1 + 0.03, "data", d, SIZE, ink=INK)
        self.errbar(sc, xd, ch.fy(sr["observed"] - sr["data_err"]),
                    ch.fy(sr["observed"] + sr["data_err"]), DATA_INK, d)
        self.disk(sc, self.P(xd, ch.fy(sr["observed"])), 5.5, DATA_INK, d, lift=1.6,
                  outline=0.1)
        self.text(sc, xd - 0.018, ch.fy(sr["observed"]), "%s ± %s" % (m["observed_text"],
                                                                     m["data_err_text"]),
                  d, SIZE, "rm", strong=True)

    def rows(self, sc, k, u, op):
        """Under the chart. Phase 1: the cut; 2: the probes; from 3: each efficiency, then the
        scale factor, then what the signal region shows."""
        m = self.meta
        if k < 2:                          # phases 1-2 draw their own rows
            return
        # from phase 3 on: the efficiencies stay, each swapped in place from its probes' row
        a = op * (self.at_u(2, 0.2) if k == 2 else 1.0)
        self.row(sc, 0, "simulation: " + m["eff_mc_row"], a, chip=INKS[0], ink=TEXT_INKS[0])
        b = op * (self.at_u(2, 0.45) if k == 2 else 1.0)
        self.row(sc, 1, "data: " + m["eff_data_row"], b, chip=DATA_INK, chip_kind="dot")
        if k >= 3:
            c = op * (self.at_u(3, 0.15) if k == 3 else 1.0)
            self.row(sc, 2, "scale factor: " + m["sf_row"], c, strong=True)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.row(sc, 0, "cut: x > " + m["cut_text"], on(u, 0.3, op), strong=True)
        self.row(sc, 1, "keeps %s of %s simulated: %d%%" % (
            format(m["signal_kept"], ","), format(m["signal_n"], ","),
            round(100 * m["working_point"])), on(u, 0.55, op), chip=INKS[0], ink=TEXT_INKS[0])

    def phase_2(self, sc, u, op):
        m = self.meta
        self.row(sc, 0, "simulation: %s probes" % format(m["n_mc"], ","), on(u, 0.22, op),
                 chip=INKS[0], ink=TEXT_INKS[0])
        self.row(sc, 1, "data: %s probes" % format(m["n_data"], ","), on(u, 0.45, op),
                 chip=DATA_INK, chip_kind="dot")
        self.row(sc, 2, "made up: data spread %g, not 0.10" % m["data_spread"], on(u, 0.65, op),
                 strong=True)

    def phase_3(self, sc, u, op):
        self.row(sc, 2, "passing: x above the cut", on(u, 0.08, op), ink=QUIET)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.row(sc, 3, "± %s: the statistical error" % m["sf_err_text"], on(u, 0.45, op))

    def phase_5(self, sc, u, op):
        m = self.meta
        self.row(sc, 3, "simulation − data = %s: %s × the error" % (
            m["excess_text"], m["pull_before_text"]), on(u, 0.7, op), strong=True)

    def phase_6(self, sc, u, op):
        m = self.meta
        self.row(sc, 3, "signal %s × %s = %s; background kept" % (
            m["signal_text"], m["sf_text"], m["signal_scaled_text"]), on(u, 0.08, op),
            strong=True)


ANIMATIONS = {"event-weights": EventWeights, "scale-factors": ScaleFactors}
