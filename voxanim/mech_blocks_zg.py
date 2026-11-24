"""Animations 273-274 of "Blocks and code", "The classifier and the likelihood ratio" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_zg`: one small program
(``examples/likelihood_ratio.py``), on made-up events.

* :class:`LearnsRatio` (``classifier-learns-the-likelihood-ratio``): two made-up hypotheses
  about one value x, as histograms (H0 wide, H1 narrow); a network trained to tell them apart,
  its loss falling to the exact ratio's own; its s / (1 - s) over the exact p1 / p0 on a log
  scale, close where both have events; where they run out, it drifts; its ROC on fresh events
  against the best test's (the exact ratio: a window of x) and a single cut's.
* :class:`Reweights` (``reweighting-with-a-classifier``): made-up "simulation" and "data" in
  x and y; a network trained to tell them apart, its weight s / (1 - s) over the plane of
  (x, y); the simulation's histograms weighted, matching the data's; x + y, never given to it,
  matching too; where it fails: data where the simulation has none, the largest weights,
  the events the weighted sample is worth, and a network trained on few events.

The alphabet: H0 blue, H1 red (as in the hypothesis tests); the network's output, and the
weights, orange (it is what the weights of the network make); the exact answer dark ink; the
data black dots, the simulation blue bars, the weighted simulation orange bars; a range where
the network goes wrong tinted red.
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
    measured,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .losschart import LossChart, sup
from .mech import AMBER, AXIS, BLUE, GRID, PAPER, RED, dark, heat, soft

ON = "made-up data"
H0_INK, H1_INK = BLUE, RED
NET = AMBER                         # the network's output, and the weights
NET_INK = dark(AMBER, 0.3)          # its text
EXACT = INK                         # the exact answer
CUT_INK = (0.55, 0.57, 0.60)        # a single cut
LOSS_INK = dark(LOSS, 0.3)
SIM = BLUE
SIM_INK = dark(BLUE, 0.25)
WRONG = RED                         # where it goes wrong

# ---------------------------------------------------------------------- 273's layout
HIST_BOX = ((0.105, 0.455), (0.395, 0.545))      # the two histograms of x
LOW_BOX = ((0.105, 0.455), (0.635, 0.855))       # the loss, then the ratio chart
ROC_BOX = ((0.105, 0.285), (0.635, 0.855))       # the ROC
NOTE_Y = 0.592                                   # the line between the two charts
ROC_COL = 0.305                                  # the ROC's numbers

# ---------------------------------------------------------------------- 274's layout
COL_X = 0.325                                    # the right column
CHARTS = {"x": ((0.105, 0.30), (0.385, 0.495)),
          "y": ((0.105, 0.30), (0.56, 0.67)),
          "x + y": ((0.105, 0.30), (0.735, 0.845))}
PLANE_C, PLANE_PX = (0.40, 0.64), 112.0         # the plane of (x, y): centre, side in px
SCATTER_BOX = ((0.36, 0.46), (0.43, 0.575))     # learned against exact weights (log, log)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def log_tick(e):
    """``10⁻⁴``, ``1``, ``10``."""
    return {0: "1", 1: "10"}.get(e, "10" + sup(e))


class Shared(Blocks):
    """What the two share: time, histograms, curves on charts, the step indicator."""

    STATION = None

    def setup(self):
        self.meta = self.study.metadata
        self._t = 0.0

    def station_for(self, k):
        return self.STATIONS[k] if hasattr(self, "STATIONS") else None

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

    # ------------------------------------------------------------------ time
    def time_of(self, k, u):
        a, b = self.spans[k]
        return a + (b - a) * u

    def prog(self, k, u0, u1):
        """0 before ``u0`` of phase ``k``, 1 after ``u1``, eased between (by the clock)."""
        t0, t1 = self.time_of(k, u0), self.time_of(k, u1)
        return ease_in_out(float(np.clip((self._t - t0) / (t1 - t0), 0.0, 1.0)))

    def at(self, k, u, width=0.5):
        """1 once phase ``k`` has reached ``u`` (fading in over ``width`` s); 0 before."""
        t0 = self.time_of(k, u)
        return smoothstep(t0, t0 + width, self._t)

    # ------------------------------------------------------------------ drawing
    def lifted(self, fx, fy):
        return self.P(float(fx), float(fy), self.ARROW_LIFT)

    def guide(self, sc, pts, ink, op, width=2.0):
        """A polyline through layout points, in front of the shapes."""
        if op > 0.01 and len(pts) > 1:
            self.polyline(sc, [self.lifted(*p) for p in pts], width, ink, op, bias=0.3)

    def dashed(self, sc, a, b, ink, op, n=9, width=1.4):
        if op <= 0.01:
            return
        a, b = np.asarray(a, float), np.asarray(b, float)
        for i in range(n):
            self.guide(sc, [a + (b - a) * i / n, a + (b - a) * (i + 0.5) / n], ink, op, width)

    def curve(self, sc, ch, xs, ys, ink, op, upto=1.0, width=2.2):
        """``ys`` over ``xs`` on ``ch`` as far as fraction ``upto`` of the points, the parts
        outside its y range left out."""
        if op <= 0.01 or upto <= 0:
            return
        lo, hi = ch.y_range
        n = max(2, int(round(len(xs) * min(upto, 1.0))))
        run = []
        for x, y in zip(xs[:n], ys[:n], strict=True):
            if lo * (1 - 1e-9) <= y <= hi * (1 + 1e-9):
                run.append((ch.fx(x), ch.fy(y)))
            else:
                self.guide(sc, run, ink, op, width)
                run = []
        self.guide(sc, run, ink, op, width)

    def bars(self, sc, ch, edges, values, ink, op, gap=1.0, front=0, outline=None):
        """A histogram on ``ch``: one bar per bin, from the bottom of the chart."""
        if op <= 0.01:
            return
        floor = ch.y_range[0]
        for a, b, v in zip(edges[:-1], edges[1:], values, strict=True):
            if v <= floor or b <= ch.x_range[0] or a >= ch.x_range[1]:
                continue
            x0, x1 = ch.fx(a), ch.fx(b)
            y0, y1 = ch.fy(float(v)), ch.fy(floor)
            w, h = self.span(x1 - x0, y1 - y0)
            if h < 0.5 or w - gap < 0.3:
                continue
            self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w - gap, h, ink, op,
                      radius=min(1.5, (w - gap) / 2, h / 2), lift=0.4 + 0.15 * front,
                      border=outline, border_w=1.0)

    def points(self, sc, ch, edges, values, ink, op, r=2.4):
        """Data as dots at the bins' centres (none where a bin is empty)."""
        if op <= 0.01:
            return
        for a, b, v in zip(edges[:-1], edges[1:], values, strict=True):
            if v <= 0 or b <= ch.x_range[0] or a >= ch.x_range[1]:
                continue
            self.disk(sc, self.P(ch.fx((a + b) / 2), ch.fy(float(v))), r, ink, op, lift=1.5,
                      outline=0.1)

    def tint(self, sc, ch, xa, xb, ink, op, strength=0.18):
        """The range ``xa..xb`` of ``ch`` tinted over its whole height."""
        if op <= 0.01:
            return
        x0, x1 = ch.fx(max(xa, ch.x_range[0])), ch.fx(min(xb, ch.x_range[1]))
        w, h = self.span(x1 - x0, ch.y1 - ch.y0)
        if w > 0.5:
            self.rect(sc, self.P((x0 + x1) / 2, (ch.y0 + ch.y1) / 2), w, h, ink,
                      op * strength, radius=0.0, lift=0.05)

    def swatch(self, sc, fx, fy, ink, op, kind="bar"):
        """A key's mark: a bar, a dot or a line, left of ``fx``."""
        if kind == "bar":
            self.rect(sc, self.P(fx - 0.012, fy), 10.0, 12.0, ink, op, radius=1.5, lift=0.4)
        elif kind == "dot":
            self.disk(sc, self.P(fx - 0.012, fy), 3.5, ink, op, lift=1.5, outline=0.1)
        else:
            self.guide(sc, [(fx - 0.022, fy), (fx - 0.004, fy)], ink, op, 2.6)

    def key(self, sc, fx, fy, text, ink, op, kind="bar", text_ink=None):
        if op <= 0.01:
            return
        self.swatch(sc, fx, fy, ink, op, kind)
        self.text(sc, fx + 0.004, fy, text, op, SIZE, "lm", ink=text_ink)


# ====================================================================== 273
class LearnsRatio(Shared):
    """Two hypotheses; training; the ratio; where it drifts; the ROC."""

    TITLE = "A classifier learns the likelihood ratio"
    PHASES = (("two hypotheses", 10.0), ("training", 11.0), ("the ratio", 12.0),
              ("no events", 10.0), ("ROC", 12.0))
    LAST = "The look-elsewhere effect"
    NEXT = "Reweighting simulation with a classifier"
    STATIONS = (None, "loss", None, None, None)

    def setup(self):
        super().setup()
        m = self.meta
        self.show = np.asarray(m["show"])
        self.grid = np.asarray(m["grid"])
        self.learned = np.asarray(m["learned"])
        self.exact = np.asarray(m["exact"])
        top = max(max(m["h0"]), max(m["h1"])) * 1.05
        self.hist = LossChart(HIST_BOX, (-2.0, 4.0), (0.0, top))
        self.ratio = LossChart(LOW_BOX, (-2.0, 4.0), (1e-4, 10.0), ylog=True)
        self.loss = LossChart(LOW_BOX, (0.0, float(m["curve_steps"][-1])), (0.44, 0.70))
        self.roc = LossChart(ROC_BOX, (0.0, 0.3), (0.0, 1.0))
        self.CODE = (
            Code("lr-hypotheses", run=((0.0, 0), (0.12, 1), (0.25, 2), (0.45, 3),
                                       (0.65, 4))),
            Code("lr-train", run=((0.0, 0), (0.12, 1), (0.25, 3), (0.4, 4), (0.55, 5),
                                  (0.65, 6), (0.75, 7))),
            Code("lr-ratio", run=((0.0, 0), (0.08, 2), (0.3, 3), (0.5, 4))),
            Code("lr-ratio", run=((0.0, 3),)),
            Code("lr-power", run=((0.0, 0), (0.15, 3), (0.3, 5), (0.45, 6)),
                 out=((0.5, "power: %s" % m["printed_power"]),)))
        self.EVIDENCE = (
            measured("20,000 drawn events of each", ON),
            measured("the loss, one training run", ON),
            measured("the network's ratio; p1 / p0 exact", ON),
            measured("events past x = %g; the ratios" % m["far_x"], ON),
            measured("ROC, fresh events; best exact", ON))

    def heading_for(self, k, u):
        return ("two made-up hypotheses about one value x",
                "a network learns: H0 or H1?",
                "s / (1 − s) follows the exact p1 / p0",
                "where events run out, it drifts",
                "its ROC: as good as the best test")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Made-up events: under H0, x is wide and centred at 0; under H1, narrow and "
            "centred at 1. 20,000 of each.",
            "A small network is trained to say H1 or H0, on equal numbers: its loss falls to "
            "%s; the exact ratio's own is %s." % (m["last_text"], m["floor_text"]),
            "Its output s, the chance of H1, gives s / (1 − s): within %s of the exact ratio "
            "for x from −0.75 to 2.25." % m["within_text"],
            "Past x = 3 there are %d H0 and %d H1 events: there the network says %s, the "
            "exact ratio %s." % (m["beyond_far0"], m["beyond_far1"], m["far_learned_text"],
                                m["far_exact_text"]),
            "Keeping 5%% of H0, it keeps %s of fresh H1 events (±%s); the best test %s, one "
            "cut %s." % (m["net_text"], m["err_text"], m["np_text"], m["cut_text"]),
        ][k]

    # ------------------------------------------------------------------ steady
    def steady(self, sc, k, u, op):
        m = self.meta
        ch = self.hist
        self.axes(sc, ch, op, xticks=[], yticks=[], grid=False)
        g0 = 1.0 if k > 0 else self.prog(0, 0.08, 0.3)
        g1 = 1.0 if k > 0 else self.prog(0, 0.32, 0.55)
        h0 = np.asarray(m["h0"], float) * g0
        h1 = np.asarray(m["h1"], float) * g1
        self.bars(sc, ch, self.show, h0, soft(H0_INK, 0.25), op * 0.95)
        self.bars(sc, ch, self.show, h1, soft(H1_INK, 0.2), op * 0.8 * min(1.0, 4 * g1),
                  front=1)
        a0, a1 = op * (1.0 if k > 0 else self.at(0, 0.12)), op * (1.0 if k > 0 else
                                                                   self.at(0, 0.36))
        self.text(sc, ch.fx(-1.6), ch.fy(0.62 * ch.y_range[1]), "H0", a0, SIZE, "mm",
                  ink=dark(H0_INK, 0.25))
        self.text(sc, ch.fx(2.3), ch.fy(0.62 * ch.y_range[1]), "H1", a1, SIZE, "mm",
                  ink=dark(H1_INK, 0.2))
        # the ratio chart, kept from phase 3 into phase 4
        if k in (2, 3):
            r = op * self.kept(2, 3)
            self.ratio_axes(sc, r)
            self.curve(sc, self.ratio, self.grid, self.exact, EXACT, r,
                       upto=self.prog(2, 0.05, 0.3), width=2.4)
            self.curve(sc, self.ratio, self.grid, self.learned, NET, r,
                       upto=self.prog(2, 0.35, 0.65), width=2.2)
        # the best test's window and a single cut, in the ROC phase
        if k == 4:
            w = self.prog(4, 0.55, 0.65)
            lo, hi = m["np_window"]
            self.tint(sc, ch, lo, hi, INK, op * w, 0.16)
            c = self.prog(4, 0.7, 0.8)
            self.dashed(sc, (ch.fx(m["cut"]), ch.y0), (ch.fx(m["cut"]), ch.y1), CUT_INK,
                        op * c, n=7, width=1.8)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        x = 0.105
        rows = ((0.64, "H0: wide, centred at 0", H0_INK), (0.70, "H1: narrow, centred at 1",
                                                           H1_INK))
        for (y, text, ink), a in zip(rows, (0.15, 0.38), strict=True):
            self.key(sc, x + 0.02, y, text, ink, on(u, a, op), text_ink=dark(ink, 0.25))
        self.text(sc, x, 0.78, "made-up: 20,000 events of each", on(u, 0.6, op), SIZE, "lm",
                  strong=True)
        self.text(sc, x, 0.83, "the exact p1 / p0 is known", on(u, 0.75, op), SIZE, "lm",
                  ink=QUIET)

    def phase_2(self, sc, u, op):
        m = self.meta
        ch = self.loss
        self.axes(sc, ch, op, xticks=[(0, "0"), (ch.x_range[1], "10,000 steps")],
                  yticks=[(0.5, "0.5"), (0.6, "0.6"), (0.7, "0.7")], grid=False)
        go = eased(u, 0.1, 0.7)
        xs, ys = np.asarray(m["curve_steps"], float), np.asarray(m["curve"])
        self.curve(sc, ch, xs, ys, LOSS_INK, op, upto=go)
        f = on(u, 0.72, op)
        self.dashed(sc, (ch.x0, ch.fy(m["floor_on"])), (ch.x1, ch.fy(m["floor_on"])), EXACT,
                    f, n=14, width=1.4)
        n = max(1, int(round(len(xs) * go)))
        now = ys[n - 1]
        self.text(sc, ch.x1, ch.fy(0.665), "loss %.4f" % now, on(u, 0.1, op), SIZE, "rm",
                  ink=LOSS_INK)
        self.text(sc, ch.x1, ch.fy(0.62), "exact ratio's: %s" % m["floor_text"], f, SIZE, "rm",
                  ink=EXACT)
        self.text(sc, 0.105, NOTE_Y, "equal numbers of each: labels 0 and 1", on(u, 0.02, op),
                  SIZE, "lm", ink=QUIET)

    def ratio_axes(self, sc, op):
        ch = self.ratio
        self.axes(sc, ch, op, xticks=[(v, "%g" % v if v >= 0 else "−%g" % -v)
                                      for v in (-1, 0, 1, 2, 3, 4)],
                  yticks=[(10.0 ** e, log_tick(e)) for e in (-4, -2, 0, 1)], grid=False)
        self.guide(sc, [(ch.x0, ch.fy(1.0)), (ch.x1, ch.fy(1.0))], GRID, 0.6 * op, 1.0)

    def phase_3(self, sc, u, op):
        m = self.meta
        ch = self.ratio
        lo, hi = m["track"]
        self.tint(sc, ch, lo, hi, NET, on(u, 0.72, op), 0.16)
        self.key(sc, 0.125, NOTE_Y, "exact", EXACT, on(u, 0.05, op), kind="line")
        self.key(sc, 0.215, NOTE_Y, "network: s / (1 − s)", NET, on(u, 0.35, op), kind="line",
                 text_ink=NET_INK)
        self.text(sc, ch.fx(0.75), ch.fy(4e-4), "within %s" % m["within_text"],
                  on(u, 0.78, op), SIZE, "mm", ink=NET_INK)

    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self.ratio
        x = m["far_x"]
        self.tint(sc, ch, ch.x_range[0], m["edge"][0], WRONG, on(u, 0.05, op), 0.14)
        self.tint(sc, ch, x, ch.x_range[1], WRONG, on(u, 0.05, op), 0.14)
        a = on(u, 0.35, op)
        for v, ink in ((m["far_learned"], NET), (m["far_exact"], EXACT)):
            self.disk(sc, self.P(ch.fx(x), ch.fy(v)), 4.5, ink, a, lift=1.6, outline=0.1)
        rows = ((1.5e-2, "at x = 3:", INK), (2.5e-3, "network: " + m["far_learned_text"],
                                             NET_INK),
                (4e-4, "exact: " + m["far_exact_text"], EXACT))
        for v, text, ink in rows:
            self.text(sc, ch.fx(0.0), ch.fy(v), text, a, SIZE, "lm", ink=ink,
                      strong=ink is INK)
        self.text(sc, ch.fx(-1.5), ch.fy(1.0) - 0.035, "no H1", on(u, 0.15, op), SIZE, "mm",
                  ink=dark(WRONG, 0.2))
        self.text(sc, 0.105, NOTE_Y, "past x = 3: %d H0, %d H1 events" % (
            m["beyond_far0"], m["beyond_far1"]), on(u, 0.6, op), SIZE, "lm", strong=True)

    def phase_5(self, sc, u, op):
        m = self.meta
        ch = self.roc
        self.axes(sc, ch, op, xticks=[(0, "0"), (0.05, "5%"), (0.3, "30%")],
                  yticks=[(0, "0"), (0.5, "50%"), (1.0, "100%")], grid=False)
        self.text(sc, ch.x0, NOTE_Y + 0.008, "H1 kept against H0 kept", on(u, 0.02, op), SIZE,
                  "lm", ink=QUIET)
        al = np.asarray(m["alphas"])
        sel = al <= 0.3 + 1e-9
        self.curve(sc, ch, al[sel], np.asarray(m["roc_np"])[sel], EXACT, op,
                   upto=eased(u, 0.05, 0.25), width=2.4)
        self.curve(sc, ch, al[sel], np.asarray(m["roc_cut"])[sel], CUT_INK, op,
                   upto=eased(u, 0.65, 0.8), width=2.0)
        dots = on(u, 0.28, op)
        if dots > 0.01:
            net = np.asarray(m["roc_net"])
            idx = [i for i in range(len(al)) if al[i] <= 0.3 + 1e-9 and i % 2 == 0]
            n = int(round(len(idx) * eased(u, 0.28, 0.45)))
            for i in idx[:n]:
                self.disk(sc, self.P(ch.fx(al[i]), ch.fy(net[i])), 2.6, NET, dots, lift=1.6,
                          outline=0.1)
        mark = on(u, 0.48, op)
        self.dashed(sc, (ch.fx(0.05), ch.y0), (ch.fx(0.05), ch.y1), AXIS, mark, n=8, width=1.2)
        rows = ((0.665, "best, a window: %s" % m["np_text"], EXACT, 0.5),
                (0.715, "network: %s" % m["net_text"], NET_INK, 0.55),
                (0.765, "one cut: %s" % m["cut_text"], CUT_INK, 0.75),
                (0.83, "at 5% of H0", QUIET, 0.5))
        for y, text, ink, a in rows:
            self.text(sc, ROC_COL, y, text, on(u, a, op), SIZE, "lm", ink=ink)


# ====================================================================== 274
class Reweights(Shared):
    """Two samples; a classifier and its weights; the weighted histograms; x + y; the
    tails; few events."""

    TITLE = "Reweighting simulation with a classifier"
    PHASES = (("two samples", 9.0), ("a classifier", 11.0), ("weighted", 11.0),
              ("x + y", 9.0), ("the tails", 10.0), ("few events", 9.0))
    LAST = "A classifier learns the likelihood ratio"
    NEXT = "A bump hunt, from fit to global significance"
    STATIONS = (None, "loss", None, None, None, None)
    MORPH = (0.15, 0.55)            # phase 3: the simulation's bars take their weights

    def setup(self):
        super().setup()
        m = self.meta
        self.edges = np.asarray(m["bins"])
        self.charts = {}
        for name, box in CHARTS.items():
            h = m["hists"][name]
            top = max(max(h["data"]), max(h["sim"]), max(h["weighted"])) * 1.08
            self.charts[name] = LossChart(box, (-5.0, 5.0), (0.0, top))
        plane = np.log10(np.clip(np.asarray(m["plane_w"]), 1e-3, 1e3))
        self.plane_tex = heat(np.clip(plane / 1.3, -1, 1)[::-1], hot=AMBER, cold=BLUE)
        we = np.asarray(m["scatter_exact"])
        wl = np.asarray(m["scatter_w"])
        self.scatter = (np.log10(we), np.log10(wl))
        self.sc_chart = LossChart(SCATTER_BOX, (-1.5, 2.5), (-1.5, 2.5))
        self.CODE = (
            Code("lr-samples", run=((0.0, 0), (0.15, 2), (0.45, 3), (0.6, 4))),
            Code("lr-judge", run=((0.0, 0), (0.12, 2), (0.25, 4), (0.4, 5))),
            Code(("lr-weights", "lr-check"), pick=(0, 1, 2, 3, 4, 7, 8),
                 run=((0.0, 0), (0.06, 4), (0.15, 5), (0.35, 6))),
            Code("lr-sum", run=((0.0, 0), (0.45, 1))),
            Code("lr-neff", run=((0.0, 1),),
                 out=((0.45, "big: %.1f" % m["w_max"]),)),
            Code("lr-neff", run=((0.0, 0),),
                 out=((0.15, "n_eff: %s" % m["n_eff_text"]),)))
        self.EVIDENCE = (
            measured("50,000 drawn events of each", ON),
            measured("the network's weights, one run", ON),
            measured("fresh samples, weighted", ON),
            measured("x + y, on fresh samples", ON),
            measured("weights against the exact ones", ON),
            measured("effective events; a small run", ON))

    def heading_for(self, k, u):
        return ("two made-up samples: simulation and data",
                "a network: simulation or data?",
                "weighted, the simulation looks like the data",
                "x + y: never given to it, matches too",
                "where it fails: the tails",
                "where it fails: few events")[k]

    def subtitle_for(self, k):
        m = self.meta
        h = m["hists"]
        return [
            "Made-up events, two values each: the data's x is shifted, its y wider, and x and "
            "y go together.",
            "Trained to tell data (1) from simulation (0), its output s gives each simulated "
            "event a weight, w = s / (1 − s).",
            "Fresh events: where %.0f%% of the data are, the weighted simulation is off by at "
            "most %s in x and %s in y." % (100 * h["y"]["middle_share"], h["x"]["off_middle_text"],
                                          h["y"]["off_middle_text"]),
            "x + y was never one of its inputs: unweighted, off by up to %s; weighted, by at "
            "most %s." % (h["x + y"]["off_before_text"], h["x + y"]["off_middle_text"]),
            "Where the simulation has no events, no weight helps: %d data events. The largest "
            "weight is %s; the exact one, %s." % (m["empty_y"], m["w_max_text"],
                                                 m["w_max_exact_text"]),
            "Big weights cost: the %s weighted events count like %s. Trained on 1,000 of each, "
            "its weights are typically off by %s, not %s." % (
                m["n_text"], m["n_eff_text"], m["few_pct"], m["bulk_pct"]),
        ][k]

    # ------------------------------------------------------------------ the histograms
    def weighted_share(self):
        """0 before phase 3's morph, 1 after."""
        return self.prog(2, *self.MORPH)

    def histogram(self, sc, name, op, morph):
        m = self.meta
        h = m["hists"][name]
        ch = self.charts[name]
        self.axes(sc, ch, op, xticks=[(-4, "−4"), (0, "0"), (4, "4")] if name == "x + y"
                  else [], yticks=[], grid=False)
        self.text(sc, ch.x0 - 0.012, (ch.y0 + ch.y1) / 2, name, op, SIZE, "rm", strong=True)
        sim, wt = np.asarray(h["sim"]), np.asarray(h["weighted"])
        vals = sim + (wt - sim) * morph
        ink = tuple(float(a + (b - a) * morph) for a, b in zip(soft(SIM, 0.25), soft(NET, 0.2),
                                                                strict=True))
        self.bars(sc, ch, self.edges, vals, ink, op * 0.95)
        self.points(sc, ch, self.edges, np.asarray(h["data"]), INK, op)

    def steady(self, sc, k, u, op):
        morph = self.weighted_share() if k >= 2 else 0.0
        a = op if k > 0 else op * self.at(0, 0.05)
        self.histogram(sc, "x", a, morph)
        self.histogram(sc, "y", op if k > 0 else op * self.at(0, 0.3), morph)
        if k >= 3:
            g = self.at(3, 0.05)
            self.histogram(sc, "x + y", op * g, self.prog(3, 0.45, 0.75))
        # the key, kept in the right column through the first four phases
        kk = op * self.kept(0, 3)
        y0, y1 = 0.40, 0.445
        self.key(sc, COL_X + 0.02, y0, "data", INK, kk * (1.0 if k else self.at(0, 0.1)),
                 kind="dot")
        if k == 2 or k == 3:
            self.key(sc, COL_X + 0.02, y1, "weighted", NET, kk * (1.0 if k == 3 else
                                                                   self.at(2, 0.15)),
                     text_ink=NET_INK)
        elif k < 2:
            self.key(sc, COL_X + 0.02, y1, "simulation", SIM, kk * (1.0 if k else
                                                                    self.at(0, 0.3)),
                     text_ink=SIM_INK)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        rows = ((0.53, "made-up:", True, 0.5), (0.575, "50,000 events each", False, 0.5),
                (0.66, "data: x shifted,", False, 0.65), (0.705, "y wider, x and y", False,
                                                           0.65),
                (0.75, "go together", False, 0.65))
        for y, text, strong, a in rows:
            self.text(sc, COL_X, y, text, on(u, a, op), SIZE, "lm", strong=strong,
                      ink=None if strong else QUIET)

    def plane(self, sc, op, dots=1.0):
        """The weight over the plane of (x, y): orange above 1, blue below; a few simulated
        events dotted on it."""
        if op <= 0.01:
            return
        m = self.meta
        cx, cy = PLANE_C
        s = PLANE_PX
        self.rect(sc, self.P(cx, cy), s + 6, s + 6, PAPER, op, border=AXIS, border_w=1.0,
                  radius=3, lift=0.1)
        draw_image(self, sc, self.P(cx, cy), self.plane_tex.astype(np.float32), s, s, op,
                   grid=0)
        hw, hh = self.frac(s / 2, s / 2)
        if dots > 0.01:
            for (x, y) in np.asarray(m["events"]):
                if abs(x) < 4 and abs(y) < 4:
                    self.disk(sc, self.P(cx + x / 4 * hw, cy - y / 4 * hh), 1.5, INK,
                              op * dots * 0.8, lift=1.5, outline=0.0)
        self.text(sc, cx, cy + hh + 0.024, "x", op, SIZE, "mm", ink=QUIET)
        self.text(sc, cx - hw - 0.016, cy, "y", op, SIZE, "mm", ink=QUIET)

    def phase_2(self, sc, u, op):
        cx, cy = PLANE_C
        hh = self.frac(0, PLANE_PX / 2)[1]
        self.text(sc, COL_X, 0.505, "w = s / (1 − s)", on(u, 0.3, op), SIZE, "lm",
                  ink=NET_INK, strong=True)
        self.plane(sc, on(u, 0.4, op), dots=self.prog(1, 0.7, 0.85))
        self.text(sc, COL_X, cy + hh + 0.06, "orange: w above 1", on(u, 0.55, op), SIZE, "lm",
                  ink=NET_INK)
        self.text(sc, COL_X, cy + hh + 0.1, "blue: w below 1", on(u, 0.6, op), SIZE, "lm",
                  ink=SIM_INK)
        del cx

    def phase_3(self, sc, u, op):
        h = self.meta["hists"]
        rows = ((0.53, "off by at most", QUIET, 0.6), (0.575, "x: %s" % h["x"]["off_middle_text"],
                                                        NET_INK, 0.65),
                (0.62, "y: %s" % h["y"]["off_middle_text"], NET_INK, 0.7),
                (0.68, "before: %s, %s" % (h["x"]["off_before_text"], h["y"]["off_before_text"]),
                 SIM_INK, 0.8))
        for y, text, ink, a in rows:
            self.text(sc, COL_X, y, text, on(u, a, op), SIZE, "lm", ink=ink)

    def phase_4(self, sc, u, op):
        h = self.meta["hists"]["x + y"]
        rows = ((0.53, "not an input", INK, 0.1), (0.62, "before: %s" % h["off_before_text"],
                                                   SIM_INK, 0.3),
                (0.67, "weighted: %s" % h["off_middle_text"], NET_INK, 0.75))
        for y, text, ink, a in rows:
            self.text(sc, COL_X, y, text, on(u, a, op), SIZE, "lm", ink=ink,
                      strong=ink is INK)

    def phase_5(self, sc, u, op):
        m = self.meta
        # the y bins with data and no simulation, ringed
        ch = self.charts["y"]
        a = on(u, 0.08, op)
        if m["empty_bins"]:
            lo = self.edges[min(m["empty_bins"])]
            hi = self.edges[max(m["empty_bins"]) + 1]
            self.tint(sc, ch, lo, hi, WRONG, a, 0.3)
        self.text(sc, COL_X, 0.70, "%d data events," % m["empty_y"], a, SIZE, "lm",
                  ink=dark(WRONG, 0.2), strong=True)
        self.text(sc, COL_X, 0.745, "no simulation", a, SIZE, "lm", ink=dark(WRONG, 0.2))
        # learned against exact weights
        sch = self.sc_chart
        b = on(u, 0.35, op)
        self.axes(sc, sch, b, xticks=[(0, "1"), (2, "100")], yticks=[(0, "1"), (2, "100")],
                  grid=False)
        self.dashed(sc, (sch.fx(-1.5), sch.fy(-1.5)), (sch.fx(2.5), sch.fy(2.5)), AXIS, b,
                    n=10, width=1.0)
        lx, ly = self.scatter
        n = int(round(len(lx) * eased(u, 0.38, 0.55)))
        for x, y in zip(lx[:n], ly[:n], strict=True):
            self.disk(sc, self.P(sch.fx(x), sch.fy(y)), 1.6, NET, b * 0.85, lift=1.5,
                      outline=0.0)
        self.text(sc, sch.x0, sch.y0 - 0.025, "network's w", b, SIZE, "lm", ink=NET_INK)
        self.text(sc, sch.x1, sch.y1 + 0.05, "exact w", b, SIZE, "rm", ink=QUIET)
        c = on(u, 0.62, op)
        self.text(sc, COL_X, 0.815, "largest: %s" % m["w_max_text"], c, SIZE, "lm",
                  ink=NET_INK, strong=True)
        self.text(sc, COL_X, 0.86, "exact: %s" % m["w_max_exact_text"], c, SIZE, "lm",
                  ink=INK)

    def phase_6(self, sc, u, op):
        m = self.meta
        rows = ((0.47, "%s events" % m["n_text"], INK, 0.05, True),
                (0.52, "count like %s" % m["n_eff_text"], NET_INK, 0.15, True),
                (0.64, "1,000 of each:", INK, 0.5, False),
                (0.69, "typically off by %s" % m["few_pct"], dark(WRONG, 0.2), 0.6, True),
                (0.74, "50,000: %s" % m["bulk_pct"], NET_INK, 0.7, False))
        for y, text, ink, a, strong in rows:
            self.text(sc, COL_X, y, text, on(u, a, op), SIZE, "lm", ink=ink, strong=strong)


ANIMATIONS = {"classifier-learns-the-likelihood-ratio": LearnsRatio,
              "reweighting-with-a-classifier": Reweights}
