"""Two animations of "Blocks and code" between ML and a physics result (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_zf`; all the data are made up.

* :class:`ErrorBars` (272, ``error-bars-on-a-network``, ``examples/ensembles.py``): the 40
  made-up points of "Minuit and SGD" across the top; under them the plane of two of the fit's
  three numbers, tau across and b up, with Minuit's minimum. One SGD run's answer (where it
  ends) is a dot; 100 runs with new seeds land in a tight clump (the optimiser's noise); one
  bootstrap resample of the 40 points (bigger dot: picked more often, hollow: left out) and its
  answer; 100 resamples' answers spread into a cloud; the cloud's spread and covariance ellipse;
  HESSE's chi-square + 1 ellipse over it, and the three spreads side by side.
* :class:`Asimov` (278, ``asimov-dataset``, ``examples/asimov.py``): a made-up histogram of 4
  bins, background with signal stacked on it; one toy experiment's counts and its
  significance; 10,000 toys' significances as a histogram, its median; the Asimov dataset (each
  bin exactly its expected count) and its significance on that median; s / sqrt(b) beside them;
  the same with a tenth of the background, where s / sqrt(b) is far too high.

The alphabet: data blue (the points, a toy's counts); the answers of runs with new seeds teal
(SGD's colour last time), of runs on resamples magenta; Minuit and HESSE dark ink; background
grey, signal amber; the Asimov significance green, s / sqrt(b) red.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    Blocks,
    Code,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AMBER, AXIS, GREEN, MAGENTA, PAPER, RED, TEAL, dark, soft

ON = "made-up data"
SEED = TEAL                          # the answers of runs with new seeds
SEED_INK = dark(TEAL, 0.35)
BOOT = MAGENTA                       # the answers of runs on resamples
BOOT_INK = dark(MAGENTA, 0.25)
DATA_INK = dark(DATA, 0.3)
SIGNAL = AMBER
SIGNAL_INK = dark(AMBER, 0.35)
BACK = (0.62, 0.64, 0.67)            # the background, grey
BACK_INK = (0.40, 0.42, 0.44)
ASIMOV = GREEN
ASIMOV_INK = dark(GREEN, 0.3)
SIMPLE = RED
SIMPLE_INK = dark(RED, 0.2)
COL_X = 0.335                        # the right column of the picture: the counter
COL_W = 0.475 - COL_X


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Base(Blocks):
    """What the two share: no training loop, the counter, timing, lifted guide lines."""

    STATION = None
    COUNT_Y = (0.60, 0.645, 0.69, 0.735)

    def setup(self):
        self.meta = self.study.metadata
        self.starts = np.cumsum([0.0] + [s for _n, s in self.PHASES])

    def station_for(self, k):
        return None

    # ------------------------------------------------------------------ time
    def time_of(self, k, u):
        return float(self.starts[k] + (self.starts[k + 1] - self.starts[k]) * u)

    def since(self, t0, width=0.4):
        """0 before time ``t0``, 1 from ``t0 + width`` (eased)."""
        return smoothstep(t0, t0 + width, getattr(self, "_t", 0.0))

    def at_u(self, k, u, width=0.4):
        return self.since(self.time_of(k, u), width)

    def prog(self, k, u0, u1):
        t0, t1 = self.time_of(k, u0), self.time_of(k, u1)
        return ease_in_out(float(np.clip((getattr(self, "_t", 0.0) - t0) / (t1 - t0), 0, 1)))

    def now(self):
        return self.where(getattr(self, "_t", 0.0))

    def after(self, k, u):
        """True once phase ``k`` has reached ``u``."""
        return getattr(self, "_t", 0.0) >= self.time_of(k, u)

    # ------------------------------------------------------------------ drawing
    def guide(self, sc, pts, ink, op, width=1.6):
        """A polyline through layout points, lifted in front of the shapes."""
        if op > 0.01 and len(pts) > 1:
            self.polyline(sc, [self.P(float(x), float(y), self.ARROW_LIFT) for x, y in pts],
                          width, ink, op, bias=0.3)

    def dashed(self, sc, a, b, ink, op, n=9, width=1.2):
        if op <= 0.01:
            return
        a, b = np.asarray(a, float), np.asarray(b, float)
        for i in range(n):
            u0, u1 = i / n, (i + 0.5) / n
            self.guide(sc, [a + (b - a) * u0, a + (b - a) * u1], ink, op, width)

    def dot(self, sc, fx, fy, ink, op, r=2.8, lift=1.5, ring=None):
        if op > 0.01:
            self.disk(sc, self.P(float(fx), float(fy)), r, ink, op, lift=lift, outline=0.15,
                      ring=ring, ring_w=1.4)

    def hollow(self, sc, fx, fy, ink, op, r=2.8, lift=1.5):
        if op > 0.01:
            self.rect(sc, self.P(float(fx), float(fy)), 2 * r, 2 * r, PAPER, op, border=ink,
                      border_w=1.3, radius=r, lift=lift)

    def lines_at(self, sc, rows, op, ys=None):
        """The counter: ``(text, size, ink)`` rows down the right column."""
        for (text, size, ink), y in zip(rows, ys or self.COUNT_Y, strict=False):
            if text:
                self.text(sc, COL_X, y, text, op, size, "lm", ink=ink)

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the title), as in
        :mod:`voxanim.mech_blocks_zd`."""
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

    # every phase's picture is drawn in steady(): nothing blanks between phases
    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        pass

    def phase_6(self, sc, u, op):
        pass


# ====================================================================== 272
STRIP = ((0.085, 0.47), (0.40, 0.49))        # the 40 points across the top
PLANE = ((0.115, 0.30), (0.60, 0.86))         # the (tau, b) plane
INSET = (0.405, 0.805)                        # phase 2: the seeds' answers magnified
INSET_W, INSET_H, ZOOM = 128.0, 80.0, 4.0
TABLE_X = (0.085, 0.215, 0.33, 0.435)         # phase 6: the row names, then 3 columns
TABLE_Y = (0.395, 0.435, 0.472, 0.509)        # its header, then a, tau, b


class ErrorBars(Base):
    """One run's answer; 100 new seeds; a resample; 100 resamples; the spread and its ellipse;
    HESSE's ellipse and errors."""

    COUNT_Y = (0.6, 0.642, 0.684, 0.726)

    TITLE = "Error bars on a network"
    PHASES = (("one run", 8.0), ("new seeds", 11.0), ("a resample", 11.0),
              ("100 resamples", 10.0), ("the spread", 10.0), ("HESSE", 10.0))
    LAST = "Minuit and SGD"
    NEXT = "The likelihood curve"
    SEEDS_U = (0.12, 0.5)           # phase 2: the 99 other seeds' answers arrive
    INSET_U = 0.58                  # phase 2: and magnified
    PICKS_U = 0.12                  # phase 3: resample 1 replaces the 40 points
    BOOT_U = 0.62                   # phase 3: its answer
    BOOTS_U = (0.08, 0.6)           # phase 4: the 99 other resamples' answers
    ELLIPSE_U = (0.12, 0.45)        # phase 5: the cloud's ellipse is drawn
    BARS_U = 0.55                   # phase 5: its shadow, the error bars
    HESSE_U = (0.12, 0.42)          # phase 6: HESSE's ellipse
    TABLE_U = 0.55                  # phase 6: the table of the three spreads

    def setup(self):
        super().setup()
        m = self.meta
        self.x, self.y = np.asarray(m["x"]), np.asarray(m["y"])
        self.seeds = np.asarray(m["seeds"])
        self.boots = np.asarray(m["boots"])
        self.picks = np.asarray(m["resample"])
        self.CODE = (
            Code("en-train", run=((0.0, 0), (0.25, 3), (0.5, 7))),
            Code("en-seeds", run=((0.0, 0), (0.1, 3), (0.65, 4)),
                 out=((0.7, m["printed_seeds"]),)),
            Code("en-bootstrap", run=((0.0, 0), (0.1, 3), (0.55, 4))),
            Code("en-bootstrap", run=((0.0, 2), (0.06, 4))),
            Code("en-spread", run=((0.0, 0), (0.1, 1), (0.45, 2)),
                 out=((0.12, m["printed_boots"]),
                      (0.45, m["printed_boots"] + "\n" + m["printed_cov"]))),
            Code("en-hesse", run=((0.0, 0), (0.06, 3), (0.5, 4)),
                 out=((0.5, m["printed_hesse"]),)))
        self.EVIDENCE = (
            measured("one SGD run, seed 0", ON),
            measured("100 SGD runs, seeds 0 to 99", ON),
            measured("resample 1 and its run, seed 100", ON),
            measured("100 runs on 100 resamples", ON),
            measured("the spread of the 100 answers", ON),
            measured("HESSE, iminuit %s, recorded" % m["iminuit"], ON))

    def heading_for(self, k, u):
        return ("one SGD run: one answer",
                "new seeds: nearly the same answer",
                "a resample: 40 picks, repeats allowed",
                "100 resamples: the answers spread out",
                "the spread, and its ellipse",
                "against HESSE's error bars")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Last time's fit on 40 made-up points, trained by SGD as a network is. A run's "
            "answer is where it ends: 3 numbers, drawn as τ and b.",
            "100 runs, each with a new seed, on the same 40 points: their answers lie within "
            "about %s of each other." % m["seeds_round"],
            "A bootstrap resample: 40 points picked from the 40 at random, repeats allowed. "
            "%d are left out; %d count twice or more." % (m["resample_out"],
                                                          m["resample_twice"]),
            "One run on each of 100 resamples: new data each time, so the answers spread "
            "out, by %s in τ and %s in b." % (m["boots_text"][1], m["boots_text"][2]),
            "Their spread and covariance make an ellipse. It tilts: when τ comes out high, b "
            "comes out low (correlation %s)." % m["corr_boots_text"],
            "HESSE's ellipse, where chi-square is 1 above its minimum, tilts alike (%s). The "
            "resamples come near its errors in a and b, a third short in τ."
            % m["corr_hesse_text"],
        ][k]

    # ------------------------------------------------------------------ charts
    def strip(self):
        return LossChart(STRIP, (0.0, 10.0), (0.0, 7.0))

    def plane(self):
        m = self.meta
        return LossChart(PLANE, tuple(m["tau_range"]), tuple(m["b_range"]))

    def at(self, p):
        """Layout point of an answer ``p = (a, tau, b)`` in the plane."""
        ch = self.plane()
        return ch.fx(p[1]), ch.fy(p[2])

    # ------------------------------------------------------------------ what is shown by now
    def n_seeds(self):
        """How many of the seeds' answers are shown: run 1 from phase 1, all 100 by the end
        of phase 2's arrival."""
        k, _u = self.now()
        if k == 0:
            return 1 if self.after(0, 0.45) else 0
        if k > 1:
            return len(self.seeds)
        return 1 + int(round((len(self.seeds) - 1) * self.prog(1, *self.SEEDS_U)))

    def n_boots(self):
        k, _u = self.now()
        if k < 2:
            return 0
        if k == 2:
            return 1 if self.after(2, self.BOOT_U) else 0
        if k > 3:
            return len(self.boots)
        return 1 + int(round((len(self.boots) - 1) * self.prog(3, *self.BOOTS_U)))

    # ------------------------------------------------------------------ the picture
    def steady(self, sc, k, u, op):
        m = self.meta
        self.draw_strip(sc, k, u, op)
        self.draw_table(sc, op * self.at_u(5, self.TABLE_U, 0.6))
        # the plane, Minuit's minimum
        ch = self.plane()
        a = op * self.at_u(0, 0.1, 0.6)
        self.axes(sc, ch, a, ylabel="b", xticks=[(v, "%.1f" % v) for v in (2.2, 2.4, 2.6, 2.8)],
                  yticks=[(v, "%.1f" % v) for v in (0.8, 1.0, 1.2)], grid=False)
        self.text(sc, ch.x1 + 0.012, ch.y1, "τ", a, SIZE, "lm", ink=layout.CAPTION_INK)
        mx, my = self.at(m["values"])
        self.dot(sc, mx, my, INK, a, r=3.8, lift=1.75, ring=PAPER)
        lab = a * (1.0 - self.at_u(1, 0.05))
        self.text(sc, mx + 0.012, my - 0.03, "Minuit", lab, SIZE, "lm", ink=INK)
        # the seeds' answers: teal, the newest bigger
        n = self.n_seeds()
        for i in range(n):
            fx, fy = self.at(self.seeds[i])
            newest = i == n - 1 and k <= 1
            self.dot(sc, fx, fy, SEED, op * self.since(self.time_of(0, 0.45)) if i == 0 else op,
                     r=3.6 if newest else 2.4, lift=1.6)
        self.inset(sc, op * self.at_u(1, self.INSET_U, 0.6) * (1.0 - self.at_u(2, 0.0, 0.5)))
        # the resamples' answers: magenta, fainter once their ellipse is drawn
        nb = self.n_boots()
        dim = 1.0 - 0.55 * self.at_u(4, self.ELLIPSE_U[0], 0.8)
        for i in range(nb):
            fx, fy = self.at(self.boots[i])
            g = self.at_u(2, self.BOOT_U) if i == 0 else 1.0
            self.dot(sc, fx, fy, BOOT, op * g * dim, r=3.6 if (i == nb - 1 and k <= 3) else 2.5,
                     lift=1.55)
        if k == 2 and nb:
            fx, fy = self.at(self.boots[0])
            self.text(sc, fx + 0.012, fy, "resample 1", op * self.at_u(2, self.BOOT_U + 0.05),
                      SIZE, "lm", ink=BOOT_INK)
        # the cloud's ellipse and its shadow, then HESSE's
        self.draw_ellipse(sc, m["boots_ellipse"], BOOT_INK, op * (k >= 4),
                          self.prog(4, *self.ELLIPSE_U) if k == 4 else 1.0, width=2.6)
        mean = m["boots_mean"]
        self.cross(sc, mean, m["boots_spread"], BOOT_INK, op * self.at_u(4, self.BARS_U, 0.5),
                   width=2.2)
        self.draw_ellipse(sc, m["hesse_ellipse"], INK, op * (k >= 5),
                          self.prog(5, *self.HESSE_U) if k == 5 else 1.0, width=2.6)
        self.cross(sc, m["values"], m["hesse"], INK, op * self.at_u(5, self.HESSE_U[1], 0.5),
                   width=2.2)
        self.counter(sc, k, u, op)

    def inset(self, sc, op):
        """Phase 2: the seeds' answers round Minuit's minimum, magnified :data:`ZOOM` times."""
        if op <= 0.01:
            return
        m = self.meta
        cx, cy = INSET
        self.rect(sc, self.P(cx, cy), INSET_W, INSET_H, PAPER, op, border=AXIS, border_w=1.2,
                  radius=6, lift=0.2)
        hw, hh = self.frac(INSET_W / 2, INSET_H / 2)
        self.text(sc, cx - hw, cy - hh - 0.016, "magnified × %g" % ZOOM, op, SIZE, "lm",
                  ink=QUIET)
        mx, my = self.at(m["values"])
        for p in self.seeds:
            fx, fy = self.at(p)
            self.dot(sc, cx + (fx - mx) * ZOOM, cy + (fy - my) * ZOOM, SEED, op, r=2.4,
                     lift=1.6)
        self.dot(sc, cx, cy, INK, op, r=3.8, lift=1.75, ring=PAPER)

    def draw_strip(self, sc, k, u, op):
        """The 40 points across the top: all alike, except in phase 3 (resample 1: each
        point's size its number of picks, hollow when left out); fading out for the table."""
        a = op * self.at_u(0, 0.02, 0.6) * (1.0 - self.at_u(5, self.TABLE_U - 0.08, 0.5))
        if a <= 0.01:
            return
        ch = self.strip()
        self.axes(sc, ch, a, xticks=[], yticks=[], grid=False)
        self.text(sc, ch.x1, ch.y0 - 0.012, "40 made-up points", a, SIZE, "rb", ink=QUIET)
        picks = self.at_u(2, self.PICKS_U, 0.6) * (1.0 - self.at_u(3, 0.0, 0.6))
        for i, (x, y) in enumerate(zip(self.x, self.y, strict=True)):
            fx, fy = ch.fx(x), ch.fy(y)
            c = int(self.picks[i])
            if picks > 0.01 and c == 0:
                self.dot(sc, fx, fy, DATA, a * (1 - picks), r=2.4)
                self.hollow(sc, fx, fy, AXIS, a * picks, r=2.6)
            else:
                r = 2.4 + picks * (1.2 * (c - 1) if c > 0 else 0.0)
                self.dot(sc, fx, fy, DATA, a, r=r)
        if picks > 0.01:
            self.text(sc, ch.x0, 0.522, "bigger: picked more often · hollow: left out",
                      a * picks, SIZE, "lm", ink=QUIET)

    def draw_table(self, sc, op):
        """Phase 6: the spreads side by side, new seeds, resamples, HESSE."""
        if op <= 0.01:
            return
        m = self.meta
        heads = (("new seeds", SEED_INK), ("resamples", BOOT_INK), ("HESSE", INK))
        for (s, ink), x in zip(heads, TABLE_X[1:], strict=True):
            self.text(sc, x, TABLE_Y[0], s, op, SIZE, "mm", ink=ink)
        cols = (m["seeds_text"], m["boots_text"], m["hesse_text"])
        for i, name in enumerate(m["names"]):
            y = TABLE_Y[1 + i]
            self.text(sc, TABLE_X[0], y, name, op, SIZE, "lm", strong=True)
            for col, x, (_h, ink) in zip(cols, TABLE_X[1:], heads, strict=True):
                self.text(sc, x, y, "± " + col[i], op, SIZE, "mm", ink=ink)

    def draw_ellipse(self, sc, pts, ink, op, u, width=1.8):
        if op <= 0.01 or u <= 0.0:
            return
        pts = np.asarray(pts)
        n = max(2, int(np.ceil(len(pts) * min(u, 1.0))))
        self.guide(sc, [self.at((0.0, p[0], p[1])) for p in pts[:n]], ink, op, width)

    def cross(self, sc, centre, spread, ink, op, width=1.6):
        """Error bars through ``centre``: one standard deviation each way along tau and b,
        with end caps."""
        if op <= 0.01:
            return
        c = np.asarray(centre, float)
        for axis in (1, 2):
            d = np.zeros(3)
            d[axis] = spread[axis]
            a, b = self.at(c - d), self.at(c + d)
            self.guide(sc, [a, b], ink, op, width)
            cap = self.frac(4.0, 4.0)
            for e in (a, b):
                if axis == 1:
                    self.guide(sc, [(e[0], e[1] - cap[1]), (e[0], e[1] + cap[1])], ink, op, width)
                else:
                    self.guide(sc, [(e[0] - cap[0], e[1]), (e[0] + cap[0], e[1])], ink, op, width)

    def counter(self, sc, k, u, op):
        m = self.meta
        if k == 0:
            rows = [("run 1, seed 0", SIZE, INK), ("τ = " + m["first_text"][1], HEAD, SEED_INK),
                    ("b = " + m["first_text"][2], HEAD, SEED_INK)]
            op *= self.at_u(0, 0.45)
        elif k == 1:
            n = self.n_seeds()
            if self.after(1, self.SEEDS_U[1] + 0.05):
                rows = [("100 new seeds", SIZE, INK), ("± " + m["seeds_round"], HEAD, SEED_INK)]
            else:
                rows = [("run %d of 100" % n, SIZE, INK), ("seed %d" % (n - 1), HEAD, SEED_INK),
                        ("same 40 points", SIZE, QUIET)]
        elif k == 2:
            rows = [("resample 1", SIZE, INK),
                    ("%d left out" % m["resample_out"], SIZE, QUIET),
                    ("%d twice or more" % m["resample_twice"], SIZE, DATA_INK)]
        elif k == 3 and not self.after(3, self.BOOTS_U[1] + 0.05):
            rows = [("resample %d of 100" % max(self.n_boots(), 1), SIZE, INK),
                    ("seed %d" % (99 + max(self.n_boots(), 1)), HEAD, BOOT_INK),
                    ("new data each run", SIZE, QUIET)]
        elif k <= 4:
            rows = [("100 resamples", SIZE, INK), ("τ ± " + m["boots_text"][1], HEAD, BOOT_INK),
                    ("b ± " + m["boots_text"][2], HEAD, BOOT_INK)]
            if k == 4:
                rows.append(("tilt " + m["corr_boots_text"], SIZE, BOOT_INK))
        else:
            rows = [("HESSE", SIZE, INK), ("τ ± " + m["hesse_text"][1], HEAD, INK),
                    ("b ± " + m["hesse_text"][2], HEAD, INK),
                    ("tilt " + m["corr_hesse_text"], SIZE, INK)]
            if not self.after(5, self.HESSE_U[1]):
                rows = [("100 resamples", SIZE, INK),
                        ("τ ± " + m["boots_text"][1], HEAD, BOOT_INK),
                        ("b ± " + m["boots_text"][2], HEAD, BOOT_INK),
                        ("tilt " + m["corr_boots_text"], SIZE, BOOT_INK)]
        self.lines_at(sc, rows, op)


# ====================================================================== 278
HIST = ((0.10, 0.30), (0.42, 0.655))          # the counts: 4 bins
COUNT_MAX = 45.0
ZBOX = ((0.10, 0.455), (0.765, 0.855))        # the toys' significance
BAR_W = 34.0                                  # a count bar's width, px
MARK_Y = (0.745, 0.728)                       # the Asimov and s/sqrt(b) dots, over the Z chart


class Asimov(Base):
    """The expected counts; one toy; 10,000 toys; the Asimov dataset; s / sqrt(b); a tenth of
    the background."""

    TITLE = "The Asimov dataset"
    PHASES = (("expected", 9.0), ("one toy", 10.0), ("10,000 toys", 10.0), ("Asimov", 10.0),
              ("s/√b", 9.0), ("less background", 12.0))
    LAST = "Unfolding: running the smearing backwards"
    NEXT = "Jets as sets of particles"
    COUNT_Y = (0.445, 0.49, 0.535, 0.58)
    TOY_U = 0.15                    # phase 2: toy 1's counts
    Z1_U = 0.55                     # phase 2: its Z
    GROW_U = (0.05, 0.6)            # phase 3: the toys pile up
    MEDIAN_U = 0.68                 # phase 3: their median
    ASIMOV_U = 0.12                 # phase 4: the Asimov counts
    ZA_U = 0.5                      # phase 4: its Z
    SIMPLE_U = 0.25                 # phase 5: s / sqrt(b)
    SHRINK_U = (0.05, 0.28)         # phase 6: the background shrinks
    MORPH_U = (0.33, 0.55)          # phase 6: the new toys
    MOVE_U = (0.6, 0.72)            # phase 6: the Asimov dot moves
    JUMP_U = (0.76, 0.88)           # phase 6: and s / sqrt(b)

    def last_line(self):
        """The catalogue's title of 277 is too wide for the heading: its first word."""
        return "last time: " + self.LAST.split(":")[0]

    def setup(self):
        super().setup()
        m = self.meta
        self.s = np.asarray(m["s"])
        self.b = np.asarray(m["b"])
        self.b_less = np.asarray(m["b_less"])
        self.z = np.asarray(m["z"])
        self.hist = np.asarray(m["hist"], float)
        self.hist_less = np.asarray(m["hist_less"], float)
        self.edges = np.asarray(m["edges"])
        self.peak = max(self.hist.max(), self.hist_less.max())
        less = "background %s" % _np_print(self.b_less)
        self.CODE = (
            Code("as-counts", run=((0.0, 1), (0.4, 0))),
            Code("as-z", run=((0.0, 0), (0.4, 1), (0.5, 3))),
            Code("as-toys", run=((0.0, 1), (0.05, 2), (0.65, 3)), out=((0.68, m["median_text"]),)),
            Code("as-asimov", run=((0.0, 0), (0.45, 1)), out=((0.5, m["asimov_text"]),)),
            Code("as-simple", run=((0.0, 0), (0.25, 1)), out=((0.25, m["simple_text"]),)),
            Code("as-less", run=((0.0, 0), (0.05, 2)),
                 out=((0.05, less), (0.5, less + "\n" + m["median_text_less"]),
                      (0.62, less + "\n%s\n%s" % (m["median_text_less"],
                                                   m["asimov_text_less"])),
                      (0.78, less + "\n%s\n%s\n%s" % (m["median_text_less"],
                                                       m["asimov_text_less"],
                                                       m["simple_text_less"])))))
        self.EVIDENCE = (
            exact("s + b in each bin, made up"),
            measured("toy 1, seed 7", ON),
            measured("10,000 toys, seed 7", ON),
            exact("Z of the Asimov dataset, n = s + b"),
            exact("s / √b bin by bin, added in quadrature"),
            measured("10,000 toys with b ÷ 10, seed 8", ON))

    def heading_for(self, k, u):
        return ("what we expect: background and signal",
                "one toy experiment: counts at random",
                "10,000 toys: their median is expected",
                "Asimov: every bin exactly as expected",
                "the quick guess, s/√b",
                "a tenth of the background: s/√b fails")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "A made-up analysis: 4 bins, background b (grey) and signal s (amber) expected in "
            "each. A real experiment sees random counts.",
            "A toy experiment: random counts round s + b. Fitting the signal gives its "
            "significance, Z = %s σ: how far it is from background alone." % m["first_z_text"],
            "%s toys give %s values of Z. Their median, %s σ, is the significance we expect."
            % (m["toys_text"], m["toys_text"], m["median_text"]),
            "The Asimov dataset puts exactly s + b in each bin. One fit gives %s σ, on the "
            "toys' median, with no toys." % m["asimov_text"],
            "The quick guess, s / √b bin by bin, gives %s σ: close, a little high, when the "
            "background is large." % m["simple_text"],
            "With a tenth of the background, s / √b says %s σ. Asimov's %s still lands on the "
            "toys' median, %s." % (m["simple_text_less"], m["asimov_text_less"],
                                   m["median_text_less"]),
        ][k]

    # ------------------------------------------------------------------ charts
    def counts(self):
        return LossChart(HIST, (0.0, 4.0), (0.0, COUNT_MAX))

    def zchart(self):
        return LossChart(ZBOX, tuple(self.meta["z_range"]), (0.0, 1.0))

    def shrink(self):
        """0 with the background as it is, 1 with a tenth of it."""
        return self.prog(5, *self.SHRINK_U) if self.now()[0] == 5 else 0.0

    def background(self):
        g = self.shrink()
        return self.b + (self.b_less - self.b) * g

    # ------------------------------------------------------------------ the picture
    def steady(self, sc, k, u, op):
        ch = self.counts()
        a = op * self.at_u(0, 0.04, 0.6)
        self.axes(sc, ch, a, ylabel="events", xticks=[],
                  yticks=[(v, "%d" % v) for v in (0, 20, 40)], grid=False)
        self.text(sc, ch.x1 + 0.008, ch.y1 + 0.022, "mass", a, SIZE, "rm",
                  ink=layout.CAPTION_INK)
        back = self.background()
        gb = self.prog(0, 0.08, 0.3)
        gs = self.prog(0, 0.35, 0.55)
        tops = []
        for i in range(4):
            cx = ch.fx(i + 0.5)
            yb = ch.fy(back[i] * gb)
            ys = ch.fy(back[i] * gb + self.s[i] * gs)
            y0 = ch.y1
            hb = self.span(0, y0 - yb)[1]
            hs = self.span(0, yb - ys)[1]
            if hb > 0.5:
                self.rect(sc, self.P(cx, (y0 + yb) / 2), BAR_W, hb, BACK, a, radius=2.0,
                          lift=0.3)
            if hs > 0.5:
                self.rect(sc, self.P(cx, (yb + ys) / 2), BAR_W, hs, SIGNAL, a, radius=2.0,
                          lift=0.35)
            tops.append((cx, ys))
        # the expected counts over the bars (phase 1, and again for the Asimov dataset)
        exp_op = a * (self.at_u(0, 0.6) * (1.0 - self.at_u(1, self.TOY_U - 0.05)))
        expected = back + self.s
        for (cx, ys), v in zip(tops, expected, strict=True):
            self.text(sc, cx, ys - 0.022, _num(v), exp_op, SIZE, "mm", ink=layout.CAPTION_INK)
        self.toy_counts(sc, k, tops, op)
        self.asimov_counts(sc, k, tops, op)
        self.z_panel(sc, k, u, op)
        self.counter(sc, k, op)

    def toy_counts(self, sc, k, tops, op):
        """Toy 1's counts: blue dots at their heights, each number over it (phase 2), then
        faded for the toys' pile-up."""
        g = op * self.at_u(1, self.TOY_U) * (1.0 - self.at_u(2, 0.2, 0.6))
        if g <= 0.01:
            return
        ch = self.counts()
        for (cx, _ys), n in zip(tops, self.meta["first"], strict=True):
            fy = ch.fy(n)
            self.dot(sc, cx, fy, DATA, g, r=4.2, lift=1.6)
            self.text(sc, cx, fy - 0.03, "%d" % n, g, SIZE, "mm", ink=DATA_INK)

    def asimov_counts(self, sc, k, tops, op):
        """The Asimov dataset: a green-ringed blue dot exactly on each bar's top, its count over
        it (phase 4 on)."""
        g = op * self.at_u(3, self.ASIMOV_U)
        if g <= 0.01:
            return
        # while the background shrinks (phase 6) the old counts fade out and the new fade in
        s = self.shrink()
        label = g * float(np.clip(abs(1.0 - 2.0 * s) * 1.6 - 0.6, 0.0, 1.0))
        final = self.s + (self.b if s < 0.5 else self.b_less)
        for (cx, ys), v in zip(tops, final, strict=True):
            self.dot(sc, cx, ys, DATA, g, r=4.2, lift=1.6, ring=ASIMOV)
            self.text(sc, cx, ys - 0.032, _num(v), label, SIZE, "mm", ink=DATA_INK)

    def z_panel(self, sc, k, u, op):
        """The toys' significance: an axis from 0 to 11, toy 1's dot, the histogram of the
        toys so far, their median, the Asimov dot and the s / sqrt(b) dot."""
        m = self.meta
        zc = self.zchart()
        a = op * self.at_u(1, self.Z1_U - 0.1, 0.6)
        if a <= 0.01:
            return
        self.axes(sc, zc, a, ylabel="toys", xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8, 10)],
                  yticks=[], grid=False)
        self.text(sc, zc.x1 + 0.008, zc.y1, "Z", a, SIZE, "lm", ink=layout.CAPTION_INK)
        # toy 1 alone
        g1 = a * (1.0 - self.at_u(2, 0.1))
        self.dot(sc, zc.fx(m["first_z"]), zc.y1 - 0.012, DATA, g1, r=4.0, lift=1.6)
        # the histogram: the toys so far (phase 3), then morphing to the new toys (phase 6)
        if k >= 2:
            if k == 2:
                n = int(round(10 ** (4 * self.prog(2, *self.GROW_U))))
                h = np.histogram(np.clip(self.z[:n], self.edges[0], self.edges[-1] - 1e-9),
                                 self.edges)[0].astype(float)
                # the few first toys stand as tall as fits; from a few hundred on, the bars
                # grow on the final scale
                h = h * self.peak / max(h.max(), self.peak * n / len(self.z))
            else:
                g = self.prog(5, *self.MORPH_U) if k == 5 else 0.0
                h = self.hist + (self.hist_less - self.hist) * g
            self.bars(sc, zc, h, op)
        # the median line, then the Asimov dot and the s / sqrt(b) dot
        med = self.blend(m["median"], m["median_less"], self.MORPH_U)
        mo = op * self.at_u(2, self.MEDIAN_U)
        if mo > 0.01:
            fx = zc.fx(med)
            self.dashed(sc, (fx, zc.y1), (fx, zc.y0 - 0.005), INK, mo, n=8, width=1.6)
        za = self.blend(m["asimov_z"], m["asimov_z_less"], self.MOVE_U)
        self.marker(sc, zc, za, MARK_Y[0], ASIMOV, op * self.at_u(3, self.ZA_U))
        sim = self.blend(m["simple"], m["simple_less"], self.JUMP_U)
        self.marker(sc, zc, sim, MARK_Y[1], SIMPLE, op * self.at_u(4, self.SIMPLE_U))

    def marker(self, sc, zc, z, fy, ink, op):
        """A significance marked over the toys: a dot over the chart, a line down to Z."""
        if op <= 0.01:
            return
        fx = zc.fx(z)
        self.guide(sc, [(fx, fy), (fx, zc.y1)], ink, 0.8 * op, width=1.4)
        self.dot(sc, fx, fy, ink, op, r=4.4, lift=1.7)

    def blend(self, first, second, span):
        """``first``, moving to ``second`` over ``span`` of phase 6."""
        g = self.prog(5, *span) if self.now()[0] == 5 else 0.0
        return first + (second - first) * g

    def bars(self, sc, zc, h, op):
        top = self.span(0, zc.y1 - zc.y0)[1]
        w = self.span(zc.fx(self.edges[1]) - zc.fx(self.edges[0]))[0] - 1.0
        for i, v in enumerate(h):
            hh = top * min(float(v) / self.peak, 1.0)          # never out of its panel
            if hh < 0.4:
                continue
            cx = zc.fx((self.edges[i] + self.edges[i + 1]) / 2)
            fy = zc.y1 - self.frac(0, hh)[1] / 2
            self.rect(sc, self.P(cx, fy), w, hh, soft(DATA, 0.35), op, radius=1.0, lift=0.3)

    def counter(self, sc, k, op):
        m = self.meta
        less_med = self.after(5, self.MORPH_U[1])
        less_za = self.after(5, self.MOVE_U[1])
        less_sim = self.after(5, self.JUMP_U[1])
        med = m["median_text_less"] if less_med else m["median_text"]
        za = m["asimov_text_less"] if less_za else m["asimov_text"]
        sim = m["simple_text_less"] if less_sim else m["simple_text"]
        if k == 0:
            rows = [("signal s", SIZE, SIGNAL_INK), ("background b", SIZE, BACK_INK),
                    ("4 made-up bins", SIZE, QUIET)]
            op *= self.at_u(0, 0.1)
        elif k == 1:
            rows = [("toy 1", SIZE, INK)]
            if self.after(1, self.Z1_U):
                rows += [("Z = %s σ" % m["first_z_text"], HEAD, DATA_INK),
                         ("signal × %s" % m["first_mu_text"], SIZE, QUIET)]
        elif k == 2:
            n = int(round(10 ** (4 * self.prog(2, *self.GROW_U))))
            rows = [("%s toy%s" % (format(n, ","), "" if n == 1 else "s"), SIZE, INK)]
            if self.after(2, self.MEDIAN_U):
                rows += [("median %s" % med, HEAD, INK)]
        elif k == 3:
            rows = [("median %s" % med, SIZE, INK)]
            if self.after(3, self.ZA_U):
                rows += [("Asimov %s" % za, HEAD, ASIMOV_INK)]
        else:
            rows = [("median %s" % med, SIZE, INK), ("Asimov %s" % za, HEAD, ASIMOV_INK)]
            if k > 4 or self.after(4, self.SIMPLE_U):
                rows += [("s/√b %s" % sim, HEAD, SIMPLE_INK)]
        self.lines_at(sc, rows, op)


def _num(v):
    v = float(v)
    return "%d" % round(v) if abs(v - round(v)) < 0.05 else "%.1f" % v


def _np_print(a):
    return str(np.asarray(a))


ANIMATIONS = {"error-bars-on-a-network": ErrorBars, "asimov-dataset": Asimov}
