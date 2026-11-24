"""Animation 277 of "Blocks and code" (see :mod:`voxanim.blocks`), "Unfolding: running the
smearing backwards", drawn from :mod:`voxanim.study_blocks_zi` (``examples/unfolding.py``,
made-up events). Nothing is trained, so the loop map lights no station.

:class:`Unfolding` (``unfolding``): the made-up true spectrum, 10 bins of 10 GeV (10,000
events, falling, a bump at 60 GeV); it moves up to sit over the response matrix R's columns,
and R is built from simulation, as the resizing animations' matrices are drawn (orange
tiles): 500 of the simulated (true, measured) pairs as dots, counted by (measured bin, true
bin), each column divided by the events simulated in its true bin; the bump's bin spread down
its column into the measured bins beside R's rows, then all of them, R @ truth = 6,345
expected, and one Poisson draw, 6,362; inverting R: the expected counts give the truth back,
the data swing from -900 to 3,475; why, row by row: R's singular values (0.75 to 0.0078),
this draw's noise along each direction, and the noise divided by them (up to 2,759);
D'Agostini's iterations from the simulation's spectrum, 0 to 10, closing in on the truth;
bias and spread over 2,000 pseudo-experiments against the number of iterations.

The alphabet: the truth is a dark outline (bars filled pale at first); what comes from
simulation is orange (R's tiles, the dots, the prior); measured counts are blue; R inverted
is red; the unfolded spectrum green; bias violet, spread teal.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import BIG, DATA, HEAD, INK, LOSS, QUIET, SIZE, WEIGHT, Code, exact, measured
from .easing import smoothstep
from .losschart import LossChart
from .mech import AXIS, GREEN, PAPER, RED, TEAL, dark, soft
from .mech_blocks_zb import Resize, during, eased, on
from .study_blocks_zi import ON

TRUTH = INK
TRUTH_FILL = soft(INK, 0.75)
SIM = WEIGHT                              # from simulation: R, the dots, the prior
SIM_INK = dark(WEIGHT, 0.35)
MEAS = DATA                               # measured counts
MEAS_INK = dark(DATA, 0.3)
INV = RED                                 # R inverted
INV_INK = dark(RED, 0.2)
UNF = GREEN                               # unfolded
UNF_INK = dark(GREEN, 0.3)
BIAS = LOSS
BIAS_INK = dark(LOSS, 0.3)
SPREAD = TEAL
SPREAD_INK = dark(TEAL, 0.35)

# ---------------------------------------------------------------------- the layout
BIG_BOX = ((0.12, 0.455), (0.44, 0.735))   # the spectrum charts: (x0, x1), (top, bottom)
MAT = (0.215, 0.665)                      # R's centre
MAT_PX = 144.0                            # its side, px
TOP_BOX = ((0.14, 0.29), (0.395, 0.515))  # the truth over R's columns
RIGHT_X = 0.302                           # the measured bars beside R's rows start here
RIGHT_LEN = 0.13 / 1000.0                 # their length per event, layout x
COUNT_MAX = 2800.0                        # the truth's charts go up to this
LINE1, LINE2 = 0.85, 0.885                # the two lines of words under a chart
TABLE_TOP, TABLE_STEP = 0.475, 0.037      # 5: the rows
S_X, S_LEN = 0.122, 0.07                  # 5: the singular values' bars (0.75: S_LEN)
N_X, N_LEN = 0.255, 0.035 / 75.0          # 5: the noise's bars, from a zero line
B_X, B_LEN = 0.385, 0.072 / 2800.0        # 5: the noise / s bars, from a zero line


def hold(u, a, b):
    """0 before ``a``, eased to 1 at ``b``."""
    return eased(u, a, b)


class Unfolding(Resize):
    """The truth; R from simulation; measured; inverted; why; iterations; bias and spread."""

    TITLE = "Unfolding: running the smearing backwards"
    PHASES = (("truth", 7.0), ("R from simulation", 10.0), ("measured", 8.5),
              ("inverted", 8.5), ("why", 9.5), ("iterations", 9.0), ("bias and spread", 8.5))
    LAST = "Simulation is not data: scale factors"
    NEXT = "The Asimov dataset"
    STATION = None
    ITER_U = (0.24, 0.84)          # 6: iterations 1 to 10 come in between these
    CURVE_U = (0.1, 0.5)           # 7: the curves draw themselves between these

    def setup(self):
        m = self.meta = self.study.metadata
        self.truth = np.asarray(m["truth"], float)
        self.R = np.asarray(m["R"], float)
        self.H = np.asarray(m["H"], float)
        self.expected = np.asarray(m["expected"], float)
        self.data = np.asarray(m["data"], float)
        self.naive = np.asarray(m["naive"], float)
        self.back = np.asarray(m["back"], float)
        self.iterates = np.asarray(m["iterates"], float)
        self.edges = np.asarray(m["edges"], float)
        self.bump = int(m["bump_bin"])
        p = m["printed"]
        self.printed = p
        self.CODE = (
            Code("uf-truth", run=((0.0, 0), (0.3, 1)), out=((0.6, p[0]),)),
            Code(("uf-simulate", "uf-response"), pick=(0, 2, 3, 4, 5, 6, 7, 8),
                 run=((0.0, 0), (0.12, 1), (0.2, 2), (0.28, 3), (0.45, 4), (0.62, 6),
                      (0.7, 7)), out=((0.5, p[1]),)),
            Code("uf-measure", run=((0.0, 0), (0.62, 1)), out=((0.7, p[2]),)),
            Code("uf-invert", run=((0.0, 0), (0.45, 1)),
                 out=((0.2, p[3]), (0.5, p[3] + "\n" + p[4]))),
            Code("uf-svd", run=((0.0, 0), (0.3, 1), (0.55, 2)), out=((0.1, p[5]),)),
            Code(("uf-prior", "uf-bayes"),
                 run=((0.0, 1), (0.12, 3), (0.2, 5), (0.27, 6), (0.86, 7)),
                 out=((0.12, p[6]), (self.iter_u(1), p[7]), (self.iter_u(4), p[8]),
                      (self.iter_u(10), p[9]))),
            Code("uf-toys", run=((0.0, 0), (0.1, 2), (0.3, 3), (0.45, 4)),
                 out=((0.6, self.toy_line(m["best"])),)))
        self.EVIDENCE = (
            exact("the made-up spectrum's expected counts"),
            measured("R from 964,628 simulated", ON),
            measured("R @ truth, and one Poisson draw", ON),
            measured("R inverted: expected, one draw", ON),
            measured("R's singular values; the noise", ON),
            measured("D'Agostini's iterations, one draw", ON),
            measured("2,000 pseudo-experiments", ON))

    def toy_line(self, k):
        m = self.meta
        line = "%d iterations: bias %s, spread %s" % (k, m["bias_text"][k], m["spread_text"][k])
        if line not in self.printed:
            raise ValueError("the script does not print %r" % line)
        return line

    # ------------------------------------------------------------------ no training
    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the long title), as
        in :mod:`voxanim.mech_blocks_zd`."""
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

    def heading_for(self, k, u):
        return ("the truth: a made-up spectrum",
                "R, the response matrix, from simulation",
                "measured = R @ truth, plus noise",
                "invert R: exact without noise, wild with it",
                "why: R's small singular values",
                "iterative Bayesian unfolding",
                "few iterations: bias; many: spread")[k]

    def subtitle_for(self, k):
        m = self.meta
        best = m["best"]
        return [
            "Made-up events: %s with energies falling from 0 GeV, plus a bump at 60 GeV, "
            "in 10 bins of 10 GeV. This is the truth." % m["truth_total_text"],
            "The detector's response matrix R is built from simulation, as everything in HEP "
            "is: column j shows where events of true bin j are measured.",
            "Each true bin spreads into its neighbours, and %s of the events are lost. One "
            "Poisson draw adds noise of about ±%s a bin." % (m["lost_text"],
                                                              m["data_noise_text"]),
            "Solving R @ truth = expected gives the truth back exactly. Solving it for the "
            "data turns noise of ±%s into swings of thousands." % m["data_noise_text"],
            "R's singular values s fall from %s to %s. Inverting divides the noise along "
            "each direction by s: the last ones blow up." % (m["s_text"][0], m["s_text"][-1]),
            "D'Agostini's iterative Bayesian unfolding starts from the simulation's spectrum, "
            "then folds it with R, compares with the data, corrects.",
            "Over %s pseudo-experiments: more iterations, less bias but more spread. "
            "Together they are smallest near %d; R inverted spreads %s." % (
                "{:,}".format(m["toys"]), best, m["inverse_spread_text"]),
        ][k]

    # ------------------------------------------------------------------ time
    def iter_u(self, j):
        """Phase 6: when iteration ``j`` (1 to 10) comes in."""
        a, b = self.ITER_U
        return a + (b - a) * (j - 1) / 9.0

    def iteration(self, u):
        """Phase 6: the iteration shown (0 to 10)."""
        return sum(u >= self.iter_u(j) for j in range(1, 11))

    def after(self, k, phase, u, at):
        """1 once ``phase`` has reached ``at`` (eased over half a second), 0 before; 1 in the
        phases after it."""
        if k < phase:
            return 0.0
        return self.at(phase, at) if k == phase else 1.0

    # ------------------------------------------------------------------ the matrix
    def mat_box(self):
        w, h = self.frac(MAT_PX, MAT_PX)
        return MAT[0] - w / 2, MAT[1] - h / 2, MAT[0] + w / 2, MAT[1] + h / 2

    def mat_cell(self):
        return self.cells(*MAT, MAT_PX, MAT_PX, 10, 10)

    def draw_matrix(self, sc, op, mix, tile_op):
        """R's frame, and its tiles (``tile_op``): the counts H (``mix`` 0) turning into R
        (``mix`` 1), each shaded on its own scale."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(*MAT), MAT_PX + 8, MAT_PX + 8, PAPER, op, border=SIM,
                  border_w=1.6, radius=3, lift=0.2)
        if mix is None or tile_op <= 0.01:
            return
        v = (1 - mix) * self.H / self.H.max() + mix * self.R / self.R.max()
        self.tiles(sc, v, *MAT, MAT_PX, MAT_PX, tile_op, ink=SIM, lo=0.0, hi=1.0, grid=0.2)

    # ------------------------------------------------------------------ histograms
    def chart(self, box, y_range, x_range=(0.0, 100.0)):
        return LossChart(box, x_range, y_range)

    def bars(self, sc, ch, values, ink, op, fill=0.6, grow=1.0, only=None):
        """A histogram: one bar per 10 GeV bin from 0 (down, for a negative count)."""
        if op <= 0.01:
            return
        for i, v in enumerate(values):
            if only is not None and i not in only:
                continue
            a, b = self.edges[i], self.edges[i + 1]
            y0, y1 = ch.fy(0.0), ch.fy(float(v) * grow)
            top, bottom = min(y0, y1), max(y0, y1)
            if bottom - top < 0.002:
                continue
            x0, x1 = ch.fx(a), ch.fx(b)
            w, h = self.span(x1 - x0, bottom - top)
            self.rect(sc, self.P((x0 + x1) / 2, (top + bottom) / 2), max(w - 2.0, 1.0), h, ink,
                      op, border=ink, border_w=1.0, radius=1.0, lift=0.4, fill_op=fill)

    def outline(self, sc, ch, values, ink, op, width=2.2):
        """A histogram as a step line in front of the bars."""
        if op <= 0.01:
            return
        pts = [(ch.fx(self.edges[0]), ch.fy(0.0))]
        for i, v in enumerate(values):
            y = ch.fy(float(v))
            pts += [(ch.fx(self.edges[i]), y), (ch.fx(self.edges[i + 1]), y)]
        pts.append((ch.fx(self.edges[-1]), ch.fy(0.0)))
        self.polyline(sc, [self.P(x, y, self.ARROW_LIFT) for x, y in pts], width, ink, op,
                      bias=0.5)

    def zero_line(self, sc, ch, op):
        if op > 0.01:
            y = ch.fy(0.0)
            self.polyline(sc, [self.P(ch.x0, y, 0.3), self.P(ch.x1, y, 0.3)], 1.0, AXIS,
                          0.8 * op, bias=0.0)

    def energy_axes(self, sc, ch, op, yticks):
        self.axes(sc, ch, op, xlabel="true energy, GeV", ylabel="events",
                  xticks=[(0, "0"), (50, "50"), (100, "100")],
                  yticks=[(v, f"{v:,}".replace("-", "−")) for v in yticks], grid=False)

    def side_bars(self, sc, values, ink, op, fill=0.6, grow=1.0):
        """Measured counts as bars beside R's rows, growing right."""
        if op <= 0.01:
            return
        cell = self.mat_cell()
        th = MAT_PX / 10 - 3.0
        for i, v in enumerate(values):
            length = RIGHT_LEN * float(v) * grow
            if length < 0.002:
                continue
            y = cell(i, 0)[1]
            w = self.span(length)[0]
            self.rect(sc, self.P(RIGHT_X + length / 2, y), w, th, ink, op, border=ink,
                      border_w=1.0, radius=1.0, lift=0.4, fill_op=fill)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.truth_chart(sc, k, u, op)
        self.matrix_part(sc, k, u, op)
        self.inverted_chart(sc, k, u, op)
        self.table(sc, k, u, op)
        self.iterations_chart(sc, k, u, op)
        self.toys_chart(sc, k, u, op)

    def truth_chart(self, sc, k, u, op):
        """Phases 1-3: the truth, big, then over R's columns."""
        a = op * self.leaving(3, k, u)
        if a <= 0.01:
            return
        s = 0.0 if k == 0 else (hold(u, 0.02, 0.14) if k == 1 else 1.0)
        (bx0, bx1), (by0, by1) = BIG_BOX
        (tx0, tx1), (ty0, ty1) = TOP_BOX
        box = ((bx0 + (tx0 - bx0) * s, bx1 + (tx1 - bx1) * s),
               (by0 + (ty0 - by0) * s, by1 + (ty1 - by1) * s))
        ch = self.chart(box, (0.0, COUNT_MAX))
        self.energy_axes(sc, ch, a * (1 - s), (0, 1000, 2000))
        self.zero_line(sc, ch, a * s)
        self.bars(sc, ch, self.truth, TRUTH_FILL, a, fill=0.85)
        self.outline(sc, ch, self.truth, TRUTH, a, width=2.2 - 0.8 * s)
        if s > 0.5:
            self.text(sc, tx0 - 0.008, (ty0 + ty1) / 2 + 0.02, "truth", a * (2 * s - 1), SIZE,
                      "rm", ink=INK)
        # the bump's bin, ringed in phase 3
        r = a * (during(u, 0.04, 0.4, 1.0) if k == 2 else 0.0)
        if r > 0.01:
            i = self.bump
            x0, x1 = ch.fx(self.edges[i]), ch.fx(self.edges[i + 1])
            y0, y1 = ch.fy(self.truth[i]), ch.fy(0.0)
            w, h = self.span(x1 - x0, y1 - y0)
            self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w + 5, h + 5, INK, r,
                      border=INK, border_w=2.2, radius=2, lift=1.2, fill_op=0.0)

    def matrix_part(self, sc, k, u, op):
        """Phases 2-3: R's frame, its labels, its tiles; the measured bars beside it."""
        a = op * self.after(k, 1, u, 0.1) * self.leaving(3, k, u)
        if a <= 0.01:
            return
        mix = None
        if k == 1 and u >= 0.44:
            mix = hold(u, 0.62, 0.74)
        elif k >= 2:
            mix = 1.0
        tile_op = a * (self.at(1, 0.44) if k == 1 else 1.0)
        self.draw_matrix(sc, a, mix, tile_op)
        x0, y0, _x1, y1 = self.mat_box()
        self.text(sc, x0 - 0.008, y0 + 0.07, "measured", a, SIZE, "rm", ink=QUIET)
        self.text(sc, x0 - 0.008, y0 + 0.105, "bin ↓", a, SIZE, "rm", ink=QUIET)
        self.text(sc, MAT[0], y1 + 0.03, "true bin →", a, SIZE, ink=QUIET)
        ra = a * (self.at(1, 0.78) if k == 1 else 1.0)
        self.text(sc, x0 - 0.012, y1 - 0.03, "R", ra, BIG, "rm", ink=SIM_INK)
        # the bump's column, ringed in phase 3
        if k == 2:
            r = a * during(u, 0.04, 0.4, 1.0)
            self.frame(sc, self.mat_cell(), 0, self.bump, 10, 1, MAT_PX / 10, MAT_PX / 10, r,
                       ink=INK, width=2.2)
        # beside the rows: the measured counts (phase 3)
        if k == 2:
            self.text(sc, RIGHT_X, y0 - 0.02, "measured", a * self.at(2, 0.06), SIZE, "lm",
                      ink=MEAS_INK)
            col = self.R[:, self.bump] * self.truth[self.bump]
            self.side_bars(sc, col, MEAS, a * during(u, 0.06, 0.42, 1.0), fill=0.3,
                           grow=hold(u, 0.1, 0.26))
            self.side_bars(sc, self.expected, MEAS, a * self.at(2, 0.4), fill=0.12,
                           grow=hold(u, 0.4, 0.55))
            self.side_bars(sc, self.data, MEAS, a * self.at(2, 0.64), fill=0.65)
        elif k == 3:
            self.text(sc, RIGHT_X, y0 - 0.02, "measured", a, SIZE, "lm", ink=MEAS_INK)
            self.side_bars(sc, self.data, MEAS, a, fill=0.65)

    def inverted_chart(self, sc, k, u, op):
        """Phase 4: the truth's outline, R inverted without noise (dots) and with (bars)."""
        a = op * self.after(k, 3, u, 0.07) * self.leaving(4, k, u)
        if a <= 0.01:
            return
        ch = self.chart(BIG_BOX, (-1000.0, 3700.0))
        self.energy_axes(sc, ch, a, (-1000, 0, 2000))
        self.zero_line(sc, ch, a)
        g = 1.0 if k > 3 else hold(u, 0.45, 0.62)
        self.bars(sc, ch, self.naive, INV, a * (self.at(3, 0.45) if k == 3 else 1.0), fill=0.55,
                  grow=g)
        self.outline(sc, ch, self.truth, TRUTH, a)
        d = a * (self.at(3, 0.16) if k == 3 else 1.0)
        if d > 0.01:
            for i, v in enumerate(self.back):
                x = ch.fx((self.edges[i] + self.edges[i + 1]) / 2)
                self.disk(sc, self.P(x, ch.fy(float(v))), 4.0, UNF, d, lift=1.6, outline=0.15)

    def table(self, sc, k, u, op):
        """Phase 5: per direction of R, its singular value, the noise along it, and the noise
        divided by it."""
        m = self.meta
        a = op * self.after(k, 4, u, 0.07) * self.leaving(5, k, u)
        if a <= 0.01:
            return
        s = np.asarray(m["s"])
        noise = np.asarray(m["noise"])
        blown = np.asarray(m["blown"])
        th = 12.0
        hy = TABLE_TOP - 0.045
        sa = a * (self.at(4, 0.08) if k == 4 else 1.0)
        na = a * (self.at(4, 0.34) if k == 4 else 1.0)
        self.text(sc, S_X + S_LEN / 2, hy, "s", sa, SIZE, ink=SIM_INK)
        self.text(sc, N_X, hy, "noise", na, SIZE, ink=MEAS_INK)
        self.text(sc, B_X, hy, "noise ÷ s", a * (self.at(4, 0.52) if k == 4 else 1.0), SIZE,
                  ink=INV_INK)
        for x in (N_X, B_X):
            top, bottom = TABLE_TOP - 0.02, TABLE_TOP + 9 * TABLE_STEP + 0.02
            self.polyline(sc, [self.P(x, top, 0.3), self.P(x, bottom, 0.3)], 1.0, AXIS,
                          0.7 * (na if x == N_X else a * (self.at(4, 0.52) if k == 4 else 1.0)),
                          bias=0.0)
        for j in range(10):
            y = TABLE_TOP + j * TABLE_STEP
            # the singular value: a bar and its number
            ra = sa * (smoothstep(0.08 + 0.015 * j, 0.1 + 0.015 * j, u) if k == 4 else 1.0)
            if ra > 0.01:
                length = S_LEN * s[j] / s[0]
                w = max(self.span(length)[0], 1.5)
                self.rect(sc, self.P(S_X + length / 2, y), w, th, SIM, ra, radius=1.5, lift=0.4)
                self.text(sc, S_X - 0.006, y, m["s_text"][j], ra, SIZE, "rm", ink=INK)
            # the noise along it
            ra = na * (smoothstep(0.34 + 0.012 * j, 0.36 + 0.012 * j, u) if k == 4 else 1.0)
            if ra > 0.01:
                length = N_LEN * noise[j]
                w = max(self.span(abs(length))[0], 1.5)
                self.rect(sc, self.P(N_X + length / 2, y), w, th, MEAS, ra, radius=1.5,
                          lift=0.4)
            # divided by s
            t0 = 0.52 + 0.025 * j
            ra = a * (smoothstep(t0, t0 + 0.02, u) if k == 4 else 1.0)
            if ra > 0.01:
                grow = hold(u, t0, t0 + 0.04) if k == 4 else 1.0
                length = B_LEN * blown[j] * grow
                w = max(self.span(abs(length))[0], 1.5)
                self.rect(sc, self.P(B_X + length / 2, y), w, th, INV, ra, radius=1.5,
                          lift=0.4)
                if j >= 7:
                    side = -1 if blown[j] > 0 else 1
                    self.text(sc, B_X + side * 0.006, y, m["blown_text"][j], ra * grow, SIZE,
                              "rm" if side < 0 else "lm", ink=INV_INK)

    def iterations_chart(self, sc, k, u, op):
        """Phase 6: the truth's outline and D'Agostini's iterates, 0 to 10."""
        a = op * self.after(k, 5, u, 0.07) * self.leaving(6, k, u)
        if a <= 0.01:
            return
        ch = self.chart(BIG_BOX, (0.0, COUNT_MAX))
        self.energy_axes(sc, ch, a, (0, 1000, 2000))
        self.zero_line(sc, ch, a)
        j = self.iteration(u) if k == 5 else 10
        ink = SIM if j == 0 else UNF
        self.bars(sc, ch, self.iterates[j], ink, a * (self.at(5, 0.1) if k == 5 else 1.0),
                  fill=0.55)
        self.outline(sc, ch, self.truth, TRUTH, a)

    def toys_chart(self, sc, k, u, op):
        """Phase 7: bias and spread against the number of iterations."""
        m = self.meta
        a = op * self.after(k, 6, u, 0.07)
        if a <= 0.01:
            return
        n = m["chart_iterations"]
        ch = self.chart(BIG_BOX, (0.0, 450.0), (0.0, float(n)))
        self.axes(sc, ch, a, xlabel="iterations", ylabel="events",
                  xticks=[(v, str(v)) for v in (0, 10, 20, 30)],
                  yticks=[(v, str(v)) for v in (0, 200, 400)], grid=False)
        upto = n * (hold(u, *self.CURVE_U) if k == 6 else 1.0)
        xs = np.arange(n + 1, dtype=float)
        for vals, ink in ((m["bias_rms"], BIAS), (m["spread_rms"], SPREAD)):
            ys = np.asarray(vals, float)[:n + 1]
            pts = [(ch.fx(x), ch.fy(y)) for x, y in zip(xs, ys, strict=True) if x <= upto]
            if upto > 0 and len(pts) < len(xs):
                pts.append((ch.fx(upto), ch.fy(float(np.interp(upto, xs, ys)))))
            if len(pts) > 1:
                self.polyline(sc, [self.P(x, y, self.ARROW_LIFT) for x, y in pts], 2.6, ink, a,
                              bias=0.5)
        la = a * (self.at(6, 0.3) if k == 6 else 1.0)
        self.text(sc, ch.fx(2.4), ch.fy(float(m["bias_rms"][2])) - 0.012, "bias", la, SIZE, "lm",
                  ink=BIAS_INK)
        self.text(sc, ch.fx(24.0), ch.fy(float(m["spread_rms"][24])) - 0.04, "spread", la,
                  SIZE, "mm", ink=SPREAD_INK)
        # the best number of iterations
        b = m["best"]
        ba = a * (self.at(6, 0.58) if k == 6 else 1.0)
        if ba > 0.01:
            x = ch.fx(float(b))
            for i in range(12):
                y0 = ch.y1 + (ch.y0 - ch.y1) * i / 12
                y1 = ch.y1 + (ch.y0 - ch.y1) * (i + 0.5) / 12
                self.polyline(sc, [self.P(x, y0, 0.5), self.P(x, y1, 0.5)], 1.2, AXIS,
                              0.8 * ba, bias=0.0)
            for vals, ink in ((m["bias_rms"], BIAS), (m["spread_rms"], SPREAD)):
                self.disk(sc, self.P(x, ch.fy(float(vals[b]))), 4.5, ink, ba, lift=1.6,
                          outline=0.15)

    # ------------------------------------------------------------------ 1: the truth
    def phase_1(self, sc, u, op):
        m = self.meta
        a = on(u, 0.2, op)
        self.text(sc, 0.3, 0.49, m["truth_total_text"] + " made-up events", a, SIZE, "lm",
                  ink=INK)
        b = on(u, 0.45, op)
        ch = self.chart(BIG_BOX, (0.0, COUNT_MAX))
        i = self.bump
        x = ch.fx((self.edges[i] + self.edges[i + 1]) / 2)
        y = ch.fy(self.truth[i])
        self.text(sc, 0.3, 0.55, "a bump at 60 GeV", b, SIZE, "lm", ink=INK)
        self.guide(sc, (0.315, 0.568), (x + 0.004, y - 0.008), AXIS, b)

    # ------------------------------------------------------------------ 2: R from simulation
    def phase_2(self, sc, u, op):
        m = self.meta
        x0, y0, x1, y1 = self.mat_box()
        n = len(m["sim_true"])
        shown = int(n * hold(u, 0.14, 0.4))
        da = op * on(u, 0.14, 1.0) * (1.0 - smoothstep(0.5, 0.6, u))
        if da > 0.01 and shown > 0:
            for tt, rr in zip(m["sim_true"][:shown], m["sim_measured"][:shown], strict=True):
                fx = x0 + (x1 - x0) * float(tt) / 100.0
                fy = y0 + (y1 - y0) * float(rr) / 100.0
                self.disk(sc, self.P(fx, fy), 1.7, SIM, da, lift=1.4, outline=0.0)
        X = RIGHT_X
        self.text(sc, X, 0.575, "simulated events", on(u, 0.1, op), SIZE, "lm", ink=QUIET)
        self.text(sc, X, 0.62, m["simulated_text"], on(u, 0.1, op), HEAD, "lm", ink=SIM_INK,
                  strong=False)
        self.text(sc, X, 0.685, "found, measured 0–100", on(u, 0.42, op), SIZE, "lm",
                  ink=QUIET)
        self.text(sc, X, 0.73, m["counted_text"], on(u, 0.42, op), HEAD, "lm", ink=SIM_INK)
        self.text(sc, X, 0.8, "each column ÷", on(u, 0.62, op), SIZE, "lm", ink=INK)
        self.text(sc, X, 0.835, "simulated in it", on(u, 0.62, op), SIZE, "lm", ink=INK)
        self.text(sc, X, 0.52, "%d dots shown" % n, during(u, 0.16, 0.56, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 3: measured
    def phase_3(self, sc, u, op):
        m = self.meta
        i = self.bump
        steps1 = ((0.0, "50–60 GeV: %s true" % m["truth_text"][i]),
                  (0.4, "all bins: %s true" % m["truth_total_text"]),
                  (0.64, "data: %s counts" % m["data_total_text"]))
        steps2 = ((0.0, "→ %s found, spread" % m["bump_found_text"]),
                  (0.4, "→ %s expected" % m["expected_total_text"]),
                  (0.64, "noise: about ±%s a bin" % m["data_noise_text"]))
        a = on(u, 0.1, op)
        self.running(sc, RIGHT_X, 0.41, steps1, u, a, size=SIZE, ink=INK)
        self.running(sc, RIGHT_X, 0.45, steps2, u, a, size=SIZE, ink=MEAS_INK)

    # ------------------------------------------------------------------ 4: inverted
    def phase_4(self, sc, u, op):
        m = self.meta
        a = on(u, 0.16, op)
        self.disk(sc, self.P(0.07, LINE1), 4.0, UNF, a, lift=1.6, outline=0.15)
        self.text(sc, 0.085, LINE1, "no noise: the truth, %s apart" % m["back_off_text"], a,
                  SIZE, "lm", ink=UNF_INK)
        b = on(u, 0.48, op)
        self.rect(sc, self.P(0.07, LINE2), 9.0, 11.0, INV, b, radius=1.5, lift=0.4,
                  fill_op=0.55)
        self.text(sc, 0.085, LINE2, "with noise: %s to %s" % (m["naive_min_text"],
                                                             m["naive_max_text"]), b, SIZE,
                  "lm", ink=INV_INK)

    # ------------------------------------------------------------------ 5: why
    def phase_5(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, LINE2, "the last 3 rows: %s of the swing" % m["last3_share_text"],
                  on(u, 0.82, op), SIZE, "lm", ink=INK)

    # ------------------------------------------------------------------ 6: iterations
    def phase_6(self, sc, u, op):
        j = self.iteration(u)
        steps = ((0.0, "start"),) + tuple((self.iter_u(n), "iteration %d" % n)
                                          for n in range(1, 11))
        a = on(u, 0.1, op)
        self.running(sc, 0.31, 0.49, steps, u, a, size=HEAD, ink=SIM_INK if j == 0 else UNF_INK)
        self.text(sc, 0.06, LINE1, "start: the simulation's shape, no bump", on(u, 0.12, op),
                  SIZE, "lm", ink=SIM_INK)
        self.text(sc, 0.06, LINE2, "each: fold with R, compare, correct", on(u, 0.3, op), SIZE,
                  "lm", ink=UNF_INK)

    # ------------------------------------------------------------------ 7: bias and spread
    def phase_7(self, sc, u, op):
        m = self.meta
        b = m["best"]
        a = on(u, 0.6, op)
        self.text(sc, 0.06, LINE1, "%d iterations: bias %s, spread %s" % (
            b, m["bias_text"][b], m["spread_text"][b]), a, SIZE, "lm", ink=INK)
        self.text(sc, 0.06, LINE2, "R inverted: spread %s" % m["inverse_spread_text"],
                  on(u, 0.75, op), SIZE, "lm", ink=INV_INK)
        self.text(sc, BIG_BOX[0][1], BIG_BOX[1][0] + 0.01, "2,000 pseudo-experiments",
                  on(u, 0.05, op), SIZE, "rm", ink=QUIET)


ANIMATIONS = {"unfolding": Unfolding}
