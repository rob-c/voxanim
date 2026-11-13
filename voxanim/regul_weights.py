"""L1 and L2 on a network's weights: the 784-64-10 MNIST MLP trained by SGD
with no penalty, L2, L1, elastic net, and L1 by its subgradient.

    784 -> 64 -> 10 (ReLU, He weights), SGD at learning rate 0.1, batches of
    64, 9,400 steps (10 epochs of all 60,000 digits); every run starts from
    the same weights and sees the same minibatches

1. **rule** -- one weight from a never-lit corner pixel: no gradient from
   the data ever reaches it, so only the penalty moves it.  Plain keeps it;
   L2 takes a fixed *fraction* of it each step (``w <- (1 - eta lam) w``,
   never 0); L1's proximal step takes a fixed *amount* and stops at 0.  The
   curve is the recurrence, checked against the run's recorded snapshots,
2. **templates** -- 36 of the 64 first-layer templates (each unit's 28 x 28
   weights) of the four runs side by side, playing through training; an
   exactly-zero weight is blank paper,
3. **never lit** -- how often each pixel is lit in training beside each
   run's per-pixel weight size: the 67 never-lit pixels (outlined) keep their
   random start in the plain network, shrink under L2, and are cut out (all
   64 outgoing weights exactly 0) by L1,
4. **histograms** -- all 50,176 first-layer weights on a log count: L1's
   spike at exactly 0 against L2's narrower bell,
5. **proximal** -- the same L1 penalty applied by its slope ``lam sign(w)``
   as an ordinary gradient term: the same accuracy and norm, but no weight
   is ever exactly 0; the small ones jitter within ``eta lam`` of it,
6. **table** -- test accuracy, the share exactly 0, the pixels cut and
   ``|W0|`` of all five.

Every number is read from :func:`voxanim.regul.weight_runs`.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .diagram import Diagram, draw_image
from .easing import remap, smoothstep
from .losschart import LossChart, pow10
from .netanim import ACT_HI, ACT_LO, gray_colors, lit_colors
from .regul import W_RUNS, W_TRAIN, cut_pixels, never_lit, weight_data, weight_runs
from .regul_plots import (
    EN_INK,
    L1_INK,
    L2_INK,
    NONE_INK,
    SLOPE_INK,
    ZERO,
    contour_segments,
    weight_mosaic,
)
from .text import num

INKS = (NONE_INK, L2_INK, L1_INK, EN_INK, SLOPE_INK)
SHOWN = (0, 1, 2, 3)                        # the four runs drawn side by side
NAMES = ("no penalty", "L2", "L1", "elastic net", "L1 by its slope")
STEPS = W_TRAIN["steps"]
ETA = W_TRAIN["lr"]
UNITS = 36                                  # templates shown per run (6 x 6)
TILE_COLS = 6
TEX_PX = 1.1                                # px per texel of a mosaic
COL_X = (0.13, 0.377, 0.623, 0.87)          # centres of the four columns (layout x)
MOSAIC_Y = 0.53
MAP_X = (0.10, 0.30, 0.50, 0.70, 0.90)      # the five pixel maps
MAP_Y, MAP_PX = 0.50, 5.0                   # px per pixel
HIST_BOX_Y = (0.31, 0.70)
HIST_X = (0.165, 0.39, 0.615, 0.84)         # centres of the four histogram panels
HIST_W = 0.19                               # layout width of a histogram panel
HIST_RANGE, HIST_BINS = 0.24, 48            # weights in (-0.24, 0.24), bins of 0.01
HIST_DECADES = 5.0                          # the log count axis: 1 .. 100,000
ZOOM_RANGE, ZOOM_BINS = 6e-5, 24            # the proximal phase's look at 0
ZOOM_BOX = (((0.07, 0.45), (0.32, 0.68)), ((0.55, 0.93), (0.32, 0.68)))
RULE_BOX = ((0.08, 0.50), (0.31, 0.74))
RULE_TEXT_X, RULE_TEXT_Y, RULE_ROW = 0.565, 0.31, 0.105
TABLE_Y0, TABLE_ROW = 0.33, 0.085
TABLE_COLS = (0.05, 0.46, 0.64, 0.80, 0.95)   # name | test accuracy | exactly 0 | cut | |W0|
PANEL = (0.955, 0.955, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
SPIKE_INK = (0.75, 0.22, 0.22)              # the bar of exact zeros
OUTLINE_INK = (0.85, 0.30, 0.25)            # the never-lit pixels' outline
DT = dict(rule=8.5, templates=10.5, maps=7.5, hist=6.5, prox=7.5, table=7.0)
HOLDS = dict(rule=1.0, templates=1.6, maps=1.6, hist=1.2, prox=1.2, table=0.0)


def lam_text(opts):
    """``"λ₁ = 2×10⁻⁴"`` and the like for a run's penalty options."""
    parts = []
    if opts.get("l1"):
        parts.append("λ₁ = " + pow10(opts["l1"]))
    if opts.get("weight_decay"):
        parts.append("λ₂ = " + pow10(opts["weight_decay"]))
    return ", ".join(parts) if parts else "λ = 0"


def penalty_path(w0, steps, l1=0.0, l2=0.0, eta=ETA):
    """``(steps + 1,)`` a weight that no data gradient reaches, under SGD at
    rate ``eta`` with the L2 term ``l2`` (a gradient step) and the L1 term
    ``l1`` (the proximal step after it): exactly what :func:`voxanim.deepnet.train`
    does to it."""
    out = np.empty(steps + 1)
    w = float(w0)
    out[0] = w
    for s in range(1, steps + 1):
        w = w - eta * l2 * w
        if l1:
            w = np.sign(w) * max(abs(w) - eta * l1, 0.0)
        out[s] = w
    return out


def window(t, a, b, fade=0.5):
    """1 inside ``[a, b]``, fading in over ``fade`` s before ``a`` ... and out after ``b``."""
    return smoothstep(a - fade, a, t) * (1.0 - smoothstep(b, b + fade, t))


class L1L2Weights(Diagram):
    """The rule, the templates, the never-lit pixels, the histograms, the
    proximal step and the table (see the module docstring)."""

    PHASES = ("rule", "templates", "never lit", "histograms", "proximal", "table")

    def __init__(self, runs=None, data=None):
        super().__init__("L1 and L2 on a network's weights", "")
        self.data = weight_data() if data is None else data
        self.runs = weight_runs(dat=self.data) if runs is None else runs
        self.hold = 2.5
        self.opts = [o for _n, o in W_RUNS]
        self.never = never_lit(self.data)
        self.lit_freq = (self.data.x_train > 0).mean(0)
        self.snap_steps = self.runs[0].steps
        self.S = len(self.snap_steps)
        self._measure()
        self._rule()
        self._textures()
        Tm, t = {}, 0.3
        for name in ("rule", "templates", "maps", "hist", "prox", "table"):
            Tm[name] = (t, t + DT[name])
            t += DT[name] + HOLDS[name]
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0] + [Tm[k][0] for k in ("templates", "maps", "hist", "prox", "table")]

    # ------------------------------------------------------------------ the numbers
    def _measure(self):
        R = self.runs
        self.W = [r.P["W0"] for r in R]
        self.acc = np.array([float(r.test_acc[-1]) for r in R])
        self.zero = np.array([float((w == 0).mean()) for w in self.W])
        self.cut = np.array([int(cut_pixels(w).sum()) for w in self.W])
        self.cut_never = np.array([int((cut_pixels(w) & self.never).sum()) for w in self.W])
        self.norm = np.array([float(np.linalg.norm(w)) for w in self.W])
        self.norm0 = float(np.linalg.norm(R[0].P0["W0"]))
        self.std = np.array([float(w.std()) for w in self.W])
        init_never = np.abs(R[0].P0["W0"][self.never]).sum()
        # share of the never-lit pixels' |w| left, per run and recorded step
        self.never_left = np.array([np.abs(r.weights["W0"][:, self.never]).sum((1, 2)) / init_never
                                    for r in R])
        self.snap_zero = np.array([(r.weights["W0"] == 0).mean((1, 2)) for r in R])
        self.snap_acc = np.array([r.test_acc for r in R])
        # the subgradient's small weights: within eta * lam of 0, none at 0
        lam = self.opts[4]["l1"]
        self.jitter = ETA * lam
        self.near = float((np.abs(self.W[4]) <= self.jitter).mean())
        self.n_weights = self.W[0].size
        self.zero_count = [int((w == 0).sum()) for w in self.W]
        self.near_count = int((np.abs(self.W[4]) <= self.jitter).sum())
        # histograms
        edges = np.linspace(-HIST_RANGE, HIST_RANGE, HIST_BINS + 1)
        self.hist = [np.histogram(w[w != 0], edges)[0] for w in self.W]
        zedges = np.linspace(-ZOOM_RANGE, ZOOM_RANGE, ZOOM_BINS + 1)
        self.zoom = [np.histogram(self.W[k], zedges)[0] for k in (2, 4)]

    def _rule(self):
        """The rule phase's weight: from a never-lit pixel, starting near 0.1."""
        P0 = self.runs[0].P0["W0"]
        cand = np.argwhere(self.never[:, None] & (P0 > 0.09) & (P0 < 0.11))
        self.rule_px, self.rule_unit = (int(v) for v in cand[0])
        w0 = float(P0[self.rule_px, self.rule_unit])
        self.rule_w0 = w0
        self.rule_paths = [penalty_path(w0, STEPS, o.get("l1", 0.0), o.get("weight_decay", 0.0))
                           for o in self.opts[:4]]
        for k in range(4):                    # the recurrence is what the network did
            rec = self.runs[k].weights["W0"][:, self.rule_px, self.rule_unit]
            assert np.allclose(self.rule_paths[k][self.snap_steps], rec, atol=1e-6)
        self.rule_zero_at = [int(np.argmax(p == 0)) if (p == 0).any() else None
                             for p in self.rule_paths]
        self.halving = float(np.log(2.0) / (ETA * self.opts[1]["weight_decay"]))
        self.rule_chart = LossChart(RULE_BOX, (0, STEPS), (0.0, 0.11))

    def _textures(self):
        """Every mosaic at every recorded step, one colour scale for all."""
        self.vmax = float(np.percentile(np.abs(np.concatenate([w.ravel() for w in self.W[:4]])),
                                        99.5))
        units = range(UNITS)
        self.tex = [[weight_mosaic(self.runs[k].weights["W0"][s], TILE_COLS, self.vmax,
                                   units=units) for s in range(self.S)] for k in SHOWN]
        # per-pixel weight size |W[p, :]| at every recorded step
        self.pix = [np.sqrt((self.runs[k].weights["W0"].astype(np.float64) ** 2).sum(2))
                    for k in SHOWN]
        self.pix_max = float(np.percentile(self.pix[0][-1], 99))
        self.pix_cut = [(self.runs[k].weights["W0"] != 0).sum(2) == 0 for k in SHOWN]
        xs = np.arange(28.0)
        self.outline = contour_segments(self.never.reshape(28, 28).astype(float), xs, xs, 0.5)

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def snap_x(self, name, t):
        """Fractional recorded-step index a playback phase has reached."""
        return remap(t, *self.Tm[name]) * (self.S - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        T = self.Tm
        nxt = {"rule": "templates", "templates": "maps", "maps": "hist", "hist": "prox",
               "prox": "table"}

        def op(name):
            end = T[nxt[name]][0] - 0.6 if name in nxt else self.t_done + 99
            start = T[name][0] + 0.5 if name != "rule" else 0.5     # after the last has gone
            return window(t, start, end) * back

        self._rule_phase(sc, t, op("rule"))
        self._templates(sc, t, op("templates"))
        self._maps(sc, t, op("maps"))
        self._histograms(sc, t, op("hist"))
        self._proximal(sc, t, op("prox"))
        self._table(sc, t, op("table") * smoothstep(0.0, 0.5, t - T["table"][0] + 0.5))
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, ph, reset)

    # -- 1. the rule ---------------------------------------------------------------
    def _rule_phase(self, sc, t, op):
        if op <= 0.01:
            return
        u = remap(t, self.Tm["rule"][0] + 0.8, self.Tm["rule"][1] - 1.0)
        upto = STEPS * u
        ch = self.rule_chart
        ch.axes(sc, self.cam, self.P, op, "training step", "the weight's value",
                [(0, "0"), (3000, "3,000"), (6000, "6,000"), (9000, "9,000")],
                [(0.0, "0"), (0.05, "0.05"), (0.1, "0.1")])
        x = np.arange(0, STEPS + 1, 50)
        for k in (0, 1, 3, 2):
            ch.curve(sc, self.P, x, self.rule_paths[k][x], upto, INKS[k], op, width=1.8)
        rows = self._rule_rows()
        for n, (head, note, ink) in enumerate(rows):
            o = op * smoothstep(0.0, 0.15, u - 0.18 * n + 0.1)
            y = RULE_TEXT_Y + n * RULE_ROW
            self.label(sc, self.P(RULE_TEXT_X, y), head, layout.MIN_TEXT, anchor="lm", ink=ink,
                       opacity=o)
            self.label(sc, self.P(RULE_TEXT_X, y + 0.045), note, layout.MIN_TEXT, anchor="lm",
                       ink=layout.NOTE_INK, opacity=o)

    def _rule_rows(self):
        p = self.rule_paths
        z1, z3 = self.rule_zero_at[2], self.rule_zero_at[3]
        return [
            ("no penalty: stays at %s" % num(self.rule_w0, 3), "its random start, forever",
             INKS[0]),
            ("L2: w ← (1 − ηλ₂) w, a fixed fraction",
             "halves every %s steps; %s at the end, never 0"
             % (f"{round(self.halving):,}", num(p[1][-1], 3)), INKS[1]),
            ("L1: w moves ηλ₁ = %s towards 0" % pow10(ETA * self.opts[2]["l1"]),
             "a fixed amount, and stops: 0 from step %s" % f"{z1:,}", INKS[2]),
            ("elastic net: both at once", "%s: 0 from step %s"
             % (lam_text(self.opts[3]), f"{z3:,}"), INKS[3]),
        ]

    # -- 2. templates ----------------------------------------------------------------
    def _snap_tex(self, k, x):
        i = int(np.floor(x))
        a = x - i
        T = self.tex[k]
        if a <= 1e-6 or i + 1 >= self.S:
            return T[min(i, self.S - 1)]
        return T[i] * (1 - a) + T[i + 1] * a

    def _templates(self, sc, t, op):
        if op <= 0.01:
            return
        x = self.snap_x("templates", t)
        i = int(round(x))
        h = self.tex[0][0].shape[0] * TEX_PX
        for col, k in enumerate(SHOWN):
            c = self.P(COL_X[col], MOSAIC_Y)
            draw_image(self, sc, c, self._snap_tex(col, x), h, h, op, grid=0.0)
            top = self.shift(c, 0, -h / 2)
            bottom = self.shift(c, 0, h / 2)
            self.label(sc, top, NAMES[k], layout.MIN_TEXT, anchor="mm", dy=-34, ink=INKS[k],
                       opacity=op)
            self.label(sc, top, lam_text(self.opts[k]), layout.MIN_TEXT, anchor="mm", dy=-14,
                       ink=layout.NOTE_INK, opacity=op)
            self.label(sc, bottom, "%.0f%% exactly 0" % (100 * self.snap_zero[k, i]),
                       layout.MIN_TEXT, anchor="mt", dy=8, ink=layout.CAPTION_INK, opacity=op)
            self.label(sc, bottom, "test %.1f%%" % (100 * self.snap_acc[k, i]), layout.MIN_TEXT,
                       anchor="mt", dy=32, ink=layout.NOTE_INK, opacity=op)
        sc.probe(("templates",), (x * 10, 0, 0), tol=6.0)

    # -- 3. never-lit pixels ------------------------------------------------------------
    def _map(self, sc, c, colors, op):
        tex = np.asarray(colors, np.float32).reshape(28, 28, 3)
        draw_image(self, sc, c, tex, 28 * MAP_PX, 28 * MAP_PX, op, grid=0.0)
        # the never-lit pixels' outline
        o = self.shift(c, -14 * MAP_PX, -14 * MAP_PX)
        for (x0, y0), (x1, y1) in self.outline:
            a = self.shift(o, (x0 + 0.5) * MAP_PX, (y0 + 0.5) * MAP_PX)
            b = self.shift(o, (x1 + 0.5) * MAP_PX, (y1 + 0.5) * MAP_PX)
            sc.line(a + self.view * 0.6, b + self.view * 0.6, width=1.2, color=OUTLINE_INK,
                    opacity=0.9 * op, bias=0.0)

    def _maps(self, sc, t, op):
        if op <= 0.01:
            return
        x = self.snap_x("maps", t)
        i = int(np.floor(x))
        a = x - i
        j = min(i + 1, self.S - 1)
        side = 28 * MAP_PX
        c = self.P(MAP_X[0], MAP_Y)
        self._map(sc, c, gray_colors(self.lit_freq / self.lit_freq.max()), op)
        top, bottom = self.shift(c, 0, -side / 2), self.shift(c, 0, side / 2)
        self.label(sc, top, "pixel lit in training", layout.MIN_TEXT, anchor="mm", dy=-15,
                   ink=layout.CAPTION_INK, opacity=op)
        self.label(sc, bottom, "%d never lit" % int(self.never.sum()), layout.MIN_TEXT,
                   anchor="mt", dy=8, ink=OUTLINE_INK, opacity=op)
        self.label(sc, bottom, "(outlined)", layout.MIN_TEXT, anchor="mt", dy=32,
                   ink=layout.NOTE_INK, opacity=op)
        k_snap = int(round(x))
        for col, k in enumerate(SHOWN):
            v = self.pix[col][i] * (1 - a) + self.pix[col][j] * a
            colors = lit_colors(v, self.pix_max, ACT_LO, ACT_HI)
            colors[self.pix_cut[col][k_snap]] = ZERO
            c = self.P(MAP_X[col + 1], MAP_Y)
            self._map(sc, c, colors, op)
            top, bottom = self.shift(c, 0, -side / 2), self.shift(c, 0, side / 2)
            self.label(sc, top, NAMES[k], layout.MIN_TEXT, anchor="mm", dy=-15, ink=INKS[k],
                       opacity=op)
            self.label(sc, bottom, "never-lit |w|: %.0f%%" % (100 * self.never_left[k, k_snap]),
                       layout.MIN_TEXT, anchor="mt", dy=8, ink=layout.CAPTION_INK, opacity=op)
            cut = int(self.pix_cut[col][k_snap].sum())
            self.label(sc, bottom, "pixels cut: %d" % cut, layout.MIN_TEXT, anchor="mt", dy=32,
                       ink=layout.STRONG_INK if cut else layout.NOTE_INK, opacity=op)
        sc.probe(("maps",), (x * 10, 0, 0), tol=6.0)

    # -- 4. histograms -------------------------------------------------------------------
    def _panel(self, sc, box, op):
        (x0, x1), (y0, y1) = box
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)

    def _bars(self, sc, box, counts, height, ink, op, lo, hi, u=1.0):
        """Bars of ``height(counts)`` (0..1 of the panel) over ``(lo, hi)``."""
        (x0, x1), (_y0, y1) = box
        n = len(counts)
        for b in range(n):
            hh = min(1.0, height(counts[b])) * u
            if hh <= 0:
                continue
            fx0, fx1 = x0 + (x1 - x0) * b / n, x0 + (x1 - x0) * (b + 1) / n
            p0, p1 = self.P(fx0, y1 - (y1 - box[1][0]) * hh), self.P(fx1, y1)
            w, h = self.px(p1) - self.px(p0)
            self.rect(sc, (p0 + p1) / 2, max(w - 1.0, 1.0), h, ink, 0.85 * op, radius=0.0,
                      lift=0.15)
        del lo, hi

    def _log_h(self, c):
        return np.log10(c + 1.0) / HIST_DECADES if c > 0 else 0.0

    def _histograms(self, sc, t, op):
        if op <= 0.01:
            return
        u = smoothstep(0.0, 1.0, remap(t, self.Tm["hist"][0], self.Tm["hist"][0] + 2.0))
        spike = smoothstep(0.0, 1.0, remap(t, self.Tm["hist"][0] + 2.0, self.Tm["hist"][0] + 3.2))
        for col, k in enumerate(SHOWN):
            cx = HIST_X[col]
            box = ((cx - HIST_W / 2, cx + HIST_W / 2), HIST_BOX_Y)
            self._panel(sc, box, op)
            self._bars(sc, box, self.hist[k], self._log_h, INKS[k], op, -HIST_RANGE, HIST_RANGE,
                       u)
            (x0, x1), (y0, y1) = box
            # the exact zeros: their own bar at 0
            if self.zero_count[k] and spike > 0:
                hh = self._log_h(self.zero_count[k]) * spike
                p0, p1 = self.P(cx, y1 - (y1 - y0) * hh), self.P(cx, y1)
                h = (self.px(p1) - self.px(p0))[1]
                self.rect(sc, (p0 + p1) / 2, 3.0, h, SPIKE_INK, op, radius=0.0, lift=0.3)
                self.label(sc, p0, "= 0", layout.MIN_TEXT, anchor="lm", dx=6, dy=2,
                           ink=SPIKE_INK, opacity=op * spike)
            # axes labels
            for v, s in ((-0.2, "−0.2"), (0.0, "0"), (0.2, "0.2")):
                fx = x0 + (x1 - x0) * (v + HIST_RANGE) / (2 * HIST_RANGE)
                self.label(sc, self.P(fx, y1), s, layout.MIN_TEXT, anchor="mt", dy=4,
                           ink=layout.NOTE_INK, opacity=op)
            self.label(sc, self.P(cx, y0), NAMES[k], layout.MIN_TEXT, anchor="mm", dy=-15,
                       ink=INKS[k], opacity=op)
            self.label(sc, self.P(cx, y1), "std %s" % num(self.std[k], 3), layout.MIN_TEXT,
                       anchor="mt", dy=26, ink=layout.CAPTION_INK, opacity=op)
            self.label(sc, self.P(cx, y1), "exactly 0: %.0f%%" % (100 * self.zero[k]),
                       layout.MIN_TEXT, anchor="mt", dy=50,
                       ink=SPIKE_INK if self.zero[k] else layout.NOTE_INK, opacity=op)
        box0 = (HIST_X[0] - HIST_W / 2, HIST_BOX_Y)
        for d, s in ((1, "10"), (3, "10³"), (5, "10⁵")):
            fy = HIST_BOX_Y[1] - (HIST_BOX_Y[1] - HIST_BOX_Y[0]) * d / HIST_DECADES
            self.label(sc, self.P(box0[0], fy), s, layout.MIN_TEXT, anchor="rm", dx=-5,
                       ink=layout.NOTE_INK, opacity=op)

    # -- 5. the proximal step against the slope -------------------------------------------
    def _proximal(self, sc, t, op):
        if op <= 0.01:
            return
        u = smoothstep(0.0, 1.0, remap(t, self.Tm["prox"][0], self.Tm["prox"][0] + 2.0))
        top = float(max(c.max() for c in self.zoom))
        for n, k in enumerate((2, 4)):
            box = ZOOM_BOX[n]
            (x0, x1), (y0, y1) = box
            self._panel(sc, box, op)
            self._bars(sc, box, self.zoom[n], lambda c: c / top, INKS[k], op, -ZOOM_RANGE,
                       ZOOM_RANGE, u)
            # dashed marks at +- eta * lam
            for v in (-self.jitter, self.jitter):
                fx = x0 + (x1 - x0) * (v + ZOOM_RANGE) / (2 * ZOOM_RANGE)
                a, b = self.P(fx, y1, 0.4), self.P(fx, y0, 0.4)
                for d in range(10):
                    sc.line(a + (b - a) * d / 10, a + (b - a) * (d + 0.5) / 10, width=0.9,
                            color=layout.NOTE_INK, opacity=0.7 * op, bias=0.0)
            for v, s in ((-self.jitter, "−ηλ"), (0.0, "0"), (self.jitter, "+ηλ")):
                fx = x0 + (x1 - x0) * (v + ZOOM_RANGE) / (2 * ZOOM_RANGE)
                self.label(sc, self.P(fx, y1), s, layout.MIN_TEXT, anchor="mt", dy=4,
                           ink=layout.NOTE_INK, opacity=op)
            head = ("L1, the proximal step" if k == 2 else "L1 by its slope, λ sign(w)")
            self.label(sc, self.P((x0 + x1) / 2, y0), head, layout.MIN_TEXT, anchor="mm", dy=-15,
                       ink=INKS[k], opacity=op)
            lines = ((("%s weights exactly 0" % f"{self.zero_count[2]:,}"),
                      "test %.1f%%, |W₀| %.1f" % (100 * self.acc[2], self.norm[2]))
                     if k == 2 else
                     ("none exactly 0; %s within ±ηλ" % f"{self.near_count:,}",
                      "test %.1f%%, |W₀| %.1f" % (100 * self.acc[4], self.norm[4])))
            for m, s in enumerate(lines):
                self.label(sc, self.P((x0 + x1) / 2, y1), s, layout.MIN_TEXT, anchor="mt",
                           dy=26 + 24 * m, ink=layout.CAPTION_INK if m == 0 else layout.NOTE_INK,
                           opacity=op)

    # -- 6. the table ----------------------------------------------------------------------
    def _table(self, sc, t, op):
        if op <= 0.01:
            return
        heads = ("", "test accuracy", "exactly 0", "pixels cut", "|W₀|")
        anchors = ("lm", "rm", "rm", "rm", "rm")
        for x, s, anc in zip(TABLE_COLS, heads, anchors):
            self.label(sc, self.P(x, TABLE_Y0), s, layout.MIN_TEXT, anchor=anc,
                       ink=layout.NOTE_INK, opacity=op)
        for k in range(len(self.runs)):
            y = TABLE_Y0 + (k + 1) * TABLE_ROW
            o = op * smoothstep(0.0, 0.4, remap(t, self.Tm["table"][0] + 0.3 * k,
                                                self.Tm["table"][0] + 0.3 * k + 0.6))
            cells = ((NAMES[k] + ("" if k == 0 else ",  " + lam_text(self.opts[k])), INKS[k]),
                     ("%.1f%%" % (100 * self.acc[k]), layout.CAPTION_INK),
                     ("%.0f%%" % (100 * self.zero[k]),
                      SPIKE_INK if self.zero[k] else layout.CAPTION_INK),
                     ("%d" % self.cut[k], layout.STRONG_INK if self.cut[k] else layout.CAPTION_INK),
                     ("%.1f" % self.norm[k], layout.CAPTION_INK))
            for x, (s, ink), anc in zip(TABLE_COLS, cells, anchors):
                self.label(sc, self.P(x, y), s, 0.03, anchor=anc, ink=ink, opacity=o)

    # ------------------------------------------------------------------ words
    def _words(self, t, ph, reset):
        if reset > 0 or ph == 0:
            return ("one weight from a never-lit corner pixel to hidden unit %d: no gradient from "
                    "the data ever reaches it, so only the penalty moves it (SGD, η = %g)"
                    % (self.rule_unit, ETA))
        if ph == 1:
            x = self.snap_x("templates", t)
            i = int(round(x))
            return ("36 of the first layer's 64 templates, step %s of %s: L1 blanks the "
                    "background (%.0f%% of weights exactly 0), L2 shrinks every weight and "
                    "zeroes none" % (f"{int(self.snap_steps[i]):,}", f"{STEPS:,}",
                                     100 * self.snap_zero[2, i]))
        if ph == 2:
            return ("%d pixels are never lit in %s training digits: the plain network keeps "
                    "their random weights forever, L2 leaves %.0f%% of them, L1 cuts %d pixels "
                    "out entirely (%d of the %d)"
                    % (int(self.never.sum()), f"{len(self.data.x_train):,}",
                       100 * self.never_left[1, -1], self.cut[2], self.cut_never[2],
                       int(self.never.sum())))
        if ph == 3:
            return ("all %s first-layer weights (log count): L1 puts %.0f%% at exactly 0, "
                    "elastic net %.0f%%; L2 narrows the bell (std %s → %s) and zeroes none"
                    % (f"{self.n_weights:,}", 100 * self.zero[2], 100 * self.zero[3],
                       num(self.std[0], 3), num(self.std[1], 3)))
        if ph == 4:
            return ("the same L1 penalty as an ordinary slope term: the same accuracy and norm, "
                    "but no weight lands on 0 — %.0f%% hop back and forth within ηλ = %s of it"
                    % (100 * self.near, pow10(self.jitter)))
        lo, hi = 100 * self.acc[1:].min(), 100 * self.acc[1:].max()
        return ("the cost: %.1f%% test accuracy plain, %.1f–%.1f%% penalised (%s test digits); "
                "L1 buys a sparse first layer: %d of 784 pixels never read"
                % (100 * self.acc[0], lo, hi, f"{len(self.data.x_test):,}", self.cut[2]))
