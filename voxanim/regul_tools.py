"""Four more regularisers, one step each with real numbers, and what each one
changes in the trained network.

The network is the race's (:mod:`voxanim.regul`): 784 -> 256 -> 256 -> 10
(ReLU, He weights), trained by Adam for 3,000 steps on 1,000 MNIST digits,
once plain and once with each regulariser at the strength the validation
set chose (:func:`voxanim.regul.race_runs`).

1. **dropout** -- one training digit's first hidden layer (256 units as a
   16 x 16 sheet) under three fresh masks, as three training steps see it:
   each unit dropped with probability p, the kept ones scaled by 1 / (1 - p).
   What it changes: drop half the hidden units of the *trained* networks at
   test time and the plain one loses far more accuracy than the one trained
   with dropout,
2. **input noise** -- the same digit plus fresh Gaussian noise of std sigma
   at four steps; what it changes: accuracy on test digits with that noise,
3. **max-norm** -- the length of each hidden unit's incoming weight vector
   (:func:`voxanim.deepnet.unit_norms`) after plain training and after
   training under the cap c, both layers, as histograms with the cap and the
   He initialisation's lengths marked: how much the cap binds, and that the
   accuracy does not move,
4. **label smoothing** -- the one-hot target against the smoothed one
   (1 - eps + eps / K on the label, eps / K elsewhere); what it changes: the
   gap between the top logit and the others on 2,000 test digits (the
   softmax meets the smoothed target at
   :func:`voxanim.regul.smoothing_gap`), the median confidence and the
   confidence of the wrong answers,
5. **summary** -- the four in a table with their strength and test
   accuracy against the plain network's, ties called ties (:data:`TIE`).

Every number is computed here from the cached runs; nothing is typed in.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import regul as R
from .deepnet import Spec, forward, unit_norms
from .diagram import Diagram, draw_image
from .easing import remap, smoothstep
from .gen_common import draw_digit, pct
from .netanim import OFF, lit_colors
from .regul_plots import NONE_INK

SEED = 0
DIGIT = 7                       # the training digit the dropout and noise steps use
MASKS = 3                       # dropout steps shown
DRAWS = 4                       # input-noise steps shown
TIE = 0.007                     # accuracies this close to the plain network's are a tie
SHEET = 16                      # 256 hidden units as a 16 x 16 sheet
CELL = 7.0                      # px per unit
DIGIT_PX = 112.0
HEAD_INK = layout.NOTE_INK
INIT_INK = (0.80, 0.81, 0.83)    # the He initialisation's lengths, behind the plain ones
KEPT_INK = np.array([0.93, 0.93, 0.90], np.float32)
BAR_H, BAR_GAP = 12.0, 4.0      # px: the comparison bars
NORM_RANGE = (1.2, 3.2)         # the max-norm histograms' x axis
NORM_BIN = 0.05
NORM_FULL = 48                  # a bin with this many units fills the row
GAP_RANGE = (0.0, 56.0)         # the logit-gap histograms' x axis
GAP_BIN = 1.0
PHASE_T = dict(dropout=11.0, noise=9.5, maxnorm=10.5, smooth=12.0, summary=8.0)


def _ink(key):
    return R.BY_KEY[key].ink


def dashed(anim, sc, a, b, ink, op, dashes=14, width=1.1):
    """A dashed line from world point ``a`` to ``b``."""
    for i in range(dashes):
        u0, u1 = i / dashes, (i + 0.55) / dashes
        anim.polyline(sc, [a + (b - a) * u0, a + (b - a) * u1], width, ink, op, bias=0.2)


def hist(values, lo, hi, width):
    """``(edges, counts)`` of ``values`` in bins of ``width`` over ``[lo, hi]``
    (values beyond the range go in the end bins)."""
    edges = np.arange(lo, hi + width * 0.5, width)
    v = np.clip(values, lo, hi - 1e-9)
    return edges, np.histogram(v, edges)[0]


class Regularisers(Diagram):
    """Dropout, input noise, max-norm and label smoothing (see the module)."""

    PHASES = ("dropout", "input noise", "max-norm", "label smoothing", "summary")

    def __init__(self, dat=None, plain=None, chosen=None):
        super().__init__("Four more regularisers", "")
        self.dat = R.race_data() if dat is None else dat
        if plain is None:
            plain, chosen = R.race_runs(dat=self.dat)
        self.plain = plain
        self.reg = {reg.key: (reg, s, r) for reg, s, r in chosen}
        self.hold = 2.5
        self._measure()
        Tm, t = {}, 0.3
        for k in ("dropout", "noise", "maxnorm", "smooth", "summary"):
            Tm[k] = (t, t + PHASE_T[k])
            t += PHASE_T[k] + (0.0 if k == "summary" else 0.6)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0] + [Tm[k][0] for k in ("noise", "maxnorm", "smooth", "summary")]

    # ------------------------------------------------------------------ the numbers
    def _measure(self):
        d, plain = self.dat, self.plain
        x, y = d.x_test, d.y_test
        self.acc_plain = float(plain.test_acc[-1])
        self.digit = d.x_train[DIGIT]
        self.digit_label = int(d.y_train[DIGIT])

        # dropout: three masks on the dropout network's first hidden layer
        _r, p, run = self.reg["dropout"]
        self.p = p
        self.h = forward(run.spec, run.P, self.digit[None])["a"][0][0]
        rng = np.random.default_rng([SEED, 1])
        self.masks = rng.random((MASKS, len(self.h))) >= p
        self.h_vmax = float(self.h.max()) / (1.0 - p)

        def dropped_acc(r):
            accs = []
            for s in range(5):
                g = np.random.default_rng([SEED, 2, s])
                masks = [g.random((len(x), w)) >= p for w in R.RACE_SIZES[1:-1]]
                sp = Spec(r.spec.sizes, "relu", "he", dropout=p)
                f = forward(sp, r.P, x, train=True, masks=masks)
                accs.append(float((f["p"].argmax(1) == y).mean()))
            return float(np.mean(accs))
        self.drop_acc = (dropped_acc(plain), dropped_acc(run))
        self.dropout_acc = float(run.test_acc[-1])

        # input noise: fresh draws on the digit; accuracy on noisy test digits
        _r, sd, run = self.reg["noise"]
        self.sd = sd
        g = np.random.default_rng([SEED, 3])
        self.draws = self.digit[None] + g.normal(0.0, sd, (DRAWS, len(self.digit)))
        xn = x + np.random.default_rng([SEED, 4]).normal(0.0, sd, x.shape)
        self.noisy_acc = tuple(float((forward(r.spec, r.P, xn)["p"].argmax(1) == y).mean())
                               for r in (plain, run))
        self.noise_acc = float(run.test_acc[-1])

        # max-norm: the units' incoming weight lengths
        _r, c, run = self.reg["maxnorm"]
        self.cap = c
        self.norms = {}
        for l in (0, 1):
            k = "W%d" % l
            self.norms[l] = dict(init=unit_norms(plain.P0[k]), plain=unit_norms(plain.P[k]),
                                 capped=unit_norms(run.P[k]))
        self.maxnorm_acc = float(run.test_acc[-1])

        # label smoothing: the target and the logit gaps
        _r, eps, run = self.reg["smooth"]
        self.eps = eps
        self.gap_line = R.smoothing_gap(eps)
        self.gaps = (R.logit_gap(plain, x), R.logit_gap(run, x))
        conf = (R.confidence(plain, x), R.confidence(run, x))
        wrong = [forward(r.spec, r.P, x)["p"].argmax(1) != y for r in (plain, run)]
        self.conf_med = tuple(float(np.median(cf)) for cf in conf)
        self.conf_wrong = tuple(float(np.median(cf[w])) for cf, w in zip(conf, wrong))
        self.sure = tuple(float((cf > 0.99).mean()) for cf in conf)
        self.smooth_acc = float(run.test_acc[-1])

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_op(self, key, t, back):
        """A phase's opacity: in over its first 0.6 s, out over its last 0.5 s
        (the summary stays to the end and fades with the reset)."""
        a, b = self.Tm[key]
        op = smoothstep(a, a + 0.6, t)
        if key != "summary":
            op *= 1.0 - smoothstep(b - 0.5, b, t)
        return op * back

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """The frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        self._dropout(sc, t, self._phase_op("dropout", t, back))
        self._noise(sc, t, self._phase_op("noise", t, back))
        self._maxnorm(sc, t, self._phase_op("maxnorm", t, back))
        self._smooth(sc, t, self._phase_op("smooth", t, back))
        self._summary(sc, t, self._phase_op("summary", t, back))
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, reset)

    # -- shared pieces -----------------------------------------------------------------
    def _sheet(self, sc, fx, fy, colors, op):
        tex = np.asarray(colors, np.float32).reshape(SHEET, SHEET, 3)
        draw_image(self, sc, self.P(fx, fy), tex, SHEET * CELL, SHEET * CELL, op, grid=0.35)

    def _under(self, sc, fx, fy, text, op, dy=66.0, ink=None):
        self.label(sc, self.P(fx, fy), text, layout.MIN_TEXT, anchor="mt", dy=dy,
                   ink=ink, opacity=op)

    def _compare(self, sc, fy, rows, names, op, grow=1.0, x_label=0.06, x_bar=0.47,
                 width=0.36, fmt=pct):
        """A heading, a legend and rows of paired horizontal bars from layout
        ``fy`` down: ``rows`` ``[(label, (plain, regularised)), ...]`` on a
        0..1 scale; ``names`` ``((text, ink), (text, ink))``."""
        if op <= 0.01:
            return
        self.label(sc, self.P(x_label, fy), "in the trained network:", layout.MIN_TEXT,
                   anchor="lm", ink=HEAD_INK, opacity=op)
        x = x_bar
        for text, ink in names:
            self.label(sc, self.P(x, fy), text, layout.MIN_TEXT, anchor="lm", ink=ink,
                       opacity=op)
            x += layout.text_width(text, layout.MIN_TEXT) / layout.ZOOM + 0.03
        for i, (text, vals) in enumerate(rows):
            ry = fy + 0.085 * (i + 1)
            self.label(sc, self.P(x_label, ry), text, layout.MIN_TEXT, anchor="lm",
                       ink=layout.CAPTION_INK, opacity=op)
            for j, ((_n, ink), v) in enumerate(zip(names, vals)):
                a = self.shift(self.P(x_bar, ry), 0.0, (j - 0.5) * (BAR_H + BAR_GAP))
                b = self.shift(self.P(x_bar + width * v * grow, ry), 0.0,
                               (j - 0.5) * (BAR_H + BAR_GAP))
                w = self.px(b)[0] - self.px(a)[0]
                if w > 1:
                    self.rect(sc, (a + b) / 2, w, BAR_H, ink, 0.85 * op, radius=2.0, lift=0.2)
                self.label(sc, b, fmt(v), layout.MIN_TEXT, anchor="lm", dx=6, ink=ink,
                           opacity=op * smoothstep(0.85, 1.0, grow))

    # -- 1. dropout -------------------------------------------------------------------
    def _mask_index(self, t):
        u = self.progress("dropout", t)
        return int(np.clip((u - 0.12) / 0.5 * MASKS, 0, MASKS - 1))

    def _dropout(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("dropout", t)
        m = self.masks[self._mask_index(t)]
        keep = 1.0 - self.p
        fy = 0.44
        draw_digit(self, sc, self.P(0.10, fy), self.digit, DIGIT_PX, op)
        self._under(sc, 0.10, fy, "a training digit", op)
        self._sheet(sc, 0.32, fy, lit_colors(self.h, self.h_vmax), op)
        self._under(sc, 0.32, fy, "layer 1: 256 units", op)
        mo = op * smoothstep(0.05, 0.12, u)
        self._sheet(sc, 0.54, fy, np.where(m[:, None], KEPT_INK[None], OFF[None]), mo)
        self._under(sc, 0.54, fy, "mask, step %d of %d" % (self._mask_index(t) + 1, MASKS), mo)
        oo = op * smoothstep(0.08, 0.15, u)
        self._sheet(sc, 0.76, fy, lit_colors(self.h * m / keep, self.h_vmax), oo)
        self._under(sc, 0.76, fy, "× mask / %s" % _num(keep), oo)
        for fx, s in ((0.21, "→"), (0.43, "×"), (0.65, "=")):
            self.label(sc, self.P(fx, fy), s, 0.04, ink=HEAD_INK, opacity=mo if s != "→" else op)
        grow = smoothstep(0.0, 1.0, remap(u, 0.66, 0.9))
        names = (("plain", NONE_INK), ("trained with dropout", _ink("dropout")))
        rows = (("test accuracy", (self.acc_plain, self.dropout_acc)),
                ("with half its units dropped", self.drop_acc))
        self._compare(sc, 0.70, rows, names, op * smoothstep(0.62, 0.68, u), grow)

    # -- 2. input noise ---------------------------------------------------------------
    def _noise(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("noise", t)
        fy, px = 0.44, 96.0
        draw_digit(self, sc, self.P(0.10, fy), self.digit, px, op)
        self._under(sc, 0.10, fy, "the digit", op, dy=58)
        for k in range(DRAWS):
            o = op * smoothstep(0.08 + 0.1 * k, 0.14 + 0.1 * k, u)
            fx = 0.30 + 0.17 * k
            draw_digit(self, sc, self.P(fx, fy), self.draws[k], px, o)
            self._under(sc, fx, fy, "step %d" % (k + 1), o, dy=58)
        self.label(sc, self.P(0.20, fy), "+", 0.04, ink=HEAD_INK, opacity=op)
        grow = smoothstep(0.0, 1.0, remap(u, 0.62, 0.88))
        names = (("plain", NONE_INK), ("trained with noise", _ink("noise")))
        rows = (("test accuracy", (self.acc_plain, self.noise_acc)),
                ("on noisy test digits, σ = %s" % _num(self.sd), self.noisy_acc))
        self._compare(sc, 0.70, rows, names, op * smoothstep(0.58, 0.64, u), grow)

    # -- 3. max-norm --------------------------------------------------------------------
    def _norm_x(self, box, v):
        (x0, x1), _ = box
        lo, hi = NORM_RANGE
        return x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)

    def _hist_row(self, sc, box, base, height, values, ink, op, grow, lo_hi, bin_w, full,
                  x_of):
        edges, counts = hist(values, *lo_hi, bin_w)
        for b, n in enumerate(counts):
            if n == 0:
                continue
            h = min(1.0, n / full) * grow
            fx0, fx1 = x_of(box, edges[b]), x_of(box, edges[b + 1])
            p0, p1 = self.P(fx0, base - height * h), self.P(fx1, base)
            w, hh = self.px(p1) - self.px(p0)
            if hh > 0.5:
                self.rect(sc, (p0 + p1) / 2, max(w - 1.0, 1.0), hh, ink, 0.9 * op, radius=0.0,
                          lift=0.15)
        return edges, counts

    def _axis(self, sc, box, base, op, ticks, x_of):
        (x0, x1), _ = box
        self.polyline(sc, [self.P(x0, base, 0.2), self.P(x1, base, 0.2)], 1.0,
                      (0.5, 0.52, 0.54), 0.7 * op)
        for v in ticks:
            fx = x_of(box, v)
            self.polyline(sc, [self.P(fx, base, 0.2), self.P(fx, base + 0.012, 0.2)], 1.0,
                          (0.5, 0.52, 0.54), 0.7 * op)
            self.label(sc, self.P(fx, base + 0.014), _num(v), layout.MIN_TEXT, anchor="mt",
                       ink=HEAD_INK, opacity=op)

    def _maxnorm(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("maxnorm", t)
        g1 = smoothstep(0.0, 1.0, remap(u, 0.05, 0.3))
        cap_op = op * smoothstep(0.32, 0.4, u)
        g2 = smoothstep(0.0, 1.0, remap(u, 0.42, 0.65))
        ink = _ink("maxnorm")
        rows = ((0.53, "plain"), (0.83, "max-norm"))
        for l, box in ((0, ((0.10, 0.46), (0.3, 0.9))), (1, ((0.56, 0.92), (0.3, 0.9)))):
            (x0, x1), _ = box
            N = self.norms[l]
            self.label(sc, self.P((x0 + x1) / 2, 0.265), "layer %d: each unit's |w|" % (l + 1),
                       layout.MIN_TEXT, anchor="mb", ink=layout.CAPTION_INK, opacity=op)
            for (base, name), vals, k, g in zip(rows, (N["plain"], N["capped"]), (0, 1),
                                                (g1, g2)):
                o = op if k == 0 else op * smoothstep(0.4, 0.45, u)
                if l == 0:
                    self.label(sc, self.P(x0 - 0.012, base - 0.08), name, layout.MIN_TEXT,
                               anchor="rm", ink=NONE_INK if k == 0 else ink, opacity=o)
                self._axis(sc, box, base, o, (1.5, 2.0, 2.5, 3.0), self._norm_x)
                if k == 0:      # the He initialisation's lengths, pale, behind
                    self._hist_row(sc, box, base, 0.2, N["init"], INIT_INK, o, g1, NORM_RANGE,
                                   NORM_BIN, NORM_FULL, self._norm_x)
                self._hist_row(sc, box, base, 0.2, vals, NONE_INK if k == 0 else ink,
                                            o, g, NORM_RANGE, NORM_BIN, NORM_FULL, self._norm_x)
                if k == 1:
                    at = int((vals >= self.cap - 1e-6).sum())
                    fx = self._norm_x(box, self.cap)
                    self.label(sc, self.P(fx, base - 0.2), "%d of %d at the cap" % (at, len(vals)),
                               layout.MIN_TEXT, anchor="lb", dx=8, ink=ink,
                               opacity=o * smoothstep(0.6, 0.7, u))
            fx = self._norm_x(box, self.cap)
            for base, _name in rows:          # through each row, clear of the tick labels
                dashed(self, sc, self.P(fx, base - 0.2, 0.5), self.P(fx, base, 0.5), ink,
                       cap_op, dashes=8)
            self.label(sc, self.P(fx, 0.33), "c = %s" % _num(self.cap), layout.MIN_TEXT,
                       anchor="lb", dx=4, dy=-2, ink=ink, opacity=cap_op)
            self.label(sc, self.P(self._norm_x(box, float(N["init"].min())), 0.33),
                       "init", layout.MIN_TEXT, anchor="rb", dx=-4, dy=-2, ink=HEAD_INK,
                       opacity=op * smoothstep(0.1, 0.2, u))

    # -- 4. label smoothing -------------------------------------------------------------
    def _targets(self, sc, box, values, ink, title, op, grow):
        (x0, x1), (top, bottom) = box
        n = len(values)
        step = (x1 - x0) / n
        for k, v in enumerate(values):
            fx = x0 + step * (k + 0.5)
            p0, p1 = self.P(fx - step * 0.32, bottom - (bottom - top) * v * grow), self.P(
                fx + step * 0.32, bottom)
            w, h = self.px(p1) - self.px(p0)
            if h > 0.5:
                self.rect(sc, (p0 + p1) / 2, w, h, ink, 0.85 * op, radius=1.0, lift=0.2)
            self.label(sc, self.P(fx, bottom), str(k), layout.MIN_TEXT, anchor="mt", dy=3,
                       ink=HEAD_INK, opacity=op)
        self.polyline(sc, [self.P(x0, bottom, 0.2), self.P(x1, bottom, 0.2)], 1.0,
                      (0.5, 0.52, 0.54), 0.7 * op)
        self.label(sc, self.P(x0, top), title, layout.MIN_TEXT, anchor="lb", dy=-4,
                   ink=ink, opacity=op)

    def _gap_x(self, box, v):
        (x0, x1), _ = box
        lo, hi = GAP_RANGE
        return x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)

    def _smooth(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("smooth", t)
        ink = _ink("smooth")
        K = 10
        hot = np.eye(K)[self.digit_label]
        soft = (1.0 - self.eps) * hot + self.eps / K
        self._targets(sc, ((0.06, 0.32), (0.33, 0.52)), hot, NONE_INK, "one-hot target", op,
                      smoothstep(0.0, 1.0, remap(u, 0.02, 0.1)))
        so = op * smoothstep(0.1, 0.16, u)
        self._targets(sc, ((0.06, 0.32), (0.64, 0.83)), soft, ink,
                      "smoothed: %s and %s" % (_num(soft[self.digit_label]), _num(self.eps / K)),
                      so,
                      smoothstep(0.0, 1.0, remap(u, 0.12, 0.22)))
        # the logit gaps on the test digits
        ho = op * smoothstep(0.3, 0.36, u)
        box = ((0.44, 0.94), (0.3, 0.9))
        self.label(sc, self.P(0.69, 0.265), "top logit minus the others' mean, 2,000 test digits",
                   layout.MIN_TEXT, anchor="mb", ink=layout.CAPTION_INK, opacity=ho)
        for k, (base, gaps, col) in enumerate(((0.52, self.gaps[0], NONE_INK),
                                               (0.80, self.gaps[1], ink))):
            o = ho if k == 0 else op * smoothstep(0.45, 0.5, u)
            g = smoothstep(0.0, 1.0, remap(u, 0.32 + 0.15 * k, 0.5 + 0.15 * k))
            counts = hist(gaps, *GAP_RANGE, GAP_BIN)[1]
            self._hist_row(sc, box, base, 0.17, gaps, col, o, g, GAP_RANGE, GAP_BIN,
                           max(int(counts.max()), 1), self._gap_x)
            self._axis(sc, box, base, o, (0, 10, 20, 30, 40, 50), self._gap_x)
            med = float(np.median(gaps))
            self.label(sc, self.P(self._gap_x(box, med), base - 0.17), "median %s" % _num(
                med, 1), layout.MIN_TEXT, anchor="lb", dx=4, ink=col,
                opacity=o * smoothstep(0.55, 0.62, u))
        lo = op * smoothstep(0.62, 0.7, u)
        fx = self._gap_x(box, self.gap_line)
        dashed(self, sc, self.P(fx, 0.33, 0.5), self.P(fx, 0.87, 0.5), ink, lo)
        self.label(sc, self.P(fx, 0.87), "gap %s: softmax = target" % _num(self.gap_line, 2),
                   layout.MIN_TEXT, anchor="lt", dx=5, ink=ink, opacity=lo)

    # -- 5. summary ---------------------------------------------------------------------
    def _summary(self, sc, t, op):
        if op <= 0.01:
            return
        cols = (0.02, 0.24, 0.42, 0.53, 0.58)
        heads = (("regulariser", "lm"), ("strength", "lm"), ("test acc.", "rm"),
                 ("vs plain", "rm"), ("what it changed", "lm"))
        y0, row = 0.33, 0.1
        for (s, anc), x in zip(heads, cols):
            self.label(sc, self.P(x, y0), s, layout.MIN_TEXT, anchor=anc, ink=HEAD_INK,
                       opacity=op)
        for i, (name, strength, acc, ink, note) in enumerate(self.summary_rows()):
            y = y0 + row * (i + 1)
            o = op * smoothstep(0.08 * i, 0.08 * i + 0.1, self.progress("summary", t))
            vs = "" if strength is None else _delta(acc - self.acc_plain)
            cells = ((cols[0], name, "lm", ink), (cols[1], strength or "", "lm", HEAD_INK),
                     (cols[2], pct(acc), "rm", layout.CAPTION_INK),
                     (cols[3], vs, "rm", layout.STRONG_INK if strength and acc - self.acc_plain
                      > TIE else HEAD_INK),
                     (cols[4], note, "lm", layout.CAPTION_INK))
            for x, s, anc, k in cells:
                self.label(sc, self.P(x, y), s, layout.MIN_TEXT, anchor=anc, ink=k, opacity=o)

    def summary_rows(self):
        """``(name, strength, test accuracy, ink, what changed)`` per row."""
        return [
            ("plain", None, self.acc_plain, NONE_INK, "memorises the 1,000 digits"),
            ("dropout", "p = %s" % _num(self.p), self.dropout_acc, _ink("dropout"),
             "half its units dropped: %s (plain %s)" % (pct(self.drop_acc[1]),
                                                       pct(self.drop_acc[0]))),
            ("input noise", "σ = %s" % _num(self.sd), self.noise_acc, _ink("noise"),
             "noisy digits: %s (plain %s)" % (pct(self.noisy_acc[1]), pct(self.noisy_acc[0]))),
            ("max-norm", "c = %s" % _num(self.cap), self.maxnorm_acc, _ink("maxnorm"),
             "%d%% of layer-1 units at the cap" % round(100 * self._at_cap(0))),
            ("label smoothing", "ε = %s" % _num(self.eps), self.smooth_acc, _ink("smooth"),
             "wrong answers %s sure (plain %s)" % (_num(self.conf_wrong[1], 2),
                                                   _num(self.conf_wrong[0], 2))),
        ]

    def _at_cap(self, l):
        return float((self.norms[l]["capped"] >= self.cap - 1e-6).mean())

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        T = self.Tm
        if reset > 0 or t < T["dropout"][0] + 0.8:
            return ("the race's network, 784 → 256 → 256 → 10, memorising 1,000 digits: four "
                    "regularisers, one training step each, and what each changes once trained")
        if t < T["noise"][0]:
            u = self.progress("dropout", t)
            m = self.masks[self._mask_index(t)]
            if u < 0.62:
                return ("dropout p = %s: each step a fresh mask switches off each unit with "
                        "probability %s (%d of %d kept here); the kept ones are scaled by 1/%s "
                        "so the next layer sees the same total on average"
                        % (_num(self.p), _num(self.p), int(m.sum()), len(m), _num(1 - self.p)))
            return ("no unit can rely on another being there: with half its hidden units "
                    "dropped at test time the plain network falls from %s to %s, the dropout "
                    "network only to %s" % (pct(self.acc_plain), pct(self.drop_acc[0]),
                                            pct(self.drop_acc[1])))
        if t < T["maxnorm"][0]:
            u = self.progress("noise", t)
            if u < 0.58:
                return ("input noise σ = %s: every step adds fresh Gaussian noise to every pixel, "
                        "so the network never sees the same digit twice" % _num(self.sd))
            return ("trained on noise, it reads noisy digits: %s against the plain network's %s "
                    "at σ = %s; clean test accuracy %s against %s"
                    % (pct(self.noisy_acc[1]), pct(self.noisy_acc[0]), _num(self.sd),
                       pct(self.noise_acc), pct(self.acc_plain)))
        if t < T["smooth"][0]:
            u = self.progress("maxnorm", t)
            N0 = self.norms[0]
            if u < 0.42:
                return ("max-norm: after each step a hidden unit whose incoming weights are "
                        "longer than c is scaled back to c; untouched, they start near %s "
                        "(He) and grow to a median %s, the longest %s"
                        % (_num(float(np.median(N0["init"])), 2),
                           _num(float(np.median(N0["plain"])), 2),
                           _num(float(N0["plain"].max()), 2)))
            return ("with c = %s, just above where He weights start, %d%% of layer 1's units "
                    "and %d%% of layer 2's sit at the cap; test accuracy %s against %s: here it "
                    "binds, but buys nothing" % (_num(self.cap), round(100 * self._at_cap(0)),
                                                 round(100 * self._at_cap(1)),
                                                 pct(self.maxnorm_acc), pct(self.acc_plain)))
        if t < T["summary"][0]:
            u = self.progress("smooth", t)
            if u < 0.3:
                return ("label smoothing ε = %s: the target keeps %s on the label and spreads "
                        "%s over each other class instead of demanding 1 and 0"
                        % (_num(self.eps), _num(1 - self.eps + self.eps / 10),
                           _num(self.eps / 10)))
            if u < 0.62:
                return ("plain, the top logit runs away from the rest (median gap %s, %d%% of "
                        "test digits above 0.99 sure); smoothed, it stops near %s"
                        % (_num(float(np.median(self.gaps[0])), 1), round(100 * self.sure[0]),
                           _num(self.gap_line, 2)))
            return ("the softmax meets the smoothed target at a gap of %s: median confidence "
                    "%s instead of %s, wrong answers %s sure instead of %s; test accuracy %s "
                    "against %s" % (_num(self.gap_line, 2), _num(self.conf_med[1], 2),
                                    _num(self.conf_med[0], 2), _num(self.conf_wrong[1], 2),
                                    _num(self.conf_wrong[0], 2), pct(self.smooth_acc),
                                    pct(self.acc_plain)))
        return self._summary_words()

    def _summary_words(self):
        rows = self.summary_rows()[1:]
        ahead = [r[0] for r in rows if r[2] - self.acc_plain > TIE]
        tied = [r[0] for r in rows if abs(r[2] - self.acc_plain) <= TIE]
        parts = []
        if ahead:
            parts.append("%s %s ahead of the plain network" % (_and(ahead),
                                                              "is" if len(ahead) == 1 else "are"))
        if tied:
            parts.append("%s %s within %.1f%% of it (2,000 test digits: a tie)"
                         % (_and(tied), "is" if len(tied) == 1 else "are", 100 * TIE))
        return "; ".join(parts) + "; each changed what it targets"


def _num(v, digits=None):
    """``0.5`` as ``0.5``, ``1.5`` as ``1.5``; ``digits`` fixes the decimals."""
    if digits is not None:
        return "%.*f" % (digits, v)
    return "%g" % float(v)


def _delta(d):
    """A difference in accuracy, signed, in points: ``+2.2``."""
    return ("+%.1f" if d >= 0 else "−%.1f") % abs(100 * d)


def _and(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
