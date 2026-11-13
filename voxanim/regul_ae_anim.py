"""A regularised autoencoder: the tied-weight MNIST autoencoder of
:mod:`voxanim.regul_ae` trained plain and under four regularisers, each one
turned up until it costs the same clean reconstruction error
(:data:`voxanim.regul.AE_BUDGET`), so what each buys can be compared.

    x (784) -> h = ReLU(x W + b) (256) -> r = sigmoid(h W^T + b') (784)

1. **objectives** -- the model on one digit (the digit, its 16 x 16 code,
   the rebuilt digit) and the five objectives: the squared error alone, plus
   L1 on the code, plus weight decay, on a noisy input (denoising), plus the
   contractive penalty.  The weights are tied: untied, a penalty on the
   code is dodged by scale (the encoder shrinks, the decoder grows),
2. **training** -- ten epochs on 60,000 digits: each model's 25 most-used
   features (columns of W) forming through the recorded snapshots, and the
   test reconstruction error of all five on a log time axis,
3. **the code** -- one test digit's 256 code units under each model (a unit
   that is exactly 0 is dark) and the digit each rebuilds,
4. **noise** -- two test digits clean and with Gaussian noise
   (:data:`voxanim.regul_ae.NOISE_SD`), rebuilt by each model, and each
   model's mean error on 2,000 noisy test digits,
5. **what each buys** -- the table: clean error, noisy error, units on per
   digit, how far the code moves under the noise, and the Jacobian's size,
   with each column's best marked (ties named).

Every number is from :func:`voxanim.regul.ae_runs` and
:func:`voxanim.regul.ae_measures`.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import regul as R
from . import regul_ae as AE
from .diagram import Diagram, draw_image
from .easing import remap, smoothstep
from .gen_common import draw_digit
from .losschart import LossChart, pow10, sup
from .netanim import lit_colors
from .regul_plots import EN_INK, L1_INK, L2_INK, NONE_INK, SLOPE_INK, mosaic, signed_texture

INKS = (NONE_INK, L1_INK, L2_INK, SLOPE_INK, EN_INK)     # plain, L1 code, decay, denoise, contract
SHORT = ("plain", "L1 on the code", "weight decay", "denoising", "contractive")
COL_X = (0.10, 0.295, 0.49, 0.685, 0.88)                  # the five columns (layout x)
HEAD_Y = 0.285                                            # column headings
FEATURES = 16                                             # features shown per model (4 x 4)
FEAT_COLS = 4
TEXEL = 1.4                                               # px per weight in a feature mosaic
FEAT_Y = 0.49                                             # the mosaics' centre (layout y)
CODE_DIGIT = 8                                            # a 3 (index into the measured digits)
NOISE_DIGITS = (8, 0)                                     # a 3 and a 6
NOISE_X = (0.215, 0.39, 0.565, 0.74, 0.915)               # the noise phase's five columns
IN_X = 0.045                                              # ... and its input column
DIGIT_PX = 44.0
CHART = ((0.10, 0.74), (0.72, 0.91))
ERR_RANGE = (1e-3, 0.3)
TIE = 0.02                                                # within 2%: a tie
T_OBJ, T_TRAIN, T_CODE, T_NOISE, T_TABLE = 9.0, 17.0, 10.0, 11.0, 11.0
GAP = 0.8                                                 # between phases


def best_of(values, low=True, tie=TIE):
    """Indices within ``tie`` (relative) of the best of ``values``."""
    v = np.asarray(values, np.float64)
    b = v.min() if low else v.max()
    if low:
        return [int(i) for i in np.nonzero(v <= b * (1 + tie))[0]]
    return [int(i) for i in np.nonzero(v >= b * (1 - tie))[0]]


def names(idx):
    """``"weight decay and contractive"``."""
    s = [SHORT[i] for i in idx]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


class RegularisedAutoencoder(Diagram):
    """The objectives, the training, the code, noise and the table (see the module)."""

    PHASES = ("objectives", "training", "the code", "noise", "what each buys")

    def __init__(self, runs=None, measures=None, digits=None):
        super().__init__("A regularised autoencoder", "")
        self.hold = 2.5
        self.runs = R.ae_runs() if runs is None else runs
        self.ms = R.ae_measures(self.runs) if measures is None else measures
        if digits is None:
            from .mnist import load
            digits = load()[2][R.AE_MEASURE[0]:R.AE_MEASURE[1]]
        self.x = digits
        # the noisy digits exactly as measure() corrupts them
        self.xn = AE.corrupt(np.random.default_rng([AE.SEED, 2]), digits, AE.NOISE_SD)
        self.specs = [r.spec for r in self.runs]
        # each model's most-used units (on the measured digits), most-used first
        self.units = []
        for r in self.runs:
            use = (AE.encode(r.P, digits)[1] > 0).mean(0)
            self.units.append(np.argsort(-use, kind="stable")[:FEATURES])
        self.chart = LossChart(CHART, (1, int(self.runs[0].steps[-1])), ERR_RANGE, xlog=True,
                               ylog=True)
        self.total = int(self.runs[0].steps[-1])
        Tm, t = {}, 0.3
        for name, d in zip(self.PHASES, (T_OBJ, T_TRAIN, T_CODE, T_NOISE, T_TABLE)):
            Tm[name] = (t, t + d)
            t += d + GAP
        self.Tm = Tm
        self.finish(t - GAP)
        self.marks = [0.0] + [Tm[p][0] - GAP / 2 for p in self.PHASES[1:]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def window(self, name, t, fade=GAP / 2):
        """Opacity of a phase's content: in at its start, out at its end."""
        a, b = self.Tm[name]
        return smoothstep(a - fade, a, t) * (1.0 - smoothstep(b, b + fade, t))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        ops = {p: self.window(p, t) for p in self.PHASES}
        end_obj = self.Tm["objectives"][1]
        ops["objectives"] = smoothstep(0.0, 0.6, t) * (
            1.0 - smoothstep(end_obj, end_obj + GAP / 2, t))
        a = self.Tm["what each buys"][0]
        ops["what each buys"] = smoothstep(a - GAP / 2, a, t) * back
        self._heads(sc, HEAD_Y, max(ops["training"], ops["the code"]))
        self._objectives(sc, ops["objectives"])
        self._training(sc, t, ops["training"])
        self._code(sc, t, ops["the code"])
        self._noise(sc, t, ops["noise"])
        self._table(sc, t, ops["what each buys"])
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, reset)

    # -- shared pieces -------------------------------------------------------------
    def _heads(self, sc, y, op, sub=None):
        for i, x in enumerate(COL_X):
            self.label(sc, self.P(x, y), SHORT[i], layout.MIN_TEXT, ink=INKS[i], opacity=op)
            if sub:
                self.label(sc, self.P(x, y), sub[i], layout.MIN_TEXT, dy=19,
                           ink=layout.NOTE_INK, opacity=op)

    def _code_texture(self, h):
        """A 256-unit code as a 16 x 16 texture: lit where on, dark where exactly 0."""
        return lit_colors(h, float(h.max()) or 1.0).reshape(16, 16, 3)

    def _strength(self, i):
        s = self.specs[i]
        sym = {"sparse": "α", "decay": "λ", "denoise": "σ", "contract": "γ"}.get(s.kind)
        if sym is None:
            return "no penalty"
        return "%s = %s" % (sym, _sci(s.strength))

    # -- 1. the objectives --------------------------------------------------------
    def _objectives(self, sc, op):
        if op <= 0.01:
            return
        x = self.x[CODE_DIGIT]
        f = AE.forward(self.runs[0].P, x[None])
        y = 0.45
        xs = (0.28, 0.50, 0.72)
        draw_digit(self, sc, self.P(xs[0], y), x, 70, op)
        draw_image(self, sc, self.P(xs[1], y), self._code_texture(f["h"][0]), 70, 70, op,
                   grid=0.25)
        draw_digit(self, sc, self.P(xs[2], y), f["r"][0], 70, op)
        for a, b, s in ((xs[0], xs[1], "× W, ReLU"), (xs[1], xs[2], "× W transposed")):
            p0, p1 = self.shift(self.P(a, y), 45, 0), self.shift(self.P(b, y), -45, 0)
            self.polyline(sc, [p0, p1], 1.4, layout.NOTE_INK, 0.8 * op)
            self.polyline(sc, [self.shift(p1, -6, -4), p1, self.shift(p1, -6, 4)], 1.4,
                          layout.NOTE_INK, 0.8 * op)
            self.label(sc, (p0 + p1) / 2, s, layout.MIN_TEXT, dy=-14, ink=layout.NOTE_INK,
                       opacity=op)
        for px, s in ((xs[0], "x: 784 pixels"), (xs[1], "h: 256 code units"),
                      (xs[2], "r: rebuilt")):
            self.label(sc, self.P(px, y), s, layout.MIN_TEXT, dy=50, opacity=op)
        forms = ("‖r − x‖²", "+ α Σ h", "+ λ/2 ‖W‖²", "x + noise in", "+ γ ‖∂h/∂x‖²")
        yy = 0.70
        self._heads(sc, yy, op)
        for i, xx in enumerate(COL_X):
            self.label(sc, self.P(xx, yy), forms[i], layout.CAPTION, dy=24,
                       ink=layout.STRONG_INK, opacity=op)
            self.label(sc, self.P(xx, yy), self._strength(i), layout.MIN_TEXT, dy=48,
                       ink=layout.NOTE_INK, opacity=op)

    # -- 2. training -----------------------------------------------------------------
    def upto(self, t):
        """The training step the animation has reached (constant speed on a log axis)."""
        u = remap(self.progress("training", t), 0.05, 0.85)
        return 10.0 ** (u * np.log10(self.total))

    def _features(self, run, units, step):
        """The units' weights at ``step``, interpolated between the snapshots."""
        ss = run.snap_steps.astype(np.float64)
        k = int(np.clip(np.searchsorted(ss, step, side="right") - 1, 0, len(ss) - 2))
        a, b = np.log10(ss[k] + 1), np.log10(ss[k + 1] + 1)
        u = float(np.clip((np.log10(step + 1) - a) / (b - a), 0, 1))
        return (1 - u) * run.W_snaps[k][:, units] + u * run.W_snaps[k + 1][:, units]

    def _training(self, sc, t, op):
        if op <= 0.01:
            return
        step = self.upto(t) if t >= self.Tm["training"][0] else 0.0
        for i, (run, units) in enumerate(zip(self.runs, self.units)):
            W = self._features(run, units, step)
            tex = mosaic([signed_texture(W[:, j], float(np.abs(W[:, j]).max()) or 1.0,
                                         mark_zero=False) for j in range(len(units))],
                         FEAT_COLS)
            h, w = tex.shape[:2]
            draw_image(self, sc, self.P(COL_X[i], FEAT_Y), tex, w * TEXEL, h * TEXEL, op,
                       grid=0.0)
        ch = self.chart
        yt = [(v, pow10(v)) for v in (1e-3, 1e-2, 1e-1)]
        xt = [(1, "1"), (10, "10"), (100, "100"), (1000, "1,000")]
        ch.axes(sc, self.cam, self.P, op, "training step (log)", "", xt, yt)
        self.label(sc, self.P(CHART[0][0], CHART[1][0]), "test reconstruction error",
                   layout.MIN_TEXT, anchor="lb", dx=8, dy=-4, opacity=op)
        for i, run in enumerate(self.runs):
            ch.curve(sc, self.P, run.steps[1:], run.test_error[1:], max(step, 1.0), INKS[i], op,
                     width=1.5)
        end = op * smoothstep(0.85, 0.95, self.progress("training", t))
        ch.hline(sc, self.P, R.AE_BUDGET, end)
        x1 = CHART[0][1] + 0.025
        errs = [float(r.test_error[-1]) for r in self.runs]
        self.label(sc, self.P(x1, ch.fy(errs[0])), "plain %s" % _err(errs[0]), layout.MIN_TEXT,
                   anchor="lm", ink=INKS[0], opacity=end)
        self.label(sc, self.P(x1, ch.fy(R.AE_BUDGET)), "the other four", layout.MIN_TEXT,
                   anchor="lm", dy=-24, ink=layout.CAPTION_INK, opacity=end)
        self.label(sc, self.P(x1, ch.fy(R.AE_BUDGET)), "%s to %s" % (_err(min(errs[1:])),
                                                                     _err(max(errs[1:]))),
                   layout.MIN_TEXT, anchor="lm", dy=-5, ink=layout.CAPTION_INK, opacity=end)

    # -- 3. the code -------------------------------------------------------------------
    def _code(self, sc, t, op):
        if op <= 0.01:
            return
        x = self.x[CODE_DIGIT]
        u = self.progress("the code", t)
        for i, run in enumerate(self.runs):
            f = AE.forward(run.P, x[None])
            h = f["h"][0]
            o = op * smoothstep(0.0, 0.25, remap(u, 0.05 * i, 0.05 * i + 0.3))
            draw_image(self, sc, self.P(COL_X[i], 0.48), self._code_texture(h), 112, 112, o,
                       grid=0.25)
            on = int((h > 0).sum())
            self.label(sc, self.P(COL_X[i], 0.48), "%d of %d on" % (on, h.size),
                       layout.MIN_TEXT, dy=70, ink=layout.STRONG_INK, opacity=o)
            ro = o * smoothstep(0.45, 0.6, u)
            draw_digit(self, sc, self.P(COL_X[i], 0.80), f["r"][0], 64, ro)
            self.label(sc, self.P(COL_X[i], 0.80), "error %s" % _err(AE.recon_error(f["r"], x)),
                       layout.MIN_TEXT, dy=44, ink=layout.CAPTION_INK, opacity=ro)

    # -- 4. noise -----------------------------------------------------------------------
    def _noise(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("noise", t)
        noisy = smoothstep(0.3, 0.45, u)
        ys = (0.46, 0.62)
        px = DIGIT_PX * 1.25
        self.label(sc, self.P(IN_X, HEAD_Y), "in", layout.MIN_TEXT, opacity=op)
        for i in range(5):
            self.label(sc, self.P(NOISE_X[i], HEAD_Y), SHORT[i], layout.MIN_TEXT,
                       ink=INKS[i], opacity=op)
        for r, d in enumerate(NOISE_DIGITS):
            xin = (1 - noisy) * self.x[d] + noisy * self.xn[d]
            draw_digit(self, sc, self.P(IN_X, ys[r]), xin, px, op)
            for i, run in enumerate(self.runs):
                rc = AE.forward(run.P, self.x[d][None])["r"][0]
                rn = AE.forward(run.P, self.xn[d][None])["r"][0]
                draw_digit(self, sc, self.P(NOISE_X[i], ys[r]), (1 - noisy) * rc + noisy * rn,
                           px, op)
        so = op * smoothstep(0.3, 0.4, u)
        self.label(sc, self.P(IN_X - 0.03, 0.735), "mean error over %s test digits"
                   % f"{len(self.x):,}", layout.MIN_TEXT, anchor="lm", ink=layout.NOTE_INK,
                   opacity=op)
        self.label(sc, self.P(IN_X, 0.80), "clean", layout.MIN_TEXT, ink=layout.NOTE_INK,
                   opacity=op)
        self.label(sc, self.P(IN_X, 0.86), "noisy", layout.MIN_TEXT, ink=layout.NOTE_INK,
                   opacity=so)
        best = self._noisy_best()
        for i, m in enumerate(self.ms):
            self.label(sc, self.P(NOISE_X[i], 0.80), _err(m["error"]), layout.MIN_TEXT,
                       ink=layout.CAPTION_INK, opacity=op)
            ink = layout.STRONG_INK if i in best else layout.CAPTION_INK
            if i in best:
                self.rect(sc, self.P(NOISE_X[i], 0.86), 62, 24, (0.86, 0.90, 0.96), so,
                          radius=4, lift=0.05)
            self.label(sc, self.P(NOISE_X[i], 0.86), _err(m["noisy_error"]), layout.MIN_TEXT,
                       ink=ink, opacity=so)

    def _noisy_best(self):
        return best_of([m["noisy_error"] for m in self.ms])

    # -- 5. the table --------------------------------------------------------------------
    COLS = (("clean error", "error", True), ("noisy error", "noisy_error", True),
            ("units on", "active", True), ("code shift", "shift", True), ("‖∂h/∂x‖", "jac", True))
    TAB_X = (0.02, 0.36, 0.52, 0.67, 0.81, 0.95)       # name, then the five columns (right edges)
    TAB_Y0, TAB_ROW = 0.31, 0.088

    def _cell(self, key, v):
        if key in ("error", "noisy_error"):
            return _err(v)
        if key == "active":
            return "%.0f%%" % (100 * v)
        if key == "shift":
            return "%.2f" % v
        return "%.1f" % v

    def _table(self, sc, t, op):
        if op <= 0.01:
            return
        u = self.progress("what each buys", t)
        y0 = self.TAB_Y0
        self.label(sc, self.P(self.TAB_X[0], y0), "regulariser", layout.MIN_TEXT, anchor="lm",
                   ink=layout.NOTE_INK, opacity=op)
        for c, (head, key, low) in enumerate(self.COLS):
            co = op * smoothstep(0.0, 0.12, remap(u, 0.05 * c, 0.05 * c + 0.15))
            xr = self.TAB_X[c + 1]
            self.label(sc, self.P(xr, y0), head, layout.MIN_TEXT, anchor="rm",
                       ink=layout.NOTE_INK, opacity=co)
            vals = [m[key] for m in self.ms]
            best = [] if key == "error" else best_of(vals, low)     # clean error: the budget
            for i in range(5):
                y = y0 + (i + 1) * self.TAB_ROW
                win = i in best and smoothstep(0.3, 0.4, u) > 0
                if win:
                    w = layout.text_width(self._cell(key, vals[i]), 0.031) * 960 + 14
                    a = self.shift(self.P(xr, y), -w / 2 + 7, 0)
                    self.rect(sc, a, w, 24, (0.86, 0.90, 0.96), co * smoothstep(0.3, 0.4, u),
                              radius=4, lift=0.05)
                ink = layout.STRONG_INK if win else layout.CAPTION_INK
                self.label(sc, self.P(xr, y), self._cell(key, vals[i]), 0.031, anchor="rm",
                           ink=ink, opacity=co)
        for i in range(5):
            y = y0 + (i + 1) * self.TAB_ROW
            self.label(sc, self.P(self.TAB_X[0], y), SHORT[i], 0.031, anchor="lm", ink=INKS[i],
                       opacity=op)
        no = op * smoothstep(0.35, 0.45, u)
        notes = ("clean error: every regulariser held to about %s; plain pays no penalty"
                 % _err(R.AE_BUDGET),
                 "marked: the best of each other column (ties within %d%%)" % round(100 * TIE),
                 "code shift: |h(noisy) − h| / |h|, the median; ‖∂h/∂x‖: the mean over the digits")
        for k, line in enumerate(notes):
            self.label(sc, self.P(self.TAB_X[0], 0.835 + 0.05 * k), line, layout.MIN_TEXT,
                       anchor="lm", ink=layout.NOTE_INK, opacity=no)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        T = self.Tm
        ms = self.ms
        if reset > 0 or t < T["training"][0] - GAP / 2:
            return ("784 → 256 → 784, tied weights (untied, a penalty on the code is dodged by "
                    "scale: the encoder shrinks, the decoder grows); every regulariser turned up "
                    "until the clean error is %s" % _err(R.AE_BUDGET))
        if t < T["the code"][0] - GAP / 2:
            step = int(round(self.upto(t)))
            return ("10 epochs, Adam, step %s of %s; each model's %d most-used features "
                    "(columns of W): plain's speckled, L1's whole strokes and digits, weight "
                    "decay's and the contractive's smooth" % (f"{step:,}", f"{self.total:,}",
                                                              FEATURES))
        if t < T["noise"][0] - GAP / 2:
            on = [int((AE.encode(r.P, self.x[CODE_DIGIT][None])[1] > 0).sum()) for r in self.runs]
            return ("one test digit's code: L1 on the code leaves %d of 256 units on (plain "
                    "%d); weight decay and contractive switch on %d and %d; on average over "
                    "%s test digits %.0f%% against plain's %.0f%%"
                    % (on[1], on[0], on[2], on[4], f"{len(self.x):,}", 100 * ms[1]["active"],
                       100 * ms[0]["active"]))
        if t < T["what each buys"][0] - GAP / 2:
            b = self._noisy_best()
            return ("Gaussian noise of std %g on the input: %s (trained on noise of std %g) "
                    "rebuilds noisy digits best, error %s against plain's %s; on clean digits "
                    "plain, with no penalty to pay, is best (%s)"
                    % (AE.NOISE_SD, names(b), self.specs[3].strength,
                       _err(ms[b[0]]["noisy_error"]), _err(ms[0]["noisy_error"]),
                       _err(ms[0]["error"])))
        shift_best = best_of([m["shift"] for m in ms])
        if t < T["what each buys"][0] + 0.5 * T_TABLE:
            return ("each buys something else: denoising a better guess under noise, L1 a sparse "
                    "code (%.0f%% of units on), %s the steadiest code (it moves %.2f under the "
                    "noise, plain's %.2f)" % (100 * ms[1]["active"], names(shift_best),
                                              ms[shift_best[0]]["shift"], ms[0]["shift"]))
        return ("weight decay and contractive end alike: for a ReLU code ∂h/∂x is the weights "
                "of the units that are on (%.0f%% of them here), so the contractive penalty is "
                "weight decay weighted by how often each unit is on" % (100 * ms[4]["active"]))


def _err(v):
    """A reconstruction error: ``0.0074``."""
    return "%.4f" % v


def _sci(v):
    """A strength: ``1.1×10⁻⁴`` (``0.55`` as it is)."""
    if v >= 0.01:
        return "%g" % v
    e = int(np.floor(np.log10(v)))
    return "%g×10%s" % (round(v / 10.0 ** e, 2), sup(e))
