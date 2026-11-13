"""What each activation leaves on: the hidden layers of the seven trained race
networks (:mod:`voxanim.actfun_race`), and how much of them is switched off.

    784 -> 128 -> 64 -> 10 after 1,500 Adam steps, one activation each

"Off" means a different thing per family, and the read-outs say which: an
activation that is exactly 0 (ReLU, and only ReLU makes exact zeros), one
within 0.01 of 0 (Leaky ReLU, GELU, SiLU, ELU: their small negatives are
alive but tiny), or one on a flat part of the function, |f′| < 0.05
(sigmoid and tanh saturate at their ends instead of switching off).

1. **one batch** -- layer 1 (128 units) on 32 test images as a sheet, one
   network after another; cells within 0.01 of 0 are dark.  ReLU leaves 35%
   of the layer exactly 0; Leaky ReLU 22% within 0.01; GELU 4%, SiLU and ELU
   1%; tanh saturates 10% of it, sigmoid 38%,
2. **over training** -- the fraction of each layer off, per network, as
   bars that follow the recorded run (log time): ReLU starts at 51% zeros
   (He weights: half the pre-activations are negative), dips to 25% while
   the layer learns, settles at 36%,
3. **one digit** -- the 192 hidden units one test digit (an 8) uses under
   each network, lit where |a| ≥ 0.01: 146 under ReLU, 187 under GELU, all
   192 under tanh, SiLU and ELU,
4. **dead units** -- units off on every one of the 2,000 test images: ReLU
   has 3 of 192 (in red); Leaky ReLU, GELU and the rest none, and none can die
   for good, because their slope is never exactly 0.

Every number is from the recorded runs (:func:`voxanim.actfun.race_runs`),
their final parameters applied to the test subset.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .actfun import ACTS, N_TEST, RACE, RACE_SIZES, data, dead_units, fractions, hidden, race_runs
from .deepnet import FLAT, NEAR_ZERO
from .diagram import Diagram
from .easing import ease_in_out, remap, smoothstep
from .netanim import OFF, amax, div_colors, lit_colors, screen_box

STEPS = RACE["steps"]
WIDTHS = RACE_SIZES[1:-1]                        # (128, 64)
N_UNITS = sum(WIDTHS)
BATCH_N = 32                                     # test images on the sheet
SHEET_LAYER = 0                                  # layer 1 is the sheet
DIGIT = 0                                        # the test image of phases 3 and 4 (an 8)
SIGNED = ("tanh", "leaky_relu", "gelu", "silu", "elu")
MEASURE = {"sigmoid": "flat", "tanh": "flat", "relu": "zero"}      # the rest: "small"
MEASURE_LABEL = {"zero": "exactly 0", "small": "|a| < %g" % NEAR_ZERO,
                 "flat": "saturated, |f′| < %g" % FLAT}
DEAD_INK = (0.85, 0.25, 0.25)

SHEET_AT, SHEET_SCALE = (0.44, 0.37), 0.45
NOTE_X = 0.745                                   # read-outs right of the sheet
STRIP_GAP, PAIR_GAP = 0.6, 3.4                   # cells between a pair's rows, between pairs
BARS = ((0.08, 0.95), (0.64, 0.86))              # the bar chart: (x0, x1), (top, baseline)
BAR_W = 0.036                                    # layout width of one bar
AXIS = (0.5, 0.52, 0.54)
ACT_DT, SWITCH = 2.0, 0.45                       # per network on the sheet; the crossfade
TRAIN_T = 9.0
STRIP_DT = 0.7
DEAD_T = 6.5


def article(digit):
    """``"an 8"``, ``"a 3"``."""
    return ("an %d" if digit == 8 else "a %d") % digit


def measure(key):
    """Which "off" the family uses: ``"zero"``, ``"small"`` or ``"flat"``."""
    return MEASURE.get(key, "small")


class ActivationSparsity(Diagram):
    """The sheet, the bars, the strips and the dead units (see the module)."""

    PHASES = ("one batch", "over training", "one digit", "dead units")

    def __init__(self, runs=None, dat=None):
        super().__init__("What each activation leaves on", "")
        dat = data() if dat is None else dat
        self.runs = race_runs(dat=dat) if runs is None else runs
        self.hold = 2.5
        x = dat.x_test[:BATCH_N]
        self.digit = int(dat.y_test[DIGIT])
        self.sheets_a = []                        # per network: layer 1 on the batch (32, 128)
        self.final = []                           # per network: (zero, small, flat) per layer
        self.digit_a = []                         # per network: per layer, the digit's activations
        self.dead = []                            # per network: per layer, dead mask
        for r in self.runs:
            _ys, As = hidden(r.spec, r.P, x)
            self.sheets_a.append(As[SHEET_LAYER])
            self.final.append(fractions(r.spec, r.P, x))
            self.digit_a.append([a[0] for a in hidden(r.spec, r.P, dat.x_test[DIGIT:DIGIT + 1])[1]])
            self.dead.append(dead_units(r.spec, r.P, dat.x_test))
        self.steps = self.runs[0].steps
        self.sheet_ = self.sheet((BATCH_N, WIDTHS[SHEET_LAYER]), SHEET_AT, SHEET_SCALE, "layer")
        self.strips = self._strips()

        Tm, t = {}, 0.3
        Tm["batch"] = (t, t + len(ACTS) * ACT_DT)
        t += len(ACTS) * ACT_DT + 1.0
        Tm["train"] = (t, t + TRAIN_T)
        t += TRAIN_T + 1.5
        Tm["digit"] = (t, t + len(ACTS) * STRIP_DT)
        t += len(ACTS) * STRIP_DT + 2.5
        Tm["dead"] = (t, t + DEAD_T)
        t += DEAD_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["train"][0], Tm["digit"][0], Tm["dead"][0]]

    def _strips(self):
        """Per network, a pair of one-row sheets (layer 1, layer 2) stacked where
        the big sheet is."""
        centre = self.cam.frac(*SHEET_AT)
        s = SHEET_SCALE
        pair_h = (2 + STRIP_GAP) * s
        total = len(ACTS) * pair_h + (len(ACTS) - 1) * PAIR_GAP * s
        z = centre[2] + total / 2
        strips = []
        for k in range(len(ACTS)):
            pair = []
            for l, w in enumerate(WIDTHS):
                sh = self.sheet((1, w), SHEET_AT, s, "strip%d_%d" % (k, l))
                # left-aligned with the big sheet
                c = centre.copy()
                c[0] = self.sheet_.block.origin[0] + sh.block.extent[0] / 2
                c[2] = z - (0.5 + l * (1 + STRIP_GAP)) * s
                sh.block.place(c)
                pair.append(sh)
            z -= pair_h + PAIR_GAP * s
            strips.append(pair)
        return strips

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ state
    def current(self, t):
        """``(k, swap)``: the network on the sheet at ``t`` and its crossfade (1 =
        steady); after the batch phase the sheet shows the ReLU network, whose
        run the training phase quotes."""
        u = (t - self.Tm["batch"][0]) / ACT_DT
        k = int(np.clip(np.floor(u), 0, len(ACTS) - 1))
        if t >= self.Tm["batch"][1]:
            relu = next(n for n, a in enumerate(ACTS) if a.key == "relu")
            return relu, smoothstep(0.0, SWITCH, t - self.Tm["batch"][1])
        frac = u - k
        swap = 1.0 if k == 0 and frac < SWITCH / ACT_DT else smoothstep(0.0, SWITCH / ACT_DT, frac)
        return k, swap

    def step_at(self, train):
        """The training step the bars show, constant speed on a log axis."""
        return 10.0 ** (remap(train, 0.0, 0.85) * np.log10(STEPS)) if train > 0 else 0.0

    def fraction(self, k, l, step=None):
        """Network ``k``'s "off" fraction of layer ``l``: at the end on the test
        batch, or at training ``step`` on the run's minibatch."""
        m = measure(ACTS[k].key)
        if step is None:
            return float(self.final[k][("zero", "small", "flat").index(m)][l])
        return float(np.interp(step, self.steps, self.runs[k].sparse[m][:, l]))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 0.6, t)
        train = self.progress("train", t)
        strips = smoothstep(0.0, 0.3, remap(t, self.Tm["digit"][0] - 1.0, self.Tm["digit"][0]))
        sheet_op = (1.0 - strips) * back * intro
        bars_op = (1.0 - strips) * back
        k, swap = self.current(t)
        self._sheet(sc, k, swap, sheet_op)
        self._bars(sc, t, train, bars_op)
        self._strips_draw(sc, t, strips * back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, k, train, reset)

    # -- the sheet -------------------------------------------------------------------------
    def _colors(self, a, key):
        """Cell colours of activations ``a``: signed or lit by family; off cells dark."""
        v = np.asarray(a, np.float64).ravel()
        c = div_colors(v, amax(v)) if key in SIGNED else lit_colors(v, amax(v))
        c[np.abs(v) < NEAR_ZERO] = OFF
        return c

    def _sheet(self, sc, k, swap, op):
        s = self.sheet_
        if op <= 0.01:
            s.visible = False
            return
        a = ACTS[k]
        s.fill(self._colors(self.sheets_a[k], a.key))
        s.cage = op
        s.fade(op * swap)
        self.caption(sc, s, "layer 1 (%d units) of the trained %s network" % (WIDTHS[0], a.label),
                     op * swap, above=True, size=layout.MIN_TEXT, max_width=0.5)
        self.caption_left(sc, s, "%d images" % BATCH_N, op)
        zero, small, flat = (self.final[k][i][SHEET_LAYER] for i in range(3))
        m = measure(a.key)
        lines = (("exactly 0: %d%%" % round(100 * zero), "zero"),
                 ("within %g of 0: %d%%" % (NEAR_ZERO, round(100 * small)), "small"),
                 ("flat part, |f′| < %g: %d%%" % (FLAT, round(100 * flat)), "flat"))
        lo, hi = s.bbox()
        _x0, y0, _x1, y1 = screen_box(self.cam, lo, hi)
        y_mid = (y0 + y1) / 2 / self.cam.height
        fx = self.fr(self.P(NOTE_X, 0.0))[0]
        for n, (text, which) in enumerate(lines):
            ink = layout.STRONG_INK if which == m else layout.NOTE_INK
            size = 0.031 if which == m else layout.MIN_TEXT
            sc.text(fx, y_mid + (n - 1) * 0.055, text, size, "regular", ink, op * swap * self._cap,
                    "lm")
        off_text = "off, for %s: %s" % (a.label, MEASURE_LABEL[m].split(",")[0])
        sc.text(fx, y_mid + 2 * 0.055, off_text, layout.MIN_TEXT, "regular", a.ink,
                op * swap * self._cap, "lm")

    # -- the bars --------------------------------------------------------------------------
    def _bar_x(self, k, l):
        """Layout x of the left edge of network ``k``'s layer ``l`` bar."""
        (x0, x1), _y = BARS
        group = (x1 - x0) / len(ACTS)
        return x0 + k * group + group / 2 - BAR_W - 0.004 + l * (BAR_W + 0.008)

    def _bars(self, sc, t, train, op):
        op *= self._cap
        if op <= 0.01:
            return
        (x0, x1), (top, base) = BARS
        sc.line(self.P(x0, base), self.P(x1, base), width=1.0, color=AXIS, opacity=0.7 * op,
                bias=0.0)
        sc.line(self.P(x0, base), self.P(x0, top), width=1.0, color=AXIS, opacity=0.7 * op,
                bias=0.0)
        for v in (0.0, 0.5, 1.0):
            fy = base - (base - top) * v
            self.label(sc, self.P(x0, fy), "%d%%" % round(100 * v), layout.MIN_TEXT, anchor="rm",
                       dx=-6, ink=layout.NOTE_INK, opacity=op)
        self.label(sc, self.P(x0, top), "share of the layer off (layer 1, layer 2)",
                   layout.MIN_TEXT, anchor="lb", dy=-6, opacity=op)
        k_now, swap = self.current(t)
        step = self.step_at(train) if train > 0 else None
        for k, a in enumerate(ACTS):
            shown = 1.0 if k < k_now or train > 0 else (swap if k == k_now else 0.0)
            if t >= self.Tm["batch"][1]:
                shown = 1.0
            if shown <= 0.01:
                continue
            for l in range(len(WIDTHS)):
                frac = self.fraction(k, l, step)
                h = (base - top) * frac
                xa = self._bar_x(k, l)
                p0, p1 = self.P(xa, base - h), self.P(xa + BAR_W, base)
                w, hh = self.px(p1) - self.px(p0)
                if hh > 0.5:
                    self.rect(sc, (p0 + p1) / 2, w, hh, a.ink, (0.95 if l == 0 else 0.55) * op
                              * shown, radius=1.5, lift=0.2)
            self.label(sc, self.P(self._bar_x(k, 0) + BAR_W + 0.004, base), a.label,
                       layout.MIN_TEXT, anchor="mt", dy=5, ink=a.ink, opacity=op * shown)
        self.label(sc, self.P((x0 + x1) / 2, base), "off: sigmoid and tanh saturated (|f′| < %g), "
                   "ReLU exactly 0, the rest |a| < %g" % (FLAT, NEAR_ZERO), layout.MIN_TEXT,
                   anchor="mt", dy=27, ink=layout.NOTE_INK, opacity=op)
        if train > 0:
            sc.probe(("bars",), (np.log10(max(step, 1.0)) * 100, 0, 0), tol=6.0)

    # -- the strips ------------------------------------------------------------------------
    def _strips_draw(self, sc, t, op):
        dead = smoothstep(0.0, 0.4, self.progress("dead", t))
        for k, (a, pair) in enumerate(zip(ACTS, self.strips)):
            t0 = self.Tm["digit"][0] + k * STRIP_DT
            on = remap(t, t0, t0 + STRIP_DT)
            if on <= 0 or op <= 0.01:
                for sh in pair:
                    sh.visible = False
                continue
            used = 0
            n_dead = 0
            for l, sh in enumerate(pair):
                v = self.digit_a[k][l]
                c = self._colors(v, a.key)
                lit = np.abs(v) >= NEAR_ZERO
                used += int(lit.sum())
                d = self.dead[k][l]
                n_dead += int(d.sum())
                if dead > 0 and d.any():
                    c[d] = c[d] + (np.asarray(DEAD_INK, np.float32) - c[d]) * dead
                sh.fill(c)
                sh.show(np.arange(len(v)) < on * len(v) * 1.1)
                sh.cage = op
                sh.fade(op)
                if dead > 0 and d.any():
                    sh.light(d.astype(float), dead)
            lo = np.minimum(pair[0].bbox()[0], pair[1].bbox()[0])
            hi = np.maximum(pair[0].bbox()[1], pair[1].bbox()[1])
            box = (lo, hi)
            self.caption_left(sc, _Box(*box), a.label, op)
            if dead > 0:
                self.caption_side(sc, box, "%d dead" % n_dead if n_dead else "none dead",
                                  op * dead)
            else:
                self.caption_side(sc, box, "%d of %d on" % (used, N_UNITS),
                                  op * smoothstep(0.8, 1.0, on))
        if op > 0.01 and t >= self.Tm["digit"][0]:
            lo = self.strips[0][0].bbox()[0]
            hi = self.strips[0][0].bbox()[1]
            head = ("one test digit (%s) through each network: its %d hidden units, lit where "
                    "|a| ≥ %g" % (article(self.digit), N_UNITS, NEAR_ZERO))
            if dead > 0:
                head = "units off on every one of the %s test images, in red" % f"{N_TEST:,}"
            self.caption(sc, (lo, hi), head, op, above=True, size=layout.MIN_TEXT, max_width=0.7)

    # ------------------------------------------------------------------ words
    def _words(self, t, k, train, reset):
        T = self.Tm
        i = {a.key: n for n, a in enumerate(ACTS)}
        if reset > 0 or t < T["batch"][0] + SWITCH:
            return ("the seven trained race networks (784 → %d → %d → 10, %s Adam steps); which "
                    "hidden units does each activation leave switched off?"
                    % (WIDTHS[0], WIDTHS[1], f"{STEPS:,}"))
        if t < T["train"][0]:
            a = ACTS[k]
            zero, small, flat = (100 * self.final[k][n][SHEET_LAYER] for n in range(3))
            return {
                "sigmoid": "sigmoid: no zeros (σ > 0 everywhere), but %d%% of the layer sits on a "
                           "flat end, σ ≈ 0 or 1, where the slope is below %g" % (flat, FLAT),
                "tanh": "tanh: no zeros; %d%% saturated at ±1 (full colour), where the slope is "
                        "below %g" % (flat, FLAT),
                "relu": "ReLU: %d%% of the layer is exactly 0 (dark): those units pass nothing "
                        "forward and no gradient back" % zero,
                "leaky_relu": "Leaky ReLU: nothing is exactly 0 (0.01 z instead); %d%% within %g "
                              "of 0, %d%% on the shallow side" % (small, NEAR_ZERO, flat),
                "gelu": "GELU: %d%% within %g of 0; the small negatives (blue) stay alive, so the "
                        "layer is far less sparse than ReLU's %d%%"
                        % (small, NEAR_ZERO, 100 * self.final[i["relu"]][0][SHEET_LAYER]),
                "silu": "SiLU: %d%% within %g of 0, %d%% on a flat part: almost everything stays "
                        "on" % (small, NEAR_ZERO, flat),
                "elu": "ELU: negatives saturate at −1 instead of 0 (blue); %d%% within %g of 0, "
                       "%d%% flat" % (small, NEAR_ZERO, flat),
            }[a.key]
        if t < T["digit"][0] - 1.0:
            step = int(round(self.step_at(train)))
            r = self.runs[i["relu"]].sparse["zero"][:, 0]
            return ("over training, step %s: ReLU's layer 1 starts %d%% zeros (He weights: half "
                    "the pre-activations negative), dips to %d%%, ends at %d%%"
                    % (f"{step:,}", round(100 * r[0]), round(100 * r.min()), round(100 * r[-1])))
        if t < T["dead"][0]:
            used = [sum(int((np.abs(v) >= NEAR_ZERO).sum()) for v in self.digit_a[n])
                    for n in range(len(ACTS))]
            return ("one digit (%s) through each network: it uses %d of %d hidden units under "
                    "ReLU, %d under Leaky ReLU, %d under GELU, %d under tanh"
                    % (article(self.digit), used[i["relu"]], N_UNITS, used[i["leaky_relu"]],
                       used[i["gelu"]], used[i["tanh"]]))
        n_dead = [sum(int(d.sum()) for d in self.dead[n]) for n in range(len(ACTS))]
        others = max(n_dead[n] for n in range(len(ACTS))
                     if ACTS[n].key not in ("relu", "leaky_relu", "gelu"))
        return ("dead units (off on all %s test images): ReLU %d of %d; Leaky ReLU %d, GELU %d, "
                "the rest %d: a slope that is never exactly 0 keeps every unit trainable"
                % (f"{N_TEST:,}", n_dead[i["relu"]], N_UNITS, n_dead[i["leaky_relu"]],
                   n_dead[i["gelu"]], others))


class _Box:
    """A world box with the :meth:`~voxanim.netanim.Sheet.bbox` interface."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def bbox(self, offsets=True):
        return self.lo, self.hi
