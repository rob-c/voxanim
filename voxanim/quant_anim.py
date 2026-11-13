"""Quantization, post-training and quantization-aware, on the trained MNIST MLP.

Both animations work on the real network of :mod:`voxanim.mnist` (784-64-32-10,
96.8% test accuracy) and draw its middle weight matrix ``W1`` (64 x 32) as a
sheet; every accuracy is the whole network's, with all three weight
matrices quantized the same way (:mod:`voxanim.quant`).

**Quantization** (``quantization``):

1. **float32** -- the sheet and the histogram of its 2,048 weights;
2. **int8** -- the 255 levels of a per-tensor symmetric quantizer appear
   over the histogram, the weights snap to them row by row (the sheet's
   colours step), the error ``W - Q(W)`` fills a second sheet, the test
   accuracy lands in the table: 96.8%, unchanged;
3. **int4** -- 15 levels: the error is visible, the accuracy drops 0.6 points;
4. **2 bits** -- three levels ``-s, 0, s``: with ``s`` the largest weight
   nine in ten weights round to 0 and the network falls to chance (13.1%);
   one scale per output channel brings 29.2%; a *calibrated* clip (the
   fraction of the range that minimises the squared error, per channel)
   78.0%;
5. **size** -- the bytes each version takes (weights packed, biases and
   scales in float32): 211 kB, 53, 27, 14; and int8 activations too.

**Quantization-aware training** (``quantization-aware-training``):

1. **rounding** -- ``Q(x) = s round(x / s)`` is a staircase whose derivative
   is 0 almost everywhere (and undefined at the steps); the straight-through
   estimator (STE) uses 1 inside the clipping range instead;
2. **forward** -- the network computes with ``Q(W)``, the 2-bit weights
   (per-tensor scale, calibrated): 76.4% to start with;
3. **backward** -- ``dL/dQ(W)`` on a real minibatch passes straight through
   to the float weights (except where they saturate); Adam moves them a
   little, and those that cross a boundary change level: the flips of the
   first steps are counted;
4. **training** -- two epochs of this: the quantized network's accuracy
   against the step for 2 and 4 bits, the sheets following the run, and
   the histogram of the float weights against the levels and boundaries --
   they do *not* gather at the levels; a third of them end within a tenth
   of a step of a boundary, oscillating across it;
5. **result** -- post-training against quantization-aware, side by side.

Every number is read from the trained model and the recorded runs; the STE
gradient is checked against its definition in ``tests/test_quant.py`` (a
finite-difference check of the rounded loss is impossible: it is flat).
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from . import quant as Q
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10
from .netanim import OFF, NetAnimation, amax, div_colors, lit_colors, particle
from .text import MINUS, TIMES, num, sub

PPU = 14.0
ROWS, COLS = 32, 64                     # W1 drawn transposed: a row per output neuron
SHEET_SCALE = 0.36
HBINS, HROWS = 64, 12                   # the weight histogram: bins over (-HRANGE, HRANGE), rows
HSCALE, HPITCH = 0.5, (1.0, 1.25)
HIST_PAD = 1.05                         # histogram range: the largest |weight| times this
HIST_INK = C.rgb("#8a8f94")             # the float histogram's grey
GRID_INK = (0.36, 0.38, 0.42)           # the quantization levels drawn over the histogram
NEAR = 0.1                              # "near" a level or boundary: within this many steps
LEVEL_IN, SNAP, LAND = 1.0, 3.0, 0.5    # seconds: levels appear, the sheet snaps, the row lands


def _ink(col):
    """A readable text colour from a line colour."""
    return tuple(float(v) for v in np.asarray(col) * 0.85)


def histogram(values, bins, half_range):
    """Fraction of ``values`` in each of ``bins`` bins over ``(-half_range, half_range)``."""
    counts, _edges = np.histogram(np.asarray(values).ravel(), bins, (-half_range, half_range))
    return counts / max(1, np.asarray(values).size)


def fill_histogram(sheet, frac, full, colors, op):
    """Draw ``frac`` (one fraction per bin) on a ``(rows, bins)`` sheet as columns,
    ``full`` being the fraction that fills a column; ``colors`` ``(bins, 3)``."""
    rows, bins = sheet.h, sheet.w
    present = np.zeros((rows, bins), bool)
    for b in range(bins):
        h = int(round(min(1.0, frac[b] / full) * rows))
        present[rows - h:, b] = True
    sheet.fill(np.repeat(np.asarray(colors)[None], rows, 0).reshape(-1, 3))
    sheet.show(present.ravel())
    sheet.cage = op
    sheet.fade(op)


def sweep_rows(rows, u, span=0.85):
    """Per-row progress of a top-to-bottom sweep at progress ``u`` (0..1)."""
    r = np.arange(rows) / max(rows - 1, 1)
    return np.array([smoothstep(span * x, span * x + (1 - span), u) for x in r])


class _Table:
    """Right-aligned columns of text at layout fractions: a small results table."""

    def __init__(self, anim, x_cols, y0, step, size=0.028):
        self.anim, self.x_cols, self.y0, self.step, self.size = anim, x_cols, y0, step, size

    def row(self, sc, n, cells, op, ink=None, kind="regular"):
        """Row ``n`` (0 = the header): ``cells`` left-aligned in the first
        column, right-aligned in the rest; None cells are skipped."""
        op *= self.anim._cap
        if op <= 0.01:
            return
        y = self.y0 + n * self.step
        for k, (x, text) in enumerate(zip(self.x_cols, cells, strict=True)):
            if text is None:
                continue
            fx, fy = self.anim.fr(x, y)
            sc.text(fx, fy, text, self.size, kind, ink or layout.CAPTION_INK, op,
                    "lm" if k == 0 else "rm")


# ------------------------------------------------------------------ post-training quantization

W_AT, E_AT = (0.19, 0.72), (0.49, 0.72)         # the weights and the error sheets
H_AT = (0.34, 0.40)                             # the histogram
TABLE_X = (0.69, 0.79, 0.885, 0.955)            # columns: name, levels, accuracy, size
TABLE_Y, TABLE_STEP = 0.30, 0.052
BYTES_COLS = 48                                 # the size bars: cells for the float32 model
BYTES_PITCH = (4.5, 1.0)

# (bits, scheme, table row) in the order the animation applies them.
STAGES = ((8, "per-tensor", 2), (4, "per-tensor", 3), (2, "per-tensor", 4),
          (2, "per-channel", 5), (2, "calibrated", 6))
ROW_NAMES = ("", "float32", "int8", "int4", "2 bits", "  per-channel", "  calibrated")


class Quantization(NetAnimation):
    """``quantization``: the trained MLP's weights at 8, 4 and 2 bits (see the module)."""

    PHASES = ("float32", "int8", "int4", "2 bits", "size")

    def __init__(self):
        super().__init__("Quantization", "", PPU)
        self.hold = 2.5
        self.params, self.acc, (xtr, _ytr, xte, yte) = Q.trained_mlp()
        self.W = self.params[Q.WATCH][0]                       # (64, 32)
        self.wmax = amax(self.W)
        self.hrange = HIST_PAD * self.wmax
        self.n_weights = sum(W.size for W, _ in self.params)
        self.table_acc = Q.ptq_accuracies(self.params, xte, yte)
        self.bytes = {b: Q.model_bytes(self.params, b) for b in (32,) + Q.BITS}
        ranges = Q.activation_ranges(self.params, xtr[:Q.CALIB_IMAGES])
        self.act_acc = {}
        for wb, ab in ((8, 8), (4, 4)):
            pq, _ = Q.quantize_params(self.params, wb, **Q.SCHEMES["calibrated"])
            self.act_acc[wb, ab] = float((Q.forward_quantized(pq, xte, ab, ranges).argmax(1)
                                          == yte).mean())
        # every stage's quantizer, its rounded weights, and what the sheets show
        self.stages = []
        for bits, scheme, row in STAGES:
            q = Q.weight_quantizer(self.W, bits, **Q.SCHEMES[scheme])
            Wq = q.fake(self.W)
            self.stages.append(dict(bits=bits, scheme=scheme, row=row, q=q, Wq=Wq,
                                    acc=self.table_acc[bits, scheme],
                                    zero=float((q.quantize(self.W) == 0).mean()),
                                    err=float(np.abs(Wq - self.W).max())))
        self.hist_float = histogram(self.W, HBINS, self.hrange)
        centres = (np.arange(HBINS) + 0.5) / HBINS * 2 * self.hrange - self.hrange
        self.hist_cols = div_colors(centres, self.hrange)

        # ---- geometry
        self.Ws = self.sheet((ROWS, COLS), W_AT, SHEET_SCALE, "W")
        self.Es = self.sheet((ROWS, COLS), E_AT, SHEET_SCALE, "error")
        self.Hs = self.sheet((HROWS, HBINS), H_AT, HSCALE, "hist", pitch=HPITCH)
        self.Bs = self.sheet((4, BYTES_COLS), H_AT, HSCALE, "bytes", pitch=BYTES_PITCH)
        self.table = _Table(self, TABLE_X, TABLE_Y, TABLE_STEP)

        # ---- timeline
        Tm, t = {}, 0.0
        Tm["intro"] = (t, t + 6.5)
        t += 6.5
        holds = (3.0, 3.0, 3.0, 2.5, 3.5)
        for stage, hold in zip(self.stages, holds, strict=True):
            stage["t0"] = t
            stage["snap"] = (t + LEVEL_IN, t + LEVEL_IN + SNAP)
            stage["land"] = t + LEVEL_IN + SNAP
            t += LEVEL_IN + SNAP + LAND + hold
        Tm["size"] = (t, t + 8.0)
        t += 8.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, self.stages[0]["t0"], self.stages[1]["t0"], self.stages[2]["t0"],
                      Tm["size"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def stage_at(self, t):
        """``(index of the stage under way or last begun, or -1)``."""
        k = -1
        for i, s in enumerate(self.stages):
            if t >= s["t0"]:
                k = i
        return k

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 0.6, t) * back
        size = self.progress("size", t)
        k = self.stage_at(t)
        self.subtitle = self.words(ph, k, t)
        self._sheets(sc, t, k, intro)
        self._histogram(sc, t, k, intro * (1 - smoothstep(0.0, 0.15, size)))
        self._bytes(sc, size, back)
        self._table(sc, t, k, intro, size)
        self.step_indicator(sc, ph, self.PHASES)

    def _snap(self, t, k):
        """``(previous rounded W or None, current rounded W, per-row progress)``
        of stage ``k``'s snap at time ``t``."""
        if k < 0:
            return None, self.W, np.ones(ROWS)
        s = self.stages[k]
        prev = self.W if k == 0 else self.stages[k - 1]["Wq"]
        return prev, s["Wq"], sweep_rows(ROWS, remap(t, *s["snap"]))

    def _sheets(self, sc, t, k, op):
        """The weights snapping to their levels, and the error sheet."""
        prev, cur, m = self._snap(t, k)
        prev = self.W if prev is None else prev
        shown = prev.T + (cur.T - prev.T) * m[:, None]           # rows are output neurons
        self.Ws.fill(div_colors(shown.ravel(), self.wmax))
        self.Ws.cage = op
        self.Ws.fade(op)
        if k >= 0:
            s = self.stages[k]
            u = remap(t, *s["snap"])
            if 0 < u < 1:
                r = int(np.clip(u * ROWS, 0, ROWS - 1))
                self.Ws.block.white[r, :, 0] = 0.5
                sc.probe(("snap",), (u * 30, 0, 0), tol=6.0)
            err_prev = np.zeros_like(self.W) if k == 0 else self.W - self.stages[k - 1]["Wq"]
            err = err_prev.T + ((self.W - cur).T - err_prev.T) * m[:, None]
            self.Es.fill(div_colors(err.ravel(), self.wmax))
            self.Es.show(np.repeat(m > 0.02, COLS) if k == 0 else np.ones(ROWS * COLS, bool))
            eop = op * (smoothstep(0.0, 0.3, m.max()) if k == 0 else 1.0)
            self.Es.cage = eop
            self.Es.fade(eop)
            what = "%s, %d levels" % (self._name(k), s["q"].levels)
            if s["scheme"] != "per-tensor":
                what += ", %d scales" % s["q"].scale.size
            self.caption(sc, self.Es, "error W %s Q(W), at most %s" % (MINUS, num(s["err"], 3)),
                         eop, size=0.027)
        else:
            self.Es.visible = False
            what = "float32"
        self.caption(sc, self.Ws, "W%s (64 %s 32): %s" % (sub(1), TIMES, what), op, size=0.027)

    def _name(self, k):
        """``int8``, ``int4`` or ``2 bits`` for stage ``k``."""
        b = self.stages[k]["bits"]
        return "2 bits" if b == 2 else "int%d" % b

    def _histogram(self, sc, t, k, op):
        """The histogram of W1's weights, morphing to the rounded values, with the
        quantizer's levels drawn over it."""
        if op <= 0.01:
            self.Hs.visible = False
            return
        prev, cur, m = self._snap(t, k)
        prev = self.W if prev is None else prev
        w = float(m.mean())
        h_prev, h_cur = histogram(prev, HBINS, self.hrange), histogram(cur, HBINS, self.hrange)
        frac = h_prev + (h_cur - h_prev) * w
        full = float(frac.max())
        grey = np.tile(HIST_INK, (HBINS, 1))
        mix = w if k == 0 else 1.0
        cols = grey + (self.hist_cols - grey) * mix
        fill_histogram(self.Hs, frac, full, cols, op)
        self._levels(sc, t, k, op)
        self.caption(sc, self.Hs, "the 2,048 weights of W%s, binned" % sub(1), op, above=True,
                     size=0.027)
        lo, hi = self.Hs.block.box((HROWS - 1, HROWS), (0, HBINS), (0, 1))
        ends = ((lo[0], num(-self.hrange), "rm", -8), (hi[0], "+" + num(self.hrange), "lm", 8))
        for x, text, anchor, dx in ends:
            p = self.cam.project([x, hi[1], (lo[2] + hi[2]) / 2])
            sc.text((p[0] + dx) / self.cam.width, p[1] / self.cam.height, text, layout.MIN_TEXT,
                    "regular", layout.NOTE_INK, op * self._cap, anchor)
        top_lo, top_hi = self.Hs.block.box((0, 1), (0, HBINS), (0, 1))
        p = self.cam.project([top_hi[0], top_hi[1], (top_lo[2] + top_hi[2]) / 2])
        sc.text((p[0] + 8) / self.cam.width, p[1] / self.cam.height,
                "tallest: %.0f%%" % (100 * full), layout.MIN_TEXT, "regular", layout.NOTE_INK,
                op * self._cap, "lm")

    def _levels(self, sc, t, k, op):
        """Vertical lines at the quantizer's levels across the histogram box."""
        if k < 0:
            return
        s = self.stages[k]
        q = s["q"]
        vis = smoothstep(0.0, LEVEL_IN, t - s["t0"]) * op
        if vis <= 0.01:
            return
        if q.per_channel:
            scales = q.scale.ravel()
            levels = np.concatenate([(np.arange(q.qmin, q.qmax + 1)) * sc_ for sc_ in scales])
            alpha = 0.3
        else:
            levels = q.grid()
            alpha = 0.18 if q.levels > 100 else 0.8     # 255 lines would hide the bars
        lo, hi = self.Hs.bbox(offsets=False)
        x0, x1 = lo[0], hi[0]
        front = hi[1]
        for v in levels:
            if abs(v) > self.hrange:
                continue
            x = x0 + (x1 - x0) * (v + self.hrange) / (2 * self.hrange)
            sc.line([x, front, lo[2] - 0.2], [x, front, hi[2] + 0.4], width=1.0, color=GRID_INK,
                    opacity=alpha * vis, bias=0.3)

    def _bytes(self, sc, size, back):
        """Phase 5: the model's bytes at each precision as bars of cells."""
        op = smoothstep(0.15, 0.4, size) * back
        if op <= 0.01:
            self.Bs.visible = False
            return
        names = (("float32", 32), ("int8", 8), ("int4", 4), ("2 bits", 2))
        present = np.zeros((4, BYTES_COLS), bool)
        colors = np.tile(OFF, (4, BYTES_COLS, 1)).astype(np.float32)
        grow = smoothstep(0.1, 0.6, size)
        inks = (layout.NOTE_INK, C.INPUT, C.INPUT, C.INPUT)
        for i, (_name, bits) in enumerate(names):
            n = int(round(BYTES_COLS * self.bytes[bits] / self.bytes[32] * grow))
            n = max(n, 1 if grow > 0.2 else 0)
            present[i, :n] = True
            colors[i, :n] = inks[i]
        self.Bs.fill(colors.reshape(-1, 3))
        self.Bs.show(present.ravel())
        self.Bs.fade(op)
        for i, (name, bits) in enumerate(names):
            lo, hi = self.Bs.block.box((i, i + 1), (0, BYTES_COLS), (0, 1))
            p = self.cam.project([lo[0], hi[1], (lo[2] + hi[2]) / 2])
            sc.text((p[0] - 10) / self.cam.width, p[1] / self.cam.height,
                    "%s  %s" % (name, _kb(self.bytes[bits])), layout.MIN_TEXT, "regular",
                    layout.CAPTION_INK, op * self._cap, "rm")
        self.caption(sc, self.Bs, "bytes on disk: weights packed, biases and scales float32",
                     op, size=0.027, above=True, max_width=0.5)

    def _table(self, sc, t, k, op, size):
        """The accuracy table on the right: a row lands as each stage finishes."""
        tb = self.table
        tb.row(sc, 0, ("", "levels", "accuracy", "bytes"), op, layout.NOTE_INK)
        size_op = smoothstep(0.0, 0.3, size)
        tb.row(sc, 1, ("float32", "—", "%.1f%%" % (100 * self.acc),
                       _kb(self.bytes[32]) if size_op > 0 else None), op,
               layout.STRONG_INK if k < 0 else None)
        for i, s in enumerate(self.stages):
            land = smoothstep(0.0, LAND, t - s["land"]) * op
            if land <= 0.01:
                continue
            first = s["scheme"] == "per-tensor"
            cells = (ROW_NAMES[s["row"]], str(s["q"].levels) if first else None,
                     "%.1f%%" % (100 * s["acc"]),
                     _kb(self.bytes[s["bits"]]) if first and size_op > 0 else None)
            tb.row(sc, s["row"], cells, land, layout.STRONG_INK if i == k and size <= 0 else None)
        if size_op > 0.01:
            y = TABLE_Y + 7.6 * TABLE_STEP
            lines = ("with int8 activations too: %.1f%%" % (100 * self.act_acc[8, 8]),
                     "int4 weights and activations: %.1f%%" % (100 * self.act_acc[4, 4]))
            for n, line in enumerate(lines):
                fx, fy = self.fr(TABLE_X[0], y + n * 0.045)
                sc.text(fx, fy, line, 0.027, "regular", layout.CAPTION_INK,
                        size_op * op * self._cap, "lm")

    def words(self, ph, k, t):
        """The subtitle for the moment."""
        if ph == 0:
            return ("the trained MLP (784-64-32-10, %.1f%% test accuracy): %s float32 weights, "
                    "4 bytes each, %s"
                    % (100 * self.acc, f"{self.n_weights:,}", _kb(self.bytes[32])))
        if ph == 4:
            return ("the weights at 8, 4 and 2 bits take a quarter, an eighth and a sixteenth of "
                    "the space; int8 costs nothing")
        s = self.stages[max(k, 0)]
        step = float(s["q"].scale.ravel().mean())
        if ph == 1:
            return ("int8: %d levels from one scale per matrix, its largest weight (step %s on "
                    "W%s): %.1f%%" % (s["q"].levels, num(step, 4), sub(1), 100 * s["acc"]))
        if ph == 2:
            return ("int4: %d levels, step %s on W%s; the error is at most half a step: %.1f%%, "
                    "%.1f points down"
                    % (s["q"].levels, num(step, 3), sub(1), 100 * s["acc"],
                       100 * (self.acc - s["acc"])))
        if s["scheme"] == "per-tensor":
            return ("2 bits: levels %ss, 0, +s with s the largest weight: %.0f%% of W%s rounds "
                    "to 0, the network falls to chance, %.1f%%"
                    % (MINUS, 100 * s["zero"], sub(1), 100 * s["acc"]))
        if s["scheme"] == "per-channel":
            return ("per-channel: one scale per output neuron, from its own largest weight; "
                    "%.0f%% of W%s is still 0: %.1f%%" % (100 * s["zero"], sub(1), 100 * s["acc"]))
        return ("calibrated: per channel, the clip that minimises the squared error; %.0f%% of "
                "W%s is 0: %.1f%%" % (100 * s["zero"], sub(1), 100 * s["acc"]))


def _kb(n):
    """Bytes as ``211 kB``."""
    return "%d kB" % round(n / 1000)


# ------------------------------------------------------------------ quantization-aware training

QAT_BITS = (2, 4)
QAT_INKS = {2: C.rgb("#d84e4e"), 4: C.rgb("#1f6fb5")}
LAT_AT, QW_AT = (0.17, 0.44), (0.17, 0.74)      # the latent float weights, their rounded value
G_AT = (0.47, 0.44)                             # the gradient dL/dQ(W)
QH_AT = (0.47, 0.74)                            # the latent weights' histogram
QHBINS, QHROWS = 40, 10
RULE_BOX = ((0.66, 0.89), (0.28, 0.50))         # the staircase Q(x)
DERIV_BOX = ((0.66, 0.89), (0.60, 0.78))        # its derivative, true and straight-through
CHART_BOX = ((0.73, 0.945), (0.33, 0.68))       # the training chart
LEGEND_AT = (0.68, 0.80)                        # the chart's legend, under it
RESULT_X, RESULT_Y = 0.68, 0.795                # the result lines, in the legend's place
RULE_INK = (0.30, 0.32, 0.36)
STE_INK = C.rgb("#c0399b")
TRUE_INK = C.INPUT
BOUNDARY_INK = (0.55, 0.57, 0.60)
FIRST_STEPS = 5                                 # steps whose flips the backward phase shows
STEP_SHOW = 1.0                                 # seconds per shown step


class QATraining(NetAnimation):
    """``quantization-aware-training``: fake quantization and the STE (see the module)."""

    PHASES = ("rounding", "forward", "backward", "training", "result")

    def __init__(self, runs=None):
        super().__init__("Quantization-aware training", "", PPU)
        self.hold = 2.5
        self.runs = runs or {b: Q.qat_train(b) for b in QAT_BITS}
        self.run = self.runs[2]
        self.q = self.run.quantizer
        self.s = float(self.run.scales[Q.WATCH])
        self.W0 = self.run.W[0]
        self.wmax = amax(self.run.W)
        self.gmax = amax(self.run.grad_q)
        self.n = self.W0.size
        self.saturated = float((self.q.ste_mask(self.W0) == 0).mean())
        qs = [self.q.quantize(self.run.W[i]) for i in range(FIRST_STEPS + 1)]
        self.flips = [int((qs[i] != qs[i + 1]).sum()) for i in range(FIRST_STEPS)]
        self.steps = int(self.run.steps[-1])
        params, _acc, (_xtr, _ytr, xte, yte) = Q.trained_mlp()
        self.acc_minmax = Q.ptq_accuracies(params, xte, yte, bits=(2,),
                                           schemes={"per-tensor": Q.SCHEMES["per-tensor"]})[
            2, "per-tensor"]
        self.hrange = HIST_PAD * self.wmax
        centres = (np.arange(QHBINS) + 0.5) / QHBINS * 2 * self.hrange - self.hrange
        self.hist_cols = div_colors(centres, self.hrange)
        self.chart = LossChart(CHART_BOX, (0, self.steps), (0.7, 1.0))

        # ---- geometry
        self.Ls = self.sheet((ROWS, COLS), LAT_AT, SHEET_SCALE, "W")
        self.Qs = self.sheet((ROWS, COLS), QW_AT, SHEET_SCALE, "Q(W)")
        self.Gs = self.sheet((ROWS, COLS), G_AT, SHEET_SCALE, "grad")
        self.Hs = self.sheet((QHROWS, QHBINS), QH_AT, HSCALE, "hist", pitch=HPITCH)

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("rule", 8.0)
        add("fwd", 7.0)
        add("grad", 2.5, 0.0)                    # the gradient appears
        add("through", 2.5, 0.0)                 # ... and passes straight through
        add("steps", FIRST_STEPS * STEP_SHOW + 1.5, 0.6)
        add("train", 13.0, 0.3)
        add("result", 5.0, 0.0)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["grad"][0], Tm["train"][0], Tm["result"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def near(self, W):
        """``(near a level, near a boundary)``: fractions of ``W`` within
        :data:`NEAR` steps of a level, and of a decision boundary between two."""
        u = W / self.s
        f = np.abs(u - np.round(u))
        return float((f < NEAR).mean()), float((f > 0.5 - NEAR).mean())

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rule = self.progress("rule", t)
        fwd = self.progress("fwd", t)
        train = self.progress("train", t)
        result = self.progress("result", t)
        self.subtitle = self.words(ph, t, train)
        i = self._snapshot(t, train)
        sheets = smoothstep(0.0, 0.5, fwd) * back
        self._rule(sc, rule, (1 - smoothstep(0.0, 0.08, train)) * back)
        self._weights(sc, t, fwd, i, sheets)
        self._gradient(sc, t, i, back)
        self._histogram(sc, i, smoothstep(0.15, 0.35, train) * back)
        self._chart(sc, train, result, back)
        self._result(sc, result, back)
        self.step_indicator(sc, ph, self.PHASES)

    def _snapshot(self, t, train):
        """Index of the recorded snapshot the sheets show at time ``t``."""
        if train > 0:
            return self.run.snapshot(self.steps * smoothstep(0.0, 0.92, train))
        if t < self.Tm["steps"][0]:
            return 0
        k = int(min(FIRST_STEPS, (t - self.Tm["steps"][0]) // STEP_SHOW + 1))
        return k

    def _step_u(self, t):
        """``(step shown, progress through it)`` during the first-steps phase."""
        if not self.Tm["steps"][0] <= t < self.Tm["steps"][1]:
            return 0, 0.0
        x = (t - self.Tm["steps"][0]) / STEP_SHOW
        return int(min(FIRST_STEPS, int(x) + 1)), x - int(x)

    # -- phase 1: the rounding function -----------------------------------------
    def _plot(self, box, xr, yr):
        """``place(x, y)`` -> world point and ``text(x, y)`` -> frame fraction for a
        data box ``xr`` x ``yr`` drawn in the layout ``box``."""
        (x0, x1), (y0, y1) = box
        (xa, xb), (ya, yb) = xr, yr

        def fx(x):
            return x0 + (x1 - x0) * (x - xa) / (xb - xa)

        def fy(y):
            return y1 - (y1 - y0) * (y - ya) / (yb - ya)

        return (lambda x, y: self.cam.frac(fx(x), fy(y))), (lambda x, y: self.fr(fx(x), fy(y)))

    def _rule(self, sc, rule, op):
        """``Q(x)`` as a staircase, then its derivative: 0 almost everywhere against
        the straight-through 1."""
        op *= smoothstep(0.0, 0.08, rule) * self._cap
        if op <= 0.01:
            return
        s = self.s
        place, text = self._plot(RULE_BOX, (-2.2 * s, 2.2 * s), (-1.4 * s, 1.4 * s))
        self._axes(sc, place, (-2.2 * s, 2.2 * s), (-1.4 * s, 1.4 * s), op)
        draw = smoothstep(0.1, 0.5, rule)
        # the staircase: three flats and two jumps, drawn left to right
        pieces = [((-2.2 * s, -s), (-0.5 * s, -s)), ((-0.5 * s, -s), (-0.5 * s, 0.0)),
                  ((-0.5 * s, 0.0), (0.5 * s, 0.0)), ((0.5 * s, 0.0), (0.5 * s, s)),
                  ((0.5 * s, s), (2.2 * s, s))]
        for n, (a, b) in enumerate(pieces):
            u = remap(draw, n / len(pieces), (n + 1) / len(pieces))
            if u <= 0:
                break
            jump = a[0] == b[0]
            end = (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)
            self._segment(sc, place, a, end, STE_INK if False else RULE_INK, op,
                          dashed=jump)
        for lv, label in ((-s, MINUS + "s"), (0.0, "0"), (s, "+s")):
            x, y = text(2.25 * s, lv)
            sc.text(x, y, label, layout.MIN_TEXT, "regular", layout.NOTE_INK, op * draw, "lm")
        x, y = text(-2.2 * s, 1.4 * s)
        sc.text(x, y - 0.012, "Q(x) = s·round(x/s), clipped   (s = %s)" % num(s, 3),
                0.027, "regular", layout.CAPTION_INK, op, "lb")
        # the derivative panel
        dop = op * smoothstep(0.5, 0.7, rule)
        if dop <= 0.01:
            return
        place2, text2 = self._plot(DERIV_BOX, (-2.2 * s, 2.2 * s), (-0.3, 1.3))
        self._axes(sc, place2, (-2.2 * s, 2.2 * s), (-0.3, 1.3), dop)
        # true derivative: 0, with the jumps marked
        self._segment(sc, place2, (-2.2 * s, 0.0), (2.2 * s, 0.0), TRUE_INK, dop)
        for xj in (-0.5 * s, 0.5 * s):
            self._segment(sc, place2, (xj, 0.0), (xj, 1.3), TRUE_INK, dop, dashed=True)
        ste = smoothstep(0.75, 0.95, rule)
        if ste > 0:
            for a, b in (((-2.2 * s, 0.0), (-1.5 * s, 0.0)), ((-1.5 * s, 1.0), (1.5 * s, 1.0)),
                         ((1.5 * s, 0.0), (2.2 * s, 0.0))):
                self._segment(sc, place2, a, b, STE_INK, dop * ste, width=2.0)
        x, y = text2(-2.2 * s, 1.3)
        sc.text(x, y - 0.012, "∂Q/∂x", 0.027, "regular", layout.CAPTION_INK, dop, "lb")
        x, y = text2(2.25 * s, 0.0)
        sc.text(x, y, "0 (true)", layout.MIN_TEXT, "regular", _ink(TRUE_INK), dop, "lm")
        x, y = text2(2.25 * s, 1.0)
        sc.text(x, y, "1 (STE)", layout.MIN_TEXT, "regular", _ink(STE_INK), dop * ste, "lm")

    def _axes(self, sc, place, xr, yr, op):
        """A light x axis through 0 and a y axis at the left of a plot."""
        sc.line(place(xr[0], 0.0), place(xr[1], 0.0), width=0.9, color=(0.5, 0.52, 0.54),
                opacity=0.6 * op, bias=0.0)
        sc.line(place(xr[0], yr[0]), place(xr[0], yr[1]), width=0.9, color=(0.5, 0.52, 0.54),
                opacity=0.6 * op, bias=0.0)

    def _segment(self, sc, place, a, b, ink, op, width=1.5, dashed=False, dashes=8):
        """A data-space segment, optionally dashed."""
        p, q = place(*a), place(*b)
        if not dashed:
            sc.line(p, q, width=width, color=ink, opacity=0.9 * op, bias=0.05)
            return
        for i in range(dashes):
            u0, u1 = i / dashes, (i + 0.5) / dashes
            sc.line(p + (q - p) * u0, p + (q - p) * u1, width=width, color=ink, opacity=0.9 * op,
                    bias=0.05)

    # -- phases 2-4: the sheets ---------------------------------------------------
    def _weights(self, sc, t, fwd, i, op):
        """The latent float weights (top) and their rounded value (bottom), following
        snapshot ``i``; the first-steps phase glints the cells that changed level."""
        if op <= 0.01:
            self.Ls.visible = self.Qs.visible = False
            return
        W = self.run.W[i]
        self.Ls.fill(div_colors(W.T.ravel(), self.wmax))
        self.Ls.cage = op
        self.Ls.fade(op)
        # the rounded sheet snaps row by row during the forward phase
        Wq = self.q.fake(W)
        m = sweep_rows(ROWS, remap(fwd, 0.15, 0.7))
        shown = W.T + (Wq.T - W.T) * m[:, None]
        self.Qs.fill(div_colors(shown.ravel(), self.wmax))
        self.Qs.cage = op
        self.Qs.fade(op)
        through = self.progress("through", t)
        if 0 < through < 1:
            # the weights that saturate get no gradient: they go dark for a moment
            dark = (self.q.ste_mask(W) == 0).T.ravel()
            cols = self.Ls.block.base[:, :, 0].reshape(-1, 3)
            k = smoothstep(0.3, 0.6, through) * (1 - smoothstep(0.85, 1.0, through))
            cols[dark] = cols[dark] + (OFF - cols[dark]) * k
            self.Ls.block.base[:, :, 0] = cols.reshape(ROWS, COLS, 3)
        step, u = self._step_u(t)
        if step > 0 and i >= 1:
            flipped = (self.q.quantize(self.run.W[i]) != self.q.quantize(self.run.W[i - 1])).T
            glint = 0.9 * (1 - smoothstep(0.3, 1.0, u))
            self.Qs.block.white[:, :, 0] = np.maximum(self.Qs.block.white[:, :, 0],
                                                      flipped * glint)
        self.caption(sc, self.Ls, "W%s: float weights (latent)" % sub(1), op, above=True,
                     size=0.027)
        if fwd < 0.7:
            what = "Q(W%s): rounding to %ss, 0, +s" % (sub(1), MINUS)
        elif step > 0:
            what = "Q(W%s), step %d: %d of %s flipped" % (sub(1), step, self.flips[step - 1],
                                                          f"{self.n:,}")
        elif i > 0:
            what = "Q(W%s) at step %d" % (sub(1), int(self.run.steps[i]))
        else:
            what = "Q(W%s): the network computes with these" % sub(1)
        self.caption(sc, self.Qs, what, op, size=0.027, max_width=0.34)

    def _gradient(self, sc, t, i, back):
        """``dL/dQ(W)`` on the first minibatch, then flying straight through to W."""
        grad = self.progress("grad", t)
        through = self.progress("through", t)
        train = self.progress("train", t)
        op = smoothstep(0.0, 0.5, grad) * (1 - smoothstep(0.0, 0.12, train)) * back
        if op <= 0.01:
            self.Gs.visible = False
            return
        g = self.run.grad_q
        self.Gs.fill(div_colors(g.T.ravel(), self.gmax))
        self.Gs.cage = op
        self.Gs.fade(op)
        self.caption(sc, self.Gs, "∂L/∂Q(W%s), a minibatch of %d" % (sub(1), Q.QAT_BATCH), op,
                     above=True, size=0.027)
        if through > 0:
            # a handful of values fly from the gradient sheet to the float weights
            K = 24
            idx = np.argsort(-np.abs(g).ravel())[:K]
            for n, flat in enumerate(idx):
                r, c = divmod(int(flat), g.shape[1])
                u = remap(through, 0.1 + 0.4 * n / K, 0.4 + 0.4 * n / K)
                col = C.POSITIVE if g[r, c] > 0 else C.NEGATIVE
                particle(sc, self.Gs.cell(c * COLS + r), self.Ls.cell(c * COLS + r), u, col,
                         0.6 * SHEET_SCALE, lift=1.5)
            if 0 < through < 1:
                sc.probe(("through",), (through * 30, 0, 0), tol=6.0)
            text = ("straight through: ∂L/∂W := ∂L/∂Q(W)\n%.0f%% saturate at ±s and get 0"
                    % (100 * self.saturated))
            self.caption(sc, self.Gs, text, op * smoothstep(0.1, 0.4, through), size=0.027,
                         max_width=0.30)
        else:
            self.caption(sc, self.Gs, "the float network's gradient\nat the rounded weights", op,
                         size=0.027, max_width=0.30)

    def _histogram(self, sc, i, op):
        """The latent weights binned, with the levels (solid) and the boundaries
        between them (dashed)."""
        if op <= 0.01:
            self.Hs.visible = False
            return
        W = self.run.W[i]
        frac = histogram(W, QHBINS, self.hrange)
        fill_histogram(self.Hs, frac, float(frac.max()), self.hist_cols, op)
        lo, hi = self.Hs.bbox(offsets=False)
        x0, x1 = lo[0], hi[0]
        front = hi[1]
        for v, dashed in ((-self.s, False), (0.0, False), (self.s, False),
                          (-0.5 * self.s, True), (0.5 * self.s, True)):
            x = x0 + (x1 - x0) * (v + self.hrange) / (2 * self.hrange)
            a, b = np.array([x, front, lo[2] - 0.2]), np.array([x, front, hi[2] + 0.4])
            if not dashed:
                sc.line(a, b, width=1.1, color=GRID_INK, opacity=0.85 * op, bias=0.3)
            else:
                for n in range(6):
                    sc.line(a + (b - a) * n / 6, a + (b - a) * (n + 0.5) / 6, width=1.0,
                            color=BOUNDARY_INK, opacity=0.85 * op, bias=0.3)
        level, boundary = self.near(W)
        self.caption(sc, self.Hs, "the float weights, binned", op, above=True, size=0.027)
        self.caption(sc, self.Hs, "levels solid, boundaries dashed; within 0.1 step of a level "
                     "%.0f%%, of a boundary %.0f%%" % (100 * level, 100 * boundary), op,
                     size=0.027, max_width=0.36)

    def _chart(self, sc, train, result, back):
        """Test accuracy of the quantized network against the step, 2 and 4 bits."""
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        upto = self.steps * smoothstep(0.0, 0.92, train)
        self.chart.axes(sc, cam, place, op, "step", "test accuracy",
                        [(s, f"{s:,}") for s in (0, 300, 600, 900)],
                        [(v, "%d%%" % round(100 * v)) for v in (0.7, 0.8, 0.9, 1.0)])
        self.chart.hline(sc, place, self.run.acc_float, op)
        for bits in QAT_BITS:
            run = self.runs[bits]
            self.chart.curve(sc, place, run.steps, run.acc, upto, QAT_INKS[bits], op)
        fx, fy = self.fr(CHART_BOX[0][1], self.chart.fy(self.run.acc_float) - 0.012)
        sc.text(fx, fy, "float32: %.1f%%" % (100 * self.run.acc_float), layout.MIN_TEXT,
                "regular", layout.NOTE_INK, op, "rb")
        legend = [("%s: %.1f%% → %.1f%%" % (_bits_name(b), 100 * self.runs[b].acc[0],
                                            100 * self.runs[b].acc[-1]), QAT_INKS[b])
                  for b in QAT_BITS]
        self.chart.legend(sc, cam, [(s, _ink(c)) for s, c in legend],
                          op * smoothstep(0.85, 1.0, train) * (1 - smoothstep(0.0, 0.2, result)),
                          LEGEND_AT)
        sc.probe(("train",), (upto * 0.05, 0, 0), tol=6.0)

    def _result(self, sc, result, back):
        """Post-training against quantization-aware, in words, under the chart."""
        op = smoothstep(0.25, 0.5, result) * back * self._cap
        if op <= 0.01:
            return
        r2, r4 = self.runs[2], self.runs[4]
        lines = (("post-training %.1f%% (min-max %.1f%%)"
                  % (100 * r2.acc[0], 100 * self.acc_minmax)),
                 ("quantization-aware %.1f%%" % (100 * r2.acc[-1])),
                 ("int4 %.1f%% → %.1f%%, float32 %.1f%%"
                  % (100 * r4.acc[0], 100 * r4.acc[-1], 100 * r2.acc_float)))
        for n, line in enumerate(lines):
            fx, fy = self.fr(RESULT_X, RESULT_Y + 0.9 * 0.049 * n)
            sc.text(fx, fy, line, 0.027, "regular",
                    layout.STRONG_INK if n == 1 else layout.CAPTION_INK, op, "lm")

    def words(self, ph, t, train):
        """The subtitle for the moment."""
        r2, r4 = self.runs[2], self.runs[4]
        if ph == 0:
            return ("rounding to a grid has derivative 0 almost everywhere, so no gradient "
                    "passes through it; the straight-through estimator pretends it is 1")
        if ph == 1:
            return ("forward: the network computes with Q(W), the weights rounded to 3 levels "
                    "(s = %s for W%s, calibrated): %.1f%% on the test set"
                    % (num(self.s, 3), sub(1), 100 * r2.acc[0]))
        if ph == 2:
            step, _u = self._step_u(t)
            if step == 0:
                return ("backward: ∂L/∂W := ∂L/∂Q(W) where |W| < 1.5 s (0 where it saturates); "
                        "then an Adam step, lr %g" % Q.QAT_LR)
            return ("a weight that crosses ±s/2 changes level: %s of the %s flipped in step%s %s"
                    % (", ".join(str(f) for f in self.flips[:step]), f"{self.n:,}",
                       "s 1 to %d" % step if step > 1 else " 1", ""))
        if ph == 3:
            if train < 0.92:
                k = int(self.steps * smoothstep(0.0, 0.92, train))
                return ("step %d of %d: the quantized accuracy climbs; the float weights spread "
                        "rather than gather at the levels" % (k, self.steps))
            level, boundary = self.near(self.run.W[-1])
            return ("%d steps: 2 bits %.1f → %.1f%%, int4 %.1f → %.1f%%; %.0f%% of the float "
                    "weights sit at a boundary, %.0f%% at a level"
                    % (self.steps, 100 * r2.acc[0], 100 * r2.acc[-1], 100 * r4.acc[0],
                       100 * r4.acc[-1], 100 * boundary, 100 * level))
        lost = r2.acc_float - r2.acc[0]
        won = r2.acc[-1] - r2.acc[0]
        return ("post-training lost %.1f points at 2 bits; training through the rounding wins "
                "%.1f back: %.1f%% against %.1f%% in float32"
                % (100 * lost, 100 * won, 100 * r2.acc[-1], 100 * r2.acc_float))


def _bits_name(b):
    """``2 bits`` or ``int4``."""
    return "2 bits" if b == 2 else "int%d" % b


_ = (lit_colors, pow10)      # imported for the precision animation's siblings; keep ruff quiet
