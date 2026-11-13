"""float32, float16, bfloat16: what fewer bits per number keep and lose.

``mixed-precision``, in five phases, every number from :mod:`voxanim.quant`:

1. **formats** -- the three bit layouts as rows of voxels, sign | exponent |
   mantissa (1|8|23, 1|5|10, 1|8|7), each holding the same real number, a
   trained weight of the MNIST MLP (``W1[0, 0] = 0.4254688``): float16
   stores 0.4255, bfloat16 0.4258;
2. **grid** -- the representable numbers within 1% of 1, 10⁻³ and 10⁻⁵ as
   ticks on a line: float16's step is the value / 1024 and bfloat16's the
   value / 128 as long as the number is normal; below 6×10⁻⁵ float16's
   step freezes at 6×10⁻⁸ (subnormals) and the ticks thin out, while
   bfloat16, with float32's exponent, keeps its 1/128;
3. **underflow** -- a real weight gradient: layer 11 of a 20-layer tanh MLP
   with weights of std 0.05, on 64 images, 100 × 100 values of RMS
   2.7×10⁻⁸.  Cast to float16, 79% of them become exactly 0 and the rest
   are subnormal with one to three significant bits; cast to bfloat16,
   all survive;
4. **loss scaling** -- the loss times 2¹⁶ makes every gradient 2¹⁶ larger
   before the cast, where float16 holds 96% of them as normal numbers and
   loses none; dividing by 2¹⁶ afterwards, in float32, gives the gradient
   back (median relative error 2×10⁻⁴);
5. **training** -- the MLP trained for two epochs with every tensor in
   float32, float16 and bfloat16 (sums accumulated in float32, as tensor
   cores do), and in float16 with float32 master weights and optimiser
   state.  Float16 alone diverges at its first update: Adam's ε = 10⁻⁸ is
   0 in float16, and so are the small squared gradients it divides by.
   bfloat16 alone trains (95.1% against float32's 95.6%); float16 with
   master weights matches float32.

bfloat16 is emulated by rounding float32 to a 7-bit mantissa (numpy has
no such type); the gradient and the runs are cached, seeded computations.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from . import quant as Q
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10, sup
from .mlp_runs import smooth
from .netanim import NetAnimation, div_colors
from .quant_anim import _ink, sweep_rows
from .text import TIMES

PPU = 14.0
BIT_SCALE, BIT_PITCH = 1.0, (1.0, 1.15)
BITS_AT = (0.45, 0.40)                          # the float32 row; the others stack under it
ROW_GAP = 6.0                                   # world units between bit rows
FIELD_INK = {"sign": C.rgb("#d84e4e"), "exponent": C.rgb("#e0952e"), "mantissa": C.INPUT}
ZERO_DIM = 0.3                                  # a 0 bit: its field's colour, this much of it
MAGNITUDES = (1.0, 1e-3, 1e-5)                  # where the grid is drawn
WINDOW = 0.01                                   # ... within this fraction of the value
GRID_X = (0.28, 0.66)                           # the number lines, layout x
GRID_Y = (0.34, 0.57, 0.80)                     # one group per magnitude
GRID_DY = 0.044                                 # between the three formats of a group
TICK_PX = 5.0
FORMAT_INK = {"float32": (0.36, 0.38, 0.42), "float16": C.rgb("#d84e4e"),
              "bfloat16": C.rgb("#1f6fb5")}
G_AT, G_SCALE = (0.25, 0.61), 0.18              # the gradient sheet
LOG_LO = np.log10(5.96e-8)                      # the gradient's brightness: log10 |g| from
LOG_DECADES = 6.0                               # float16's floor, over this many decades
TEXT_X = 0.53                                   # the read-out column of phases 3 and 4
TEXT_Y = 0.30
CHART_BOX = ((0.10, 0.62), (0.30, 0.82))        # the training chart
LEGEND_AT = (0.66, 0.42)
SMOOTH = 20                                     # batches the drawn loss is averaged over
RUN_INKS = ((0.36, 0.38, 0.42), C.rgb("#d84e4e"), C.rgb("#1f6fb5"), C.rgb("#3d9e45"))
RUN_WIDTHS = (2.4, 1.5, 1.5, 1.0)              # the curves lie on each other: thick to thin


def mag(v):
    """A magnitude as ``2.4×10⁻⁷``, or plainly (``0.0155``) when it is not tiny."""
    return pow10(v, 1) if abs(v) < 0.01 else "%.3g" % v


class MixedPrecision(NetAnimation):
    """``mixed-precision``: the three formats on real numbers (see the module)."""

    PHASES = ("formats", "grid", "underflow", "loss scaling", "training")

    def __init__(self, runs=None):
        super().__init__("float32, float16, bfloat16", "", PPU)
        self.hold = 2.5
        params, _acc, _data = Q.trained_mlp()
        self.value = float(np.float32(params[Q.WATCH][0][0, 0]))
        self.stored = {f: float(Q.round_to(self.value, f)) for f in Q.FORMATS}
        self.fields = {f: Q.bit_fields(self.value, f) for f in Q.FORMATS}
        u = Q.underflow_gradient()
        self.g, self.g_rms = u["g"], u["rms"]
        self.g_max = float(np.abs(self.g).max())
        self.g16 = Q.round_to(self.g, "float16").astype(np.float64)
        self.zero16 = (self.g16 == 0) & (self.g != 0)
        self.n_zero = int(self.zero16.sum())
        self.n_sub = int(((np.abs(self.g16) > 0)
                          & (np.abs(self.g16) < Q.format_info("float16")["tiny"])).sum())
        kept = ~self.zero16 & (self.g != 0)
        self.err16 = float(np.median(np.abs(self.g16 - self.g)[kept] / np.abs(self.g[kept])))
        scaled = Q.loss_scaled(self.g, Q.LOSS_SCALE, "float16")
        nz = self.g != 0
        self.err_scaled = float(np.median(np.abs(scaled - self.g)[nz] / np.abs(self.g[nz])))
        self.lost_scaled = int(((scaled == 0) & nz).sum())
        self.sub_scaled = Q.classify(self.g * Q.LOSS_SCALE, "float16")[1]
        self.runs = runs or Q.precision_runs()
        self.curves = [smooth(r.batch_loss, SMOOTH) for r in self.runs]
        self.steps = self.runs[0].steps
        self.chart = LossChart(CHART_BOX, (0, self.steps), (0.0, 2.5))

        # ---- geometry: the bit rows, left-aligned under each other
        self.rows = {}
        first = None
        for n, f in enumerate(Q.FORMATS):
            s = self.sheet((1, Q.format_info(f)["bits"]), BITS_AT, BIT_SCALE, f, pitch=BIT_PITCH)
            if first is None:
                first = s.block.origin.copy()
            s.block.origin = first + np.array([0.0, 0.0, -ROW_GAP * n])
            self.rows[f] = s
        self.Gs = self.sheet(self.g.shape, G_AT, G_SCALE, "grad")

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("formats", 9.0)
        add("grid", 9.0)
        add("under", 10.0)
        add("scale", 9.0)
        add("train", 14.0)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["grid"][0], Tm["under"][0], Tm["scale"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        formats = self.progress("formats", t)
        grid = self.progress("grid", t)
        under = self.progress("under", t)
        scale = self.progress("scale", t)
        train = self.progress("train", t)
        self.subtitle = self.words(ph, under, scale)
        bits_op = smoothstep(0.0, 0.5, formats) * (1 - smoothstep(0.0, 0.15, grid)) * back
        self._bits(sc, formats, bits_op)
        self._grid(sc, grid, (1 - smoothstep(0.0, 0.12, under)) * back)
        sheet_op = smoothstep(0.0, 0.12, under) * (1 - smoothstep(0.0, 0.12, train)) * back
        self._gradient(sc, under, scale, sheet_op)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- phase 1: the bit layouts --------------------------------------------------
    def _bits(self, sc, formats, op):
        """The three rows of bits holding :attr:`value`, lit where the bit is 1."""
        if op <= 0.01:
            for s in self.rows.values():
                s.visible = False
            return
        for n, f in enumerate(Q.FORMATS):
            s = self.rows[f]
            sign, exp, man = self.fields[f]
            bits = sign + exp + man
            kinds = ["sign"] * len(sign) + ["exponent"] * len(exp) + ["mantissa"] * len(man)
            cols = np.array([FIELD_INK[k] for k in kinds], np.float32)
            ones = np.array([b == "1" for b in bits])
            # the row lights up left to right, rows one after another
            u = remap(formats, 0.08 + 0.22 * n, 0.3 + 0.22 * n)
            lit = np.array([smoothstep(k / len(bits) - 0.1, k / len(bits) + 0.1, u)
                            for k in range(len(bits))])
            dim = cols * ZERO_DIM
            full = cols
            shown = dim + (full - dim) * (ones * lit)[:, None]
            s.fill(shown, glow=0.25 * ones * lit)
            s.cage = op
            s.fade(op)
            info = Q.format_info(f)
            self.caption_left(sc, s, "%s  (1 | %d | %d)" % (f, info["exponent"], info["mantissa"]),
                              op, size=0.028)
            shown_op = op * smoothstep(0.9, 1.0, u)
            err = self.stored[f] - self.value
            text = "= %.7f" % self.stored[f]
            if abs(err) > 0:
                text += "   (off by %s)" % pow10(err, 1)
            self.caption_side(sc, s, text, shown_op, size=0.028)
            if n == 0:                      # the field names, above the first row
                e, m = len(exp), len(man)
                for j0, j1, label in ((0, 1, "sign"), (1, 1 + e, "exponent"),
                                      (1 + e, 1 + e + m, "mantissa")):
                    self.caption(sc, s.block.box((0, 1), (j0, j1), (0, 1)), label, op,
                                 size=0.027, above=True)

    # -- phase 2: the representable grid -------------------------------------------
    def _grid(self, sc, grid, op):
        """Ticks at the representable numbers of each format around each magnitude."""
        op *= smoothstep(0.0, 0.15, grid) * self._cap
        if op <= 0.01:
            return
        x0, x1 = GRID_X
        cam = self.cam
        for n, (v, gy) in enumerate(zip(MAGNITUDES, GRID_Y, strict=True)):
            gop = op * smoothstep(0.1 + 0.25 * n, 0.3 + 0.25 * n, grid)
            if gop <= 0.01:
                continue
            lo, hi = v * (1 - WINDOW), v * (1 + WINDOW)
            fx, fy = self.fr(x0, gy - GRID_DY - 0.045)
            sc.text(fx, fy, "within 1%% of %s" % pow10(v), 0.028, "regular", layout.CAPTION_INK,
                    gop, "lm")
            for k, f in enumerate(Q.FORMATS):
                y = gy + (k - 1) * GRID_DY
                a, b = cam.frac(x0, y), cam.frac(x1, y)
                ink = FORMAT_INK[f]
                pts = Q.representable(lo, hi, f)
                if pts is None:
                    # too fine to draw: a solid bar stands for the grid
                    sc.line(a, b, width=3.0, color=ink, opacity=0.8 * gop, bias=0.0)
                    count = int(round(2 * WINDOW * v / Q.ulp(v, f)))
                    note = "step %s, %s values" % (pow10(Q.ulp(v, f), 1), f"{count:,}")
                else:
                    sc.line(a, b, width=1.0, color=ink, opacity=0.6 * gop, bias=0.0)
                    up = cam.frac(x0, y - 0.012) - a
                    for p in pts:
                        c = a + (b - a) * (p - lo) / (hi - lo)
                        sc.line(c - up * 0.6, c + up, width=1.6, color=ink, opacity=gop, bias=0.0)
                    note = "step %s, %d values" % (pow10(Q.ulp(v, f), 1), len(pts))
                    if v < Q.format_info(f)["tiny"]:
                        note = "step %s, %d (subnormal)" % (pow10(Q.ulp(v, f), 1), len(pts))
                fx, fy = self.fr(x0 - 0.012, y)
                sc.text(fx, fy, f, 0.028, "regular", _ink(ink), gop, "rm")
                fx, fy = self.fr(x1 + 0.012, y)
                sc.text(fx, fy, note, 0.028, "regular", layout.NOTE_INK, gop, "lm")

    # -- phases 3 and 4: the gradient --------------------------------------------
    def _colours(self, g, present):
        """Sheet colours of gradient values ``g``: orange / blue by sign, brightness
        the log of |g| above float16's smallest subnormal; absent cells OFF."""
        with np.errstate(divide="ignore"):
            m = (np.log10(np.abs(g)) - LOG_LO) / LOG_DECADES
        m = np.clip(m, 0.08, 1.0) * np.sign(g)
        return div_colors(m.ravel(), 1.0, mask=(present & (g != 0)).ravel())

    def _gradient(self, sc, under, scale, op):
        """The 100 x 100 gradient: float32, cast to float16 (cells vanish), back in
        bfloat16, then scaled, cast and unscaled."""
        if op <= 0.01:
            self.Gs.visible = False
            return
        g = self.g
        rows = g.shape[0]
        cast = sweep_rows(rows, remap(under, 0.25, 0.55))          # the float16 cast sweeps down
        relight = smoothstep(0.75, 0.9, under)                     # bfloat16 keeps them all
        present = ~(self.zero16 & (cast[:, None] > 0.5) & (relight < 0.5))
        shown = g
        mult = smoothstep(0.05, 0.3, scale)                        # x 2^16 ...
        div = smoothstep(0.6, 0.85, scale)                         # ... and back
        factor = Q.LOSS_SCALE ** (mult - div)
        shown = g * factor
        self.Gs.fill(self._colours(shown, present))
        self.Gs.cage = op
        self.Gs.fade(op)
        u = remap(under, 0.25, 0.55)
        if 0 < u < 1:
            r = int(np.clip(u * rows, 0, rows - 1))
            self.Gs.block.white[r, :, 0] = 0.5
            sc.probe(("cast",), (u * 30, 0, 0), tol=6.0)
        self._gradient_text(sc, under, scale, op)
        self.caption(sc, self.Gs, "colour: sign; brightness: log|g| above %s"
                     % pow10(10 ** LOG_LO, 1), op, size=0.027, max_width=0.40)
        self.caption(sc, self.Gs, "∂L/∂W of layer %d: 100 %s 100 values"
                     % (Q.DEEP_LAYER + 1, TIMES), op, above=True, size=0.027)

    def _gradient_text(self, sc, under, scale, op):
        """The read-outs beside the sheet, in the order the story needs them."""
        f16 = Q.format_info("float16")
        n = self.g.size
        lines = [
            ("float32: RMS %s, largest %s" % (mag(self.g_rms), mag(self.g_max)),
             layout.CAPTION_INK, smoothstep(0.05, 0.2, under)),
            ("float16: normal from %s, subnormal from %s"
             % (pow10(f16["tiny"], 1), pow10(f16["smallest_subnormal"], 1)),
             layout.CAPTION_INK, smoothstep(0.15, 0.25, under)),
            ("cast to float16: %s of %s become 0" % (f"{self.n_zero:,}", f"{n:,}"),
             layout.STRONG_INK, smoothstep(0.5, 0.6, under)),
            ("the other %s are subnormal: median error %.0f%%"
             % (f"{self.n_sub:,}", 100 * self.err16), layout.CAPTION_INK,
             smoothstep(0.55, 0.65, under)),
            ("bfloat16 keeps float32's exponent: all %s survive" % f"{n:,}", layout.CAPTION_INK,
             smoothstep(0.75, 0.85, under)),
        ]
        if scale > 0:
            lines = [
                ("loss %s 2%s: every gradient %s %s" % (TIMES, sup(16), TIMES,
                                                       f"{int(Q.LOSS_SCALE):,}"),
                 layout.STRONG_INK, smoothstep(0.0, 0.1, scale)),
                ("RMS %s, largest %s: %.0f%% normal in float16"
                 % (mag(self.g_rms * Q.LOSS_SCALE), mag(self.g_max * Q.LOSS_SCALE),
                    100 * (1 - self.sub_scaled)),
                 layout.CAPTION_INK, smoothstep(0.25, 0.35, scale)),
                ("cast to float16: %d lost" % self.lost_scaled, layout.CAPTION_INK,
                 smoothstep(0.4, 0.5, scale)),
                ("÷ 2%s in float32: the gradient back, median error %s"
                 % (sup(16), pow10(self.err_scaled, 1)), layout.STRONG_INK,
                 smoothstep(0.6, 0.7, scale)),
                ("unscaled, the %s survivors had %.0f%%" % (f"{self.n_sub:,}", 100 * self.err16),
                 layout.NOTE_INK, smoothstep(0.75, 0.85, scale)),
            ]
        for k, (text, ink, vis) in enumerate(lines):
            o = op * vis * self._cap
            if o <= 0.01:
                continue
            fx, fy = self.fr(TEXT_X, TEXT_Y + 0.085 * k)
            sc.text(fx, fy, text, 0.028, "regular", ink, o, "lm")

    # -- phase 5: training -------------------------------------------------------------
    def _chart(self, sc, train, back):
        """The four runs' smoothed batch loss against the step."""
        op = smoothstep(0.1, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        upto = self.steps * smoothstep(0.0, 0.92, train)
        self.chart.axes(sc, cam, place, op, "step (batch loss, mean of %d)" % SMOOTH, "loss",
                        [(s, f"{s:,}") for s in (0, 300, 600, 900)],
                        [(v, "%g" % v) for v in (0, 1, 2)])
        steps = np.arange(1, self.steps + 1)
        legend = []
        for (label, _fmt, _master), run, curve, ink, width in zip(
                Q.PRECISION_RUNS, self.runs, self.curves, RUN_INKS, RUN_WIDTHS, strict=True):
            d = run.diverged
            if d >= 0:
                # drawn up to its last finite loss; the divergence marked where it happens
                self.chart.curve(sc, place, steps[:d], curve[:d], upto, ink, op, width=width)
                if upto >= d:
                    p = place(self.chart.fx(d + 1), self.chart.fy(self.chart.y_range[1]))
                    sc.voxel(p, ink, op, glow=0.4, shadow=False, reflect=False, size=0.7)
                    fx, fy = self.fr(self.chart.fx(d + 1) + 0.012, self.chart.fy(2.5) + 0.01)
                    sc.text(fx, fy, "float16: ∞ at step %d" % (d + 1), layout.MIN_TEXT, "regular",
                            _ink(ink), op, "lt")
                legend.append(("%s: diverges at step %d" % (label, d + 1), _ink(ink)))
            else:
                self.chart.curve(sc, place, steps, curve, upto, ink, op, width=width)
                legend.append(("%s: %.1f%%" % (label, 100 * run.acc), _ink(ink)))
        self.chart.legend(sc, cam, legend, op * smoothstep(0.8, 1.0, train), LEGEND_AT, size=0.028)
        fx, fy = self.fr(LEGEND_AT[0], LEGEND_AT[1] + 0.042 * len(legend) + 0.02)
        sc.text(fx, fy, "test accuracy after %d epochs" % Q.PRECISION_EPOCHS, layout.MIN_TEXT,
                "regular", layout.NOTE_INK, op * smoothstep(0.8, 1.0, train), "lm")
        sc.probe(("train",), (upto * 0.05, 0, 0), tol=6.0)

    def words(self, ph, under, scale):
        """The subtitle for the moment."""
        if ph == 0:
            return ("sign, exponent (the range) and mantissa (the precision): the trained MLP's "
                    "weight W₁[0,0] = %.7f in each format" % self.value)
        if ph == 1:
            return ("the representable numbers within 1%% of 1, 10⁻³ and 10⁻⁵: float16 steps by "
                    "the value/1024, bfloat16 by the value/128, until float16's subnormals "
                    "(below %s)" % pow10(Q.format_info("float16")["tiny"], 1))
        if ph == 2:
            if under < 0.5:
                return ("a real gradient: layer %d of a 20-layer tanh MLP (weights of std %g), "
                        "on %d images: RMS %s, below float16's smallest normal"
                        % (Q.DEEP_LAYER + 1, Q.DEEP_STD, Q.DEEP_BATCH, pow10(self.g_rms, 1)))
            return ("cast to float16, %.0f%% of its %s values become 0 and the rest keep one to "
                    "three significant bits; bfloat16 keeps them all"
                    % (100 * self.n_zero / self.g.size, f"{self.g.size:,}"))
        if ph == 3:
            return ("loss scaling: the loss %s 2%s makes every gradient 2%s larger before the "
                    "cast (none underflows), then ÷ 2%s in float32" % (TIMES, sup(16), sup(16),
                                                                        sup(16)))
        r = self.runs
        return ("training the MLP for %d epochs: float32 %.1f%%, bfloat16 %.1f%%, float16 with "
                "float32 master weights %.1f%%; float16 alone diverges at step %d (ε = 10⁻⁸ is 0 "
                "in float16)" % (Q.PRECISION_EPOCHS, 100 * r[0].acc, 100 * r[2].acc,
                                 100 * r[3].acc, r[1].diverged + 1))

