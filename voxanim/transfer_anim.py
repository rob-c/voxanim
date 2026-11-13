"""Three animations on fine-tuning and transfer, from the recorded runs of
:mod:`voxanim.transfer`.

**``transfer-learning``** -- *Transfer learning.*  The CNN trains on the
digits 0-4 (its first-layer filters settle as it does); then the digits 5-9
are learned from 100 images a class in three ways, one after another, each
row of filters live and each accuracy curve drawn as its run plays: from
scratch, with the pretrained conv layers frozen (the filters do not move:
they are the same arrays, bit for bit) and a new head, and with fine-tuning,
the pretrained layers at a tenth of the rate.  The final numbers, and who
wins with little data.

**``catastrophic-forgetting``** -- *Catastrophic forgetting.*  The network
trained on 0-4 has a bar per digit (its test accuracy); fine-tuned on 5-9,
the old digits' bars collapse while the new ones rise, with the minibatch's
images shown underneath.  Then the same fine-tuning with rehearsal (a share
of every minibatch drawn from a few hundred old images: the bars hold up)
and with lower rates (the conv layers at a hundredth, the head at a tenth:
the bars hold up, the new digits come slowly), and the trade-off in numbers.

**``learning-rate-finder-and-restarts``** -- *Learning-rate finder and
restarts.*  The MLP's loss against an exponentially rising learning rate,
drawn live over 300 steps; the knee (steepest fall), the minimum and the
divergence are marked.  Then training with a cosine schedule restarted every
epoch, peak at the knee: the schedule, the loss curve against one cosine,
and minibatch SGD on the two-weight landscape under the same shape of
schedule, kicked out of the basin at every restart and settling again.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import landscape as LS
from . import layout
from . import transfer as T
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep
from .losschart import pow10
from .netanim import INK_HI, INK_LO, NetAnimation, amax, div_colors, lit_colors, screen_box
from .optim_anim import BOX_CAP, BOX_HI, BOX_LO, LandscapeAnimation, _ink
from .tensor import Block
from .text import TIMES

DASH = "–"                                              # an en dash: the range 0–4
WAY_NAMES = dict(scratch="from scratch", frozen="frozen features", finetune="fine-tuned")
WAY_COLORS = dict(scratch=C.rgb("#d84e4e"), frozen=C.rgb("#1f6fb5"), finetune=C.rgb("#3d9e45"))
REMEDY_NAMES = dict(plain="plain fine-tuning", rehearsal="with rehearsal", lowlr="lower rates")
REMEDY_COLORS = dict(plain=C.rgb("#d84e4e"), rehearsal=C.rgb("#3d9e45"), lowlr=C.rgb("#1f6fb5"))
OLD_COL, NEW_COL = C.INPUT, C.rgb("#e0952e")          # the old digits teal, the new ones orange
DIGIT_SCALE = 0.2                                       # voxel size of a 28x28 thumbnail
FILTER_SCALE = 1.5                                      # voxel size of a 3x3 filter
PPU = 9.0
OLD_RANGE = "%d%s%d" % (T.OLD[0], DASH, T.OLD[-1])      # "0–4"
NEW_RANGE = "%d%s%d" % (T.NEW[0], DASH, T.NEW[-1])      # "5–9"


def pct(v):
    """``0.9312`` as ``93.1%``."""
    return "%.1f%%" % (100 * v)


def thousands(n):
    """``30596`` as ``30,596``."""
    return f"{int(n):,}"


def interp_rows(steps, values, step):
    """``values`` recorded at ``steps`` (leading axis), linearly interpolated at ``step``."""
    i = int(np.searchsorted(steps, step, side="right")) - 1
    i = max(0, min(i, len(steps) - 2))
    u = np.clip((step - steps[i]) / max(1e-9, steps[i + 1] - steps[i]), 0.0, 1.0)
    return values[i] * (1 - u) + values[i + 1] * u


def per_step(steps, values, n):
    """Recorded ``values`` as one value per step ``0..n`` (for a chart curve)."""
    return np.interp(np.arange(n + 1), steps, values)


def digit_colors(x, tint=None):
    """Colours of a digit's pixels: ink on the dark background, in ``tint`` if given."""
    hi = INK_HI if tint is None else C.mix(tint, (1, 1, 1), 0.4)
    lo = INK_LO if tint is None else C.mix(tint, (0, 0, 0), 0.4)
    return lit_colors(x, 1.0, lo, hi)


class Row:
    """A row of ``n`` equal sheets, stacked in world space about a centre."""

    def __init__(self, anim, n, shape, centre, scale, gap, name):
        self.sheets = [anim.sheet(shape, (0.5, 0.5), scale, "%s%d" % (name, i)) for i in range(n)]
        w = shape[1] * scale
        for i, s in enumerate(self.sheets):
            s.block.place(np.asarray(centre) + np.array([(i - (n - 1) / 2) * (w + gap), 0.0, 0.0]))

    def bbox(self):
        """World box of the whole row."""
        lo = np.min([s.bbox()[0] for s in self.sheets], axis=0)
        hi = np.max([s.bbox()[1] for s in self.sheets], axis=0)
        return lo, hi

    def fade(self, op):
        """Fade every sheet (invisible below 1%)."""
        for s in self.sheets:
            s.fade(op)
            s.visible = op > 0.01

    def side(self, anim):
        """Frame ``(x_left, y_centre, x_right)`` of the row on screen."""
        x0, y0, x1, y1 = screen_box(anim.cam, *self.bbox())
        cam = anim.cam
        return x0 / cam.width, (y0 + y1) / 2 / cam.height, x1 / cam.width


def phase_index(marks, t):
    """Index of the last mark at or before ``t``."""
    return int(np.searchsorted(marks, t, side="right") - 1)


# ------------------------------------------------------------------ 1. transfer learning

TL_DIGITS_AT = (0.37, 0.27)         # layout centre of the example digits
TL_GRID_AT = (0.39, 0.675)          # layout centre of the filter grid (4 rows)
TL_ROW_DZ = 6.6                     # world units between filter rows
TL_CHART = ((0.66, 0.945), (0.32, 0.64))
TL_ACC_RANGE = (0.5, 1.0)           # the accuracy axis (the three runs end within a few points)
TL_READ = (0.66, 0.72)              # frame position of the legend under the chart
TL_SPAN = 9.0                       # seconds each run plays for


class TransferLearning(NetAnimation):
    """``transfer-learning``: the pretrained filters, then the three ways."""

    PHASES = ("pretrain", "scratch", "frozen", "fine-tune", "results")

    def __init__(self, data, pre, runs):
        super().__init__("Transfer learning", "", PPU)
        self.data, self.pre, self.runs = data, pre, runs
        self.hold = 2.5
        F = T.FILTERS[0]
        c = self.cam.frac(*TL_DIGITS_AT)
        self.digits = Row(self, len(T.OLD), (28, 28), c, DIGIT_SCALE, 1.5, "digit")
        self.old_x = [data.x_old[np.nonzero(data.y_old == d)[0][0]] for d in T.OLD]
        self.new_x = [data.x_few[np.nonzero(data.y_few == d - T.NEW[0])[0][0]] for d in T.NEW]
        g = self.cam.frac(*TL_GRID_AT)
        self.rows = {}
        for n, key in enumerate(("pre", *T.WAYS)):
            at = g + np.array([0.0, 0.0, (1.5 - n) * TL_ROW_DZ])
            self.rows[key] = Row(self, F, (3, 3), at, FILTER_SCALE, 1.0, key)
        self.wmax = max(amax(pre.W1), *(amax(r.W1) for r in runs.values()))
        self.chart = LS.LossChart(self, TL_CHART, T.STEPS, *TL_ACC_RANGE)
        self.curves = {w: per_step(r.steps, r.acc, T.STEPS) for w, r in runs.items()}
        self.moved = np.abs(T.filters(runs["finetune"].P["W1"]) - pre.W1[-1]).max() / amax(pre.W1)
        Tm, t = {}, 0.3
        Tm["digits"] = (t, t + 1.5)
        Tm["pretrain"] = (t + 1.0, t + 6.0)
        t += 7.5
        Tm["switch"] = (t, t + 1.8)
        t += 2.4
        for way in T.WAYS:
            Tm[way] = (t, t + TL_SPAN)
            t += TL_SPAN + 1.2
        Tm["results"] = (t, t + 1.5)
        t += 5.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["switch"][0], *(Tm[w][0] for w in T.WAYS[1:]), Tm["results"][0]]

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else phase_index(self.marks, t)
        self._digits(sc, t, back)
        self._filters(sc, t, back)
        self._chart(sc, t, back)
        self.subtitle = self._words(t, ph)
        self.step_indicator(sc, ph, self.PHASES)

    def _digits(self, sc, t, back):
        """Five example digits: the old task's, then the new task's."""
        rv = ease_out(self.progress("digits", t))
        switch = self.progress("switch", t)
        new = switch >= 0.5
        swap = 1 - smoothstep(0.0, 0.5, switch) + smoothstep(0.5, 1.0, switch)   # out, then in
        for i, s in enumerate(self.digits.sheets):
            x = self.new_x[i] if new else self.old_x[i]
            s.fill(digit_colors(x, NEW_COL if new else OLD_COL))
            s.light(x, 0.6)
            s.fade(back * rv * swap)
        n_old, n_new = len(self.data.x_old), len(self.data.x_few)
        text = ("digits %s: %s training images" % (OLD_RANGE, thousands(n_old)) if not new
                else "digits %s: %d of each, %d in all" % (NEW_RANGE, T.N_FEW, n_new))
        self.caption(sc, self.digits.bbox(), text, rv * back * swap, size=0.028, max_width=0.45)

    def _filters(self, sc, t, back):
        """The four rows of first-layer filters: the pretraining, then one row a way."""
        pre = self.pre
        u = smoothstep(0.0, 1.0, self.progress("pretrain", t))
        step = u * pre.steps[-1]
        on = smoothstep(0.0, 0.2, self.progress("pretrain", t)) * back
        self._row(sc, "pre", interp_rows(pre.steps, pre.W1, step), on, "trained on %s" % OLD_RANGE)
        for way in T.WAYS:
            run = self.runs[way]
            u = self.progress(way, t)
            if u <= 0:
                self.rows[way].fade(0.0)
                continue
            step = smoothstep(0.0, 1.0, u) * T.STEPS
            self._row(sc, way, interp_rows(run.steps, run.W1, step), smoothstep(0.0, 0.1, u) * back,
                      WAY_NAMES[way], WAY_COLORS[way])

    def _row(self, sc, key, W1, op, name, color=None):
        """One row of filters with its name on the left."""
        row = self.rows[key]
        for f, s in enumerate(row.sheets):
            s.fill(div_colors(W1[..., f].ravel(), self.wmax))
            s.cage = 1.0
        row.fade(op)
        if op <= 0.01:
            return
        x0, y, _x1 = row.side(self)
        ink = layout.CAPTION_INK if color is None else _ink(color)
        sc.text(x0 - 0.012, y, name, 0.027, "regular", ink, op * self._cap, "rm")

    def _chart(self, sc, t, back):
        """Accuracy on the new digits against the step, one curve a way."""
        on = smoothstep(0.0, 0.5, self.progress(T.WAYS[0], t)) * back
        if on <= 0.01:
            return
        self.chart.frame(sc, on, xlabel="step", ylabel="test accuracy on %s" % NEW_RANGE)
        for v in np.linspace(*TL_ACC_RANGE, 6):
            self.chart.label(sc, 0, v, "%d%%" % round(100 * v), layout.NOTE_INK, on, dx=-0.006,
                             anchor="rm", size=0.024)
        for way in T.WAYS:
            u = self.progress(way, t)
            if u <= 0:
                continue
            upto = smoothstep(0.0, 1.0, u) * T.STEPS
            self.chart.curve(sc, self.curves[way], max(upto, 1e-6), WAY_COLORS[way], on,
                             width=1.4, marker=u < 1)
        self._legend(sc, t, on)

    def _legend(self, sc, t, op):
        """The ways in their colours; the step and accuracy while a run plays,
        the final test accuracy once it is done."""
        op *= self._cap
        x, y = TL_READ
        for n, way in enumerate(T.WAYS):
            u = self.progress(way, t)
            if u <= 0:
                continue
            yy = y + n * 0.045
            sc.text(x, yy, WAY_NAMES[way], 0.027, "regular", _ink(WAY_COLORS[way]), op, "lm")
            run = self.runs[way]
            if u < 1:
                step = smoothstep(0.0, 1.0, u) * T.STEPS
                text = "step %d: %s" % (int(step), pct(np.interp(step, run.steps, run.acc)))
                ink = layout.CAPTION_INK
            else:
                text, ink = pct(run.test_acc), layout.STRONG_INK
            sc.text(0.945, yy, text, 0.027, "mono", ink, op, "rm")
        res = self.progress("results", t)
        if res > 0:
            n_test = int(np.isin(self.data.y_test, T.NEW).sum())
            sc.text(x, y + 3 * 0.045 + 0.012, "on the %s test images of %s"
                    % (thousands(n_test), NEW_RANGE), 0.025, "regular", layout.NOTE_INK,
                    op * res, "lm")

    def _words(self, t, ph):
        """The subtitle for the moment."""
        r, pre = self.runs, self.pre
        if ph == 0:
            if t >= self.t_done + self.hold or t < self.Tm["pretrain"][0] + 0.3:
                return ("a CNN (conv %d, pool, conv %d, pool, dense) learns the digits %s from "
                        "%s images" % (*T.FILTERS, OLD_RANGE, thousands(len(self.data.x_old))))
            return ("one epoch of Adam (%d steps of %d): %s of the %s test digits right; it has "
                    "never seen a %s"
                    % (pre.steps[-1], T.PRE["batch"], pct(pre.old_acc), OLD_RANGE, NEW_RANGE))
        if ph == 1:
            if t < self.Tm["scratch"][0] + 0.3:
                return ("now the digits %s, from only %d images of each: three ways, %d steps "
                        "each (batch %d, η = %g)" % (NEW_RANGE, T.N_FEW, T.STEPS, T.BATCH, T.LR))
            return ("from scratch: a fresh network, random filters, trained on the %d images "
                    "alone: %s" % (len(self.data.x_few), pct(r["scratch"].test_acc)))
        if ph == 2:
            return ("frozen: the conv layers are kept exactly as pretrained; only a new 5-way "
                    "head learns: %s" % pct(r["frozen"].test_acc))
        if ph == 3:
            return ("fine-tuned: pretrained layers at η/%d, the new head at η; the filters "
                    "change by at most %d%%: %s"
                    % (round(1 / T.LOW), round(100 * self.moved), pct(r["finetune"].test_acc)))
        order = sorted(T.WAYS, key=lambda w: -r[w].test_acc)
        short = dict(scratch="scratch", frozen="frozen", finetune="fine-tuned")
        return ("%d a class: %s: the strokes learned on %s carry over"
                % (T.N_FEW, ", ".join("%s %s" % (short[w], pct(r[w].test_acc)) for w in order),
                   OLD_RANGE))


# ------------------------------------------------------------------ 2. catastrophic forgetting

CF_LEVELS = 20                      # voxels of a full bar (5% each)
CF_BAR_W, CF_GAP = 2, 1             # a bar is CF_BAR_W voxels wide, CF_GAP from the next
CF_BARS_AT = (0.28, 0.44)           # layout centre of the bar panel
CF_BAR_SCALE = 1.2
CF_STRIP_AT = (0.30, 0.835)         # layout centre of the minibatch strip
CF_STRIP_SCALE = 0.15               # voxel size of its thumbnails
CF_CHART = ((0.63, 0.945), (0.30, 0.58))
CF_READ = (0.63, 0.66)              # frame position of the legend under the chart
CF_SPAN = 10.0                      # seconds each run plays for


class CatastrophicForgetting(NetAnimation):
    """``catastrophic-forgetting``: a bar per digit through three fine-tunings."""

    PHASES = ("before", "forgetting", "rehearsal", "lower rates", "trade-off")

    def __init__(self, data, pre, runs):
        super().__init__("Catastrophic forgetting", "", PPU)
        self.data, self.pre, self.runs = data, pre, runs
        self.hold = 2.5
        cols = T.CLASSES * (CF_BAR_W + CF_GAP) - CF_GAP
        self.bars = Block((CF_LEVELS, cols, 1), scale=CF_BAR_SCALE, name="bars")
        self.bars.place(self.cam.frac(*CF_BARS_AT))
        self.blocks_extra.append(self.bars)
        c = self.cam.frac(*CF_STRIP_AT)
        self.strip = Row(self, T.N_SHOWN, (28, 28), c, CF_STRIP_SCALE, 1.0, "shown")
        self.chart = LS.LossChart(self, CF_CHART, T.FT_STEPS, 0.0, 1.0)
        self.curves = {k: (per_step(r.steps, r.per_digit[:, list(T.OLD)].mean(1), T.FT_STEPS),
                           per_step(r.steps, r.per_digit[:, list(T.NEW)].mean(1), T.FT_STEPS))
                       for k, r in runs.items()}
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 2.0)
        t += 6.0
        for remedy in T.REMEDIES:
            Tm[remedy] = (t, t + CF_SPAN)
            t += CF_SPAN + 1.5
        Tm["table"] = (t, t + 1.5)
        t += 5.5
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, *(Tm[r][0] for r in T.REMEDIES), Tm["table"][0]]

    def current(self, t):
        """``(remedy, step)`` on show at ``t``: the run playing and its fractional step
        (None before the first run; the last run's end afterwards)."""
        for remedy in reversed(T.REMEDIES):
            t0, t1 = self.Tm[remedy]
            if t >= t0:
                return remedy, smoothstep(0.0, 1.0, remap(t, t0, t1)) * T.FT_STEPS
        return None, 0.0

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else phase_index(self.marks, t)
        remedy, step = self.current(t)
        self._bars(sc, t, remedy, step, back)
        self._strip(sc, t, remedy, step, back)
        self._chart(sc, t, remedy, step, back)
        self.subtitle = self._words(t, ph, remedy, step)
        self.step_indicator(sc, ph, self.PHASES)

    @staticmethod
    def columns(d):
        """The half-open column range of digit ``d``'s bar."""
        c0 = d * (CF_BAR_W + CF_GAP)
        return c0, c0 + CF_BAR_W

    def _accuracy(self, remedy, step):
        """Per-digit accuracy ``(10,)`` on show."""
        if remedy is None:
            return self.pre.acc
        run = self.runs[remedy]
        return interp_rows(run.steps, run.per_digit, step)

    def _bars(self, sc, t, remedy, step, back):
        """One bar per digit, as tall as its accuracy; the group means above."""
        acc = self._accuracy(remedy, step)
        rise = ease_out(self.progress("rise", t))
        b = self.bars
        b.present[...] = False
        for d in range(T.CLASSES):
            n = int(round(acc[d] * CF_LEVELS * rise))
            c0, c1 = self.columns(d)
            b.present[CF_LEVELS - n:, c0:c1, 0] = True
            b.base[:, c0:c1, 0] = OLD_COL if d in T.OLD else NEW_COL
        b.opacity *= back
        op = back * rise * self._cap
        if op <= 0.01:
            return
        for d in range(T.CLASSES):
            lo, hi = b.box((CF_LEVELS - 1, CF_LEVELS), self.columns(d), (0, 1))
            self.caption(sc, (lo, hi), str(d), op, size=0.028)
        for digits, name, col in ((T.OLD, OLD_RANGE, OLD_COL), (T.NEW, NEW_RANGE, NEW_COL)):
            lo, hi = b.box((0, CF_LEVELS), (self.columns(digits[0])[0],
                                            self.columns(digits[-1])[1]), (0, 1))
            mean = float(acc[list(digits)].mean())
            self.caption(sc, (lo, hi), "%s: %s" % (name, pct(mean)), op, size=0.029, above=True,
                         ink=_ink(col))

    def _strip(self, sc, t, remedy, step, back):
        """The first images of the minibatch at the recorded step on show."""
        if remedy is None:
            self.strip.fade(0.0)
            return
        run = self.runs[remedy]
        i = int(np.searchsorted(run.steps, step, side="right")) - 1
        i = max(0, min(i, len(run.steps) - 1))
        on = smoothstep(0.0, 0.08, self.progress(remedy, t)) * back
        for j, s in enumerate(self.strip.sheets):
            x, y = run.shown[i, j], int(run.shown_y[i, j])
            s.fill(digit_colors(x, OLD_COL if y in T.OLD else NEW_COL))
            s.light(x, 0.6)
        self.strip.fade(on)
        if remedy == "rehearsal":
            text = ("a minibatch: %d images of %s + %d from a buffer of %d old digits"
                    % (T.FT_BATCH - T.REHEARSE, NEW_RANGE, T.REHEARSE, len(self.data.x_buffer)))
        else:
            text = "a minibatch: %d images, all %s" % (T.FT_BATCH, NEW_RANGE)
        # under the strip: its lowest (left) end, clear of the digit labels above it
        self.caption(sc, self.strip.bbox(), text, on, size=0.027, max_width=0.5)

    def _chart(self, sc, t, remedy, step, back):
        """Mean accuracy on the old (solid) and new (dashed) digits against the step."""
        if remedy is None:
            return
        on = smoothstep(0.0, 0.3, self.progress(T.REMEDIES[0], t)) * back
        self.chart.frame(sc, on, xlabel="step", ylabel="test accuracy")
        for v in (0.25, 0.5, 0.75, 1.0):
            self.chart.label(sc, 0, v, "%d%%" % round(100 * v), layout.NOTE_INK, on, dx=-0.006,
                             anchor="rm", size=0.024)
        for r in T.REMEDIES:
            u = self.progress(r, t)
            if u <= 0:
                continue
            upto = max(1e-6, smoothstep(0.0, 1.0, u) * T.FT_STEPS)
            fade = 1.0 if r == remedy else 0.45          # earlier runs stay, fainter
            old, new = self.curves[r]
            self.chart.curve(sc, old, upto, REMEDY_COLORS[r], on * fade, width=1.4,
                             marker=u < 1)
            self.chart.curve(sc, new, upto, REMEDY_COLORS[r], on * fade, width=1.2,
                             marker=False, dashed=True)
        self._legend(sc, t, remedy, step, on)

    def _legend(self, sc, t, remedy, step, op):
        """The remedies in their colours, with old / new accuracy; solid and dashed explained."""
        op *= self._cap
        x, y = CF_READ
        sc.text(x, y, "solid: %s   dashed: %s" % (OLD_RANGE, NEW_RANGE), 0.025, "regular",
                layout.NOTE_INK, op, "lm")
        sc.text(0.945, y + 0.045, "%s     %s" % (OLD_RANGE, NEW_RANGE), 0.025, "mono",
                layout.NOTE_INK, op, "rm")
        for n, r in enumerate(T.REMEDIES):
            u = self.progress(r, t)
            if u <= 0:
                continue
            yy = y + 0.045 * (n + 2)
            sc.text(x, yy, REMEDY_NAMES[r], 0.027, "regular", _ink(REMEDY_COLORS[r]), op, "lm")
            run = self.runs[r]
            if u < 1:
                acc = interp_rows(run.steps, run.per_digit, step)
                a_old, a_new = acc[list(T.OLD)].mean(), acc[list(T.NEW)].mean()
                ink = layout.CAPTION_INK
            else:
                a_old, a_new, ink = run.test_old, run.test_new, layout.STRONG_INK
            sc.text(0.945, yy, "%5.1f%%  %5.1f%%" % (100 * a_old, 100 * a_new), 0.027, "mono",
                    ink, op, "rm")

    def _words(self, t, ph, remedy, step):
        """The subtitle for the moment."""
        r, pre = self.runs, self.pre
        if ph == 0:
            if t >= self.t_done + self.hold or t < self.Tm["rise"][1] + 0.5:
                return ("trained on %s: %s of those digits right; the %s outputs have only ever "
                        "been pushed down: %s"
                        % (OLD_RANGE, pct(pre.old_acc), NEW_RANGE,
                           pct(pre.acc[list(T.NEW)].mean())))
            return ("fine-tune it on %s only (%s images, batch %d, Adam η = %g) and watch the "
                    "bars of %s" % (NEW_RANGE, thousands(len(self.data.x_new)), T.FT_BATCH, T.LR,
                                    OLD_RANGE))
        k = int(step)
        if ph == 1:
            run = r["plain"]
            if step < T.FT_STEPS:
                acc = interp_rows(run.steps, run.per_digit, step)
                return ("step %d: %s %s, %s %s; every minibatch says 'not a %s', and nothing "
                        "says otherwise" % (k, OLD_RANGE, pct(acc[list(T.OLD)].mean()), NEW_RANGE,
                                             pct(acc[list(T.NEW)].mean()), OLD_RANGE))
            return ("catastrophic forgetting: after %d steps %s is at %s (from %s) while %s "
                    "reaches %s" % (T.FT_STEPS, OLD_RANGE, pct(run.test_old), pct(pre.old_acc),
                                     NEW_RANGE, pct(run.test_new)))
        if ph == 2:
            run = r["rehearsal"]
            if step < T.FT_STEPS:
                return ("rehearsal: %d of each %d images come from a buffer of %d old digits: %s "
                        "stays in the loss (step %d)"
                        % (T.REHEARSE, T.FT_BATCH, len(self.data.x_buffer), OLD_RANGE, k))
            return ("with rehearsal: %s %s, %s %s; %d old images a batch keep most of what "
                    "%s taught" % (OLD_RANGE, pct(run.test_old), NEW_RANGE, pct(run.test_new),
                                    T.REHEARSE, thousands(len(self.data.x_old))))
        if ph == 3:
            run = r["lowlr"]
            if step < T.FT_STEPS:
                return ("lower rates: conv at η/%d, head at η/%d (at η the head alone forgets): "
                        "smaller steps lose less (step %d)"
                        % (round(1 / T.FT_LOW[0]), round(1 / T.FT_LOW[1]), k))
            return ("lower rates: %s %s, %s %s after %d steps: what is kept is paid for in "
                    "slower learning" % (OLD_RANGE, pct(run.test_old), NEW_RANGE,
                                         pct(run.test_new), T.FT_STEPS))
        return ("the trade-off (%s / %s): plain %s / %s, rehearsal %s / %s, lower rates %s / %s"
                % (OLD_RANGE, NEW_RANGE, pct(r["plain"].test_old), pct(r["plain"].test_new),
                   pct(r["rehearsal"].test_old), pct(r["rehearsal"].test_new),
                   pct(r["lowlr"].test_old), pct(r["lowlr"].test_new)))


# ------------------------------------------------------------------ 3. the finder and restarts

RANGE_CHART = ((0.08, 0.52), (0.30, 0.80))      # the range test's chart
RANGE_LOSS_HI = 3.0                              # top of its loss axis
RANGE_READ = (0.60, 0.33)                        # frame position of the read-outs beside it
LR_CHART = ((0.715, 0.945), (0.27, 0.40))        # the schedule (as schedule_anim)
LEGEND_Y = 0.445                                 # the two runs' names, under the schedule
LOSS_CHART = ((0.715, 0.945), (0.56, 0.78))      # the training loss
READ_Y = 0.83                                    # the step read-out / the table under it
LOSS_RANGE = (0.08, 2.6)                         # its (log) range
RESTART_INK = C.rgb("#1f6fb5")
COSINE_INK = (0.5, 0.52, 0.54)
MARK_COLORS = dict(knee=C.rgb("#3d9e45"), minimum=C.rgb("#1f6fb5"), diverge=C.rgb("#d84e4e"))
MARK_NAMES = dict(knee="steepest fall", minimum="lowest loss", diverge="back above the start")
TRAIN_SPAN = 18.0                                # seconds the restart training plays for


def sci(v):
    """``8.4e-4`` as ``8.4×10⁻⁴``; rates of 0.01 and up plainly, to two figures (``0.023``)."""
    return "%.2g" % v if v >= 0.01 else pow10(v, 1)


class LearningRateFinder(LandscapeAnimation):
    """``learning-rate-finder-and-restarts``: the range test, then warm restarts
    on the MLP and the landscape together."""

    PHASES = ("range test", "the knee", "restarts", "results")

    def __init__(self, lrs, losses, peak, mlp, land_run):
        prob = LS.problem()
        self.lrs, self.losses = lrs, losses
        self.an = T.lr_analysis(lrs, losses)
        self.peak, self.mlp, self.land_run = peak, mlp, land_run
        super().__init__("Learning-rate finder and restarts", prob, BOX_LO, BOX_HI, BOX_CAP,
                         T.LAND_STEPS, float(land_run.loss[0]) + 0.1)
        self.steps = mlp["restarts"].steps
        self.range_chart = LS.LossChart(self, RANGE_CHART, T.LR_STEPS - 1, 0.0, RANGE_LOSS_HI)
        self.lr_chart = LS.LossChart(self, LR_CHART, self.steps, self.peak * 1e-3, self.peak * 2,
                                     log=True)
        self.loss_chart = LS.LossChart(self, LOSS_CHART, self.steps, *LOSS_RANGE, log=True)
        self.sub = {k: per_step(r.sub_steps, r.sub_loss, self.steps) for k, r in mlp.items()}
        # ---- timeline
        Tm, t = {}, 0.3
        tempo = Tempo(slow=0.12, fast=0.03, ramp_in=15.0, ramp_out=12.0)
        d = self.sweep("range", T.LR_STEPS, tempo)
        Tm["range"] = (t, t + d)
        t += d + 0.8
        Tm["marks"] = (t, t + 6.0)
        t += 6.0 + 1.5
        Tm["rise"] = (t, t + 2.5)
        Tm["sched"] = (t + 0.5, t + 3.0)
        t += 3.5
        Tm["train"] = (t, t + TRAIN_SPAN)
        t += TRAIN_SPAN + 1.0
        Tm["results"] = (t, t + 1.5)
        t += 5.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["marks"][0], Tm["rise"][0], Tm["results"][0]]

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        pos = self.step_at("range", t)
        marks = self.progress("marks", t)
        rise = self.progress("rise", t)
        sched = self.progress("sched", t)
        train = self.progress("train", t)
        leave = 1 - smoothstep(0.0, 0.5, rise)          # the range test makes way
        # the chart fades in with the captions, so the loop's first frame is bare
        self._range(sc, pos, marks, smoothstep(0.0, 0.5, t) * leave * back)
        self.landscape(sc, t, back, rise)
        self.minimum_marker(sc, smoothstep(0.6, 1.0, rise) * back)
        self._schedule(sc, sched, smoothstep(0.0, 0.3, sched) * back)
        res = self.progress("results", t)
        self._training(sc, train, res, smoothstep(0.0, 0.1, train) * back)
        self._table(sc, smoothstep(0.3, 1.0, res) * back)
        self.subtitle = self._words(t, ph, pos, marks, train)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the range test ------------------------------------------------------------
    def _range(self, sc, pos, marks, op):
        """The loss against the rising learning rate, raw (faint) and smoothed."""
        if op <= 0.01:
            return
        ch = self.range_chart
        ch.frame(sc, op, xlabel="", ylabel="minibatch loss")
        ch.label(sc, (T.LR_STEPS - 1) / 2, 0.0, "learning rate η", layout.CAPTION_INK, op, dx=0.0,
                 dy=0.05, anchor="mt")
        for e in range(-5, 1):
            k = (e - np.log10(T.LR_LO)) / np.log10(T.LR_HI / T.LR_LO) * (T.LR_STEPS - 1)
            ch.label(sc, k, 0.0, pow10(10.0 ** e), layout.NOTE_INK, op, dx=0.0, dy=0.012,
                     anchor="mt", size=0.024)
        for v in (1, 2, 3):
            ch.label(sc, 0, v, "%d" % v, layout.NOTE_INK, op, dx=-0.006, anchor="rm", size=0.024)
        upto = min(pos, T.LR_STEPS - 1)
        ch.curve(sc, self.losses, upto, (0.62, 0.64, 0.66), 0.6 * op, width=1.0, marker=False)
        ch.curve(sc, self.an.smooth, upto, layout.STRONG_INK, op, width=1.6, marker=upto < pos)
        k = int(min(pos, T.LR_STEPS - 1))
        if marks <= 0:
            self.readout(sc, ["step %d of %d" % (k, T.LR_STEPS), "η = %s" % sci(self.lrs[k]),
                              "loss %.2f" % self.losses[k], "smoothed %.2f" % self.an.smooth[k]],
                         op, x=RANGE_READ[0], y=RANGE_READ[1], dy=0.045)
            return
        for n, key in enumerate(("knee", "minimum", "diverge")):
            show = smoothstep(0.0, 0.25, remap(marks, n / 3, (n + 0.8) / 3)) * op
            if show <= 0.01:
                continue
            i = getattr(self.an, key)
            p = ch.pt(i, self.an.smooth[i])
            col = MARK_COLORS[key]
            sc.voxel(p, col, show, glow=0.4, shadow=False, reflect=False, size=0.8)
            top, bottom = ch.pt(i, RANGE_LOSS_HI), ch.pt(i, 0.0)
            for j in range(10):
                a, b = top + (bottom - top) * j / 10, top + (bottom - top) * (j + 0.5) / 10
                sc.line(a, b, width=0.9, color=col, opacity=0.7 * show, bias=0.0)
            y = RANGE_READ[1] + n * 0.09
            sc.text(RANGE_READ[0], y, MARK_NAMES[key], 0.028, "regular", _ink(col),
                    show * self._cap, "lm")
            sc.text(RANGE_READ[0], y + 0.04, "η = %s, loss %.2f" % (sci(self.lrs[i]),
                                                                    self.an.smooth[i]),
                    0.027, "mono", layout.CAPTION_INK, show * self._cap, "lm")

    # -- the restarts --------------------------------------------------------------
    def _schedule(self, sc, sched, op):
        """The restarted cosine (and the single cosine, faint) against the step."""
        if op <= 0.01:
            return
        ch = self.lr_chart
        ch.frame(sc, op, xlabel="", ylabel="learning rate")     # the loss chart below has "step"
        upto = smoothstep(0.0, 1.0, sched) * self.steps
        ch.curve(sc, self.mlp["cosine"].lrs, upto, COSINE_INK, 0.7 * op, width=1.1, marker=False)
        ch.curve(sc, self.mlp["restarts"].lrs, upto, RESTART_INK, op, width=1.4, marker=sched < 1)
        ch.label(sc, 0, self.peak, sci(self.peak), layout.NOTE_INK, op, dx=-0.006, anchor="rm",
                 size=0.024)
        ch.label(sc, 0, self.peak / 100, sci(self.peak / 100), layout.NOTE_INK, op, dx=-0.006,
                 anchor="rm", size=0.024)

    def _training(self, sc, train, res, op):
        """The MLP's training loss and the landscape walker, in step; the read-out
        makes way for the table as ``res`` (the results phase) begins."""
        if op <= 0.01:
            return
        u = smoothstep(0.0, 1.0, train)
        k = u * self.steps
        ch = self.loss_chart
        ch.frame(sc, op, xlabel="step", ylabel="training loss (2,000 images)")
        for v in (2, 1, 0.3, 0.1):
            ch.label(sc, 0, v, "%g" % v, layout.NOTE_INK, op, dx=-0.006, anchor="rm", size=0.024)
        length = self.steps // T.CYCLES
        for c in range(1, T.CYCLES):
            if k >= c * length:
                top, bottom = ch.pt(c * length, LOSS_RANGE[1]), ch.pt(c * length, LOSS_RANGE[0])
                for j in range(8):
                    a, b = top + (bottom - top) * j / 8, top + (bottom - top) * (j + 0.5) / 8
                    sc.line(a, b, width=0.8, color=LS.CHART_INK, opacity=0.6 * op, bias=0.0)
        ch.curve(sc, self.sub["cosine"], max(k, 1e-6), COSINE_INK, 0.8 * op, width=1.1,
                 marker=False)
        ch.curve(sc, self.sub["restarts"], max(k, 1e-6), RESTART_INK, op, width=1.5,
                 marker=train < 1)
        # the landscape walker under the same shape of schedule
        pos = u * T.LAND_STEPS
        self.walker(sc, self.land_run, pos, RESTART_INK, op, ("restart",), size=0.9, fade_len=60)
        i = int(k)
        run = self.mlp["restarts"]
        lines = ["step %d of %d" % (i, self.steps),
                 "η = %s" % sci(run.lrs[min(i, self.steps - 1)]),
                 "loss %.3f  (one cosine %.3f)" % (self.sub["restarts"][i], self.sub["cosine"][i])]
        self.readout(sc, lines, op * (1 - smoothstep(0.0, 0.3, res)), y=READ_Y, dy=0.04)
        self.legend(sc, op)

    def legend(self, sc, op):
        """The two runs' names under the schedule chart."""
        op *= self._cap
        if op <= 0.01:
            return
        sc.text(0.715, LEGEND_Y, "cosine, restarted every epoch", 0.026, "regular",
                _ink(RESTART_INK), op, "lm")
        sc.text(0.715, LEGEND_Y + 0.034, "one cosine over all %d steps" % self.steps, 0.026,
                "regular", COSINE_INK, op, "lm")

    def _table(self, sc, op):
        """Test accuracy at the end of each cycle, and of the single cosine."""
        op *= self._cap
        if op <= 0.01:
            return
        r, c = self.mlp["restarts"], self.mlp["cosine"]
        x, y = 0.715, READ_Y - 0.01
        lines = (("test accuracy, cycle by cycle", layout.STRONG_INK),
                 (" → ".join("%.1f" % (100 * a) for a in r.cycle_acc) + "%", _ink(RESTART_INK)),
                 ("one cosine: %s" % pct(c.acc), COSINE_INK))
        for n, (text, ink) in enumerate(lines):
            sc.text(x, y + 0.036 * n, text, 0.026, "regular", ink, op, "lm")

    def _words(self, t, ph, pos, marks, train):
        """The subtitle for the moment."""
        an, lrs = self.an, self.lrs
        if ph == 0:
            if t >= self.t_done + self.hold or pos <= 0:
                return ("the range test: %d steps of Adam on the MLP while η rises %d%% a step, "
                        "from %s to %s; the minibatch loss is recorded"
                        % (T.LR_STEPS, round(100 * ((T.LR_HI / T.LR_LO) ** (1 / (T.LR_STEPS - 1))
                                                    - 1)), pow10(T.LR_LO), pow10(T.LR_HI)))
            k = int(min(pos, T.LR_STEPS - 1))
            if k < an.knee:
                return ("too small a rate and nothing happens: at η = %s the loss is still %.2f "
                        "(it started at %.2f)" % (sci(lrs[k]), an.smooth[k], an.smooth[0]))
            if k < an.diverge:
                return ("the loss falls fastest around η = %s, bottoms out at η = %s (%.2f) and "
                        "then climbs" % (sci(lrs[an.knee]), sci(lrs[an.minimum]),
                                         an.smooth[an.minimum]))
            return ("past η = %s the steps overshoot: the loss is back above where it started, "
                    "and at %s it is %.1f" % (sci(lrs[an.diverge]), sci(lrs[k]), an.smooth[k]))
        if ph == 1:
            if marks < 0.4:
                return ("the knee: where the smoothed loss falls fastest against log η, at "
                        "η = %s (the usual 10⁻³)" % sci(lrs[an.knee]))
            return ("lowest loss at η = %s; back above its start from η = %s: a working range "
                    "of about %d%s"
                    % (sci(lrs[an.minimum]), sci(lrs[an.diverge]),
                       int(round(lrs[an.diverge] / lrs[an.knee])), TIMES))
        if ph == 2:
            k = int(smoothstep(0.0, 1.0, train) * self.steps)
            length = self.steps // T.CYCLES
            r = self.sub["restarts"]
            if train <= 0:
                return ("warm restarts: a cosine from the knee rate %s to 0 every epoch (%d "
                        "steps), restarted %d times" % (sci(self.peak), length, T.CYCLES - 1))
            c = k // length
            if c == 0:
                return ("cycle 1, a plain cosine: on the landscape minibatch SGD (B = %d, peak "
                        "η = %g) settles as η falls" % (T.LAND_BATCH, T.LAND_PEAK))
            if k % length < 60:
                i0 = c * length
                return ("restart %d: η jumps back to %s; the MLP's loss pauses (%.2f → %.2f in 60 "
                        "steps), the walker is thrown out of the basin"
                        % (c, sci(self.peak), r[i0], r[min(i0 + 60, self.steps)]))
            return ("η falls: the loss settles lower (step %d: %.3f, one cosine %.3f); the walker "
                    "drifts back to the floor" % (k, r[k], self.sub["cosine"][k]))
        r, c = self.mlp["restarts"], self.mlp["cosine"]
        return ("%d epochs: restarts %s, one cosine %s; gentle restarts, and a model at every "
                "cycle's end" % (T.EPOCHS, pct(r.acc), pct(c.acc)))
