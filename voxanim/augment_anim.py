"""Data augmentation: the same small-data classifier trained with and
without random transforms of its 500 training digits.

The problem: 500 MNIST training images, ten classes, a 784-128-10 ReLU MLP
(He init) trained by Adam (lr 0.001, minibatches of 50) for 200 epochs
(2,000 steps) -- far too little data for the network, which memorises it.
The augmentation (:class:`voxanim.generalise.Augment`): every time a digit
enters a minibatch it is shifted by up to +-3 pixels in each direction,
rotated by up to +-15 degrees and scaled by up to +-15%, all drawn at random
and resampled bilinearly in numpy, so the network never sees the same image
twice.

1. **a digit** -- one training image and its label,
2. **transforms** -- ten random transforms of it drawn by the same generator
   training uses, one after another, each with its shift, angle and scale,
3. **training** -- both runs recorded by :func:`voxanim.deepnet.train`: the
   minibatch on screen at every recorded step (the first eight images, as
   stored and as augmented) and the training and test loss curves: without
   augmentation the training loss reaches 0.001 while the test loss climbs
   from its low of 0.38 back up to 0.48; with it the test loss ends at 0.18,
4. **invariance** -- accuracy on the 10,000 test digits, on 2,000 of them
   shifted by up to +-3 px and on 2,000 rotated by up to +-15 degrees, for
   both networks: 87% / 47% / 86% without augmentation, 94% / 92% / 93%
   with it.  (The numbers are this machine's; CI retrains.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import Data, Spec, subset, train
from .diagram import EDGE, Diagram
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gen_common import (
    DIGIT_INK,
    bar,
    draw_digit,
    note,
    pct,
    readout,
    strong,
)
from .generalise import Augment, affine, error_rate
from .losschart import LossChart
from .sweep import Sweep
from .text import MINUS, TIMES

N_TRAIN, N_TEST, SEED = 500, 10000, 0
SIZES = (784, 128, 10)
TRAIN = dict(steps=2000, batch=50, lr=0.001, opt="adam")
SHIFT, DEGREES, SCALE = 3.0, 15.0, 0.15          # the augmentation's ranges
PROBE_ROWS = 8                                   # minibatch images recorded per step
SHOW_INDEX = 8                                   # the training image the transforms phase uses: a 3
N_SHOW = 10                                      # transforms shown of it
SHOW_SEED = 3                                    # the generator that draws them
N_INVARIANCE, INVARIANCE_SEED = 2000, 7          # test digits transformed for the measurement
AUG_INK = C.rgb("#3d9e45")                       # the augmented run
PLAIN_INK = (0.46, 0.48, 0.52)                   # the plain run
BIG_PX, THUMB_PX, BATCH_PX, GRID_PX = 120.0, 30.0, 34.0, 24.0    # digit sizes
DIGIT_AT = (0.16, 0.54)                          # the original, transforms phase (layout)
RESULT_AT = (0.40, 0.54)                         # the transformed copy
THUMBS_AT = (0.28, 0.83)                         # row of thumbnails: centre
GRID_AT, GRID_COLS, GRID_N = (0.76, 0.54), 12, 72    # some of the training set, first phases
STRIPS_AT = (0.27, 0.56)                         # the two minibatch strips, training phase
BARS = ((0.08, 0.50), 0.84, 190.0)               # invariance bars: x span, bottom y, max px
CHART = ((0.57, 0.94), (0.34, 0.66))             # the loss chart
NOTE_AT = (0.57, 0.77)                           # read-outs under it
LOSS_RANGE = (0.0, 2.5)


def augmentation():
    """The transform training uses."""
    return Augment(SHIFT, DEGREES, SCALE)


def dataset(cache_dir=None):
    """The 500-image problem (test set: all 10,000 test digits)."""
    raw = subset(N_TRAIN, N_TEST, None, SEED, cache_dir)
    return Data(raw.x_train, raw.y_train, raw.x_test, raw.y_test, 10)


def runs(cache_dir=None):
    """``(data, plain, augmented)``: the problem and the two recorded runs,
    each with the first eight images of every recorded step's minibatch."""
    data = dataset(cache_dir)
    spec = Spec(SIZES, "relu", "he")
    kw = dict(TRAIN, probe_layer=0, probe_keys=("h",), probe_rows=PROBE_ROWS)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    plain = train(spec, data, **kw)
    aug = train(spec, data, augment=augmentation(), **kw)
    return data, plain, aug


def transformed_test_sets(data):
    """``(shifted, rotated)``: the first :data:`N_INVARIANCE` test images
    shifted by up to +-SHIFT px, and rotated by up to +-DEGREES, seeded."""
    rng = np.random.default_rng(INVARIANCE_SEED)
    x = data.x_test[:N_INVARIANCE]
    n = len(x)
    shifted = affine(x, rng.uniform(-SHIFT, SHIFT, n), rng.uniform(-SHIFT, SHIFT, n), 0.0, 1.0)
    rotated = affine(x, 0.0, 0.0, np.radians(rng.uniform(-DEGREES, DEGREES, n)), 1.0)
    return shifted, rotated


def invariance(data, run):
    """``(clean, shifted, rotated)`` accuracies of a trained run."""
    spec = run.spec
    shifted, rotated = transformed_test_sets(data)
    y = data.y_test[:N_INVARIANCE]
    return (1 - error_rate(spec, run.P, data.x_test, data.y_test),
            1 - error_rate(spec, run.P, shifted, y), 1 - error_rate(spec, run.P, rotated, y))


def shown_transforms():
    """The ``N_SHOW`` transforms of the transforms phase: ``(dx, dy, angle, scale)`` arrays."""
    return augmentation().draw(np.random.default_rng(SHOW_SEED), N_SHOW)


def describe(dx, dy, angle, scale):
    """``shift (+2, −1) px · rotate −9° · scale ×1.08``."""
    def signed(v):
        return ("+%d" % round(v)) if v >= 0.5 else (MINUS + "%d" % round(-v) if v <= -0.5 else "0")
    deg = np.degrees(angle)
    return "shift (%s, %s) px   rotate %s°   scale %s%.2f" % (
        signed(dx), signed(-dy), signed(deg), TIMES, scale)


@dataclass
class Frame:
    """What one instant shows."""

    back: float
    phase: int
    k: int          # transforms phase: transforms finished
    cur: int        # ... the one under way (-1 none)
    cu: float       # ... its progress
    x: float        # training phase: fractional recorded-step index
    grow: float     # invariance phase: bars' growth


class Augmentation(Diagram):
    """Training with and without random transforms of the digits (see the module docstring)."""

    PHASES = ("a digit", "transforms", "training", "invariance")

    def __init__(self, data, plain, aug):
        super().__init__("Data augmentation", "")
        self.data, self.plain, self.aug = data, plain, aug
        self.hold = 2.5
        self.S = len(plain.steps)
        self.img = data.x_train[SHOW_INDEX]
        self.params = shown_transforms()
        self.copies = [affine(self.img[None], *[p[i] for p in self.params])[0]
                       for i in range(N_SHOW)]
        self.acc = dict(plain=invariance(data, plain), aug=invariance(data, aug))
        self.chart = LossChart(CHART, (1, TRAIN["steps"]), LOSS_RANGE, xlog=True)
        Tm, t = {}, 0.0
        Tm["digit"] = (t, t + 3.5)
        t += 3.5 + 0.3
        self.sweep = Sweep(t, N_SHOW, Tempo(slow=1.3, fast=0.55, ramp_in=3.0, ramp_out=2.0))
        Tm["transforms"] = (t, self.sweep.t1 + 1.2)
        t = self.sweep.t1 + 1.2 + 0.3
        Tm["training"] = (t, t + 15.0)
        t += 15.0 + 2.0
        Tm["invariance"] = (t, t + 4.0)
        t += 4.0 + 2.5
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["transforms"][0], Tm["training"][0], Tm["invariance"][0]]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else int(np.searchsorted(self.marks, t, side="right") - 1)
        k, cur, cu = self.sweep.at(t) if ph == 1 else (0, -1, 0.0)
        x = self.progress("training", t) * (self.S - 1) if ph >= 2 else 0.0
        if ph == 3:
            x = float(self.S - 1)
        grow = ease_out(self.progress("invariance", t)) if ph == 3 else 0.0
        return Frame(back, ph, k, cur, cu, x, grow)

    def at(self, run, x):
        """``(step, train loss, test loss)`` of ``run`` at fractional index ``x``."""
        i = int(round(x))
        return int(run.steps[i]), float(run.train_loss[i]), float(run.test_loss[i])

    # ------------------------------------------------------------------ drawing
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        op = f.back * smoothstep(0.0, 0.5, t)
        if f.phase <= 1:
            self._digit_and_transforms(sc, t, f, op)
        elif f.phase == 2:
            self._strips(sc, f, op)
        else:
            self._bars(sc, f, op)
        if f.phase >= 2:
            self._chart(sc, f, op)
            self._notes(sc, f, op)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _digit_and_transforms(self, sc, t, f, op):
        """The original (always), the current transform and the thumbnails."""
        c = self.P(*DIGIT_AT)
        pop = smoothstep(0.0, 0.6, self.progress("digit", t)) if f.phase == 0 else 1.0
        draw_digit(self, sc, c, self.img, BIG_PX, op * pop)
        self.label(sc, self.shift(c, 0, -BIG_PX / 2 - 10), "training image %d, label %d"
                   % (SHOW_INDEX, self.data.y_train[SHOW_INDEX]), anchor="mb", opacity=op * pop)
        self._grid(sc, op * pop)
        if f.phase != 1:
            return
        r = self.P(*RESULT_AT)
        # the arrow between the two
        a, b = self.shift(c, BIG_PX / 2 + 14, 0), self.shift(r, -BIG_PX / 2 - 14, 0)
        self.polyline(sc, [a, b], width=1.4, color=EDGE, opacity=0.8 * op)
        for s in (1, -1):
            self.polyline(sc, [b, self.shift(b, -9, 6 * s)], width=1.4, color=EDGE,
                          opacity=0.8 * op)
        self.label(sc, self.shift((a + b) / 2, 0, -12), "random affine", anchor="mb",
                   opacity=op, ink=layout.NOTE_INK)
        if 0 <= f.cur < N_SHOW:
            on = smoothstep(0.0, 0.35, f.cu) * op
            draw_digit(self, sc, r, self.copies[f.cur], BIG_PX, on, ink=AUG_INK)
            self.label(sc, self.shift((c + r) / 2, 0, BIG_PX / 2 + 10),
                       describe(*[p[f.cur] for p in self.params]), anchor="mt", opacity=on)
        # thumbnails of the transforms done so far
        n_done = min(f.k, N_SHOW)
        tc = self.P(*THUMBS_AT)
        for i in range(n_done):
            q = self.shift(tc, (i - (N_SHOW - 1) / 2) * (THUMB_PX + 4), 0)
            draw_digit(self, sc, q, self.copies[i], THUMB_PX, op, ink=AUG_INK)
        if n_done:
            self.label(sc, self.shift(tc, 0, THUMB_PX / 2 + 8), "%d of the %d copies the "
                       "network will see of it" % (n_done, TRAIN["steps"] * TRAIN["batch"]
                                                   // N_TRAIN), anchor="mt", opacity=op,
                       ink=layout.NOTE_INK)

    def _grid(self, sc, op):
        """Some of the training set (the chosen digit is among them)."""
        rows = -(-GRID_N // GRID_COLS)
        c = self.P(*GRID_AT)
        pitch = GRID_PX + 3
        for i in range(GRID_N):
            r, col = divmod(i, GRID_COLS)
            q = self.shift(c, (col - (GRID_COLS - 1) / 2) * pitch, (r - (rows - 1) / 2) * pitch)
            ink = AUG_INK if i == SHOW_INDEX else DIGIT_INK
            draw_digit(self, sc, q, self.data.x_train[i], GRID_PX, op, ink=ink)
        self.label(sc, self.shift(c, 0, -rows / 2 * pitch - 10),
                   "%d of the %d training images" % (GRID_N, N_TRAIN), anchor="mb", opacity=op)

    def _strips(self, sc, f, op):
        """The current recorded step's minibatch, as stored and as augmented."""
        i = int(round(f.x))
        step = int(self.plain.steps[i])
        c = self.P(*STRIPS_AT)
        rows = (("as stored", self.plain.probe["h"][i], DIGIT_INK),
                ("augmented", self.aug.probe["h"][i], AUG_INK))
        pitch = BATCH_PX + 4
        for r, (name, imgs, ink) in enumerate(rows):
            y = (r - 0.5) * (BATCH_PX + 34)
            for j in range(PROBE_ROWS):
                q = self.shift(c, (j - (PROBE_ROWS - 1) / 2) * pitch, y)
                draw_digit(self, sc, q, imgs[j], BATCH_PX, op, ink=ink)
            self.label(sc, self.shift(c, 0, y - BATCH_PX / 2 - 6), name, anchor="mb",
                       opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.shift(c, 0, -(BATCH_PX + 34) - BATCH_PX / 2 - 30),
                   "the minibatch at step %s (first %d of %d images)"
                   % (f"{step:,}", PROBE_ROWS, TRAIN["batch"]), anchor="mb", opacity=op)

    def _bars(self, sc, f, op):
        """Accuracy on clean, shifted and rotated test digits, both networks."""
        (x0, x1), yb, hmax = BARS
        groups = ("clean", "shifted ±%d px" % SHIFT, "rotated ±%d°" % DEGREES)
        w, gap = 44.0, 16.0
        for g, name in enumerate(groups):
            fx = x0 + (x1 - x0) * (g + 0.5) / len(groups)
            base = self.P(fx, yb)
            for s, (key, ink) in enumerate((("plain", PLAIN_INK), ("aug", AUG_INK))):
                acc = self.acc[key][g]
                h = hmax * acc * f.grow
                cb = self.shift(base, (s - 0.5) * (w + gap), 0)
                bar(self, sc, cb, w, h, ink, op)
                self.label(sc, self.shift(cb, 0, -h - 6), pct(acc), anchor="mb",
                           opacity=op * smoothstep(0.6, 1.0, f.grow), ink=layout.STRONG_INK)
            self.label(sc, self.shift(base, 0, 10), name, anchor="mt", opacity=op)
        # baseline and legend
        self.polyline(sc, [self.P(x0, yb), self.P(x1, yb)], width=1.0, color=EDGE,
                      opacity=0.8 * op)
        for s, (text, ink) in enumerate((("without augmentation", PLAIN_INK),
                                         ("with augmentation", AUG_INK))):
            q = self.P(x0 + 0.02, yb - hmax / self.cam.height / self.cam.zoom - 0.10 + 0.045 * s)
            self.rect(sc, q, 12, 12, ink, op, radius=2.0)
            self.label(sc, q, text, anchor="lm", dx=12, opacity=op)
        self.label(sc, self.P((x0 + x1) / 2, yb - hmax / self.cam.height / self.cam.zoom - 0.15),
                   "test accuracy (%s digits; %s transformed)"
                   % (f"{N_TEST:,}", f"{N_INVARIANCE:,}"), anchor="mb", opacity=op)

    def _chart(self, sc, f, op):
        """Training and test loss of both runs, drawn as training plays."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(s, f"{s:,}") for s in (1, 10, 100, 1000)]
        self.chart.axes(sc, self.cam, place, op, "training steps", "loss", ticks,
                        [(v, "%g" % v) for v in (0, 1, 2)])
        i = int(np.floor(f.x))
        a = f.x - i
        steps = self.plain.steps
        upto = steps[i] if i + 1 >= self.S else steps[i] * (1 - a) + steps[i + 1] * a
        upto = max(float(upto), 1.0)
        for run, ink in ((self.plain, PLAIN_INK), (self.aug, AUG_INK)):
            self.chart.curve(sc, place, run.steps, run.train_loss, upto, ink, 0.45 * op,
                             width=1.0, dot=False)
            self.chart.curve(sc, place, run.steps, run.test_loss, upto, ink, op)
        self.chart.legend(sc, self.cam, [("test loss, no augmentation", PLAIN_INK),
                                         ("test loss, augmented", AUG_INK),
                                         ("training losses, faint", layout.NOTE_INK)],
                          op, (CHART[0][1] - 0.01, CHART[1][0] + 0.03), step=0.038, anchor="rm")

    def _notes(self, sc, f, op):
        """The step and the losses at it."""
        step, ptr, pte = self.at(self.plain, f.x)
        _s, atr, ate = self.at(self.aug, f.x)
        lines = [note("step %s" % f"{step:,}"),
                 strong("without: training loss %.4f   test loss %.2f" % (ptr, pte)),
                 strong("with:      training loss %.4f   test loss %.2f" % (atr, ate))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1], lines, op, step=0.052)

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        plain, aug = self.plain, self.aug
        end = self.S - 1
        low = int(np.argmin(plain.test_loss))
        pa, aa = self.acc["plain"], self.acc["aug"]
        return [
            "%d MNIST training images, ten classes, a %d-%d-%d ReLU network: too few, it will "
            "memorise them" % (N_TRAIN, *SIZES),
            "every time a digit enters a minibatch it is shifted up to ±%d px, rotated up to "
            "±%d°, scaled by up to ±%d%%, at random" % (SHIFT, DEGREES, round(100 * SCALE)),
            "%d epochs of Adam: without augmentation the training loss hits %.4f and the test "
            "loss climbs from %.2f (step %d) to %.2f; with it %.2f"
            % (TRAIN["steps"] * TRAIN["batch"] // N_TRAIN, plain.train_loss[end],
               plain.test_loss[low], plain.steps[low], plain.test_loss[end], aug.test_loss[end]),
            "test accuracy %s → %s; on shifted test digits %s → %s, on rotated ones %s → %s: "
            "the invariance was learned" % (pct(pa[0]), pct(aa[0]), pct(pa[1]), pct(aa[1]),
                                             pct(pa[2]), pct(aa[2])),
        ][f.phase]
