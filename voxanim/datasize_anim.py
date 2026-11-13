"""Data against model size: learning curves of three one-hidden-layer MLPs
on MNIST, every point a real training.

Three networks, 784-8-10, 784-32-10 and 784-256-10 (ReLU, He init: 6,360,
25,450 and 203,530 parameters), each trained on the first 100, 300, 1,000,
3,000, 10,000 and 60,000 images of a seeded shuffle of the training set:
eighteen runs of :func:`voxanim.deepnet.train`, every one the same short
budget (2,000 Adam steps, lr 0.001, minibatches of 64) so that only the
data and the width differ.  The test error is measured on all 10,000 test
digits.

1. **setup** -- the three networks, their hidden layers drawn to size,
2. **learning curves** -- the runs one by one, size by size: each drops its
   test error onto the chart (log size axis),
3. **small saturates** -- the 8-unit network: 25.7% error with 100 images,
   9.3% with 10,000 and 8.7% with 60,000: six times the data buys 0.6 points,
4. **bigger pays** -- the 256-unit network: 21.5% with 100 images, 2.8%
   with 60,000; with little data the three are a few points apart, with all
   of it the big one makes a third of the small one's mistakes,
5. **mistakes** -- test digits each regime gets wrong: the small network on
   60,000 images, the big one on 100 and on 60,000, with what it said.

(The numbers are this machine's; CI retrains them.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import Data, Spec, subset, train
from .diagram import Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gen_common import PAPER, WRONG_INK, draw_digit, note, readout, strong
from .generalise import error_rate, misclassified, predictions
from .losschart import LossChart
from .sweep import Sweep

SIZES_N = (100, 300, 1000, 3000, 10000, 60000)
N_TEST, SEED = 10000, 0
MODELS = (("small", 8), ("medium", 32), ("large", 256))      # (name, hidden units)
TRAIN = dict(steps=2000, batch=64, lr=0.001, opt="adam")
INKS = (C.rgb("#e0952e"), C.rgb("#1f6fb5"), C.rgb("#3d9e45"))   # small, medium, large
GLYPH_AT = (0.72, 0.50)                          # the three networks, layout centre
GLYPH_GAP = 0.17                                 # layout x between them
UNIT_PX = (12.0, 8.0, 4.0)                       # cell size of each hidden layer's grid
UNIT_COLS = (8, 16, 32)                          # ... drawn this many across
CHART = ((0.09, 0.45), (0.32, 0.74))             # the learning curves
ERR_RANGE = (0.0, 0.30)
NOTE_AT = (0.57, 0.80)                           # read-outs under the glyphs
SHOW = ((0, 5), (2, 0), (2, 5))                  # mistakes phase: (model, size index) rows
N_WRONG = 8                                      # wrong test digits per row
WRONG_PX = 30.0
WRONG_AT = (0.74, 0.55)                          # the rows' centre
WRONG_PITCH = 104.0                              # px between the rows


def params(w):
    """Parameters of a 784-w-10 network."""
    return 784 * w + w + w * 10 + 10


def compact(n):
    """``6370`` as ``6.4k``, ``203530`` as ``204k``."""
    return "%.1fk" % (n / 1000) if n < 10000 else "%dk" % round(n / 1000)


def runs(cache_dir=None):
    """``(data, table)``: the full shuffled training set and a ``(3, 6)``
    table of recorded runs, ``table[m][j]`` model ``m`` on the first
    ``SIZES_N[j]`` images."""
    full = subset(max(SIZES_N), N_TEST, None, SEED, cache_dir)
    kw = dict(TRAIN)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    table = []
    for _name, w in MODELS:
        spec = Spec((784, w, 10), "relu", "he")
        row = []
        for n in SIZES_N:
            data = Data(full.x_train[:n], full.y_train[:n], full.x_test, full.y_test, 10)
            row.append(train(spec, data, batch=min(kw["batch"], n),
                             **{k: v for k, v in kw.items() if k != "batch"}))
        table.append(row)
    return full, table


@dataclass
class Frame:
    """What one instant shows."""

    back: float
    phase: int
    done: int        # trainings finished (sweep)
    cur: int         # the one under way (-1: none)
    cu: float        # its progress
    focus: int       # model highlighted (-1: all)


class DataVsModelSize(Diagram):
    """Learning curves of three model sizes (see the module docstring)."""

    PHASES = ("setup", "learning curves", "small saturates", "bigger pays", "mistakes")

    def __init__(self, data, table):
        super().__init__("Data against model size", "")
        self.data, self.table = data, table
        self.hold = 2.5
        # test error on all 10,000 digits, (3, 6)
        self.err = np.array([[error_rate(r.spec, r.P, data.x_test, data.y_test) for r in row]
                             for row in table])
        self.wrong = [self._wrong(m, j) for m, j in SHOW]
        self.chart = LossChart(CHART, (SIZES_N[0], SIZES_N[-1]), ERR_RANGE, xlog=True)
        n_runs = len(SIZES_N) * len(MODELS)
        Tm, t = {}, 0.0
        Tm["setup"] = (t, t + 5.0)
        t += 5.0 + 0.3
        self.sweep = Sweep(t, n_runs, Tempo(slow=1.2, fast=0.45, ramp_in=4.0, ramp_out=2.0))
        Tm["curves"] = (t, self.sweep.t1 + 1.0)
        t = self.sweep.t1 + 1.0 + 0.3
        for name in ("small", "big", "mistakes"):
            Tm[name] = (t, t + 7.0)
            t += 7.0 + 0.3
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["curves"][0], Tm["small"][0], Tm["big"][0], Tm["mistakes"][0]]

    def _wrong(self, m, j):
        """``(indices, predictions)`` of the first ``N_WRONG`` test digits run
        ``(m, j)`` misclassifies."""
        r = self.table[m][j]
        pred = predictions(r.spec, r.P, self.data.x_test)
        idx = misclassified(pred, self.data.y_test)[:N_WRONG]
        return idx, pred[idx]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else int(np.searchsorted(self.marks, t, side="right") - 1)
        if ph == 0:
            return Frame(back, 0, 0, -1, 0.0, -1)
        done, cur, cu = self.sweep.at(t)
        focus = {2: 0, 3: 2}.get(ph, -1)
        return Frame(back, ph, done, cur if ph == 1 else -1, cu, focus)

    @staticmethod
    def order(k):
        """Sweep step ``k`` -> ``(model, size index)``: size by size, the three models in turn."""
        return k % len(MODELS), k // len(MODELS)

    # ------------------------------------------------------------------ drawing
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        op = f.back * smoothstep(0.0, 0.5, t)
        if f.phase < 4:
            self._glyphs(sc, t, f, op)
        else:
            self._mistakes(sc, t, f, op)
        if f.phase >= 1:
            self._chart(sc, f, op)
            self._notes(sc, f, op)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _glyphs(self, sc, t, f, op):
        """The three networks: input bar, hidden layer to size, output row."""
        pop = self.progress("setup", t) if f.phase == 0 else 1.0
        cur_m = self.order(f.cur)[0] if f.cur >= 0 else -1
        for m, ((name, w), ink) in enumerate(zip(MODELS, INKS)):
            on = smoothstep(0.0, 0.3, remap(pop, 0.3 * m, 0.3 * m + 0.4)) * op
            if f.focus >= 0 and m != f.focus:
                on *= 0.35
            c = self.P(GLYPH_AT[0] + (m - 1) * GLYPH_GAP, GLYPH_AT[1])
            cols = UNIT_COLS[m]
            rows = w // cols
            px = UNIT_PX[m]
            tex = np.broadcast_to(ink[None, None], (rows, cols, 3))
            draw_image(self, sc, c, tex, cols * px, rows * px, on, grid=0.6, lift=0.2)
            if m == cur_m:
                hw, hh = cols * px / 2 + 5, rows * px / 2 + 5
                self.rect(sc, c, 2 * hw, 2 * hh, ink, on, border=ink, border_w=1.5, radius=4.0,
                          fill_op=0.0, lift=0.1)
            half = rows * px / 2
            self.label(sc, self.shift(c, 0, -half - 10), "%s: %d units" % (name, w),
                       anchor="mb", opacity=on, ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, half + 10), "%s parameters" % compact(params(w)),
                       anchor="mt", opacity=on, ink=layout.NOTE_INK)
        self.label(sc, self.P(GLYPH_AT[0], GLYPH_AT[1] - 0.17), "784 pixels in, 10 classes "
                   "out, one hidden layer of", anchor="mb", opacity=op * pop)

    def _mistakes(self, sc, t, f, op):
        """Rows of test digits each regime gets wrong, with its answer."""
        c = self.P(*WRONG_AT)
        pitch = WRONG_PX + 6
        reveal = self.progress("mistakes", t)
        for r, ((m, j), (idx, pred)) in enumerate(zip(SHOW, self.wrong)):
            y = (r - 1) * WRONG_PITCH
            on = smoothstep(0.0, 0.3, remap(reveal, 0.25 * r, 0.25 * r + 0.3)) * op
            name = MODELS[m][0]
            self.label(sc, self.shift(c, 0, y - WRONG_PX / 2 - 22), "%s network, %s images: "
                       "%.1f%% wrong" % (name, f"{SIZES_N[j]:,}", 100 * self.err[m, j]),
                       anchor="mb", opacity=on, ink=INKS[m])
            for k, (i, p) in enumerate(zip(idx, pred)):
                q = self.shift(c, (k - (N_WRONG - 1) / 2) * pitch, y)
                draw_digit(self, sc, q, self.data.x_test[i], WRONG_PX, on, paper=PAPER)
                self.label(sc, self.shift(q, 0, WRONG_PX / 2 + 4), "%d" % p, anchor="mt",
                           opacity=on, ink=WRONG_INK)

    def _chart(self, sc, f, op):
        """Test error against training-set size, a curve per model."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(n, f"{n:,}") for n in (100, 1000, 10000, 60000)]
        self.chart.axes(sc, self.cam, place, op, "training images (log scale)", "test error",
                        ticks, [(v, "%d%%" % round(100 * v)) for v in (0.1, 0.2, 0.3)])
        done = min(f.done, len(SIZES_N) * len(MODELS))
        for m, ink in enumerate(INKS):
            k = sum(1 for q in range(done) if self.order(q)[0] == m)      # sizes done
            a = op * (0.35 if f.focus >= 0 and m != f.focus else 1.0)
            if k == 0:
                continue
            xs, ys = np.array(SIZES_N[:k], float), self.err[m, :k]
            self.chart.curve(sc, place, xs, ys, xs[-1], ink, a, dot=False)
            for x, y in zip(xs, ys):
                self.disk(sc, place(self.chart.fx(x), self.chart.fy(y)), 3.6, ink, a, lift=0.6,
                          outline=0.3)
        if 0 <= f.cur < len(SIZES_N) * len(MODELS):
            m, j = self.order(f.cur)
            x, y = SIZES_N[j], self.err[m, j]
            p = place(self.chart.fx(x), self.chart.fy(y))
            u = ease_out(f.cu)
            self.disk(sc, p, 3.6 + 6 * (1 - u), INKS[m], op * u, lift=0.7, glow=0.4 * (1 - u),
                      outline=0.3)
        self.chart.legend(sc, self.cam, [("%s, %d units" % (n, w), ink)
                                         for (n, w), ink in zip(MODELS, INKS)],
                          op, (CHART[0][1] - 0.01, CHART[1][0] + 0.03), step=0.038, anchor="rm")

    def _notes(self, sc, f, op):
        """What is training, or the numbers of the phase."""
        if f.phase == 1:
            if not 0 <= f.cur < len(SIZES_N) * len(MODELS):
                return
            m, j = self.order(f.cur)
            name, w = MODELS[m]
            lines = [note("training %d of %d" % (f.cur + 1, len(SIZES_N) * len(MODELS))),
                     strong("%s network on %s images" % (name, f"{SIZES_N[j]:,}")),
                     strong("test error %.1f%%" % (100 * self.err[m, j]))]
        elif f.phase in (2, 3):
            m = f.focus
            name, w = MODELS[m]
            lines = [strong("%s network, %d units:" % (name, w))] + [
                note("%s images: %.1f%% error" % (f"{n:,}", 100 * e))
                for n, e in zip(SIZES_N, self.err[m])]
        else:
            return
        readout(self, sc, NOTE_AT[0], NOTE_AT[1] - 0.045 * (len(lines) - 3), lines, op,
                step=0.045)

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        e = self.err
        return [
            "three ReLU networks with 8, 32 and 256 hidden units, each trained on 100 to %s "
            "MNIST images, the same %s Adam steps" % (f"{SIZES_N[-1]:,}", f"{TRAIN['steps']:,}"),
            "each training drops its error on %s test digits onto the chart: more data helps "
            "every network, but not equally" % f"{N_TEST:,}",
            "the 8-unit network: %.1f%% error with 100 images, %.1f%% with 10,000, %.1f%% with "
            "60,000: six times the data buys %.1f points"
            % (100 * e[0, 0], 100 * e[0, 4], 100 * e[0, 5], 100 * (e[0, 4] - e[0, 5])),
            "the 256-unit network: %.1f%% with 100 images, %.1f%% with 60,000; with all the "
            "data it makes a third of the small one's mistakes"
            % (100 * e[2, 0], 100 * e[2, 5]),
            "the first eight test digits each run gets wrong, its answer in red: too small a "
            "model, too little data, and the hard ones that remain",
        ][f.phase]

