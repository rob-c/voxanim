"""Train short, test long: six ways of telling a transformer where it is.

Six identical tiny causal transformers (:mod:`voxanim.posenc`: one layer, 4
heads, width 32) differ only in how they see position: none at all, a
learned table, the sinusoidal table, RoPE, ALiBi and T5's relative bias.
Each learns an order-2 Markov language over 8 symbols on sequences of 32 and
is tested on sequences of 128.  Four phases:

1. **the task** -- a sampled sequence with the two symbols that decide the
   next one bracketed, and the true distribution of the next symbol
   (:func:`posenc.markov_table`); the best possible loss knowing the last
   two, knowing only the current one and knowing nothing
   (:func:`posenc.entropies`); the six models,
2. **training** -- the six training-loss curves (``Run.train_loss``, a
   moving average of 50 steps) over the 4,000 steps, against the best
   possible loss,
3. **the test** -- the loss at every position of 512 test sequences of 128
   (``Run.test_loss``), the region past 32 shaded: no training sequence
   reached it.  Curves revealed left to right,
4. **the table** -- the mean loss inside 32 and beyond for each scheme,
   ranked, with what each scheme is, and the lesson: what extrapolates
   depends only on the offset and has a trained answer for every offset it
   meets (T5's last bucket, ALiBi's fixed slope); learned and sinusoidal
   tables meet positions, and RoPE angles, never trained on.  The
   previous-token weights quoted for T5 and ALiBi are measured on 16 test
   sequences with :func:`posenc.attention`.

Every number is read from :func:`posenc.runs` and :func:`posenc.entropies`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout, posenc
from .diagram import Diagram
from .easing import ease_in_out, ease_out, remap, smoothstep

N_SHOW = 18                         # symbols of the example sequence
SEQ_SEED = 3
SMOOTH = 50                         # steps in the training-loss moving average
WINDOWS = ((2, 31), (32, 63), (64, 127))   # by the position of the predicted symbol
N_ATT = 16                          # sequences behind the quoted attention weights
ATT_SEED = 7
INKS = {"none": (0.45, 0.47, 0.49), "learned": C.rgb("#d84e4e"),
        "sinusoidal": C.rgb("#a1814b"), "rope": C.rgb("#6370eb"),
        "alibi": C.rgb("#3d9e45"), "t5": C.rgb("#be3c98")}
WHAT = {"none": ("nothing added", "", "no position"),
        "learned": ("a trained row", "per index", "absolute"),
        "sinusoidal": ("fixed sin, cos", "per index", "absolute"),
        "rope": ("q, k turned by", "the position", "relative"),
        "alibi": ("score − slope", "× distance", "relative"),
        "t5": ("a learned bias", "per offset bucket", "relative")}
KIND_TEXT = {"none": "nothing", "learned": "absolute, learned", "sinusoidal": "absolute, fixed",
             "rope": "relative, fixed angles", "alibi": "relative, fixed slope",
             "t5": "relative, learned buckets"}
BEST_INK = (0.14, 0.16, 0.18)
AXIS = (0.5, 0.52, 0.54)
SHADE = C.rgb("#e9d6cf")
SYM_PX = 30.0
SEQ_Y = 0.33
TRAIN_PLOT = (0.09, 0.33, 0.66, 0.80)
TEST_PLOT = (0.09, 0.33, 0.66, 0.80)
LEGEND_X = 0.72


def window_means(curve):
    """Mean of a per-position loss over each of :data:`WINDOWS`."""
    return tuple(float(curve[a - 1:b].mean()) for a, b in WINDOWS)


def smooth(x, k=SMOOTH):
    """Moving average over ``k`` steps (``len(x) − k + 1`` values)."""
    return np.convolve(x, np.ones(k) / k, "valid")


def previous_weight(run, rows, seqs):
    """Each head's mean attention on the previous token over query ``rows``."""
    out = np.zeros(posenc.HEADS)
    for x in seqs:
        A = posenc.attention(run, x)
        out += np.array([A[h, rows, rows - 1].mean() for h in range(posenc.HEADS)])
    return out / len(seqs)


class LengthExtrapolation(Diagram):
    """Six position schemes trained at length 32 and tested at 128."""

    PHASES = ("the task", "training", "the test", "the table")

    def __init__(self):
        super().__init__("Train short, test long", "", 10.0)
        self.hold = 3.0
        self.runs = {r.kind: r for r in posenc.runs()}
        self.h2, self.h1, self.h0 = posenc.entropies()
        self.table = posenc.markov_table()
        self.seq = posenc.sample(1, N_SHOW, np.random.default_rng(SEQ_SEED))[0]
        self.train = {k: smooth(r.train_loss) for k, r in self.runs.items()}
        self.final = {k: float(r.train_loss[-200:].mean()) for k, r in self.runs.items()}
        self.means = {k: window_means(r.test_loss) for k, r in self.runs.items()}
        self.rank = sorted(posenc.KINDS, key=lambda k: self.means[k][2])
        # how close the good ones get inside 32, and what T5 and ALiBi attend to
        self.near = [k for k in posenc.KINDS if self.means[k][0] - self.h2 < 0.05]
        self.near_gap = max(self.means[k][0] - self.h2 for k in self.near)
        self.final_near = [k for k in posenc.KINDS if self.final[k] - self.h2 < 0.05]
        self.final_gap = max(self.final[k] - self.h2 for k in self.final_near)
        bucket = posenc.t5_bucket(np.arange(posenc.TRAIN_LEN), posenc.T5_BUCKETS, posenc.T5_MAX)
        self.t5_last = int(np.argmax(bucket == posenc.T5_BUCKETS - 1))
        seqs = posenc.sample(N_ATT, posenc.TEST_LEN, np.random.default_rng(ATT_SEED))
        inside = np.arange(2, posenc.TRAIN_LEN)
        far = np.arange(2 * posenc.TRAIN_LEN, posenc.TEST_LEN)
        self.att = {}
        for k in ("t5", "alibi"):
            a_in = previous_weight(self.runs[k], inside, seqs)
            a_far = previous_weight(self.runs[k], far, seqs)
            h = int(np.argmax(a_in))
            self.att[k] = (h, float(a_in[h]), float(a_far[h]))
        self.alibi_slope = float(posenc.alibi_slopes(posenc.HEADS)[self.att["alibi"][0]])

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("task", 14.0, 1.0)
        add("train", 11.0, 1.0)
        add("test", 13.0, 1.0)
        add("table", 13.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["train"][0] - 0.5, Tm["test"][0] - 0.5, Tm["table"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        Tm = self.Tm
        ends = {"task": Tm["train"][0] - 0.5, "train": Tm["test"][0] - 0.5,
                "test": Tm["table"][0] - 0.5}
        starts = {"train": Tm["train"][0] - 0.4, "test": Tm["test"][0] - 0.4,
                  "table": Tm["table"][0] - 0.4}
        op = {}
        op["task"] = 1.0 - smoothstep(ends["task"] - 0.6, ends["task"], t)
        for key in ("train", "test"):
            op[key] = smoothstep(starts[key], starts[key] + 0.4, t) * (
                1.0 - smoothstep(ends[key] - 0.6, ends[key], t))
        op["table"] = smoothstep(starts["table"], starts["table"] + 0.4, t)
        return {k: v * back for k, v in op.items()}

    def _step_k(self, task):
        """The index of the symbol being predicted in the example sequence."""
        u = remap(task, 0.04, 0.55)
        return 2 + min(N_SHOW - 3, int(u * (N_SHOW - 2)))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        task = self.progress("task", t)
        train = self.progress("train", t)
        test = self.progress("test", t)
        table = self.progress("table", t)
        k = self._step_k(task) if reset == 0 else self._step_k(0.0)
        a, b, c = (int(v) for v in self.seq[k - 2:k + 1])
        m = self.means
        best = self.rank
        self.subtitle = [
            "an order-2 Markov language over 8 symbols: the next depends on the last two. "
            "After %d %d the next is %d, which had probability %.2f"
            % (a, b, c, self.table[a, b, c]),
            "4000 steps on sequences of %d: %d of the six end within %.2f of the best "
            "possible %.2f; ALiBi at %.2f, no positions at %.2f"
            % (posenc.TRAIN_LEN, len(self.final_near), self.final_gap, self.h2,
               self.final["alibi"], self.final["none"]),
            "tested at %d: inside %d, %d schemes are within %.2f of the best %.2f. Past it "
            "only T5 (%.2f) and ALiBi (%.2f) hold; RoPE reaches %.2f"
            % (posenc.TEST_LEN, posenc.TRAIN_LEN, len(self.near), self.near_gap, self.h2,
               m["t5"][2], m["alibi"][2], m["rope"][2]),
            "ranked by loss at 64–127: " + ", ".join(
                "%s %.2f" % (posenc.LABELS[q], m[q][2]) for q in best) +
            "; best possible %.2f" % self.h2,
        ][ph]
        if ops["task"] > 0.01:
            self._task(sc, task, k, ops["task"])
        if ops["train"] > 0.01:
            self._training(sc, train, ops["train"])
        if ops["test"] > 0.01:
            self._test(sc, test, ops["test"])
        if ops["table"] > 0.01:
            self._table(sc, table, ops["table"])
        self.step_indicator(sc, ph, self.PHASES)

    # -- helpers ------------------------------------------------------------------------------
    def _dashed(self, sc, a, b, ink, op, width=1.2, dash=6.0):
        """A dashed line from ``a`` to ``b`` (dash length in px)."""
        n = max(1, int(np.linalg.norm(self.px(b) - self.px(a)) / dash))
        for j in range(0, n, 2):
            sc.line(a + (b - a) * j / n, a + (b - a) * min(j + 1, n) / n, width=width,
                    color=ink, opacity=op, bias=0.3)

    def _axes(self, sc, box, op, ymin, ymax, yticks, xticks, xmax, xlabel, ylabel, x_of,
              above=(False, True)):
        """Axes, tick labels and the best-possible and current-symbol-only lines."""
        x0, y0, x1, y1 = box

        def pt(x, v):
            return self.P(x0 + (x1 - x0) * x_of(x),
                          y1 - (y1 - y0) * (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin))

        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, AXIS, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, AXIS, 0.7 * op)
        for v in yticks:
            self.label(sc, pt(xticks[0], v), "%.1f" % v if v % 1 else "%.0f" % v, 0.024,
                       anchor="rm", dx=-6, opacity=op, ink=layout.NOTE_INK)
        for x in xticks[1:]:
            self.label(sc, pt(x, ymin), str(x), 0.024, anchor="mt", dy=5, opacity=op,
                       ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), xlabel, 0.026, anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.02), ylabel, 0.026, anchor="lb", opacity=op)
        for v, text, ink, up in ((self.h2, "best possible %.2f", BEST_INK, above[0]),
                                 (self.h1, "current symbol only %.2f", layout.NOTE_INK,
                                  above[1])):
            self._dashed(sc, pt(xticks[0], v), pt(xmax, v), ink, 0.8 * op)
            if up is None:              # labelled in the legend instead
                continue
            self.label(sc, pt(xmax, v), text % v, 0.024, anchor="rb" if up else "rt",
                       dy=-3 if up else 3, opacity=op, ink=tuple(float(c) for c in ink))
        return pt

    def _legend(self, sc, op, values=None, fmt="%.2f", order=posenc.KINDS, y0=0.36):
        """Names in their inks down the right, with a value each."""
        for n, k in enumerate(order):
            y = y0 + 0.062 * n
            ink = tuple(float(v) for v in INKS[k])
            self.polyline(sc, [self.P(LEGEND_X, y), self.P(LEGEND_X + 0.025, y)], 2.4, ink, op)
            self.label(sc, self.P(LEGEND_X + 0.035, y), posenc.LABELS[k], 0.026, anchor="lm",
                       opacity=op, ink=ink)
            if values is not None:
                self.label(sc, self.P(0.95, y), fmt % values[k], 0.026, anchor="rm",
                           opacity=op, ink=layout.STRONG_INK)

    # -- phase 1: the task ---------------------------------------------------------------------
    def _task(self, sc, task, k, op):
        """The example sequence, the two symbols that decide the next, its distribution, the
        three entropies and the six models."""
        show = smoothstep(0.0, 0.05, task) * op
        xs = np.linspace(0.08, 0.92, N_SHOW)
        stepping = task < 0.6
        for j, (x, s) in enumerate(zip(xs, self.seq)):
            c = self.P(x, SEQ_Y)
            seen = j <= k if stepping else True
            fill = C.lighten(C.FILTERS[int(s)], 0.55)
            border = None
            if stepping and j in (k - 2, k - 1):
                border = (0.20, 0.22, 0.24)
            elif stepping and j == k:
                border = C.rgb("#d84e4e")
            self.rect(sc, c, SYM_PX, SYM_PX, fill, show * (1.0 if seen else 0.35),
                      border=border, border_w=2.2, radius=5)
            self.label(sc, c, str(int(s)), 0.029, opacity=show * (1.0 if seen else 0.35),
                       ink=layout.STRONG_INK)
        if stepping:
            fade = 1.0 - smoothstep(0.55, 0.6, task)
            # an arc from the two symbols to the next
            mid = self.shift(self.P((xs[k - 2] + xs[k - 1]) / 2, SEQ_Y), 0, -SYM_PX / 2 - 4)
            end = self.shift(self.P(xs[k], SEQ_Y), 0, -SYM_PX / 2 - 4)
            pts = [mid + (end - mid) * u + self.Mi @ np.array([0.0, -26.0 * 4 * u * (1 - u)])
                   for u in np.linspace(0, 1, 16)]
            self.polyline(sc, pts, 1.6, C.rgb("#d84e4e"), show * fade, bias=0.6)
            self.disk(sc, end, 3.0, C.rgb("#d84e4e"), show * fade, lift=1.0, outline=0.0)
            # the next symbol's distribution
            a, b, nxt = (int(v) for v in self.seq[k - 2:k + 1])
            p = self.table[a, b]
            bx0, bw, base, hmax = 0.10, 0.034, 0.62, 0.15
            self.label(sc, self.P(bx0 - 0.01, 0.43), "after %d %d, the next symbol:" % (a, b),
                       0.026, anchor="lm", opacity=show * fade, ink=layout.STRONG_INK)
            for s in range(posenc.VOCAB):
                x = bx0 + bw * s
                h = p[s] * hmax
                c = self.P(x, base - h / 2)
                px_h = max(1.0, float(self.px(self.P(x, base))[1] -
                                      self.px(self.P(x, base - h))[1]))
                ink = C.FILTERS[s] if s == nxt else C.lighten(C.FILTERS[s], 0.35)
                self.rect(sc, c, 20.0, px_h, ink, show * fade, radius=1.5)
                self.label(sc, self.P(x, base), str(s), 0.024, anchor="mt", dy=4,
                           opacity=show * fade, ink=layout.NOTE_INK)
            self.label(sc, self.P(bx0 - 0.01, base + 0.065), "probability of each, from the "
                       "language's table", 0.025, anchor="lm", opacity=show * fade,
                       ink=layout.NOTE_INK)
        # the three entropies
        ent = smoothstep(0.25, 0.35, task) * op
        rows = (("the loss a model can reach, in nats:", layout.NOTE_INK),
                ("knowing the last two: %.2f (the best)" % self.h2, BEST_INK),
                ("knowing the current one only: %.2f" % self.h1, layout.CAPTION_INK),
                ("knowing nothing: %.2f" % self.h0, layout.CAPTION_INK))
        for n, (text, ink) in enumerate(rows):
            self.label(sc, self.P(0.54, 0.46 + 0.045 * n), text, 0.026, anchor="lm",
                       opacity=ent, ink=ink)
        fade = 1.0 - smoothstep(0.55, 0.6, task)
        self.label(sc, self.P(0.54, 0.645), "to use the symbol before the current", 0.025,
                   anchor="lm", opacity=ent * fade, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.54, 0.69), "one, a model must know where it is", 0.025,
                   anchor="lm", opacity=ent * fade, ink=layout.NOTE_INK)
        # the six models
        mods = smoothstep(0.6, 0.68, task) * op
        self.label(sc, self.P(0.5, 0.70), "six transformers, 1 layer, 4 heads, width %d: "
                   "identical but for position" % posenc.D, 0.027, opacity=mods,
                   ink=layout.STRONG_INK)
        cx = np.linspace(0.11, 0.89, len(posenc.KINDS))
        for n, (kind, x) in enumerate(zip(posenc.KINDS, cx)):
            on = smoothstep(0.62 + 0.03 * n, 0.68 + 0.03 * n, task) * op
            ink = tuple(float(v) for v in INKS[kind])
            line1, line2, tag = WHAT[kind]
            self.label(sc, self.P(x, 0.76), posenc.LABELS[kind], 0.026, opacity=on, ink=ink)
            self.label(sc, self.P(x, 0.81), line1, 0.024, opacity=on)
            self.label(sc, self.P(x, 0.853), line2, 0.024, opacity=on)
            self.label(sc, self.P(x, 0.905), tag, 0.024, opacity=on, ink=layout.NOTE_INK)

    # -- phase 2: training ----------------------------------------------------------------------
    def _training(self, sc, prog, op):
        """The six smoothed training-loss curves, revealed left to right."""
        n = len(self.train["none"])
        top = max(float(c[0]) for c in self.train.values())
        ymax = float(np.ceil(top * 5) / 5)
        ymin = 1.0
        last = posenc.STEPS

        pt = self._axes(sc, TRAIN_PLOT, op, ymin, ymax, (1.0, 1.5, 2.0), (0, 1000, 2000, 3000,
                        4000), last, "training step", "training loss, nats (mean of %d steps)"
                        % SMOOTH, lambda x: x / last)
        reveal = ease_out(remap(prog, 0.05, 0.75))
        k = max(2, int(reveal * n))
        idx = np.unique(np.linspace(0, k - 1, min(k, 200)).astype(int))
        for kind in posenc.KINDS:
            c = self.train[kind]
            ink = tuple(float(v) for v in INKS[kind])
            pts = [pt(i + SMOOTH, c[i]) for i in idx]
            self.polyline(sc, pts, 1.6, ink, op, bias=0.6)
            self.disk(sc, pts[-1], 2.6, ink, op, lift=1.0, outline=0.0)
        sc.probe(("train",), pt(idx[-1] + SMOOTH, 1.0), tol=6.0)
        done = smoothstep(0.75, 0.85, prog) * op
        self.label(sc, self.P(LEGEND_X, 0.31), "last 200 steps", 0.025, anchor="lm",
                   opacity=done, ink=layout.NOTE_INK)
        order = sorted(posenc.KINDS, key=lambda q: self.final[q])
        self._legend(sc, op, None, order=order)
        for n_, q in enumerate(order):
            self.label(sc, self.P(0.95, 0.36 + 0.062 * n_), "%.2f" % self.final[q], 0.026,
                       anchor="rm", opacity=done, ink=layout.STRONG_INK)

    # -- phase 3: the test ----------------------------------------------------------------------
    def _test(self, sc, prog, op):
        """Per-position test loss of the six at length 128, the untrained region shaded."""
        n = posenc.TEST_LEN - 1
        top = max(float(r.test_loss.max()) for r in self.runs.values())
        ymin, ymax = 0.8, float(np.ceil(top * 2) / 2) + 0.5
        x0, y0, x1, y1 = TEST_PLOT
        A = self.P(x0 + (x1 - x0) * (posenc.TRAIN_LEN - 0.5 - 1) / (n - 1), y0)
        B = self.P(x1, y1)
        a, b = self.px(A), self.px(B)
        self.rect(sc, (A + B) / 2, b[0] - a[0], b[1] - a[1], SHADE, 0.55 * op, radius=0.0,
                  lift=-0.5)
        self.label(sc, (A + self.P(x1, y0)) / 2, "never trained on", 0.025, anchor="mt", dy=4,
                   opacity=op, ink=layout.NOTE_INK)
        pt = self._axes(sc, TEST_PLOT, op, ymin, ymax, tuple(np.arange(1.0, ymax - 0.25, 1.0)),
                        (1, 32, 64, 96, 127), n, "position of the predicted symbol",
                        "test loss per position, nats (%d sequences of %d)"
                        % (posenc.N_EVAL, posenc.TEST_LEN), lambda x: (x - 1) / (n - 1),
                        above=(False, None))
        reveal = ease_in_out(remap(prog, 0.05, 0.7))
        k = max(2, int(reveal * n))
        for kind in posenc.KINDS:
            c = self.runs[kind].test_loss
            ink = tuple(float(v) for v in INKS[kind])
            pts = [pt(i + 1, c[i]) for i in range(k)]
            self.polyline(sc, pts, 1.4, ink, op, bias=0.6)
            self.disk(sc, pts[-1], 2.6, ink, op, lift=1.0, outline=0.0)
        sc.probe(("test",), pt(k, 1.0), tol=6.0)
        done = smoothstep(0.72, 0.82, prog) * op
        self.label(sc, self.P(LEGEND_X, 0.31), "mean at 64–127", 0.025, anchor="lm",
                   opacity=done, ink=layout.NOTE_INK)
        order = self.rank
        self._legend(sc, op, None, order=order)
        y = 0.36 + 0.062 * len(order)
        self._dashed(sc, self.P(LEGEND_X, y), self.P(LEGEND_X + 0.025, y), layout.NOTE_INK, op,
                     dash=4.0)
        self.label(sc, self.P(LEGEND_X + 0.035, y), "current only", 0.026, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.95, y), "%.2f" % self.h1, 0.026, anchor="rm", opacity=op,
                   ink=layout.NOTE_INK)
        for n_, q in enumerate(order):
            self.label(sc, self.P(0.95, 0.36 + 0.062 * n_), "%.2f" % self.means[q][2], 0.026,
                       anchor="rm", opacity=done, ink=layout.STRONG_INK)

    # -- phase 4: the table ---------------------------------------------------------------------
    def _table(self, sc, prog, op):
        """Mean test loss in each window, ranked by the last, and the lesson."""
        cols = (0.50, 0.59, 0.68)
        bar_x0, bar_w = 0.73, 0.21
        top = 0.33
        row = 0.058
        hdr = smoothstep(0.0, 0.08, prog) * op
        self.label(sc, self.P(0.06, top), "scheme", 0.025, anchor="lm", opacity=hdr,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(0.24, top), "how it sees position", 0.025, anchor="lm",
                   opacity=hdr, ink=layout.NOTE_INK)
        for x, (a, b) in zip(cols, WINDOWS):
            self.label(sc, self.P(x, top), "%d–%d" % (a, b), 0.025, opacity=hdr,
                       ink=layout.NOTE_INK)
        self.label(sc, self.P(bar_x0, top), "loss at 64–127", 0.025, anchor="lm",
                   opacity=hdr, ink=layout.NOTE_INK)
        vmax = max(self.means[k][2] for k in posenc.KINDS)
        for n, kind in enumerate(self.rank):
            y = top + row * (n + 1)
            on = smoothstep(0.05 + 0.06 * n, 0.12 + 0.06 * n, prog) * op
            ink = tuple(float(v) for v in INKS[kind])
            self.label(sc, self.P(0.06, y), posenc.LABELS[kind], 0.026, anchor="lm", opacity=on,
                       ink=ink)
            self.label(sc, self.P(0.24, y), KIND_TEXT[kind], 0.025, anchor="lm", opacity=on)
            for x, v in zip(cols, self.means[kind]):
                self.label(sc, self.P(x, y), "%.2f" % v, 0.026, opacity=on,
                           ink=layout.STRONG_INK)
            w = bar_w * self.means[kind][2] / vmax
            a, b = self.px(self.P(bar_x0, y)), self.px(self.P(bar_x0 + w, y))
            self.rect(sc, self.P(bar_x0 + w / 2, y), b[0] - a[0], 12.0, INKS[kind], on,
                      radius=2.0)
        # the best possible loss as a tick across the bars
        best_x = bar_x0 + bar_w * self.h2 / vmax
        tick = smoothstep(0.45, 0.5, prog) * op
        self._dashed(sc, self.P(best_x, top + row * 0.5), self.P(best_x, top + row * 6.5),
                     BEST_INK, tick, dash=4.0)
        self.label(sc, self.P(best_x, top + row * 6.5), "best %.2f" % self.h2, 0.024,
                   anchor="mt", dy=2, opacity=tick, ink=BEST_INK)
        _h, a_in, a_far = self.att["t5"]
        lessons = [
            ("extrapolates: a score that depends only on the offset, with a trained answer "
             "for every offset it meets", layout.STRONG_INK),
            ("T5: offsets past %d all share the bucket training used at %d–%d; previous-token "
             "weight %.2f → %.2f" % (self.t5_last - 1, self.t5_last, posenc.TRAIN_LEN - 1,
                                     a_in, a_far), layout.CAPTION_INK),
            ("ALiBi: the same fixed slope at any distance; its steepest head (slope %.2f) keeps "
             "the previous token, %.2f → %.2f" % ((self.alibi_slope,) + self.att["alibi"][1:]),
             layout.CAPTION_INK),
            ("fails: learned and sinusoidal meet positions never trained on, RoPE angles "
             "never trained on", layout.STRONG_INK),
        ]
        for n, (text, ink) in enumerate(lessons):
            on = smoothstep(0.55 + 0.08 * n, 0.62 + 0.08 * n, prog) * op
            self.label(sc, self.P(0.06, 0.775 + 0.042 * n), text, 0.025, anchor="lm",
                       opacity=on, ink=ink)
