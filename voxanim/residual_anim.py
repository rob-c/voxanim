"""Skip connections: a 30-layer plain MLP against the same network with
residual connections, the depth-wise twin of :mod:`voxanim.gated_anim`.

    784 -> 16 x 30 (ReLU) -> 10, Xavier init, plain SGD on 10,000 MNIST images

The residual network adds a skip every two layers, ``h -> f(h) + h`` (the
ResNet block); nothing else differs.  Each is drawn as a band of 16 units x
30 layers, one above the other, with one test digit going through the
untrained weights:

1. **networks** -- the two bands and the skips,
2. **forward** -- the activations layer by layer: the plain network's fade
   (std 0.23 at layer 1, 6×10⁻⁷ at layer 30: each ReLU layer under Xavier
   halves the variance), the residual network's grow (to 2.9),
3. **backward** -- the gradient of the loss flowing back, drawn as its
   magnitude on a log scale: through the plain network it shrinks to
   5×10⁻⁶ of itself before it reaches layer 1; along the skips it arrives
   13 times larger than it left,
4. **training** -- 1500 SGD steps: the plain network never leaves chance,
   the residual one reaches 90% test accuracy.

The setting was chosen so that the contrast is real: at this depth and
width a plain ReLU network with Xavier weights does not train at all (with
He weights it crawls, see the README), while a skip every layer without
normalisation overflows.  Every number comes from recorded runs of
:mod:`voxanim.deepnet`; the bands show one example through the initial
weights ``P0`` of those runs.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import Spec, backward, forward, subset, train
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10
from .netanim import ACT_HI, NetAnimation, lit_colors, particle

DEPTH, WIDTH = 30, 16
SIZES = (784,) + (WIDTH,) * DEPTH + (10,)
NETS = (("plain", Spec(SIZES, "relu", "xavier")),
        ("residual", Spec(SIZES, "relu", "xavier", residual=2)))
TRAIN = dict(steps=1500, batch=64, lr=0.05, opt="sgd", every=50)
N_TRAIN, N_TEST = 10000, 2000
EXAMPLE = 0                                   # the test image of the subset drawn through
PLAIN_INK = (0.75, 0.22, 0.22)
RES_INK = tuple(float(v) for v in C.INPUT)
INKS = (PLAIN_INK, RES_INK)
DARK, GRAD_HI = C.rgb("#3a3f44"), C.rgb("#c9fff6")      # a log colour scale: dark .. lit
ACT_RANGE = (-6.0, 0.5)                       # log10 activation at the dark and the lit end
GRAD_RANGE = (-9.0, 1.0)                      # log10 |gradient| at the dark and the lit end
ROWS = (("plain", "plain: 30 layers", 0.0), ("residual", "residual: skip every 2", 4.0))
STACK_AT = (0.42, 0.58)                       # frame position of the stack's centre
PITCH = (1.0, 1.25)                           # cell pitch (units, layers)
SCALE = 0.9
PPU = 10.5
ARC = 1.6                                     # height of the skip arcs above the band, world units
ARC_INK = (0.40, 0.42, 0.45)
NOTE_X = 0.655                                # frame x of the read-outs right of the bands
CHART = ((0.70, 0.94), (0.33, 0.78))          # the training chart ((x0, x1), (top, bottom))
SMOOTH = 20                                   # batches the drawn loss is averaged over
FRAME = (0.96, 0.96, 0.96)
CHANCE = np.log(10.0)                         # cross-entropy of guessing: ln 10 = 2.30


def runs(cache_dir=None):
    """``(runs, x, label)``: the two recorded runs and the example digit."""
    data = subset(N_TRAIN, N_TEST, seed=0, cache_dir=cache_dir)
    kw = dict(TRAIN)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    rs = [train(spec, data, **kw) for _name, spec in NETS]
    return rs, data.x_test[EXAMPLE], int(data.y_test[EXAMPLE])


def log_scale(v, lo_hi):
    """``|v|`` on a log scale: 0 at ``10 ** lo`` and below (an exact 0 stays 0,
    which :func:`~voxanim.netanim.lit_colors` draws as OFF), 1 at ``10 ** hi``."""
    lo, hi = lo_hi
    return np.clip((np.log10(np.abs(v) + 1e-300) - lo) / (hi - lo), 0.0, 1.0)


def example_through(run, x, label):
    """``(h (WIDTH, DEPTH), dh (WIDTH, DEPTH))``: one digit's hidden activations
    and the gradient of its loss at each hidden layer, through ``run.P0``."""
    f = forward(run.spec, run.P0, x[None])
    _G, D = backward(run.spec, run.P0, f, np.eye(10)[[label]])
    h = np.array([f["h"][l + 1][0] for l in range(DEPTH)]).T
    dh = np.array([D["dh"][l + 1][0] for l in range(DEPTH)]).T
    return h, dh


class SkipConnections(NetAnimation):
    """A plain and a residual deep MLP side by side (see the module docstring).

    ``runs`` are the two :class:`voxanim.deepnet.Run` s of :data:`NETS`, ``x``
    and ``label`` the example digit.
    """

    PHASES = ("networks", "forward", "backward", "training")

    def __init__(self, runs, x, label):
        super().__init__("Skip connections", "", PPU)
        self.runs = runs
        self.hold = 2.5
        self.rows = {}
        centre = self.cam.frac(*STACK_AT)
        height = sum(WIDTH + gap for _k, _l, gap in ROWS)
        z = centre[2] + height / 2 * SCALE
        for key, label_, gap in ROWS:
            s = self.sheet((WIDTH, DEPTH), STACK_AT, SCALE, key, pitch=PITCH)
            z -= (gap + WIDTH / 2) * SCALE
            c = centre.copy()
            c[2] = z
            s.block.place(c)
            z -= WIDTH / 2 * SCALE
            self.rows[key] = (s, label_)
        self.h, self.dh = {}, {}
        for (key, _l, _g), run in zip(ROWS, runs):
            self.h[key], self.dh[key] = example_through(run, x, label)
        self.act = [r.act_std[0] for r in runs]            # batch statistics at step 0
        self.grad = [r.grad_rms[0] for r in runs]
        self.ratio = [float(g[0] / g[-1]) for g in self.grad]
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = [np.convolve(r.batch_loss, kernel, "valid") for r in runs]
        self.acc = [float(r.test_acc[-1]) for r in runs]
        self.chart = LossChart(CHART, (0, TRAIN["steps"]), (0.0, 3.0))

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("networks", 4.5, 0.3)
        fw = Tempo(slow=0.55, fast=0.11, ramp_in=3.0, ramp_out=2.0).durations(DEPTH)
        self.f_starts, self.f_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("fwd", float(fw.sum()), 2.0)
        self.b_starts, self.b_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("bwd", float(fw.sum()), 2.2)
        add("train", 11.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["bwd"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """``(layers done, current layer, its progress)`` of a column sweep;
        the current layer is -1 before the sweep and ``DEPTH`` after it."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return DEPTH, DEPTH, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        a = self.act
        self.subtitle = [
            "two %d-layer ReLU MLPs (784 → %d × %d → 10), the same Xavier weights; the second "
            "adds a skip every two layers: h → f(h) + h" % (DEPTH, WIDTH, DEPTH),
            "forward: one digit through the untrained weights; the plain activations fade "
            "(std %s at layer 1, %s at layer %d), the residual ones grow to %s"
            % (pow10(a[0][0]), pow10(a[0][-1]), DEPTH, pow10(a[1][-1])),
            "backward: the gradient flowing back, log scale; reaching layer 1 it is %s of what "
            "left layer %d in the plain network, %s× along the skips"
            % (pow10(self.ratio[0]), DEPTH, pow10(self.ratio[1])),
            "training: %s SGD steps (learning rate %g); the plain network stays at chance (loss "
            "ln 10 = %.2f), the residual one reaches %d%% test accuracy"
            % (f"{TRAIN['steps']:,}", TRAIN["lr"], CHANCE, round(100 * self.acc[1])),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        intro = smoothstep(0.0, 0.6, t)
        done_f, nf, uf = self._sweep(t, "fwd", self.f_starts, self.f_durs)
        done_b, nb, ub = self._sweep(t, "bwd", self.b_starts, self.b_durs)
        train = self.progress("train", t)
        notes = (1.0 - smoothstep(0.0, 0.12, train)) * back
        self._bands(sc, t, done_f, nf, uf, done_b, nb, ub, back * intro)
        if 0 <= nf < DEPTH or 0 <= nb < DEPTH:
            self._column_frame(sc, nf if nf < DEPTH else DEPTH - 1 - nb)
        self._skips(sc, t, back * intro)
        self._notes(sc, t, done_f, done_b, notes)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the bands ---------------------------------------------------------------------
    def _bands(self, sc, t, done_f, nf, uf, done_b, nb, ub, back):
        """The two bands: activations as they are computed, then the gradient
        magnitude column by column on the way back."""
        col = np.tile(np.arange(DEPTH), WIDTH)
        for key, label in ((k, l) for k, l, _g in ROWS):
            s, _ = self.rows[key]
            h = self.h[key]
            c = lit_colors(log_scale(h, ACT_RANGE).ravel(), 1.0, DARK, ACT_HI)
            if t >= self.Tm["bwd"][0]:
                gc = lit_colors(log_scale(self.dh[key], GRAD_RANGE).ravel(), 1.0, DARK, GRAD_HI)
                got = col >= DEPTH - done_b
                if 0 <= nb < DEPTH and ub > 0.35:
                    got |= col == DEPTH - 1 - nb
                c[got] = gc[got]
            shown = col < done_f
            if 0 <= nf < DEPTH and uf >= 0.5:
                shown |= col == nf
            if t >= self.Tm["fwd"][1]:
                shown[:] = True
            s.fill(c)
            s.show(shown)
            s.cage = 1.0
            s.fade(back)
            s.block.white[:, :, 0] = (((col == nf) & (uf < 1)) * (1 - smoothstep(0.7, 1.0, uf))
                                      * 0.6).reshape(WIDTH, DEPTH)
            text = label if t < self.Tm["bwd"][0] else label.split(":")[0] + ": gradient"
            self.caption_left(sc, s, text, back)
            # the values in flight: into the next column, or back to the previous one
            if 0 < nf < DEPTH and t < self.Tm["fwd"][1]:
                m = log_scale(h, ACT_RANGE)
                for r in range(WIDTH):
                    colr = lit_colors(m[r, nf - 1:nf], 1.0, DARK, ACT_HI)[0]
                    particle(sc, s.cell(r * DEPTH + nf - 1), s.cell(r * DEPTH + nf),
                             remap(uf, 0.05, 0.45), colr, 0.5, lift=0.4)
            if 0 <= nb < DEPTH and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
                s_ = DEPTH - 1 - nb
                if s_ < DEPTH - 1:
                    m = log_scale(self.dh[key], GRAD_RANGE)
                    for r in range(WIDTH):
                        gcol = lit_colors(m[r, s_ + 1:s_ + 2], 1.0, DARK, GRAD_HI)[0]
                        particle(sc, s.cell(r * DEPTH + s_ + 1), s.cell(r * DEPTH + s_),
                                 remap(ub, 0.05, 0.45), gcol, 0.55, lift=-0.4)
        if 0 <= nf < DEPTH and t < self.Tm["fwd"][1]:
            sc.probe(("fwd",), (nf * 20 + uf * 20, 0, 0), tol=10.0)
        if 0 <= nb < DEPTH and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
            sc.probe(("bwd",), (nb * 20 + ub * 20, 0, 0), tol=10.0)

    def _column_frame(self, sc, col):
        """A white frame around layer ``col`` of both bands: the layer at work."""
        ss = [s for s, _l in self.rows.values()]
        lo = np.min([s.block.box((0, WIDTH), (col, col + 1), (0, 1))[0] for s in ss], 0)
        hi = np.max([s.block.box((0, WIDTH), (col, col + 1), (0, 1))[1] for s in ss], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=FRAME, width=1.6, opacity=0.95, bias=0.1)

    def _skips(self, sc, t, op):
        """The residual network's skips: an arc over every block of two layers
        (from the block's input column to its output column)."""
        intro = smoothstep(0.3, 0.6, self.progress("networks", t)) if t < self.Tm["fwd"][0] \
            else 1.0
        op *= intro
        if op <= 0.01:
            return
        s, _ = self.rows["residual"]
        spec = self.runs[1].spec
        for l in range(DEPTH):
            if not spec.skip(l):
                continue
            a = s.block.box((0, 1), (l - spec.residual, l - spec.residual + 1), (0, 1))
            b = s.block.box((0, 1), (l, l + 1), (0, 1))
            p0 = np.array([(a[0][0] + a[1][0]) / 2, a[1][1], a[1][2] + 0.15])
            p1 = np.array([(b[0][0] + b[1][0]) / 2, b[1][1], b[1][2] + 0.15])
            pts = [p0 + (p1 - p0) * u + np.array([0, 0, ARC * 4 * u * (1 - u)])
                   for u in np.linspace(0, 1, 7)]
            for q0, q1 in itertools.pairwise(pts):
                sc.line(q0, q1, width=1.2, color=ARC_INK, opacity=0.85 * op, bias=0.2)

    # -- read-outs ----------------------------------------------------------------------
    def _notes(self, sc, t, done_f, done_b, op):
        """Right of the bands: the activation std, then the gradient, per network."""
        op *= self._cap
        if op <= 0.01 or (done_f == 0 and done_b == 0):
            return
        for i, (key, _l, _g) in enumerate(ROWS):
            s, _ = self.rows[key]
            lo, hi = s.bbox()
            y = self.cam.project((lo + hi) / 2)[1] / self.cam.height
            if done_b > 0:
                l = max(0, DEPTH - done_b)
                head = "gradient reaching layer %d" % (l + 1)
                value = pow10(self.grad[i][l])
                tail = "(%s of layer %d's)" % (pow10(self.grad[i][l] / self.grad[i][-1]), DEPTH) \
                    if l < DEPTH - 1 else ""
            else:
                head = "activations at layer %d" % done_f
                value = "std " + pow10(self.act[i][done_f - 1])
                tail = ""
            sc.text(NOTE_X, y - 0.02, head, 0.027, "regular", layout.CAPTION_INK, op, "lm")
            sc.text(NOTE_X, y + 0.022, value, 0.031, "regular", layout.STRONG_INK, op, "lm")
            if tail:
                sc.text(NOTE_X, y + 0.062, tail, 0.026, "regular", layout.NOTE_INK, op, "lm")

    # -- the training curves ----------------------------------------------------------------
    def _chart(self, sc, train, back):
        """Both networks' training loss (batch loss averaged over SMOOTH batches)."""
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        n = len(self.curves[0])
        upto = SMOOTH + (n - 1) * smoothstep(0.0, 0.92, train)
        ticks = [(s, f"{s:,}") for s in (0, 500, 1000, 1500)]
        self.chart.axes(sc, cam, place, op, "steps (batch loss, mean of %d)" % SMOOTH, "loss",
                        ticks, [(v, "%g" % v) for v in (0, 1, 2, 3)])
        # chance: ln 10
        a, b = place(self.chart.fx(0), self.chart.fy(CHANCE)), place(self.chart.x1,
                                                                    self.chart.fy(CHANCE))
        for i in range(12):
            u0, u1 = i / 12, (i + 0.5) / 12
            sc.line(a + (b - a) * u0, a + (b - a) * u1, width=0.9, color=(0.5, 0.52, 0.54),
                    opacity=0.6 * op, bias=0.0)
        steps = np.arange(n) + SMOOTH
        for i in range(2):
            self.chart.curve(sc, place, steps, self.curves[i], upto, INKS[i], op, every=4)
        legend = [("plain: %d%% test accuracy" % round(100 * self.acc[0]), PLAIN_INK),
                  ("residual: %d%%" % round(100 * self.acc[1]), RES_INK)]
        self.chart.legend(sc, cam, legend, op * smoothstep(0.85, 1.0, train),
                          (CHART[0][1] - 0.01, CHART[1][0] + 0.19), anchor="rm")
        sc.probe(("train",), (upto * 0.1, 0, 0), tol=6.0)
