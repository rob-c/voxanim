"""Vision: animations 51-58 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the mechanism on a real MNIST instance first and the measurement
after it. The numbers are the study's (:mod:`voxanim.next_vision`, and
:mod:`voxanim.study_vision` for the three experiments that were redesigned:
52, 57 and 58); what a mechanism needs beyond them (per-step states, feature
maps, a second forward pass) is computed here from the study's own weights
and functions.
"""

from __future__ import annotations

import numpy as np

from . import next_vision as nv
from . import next_vision_models as vm
from . import study_vision as sv
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    PAPER,
    RED,
    TEAL,
    TEXT,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    heat,
    soft,
)

INK = (0.16, 0.17, 0.19)


def study_arrays(name):
    """The arrays the study ``name`` cached (its run: weights, per-example
    results), by listening to the study's own call for them."""
    got, real = {}, nv.cached

    def spy(key, build, **settings):
        got["d"] = real(key, build, **settings)
        return got["d"]
    nv.cached = spy
    try:
        nv._BUILDERS[name]()
    finally:
        nv.cached = real
    return got["d"]


def paper(v):
    """A grey image in [0, 1] as dark ink on paper, ``(rows, cols, 3)``."""
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = v.reshape(int(round(np.sqrt(v.size))), -1)
    v = np.clip(v, 0, 1)[..., None]
    return PAPER * (1 - v) + np.asarray(DIGIT_INK) * v


def tint(rgb, amount, ink):
    """``rgb`` moved towards ``ink`` by ``amount`` (per pixel, 0..1)."""
    a = np.clip(np.asarray(amount, float), 0, 1)[..., None]
    return np.asarray(rgb) * (1 - a) + np.asarray(ink) * a


class Vision(Mechanism):
    """What the vision animations share: pictures with addressable cells."""

    def picture(self, sc, rgb, fx, fy, w, h, op, frame=AXIS, grid=0.0):
        """An ``(rows, cols, 3)`` picture ``w x h`` px centred at ``(fx, fy)``.
        Returns ``at(row, col) -> (fx, fy)``, the centre of a cell (fractions of
        a cell allowed)."""
        rgb = np.asarray(rgb, np.float32)
        c = self.P(float(fx), float(fy))
        if frame is not None:
            self.rect(sc, c, w + 6, h + 6, frame, op, radius=3, lift=0.1)
        draw_image(self, sc, c, rgb, w, h, op, grid=grid)
        return self.cells(fx, fy, w, h, rgb.shape[0], rgb.shape[1])

    def cells(self, fx, fy, w, h, rows, cols):
        dw, dh = self.frac(w, h)

        def at(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / cols, fy - dh / 2 + (r + 0.5) * dh / rows
        return at

    def outline(self, sc, fx, fy, w, h, ink, op, width=2.6, radius=2.0, lift=0.9):
        """An unfilled ``w x h`` px rectangle centred at ``(fx, fy)``."""
        self.rect(sc, self.P(float(fx), float(fy)), w, h, ink, op, border=ink, border_w=width,
                  radius=radius, lift=lift, fill_op=0.0)

    def patch(self, sc, fx, fy, w, h, ink, op, lift=0.6, radius=2.0):
        """A filled ``w x h`` px rectangle centred at ``(fx, fy)``."""
        self.rect(sc, self.P(float(fx), float(fy)), w, h, ink, op, radius=radius, lift=lift)

    def vbar(self, sc, fx, base, w, height, ink, op):
        """A bar ``height`` px tall standing on layout y ``base``."""
        height = max(float(height), 0.6)
        at = self.shift(self.P(float(fx), float(base)), 0, -height / 2)
        self.rect(sc, at, w, height, ink, op, radius=min(3.0, height / 2))

    def line(self, sc, a, b, ink, op, width=1.6, lift=1.2):
        """A straight line between layout points, drawn over pictures."""
        self.polyline(sc, [self.P(float(a[0]), float(a[1]), lift),
                           self.P(float(b[0]), float(b[1]), lift)], width, ink, op, bias=0.3)

    def arc_over(self, sc, a, b, ink, op, u=1.0, bend=0.0, width=2.2, head=True, lift=1.2):
        """:meth:`arc`, drawn over pictures instead of under them."""
        if op <= 0.01 or u <= 0.0:
            return
        x, y = self._curve_points(a, b, bend)
        m = max(2, int(np.ceil(len(x) * min(u, 1.0))))
        pts = [self.P(float(px), float(py), lift) for px, py in zip(x[:m], y[:m])]
        self.polyline(sc, pts, width, ink, op, bias=0.3)
        if head:
            self._head(sc, pts[-2], pts[-1], ink, op, width)


# ====================================================================== 51 receptive field
class ReceptiveField(Vision):
    """How much of the image one unit of the MNIST CNN can see: a 3x3 filter
    sliding, the field growing with depth (exact arithmetic), the gradient of
    one pooled output against its theoretical square, then the logit's gradient."""

    TITLE = "Receptive field"
    PHASES = (("one filter", 11.0), ("deeper", 15.0), ("one output", 13.0), ("the logit", 12.0))
    STAGES = (("one pixel", AXIS), ("conv 3×3", RED), ("+ pool 2×2", BLUE),
              ("+ conv 3×3", GREEN), ("2 dilated convs", VIOLET), ("dense layer", AMBER))

    def setup(self):
        from . import mnist

        s = self.study
        self.x = np.asarray(s.panels[0].data["values"][0], float)
        self.sizes = [int(v) for v in s.panels[1].data["values"]]
        self.params = mnist.cnn_train()[0]
        f = mnist._cnn_forward(self.params, self.x.reshape(1, 784))
        self.fmap = f["a1"][0, :, :, 0]
        # the pooled output to follow: the one whose four inputs differ most (the
        # study's own cell, (6, 6), lies on blank paper where all four are equal)
        a1 = self.fmap.reshape(13, 2, 13, 2).transpose(0, 2, 1, 3).reshape(13, 13, 4)
        self.cell = divmod(int(np.argmax(a1.max(-1) - a1.min(-1))), 13)
        pr, pc = self.cell
        self.four = self.fmap[2 * pr:2 * pr + 2, 2 * pc:2 * pc + 2]
        self.winner = divmod(int(f["arg"][0, pr, pc, 0]), 2)
        self.support = self.pool_gradient(self.params, f, pr, pc)
        self.full = np.asarray(s.panels[3].data["values"], float)
        self.klass = int(mnist.load()[3][0])
        self.nonzero = int(np.count_nonzero(self.full))
        order = np.argsort(-self.full.ravel())
        self.order = order
        self.share = np.r_[0.0, np.cumsum(self.full.ravel()[order]) / self.full.sum()]
        self.half = int(np.searchsorted(self.share, 0.5))

    def subtitle_for(self, k):
        z = self.sizes
        return [
            "one filter of the MNIST CNN slides over the digit: each output cell is computed "
            "from a 3 × 3 window, so it sees 9 of the 784 pixels",
            "exact arithmetic: stacking layers widens what one cell sees, %d then %d then %d "
            "pixels across; two dilated filters reach %d, a dense layer all %d" % (
                z[1], z[2], z[3], z[4], z[5]),
            "one pooled cell could see 4 × 4 = 16 pixels, but max-pooling passes the gradient "
            "to the winning cell only: it reaches %d pixels" % int(np.count_nonzero(self.support)),
            "the logit for %d sits behind a dense layer and sees everything: its gradient is "
            "non-zero on %d of 784 pixels, but half of it sits on just %d" % (
                self.klass, self.nonzero, self.half),
        ][k]

    def phase_1(self, sc, u, op):
        S, F = 232.0, 234.0
        ix, iy, fx, fy = 0.24, 0.61, 0.76, 0.61
        at = self.picture(sc, paper(self.x), ix, iy, S, S, op)
        k = int(min(u / 0.92, 1.0) ** 1.7 * 675)
        r, c = divmod(k, 26)
        self.outline(sc, *at(r + 1, c + 1), 3 * S / 28 + 2, 3 * S / 28 + 2, RED, op, 2.8)
        rgb = heat(self.fmap / self.fmap.max(), RED).reshape(676, 3)
        rgb[k + 1:] = PAPER * 0.90
        cell = self.picture(sc, rgb.reshape(26, 26, 3), fx, fy, F, F, op)
        self.outline(sc, *cell(r, c), F / 26 + 3, F / 26 + 3, RED, op, 2.2)
        a, b = at(r + 1, c + 2.7), cell(r, c - 0.8)
        self.arrow(sc, a, b, RED, op * 0.85, width=1.8)
        self.text(sc, ix, 0.32, "the digit, 28 × 28 pixels", op, strong=True, size=0.034)
        self.text(sc, fx, 0.32, "one filter's output, 26 × 26", op, strong=True, size=0.034)
        self.text(sc, 0.5, 0.905, "output cell %d of 676: a 3 × 3 window, 9 pixels" % (k + 1), op,
                  size=0.034, ink=dark(RED))

    def _square(self, n):
        """Centre (row = col) of an ``n``-pixel field around pixel 13."""
        return 13 - (n - 1) // 2 + (n - 1) / 2

    def phase_2(self, sc, u, op):
        S = 226.0
        ix, iy = 0.26, 0.61
        at = self.picture(sc, paper(self.x * 0.35), ix, iy, S, S, op, grid=0.12)
        stage = min(int(u * 6.25), 5)
        within = smoothstep(0.0, 0.55, u * 6.25 - stage) if stage < 6 else 1.0
        px = S / 28
        prev = {0: 1, 1: 1, 2: 3, 3: 4, 4: 1, 5: 9}[stage]
        for i in range(1, stage):                       # the fields already reached
            if (stage == 4 and i < 4) or (stage == 5 and i < 5):
                continue
            n = self.sizes[i]
            m = self._square(n)
            self.outline(sc, *at(m, m), n * px, n * px, self.STAGES[i][1], op * 0.55, 1.6)
        ink = self.STAGES[stage][1]
        if stage == 4:                                  # the nine taps of a filter dilated by 2
            for dr in (-2, 0, 2):
                for dc in (-2, 0, 2):
                    self.patch(sc, *at(13 + dr, 13 + dc), px - 1, px - 1, VIOLET,
                               op * (1 - 0.6 * within))
        n = prev + (self.sizes[stage] - prev) * within
        m = self._square(prev) + (self._square(self.sizes[stage]) - self._square(prev)) * within
        self.outline(sc, *at(m, m), n * px + 2, n * px + 2, dark(ink, 0.3) if stage == 0 else ink,
                     op, 3.0)
        self.text(sc, ix, 0.325, "what one cell can see", op, strong=True, size=0.034)
        full = self.span(0.22)[0]
        for i, (name, colour) in enumerate(self.STAGES):
            if i > stage:
                break
            grow = within if i == stage else 1.0
            z = self.sizes[i]
            self.hbar(sc, 0.66, 0.39 + i * 0.086, full * z / 28.0, dark(colour, 0.3) if i == 0
                      else colour, op, name, "%d × %d" % (z, z), grow, thick=24.0, track=full)
        z = self.sizes
        note = [
            "start from a single pixel",
            "a 3 × 3 filter reads %d × %d pixels" % (z[1], z[1]),
            "pooling joins 2 neighbours: %d + 1 = %d, and halves the grid" % (z[1], z[2]),
            "one step is now 2 pixels: %d + 2 × 2 = %d" % (z[2], z[3]),
            "or spread the taps 2 apart: 1 + 4 + 4 = %d in two layers" % z[4],
            "a dense layer reads every pixel: %d × %d" % (z[5], z[5]),
        ][stage]
        self.text(sc, 0.5, 0.905, note, op, size=0.034, ink=dark(ink, 0.3))

    @staticmethod
    def pool_gradient(params, f, pr, pc, channel=0):
        """|d pooled[pr, pc] / d pixel| of the CNN, ``(28, 28)``: the filter's
        weights under the 2x2 window's winner (nothing if the ReLU there is off)."""
        out = np.zeros((28, 28))
        r, c = divmod(int(f["arg"][0, pr, pc, channel]), 2)
        r, c = 2 * pr + r, 2 * pc + c
        if f["z1"][0, r, c, channel] > 0:
            out[r:r + 3, c:c + 3] = np.abs(params[0][0][:, :, channel])
        return out

    def phase_3(self, sc, u, op):
        S = 204.0
        ix, zx, y = 0.17, 0.48, 0.61
        R0, C0 = 2 * self.cell[0], 2 * self.cell[1]
        at = self.picture(sc, paper(self.x), ix, y, S, S, op)
        self.outline(sc, *at(R0 + 1.5, C0 + 1.5), 4 * S / 28 + 2, 4 * S / 28 + 2, BLUE, op, 2.4)
        # the same square, ten pixels of the digit enlarged
        lo, lc = int(np.clip(R0 - 3, 0, 18)), int(np.clip(C0 - 3, 0, 18))
        crop = paper(self.x[lo:lo + 10, lc:lc + 10] * 0.8)
        won = smoothstep(0.55, 0.70, u)
        g = self.support[lo:lo + 10, lc:lc + 10]
        crop = tint(crop, 0.9 * won * np.sqrt(g / max(g.max(), 1e-12)), RED)
        zat = self.picture(sc, crop, zx, y, S, S, op, grid=0.2)
        cellpx = S / 10
        self.outline(sc, *zat(R0 + 1.5 - lo, C0 + 1.5 - lc), 4 * cellpx + 2, 4 * cellpx + 2, BLUE,
                     op, 3.0)
        for dy in (-1, 1):
            self.line(sc, at(R0 + 1.5 + 2 * dy, C0 + 3.5), zat(4.5 + 5 * dy, -0.5), BLUE,
                      op * 0.6, 1.2)
        self.text(sc, ix, 0.325, "one pooled cell", op, strong=True, size=0.034)
        self.text(sc, zx, 0.325, "its 4 × 4 pixels, enlarged", op, strong=True, size=0.034)
        # the four filter outputs under the pool, each with its own 3 x 3 window
        bx, by, step = 0.815, 0.56, 0.13
        turn = min(int(smoothstep(0.18, 0.55, u) * 4), 3)
        show = smoothstep(0.12, 0.22, u)
        wr, wc = self.winner
        for r in range(2):
            for c in range(2):
                mine = (r, c) == (wr, wc)
                lit = (u < 0.55 and (r * 2 + c) == turn) or (u >= 0.55 and mine)
                x, yy = bx + (c - 0.5) * 0.095, by + (r - 0.5) * step
                fill = RED if (mine and u >= 0.55) else (soft(RED) if lit else PAPER)
                self.box(sc, x, yy, 72, 52, fill, op * show * (1.0 if lit or u < 0.55 else 0.4),
                         "%.2f" % self.four[r, c], ink=WHITE if (mine and u >= 0.55) else None,
                         border=RED, size=0.038)
                if lit and show > 0:
                    self.outline(sc, *zat(R0 + r + 1 - lo, C0 + c + 1 - lc), 3 * cellpx,
                                 3 * cellpx, RED, op * show, 2.8)
        self.text(sc, bx, 0.325, "the 4 filter outputs", op * show, strong=True, size=0.034)
        self.text(sc, bx, 0.375, "under that cell", op * show)
        self.text(sc, bx, 0.775, "max-pool keeps %.2f" % self.four.max(), op * won,
                  ink=dark(BLUE), size=0.034)
        wx, wy = bx + (wc - 0.5) * 0.095, by + (wr - 0.5) * step
        self.pulse(sc, (wx - 0.045, wy), zat(R0 + wr + 1 - lo, C0 + wc + 2.6 - lc),
                   (u - 0.66) / 0.24, RED, op, bend=0.06, r=6.0)
        self.text(sc, 0.5, 0.905, "it could see 16 pixels; the gradient reaches %d: only the "
                  "winner passes it back" % int(np.count_nonzero(self.support)), op * won,
                  size=0.034, ink=dark(RED))

    def phase_4(self, sc, u, op):
        S = 216.0
        hx, y = 0.23, 0.61
        go = min(u / 0.85, 1.0)
        k = int(go ** 1.6 * 784)
        shown = np.zeros(784)
        shown[self.order[:k]] = self.full.ravel()[self.order[:k]]
        rgb = tint(paper(self.x * 0.22), np.sqrt(shown.reshape(28, 28)), RED)
        self.picture(sc, rgb, hx, y, S, S, op)
        self.outline(sc, hx, y, S + 8, S + 8, AMBER, op, 2.6)
        self.text(sc, hx, 0.335, "gradient of the logit for %d" % self.klass, op, strong=True,
                  size=0.034)
        self.text(sc, hx, 0.895, "strongest %d pixels shown" % k, op, ink=dark(RED))
        xs = np.arange(785.0)
        chart = self.chart(((0.50, 0.93), (0.37, 0.80)), xs, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "pixels, strongest first", "share of the gradient",
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)],
                  xticks=[(v, "%d" % v) for v in (0, 200, 400, 600, 784)])
        self.plot(sc, chart, xs, self.share, RED, op, k / 784.0, width=2.6)
        if k >= self.half:
            hxf, hyf = chart.fx(self.half), chart.fy(0.5)
            self.line(sc, (hxf, chart.fy(0.0)), (hxf, hyf), AXIS, op * 0.8, 1.2)
            self.text(sc, hxf + 0.012, hyf + 0.06, "half of it: %d pixels" % self.half, op,
                      anchor="lm", strong=True)


# ====================================================================== 52 ViT attention
class VitAttention(Vision):
    """A three-block vision transformer on MNIST (:func:`voxanim.study_vision.vit_attention`):
    the digit cut into 49 patch tokens behind a CLS token, what CLS reads in the first and
    the last block, how much of that reading lands on the strokes, and how far attention
    travels across the patch grid (no further and no nearer than attention spread evenly:
    the measured answer, drawn as it came out)."""

    TITLE = "What a vision transformer attends to"
    PHASES = (("patches", 9.0), ("block 1", 10.0), ("block 3", 10.0), ("focus", 12.0),
              ("reach", 12.0))
    HEADS = (RED, BLUE, GREEN, AMBER)
    BLOCKS = (TEAL, BLUE, VIOLET)

    def setup(self):
        s = self.study
        d = sv.vit_run()
        self.tiles = np.asarray(s.panels[0].data["values"], float)            # (49, 4, 4)
        self.digits = np.asarray(d["example"], float)                         # (8, 784)
        self.said = [int(v) for v in d["read"]]
        att = np.asarray(d["attention"], float)                 # block, digit, head, from, to
        cls = att[:, :, :, 0, 1:]
        self.cls = cls / cls.sum(-1, keepdims=True)             # block, digit, head, patch
        among = att[:, 0, :, 1:, 1:].mean(1)                    # block, from, to: first digit
        self.among = among / among.sum(-1, keepdims=True)
        self.has_ink = sv.has_ink(self.digits)                  # digit, patch
        self.distance = sv.patch_distance()
        reach, read = s.panels[2].data, s.panels[3].data
        self.reach = [float(v) for v in reach["values"][:-1]]
        self.uniform = float(reach["values"][-1])               # exact: attention spread evenly
        self.reach_heads = np.asarray(reach["heads"], float)    # block, head (mean of seeds)
        self.read = [float(v) for v in read["values"][:-1]]     # patches CLS reads, per block
        self.read_seeds = np.asarray(read["seeds"], float)      # seed, block
        self.on_ink = [float(v) for v in read["on_ink"]]        # share of it on strokes
        self.on_ink_heads = np.asarray(read["on_ink_heads"], float)     # seed, block, head
        self.ink_share = float(read["ink_share"])
        self.accuracy = [float(v) for v in s.metadata["test_accuracy"]]
        marked = np.flatnonzero(self.has_ink[0])
        self.queries = [int(marked[i]) for i in np.linspace(0, len(marked) - 1, 4).astype(int)]

    def top3(self, block):
        """Share of CLS's weight on each head's three favourite patches (the digit shown)."""
        return np.sort(self.cls[block, 0], axis=-1)[:, -3:].sum(-1)

    def subtitle_for(self, k):
        first, last = 100 * self.top3(0), 100 * self.top3(-1)
        return [
            "the digit is cut into 49 patches of 4 × 4 pixels; each becomes a token, and one "
            "extra token, CLS, is put in front to collect the answer: 50 tokens",
            "block 1: CLS reads the patches, each of the four heads its own way. The weight is "
            "spread thin: the three patches a head favours hold %.0f%% to %.0f%% of it" % (
                first.min(), first.max()),
            "block 3, the last: the heads have picked their patches. The top three now hold "
            "%.0f%% to %.0f%% of a head's weight, and from CLS the model reads this digit as "
            "%d" % (last.min(), last.max(), self.said[0]),
            "over 500 test digits and three trained models: CLS spreads its attention over "
            "about %.0f of the 49 patches in block 1, then %.0f and %.0f. %.0f%% to %.0f%% of it "
            "falls on strokes, %.0f%% if read evenly. Test accuracy: %.1f%%" % (
                *self.read, 100 * min(self.on_ink), 100 * max(self.on_ink),
                100 * self.ink_share, 100 * float(np.mean(self.accuracy))),
            "and it does not stay near: between patches attention travels %.2f, %.2f and %.2f "
            "patch widths in blocks 1 to 3, against %.2f if spread evenly. Unlike a "
            "convolution, nothing here favours a patch's neighbours" % (
                *self.reach, self.uniform),
        ][k]

    def _tile(self, sc, i, fx, fy, size, op, frame=AXIS):
        self.picture(sc, paper(self.tiles[i] * 0.9), fx, fy, size, size, op, frame=frame)

    def _grid_at(self, fx, fy, size, gap=4.0):
        """Centres of the 49 tiles of a patch grid ``size`` px across."""
        dx, dy = self.frac((size + gap) / 7, (size + gap) / 7)
        return [(fx + (i % 7 - 3) * dx, fy + (i // 7 - 3) * dy) for i in range(49)]

    def _attended(self, sc, digit, w, ink, fx, fy, size, op, frame=AXIS, top=None):
        """Digit ``digit`` with each patch's paper coloured in proportion to its weight in
        ``w`` (49,), the deepest at ``top`` (default: the largest weight). Returns
        ``centre(patch) -> (fx, fy)``."""
        top = float(w.max()) if top is None else top
        a = np.kron(0.92 * (w / top).reshape(7, 7), np.ones((4, 4)))[..., None]
        v = np.clip(self.digits[digit].reshape(28, 28), 0, 1)[..., None]
        rgb = (PAPER * (1 - a) + np.asarray(ink) * a) * (1 - v) + np.asarray(DIGIT_INK) * v
        at = self.picture(sc, rgb, fx, fy, size, size, op, frame=frame)
        return lambda i: at(4 * (i // 7) + 1.5, 4 * (i % 7) + 1.5)

    def phase_1(self, sc, u, op):
        S = 196.0
        gap = 21.0 * smoothstep(0.05, 0.25, u)
        home = self._grid_at(0.18, 0.62, S + gap, gap)
        slots = [(0.435 + (j % 10) * 0.054, 0.44 + (j // 10) * 0.092) for j in range(50)]
        self.text(sc, 0.18, 0.325, "one digit, 28 × 28", op, strong=True, size=0.034)
        self.text(sc, 0.678, 0.325, "the sequence the model reads: 50 tokens", op, strong=True,
                  size=0.034)
        self.box(sc, slots[0][0], slots[0][1], 40, 34, AMBER, op * smoothstep(0.2, 0.3, u), "CLS",
                 ink=WHITE, size=0.028)
        for i in range(49):
            go = smoothstep(0.25 + 0.011 * i, 0.37 + 0.011 * i, u)
            (hx, hy), (tx, ty) = home[i], slots[i + 1]
            self._tile(sc, i, hx, hy, S / 7, op * (0.25 if go > 0 else 1.0), frame=None)
            if go > 0:
                self._tile(sc, i, hx + (tx - hx) * go, hy + (ty - hy) * go - 0.05 * go * (1 - go),
                           S / 7 + 6 * go, op)
        self.text(sc, 0.678, 0.905, "a patch is 4 × 4 = 16 pixels; the model turns each into "
                  "32 numbers", op * smoothstep(0.88, 0.96, u), ink=dark(AMBER, 0.35))

    def _reading(self, sc, u, op, block):
        S = 140.0
        for head, ink in enumerate(self.HEADS):
            fx = 0.135 + head * 0.243
            w = self.cls[block, 0, head]
            upto = 49 if u >= 0.4 else int(u / 0.4 * 49)         # patches read so far
            centre = self._attended(sc, 0, np.where(np.arange(49) < upto, w, 0.0), ink, fx,
                                    0.645, S, op, top=w.max())
            cx, cy = fx + 0.02, 0.35
            self.box(sc, cx, cy, 50, 34, AMBER, op, "CLS", ink=WHITE, size=0.028)
            self.text(sc, cx - 0.036, cy, "head %d" % (head + 1), op, anchor="rm", ink=dark(ink),
                      size=0.032)
            best = np.argsort(-w)[:3]
            for rank, i in enumerate(best):
                start = 0.44 + 0.06 * rank
                a, b = centre(i), (cx + (rank - 1) * 0.014, cy + 0.04)
                self.arc_over(sc, a, b, dark(ink, 0.25), op, smoothstep(start, start + 0.15, u),
                              0.0, 1.4 + 7.0 * w[i])
                self.pulse(sc, a, b, (u - start - 0.12) / 0.25, ink, op, r=4.0 + 7 * w[i])
                self.outline(sc, a[0], a[1], S / 7 + 2, S / 7 + 2, dark(ink, 0.3),
                             op * smoothstep(start, start + 0.1, u), 2.4)
            done = op * smoothstep(0.62, 0.72, u)
            self.text(sc, fx, 0.845, "top 3 patches: %.0f%%" % (100 * w[best].sum()), done,
                      ink=dark(ink), size=0.032)
            self.text(sc, fx, 0.895, "on strokes: %.0f%%" % (100 * w[self.has_ink[0]].sum()),
                      done, ink=INK, size=0.032)
        self.text(sc, 0.5, 0.945, "deeper colour: more of CLS's attention on that patch", op)

    def phase_2(self, sc, u, op):
        self._reading(sc, u, op, 0)

    def phase_3(self, sc, u, op):
        self._reading(sc, u, op, len(self.BLOCKS) - 1)

    def phase_4(self, sc, u, op):
        S = 100.0
        self.text(sc, 0.25, 0.315, "block 3, the four heads together", op, strong=True,
                  size=0.032)
        for j in range(6):
            n = j + 1                                    # six other test digits
            fx, fy = 0.105 + (j % 3) * 0.145, 0.475 + (j // 3) * 0.275
            w = self.cls[-1, n].mean(0)
            show = op * smoothstep(0.04 * j, 0.04 * j + 0.1, u)
            self._attended(sc, n, w, VIOLET, fx, fy, S, show)
            self.text(sc, fx, fy + 0.125, "%.0f%% on strokes" % (100 * w[self.has_ink[n]].sum()),
                      show, ink=dark(VIOLET), size=0.032)
        show = smoothstep(0.35, 0.45, u)
        full = self.span(0.17)[0]
        self.text(sc, 0.57, 0.335, "patches CLS spreads its attention over", op * show,
                  anchor="lm", strong=True)
        rows = [("block %d" % (i + 1), self.BLOCKS[i], v) for i, v in enumerate(self.read)]
        rows.append(("read evenly", AXIS, 49.0))
        for row, (name, ink, value) in enumerate(rows):
            self.hbar(sc, 0.70, 0.39 + row * 0.058, full * value / 49.0, ink, op * show, name,
                      "%.0f" % value, (u - 0.4 - 0.04 * row) / 0.2, thick=18.0, track=full)
        show = smoothstep(0.62, 0.72, u)
        self.text(sc, 0.57, 0.655, "share of it on patches with a stroke", op * show, anchor="lm",
                  strong=True)
        rows = [("block %d" % (i + 1), self.BLOCKS[i], v) for i, v in enumerate(self.on_ink)]
        rows.append(("read evenly", AXIS, self.ink_share))
        for row, (name, ink, value) in enumerate(rows):
            self.hbar(sc, 0.70, 0.71 + row * 0.058, full * value, ink, op * show, name,
                      "%.0f%%" % (100 * value), (u - 0.66 - 0.04 * row) / 0.2, thick=18.0,
                      track=full)
        heads = 100 * self.on_ink_heads[:, -1]
        self.text(sc, 0.57, 0.935, "block 3, head by head: %.0f%% to %.0f%%" % (
            heads.min(), heads.max()), op * smoothstep(0.85, 0.93, u), anchor="lm")

    def phase_5(self, sc, u, op):
        S = 150.0
        turn = min(int(u / 0.6 * len(self.queries)), len(self.queries) - 1)
        within = min(u / 0.6 * len(self.queries) - turn, 1.0) if u < 0.6 else 1.0
        q = self.queries[turn]
        for k, (block, fx) in enumerate(((0, 0.15), (len(self.BLOCKS) - 1, 0.41))):  # noqa: B007
            ink = self.BLOCKS[block]
            w = self.among[block, q]
            centre = self._attended(sc, 0, w, ink, fx, 0.61, S, op)
            self.outline(sc, *centre(q), S / 7 + 3, S / 7 + 3, AMBER, op, 3.0)
            for i in np.argsort(-w)[:3]:
                if i != q:
                    self.arc_over(sc, centre(q), centre(i), AMBER, op,
                                  smoothstep(0.1, 0.5, within), 0.04, 2.4)
            self.text(sc, fx, 0.36, "block %d" % (block + 1), op, ink=dark(ink), size=0.036)
            self.text(sc, fx, 0.855, "from the amber patch:", op, ink=dark(ink), size=0.032)
            self.text(sc, fx, 0.905, "%.2f patch widths" % float((w * self.distance[q]).sum()),
                      op, ink=dark(ink), size=0.032)
        show = smoothstep(0.6, 0.7, u)
        full = self.span(0.18)[0]
        self.text(sc, 0.60, 0.36, "average distance, 500 digits", op * show, anchor="lm",
                  strong=True, size=0.032)
        rows = [("block %d" % (i + 1), self.BLOCKS[i], v) for i, v in enumerate(self.reach)]
        rows.append(("spread evenly", AXIS, self.uniform))
        for row, (name, ink, value) in enumerate(rows):
            self.hbar(sc, 0.745, 0.46 + row * 0.095, full * value / 4.0, ink, op * show, name,
                      "%.2f" % value, (u - 0.62 - 0.04 * row) / 0.2, thick=24.0, track=full)
        self.text(sc, 0.60, 0.855, "head by head: %.2f to %.2f" % (
            self.reach_heads.min(), self.reach_heads.max()), op * smoothstep(0.8, 0.9, u),
            anchor="lm", size=0.032)


# ====================================================================== 53 CLIP
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_CLIP = {}


def clip_run(every=10):
    """The study's contrastive run again (same seed, data and updates), keeping
    the 10 x 10 score matrix every ``every`` updates. Returns ``(steps, snaps,
    losses, images)``; ``images`` are the ten 7x7 digits of the matrix rows."""
    if every not in _CLIP:
        from .optim import Adam

        x, y, _, _ = nv.mnist_data()
        z, y = vm.downsample(x[:12000]), y[:12000]
        rng = np.random.default_rng(53)
        p = dict(I=rng.normal(0, 0.12, (z.shape[1], 16)), T=rng.normal(0, 0.2, (10, 16)))
        opt, losses, snaps, steps = Adam(p, 0.006), [], [], []
        fixed = np.array([np.flatnonzero(y == i)[0] for i in range(10)])
        for step in range(901):
            if step % every == 0:
                steps.append(step)
                snaps.append(vm.info_nce(p, z[fixed], y[fixed])[2])
            if step == 900:
                break
            idx = rng.choice(len(z), 64)
            loss, g, _ = vm.info_nce(p, z[idx], y[idx])
            opt.step(p, g)
            losses.append(loss)
        _CLIP[every] = (np.array(steps), np.array(snaps), np.array(losses), z[fixed])
    return _CLIP[every]


class ClipContrastive(Vision):
    """Contrastive image-text training on MNIST digits and their ten words: the
    similarity matrix filled cell by cell, sharpening to its diagonal during
    training, the temperature that sharpens it, and matching by nearest word."""

    TITLE = "Contrastive image-text training"
    PHASES = (("compare", 12.0), ("train", 14.0), ("temperature", 12.0), ("match", 13.0))
    TEMP = 0.15
    MAT = (0.385, 0.665, 470.0, 236.0)      # the matrix: x, y, width, height in px

    def setup(self):
        s = self.study
        d = study_arrays("clip-contrastive")
        self.steps, self.snaps, self.losses, self.rows = clip_run()
        self.loss = np.asarray(s.panels[1].data["ys"], float)[0]
        self.temps = np.array([0.03, 0.07, 0.15, 0.4, 1.0])
        self.entropy = [float(v) for v in s.panels[3].data["values"]]
        self.scores = np.asarray(d["scores"], float)             # cosine / 0.15, 1000 x 10
        self.pred, self.target = np.asarray(d["predictions"]), np.asarray(d["targets"])
        self.accuracy = float(np.mean(self.pred == self.target))
        _, _, xt, _ = nv.mnist_data()
        self.test = vm.downsample(xt[:1000])
        wrong = np.flatnonzero(self.pred != self.target)
        right = np.flatnonzero(self.pred == self.target)
        self.shown = [int(right[0]), int(right[1]), int(wrong[0]), int(right[2]), int(right[3]),
                      int(wrong[1])]

    def diagonal_wins(self, snap):
        """How many of the ten images score highest with their own word."""
        return int(np.sum(np.asarray(snap).argmax(1) == np.arange(10)))

    def subtitle_for(self, k):
        return [
            "ten digit images and the ten digit words each become 16 numbers; a cell is how "
            "closely an image and a word point the same way. Before training: no pattern",
            "training pulls each image towards its own word and away from the other nine: over "
            "900 updates the loss falls from %.2f to %.2f and the diagonal lights up" % (
                self.loss[0], self.loss[-1]),
            "temperature divides the similarities before they become a choice: at %s the choice "
            "is spread (%.2f nats of uncertainty), at %s it is all on one word (%.2f)" % (
                "1.0", self.entropy[-1], "0.03", self.entropy[0]),
            "a held-out image is matched to the nearest of the same ten words: right for %.1f%% "
            "of 1,000 images. These are the words it trained with, not new ones" % (
                100 * self.accuracy),
        ][k]

    def _matrix(self, sc, op, values, shown=100, diagonal=0.0):
        """The score matrix, one column per image and one row per word; the
        first ``shown`` cells (image by image) are filled in."""
        mx, my, w, h = self.MAT
        vmax = float(np.abs(self.snaps).max())
        rgb = heat(np.clip(np.asarray(values) / vmax, -1, 1)).reshape(100, 3)
        rgb[shown:] = PAPER * 0.90
        at = self.picture(sc, rgb.reshape(10, 10, 3).transpose(1, 0, 2), mx, my, w, h, op,
                          grid=0.25)
        for i in range(10):
            x, y = at(-1.5, i)
            self.picture(sc, paper(self.rows[i].reshape(7, 7)), x, y, 38, 38, op, frame=AXIS)
            x, y = at(i, -0.75)
            self.text(sc, x, y, WORDS[i], op, ink=INK, anchor="rm")
            if diagonal > 0:
                self.outline(sc, *at(i, i), w / 10, h / 10, GREEN, op * diagonal, 2.6)
        return at

    def phase_1(self, sc, u, op):
        k = int(min(u / 0.88, 1.0) * 100)
        at = self._matrix(sc, op, self.snaps[0], k)
        _mx, _my, w, h = self.MAT
        i, j = divmod(min(k, 99), 10)                  # image i, word j
        if k < 100:
            self.outline(sc, *at(j, i), w / 10 + 2, h / 10 + 2, AMBER, op, 2.8)
            self.outline(sc, *at(-1.5, i), 46, 46, AMBER, op, 2.6)
            self.line(sc, at(-0.5, i), at(j - 0.5, i), AMBER, op * 0.9, 1.8)
            self.line(sc, at(j, -0.62), at(j, i - 0.5), AMBER, op * 0.9, 1.8)
        cosine = self.snaps[0][i, j] * self.TEMP
        self.lines(sc, 0.70, 0.36, [
            ("image of a %d" % i, INK),
            ("word “%s”" % WORDS[j], INK),
            ("similarity %+.2f" % cosine, dark(RED) if cosine > 0 else dark(BLUE)),
        ], op, size=0.036, step=0.062)
        self.lines(sc, 0.70, 0.59, [
            ("+1: pointing the same way", dark(RED)),
            ("0: unrelated", None),
            ("−1: pointing opposite ways", dark(BLUE)),
        ], op, step=0.05)
        self.lines(sc, 0.70, 0.79, ["before training, %d of 10 images" % (
            self.diagonal_wins(self.snaps[0])), "are closest to their own word"],
            op * smoothstep(0.88, 0.95, u), step=0.05)

    def phase_2(self, sc, u, op):
        go = min(u / 0.88, 1.0) ** 2.6                 # the first updates do most: dwell on them
        n = int(go * (len(self.steps) - 1))
        self._matrix(sc, op, self.snaps[n], diagonal=0.9)
        step = int(self.steps[n])
        chart = self.chart(((0.72, 0.95), (0.43, 0.74)), np.arange(900.0), self.loss)
        self.axes(sc, chart, op, "update", "loss", xticks=[(v, "%d" % v) for v in (0, 300, 600,
                                                                                 900)])
        self.plot(sc, chart, np.arange(900.0), self.loss, AMBER, op, step / 899.0, width=2.2)
        self.text(sc, 0.835, 0.33, "update %d" % step, op, strong=True, size=0.038)
        self.lines(sc, 0.835, 0.875, [("%d of 10 images closest" % self.diagonal_wins(self.snaps[n]),  # noqa: E501
                                       dark(GREEN, 0.3)), ("to their own word", dark(GREEN, 0.3))],
                   op, anchor="mm", step=0.05, size=0.032)

    def phase_3(self, sc, u, op):
        # from the widest temperature to the sharpest, resting on each one the study measured
        stops = self.temps[::-1]
        pos = min(u / 0.9, 1.0) * (len(stops) - 1)
        i = min(int(pos), len(stops) - 2)
        f = smoothstep(0.35, 1.0, pos - i)
        t = float(np.exp(np.log(stops[i]) * (1 - f) + np.log(stops[i + 1]) * f))
        cosine = self.scores[0] * self.TEMP
        z = cosine / t
        p = np.exp(z - z.max())
        p /= p.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-20)))
        self.picture(sc, paper(self.test[0].reshape(7, 7)), 0.13, 0.50, 120, 120, op)
        self.text(sc, 0.13, 0.325, "a held-out image", op, strong=True, size=0.034)
        self.text(sc, 0.13, 0.70, "temperature %.2f" % t, op, strong=True, size=0.038)
        self.text(sc, 0.13, 0.765, "uncertainty %.2f nats" % entropy, op, ink=dark(AMBER, 0.35),
                  size=0.034)
        # the temperature dial: a log scale with the five measured settings
        x0, x1, y = 0.05, 0.215, 0.86
        self.line(sc, (x0, y), (x1, y), AXIS, op, 2.0)
        where = lambda v: x0 + (x1 - x0) * (np.log(v) - np.log(0.03)) / (np.log(1.0) - np.log(0.03))  # noqa: E731
        for v in self.temps:
            self.patch(sc, where(v), y, 3, 12, AXIS, op)
        self.disk(sc, self.P(float(where(t)), y), 7.0, AMBER, op, lift=1.4, outline=0.0)
        self.text(sc, x0, y + 0.05, "sharp", op, anchor="lm")
        self.text(sc, x1, y + 0.05, "spread", op, anchor="rm")
        base, top = 0.80, self.span(0, 0.40)[1]
        self.text(sc, 0.62, 0.325, "how the image's choice is shared among the ten words", op,
                  strong=True, size=0.032)
        for j in range(10):
            x = 0.32 + j * 0.066
            best = j == int(cosine.argmax())
            self.vbar(sc, x, base, 40, top * p[j], GREEN if best else DIM, op)
            self.text(sc, x, base + 0.045, WORDS[j], op, ink=INK if best else None)
            if p[j] >= 0.095:
                self.text(sc, x, base - self.frac(0, top * p[j])[1] - 0.03, "%.0f%%" % (100 * p[j]),
                          op, strong=True)
        self.line(sc, (0.29, base), (0.95, base), AXIS, op, 1.4)

    def phase_4(self, sc, u, op):
        n = len(self.shown)
        turn = min(int(u / 0.72 * n), n - 1)
        within = min(u / 0.72 * n - turn, 1.0) if u < 0.72 else 1.0
        idx = self.shown[turn]
        z = self.scores[idx]
        p = np.exp(z - z.max())
        p /= p.sum()
        said, truth = int(self.pred[idx]), int(self.target[idx])
        ok = said == truth
        self.picture(sc, paper(self.test[idx].reshape(7, 7)), 0.13, 0.56, 130, 130, op)
        self.text(sc, 0.13, 0.325, "held-out image %d of 6" % (turn + 1), op, strong=True,
                  size=0.034)
        self.text(sc, 0.13, 0.80, "it is a %s" % WORDS[truth], op, size=0.034)
        full = self.span(0.26)[0]
        grow = smoothstep(0.05, 0.45, within)
        for j in range(10):
            y = 0.345 + j * 0.058
            ink = (GREEN if ok else RED) if j == said else DIM
            self.hbar(sc, 0.40, y, full * p[j], ink, op, WORDS[j], "%.0f%%" % (100 * p[j])
                      if p[j] >= 0.05 else "", grow, thick=18.0, track=full)
        self.pulse(sc, (0.215, 0.56), (0.325, 0.345 + said * 0.058), (within - 0.05) / 0.4,
                   GREEN if ok else RED, op, bend=0.03, r=6.0)
        verdict = smoothstep(0.45, 0.55, within)
        self.text(sc, 0.13, 0.86, "nearest word: %s" % WORDS[said], op * verdict,
                  ink=dark(GREEN if ok else RED), size=0.034)
        done = smoothstep(0.74, 0.84, u)
        self.text(sc, 0.85, 0.44, "%.1f%%" % (100 * self.accuracy), op * done, strong=True,
                  size=0.075)
        self.lines(sc, 0.85, 0.56, ["of 1,000 held-out", "images matched to", "the right word"],
                   op * done, anchor="mm", step=0.05, size=0.032)
        self.lines(sc, 0.85, 0.76, ["the same ten words", "it trained with"], op * done,
                   anchor="mm", step=0.046)


# ====================================================================== 54 adversarial
def as_layers(d, prefix):
    """The MLP stored under ``prefix`` in a study's arrays, as ``mnist`` layers."""
    return [[np.asarray(d["%sW%d" % (prefix, k)]), np.asarray(d["%sb%d" % (prefix, k)])]
            for k in range(3)]


def input_gradient(params, x, y):
    """d cross-entropy / d pixels of the MNIST MLP, and its class probabilities."""
    from . import mnist

    acts, pre, pr = mnist._forward(params, x)
    return mnist._backward(params, acts, pre, pr, np.eye(10)[y])[2][0], pr


def pgd_steps(params, x, y, eps, steps=12):
    """The study's PGD attack on one image, every step kept: ``(images,
    probabilities)``, each with ``steps + 1`` rows (row 0: the clean image)."""
    z, zs, ps = x.copy(), [], []
    for _ in range(steps):
        g, pr = input_gradient(params, z, y)
        zs.append(z[0].copy())
        ps.append(pr[0])
        z = np.clip(z + eps / 4 * np.sign(g), x - eps, x + eps).clip(0, 1)
    zs.append(z[0].copy())
    ps.append(input_gradient(params, z, y)[1][0])
    return np.array(zs), np.array(ps)


class AdversarialExamples(Vision):
    """A change too small to notice that flips the MNIST classifier: the attack
    built step by step, the budget widened, accuracy on 160 held-out digits
    against the budget, and a short adversarial fine-tune."""

    TITLE = "Adversarial examples"
    PHASES = (("the attack", 14.0), ("the budget", 10.0), ("accuracy", 11.0),
              ("training on attacks", 13.0))
    EPS = 0.1

    def setup(self):
        s = self.study
        d = study_arrays("adversarial-examples")
        self.base, self.tuned = as_layers(d, "base_weights_"), as_layers(d, "weights_")
        train_x, train_y, x, y = nv.mnist_data()
        self.truth = int(y[0])
        self.clean = np.asarray(d["clean"], float)
        self.traj, self.probs = pgd_steps(self.base, x[:1], y[:1], self.EPS)
        self.eps = np.asarray(s.panels[2].data["x"], float)
        self.fgsm, self.pgd = np.asarray(s.panels[2].data["ys"], float)
        self.examples = np.asarray(d["examples"], float)
        from . import mnist

        self.reads = mnist._forward(self.base, self.examples)[2]
        self.bars = [float(v) for v in s.panels[3].data["values"]]   # base/tuned clean, base/tuned attacked  # noqa: E501
        g, _ = input_gradient(self.base, train_x[:4], train_y[:4])
        self.batch = train_x[:4]
        self.attacked = np.clip(train_x[:4] + self.EPS * np.sign(g), 0, 1)
        self.flip = int(self.probs[-1].argmax())

    def subtitle_for(self, k):
        e = list(self.eps).index(self.EPS)
        return [
            "the classifier reads this digit as %d (%.0f%% sure). Twelve small steps, each "
            "nudging every pixel the way that hurts most, never more than %.1f in total: now it "
            "reads %d (%.0f%% sure)" % (self.truth, 100 * self.probs[0][self.truth], self.EPS,
                                       self.flip, 100 * self.probs[-1][self.flip]),
            "the budget is the largest change allowed to any pixel, on a scale where black is 1. "
            "At %s the digit still reads %d; from %s the classifier is wrong" % (
                self._eps(self._last_right()), self.truth, self._eps(self._last_right() + 1)),
            "the same attacks on 160 held-out digits: at a budget of %.1f the classifier is "
            "right on %.1f%% after one step (FGSM) and %.1f%% after twelve (PGD), down from "
            "%.1f%%" % (self.EPS, 100 * self.fgsm[e], 100 * self.pgd[e], 100 * self.fgsm[0]),
            "220 more updates on digits together with their attacked copies: accuracy under "
            "the one-step attack rises from %.1f%% to %.1f%%, and on clean digits falls from "
            "%.1f%% to %.1f%%" % (100 * self.bars[2], 100 * self.bars[3], 100 * self.bars[0],
                                  100 * self.bars[1]),
        ][k]

    def _last_right(self):
        ok = self.reads.argmax(1) == self.truth
        return int(np.flatnonzero(ok)[-1])

    @staticmethod
    def _eps(v_or_index, eps=(0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3)):
        return ("%.3f" % eps[v_or_index]).rstrip("0").rstrip(".") if eps[v_or_index] else "0"

    def _bars(self, sc, p, fx, fy, op, width=0.25, height=120.0):
        """Ten class probabilities as bars on a baseline at ``fy``."""
        for j in range(10):
            x = fx + (j - 4.5) * width / 10
            ink = GREEN if j == self.truth else RED if j == self.flip else DIM
            self.vbar(sc, x, fy, 17, height * p[j], ink, op)
            self.text(sc, x, fy + 0.04, str(j), op, ink=dark(ink) if ink is not DIM else None)
        self.line(sc, (fx - width / 2, fy), (fx + width / 2, fy), AXIS, op, 1.4)

    def phase_1(self, sc, u, op):
        S = 150.0
        at = 12 * min(u / 0.88, 1.0) ** 2.2             # the first steps decide: dwell on them
        n = int(at)
        z, p = self.traj[n], self.probs[n]
        delta = z - self.clean
        y = 0.50
        self.picture(sc, paper(self.clean), 0.12, y, S, S, op)
        self.picture(sc, heat(delta.reshape(28, 28) / self.EPS), 0.345, y, S, S, op)
        self.picture(sc, paper(z), 0.57, y, S, S, op)
        self.text(sc, 0.232, y, "+", op, strong=True, size=0.06)
        self.text(sc, 0.458, y, "=", op, strong=True, size=0.06)
        self.text(sc, 0.12, 0.31, "the digit", op, strong=True, size=0.034)
        self.text(sc, 0.345, 0.31, "the change, ×%d" % round(1 / self.EPS), op, strong=True,
                  size=0.034)
        self.text(sc, 0.57, 0.31, "what the classifier sees", op, strong=True, size=0.034)
        self.text(sc, 0.345, 0.70, "step %d of 12" % n, op, size=0.034, ink=INK)
        self.lines(sc, 0.345, 0.77, [("red: pixel made darker", dark(RED)),
                                     ("blue: pixel made lighter", dark(BLUE))], op,
                   anchor="mm", step=0.045)
        self.text(sc, 0.57, 0.70, "largest change %.3f" % float(np.abs(delta).max()), op,
                  size=0.034, ink=INK)
        self._bars(sc, p, 0.83, 0.62, op, 0.24, 130.0)
        said = int(p.argmax())
        self.text(sc, 0.83, 0.31, "its answer", op, strong=True, size=0.034)
        self.text(sc, 0.83, 0.75, "reads %d, %.0f%% sure" % (said, 100 * p[said]), op,
                  ink=dark(GREEN if said == self.truth else RED), size=0.038)
        # the gradient: from the answer back to the pixels
        self.pulse(sc, (0.72, 0.80), (0.42, 0.84), at - n, AMBER,
                   op * (1.0 if n < 12 else 0.0), bend=-0.05, r=5.5)
        self.text(sc, 0.56, 0.905, "each step asks: which way does each pixel hurt the right "
                  "answer most?", op, ink=dark(AMBER, 0.35))

    def phase_2(self, sc, u, op):
        S = 96.0
        for i, e in enumerate(self.eps):  # noqa: B007
            show = smoothstep(0.05 + 0.09 * i, 0.12 + 0.09 * i, u)
            x = 0.095 + i * 0.135
            said = int(self.reads[i].argmax())
            ok = said == self.truth
            self.picture(sc, paper(self.examples[i]), x, 0.52, S, S, op * show,
                         frame=GREEN if ok else RED)
            self.text(sc, x, 0.355, "budget %s" % self._eps(i), op * show, strong=True)
            self.text(sc, x, 0.68, "reads %d" % said, op * show, ink=dark(GREEN if ok else RED),
                      size=0.036)
            self.text(sc, x, 0.735, "%.0f%% sure" % (100 * self.reads[i][said]), op * show)
        self.text(sc, 0.5, 0.86, "the same digit after the twelve-step attack, with a growing "
                  "budget", op, strong=True, size=0.034)

    def phase_3(self, sc, u, op):
        chart = self.chart(((0.10, 0.66), (0.37, 0.82)), self.eps, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "budget: the largest change to any pixel", "digits read correctly",
                  xticks=[(v, self._eps(i)) for i, v in enumerate(self.eps) if i != 1],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        self.plot(sc, chart, self.eps, self.fgsm, BLUE, op, u / 0.7, width=2.6)
        self.plot(sc, chart, self.eps, self.pgd, RED, op, u / 0.7, width=2.6)
        e = list(self.eps).index(self.EPS)
        self.lines(sc, 0.72, 0.42, [
            ("one step (FGSM)", dark(BLUE)),
            ("%.1f%% right at %.1f" % (100 * self.fgsm[e], self.EPS), dark(BLUE)),
        ], op, size=0.034, step=0.052)
        self.lines(sc, 0.72, 0.58, [
            ("twelve steps (PGD)", dark(RED)),
            ("%.1f%% right at %.1f" % (100 * self.pgd[e], self.EPS), dark(RED)),
        ], op, size=0.034, step=0.052)
        self.lines(sc, 0.72, 0.74, ["untouched: %.1f%% right" % (100 * self.fgsm[0]),
                                    "160 held-out digits"], op, step=0.048)

    def phase_4(self, sc, u, op):
        S = 76.0
        xs = [0.075 + i * 0.1 for i in range(4)]
        for i, x in enumerate(xs):
            self.picture(sc, paper(self.batch[i]), x, 0.43, S, S, op, frame=AXIS)
            self.picture(sc, paper(self.attacked[i]), x, 0.66, S, S,
                         op * smoothstep(0.05 + 0.05 * i, 0.15 + 0.05 * i, u), frame=RED)
            self.pulse(sc, (x, 0.50), (x, 0.59), (u - 0.02 - 0.05 * i) / 0.14, RED, op)
        self.text(sc, 0.225, 0.315, "training digits", op, strong=True)
        self.text(sc, 0.225, 0.545, "their attacked copies (budget %.1f)" % self.EPS, op,
                  ink=dark(RED))
        self.box(sc, 0.225, 0.87, 300, 44, soft(GREEN), op * smoothstep(0.25, 0.35, u),
                 "both go into 220 more updates", border=GREEN, size=0.030)
        for x in (xs[0], xs[-1]):
            self.pulse(sc, (x, 0.75), (0.225, 0.83), (u - 0.3) / 0.2, GREEN, op, bend=-0.02)
        full = self.span(0.25)[0]
        show = smoothstep(0.45, 0.55, u)
        groups = (("on attacked digits (one step, budget %.1f)" % self.EPS, self.bars[2],
                   self.bars[3], 0.40), ("on clean digits", self.bars[0], self.bars[1], 0.68))
        for title, before, after, y in groups:
            self.text(sc, 0.50, y - 0.055, title, op * show, anchor="lm", strong=True, size=0.032)
            self.hbar(sc, 0.62, y + 0.01, full * before, BLUE, op * show, "before",
                      "%.1f%%" % (100 * before), (u - 0.5) / 0.2, thick=24.0, track=full)
            self.hbar(sc, 0.62, y + 0.08, full * after, GREEN, op * show, "after",
                      "%.1f%%" % (100 * after), (u - 0.6) / 0.2, thick=24.0, track=full)


# ====================================================================== 55 object detection
_DETECTOR = {}


def detector_run(every=5):
    """The study's box regressor trained again (same scenes, seed and updates),
    keeping its two boxes for held-out scene 0 every ``every`` updates.
    Returns ``(steps, boxes (n, 2, 4) in pixels, losses, final parameters)``."""
    if every not in _DETECTOR:
        from . import toy2d as toy
        from .optim import Adam

        x, y, xt, yt = nv.mnist_data()
        xx, _, target = nv.composed_digits(x, y, 1200, nv.SEED)
        scene = nv.composed_digits(xt, yt, 80, nv.SEED + 1)[0][:1].reshape(1, -1)
        xx, target = xx.reshape(1200, -1), target.reshape(1200, -1) / 16
        rng = np.random.default_rng(51)
        p = toy.init((xx.shape[1], 64, 32, target.shape[1]), rng)
        opt, losses, steps, boxes = Adam(p, 0.004), [], [], []
        for step in range(1101):
            if step % every == 0:
                steps.append(step)
                boxes.append(toy.forward(p, scene)[0].reshape(2, 4) * 16)
            if step == 1100:
                break
            idx = rng.choice(len(xx), 96, replace=False)
            loss, g = toy.loss_grads(p, xx[idx], target[idx])
            opt.step(p, g)
            losses.append(loss)
        _DETECTOR[every] = (np.array(steps), np.array(boxes), np.array(losses), p)
    return _DETECTOR[every]


def nms_trace(boxes, scores, threshold=0.4):
    """Non-maximum suppression as the study runs it, step by step:
    ``[(kept, [(other, IoU with kept, removed), ...]), ...]``."""
    order, trace = list(np.argsort(-np.asarray(scores), kind="stable")), []
    while order:
        i, rest = order[0], order[1:]
        overlaps = nv.iou(boxes[i], boxes[rest]) if rest else []
        trace.append((int(i), [(int(j), float(v), bool(v > threshold))
                               for j, v in zip(rest, overlaps)]))
        order = [j for j, v in zip(rest, overlaps) if v <= threshold]
    return trace


class ObjectDetection(Vision):
    """Boxes around two MNIST digits: a box as four numbers, a regressor
    learning them, intersection over union as the score, and non-maximum
    suppression removing duplicate proposals."""

    TITLE = "Object detection: boxes, IoU, suppression"
    PHASES = (("a box", 9.0), ("learning boxes", 13.0), ("overlap", 12.0), ("duplicates", 15.0))
    SCENE = (0.23, 0.615, 232.0)
    WHO = ("the one", "the seven")
    SCORES = (0.9, 0.7, 0.6, 0.9, 0.7, 0.6)

    def setup(self):
        from . import toy2d as toy

        s = self.study
        d = study_arrays("object-detection-basics")
        self.scenes = np.asarray(d["test_images"], float)[..., 0]
        self.truth = np.asarray(d["test_boxes"], float)
        weights = {k[8:]: np.asarray(v) for k, v in d.items() if k.startswith("weights_")}
        self.pred = toy.forward(weights, self.scenes.reshape(80, -1))[0].reshape(80, 2, 4) * 16
        self.loss = np.asarray(s.panels[1].data["ys"], float)[0]
        self.overlaps = np.asarray(d["overlaps"], float).reshape(80, 2)
        self.mean_iou = float(self.overlaps.mean())
        self.proposals = np.asarray(d["boxes"], float)
        self.keep = [int(k) for k in d["keep"]]
        self.trace = nms_trace(self.proposals, np.array(self.SCORES))
        self.steps, self.path, _, _ = detector_run()

    def subtitle_for(self, k):
        t = self.truth[0].astype(int)
        return [
            "a scene of two real MNIST digits. Finding an object means drawing a box round it, "
            "and a box is four numbers: left %d, top %d, right %d, bottom %d for the one" % (
                t[0][0], t[0][1], t[0][2], t[0][3]),
            "a small network learns to output both boxes from the pixels: 1,100 updates on "
            "1,200 scenes. Its boxes for a scene it never saw close in on the true ones",
            "the score for a box is its overlap with the true one divided by the area the two "
            "cover together (IoU). Over 80 held-out scenes it averages %.3f" % self.mean_iou,
            "detectors propose many boxes per object. Here three copies of each box, a pixel "
            "apart, scored by hand: keep the best, drop any that overlap it by more than 0.4. "
            "%d of %d remain" % (len(self.keep), len(self.proposals)),
        ][k]

    def _scene(self, sc, op, i=0, dim=1.0):
        x, y, S = self.SCENE
        return self.picture(sc, paper(self.scenes[i] * dim), x, y, S, S, op, grid=0.12)

    def _box(self, sc, at, box, ink, op, width=3.0, fill=0.0):
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return
        cell = self.SCENE[2] / 16
        c = at((y1 + y2) / 2 - 0.5, (x1 + x2) / 2 - 0.5)
        if fill > 0:
            self.rect(sc, self.P(*c, 0.5), (x2 - x1) * cell, (y2 - y1) * cell, ink, op * fill,
                      radius=1, lift=0.6)
        self.outline(sc, c[0], c[1], (x2 - x1) * cell, (y2 - y1) * cell, ink, op, width)

    def phase_1(self, sc, u, op):
        at = self._scene(sc, op)
        self.text(sc, self.SCENE[0], 0.325, "16 × 16 pixels", op, strong=True, size=0.034)
        names = ("left", "top", "right", "bottom")
        for k, (who, ink) in enumerate(zip(self.WHO, (RED, BLUE))):
            x1, y1, x2, y2 = self.truth[0][k]
            start = 0.08 + 0.42 * k
            sides = (((y1, x1), (y2, x1)), ((y1, x1), (y1, x2)), ((y1, x2), (y2, x2)),
                     ((y2, x1), (y2, x2)))
            self.text(sc, 0.52 + 0.24 * k, 0.40, who, op * smoothstep(start, start + 0.05, u),
                      anchor="lm", ink=dark(ink), size=0.040)
            for j, (a, b) in enumerate(sides):
                t = start + 0.08 * j
                grow = smoothstep(t, t + 0.07, u)
                if grow > 0:
                    a, b = at(a[0] - 0.5, a[1] - 0.5), at(b[0] - 0.5, b[1] - 0.5)
                    self.line(sc, a, (a[0] + (b[0] - a[0]) * grow, a[1] + (b[1] - a[1]) * grow),
                              ink, op, 3.2)
                self.text(sc, 0.52 + 0.24 * k, 0.49 + 0.075 * j, "%s  %d" % (
                    names[j], self.truth[0][k][j]), op * grow, anchor="lm", size=0.036, ink=INK)
        self.text(sc, 0.52, 0.86, "pixels are counted from the top-left corner", op, anchor="lm")

    def phase_2(self, sc, u, op):
        at = self._scene(sc, op)
        go = min(u / 0.88, 1.0) ** 2.4
        n = int(go * (len(self.steps) - 1))
        for k in range(2):
            self._box(sc, at, self.truth[0][k], GREEN, op, 2.4)
            self._box(sc, at, np.clip(self.path[n][k], -0.4, 16.4), AMBER, op, 3.2)
        step = int(self.steps[n])
        self.text(sc, self.SCENE[0], 0.325, "a scene it never trained on", op, strong=True,
                  size=0.034)
        self.text(sc, self.SCENE[0], 0.895, "green: the true boxes", op, ink=dark(GREEN, 0.3))
        self.text(sc, self.SCENE[0], 0.94, "amber: the network's", op, ink=dark(AMBER, 0.35))
        xs = np.arange(1.0, len(self.loss) + 1)
        chart = self.chart(((0.55, 0.95), (0.42, 0.78)), xs, self.loss, ylog=True)
        self.axes(sc, chart, op, "update", "box error",
                  xticks=[(v, "%d" % v) for v in (1, 300, 600, 900, 1100)],
                  yticks=[(v, "%g" % v) for v in (0.0001, 0.001, 0.01, 0.1)])
        self.plot(sc, chart, xs, self.loss, AMBER, op, max(step, 1) / len(self.loss), width=2.0)
        both = [float(nv.iou(self.truth[0][k], self.path[n][k:k + 1])[0]) for k in range(2)]
        self.text(sc, 0.75, 0.325, "update %d" % step, op, strong=True, size=0.038)
        self.text(sc, 0.75, 0.895, "overlap with the truth: %.2f and %.2f" % tuple(both), op,
                  ink=INK, size=0.032)

    def phase_3(self, sc, u, op):
        shown = (0, 1, 2)
        turn = min(int(u / 0.72 * len(shown)), len(shown) - 1)
        within = min(u / 0.72 * len(shown) - turn, 1.0) if u < 0.72 else 1.0
        i = shown[turn]
        at = self._scene(sc, op, i, 0.55)
        cut = smoothstep(0.15, 0.4, within)
        for k in range(2):
            t, p = self.truth[i][k], self.pred[i][k]
            inter = np.r_[np.maximum(t[:2], p[:2]), np.minimum(t[2:], p[2:])]
            self._box(sc, at, inter, VIOLET, op * cut, 1.0, fill=0.38)
            self._box(sc, at, t, GREEN, op, 2.6)
            self._box(sc, at, p, AMBER, op, 2.6)
        self.text(sc, self.SCENE[0], 0.325, "held-out scene %d" % (i + 1), op, strong=True,
                  size=0.034)
        self.lines(sc, 0.50, 0.37, [
            ("green: true box     amber: predicted box", INK),
            ("violet: the part they share", dark(VIOLET)),
        ], op, size=0.032, step=0.055)
        for k, who in enumerate(self.WHO):
            t, p = self.truth[i][k], self.pred[i][k]
            inter = float(np.prod(np.maximum(np.minimum(t[2:], p[2:]) - np.maximum(t[:2], p[:2]),
                                             0)))
            union = float(np.prod(t[2:] - t[:2]) + np.prod(p[2:] - p[:2]) - inter)
            self.text(sc, 0.50, 0.535 + 0.07 * k, "%s:  %.1f shared ÷ %.1f covered = IoU %.2f" % (
                who, inter, union, self.overlaps[i, k]), op * smoothstep(0.4, 0.55, within),
                anchor="lm", size=0.034, ink=INK)
        done = smoothstep(0.74, 0.84, u)
        full = self.span(0.30)[0]
        self.text(sc, 0.50, 0.735, "average over 160 boxes in 80 held-out scenes", op * done,
                  anchor="lm", strong=True)
        self.hbar(sc, 0.50, 0.81, full * self.mean_iou, GREEN, op * done, "",
                  "IoU %.3f" % self.mean_iou, (u - 0.76) / 0.15, thick=28.0, track=full)
        self.text(sc, 0.50, 0.885, "1 would be a perfect fit", op * done, anchor="lm")

    def phase_4(self, sc, u, op):
        at = self._scene(sc, op, 0, 0.45)
        # one beat to show the six, then two beats per kept box: choose it, compare the rest
        beats = 1 + 2 * len(self.trace)
        beat = min(int(u / 0.9 * beats), beats - 1) if u < 0.9 else beats - 1
        within = min(u / 0.9 * beats - beat, 1.0) if u < 0.9 else 1.0
        kept, gone, current, judged = [], {}, None, {}
        for n, (i, others) in enumerate(self.trace):
            if beat >= 1 + 2 * n:
                current = i
                kept.append(i)
            if beat >= 2 + 2 * n:
                for j, v, removed in others:
                    judged[j] = v
                    if removed:
                        gone[j] = 1.0 if beat > 2 + 2 * n else smoothstep(0.3, 0.8, within)
        order = [i for i, _ in sorted(enumerate(self.SCORES), key=lambda kv: (-kv[1], kv[0]))]
        self.text(sc, self.SCENE[0], 0.325, "six proposals", op, strong=True, size=0.034)
        self.text(sc, 0.50, 0.325, "proposal", op, anchor="lm", strong=True)
        self.text(sc, 0.735, 0.325, "score", op, strong=True)
        self.text(sc, 0.86, 0.325, "overlap with kept", op, strong=True)
        for row, i in enumerate(order):
            ink = RED if i < 3 else BLUE
            fade = 1.0 - 0.85 * gone.get(i, 0.0)
            is_kept = i in kept
            self._box(sc, at, np.clip(self.proposals[i], -0.4, 16.4), GREEN if is_kept else ink,
                      op * fade, 3.6 if is_kept else 1.8)
            y = 0.40 + row * 0.075
            self.patch(sc, 0.515, y, 16, 16, GREEN if is_kept else ink, op * fade)
            self.text(sc, 0.54, y, "%s, copy %d" % (self.WHO[i // 3], i % 3 + 1), op * fade,
                      anchor="lm", ink=INK)
            self.text(sc, 0.735, y, "%.1f" % self.SCORES[i], op * fade, ink=INK, size=0.034)
            if is_kept:
                self.text(sc, 0.86, y, "kept: the best left", op, ink=dark(GREEN, 0.3))
            elif i in judged:
                removed = i in gone
                self.text(sc, 0.86, y, "%.2f  %s" % (judged[i], "removed" if removed else
                                                    "stays"), op * (fade if removed else 1.0),
                          ink=dark(RED) if removed else INK)
            if i == current and beat % 2 == 1:
                self.outline(sc, 0.73, y, 410, 32, GREEN, op * (1 - smoothstep(0.7, 1.0, within))
                             if beat < beats - 1 else 0.0, 2.0, radius=5)
        self.text(sc, 0.725, 0.885, "%d of %d proposals remain: one box per digit" % (
            len(self.keep), len(self.proposals)), op * smoothstep(0.9, 0.97, u), strong=True,
            size=0.034)


# ====================================================================== 56 segmentation
CLASS_INKS = (PAPER, RED, BLUE)        # background, the one, the seven


def mask_rgb(mask, soften=0.25):
    """A class mask (0 background, 1 one, 2 seven) in the class colours."""
    inks = np.array([PAPER, soft(RED, soften), soft(BLUE, soften)])
    return inks[np.asarray(mask, int)]


def unet_parts(p, x, skip=1.0, coarse=1.0):
    """The study's U-Net forward pass (:func:`vm.unet_loss`) with its pieces
    kept: ``(predicted classes, h0 (N,16,16,4), h1 (N,8,8,8))``. ``skip`` and
    ``coarse`` scale the two inputs of the last layer (0 cuts that path)."""
    from . import autoencoder as ae

    h0 = np.tanh(ae.conv(x, p["W0"], p["b0"], 1, 1)[0])
    n, h, w, c = h0.shape
    pooled = h0.reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4))
    h1 = np.tanh(ae.conv(pooled, p["W1"], p["b1"], 1, 1)[0])
    joined = np.concatenate((h0 * skip, ae.up2(h1) * coarse), axis=-1)
    logits = ae.conv(joined, p["W2"], p["b2"], 1, 0)[0]
    return logits.argmax(-1), h0, h1


def class_iou(pred, truth):
    return np.array([np.sum((pred == c) & (truth == c)) / max(np.sum((pred == c) | (truth == c)), 1)
                     for c in range(3)])


class SemanticSegmentation(Vision):
    """A label for every pixel of a two-digit MNIST scene: the U-Net drawn with
    its real feature maps, what the skip connection and the coarse path each
    carry, and intersection over union per class."""

    TITLE = "Semantic segmentation with a U-Net"
    PHASES = (("a label per pixel", 9.0), ("the U", 17.0), ("two paths", 13.0),
              ("per class", 12.0))
    NAMES = ("background", "the one", "the seven")

    def setup(self):
        s = self.study
        d = study_arrays("semantic-segmentation")
        self.p = {k[8:]: np.asarray(v) for k, v in d.items() if k.startswith("weights_")}
        self.images = np.asarray(d["test_images"], float)
        self.truth = np.asarray(d["test_masks"], int)
        self.pred, self.h0, self.h1 = unet_parts(self.p, self.images)
        self.iou = [float(v) for v in s.panels[3].data["values"]]
        self.updates = len(s.panels[1].data["ys"][0])
        self.no_skip = unet_parts(self.p, self.images, skip=0.0)[0]
        self.no_coarse = unet_parts(self.p, self.images, coarse=0.0)[0]
        self.iou_no_skip = class_iou(self.no_skip, self.truth)
        self.iou_no_coarse = class_iou(self.no_coarse, self.truth)

    def subtitle_for(self, k):
        return [
            "segmentation gives every pixel a label. In this 16 × 16 scene of two real MNIST "
            "digits each of the 256 pixels is background, part of the one, or part of the seven",
            "the U-Net, trained for %s updates: 4 feature maps at full size, shrunk to 8 maps "
            "at half size that see more of the scene, enlarged again and joined with the "
            "full-size maps, which skip across" % f"{self.updates:,}",
            "cut one path of the trained network and score how well its pixels overlap the true "
            "ones (1 is perfect): for the one, %.2f whole, %.2f without the skip, %.2f without "
            "the half-size path. It needs both" % (
                self.iou[1], self.iou_no_skip[1], self.iou_no_coarse[1]),
            "the score per class is the overlap of predicted and true pixels over their union, "
            "on 80 held-out scenes: background %.3f, the one %.3f, the seven %.3f" % tuple(
                self.iou),
        ][k]

    def _legend(self, sc, fx, fy, op, step=0.05):
        for k, name in enumerate(self.NAMES):
            self.rect(sc, self.P(fx, fy + k * step), 18, 18, mask_rgb([k])[0], op, border=AXIS,
                      border_w=1.0, radius=2)
            self.text(sc, fx + 0.022, fy + k * step, name, op, anchor="lm", ink=INK)

    def phase_1(self, sc, u, op):
        S = 220.0
        k = int(min(u / 0.85, 1.0) * 256)
        at = self.picture(sc, paper(self.images[0, ..., 0]), 0.22, 0.61, S, S, op, grid=0.15)
        rgb = mask_rgb(self.truth[0]).reshape(256, 3)
        rgb[k:] = PAPER * 0.90
        mat = self.picture(sc, rgb.reshape(16, 16, 3), 0.60, 0.61, S, S, op, grid=0.15)
        if k < 256:
            r, c = divmod(k, 16)
            self.outline(sc, *at(r, c), S / 16 + 2, S / 16 + 2, AMBER, op, 2.4)
            self.outline(sc, *mat(r, c), S / 16 + 2, S / 16 + 2, AMBER, op, 2.4)
        self.text(sc, 0.22, 0.325, "the scene", op, strong=True, size=0.034)
        self.text(sc, 0.60, 0.325, "its true labels", op, strong=True, size=0.034)
        self._legend(sc, 0.79, 0.50, op, 0.065)
        self.text(sc, 0.41, 0.905, "pixel %d of 256" % min(k + 1, 256), op, ink=dark(AMBER, 0.35),
                  size=0.034)

    def _maps(self, sc, maps, fx, fy, size, cols, op, gap=5.0):
        """Feature maps ``(h, w, n)`` as a block of small pictures; returns the
        block's half-extent in layout fractions."""
        n = maps.shape[-1]
        rows = int(np.ceil(n / cols))
        dx, dy = self.frac(size + gap, size + gap)
        for i in range(n):
            x = fx + (i % cols - (cols - 1) / 2) * dx
            y = fy + (i // cols - (rows - 1) / 2) * dy
            self.picture(sc, heat(maps[..., i], hot=VIOLET, cold=TEAL), x, y, size, size, op,
                         frame=AXIS)
        return cols * dx / 2, rows * dy / 2

    def phase_2(self, sc, u, op):
        top, low = 0.51, 0.82
        x_in, x_h0, x_h1, x_join, x_out = 0.085, 0.29, 0.50, 0.71, 0.905
        self.picture(sc, paper(self.images[0, ..., 0]), x_in, top, 104, 104, op)
        self.text(sc, x_in, 0.315, "scene", op, strong=True)
        s1 = smoothstep(0.06, 0.16, u)
        self.pulse(sc, (x_in + 0.07, top), (x_h0 - 0.09, top), (u - 0.02) / 0.14, VIOLET, op, r=6)
        self._maps(sc, self.h0[0], x_h0, top, 62, 2, op * s1)
        self.text(sc, x_h0, 0.315, "4 maps, 16 × 16", op * s1, strong=True)
        s2 = smoothstep(0.26, 0.36, u)
        self.arc(sc, (x_h0 + 0.02, top + 0.15), (x_h1 - 0.165, low), VIOLET, op * s1,
                 smoothstep(0.16, 0.28, u), -0.05, 2.6)
        self.pulse(sc, (x_h0 + 0.02, top + 0.15), (x_h1 - 0.165, low), (u - 0.18) / 0.16, VIOLET,
                   op, -0.05, r=6)
        self._maps(sc, self.h1[0], x_h1, low, 54, 4, op * s2)
        self.text(sc, x_h1 - 0.26, low + 0.045, "halve, then 8 maps, 8 × 8", op * s2,
                  ink=dark(VIOLET))
        self.text(sc, x_h1 + 0.265, low + 0.045, "double the size again", op *
                  smoothstep(0.42, 0.5, u), ink=dark(VIOLET))
        s3 = smoothstep(0.50, 0.60, u)
        self.arc(sc, (x_h1 + 0.165, low), (x_join - 0.01, top + 0.115), VIOLET, op * s2,
                 smoothstep(0.38, 0.5, u), -0.05, 2.6)
        self.pulse(sc, (x_h1 + 0.165, low), (x_join - 0.01, top + 0.115), (u - 0.40) / 0.16,
                   VIOLET, op, -0.05, r=6)
        # the skip: the full-size maps go straight across
        self.arc(sc, (x_h0 + 0.09, top - 0.04), (x_join - 0.085, top - 0.04), AMBER, op * s1,
                 smoothstep(0.52, 0.66, u), 0.03, 3.0)
        for lag in (0.0, 0.05, 0.10):
            self.pulse(sc, (x_h0 + 0.09, top - 0.04), (x_join - 0.085, top - 0.04),
                       (u - 0.54 - lag) / 0.16, AMBER, op, 0.03, r=6)
        self.text(sc, (x_h0 + x_join) / 2 + 0.035, top - 0.115, "the skip: full-size maps, unchanged",  # noqa: E501
                  op * smoothstep(0.52, 0.6, u), ink=dark(AMBER, 0.35), size=0.032)
        self.box(sc, x_join, top, 138, 96, soft(GREEN, 0.7), op * s3, border=GREEN)
        self.lines(sc, x_join, top - 0.045, ["4 + 8 maps", "per pixel:", "3 scores"], op * s3,
                   anchor="mm", step=0.045)
        self.text(sc, x_join, 0.315, "join", op * s3, strong=True)
        s4 = smoothstep(0.72, 0.80, u)
        self.pulse(sc, (x_join + 0.085, top), (x_out - 0.07, top), (u - 0.68) / 0.12, GREEN, op,
                   r=6)
        k = int(smoothstep(0.76, 0.95, u) * 256)
        rgb = mask_rgb(self.pred[0]).reshape(256, 3)
        rgb[k:] = PAPER * 0.90
        self.picture(sc, rgb.reshape(16, 16, 3), x_out, top, 104, 104, op * s4)
        self.text(sc, x_out, 0.315, "labels", op * s4, strong=True)

    def phase_3(self, sc, u, op):
        S = 150.0
        cols = (
            ("the truth", self.truth[0], None, 0.0),
            ("the whole U-Net", self.pred[0], class_iou(self.pred, self.truth), 0.12),
            ("skip cut", self.no_skip[0], self.iou_no_skip, 0.36),
            ("half-size path cut", self.no_coarse[0], self.iou_no_coarse, 0.60),
        )
        for i, (name, mask, iou, start) in enumerate(cols):
            x = 0.14 + i * 0.24
            show = smoothstep(start, start + 0.1, u)
            self.picture(sc, mask_rgb(mask), x, 0.535, S, S, op * show, grid=0.15)
            self.text(sc, x, 0.335, name, op * show, strong=True, size=0.034)
            if iou is not None:
                self.text(sc, x, 0.735, "the one  %.2f" % iou[1], op * show,
                          ink=dark(RED), size=0.034)
                self.text(sc, x, 0.79, "the seven  %.2f" % iou[2], op * show,
                          ink=dark(BLUE), size=0.034)
        self.text(sc, 0.14, 0.76, "overlap, 80 scenes:", op, ink=INK)
        self.text(sc, 0.5, 0.895, "the skip says where the ink is; the half-size path, seeing "
                  "more of the scene, says which digit it belongs to", op * smoothstep(0.75, 0.85, u),  # noqa: E501
                  ink=dark(AMBER, 0.35))

    def phase_4(self, sc, u, op):
        S = 170.0
        shown = (0, 1, 2)
        turn = min(int(u / 0.7 * len(shown)), len(shown) - 1)
        i = shown[turn]
        wrong = self.pred[i] != self.truth[i]
        rgb = mask_rgb(self.pred[i])
        rgb[wrong] = AMBER
        self.picture(sc, paper(self.images[i, ..., 0]), 0.14, 0.58, S, S, op, grid=0.15)
        self.picture(sc, rgb, 0.385, 0.58, S, S, op, grid=0.15)
        self.text(sc, 0.14, 0.335, "held-out scene %d" % (turn + 1), op, strong=True, size=0.034)
        self.text(sc, 0.385, 0.335, "predicted labels", op, strong=True, size=0.034)
        self.text(sc, 0.385, 0.815, "amber: %d of 256 pixels wrong" % int(wrong.sum()), op,
                  ink=dark(AMBER, 0.35), size=0.032)
        full = self.span(0.24)[0]
        self.text(sc, 0.58, 0.40, "overlap ÷ union, 80 held-out scenes", op, anchor="lm",
                  strong=True)
        for k, (name, ink) in enumerate(zip(self.NAMES, (AXIS, RED, BLUE))):
            y = 0.52 + k * 0.12
            self.text(sc, 0.58, y - 0.048, name, op, anchor="lm", ink=dark(ink, 0.3))
            self.hbar(sc, 0.58, y, full * self.iou[k], ink, op, "", "%.3f" % self.iou[k],
                      u / 0.35, thick=24.0, track=full)


# ====================================================================== 57 batch norm
class BatchNormInCnns(Vision):
    """Batch normalisation in a CNN trained with it (:func:`voxanim.study_vision.batchnorm`):
    the statistics gathered per filter, the normalisation itself, stored statistics against
    a batch's own, and what using the batch's own at test time costs in accuracy."""

    TITLE = "Batch normalisation in a CNN"
    PHASES = (("statistics", 12.0), ("normalise", 11.0), ("which statistics", 12.0),
              ("the cost", 13.0))
    STORED, BATCH, ONE = GREEN, TEAL, RED
    SHOWN = 4                    # of the first layer's eight filters

    def setup(self):
        s = self.study
        d = sv.batchnorm_run()
        p, _stats = sv.part(d, "weights_"), sv.part(d, "stats_")
        x = sv.mnist_data()[0]
        xt = sv.bn_test_set()[0]
        n = self.SHOWN
        self.train = sv.cnn_forward(p, x[:256])["z1"][..., :n]
        test = sv.cnn_forward(p, xt[:32])["z1"][..., :n]
        self.mu, self.var = self.train.mean((0, 1, 2)), self.train.var((0, 1, 2))
        self.stored = np.asarray(s.panels[0].data["values"], float)[:, :n]    # mean, variance
        self.means = np.vstack((self.stored[0], test.mean((0, 1, 2)), test[:1].mean((0, 1, 2))))
        self.accuracy = np.asarray(s.panels[2].data["seeds"], float)     # seed, [stored, 256..1]
        self.changed = np.asarray(s.panels[3].data["seeds"], float)      # seed, [256..1]
        self.one_digit = float(np.mean(d["one_digit"]))
        self.sizes = sv.BN_BATCHES
        self.reach = float(np.ceil(4 * np.sqrt(self.var.max()) * 2) / 2)  # histogram half-width
        self.bins = np.linspace(-self.reach, self.reach, 37)
        self.counts = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        flat = self.train.reshape(256, -1, n)
        self.partial = [(flat[:k].mean((0, 1)), flat[:k].var((0, 1)),
                         [np.histogram(flat[:k, :, c], self.bins)[0] / (k * flat.shape[1])
                          for c in range(n)]) for k in self.counts]
        self.flat = flat[:64].reshape(-1, n)
        # the tallest bar that is not the pile of identical blank-paper values
        self.cut = 1.4 * max(float(np.sort(h)[-2]) for h in self.partial[-1][2])

    def subtitle_for(self, k):
        m, v = self.mu, self.var
        acc = 100 * self.accuracy
        return [
            "this CNN was trained with batch normalisation. Each filter gets its own "
            "statistics, over every position of every image in the batch: after 256 images "
            "filter 1 has mean %.2f and variance %.2f, filter 2 %.2f and %.2f" % (
                m[0], v[0], m[1], v[1]),
            "batch normalisation subtracts each filter's mean and divides by its spread, so "
            "all four come out centred on 0 with variance 1, whatever they were before",
            "which mean? While training, the batch's own. Afterwards, the one stored during "
            "training: for filter 2 that is %.2f, a batch of 32 test images gives %.2f, and one "
            "image alone gives %.2f" % (self.means[0][1], self.means[1][1], self.means[2][1]),
            "the classic bug: still using the batch's own statistics at test time. Batches of "
            "256 lose nothing; one image at a time, accuracy falls from %.1f%% to %.1f%% "
            "(three trained networks: %.1f%% to %.1f%%)" % (
                acc[:, 0].mean(), acc[:, -1].mean(), acc[:, -1].min(), acc[:, -1].max()),
        ][k]

    def _hist(self, sc, share, fx, base, width, height, ink, op, top=None):
        """A histogram (shares per bin) ``width`` wide (layout) on baseline ``base``."""
        top = float(max(share)) if top is None else top
        n = len(share)
        bar = self.span(width)[0] / n
        for i, v in enumerate(share):
            if v > 0:                      # a bar taller than the scale is cut off, and darker
                self.vbar(sc, fx - width / 2 + (i + 0.5) * width / n, base, bar + 0.6,
                          height * min(v / top, 1.0), dark(ink, 0.45) if v > top else ink, op)

    def phase_1(self, sc, u, op):
        k = min(int(min(u / 0.85, 1.0) ** 1.5 * len(self.counts)), len(self.counts) - 1)
        n = self.counts[k]
        mean, var, hists = self.partial[k]
        self.text(sc, 0.5, 0.31, "image %d of 256 through the filters" % n, op, strong=True,
                  size=0.034)
        for c in range(4):
            x = 0.155 + c * 0.23
            fmap = self.train[n - 1, :, :, c]
            self.picture(sc, heat(fmap / self.reach, hot=VIOLET, cold=TEAL), x, 0.465, 104, 104,
                         op)
            self.text(sc, x, 0.605, "filter %d" % (c + 1), op, ink=dark(VIOLET), size=0.032)
            self._hist(sc, hists[c], x, 0.80, 0.19, 62.0, VIOLET, op, top=self.cut)
            self.line(sc, (x - 0.095, 0.80), (x + 0.095, 0.80), AXIS, op, 1.2)
            mx = x + 0.19 * mean[c] / (2 * self.reach)
            self.line(sc, (mx, 0.80), (mx, 0.655), AMBER, op, 2.2)
            self.text(sc, x, 0.845, "mean %.2f" % (mean[c] if abs(mean[c]) >= 0.005 else 0.0), op,
                      ink=dark(AMBER, 0.35), size=0.032)
            self.text(sc, x, 0.895, "variance %.2f" % var[c], op, ink=INK, size=0.032)
        self.text(sc, 0.5, 0.945, "how often each value occurs (the dark bar, blank paper, is "
                  "cut off)", op)

    def phase_2(self, sc, u, op):
        s = smoothstep(0.2, 0.8, u)
        x0, x1 = 0.36, 0.94
        centre = (x0 + x1) / 2
        bins = np.linspace(-4.0, 4.0, 65)
        for c in range(4):
            base = 0.44 + c * 0.125
            sd = np.sqrt(self.var[c] + 1e-5)
            v = (self.flat[:, c] - s * self.mu[c]) / (1 + s * (sd - 1))
            share = np.histogram(v, bins)[0] / len(v)
            self._hist(sc, share, centre, base, x1 - x0, 50.0, VIOLET, op, top=0.05)
            self.line(sc, (x0, base), (x1, base), AXIS, op, 1.0)
            self.text(sc, x0 - 0.015, base - 0.03, "filter %d" % (c + 1), op, anchor="rm",
                      ink=dark(VIOLET), size=0.032)
            mx = centre + (x1 - x0) * float(v.mean()) / 8.0
            self.line(sc, (mx, base), (mx, base - 0.105), AMBER, op, 2.2)
        for tick in (-4, -2, 0, 2, 4):
            self.text(sc, centre + (x1 - x0) * tick / 8.0, 0.865, "%d" % tick, op)
        self.text(sc, centre, 0.915, "value of the filter's output", op)
        self.lines(sc, 0.04, 0.42, [("for each filter:", INK)], op, size=0.034)
        self.lines(sc, 0.04, 0.50, ["subtract its mean", "(the amber line)"], op * smoothstep(
            0.1, 0.2, u), step=0.045)
        self.lines(sc, 0.04, 0.63, ["divide by its spread"], op * smoothstep(0.3, 0.4, u))
        self.lines(sc, 0.04, 0.74, [("every filter now:", INK), ("mean 0, variance 1", INK)],
                   op * smoothstep(0.8, 0.9, u), size=0.034, step=0.05)

    def phase_3(self, sc, u, op):
        sources = (("stored from training", "a running average", self.STORED, 0.08),
                   ("this batch", "32 test images", self.BATCH, 0.34),
                   ("a batch of one", "1 test image", self.ONE, 0.60))
        zero, scale = 0.60, 100.0 / float(np.abs(self.means).max())     # px per unit of mean
        for k, (name, detail, ink, start) in enumerate(sources):
            show = smoothstep(start, start + 0.08, u)
            y = 0.42 + k * 0.16
            self.rect(sc, self.P(0.055, y), 22, 22, ink, op * show, radius=3)
            self.text(sc, 0.08, y - 0.022, name, op * show, anchor="lm", ink=dark(ink, 0.3),
                      size=0.034)
            self.text(sc, 0.08, y + 0.03, detail, op * show, anchor="lm")
            for c in range(4):
                x = 0.365 + c * 0.166 + (k - 1) * 0.054
                v = float(self.means[k][c])
                h = abs(v) * scale * smoothstep(start + 0.04, start + 0.2, u)
                at = self.shift(self.P(x, zero), 0, -np.sign(v) * h / 2)
                self.rect(sc, at, 30, max(h, 0.6), ink, op * show, radius=2)
                dy = self.frac(0, h)[1] + 0.03
                self.text(sc, x, zero - np.sign(v) * dy, ("%.2f" % (v + 0.0 if abs(v) >= 0.005
                                                                     else 0.0)).replace("-", "−"),
                          op * smoothstep(start + 0.16, start + 0.22, u))
                self.pulse(sc, (0.30, y), (x, zero), (u - start - 0.02) / 0.16, ink, op, 0.03)
        for c in range(4):
            x = 0.365 + c * 0.166
            self.text(sc, x, 0.325, "filter %d" % (c + 1), op, ink=dark(VIOLET), size=0.034)
            self.line(sc, (x - 0.08, zero), (x + 0.08, zero), AXIS, op, 1.4)
        self.text(sc, 0.65, 0.905, "the mean each source would subtract", op, strong=True)

    def phase_4(self, sc, u, op):
        wrong = 100 * (1 - self.accuracy)                       # seed, [stored, 256 .. 1]
        mean, top = wrong.mean(0), float(wrong.max())
        base, tall = 0.80, 190.0
        names = ["stored"] + [str(b) for b in self.sizes]
        self.text(sc, 0.345, 0.31, "test digits read wrongly, %", op, strong=True, size=0.034)
        for i, name in enumerate(names):
            x = 0.085 + i * 0.104
            ink = self.STORED if i == 0 else (self.ONE if i == len(names) - 1 else self.BATCH)
            grow = smoothstep(0.04 + 0.09 * i, 0.16 + 0.09 * i, u)
            self.vbar(sc, x, base, 46, tall * mean[i] / top * grow, ink, op)
            for k, v in enumerate(wrong[:, i]):
                self.disk(sc, self.P(x + (k - 1) * 0.017, base - self.frac(0, tall * v / top)[1]),
                          4.5, dark(ink, 0.45), op * smoothstep(0.6, 1.0, grow), lift=1.4,
                          outline=0.0)
            self.text(sc, x, base - self.frac(0, tall * wrong[:, i].max() / top)[1] - 0.035,
                      "%.1f" % mean[i], op * smoothstep(0.8, 1.0, grow), ink=INK, size=0.032)
            self.text(sc, x, base + 0.04, name, op, ink=dark(ink, 0.3), size=0.032)
        self.line(sc, (0.03, base), (0.66, base), AXIS, op, 1.4)
        self.text(sc, 0.085, 0.905, "statistics", op, ink=dark(self.STORED, 0.3))
        self.text(sc, 0.397, 0.905, "the batch's own statistics, by images per batch", op,
                  ink=dark(self.BATCH, 0.3))
        show = smoothstep(0.62, 0.72, u)
        self.lines(sc, 0.70, 0.40, [
            ("bars: mean of three", INK), ("trained networks", INK), ("dots: each network", INK),
        ], op, step=0.045)
        self.lines(sc, 0.70, 0.58, [
            ("one image per batch:", dark(self.ONE, 0.3)),
            ("%.1f%% of the answers" % (100 * self.changed[:, -1].mean()), INK),
            ("change", INK),
        ], op * show, step=0.045, size=0.032)
        self.lines(sc, 0.70, 0.76, [
            ("32 images of one digit", dark(self.BATCH, 0.3)),
            ("per batch: %.1f%% wrong" % (100 * (1 - self.one_digit)), INK),
        ], op * smoothstep(0.76, 0.86, u), step=0.045, size=0.032)


# ====================================================================== 58 style transfer
class StyleTransfer(Vision):
    """Style as which filters fire together (:func:`voxanim.study_vision.style_transfer`):
    the Gram matrix of a trained MNIST CNN's feature maps built cell by cell, pixels
    optimised to match a heavy digit's Gram matrices while keeping another digit's maps,
    then twenty digits rendered in a thin and in a heavy digit's style."""

    TITLE = "Style transfer with Gram matrices"
    PHASES = (("feature maps", 10.0), ("Gram matrix", 15.0), ("optimise pixels", 15.0),
              ("result", 12.0))
    FILTERS = (RED, BLUE, GREEN, AMBER)
    STYLES = (TEAL, VIOLET)             # thin, heavy
    SHOWN = 1                           # the style the first three phases follow: heavy

    def setup(self):
        s = self.study
        d = sv.style_run()
        p = sv.part(d, "weights_")
        self.contents = np.asarray(d["content"], float)                  # (20, 28, 28)
        self.styles = np.asarray(d["style"], float)                      # thin, heavy
        self.results = np.asarray(d["result"], float)                    # style, digit, 28, 28
        self.content, self.style = self.contents[0], self.styles[self.SHOWN]
        f = sv.cnn_forward(p, np.stack((self.content, self.style)), pad=True)
        # the four first-layer filters the style digit excites most, in filter order
        self.picked = np.sort(np.argsort(-f["a1"][1].mean((0, 1)))[:4])
        self.maps = [f["a1"][i][..., self.picked] for i in (0, 1)]
        self.top = float(max(m.max() for m in self.maps))
        self.full = np.asarray(s.panels[1].data["values"], float)        # 16 x 16, layer 1
        self.gram = self.full[np.ix_(self.picked, self.picked)]
        self.gmax = float(self.gram.max())
        self.images = np.asarray(d["snaps"], float)[self.SHOWN, :, 0]    # every SNAP steps
        self.steps = np.arange(len(self.images)) * sv.SNAP
        self.grams = sv.grams(sv.cnn_forward(p, self.images, pad=True))[1]
        self.loss = np.asarray(d["mismatch"], float)[self.SHOWN, :, 0]   # this digit, per step
        self.mean = np.asarray(s.panels[3].data["ys"], float)            # style, step
        self.ink = s.metadata["ink"]
        self.same = s.metadata["read_as_same_digit"]

    def subtitle_for(self, k):
        g, f = self.gram, self.picked + 1
        ink = self.ink
        return [
            "two real digits go through a CNN trained on MNIST: 16 filters, then 32 (four of "
            "the 16 are shown). One digit supplies the content, where the strokes are; the "
            "other the style, read off the same maps in a different way",
            "style is which filters fire together, wherever they fire: multiply two feature "
            "maps position by position and average. Filters %d and %d give %.2f, filters %d "
            "and %d %.2f; over all 16 filters this table is the Gram matrix" % (
                f[1], f[3], g[1, 3], f[0], f[2], g[0, 2]),
            "start from the content digit and change its pixels, 300 small steps, so that its "
            "Gram matrices move towards the heavy digit's while its second-layer maps stay "
            "close: the mismatch falls from %.2f to %.2f" % (self.loss[0], self.loss[-1]),
            "twenty digits, two styles. Towards the thin digit their ink falls from %.0f "
            "pixels' worth to %.0f; towards the heavy one it rises to %.0f. The CNN still reads "
            "%d and %d of the 20 as the same digit" % (
                ink["content"], ink["thin_result"], ink["heavy_result"], *self.same),
        ][k]

    def _fmap(self, sc, m, c, fx, fy, size, op, top=None):
        top = self.top if top is None else top
        self.picture(sc, tint(np.full(m.shape + (3,), PAPER), np.sqrt(np.clip(m / top, 0, 1)),
                              dark(self.FILTERS[c], 0.1)), fx, fy, size, size, op,
                     frame=self.FILTERS[c])

    def phase_1(self, sc, u, op):
        for row, (name, img, maps, note) in enumerate((
                ("content", self.content, self.maps[0], "kept: where each filter fires"),
                ("style", self.style, self.maps[1], "kept: which filters fire together"))):
            y = 0.44 + row * 0.31
            start = 0.05 + 0.42 * row
            self.picture(sc, paper(img), 0.12, y, 112, 112, op)
            self.text(sc, 0.12, y - 0.145, name, op, strong=True, size=0.034)
            for c in range(4):
                x = 0.34 + c * 0.135
                t = start + 0.07 * c
                self.pulse(sc, (0.19, y), (x - 0.05, y), (u - t) / 0.12, self.FILTERS[c], op, r=5)
                self._fmap(sc, maps[..., c], c, x, y, 96, op * smoothstep(t + 0.08, t + 0.16, u))
                if row == 0:
                    self.text(sc, x, 0.295, "filter %d" % (self.picked[c] + 1), op,
                              ink=dark(self.FILTERS[c]))
            self.lines(sc, 0.875, y - 0.025, note.split(": ")[0:1] + fit_two(note.split(": ")[1]),
                       op * smoothstep(start + 0.3, start + 0.4, u), anchor="mm", step=0.045)

    def phase_2(self, sc, u, op):
        maps = self.maps[1]
        k = min(int(min(u / 0.9, 1.0) * 16), 15)
        within = min(u / 0.9 * 16 - k, 1.0) if u < 0.9 else 1.0
        i, j = divmod(k, 4)
        G, T = 192.0, 44.0
        gx, gy = 0.30, 0.68
        shown = np.full(16, np.nan)
        shown[:k + (1 if within > 0.5 or u >= 0.9 else 0)] = self.gram.ravel()[
            :k + (1 if within > 0.5 or u >= 0.9 else 0)]
        rgb = heat(np.nan_to_num(shown.reshape(4, 4)) / self.gmax, hot=VIOLET)
        rgb[np.isnan(shown.reshape(4, 4))] = PAPER * 0.90
        at = self.picture(sc, rgb, gx, gy, G, G, op, grid=0.3)
        for c in range(4):
            x, y = at(-1.15, c)
            self._fmap(sc, maps[..., c], c, x, y, T, op)
            x, y = at(c, -1.15)
            self._fmap(sc, maps[..., c], c, x, y, T, op)
        for r in range(4):
            for c in range(4):
                if not np.isnan(shown[r * 4 + c]):
                    self.text(sc, *at(r, c), "%.2f" % self.gram[r, c], op, ink=INK
                              if self.gram[r, c] < 0.6 * self.gmax else WHITE, size=0.032)
        if u < 0.9:
            self.outline(sc, *at(i, j), G / 4 + 2, G / 4 + 2, INK, op, 2.6)
            self.outline(sc, *at(-1.15, j), T + 9, T + 9, INK, op, 2.4)
            self.outline(sc, *at(i, -1.15), T + 9, T + 9, INK, op, 2.4)
        self.text(sc, gx - 0.03, 0.315, "four of the style digit's maps, against each other", op,
                  strong=True)
        # the product being averaged
        product = maps[..., i] * maps[..., j]
        px, py, M = 0.60, 0.55, 92.0
        self._fmap(sc, maps[..., i], i, px, py, M, op)
        self._fmap(sc, maps[..., j], j, px + 0.14, py, M, op)
        self.text(sc, px + 0.07, py, "×", op, strong=True, size=0.045)
        self.text(sc, px + 0.21, py, "=", op, strong=True, size=0.045)
        self.picture(sc, heat(np.sqrt(product) / self.top, hot=VIOLET), px + 0.28, py, M, M, op,
                     frame=VIOLET)
        self.text(sc, px + 0.14, 0.375, "filter %d × filter %d, at every position" % (
            self.picked[i] + 1, self.picked[j] + 1), op, strong=True, size=0.034)
        both = int(np.count_nonzero(product))
        self.lines(sc, px + 0.14, 0.73, [
            "both fire at %d of %d positions" % (both, product.size),
            "average of the products: %.2f" % self.gram[i, j],
        ], op, anchor="mm", step=0.055, size=0.032)
        self.text(sc, px + 0.14, 0.885, "where they fire is thrown away", op * smoothstep(
            0.5, 0.6, u), ink=dark(VIOLET))

    def phase_3(self, sc, u, op):
        go = min(u / 0.88, 1.0)
        n = int(go * (len(self.steps) - 1))
        step = int(self.steps[n])
        S, G = 150.0, 132.0
        top = float(self.full.max())
        self.picture(sc, paper(self.images[n]), 0.125, 0.56, S, S, op)
        self.text(sc, 0.125, 0.325, "the image, step %d" % step, op, strong=True, size=0.034)
        self.picture(sc, heat(self.grams[n] / top, hot=VIOLET), 0.355, 0.56, G, G, op)
        self.text(sc, 0.355, 0.325, "its Gram matrix", op, strong=True, size=0.034)
        self.picture(sc, heat(self.full / top, hot=VIOLET), 0.555, 0.56, G, G, op)
        self.text(sc, 0.555, 0.325, "the style's", op, strong=True, size=0.034)
        self.text(sc, 0.455, 0.56, "→", op, size=0.045, strong=True)
        self.text(sc, 0.455, 0.76, "16 × 16, first layer", op)
        self.text(sc, 0.455, 0.81, "mismatch %.2f" % self.loss[step], op, ink=dark(VIOLET),
                  size=0.036)
        self.pulse(sc, (0.355, 0.73), (0.125, 0.75), (go * 12) % 1.0, VIOLET,
                   op * (1.0 if go < 1 else 0.0), bend=-0.06, r=5.5)
        self.text(sc, 0.30, 0.905, "each step nudges the pixels to shrink the mismatch", op,
                  ink=dark(VIOLET))
        xs = np.arange(len(self.loss), dtype=float)
        chart = self.chart(((0.73, 0.95), (0.42, 0.76)), xs, np.r_[self.loss, 0.0])
        self.axes(sc, chart, op, "step", "mismatch",
                  xticks=[(v, "%d" % v) for v in (0, 100, 200, 300)])
        self.plot(sc, chart, xs, self.loss, VIOLET, op, step / 300.0, width=2.4)

    def phase_4(self, sc, u, op):
        S = 68.0
        ys = (0.385, 0.565, 0.745)
        self.text(sc, 0.075, ys[0], "content", op, strong=True, size=0.032)
        for row, ink in enumerate(self.STYLES):
            y = ys[row + 1]
            show = smoothstep(0.1 + 0.25 * row, 0.2 + 0.25 * row, u)
            self.picture(sc, paper(self.styles[row]), 0.075, y - 0.02, S - 10, S - 10, op * show,
                         frame=ink)
            self.text(sc, 0.075, y + 0.065, ("thin", "heavy")[row] + " style", op * show,
                      ink=dark(ink, 0.3), size=0.030)
        for j in range(6):
            x = 0.19 + j * 0.083
            self.picture(sc, paper(self.contents[j]), x, ys[0], S, S, op)
            for row, ink in enumerate(self.STYLES):
                start = 0.15 + 0.25 * row + 0.02 * j
                self.picture(sc, paper(self.results[row, j]), x, ys[row + 1], S, S,
                             op * smoothstep(start, start + 0.1, u), frame=ink)
        self.text(sc, 0.36, 0.875, "same digits, the style's strokes", op * smoothstep(
            0.6, 0.7, u), ink=INK, size=0.032)
        self.text(sc, 0.36, 0.925, "the table says how much, not where: some ink lands off "
                  "the strokes", op * smoothstep(0.7, 0.8, u))
        show = smoothstep(0.62, 0.72, u)
        full = self.span(0.11)[0]
        ink = self.ink
        self.text(sc, 0.70, 0.33, "ink in the image", op * show, anchor="lm", strong=True,
                  size=0.032)
        rows = (("content", AXIS, ink["content"]),
                ("made thin", self.STYLES[0], ink["thin_result"]),
                ("thin style", soft(self.STYLES[0], 0.35), ink["thin_style"]),
                ("made heavy", self.STYLES[1], ink["heavy_result"]),
                ("heavy style", soft(self.STYLES[1], 0.35), ink["heavy_style"]))
        top = max(v for _, _, v in rows)
        for row, (name, colour, value) in enumerate(rows):
            y = 0.41 + row * 0.085 + (0.03 if row else 0) + (0.03 if row > 2 else 0)
            self.hbar(sc, 0.80, y, full * value / top, colour, op * show, name, "%.0f" % value,
                      (u - 0.66 - 0.03 * row) / 0.2, thick=20.0, track=full)
        self.text(sc, 0.70, 0.875, "mean of 20 digits; 1 = a black pixel", op * show,
                  anchor="lm")


def fit_two(text):
    """``text`` as one or two lines (a caption beside a row of pictures)."""
    from .mech import fit

    return fit(text, TEXT, 0.16)


ANIMATIONS = {
    "receptive-field": ReceptiveField,
    "vit-attention": VitAttention,
    "clip-contrastive": ClipContrastive,
    "adversarial-examples": AdversarialExamples,
    "object-detection-basics": ObjectDetection,
    "semantic-segmentation": SemanticSegmentation,
    "batchnorm-in-cnns": BatchNormInCnns,
    "style-transfer": StyleTransfer,
}
