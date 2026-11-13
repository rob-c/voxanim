"""Bottlenecks, animated: how narrow a code can be, and what a layer that is
too narrow throws away.  Two animations on the models of :mod:`voxanim.capacity`.

**bottleneck-sweep** -- the dense autoencoder of :mod:`voxanim.autoencoder`
trained six times, alike, with a latent of 1, 2, 3, 8, 32 and 256 numbers:

1. **widths** -- four test digits down the left, and column by column each
   width's reconstruction of them (the test error under each column),
2. **error** -- the mean test reconstruction error against the width, on a
   log axis: it falls steeply to 8 and flattens; from 32 to 256 it barely
   moves,
3. **codes** -- the codes of 600 test digits for L = 1 (a line: where codes
   coincide they stack), 2 (a plane) and 3 (a cube), coloured by label, and
   how well the nearest class centroid classifies each width's codes.

**information-bottleneck** -- a classifier ``784 -> 64 -> 2 -> 10`` (the
middle layer linear) against the same with 64 units in the middle:

1. **networks** -- the two networks' layers on one test digit, untrained,
   and the 2-unit layer's plane with the 600 test digits in one clump,
2. **training** -- the 600 digits' two coordinates move and sort themselves
   by class as the run plays (snapshots at geometrically spaced steps, as
   :func:`voxanim.autoencoder.training_run`), the layers live, both test
   accuracies drawing themselves,
3. **confusions** -- the 2-unit network's confusion matrix on the 10,000
   test digits: the pairs two numbers cannot keep apart.

Both draw their latent spaces with :class:`LatentSpace`: a line, a plane
facing the viewer or a wire cube, with the robust (median, MAD, arcsinh)
axes of :class:`voxanim.ae_anim.AutoencoderAnimation` so that the clusters
spread over the space rather than crowd a corner.
"""

from __future__ import annotations

import numpy as np

from . import capacity as CAP
from . import layout
from .ae_anim import DIGIT_COLORS, corners
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart
from .netanim import (
    ACT_HI,
    ACT_LO,
    INK_HI,
    INK_LO,
    NetAnimation,
    amax,
    div_colors,
    gray_colors,
    lit_colors,
    screen_box,
)
from .sweep import Sweep
from .tensor import Block
from .text import sub

PPU = 9.0
FRAME_INK = layout.NOTE_INK                 # the latent spaces' outlines
POINT = 0.5                                 # voxel size of a code in a plane or cube
BINS = 40                                   # columns the 1-D codes stack into
LEGEND_STEP = 0.041                         # layout y between the legend's swatches


def fit_scale(cam, shape, width, pitch=(1, 1)):
    """Voxel scale at which a :class:`~voxanim.tensor.Block` of ``shape`` (with
    cell ``pitch``) is ``width`` of the frame wide on screen."""
    probe = Block(shape, scale=1.0, pitch=pitch)
    s = cam.project(corners(probe.origin, probe.origin + probe.extent))
    return width / ((s[:, 0].max() - s[:, 0].min()) / cam.width)


def screen_union(cam, blocks):
    """Screen bounds ``(x0, y0, x1, y1)``, in pixels, of several blocks: the
    union of their own screen boxes (the screen box of their world union
    would reach higher, from a corner none of them has)."""
    boxes = np.array([screen_box(cam, b.origin, b.origin + b.extent) for b in blocks])
    return boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()


def caption_over(anim, sc, blocks, text, op, size=0.027, above=True, gap=10):
    """A caption centred over (or under) several blocks, from their screen union."""
    op *= anim._cap
    if op <= 0.01:
        return
    x0, y0, x1, y1 = screen_union(anim.cam, blocks)
    cam = anim.cam
    if above:
        sc.text((x0 + x1) / 2 / cam.width, (y0 - gap) / cam.height, text, size, "regular",
                layout.CAPTION_INK, op, "mb")
    else:
        sc.text((x0 + x1) / 2 / cam.width, (y1 + gap) / cam.height, text, size, "regular",
                layout.CAPTION_INK, op, "mt")


def robust_axes(codes):
    """``(median, MAD, lo, hi)`` of ``codes`` ``(m, L)``: the arcsinh-compressed
    axes of :class:`~voxanim.ae_anim.AutoencoderAnimation`, 1st to 99th percentile."""
    med = np.median(codes, axis=0)
    mad = np.median(np.abs(codes - med), axis=0) + 1e-9
    w = np.arcsinh((codes - med) / (2.0 * mad))
    return np.array([med, mad, np.percentile(w, 1, axis=0), np.percentile(w, 99, axis=0)])


def digit_legend(sc, cam, fx, fy, op, hit=(), step=LEGEND_STEP):
    """A column of the ten digit colours with their digits, from layout ``(fx, fy)``."""
    if op <= 0.01:
        return
    for d in range(10):
        p = cam.frac(fx, fy + step * d)
        on = d in hit
        sc.voxel(p, DIGIT_COLORS[d], op, shadow=False, reflect=False, size=1.1 if on else 0.7,
                 glow=0.3 if on else 0.0)
        q = cam.project(p)
        sc.text(q[0] / cam.width + 0.016, q[1] / cam.height, str(d), 0.029, "regular",
                layout.STRONG_INK if on else layout.CAPTION_INK, op, "lm")


class LatentSpace:
    """A 1-, 2- or 3-D space of codes at layout position ``at``: a line
    ``width`` of the frame long (its codes stacked into a histogram up to
    ``height``), a plane ``width`` by ``height`` facing the viewer, or a cube
    as wide as ``width`` on screen.

    :meth:`axes` sets the robust axes from a cloud of codes (or interpolated
    ones, for a run that is played back); :meth:`world` maps codes inside.
    """

    def __init__(self, cam, dim, at, width, height=None):
        self.cam, self.dim = cam, dim
        self.centre = cam.frac(*at)
        M, k = cam.M, cam.zoom
        h = np.array([M[1, 1], -M[1, 0], 0.0])          # horizontal on screen, in the floor plane
        h /= np.linalg.norm(h)
        px = width * k * cam.width
        self.e1 = h * px / abs(M[0] @ h)
        height = width if height is None else height
        self.e2 = np.array([0.0, 0.0, 1.0]) * height * k * cam.height / abs(M[1, 2])
        side = px / (abs(M[0, 0]) + abs(M[0, 1]))        # a cube this wide on screen
        self.lo, self.hi = self.centre - side / 2, self.centre + side / 2
        self.ax = None

    def axes(self, ax):
        """Use the ``(median, MAD, lo, hi)`` of :func:`robust_axes`."""
        self.ax = np.asarray(ax, np.float64)
        return self

    def unit(self, z):
        """Codes ``(..., dim)`` -> coordinates in ``[0.02, 0.98]^dim`` of the space."""
        med, mad, lo, hi = self.ax
        w = np.arcsinh((np.asarray(z, np.float64) - med) / (2.0 * mad))
        return np.clip((w - lo) / (hi - lo), 0.02, 0.98)

    def unwarp(self, w):
        """Latent values at warped coordinates ``w`` (the inverse of :meth:`unit`'s arcsinh)."""
        med, mad, _lo, _hi = self.ax
        return med + 2.0 * mad * np.sinh(w)

    def world(self, z):
        """World positions of codes ``(..., dim)`` inside the space."""
        u = self.unit(z)
        if self.dim == 3:
            return self.lo + u * (self.hi - self.lo)
        p = self.centre + (u[..., :1] - 0.5) * self.e1
        if self.dim == 2:
            p = p + (u[..., 1:] - 0.5) * self.e2
        return p

    def frame(self, sc, op):
        """The outline: a base line, a rectangle or a wire cube."""
        if op <= 0.01:
            return
        c, a, b = self.centre, self.e1 / 2, self.e2 / 2
        if self.dim == 3:
            sc.wire_box(self.lo, self.hi, color=FRAME_INK, width=1.0, opacity=0.75 * op, bias=0.0)
        elif self.dim == 2:
            pts = [c - a - b, c + a - b, c + a + b, c - a + b]
            for i in range(4):
                sc.line(pts[i], pts[(i + 1) % 4], width=1.0, color=FRAME_INK, opacity=0.75 * op,
                        bias=0.0)
        else:
            sc.line(c - a, c + a, width=1.0, color=FRAME_INK, opacity=0.75 * op, bias=0.0)

    def points(self, sc, codes, labels, op, size=POINT):
        """The codes as voxels coloured by label (1-D: a stacked histogram)."""
        if op <= 0.01:
            return
        if self.dim == 1:
            self.histogram(sc, codes, labels, op)
            return
        for p, lab in zip(self.world(codes), labels):
            sc.voxel(p, DIGIT_COLORS[lab], op, shadow=False, reflect=False, size=size)

    def counts(self, codes, labels):
        """1-D codes binned along the line: ``(BINS, 10)`` counts per bin and label."""
        u = self.unit(np.asarray(codes).reshape(-1, 1))[:, 0]
        b = np.minimum((u * BINS).astype(int), BINS - 1)
        counts = np.zeros((BINS, 10), int)
        np.add.at(counts, (b, np.asarray(labels)), 1)
        return counts

    def histogram(self, sc, codes, labels, op):
        """The 1-D codes as a stacked histogram over the line: one column per
        bin, a segment per label, the tallest column ``height`` tall."""
        counts = self.counts(codes, labels)
        unit = self.e2 / counts.sum(1).max()               # world vector per digit
        px = abs(self.cam.M[0] @ self.e1) / BINS           # a bin's width on screen
        for k in range(BINS):
            base = self.centre + ((k + 0.5) / BINS - 0.5) * self.e1
            acc = 0
            for d in range(10):
                n = counts[k, d]
                if n == 0:
                    continue
                sc.line(base + unit * acc, base + unit * (acc + n), width=max(1.0, px - 1.0),
                        color=DIGIT_COLORS[d], opacity=op, bias=0.0)
                acc += n

    def bbox(self):
        """World box of the space (for captions)."""
        if self.dim == 3:
            return self.lo, self.hi
        c, a = self.centre, np.abs(self.e1) / 2
        if self.dim == 2:
            b = np.abs(self.e2) / 2
            return c - a - b, c + a + b
        return c - a, c + a + self.e2

    def axis_names(self, sc, names, op, size=0.029):
        """Axis names just outside the space, on the edges nearest the viewer."""
        if op <= 0.01:
            return
        cam, c, a, b = self.cam, self.centre, self.e1 / 2, self.e2 / 2
        if self.dim == 3:
            lo, hi, mid = self.lo, self.hi, (self.lo + self.hi) / 2
            marks = [(np.array([mid[0], hi[1], lo[2]]), names[0], (0.0, 0.03), "mt"),
                     (np.array([lo[0], mid[1], lo[2]]), names[1], (-0.012, 0.012), "rt"),
                     (np.array([lo[0], lo[1], mid[2]]), names[2], (-0.012, 0.0), "rm")]
        elif self.dim == 2:
            marks = [(c - b, names[0], (0.0, 0.012), "mt"), (c - a, names[1], (-0.012, 0.0), "rm")]
        else:
            marks = [(c + a, names[0], (0.012, 0.0), "lm")]
        for p, name, (dx, dy), anchor in marks:
            q = cam.project(p)
            sc.text(q[0] / cam.width + dx, q[1] / cam.height + dy, name, size, "regular",
                    layout.CAPTION_INK, op, anchor)

    def scale_marks(self, sc, op, size=0.026):
        """2-D only: the latent values at the ends of both axes (the box follows
        the cloud; these say how far it has grown)."""
        if self.dim != 2 or op <= 0.01:
            return
        z_lo, z_hi = self.unwarp(self.ax[2]), self.unwarp(self.ax[3])
        c, a, b, cam = self.centre, self.e1 / 2, self.e2 / 2, self.cam
        marks = [(c - a - b, "%.1f" % z_lo[0], (0.0, 0.012), "lt"),
                 (c + a - b, "%.1f" % z_hi[0], (0.0, 0.012), "rt"),
                 (c - a - b, "%.1f" % z_lo[1], (-0.012, 0.0), "rb"),
                 (c - a + b, "%.1f" % z_hi[1], (-0.012, 0.0), "rt")]
        for p, s, (dx, dy), anchor in marks:
            q = cam.project(p)
            sc.text(q[0] / cam.width + dx, q[1] / cam.height + dy, s, size, "mono",
                    layout.NOTE_INK, op, anchor)


# ================================================================== bottleneck-sweep

GRID_X = np.linspace(0.075, 0.925, 7)       # layout x of the seven columns: input, six widths
GRID_Y = 0.53                               # layout y of the grid's centre
SHEET_W = 0.072                             # on-screen width of a 28x28 sheet, frame fraction
ROW_GAP = 3                                 # voxels between the grid's rows
CURVE = ((0.22, 0.78), (0.30, 0.72))        # the error chart ((x0, x1), (top, bottom)), layout
CURVE_INK = (0.75, 0.22, 0.22)
SPACES = ((1, (0.185, 0.68), 0.27, 0.30), (2, (0.505, 0.51), 0.25, 0.40),
          (3, (0.775, 0.50), 0.22, None))   # (dim, centre, width, height) of the three spaces
LEGEND_AT = (0.935, 0.27)
SPACE_EVERY = 3.4                           # seconds between the three spaces appearing
FADE = 0.7                                  # seconds a phase's stage takes to fade out


class BottleneckSweep(NetAnimation):
    """Six latent widths of one autoencoder side by side (see the module docstring).

    ``sw`` is a :class:`voxanim.capacity.Sweep`.
    """

    PHASES = ("widths", "error", "codes")

    def __init__(self, sw):
        super().__init__("How narrow can the code be?", "", PPU)
        self.sw = sw
        self.hold = 2.5
        cam = self.cam
        n_rows = len(sw.images)
        s = fit_scale(cam, (28, 28, 1), SHEET_W)
        self.grid = []                                   # [column][row] blocks
        for c, fx in enumerate(GRID_X):
            col = []
            for r in range(n_rows):
                b = Block((28, 28, 1), scale=s, name="g%d%d" % (c, r))
                centre = cam.frac(fx, GRID_Y)
                centre[2] += ((n_rows - 1) / 2 - r) * (28 + ROW_GAP) * s
                b.place(centre)
                col.append(b)
            self.grid.append(col)
        self.blocks_extra = [b for col in self.grid for b in col]
        self.colors = [[lit_colors(x, 1.0, INK_LO, INK_HI) for x in sw.images]] + \
            [[gray_colors(r) for r in rec] for rec in sw.recon]
        top = float(sw.losses.max())
        self.y_top = np.ceil(top / 0.02) * 0.02
        self.chart = LossChart(CURVE, (1, sw.widths[-1]), (0.0, self.y_top), xlog=True)
        self.spaces = []
        for dim, at, w, h in SPACES:
            sp = LatentSpace(cam, dim, at, w, h).axes(robust_axes(sw.clouds[dim]))
            self.spaces.append(sp)

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0
        Tm["input"] = (0.3, 1.5)
        self.sweep = Sweep(1.8, len(sw.widths), Tempo(slow=2.6, fast=1.5, ramp_in=2.0,
                                                      ramp_out=1.5))
        t = self.sweep.t1 + 2.0
        Tm["error"] = (t, t + FADE + 1.2 * len(sw.widths) + 2.5)
        t = Tm["error"][1]
        Tm["codes"] = (t, t + FADE + SPACE_EVERY * (len(SPACES) - 1) + 4.5)
        self.Tm = Tm
        self.finish(Tm["codes"][1])
        self.marks = [0.0, Tm["error"][0], Tm["codes"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        err = self.progress("error", t)
        codes = self.progress("codes", t)
        grid_op = back * (1.0 - smoothstep(0.0, FADE, t - self.Tm["error"][0]))
        chart_op = back * smoothstep(0.0, FADE, t - self.Tm["error"][0]) * \
            (1.0 - smoothstep(0.0, FADE, t - self.Tm["codes"][0]))
        self._grid(sc, t, grid_op)
        self._curve(sc, t, chart_op)
        self._spaces(sc, t, back * smoothstep(0.0, FADE, t - self.Tm["codes"][0]))
        if reset > 0 or t < self.sweep.t0:
            subtitle = ("the dense autoencoder 784 → 256 → 64 → L → 64 → 256 → 784, trained alike "
                        "(%d epochs) with L = %s numbers in the code"
                        % (self.sw.epochs, ", ".join(str(w) for w in self.sw.widths)))
        elif err <= 0:
            subtitle = self._sweep_subtitle(t)
        elif codes <= 0:
            subtitle = self._error_subtitle(t)
        else:
            subtitle = self._codes_subtitle(t)
        self.subtitle = subtitle
        self.step_indicator(sc, ph, self.PHASES)

    # -- the grid of reconstructions ----------------------------------------------------
    def _column_progress(self, c, t):
        """How far column ``c`` (0: the inputs) has filled in, 0..1."""
        if c == 0:
            return self.progress("input", t)
        done, n, u = self.sweep.at(t)
        k = c - 1
        if k < done:
            return 1.0
        return remap(u, 0.0, 0.8) if k == n else 0.0

    def _grid(self, sc, t, op):
        """The input digits and each width's reconstructions, filling row by row."""
        rows = np.arange(28)[:, None, None] / 28
        for c, col in enumerate(self.grid):
            p = self._column_progress(c, t)
            for r, b in enumerate(col):
                b.base[:, :, 0] = self.colors[c][r].reshape(28, 28, 3)
                b.present[...] = (p * 1.12 - 0.1 > rows) & np.ones((1, 28, 1), bool) if p > 0 \
                    else False
                b.opacity *= op
            if p > 0 and op > 0.01:
                lo, hi = col[-1].origin, col[-1].origin + col[-1].extent
                text = "input" if c == 0 else "L = %d\nerror %.3f" % (self.sw.widths[c - 1],
                                                                     self.sw.losses[c - 1])
                self.caption(sc, (lo, hi), text, op * smoothstep(0.0, 0.3, p), size=0.028)
        _done, n, u = self.sweep.at(t)
        if 0 <= n < len(self.sw.widths) and op > 0.01:
            col = self.grid[n + 1]
            lo = np.minimum.reduce([b.origin for b in col])
            hi = np.maximum.reduce([b.origin + b.extent for b in col])
            sc.wire_box(lo, hi, inflate=0.6 * col[0].scale, color=(0.96, 0.96, 0.96), width=1.2,
                        opacity=0.9 * op * (1.0 - smoothstep(0.8, 1.0, u)), bias=0.1)

    def _sweep_subtitle(self, t):
        done, n, _u = self.sweep.at(t)
        k = min(n, len(self.sw.widths) - 1) if n >= 0 else 0
        L = self.sw.widths[k]
        if done == len(self.sw.widths):
            return ("every width rebuilds the digits; the error (mean (x − x̂)² over 10,000 test "
                    "digits) falls from %.3f at L = 1 to %.3f at L = 256"
                    % (self.sw.losses[0], self.sw.losses[-1]))
        return ("L = %d: each digit squeezed to %s and rebuilt from %s; test error %.3f"
                % (L, "one number" if L == 1 else "%d numbers" % L,
                   "it" if L == 1 else "them", self.sw.losses[k]))

    # -- the error curve -----------------------------------------------------------------------
    def _curve(self, sc, t, op):
        """Test error against width, log axis, one point at a time."""
        op *= self._cap
        if op <= 0.01:
            return
        sw, cam, place = self.sw, self.cam, self.cam.frac
        yt = [(v, "%.2f" % v) for v in np.arange(0, self.y_top + 1e-9, 0.02)]
        self.chart.axes(sc, cam, place, op, "latent width L (log scale)", "test error",
                        [(w, str(w)) for w in sw.widths], yt)
        reveal = (t - self.Tm["error"][0] - FADE) / 1.2             # points shown so far
        n = len(sw.widths)
        pts = [place(self.chart.fx(w), self.chart.fy(e)) for w, e in zip(sw.widths, sw.losses)]
        for i in range(n):
            u = remap(reveal, i, i + 1)
            if u <= 0:
                break
            if i > 0:
                q = pts[i - 1] + (pts[i] - pts[i - 1]) * ease_in_out(u)
                sc.line(pts[i - 1], q, width=1.4, color=CURVE_INK, opacity=0.9 * op, bias=0.0)
            if u >= 1 or i == 0:
                sc.voxel(pts[i], CURVE_INK, op, glow=0.3, shadow=False, reflect=False, size=0.7)
                q = cam.project(pts[i])
                sc.text(q[0] / cam.width + 0.012, q[1] / cam.height - 0.012, "%.3f" % sw.losses[i],
                        0.028, "regular", layout.STRONG_INK,
                        op * smoothstep(0.0, 0.5, remap(reveal, i, i + 0.5)), "lm")
        sc.probe(("curve",), (reveal * 20, 0, 0), tol=8.0)

    def _error_subtitle(self, t):
        sw = self.sw
        reveal = (t - self.Tm["error"][0] - FADE) / 1.2
        if reveal < len(sw.widths):
            return ("the test error against the width of the code: %s"
                    % ", ".join("%.3f at L = %d" % (e, w) for w, e in
                                zip(sw.widths[:3], sw.losses[:3])))
        i32 = list(sw.widths).index(32)
        gain = 100 * (1 - sw.losses[-1] / sw.losses[i32])
        return ("steep to L = 8, then flat: 32 → 256 numbers cuts the error by only %d%%; "
                "a digit has far fewer than 784 degrees of freedom" % round(gain))

    # -- the three latent spaces ------------------------------------------------------------------
    def _space_op(self, j, t):
        """Opacity of space ``j`` at time ``t`` (they appear one after another)."""
        t0 = self.Tm["codes"][0] + FADE + SPACE_EVERY * j
        return smoothstep(t0, t0 + 0.8, t)

    def _spaces(self, sc, t, op):
        """The line, the plane and the cube of codes, with the legend and captions."""
        if op <= 0.01:
            return
        sw = self.sw
        shown = set()
        for j, sp in enumerate(self.spaces):
            o = op * self._space_op(j, t)
            if o <= 0.01:
                continue
            dim = sp.dim
            sp.frame(sc, o)
            sp.points(sc, sw.clouds[dim], sw.cloud_labels, o)
            sp.axis_names(sc, ["z" + sub(i + 1) for i in range(dim)], o * self._cap)
            k = list(sw.widths).index(dim)
            self.caption(sc, sp.bbox(), "L = %d: %s\nnearest centroid %d%%"
                         % (dim, ("a line", "a plane", "a cube")[dim - 1],
                            round(100 * sw.centroid_acc[k])), o,
                         offset=0.05 if dim == 3 else 0.01)      # the cube's z₁ sits below it
            shown.add(dim)
        digit_legend(sc, self.cam, *LEGEND_AT, op * self._cap)

    def _codes_subtitle(self, t):
        sw = self.sw
        j = min(len(self.spaces) - 1, int((t - self.Tm["codes"][0] - FADE) // SPACE_EVERY))
        acc = [round(100 * sw.centroid_acc[list(sw.widths).index(d)]) for d in (1, 2, 3)]
        full = round(100 * sw.centroid_acc[-1])
        return [
            "L = 1: the %d test codes on one axis (stacked where they coincide), coloured by "
            "label; the classes pile on top of each other" % len(sw.cloud_labels),
            "L = 2: on a plane the digits begin to find regions of their own; L = 3: a cube, "
            "more room still",
            "what the code keeps about the digit: the nearest class centroid gets %d%%, %d%% and "
            "%d%% of the codes right for L = 1, 2, 3 (%d%% at L = 256)" % (*acc, full),
        ][j]


# ================================================================== information-bottleneck

PLANE = ((0.265, 0.52), 0.36, 0.46)         # the 2-unit layer's plane: centre, width, height
IB_LEGEND = (0.485, 0.30)
ACC_CHART = ((0.615, 0.95), (0.25, 0.42))   # accuracy chart ((x0, x1), (top, bottom)), layout
LEGEND_DROP = 0.125                         # the legend line sits this far under the chart
INKS = ((0.75, 0.22, 0.22), (0.20, 0.52, 0.50))             # the squeezed and the roomy network
STRIP_X = (0.665, 0.775, 0.885)             # layout x of a network's three sheets
STRIP_Y = 0.785                             # layout y of the two rows' centre
STRIP_W = 0.055                             # on-screen width of an 8x8 sheet
ROW_GAP_Y = 0.21                            # layout y between the two rows (a caption fits between)
CONF_AT = (0.79, 0.75)                      # the confusion matrix's centre
CONF_W = 0.175                              # its on-screen width
CONF_PITCH = (1.6, 1.6)                     # cells spaced so each row and column takes a label
SPAN = 25.0                                 # seconds the training phase takes
NAMES = ("squeezed", "roomy")


class InformationBottleneck(NetAnimation):
    """A 2-unit middle layer against a 64-unit one (see the module docstring).

    ``runs`` are the :class:`voxanim.capacity.SqueezeRun` s for hidden 2 and 64.
    """

    PHASES = ("networks", "training", "confusions")

    def __init__(self, runs):
        super().__init__("A layer that must forget", "", PPU)
        self.runs = runs
        self.hold = 2.5
        cam = self.cam
        narrow = runs[0]
        self.S = len(narrow.steps)
        at, w, h = PLANE
        self.plane = LatentSpace(cam, 2, at, w, h)
        self.ax = np.array([robust_axes(c) for c in narrow.cloud])          # (S, 4, 2)
        self.chart = LossChart(ACC_CHART, (1, int(narrow.steps[-1])), (0.0, 100.0), xlog=True)
        # the two networks' layers: 64 (8x8), the middle layer, 10 (1x10)
        s = fit_scale(cam, (8, 8, 1), STRIP_W)
        self.strips = []
        for i, run in enumerate(runs):
            fy = STRIP_Y + (i - 0.5) * ROW_GAP_Y             # the squeezed network on top
            h = run.hidden
            shapes = [(8, 8, 1), (1, h, 1) if h <= 8 else (8, 8, 1), (1, 10, 1)]
            row = []
            for fx, shape in zip(STRIP_X, shapes):
                scale = s * (2.0 if shape[1] == 2 else 1.0)
                row.append(Block(shape, scale=scale, name="n%d" % i).place(cam.frac(fx, fy)))
            self.strips.append(row)
        self.conf = Block((10, 10, 1), scale=fit_scale(cam, (10, 10, 1), CONF_W, CONF_PITCH),
                          pitch=CONF_PITCH, name="conf")
        self.conf.place(cam.frac(*CONF_AT))
        self.blocks_extra = [b for row in self.strips for b in row] + [self.conf]
        self.conf_colors = self._confusion_colors(narrow.confusion)

        # ---- timeline -----------------------------------------------------------------
        Tm = {"intro": (0.0, 4.0)}
        Tm["train"] = (4.3, 4.3 + SPAN)
        t = Tm["train"][1] + 0.7
        Tm["confusions"] = (t, t + 8.5)
        self.Tm = Tm
        self.finish(Tm["confusions"][1])
        self.marks = [0.0, Tm["train"][0], Tm["confusions"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def snapshot(self, pos):
        """The runs' state at fractional snapshot ``pos`` (linear between
        snapshots): ``cloud`` of the squeezed network, each network's ``acc``
        and ``example`` layers, the ``step``."""
        i = min(int(pos), self.S - 2)
        f = pos - i

        def lerp(a):
            return a[i] + (a[i + 1] - a[i]) * f

        self.plane.axes(lerp(self.ax))
        return dict(cloud=lerp(self.runs[0].cloud), step=float(lerp(self.runs[0].steps)),
                    acc=[float(lerp(r.test_acc)) for r in self.runs],
                    example=[[lerp(e) for e in r.example] for r in self.runs], i=i)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        u = self.progress("train", t)
        conf = smoothstep(0.0, 0.8, t - self.Tm["confusions"][0])
        pos = (self.S - 1) * smoothstep(0.0, 1.0, u)
        snap = self.snapshot(pos)
        intro = smoothstep(0.0, 0.8, t) * back
        self._plane(sc, snap, intro)
        self._strips(sc, snap, intro * (1.0 - conf))
        self._chart(sc, snap, u, intro * smoothstep(0.6, 1.4, t))
        self._confusion(sc, back * conf)
        if reset > 0 or u <= 0:
            subtitle = ("two classifiers of one digit: 784 → 64 → 2 → 10 and 784 → 64 → 64 → 10; "
                        "the 2-unit layer must say it all in two numbers")
        elif conf <= 0:
            subtitle = self._training_subtitle(u, snap)
        else:
            subtitle = self._confusion_subtitle()
        self.subtitle = subtitle
        self.step_indicator(sc, ph, self.PHASES)

    def _training_subtitle(self, u, snap):
        run = self.runs[0]
        step = int(round(snap["step"]))
        a2, a64 = (round(100 * a) for a in snap["acc"])
        if u >= 1:
            return ("after %d epochs: %d%% test accuracy through two numbers, %d%% through 64; "
                    "the plane shows what two numbers can and cannot separate"
                    % (run.steps[-1] // run.steps_per_epoch, a2, a64))
        if u < 0.4:
            return ("training, step %d of %d: the two coordinates of 600 test digits spread out; "
                    "accuracy %d%% (2 units) and %d%% (64)" % (step, run.steps[-1], a2, a64))
        return ("training, step %d (epoch %d): the classes sort into wedges round the origin; "
                "accuracy %d%% against %d%%"
                % (step, int(snap["step"] // run.steps_per_epoch) + 1, a2, a64))

    # -- the plane ----------------------------------------------------------------------------
    def _plane(self, sc, snap, op):
        """The squeezed network's 2-unit layer: 600 test digits as points."""
        if op <= 0.01:
            return
        run = self.runs[0]
        self.plane.frame(sc, op)
        self.plane.points(sc, snap["cloud"], run.cloud_labels, op, size=0.55)
        self.plane.axis_names(sc, ["h" + sub(1), "h" + sub(2)], op * self._cap)
        self.plane.scale_marks(sc, op * self._cap)
        self.caption(sc, self.plane.bbox(), "%d test digits in the 2-unit layer"
                     % len(run.cloud_labels), op, offset=0.034, max_width=0.4)
        digit_legend(sc, self.cam, *IB_LEGEND, op * self._cap)
        for p in self.plane.world(snap["cloud"][:4]):
            sc.probe(("cloud", tuple(np.round(p, 1))), p, tol=4.0)

    # -- the two networks on one digit ------------------------------------------------------------
    def _strips(self, sc, snap, op):
        """Each network's layers on test digit EXAMPLE, live: 64 ReLU units, the
        middle layer (signed) and the ten probabilities."""
        for i, (run, row) in enumerate(zip(self.runs, self.strips)):
            h1, h2, p = snap["example"][i]
            cols = [lit_colors(h1, amax(h1), ACT_LO, ACT_HI), div_colors(h2, amax(h2)),
                    lit_colors(p, 1.0, ACT_LO, ACT_HI)]
            for b, c in zip(row, cols):
                b.base[:, :, 0] = c.reshape(b.H, b.W, 3)
                b.opacity *= op
            row[2].glow[0, int(np.argmax(p)), 0] = 0.3
            if op > 0.01:
                lab = int(run.cloud_labels[CAP.EXAMPLE])
                caption_over(self, sc, row, "%s: 784 → 64 → %d → 10, p(%d) = %.2f"
                             % (NAMES[i], run.hidden, lab, p[lab]), op)

    # -- the accuracy curves ------------------------------------------------------------------
    def _chart(self, sc, snap, u, op):
        """Both networks' test accuracy against the step (log axis), drawn as the run plays."""
        op *= self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        total = int(self.runs[0].steps[-1])
        ticks = [(s, str(s)) for s in (1, 10, 100, 1000) if s <= total]
        self.chart.axes(sc, cam, place, op, "training steps (log scale)", "test accuracy",
                        ticks, [(v, "%d%%" % v) for v in (0, 50, 100)])
        upto = max(1.0, snap["step"])
        for run, ink in zip(self.runs, INKS):
            self.chart.curve(sc, place, np.maximum(run.steps, 1), 100 * run.test_acc, upto, ink,
                             op)
        # the legend is one line under the axis label, clear of the curves
        (x0, _x1), (_y0, y1) = ACC_CHART
        for run, a, ink, fx in zip(self.runs, snap["acc"], INKS, (x0, x0 + 0.16)):
            self.chart.legend(sc, cam, [("%d units: %d%%" % (run.hidden, round(100 * a)), ink)],
                              op * smoothstep(0.05, 0.2, u), (fx, y1 + LEGEND_DROP))

    # -- the confusion matrix ------------------------------------------------------------------
    @staticmethod
    def _confusion_colors(conf):
        """Off-diagonal counts lit on their own maximum; the diagonal (right
        answers, in the hundreds) a plain grey so the mistakes stand out."""
        off = conf * (1 - np.eye(10, dtype=int))
        cols = lit_colors(off.ravel(), amax(off), ACT_LO, ACT_HI)
        cols[np.eye(10, dtype=bool).ravel()] = (0.52, 0.54, 0.56)
        return cols.reshape(10, 10, 3)

    def _confusion(self, sc, op):
        """The squeezed network's confusion matrix with its digit labels."""
        b = self.conf
        b.base[:, :, 0] = self.conf_colors
        b.opacity *= op
        b.present[...] = op > 0.01
        if op <= 0.01:
            return
        h = 0.5 * b.scale
        for d in range(10):
            p = b.cell_center(9, d, 0)
            self.caption(sc, (p - h, p + h), str(d), op, size=0.026)
            q = b.cell_center(d, 0, 0)
            x0 = self.cam.project(q - h)[0]
            y = self.cam.project(q)[1]
            sc.text(x0 / self.cam.width - 0.012, y / self.cam.height, str(d), 0.026, "regular",
                    layout.CAPTION_INK, op * self._cap, "rm")

    def _confusion_subtitle(self):
        narrow, wide = self.runs
        pairs = ", ".join("%d ↔ %d (%d)" % (a, b, n) for a, b, n in narrow.confusions(3))
        return ("what two numbers throw away (rows: the digit, columns: the squeezed network's "
                "answer, %d of 10,000 wrong): %s; the roomy network gets %d wrong"
                % (narrow.errors, pairs, wide.errors))
