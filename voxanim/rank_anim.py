"""The rank of a layer: the trained MNIST MLP's weight matrices under a
truncated singular value decomposition (:func:`voxanim.capacity.rank_study`).

    W₁ (784 × 64) and W₂ (64 × 32) of 784 -> 64 -> 32 -> 10 (ReLU, 96.8% test accuracy)

1. **matrices** -- W₁ as weight images (16 of its 64 columns, each a 28×28
   picture of what that unit looks for) and W₂ as a 64 × 32 sheet,
2. **singular values** -- W = U Σ Vᵀ: the singular values of each matrix as
   bars, largest first, and how much of ‖W‖² the top few hold,
3. **truncation** -- both matrices rebuilt from their top 1, 2, 4, 8, 16 and
   32 components, then whole again: the weight images re-form, the test
   accuracy climbs back (7.6% at rank 1, 62% at 8, 96.3% at 32 against 96.8%
   in full) while the fraction of the parameters kept rises (a rank-k matrix
   is two thin factors, k(m + n) numbers for an m × n matrix),
4. **LoRA** -- the fact low-rank adaptation relies on: what a matrix carries
   lives in a few directions, so a rank-8 update to a frozen matrix can carry
   most of what fine-tuning changes (``gpt2-lora``).

The SVD is numpy's; a test checks ``U Σ Vᵀ`` against each matrix to machine
precision.  The bars are drawn to their exact height on a linear scale.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .bottleneck_anim import caption_over, fit_scale
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart
from .netanim import NetAnimation, amax, div_colors
from .sweep import Sweep
from .tensor import Block
from .text import sub

PPU = 9.0
SHOWN = 16                                  # columns of W₁ drawn as weight images (of 64)
IMAGE_X = (0.085, 0.19, 0.295, 0.40)        # layout x of the weight images' four columns
IMAGE_Y = 0.555                             # layout y of the image grid's centre
IMAGE_W = 0.08                              # on-screen width of a weight image
IMAGE_GAP = 4                               # voxels between rows of images
SHEET_AT = (0.555, 0.56)                    # W₂'s sheet (clear of the images' captions)
SHEET_W = 0.11
BARS = (((0.67, 0.95), (0.26, 0.37)), ((0.67, 0.95), (0.47, 0.56)))  # the two bar charts' boxes
BAR_PX = (2.6, 5.0)                         # bar width, px, for 64 and 32 bars
BAR_INK = (0.20, 0.52, 0.50)
DROPPED = 0.22                              # opacity of a component a truncation drops
CHART = ((0.69, 0.91), (0.665, 0.84))       # accuracy against rank ((x0, x1), (top, bottom))
CHART_INK = (0.75, 0.22, 0.22)
KEPT_INK = (0.36, 0.38, 0.42)
NAMES = ("W" + sub(1), "W" + sub(2))


class RankOfALayer(NetAnimation):
    """Two weight matrices, their singular values and their truncations (see the module).

    ``st`` is a :class:`voxanim.capacity.RankStudy`.
    """

    PHASES = ("matrices", "singular values", "truncation", "LoRA")

    def __init__(self, st):
        super().__init__("The rank of a layer", "", PPU)
        self.st = st
        self.hold = 2.5
        cam = self.cam
        # the weight images: columns 0 .. SHOWN - 1 of W₁ as 28x28 pictures
        s = fit_scale(cam, (28, 28, 1), IMAGE_W)
        self.images = []
        side = int(np.sqrt(SHOWN))
        for n in range(SHOWN):
            r, c = divmod(n, side)
            centre = cam.frac(IMAGE_X[c], IMAGE_Y)
            centre[2] += ((side - 1) / 2 - r) * (28 + IMAGE_GAP) * s
            self.images.append(Block((28, 28, 1), scale=s, name="w1_%d" % n).place(centre))
        h, w = st.Ws[1].shape
        self.sheet = Block((h, w, 1), scale=fit_scale(cam, (h, w, 1), SHEET_W), name="w2")
        self.sheet.place(cam.frac(*SHEET_AT))
        # the singular-value bars: a chart per matrix, index along x, value up
        self.bars = [LossChart(box, (0, len(st.svd[m][1]) + 1), (0.0, float(st.svd[m][1][0])))
                     for m, box in enumerate(BARS)]
        self.blocks_extra = self.images + [self.sheet]
        # the truncated matrices at every rank (the full one last)
        self.trunc = [[st.truncated(m, k) for k in st.ranks] for m in range(2)]
        self.vmax = [amax(st.Ws[0][:, :SHOWN]), amax(st.Ws[1])]
        self.chart = LossChart(CHART, (1, st.ranks[-1]), (0.0, 100.0), xlog=True)

        # ---- timeline -----------------------------------------------------------------
        Tm = {"matrices": (0.0, 4.0)}
        self.svd = Sweep(4.3, len(st.svd[0][1]), Tempo(slow=0.45, fast=0.055, ramp_in=3.0,
                                                       ramp_out=2.0))
        Tm["svd"] = (self.svd.t0, self.svd.t1 + 2.5)
        self.trunc_sweep = Sweep(Tm["svd"][1], len(st.ranks), Tempo(slow=3.0, fast=2.2,
                                                                     ramp_in=2.0, ramp_out=1.0))
        Tm["truncation"] = (self.trunc_sweep.t0, self.trunc_sweep.t1 + 1.0)
        Tm["lora"] = (Tm["truncation"][1], Tm["truncation"][1] + 6.0)
        self.Tm = Tm
        self.finish(Tm["lora"][1])
        self.marks = [0.0, Tm["svd"][0], Tm["truncation"][0], Tm["lora"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def rank_at(self, t):
        """``(index into ranks, progress of that step)`` of the truncation at ``t``
        (the full matrices before the sweep and after it)."""
        done, n, u = self.trunc_sweep.at(t)
        if n < 0 or done == len(self.st.ranks):
            return len(self.st.ranks) - 1, 1.0
        return n, u

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 0.8, t) * back
        k_idx, u = self.rank_at(t)
        self._matrices(sc, k_idx, u, intro)
        self._bars(sc, t, k_idx, intro * smoothstep(0.0, 0.5, t - self.Tm["svd"][0]))
        self._chart(sc, t, back * smoothstep(0.0, 0.6, t - self.Tm["truncation"][0]))
        self.subtitle = self._subtitle(t, ph, k_idx)
        self.step_indicator(sc, ph, self.PHASES)

    def _subtitle(self, t, ph, k_idx):
        st = self.st
        if ph == 0:
            return ("the trained MLP 784 → 64 → 32 → 10: W%s is 784 × 64 (each column a 28×28 "
                    "image of what one unit looks for), W%s is 64 × 32" % (sub(1), sub(2)))
        if ph == 1:
            done = self.svd.at(t)[0]
            if done < len(st.svd[0][1]):
                return ("the SVD W = U Σ V' splits W into rank-1 pieces weighted by σ%s ≥ σ%s ≥ …; "
                        "the larger σ, the more of W that piece carries" % (sub(1), sub(2)))
            return ("the top 8 of W%s's 64 singular values hold %d%% of ‖W%s‖², the top 8 of "
                    "W%s's 32 hold %d%%: neither matrix is nearly rank 8"
                    % (sub(1), round(100 * st.energy(0, 8)), sub(1), sub(2),
                       round(100 * st.energy(1, 8))))
        if ph == 2:
            k = st.ranks[k_idx]
            if k_idx == len(st.ranks) - 1:
                return ("whole again: %.1f%%; rank 32 kept %.1f%% with %d%% of the parameters, "
                        "rank 4 only %.0f%% with %d%%"
                        % (100 * st.full_acc, 100 * st.acc[-2], round(100 * st.kept[-2]),
                           100 * st.acc[2], round(100 * st.kept[2])))
            return ("rank %d: both matrices rebuilt from their top %d component%s (%d%% of the "
                    "parameters); test accuracy %.1f%%"
                    % (k, k, "" if k == 1 else "s", round(100 * st.kept[k_idx]),
                       100 * st.acc[k_idx]))
        return ("LoRA relies on this: a fine-tuning update ΔW of rank 4 to 16, two thin factors "
                "added to a frozen W, carries most of what training changes (see gpt2-lora)")

    # -- the matrices -----------------------------------------------------------------------
    def _matrices(self, sc, k_idx, u, op):
        """The weight images and the sheet at the current truncation, with captions."""
        st = self.st
        W1, W2 = self.trunc[0][k_idx], self.trunc[1][k_idx]
        for n, b in enumerate(self.images):
            b.base[:, :, 0] = div_colors(W1[:, n], self.vmax[0]).reshape(28, 28, 3)
            b.opacity *= op
        self.sheet.base[:, :, 0] = div_colors(W2.ravel(), self.vmax[1]).reshape(self.sheet.H,
                                                                               self.sheet.W, 3)
        self.sheet.opacity *= op
        # a flash of white as a new truncation lands
        flash = 0.5 * (1.0 - smoothstep(0.0, 0.25, u)) if 0 < u < 1 else 0.0
        for b in self.images + [self.sheet]:
            b.white[...] = flash
        if op <= 0.01:
            return
        k = st.ranks[k_idx]
        full = k_idx == len(st.ranks) - 1
        what = "" if full else " at rank %d" % k
        caption_over(self, sc, self.images, "%s%s: %d of its 64 columns as 28×28 images"
                     % (NAMES[0], what, SHOWN), op, size=0.029, above=False)
        self.caption(sc, (self.sheet.origin, self.sheet.origin + self.sheet.extent),
                     "%s: 64 × 32" % NAMES[1] if full else "%s, rank %d" % (NAMES[1], k), op)

    # -- the singular values ----------------------------------------------------------------------
    def _bars(self, sc, t, k_idx, op):
        """Both matrices' singular values as bars, appearing in index order (the
        bar under way grows); the components a truncation drops are dimmed."""
        op *= self._cap
        if op <= 0.01:
            return
        done, n, u = self.svd.at(t)
        k = self.st.ranks[k_idx]
        truncating = t >= self.Tm["truncation"][0]
        cam, place = self.cam, self.cam.frac
        for m, chart in enumerate(self.bars):
            s = self.st.svd[m][1]
            cols = len(s)
            label = "%s: σ%s = %.2f … σ%s = %.2f" % (NAMES[m], sub(1), s[0], sub(cols), s[-1])
            if truncating:
                label = "%s: top %d of %d components kept" % (NAMES[m], min(k, cols), cols)
            chart.axes(sc, cam, place, op, "", label, [(1, "1"), (cols, str(cols))],
                       [(0, "0"), (float(s[0]), "%.1f" % s[0])], size=0.027)
            for j in range(cols):
                grow = 1.0 if j < done or t >= self.Tm["svd"][1] else (
                    remap(u, 0.0, 0.6) if j == n else 0.0)
                if grow <= 0:
                    continue
                a = place(chart.fx(j + 1), chart.fy(0.0))
                b = place(chart.fx(j + 1), chart.fy(float(s[j]) * grow))
                dim = DROPPED if truncating and j >= k else 1.0
                sc.line(a, b, width=BAR_PX[m], color=BAR_INK, opacity=0.9 * op * dim, bias=0.0)
            if truncating and k < cols:
                chart.mark(sc, place, k + 0.5, KEPT_INK, op, dashes=8)
        if 0 <= n < len(self.st.svd[0][1]) and t < self.Tm["svd"][1]:
            sc.probe(("svd",), (n * 10 + u * 10, 0, 0), tol=8.0)

    # -- accuracy against rank ---------------------------------------------------------------------
    def _chart(self, sc, t, op):
        """Test accuracy after each truncation, point by point, with the parameters kept."""
        op *= self._cap
        if op <= 0.01:
            return
        st, cam, place = self.st, self.cam, self.cam.frac
        self.chart.axes(sc, cam, place, op, "rank kept (log scale)", "test accuracy",
                        [(k, str(k)) for k in st.ranks], [(v, "%d%%" % v) for v in (0, 50, 100)])
        done, n, u = self.trunc_sweep.at(t)
        shown = done + (1 if 0 <= n < len(st.ranks) and u > 0.3 else 0)
        pts = [place(self.chart.fx(k), self.chart.fy(100 * a)) for k, a in zip(st.ranks, st.acc)]
        for i in range(min(shown, len(pts))):
            if i > 0:
                sc.line(pts[i - 1], pts[i], width=1.4, color=CHART_INK, opacity=0.9 * op, bias=0.0)
            sc.voxel(pts[i], CHART_INK, op, glow=0.3, shadow=False, reflect=False, size=0.7)
        if shown > 0:
            i = min(shown, len(pts)) - 1
            q = cam.project(pts[i])
            sc.text(q[0] / cam.width + 0.012, q[1] / cam.height, "%.1f%%" % (100 * st.acc[i]),
                    0.028, "regular", layout.STRONG_INK, op, "lm")
            self.chart.legend(sc, cam, [("parameters kept: %d%%" % round(100 * st.kept[i]),
                                         KEPT_INK)], op, (CHART[0][0], CHART[1][1] + 0.10))
