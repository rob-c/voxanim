"""PCA and k-means: the two classical unsupervised methods on MNIST, drawn
on the autoencoders' stage so the two can be compared.

    mean   PC 1   PC 2   ...   PC 7            (top band: images, a variance bar each)

    the PC 1 / PC 2 plane                       original  2  10  50   (reconstructions)
    600 test digits, coloured      legend       later: the clusters-by-label grid
    by label

1. **mean** -- the mean of the 60,000 training images,
2. **components** -- the first seven principal components as images
   (orange +, blue -), each with a bar for the variance it explains,
3. **projection** -- the 600 test digits of the autoencoders' latent clouds
   projected onto PC 1 and PC 2 (:data:`voxanim.ae_anim.DIGIT_COLORS`, the
   same plane conventions as ``autoencoder-dense-2d``),
4. **rebuild** -- one digit rebuilt from its first 2, 10 and 50 coordinates,
   with the error of each,
5. **k-means** -- ten centroids dropped on the plane at random points; then,
   to convergence: every point joins its nearest centroid (its colour
   switches to the centroid's), every centroid walks to the mean of its
   points; the inertia reads out at every iteration and the
   clusters-by-label grid fills in,
6. **result** -- how often the clusters agree with the labels, under the
   best one-to-one matching: honestly, not very.

Each cluster is coloured by the digit it is matched to at the end, so a
point that keeps its colour when the assignment is made is one k-means gets
right.  Every number comes from :func:`voxanim.pca_kmeans.trace`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .ae_anim import DIGIT_COLORS, FRAME
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .netanim import NetAnimation, column_pitch, div_colors, gray_colors, hash01, lit_colors
from .pca_kmeans import K_CLUSTERS, RECON_K
from .tensor import Block
from .text import num

SHOWN_PCS = 7                                  # components drawn as images
BAND_Y = 0.30                                  # layout y of the top band
BAND_X = (0.07, 0.93)                          # layout x of its first and last sheet
BAND_WIDTH = 0.085                             # on-screen width of a band sheet (frame fraction)
BAR_WIDTH = 6.0                                # the variance bars, px
BAR_INK = tuple(float(v) for v in C.INPUT)
PLANE_AT = (0.255, 0.735)                      # layout centre of the PC plane
PLANE_S = 27.0                                 # the plane's height, world units
PLANE_MARGIN = 0.04                            # of the data range, kept clear at each edge
LEGEND_X, LEGEND_Y0, LEGEND_DY = 0.49, 0.555, 0.041
RECON_X = (0.585, 0.70, 0.815, 0.93)           # layout x of the original and the rebuilds
RECON_Y = 0.60
GRID_AT = (0.80, 0.70)                         # the clusters-by-label grid
GRID_SCALE = 1.2
GRID_ROW_PITCH = 1.3                           # rows a little apart (the view foreshortens them)
GRID_LO = np.array([0.80, 0.81, 0.82])         # a count of 0 in the grid: pale ...
GRID_HI = np.array([0.16, 0.18, 0.20])         # ... the largest count: ink
POINT = 0.5                                    # size of a cloud point, world units
READOUT_DROP = 0.055                           # the inertia read-out below the plane's bottom edge
CENTROID = 1.5                                 # size of a centroid
ASSIGN, MOVE = 0.35, 0.75                      # of an iteration: colours switch by, walk until


class PCAKMeans(NetAnimation):
    """The six phases of the module docstring on a :class:`voxanim.pca_kmeans.PCATrace`."""

    PHASES = ("mean", "components", "projection", "rebuild", "k-means", "result")

    def __init__(self, tr):
        super().__init__("PCA and k-means", "", 9.0)
        self.tr = tr
        self.hold = 2.5
        cam = self.cam
        scale = self._fit_scale(BAND_WIDTH)
        xs = np.linspace(BAND_X[0], BAND_X[1], SHOWN_PCS + 1)
        self.band = [self.sheet((28, 28), (x, BAND_Y), scale, "pc%d" % i) for i, x in enumerate(xs)]
        self.recon = [self.sheet((28, 28), (x, RECON_Y), scale, "r%d" % i)
                      for i, x in enumerate(RECON_X)]
        self.grid = Block((K_CLUSTERS, 10, 1), scale=GRID_SCALE, name="grid",
                          pitch=(GRID_ROW_PITCH, column_pitch(cam, GRID_SCALE)))
        self.grid.place(cam.frac(*GRID_AT))
        self.blocks_extra = [self.grid]
        # the plane: a flat rectangle facing the viewer, equal scale on both axes
        z = tr.cloud_z
        span = (z.max(0) - z.min(0)) * (1 + 2 * PLANE_MARGIN)
        self.lo = z.min(0) - PLANE_MARGIN * (z.max(0) - z.min(0))
        self.hi = self.lo + span
        M = cam.M
        h = np.array([M[1, 1], -M[1, 0], 0.0])
        self.centre = cam.frac(*PLANE_AT)
        self.e1 = h / np.linalg.norm(h) * PLANE_S * span[0] / span[1]
        self.e2 = np.array([0.0, 0.0, 1.0]) * PLANE_S
        self.cloud_pts = self.to_plane(z)
        self.pc_max = float(np.abs(tr.components[:, :SHOWN_PCS]).max())
        # k-means: the clusters coloured by the digit each is matched to at the end
        self.purity, self.mapping = tr.matching()
        self.order = sorted(range(K_CLUSTERS), key=lambda j: self.mapping[j])   # grid rows
        self.confusions = [self._confusion(a) for a in tr.assignments]

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.3

        def add(name, d, gap=0.0):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("mean", 2.7)
        add("pcs", 6.5, 0.6)
        add("project", 5.5, 0.4)
        add("rebuild", 7.5, 0.6)
        add("seed", 1.6, 0.2)
        it = Tempo(slow=1.8, fast=0.55, ramp_in=3.0, ramp_out=2.0).durations(tr.iterations)
        self.i_starts, self.i_durs = t + np.concatenate([[0.0], np.cumsum(it)[:-1]]), it
        add("iterate", float(it.sum()), 0.4)
        add("result", 3.5)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["pcs"][0], Tm["project"][0], Tm["rebuild"][0], Tm["seed"][0],
                      Tm["result"][0]]

    # ------------------------------------------------------------------ setup
    def _fit_scale(self, width):
        """Voxel scale at which a 28 x 28 sheet is ``width`` of the frame wide."""
        probe = Block((28, 28, 1), scale=1.0)
        lo, hi = probe.origin, probe.origin + probe.extent
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        return width / ((s[:, 0].max() - s[:, 0].min()) / self.cam.width)

    def _confusion(self, lab):
        """``(K, 10)`` counts of assignment ``lab``, rows in :attr:`order`."""
        M = np.zeros((K_CLUSTERS, 10), int)
        np.add.at(M, (lab, self.tr.cloud_labels), 1)
        return M[self.order]

    def to_plane(self, z):
        """World position(s) of coordinates ``z`` ``(2,)`` or ``(m, 2)`` on the plane."""
        u = (np.asarray(z, np.float64) - self.lo) / (self.hi - self.lo)
        return self.centre + (u[..., :1] - 0.5) * self.e1 + (u[..., 1:] - 0.5) * self.e2

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def iteration(self, t):
        """k-means at time ``t``: ``(iterations finished, progress through the
        current one)``; before the sweep ``(0, 0)``, after it ``(I, 0)``."""
        if t < self.Tm["iterate"][0]:
            return 0, 0.0
        if t >= self.Tm["iterate"][1]:
            return self.tr.iterations, 0.0
        k = int(np.searchsorted(self.i_starts, t, side="right") - 1)
        return k, (t - self.i_starts[k]) / self.i_durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        self._band(sc, t, back)
        self._plane(sc, t, back)
        self._rebuild(sc, t, back)
        self._grid(sc, t, back)
        subtitle = self._subtitle(t, ph)
        if t < self.Tm["mean"][0] or reset > 0:            # the loop returns to the opening
            subtitle = self._subtitle(0.0, 0)
        self.subtitle = subtitle
        self.step_indicator(sc, ph, self.PHASES)

    # -- the top band ----------------------------------------------------------------
    def _band(self, sc, t, back):
        """The mean digit and the first components, each with its variance bar."""
        tr = self.tr
        e = tr.explained
        mean_in = smoothstep(0.0, 0.5, self.progress("mean", t)) * back
        s = self.band[0]
        s.fill(gray_colors(tr.mean))
        s.cage = mean_in
        s.fade(mean_in)
        s.visible = mean_in > 0
        self.caption(sc, s, "mean digit", mean_in)
        pcs = self.progress("pcs", t)
        for i in range(1, SHOWN_PCS + 1):
            op = smoothstep(0.0, 0.12, remap(pcs, (i - 1) / SHOWN_PCS, 1.0)) * back
            s = self.band[i]
            s.visible = op > 0
            if op <= 0:
                continue
            s.fill(div_colors(tr.components[:, i - 1], self.pc_max))
            s.cage = op
            s.fade(op)
            self._variance_bar(sc, s, e[i - 1] / e[0], op)
            self.caption(sc, s, "PC %d  %.1f%%" % (i, 100 * e[i - 1]), op, offset=0.03)

    def _variance_bar(self, sc, sheet, length, op):
        """A bar under ``sheet``, ``length`` (0..1) of its width, in the diverging
        scale's neutral grey."""
        lo, hi = sheet.bbox()
        y = lo[1]
        z = lo[2] - 1.6 * sheet.block.scale
        x0 = lo[0]
        x1 = lo[0] + (hi[0] - lo[0]) * length
        sc.line([x0, y, z], [x1, y, z], width=BAR_WIDTH, color=BAR_INK, opacity=0.9 * op,
                bias=0.0)

    # -- the plane and the cloud ------------------------------------------------------
    def _colours(self, t):
        """Colour of every cloud point at time ``t``: its label, or (as k-means
        assigns it) its cluster's."""
        tr = self.tr
        lab = tr.cloud_labels
        by_label = np.array([DIGIT_COLORS[v] for v in lab])
        done, u = self.iteration(t)
        if done == 0 and u <= 0:
            return by_label
        cur = min(done, tr.iterations - 1)
        prev = np.array([DIGIT_COLORS[self.mapping[j]] for j in tr.assignments[cur - 1]]) \
            if cur > 0 else by_label
        if done >= tr.iterations:
            return np.array([DIGIT_COLORS[self.mapping[j]] for j in tr.assignments[-1]])
        new = np.array([DIGIT_COLORS[self.mapping[j]] for j in tr.assignments[cur]])
        w = smoothstep(0.05, ASSIGN, u)
        return prev + (new - prev) * w

    def _centroids(self, t):
        """Centroid positions ``(K, 2)`` at time ``t`` (walking in the MOVE part of an
        iteration), or None before they are placed."""
        tr = self.tr
        seed = self.progress("seed", t)
        if seed <= 0:
            return None
        done, u = self.iteration(t)
        if done >= tr.iterations:
            return tr.centroids[-1]
        a, b = tr.centroids[done], tr.centroids[min(done + 1, len(tr.centroids) - 1)]
        w = ease_in_out(remap(u, ASSIGN + 0.05, MOVE))
        return a + (b - a) * w

    def _plane(self, sc, t, back):
        """The PC 1 / PC 2 plane, the cloud, the legend, the axes; the centroids."""
        tr = self.tr
        fade = smoothstep(0.0, 0.8, self.progress("project", t)) * back
        if fade <= 0.01:
            return
        cam, c, a, b = self.cam, self.centre, self.e1 / 2, self.e2 / 2
        corners = [c - a - b, c + a - b, c + a + b, c - a + b]
        for i in range(4):
            sc.line(corners[i], corners[(i + 1) % 4], width=1.0, color=layout.NOTE_INK,
                    opacity=0.75 * fade, bias=0.0)
        cols = self._colours(t)
        arrive = self.progress("project", t)
        for n, (p, col) in enumerate(zip(self.cloud_pts, cols)):
            op = smoothstep(0.0, 0.25, remap(arrive, 0.1 + 0.6 * hash01(n), 1.0)) * back
            if op > 0.01:
                sc.voxel(p, col, op, shadow=False, reflect=False, size=POINT)
        # the rebuilt digit's own point
        rb = self.progress("rebuild", t)
        if rb > 0:
            op = smoothstep(0.0, 0.2, rb) * (1 - smoothstep(0.0, 0.3, self.progress("seed", t)))
            if op > 0.01:
                p = self.to_plane(tr.digit_z)
                sc.voxel(p, (1.0, 1.0, 1.0), op * back, glow=0.4, shadow=False, reflect=False,
                         size=1.1)
                sc.wire_box(p - 1.0, p + 1.0, color=FRAME, width=1.2, opacity=0.9 * op * back,
                            bias=0.1)
        # the centroids
        cen = self._centroids(t)
        if cen is not None:
            seed = self.progress("seed", t)
            done, u = self.iteration(t)
            coloured = smoothstep(0.0, ASSIGN, u) if done == 0 else 1.0
            for j in range(K_CLUSTERS):
                op = smoothstep(0.0, 0.3, remap(seed, 0.08 * j, 0.08 * j + 0.3)) * back
                if op <= 0.01:
                    continue
                p = self.to_plane(cen[j])
                col = np.array((0.95, 0.95, 0.95)) + \
                    (DIGIT_COLORS[self.mapping[j]] - 0.95) * coloured
                sc.voxel(p, col, op, shadow=False, reflect=False, size=CENTROID)
                sc.wire_box(p - 1.15, p + 1.15, color=FRAME, width=1.3, opacity=0.9 * op,
                            bias=0.1)
                sc.probe(("cen", j), p, tol=4.0)
        # legend, right of the plane
        for d in range(10):
            p = cam.frac(LEGEND_X, LEGEND_Y0 + LEGEND_DY * d)
            sc.voxel(p, DIGIT_COLORS[d], fade, shadow=False, reflect=False, size=0.7)
            q = cam.project(p)
            sc.text(q[0] / cam.width + 0.016, q[1] / cam.height, str(d), 0.029, "regular",
                    layout.CAPTION_INK, fade * self._cap, "lm")
        # axis names and the coordinate range
        op = fade * self._cap
        marks = [(c - b, "PC 1", (0.0, 0.012), "mt", 0.029, layout.CAPTION_INK),
                 (c - a, "PC 2", (-0.012, 0.0), "rm", 0.029, layout.CAPTION_INK),
                 (c - a - b, num(self.lo[0], 1), (0.0, 0.012), "lt", 0.026, layout.NOTE_INK),
                 (c + a - b, num(self.hi[0], 1), (0.0, 0.012), "rt", 0.026, layout.NOTE_INK),
                 (c - a - b, num(self.lo[1], 1), (-0.012, 0.0), "rb", 0.026, layout.NOTE_INK),
                 (c - a + b, num(self.hi[1], 1), (-0.012, 0.0), "rt", 0.026, layout.NOTE_INK)]
        for p, s, (dx, dy), anchor, size, ink in marks:
            q = cam.project(p)
            sc.text(q[0] / cam.width + dx, q[1] / cam.height + dy, s, size,
                    "mono" if ink is layout.NOTE_INK else "regular", ink, op, anchor)
        # the inertia read-out under the plane's axis label
        done, u = self.iteration(t)
        if done > 0 or u > 0:
            k = max(min(done + (1 if u > 0.05 else 0), tr.iterations), 1)
            text = "iteration %d of %d    inertia %.0f" % (k, tr.iterations, tr.inertias[k - 1])
            q = cam.project(c - b)
            sc.text(q[0] / cam.width, q[1] / cam.height + READOUT_DROP, text, 0.03, "regular",
                    layout.STRONG_INK, back * self._cap, "mt")

    # -- the reconstructions -------------------------------------------------------------
    def _rebuild(self, sc, t, back):
        """The digit and its rebuilds from RECON_K components, in turn."""
        tr = self.tr
        rb = self.progress("rebuild", t)
        gone = 1.0 - smoothstep(0.0, 0.4, self.progress("seed", t))
        texts = ["original"] + ["k = %d\nerror %.3f" % (k, e)
                                for k, e in zip(RECON_K, tr.recon_errors)]
        imgs = [tr.digit] + list(tr.recons)
        for i, s in enumerate(self.recon):
            op = smoothstep(0.0, 0.15, remap(rb, i * 0.22, 1.0)) * gone * back
            s.visible = op > 0.01
            if not s.visible:
                continue
            s.fill(gray_colors(np.clip(imgs[i], 0, 1)))
            s.cage = op
            s.fade(op)
            self.caption(sc, s, texts[i], op, max_width=0.12)

    # -- the clusters-by-label grid ---------------------------------------------------------
    def _grid(self, sc, t, back):
        """``K x 10`` counts: cluster rows (ordered and coloured by the digit each
        is matched to) against the true label, filled as k-means runs."""
        tr = self.tr
        done, u = self.iteration(t)
        show = smoothstep(0.0, 0.5, remap(t, self.Tm["seed"][1] - 0.2, self.Tm["iterate"][0]))
        op = show * back
        g = self.grid
        if op <= 0.01:
            g.present[...] = False
            return
        k = min(max(done + (1 if u > ASSIGN else 0), 1), tr.iterations)
        assigned = done > 0 or u > ASSIGN
        M = self.confusions[k - 1] if assigned else np.zeros_like(self.confusions[0])
        vmax = max(1.0, float(self.confusions[-1].max()))
        cols = lit_colors(M.ravel(), vmax, GRID_LO, GRID_HI)
        cols[M.ravel() == 0] = GRID_LO
        g.base[:, :, 0] = cols.reshape(K_CLUSTERS, 10, 3)
        g.opacity *= op
        g.shadow = False
        h = 0.5 * g.scale
        for r, j in enumerate(self.order):                  # a swatch per cluster row
            p = g.cell_center(r, 0, 0) - np.array([1.4 * g.scale, 0.0, 0.0])
            sc.voxel(p, DIGIT_COLORS[self.mapping[j]], op, shadow=False, reflect=False,
                     size=0.6)
        for d in range(10):                                  # the true labels below
            p = g.cell_center(K_CLUSTERS - 1, d, 0)
            self.caption(sc, (p - h, p + h), str(d), op, size=0.029)
        lo, hi = g.origin, g.origin + g.extent
        self.caption(sc, (lo, hi), "clusters (rows) against labels", op, above=True)
        self.caption(sc, (lo, hi), "true label", op, offset=0.045)
        matched = int(np.trace(M))
        if done >= tr.iterations:
            self.caption(sc, (lo, hi), "on the diagonal: %d of %d"
                         % (matched, len(tr.cloud_labels)), op, offset=0.085,
                         ink=layout.STRONG_INK, size=0.03)

    # -- words -----------------------------------------------------------------------------
    def _subtitle(self, t, ph):
        """The subtitle for phase ``ph`` at time ``t``."""
        tr = self.tr
        e = tr.explained
        cum = np.cumsum(e)
        err = tr.recon_errors
        if ph == 0:
            return ("PCA: the mean of the %d training images, and the directions in which "
                    "they vary most" % 60000)
        if ph == 1:
            return ("the principal components, as images: PC 1 explains %.1f%% of the variance, "
                    "PC 2 %.1f%%; the first 10 together %.0f%%, the first 50 %.0f%%"
                    % (100 * e[0], 100 * e[1], 100 * cum[9], 100 * cum[49]))
        if ph == 2:
            return ("%d test digits projected onto PC 1 and PC 2 (two numbers each), coloured "
                    "by label: compare the autoencoder's 2-D latent space" % len(tr.cloud_labels))
        if ph == 3:
            rb = self.progress("rebuild", t)
            if rb < 0.22:
                return ("rebuilding a %d from its coordinates: mean + z₁ PC 1 + z₂ PC 2 + …; "
                        "its (z₁, z₂) = (%s, %s) is its point in the plane"
                        % (tr.digit_label, num(tr.digit_z[0]), num(tr.digit_z[1])))
            return ("from 2 components the error is %.3f, from 10 it is %.3f, from 50 it is "
                    "%.3f: the dropped components carried the detail" % (err[0], err[1], err[2]))
        if ph == 4:
            done, u = self.iteration(t)
            if done == 0 and u <= 0:
                return ("k-means with k = %d on the two coordinates: %d random points become "
                        "the first centroids" % (K_CLUSTERS, K_CLUSTERS))
            k = min(done + (1 if u > 0 else 0), tr.iterations)
            if u <= ASSIGN and k <= 2:
                return ("iteration %d: every point joins its nearest centroid (its colour "
                        "becomes the centroid's)" % k)
            if k <= 2:
                return ("iteration %d: every centroid moves to the mean of its points; the "
                        "inertia (summed squared distances) falls" % k)
            return ("repeat: assign, move; the inertia falls at every step, %.0f → %.0f, "
                    "until no point changes cluster" % (tr.inertias[0], tr.inertias[k - 1]))
        return ("converged after %d iterations: matched one-to-one, the clusters agree with "
                "the labels %.0f%% of the time; two numbers per digit are not enough"
                % (tr.iterations, 100 * self.purity))
