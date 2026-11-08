"""Autoencoders, animated: digits squeezed through a 2-D or 3-D latent space
and rebuilt.

    image -> encoder -> code z (2 or 3 numbers) -> decoder -> reconstruction
                              |
                       the latent space: a cube (3-D) or plane (2-D) holding
                       600 encoded test digits, coloured by their label

The chain runs along the bottom; the latent space sits above the code.
Three digits go through in turn -- each lands at its point in the latent
space, next to the other digits of its kind -- and then a walk through the
latent space shows the decoder drawing every point in between.

Dense and convolutional autoencoders share the choreography; each layer
transition is drawn as what it is:

* dense: the strongest weighted inputs fly across, the layer fills in,
* conv: a frame sweeps the input while the feature map fills row by row
  (the decoder's "up x2 + conv" sweeps at half speed),

and after ReLU only values > 0 light up.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .easing import ease_in_out, ease_out, remap, smoothstep
from .mlp import (ACT_HI, ACT_LO, INK_HI, INK_LO, X, NetAnimation, amax, div_colors,
                  gray_colors, lit_colors, particle)
from .tensor import Block

# Ten distinguishable hues (categorical; no two neighbours alike).
DIGIT_COLORS = [C.rgb(h) for h in ("#e6194b", "#f58231", "#ffd31a", "#3cb44b", "#42d4f4",
                                   "#4363d8", "#911eb4", "#f032e6", "#9a6324", "#469990")]
CHAIN_Y = 0.70
CHAIN_X = (0.075, 0.19, 0.305, 0.50, 0.695, 0.81, 0.925)
WIDTH = (0.085, 0.085, 0.075, 0.0, 0.075, 0.085, 0.085)


def _shapes(kind):
    if kind == "dense":
        return [(28, 28, 1), (16, 16, 1), (8, 8, 1), None, (8, 8, 1), (16, 16, 1), (28, 28, 1)]
    return [(28, 28, 1), (14, 14, 8), (7, 7, 16), None, (7, 7, 16), (14, 14, 8), (28, 28, 1)]


def _ops(kind):
    if kind == "dense":
        return ["dense"] * 6
    return ["conv", "conv", "dense", "dense", "upconv", "upconv"]


class AutoencoderAnimation(NetAnimation):
    def __init__(self, tr, title=None):
        kind, L = tr.kind, tr.latent
        title = title or "%s autoencoder, %d-D latent" % ("Dense" if kind == "dense" else
                                                        "Conv", L)
        super().__init__(title, "", 9.0)
        self.tr, self.kind, self.L = tr, kind, L
        self.shapes, self.ops = _shapes(kind), _ops(kind)
        self.hold = 2.5
        cam = self.cam

        # ---- the chain -------------------------------------------------------------
        self.blocks = []
        for k, shp in enumerate(self.shapes):
            if shp is None:
                self.blocks.append(None)
                continue
            self.blocks.append(self._fit_block(shp, CHAIN_X[k], WIDTH[k], "t%d" % k))
        self.code = Block((1, L, 1), scale=2.0, name="code").place(cam.frac(0.50, CHAIN_Y))
        self.blocks_extra = [b for b in self.blocks if b is not None] + [self.code]

        # ---- the latent space ------------------------------------------------------------
        cloud = tr.cloud
        # Robust axes: centre on the median and compress the long tails, so
        # the clusters spread over the space instead of one crowded corner.
        self.med = np.median(cloud, axis=0)
        self.mad = np.median(np.abs(cloud - self.med), axis=0) + 1e-9
        w = self._warp(cloud)
        self.lo, self.hi = np.percentile(w, 1, axis=0), np.percentile(w, 99, axis=0)
        S = 21.0
        self.S = S
        c = cam.frac(0.45, 0.38)
        self.centre = c
        if L == 3:
            self.box_lo, self.box_hi = c - S / 2, c + S / 2
        else:
            # a flat rectangle facing the viewer: one axis horizontal on screen
            M = cam.M
            h = np.array([M[1, 1], -M[1, 0], 0.0])
            self.e1 = h / np.linalg.norm(h) * 1.25 * S
            self.e2 = np.array([0.0, 0.0, 1.0]) * S
            self.box_lo = c - self.e1 / 2 - self.e2 / 2
            self.box_hi = c + self.e1 / 2 + self.e2 / 2
        self.cloud_pts = np.array([self.to_box(z) for z in cloud])

        # ---- per-digit values (block colours and particles) ------------------------------
        self.vals = [self._values(f) for f in tr.fwd]
        self.parts = [self._particles(f) for f in tr.fwd]
        self.codes = [np.asarray(f["z"]).reshape(-1) for f in tr.fwd]
        self.zmax = np.abs(cloud).max(0)

        # ---- timeline ------------------------------------------------------------------------
        Tm, t = {}, 1.6                      # the latent cloud fades in first
        self.digit_t = []
        for d in range(len(tr.images)):
            slow = d == 0
            durs = [2.0, 1.6, 1.4, 1.6, 1.4, 1.6, 2.0, 1.8] if slow else \
                [0.9, 0.8, 0.7, 1.0, 0.7, 0.8, 0.9, 1.4]
            starts = []
            for dd in durs:
                starts.append((t, t + dd))
                t += dd
            self.digit_t.append(starts)
            t += 0.5
        Tm["walk"] = (t, t + 7.0)
        t += 7.4
        self.Tm = Tm
        self.finish(t)
        # the walk: through the three digits' codes and back to the first
        self.walk = [self.codes[0], self.codes[1], self.codes[2], self.codes[0]]

    # ------------------------------------------------------------------ setup helpers
    def _fit_block(self, shp, fx, width, name):
        """A block whose on-screen width is ``width`` (a fraction of the frame).
        Multi-channel tensors get a gap between channels, so each feature map
        reads as its own slab."""
        gap = 1.2 if shp[2] > 1 else 0.0
        probe = Block(shp, scale=1.0, group=1, gap=gap)
        lo, hi = probe.origin, probe.origin + probe.extent
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                      for z in (lo[2], hi[2])])
        s = self.cam.project(c)
        w = (s[:, 0].max() - s[:, 0].min()) / self.cam.width
        b = Block(shp, scale=width / w, group=1, gap=gap, name=name)
        b.place(self.cam.frac(fx, CHAIN_Y))
        return b

    def _warp(self, z):
        return np.arcsinh((np.asarray(z) - self.med) / (2.0 * self.mad))

    def to_box(self, z):
        u = np.clip((self._warp(z) - self.lo) / (self.hi - self.lo), 0.02, 0.98)
        if self.L == 3:
            return self.box_lo + u * (self.box_hi - self.box_lo)
        return self.centre + (u[0] - 0.5) * self.e1 + (u[1] - 0.5) * self.e2

    def _acts(self, f):
        """The seven tensors of the chain as HWC arrays (code at index 3)."""
        a = f["acts"]
        if self.kind == "dense":
            return [np.asarray(v).reshape(s) if s else np.asarray(v)
                    for v, s in zip(a, self.shapes)]
        return [np.asarray(v) for v in a]

    def _values(self, f):
        acts = self._acts(f)
        out = []
        for k, (a, shp) in enumerate(zip(acts, self.shapes)):
            if shp is None:
                out.append(None)
                continue
            v = a.reshape(-1)
            if k == 0:
                col = lit_colors(v, 1.0, INK_LO, INK_HI)    # the input's zeros are real zeros
            elif k == 6:
                col = gray_colors(v)       # a sigmoid output is never exactly 0: plain greyscale
            else:
                col = lit_colors(v, amax(v), ACT_LO, ACT_HI)
            out.append((col.reshape(shp + (3,)), a.reshape(shp)))
        return out

    WEIGHT_KEYS = None          # subclasses: transition -> parameter name
    WALK_SUBTITLE = ("walking through latent space: the decoder draws every point between "
                     "the digits")

    def _weights(self, k):
        P = self.tr.P
        if self.WEIGHT_KEYS:
            return P[self.WEIGHT_KEYS[k]]
        if self.kind == "dense":
            return P["W%d" % k]
        return {2: P["Wz"], 3: P["Wd"]}[k]

    def _particles(self, f):
        """Per dense transition: the strongest weighted inputs (src, dst, sign)."""
        acts = self._acts(f)
        parts = {}
        for k, op in enumerate(self.ops):
            if op != "dense":
                continue
            src = acts[k].reshape(-1)
            W = self._weights(k)
            c = src[:, None] * W
            K = min(70, int((np.abs(c) > 0).sum()))
            idx = np.argpartition(-np.abs(c).ravel(), K - 1)[:K]
            cmax = amax(c)
            parts[k] = [(divmod(int(m), W.shape[1]), c.ravel()[m] / cmax,
                         0.5 * ((m * 2654435761) % 997) / 997) for m in idx]
        return parts

    # ------------------------------------------------------------------ geometry helpers
    def node(self, k, flat):
        """World centre of flat index ``flat`` of chain tensor ``k``."""
        if k == 3:
            return self.code.cell_center(0, int(flat), 0)
        b = self.blocks[k]
        r, c, ch = np.unravel_index(int(flat), (b.H, b.W, b.C))
        return b.cell_center(r, c, ch)

    def step_of(self, t):
        for d in range(len(self.digit_t) - 1, -1, -1):
            if t >= self.digit_t[d][0][0]:
                return d + 1 if t < self.Tm["walk"][0] else len(self.digit_t) + 1
        return 0

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        nd = len(self.digit_t)
        phases = ["latent space"] + ["digit %d" % (d + 1) for d in range(nd)] + ["latent walk"]
        ph = 0 if reset > 0 else self.step_of(t)
        walk = remap(t, *self.Tm["walk"])
        self._landed = set()
        if t >= self.Tm["walk"][0]:
            self._walk(sc, walk, back)
            self.subtitle = self.WALK_SUBTITLE
        else:
            d = max(0, ph - 1)
            self._digit(sc, t, d, back, active=ph >= 1)
        self._cloud(sc, t, back)             # after the digits: the legend knows what landed
        if t < self.digit_t[0][0][0] or reset > 0:          # the loop returns to the opening
            self.subtitle = ("%d test digits encoded to %d numbers each, coloured by their "
                             "label (the autoencoder never sees labels)"
                             % (len(self.tr.cloud), self.L))
        self.step_indicator(sc, ph, phases)
        self._captions(sc, back)

    # -- the latent space ------------------------------------------------------------------
    def _cloud(self, sc, t, back):
        fade = smoothstep(0.1, 1.4, t) * back
        if fade <= 0.01:
            return
        if self.L == 3:
            sc.wire_box(self.box_lo, self.box_hi, color=layout.NOTE_INK, width=1.0,
                        opacity=0.75 * fade, bias=0.0)
        else:
            c, a, b = self.centre, self.e1 / 2, self.e2 / 2
            corners = [c - a - b, c + a - b, c + a + b, c - a + b]
            for i in range(4):
                sc.line(corners[i], corners[(i + 1) % 4], width=1.0, color=layout.NOTE_INK,
                        opacity=0.75 * fade, bias=0.0)
        for p, lab in zip(self.cloud_pts, self.tr.cloud_labels):
            sc.voxel(p, DIGIT_COLORS[lab], fade, shadow=False, reflect=False, size=0.5)
        # legend: one swatch per digit, in a column right of the latent space
        for dgt in range(10):
            fx, fy = 0.735, 0.20 + 0.041 * dgt
            p = self.cam.frac(fx, fy)
            hit = dgt in self._landed          # a digit we just put through: highlighted
            sc.voxel(p, DIGIT_COLORS[dgt], fade, shadow=False, reflect=False,
                     size=1.1 if hit else 0.7, glow=0.3 if hit else 0.0)
            q = self.cam.project(p)
            sc.text(q[0] / self.cam.width + 0.016, q[1] / self.cam.height,
                    str(dgt), 0.031 if hit else 0.029, "regular",
                    layout.STRONG_INK if hit else layout.CAPTION_INK, fade * self._cap, "lm")
        # axis names on the edges nearest the viewer, just outside the box
        lo, hi = self.box_lo, self.box_hi
        mid = (lo + hi) / 2
        cam = self.cam
        if self.L == 3:
            marks = [(np.array([mid[0], hi[1], lo[2]]), "z1", (0.0, 0.03), "mt"),
                     (np.array([lo[0], lo[1], mid[2]]), "z3", (-0.012, 0.0), "rm"),
                     (np.array([lo[0], mid[1], lo[2]]), "z2", (-0.012, 0.012), "rt")]
        else:
            c = self.centre
            marks = [(c - self.e2 / 2, "z1", (0.0, 0.012), "mt"),
                     (c - self.e1 / 2, "z2", (-0.012, 0.0), "rm")]
        for p, name, (dx, dy), anchor in marks:
            q = cam.project(p)
            sc.text(q[0] / cam.width + dx, q[1] / cam.height + dy, name, 0.031, "regular",
                    layout.CAPTION_INK, fade * self._cap, anchor)

    # -- one digit through the chain -----------------------------------------------------------
    def _digit(self, sc, t, d, back, active):
        if not active:
            for b in self.blocks:
                if b is not None:
                    b.present[...] = False
            self.code.present[...] = False
            return
        segs = self.digit_t[d]
        vals, parts = self.vals[d], self.parts[d]
        lab = int(self.tr.labels[d])
        # transitions 0..2: encoder, 3: code -> latent point, 4..6: decoder (seg 4..6 = k 3..5)
        prog = [remap(t, a, b) for a, b in segs]
        # the input is there from the start; tensor k is made by transition k - 1,
        # which runs in segment k - 1 (encoder) or k (decoder: segment 3 is the
        # code's flight into the latent space)
        self._fill(0, vals[0], 1.0, back)
        for k in (1, 2, 4, 5, 6):
            self._fill(k, vals[k], prog[k - 1] if k < 3 else prog[k], back)
        # the code vector
        z = self.codes[d]
        cp = prog[2]
        self.code.base[0, :, 0] = div_colors(z / self.zmax, 1.0)
        self.code.present[0, :, 0] = np.arange(self.L) < cp * (self.L + 1)
        self.code.opacity *= back
        # transitions
        for k, op in enumerate(self.ops):
            seg = k if k < 3 else k + 1
            u = prog[seg]
            if not 0 < u < 1:
                continue
            if op == "dense":
                for (i, j), v, start in parts[k]:
                    uu = remap(u, start, start + 0.5)
                    particle(sc, self.node(k, i), self.node(k + 1, j), uu,
                             C.POSITIVE if v > 0 else C.NEGATIVE, 0.5,
                             opacity=0.35 + 0.65 * abs(v), lift=2.0)
            else:
                self._sweep(sc, k, u)
            sc.probe(("tr", k), (u * 30, 0, 0), tol=8.0)
        # the code flies into the latent space and stays there
        fly = prog[3]
        dest = self.to_box(z)
        for dd in range(d + 1):
            if dd < d or fly >= 1:
                pd = self.to_box(self.codes[dd])
                lab_d = int(self.tr.labels[dd])
                sc.voxel(pd, DIGIT_COLORS[lab_d], back, glow=0.35, shadow=False,
                         reflect=False, size=1.5)
                self._landed.add(lab_d)
        if 0 < fly < 1:
            particle(sc, self.code.center, dest, ease_out(fly), DIGIT_COLORS[lab], 1.3,
                     lift=3.0)
            sc.probe(("fly",), (fly * 30, 0, 0), tol=8.0)
        if fly >= 1:
            sc.wire_box(dest - 1.0, dest + 1.0, color=(0.96, 0.96, 0.96), width=1.2,
                        opacity=0.9 * (1 - smoothstep(0.6, 1.0, prog[4])) * back, bias=0.1)
        # subtitle
        names = {"dense": "dense layer", "conv": "3%s3 conv, stride 2" % X,
                 "upconv": "upsample %s2, then 3%s3 conv" % (X, X)}
        cur = next((s for s in range(8) if 0 < prog[s] < 1), None)
        if cur is None:
            cur = 7 if prog[7] > 0 else 0
        zs = ", ".join("%.2f" % v for v in z)
        if cur < 3:
            k = cur
            self.subtitle = "encoder: %s  (digit %d)" % (names[self.ops[k]], lab)
        elif cur == 3:
            self.subtitle = "the code z = (%s) is a point in the latent space" % zs
        elif cur < 7:
            k = cur - 1
            self.subtitle = "decoder: %s, rebuilding the image from z" % names[self.ops[k]]
        else:
            err = float(np.mean((vals[6][1].reshape(-1) - self.tr.images[d]) ** 2))
            self.subtitle = "reconstruction error: mean (x - x%s)%s = %.4f" % (
                "̂", "²", err)
        self._err = prog[7]

    def _fill(self, k, val, p, back):
        """Tensor ``k`` appears as the transition producing it runs (row order)."""
        b = self.blocks[k]
        cols, a = val
        b.base[...] = cols
        rows = np.arange(b.H)[:, None, None] / b.H
        b.present[...] = (p * 1.12 - 0.1 > rows) & np.ones((1, b.W, b.C), bool)
        if k in (0, 6):
            on = a > 0
        else:
            on = a > 0
        b.glow[...] = 0.06 * on * smoothstep(0.85, 1.0, p)
        b.opacity *= back

    def _sweep(self, sc, k, u):
        """A frame sweeping the source of conv transition ``k``."""
        src, dst = self.blocks[k], self.blocks[k + 1]
        rows = dst.H
        r = min(rows - 1, int(u * rows))
        if self.ops[k] == "upconv":
            sr = r // 2
            lo, hi = src.box((sr, sr + 1), (0, src.W), (0, src.C))
        else:
            s0 = max(0, 2 * r - 1)
            s1 = min(src.H, 2 * r + 2)
            lo, hi = src.box((s0, s1), (0, src.W), (0, src.C))
        sc.wire_box(lo, hi, inflate=0.05, color=(0.96, 0.96, 0.96), width=1.2, opacity=0.9,
                    bias=0.1)

    # -- the latent walk --------------------------------------------------------------------------
    def _walk(self, sc, u, back):
        seg = min(len(self.walk) - 2, int(u * (len(self.walk) - 1)))
        w = ease_in_out(u * (len(self.walk) - 1) - seg)
        z = self.walk[seg] + (self.walk[seg + 1] - self.walk[seg]) * w
        acts, img = self.tr.decode(z)
        # the encoder side rests; the decoder side shows the walk
        for k in (0, 1, 2):
            b = self.blocks[k]
            b.present[...] = False
        self.code.base[0, :, 0] = div_colors(z / self.zmax, 1.0)
        self.code.opacity *= back
        dec = [None, None, None, None] + list(acts[1:])
        for k in (4, 5, 6):
            a = np.asarray(dec[k])
            shp = self.shapes[k]
            v = a.reshape(-1)
            if k == 6:
                col = gray_colors(v)
            else:
                col = lit_colors(v, amax(v), ACT_LO, ACT_HI)
            b = self.blocks[k]
            b.base[...] = col.reshape(shp + (3,))
            b.opacity *= back
        p = self.to_box(z)
        lab = int(self.tr.labels[min(seg + (w > 0.5), 2) % 3])
        sc.voxel(p, (1.0, 1.0, 1.0), back, glow=0.5, shadow=False, reflect=False, size=1.4)
        sc.wire_box(p - 1.1, p + 1.1, color=DIGIT_COLORS[lab], width=1.3, opacity=0.9 * back,
                    bias=0.1)
        # the path so far
        prev = None
        for s in np.linspace(0, u, max(2, int(u * 60))):
            sg = min(len(self.walk) - 2, int(s * (len(self.walk) - 1)))
            ww = ease_in_out(s * (len(self.walk) - 1) - sg)
            q = self.to_box(self.walk[sg] + (self.walk[sg + 1] - self.walk[sg]) * ww)
            if prev is not None:
                sc.line(prev, q, width=1.4, color=(0.96, 0.96, 0.96), opacity=0.9 * back,
                        bias=0.1)
            prev = q
        sc.probe(("walk",), p, tol=4.0)
        self._err = 0.0

    # -- captions -------------------------------------------------------------------------------------
    def _captions(self, sc, back):
        conv = self.kind == "conv"
        names = ["input\n28%s28" % X,
                 "14%s14\n8 channels" % X if conv else "256",
                 "7%s7\n16 channels" % X if conv else "64",
                 None,
                 "7%s7\n16 channels" % X if conv else "64",
                 "14%s14\n8 channels" % X if conv else "256",
                 "output\n28%s28" % X]
        for k, b in enumerate(self.blocks):
            if b is None or not b.present.any():
                continue
            lo, hi = b.origin, b.origin + b.extent
            self.caption(sc, (lo, hi), names[k], back)
        if self.code.present.any():
            lo, hi = self.code.origin, self.code.origin + self.code.extent
            self.caption(sc, (lo, hi), "code z (%d)" % self.L, back)

