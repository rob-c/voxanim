"""Inside GPT-2: one attention head up close, then its layer, then the model.

One world and one continuous camera move; every number comes from the real
GPT-2 small (:mod:`voxanim.gpt2`) reading "One, two, three,":

1. tokens: each token's row of the embedding table plus its position's row
   gives the residual stream x, 6 tokens x 768 numbers,
2. fly into layer 5, head 12 (GPT-2's "previous-token" head):
   Q = LN(x) W_Q one row at a time, then K and V the same way,
3. the transpose: K flips over its diagonal into K^T,
4. Q K^T one score at a time: a row of Q turns to lie beside a column of K^T,
   they multiply element by element and the 64 products sum into one score
   (/ sqrt 64); then the causal mask,
5. softmax, row by row,
6. A V: each output row is a weighted mix of the value rows,
7. zoom out: the other 11 heads; their outputs are concatenated, projected
   by W_O and added to the stream; then the MLP,
8. zoom out again: all 12 layers, information flowing through them,
9. the last position's vector becomes next-token probabilities.

Matrices are flat textured panels (the "Falk scheme": A left of A.B, B above
it, so rows and columns line up).  Textures are mip-mapped per axis, so a
6 x 3072 matrix neither shimmers when small nor blurs its 6 rows.  The
zoom is a transform of the whole world (``Z``), so one model-space layout
serves every shot.
"""

from __future__ import annotations

import math

import numpy as np

from . import color as C
from . import gpt2 as G
from . import layout
from .easing import ease_in_out, ease_out, remap, smoothstep
from .mlp import ACT_HI, ACT_LO, OFF, NetAnimation

LAYER, HEAD = 4, 11          # the head we dive into: layer 5, head 12 (1-based)
TP = 16.0                    # model units per token (rows); a feature is 1 unit.
HT = 96.0                    # ... set so T tokens always span HT (see GPT2Animation)
MAX_TOKENS = 8
MOD = 3050.0                 # one layer's width along x
EMBED_X = -1050.0
CARD_W, CARD_H = 236.0, 198.0
GRID_COL, GRID_ROW = 260.0, 230.0
# card layout (Falk scheme), top-left corners relative to the card's bottom-left
Q_AT, S_AT, KT_AT, O_AT, V_AT = (0, 96), (70, 96), (70, 166), (172, 96), (172, 198)
EMB_TOK, EMB_POS = 340.0, 180.0
HALF = 0.5                   # W_Q / LN(x) draw their 768-long side at half scale


def bump(x, width):
    return math.exp(-(x / width) ** 2)


# ------------------------------------------------------------------ colours


def diverging(a):
    a = np.asarray(a, np.float64)
    m = float(np.percentile(np.abs(a), 99)) or 1.0
    v = np.clip(a / m, -1, 1)
    t = (np.abs(v) ** 0.75)[..., None]
    tgt = np.where(v[..., None] >= 0, C.POSITIVE, C.NEGATIVE)
    return (C.NEUTRAL + (tgt - C.NEUTRAL) * t).astype(np.float32)


def lit(a, vmax=None):
    """Only values > 0 light up."""
    a = np.asarray(a, np.float64)
    if vmax is None:
        vmax = float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0
    t = np.clip(a / vmax, 0, 1) ** 0.6
    out = ACT_LO + (ACT_HI - ACT_LO) * t[..., None]
    out[a <= 1e-3 * vmax] = OFF
    return out.astype(np.float32)


# ------------------------------------------------------------------ panels


class Panel:
    """A matrix drawn as a textured parallelogram.  ``origin`` is its top-left
    corner in model space; columns run along +x, rows down -z."""

    def __init__(self, colors, origin, cell=(TP, 1.0)):
        self.tex = np.asarray(colors, np.float32)
        self.R, self.C = self.tex.shape[:2]
        self.origin = np.asarray(origin, np.float64)
        self.e1 = np.array([self.C * cell[1], 0.0, 0.0])
        self.e2 = np.array([0.0, 0.0, -self.R * cell[0]])
        self._mips = {}

    def box(self):
        p = np.array([self.origin, self.origin + self.e1, self.origin + self.e2,
                      self.origin + self.e1 + self.e2])
        return p.min(0), p.max(0)

    def mip(self, lr, lc, tex=None):
        """The texture averaged over ``2^lr`` rows and ``2^lc`` columns."""
        own = tex is None
        if own and (lr, lc) in self._mips:
            return self._mips[lr, lc]
        t = self.tex if own else tex
        for axis, lev in ((0, lr), (1, lc)):
            for _ in range(lev):
                n = t.shape[axis]
                if n < 2:
                    break
                if n % 2:
                    t = np.concatenate([t, np.take(t, [-1], axis)], axis)
                t = (np.take(t, range(0, t.shape[axis], 2), axis) +
                     np.take(t, range(1, t.shape[axis], 2), axis)) / 2
        if own:
            self._mips[lr, lc] = t
        return t

    def draw(self, anim, sc, opacity=1.0, highlight=None, tex=None, alpha=None,
             e1=None, e2=None, origin=None, lift=0.0, glow=0.0):
        """``highlight`` (scalar or ``(R, C)``) blends towards white; ``alpha``
        ``(R, C)`` per-cell opacity; ``tex`` replaces the colours this frame."""
        if opacity <= 0.01:
            return
        cam = anim.cam
        p0 = anim.Z(self.origin if origin is None else origin) + cam.view * lift
        a1 = (self.e1 if e1 is None else e1) * anim.s
        a2 = (self.e2 if e2 is None else e2) * anim.s
        sa, sb = cam.M @ a1, cam.M @ a2
        if abs(sa[0] * sb[1] - sa[1] * sb[0]) < 0.25:          # edge-on or sub-pixel
            return
        s0 = cam.project(p0)
        pts = np.array([s0, s0 + sa, s0 + sb, s0 + sa + sb])
        if pts[:, 0].max() < -4 or pts[:, 0].min() > cam.width + 4 or \
                pts[:, 1].max() < -4 or pts[:, 1].min() > cam.height + 4:
            return
        n = np.cross(a1, a2)
        d1 = np.cross(a2, n) / np.dot(a1, np.cross(a2, n))
        d2 = np.cross(n, a1) / np.dot(a2, np.cross(n, a1))
        R, Cn = self.R, self.C
        hl = highlight
        own = tex is None

        def shader(P):
            d = P - p0
            u = np.clip(d @ d1, 0.0, 1 - 1e-9)
            v = np.clip(d @ d2, 0.0, 1 - 1e-9)
            # texels per pixel along each axis -> mip level per axis
            if u.shape[0] > 1 and u.shape[1] > 1:
                fu = Cn * math.hypot(u[0, 1] - u[0, 0], u[1, 0] - u[0, 0])
                fv = R * math.hypot(v[0, 1] - v[0, 0], v[1, 0] - v[0, 0])
            else:
                fu = fv = 0.0
            lc = int(math.log2(fu)) if fu > 1.5 else 0
            lr = int(math.log2(fv)) if fv > 1.5 else 0
            t = self.mip(lr, lc, None if own else tex)
            tr, tc = t.shape[:2]
            row = np.minimum((v * tr).astype(np.int32), tr - 1)
            col = np.minimum((u * tc).astype(np.int32), tc - 1)
            rgb = t[row, col]
            a = np.full(u.shape, opacity, np.float32)
            if fu > 0 and fu < 1 / 9 and fv < 1 / 9:           # big cells: show the grid
                gu, gv = (u * Cn) % 1.0, (v * R) % 1.0
                g = 0.045
                edge = (gu < g) | (gu > 1 - g) | (gv < g) | (gv > 1 - g)
                rgb = rgb * np.where(edge, 0.86, 1.0)[..., None].astype(np.float32)
            ri = np.minimum((v * R).astype(np.int32), R - 1)
            ci = np.minimum((u * Cn).astype(np.int32), Cn - 1)
            if hl is not None:
                h = hl[ri, ci] if np.ndim(hl) else np.float32(hl)
                rgb = rgb + (1.0 - rgb) * (0.7 * np.asarray(h, np.float32))[..., None]
            if alpha is not None:
                a = a * alpha[ri, ci]
            if glow > 0:
                return rgb.astype(np.float32), a, np.full(u.shape, glow, np.float32)
            return rgb.astype(np.float32), a

        sc.quad(p0, a1, a2, shader)


def rotate(v, axis, theta):
    """Rodrigues: rotate ``v`` about ``axis`` by ``theta``."""
    k = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + \
        k * np.dot(k, v) * (1 - math.cos(theta))


DIAG = np.array([1.0, 0.0, -1.0])          # flipping about this swaps rows and columns


# ------------------------------------------------------------------ the animation


class GPT2Animation(NetAnimation):
    PHASES = ("tokens", "Q K V", "flip", "scores", "softmax", "A·V", "heads", "layers",
              "output")

    def __init__(self, text="One, two, three,"):
        super().__init__("Inside GPT-2", "", 10.0, azimuth=8.0, elevation=16.0)
        global TP
        tr = self.tr = G.run(text)
        T = self.T = len(tr.ids)
        if not 2 <= T <= MAX_TOKENS:
            raise SystemExit("gpt2-attention: the prompt must be 2 to %d tokens, not %d"
                             % (MAX_TOKENS, T))
        TP = HT / T              # the layout is built for HT = T * TP
        W = G.weights()
        L, H = LAYER, HEAD
        self.hold, self.reset = 3.0, 3.5
        self.floor_z = -1e6
        self.FX = self.Mx(G.N_LAYER)

        # ---- residual stream and embeddings ---------------------------------------
        self.tok_e = Panel(diverging(tr.emb), (EMBED_X, 0, EMB_TOK))
        self.pos_e = Panel(diverging(tr.pos), (EMBED_X, 0, EMB_POS))
        self.x0 = Panel(diverging(tr.resid[0]), (EMBED_X, 0, 0))
        self.seg_in, self.seg_mid, self.seg_out = [], [], []
        self.concat, self.attn_o, self.mlp_h, self.mlp_o = [], [], [], []
        self.cards = []
        for l in range(G.N_LAYER):
            m = self.Mx(l)
            self.seg_in.append(Panel(diverging(tr.resid[l]), (m, 0, 0)))
            self.seg_mid.append(Panel(diverging(tr.resid[l] + tr.attn_out[l]), (m + 1100, 0, 0)))
            self.seg_out.append(Panel(diverging(tr.resid[l + 1]), (m + 2050, 0, 0)))
            cat = tr.head_out[l].transpose(1, 0, 2).reshape(T, -1)
            self.concat.append(Panel(diverging(cat), (m + 1100, 0, 400)))
            self.attn_o.append(Panel(diverging(tr.attn_out[l]), (m + 1100, 0, 230)))
            self.mlp_h.append(Panel(lit(tr.mlp_hidden[l]), (m + 2050, 0, 400), cell=(TP, 0.25)))
            self.mlp_o.append(Panel(diverging(tr.mlp_out[l]), (m + 2050, 0, 230)))
            self.cards.append([self._card(l, h) for h in range(G.N_HEAD)])
        final = G._ln(tr.resid[-1], W["ln_f.weight"], W["ln_f.bias"])
        self.lnf = Panel(diverging(final), (self.FX, 0, 0))
        lg = tr.logits[-1]
        self.logits = Panel(diverging(lg[None] - lg.mean()), (self.FX, 0, 230), cell=(TP, 768.0 / len(lg)))

        # ---- the dive: layer 5, head 12 ---------------------------------------------------------
        cx, cz = self.card_origin(L, H)
        self.cq = np.array([cx, 0.0, cz])
        self.ln = Panel(diverging(tr.ln1[L]), (cx - 6 - 768 * HALF, 0, cz + 96),
                        cell=(TP, HALF))
        wq = W["h.%d.attn.c_attn.weight" % L][:, H * 64:(H + 1) * 64]
        self.wq = Panel(diverging(wq), (cx, 0, cz + 96 + 6 + 768 * HALF), cell=(HALF, 1.0))
        self.k_plain = Panel(diverging(tr.k[L, H]), (cx + KT_AT[0], 0, cz + KT_AT[1]))

        # ---- timeline -----------------------------------------------------------------------------------
        Tm = self.Tm = dict(
            labels=(0.2, 1.0), embed=(1.0, 3.6), fly=(4.4, 8.6), proj=(8.8, 12.4),
            kv=(12.4, 13.8), close=(13.6, 15.0), transpose=(15.0, 17.4), qk=(17.8, 29.8),
            mask=(29.8, 31.2), softmax=(31.4, 34.0), av=(34.2, 38.6), heads=(39.2, 42.6),
            concat=(42.8, 45.2), wo=(45.2, 46.6), add1=(46.6, 47.8), mlp=(47.8, 50.4),
            wide=(50.8, 53.4), flow=(53.4, 69.4), full=(69.4, 70.6), out=(71.4, 74.0),
            predict=(74.0, 77.6))
        self.t_done = 78.0
        self.duration = self.t_done + self.hold + self.reset
        cells = [(i, j) for i in range(T) for j in range(T)]
        slow = [(1, 0), (1, 1), (2, 1)]
        rest = [c for c in cells if c not in slow]
        a, b = Tm["qk"]
        t, self.qk_cells = a, []
        for c in slow:
            self.qk_cells.append((c, t, 2.7, True))
            t += 2.8
        dt = (b - t) / len(rest)
        for c in rest:
            self.qk_cells.append((c, t, dt, False))
            t += dt

        # ---- camera keyframes: (time, focus, span: model units across 86% of the width) --------
        card_c = self.cq + np.array([CARD_W / 2 - 30, 0, CARD_H / 2 + 10])
        proj_c = self.cq + np.array([-163.0, 0, 240.0])
        layer_c = np.array([self.Mx(L) + 1350, 0, 380.0])
        full_c = np.array([(EMBED_X + self.FX + 1500) / 2, 0, 380.0])
        full_s = self.FX + 1500 - EMBED_X + 600
        out_c = np.array([self.FX + 1050, 0, 210.0])
        open_c = np.array([EMBED_X + 300, 0, 135.0])
        tok_c = np.array([EMBED_X + 190, 0, EMB_TOK - 40.0])
        mid_c = (open_c + proj_c) / 2 + np.array([0, 0, 300.0])
        hold_end = self.t_done + self.hold
        self.keys = [
            (0.0, tok_c, 500.0), (2.2, tok_c, 500.0), (3.2, open_c, 850.0),
            (Tm["fly"][0], open_c, 850.0),
            (6.4, mid_c, 9000.0), (Tm["fly"][1], proj_c, 1150.0),
            (Tm["kv"][1] - 0.2, proj_c, 1150.0), (Tm["close"][1], card_c, 470.0),
            (Tm["heads"][0], card_c, 470.0), (Tm["heads"][1], layer_c, 2950.0),
            (Tm["wide"][0], layer_c, 2950.0), (Tm["wide"][1], self._track, 9000.0),
            (Tm["flow"][1], self._track, 9000.0), (Tm["full"][1], full_c, full_s),
            (Tm["out"][0], full_c, full_s), (Tm["out"][1] - 0.4, out_c, 2400.0),
            (hold_end, out_c, 2400.0), (hold_end + 1.7, full_c, full_s),
            (self.duration, tok_c, 500.0),
        ]

    # ---------------------------------------------------------------- layout
    @staticmethod
    def Mx(l):
        return l * MOD

    def card_origin(self, l, h):
        r, c = divmod(h, 4)
        return self.Mx(l) - 124 + c * GRID_COL, 150.0 + (2 - r) * GRID_ROW

    def _card(self, l, h):
        tr = self.tr
        cx, cz = self.card_origin(l, h)
        at = lambda p: (cx + p[0], 0.0, cz + p[1])           # noqa: E731
        A = lit(tr.attn[l, h], 1.0)
        return dict(Q=Panel(diverging(tr.q[l, h]), at(Q_AT)),
                    KT=Panel(diverging(tr.k[l, h].T), at(KT_AT), cell=(1.0, TP)),
                    S=Panel(diverging(tr.scores[l, h]), at(S_AT), cell=(TP, TP)),
                    A=Panel(A, at(S_AT), cell=(TP, TP)),
                    V=Panel(diverging(tr.v[l, h]), at(V_AT)),
                    O=Panel(diverging(tr.head_out[l, h]), at(O_AT)))

    # ---------------------------------------------------------------- camera
    def pulse_x(self, t):
        """Where the information pulse is on the stream while it flows."""
        u = self.p_("flow", t)
        return EMBED_X - 600 + u * (self.FX + 200 - EMBED_X + 600)

    def _track(self, t):
        """The tracking shot follows the pulse through the layers."""
        return np.array([self.pulse_x(t) + 800, 0, 380.0])

    def camera_at(self, t):
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1], keys[0][2]
        for (t0, f0, s0), (t1, f1, s1) in zip(keys, keys[1:]):
            if t0 <= t <= t1:
                f0 = f0(t) if callable(f0) else f0
                f1 = f1(t) if callable(f1) else f1
                u = ease_in_out(remap(t, t0, t1))
                span = math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * u)
                # the focus moves in step with the view size, so the target
                # stays in frame while zooming (a "zoom-pan")
                w = (span - s0) / (s1 - s0) if abs(s1 - s0) > 1.0 else u
                return f0 + (f1 - f0) * w, span
        return keys[-1][1], keys[-1][2]

    def set_view(self, t):
        f, span = self.camera_at(t)
        px_per_unit = self.cam.M[0, 0]
        self.s = 0.86 * self.cam.width / (span * px_per_unit)
        self.focus = np.asarray(f, np.float64)
        self.anchor = self.cam.frac(0.5, 0.58)

    def Z(self, p):
        return self.anchor + (np.asarray(p, np.float64) - self.focus) * self.s

    def zbox(self, panel_or_lo, hi=None):
        lo, hi = panel_or_lo.box() if hi is None else (panel_or_lo, hi)
        return self.Z(lo), self.Z(hi)

    def px(self, units):
        """Screen pixels (at 960 wide) of a model length."""
        return units * self.s * self.cam.M[0, 0]

    def at_px(self, p):
        q = self.cam.project(self.Z(p))
        return q[0] / self.cam.width, q[1] / self.cam.height

    # ---------------------------------------------------------------- frame
    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def step_of(self, t):
        Tm = self.Tm
        order = [("labels", 0), ("fly", 1), ("transpose", 2), ("qk", 3), ("softmax", 4),
                 ("av", 5), ("heads", 6), ("wide", 7), ("out", 8)]
        cur = 0
        for name, k in order:
            if t >= Tm[name][0]:
                cur = k
        return cur

    def draw(self, sc, t, reset):
        self.set_view(t)
        self._cap *= 1.0 - smoothstep(0.0, 0.12, reset)     # text leaves before the camera
        back = 1.0 - smoothstep(0.55, 1.0, reset)
        self.back = back
        ph = 0 if reset > 0.5 else self.step_of(t)
        self.subtitle = ""
        Tm = self.Tm
        # probes: the camera move drives motion blur
        sc.probe(("cam", 0), self.Z((0.0, 0.0, 0.0)))
        sc.probe(("cam", 1), self.Z((self.FX, 0.0, 800.0)))
        sc.probe(("cam", 2), self.Z((EMBED_X, 0.0, 0.0)))

        self._embedding(sc, t, back)
        # the whole model is there during the fly-in and from the zoom-out on;
        # around the dive only the head's own layer stream stays
        model = max(1 - smoothstep(Tm["fly"][1] - 1.6, Tm["fly"][1] - 0.4, t),
                    smoothstep(*Tm["heads"], t)) * smoothstep(Tm["fly"][0], Tm["fly"][0] + 1, t)
        model *= back
        flow = self.p_("flow", t)
        pulse_x = self.pulse_x(t) if 0 < flow < 1 else None
        for l in range(G.N_LAYER):
            if l == LAYER:
                self._dive_layer(sc, t, back, model, pulse_x)
            else:
                self._layer(sc, l, model, pulse_x)
        self._stream_lines(sc, model, 1.0 if t < Tm["fly"][1] else
                           smoothstep(Tm["mlp"][1] - 0.3, Tm["mlp"][1], t))
        self._output(sc, t, back, model, pulse_x)
        if pulse_x is not None:
            self._pulse(sc, pulse_x, back)
        self._captions(sc, t, back, model)
        if not self.subtitle:
            self.subtitle = self._subtitle(t)
        if reset > 0.3:                       # the loop restarts where it began
            self.subtitle = self._subtitle(0.0)
        self._keep_in_frame(sc)
        self.step_indicator(sc, ph, self.PHASES)

    def _keep_in_frame(self, sc):
        """The camera pans past labelled things: a label whose object is partly
        out of shot slides to stay inside the title-safe area, and fades as
        its object leaves (it never piles up at the edge)."""
        W, H = self.cam.width, self.cam.height
        lo_x, hi_x = W * layout.SAFE, W * (1 - layout.SAFE)
        lo_y, hi_y = H * layout.SAFE, H * (1 - layout.SAFE)
        keep = []
        for tx in sc.texts:
            x0, y0, x1, y1 = layout.text_box(tx)
            w, h = x1 - x0, y1 - y0
            dx = max(0.0, lo_x + 1 - x0) - max(0.0, x1 - hi_x + 1)
            dy = max(0.0, lo_y + 1 - y0) - max(0.0, y1 - hi_y + 1)
            if dx or dy:
                over = max(abs(dx) / max(w, 1.0), abs(dy) / max(h, 1.0))
                tx.opacity *= float(np.clip(1.0 - (over - 0.25) / 0.35, 0.0, 1.0))
                tx.x += dx / W
                tx.y += dy / H
            if tx.opacity > 0.004:
                keep.append(tx)
        sc.texts[:] = keep

    # -- pieces --------------------------------------------------------------------------------------------
    def _hl(self, panel, pulse_x):
        if pulse_x is None:
            return None
        lo, hi = panel.box()
        return 0.55 * bump((lo[0] + hi[0]) / 2 - pulse_x, 900.0)

    def _layer(self, sc, l, op, pulse_x):
        if op <= 0.01:
            return
        for h in range(G.N_HEAD):
            self._card_draw(sc, l, h, op, pulse_x)
        for P in (self.seg_in[l], self.seg_mid[l], self.seg_out[l], self.concat[l],
                  self.attn_o[l], self.mlp_h[l], self.mlp_o[l]):
            P.draw(self, sc, op, highlight=self._hl(P, pulse_x))

    def _card_draw(self, sc, l, h, op, pulse_x=None):
        c = self.cards[l][h]
        for k in ("Q", "KT", "A", "V", "O"):
            c[k].draw(self, sc, op, highlight=self._hl(c[k], pulse_x))

    def _stream_lines(self, sc, op, dive_op=1.0):
        """Thin connectors: the stream between its segments, and each layer's
        branches (stream -> heads -> concat -> W_O -> add; stream -> MLP -> add)."""
        if op <= 0.01:
            return
        ink = (0.62, 0.64, 0.66)
        mid = -TP * self.T / 2
        Z = self.Z
        xs = [EMBED_X + 768]
        for l in range(G.N_LAYER):
            m = self.Mx(l)
            for a, b in ((m - 52, m), (m + 768, m + 1100), (m + 1868, m + 2050)):
                sc.line(Z((a, 0, mid)), Z((b, 0, mid)), 1.2, ink, op * 0.8)
            # branches
            op_l = op
            op = op * (dive_op if l == LAYER else 1.0)
            sc.line(Z((m + 384, 0, 0)), Z((m + 384, 0, 150)), 1.0, ink, op * 0.6)
            sc.line(Z((m + 892, 0, 500)), Z((m + 1100, 0, 400 - 3 * TP)), 1.0, ink, op * 0.6)
            sc.line(Z((m + 1160, 0, 400 - TP * self.T)), Z((m + 1160, 0, 230)), 1.0, ink, op * .6)
            sc.line(Z((m + 1160, 0, 230 - TP * self.T)), Z((m + 1160, 0, 0)), 1.0, ink, op * .6)
            sc.line(Z((m + 1868, 0, mid)), Z((m + 1950, 0, mid)), 1.0, ink, op * 0.6)
            sc.line(Z((m + 1950, 0, mid)), Z((m + 1950, 0, 400 - 3 * TP)), 1.0, ink, op * 0.6)
            sc.line(Z((m + 1950, 0, 400 - 3 * TP)), Z((m + 2050, 0, 400 - 3 * TP)), 1.0, ink,
                    op * 0.6)
            sc.line(Z((m + 2110, 0, 400 - TP * self.T)), Z((m + 2110, 0, 230)), 1.0, ink, op * .6)
            sc.line(Z((m + 2110, 0, 230 - TP * self.T)), Z((m + 2110, 0, 0)), 1.0, ink, op * 0.6)
            xs.append(m + 2818)
            op = op_l
        sc.line(Z((EMBED_X + 768, 0, mid)), Z((-52, 0, mid)), 1.2, ink, op * 0.8)
        sc.line(Z((self.FX - 232, 0, mid)), Z((self.FX, 0, mid)), 1.2, ink, op * 0.8)
        for l in range(G.N_LAYER - 1):
            m = self.Mx(l)
            sc.line(Z((m + 2818, 0, mid)), Z((self.Mx(l + 1) - 52, 0, mid)), 1.2, ink, op * 0.8)

    def _pulse(self, sc, x, back):
        h = TP * self.T
        w = max(160.0, 26.0 / max(self.px(1.0), 1e-9))
        lo = self.Z((x - w / 2, 0, 30)) + self.cam.view * 50
        a1 = np.array([w, 0, 0]) * self.s
        a2 = np.array([0, 0, -(h + 60)]) * self.s

        def shader(P):
            u = np.clip((P - lo) @ (a1 / (a1 @ a1)), 0, 1)
            a = (np.sin(np.pi * u) ** 2).astype(np.float32)
            return (np.broadcast_to(np.float32(ACT_HI), P.shape).copy(), 0.95 * back * a,
                    1.2 * a)

        sc.quad(lo, a1, a2, shader)
        sc.probe(("pulse",), self.Z((x, 0, 0)), tol=3.0)

    # -- tokens and embeddings ---------------------------------------------------------------------------
    def _embedding(self, sc, t, back):
        Tm = self.Tm
        e = self.p_("embed", t)
        on = back * smoothstep(0.0, 0.15, e)
        # each token's embedding row is looked up in turn
        rows = np.arange(self.T)[:, None] * np.ones((1, 768))
        got = (e * 1.6 * self.T > rows).astype(np.float32)
        self.tok_e.draw(self, sc, on, alpha=0.15 + 0.85 * got)
        self.pos_e.draw(self, sc, on * smoothstep(0.45, 0.6, e))
        x = smoothstep(0.65, 0.95, e)
        if 0 < x < 1:                                          # both slide down into x
            for P in (self.tok_e, self.pos_e):
                o = P.origin + (self.x0.origin - P.origin) * ease_in_out(x)
                P.draw(self, sc, on * 0.7 * (1 - x), origin=o, lift=40)
        self.x0.draw(self, sc, back * smoothstep(0.85, 1.0, e))
        # token labels beside the rows they pick
        op = smoothstep(*Tm["labels"], t) * back * self._cap
        op *= 1 - smoothstep(Tm["fly"][0], Tm["fly"][0] + 0.8, t) * (1 - smoothstep(
            self.t_done + self.hold + 1.7, self.duration - 0.6, t))
        op *= smoothstep(19.5, 21.0, self.px(TP) * 0.815)        # row height in px
        if op > 0.01:
            for i, tok in enumerate(self.tr.tokens):
                fx, fy = self.at_px((EMBED_X - 14, 0, EMB_TOK - (i + 0.5) * TP))
                sc.text(fx, fy, '"%s"' % tok, layout.MIN_TEXT, "mono", layout.CAPTION_INK, op,
                        "rm")

    # -- the dive layer ------------------------------------------------------------------------------------
    def _dive_layer(self, sc, t, back, model, pulse_x):
        L = LAYER
        self.seg_in[L].draw(self, sc, model, highlight=self._hl(self.seg_in[L], pulse_x))
        for h in range(G.N_HEAD):
            if h != HEAD:
                self._card_draw(sc, L, h, model, pulse_x)
        self._dive(sc, t, back, pulse_x)
        self._after_heads(sc, t, back, model, pulse_x)

    def _dive(self, sc, t, back, pulse_x):
        L, H, T = LAYER, HEAD, self.T
        Tm = self.Tm
        card = self.cards[L][H]
        op = back * max(smoothstep(Tm["fly"][1] - 1.2, Tm["fly"][1], t),
                        1 - smoothstep(Tm["fly"][0], Tm["fly"][0] + 1.0, t))
        if op <= 0.01:
            return
        full = t >= Tm["av"][1] or t < Tm["fly"][1] - 1.2
        if full:
            self._card_draw(sc, L, H, op, pulse_x)
            return
        empty = lambda P: (np.full((P.R, P.C), 0.15, np.float32))      # noqa: E731
        # ---- Q = LN(x) W_Q, one row at a time
        pr = self.p_("proj", t)
        lw = op * smoothstep(Tm["fly"][1] - 0.6, Tm["fly"][1], t) * \
            (1 - smoothstep(Tm["kv"][1] - 0.6, Tm["kv"][1], t))
        rowq = min(T - 1, int(pr * T)) if 0 < pr < 1 else None
        hl_ln = None
        if rowq is not None:
            hl_ln = np.zeros((T, 768), np.float32)
            hl_ln[rowq] = 0.8
        self.ln.draw(self, sc, lw, highlight=hl_ln)
        self.wq.draw(self, sc, lw, highlight=0.25 if rowq is not None else None)
        q_done = np.clip(pr * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 64))
        al = np.where(q_done >= 1, 1.0, 0.15 + 0.85 * q_done).astype(np.float32)
        tex = np.where(q_done[..., None] > 0, card["Q"].tex, OFF).astype(np.float32)
        card["Q"].draw(self, sc, op, tex=tex if pr < 1 else None, alpha=al if pr < 1 else None,
                       highlight=self._row_hl(T, 64, rowq, 0.5) if rowq is not None else
                       self._qk_hl(t, "Q"))
        if rowq is not None:          # a row of LN(x) streams up the columns of W_Q
            u = (pr * T) % 1.0
            sc.probe(("proj",), (u * 60, 0, 0), tol=6.0)
        # ---- K and V
        kv = self.p_("kv", t)
        tp = self.p_("transpose", t)
        kd = np.clip(kv * 1.4 * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 64))
        vd = np.clip((kv - 0.25) * 1.4 * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 64))
        kv_on = op * smoothstep(0.0, 0.1, kv)
        if kv > 0 and tp <= 0:
            self.k_plain.draw(self, sc, kv_on, alpha=(0.15 + 0.85 * kd).astype(np.float32),
                              tex=np.where(kd[..., None] > 0, self.k_plain.tex, OFF)
                              .astype(np.float32))
        if kv > 0:
            card["V"].draw(self, sc, kv_on, alpha=(0.15 + 0.85 * vd).astype(np.float32),
                           tex=np.where(vd[..., None] > 0, card["V"].tex, OFF).astype(np.float32),
                           highlight=self._av_hl(t))
        # ---- the transpose: K flips over its diagonal, in place
        if tp > 0:
            th = math.pi * ease_in_out(remap(tp, 0.1, 0.8))
            e1 = rotate(self.k_plain.e1, DIAG, th)
            e2 = rotate(self.k_plain.e2, DIAG, th)
            if th < math.pi - 1e-6:
                self.k_plain.draw(self, sc, op, e1=e1, e2=e2, lift=0.0)
                sc.probe(("flip",), self.Z(self.k_plain.origin + e1 + e2), tol=4.0)
            else:
                card["KT"].draw(self, sc, op, highlight=self._qk_hl(t, "KT"))
        # ---- scores, softmax
        if t >= Tm["qk"][0]:
            self._scores(sc, t, card, op)
        elif t >= Tm["kv"][0]:
            card["S"].draw(self, sc, op * 0.6, tex=np.full((T, T, 3), OFF, np.float32),
                           alpha=empty(card["S"]))
        # ---- A V
        av = self.p_("av", t)
        o_done = np.clip(av * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 64))
        if t >= Tm["kv"][0]:
            card["O"].draw(self, sc, op, tex=np.where(o_done[..., None] > 0, card["O"].tex, OFF)
                           .astype(np.float32), alpha=(0.15 + 0.85 * o_done).astype(np.float32),
                           highlight=self._row_hl(T, 64, min(T - 1, int(av * T)), 0.5)
                           if 0 < av < 1 else None)

    @staticmethod
    def _row_hl(R, Cn, i, k):
        h = np.zeros((R, Cn), np.float32)
        h[i] = k
        return h

    def _current_cell(self, t):
        for (i, j), t0, d, slow in self.qk_cells:
            if t0 <= t < t0 + d:
                return (i, j), (t - t0) / d, slow
        return None, 0.0, False

    def _qk_hl(self, t, which):
        """Highlight the row of Q / column of K^T being multiplied."""
        if not self.Tm["qk"][0] <= t < self.Tm["qk"][1]:
            return None
        (cell, u, slow) = self._current_cell(t)
        if cell is None:
            return None
        i, j = cell
        k = 0.55 * smoothstep(0.0, 0.08, u) if slow else 0.45
        if which == "Q":
            return self._row_hl(self.T, 64, i, k)
        h = np.zeros((64, self.T), np.float32)
        h[:, j] = k
        return h

    def _av_hl(self, t):
        av = self.p_("av", t)
        if not 0 < av < 1:
            return None
        i = min(self.T - 1, int(av * self.T))
        w = self.tr.attn[LAYER, HEAD, i]
        return (np.repeat(w[:, None], 64, 1) * 0.7).astype(np.float32)

    def _scores(self, sc, t, card, op):
        tr, L, H, T = self.tr, LAYER, HEAD, self.T
        Tm = self.Tm
        S = tr.scores[L, H]
        masked = np.triu(np.ones((T, T), bool), 1)
        sm = self.p_("softmax", t)
        mk = self.p_("mask", t)
        Sc, Ac = card["S"].tex, card["A"].tex
        if sm > 0:
            m = np.clip(sm * 1.25 * T - np.arange(T), 0, 1)[:, None, None] * np.ones((1, T, 1))
            col = Sc + (Ac - Sc) * m
            col[masked] = OFF
            i = min(T - 1, int(sm * 1.25 * T))
            hl = self._row_hl(T, T, i, 0.4) if sm < 0.8 else None
            card["S"].draw(self, sc, op, tex=col.astype(np.float32), highlight=hl)
            if sm >= 1 and t < Tm["av"][1]:
                av = self.p_("av", t)
                if 0 < av < 1:
                    hl = self._row_hl(T, T, min(T - 1, int(av * T)), 0.5)
                    card["A"].draw(self, sc, op, highlight=hl, lift=0.5)
            return
        done = np.zeros((T, T), bool)
        cur, u, slow = self._current_cell(t)
        for (i, j), t0, d, _s in self.qk_cells:
            done[i, j] = t >= t0 + d
        if t >= Tm["qk"][1]:
            done[:] = True
        col = np.where(done[..., None], Sc, OFF).astype(np.float32)
        if mk > 0:
            k = smoothstep(0.0, 0.7, mk)
            col[masked] = col[masked] + (np.float32(OFF) - col[masked]) * k
        al = np.where(done, 1.0, 0.15).astype(np.float32)
        hl = None
        if mk > 0 and mk < 1:
            hl = (masked * 0.5 * bump(mk - 0.35, 0.3)).astype(np.float32)
        card["S"].draw(self, sc, op, tex=col, alpha=al, highlight=hl)
        if cur is None:
            return
        i, j = cur
        if not slow:
            return
        # the slow cells: row i of Q turns (a transpose of one row) to stand beside
        # column j of K^T; element-wise products; they sum into one score
        q = card["Q"]
        kt = card["KT"]
        cx, cz = self.cq[0], self.cq[2]
        row = Panel(q.tex[i:i + 1], (cx, 0, cz + 96 - i * TP))
        go = ease_in_out(remap(u, 0.08, 0.38))
        th = math.pi * go
        dest_q = np.array([cx + 70 - 2 * TP - 6, 0, cz + 166])        # beside K^T's left edge
        dest_k = dest_q + np.array([TP + 2, 0, 0])
        kcol = Panel(kt.tex[:, j:j + 1], (cx + 70 + j * TP, 0, cz + 166), cell=(1.0, TP))
        mrg = ease_in_out(remap(u, 0.42, 0.55))
        mid = (dest_q + dest_k) / 2
        if u < 0.55:
            o = row.origin + (dest_q - row.origin) * go
            o = o + (mid - dest_q) * mrg
            row.draw(self, sc, op, e1=rotate(row.e1, DIAG, th), e2=rotate(row.e2, DIAG, th),
                     origin=o, lift=30, highlight=0.15)
            ok = kcol.origin + (dest_k - kcol.origin) * go + (mid - dest_k) * mrg
            kcol.draw(self, sc, op, origin=ok, lift=30, highlight=0.15)
            sc.probe(("qrow",), self.Z(o), tol=4.0)
            sc.probe(("kcol",), self.Z(ok), tol=4.0)
        else:
            prod = tr.q[L, H, i] * tr.k[L, H, j]
            pp = Panel(diverging(prod[:, None]), mid, cell=(1.0, TP))
            sh = ease_in_out(remap(u, 0.68, 0.92))
            tgt = np.array([cx + S_AT[0] + j * TP, 0, cz + S_AT[1] - i * TP])
            o = mid + (tgt - mid) * sh
            e2 = np.array([0, 0, -(64 * (1 - sh) + TP * sh)])
            tex = pp.tex if sh < 0.6 else np.broadcast_to(Sc[i, j], pp.tex.shape)
            pp.draw(self, sc, op, origin=o, e2=e2, lift=30, tex=np.ascontiguousarray(tex))
            sc.probe(("prod",), self.Z(o + e2), tol=4.0)
        raw = S[i, j] * 8.0
        if u < 0.42:
            self.subtitle = ("row %d of Q turns to lie beside column %d of K^T (a transpose of "
                             "one row)" % (i + 1, j + 1))
        elif u < 0.66:
            self.subtitle = ("multiply element by element: 64 products q%d[d] · k%d[d]"
                             % (i + 1, j + 1))
        else:
            self.subtitle = ("sum the 64 products: q%d · k%d = %.1f;  ÷ √64 = "
                             "%.2f  =  score[%d, %d]" % (i + 1, j + 1, raw, S[i, j], i + 1, j + 1))

    # -- concat, W_O, residual add, MLP for the dive layer ------------------------------------------
    def _after_heads(self, sc, t, back, model, pulse_x):
        L, Tm = LAYER, self.Tm
        m = self.Mx(L)
        T = self.T
        if t < Tm["concat"][0] or back <= 0.01:
            # before the layer view: show nothing of the rest of this layer
            # (during the fly-in the full model is visible: draw it complete)
            if model > 0.01:
                for P in (self.concat[L], self.attn_o[L], self.mlp_h[L], self.mlp_o[L],
                          self.seg_mid[L], self.seg_out[L]):
                    P.draw(self, sc, model * (t < Tm["fly"][1]))
            return
        ct = self.p_("concat", t)
        # the 12 head outputs fly to their slots in the concatenation
        arrived = np.zeros((T, 768), np.float32)
        for h in range(G.N_HEAD):
            u = ease_in_out(remap(ct, h * 0.04, h * 0.04 + 0.5))
            if u >= 1:
                arrived[:, h * 64:(h + 1) * 64] = 1
            elif u > 0:
                O = self.cards[L][h]["O"]
                o = O.origin + (np.array([m + 1100 + h * 64, 0, 400]) - O.origin) * u
                O.draw(self, sc, back, origin=o, lift=40 * math.sin(math.pi * u))
                if h == 0:
                    sc.probe(("cat",), self.Z(o), tol=4.0)
        hl = lambda P: self._hl(P, pulse_x)                  # noqa: E731
        self.concat[L].draw(self, sc, back, alpha=np.maximum(arrived, 0.15) if ct < 1 else None,
                            tex=None, highlight=hl(self.concat[L]))
        wo = self.p_("wo", t)
        a1 = self.p_("add1", t)
        ml = self.p_("mlp", t)
        rows = np.clip(wo * T * 1.2 - np.arange(T), 0, 1)[:, None] * np.ones((1, 768))
        self.attn_o[L].draw(self, sc, back * (wo > 0),
                            alpha=(0.15 + 0.85 * rows).astype(np.float32) if wo < 1 else None,
                            highlight=hl(self.attn_o[L]))
        if 0 < a1 < 1:                       # x + attention(x): the add
            P = self.attn_o[L]
            o = P.origin + (self.seg_mid[L].origin - P.origin) * ease_in_out(a1)
            P.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a1)), origin=o, lift=40)
            Pi = self.seg_in[L]
            o2 = Pi.origin + (self.seg_mid[L].origin - Pi.origin) * ease_in_out(a1)
            Pi.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a1)), origin=o2, lift=30)
            sc.probe(("add1",), self.Z(o2), tol=4.0)
        self.seg_mid[L].draw(self, sc, back * smoothstep(0.8, 1.0, a1), highlight=hl(
            self.seg_mid[L]))
        hr = np.clip(ml * 2.2 * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 3072))
        mr = np.clip((ml - 0.35) * 2.2 * T - np.arange(T), 0, 1)[:, None] * np.ones((1, 768))
        self.mlp_h[L].draw(self, sc, back * (ml > 0),
                           alpha=(0.15 + 0.85 * hr).astype(np.float32) if ml < 1 else None,
                           highlight=hl(self.mlp_h[L]))
        self.mlp_o[L].draw(self, sc, back * (ml > 0.35),
                           alpha=(0.15 + 0.85 * mr).astype(np.float32) if ml < 1 else None,
                           highlight=hl(self.mlp_o[L]))
        a2 = remap(ml, 0.8, 1.0)
        if 0 < a2 < 1:
            for P, lift in ((self.mlp_o[L], 40), (self.seg_mid[L], 30)):
                o = P.origin + (self.seg_out[L].origin - P.origin) * ease_in_out(a2)
                P.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a2)), origin=o, lift=lift)
        self.seg_out[L].draw(self, sc, back * smoothstep(0.8, 1.0, a2), highlight=hl(
            self.seg_out[L]))

    # -- output ----------------------------------------------------------------------------------------
    def _output(self, sc, t, back, model, pulse_x):
        tr = self.tr
        on = max(model * (t < self.Tm["fly"][1]), smoothstep(*self.Tm["wide"], t)) * back
        self.lnf.draw(self, sc, on, highlight=self._hl(self.lnf, pulse_x))
        pr = self.p_("predict", t)
        self.logits.draw(self, sc, back * smoothstep(0.0, 0.15, pr))
        if pr <= 0:
            return
        x0 = self.FX + 950
        for n, (tok, p) in enumerate(tr.top):
            g = ease_out(remap(pr, 0.15 + 0.08 * n, 0.55 + 0.08 * n))
            h = max(4.0, 700.0 * p * g)
            col = np.float32(ACT_HI if n == 0 else ACT_LO)
            lo = self.Z((x0 + n * 230, 0, 0))
            a1 = np.array([150.0, 0, 0]) * self.s
            a2 = np.array([0, 0, h]) * self.s
            glow = 0.6 if n == 0 else 0.0
            sc.quad(lo, a1, a2, lambda P, c=col, gl=glow: (
                np.broadcast_to(c, P.shape).copy(), np.full(P.shape[:2], back, np.float32),
                np.full(P.shape[:2], gl, np.float32)))
            op = back * self._cap * smoothstep(0.15, 0.3, pr)
            fx, fy = self.at_px((x0 + n * 230 + 75, 0, -14))
            sc.text(fx, fy, tok.strip() or repr(tok), layout.CAPTION, "mono",
                    layout.STRONG_INK if n == 0 else layout.CAPTION_INK, op, "mt")
            fx, fy = self.at_px((x0 + n * 230 + 75, 0, h + 14))
            sc.text(fx, fy, "%.0f%%" % (100 * p * g if g < 1 else 100 * p), layout.MIN_TEXT,
                    "regular", layout.CAPTION_INK, op, "mb")
        sc.probe(("bars",), (pr * 40, 0, 0), tol=4.0)

    # -- captions ------------------------------------------------------------------------------------------
    def _captions(self, sc, t, back, model):
        Tm, L = self.Tm, LAYER
        cap = self.caption
        e = self.p_("embed", t)
        before_fly = 1 - smoothstep(Tm["fly"][0], Tm["fly"][0] + 0.6, t)
        cap(sc, self.zbox(self.tok_e), "token embeddings: each token's row of a 50,257 × "
            "768 table", back * smoothstep(0.0, 0.15, e) * before_fly, above=True, max_width=0.6)
        cap(sc, self.zbox(self.pos_e), "+ position embeddings (one row per position, 0 to %d)" % (self.T - 1), back * smoothstep(0.45, 0.6, e) *
            before_fly, above=True, max_width=0.6)
        cap(sc, self.zbox(self.x0), "x: 6 tokens × 768 numbers, the residual stream",
            back * smoothstep(0.85, 1.0, e) * before_fly, max_width=0.6)
        card = self.cards[L][HEAD]
        dive = smoothstep(Tm["fly"][1] - 0.4, Tm["fly"][1] + 0.2, t) * \
            (1 - smoothstep(Tm["heads"][0], Tm["heads"][0] + 0.8, t)) * back
        proj = dive * (1 - smoothstep(Tm["kv"][1] - 0.6, Tm["kv"][1], t))
        cap(sc, self.zbox(self.ln), "LN(x): 6 × 768", proj, above=True)
        cap(sc, self.zbox(self.wq), "W_Q for head 12: 768 × 64", proj, above=True)
        mono = dict(size=layout.CAPTION)
        close = dive * smoothstep(Tm["close"][0], Tm["close"][1], t)
        self._label(sc, card["Q"], "Q", dive, "below", **mono)
        if t >= Tm["kv"][0]:
            kt = t >= Tm["transpose"][0] + 0.5 * (Tm["transpose"][1] - Tm["transpose"][0])
            self._label(sc, card["KT"] if kt else self.k_plain, "Kᵀ" if kt else "K",
                        dive * smoothstep(Tm["kv"][0], Tm["kv"][0] + 0.3, t), "above",
                        gap=30 if t >= Tm["transpose"][1] and self.px(TP) > 14 else 6, **mono)
            self._label(sc, card["V"], "V", dive * smoothstep(Tm["kv"][0], Tm["kv"][0] + .3, t),
                        "above", **mono)
            name = "A" if t >= Tm["softmax"][0] else "scores"
            self._label(sc, card["S"], name, dive, "below", **mono)
            self._label(sc, card["O"], "A·V", dive * (t >= Tm["av"][0]), "below", **mono)
        # token numbers beside rows of Q / A and above columns of K^T
        close *= smoothstep(19.5, 21.0, self.px(TP) * 0.815)
        if close > 0.01:
            cx, cz = self.cq[0], self.cq[2]
            for i, tok in enumerate(self.tr.tokens):
                fx, fy = self.at_px((cx - 6, 0, cz + 96 - (i + 0.5) * TP))
                sc.text(fx, fy, '%d "%s"' % (i + 1, tok.strip() or tok), layout.MIN_TEXT,
                        "mono", layout.CAPTION_INK, close * self._cap, "rm")
                if t >= Tm["transpose"][1]:
                    fx, fy = self.at_px((cx + 70 + (i + 0.5) * TP, 0, cz + 166 + 6))
                    sc.text(fx, fy, "%d" % (i + 1), layout.MIN_TEXT, "mono", layout.NOTE_INK,
                            close * self._cap, "mb")
        # the layer view
        lay = smoothstep(Tm["heads"][1] - 0.6, Tm["heads"][1], t) * \
            (1 - smoothstep(Tm["wide"][0], Tm["wide"][0] + 0.8, t)) * back
        if lay > 0.01:
            m = self.Mx(L)
            cap(sc, self.zbox(np.array([m - 124, 0, 150]), np.array([m + 892, 0, 150 + 2 * GRID_ROW
                                                                        + CARD_H])),
                "12 attention heads, all at once", lay, above=True)
            ct = self.p_("concat", t)
            cap(sc, self.zbox(self.concat[L]), "concatenate: 6 × (12 × 64)",
                lay * (ct > 0), above=True)
            cap(sc, self.zbox(self.attn_o[L]), "× W_O (768 × 768)",
                lay * (self.p_("wo", t) > 0) * (1 - bump(self.p_("add1", t) - 0.5, 0.3)))
            cap(sc, self.zbox(self.mlp_h[L]), "MLP: 768 to 3072, GELU",
                lay * (self.p_("mlp", t) > 0), above=True)
            cap(sc, self.zbox(self.mlp_o[L]), "3072 to 768", lay * (self.p_("mlp", t) > 0.35))
            cap(sc, self.zbox(self.seg_in[L]), "stream in", lay)
            cap(sc, self.zbox(self.seg_mid[L]), "+ attention", lay * (self.p_("add1", t) > 0.8))
            cap(sc, self.zbox(self.seg_out[L]), "+ MLP", lay * (self.p_("mlp", t) > 0.95))
        # the whole model
        wide = smoothstep(Tm["wide"][1] - 0.8, Tm["wide"][1], t) * \
            (1 - smoothstep(Tm["out"][0], Tm["out"][0] + 0.6, t))
        wide = max(wide, smoothstep(self.t_done + self.hold + 1.2, self.t_done + self.hold + 1.7,
                                    t) * (1 - smoothstep(self.duration - 1.4, self.duration - .8,
                                                         t))) * back
        if wide > 0.01:
            for l in range(G.N_LAYER):
                m = self.Mx(l)
                cap(sc, self.zbox(np.array([m - 124, 0, 0]), np.array([m + 2818, 0, 808])),
                    ("layer %d" if self.px(MOD) > 180 else "%d") % (l + 1), wide, above=True)
            cap(sc, self.zbox(self.x0), "embed", wide)
            cap(sc, self.zbox(self.lnf), "output", wide)
        pr = self.p_("predict", t)
        if pr > 0:
            op = back * smoothstep(0.0, 0.2, pr)
            cap(sc, self.zbox(self.lnf), "final LayerNorm", op)
            cap(sc, self.zbox(self.logits), "× W_E^T: 50,257 logits (last token)", op,
                above=True)

    def _label(self, sc, P, text, op, where="below", size=layout.CAPTION, gap=6):
        op *= self._cap
        if op <= 0.01:
            return
        lo, hi = self.zbox(P)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        x = (s[:, 0].min() + s[:, 0].max()) / 2 / self.cam.width
        if where == "above":
            sc.text(x, (s[:, 1].min() - gap) / self.cam.height, text, size, "mono",
                    layout.STRONG_INK, op, "mb")
        else:
            sc.text(x, (s[:, 1].max() + 6) / self.cam.height, text, size, "mono",
                    layout.STRONG_INK, op, "mt")

    def _subtitle(self, t):
        Tm = self.Tm
        tr = self.tr
        if t < Tm["embed"][0]:
            return 'the text "%s" is split into %d tokens' % (tr.text, self.T)
        if t < Tm["fly"][0]:
            return "each token's embedding + its position's embedding: one row of 768 per token"
        if t < Tm["fly"][1]:
            return ("GPT-2 small: 12 layers, each with 12 attention heads. "
                    "Fly into layer 5, head 12")
        if t < Tm["kv"][0]:
            return ("queries: Q = LN(x) · W_Q. Each row of LN(x) times the 768 × 64 "
                    "weights gives one row of Q")
        if t < Tm["transpose"][0]:
            return "keys and values the same way: K = LN(x) · W_K,  V = LN(x) · W_V"
        if t < Tm["qk"][0]:
            return "the transpose: K flips over its diagonal, so its 6 rows become the columns " \
                   "of K^T"
        if t < Tm["mask"][0]:
            return "scores = Q · K^T ÷ √64: one dot product for every pair of tokens"
        if t < Tm["softmax"][0]:
            return "causal mask: a token may not look ahead, so later tokens' scores become " \
                   "−∞"
        if t < Tm["av"][0]:
            return "softmax along each row: the weights sum to 1. This head looks at the " \
                   "previous token"
        if t < Tm["heads"][0]:
            av = self.p_("av", t)
            i = min(self.T - 1, int(av * self.T))
            w = tr.attn[LAYER, HEAD, i]
            top = [j for j in np.argsort(-w)[:2] if w[j] > 0.01]
            mix = " + ".join("%.2f · v%d" % (w[j], j + 1) for j in top)
            return "A · V: output row %d = %s, a weighted mix of the value rows" % (i + 1,
                                                                                        mix)
        if t < Tm["concat"][0]:
            return "this was 1 of 12 heads in layer 5, all computed at the same time"
        if t < Tm["wo"][0]:
            return "the 12 head outputs (6 × 64 each) are concatenated: 6 × 768"
        if t < Tm["add1"][0]:
            return "projected by W_O (768 × 768)"
        if t < Tm["mlp"][0]:
            return "and added to the residual stream: x + attention(x)"
        if t < Tm["wide"][0]:
            return "then the MLP, applied to each token separately (GELU: only > 0 light up), " \
                   "added to the stream"
        if t < Tm["full"][0]:
            return "every layer reads the stream and adds to it: attention mixes information " \
                   "between tokens, the MLP works on each token"
        if t < Tm["out"][0]:
            return "12 layers × 12 heads = 144 attention heads; every layer adds to the " \
                   "stream"
        if t < Tm["predict"][0]:
            return "after layer 12: the final LayerNorm"
        top, p = tr.top[0]
        return 'the last token\'s row × W_E^T gives 50,257 logits; softmax: "%s" %.0f%%' % (
            top, 100 * p)
