"""One training step of a 3-head self-attention classifier (a tiny vision
transformer) on one MNIST image.

Layout, left to right:

    image -> 16 patches        three head bands, one per head:            CLS output y
    tokens T (17 x 24)         Q K V (17 x 8 each) | A (17 x 17) | O       MLP, logits,
                               | where the CLS token looks (4 x 4)         softmax, one-hot

The forward pass:

1. the image is cut into 16 patches of 7 x 7; each becomes a token row, after
   a learned CLS row; positions are added,
2. every head projects the tokens to queries, keys and values,
3. scores = Q K^T / sqrt(8): head 1 is built slowly (the CLS query against
   every key), the rest quickly; softmax turns each row into weights that
   sum to 1 -- big weights light up,
4. each token's output is the weighted sum of the values; the CLS token's
   weights, laid back on the 4 x 4 patch grid, show where each head looks,
5. the CLS row (+ residual) goes through a small MLP to 10 logits.

Backward: only the CLS row is read, so only its rows of O, A and Q get
gradient, while every key and value does (the CLS query attended to them all).
Update: the per-head query/key/value weights step against their gradients.
Forward again: the attention shifts and p(label) rises.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .easing import ease_in_out, remap, smoothstep, soft_band
from .netanim import (
    INK_HI,
    INK_LO,
    NetAnimation,
    OutputPanel,
    amax,
    div_colors,
    gray_colors,
    lit_colors,
    particle,
)
from .text import TIMES
from .vit import DH, HEADS, N_TOK

BAND_Y = (0.33, 0.555, 0.78)
COL = dict(Q=0.255, K=0.305, V=0.355, A=0.445, O=0.53, map=0.597)
GRID = 4


def _weights_lit(a):
    """Attention weights (a row of A) -> lit colours, scaled by the row's max."""
    return lit_colors(a, amax(a))


class AttentionTrainingStep(NetAnimation):
    PHASES = ("forward", "loss", "backward", "update", "forward again")

    def __init__(self, tr, tr_after, lr):
        super().__init__("Attention: one training step", "", 10.0)
        self.tr, self.tr2, self.lr = tr, tr_after, lr
        self.hold = 2.5
        s06 = 0.6
        self.img = self.sheet((28, 28), (0.105, 0.37), 0.30, "image")
        self.tok = self.sheet((N_TOK, 24), (0.11, 0.76), 0.45, "tokens")
        self.heads = []
        for h, y in enumerate(BAND_Y):
            d = {}
            for key in "QKV":
                d[key] = self.sheet((N_TOK, DH), (COL[key], y), s06, key + str(h))
                d["W" + key] = self.sheet((24, DH), (COL[key], y), s06 * N_TOK / 24,
                                          "W%s%d" % (key, h))
            d["A"] = self.sheet((N_TOK, N_TOK), (COL["A"], y), s06, "A%d" % h)
            d["O"] = self.sheet((N_TOK, DH), (COL["O"], y), s06, "O%d" % h)
            d["map"] = self.sheet((GRID, GRID), (COL["map"], y), 1.4, "map%d" % h)
            self.heads.append(d)
        self.ycls = self.sheet((4, 6), (0.675, 0.42), 0.8, "y")
        self.u = self.sheet((4, 8), (0.855, 0.30), 0.8, "u")
        self.panel = OutputPanel(self, tr, (0.82, 0.67), 0.8)

        Tm, t = {}, 0.0

        def add(name, d, gap=0.2):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("reveal", 0.9)
        add("patch", 1.3)
        add("embed", 2.0)
        add("qkv", 1.8, 0.3)
        add("score0", 3.6)
        add("score12", 1.2)
        add("softmax", 1.4, 0.3)
        add("attend", 2.0)
        add("maps", 1.0, 0.3)
        add("cls", 1.4)
        add("mlp", 1.0)
        add("logits", 1.0, 0.2)
        add("bars", 1.0, 0.15)
        add("hot", 0.6, 0.15)
        add("loss", 0.6, 0.5)
        add("err", 0.8)
        add("back_mlp", 1.2)
        add("back_o", 1.6)
        add("back_a", 2.0)
        add("back_t", 1.4, 0.4)
        add("update", 3.6, 0.4)
        add("again", 4.2, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["bars"][0], Tm["err"][0], Tm["update"][0], Tm["again"][0]]

    def p(self, name, t):
        return remap(t, *self.Tm[name])

    def step_of(self, t):
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        tr, tr2 = self.tr, self.tr2
        ph = self.step_of(t)
        label_ph = 0 if reset > 0 else ph
        self.subtitle = [
            "forward: patches become tokens; each head matches queries with keys and mixes "
            "the values",
            "loss: the CLS token's output is classified; softmax vs the one-hot label",
            "backward: only the CLS row is read: its query gets gradient, and every key "
            "and value",
            "update: each head's query, key and value weights take a step  (lr = %g)" % self.lr,
            "forward again: the heads attend a little differently: p(%d) rises from %.2f "
            "to %.2f" % (tr.label, tr.p[tr.label], tr2.p[tr.label]),
        ][label_ph]
        self.back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        self.again = self.p("again", t)
        self._image(sc, t, reset)
        self._tokens(sc, t)
        for h in range(HEADS):
            self._head(sc, t, h)
        self._top(sc, t)
        self.step_indicator(sc, label_ph, self.PHASES)

    # -- image and patches ----------------------------------------------------
    def _image(self, sc, t, reset):
        x = self.tr.x
        rv = smoothstep(0.0, 0.9, t) * (1.0 - ease_in_out(remap(reset, 0.3, 1.0)))
        g = gray_colors(x)
        self.img.fill(g + (lit_colors(x, 1.0, INK_LO, INK_HI) - g) * rv)
        self.img.light(x, rv)
        b = self.img.block
        if b.offset is None:
            b.offset = np.zeros(b.base.shape[:3] + (3,))
        spread = ease_in_out(self.p("patch", t)) * (1.0 - ease_in_out(remap(reset, 0.2, 0.9)))
        r, c = np.mgrid[0:28, 0:28]
        gap = 0.9 * spread
        b.offset[:, :, 0, 0] += (c // 7 - 1.5) * gap
        b.offset[:, :, 0, 2] -= (r // 7 - 1.5) * gap
        if 0 < spread < 1:
            sc.probe(("patch",), (spread * 20, 0, 0), tol=8.0)
        self.caption(sc, self.img, "16 patches of 7%s7" % TIMES if spread > 0.5 else
                     "image 28%s28" % TIMES)

    def _patch_center(self, p):
        r, c = divmod(p, GRID)
        return self.img.block.cell_center(r * 7 + 3, c * 7 + 3, 0) + \
            self.img.block.offset[r * 7 + 3, c * 7 + 3, 0]

    # -- tokens ------------------------------------------------------------------
    def _tokens(self, sc, t):
        tr = self.tr
        T = tr.f["T"]
        e = self.p("embed", t)
        cols = div_colors(T.ravel(), amax(T))
        rows = np.repeat(np.arange(N_TOK), 24)
        show = rows <= np.floor(e * (N_TOK + 2)) - 1
        if self.Tm["back_t"][0] <= t < self.Tm["again"][0]:
            dT = tr.d["dT"]
            m = smoothstep(0.2, 0.9, self.p("back_t", t))
            cols = cols + (div_colors(dT.ravel(), amax(dT)) - cols) * m
        if self.again > 0:
            T2 = self.tr2.f["T"]
            cols = div_colors(T2.ravel(), amax(T))
        self.tok.fill(cols)
        self.tok.show(show)
        self.tok.cage = 1.0
        self.tok.fade(self.back)
        glint = soft_band(e, 0.86, 1.0, 0.06)              # + position
        self.tok.block.white[...] = 0.5 * glint
        if 0 < e < 1:
            for p in range(16):
                u = remap(e, 0.06 * (p + 1) * 0.9, 0.06 * (p + 1) * 0.9 + 0.2)
                particle(sc, self._patch_center(p), self.tok.cell((p + 1) * 24 + 12), u,
                         INK_HI, 0.6, lift=1.0)
            sc.probe(("embed",), (e * 30, 0, 0), tol=8.0)
        if t >= self.Tm["back_t"][0] + 0.3 and t < self.Tm["again"][0]:
            text = "gradient at the tokens"
        elif e > 0.9:
            text = "tokens: CLS + 16 patches"
        else:
            text = "tokens (17 %s 24)" % TIMES
        self.caption(sc, self.tok, text)

    # -- one head band -----------------------------------------------------------
    def _head(self, sc, t, h):
        tr, tr2 = self.tr, self.tr2
        f, d = tr.f, tr.d
        H = self.heads[h]
        back, again = self.back, self.again
        sl = slice(h * DH, (h + 1) * DH)
        qkv = self.p("qkv", t)
        bwd_a = self.p("back_a", t)
        upd = self.p("update", t)
        in_update = self.Tm["update"][0] <= t < self.Tm["update"][1]
        for key in "QKV":
            s = H[key]
            M = f[key][:, sl]
            col = div_colors(M.ravel(), amax(M))
            if bwd_a > 0 and t < self.Tm["again"][0]:
                G = d["d" + key][:, sl]
                gm = G != 0
                gcol = div_colors(G.ravel(), amax(G), mask=gm.ravel())
                m = smoothstep(0.55, 0.9, bwd_a)
                col = col + (gcol - col) * m
            if again > 0:
                M2 = tr2.f[key][:, sl]
                col = col + (div_colors(M2.ravel(), amax(M)) - col) * smoothstep(0.0, 0.2, again)
            s.fill(col)
            rows = np.repeat(np.arange(N_TOK), DH)
            s.show(rows < qkv * (N_TOK + 3) - 2)
            s.cage = 1.0
            s.visible = not in_update
            s.fade(back)
            # its weights during the update
            w = H["W" + key]
            w.visible = in_update
            if in_update:
                W = tr.P["W" + key.lower()][:, sl]
                dW = tr.g["W" + key.lower()][:, sl]
                W2 = tr2.P["W" + key.lower()][:, sl]
                ch = -self.lr * dW
                c = div_colors(W.ravel(), amax(W))
                c = c + (div_colors(ch.ravel(), amax(ch)) - c) * smoothstep(0.15, 0.35, upd)
                c = c + (div_colors(W2.ravel(), amax(W)) - c) * smoothstep(0.65, 0.85, upd)
                w.fill(c)
                w.cage = 1.0
                w.block.white[...] = 0.7 * soft_band(upd, 0.68, 0.86, 0.05)
            if h == 0:
                if in_update:
                    text = "W" + key.lower()
                else:
                    text = key if bwd_a <= 0.55 or t >= self.Tm["again"][0] else "d" + key
                self.caption(sc, w if in_update else s, text, size=0.026, above=True)
        if qkv > 0 and qkv < 1:
            for key in "QKV":
                particle(sc, self.tok.center, H[key].center, remap(qkv, 0.0, 0.5),
                         C.NEUTRAL, 0.7, lift=1.5)
            sc.probe(("qkv",), (qkv * 30, 0, 0), tol=8.0)

        # ---- scores and weights ---------------------------------------------------
        A = H["A"]
        S, W = f["S"][h], f["A"][h]
        scol = div_colors(S.ravel(), amax(S))
        wcol = np.concatenate([_weights_lit(W[i]) for i in range(N_TOK)])
        sm = smoothstep(0.0, 1.0, self.p("softmax", t))
        rows = np.repeat(np.arange(N_TOK), N_TOK)
        cols_ = np.tile(np.arange(N_TOK), N_TOK)
        row_wave = np.clip(sm * 1.6 - rows / N_TOK * 0.6, 0, 1)[:, None]
        col = scol + (wcol - scol) * row_wave
        if h == 0:
            sp = self.p("score0", t)
            n0 = sp / 0.45 * N_TOK                           # CLS row, cell by cell
            nr = (sp - 0.45) / 0.55 * (N_TOK - 1)            # then the other rows
            show = ((rows == 0) & (cols_ < n0)) | ((rows >= 1) & (rows - 1 < nr))
            if 0 < sp < 0.45:
                j = int(min(N_TOK - 1, n0))
                for key, r in (("Q", 0), ("K", j)):
                    lo, hi = H[key].block.box((r, r + 1), (0, DH), (0, 1))
                    sc.wire_box(lo, hi, inflate=0.05, color=(0.96, 0.96, 0.96), width=1.3,
                                opacity=0.95, bias=0.1)
                lo, hi = A.block.box((0, 1), (j, j + 1), (0, 1))
                sc.wire_box(lo, hi, inflate=0.05, color=(0.96, 0.96, 0.96), width=1.3,
                            opacity=0.95, bias=0.1)
                sc.probe(("score",), (n0 * 4, 0, 0), tol=8.0)
        else:
            sp = self.p("score12", t)
            show = rows < sp * (N_TOK + 1)
        if self.Tm["back_a"][0] <= t < self.Tm["again"][0]:
            dA = d["dA"][h]
            gcol = div_colors(dA.ravel(), amax(dA), mask=(dA != 0).ravel())
            col = col + (gcol - col) * smoothstep(0.0, 0.4, self.p("back_a", t))
        if again > 0:
            W2 = tr2.f["A"][h]
            w2col = np.concatenate([_weights_lit(W2[i]) for i in range(N_TOK)])
            col = col + (w2col - col) * smoothstep(0.15, 0.4, again)
        A.fill(col)
        A.show(show)
        A.cage = 1.0
        A.fade(back)
        if h == 0:
            if self.Tm["back_a"][0] <= t < self.Tm["again"][0]:
                text = "dA"
            elif sm > 0.5:
                text = "softmax"
            else:
                text = "Q·K / √8"
            self.caption(sc, A, text, size=0.026, above=True)
        self.caption_left(sc, H["Q"], "head %d" % (h + 1), size=0.028)
        if h == HEADS - 1 and self.Tm["back_o"][0] <= t < self.Tm["again"][0]:
            lo = H["A"].bbox()[0]
            hi = H["O"].bbox()[1]
            self.caption(sc, (lo, hi), "only the CLS row is read, so only it gets gradient",
                         size=0.026)

        # ---- weighted sum of the values ------------------------------------------------
        O = H["O"]
        Oh = f["Oh"][h]
        at = self.p("attend", t)
        ocol = div_colors(Oh.ravel(), amax(Oh))
        if self.Tm["back_o"][0] <= t < self.Tm["again"][0]:
            dO = d["dO"][:, sl]
            gcol = div_colors(dO.ravel(), amax(dO), mask=(dO != 0).ravel())
            ocol = ocol + (gcol - ocol) * smoothstep(0.2, 0.7, self.p("back_o", t))
        if again > 0:
            O2 = tr2.f["Oh"][h]
            ocol = ocol + (div_colors(O2.ravel(), amax(Oh)) - ocol) * smoothstep(0.3, 0.5, again)
        O.fill(ocol)
        O.show(np.repeat(np.arange(N_TOK), DH) < at * (N_TOK + 2) - 1)
        O.cage = 1.0
        O.fade(back)
        if 0 < at < 0.5:
            w = W[0]
            for j in np.argsort(-w)[:6]:
                particle(sc, H["V"].cell(j * DH + DH // 2), O.cell(DH // 2), remap(at, 0.0, 0.45),
                         INK_HI, 0.5 + 0.5 * w[j] / w.max(), opacity=0.4 + 0.6 * w[j] / w.max(),
                         lift=1.0)
            sc.probe(("attend", h), (at * 30, 0, 0), tol=8.0)
        if h == 0:
            text = "dO" if self.Tm["back_o"][0] <= t < self.Tm["again"][0] else "A·V"
            self.caption(sc, O, text, size=0.026, above=True)

        # ---- where the CLS token looks ---------------------------------------------------
        mp = H["map"]
        mshow = self.p("maps", t)
        a = f["A"][h][0, 1:]
        mc = _weights_lit(a)
        if again > 0:
            a2 = tr2.f["A"][h][0, 1:]
            mc = mc + (lit_colors(a2, amax(a)) - mc) * smoothstep(0.15, 0.4, again)
        mp.fill(mc)
        mp.light(a - 0.5 * a.max(), mshow)
        mp.fade(mshow * back)
        mp.cage = mshow * back
        if h == 0:
            self.caption_side(sc, mp, "where CLS looks", mshow)

    # -- CLS output, MLP and the output panel ---------------------------------------------
    def _top(self, sc, t):
        tr, tr2 = self.tr, self.tr2
        f, d = tr.f, tr.d
        back, again = self.back, self.again
        z = f["z"]
        cl = self.p("cls", t)
        zcol = div_colors(z, amax(z))
        if self.Tm["back_mlp"][0] <= t < self.Tm["again"][0]:
            dz = d["dY"][0]
            zcol = zcol + (div_colors(dz, amax(dz)) - zcol) * smoothstep(0.5, 0.9,
                                                                         self.p("back_mlp", t))
        if again > 0:
            zcol = zcol + (div_colors(tr2.f["z"], amax(z)) - zcol) * smoothstep(0.5, 0.65, again)
        self.ycls.fill(zcol)
        self.ycls.show(np.arange(24) < cl * 26)
        self.ycls.cage = 1.0
        self.ycls.fade(back)
        if 0 < cl < 0.6:
            for h in range(HEADS):
                particle(sc, self.heads[h]["O"].cell(DH // 2), self.ycls.center,
                         remap(cl, 0.0, 0.5), INK_HI, 0.7, lift=1.0)
            particle(sc, self.tok.cell(12), self.ycls.center, remap(cl, 0.05, 0.55),
                     C.NEUTRAL, 0.7, lift=2.0)
            sc.probe(("cls",), (cl * 30, 0, 0), tol=8.0)
        self.caption(sc, self.ycls, "CLS output", size=0.026)

        mlp = self.p("mlp", t)
        u = f["u"]
        ucol = lit_colors(u, amax(u))
        if self.Tm["back_mlp"][0] <= t < self.Tm["again"][0]:
            du = (d["dlog"] @ tr.P["W2"].T) * (f["pre1"] > 0)
            ucol = ucol + (div_colors(du, amax(du), mask=du != 0) - ucol) * \
                smoothstep(0.0, 0.5, self.p("back_mlp", t))
        if again > 0:
            u2 = tr2.f["u"]
            ucol = ucol + (lit_colors(u2, amax(u)) - ucol) * smoothstep(0.6, 0.75, again)
            self.u.light(u2, smoothstep(0.6, 0.75, again))
        elif not (self.Tm["back_mlp"][0] <= t):
            self.u.light(u, smoothstep(0.6, 1.0, mlp))
        self.u.fill(ucol)
        self.u.show(np.arange(32) < mlp * 34)
        self.u.cage = 1.0
        self.u.fade(back)
        self.caption(sc, self.u, "MLP (ReLU)", size=0.026)
        if 0 < mlp < 1:
            particle(sc, self.ycls.center, self.u.center, remap(mlp, 0.0, 0.5), INK_HI, 0.8,
                     lift=0.5)

        # output panel
        lg = self.p("logits", t)
        mask = np.arange(10) < lg * 11
        logits = div_colors(f["logits"], amax(f["logits"]))
        err = div_colors(d["dlog"], amax(d["dlog"]))
        col = logits + (err - logits) * ease_in_out(self.p("err", t))
        if again > 0:
            col = col + (div_colors(tr2.f["logits"], amax(tr2.f["logits"])) - col) * \
                smoothstep(0.7, 0.85, again)
        w = smoothstep(0.8, 1.0, again)
        self.panel.set_probs(tr.p + (tr2.p - tr.p) * w, tr.loss + (tr2.loss - tr.loss) * w)
        tb = self.p("bars", t) * back
        th = self.p("hot", t) * back
        tl = (self.p("loss", t) * (1 - self.p("err", t)) + smoothstep(0.9, 1.0, again)) * back
        self.panel.draw(sc, tb, th, tl, mask, col, fade=back)
        self.panel.emit(sc)
        self.panel.captions(sc, tb, th, tl, "error p - y" if t >= self.Tm["err"][0] and
                            again <= 0.7 else "logits")
        if 0 < lg < 1:
            particle(sc, self.u.center, self.panel.logits.center, remap(lg, 0.0, 0.6), INK_HI,
                     0.8, lift=1.0)
        self.panel.note(sc, "before the step: p = %.2f, loss %.2f" % (tr.p[tr.label], tr.loss),
                        smoothstep(0.9, 1.0, again) * back)
