"""Inside GPT-2: one attention head up close, then its layer, then the model.

One world and one continuous camera move; every number comes from the real
GPT-2 small (:mod:`voxanim.gpt2`) reading "One, two, three," (by default):

1. tokens: each token's row of the embedding table plus its position's row
   gives the residual stream x, T tokens x 768 numbers,
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

Matrices are flat textured panels (:mod:`voxanim.gpt2_panel`) laid out once
in model space (:mod:`voxanim.gpt2_layout`).  The zoom is a transform of the
whole world (:meth:`GPT2Animation.Z`), driven by a keyframed
:class:`~voxanim.gpt2_camera.CameraPath`, so one layout serves every shot.
"""

from __future__ import annotations

import math

import numpy as np

from . import gpt2 as G
from . import layout
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2_camera import CameraPath
from .gpt2_layout import (
    BAR_H,
    BAR_PITCH,
    BAR_W,
    BARS_DX,
    CARD_H,
    CARD_W,
    CARDS_Z,
    DROP_DX,
    EMB_POS,
    EMB_TOK,
    EMBED_X,
    GAP,
    GRID_RIGHT,
    GRID_TOP,
    GRID_X,
    HALF,
    HEADS_OUT_Z,
    HT,
    KT_AT,
    LEAD,
    MAX_TOKENS,
    MLP_FORK_X,
    MOD,
    O_AT,
    OUT_X,
    PROJ_Z,
    Q_AT,
    S_AT,
    SEG_IN_X,
    SEG_MID_X,
    SEG_OUT_X,
    STREAM_W,
    STREAM_Z,
    V_AT,
    WIDE_Z,
    card_origin,
    layer_x,
)
from .gpt2_panel import DIAG, Panel, bump, diverging, lit, rotate
from .netanim import ACT_HI, ACT_LO, OFF, NetAnimation
from .text import MINUS, num

__all__ = ["GPT2Animation", "Panel"]

LAYER, HEAD = 4, 11          # the head we dive into: layer 5, head 12 (1-based)

# phase -> (start, end) in seconds
TIMELINE = dict(
    labels=(0.2, 1.0), embed=(1.0, 3.6), fly=(4.4, 8.6), proj=(8.8, 12.4),
    kv=(12.4, 13.8), close=(13.6, 15.0), transpose=(15.0, 17.4), qk=(17.8, 29.8),
    mask=(29.8, 31.2), softmax=(31.4, 34.0), av=(34.2, 38.6), heads=(39.2, 42.6),
    concat=(42.8, 45.2), wo=(45.2, 46.6), add1=(46.6, 47.8), mlp=(47.8, 50.4),
    wide=(50.8, 53.4), flow=(53.4, 69.4), full=(69.4, 70.6), out=(71.4, 74.0),
    predict=(74.0, 77.6))
# the step indicator: (phase that starts the step, index into PHASES)
STEP_STARTS = (("labels", 0), ("fly", 1), ("transpose", 2), ("qk", 3), ("softmax", 4),
               ("av", 5), ("heads", 6), ("wide", 7), ("out", 8))
# the scores worked in full, (row, column), each SLOW_DT s, SLOW_PITCH apart;
# the rest of the T x T scores share what is left of the "qk" phase
SLOW_CELLS = ((1, 0), (1, 1), (2, 1))
SLOW_DT, SLOW_PITCH = 2.7, 2.8
INK = (0.62, 0.64, 0.66)     # connector lines


def _rows_done(x, T, n):
    """``(T, n)``: row ``i`` is ``clip(x - i, 0, 1)`` done (``x`` counts rows)."""
    return np.clip(x - np.arange(T), 0, 1)[:, None] * np.ones((1, n))


def _reveal(panel, done):
    """``(tex, alpha)`` for a panel filling in: cells not started are OFF and faint."""
    tex = np.where(done[..., None] > 0, panel.tex, OFF).astype(np.float32)
    return tex, (0.15 + 0.85 * done).astype(np.float32)


def _row_hl(R, Cn, i, k):
    """``(R, Cn)`` highlight: row ``i`` at strength ``k``, the rest 0."""
    h = np.zeros((R, Cn), np.float32)
    h[i] = k
    return h


def _thousands(n):
    """``50257`` -> ``"50,257"``."""
    return format(n, ",")


class GPT2Animation(NetAnimation):
    """The ``gpt2-attention`` animation for a prompt of 2 to :data:`MAX_TOKENS` tokens.

    Model space is laid out by :mod:`voxanim.gpt2_layout`; a token's row is
    ``self.tp = HT / T`` model units tall.  Each frame :meth:`set_view`
    picks the zoom (``self.s``, screen units per model unit before the
    camera) and focus, and :meth:`Z` maps model space to the world.
    """

    PHASES = ("tokens", "Q K V", "flip", "scores", "softmax", "A·V", "heads", "layers",
              "output")
    hold, reset = 3.0, 3.5

    def __init__(self, text="One, two, three,"):
        super().__init__("Inside GPT-2", "", 10.0, azimuth=8.0, elevation=16.0)
        self.tr = G.run(text)
        self.T = len(self.tr.ids)
        if not 2 <= self.T <= MAX_TOKENS:
            raise SystemExit("gpt2-attention: the prompt must be 2 to %d tokens, not %d"
                             % (MAX_TOKENS, self.T))
        self.tp = HT / self.T                # model units per token row
        self._build_stream()
        self._build_dive()
        self.Tm = dict(TIMELINE)
        self.qk_cells = self._qk_schedule()
        self.finish(max(b for _a, b in self.Tm.values()) + 0.4)
        self.camera = self._camera_path()
        self.subtitles = self._subtitle_table()

    def finish(self, t_done):
        """Set ``t_done`` / ``duration`` (there are no Sheets, so no floor)."""
        self.t_done = t_done
        self.duration = t_done + self.hold + self.reset
        self.floor_z = -1e6

    # ---------------------------------------------------------------- building
    def _rows(self, colors, origin, width=1.0):
        """A panel with one row per token (``self.tp`` tall), columns ``width`` wide."""
        return Panel(colors, origin, cell=(self.tp, width))

    def _build_stream(self):
        """The embeddings, every layer's stream / concat / W_O / MLP panels and
        cards, the final LayerNorm and the logits."""
        tr, W = self.tr, G.weights()
        self.tok_e = self._rows(diverging(tr.emb), (EMBED_X, 0, EMB_TOK))
        self.pos_e = self._rows(diverging(tr.pos), (EMBED_X, 0, EMB_POS))
        self.x0 = self._rows(diverging(tr.resid[0]), (EMBED_X, 0, STREAM_Z))
        self.seg_in, self.seg_mid, self.seg_out = [], [], []
        self.concat, self.attn_o, self.mlp_h, self.mlp_o = [], [], [], []
        self.cards = []
        for l in range(G.N_LAYER):
            m = layer_x(l)
            mid, out = m + SEG_MID_X, m + SEG_OUT_X
            self.seg_in.append(self._rows(diverging(tr.resid[l]), (m + SEG_IN_X, 0, STREAM_Z)))
            self.seg_mid.append(self._rows(diverging(tr.resid[l] + tr.attn_out[l]),
                                           (mid, 0, STREAM_Z)))
            self.seg_out.append(self._rows(diverging(tr.resid[l + 1]), (out, 0, STREAM_Z)))
            cat = tr.head_out[l].transpose(1, 0, 2).reshape(self.T, -1)
            self.concat.append(self._rows(diverging(cat), (mid, 0, WIDE_Z)))
            self.attn_o.append(self._rows(diverging(tr.attn_out[l]), (mid, 0, PROJ_Z)))
            self.mlp_h.append(self._rows(lit(tr.mlp_hidden[l]), (out, 0, WIDE_Z),
                                         STREAM_W / G.D_MLP))
            self.mlp_o.append(self._rows(diverging(tr.mlp_out[l]), (out, 0, PROJ_Z)))
            self.cards.append([self._card(l, h) for h in range(G.N_HEAD)])
        final = G.layer_norm(tr.resid[-1], W["ln_f.weight"], W["ln_f.bias"])
        self.lnf = self._rows(diverging(final), (OUT_X, 0, STREAM_Z))
        lg = tr.logits[-1]
        self.logits = self._rows(diverging(lg[None] - lg.mean()), (OUT_X, 0, PROJ_Z),
                                 STREAM_W / len(lg))

    def _card(self, l, h):
        """Layer ``l``, head ``h``: ``{Q, KT, S, A, V, O}`` panels (Falk scheme)."""
        tr, tp = self.tr, self.tp
        cx, cz = card_origin(l, h)

        def at(p):
            return (cx + p[0], 0.0, cz + p[1])

        return dict(Q=self._rows(diverging(tr.q[l, h]), at(Q_AT)),
                    KT=Panel(diverging(tr.k[l, h].T), at(KT_AT), cell=(1.0, tp)),
                    S=Panel(diverging(tr.scores[l, h]), at(S_AT), cell=(tp, tp)),
                    A=Panel(lit(tr.attn[l, h], 1.0), at(S_AT), cell=(tp, tp)),
                    V=self._rows(diverging(tr.v[l, h]), at(V_AT)),
                    O=self._rows(diverging(tr.head_out[l, h]), at(O_AT)))

    def _build_dive(self):
        """The dive head's extras: LN(x) left of Q, W_Q above it, K before its flip."""
        tr, L, H = self.tr, LAYER, HEAD
        cx, cz = card_origin(L, H)
        self.cq = np.array([cx, 0.0, cz])
        self.ln = self._rows(diverging(tr.ln1[L]),
                             (cx - GAP - G.D_MODEL * HALF, 0, cz + Q_AT[1]), HALF)
        wq = G.weights()["h.%d.attn.c_attn.weight" % L][:, H * G.D_HEAD:(H + 1) * G.D_HEAD]
        self.wq = Panel(diverging(wq), (cx, 0, cz + Q_AT[1] + GAP + G.D_MODEL * HALF),
                        cell=(HALF, 1.0))
        self.k_plain = self._rows(diverging(tr.k[L, H]), (cx + KT_AT[0], 0, cz + KT_AT[1]))

    def _qk_schedule(self):
        """``[((i, j), start s, duration s, slow)]`` for every score of the dive head."""
        T = self.T
        slow = [c for c in SLOW_CELLS if max(c) < T]
        rest = [(i, j) for i in range(T) for j in range(T) if (i, j) not in slow]
        a, b = self.Tm["qk"]
        t, cells = a, []
        for c in slow:
            cells.append((c, t, SLOW_DT, True))
            t += SLOW_PITCH
        dt = (b - t) / len(rest)
        for c in rest:
            cells.append((c, t, dt, False))
            t += dt
        return cells

    def _camera_path(self):
        """The shots, as a :class:`CameraPath` (spans in model units)."""
        Tm = self.Tm
        card_c = self.cq + np.array([CARD_W / 2 - 30, 0, CARD_H / 2 + 10])     # the dive head
        proj_c = self.cq + np.array([-163.0, 0, 240.0])        # LN(x), W_Q and Q
        layer_c = np.array([layer_x(LAYER) + 1350, 0, 380.0])  # the dive layer
        full_c = np.array([(EMBED_X + OUT_X + 1500) / 2, 0, 380.0])
        full_s = OUT_X + 1500 - EMBED_X + 600                   # the whole model
        out_c = np.array([OUT_X + 1050, 0, 210.0])             # LN_f, logits and bars
        open_c = np.array([EMBED_X + 300, 0, 135.0])           # the embeddings and x
        tok_c = np.array([EMBED_X + 190, 0, EMB_TOK - 40.0])   # the token embeddings
        mid_c = (open_c + proj_c) / 2 + np.array([0, 0, 300.0])
        hold_end = self.t_done + self.hold
        return CameraPath([
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
        ])

    # ---------------------------------------------------------------- camera
    def pulse_x(self, t):
        """Model x of the information pulse on the stream during "flow"."""
        u = self.progress("flow", t)
        return EMBED_X - 600 + u * (OUT_X + 200 - EMBED_X + 600)

    def _track(self, t):
        """The tracking shot's focus: just ahead of the pulse."""
        return np.array([self.pulse_x(t) + 800, 0, 380.0])

    def camera_at(self, t):
        """``(focus (3,), span)`` in model units at time ``t`` (s)."""
        return self.camera(t)

    def set_view(self, t):
        """Set this frame's zoom ``self.s``, ``self.focus`` and screen ``self.anchor``."""
        f, span = self.camera_at(t)
        px_per_unit = self.cam.M[0, 0]
        self.s = 0.86 * self.cam.width / (span * px_per_unit)
        self.focus = np.asarray(f, np.float64)
        self.anchor = self.cam.frac(0.5, 0.58)

    def Z(self, p):
        """Model-space point(s) ``p`` -> world (the focus lands on the anchor)."""
        return self.anchor + (np.asarray(p, np.float64) - self.focus) * self.s

    def zbox(self, panel_or_lo, hi=None):
        """World ``(lo, hi)`` of a panel, or of a model-space box ``lo, hi``."""
        lo, hi = panel_or_lo.box() if hi is None else (panel_or_lo, hi)
        return self.Z(lo), self.Z(hi)

    def px(self, units):
        """Screen pixels of a model length (at the current zoom)."""
        return units * self.s * self.cam.M[0, 0]

    def at_px(self, p):
        """Model point ``p`` -> ``(x, y)`` as fractions of the frame."""
        q = self.cam.project(self.Z(p))
        return q[0] / self.cam.width, q[1] / self.cam.height

    def _rows_legible(self):
        """0..1: whether a token row is tall enough on screen to label (~24 px)."""
        return smoothstep(19.5, 21.0, self.px(self.tp) * 0.815)

    # ---------------------------------------------------------------- frame

    def step_of(self, t):
        """Index into :attr:`PHASES` of the step shown at ``t``."""
        cur = 0
        for name, k in STEP_STARTS:
            if t >= self.Tm[name][0]:
                cur = k
        return cur

    def draw(self, sc, t, reset):
        """Draw frame ``t`` (s) into ``sc``; ``reset`` 0..1 is the loop back."""
        self.set_view(t)
        self._cap *= 1.0 - smoothstep(0.0, 0.12, reset)     # text leaves before the camera
        back = 1.0 - smoothstep(0.55, 1.0, reset)
        ph = 0 if reset > 0.5 else self.step_of(t)
        self.subtitle = ""
        Tm = self.Tm
        # probes: the camera move drives motion blur
        sc.probe(("cam", 0), self.Z((0.0, 0.0, 0.0)))
        sc.probe(("cam", 1), self.Z((OUT_X, 0.0, 800.0)))
        sc.probe(("cam", 2), self.Z((EMBED_X, 0.0, 0.0)))

        self._embedding(sc, t, back)
        # the whole model is there during the fly-in and from the zoom-out on;
        # around the dive only the head's own layer stream stays
        model = max(1 - smoothstep(Tm["fly"][1] - 1.6, Tm["fly"][1] - 0.4, t),
                    smoothstep(*Tm["heads"], t)) * smoothstep(Tm["fly"][0], Tm["fly"][0] + 1, t)
        model *= back
        flow = self.progress("flow", t)
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
        self._captions(sc, t, back)
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

    # -- layers and the stream ---------------------------------------------------------------------
    def _hl(self, panel, pulse_x):
        """Highlight of a panel as the pulse (model x ``pulse_x``) passes it."""
        if pulse_x is None:
            return None
        lo, hi = panel.box()
        return 0.55 * bump((lo[0] + hi[0]) / 2 - pulse_x, 900.0)

    def _layer(self, sc, l, op, pulse_x):
        """A layer other than the dive's, complete, at opacity ``op``."""
        if op <= 0.01:
            return
        for h in range(G.N_HEAD):
            self._card_draw(sc, l, h, op, pulse_x)
        for P in (self.seg_in[l], self.seg_mid[l], self.seg_out[l], self.concat[l],
                  self.attn_o[l], self.mlp_h[l], self.mlp_o[l]):
            P.draw(self, sc, op, highlight=self._hl(P, pulse_x))

    def _card_draw(self, sc, l, h, op, pulse_x=None):
        """Head ``h`` of layer ``l`` as computed: Q, K^T, A, V, A·V."""
        c = self.cards[l][h]
        for k in ("Q", "KT", "A", "V", "O"):
            c[k].draw(self, sc, op, highlight=self._hl(c[k], pulse_x))

    def _stream_lines(self, sc, op, dive_op=1.0):
        """Thin connectors: the stream between its segments, and each layer's
        branches (stream -> heads -> concat -> W_O -> add; stream -> MLP -> add).
        The dive layer's branches are at ``op * dive_op``."""
        if op <= 0.01:
            return
        mid = STREAM_Z - HT / 2
        for l in range(G.N_LAYER):
            m = layer_x(l)
            for a, b in ((m - LEAD, m), (m + STREAM_W, m + SEG_MID_X),
                         (m + SEG_MID_X + STREAM_W, m + SEG_OUT_X)):
                self._line(sc, (a, mid), (b, mid), 1.2, op * 0.8)
            br = op * (dive_op if l == LAYER else 1.0)
            for p, q in self._branches(m):
                self._line(sc, p, q, 1.0, br * 0.6)
        self._line(sc, (EMBED_X + STREAM_W, mid), (layer_x(0) - LEAD, mid), 1.2, op * 0.8)
        end = layer_x(G.N_LAYER - 1) + SEG_OUT_X + STREAM_W
        self._line(sc, (end, mid), (OUT_X, mid), 1.2, op * 0.8)
        for l in range(G.N_LAYER - 1):
            self._line(sc, (layer_x(l) + SEG_OUT_X + STREAM_W, mid),
                       (layer_x(l + 1) - LEAD, mid), 1.2, op * 0.8)

    @staticmethod
    def _branches(m):
        """The branch connectors of the layer at x ``m``: ``[((x, z), (x, z))]``."""
        mid, half = STREAM_Z - HT / 2, WIDE_Z - HT / 2      # stream / wide-row middles
        att, mlp = m + SEG_MID_X + DROP_DX, m + SEG_OUT_X + DROP_DX
        fork = m + MLP_FORK_X
        return [((m + SEG_IN_X + STREAM_W / 2, STREAM_Z), (m + SEG_IN_X + STREAM_W / 2, CARDS_Z)),
                ((m + GRID_RIGHT, HEADS_OUT_Z), (m + SEG_MID_X, half)),
                ((att, WIDE_Z - HT), (att, PROJ_Z)),
                ((att, PROJ_Z - HT), (att, STREAM_Z)),
                ((m + SEG_MID_X + STREAM_W, mid), (fork, mid)),
                ((fork, mid), (fork, half)),
                ((fork, half), (m + SEG_OUT_X, half)),
                ((mlp, WIDE_Z - HT), (mlp, PROJ_Z)),
                ((mlp, PROJ_Z - HT), (mlp, STREAM_Z))]

    def _line(self, sc, a, b, width, op):
        """A connector from model ``(x, z)`` ``a`` to ``b``; ``width`` in px."""
        sc.line(self.Z((a[0], 0, a[1])), self.Z((b[0], 0, b[1])), width, INK, op)

    def _pulse(self, sc, x, back):
        """The glowing band of information at model x ``x``."""
        w = max(160.0, 26.0 / max(self.px(1.0), 1e-9))
        lo = self.Z((x - w / 2, 0, 30)) + self.cam.view * 50
        a1 = np.array([w, 0, 0]) * self.s
        a2 = np.array([0, 0, -(HT + 60)]) * self.s

        def shader(P):
            u = np.clip((P - lo) @ (a1 / (a1 @ a1)), 0, 1)
            a = (np.sin(np.pi * u) ** 2).astype(np.float32)
            return (np.broadcast_to(np.float32(ACT_HI), P.shape).copy(), 0.95 * back * a,
                    1.2 * a)

        sc.quad(lo, a1, a2, shader)
        sc.probe(("pulse",), self.Z((x, 0, 0)), tol=3.0)

    # -- tokens and embeddings ---------------------------------------------------------------------
    def _embedding(self, sc, t, back):
        """The token rows looked up, the positions added, both sliding into x."""
        Tm = self.Tm
        e = self.progress("embed", t)
        on = back * smoothstep(0.0, 0.15, e)
        # each token's embedding row is looked up in turn
        rows = np.arange(self.T)[:, None] * np.ones((1, G.D_MODEL))
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
        op *= self._rows_legible()
        if op > 0.01:
            for i, tok in enumerate(self.tr.tokens):
                fx, fy = self.at_px((EMBED_X - 14, 0, EMB_TOK - (i + 0.5) * self.tp))
                sc.text(fx, fy, '"%s"' % tok, layout.MIN_TEXT, "mono", layout.CAPTION_INK, op,
                        "rm")

    # -- the dive layer ----------------------------------------------------------------------------
    def _dive_layer(self, sc, t, back, model, pulse_x):
        """Layer LAYER: its other heads, the dive head step by step, and the rest."""
        L = LAYER
        self.seg_in[L].draw(self, sc, model, highlight=self._hl(self.seg_in[L], pulse_x))
        for h in range(G.N_HEAD):
            if h != HEAD:
                self._card_draw(sc, L, h, model, pulse_x)
        self._dive(sc, t, back, pulse_x)
        self._after_heads(sc, t, back, model, pulse_x)

    def _dive(self, sc, t, back, pulse_x):
        """The dive head: Q, K, V, the flip, the scores, softmax and A·V."""
        Tm = self.Tm
        card = self.cards[LAYER][HEAD]
        op = back * max(smoothstep(Tm["fly"][1] - 1.2, Tm["fly"][1], t),
                        1 - smoothstep(Tm["fly"][0], Tm["fly"][0] + 1.0, t))
        if op <= 0.01:
            return
        if t >= Tm["av"][1] or t < Tm["fly"][1] - 1.2:          # complete
            self._card_draw(sc, LAYER, HEAD, op, pulse_x)
            return
        self._dive_q(sc, t, card, op)
        self._dive_kv(sc, t, card, op)
        if self.progress("transpose", t) > 0:
            self._dive_flip(sc, t, card, op)
        T = self.T
        if t >= Tm["qk"][0]:
            self._scores(sc, t, card, op)
        elif t >= Tm["kv"][0]:
            card["S"].draw(self, sc, op * 0.6, tex=np.full((T, T, 3), OFF, np.float32),
                           alpha=np.full((T, T), 0.15, np.float32))
        av = self.progress("av", t)
        if t >= Tm["kv"][0]:
            tex, al = _reveal(card["O"], _rows_done(av * T, T, G.D_HEAD))
            card["O"].draw(self, sc, op, tex=tex, alpha=al,
                           highlight=_row_hl(T, G.D_HEAD, min(T - 1, int(av * T)), 0.5)
                           if 0 < av < 1 else None)

    def _dive_q(self, sc, t, card, op):
        """Q = LN(x) W_Q, one row at a time, under W_Q and beside LN(x)."""
        Tm, T = self.Tm, self.T
        pr = self.progress("proj", t)
        lw = op * smoothstep(Tm["fly"][1] - 0.6, Tm["fly"][1], t) * \
            (1 - smoothstep(Tm["kv"][1] - 0.6, Tm["kv"][1], t))
        rowq = min(T - 1, int(pr * T)) if 0 < pr < 1 else None
        hl_ln = None if rowq is None else _row_hl(T, G.D_MODEL, rowq, 0.8)
        self.ln.draw(self, sc, lw, highlight=hl_ln)
        self.wq.draw(self, sc, lw, highlight=0.25 if rowq is not None else None)
        q_done = _rows_done(pr * T, T, G.D_HEAD)
        al = np.where(q_done >= 1, 1.0, 0.15 + 0.85 * q_done).astype(np.float32)
        tex = np.where(q_done[..., None] > 0, card["Q"].tex, OFF).astype(np.float32)
        card["Q"].draw(self, sc, op, tex=tex if pr < 1 else None, alpha=al if pr < 1 else None,
                       highlight=_row_hl(T, G.D_HEAD, rowq, 0.5) if rowq is not None else
                       self._qk_hl(t, "Q"))
        if rowq is not None:          # a row of LN(x) streams up the columns of W_Q
            u = (pr * T) % 1.0
            sc.probe(("proj",), (u * 60, 0, 0), tol=6.0)

    def _dive_kv(self, sc, t, card, op):
        """K and V fill in row by row (K unflipped until the transpose)."""
        T = self.T
        kv = self.progress("kv", t)
        if kv <= 0:
            return
        on = op * smoothstep(0.0, 0.1, kv)
        if self.progress("transpose", t) <= 0:
            tex, al = _reveal(self.k_plain, _rows_done(kv * 1.4 * T, T, G.D_HEAD))
            self.k_plain.draw(self, sc, on, alpha=al, tex=tex)
        tex, al = _reveal(card["V"], _rows_done((kv - 0.25) * 1.4 * T, T, G.D_HEAD))
        card["V"].draw(self, sc, on, alpha=al, tex=tex, highlight=self._av_hl(t))

    def _dive_flip(self, sc, t, card, op):
        """The transpose: K turns about its diagonal, in place, into K^T."""
        th = math.pi * ease_in_out(remap(self.progress("transpose", t), 0.1, 0.8))
        e1 = rotate(self.k_plain.e1, DIAG, th)
        e2 = rotate(self.k_plain.e2, DIAG, th)
        if th < math.pi - 1e-6:
            self.k_plain.draw(self, sc, op, e1=e1, e2=e2, lift=0.0)
            sc.probe(("flip",), self.Z(self.k_plain.origin + e1 + e2), tol=4.0)
        else:
            card["KT"].draw(self, sc, op, highlight=self._qk_hl(t, "KT"))

    def _current_cell(self, t):
        """``((i, j), progress 0..1, slow)`` of the score being computed at ``t``.

        ``(None, 0.0, False)`` between cells or outside the "qk" phase.
        """
        for (i, j), t0, d, slow in self.qk_cells:
            if t0 <= t < t0 + d:
                return (i, j), (t - t0) / d, slow
        return None, 0.0, False

    def _qk_hl(self, t, which):
        """Highlight the row of Q (``which="Q"``) / column of K^T being multiplied."""
        if not self.Tm["qk"][0] <= t < self.Tm["qk"][1]:
            return None
        (cell, u, slow) = self._current_cell(t)
        if cell is None:
            return None
        i, j = cell
        k = 0.55 * smoothstep(0.0, 0.08, u) if slow else 0.45
        if which == "Q":
            return _row_hl(self.T, G.D_HEAD, i, k)
        h = np.zeros((G.D_HEAD, self.T), np.float32)
        h[:, j] = k
        return h

    def _av_hl(self, t):
        """Highlight V's rows by the weights of the A·V row being computed."""
        av = self.progress("av", t)
        if not 0 < av < 1:
            return None
        i = min(self.T - 1, int(av * self.T))
        w = self.tr.attn[LAYER, HEAD, i]
        return (np.repeat(w[:, None], G.D_HEAD, 1) * 0.7).astype(np.float32)

    # -- the scores --------------------------------------------------------------------------------
    def _scores(self, sc, t, card, op):
        """Q K^T: the scores fill in, the mask, then softmax row by row."""
        if self.progress("softmax", t) > 0:
            self._scores_softmax(sc, t, card, op)
            return
        self._scores_fill(sc, t, card, op)
        cur, u, slow = self._current_cell(t)
        if cur is not None and slow:
            self._slow_cell(sc, card, op, *cur, u)
            self.subtitle = self._slow_cell_subtitle(*cur, u)

    def _scores_softmax(self, sc, t, card, op):
        """Scores turn into attention weights row by row; then A is highlighted
        row by row during A·V."""
        T = self.T
        sm = self.progress("softmax", t)
        Sc, Ac = card["S"].tex, card["A"].tex
        m = np.clip(sm * 1.25 * T - np.arange(T), 0, 1)[:, None, None] * np.ones((1, T, 1))
        col = Sc + (Ac - Sc) * m
        col[np.triu(np.ones((T, T), bool), 1)] = OFF
        i = min(T - 1, int(sm * 1.25 * T))
        hl = _row_hl(T, T, i, 0.4) if sm < 0.8 else None
        card["S"].draw(self, sc, op, tex=col.astype(np.float32), highlight=hl)
        if sm >= 1 and t < self.Tm["av"][1]:
            av = self.progress("av", t)
            if 0 < av < 1:
                hl = _row_hl(T, T, min(T - 1, int(av * T)), 0.5)
                card["A"].draw(self, sc, op, highlight=hl, lift=0.5)

    def _scores_fill(self, sc, t, card, op):
        """The computed scores so far; then the causal mask darkens the upper triangle."""
        T = self.T
        masked = np.triu(np.ones((T, T), bool), 1)
        mk = self.progress("mask", t)
        done = np.zeros((T, T), bool)
        for (i, j), t0, d, _s in self.qk_cells:
            done[i, j] = t >= t0 + d
        if t >= self.Tm["qk"][1]:
            done[:] = True
        col = np.where(done[..., None], card["S"].tex, OFF).astype(np.float32)
        if mk > 0:
            k = smoothstep(0.0, 0.7, mk)
            col[masked] = col[masked] + (np.float32(OFF) - col[masked]) * k
        al = np.where(done, 1.0, 0.15).astype(np.float32)
        hl = None
        if mk > 0 and mk < 1:
            hl = (masked * 0.5 * bump(mk - 0.35, 0.3)).astype(np.float32)
        card["S"].draw(self, sc, op, tex=col, alpha=al, highlight=hl)

    def _slow_cell(self, sc, card, op, i, j, u):
        """Score (i, j) in full, ``u`` 0..1: row i of Q turns (a transpose of one
        row) to stand beside column j of K^T; their element-wise products
        collapse into the score's cell."""
        tp = self.tp
        cx, cz = self.cq[0], self.cq[2]
        row = self._rows(card["Q"].tex[i:i + 1], (cx, 0, cz + Q_AT[1] - i * tp))
        go = ease_in_out(remap(u, 0.08, 0.38))
        th = math.pi * go
        dest_q = np.array([cx + KT_AT[0] - 2 * tp - GAP, 0, cz + KT_AT[1]])  # left of K^T
        dest_k = dest_q + np.array([tp + 2, 0, 0])
        kcol = Panel(card["KT"].tex[:, j:j + 1], (cx + KT_AT[0] + j * tp, 0, cz + KT_AT[1]),
                     cell=(1.0, tp))
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
            return
        tr = self.tr
        prod = tr.q[LAYER, HEAD, i] * tr.k[LAYER, HEAD, j]
        pp = Panel(diverging(prod[:, None]), mid, cell=(1.0, tp))
        sh = ease_in_out(remap(u, 0.68, 0.92))
        tgt = np.array([cx + S_AT[0] + j * tp, 0, cz + S_AT[1] - i * tp])
        o = mid + (tgt - mid) * sh
        e2 = np.array([0, 0, -(G.D_HEAD * (1 - sh) + tp * sh)])
        tex = pp.tex if sh < 0.6 else np.broadcast_to(card["S"].tex[i, j], pp.tex.shape)
        pp.draw(self, sc, op, origin=o, e2=e2, lift=30, tex=np.ascontiguousarray(tex))
        sc.probe(("prod",), self.Z(o + e2), tol=4.0)

    def _slow_cell_subtitle(self, i, j, u):
        """What the worked score (i, j) is doing at progress ``u``."""
        d = G.D_HEAD
        if u < 0.42:
            return ("row %d of Q turns to lie beside column %d of K^T (a transpose of "
                    "one row)" % (i + 1, j + 1))
        if u < 0.66:
            return "multiply element by element: %d products q%d[d] · k%d[d]" % (d, i + 1, j + 1)
        s = self.tr.scores[LAYER, HEAD, i, j]
        return ("sum the %d products: q%d · k%d = %.1f;  ÷ √%d = %.2f  =  score[%d, %d]"
                % (d, i + 1, j + 1, s * math.sqrt(d), d, s, i + 1, j + 1))

    # -- concat, W_O, residual add, MLP for the dive layer -----------------------------------------
    def _after_heads(self, sc, t, back, model, pulse_x):
        """The dive layer after its heads: concat, W_O, the add, the MLP, the add."""
        L, Tm = LAYER, self.Tm
        if t < Tm["concat"][0] or back <= 0.01:
            # before the layer view: show nothing of the rest of this layer
            # (during the fly-in the full model is visible: draw it complete)
            if model > 0.01:
                for P in (self.concat[L], self.attn_o[L], self.mlp_h[L], self.mlp_o[L],
                          self.seg_mid[L], self.seg_out[L]):
                    P.draw(self, sc, model * (t < Tm["fly"][1]))
            return
        self._concat_heads(sc, t, back, pulse_x)
        self._project_add(sc, t, back, pulse_x)
        self._mlp(sc, t, back, pulse_x)

    def _concat_heads(self, sc, t, back, pulse_x):
        """The 12 head outputs fly to their slots in the concatenation."""
        L, T = LAYER, self.T
        ct = self.progress("concat", t)
        slot0 = layer_x(L) + SEG_MID_X
        arrived = np.zeros((T, G.D_MODEL), np.float32)
        for h in range(G.N_HEAD):
            u = ease_in_out(remap(ct, h * 0.04, h * 0.04 + 0.5))
            if u >= 1:
                arrived[:, h * G.D_HEAD:(h + 1) * G.D_HEAD] = 1
            elif u > 0:
                O = self.cards[L][h]["O"]
                o = O.origin + (np.array([slot0 + h * G.D_HEAD, 0, WIDE_Z]) - O.origin) * u
                O.draw(self, sc, back, origin=o, lift=40 * math.sin(math.pi * u))
                if h == 0:
                    sc.probe(("cat",), self.Z(o), tol=4.0)
        self.concat[L].draw(self, sc, back, alpha=np.maximum(arrived, 0.15) if ct < 1 else None,
                            tex=None, highlight=self._hl(self.concat[L], pulse_x))

    def _project_add(self, sc, t, back, pulse_x):
        """× W_O row by row, then x + attention(x): both slide into the stream."""
        L, T = LAYER, self.T
        wo, a1 = self.progress("wo", t), self.progress("add1", t)
        rows = _rows_done(wo * T * 1.2, T, G.D_MODEL)
        self.attn_o[L].draw(self, sc, back * (wo > 0),
                            alpha=(0.15 + 0.85 * rows).astype(np.float32) if wo < 1 else None,
                            highlight=self._hl(self.attn_o[L], pulse_x))
        if 0 < a1 < 1:
            P = self.attn_o[L]
            o = P.origin + (self.seg_mid[L].origin - P.origin) * ease_in_out(a1)
            P.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a1)), origin=o, lift=40)
            Pi = self.seg_in[L]
            o2 = Pi.origin + (self.seg_mid[L].origin - Pi.origin) * ease_in_out(a1)
            Pi.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a1)), origin=o2, lift=30)
            sc.probe(("add1",), self.Z(o2), tol=4.0)
        self.seg_mid[L].draw(self, sc, back * smoothstep(0.8, 1.0, a1),
                             highlight=self._hl(self.seg_mid[L], pulse_x))

    def _mlp(self, sc, t, back, pulse_x):
        """The MLP's hidden rows then its output rows, then the second add."""
        L, T = LAYER, self.T
        ml = self.progress("mlp", t)
        hr = _rows_done(ml * 2.2 * T, T, G.D_MLP)
        mr = _rows_done((ml - 0.35) * 2.2 * T, T, G.D_MODEL)
        self.mlp_h[L].draw(self, sc, back * (ml > 0),
                           alpha=(0.15 + 0.85 * hr).astype(np.float32) if ml < 1 else None,
                           highlight=self._hl(self.mlp_h[L], pulse_x))
        self.mlp_o[L].draw(self, sc, back * (ml > 0.35),
                           alpha=(0.15 + 0.85 * mr).astype(np.float32) if ml < 1 else None,
                           highlight=self._hl(self.mlp_o[L], pulse_x))
        a2 = remap(ml, 0.8, 1.0)
        if 0 < a2 < 1:
            for P, lift in ((self.mlp_o[L], 40), (self.seg_mid[L], 30)):
                o = P.origin + (self.seg_out[L].origin - P.origin) * ease_in_out(a2)
                P.draw(self, sc, back * (1 - smoothstep(0.8, 1.0, a2)), origin=o, lift=lift)
        self.seg_out[L].draw(self, sc, back * smoothstep(0.8, 1.0, a2),
                             highlight=self._hl(self.seg_out[L], pulse_x))

    # -- output ------------------------------------------------------------------------------------
    def _output(self, sc, t, back, model, pulse_x):
        """Final LayerNorm, the logits and the top-5 next-token bars."""
        on = max(model * (t < self.Tm["fly"][1]), smoothstep(*self.Tm["wide"], t)) * back
        self.lnf.draw(self, sc, on, highlight=self._hl(self.lnf, pulse_x))
        pr = self.progress("predict", t)
        self.logits.draw(self, sc, back * smoothstep(0.0, 0.15, pr))
        if pr <= 0:
            return
        for n, (tok, p) in enumerate(self.tr.top):
            self._bar(sc, n, tok, p, pr, back)
        sc.probe(("bars",), (pr * 40, 0, 0), tol=4.0)

    def _bar(self, sc, n, tok, p, pr, back):
        """Bar ``n`` (token ``tok``, probability ``p``) growing at predict progress ``pr``."""
        x = OUT_X + BARS_DX + n * BAR_PITCH
        g = ease_out(remap(pr, 0.15 + 0.08 * n, 0.55 + 0.08 * n))
        h = max(4.0, BAR_H * p * g)
        col = np.float32(ACT_HI if n == 0 else ACT_LO)
        lo = self.Z((x, 0, 0))
        a1 = np.array([BAR_W, 0, 0]) * self.s
        a2 = np.array([0, 0, h]) * self.s
        glow = 0.6 if n == 0 else 0.0
        sc.quad(lo, a1, a2, lambda P, c=col, gl=glow: (
            np.broadcast_to(c, P.shape).copy(), np.full(P.shape[:2], back, np.float32),
            np.full(P.shape[:2], gl, np.float32)))
        op = back * self._cap * smoothstep(0.15, 0.3, pr)
        fx, fy = self.at_px((x + BAR_W / 2, 0, -14))
        sc.text(fx, fy, tok.strip() or repr(tok), layout.CAPTION, "mono",
                layout.STRONG_INK if n == 0 else layout.CAPTION_INK, op, "mt")
        fx, fy = self.at_px((x + BAR_W / 2, 0, h + 14))
        sc.text(fx, fy, "%.0f%%" % (100 * p * g if g < 1 else 100 * p), layout.MIN_TEXT,
                "regular", layout.CAPTION_INK, op, "mb")

    # -- captions ----------------------------------------------------------------------------------
    def _captions(self, sc, t, back):
        """Every label, shot by shot (their order is their drawing order)."""
        self._cap_embed(sc, t, back)
        self._cap_dive(sc, t, back)
        self._cap_layer(sc, t, back)
        self._cap_model(sc, t, back)
        self._cap_output(sc, t, back)

    def _cap_embed(self, sc, t, back):
        """The embedding tables and x."""
        Tm, cap = self.Tm, self.caption
        e = self.progress("embed", t)
        before_fly = 1 - smoothstep(Tm["fly"][0], Tm["fly"][0] + 0.6, t)
        cap(sc, self.zbox(self.tok_e), "token embeddings: each token's row of a %s × %d table"
            % (_thousands(G.N_VOCAB), G.D_MODEL), back * smoothstep(0.0, 0.15, e) * before_fly,
            above=True, max_width=0.6)
        cap(sc, self.zbox(self.pos_e), "+ position embeddings (one row per position, 0 to %d)"
            % (self.T - 1), back * smoothstep(0.45, 0.6, e) * before_fly, above=True,
            max_width=0.6)
        cap(sc, self.zbox(self.x0), "x: %d tokens × %d numbers, the residual stream"
            % (self.T, G.D_MODEL), back * smoothstep(0.85, 1.0, e) * before_fly, max_width=0.6)

    def _cap_dive(self, sc, t, back):
        """LN(x) and W_Q, the head's matrix names, and the token numbers."""
        Tm, cap = self.Tm, self.caption
        card = self.cards[LAYER][HEAD]
        dive = smoothstep(Tm["fly"][1] - 0.4, Tm["fly"][1] + 0.2, t) * \
            (1 - smoothstep(Tm["heads"][0], Tm["heads"][0] + 0.8, t)) * back
        proj = dive * (1 - smoothstep(Tm["kv"][1] - 0.6, Tm["kv"][1], t))
        cap(sc, self.zbox(self.ln), "LN(x): %d × %d" % (self.T, G.D_MODEL), proj, above=True)
        cap(sc, self.zbox(self.wq), "W_Q for head %d: %d × %d" % (HEAD + 1, G.D_MODEL, G.D_HEAD),
            proj, above=True)
        self._label(sc, card["Q"], "Q", dive, "below")
        if t >= Tm["kv"][0]:
            kt = t >= Tm["transpose"][0] + 0.5 * (Tm["transpose"][1] - Tm["transpose"][0])
            on = dive * smoothstep(Tm["kv"][0], Tm["kv"][0] + 0.3, t)
            self._label(sc, card["KT"] if kt else self.k_plain, "Kᵀ" if kt else "K", on,
                        "above",
                        gap=30 if t >= Tm["transpose"][1] and self.px(self.tp) > 14 else 6)
            self._label(sc, card["V"], "V", on, "above")
            self._label(sc, card["S"], "A" if t >= Tm["softmax"][0] else "scores", dive, "below")
            self._label(sc, card["O"], "A·V", dive * (t >= Tm["av"][0]), "below")
        close = dive * smoothstep(Tm["close"][0], Tm["close"][1], t) * self._rows_legible()
        if close > 0.01:
            self._cap_token_numbers(sc, t, close)

    def _cap_token_numbers(self, sc, t, op):
        """Token numbers beside the rows of Q / A and above the columns of K^T."""
        cx, cz, tp = self.cq[0], self.cq[2], self.tp
        for i, tok in enumerate(self.tr.tokens):
            fx, fy = self.at_px((cx - GAP, 0, cz + Q_AT[1] - (i + 0.5) * tp))
            sc.text(fx, fy, '%d "%s"' % (i + 1, tok.strip() or tok), layout.MIN_TEXT,
                    "mono", layout.CAPTION_INK, op * self._cap, "rm")
            if t >= self.Tm["transpose"][1]:
                fx, fy = self.at_px((cx + KT_AT[0] + (i + 0.5) * tp, 0, cz + KT_AT[1] + GAP))
                sc.text(fx, fy, "%d" % (i + 1), layout.MIN_TEXT, "mono", layout.NOTE_INK,
                        op * self._cap, "mb")

    def _cap_layer(self, sc, t, back):
        """The layer view: heads, concat, W_O, the MLP and the stream's three states."""
        Tm, cap, L = self.Tm, self.caption, LAYER
        lay = smoothstep(Tm["heads"][1] - 0.6, Tm["heads"][1], t) * \
            (1 - smoothstep(Tm["wide"][0], Tm["wide"][0] + 0.8, t)) * back
        if lay <= 0.01:
            return
        m, T, d, H = layer_x(L), self.T, G.D_MODEL, G.N_HEAD
        ml = self.progress("mlp", t)
        cap(sc, self.zbox(np.array([m + GRID_X, 0, CARDS_Z]), np.array([m + GRID_RIGHT, 0,
                                                                         GRID_TOP])),
            "%d attention heads, all at once" % H, lay, above=True)
        cap(sc, self.zbox(self.concat[L]), "concatenate: %d × (%d × %d)" % (T, H, G.D_HEAD),
            lay * (self.progress("concat", t) > 0), above=True)
        cap(sc, self.zbox(self.attn_o[L]), "× W_O (%d × %d)" % (d, d),
            lay * (self.progress("wo", t) > 0) * (1 - bump(self.progress("add1", t) - 0.5, 0.3)))
        cap(sc, self.zbox(self.mlp_h[L]), "MLP: %d to %d, GELU" % (d, G.D_MLP), lay * (ml > 0),
            above=True)
        cap(sc, self.zbox(self.mlp_o[L]), "%d to %d" % (G.D_MLP, d), lay * (ml > 0.35))
        cap(sc, self.zbox(self.seg_in[L]), "stream in", lay)
        cap(sc, self.zbox(self.seg_mid[L]), "+ attention", lay * (self.progress("add1", t) > 0.8))
        cap(sc, self.zbox(self.seg_out[L]), "+ MLP", lay * (ml > 0.95))

    def _cap_model(self, sc, t, back):
        """The wide shots: "layer n" over each layer, "embed", "output"."""
        Tm, cap = self.Tm, self.caption
        end = self.t_done + self.hold
        wide = smoothstep(Tm["wide"][1] - 0.8, Tm["wide"][1], t) * \
            (1 - smoothstep(Tm["out"][0], Tm["out"][0] + 0.6, t))
        wide = max(wide, smoothstep(end + 1.2, end + 1.7, t) *
                   (1 - smoothstep(self.duration - 1.4, self.duration - .8, t))) * back
        if wide <= 0.01:
            return
        for l in range(G.N_LAYER):
            m = layer_x(l)
            cap(sc, self.zbox(np.array([m + GRID_X, 0, STREAM_Z]),
                              np.array([m + SEG_OUT_X + STREAM_W, 0, GRID_TOP])),
                ("layer %d" if self.px(MOD) > 180 else "%d") % (l + 1), wide, above=True)
        cap(sc, self.zbox(self.x0), "embed", wide)
        cap(sc, self.zbox(self.lnf), "output", wide)

    def _cap_output(self, sc, t, back):
        """The final LayerNorm and the logits."""
        pr = self.progress("predict", t)
        if pr <= 0:
            return
        op = back * smoothstep(0.0, 0.2, pr)
        self.caption(sc, self.zbox(self.lnf), "final LayerNorm", op)
        self.caption(sc, self.zbox(self.logits), "× W_E^T: %s logits (last token)"
                     % _thousands(G.N_VOCAB), op, above=True)

    def _label(self, sc, P, text, op, where="below", size=layout.CAPTION, gap=6):
        """A matrix's name centred ``gap`` px above or below panel ``P`` on screen."""
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
            sc.text(x, (s[:, 1].max() + gap) / self.cam.height, text, size, "mono",
                    layout.STRONG_INK, op, "mt")

    # -- subtitles ---------------------------------------------------------------------------------
    def _subtitle_table(self):
        """``[(until, text or t -> text)]``: each line shows until ``until`` s
        (``None``: to the end)."""
        T, d, dh = self.T, G.D_MODEL, G.D_HEAD
        nl, nh = G.N_LAYER, G.N_HEAD
        start = {k: a for k, (a, _b) in self.Tm.items()}
        return [
            (start["embed"], 'the text "%s" is split into %d tokens' % (self.tr.text, T)),
            (start["fly"], "each token's embedding + its position's embedding: one row of %d "
                           "per token" % d),
            (self.Tm["fly"][1], "GPT-2 small: %d layers, each with %d attention heads. Fly "
                                "into layer %d, head %d" % (nl, nh, LAYER + 1, HEAD + 1)),
            (start["kv"], "queries: Q = LN(x) · W_Q. Each row of LN(x) times the %d × %d "
                          "weights gives one row of Q" % (d, dh)),
            (start["transpose"], "keys and values the same way: K = LN(x) · W_K,  "
                                 "V = LN(x) · W_V"),
            (start["qk"], "the transpose: K flips over its diagonal, so its %d rows become "
                          "the columns of K^T" % T),
            (start["mask"], "scores = Q · K^T ÷ √%d: one dot product for every pair of "
                            "tokens" % dh),
            (start["softmax"], "causal mask: a token may not look ahead, so later tokens' "
                               "scores become %s∞" % MINUS),
            # true of layer 5, head 12: GPT-2's previous-token head
            (start["av"], "softmax along each row: the weights sum to 1. This head looks at "
                          "the previous token"),
            (start["heads"], self._av_subtitle),
            (start["concat"], "this was 1 of %d heads in layer %d, all computed at the same "
                              "time" % (nh, LAYER + 1)),
            (start["wo"], "the %d head outputs (%d × %d each) are concatenated: %d × %d"
                          % (nh, T, dh, T, nh * dh)),
            (start["add1"], "projected by W_O (%d × %d)" % (d, d)),
            (start["mlp"], "and added to the residual stream: x + attention(x)"),
            (start["wide"], "then the MLP, applied to each token separately (GELU: only > 0 "
                            "light up), added to the stream"),
            (start["full"], "every layer reads the stream and adds to it: attention mixes "
                            "information between tokens, the MLP works on each token"),
            (start["out"], "%d layers × %d heads = %d attention heads; every layer adds to "
                           "the stream" % (nl, nh, nl * nh)),
            (start["predict"], "after layer %d: the final LayerNorm" % nl),
            (None, self._predict_subtitle),
        ]

    def _subtitle(self, t):
        """The subtitle at ``t`` (worked scores set their own in :meth:`_scores`)."""
        for until, text in self.subtitles:
            if until is None or t < until:
                return text(t) if callable(text) else text
        return ""

    def _av_subtitle(self, t):
        """A·V: the row being computed as its two biggest weights times value rows."""
        i = min(self.T - 1, int(self.progress("av", t) * self.T))
        w = self.tr.attn[LAYER, HEAD, i]
        top = [j for j in np.argsort(-w)[:2] if w[j] > 0.01]
        mix = " + ".join("%s · v%d" % (num(w[j]), j + 1) for j in top)
        return "A · V: output row %d = %s, a weighted mix of the value rows" % (i + 1, mix)

    def _predict_subtitle(self, t):
        """The logits and the top next token."""
        top, p = self.tr.top[0]
        return 'the last token\'s row × W_E^T gives %s logits; softmax: "%s" %.0f%%' % (
            _thousands(G.N_VOCAB), top, 100 * p)
