"""Attention between two sequences: an encoder-decoder reversing six digits.

The model is :mod:`voxanim.seq2seq` (a GRU encoder, a GRU decoder, additive
attention), trained to read ``3 7 2 9 4 1`` and say ``1 4 9 2 7 3``.  The
picture is a table whose rows are the decoder's steps, under the encoder:

    input x            3 7 2 9 4 1          (one teal voxel per token)
    encoder h          12 x 6 band          (one column per token, as recurrent.py)

    context c  |  alignment a  |  decoder s  |  output
    8 cells    |  6 x 6 cells  |  12 cells   |  a digit        ... one row per step

The alignment's columns sit exactly under the encoder's, so a row of it
reads "how much of each input this step looked at".

Five phases:

1. the task,
2. encode -- the GRU reads the tokens left to right; the states appear
   column by column and the state carries along the band,
3. decode -- at each step the decoder's state queries the encoder states:
   the row of attention weights appears in the alignment, particles fly
   from the input tokens to the context row weighted by those weights (the
   context *is* that mix), the decoder state updates and the output digit
   appears,
4. alignment -- the finished matrix: the anti-diagonal, each output reading
   the input it copies,
5. training -- the recorded run replays on a log time axis: the matrix is
   flat before training and the output wrong; the anti-diagonal forms as
   the held-out accuracy climbs to 100%.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep, soft_band
from .netanim import NetAnimation, amax, div_colors, particle, screen_box
from .text import MINUS, sub

PPU = 16.0                         # pixels per voxel
TOP, CENTRE_X = 0.26, 0.50         # frame y of the picture's top (under the digits), its centre x
COL_PITCH = 1.5                    # cell pitch along the input tokens (room for digit labels)
ROW_PITCH = 1.6                    # cell pitch down the decoder steps (room for output digits)
GAP_ENC = 1.8                      # world units between the matrix and the encoder band
GAP_IN = 0.6                       # between the encoder band and the input row
GAP_SIDE = 1.8                     # between the matrix and the context / decoder sheets
ATT_LO, ATT_HI = C.rgb("#3a3f44"), C.rgb("#c9fff6")   # attention weight 0 .. 1 (a heat map)
WRONG_INK = (0.75, 0.22, 0.22)     # an output digit that is not the target
ALPHA_FLOOR = 0.02                 # attention weights below this send no particle
REPLAY_BLEND = 0.06                # fraction of the training phase spent fading to step 0
REPLAY_END = 0.88                  # ... and the fraction by which it has reached the last step
NOTE_X, NOTE_Y = 0.945, 0.80       # frame position (right edge) of the training read-outs


def attention_colors(alpha):
    """Attention weights in 0..1 as a heat map: dark for nothing, lit for all of it."""
    a = np.clip(np.asarray(alpha, np.float64).ravel(), 0, 1)[:, None] ** 1.3
    return (ATT_LO[None] + (ATT_HI - ATT_LO)[None] * a).astype(np.float32)


class Seq2SeqAttention(NetAnimation):
    """The :func:`voxanim.seq2seq.example` through the trained encoder-decoder.

    ``tr`` is a :class:`voxanim.seq2seq.Trace`; its ``run`` holds the recorded
    training (the alignment and the output after every step).
    """

    PHASES = ("task", "encode", "decode", "alignment", "training")

    def __init__(self, tr):
        super().__init__("Attention between two sequences", "", PPU)
        self.tr = tr
        self.L, self.n, self.d = tr.alpha.shape[0], tr.h.shape[1], tr.c.shape[1]
        self.N = len(tr.run.losses)
        self.hold = 2.5
        L, n, d = self.L, self.n, self.d
        # ---- sheets, placed in world space around the matrix -------------------------
        at = (0.5, 0.5)                                   # placed properly by _fit
        self.matrix = self.sheet((L, L), at, 1.0, "alpha", pitch=(ROW_PITCH, COL_PITCH))
        self.enc = self.sheet((n, L), at, 1.0, "h", pitch=(1, COL_PITCH))
        self.inp = self.sheet((1, L), at, 1.0, "x", pitch=(1, COL_PITCH))
        self.ctx = self.sheet((L, d), at, 1.0, "c", pitch=(ROW_PITCH, 1))
        self.dec = self.sheet((L, n), at, 1.0, "s", pitch=(ROW_PITCH, 1))
        m = self.matrix.block
        self.enc.block.origin = m.origin + np.array([0.0, 0.0, m.extent[2] + GAP_ENC])
        e = self.enc.block
        self.inp.block.origin = e.origin + np.array([0.0, 0.0, e.extent[2] + GAP_IN])
        self.ctx.block.origin = m.origin - np.array([self.ctx.block.extent[0] + GAP_SIDE, 0, 0])
        self.dec.block.origin = m.origin + np.array([m.extent[0] + GAP_SIDE, 0.0, 0.0])
        self._fit()
        # ---- values ----------------------------------------------------------------------
        self.h_max, self.c_max, self.s_max = amax(tr.h), amax(tr.c), amax(tr.s)
        self.anti = float(np.mean([tr.alpha[j, L - 1 - j] for j in range(L)]))

        # ---- timeline ----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, dur, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + dur)
            t += dur + gap

        add("task", 3.2)
        enc = Tempo(slow=1.3, fast=0.55, ramp_in=2.0, ramp_out=1.0).durations(L)
        self.e_starts, self.e_durs = t + np.concatenate([[0.0], np.cumsum(enc)[:-1]]), enc
        add("encode", float(enc.sum()), 0.5)
        dec = Tempo(slow=3.4, fast=2.1, ramp_in=2.0, ramp_out=1.0).durations(L)
        self.d_starts, self.d_durs = t + np.concatenate([[0.0], np.cumsum(dec)[:-1]]), dec
        add("decode", float(dec.sum()), 0.5)
        add("alignment", 4.0)
        add("training", 9.0)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["encode"][0], Tm["decode"][0], Tm["alignment"][0],
                      Tm["training"][0]]

    def _fit(self):
        """Slide the whole picture so it is centred at :data:`CENTRE_X` with its
        top (the input row) at :data:`TOP`, whatever the sizes."""
        cam = self.cam
        boxes = [screen_box(cam, *s.bbox()) for s in self.sheets]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        want = np.array([CENTRE_X * cam.width, TOP * cam.height])
        have = np.array([(x0 + x1) / 2, y0])
        delta = cam.unproject(*want, 0.0) - cam.unproject(*have, 0.0)
        for s in self.sheets:
            s.block.origin = s.block.origin + delta

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """A step sweep at time ``t``: ``(steps finished, current step, its
        progress)``; the current step is -1 before the sweep, ``L`` after."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return self.L, self.L, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    def _replay(self, t):
        """The training phase's state at ``t``, or None before it: ``k`` (the
        step shown), ``w`` (how far the picture has faded from the trained
        model to the recorded one), ``alpha``, ``pred`` and ``acc``."""
        u = self.progress("training", t)
        if u <= 0:
            return None
        run = self.tr.run
        k = int(round(self.N ** smoothstep(0.0, REPLAY_END, u)))
        k = 0 if u < REPLAY_BLEND else k
        w = smoothstep(0.0, REPLAY_BLEND, u) if u < 0.5 else 1.0 - smoothstep(REPLAY_END, 1.0, u)
        return dict(k=k, w=w, alpha=run.alpha[k], pred=run.pred[k], acc=run.accuracy_at(k))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        tr, L = self.tr, self.L
        ph = 0 if reset > 0 else self.step_of(t)
        digits = " ".join(str(v) for v in tr.x)
        rev = " ".join(str(v) for v in tr.y)
        run = tr.run
        self.subtitle = [
            "the task: read %s and say it backwards, %s: every output token copies one input "
            "token" % (digits, rev),
            "encoder: a GRU reads the tokens one by one; each state h%s knows what it has read "
            "and where it is" % sub("t"),
            "decoder: at each step the attention weights over the encoder states pick which "
            "input to read; the context carries that token",
            "alignment, one row per output: output j reads input %d%sj, the anti-diagonal; "
            "%.2f of each row's weight is on the token it copies" % (L - 1, MINUS, self.anti),
            "training: the alignment is learned; %d Adam steps take it from flat (%d%% of tokens "
            "right) to the anti-diagonal (%d%%)" % (self.N, round(100 * run.accuracy_at(0)),
                                                    round(100 * run.accuracy_at(self.N))),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        task = self.progress("task", t)
        done_e, ne, ue = self._sweep(t, "encode", self.e_starts, self.e_durs)
        done_d, nd, ud = self._sweep(t, "decode", self.d_starts, self.d_durs)
        live = self._replay(t)
        self._input(sc, t, task, ne, ue, nd, ud, back)
        self._encoder(sc, t, done_e, ne, ue, nd, ud, back)
        self._decoder(sc, t, done_d, nd, ud, live, back)
        self._training_notes(sc, live, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the input tokens --------------------------------------------------------------
    def _input(self, sc, t, task, ne, ue, nd, ud, back):
        """The input row: teal voxels with their digits above, read one at a time
        by the encoder, then sending their embeddings to the contexts."""
        s, tr, L = self.inp, self.tr, self.L
        col = np.arange(L)
        shown = col < task * (L + 1) * 1.1 if t < self.Tm["encode"][0] else np.ones(L, bool)
        s.fill(np.tile(C.INPUT, (L, 1)))
        s.show(shown)
        s.cage = 1.0
        s.fade(back)
        reading = ((col == ne) & (ue < 0.4)) * 0.6
        if 0 <= nd < L:                                   # attended: lit by its weight
            reading = reading + tr.alpha[nd] * 0.7 * soft_band(ud, 0.22, 0.5, 0.08)
        s.block.white[0, :, 0] = np.clip(reading, 0, 1)
        self.caption_left(sc, s, "input x", back)
        for j in range(L):
            if shown[j]:
                p = s.cell(j)
                h = 0.5
                self.caption(sc, (p - h, p + h), str(tr.x[j]), back, size=0.03, above=True,
                             ink=layout.STRONG_INK)

    # -- the encoder band ----------------------------------------------------------------
    def _encoder(self, sc, t, done_e, ne, ue, nd, ud, back):
        """The encoder states, column by column, and what flows into them."""
        s, tr, L, n = self.enc, self.tr, self.L, self.n
        col = np.tile(np.arange(L), n)
        shown = col < done_e
        if 0 <= ne < L and ue >= 0.55:
            shown |= col == ne
        if t >= self.Tm["encode"][1]:
            shown[:] = True
        s.fill(div_colors(tr.h.T.ravel(), self.h_max))
        s.show(shown)
        s.cage = 1.0
        s.fade(back)
        white = ((col == ne) & (ue < 1)) * (1 - smoothstep(0.7, 1.0, ue)) * 0.6
        if 0 <= nd < L:                                   # the keys, lit by their weight
            white = white + tr.alpha[nd][col] * 0.8 * soft_band(ud, 0.1, 0.3, 0.06)
        s.block.white[:, :, 0] = np.clip(white, 0, 1).reshape(n, L)
        self.caption_left(sc, s, "encoder h", back)
        if 0 <= ne < L and t < self.Tm["encode"][1]:
            self._column_frame(sc, ne)
            particle(sc, self.inp.cell(ne), s.cell(n // 2 * L + ne), remap(ue, 0.0, 0.35),
                     C.INPUT, 0.6, lift=0.0)
            if ne > 0:
                for r in range(n):
                    colr = div_colors(tr.h[ne - 1, r:r + 1], self.h_max)[0]
                    particle(sc, s.cell(r * L + ne - 1), s.cell(r * L + ne),
                             remap(ue, 0.1, 0.5), colr, 0.5, lift=0.4)
            sc.probe(("enc",), (ne * 20 + ue * 20, 0, 0), tol=10.0)

    def _column_frame(self, sc, col):
        """A white frame around input ``col`` through the input row and the band."""
        lo = np.min([self.inp.block.box((0, 1), (col, col + 1), (0, 1))[0],
                     self.enc.block.box((0, self.n), (col, col + 1), (0, 1))[0]], 0)
        hi = np.max([self.inp.block.box((0, 1), (col, col + 1), (0, 1))[1],
                     self.enc.block.box((0, self.n), (col, col + 1), (0, 1))[1]], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=(0.96, 0.96, 0.96), width=1.6, opacity=0.95,
                    bias=0.1)

    # -- the decoder's rows -------------------------------------------------------------
    def _decoder(self, sc, t, done_d, nd, ud, live, back):
        """Contexts, alignment rows, decoder states and output digits, one row per step."""
        tr, L, n, d = self.tr, self.L, self.n, self.d
        row = np.arange(L)
        done = row < done_d
        if t >= self.Tm["decode"][1]:
            done[:] = True
        cur = 0 <= nd < L and t < self.Tm["decode"][1]
        # the alignment: the trained rows, then the replayed matrix
        alpha = tr.alpha
        if live is not None:
            alpha = alpha + (live["alpha"] - alpha) * live["w"]
        a_shown = np.repeat(done | ((row == nd) & (ud >= 0.12) & cur), L)
        self.matrix.fill(attention_colors(alpha))
        self.matrix.show(a_shown)
        a_white = np.zeros((L, L))
        if cur:
            a_white[nd] = soft_band(ud, 0.12, 0.3, 0.06) * 0.5
        self.matrix.block.white[:, :, 0] = a_white
        self.matrix.cage = 1.0
        self.matrix.fade(back)
        # the context and decoder rows (the replay has none: they fade out)
        rows_op = back * (1.0 if live is None else 1.0 - live["w"])
        c_shown = np.repeat(done | ((row == nd) & (ud >= 0.5) & cur), d)
        self.ctx.fill(div_colors(tr.c.ravel(), self.c_max))
        self.ctx.show(c_shown)
        self.ctx.cage = 1.0
        self.ctx.fade(rows_op)
        s_shown = np.repeat(done | ((row == nd) & (ud >= 0.78) & cur), n)
        self.dec.fill(div_colors(tr.s.ravel(), self.s_max))
        self.dec.show(s_shown)
        self.dec.cage = 1.0
        self.dec.fade(rows_op)
        s_white = np.zeros((L, n))
        if cur and nd > 0:                                # the query: the previous state
            s_white[nd - 1] = soft_band(ud, 0.0, 0.2, 0.06) * 0.6
        self.dec.block.white[:, :, 0] = s_white
        # labels
        self.caption(sc, self.ctx, "context c", rows_op, size=0.027)
        self.caption(sc, self.matrix, "alignment α", back, size=0.027)
        self.caption(sc, self.dec, "decoder state s", rows_op, size=0.027)
        self._outputs(sc, done, nd, ud, cur, live, back)
        if cur:
            self._flows(sc, nd, ud)

    def _outputs(self, sc, done, nd, ud, cur, live, back):
        """The output digits right of the decoder rows (the replay's are the
        recorded ones, wrong ones in red)."""
        tr, L, n = self.tr, self.L, self.n
        pred = tr.pred if live is None else live["pred"]
        op = back * self._cap
        if op <= 0.01:
            return
        p0 = self.dec.cell(n - 1) + np.array([1.5, 0.0, 1.1])
        s = self.cam.project(p0)
        sc.text(s[0] / self.cam.width, s[1] / self.cam.height, "output", 0.027, "regular",
                layout.CAPTION_INK, op, "lb")
        for j in range(L):
            show = done[j] or (cur and j == nd and ud >= 0.9)
            if not show:
                continue
            p = self.dec.cell(j * n + n - 1) + np.array([1.5, 0.0, 0.0])
            s = self.cam.project(p)
            ink = layout.STRONG_INK if pred[j] == tr.y[j] else WRONG_INK
            sc.text(s[0] / self.cam.width, s[1] / self.cam.height, str(int(pred[j])), 0.031,
                    "regular", ink, op, "lm")

    def _flows(self, sc, j, u):
        """What moves in decoder step ``j`` at progress ``u``: the weighted input
        embeddings into the context, the context and previous state into the
        new state."""
        tr, L, n, d = self.tr, self.L, self.n, self.d
        for t_in in range(L):
            a = float(tr.alpha[j, t_in])
            if a < ALPHA_FLOOR:
                continue
            particle(sc, self.inp.cell(t_in), self.ctx.cell(j * d + d // 2),
                     remap(u, 0.25, 0.55), C.INPUT, 0.45 + 0.5 * a, opacity=0.35 + 0.65 * a,
                     lift=1.5)
        particle(sc, self.ctx.cell(j * d + d // 2), self.dec.cell(j * n + n // 2),
                 remap(u, 0.55, 0.8), div_colors(tr.c[j, d // 2:d // 2 + 1], self.c_max)[0],
                 0.6, lift=1.2)
        if j > 0:
            for r in range(n):
                colr = div_colors(tr.s[j - 1, r:r + 1], self.s_max)[0]
                particle(sc, self.dec.cell((j - 1) * n + r), self.dec.cell(j * n + r),
                         remap(u, 0.6, 0.8), colr, 0.5, lift=0.3)
        sc.probe(("dec",), (j * 30 + u * 30, 0, 0), tol=10.0)

    # -- the training replay's read-outs ------------------------------------------------
    def _training_notes(self, sc, live, back):
        """The step shown and the held-out accuracy at that step."""
        if live is None:
            return
        op = live["w"] * back * self._cap if live["k"] < self.N else back * self._cap
        if op <= 0.01:
            return
        x, y = NOTE_X, NOTE_Y
        sc.text(x, y, "after %d step%s" % (live["k"], "s" * (live["k"] != 1)), 0.03, "regular",
                layout.STRONG_INK, op, "rm")
        sc.text(x, y + 0.045, "held-out tokens right: %d%%" % round(100 * live["acc"]), 0.027,
                "regular", layout.CAPTION_INK, op, "rm")
        sc.probe(("train",), (np.log10(live["k"] + 1) * 40, 0, 0), tol=6.0)
