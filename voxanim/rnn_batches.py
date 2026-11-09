"""A 2-layer RNN training on batches of sine waves, with the model fixed in
the centre of the frame and the data streaming through it.

    input queue  ->  x -> [h1] -> [h2] -> y  ->  predictions vs targets
    (this batch)        loop     loop
                       Wx1  W12  Wy   (+ Wh1, Wh2 on the recurrent loops)

                         loss curve over training steps (log-log axes)

Each batch has its own colour and each sequence in it a shade of that
colour.  For every sequence, the values enter one time step at a time: x
feeds h1, h1 feeds h2 and itself (the loop), h2 feeds the prediction.  When a
batch is done, the gradients averaged over its sequences update the weights
once, and its loss joins the curve.  Batches 1-3 are shown in full, then
training fast-forwards to the last step, where the predictions now follow
the targets.

All numbers come from :func:`voxanim.sequence.rnn2_training`.
"""

from __future__ import annotations

import bisect

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep, soft_band
from .netanim import NetAnimation, amax, div_colors, particle
from .recurrent import GHOST
from .text import TIMES

# Batches 1-3 and the last take the first four; the fast-forward cycles the rest.
BATCH_COLORS = [C.FILTERS[5], C.FILTERS[0], C.FILTERS[3], C.FILTERS[6], C.FILTERS[1],
                C.FILTERS[7], C.FILTERS[4], C.FILTERS[2]]
LOSS_RANGE = (-2.8, -0.6)          # log10 of the loss at the bottom and top of the curve


class RNNBatchTraining(NetAnimation):
    """Batches streaming through a fixed 2-layer RNN; ``run`` is a
    :class:`voxanim.sequence.TrainingRun` whose ``shown`` holds at least the
    first four steps and the last."""

    PHASES = ("batch 1", "batch 2", "batch 3", "fast-forward", "last batch")

    def __init__(self, run):
        super().__init__("2-layer RNN: training in batches", "", 9.0)
        self.run = run
        recs = run.shown
        self.detail = [recs[0], recs[1], recs[2], recs[-1]]     # batches shown in full
        self.P_after = [recs[1].P, recs[2].P, recs[3].P, run.P_final]
        self.B, self.T = recs[0].x.shape
        self.n = len(recs[0].P["b1"])
        self.steps = len(run.losses)
        self.hold = 3.0

        # ---- the model, in the centre ------------------------------------------
        self.hx = self.sheet((1, 1), (0.335, 0.44), 2.6, "x")
        self.h1 = self.sheet((2, self.n // 2), (0.44, 0.44), 2.5, "h1")
        self.h2 = self.sheet((2, self.n // 2), (0.575, 0.44), 2.5, "h2")
        self.hy = self.sheet((1, 1), (0.675, 0.44), 2.6, "y")
        n = self.n
        self.W = {
            "Wx1": self.sheet((1, n), (0.385, 0.615), 1.15, "Wx1"),
            "Wh1": self.sheet((n, n), (0.44, 0.255), 1.15, "Wh1"),
            "W12": self.sheet((n, n), (0.51, 0.665), 1.15, "W12"),
            "Wh2": self.sheet((n, n), (0.575, 0.255), 1.15, "Wh2"),
            "Wy": self.sheet((1, n), (0.63, 0.615), 1.15, "Wy"),
        }
        self.wmax = {k: amax(self.detail[0].P[k]) for k in self.W}

        # ---- timeline ---------------------------------------------------------------
        ticks = []                  # (start, dur, batch, entry, step)
        t = 1.0
        self.batch_t = []           # per shown batch: (start, update_start, end)
        for bi in range(4):
            if bi == 3:
                self.t_ff = (t, t + 5.0)
                t += 5.4
            b0 = t
            for e in range(self.B):
                if bi == 0 and e == 0:
                    durs = Tempo(slow=1.2, fast=0.16, ramp_in=2.0, ramp_out=0.5).durations(self.T)
                elif bi == 0 or e == 0:
                    durs = np.full(self.T, 0.09)
                else:
                    durs = np.full(self.T, 0.05)
                for s in range(self.T):
                    ticks.append((t, float(durs[s]), bi, e, s))
                    t += float(durs[s])
                t += 0.15
            u0 = t
            t += 2.2 if bi == 0 else 1.2
            self.batch_t.append((b0, u0, t))
            t += 0.3
        self.ticks = ticks
        self.tick_starts = [k[0] for k in ticks]
        self.finish(t)

    # ------------------------------------------------------------------ helpers
    def batch_color(self, bi):
        """Colour of shown batch ``bi`` (0..3)."""
        return BATCH_COLORS[bi % len(BATCH_COLORS)]

    def entry_color(self, bi, e):
        """Colour of sequence ``e`` of shown batch ``bi``: a shade of the batch's."""
        return C.shades(self.batch_color(bi), self.B, -0.14, 0.12)[e]

    def fr(self, fx, fy):
        """Layout fraction -> frame fraction (the camera's safe-area zoom), so
        text lines up with geometry placed at the same layout fraction."""
        k = self.cam.zoom
        return 0.5 + (fx - 0.5) * k, 0.5 + (fy - 0.5) * k

    def _pt(self, fx, fy, dz=0.0):
        """World point at layout fraction ``(fx, fy)``, raised by ``dz``."""
        p = self.cam.frac(fx, fy)
        return p + np.array([0.0, 0.0, dz])

    def strip(self, side, e, s, v):
        """World point of time step ``s`` of entry ``e``'s wave (``v`` its value)."""
        x0 = 0.05 if side == "in" else 0.745
        fx = x0 + 0.205 * s / (self.T - 1)
        fy = 0.31 + 0.095 * e
        return self._pt(fx, fy, 0.9 * v)

    def loss_xy(self, step, loss):
        """Loss-curve point (log-log) as a frame fraction."""
        fx = 0.12 + 0.76 * np.log10(step) / np.log10(self.steps)
        lo, hi = LOSS_RANGE
        fy = 0.885 - 0.15 * (np.clip(np.log10(loss), lo, hi) - lo) / (hi - lo)
        return fx, fy

    def which(self, t):
        """The tick at time ``t``, ``((start, dur, batch, entry, step), index)``;
        index -1 before the first tick (then the first tick is returned)."""
        n = bisect.bisect_right(self.tick_starts, t) - 1
        return self.ticks[max(n, 0)], n

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase at time ``t``."""
        if t >= self.batch_t[3][0]:
            return 4                                    # the last batch
        if t >= self.t_ff[0]:
            return 3                                    # fast-forward
        for bi in (2, 1, 0):
            if t >= self.batch_t[bi][0]:
                return bi
        return 0

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        (t0, dur, bi, e, s), n = self.which(t)
        in_ff = self.t_ff[0] <= t < self.t_ff[1]
        if in_ff:
            bi = 3
        b0, u0, b1 = self.batch_t[bi]
        u_tick = (t - t0) / dur if n >= 0 else 0.0
        rec = self.detail[bi]
        updating = u0 <= t < b1 and not in_ff
        ff = remap(t, *self.t_ff)

        ph = 0 if reset > 0 else self.step_of(t)
        subtitle = [
            "each batch: %d sequences, one time step at a time, then one weight update"
            % self.B,
            "batch 2: new sequences, slightly better weights",
            "batch 3: every batch nudges the weights a little",
            "fast-forward: %d more batches, each its own colour" % (self.steps - 4),
            "after %d batches the predictions follow the waves" % self.steps,
        ][ph]
        if t < b0 and not in_ff and bi == 0:
            s, e, u_tick = -1, 0, 0.0          # before the first value arrives

        self.subtitle = self._weights(sc, t, bi, updating, u0, b1, ff, in_ff, reset) or subtitle
        if not in_ff:
            self._streams(sc, t, bi, e, s, u_tick, rec, updating, back)
        self._model(sc, bi, e, s, u_tick, rec, in_ff, back)
        self._loss_curve(sc, t, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- input queue and predictions ----------------------------------------------------
    def _streams(self, sc, t, bi, e_now, s_now, u, rec, updating, back):
        """Batch ``bi``'s input queue, targets and predictions, with labels.

        ``e_now``, ``s_now``: the sequence and time step being read (``-1``:
        none yet); ``u``: progress through that tick.
        """
        B, T = self.B, self.T
        appear = smoothstep(self.batch_t[bi][0] - 0.8, self.batch_t[bi][0] - 0.1, t) \
            if bi else smoothstep(0.2, 0.9, t)
        leave = smoothstep(self.batch_t[bi][2] - 0.3, self.batch_t[bi][2] + 0.2, t)
        op = appear * (1 - leave) * back
        if op <= 0.01:
            return
        done_all = updating or t >= self.batch_t[bi][1]
        for e in range(B):
            col = self.entry_color(bi, e)
            for s in range(T):
                consumed = done_all or e < e_now or (e == e_now and s < s_now) or \
                    (e == e_now and s == s_now and u > 0.25)
                # inputs: waiting values are solid, consumed ones fade
                sc.voxel(self.strip("in", e, s, rec.x[e, s]), col,
                         op * (0.28 if consumed else 1.0), shadow=False, size=0.62)
                # targets (ghosts) and predictions
                sc.voxel(self.strip("out", e, s, rec.target[e, s]), GHOST, 0.45 * op,
                         shadow=False, size=0.55)
                made = done_all or e < e_now or (e == e_now and s < s_now) or \
                    (e == e_now and s == s_now and u > 0.85)
                if made:
                    sc.voxel(self.strip("out", e, s, rec.f["y"][e, s]), col, op,
                             shadow=False, size=0.62)
            if e == e_now and not done_all:
                # the sequence being read: a soft glow on its values, no frame
                for s in range(T):
                    if not ((s < s_now) or (s == s_now and u > 0.25)):
                        sc.voxel(self.strip("in", e, s, rec.x[e, s]), col, op, glow=0.25,
                                 shadow=False, size=0.62)
        # labels, in layout fractions like the strips
        txt = self.fr
        sc.text(*txt(0.152, 0.245), "input: batch %d, %d sequences"
                % (self.detail[bi].step, B), 0.029, "regular", layout.CAPTION_INK,
                op * self._cap, "mb")
        sc.text(*txt(0.847, 0.245), "prediction vs target (white)", 0.029, "regular",
                layout.CAPTION_INK, op * self._cap, "mb")
        if not done_all and s_now >= 0:
            sc.text(*txt(0.152, 0.665), "sequence %d of %d,  t = %d / %d"
                    % (e_now + 1, B, s_now + 1, T), 0.029, "regular", layout.CAPTION_INK,
                    op * self._cap, "mt")
        if done_all:
            sc.text(*txt(0.847, 0.665), "batch loss = %.3f" % rec.loss, 0.031, "regular",
                    layout.STRONG_INK, op * self._cap, "mt")

    # -- the model -------------------------------------------------------------------------
    def _model(self, sc, bi, e, s, u, rec, in_ff, back):
        """The model's state for sequence ``e``, step ``s``, and the signal through it."""
        f = rec.f
        col = self.entry_color(bi, max(e, 0))
        valid = s >= 0 and not in_ff
        h1 = f["h1"][e, s + 1] if valid else np.zeros(self.n)
        h2 = f["h2"][e, s + 1] if valid else np.zeros(self.n)
        h1p = f["h1"][e, s] if valid else np.zeros(self.n)
        h2p = f["h2"][e, s] if valid else np.zeros(self.n)
        # cells switch to the new state as the signal reaches them
        a1 = smoothstep(0.30, 0.45, u) if valid else 0.0
        a2 = smoothstep(0.55, 0.70, u) if valid else 0.0
        s1 = div_colors((h1p + (h1 - h1p) * a1) * back, 1.0)   # the loop rewinds to 0
        s2 = div_colors((h2p + (h2 - h2p) * a2) * back, 1.0)
        if in_ff:
            s1 = s2 = np.tile(C.NEUTRAL, (self.n, 1))
        self.h1.fill(s1)
        self.h2.fill(s2)
        xv = f["x"][e, s] if valid else 0.0
        yv = f["y"][e, s] if valid else 0.0
        self.hx.fill(div_colors(np.array([xv * back]), 1.0))
        self.hy.fill(div_colors(np.array([yv * back]), 1.0))
        for sh in (self.hx, self.h1, self.h2, self.hy):
            sh.cage = 1.0                    # the model never leaves the frame
            sh.block.white[...] = 0.0
        # the signal travelling through
        if valid:
            particle(sc, self.strip("in", e, s, f["x"][e, s]), self.hx.center,
                     remap(u, 0.0, 0.25), col, 0.8, lift=1.5)
            particle(sc, self.hx.center, self.h1.center, remap(u, 0.22, 0.38), col, 0.8,
                     lift=0.5)
            particle(sc, self.h1.center, self.h2.center, remap(u, 0.45, 0.6), col, 0.8,
                     lift=0.5)
            particle(sc, self.h2.center, self.hy.center, remap(u, 0.68, 0.8), col, 0.8,
                     lift=0.5)
            particle(sc, self.hy.center, self.strip("out", e, s, f["y"][e, s]),
                     remap(u, 0.8, 0.95), col, 0.8, lift=1.5)
            # the recurrent loops carry the previous state back in
            for sh, w in ((self.h1, "Wh1"), (self.h2, "Wh2")):
                particle(sc, sh.center, self.W[w].center, remap(u, 0.05, 0.18), col, 0.55,
                         lift=1.2)
                particle(sc, self.W[w].center, sh.center, remap(u, 0.18, 0.30), col, 0.55,
                         lift=-1.2)
            sc.probe(("tick",), (u * 30, 0, 0), tol=8.0)
        # captions
        self.caption(sc, self.hx, "x_t")
        self.caption(sc, self.h1, "layer 1  h1")
        self.caption(sc, self.h2, "layer 2  h2")
        self.caption(sc, self.hy, "y_t")

    # -- weights ------------------------------------------------------------------------------
    def _weights(self, sc, t, bi, updating, u0, b1, ff, in_ff, reset):
        """The five weight matrices, blending to each batch's update.

        Returns the narration line during an update, else ``None``.
        """
        if in_ff:                                                # step 4 .. the last step
            Pa, Pb, m = self.P_after[2], self.detail[3].P, ff
        else:
            Pa, Pb = self.detail[bi].P, self.P_after[bi]
            m = smoothstep(0.45, 0.85, remap(t, u0, b1)) if updating else 0.0
            if t >= b1:
                m = 1.0
        g = self.detail[bi].g
        for k, sh in self.W.items():
            W = Pa[k] + (Pb[k] - Pa[k]) * m
            if reset > 0:                    # back to the untrained weights for the loop
                W = W + (self.detail[0].P[k] - W) * ease_in_out(reset)
            col = div_colors(np.ravel(W), self.wmax[k])
            if updating:
                gc = div_colors(np.ravel(g[k]), amax(g[k]))
                grad = soft_band(remap(t, u0, b1), 0.05, 0.45, 0.08)
                col = col + (gc - col) * grad
            sh.fill(col)
            sh.cage = 1.0
            glint = soft_band(remap(t, u0, b1), 0.5, 0.75, 0.06) if updating else 0.0
            if in_ff:
                glint = 0.35 * (0.5 + 0.5 * np.sin(ff * 60))
            sh.block.white[...] = 0.6 * glint
            self.caption(sc, sh, k, size=0.029)
        if not updating:
            return None
        u = remap(t, u0, b1)
        sc.probe(("upd",), (u * 30, 0, 0), tol=8.0)
        # the narration line says what is happening to the weights
        if u < 0.45:
            return ("backward through time: gradients from all %d sequences are averaged"
                    % self.B)
        return "one update for the whole batch:  W = W - lr %s dW" % TIMES

    # -- loss curve -------------------------------------------------------------------------------
    def _loss_curve(self, sc, t, back):
        """The loss of every step reported so far, on log-log axes."""
        losses = self.run.losses
        # which steps have reported their loss so far
        last = 0
        for bi in range(3):
            if t >= self.batch_t[bi][2] - 0.2:
                last = bi + 1
        if t >= self.t_ff[0]:
            last = 3 + int(remap(t, *self.t_ff) * (self.steps - 4))
        if t >= self.batch_t[3][2] - 0.2:
            last = self.steps
        if back < 1:
            last = int(last * back)                      # the loop rewinds the curve
        # axes
        x0, y0 = 0.12, 0.885
        ax = [self._pt(x0, y0), self._pt(0.88, y0), self._pt(x0, 0.73)]
        for end in (ax[1], ax[2]):           # the axes stay put through the loop
            sc.line(ax[0], end, width=1.0, color=(0.5, 0.52, 0.54), opacity=0.7, bias=0.0)
        txt = self.fr
        sc.text(*txt(x0 - 0.012, 0.80), "loss", 0.029, "regular", layout.CAPTION_INK,
                back * self._cap, "rm")
        for step in (1, 10, 100, self.steps):
            fx, _ = self.loss_xy(step, 1.0)
            sc.text(*txt(fx, y0 + 0.012), str(step), 0.029, "regular", layout.NOTE_INK,
                    back * self._cap, "mt")
        sc.text(*txt(0.95, y0), "batch", 0.029, "regular", layout.CAPTION_INK,
                back * self._cap, "lm")
        prev = None
        for k in range(last):
            step = k + 1
            p = self._pt(*self.loss_xy(step, losses[k]))
            shown = k < 3 or step == self.steps          # batches animated in full
            if shown:
                col = self.batch_color(k if k < 3 else 3)
            else:
                col = BATCH_COLORS[4 + k % 4]            # every batch its own colour
            sc.voxel(p, col, back, shadow=False, reflect=False, size=0.75 if shown else 0.35,
                     glow=0.2 if shown else 0.0)
            if k % 3 == 0 or shown:
                if prev is not None:
                    sc.line(prev, p, width=0.8, color=(0.55, 0.57, 0.59), opacity=0.5 * back,
                            bias=0.0)
                prev = p
