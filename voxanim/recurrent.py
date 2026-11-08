"""One training step of a recurrent network (RNN or LSTM) on a sine wave.

The network is drawn *unrolled through time*, one column per step, left to
right:

    prediction plot   y_t (red) vs the target x_{t+1} (white ghost)
    hidden bands      RNN: h            LSTM: h, c, o, f, i, g
    input wave        x_t (teal)

Each band is ``units x 16``; signed values use the diverging scale, gates
(sigmoids in 0..1) use the lit scale: an open gate is lit.  The weights
(shared by every step) sit on the right.

Five steps, like the other training-step animations:

1. forward -- the same cell runs at every step; the hidden state carries
   information to the right,
2. loss -- the errors between predictions and targets,
3. backward (BPTT) -- gradients flow down from every prediction and *left*
   through time; the weight gradients accumulate one step at a time,
4. update -- the shared weights step against the summed gradient,
5. forward again -- the predictions move towards the targets.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep, soft_band
from .mlp import X, NetAnimation, amax, div_colors, particle

PRED = C.FILTERS[0]
GHOST = np.array([0.97, 0.97, 0.96], np.float32)
GATE_LO = C.rgb("#3a3f44")
GATE_HI = C.rgb("#c9fff6")

KIND = {
    "rnn": dict(ppu=15.5, scale=1.0, amp=2.4, gap=3.6, title="RNN: one training step",
                bands=[("h", "hidden h", "signed")]),
    "lstm": dict(ppu=8.4, scale=0.78, amp=2.6, gap=2.4, title="LSTM: one training step",
                 bands=[("h", "h = o tanh(c)", "signed"), ("c", "cell state c", "signed"),
                        ("o", "output gate o", "gate"), ("f", "forget gate f", "gate"),
                        ("i", "input gate i", "gate"), ("g", "candidate g", "signed")]),
}
GATE_IDX = dict(i=0, f=1, g=2, o=3)


class RecurrentTrainingStep(NetAnimation):
    PHASES = ("forward", "loss", "backward (BPTT)", "update", "forward again")

    def __init__(self, tr, tr_after, lr):
        k = KIND[tr.kind]
        super().__init__(k["title"], "", k["ppu"])
        self.tr, self.tr2, self.lr, self.k = tr, tr_after, lr, k
        self.hold = 2.5
        self.T = len(tr.x)
        n = len(tr.P["Wy"])
        self.n = n
        pitch = (1.0, 1.5)
        # ---- bands, stacked top to bottom around the centre ----------------
        centre = self.cam.frac(0.42, 0.55)
        names = k["bands"]
        h_each = n * k["scale"] + 0.7
        top = centre[2] + h_each * len(names) / 2
        self.bands = {}
        for b, (key, label, style) in enumerate(names):
            s = self.sheet((n, self.T), (0.42, 0.55), k["scale"], key, pitch=pitch)
            c = centre.copy()
            c[2] = top - h_each * (b + 0.5)
            s.block.place(c)
            self.bands[key] = (s, label, style)
        first = self.bands[names[0][0]][0]
        last = self.bands[names[-1][0]][0]
        self.z_pred = first.block.origin[2] + first.block.extent[2] + k["gap"] + k["amp"]
        self.z_in = last.block.origin[2] - k["gap"] - k["amp"]
        self.y_plot = first.block.origin[1] + 0.5
        self.xs = [first.block.cell_center(0, t, 0)[0] for t in range(self.T)]

        # ---- weights panel ----------------------------------------------------------
        P = tr.P
        if tr.kind == "rnn":
            mats = [("Wh", "Wh", P["Wh"]), ("Wx", "Wx", P["Wx"][:, None]),
                    ("Wy", "Wy", P["Wy"][None, :])]
            where = [(0.80, 0.42), (0.91, 0.42), (0.84, 0.66)]
            scales = [1.0, 1.0, 1.0]
        else:
            mats = [("W", "W: [x, h] into 4 gates", P["W"]),
                    ("Wy", "Wy", P["Wy"][None, :])]
            where = [(0.85, 0.44), (0.85, 0.80)]
            scales = [1.0, 1.0]
        self.wpanels = []
        for (key, label, W), at, sc in zip(mats, where, scales):
            s = self.sheet(W.shape, at, sc, "W" + key)
            self.wpanels.append((key, label, s))

        # ---- timeline ---------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.25):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("intro", 1.0, 0.2)
        fw = Tempo(slow=2.2 if tr.kind == "rnn" else 3.0, fast=0.32, ramp_in=2.0,
                   ramp_out=1.0).durations(self.T)
        self.f_starts = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]])
        self.f_durs = fw
        add("fwd", float(fw.sum()), 0.3)
        add("loss", 1.8, 0.4)
        bw = Tempo(slow=1.8, fast=0.3, ramp_in=2.0, ramp_out=1.0).durations(self.T)
        self.b_starts = t + np.concatenate([[0.0], np.cumsum(bw)[:-1]])
        self.b_durs = bw
        add("bwd", float(bw.sum()), 0.4)
        add("update", 3.0, 0.4)
        add("again", 3.0, 0.4)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["loss"][0], Tm["bwd"][0], Tm["update"][0], Tm["again"][0]]

        # ---- values per band ----------------------------------------------------------
        self.vals = self._band_values(tr)
        self.vals2 = self._band_values(tr_after)
        self.grads = self._band_grads(tr)
        self.contrib = self._contributions(tr)           # per step, per weight panel
        self.total = [c.sum(0) for c in self.contrib]

    # -- data ------------------------------------------------------------------------
    def _band_values(self, tr):
        f = tr.f
        out = {"h": f["h"][1:]}
        if tr.kind == "lstm":
            out["c"] = f["c"][1:]
            for g, j in GATE_IDX.items():
                out[g] = f["gates"][:, j]
        return out

    def _band_grads(self, tr):
        d, n = tr.d, self.n
        out = {"h": d["dh"]}
        if tr.kind == "lstm":
            out["c"] = d["dc"]
            for g, j in GATE_IDX.items():
                out[g] = d["dz"][:, j * n:(j + 1) * n]
        return out

    def _contributions(self, tr):
        f, d, x = tr.f, tr.d, tr.x
        hp, hn = f["h"][:-1], f["h"][1:]                  # h_{t-1}, h_t
        wy = d["dy"][:, None] * hn                         # (T, n)
        if tr.kind == "rnn":
            return [np.einsum("ti,tj->tij", d["dz"], hp), (d["dz"] * x[:, None])[..., None],
                    wy[:, None, :]]
        inp = np.concatenate([x[:, None], hp], 1)
        return [np.einsum("ti,tj->tij", d["dz"], inp), wy[:, None, :]]

    def step_of(self, t):
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def p(self, name, t):
        return remap(t, *self.Tm[name])

    def _colors(self, key, vals, style, vmax=None):
        v = vals.T.ravel()                                 # (n, T) row-major
        if style == "gate":
            return (GATE_LO[None] + (GATE_HI - GATE_LO)[None] *
                    np.clip(v, 0, 1)[:, None] ** 1.3).astype(np.float32)
        return div_colors(v, vmax or amax(vals))

    def _z(self, base, v):
        return base + self.k["amp"] * float(v)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        tr, tr2, T = self.tr, self.tr2, self.T
        ph = self.step_of(t)
        label_ph = 0 if reset > 0 else ph
        rec = "h carries" if tr.kind == "rnn" else "c and h carry"
        self.subtitle = [
            "forward: the same cell runs at every step; %s what it has seen to the "
            "right" % rec,
            "loss: mean squared error between each prediction and the next value",
            "backward through time: gradients flow down from every prediction and left "
            "through the steps",
            "update: the shared weights step against the gradient summed over all steps  "
            "(lr = %g)" % self.lr,
            "forward again: the predictions move towards the targets: loss %.3f to %.3f"
            % (tr.loss, tr2.loss),
        ][label_ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        appear = smoothstep(0.0, 0.8, t) * back
        again = self.p("again", t)

        # where the forward sweep is: steps fully done, current step progress
        nf = int(np.searchsorted(self.f_starts, t, side="right") - 1)
        uf = (t - self.f_starts[max(nf, 0)]) / self.f_durs[max(nf, 0)] if nf >= 0 else 0.0
        done_f = nf if nf >= 0 and uf < 1 else nf + 1
        if t >= self.Tm["fwd"][1]:
            done_f = T
        nb = int(np.searchsorted(self.b_starts, t, side="right") - 1)
        ub = (t - self.b_starts[max(nb, 0)]) / self.b_durs[max(nb, 0)] if nb >= 0 else 0.0
        if t >= self.Tm["bwd"][1]:
            nb, ub = T, 0.0
        n_again = again * (T + 2)                          # sweep position, in steps

        self._waves(sc, t, done_f, nf, uf, nb, ub, n_again, appear, back)
        self._bands(sc, t, done_f, nf, uf, nb, ub, n_again, back)
        self._weights(sc, t, nf, uf, nb, ub, reset)
        self._loss_text(sc, t, again, back)
        self.step_indicator(sc, label_ph, self.PHASES)

    # -- waves: input below, predictions vs targets above ------------------------------
    def _waves(self, sc, t, done_f, nf, uf, nb, ub, n_again, appear, back):
        tr, tr2, T = self.tr, self.tr2, self.T
        y_ = self.y_plot
        if appear <= 0.01:
            return
        for base in (self.z_in, self.z_pred):
            sc.line((self.xs[0] - 0.8, y_, base), (self.xs[-1] + 0.8, y_, base), width=1.0,
                    color=(0.55, 0.57, 0.59), opacity=0.5 * appear, bias=0.0)
        loss_on = self.p("loss", t)
        for s in range(T):
            xin = np.array([self.xs[s], y_, self._z(self.z_in, tr.x[s])])
            reading = 1.0 if (s == nf and 0 <= uf < 0.4) else 0.0
            sc.voxel(xin, C.INPUT, appear, white=0.5 * reading, glow=0.15 * reading,
                     shadow=False, size=0.75)
            if s:
                prev = np.array([self.xs[s - 1], y_, self._z(self.z_in, tr.x[s - 1])])
                sc.line(prev, xin, width=1.1, color=(0.45, 0.62, 0.62), opacity=0.6 * appear)
            ztgt = self._z(self.z_pred, tr.target[s])
            tgt = np.array([self.xs[s], y_, ztgt])
            sc.voxel(tgt, GHOST, 0.45 * appear, shadow=False, size=0.75)
            # the prediction: appears when its step is done; moves in "again"
            if s < done_f:
                m = smoothstep(0.0, 1.0, n_again - s - 0.6)
                yv = tr.f["y"][s] + (tr2.f["y"][s] - tr.f["y"][s]) * m
                pp = np.array([self.xs[s], y_ - 0.05, self._z(self.z_pred, yv)])
                col = PRED
                white = 0.0
                if self.Tm["bwd"][0] <= t < self.Tm["update"][0] and s >= T - 1 - min(nb, T):
                    # its gradient: orange = prediction too high, blue = too low
                    dy = tr.d["dy"]
                    col = div_colors(dy[s:s + 1], amax(dy))[0]
                fresh = s == nf and uf < 1
                if fresh:
                    white = 1.0 - smoothstep(0.85, 1.0, uf)
                sc.voxel(pp, col, back, white=white, shadow=False, size=0.8)
                err = (loss_on > 0 and t < self.Tm["again"][0]) or n_again > s + 1.5
                if err:
                    op = smoothstep(0.0, 0.3, loss_on - s / T * 0.6) if t < self.Tm["again"][0] \
                        else 1.0
                    sc.line(pp, tgt, width=1.6, color=(0.75, 0.22, 0.22), opacity=0.8 * op * back,
                            bias=0.05)
                if s:
                    prev_y = tr.f["y"][s - 1] + (tr2.f["y"][s - 1] - tr.f["y"][s - 1]) * \
                        smoothstep(0.0, 1.0, n_again - s + 1 - 0.6)
                    q = np.array([self.xs[s - 1], y_ - 0.05, self._z(self.z_pred, prev_y)])
                    sc.line(q, pp, width=1.2, color=(0.80, 0.40, 0.40), opacity=0.6 * back)
            if 0 < n_again - s - 0.6 < 1:
                sc.probe(("again", s), (n_again * 20, 0, 0), tol=8.0)

    # -- the bands of hidden state ----------------------------------------------------
    def _bands(self, sc, t, done_f, nf, uf, nb, ub, n_again, back):
        tr, T, n = self.tr, self.T, self.n
        lstm = tr.kind == "lstm"
        # sub-timing inside a forward step: gates, then c, then h
        order = {"h": 0.62} if not lstm else dict(f=0.25, i=0.25, g=0.25, o=0.25,
                                                  c=0.45, h=0.62)
        bwd_on = t >= self.Tm["bwd"][0]
        for key, (s, label, style) in self.bands.items():
            vals, vals2 = self.vals[key], self.vals2[key]
            cols = self._colors(key, vals, style)
            cols2 = self._colors(key, vals2, style, amax(vals) if style == "signed" else None)
            g = self.grads[key]
            gcols = div_colors(g.T.ravel(), amax(g))
            col_t = np.tile(np.arange(T), n)               # column index of each cell
            # computed so far (forward)
            shown = col_t < done_f
            if 0 <= nf < T and uf >= order[key]:
                shown |= col_t == nf
            if t >= self.Tm["fwd"][1]:
                shown[:] = True
            c = cols.copy()
            if bwd_on:
                # gradient arrives column by column, from the right
                got = col_t >= T - nb if nb < T else np.ones_like(col_t, bool)
                cur = (col_t == T - 1 - nb) & (ub > 0.35) if nb < T else np.zeros_like(got)
                m = got | cur
                c[m] = gcols[m]
            if n_again > 0:                  # the new forward pass sweeps over the gradients
                m = np.clip(n_again - col_t - 0.3, 0, 1)[:, None]
                c = c + (cols2 - c) * m
            s.fill(c)
            s.show(shown)
            s.cage = 1.0
            s.fade(back)
            fresh = (col_t == nf) & (uf < 1)
            s.block.white[:, :, 0] = (fresh * (1 - smoothstep(0.7, 1.0, uf)) * 0.6).reshape(n, T)
            grad_phase = bwd_on and t < self.Tm["again"][0]
            self._band_label(sc, s, ("gradient at " + key) if grad_phase else label)

        # the cell at the current step, and what flows
        anchor = self.bands["h"][0]
        bottom = self.bands[list(self.bands)[-1]][0]
        y_ = self.y_plot
        if 0 <= nf < T and t < self.Tm["fwd"][1]:
            self._cell_frame(sc, nf)
            xin = np.array([self.xs[nf], y_, self._z(self.z_in, tr.x[nf])])
            particle(sc, xin, bottom.cell(n // 2 * T + nf), remap(uf, 0.0, 0.3), C.INPUT, 0.6,
                     lift=0.0)
            if nf > 0:
                src = self.bands["c" if lstm else "h"][0]
                for r in range(n):
                    a = src.cell(r * T + nf - 1)
                    b = src.cell(r * T + nf)
                    op = 1.0 if not lstm else float(self.vals["f"][nf, r])
                    col = div_colors(self.vals["c" if lstm else "h"][nf - 1, r:r + 1],
                                     amax(self.vals["c" if lstm else "h"]))[0]
                    particle(sc, a, b, remap(uf, 0.05, 0.45), col, 0.55, opacity=op, lift=0.4)
            top = anchor.cell(nf)
            pred = np.array([self.xs[nf], y_, self._z(self.z_pred, tr.f["y"][nf])])
            particle(sc, top, pred, remap(uf, 0.65, 0.88), PRED, 0.6, lift=0.0)
            sc.probe(("fwd",), (nf * 20 + uf * 20, 0, 0), tol=10.0)
        if 0 <= nb < T and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
            s_ = T - 1 - nb
            self._cell_frame(sc, s_)
            pred = np.array([self.xs[s_], y_, self._z(self.z_pred, tr.f["y"][s_])])
            dy = tr.d["dy"]
            col = div_colors(dy[s_:s_ + 1], amax(dy))[0]
            particle(sc, pred, anchor.cell(s_), remap(ub, 0.0, 0.35), col, 0.6, lift=0.0)
            if s_ < T - 1:
                src = self.bands["c" if lstm else "h"][0]
                gk = self.grads["c" if lstm else "h"]
                for r in range(n):
                    gc = div_colors(gk[s_ + 1, r:r + 1], amax(gk))[0]
                    particle(sc, src.cell(r * T + s_ + 1), src.cell(r * T + s_),
                             remap(ub, 0.05, 0.45), gc, 0.55, lift=-0.4)
            sc.probe(("bwd",), (nb * 20 + ub * 20, 0, 0), tol=10.0)

    def _cell_frame(self, sc, col):
        ss = [s for s, _l, _st in self.bands.values()]
        lo = np.min([s.block.box((0, s.h), (col, col + 1), (0, 1))[0] for s in ss], 0)
        hi = np.max([s.block.box((0, s.h), (col, col + 1), (0, 1))[1] for s in ss], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=(0.96, 0.96, 0.96), width=1.6, opacity=0.95,
                    bias=0.1)

    def _band_label(self, sc, s, text):
        lo, hi = s.bbox()
        p = self.cam.project(np.array([lo[0], (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]))
        sc.text(p[0] / self.cam.width - 0.012, p[1] / self.cam.height, text, 0.026,
                "regular", layout.CAPTION_INK, self._cap, "rm")

    # -- shared weights ----------------------------------------------------------------
    def _weights(self, sc, t, nf, uf, nb, ub, reset):
        T = self.T
        bwd = self.Tm["bwd"][0] <= t < self.Tm["update"][0]
        up = self.p("update", t)
        for (key, label, s), contrib, total in zip(self.wpanels, self.contrib, self.total):
            W = self._weight(key, self.tr)
            W2 = self._weight(key, self.tr2)
            wmax = amax(W)
            w = div_colors(W.ravel(), wmax)
            col = w
            if bwd or up > 0:
                if bwd:
                    k = min(nb, T)
                    part = contrib[T - k:].sum(0) if k > 0 else np.zeros_like(total)
                    if k < T and ub > 0.4:
                        part = part + contrib[T - 1 - k] * smoothstep(0.4, 0.8, ub)
                else:
                    part = total
                col = div_colors(part.ravel(), amax(total))
            if up > 0:
                ch = -self.lr * total
                ccol = div_colors(ch.ravel(), amax(ch))
                col = col + (ccol - col) * smoothstep(0.0, 0.2, up)
                col = col + (div_colors(W2.ravel(), wmax) - col) * smoothstep(0.55, 0.8, up)
            if t >= self.Tm["update"][1]:
                col = div_colors(W2.ravel(), wmax)
            if reset > 0:
                col = col + (w - col) * ease_in_out(remap(reset, 0.2, 0.9))
            s.fill(col)
            s.cage = 1.0
            glint = 0.0
            if 0 <= nf < T and t < self.Tm["fwd"][1]:
                glint = 0.35 * soft_band(uf, 0.15, 0.55, 0.1)    # the same weights, every step
            glint = max(glint, 0.7 * soft_band(up, 0.58, 0.8, 0.05))
            s.block.white[...] = glint
            if up <= 0 or up >= 1:
                text = label if not bwd else "d" + key
            elif up < 0.55:
                text = "change"
            else:
                text = "updated"
            self.caption(sc, s, text, size=0.026)
        if up >= 0.55 and up < 1:
            rel = max(np.abs(self.lr * tot).max() / amax(self._weight(k, self.tr))
                      for (k, _l, _s), tot in zip(self.wpanels, self.total))
            self.caption(sc, self.wpanels[-1][2], "largest change: %.2g%% of a weight"
                         % (100 * rel), size=0.024, offset=0.035)
        if bwd:
            self.caption(sc, self.wpanels[-1][2], "gradients summed over the steps",
                         size=0.024, offset=0.035)
        elif 0 < up < 0.55:
            self.caption(sc, self.wpanels[-1][2], "change = -lr %s gradient" % X,
                         size=0.024, offset=0.035)

    def _weight(self, key, tr):
        P = tr.P
        if key == "Wx":
            return P["Wx"][:, None]
        if key == "Wy":
            return P["Wy"][None, :]
        return P[key]

    # -- loss text -------------------------------------------------------------------------
    def _loss_text(self, sc, t, again, back):
        tr, tr2 = self.tr, self.tr2
        op = self.p("loss", t) * back * self._cap
        if op <= 0.01:
            return
        w = smoothstep(0.75, 1.0, again)
        loss = tr.loss + (tr2.loss - tr.loss) * w
        p = self.cam.project(np.array([self.xs[-1] + 1.2, self.y_plot, self.z_pred]))
        x0, y0 = p[0] / self.cam.width, p[1] / self.cam.height
        sc.text(x0, y0 - 0.03, "prediction (red) vs next value (white)", 0.026, "regular",
                layout.CAPTION_INK, op, "lm")
        sc.text(x0, y0 + 0.012, "loss = %.3f" % loss, 0.03, "regular", layout.STRONG_INK, op,
                "lm")
        if w > 0:
            sc.text(x0, y0 + 0.05, "before the step: %.3f" % tr.loss, 0.024, "regular",
                    layout.NOTE_INK, w * op, "lm")
        p2 = self.cam.project(np.array([self.xs[0] - 1.0, self.y_plot,
                                        self.z_in - self.k["amp"] - 0.6]))
        sc.text(p2[0] / self.cam.width, p2[1] / self.cam.height + 0.01,
                "input: sin wave x_t", 0.026, "regular", layout.CAPTION_INK, self._cap * back,
                "lt")
