"""Curiosities: animations 97-100 of the "next 100" (see :mod:`voxanim.mech`).

A Hopfield memory, a neural cellular automaton, a neural ODE and evolution
strategies. The first three draw the redesigned experiments of
:mod:`voxanim.study_curios`, which record every state the drawings show (each
single-pixel update of a recall, every cell of the automaton step by step, each
point's path along the flow): nothing is recomputed here but layout. Evolution
strategies replays its study's run with the study's seed and functions, and
the tests hold the replay to the study's numbers.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import layout
from . import next_systems as S
from . import study_curios as SC
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    GREEN,
    GRID,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    soft,
)


def _settings(*files):
    from . import realdata

    return dict(version=1, seed=17, source=realdata.source_hash(*files))


# ====================================================================== shared
def _rgb(c):
    return np.asarray(c, np.float32)


def small(v):
    """An error as text: three decimals, or five where three would read 0.000."""
    return "%.3f" % v if v >= 0.0005 else "%.5f" % v


def progress(anim, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, xlog=False, ylog=False,
             dot=True):
    """``ys`` against ``xs`` on ``chart``, drawn as far as fraction ``go`` of the x axis
    (of its logarithm when ``xlog``), through every measured point, ending in a lit dot."""
    if op <= 0.01 or go <= 0:
        return None
    tx = np.log(np.asarray(xs, float)) if xlog else np.asarray(xs, float)
    ty = np.log(np.asarray(ys, float)) if ylog else np.asarray(ys, float)
    upto = tx[0] + (tx[-1] - tx[0]) * min(float(go), 1.0)
    lx = np.append(tx[tx < upto], upto)
    ly = np.interp(lx, tx, ty)
    pts = [anim.P(chart.fx(float(np.exp(a) if xlog else a)),
                  chart.fy(float(np.exp(b) if ylog else b))) for a, b in zip(lx, ly)]
    if len(pts) > 1:
        anim.polyline(sc, pts, width, ink, op, bias=0.3)
    if dot:
        anim.disk(sc, pts[-1], 5.0, ink, op, glow=0.4, lift=1.4, outline=0.0)
    return pts[-1]


# ====================================================================== 97: Hopfield
HOP_INK = (RED, AMBER, GREEN, BLUE)        # Hebbian, centred, pseudo-inverse, modern
HOP_NAMES = ("Hebbian", "minus the mean", "pseudo-inverse", "modern (softmax)")


class HopfieldNetwork(Mechanism):
    """Three MNIST digits stored in one weight matrix; a corrupted one recalled
    pixel by pixel while the energy falls; the blend it settles in; the same
    recall with the digits' overlap removed; the same lookup done as attention;
    and how many digits each rule holds, counted."""

    TITLE = "A Hopfield memory"
    PHASES = (("store", 10.0), ("recall", 12.0), ("a blend", 7.0), ("decorrelate", 10.0),
              ("as attention", 9.0), ("capacity", 11.0))

    def setup(self):
        d = self.d = SC.hopfield_data()
        s = self.study
        self.energy = np.asarray(s.panels[1].data["ys"], float)[0]
        self.loads = [int(v) for v in s.panels[2].data["x"]]
        self.by_load = np.asarray(s.panels[2].data["ys"], float)          # (rule, load)
        self.flips = np.asarray(s.panels[3].data["x"], float)
        self.by_flip = np.asarray(s.panels[3].data["ys"], float)          # (rule, flip)
        self.prob = d["prob"]
        pat = d["patterns"]
        self.differ = [int((d["final"] != p).sum()) for p in pat]
        self.corrupt = int((d["start"] != pat[1]).sum())
        self.flipped = int((np.diff(np.vstack([d["start"], d["states"]]), axis=0) != 0).sum())
        self.off = [int((d[k] != pat[1]).sum())
                    for k in ("final", "centred_final", "pinv_final", "modern_final")]
        # the largest load each rule still recalls 9 times in 10
        self.holds = [max([k for k, r in zip(self.loads, row) if r >= 0.9], default=0)
                      for row in self.by_load]
        # the pixel whose weights are shown: ink in the one only, nearest the middle
        rows, cols = np.divmod(np.arange(784), 28)
        middle = np.hypot(rows - 13.5, cols - 13.5)
        only = (pat[1] > 0) & (pat[0] < 0) & (pat[2] < 0)
        self.j = int(np.argmin(np.where(only, middle, 1e9)))
        far = np.hypot(rows - rows[self.j], cols - cols[self.j]) >= 5
        agree = (pat * pat[:, self.j:self.j + 1]).sum(0)          # -3, -1, 1 or 3
        self.partners = [int(np.argmin(np.where(far & (agree == want) & inked, middle, 1e9)))
                         for want, inked in ((3, pat[1] > 0), (-3, pat[0] > 0), (-1, pat[2] > 0))]
        blend = self.prob @ pat
        self.blend = (blend + 1) / 2
        self.blend_differ = int((np.where(blend >= 0, 1.0, -1.0) != pat[1]).sum())

    def subtitle_for(self, k):
        d = self.d
        return [
            "three MNIST digits are stored in one table of weights: two pixels that agree in a "
            "digit are tied together, two that differ are pushed apart",
            "recall: %d of the 784 pixels of the 1 are flipped, then each pixel in turn takes "
            "the side its neighbours vote for; the energy falls from %.0f to %.0f" % (
                self.corrupt, self.energy[0], self.energy[-1]),
            "it settles after one pass, %d pixels away from the stored 1 and %d from the 2: "
            "a blend of the memories, lower in energy than any of them" % (
                self.differ[1], self.differ[2]),
            "the fault is in the digits, which share most of their pixels: store what differs "
            "from their mean image, or use the pseudo-inverse rule, and the same corrupted 1 "
            "comes back with %d and %d pixels wrong" % (self.off[1], self.off[2]),
            "the modern version scores the corrupted picture against each memory and takes a "
            "softmax, as attention does: weights %.2f, %.2f, %.2f, and the 1 comes back" % tuple(
                d["prob"]),
            "counted on %d random sets of digits per load, a fifth of the pixels flipped: 9 "
            "recalls in 10 up to %d digits (Hebbian), %d (minus the mean), %d (pseudo-inverse); "
            "the modern rule, %d in 100 at %d" % (
                SC.HOP_TRIALS, self.holds[0], self.holds[1], self.holds[2],
                round(100 * self.by_load[3, -1]), self.loads[-1]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def picture(self, sc, state, fx, fy, size, op, ref=None, ring=(), frame=AXIS):
        """A 28 x 28 pattern of -1/+1 as ink on paper; where it differs from
        ``ref`` the pixel is red. ``ring``: ``(pixel, ink)`` outlined."""
        v = np.asarray(state, float).reshape(28, 28)
        rgb = np.where(v[..., None] > 0, _rgb(DIGIT_INK), _rgb(PAPER))
        if ref is not None:
            wrong = v != np.asarray(ref, float).reshape(28, 28)
            rgb = np.where(wrong[..., None],
                           np.where(v[..., None] > 0, _rgb(RED), _rgb(soft(RED, 0.45))), rgb)
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, size + 8, size + 8, PAPER, op, border=frame, border_w=1.6, radius=5,
                  lift=0.1)
        draw_image(self, sc, c, rgb.astype(np.float32), size, size, op, grid=0)
        dw, dh = self.frac(size, size)
        for pixel, ink in ring:
            r, k = divmod(int(pixel), 28)
            at = self.P(fx - dw / 2 + (k + 0.5) * dw / 28, fy - dh / 2 + (r + 0.5) * dh / 28)
            side = max(size / 28 + 5, 11)
            self.rect(sc, at, side, side, ink, op, border=ink, border_w=2.4, radius=2, lift=0.8,
                      fill_op=0.0)

    def phase_1(self, sc, u, op):
        d = self.d
        pat = d["patterns"]
        turn = min(int(u * 3.2), 2)
        within = min(u * 3.2 - turn, 1.0)
        i, j = self.partners[turn], self.j
        xs = (0.115, 0.285, 0.455)
        self.text(sc, xs[1], 0.31, "three stored digits", op, strong=True, size=0.034)
        total = 0
        for n, x in enumerate(xs):
            self.picture(sc, pat[n], x, 0.50, 128, op, ring=[(j, AMBER), (i, VIOLET)])
            same = pat[n, i] == pat[n, j]
            total += 1 if same else -1
            show = op * smoothstep(0.10 + 0.15 * n, 0.25 + 0.15 * n, within)
            self.text(sc, x, 0.685, "same  +1" if same else "differ  −1", show,
                      ink=dark(RED if same else BLUE), size=0.034)
        self.text(sc, xs[1], 0.765, "this pair's weight: %+d / 784" % total,
                  op * smoothstep(0.55, 0.7, within), strong=True, size=0.036)
        self.text(sc, xs[1], 0.85, "the orange pixel and the violet pixel, digit by digit", op)
        # every weight of the orange pixel, as a picture
        wj = d["W"][j].reshape(28, 28)
        cell = self.grid(sc, wj, 0.80, 0.585, 236, 236, op, lines=0)
        self.text(sc, 0.80, 0.31, "the orange pixel's 783 weights", op, strong=True, size=0.034)
        for pixel, ink in ((j, AMBER), (i, VIOLET)):
            r, k = divmod(pixel, 28)
            self.rect(sc, self.P(*cell(r, k)), 14, 14, ink, op, border=ink, border_w=2.4,
                      radius=2, lift=0.8, fill_op=0.0)
        self.text(sc, 0.80, 0.875, "red: be like me    blue: be my opposite", op)

    def _recall_at(self, u):
        """How many single-pixel updates have run at phase fraction ``u``: the
        first few slowly, then the rest of the sweep."""
        return int(min(784.0 * (u / 0.82) ** 3, 784))

    def phase_2(self, sc, u, op):
        d = self.d
        n = self._recall_at(u)
        one = d["patterns"][1]
        state = d["start"] if n == 0 else d["states"][n - 1]
        k = min(n, 783)                                   # the pixel being asked now
        pixel, vote = int(d["order"][k]), float(d["votes"][k])
        self.picture(sc, state, 0.20, 0.60, 240, op, ref=one,
                     ring=[(pixel, AMBER)] if n < 784 else ())
        self.text(sc, 0.20, 0.30, "the corrupted 1", op, strong=True, size=0.034)
        self.text(sc, 0.20, 0.90, "red: %d pixels differ from the stored 1" % int(
            (state != one).sum()), op, ink=dark(RED))
        steps = np.arange(785)
        chart = self.chart(((0.50, 0.93), (0.36, 0.70)), steps, d["energy"])
        self.axes(sc, chart, op, "pixels asked", "energy",
                  xticks=[(v, "%d" % v) for v in (0, 200, 400, 600, 784)])
        chart.hline(sc, self.P, float(d["memory_energy"][1]), op)
        self.text(sc, 0.515, chart.fy(float(d["memory_energy"][1])) - 0.04,
                  "the stored 1: %.0f" % d["memory_energy"][1], op, anchor="lm")
        self.plot(sc, chart, steps, d["energy"], AMBER, op, n / 784.0, width=2.2)
        if n < 784:
            now = state[pixel] > 0
            want = now if vote == 0 else vote > 0
            verdict = ("stays " if want == now else "turns ") + ("ink" if want else "paper")
            say = "pixel %d of 784: its neighbours vote %+.2f, it %s" % (n + 1, vote, verdict)
        else:
            say = "all 784 asked: %d flipped; a second pass changes nothing" % self.flipped
        self.text(sc, 0.44, 0.84, say, op, anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.44, 0.895, "energy now %.0f" % d["energy"][n], op, anchor="lm",
                  ink=dark(AMBER, 0.3), size=0.032)

    def phase_3(self, sc, u, op):
        d = self.d
        pat = d["patterns"]
        self.picture(sc, d["final"], 0.17, 0.57, 200, op, ref=pat[1], frame=AMBER)
        self.text(sc, 0.17, 0.315, "where it settled", op, strong=True, size=0.034)
        self.text(sc, 0.17, 0.83, "energy %.0f" % d["sweeps"][-1], op, ink=dark(AMBER, 0.3),
                  size=0.034)
        self.text(sc, 0.17, 0.885, "red: differs from the stored 1", op, ink=dark(RED))
        full = self.span(0.20)[0]
        self.text(sc, 0.52, 0.315, "pixels that differ from each memory", op, anchor="lm",
                  strong=True, size=0.034)
        for n in range(3):
            y = 0.44 + 0.17 * n
            show = op * smoothstep(0.08 + 0.2 * n, 0.2 + 0.2 * n, u)
            self.picture(sc, pat[n], 0.42, y, 68, op)
            self.hbar(sc, 0.52, y - 0.025, full * self.differ[n] / 120.0, RED, show, "",
                      "%d of 784" % self.differ[n], (u - 0.08 - 0.2 * n) / 0.2, thick=22.0,
                      track=full)
            self.text(sc, 0.52, y + 0.04, "its own energy %.0f" % d["memory_energy"][n], show,
                      anchor="lm")
        self.text(sc, 0.52, 0.895, "no stored digit: a memory nobody put there", op * smoothstep(
            0.7, 0.85, u), anchor="lm", strong=True, size=0.032)

    def phase_4(self, sc, u, op):
        d = self.d
        one = d["patterns"][1]
        n = int(min(u / 0.72, 1.0) ** 2 * 784)
        self.picture(sc, d["start"], 0.105, 0.575, 116, op, ref=one)
        self.text(sc, 0.105, 0.40, "the corrupted 1", op, strong=True, size=0.034)
        self.text(sc, 0.105, 0.75, "%d pixels flipped" % self.corrupt, op, ink=dark(RED))
        columns = (
            ("Hebbian rule", "the raw digits", d["final"], RED),
            ("minus the mean image", "what the digits share, removed",
             d["start"] if n == 0 else d["centred_states"][n - 1], AMBER),
            ("pseudo-inverse rule", "their overlaps cancelled",
             d["start"] if n == 0 else d["pinv_states"][n - 1], GREEN),
        )
        for k, (name, how, state, ink) in enumerate(columns):
            x = 0.345 + 0.24 * k
            wrong = int((np.asarray(state, float) != one).sum())
            self.picture(sc, state, x, 0.575, 150, op, ref=one, frame=ink)
            self.text(sc, x, 0.335, name, op, ink=dark(ink, 0.3), size=0.034)
            self.text(sc, x, 0.385, how, op)
            self.text(sc, x, 0.79, "%d pixels wrong" % wrong, op, strong=True, size=0.036)
        say = ("pixel %d of 784 asked" % (n + 1)) if n < 784 else "all 784 pixels asked once"
        self.text(sc, 0.585, 0.865, say + ": same corrupted picture, same updates, "
                  "different weights", op)

    def phase_5(self, sc, u, op):
        d = self.d
        pat = d["patterns"]
        self.picture(sc, d["start"], 0.13, 0.60, 150, op, ref=pat[1])
        self.text(sc, 0.13, 0.385, "the corrupted 1", op, strong=True, size=0.034)
        ys = (0.385, 0.60, 0.815)
        for n, y in enumerate(ys):
            a, b = (0.225, 0.60), (0.405, y)
            bend = 0.05 * (1 - n)
            self.arc(sc, a, b, BLUE, op, smoothstep(0.02 + 0.08 * n, 0.2 + 0.08 * n, u), bend, 2.4)
            self.pulse(sc, a, b, (u - 0.1 - 0.08 * n) / 0.22, BLUE, op, bend)
            self.picture(sc, pat[n], 0.46, y, 84, op)
            show = op * smoothstep(0.3 + 0.08 * n, 0.4 + 0.08 * n, u)
            self.text(sc, 0.525, y - 0.03, "score %.1f" % d["scores"][n], show, anchor="lm")
            self.text(sc, 0.525, y + 0.03, "weight %.2f" % self.prob[n], show, anchor="lm",
                      strong=True, size=0.036)
            out = smoothstep(0.55, 0.75, u)
            self.arrow(sc, (0.665, y), (0.765, 0.60 + (y - 0.60) * 0.35), BLUE,
                       op * (0.25 + 0.75 * float(self.prob[n])), out,
                       width=1.2 + 3.0 * float(self.prob[n]))
        out = op * smoothstep(0.7, 0.85, u)
        self.digit(sc, self.blend, 0.86, 0.60, 150, out, frame=BLUE)
        self.text(sc, 0.86, 0.385, "memories × weights", out, strong=True, size=0.034)
        self.text(sc, 0.86, 0.815, "rounded to ink or paper:", out)
        self.text(sc, 0.86, 0.86, "%d pixels off the stored 1" % self.blend_differ, out,
                  strong=True)

    def phase_6(self, sc, u, op):
        loads = np.asarray(self.loads, float)
        left = self.chart(((0.09, 0.48), (0.43, 0.72)), loads, [0.0, 1.0], xlog=True)
        self.axes(sc, left, op, "digits stored", "recalled",
                  xticks=[(v, "%d" % v) for v in (2, 4, 8, 16, 32, 64, 128, 256)],
                  yticks=[(0, "0"), (0.5, "half"), (1, "all")])
        right = self.chart(((0.60, 0.95), (0.43, 0.72)), 100 * self.flips, [0.0, 1.0])
        self.axes(sc, right, op, "pixels flipped, %", "recalled",
                  xticks=[(v, "%d" % v) for v in (0, 15, 30, 45)],
                  yticks=[(0, "0"), (0.5, "half"), (1, "all")])
        self.text(sc, 0.285, 0.325, "more digits, a fifth of the pixels flipped", op,
                  strong=True, size=0.034)
        self.text(sc, 0.775, 0.325, "%d digits, more pixels flipped" % SC.HOP_FLIP_LOADS[0], op,
                  strong=True, size=0.034)
        for r in (0, 1, 2, 3):
            go = (u - 0.04 - 0.14 * r) / 0.26
            width = (2.6, 2.6, 4.2, 2.0)[r]
            progress(self, sc, left, loads, self.by_load[r], HOP_INK[r], op, go, width,
                     xlog=True)
            progress(self, sc, right, 100 * self.flips, self.by_flip[r], HOP_INK[r], op, go,
                     width)
            x = 0.115 + 0.225 * r
            show = op * smoothstep(0.04 + 0.14 * r, 0.10 + 0.14 * r, u)
            self.box(sc, x, 0.895, 14, 14, HOP_INK[r], show, radius=3)
            self.text(sc, x + 0.016, 0.895, HOP_NAMES[r], show, anchor="lm",
                      ink=dark(HOP_INK[r], 0.3), size=0.032)


# ====================================================================== 98: cellular automaton
NCA_AT = 12                               # the growth step whose cells phase 1 looks into
NCA_INK = (AMBER, VIOLET, TEAL)           # the three damages


class NeuralCellularAutomata(Mechanism):
    """A 14 x 14 sheet of cells that all run one small learned rule: what a
    cell senses and computes, a digit growing from one seed cell, how the rule
    was trained on its own past results, the digit held for 960 steps, and
    three damages growing back."""

    TITLE = "A neural cellular automaton"
    PHASES = (("one cell", 11.0), ("growing", 10.0), ("training", 11.0), ("holding", 10.0),
              ("regrowing", 13.0))
    CELLS = ((5, 9), (9, 5), (4, 6))      # the cells whose update phase 1 follows
    SENSES = ("itself", "slope, left to right", "slope, top to bottom")
    HOLD = (240, 480, 960)

    def setup(self):
        d = self.d = SC.nca_data()
        s = self.study
        self.loss = np.asarray(s.panels[1].data["ys"], float)            # (rule, update)
        self.errors = np.asarray(s.panels[3].data["all_errors"], float)  # (rule, run, step)
        self.seeds = len(SC.SEEDS)
        self.pool, self.only = d["states"][0], d["states"][1]            # (run, step, 14, 14, C)
        self.target = d["target"]
        self.channels = self.pool.shape[-1]
        self.weights = int(d["A"][0].size + d["b"][0].size + d["B"][0].size)
        state = self.pool[0, NCA_AT]
        self.sense = SC.nca_sense(state).reshape(14, 14, 3, self.channels)
        self.next = self.pool[0, NCA_AT + 1]
        self.fires = SC.nca_eval_masks(NCA_AT + 1)[NCA_AT] > 0
        e, n, cut = self.errors, self.seeds, SC.NCA_CUT
        self.grown = e[0, 0, cut - 1]
        self.held = e[:n, 0, -1]                       # pool-trained, left alone, step 960
        self.lost = e[n:, 0, -1]                       # seed-only, left alone, step 960
        self.cut = e[0, 1:, cut]                       # the drawn rule just after each damage
        self.mended = e[0, 1:, 2 * cut]                # ... and 48 steps later
        self.worst = float(e[:n, 1:, 2 * cut].max())   # worst pool seed and damage
        self.unmended = e[n:, 1:, 2 * cut]             # seed-only rules, 48 steps after the cut
        k = 25
        self.smooth = np.array([np.convolve(row, np.ones(k) / k, "valid")
                                for row in self.loss[:n]])

    def subtitle_for(self, k):
        cut = SC.NCA_CUT
        return [
            "every cell holds %d numbers and senses only its 3 by 3 neighbourhood: its own "
            "numbers and how they slope across it; one small network of %d weights, the same "
            "in every cell, says how to change them" % (self.channels, self.weights),
            "all 196 cells run the rule together, each on about half the steps: from one "
            "seed cell a 0 grows, and by step %d its error against the target is %s" % (
                cut - 1, small(self.grown)),
            "trained on one MNIST 0 for %d updates: most runs start from the rule's own "
            "earlier results, kept in a pool, and two in every eight have a disc cut out, "
            "so it must hold the digit and mend it" % self.loss.shape[1],
            "left running for %d steps, %d times the longest training run, the rule holds "
            "the 0: error %s at most over three seeds. Trained from the seed only, it "
            "drifts to %s, %s and %s" % ((SC.NCA_RUN, SC.NCA_RUN // SC.NCA_STEPS[1],
                                           small(self.held.max()))
                                          + tuple(small(v) for v in self.lost)),
            "three damages at step %d, two of them larger than any in training: %d steps "
            "later the error is back from %.3f, %.3f and %.3f to %s, %s and %s; the worst of "
            "three seeds is %s" % ((cut, cut) + tuple(self.cut)
                                    + tuple(small(v) for v in self.mended)
                                    + (small(self.worst),)),
        ][k]

    # ------------------------------------------------------------------ drawing
    def sheet(self, sc, img, fx, fy, size, op, ring=(), frame=AXIS, lines=0.22):
        """The cells' visible channel as ink on paper, ``size`` px wide.
        ``ring``: ``(row, col, ink)`` cells outlined. Returns ``cell(row, col)``."""
        v = np.clip(np.asarray(img, float), 0, 1)[..., None]
        n = v.shape[0]
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, size + 8, size + 8, PAPER, op, border=frame, border_w=1.8, radius=5,
                  lift=0.1)
        draw_image(self, sc, c, (_rgb(PAPER) * (1 - v) + _rgb(DIGIT_INK) * v).astype(np.float32),
                   size, size, op, grid=lines)
        dw, dh = self.frac(size, size)

        def cell(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / n, fy - dh / 2 + (r + 0.5) * dh / n
        for r, k, ink in ring:
            self.rect(sc, self.P(*cell(r, k)), size / n + 1, size / n + 1, ink, op, border=ink,
                      border_w=2.4, radius=2, lift=0.8, fill_op=0.0)
        return cell

    def phase_1(self, sc, u, op):
        turn = min(int(u * 3.15), 2)
        within = min(u * 3.15 - turn, 1.0)
        r, k = self.CELLS[turn]
        near = [(r + a, k + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
        cell = self.sheet(sc, self.pool[0, NCA_AT, ..., 0], 0.165, 0.60, 230, op,
                          ring=[(a, b, BLUE) for a, b in near] + [(r, k, AMBER)])
        self.text(sc, 0.165, 0.30, "196 cells, at growth step %d" % NCA_AT, op, strong=True,
                  size=0.034)
        self.text(sc, 0.165, 0.895, "orange: one cell   blue: its neighbours", op)
        # what the cell senses: 3 x 8 numbers
        C = self.channels
        gx, gy, gw, gh = 0.60, 0.50, 176.0, 96.0
        grid = self.grid(sc, self.sense[r, k], gx, gy, gw, gh, op,
                         reveal=smoothstep(0.05, 0.4, within), vmax=1.0)
        self.text(sc, gx, 0.335, "what it senses: 3 × %d numbers" % C, op, strong=True,
                  size=0.034)
        for row, name in enumerate(self.SENSES):
            self.text(sc, gx - 0.112, grid(row, 0)[1], name, op, anchor="rm")
        fires = bool(self.fires[r, k])
        self.text(sc, gx, 0.66, "a coin decides whether it acts now: %s" % (
            "yes" if fires else "not this step"), op)
        a, b = cell(r, k), (gx - 0.10, gy - 0.13)
        self.arc(sc, (a[0], a[1] - 0.02), b, AMBER, op, smoothstep(0.0, 0.3, within), 0.10, 2.2)
        self.pulse(sc, (a[0], a[1] - 0.02), b, within / 0.4, AMBER, op, 0.10)
        # the rule, and the numbers that come out
        rx = 0.875
        self.arrow(sc, (gx + 0.105, gy), (rx - 0.08, gy), VIOLET, op,
                   smoothstep(0.35, 0.5, within), 2.4)
        self.pulse(sc, (gx + 0.105, gy), (rx - 0.08, gy), (within - 0.35) / 0.25, VIOLET, op)
        self.box(sc, rx, gy, 116, 84, VIOLET, op, "the rule", ink=(1, 1, 1), size=0.038,
                 glow=0.2 * smoothstep(0.5, 0.6, within) * (1 - smoothstep(0.7, 0.8, within)))
        self.text(sc, rx, 0.335, "%d weights" % self.weights, op, strong=True, size=0.034)
        self.text(sc, rx, 0.38, "same in every cell", op)
        out = smoothstep(0.6, 0.75, within)
        self.arrow(sc, (rx, gy + 0.095), (rx, 0.725), VIOLET, op, out, 2.4)
        self.grid(sc, self.next[r, k][None], rx - 0.035, 0.775, 176.0, 28.0, op * out, vmax=1.0)
        self.text(sc, rx - 0.035, 0.85, "its %d numbers at step %d" % (C, NCA_AT + 1), op * out,
                  strong=True)
        self.text(sc, rx - 0.035, 0.895, "the first is the ink we see: %.2f" % self.next[r, k, 0],
                  op * out)

    def phase_2(self, sc, u, op):
        last = SC.NCA_CUT - 1
        step = int(min(u / 0.86, 1.0) * last)
        state = self.pool[0, step]
        mse = self.errors[0, 0, :last + 1]
        self.sheet(sc, state[..., 0], 0.165, 0.60, 240, op,
                   ring=[(7, 7, AMBER)] if step < 3 else ())
        self.text(sc, 0.165, 0.295, "step %d" % step, op, strong=True, size=0.038)
        self.text(sc, 0.165, 0.90, "the ink channel of all 196 cells", op)
        self.sheet(sc, self.target, 0.435, 0.475, 112, op, frame=GREEN, lines=0)
        self.text(sc, 0.435, 0.315, "the target", op, ink=dark(GREEN), size=0.034)
        steps = np.arange(last + 1)
        chart = self.chart(((0.62, 0.94), (0.36, 0.58)), steps, mse)
        self.axes(sc, chart, op, "growth step", "error against the target",
                  xticks=[(v, "%d" % v) for v in (0, 12, 24, 36, last)])
        self.plot(sc, chart, steps, mse, AMBER, op, step / float(last), width=2.2)
        self.text(sc, 0.435, 0.645, "error %s" % small(mse[step]), op, strong=True, size=0.036)
        self.text(sc, 0.67, 0.735, "each cell's %d numbers: one of ink, %d hidden" % (
            self.channels, self.channels - 1), op, strong=True)
        for n in range(self.channels):
            x = 0.405 + 0.076 * n
            self.grid(sc, state[..., n], x, 0.84, 56, 56, op, vmax=1.0, lines=0)

    def phase_3(self, sc, u, op):
        updates = self.loss.shape[1]
        xs = np.arange(self.smooth.shape[1]) + 12
        chart = self.chart(((0.09, 0.52), (0.37, 0.70)), [0, updates], [1e-4, 0.3], ylog=True)
        from .losschart import pow10

        self.axes(sc, chart, op, "training update", "error after a run",
                  yticks=[(v, pow10(v)) for v in (1e-4, 1e-3, 1e-2, 1e-1)])
        go = min(u / 0.8, 1.0)
        for n in range(self.seeds - 1, -1, -1):
            progress(self, sc, chart, xs[::6], self.smooth[n, ::6], VIOLET if n == 0 else
                     soft(VIOLET, 0.5), op, go,
                     2.4 if n == 0 else 1.4, ylog=True, dot=n == 0)
        self.text(sc, 0.52, 0.335, "three seeds, mean of 25 updates", op, anchor="rm")
        self.lines(sc, 0.06, 0.815, [
            ("each update: 8 states from a pool of %d of its own results," % SC.NCA_POOL, None),
            ("the worst replaced by the seed, two damaged; run %d to %d steps," % SC.NCA_STEPS,
             None),
            ("compare with the digit, adjust the weights, put the results back", None)],
            op, step=0.043)
        self.text(sc, 0.775, 0.31, "what the rule grows in %d steps" % SC.NCA_CUT, op,
                  strong=True, size=0.034)
        for n, at in enumerate(SC.NCA_SNAPS):
            x, y = 0.66 + 0.115 * (n % 3), 0.46 + 0.28 * (n // 3)
            show = op * (smoothstep(at - 30, at - 4, go * updates) if at else 1.0)
            self.sheet(sc, self.d["snaps"][n], x, y, 84, show, lines=0, frame=VIOLET)
            self.text(sc, x, y + 0.125, "update %d" % at, show)
        self.sheet(sc, self.target, 0.89, 0.74, 84, op, frame=GREEN, lines=0)
        self.text(sc, 0.89, 0.865, "the target", op, ink=dark(GREEN))

    def _held_at(self, u):
        """The step shown at phase fraction ``u`` of "holding", and which stored
        late frame it is (None while the kept states last)."""
        if u < 0.55:
            return int(u / 0.55 * SC.NCA_KEEP), None
        k = min(int((u - 0.55) / 0.12), len(SC.NCA_LATE) - 1)
        return SC.NCA_LATE[k], k

    def phase_4(self, sc, u, op):
        step, late = self._held_at(u)
        n = self.seeds
        pictures = ((0.115, "from the seed only", self.only, 1, RED, n),
                    (0.335, "with the pool", self.pool, 0, GREEN, 0))
        for x, name, states, which, ink, rule in pictures:
            img = states[0, step, ..., 0] if late is None else self.d["late"][which, 0, late]
            self.sheet(sc, img, x, 0.59, 160, op, frame=ink, lines=0.16)
            self.text(sc, x, 0.39, name, op, ink=dark(ink, 0.3), size=0.034)
            self.text(sc, x, 0.795, "error %s" % small(self.errors[rule, 0, step]), op,
                      strong=True, size=0.034)
        self.text(sc, 0.225, 0.32, "step %d" % step, op, strong=True, size=0.038)
        self.text(sc, 0.225, 0.875, "trained on runs of at most %d steps" % SC.NCA_STEPS[1], op)
        from .losschart import pow10

        xs = np.arange(0, SC.NCA_RUN + 1, 8)
        chart = self.chart(((0.57, 0.95), (0.37, 0.70)), xs, [1e-5, 1.0], ylog=True)
        self.axes(sc, chart, op, "step", "error against the target",
                  xticks=[(v, "%d" % v) for v in (0, 240, 480, 720, 960)],
                  yticks=[(v, pow10(v)) for v in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)])
        go = step / float(SC.NCA_RUN)
        for rule in range(2 * n):
            ink = GREEN if rule < n else RED
            progress(self, sc, chart, xs, np.maximum(self.errors[rule, 0, ::8], 1e-5), ink, op,
                     go, 2.6 if rule in (0, n) else 1.4, ylog=True, dot=rule in (0, n))
        self.text(sc, 0.76, 0.815, "three seeds each; thick: the two rules shown", op)
        self.text(sc, 0.76, 0.865, "red: seed only    green: with the pool", op)

    def phase_5(self, sc, u, op):
        cut = SC.NCA_CUT
        first, last = cut - 4, cut + 96
        step = first + int(min(u / 0.82, 1.0) * (last - first))
        done = step >= cut
        boxes = ((3, 3, 8, 8), (0, 7, 13, 13), (7, 0, 13, 13))     # row, col, row, col
        self.text(sc, 0.06, 0.315, "step %d" % step, op, strong=True, size=0.038, anchor="lm")
        self.text(sc, 0.20, 0.315, "cut at step %d" % cut if done else "grown from the seed",
                  op, anchor="lm", ink=dark(RED) if done else None)
        for k, name in enumerate(SC.NCA_DAMAGES):
            x = 0.125 + 0.20 * k
            cell = self.sheet(sc, self.pool[k + 1, step, ..., 0], x, 0.575, 150, op,
                              frame=NCA_INK[k], lines=0.16)
            if done:
                r0, c0, r1, c1 = boxes[k]
                a, b = cell(r0, c0), cell(r1, c1)
                fade = 1.0 - 0.7 * smoothstep(cut + 2, cut + 14, step)
                self.rect(sc, self.P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                          150 * (c1 - c0 + 1) / 14, 150 * (r1 - r0 + 1) / 14, RED, op * fade,
                          border=RED, border_w=2.6, radius=2, lift=0.8, fill_op=0.0)
            self.text(sc, x, 0.385, name, op, ink=dark(NCA_INK[k], 0.3), size=0.034)
            self.text(sc, x, 0.765, "error %s" % small(self.errors[0, k + 1, step]), op,
                      strong=True, size=0.034)
        steps = np.arange(first, last + 1)
        chart = self.chart(((0.70, 0.95), (0.37, 0.66)), steps, self.errors[0, 1:, first:last + 1])
        self.axes(sc, chart, op, "step", "error",
                  xticks=[(v, "%d" % v) for v in (cut, cut + 48, cut + 96)])
        for k in range(3):
            progress(self, sc, chart, steps, self.errors[0, k + 1, first:last + 1], NCA_INK[k],
                     op, (step - first) / float(last - first), 2.4)
        show = op * smoothstep(0.84, 0.92, u)
        low, high = self.unmended.min(), self.unmended.max()
        self.lines(sc, 0.06, 0.845, [
            ("%d steps after the cut, three seeds, three damages: at most %s"
             % (cut, small(self.worst)), None),
            ("the rule trained from the seed only: still %.3f to %.3f" % (low, high),
             dark(RED))], show, step=0.05, size=0.032)


# ====================================================================== shared: a 2-D plane
class Plane:
    """A square window of a 2-D space drawn ``size`` px wide at layout
    ``(fx, fy)``: maps points of the space to layout fractions."""

    def __init__(self, anim, fx, fy, size, centre, half):
        self.anim, self.fx, self.fy, self.size = anim, fx, fy, size
        self.centre, self.half = np.asarray(centre, float), float(half)

    def to(self, xy):
        dw, dh = self.anim.frac(self.size, self.size)
        q = (np.asarray(xy, float) - self.centre) / (2 * self.half)
        return self.fx + float(q[0]) * dw, self.fy - float(q[1]) * dh

    def inside(self, xy, margin=0.0):
        return bool(np.all(np.abs(np.asarray(xy, float) - self.centre) <= self.half - margin))

    def draw(self, sc, op, border=AXIS):
        a = self.anim
        a.rect(sc, a.P(float(self.fx), float(self.fy)), self.size + 8, self.size + 8, PAPER, op,
               border=border, border_w=1.8, radius=5, lift=-0.5)     # behind the plane's lines

    def dot(self, sc, xy, r, ink, op, hollow=False):
        if not self.inside(xy) or op <= 0.01:
            return
        a = self.anim
        at = a.P(*self.to(xy))
        if hollow:
            a.disk(sc, at, r, PAPER, op, ring=ink, ring_w=2.2, outline=0.0, lift=1.2)
        else:
            a.disk(sc, at, r, ink, op, outline=0.0, lift=1.4)

    def path(self, sc, pts, ink, op, width=2.2):
        """A polyline through ``pts`` (the parts inside the window)."""
        a = self.anim
        run = []
        for q in pts:
            if self.inside(q):
                run.append(a.P(*self.to(q)))
            else:
                if len(run) > 1:
                    a.polyline(sc, run, width, ink, op, bias=0.3)
                run = []
        if len(run) > 1:
            a.polyline(sc, run, width, ink, op, bias=0.3)

    def arrow(self, sc, p, q, ink, op, u=1.0, width=2.2):
        """A straight arrow, drawn as one segment (a field's arrows are short)."""
        if self.inside(p) and self.inside(q) and op > 0.01 and u > 0:
            a = self.anim
            p = np.asarray(p, float)
            start, end = a.P(*self.to(p)), a.P(*self.to(p + (np.asarray(q, float) - p) * min(u, 1)))
            a.polyline(sc, [start, end], width, ink, op, bias=0.3)
            a._head(sc, start, end, ink, op, width * 0.7)


def partway(traj, s):
    """Positions ``s`` steps (fractional) along ``traj`` (steps + 1, ..., 2)."""
    s = float(np.clip(s, 0, len(traj) - 1))
    i = min(int(s), len(traj) - 2)
    return traj[i] + (traj[i + 1] - traj[i]) * (s - i)


# ====================================================================== 99: neural ODE
class View:
    """A rectangular window of the plane drawn ``w x h`` px at layout ``(fx, fy)``,
    ``scale`` px to the unit both ways. The plane is shown turned: its second
    coordinate runs across the screen and its first runs up, because the learned
    flow carries the points furthest along the second. ``centre`` is the point of
    the plane at the middle of the window."""

    def __init__(self, anim, fx, fy, w, h, centre, scale):
        self.anim, self.fx, self.fy, self.w, self.h = anim, fx, fy, float(w), float(h)
        self.centre, self.scale = np.asarray(centre, float), float(scale)

    @classmethod
    def around(cls, anim, fx, fy, w, h, points, margin=1.08):
        """The view of at most ``w x h`` px that holds every one of ``points``."""
        pts = np.asarray(points, float).reshape(-1, 2)
        lo, hi = pts.min(0), pts.max(0)
        across, up = max(hi[1] - lo[1], 1e-6), max(hi[0] - lo[0], 1e-6)
        scale = min(w / across, h / up) / margin
        return cls(anim, fx, fy, w, h, (lo + hi) / 2, scale)

    def between(self, other, t):
        """The view a fraction ``t`` of the way from this one to ``other``."""
        def mix(a, b):
            return a + (b - a) * t
        return View(self.anim, mix(self.fx, other.fx), mix(self.fy, other.fy),
                    mix(self.w, other.w), mix(self.h, other.h),
                    mix(self.centre, other.centre),
                    float(np.exp(mix(np.log(self.scale), np.log(other.scale)))))

    def px(self, xy):
        """``xy`` as px right of and above the window's middle."""
        q = (np.asarray(xy, float) - self.centre) * self.scale
        return np.array([q[..., 1], q[..., 0]])

    def at(self, px):
        dw, dh = self.anim.frac(float(px[0]), float(px[1]))
        return self.fx + dw, self.fy - dh

    def to(self, xy):
        return self.at(self.px(xy))

    def inside(self, xy):
        a, b = self.px(xy)
        return bool(abs(a) <= self.w / 2 and abs(b) <= self.h / 2)

    def draw(self, sc, op, border=AXIS):
        a = self.anim
        a.rect(sc, a.P(float(self.fx), float(self.fy)), self.w + 8, self.h + 8, PAPER, op,
               border=border, border_w=1.8, radius=5, lift=-0.5)

    def dot(self, sc, xy, r, ink, op, hollow=False):
        if op <= 0.01 or not self.inside(xy):
            return
        a = self.anim
        at = a.P(*self.to(xy))
        if hollow:
            a.disk(sc, at, r, PAPER, op, ring=ink, ring_w=2.2, outline=0.0, lift=1.2)
        else:
            a.disk(sc, at, r, ink, op, outline=0.0, lift=1.4)

    def clip(self, p, q):
        """The part of the segment ``p``-``q`` (px) inside the window, or None."""
        t0, t1 = 0.0, 1.0
        d = q - p
        for k, half in ((0, self.w / 2), (1, self.h / 2)):
            for sign in (1.0, -1.0):
                den, num = sign * d[k], half - sign * p[k]
                if abs(den) < 1e-12:
                    if num < 0:
                        return None
                    continue
                t = num / den
                if den > 0:
                    t1 = min(t1, t)
                else:
                    t0 = max(t0, t)
        if t0 >= t1:
            return None
        return p + d * t0, p + d * t1

    def path(self, sc, pts, ink, op, width=2.2, head=False):
        """A polyline through ``pts``, cut where it leaves the window."""
        if op <= 0.01:
            return
        a = self.anim
        px = self.px(np.asarray(pts, float)).T
        run, last = [], None
        for p, q in itertools.pairwise(px):
            cut = self.clip(p, q)
            if cut is None:
                if len(run) > 1:
                    a.polyline(sc, run, width, ink, op, bias=0.3)
                run = []
                continue
            s, e = (a.P(*self.at(v)) for v in cut)
            if not run or not np.allclose(cut[0], p):
                if len(run) > 1:
                    a.polyline(sc, run, width, ink, op, bias=0.3)
                run = [s]
            run.append(e)
            last = (s, e)
            if not np.allclose(cut[1], q):
                a.polyline(sc, run, width, ink, op, bias=0.3)
                run = []
        if len(run) > 1:
            a.polyline(sc, run, width, ink, op, bias=0.3)
        if head and last is not None:
            a._head(sc, last[0], last[1], ink, op, width * 0.7)

    def line(self, sc, w, c, ink, op, go=1.0, width=2.8):
        """The straight line ``w . z + c = 0``, drawn across the window as far as ``go``."""
        w = np.asarray(w, float)
        foot = -c * w / (w @ w)
        along = np.array([-w[1], w[0]]) / np.linalg.norm(w)
        reach = (self.w + self.h) / self.scale
        s = np.linspace(-reach, reach, 2)
        ends = self.clip(*self.px(foot + s[:, None] * along).T)
        if ends is None or go <= 0:
            return
        a = self.anim
        start = ends[0]
        a.polyline(sc, [a.P(*self.at(start)),
                        a.P(*self.at(start + (ends[1] - start) * min(go, 1.0)))], width, ink,
                   op, bias=0.3)


ODE_INK = {0: BLUE, 5: RED, 8: GREEN}


class NeuralODE(Mechanism):
    """Held-out MNIST 0s, 5s and 8s as points in a plane; a learned field that
    carries the 0s and 8s away from the 5s so that a straight line can tell
    them apart; residual layers as Euler steps of that flow; and the adjoint's
    backward recomputation."""

    TITLE = "A neural ODE"
    PHASES = (("the task", 11.0), ("the flow", 14.0), ("layers as steps", 15.0),
              ("the adjoint", 13.0))
    COUNTS = (1, 2, 4, 8, 16)
    INK = {1: RED, 2: AMBER, 4: VIOLET, 8: TEAL, 16: BLUE}  # noqa: RUF012
    SEED_INK = (RED, AMBER, VIOLET)

    def setup(self):
        d = self.d = SC.ode_data()
        s = self.study
        chart = s.panels[2].data
        self.steps = [int(v) for v in chart["x"]]
        self.acc = np.asarray(chart["ys"], float)                 # (seed, steps)
        self.ref = np.asarray(chart["reference"], float)
        self.straight = float(chart["line"][0])
        self.adj_steps = [int(v) for v in s.panels[3].data["x"]]
        self.adjoint = np.asarray(s.panels[3].data["ys"], float)[0]
        self.stored = [int(v) for v in s.panels[3].data["stored"]]
        self.kept = int(s.panels[3].data["kept"][0])
        self.count = len(d["test"])
        self.never = float((d["test_digits"] != 5).mean())       # right by never answering 5
        self.model = SC.ode_model(d)
        self.digits = d["test_digits"][:SC.ODE_DRAWN]
        self.paths = {n: d["path_%d" % n] for n in SC.ODE_STEPS}
        self.flow = self.paths[SC.ODE_TRAIN_STEPS]
        exact = d["reference_ends"][0][:SC.ODE_DRAWN]
        miss = np.linalg.norm(self.paths[1][-1] - exact, axis=1)
        self.focus = int(np.argsort(miss)[len(miss) // 2])      # the median one-step miss
        self.exact = exact[self.focus]
        f = self.focus
        self.track = {n: self.paths[n][:, f] for n in self.paths}
        self.back = d["back_16"][:, f]
        self.back_gap = float(np.linalg.norm(self.back[-1] - self.track[16][0]))
        self.end_gap = {n: float(np.linalg.norm(self.track[n][-1] - self.exact))
                        for n in self.COUNTS}
        self.examples = [int(np.flatnonzero(self.digits == c)[0]) for c in SC.ODE_DIGITS]
        self.near = View.around(self, 0.20, 0.605, 290, 290, self.flow[0])
        self.wide = View.around(self, 0.50, 0.555, 610, 170, self.flow, margin=1.04)
        self.wide.w, self.wide.h = 610.0, min(170.0, 1.06 * self.wide.scale * float(
            np.ptp(self.flow[..., 0])))
        lattice = d["grid"][0]
        g = np.linspace(0, 1, 9)
        lo, hi = lattice.reshape(-1, 2).min(0), lattice.reshape(-1, 2).max(0)
        self.lattice = lo + np.stack(np.meshgrid(g, g), -1).reshape(-1, 2) * (hi - lo)
        self.field = SC.ode_field(self.model, self.lattice)[0]

    def subtitle_for(self, k):
        a, steps = 100 * self.acc, self.steps
        at = {n: steps.index(n) for n in (1, 4, 16)}
        return [
            "each dot is a held-out MNIST digit placed by its two main PCA coordinates; the "
            "5s lie between the 0s and the 8s, so the best straight line is right only %.1f%% "
            "of the time: no better than never answering 5 (%.1f%%)" % (
                100 * self.straight, 100 * self.never),
            "a neural ODE learns a field of velocities and lets every point drift along it "
            "for one unit of time; a straight line then gets %.1f%% of the %d held-out digits "
            "right (three seeds: %.1f to %.1f)" % (
                100 * self.ref[0], self.count, 100 * self.ref.min(), 100 * self.ref.max()),
            "a residual layer is one Euler step of that drift, z + f(z)/n: the same weights as "
            "1 layer get %.1f%%, as 4 %.1f%%, as 16 %.1f%% (exact flow %.1f%%); the other two "
            "seeds lose only %.1f and %.1f points at 1 layer" % (
                a[0, at[1]], a[0, at[4]], a[0, at[16]], 100 * self.ref[0],
                100 * self.ref[1] - a[1, at[1]], 100 * self.ref[2] - a[2, at[1]]),
            "backprop stores every layer's output; the adjoint method keeps only the last "
            "and steps backwards, recomputing. Its gradient is off by %.0f%% at %d steps and "
            "%.1f%% at %d: right only as the steps shrink" % (
                100 * self.adjoint[self.adj_steps.index(16)], 16, 100 * self.adjoint[-1],
                self.adj_steps[-1]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _dots(self, sc, view, pts, op, r=2.7):
        for xy, digit in zip(pts, self.digits):
            view.dot(sc, xy, r, ODE_INK[int(digit)], op)

    def _key(self, sc, fx, fy, op, across=False):
        rows = [("blue: 0s", BLUE), ("red: 5s", RED), ("green: 8s", GREEN)]
        for n, (say, ink) in enumerate(rows):
            self.text(sc, fx + (0.13 * n if across else 0.0), fy + (0.0 if across else 0.05 * n),
                      say, op, ink=dark(ink, 0.25), anchor="lm", size=0.034)

    def phase_1(self, sc, u, op):
        d = self.d
        view = self.near
        view.draw(sc, op)
        self._dots(sc, view, self.flow[0], op)
        for n, i in enumerate(self.examples):
            y = 0.40 + 0.20 * n
            ink = ODE_INK[int(self.digits[i])]
            self.digit(sc, d["pixels"][i], 0.47, y, 74, op, frame=ink)
            self.arrow(sc, (0.425, y), view.to(self.flow[0, i]), ink, op * 0.9,
                       smoothstep(0.02, 0.2, u), 1.8, head=False)
        self._key(sc, 0.55, 0.36, op)
        self.lines(sc, 0.55, 0.535, [
            ("%d held-out digits (%d drawn)," % (self.count, SC.ODE_DRAWN), None),
            ("784 pixels squeezed to 2 numbers", None)], op, step=0.048, size=0.032)
        go = smoothstep(0.45, 0.7, u)
        view.line(sc, d["line_w"], d["line_c"], DIGIT_INK, op, go)
        show = op * smoothstep(0.65, 0.8, u)
        self.lines(sc, 0.55, 0.69, [
            ("the question: is it a 5?", layout.STRONG_INK),
            ("the best straight line: %.1f%% right" % (100 * self.straight), layout.STRONG_INK),
            ("never answering 5: %.1f%% right" % (100 * self.never), None)],
            show, step=0.052, size=0.034)

    def _grid(self, sc, view, s, op):
        lines = partway(self.d["grid"], s / 4.0)
        for line in lines:
            view.path(sc, line[::2], GRID, op, 1.1)

    def phase_2(self, sc, u, op):
        view = self.near.between(self.wide, smoothstep(0.03, 0.16, u))
        last = len(self.flow) - 1
        s = float(np.clip((u - 0.22) / 0.46, 0, 1)) * last
        view.draw(sc, op)
        self._grid(sc, view, s, op * 0.9)
        arrows = op * smoothstep(0.10, 0.17, u) * (1.0 - smoothstep(0.24, 0.34, u))
        if arrows > 0.01:
            for g, v in zip(self.lattice, self.field):
                reach = np.linalg.norm(v)
                tip = g + v / max(reach, 1e-9) * min(0.06 * reach, 0.5)
                view.path(sc, [g, tip], BLUE, arrows, 1.6, head=True)
        self._dots(sc, view, partway(self.flow, s), op)
        done = smoothstep(0.72, 0.84, u)
        view.line(sc, self.model["w"], self.model["c"], DIGIT_INK, op, done)
        said = op * smoothstep(0.13, 0.19, u)              # once the view has widened
        self.text(sc, 0.185, 0.335, "time %.2f" % (s / last), said, anchor="lm", strong=True,
                  size=0.038)
        self.text(sc, 0.815, 0.335, "every point drifts along the field: dz/dt = f(z)", said,
                  anchor="rm", size=0.034)
        self._key(sc, 0.19, 0.765, said, across=True)
        self.text(sc, 0.815, 0.765, "grey: a grid carried along", said, anchor="rm")
        show = op * smoothstep(0.80, 0.9, u)
        self.text(sc, 0.5, 0.845, "a straight line is now right for %.1f%% of them "
                  "(before the flow: %.1f%%)" % (100 * self.ref[0], 100 * self.straight), show,
                  strong=True, size=0.036)
        self.text(sc, 0.5, 0.895, "three trainings: %s" % ",  ".join(
            "%.1f%%" % (100 * v) for v in self.ref), show)

    def phase_3(self, sc, u, op):
        counts = self.COUNTS
        turn = min(int(u / 0.80 * len(counts)), len(counts) - 1)
        within = float(np.clip(u / 0.80 * len(counts) - turn, 0, 1))
        n = counts[turn]
        pts = [self.track[m] for m in counts] + [self.track[SC.ODE_TRAIN_STEPS]]
        view = View.around(self, 0.29, 0.515, 470, 150, np.vstack(pts), margin=1.12)
        view.draw(sc, op, border=BLUE)
        view.path(sc, self.track[SC.ODE_TRAIN_STEPS], DIGIT_INK, op * 0.6, 1.8)
        view.dot(sc, self.exact, 6.5, DIGIT_INK, op, hollow=True)
        for k, m in enumerate(counts[:turn + 1]):
            ink, tr = self.INK[m], self.track[m]
            s = m * (smoothstep(0.0, 0.7, within) if k == turn else 1.0)
            if k == turn:
                view.path(sc, list(tr[:int(s) + 1]) + [partway(tr, s)], ink, op, 2.8)
                for q in tr[1:int(s) + 1]:
                    view.dot(sc, q, 3.4, ink, op)
            if s >= m:
                view.dot(sc, tr[-1], 5.0, ink, op)
                self.text(sc, 0.445 + 0.024 * k, 0.335, "%d" % m, op, ink=dark(ink, 0.2),
                          size=0.034)
        view.dot(sc, self.track[1][0], 5.0, ODE_INK[int(self.digits[self.focus])], op)
        self.text(sc, 0.045, 0.335, "one point, the same weights as %d layer%s" % (
            n, "" if n == 1 else "s"), op, anchor="lm", strong=True, size=0.034)
        # the stack of layers that does it
        x0, x1, y = 0.05, 0.53, 0.735
        lit = min(int(n * smoothstep(0.0, 0.7, within)), n - 1)
        moving = 0.0 < within < 0.7
        for k in range(n):
            x = x0 + (x1 - x0) * (k + 0.5) / n
            on = k == lit and moving
            fill = self.INK[n] if on else soft(self.INK[n], 0.6) if k < lit or not moving \
                else PAPER
            self.box(sc, x, y, self.span((x1 - x0) / n)[0] - 4, 34, fill, op,
                     "z + f(z)/%d" % n if n <= 4 else None, ink=(1, 1, 1) if on else None,
                     border=self.INK[n], size=0.032, radius=4, glow=0.15 if on else 0.0)
        self.text(sc, 0.045, 0.82, "it ends %.2f from where the exact flow ends (ring; thin "
                  "line: 64 steps)" % self.end_gap[n], op, anchor="lm",
                  ink=dark(self.INK[n], 0.3))
        show = op * smoothstep(0.84, 0.93, u)
        self.text(sc, 0.045, 0.885, "16 layers with a copy of the weights each: the same "
                  "outputs, difference %g" % float(self.d["resnet_difference"]), show,
                  anchor="lm")
        # held-out accuracy against the number of steps, every seed
        xs = np.asarray(self.steps, float)
        chart = self.chart(((0.655, 0.95), (0.37, 0.67)), xs, [0.45, 0.85], xlog=True, pad=0.0)
        self.axes(sc, chart, op, "layers (Euler steps)", "held-out accuracy",
                  xticks=[(v, "%d" % v) for v in self.steps],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.5, 0.6, 0.7, 0.8)])
        chart.hline(sc, self.P, self.straight, op)
        self.text(sc, 0.945, chart.fy(self.straight) + 0.035, "straight line, no flow", op,
                  anchor="rm")
        base = (turn - 1 + smoothstep(0.2, 0.8, within)) if turn else 0.0
        go = (base + 2.0 * smoothstep(0.82, 0.92, u)) / (len(self.steps) - 1)
        for seed in (2, 1, 0):
            progress(self, sc, chart, xs, self.acc[seed], self.SEED_INK[seed], op,
                     max(go, 1e-3), 3.0 if seed == 0 else 1.6, xlog=True, dot=seed == 0)
        self.text(sc, 0.80, 0.80, "three trainings; red: the one drawn", op)

    def phase_4(self, sc, u, op):
        tr, back = self.track[16], self.back
        fwd = float(np.clip(u / 0.28, 0, 1)) * 16
        rev = float(np.clip((u - 0.36) / 0.34, 0, 1)) * 16
        view = View.around(self, 0.185, 0.555, 290, 150, np.vstack([tr, back]), margin=1.15)
        view.draw(sc, op, border=BLUE)
        self.text(sc, 0.185, 0.335, "the same point, 16 steps", op, strong=True, size=0.034)
        view.path(sc, list(tr[:int(fwd) + 1]) + [partway(tr, fwd)], RED, op, 2.6)
        for q in tr[:int(fwd) + 1]:
            view.dot(sc, q, 3.6, RED, op)
        if rev > 0:
            view.path(sc, list(back[:int(rev) + 1]) + [partway(back, rev)], VIOLET, op, 2.2)
            view.dot(sc, partway(back, rev), 5.0, VIOLET, op)
        self.text(sc, 0.185, 0.76, "forward: every state stored" if rev <= 0 else
                  "backward: each state recomputed", op, ink=dark(RED if rev <= 0 else VIOLET),
                  size=0.034)
        self.text(sc, 0.185, 0.815, "it arrives %.2f from where it started" % self.back_gap,
                  op * smoothstep(0.98, 1.0, rev / 16), strong=True)
        # what each method holds in memory
        x0, step = 0.40, 0.0142
        self.text(sc, 0.515, 0.335, "backprop keeps", op, ink=dark(RED), size=0.034)
        self.text(sc, 0.515, 0.475, "the adjoint keeps", op, ink=dark(VIOLET), size=0.034)
        kept = int(fwd)
        at = 16 - int(rev) if rev > 0 else kept
        for k in range(17):
            self.box(sc, x0 + k * step, 0.39, 10, 26, RED if k <= kept else PAPER, op, border=RED,
                     radius=2)
            self.box(sc, x0 + k * step, 0.53, 10, 26, VIOLET if k == at else PAPER, op,
                     border=VIOLET, radius=2)
        at64 = self.adj_steps.index(64)
        self.lines(sc, 0.40, 0.63, [
            ("numbers held, %s points" % f"{self.count:,}", None),
            ("at 64 steps (exact):", None),
            (f"backprop  {self.stored[at64]:,}", dark(RED)),
            (f"adjoint  {self.kept:,}", dark(VIOLET))],
            op * smoothstep(0.3, 0.4, u), step=0.048, size=0.032)
        # the study's measurement of what that costs the gradient
        xs = np.asarray(self.adj_steps, float)
        pct = 100 * self.adjoint
        chart = self.chart(((0.715, 0.95), (0.385, 0.70)), xs, [1.0, 100.0], xlog=True,
                           ylog=True)
        self.axes(sc, chart, op, "steps", "adjoint gradient error",
                  xticks=[(v, "%d" % v) for v in (4, 16, 64, 256)],
                  yticks=[(v, "%g%%" % v) for v in (1, 3, 10, 30, 100)])
        progress(self, sc, chart, xs, pct, VIOLET, op, smoothstep(0.55, 0.85, u), 3.0,
                 xlog=True, ylog=True)
        self.text(sc, 0.83, 0.82, "against backprop,", op * smoothstep(0.8, 0.9, u))
        self.text(sc, 0.83, 0.865, "same number of steps", op * smoothstep(0.8, 0.9, u))


# ====================================================================== 100: evolution strategies
ES_H = np.array([[4.0, 1.5], [1.5, 1.0]])        # the study's exact quadratic bowl
ES_SIGMA, ES_NET_SIGMA = 0.05, 0.01


def es_trace():
    """The study's evolution-strategies run (:func:`voxanim.next_systems.evolution`)
    replayed with its seed, keeping what it threw away: the 64 nudges of the
    first step on the bowl with the loss of each twin, and for the MNIST
    network its weights, backprop's gradient, the first update's nudges and
    twin losses, and the estimator's error at every population size."""

    def build():
        rng = np.random.default_rng(17)
        H = ES_H
        paths, losses, alignment, keep = [], [], [], {}
        for method in range(2):
            w = np.array([2.0, -2.0])
            path, ls, cs = [], [], []
            for step in range(101):
                true = H @ w
                path.append(w.copy())
                ls.append(0.5 * w @ H @ w)
                if method:
                    eps = rng.normal(size=(64, 2))
                    plus = w + ES_SIGMA * eps
                    minus = w - ES_SIGMA * eps
                    fp = np.einsum("ni,ij,nj->n", plus, H, plus) / 2
                    fm = np.einsum("ni,ij,nj->n", minus, H, minus) / 2
                    estimate = ((fp - fm)[:, None] * eps).mean(0) / 0.1
                    if step == 0:
                        keep.update(eps0=eps, fp0=fp, fm0=fm, est0=estimate, true0=true)
                else:
                    estimate = true
                cs.append(estimate @ true
                          / max(np.linalg.norm(estimate) * np.linalg.norm(true), 1e-12))
                if step == 100:
                    break
                w -= 0.12 * estimate
            paths.append(path)
            losses.append(ls)
            alignment.append(cs)
        x, y, _, _ = S._digit_data(48)
        P = S.mlp_init(49, 5, 10, rng)
        theta = S._pack(P)
        true = S._pack(S.mlp_loss(P, x, y)[1])
        sizes, errors = np.array([8, 32, 128, 512]), []
        for n in sizes:
            eps = rng.normal(size=(n, len(theta)))
            estimate = np.zeros_like(theta)
            for e in eps:
                a = S.mlp_loss(S._unpack(theta + 0.005 * e, P), x, y, False)[0]
                b = S.mlp_loss(S._unpack(theta - 0.005 * e, P), x, y, False)[0]
                estimate += (a - b) * e / (0.01 * n)
            errors.append(np.linalg.norm(estimate - true) / np.linalg.norm(true))
        network = []
        for method in range(2):
            weights = theta.copy()
            trajectory = []
            for step in range(81):
                model = S._unpack(weights, P)
                if method:
                    loss, _ = S.mlp_loss(model, x, y, False)
                else:
                    loss, grads = S.mlp_loss(model, x, y)
                trajectory.append(loss)
                if step == 80:
                    break
                if method:
                    perturb = rng.normal(size=(64, len(weights)))
                    gradient = np.zeros_like(weights)
                    twins = []
                    for eps in perturb:
                        plus = S.mlp_loss(S._unpack(weights + ES_NET_SIGMA * eps, P), x, y,
                                          False)[0]
                        minus = S.mlp_loss(S._unpack(weights - ES_NET_SIGMA * eps, P), x, y,
                                           False)[0]
                        twins.append((plus, minus))
                        gradient += (plus - minus) * eps / (2 * ES_NET_SIGMA * len(perturb))
                    if step == 0:
                        keep.update(net_eps=perturb, net_twins=np.array(twins), net_est=gradient)
                else:
                    gradient = S._pack(grads)
                weights -= 0.15 * gradient
            network.append(trajectory)
        return dict(paths=np.array(paths), loss=np.array(losses), alignment=np.array(alignment),
                    sizes=sizes, error=np.array(errors), network_loss=np.array(network),
                    theta=theta, net_true=true, images=x[:6], **keep)

    from .cache import cached_npz

    return {k: np.asarray(v) for k, v in cached_npz(
        "mech-evolution-strategies", _settings("next_systems.py", "mech_curios.py"),
        build).items()}


def level_curve(H, level, n=720):
    """Points of the ellipse ``w' H w / 2 = level`` (closed)."""
    lam, V = np.linalg.eigh(H)
    a = np.linspace(0, 2 * np.pi, n + 1)
    return (np.stack([np.cos(a), np.sin(a)], 1) * np.sqrt(2 * level / lam)) @ V.T


class EvolutionStrategies(Mechanism):
    """A cloud of nudged copies of the weights and their losses; the pairs'
    votes averaging into a direction; that direction against the true
    gradient on an exact bowl; and the same estimator training a small MNIST
    network of 310 weights."""

    TITLE = "Evolution strategies"
    PHASES = (("a population", 11.0), ("the vote", 13.0), ("same bowl", 11.0),
              ("310 weights", 13.0), ("training", 10.0))
    ZOOM = 0.17                         # half-width of the enlarged patch around the weights

    def setup(self):
        d = self.d = es_trace()
        s = self.study
        xy = np.asarray(s.panels[0].data["xy"], float)
        self.paths = xy.reshape(2, 101, 2)
        self.bowl = np.asarray(s.panels[1].data["ys"], float)
        self.align = np.asarray(s.panels[2].data["ys"], float)
        self.net = np.asarray(s.panels[3].data["ys"], float)
        self.w0 = self.paths[1, 0]
        self.rings = [level_curve(ES_H, v) for v in (0.1, 0.5, 1.5, 3.0, 5.0, 8.0)]
        f0 = 0.5 * self.w0 @ ES_H @ self.w0
        self.zoom_rings = [level_curve(ES_H, f0 + dv, 9000) for dv in (-0.6, -0.3, 0, 0.3, 0.6)]
        self.zoom_rings = [r[np.all(np.abs(r - self.w0) < self.ZOOM * 1.3, 1)]
                           for r in self.zoom_rings]
        eps, diff = d["eps0"], d["fp0"] - d["fm0"]
        votes = diff[:, None] * eps / (2 * ES_SIGMA)
        self.running = np.cumsum(votes, 0) / np.arange(1, 65)[:, None]
        true = d["true0"]
        self.running_cos = self.running @ true / (
            np.linalg.norm(self.running, axis=1) * np.linalg.norm(true))
        self.angle = float(np.degrees(np.arccos(self.align[1, 0])))
        twins = d["net_twins"]
        nvotes = (twins[:, 0] - twins[:, 1])[:, None] * d["net_eps"] / (2 * ES_NET_SIGMA)
        self.net_running = np.cumsum(nvotes, 0) / np.arange(1, 65)[:, None]
        nt = d["net_true"]
        self.net_cos = self.net_running @ nt / (
            np.linalg.norm(self.net_running, axis=1) * np.linalg.norm(nt))
        self.error = [float(v) for v in d["error"]]
        self.sizes = [int(v) for v in d["sizes"]]

    def subtitle_for(self, k):
        return [
            "evolution strategies uses no gradient: around the current weights it tries 64 "
            "random nudges, each one forwards and backwards, and only measures the loss of each",
            "each pair votes for its better twin, by as much as that twin is better; the "
            "average of the 64 votes lands %.0f° from the true downhill direction "
            "(cosine %.2f)" % (self.angle, self.align[1, 0]),
            "on this exact bowl both take 100 steps of the same size: gradient descent ends "
            "at loss %.5f and evolution strategies at %.5f" % (
                self.bowl[0, -1], self.bowl[1, -1]),
            "a small MNIST network has 310 weights, not 2, and a pair says less about each: "
            "from %d pairs the estimate is off by %.1f times the gradient's size, from %d "
            "pairs by %.2f" % (self.sizes[0], self.error[0], self.sizes[-1], self.error[-1]),
            "yet it trains: after 80 updates the loss is %.2f with backprop and %.2f with "
            "evolution strategies, at 128 forward passes per update and no backward pass" % (
                self.net[0, -1], self.net[1, -1]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _bowl(self, sc, op, fx, size):
        plane = Plane(self, fx, 0.605, size, (0.0, 0.0), 2.75)
        plane.draw(sc, op)
        for ring in self.rings:
            plane.path(sc, ring[::8], AXIS, op * 0.55, 1.2)
        plane.dot(sc, (0.0, 0.0), 3.5, AXIS, op)
        return plane

    def _zoom(self, sc, op, fx, size):
        plane = Plane(self, fx, 0.605, size, self.w0, self.ZOOM)
        plane.draw(sc, op, border=AMBER)
        for ring in self.zoom_rings:
            plane.path(sc, ring, AXIS, op * 0.5, 1.2)
        return plane

    def _twins(self, sc, plane, k, op, r=4.0):
        """Pair ``k``: the better twin green, the other violet."""
        d = self.d
        e = d["eps0"][k] * ES_SIGMA
        better = d["fp0"][k] < d["fm0"][k]
        plane.dot(sc, self.w0 + e, r, GREEN if better else VIOLET, op)
        plane.dot(sc, self.w0 - e, r, VIOLET if better else GREEN, op)

    def phase_1(self, sc, u, op):
        bowl = self._bowl(sc, op, 0.165, 226)
        bowl.dot(sc, self.w0, 5.5, AMBER, op)
        self.text(sc, 0.165, 0.335, "the loss: an exact bowl", op, strong=True, size=0.034)
        self.text(sc, 0.165, 0.88, "lines: equal loss   dot: lowest", op)
        zoom = self._zoom(sc, op * smoothstep(0.08, 0.2, u), 0.505, 290)
        show = op * smoothstep(0.08, 0.2, u)
        a = bowl.to(self.w0)
        for dy in (-0.29, 0.29):
            self.arrow(sc, a, (0.505 - 0.168, 0.605 + dy), AMBER, show * 0.7, 1.0, 1.4,
                       head=False)
        self.text(sc, 0.505, 0.925, "around the weights, ×%d" % round(
            (290 / (2 * self.ZOOM)) / (226 / 5.5)), show, ink=dark(AMBER, 0.3))
        count = int(np.clip((u - 0.25) / 0.6, 0, 1) * 64)
        for k in range(count):
            self._twins(sc, zoom, k, op)
        zoom.dot(sc, self.w0, 6.5, AMBER, show)
        rows = [("orange: the weights now", dark(AMBER, 0.3)),
                ("a nudge ε is random; its twin", None),
                ("is the opposite nudge", None),
                ("green: the twin with lower loss", dark(GREEN)),
                ("violet: the one with higher", dark(VIOLET))]
        self.lines(sc, 0.70, 0.38, rows, op, step=0.05, size=0.032)
        self.text(sc, 0.70, 0.70, "pairs tried: %d of 64" % count, op, anchor="lm", strong=True,
                  size=0.036)
        self.text(sc, 0.70, 0.76, "losses measured: %d" % (2 * count), op, anchor="lm",
                  size=0.034)

    def _vote_at(self, u):
        return int(np.clip(64.0 * (u / 0.72) ** 2, 0, 64))

    def phase_2(self, sc, u, op):
        d = self.d
        zoom = self._zoom(sc, op, 0.19, 280)
        count = self._vote_at(u)
        for k in range(64):
            self._twins(sc, zoom, k, op * (0.9 if k < count else 0.3), 3.4)
        k = min(count, 63)
        e = d["eps0"][k] * ES_SIGMA
        if count < 64:
            zoom.path(sc, [self.w0 - e, self.w0 + e], AXIS, op, 1.6)
            self._twins(sc, zoom, k, op, 6.0)
        scale = 0.13 / float(np.linalg.norm(d["true0"]))
        if count > 0:
            vote = -self.running[count - 1] * scale
            vote *= min(1.0, 0.16 / float(np.abs(vote).max()))       # the first few overshoot
            zoom.arrow(sc, self.w0, self.w0 + vote, RED, op, 1.0, 3.2)
        done = smoothstep(0.76, 0.86, u)
        zoom.arrow(sc, self.w0, self.w0 - d["true0"] * scale, BLUE, op * done, done, 3.2)
        zoom.dot(sc, self.w0, 6.0, AMBER, op)
        self.text(sc, 0.19, 0.925, "red: the votes so far   blue: true downhill", op)
        rows = [("pair %d of 64" % (k + 1), None),
                ("loss with the nudge:  %.3f" % d["fp0"][k], None),
                ("loss with its twin:  %.3f" % d["fm0"][k], None),
                ("its vote: towards the lower one,", None),
                ("%.3f strong" % abs(d["fp0"][k] - d["fm0"][k]), None)]
        if count > 0:
            rows.append(("the %d votes so far agree with true downhill: %.2f" % (
                count, self.running_cos[count - 1]), dark(RED)))
        self.lines(sc, 0.47, 0.325, rows, op, step=0.043, size=0.032)
        xs = np.arange(1, 65)
        chart = self.chart(((0.52, 0.95), (0.65, 0.84)), xs, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "pairs counted", "agreement with the true direction (cosine)",
                  xticks=[(v, "%d" % v) for v in (1, 16, 32, 48, 64)],
                  yticks=[(v, "%g" % v) for v in (0, 0.5, 1)])
        if count > 0:
            self.plot(sc, chart, xs, np.clip(self.running_cos, 0, 1), RED, op,
                      (count - 1) / 63.0, width=2.0)

    def phase_3(self, sc, u, op):
        bowl = self._bowl(sc, op, 0.20, 290)
        go = float(np.clip(u / 0.8, 0, 1))
        n = 100 * go ** 2                      # the early, long steps slowly
        for path, ink, width, r in ((self.paths[0], BLUE, 4.6, 6.5),
                                    (self.paths[1], RED, 2.0, 4.0)):
            bowl.path(sc, list(path[:int(n) + 1]) + [partway(path, n)], ink, op, width)
            bowl.dot(sc, partway(path, n), r, ink, op)
        self.text(sc, 0.20, 0.925, "update %d of 100" % int(n), op, strong=True)
        from .losschart import pow10

        steps = np.arange(101)
        chart = self.chart(((0.52, 0.95), (0.36, 0.68)), steps, [5e-5, 10.0], ylog=True)
        self.axes(sc, chart, op, "update", "loss",
                  yticks=[(v, pow10(v)) for v in (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)])
        self.plot(sc, chart, steps, self.bowl[0], BLUE, op, n / 100.0, width=3.4, dot=False)
        self.plot(sc, chart, steps, self.bowl[1], RED, op, n / 100.0, width=1.4)
        i = int(n)
        self.lines(sc, 0.52, 0.80, [
            ("gradient descent (true gradient)   %.5f" % self.bowl[0, i], dark(BLUE)),
            ("evolution strategies (64 pairs)   %.5f" % self.bowl[1, i], dark(RED)),
            ("worst agreement of the 100 estimates: cosine %.2f" % self.align[1].min(), None)],
            op, step=0.048, size=0.032)

    def _strip(self, sc, values, fy, op, label, ink=None, vmax=None):
        v = np.asarray(values, float)
        self.grid(sc, v[None], 0.36, fy, 520.0, 22.0, op, vmax=vmax, lines=0)
        self.text(sc, 0.06, fy - 0.057, label, op, anchor="lm", ink=ink, size=0.032)

    def phase_4(self, sc, u, op):
        d = self.d
        count = int(np.clip(64.0 * (u / 0.7) ** 2, 0, 64))
        k = min(count, 63)
        self._strip(sc, d["theta"], 0.385, op, "the network's 310 weights θ  (49 pixels → 5 → "
                    "10 digits)", dark(AMBER, 0.3))
        self._strip(sc, d["net_eps"][k], 0.50, op, "pair %d of 64: a random nudge ε for every "
                    "weight" % (k + 1), vmax=3.0)
        plus, minus = d["net_twins"][k]
        self.box(sc, 0.20, 0.59, 236, 34, soft(GREEN if plus < minus else VIOLET), op,
                 "loss(θ + σε) = %.4f" % plus, size=0.032)
        self.box(sc, 0.52, 0.59, 236, 34, soft(GREEN if minus < plus else VIOLET), op,
                 "loss(θ − σε) = %.4f" % minus, size=0.032)
        scale = float(max(np.abs(d["net_true"]).max(), np.abs(self.net_running[-1]).max()))
        shown = self.net_running[count - 1] if count else np.zeros(310)
        self._strip(sc, shown, 0.725, op, "evolution strategies: the %d votes so far, averaged"
                    % count, dark(RED), vmax=scale)
        self._strip(sc, d["net_true"], 0.845, op, "backprop: the true gradient (same colour "
                    "scale)", dark(BLUE), vmax=scale)
        self.text(sc, 0.36, 0.915, "red: this weight raises the loss   blue: lowers it", op)
        self.text(sc, 0.835, 0.335, "agreement", op, strong=True, size=0.034)
        self.text(sc, 0.835, 0.385, "cosine %.2f" % (self.net_cos[count - 1] if count else 0.0),
                  op, ink=dark(RED), size=0.040)
        show = op * smoothstep(0.72, 0.82, u)
        self.text(sc, 0.835, 0.50, "estimate's error,", show, strong=True)
        self.text(sc, 0.835, 0.54, "relative to the gradient", show, strong=True)
        full = self.span(0.10)[0]
        for n, (size, err) in enumerate(zip(self.sizes, self.error)):
            self.hbar(sc, 0.80, 0.61 + 0.07 * n, full * err / max(self.error), RED, show,
                      "%d pairs" % size, "%.2f" % err, (u - 0.72 - 0.03 * n) / 0.1, thick=18.0)

    def phase_5(self, sc, u, op):
        steps = np.arange(81)
        go = float(np.clip(u / 0.8, 0, 1))
        chart = self.chart(((0.09, 0.56), (0.37, 0.76)), steps, self.net)
        self.axes(sc, chart, op, "update", "training loss, 48 MNIST digits")
        self.plot(sc, chart, steps, self.net[0], BLUE, op, go, width=2.2)
        self.plot(sc, chart, steps, self.net[1], RED, op, go, width=2.2)
        i = int(go * 80)
        self.lines(sc, 0.09, 0.885, [
            ("update %d:   backprop %.3f    evolution strategies %.3f" % (
                i, self.net[0, i], self.net[1, i]), None)], op, size=0.034)
        # what one update costs each method
        self.text(sc, 0.66, 0.335, "one update with backprop", op, anchor="lm", ink=dark(BLUE),
                  size=0.034)
        self.box(sc, 0.715, 0.405, 92, 30, BLUE, op, "forward", ink=(1, 1, 1))
        self.box(sc, 0.835, 0.405, 92, 30, soft(BLUE), op, "backward", ink=dark(BLUE, 0.4))
        self.text(sc, 0.66, 0.50, "one update with evolution strategies", op, anchor="lm",
                  ink=dark(RED), size=0.034)
        beat = (go * 10) % 1.0 if go < 1 else 1.0       # the 128 passes of one update, slowed
        lit = int(beat * 128) + 1
        for n in range(128):
            r, c = divmod(n, 16)
            self.box(sc, 0.668 + c * 0.0183, 0.56 + r * 0.034, 12, 12, RED if n < lit else PAPER,
                     op, border=RED, radius=2)
        self.text(sc, 0.66, 0.85, "128 forward passes: 64 pairs of twins", op, anchor="lm")
        self.text(sc, 0.66, 0.895, "no backward pass", op, anchor="lm", strong=True)


ANIMATIONS = {
    "hopfield-network": HopfieldNetwork,
    "neural-cellular-automata": NeuralCellularAutomata,
    "neural-ode": NeuralODE,
    "evolution-strategies": EvolutionStrategies,
}
