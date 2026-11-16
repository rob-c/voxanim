"""Label embeddings: animations 106-109 of the "next" catalogue (see :mod:`voxanim.mech`).

A class label as a learned vector - a row of a small table, looked up by the label -
on MNIST's ten digit labels: the table and what its geometry comes to mean; the table
that every classifier already has as its last layer; the row handed to an
autoencoder's decoder; and what swapping that row draws.

The four draw the experiments of :mod:`voxanim.study_labels`, which record every array
shown here (tables, their states along training, features, scores, codes, decoded
pictures): nothing is computed in this module but layout.

One colour means one thing throughout: blue is the one-hot / plain model, red the
learned or label-told one, violet the variant told at both ends, ochre the control
(a table never trained, a random label), green label smoothing.
"""

from __future__ import annotations

import numpy as np

from . import study_labels as S
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    soft,
)

BIG = 0.034                                   # headings over a drawing
GREY = (0.56, 0.58, 0.60)
# one colour per digit where ten clouds share a plane
DIGIT10 = (BLUE, GREEN, RED, AMBER, VIOLET, TEAL, MAGENTA, OCHRE, DIGIT_INK, GREY)


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def closeness(dist):
    """A distance matrix as closeness in 0..1 for drawing: the nearest pair 1, the
    farthest 0, the diagonal left empty; a matrix whose pairs are all equally far is
    one flat half."""
    d = np.asarray(dist, float)
    off = d[S.PAIRS]
    span = off.max() - off.min()
    c = np.full(d.shape, 0.5) if span < 1e-9 else (off.max() - d) / span
    return np.clip(c, 0.0, 1.0) * (1 - np.eye(len(d)))


def cloud_texture(xy, classes, upto=10, side=96, lim=None):
    """Points ``xy`` coloured by digit as a ``side`` x ``side`` picture (digits below
    ``upto`` only); ``lim``: ``(x0, x1, y0, y1)``, by default the middle 98% of the cloud."""
    xy = np.asarray(xy, float)
    if lim is None:
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        pad = 0.06 * (hi - lo)
        lim = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    tex = np.tile(np.asarray(PAPER, np.float32), (side, side, 1))
    col = np.clip(((xy[:, 0] - lim[0]) / (lim[1] - lim[0]) * side).astype(int), 0, side - 2)
    row = np.clip(((lim[3] - xy[:, 1]) / (lim[3] - lim[2]) * side).astype(int), 0, side - 2)
    for c in range(min(int(upto), 10)):
        m = classes == c
        for dr in (0, 1):
            for dc in (0, 1):
                tex[row[m] + dr, col[m] + dc] = DIGIT10[c]
    return tex


def spread_apart(points, gap, limit, rounds=80):
    """Where to write the labels of ``points`` (px, centred): each at its point unless
    two would sit within ``gap`` of each other, in which case they are pushed apart."""
    p = np.asarray(points, float)
    q = p.copy()
    n = len(q)
    for _ in range(rounds):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = q[j] - q[i]
                dist = float(np.hypot(*d))
                if dist >= gap:
                    continue
                if dist < 1e-6:                       # the same spot: part them by index
                    angle = 2 * np.pi * (i * 0.37 + j * 0.11)
                    d, dist = np.array([np.cos(angle), np.sin(angle)]), 1.0
                push = (gap - dist) / 2 + 0.25
                q[i] -= d / dist * push
                q[j] += d / dist * push
                moved = True
        q = np.clip(q, -limit, limit)
        if not moved:
            break
    return q


class Labels(Mechanism):
    """What the four animations share: curves with their seed spread, legends, digit
    axes on a matrix, and the boxes of a small network."""

    def trace(self, sc, chart, xs, ys, ink, op, go=1.0, lo=None, hi=None, width=2.6,
              xlog=False):
        """``ys`` against ``xs`` through every measured point, drawn as far as fraction
        ``go`` of the axis; ``lo`` / ``hi``: the seeds' range, as a whisker at each."""
        if op <= 0.01 or go <= 0:
            return
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        tx = np.log(xs) if xlog else xs
        upto = tx[0] + (tx[-1] - tx[0]) * min(float(go), 1.0)
        lx = np.append(tx[tx < upto], upto)
        ly = np.interp(lx, tx, ys)
        pts = [self.P(chart.fx(float(np.exp(a) if xlog else a)), chart.fy(float(b)))
               for a, b in zip(lx, ly)]
        if len(pts) > 1:
            self.polyline(sc, pts, width, ink, op, bias=0.3)
        for i in np.flatnonzero(tx <= upto + 1e-9):
            x = chart.fx(float(xs[i]))
            if lo is not None:
                self.polyline(sc, [self.P(x, chart.fy(float(lo[i]))),
                                   self.P(x, chart.fy(float(hi[i])))], 1.8, ink, 0.8 * op)
            self.disk(sc, self.P(x, chart.fy(float(ys[i]))), 4.2, ink, op, lift=1.4, outline=0.0)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=None):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``."""
        for n, (name, ink) in enumerate(entries):
            y = fy + n * step
            self.box(sc, fx, y, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.016, y, name, op, anchor="lm", ink=dark(ink, 0.3),
                      size=size or 0.031)

    def matrix(self, sc, values, fx, fy, side, op, hot=RED, cold=BLUE, vmax=None, ring=(),
               ring_op=1.0, digits=True, reveal=1.0):
        """A 10 x 10 matrix with the digits 0-9 down its left side; ``ring``: cells
        outlined (with their mirror images)."""
        cell = self.grid(sc, values, fx, fy, side, side, op, vmax=vmax, hot=hot, cold=cold,
                         reveal=reveal)
        if digits:
            x0 = cell(0, 0)[0] - self.frac(side / 10.0)[0] * 0.5 - 0.014
            for r in range(10):
                self.text(sc, x0, cell(r, 0)[1], str(r), op, size=0.029, anchor="mm")
        for r, k in ring:
            for a, b in ((r, k), (k, r)):
                self.rect(sc, self.P(*cell(a, b)), side / 10.0 + 3, side / 10.0 + 3, DIGIT_INK,
                          op * ring_op, border=DIGIT_INK, border_w=2.4, radius=2, lift=0.8,
                          fill_op=0.0)
        return cell

    def row_ring(self, sc, cell, r, cols, w, h, rows, ink, op):
        """An outline around row ``r`` of a ``w`` x ``h`` px grid of ``rows`` rows."""
        x = (cell(r, 0)[0] + cell(r, cols - 1)[0]) / 2
        self.rect(sc, self.P(x, cell(r, 0)[1]), w + 5, h / rows + 3, ink, op, border=ink,
                  border_w=2.6, radius=3, lift=0.8, fill_op=0.0)

    def small(self, sc, img, fx, fy, size, op, mark=None):
        """A small picture; ``mark``: a colour to outline it in."""
        self.digit(sc, img, fx, fy, size, op)
        if mark is not None:
            self.rect(sc, self.P(fx, fy), size + 1, size + 1, mark, op, border=mark,
                      border_w=2.2, radius=2, lift=0.8, fill_op=0.0)

    def stage(self, sc, fx, fy, text, ink, op, w=112, h=58):
        """A part of a network as a box."""
        self.box(sc, fx, fy, w, h, soft(ink, 0.72), op, text, ink=dark(ink, 0.45), border=ink,
                 size=0.032)

    def numbers(self, sc, values, fx, fy, op, ink, w=58, h=30, gap=0.04, fmt="%+.1f"):
        """A short vector as a column of boxed numbers centred at ``(fx, fy)``."""
        n = len(values)
        for i, v in enumerate(values):
            y = fy + (i - (n - 1) / 2) * gap * (1.0 if n > 1 else 0.0)
            self.box(sc, fx, y, w, h, soft(ink, 0.6), op, fmt % v, border=ink, size=0.030)

    def cloud(self, sc, tex, fx, fy, side, op, title=None, ink=None, below=None):
        self.rect(sc, self.P(fx, fy), side + 8, side + 8, PAPER, op,
                  border=AXIS if ink is None else ink,
                  border_w=1.8, radius=5, lift=0.1)
        draw_image(self, sc, self.P(fx, fy), tex, side, side, op, grid=0)
        dh = self.frac(0, side)[1]
        if title:
            self.text(sc, fx, fy - dh / 2 - 0.04, title, op, size=BIG,
                      ink=None if ink is None else dark(ink, 0.3), strong=ink is None)
        if below:
            self.text(sc, fx, fy + dh / 2 + 0.042, below, op, strong=True, size=0.032)


# ====================================================================== 106: the table
class LabelEmbedding(Labels):
    """Ten labels as one-hot markers, all equally far apart; the same labels as rows
    of a table; the table trained to draw each digit from its label, its rows moving
    apart; which labels end up near which, against the digits' mean images and a
    classifier's confusion; and what the width, the seed and the start do to it."""

    TITLE = "A label as a vector"
    PHASES = (("one-hot", 8.0), ("a table", 10.0), ("training", 12.0), ("geometry", 11.0),
              ("seeds", 11.0), ("the start", 8.0))
    SHOWN = (4, 9)                             # the labels looked up in phase 2

    def setup(self):
        d = self.d = S.table_data()
        s = self.study
        m = self.meta = s.metadata
        self.widths = [int(w) for w in s.panels[1].data["x"]]
        self.agree = np.asarray(s.panels[1].data["ys"], float)            # (4, width)
        self.agree_seeds = np.asarray(s.panels[1].data["seeds"], float)   # (4, width, seed)
        self.geometry = np.asarray(s.panels[2].data["geometry"], float)   # (width, pair)
        self.numbers_r = np.asarray(s.panels[2].data["numbers"], float)
        self.starts = np.asarray(s.panels[2].data["start_sweep"], float)  # (3, start)
        self.w8 = self.widths.index(8)
        self.table = d["table_8"][0]
        self.dist8 = np.asarray(s.panels[0].data["values"], float)
        self.pairs = [tuple(int(v) for v in p) for p in m["closest_pairs_width_8"][:3]]
        self.floor = float(m["held_out_error"]["floor"])
        self.map_lim = 1.12 * float(np.abs(d["snaps_2"][:3, -1]).max())

    def subtitle_for(self, k):
        m, w8 = self.meta, self.w8
        a, st = m["agreement"], m["start_size"]
        return [
            "a class label is usually given to a network as a one-hot marker: ten slots, one "
            "of them on. Every label is then exactly as far from every other: 4 is no nearer "
            "to 9 than to 1",
            "a label embedding is a table with one row of numbers per label. The label picks "
            "its row and the row is all the network gets: here it must draw the digit from it",
            "the table starts as tiny random numbers and is trained with the network on %s "
            "digits. From the label alone it can only draw each digit's average: held-out "
            "error %.3f, the averages' own %.3f" % (
                f"{S.N_TRAIN:,}", float(self.d["curve_2"][0, -1]), self.floor),
            "nobody said which digits look alike, yet with 8 numbers a row the closest labels "
            "are %d and %d. The table ranks the 45 pairs as the average images do (%.2f) and "
            "as a classifier's mix-ups do (%.2f); untrained, %.2f" % (
                self.pairs[0] + (a["trained_vs_mean_images"][w8], a["trained_vs_confusion"][w8],
                                 a["untrained_vs_mean_images"][w8])),
            "another seed gives different numbers (correlation %.2f at 8 a row) but much the "
            "same geometry (%.2f). Wider tables agree more between seeds: %.2f with 2 numbers "
            "a row, %.2f with 32" % (
                m["between_seeds"]["numbers"][w8], m["between_seeds"]["geometry"][w8],
                m["between_seeds"]["geometry"][0], m["between_seeds"]["geometry"][-1]),
            "the start matters too. A table started as large random numbers stays what it was "
            "(%.2f like its start, %.2f like the images); started small, it becomes what the "
            "task makes it (%.2f like the images)" % (
                st["trained_vs_its_own_start"][0], st["trained_vs_mean_images"][0],
                st["trained_vs_mean_images"][2]),
        ][k]

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.27, 0.315, "ten labels, ten one-hot markers", op, strong=True, size=BIG)
        pick = 4 if u < 0.5 else 9
        cell = self.matrix(sc, np.eye(10), 0.27, 0.605, 240, op, hot=BLUE)
        self.row_ring(sc, cell, pick, 10, 240, 240, 10, AMBER, op * smoothstep(0.08, 0.2, u))
        self.text(sc, 0.27, 0.885, "label %d: slot %d on, the rest off" % (pick, pick), op,
                  ink=dark(BLUE, 0.3), size=0.032)
        show = op * smoothstep(0.3, 0.5, u)
        self.text(sc, 0.73, 0.315, "how close is each pair?", show, strong=True, size=BIG)
        self.matrix(sc, closeness(self.d["one_hot_dist"]), 0.73, 0.605, 240, show, hot=BLUE,
                    vmax=1.0, ring=[(4, 9), (1, 4)], ring_op=smoothstep(0.6, 0.75, u))
        self.text(sc, 0.73, 0.885, "4 to 9: 1.41      4 to 1: 1.41      every pair: 1.41",
                  op * smoothstep(0.6, 0.75, u), ink=dark(BLUE, 0.3), size=0.032)

    def phase_2(self, sc, u, op):
        d = self.d
        turn = 0 if u < 0.52 else 1
        lab = self.SHOWN[turn]
        w = (u - 0.52 * turn) / (0.52 if turn == 0 else 0.48)
        vmax = float(np.abs(self.table).max())
        self.text(sc, 0.30, 0.315, "the table: 8 numbers per label", op, strong=True, size=BIG)
        cell = self.matrix(sc, self.table, 0.30, 0.62, 200, op, vmax=vmax)
        self.box(sc, 0.085, 0.62, 54, 54, soft(AMBER), op, str(lab), border=AMBER, size=0.05)
        self.text(sc, 0.085, 0.535, "label", op)
        ry = cell(lab, 0)[1]
        self.arrow(sc, (0.118, 0.62), (0.168, ry), AMBER, op, smoothstep(0.02, 0.16, w), 2.6)
        self.row_ring(sc, cell, lab, 8, 200, 200, 10, AMBER, op * smoothstep(0.14, 0.22, w))
        # the row leaves the table
        out = smoothstep(0.24, 0.42, w)
        self.arc(sc, (0.41, ry), (0.515, 0.435), AMBER, op, out, 0.04 if ry > 0.5 else -0.04, 2.6)
        self.pulse(sc, (0.41, ry), (0.515, 0.435), (w - 0.24) / 0.2, AMBER, op,
                   0.04 if ry > 0.5 else -0.04)
        row = op * smoothstep(0.4, 0.5, w)
        self.grid(sc, self.table[lab][None], 0.635, 0.435, 208, 26, row, vmax=vmax)
        self.text(sc, 0.635, 0.375, "row %d: the label's embedding" % lab, row, strong=True,
                  size=0.032)
        self.arrow(sc, (0.635, 0.47), (0.635, 0.555), AMBER, op, smoothstep(0.5, 0.6, w), 2.6)
        self.stage(sc, 0.635, 0.615, "network", RED, op, w=130, h=50)
        self.arrow(sc, (0.635, 0.665), (0.635, 0.715), RED, op, smoothstep(0.62, 0.7, w), 2.6)
        drawn = op * smoothstep(0.68, 0.8, w)
        self.digit(sc, d["drawn"][lab], 0.635, 0.815, 84, drawn, frame=RED)
        self.text(sc, 0.73, 0.815, "what it draws", drawn, anchor="lm", ink=dark(RED, 0.3))
        self.digit(sc, d["means"][lab], 0.885, 0.615, 84, drawn, frame=TEAL)
        self.text(sc, 0.885, 0.515, "the average %d" % lab, drawn, ink=dark(TEAL, 0.3))

    def _snap(self, u):
        """The recorded table state at phase fraction ``u`` (its index, and how far
        towards the next)."""
        n = len(self.d["snaps_2"][0]) - 1
        k = min(u / 0.86, 1.0) * n
        i = int(min(k, n - 1))
        return i, k - i

    def _map(self, sc, pts, fx, fy, side, op, lim, trail=None, size=22):
        """A 2-number table as a plane: a dot where each label's row puts it, its
        digit beside the dot (digits that would cover each other are moved apart)."""
        self.rect(sc, self.P(fx, fy), side + 8, side + 8, PAPER, op, border=AXIS, border_w=1.2,
                  radius=5, lift=0.1)
        half = side / 2.0 - size / 2.0 - 2.0
        true = np.clip(np.asarray(pts, float) / lim, -1, 1) * np.array([1.0, -1.0]) * half
        if trail is not None:
            for c in range(10):
                path = np.clip(trail[:, c] / lim, -1, 1) * np.array([1.0, -1.0]) * half
                self.polyline(sc, [self.shift(self.P(fx, fy, lift=0.4), *p) for p in path], 1.2,
                              DIM, 0.45 * op)
        shown = spread_apart(true, size + 3.0, half)
        centre = self.P(fx, fy)
        for c in range(10):
            dot, lab = self.shift(centre, *true[c]), self.shift(centre, *shown[c])
            self.polyline(sc, [dot, lab], 1.2, AMBER, 0.45 * op)
            self.disk(sc, dot, 3.2, AMBER, op, lift=0.9, outline=0.0)
            self.rect(sc, lab, size, size, soft(AMBER, 0.35), op, border=AMBER, radius=4,
                      lift=1.2)
            self.label(sc, lab, str(c), 0.030, opacity=op, ink=dark(AMBER, 0.6))

    def phase_3(self, sc, u, op):
        d = self.d
        i, f = self._snap(u)
        snaps, steps = d["snaps_2"][0], d["steps_2"][0]
        pts = snaps[i] * (1 - f) + snaps[i + 1] * f
        last = int(np.argmax(steps))                     # the first state at the final step
        self.text(sc, 0.215, 0.315, "a table of 2 numbers per label", op, strong=True, size=BIG)
        self._map(sc, pts, 0.215, 0.605, 244, op, self.map_lim, trail=snaps[:i + 1])
        self.text(sc, 0.215, 0.89, "a dot where each label's row puts it", op)
        now = i + (1 if f > 0.5 else 0)
        self.text(sc, 0.70, 0.315, "what the ten rows draw, after %d updates" % steps[now], op,
                  strong=True, size=BIG)
        for c in range(10):
            self.digit(sc, d["drawn_snaps_2"][now, c], 0.475 + 0.05 * c, 0.405, 42, op)
        xs = np.maximum(steps[:last + 1], 0.7)
        chart = self.chart(((0.50, 0.93), (0.56, 0.80)), [0.7, steps[last]],
                           [0.04, float(d["curve_2"][0].max())], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "updates", "held-out error",
                  xticks=[(v, str(v)) for v in (1, 10, 100, 939)],
                  yticks=[(v, "%.2f" % v) for v in (0.05, 0.10, 0.15, 0.20)])
        chart.hline(sc, self.P, self.floor, op)
        self.text(sc, 0.52, chart.fy(self.floor) - 0.035,
                  "the ten average digits: %.3f" % self.floor, op, anchor="lm",
                  ink=dark(TEAL, 0.3))
        go = (np.log(max(steps[now], 0.7)) - np.log(0.7)) / (np.log(steps[last]) - np.log(0.7))
        self.trace(sc, chart, xs, d["curve_2"][0][:last + 1], RED, op, go, xlog=True)

    def phase_4(self, sc, u, op):
        d, m, w8 = self.d, self.meta, self.w8
        a = m["agreement"]
        ring = self.pairs[:1 + int(u > 0.45) + int(u > 0.6)] if u > 0.3 else []
        cols = (
            (0.185, closeness(self.dist8), AMBER, "the trained table",
             "how close its rows are", None),
            (0.515, closeness(d["mean_dist"]), TEAL, "the average images",
             "table agrees %.2f" % a["trained_vs_mean_images"][w8], 0.12),
            (0.845, d["soft"] * (1 - np.eye(10)), VIOLET, "a classifier's confusion",
             "table agrees %.2f" % a["trained_vs_confusion"][w8], 0.24),
        )
        for x, values, ink, name, note, t0 in cols:
            show = op if t0 is None else op * smoothstep(t0, t0 + 0.12, u)
            self.text(sc, x, 0.315, name, show, ink=dark(ink, 0.3), size=BIG)
            self.matrix(sc, values, x, 0.585, 212, show, hot=ink, ring=ring)
            self.text(sc, x, 0.83, note, show, strong=True, size=0.032)
        named = "ringed: " + ", ".join("%d and %d" % p for p in self.pairs[:len(ring)])
        self.text(sc, 0.185, 0.895, named if ring else "", op, ink=dark(AMBER, 0.3))
        self.text(sc, 0.68, 0.895, "an untrained table agrees %.2f and %.2f" % (
            a["untrained_vs_mean_images"][w8], a["untrained_vs_confusion"][w8]),
            op * smoothstep(0.75, 0.88, u), ink=dark(OCHRE, 0.3))

    def phase_5(self, sc, u, op):
        d, m, w8 = self.d, self.meta, self.w8
        tables = d["table_8"]
        vmax = float(np.abs(tables[:2]).max())
        self.text(sc, 0.225, 0.315, "two seeds, 8 numbers per label", op, strong=True, size=BIG)
        for n in (0, 1):
            x = 0.115 + 0.20 * n
            self.grid(sc, tables[n], x, 0.49, 112, 124, op, vmax=vmax)
            self.matrix(sc, closeness(d["dist"][w8, n]), x, 0.785, 112, op, hot=AMBER,
                        digits=False)
        self.text(sc, 0.405, 0.465, "the numbers", op, anchor="lm")
        self.text(sc, 0.405, 0.515, "r = %.2f" % m["between_seeds"]["numbers"][w8], op,
                  anchor="lm", strong=True, size=BIG)
        self.text(sc, 0.405, 0.76, "closeness", op, anchor="lm")
        self.text(sc, 0.405, 0.81, "r = %.2f" % m["between_seeds"]["geometry"][w8], op,
                  anchor="lm", strong=True, size=BIG)
        ws = np.asarray(self.widths, float)
        chart = self.chart(((0.60, 0.94), (0.37, 0.70)), ws, [-0.1, 1.0], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers per label", "agreement",
                  xticks=[(w, str(int(w))) for w in ws],
                  yticks=[(v, "%.1f" % v) for v in (0.0, 0.5, 1.0)])
        rows = ((self.geometry, AMBER, "two seeds' geometry", 0.10),
                (self.agree_seeds[0], TEAL, "table and average images", 0.32),
                (self.numbers_r, GREY, "two seeds' numbers", 0.54))
        for values, ink, name, t0 in rows:  # noqa: B007
            self.trace(sc, chart, ws, values.mean(1), ink, op, (u - t0) / 0.3, values.min(1),
                       values.max(1), xlog=True)
        self.key(sc, 0.60, 0.835, [(name, ink) for _, ink, name, _ in rows], op, step=0.042)

    def phase_6(self, sc, u, op):
        d, m = self.d, self.meta
        st = m["start_size"]
        cols = ((0.17, closeness(d["sweep_dist"][0, 0]), OCHRE, "started large (1)"),
                (0.50, closeness(d["sweep_dist"][2, 0]), AMBER, "started small (0.01)"),
                (0.83, closeness(d["mean_dist"]), TEAL, "the average images"))
        for x, values, ink, name in cols:
            self.text(sc, x, 0.315, name, op, ink=dark(ink, 0.3), size=BIG)
            self.matrix(sc, values, x, 0.51, 150, op, hot=ink, digits=False)
        full = self.span(0.115)[0]
        self.text(sc, 0.05, 0.725, "table started at", op, anchor="lm")
        self.text(sc, 0.05, 0.805, "like its own start", op, anchor="lm", ink=dark(OCHRE, 0.3))
        self.text(sc, 0.05, 0.875, "like the images", op, anchor="lm", ink=dark(TEAL, 0.3))
        for n, size in enumerate(st["sizes"]):
            x = 0.285 + 0.17 * n
            go = (u - 0.12 - 0.12 * n) / 0.2
            show = op * smoothstep(0.1 + 0.12 * n, 0.2 + 0.12 * n, u)
            self.text(sc, x + 0.05, 0.725, "%g" % size, show, strong=True, size=BIG)
            for y, key, ink in ((0.805, "trained_vs_its_own_start", OCHRE),
                                (0.875, "trained_vs_mean_images", TEAL)):
                v = float(st[key][n])
                self.hbar(sc, x, y, full * max(v, 0.0), ink, show, "", "%.2f" % v, go, thick=18.0,
                          track=full)


# ====================================================================== 107: the classifier
CLF_INK = (BLUE, GREEN, RED, OCHRE)           # one-hot, smoothed, learned, random
CLF_NAMES = ("one-hot targets", "smoothed targets", "learned table", "fixed random table")


class LabelEmbeddingClassifier(Labels):
    """The last layer of a digit classifier as a table of ten label vectors: an
    image's feature scored against each row; four ways to treat that table; their
    held-out accuracy; what the rows come to look like; and where the mistakes go."""

    TITLE = "Label vectors in a classifier"
    PHASES = (("last layer", 12.0), ("four ways", 9.0), ("accuracy", 9.0), ("geometry", 12.0),
              ("mistakes", 10.0))
    IMAGE = 3                                  # the held-out image scored in phase 1 (a 4)

    def setup(self):
        d = self.d = S.classifier_data()
        s = self.study
        self.meta = s.metadata
        self.widths = [int(w) for w in s.panels[0].data["x"]]
        self.acc = np.asarray(s.panels[0].data["seeds"], float)           # (variant, width, seed)
        self.cos = np.asarray(s.panels[1].data["first_seed"], float)      # (variant, 10, 10)
        self.same = np.asarray(s.panels[3].data["same_pairs"], float)     # (variant, width)
        self.wrong = np.asarray(s.panels[3].data["by_variant"], float)    # (variant, 10, 10)
        wide = len(self.widths) - 1
        self.table = S.clf_table(d, "one-hot", self.widths[wide], 0)
        self.feature = d["feats"][0, wide, self.IMAGE]
        self.scores = d["scores"][0, wide, self.IMAGE]
        _x, _y, xt, yt, _ = S.data()
        self.image, self.truth = xt[self.IMAGE], int(yt[self.IMAGE])
        self.order = np.argsort(-self.scores)

    def subtitle_for(self, k):
        m = self.meta
        acc, wide = m["accuracy"], len(self.widths) - 1
        cc, _fn = m["cosine_vs_confusion"], m["four_nine_cosine"]
        same = m["mistakes_same_pairs_as_one_hot"]
        return [
            "every classifier already has a table of label vectors: its last layer. The image "
            "becomes a feature of 16 numbers, and each digit's score is that feature times "
            "the digit's row: here %d wins with %.1f" % (
                self.order[0], self.scores[self.order[0]]),
            "four ways to treat that table, from the same start on the same %s digits: "
            "one-hot targets; targets smoothed by %.1f; a table the feature must point along; "
            "and that table never trained" % (f"{S.N_TRAIN:,}", S.CLF_SMOOTH),
            "held-out accuracy with 16-number rows, %d seeds: %s, %s, %s and %s. Learning the "
            "table buys %.1f points over leaving it random, less than the seeds differ; with "
            "2-number rows, %.0f" % (
                len(S.CLF_SEEDS), pct(acc["one-hot"][wide]), pct(acc["smoothed"][wide]),
                pct(acc["learned"][wide]), pct(acc["random"][wide]),
                100 * (acc["learned"][wide] - acc["random"][wide]),
                100 * (acc["learned"][0] - acc["random"][0])),
            "training shapes the table: rows of digits that get confused point the same way. "
            "Agreement with a separate classifier's mix-ups: %.2f one-hot, %.2f learned, %.2f "
            "random; smoothing evens the rows out: %.2f" % (
                cc["one-hot"][wide], cc["learned"][wide], cc["random"][wide],
                cc["smoothed"][wide]),
            "mistakes go to digits that look alike, whatever the table: with 16-number rows "
            "the random table errs on the same pairs as one-hot (%.2f). With 2-number rows "
            "its mistakes follow its table instead (%.2f)" % (
                same["random"][wide], same["random"][0]),
        ][k]

    def phase_1(self, sc, u, op):
        vmax = float(np.abs(self.table).max())
        self.digit(sc, self.image, 0.10, 0.44, 104, op)
        self.text(sc, 0.10, 0.315, "a held-out %d" % self.truth, op, strong=True, size=BIG)
        self.arrow(sc, (0.10, 0.55), (0.10, 0.62), BLUE, op, smoothstep(0.02, 0.08, u), 2.6)
        self.stage(sc, 0.10, 0.68, "network", BLUE, op, w=120, h=50)
        self.arc(sc, (0.165, 0.66), (0.308, 0.375), BLUE, op, smoothstep(0.08, 0.16, u), 0.07, 2.6)
        self.pulse(sc, (0.165, 0.66), (0.308, 0.375), (u - 0.06) / 0.12, BLUE, op, 0.07)
        fshow = op * smoothstep(0.14, 0.2, u)
        self.grid(sc, self.feature[None], 0.44, 0.375, 230, 20, fshow)
        self.text(sc, 0.44, 0.315, "its feature: 16 numbers", fshow, strong=True, size=BIG)
        cell = self.matrix(sc, self.table, 0.44, 0.655, 230, op, vmax=vmax)
        self.text(sc, 0.44, 0.91, "the last layer: a row of 16 numbers per digit", op)
        # one row at a time: feature times row = score
        k = (u - 0.22) / 0.56 * 10
        zero, full = 0.75, self.span(0.115)[0]
        top = float(np.abs(self.scores).max())
        self.text(sc, 0.80, 0.315, "feature × row, plus a bias", op * smoothstep(0.2, 0.26, u),
                  strong=True, size=BIG)
        for r in range(10):
            if k < r:
                continue
            y = cell(r, 0)[1]
            w = min(k - r, 1.0)
            if w < 1.0:
                self.row_ring(sc, cell, r, 16, 230, 230, 10, AMBER, op)
                self.pulse(sc, (0.44, 0.39), (0.44, y), w * 1.4, BLUE, op)
            v = float(self.scores[r])
            length = full * abs(v) / top * smoothstep(0.3, 1.0, w)
            ink = RED if v >= 0 else BLUE
            at = self.P(zero + np.sign(v) * self.frac(length / 2)[0], y)
            self.rect(sc, at, max(length, 0.5), 15, ink, op, radius=2)
            best = r == self.order[0] and u > 0.82
            self.text(sc, 0.945, y, "%+.1f" % v, op * smoothstep(0.5, 1.0, w), anchor="rm",
                      strong=best, ink=None if best else dark(ink, 0.3), size=0.030)
        self.polyline(sc, [self.P(zero, cell(0, 0)[1] - 0.03), self.P(zero, cell(9, 0)[1] + 0.025)],
                      1.0, AXIS, 0.7 * op)
        if u > 0.82:
            self.row_ring(sc, cell, int(self.order[0]), 16, 230, 230, 10, RED,
                          op * smoothstep(0.82, 0.9, u))
        self.text(sc, 0.82, 0.91, "highest score: %d" % self.order[0],
                  op * smoothstep(0.82, 0.9, u), strong=True, size=0.032)

    def phase_2(self, sc, u, op):
        d = self.d
        wide = len(self.widths) - 1
        after = smoothstep(0.42, 0.56, u)
        notes = (("feature × row", "one label gets 1"),
                 ("feature × row", "the label 0.91, others 0.01"),
                 ("cosine to the row", "table trained"),
                 ("cosine to the row", "table never updated"))
        start = d["start"][wide, 0]
        for v in range(4):
            x = 0.155 + 0.23 * v
            ink = CLF_INK[v]
            self.text(sc, x, 0.315, CLF_NAMES[v], op, ink=dark(ink, 0.3), size=0.032)
            self.rect(sc, self.P(x, 0.555), 168, 168, ink, op, border=ink, border_w=2.0, radius=5,
                      lift=0.05, fill_op=0.0)
            self.grid(sc, start, x, 0.555, 150, 150, op * (1 - after), frame=False)
            self.grid(sc, d["table"][v, wide, 0], x, 0.555, 150, 150, op * after, frame=False)
            self.text(sc, x, 0.79, notes[v][0], op, strong=True, size=0.031)
            self.text(sc, x, 0.84, notes[v][1], op)
        total = S.CLF_EPOCHS * int(np.ceil(S.N_TRAIN / S.CLF_BATCH))
        self.text(sc, 0.5, 0.905, "the four tables before training: identical" if after < 0.5
                  else "the same four after %s updates" % f"{total:,}", op, strong=True,
                  size=0.032)

    def phase_3(self, sc, u, op):
        acc = self.acc
        ws = np.asarray(self.widths, float)
        chart = self.chart(((0.10, 0.56), (0.36, 0.80)), ws, [0.55, 1.0], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers per label vector", "held-out accuracy",
                  xticks=[(w, str(int(w))) for w in ws],
                  yticks=[(v, pct(v, 0)) for v in (0.6, 0.7, 0.8, 0.9, 1.0)])
        for v in (0, 1, 3, 2):
            self.trace(sc, chart, ws, acc[v].mean(1), CLF_INK[v], op, (u - 0.05 - 0.1 * v) / 0.3,
                       acc[v].min(1), acc[v].max(1), xlog=True)
        self.text(sc, 0.80, 0.335, "16 numbers per label", op, strong=True, size=BIG)
        full = self.span(0.13)[0]
        wide = len(self.widths) - 1
        for v in range(4):
            y = 0.44 + 0.125 * v
            a = acc[v, wide]
            show = op * smoothstep(0.3 + 0.1 * v, 0.4 + 0.1 * v, u)
            self.text(sc, 0.655, y - 0.04, CLF_NAMES[v], show, anchor="lm",
                      ink=dark(CLF_INK[v], 0.3))
            self.hbar(sc, 0.655, y, full * (a.mean() - 0.9) / 0.1, CLF_INK[v], show, "",
                      "%s  (%s to %s)" % (pct(a.mean()), "%.1f" % (100 * a.min()),
                                          "%.1f" % (100 * a.max())),
                      (u - 0.3 - 0.1 * v) / 0.2, thick=16.0)
        self.text(sc, 0.655, 0.905, "bars start at 90%; brackets: the 5 seeds", op, anchor="lm")

    def phase_4(self, sc, u, op):
        m = self.meta
        wide = len(self.widths) - 1
        off = 1 - np.eye(10)
        vmax = float(np.abs(self.cos * off).max())
        cc, fn = m["cosine_vs_confusion"], m["four_nine_cosine"]
        keys = ("one-hot", "smoothed", "learned", "random")
        xs = (0.125, 0.315, 0.505, 0.695)
        order = (0, 2, 3, 1)                          # one-hot, learned, random, then smoothing
        for n, v in enumerate(order):
            x = xs[n]
            show = op * smoothstep(0.02 + 0.13 * n, 0.12 + 0.13 * n, u)
            self.text(sc, x, 0.315, CLF_NAMES[v].replace(" targets", "").replace(" table", ""),
                      show, ink=dark(CLF_INK[v], 0.3), size=BIG)
            self.matrix(sc, self.cos[v] * off, x, 0.535, 150, show, vmax=vmax, ring=[(4, 9)],
                        digits=False)
            self.text(sc, x, 0.735, "agrees %.2f" % cc[keys[v]][wide], show, strong=True,
                      size=0.032)
            self.text(sc, x, 0.785, "4 and 9: %+.2f" % fn[keys[v]][wide], show)
        show = op * smoothstep(0.55, 0.68, u)
        self.text(sc, 0.885, 0.315, "confusion", show, ink=dark(VIOLET, 0.3), size=BIG)
        self.matrix(sc, self.d["soft"] * off, 0.885, 0.535, 150, show, hot=VIOLET, ring=[(4, 9)],
                    digits=False)
        self.text(sc, 0.885, 0.735, "a separate", show)
        self.text(sc, 0.885, 0.785, "classifier's", show)
        self.text(sc, 0.5, 0.865, "each square: digits 0 to 9 against 0 to 9    red: two rows "
                  "point the same way    ringed: 4 and 9", op)
        self.text(sc, 0.5, 0.91, "one seed's tables drawn; the numbers are means of %d seeds"
                  % len(S.CLF_SEEDS), op)

    def phase_5(self, sc, u, op):
        m = self.meta
        vmax = float(self.wrong.max())
        worst = tuple(m["most_confused_pair_widest"]["one-hot"]["pair"])
        for n, v in enumerate((0, 3)):
            x = 0.145 + 0.225 * n
            self.text(sc, x, 0.315, CLF_NAMES[v].replace(" targets", "").replace(" table", ""),
                      op, ink=dark(CLF_INK[v], 0.3), size=BIG)
            self.matrix(sc, self.wrong[v], x, 0.545, 170, op, hot=CLF_INK[v], vmax=vmax,
                        digits=n == 0, ring=[worst])
        top = m["most_confused_pair_widest"]["one-hot"]
        self.text(sc, 0.255, 0.765, "mistakes, 16-number rows, 5 seeds", op, strong=True,
                  size=0.031)
        self.text(sc, 0.255, 0.815, "down: the digit    across: the answer", op)
        self.text(sc, 0.255, 0.865, "ringed: %d and %d, %d of the one-hot %s" % (
            top["pair"][0], top["pair"][1], top["mistakes"], "{:,}".format(top["of"])), op)
        ws = np.asarray(self.widths, float)
        chart = self.chart(((0.57, 0.94), (0.37, 0.70)), ws, [0.0, 1.0], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers per label vector", "same pairs as one-hot",
                  xticks=[(w, str(int(w))) for w in ws],
                  yticks=[(v, "%.1f" % v) for v in (0.0, 0.5, 1.0)])
        for n, v in enumerate((1, 2, 3)):
            self.trace(sc, chart, ws, self.same[v], CLF_INK[v], op, (u - 0.15 - 0.15 * n) / 0.3,
                       xlog=True)
        self.key(sc, 0.58, 0.835, [(CLF_NAMES[v], CLF_INK[v]) for v in (1, 2, 3)], op,
                 step=0.04)


# ====================================================================== 108, 109: shared
AE_INK = (BLUE, RED, VIOLET, OCHRE)           # plain, decoder told, both told, random label
AE_NAMES = ("plain", "decoder told", "both told", "random label")


class AutoencoderLabels(Labels):
    """The pipeline both autoencoder animations draw: image, encoder, code, decoder,
    picture, with a label's row joining the code."""

    def setup(self):
        self.a = S.autoencoder_data()
        self.codes = [int(c) for c in S.AE_CODES]

    def code_of(self, variant, code, digit):
        """The recorded code of the first held-out image of ``digit``."""
        a = self.a
        return a["cloud"][variant, self.codes.index(code), int(a["first"][digit]), :code]

    def pipeline(self, sc, op, u, y, image, code, picture, ink, picture_op=1.0, size=118):
        """image -> encoder -> code -> decoder -> picture along the line ``y``; ``u``
        runs the value through it. Returns nothing: the fixed positions are attributes."""
        self.digit(sc, image, 0.105, y, size, op)
        self.arrow(sc, (0.175, y), (0.215, y), ink, op, smoothstep(0.0, 0.1, u), 2.6)
        self.stage(sc, 0.28, y, "encoder", ink, op, w=112, h=64)
        self.arrow(sc, (0.345, y), (0.395, y), ink, op, smoothstep(0.1, 0.2, u), 2.6)
        self.pulse(sc, (0.175, y), (0.395, y), u / 0.22, ink, op)
        self.numbers(sc, code, 0.44, y, op * smoothstep(0.18, 0.26, u), ink,
                     gap=0.062 if len(code) <= 2 else 0.058)
        self.arrow(sc, (0.485, y), (0.545, y), ink, op, smoothstep(0.26, 0.36, u), 2.6)
        self.stage(sc, 0.61, y, "decoder", ink, op, w=112, h=64)
        self.arrow(sc, (0.675, y), (0.735, y), ink, op, smoothstep(0.36, 0.46, u), 2.6)
        self.pulse(sc, (0.485, y), (0.735, y), (u - 0.26) / 0.22, ink, op)
        self.digit(sc, picture, 0.815, y, size, op * smoothstep(0.44, 0.54, u) * picture_op,
                   frame=ink)


# ====================================================================== 108: the autoencoder
class LabelEmbeddingAutoencoder(AutoencoderLabels):
    """A plain autoencoder squeezing a digit through two numbers; the same one with
    the label's row handed to its decoder; ten digits rebuilt by each; the held-out
    error at four code sizes; the codes themselves, sorted by digit or not; and how
    well the digit can be read from the code alone."""

    TITLE = "Telling the decoder the label"
    PHASES = (("plain", 8.0), ("told", 11.0), ("rebuilt", 9.0), ("error", 9.0),
              ("the codes", 9.0), ("what is left", 9.0))
    DIGIT = 4

    def setup(self):
        super().setup()
        s = self.study
        self.meta = s.metadata
        self.loss = np.asarray(s.panels[0].data["seeds"], float)          # (variant, code, seed)
        self.knn = np.asarray(s.panels[1].data["seeds"], float)
        self.cloud_y = np.asarray(s.panels[2].data["classes"])
        self.clouds = [np.asarray(s.panels[2].data[k], float)
                       for k in ("xy", "decoder", "both", "shuffled")]
        self.rebuilt = np.asarray(s.panels[3].data["values"], float).reshape(5, 10, 784)
        self._tex = {}

    def subtitle_for(self, k):
        m = self.meta
        err, knn = m["held_out_error"], m["digit_from_code_neighbours"]
        read = m["reconstruction_read_right"]
        return [
            "an autoencoder squeezes an image into a short code and rebuilds it from the "
            "code alone. Here the code is 2 numbers: which digit it is and how it was "
            "written both have to fit in them",
            "now look the digit's label up in a table of 8-number rows and hand that row to "
            "the decoder beside the code. The table is trained with the autoencoder; a second "
            "variant tells the encoder too",
            "ten held-out digits rebuilt from 2-number codes. A separate classifier reads %s "
            "of the plain rebuilds as the right digit, %s of the told ones, and %s when the "
            "decoder is handed a random label's row" % (
                pct(read["plain"][0]), pct(read["decoder"][0]), pct(read["shuffled"][0])),
            "held-out error, %d seeds: with 2-number codes %.4f plain and %.4f told, %.0f%% "
            "less. The gain shrinks as the code grows: %.4f against %.4f with 16 numbers. "
            "Telling the encoder too changes little" % (
                len(S.AE_SEEDS), err["plain"][0], err["decoder"][0],
                100 * (1 - err["decoder"][0] / err["plain"][0]), err["plain"][-1],
                err["decoder"][-1]),
            "the 2-number codes of %s held-out digits. Plain, each digit keeps its own "
            "region: the code has to say which digit. Told the label, the digits pile on top "
            "of each other: the code is free for the handwriting" % f"{S.AE_CLOUD:,}",
            "can the digit be read from the code alone, by its %d nearest codes? Plain: %s "
            "with 2 numbers. Decoder told: %s. But with 16 numbers, %s and %s: given room, "
            "the code keeps the digit anyway" % (
                S.AE_NEIGHBOURS, pct(knn["plain"][0], 0), pct(knn["decoder"][0], 0),
                pct(knn["plain"][-1], 0), pct(knn["decoder"][-1], 0)),
        ][k]

    def phase_1(self, sc, u, op):
        a, g = self.a, self.DIGIT
        self.pipeline(sc, op, u / 0.8, 0.59, a["sources"][g], self.code_of(0, 2, g),
                      a["pictures"][0, 0, g, g], BLUE, size=140)
        self.text(sc, 0.105, 0.385, "a held-out %d" % g, op, strong=True, size=BIG)
        self.text(sc, 0.44, 0.385, "its code", op * smoothstep(0.15, 0.22, u), strong=True,
                  size=BIG)
        self.text(sc, 0.44, 0.74, "2 numbers", op * smoothstep(0.15, 0.22, u))
        self.text(sc, 0.815, 0.385, "rebuilt", op * smoothstep(0.4, 0.5, u), strong=True, size=BIG)
        self.text(sc, 0.5, 0.86, "everything the decoder knows about the image is in those "
                  "two numbers", op * smoothstep(0.55, 0.7, u), size=0.032)

    def phase_2(self, sc, u, op):
        a, g = self.a, self.DIGIT
        both = u > 0.66
        v = 2 if both else 1
        ink = AE_INK[v]
        y = 0.725
        self.pipeline(sc, op, min(u / 0.3, 1.0) if not both else 1.0, y, a["sources"][g],
                      self.code_of(v, 2, g), a["pictures"][v, 0, g, g], ink, size=112)
        table = a["table"][v, 0, 0]
        vmax = float(np.abs(table).max())
        show = op * smoothstep(0.2, 0.3, u)
        self.box(sc, 0.215, 0.40, 50, 50, soft(AMBER), show, str(g), border=AMBER, size=0.05)
        self.text(sc, 0.05, 0.40, "its label", show, strong=True, size=0.032, anchor="lm")
        cell = self.grid(sc, table, 0.44, 0.41, 150, 132, show, vmax=vmax)
        self.text(sc, 0.05, 0.50, "looked up in the label table", show, anchor="lm")
        ry = cell(g, 0)[1]
        self.arrow(sc, (0.248, 0.40), (0.352, ry), AMBER, show, smoothstep(0.28, 0.38, u), 2.6)
        self.row_ring(sc, cell, g, 8, 150, 132, 10, AMBER, show * smoothstep(0.36, 0.42, u))
        self.arc(sc, (0.525, ry), (0.61, 0.655), AMBER, show, smoothstep(0.42, 0.54, u), 0.05, 2.6)
        self.pulse(sc, (0.525, ry), (0.61, 0.655), (u - 0.42) / 0.16, AMBER, op, 0.05)
        row = op * smoothstep(0.46, 0.54, u)
        self.grid(sc, table[g][None], 0.75, 0.41, 170, 22, row, vmax=vmax)
        self.text(sc, 0.75, 0.355, "row %d: 8 numbers" % g, row, strong=True, size=0.032)
        self.text(sc, 0.75, 0.47, "goes in beside the code", row)
        if both:
            w = (u - 0.66) / 0.34
            self.arc(sc, (0.36, 0.48), (0.28, 0.655), VIOLET, op, smoothstep(0.0, 0.3, w), -0.03,
                     2.6)
            self.pulse(sc, (0.36, 0.48), (0.28, 0.655), w / 0.4, VIOLET, op, -0.03)
        self.text(sc, 0.5, 0.905, "second variant: the encoder is told the label as well" if both
                  else "the decoder is told which digit it is drawing",
                  op * smoothstep(0.5, 0.6, u),
                  ink=dark(ink, 0.3), size=0.032)

    def phase_3(self, sc, u, op):
        names = ("held-out digit",) + AE_NAMES
        inks = (None,) + AE_INK
        self.text(sc, 0.05, 0.305, "rebuilt from 2 numbers", op, anchor="lm", strong=True,
                  size=0.032)
        for r in range(5):
            y = 0.385 + 0.122 * r
            show = op * smoothstep(0.04 + 0.14 * r, 0.14 + 0.14 * r, u)
            self.text(sc, 0.05, y, names[r], show, anchor="lm", size=0.032,
                      ink=None if inks[r] is None else dark(inks[r], 0.3), strong=inks[r] is None)
            for c in range(10):
                self.digit(sc, self.rebuilt[r, c], 0.285 + 0.07 * c, y, 46, show, frame=inks[r])

    def _chart(self, sc, u, op, values, ylabel, yrange, yticks, order=(0, 3, 1, 2)):
        cs = np.asarray(self.codes, float)
        chart = self.chart(((0.10, 0.57), (0.37, 0.80)), cs, yrange, pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers in the code", ylabel,
                  xticks=[(c, str(int(c))) for c in cs], yticks=yticks)
        for n, v in enumerate(order):
            self.trace(sc, chart, cs, values[v].mean(1), AE_INK[v], op, (u - 0.04 - 0.12 * n) / 0.3,
                       values[v].min(1), values[v].max(1), xlog=True)
        self.key(sc, 0.66, 0.40, [(AE_NAMES[v], AE_INK[v]) for v in order], op, step=0.05)

    def phase_4(self, sc, u, op):
        self._chart(sc, u, op, self.loss, "held-out error", [0.012, 0.047],
                    [(v, "%.2f" % v) for v in (0.02, 0.03, 0.04)])
        m = self.meta["held_out_error"]
        show = op * smoothstep(0.55, 0.7, u)
        self.text(sc, 0.645, 0.65, "2 numbers: %.4f to %.4f" % (m["plain"][0], m["decoder"][0]),
                  show, anchor="lm", strong=True, size=0.032)
        self.text(sc, 0.645, 0.70, "%.0f%% less error" % (
            100 * (1 - m["decoder"][0] / m["plain"][0])), show, anchor="lm",
            ink=dark(RED, 0.3))
        self.text(sc, 0.645, 0.78, "16 numbers: %.4f to %.4f" % (
            m["plain"][-1], m["decoder"][-1]), show, anchor="lm", strong=True, size=0.032)
        self.text(sc, 0.645, 0.83, "%.0f%% less error" % (
            100 * (1 - m["decoder"][-1] / m["plain"][-1])), show, anchor="lm",
            ink=dark(RED, 0.3))
        self.text(sc, 0.645, 0.90, "whiskers: the %d seeds" % len(S.AE_SEEDS), op, anchor="lm")

    def _cloud_tex(self, v, upto):
        key = (v, upto)
        if key not in self._tex:
            self._tex[key] = cloud_texture(self.clouds[v], self.cloud_y, upto, side=112)
        return self._tex[key]

    def phase_5(self, sc, u, op):
        knn = self.meta["digit_from_code_neighbours"]
        upto = int(np.clip(u / 0.6 * 10 + 1, 1, 10))
        for x, v, key in ((0.22, 0, "plain"), (0.60, 1, "decoder")):
            self.cloud(sc, self._cloud_tex(v, upto), x, 0.60, 236, op, AE_NAMES[v], AE_INK[v],
                       "neighbours name the digit: %s" % pct(knn[key][0], 0))
        self.text(sc, 0.875, 0.345, "digit", op, strong=True, size=0.032)
        for c in range(10):
            y = 0.40 + 0.047 * c
            on = op * (1.0 if c < upto else 0.25)
            self.box(sc, 0.86, y, 15, 15, DIGIT10[c], on, radius=3)
            self.text(sc, 0.89, y, str(c), on, size=0.031)

    def phase_6(self, sc, u, op):
        self._chart(sc, u, op, self.knn, "digit read from the code", [0.15, 1.0],
                    [(v, pct(v, 0)) for v in (0.25, 0.5, 0.75, 1.0)])
        m = self.meta["digit_from_code_neighbours"]
        show = op * smoothstep(0.55, 0.7, u)
        rows = [("2 numbers: %s plain, %s told" % (pct(m["plain"][0], 0),
                                                   pct(m["decoder"][0], 0)), None),
                ("16 numbers: %s plain, %s told" % (pct(m["plain"][-1], 0),
                                                    pct(m["decoder"][-1], 0)), None)]
        self.text(sc, 0.645, 0.66, rows[0][0], show, anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.645, 0.71, rows[1][0], show, anchor="lm", strong=True, size=0.031)
        self.lines(sc, 0.645, 0.79, ["nothing forces the digit out", "of the code: it leaves only",
                                     "when the room is needed"], show, step=0.045)


# ====================================================================== 109: generation
STYLE_INK = (AMBER, TEAL, MAGENTA)            # ink, slant, stroke thickness
STYLE_NAMES = ("ink", "slant", "stroke thickness")


class LabelEmbeddingGeneration(AutoencoderLabels):
    """Keep an image's code and change the label row its decoder is handed: the same
    handwriting drawn as another digit; every digit as every digit; how often a
    separate classifier reads the asked digit, by code size; how much of the
    handwriting survives; and a row slid between two labels."""

    TITLE = "Change the label, keep the style"
    PHASES = (("swap the row", 11.0), ("every digit", 10.0), ("how often", 10.0),
              ("the style", 10.0), ("in between", 11.0))
    SOURCE = 3                                 # the held-out digit whose row is swapped
    CODE = 4                                   # the code size drawn

    def setup(self):
        super().setup()
        s = self.study
        self.meta = s.metadata
        self.ci = self.codes.index(self.CODE)
        self.grid_pictures = np.asarray(s.panels[0].data["values"], float).reshape(10, 10, 784)
        self.read = np.asarray(s.panels[0].data["read"])[1]               # decoder: (code, s, a)
        self.asked = np.asarray(s.panels[1].data["seeds"], float)         # (3 variants, code, seed)
        self.plain = np.asarray(s.panels[1].data["plain"], float)
        self.style = np.asarray(s.panels[2].data["seeds"], float)         # (code, seed, 3)
        self.mix = np.asarray(s.panels[3].data["ys"], float)              # (3, alpha)
        self.alphas = np.asarray(s.panels[3].data["x"], float)
        self.strips = np.asarray(s.panels[3].data["strips"], float)[self.ci]   # (pair, alpha, 784)
        self.strip_read = np.asarray(s.panels[3].data["strip_read"], float)[self.ci]
        off = ~np.eye(10, dtype=bool)
        self.grid_right = int((self.read[self.ci] == np.arange(10)[None])[off].sum())
        self.sequence = [self.SOURCE] + [c for c in range(10) if c != self.SOURCE]
        measures = self.a["source_measures"]
        # the three sources of the style phase: the thinnest, the thickest, the most slanted
        self.examples = [int(np.argmin(measures[0])), int(np.argmax(measures[0])),
                         int(np.argmax(np.abs(measures[1])))]

    def subtitle_for(self, k):
        m = self.meta
        asked, style = m["read_as_asked"], m["style_swap"]["decoder"]
        mix = m["in_between"]["decoder_code_4"]
        _ci, half = self.ci, len(self.alphas) // 2
        return [
            "an autoencoder whose decoder is told the label can be told a different one. "
            "Keep this %d's code, hand the decoder another digit's row, and it draws that "
            "digit in the same hand; a separate classifier reads the result" % self.SOURCE,
            "ten held-out digits, each drawn as all ten from a %d-number code. Slant and "
            "weight run along each row. The classifier reads %d of these 90 swaps as the "
            "digit asked for" % (self.CODE, self.grid_right),
            "over %s swaps and %d seeds: read as asked %s with 2-number codes, %s with 4, %s "
            "with 8, %s with 16. A big code holds the digit itself, so the label is ignored; "
            "plain or random-label, never above %s" % (
                f"{9 * S.AE_SWAP:,}", len(S.AE_SEEDS), pct(asked["decoder"][0], 0),
                pct(asked["decoder"][1], 0), pct(asked["decoder"][2], 0),
                pct(asked["decoder"][3], 0),
                pct(max(max(asked["plain"]), max(asked["shuffled"])), 0)),
            "how much handwriting survives: source against swap, slant agrees %.2f with "
            "2-number codes and %.2f with 8; an unrelated digit would score 0. Bigger codes "
            "keep more style and obey the label less" % (style["slant"][0], style["slant"][2]),
            "slide the row from one label to another. Halfway, %s of pictures still read as "
            "the source, %s as the digit asked and %s as neither: the digit turns over in a "
            "few steps, with some blends between" % (
                pct(mix[0][half], 0), pct(mix[1][half], 0), pct(mix[2][half], 0)),
        ][k]

    def phase_1(self, sc, u, op):
        a, g = self.a, self.SOURCE
        y = 0.70
        turn = int(min(u / 0.92 * len(self.sequence), len(self.sequence) - 1))
        lab = self.sequence[turn]
        w = u / 0.92 * len(self.sequence) - turn
        self.pipeline(sc, op, 1.0 if turn else u * len(self.sequence) / 0.92, y, a["sources"][g],
                      self.code_of(1, self.CODE, g), self.grid_pictures[g, lab], RED, size=112)
        self.text(sc, 0.105, 0.845, "a held-out %d" % g, op, strong=True, size=0.032)
        self.text(sc, 0.44, 0.895, "its code: kept", op, strong=True, size=0.032)
        xs = self.tokens(sc, [str(c) for c in range(10)], 0.33, 0.89, 0.345, op,
                         inks=[AMBER], active={lab}, w=42, h=38, size=0.04)
        self.text(sc, 0.19, 0.345, "the label row", op, strong=True, size=0.032)
        table = a["table"][1, self.ci, 0]
        self.grid(sc, table[lab][None], 0.61, 0.50, 150, 20, op, vmax=float(np.abs(table).max()))
        self.arrow(sc, (xs[lab], 0.385), (0.61, 0.475), AMBER, op, smoothstep(0.0, 0.3, w), 2.4)
        self.arrow(sc, (0.61, 0.525), (0.61, 0.635), AMBER, op, 1.0, 2.4)
        self.pulse(sc, (0.61, 0.525), (0.61, 0.635), w / 0.6, AMBER, op)
        said = int(self.read[self.ci, g, lab])
        self.text(sc, 0.815, 0.845, "asked for %d" % lab, op, ink=dark(AMBER, 0.3), size=0.032)
        self.text(sc, 0.815, 0.895, "the classifier reads %d" % said, op, strong=True,
                  size=0.032)

    def phase_2(self, sc, u, op):
        a = self.a
        x0, y0, dx, dy = 0.385, 0.35, 0.0365, 0.0635
        self.text(sc, 0.05, 0.42, "down: ten held-out", op, anchor="lm", size=0.032)
        self.text(sc, 0.05, 0.47, "digits, their codes kept", op, anchor="lm", size=0.032)
        self.text(sc, 0.05, 0.57, "across: the label row", op, anchor="lm", size=0.032,
                  ink=dark(AMBER, 0.3))
        self.text(sc, 0.05, 0.62, "the decoder is handed", op, anchor="lm", size=0.032,
                  ink=dark(AMBER, 0.3))
        for c in range(10):
            self.text(sc, x0 + dx * c, 0.295, str(c), op, ink=dark(AMBER, 0.3), size=0.032)
        rows = u / 0.7 * 10
        right = 0
        for r in range(10):
            y = y0 + dy * r
            show = op * smoothstep(r, r + 0.8, rows)
            self.digit(sc, a["sources"][r], 0.325, y, 28, op)
            for c in range(10):
                ok = self.read[self.ci, r, c] == c
                right += int(ok and r != c and rows >= r + 0.8)
                frame = AMBER if r == c else (None if ok else RED)
                self.small(sc, self.grid_pictures[r, c], x0 + dx * c, y, 28, show, frame)
        self.text(sc, 0.76, 0.42, "amber frame: its own label", op, anchor="lm",
                  ink=dark(AMBER, 0.3))
        self.text(sc, 0.76, 0.47, "red frame: the classifier", op, anchor="lm", ink=dark(RED, 0.3))
        self.text(sc, 0.76, 0.515, "reads another digit", op, anchor="lm", ink=dark(RED, 0.3))
        self.text(sc, 0.76, 0.62, "read as asked so far", op, anchor="lm")
        self.text(sc, 0.76, 0.675, "%d of %d swaps" % (right, 9 * int(min(rows + 0.2, 10))), op,
                  anchor="lm", strong=True, size=BIG)

    def phase_3(self, sc, u, op):
        cs = np.asarray(self.codes, float)
        chart = self.chart(((0.10, 0.55), (0.37, 0.80)), cs, [0.0, 1.0], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers in the code", "read as the digit asked for",
                  xticks=[(c, str(int(c))) for c in cs],
                  yticks=[(v, pct(v, 0)) for v in (0.0, 0.5, 1.0)])
        for n, v in enumerate((0, 1, 2)):                # decoder, both, shuffled
            vals = self.asked[v]
            self.trace(sc, chart, cs, vals.mean(1), AE_INK[v + 1], op, (u - 0.04 - 0.1 * n) / 0.3,
                       vals.min(1), vals.max(1), xlog=True)
        self.trace(sc, chart, cs, self.plain, AE_INK[0], op, (u - 0.34) / 0.3, xlog=True)
        self.key(sc, 0.11, 0.905, [(AE_NAMES[1], RED)], op)
        self.key(sc, 0.27, 0.905, [(AE_NAMES[2], VIOLET)], op)
        self.key(sc, 0.41, 0.905, [("plain, random label", OCHRE)], op)
        g = self.SOURCE
        self.text(sc, 0.79, 0.315, "the %d as 0 to 9, by code size" % g, op, strong=True,
                  size=0.032)
        for r, code in enumerate(self.codes):
            y = 0.42 + 0.115 * r
            show = op * smoothstep(0.3 + 0.12 * r, 0.42 + 0.12 * r, u)
            self.text(sc, 0.615, y, str(code), show, strong=True, size=0.034)
            for c in range(10):
                ok = self.read[r, g, c] == c
                self.small(sc, self.a["pictures"][1, r, g, c], 0.655 + 0.0325 * c, y, 26, show,
                           AMBER if c == g else (None if ok else RED))
        self.text(sc, 0.80, 0.875, "red: not read as asked", op, ink=dark(RED, 0.3))

    def phase_4(self, sc, u, op):
        a = self.a
        self.text(sc, 0.24, 0.315, "thinnest, thickest, most slanted", op, strong=True, size=0.032)
        for n, g in enumerate(self.examples):
            y = 0.44 + 0.16 * n
            self.digit(sc, a["sources"][g], 0.085, y, 56, op)
            for c in range(10):
                self.small(sc, self.grid_pictures[g, c], 0.16 + 0.0325 * c, y, 26,
                           op * smoothstep(0.03 * c, 0.03 * c + 0.1, u),
                           AMBER if c == g else None)
        self.text(sc, 0.24, 0.895, "three of the ten sources, each as 0 to 9", op)
        cs = np.asarray(self.codes, float)
        chart = self.chart(((0.55, 0.94), (0.37, 0.70)), cs, [0.0, 1.0], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "numbers in the code", "source and swap agree",
                  xticks=[(c, str(int(c))) for c in cs],
                  yticks=[(v, "%.1f" % v) for v in (0.0, 0.5, 1.0)])
        for k in range(3):
            vals = self.style[:, :, k]
            self.trace(sc, chart, cs, vals.mean(1), STYLE_INK[k], op, (u - 0.2 - 0.1 * k) / 0.3,
                       vals.min(1), vals.max(1), xlog=True)
        vals = self.asked[0]
        self.trace(sc, chart, cs, vals.mean(1), RED, op, (u - 0.55) / 0.3, vals.min(1),
                   vals.max(1), width=2.0, xlog=True)
        self.key(sc, 0.56, 0.845, [(STYLE_NAMES[0], AMBER), (STYLE_NAMES[1], TEAL)], op,
                 step=0.045)
        self.key(sc, 0.74, 0.845, [(STYLE_NAMES[2], MAGENTA), ("read as asked", RED)], op,
                 step=0.045)

    def phase_5(self, sc, u, op):
        n = len(self.alphas)
        at = min(u / 0.8, 1.0) * (n - 1)
        now = int(round(at))
        for r, (lo, hi) in enumerate(S.AE_STRIPS[:2]):
            y = 0.375 + 0.135 * r
            self.text(sc, 0.085, y, "%d to %d" % (lo, hi), op, strong=True, size=0.034)
            for i in range(n):
                x = 0.175 + 0.0705 * i
                on = i <= now
                self.digit(sc, self.strips[r, i], x, y, 56, op * (1.0 if on else 0.25),
                           frame=AMBER if i == now else None)
        self.text(sc, 0.175, 0.59, "its own row", op, anchor="lm", ink=dark(AMBER, 0.3))
        self.text(sc, 0.88 + 0.03, 0.59, "the other label's row", op, anchor="rm",
                  ink=dark(AMBER, 0.3))
        chart = self.chart(((0.29, 0.75), (0.665, 0.855)), self.alphas, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "how far the row has slid", "read as",
                  xticks=[(0.5, "halfway"), (1.0, "all the way")],
                  yticks=[(v, pct(v, 0)) for v in (0.0, 0.5, 1.0)])
        inks = (BLUE, RED, GREY)
        for k in range(3):
            self.trace(sc, chart, self.alphas, self.mix[k], inks[k], op, at / (n - 1))
        self.key(sc, 0.80, 0.70, [("the source digit", BLUE), ("the digit asked", RED),
                                  ("neither", GREY)], op, step=0.05)
        self.text(sc, 0.05, 0.76, "over all swaps", op, anchor="lm")
        self.text(sc, 0.05, 0.805, "%d-number codes" % self.CODE, op, anchor="lm")


ANIMATIONS = {
    "label-embedding": LabelEmbedding,
    "label-embedding-classifier": LabelEmbeddingClassifier,
    "label-embedding-autoencoder": LabelEmbeddingAutoencoder,
    "label-embedding-generation": LabelEmbeddingGeneration,
}
