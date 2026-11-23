"""Animation 159 of the "next" catalogue: training on a CPU and on a GPU (see
:mod:`voxanim.mech`).

:class:`CpuAndGpuTraining` keeps one map of a computer on screen for the whole
animation - storage, main memory, a CPU of a few cores, a bus, and a GPU with its own
memory and many small cores - and moves three things over it: the model's weights (a
violet grid, drawn on a compressed scale and labelled with how many numbers it holds),
batches of real MNIST digits (tiles), and an amber light on the cores doing arithmetic.
Beside the map a panel carries a stopwatch, a count of steps, and the figures.

It is deliberately long (about three minutes and a quarter, eleven phases): its subject
is waiting. Phase 5 is about precision: how many bits a number gets (a CPU's 64 against
the 32, 16, 8 and 4 a graphics card is built for), what each width costs in speed, and
what storing the trained classifier's weights in fewer bits does to it.

The speeds are representative round figures, not measurements: one table,
:data:`voxanim.study_hardware.REPRESENTATIVE`, and arithmetic from it. Every curve, and
the accuracy of the classifier stored in fewer bits, is numpy on MNIST; the formats are
exact arithmetic. Each phase says, bottom right, which of these it shows.
Nothing is computed here but layout.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import study_hardware as S
from .diagram import draw_image
from .easing import ease_in_out, remap, smoothstep
from .losschart import pow10
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    TEXT,
    TRACK,
    VIOLET,
    Mechanism,
    dark,
    heat,
    soft,
)

GREY = (0.56, 0.58, 0.60)
INK = {"cpu": BLUE, "gpu": GREEN}
BIG = 0.034

# ---------------------------------------------------------------------- the map (layout)
RAM = (0.015, 0.285, 0.29, 0.56)              # x0, x1, top, bottom
STO = (0.015, 0.13, 0.715, 0.935)
CPU = (0.165, 0.285, 0.715, 0.935)
GPU = (0.33, 0.50, 0.29, 0.935)
GMEM = (0.339, 0.491, 0.35, 0.60)
BUS = (0.05, 0.33, 0.622, 0.658)
BUS_Y = 0.64
RAM_TAP, STO_TAP, CPU_TAP, GPU_TAP = 0.1365, 0.0725, 0.225, 0.352
W_RAM, D_RAM = (0.07, 0.435), (0.215, 0.44)  # weights and the digits, in main memory
W_GPU, B_GPU = (0.385, 0.49), (0.461, 0.49)   # weights and a batch, in GPU memory
W_CPU = (0.262, 0.75)                         # the weights the CPU is working on
W_STO = (0.0725, 0.9)                         # the weights saved to a file
F_STO = (0.0725, 0.815)                       # the file of digits
C_CPU, C_GPU = (0.225, 0.845), (0.415, 0.79)  # the cores
CAPTION_Y = 0.54                              # of what main memory holds
CELL = 4.0                                    # px side of one cell of a weights block
TILE = 22.0
PX0, PX1 = 0.535, 0.985                       # the panel beside the map
TAG_Y = 0.972
ASPECT = np.array([960.0, 540.0])

VIA_STO_RAM = [(STO_TAP, STO[2]), (STO_TAP, BUS_Y), (RAM_TAP, BUS_Y), (RAM_TAP, RAM[3])]
VIA_RAM_CPU = [(RAM_TAP, RAM[3]), (RAM_TAP, BUS_Y), (CPU_TAP, BUS_Y), (CPU_TAP, CPU[2])]
VIA_RAM_GPU = [(RAM_TAP, RAM[3]), (RAM_TAP, BUS_Y), (GPU_TAP, BUS_Y)]
VIA_GPU_CPU = [(GPU_TAP, BUS_Y), (CPU_TAP, BUS_Y), (CPU_TAP, CPU[2])]


def seg(u, a, b):
    """0..1 as ``u`` goes from ``a`` to ``b``."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


def along(points, s):
    """The layout point a fraction ``s`` of the way along the path through ``points``
    (by distance on screen)."""
    p = np.asarray(points, float)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*((p[1:] - p[:-1]) * ASPECT).T))])
    at = float(np.clip(s, 0.0, 1.0)) * d[-1]
    return float(np.interp(at, d, p[:, 0])), float(np.interp(at, d, p[:, 1]))


def ms(v):
    """Milliseconds as a label: ``0.05 ms``, ``1 ms``, ``40 ms``."""
    return "%.3g ms" % float(v)


def times(r):
    """A ratio as a label: ``2×``, ``2.6×``, ``53×``."""
    return ("%.2g×" if r < 99.5 else "%.0f×") % r


def count(v):
    """A large count as a label: ``101,770``, ``5.4 million``, ``1.9 billion``."""
    v = float(v)
    if v >= 1e9:
        return "%.1f billion" % (v / 1e9)
    if v >= 1e6:
        return ("%.1f million" if v < 1e7 else "%.0f million") % (v / 1e6)
    return f"{int(v):,}"


def short(v):
    """A count as a caption: ``101,770``, ``5.4M``."""
    return "%.1fM" % (v / 1e6) if v >= 1e6 else f"{int(v):,}"


class CpuAndGpuTraining(Mechanism):
    """The parts of a computer; a training step on the CPU, slowly and then at speed;
    the same with a GPU; why many small cores suit this arithmetic and what they do not
    speed up; a small classifier and a larger autoencoder, timed on both; the four
    times on one scale; a race on a scaled clock; and what the times multiply up to."""

    TITLE = "Training on a CPU and on a GPU"
    PHASES = (("parts", 14.0), ("CPU", 22.0), ("GPU", 22.0), ("cores", 18.0),
              ("precision", 24.0), ("small", 16.0), ("large", 16.0), ("bars", 14.0),
              ("race", 22.0), ("scale", 16.0), ("plan", 12.0))
    RACE = (0.06, 0.90)                       # the part of phase 8 in which the clock runs
    PICTURE = "a picture of the parts: no figures yet"
    FIGURES = "representative figures, not measured"
    ARITHMETIC = "arithmetic from representative figures"
    RACED = "clock: representative figures, not measured; curves: measured on MNIST"
    BITS = "speeds: representative figures, not measured; accuracy: measured on MNIST"

    # ------------------------------------------------------------------ data
    def setup(self):
        s, c = self.study, S.curves()
        m = self.meta = s.metadata
        R = S.REPRESENTATIVE
        self.batch = int(m["batch"])
        self.params = {n: int(m["models"][n]["parameters"]) for n in S.NAMES}
        self.work = {n: int(m["models"][n]["multiply_adds_per_step"]) for n in S.NAMES}
        self.step = {k: float(v) for k, v in zip(S.FOUR, s.panels[0].data["values"])}
        self.epoch = {k: float(v) for k, v in zip(S.FOUR, s.panels[0].data["epoch_seconds"])}
        self.part = {k + (p,): v for k in S.FOUR
                     for p, v in zip(("prepare", "move", "compute"), S.parts_ms(*k))}
        self.fixed = {k: float(R["fixed_ms"][k]) for k in S.FOUR}
        self.batches = np.asarray(s.panels[1].data["x"], float)
        self.by_batch = dict(zip(S.FOUR, np.asarray(s.panels[1].data["ys"], float)))
        self.ratio = {n: float(m["cpu_over_gpu"][n]) for n in S.NAMES}
        self.ahead_from = S.crossover("classifier")
        self.copy = {n: (float(R["copy_ms"][n]), float(R["megabytes"][n])) for n in S.NAMES}
        self.steps_epoch = S.N_TRAIN // self.batch
        self.first_loss = float(c["classifier_first"][0, 0])
        self.acc_steps = np.asarray(s.panels[2].data["x"], float)
        self.acc = np.asarray(s.panels[2].data["ys"], float)
        self.ae_steps = np.asarray(s.panels[3].data["x"], float)
        self.ae = np.asarray(s.panels[3].data["ys"], float)
        self.total_steps = int(self.acc_steps[-1])
        self.race_s = {k: m["epochs"] * self.epoch[k] for k in S.FOUR}
        window = (self.RACE[1] - self.RACE[0]) * self.seconds("race")
        self.race_scale = float(max(1, int(np.ceil(max(self.race_s.values()) / window))))
        self.race_span = window * self.race_scale
        self.projections = [(p["what"], p["seconds"]) for p in m["projections"]]
        # precision: the formats (exact), the speeds at each width (representative), the
        # classifier with its weights in each format (measured)
        self.formats = [(f[0], f[1], f[2], f[3], m["formats"][f[0]]) for f in S.FORMATS]
        self.madds = {k: float(v) for k, v in R["madds_per_s"].items()}
        self.fewer = m["weights_in_fewer_bits"]
        self.example = m["example"]
        # pictures
        self.digits = np.asarray(c["batch_digits"], float)
        self.train_digits = np.asarray(c["train_digits"], float)
        self.held = np.asarray(c["held_digits"], float)
        self.rebuilt = np.asarray(c["autoencoder_shown"], float)
        self.named = [int(v) for v in np.asarray(c["classifier_shown"]).argmax(1)]
        self.hidden = np.asarray(c["hidden"], float)
        self.patch = {n: np.asarray(c[n + "_patch"], float) for n in S.NAMES}
        # the weights block: one cell a side per fourth root of the count over 3 (a
        # compressed scale: 6 x 6 cells for the classifier, 16 x 16 for the autoencoder)
        self.cells = {n: int(round(self.params[n] ** 0.25 / 3.0)) for n in S.NAMES}
        self.side = {n: CELL * self.cells[n] for n in S.NAMES}
        self.data_slots = self.slots(D_RAM, 4, 3, 23.0)
        self.picks = [self.data_slots[i] for i in (0, 3, 5, 6, 9, 10)]   # a batch lifts off
        self.cpu_slots = self.slots(C_CPU, 3, 2, 33.0)
        self.gpu_slots = self.slots(B_GPU, 2, 3, 25.0)
        self.file_slots = self.slots(F_STO, 2, 2, 21.0)

    def subtitle_for(self, k):
        st, ep = self.step, self.epoch
        c, a = "classifier", "autoencoder"

        def versus(name):
            return self.versus(self.ratio[name])
        gpu_fixed = self.fixed[c, "gpu"] / st[c, "gpu"]
        return [
            "training a model moves two things around a computer: the model's weights, and "
            "batches of data. Here are the parts they move between, and where the "
            "arithmetic can happen",
            "no GPU: the CPU's few cores do every sum, about %s a step, %s for all 60,000 "
            "digits. Times here are typical round figures for a desktop CPU and graphics "
            "card; real machines differ"
            % (ms(st[c, "cpu"]), S.duration(ep[c, "cpu"])),
            "with a GPU the weights cross the bus once and stay, each batch crosses, and "
            "only the loss comes back. A typical step for this small model: about %s, "
            "against %s on the CPU"
            % (ms(st[c, "gpu"]), ms(st[c, "cpu"])),
            "a layer is thousands of independent sums: a few fast cores take them in "
            "turn, a GPU's many small cores all at once. But picking the batch, the trip "
            "and starting the work get no faster",
            "a CPU is built for 64-bit numbers: its 64-bit sums run at full speed. A graphics "
            "card is built for 32-bit, and runs 16-, 8- and 4-bit sums %s faster, 64-bit "
            "%s slower"
            % (self.bits_range(), times(self.madds["gpu", "FP32"] / self.madds["gpu", "FP64"])),
            "the small classifier: about %s a step on the CPU, %s on the GPU, only %s. "
            "The sums are few, so fixed costs are %d%% of its GPU step; %s"
            % (ms(st[c, "cpu"]), ms(st[c, "gpu"]), versus(c), round(100 * gpu_fixed),
               "the GPU is behind at every batch size" if self.ahead_from is None
               else "below batches of %d the CPU wins" % self.ahead_from),
            "the autoencoder: %s the weights, %s the arithmetic. About %s a step on the "
            "CPU and %s on the GPU: the arithmetic is now most of the step, and the GPU "
            "is %s"
            % (times(self.params[a] / self.params[c]), times(self.work[a] / self.work[c]),
               ms(st[a, "cpu"]), ms(st[a, "gpu"]), versus(a)),
            "four figures, one scale. The autoencoder takes %s the classifier's time on the "
            "CPU and %s on the GPU: slower on both. Yet on the GPU it is still %s than on "
            "the CPU; the classifier, only %s"
            % (times(st[a, "cpu"] / st[c, "cpu"]), times(st[a, "gpu"] / st[c, "gpu"]),
               versus(a), versus(c)),
            "a race over %d epochs (%s steps), the clock running %d× fast. The classifier "
            "is done in %s; the autoencoder takes %s on the GPU and %s on the CPU, its "
            "held-out loss falling all the while"
            % (self.meta["epochs"], f"{self.total_steps:,}", self.race_scale,
               S.duration(min(self.race_s[c, "cpu"], self.race_s[c, "gpu"])),
               S.duration(self.race_s[a, "gpu"]), S.duration(self.race_s[a, "cpu"])),
            "real projects are bigger. By the same arithmetic, %d epochs of a model 100× "
            "larger on 1,000× the data: about %s on the GPU, %s on the CPU. A phone app "
            "answers within a second; training does not"
            % (self.meta["projection_epochs"], S.duration(self.projections[3][1]["gpu"]),
               S.duration(self.projections[3][1]["cpu"])),
            "so plan for the wait: time a few steps, multiply up before starting, and let "
            "the long run go unattended. That is what batch jobs and shared GPU clusters "
            "are for",
        ][k]

    def seconds(self, name):
        """The length of the phase called ``name``."""
        return dict(self.PHASES)[name]

    def bits_range(self):
        """How much faster than 32-bit a graphics card's narrow sums are: ``4-16×``."""
        lo = self.madds["gpu", "FP16"] / self.madds["gpu", "FP32"]
        hi = self.madds["gpu", "FP4"] / self.madds["gpu", "FP32"]
        return "%s-%s" % (("%g" % lo), times(hi))

    @staticmethod
    def versus(r):
        """A CPU-over-GPU ratio of times in words: the GPU is ``2.6× faster``."""
        return "%s %s" % (times(max(r, 1 / r)), "faster" if r > 1 else "slower")

    # ------------------------------------------------------------------ the frame
    def step_indicator(self, sc, current, phases):
        """Ten phases do not fit by name beside the title: the one under way is named,
        the others are their numbers."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.014
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        self._indicator_left = x
        for n, (s, w) in enumerate(zip(items, widths)):
            col = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else GREY
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", col, 1.0, "la")
            x += w + gap

    def draw(self, sc, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        self.platform(sc, smoothstep(0.0, self.FADE, t) * back)
        super().draw(sc, t, reset)

    # ------------------------------------------------------------------ drawing: the map
    def slots(self, centre, cols, rows, pitch):
        """Layout points of a ``cols x rows`` grid of things ``pitch`` px apart."""
        dx, dy = self.frac(pitch, pitch)
        return [(centre[0] + (c - (cols - 1) / 2) * dx, centre[1] + (r - (rows - 1) / 2) * dy)
                for r in range(rows) for c in range(cols)]

    def area(self, sc, box, fill, op, border=AXIS, lift=0.0, radius=7.0, width=1.5, fill_op=1.0):
        """A rounded box filling the layout rectangle ``(x0, x1, top, bottom)``."""
        x0, x1, y0, y1 = box
        w, h = self.span(x1 - x0, y1 - y0)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, fill, op, border=border,
                  border_w=width, radius=radius, lift=lift, fill_op=fill_op)

    def ring(self, sc, box, ink, op, grow=0.006):
        """An outline round a part of the map: the part meant, or the part at work."""
        x0, x1, y0, y1 = box
        self.area(sc, (x0 - grow, x1 + grow, y0 - grow * 1.78, y1 + grow * 1.78), ink, op,
                  border=ink, lift=0.9, radius=9.0, width=3.0, fill_op=0.0)

    def platform(self, sc, op):
        """The map that stays: storage, main memory, the CPU, the bus, the GPU."""
        if op <= 0.01:
            return
        t = 0.004                                     # half the bus's thickness, across
        for box in (BUS, (STO_TAP - t, STO_TAP + t, BUS_Y, STO[2]),
                    (RAM_TAP - t, RAM_TAP + t, RAM[3], BUS_Y),
                    (CPU_TAP - t, CPU_TAP + t, BUS_Y, CPU[2])):
            x0, x1, y0, y1 = box
            if box is BUS:
                y0, y1 = BUS_Y - t * 1.78, BUS_Y + t * 1.78
            self.area(sc, (x0, x1, y0, y1), TRACK, op, border=None, lift=-0.05, radius=3.0)
        self.area(sc, RAM, PAPER, op)
        self.area(sc, STO, PAPER, op)
        self.area(sc, CPU, soft(BLUE, 0.78), op, border=BLUE)
        self.area(sc, GPU, soft(GREEN, 0.78), op, border=GREEN)
        self.area(sc, GMEM, PAPER, op, lift=0.02)
        self.text(sc, RAM[0] + 0.01, 0.318, "RAM (main memory)", op, anchor="lm", strong=True)
        self.text(sc, STO[0] + 0.008, 0.745, "storage", op, anchor="lm", strong=True)
        self.text(sc, CPU[0] + 0.009, 0.745, "CPU", op, anchor="lm", ink=dark(BLUE, 0.4),
                  size=BIG)
        self.text(sc, GPU[0] + 0.01, 0.32, "GPU", op, anchor="lm", ink=dark(GREEN, 0.45),
                  size=BIG)
        self.text(sc, GMEM[0] + 0.007, 0.375, "GPU memory", op, anchor="lm")
        self.text(sc, 0.322, 0.605, "bus", op, anchor="rm")
        for p, img in zip(self.file_slots, self.train_digits[8:12]):
            self.tile(sc, img, p[0], p[1], 18.0, op, lift=0.3)
        self.cpu_cores(sc, np.zeros(6), op, base=True)
        self.gpu_cores(sc, 0.0, op, base=True)

    def cpu_cores(self, sc, lit, op, base=False):
        """The CPU's cores; ``lit`` (0..1 each): how hard each is doing arithmetic now."""
        for p, v in zip(self.cpu_slots, lit):
            v = float(np.clip(v, 0, 1))
            if base:
                self.rect(sc, self.P(*p), 27, 27, soft(BLUE, 0.35), op, border=BLUE, radius=4,
                          lift=0.2)
            elif v > 0.02:
                self.rect(sc, self.P(*p), 27, 27, AMBER, op * v, border=dark(AMBER, 0.25),
                          radius=4, lift=0.7, glow=0.5 * v)

    def gpu_cores(self, sc, lit, op, base=False, cols=10, rows=8):
        """The GPU's many small cores, as one grid; ``lit``: 0 idle .. 1 all at work."""
        lit = float(np.clip(lit, 0, 1))
        if not base and lit <= 0.02:
            return
        ink = soft(GREEN, 0.30) if base else AMBER
        tex = np.broadcast_to(np.asarray(ink, np.float32), (rows, cols, 3)).copy()
        draw_image(self, sc, self.P(*C_GPU), tex, cols * 13.0, rows * 13.0,
                   op if base else op * lit, grid=0.55, lift=0.2 if base else 0.7)

    def tile(self, sc, img, fx, fy, size, op, lift=1.2):
        """One MNIST digit as a tile."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, size + 3, size + 3, DIGIT_INK, op, radius=2, lift=lift - 0.05)
        v = np.clip(np.asarray(img, float).reshape(28, 28), 0, 1)[..., None]
        draw_image(self, sc, c, (PAPER * (1 - v) + np.asarray(DIGIT_INK) * v).astype(np.float32),
                   size, size, op, grid=0, lift=lift)

    def block(self, sc, name, at, op, lift=1.3, tint=1.0):
        """The model's weights at layout point ``at``: a grid of real initial weights.
        Its size is on a compressed scale (the fourth root of the count), so that a
        small model's weights can still be seen: the label says how many there are."""
        op = op * tint
        if op <= 0.01:
            return
        side, cells = self.side[name], self.cells[name]
        tex = heat(np.clip(self.patch[name][:cells, :cells] / 0.08, -1, 1), VIOLET, VIOLET)
        tex = tex * 0.55 + np.asarray(VIOLET, np.float32) * 0.45
        c = self.P(float(at[0]), float(at[1]))
        self.rect(sc, c, side + 4, side + 4, dark(VIOLET, 0.35), op, radius=2, lift=lift - 0.05)
        draw_image(self, sc, c, tex.astype(np.float32), side, side, op, grid=0.45, lift=lift)

    def memory(self, sc, op, name="classifier", shown=1.0):
        """What main memory holds: the training digits (twelve of the 60,000 drawn), and
        how many weights the model named has."""
        n = len(self.data_slots)
        for i, (p, img) in enumerate(zip(self.data_slots, self.train_digits)):
            if i < int(round(shown * n)):
                self.tile(sc, img, p[0], p[1], 19.0, op, lift=0.4)
        self.text(sc, RAM[1] - 0.008, CAPTION_Y, "60,000 digits", op * float(shown >= 1.0),
                  anchor="rm")
        self.text(sc, RAM[0] + 0.008, CAPTION_Y, "%s weights" % short(self.params[name]), op,
                  anchor="lm")

    def trip(self, sc, imgs, srcs, via, dsts, s, op, size=TILE, stagger=0.4):
        """Digit tiles travelling one behind another from ``srcs`` through ``via`` to
        ``dsts``; ``s``: 0 (all at the start) .. 1 (all arrived)."""
        n = len(imgs)
        for i, img in enumerate(imgs):
            si = seg(s * (1 + stagger) - stagger * i / max(n - 1, 1), 0.0, 1.0)
            p = along([srcs[i % len(srcs)]] + list(via) + [dsts[i % len(dsts)]], ease_in_out(si))
            self.tile(sc, img, p[0], p[1], size, op)

    def carry(self, sc, name, src, via, dst, s, op):
        """The weights block travelling from ``src`` to ``dst``."""
        self.block(sc, name, along([src] + list(via) + [dst], ease_in_out(s)), op, lift=1.5)

    def batch_imgs(self, k=0):
        return [self.digits[(3 * k + i) % len(self.digits)] for i in range(6)]

    # ------------------------------------------------------------------ drawing: the panel
    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, PX1, TAG_Y, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def status(self, sc, op, clock, right="", note=""):
        """The stopwatch and the count of steps, at the top of the panel."""
        c = self.P(PX0 + 0.012, 0.312)
        self.disk(sc, c, 9.0, PAPER, op, ring=AXIS, ring_w=2.2, outline=0.0)
        self.rect(sc, self.shift(c, 0, -11.5), 5, 4, AXIS, op, radius=1, lift=0.6)
        self.rect(sc, self.shift(c, 1.6, -2.6), 2.2, 7, RED, op, radius=1, lift=0.9)
        self.text(sc, PX0 + 0.032, 0.312, clock, op, anchor="lm", strong=True, size=BIG)
        if right:
            self.text(sc, PX1, 0.312, right, op, anchor="rm", strong=True)
        if note:
            self.text(sc, PX0, 0.357, note, op, anchor="lm", ink=layout.NOTE_INK)

    def chip(self, sc, fx, fy, ink, op, text, size=TEXT, strong=False, on=True):
        """A colour chip and what the colour means."""
        self.box(sc, fx + 0.007, fy, 13, 13, ink if on else DIM, op, radius=3)
        self.text(sc, fx + 0.022, fy, text, op * (1.0 if on else 0.55), anchor="lm", size=size,
                  strong=strong)

    def stacked(self, sc, fx, fy, parts, scale, op, u=1.0, thick=16.0):
        """A bar of several ``(milliseconds, ink)`` parts end to end from ``(fx, fy)``,
        ``scale`` px per millisecond. Returns the layout x of its end."""
        x = 0.0
        for v, ink in parts:
            w = float(v) * scale * smoothstep(0.0, 1.0, u)
            if w > 0.4:
                self.rect(sc, self.shift(self.P(fx, fy), x + w / 2, 0), w, thick, ink, op,
                          radius=min(3.0, w / 2))
            x += w
        return fx + self.frac(x)[0]

    def lines_chart(self, sc, box, xs, series, op, go=1.0, xticks=None, mark=None):
        """Log-log curves through points: ``series`` is ``[(ys, ink)]``."""
        ys = np.concatenate([np.asarray(s[0], float) for s in series])
        lo, hi = np.floor(np.log10(ys.min())), np.ceil(np.log10(ys.max()))
        chart = self.chart(box, xs, [10.0 ** lo, 10.0 ** hi], xlog=True, ylog=True)
        self.axes(sc, chart, op, xticks=xticks,
                  yticks=[(10.0 ** e, "%g" % 10.0 ** e)
                          for e in np.arange(lo, hi + 0.5, 2 if hi - lo > 4 else 1)])
        if mark is not None:
            chart.mark(sc, self.P, mark, AXIS, op, dashes=7)
        n = max(2, int(round(go * len(xs)))) if go > 0 else 0
        for values, ink in series:
            pts = [self.P(chart.fx(float(x)), chart.fy(float(y)), lift=0.8)
                   for x, y in zip(xs[:n], values[:n])]
            if len(pts) > 1:
                self.polyline(sc, pts, 2.8, ink, op, bias=0.3)
            for p in pts:
                self.disk(sc, p, 3.6, ink, op, lift=1.0, outline=0.0)
        return chart

    # ------------------------------------------------------------------ 1: the parts
    PARTS = (("storage", "keeps files while the power is off", STO, GREY),
             ("RAM (main memory)", "what running programs hold right now", RAM, GREY),
             ("CPU", "a few fast cores: runs the program", CPU, BLUE),
             ("bus", "the path data takes between the parts", BUS, GREY),
             ("GPU", "its own memory, and many small cores", GPU, GREEN))

    def phase_1(self, sc, u, op):
        now = int(np.clip((u - 0.06) / 0.86 * len(self.PARTS), 0, len(self.PARTS) - 1))
        self.status(sc, op, "0 ms", "steps done: 0")
        for n, (name, what, box, ink) in enumerate(self.PARTS):
            y = 0.415 + 0.102 * n
            seen = op * smoothstep(0.02 + 0.172 * n, 0.08 + 0.172 * n, u)
            self.chip(sc, PX0, y, AMBER if n == now else ink, seen, name, size=BIG, strong=True)
            self.text(sc, PX0 + 0.022, y + 0.045, what, seen, anchor="lm")
            if n == now:
                self.ring(sc, box, AMBER, seen)
        self.tag(sc, self.PICTURE, op)

    # ------------------------------------------------------------------ 2: CPU only
    STEPS_CPU = ("forward: the digits through the layers", "loss: how wrong the answers were",
                 "backward: a gradient for every weight", "update: every weight nudged")

    def legend(self, sc, op, y=0.78):
        self.chip(sc, PX0, y, VIOLET, op,
                  "the weights: %s numbers" % count(self.params["classifier"]))
        self.tile(sc, self.digits[0], PX0 + 0.007, y + 0.052, 13.0, op)
        self.text(sc, PX0 + 0.022, y + 0.052, "a batch: %d digits (6 drawn)" % self.batch, op,
                  anchor="lm")
        self.chip(sc, PX0, y + 0.104, AMBER, op, "arithmetic is happening here")

    def phase_2(self, sc, u, op):
        name, key = "classifier", ("classifier", "cpu")
        load = seg(u, 0.0, 0.08)
        self.memory(sc, op, shown=load)
        if 0 < load < 1:
            self.trip(sc, self.train_digits[:6], [F_STO], VIA_STO_RAM, [D_RAM], load, op,
                      size=19.0)
        lit, done, slow = np.zeros(6), 0, u < 0.60
        if slow:
            go, work, back = seg(u, 0.10, 0.22), seg(u, 0.22, 0.46), seg(u, 0.46, 0.56)
            self.block(sc, name, W_RAM, op, tint=1.0 if go == 0 or back == 1 else 0.35)
            if go > 0 and work < 0.12:
                self.trip(sc, self.batch_imgs(), self.picks, VIA_RAM_CPU, self.cpu_slots, go,
                          op * (1 - seg(work, 0.0, 0.12)))
            if go > 0 and back == 0:
                self.carry(sc, name, W_RAM, VIA_RAM_CPU, W_CPU, go, op)
            elif 0 < back < 1:
                self.carry(sc, name, W_CPU, VIA_RAM_CPU[::-1], W_RAM, back, op)
            if 0 < work < 1:
                lit = 0.5 + 0.5 * np.cos(2 * np.pi * (work * 5 - np.arange(6) / 6))
                self.ring(sc, CPU, AMBER, op)
            stage = int(np.clip(work * 4, 0, 3)) if 0 < work < 1 else -1
            done = int(back == 1)
            clock = ms(self.step[key] * seg(u, 0.10, 0.56))
            note = "one step, shown %s× slower than it would be" % count(float(
                "%.2g" % (0.46 * self.PHASES[1][1] / (self.step[key] / 1e3))))
        else:
            v = seg(u, 0.62, 0.97)
            k, w = divmod(v * 14, 1.0)
            self.block(sc, name, W_RAM, op)
            if v < 1:
                self.trip(sc, self.batch_imgs(int(k)), self.picks, VIA_RAM_CPU, self.cpu_slots,
                          seg(w, 0.0, 0.5), op * (1 - seg(w, 0.5, 0.6)), stagger=0.15)
                lit = np.full(6, seg(w, 0.45, 0.6) * (1 - seg(w, 0.85, 1.0)))
                if w > 0.6:
                    self.carry(sc, name, W_CPU, VIA_RAM_CPU[::-1], W_RAM, seg(w, 0.6, 1.0), op)
                self.ring(sc, CPU, AMBER, op * float(lit[0]))
            stage = -2
            done = int(round(ease_in_out(v) * self.steps_epoch))
            clock = S.duration(self.epoch[key] * done / self.steps_epoch)
            note = "every digit once, shown %s× slower" % count(float(
                "%.2g" % (0.35 * self.PHASES[1][1] / self.epoch[key])))
        self.cpu_cores(sc, lit, op)
        self.status(sc, op, clock, "steps done: %s" % f"{done:,}", note)
        for n, row in enumerate(self.STEPS_CPU):
            self.chip(sc, PX0, 0.425 + 0.052 * n, AMBER, op, row, on=stage in (n, -2),
                      strong=stage == n)
        self.text(sc, PX0 + 0.022, 0.655, "loss of this batch: %.2f" % self.first_loss,
                  op * (smoothstep(0.28, 0.32, u) if slow else 0.0), anchor="lm",
                  ink=dark(RED, 0.2))
        self.legend(sc, op)
        self.tag(sc, self.FIGURES, op)

    # ------------------------------------------------------------------ 3: with a GPU
    def phase_3(self, sc, u, op):
        name, key = "classifier", ("classifier", "gpu")
        part = {p: self.part[name, "gpu", p] for p in ("prepare", "move", "compute")}
        self.memory(sc, op)
        across, home, saved = seg(u, 0.02, 0.12), seg(u, 0.86, 0.93), seg(u, 0.93, 0.99)
        on_gpu = across == 1 and home == 0
        self.block(sc, name, W_RAM, op, tint=0.3 if across > 0 and home < 1 else 1.0)
        if 0 < across < 1:
            self.carry(sc, name, W_RAM, VIA_RAM_GPU, W_GPU, across, op)
        if on_gpu:
            self.block(sc, name, W_GPU, op)
        if 0 < home < 1:
            self.carry(sc, name, W_GPU, VIA_RAM_GPU[::-1], W_RAM, home, op)
        if saved > 0:
            self.carry(sc, name, W_RAM, VIA_STO_RAM[::-1], W_STO, saved, op)
        lit_cpu, lit_gpu, done, stage = np.zeros(6), 0.0, 0, -1
        if u < 0.14:
            clock, note, stage = ms(self.copy[name][0] * across), "the weights cross once", 0
        elif u < 0.52:
            pick, go, work, back = (seg(u, 0.14, 0.19), seg(u, 0.19, 0.30), seg(u, 0.30, 0.42),
                                    seg(u, 0.42, 0.51))
            lit_cpu[0] = float(0 < pick < 1)
            stage = 1 if pick < 1 else 2 if go < 1 else 3 if work < 1 else 4
            if pick > 0 and back < 1:
                self.trip(sc, self.batch_imgs(), self.picks, VIA_RAM_GPU, self.gpu_slots, go,
                          op * (1 - seg(back, 0.6, 1.0)))
            if 0 < work < 1:
                lit_gpu = 0.55 + 0.45 * np.sin(2 * np.pi * work * 4) ** 2
                self.ring(sc, GPU, AMBER, op)
            if 0 < back < 1:
                p = along([C_GPU, (GPU_TAP, 0.675), (GPU_TAP, BUS_Y), (W_CPU[0], BUS_Y), W_CPU],
                          ease_in_out(back))
                self.box(sc, p[0], p[1], 46, 24, soft(RED, 0.25), op, "%.2f" % self.first_loss,
                         border=RED, glow=0.3)
            done = int(back == 1)
            clock = ms(part["prepare"] * pick + part["move"] * go + part["compute"] * work)
            note = "one step, shown %s× slower than it would be" % count(float(
                "%.2g" % (0.37 * self.PHASES[2][1] / (self.step[key] / 1e3))))
        else:
            v = seg(u, 0.54, 0.85)
            k, w = divmod(v * 12, 1.0)
            if v < 1:
                self.trip(sc, self.batch_imgs(int(k)), self.picks, VIA_RAM_GPU, self.gpu_slots,
                          seg(w, 0.0, 0.55), op * (1 - seg(w, 0.85, 1.0)), stagger=0.15)
                lit_gpu = seg(w, 0.5, 0.6) * (1 - seg(w, 0.85, 1.0))
                self.ring(sc, GPU, AMBER, op * lit_gpu)
            stage = -2 if home == 0 else 5
            done = int(round(ease_in_out(v) * self.steps_epoch))
            clock = S.duration(self.epoch[key] * done / self.steps_epoch)
            note = "every digit once, shown %s× slower" % count(float(
                "%.2g" % (0.31 * self.PHASES[2][1] / self.epoch[key])))
        self.cpu_cores(sc, lit_cpu, op)
        self.gpu_cores(sc, lit_gpu, op)
        self.status(sc, op, clock, "steps done: %s" % f"{done:,}", note)
        rows = (("weights copied to the GPU, once", ms(self.copy[name][0]), VIOLET),
                ("the CPU picks a batch", ms(part["prepare"]), GREY),
                ("the batch crosses to the GPU", ms(part["move"]), OCHRE),
                ("GPU: forward, loss, backward, update", ms(part["compute"]), AMBER),
                ("only the loss comes back: %.2f" % self.first_loss, "", RED),
                ("at the end: weights back, and saved", ms(self.copy[name][0]), VIOLET))
        for n, (row, t, ink) in enumerate(rows):
            y = 0.415 + 0.05 * n
            self.chip(sc, PX0, y, ink, op, row, on=stage in (n, -2), strong=stage == n)
            self.text(sc, PX1, y, t, op * (1.0 if stage in (n, -2) else 0.55), anchor="rm")
        self.legend(sc, op)
        self.tag(sc, self.FIGURES, op)

    # ------------------------------------------------------------------ 4: many cores
    def phase_4(self, sc, u, op):
        a_op = op * (1 - smoothstep(0.44, 0.50, u))
        b_op = op * smoothstep(0.50, 0.56, u)
        self.memory(sc, op)
        self.block(sc, "classifier", W_RAM, op)
        H, n = self.hidden, len(self.hidden)
        vmax = float(np.percentile(H, 99))
        turn, once = seg(u, 0.05, 0.40), seg(u, 0.12, 0.40)
        # the CPU: a strip of the answer at a time; the GPU: every cell together
        shown = np.zeros_like(H)
        shown[:, :int(round(turn * n))] = H[:, :int(round(turn * n))]
        flash = min(1.0, np.floor(once * 4) / 3)
        if 0 < turn < 1 and u < 0.44:
            lit = np.full(6, 0.3)
            lit[int(turn * 30) % 6] = 1.0
            self.cpu_cores(sc, lit, op)
            self.gpu_cores(sc, float(0 < once < 1 and (once * 4) % 1 < 0.6), op)
        self.status(sc, op, ms(self.step["classifier", "gpu"]), "one step on the GPU")
        self.text(sc, PX0, 0.365, "one layer is one matrix multiplication", a_op, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, PX0, 0.412, "%d digits × %d units = %s sums, each" % (
            n, H.shape[1], f"{H.size:,}"), a_op, anchor="lm")
        self.text(sc, PX0, 0.452, "of 784 products; none waits for another", a_op, anchor="lm")
        for cx, values, title, ink in ((0.645, shown, "CPU: a few at a time", BLUE),
                                       (0.875, H * flash, "GPU: all at once", GREEN)):
            self.text(sc, cx, 0.508, title, a_op, ink=dark(ink, 0.4), strong=True)
            self.grid(sc, values, cx, 0.69, 140, 140, a_op, vmax=vmax, hot=ink, lines=0.0)
        self.text(sc, (PX0 + PX1) / 2, 0.875, "the same %s numbers, real ones: the first"
                  % f"{H.size:,}", a_op)
        self.text(sc, (PX0 + PX1) / 2, 0.915, "layer's outputs for this batch", a_op)
        # what the many cores do not speed up: a GPU step of each model, split
        self.text(sc, PX0, 0.365, "what many cores do not speed up", b_op, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, PX0, 0.412, "one step on the GPU, batch of %d, as shares:" % self.batch,
                  b_op, anchor="lm")
        grow, wide = seg(u, 0.56, 0.74), self.span(PX1 - PX0 - 0.01)[0]
        for n, name in enumerate(S.NAMES):
            key, y = (name, "gpu"), 0.49 + 0.20 * n
            pick, move = self.part[key + ("prepare",)], self.part[key + ("move",)]
            start = self.fixed[key] - pick - move
            sums = self.step[key] - self.fixed[key]
            self.text(sc, PX0, y, "%s: %s a step" % (name, ms(self.step[key])), b_op,
                      anchor="lm", strong=True)
            self.stacked(sc, PX0 + 0.005, y + 0.05, [(pick, GREY), (move, OCHRE), (start, RED),
                                                    (sums, AMBER)], wide / self.step[key],
                         b_op, grow, thick=20.0)
            self.text(sc, PX0, y + 0.10, "arithmetic %s (%d%%); the rest is fixed" % (
                ms(sums), round(100 * sums / self.step[key])), b_op * grow, anchor="lm")
        for x, ink, word in ((PX0, GREY, "pick batch"), (PX0 + 0.135, OCHRE, "move it"),
                             (PX0 + 0.235, RED, "start"), (PX0 + 0.315, AMBER, "arithmetic")):
            self.chip(sc, x, 0.905, ink, b_op, word)
        self.tag(sc, self.FIGURES, op)

    # ------------------------------------------------------------------ 5: precision
    SIGN, EXPO, FRAC = (0.82, 0.30, 0.26), OCHRE, TEAL     # the parts of a number's bits
    WIDTHS = (("cpu", "FP64", "CPU, 64-bit"), ("cpu", "FP32", "CPU, 32-bit"),
              ("gpu", "FP64", "GPU, 64-bit"), ("gpu", "FP32", "GPU, 32-bit"),
              ("gpu", "FP16", "GPU, 16-bit"), ("gpu", "FP8", "GPU, 8-bit"),
              ("gpu", "FP4", "GPU, 4-bit"))

    def phase_5(self, sc, u, op):
        """Bits a number gets; how fast each width is; what fewer bits do to the model."""
        a_op = op * (1 - smoothstep(0.33, 0.37, u))
        b_op = op * smoothstep(0.37, 0.41, u) * (1 - smoothstep(0.64, 0.68, u))
        c_op = op * smoothstep(0.68, 0.72, u)
        self.memory(sc, op)
        self.block(sc, "classifier", W_RAM, op)
        self.block(sc, "classifier", W_GPU, op)
        # the map says what each is built for
        self.text(sc, CPU[0] + 0.062, 0.745, "64-bit", op, anchor="lm", ink=dark(BLUE, 0.4),
                  strong=True)
        self.text(sc, (GPU[0] + GPU[1]) / 2, 0.645, "32-bit; also 16, 8, 4", op,
                  ink=dark(GREEN, 0.45), strong=True)
        lit = 0.5 + 0.5 * np.sin(2 * np.pi * u * 6) ** 2
        if u < 0.37 or u > 0.68:
            self.cpu_cores(sc, np.full(6, 0.35 * lit), op)
        else:
            self.cpu_cores(sc, np.full(6, lit), op * float(u < 0.52))
            self.gpu_cores(sc, lit, op * float(u >= 0.52))
        self.precision_bits(sc, seg(u, 0.03, 0.30), a_op)
        self.precision_speed(sc, seg(u, 0.41, 0.60), b_op)
        self.precision_model(sc, seg(u, 0.72, 0.92), c_op)
        self.tag(sc, self.BITS, op)

    def precision_bits(self, sc, u, op):
        """Each format's bits as a bar, sign, exponent and fraction, and the example
        number as each stores it."""
        if op <= 0.01:
            return
        self.text(sc, PX0, 0.33, "how many bits a number gets", op, anchor="lm", strong=True,
                  size=BIG)
        self.text(sc, PX0, 0.376, "the bits; %g as each stores it; how far off" % self.example,
                  op, anchor="lm")
        full = self.span(0.16)[0] / 64.0                  # px per bit
        for n, (name, sg, ex, fr, f) in enumerate(self.formats):
            y = 0.43 + 0.062 * n
            seen = op * smoothstep(0.02 + 0.13 * n, 0.10 + 0.13 * n, u)
            self.text(sc, PX0, y, name, seen, anchor="lm", strong=True)
            self.stacked(sc, PX0 + 0.055, y, [(sg, self.SIGN), (ex, self.EXPO),
                                               (fr, self.FRAC)], full, seen, thick=15.0)
            v = f["stored_example"]
            shown = "0" if v == 0 else "%.10g" % v
            self.text(sc, PX0 + 0.235, y, shown, seen, anchor="lm",
                      ink=dark(RED, 0.2) if v == 0 else None, strong=v == 0)
            self.text(sc, PX1, y, pow10(f["example_error"]), seen, anchor="rm",
                      ink=dark(RED, 0.2) if v == 0 else layout.NOTE_INK)
        y = 0.43 + 0.062 * len(self.formats) + 0.01
        for x, ink, word in ((PX0, self.SIGN, "sign"), (PX0 + 0.08, self.EXPO, "exponent: range"),
                             (PX0 + 0.25, self.FRAC, "fraction: precision")):
            self.chip(sc, x, y, ink, op * smoothstep(0.75, 0.85, u), word)
        grid = S.small_grid("FP4")
        self.text(sc, PX0, y + 0.052, "FP4 holds only %d sizes: %s" % (
            len(grid), ", ".join("%g" % v for v in grid)), op * smoothstep(0.85, 0.95, u),
            anchor="lm")

    def precision_speed(self, sc, u, op):
        """The representative speed of each width, on a log scale."""
        if op <= 0.01:
            return
        self.text(sc, PX0, 0.33, "how fast each width is", op, anchor="lm", strong=True,
                  size=BIG)
        self.text(sc, PX0, 0.376, "trillion multiply-adds a second; log scale", op,
                  anchor="lm")
        lo, hi = -1.0, np.log10(max(self.madds.values())) + 0.35
        wide = self.span(0.21)[0]
        for n, (dev, fmt, label) in enumerate(self.WIDTHS):
            y = 0.43 + 0.058 * n
            v = self.madds[dev, fmt]
            grow = seg(u, 0.05 + 0.10 * n, 0.15 + 0.10 * n)
            self.text(sc, PX0 + 0.115, y, label, op, anchor="rm", ink=dark(INK[dev], 0.4),
                      strong=True)
            self.hbar(sc, PX0 + 0.125, y, wide * (np.log10(v) - lo) / (hi - lo), INK[dev], op,
                      value="%.3g" % v, u=grow, thick=14.0)
        say = op * smoothstep(0.75, 0.85, u)
        cpu64, gpu64 = self.madds["cpu", "FP64"], self.madds["gpu", "FP64"]
        self.text(sc, PX0, 0.855, "CPU: 64-bit at full speed, 32-bit %s" % times(
            self.madds["cpu", "FP32"] / cpu64), say, anchor="lm")
        self.text(sc, PX0, 0.897, "GPU: 64-bit at 1/%d of its 32-bit: %s the CPU's" % (
            round(self.madds["gpu", "FP32"] / gpu64),
            "below" if gpu64 < cpu64 else "about"), say, anchor="lm")

    def precision_model(self, sc, u, op):
        """The trained classifier with its weights stored in each format (measured)."""
        if op <= 0.01:
            return
        self.text(sc, PX0, 0.33, "the classifier's weights in fewer bits", op, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, PX0, 0.376, "held-out accuracy; below 64-bit, sums in 32-bit", op,
                  anchor="lm")
        wide = self.span(0.15)[0]
        rows = list(S.VARIANTS)
        for n, v in enumerate(rows):
            y = 0.425 + 0.05 * n
            r = self.fewer[v]
            bad = r["accuracy"] < 0.5
            label = v.replace(" scaled", ", a scale per %d" % S.BLOCK)
            grow = seg(u, 0.04 + 0.08 * n, 0.14 + 0.08 * n)
            self.text(sc, PX0 + 0.175, y, label, op, anchor="rm", strong=True)
            self.hbar(sc, PX0 + 0.185, y, wide * r["accuracy"], RED if bad else VIOLET, op,
                      value="%.1f%%" % (100 * r["accuracy"]), u=grow, thick=13.0)
        z = self.fewer["FP4"]["zero_weights"]
        say = op * smoothstep(0.85, 0.95, u)
        self.text(sc, PX0, 0.83, "plain FP4 rounds %d%% of the weights to 0;" % round(100 * z),
                  say, anchor="lm", ink=dark(RED, 0.2))
        self.text(sc, PX0, 0.87, "a shared scale per %d weights brings them back" % S.BLOCK,
                  say, anchor="lm")
        mb = {f[0]: f[4]["autoencoder_mb"] for f in self.formats}
        self.text(sc, PX0, 0.915, "autoencoder weights: %.2g MB in 64-bit, %.2g in 4-bit"
                  % (mb["FP64"], mb["FP4"]), say, anchor="lm")

    # ------------------------------------------------------------------ 6, 7: the two models
    def model(self, sc, u, op, name):
        cpu, gpu = (name, "cpu"), (name, "gpu")
        self.memory(sc, op, name)
        across = seg(u, 0.06, 0.24)
        self.block(sc, name, W_RAM, op, tint=1.0 if across < 1 else 0.35)
        if 0 < across < 1:
            self.carry(sc, name, W_RAM, VIA_RAM_GPU, W_GPU, across, op)
        elif across == 1:
            self.block(sc, name, W_GPU, op)
        v = seg(u, 0.26, 0.98)
        k, w = divmod(v * 10, 1.0)
        if 0 < v < 1:
            self.trip(sc, self.batch_imgs(int(k)), self.picks, VIA_RAM_GPU,
                      [(GPU_TAP + 0.01, BUS_Y)], seg(w, 0.0, 0.7), op * (1 - seg(w, 0.6, 0.7)),
                      stagger=0.15)
            self.gpu_cores(sc, seg(w, 0.6, 0.7) * (1 - seg(w, 0.9, 1.0)), op)
        slow = u < 0.5
        self.status(sc, op, ms(self.step[cpu] if slow else self.step[gpu]),
                    "one step on the %s" % ("CPU" if slow else "GPU"))
        self.text(sc, PX0, 0.365, "the " + name, op, anchor="lm", strong=True, size=BIG)
        i = 1                                          # what it does, on a held-out digit
        self.tile(sc, self.held[i], 0.845, 0.365, 22.0, op)
        self.arrow(sc, (0.866, 0.365), (0.905, 0.365), AXIS, op, 1.0, 2.2)
        if name == "classifier":
            self.text(sc, 0.918, 0.365, "“%d”" % self.named[i], op, anchor="lm", strong=True,
                      size=BIG)
        else:
            self.tile(sc, self.rebuilt[i], 0.932, 0.365, 22.0, op)
        self.text(sc, PX0, 0.412, " → ".join("%d" % v for v in S.MODELS[name]["sizes"]), op,
                  anchor="lm")
        if name == "classifier":
            facts = "%s weights; %s multiply-adds a step" % (count(self.params[name]),
                                                              count(self.work[name]))
        else:
            facts = "%s weights (%s); %s the arithmetic" % (
                count(self.params[name]), times(self.params[name] / self.params["classifier"]),
                times(self.work[name] / self.work["classifier"]))
        self.text(sc, PX0, 0.452, facts, op, anchor="lm")
        scale = self.span(0.20)[0] / max(self.step[cpu], self.step[gpu])
        grow = seg(u, 0.10, 0.32)
        for n, key in enumerate((cpu, gpu)):
            y = 0.512 + 0.05 * n
            self.text(sc, PX0 + 0.04, y, S.LABEL[key[1]], op, anchor="rm",
                      ink=dark(INK[key[1]], 0.4), strong=True)
            end = self.stacked(sc, PX0 + 0.05, y, [(self.part[key + ("prepare",)], GREY),
                               (self.part[key + ("move",)], OCHRE),
                               (self.part[key + ("compute",)], INK[key[1]])], scale, op, grow)
            self.text(sc, end + 0.01, y, ms(self.step[key]) + " a step", op * grow, anchor="lm",
                      strong=True)
        for x, ink, word in ((PX0, GREY, "pick batch"), (PX0 + 0.14, OCHRE, "move it"),
                             (PX0 + 0.25, AXIS, "arithmetic")):
            self.chip(sc, x, 0.618, ink, op, word)
        self.text(sc, PX0, 0.67, "ms a step = fixed + per digit × batch (a model):", op,
                  anchor="lm")
        xt = [(v, "%d" % v) for v in self.batches[1::2]]
        self.lines_chart(sc, ((0.62, 0.965), (0.71, 0.875)), self.batches,
                         [(self.by_batch[cpu], BLUE), (self.by_batch[gpu], GREEN)], op,
                         go=seg(u, 0.40, 0.72), xticks=xt, mark=self.batch)
        self.tag(sc, self.FIGURES, op)

    def phase_6(self, sc, u, op):
        self.model(sc, u, op, "classifier")

    def phase_7(self, sc, u, op):
        self.model(sc, u, op, "autoencoder")

    # ------------------------------------------------------------------ 8: four bars
    def phase_8(self, sc, u, op):
        self.memory(sc, op)
        self.block(sc, "classifier", W_RAM, op)
        self.block(sc, "autoencoder", W_GPU, op)
        full = max(self.step.values())
        scale = self.span(0.17)[0] / full
        self.status(sc, op, ms(full), "the slowest of the four",
                    "time per step, batch of %d: one linear scale" % self.batch)
        row = 0
        for g, name in enumerate(S.NAMES):
            y0 = 0.42 + 0.155 * g
            self.text(sc, PX0, y0, "%s: %s weights" % (name, count(self.params[name])), op,
                      anchor="lm", strong=True)
            for n, dev in enumerate(S.DEVICES):
                y = y0 + 0.048 * (n + 1)
                grow = seg(u, 0.06 + 0.10 * row, 0.20 + 0.10 * row)
                row += 1
                self.text(sc, PX0 + 0.04, y, S.LABEL[dev], op, anchor="rm",
                          ink=dark(INK[dev], 0.4), strong=True)
                end = self.stacked(sc, PX0 + 0.05, y, [(self.step[name, dev], INK[dev])], scale,
                                   op, grow)
                self.text(sc, end + 0.01, y, "%s; an epoch %s" % (
                    ms(self.step[name, dev]), S.duration(self.epoch[name, dev])), op * grow,
                    anchor="lm", strong=True)
        say = op * smoothstep(0.50, 0.60, u)
        top = int(self.batches[-1])
        for n, name in enumerate(S.NAMES[::-1]):
            r, wide = self.ratio[name], self.by_batch[name, "cpu"][-1] / \
                self.by_batch[name, "gpu"][-1]
            y = 0.762 + 0.094 * n
            self.chip(sc, PX0, y, GREEN if r > 1 else BLUE, say, "%s: the GPU is %s" % (
                name, self.versus(r)), strong=True)
            self.text(sc, PX0 + 0.022, y + 0.042, "at batches of %s: %s" % (
                f"{top:,}", self.versus(wide)), say, anchor="lm")
        self.tag(sc, self.FIGURES, op)

    # ------------------------------------------------------------------ 9: the race
    def phase_9(self, sc, u, op):
        now = seg(u, *self.RACE) * self.race_span          # seconds on the race's clock
        a = "autoencoder"
        self.memory(sc, op, a)
        self.block(sc, a, W_GPU, op)
        self.block(sc, a, W_RAM, op)
        cpu_on = any(0 < now < self.race_s[n, "cpu"] for n in S.NAMES)
        gpu_on = any(0 < now < self.race_s[n, "gpu"] for n in S.NAMES)
        k, beat = divmod(u * self.seconds("race") * 1.6, 1.0)
        if cpu_on:
            self.trip(sc, self.batch_imgs(int(k))[:3], self.picks[:3], VIA_RAM_CPU,
                      self.cpu_slots[:3], seg(beat, 0, 0.7), op * (1 - seg(beat, 0.6, 0.7)),
                      stagger=0.15)
            self.ring(sc, CPU, AMBER, op * 0.8)
            self.cpu_cores(sc, np.full(6, 0.4 + 0.6 * (beat > 0.6)), op)
        if gpu_on:
            self.trip(sc, self.batch_imgs(int(k) + 1)[3:], self.picks[3:], VIA_RAM_GPU,
                      [(GPU_TAP + 0.01, BUS_Y)], seg(beat, 0, 0.7),
                      op * (1 - seg(beat, 0.6, 0.7)), stagger=0.15)
            self.ring(sc, GPU, AMBER, op * 0.8)
            self.gpu_cores(sc, 0.4 + 0.6 * (beat > 0.6), op)
        steps = self.total_steps
        self.status(sc, op, "%.1f s" % now, "a real clock" if self.race_scale == 1 else
                    "clock runs %d× fast" % self.race_scale,
                    "%d epochs each: %s steps of %d digits" % (self.meta["epochs"], f"{steps:,}",
                                                               self.batch))
        track = self.span(0.095)[0]
        for n, key in enumerate(S.FOUR):
            name, dev = key
            y = 0.412 + 0.045 * n
            frac = min(1.0, now / self.race_s[key])
            self.text(sc, PX0 + 0.14, y, "%s, %s" % (name, S.LABEL[dev]), op, anchor="rm",
                      ink=dark(INK[dev], 0.4))
            self.hbar(sc, PX0 + 0.15, y, track * frac, INK[dev], op, thick=13.0, track=track)
            at = frac * steps
            if name == "classifier":
                value = "%.1f%% right" % (100 * np.interp(at, self.acc_steps,
                                                          self.acc.mean(0)))
            else:
                value = "loss %.4f" % np.interp(at, self.ae_steps, self.ae.mean(0))
            done = frac >= 1
            self.text(sc, PX0 + 0.256, y, "%s in %s" % (value, S.duration(self.race_s[key]))
                      if done else value, op, anchor="lm", strong=done)
        self.text(sc, PX0, 0.612, "autoencoder: held-out loss (log scale)", op, anchor="lm",
                  strong=True)
        lo, hi = float(self.ae.min()) * 0.8, float(self.ae.max()) * 1.1
        chart = self.chart(((0.62, 0.965), (0.65, 0.875)), [0.0, self.race_span], [lo, hi],
                           ylog=True)
        tick = min((1, 2, 5, 10, 20, 30, 60), key=lambda v: abs(self.race_span / 4 - v))
        self.axes(sc, chart, op, yticks=[(v, "%g" % v) for v in (0.01, 0.03, 0.1) if lo < v < hi],
                  xticks=[(float(v), "%d s" % v)
                          for v in range(0, int(self.race_span) + 1, tick)])
        for dev in S.DEVICES:
            secs = self.ae_steps / steps * self.race_s[a, dev]
            for seed in self.ae:
                chart.curve(sc, self.P, secs, seed, now, INK[dev], op * 0.9, width=1.0,
                            dot=False)
            if now > 0:
                t = min(now, secs[-1])
                y = float(np.interp(t, secs, self.ae.mean(0)))
                self.disk(sc, self.P(chart.fx(t), chart.fy(y), lift=1.0), 5.0, INK[dev], op,
                          glow=0.4, lift=1.4, outline=0.0)
        self.chip(sc, 0.84, 0.685, BLUE, op, "CPU")
        self.chip(sc, 0.84, 0.729, GREEN, op, "GPU")
        self.tag(sc, self.RACED, op)

    # ------------------------------------------------------------------ 10: scaling up
    def phase_10(self, sc, u, op):
        self.memory(sc, op, "autoencoder")
        self.block(sc, "autoencoder", W_GPU, op)
        self.block(sc, "autoencoder", W_RAM, op, tint=0.35)
        self.gpu_cores(sc, 0.5 + 0.5 * np.sin(2 * np.pi * u * 9) ** 2, op)
        ep = int(self.meta["projection_epochs"])
        shown = int(np.clip((u - 0.14) / 0.60 * 4, 0, 3))
        self.status(sc, op, S.duration(self.projections[shown][1]["gpu"]),
                    "on the GPU, by arithmetic",
                    "%d epochs: steps × time per step × how much larger" % ep)
        lo, hi = -1.0, np.log10(max(self.projections[-1][1].values())) + 0.2
        wide = self.span(0.17)[0]

        def at(seconds):
            return wide * (np.log10(seconds) - lo) / (hi - lo)

        def bar(y, seconds, ink, seen, grow=1.0):
            w = max(at(seconds) * smoothstep(0, 1, grow), 0.5)
            self.rect(sc, self.shift(self.P(PX0 + 0.005, y), w / 2, 0), w, 8.0, ink, seen,
                      radius=3)
        self.text(sc, PX0, 0.405, "a whole training (bars: log scale, ×10 a step)", op,
                  anchor="lm", strong=True)
        self.text(sc, PX0, 0.455, "a video app on a phone opens:", op, anchor="lm")
        bar(0.497, 1.0, GREY, op)
        self.text(sc, PX0 + 0.015 + self.frac(at(1.0))[0], 0.497,
                  "under 1 s (an everyday wait)", op, anchor="lm")
        for n, (what, seconds) in enumerate(self.projections):
            y = 0.56 + 0.095 * n
            seen = op * smoothstep(0.08 + 0.15 * n, 0.16 + 0.15 * n, u)
            grow = seg(u, 0.10 + 0.15 * n, 0.22 + 0.15 * n)
            self.text(sc, PX0, y, what + (": %d epochs" % ep if n == 0 else ""),
                      seen, anchor="lm", strong=True)
            bar(y + 0.034, seconds["gpu"], GREEN, seen, grow)
            bar(y + 0.054, seconds["cpu"], BLUE, seen, grow)
            self.text(sc, PX0 + 0.015 + self.frac(at(max(seconds.values())))[0], y + 0.044,
                      "GPU %s, CPU %s" % (S.duration(seconds["gpu"]), S.duration(seconds["cpu"])),
                      seen * grow, anchor="lm")
        self.tag(sc, self.ARITHMETIC, op)

    # ------------------------------------------------------------------ 11: what to do
    def phase_11(self, sc, u, op):
        a = "autoencoder"
        self.memory(sc, op, a)
        self.block(sc, a, W_GPU, op)
        self.block(sc, a, W_RAM, op, tint=0.35)
        self.gpu_cores(sc, 0.5 + 0.5 * np.sin(2 * np.pi * u * 7) ** 2, op)
        self.ring(sc, GPU, AMBER, op * 0.8)
        test = 100 * self.step[a, "gpu"] / 1e3
        self.status(sc, op, S.duration(test), "100 test steps, GPU")
        rows = (("time a small test first", "100 steps of this autoencoder: %s"
                 % S.duration(test)),
                ("multiply up before you start", "steps × time per step = the wait"),
                ("let long runs go unattended", "a batch job: no screen, results to a file"),
                ("share the big machines", "a GPU cluster and its queue, for everyone"))
        for n, (head, what) in enumerate(rows):
            y = 0.42 + 0.125 * n
            seen = op * smoothstep(0.03 + 0.18 * n, 0.11 + 0.18 * n, u)
            self.box(sc, PX0 + 0.013, y, 24, 24, soft(AMBER, 0.3), seen, "%d" % (n + 1),
                     border=AMBER, size=0.031)
            self.text(sc, PX0 + 0.038, y, head, seen, anchor="lm", strong=True, size=BIG)
            self.text(sc, PX0 + 0.038, y + 0.046, what, seen, anchor="lm")
        self.tag(sc, self.ARITHMETIC, op)


ANIMATIONS = {"cpu-and-gpu-training": CpuAndGpuTraining}
