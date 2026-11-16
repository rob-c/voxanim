"""The transformer block, and what is built from it: animations 13-20 of the
"next 100" (see :mod:`voxanim.mech`).

Each draws the mechanism first and the measurement after it. The measurements
are the study's: NumPy transformers trained on Tiny Shakespeare, or the pinned
GPT-2 (:mod:`voxanim.next_attention`), and for five of them the redesigned
experiments of :mod:`voxanim.study_block`, which train every variant over
several seeds. What a mechanism needs beyond them is computed here from the
same trained models and the same text.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from functools import cache

import numpy as np

from . import next_attention as NA
from . import realdata as R
from . import study_block as SB
from . import tinylm as T
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    nice_ticks,
    number,
    soft,
)

# One meaning per colour in every animation of this module.
ATTN, MLP, NORM = TEAL, VIOLET, AMBER
STREAM = (0.30, 0.32, 0.35)


# ---------------------------------------------------------------------- the trained models
def base_config():
    """The study's base model (``next_attention._make``)."""
    return T.Config(vocab=len(R.vocabulary()), width=16, heads=4, kv_heads=4, ff=32, max_len=48)


def study_configs(name):
    """``(configs, cache name, updates)`` of the models the first study of
    ``name`` trained: the same arguments ``next_attention`` passes to
    ``compare``, so the checkpoints come back from its cache."""
    c = base_config()
    if name == "weight-tying":
        return [c, replace(c, tied=True)], "reuse-the-input-embedding", 240
    raise KeyError(name)


@cache
def trained(name):
    """``(configs, runs, heldout losses, test arrays)``: the study's own
    checkpoints (``runs[i]["P"]``) and the held-out windows it scored them on.
    For a study trained over several seeds (:mod:`voxanim.study_block`) these
    are each variant's first seed: the model a mechanism is drawn from."""
    if name in SB.STUDIES:
        e = SB.experiment(name)
        return e["configs"], [rs[0] for rs in e["runs"]], e["heldout"][:, 0], e["test"]
    configs, label, steps = study_configs(name)
    runs, losses, _train, test = NA.compare(configs, [""] * len(configs), steps=steps, name=label)
    return configs, runs, losses, test


def show(ch):
    """A character as it is drawn in a token box."""
    return {" ": "␣", "\n": "↵", "\x00": "?", "\ufffd": "?"}.get(ch, ch)


def test_window(row=0):
    """One held-out window the studies scored: ``(ids, characters as drawn)``."""
    ids = R.language_data(n=64, length=12, split="test", seed=73)[0][row]
    return ids, [show(ch) for ch in R.decode(ids)]


def block_trace(P, ids, c):
    """Every vector of the first block for the tokens ``ids`` (one window):
    the arithmetic of ``tinylm._forward`` kept step by step. ``{name: (n, width)}``:
    ``x`` the stream entering, ``attn`` what attention writes, ``mid_raw`` the
    stream after that write and ``mid`` after post-LN's norm (the same for
    pre-LN), ``reads`` what the MLP is given (after its norm, for pre-LN),
    ``pre`` the hidden units before the activation, ``gate`` (SwiGLU) the
    multipliers, ``hidden`` the units after it, ``mlp`` what the MLP writes,
    ``out_raw`` / ``out`` the stream leaving."""
    V = {k: T.variable(v) for k, v in P.items()}
    x = np.asarray(ids, int)[None]
    n = x.shape[1]
    h = V["E"][x] + V["pos"][np.arange(n)]
    p, pre = "l0.", c.placement == "pre"
    hn = T._norm(h, V, p + "n1", c) if pre else h
    a = T._attention(hn, hn, V, p, c, T.attention_mask(n, c), [])
    mid_raw = h + a
    mid = mid_raw if pre else T._norm(mid_raw, V, p + "n1", c)
    z = h if c.parallel else mid
    if pre:
        z = T._norm(z, V, p + "n2", c)
    z1 = z @ V[p + "W1"]
    out = dict(x=h.v, attn=a.v, mid_raw=mid_raw.v, mid=mid.v, reads=z.v, pre=z1.v)
    if c.activation == "gelu":
        hidden = T.gelu(z1)
    else:
        silu = z1 * (1 / (1 + (-z1).exp()))
        gate = z @ V[p + "gate"]
        hidden = silu * gate
        out.update(silu=silu.v, gate=gate.v)
    mlp = hidden @ V[p + "W2"]
    out_raw = mid + mlp
    end = out_raw if pre else T._norm(out_raw, V, p + "n2", c)
    out.update(hidden=hidden.v, mlp=mlp.v, out_raw=out_raw.v, out=end.v)
    return {k: v[0] for k, v in out.items()}


def size(v):
    """Length of a vector."""
    return float(np.linalg.norm(v))


# ---------------------------------------------------------------------- shared drawing
def seg(u, a, b):
    """0..1 as ``u`` runs from ``a`` to ``b``."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


class Built(Mechanism):
    """What the eight animations share: wires a value rides along, bars about
    a zero line, the training race and the held-out result."""

    NAMES: tuple = ()
    COLOURS: tuple = ()

    # -- wires
    def _along(self, pts, u):
        """The part of the path ``pts`` (layout points) up to fraction ``u``
        of its length, as a list of points ending at that position."""
        pts = [np.asarray(p, float) for p in pts]
        lens = [float(np.hypot((b[0] - a[0]) * 960, (b[1] - a[1]) * 540))
                for a, b in itertools.pairwise(pts)]
        want = sum(lens) * float(np.clip(u, 0.0, 1.0))
        out = [pts[0]]
        for a, b, d in zip(pts, pts[1:], lens):
            if want >= d:
                out.append(b)
                want -= d
            else:
                out.append(a + (b - a) * (want / max(d, 1e-9)))
                break
        return out

    def _cut(self, pts, boxes):
        """The path ``pts`` with the parts inside ``boxes`` (``(fx, fy, w px,
        h px)``) removed: a wire stops at the edge of what it enters."""
        paths = [[np.asarray(pts[0], float)]]
        for a, b in itertools.pairwise(pts):
            a, b = np.asarray(a, float), np.asarray(b, float)
            holes = []
            for cx, cy, w, h in boxes:
                hw, hh = self.frac(w / 2 + 2.0, h / 2 + 2.0)
                t0, t1, hit = 0.0, 1.0, True
                for p, d, c, half in ((a[0], b[0] - a[0], cx, hw), (a[1], b[1] - a[1], cy, hh)):
                    if abs(d) < 1e-12:
                        hit = hit and abs(p - c) < half
                    else:
                        lo, hi = sorted(((c - half - p) / d, (c + half - p) / d))
                        t0, t1 = max(t0, lo), min(t1, hi)
                if hit and t1 > t0:
                    holes.append((t0, t1))
            t = 0.0
            for h0, h1 in sorted(holes):
                if h0 > t + 1e-9:
                    paths[-1].append(a + (b - a) * h0)
                    paths.append([a + (b - a) * h1])
                else:
                    paths[-1][-1] = a + (b - a) * max(h1, t)
                t = max(t, h1)
            if t < 1.0 - 1e-9:
                paths[-1].append(b)
        return [q for q in paths if len(q) > 1]

    def wire(self, sc, pts, ink, op, u=1.0, width=2.6, head=True, boxes=()):
        """A path through layout points, drawn as far as ``u``; it stops at
        the edge of each of ``boxes`` and starts again on the far side."""
        if u <= 0.0 or op <= 0.01:
            return
        for part in self._cut(self._along(pts, u), boxes):
            for i, (a, b) in enumerate(itertools.pairwise(part)):
                last = i == len(part) - 2
                if np.hypot((b[0] - a[0]) * 960, (b[1] - a[1]) * 540) > 1.0:
                    self.arrow(sc, tuple(a), tuple(b), ink, op, 1.0, width, head and last)

    def ride(self, sc, pts, u, ink, op, r=6.5):
        """A value travelling along the path ``pts``."""
        if not 0.0 < u < 1.0 or op <= 0.01:
            return
        at = self._along(pts, u)[-1]
        fade = min(1.0, u / 0.08, (1 - u) / 0.08)
        self.disk(sc, self.P(float(at[0]), float(at[1])), r, ink, op * fade, glow=0.7, lift=1.8,
                  outline=0.0)

    def dashes(self, sc, a, b, ink, op, n=18, width=1.6):
        """A dashed straight line between two layout points."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        for i in range(n):
            p, q = a + (b - a) * (i / n), a + (b - a) * ((i + 0.55) / n)
            self.polyline(sc, [self.P(*map(float, p)), self.P(*map(float, q))], width, ink,
                          op * 0.45, bias=0.0)

    # -- things
    def vbars(self, sc, values, fx0, fx1, fy0, unit, ink, op, fill=0.72, neg=None):
        """One bar per value, standing on (or hanging from) the zero line
        ``fy0``; ``unit``: layout height of 1.0. Returns the bars' x."""
        values = np.asarray(values, float)
        n = len(values)
        pitch = (fx1 - fx0) / n
        w = self.span(pitch)[0] * fill
        xs = [fx0 + (i + 0.5) * pitch for i in range(n)]
        for x, v in zip(xs, values):
            h = abs(v) * unit
            hpx = max(self.span(0, h)[1], 1.0)
            col = ink if v >= 0 or neg is None else neg
            self.rect(sc, self.P(float(x), float(fy0 - np.sign(v) * h / 2)), w, hpx, col, op,
                      radius=min(2.5, w / 2, hpx / 2))
        return xs

    def strip(self, sc, v, fx, fy, op, cell=12.0, wide=20.0, vmax=None, ink=RED):
        """A vector as a column of cells centred at ``(fx, fy)``."""
        v = np.asarray(v, float)
        self.grid(sc, v[:, None], fx, fy, wide, cell * len(v), op, vmax=vmax, hot=ink)

    def node(self, sc, fx, fy, text, ink, op, on=0.0, w=92.0, h=46.0, size=0.036):
        """A sublayer: pale until a value is inside it (``on`` 0..1)."""
        on = float(np.clip(on, 0.0, 1.0))
        fill = tuple(np.asarray(soft(ink, 0.62)) * (1 - on) + np.asarray(ink) * on)
        self.box(sc, fx, fy, w, h, fill, op, text,
                 ink=tuple(np.asarray(dark(ink, 0.45)) * (1 - on) + np.asarray(WHITE) * on),
                 border=ink, size=size, glow=0.18 * on)

    def plus(self, sc, fx, fy, op, on=0.0):
        """Where a branch adds what it computed into the stream."""
        self.node(sc, fx, fy, "+", STREAM, op, on, w=34.0, h=34.0, size=0.040)

    # -- one block, its two sublayers one after the other
    def block(self, sc, u, op, f, ink, post, caption, cell=14.0):
        """One block as a diagram: the stream running left to right, the
        attention branch and then the MLP branch leaving it and adding back,
        the normalisation on the branches (pre-LN) or on the stream (``post``).
        ``f``: its :func:`block_trace`; the stream's vector for character
        ``self.at`` is drawn where it enters, between the branches and where
        it leaves."""
        ys, yb = self.YS, self.YB
        if post:
            tap_a, attn, plus_a, ln_a, mid, tap_b, mlp, plus_b, ln_b = (
                0.165, 0.265, 0.365, 0.44, 0.515, 0.575, 0.67, 0.765, 0.835)
        else:
            tap_a, ln_a, attn, plus_a, mid, tap_b, ln_b, mlp, plus_b = (
                0.165, 0.245, 0.355, 0.455, 0.515, 0.575, 0.65, 0.755, 0.855)
        x_in, x_out = 0.10, 0.905
        wide = {"attention": 124.0, "MLP": 84.0}
        boxes = [(attn, yb, wide["attention"], 50.0), (mlp, yb, wide["MLP"], 50.0),
                 (plus_a, ys, 34.0, 34.0), (plus_b, ys, 34.0, 34.0),
                 (ln_a, ys if post else yb, 50.0, 50.0), (ln_b, ys if post else yb, 50.0, 50.0),
                 (x_in, ys, 30.0, 8), (mid, ys, 30.0, 8), (x_out, ys, 30.0, 8)]
        # the stream
        self.wire(sc, [(x_in, ys), (x_out + 0.045, ys)], STREAM, op, seg(u, 0.0, 0.10), 5.0,
                  boxes=boxes)
        halves = ((tap_a, ln_a, attn, plus_a, "attention", ATTN, 0.12, 0.47),
                  (tap_b, ln_b, mlp, plus_b, "MLP", MLP, 0.50, 0.85))
        for tap, ln, sub, plus, name, colour, t0, t1 in halves:
            w = seg(u, t0, t1)
            path = [(tap, ys), (tap, yb), (plus, yb), (plus, ys)]
            self.wire(sc, path, colour, op * smoothstep(0.0, 0.08, u), 1.0, 2.6, boxes=boxes)
            self.ride(sc, path, seg(w, 0.10, 0.90), colour, op)
            self.ride(sc, [(tap - 0.05, ys), (plus, ys)], seg(w, 0.0, 0.90), STREAM, op)
            inside = 1.0 - abs(seg(w, 0.10, 0.90) - 0.5) * 2
            self.node(sc, sub, yb, name, colour, op, inside if w < 1 else 0.0, w=wide[name],
                      h=50.0)
            self.plus(sc, plus, ys, op, smoothstep(0.85, 0.95, w) * (1 - smoothstep(0.97, 1.0, w)))
            if post:
                self.node(sc, ln, ys, "LN", NORM, op,
                          smoothstep(0.92, 1.0, w) * (1 - seg(u, t1, t1 + 0.06)), w=50.0, h=50.0)
            else:
                self.node(sc, ln, yb, "LN", NORM, op, smoothstep(0.15, 0.3, w) * (1 - seg(
                    w, 0.35, 0.5)), w=50.0, h=50.0)
        # the stream's vector for one character, where it is measured
        for x, key, t in ((x_in, "x", 0.04), (mid, "mid", 0.47), (x_out, "out", 0.85)):
            a = op * (0.3 + 0.7 * smoothstep(t, t + 0.06, u))
            v = f[key][self.at]
            self.strip(sc, v, x, ys, a, cell=cell, wide=26.0, vmax=self.vmax, ink=ink)
            self.text(sc, x, ys + (cell * len(v) / 2 + 22.0) / 540.0, "length %.1f" % size(v),
                      a, strong=True, size=0.034)
        self.text(sc, x_in - 0.045, self.YB - 0.085, "the stream for the character  %s" %
                  self.chars[self.at], op, anchor="lm", size=0.032)
        self.text(sc, 0.5, 0.90, caption, op, strong=True, size=0.034)


    # -- results
    def race(self, sc, u, op, steps, traces, names, inks, box=((0.10, 0.56), (0.52, 0.84)),
             at=(0.66, 0.58), upto=0.85):
        """The models training side by side, with the loss each has reached."""
        chart = self.chart(box, steps, traces)
        self.axes(sc, chart, op, "training update", "loss (nats)")
        for trace, ink in zip(traces, inks):
            self.plot(sc, chart, steps, trace, ink, op, u / upto)
        now = int(min(u / upto, 1.0) * (len(steps) - 1))
        self.lines(sc, at[0], at[1], [("update %d" % steps[now], None)] + [
            ("%s  %.2f" % (name, trace[now]), dark(ink))
            for name, trace, ink in zip(names, traces, inks)], op, step=0.06, size=0.036)
        return chart

    def result(self, sc, u, op, values, names, inks, fx=0.26, fy=0.56, wide=0.40,
               title="loss on held-out text (lower is better)", fmt="%.2f nats", step=0.10,
               thick=34.0, upto=0.5):
        """The held-out measurement as bars on one scale."""
        full = self.span(wide)[0]
        worst = max(values)
        self.text(sc, fx, fy - 0.075, title, op, anchor="lm", strong=True)
        for row, (name, ink, value) in enumerate(zip(names, inks, values)):
            self.hbar(sc, fx, fy + row * step, full * value / worst, ink, op, name, fmt % value,
                      u / upto, thick=thick, track=full)
        return fy + len(values) * step


    def seed_dots(self, sc, u, op, held, names, inks, fx=0.26, fy=0.55, wide=0.42, step=0.115,
                  title="held-out loss of every run (nats, lower is better)", upto=0.6,
                  fmt="%.3f"):
        """Every run of every variant as a dot on one scale, zoomed to the
        runs' own range, with each variant's mean marked: ``held`` is
        ``(variants, seeds)``. Returns ``value -> layout x``."""
        held = np.asarray(held, float)
        lo, hi = float(held.min()), float(held.max())
        pad = max((hi - lo) * 0.12, 1e-4)
        lo, hi = lo - pad, hi + pad

        def at(v):
            return fx + wide * (float(v) - lo) / (hi - lo)
        self.text(sc, fx, fy - 0.085, title, op, anchor="lm", strong=True)
        base = fy + (len(held) - 1) * step + 0.065
        for v in nice_ticks(lo, hi, 4):
            if lo <= v <= hi:
                self.dashes(sc, (at(v), fy - 0.045), (at(v), base), AXIS, op, n=16, width=1.0)
                self.text(sc, at(v), base + 0.028, ("%.4f" % v).rstrip("0").rstrip("."), op)
        n = held.shape[1]
        for row, (name, ink, values) in enumerate(zip(names, inks, held)):
            y = fy + row * step
            self.polyline(sc, [self.P(fx, y), self.P(fx + wide, y)], 1.4, AXIS, op, bias=0.0)
            self.text(sc, fx - 0.015, y, name, op, anchor="rm", ink=dark(ink), size=0.036)
            for k, v in enumerate(values):
                a = smoothstep(0.0, 1.0, seg(u / upto, k / (n + 1.0), (k + 1.5) / (n + 1.0)))
                self.disk(sc, self.P(at(v), y), 7.5, ink, op * a * 0.85, lift=1.0 + 0.1 * k,
                          outline=0.0)
            a = op * smoothstep(upto, upto + 0.12, u)
            mean = float(values.mean())
            self.polyline(sc, [self.P(at(mean), y - 0.04), self.P(at(mean), y + 0.04)], 4.0,
                          dark(ink, 0.35), a, bias=0.6)
            self.text(sc, fx + wide + 0.015, y, "mean " + fmt % mean, a, anchor="lm",
                      strong=True)
        return at


# two-sided 95% points of Student's t, by degrees of freedom (seeds - 1)
T95 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def paired_gap(held):
    """Two variants trained from the same seeds, ``held`` ``(2, seeds)``:
    ``(mean gap second - first, largest gap between two seeds of one variant,
    seeds in which the second is better, whether the mean gap is beyond what
    the seeds' differences explain)``. The last is the paired t-test at 95%."""
    held = np.asarray(held, float)
    diff = held[1] - held[0]
    n = len(diff)
    error = float(diff.std(ddof=1) / np.sqrt(n))
    return (float(diff.mean()), float((held.max(1) - held.min(1)).max()), int((diff < 0).sum()),
            bool(abs(diff.mean()) > T95[n - 1] * error))


class Paired(Built):
    """A model built two ways and trained from the same seeds: the training
    of both, then every run's held-out loss and what the gap amounts to."""

    def measured(self):
        """Read the study's measurement (:func:`voxanim.study_block.paired`)."""
        s = self.study
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.by_seed = np.asarray(s.metadata["heldout_loss_by_seed"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.params = [int(v) for v in s.panels[3].data["values"]]
        self.updates = int(s.metadata["steps"])
        self.seeds = self.by_seed.shape[1]
        self.mean_gap, self.noise, self.wins, self.real = paired_gap(self.by_seed)

    def trained_subtitle(self, what):
        return ("%s (%s learned numbers), each trained %d times from different starting "
                "weights: %s updates on Tiny Shakespeare. The curves are the averages" % (
                    what, f"{self.params[0]:,}", self.seeds,
                    f"{self.updates:,}"))

    def gap_subtitle(self):
        a, b = self.NAMES
        better, wins = (b, self.wins) if self.mean_gap < 0 else (a, self.seeds - self.wins)
        head = "held-out text, mean of %d runs: %s %.3f nats, %s %.3f. " % (
            self.seeds, a, self.heldout[0], b, self.heldout[1])
        if self.real:
            return head + "%s is better in %d of the %d seeds, by %.3f on average: a real " \
                "difference at this size" % (better[0].upper() + better[1:], wins, self.seeds,
                                             abs(self.mean_gap))
        return head + "The %.3f gap is within the noise: two seeds of one model differ by up " \
            "to %.3f, and %s wins %d of %d" % (abs(self.mean_gap), self.noise, better, wins,
                                               self.seeds)

    def verdict(self, sc, u, op, fx=0.26, fy=0.845):
        a, b = self.NAMES
        better, wins = (b, self.wins) if self.mean_gap < 0 else (a, self.seeds - self.wins)
        self.lines(sc, fx, fy, [
            "gap between the means: %.3f nats (%.1f%% of the loss)" % (
                abs(self.mean_gap), 100 * abs(self.mean_gap) / self.heldout[0]),
            ("%s better in %d of %d seeds: %s" % (
                better, wins, self.seeds,
                "a real difference" if self.real else "within the noise between seeds"),
             dark(GREEN, 0.35) if self.real else dark(AMBER, 0.35))],
            op * smoothstep(0.7, 0.85, u), size=0.034, step=0.05)

    def phase_training(self, sc, u, op):
        self._recap(sc, op)
        self.race(sc, u, op, self.steps, self.traces, self.NAMES, self.COLOURS,
                  box=((0.10, 0.58), (0.50, 0.84)), at=(0.68, 0.58))

    def phase_heldout(self, sc, u, op):
        self._recap(sc, op)
        self.seed_dots(sc, u, op, self.by_seed, self.NAMES, self.COLOURS, fy=0.56, step=0.10)
        self.verdict(sc, u, op)


# ====================================================================== 13 pre-LN and post-LN
@cache
def depth_sizes():
    """The stream's size through the study's 12 untrained blocks, pre-LN and
    post-LN: ``(2, 13)``, the mean length of the stream vector entering block
    1 and leaving each block, on the windows and the (first seed's) initial
    weights the study measured its gradients with."""
    e = SB.experiment("pre-ln-vs-post-ln")
    x = e["train"][0][:e["batch"]]
    rows = []
    for cfg in e["configs"][3:5]:
        _logits, rec = T.forward(T.init(cfg, e["seeds"][0]), x, cfg, keep=True)
        rows.append([float(np.linalg.norm(r, axis=-1).mean()) for r in rec["residual"]])
    return np.array(rows)


@cache
def frequency_loss():
    """Held-out loss of guessing every next character by how common each
    character is in the training text (exact; counts plus one)."""
    count = np.bincount(R.encode(R.text("train")), minlength=len(R.vocabulary())) + 1.0
    targets = SB.experiment("pre-ln-vs-post-ln")["test"][1]
    return float(-np.log(count / count.sum())[targets].mean())


class PreLnVsPostLn(Built):
    """Where the normalisation sits: on a branch (pre-LN) or on the stream
    itself (post-LN); what that does to the stream's size and to the gradient
    through 12 blocks; then both trained at one fixed rate, and post-LN again
    with a warm-up, at 2 and at 12 blocks over three seeds."""

    TITLE = "Pre-LN and post-LN"
    PHASES = (("pre-LN", 9.0), ("post-LN", 9.0), ("forward", 8.0), ("gradient", 9.0),
              ("training", 20.0))
    NAMES = ("pre-LN", "post-LN", "post-LN + warm-up")
    COLOURS = (RED, BLUE, GREEN)
    YS, YB = 0.66, 0.40            # the stream, the branches
    STUCK = 0.05                   # nats from the letter-frequency loss that count as stuck

    def setup(self):
        s, m = self.study, self.study.metadata
        self.grads = np.asarray(s.panels[0].data["ys"], float)            # (2, 12)
        self.sizes = depth_sizes()                                        # (2, 13)
        self.blocks = self.grads.shape[1]
        self.depths = tuple(int(d) for d in m["depths"])                  # shallow, deep
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.stack([np.asarray(s.panels[2].data["ys"], float),
                                np.asarray(s.panels[1].data["ys"], float)])   # (depth, arm, t)
        self.by_seed = np.asarray(m["heldout_loss_by_seed"], float).reshape(2, 3, -1)
        self.heldout = self.by_seed.mean(-1)                              # (depth, arm)
        self.updates, self.seeds = int(m["steps"]), self.by_seed.shape[-1]
        self.warmup, self.rate = int(m["warmup_updates"][2]), float(m["learning_rate"])
        self.frequency = frequency_loss()
        configs, runs, _losses, _test = trained("pre-ln-vs-post-ln")
        self.ids, self.chars = test_window()
        self.at = 5
        self.flow = [block_trace(runs[i]["P"], self.ids, configs[i]) for i in (0, 1)]
        self.vmax = max(float(np.abs(f[k][self.at]).max()) for f in self.flow
                        for k in ("x", "mid", "out"))

    def lengths(self, row):
        """The stream's length at the three stations of the block drawn."""
        return tuple(size(self.flow[row][k][self.at]) for k in ("x", "mid", "out"))

    def stuck(self, depth, arm):
        """How many seeds of an arm ended within :attr:`STUCK` nats of the
        loss of guessing by letter frequency: they learned nothing more."""
        return int((self.by_seed[depth, arm] > self.frequency - self.STUCK).sum())

    def outcome(self, depth, arm):
        """What an arm did at a depth, in words, from its seeds."""
        stuck, runs = self.stuck(depth, arm), self.by_seed[depth, arm]
        if stuck == self.seeds:
            return "stuck at %.2f (letter frequencies)" % runs.mean()
        if stuck:
            return "trains in %d of %d seeds" % (self.seeds - stuck, self.seeds)
        return "%.2f nats" % runs.mean()

    def subtitle_for(self, k):
        g, z = self.grads, self.sizes
        if k == 4:
            if not any(self.stuck(0, arm) for arm in range(3)):
                shallow = "all three train"
            else:
                shallow = "; ".join(self.outcome(0, arm) for arm in range(3))
            return "%d updates, one fixed rate, %d seeds. %d blocks: %s. %d blocks: pre-LN " \
                "%s; post-LN %s; with warm-up, %s" % (
                    self.updates, self.seeds, self.depths[0], shallow, self.depths[1],
                    self.outcome(1, 0), self.outcome(1, 1), self.outcome(1, 2))
        return [
            "pre-LN: each branch normalises its own copy and adds its result back; the stream "
            "itself is never rescaled (its length here goes %.1f, %.1f, %.1f)" % self.lengths(0),
            "post-LN: the normalisation sits on the stream, after each addition; everything "
            "passing is rescaled, every block (lengths %.1f, %.1f, %.1f)" % self.lengths(1),
            "%d untrained blocks: the pre-LN stream grows from %.1f to %.1f as branches keep "
            "adding; post-LN resets it to %.1f after every block" % (
                self.blocks, z[0, 0], z[0, -1], z[1, -1]),
            "the gradient coming back, measured at each block before any update: pre-LN "
            "between %.2g and %.2g; post-LN between %.2g and %.2g. At block 1 post-LN's is "
            "%.0f times %s than pre-LN's" % (
                g[0].min(), g[0].max(), g[1].min(), g[1].max(),
                max(g[0, 0] / g[1, 0], g[1, 0] / g[0, 0]),
                "smaller" if g[1, 0] < g[0, 0] else "larger"),
        ][k]

    def phase_1(self, sc, u, op):
        self.block(sc, u, op, self.flow[0], RED, False,
                   "LN on the branches: the stream runs straight through", cell=8.5)

    def phase_2(self, sc, u, op):
        self.block(sc, u, op, self.flow[1], BLUE, True,
                   "LN on the stream: what passes is rescaled each time", cell=8.5)

    # ------------------------------------------------------------------ the deep model
    X0, X1, YROW = 0.13, 0.86, 0.845

    def _tower(self, sc, op, lit, back):
        """The blocks in a row, the loss at the far end; ``lit``: how many
        have been reached (from the left, or from the right when ``back``)."""
        n = self.blocks
        pitch = (self.X1 - self.X0) / n
        xs = [self.X0 + (i + 0.5) * pitch for i in range(n)]
        for i, x in enumerate(xs):
            on = (n - 1 - i < lit) if back else (i < lit)
            self.box(sc, x, self.YROW, 44.0, 30.0, soft(STREAM, 0.25 if on else 0.72), op,
                     radius=4)
            self.text(sc, x, self.YROW + 0.058, "%d" % (i + 1), op)
        self.text(sc, self.X0 - 0.012, self.YROW + 0.058, "block", op, anchor="rm")
        self.box(sc, 0.905, self.YROW, 56.0, 30.0, soft(GREEN, 0.3), op, "loss",
                 ink=dark(GREEN, 0.5), border=GREEN)
        return xs, pitch

    def phase_3(self, sc, u, op):
        n = self.blocks
        reach = seg(u, 0.08, 0.80) * n
        xs, pitch = self._tower(sc, op, int(reach), back=False)
        top, base = 0.33, self.YROW - 0.06
        unit = (base - top) / (self.sizes.max() * 1.05)
        for v in (10, 20, 30, 40):
            if v < self.sizes.max():
                y = base - v * unit
                self.dashes(sc, (self.X0, y), (self.X1, y), AXIS, op, n=40, width=1.0)
                self.text(sc, self.X0 - 0.012, y, "%d" % v, op, anchor="rm")
        self.text(sc, self.X0, top - 0.025, "length of the stream leaving each block", op,
                  anchor="lm", strong=True)
        for row, ink in enumerate(self.COLOURS[:2]):
            for i, x in enumerate(xs):
                grow = seg(reach, i, i + 1)
                if grow <= 0:
                    continue
                dx = (row - 0.5) * pitch * 0.42
                h = self.sizes[row, i + 1] * unit * grow
                self.rect(sc, self.P(x + dx, base - h / 2), self.span(pitch * 0.36)[0],
                          max(self.span(0, h)[1], 1.0), ink, op, radius=2)
        if 0 < reach < n:
            at = self.X0 + reach * pitch
            self.disk(sc, self.P(at, self.YROW), 7.0, STREAM, op, glow=0.7, lift=1.8, outline=0.0)
        k = min(int(reach), n)
        self.lines(sc, 0.905, 0.42, [
            ("block %d" % max(k, 1), None),
            ("pre-LN", dark(RED)), ("%.1f" % self.sizes[0, k], dark(RED)),
            ("post-LN", dark(BLUE)), ("%.1f" % self.sizes[1, k], dark(BLUE))], op, step=0.05,
            size=0.032, anchor="mm")

    def phase_4(self, sc, u, op):
        n = self.blocks
        reach = seg(u, 0.08, 0.80) * n
        xs, pitch = self._tower(sc, op, int(reach), back=True)
        top, base = 0.35, self.YROW - 0.06
        decades = np.arange(np.floor(np.log10(self.grads.min())),
                            np.ceil(np.log10(self.grads.max())) + 1)
        lo, hi = decades[0] - 0.15, decades[-1]             # log10 of the gradient's size

        def height(g):
            return (np.log10(g) - lo) / (hi - lo) * (base - top)
        for d in decades:
            y = base - height(10.0 ** d)
            self.dashes(sc, (self.X0, y), (self.X1, y), AXIS, op, n=40, width=1.0)
            self.text(sc, self.X0 - 0.012, y, number(10.0 ** d), op, anchor="rm")
        self.text(sc, self.X0, top - 0.045, "size of the gradient reaching each block "
                  "(each line is ten times the one below)", op, anchor="lm", strong=True)
        for row, ink in enumerate(self.COLOURS[:2]):
            for i, x in enumerate(xs):
                grow = seg(reach, n - 1 - i, n - i)
                if grow <= 0:
                    continue
                dx = (row - 0.5) * pitch * 0.42
                h = height(self.grads[row, i]) * grow
                self.rect(sc, self.P(x + dx, base - h / 2), self.span(pitch * 0.36)[0],
                          max(self.span(0, h)[1], 1.0), ink, op, radius=2)
        if 0 < reach < n:
            at = self.X1 - reach * pitch
            self.disk(sc, self.P(at, self.YROW), 7.0, GREEN, op, glow=0.7, lift=1.8, outline=0.0)
        k = int(np.clip(n - 1 - int(reach), 0, n - 1))
        self.lines(sc, 0.905, 0.42, [
            ("block %d" % (k + 1), None),
            ("pre-LN", dark(RED)), ("%.2g" % self.grads[0, k], dark(RED)),
            ("post-LN", dark(BLUE)), ("%.2g" % self.grads[1, k], dark(BLUE))], op, step=0.05,
            size=0.032, anchor="mm")

    # ------------------------------------------------------------------ training
    def phase_5(self, sc, u, op):
        half = min(int(u * 2.0), 1)                          # the shallow model, then the deep
        w = u * 2.0 - half
        chart = self.chart(((0.10, 0.52), (0.42, 0.84)), self.steps,
                           np.append(self.traces.ravel(), self.frequency))
        self.axes(sc, chart, op, "training update", "loss (nats)")
        y = chart.fy(self.frequency)
        self.dashes(sc, (chart.x0, y), (chart.x1, y), STREAM, op * 1.6, n=30, width=1.6)
        above = [v for v in nice_ticks(*chart.y_range, 4) if v > self.frequency][:2]
        if len(above) == 2:                               # between the grid lines above it
            y = (chart.fy(above[0]) + chart.fy(above[1])) / 2
        self.text(sc, chart.x1, y, "dashed: guessing by letter frequency, %.2f" %
                  self.frequency, op, anchor="rm")
        shown = smoothstep(0.0, 0.08, w) * (1.0 - smoothstep(0.94, 1.0, w) * (half < 1))
        for arm, ink in enumerate(self.COLOURS):
            self.plot(sc, chart, self.steps, self.traces[half, arm], ink, op * shown, w / 0.7)
        # the model being trained: its blocks
        depth = self.depths[half]
        self.text(sc, 0.10, 0.325, "%d blocks" % depth, op * shown, anchor="lm", strong=True,
                  size=0.040)
        for i in range(depth):
            self.box(sc, 0.235 + i * 0.0165, 0.325, 12.0, 24.0, soft(STREAM, 0.35), op * shown,
                     radius=2)
        # held-out loss: a bar per arm (the seeds' mean) with each seed marked on it
        fx, full = 0.735, self.span(0.165)[0]
        worst = float(self.by_seed.max())
        self.text(sc, 0.63, 0.325, "held-out loss: mean, and each of %d seeds" % self.seeds, op,
                  anchor="lm", strong=True)
        for d in range(2):
            y0 = 0.405 + d * 0.235
            done = 1.0 if d < half else (seg(w, 0.7, 0.9) if d == half else 0.0)
            self.text(sc, 0.63, y0 + 0.058, "%d blocks" % self.depths[d],
                      op * (0.35 + 0.65 * (done > 0)), anchor="lm")
            for arm, ink in enumerate(self.COLOURS):
                yy = y0 + arm * 0.058
                value = self.heldout[d, arm]
                self.hbar(sc, fx, yy, full * value / worst, soft(ink, 0.3), op * (done > 0), "",
                          "", done, thick=22.0, track=full)
                a = op * smoothstep(0.9, 1.0, done)
                for k, v in enumerate(self.by_seed[d, arm]):
                    x = fx + self.frac(full * v / worst)[0]
                    self.disk(sc, self.P(x, yy), 4.5, dark(ink, 0.5), a, lift=1.2 + 0.1 * k,
                              outline=0.0)
                self.text(sc, fx + self.frac(full)[0] + 0.012, yy, "%.2f" % value, a,
                          anchor="lm", strong=True)
        for arm, (name, x) in enumerate(zip(self.NAMES, (0.60, 0.69, 0.785))):
            self.text(sc, x, 0.895, name, op, anchor="lm", ink=dark(self.COLOURS[arm]),
                      size=0.034)


# ====================================================================== 14 RMSNorm and LayerNorm
def layer_norm(v):
    """LayerNorm's arithmetic on one vector (before its learned scale and
    shift): ``(mean, spread, result)``."""
    v = np.asarray(v, float)
    mean = float(v.mean())
    spread = float(np.sqrt(np.mean((v - mean) ** 2) + 1e-5))
    return mean, spread, (v - mean) / spread


def rms_norm(v):
    """RMSNorm's arithmetic on one vector: ``(size, result)``."""
    v = np.asarray(v, float)
    scale = float(np.sqrt(np.mean(v ** 2) + 1e-5))
    return scale, v / scale


class RmsNormVsLayerNorm(Paired):
    """The two normalisations applied to one real vector, bar by bar:
    LayerNorm subtracts the mean and divides by the spread, RMSNorm only
    divides by the size; then a model trained with each, over eight seeds."""

    TITLE = "RMSNorm and LayerNorm"
    PHASES = (("a vector", 8.0), ("LayerNorm", 13.0), ("RMSNorm", 12.0), ("training", 10.0),
              ("held out", 10.0))
    NAMES = ("LayerNorm", "RMSNorm")
    COLOURS = (RED, BLUE)
    BARS = (0.12, 0.60, 0.66)              # x0, x1, the zero line
    AT = 2

    def setup(self):
        configs, runs, _losses, _test = trained("rmsnorm-vs-layernorm")
        self.ids, self.chars = test_window()
        self.v = block_trace(runs[0]["P"], self.ids, configs[0])["out"][self.AT]
        self.mean, self.spread, self.ln = layer_norm(self.v)
        self.scale, self.rn = rms_norm(self.v)
        self.gap = float(np.abs(self.ln - self.rn).max())
        self.unit = 0.255 / max(float(np.abs(a).max()) for a in (self.v, self.ln, self.rn,
                                                                 self.v - self.mean))
        self.measured()

    def subtitle_for(self, k):
        n = len(self.v)
        return [
            "inside a trained model each character is a list of %d numbers; this is the one "
            "for %s. Its mean is %.2f and its size (root mean square) %.2f" % (
                n, self.chars[self.AT], self.mean, self.scale),
            "LayerNorm: subtract the mean, %.2f, from every number, then divide by their "
            "spread, %.2f. The result has mean 0 and spread 1" % (self.mean, self.spread),
            "RMSNorm drops the mean: one step, divide by the size %.2f. The result keeps a "
            "mean of %.2f and differs from LayerNorm's by at most %.2f" % (
                self.scale, float(self.rn.mean()), self.gap),
            self.trained_subtitle("the same model with LayerNorm and with RMSNorm"),
            self.gap_subtitle(),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _stage(self, sc, op, values, ink, mean=None, unit_lines=False, ghost=None):
        """The vector as bars about its zero line; ``mean``: a dashed line
        there; ``unit_lines``: guides at +1 and -1; ``ghost``: values marked
        with a tick on each bar's column (the other method's result)."""
        x0, x1, y0 = self.BARS
        self.polyline(sc, [self.P(x0 - 0.01, y0), self.P(x1 + 0.01, y0)], 1.4, AXIS, op,
                      bias=0.0)
        self.text(sc, x0 - 0.02, y0, "0", op, anchor="rm")
        if unit_lines:
            for v in (1, -1):
                y = y0 - v * self.unit
                self.dashes(sc, (x0 - 0.01, y), (x1 + 0.01, y), AXIS, op, n=30, width=1.0)
                self.text(sc, x0 - 0.02, y, "%+d" % v, op, anchor="rm")
        xs = self.vbars(sc, values, x0, x1, y0, self.unit, ink, op)
        if ghost is not None:
            for x, g in zip(xs, ghost[0]):
                y = y0 - g * self.unit
                self.polyline(sc, [self.P(x - 0.011, y), self.P(x + 0.011, y)], 3.0, ghost[1],
                              op * ghost[2], bias=0.4)
        if mean is not None:
            y = y0 - mean * self.unit
            self.dashes(sc, (x0 - 0.01, y), (x1 + 0.035, y), NORM, op * 2.0, n=30, width=2.2)
            self.text(sc, x1 + 0.045, y, "mean", op, anchor="lm", ink=dark(NORM, 0.35))
        return xs

    def _window(self, sc, op):
        x0, x1, _ = self.BARS
        shown = ["" if i == self.AT else ch for i, ch in enumerate(self.chars)]
        xs = self.tokens(sc, shown, x0, x1, 0.325, op, w=38.0, h=38.0, size=0.036)
        self.box(sc, xs[self.AT], 0.325, 38, 38, STREAM, op, self.chars[self.AT], ink=WHITE,
                 size=0.036)
        return xs

    def _steps(self, sc, op, rows, lit):
        """The method's steps as a list at the right, the one under way lit."""
        y = 0.47
        for i, (head, detail) in enumerate(rows):
            on = 1.0 if i == lit else 0.45
            self.text(sc, 0.70, y, head, op * on, anchor="lm", strong=True, size=0.036)
            self.text(sc, 0.70, y + 0.05, detail, op * on, anchor="lm", size=0.032)
            y += 0.135

    def phase_1(self, sc, u, op):
        _x0, _x1, _y0 = self.BARS
        xs = self._window(sc, op)
        grow = np.array([seg(u, 0.12 + 0.02 * i, 0.30 + 0.02 * i) for i in range(len(self.v))])
        self.arrow(sc, (xs[self.AT], 0.365), (xs[self.AT], 0.405), STREAM, op, seg(u, 0.05, 0.15))
        self._stage(sc, op, self.v * grow, STREAM,
                    mean=self.mean if u > 0.62 else None)
        self.text(sc, 0.70, 0.325, "its %d numbers" % len(self.v), op, anchor="lm", strong=True,
                  size=0.036)
        a = op * smoothstep(0.62, 0.72, u)
        self.lines(sc, 0.70, 0.52, [("mean  %.2f" % self.mean, dark(NORM, 0.35))], a, size=0.036)
        self.lines(sc, 0.70, 0.60, ["size  %.2f" % self.scale,
                                    "(root of the mean square)"],
                   op * smoothstep(0.78, 0.88, u), size=0.036, step=0.048)

    def phase_2(self, sc, u, op):
        self._window(sc, op)
        a, b = smoothstep(0.10, 0.42, u), smoothstep(0.52, 0.84, u)
        shifted = self.v - self.mean * a
        values = shifted + (self.ln - (self.v - self.mean)) * b
        self._stage(sc, op, values, RED, mean=self.mean * (1 - a) if b <= 0 else None,
                    unit_lines=True)
        self.text(sc, 0.70, 0.325, "LayerNorm", op, anchor="lm", ink=dark(RED), size=0.040)
        self._steps(sc, op, [
            ("1  subtract the mean", "every bar moves up by %.2f" % -self.mean),
            ("2  divide by the spread", "every bar is %.2f times shorter" % self.spread),
            ("mean 0, spread 1", "whatever came in")], 0 if u < 0.47 else 1 if u < 0.86 else 2)

    def phase_3(self, sc, u, op):
        self._window(sc, op)
        b = smoothstep(0.12, 0.48, u)
        values = self.v + (self.rn - self.v) * b
        self._stage(sc, op, values, BLUE, unit_lines=True,
                    ghost=(self.ln, RED, smoothstep(0.55, 0.65, u)))
        self.text(sc, 0.70, 0.325, "RMSNorm", op, anchor="lm", ink=dark(BLUE), size=0.040)
        n = len(self.v)
        self._steps(sc, op, [
            ("1  divide by the size", "every bar is %.2f times shorter" % self.scale),
            ("red marks: LayerNorm", "largest gap %.2f" % self.gap),
            ("the work saved (exact)", "%d additions, %d subtractions" % (n, n))],
            0 if u < 0.52 else 1 if u < 0.78 else 2)

    def _recap(self, sc, op):
        """Both results, small, above the measurement."""
        for row, (values, ink, name) in enumerate(zip((self.ln, self.rn), self.COLOURS,
                                                      self.NAMES)):
            x0 = 0.10 + row * 0.30
            self.vbars(sc, values, x0, x0 + 0.17, 0.355, 0.022, ink, op)
            self.text(sc, x0 + 0.18, 0.355, name, op, anchor="lm", ink=dark(ink), size=0.034)

    def phase_4(self, sc, u, op):
        self.phase_training(sc, u, op)

    def phase_5(self, sc, u, op):
        self.phase_heldout(sc, u, op)


# ====================================================================== 18 the parallel block
class ParallelBlock(Paired):
    """Attention then MLP, against attention beside MLP: the two dataflows
    with one character's real vectors on them, what each takes in steps, and
    the two trained on the same text over eight seeds."""

    TITLE = "Sequential and parallel blocks"
    PHASES = (("sequential", 12.0), ("parallel", 13.0), ("training", 11.0), ("held out", 10.0))
    NAMES = ("sequential", "parallel")
    COLOURS = (RED, BLUE)
    YS, YB = 0.66, 0.40
    AT = 5

    def setup(self):
        configs, runs, _losses, _test = trained("parallel-block")
        self.ids, self.chars = test_window()
        self.at = self.AT
        self.flow = [block_trace(run["P"], self.ids, c) for run, c in zip(runs, configs)]
        self.vmax = max(float(np.abs(f[k][self.at]).max()) for f in self.flow
                        for k in ("x", "mid", "out"))
        self.measured()

    def lengths(self, row):
        return tuple(size(self.flow[row][k][self.at]) for k in ("x", "mid", "out"))

    def subtitle_for(self, k):
        return [
            "sequential: attention writes into the stream first (its length goes %.1f to %.1f); "
            "only then does the MLP read it and write too (%.1f). Two steps" % self.lengths(0),
            "parallel: attention and the MLP both read the same incoming stream and both add "
            "to it (length %.1f to %.1f). Neither waits for the other: one step" % (
                self.lengths(1)[0], self.lengths(1)[2]),
            self.trained_subtitle("the same model built both ways"),
            self.gap_subtitle(),
        ][k]

    def phase_1(self, sc, u, op):
        self.block(sc, u, op, self.flow[0], RED, False,
                   "the MLP waits: it reads what attention has written", cell=8.5)
        for i, (name, ink, t) in enumerate((("step 1", ATTN, 0.12), ("step 2", MLP, 0.50))):
            self.text(sc, (0.355, 0.755)[i], self.YB + 0.08, name,
                      op * (0.4 + 0.6 * smoothstep(t, t + 0.05, u)), ink=dark(ink, 0.35),
                      size=0.034)

    def phase_2(self, sc, u, op):
        f = self.flow[1]
        ys, up, down = 0.61, 0.40, 0.82
        x_in, tap, ln, sub, plus, x_out = 0.10, 0.185, 0.275, 0.40, 0.515, 0.60
        wide = {"attention": 124.0, "MLP": 84.0}
        boxes = [(ln, up, 50.0, 50.0), (ln, down, 50.0, 50.0), (sub, up, wide["attention"], 50.0),
                 (sub, down, wide["MLP"], 50.0), (plus, ys, 34.0, 34.0), (x_in, ys, 30.0, 8),
                 (x_out, ys, 30.0, 8)]
        self.wire(sc, [(x_in, ys), (x_out + 0.045, ys)], STREAM, op, seg(u, 0.0, 0.08), 5.0,
                  boxes=boxes)
        w = seg(u, 0.12, 0.52)
        inside = 1.0 - abs(seg(w, 0.10, 0.90) - 0.5) * 2
        for y, name, colour in ((up, "attention", ATTN), (down, "MLP", MLP)):
            path = [(tap, ys), (tap, y), (plus, y), (plus, ys)]
            self.wire(sc, path, colour, op * smoothstep(0.0, 0.08, u), 1.0, 2.6, boxes=boxes)
            self.ride(sc, path, seg(w, 0.10, 0.90), colour, op)
            self.node(sc, ln, y, "LN", NORM, op, smoothstep(0.15, 0.3, w) * (1 - seg(w, 0.35, 0.5)),
                      w=50.0, h=50.0)
            self.node(sc, sub, y, name, colour, op, inside if w < 1 else 0.0, w=wide[name], h=50.0)
        self.ride(sc, [(tap - 0.05, ys), (plus, ys)], seg(w, 0.0, 0.90), STREAM, op)
        self.plus(sc, plus, ys, op, smoothstep(0.85, 0.95, w) * (1 - smoothstep(0.97, 1.0, w)))
        for x, key, t in ((x_in, "x", 0.04), (x_out, "out", 0.52)):
            a = op * (0.3 + 0.7 * smoothstep(t, t + 0.06, u))
            v = f[key][self.at]
            self.strip(sc, v, x, ys, a, cell=8.5, wide=26.0, vmax=self.vmax, ink=BLUE)
            self.text(sc, x, ys + 0.168, "length %.1f" % size(v), a, strong=True, size=0.034)
        self.text(sc, x_in - 0.045, 0.315, "the stream for the character  %s" %
                  self.chars[self.at], op, anchor="lm", size=0.032)
        # the steps each layout takes, drawn as a timetable
        a = op * smoothstep(0.58, 0.68, u)
        x0, cw = 0.705, 0.085
        self.text(sc, x0, 0.36, "time taken (exact)", a, anchor="lm", strong=True, size=0.034)
        self.text(sc, x0, 0.43, "sequential: 2 steps", a, anchor="lm", ink=dark(RED))
        self.node(sc, x0 + cw / 2, 0.495, "attn", ATTN, a, 1.0, w=76.0, h=36.0, size=0.032)
        self.node(sc, x0 + cw * 1.5 + 0.004, 0.495, "MLP", MLP, a, 1.0, w=76.0, h=36.0,
                  size=0.032)
        b = op * smoothstep(0.72, 0.82, u)
        self.text(sc, x0, 0.60, "parallel: 1 step", b, anchor="lm", ink=dark(BLUE))
        self.node(sc, x0 + cw / 2, 0.665, "attn", ATTN, b, 1.0, w=76.0, h=36.0, size=0.032)
        self.node(sc, x0 + cw / 2, 0.74, "MLP", MLP, b, 1.0, w=76.0, h=36.0, size=0.032)
        self.lines(sc, x0, 0.83, ["with hardware that can", "run both at once"], b, step=0.042)

    def _recap(self, sc, op):
        """The two layouts, small."""
        for row, (name, ink) in enumerate(zip(self.NAMES, self.COLOURS)):
            x = 0.10 + row * 0.42
            self.text(sc, x, 0.345, name, op, anchor="lm", ink=dark(ink), size=0.036)
            x += 0.145
            if row == 0:
                self.node(sc, x + 0.035, 0.345, "attn", ATTN, op, 0.0, w=64.0, h=32.0, size=0.030)
                self.node(sc, x + 0.115, 0.345, "MLP", MLP, op, 0.0, w=64.0, h=32.0, size=0.030)
                self.arrow(sc, (x + 0.071, 0.345), (x + 0.079, 0.345), STREAM, op, 1.0, 2.0)
            else:
                self.node(sc, x + 0.035, 0.31, "attn", ATTN, op, 0.0, w=64.0, h=30.0, size=0.030)
                self.node(sc, x + 0.035, 0.38, "MLP", MLP, op, 0.0, w=64.0, h=30.0, size=0.030)

    def phase_3(self, sc, u, op):
        self.phase_training(sc, u, op)

    def phase_4(self, sc, u, op):
        self.phase_heldout(sc, u, op)


# ====================================================================== 15 the gated MLP
class SwigluMlp(Paired):
    """The MLP's hidden units for one real character, unit by unit: through
    GELU, and through SwiGLU, where a second projection of the same input
    multiplies each unit. Then the weights of both at equal size, and both
    trained over eight seeds."""

    TITLE = "A gated MLP: SwiGLU"
    PHASES = (("GELU", 10.0), ("the gate", 14.0), ("same size", 9.0), ("training", 10.0),
              ("held out", 10.0))
    NAMES = ("GELU", "SwiGLU")
    COLOURS = (RED, BLUE)
    GATE = GREEN
    X0, X1 = 0.25, 0.93
    AT = 5

    def setup(self):
        self.configs, runs, _losses, _test = trained("swiglu-mlp")
        self.ids, self.chars = test_window()
        self.flow = [block_trace(run["P"], self.ids, c) for run, c in zip(runs, self.configs)]
        g, w = self.flow[0], self.flow[1]
        self.pre, self.after = g["pre"][self.AT], g["hidden"][self.AT]
        self.silu, self.gate, self.product = (w[k][self.AT] for k in ("silu", "gate", "hidden"))
        self.weights = [[runs[0]["P"]["l0.W1"], runs[0]["P"]["l0.W2"]],
                        [runs[1]["P"]["l0.W1"], runs[1]["P"]["l0.gate"], runs[1]["P"]["l0.W2"]]]
        self.mlp_params = [sum(int(m.size) for m in ws) for ws in self.weights]
        # the unit the SwiGLU sweep stops on: the one the gate changes most
        self.star = int(np.argmax(np.abs(self.product - self.silu)))
        self.measured()

    def subtitle_for(self, k):
        j = self.star
        return [
            "the usual MLP: the character's vector is projected onto %d hidden units, and "
            "GELU passes the positive ones and squashes the negative ones towards zero" % len(
                self.pre),
            "SwiGLU has %d units and a second projection of the same input, the gate: each "
            "unit is multiplied by its own gate value (unit %d: %.2f x %.2f = %.2f)" % (
                len(self.silu), j + 1, self.silu[j], self.gate[j], self.product[j]),
            "the gate is a third weight matrix, so SwiGLU gets %d units where GELU has %d: "
            "both MLPs hold exactly %s weights" % (
                len(self.silu), len(self.pre), f"{self.mlp_params[0]:,}"),
            self.trained_subtitle("the same model with each MLP"),
            self.gap_subtitle(),
        ][k]

    # ------------------------------------------------------------------ rows of units
    def _row(self, sc, op, values, y, unit, ink, name, detail, shown=None):
        """One row of hidden units as bars about a zero line, named at the left."""
        values = np.asarray(values, float)
        if shown is not None:
            values = values * shown
        self.polyline(sc, [self.P(self.X0 - 0.008, y), self.P(self.X1 + 0.008, y)], 1.2, AXIS,
                      op, bias=0.0)
        xs = self.vbars(sc, values, self.X0, self.X1, y, unit, ink, op, fill=0.68)
        self.text(sc, self.X0 - 0.025, y - 0.022, name, op, anchor="rm", ink=dark(ink),
                  size=0.036)
        self.text(sc, self.X0 - 0.025, y + 0.024, detail, op, anchor="rm")
        return xs

    def _cursor(self, sc, op, x, top, bottom, n):
        """The unit being computed: a band over its column."""
        w = self.span((self.X1 - self.X0) / n)[0]
        self.rect(sc, self.P(x, (top + bottom) / 2), w + 3, self.span(0, bottom - top)[1], AMBER,
                  op * 0.9, border=AMBER, border_w=2.0, radius=3, lift=0.2, fill_op=0.16)

    def _input(self, sc, op, ink):
        """Where the units come from: the character, and its vector."""
        x0 = 0.06
        shown = ["" if i == self.AT else ch for i, ch in enumerate(self.chars)]
        xs = self.tokens(sc, shown, x0, x0 + 0.40, 0.315, op, w=32.0, h=32.0, size=0.034)
        self.box(sc, xs[self.AT], 0.315, 32, 32, STREAM, op, self.chars[self.AT], ink=WHITE,
                 size=0.034)
        v = self.flow[0 if ink is RED else 1]["reads"][self.AT]
        self.arrow(sc, (0.47, 0.315), (0.515, 0.315), STREAM, op, 1.0, 2.4)
        self.grid(sc, v[None, :], 0.655, 0.315, 250.0, 22.0, op)
        self.text(sc, 0.805, 0.315, "its %d numbers" % len(v), op, anchor="lm")

    def phase_1(self, sc, u, op):
        n = len(self.pre)
        self._input(sc, op, RED)
        unit = 0.085 / float(np.abs(self.pre).max())
        sweep = seg(u, 0.22, 0.86) * n
        shown = np.array([seg(sweep, j, j + 1) for j in range(n)])
        xs = self._row(sc, op, self.pre, 0.50, unit, soft(RED, 0.35), "x W", "%d hidden units" % n,
                       np.array([seg(u, 0.04 + 0.003 * j, 0.16 + 0.003 * j) for j in range(n)]))
        self._row(sc, op, self.after, 0.76, unit, RED, "GELU", "what passes", shown)
        j = int(min(sweep, n - 1))
        if 0 < sweep < n:
            self._cursor(sc, op, xs[j], 0.40, 0.86, n)
        a = op * smoothstep(0.22, 0.28, u)
        self.text(sc, (self.X0 + self.X1) / 2, 0.905, "unit %d:  %.2f  becomes  %.2f" % (
            j + 1, self.pre[j], self.after[j]), a, strong=True, size=0.036)

    def phase_2(self, sc, u, op):
        n = len(self.silu)
        self._input(sc, op, BLUE)
        unit = 0.062 / float(max(np.abs(self.silu).max(), np.abs(self.gate).max(),
                                 np.abs(self.product).max()))
        sweep = seg(u, 0.26, 0.74) * n
        if u >= 0.74:                                     # then back to the unit it changed most
            sweep = n + 1
        shown = np.array([seg(sweep, j, j + 1) for j in range(n)])
        rise = [np.array([seg(u, t + 0.004 * j, t + 0.10 + 0.004 * j) for j in range(n)])
                for t in (0.03, 0.12)]
        xs = self._row(sc, op, self.silu, 0.45, unit, soft(BLUE, 0.35), "SiLU(x W)",
                       "%d hidden units" % n, rise[0])
        self._row(sc, op, self.gate, 0.615, unit, self.GATE, "gate  x V", "one per unit", rise[1])
        self._row(sc, op, self.product, 0.80, unit, BLUE, "product", "what passes", shown)
        j = int(min(sweep, n - 1)) if sweep <= n else self.star
        if sweep > 0:
            self._cursor(sc, op, xs[j], 0.375, 0.87, n)
        a = op * smoothstep(0.26, 0.32, u)
        self.text(sc, (self.X0 + self.X1) / 2, 0.91, "unit %d:  %.2f  x  %.2f  =  %.2f" % (
            j + 1, self.silu[j], self.gate[j], self.product[j]), a, strong=True, size=0.036)

    # ------------------------------------------------------------------ equal weights
    def phase_3(self, sc, u, op):
        px = 4.6                                          # screen px per weight, both models
        layouts = ((0.08, ("W", "W2"), RED, "GELU: %d units" % len(self.pre)),
                   (0.54, ("W", "V, the gate", "W2"), BLUE, "SwiGLU: %d units" % len(self.silu)))
        for row, (x0, names, ink, title) in enumerate(layouts):
            self.text(sc, x0, 0.30, title, op, anchor="lm", ink=dark(ink), size=0.038)
            total = 0
            for i, (w, name) in enumerate(zip(self.weights[row], names)):
                a = op * smoothstep(0.06 + 0.2 * i, 0.2 + 0.2 * i, u)
                m = w if w.shape[0] < w.shape[1] else w.T          # drawn wide: a column per unit
                wide, tall = m.shape[1] * px, m.shape[0] * px
                y = 0.435 + i * 0.18
                self.grid(sc, m, x0 + self.frac(wide / 2)[0], y, wide, tall, a, lines=0)
                x = x0 + self.frac(wide)[0] + 0.022
                self.text(sc, x, y - 0.03, name, a, anchor="lm", strong=True, size=0.036)
                self.text(sc, x, y + 0.02, "%d x %d = %d" % (w.shape + (w.size,)), a, anchor="lm")
                total += int(w.size) if a > 0.5 else 0
            if row == 0:
                self.text(sc, x0, 0.795, "no gate: those weights buy 16 more units",
                          op * smoothstep(0.46, 0.6, u), anchor="lm")
            self.text(sc, x0, 0.912, "%s weights" % f"{total:,}", op, anchor="lm",
                      strong=True, size=0.040)

    def _recap(self, sc, op):
        """What passes each MLP for the character, small."""
        for row, (values, ink, name) in enumerate(zip((self.after, self.product), self.COLOURS,
                                                      self.NAMES)):
            x0 = 0.10 + row * 0.40
            unit = 0.035 / float(np.abs(values).max())
            self.vbars(sc, values, x0, x0 + 0.24, 0.36, unit, ink, op)
            self.text(sc, x0 + 0.25, 0.345, name, op, anchor="lm", ink=dark(ink), size=0.034)

    def phase_4(self, sc, u, op):
        self.phase_training(sc, u, op)

    def phase_5(self, sc, u, op):
        self.phase_heldout(sc, u, op)


# ====================================================================== 17 weight tying
@cache
def tied_gradient():
    """The gradient of a tied embedding table, split by the end it arrives
    from: ``(whole, from the input end, from the output end, input ids)``,
    each ``(vocabulary, width)``, at the tied model's initial weights on the
    study's first 12 training windows. The halves come from the same model
    with the output projection held as a separate copy of the table."""
    tied = study_configs("weight-tying")[0][1]
    P = T.init(tied, 0)
    x, y, w = NA.text_task(n=12)
    _loss, whole = T.loss_grads(P, x, y, tied, w)
    copy = dict(P, U=P["E"].T.copy())
    _loss, halves = T.loss_grads(copy, x, y, replace(tied, tied=False), w)
    return whole["E"], halves["E"], halves["U"].T, x


def predict(P, ids, c, at):
    """What the model expects after position ``at`` of the window ``ids``:
    ``(character, probability, the final vector's scores for every character)``."""
    logits = T.forward(P, np.asarray(ids)[None], c)[0, at]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    k = int(p.argmax())
    return k, float(p[k]), logits


class WeightTying(Built):
    """The embedding table at both ends of the model: looked up by row on the
    way in, compared row by row on the way out. Untied these are two tables;
    tied they are one, whose gradient arrives from both ends."""

    TITLE = "Weight tying"
    PHASES = (("two tables", 11.0), ("one table", 11.0), ("both ends", 12.0), ("training", 10.0),
              ("held out", 10.0))
    NAMES = ("untied", "tied")
    COLOURS = (RED, BLUE)
    TABLE = (144.0, 264.0)             # px: 16 columns, one row per character
    Y = 0.61
    AT = 5

    def setup(self):
        s = self.study
        self.configs, runs, _losses, _test = trained("weight-tying")
        self.ids, self.chars = test_window()
        self.vocab = len(R.vocabulary())
        models = [run["P"] for run in runs]
        self.tables = [(models[0]["E"], models[0]["U"].T), (models[1]["E"], models[1]["E"])]
        self.said = [predict(P, self.ids, c, self.AT) for P, c in zip(models, self.configs)]
        self.saved_gpt2 = int(s.panels[0].data["values"][0])
        self.saved_bytes = int(s.panels[0].data["values"][1])
        self.whole, self.g_in, self.g_out, _x = tied_gradient()
        self.rows_in = int((np.abs(self.g_in).sum(1) > 0).sum())
        self.rows_out = int((np.abs(self.g_out).sum(1) > 0).sum())
        self.steps = np.asarray(s.panels[1].data["x"], float) + 1
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.params = [int(v) for v in s.panels[3].data["values"]]
        self.updates = int(s.metadata["steps"])

    def subtitle_for(self, k):
        cells = self.vocab * self.tables[0][0].shape[1]
        return [
            "a character enters by looking up its row in one table, and the answer leaves by "
            "scoring every row of another: two tables of %d x %d = %s numbers" % (
                self.vocab, self.tables[0][0].shape[1], f"{cells:,}"),
            "tied: the same table is used at both ends. This model drops from %s learned "
            "numbers to %s; GPT-2 saves %s x %s = %s the same way" % (
                f"{self.params[0]:,}", f"{self.params[1]:,}", "50,257", "768",
                f"{self.saved_gpt2:,}"),
            "training then corrects the one table from both ends: the input end touches the "
            "%d rows of characters just read, the output end all %d" % (
                self.rows_in, self.rows_out),
            "both models, %d updates on the same Tiny Shakespeare windows; the tied one has "
            "%s fewer numbers to learn" % (self.updates, f"{self.params[0] - self.params[1]:,}"),
            "on held-out text the untied model scores %.2f nats and the tied one %.2f: "
            "%.2f nats %s, with %d%% fewer learned numbers" % (
                self.heldout[0], self.heldout[1], abs(self.heldout[0] - self.heldout[1]),
                "better" if self.heldout[1] < self.heldout[0] else "worse",
                round(100 * (1 - self.params[1] / self.params[0]))),
        ][k]

    # ------------------------------------------------------------------ the pipeline
    def _table(self, sc, values, fx, op, row=None, ink=AMBER, vmax=None, lit=1.0):
        w, h = self.TABLE
        cell = self.grid(sc, values, fx, self.Y, w, h, op, vmax=vmax, lines=0)
        if row is not None and lit > 0.01:
            self.rect(sc, self.P(fx, cell(row, 0)[1]), w + 10, 9.0, ink, op * lit, border=ink,
                      border_w=2.4, radius=2, lift=0.8, fill_op=0.0)
        return cell

    def _pipeline(self, sc, u, op, which):
        """A character in, a prediction out, through the model ``which``."""
        ink = self.COLOURS[which]
        table_in, table_out = self.tables[which]
        k, prob, _scores = self.said[which]
        here = int(self.ids[self.AT])
        y = self.Y
        x_char, x_in, x_blocks, x_out, x_said = 0.085, 0.235, 0.49, 0.745, 0.895
        top = y - self.frac(0, self.TABLE[1] / 2)[1] - 0.045
        w = self.frac(self.TABLE[0] / 2 + 8)[0]
        # in
        self.box(sc, x_char, y, 46, 46, STREAM, op, self.chars[self.AT], ink=WHITE, size=0.040)
        cell = self._table(sc, table_in, x_in, op, here, lit=smoothstep(0.10, 0.18, u))
        row_y = cell(here, 0)[1]
        self.arrow(sc, (x_char + 0.03, y), (x_in - w - 0.006, row_y), STREAM, op,
                   seg(u, 0.03, 0.12), 2.6)
        self.text(sc, x_in, top, "table in", op, strong=True, size=0.034)
        # through the model
        a, b = (x_in + w + 0.004, row_y), (x_blocks - 0.062, y)
        self.arrow(sc, a, b, STREAM, op, seg(u, 0.18, 0.28), 2.6)
        self.pulse(sc, a, b, seg(u, 0.20, 0.34), ink, op, r=6.0)
        self.node(sc, x_blocks, y, "the model", STREAM, op,
                  smoothstep(0.30, 0.36, u) * (1 - smoothstep(0.44, 0.50, u)), w=110.0, h=60.0,
                  size=0.034)
        # out
        cell = self._table(sc, table_out, x_out, op, k, lit=smoothstep(0.66, 0.74, u))
        said_y = cell(k, 0)[1]
        a, b = (x_blocks + 0.062, y), (x_out - w - 0.006, y)
        self.arrow(sc, a, b, STREAM, op, seg(u, 0.44, 0.54), 2.6)
        self.pulse(sc, a, b, seg(u, 0.44, 0.58), ink, op, r=6.0)
        # every row is scored against the vector that arrives: a sweep down the table
        sweep = seg(u, 0.54, 0.68)
        if 0 < sweep < 1:
            yy = y + (sweep - 0.5) * self.frac(0, self.TABLE[1])[1]
            self.rect(sc, self.P(x_out, yy), self.TABLE[0] + 10, 7.0, ink, op, radius=2, lift=0.8,
                      fill_op=0.55)
        self.text(sc, x_out, top, "table out", op, strong=True, size=0.034)
        got = op * smoothstep(0.70, 0.78, u)
        self.arrow(sc, (x_out + w + 0.004, said_y), (x_said - 0.03, y), STREAM, got, 1.0, 2.6)
        self.box(sc, x_said, y, 46, 46, ink, got, show(R.decode([k])), ink=WHITE, size=0.040)
        self.text(sc, x_said, y + 0.075, "%d%%" % round(100 * prob), got, strong=True, size=0.034)
        self.text(sc, x_char, y + 0.075, "read", op)
        self.text(sc, x_said, y - 0.075, "next?", got)
        self.text(sc, x_said, y + 0.125, "text: %s" % self.chars[self.AT + 1], got)
        return x_in, x_out, top

    def phase_1(self, sc, u, op):
        x_in, x_out, _top = self._pipeline(sc, u, op, 0)
        cells = f"{self.tables[0][0].size:,}"
        for x in (x_in, x_out):
            self.text(sc, x, 0.905, "%s numbers" % cells, op * smoothstep(0.8, 0.9, u))

    def phase_2(self, sc, u, op):
        x_in, x_out, top = self._pipeline(sc, u * 1.25, op, 1)
        link = op * smoothstep(0.02, 0.12, u)
        self.arc(sc, (x_in + 0.05, top - 0.03), (x_out - 0.05, top - 0.03), BLUE, link, 1.0,
                 bend=0.035, width=3.0, head=False)
        self.text(sc, 0.49, top - 0.005, "one table, used twice", link, ink=dark(BLUE), size=0.036)
        a = op * smoothstep(0.80, 0.90, u)
        self.text(sc, 0.49, 0.80, "%s fewer" % f"{self.params[0] - self.params[1]:,}", a,
                  strong=True, size=0.036)
        self.text(sc, 0.49, 0.85, "numbers to learn", a)

    # ------------------------------------------------------------------ the gradient
    def phase_3(self, sc, u, op):
        vmax = float(max(np.abs(self.g_in).max(), np.abs(self.g_out).max()))
        top = self.Y - self.frac(0, self.TABLE[1] / 2)[1] - 0.045
        w = self.frac(self.TABLE[0] / 2 + 8)[0]
        xs = (0.20, 0.50, 0.80)
        a_in, a_out = smoothstep(0.06, 0.20, u), smoothstep(0.30, 0.44, u)
        merged = smoothstep(0.58, 0.74, u)
        self._table(sc, self.g_in * a_in, xs[0], op, vmax=vmax)
        self._table(sc, self.g_out * a_out, xs[2], op, vmax=vmax)
        self._table(sc, self.g_in * a_in * merged + self.g_out * a_out * merged, xs[1], op,
                    vmax=vmax)
        self.text(sc, xs[0], top, "from the input end", op, strong=True, size=0.034)
        self.text(sc, xs[2], top, "from the output end", op, strong=True, size=0.034)
        self.text(sc, xs[1], top, "the table's correction", op, ink=dark(BLUE), size=0.034)
        for x, side, t in ((xs[0], 1, 0.46), (xs[2], -1, 0.46)):
            a, b = (x + side * (w + 0.004), self.Y), (xs[1] - side * (w + 0.008), self.Y)
            self.arrow(sc, a, b, BLUE, op, seg(u, t, t + 0.1), 3.0)
            for k in range(3):
                self.pulse(sc, a, b, seg(u, t + 0.06 + 0.05 * k, t + 0.22 + 0.05 * k), BLUE, op,
                           r=6.0)
        self.text(sc, xs[0], 0.905, "%d of %d rows" % (self.rows_in, self.vocab), op * a_in,
                  size=0.034)
        self.text(sc, xs[2], 0.905, "%d of %d rows" % (self.rows_out, self.vocab), op * a_out,
                  size=0.034)
        self.text(sc, xs[1], 0.905, "their sum", op * merged, size=0.034)

    def _recap(self, sc, op):
        """The two models as their tables, small."""
        for row, (name, ink) in enumerate(zip(self.NAMES, self.COLOURS)):
            x = 0.10 + row * 0.40
            self.text(sc, x, 0.35, name, op, anchor="lm", ink=dark(ink), size=0.036)
            first, second = self.tables[row]
            self.grid(sc, first[:, :16].T, x + 0.15, 0.35, 96.0, 40.0, op, lines=0)
            if row == 0:
                self.grid(sc, second[:, :16].T, x + 0.272, 0.35, 96.0, 40.0, op, lines=0)

    def phase_4(self, sc, u, op):
        self._recap(sc, op)
        self.race(sc, u, op, self.steps, self.traces, self.NAMES, self.COLOURS,
                  box=((0.10, 0.58), (0.50, 0.84)), at=(0.68, 0.58))

    def phase_5(self, sc, u, op):
        self._recap(sc, op)
        self.result(sc, u, op, self.heldout, self.NAMES, self.COLOURS, fy=0.56, step=0.085,
                    thick=30.0)
        full = self.span(0.40)[0]
        a = op * smoothstep(0.4, 0.6, u)
        self.text(sc, 0.26, 0.745, "numbers each model has to learn", a, anchor="lm", strong=True)
        for row, (name, ink, n) in enumerate(zip(self.NAMES, self.COLOURS, self.params)):
            self.hbar(sc, 0.26, 0.805 + row * 0.07, full * n / max(self.params), ink, a, name,
                      f"{n:,}", seg(u, 0.4, 0.8), thick=24.0, track=full)


# ====================================================================== 19 the three families
def family_configs():
    """The study's encoder, causal decoder and cross-attending decoder."""
    enc = T.Config(vocab=len(R.vocabulary()), width=16, heads=4, kv_heads=4, ff=32, layers=1,
                   max_len=16, mask="bidirectional")
    return enc, replace(enc, mask="causal"), replace(enc, mask="causal", cross_attention=True)


def family_mask():
    """Who may read whom in the encoder-decoder, as one 12 x 12 matrix (exact):
    the six encoder positions read each other, the six decoder positions read
    every encoder position and the decoder positions up to their own."""
    enc, _causal, dec = family_configs()
    top = np.concatenate([T.attention_mask(6, enc), np.zeros((6, 6), bool)], axis=1)
    bottom = np.concatenate([np.ones((6, 6), bool), T.attention_mask(6, dec)], axis=1)
    return np.concatenate([top, bottom]).astype(float)


def _checkpoint(path, prefix):
    with np.load(path, allow_pickle=False) as d:
        return {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}


@cache
def family_answers():
    """What the study's three trained models write for its 64 held-out spans:
    ``(source, shifted, target, said)``, ``said`` ``(3, 64, 6)`` character
    ids (encoder only, decoder only, encoder-decoder), from the checkpoints
    the study's metadata names."""
    meta = NA.make("encoder-decoder-family").metadata
    enc, causal, dec = family_configs()
    source, shifted, target = NA.denoising_data(64, "test", 73)
    first = _checkpoint(meta["runs"][0]["artifact"]["path"], "P_")
    second = _checkpoint(meta["runs"][1]["artifact"]["path"], "P_")
    joint = _checkpoint(meta["joint_run"]["artifact"], "parameter_")
    slots = np.concatenate([source, np.zeros_like(source)], axis=1)
    forced = np.concatenate([source, shifted], axis=1)
    said = [T.forward(first, slots, enc)[:, 6:].argmax(-1),
            T.forward(second, forced, causal)[:, 6:].argmax(-1),
            NA.encoder_decoder_loss(joint, source, shifted, target, enc, dec, False)[1].argmax(-1)]
    return source, shifted, target, np.array(said)


class EncoderDecoderFamily(Built):
    """Encoder only, decoder only and encoder-decoder restoring the same
    damaged span: what each is given, who may read whom (the arcs and the
    mask), and what each trained model writes."""

    TITLE = "Encoder, decoder, encoder-decoder"
    PHASES = (("encoder", 11.0), ("decoder", 11.0), ("both", 13.0), ("training", 10.0),
              ("held out", 10.0))
    NAMES = ("encoder only", "decoder only", "encoder-decoder")
    COLOURS = (RED, BLUE, GREEN)
    ROW = 2                              # the held-out span drawn
    HIDDEN = (0.16, 0.17, 0.19)
    MASK_AT = (0.815, 0.585, 228.0)

    def setup(self):
        s = self.study
        source, shifted, target, said = family_answers()
        r = self.ROW
        self.source = [show(ch) for ch in R.decode(source[r])]
        self.shifted = [show(ch) for ch in R.decode(shifted[r])]
        self.target = [show(ch) for ch in R.decode(target[r])]
        self.said = [[show(ch) for ch in R.decode(row[r])] for row in said]
        self.right = [[int(a) == int(b) for a, b in zip(row[r], target[r])] for row in said]
        self.accuracy = [float((row == target).mean()) for row in said]
        self.masks = [np.asarray(s.panels[0].data["values"], float),
                      np.asarray(s.panels[1].data["values"], float), family_mask()]
        self.traces = np.asarray(s.panels[2].data["ys"], float)
        self.steps = np.arange(1, self.traces.shape[1] + 1)
        self.heldout = [float(v) for v in s.panels[3].data["values"]]
        self.updates = self.traces.shape[1]

    def subtitle_for(self, k):
        hidden = sum(ch == "?" for ch in self.source)
        return [
            "six characters of Shakespeare with %d hidden. An encoder reads the damaged text "
            "and six blank slots, every position reading every other, and fills the slots" % (
                hidden),
            "a decoder reads left to right only: the damaged text, then its own answer so far, "
            "writing one character at a time (%d of 144 pairs allowed)" % int(
                self.masks[1].sum()),
            "encoder-decoder: an encoder reads the damaged text both ways; a decoder writes "
            "the answer left to right, reading the encoder's whole output at every step",
            "all three trained on the same spans for %d updates. They are given different "
            "inputs and differ in size, so this is a tour, not a ranking" % self.updates,
            "held-out spans: encoder only %.2f nats, decoder only %.2f, encoder-decoder %.2f "
            "(%.1f%% of its characters right) after %d updates each" % (
                self.heldout[0], self.heldout[1], self.heldout[2], 100 * self.accuracy[2],
                self.updates),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _token(self, sc, x, y, ch, op, w, h=44.0, ink=None, size=0.038):
        """One character: the hidden symbol dark, the reader in ``ink``."""
        if ink is not None:
            self.box(sc, x, y, w, h, ink, op, ch, ink=WHITE, size=size, glow=0.15)
        elif ch == "?":
            self.box(sc, x, y, w, h, self.HIDDEN, op, "?", ink=WHITE, size=size)
        else:
            self.box(sc, x, y, w, h, PAPER, op, ch, border=AXIS, size=size)

    def _answers(self, sc, op, xs, y, which, done, w):
        """What the model has written so far, and the text it should match."""
        ink = self.COLOURS[which]
        for i, x in enumerate(xs):
            a = op * float(np.clip(done - i, 0.0, 1.0))
            ok = self.right[which][i]
            self.box(sc, x, y, w, 40.0, ink if ok else soft(ink, 0.7), a, self.said[which][i],
                     ink=WHITE if ok else dark(ink, 0.3), border=ink, size=0.038)
            self.box(sc, x, y + 0.095, w, 36.0, PAPER, op, self.target[i], border=AXIS,
                     size=0.036)
        return y + 0.095

    def _mask(self, sc, op, which, rows, turn):
        mx, my, side = self.MASK_AT
        cell = self.grid(sc, self.masks[which], mx, my, side, side, op, reveal=rows / 12.0,
                         ring=[(turn, turn)], hot=self.COLOURS[which])
        self.text(sc, mx, my - 0.27, "who may read whom", op, strong=True, size=0.034)
        half = self.frac(side / 4)[0]
        self.text(sc, mx - half, my + 0.245, "damaged text", op)
        self.text(sc, mx + half, my + 0.245, "answer", op)
        self.text(sc, mx, my + 0.29, "row: the reader", op)
        return cell

    def _one_stack(self, sc, u, op, which, row):
        """A single stack over twelve positions (the damaged text, then six
        answer positions), the answer positions reading in turn."""
        ink = self.COLOURS[which]
        mask = self.masks[which]
        x0, x1, y = 0.05, 0.62, 0.61
        n = 12
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        turn = 6 + min(int(u * 1.1 * 6), 5)
        within = min(u * 1.1 * 6 - (turn - 6), 1.0)
        for j, (x, ch) in enumerate(zip(xs, row)):
            seen = j == turn or mask[turn, j]
            self._token(sc, x, y, ch, op * (1.0 if seen else 0.4), 40.0,
                        ink=ink if j == turn else None)
        for j in range(n):
            if j == turn or not mask[turn, j]:
                continue
            bend = 0.07 + 0.024 * abs(turn - j)
            a, b = (xs[j], y - 0.052), (xs[turn], y - 0.052)
            self.arc(sc, a, b, ink, op, smoothstep(0.0, 0.5, within), bend, 2.2)
            self.pulse(sc, a, b, (within - 0.3) / 0.6, ink, op, bend)
        done = (turn - 6) + smoothstep(0.75, 0.95, within)
        bottom = self._answers(sc, op, xs[6:], y + 0.105, which, done, 40.0)
        self.text(sc, xs[5] + 0.018, y + 0.105, "it writes", op, anchor="rm", ink=dark(ink),
                  size=0.034)
        self.text(sc, xs[5] + 0.018, bottom, "the text", op, anchor="rm", size=0.034)
        self.text(sc, x0 + 0.005, 0.315, "%s:  position %d reads %d of the 12" % (
            self.NAMES[which], turn + 1, int(mask[turn].sum())), op, anchor="lm", strong=True,
            size=0.034)
        self._mask(sc, op, which, turn + 1, turn)

    def phase_1(self, sc, u, op):
        self._one_stack(sc, u, op, 0, self.source + ["?"] * 6)

    def phase_2(self, sc, u, op):
        self._one_stack(sc, u, op, 1, self.source + self.shifted)

    def phase_3(self, sc, u, op):
        ink = GREEN
        x0, x1, ye, yd = 0.16, 0.61, 0.43, 0.665
        xs = [x0 + (x1 - x0) * (i + 0.5) / 6 for i in range(6)]
        turn = min(int(seg(u, 0.14, 1.0) * 1.1 * 6), 5)
        within = min(seg(u, 0.14, 1.0) * 1.1 * 6 - turn, 1.0) if u > 0.14 else 0.0
        # the encoder reads the damaged text, every position every other
        own = seg(u, 0.0, 0.14)
        for i in range(6):
            for j in range(i + 1, 6):
                bend = 0.05 + 0.02 * (j - i)
                self.arc(sc, (xs[i], ye - 0.052), (xs[j], ye - 0.052), ink,
                         op * (0.9 if own < 1 else 0.32), own, bend, 1.8, head=False)
        for x, ch in zip(xs, self.source):
            self._token(sc, x, ye, ch, op, 52.0)
        # the decoder writes left to right, reading the encoder at every step
        for j, (x, ch) in enumerate(zip(xs, self.shifted)):
            self._token(sc, x, yd, ch, op * (1.0 if j <= turn or u <= 0.14 else 0.4), 52.0,
                        ink=ink if j == turn and u > 0.14 else None)
        if u > 0.14:
            for j in range(6):
                a, b = (xs[j], ye + 0.05), (xs[turn], yd - 0.05)
                self.arrow(sc, a, b, ink, op, smoothstep(0.0, 0.45, within), 2.2)
                self.pulse(sc, a, b, (within - 0.25) / 0.6, ink, op)
        done = turn + smoothstep(0.75, 0.95, within) if u > 0.14 else 0.0
        bottom = self._answers(sc, op, xs, yd + 0.105, 2, done, 52.0)
        for y, name, strong in ((ye, "encoder", True), (yd, "decoder", True),
                                (yd + 0.105, "it writes", False), (bottom, "the text", False)):
            self.text(sc, x0 - 0.015, y, name, op, anchor="rm", strong=strong, size=0.034,
                      ink=None if strong or name == "the text" else dark(ink))
        self._mask(sc, op, 2, 6 + (turn + 1 if u > 0.14 else 0), 6 + turn)

    def _task(self, sc, op):
        """The task all three train on."""
        xs = [0.10 + 0.042 * i for i in range(6)]
        for x, ch in zip(xs, self.source):
            self._token(sc, x, 0.345, ch, op, 36.0, h=36.0, size=0.034)
        self.arrow(sc, (xs[-1] + 0.035, 0.345), (xs[-1] + 0.075, 0.345), STREAM, op, 1.0, 2.4)
        for x, ch in zip(xs, self.target):
            self._token(sc, x + 0.32, 0.345, ch, op, 36.0, h=36.0, size=0.034)
        self.text(sc, xs[-1] + 0.36, 0.345, "the task: restore the hidden characters", op,
                  anchor="lm", strong=True)

    def phase_4(self, sc, u, op):
        self._task(sc, op)
        self.race(sc, u, op, self.steps, self.traces, self.NAMES, self.COLOURS,
                  box=((0.10, 0.58), (0.50, 0.84)), at=(0.66, 0.56))

    def phase_5(self, sc, u, op):
        self._task(sc, op)
        bottom = self.result(sc, u, op, self.heldout, self.NAMES, self.COLOURS, fx=0.30,
                             fy=0.535, step=0.085, thick=30.0,
                             title="loss on held-out spans (lower is better)")
        self.lines(sc, 0.30, bottom + 0.02, [
            "this span: %d, %d and %d of its 6 characters right" % tuple(
                sum(r) for r in self.right),
            "different inputs, different sizes: a tour, not a ranking"],
            op * smoothstep(0.5, 0.75, u), size=0.034, step=0.05)


# ====================================================================== 20 depth and width
@cache
def layer_attention():
    """Where each character looks in every block of the study's four trained
    shapes (the first seed of each), for one held-out window: a list of
    ``(layers, 12, 12)`` arrays (rows: the reader; the heads averaged)."""
    configs, runs, _losses, _test = trained("transformer-depth-vs-width")
    ids, _chars = test_window()
    maps = []
    for c, run in zip(configs, runs):
        _logits, rec = T.forward(run["P"], ids[None], c, keep=True)
        maps.append(np.array([a[0].mean(0) for a in rec["attention"]]))
    return maps


def blocks(n):
    return "%d block%s" % (n, "" if n == 1 else "s")


class TransformerDepthVsWidth(Built):
    """The same number of weights spent four ways: one wide block, two, four,
    eight narrow ones. Each block is one more round of every character reading
    the earlier ones; then the four trained on text none of them sees twice,
    three seeds each, their loss on new text, and how much of it each gets
    from a longer context."""

    TITLE = "Depth and width"
    PHASES = (("four shapes", 8.0), ("rounds", 13.0), ("training", 10.0), ("held out", 12.0),
              ("context", 11.0))
    COLOURS = (RED, VIOLET, BLUE, GREEN)
    LATE = 8                            # "a long context": more characters seen than this

    def setup(self):
        s, m = self.study, self.study.metadata
        self.configs, _runs, _losses, _test = trained("transformer-depth-vs-width")
        self.ids, self.chars = test_window()
        self.maps = layer_attention()
        self.widths = [c.width for c in self.configs]
        self.depths = [c.layers for c in self.configs]
        self.NAMES = tuple("%s x %d" % (blocks(d), w) for d, w in zip(self.depths, self.widths))
        self.short = tuple(blocks(d) for d in self.depths)
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.fitted = [float(v) for v in m["final_training_loss"]]
        self.by_seed = np.asarray(m["heldout_loss_by_seed"], float)       # (shapes, seeds)
        self.heldout = self.by_seed.mean(1)
        self.params = [int(v) for v in s.panels[0].data["values"]]
        self.updates, self.seeds = int(m["steps"]), self.by_seed.shape[1]
        self.windows, self.length = int(m["training_windows"]), int(m["window_length"])
        self.guess = float(m["uniform_guess"])
        self.seen = np.asarray(s.panels[3].data["x"], float)              # characters seen
        self.context = np.asarray(s.panels[3].data["ys"], float)          # (shapes, length)
        self.late = self.context[:, self.LATE:].mean(1)
        # deeper pays with a long context if some deeper shape beats one block there by
        # more than the largest gap between two seeds of a shape on the whole text
        self.deeper_reads_further = bool(self.late[0] - self.late[1:].min() > (
            self.by_seed.max(1) - self.by_seed.min(1)).max())
        self.best = int(np.argmin(self.heldout))
        # shapes whose seeds overlap the best one's, and those worse in every seed
        top = self.by_seed[self.best].max()
        self.tied = [i for i in range(len(self.depths)) if self.by_seed[i].min() <= top]
        self.worse = [i for i in range(len(self.depths)) if i not in self.tied]

    def ranking(self):
        """What the seeds allow to be said about the order of the shapes."""
        def listed(ids):
            return " and ".join("%d" % self.depths[i] for i in ids)
        parts = []
        if len(self.tied) > 1:
            parts.append("%s overlap across seeds" % listed(self.tied))
        if self.worse:
            parts.append("%s %s worse in every seed" % (
                listed(self.worse), "is" if len(self.worse) == 1 else "are"))
        deepest = self.best == len(self.depths) - 1
        return "; ".join(parts) + (": the deepest is best" if deepest else
                                   ": past %s, depth did not pay here" % blocks(
                                       self.depths[max(self.tied)]))

    def subtitle_for(self, k):
        w, d, h = self.widths, self.depths, self.heldout
        size = f"{int(round(np.mean(self.params), -2)):,}"
        return [
            "four models of the same size, about %s weights each (within %.1f%%): 1 block %d "
            "numbers wide, 2 blocks of %d, 4 of %d, 8 of %d" % (
                size, 100 * (max(self.params) - min(self.params)) / min(self.params),
                w[0], w[1], w[2], w[3]),
            "each block is one round in which every character reads the ones before it. "
            "The widest model gets %d round, the narrowest %d: here is where each looks" % (
                d[0], d[-1]),
            "%s updates each on %s windows of Tiny Shakespeare, each drawn about once: too "
            "many to memorise. %d seeds per shape; the curves are their averages" % (
                f"{self.updates:,}", f"{self.windows:,}", self.seeds),
            "held-out, mean of %d seeds: %s. %s" % (
                self.seeds, ", ".join("%s %.2f" % (n, v) for n, v in zip(self.short, h)),
                self.ranking()),
            "loss by characters seen: %.2f after one, about %.2f after four, in every shape. "
            "Past %d characters: %s for 1, 2, 4, 8 blocks: %s" % (
                self.context[:, 0].mean(), self.context[:, 3].mean(), self.LATE,
                ", ".join("%.2f" % v for v in self.late),
                "the deeper models make more of a long context" if self.deeper_reads_further
                else "depth did not make more of a long context"),
        ][k]

    # ------------------------------------------------------------------ the four shapes
    def phase_1(self, sc, u, op):
        base, tall, px = 0.81, 26.0, 3.1
        for m, (x, ink) in enumerate(zip((0.15, 0.385, 0.62, 0.855), self.COLOURS)):
            wide = self.widths[m] * px
            for i in range(self.depths[m]):
                a = op * smoothstep(0.08 + 0.07 * i, 0.16 + 0.07 * i, u)
                y = base - self.frac(0, (i + 0.5) * (tall + 4.0))[1]
                self.box(sc, x, y, wide, tall, soft(ink, 0.25), a, border=ink, radius=4)
            top = base - self.frac(0, self.depths[m] * (tall + 4.0))[1]
            self.arrow(sc, (x, base + 0.012), (x, top - 0.04), STREAM, op * 0.45,
                       seg(u, 0.1, 0.1 + 0.09 * self.depths[m]), 2.0)
            self.text(sc, x, base + 0.045, self.NAMES[m], op, ink=dark(ink), size=0.036)
            self.text(sc, x, base + 0.095, "%s weights" % f"{self.params[m]:,}", op)
        self.text(sc, 0.05, 0.315, "a block's width: how many numbers describe each character",
                  op, anchor="lm", strong=True)

    # ------------------------------------------------------------------ rounds of reading
    ROWS = (0.415, 0.565, 0.715, 0.865)
    MAP = 62.0

    def _maps(self, sc, op, reach):
        """Every block of every model as the map of where its characters
        look; ``reach``: how many rounds have run."""
        x0, pitch = 0.237, 0.0955
        for m, (y, ink) in enumerate(zip(self.ROWS, self.COLOURS)):
            n = self.depths[m]
            self.text(sc, 0.05, y - 0.025, self.NAMES[m], op, anchor="lm", ink=dark(ink),
                      size=0.036)
            self.text(sc, 0.05, y + 0.025, "%d round%s" % (n, "" if n == 1 else "s"), op,
                      anchor="lm")
            xs = [x0 + pitch * i for i in range(n)]
            thick = max(self.widths[m] / 6.0, 2.0)
            self.polyline(sc, [self.P(x0 - 0.05, y), self.P(xs[-1] + 0.045, y)], thick, ink,
                          op * 0.45, bias=0.0)
            for i, x in enumerate(xs):
                lit = seg(reach, i, i + 0.6)
                self.grid(sc, self.maps[m][i] * lit, x, y, self.MAP, self.MAP, op, vmax=1.0,
                          hot=ink, lines=0)
            if 0 < reach < n:
                at = x0 - 0.05 + (reach / n) * (xs[-1] + 0.045 - (x0 - 0.05))
                self.disk(sc, self.P(at, y), 7.0, ink, op, glow=0.7, lift=1.8, outline=0.0)
            elif reach >= n:
                if n < 8:
                    self.text(sc, xs[-1] + 0.06, y, "done", op * smoothstep(n, n + 0.5, reach),
                              anchor="lm", ink=dark(ink), size=0.034)

    def phase_2(self, sc, u, op):
        self._maps(sc, op, seg(u, 0.06, 0.86) * 8.0)
        self.text(sc, 0.05, 0.305, "each square is one block.  row: a character reading;  "
                  "column: the earlier one it reads", op, anchor="lm", strong=True)

    def _recap(self, sc, op):
        """The four shapes, small."""
        for m, (x, ink) in enumerate(zip((0.07, 0.29, 0.51, 0.73), self.COLOURS)):
            self.text(sc, x, 0.35, self.short[m], op, anchor="lm", ink=dark(ink), size=0.034)
            for i in range(self.depths[m]):
                self.box(sc, x + 0.095 + i * 0.0105, 0.35, 7.0, self.widths[m] * 0.62,
                         soft(ink, 0.25), op, radius=2)

    def phase_3(self, sc, u, op):
        self._recap(sc, op)
        self.race(sc, u, op, self.steps, self.traces, self.short, self.COLOURS,
                  box=((0.10, 0.58), (0.50, 0.84)), at=(0.66, 0.52))

    def phase_4(self, sc, u, op):
        self._recap(sc, op)
        fx, wide = 0.19, 0.36
        self.seed_dots(sc, u, op, self.by_seed, self.short, self.COLOURS, fx=fx, fy=0.535,
                       wide=wide, step=0.085, fmt="%.2f",
                       title="held-out loss of every run (nats, lower is better)")
        a = op * smoothstep(0.72, 0.86, u)
        self.text(sc, 0.72, 0.45, "its last 100", a, anchor="lm", strong=True)
        self.text(sc, 0.72, 0.487, "training batches", a, anchor="lm", strong=True)
        for m in range(len(self.depths)):
            self.text(sc, 0.72, 0.535 + m * 0.085, "%.2f" % self.fitted[m], a, anchor="lm")
        self.text(sc, 0.72, 0.535 + len(self.depths) * 0.085, "guessing evenly: %.2f" %
                  self.guess, a, anchor="lm")

    # ------------------------------------------------------------------ what context buys
    def phase_5(self, sc, u, op):
        self._recap(sc, op)
        chart = self.chart(((0.10, 0.56), (0.47, 0.84)), self.seen, self.context)
        self.axes(sc, chart, op, "characters the model has seen", "held-out loss (nats)",
                  xticks=[(v, "%d" % v) for v in (1, 4, 8, 12, 16) if v <= self.seen[-1]])
        for m, ink in enumerate(self.COLOURS):
            self.plot(sc, chart, self.seen, self.context[m], ink, op, u / 0.6)
        x = chart.fx(self.LATE + 0.5)
        a = op * smoothstep(0.62, 0.74, u)
        self.dashes(sc, (x, chart.fy(chart.y_range[1])), (x, chart.fy(chart.y_range[0])),
                    STREAM, a * 1.6, n=18, width=1.6)
        self.lines(sc, 0.66, 0.50, [("more than %d characters seen" % self.LATE, None)] + [
            ("%s  %.2f" % (name, v), dark(ink))
            for name, v, ink in zip(self.short, self.late, self.COLOURS)], a, step=0.06,
            size=0.036)


# ====================================================================== 16 the residual stream
@cache
def stream_trace():
    """GPT-2 on the study's text, for the last token: the tokens, the stream's
    length entering and after every layer (13), the top token and its
    probability read out of the stream at each of the 25 points (the logit
    lens), the final top five, and the first 48 of the 768 numbers entering."""
    from . import gpt2

    tok = gpt2.Tokenizer()
    ids = tok.encode(R.text("test")[:64])
    tr = gpt2.run(tok.decode(ids))
    lens = gpt2.logit_lens(tr)
    tops = lens.argmax(-1)
    return dict(tokens=list(tr.tokens),
                length=np.array([float(np.linalg.norm(r[-1])) for r in tr.resid]),
                top=[tok.decode([int(i)]) for i in tops],
                top_p=lens[np.arange(len(lens)), tops],
                final=list(tr.top), enters=np.asarray(tr.resid[0][-1][:48], float),
                answer=tok.decode([int(tops[-1])]))


def glyphs(token):
    """A GPT-2 token as it is drawn in a box (punctuation in quotes: a bare
    comma is too small to read)."""
    shown = token.replace("\n", "↵").replace(" ", "␣")
    return shown if any(ch.isalnum() or ch in "↵␣" for ch in shown) else '"%s"' % shown


class ResidualStream(Built):
    """GPT-2's stream for one token as a bus: twelve layers, each with an
    attention tap and an MLP tap that read it and write back; how large each
    write is, and what the model would answer if it stopped after each."""

    TITLE = "The residual stream"
    PHASES = (("the stream", 9.0), ("writes", 17.0), ("read out", 15.0), ("the answer", 10.0))
    BUS = 0.47
    UP, DOWN = 0.36, 0.58
    X0, STEP = 0.14, 0.03              # the 25 read-out points along the bus

    def setup(self):
        s = self.study
        self.attn, self.mlp = np.asarray(s.panels[0].data["ys"], float)
        self.lens = np.asarray(s.panels[1].data["ys"], float)[0]            # 25 points
        self.t = stream_trace()
        self.tokens_ = [glyphs(t) for t in self.t["tokens"][-8:]]
        self.answer = glyphs(self.t["answer"])
        self.big = float(max(self.attn.max(), self.mlp.max()))

    def at(self, point):
        """Layout x of read-out point ``point`` (0: entering; 2l+1 after layer
        l+1's attention; 2l+2 after its MLP)."""
        return self.X0 + self.STEP * point

    def subtitle_for(self, k):
        t = self.t
        first = next(i for i, tok in enumerate(t["top"]) if tok == t["answer"])
        return [
            "GPT-2 holds %d numbers for each token: its stream. Nothing replaces them; each "
            "of the 12 layers reads the stream and adds to it, twice" % 768,
            "the writes, measured for the token %s: small at first (attention %.0f, MLP %.0f "
            "in layer 2), largest in layer 12 (%.0f and %.0f). The stream grows from %.0f to "
            "%.0f" % (self.tokens_[-1], self.attn[1], self.mlp[1], self.attn[-1], self.mlp[-1],
                      t["length"][0], t["length"][-1]),
            "reading the answer out of the stream early: it still says %s until layer %d, and "
            "%s from layer %d on, peaking at %.0f%%" % (
                glyphs(t["top"][0]), (first - 1) // 2, self.answer, (first + 1) // 2,
                100 * self.lens.max()),
            "the last layer's writes are the largest, and pull the answer back: %s falls from "
            "%.0f%% to %.0f%%, with %s at %.0f%%" % (
                self.answer, 100 * self.lens[22], 100 * self.lens[24], glyphs(t["final"][1][0]),
                100 * t["final"][1][1]),
        ][k]

    # ------------------------------------------------------------------ the bus
    def _bus(self, sc, op, drawn=1.0, lit=None, dim=1.0):
        """The stream left to right with its 24 taps; ``lit``: ``(layer,
        'a' | 'm', 0..1)`` the tap at work."""
        y = self.BUS
        self.box(sc, 0.075, y, 56.0, 40.0, STREAM, op, self.tokens_[-1], ink=WHITE, size=0.034)
        self.wire(sc, [(0.108, y), (0.905, y)], STREAM, op, drawn, 6.0)
        for layer in range(12):
            for kind, colour, ty, point in (("a", ATTN, self.UP, 2 * layer + 1),
                                            ("m", MLP, self.DOWN, 2 * layer + 2)):
                x = self.at(point)
                a = op * dim * smoothstep(layer / 14.0, layer / 14.0 + 0.12, drawn)
                on = lit[2] if lit and lit[:2] == (layer, kind) else 0.0
                side = -1.0 if kind == "a" else 1.0
                edge = ty - side * 0.034
                self.arrow(sc, (x - 0.007, y + side * 0.012), (x - 0.007, edge), colour, a, 1.0,
                           1.8)
                self.arrow(sc, (x + 0.007, edge), (x + 0.007, y + side * 0.012), colour, a, 1.0,
                           1.8)
                self.node(sc, x, ty, "A" if kind == "a" else "M", colour, a, on, w=26.0, h=30.0,
                          size=0.030)
                if on > 0:
                    self.ride(sc, [(x - 0.007, y), (x - 0.007, edge)], seg(on, 0.0, 0.45), colour,
                              op, r=5.0)
                    self.ride(sc, [(x + 0.007, edge), (x + 0.007, y)], seg(on, 0.5, 1.0), colour,
                              op, r=5.0)

    def _layers(self, sc, op, y):
        for layer in range(12):
            self.text(sc, self.at(2 * layer + 1.5), y, "%d" % (layer + 1), op)
        self.text(sc, self.at(0) - 0.015, y, "layer", op, anchor="rm")

    def phase_1(self, sc, u, op):
        self._bus(sc, op, seg(u, 0.05, 0.75))
        self._layers(sc, op * smoothstep(0.3, 0.5, u), 0.66)
        self.text(sc, 0.05, 0.305, "the stream for the token  %s" % self.tokens_[-1], op,
                  anchor="lm", strong=True, size=0.034)
        a = op * smoothstep(0.55, 0.7, u)
        self.node(sc, 0.10, 0.76, "A", ATTN, a, 0.0, w=26.0, h=30.0, size=0.030)
        self.text(sc, 0.125, 0.76, "attention: reads the stream of this token and the earlier "
                  "ones, adds its result", a, anchor="lm")
        self.node(sc, 0.10, 0.83, "M", MLP, a, 0.0, w=26.0, h=30.0, size=0.030)
        self.text(sc, 0.125, 0.83, "the MLP: reads this token's stream, adds its result", a,
                  anchor="lm")
        b = op * smoothstep(0.75, 0.9, u)
        self.grid(sc, self.t["enters"][None, :], 0.42, 0.90, 432.0, 16.0, b, lines=0)
        self.text(sc, 0.68, 0.90, "the first 48 of its 768 numbers", b, anchor="lm")

    def phase_2(self, sc, u, op):
        w = seg(u, 0.04, 0.90) * 24.0                    # one unit per write
        k = int(min(w, 23))
        layer, kind = k // 2, "am"[k % 2]
        self._bus(sc, op, lit=(layer, kind, w - k if w < 24 else 0.0))
        self._layers(sc, op, 0.66)
        base, top = 0.925, 140.0                         # the scale ends at ``top``
        unit = 0.17 / top
        for v in (50, 100):
            self.dashes(sc, (self.at(0.4), base - v * unit), (self.at(24.6), base - v * unit),
                        AXIS, op, n=44, width=1.0)
            self.text(sc, self.at(0) - 0.015, base - v * unit, "%d" % v, op, anchor="rm")
        for i in range(24):
            grow = seg(w, i + 0.55, i + 1.0)
            if grow <= 0:
                continue
            value = (self.attn if i % 2 == 0 else self.mlp)[i // 2]
            ink = ATTN if i % 2 == 0 else MLP
            h = min(value, top) * grow * unit
            self.rect(sc, self.P(self.at(i + 1), base - h / 2), 18.0, max(self.span(0, h)[1], 1.0),
                      ink, op, radius=2)
            if value > 100:                               # the two largest carry their size
                over = value > top                        # and one runs off the scale
                if over:
                    for k in range(2):                    # a broken bar: it continues
                        self.rect(sc, self.P(self.at(i + 1), base - h - 0.008 - 0.012 * k), 18.0,
                                  3.5, ink, op * grow, radius=1)
                if over:
                    self.text(sc, self.at(i + 1) - 0.016, base - h + 0.012, "%.0f" % value,
                              op * grow, anchor="rm", ink=dark(ink, 0.35), size=0.034)
                else:
                    self.text(sc, self.at(i + 1), base - h - 0.03, "%.0f" % value, op * grow,
                              ink=dark(ink, 0.35), size=0.034)
        if w < 24:
            self.disk(sc, self.P(self.at(k + 1), self.BUS), 7.5, WHITE, op, glow=0.8, lift=1.9,
                      outline=0.0)
        done = min(int(w + 0.45), 24)
        self.text(sc, 0.05, 0.305, "layer %d:" % (layer + 1), op, anchor="lm", strong=True,
                  size=0.036)
        self.text(sc, 0.175, 0.305, "attention writes %.0f" % self.attn[layer],
                  op * (1.0 if done >= 2 * layer + 1 else 0.3), anchor="lm", ink=dark(ATTN, 0.35),
                  size=0.036)
        self.text(sc, 0.44, 0.305, "the MLP writes %.0f" % self.mlp[layer],
                  op * (1.0 if done >= 2 * layer + 2 else 0.3), anchor="lm", ink=dark(MLP, 0.3),
                  size=0.036)
        self.text(sc, 0.95, 0.305, "stream length %.0f" % self.t["length"][min(done // 2, 12)],
                  op, anchor="rm", size=0.036)
        self.text(sc, self.at(0.4), 0.705, "size of each write (the scale stops at %d)" % top,
                  op, anchor="lm", strong=True)

    def phase_3(self, sc, u, op):
        t = self.t
        self._bus(sc, op, dim=0.55)
        w = seg(u, 0.06, 0.88) * 24.0
        p = int(min(w, 24))
        self.disk(sc, self.P(self.at(min(w, 24.0)), self.BUS), 7.5, WHITE, op, glow=0.8, lift=1.9,
                  outline=0.0)
        row = 0.70
        for i in range(13):                              # entering, then after each layer
            point = 2 * i
            a = op * smoothstep(point - 0.2, point + 0.4, w + 0.3)
            x = self.at(point) if i else self.at(0)
            same = t["top"][point] == t["answer"]
            self.arrow(sc, (x + 0.015, self.BUS + 0.018), (x + 0.015, row - 0.045), GREEN, a * 0.45,
                       1.0, 1.6)
            self.box(sc, x + 0.015, row, 46.0, 36.0, GREEN if same else PAPER, a,
                     glyphs(t["top"][point]), ink=WHITE if same else None,
                     border=GREEN if same else AXIS, size=0.032)
        chart = self.chart(((self.at(0) + 0.015, self.at(24) + 0.015), (0.775, 0.915)),
                           np.arange(25), np.array([0.0, 1.0]), pad=0.0)
        self.axes(sc, chart, op, "", "", xticks=[], yticks=[(0, "0"), (0.5, "50%"), (1, "100%")])
        self.plot(sc, chart, np.arange(25), self.lens, GREEN, op, w / 24.0, width=2.2)
        where = "entering" if p == 0 else "after layer %d's %s" % (
            (p + 1) // 2, "attention" if p % 2 else "MLP")
        self.text(sc, 0.05, 0.305, "read out %s:" % where, op, anchor="lm", strong=True,
                  size=0.036)
        self.text(sc, 0.47, 0.305, "top answer  %s  %.0f%%" % (
            glyphs(t["top"][p]), 100 * t["top_p"][p]), op, anchor="lm", size=0.036)
        self.text(sc, 0.95, 0.305, "chance of  %s  %.0f%%" % (self.answer, 100 * self.lens[p]),
                  op, anchor="rm", ink=dark(GREEN, 0.35), size=0.036)

    def phase_4(self, sc, u, op):
        t = self.t
        xs = self.tokens(sc, self.tokens_ + [""], 0.08, 0.62, 0.345, op, w=56.0, h=40.0,
                         size=0.034)
        self.box(sc, xs[-1], 0.345, 56, 40, GREEN, op * smoothstep(0.1, 0.25, u), self.answer,
                 ink=WHITE, size=0.036)
        self.text(sc, 0.65, 0.345, "the text, and GPT-2's next token", op, anchor="lm")
        full = self.span(0.26)[0]
        self.text(sc, 0.16, 0.47, "its answer after layer 12", op, anchor="lm", strong=True)
        for i, (token, p) in enumerate(t["final"]):
            self.hbar(sc, 0.16, 0.54 + i * 0.075, full * p / t["final"][0][1],
                      GREEN if i == 0 else soft(GREEN, 0.45), op, glyphs(token),
                      "%.1f%%" % (100 * p), seg(u, 0.1 + 0.05 * i, 0.4 + 0.05 * i), thick=26.0,
                      track=full)
        a = op * smoothstep(0.45, 0.6, u)
        self.text(sc, 0.58, 0.47, "the chance of  %s  on the way" % self.answer, a, anchor="lm",
                  strong=True)
        rows = (("after layer 11", self.lens[22], None),
                ("layer 12's attention writes %.0f" % self.attn[-1], self.lens[23], ATTN),
                ("layer 12's MLP writes %.0f" % self.mlp[-1], self.lens[24], MLP))
        for i, (what, p, ink) in enumerate(rows):
            b = a * smoothstep(0.5 + 0.1 * i, 0.6 + 0.1 * i, u)
            self.text(sc, 0.58, 0.55 + i * 0.105, what, b, anchor="lm",
                      ink=None if ink is None else dark(ink, 0.35), size=0.032)
            self.text(sc, 0.58, 0.55 + i * 0.105 + 0.045, "%.0f%%" % (100 * p), b, anchor="lm",
                      strong=True, size=0.038)


ANIMATIONS = {
    "pre-ln-vs-post-ln": PreLnVsPostLn,
    "rmsnorm-vs-layernorm": RmsNormVsLayerNorm,
    "parallel-block": ParallelBlock,
    "swiglu-mlp": SwigluMlp,
    "weight-tying": WeightTying,
    "encoder-decoder-family": EncoderDecoderFamily,
    "transformer-depth-vs-width": TransformerDepthVsWidth,
    "residual-stream": ResidualStream,
}
