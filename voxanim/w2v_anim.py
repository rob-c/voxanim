"""word2vec: skip-gram with negative sampling, trained on a small generated corpus.

Every number is read from the study :mod:`voxanim.w2v` (39 words, 16-wide
vectors, 14,034 SGD steps over 898,216 (centre, context) pairs).  Four phases:

1. **corpus** -- the first sentences of the generated corpus (30,000
   sentences from 8 templates: a toy, said so), then one sentence with the
   window of :data:`w2v.WINDOW` = 4 words sliding across it, listing the
   (centre, context) pairs each centre gives (from :func:`w2v.pairs`),
2. **one step** -- one SGNS update with real numbers: the centre ``baker``'s
   input vector ``v``, the output vectors ``u`` of its true context
   ``bread`` and of 5 noise words drawn (seed 0) from the study's noise
   distribution (unigram ^ 3/4); each ``u · v`` and ``σ(u · v)`` before and
   after the step, and the pair's loss.  The step is taken at the
   **exact** training state of snapshot 6 (step 1,403), with the learning
   rate the schedule had there (0.45): the study stores ``W_in`` at each
   snapshot but not ``W_out``, so :func:`replay` reruns the study's own
   training loop (same seed, same order of random draws) up to that step
   and checks that its ``W_in`` equals the stored snapshot bit for bit;
   its ``W_out`` is then the real one, not the final one standing in.
   The step itself is :func:`w2v.sgns_step` on those vectors,
3. **training** -- the 24 category words' unit vectors in one fixed 2-D
   view (:func:`w2v.projection` of the final vectors: their mean and top two
   principal axes, used for every snapshot), playing through the 61
   snapshots from a random cloud to four clusters; beside it the mean
   cosine between words of the same category and of different categories
   per snapshot (:func:`w2v.category_cosines`) and the batch loss averaged
   over 200 steps,
4. **neighbours** -- the 5 nearest words by cosine to cat, baker, bread and
   village in the final vectors (:func:`w2v.neighbours`).
"""

from __future__ import annotations

import dataclasses
from functools import cache

import numpy as np

from . import color as C
from . import layout, w2v
from .diagram import DATA, EDGE, Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .netanim import amax, div_colors
from .text import num

STEP_SNAP = 6                         # the SGNS step is taken at this snapshot (step 1,403)
CENTRE, CONTEXT = "baker", "bread"
NEG_SEED = 0                          # the noise words' draw
N_LISTED = 6                          # corpus sentences listed
NEAR_WORDS = ("cat", "baker", "bread", "village")
N_NEAR = 5
LOSS_SMOOTH = 200                     # steps averaged in the loss curve
SNAP_POWER = 1.8                      # the snapshots play slowly early, where most happens
CAT_ORDER = ("animal", "person", "food", "place")
CAT_NAMES = {"animal": "animals", "person": "people", "food": "foods", "place": "places"}
CAT_INKS = {"animal": C.rgb("#d84e4e"), "person": C.rgb("#6370eb"),
            "food": C.rgb("#a1814b"), "place": C.rgb("#3d9e45")}
# where each category's name sits beside its cluster (the clusters' final layout)
LABEL_SIDE = {"animal": "right", "person": "below", "food": "above", "place": "below"}
PULL_INK = C.rgb("#3d9e45")           # the true context, pulled closer
PUSH_INK = C.rgb("#d84e4e")           # the noise words, pushed away
PANEL = (0.96, 0.96, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
SAME_INK, CROSS_INK = (0.14, 0.16, 0.18), (0.55, 0.57, 0.59)
STRIP_W, STRIP_H = 224.0, 14.0        # px: a 16-number vector


def text_ink(c):
    """A category colour darkened enough to read as text on the backdrop."""
    return tuple(float(v) for v in C.darken(c, 0.25))


# ------------------------------------------------------------------ study helpers


@cache
def replay(snap):
    """The exact ``(W_in, W_out)`` of the study's training at snapshot ``snap``.

    :func:`w2v.train` keeps ``W_in`` at every snapshot but only the final
    ``W_out``; this reruns its loop (the same seed and the same sequence of
    random draws) up to the snapshot's step, and checks the result's ``W_in``
    against the stored snapshot, bit for bit.  Also returns the noise
    distribution and the number of training pairs."""
    run = w2v.train()
    index = run.index
    sents = w2v.corpus()
    P = w2v.pairs(sents, index)
    noise = w2v.unigram(sents, index) ** 0.75
    noise /= noise.sum()
    V = len(run.words)
    rng = np.random.default_rng(w2v.TRAIN_SEED)
    W_in = rng.uniform(-0.5, 0.5, (V, w2v.DIM)) / w2v.DIM
    W_out = np.zeros((V, w2v.DIM))
    per_epoch = len(P) // w2v.BATCH
    steps = w2v.EPOCHS * per_epoch
    target = int(run.snap_steps[snap])
    k = 0
    for _ep in range(w2v.EPOCHS):
        if k >= target:
            break
        order = rng.permutation(len(P))
        for b in range(per_epoch):
            if k >= target:
                break
            idx = order[b * w2v.BATCH:(b + 1) * w2v.BATCH]
            c, o = P[idx, 0], P[idx, 1]
            neg = rng.choice(V, (len(idx), w2v.NEGATIVES), p=noise)
            lr = w2v.LR * (1.0 - k / steps)
            _loss, ri, gi, ro, go = w2v.sgns_grads(W_in, W_out, c, o, neg)
            np.add.at(W_in, ri, -lr * gi)
            np.add.at(W_out, ro, -lr * go)
            k += 1
    if not np.array_equal(W_in, run.snaps[snap]):
        raise RuntimeError("the replayed training does not match the stored snapshot")
    return W_in, W_out, noise, len(P)


@cache
def training_step(snap=STEP_SNAP, centre=CENTRE, context=CONTEXT, seed=NEG_SEED):
    """One SGNS step on ``(centre, context)`` at the exact state of snapshot
    ``snap``, with :data:`w2v.NEGATIVES` noise words drawn with ``seed``:
    :func:`w2v.sgns_step`'s numbers, plus both whole tables before and after."""
    run = w2v.train()
    W_in, W_out, noise, _n = replay(snap)
    rng = np.random.default_rng(seed)
    negatives = [run.words[i] for i in rng.choice(len(run.words), w2v.NEGATIVES, p=noise)]
    lr = w2v.LR * (1.0 - run.snap_steps[snap] / run.steps)   # the schedule's rate there
    exact = dataclasses.replace(run, W_out=W_out)
    d = w2v.sgns_step(exact, snap, centre, context, negatives, lr=lr)
    # the same update applied to whole copies of the tables
    W_in1, W_out1 = W_in.copy(), W_out.copy()
    _l, ri, gi, ro, go = w2v.sgns_grads(W_in, W_out, np.array([d["c"]]), np.array([d["o"]]),
                                        d["neg"][None])
    np.add.at(W_in1, ri, -lr * gi)
    np.add.at(W_out1, ro, -lr * go)
    if not np.array_equal(W_in1[d["c"]], d["v1"]):
        raise RuntimeError("the whole-table update disagrees with sgns_step")
    d.update(W_in0=W_in, W_out0=W_out, W_in1=W_in1, W_out1=W_out1, negatives=negatives,
             step=int(run.snap_steps[snap]), steps=run.steps, centre=centre, context=context)
    return d


def demo_sentence(sents, centre=CENTRE, context=CONTEXT, window=w2v.WINDOW):
    """Index of the first corpus sentence in which ``context`` is within
    ``window`` of ``centre``: the pair the step trains on really occurs."""
    for n, s in enumerate(sents):
        for i, w in enumerate(s):
            if w == centre and context in s[max(0, i - window):i + window + 1]:
                return n
    raise ValueError("the pair never occurs")


def smooth(x, n):
    """Trailing mean of ``x`` over ``n`` steps (fewer at the start)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(0, i - n)
    return (c[i] - c[lo]) / (i - lo)


class Word2VecTraining(Diagram):
    """Skip-gram with negative sampling on the study's corpus (see the module docstring)."""

    PHASES = ("corpus", "one step", "training", "neighbours")

    def __init__(self):
        super().__init__("word2vec: skip-gram with negative sampling", "", 10.0)
        self.hold = 2.5
        run = self.run = w2v.train()
        words = run.words
        sents = w2v.corpus()
        self.listed = [" ".join(s) for s in sents[:N_LISTED]]
        self.demo = sents[demo_sentence(sents)]
        index = run.index
        self.demo_pairs = [(words[c], words[o]) for c, o in w2v.pairs([self.demo], index)]
        self.n_pairs = replay(STEP_SNAP)[3]
        self.st = training_step()
        st = self.st
        self.step_rows = [st["context"]] + list(st["negatives"])
        self.u0 = [st["W_out0"][st["o"]]] + [st["W_out0"][i] for i in st["neg"]]
        self.u1 = [st["W_out1"][st["o"]]] + [st["W_out1"][i] for i in st["neg"]]
        self.s0 = [st["s_pos0"]] + list(st["s_neg0"])
        self.s1 = [st["s_pos1"]] + list(st["s_neg1"])
        # the step does what it is meant to: the context's score up, every noise word's down
        self.as_meant = self.s1[0] > self.s0[0] and all(
            b < a for a, b in zip(self.s0[1:], self.s1[1:]))
        self.vmax = max(amax(np.array(self.u0 + self.u1)), amax(st["v0"]), amax(st["v1"]))
        # training: a fixed view of the category words, through every snapshot
        self.cat_idx = [i for i, w in enumerate(words) if w2v.category_of(w)]
        self.cats = [w2v.category_of(words[i]) for i in self.cat_idx]
        mu, axes = w2v.projection(run.W_in, words)
        self.xy = np.array([(w2v.unit(S)[self.cat_idx] - mu) @ axes.T for S in run.snaps])
        lo, hi = self.xy.reshape(-1, 2).min(0), self.xy.reshape(-1, 2).max(0)
        self.xy_mid, self.xy_span = (lo + hi) / 2, hi - lo
        self.cos = np.array([w2v.category_cosines(S, words) for S in run.snaps])
        self.loss = smooth(run.loss, LOSS_SMOOTH)
        self.peak = int(np.argmax(self.cos[:, 1]))           # the snapshot where all align
        self.near = {w: w2v.neighbours(run.W_in, words, w, N_NEAR) for w in NEAR_WORDS}
        self.near_same = all(w2v.category_of(n) == w2v.category_of(w)
                             for w, ns in self.near.items() for n, _c in ns)

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("sentences", 3.0, 0.3)
        add("window", 9.0, 1.2)
        add("rows", 3.0, 0.5)
        add("update", 4.5, 3.0)
        add("train", 16.0, 1.5)
        add("near", 7.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["rows"][0] - 0.5, Tm["train"][0] - 0.5, Tm["near"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        keys = ("corpus", "step", "train", "near")
        ends = self.marks[1:]
        op = {}
        for n, key in enumerate(keys):
            a = 1.0 if n == 0 else smoothstep(ends[n - 1] + 0.1, ends[n - 1] + 0.5, t)
            b = 1.0 if n == len(keys) - 1 else 1.0 - smoothstep(ends[n] - 0.6, ends[n], t)
            op[key] = a * b * back
        return op

    def _snap_at(self, t):
        """The (fractional) snapshot shown at time ``t`` of the training phase."""
        u = self.progress("train", t)
        return (len(self.run.snaps) - 1) * u ** SNAP_POWER

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        st = self.st
        f = self._snap_at(t) if reset == 0 else 0.0
        k = int(np.floor(f + 1e-9))
        same, cross = self.cos[k]
        sig = w2v.sigmoid
        near = "; ".join("%s → %s %.3f" % (w, ns[0][0], ns[0][1]) for w, ns in self.near.items())
        self.subtitle = [
            "a small generated corpus: %s sentences from %d templates; a window of %d words "
            "either side makes (centre, context) pairs, %s in all"
            % (format(w2v.N_SENTENCES, ","), len(w2v.TEMPLATES), w2v.WINDOW,
               format(self.n_pairs, ",")),
            "one step at training step %s: v = W_in[%s] is pulled towards u_%s (σ %.2f → %.2f) "
            "and pushed from %d noise words; the pair's loss %.2f → %.2f"
            % (format(st["step"], ","), st["centre"], st["context"], sig(st["s_pos0"]),
               sig(st["s_pos1"]), len(st["negatives"]), st["loss0"], st["loss1"]),
            "step %s of %s: the mean cosine between words of the same category is %s,\n"
            "between words of different categories %s" % (format(int(self.run.snap_steps[k]), ","),
                               format(self.run.steps, ","), num(same, 3), num(cross, 3)),
            ("after %s steps each word's nearest neighbours are from its own category: %s"
             if self.near_same else "after %s steps, the nearest words: %s")
            % (format(self.run.steps, ","), near),
        ][ph]
        if ops["corpus"] > 0.01:
            self._corpus(sc, t, ops["corpus"])
        if ops["step"] > 0.01:
            self._step(sc, t, ops["step"])
        if ops["train"] > 0.01:
            self._training(sc, f, ops["train"])
        if ops["near"] > 0.01:
            self._neighbours(sc, t, ops["near"])
        self.step_indicator(sc, ph, self.PHASES)

    # -- phase 1: the corpus and the window ---------------------------------------------------
    def _corpus(self, sc, t, op):
        """The first sentences of the corpus; one sentence with the window sliding along."""
        sen = self.progress("sentences", t)
        self.label(sc, self.P(0.0, 0.29), "the corpus, generated: its first %d sentences"
                   % N_LISTED, 0.028, anchor="lm", opacity=op * smoothstep(0.0, 0.15, sen),
                   ink=layout.STRONG_INK)
        for n, s in enumerate(self.listed):
            k = smoothstep(0.1 + 0.12 * n, 0.25 + 0.12 * n, sen)
            self.label(sc, self.P(0.0, 0.36 + 0.055 * n), s, 0.027, anchor="lm",
                       opacity=op * k)
        late = op * smoothstep(0.85, 1.0, sen)
        self.label(sc, self.P(0.0, 0.36 + 0.055 * N_LISTED), "… %s more" % format(
            w2v.N_SENTENCES - N_LISTED, ","), 0.027, anchor="lm", opacity=late,
            ink=layout.NOTE_INK)
        self.label(sc, self.P(0.0, 0.79), "slots filled from four word lists: animals,",
                   0.026, anchor="lm", opacity=late, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.0, 0.84), "people, foods, places (a toy: real text is",
                   0.026, anchor="lm", opacity=late, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.0, 0.89), "the same structure, buried in noise)",
                   0.026, anchor="lm", opacity=late, ink=layout.NOTE_INK)
        # the window, sliding along one sentence
        win = self.progress("window", t)
        show = op * smoothstep(0.0, 0.06, win)
        if show <= 0.01:
            return
        words = self.demo
        n = len(words)
        i = min(n - 1, int(remap(win, 0.04, 0.92) * n))
        xs = [0.50 + 0.074 * j for j in range(n)]
        y = 0.40
        self.label(sc, self.P(0.725, 0.29), "the window: %d words either side of the centre"
                   % w2v.WINDOW, 0.027, opacity=show, ink=layout.STRONG_INK)
        lo, hi = max(0, i - w2v.WINDOW), min(n - 1, i + w2v.WINDOW)
        a, b = self.P(xs[lo] - 0.034, y), self.P(xs[hi] + 0.034, y)
        w = float((self.px(b) - self.px(a))[0])
        self.rect(sc, (a + b) / 2, w, 34, DATA, show, border=DATA, border_w=1.2, radius=5,
                  lift=0.1, fill_op=0.12)
        self.rect(sc, self.P(xs[i], y), 58, 26, DATA, show, radius=4, lift=0.3, fill_op=0.35)
        sc.probe(("window",), self.P(xs[i], y), tol=6.0)
        for j, word in enumerate(words):
            ink = layout.STRONG_INK if j == i else (
                layout.CAPTION_INK if lo <= j <= hi else layout.NOTE_INK)
            self.label(sc, self.P(xs[j], y), word, 0.029, opacity=show, ink=ink)
        centre = words[i]
        mine = [(c, o) for (c, o), (ci, _oj) in zip(self.demo_pairs, self._pair_pos())
                if ci == i]
        self.label(sc, self.P(0.50, 0.49), "pairs with the centre \"%s\":" % centre, 0.027,
                   anchor="lm", dx=-30, opacity=show, ink=layout.CAPTION_INK)
        for m, (c, o) in enumerate(mine):
            col, row = divmod(m, 4)
            self.label(sc, self.P(0.50 + 0.24 * col, 0.555 + 0.055 * row), "(%s, %s)" % (c, o),
                       0.027, anchor="lm", dx=-30, opacity=show, kind="mono")
        done = sum(1 for ci, _oj in self._pair_pos() if ci <= i)
        self.label(sc, self.P(0.50, 0.81), "this sentence: %d words, %d of its %d pairs" % (
            n, done, len(self.demo_pairs)), 0.027, anchor="lm", dx=-30, opacity=show,
            ink=layout.STRONG_INK)
        self.label(sc, self.P(0.50, 0.865), "the corpus: %s pairs, read twice in batches of %d"
                   % (format(self.n_pairs, ","), w2v.BATCH), 0.027, anchor="lm", dx=-30,
                   opacity=show * smoothstep(0.92, 1.0, win), ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.50, 0.92), "= %s SGD steps" % format(self.run.steps, ","),
                   0.027, anchor="lm", dx=-30, opacity=show * smoothstep(0.92, 1.0, win),
                   ink=layout.CAPTION_INK)

    def _pair_pos(self):
        """(centre position, context position) of each of the demo sentence's pairs, in the
        order :func:`w2v.pairs` lists them."""
        n, out = len(self.demo), []
        for i in range(n):
            for j in range(max(0, i - w2v.WINDOW), min(n, i + w2v.WINDOW + 1)):
                if j != i:
                    out.append((i, j))
        return out

    # -- phase 2: one step --------------------------------------------------------------------
    def _strip(self, sc, c, v, op):
        """A 16-number vector as a strip of cells on the shared diverging scale."""
        tex = div_colors(v, self.vmax).reshape(1, len(v), 3)
        draw_image(self, sc, c, tex, STRIP_W, STRIP_H, op, grid=0.3)

    def _step(self, sc, t, op):
        """v, the six output vectors, their scores and sigmoids, before and after the step."""
        st = self.st
        rows = self.progress("rows", t)
        upd = ease_in_out(self.progress("update", t))
        sig = w2v.sigmoid
        name_x, strip_x, score_x, bar0, bar1 = 0.165, 0.32, 0.535, 0.62, 0.835
        top = 0.30
        v = st["v0"] + (st["v1"] - st["v0"]) * upd
        self.label(sc, self.P(name_x, top), "v = W_in[%s]" % st["centre"], 0.028, anchor="rm",
                   opacity=op, ink=layout.STRONG_INK)
        self._strip(sc, self.P(strip_x, top), v, op)
        self.label(sc, self.P(0.48, top), "training step %s, learning rate %.2f" % (
            format(st["step"], ","), st["lr"]), 0.027, anchor="lm", opacity=op,
            ink=layout.CAPTION_INK)
        hy = 0.39
        for x, h in ((strip_x, "output vectors u = W_out[word]"), (score_x, "u · v"),
                     ((bar0 + bar1) / 2, "σ(u · v)")):
            self.label(sc, self.P(x, hy), h, 0.026, opacity=op * smoothstep(0.0, 0.2, rows),
                       ink=layout.NOTE_INK)
        for r, word in enumerate(self.step_rows):
            k = op * smoothstep(0.1 + 0.12 * r, 0.3 + 0.12 * r, rows)
            if k <= 0.01:
                continue
            y = 0.46 + 0.068 * r
            role = "context" if r == 0 else "noise"
            ink = PULL_INK if r == 0 else PUSH_INK
            self.label(sc, self.P(0.0, y), role, 0.026, anchor="lm", opacity=k,
                       ink=text_ink(ink))
            self.label(sc, self.P(name_x, y), word, 0.028, anchor="rm", opacity=k,
                       ink=layout.STRONG_INK)
            u = self.u0[r] + (self.u1[r] - self.u0[r]) * upd
            self._strip(sc, self.P(strip_x, y), u, k)
            s0, s1 = self.s0[r], self.s1[r]
            text = num(s0) if upd <= 0 else "%s → %s" % (num(s0), num(s1))
            self.label(sc, self.P(score_x, y), text, 0.027, opacity=k, kind="mono",
                       ink=layout.STRONG_INK)
            # σ as a bar: the track, the value (moving), a tick where it started
            a, b = self.P(bar0, y), self.P(bar1, y)
            wpx = float((self.px(b) - self.px(a))[0])
            self.rect(sc, (a + b) / 2, wpx, 10, PANEL, k, border=PANEL_EDGE, border_w=1.0,
                      radius=2, lift=0.1)
            p = sig(s0) + (sig(s1) - sig(s0)) * upd
            fill_c = self.shift(a, wpx * p / 2, 0)
            self.rect(sc, fill_c, max(wpx * p, 2.0), 8, ink, k, radius=2, lift=0.3)
            tick = self.shift(a, wpx * sig(s0), 0)
            sc.line(self.shift(tick, 0, -8), self.shift(tick, 0, 8), width=1.4, color=EDGE,
                    opacity=k * smoothstep(0.0, 0.1, upd), bias=0.6)
            sc.probe(("sig", r), (wpx * p, 0, 0), tol=4.0)
            self.label(sc, b, ("%.2f" % sig(s0)) if upd <= 0 else "%.2f → %.2f" % (
                sig(s0), sig(s1)), 0.026, anchor="lm", dx=8, opacity=k, kind="mono")
        sc.probe(("upd",), (upd * 40, 0, 0), tol=4.0)
        fin = op * smoothstep(0.0, 0.3, self.progress("update", t))
        self.label(sc, self.P(0.5, 0.89), "loss = −log σ(u_%s · v) − Σ log σ(−u_noise · v):  "
                   "%.3f → %.3f" % (st["context"], st["loss0"], st["loss1"]) if upd > 0.5 else
                   "loss = −log σ(u_%s · v) − Σ log σ(−u_noise · v):  %.3f" % (
                       st["context"], st["loss0"]), 0.028, opacity=op * smoothstep(0.6, 1.0, rows),
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.945), "the step raises σ for the true context and lowers "
                   "it for the noise words", 0.026,
                   opacity=fin * smoothstep(0.6, 1.0, upd) * self.as_meant,
                   ink=layout.NOTE_INK)

    # -- phase 3: training --------------------------------------------------------------------
    def _training(self, sc, f, op):
        """The category words in the fixed view, through the snapshots; the cosine and loss
        curves with a cursor at the current step."""
        n_snap = len(self.run.snaps)
        k = min(int(np.floor(f + 1e-9)), n_snap - 1)
        a = f - k
        xy = self.xy[k] if k + 1 >= n_snap else self.xy[k] + (self.xy[k + 1] - self.xy[k]) * a
        # the scatter panel
        x0, y0, x1, y1 = 0.0, 0.33, 0.52, 0.97
        pa, pb = self.P(x0, y0), self.P(x1, y1)
        w, h = (self.px(pb) - self.px(pa))
        centre = (pa + pb) / 2
        self.rect(sc, centre, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        self.label(sc, self.P(x0, 0.285), "the %d category words' vectors (unit length), one "
                   "fixed 2-D view" % len(self.cat_idx), 0.026, anchor="lm", opacity=op,
                   ink=layout.CAPTION_INK)
        scale = min((w - 40) / self.xy_span[0], (h - 40) / self.xy_span[1])

        def at(p):
            d = (p - self.xy_mid) * scale
            return self.shift(centre, d[0], -d[1])

        for (p, cat) in zip(xy, self.cats):
            self.disk(sc, at(p), 4.5, CAT_INKS[cat], op, lift=0.8, outline=0.25)
        sc.probe(("cloud",), at(xy[0]), tol=4.0)
        sc.probe(("cloud", 1), at(xy[len(xy) // 2]), tol=4.0)
        # category names beside their clusters, once they have formed
        named = op * smoothstep(14.0, 22.0, f)
        for cat in CAT_ORDER:
            pts = xy[[n for n, c in enumerate(self.cats) if c == cat]]
            cen = pts.mean(0)
            side = LABEL_SIDE[cat]
            if side == "right":
                pos, anchor = self.shift(at((pts[:, 0].max(), cen[1])), 12, 0), "lm"
            elif side == "above":
                pos, anchor = self.shift(at((cen[0], pts[:, 1].max())), 0, -12), "mb"
            else:
                pos, anchor = self.shift(at((cen[0], pts[:, 1].min())), 0, 12), "mt"
            fx, fy = self.px(pos) - self.px(centre)
            pos = self.shift(centre, float(np.clip(fx, -w / 2 + 40, w / 2 - 40)),
                             float(np.clip(fy, -h / 2 + 24, h / 2 - 24)))
            self.label(sc, pos, CAT_NAMES[cat], 0.028, anchor=anchor, opacity=named,
                       ink=text_ink(CAT_INKS[cat]))
        peak_step = int(self.run.snap_steps[self.peak])
        early = op * smoothstep(1.0, 2.0, f) * (1 - smoothstep(6.0, 9.0, f))
        self.label(sc, self.P(0.26, 0.87), "first they all swing one way:", 0.026,
                   opacity=early, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.26, 0.915), "cosine across categories %.4f at step %s" % (
            self.cos[self.peak, 1], format(peak_step, ",")), 0.026, opacity=early,
            ink=layout.NOTE_INK)
        # the curves
        steps = self.run.snap_steps
        total = self.run.steps
        now = float(steps[k] + (steps[min(k + 1, n_snap - 1)] - steps[k]) * a)
        X0, X1 = 0.62, 0.985

        def plot(box_y0, box_y1, xs, ys, lo, hi, ink, width=1.8):
            def pt(x, yv):
                return self.P(X0 + (X1 - X0) * x / total,
                              box_y1 - (box_y1 - box_y0) * (yv - lo) / (hi - lo), 0.3)
            keep = xs <= now
            if keep.sum() > 1:
                self.polyline(sc, [pt(x, yv) for x, yv in zip(xs[keep], ys[keep])], width,
                              ink, op)
            return pt

        def frame(box_y0, box_y1, lo, hi, ticks):
            pa, pb = self.P(X0, box_y0), self.P(X1, box_y1)
            ww, hh = self.px(pb) - self.px(pa)
            self.rect(sc, (pa + pb) / 2, ww + 12, hh + 12, PANEL, op, border=PANEL_EDGE,
                      border_w=1.0, radius=4, lift=0.1)
            for v in ticks:
                yy = box_y1 - (box_y1 - box_y0) * (v - lo) / (hi - lo)
                self.label(sc, self.P(X0, yy), num(v, 1), 0.025, anchor="rm", dx=-10,
                           opacity=op, ink=layout.NOTE_INK, kind="mono")
            cx = X0 + (X1 - X0) * now / total
            sc.line(self.P(cx, box_y0, 0.3), self.P(cx, box_y1, 0.3), width=1.0, color=EDGE,
                    opacity=0.6 * op, bias=0.3)

        cy0, cy1 = 0.36, 0.52
        frame(cy0, cy1, 0.0, 1.0, (0.0, 1.0))
        plot(cy0, cy1, steps.astype(float), self.cos[:, 0], 0.0, 1.0, SAME_INK)
        plot(cy0, cy1, steps.astype(float), self.cos[:, 1], 0.0, 1.0, CROSS_INK)
        sc.probe(("cursor",), self.P(X0 + (X1 - X0) * now / total, cy0), tol=4.0)
        self.label(sc, self.P(X0, 0.285), "mean cosine:", 0.026, anchor="lm", dx=-12,
                   opacity=op, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.745, 0.285), "same category %s" % num(self.cos[k, 0], 3), 0.026,
                   anchor="lm", opacity=op, ink=SAME_INK)
        self.label(sc, self.P(0.745, 0.325), "different %s" % num(self.cos[k, 1], 3), 0.026,
                   anchor="lm", opacity=op, ink=CROSS_INK)
        ly0, ly1 = 0.63, 0.735
        lo_l, hi_l = 2.0, float(np.ceil(self.loss.max() * 2) / 2)
        frame(ly0, ly1, lo_l, hi_l, (lo_l, hi_l))
        idx = np.unique(np.linspace(0, total - 1, 240).astype(int))
        plot(ly0, ly1, idx.astype(float), self.loss[idx], lo_l, hi_l, DATA)
        li = min(total - 1, max(0, int(now) - 1))
        self.label(sc, self.P(X0, 0.585), "loss, mean of %d steps: %.2f" % (
            LOSS_SMOOTH, self.loss[li]), 0.026, anchor="lm", dx=-12, opacity=op,
            ink=layout.CAPTION_INK)
        self.label(sc, self.P(X0, ly1), "0", 0.025, anchor="lt", dy=12, opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(X1, ly1), "%s steps" % format(total, ","), 0.025, anchor="rt",
                   dy=12, opacity=op, ink=layout.NOTE_INK)
        # the legend
        for n, cat in enumerate(CAT_ORDER):
            y = 0.845 + 0.04 * n
            self.disk(sc, self.P(X0 - 0.03, y), 4.5, CAT_INKS[cat], op, lift=0.8)
            ws = [self.run.words[i] for i, c in zip(self.cat_idx, self.cats) if c == cat]
            self.label(sc, self.P(X0 - 0.015, y), "%s: %s" % (CAT_NAMES[cat], " ".join(ws)),
                       0.025, anchor="lm", opacity=op, ink=text_ink(CAT_INKS[cat]))

    # -- phase 4: nearest neighbours ----------------------------------------------------------
    def _neighbours(self, sc, t, op):
        """Each word's nearest five by cosine, with a bar per cosine."""
        u = self.progress("near", t)
        self.label(sc, self.P(0.5, 0.29), "nearest words by cosine similarity, final vectors",
                   0.027, opacity=op, ink=layout.CAPTION_INK)
        for c, (word, ns) in enumerate(self.near.items()):
            x = 0.12 + 0.253 * c
            cat = w2v.category_of(word)
            k0 = op * smoothstep(0.04 * c, 0.04 * c + 0.12, u)
            self.label(sc, self.P(x, 0.37), word, 0.031, opacity=k0,
                       ink=text_ink(CAT_INKS[cat]))
            for r, (nb, cv) in enumerate(ns):
                k = op * smoothstep(0.1 + 0.06 * r + 0.02 * c, 0.22 + 0.06 * r + 0.02 * c, u)
                y = 0.45 + 0.075 * r
                self.label(sc, self.P(x - 0.09, y), nb, 0.027, anchor="lm", opacity=k,
                           ink=text_ink(CAT_INKS[w2v.category_of(nb)]))
                self.label(sc, self.P(x + 0.095, y), "%.3f" % cv, 0.027, anchor="rm",
                           opacity=k, kind="mono", ink=layout.STRONG_INK)
                a = self.P(x - 0.09, y + 0.028)
                full = float((self.px(self.P(x + 0.095, y)) - self.px(a))[0])
                wpx = max(2.0, full * max(cv, 0.0))
                self.rect(sc, self.shift(a, wpx / 2, 0), wpx, 4, CAT_INKS[w2v.category_of(nb)],
                          k * 0.8, radius=1.5, lift=0.3)
        late = op * smoothstep(0.55, 0.7, u)
        if self.near_same:
            self.label(sc, self.P(0.5, 0.86), "every neighbour is from the word's own category: "
                       "they fill the same slots, so they share contexts", 0.027, opacity=late)
            self.label(sc, self.P(0.5, 0.915), "training never saw the categories, only which "
                       "words occur near which", 0.026, opacity=late, ink=layout.NOTE_INK)
        sc.probe(("near",), (u * 40, 0, 0), tol=4.0)

