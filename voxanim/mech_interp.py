"""Looking inside a trained model: animations 43-50 of the "next 100".

Each draws the mechanism first and the measurement after it. The numbers are
the study's (:mod:`voxanim.next_language`); what a study did not keep (the
weights it trained, per-token activations, every optimisation step) is
computed again here by the same code path, seeded the same, and the tests
hold the two to each other. Four of them (44, 47, 48 and 50) draw the
redesigned experiments of :mod:`voxanim.study_interp`, which keeps everything
they need: per-passage ablations, held-out feature contexts, every step of
the ascents, each seed's training curve.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import layout
from . import next_language as NL
from . import study_interp as SI
from .cache import cached_npz
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    GRID,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    heat,
    number,
    soft,
)

DATA_VERSION = 1
INK5 = (RED, BLUE, GREEN, AMBER, VIOLET)


def show(token):
    """A GPT-2 token (or a character) as it is drawn in a box."""
    s = str(token).replace("\n", "↵")
    return "␣" if s == " " else s.strip() or "↵"


def spread(words, x0, x1, size, pad=0.016, least=0.04, left=False):
    """Centres and widths (frame fractions) of a row of boxes from ``x0`` to
    ``x1``, each as wide as its word needs, the row centred in that span."""
    ws = [max(layout.text_width(w, size) + pad, least) for w in words]
    gap = 0.007
    total = sum(ws) + gap * (len(ws) - 1)
    if total > x1 - x0:
        ws = [w * (x1 - x0 - gap * (len(ws) - 1)) / sum(ws) for w in ws]
        total = x1 - x0
    x = x0 if left else (x0 + x1 - total) / 2
    xs = []
    for w in ws:
        xs.append(x + w / 2)
        x += w + gap
    return xs, ws


class Inside(Mechanism):
    """What the animations of this module share."""

    def row(self, sc, words, x0, x1, y, op, inks=None, h=46.0, size=0.038, lit=None,
            dim=(), left=False):
        """A row of token boxes, each as wide as its word; ``inks``: index ->
        colour (filled soft); ``lit``: index -> colour (filled solid, white
        text); ``dim``: indices drawn faint. Returns the centre xs."""
        xs, ws = spread(words, x0, x1, size, left=left)
        for i, (word, x, w) in enumerate(zip(words, xs, ws)):
            px = self.span(w)[0]
            if lit and i in lit:
                self.box(sc, x, y, px, h, lit[i], op, word, ink=WHITE, size=size, glow=0.12)
            elif inks and i in inks:
                self.box(sc, x, y, px, h, soft(inks[i]), op, word, ink=dark(inks[i], 0.35),
                         border=inks[i], size=size)
            else:
                self.box(sc, x, y, px, h, PAPER, op * (0.4 if i in dim else 1.0), word,
                         border=AXIS, size=size)
        return xs

    def outline(self, sc, fx, fy, w, h, ink, op, width=2.4, radius=3.0):
        """An empty ``w x h`` px frame centred at ``(fx, fy)``."""
        self.rect(sc, self.P(float(fx), float(fy)), w, h, ink, op, border=ink, border_w=width,
                  radius=radius, lift=0.8, fill_op=0.0)

    def image(self, sc, rgb, fx, fy, w, h, op, frame=True, lines=0.0):
        """An RGB array as a ``w x h`` px picture centred at ``(fx, fy)``."""
        c = self.P(float(fx), float(fy))
        if frame:
            self.rect(sc, c, w + 8, h + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5,
                      lift=0.1)
        draw_image(self, sc, c, np.asarray(rgb, np.float32), w, h, op, grid=lines)

    def over(self, sc, a, b, ink, op, u=1.0, bend=0.08, width=2.4, lift=3.0):
        """:meth:`arc`, but drawn above pictures and boxes instead of under them."""
        if op <= 0.01 or u <= 0.0:
            return
        x, y = self._curve_points(a, b, bend)
        m = max(2, int(np.ceil(len(x) * min(u, 1.0))))
        pts = [self.P(float(px), float(py), lift) for px, py in zip(x[:m], y[:m])]
        self.polyline(sc, pts, width, ink, op, bias=0.3)
        self._head(sc, pts[-2], pts[-1], ink, op, width)

    def at(self, fx, fy, w, h, rows, cols):
        """``cell(row, col) -> (fx, fy)`` of a ``w x h`` px picture centred at ``(fx, fy)``."""
        dw, dh = self.frac(w, h)
        return lambda r, k: (fx - dw / 2 + (k + 0.5) * dw / cols,
                             fy - dh / 2 + (r + 0.5) * dh / rows)


def _settings(name, **extra):
    return dict(version=DATA_VERSION, study_version=NL.VERSION, seed=NL.SEED, name=name, **extra)


# ---------------------------------------------------------------------- 43 superposition
SUPER_Q = (0.1, 0.5, 0.9, 0.98)
SUPER_AT = ((2, 2), (2, 9), (6, 6), (9, 2), (9, 9))


def superposition_features(images):
    """The study's five features: mean intensity of five 3 x 3 patches of a 14 x 14 digit."""
    images = np.asarray(images).reshape(-1, 14, 14)
    return np.stack([images[:, i:i + 3, j:j + 3].mean((1, 2)) for i, j in SUPER_AT], 1)


def superposition_runs():
    """The study's four trainings again, keeping what it dropped: the feature
    directions every 8 updates (``W``: quantile, snapshot, 2, 5), the final
    bias, the thresholds, and one held-out digit to follow through the model."""
    from . import realdata
    from .optim import Adam

    def build():
        x, _, _ = realdata.mnist_split(2048, "train", 14, NL.SEED)
        xt, _, _ = realdata.mnist_split(512, "test", 14, NL.SEED + 1)
        raw, held = superposition_features(x), superposition_features(xt)
        Ws, bs, ths, held_loss = [], [], [], []
        for q in SUPER_Q:
            threshold = np.quantile(raw, q, axis=0)
            data = np.where(raw > threshold, raw, 0)
            test = np.where(held > threshold, held, 0)
            rng = np.random.default_rng(NL.SEED)
            P = {"W": rng.normal(0, 0.3, (2, 5)), "b": np.zeros(5)}
            opt = Adam(P, 0.01)
            snaps = [P["W"].copy()]
            for step in range(800):
                ix = rng.integers(len(data), size=128)
                _, g = NL.superposition_loss(P, data[ix])
                opt.step(P, g)
                if step % 8 == 7:
                    snaps.append(P["W"].copy())
            Ws.append(snaps)
            bs.append(P["b"].copy())
            ths.append(threshold)
            held_loss.append(NL.superposition_loss(P, test)[0])
        # the held-out digit with the most patches lit (the first such)
        pick = int(np.argmax((held > ths[0]).sum(1)))
        return dict(W=np.asarray(Ws), b=np.asarray(bs), thresholds=np.asarray(ths),
                    heldout_loss=np.asarray(held_loss), image=xt[pick].reshape(14, 14),
                    features=held[pick])
    return cached_npz("mech-interp-superposition", _settings("superposition"), build)


# ---------------------------------------------------------------------- 45 linear probes
def probe_run():
    """The study's probes trained again, keeping their answers: every held-out
    token's score under each layer's punctuation probe, with its true label
    (and the control's accuracies, to hold this run to the study's)."""
    from . import gpt2_hooks

    def build():
        tok = NL._tok()

        def data(split, count):
            text = NL._corpus()[split]
            features, labels, controls, tokens = [[] for _ in range(13)], [], [], []
            for start in np.linspace(0, len(text) - 160, count).astype(int):
                ids = tok.encode(text[start:start + 160])
                r = gpt2_hooks.forward(ids, record=True)
                for i in range(1, len(ids)):
                    previous = tok.decode([ids[i - 1]])
                    labels.append(float(any(ch in previous for ch in ".,;:!?")))
                    controls.append(float((ids[i] * 2654435761 % 11) < 5))
                    tokens.append(tok.decode([ids[i]]))
                    for layer in range(13):
                        features[layer].append(r["resid"][layer, i])
            return ([np.asarray(v) for v in features], np.asarray(labels),
                    np.asarray(controls), tokens)

        train, y, cy, _ = data("train", 24)
        test, yt, ct, tokens = data("test", 8)
        first = NL._tok().decode([tok.encode(NL._corpus()["test"][:160])[0]])

        def balanced(pred, target):
            return np.mean([(pred[target == v] == v).mean() for v in (0, 1)])

        scores, accuracy, control = [], [], []
        for tr, te in zip(train, test):
            mean, scale = tr.mean(0), tr.std(0) + 0.1
            tr = np.c_[(tr - mean) / scale, np.ones(len(tr))]
            te = np.c_[(te - mean) / scale, np.ones(len(te))]
            for target, testtarget, out in ((y, yt, accuracy), (cy, ct, control)):
                w = np.zeros(tr.shape[1])
                sample = np.where(target > 0, 0.5 / max(target.mean(), 1e-6),
                                  0.5 / max(1 - target.mean(), 1e-6))
                for _ in range(120):
                    p = 1 / (1 + np.exp(np.clip(-tr @ w, -60, 60)))
                    w -= 0.1 * (tr.T @ ((p - target) * sample) / len(tr) + 0.05 * w)
                out.append(balanced(te @ w > 0, testtarget))
                if out is accuracy:
                    scores.append(te @ w)
        return dict(scores=np.asarray(scores), labels=yt, controls=ct,
                    accuracy=np.asarray(accuracy), control=np.asarray(control),
                    tokens=np.asarray(tokens), first=np.asarray(first),
                    train_tokens=np.asarray(len(y)))
    return cached_npz("mech-interp-probes", _settings("probes"), build)


# ---------------------------------------------------------------------- 46 activation patching
@lru_cache(maxsize=1)
def patching_prompts():
    """The study's two prompts as GPT-2 tokens: ``(clean, corrupted, target)`` strings."""
    tok = NL._tok()
    observed = tok.encode(NL._corpus()["test"][:160])
    clean = list(observed[:9])
    corrupt = list(clean)
    corrupt[2:4] = tok.encode(NL._corpus()["train"][200:300])[:2]
    return ([tok.decode([i]) for i in clean], [tok.decode([i]) for i in corrupt],
            tok.decode([observed[9]]))


# ---------------------------------------------------------------------- 47 induction heads
def realdata_char(i):
    """Character ``i`` of the Tiny Shakespeare vocabulary."""
    from . import realdata

    return realdata.decode([i])


def successor_targets(ids):
    """For each position ``t``: the positions just after an earlier copy of
    token ``t`` (the study's rule: the copy is at least two back)."""
    return {t: [j + 1 for j in range(t - 1) if ids[j] == ids[t]] for t in range(2, len(ids))
            if any(ids[j] == ids[t] for j in range(t - 1))}


def induction_run():
    """GPT-2's attention on the study's 220 held-out characters: the tokens,
    every head's attention from each token, as the study measured it (the
    study kept only each head's average)."""
    def build():
        tok = NL._tok()
        text = NL._corpus()["test"][:220]
        ids = tok.encode(text)
        attn = np.asarray(NL._forward(text, record=True)["attn"])
        targets = successor_targets(ids)
        mass = np.mean([attn[..., t, js].sum(-1) for t, js in targets.items()], axis=0)
        best = np.argsort(-mass.ravel())[:6]
        return dict(tokens=np.asarray([tok.decode([i]) for i in ids]), ids=np.asarray(ids),
                    mass=mass, best=best,
                    attention=np.asarray([attn[b // 12, b % 12] for b in best]))
    return cached_npz("mech-interp-induction", _settings("induction"), build)


# ---------------------------------------------------------------------- 49 saliency
PATH_STEPS = 512


@lru_cache(maxsize=1)
def saliency_run():
    """The study's attributions as numbers (it kept them as colours), and the
    path integrated gradients walks: the target logit at each of the 512
    midpoints from black to the digit, and the attribution summed so far."""
    from . import mnist

    params, _ = mnist.train()
    _, _, test, y = mnist.load()
    x, target = test[0], int(y[0])
    gradient = NL._input_gradient(params, x[None], target)[0]
    alphas = (np.arange(PATH_STEPS) + 0.5) / PATH_STEPS
    grads = NL._input_gradient(params, alphas[:, None] * x[None], target)
    running = x * np.cumsum(grads, 0) / PATH_STEPS            # attribution after each step
    ends = np.linspace(0.0, 1.0, PATH_STEPS + 1)
    _, zs, _ = mnist._forward(params, ends[:, None] * x[None])
    return dict(x=x, target=target, gradient=gradient, times_input=gradient * x,
                ig=NL.integrated_gradients(params, x, target, PATH_STEPS), running=running,
                alphas=ends, path_logit=zs[-1][:, target], logits=zs[-1][-1])


# ====================================================================== 43
class Superposition(Inside):
    """Five features of a digit squeezed into two numbers: each feature's
    direction in the plane of those two numbers, moving as the model trains,
    for inputs that are rarely zero and for inputs that are mostly zero."""

    TITLE = "Superposition"
    PHASES = (("squeeze", 10.0), ("dense", 11.0), ("sparse", 12.0), ("interference", 12.0),
              ("measured", 10.0))
    LIM = 1.35                           # the plane shows -LIM..LIM on both axes

    def setup(self):
        s = self.study
        d = superposition_runs()
        self.W = d["W"]                                  # quantile, snapshot, 2, 5
        self.b = d["b"]
        self.image, self.features = d["image"], d["features"]
        self.zero = [float(v) for v in s.metadata["actual_zero_fraction"]]
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.gram = np.asarray(s.panels[2].data["values"], float)
        self.final = [np.asarray(s.panels[0].data["xy"], float).T, self.W[1, -1], self.W[2, -1],
                      np.asarray(s.panels[1].data["xy"], float).T]
        # the dense model on the one digit: in, the two numbers, out
        x = np.where(self.features > d["thresholds"][0], self.features, 0.0)
        self.x, self.h = x, self.final[0] @ x
        self.out = np.maximum(self.h @ self.final[0] + self.b[0], 0.0)
        angles = np.sort(np.degrees(np.arctan2(self.final[3][1], self.final[3][0])) % 360)
        self.gaps = np.diff(np.r_[angles, angles[0] + 360])
        off = self.gram[~np.eye(5, dtype=bool)]
        self.leak, self.clash = float(off.max()), float(off.min())

    def subtitle_for(self, k):
        z, h = self.zero, self.heldout
        return [
            "five features of a digit, the brightness of five patches, are squeezed into two "
            "numbers and rebuilt from them: two numbers cannot hold five, so the rebuilt ones "
            "are off",
            "each feature gets a direction in the plane of the two numbers; with %.0f%% of the "
            "values zero, training leaves two of them short: lengths %.2f and %.2f" % (
                (100 * z[0],) + tuple(np.sort(np.linalg.norm(self.final[0], axis=0))[:2])),
            "with %.0f%% of the values zero the five directions spread into a rough pentagon, "
            "neighbours %.0f to %.0f degrees apart, every length between %.2f and %.2f" % (
                100 * z[3], self.gaps.min(), self.gaps.max(),
                np.linalg.norm(self.final[3], axis=0).min(),
                np.linalg.norm(self.final[3], axis=0).max()),
            "the price is interference: one feature switched on reads as up to %.2f on a "
            "neighbour; readings as low as %.2f on the others are cut to zero" % (
                self.leak, self.clash),
            "held-out error falls from %.3f to %.4f as zeros go from %.0f%% to %.0f%%; part of "
            "that fall is only that mostly-zero inputs are easier to rebuild" % (
                h[0], h[3], 100 * z[0], 100 * z[3]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _plane(self, sc, W, cx, cy, R, op, lit=None, width=3.2, names=True):
        """The plane of the two hidden numbers with each feature's direction
        (a column of ``W``) as an arrow from the origin. Returns the tips."""
        dx, dy = self.frac(R, R)
        self.rect(sc, self.P(cx, cy), 2 * R + 10, 2 * R + 10, PAPER, op, border=AXIS,
                  border_w=1.2, radius=6, lift=-2.0)
        for a, b in (((cx - dx, cy), (cx + dx, cy)), ((cx, cy - dy), (cx, cy + dy))):
            self.polyline(sc, [self.P(*a), self.P(*b)], 1.0, GRID, 0.8 * op, bias=0.0)
        ring = np.linspace(0, 2 * np.pi, 49)
        self.polyline(sc, [self.P(cx + np.cos(t) * dx / self.LIM, cy + np.sin(t) * dy / self.LIM)
                           for t in ring], 1.0, GRID, 0.8 * op, bias=0.0)
        tips = []
        for i in range(5):
            tip = (cx + W[0, i] / self.LIM * dx, cy - W[1, i] / self.LIM * dy)
            on = lit is None or i in lit
            self.arrow(sc, (cx, cy), tip, INK5[i], op * (1.0 if on else 0.22), width=width)
            tips.append(tip)
        return tips

    def _lengths(self, sc, W, x, y, op, title):
        """How long each feature's direction is, as bars."""
        self.text(sc, x, y, title, op, anchor="lm", strong=True, size=0.034)
        full = self.span(0.26)[0]
        for i in range(5):
            n = float(np.linalg.norm(W[:, i]))
            self.hbar(sc, x + 0.075, y + 0.075 + i * 0.066, full * n / self.LIM, INK5[i], op,
                      "patch %d" % (i + 1), "%.2f" % n, thick=24.0, track=full)

    def _digit(self, sc, x, y, side, op):
        """The digit with its five patches outlined; returns the patch centres."""
        self.digit(sc, self.image, x, y, side, op, frame=PAPER)
        cell = self.at(x, y, side, side, 14, 14)
        centres = []
        for k, (i, j) in enumerate(SUPER_AT):
            c = cell(i + 1, j + 1)
            self.outline(sc, c[0], c[1], side * 3 / 14, side * 3 / 14, INK5[k], op, 3.0)
            centres.append(c)
        return centres

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        centres = self._digit(sc, 0.16, 0.63, 220.0, op)
        self.text(sc, 0.16, 0.32, "a digit, five patches", op, strong=True, size=0.034)
        cols = (0.385, 0.615, 0.845)
        heads = ("5 features", "2 numbers", "5 rebuilt")
        for x, head in zip(cols, heads):
            self.text(sc, x, 0.32, head, op, strong=True, size=0.034)
        ys = [0.41 + 0.11 * i for i in range(5)]
        hy = (0.575, 0.685)
        a = smoothstep(0.05, 0.30, u)                  # features read off the digit
        b = smoothstep(0.38, 0.60, u)                  # squeezed
        c = smoothstep(0.68, 0.90, u)                  # rebuilt
        for i, y in enumerate(ys):
            self.arrow(sc, (centres[i][0] + 0.03, centres[i][1]), (cols[0] - 0.065, y), INK5[i],
                       op * 0.8, a, width=1.8)
            self.box(sc, cols[0], y, 104, 40, soft(INK5[i]), op * a, "%.2f" % self.x[i],
                     ink=dark(INK5[i], 0.4), border=INK5[i], size=0.038)
            for k in range(2):
                p, q = (cols[0] + 0.062, y), (cols[1] - 0.062, hy[k])
                self.arrow(sc, p, q, DIM, op * 0.9 * b, b, width=1.4, head=False)
                self.pulse(sc, p, q, (u - 0.38) / 0.22, INK5[i], op)
                p, q = (cols[1] + 0.062, hy[k]), (cols[2] - 0.062, y)
                self.arrow(sc, p, q, DIM, op * 0.9 * c, c, width=1.4, head=False)
                self.pulse(sc, p, q, (u - 0.68) / 0.22, TEAL, op)
            off = abs(self.out[i] - self.x[i])
            self.box(sc, cols[2], y, 104, 40, PAPER, op * c, "%.2f" % self.out[i],
                     border=INK5[i], size=0.038)
            self.text(sc, cols[2] + 0.062, y, "off %.2f" % off, op * smoothstep(0.88, 0.97, u),
                      anchor="lm", size=0.030)
        for k in range(2):
            self.box(sc, cols[1], hy[k], 104, 46, TEAL, op * b, "%.2f" % self.h[k], ink=WHITE,
                     size=0.040)

    def _training(self, sc, u, op, q, label):
        snaps = self.W[q]
        k = min(u / 0.72, 1.0) * (len(snaps) - 1)
        i = int(min(k, len(snaps) - 2))
        W = snaps[i] + (snaps[i + 1] - snaps[i]) * (k - i)
        tips = self._plane(sc, W, 0.27, 0.645, 140.0, op)
        self.text(sc, 0.27, 0.32, "the plane of the two numbers", op, strong=True, size=0.034)
        self.text(sc, 0.62, 0.32, label, op, anchor="lm", strong=True, size=0.034)
        self.text(sc, 0.62, 0.375, "update %d of 800" % round(k * 8), op, anchor="lm",
                  size=0.034)
        self._lengths(sc, W, 0.545, 0.46, op, "length of each feature's direction")
        return tips, min(u / 0.72, 1.0)

    def phase_2(self, sc, u, op):
        self._training(sc, u, op, 0, "%.0f%% of the values are zero" % (100 * self.zero[0]))
        self.text(sc, 0.545, 0.89, "short arrow: that feature is barely stored", op *
                  smoothstep(0.75, 0.9, u), anchor="lm", size=0.032)

    def phase_3(self, sc, u, op):
        tips, done = self._training(sc, u, op, 3,
                                    "%.0f%% of the values are zero" % (100 * self.zero[3]))
        show = op * smoothstep(0.76, 0.9, u)
        order = np.argsort(np.arctan2(self.final[3][1], self.final[3][0]))
        if done >= 1.0:
            ring = [tips[i] for i in order] + [tips[order[0]]]
            self.polyline(sc, [self.P(*p) for p in ring], 1.6, AXIS, 0.7 * show, bias=0.1)
        self.text(sc, 0.545, 0.89, "five directions, %.0f to %.0f degrees apart" % (
            self.gaps.min(), self.gaps.max()), show, anchor="lm", size=0.032)

    def phase_4(self, sc, u, op):
        turn = min(int(u * 5 * 1.04), 4)
        within = min(u * 5 * 1.04 - turn, 1.0)
        W = self.final[3]
        tips = self._plane(sc, W, 0.235, 0.645, 140.0, op, lit=[turn])
        self.text(sc, 0.235, 0.32, "patch %d switched on" % (turn + 1), op, strong=True,
                  size=0.034, ink=dark(INK5[turn], 0.3))
        mx, my, side = 0.62, 0.655, 235.0
        cell = self.grid(sc, self.gram, mx, my, side, side, op, reveal=(turn + 1) / 5,
                         ring=[(turn, turn)], ring_ink=INK5[turn])
        self.text(sc, mx, 0.32, "what each feature then reads", op, strong=True, size=0.034)
        for r in range(turn + 1):
            for k in range(5):
                shown = op * (smoothstep(0.15, 0.5, within) if r == turn else 1.0)
                self.text(sc, *cell(r, k), "%+.2f" % self.gram[r, k], shown, size=0.030,
                          ink=WHITE if abs(self.gram[r, k]) > 0.8 else None)
        for k in range(5):
            x, y = cell(0, k)
            self.box(sc, x, my - 0.25, 30, 12, INK5[k], op, radius=3)
            x, y = cell(k, 0)
            self.box(sc, mx - 0.152, y, 14, 30, INK5[k], op, radius=3)
            if k != turn:
                # the switched-on direction's shadow on each other one
                self.pulse(sc, tips[turn], tips[k], (within - 0.05) / 0.5, INK5[turn], op,
                           bend=0.03)
        row = self.gram[turn]
        leak = [(k, row[k]) for k in range(5) if k != turn and row[k] > 0]
        cut = [k for k in range(5) if k != turn and row[k] <= 0]
        note = op * smoothstep(0.45, 0.7, within)
        self.lines(sc, 0.775, 0.56, [
            ("leaks into", None),
            ("  ".join("patch %d" % (k + 1) for k, _ in leak), dark(RED)),
            ("cut to zero", None),
            ("  ".join("patch %d" % (k + 1) for k in cut), dark(BLUE))],
            note, size=0.032, step=0.05)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.5, 0.31, "the arrangement learned as the features get sparser", op,
                  strong=True, size=0.034)
        worst = max(self.heldout)
        full = self.span(0.16)[0]
        for q, x in enumerate((0.165, 0.39, 0.615, 0.84)):
            shown = op * smoothstep(q * 0.12, q * 0.12 + 0.15, u)
            self._plane(sc, self.final[q], x, 0.535, 78.0, shown, width=2.6)
            self.text(sc, x, 0.745, "%.0f%% zeros" % (100 * self.zero[q]), shown, strong=True,
                      size=0.036)
            self.hbar(sc, x - 0.085, 0.82, full * self.heldout[q] / worst, TEAL, shown,
                      u=(u - 0.3 - q * 0.08) / 0.3, thick=22.0, track=full)
            self.text(sc, x, 0.885, "error %s" % number(self.heldout[q]), shown *
                      smoothstep(0.5, 0.7, u), size=0.032)


# ====================================================================== 44
class SparseAutoencoderFeatures(Inside):
    """GPT-2's layer-6 activations rewritten by a sparse autoencoder: a
    token's 768 numbers in, a few of 1,024 features firing, 768 numbers
    rebuilt; then what features fire on in held-out text, and how many of the
    dictionary are silent or read as one word."""

    TITLE = "Sparse autoencoder features"
    PHASES = (("activations", 10.0), ("rewrite", 15.0), ("features", 15.0),
              ("dictionary", 13.0))
    FROM = 4                             # the held-out tokens shown start at the first line
    TURNS = (1, 2, 4, 5, 6)              # those rewritten in turn
    INK = (GREEN, VIOLET, BLUE, AMBER)   # the features shown; the last is the typical one
    KINDS = (DIM, AMBER, soft(TEAL, 0.3))    # silent, fires on one word, the rest

    def setup(self):
        s = self.study
        d = SI.sae_data()
        self.d = d
        self.m = s.metadata["summary"]
        self.words = [show(t) for t in d["tokens"][self.FROM:self.FROM + 10]]
        self.x, self.h, self.recon = (d[k][self.FROM:self.FROM + 10] for k in ("x", "h", "recon"))
        self.size = len(d["fires"])
        self.side = int(round(np.sqrt(self.size)))
        self.unexplained = float(s.panels[3].data["values"][0])
        self.k = int(s.metadata["settings"]["k"])
        self.updates = int(s.metadata["settings"]["updates"])
        self.n_train = int(s.metadata["train_residuals"])
        self.ratio = float(s.metadata["first_token_norm_ratio"])
        self.picks = [int(j) for j in self.m["picks"]]
        scored = d["scored"].astype(bool)
        self.kind = np.where(d["fires"] == 0, 0,
                             np.where(scored & (d["purity"] >= SI.READABLE), 1, 2))
        self.counts = [int((self.kind == k).sum()) for k in range(3)]
        self.sweep_k = [int(k) for k in s.panels[1].data["x"]]
        self.sweep = [float(v) for v in s.panels[1].data["ys"][0]]
        self.sweep_silent = [float(v) for v in s.panels[1].data["ys"][1]]

    def subtitle_for(self, k):
        return [
            "GPT-2 reads held-out Shakespeare; halfway up, at layer 6, every token has become "
            "768 numbers, a pattern nobody can read off directly",
            "a sparse autoencoder rewrites the 768 numbers as %s features, of which only %d "
            "may fire, and rebuilds them: on held-out text it recovers %.0f%% of the variance"
            % (number(self.size), self.k, 100 * (1 - self.unexplained)),
            "what makes a feature fire? its strongest firings on held-out text, with the words "
            "before. %d of %d features fire on one word, like the first three; the fourth is a "
            "typical one that does not" % (self.counts[1], self.m["scored"]),
            "trained for %s updates on %s tokens, each passage's first token left out: %d of "
            "the %s features are silent on held-out text, and %.0f%% read as one word" % (
                number(self.updates), f"{self.n_train:,}", self.counts[0],
                number(self.size), 100 * self.m["readable_share"]),
        ][k]

    def _tile(self, sc, v, fx, fy, w, h, op, vmax=2.5):
        """768 activations as a 24 x 32 block of colour."""
        self.image(sc, heat(np.clip(np.asarray(v).reshape(24, 32) / vmax, -1, 1)), fx, fy, w, h,
                   op)

    def phase_1(self, sc, u, op):
        n = 9
        xs = self.tokens(sc, self.words[:n], 0.05, 0.95, 0.875, op, w=80.0, h=44.0, size=0.036)
        self.box(sc, 0.5, 0.715, self.span(0.88)[0], 46, soft(BLUE, 0.7), op,
                 "GPT-2, layers 1 to 6", border=BLUE, size=0.036)
        self.text(sc, 0.5, 0.33, "layer 6: 768 numbers for every token", op, strong=True,
                  size=0.036)
        self.text(sc, 0.5, 0.385, "red above the average, blue below", op, size=0.032)
        for i, x in enumerate(xs):
            a = (u - 0.08 - i * 0.07) / 0.3
            self.arrow(sc, (x, 0.83), (x, 0.765), AXIS, op, smoothstep(0, 0.4, a), width=1.8)
            self.pulse(sc, (x, 0.83), (x, 0.59), a, BLUE, op)
            self._tile(sc, self.x[i], x, 0.535, 76, 100, op * smoothstep(0.75, 1.0, a))

    def phase_2(self, sc, u, op):
        turn = min(int(u * len(self.TURNS) * 1.03), len(self.TURNS) - 1)
        within = min(u * len(self.TURNS) * 1.03 - turn, 1.0)
        i = self.TURNS[turn] - 1
        self.row(sc, self.words, 0.05, 0.95, 0.895, op, h=40.0, size=0.034, lit={i: TEAL},
                 dim=range(len(self.words)))
        cols = (0.17, 0.50, 0.83)
        y = 0.565
        for x, head in zip(cols, ("768 numbers", "%s features" % number(self.size),
                                  "768 rebuilt")):
            self.text(sc, x, 0.315, head, op, strong=True, size=0.036)
        a, b, c = (smoothstep(0.03, 0.25, within), smoothstep(0.25, 0.45, within),
                   smoothstep(0.5, 0.75, within))
        self._tile(sc, self.x[i], cols[0], y, 216, 162, op)
        self.arrow(sc, (cols[0] + 0.125, y), (cols[1] - 0.14, y), AXIS, op, a, width=2.6)
        self.pulse(sc, (cols[0] + 0.125, y), (cols[1] - 0.14, y), (within - 0.03) / 0.25, TEAL, op)
        h = self.h[i].reshape(self.side, self.side)
        top = float(self.h[i].max())
        self.image(sc, heat(np.where(h > 0, np.clip(h / top, 0.45, 1.0), 0.0) * b, hot=TEAL),
                   cols[1], y, 200, 200, op)
        cell = self.at(cols[1], y, 200, 200, self.side, self.side)
        for r, k in np.argwhere(h > 0):
            self.outline(sc, *cell(r, k), 11, 11, TEAL, op * b, 2.0)
        self.arrow(sc, (cols[1] + 0.14, y), (cols[2] - 0.125, y), AXIS, op, c, width=2.6)
        self.pulse(sc, (cols[1] + 0.14, y), (cols[2] - 0.125, y), (within - 0.5) / 0.25, TEAL, op)
        self._tile(sc, self.recon[i], cols[2], y, 216, 162, op * c)
        fires = int((h > 0).sum())
        self.text(sc, cols[1], 0.80, "%d of %s fire" % (fires, number(self.size)), op * b,
                  size=0.034, ink=dark(TEAL))
        left = float(np.sum((self.recon[i] - self.x[i]) ** 2) / np.sum(self.x[i] ** 2))
        self.text(sc, cols[2], 0.76, "%.0f%% of it rebuilt" % (100 * (1 - left)), op * c,
                  size=0.034)
        self.text(sc, cols[0], 0.76, "the token  %s" % self.words[i], op, size=0.034)

    def phase_3(self, sc, u, op):
        d = self.d
        for f, j in enumerate(self.picks):
            ink = self.INK[f]
            y = 0.30 + f * 0.158
            start = f * 0.17
            shown = op * smoothstep(start, start + 0.08, u)
            self.box(sc, 0.062, y, 18, 18, ink, shown, radius=3)
            agree = int(round(float(d["purity"][j]) * SI.SAE["top"]))
            self.text(sc, 0.085, y, "feature %d%s: fires on %d held-out tokens; %d of its top "
                      "%d are one word" % (j, " (typical)" if f == len(self.picks) - 1 else "",
                                           int(d["fires"][j]), agree, SI.SAE["top"]),
                      shown, anchor="lm", strong=True, size=0.032)
            for c in range(4):
                words = [show(w) for w in d["contexts"][j][c][-3:] if str(w)]
                x0 = 0.085 + c * 0.218
                one = shown * smoothstep(0, 0.06, u - start - 0.05 - c * 0.03)
                self.row(sc, words, x0, x0 + 0.205, y + 0.068, one, h=40.0, size=0.032,
                         lit={len(words) - 1: ink}, left=True)

    def phase_4(self, sc, u, op):
        inks = np.asarray(self.KINDS, np.float32)
        gx, gy, w = 0.25, 0.645, 272.0
        shown = int(np.ceil(self.side * min(u / 0.5, 1.0)))
        rgb = inks[self.kind.reshape(self.side, self.side)]
        rgb[shown:] = PAPER * 0.97
        self.image(sc, rgb, gx, gy, w, w, op, lines=0.25)
        self.text(sc, gx, 0.295, "the %s features" % number(self.size), op, strong=True,
                  size=0.034)
        full = self.span(0.33)[0]
        rows = (("silent on held-out text", 0), ("fire on one word", 1), ("the rest", 2))
        for r, (name, k) in enumerate(rows):
            y = 0.345 + r * 0.118
            self.text(sc, 0.55, y, name, op, anchor="lm", size=0.034)
            self.hbar(sc, 0.55, y + 0.05, full * self.counts[k] / self.size, self.KINDS[k], op,
                      value="%d" % self.counts[k], u=(u - 0.15 - r * 0.12) / 0.3, thick=24.0,
                      track=full)
        self.lines(sc, 0.55, 0.72, [
            ("held-out variance left unexplained", layout.STRONG_INK)] + [
            "%d features per token: %.0f%%" % (k, 100 * v)
            for k, v in zip(self.sweep_k, self.sweep)],
            op * smoothstep(0.6, 0.8, u), size=0.034, step=0.05)


# ====================================================================== 45
class LinearProbes(Inside):
    """A linear probe per GPT-2 layer for "the token before this one is
    punctuation": the label drawn on real tokens, every held-out token's
    probe score sliding as the probe moves up the layers, accuracy by layer,
    and a made-up label that probes decode too."""

    TITLE = "Linear probes"
    PHASES = (("the label", 10.0), ("a probe", 17.0), ("every layer", 12.0),
              ("the control", 12.0))
    YES, NO, PROBE, CONTROL = RED, BLUE, TEAL, AMBER
    AXIS_X = (0.20, 0.92)                # the score axis
    LIMIT = (-6.0, 10.0)

    def setup(self):
        s = self.study
        d = probe_run()
        self.scores, self.labels = d["scores"], d["labels"].astype(bool)
        self.accuracy = np.asarray(s.panels[1].data["ys"][0], float)
        self.control = np.asarray(s.panels[2].data["ys"][0], float)
        self.layers = np.asarray(s.panels[1].data["x"], float)
        self.words = [show(t) for t in d["tokens"][3:14]]
        self.word_yes = self.labels[3:14]
        self.word_control = d["controls"][3:14].astype(bool)
        self.positive = float(s.panels[0].data["values"][1])
        self.best = int(np.argmax(self.accuracy))
        rng = np.random.default_rng(0)
        self.jitter = rng.uniform(-1, 1, len(self.labels))

    def subtitle_for(self, k):
        a, c = self.accuracy, self.control
        return [
            "the question put to GPT-2's activations: does the token just before this one "
            "contain punctuation? true for %.0f%% of %d held-out tokens" % (
                100 * self.positive, len(self.labels)),
            "a probe is one weighted sum of a token's 768 numbers at one layer; above zero it "
            "answers yes. accuracy, balanced over yes and no, is %.2f at layer 0 and %.2f at "
            "layer %d" % (a[0], a[self.best], self.best),
            "one probe per layer: the answer is easiest to read in the middle, %.2f at layers "
            "4 to 7, and fades to %.2f by the last layer" % (a[4:8].min(), a[-1]),
            "a control: a made-up yes or no fixed by the current token's identity is decoded "
            "at %.2f from layer 0. a probe shows what can be read, not what the model uses" % (
                c[0]),
        ][k]

    def _words(self, sc, y, op, yes, ink, h=54.0, size=0.044):
        inks = {i: ink for i in range(len(self.words)) if yes[i]}
        return self.row(sc, self.words, 0.05, 0.95, y, op, inks=inks, h=h, size=size)

    def phase_1(self, sc, u, op):
        y = 0.58
        xs = self._words(sc, y, op, self.word_yes * (u > 0.25), self.YES)
        self.text(sc, 0.5, 0.33, "is the token before this one punctuation?", op, strong=True,
                  size=0.038)
        for i in range(1, len(self.words)):
            a = (u - 0.12 - i * 0.055) / 0.2
            yes = bool(self.word_yes[i])
            ink = self.YES if yes else self.NO
            self.arc(sc, (xs[i - 1], y - 0.062), (xs[i], y - 0.062), ink, op * (1.0 if yes
                     else 0.45), smoothstep(0, 1, a), bend=0.10, width=2.8 if yes else 1.6)
            self.text(sc, xs[i], y + 0.105, "yes" if yes else "no", op * smoothstep(0.8, 1.2, a),
                      size=0.038, ink=dark(ink))
        self.lines(sc, 0.5, 0.81, ["the label comes from the text itself; GPT-2 is never told it",
                                   "can it be read back out of GPT-2's activations?"],
                   op * smoothstep(0.75, 0.9, u), size=0.034, step=0.055, anchor="mm")

    def _layers(self, sc, x, op, at, ink):
        """GPT-2's thirteen read-out points as a column, the one probed lit."""
        for layer in range(13):
            y = 0.87 - layer * 0.041
            on = layer == at
            self.box(sc, x, y, 62, 18, ink if on else PAPER, op * (1.0 if on else 0.7),
                     border=None if on else AXIS, radius=3, glow=0.2 if on else 0.0)
        self.text(sc, x, 0.33, "layer %d" % at, op, strong=True, size=0.036, ink=dark(ink))
        self.text(sc, x, 0.915, "input", op, size=0.030)

    def phase_2(self, sc, u, op):
        stops = (0, 2, 5, 9, 12)                       # the layers the probe visits
        k = min(u / 0.92, 1.0) * (len(stops) - 1)
        i = int(min(k, len(stops) - 2))
        f = smoothstep(0.35, 1.0, k - i)               # rest on a layer, then move
        at = stops[i + 1] if f > 0.5 else stops[i]
        score = self.scores[stops[i]] + (self.scores[stops[i + 1]] - self.scores[stops[i]]) * f
        self._layers(sc, 0.09, op, at, self.PROBE)
        x0, x1 = self.AXIS_X
        lo, hi = self.LIMIT

        def fx(v):
            return x0 + (np.clip(v, lo, hi) - lo) / (hi - lo) * (x1 - x0)
        rows = ((True, 0.50, self.YES, "after punctuation"), (False, 0.74, self.NO, "not"))
        zero = fx(0.0)
        self.polyline(sc, [self.P(zero, 0.39), self.P(zero, 0.86)], 2.0, AXIS, op, bias=0.0)
        self.text(sc, zero + 0.012, 0.365, "the probe says yes", op, anchor="lm", size=0.032)
        self.text(sc, zero - 0.012, 0.365, "says no", op, anchor="rm", size=0.032)
        for yes, y, ink, name in rows:
            pick = self.labels == yes
            self.polyline(sc, [self.P(x0, y + 0.085), self.P(x1, y + 0.085)], 1.0, GRID, op,
                          bias=0.0)
            for v, j in zip(score[pick], self.jitter[pick]):
                self.disk(sc, self.P(float(fx(v)), y + 0.065 * j), 4.2, ink, op * 0.8,
                          outline=0.0, lift=1.0)
            right = float(((score[pick] > 0) == yes).mean())
            self.text(sc, x1, y - 0.092, "%s: %.0f%% on the right side" % (name, 100 * right),
                      op, anchor="rm", size=0.032, ink=dark(ink))
        self.text(sc, (x0 + x1) / 2, 0.905, "each dot: one held-out token's score at layer %d  "
                  "(balanced accuracy %.2f)" % (at, self.accuracy[at]), op, strong=True,
                  size=0.032)

    def _chart(self, sc, op):
        chart = self.chart(((0.27, 0.90), (0.36, 0.80)), self.layers, [0.5, 1.0], pad=0.0)
        self.axes(sc, chart, op, "layer the probe reads", "balanced accuracy",
                  xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8, 10, 12)],
                  yticks=[(v, "%.1f" % v) for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
        return chart

    def phase_3(self, sc, u, op):
        go = min(u / 0.7, 1.0)
        self._layers(sc, 0.09, op, int(round(go * 12)), self.PROBE)
        chart = self._chart(sc, op)
        self.plot(sc, chart, self.layers, self.accuracy, self.PROBE, op, go, width=3.0)
        for layer in range(int(go * 12 + 1e-6) + 1):
            self.disk(sc, self.P(chart.fx(layer), chart.fy(self.accuracy[layer])), 4.5,
                      self.PROBE, op, outline=0.0, lift=1.2)
        self.text(sc, chart.fx(0) + 0.015, chart.fy(0.5) - 0.03, "0.5: a coin flip", op,
                  anchor="lm", size=0.032)
        note = op * smoothstep(0.72, 0.85, u)
        self.text(sc, chart.fx(self.best), chart.fy(self.accuracy[self.best]) - 0.04,
                  "%.2f" % self.accuracy[self.best], note, strong=True, size=0.034)
        self.text(sc, chart.fx(12) - 0.01, chart.fy(self.accuracy[12]) + 0.045,
                  "%.2f" % self.accuracy[12], note, strong=True, size=0.034)
        self.text(sc, chart.fx(0) + 0.03, chart.fy(self.accuracy[0]) + 0.03,
                  "%.2f" % self.accuracy[0], note, strong=True, size=0.034)

    def phase_4(self, sc, u, op):
        xs = self._words(sc, 0.335, op, self.word_control, self.CONTROL, h=36.0, size=0.032)
        chart = self.chart(((0.27, 0.90), (0.50, 0.82)), self.layers, [0.5, 1.0], pad=0.0)
        self.axes(sc, chart, op, "layer the probe reads", "balanced accuracy",
                  xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8, 10, 12)],
                  yticks=[(v, "%.1f" % v) for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)])
        self.plot(sc, chart, self.layers, self.accuracy, self.PROBE, op, width=3.0, dot=False)
        go = (u - 0.2) / 0.5
        self.plot(sc, chart, self.layers, self.control, self.CONTROL, op * (go > 0), go,
                  width=3.0)
        self.text(sc, xs[0] - 0.03, 0.405, "a made-up label: yes for the coloured tokens, "
                  "whatever surrounds them", op, anchor="lm", size=0.032, ink=dark(self.CONTROL))
        self.lines(sc, 0.04, 0.58, [("punctuation", dark(self.PROBE)), ("before", dark(
            self.PROBE)), ("", None), ("made-up", dark(self.CONTROL)), ("label", dark(
                self.CONTROL))], op, size=0.034, step=0.045)
        note = op * smoothstep(0.7, 0.85, u)
        self.text(sc, chart.fx(0) + 0.03, chart.fy(self.control[0]) - 0.035,
                  "%.2f" % self.control[0], note, strong=True, size=0.034)
        self.text(sc, chart.fx(12) - 0.01, chart.fy(self.control[12]) + 0.04,
                  "%.2f" % self.control[12], note, strong=True, size=0.034)


# ====================================================================== 46
class ActivationPatching(Inside):
    """A clean and a corrupted run of GPT-2 side by side as lattices of
    (layer, token) activations; one clean activation copied into the
    corrupted run and the answer's probability moving; then every such copy,
    one at a time, filling the heatmap of how much each recovers."""

    TITLE = "Activation patching"
    PHASES = (("two runs", 11.0), ("one patch", 16.0), ("every patch", 14.0),
              ("where it flows", 12.0))
    CLEAN, BAD, PATCH = GREEN, RED, AMBER
    SHOWN = ((6, 4), (0, 2), (6, 7))     # the single patches followed in phase 2
    TRACK = (AMBER, VIOLET, TEAL)

    def setup(self):
        s = self.study
        clean, corrupt, target = patching_prompts()
        self.clean, self.corrupt = [show(t) for t in clean], [show(t) for t in corrupt]
        self.swapped = [i for i in range(len(clean)) if clean[i] != corrupt[i]]
        self.target = show(target)
        self.p_clean, self.p_bad = [float(v) for v in s.panels[0].data["values"]]
        self.p = np.asarray(s.panels[1].data["values"], float)            # layer, token
        self.recovery = np.asarray(s.panels[2].data["values"], float)
        self.layers, self.n = self.p.shape
        self.path = self.recovery.argmax(1)
        runs = []                                    # (token, first layer, last layer)
        for layer, k in enumerate(self.path):
            if runs and runs[-1][0] == k:
                runs[-1][2] = layer
            else:
                runs.append([int(k), layer, layer])
        self.runs = runs
        self.columns = [r[0] for r in runs][:3]

    def _name(self, k):
        if k == self.n - 1:
            return "the last token"
        if k in self.swapped:
            return "the swapped token"
        return 'the token "%s"' % self.clean[k]

    def subtitle_for(self, k):
        flow = ", ".join("at %s in layers %d to %d" % (self._name(t), a, b)
                         for t, a, b in self.runs)
        return [
            "GPT-2 reads a held-out passage and gives its true next token, a comma, "
            "probability %.3f; with two earlier tokens swapped for others it gives %.3f" % (
                self.p_clean, self.p_bad),
            "patching: run the corrupted prompt again, but copy in one activation, one token "
            "at one layer, from the clean run, and see how far the probability comes back",
            "all %d patches, one at a time: each cell is how much of the gap from %.3f back "
            "to %.3f that single copied activation recovers" % (
                self.p.size, self.p_bad, self.p_clean),
            "the patch that works best moves: %s. one passage, not a map of where facts live"
            % flow,
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _lattice(self, sc, rgb, x, y, cw, ch, op, words, inks=None, tw=None):
        """A (layer, token) lattice, layer 0 at the bottom, over its tokens.
        Returns ``cell(layer, token) -> (fx, fy)``."""
        w, h = cw * self.n, ch * self.layers
        self.image(sc, np.asarray(rgb)[::-1], x, y, w, h, op, lines=0.3)
        dw, dh = self.frac(w, h)
        self.tokens(sc, words, x - dw / 2, x + dw / 2, y + dh / 2 + 0.058, op, w=cw - 4.0,
                    h=34.0, size=0.029, inks=inks)
        at = self.at(x, y, w, h, self.layers, self.n)
        return lambda layer, k: at(self.layers - 1 - layer, k)

    def _tint(self, bad, upto=13.0):
        """Lattice colours: the clean run's activations, or the corrupted
        run's (which differ from the swapped token on), up to a layer."""
        rgb = np.tile(np.asarray(PAPER * 0.97, np.float32), (self.layers, self.n, 1))
        for layer in range(self.layers):
            if layer <= upto:
                for k in range(self.n):
                    differs = bad and k >= self.swapped[0]
                    rgb[layer, k] = soft(self.BAD if differs else self.CLEAN, 0.45)
        return rgb

    def _inks(self, bad):
        return [self.BAD if (bad and k in self.swapped) else None for k in range(self.n)]

    def _answer(self, sc, x, y, p, op, ink):
        self.box(sc, x - 0.05, y, 40, 40, soft(ink), op, self.target, border=ink, size=0.040)
        self.text(sc, x - 0.02, y, "p = %.3f" % p, op, anchor="lm", strong=True, size=0.038)

    def _pair(self, sc, op, upto=13.0):
        cells = []
        for x, bad, name in ((0.27, False, "the passage"), (0.73, True, "two tokens swapped")):
            words = self.corrupt if bad else self.clean
            cells.append(self._lattice(sc, self._tint(bad, upto), x, 0.615, 38.0, 15.0, op,
                                       words, inks=None if not bad else self._inks(bad)))
            self.text(sc, x - 0.178, 0.378, name, op, anchor="lm", size=0.032,
                      ink=dark(self.BAD if bad else self.CLEAN))
        return cells

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        rise = u / 0.6 * 13.0
        self._pair(sc, op, rise)
        done = op * smoothstep(0.6, 0.72, u)
        self._answer(sc, 0.33, 0.305, self.p_clean, done, self.CLEAN)
        self._answer(sc, 0.79, 0.305, self.p_bad, done, self.BAD)
        self.text(sc, 0.5, 0.48, "layers", op, size=0.030)
        self.arrow(sc, (0.5, 0.78), (0.5, 0.52), AXIS, op, smoothstep(0.0, 0.6, u), width=2.2)
        self.text(sc, 0.5, 0.945, "every cell: one token's activation at one of %d layers"
                  % self.layers, op * smoothstep(0.2, 0.4, u), size=0.034)

    def phase_2(self, sc, u, op):
        clean, bad = self._pair(sc, op)
        turn = min(int(u * 3 * 1.02), 2)
        within = min(u * 3 * 1.02 - turn, 1.0)
        layer, k = self.SHOWN[turn]
        a, b = clean(layer, k), bad(layer, k)
        self.outline(sc, a[0], a[1], 40, 17, self.PATCH, op, 3.0)
        fly = (within - 0.08) / 0.4
        self.over(sc, a, b, self.PATCH, op, smoothstep(0, 1, fly), bend=0.14, width=3.0)
        landed = smoothstep(0.48, 0.55, within)
        self.box(sc, b[0], b[1], 38, 15, self.CLEAN, op * landed, radius=2, glow=0.2)
        self.outline(sc, b[0], b[1], 40, 17, self.PATCH, op * landed, 3.0)
        p = self.p_bad + (self.p[layer, k] - self.p_bad) * smoothstep(0.55, 0.8, within)
        self._answer(sc, 0.33, 0.305, self.p_clean, op, self.CLEAN)
        self._answer(sc, 0.79, 0.305, p, op, self.PATCH if landed > 0.5 else self.BAD)
        self.text(sc, 0.49, 0.945, "copy layer %d at %s:" % (layer, self._name(k)), op,
                  anchor="rm", strong=True, size=0.034)
        self.text(sc, 0.51, 0.945, "%.0f%% of the gap recovered" % (
            100 * self.recovery[layer, k]), op * smoothstep(0.75, 0.9, within), anchor="lm",
            size=0.034, ink=dark(self.PATCH, 0.3))

    def _map(self, sc, op, count=None, ring=()):
        shown = self.recovery.copy()
        if count is not None:
            shown.ravel()[count:] = 0.0
        rgb = heat(np.clip(shown, -1, 1), hot=self.CLEAN, cold=BLUE)
        if count is not None:
            rgb.reshape(-1, 3)[count:] = PAPER * 0.97
        cell = self._lattice(sc, rgb, 0.30, 0.61, 46.0, 17.0, op, self.corrupt,
                             inks=self._inks(True))
        for layer in (0, 6, 12):
            self.text(sc, 0.30 - 0.225, cell(layer, 0)[1], "layer %d" % layer, op, anchor="rm",
                      size=0.030)
        for layer, k in ring:
            x, y = cell(layer, k)
            self.outline(sc, x, y, 46, 17, self.PATCH, op, 2.6)
        return cell

    def phase_3(self, sc, u, op):
        count = int(min(u / 0.82, 1.0) * self.p.size)
        now = min(count, self.p.size - 1)
        layer, k = divmod(now, self.n)
        self._map(sc, op, count, ring=[(layer, k)] if count < self.p.size else ())
        self.text(sc, 0.30, 0.325, "share of the gap recovered by each patch", op, strong=True,
                  size=0.034)
        x, full = 0.63, self.span(0.26)[0]
        self.text(sc, x, 0.42, "probability of the comma", op, anchor="lm", strong=True,
                  size=0.034)
        rows = (("clean run", self.CLEAN, self.p_clean), ("corrupted run", self.BAD, self.p_bad),
                ("with this patch", self.PATCH, self.p[layer, k]))
        for r, (name, ink, value) in enumerate(rows):
            y = 0.49 + r * 0.125
            self.text(sc, x, y, name, op, anchor="lm", size=0.032, ink=dark(ink, 0.3))
            self.hbar(sc, x, y + 0.05, full * value / 0.26, ink, op, value="%.3f" % value,
                      thick=24.0, track=full)
        self.text(sc, x, 0.86, 'layer %d, token "%s"' % (layer, self.corrupt[k]), op, anchor="lm",
                  size=0.034)

    def phase_4(self, sc, u, op):
        go = min(u / 0.6, 1.0)
        upto = int(go * (self.layers - 1))
        cell = self._map(sc, op, ring=[(layer, self.path[layer]) for layer in range(upto + 1)])
        self.text(sc, 0.30, 0.325, "the best patch at each layer", op, strong=True, size=0.034)
        layers = np.arange(self.layers, dtype=float)
        chart = self.chart(((0.66, 0.93), (0.39, 0.68)), layers, [-0.2, 1.05], pad=0.0)
        self.axes(sc, chart, op, "layer patched", "recovered",
                  xticks=[(v, "%d" % v) for v in (0, 4, 8, 12)],
                  yticks=[(v, "%.1f" % v) for v in (0.0, 0.5, 1.0)])
        rows = []
        for ink, k in zip(self.TRACK, self.columns):
            self.plot(sc, chart, layers, self.recovery[:, k], ink, op, go, width=2.8)
            x, y = cell(self.layers - 1, k)
            self.box(sc, x, y - 0.035, 40, 9, ink, op, radius=3)
            rows.append(("patched at %s" % self._name(k), dark(ink, 0.25)))
        self.lines(sc, 0.63, 0.82, rows, op, size=0.032, step=0.045)


# ====================================================================== 47
class InductionHeads(Inside):
    """A passage that repeats itself, and the attention head that uses the
    repeat: from the second copy of a token, GPT-2's head 5.1 looks at what
    followed the first copy. Then every head scored for it; then a small
    two-layer model trained here on lines read over and over, the head it
    grows, and the moment in training when that head appears."""

    TITLE = "Induction heads"
    PHASES = (("a repeat", 9.0), ("what came next", 15.0), ("every head", 10.0),
              ("from scratch", 13.0), ("it appears", 12.0))
    OURS, CONTROL = TEAL, AXIS
    FIRST, SECOND = (4, 14), (21, 32)    # the two stretches of the passage shown
    QUERIES = (23, 24, 25, 26, 27)
    MATCH, READ, NEXT = TEAL, RED, AMBER
    ROWS = (0.45, 0.72)

    def setup(self):
        s = self.study
        d = induction_run()
        self.tokens_all = [show(t) for t in d["tokens"]]
        self.ids = [int(i) for i in d["ids"]]
        self.targets = successor_targets(self.ids)
        self.mass = np.asarray(s.panels[3].data["values"], float)
        self.best = [(int(b) // 12, int(b) % 12) for b in np.argsort(-self.mass.ravel())[:5]]
        # the head followed: the best one that is among the kept attention maps
        kept = [(int(b) // 12, int(b) % 12) for b in d["best"]]
        self.head = (5, 1)
        self.attn = d["attention"][kept.index(self.head)]
        self.top = list(range(*self.FIRST))
        self.bottom = list(range(*self.SECOND))
        from . import realdata
        sm = SI.induction_data()
        self.sm = s.metadata["summary"]
        self.updates = int(s.metadata["settings"]["updates"])
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.later = np.asarray(s.panels[1].data["seeds"], float)   # kind, seed, step
        seeds = self.later.shape[1]
        self.score = np.asarray(s.panels[2].data["ys"], float).reshape(2, seeds, -1)
        e = int(sm["shown"])
        self.length = int(sm["held_len"][e])
        self.window = [int(i) for i in sm["held"][e]]
        self.chars = [show(c) for c in realdata.decode(self.window[:-1])]
        # seed 0's models: the one trained on repeated lines, and the control
        heads = [int(sm["copy"][kind, 0, -1, 1].argmax()) for kind in range(2)]
        self.small_head = heads[0]
        row = sm["attention"][0, 1, heads[0]]

        def share(t, row=row):
            return float(row[t, SI.aligned_successors(t, self.length)].sum())
        # a typical character of the second reading, not the luckiest: the median one by share
        ranked = sorted(range(self.length, 2 * self.length), key=share)
        self.small_query = q = ranked[len(ranked) // 2]
        self.small_targets = [int(j) for j in SI.aligned_successors(q, self.length)]
        self.small_row = row[q]
        self.small_share = share(q)
        self.control_share = float(sm["attention"][1, 1, heads[1]][q, self.small_targets].sum())
        self.p_next = [float(v) for v in sm["p_true"][:, q]]
        self.example = float(self.attn[24, 8])

    def subtitle_for(self, k):
        b = self.best
        return [
            "this held-out passage says the same words twice. a model that notices the repeat "
            "can guess that whatever followed last time will follow again",
            "GPT-2's head %d.%d does exactly that: reading the second \"mor\" it puts %.2f of "
            "its attention on \"row\", the token that followed the first \"mor\"" % (
                self.head + (self.example,)),
            "every head scored on the passage's natural repeats: the share of its attention "
            "that lands just after an earlier copy. heads %s lead with %s" % (
                ", ".join("%d.%d" % h for h in b[:3]),
                ", ".join("%.2f" % self.mass[h] for h in b[:3])),
            "can a small model grow such a head? a two-layer attention-only model trained here "
            "on lines of Shakespeare read over and over does: one head puts %.2f of its "
            "attention on what came next last time" % self.small_share,
            "the head appears between updates %d and %d in all three seeds; loss on the later "
            "readings falls to %.2f nats. the same model trained on ordinary text grows none: "
            "its best head scores %.2f" % (
                min(self.sm["appears_at"]), max(self.sm["appears_at"]),
                float(np.mean(self.sm["twice"]["later_loss"])),
                float(np.max(self.sm["plain"]["best_head"]))),
        ][k]

    def _rows(self, sc, op, matched=(), query=None, ring=None):
        """Both stretches, one above the other; returns each token's centre."""
        where = {}
        for idxs, y in zip((self.top, self.bottom), self.ROWS):
            words = [self.tokens_all[i] for i in idxs]
            inks = {k: self.MATCH for k, i in enumerate(idxs) if i in matched}
            lit = {k: self.READ for k, i in enumerate(idxs) if i == query}
            xs = self.row(sc, words, 0.07, 0.95, y, op, inks=inks, lit=lit, h=60.0, size=0.048,
                          left=True)
            for i, x in zip(idxs, xs):
                where[i] = (x, y)
        if ring is not None:
            w = self.span(spread([self.tokens_all[ring]], 0, 1, 0.048)[1][0])[0]
            self.outline(sc, where[ring][0], where[ring][1], w + 8, 68, self.NEXT, op, 3.4, 8.0)
        self.text(sc, 0.95, self.ROWS[0], "earlier", op, anchor="rm", size=0.034)
        self.text(sc, 0.95, self.ROWS[1], "7 tokens later", op, anchor="rm", size=0.034)
        return where

    def phase_1(self, sc, u, op):
        pairs = [(i, i - 17) for i in range(22, 28)]            # the repeat: 22..27 are 5..10
        shown = [p for k, p in enumerate(pairs) if u > 0.15 + k * 0.1]
        where = self._rows(sc, op, matched=[i for p in shown for i in p])
        for k, (later, earlier) in enumerate(pairs):
            a = smoothstep(0.0, 0.1, u - 0.15 - k * 0.1)
            (x1, y1), (x0, y0) = where[later], where[earlier]
            self.arrow(sc, (x1, y1 - 0.068), (x0, y0 + 0.068), self.MATCH, op * 0.8, a,
                       width=2.0, head=False)
        self.text(sc, 0.07, 0.885, "the same six tokens, in the same order", op *
                  smoothstep(0.75, 0.9, u), anchor="lm", strong=True, size=0.036)

    def phase_2(self, sc, u, op):
        turn = min(int(u * len(self.QUERIES) * 1.02), len(self.QUERIES) - 1)
        within = min(u * len(self.QUERIES) * 1.02 - turn, 1.0)
        q = self.QUERIES[turn]
        nxt = self.targets[q][-1]                       # what followed the earlier copy
        where = self._rows(sc, op, matched=[q - 17], query=q,
                           ring=nxt if within > 0.3 else None)
        weights = self.attn[q]
        for i in self.top + [j for j in self.bottom if j < q]:
            w = float(weights[i])
            if w < 0.03:
                continue
            (x0, y0), (x1, y1) = where[i], where[q]
            if y0 < y1:
                a, b, bend = (x0, y0 + 0.07), (x1, y1 - 0.07), 0.0
            else:
                a, b, bend = (x0, y0 - 0.07), (x1, y1 - 0.07), 0.05
            ink = self.NEXT if i == nxt else AXIS
            self.arc(sc, a, b, ink, op, smoothstep(0.2, 0.55, within), bend, 1.5 + 6.0 * w)
            self.pulse(sc, a, b, (within - 0.5) / 0.4, ink, op, bend, r=3.0 + 4.0 * w)
        said = op * smoothstep(0.45, 0.6, within)
        self.text(sc, where[nxt][0], self.ROWS[0] - 0.10, "%.2f" % weights[nxt], said,
                  strong=True, size=0.040, ink=dark(self.NEXT, 0.3))
        self.lines(sc, 0.07, 0.865, [
            ("reading \"%s\": it was here before, followed by \"%s\"" % (
                self.tokens_all[q], self.tokens_all[nxt]), None),
            ("head %d.%d puts %.2f of its attention on that \"%s\"" % (
                self.head + (weights[nxt], self.tokens_all[nxt])), dark(self.NEXT, 0.3))],
            op, size=0.036, step=0.055)

    def phase_3(self, sc, u, op):
        mx, my, side = 0.29, 0.63, 276.0
        go = min(u / 0.6, 1.0)
        ring = [h for h in self.best if h[0] < int(np.ceil(go * 12))]
        # squared, so that colour deepens in proportion (the kit's scale lifts small values)
        cell = self.grid(sc, (self.mass / 0.45) ** 2, mx, my, side, side, op, reveal=go,
                         ring=ring, ring_ink=self.READ, hot=self.NEXT, vmax=1.0)
        self.text(sc, mx, 0.315, "GPT-2's 144 heads", op, strong=True, size=0.036)
        for layer in (0, 5, 11):
            self.text(sc, mx - 0.175, cell(layer, 0)[1], "layer %d" % layer, op, anchor="rm",
                      size=0.030)
        self.text(sc, mx, my + 0.305, "head 0 to 11", op, size=0.030)
        x, full = 0.66, self.span(0.24)[0]
        self.text(sc, 0.58, 0.36, "attention just after an earlier copy", op, anchor="lm",
                  strong=True, size=0.034)
        for r, h in enumerate(self.best):
            v = float(self.mass[h])
            self.hbar(sc, x, 0.445 + r * 0.085, full * v / 0.5, self.NEXT, op,
                      "%d.%d" % h, "%.2f" % v, u=(u - h[0] / 12 * 0.6) / 0.2, thick=28.0,
                      track=full)
        self.text(sc, 0.58, 0.885, "layer.head", op, anchor="lm", size=0.030)

    def phase_4(self, sc, u, op):
        y, q, n = 0.56, self.small_query, len(self.chars)
        xs = self.tokens(sc, [c if i != q else "" for i, c in enumerate(self.chars)], 0.04,
                         0.96, y, op, w=16.5, h=34.0, size=0.030)
        self.box(sc, xs[q], y, 16.5, 34, self.READ, op, self.chars[q], ink=WHITE, size=0.030)
        for m, start in enumerate(range(0, n, self.length)):         # one bracket per reading
            end = min(start + self.length, n) - 1
            ink = (TEAL, VIOLET)[m % 2]
            self.arrow(sc, (xs[start] - 0.007, y + 0.055), (xs[end] + 0.007, y + 0.055), ink, op,
                       width=3.0, head=False)
            if end - start >= 6:
                self.text(sc, (xs[start] + xs[end]) / 2, y + 0.095,
                          "read once" if m == 0 else "again", op, size=0.030, ink=dark(ink))
        w = self.small_row
        grow = smoothstep(0.1, 0.35, u)
        for i in range(q):
            if w[i] < 0.03:
                continue
            ink = self.NEXT if i in self.small_targets else AXIS
            self.arc(sc, (xs[i], y - 0.042), (xs[q], y - 0.042), ink, op, grow,
                     0.04 + 0.004 * (q - i), 1.5 + 6.0 * float(w[i]))
        for i in self.small_targets:
            self.outline(sc, xs[i], y, 22, 40, self.NEXT, op, 3.0)
        before, after = self.chars[q], self.chars[self.small_targets[-1]]
        self.lines(sc, 0.04, 0.30, [
            ("a model trained here on lines read over and over: a layer-2 head, reading the red "
             "\"%s\"" % before, layout.STRONG_INK),
            "%.2f of its attention goes to the boxed \"%s\", which followed \"%s\" on the "
            "first reading" % (self.small_share, after, before)], op, size=0.034, step=0.05)
        self.lines(sc, 0.04, 0.76, [
            ("so the model gives the true next character, \"%s\", probability %.2f" % (
                show(realdata_char(self.window[q + 1])), self.p_next[0]), dark(self.NEXT, 0.3)),
            "the same model trained on ordinary text: %.2f of its attention there, "
            "probability %.2f" % (self.control_share, self.p_next[1])],
            op * smoothstep(0.5, 0.65, u), size=0.034, step=0.055)

    def phase_5(self, sc, u, op):
        go = (u - 0.05) / 0.6
        panels = ((self.later, ((0.09, 0.44), (0.40, 0.78)), "loss (nats)",
                   "loss on the later readings"),
                  (self.score, ((0.60, 0.95), (0.40, 0.78)), "attention",
                   "best head: attention to what came next"))
        for values, box, ylabel, title in panels:
            chart = self.chart(box, self.steps, values)
            self.axes(sc, chart, op, "training update", ylabel)
            for kind, ink in ((1, self.CONTROL), (0, self.OURS)):
                for seed in range(values.shape[1]):
                    self.plot(sc, chart, self.steps, values[kind, seed], ink, op, go,
                              width=2.6 if kind == 0 else 1.8, dot=kind == 0)
            self.text(sc, (box[0][0] + box[0][1]) / 2, 0.325, title, op, strong=True, size=0.034)
        said = op * smoothstep(0.6, 0.8, u)
        self.box(sc, 0.10, 0.905, 18, 18, self.OURS, said, radius=3)
        self.text(sc, 0.12, 0.905, "trained on lines read over and over, 3 seeds", said,
                  anchor="lm", size=0.032)
        self.box(sc, 0.56, 0.905, 18, 18, self.CONTROL, said, radius=3)
        self.text(sc, 0.58, 0.905, "trained on ordinary text, 3 seeds", said, anchor="lm",
                  size=0.032)


# ====================================================================== 48
class HeadAblation(Inside):
    """GPT-2's 144 attention heads as a 12 x 12 lattice over a passage: one
    head switched off and the passage's loss read again; then every head on
    64 held-out passages, the map of mean changes settling as passages are
    added, and the check that two halves of the passages rank the heads alike."""

    TITLE = "Head ablation"
    PHASES = (("144 heads", 9.0), ("switch one off", 15.0), ("64 passages", 15.0),
              ("does it hold?", 14.0))
    UP, DOWN, OFF = RED, BLUE, (0.16, 0.17, 0.19)
    AT = (0.30, 0.615, 252.0)            # the lattice: x, y, side in px
    SHOWN = ((0, 7), (2, 6), (7, 3))     # the knock-outs followed on the first passage
    VMAX = 0.5                           # nats at full colour

    def setup(self):
        s = self.study
        d = SI.ablation_data()
        self.each = np.asarray(d["change"], float)            # passage, layer, head
        self.change = np.asarray(s.panels[1].data["values"], float)
        self.sem = np.asarray(s.panels[2].data["values"], float)
        self.base = [float(v) for v in s.panels[0].data["values"]]
        self.n = len(self.each)
        self.per = int(s.metadata["settings"]["tokens"])
        self.words = [show(t) for t in d["tokens"][0]]
        self.running = np.cumsum(self.each, 0) / np.arange(1, self.n + 1)[:, None, None]
        order = np.argsort(self.change.ravel())
        self.worst = [divmod(int(i), 12) for i in order[::-1][:3]]     # loss rose most
        self.eased = [divmod(int(i), 12) for i in order[:3]]           # loss fell most
        self.xy = np.asarray(s.panels[3].data["xy"], float)
        self.rho = float(s.metadata["rank_correlation"])
        self.single = float(s.metadata["single_passage_rank_correlation"])
        self.clear = int(s.metadata["heads_beyond_two_standard_errors"])
        both = np.concatenate([self.each[:, a, b] for a, b in self.SHOWN[:2]])
        self.strip = (float(np.floor(both.min() * 2) / 2), float(np.ceil(both.max() * 2) / 2))

    def subtitle_for(self, k):
        (a, b), (c, e) = self.SHOWN[:2]
        base = self.base[0]
        w = self.worst[0]
        return [
            "GPT-2 has 12 layers of 12 attention heads. on this held-out passage its loss, "
            "averaged over the %d next tokens it has to predict, is %.2f nats" % (self.per, base),
            "ablation: switch one head off, run the model again and read the loss. without "
            "head %d.%d this passage's loss goes from %.2f to %.2f; without head %d.%d it "
            "falls to %.2f" % (a, b, base, base + self.each[0, a, b], c, e,
                               base + self.each[0, c, e]),
            "one passage cannot rank 144 heads: its ranking agrees with the other %d only at "
            "rank correlation %.2f. so each head is switched off on %d held-out passages and "
            "the changes averaged" % (self.n - 1, self.single, self.n),
            "the two halves of the passages rank the heads alike, correlation %.2f. losing "
            "head %d.%d costs %.2f ± %.2f nats, and %d of the 144 changes stand clear of twice "
            "their standard error" % (self.rho, w[0], w[1], self.change[w], self.sem[w],
                                      self.clear),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _heads(self, sc, op, values=None, count=None, off=None, ring=(), ring_ink=AMBER):
        """The lattice of heads, layer 0 at the bottom; ``values`` (nats) colours
        the first ``count`` (in the order they are ablated). Returns the cell centres."""
        x, y, side = self.AT
        if values is None:
            rgb = np.tile(np.asarray(soft(TEAL, 0.75), np.float32), (12, 12, 1))
        else:
            rgb = heat(np.clip(np.asarray(values) / self.VMAX, -1, 1), self.UP, self.DOWN)
            if count is not None:
                rgb.reshape(-1, 3)[count:] = soft(TEAL, 0.75)
        self.image(sc, rgb[::-1], x, y, side, side, op, lines=0.35)
        at = self.at(x, y, side, side, 12, 12)

        def cell(layer, head):
            return at(11 - layer, head)
        for layer in (0, 5, 11):
            self.text(sc, x - 0.16, cell(layer, 0)[1], "layer %d" % layer, op, anchor="rm",
                      size=0.030)
        self.text(sc, x, y + 0.28, "head 0 to 11", op, size=0.030)
        if off is not None:
            self.box(sc, *cell(*off), side / 12, side / 12, self.OFF, op, radius=2)
        for h in ring:
            self.outline(sc, *cell(*h), side / 12 + 2, side / 12 + 2, ring_ink, op, 2.6)
        return cell

    def _prompt(self, sc, op):
        self.row(sc, self.words[-7:], 0.54, 0.95, 0.90, op, h=40.0, size=0.034)
        self.text(sc, 0.95, 0.825, "the end of the passage GPT-2 reads", op, anchor="rm",
                  size=0.030)

    def _reading(self, sc, op, loss, ink=TEAL, was=None):
        x, full = 0.58, self.span(0.30)[0]
        self.text(sc, x, 0.35, "its %d next tokens, each predicted" % self.per, op, anchor="lm",
                  size=0.034)
        self.text(sc, x, 0.405, "from the ones before it", op, anchor="lm", size=0.034)
        self.text(sc, x, 0.50, "loss (nats per token): lower is better", op, anchor="lm",
                  size=0.034)
        self.hbar(sc, x, 0.565, full * min(loss, 8.0) / 8.0, ink, op, value="%.2f" % loss,
                  thick=30.0, track=full)
        if was is not None:
            self.text(sc, x, 0.64, "with every head on: %.2f" % was, op, anchor="lm",
                      size=0.032)

    def _spread(self, sc, op, head, y, n, ink):
        """One head's loss change on each of the first ``n`` passages, as dots on
        an axis, with their mean."""
        v = self.each[:, head[0], head[1]]
        x0, x1 = 0.58, 0.94
        lo, hi = self.strip

        def fx(value):
            return x0 + (value - lo) / (hi - lo) * (x1 - x0)
        self.arrow(sc, (x0, y), (x1, y), AXIS, op, width=1.4, head=False)
        for tick in np.arange(np.ceil(lo), hi + 1e-9, 1.0):
            self.arrow(sc, (fx(tick), y - 0.012), (fx(tick), y + 0.012), AXIS, op, width=1.4,
                       head=False)
            self.text(sc, fx(tick), y + 0.045, "%+d" % tick if tick else "0", op, size=0.030)
        for i in range(n):
            self.disk(sc, self.P(fx(v[i]), y + ((i * 7) % 5 - 2) * 0.0045), 3.4, ink, op * 0.75,
                      lift=1.2, outline=0.0)
        mean = float(v[:n].mean())
        self.arrow(sc, (fx(mean), y - 0.034), (fx(mean), y + 0.034), dark(ink, 0.35), op,
                   width=3.4, head=False)
        self.text(sc, x0, y - 0.065, "without head %d.%d: mean %+.2f nats" % (
            head[0], head[1], mean), op, anchor="lm", strong=True, size=0.034)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        cell = self._heads(sc, op)
        self.text(sc, self.AT[0], 0.30, "GPT-2: 12 layers of 12 heads", op, strong=True,
                  size=0.036)
        for head in (1, 5, 9):
            self.pulse(sc, cell(0, head), cell(11, head), (u * 2.2) % 1.1, TEAL, op, r=5.0)
        self._prompt(sc, op)
        self._reading(sc, op * smoothstep(0.3, 0.5, u), self.base[0])

    def phase_2(self, sc, u, op):
        turn = min(int(u * 3 * 1.02), 2)
        within = min(u * 3 * 1.02 - turn, 1.0)
        off = self.SHOWN[turn]
        gone = within > 0.15
        cell = self._heads(sc, op, off=off if gone else None, ring=[off])
        self.text(sc, self.AT[0], 0.30, "head %d.%d switched off" % off, op, strong=True,
                  size=0.036)
        for head in (1, 5, 9):
            self.pulse(sc, cell(0, head), cell(11, head), (within - 0.2) / 0.35, TEAL, op, r=5.0)
        self._prompt(sc, op)
        change = float(self.each[0][off])
        delta = change * smoothstep(0.5, 0.75, within)
        ink = TEAL if abs(delta) < 0.005 else self.UP if delta > 0 else self.DOWN
        self._reading(sc, op, self.base[0] + delta, ink, was=self.base[0])
        self.text(sc, 0.58, 0.72, "change: %+.2f nats" % change,
                  op * smoothstep(0.75, 0.9, within), anchor="lm", strong=True, size=0.036)

    def phase_3(self, sc, u, op):
        n = 1 + int(min(u / 0.8, 1.0) * (self.n - 1))
        self._heads(sc, op, self.running[n - 1], ring=self.SHOWN[:2])
        self.text(sc, self.AT[0], 0.30, "mean loss change over %d passage%s" % (
            n, "" if n == 1 else "s"), op, strong=True, size=0.036)
        self.text(sc, 0.58, 0.31, "one dot per passage", op, anchor="lm", size=0.032)
        for r, (head, ink) in enumerate(zip(self.SHOWN[:2], (self.UP, self.DOWN))):
            self._spread(sc, op, head, 0.47 + r * 0.22, n, ink)
        self.text(sc, 0.58, 0.865, "red: the loss rose without the head   blue: it fell", op,
                  anchor="lm", size=0.030)

    def phase_4(self, sc, u, op):
        x, y = self.xy[:, 0], self.xy[:, 1]
        chart = self.chart(((0.10, 0.44), (0.44, 0.80)), np.r_[x, y], np.r_[x, y])
        self.axes(sc, chart, op, "mean change, even passages", "odd passages")
        lo, hi = chart.x_range
        self.polyline(sc, [self.P(chart.fx(lo), chart.fy(lo)), self.P(chart.fx(hi), chart.fy(hi))],
                      1.4, AXIS, 0.6 * op)
        named = self.worst + self.eased
        count = int(np.ceil(144 * smoothstep(0.0, 0.5, u)))
        for i in range(count):
            head = divmod(i, 12)
            ink = self.UP if head in self.worst else self.DOWN if head in self.eased else TEAL
            self.disk(sc, self.P(chart.fx(x[i]), chart.fy(y[i])), 5.0 if head in named else 3.2,
                      ink, op * (1.0 if head in named else 0.7), lift=1.2, outline=0.0)
        self.text(sc, 0.27, 0.295, "one dot per head: two halves of the passages", op,
                  strong=True, size=0.034)
        self.text(sc, 0.27, 0.345, "rank correlation %.2f" % self.rho,
                  op * smoothstep(0.5, 0.65, u), size=0.034, ink=dark(TEAL))
        bx, full = 0.70, self.span(0.13)[0]
        groups = (("the loss rose most without", self.worst, self.UP, 0.30),
                  ("the loss fell most without", self.eased, self.DOWN, 0.60))
        for g, (name, heads, ink, top) in enumerate(groups):
            self.text(sc, 0.58, top, name, op, anchor="lm", strong=True, size=0.034)
            most = abs(float(self.change[heads[0]]))
            for r, h in enumerate(heads):
                v = float(self.change[h])
                self.hbar(sc, bx, top + 0.065 + r * 0.062, full * abs(v) / most, ink, op,
                          "head %d.%d" % h, "%+.2f ± %.2f" % (v, self.sem[h]),
                          u=(u - 0.1 - g * 0.25 - r * 0.05) / 0.25, thick=24.0, track=full)


# ====================================================================== 49
class SaliencyAndIntegratedGradients(Inside):
    """Which pixels earned the 7 its score: the gradient swept in pixel by
    pixel, the same times the image, then integrated gradients built up along
    the path from a black image, its running sum chasing the logit."""

    TITLE = "Saliency and integrated gradients"
    PHASES = (("the score", 9.0), ("gradient", 14.0), ("the path", 17.0), ("adds up?", 12.0))
    PLUS, MINUS, LOGIT = RED, BLUE, TEAL

    def setup(self):
        s = self.study
        d = saliency_run()
        self.d = d
        self.x = d["x"].reshape(28, 28)
        self.target = int(d["target"])
        self.gap, self.ig_sum, self.err = [float(v) for v in s.panels[3].data["values"]]
        self.gxi_sum = float(d["times_input"].sum())
        g = np.abs(d["gradient"])
        self.blank = float(g[d["x"] == 0].sum() / g.sum())
        self.steps = int(s.metadata["settings"]["integration_steps"])
        self.rise = d["path_logit"] - d["path_logit"][0]

    def subtitle_for(self, k):
        return [
            "a trained MNIST network gives this image a score for each digit, and the %d wins "
            "with %.1f. which pixels earned it that score?" % (
                self.target, self.d["logits"][self.target]),
            "the gradient asks how the %d's score would move if one pixel were nudged. it is "
            "noisy: %.0f%% of its size sits on blank pixels. times the image, only inked "
            "pixels remain" % (self.target, 100 * self.blank),
            "integrated gradients walks from a black image to the digit in %d steps, adding "
            "up the gradient on the way, so each pixel's share covers the whole climb" % (
                self.steps),
            "do the shares add up? integrated gradients' sum to %.3f and the score rose %.3f "
            "from black: off by %.3f. gradient × input sums to %.2f" % (
                self.ig_sum, self.gap, self.err, self.gxi_sum),
        ][k]

    def _map(self, sc, v, x, y, side, op, reveal=1.0):
        self.grid(sc, np.asarray(v).reshape(28, 28), x, y, side, side, op, reveal=reveal,
                  hot=self.PLUS, cold=self.MINUS, lines=0)

    def phase_1(self, sc, u, op):
        logits = self.d["logits"]
        self.digit(sc, self.x, 0.17, 0.61, 220.0, op, frame=PAPER)
        self.text(sc, 0.17, 0.335, "28 × 28 pixels", op, strong=True, size=0.034)
        self.box(sc, 0.415, 0.61, 110, 170, soft(self.LOGIT, 0.7), op, "network",
                 border=self.LOGIT, size=0.036)
        self.arrow(sc, (0.30, 0.61), (0.35, 0.61), AXIS, op, smoothstep(0.05, 0.25, u), width=2.6)
        self.arrow(sc, (0.48, 0.61), (0.53, 0.61), AXIS, op, smoothstep(0.3, 0.5, u), width=2.6)
        self.pulse(sc, (0.29, 0.61), (0.53, 0.61), (u - 0.05) / 0.45, self.LOGIT, op, r=6.0)
        zero, scale = 0.745, 0.18 / float(np.abs(logits).max())
        grow = smoothstep(0.45, 0.75, u)
        self.text(sc, 0.745, 0.305, "score for each digit", op, strong=True, size=0.034)
        for i, v in enumerate(logits):
            y = 0.365 + i * 0.056
            on = i == self.target
            w = self.span(abs(float(v)) * scale * grow)[0]
            ink = self.LOGIT if on else DIM
            self.rect(sc, self.shift(self.P(zero, y), np.sign(v) * w / 2, 0), max(w, 0.5), 20, ink,
                      op, radius=3)
            self.text(sc, 0.555, y, "%d" % i, op, size=0.036, strong=on)
            if on:
                self.text(sc, zero + 0.19, y, "%.1f" % v, op * grow, anchor="lm", strong=True,
                          size=0.036)

    def phase_2(self, sc, u, op):
        d = self.d
        xs, y, side = (0.19, 0.50, 0.81), 0.60, 196.0
        sweep = min(u / 0.45, 1.0)
        self._map(sc, d["gradient"], xs[0], y, side, op, reveal=sweep)
        self.text(sc, xs[0], 0.335, "gradient", op, strong=True, size=0.036)
        self.text(sc, xs[0], 0.845, "each pixel: how fast the", op, size=0.030)
        self.text(sc, xs[0], 0.885, "score moves if it is nudged", op, size=0.030)
        b = op * smoothstep(0.5, 0.62, u)
        self.digit(sc, self.x, xs[1], y, side, b, frame=PAPER)
        self.text(sc, xs[1], 0.335, "the image", b, strong=True, size=0.036)
        self.text(sc, (xs[0] + xs[1]) / 2, y, "×", b, strong=True, size=0.06)
        c = op * smoothstep(0.68, 0.82, u)
        self._map(sc, d["times_input"], xs[2], y, side, c)
        self.text(sc, xs[2], 0.335, "gradient × input", c, strong=True, size=0.036)
        self.text(sc, (xs[1] + xs[2]) / 2, y, "=", c, strong=True, size=0.06)
        self.text(sc, xs[2], 0.845, "red: raises the %d's score" % self.target, c, size=0.030,
                  ink=dark(self.PLUS))
        self.text(sc, xs[2], 0.885, "blue: lowers it", c, size=0.030, ink=dark(self.MINUS))

    def phase_3(self, sc, u, op):
        d = self.d
        alpha = min(u / 0.85, 1.0)
        k = int(round(alpha * self.steps))
        stops = (0.0, 0.25, 0.5, 0.75, 1.0)
        at = int(np.argmin([abs(alpha - a) for a in stops]))
        for i, a in enumerate(stops):
            x = 0.095 + i * 0.105
            self.digit(sc, self.x * a, x, 0.435, 84.0, op, frame=AMBER if i == at else PAPER)
        self.text(sc, 0.095, 0.30, "black", op, size=0.032)
        self.text(sc, 0.515, 0.30, "the digit", op, size=0.032)
        self.arrow(sc, (0.15, 0.30), (0.44, 0.30), AXIS, op, width=1.8)
        chart = self.chart(((0.10, 0.52), (0.63, 0.86)), d["alphas"], self.rise)
        self.axes(sc, chart, op, "how far along the path", "rise of the %d's score" % self.target,
                  xticks=[(a, "%.2g" % a) for a in stops])
        self.plot(sc, chart, d["alphas"], self.rise, self.LOGIT, op, alpha, width=3.0)
        gx, gy, side = 0.775, 0.59, 216.0
        vmax = float(np.abs(d["ig"]).max())
        run = d["running"][k - 1] if k > 0 else np.zeros(784)
        self.grid(sc, run.reshape(28, 28), gx, gy, side, side, op, vmax=vmax, hot=self.PLUS,
                  cold=self.MINUS, lines=0)
        self.text(sc, gx, 0.30, "each pixel's share so far", op, strong=True, size=0.034)
        self.text(sc, gx, 0.855, "shares sum to %.2f" % run.sum(), op, strong=True, size=0.036)
        self.text(sc, gx, 0.905, "the score has risen %.2f" % self.rise[k], op, size=0.034,
                  ink=dark(self.LOGIT))

    def phase_4(self, sc, u, op):
        d = self.d
        xs, y, side = (0.19, 0.50, 0.81), 0.535, 170.0
        maps = (("gradient", d["gradient"], None), ("gradient × input", d["times_input"],
                self.gxi_sum), ("integrated gradients", d["ig"], self.ig_sum))
        full = self.span(0.22)[0]
        for x, (name, v, total) in zip(xs, maps):
            self._map(sc, v, x, y, side, op)
            self.text(sc, x, 0.30, name, op, strong=True, size=0.034)
            if total is None:
                self.text(sc, x, 0.775, "a slope per pixel:", op, size=0.032)
                self.text(sc, x, 0.82, "not shares of the score", op, size=0.032)
                continue
            self.hbar(sc, x - 0.11, 0.775, full * total / self.gap, self.PLUS, op,
                      u=(u - 0.15) / 0.4, thick=24.0, track=full)
            self.text(sc, x, 0.83, "sums to %s" % ("%.3f" % total if x > 0.6 else
                                                     "%.2f" % total), op, size=0.034)
        self.text(sc, 0.5, 0.905, "the %d's score rose %.3f from the black image: the full "
                  "length of each bar" % (self.target, self.gap), op * smoothstep(0.5, 0.7, u),
                  strong=True, size=0.034)


# ====================================================================== 50
class FeatureVisualisation(Inside):
    """One output unit of the MNIST CNN and the real digit it likes best;
    then an image of noise pushed uphill on that unit's response, plainly and
    with the usual regularisers, every step drawn; then all ten units' images
    beside the real digits, and whether a second network can read them."""

    TITLE = "Feature visualisation"
    PHASES = (("the unit", 11.0), ("climb", 14.0), ("regularised", 14.0), ("measured", 14.0))
    PLAIN, SMOOTH, REAL, FIRE = RED, BLUE, GREEN, AMBER
    DIGIT = 0                            # the unit followed: the first

    def setup(self):
        s = self.study
        d = SI.feature_data()
        self.d = d
        self.m = s.metadata["summary"]
        g = self.DIGIT
        self.frames = d["frames"][:, g]                       # run, step, 28, 28
        self.traces = d["traces"][:, 0, g]                    # run, step
        self.steps = np.asarray(s.panels[3].data["x"], float)
        self.finals = [np.asarray(s.panels[k].data["values"], float) for k in range(3)]
        self.real = d["real_best_images"][g]
        self.real_value = [float(v) for v in d["real_best_value"][g]]
        from . import mnist
        params, _ = mnist.cnn_train()
        self.scores = SI.cnn_logits(params, self.real[:1])[0]
        self.said = d["judge"][:, 0, g]                       # run, class: the second network
        self.trials = int(self.m["trials"])

    def subtitle_for(self, k):
        t, m, g = self.traces, self.m, self.DIGIT
        return [
            "the MNIST CNN has one output unit per digit. of the 10,000 test images, the one "
            "the unit for %d answers most is this: a response of %.1f, its score minus the "
            "mean of the other nine" % (g, self.real_value[0]),
            "feature visualisation: start from noise and change the pixels themselves, %d "
            "steps uphill on that response. it climbs to %.0f, %.1f times the best real digit, "
            "and the image is speckle" % (len(self.steps) - 1, t[0, -1],
                                          t[0, -1] / self.real_value[0]),
            "the same climb with the usual regularisers: shift the image a few pixels each "
            "step, blur the gradient, fade toward blank paper. the response stops at %.0f and "
            "what survives is strokes" % t[1, -1],
            "all ten units from three starting noises: a second network reads the intended "
            "digit in %d of %d plain images and %d of %d regularised ones, which are also more "
            "like the average real digit, %.2f against %.2f" % (
                m["plain"]["judge_right"], self.trials, m["regularised"]["judge_right"],
                self.trials, m["regularised"]["likeness"], m["plain"]["likeness"]),
        ][k]

    def phase_1(self, sc, u, op):
        g = self.DIGIT
        self.digit(sc, self.real[0], 0.14, 0.58, 190.0, op, frame=PAPER)
        self.text(sc, 0.14, 0.335, "a real test digit", op, strong=True, size=0.036)
        self.box(sc, 0.345, 0.58, 96, 150, soft(BLUE, 0.7), op, "CNN", border=BLUE, size=0.040)
        self.arrow(sc, (0.25, 0.58), (0.288, 0.58), AXIS, op, width=2.4)
        self.pulse(sc, (0.25, 0.58), (0.288, 0.58), (u * 3) % 1.2, BLUE, op)
        self.arrow(sc, (0.402, 0.58), (0.44, 0.58), AXIS, op, smoothstep(0.1, 0.25, u),
                   width=2.4)
        self.text(sc, 0.56, 0.335, "ten output units", op, strong=True, size=0.036)
        full, top = self.span(0.15)[0], float(self.scores.max())
        for c in range(10):
            y = 0.395 + c * 0.047
            v = float(self.scores[c])
            self.hbar(sc, 0.49, y, full * max(v, 0.0) / top, self.FIRE if c == g else DIM, op,
                      "%d" % c, "%.1f" % v, u=(u - 0.2 - c * 0.02) / 0.25, thick=17.0)
        shown = op * smoothstep(0.55, 0.7, u)
        self.lines(sc, 0.74, 0.40, [
            ("the unit for %d" % g, dark(self.FIRE, 0.3)),
            "its score minus the mean",
            "of the other nine:",
            ("response %.1f" % self.real_value[0], layout.STRONG_INK)], shown, size=0.034,
            step=0.052)
        self.lines(sc, 0.74, 0.67, [
            "no real digit scores",
            "higher on this unit.",
            "what would?"], op * smoothstep(0.75, 0.9, u), size=0.034, step=0.052)

    def _climb(self, sc, u, op, run, ink):
        last = len(self.steps) - 1
        step = int(min(u / 0.85, 1.0) * last)
        image = self.frames[run, step]
        side, y, x = 208.0, 0.60, 0.155
        if run == 1 and step < last:
            dy, dx = self.d["shifts"][self.DIGIT][step]
            fx, fy = self.frac(side / 28 * dx, side / 28 * dy)
            self.digit(sc, image, x + fx, y + fy, side, op, frame=soft(ink, 0.3))
            self.text(sc, x, 0.895, "this step's shift: %+d, %+d pixels" % (dx, dy), op,
                      size=0.030)
        else:
            self.digit(sc, image, x, y, side, op, frame=PAPER)
        self.text(sc, x, 0.325, "the image, step %d" % step, op, strong=True, size=0.034)
        self.digit(sc, self.real[0], 0.42, y, 150.0, op, frame=soft(self.REAL, 0.3))
        self.text(sc, 0.42, 0.325, "best real %d" % self.DIGIT, op, strong=True, size=0.034,
                  ink=dark(self.REAL))
        self.text(sc, 0.42, 0.79, "response %.1f" % self.real_value[0], op, size=0.032)
        chart = self.chart(((0.63, 0.94), (0.42, 0.78)), self.steps, self.traces)
        self.axes(sc, chart, op, "step", "response")
        mark = chart.fy(self.real_value[0])
        self.polyline(sc, [self.P(chart.x0, mark), self.P(chart.x1, mark)], 1.8, self.REAL,
                      0.9 * op)
        if run == 1:
            self.plot(sc, chart, self.steps, self.traces[0], self.PLAIN, op * 0.6, width=2.4,
                      dot=False)
        self.plot(sc, chart, self.steps, self.traces[run], ink, op, step / last, width=3.0)
        self.text(sc, 0.785, 0.325, "response %.1f" % self.traces[run, step], op, strong=True,
                  size=0.036, ink=dark(ink))
        if step == last:
            read = int(self.said[run].argmax())
            self.text(sc, x, 0.895, "a second network reads: %d" % read, op, size=0.032,
                      ink=dark(ink))
        return step

    def phase_2(self, sc, u, op):
        step = self._climb(sc, u, op, 0, self.PLAIN)
        if step < len(self.steps) - 1:
            self.text(sc, 0.155, 0.895, "each step: nudge every pixel uphill", op, size=0.030)

    def phase_3(self, sc, u, op):
        self._climb(sc, u, op, 1, self.SMOOTH)

    def phase_4(self, sc, u, op):
        m = self.m
        rows = (("plain climb", self.finals[0], self.PLAIN,
                 "read right %d of %d" % (m["plain"]["judge_right"], self.trials),
                 "response %.0f" % m["plain"]["objective"]),
                ("regularised", self.finals[1], self.SMOOTH,
                 "read right %d of %d" % (m["regularised"]["judge_right"], self.trials),
                 "response %.0f" % m["regularised"]["objective"]),
                ("best real digit", self.finals[2], self.REAL, "",
                 "response %.0f" % m["real_best"]))
        for r, (name, images, ink, read, response) in enumerate(rows):
            y = 0.385 + r * 0.165
            shown = op * smoothstep(r * 0.15, r * 0.15 + 0.12, u)
            self.lines(sc, 0.04, y - (0.045 if read else 0.025), [
                (name, dark(ink)), response] + ([read] if read else []), shown, size=0.032,
                step=0.045)
            for c in range(10):
                one = shown * smoothstep(0, 0.1, u - r * 0.15 - c * 0.012)
                self.digit(sc, images[c], 0.275 + c * 0.0725, y, 58.0, one, frame=soft(ink, 0.3))
        for c in range(10):
            self.text(sc, 0.275 + c * 0.0725, 0.295, "%d" % c, op, size=0.032)
        self.text(sc, 0.5, 0.875, "like the average real digit: plain %.2f, regularised %.2f "
                  "(1 would be identical)" % (m["plain"]["likeness"],
                                              m["regularised"]["likeness"]),
                  op * smoothstep(0.6, 0.8, u), size=0.034)


ANIMATIONS = {
    "superposition": Superposition,
    "sparse-autoencoder-features": SparseAutoencoderFeatures,
    "linear-probes": LinearProbes,
    "activation-patching": ActivationPatching,
    "induction-heads": InductionHeads,
    "head-ablation": HeadAblation,
    "saliency-and-integrated-gradients": SaliencyAndIntegratedGradients,
    "feature-visualisation": FeatureVisualisation,
}
