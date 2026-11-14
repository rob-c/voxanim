"""Patch embeddings: a vision transformer's first layer, on MNIST.

Two classifiers from :mod:`voxanim.patch_embed`, trained the same way (Adam,
seeded, cached), read a digit as 16 patch tokens:

    h_k = ReLU(x_k E + b + pos[k]),   logits = mean_k(h_k) W + c

one without the position vectors ``pos`` and one with them.  Five phases:

1. **patches** -- a test digit cut into a 4 × 4 grid of 7 × 7 patches, which
   lift apart into a row of 16 tokens, read row by row,
2. **embed** -- the shared matrix ``E`` (49 × 64) learning through training
   (41 snapshots), with 16 of its 64 columns -- the ones of largest norm --
   drawn as 7 × 7 filters; then one patch's 49 pixels, flattened, times
   ``E``: 64 numbers.  The same as a 7 × 7 convolution with stride 7,
3. **positions** -- the 16 tokens ``x_k E + b`` (a 16 × 64 matrix) plus the
   16 learned position vectors: the blank patches' tokens are ``b + pos[k]``,
   place alone; the learned vectors draw no map of the grid (their mean
   cosine one step apart is about that of places far apart),
4. **shuffle** -- the digit's patches scrambled with the study's fixed
   permutation (:func:`voxanim.patch_embed.shuffle_order`); both models'
   predictions on the original and the scrambled digit.  The digit is the
   first test digit that both models get right, that the model without
   positions still gets right scrambled, and that the model with positions
   gets wrong scrambled,
5. **accuracy** -- the 10,000 test digits as they are and scrambled: 83.8%
   and 83.8% without positions (every prediction the same), 94.3% and
   26.3% with them.

Every number is computed at build time from the cached runs and MNIST.
"""

from __future__ import annotations

import numpy as np

from . import layout, mnist
from . import patch_embed as pe
from .diagram import Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import div_colors, gray_colors
from .text import num

SIDE, PP = pe.SIDE, pe.P                 # 4 × 4 patches of 7 × 7 pixels
N_PATCH, DIM = pe.N_PATCH, pe.DIM
N_FILTERS = 16                            # filters drawn (the largest columns of E)
DIGIT_PX = 196.0                          # the digit in phase 1 (7 px per pixel)
TOKEN_PX, TOKEN_GAP = 34.0, 8.0           # the row of tokens
SPLIT_GAP = 8.0                           # px between patches when the grid opens
DIGIT_AT = (0.5, 0.50)
ROW_Y = 0.83
SHUF_PX = 168.0                           # the two digits of phase 4
SHUF_AT = ((0.24, 0.50), (0.62, 0.50))
NO_POS_INK = (0.28, 0.30, 0.32)
POS_INK = (0.12, 0.36, 0.66)
WRONG_INK = (0.74, 0.20, 0.16)


def softmax(z):
    """Softmax of a vector."""
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def pick_digit(y, pred_a, pred_a_shuf, pred_b, pred_b_shuf):
    """The first test digit both models get right, that the model without positions
    (``a``) still gets right scrambled and the model with positions (``b``) does not."""
    ok = (pred_a == y) & (pred_a_shuf == y) & (pred_b == y) & (pred_b_shuf != y)
    return int(np.flatnonzero(ok)[0])


def grid_distance():
    """``(16, 16)`` city-block distance between grid places."""
    r, c = np.divmod(np.arange(N_PATCH), SIDE)
    return np.abs(r[:, None] - r[None]) + np.abs(c[:, None] - c[None])


class PatchEmbeddings(Diagram):
    """A digit as 16 patch tokens, with and without learned position vectors."""

    PHASES = ("patches", "embed", "positions", "shuffle", "accuracy")

    def __init__(self):
        super().__init__("Patch embeddings", "", 10.0)
        self.hold = 2.5
        self.a, self.b = pe.train(False), pe.train(True)
        _, _, xte, yte = mnist.load()
        xp = pe.patches(xte)
        order = pe.shuffle_order()
        self.order = order
        pa, pas = pe.predict(self.a.P, xp), pe.predict(self.a.P, xp[:, order])
        pb, pbs = pe.predict(self.b.P, xp), pe.predict(self.b.P, xp[:, order])
        self.n_test = len(yte)
        self.acc = {("a", 0): float((pa == yte).mean()), ("a", 1): float((pas == yte).mean()),
                    ("b", 0): float((pb == yte).mean()), ("b", 1): float((pbs == yte).mean())}
        self.n_same = int((pa == pas).sum())
        i = pick_digit(yte, pa, pas, pb, pbs)
        self.idx, self.label_y = i, int(yte[i])
        self.img = xte[i].reshape(28, 28)
        self.xp = xp[i]                                           # (16, 49)
        self.xs = xp[i][order]                                    # scrambled tokens
        self.img_shuf = pe.unpatch(self.xs[None])[0].reshape(28, 28)
        self.probs = {}
        for name, run in (("a", self.a), ("b", self.b)):
            for s, x in enumerate((self.xp, self.xs)):
                self.probs[name, s] = softmax(pe.forward(run.P, x[None])[0][0])
        # E, the filters and the tokens of the model with positions
        E = self.b.P["E"]
        self.E = E
        self.E_snaps = self.b.E_snaps
        self.snap_steps = self.b.snap_steps
        self.e_max = float(np.abs(E).max())
        norms = np.linalg.norm(E, axis=0)
        self.filters = np.argsort(-norms, kind="stable")[:N_FILTERS]
        self.f_max = np.abs(E[:, self.filters]).max(0)            # per-filter colour scale
        ink = self.xp.sum(1)
        self.k_show = int(np.argmax(ink))                         # the inkiest patch
        self.xE = self.xp @ E                                     # (16, 64)
        self.tok0 = self.xE + self.b.P["b"]
        self.pos = self.b.P["pos"]
        self.tok = self.tok0 + self.pos
        self.blank = [int(k) for k in np.flatnonzero(ink == 0)]
        S = pe.position_similarity(self.pos)
        D = grid_distance()
        self.cos_near = float(S[D == 1].mean())
        self.cos_far = float(S[D >= 3].mean())
        self.pos_norm = np.linalg.norm(self.pos, axis=1)

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("digit_in", 1.5, 0.0)
        add("cut", 2.0, 0.2)
        add("lift", 4.5, 2.0)
        add("embed", 13.0, 1.0)
        add("positions", 11.0, 1.0)
        add("shuffle", 11.5, 1.0)
        add("accuracy", 8.5, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.starts = [Tm[k][0] for k in ("embed", "positions", "shuffle", "accuracy")]
        self.marks = [0.0] + [s - 0.5 for s in self.starts]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        ops = []
        for k in range(len(self.PHASES)):
            op = 1.0 if k == 0 else smoothstep(self.starts[k - 1] - 0.4, self.starts[k - 1], t)
            if k < len(self.starts):
                end = self.starts[k] - 0.5
                op *= 1.0 - smoothstep(end - 0.6, end, t)
            ops.append(op * back)
        return ops

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        pa0, pa1 = self.probs["a", 0], self.probs["a", 1]
        pb1 = self.probs["b", 1]
        self.subtitle = [
            "a %d × %d test digit cut into a %d × %d grid of %d × %d patches: %d tokens, "
            "read row by row" % (28, 28, SIDE, SIDE, PP, PP, N_PATCH),
            "each patch's %d pixels times one shared matrix E (%d × %d): a %d × %d convolution "
            "with %d filters and stride %d" % (PP * PP, PP * PP, DIM, PP, PP, DIM, PP),
            "one learned vector per grid place, added to the patch's embedding: the token "
            "now carries where it came from",
            "shuffled, the model without positions still says %d, just as sure (%.3f, before "
            "%.3f); the model with positions says %d" % (int(pa1.argmax()), pa1.max(),
                                                         pa0.max(), int(pb1.argmax())),
            "%s test digits: without positions %.1f%%, and %.1f%% shuffled; with them "
            "%.1f%%, and %.1f%%" % (f"{self.n_test:,}", 100 * self.acc["a", 0],
                                    100 * self.acc["a", 1], 100 * self.acc["b", 0],
                                    100 * self.acc["b", 1]),
        ][ph]
        draws = (self._patches, self._embed, self._positions, self._shuffle, self._accuracy)
        for fn, op in zip(draws, ops):
            if op > 0.01:
                fn(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    def _patch_tex(self, x):
        """A patch's 49 pixels as a 7 × 7 greyscale texture."""
        return gray_colors(x).reshape(PP, PP, 3)

    # -- phase 1: the grid of patches -------------------------------------------------------
    def _patches(self, sc, t, op):
        """The digit, its 4 × 4 grid opening, the patches flying to a row of 16 tokens."""
        show = smoothstep(0.0, 1.0, self.progress("digit_in", t))
        cut = ease_in_out(self.progress("cut", t))
        lift = self.progress("lift", t)
        S = DIGIT_PX / SIDE
        c0 = self.P(*DIGIT_AT)
        row_c = self.P(0.5, ROW_Y)
        if cut <= 0.0:
            # whole, before the cut: one image, no seams between the tiles
            draw_image(self, sc, c0, gray_colors(self.img.ravel()).reshape(28, 28, 3), DIGIT_PX,
                       DIGIT_PX, op * show, grid=0.0)
        for k in range(N_PATCH if cut > 0.0 else 0):
            r, c = divmod(k, SIDE)
            gap = SPLIT_GAP * cut
            p0 = self.shift(c0, (c - 1.5) * (S + gap), (r - 1.5) * (S + gap))
            p1 = self.shift(row_c, (k - (N_PATCH - 1) / 2) * (TOKEN_PX + TOKEN_GAP), 0)
            u = ease_in_out(remap(lift, 0.04 * k, 0.04 * k + 0.4))
            p = p0 + (p1 - p0) * u
            p = self.shift(p, 0, -30.0 * np.sin(np.pi * u))
            size = S + (TOKEN_PX - S) * u
            draw_image(self, sc, p, self._patch_tex(self.xp[k]), size, size, op * show,
                       grid=0.0, lift=0.3 + 0.3 * (0 < u < 1))
            if 0 < u < 1:
                sc.probe(("tile", k), p, tol=6.0)
            # the place number, under the token's place in the row, as the patch lands
            self.label(sc, self.shift(p1, 0, TOKEN_PX / 2 + 6), str(k), 0.026, anchor="mt",
                       opacity=op * smoothstep(0.85, 1.0, u), ink=layout.NOTE_INK)
        gone = 1.0 - smoothstep(0.0, 0.1, lift)
        self.label(sc, self.shift(c0, 0, -DIGIT_PX / 2 - 22), "test digit %d, a %d" % (
            self.idx, self.label_y), 0.028, anchor="mb", opacity=op * show * gone,
            ink=layout.STRONG_INK)
        side = self.shift(c0, DIGIT_PX / 2 + 30, 0)
        self.label(sc, self.shift(side, 0, -12), "%d × %d pixels" % (28, 28), 0.027,
                   anchor="lm", opacity=op * show * (1 - smoothstep(0.0, 0.4, cut)))
        self.label(sc, self.shift(side, 0, -12), "%d × %d patches" % (SIDE, SIDE), 0.027,
                   anchor="lm", opacity=op * smoothstep(0.5, 1.0, cut) * gone,
                   ink=layout.STRONG_INK)
        self.label(sc, self.shift(side, 0, 12), "of %d × %d pixels" % (PP, PP), 0.027,
                   anchor="lm", opacity=op * smoothstep(0.5, 1.0, cut) * gone)
        self.label(sc, self.P(0.5, ROW_Y - 0.12), "%d tokens, one per patch, in reading order"
                   % N_PATCH, 0.028, opacity=op * smoothstep(0.85, 1.0, lift),
                   ink=layout.STRONG_INK)

    # -- phase 2: the shared matrix E -------------------------------------------------------
    def _embed(self, sc, t, op):
        """E learning through training, 16 of its columns as filters; one patch times E."""
        u = self.progress("embed", t)
        train = ease_in_out(remap(u, 0.05, 0.5))
        s = int(round(train * (len(self.E_snaps) - 1)))
        E = self.E_snaps[s]
        k = self.k_show
        x = self.xp[k]
        # the patch and its 49 pixels as a column
        cell = 4.0
        e_c = self.P(0.42, 0.53)
        Ew, Eh = cell * DIM, cell * PP * PP
        col_c = self.shift(e_c, -Ew / 2 - 22, 0)
        patch_c = self.shift(col_c, -82, 0)
        draw_image(self, sc, patch_c, self._patch_tex(x), 70, 70, op, grid=0.0)
        self.label(sc, self.shift(patch_c, 0, -35 - 8), "patch %d" % k, 0.027, anchor="mb",
                   opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.shift(patch_c, 0, 35 + 8), "flattened", 0.026, anchor="mt",
                   opacity=op, ink=layout.NOTE_INK)
        sc.line(self.shift(patch_c, 40, 0), self.shift(col_c, -12, 0), width=1.4,
                color=(0.42, 0.44, 0.46), opacity=op, bias=0.3)
        draw_image(self, sc, col_c, gray_colors(x).reshape(PP * PP, 1, 3), 12, Eh, op, grid=0.0)
        self.label(sc, self.shift(col_c, 0, -Eh / 2 - 8), "x", 0.027, anchor="mb", opacity=op,
                   ink=layout.STRONG_INK)
        # E itself, as the current snapshot
        draw_image(self, sc, e_c, div_colors(E.ravel(), self.e_max).reshape(PP * PP, DIM, 3),
                   Ew, Eh, op, grid=0.0)
        self.label(sc, self.shift(e_c, 0, -Eh / 2 - 8), "E: %d × %d, the same for every patch"
                   % (PP * PP, DIM), 0.027, anchor="mb", opacity=op, ink=layout.STRONG_INK)
        # x E, filled column by column once training is over
        mult = remap(u, 0.55, 0.8)
        out_c = self.shift(e_c, 0, Eh / 2 + 26)
        n_out = int(round(mult * DIM))
        vmax = float(np.abs(self.xE).max())
        tex = div_colors(self.xE[k], vmax).reshape(1, DIM, 3)
        if n_out < DIM:
            tex = tex.copy()
            tex[0, n_out:] = (0.875, 0.875, 0.87)
        draw_image(self, sc, out_c, tex, Ew, 14, op * smoothstep(0.5, 0.56, u), grid=0.0)
        self.label(sc, self.shift(out_c, -Ew / 2 - 8, 0), "x E", 0.027, anchor="rm",
                   opacity=op * smoothstep(0.5, 0.56, u), ink=layout.STRONG_INK)
        self.label(sc, self.shift(out_c, 0, 16), "%d numbers: the patch against each filter" % DIM,
                   0.026, anchor="mt", opacity=op * smoothstep(0.5, 0.56, u),
                   ink=layout.NOTE_INK)
        if 0 < mult < 1:
            j = min(DIM - 1, n_out)
            top = self.shift(e_c, -Ew / 2 + cell * (j + 0.5), -Eh / 2)
            sc.probe(("col",), top, tol=6.0)
            self.rect(sc, self.shift(top, 0, Eh / 2), cell + 2, Eh + 4, (1, 1, 1), op,
                      border=layout.STRONG_INK, border_w=1.5, radius=1.0, lift=0.6, fill_op=0.0)
        # the filters
        f_c = self.P(0.82, 0.53)
        fs, fg = 40.0, 7.0
        for n, j in enumerate(self.filters):
            r, c = divmod(n, 4)
            p = self.shift(f_c, (c - 1.5) * (fs + fg), (r - 1.5) * (fs + fg))
            tex = div_colors(E[:, j], self.f_max[n]).reshape(PP, PP, 3)
            draw_image(self, sc, p, tex, fs, fs, op, grid=0.0)
        top = self.shift(f_c, 0, -2 * (fs + fg) - 6)
        self.label(sc, self.shift(top, 0, -22), "%d of the %d columns of E," % (N_FILTERS, DIM),
                   0.027, anchor="mb", opacity=op)
        self.label(sc, self.shift(top, 0, 0), "each as a %d × %d filter" % (PP, PP), 0.027,
                   anchor="mb", opacity=op)
        bot = self.shift(f_c, 0, 2 * (fs + fg) + 4)
        self.label(sc, bot, "training step %s of %s" % (f"{int(self.snap_steps[s]):,}",
                                                        f"{int(self.snap_steps[-1]):,}"),
                   0.027, anchor="mt", opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.shift(bot, 0, 22), "orange +, blue −", 0.026, anchor="mt",
                   opacity=op, ink=layout.NOTE_INK)
        sc.probe(("snap",), (s * 4.0, 0, 0), tol=6.0)
        self.label(sc, self.P(0.5, 0.90), "the same %d × %d numbers as sliding the %d filters "
                   "over the digit %d pixels at a time" % (N_PATCH, DIM, DIM, PP), 0.027,
                   opacity=op * smoothstep(0.85, 0.95, u), ink=layout.STRONG_INK)

    # -- phase 3: position vectors ----------------------------------------------------------
    def _positions(self, sc, t, op):
        """x_k E + b, plus pos[k], equals the token: three 16 × 64 matrices."""
        u = self.progress("positions", t)
        w, h = 3.0 * DIM, 9.0 * N_PATCH
        xs = (0.20, 0.50, 0.80)
        y = 0.49
        vmax = float(np.percentile(np.abs(self.tok), 98))     # the largest few saturate
        mats = (("x E + b", self.tok0, 0.0), ("pos", self.pos, 0.15),
                ("token", self.tok, 0.32))
        for (name, M, at), x in zip(mats, xs):
            show = smoothstep(at, at + 0.1, u)
            c = self.P(x, y)
            draw_image(self, sc, c, div_colors(M.ravel(), vmax).reshape(N_PATCH, DIM, 3), w, h,
                       op * show, grid=0.0)
            self.label(sc, self.shift(c, 0, -h / 2 - 8), name, 0.029, anchor="mb",
                       opacity=op * show, ink=layout.STRONG_INK)
            for k in range(0, N_PATCH, SIDE):
                self.label(sc, self.shift(c, -w / 2 - 6, (k + 0.5 - N_PATCH / 2) * h / N_PATCH),
                           str(k), 0.026, anchor="rm", opacity=op * show, ink=layout.NOTE_INK)
        for x0, x1, sym, at in ((xs[0], xs[1], "+", 0.15), (xs[1], xs[2], "=", 0.32)):
            self.label(sc, self.P((x0 + x1) / 2, y), sym, 0.031,
                       opacity=op * smoothstep(at, at + 0.1, u), ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, y + 0.205), "row k: patch k (0 .. 15) and grid place k, "
                   "64 numbers each", 0.026, opacity=op * smoothstep(0.0, 0.1, u),
                   ink=layout.NOTE_INK)
        blank = ", ".join(str(k) for k in self.blank)
        self.label(sc, self.P(0.5, 0.79), "the blank patches (%s) give x E = 0: their tokens "
                   "are b + pos[k], place alone" % blank, 0.027,
                   opacity=op * smoothstep(0.5, 0.6, u), ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.845), "learned, not a map of the grid: mean cosine %s "
                   "between neighbouring places, %s between places 3+ apart" % (
                       num(self.cos_near), num(self.cos_far)), 0.026,
                   opacity=op * smoothstep(0.65, 0.75, u), ink=layout.NOTE_INK)
        self.label(sc, self.P(0.5, 0.90), "then ReLU, the mean of the 16 tokens, and a 64 × 10 "
                   "layer: the 10 class scores", 0.026, opacity=op * smoothstep(0.8, 0.9, u),
                   ink=layout.NOTE_INK)

    # -- phase 4: the shuffle ---------------------------------------------------------------
    def _shuffle(self, sc, t, op):
        """The digit and a copy whose patches move to the shuffled places; the four
        predictions."""
        u = self.progress("shuffle", t)
        S = SHUF_PX / SIDE
        c_a, c_b = self.P(*SHUF_AT[0]), self.P(*SHUF_AT[1])
        draw_image(self, sc, c_a, gray_colors(self.img.ravel()).reshape(28, 28, 3), SHUF_PX,
                   SHUF_PX, op, grid=0.0)
        move = remap(u, 0.08, 0.38)
        # the scrambled image: place k shows patch order[k], which starts at its own place
        whole = self.img if move <= 0.0 else self.img_shuf if move >= 1.0 else None
        if whole is not None:
            draw_image(self, sc, c_b, gray_colors(whole.ravel()).reshape(28, 28, 3), SHUF_PX,
                       SHUF_PX, op, grid=0.0)
        for k in range(N_PATCH if whole is None else 0):
            src = int(self.order[k])
            r0, c0 = divmod(src, SIDE)
            r1, c1 = divmod(k, SIDE)
            v = ease_in_out(remap(move, 0.03 * src, 0.03 * src + 0.55))
            dx = ((c0 + (c1 - c0) * v) - 1.5) * S
            dy = ((r0 + (r1 - r0) * v) - 1.5) * S
            p = self.shift(c_b, dx, dy)
            draw_image(self, sc, p, self._patch_tex(self.xp[src]), S, S, op, grid=0.0,
                       lift=0.3 + 0.3 * (0 < v < 1))
            if 0 < v < 1:
                sc.probe(("shuf", k), p, tol=6.0)
        for c, name in ((c_a, "as it is"), (c_b, "patches shuffled")):
            self.label(sc, self.shift(c, 0, -SHUF_PX / 2 - 10), name, 0.028, anchor="mb",
                       opacity=op, ink=layout.STRONG_INK)
        # the predictions under each digit
        rows = (("a", "no positions", NO_POS_INK), ("b", "with positions", POS_INK))
        for s, c in enumerate((c_a, c_b)):
            show = smoothstep(0.42 + 0.12 * s, 0.52 + 0.12 * s, u)
            for r, (m, name, ink) in enumerate(rows):
                p = self.probs[m, s]
                guess = int(p.argmax())
                wrong = guess != self.label_y
                text = "%s: %d (p %.2f)%s" % (name, guess, p.max(), "  wrong" if wrong else "")
                self.label(sc, self.shift(c, 0, SHUF_PX / 2 + 14 + 24 * r), text, 0.027,
                           anchor="mt", opacity=op * show, ink=WRONG_INK if wrong else ink)
        notes = ("the same 16 patches,", "so the same mean:", "only the places",
                 "tell them apart")
        for r, line in enumerate(notes):
            self.label(sc, self.P(0.86, 0.42 + 0.05 * r), line, 0.027, opacity=op *
                       smoothstep(0.7 + 0.04 * r, 0.8 + 0.04 * r, u),
                       ink=layout.STRONG_INK if r < 2 else layout.CAPTION_INK)
        self.label(sc, self.P(0.43, 0.92), "test digit %d: the first that both models get "
                   "right, and only the one with positions misreads shuffled" % self.idx,
                   0.025, opacity=op * smoothstep(0.85, 0.95, u), ink=layout.NOTE_INK)

    # -- phase 5: the accuracies ------------------------------------------------------------
    def _accuracy(self, sc, t, op):
        """Test accuracy of both models, as the digits are and scrambled, as bars."""
        u = self.progress("accuracy", t)
        grow = ease_out(remap(u, 0.05, 0.45))
        x_name, x_bar = 0.27, 0.30
        bar_px = 260.0
        self.label(sc, self.P(x_bar, 0.33), "accuracy on the %s test digits" % (
            f"{self.n_test:,}"), 0.028, anchor="lm", opacity=op, ink=layout.NOTE_INK)
        rows = (("a", 0, "no positions", "as they are", NO_POS_INK),
                ("a", 1, "", "shuffled", NO_POS_INK),
                ("b", 0, "with positions", "as they are", POS_INK),
                ("b", 1, "", "shuffled", POS_INK))
        for r, (m, s, name, kind, ink) in enumerate(rows):
            y = 0.42 + 0.075 * r + 0.04 * (r >= 2)
            acc = self.acc[m, s]
            if name:
                self.label(sc, self.P(x_name, y), name, 0.028, anchor="rm", opacity=op,
                           ink=ink)
            L = max(2.0, bar_px * acc * grow)
            p = self.P(x_bar, y)
            fill = np.array(ink) if s == 0 else np.array(ink) * 0.5 + 0.5
            self.rect(sc, self.shift(p, L / 2, 0), L, 20, fill, op, radius=3.0)
            self.label(sc, self.shift(p, L + 10, 0), "%.1f%%  %s" % (100 * acc, kind),
                       0.027, anchor="lm", opacity=op * smoothstep(0.6, 1.0, grow),
                       ink=layout.STRONG_INK)
        late = smoothstep(0.5, 0.6, u)
        same = ("all %s" % f"{self.n_test:,}" if self.n_same == self.n_test else
                "%s of the %s" % (f"{self.n_same:,}", f"{self.n_test:,}"))
        self.label(sc, self.P(0.5, 0.80), "without positions, %s predictions are the same "
                   "either way: a mean cannot see order" % same, 0.027, opacity=op * late,
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.86), "a ViT's attention, like the mean, reads its tokens as "
                   "a set: the position vectors say where each patch was", 0.026,
                   opacity=op * smoothstep(0.7, 0.8, u), ink=layout.NOTE_INK)
        sc.probe(("acc",), (grow * 40, 0, 0), tol=6.0)
