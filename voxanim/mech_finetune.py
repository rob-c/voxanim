"""From a language model to an assistant, and what tuning does to it:
animations 35-42 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the mechanism first and the measurement after it. The numbers of
35-41 are the redesigned studies' (:mod:`voxanim.study_finetune`), which keep
what a mechanism needs beyond the four panels (the tuned models' answers at
every checkpoint, the policy at every update, the plane of merges); 42's are
the first study's (:mod:`voxanim.next_language`), and :func:`edit_trace` builds
its edit again, deterministically, to keep the edit itself.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import gpt2, realdata, theme
from . import study_finetune as F
from .cache import cached_npz
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
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

VERSION = 2
INK = (0.16, 0.17, 0.19)


def shown(ch):
    """A character as it is drawn in a token box."""
    return "␣" if ch == " " else "↵" if ch == "\n" else ch


def _settings(name, **extra):
    from . import next_language as L
    return dict(version=VERSION, name=name, study_version=L.VERSION, seed=L.SEED,
                dataset=realdata.TEXT_SHA256, **extra)


# ---------------------------------------------------------------------- data the studies did not keep  # noqa: E501
EDIT_SHOWN = 16         # entries of each factor of the edit that are drawn


@lru_cache(maxsize=1)
def edit_trace(layer, strength=40.0):
    """The knowledge-editing study's rank-one edit built again at one
    ``strength``: the passage's tokens, GPT-2's likeliest next tokens
    before and after, and the two factors of the edit (their first
    ``EDIT_SHOWN`` entries)."""
    def build():
        from . import gpt2_hooks
        from . import next_language as L
        tok = L._tok()
        ids = tok.encode(L._corpus()["test"][:160])
        clean, observed = np.asarray(ids[:9]), int(ids[9])
        target = next(i for i in tok.encode(L._corpus()["train"][400:500]) if i != observed)
        hold = tok.encode(L._corpus()["test"][300:430])
        control, ctarget = np.asarray(hold[:9]), int(hold[9])
        cr = gpt2_hooks.forward(clean, record=True)
        key = cr["mlp_hidden"][layer, -1].astype(float)
        _, records = L._activation_samples()
        background = np.concatenate([r["mlp_hidden"][layer] for r in records])
        W = gpt2.weights()
        direction = (W["wte.weight"][target] - W["wte.weight"][observed]).astype(float)
        direction /= max(np.linalg.norm(direction), 1e-12)
        name = "h.%d.mlp.c_proj.weight" % layer
        edit = L.rank_one_edit(key, direction * strength, background)
        # the edit is an outer product: (a column the key selects) x (the push)
        column = edit @ direction / strength
        before = gpt2.softmax(cr["logits"][-1])
        after = gpt2.softmax(gpt2_hooks.forward(clean, weight_edits={name: W[name] + edit})[
            "logits"][-1])
        cbefore = gpt2.softmax(gpt2_hooks.forward(control)["logits"][-1])
        cafter = gpt2.softmax(gpt2_hooks.forward(control, weight_edits={name: W[name] + edit})[
            "logits"][-1])
        top = np.argsort(-before)[:4]
        top_after = np.argsort(-after)[:4]
        ctop = np.argsort(-cbefore)[:4]

        def words(seq):
            return np.asarray([tok.decode([int(i)]) for i in seq])
        return dict(
            tokens=words(clean), observed=words([observed]), target=words([target]),
            control=words(control), ctarget=words([ctarget]),
            p_before=np.asarray([before[target], before[observed]]),
            p_after=np.asarray([after[target], after[observed]]),
            c_before=np.asarray(cbefore[ctarget]), c_after=np.asarray(cafter[ctarget]),
            top=words(top), top_p=np.c_[before[top], after[top]],
            top_after=words(top_after), top_after_p=np.c_[before[top_after], after[top_after]],
            ctop=words(ctop), ctop_p=np.c_[cbefore[ctop], cafter[ctop]],
            shape=np.asarray(edit.shape), column=column[:EDIT_SHOWN],
            push=(direction * strength)[:EDIT_SHOWN],
            block=edit[:EDIT_SHOWN, :EDIT_SHOWN], weight=W[name][:EDIT_SHOWN, :EDIT_SHOWN],
            moved=np.asarray(key @ edit @ direction),
            edit_norm=np.asarray(np.linalg.norm(edit)), weight_norm=np.asarray(
                np.linalg.norm(W[name])),
            samples=np.asarray(len(background)))
    return cached_npz("mech-finetune-edit", _settings("edit", layer=int(layer),
                                                      strength=float(strength),
                                                      gpt2=gpt2.REVISION), build)


# ---------------------------------------------------------------------- shared drawing
def pct(v):
    """A probability as a percentage."""
    v = 100.0 * float(v)
    if v >= 9.95:
        return "%.0f%%" % v
    if v >= 0.095:
        return "%.1f%%" % v
    return "%.2f%%" % v if v >= 0.005 or v == 0 else "%.1g%%" % v


def whole(v):
    """A share of answers as a percentage, none of them as 'none'."""
    return "none" if v == 0 else pct(v)


class Kit(Mechanism):
    """What these eight animations draw with, beyond :class:`Mechanism`."""

    def dash(self, sc, a, b, ink, op, n=14, width=1.3):
        """A dashed line between two layout points."""
        if op <= 0.01:
            return
        pa, pb = self.P(*a), self.P(*b)
        for i in range(n):
            sc.line(pa + (pb - pa) * (i / n), pa + (pb - pa) * ((i + 0.5) / n), width=width,
                    color=ink, opacity=op, bias=0.0)

    def spot(self, sc, fx, fy, r, ink, op, glow=0.0):
        """A flat dot at a layout point."""
        self.disk(sc, self.P(float(fx), float(fy)), r, ink, op, glow=glow, lift=1.4, outline=0.0)

    def wordbars(self, sc, x, y, words, values, full, ink, op, vmax=1.0, step=0.062, thick=22.0,
                 ref=None, fmt=pct, size=0.034, inks=None):
        """Rows of bars growing right from ``x``: one per word, ``full`` px at
        ``vmax``, each word to its left and its value at its end. ``ref``:
        a second value per row, drawn as a dark tick (where the bar started)."""
        for i, (word, v) in enumerate(zip(words, values)):
            at = self.P(x, y + i * step)
            colour = ink if inks is None else inks[i]
            self.rect(sc, self.shift(at, full / 2, 0), full, thick, (0.78, 0.79, 0.80), 0.45 * op,
                      radius=4, lift=0.3)
            w = max(full * min(float(v) / vmax, 1.0), 0.5)
            self.rect(sc, self.shift(at, w / 2, 0), w, thick, colour, op, radius=min(4.0, w / 2))
            end = w
            if ref is not None:
                r = full * min(float(ref[i]) / vmax, 1.0)
                self.rect(sc, self.shift(at, r, 0), 3.0, thick + 8, INK, op, radius=1, lift=0.8)
                end = max(end, r + 2)
            self.label(sc, self.shift(at, -10, 0), word, size, anchor="rm", opacity=op)
            self.label(sc, self.shift(at, end + 8, 0), fmt(v), size, anchor="lm", opacity=op,
                       ink=INK)

    def over(self, sc, a, b, ink, op, u=1.0, width=2.2, head=True, lift=1.0):
        """A straight arrow between two layout points, drawn above images
        and boxes (whose own lift hides :meth:`arrow`)."""
        if op <= 0.01 or u <= 0.0:
            return
        pa, pb = self.P(a[0], a[1], lift), self.P(b[0], b[1], lift)
        end = pa + (pb - pa) * min(float(u), 1.0)
        self.polyline(sc, [pa, end], width, ink, op, bias=0.3)
        if head:
            self._head(sc, pa, end, ink, op, width)

    def plate(self, sc, fx, fy, w, h, fill, op, border=None, radius=3.0, glow=0.0, lift=1.0):
        """A box drawn above the boxes it sits on."""
        self.rect(sc, self.P(float(fx), float(fy)), w, h, fill, op, border=border, radius=radius,
                  glow=glow, lift=lift)

    def stage(self, u, n):
        """Which of ``n`` equal turns ``u`` is in, and how far through it."""
        k = min(int(u * n), n - 1)
        return k, float(u * n - k)


# ---------------------------------------------------------------------- 35
class InstructionTuning(Kit):
    """Supervised fine-tuning on (instruction, answer) examples: one real line
    under two instructions, which of the model's guesses each loss grades,
    both fine-tunes measured on held-out answers, and what they then write."""

    TITLE = "Instruction tuning"
    PHASES = (("two instructions", 11.0), ("which guesses are graded", 12.0),
              ("fine-tuning", 13.0), ("its answers", 14.0))
    PAIR = (0, 1)                       # a test line under Echo: and under Caps:
    HELD = (2, 3)                       # the line answered in the last phase, both ways
    ONLY, EVERY = GREEN, RED            # answer-only loss, all-token loss

    def setup(self):
        s, r = self.study, F.instruction_runs()
        self.rows = [[shown(c) for c in realdata.decode(np.r_[x, y[-1:]])]
                     for x, y in zip(r["show_x"], r["show_y"])]         # 22 characters each
        self.mask = np.asarray(s.panels[1].data["values"], float)
        self.steps = np.asarray(s.panels[2].data["x"], float)
        self.acc = np.asarray(s.panels[3].data["ys"], float)            # mean of the seeds
        self.each = np.asarray(s.panels[3].data["each"], float)         # seed, run, checkpoint
        self.base = float(np.mean(s.panels[3].data["base"]))
        self.reach = F.reached(r["trace"])                              # seed, run
        self.guess, self.base_guess = r["guess"], r["base_guess"]
        self.truth = r["show_y"][:, -F.LINE:]
        self.counts = [int(v) for v in r["counts"]]
        self.written = r["written"].mean(0)
        self.base_written = float(r["base_written"].mean())
        self.seeds = len(self.each)
        self.updates = int(s.metadata["settings"]["adaptation_steps"])
        trace = np.asarray(r["trace"], float)
        self.line = (float(r["base"][:, 3].mean()), *trace[:, :, 3, -1].mean(0))

    def by(self, run, short=False):
        """When a loss's runs reach 90% of held-out answer characters, in words."""
        at = self.reach[:, run]
        if (at > self.updates).all():
            return ("not by %d" if short else "not within %d updates in any run") % self.updates
        if (at > self.updates).any():
            return ("%d of %d runs" if short else "in %d of %d runs") % (
                (at <= self.updates).sum(), len(at))
        return "%d to %d" % (at.min(), at.max()) if short else "by update %d on average" % (
            round(at.mean() / 10) * 10)

    def subtitle_for(self, k):
        n, g = self.mask.shape[1], int(self.mask[0].sum())
        return [
            "one training example: an instruction, eight characters of real Shakespeare, and "
            "the answer, which is those characters again for Echo: and in capitals for Caps:",
            "the model guesses every next character; the all-token loss grades all %d guesses, "
            "the answer-only loss just the %d that spell the answer" % (n, g),
            "%d base models, each tuned both ways on the same batches: held-out answers are "
            "90%% right %s with the answer-only loss; with the all-token loss, %s" % (
                self.seeds, self.by(0), self.by(1)),
            "the answer-only models write %s of %d held-out answers out whole, the all-token "
            "models %s, the untuned bases %s: graded on everything, the updates went to "
            "guessing the line" % (whole(self.written[0]), self.counts[2],
                                   whole(self.written[1]), whole(self.base_written)),
        ][k]

    # the 22 characters in a row: what part each is
    PARTS = (("instruction", AMBER, 0, 5), ("a real line", BLUE, 5, 13), ("", AMBER, 13, 14),
             ("the answer", GREEN, 14, 22))

    def _example(self, sc, chars, op, x0, x1, y, w, h, show=(1, 1, 1), size=0.034):
        """An example's 22 characters; ``show``: opacity of the line, of the
        wrapping and of the answer. Returns their x fractions."""
        n = len(chars)
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        for name, ink, a, b in self.PARTS:
            o = op * show[{"a real line": 0, "the answer": 2}.get(name, 1)]
            for i in range(a, b):
                self.box(sc, xs[i], y, w, h, soft(ink), o, chars[i], ink=dark(ink, 0.35),
                         border=ink, size=size)
        return xs

    def phase_1(self, sc, u, op):
        wrap, ans = smoothstep(0.22, 0.34, u), smoothstep(0.48, 0.60, u)
        for row, e in enumerate(self.PAIR):
            xs = self._example(sc, self.rows[e], op, 0.05, 0.95, 0.44 + row * 0.15, 35.0, 46.0,
                               (1.0, wrap, ans), 0.038)
        self.text(sc, 0.5, 0.34, "eight characters in a row from Tiny Shakespeare", op * (
            1 - smoothstep(0.18, 0.24, u)), size=0.036, strong=True)
        self.text(sc, 0.5, 0.34, "wrapped in an instruction: Echo: or Caps:", op * wrap * (
            1 - smoothstep(0.44, 0.50, u)), size=0.036, strong=True)
        self.text(sc, 0.5, 0.34, "the answer depends on the instruction", op * ans, size=0.036,
                  strong=True)
        for name, ink, a, b in self.PARTS:
            if not name:
                continue
            o = op * {"a real line": 1.0, "the answer": ans}.get(name, wrap)
            mid = (xs[a] + xs[b - 1]) / 2
            self.polyline(sc, [self.P(xs[a] - 0.016, 0.655), self.P(xs[a] - 0.016, 0.67),
                               self.P(xs[b - 1] + 0.016, 0.67), self.P(xs[b - 1] + 0.016, 0.655)],
                          2.0, ink, o)
            self.text(sc, mid, 0.715, name, o, size=0.038, ink=dark(ink))
        self.lines(sc, 0.5, 0.82, [
            "%s examples to train on, none seen twice; %d more held out" % (
                f"{self.counts[0]:,}", self.counts[2]),
            "no person wrote these: every line is from the play, every answer follows from it"],
            op * smoothstep(0.72, 0.84, u), size=0.034, step=0.058, anchor="mm")

    def phase_2(self, sc, u, op):
        x0, x1, y = 0.25, 0.95, 0.40
        xs = self._example(sc, self.rows[self.PAIR[1]], op, x0, x1, y, 27.0, 42.0, size=0.032)
        n = self.mask.shape[1]                                  # 21 guesses
        sweep = min(u / 0.55, 1.0)
        turn = min(int(sweep * n), n - 1)
        within = sweep * n - turn if sweep < 1 else 1.0
        done = turn + (1 if within > 0.7 else 0)
        if sweep < 1:
            a, b = (xs[turn], y - 0.052), (xs[turn + 1], y - 0.052)
            self.arc(sc, a, b, INK, op, smoothstep(0, 0.5, within), 0.05, 2.4)
            self.pulse(sc, a, b, within, INK, op, 0.05)
        self.text(sc, 0.05, y, "guess the next one", op, anchor="lm", size=0.034, strong=True)
        for row, (name, ink, yy) in enumerate((("all-token loss", self.EVERY, 0.515),
                                               ("answer-only loss", self.ONLY, 0.585))):
            graded = 0
            for g in range(n):
                counts = row == 0 or self.mask[0, g] > 0
                if g < done:
                    graded += counts
                    self.box(sc, xs[g + 1], yy, 25, 22, ink if counts else DIM, op * (
                        1.0 if counts else 0.55), radius=4)
            self.text(sc, 0.05, yy, name, op, anchor="lm", size=0.034, ink=dark(ink))
            self.text(sc, 0.235, yy, "%d" % graded, op, anchor="rm", size=0.036, strong=True)
        # the study's mask for six examples, its columns under the guesses they grade
        o = op * smoothstep(0.58, 0.66, u)
        w = self.span(xs[n] - xs[1])[0] * n / (n - 1)
        self.grid(sc, self.mask, (xs[1] + xs[n]) / 2, 0.76, w, 118.0, o,
                  reveal=smoothstep(0.62, 0.9, u), hot=self.ONLY, mask=True)
        self.lines(sc, 0.05, 0.72, ["the answer-only", "mask, six examples:", "green = graded"], o,
                   size=0.032, step=0.045)

    def _legend(self, sc, x, y, op, values=None, step=0.06, size=0.036):
        for row, (name, ink) in enumerate((("answer-only", self.ONLY), ("all-token", self.EVERY))):
            s = name if values is None else "%s  %s" % (name, values[row])
            self.text(sc, x, y + row * step, s, op, anchor="lm", size=size, ink=dark(ink))

    def phase_3(self, sc, u, op):
        # one base model, copied, tuned two ways
        self.box(sc, 0.14, 0.36, 150, 44, PAPER, op, "base model", border=AXIS, size=0.034)
        for row, (name, ink) in enumerate((("graded on the answer only", self.ONLY),
                                           ("graded on every character", self.EVERY))):
            yy = 0.33 + row * 0.07
            self.arrow(sc, (0.225, 0.36), (0.33, yy), ink, op, smoothstep(0.0, 0.12, u))
            self.pulse(sc, (0.225, 0.36), (0.33, yy), (u * 6) % 1.0, ink, op * (u < 0.86))
            self.box(sc, 0.48, yy, 270, 30, soft(ink), op, name, ink=dark(ink, 0.35), border=ink,
                     size=0.032)
        chart = self.chart(((0.10, 0.60), (0.50, 0.84)), self.steps, [0.0, 1.0], pad=0.03)
        self.axes(sc, chart, op, "fine-tuning update", "held-out answer characters right",
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        chart.hline(sc, self.P, self.base, op, dashes=20, width=1.3)
        self.text(sc, 0.595, chart.fy(self.base) + 0.03, "untuned: %s" % pct(self.base), op,
                  anchor="rm", size=0.03)
        go = u / 0.85
        for run, ink in enumerate((self.ONLY, self.EVERY)):
            for seed in range(self.seeds):
                self.plot(sc, chart, self.steps, self.each[seed, run], ink, op * 0.4, go,
                          width=1.2, dot=False)
            self.plot(sc, chart, self.steps, self.acc[run], ink, op, go, width=2.6)
        now = int(min(go, 1.0) * (len(self.steps) - 1))
        self.text(sc, 0.68, 0.50, "update %d" % self.steps[now], op, anchor="lm", size=0.036,
                  strong=True)
        self._legend(sc, 0.68, 0.57, op, [pct(self.acc[0][now]), pct(self.acc[1][now])])
        self.lines(sc, 0.68, 0.70, [
            "90% right by update:",
            "answer-only  %s" % self.by(0, True),
            "all-token  %s" % self.by(1, True),
            "(thin lines: the %d seeds)" % self.seeds],
            op * smoothstep(0.6, 0.75, u), size=0.032, step=0.046)

    def phase_4(self, sc, u, op):
        k = int(min(u / 0.8, 1.0) * (len(self.steps) - 1))
        pitch = 0.034
        for block, e in enumerate(self.HELD):
            y = 0.345 + block * 0.27
            chars = self.rows[e]
            xs = [0.094] + [0.165 + pitch * i for i in range(9)]
            for i, (ch, x) in enumerate(zip(["".join(chars[:5])] + chars[5:14], xs)):
                ink = AMBER if i in (0, 9) else BLUE
                self.box(sc, x, y, 84.0 if i == 0 else 29.0, 40.0, soft(ink), op, ch,
                         ink=dark(ink, 0.35), border=ink, size=0.032)
            truth = self.truth[e]
            rows = (("the right answer", truth, GREEN), ("untuned base", self.base_guess[e], DIM),
                    ("answer-only", self.guess[0, k, e], self.ONLY),
                    ("all-token", self.guess[1, k, e], self.EVERY))
            self.arrow(sc, (xs[-1] + 0.026, y), (0.50, y), AXIS, op)
            for r, (name, ids, ink) in enumerate(rows):
                yy = y + r * 0.058
                for j in range(F.LINE):
                    ok = ids[j] == truth[j]
                    self.box(sc, 0.53 + pitch * j, yy, 29.0, 27.0 if r else 40.0,
                             soft(GREEN) if ok else PAPER, op,
                             shown(realdata.decode([int(ids[j])])),
                             ink=dark(GREEN, 0.35) if ok else None,
                             border=GREEN if ok else DIM, size=0.032)
                self.text(sc, 0.80, yy, name, op, anchor="lm", size=0.032,
                          ink=dark(ink) if r > 1 else None, strong=not r)
        self.lines(sc, 0.05, 0.425, ["one held-out line, asked both ways;",
                                     "a green box is a correct character"], op,
                   size=0.032, step=0.045)
        self.text(sc, 0.05, 0.895, "update %d" % self.steps[k], op, anchor="lm", size=0.036,
                  strong=True)
        self.text(sc, 0.22, 0.895, "right, over all %d held-out answers:" % self.counts[2], op,
                  anchor="lm", size=0.034)
        self.text(sc, 0.585, 0.895, "answer-only  %s" % pct(self.acc[0][k]), op, anchor="lm",
                  size=0.036, ink=dark(self.ONLY))
        self.text(sc, 0.79, 0.895, "all-token  %s" % pct(self.acc[1][k]), op, anchor="lm",
                  size=0.036, ink=dark(self.EVERY))


# ---------------------------------------------------------------------- the preference studies
def preferences():
    """The preference pairs, the reward models and everything optimised
    against them: what the four preference studies share
    (:func:`voxanim.study_finetune.preferences`)."""
    return F.preferences()


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-np.asarray(v, float)))


def signed(v, digits=2):
    """A score with a true minus sign."""
    return ("%+.*f" % (digits, float(v))).replace("-", "−")


# ---------------------------------------------------------------------- 36
class RewardModel(Kit):
    """Preference pairs made from the text, a scorer trained so the preferred
    word scores higher (Bradley-Terry), what it computes a score from, and how
    well its ranking holds on held-out pairs beside two weaker scorers."""

    TITLE = "A reward model from preferences"
    PHASES = (("preference pairs", 10.0), ("score the pair", 15.0), ("training", 12.0),
              ("the ranking", 13.0))
    WIN, LOSE, MODEL, HELD, PREV = GREEN, RED, AMBER, TEAL, BLUE
    KINDS = (("two word vectors", AMBER), ("a score per word", VIOLET),
             ("three spelling facts", OCHRE), ("a coin flip", DIM))

    def setup(self):
        s, d = self.study, preferences()
        self.words = list(d["words"])
        self.cands = list(d["candidates"])
        self.pairs, self.rejected = np.asarray(d["train_pairs"]), np.asarray(d["train_rejected"])
        self.u, self.v, self.b = (np.asarray(d[k], float) for k in "uvb")
        self.table = self.u @ self.v.T + self.b                 # score[previous, next]
        self.true, self.reward = np.asarray(s.panels[2].data["ys"], float)
        self.rank = float(s.panels[2].data["rank_correlation"][0])
        self.gap_x = np.asarray(s.panels[0].data["x"], float)
        self.link = np.asarray(s.panels[0].data["ys"], float)[0]
        self.loss = np.asarray(s.panels[1].data["ys"], float)[0]
        self.val = np.asarray(s.panels[1].data["val"], float)
        self.val_x = np.asarray(s.panels[1].data["val_x"], float)
        self.stop = int(s.panels[1].data["stop"])
        self.accuracy = np.asarray(s.panels[3].data["values"], float)
        self.low = np.asarray(s.panels[3].data["low"], float)
        self.high = np.asarray(s.panels[3].data["high"], float)
        self.counts = [int(v) for v in d["counts"]]
        gaps = self.gap(np.arange(len(self.pairs)))
        wrong = int(np.argmax(gaps < -0.3))                 # one the trained scorer gets wrong
        self.scored = [0, 1, wrong, 4]
        self.train_right = float((gaps > 0).mean())

    def gap(self, i):
        """Score of pair ``i``'s preferred word minus its rejected word's."""
        c, w = self.pairs[i].T
        return self.table[c, w] - self.table[c, self.rejected[i]]

    def subtitle_for(self, k):
        a = self.accuracy
        return [
            "after each word of the play, the word that really comes next is preferred to a "
            "word drawn at random: %s such pairs from the training text, not judged by "
            "people" % f"{self.counts[0]:,}",
            "a reward model gives each word a score; the gap between two scores, read off this "
            "curve, is how sure it is that the first word is the preferred one",
            "%d updates push the preferred word's score above the other's: the pair loss falls "
            "from %.3f to %.3f, and to %.3f on validation pairs; a coin flip scores %.3f" % (
                len(self.loss), self.loss[0], self.loss[-1], self.val.min(), np.log(2)),
            "on %s held-out pairs it ranks the real word higher %.1f%% of the time; a score "
            "per word with no context %.1f%%, three facts about its spelling %.1f%%" % (
                f"{self.counts[2]:,}", 100 * a[0], 100 * a[1], 100 * a[2]),
        ][k]

    def _word(self, sc, x, y, i, ink, op, w=116.0, h=46.0):
        self.box(sc, x, y, w, h, soft(ink), op, self.words[i], ink=dark(ink, 0.35), border=ink,
                 size=0.038)

    def phase_1(self, sc, u, op):
        self.text(sc, 0.15, 0.33, "a word of the play", op, size=0.034, ink=dark(self.PREV))
        self.text(sc, 0.47, 0.33, "what really came next", op, size=0.034, ink=dark(self.WIN))
        self.text(sc, 0.83, 0.33, "a word drawn at random", op, size=0.034, ink=dark(self.LOSE))
        for r in range(5):
            start = 0.04 + r * 0.15
            o = op * smoothstep(start + 0.07, start + 0.12, u)
            y = 0.42 + r * 0.095
            c, w = self.pairs[r]
            self._word(sc, 0.15, y, c, self.PREV, op * smoothstep(start, start + 0.05, u))
            a, b = (0.22, y), (0.40, y)
            self.arrow(sc, a, b, self.WIN, op * 0.8, smoothstep(start, start + 0.08, u), 1.8)
            self.pulse(sc, a, b, (u - start) / 0.1, self.WIN, op)
            self._word(sc, 0.47, y, w, self.WIN, o)
            self.text(sc, 0.65, y, "preferred to", o, size=0.032)
            self._word(sc, 0.83, y, self.rejected[r], self.LOSE, o)
        self.text(sc, 0.5, 0.915, "... and %s more pairs like these, with a new random word "
                  "each time one is used" % f"{self.counts[0] - 5:,}",
                  op * smoothstep(0.8, 0.9, u), size=0.034, strong=True)

    def phase_2(self, sc, u, op):
        k, t = self.stage(u, len(self.scored))
        pair = self.scored[k]
        c, w = (int(v) for v in self.pairs[pair])
        l = int(self.rejected[pair])
        rw, rl = float(self.table[c, w]), float(self.table[c, l])
        gap = rw - rl
        fade = op * smoothstep(0.0, 0.06, t) * (1 - smoothstep(0.94, 1.0, t) * (k < 3))
        self.box(sc, 0.305, 0.54, 120, 150, soft(self.MODEL, 0.35), op, border=self.MODEL)
        self.lines(sc, 0.305, 0.52, ["reward", "model"], op, size=0.036, step=0.05, anchor="mm")
        self.text(sc, 0.05, 0.325, "after '%s':" % self.words[c], fade, anchor="lm", size=0.036,
                  ink=dark(self.PREV), strong=True)
        for row, (i, ink, r, y) in enumerate(((w, self.WIN, rw, 0.44), (l, self.LOSE, rl, 0.65))):  # noqa: B007
            self._word(sc, 0.115, y, i, ink, fade)
            a, b = (0.18, y), (0.238, y)
            self.arrow(sc, a, b, ink, fade, smoothstep(0.05, 0.15, t))
            self.pulse(sc, a, b, (t - 0.1) / 0.15, ink, fade)
            o = fade * smoothstep(0.25, 0.32, t)
            self.arrow(sc, (0.372, y), (0.40, y), ink, o)
            self.text(sc, 0.41, y, signed(r), o, size=0.042, anchor="lm", ink=dark(ink))
        self.text(sc, 0.115, 0.525, "preferred", fade, size=0.032, ink=dark(self.WIN))
        self.text(sc, 0.115, 0.735, "rejected", fade, size=0.032, ink=dark(self.LOSE))
        o = fade * smoothstep(0.36, 0.44, t)
        self.text(sc, 0.05, 0.86, "gap between the scores: %s" % signed(gap), o, size=0.038,
                  anchor="lm", strong=True)
        # the curve, and the gap read off it
        keep = np.abs(self.gap_x) <= 6.0 + 1e-9
        chart = self.chart(((0.60, 0.94), (0.37, 0.74)), self.gap_x[keep], [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "gap between the two scores", "how sure it is (probability)",
                  xticks=[(v, signed(v, 0) if v else "0") for v in (-4, -2, 0, 2, 4)],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        self.plot(sc, chart, self.gap_x[keep], self.link[keep], self.MODEL, op, 1.0, dot=False)
        at = float(np.clip(gap, -6, 6)) * smoothstep(0.44, 0.7, t)
        prob = float(sigmoid(at))
        x, y = chart.fx(at), chart.fy(prob)
        self.dash(sc, (x, chart.y1), (x, y), INK, o * 0.45, n=8)
        self.dash(sc, (chart.x0, y), (x, y), INK, o * 0.45, n=10)
        self.spot(sc, x, y, 7.5, self.WIN if gap > 0 else self.LOSE, o, glow=0.5)
        done = fade * smoothstep(0.7, 0.78, t)
        final = float(sigmoid(gap))
        self.text(sc, 0.60, 0.86, "%s sure of the preferred word" % pct(final), done, size=0.038,
                  anchor="lm", strong=True)
        self.text(sc, 0.60, 0.915, "loss on this pair: %.2f%s" % (
            -np.log(final), ": the wrong way round" if gap < 0 else ""), done,
            size=0.032, anchor="lm")

    def phase_3(self, sc, u, op):
        steps = np.arange(1, len(self.loss) + 1)
        chart = self.chart(((0.10, 0.50), (0.48, 0.80)), steps, np.r_[self.loss, self.val,
                                                                    np.log(2)])
        self.axes(sc, chart, op, "training update", "pair loss")
        chart.hline(sc, self.P, float(np.log(2)), op, dashes=20, width=1.3)
        self.text(sc, 0.49, chart.fy(np.log(2)) + 0.03, "a coin flip: %.3f" % np.log(2), op,
                  anchor="rm", size=0.032)
        go = u / 0.7
        self.plot(sc, chart, steps, self.loss, self.MODEL, op, go, width=2.3)
        now = int(min(go, 1.0) * (len(steps) - 1))
        for x, v in zip(self.val_x, self.val):
            if x <= steps[now]:
                self.spot(sc, chart.fx(max(x, 1)), chart.fy(v), 3.6, self.HELD, op)
        self.text(sc, 0.10, 0.34, "update %d    loss %.3f" % (steps[now], self.loss[now]), op,
                  anchor="lm", size=0.036, strong=True)
        self.text(sc, 0.10, 0.385, "dots: pairs from validation text it never trains on", op,
                  anchor="lm", size=0.03, ink=dark(self.HELD))
        # what a score is made of: the pair's two learned vectors, multiplied and added up
        c, w = (int(v) for v in self.pairs[self.scored[0]])
        o = op * smoothstep(0.1, 0.25, u)
        self.text(sc, 0.77, 0.34, "what a score is computed from", op, size=0.034, strong=True)
        top = float(max(np.abs(self.u[c]).max(), np.abs(self.v[w]).max()))
        for r, (name, vec, ink) in enumerate((("'%s' as the word before" % self.words[c],
                                               self.u[c], self.PREV),
                                              ("'%s' as the word after" % self.words[w],
                                               self.v[w], self.WIN))):
            y = 0.455 + r * 0.135
            self.text(sc, 0.77, y - 0.062, name, o, size=0.032, ink=dark(ink))
            self.grid(sc, vec[None, :], 0.77, y, 264.0, 28.0, o, vmax=top)
        prod = self.u[c] * self.v[w]
        o2 = op * smoothstep(0.4, 0.55, u)
        self.text(sc, 0.77, 0.665, "multiplied, slot by slot", o2, size=0.032)
        self.grid(sc, prod[None, :], 0.77, 0.725, 264.0, 28.0, o2, vmax=float(
            np.abs(prod).max()), hot=self.MODEL, cold=VIOLET)
        o3 = op * smoothstep(0.6, 0.75, u)
        self.text(sc, 0.77, 0.805, "added up %s, plus %s for '%s' itself" % (
            signed(prod.sum()), signed(self.b[w]), self.words[w]), o3, size=0.032)
        self.text(sc, 0.77, 0.865, "score %s" % signed(self.table[c, w]), o3, size=0.040,
                  ink=dark(self.MODEL), strong=True)
        self.text(sc, 0.10, 0.92, "eight learned numbers a word in each role, for %d words" % len(
            self.words), op * smoothstep(0.8, 0.9, u), anchor="lm", size=0.032)

    def phase_4(self, sc, u, op):
        share = np.exp(self.true)
        chart = self.chart(((0.11, 0.45), (0.40, 0.80)), self.reward, share, ylog=True)
        self.axes(sc, chart, op, "its score for the word after 'the'",
                  "how often it follows 'the' in held-out text",
                  yticks=[(v, pct(v)) for v in (0.003, 0.01, 0.03, 0.1)], grid=False)
        order = np.argsort(self.reward)
        upto = int(len(order) * smoothstep(0.0, 0.45, u))
        for i in order[:upto]:
            self.spot(sc, chart.fx(self.reward[i]), chart.fy(share[i]), 5.0, self.HELD, op)
        best = int(np.argmax(self.reward))
        if best in order[:upto]:
            x, y = chart.fx(self.reward[best]), chart.fy(share[best])
            self.spot(sc, x, y, 6.5, self.MODEL, op, glow=0.4)
            self.text(sc, x + 0.014, y, self.cands[best], op, size=0.034, anchor="lm",
                      strong=True)
        o = op * smoothstep(0.5, 0.6, u)
        full = self.span(0.25)[0]
        self.text(sc, 0.56, 0.335, "on %s held-out pairs, the real word" % f"{self.counts[2]:,}", o, anchor="lm", size=0.032, strong=True)  # noqa: E501
        self.text(sc, 0.56, 0.38, "is ranked higher by a scorer that sees", o, anchor="lm",
                  size=0.032, strong=True)
        grow = (u - 0.52) / 0.25
        for r, ((name, ink), v, lo, hi) in enumerate(zip(self.KINDS, self.accuracy, self.low,  # noqa: B007
                                                         self.high)):
            y = 0.485 + r * 0.108
            self.text(sc, 0.56, y - 0.045, name, o, anchor="lm", size=0.032,
                      ink=dark(ink) if ink is not DIM else None)
            self.hbar(sc, 0.56, y, full * v, ink, o, "", "%.1f%%" % (100 * v), grow, thick=24.0,
                      track=full)
        self.lines(sc, 0.11, 0.905, ["each dot is one of the %d words tried after 'the'; rank "
                                     "agreement %.2f (1 would be perfect)" % (
                                         len(self.cands), self.rank)],
                   op * smoothstep(0.8, 0.9, u), size=0.03, step=0.046)


# ---------------------------------------------------------------------- 37
class RlhfPpo(Kit):
    """A policy over the next word, pushed by PPO towards what the reward
    model scores highly and held back by a KL penalty to the reference: the
    loop, the policy moving at three penalty strengths, and reward against
    drift."""

    TITLE = "RLHF with PPO"
    PHASES = (("the loop", 12.0), ("no leash", 12.0), ("a leash", 13.0),
              ("reward against drift", 12.0))
    INKS = (RED, VIOLET, BLUE)
    MODEL = AMBER
    SAMPLED = ("king", "the", "queen", "lord")

    def setup(self):
        s, d = self.study, preferences()
        self.words = list(s.metadata["settings"]["candidate_words"])
        self.BETAS = tuple(s.metadata["settings"]["betas"])
        self.score, self.kl, self.held = (np.asarray(p.data["ys"], float) for p in s.panels[:3])
        self.final = np.asarray(s.panels[3].data["ys"], float)[:3]
        self.ref = np.asarray(s.panels[3].data["ys"], float)[3]
        self.reward = np.asarray(d["rewards"], float)[0]
        self.start = float(self.ref @ self.reward)          # the reference policy's own score
        self.dists = list(np.asarray(d["ppo_p"], float))
        self.seeds_kl = np.asarray(s.panels[1].data["each"], float)[:, :, -1]   # seed, beta
        self.seeds_score = np.asarray(s.panels[0].data["each"], float)[:, :, -1]
        top = list(np.argsort(-self.ref)[:5])
        for row in self.final:
            top += [i for i in np.argsort(-row)[:3] if i not in top]
        self.rows = top[:7]
        self.updates = self.score.shape[1]

    def subtitle_for(self, k):
        sc, kl = self.score[:, -1], self.kl[:, -1]
        best = int(np.argmax(self.final[0]))
        return [
            "the loop: the policy picks a word to follow 'the', the reward model scores it, and "
            "PPO shifts the policy towards words that score above its average",
            "with no penalty for leaving the reference, %d updates put %s of the policy on '%s' "
            "and its drift from the reference (the KL divergence) reaches %.2f nats" % (
                self.updates, pct(self.final[0][best]), self.words[best], kl[0]),
            "a penalty of β times the drift is the leash: at β = %s the policy still moves "
            "%.2f nats, at β = %s only %.2f" % (
                number(self.BETAS[1]), kl[1], number(self.BETAS[2]), kl[2]),
            "the trade: reward model score %.2f for %.2f nats of drift with no leash, %.2f for "
            "%.2f at β = %s, and only %.2f for %.2f at β = %s" % (
                sc[0], kl[0], sc[1], kl[1], number(self.BETAS[1]), sc[2], kl[2],
                number(self.BETAS[2])),
        ][k]

    def phase_1(self, sc, u, op):
        k, t = self.stage(u, len(self.SAMPLED))
        i = self.words.index(self.SAMPLED[k])
        r = float(self.reward[i])
        up = r > self.start
        ink = GREEN if up else RED
        pol, tok, rm, ref = (0.20, 0.50), (0.50, 0.37), (0.80, 0.50), (0.20, 0.82)
        self.box(sc, *pol, 220, 96, PAPER, op, border=INK)
        self.lines(sc, pol[0], pol[1] - 0.025, ["the policy", "a chance for each word"], op,
                   size=0.034, step=0.05, anchor="mm")
        self.box(sc, *rm, 190, 96, soft(self.MODEL, 0.35), op, border=self.MODEL)
        self.lines(sc, rm[0], rm[1] - 0.025, ["reward model", "scores the word"], op, size=0.034,
                   step=0.05, anchor="mm")
        self.box(sc, *ref, 220, 56, soft(DIM, 0.3), op, "the reference", border=AXIS, size=0.034)
        self.dash(sc, (pol[0], pol[1] + 0.095), (ref[0], ref[1] - 0.058), INK, op * 0.7, n=5,
                  width=2.0)
        self.lines(sc, 0.335, 0.80, ["the leash: a penalty of β × how far",
                                     "the policy has drifted from it"], op, size=0.032,
                   step=0.045)
        fade = op * smoothstep(0.0, 0.06, t) * (1 - smoothstep(0.93, 1.0, t))
        # policy -> the word -> reward model -> its score -> back to the policy
        a, b = (pol[0] + 0.12, pol[1] - 0.06), (tok[0] - 0.075, tok[1])
        self.arc(sc, a, b, INK, op * 0.8, 1.0, 0.05, 2.0)
        self.pulse(sc, a, b, t / 0.2, INK, op, 0.05)
        self.box(sc, *tok, 120, 46, soft(ink), fade * smoothstep(0.15, 0.22, t),
                 self.words[i], ink=dark(ink, 0.35), border=ink, size=0.040)
        self.text(sc, 0.34, 0.345, "picks", op, size=0.032)
        a, b = (tok[0] + 0.075, tok[1]), (rm[0] - 0.105, rm[1] - 0.06)
        self.arc(sc, a, b, INK, op * 0.8, 1.0, 0.05, 2.0)
        self.pulse(sc, a, b, (t - 0.22) / 0.2, ink, op, 0.05)
        a, b = (rm[0] - 0.105, rm[1] + 0.06), (pol[0] + 0.12, pol[1] + 0.06)
        self.arc(sc, a, b, self.MODEL, op * 0.9, 1.0, -0.16, 2.2)
        self.pulse(sc, a, b, (t - 0.5) / 0.3, ink, op, -0.16, r=6.0)
        o = fade * smoothstep(0.42, 0.5, t)
        self.text(sc, 0.50, 0.50, "score %s" % signed(r), o, size=0.042, ink=dark(ink))
        self.text(sc, 0.64, 0.69, "%s the policy's average, %s:" % (
            "above" if up else "below", signed(self.start)), o, size=0.032)
        self.text(sc, 0.64, 0.735, "make '%s' %s likely" % (
            self.words[i], "more" if up else "less"), fade * smoothstep(0.7, 0.8, t), size=0.034,
            strong=True)

    def _at(self, u, span=0.85):
        """The update being shown: early ones, where most happens, linger."""
        return int(round(self.updates * min(u / span, 1.0) ** 1.7))

    def _status(self, sc, x, y, run, n, op):
        m = max(n - 1, 0)
        score = self.score[run][m] if n else self.start
        kl = self.kl[run][m] if n else 0.0
        self.text(sc, x, y, "score %.2f    drift %.2f nats" % (score, kl), op, anchor="lm",
                  size=0.034, ink=dark(self.INKS[run]))

    def phase_2(self, sc, u, op):
        n = self._at(u)
        words = [self.words[i] for i in self.rows]
        self.text(sc, 0.05, 0.325, "the policy after 'the'  (%s mark: the reference)"
                  % theme.word("dark", "white"), op,
                  anchor="lm", size=0.032, strong=True)
        self.wordbars(sc, 0.15, 0.39, words, self.dists[0][n][self.rows], 230.0, self.INKS[0], op,
                      ref=self.ref[self.rows], step=0.066)
        self.text(sc, 0.05, 0.875, "update %d" % n, op, anchor="lm", size=0.036, strong=True)
        self._status(sc, 0.20, 0.875, 0, n, op)
        steps = np.arange(1, self.updates + 1)
        chart = self.chart(((0.60, 0.94), (0.40, 0.76)), np.r_[0, steps], np.r_[0, self.kl[0]])
        self.axes(sc, chart, op, "PPO update", "drift from the reference (KL, nats)")
        if n:
            self.plot(sc, chart, np.r_[0, steps], np.r_[0, self.kl[0]], self.INKS[0], op,
                      n / self.updates, width=2.3)

    def phase_3(self, sc, u, op):
        n = self._at(u)
        words = [self.words[i] for i in self.rows]
        for col, run in enumerate((1, 2)):
            x = 0.15 + col * 0.46
            self.text(sc, x, 0.325, "leash β = %s" % number(self.BETAS[run]), op, anchor="lm",
                      size=0.036, ink=dark(self.INKS[run]))
            self.wordbars(sc, x, 0.39, words, self.dists[run][n][self.rows], 220.0,
                          self.INKS[run], op, ref=self.ref[self.rows], step=0.066)
            self._status(sc, x, 0.875, run, n, op)
        self.text(sc, 0.05, 0.925, "update %d" % n, op, anchor="lm", size=0.036, strong=True)

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.10, 0.60), (0.38, 0.80)), np.r_[0, self.kl.ravel()],
                           np.r_[self.start, self.score.ravel()])
        self.axes(sc, chart, op, "drift from the reference (KL, nats)", "reward model score")
        n = int(round(self.updates * min(u / 0.75, 1.0) ** 1.7))
        for run, ink in enumerate(self.INKS):
            xs, ys = np.r_[0, self.kl[run]][:n + 1], np.r_[self.start, self.score[run]][:n + 1]
            pts = [self.P(chart.fx(x), chart.fy(y)) for x, y in zip(xs, ys)]
            self.polyline(sc, pts, 3.2, ink, op, bias=0.0)
            self.spot(sc, chart.fx(xs[-1]), chart.fy(ys[-1]), 6.0, ink, op, glow=0.45)
        self.text(sc, 0.68, 0.36, "update %d" % n, op, anchor="lm", size=0.036, strong=True)
        m = max(n - 1, 0)
        for run, ink in enumerate(self.INKS):
            y = 0.45 + run * 0.125
            name = "no leash" if run == 0 else "β = %s" % number(self.BETAS[run])
            self.text(sc, 0.68, y, name, op, anchor="lm", size=0.036, ink=dark(ink))
            self.text(sc, 0.68, y + 0.05, "score %.2f  drift %.2f" % (
                self.score[run][m] if n else self.start, self.kl[run][m] if n else 0.0), op,
                anchor="lm", size=0.032)
        lo, hi = self.seeds_kl.min(0), self.seeds_kl.max(0)
        self.lines(sc, 0.68, 0.825, [
            "a tighter leash buys less", "reward and less drift;",
            "drift at β = %s over %d reward" % (number(self.BETAS[2]), len(self.seeds_kl)),
            "models: %.2f to %.2f nats" % (lo[2], hi[2])],
            op * smoothstep(0.8, 0.9, u), size=0.03, step=0.04)


# ---------------------------------------------------------------------- 38
class Dpo(Kit):
    """Direct preference optimisation: the same pairs with no reward model in
    between, the implicit reward log(policy / reference) and the loss on one
    pair, the policy moving as it trains, and where it ends against PPO's."""

    TITLE = "Direct preference optimisation"
    PHASES = (("two routes", 10.0), ("the loss on one pair", 14.0), ("training", 12.0),
              ("against PPO", 12.0))
    WIN, LOSE, MODEL, DPO, PPO = GREEN, RED, AMBER, TEAL, VIOLET

    def setup(self):
        s, d = self.study, preferences()
        self.words = list(s.metadata["settings"]["candidate_words"])
        self.beta = float(s.metadata["settings"]["beta"])
        self.loss = np.asarray(s.panels[0].data["ys"], float)[0]
        self.held = np.asarray(s.panels[2].data["ys"], float)           # DPO, PPO
        self.kl, self.ppo_kl = np.asarray(s.panels[3].data["ys"], float)
        self.between = np.asarray(s.panels[3].data["between"], float)   # per seed
        self.ends = np.asarray(s.panels[2].data["each"], float)[:, :, -1]   # seed, DPO | PPO
        self.winners, self.losers = np.asarray(d["winners"]), np.asarray(d["losers"])[0]
        self.ref = np.asarray(d["ref"], float)
        self.p = np.asarray(d["dpo_p"], float)
        self.ppo_p = np.asarray(d["ppo_p"], float)[list(F.BETAS).index(self.beta)][-1]
        self.updates = len(self.loss)
        king = self.words.index("king")
        self.pair = int(np.argmax(self.winners == king))
        top = list(np.argsort(-self.ref)[:4])
        top += [i for i in np.argsort(-self.p[-1])[:5] if i not in top]
        self.rows = top[:7]

    def subtitle_for(self, k):
        return [
            "RLHF trains a reward model on preference pairs, then a policy against it; DPO "
            "trains the policy on the %s pairs about 'the' directly" % f"{len(self.winners):,}",
            "DPO reads a word's reward off the policy itself: the log of its chance now over "
            "its chance in the reference; the loss wants the preferred word's to be higher",
            "%d updates of that one loss: it falls from %.3f to %.3f as the policy moves away "
            "from the reference, towards words that were preferred" % (
                self.updates, self.loss[0], self.loss[-1]),
            "scored on held-out text, DPO ends at %.2f and PPO at %.2f (higher is better); the "
            "two final policies are %.2f nats apart, and %.2f to %.2f over %d seeds" % (
                self.held[0][-1], self.held[1][-1], self.between[0], self.between.min(),
                self.between.max(), len(self.between)),
        ][k]

    def _chain(self, sc, y, boxes, u, op, ink, hops):
        """Boxes in a row joined by arrows, a value passing along them."""
        xs = [b[0] for b in boxes]
        for x, name, fill, border, w in boxes:
            self.box(sc, x, y, w, 72, fill, op, name, border=border, size=0.038)
        for j in range(len(boxes) - 1):
            a = (xs[j] + self.frac(boxes[j][4] / 2 + 6)[0], y)
            b = (xs[j + 1] - self.frac(boxes[j + 1][4] / 2 + 8)[0], y)
            self.arrow(sc, a, b, ink, op)
            self.pulse(sc, a, b, u * hops - j, ink, op, r=6.0)

    def phase_1(self, sc, u, op):
        t = (u * 2.5) % 1.0
        self.text(sc, 0.05, 0.34, "RLHF: three stages", op, anchor="lm", size=0.040,
                  ink=dark(self.PPO))
        self._chain(sc, 0.46, [(0.135, "pairs", PAPER, AXIS, 150),
                               (0.39, "reward model", soft(self.MODEL, 0.35), self.MODEL, 190),
                               (0.63, "PPO", soft(self.PPO, 0.35), self.PPO, 120),
                               (0.85, "policy", soft(self.PPO), self.PPO, 150)], t, op,
                    self.PPO, 3)
        self.text(sc, 0.05, 0.635, "DPO: one loss, no reward model", op, anchor="lm", size=0.040,
                  ink=dark(self.DPO))
        self._chain(sc, 0.755, [(0.135, "pairs", PAPER, AXIS, 150),
                                (0.85, "policy", soft(self.DPO), self.DPO, 150)], t, op,
                    self.DPO, 1)
        self.text(sc, 0.49, 0.71, "the DPO loss", op, size=0.036, ink=dark(self.DPO))
        self.text(sc, 0.5, 0.905, "the pairs: the word the play has after 'the', preferred to "
                  "a random one; both policies start at the same reference",
                  op * smoothstep(0.5, 0.65, u), size=0.03)

    def phase_2(self, sc, u, op):
        n = int(round(self.updates * min(u / 0.85, 1.0) ** 1.6))
        w, l = int(self.winners[self.pair]), int(self.losers[self.pair])
        p, full = self.p[n], 250.0
        implicit = np.log(p / self.ref)
        top = max(float(self.p[:, w].max()), float(self.ref[l]), float(self.p[:, l].max()))
        self.text(sc, 0.24, 0.33, "its chance of being picked", op, anchor="lm", size=0.032,
                  strong=True)
        self.text(sc, 0.95, 0.33, "implicit reward", op, anchor="rm", size=0.032, strong=True)
        for row, (i, ink, name) in enumerate(((w, self.WIN, "preferred"),
                                              (l, self.LOSE, "rejected"))):
            y = 0.44 + row * 0.20
            self.box(sc, 0.115, y, 110, 46, soft(ink), op, self.words[i], ink=dark(ink, 0.35),
                     border=ink, size=0.038)
            self.text(sc, 0.115, y + 0.07, name, op, size=0.032, ink=dark(ink))
            for b, (label, v, colour) in enumerate((("policy", p[i], self.DPO),
                                                    ("reference", self.ref[i], DIM))):
                yy = y - 0.03 + b * 0.062
                self.hbar(sc, 0.335, yy, full * v / top, colour, op, label, pct(v), thick=20.0,
                          track=full)
            self.text(sc, 0.95, y - 0.035, "ln(%s ÷ %s)" % (pct(p[i]), pct(self.ref[i])), op,
                      anchor="rm", size=0.032)
            self.text(sc, 0.95, y + 0.03, "= " + signed(implicit[i]), op, anchor="rm", size=0.042,
                      ink=dark(ink))
        gap = float(implicit[w] - implicit[l])
        pair_loss = float(np.logaddexp(0, -self.beta * gap))
        self.text(sc, 0.05, 0.80, "gap between the two: %s;  the wider, the smaller the loss" % (
            signed(gap)), op, anchor="lm", size=0.034)
        self.text(sc, 0.95, 0.80, "loss on this pair  %.3f" % pair_loss, op, anchor="rm",
                  size=0.036, strong=True)
        self.text(sc, 0.05, 0.875, "update %d" % n, op, anchor="lm", size=0.036, strong=True)
        self.text(sc, 0.95, 0.875, "over all %s pairs  %.3f" % (
            f"{len(self.winners):,}", self.loss[min(n, self.updates - 1)]), op,
            anchor="rm", size=0.036, ink=dark(self.DPO))

    def phase_3(self, sc, u, op):
        n = int(round(self.updates * min(u / 0.85, 1.0) ** 1.6))
        words = [self.words[i] for i in self.rows]
        self.text(sc, 0.05, 0.325, "the policy after 'the'  (%s mark: the reference)"
                  % theme.word("dark", "white"), op,
                  anchor="lm", size=0.032, strong=True)
        self.wordbars(sc, 0.15, 0.39, words, self.p[n][self.rows], 230.0, self.DPO, op,
                      ref=self.ref[self.rows], step=0.066)
        self.text(sc, 0.05, 0.875, "update %d" % n, op, anchor="lm", size=0.036, strong=True)
        steps = np.arange(self.updates)
        chart = self.chart(((0.60, 0.94), (0.40, 0.76)), steps, self.loss)
        self.axes(sc, chart, op, "DPO update", "DPO loss over the pairs")
        self.plot(sc, chart, steps, self.loss, self.DPO, op, max(n - 1, 0) / (self.updates - 1),
                  width=2.3)
        self.text(sc, 0.60, 0.875, "loss %.3f" % self.loss[min(n, self.updates - 1)], op,
                  anchor="lm", size=0.036, ink=dark(self.DPO))

    def phase_4(self, sc, u, op):
        # the two final policies, word by word
        words = [self.words[i] for i in self.rows]
        grow = smoothstep(0.0, 0.35, u)
        top = float(max(self.p[-1].max(), self.ppo_p.max()))
        self.text(sc, 0.05, 0.325, "where each policy ends, after 'the'", op, anchor="lm",
                  size=0.032, strong=True)
        for r, i in enumerate(self.rows):
            y = 0.40 + r * 0.066
            self.text(sc, 0.14, y, words[r], op, anchor="rm", size=0.034)
            for b, (v, ink) in enumerate(((self.p[-1][i], self.DPO), (self.ppo_p[i], self.PPO))):
                at = self.P(0.15, y + (b - 0.5) * 0.027)
                length = max(250.0 * v / top * grow, 0.5)
                self.rect(sc, self.shift(at, length / 2, 0), length, 11, ink, op, radius=2)
            self.text(sc, 0.15 + self.frac(250.0 * max(self.p[-1][i], self.ppo_p[i]) / top
                                           * grow + 10)[0], y,
                      "%s  %s" % (pct(self.p[-1][i]), pct(self.ppo_p[i])), op, anchor="lm",
                      size=0.03)
        o = op * smoothstep(0.35, 0.5, u)
        for row, (name, ink, kl) in enumerate((
                ("DPO", self.DPO, self.kl[-1]),
                ("PPO, β = %s" % number(self.beta), self.PPO, self.ppo_kl[-1]))):
            y = 0.40 + row * 0.135
            self.text(sc, 0.62, y, "%s   %.2f" % (name, self.held[row][-1]), o, anchor="lm",
                      size=0.036, ink=dark(ink))
            self.text(sc, 0.62, y + 0.05, "held-out score;  drift %.2f nats" % kl, o,
                      anchor="lm", size=0.03)
        lo, hi = self.ends.min(0), self.ends.max(0)
        self.lines(sc, 0.62, 0.69, [
            "between the two policies:",
            "%.2f nats" % self.between[0],
            "over %d seeds of the pairs:" % len(self.between),
            "DPO %.2f to %.2f" % (lo[0], hi[0]),
            "PPO %.2f to %.2f" % (lo[1], hi[1])],
            op * smoothstep(0.6, 0.75, u), size=0.032, step=0.045)


# ---------------------------------------------------------------------- 39
class RewardHacking(Kit):
    """Optimising against an imperfect reward model: every word's proxy score
    beside what held-out text does, the policy pushed along the proxy while
    both are measured, the held-out score's peak and fall, where the
    probability went, and the same push on thirty reward models."""

    TITLE = "Reward hacking"
    PHASES = (("two scores", 9.0), ("push", 13.0), ("the peak", 9.0), ("where it went", 9.0),
              ("thirty models", 12.0))
    PROXY, HELD, POLICY = AMBER, TEAL, VIOLET
    SIZE_INKS = (RED, VIOLET, BLUE)

    def setup(self):
        s = self.study
        st = s.metadata["settings"]
        self.words = list(st["candidate_words"])
        self.proxy, self.true = np.asarray(s.panels[0].data["ys"], float)
        self.values = np.asarray(s.panels[1].data["ys"], float)       # proxy, held-out
        self.steps = np.arange(self.values.shape[1])
        self.dist = np.asarray(s.panels[2].data["values"], float)     # every 6th update
        self.every = (len(self.steps) - 1) // (len(self.dist) - 1)
        self.start, self.best, self.end = (float(v) for v in s.panels[3].data["marks"])
        self.peak = int(np.argmax(self.values[1]))
        last = s.panels[3].data
        self.each = np.asarray(last["each"], float)                   # size, seed, update
        self.mean = np.asarray(last["ys"], float)
        self.worse = [int(v) for v in last["ends_worse"]]
        self.accuracy = np.asarray(last["accuracy"], float)[:, :, 0].mean(1)
        self.rose = bool((np.asarray(last["proxy"], float)[:, :, -1]
                          > np.asarray(last["proxy"], float)[:, :, 0]).all())
        self.sizes = [int(v) for v in st["sizes"]]
        self.pairs = int(st["training_pairs"])
        self.drawn = int(st["drawn_size"]) or self.pairs
        share = np.exp(self.true)
        top = list(np.argsort(-self.proxy)[:3])
        top += [i for i in np.argsort(-share)[:3] if i not in top] + [0]
        self.pairs_shown = list(dict.fromkeys(top))[:6]
        rows = list(np.argsort(-self.dist.max(0))[:7])
        self.rows = sorted(rows, key=lambda i: -self.dist[0][i])
        # the words the chance sits on at the start, at the held-out best, at the end
        at = [0, self.peak // self.every, len(self.dist) - 1]
        self.movers = list(dict.fromkeys(int(np.argmax(self.dist[i])) for i in at))
        for i in np.argsort(-self.dist.max(0)):
            if len(self.movers) < 3 and int(i) not in self.movers:
                self.movers.append(int(i))

    def size_name(self, k):
        return f"{self.sizes[k] or self.pairs:,} pairs"

    def subtitle_for(self, k):
        v = self.values
        fav = self.words[int(np.argmax(self.proxy))]
        return [
            "every word has two scores: the proxy, from a reward model trained on %s pairs, "
            "and how often the word really follows 'the' in held-out text; they do not "
            "agree" % f"{self.drawn:,}",
            "%d updates each move the policy towards words the proxy scores highly; its proxy "
            "score only rises, from %.2f to %.2f, while the held-out score is measured too" % (
                len(self.steps) - 1, v[0][0], v[0][-1]),
            "the held-out score rises from %.3f to its best, %.3f, at update %d, then falls to "
            "%.3f while the proxy score keeps climbing" % (
                self.start, self.best, self.peak, self.end),
            "pushed past the peak, the chance piles onto '%s', the proxy's favourite, which "
            "held-out text has after 'the' only %s of the time" % (
                fav, pct(np.exp(self.true[int(np.argmax(self.proxy))]))),
            "this was the worst of thirty runs: held-out score ended below its start for %d "
            "of 10 reward models trained on %s, %d of 10 on %s and %d of 10 on all %s" % (
                self.worse[0], self.size_name(0), self.worse[1], self.size_name(1),
                self.worse[2], self.size_name(2)),
        ][k]

    def phase_1(self, sc, u, op):
        share = np.exp(self.true)
        self.text(sc, 0.38, 0.33, "proxy score", op, size=0.034, ink=dark(self.PROXY))
        self.text(sc, 0.60, 0.33, "how often it follows 'the' in held-out text", op, anchor="lm",
                  size=0.034, ink=dark(self.HELD))
        top, full = float(np.abs(self.proxy).max()), self.span(0.25)[0]
        for r, i in enumerate(self.pairs_shown):
            o = op * smoothstep(r * 0.11, r * 0.11 + 0.08, u)
            grow = smoothstep(r * 0.11 + 0.04, r * 0.11 + 0.18, u)
            y = 0.415 + r * 0.082
            self.text(sc, 0.20, y, self.words[i], o, anchor="rm", size=0.040, strong=True)
            v = float(self.proxy[i])
            at = self.P(0.38, y)
            self.rect(sc, at, 240, 24, (0.78, 0.79, 0.80), 0.45 * o, radius=4, lift=0.3)
            length = 118.0 * abs(v) / top * grow
            self.rect(sc, self.shift(at, np.sign(v) * length / 2, 0), max(length, 0.5), 24,
                      self.PROXY, o, radius=3)
            self.rect(sc, at, 2.5, 32, INK, o, radius=1, lift=0.8)
            self.label(sc, self.shift(at, 128, 0), signed(v), 0.034, anchor="lm", opacity=o,
                       ink=INK)
            self.hbar(sc, 0.60, y, full * share[i] / float(share.max()), self.HELD, o, "",
                      pct(share[i]), grow, thick=24.0, track=full)
        a, b = self.words[int(np.argmax(self.proxy))], self.words[int(np.argmax(share))]
        self.text(sc, 0.5, 0.915, "the proxy's favourite is '%s'; held-out text's is '%s'" % (
            a, b), op * smoothstep(0.78, 0.9, u), size=0.036, strong=True)

    def _now(self, u, span=0.88):
        return int(round((len(self.steps) - 1) * min(u / span, 1.0)))

    def phase_2(self, sc, u, op):
        n = self._now(u)
        row = self.dist[n // self.every]
        words = [self.words[i] for i in self.rows]
        self.text(sc, 0.05, 0.325, "the policy after 'the'", op, anchor="lm", size=0.032,
                  strong=True)
        self.wordbars(sc, 0.14, 0.39, words, row[self.rows], 200.0, self.POLICY, op,
                      vmax=float(self.dist.max()), step=0.066,
                      fmt=lambda v: pct(v) if v >= 5e-5 else "0%")
        self.text(sc, 0.05, 0.875, "update %d" % (n // self.every * self.every), op, anchor="lm",
                  size=0.036, strong=True)
        for which, (name, ink, box) in enumerate((
                ("proxy score", self.PROXY, ((0.52, 0.94), (0.36, 0.53))),
                ("held-out score", self.HELD, ((0.52, 0.94), (0.65, 0.82))))):
            chart = self.chart(box, self.steps, self.values[which], pad=0.1)
            ticks = nice_ticks(*chart.y_range, 2)
            self.axes(sc, chart, op, "update" if which else "", "%s  %s" % (
                name, signed(self.values[which][n])), yticks=[(v, number(v)) for v in ticks])
            self.plot(sc, chart, self.steps, self.values[which], ink, op,
                      n / (len(self.steps) - 1), width=2.3)

    def phase_3(self, sc, u, op):
        chart = self.chart(((0.10, 0.60), (0.38, 0.80)), self.steps, self.values[1], pad=0.12)
        self.axes(sc, chart, op, "update (each pushes the policy further along the proxy)",
                  "held-out score (higher is better)")
        go = u / 0.6
        self.plot(sc, chart, self.steps, self.values[1], self.HELD, op, go, width=2.5)
        n = int(min(go, 1.0) * (len(self.steps) - 1))
        marks = ((0, "start", self.start), (self.peak, "best", self.best),
                 (len(self.steps) - 1, "end", self.end))
        for row, (at, name, value) in enumerate(marks):
            if n < at:
                continue
            x, y = chart.fx(at), chart.fy(value)
            self.spot(sc, x, y, 7.0, INK, op)
            yy = 0.42 + row * 0.15
            self.text(sc, 0.68, yy, "%s: update %d" % (name, at), op, anchor="lm", size=0.034,
                      strong=True)
            self.text(sc, 0.68, yy + 0.05, "held-out %s" % signed(value, 3), op, anchor="lm",
                      size=0.034, ink=dark(self.HELD))
            self.text(sc, 0.68, yy + 0.095, "proxy %s" % signed(self.values[0][at], 3), op,
                      anchor="lm", size=0.034, ink=dark(self.PROXY))
        if n >= self.peak:
            chart.mark(sc, self.P, self.peak, INK, op * 0.6, dashes=14, width=1.3)

    def phase_4(self, sc, u, op):
        inks = (OCHRE, BLUE, MAGENTA)
        at = np.arange(len(self.dist)) * self.every
        ids = self.movers
        chart = self.chart(((0.10, 0.60), (0.38, 0.80)), at, np.r_[self.dist[:, ids].ravel(), 0],
                           pad=0.08)
        self.axes(sc, chart, op, "update", "the policy's chance of picking the word",
                  yticks=[(v, "%d%%" % round(100 * v)) for v in nice_ticks(
                      0.0, float(self.dist[:, ids].max()), 3)])
        go = u / 0.6
        chart.mark(sc, self.P, self.peak, INK, op * 0.6, dashes=14, width=1.3)
        self.text(sc, chart.fx(self.peak) + 0.01, 0.40, "held-out score's best", op, anchor="lm",
                  size=0.03)
        for i, ink in zip(ids, inks):
            self.plot(sc, chart, at, self.dist[:, i], ink, op, go, width=2.5)
        share = np.exp(self.true)
        for row, (i, ink) in enumerate(zip(ids, inks)):
            y = 0.40 + row * 0.165
            self.box(sc, 0.725, y, 100, 40, soft(ink), op, self.words[i], ink=dark(ink, 0.4),
                     border=ink, size=0.036)
            self.text(sc, 0.79, y - 0.022, "proxy %s" % signed(self.proxy[i]), op, anchor="lm",
                      size=0.032, ink=dark(self.PROXY))
            self.text(sc, 0.79, y + 0.026, "held-out %s" % pct(share[i]), op, anchor="lm",
                      size=0.032, ink=dark(self.HELD))
        self.text(sc, 0.5, 0.925, "the proxy scores '%s' highest; held-out text says "
                  "otherwise, and the proxy cannot tell" % self.words[int(np.argmax(self.proxy))],
                  op * smoothstep(0.7, 0.85, u), size=0.034, strong=True)

    def phase_5(self, sc, u, op):
        chart = self.chart(((0.10, 0.56), (0.38, 0.80)), self.steps, self.each, pad=0.06)
        self.axes(sc, chart, op, "update along each model's own proxy",
                  "held-out score (higher is better)")
        chart.hline(sc, self.P, self.start, op, dashes=22, width=1.3)
        self.text(sc, 0.555, chart.fy(self.start) - 0.028, "start", op, anchor="rm", size=0.03)
        for k, ink in enumerate(self.SIZE_INKS):
            a = k * 0.2
            go = (u - a) / 0.3
            if go <= 0:
                continue
            for run in self.each[k]:
                self.plot(sc, chart, self.steps, run, ink, op * 0.55, go, width=1.4, dot=False)
            o = op * smoothstep(a + 0.05, a + 0.15, u)
            y = 0.385 + k * 0.165
            self.text(sc, 0.62, y, "trained on %s" % self.size_name(k), o, anchor="lm",
                      size=0.036, ink=dark(ink))
            self.text(sc, 0.62, y + 0.047, "right on %.0f%% of held-out pairs" % (
                100 * self.accuracy[k]), o, anchor="lm", size=0.03)
            self.text(sc, 0.62, y + 0.089, "ends below its start: %d of %d" % (
                self.worse[k], self.each.shape[1]), o, anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.10, 0.925, "every line is one reward model; in all thirty the proxy "
                  "score rose" if self.rose else "every line is one reward model",
                  op * smoothstep(0.8, 0.9, u), anchor="lm", size=0.032)


# ---------------------------------------------------------------------- 40
class PromptAndPrefixTuning(Kit):
    """Four ways to adapt a frozen model: where a soft prompt, a key-value
    prefix, LoRA and full tuning put their trainable numbers, those numbers
    drawn to scale, and how far each gets on held-out text."""

    TITLE = "Prompt and prefix tuning"
    PHASES = (("a frozen model", 9.0), ("the new numbers", 18.0), ("to scale", 9.0),
              ("adapting", 11.0), ("held out", 10.0))
    INKS = (RED, BLUE, GREEN, AMBER)
    NAMES = ("soft prompt", "key-value prefix", "LoRA", "full tuning")
    BLOCKS = ((0.19, 150, "embedding", ("E", "pos")),
              (0.435, 200, "attention", ("l0.Q", "l0.K", "l0.V", "l0.O", "l0.n1.g", "l0.n1.b")),
              (0.655, 120, "MLP", ("l0.W1", "l0.W2", "l0.n2.g", "l0.n2.b")),
              (0.85, 150, "output", ("U", "final.g", "final.b")))
    MID, TALL = 0.625, 150.0

    def setup(self):
        s, r = self.study, F.peft_runs()
        self.sizes = [int(v) for v in s.panels[0].data["values"]]
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)         # mean of the seeds
        self.final = [float(v) for v in s.panels[2].data["values"]]
        self.low = [float(v) for v in s.panels[2].data["low"]]
        self.high = [float(v) for v in s.panels[2].data["high"]]
        self.closed = [float(v) for v in s.panels[2].data["closed"]]
        self.share = [float(v) for v in s.panels[3].data["values"]]
        self.base = float(s.panels[2].data["base"])     # the untouched base on the same text
        self.seeds = len(s.panels[2].data["each"])
        self.rank = int(s.metadata["settings"]["rank"])
        self.config = F.config()
        self.block_sizes = [sum(int(F.base_model(F.SEEDS[0])[k].size) for k in keys)
                            for _x, _w, _name, keys in self.BLOCKS]
        self.chars = [shown(c) for c in realdata.decode(r["sample"][0][:5])]

    def subtitle_for(self, k):
        z, f = self.sizes, self.final
        return [
            "a small transformer trained on Tiny Shakespeare: %s numbers, all frozen; the job "
            "is to adapt it to the same play IN CAPITALS, which it reads badly" %
            f"{z[3]:,}",
            "a soft prompt learns %d numbers in front of the text, a key-value prefix %d inside "
            "attention, LoRA %d beside the output matrix; full tuning moves all %s" % (
                z[0], z[1], z[2], f"{z[3]:,}"),
            "drawn to scale, one small square per number: the three add-ons train %.1f%%, %.1f%% "
            "and %.1f%% of what full tuning trains" % tuple(100 * v for v in self.share[:3]),
            "%d updates on fresh text in capitals, measured on held-out text, mean of %d "
            "seeds: from %.2f the add-ons reach about %.1f, full tuning %.2f" % (
                self.steps[-1], self.seeds, self.base, np.mean(self.traces[:3, -1]),
                self.traces[3][-1]),
            "held-out loss: soft prompt %.2f, prefix %.2f, LoRA %.2f, full tuning %.2f, against "
            "%.2f untouched; with under 1%% of the numbers the add-ons go %.0f%% to %.0f%% of "
            "the way" % (f[0], f[1], f[2], f[3], self.base, 100 * min(self.closed[:3]),
                         100 * max(self.closed[:3])),
        ][k]

    def _model(self, sc, op, lit=(0, 0, 0, 0), counts=1.0):
        """The frozen model as four blocks; ``lit``: how strongly each
        method's numbers are shown."""
        y, h = self.MID, self.TALL
        full = lit[3]
        for (x, w, name, _keys), size in zip(self.BLOCKS, self.block_sizes):
            fill = tuple(np.asarray(soft(DIM, 0.3)) * (1 - full) + np.asarray(soft(AMBER, 0.2))
                         * full)
            self.box(sc, x, y, w, h, fill, op, border=AMBER if full > 0.5 else AXIS)
            self.text(sc, x, y - 0.118, name, op, size=0.034, strong=True)
            self.text(sc, x, y + 0.20, f"{size:,} numbers", op * counts, size=0.032,
                      ink=dark(AMBER) if full > 0.5 else None)
        for a, b in zip(self.BLOCKS, self.BLOCKS[1:]):
            self.arrow(sc, (a[0] + self.frac(a[1] / 2 + 4)[0], y),
                       (b[0] - self.frac(b[1] / 2 + 6)[0], y), AXIS, op)
        # the text going in, the soft prompt in front of it
        xs = [0.19 + 0.047 * i for i in range(5)]
        for x, ch in zip(xs, self.chars):
            self.box(sc, x, 0.36, 36, 40, PAPER, op, ch, border=AXIS, size=0.034)
        self.arrow(sc, (0.19, 0.405), (0.19, 0.462), AXIS, op)
        for j in range(2):
            self.box(sc, xs[0] - 0.047 * (2 - j), 0.36, 36, 40, soft(RED, 0.25), op * lit[0],
                     border=RED, glow=0.2)
        # what is inside: a table of vectors, keys and values, two layers, a matrix
        x = self.BLOCKS[0][0]
        for r in range(5):
            self.plate(sc, x, y - 0.045 + r * 0.03, 96, 9, DIM, op, radius=2)
        self.text(sc, x, y + 0.118, "a table of vectors", op, size=0.03)
        for col, name in enumerate(("keys", "values")):
            x = 0.39 + col * 0.09
            self.text(sc, x, y + 0.118, name, op, size=0.03)
            for r in range(5):
                self.plate(sc, x, y - 0.012 + r * 0.022, 56, 7, DIM, op, radius=2)
            for r in range(2):
                self.plate(sc, x, y - 0.062 + r * 0.022, 56, 7, BLUE, op * lit[1], radius=2,
                           glow=0.2)
        x = self.BLOCKS[2][0]
        for r in range(2):
            self.plate(sc, x, y - 0.02 + r * 0.06, 70, 16, DIM, op, radius=2)
        self.text(sc, x, y + 0.118, "two layers", op, size=0.03)
        # the output matrix, and LoRA's two thin factors beside it
        x = self.BLOCKS[3][0]
        self.plate(sc, x + 0.012, y - 0.015, 88, 34, DIM, op, radius=2)
        self.plate(sc, x - 0.055, y - 0.015, 7, 34, GREEN, op * lit[2], radius=1, glow=0.2)
        self.plate(sc, x + 0.012, y + 0.048, 88, 7, GREEN, op * lit[2], radius=1, glow=0.2)
        self.text(sc, x, y + 0.118, "one matrix", op, size=0.03)

    def phase_1(self, sc, u, op):
        self._model(sc, op, counts=smoothstep(0.25, 0.4, u))
        self.text(sc, 0.43, 0.36, "the new text: the play in capitals", op, anchor="lm",
                  size=0.032)
        self.text(sc, 0.5, 0.915, "%s numbers in all: frozen, none of them will change" %
                  f"{sum(self.block_sizes):,}", op * smoothstep(0.5, 0.65, u), size=0.036,
                  strong=True)

    def phase_2(self, sc, u, op):
        k, t = self.stage(u, 4)
        lit = [1.0 if j == k else (0.45 if j < k and k < 3 else 0.0) for j in range(4)]
        lit[k] = smoothstep(0.0, 0.15, t)
        self._model(sc, op, lit, counts=1.0 if k == 3 else 0.0)
        w, r, v = self.config.width, self.rank, self.config.vocab
        heads = self.config.heads
        notes = (
            ("soft prompt", "%d learned vectors put in front of the text" % r,
             "%d × %d = %d numbers" % (r, w, self.sizes[0])),
            ("key-value prefix", "%d extra keys and %d extra values for each of the %d "
             "attention heads" % (r, r, heads), "%d × %d × %d × %d = %d numbers" % (
                 2, r, heads, w // heads, self.sizes[1])),
            ("LoRA", "a thin %d × %d matrix times a thin %d × %d one, added to the "
             "output matrix" % (w, r, r, v), "%d + %d = %d numbers" % (
                 w * r, r * v, self.sizes[2])),
            ("full tuning", "nothing added: every number in the model is trained",
             "%s numbers" % f"{self.sizes[3]:,}"))
        name, what, count = notes[k]
        o = op * smoothstep(0.05, 0.2, t) * (1 - smoothstep(0.93, 1.0, t) * (k < 3))
        ink = dark(self.INKS[k])
        self.text(sc, 0.05, 0.875, name, o, anchor="lm", size=0.040, ink=ink)
        self.text(sc, 0.05, 0.925, what, o, anchor="lm", size=0.032)
        self.text(sc, 0.95, 0.875, count, o, anchor="rm", size=0.040, ink=ink)

    def phase_3(self, sc, u, op):
        side = 236.0
        cx, cy = 0.74, 0.60
        grow = smoothstep(0.45, 0.7, u)
        self.box(sc, cx, cy, side * grow, side * grow, AMBER, op * smoothstep(0.42, 0.5, u),
                 radius=3)
        self.text(sc, cx, cy + 0.285, "full tuning: %s numbers" % f"{self.sizes[3]:,}",
                  op * smoothstep(0.6, 0.7, u), size=0.036, ink=dark(AMBER))
        dw, dh = self.frac(side, side)
        offset = 8.0
        for r in range(3):
            s = side * np.sqrt(self.sizes[r] / self.sizes[3])
            o = op * smoothstep(r * 0.13, r * 0.13 + 0.1, u)
            y = 0.40 + r * 0.14
            self.box(sc, 0.09, y, s, s, self.INKS[r], o, radius=1.5)
            self.text(sc, 0.135, y - 0.022, self.NAMES[r], o, anchor="lm", size=0.038,
                      ink=dark(self.INKS[r]))
            self.text(sc, 0.135, y + 0.028, "%d numbers: %.1f%% of the model" % (
                self.sizes[r], 100 * self.share[r]), o, anchor="lm", size=0.032)
            # the same square again, inside the full one
            inside = self.frac(offset + s / 2, 8.0 + s / 2)
            self.plate(sc, cx - dw / 2 + inside[0], cy - dh / 2 + inside[1], s, s, self.INKS[r],
                       op * smoothstep(0.75, 0.85, u), border=WHITE, radius=1.5)
            offset += s + 6.0
        self.text(sc, 0.05, 0.84, "the same three squares, inside the big one",
                  op * smoothstep(0.78, 0.9, u), anchor="lm", size=0.032)

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.10, 0.60), (0.38, 0.80)), self.steps, np.r_[
            self.traces.ravel(), self.base])
        self.axes(sc, chart, op, "update on the text in capitals",
                  "loss on held-out capitals (nats)")
        go = u / 0.8
        for trace, ink in zip(self.traces, self.INKS):
            self.plot(sc, chart, self.steps, trace, ink, op, go, width=2.3)
        now = int(min(go, 1.0) * (len(self.steps) - 1))
        self.text(sc, 0.68, 0.36, "update %d" % self.steps[now], op, anchor="lm", size=0.036,
                  strong=True)
        for r, (name, ink) in enumerate(zip(self.NAMES, self.INKS)):
            y = 0.45 + r * 0.105
            self.text(sc, 0.68, y, name, op, anchor="lm", size=0.034, ink=dark(ink))
            self.text(sc, 0.95, y, "%.3f" % self.traces[r][now], op, anchor="rm", size=0.036,
                      strong=True)
            self.text(sc, 0.68, y + 0.042, "%s numbers trained" % f"{self.sizes[r]:,}",
                      op, anchor="lm", size=0.03)
        self.text(sc, 0.68, 0.875, "mean of %d seeds" % self.seeds, op, anchor="lm", size=0.03)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.05, 0.335, "held-out loss, each method stopped where its validation "
                  "loss was lowest", op, anchor="lm", size=0.034, strong=True)
        full = self.span(0.40)[0]
        rows = [("untouched base", self.base, DIM, None, None)] + [
            (name, self.final[r], self.INKS[r], self.closed[r], (self.low[r], self.high[r]))
            for r, name in enumerate(self.NAMES)]
        for r, (name, v, ink, closed, spread) in enumerate(rows):
            y = 0.425 + r * 0.088
            grow = (u - 0.05 - r * 0.08) / 0.25
            o = op * smoothstep(r * 0.08, r * 0.08 + 0.08, u)
            self.text(sc, 0.28, y, name, o, anchor="rm", size=0.036,
                      ink=dark(ink) if r else None)
            self.hbar(sc, 0.30, y, full * v / self.base, ink, o, "", "%.2f" % v, grow, thick=28.0,
                      track=full)
            if closed is not None:
                at = [self.shift(self.P(0.30, y), full * s / self.base, 0) for s in spread]
                for p in at:
                    self.rect(sc, p, 2.0, 36, INK, o * smoothstep(0.9, 1.0, grow), radius=1,
                              lift=0.8)
                self.text(sc, 0.95, y, "%.0f%% of the way" % (100 * closed),
                          o * smoothstep(0.5, 0.65, u), anchor="rm", size=0.034, strong=True)
        self.text(sc, 0.05, 0.895, "'of the way': from the untouched base to full tuning; "
                  "%s marks: lowest and highest of %d seeds"
                  % (theme.word("dark", "white"), self.seeds),
                  op * smoothstep(0.6, 0.75, u), anchor="lm", size=0.03)


# ---------------------------------------------------------------------- 41
class ModelMerging(Kit):
    """Task vectors: two fine-tunes of one base, each one's change of weights
    as a vector, those vectors added, averaged and subtracted, and the
    held-out loss of every such mix."""

    TITLE = "Model merging with task vectors"
    PHASES = (("two fine-tunes", 10.0), ("task vectors", 11.0), ("weight arithmetic", 14.0),
              ("measured", 14.0))
    A, B = RED, BLUE
    MIXES = (("base", 0.0, 0.0, INK), ("tuned on A", 1.0, 0.0, RED),
             ("tuned on B", 0.0, 1.0, BLUE), ("average", 0.5, 0.5, VIOLET),
             ("sum", 1.0, 1.0, AMBER), ("A minus B", 1.0, -1.0, TEAL),
             ("minus B", 0.0, -1.0, OCHRE))
    MAP = (0.33, 0.59, 440.0, 250.0)            # centre x, y; width, height in px
    TASKS = (("A", "the play IN CAPITALS"), ("B", "the play backwards"))

    def setup(self):
        s, r = self.study, F.merging_runs()
        self.names = list(s.panels[2].data["labels"])
        self.norms = [float(v) for v in r["norms"][0]]          # the drawn seed's
        self.cosine = float(r["cosine"][0])
        self.scores = np.asarray([s.panels[2].data["values"], s.panels[3].data["values"]], float)
        self.each = np.asarray([s.panels[2].data["each"], s.panels[3].data["each"]], float)
        self.seeds = self.each.shape[1]
        self.stops = [int(v) for v in r["stop"][0]]
        self.plane = r["plane"][:2]                             # task, b, a
        self.sample = [[realdata.decode(row).replace("\n", "↵") for row in block]
                       for block in r["sample"]]
        self.size = int(r["size"])
        mean = self.plane.mean(0)
        self.rise = (mean - mean[list(F.MERGE_B).index(0.0), list(F.MERGE_A).index(0.0)]).T[::-1]

    def subtitle_for(self, k):
        a, b = self.scores
        angle = np.degrees(np.arccos(self.cosine))
        return [
            "one base model, trained on Tiny Shakespeare, copied and fine-tuned twice: once on "
            "the play in capitals (task A), once on the play backwards (task B)",
            "a task vector is the tuned weights minus the base weights, %s numbers; these two "
            "have lengths %.2f and %.2f and point %.0f degrees apart" % (
                f"{self.size:,}", self.norms[0], self.norms[1], angle),
            "merging is arithmetic on those vectors: add both to the base, add half of each, or "
            "subtract one; each gives a new model with no more training",
            "held-out loss on A and on B over %d seeds: base %.2f and %.2f, each fine-tune "
            "%.2f and %.2f on its own task; the average %.2f and %.2f, the full sum only %.2f "
            "and %.2f" % (self.seeds, a[0], b[0], a[1], b[2], a[3], b[3], a[4], b[4]),
        ][k]

    def phase_1(self, sc, u, op):
        base = (0.14, 0.60)
        self.box(sc, *base, 150, 80, PAPER, op, border=INK)
        self.lines(sc, base[0], base[1] - 0.022, ["base", "model"], op, size=0.036, step=0.048,
                   anchor="mm")
        for r, ((name, what), ink) in enumerate(zip(self.TASKS, (self.A, self.B))):
            y = 0.44 + r * 0.33
            o = op * smoothstep(r * 0.2, r * 0.2 + 0.12, u)
            self.box(sc, 0.50, y, 250, 112, soft(ink, 0.75), o, border=ink)
            self.text(sc, 0.50, y - 0.14, "task %s: %s" % (name, what), o, size=0.032,
                      ink=dark(ink))
            for j, row in enumerate(self.sample[r]):
                self.label(sc, self.P(0.50, y - 0.065 + j * 0.065), row, 0.034, opacity=o,
                           ink=dark(ink, 0.4), kind="mono")
            a, b = (base[0] + 0.095, base[1] + (r - 0.5) * 0.08), (0.345, y)
            self.arrow(sc, a, b, ink, o)
            a2, b2 = (0.655, y), (0.755, y)
            self.arrow(sc, a2, b2, ink, o)
            t = ((u - r * 0.2) * 5) % 1.0
            self.pulse(sc, a, b, t * 2, ink, o)
            self.pulse(sc, a2, b2, t * 2 - 1, ink, o)
            self.box(sc, 0.85, y, 150, 80, soft(ink), o, border=ink)
            self.lines(sc, 0.85, y - 0.022, ["tuned on %s" % name,
                                             "%d updates" % self.stops[r]], o,
                       size=0.034, step=0.048, anchor="mm")

    def phase_2(self, sc, u, op):
        o0, scale = (0.16, 0.84), 21.0                          # px per unit of weight change
        angle = np.arccos(self.cosine)
        ends = []
        for r, (ink, theta) in enumerate(((self.A, 0.0), (self.B, angle))):
            dx, dy = self.frac(self.norms[r] * scale * np.cos(theta),
                               self.norms[r] * scale * np.sin(theta))
            ends.append((o0[0] + dx, o0[1] - dy))
            if u > 0.12 + r * 0.25:
                self.arrow(sc, o0, ends[r], ink, op,
                           smoothstep(0.1 + r * 0.25, 0.3 + r * 0.25, u), width=4.0)
        self.spot(sc, *o0, 9.0, INK, op)
        self.text(sc, o0[0], o0[1] + 0.055, "base weights", op, size=0.034, strong=True)
        oa, ob = op * smoothstep(0.3, 0.38, u), op * smoothstep(0.55, 0.63, u)
        self.spot(sc, *ends[0], 9.0, self.A, oa)
        self.text(sc, ends[0][0] + 0.02, ends[0][1], "tuned on A", oa, anchor="lm", size=0.034,
                  ink=dark(self.A))
        self.text(sc, (o0[0] + ends[0][0]) / 2, o0[1] - 0.045, "length %.2f" % self.norms[0], oa,
                  size=0.034, ink=dark(self.A))
        self.spot(sc, *ends[1], 9.0, self.B, ob)
        self.text(sc, ends[1][0] + 0.02, ends[1][1], "tuned on B", ob, anchor="lm", size=0.034,
                  ink=dark(self.B))
        self.text(sc, (o0[0] + ends[1][0]) / 2 + 0.02, (o0[1] + ends[1][1]) / 2,
                  "length %.2f" % self.norms[1], ob, anchor="lm", size=0.034, ink=dark(self.B))
        o = op * smoothstep(0.7, 0.8, u)
        self.lines(sc, 0.60, 0.40, [
            ("task vector =", None), ("tuned weights − base weights", None),
            ("", None),
            ("%s numbers each, drawn here" % f"{self.size:,}", None),
            ("in the flat sheet the two span:", None),
            ("true lengths, true angle", None),
            ("", None),
            ("%.0f° apart (cosine %.2f):" % (np.degrees(angle), self.cosine), None),
            ("the two fine-tunes mostly", None), ("changed different things", None)],
            o, size=0.034, step=0.047)

    def _spot(self, a, b):
        """Layout point of ``base + a * A + b * B`` on the map."""
        fx, fy, w, h = self.MAP
        dw, dh = self.frac(w, h)
        A, B = F.MERGE_A, F.MERGE_B
        step_b, step_a = B[1] - B[0], A[1] - A[0]
        x = fx - dw / 2 + (b - B[0] + step_b / 2) / (step_b * len(B)) * dw
        y = fy + dh / 2 - (a - A[0] + step_a / 2) / (step_a * len(A)) * dh
        return x, y

    LABEL = {"base": (0.0, 0.05, "mm"), "tuned on A": (0.0, -0.05, "mm"),  # noqa: RUF012
             "tuned on B": (0.0, 0.05, "mm"), "average": (0.02, 0.04, "lm"),
             "sum": (0.0, -0.05, "mm"), "A minus B": (0.0, -0.05, "mm"),
             "minus B": (0.0, 0.05, "mm")}

    def _map(self, sc, op, heat, shown_mixes, labels=True):
        fx, fy, w, h = self.MAP
        values = self.rise * heat
        self.grid(sc, values, fx, fy, w, h, op, vmax=3.0, hot=MAGENTA, cold=GREEN, lines=0.10)
        dw, dh = self.frac(w, h)
        self.text(sc, fx, fy + dh / 2 + 0.045, "← subtract B's vector        add B's vector →", op,
                  size=0.032, ink=dark(self.B))
        self.text(sc, fx - dw / 2, fy - dh / 2 - 0.04, "↑ more of A's vector", op, anchor="lm",
                  size=0.032, ink=dark(self.A))
        for (name, a, b, ink), o in zip(self.MIXES, shown_mixes):
            x, y = self._spot(a, b)
            self.disk(sc, self.P(x, y), 8.5, ink, op * o, lift=2.0, outline=0.0)
            self.disk(sc, self.P(x, y), 3.0, WHITE, op * o, lift=2.4, outline=0.0)
            if labels:
                dx, dy, anchor = self.LABEL[name]
                self.text(sc, x + dx, y + dy, name, op * o, anchor=anchor, size=0.032, strong=True)

    def phase_3(self, sc, u, op):
        at = [smoothstep(a, a + 0.08, u) for a in (0.0, 0.12, 0.22, 0.54, 0.40, 0.68, 0.82)]
        self._map(sc, op, 0.0, at)
        o, a, b = self._spot(0, 0), self._spot(1, 0), self._spot(0, 1)
        self.over(sc, o, a, self.A, op, smoothstep(0.05, 0.14, u), width=3.2)
        self.over(sc, o, b, self.B, op, smoothstep(0.15, 0.24, u), width=3.2)
        s = self._spot(1, 1)                                    # the sum: A's vector, then B's
        self.over(sc, a, s, self.B, op * 0.7, smoothstep(0.32, 0.40, u), width=2.0)
        self.over(sc, b, s, self.A, op * 0.7, smoothstep(0.32, 0.40, u), width=2.0)
        d = self._spot(1, -1)                                   # A's vector, then B's backwards
        self.over(sc, a, d, self.B, op * 0.7, smoothstep(0.60, 0.68, u), width=2.0)
        self.over(sc, o, self._spot(0, -1), self.B, op * 0.7, smoothstep(0.74, 0.82, u),
                  width=2.0)
        rows = (("tuned on A", "base + A's vector", 0.12), ("tuned on B", "base + B's vector",
                                                            0.22),
                ("sum", "base + A's + B's", 0.40), ("average", "base + ½ A's + ½ B's", 0.54),
                ("A minus B", "base + A's − B's", 0.68), ("minus B", "base − B's", 0.82))
        inks = {m[0]: m[3] for m in self.MIXES}
        for r, (name, how, start) in enumerate(rows):
            y = 0.35 + r * 0.094
            oo = op * smoothstep(start, start + 0.08, u)
            self.text(sc, 0.63, y, name, oo, anchor="lm", size=0.036, ink=dark(inks[name]))
            self.text(sc, 0.63, y + 0.047, how, oo, anchor="lm", size=0.03)

    def phase_4(self, sc, u, op):
        self._map(sc, op, smoothstep(0.05, 0.4, u), [1.0] * len(self.MIXES), labels=False)
        fx, fy, _w, h = self.MAP
        dh = self.frac(0, h)[1]
        self.text(sc, fx, fy + dh / 2 + 0.095, "green: held-out loss below the base's; pink: "
                  "above it (one seed)", op * smoothstep(0.3, 0.45, u) * (
                      1 - smoothstep(0.74, 0.8, u)), size=0.03)
        a, b = self.scores
        last = len(self.MIXES) - 1
        self.text(sc, fx, fy + dh / 2 + 0.095, "subtracting B's vector un-learns B (%.2f → %.2f)"
                  ", and A with it (%.2f → %.2f)" % (b[0], b[last], a[0], a[last]),
                  op * smoothstep(0.82, 0.9, u), size=0.03, strong=True)
        self.text(sc, 0.635, 0.335, "held-out loss", op, anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.79, 0.335, "on A", op, anchor="lm", size=0.032, ink=dark(self.A))
        self.text(sc, 0.88, 0.335, "on B", op, anchor="lm", size=0.032, ink=dark(self.B))
        top, full = float(self.scores.max()), self.span(0.08)[0]
        for r, (name, a, b, ink) in enumerate(self.MIXES):  # noqa: B007
            o = op * smoothstep(0.3 + r * 0.06, 0.38 + r * 0.06, u)
            y = 0.405 + r * 0.071
            self.spot(sc, 0.645, y, 7.5, ink, o)
            self.text(sc, 0.665, y, name, o, anchor="lm", size=0.034, ink=dark(ink))
            for col, colour in enumerate((self.A, self.B)):
                v = self.scores[col][r]
                xx = 0.79 + col * 0.09
                self.rect(sc, self.shift(self.P(xx, y + 0.028), full * v / top / 2, 0),
                          full * v / top, 5, colour, o, radius=2)
                self.text(sc, xx, y - 0.004, "%.2f" % v, o, anchor="lm", size=0.034, strong=True)
        self.text(sc, 0.635, 0.905, "mean of %d seeds; lower is better" % self.seeds,
                  op * smoothstep(0.75, 0.85, u), anchor="lm", size=0.03)


# ---------------------------------------------------------------------- 42
class KnowledgeEditing(Kit):
    """A rank-one edit to one GPT-2 MLP: the passage and what GPT-2 expects
    next, the layer picked by patching activations back, the edit as one
    column times one row, the prediction as the edit is turned up, and a
    second passage it leaves alone."""

    TITLE = "Knowledge editing"
    PHASES = (("the passage", 9.0), ("find the layer", 13.0), ("a rank-one edit", 14.0),
              ("turn it up", 11.0), ("side effects", 10.0))
    SEEN, PLANT, EDIT, CONTROL = BLUE, RED, AMBER, TEAL

    def setup(self):
        s = self.study
        self.layer = int(s.metadata["settings"]["layer"])
        self.patch = np.asarray(s.panels[0].data["values"], float)
        self.clean_p = float(s.panels[1].data["values"][0])
        self.dirty_p = float(self.patch[0, 0])          # an unchanged position patched: no change
        self.scales = np.asarray(s.panels[2].data["x"], float)
        self.response = np.asarray(s.panels[2].data["ys"], float)     # planted, observed
        self.side = np.asarray(s.panels[3].data["ys"], float)[0]
        e = edit_trace(self.layer, float(self.scales[-1]))
        self.e = e
        self.tokens = [shown_token(t) for t in e["tokens"]]
        self.control = [shown_token(t) for t in e["control"]]
        self.seen, self.plant = shown_token(e["observed"][0]), shown_token(e["target"][0])
        self.ctarget = shown_token(e["ctarget"][0])
        self.row = self.layer + 1                       # the stream that block's output enters
        self.restore = (self.patch - self.dirty_p) / (self.clean_p - self.dirty_p)

    def subtitle_for(self, k):
        e, r = self.e, self.response
        shift = float(np.abs(e["ctop_p"][:, 1] - e["ctop_p"][:, 0]).max())
        return [
            "nine tokens from the play; GPT-2 gives the comma that really follows %s; the plan "
            "is to make it say '%s' here instead, by changing one matrix" % (
                pct(self.clean_p), self.plant),
            "swap two tokens and the comma falls to %s; putting back one clean activation at a "
            "time shows which restore it: the last token after block %d brings back %s" % (
                pct(self.dirty_p), self.layer, pct(self.patch[self.row, -1])),
            "the edit to block %d's output matrix is one column times one row: %s numbers that "
            "set all %s changes, built so this passage's activation is pushed towards '%s'" % (
                self.layer, "{:,}".format(int(e["shape"].sum())),
                number(float(e["shape"].prod())), self.plant),
            "turning the edit up: at strength %d GPT-2 gives '%s' %s and the comma %s; at "
            "strength %d, %s and %s" % (
                self.scales[-2], self.plant, pct(r[0][-2]), pct(r[1][-2]), self.scales[-1],
                pct(r[0][-1]), pct(r[1][-1])),
            "a second passage, not used to build the edit: its real next token stays at %s and "
            "GPT-2's top guesses move by at most %.1f points" % (
                pct(self.side[-1]), 100 * shift),
        ][k]

    def _passage(self, sc, tokens, x0, x1, y, op, marked=(), ink=None):
        xs = self.tokens_row(sc, tokens, x0, x1, y, op, marked, ink)
        return xs

    def tokens_row(self, sc, tokens, x0, x1, y, op, marked=(), ink=None, h=44.0):
        n = len(tokens)
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        w = self.span((x1 - x0) / n)[0] - 5.0
        for i, (tok, x) in enumerate(zip(tokens, xs)):
            on = i in marked
            self.box(sc, x, y, w, h, soft(ink) if on else PAPER, op, tok,
                     ink=dark(ink, 0.35) if on else None, border=ink if on else AXIS,
                     size=0.036 if len(tok) < 4 else 0.03)
        return xs

    def _guesses(self, sc, x, y, words, values, inks, op, full=210.0, vmax=0.5, step=0.07):
        self.wordbars(sc, x, y, ["' %s '" % w for w in words], values, full, INK, op, vmax=vmax,
                      step=step, thick=24.0, inks=inks, size=0.036)

    def phase_1(self, sc, u, op):
        self.tokens_row(sc, self.tokens, 0.05, 0.62, 0.40, op, h=48.0)
        self.text(sc, 0.05, 0.325, "nine tokens of the play, as GPT-2 reads them", op,
                  anchor="lm", size=0.032, strong=True)
        self.box(sc, 0.78, 0.40, 130, 60, soft(DIM, 0.3), op, "GPT-2", border=INK, size=0.040)
        self.arrow(sc, (0.625, 0.40), (0.695, 0.40), INK, op)
        self.pulse(sc, (0.625, 0.40), (0.695, 0.40), (u * 4) % 1.0, INK, op * (u < 0.5), r=6.0)
        e = self.e
        words = [shown_token(t) for t in e["top"]] + [self.plant]
        values = list(e["top_p"][:, 0]) + [float(e["p_before"][0])]
        inks = [self.SEEN if w == self.seen else DIM for w in words[:-1]] + [self.PLANT]
        o = op * smoothstep(0.25, 0.4, u)
        self.text(sc, 0.56, 0.535, "its guesses for the next token", o, anchor="lm", size=0.032,
                  strong=True)
        self._guesses(sc, 0.63, 0.60, words, values, inks, o, step=0.066)
        o = op * smoothstep(0.5, 0.65, u)
        self.lines(sc, 0.05, 0.60, [
            ("the play goes on with a comma:", dark(self.SEEN)),
            ("GPT-2's second guess, %s" % pct(self.clean_p), dark(self.SEEN)),
            ("", None),
            ("the aim: make it say ' %s ' here," % self.plant, dark(self.PLANT)),
            ("which it now gives %s" % pct(e["p_before"][0]), dark(self.PLANT))],
            o, size=0.034, step=0.05)

    def phase_2(self, sc, u, op):
        x0, x1 = 0.17, 0.67
        self.tokens_row(sc, self.tokens, x0, x1, 0.335, op, marked=(2, 3), ink=self.PLANT,
                             h=38.0)
        len(self.patch)
        reveal = smoothstep(0.08, 0.7, u)
        w = self.span(x1 - x0)[0] - 5.0
        cell = self.grid(sc, self.restore, (x0 + x1) / 2, 0.655, w, 248.0, op, reveal=reveal,
                         vmax=1.0, ring=[(self.row, len(self.tokens) - 1)] if reveal >= 1 else (),
                         ring_ink=INK)
        for r, name in ((0, "going in"), (self.row, "after block %d" % self.layer)):
            self.text(sc, x0 - 0.012, cell(r, 0)[1], name, op, anchor="rm", size=0.03,
                      strong=r == self.row)
        self.text(sc, x0 - 0.012, cell(5, 0)[1], "depth ↓", op, anchor="rm", size=0.03)
        self.lines(sc, 0.705, 0.335, [
            ("two tokens swapped (red):", dark(self.PLANT)),
            ("the comma falls from", None),
            ("%s to %s" % (pct(self.clean_p), pct(self.dirty_p)), None)], op, size=0.032,
            step=0.045)
        self.lines(sc, 0.705, 0.515, [
            "each square: put back one", "clean activation, at that", "token and that depth",
            "red: the comma comes back", "blue: it falls further"],
            op * smoothstep(0.1, 0.2, u), size=0.032, step=0.045)
        self.lines(sc, 0.705, 0.80, [
            ("ringed: after block %d, the" % self.layer, None),
            ("last token restores %s" % pct(self.patch[self.row, -1]), None),
            ("so block %d is edited" % self.layer, None)],
            op * smoothstep(0.72, 0.82, u), size=0.032, step=0.045)

    def phase_3(self, sc, u, op):
        e, n = self.e, EDIT_SHOWN
        side = 176.0
        y = 0.63
        self.grid(sc, e["weight"], 0.15, y, side, side, op)
        self.text(sc, 0.15, y - 0.225, "block %d's output matrix" % self.layer, op, size=0.032,
                  strong=True)
        self.text(sc, 0.15, y + 0.225, "a corner of its %d × %d" % tuple(e["shape"]), op,
                  size=0.03)
        self.text(sc, 0.285, y, "+", op, size=0.06, strong=True)
        o1, o2 = op * smoothstep(0.05, 0.15, u), op * smoothstep(0.18, 0.28, u)
        strip = side / n
        cx = 0.35
        px = 0.385 + self.frac(side / 2)[0]
        self.grid(sc, e["column"][:, None], cx, y, strip, side, o1, hot=self.EDIT, cold=VIOLET)
        self.grid(sc, e["push"][None, :], px, y - self.frac(0, side / 2 + 20)[1], side, strip, o2,
                  hot=self.EDIT, cold=VIOLET)
        self.text(sc, px - self.frac(side / 2)[0], y - 0.27, "× one row: the push from the "
                  "comma towards ' %s '" % self.plant, o2, anchor="lm", size=0.03)
        self.text(sc, cx, y + 0.225, "one column", o1, size=0.03)
        reveal = smoothstep(0.3, 0.75, u)
        self.grid(sc, e["block"], px, y, side, side, op * smoothstep(0.28, 0.34, u),
                  reveal=reveal, hot=self.EDIT, cold=VIOLET)
        self.text(sc, px, y + 0.272, "every change = its column entry × its row entry",
                  op * smoothstep(0.3, 0.4, u), size=0.03)
        o = op * smoothstep(0.78, 0.88, u)
        self.lines(sc, 0.71, 0.42, [
            ("%s + %s numbers" % tuple(f"{int(v):,}" for v in e["shape"]), None),
            ("fix all %s changes" % number(float(e["shape"].prod())), None),
            ("", None),
            ("size of the whole edit: %.1f" % float(e["edit_norm"]), None),
            ("size of the matrix: %.0f" % float(e["weight_norm"]), None),
            ("", None),
            ("on this passage the block's", None),
            ("output moves %.0f units" % float(e["moved"]), None),
            ("towards ' %s '" % self.plant, None)], o, size=0.032, step=0.045)

    def phase_4(self, sc, u, op):
        k = min(int(u / 0.8 * len(self.scales)), len(self.scales) - 1)
        chart = self.chart(((0.10, 0.58), (0.38, 0.80)), self.scales, np.r_[
            self.response.ravel(), 0.0], pad=0.08)
        self.axes(sc, chart, op, "strength of the edit", "chance GPT-2 gives the token",
                  xticks=[(v, number(v)) for v in self.scales if v != 2],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.1, 0.2, 0.3)])
        for trace, ink in zip(self.response, (self.PLANT, self.SEEN)):
            pts = [self.P(chart.fx(x), chart.fy(v)) for x, v in zip(self.scales[:k + 1], trace)]
            self.polyline(sc, pts, 3.4, ink, op, bias=0.0)
            for x, v in zip(self.scales[:k + 1], trace):
                self.spot(sc, chart.fx(x), chart.fy(v), 6.0, ink, op)
        self.text(sc, 0.66, 0.36, "edit strength %d" % self.scales[k], op, anchor="lm",
                  size=0.040, strong=True)
        self._guesses(sc, 0.72, 0.47, [self.plant, self.seen], self.response[:, k],
                      [self.PLANT, self.SEEN], op, full=150.0, vmax=0.4, step=0.09)
        self.lines(sc, 0.66, 0.66, [
            ("' %s ': planted by the edit" % self.plant, dark(self.PLANT)),
            ("' %s ': what the play has" % self.seen, dark(self.SEEN))], op, size=0.032,
            step=0.047)
        self.lines(sc, 0.66, 0.79, ["six strengths were tried:", "each point is GPT-2 run",
                                    "with the edited matrix"], op, size=0.03, step=0.042)

    def phase_5(self, sc, u, op):
        self.tokens_row(sc, self.control, 0.05, 0.62, 0.40, op, h=48.0)
        self.text(sc, 0.05, 0.325, "another passage of the play, with the edit at strength %d" %
                  self.scales[-1], op, anchor="lm", size=0.032, strong=True)
        self.box(sc, 0.78, 0.40, 150, 60, soft(self.EDIT, 0.35), op, "edited GPT-2",
                 border=self.EDIT, size=0.036)
        self.arrow(sc, (0.625, 0.40), (0.685, 0.40), INK, op)
        self.pulse(sc, (0.625, 0.40), (0.685, 0.40), (u * 4) % 1.0, INK, op * (u < 0.5), r=6.0)
        e = self.e
        words = [shown_token(t) for t in e["ctop"]] + [self.ctarget]
        before = list(e["ctop_p"][:, 0]) + [float(e["c_before"])]
        after = list(e["ctop_p"][:, 1]) + [float(e["c_after"])]
        o = op * smoothstep(0.15, 0.3, u)
        self.text(sc, 0.14, 0.535, "its guesses for the next token: before, and after the edit",
                  o, anchor="lm", size=0.032, strong=True)
        for r, (word, b, a) in enumerate(zip(words, before, after)):
            y = 0.60 + r * 0.064
            last = r == len(words) - 1
            self.text(sc, 0.13, y, "' %s '" % word, o, anchor="rm", size=0.036,
                      ink=dark(self.CONTROL) if last else None)
            for col, (v, ink) in enumerate(((b, DIM), (a, self.CONTROL))):
                at = self.P(0.14, y + (col - 0.5) * 0.026)
                length = max(300.0 * v / 0.3, 0.5)
                self.rect(sc, self.shift(at, length / 2, 0), length, 10, ink, o, radius=2)
            self.text(sc, 0.14 + self.frac(300.0 * max(a, b) / 0.3 + 10)[0], y,
                      "%s → %s" % (pct(b), pct(a)), o, anchor="lm", size=0.032, strong=True)
        self.lines(sc, 0.62, 0.62, [
            ("' %s ' is what the play has" % self.ctarget, dark(self.CONTROL)),
            ("next here", dark(self.CONTROL)),
            ("", None),
            ("the edit was aimed at one", None), ("activation; this passage's", None),
            ("is different, so little moves", None), ("", None),
            ("one passage and one control:", None), ("a measurement, not proof", None)],
            op * smoothstep(0.5, 0.65, u), size=0.03, step=0.042)


def shown_token(tok):
    """A GPT-2 token as it is drawn: its leading space and newlines visible."""
    return str(tok).replace("\n", "↵").replace(" ", "␣")


ANIMATIONS = {
    "instruction-tuning": InstructionTuning,
    "reward-model": RewardModel,
    "rlhf-ppo": RlhfPpo,
    "dpo": Dpo,
    "reward-hacking": RewardHacking,
    "prompt-and-prefix-tuning": PromptAndPrefixTuning,
    "model-merging": ModelMerging,
    "knowledge-editing": KnowledgeEditing,
}
