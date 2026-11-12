"""Why a model makes things up: the real GPT-2 small on a real and an
invented country.

    The capital of Germany is Berlin. The capital of Italy is Rome.
    The capital of France is ___          The capital of Freedonia is ___

Both prompts have the same shape (two worked examples, then the question);
only the subject differs.  The model does not look anything up: it predicts
the next token from the text, and the text looks the same either way.

1. **prompts** -- the two prompts side by side,
2. **next token** -- GPT-2's next-token distribution after each, as bars
   (the six most probable tokens, their probabilities, the entropy of the
   whole distribution in bits).  After the real prompt one token dominates;
   after the invented one the distribution is flatter, but a city name
   is still on top: a fluent answer with no fact behind it,
3. **samples** -- six continuations sampled from each distribution (seeds
   0 to 5, temperature 1).  The first word is the answer, with the
   probability the model gave it.  The real prompt mostly agrees with
   itself; the invented one names a different city every time.  That
   disagreement -- self-consistency -- is a signal the text alone does not
   carry,
4. **why** -- the distribution is shaped by what followed similar text in
   training: after "The capital of X is" came a city name.  "unknown" is
   not a probable continuation after either prompt (its probability and
   rank are read out): the model predicts plausible text, not truth.

Every number is GPT-2 small's (:mod:`voxanim.gpt2_probe`), computed once
and cached (:func:`trace`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import color as C
from . import gpt2_probe as P
from . import layout
from .cache import cached_npz
from .diagram import Diagram
from .easing import ease_out, remap, smoothstep
from .prob_bars import BAR, LABEL_PX, ROW_PX, draw_bars
from .text import num

# ------------------------------------------------------------------ the prompts

FRAME = ("The capital of Germany is Berlin.", "The capital of Italy is Rome.")
QUESTION = "The capital of %s is"
REAL, INVENTED = "France", "Freedonia"
TOP_K = 6                       # tokens shown as bars
N_SAMPLES, N_TOKENS = 6, 4      # continuations sampled per prompt, tokens each
UNKNOWN = " unknown"            # the token whose probability the last phase reads out
SEEDS = tuple(range(N_SAMPLES))


TINY = 1e-4                     # probabilities below this read "< 0.0001"


def tiny(p):
    """A probability for a read-out: ``= 0.0123``, or ``< 0.0001`` when it is smaller."""
    return "< %g" % TINY if p < TINY else "= " + num(p, 4)


def prompt_text(subject):
    """The full prompt for ``subject``: the two examples and the question."""
    return " ".join(FRAME) + " " + QUESTION % subject


@dataclass
class Side:
    """One prompt's numbers, decoded for display."""

    subject: str
    top: list                   # [(token string, probability)], most probable first
    entropy: float              # bits
    unknown_p: float            # probability of UNKNOWN as the next token
    unknown_rank: int           # its rank (1 = most probable)
    answers: list               # [(first word of each sample, its probability)]

    @property
    def mode(self):
        """``(answer, count)``: the most frequent sampled answer and how often it came."""
        return Counter(a for a, _p in self.answers).most_common(1)[0]

    @property
    def distinct(self):
        """How many different answers the samples gave."""
        return len({a for a, _p in self.answers})


def _probe(subject):
    """Arrays for one prompt (see :func:`trace`)."""
    ids = P.encode(prompt_text(subject))
    p = P.next_probs(ids)
    order = np.argsort(-p)
    top = order[:TOP_K]
    uid = P.token_id(UNKNOWN)
    rank = int(np.nonzero(order == uid)[0][0]) + 1
    samples = np.zeros((N_SAMPLES, N_TOKENS), np.int64)
    probs = np.zeros((N_SAMPLES, N_TOKENS))
    words = np.zeros(N_SAMPLES, np.int64)
    for k, seed in enumerate(SEEDS):
        out, pr = P.sample(ids, N_TOKENS, seed)
        samples[k], probs[k] = out, pr
        words[k] = P.first_word(out)
    return dict(top_ids=top, top_p=p[top], entropy=np.array(P.entropy(p)),
                unknown=np.array([p[uid], rank]), samples=samples, sample_p=probs, words=words)


def trace(cache_dir=None):
    """Both prompts' numbers as :class:`Side` objects, ``(real, invented)``.

    Cached by :func:`voxanim.cache.cached_npz` under every setting that
    affects them (prompts, k, seeds, the token read out).
    """
    settings = dict(frame=FRAME, question=QUESTION, subjects=(REAL, INVENTED), top_k=TOP_K,
                    n_tokens=N_TOKENS, seeds=SEEDS, unknown=UNKNOWN)

    def build():
        out = {}
        for tag, subject in (("real", REAL), ("inv", INVENTED)):
            for k, v in _probe(subject).items():
                out[tag + "_" + k] = v
        return out

    d = cached_npz("gpt2-hallucination", settings, build, cache_dir)
    sides = []
    for tag, subject in (("real", REAL), ("inv", INVENTED)):
        g = {k[len(tag) + 1:]: v for k, v in d.items() if k.startswith(tag + "_")}
        top = [(P.decode([i]), float(p)) for i, p in zip(g["top_ids"], g["top_p"])]
        answers = []
        for ids, pr, n in zip(g["samples"], g["sample_p"], g["words"]):
            answers.append((P.decode(ids[:n]).strip(), float(np.prod(pr[:n]))))
        sides.append(Side(subject, top, float(g["entropy"]), float(g["unknown"][0]),
                          int(g["unknown"][1]), answers))
    return tuple(sides)


# ------------------------------------------------------------------ layout (fractions, px)

COL_X = (0.03, 0.53)                    # left edge of each column (layout fraction)
COL_W = 0.44                            # column width (layout fraction)
PROMPT_Y, LINE_PX = 0.20, 22.0          # first prompt line; line pitch (px)
HEAD_Y, BARS_Y = 0.36, 0.405            # the bars' header and first row
UNKNOWN_PX = 10.0                       # gap above the 'unknown' row (px)
SAMPLES_Y, CHIP_Y, CHIP_PX = 0.775, 0.825, 29.0   # samples header, first chip row, row pitch
TALLY_Y = 0.95
CHIP_H, CHIP_PAD = 22.0, 14.0           # chip height and horizontal padding (px)
PANEL = (0.955, 0.955, 0.945)           # chip fill
PANEL_EDGE = (0.70, 0.71, 0.72)
AGREE = BAR                             # chips that agree with the most frequent answer
FRAME_HL = C.rgb("#f3e3b8")             # highlight behind the shared frame (phase 4)

TIMELINE = dict(intro=(0.2, 1.2), prompt_a=(1.0, 2.6), prompt_b=(2.6, 4.2),
                bars_a=(4.6, 7.6), bars_b=(8.0, 11.0), entropy=(11.4, 13.0),
                samples=(14.0, 29.0), tally=(29.4, 31.4), why=(33.0, 35.0),
                unknown=(36.5, 38.5))
T_DONE = 42.5
SAMPLE_DT = 2.4                         # seconds per sampled answer (both columns together)


class Hallucination(Diagram):
    """The ``gpt2-hallucination`` animation (see the module docstring)."""

    PHASES = ("prompts", "next token", "samples", "why")

    def __init__(self):
        super().__init__("Why a model makes things up", "")
        self.sides = trace()
        self.hold, self.reset = 3.0, 1.0
        self.Tm = TIMELINE
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "why"), (2, "samples"), (1, "bars_a")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        why = smoothstep(0.0, 1.0, self.progress("why", t)) * back
        unk = ease_out(self.progress("unknown", t)) * back
        for n, side in enumerate(self.sides):
            x0 = COL_X[n]
            on = smoothstep(0.0, 1.0, self.progress("prompt_b" if n else "prompt_a", t)) * back
            self._prompt(sc, x0, side, on, why)
            grow = ease_out(self.progress("bars_b" if n else "bars_a", t))
            self._bars(sc, x0, side, grow, back, why, unk)
            self._samples(sc, x0, side, t, back, why)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)
        del intro

    def _prompt(self, sc, x0, side, on, why):
        """The prompt as three lines: the two examples (the shared frame) and
        the question with its subject; the frame is highlighted in phase 4."""
        if on <= 0.01:
            return
        lines = list(FRAME) + [QUESTION % side.subject + " ___"]
        top = self.P(x0, PROMPT_Y)
        if why > 0:
            w = COL_W * layout.REF_W * self.cam.zoom
            c = self.shift(top, w / 2 - 6, LINE_PX / 2)
            self.rect(sc, c, w, 2 * LINE_PX + 10, FRAME_HL, 0.8 * why, radius=5, lift=0.1)
        for k, line in enumerate(lines):
            ink = layout.NOTE_INK if k < 2 else layout.STRONG_INK
            self.label(sc, self.shift(top, 0, k * LINE_PX), line, layout.MIN_TEXT, anchor="lm",
                       ink=ink, opacity=on)

    def _bars(self, sc, x0, side, grow, back, why, unk):
        """The top-k bars, the entropy read-out and (phase 4) the 'unknown' row."""
        if grow <= 0:
            return
        head = self.P(x0, HEAD_Y)
        self.label(sc, head, "next token: p", layout.MIN_TEXT, anchor="lm", opacity=back,
                   ink=layout.NOTE_INK)
        ent = smoothstep(0.0, 1.0, remap(grow, 0.9, 1.0)) * back
        right = self.shift(self.P(x0 + COL_W, HEAD_Y), 0, 0)
        self.label(sc, right, "entropy %s bits" % num(side.entropy, 1), layout.MIN_TEXT,
                   anchor="rm", opacity=ent, ink=layout.STRONG_INK)
        draw_bars(self, sc, self.P(x0, BARS_Y), side.top, grow, back, dim=why > 0.5)
        if unk > 0:
            row = self.shift(self.P(x0, BARS_Y), 0, TOP_K * ROW_PX + UNKNOWN_PX)
            self.label(sc, row, UNKNOWN.strip(), layout.MIN_TEXT, anchor="lm", opacity=unk,
                       ink=layout.STRONG_INK, kind="mono")
            text = "p %s   rank %s" % (tiny(side.unknown_p), format(side.unknown_rank, ","))
            self.label(sc, self.shift(row, LABEL_PX, 0), text, layout.MIN_TEXT, anchor="lm",
                       opacity=unk, ink=layout.STRONG_INK, kind="mono")

    def _sample_on(self, t, k):
        """How far sampled answer ``k`` has appeared (0..1)."""
        t0 = self.Tm["samples"][0] + k * SAMPLE_DT
        return ease_out(remap(t, t0, t0 + 0.6))

    def _samples(self, sc, x0, side, t, back, why):
        """The sampled answers as chips (three per row) and the agreement tally."""
        if t < self.Tm["samples"][0]:
            return
        fade = back * (1.0 - 0.6 * why)
        self.label(sc, self.P(x0, SAMPLES_Y), "%d sampled continuations, first word:"
                   % N_SAMPLES, layout.MIN_TEXT, anchor="lm", opacity=fade, ink=layout.NOTE_INK)
        mode, count = side.mode
        agree = count > 1
        pitch = COL_W / 3
        for k, (word, p) in enumerate(side.answers):
            on = self._sample_on(t, k) * fade
            if on <= 0.01:
                continue
            r, c = divmod(k, 3)
            centre = self.shift(self.P(x0 + pitch * (c + 0.5), CHIP_Y), 0, r * CHIP_PX)
            text = "%s  %s" % (word, num(p))
            w = layout.text_width(text, layout.MIN_TEXT) * layout.REF_W + CHIP_PAD
            edge = AGREE if (agree and word == mode) else PANEL_EDGE
            self.rect(sc, centre, w, CHIP_H, PANEL, on * 0.95, border=edge, border_w=1.4,
                      radius=5, lift=0.2)
            self.label(sc, centre, text, layout.MIN_TEXT, opacity=on, ink=layout.STRONG_INK)
        tally = ease_out(self.progress("tally", t)) * fade
        if tally > 0:
            if side.distinct == N_SAMPLES:
                text = "%d different answers" % side.distinct
            else:
                text = "%d of %d agree: %s" % (count, N_SAMPLES, mode)
            self.label(sc, self.P(x0 + COL_W / 2, TALLY_Y), text, layout.CAPTION, opacity=tally,
                       ink=layout.STRONG_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle: the first script entry whose beat has not started."""
        script = self._script()
        if reset > 0 or t < self.Tm["prompt_b"][0]:
            return script[0][1]
        for beat, text in script[1:]:
            if beat is None or t < self.Tm[beat][0]:
                return text
        raise AssertionError("unreachable: the script ends with a catch-all")

    def _script(self):
        """``[(shown until this beat starts, text)]``, with the real numbers."""
        real, inv = self.sides
        rt, it = real.top[0], inv.top[0]
        mode, count = real.mode
        return [
            (None, "two prompts of the same shape: two worked examples, then a question about "
                   "a real country and about an invented one"),
            ("bars_a", "%s exists, %s does not. GPT-2 small cannot tell: it only sees text, "
                       "and predicts the next token" % (REAL, INVENTED)),
            ("bars_b", "after the real prompt the next-token distribution is sharp: '%s' %s, "
                       "entropy %s bits" % (rt[0].strip(), num(rt[1]), num(real.entropy, 1))),
            ("samples", "after the invented one it is flatter (entropy %s bits), but a city "
                        "name is still on top: '%s' %s. Fluent either way"
                        % (num(inv.entropy, 1), it[0].strip(), num(it[1]))),
            ("tally", "sample from each distribution %d times (seeds 0 to %d): the first word "
                      "is the answer, with the probability the model gave it"
                      % (N_SAMPLES, N_SAMPLES - 1)),
            ("why", "the real prompt agrees with itself (%d of %d say %s); the invented one "
                    "gives %d different cities. Self-consistency is a signal"
                    % (count, N_SAMPLES, mode, inv.distinct)),
            ("unknown", "why: the distribution is shaped by what followed text like this in "
                        "training, and after 'The capital of … is' came a city name"),
            (None, "'unknown' is not a probable continuation: p %s (rank %s) and %s (rank %s). "
                   "The model predicts plausible text, not truth"
                   % (tiny(real.unknown_p), format(real.unknown_rank, ","),
                      tiny(inv.unknown_p), format(inv.unknown_rank, ","))),
        ]
