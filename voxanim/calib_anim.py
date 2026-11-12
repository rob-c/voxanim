"""Confidence is not correctness: GPT-2 small on 40 capital cities, its
confidence against whether it is right.

Every prompt has the same two worked examples in front of it (without them
GPT-2 small answers "the" to every country):

    The capital of Kenya is Nairobi. The capital of Argentina is Buenos Aires.
    The capital of <country> is ___

The model's answer is its most probable next token and its *confidence* is
that token's probability.  The 40 countries (:data:`COUNTRIES`) are ones
whose name and whose capital are each a single GPT-2 token, so the answer
is one token and right or wrong is unambiguous.

1. **one prompt** -- the first country worked in full: the top tokens as
   bars, the answer, right or wrong, and a dot placed at its confidence,
2. **40 prompts** -- the rest go by: each answer flashes up and its dot
   lands on the confidence axis, coloured right or wrong,
3. **reliability** -- the dots gather into five confidence bins; for each
   bin the bar is the fraction right (the accuracy) and the grey marker the mean
   confidence.  A perfectly calibrated model would put every bar on the
   diagonal.  The expected calibration error (ECE, the count-weighted mean
   gap between confidence and accuracy) is read out,
4. **same number** -- a right answer and a wrong one given the same
   confidence: from inside the model they are the same number.

Every number is GPT-2 small's (:mod:`voxanim.gpt2_probe`), computed once
and cached (:func:`trace`).  :func:`reliability` and :func:`ece` are the
plain numpy definitions, tested on hand-made data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import gpt2_probe as P
from . import layout
from .cache import cached_npz
from .conv import Tempo
from .diagram import EDGE, Diagram
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import hash01
from .prob_bars import draw_bars
from .text import num

# ------------------------------------------------------------------ the prompts

FRAME = "The capital of Kenya is Nairobi. The capital of Argentina is Buenos Aires. "
QUESTION = "The capital of %s is"
# (country, capital): both single GPT-2 tokens with a leading space (a test checks)
COUNTRIES = (
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("Portugal", "Lisbon"), ("Japan", "Tokyo"), ("China", "Beijing"), ("Russia", "Moscow"),
    ("Egypt", "Cairo"), ("Greece", "Athens"), ("Austria", "Vienna"), ("Hungary", "Budapest"),
    ("Poland", "Warsaw"), ("Ireland", "Dublin"), ("Norway", "Oslo"), ("Sweden", "Stockholm"),
    ("Denmark", "Copenhagen"), ("Finland", "Helsinki"), ("Belgium", "Brussels"),
    ("Netherlands", "Amsterdam"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"),
    ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Peru", "Lima"), ("Cuba", "Havana"),
    ("Chile", "Santiago"), ("Canada", "Ottawa"), ("Australia", "Canberra"), ("India", "Delhi"),
    ("Pakistan", "Islamabad"), ("Afghanistan", "Kabul"), ("Thailand", "Bangkok"),
    ("Indonesia", "Jakarta"), ("Philippines", "Manila"), ("Tunisia", "Tunis"),
    ("Libya", "Tripoli"), ("Switzerland", "Bern"), ("Jamaica", "Kingston"))
N = len(COUNTRIES)
TOP_K = 5                       # bars shown for the worked prompt
N_BINS = 5                      # confidence bins of the reliability diagram


def prompt_text(country):
    """The full prompt for ``country``."""
    return FRAME + QUESTION % country


# ------------------------------------------------------------------ calibration arithmetic


def reliability(conf, right, bins=N_BINS):
    """Bin confidences into ``bins`` equal intervals of [0, 1].

    Returns ``(edges, count, mean_conf, accuracy)``: the ``bins + 1`` edges,
    and per bin how many items, their mean confidence and the fraction
    right (0 for an empty bin).  The first bin includes 0; every other bin
    is open on the left, closed on the right.
    """
    conf, right = np.asarray(conf, np.float64), np.asarray(right, np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    count = np.bincount(idx, minlength=bins)
    safe = np.maximum(count, 1)
    mean_conf = np.bincount(idx, conf, minlength=bins) / safe
    acc = np.bincount(idx, right, minlength=bins) / safe
    return edges, count, mean_conf, acc


def ece(conf, right, bins=N_BINS):
    """Expected calibration error: the count-weighted mean over bins of
    |accuracy - mean confidence|."""
    _edges, count, mean_conf, acc = reliability(conf, right, bins)
    return float(np.sum(count / count.sum() * np.abs(acc - mean_conf)))


# ------------------------------------------------------------------ the model's numbers


@dataclass
class Item:
    """One country: the model's answer and whether it is the capital."""

    country: str
    capital: str
    answer: str                 # the top token, stripped
    conf: float                 # its probability
    right: bool
    p_true: float               # the probability of the capital
    top: list                   # [(token string, probability)] for the bars (first item only)


def _probe(country, capital):
    """Top-k ids and probabilities, and the capital's probability, for one country."""
    p = P.next_probs(P.encode(prompt_text(country)))
    top = np.argsort(-p)[:TOP_K]
    return top, p[top], p[P.token_id(" " + capital)]


def trace(cache_dir=None):
    """The :class:`Item` for every country, cached by :func:`voxanim.cache.cached_npz`."""
    settings = dict(frame=FRAME, question=QUESTION, countries=COUNTRIES, top_k=TOP_K)

    def build():
        tops, probs, p_true = [], [], []
        for country, capital in COUNTRIES:
            top, pr, pt = _probe(country, capital)
            tops.append(top)
            probs.append(pr)
            p_true.append(pt)
        return dict(top_ids=np.array(tops), top_p=np.array(probs), p_true=np.array(p_true))

    d = cached_npz("gpt2-calibration", settings, build, cache_dir)
    items = []
    for (country, capital), ids, pr, pt in zip(COUNTRIES, d["top_ids"], d["top_p"], d["p_true"]):
        top = [(P.decode([i]), float(p)) for i, p in zip(ids, pr)]
        answer = top[0][0].strip()
        items.append(Item(country, capital, answer, top[0][1], answer == capital, float(pt), top))
    return items


def same_number(items):
    """``(right item, wrong item)`` whose confidences are closest, preferring
    confident ones: the pair the last phase holds up."""
    rights = [i for i in items if i.right]
    wrongs = [i for i in items if not i.right]
    best = min(((abs(r.conf - w.conf) - 0.1 * min(r.conf, w.conf), r, w)
                for r in rights for w in wrongs), key=lambda x: x[0])
    return best[1], best[2]


# ------------------------------------------------------------------ layout (fractions, px)

COL_X, COL_W = 0.03, 0.42                   # the left column
PROMPT_Y, LINE_PX = 0.20, 22.0              # the prompt lines
HEAD_Y, BARS_Y = 0.36, 0.405                # the bars' header and first row
VERDICT_Y = 0.70                            # the answer line under the bars
TALLY_Y = 0.78                              # "k of n right so far"
TABLE_Y, TABLE_PX = 0.36, 26.0              # the per-bin table (phase 3)
ECE_Y = 0.70
PAIR_Y = 0.82                               # the two held-up answers (phase 4)
CHART = (0.56, 0.24, 0.98, 0.96)            # the chart box (x0, y0, x1, y1), layout
STRIP_TOP = 0.68                            # the box's top while it is only the confidence axis
CHART_PAD = (34.0, 14.0, 12.0, 30.0)        # inner margins: left, top, right, bottom (px)
DOT_R = 4.2                                 # a prompt's dot (px)
TICK_R = 3.2                                # the mean-confidence marker on each bar (px)
SWARM_PX = 9.5                              # dot spacing in the strip and the stacks (px)
STRIP_COLS = 40                             # strip slots along the confidence axis
STACK_ACROSS = 4                            # dots per row inside a bin column
RIGHT = C.rgb("#2a9d8f")                    # a right answer
WRONG = C.rgb("#d1495b")                    # a wrong one
PANEL = (0.955, 0.955, 0.945)
PANEL_EDGE = (0.74, 0.75, 0.76)
ACC_BAR = C.rgb("#c9d6d4")                  # the accuracy bars
DIAGONAL = (0.55, 0.57, 0.59)

TIMELINE = dict(intro=(0.2, 1.2), prompt=(1.0, 2.2), bars=(2.4, 4.4), verdict=(4.6, 5.6),
                drop=(5.8, 6.8), sweep=(7.6, 26.6), gather=(28.4, 30.4), accbars=(30.6, 32.4),
                table=(30.8, 33.8), ece=(34.4, 35.4), pair=(37.0, 38.5))
T_DONE = 42.0
SWEEP_TEMPO = Tempo(slow=1.2, fast=0.36, ramp_in=4.0, ramp_out=2.0)


class Calibration(Diagram):
    """The ``gpt2-calibration`` animation (see the module docstring)."""

    PHASES = ("one prompt", "40 prompts", "reliability", "same number")

    def __init__(self):
        super().__init__("Confidence is not correctness", "")
        self.items = trace()
        conf = np.array([i.conf for i in self.items])
        right = np.array([i.right for i in self.items], float)
        self.edges, self.count, self.mean_conf, self.acc = reliability(conf, right)
        self.ece = ece(conf, right)
        self.accuracy = float(right.mean())
        self.pair = same_number(self.items)
        self.hold, self.reset = 3.0, 1.0
        self.Tm = TIMELINE
        d = SWEEP_TEMPO.durations(N - 1)
        t0 = TIMELINE["sweep"][0]
        self.sweep_t = t0 + np.concatenate([[0.0], np.cumsum(d)]) * (
            (TIMELINE["sweep"][1] - t0) / d.sum())        # when each item 1.. appears
        self.strip_slots = self._strip_slots()
        self.stack_slots = self._stack_slots()
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "pair"), (2, "gather"), (1, "sweep")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ dot positions
    def _strip_slots(self):
        """Each dot's (column, row) in the strip along the confidence axis: dots
        that share a column stack upwards in order of appearance."""
        used, slots = {}, []
        for it in self.items:
            col = min(int(it.conf * STRIP_COLS), STRIP_COLS - 1)
            row = used.get(col, 0)
            used[col] = row + 1
            slots.append((col, row))
        return slots

    def _stack_slots(self):
        """Each dot's (bin, row, column) inside its bin's column: right answers
        first from the bottom, wrong ones on top, STACK_ACROSS per row."""
        order = sorted(range(N), key=lambda i: (self.items[i].conf, not self.items[i].right))
        idx = np.clip(np.ceil(np.array([self.items[i].conf for i in order]) * N_BINS)
                      .astype(int) - 1, 0, N_BINS - 1)
        used, slots = {}, [None] * N
        by_bin = [[i for i, b in zip(order, idx) if b == k] for k in range(N_BINS)]
        for k, members in enumerate(by_bin):
            members.sort(key=lambda i: (not self.items[i].right, self.items[i].conf))
            for n, i in enumerate(members):
                slots[i] = (k, n // STACK_ACROSS, n % STACK_ACROSS)
            used[k] = len(members)
        return slots

    def _chart_pt(self, x, y):
        """World point in the chart for confidence ``x`` and accuracy ``y`` (0..1)."""
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        left, top, right, bottom = CHART_PAD
        return self.shift(a, left + (w - left - right) * x, top + (h - top - bottom) * (1 - y))

    def _strip_pos(self, i):
        col, row = self.strip_slots[i]
        return self.shift(self._chart_pt((col + 0.5) / STRIP_COLS, 0.0), 0.0,
                          -(row + 0.6) * SWARM_PX)

    def _stack_pos(self, i):
        k, row, col = self.stack_slots[i]
        x = (k + 0.5) / N_BINS
        dx = (col - (STACK_ACROSS - 1) / 2) * SWARM_PX
        return self.shift(self._chart_pt(x, 0.0), dx, -(row + 0.6) * SWARM_PX)

    # ------------------------------------------------------------------ state
    def _shown(self, t):
        """``(index of the item on show, how far its beat has run 0..1)``."""
        if t < self.Tm["sweep"][0]:
            return 0, 1.0
        k = int(np.searchsorted(self.sweep_t, t, side="right") - 1)
        k = max(0, min(k, N - 2))
        dt = (self.sweep_t[k + 1] - self.sweep_t[k]) if k + 1 < N else 1.0
        return k + 1, remap(t, self.sweep_t[k], self.sweep_t[k] + min(0.5, 0.8 * dt))

    def _dot_on(self, t, i):
        """How far dot ``i`` has landed (0..1)."""
        if i == 0:
            return ease_out(self.progress("drop", t))
        if t < self.Tm["sweep"][0]:
            return 0.0
        k = i - 1
        dt = (self.sweep_t[k + 1] - self.sweep_t[k]) if k + 1 < N else 0.6
        return ease_out(remap(t, self.sweep_t[k] + 0.15 * dt, self.sweep_t[k] + 0.8 * dt))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        gather = ease_in_out(self.progress("gather", t))
        i, _beat = self._shown(t)
        it = self.items[i]
        self._prompt(sc, it, t, back, gather)
        self._chart(sc, t, back, gather)
        self._table(sc, t, back)
        self._pair(sc, t, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, i, reset)

    def _prompt(self, sc, it, t, back, gather):
        """The prompt (its last line changes with the country), the bars for the
        first one, the answer and the running tally; all fade when the dots gather."""
        on = smoothstep(0.0, 1.0, self.progress("prompt", t)) * back * (1 - gather)
        if on <= 0.01:
            return
        top = self.P(COL_X, PROMPT_Y)
        lines = FRAME.strip().split(". ")
        lines = [lines[0] + ".", lines[1], QUESTION % it.country + " ___"]
        for k, line in enumerate(lines):
            ink = layout.NOTE_INK if k < 2 else layout.STRONG_INK
            self.label(sc, self.shift(top, 0, k * LINE_PX), line, layout.MIN_TEXT, anchor="lm",
                       ink=ink, opacity=on)
        first = t < self.Tm["sweep"][0]
        grow = ease_out(self.progress("bars", t))
        if grow > 0:
            head = self.P(COL_X, HEAD_Y)
            items = it.top if first else it.top[:3]
            self.label(sc, head, "next token: p", layout.MIN_TEXT, anchor="lm", opacity=on,
                       ink=layout.NOTE_INK)
            draw_bars(self, sc, self.P(COL_X, BARS_Y), items, grow if first else 1.0, on)
        verdict = ease_out(self.progress("verdict", t)) if first else 1.0
        if verdict > 0:
            mark = "right" if it.right else "wrong (%s %s)" % (it.capital, num(it.p_true))
            text = "%s → %s   %s   %s" % (it.country, it.answer, num(it.conf), mark)
            ink = RIGHT if it.right else WRONG
            self.label(sc, self.P(COL_X, VERDICT_Y), text, layout.CAPTION, anchor="lm",
                       ink=tuple(float(v) for v in C.darken(ink, 0.25)), opacity=on * verdict)
        done = [k for k in range(N) if self._dot_on(t, k) >= 1.0]
        if len(done) > 1:
            n_right = sum(self.items[k].right for k in done)
            self.label(sc, self.P(COL_X, TALLY_Y), "%d of %d right so far" % (n_right, len(done)),
                       layout.CAPTION, anchor="lm", opacity=on, ink=layout.STRONG_INK)

    def _chart(self, sc, t, back, gather):
        """The confidence axis with the dots, becoming the reliability diagram."""
        x0, y0, x1, y1 = CHART
        top = STRIP_TOP + (y0 - STRIP_TOP) * gather          # the box grows up as the dots gather
        a, b = self.P(x0, top), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        on = smoothstep(0.0, 1.0, self.progress("verdict", t)) * back
        if on <= 0.01:
            return
        self.rect(sc, (a + b) / 2, w, h, PANEL, on, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        for x in np.linspace(0, 1, N_BINS + 1):
            q = self._chart_pt(x, 0.0)
            self.label(sc, q, num(x, 1) if x not in (0, 1) else "%d" % x, layout.MIN_TEXT,
                       anchor="mt", dy=6, opacity=on, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, self._chart_pt(0.5, 0.0), "confidence: p of the answer", layout.MIN_TEXT,
                   anchor="mt", dy=22, opacity=on, ink=layout.NOTE_INK)
        sc.line(self._chart_pt(0, 0) + self.view * 0.3, self._chart_pt(1, 0) + self.view * 0.3,
                width=1.2, color=EDGE, opacity=0.8 * on, bias=0.3)
        # the reliability diagram: y axis, diagonal, accuracy bars, mean-confidence ticks
        rel = ease_out(self.progress("accbars", t)) * back
        if gather > 0:
            for y in (0.5, 1.0):
                q = self._chart_pt(0.0, y)
                self.label(sc, q, num(y, 1) if y < 1 else "1", layout.MIN_TEXT, anchor="rm",
                           dx=-6, opacity=on * gather, ink=layout.NOTE_INK, kind="mono")
            self.label(sc, self._chart_pt(0.0, 1.0), "accuracy", layout.MIN_TEXT, anchor="lb",
                       dx=4, dy=-6, opacity=on * gather, ink=layout.NOTE_INK)
            sc.line(self._chart_pt(0, 0) + self.view * 0.3, self._chart_pt(1, 1) + self.view * 0.3,
                    width=1.2, color=DIAGONAL, opacity=0.7 * on * gather, bias=0.3)
            self.label(sc, self._chart_pt(0.86, 0.86), "calibrated", layout.MIN_TEXT, anchor="rb",
                       dx=-4, dy=-2, opacity=on * gather, ink=layout.NOTE_INK)
        if rel > 0:
            for k in range(N_BINS):
                if self.count[k] == 0:
                    continue
                lo, hi = self._chart_pt(k / N_BINS, 0.0), self._chart_pt((k + 1) / N_BINS,
                                                                           self.acc[k] * rel)
                c = (lo + hi) / 2
                bw, bh = np.abs(self.px(hi) - self.px(lo))
                self.rect(sc, c, bw - 6, max(bh, 0.5), ACC_BAR, 0.85 * back, radius=2, lift=0.15)
                q = self._chart_pt(self.mean_conf[k], self.acc[k] * rel)
                self.disk(sc, q, TICK_R, DIAGONAL, 0.95 * rel, lift=0.7, outline=0.0)
        hl = ease_in_out(self.progress("pair", t)) * back
        for i, it in enumerate(self.items):
            land = self._dot_on(t, i)
            if land <= 0:
                continue
            strip = self._strip_pos(i)
            if gather > 0:
                p = strip + (self._stack_pos(i) - strip) * ease_in_out(
                    remap(gather, 0.3 * hash01(i), 0.3 * hash01(i) + 0.7))
            else:
                start = self.shift(self.P(COL_X, VERDICT_Y), 120, 0)
                p = start + (strip - start) * land
                if land < 1:
                    p = self.shift(p, 0, -40 * land * (1 - land))
            col = RIGHT if it.right else WRONG
            held = hl > 0 and it in self.pair
            r = DOT_R * (1 + 0.6 * hl) if held else DOT_R
            self.disk(sc, p, r, col, back * (1 - 0.55 * hl * (not held)), lift=1.0, outline=0.1,
                      ring=col if held else None, ring_w=2.0, ring_op=hl)
            if land < 1:
                sc.probe(("dot", i), p, tol=6.0)

    def _table(self, sc, t, back):
        """Per-bin read-outs and the ECE, in the left column (phase 3)."""
        on = ease_out(self.progress("table", t)) * back
        if on <= 0.01:
            return
        top = self.P(COL_X, TABLE_Y)
        head = "%-10s  %4s  %4s  %5s" % ("confidence", "n", "conf", "right")
        self.label(sc, top, head, layout.MIN_TEXT, anchor="lm", opacity=on, ink=layout.NOTE_INK,
                   kind="mono")
        for k in range(N_BINS):
            row = self.shift(top, 0, (k + 1) * TABLE_PX)
            lo, hi = self.edges[k], self.edges[k + 1]
            if self.count[k]:
                text = "%s to %s  %4d  %s   %s" % (num(lo, 1), num(hi, 1), self.count[k],
                                                   num(self.mean_conf[k]), num(self.acc[k]))
            else:
                text = "%s to %s  %4d" % (num(lo, 1), num(hi, 1), 0)
            k_on = smoothstep(0.0, 1.0, remap(on, k / N_BINS, (k + 1) / N_BINS))
            self.label(sc, row, text, layout.MIN_TEXT, anchor="lm", opacity=k_on * back,
                       ink=layout.CAPTION_INK, kind="mono")
        e = ease_out(self.progress("ece", t)) * back
        if e > 0:
            text = "ECE = Σ n/N · |right − conf| = %s" % num(self.ece, 3)
            self.label(sc, self.P(COL_X, ECE_Y), text, layout.CAPTION, anchor="lm", opacity=e,
                       ink=layout.STRONG_INK)
            self.label(sc, self.shift(self.P(COL_X, ECE_Y), 0, 26),
                       "accuracy %s, mean confidence %s" % (
                           num(self.accuracy), num(np.mean([i.conf for i in self.items]))),
                       layout.MIN_TEXT, anchor="lm", opacity=e, ink=layout.CAPTION_INK)

    def _pair(self, sc, t, back):
        """The right and the wrong answer with the same confidence (phase 4)."""
        on = ease_out(self.progress("pair", t)) * back
        if on <= 0.01:
            return
        for k, it in enumerate(self.pair):
            text = "%s → %s   %s   %s" % (it.country, it.answer, num(it.conf),
                                          "right" if it.right else "wrong: %s" % it.capital)
            ink = RIGHT if it.right else WRONG
            self.label(sc, self.shift(self.P(COL_X, PAIR_Y), 0, k * 26), text, layout.CAPTION,
                       anchor="lm", opacity=on, ink=tuple(float(v) for v in C.darken(ink, 0.25)))

    # ------------------------------------------------------------------ words
    def _words(self, t, i, reset):
        """The subtitle at ``t``, with the real numbers."""
        it0 = self.items[0]
        T = self.Tm
        if reset > 0 or t < T["bars"][0]:
            return ("%d countries whose capital is one token, each asked the same way after two "
                    "worked examples; GPT-2 small's answer is its most probable next token" % N)
        if t < T["drop"][0]:
            return ("%s: '%s' with p = %s, which is %s. That probability is the model's "
                    "confidence" % (it0.country, it0.answer, num(it0.conf),
                                    "right" if it0.right else "wrong"))
        if t < T["sweep"][0]:
            return "each answer becomes a dot at its confidence: %s right, %s wrong" % (
                "teal", "red")
        if t < T["gather"][0]:
            it = self.items[i]
            return ("the other %d: %s → %s (%s, %s). The wrong answers are plausible big cities"
                    % (N - 1, it.country, it.answer, num(it.conf),
                       "right" if it.right else "wrong, %s" % it.capital))
        if t < T["ece"][0]:
            return ("reliability diagram: %d bins of confidence; the bar is how often the model "
                    "was right in that bin, the tick its mean confidence" % N_BINS)
        if t < T["pair"][0]:
            k = int(np.argmax(self.count))
            above = int(np.sum((self.acc > self.mean_conf) & (self.count > 0)))
            return ("%d of %d bars sit above the diagonal: the model is right more often than "
                    "it says (%s to %s: confidence %s, right %d of %d). ECE %s"
                    % (above, int(np.sum(self.count > 0)), num(self.edges[k], 1),
                       num(self.edges[k + 1], 1), num(self.mean_conf[k]),
                       round(self.acc[k] * self.count[k]), self.count[k], num(self.ece, 3)))
        r, w = self.pair
        return ("%s %s and %s %s: one is right, one is wrong, and from inside the model they "
                "are the same number" % (r.answer, num(r.conf), w.answer, num(w.conf)))
