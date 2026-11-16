"""Decoding and serving: animations 29-34 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the decoder at work on one worked example and then the numbers its study
measured. Beam search and batching draw the first experiments
(:mod:`voxanim.next_language`); the other four draw their redesigned ones
(:mod:`voxanim.study_decoding`: speculative decoding with a draft chosen by measurement,
and the three logit-reshaping decoders measured over held-out passages as well as on
their worked prompt). What a mechanism needs and a study's four panels do not keep - the
beam's hypotheses, the drafted and replaced tokens, the rollouts under each penalty, what
the grammar removed - comes from the experiments' own arrays (``beam_trace`` here, the
``*_data`` functions there) and is checked against the study's numbers by the tests.
"""

from __future__ import annotations

import numpy as np

from . import gpt2, gpt2_hooks, layout
from . import study_decoding as SD
from .cache import cached_npz
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    INKS,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    TRACK,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    number,
    soft,
)
from .study_decoding import CONTRAST_PROMPT, GRAMMAR, GRAMMAR_PROMPT, PENALTY_PROMPT, RULES

TRACE_VERSION = 2
INK_DARK = (0.16, 0.17, 0.19)


# ---------------------------------------------------------------------- real data
def _tok():
    from .next_language import _tok as tok
    return tok()


def show(token):
    """A GPT-2 token as it is written on screen: its text, without the
    leading space that marks a new word."""
    s = _tok().decode([int(token)])
    s = s.replace("\n", "↵").strip()
    return s if s else "␣"


def _cached(name, build, **settings):
    return cached_npz("mech-decoding-" + name,
                      dict(version=TRACE_VERSION, gpt2_revision=gpt2.REVISION, **settings), build)


def _logp(seq, memo, n_layers=12):
    key = (tuple(int(i) for i in seq), n_layers)
    if key not in memo:
        p = gpt2.softmax(gpt2_hooks.forward(list(key[0]), n_layers=n_layers)["logits"][-1])
        memo[key] = np.log(np.maximum(p, 1e-300))
    return memo[key]


BEAM_PROMPT = "The best way to learn is"
BEAM_STEPS = 5


def beam_trace(widths=(1, 2, 4, 8)):
    """The study's beam search (``next_language.beam_search``), keeping what it
    threw away: for each width ``w``, per step, every candidate's new token,
    cumulative log probability and parent (an index into the hypotheses kept
    at the step before), best first - the first ``w`` are the ones kept.
    ``alt_tok`` / ``alt_p`` are the three likeliest next tokens at each step
    of the greedy (width 1) path."""
    def build():
        ids = _tok().encode(BEAM_PROMPT)
        memo, out = {}, {}
        for w in widths:
            beams = [(list(ids), 0.0)]
            tok = np.full((BEAM_STEPS, w * w), -1, np.int64)
            score = np.full((BEAM_STEPS, w * w), np.nan)
            parent = np.full((BEAM_STEPS, w * w), -1, np.int64)
            for s in range(BEAM_STEPS):
                cands = []
                for b, (seq, sc) in enumerate(beams):
                    lp = _logp(seq, memo)
                    for token in np.argsort(-lp)[:w]:
                        cands.append((seq + [int(token)], sc + float(lp[token]), b))
                cands.sort(key=lambda item: item[1], reverse=True)
                for c, (seq, sc, b) in enumerate(cands):
                    tok[s, c], score[s, c], parent[s, c] = seq[-1], sc, b
                beams = [(seq, sc) for seq, sc, _ in cands[:w]]
            out["tok%d" % w], out["score%d" % w], out["parent%d" % w] = tok, score, parent
            out["best%d" % w] = np.asarray(beams[0][0][len(ids):])
            if w == 1:
                seq, alt_tok, alt_p = list(ids), [], []
                for s in range(BEAM_STEPS):  # noqa: B007
                    lp = _logp(seq, memo)
                    top = np.argsort(-lp)[:3]
                    alt_tok.append(top)
                    alt_p.append(np.exp(lp[top]))
                    seq.append(int(top[0]))
                out["alt_tok"], out["alt_p"] = np.asarray(alt_tok), np.asarray(alt_p)
        return out
    return _cached("beam", build, prompt=BEAM_PROMPT, steps=BEAM_STEPS, widths=list(widths))


# ---------------------------------------------------------------------- shared drawing
def seg(u, a, b):
    """0..1 as ``u`` runs from ``a`` to ``b``."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


def prob(v):
    """A probability to two significant figures, written out (0.00003, not 3e-05)."""
    v = float(v)
    if v <= 0:
        return "0"
    return "%.*f" % (max(2, 1 - int(np.floor(np.log10(v)))), v)


def pct(v):
    """A probability as a percentage with two significant figures."""
    v = 100.0 * float(v)
    return ("%.2g%%" % v) if v < 10 else ("%.0f%%" % v)


class Decoder(Mechanism):
    """What the six animations share: token chips, meters, crosses."""

    def chip(self, sc, fx, fy, w, h, ink, op, text="", size=0.034, solid=False, border=None):
        """A token in the colour ``ink`` (None: plain paper); ``solid``: filled, white text."""
        if ink is None:
            self.box(sc, fx, fy, w, h, PAPER, op, text, border=AXIS if border is None else border,
                     size=size)
        elif solid:
            self.box(sc, fx, fy, w, h, ink, op, text, ink=WHITE, size=size, border=border)
        else:
            self.box(sc, fx, fy, w, h, soft(ink), op, text, ink=dark(ink, 0.4),
                     border=ink if border is None else border, size=size)

    def meter(self, sc, fx, fy, length, ink, op, label="", value="", thick=24.0, size=0.032,
              track=None, ghost=None, label_ink=None, value_at=None):
        """A bar ``length`` px long growing right from ``(fx, fy)``; ``ghost``
        (px) outlines another length behind it; ``track`` (px) the full scale."""
        at = self.P(float(fx), float(fy))
        if track:
            self.rect(sc, self.shift(at, track / 2, 0), track, thick, TRACK, 0.5 * op, radius=4,
                      lift=0.3)
        w = max(float(length), 0.1)
        self.rect(sc, self.shift(at, w / 2, 0), w, thick, ink, op, radius=min(4.0, w / 2))
        if ghost:
            self.rect(sc, self.shift(at, ghost / 2, 0), ghost, thick, AXIS, op, border=AXIS,
                      border_w=1.6, radius=min(4.0, ghost / 2), lift=0.7, fill_op=0.0)
        if label:
            self.label(sc, self.shift(at, -10, 0), str(label), size, anchor="rm", opacity=op,
                       ink=label_ink)
        if value != "":
            end = max(w, ghost or 0.0) if value_at is None else value_at
            self.label(sc, self.shift(at, end + 8, 0), str(value), size, anchor="lm", opacity=op,
                       ink=layout.STRONG_INK)

    def cross(self, sc, fx, fy, r, ink, op, width=3.0):
        """An X of half-size ``r`` px at ``(fx, fy)``."""
        c = self.P(float(fx), float(fy))
        for s in (1, -1):
            self.polyline(sc, [self.shift(c, -r, -r * s), self.shift(c, r, r * s)], width, ink,
                          op, bias=0.9)

    def segment(self, sc, a, b, ink, op, width=2.0):
        """A straight line between two layout points."""
        if op > 0.01:
            self.polyline(sc, [self.P(*a), self.P(*b)], width, ink, op, bias=0.3)

    def tally(self, sc, fx, fy, flags, ink, op, u=1.0, pitch=0.0265, size=20.0):
        """One square per passage from ``(fx, fy)`` rightwards, filled in ``ink`` where
        ``flags`` is set; ``u`` (0..1) shows that share of them."""
        shown = int(np.ceil(len(flags) * float(np.clip(u, 0.0, 1.0)) - 1e-9))
        for i, flag in enumerate(flags[:shown]):
            self.box(sc, fx + pitch * i, fy, size, size, ink if flag else PAPER, op,
                     border=ink if flag else AXIS, radius=4)


# ---------------------------------------------------------------------- 29 beam search
class BeamSearch(Decoder):
    """Greedy decoding as a chain that keeps one token a step, then the beam of
    four as a tree growing column by column - sixteen candidates, four kept,
    twelve cut - then the two paths through it and what each width found."""

    TITLE = "Beam search"
    PHASES = (("greedy", 11.0), ("beam of 4", 20.0), ("two paths", 11.0), ("widths", 11.0))
    W = 4
    COLS = [0.232 + 0.158 * s for s in range(5)]  # noqa: RUF012
    ROWS = [0.47 + 0.118 * r for r in range(4)]  # noqa: RUF012
    NODE = (88.0, 50.0)
    HALF = 0.048                      # half a node's width, with the gap a line leaves

    def setup(self):
        s = self.study
        self.trace = t = beam_trace()
        self.prompt = [show(i) for i in _tok().encode(BEAM_PROMPT)]
        self.widths = [int(w) for w in s.metadata["settings"]["widths"]]
        self.final = [float(v) for v in s.panels[2].data["values"]]       # best score per width
        self.texts = {w: [show(i) for i in t["best%d" % w]] for w in self.widths}
        self.greedy_sum = [float(v) for v in s.panels[3].data["ys"][0]]   # greedy, step by step
        self.tok, self.score, self.parent = t["tok4"], t["score4"], t["parent4"]
        self.count = [int((self.tok[k] >= 0).sum()) for k in range(BEAM_STEPS)]
        self.greedy_path = self.follow(t["best1"])
        self.best_path = self.follow(t["best4"])

    def follow(self, tokens):
        """Where the continuation ``tokens`` sits among the width-4 candidates
        at each step (an index, best first; None once it has been cut)."""
        path, at = [], 0
        for k, token in enumerate(tokens):
            hit = None
            if at is not None:
                for c in range(self.count[k]):
                    if self.tok[k, c] == token and (k == 0 or self.parent[k, c] == at):
                        hit = c
                        break
            path.append(hit)
            at = hit if hit is not None and hit < self.W else None
        return path

    def subtitle_for(self, k):
        g, b = self.greedy_path, self.best_path
        lost = next(i for i, c in enumerate(g) if c is None or c >= self.W)
        low = max(range(BEAM_STEPS), key=lambda i: b[i])
        ratio = float(np.exp(self.final[2] - self.final[0]))
        return [
            "greedy decoding: GPT-2 names its likeliest next token, that one is kept and the "
            "rest are dropped. Five tokens: “%s”, log-probability %.2f" % (
                " ".join(self.texts[1]), self.final[0]),
            "beam search, width 4: each of the 4 kept continuations proposes its 4 likeliest "
            "next tokens; of the 16 candidates the 4 with the highest total survive",
            "greedy's path stayed in the beam until token %d, where it ranked %d of %d and was "
            "cut; the winner, “%s”, was only %d of 4 at token %d" % (
                lost + 1, g[lost] + 1, self.count[lost], " ".join(self.texts[4]), b[low] + 1,
                low + 1),
            "the best five tokens found: widths 1 and 2, log-probability %.2f; widths 4 and 8, "
            "%.2f, a continuation %.1f times as probable" % (self.final[0], self.final[2], ratio),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _prompt(self, sc, op, note, ink=None):
        self.tokens(sc, self.prompt, 0.05, 0.43, 0.33, op, w=56.0, h=38.0, size=0.034)
        self.text(sc, 0.455, 0.33, note, op, anchor="lm", size=0.034, ink=ink, strong=ink is None)

    def _node(self, sc, x, y, token, second, op, ink=None, solid=False, border=None):
        w, h = self.NODE
        if ink is None:
            self.rect(sc, self.P(x, y), w, h, PAPER, op, border=border or AXIS, radius=6.0)
            top = low = None
        elif solid:
            self.rect(sc, self.P(x, y), w, h, ink, op, border=border, radius=6.0, border_w=2.6)
            top = low = WHITE
        else:
            self.rect(sc, self.P(x, y), w, h, soft(ink, 0.7), op, border=border or ink,
                      radius=6.0, border_w=2.2)
            top = low = dark(ink, 0.45)
        self.text(sc, x, y - 0.021, token, op, size=0.034, ink=top, strong=top is None)
        self.text(sc, x, y + 0.022, second, op, ink=low)

    def _root(self, sc, y, op):
        self.box(sc, 0.085, y, 58, 44, INK_DARK, op, "… is", ink=WHITE, size=0.034)
        return 0.085 + 0.034, y

    # ------------------------------------------------------------------ 1 greedy
    def phase_1(self, sc, u, op):
        t = self.trace
        self._prompt(sc, op, "greedy: keep one token, drop the rest", dark(RED))
        ys = self.ROWS[:3]
        src = self._root(sc, ys[0], op)
        now = min(int(u / 0.86 * BEAM_STEPS), BEAM_STEPS - 1)
        for k in range(now + 1):
            v = 1.0 if k < now else seg(u / 0.86 * BEAM_STEPS - k, 0.0, 1.0)
            x = self.COLS[k]
            chosen = v > 0.62
            for r in range(3):
                self.segment(sc, src, (src[0] + (x - self.HALF - src[0]) * smoothstep(0, 0.35, v),
                                       src[1] + (ys[r] - src[1]) * smoothstep(0, 0.35, v)),
                             RED if r == 0 else DIM, op * (1.0 if r == 0 else 0.9), 2.4)
                show_op = op * smoothstep(0.25, 0.5, v)
                label, second = show(t["alt_tok"][k, r]), "p %.2f" % t["alt_p"][k, r]
                if r == 0 and chosen:
                    self._node(sc, x, ys[r], label, second, show_op, RED, solid=True)
                else:
                    self._node(sc, x, ys[r], label, second,
                               show_op * (1 - 0.6 * smoothstep(0.62, 0.85, v)))
            src = (x + self.HALF, ys[0])
        done = now if u / 0.86 * BEAM_STEPS - now < 0.6 else now + 1
        if done:
            self.text(sc, 0.05, 0.81, "the kept tokens' log-probabilities, added up: %.2f" % (
                self.greedy_sum[done - 1]), op, anchor="lm", size=0.036, strong=True)
        self.text(sc, 0.05, 0.875, "p: GPT-2's probability of that token, given the ones kept "
                  "before it", op, anchor="lm", size=0.030)

    # ------------------------------------------------------------------ the tree of the beam
    def _stub(self, k, c):
        """Where the cut candidate ``c`` of step ``k`` ends: a short branch off its parent."""
        p = int(self.parent[k, c])
        rank = [i for i in range(self.count[k]) if self.parent[k, i] == p].index(c)
        return (self.COLS[k - 1] + self.HALF + 0.034, self.ROWS[p] + (rank - 1.5) * 0.026)

    def _tree(self, sc, op, upto, v=1.0, marks=()):
        """The width-4 search up to step ``upto`` (that one ``v`` of the way
        through: candidates fan out, the best four become nodes, the rest are
        cut). ``marks``: ``(path, ink, steps lit)`` paths to pick out."""
        root = self._root(sc, (self.ROWS[1] + self.ROWS[2]) / 2, op)
        lit = {}
        for path, ink, n in marks:
            for k in range(min(n, BEAM_STEPS)):
                if path[k] is not None:
                    lit.setdefault((k, path[k]), []).append(ink)
        for k in range(min(upto + 1, BEAM_STEPS)):
            w = 1.0 if k < upto else v
            grow, appear, cut = smoothstep(0.0, 0.35, w), smoothstep(0.3, 0.55, w), \
                smoothstep(0.62, 0.85, w)
            for c in range(self.count[k]):
                src = root if k == 0 else (self.COLS[k - 1] + self.HALF,
                                           self.ROWS[int(self.parent[k, c])])
                inks = lit.get((k, c), [])
                if c < self.W:
                    end = (self.COLS[k] - self.HALF, self.ROWS[c])
                    self.segment(sc, src, (src[0] + (end[0] - src[0]) * grow,
                                           src[1] + (end[1] - src[1]) * grow),
                                 inks[-1] if inks else BLUE, op * (1.0 if inks else 0.55),
                                 3.4 if inks else 1.8)
                    second = "%.2f" % self.score[k, c]
                    if inks:
                        self._node(sc, self.COLS[k], self.ROWS[c], show(self.tok[k, c]), second,
                                   op * appear, inks[-1], solid=True,
                                   border=inks[0] if len(inks) > 1 else None)
                    else:
                        self._node(sc, self.COLS[k], self.ROWS[c], show(self.tok[k, c]), second,
                                   op * appear, BLUE)
                else:
                    end = self._stub(k, c)
                    ink = inks[-1] if inks else tuple(np.asarray(BLUE) * (1 - cut)
                                                      + np.asarray(DIM) * cut)
                    tip = (src[0] + (end[0] - src[0]) * grow, src[1] + (end[1] - src[1]) * grow)
                    self.segment(sc, src, tip, ink, op * (1.0 if inks else 0.8),
                                 3.0 if inks else 1.6)
                    self.disk(sc, self.P(*tip), 5.0 if inks else 3.4, ink, op * grow, outline=0.0)

    def phase_2(self, sc, u, op):
        q = u / 0.9 * BEAM_STEPS
        now = min(int(q), BEAM_STEPS - 1)
        v = seg(q - now, 0.0, 1.0)
        self._prompt(sc, op, "beam of 4: token %d of 5" % (now + 1), dark(BLUE))
        self._tree(sc, op, now, v)
        n = self.count[now]
        if n > self.W:
            note = "%d candidates: the 4 best are kept (last in: %s %.2f), %d are cut (first " \
                "out: %s %.2f)" % (n, show(self.tok[now, 3]), self.score[now, 3], n - self.W,
                                   show(self.tok[now, 4]), self.score[now, 4])
        else:
            note = "the 4 likeliest first tokens: all are kept. Under each, its log-probability"
        self.text(sc, 0.05, 0.92, note, op * smoothstep(0.3, 0.5, v), anchor="lm", size=0.030)

    # ------------------------------------------------------------------ 3 two paths
    def phase_3(self, sc, u, op):
        g = int(seg(u, 0.04, 0.40) * BEAM_STEPS + 1e-9)
        b = int(seg(u, 0.48, 0.84) * BEAM_STEPS + 1e-9)
        self._prompt(sc, op, "the same tree, two paths through it")
        self._tree(sc, op, BEAM_STEPS, 1.0, [(self.greedy_path, RED, g), (self.best_path, BLUE, b)])
        self.text(sc, 0.05, 0.92, "greedy: %s  %.2f" % (" ".join(self.texts[1]), self.final[0]),
                  op * smoothstep(0.04, 0.12, u), anchor="lm", size=0.032, ink=dark(RED))
        self.text(sc, 0.52, 0.92, "beam of 4: %s  %.2f" % (" ".join(self.texts[4]),
                                                           self.final[2]),
                  op * smoothstep(0.48, 0.56, u), anchor="lm", size=0.032, ink=dark(BLUE))

    # ------------------------------------------------------------------ 4 widths
    def phase_4(self, sc, u, op):
        self._prompt(sc, op, "the best five tokens each width finds")
        full = self.span(0.19)[0]
        best = max(self.final)
        self.text(sc, 0.585, 0.415, "probability of all five together", op, anchor="lm",
                  strong=True, size=0.030)
        for row, (w, score) in enumerate(zip(self.widths, self.final)):
            y = 0.49 + row * 0.105
            a = op * smoothstep(row * 0.13, row * 0.13 + 0.15, u)
            ink = RED if self.texts[w] == self.texts[1] else BLUE
            self.text(sc, 0.05, y, "width %d" % w, a, anchor="lm", size=0.034, strong=True)
            xs = [0.185 + 0.078 * i for i in range(BEAM_STEPS)]
            for x, word in zip(xs, self.texts[w]):
                self.chip(sc, x, y, 64, 42, ink, a, word, size=0.032)
            self.meter(sc, 0.585, y, full * float(np.exp(score - best)), ink, a,
                       value="%s  (log %.2f)" % (pct(np.exp(score)), score),
                       thick=30.0, size=0.030, track=full, value_at=full)
        self.text(sc, 0.05, 0.905, "a wider beam finds likelier text, not better text: here "
                  "width 8 found nothing width 4 had not", op * smoothstep(0.6, 0.8, u),
                  anchor="lm", size=0.030)


# ---------------------------------------------------------------------- 30 speculative decoding
def short(word, n=9):
    """``word`` cut to ``n`` characters, for a chip."""
    return word if len(word) <= n else word[:n - 1] + "…"


class SpeculativeDecoding(Decoder):
    """A cheap draft guesses a few tokens and GPT-2 checks them all in one
    pass: the first round of a real run token by token, then every round of
    it, then why the result is exactly GPT-2's distribution; then the
    experiment - which draft is worth using (how often each candidate agrees
    with GPT-2 on held-out text, what it costs, and the one the rule picked)
    and what that draft and the first design measured over several passages
    and seeds."""

    TITLE = "Speculative decoding"
    PHASES = (("draft", 9.0), ("verify", 11.0), ("the run", 11.0), ("exact", 9.0),
              ("which draft", 10.0), ("measured", 9.0))
    Y = 0.54                                        # the token row
    PROMPT_X = (0.05, 0.40)
    SHOWN = 4                                       # prompt tokens drawn (after an ellipsis)
    DRAFT_AT, GPT_AT = (0.25, 0.335), (0.715, 0.335)
    READS = {"unigram": 0, "bigram": 1, "trigram": 2}  # noqa: RUF012

    def setup(self):
        s = self.study
        self.d = d = SD.speculative_data()
        self.trace = t = dict({key[6:]: d[key] for key in d if key.startswith("trace_")},
                              **{key: d[key] for key in ("id_tok", "id_p", "id_q", "id_freq",
                                                         "id_spot", "id_context", "overlap")})
        settings = s.metadata["settings"]
        self.block = int(settings["block"])
        self.draft_name, self.control_name = settings["chosen"], settings["control"]
        self.control_block = int(settings["control_block"])
        chosen = int(d["chosen"])
        self.kind = SD.KIND_NAMES[int(d["cand_kind"][chosen])]
        self.layer = int(d["cand_layer"][chosen])
        self.names = list(s.panels[1].data["labels"])
        self.alpha, self.cost = (np.asarray(s.panels[1].data[k], float) for k in ("alpha", "cost"))
        self.speed = np.asarray(s.panels[1].data["values"], float)        # at the best block
        self.chosen = chosen
        # the measurement: passes per token (plain, chosen draft, first design), and the rest
        m = s.panels[2].data
        self.passes = [float(v) for v in m["values"]]
        self.per_seed = np.asarray(m["per_seed"], float)
        self.kept_share = [float(v) for v in m["kept"]]
        self.seconds = [float(v) for v in m["seconds"]]
        self.total_tokens = int(s.metadata["results"]["tokens"])
        self.n_prompts, self.n_seeds, self.new = (int(settings[k]) for k in (
            "prompts", "seeds", "new_tokens"))
        self.identity_error = float(d["identity_error"])
        self.mc = float(np.mean(d["mc_spec"])), float(np.mean(d["mc_direct"]))
        # the worked run: the first passage, the first seed
        ids = [int(i) for i in d["prompts"][0]]
        self.prompt = ["…"] + [show(i) for i in ids[-self.SHOWN:]]
        self.output = [int(i) for i in d["outputs"][0, 0, 0]]
        self.rounds = len(t["start"])
        self.size = [int((row >= 0).sum()) for row in t["prop"]]          # guesses per round
        self.n_kept = [int((row == 1).sum()) for row in t["state"]]
        self.n_tokens, self.n_passes = len(self.output), self.rounds
        self.checked = int((t["state"] > 0).sum())
        self.accepted = int((t["state"] == 1).sum())
        pitch = min(0.13, (0.93 - 0.455 - 0.05) / max(self.block - 1, 1))
        self.SLOTS = [0.455 + pitch * i for i in range(self.block)]
        self.CHIP = min(106.0, self.span(pitch)[0] - 12.0)
        self.shortlist = self._shortlist()

    def _shortlist(self):
        """The candidates worth a row: the chosen one, the best of each other kind, the
        first design's draft and the deepest exit."""
        kinds = [SD.KIND_NAMES[int(k)] for k in self.d["cand_kind"]]
        layers = [int(v) for v in self.d["cand_layer"]]
        rows = [self.chosen]
        for family in (("tuned",), ("exit",), SD.NGRAMS):
            rows.append(max((i for i, k in enumerate(kinds) if k in family),
                            key=lambda i: self.speed[i]))
        rows.append(next(i for i, (k, v) in enumerate(zip(kinds, layers))
                         if (k, v) == SD.CONTROL[:2]))
        rows.append(next(i for i, (k, v) in enumerate(zip(kinds, layers))
                         if (k, v) == ("tuned", max(layers))))
        return list(dict.fromkeys(rows))

    def _control(self):
        return next(i for i, name in enumerate(self.names) if name == self.control_name)

    def subtitle_for(self, k):
        t = self.trace
        kept, size = self.n_kept[0], self.size[0]
        first = "“%s”: p/q = %s / %s = %s" % (
            show(t["prop"][0, 0]), prob(t["p"][0, 0]), prob(t["q"][0, 0]),
            pct(min(1.0, t["p"][0, 0] / t["q"][0, 0])))
        if kept == size:
            outcome = "here all %d were kept, and the pass gave the token after them too" % size
        elif kept == 0:
            outcome = "here the first was rejected, so the guesses after it were thrown away"
        else:
            outcome = "here %d %s kept, the next was rejected and the rest thrown away" % (
                kept, "was" if kept == 1 else "were")
        how = {
            "bigram": "a table counted from Shakespeare: after each token, how often every "
                      "token came next",
            "trigram": "a table counted from Shakespeare: after each pair of tokens, how often "
                       "every token came next",
            "unigram": "a table of how often each token occurs in Shakespeare",
        }.get(self.kind, self.draft_name)
        c = self._control()
        best_tuned = max((i for i, n in enumerate(self.names) if "trained" in n),
                         key=lambda i: self.alpha[i])
        ratio = self.seconds[0] / self.seconds[1]
        return [
            "the draft is cheap: %s. It guesses the next %d tokens, one %s each; q is its own "
            "probability of each guess" % (how, self.block,
                                           "look-up" if self.kind in SD.NGRAMS else "call"),
            "GPT-2 reads the text and all %d guesses in one pass. Each guess is kept with "
            "chance p/q, in order. %s; %s" % (size, first, outcome),
            "this run: %d of %d checked guesses were kept (%s); GPT-2 needed %d passes for %d "
            "tokens, where plain decoding needs %d" % (
                self.accepted, self.checked, pct(self.accepted / self.checked), self.n_passes,
                self.n_tokens, self.n_tokens),
            "what is kept is the part both models agree on, %s of the probability here; the "
            "rest is redrawn where GPT-2 is higher. The total is exactly GPT-2's distribution "
            "(largest gap measured: %.0e)" % (
                pct(float(t["overlap"])), max(self.identity_error, 1e-17)),
            "a draft must be cheap and agree with GPT-2. Its own first %d layers agree on %s "
            "of the probability and cost %s of a call; a trained read-out reaches %s, at %s; "
            "the %s: %s, for %s" % (
                SD.CONTROL[1], pct(self.alpha[c]), pct(self.cost[c]), pct(self.alpha[best_tuned]),
                pct(self.cost[best_tuned]), self.draft_name, pct(self.alpha[self.chosen]),
                "nothing" if self.cost[self.chosen] == 0 else pct(self.cost[self.chosen])),
            "measured on %d tokens: %.2f GPT-2 passes per token with the %s, %.2f with the "
            "first design, 1 plain. Wall-clock in numpy: %.1f s plain, %.1f s speculative, "
            "%.2f times %s" % (
                self.total_tokens, self.passes[1], self.draft_name, self.passes[2],
                self.seconds[0], self.seconds[1], ratio if ratio >= 1 else 1 / ratio,
                "faster" if ratio >= 1 else "slower"),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _models(self, sc, op, draft=1.0, big=1.0):
        self.box(sc, *self.DRAFT_AT, 300, 44, AMBER, op * draft, "draft: %s" % self.draft_name,
                 ink=WHITE, size=0.032)
        self.box(sc, *self.GPT_AT, 300, 44, BLUE, op * big, "GPT-2: all 12 layers", ink=WHITE,
                 size=0.032)

    def _prompt(self, sc, op):
        return self.tokens(sc, [short(w, 8) for w in self.prompt], *self.PROMPT_X, self.Y, op,
                           w=62.0, h=46.0, size=0.030)

    def _wire(self, sc, x, box, k, n, ink, op, grow, pulse, down=False):
        """A line between the token at ``x`` and a model box (its ``k``-th of
        ``n`` ports), with a value travelling along it."""
        port = (box[0] + (k - (n - 1) / 2) * 0.034, box[1] + 0.05)
        top = (x, self.Y - 0.052)
        a, b = (port, top) if down else (top, port)
        tip = (a[0] + (b[0] - a[0]) * grow, a[1] + (b[1] - a[1]) * grow)
        self.segment(sc, a, tip, ink, op * 0.75, 2.0)
        self.pulse(sc, a, b, pulse, ink, op)

    def _guess(self, sc, i, ink, op, solid=False):
        self.chip(sc, self.SLOTS[i], self.Y, self.CHIP, 46, ink, op,
                  short(show(self.trace["prop"][0, i])), size=0.032, solid=solid)

    # ------------------------------------------------------------------ 1 draft
    def phase_1(self, sc, u, op):
        t = self.trace
        self._models(sc, op, big=0.3)
        xs = self._prompt(sc, op)
        size = self.size[0]
        q = u / 0.9 * size
        now = min(int(q), size - 1)
        reads = self.READS.get(self.kind, 1 if self.layer == 0 else None)
        for i in range(now + 1):
            v = 1.0 if i < now else seg(q - i, 0.0, 1.0)
            context = xs[1:] + list(self.SLOTS[:i])
            if reads is not None:
                context = context[len(context) - reads:]
            if i == now:
                for k, x in enumerate(context):
                    self._wire(sc, x, self.DRAFT_AT, k, len(context) + 1, AMBER, op,
                               smoothstep(0.0, 0.25, v), seg(v, 0.05, 0.45))
                self._wire(sc, self.SLOTS[i], self.DRAFT_AT, len(context), len(context) + 1,
                           AMBER, op, smoothstep(0.4, 0.6, v), seg(v, 0.42, 0.78), down=True)
            a = op * smoothstep(0.7, 0.85, v)
            self._guess(sc, i, AMBER, a)
            self.text(sc, self.SLOTS[i], self.Y + 0.09, "q %s" % prob(t["q"][0, i]), a,
                      ink=dark(AMBER), size=0.032)
        before = show(self.trace["prop"][0, now - 1]) if now else self.prompt[-1]
        what = {0: "it reads nothing: the same table every time",
                1: "it reads one token, “%s”, and samples what follows it" % before,
                2: "it reads the last two tokens and samples what follows them",
                None: "it reads every token so far and samples the next"}[reads]
        self.text(sc, 0.05, 0.75, "draft %s %d of %d: %s" % (
            "look-up" if self.kind in SD.NGRAMS else "call", now + 1, size, what), op,
            anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.05, 0.81, "q: the draft's probability of the token it picked. The text "
                  "is a passage of Shakespeare; its last %d tokens are shown" % self.SHOWN, op,
                  anchor="lm", size=0.030)

    # ------------------------------------------------------------------ 2 verify
    def phase_2(self, sc, u, op):
        t = self.trace
        self._models(sc, op, draft=0.3)
        xs = self._prompt(sc, op)
        size, kept = self.size[0], self.n_kept[0]
        every = xs + list(self.SLOTS[:size])
        for k, x in enumerate(every):
            self._wire(sc, x, self.GPT_AT, k, len(every), BLUE, op * (1 - smoothstep(0.5, 0.6, u)),
                       smoothstep(0.0, 0.1, u), seg(u, 0.04, 0.2))
        state = t["state"][0]
        judged, replaced = smoothstep(0.5, 0.58, u), smoothstep(0.72, 0.8, u)
        for i in range(size):
            x = self.SLOTS[i]
            if judged < 0.5:
                self._guess(sc, i, AMBER, op)
            elif state[i] == 1:
                self._guess(sc, i, GREEN, op)
            elif state[i] == 2:
                self._guess(sc, i, RED, op)
                self.cross(sc, x + self.frac(self.CHIP / 2)[0] + 0.012, self.Y - 0.065, 7, RED,
                           op * judged)
            else:
                self.box(sc, x, self.Y, self.CHIP, 46, DIM, op * 0.8,
                         short(show(t["prop"][0, i])), size=0.032)
            self.text(sc, x, self.Y + 0.09, prob(t["q"][0, i]), op, ink=dark(AMBER), size=0.032)
            a = op * smoothstep(0.2 + 0.03 * i, 0.28 + 0.03 * i, u)
            self.text(sc, x, self.Y + 0.15, prob(t["p"][0, i]), a, ink=dark(BLUE), size=0.032)
            if state[i] > 0:
                self.text(sc, x, self.Y + 0.21, pct(min(1.0, t["p"][0, i] / t["q"][0, i])),
                          op * smoothstep(0.34, 0.42, u), size=0.032, strong=True)
            verdict = {1: ("kept", dark(GREEN)), 2: ("rejected", dark(RED)),
                       0: ("thrown away", None)}[int(state[i])]
            self.text(sc, x, self.Y + 0.27, verdict[0], op * judged, ink=verdict[1], size=0.032)
        for row, (label, ink, at) in enumerate((("the draft's probability, q", dark(AMBER), 0.0),
                                                ("GPT-2's probability, p", dark(BLUE), 0.2),
                                                ("chance it is kept, p/q", None, 0.34))):
            self.text(sc, self.SLOTS[0] - 0.07, self.Y + 0.09 + 0.06 * row, label,
                      op * smoothstep(at, at + 0.08, u), anchor="rm", ink=ink, size=0.032)
        at = self.SLOTS[min(kept, self.block - 1)]
        self.chip(sc, at, self.Y + 0.36, self.CHIP, 46, BLUE, op * replaced,
                  short(show(t["extra"][0])), size=0.032, solid=True)
        note = "GPT-2's own token takes its place" if kept < size else \
            "all kept: the same pass gives the next token too"
        self.text(sc, at - 0.07, self.Y + 0.36, "%s: one pass, %d token%s" % (
            note, kept + 1, "" if kept == 0 else "s"), op * replaced, anchor="rm", size=0.032,
            strong=True)

    # ------------------------------------------------------------------ 3 the run
    STATE_INK = {1: GREEN, 2: RED}  # noqa: RUF012

    def _round(self, sc, r, x0, y, op, v, h=28.0):
        """Round ``r`` of the run as a row of chips: its guesses, then GPT-2's token."""
        t = self.trace
        judged = v > 0.5
        self.text(sc, x0, y, str(r + 1), op, anchor="lm", size=0.030, strong=True)
        for i in range(self.size[r]):
            x = x0 + 0.066 + 0.078 * i
            word, state = short(show(t["prop"][r, i]), 7), int(t["state"][r, i])
            if not judged:
                self.chip(sc, x, y, 68, h, AMBER, op, word, size=0.029)
            elif state > 0:
                self.chip(sc, x, y, 68, h, self.STATE_INK[state], op, word, size=0.029)
            else:
                self.box(sc, x, y, 68, h, DIM, op * 0.7, word, size=0.029)
        self.chip(sc, x0 + 0.066 + 0.078 * self.size[r], y, 68, h, BLUE,
                  op * smoothstep(0.7, 0.95, v), short(show(t["extra"][r]), 7), size=0.029,
                  solid=True)

    def phase_3(self, sc, u, op):
        t = self.trace
        q = u / 0.86 * self.rounds
        half = (self.rounds + 1) // 2
        step = min(0.092, 0.40 / max(half - 1, 1))
        tall = min(34.0, self.span(0.0, step)[1] - 5.0)
        shown = done = 0
        for r in range(min(int(q) + 1, self.rounds)):
            v = seg(q - r, 0.0, 1.0)
            self._round(sc, r, 0.05 if r < half else 0.53, 0.335 + step * (r % half), op, v,
                        tall)
            if v > 0.7:
                shown = int(t["start"][r]) + self.n_kept[r] + 1
                done = r + 1
        words = _tok().decode(self.output[:shown]).replace("\n", " ↵ ")
        self.lines(sc, 0.05, 0.80, ["… " + " ".join(self.prompt[1:]) + words], op, size=0.030,
                   step=0.047, width=0.8)
        checked = int((t["state"][:done] > 0).sum())
        kept = int((t["state"][:done] == 1).sum())
        self.text(sc, 0.95, 0.935, "passes %d · tokens %d · kept %d of %d guesses" % (
            done, shown, kept, checked), op, anchor="rm", size=0.030, strong=True)
        x = 0.05
        for label, ink in (("kept", GREEN), ("rejected", RED), ("never checked", DIM),
                           ("GPT-2's token", BLUE)):
            self.box(sc, x + 0.008, 0.935, 14, 14, ink, op, radius=3)
            self.text(sc, x + 0.022, 0.935, label, op, anchor="lm", size=0.029)
            x += 0.036 + layout.text_width(label, 0.029) / self.cam.zoom

    # ------------------------------------------------------------------ 4 exact
    def phase_4(self, sc, u, op):
        t = self.trace
        p, q = t["id_p"], t["id_q"]
        scale = self.span(0.29)[0] / float(max(p.max(), q.max()))
        trim, refill = smoothstep(0.30, 0.50, u), smoothstep(0.58, 0.80, u)
        r, i = (int(v) for v in t["id_spot"])
        self.text(sc, 0.05, 0.33, "the token after “… %s”, where the guess “%s” was %s" % (
            " ".join(show(v) for v in t["id_context"]), show(t["prop"][r, i]),
            "rejected" if t["state"][r, i] == 2 else "kept"), op, anchor="lm", size=0.032,
            strong=True)
        overlap = float(t["overlap"])
        for row in range(len(p)):
            y = 0.41 + row * 0.068
            both = min(p[row], q[row])
            at = self.P(0.19, y)
            self.text(sc, 0.18, y, short(show(t["id_tok"][row]), 10), op, anchor="rm", size=0.034)
            grow = smoothstep(0.02 + 0.02 * row, 0.2 + 0.02 * row, u)
            # the draft's proposal, the part of it that is kept, GPT-2's refill
            w = max(q[row] * scale * grow, 0.1)
            self.rect(sc, self.shift(at, w / 2, 0), w, 26, AMBER, op * (1 - 0.72 * trim),
                      radius=min(4.0, w / 2))
            w = max(both * scale, 0.1)
            self.rect(sc, self.shift(at, w / 2, 0), w, 26, GREEN, op * trim * grow,
                      radius=min(4.0, w / 2), lift=0.7)
            if p[row] > q[row]:
                extra = (p[row] - q[row]) * scale * refill
                if extra > 0.3:
                    self.rect(sc, self.shift(at, both * scale + extra / 2, 0), extra, 26, BLUE,
                              op, radius=min(4.0, extra / 2), lift=0.8)
            mark = self.shift(at, p[row] * scale, 0)
            self.polyline(sc, [self.shift(mark, 0, -17), self.shift(mark, 0, 17)], 2.6,
                          dark(BLUE, 0.3), op, bias=0.9)
            self.label(sc, self.shift(at, max(p[row], q[row]) * scale + 10, 0),
                       "p %s" % prob(p[row]), 0.030, anchor="lm", opacity=op, ink=dark(BLUE))
        rows = [("amber: what the draft proposes, q", dark(AMBER), 0.02),
                ("blue line: what GPT-2 wants, p", dark(BLUE), 0.02),
                ("green: the proposals that are kept. Each is kept with chance p/q, which "
                 "leaves the smaller of p and q: %s in all" % pct(overlap), dark(GREEN), 0.30),
                ("blue: the other %s is redrawn only where p is above q, filling each bar up "
                 "to the line" % pct(1 - overlap), dark(BLUE), 0.58),
                ("so every token ends with exactly GPT-2's probability", None, 0.82)]
        y = 0.40
        for s, ink, at in rows:
            y = self.lines(sc, 0.60, y, [(s, ink)], op * smoothstep(at, at + 0.08, u),
                           size=0.030, step=0.04, width=0.31) + 0.014
        self.text(sc, 0.05, 0.925, "%s simulated tokens: distance %.3f from GPT-2's "
                  "distribution (as many drawn from it directly: %.3f)" % (
                      f"{SD.MC_DRAWS:,}", self.mc[0], self.mc[1]),
                  op * smoothstep(0.85, 0.95, u), anchor="lm", size=0.030)

    # ------------------------------------------------------------------ 5 which draft
    def phase_5(self, sc, u, op):
        d = self.d
        layers = np.asarray(d["layers"], float)
        n = len(layers)
        own, tuned = self.alpha[:n], self.alpha[n:2 * n]
        self.text(sc, 0.05, 0.335, "how often a draft agrees with GPT-2, on %s held-out "
                  "positions" % "{:,}".format(int(d["positions"])), op, anchor="lm", size=0.032,
                  strong=True)
        chart = self.chart(((0.10, 0.43), (0.42, 0.80)), layers, [0.0, max(0.8, tuned.max())],
                           pad=0.0)
        self.axes(sc, chart, op, xlabel="layers of GPT-2 the draft runs",
                  ylabel="probability shared",
                  xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8, 10)],
                  yticks=[(v, pct(v) if v else "0") for v in (0.0, 0.2, 0.4, 0.6, 0.8)])
        g1, g2 = seg(u, 0.03, 0.25), seg(u, 0.25, 0.47)
        self.plot(sc, chart, layers, own, VIOLET, op, g1, width=2.6)
        self.plot(sc, chart, layers, tuned, TEAL, op * (g2 > 0), g2, width=2.6)
        self.text(sc, chart.fx(6.5), chart.fy(float(np.interp(6.5, layers, own))) + 0.05,
                  "GPT-2's own read-out", op * smoothstep(0.2, 0.28, u), size=0.030,
                  ink=dark(VIOLET))
        self.text(sc, chart.fx(10.0), chart.fy(tuned[-1]) - 0.04, "trained read-out",
                  op * smoothstep(0.42, 0.5, u), anchor="rm", size=0.030, ink=dark(TEAL))
        grams = self.alpha[2 * n:]
        best = int(np.argmax(grams))
        a = op * smoothstep(0.5, 0.58, u)
        self.segment(sc, (chart.fx(0.0), chart.fy(grams[best])),
                     (chart.fx(10.0), chart.fy(grams[best])), AMBER, a, 2.4)
        self.text(sc, chart.fx(3.4), chart.fy(grams[best]) + 0.032, "%s table (no layers)" % (
            SD.NGRAMS[best]), a, size=0.030, ink=dark(AMBER))
        # the short list: agreement, cost, what the rule predicts
        b = op * smoothstep(0.55, 0.65, u)
        for x, head in ((0.80, "agrees"), (0.87, "cost"), (0.95, "speed")):
            self.text(sc, x, 0.42, head, b, anchor="rm", size=0.030)
        for r, i in enumerate(self.shortlist):
            y = 0.49 + r * 0.068
            c = op * smoothstep(0.58 + 0.05 * r, 0.66 + 0.05 * r, u)
            picked = i == self.chosen
            if picked:
                self.box(sc, 0.725, y, self.span(0.47)[0], 30, soft(GREEN, 0.6), c, radius=5)
            self.text(sc, 0.50, y, self.names[i], c, anchor="lm", size=0.030, strong=picked)
            self.text(sc, 0.80, y, pct(self.alpha[i]), c, anchor="rm", size=0.030)
            self.text(sc, 0.87, y, pct(self.cost[i]) if self.cost[i] else "0", c, anchor="rm",
                      size=0.030)
            self.text(sc, 0.95, y, "%.2f×" % self.speed[i], c, anchor="rm", size=0.032,
                      ink=dark(GREEN if self.speed[i] > 1 else RED, 0.3))
        self.lines(sc, 0.50, 0.49 + len(self.shortlist) * 0.068 + 0.012, [
            "cost: the draft's multiply-adds per token over GPT-2's. Speed: the predicted "
            "speed-up, tokens per GPT-2 pass ÷ (1 + guesses × cost)"],
            op * smoothstep(0.85, 0.95, u), size=0.029, step=0.04, width=0.40)

    # ------------------------------------------------------------------ 6 measured
    def phase_6(self, sc, u, op):
        self.text(sc, 0.05, 0.335, "measured: %d tokens = %d passages × %d seeds × %d tokens, "
                  "sampled from GPT-2" % (self.total_tokens, self.n_prompts, self.n_seeds,
                                          self.new), op, anchor="lm", size=0.032, strong=True)
        full = self.span(0.20)[0]
        x0 = 0.38
        self.text(sc, x0, 0.405, "GPT-2 passes per token", op, anchor="lm", size=0.030,
                  strong=True)
        rows = (("plain decoding", BLUE, ""),
                ("draft: %s" % self.draft_name, GREEN, "%s of guesses kept" % pct(
                    self.kept_share[0])),
                ("first design: %s" % self.control_name, AMBER, "%s kept" % pct(
                    self.kept_share[1])))
        for r, (label, ink, note) in enumerate(rows):
            y = 0.47 + r * 0.068
            g = smoothstep(0.04 + 0.1 * r, 0.24 + 0.1 * r, u)
            self.meter(sc, x0, y, full * self.passes[r] * g, ink, op, label,
                       "%.2f" % self.passes[r], thick=26.0, size=0.032, track=full,
                       value_at=full)
            if r:
                self.text(sc, x0 + 0.275, y, "seeds %.2f–%.2f · %s" % (
                    self.per_seed[r - 1].min(), self.per_seed[r - 1].max(), note),
                    op * smoothstep(0.24 + 0.1 * r, 0.32 + 0.1 * r, u), anchor="lm", size=0.030)
        b = op * smoothstep(0.5, 0.62, u)
        self.text(sc, x0, 0.70, "wall-clock, numpy on a CPU (hardware decides this; not the "
                  "passes)", b, anchor="lm", size=0.030, strong=True)
        most = max(self.seconds)
        for r, (label, ink, _) in enumerate(rows):
            y = 0.765 + r * 0.068
            g = smoothstep(0.55 + 0.08 * r, 0.75 + 0.08 * r, u)
            self.meter(sc, x0, y, full * self.seconds[r] / most * g, ink, b,
                       label.split(":")[0], "%.1f s" % self.seconds[r], thick=26.0, size=0.030,
                       track=full, value_at=full)


# ---------------------------------------------------------------------- 31 repetition
class RepetitionAndPenalties(Decoder):
    """Greedy GPT-2 caught in a three-word loop, then the three rules that
    reshape its next-token scores - each drawn on the prompt it acts on, with
    the arithmetic on the looping token and the distribution before and
    after - what GPT-2 writes under each, and how often each rule breaks the
    loop over held-out passages."""

    TITLE = "Repetition and penalties"
    PHASES = (("the loop", 10.0), ("repetition", 9.0), ("frequency", 9.0), ("trigram", 9.0),
              ("rollouts", 10.0), ("%d passages" % SD.PROMPTS, 11.0))
    WORD_INK = (RED, BLUE, GREEN)                 # the three words of the loop
    RULE_INK = {"repetition": AMBER, "frequency": VIOLET, "trigram": TEAL}  # noqa: RUF012
    PITCH = 0.062

    def setup(self):
        s = self.study
        self.trace = t = SD.penalty_data()
        self.ids = _tok().encode(PENALTY_PROMPT)
        self.prompt = [show(i) for i in self.ids]
        self.ink = {tok: self.WORD_INK[k % 3] for k, tok in enumerate(dict.fromkeys(self.ids))}
        self.labels = list(s.panels[0].data["labels"])
        self.probs = [np.asarray(row, float) for row in s.panels[0].data["values"]]  # per rule
        self.repeat = float(s.metadata["settings"]["repeat_penalty"])
        self.frequency = float(s.metadata["settings"]["frequency_penalty"])
        self.banned = int(s.panels[0].notes[0].split(":")[1])
        # the same four rules on held-out passages: rules x passages
        self.loops = np.asarray(s.panels[1].data["each"]) > 0.5
        self.distinct = np.asarray(s.panels[2].data["values"], float)
        self.own = np.asarray(s.panels[3].data["values"], float)
        self.n_passages = self.loops.shape[1]
        self.new_tokens = int(s.metadata["settings"]["new_tokens"])
        self.logit = np.asarray(t["logit"], float)
        self.count = int(np.sum(np.asarray(self.ids) == int(t["top"][0])))
        self.roll = [[show(i) for i in row] for row in t["roll"]]

    def subtitle_for(self, k):
        t = self.trace
        p, top = self.probs, self.labels[0]
        rounds = [float(v) for v in t["roll_p"][0][::3]]
        after = ["“%s” leads at %.2f" % (self.labels[int(np.argmax(v))], v.max()) for v in p]
        still = "and is still the likeliest" if int(np.argmax(p[2])) == 0 else \
            "and “%s” leads at %.2f" % (self.labels[int(np.argmax(p[2]))], p[2].max())
        loops = sum(a == b for a, b in zip(self.roll[0], (self.prompt * 2)[:len(self.roll[0])]))
        return [
            "GPT-2 continues “%s” by taking its likeliest token each time. It writes the same "
            "three words again, surer every round: %s at %s" % (
                PENALTY_PROMPT, top, ", then ".join("%.2f" % v for v in rounds)),
            "repetition penalty %.1f: the score (logit) of every token already used is made "
            "%.1f times worse. GPT-2's logits here are all negative, so %s falls from %.2f to "
            "%.2f; %s" % (self.repeat, self.repeat, top, p[0][0], p[1][0], after[1]),
            "frequency penalty %.1f: a token's logit loses %.1f for each time it was used. "
            "%s was used %d times: it falls from %.2f to %.2f %s" % (
                self.frequency, self.frequency, top, self.count, p[0][0], p[2][0], still),
            "no repeated trigram: “%s” was followed by %s before, so %s is forbidden outright "
            "(%d token banned). The others share its probability: %s" % (
                " ".join(self.prompt[-2:]), top, top, self.banned, after[3]),
            "eight more tokens under each rule. With no rule, %d of 8 repeat the loop; with the "
            "frequency penalty GPT-2 still begins “%s”; the other two rules begin “%s”" % (
                loops, self.roll[2][0], self.roll[1][0]),
            "one prompt is not a result. After %d passages of Shakespeare, %d greedy tokens "
            "each: with no rule %d continuations repeat themselves; frequency penalty %d, "
            "repetition penalty %d. Every rule costs likelihood"
            % (self.n_passages, self.new_tokens, self.loops[0].sum(), self.loops[2].sum(),
               self.loops[1].sum()),
        ][k]

    def _x(self, i):
        return 0.05 + self.PITCH * (i + 0.5)

    # ------------------------------------------------------------------ 1 the loop
    def phase_1(self, sc, u, op):
        t = self.trace
        y0, y1, base = 0.35, 0.53, 0.88
        for i, (tok, word) in enumerate(zip(self.ids, self.prompt)):
            self.chip(sc, self._x(i), y0, 48, 42, self.ink[tok], op, word, size=0.032)
        self.text(sc, self._x(9) + 0.01, y0, "the prompt", op, anchor="lm", size=0.034, strong=True)
        n = len(t["roll"][0])
        q = u / 0.88 * n
        for i in range(min(int(q) + 1, n)):
            v = seg(q - i, 0.0, 1.0)
            tok = int(t["roll"][0][i])
            ink = self.ink.get(tok, AXIS)
            x = self._x(i)
            if i >= 3:
                a, b = (self._x(i - 3), y1 - 0.05), (x, y1 - 0.05)
                self.arc(sc, a, b, ink, op * 0.8, smoothstep(0.0, 0.5, v), 0.085, 2.2)
                self.pulse(sc, a, b, seg(v, 0.1, 0.6), ink, op, 0.085)
            else:
                a, b = (self._x(i + 6), y0 + 0.05), (x + 0.012, y1 - 0.05)
                g = smoothstep(0.0, 0.5, v)
                self.segment(sc, a, (a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g), ink,
                             op * 0.6, 2.0)
                self.pulse(sc, a, b, seg(v, 0.1, 0.6), ink, op)
            a = op * smoothstep(0.5, 0.7, v)
            self.chip(sc, x, y1, 48, 42, ink, a, self.roll[0][i], size=0.032, solid=True)
            p = float(t["roll_p"][0][i])
            h = self.span(0, 0.26)[1] * p * smoothstep(0.55, 0.9, v)
            self.rect(sc, self.shift(self.P(x, base), 0, -h / 2), 34, max(h, 0.1), ink, a,
                      radius=3)
            self.label(sc, self.shift(self.P(x, base), 0, -h - 12), "%.2f" % p, 0.030, opacity=a,
                       ink=layout.STRONG_INK)
        self.text(sc, self._x(8) + 0.01, y1, "GPT-2 continues: the likeliest token, every time",
                  op, anchor="lm", size=0.032, strong=True)
        self.lines(sc, self._x(8) + 0.01, 0.70, ["under each token: the probability",
                                                 "GPT-2 gave it. The loop feeds itself:",
                                                 "each repeat makes the next surer"],
                   op, size=0.030, step=0.05)

    # ------------------------------------------------------------------ 2-4 one rule
    def _bars(self, sc, u, op, r, ink):
        before, after = self.probs[0], self.probs[r]
        full = self.span(0.20)[0] / float(before.max())
        move = smoothstep(0.42, 0.72, u)
        self.text(sc, 0.635, 0.415, "probability of the next token", op, anchor="lm", strong=True,
                  size=0.030)
        for i, word in enumerate(self.labels):
            y = 0.475 + i * 0.056
            now = before[i] + (after[i] - before[i]) * move
            self.meter(sc, 0.70, y, full * now, ink if move > 0 else AXIS, op, word,
                       "%.2f" % now, thick=22.0, size=0.032, ghost=full * before[i])
        self.text(sc, 0.70, 0.925, "outline: before the rule", op, anchor="lm", size=0.030)

    def _rule(self, sc, u, op, r):
        rule = RULES[r]
        ink = self.RULE_INK[rule]
        top = int(self.trace["top"][0])
        seen = smoothstep(0.05, 0.2, u)
        last2 = len(self.ids) - 2
        for i, (tok, word) in enumerate(zip(self.ids, self.prompt)):
            x = 0.05 + 0.056 * (i + 0.5)
            mark = None
            if rule == "repetition" or (rule == "frequency" and tok == top):
                mark = ink
            elif rule == "trigram":
                follows = i >= 2 and tuple(self.ids[i - 2:i]) == tuple(self.ids[-2:])
                mark = RED if follows else ink if (i >= last2 or tuple(
                    self.ids[i - (self.ids[i] == self.ids[-1]):][:2]) == tuple(self.ids[-2:])) \
                    else None
            if mark is not None and seen > 0.5:
                self.chip(sc, x, 0.335, 44, 40, mark, op, word, size=0.030,
                          solid=mark is RED or (rule == "trigram" and i >= last2))
            else:
                self.chip(sc, x, 0.335, 44, 40, None, op, word, size=0.030)
        name = {"repetition": "repetition penalty %.1f" % self.repeat,
                "frequency": "frequency penalty %.1f" % self.frequency,
                "trigram": "no repeated trigram"}[rule]
        self.text(sc, 0.58, 0.335, name, op, anchor="lm", size=0.038, ink=dark(ink))
        a, b = self.logit[0], self.logit[1]
        word, other = self.labels[0], self.labels[1]
        if rule == "repetition":
            rows = ["every token already used (amber):", "its logit is multiplied by %.1f" % (
                self.repeat), ("%s:  %.1f × %.1f = %.1f" % (word, a, self.repeat, a * self.repeat),
                               dark(ink)), "%s (not used):  %.1f, unchanged" % (other, b)]
        elif rule == "frequency":
            rows = ["%s was used %d times (violet):" % (word, self.count),
                    "its logit loses %.1f for each" % self.frequency,
                    ("%s:  %.1f − %.1f × %d = %.1f" % (word, a, self.frequency, self.count,
                                                       a - self.frequency * self.count), dark(ink)),
                    "%s (not used):  %.1f, unchanged" % (other, b)]
        else:
            rows = ["the last two tokens: %s" % " ".join(self.prompt[-2:]),
                    "earlier, they were followed by %s (red)" % word,
                    ("%s would repeat three in a row:" % word, dark(ink)),
                    ("its logit is set to −∞", dark(ink))]
        y = 0.47
        for k, row in enumerate(rows):
            y = self.lines(sc, 0.05, y, [row], op * smoothstep(0.1 + 0.08 * k, 0.2 + 0.08 * k, u),
                           size=0.034 if k < 2 or rule == "trigram" else 0.036, step=0.062)
        if rule == "frequency" and self.prompt.count(word) != self.count:
            self.text(sc, 0.05, 0.725, "(the opening %s has no space before it: to GPT-2 that "
                      "is another token)" % word, op * smoothstep(0.42, 0.52, u), anchor="lm")
        after = self.probs[r]
        self.text(sc, 0.05, 0.80, "%s: %.2f → %.2f" % (word, self.probs[0][0], after[0]),
                  op * smoothstep(0.55, 0.7, u), anchor="lm", size=0.040, strong=True)
        lead = int(np.argmax(after))
        self.text(sc, 0.05, 0.87, "the likeliest token is now %s, %.2f" % (
            self.labels[lead], after[lead]) if lead else "%s is still the likeliest token" % word,
            op * smoothstep(0.7, 0.85, u), anchor="lm", size=0.032)
        self._bars(sc, u, op, r, ink)

    def phase_2(self, sc, u, op):
        self._rule(sc, u, op, 1)

    def phase_3(self, sc, u, op):
        self._rule(sc, u, op, 2)

    def phase_4(self, sc, u, op):
        self._rule(sc, u, op, 3)

    # ------------------------------------------------------------------ 5 rollouts
    def phase_5(self, sc, u, op):
        t = self.trace
        self.text(sc, 0.05, 0.33, "“%s” …" % PENALTY_PROMPT, op, anchor="lm", size=0.036,
                  strong=True)
        names = ("no rule", "repetition %.1f" % self.repeat, "frequency %.1f" % self.frequency,
                 "no trigram")
        n = len(self.roll[0])
        shown = min(int(u / 0.8 * n + 1e-9), n)
        for r, name in enumerate(names):
            y = 0.44 + r * 0.118
            ink = self.RULE_INK.get(RULES[r])
            self.text(sc, 0.05, y, name, op, anchor="lm", size=0.032,
                      ink=None if ink is None else dark(ink), strong=ink is None)
            for i in range(shown):
                tok = int(t["roll"][r][i])
                loop = tok in self.ink
                self.chip(sc, 0.27 + 0.082 * i, y, 66, 42, self.ink[tok] if loop else ink, op,
                          self.roll[r][i], size=0.030, solid=loop)
        self.text(sc, 0.05, 0.915, "filled: a word of the loop again", op, anchor="lm",
                  size=0.030)

    # ------------------------------------------------------------------ 6 held-out passages
    def phase_6(self, sc, u, op):
        n = self.n_passages
        self.text(sc, 0.05, 0.335, "%d passages of Shakespeare, %d greedy tokens after each: "
                  "does the continuation repeat itself?" % (n, self.new_tokens), op, anchor="lm",
                  size=0.032, strong=True)
        for x, head in ((0.30, "one square a passage"), (0.665, "loop"),
                        (0.775, "distinct tokens"), (0.93, "log p")):
            self.text(sc, x, 0.405, head, op, anchor="lm" if x < 0.6 else "mm", size=0.030)
        names = ("no rule", "repetition %.1f" % self.repeat, "frequency %.1f" % self.frequency,
                 "no trigram")
        for r, name in enumerate(names):
            y = 0.48 + r * 0.095
            ink = self.RULE_INK.get(RULES[r])
            a = op * smoothstep(0.04 + 0.16 * r, 0.12 + 0.16 * r, u)
            self.text(sc, 0.05, y, name, a, anchor="lm", size=0.032,
                      ink=None if ink is None else dark(ink), strong=ink is None)
            self.tally(sc, 0.31, y, self.loops[r], RED, a, seg(u, 0.06 + 0.16 * r, 0.2 + 0.16 * r))
            b = op * smoothstep(0.2 + 0.16 * r, 0.26 + 0.16 * r, u)
            self.text(sc, 0.665, y, "%d of %d" % (self.loops[r].sum(), n), b, size=0.034,
                      strong=True)
            self.text(sc, 0.775, y, pct(self.distinct[r]), b, size=0.032)
            self.text(sc, 0.93, y, "%.2f" % self.own[r], b, size=0.032)
        self.lines(sc, 0.05, 0.865, [
            "red: the continuation completes a three-token run already written (just what the "
            "trigram rule forbids)",
            "log p: the mean log-probability GPT-2 gave the tokens it wrote; lower is further "
            "from what it wanted"], op * smoothstep(0.7, 0.85, u), size=0.030, step=0.05)


# ---------------------------------------------------------------------- 32 constrained decoding
def plain(token):
    """A token's text for a label, with the unprintable ones named."""
    s = _tok().decode([int(token)])
    if "�" in s:
        return "(broken character)"
    return s.strip() if s.strip() else "(blank)"


class ConstrainedDecoding(Decoder):
    """A five-step grammar as a mask over GPT-2's vocabulary: the mask itself,
    then one step in full (what GPT-2 wanted, what the mask removes, the
    survivors rescaled), then all five steps, then how much probability the
    grammar left standing - on the worked prompt, and after held-out passages."""

    TITLE = "Constrained decoding"
    PHASES = (("the grammar", 10.0), ("one step", 14.0), ("five steps", 13.0), ("the cost", 10.0),
              ("%d passages" % SD.PROMPTS, 11.0))
    COLUMNS = [str(d) for d in range(10)] + [","]

    def setup(self):
        s = self.study
        self.trace = t = SD.grammar_data()
        self.mask = np.asarray(s.panels[0].data["values"], float)            # steps x digits
        self.mass = np.asarray(s.panels[1].data["values"], float)
        self.emitted = list(s.panels[3].data["labels"])
        self.output = s.panels[3].notes[0].split(": ")[1]
        self.steps = len(GRAMMAR)
        # each step's row of the mask over (0..9, comma), and the rescaled probabilities
        self.allowed = np.c_[self.mask, 1.0 - self.mask[:, :1]]
        raw = np.c_[t["digit_p"], t["comma_p"]]
        self.rescaled = raw * self.allowed / (raw * self.allowed).sum(1, keepdims=True)
        self.pick = [int(np.argmax(row)) for row in self.rescaled]
        self.free = [plain(i) for i in t["free"]]
        # the same grammar after held-out passages: alone, and followed by the instruction
        across = s.panels[2].data
        self.across = (np.asarray(across["alone"], float), np.asarray(across["values"], float))
        self.valid = int(np.sum(across["valid"])), int(np.sum(across["free_valid"]))
        self.runs = int(np.size(across["valid"]))

    def subtitle_for(self, k):
        digits = sum(w.isdigit() for w in self.free)
        return [
            "a grammar for “two digits, a comma, two digits”: five steps, each with a list of "
            "the tokens allowed next. Of GPT-2's %s tokens, %d are single digits and one is "
            "the comma" % (f"{gpt2.N_VOCAB:,}", int(self.mask[0].sum())),
            "step 1, a digit is due. None of GPT-2's five favourites is one, so they are "
            "removed: %s of the probability survives. Divided by that, the survivors become "
            "the whole choice, and %s wins" % (pct(self.mass[0]), self.emitted[0]),
            "every step does the same: mask, rescale, take the likeliest token still allowed. "
            "The output is %s; it could only ever have been two digits, a comma, two digits"
            % self.output,
            "the grammar let through %s of GPT-2's probability. Left alone, GPT-2 writes %s in "
            "five tokens. The grammar fixes the form of the answer, not its sense" % (
                ", ".join(pct(m) for m in self.mass), "%d digits" % digits if digits else
                "no digit"),
            "the same grammar after %d Shakespeare passages, with and without the "
            "instruction: GPT-2 alone wrote the form in %d of %d runs, masked in all %d. "
            "Median kept: %s at the first digit, %s at the comma" % (
                self.across[0].shape[0], self.valid[1], self.runs, self.valid[0],
                pct(np.median(self.across[1][:, 0])), pct(np.median(self.across[1][:, 2]))),
        ][k]

    # ------------------------------------------------------------------ 1 the grammar
    def _states(self, sc, y, op, now=None, done=0):
        xs = [0.17 + 0.15 * k for k in range(self.steps)]
        for k, (x, kind) in enumerate(zip(xs, GRAMMAR)):
            if k:
                self.arrow(sc, (xs[k - 1] + 0.058, y), (x - 0.058, y), AXIS, op, width=2.0)
            self.chip(sc, x, y, 92, 42, GREEN if k == now or k < done else None, op, kind,
                      size=0.034, solid=k == now)
        return xs

    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.33, "“%s”" % GRAMMAR_PROMPT.strip(), op, anchor="lm", size=0.034,
                  strong=True)
        now = min(int(u / 0.85 * self.steps), self.steps - 1)
        self._states(sc, 0.425, op, now)
        values = np.where(self.allowed > 0, 1.0, -0.22)
        cell = self.grid(sc, values, 0.45, 0.75, 429, 180, op, reveal=(now + 1) / self.steps,
                         vmax=1.0, hot=GREEN, cold=RED)
        for c, name in enumerate(self.COLUMNS):
            self.text(sc, cell(0, c)[0], 0.525, name, op, size=0.034, strong=True)
        for k, kind in enumerate(GRAMMAR):  # noqa: B007
            if k <= now:
                n = int(self.allowed[k].sum())
                self.text(sc, 0.195, cell(k, 0)[1], "step %d" % (k + 1), op, anchor="rm",
                          size=0.030)
                self.text(sc, 0.715, cell(k, 0)[1], "%d of %s allowed" % (
                    n, f"{gpt2.N_VOCAB:,}"), op, anchor="lm", size=0.030)
        self.text(sc, 0.715, 0.525, "green: allowed", op, anchor="lm", size=0.030,
                  ink=dark(GREEN))

    # ------------------------------------------------------------------ 2 one step
    def phase_2(self, sc, u, op):
        t = self.trace
        masked, rescale, emit = smoothstep(0.25, 0.4, u), smoothstep(0.52, 0.78, u), \
            smoothstep(0.85, 0.93, u)
        self.text(sc, 0.05, 0.33, "step 1: the grammar allows only a digit", op, anchor="lm",
                  size=0.036, strong=True)
        self.text(sc, 0.05, 0.405, "GPT-2's five likeliest tokens", op, anchor="lm", size=0.030,
                  strong=True)
        full = self.span(0.17)[0] / float(t["want_p"][0][0])
        for i in range(5):
            y = 0.465 + i * 0.058
            grow = smoothstep(0.02 + 0.03 * i, 0.16 + 0.03 * i, u)
            self.meter(sc, 0.25, y, full * t["want_p"][0][i] * grow, BLUE if masked < 0.5 else DIM,
                       op, plain(t["want_tok"][0][i]), pct(t["want_p"][0][i]), thick=22.0,
                       size=0.030)
            self.cross(sc, 0.25 + self.frac(full * t["want_p"][0][i])[0] + 0.085, y, 7, RED,
                       op * masked)
        self.text(sc, 0.05, 0.79, "not digits: removed", op * masked, anchor="lm", size=0.034,
                  ink=dark(RED))
        self.text(sc, 0.05, 0.85, "all the grammar keeps: %s" % pct(self.mass[0]), op * masked,
                  anchor="lm", size=0.034, strong=True)
        raw = t["digit_p"][0]
        new = self.rescaled[0][:10]
        scale = self.span(0.20)[0] / float(new.max())
        self.text(sc, 0.60, 0.405, "the ten digits, %s" % (
            "divided by %s" % number(float("%.2g" % self.mass[0])) if rescale > 0 else
            "as GPT-2 scored them"), op * masked, anchor="lm", size=0.030, strong=True)
        for d in range(10):
            y = 0.455 + d * 0.047
            value = raw[d] + (new[d] - raw[d]) * rescale
            best = d == self.pick[0] and emit > 0.5
            self.meter(sc, 0.63, y, scale * value, GREEN, op * masked, str(d), pct(value),
                       thick=18.0, size=0.030, label_ink=dark(GREEN) if best else None)
        self.chip(sc, 0.90, 0.33, 44, 44, GREEN, op * emit, self.emitted[0], size=0.038,
                  solid=True)
        self.text(sc, 0.865, 0.33, "written:", op * emit, anchor="rm", size=0.032)

    # ------------------------------------------------------------------ 3 five steps
    def phase_3(self, sc, u, op):
        t = self.trace
        q = u / 0.85 * self.steps
        now = min(int(q), self.steps - 1)
        v = seg(q - now, 0.0, 1.0)
        done = now + (v > 0.6)
        values = np.where(self.allowed > 0, np.maximum(self.rescaled, 0.02), -0.22)
        cell = self.grid(sc, values, 0.43, 0.61, 418, 220, op, reveal=(now + 1) / self.steps,
                         vmax=1.0, hot=GREEN, cold=RED,
                         ring=[(k, self.pick[k]) for k in range(done)], ring_ink=INK_DARK)
        for c, name in enumerate(self.COLUMNS):
            self.text(sc, cell(0, c)[0], 0.35, name, op, size=0.034, strong=True)
        self.text(sc, 0.745, 0.35, "kept", op, size=0.030, strong=True)
        self.text(sc, 0.885, 0.35, "GPT-2's own pick", op, size=0.030, strong=True)
        for k, kind in enumerate(GRAMMAR):
            if k > now:
                continue
            y = cell(k, 0)[1]
            self.text(sc, 0.175, y, "%d %s" % (k + 1, kind), op, anchor="rm", size=0.032)
            self.text(sc, 0.745, y, pct(self.mass[k]), op, size=0.032, strong=True)
            self.text(sc, 0.885, y, plain(t["want_tok"][k][0]), op, size=0.032, ink=dark(BLUE))
        self.text(sc, 0.175, 0.905, "written:", op, anchor="rm", size=0.032)
        for k in range(done):
            self.chip(sc, 0.21 + 0.055 * k, 0.905, 40, 40, GREEN, op, self.emitted[k], size=0.036,
                      solid=True)
        self.text(sc, 0.52, 0.905, "each row: the allowed tokens' share after rescaling", op,
                  anchor="lm", size=0.030)

    # ------------------------------------------------------------------ 4 the cost
    def phase_4(self, sc, u, op):
        self.text(sc, 0.05, 0.34, "how much of GPT-2's probability the grammar kept", op,
                  anchor="lm", size=0.032, strong=True)
        full = self.span(0.30)[0]
        for k, kind in enumerate(GRAMMAR):
            y = 0.425 + k * 0.072
            g = smoothstep(0.04 + 0.08 * k, 0.24 + 0.08 * k, u)
            self.meter(sc, 0.17, y, full * self.mass[k] * g, GREEN, op, "%d %s" % (k + 1, kind),
                       pct(self.mass[k]), thick=26.0, size=0.032, track=full, value_at=full)
        a = op * smoothstep(0.45, 0.6, u)
        self.text(sc, 0.63, 0.425, "with the grammar", a, anchor="lm", size=0.032,
                  ink=dark(GREEN))
        for k, word in enumerate(self.emitted):
            self.chip(sc, 0.652 + 0.052 * k, 0.50, 40, 42, GREEN, a, word, size=0.036, solid=True)
        b = op * smoothstep(0.6, 0.75, u)
        self.text(sc, 0.63, 0.60, "GPT-2 left alone", b, anchor="lm", size=0.032, ink=dark(BLUE))
        x = 0.63
        for word in self.free:
            w = max(layout.text_width(word, 0.030) * 960 + 16, 40)
            dx = self.frac(w + 5)[0]
            self.chip(sc, x + dx / 2, 0.675, w, 42, BLUE, b, word, size=0.030)
            x += dx
        self.text(sc, 0.05, 0.83, "the mask guarantees the form, not that the digits are right",
                  op * smoothstep(0.75, 0.9, u), anchor="lm", size=0.034, strong=True)

    # ------------------------------------------------------------------ 5 held-out passages
    def phase_5(self, sc, u, op):
        n = self.across[0].shape[0]
        self.text(sc, 0.05, 0.335, "the same grammar after %d passages of Shakespeare: how "
                  "much probability it keeps" % n, op, anchor="lm", size=0.032, strong=True)
        for k, (cx, name, values) in enumerate((
                (0.20, "the passage alone", self.across[0]),
                (0.50, "passage, then the instruction", self.across[1]))):
            a = op * smoothstep(0.02 + 0.25 * k, 0.12 + 0.25 * k, u)
            self.text(sc, cx, 0.40, name, a, size=0.030, strong=True)
            cell = self.grid(sc, values, cx, 0.69, 190, 192, a, vmax=1.0, hot=GREEN, cold=RED,
                             reveal=seg(u, 0.05 + 0.25 * k, 0.3 + 0.25 * k))
            for c, kind in enumerate(GRAMMAR):
                self.text(sc, cell(0, c)[0], 0.47, "," if kind == "comma" else "digit", a,
                          size=0.029)
        self.text(sc, 0.05, 0.925, "a row: one passage. A column: one step of the grammar. "
                  "Greener: more of GPT-2's probability was allowed", op, anchor="lm", size=0.030)
        b = op * smoothstep(0.55, 0.68, u)
        y = self.lines(sc, 0.675, 0.46, ["median kept, after the instruction"], b, size=0.030)
        steps = ["digit", "digit", "comma", "digit", "digit"]
        for c, kind in enumerate(steps):
            self.text(sc, 0.69, y + 0.046 * c, "%d %s" % (c + 1, kind), b, anchor="lm",
                      size=0.030)
            self.text(sc, 0.90, y + 0.046 * c, pct(np.median(self.across[1][:, c])), b,
                      anchor="rm", size=0.032, strong=True)
        c = op * smoothstep(0.72, 0.85, u)
        y += 0.046 * 5 + 0.03
        self.text(sc, 0.675, y, "the form, in %d runs" % self.runs, c, anchor="lm", size=0.030)
        self.text(sc, 0.675, y + 0.052, "with the mask: %d" % self.valid[0], c, anchor="lm",
                  size=0.034, ink=dark(GREEN))
        self.text(sc, 0.675, y + 0.104, "GPT-2 alone: %d" % self.valid[1], c, anchor="lm",
                  size=0.034, ink=dark(BLUE))


# ---------------------------------------------------------------------- 33 contrastive decoding
class ContrastiveDecoding(Decoder):
    """The same prompt read by all twelve layers of GPT-2 and by its first
    four, their two distributions, the subtraction token by token, the
    plausibility filter that keeps the subtraction honest, the word it
    picks, and how its picks fare over held-out passages."""

    TITLE = "Contrastive decoding"
    PHASES = (("two models", 11.0), ("subtract", 13.0), ("the filter", 12.0), ("the pick", 10.0),
              ("%d passages" % SD.PROMPTS, 12.0))
    EXPERT, AMATEUR = BLUE, AMBER

    def setup(self):
        s = self.study
        self.trace = t = SD.contrast_data()
        self.prompt = [show(i) for i in _tok().encode(CONTRAST_PROMPT)]
        self.labels = list(s.panels[0].data["labels"])
        self.ep, self.ap, self.score = (np.asarray(p.data["values"], float)
                                        for p in s.panels[:3])
        self.contrast = np.asarray(s.panels[2].data["contrast"], float)
        self.layers = int(s.metadata["settings"]["amateur_layers"])
        self.alpha = float(s.metadata["settings"]["alpha"])
        self.kept = int(s.panels[2].notes[0].split(":")[1])
        self.selected = s.panels[2].notes[1].split(": ")[1]
        self.across = dict(s.metadata["results"])          # over the held-out passages
        self.loops = np.asarray(t["loops"], bool)           # (expert, contrast) x passages
        self.new_tokens = int(s.metadata["settings"]["new_tokens"])
        self.am_words = [show(i) for i in t["am_tok"]]
        self.am_score = float(np.log(t["am_ep"][0] / t["am_ap"][0]))
        self.threshold = self.alpha * float(self.ep.max())

    def subtitle_for(self, k):
        t = self.trace
        a = self.across
        best = int(np.argmax(self.score))
        return [
            "one prompt, read twice: by all 12 layers of GPT-2, the expert, and by the same "
            "model stopped after layer %d, the amateur. The expert's favourite next word is %s "
            "(%.2f); the amateur's is %s (%.2f)" % (
                self.layers, self.labels[0], self.ep[0], self.am_words[0], t["am_ap"][0]),
            "contrastive decoding scores a token by how much more the expert believes in it "
            "than the amateur: the difference of their log-probabilities. %s scores %+.1f; %s, "
            "the amateur's favourite, %+.1f" % (self.labels[best], self.score[best],
                                               self.am_words[0], self.am_score),
            "over all %s tokens the largest differences belong to tokens neither model expects. "
            "So only tokens with at least %d%% of the expert's top probability may compete: "
            "%d remain" % (f"{gpt2.N_VOCAB:,}", round(100 * self.alpha), self.kept),
            "among those %d, the differences are turned back into probabilities: %s gets %.2f "
            "and is picked, where the expert alone picks %s" % (
                self.kept, self.selected, self.contrast.max(), self.labels[0]),
            "over %d Shakespeare passages the picks differ at %d of %d positions. The real "
            "next token is the expert's pick %s of the time, the contrast's %s. Written out, "
            "%d expert and %d contrast continuations loop" % (
                self.loops.shape[1], a["differ"], a["positions"],
                pct(a["hit_expert"] / a["positions"]), pct(a["hit_contrast"] / a["positions"]),
                a["loops"][0], a["loops"][1]),
        ][k]

    def _prompt(self, sc, op):
        return self.tokens(sc, self.prompt, 0.05, 0.95, 0.335, op, w=80.0, h=38.0, size=0.032)

    # ------------------------------------------------------------------ 1 two models
    def _stack(self, sc, cx, op, lit, used, ink):
        """Twelve layers in a row, ``lit`` reached so far; the model stops after ``used``."""
        dx = self.frac(30)[0]
        for k in range(gpt2.N_LAYER):
            x = cx + (k - 5.5) * dx
            if k < min(lit, used):
                self.box(sc, x, 0.475, 26, 34, ink, op, radius=4)
            else:
                self.box(sc, x, 0.475, 26, 34, PAPER, op * (1.0 if k < used else 0.45),
                         border=AXIS if k < used else DIM, radius=4)
        return cx - 6 * dx, cx + (used - 6) * dx

    def phase_1(self, sc, u, op):
        t = self.trace
        self._prompt(sc, op)
        lit = seg(u, 0.06, 0.62) * gpt2.N_LAYER
        sides = (
            (0.27, self.EXPERT, gpt2.N_LAYER, "the expert: all 12 layers of GPT-2", self.labels,
             self.ep),
            (0.73, self.AMATEUR, self.layers, "the amateur: stopped after layer %d" % self.layers,
             self.am_words, t["am_ap"]))
        full = self.span(0.17)[0] / float(max(self.ep.max(), t["am_ap"].max()))
        for cx, ink, used, name, words, probs in sides:
            self.text(sc, cx, 0.407, name, op, size=0.030, ink=dark(ink))
            x0, x1 = self._stack(sc, cx, op, int(lit), used, ink)
            self.pulse(sc, (x0, 0.475), (x1, 0.475), lit / used, WHITE, op, r=6.0)
            out = op * smoothstep(used, used + 1.2, lit)
            for i in range(5):
                self.meter(sc, cx - 0.075, 0.575 + i * 0.058, full * probs[i], ink, out, words[i],
                           "%.2f" % probs[i], thick=22.0, size=0.032)
            self.text(sc, cx, 0.89, "its favourite next word: %s" % words[0], out, size=0.032,
                      strong=True)

    # ------------------------------------------------------------------ 2 subtract
    def phase_2(self, sc, u, op):
        self._prompt(sc, op)
        full = self.span(0.13)[0] / float(max(self.ep.max(), self.ap.max()))
        unit = self.span(0.165)[0] / float(np.abs(self.score).max())
        zero = 0.725
        self.text(sc, 0.16, 0.405, "expert's p", op, anchor="lm", size=0.030, ink=dark(BLUE))
        self.text(sc, 0.39, 0.405, "amateur's p", op, anchor="lm", size=0.030, ink=dark(AMBER))
        self.text(sc, 0.62, 0.405, "log p(expert) − log p(amateur)", op, anchor="lm", size=0.030,
                  strong=True)
        self.segment(sc, (zero, 0.435), (zero, 0.895), AXIS, op * 0.45, 1.4)
        for i, word in enumerate(self.labels):
            y = 0.465 + i * 0.056
            grow = smoothstep(0.0, 0.1, u)
            self.meter(sc, 0.16, y, full * self.ep[i] * grow, BLUE, op, word,
                       prob(self.ep[i]), thick=20.0, size=0.030)
            self.meter(sc, 0.39, y, full * self.ap[i] * grow, AMBER, op, "",
                       prob(self.ap[i]), thick=20.0, size=0.030)
            a = 0.12 + 0.085 * i
            v = seg(u, a, a + 0.12)
            self.pulse(sc, (0.30, y), (zero, y), v, BLUE, op, bend=0.02)
            self.pulse(sc, (0.53, y), (zero, y), v, AMBER, op, bend=-0.02)
            w = abs(self.score[i]) * unit * smoothstep(0.6, 1.0, v)
            if w > 0.2:
                sign = 1 if self.score[i] > 0 else -1
                self.rect(sc, self.shift(self.P(zero, y), sign * w / 2, 0), w, 20,
                          GREEN if sign > 0 else RED, op, radius=min(4.0, w / 2))
                self.label(sc, self.shift(self.P(zero, y), max(sign * w, 0) + 8, 0),
                           "%+.1f" % self.score[i], 0.030, anchor="lm", opacity=op,
                           ink=layout.STRONG_INK)
        self.text(sc, 0.05, 0.935, "green: the expert believes in it more than the amateur. "
                  "Red: less", op, anchor="lm", size=0.030)

    # ------------------------------------------------------------------ 3 the filter
    def phase_3(self, sc, u, op):
        t = self.trace
        self.text(sc, 0.05, 0.34, "largest differences over all %s tokens" % (
            f"{gpt2.N_VOCAB:,}"), op, anchor="lm", size=0.030, strong=True)
        for i in range(3):
            y = 0.42 + i * 0.085
            a = op * smoothstep(0.03 + 0.07 * i, 0.12 + 0.07 * i, u)
            self.chip(sc, 0.105, y, 92, 40, RED, a, show(t["raw_tok"][i]), size=0.030)
            self.text(sc, 0.17, y, "%+.1f    expert's p: 1 in %s" % (
                t["raw_score"][i], number(float("%.2g" % (1.0 / t["raw_ep"][i])))), a,
                anchor="lm", size=0.030)
        self.lines(sc, 0.05, 0.69, ["the expert finds these all but impossible;",
                                    "the amateur, even more so. The difference",
                                    "alone rewards noise"], op * smoothstep(0.26, 0.36, u),
                   size=0.032, step=0.05)
        b = op * smoothstep(0.38, 0.48, u)
        sweep, cut = smoothstep(0.5, 0.68, u), smoothstep(0.68, 0.76, u)
        self.lines(sc, 0.56, 0.34, ["the filter: keep a token only if the expert",
                                    "gives it %d%% of its top probability" % round(
                                        100 * self.alpha)], b, size=0.030, step=0.045)
        full = self.span(0.20)[0] / float(self.ep.max())
        x0 = 0.67
        for i, word in enumerate(self.labels):
            y = 0.475 + i * 0.052
            out = self.ep[i] < self.threshold and cut > 0.5
            self.meter(sc, x0, y, full * self.ep[i], DIM if out else BLUE, b, word,
                       "cut" if out else "%.3f" % self.ep[i], thick=20.0, size=0.030,
                       value_at=self.span(0.21)[0])
        tx = x0 + self.frac(full * self.threshold)[0]
        self.rect(sc, self.P(tx, 0.657), 2.6, self.span(0, 0.42 * sweep)[1] + 0.1, INK_DARK, b,
                  radius=1, lift=0.9)
        self.text(sc, x0, 0.925, "the line: %.3f" % self.threshold, b * sweep, anchor="lm",
                  size=0.030)
        self.text(sc, 0.05, 0.89, "%d of %s tokens pass the filter" % (
            self.kept, f"{gpt2.N_VOCAB:,}"), op * smoothstep(0.76, 0.86, u), anchor="lm",
            size=0.036, strong=True)

    # ------------------------------------------------------------------ 4 the pick
    def phase_4(self, sc, u, op):
        keep = [i for i in range(len(self.labels)) if self.ep[i] >= self.threshold]
        move = smoothstep(0.2, 0.55, u)
        full = self.span(0.33)[0] / float(self.contrast.max())
        self.text(sc, 0.05, 0.335, "the %d survivors: the expert's probability (outline), then "
                  "the contrast's" % len(keep), op, anchor="lm", size=0.030, strong=True)
        for row, i in enumerate(keep):
            y = 0.415 + row * 0.066
            now = self.ep[i] + (self.contrast[i] - self.ep[i]) * move
            self.meter(sc, 0.17, y, full * now, GREEN if move > 0 else BLUE, op, self.labels[i],
                       "%.2f" % now, thick=26.0, size=0.034, ghost=full * self.ep[i])
        a = op * smoothstep(0.05, 0.15, u)
        self.text(sc, 0.70, 0.43, "the expert alone writes", a, anchor="lm", size=0.030)
        self.chip(sc, 0.77, 0.51, 120, 46, BLUE, a, self.labels[0], size=0.038, solid=True)
        b = op * smoothstep(0.6, 0.72, u)
        self.text(sc, 0.70, 0.63, "the contrast writes", b, anchor="lm", size=0.030)
        self.chip(sc, 0.77, 0.71, 120, 46, GREEN, b, self.selected, size=0.038, solid=True)
        self.text(sc, 0.05, 0.905, "“%s %s”" % (CONTRAST_PROMPT, self.selected), b, anchor="lm",
                  size=0.034, strong=True)

    # ------------------------------------------------------------------ 5 held-out passages
    def phase_5(self, sc, u, op):
        a = self.across
        n, passages = a["positions"], self.loops.shape[1]
        self.text(sc, 0.05, 0.335, "one step: %d positions in %d passages of Shakespeare" % (
            n, passages), op, anchor="lm", size=0.032, strong=True)
        full = self.span(0.25)[0]
        rows = (("the two picks differ", a["differ"], VIOLET),
                ("the expert's pick is the real next token", a["hit_expert"], BLUE),
                ("the contrast's pick is the real next token", a["hit_contrast"], GREEN))
        for r, (label, count, ink) in enumerate(rows):
            y = 0.405 + r * 0.105
            g = smoothstep(0.04 + 0.12 * r, 0.2 + 0.12 * r, u)
            self.text(sc, 0.05, y, label, op, anchor="lm", size=0.030)
            self.meter(sc, 0.05, y + 0.047, full * count / n * g, ink, op,
                       value="%d of %d  (%s)" % (count, n, pct(count / n)), thick=24.0,
                       size=0.030, track=full, value_at=full)
        self.lines(sc, 0.05, 0.735, [
            "where they differ (%d positions): the expert's pick is the real token at %d, the "
            "contrast's at %d" % (a["differ"], a["expert_right"], a["contrast_right"])],
            op * smoothstep(0.42, 0.55, u), size=0.030, step=0.046, width=0.42)
        b = op * smoothstep(0.5, 0.6, u)
        self.text(sc, 0.555, 0.335, "written out: %d tokens after each passage" % (
            self.new_tokens), b, anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.555, 0.405, "red: the continuation repeats itself", b, anchor="lm",
                  size=0.030)
        for m, (name, ink) in enumerate((("expert", BLUE), ("contrast", GREEN))):
            y = 0.49 + m * 0.14
            c = op * smoothstep(0.58 + 0.12 * m, 0.66 + 0.12 * m, u)
            self.text(sc, 0.555, y, name, c, anchor="lm", size=0.034, ink=dark(ink))
            self.tally(sc, 0.675, y, self.loops[m], RED, c, seg(u, 0.6 + 0.12 * m, 0.76 + 0.12 * m),
                       pitch=0.0235)
            self.text(sc, 0.555, y + 0.058, "%d of %d repeat · distinct %s · log p %.2f"
                      % (a["loops"][m], passages, pct(a["distinct"][m]), a["own"][m]),
                      op * smoothstep(0.7 + 0.12 * m, 0.78 + 0.12 * m, u), anchor="lm",
                      size=0.030)
        self.lines(sc, 0.05, 0.865, [
            "the expert is trained to name the real next token; the contrast is meant to buy "
            "text that does not go round in circles",
            "log p: the expert's mean log-probability of the tokens written"],
            op * smoothstep(0.85, 0.95, u), size=0.030, step=0.05)


# ---------------------------------------------------------------------- 34 batching
def static_schedule(durations, slots=4):
    """Fixed batches of ``slots`` requests in arrival order, each batch
    holding its slots until its longest request ends: ``(slots, steps)``
    request indices, -1 where a slot idles (exact arithmetic)."""
    rows = [[] for _ in range(slots)]
    for b in range(0, len(durations), slots):
        batch = list(durations[b:b + slots])
        for k in range(slots):
            n = batch[k] if k < len(batch) else 0
            rows[k] += [b + k] * n + [-1] * (max(batch) - n)
    return np.asarray(rows, float)


class BatchingAndThroughput(Decoder):
    """Requests stacked into one matrix multiply, the measured throughput at
    each batch size, the exact count of arithmetic per byte moved that
    explains it, and a four-slot server filling its slots continuously
    against one that waits for whole batches."""

    TITLE = "Batching and throughput"
    PHASES = (("one multiply", 12.0), ("throughput", 11.0), ("why", 12.0), ("slots", 16.0))
    N = 384

    def setup(self):
        s = self.study
        from .next_language import SEED
        self.batches = [int(b) for b in s.panels[0].data["x"]]
        self.rate = np.asarray(s.panels[0].data["ys"][0], float)
        self.secs = np.asarray(s.panels[1].data["ys"][0], float)
        self.intensity = np.asarray(s.panels[2].data["ys"][0], float)
        self.slots = np.asarray(s.panels[3].data["values"], float)
        self.repeats = int(s.metadata["settings"]["repeats"])
        self.N = int(s.metadata["settings"]["matrix"])
        # the study's own weight matrix (same generator, same first draw): a corner of it
        self.corner = np.random.default_rng(SEED).normal(
            size=(self.N, self.N)).astype(np.float32)[:24, :24]
        ids = self.slots[self.slots >= 0].astype(int)
        self.durations = np.bincount(ids)
        self.static = static_schedule(self.durations, len(self.slots))
        self.steps = self.slots.shape[1]
        self.done_at = [int(np.flatnonzero((m >= 0).any(0)).max()) + 1
                        for m in (self.slots, self.static)]

    def bytes_flops(self, b):
        """Bytes moved (weights, requests and answers) and multiply-adds for a batch of ``b``."""
        n = self.N
        return 4 * n * n, 4 * 2 * b * n, 2 * b * n * n

    def subtitle_for(self, k):
        best = int(np.argmax(self.rate))
        dip = " (batch 2 came out slower than batch 1)" if self.rate[1] < self.rate[0] else ""
        return [
            "serving one request here is one multiply of its %d numbers by a %d × %d weight "
            "matrix. A batch stacks many requests into a single multiply: alone, %s requests a "
            "second; in a batch of %d, %s" % (
                self.N, self.N, self.N, number(self.rate[0]), self.batches[best],
                number(self.rate[best])),
            "measured on this machine, %d multiplies averaged per size: throughput peaks at "
            "batch %d with %s requests a second, %.1f times the rate of one at a time%s" % (
                self.repeats, self.batches[best], number(self.rate[best]),
                self.rate[best] / self.rate[0], dip),
            "why: the weights must be read from memory whatever the batch, and a bigger batch "
            "does more arithmetic with them. Operations per byte moved: %.1f at batch 1, %.0f "
            "at batch %d (exact counts)" % (self.intensity[0], self.intensity[-1],
                                           self.batches[-1]),
            "a simulated server, %d slots, %d requests of %d to %d steps: refilling a slot the "
            "moment its request ends finishes all of them in %d steps; fixed batches of %d wait "
            "for their slowest and take %d" % (
                len(self.slots), len(self.durations), self.durations.min(), self.durations.max(),
                self.done_at[0], len(self.slots), self.done_at[1]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _stack(self, sc, cx, cy, b, ink, op, w=140.0, tall=168.0):
        """``b`` requests as a stack of rows centred at ``(cx, cy)``."""
        pitch = min(15.0, tall / b)
        h = pitch * (0.8 if b <= 16 else 0.62)
        for i in range(b):
            self.rect(sc, self.shift(self.P(cx, cy), 0, (i - (b - 1) / 2) * pitch), w, h, ink, op,
                      radius=min(3.0, h / 2))

    def _at(self, u, span=0.9):
        q = u / span * len(self.batches)
        k = min(int(q), len(self.batches) - 1)
        return k, seg(q - k, 0.0, 1.0)

    # ------------------------------------------------------------------ 1 one multiply
    def phase_1(self, sc, u, op):
        k, v = self._at(u)
        b = self.batches[k]
        y = 0.585
        self.text(sc, 0.20, 0.345, "a batch of %d request%s" % (b, "" if b == 1 else "s"), op,
                  size=0.032, strong=True)
        self.text(sc, 0.50, 0.345, "the weights, %d × %d" % (self.N, self.N), op, size=0.032,
                  strong=True)
        self.text(sc, 0.80, 0.345, "%d answer%s" % (b, "" if b == 1 else "s"), op, size=0.032,
                  strong=True)
        self._stack(sc, 0.20, y, b, BLUE, op * smoothstep(0.0, 0.15, v))
        self.grid(sc, self.corner, 0.50, y, 180, 180, op, hot=AMBER, cold=TEAL)
        self.text(sc, 0.50, 0.805, "(a 24 × 24 corner)", op)
        self.arrow(sc, (0.295, y), (0.385, y), AXIS, op, width=2.4)
        self.arrow(sc, (0.615, y), (0.705, y), AXIS, op, width=2.4)
        self.pulse(sc, (0.295, y), (0.385, y), seg(v, 0.12, 0.5), BLUE, op, r=7.0)
        self.pulse(sc, (0.615, y), (0.705, y), seg(v, 0.42, 0.8), GREEN, op, r=7.0)
        out = op * smoothstep(0.7, 0.85, v)
        self._stack(sc, 0.80, y, b, GREEN, out)
        self.text(sc, 0.50, 0.875, "one multiply: %.1f µs    %s requests a second" % (
            self.secs[k] * 1e6, number(self.rate[k])), op, size=0.036, strong=True)
        self.text(sc, 0.50, 0.93, "each request: %d numbers in, %d out" % (self.N, self.N), op,
                  size=0.030)

    # ------------------------------------------------------------------ 2 throughput
    def _curve(self, sc, chart, ys, ink, op, k, v):
        """The measured points up to batch ``k`` (the last segment ``v`` drawn)."""
        pts = [(chart.fx(b), chart.fy(y)) for b, y in zip(self.batches, ys)]
        for i in range(k - 1):
            self.segment(sc, pts[i], pts[i + 1], ink, op, 3.2)
        if k and v > 0:
            a, b = pts[k - 1], pts[k]
            self.segment(sc, a, (a[0] + (b[0] - a[0]) * v, a[1] + (b[1] - a[1]) * v), ink, op, 3.2)
        for i in range(k + 1):
            if i < k or v >= 1.0 or k == 0:
                self.disk(sc, self.P(*pts[i]), 5.5, ink, op, glow=0.3 if i == k else 0.0,
                          lift=1.4, outline=0.0)

    def phase_2(self, sc, u, op):
        k, v = self._at(u, 0.85)
        chart = self.chart(((0.14, 0.62), (0.40, 0.80)), self.batches, np.r_[0.0, self.rate],
                           xlog=True)
        self.axes(sc, chart, op, "requests in the batch", "requests answered per second",
                  xticks=[(b, str(b)) for b in self.batches])
        self._curve(sc, chart, self.rate, BLUE, op, k, smoothstep(0.0, 0.5, v))
        b = self.batches[k]
        self._stack(sc, 0.82, 0.50, b, BLUE, op, w=120.0, tall=120.0)
        self.text(sc, 0.82, 0.68, "batch of %d" % b, op, size=0.034, strong=True)
        self.text(sc, 0.82, 0.74, "%s a second" % number(self.rate[k]), op, size=0.036,
                  ink=dark(BLUE))
        self.text(sc, 0.82, 0.80, "%.1f × one at a time" % (self.rate[k] / self.rate[0]), op,
                  size=0.030)

    # ------------------------------------------------------------------ 3 why
    def phase_3(self, sc, u, op):
        k, v = self._at(u, 0.85)
        b = self.batches[k]
        weights, data, flops = self.bytes_flops(b)
        w_max, d_max, f_max = self.bytes_flops(self.batches[-1])
        full = self.span(0.36)[0]
        self.text(sc, 0.05, 0.345, "batch of %d" % b, op, anchor="lm", size=0.038, strong=True)
        self.text(sc, 0.05, 0.42, "bytes moved through memory: %s" % f"{weights + data:,}",
                  op, anchor="lm", size=0.030)
        at = self.P(0.05, 0.475)
        ww, dw = full * weights / (w_max + d_max), full * data / (w_max + d_max)
        self.rect(sc, self.shift(at, ww / 2, 0), ww, 28, OCHRE, op, radius=4)
        self.rect(sc, self.shift(at, ww + dw / 2 + 1, 0), max(dw, 0.6), 28, BLUE, op, radius=2)
        self.text(sc, 0.05, 0.53, "brown: the weights, %s whatever the batch" % (
            f"{weights:,}"), op, anchor="lm", ink=dark(OCHRE))
        self.text(sc, 0.05, 0.62, "multiply-adds done: %s" % f"{flops:,}", op,
                  anchor="lm", size=0.030)
        fw = full * flops / f_max
        self.rect(sc, self.shift(self.P(0.05, 0.675), fw / 2, 0), fw, 28, AMBER, op,
                  radius=min(4.0, fw / 2))
        self.text(sc, 0.05, 0.78, "%.1f operations per byte moved" % self.intensity[k], op,
                  anchor="lm", size=0.038, strong=True)
        self.text(sc, 0.05, 0.85, "few: the time goes on fetching.  Many: on arithmetic", op,
                  anchor="lm", size=0.030)
        chart = self.chart(((0.56, 0.95), (0.42, 0.80)), self.batches, np.r_[0.0, self.intensity],
                           xlog=True)
        self.axes(sc, chart, op, "requests in the batch", "operations per byte (exact)",
                  xticks=[(x, str(x)) for x in self.batches])
        self._curve(sc, chart, self.intensity, AMBER, op, k, smoothstep(0.0, 0.5, v))

    # ------------------------------------------------------------------ 4 slots
    def _ink(self, i):
        ink = INKS[i % len(INKS)]
        return ink if i < len(INKS) else soft(ink, 0.45)

    def _gantt(self, sc, grid, cy, op, now, title, done_at):
        x0, x1 = 0.17, 0.83
        rows, cols = grid.shape
        tex = np.empty((rows, cols, 3), np.float32)
        shown = int(np.ceil(now))
        for r in range(rows):
            for c in range(cols):
                i = int(grid[r, c])
                tex[r, c] = PAPER * 0.97 if c >= shown else self._ink(i) if i >= 0 else DIM
        w, h = self.span(x1 - x0)[0], 28.0 * rows
        centre = self.P((x0 + x1) / 2, cy)
        self.rect(sc, centre, w + 8, h + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5,
                  lift=0.1)
        draw_image(self, sc, centre, tex, w, h, op, grid=0.2)
        dx, dy = (x1 - x0) / cols, self.frac(0, 28.0)[1]
        for r in range(rows):
            y = cy + (r - (rows - 1) / 2) * dy
            self.text(sc, x0 - 0.012, y, "slot %d" % (r + 1), op, anchor="rm")
            for c in range(min(shown, cols)):
                i = int(grid[r, c])
                if i >= 0 and (c == 0 or grid[r, c - 1] != i):
                    self.text(sc, x0 + (c + 0.5) * dx, y, str(i + 1), op, ink=WHITE)
        self.rect(sc, self.P(x0 + min(now, cols) * dx, cy), 3.0, h + 16, INK_DARK, op, radius=1,
                  lift=0.9)
        self.text(sc, x0, cy - dy * rows / 2 - 0.04, title, op, anchor="lm", size=0.030,
                  strong=True)
        ends = [int(np.flatnonzero(grid == i).max() % cols) + 1 for i in range(len(self.durations))]
        done = sum(e <= now for e in ends)
        self.text(sc, x1 + 0.018, cy - 0.022, "%d of %d" % (done, len(ends)), op, anchor="lm",
                  size=0.034, strong=True)
        self.text(sc, x1 + 0.018, cy + 0.03, "done" if now < done_at else
                  "in %d steps" % done_at, op, anchor="lm", size=0.030)

    def phase_4(self, sc, u, op):
        now = seg(u, 0.03, 0.85) * self.steps
        self._gantt(sc, self.slots, 0.475, op, now,
                    "continuous: a finished request's slot is refilled at once", self.done_at[0])
        self._gantt(sc, self.static, 0.775, op, now,
                    "fixed batches of %d: the next batch waits for the slowest" % len(self.slots),
                    self.done_at[1])
        busy = [float((m[:, :d] >= 0).mean()) for m, d in zip((self.slots, self.static),
                                                             self.done_at)]
        self.text(sc, 0.17, 0.94, "step %d of %d.  Slots busy until done: %s continuous, %s in "
                  "fixed batches (grey: idle)" % (min(int(now), self.steps), self.steps,
                                                  pct(busy[0]), pct(busy[1])), op, anchor="lm",
                  size=0.030)


ANIMATIONS = {
    "beam-search": BeamSearch,
    "speculative-decoding": SpeculativeDecoding,
    "repetition-and-penalties": RepetitionAndPenalties,
    "constrained-decoding": ConstrainedDecoding,
    "contrastive-decoding": ContrastiveDecoding,
    "batching-and-throughput": BatchingAndThroughput,
}
