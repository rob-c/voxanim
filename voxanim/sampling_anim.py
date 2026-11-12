"""Sampling the next token: what is done with GPT-2's output distribution.

After "The cat sat on the" GPT-2 small gives 50,257 logits; softmax makes
them a distribution, and a decoding rule picks a token from it.  The
distribution here is flat (" floor" 7.6%, " bed" 6.5%, " couch" 5.4%, ...;
entropy 7.8 bits), which is what makes the rules matter.

1. **greedy** -- always the largest: " floor",
2. **temperature** -- the logits are divided by T before the softmax: T =
   0.5 sharpens (" floor" 25%, 3.5 bits), T = 1.5 flattens (1.8%, 11.7
   bits); the bars reshape as T moves,
3. **top-k** -- keep the k = 10 likeliest (42.6% of the mass), zero the
   rest, renormalise,
4. **top-p** -- nucleus sampling keeps the smallest set whose mass reaches
   p = 0.9: sorted by probability, the cumulative mass crosses 0.9 at rank
   728 here (for "One, two, three," at 102: the set adapts),
5. **generate** -- six tokens by nucleus sampling (p = 0.9, seed 4), each
   step's distribution shown with the random number that picked from it:
   "The cat sat on the sofa and yawned a little",
6. **KV cache** -- why step t only needs the new token's Q against the
   stored K and V: one head's K cache drawn growing by one row per token,
   the new q against it, and the real shapes (12 layers x 12 heads x T x
   64).

The sampling rules are plain functions here (:func:`softmax_t`,
:func:`top_k_mask`, :func:`top_p_mask`, :func:`pick`) and tested on small
hand-made distributions; the generation runs the real model once per step
(:func:`generate`, cached by :func:`voxanim.cache.cached_npz`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache

import numpy as np

from . import color as C
from . import gpt2 as G
from . import layout
from .cache import cached_npz
from .diagram import DATA, DATA_HI, NODE, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2_panel import diverging
from .text import num

PROMPT = "The cat sat on the"
CONTRAST = "One, two, three,"          # a peaked distribution, for comparison
TEMPERATURES = (0.5, 1.0, 1.5)
TOP_K = 10
TOP_P = 0.9
N_GEN, SEED = 6, 4                      # tokens generated, and the generator's seed
N_BARS = 12                             # tokens listed in the bar chart
N_GEN_BARS = 8                          # ...per generation step
N_TAIL = 1200                           # ranks drawn in the sorted plot
HEAD = (4, 11)                          # the head whose K cache is drawn: layer 5, head 12

# ------------------------------------------------------------------ the rules


def softmax_t(logits, temperature=1.0):
    """Softmax of ``logits / temperature`` (``temperature`` > 0)."""
    return G.softmax(np.asarray(logits, np.float64) / temperature)


def entropy_bits(p):
    """Shannon entropy of distribution ``p`` in bits (zeros contribute nothing)."""
    p = np.asarray(p, np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def top_k_mask(p, k):
    """Boolean mask of the ``k`` largest entries of ``p``."""
    p = np.asarray(p)
    m = np.zeros(p.shape, bool)
    m[np.argsort(-p, kind="stable")[:k]] = True
    return m


def top_p_mask(p, top_p):
    """Boolean mask of the nucleus: the smallest set of the largest entries
    whose mass reaches ``top_p`` (always at least one)."""
    p = np.asarray(p, np.float64)
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order])
    n = int(np.searchsorted(cum, top_p * (1 - 1e-12))) + 1
    m = np.zeros(p.shape, bool)
    m[order[:min(n, len(p))]] = True
    return m


def renormalised(p, mask):
    """``p`` with everything outside ``mask`` set to 0 and the rest scaled to sum 1."""
    q = np.where(mask, p, 0.0)
    return q / q.sum()


def pick(p, u):
    """Inverse-CDF sampling: the index whose cumulative probability first exceeds ``u``."""
    return int(min(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1))


@cache
def distribution(text=PROMPT):
    """``(logits, p)`` for the token after ``text`` (the model's real output)."""
    lg = np.asarray(G.run(text).logits[-1], np.float64)
    return lg, G.softmax(lg)


def generate(text=PROMPT, n=N_GEN, top_p=TOP_P, seed=SEED, temperature=1.0):
    """``n`` tokens by nucleus sampling after ``text``, each step running the
    model on the text so far.  Returns ``{"ids", "probs", "kept", "u"}``:
    the chosen ids, each step's full distribution (n, N_VOCAB), the nucleus
    sizes and the uniform random numbers that picked (cached on disk)."""
    settings = dict(text=text, n=n, top_p=top_p, seed=seed, temperature=temperature)

    def build():
        rng = np.random.default_rng(seed)
        tok = G.Tokenizer()
        ids = tok.encode(text)
        out = dict(ids=[], probs=[], kept=[], u=[])
        for _ in range(n):
            p = softmax_t(G.run(tok.decode(ids)).logits[-1], temperature)
            m = top_p_mask(p, top_p)
            u = float(rng.random())
            j = pick(renormalised(p, m), u)
            ids.append(j)
            out["ids"].append(j)
            out["probs"].append(p.astype(np.float32))
            out["kept"].append(int(m.sum()))
            out["u"].append(u)
        return {k: np.array(v) for k, v in out.items()}

    return cached_npz("gpt2-sampling", settings, build)


# ------------------------------------------------------------------ layout (fractions, px)

NAME_X = 0.215                          # token names end here
BAR_X0 = 0.23                           # bars start here
BAR_MAX_W = 300.0                       # px at P_FULL
P_FULL = 0.30                           # the probability drawn at BAR_MAX_W
ROWS_Y = 0.275                          # the first bar
ROW_DY = 23.0                           # px between bars
BAR_H = 13.0
READ_X, READ_Y = 0.66, 0.29             # the read-outs at the right
READ_DY = 27.0                          # px between read-out lines
CHART = (0.075, 0.30, 0.58, 0.82)       # the sorted plot: x0, y0, x1, y1
CHART_PAD = 10.0
TEXT_AT = (0.07, 0.25)                  # the growing text of the generation
TOKEN_PAD = 8.0
BOX_H = 24.0
KV_AT = (0.31, 0.56)                    # centre of the K cache
KV_CELL = (22.0, 5.6)                   # px per row (token) and column (feature)
KV_GAP = 10.0                           # between q, K and the scores
PANEL = (0.96, 0.96, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
BAR = C.mix(DATA, (0.6, 0.62, 0.64), 0.5)
BAR_HI = DATA
BAR_OFF = (0.80, 0.81, 0.82)            # a bar cut by top-k
CUT = (0.80, 0.25, 0.20)                # cut lines
CUM = (0.75, 0.32, 0.58)                # the cumulative mass
BOX_FILL = (0.975, 0.975, 0.97)
BOX_EDGE = (0.62, 0.63, 0.65)
HOT_FILL = C.mix(NODE, DATA_HI, 0.45)

# ------------------------------------------------------------------ timeline (seconds)

TIMELINE = dict(intro=(0.2, 1.0), bars=(0.6, 2.6), greedy=(2.6, 4.0), temp=(7.4, 17.4),
                topk_cut=(17.8, 19.0), topk_norm=(19.2, 20.6), topp=(24.2, 25.2),
                cum=(25.4, 29.4), nucleus=(29.6, 30.6), gen=(33.6, 46.8), kv=(47.2, 48.2),
                kv_grow=(48.4, 53.2), kv_q=(53.6, 56.0))
T_DONE = 58.6
# the temperature's path through the phase: (progress, T), eased between
T_PATH = ((0.0, 1.0), (0.2, 0.5), (0.4, 0.5), (0.65, 1.5), (0.85, 1.5), (1.0, 1.0))
GEN_STEP = (TIMELINE["gen"][1] - TIMELINE["gen"][0]) / N_GEN


@dataclass
class Facts:
    """What the animation reads from the model (see :func:`facts`)."""

    logits: np.ndarray
    p: np.ndarray
    order: np.ndarray                   # tokens by falling probability
    names: list                         # the top N_BARS tokens' strings
    entropy: dict                       # T -> bits
    k_mass: float                       # mass of the top TOP_K
    nucleus: int                        # tokens the nucleus keeps
    nucleus_contrast: int               # ...for CONTRAST
    gen: dict                           # generate()
    gen_names: list                     # the generated tokens' strings
    gen_tops: list                      # per step: [(token, p)] for the top N_GEN_BARS
    k_cache: np.ndarray                 # (T, 64) K of HEAD for the finished text
    q_last: np.ndarray                  # (64,) the last token's q
    scores: np.ndarray                  # (T,) q . k / 8
    tokens: list                        # the finished text's tokens


@cache
def facts():
    """Run the model (once per distinct text) and gather every number shown."""
    tok = G.Tokenizer()
    logits, p = distribution(PROMPT)
    order = np.argsort(-p, kind="stable")
    names = [tok.decode([int(i)]) for i in order[:N_BARS]]
    entropy = {T: entropy_bits(softmax_t(logits, T)) for T in TEMPERATURES}
    gen = generate()
    gen_names = [tok.decode([int(i)]) for i in gen["ids"]]
    tops = []
    for row in gen["probs"]:
        o = np.argsort(-row, kind="stable")[:N_GEN_BARS]
        tops.append([(tok.decode([int(i)]), float(row[i])) for i in o])
    final = G.run(PROMPT + "".join(gen_names))
    l, h = HEAD
    k = final.k[l, h]
    q = final.q[l, h, -1]
    return Facts(logits, p, order, names, entropy, float(p[order[:TOP_K]].sum()),
                 int(top_p_mask(p, TOP_P).sum()), int(top_p_mask(distribution(CONTRAST)[1],
                                                                 TOP_P).sum()),
                 gen, gen_names, tops, k, q, final.scores[l, h, -1], final.tokens)


class SamplingAnimation(Diagram):
    """The ``gpt2-sampling`` animation (see the module docstring)."""

    PHASES = ("greedy", "temperature", "top-k", "top-p", "generate", "KV cache")

    def __init__(self):
        super().__init__("Sampling the next token", "")
        self.f = facts()
        self.Tm = dict(TIMELINE)
        self.hold, self.reset = 3.0, 1.5
        self.T0 = len(G.Tokenizer().encode(PROMPT))
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((5, "kv"), (4, "gen"), (3, "topp"), (2, "topk_cut"), (1, "temp")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def temperature(self, t):
        """T at time ``t``: along :data:`T_PATH` during the temperature phase, else 1."""
        u = self.progress("temp", t)
        if u <= 0 or u >= 1:
            return 1.0
        for (u0, t0), (u1, t1) in itertools.pairwise(T_PATH):
            if u0 <= u <= u1:
                return t0 + (t1 - t0) * ease_in_out(remap(u, u0, u1))
        return 1.0

    def gen_step(self, t):
        """``(step, u)``: the generation step under way (0-based) and its progress."""
        g = self.progress("gen", t)
        if g <= 0:
            return -1, 0.0
        if g >= 1:
            return N_GEN - 1, 1.0
        k = min(N_GEN - 1, int(g * N_GEN))
        return k, g * N_GEN - k

    def shown(self, t):
        """The distribution the bars show at ``t`` and its masks: ``(p, keep, scale)``
        where ``keep`` is the rows kept by top-k (None otherwise) and ``scale``
        the renormalisation applied so far."""
        p = softmax_t(self.f.logits, self.temperature(t))
        cut = self.progress("topk_cut", t)
        if cut <= 0:
            return p, None, 1.0
        keep = top_k_mask(p, TOP_K)
        norm = ease_out(self.progress("topk_norm", t))
        scale = 1 + (1 / self.f.k_mass - 1) * norm
        return p, keep, scale

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        Tm = self.Tm
        bars = smoothstep(0.0, 1.0, self.progress("bars", t)) * \
            (1 - smoothstep(Tm["topp"][0] - 0.8, Tm["topp"][0], t)) * back
        self._bars(sc, t, bars)
        self._readouts(sc, t, bars)
        plot = smoothstep(*Tm["topp"], t) * (1 - smoothstep(Tm["gen"][0] - 0.8, Tm["gen"][0], t))
        self._sorted_plot(sc, t, plot * back)
        gen = smoothstep(Tm["gen"][0] - 0.3, Tm["gen"][0] + 0.5, t) * \
            (1 - smoothstep(Tm["kv"][0] - 0.6, Tm["kv"][0], t))
        self._generation(sc, t, gen * back)
        self._kv(sc, t, smoothstep(*Tm["kv"], t) * back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    def _bar_row(self, sc, c, name, p, op, hot=0.0, off=False, grow=1.0, quote=True):
        """One bar: the token's name, the bar (``p`` of P_FULL wide) and its probability."""
        self.label(sc, c, '"%s"' % name if quote else name, layout.MIN_TEXT, anchor="rm", dx=-10,
                   opacity=op, kind="mono", ink=layout.STRONG_INK if hot else layout.CAPTION_INK)
        w = max(1.5, BAR_MAX_W * min(1.0, p / P_FULL) * grow)
        col = BAR_OFF if off else C.mix(BAR, BAR_HI, hot)
        self.rect(sc, self.shift(c, w / 2, 0.0), w, BAR_H, col, op, radius=2, lift=0.3)
        self.label(sc, c, "%.1f%%" % (100 * p) if p >= 0.0005 else "%.2f%%" % (100 * p),
                   layout.MIN_TEXT, anchor="lm", dx=w + 7, opacity=op * (0.55 if off else 1.0),
                   kind="mono", ink=layout.STRONG_INK if hot else layout.CAPTION_INK)

    def _bars(self, sc, t, op):
        """The top tokens as bars (at the current temperature; cut by top-k), and
        the rest of the mass as one more bar."""
        if op <= 0.01:
            return
        f = self.f
        p, keep, scale = self.shown(t)
        greedy = smoothstep(0.0, 1.0, self.progress("greedy", t))
        grow = ease_out(self.progress("bars", t))
        c0 = self.P(BAR_X0, ROWS_Y)
        for n, i in enumerate(f.order[:N_BARS]):
            c = self.shift(c0, 0.0, n * ROW_DY)
            off = keep is not None and not keep[i]
            q = p[i] * (scale if keep is not None and keep[i] else 1.0)
            self._bar_row(sc, c, f.names[n], q, op, greedy if n == 0 else 0.0, off, grow)
        rest = 1 - p[f.order[:N_BARS]].sum()
        c = self.shift(c0, 0.0, (N_BARS + 0.5) * ROW_DY)
        self._bar_row(sc, c, "%s others" % format(G.N_VOCAB - N_BARS, ","), rest, op * 0.8,
                      off=keep is not None, grow=grow, quote=False)
        cut = self.progress("topk_cut", t)
        if cut > 0:
            y = (TOP_K - 0.5) * ROW_DY
            a = self.shift(c0, -layout.text_width('"' + max(f.names, key=len) + '"',
                                                  layout.MIN_TEXT, "mono") * self.cam.width - 14, y)
            b = self.shift(c0, BAR_MAX_W * 0.75, y)
            k = ease_out(cut)
            sc.line(a + self.view * 0.5, a + (b - a) * k + self.view * 0.5, width=1.6, color=CUT,
                    opacity=0.9 * op, bias=0.3)
            self.label(sc, b, "k = %d" % TOP_K, layout.MIN_TEXT, anchor="lm", dx=8, opacity=op * k,
                       ink=CUT)

    def _readouts(self, sc, t, op):
        """Temperature, entropy, mass and the pick, at the right."""
        if op <= 0.01:
            return
        f = self.f
        T = self.temperature(t)
        p, keep, scale = self.shown(t)
        H = entropy_bits(p if keep is None else renormalised(p, keep))
        top = f.names[0]
        lines = [("softmax over %s logits" % format(G.N_VOCAB, ","), layout.NOTE_INK)]
        if self.progress("temp", t) > 0:
            lines.append(("T = %.2f:  softmax(z ÷ T)" % T, layout.STRONG_INK))
        lines.append(("entropy %.1f bits" % H, layout.STRONG_INK))
        lines.append(("≈ %s equally likely tokens" % format(int(round(2 ** H)), ","),
                      layout.CAPTION_INK))
        lines.append(("top %d hold %.0f%% of the mass" % (N_BARS, 100 * p[f.order[:N_BARS]].sum())
                      if keep is None else "top-%d: %.1f%% kept, scaled × %.2f"
                      % (TOP_K, 100 * f.k_mass, scale), layout.CAPTION_INK))
        lines.append(("tokens with p > 1%%: %d" % int((p > 0.01).sum()), layout.CAPTION_INK))
        if self.progress("greedy", t) > 0 and keep is None:
            lines.append(('greedy: "%s" every time' % top, layout.STRONG_INK))
        c0 = self.P(READ_X, READ_Y)
        for n, (text, ink) in enumerate(lines):
            self.label(sc, self.shift(c0, 0.0, n * READ_DY), text, layout.MIN_TEXT, anchor="lm",
                       opacity=op, ink=ink)

    def _sorted_plot(self, sc, t, op):
        """The distribution sorted by rank with its cumulative mass, the nucleus cut
        at p = 0.9 and the top-k cut for comparison."""
        if op <= 0.01:
            return
        f = self.f
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        ps = f.p[f.order[:N_TAIL]]
        cum = np.cumsum(ps)
        pmax = float(ps[0])

        def pt(rank, v):
            fx = CHART_PAD + (w - 2 * CHART_PAD) * rank / N_TAIL
            fy = CHART_PAD + (h - 2 * CHART_PAD) * (1 - v)
            return self.shift(a, fx, fy) + self.view * 0.3

        ranks = np.unique(np.concatenate([np.arange(0, 40), np.linspace(40, N_TAIL - 1, 150)])
                          .astype(int))
        self.polyline(sc, [pt(r, ps[r] / pmax) for r in ranks], width=2.0, color=DATA,
                      opacity=0.95 * op)
        cu = ease_out(self.progress("cum", t))
        if cu > 0:
            upto = max(1, int(cu * N_TAIL))
            rs = [r for r in ranks if r < upto]
            self.polyline(sc, [pt(r, cum[r]) for r in rs], width=2.0, color=CUM,
                          opacity=0.95 * op)
            r = rs[-1]
            self.label(sc, pt(r, cum[r]), "%.2f" % cum[r], layout.MIN_TEXT, anchor="lt", dx=6,
                       dy=6, opacity=op, ink=CUM, kind="mono")
        nu = smoothstep(0.0, 1.0, self.progress("nucleus", t))
        if nu > 0:
            top = pt(0, 1.0) - (pt(0, 0.0) - pt(0, 1.0)) * (CHART_PAD / (h - 2 * CHART_PAD))
            for r, text, k in ((f.nucleus, "p = %.1f: %d tokens" % (TOP_P, f.nucleus), 0.9),
                               (TOP_K, "k = %d" % TOP_K, 0.6)):
                sc.line(pt(r, 0.0), pt(r, 0.0) + (top - pt(0, 0.0)) * nu, width=1.6, color=CUT,
                        opacity=k * op, bias=0.3)
                self.label(sc, pt(r, 1.0), text, layout.MIN_TEXT, anchor="lb", dx=-4,
                           dy=-CHART_PAD - 4, opacity=op * nu, ink=CUT)
        self.label(sc, pt(0, 1.0), "p (top: %.1f%%)" % (100 * pmax), layout.MIN_TEXT,
                   anchor="lt", dx=6, dy=4, opacity=op, ink=layout.NOTE_INK)
        self.label(sc, pt(0, 1.0), "cumulative mass", layout.MIN_TEXT, anchor="lt", dx=6,
                   dy=4 + READ_DY * 0.8, opacity=op, ink=CUM)
        self.label(sc, pt(0, 0.0), "rank 1", layout.MIN_TEXT, anchor="lt", dy=10, opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, pt(N_TAIL, 0.0), "%s" % format(N_TAIL, ","), layout.MIN_TEXT, anchor="rt",
                   dy=10, opacity=op, ink=layout.NOTE_INK, kind="mono")
        for v in (1.0, 0.5, 0.0):
            self.label(sc, pt(N_TAIL, v), "%.1f" % v, layout.MIN_TEXT, anchor="lm", dx=10,
                       opacity=op, ink=CUM, kind="mono")

    def _text_row(self, sc, t, op, n_new, hot):
        """The prompt and the tokens generated so far as boxes; returns the row's end."""
        p0 = self.P(*TEXT_AT)
        x = 0.0
        toks = self.f.tokens[:self.T0 + n_new]
        for i, tokn in enumerate(toks):
            w = layout.text_width(tokn, layout.MIN_TEXT, "mono") * self.cam.width + 12
            c = self.shift(p0, x + w / 2, 0.0)
            new = i >= self.T0
            k = hot if i == len(toks) - 1 and new else (0.5 if new else 0.0)
            self.rect(sc, c, w, BOX_H, C.mix(BOX_FILL, HOT_FILL, k), op,
                      border=C.mix(BOX_EDGE, DATA, k), border_w=1.2, radius=5, lift=0.2)
            self.label(sc, c, tokn, layout.MIN_TEXT, opacity=op, kind="mono",
                       ink=layout.STRONG_INK)
            x += w + TOKEN_PAD
        return self.shift(p0, x, 0.0)

    def _generation(self, sc, t, op):
        """Six steps of nucleus sampling: the text so far, the step's top tokens,
        the nucleus size, the random number and the pick."""
        if op <= 0.01:
            return
        f = self.f
        k, u = self.gen_step(t)
        k = max(k, 0)
        appear = smoothstep(0.0, 0.25, u)                   # the step's bars
        picked = smoothstep(0.55, 0.75, u)                  # the pick lights up
        done = smoothstep(0.8, 0.95, u)                     # ...and joins the text
        end = self._text_row(sc, t, op, k + int(done > 0), picked)
        self.label(sc, end, "step %d of %d" % (k + 1, N_GEN), layout.MIN_TEXT, anchor="lm",
                   dx=6, opacity=op, ink=layout.NOTE_INK)
        c0 = self.P(BAR_X0, ROWS_Y + 0.07)
        tops = f.gen_tops[k]
        chosen = f.gen_names[k]
        in_top = any(name == chosen for name, _p in tops)
        p_chosen = float(f.gen["probs"][k][f.gen["ids"][k]])
        rows = list(tops) + ([] if in_top else [(chosen, p_chosen)])
        for n, (name, p) in enumerate(rows):
            c = self.shift(c0, 0.0, n * ROW_DY + (ROW_DY * 0.5 if n >= len(tops) else 0.0))
            hot = picked if name == chosen else 0.0
            self._bar_row(sc, c, name, p, op * appear, hot)
        lines = [("%d tokens in, %s logits out" % (self.T0 + k, format(G.N_VOCAB, ",")),
                  layout.NOTE_INK),
                 ("nucleus p = %.1f keeps %d token%s" % (TOP_P, f.gen["kept"][k],
                                                         "" if f.gen["kept"][k] == 1 else "s"),
                  layout.CAPTION_INK),
                 ("u = %.3f  →  \"%s\"" % (f.gen["u"][k], chosen), layout.STRONG_INK)]
        c1 = self.P(READ_X, READ_Y + 0.07)
        for n, (text, ink) in enumerate(lines):
            k_on = appear if n < 2 else picked
            self.label(sc, self.shift(c1, 0.0, n * READ_DY), text, layout.MIN_TEXT, anchor="lm",
                       opacity=op * k_on, ink=ink)

    def _kv(self, sc, t, op):
        """One head's K cache growing a row per generated token, then the new
        token's q against every stored row, with the real shapes."""
        if op <= 0.01:
            return
        f = self.f
        T = len(f.tokens)
        grow = self.progress("kv_grow", t)
        n_rows = self.T0 + int(min(N_GEN, grow * (N_GEN + 0.999))) if grow > 0 else self.T0
        n_rows = min(T, n_rows)
        rh, cw = KV_CELL
        kw, kh = G.D_HEAD * cw, T * rh
        c = self.P(*KV_AT)
        top_left = self.shift(c, -kw / 2, -kh / 2)
        K = diverging(f.k_cache)
        for i in range(n_rows):
            rc = self.shift(top_left, kw / 2, (i + 0.5) * rh)
            draw_image(self, sc, rc, K[i:i + 1], kw, rh - 1.5, op, grid=0.0)
            self.label(sc, rc, f.tokens[i], layout.MIN_TEXT, anchor="rm", dx=-(kw / 2 + 8),
                       opacity=op, kind="mono",
                       ink=layout.STRONG_INK if i >= self.T0 else layout.CAPTION_INK)
        self.label(sc, self.shift(top_left, kw / 2, -KV_GAP - rh), "K cache of one head: T × %d"
                   % G.D_HEAD, layout.MIN_TEXT, anchor="mb", dy=-4, opacity=op)
        qk = self.progress("kv_q", t)
        if qk > 0:
            on = smoothstep(0.0, 0.3, qk) * op
            qc = self.shift(top_left, kw / 2, -KV_GAP - rh / 2)
            draw_image(self, sc, qc, diverging(f.q_last)[None], kw, rh - 1.5, on, grid=0.0)
            self.label(sc, qc, "new q", layout.MIN_TEXT, anchor="rm", dx=-(kw / 2 + 8),
                       opacity=on, kind="mono", ink=layout.STRONG_INK)
            sw = ease_out(remap(qk, 0.3, 1.0))
            n_sc = int(sw * T + 1e-9)
            for i in range(min(T, n_sc + (1 if sw < 1 else 0))):
                rc = self.shift(top_left, kw + KV_GAP + 20, (i + 0.5) * rh)
                self.label(sc, rc, num(f.scores[i], 1), layout.MIN_TEXT, anchor="lm",
                           opacity=on * (1.0 if i < n_sc else remap(sw * T - i, 0.0, 1.0)),
                           kind="mono", ink=layout.CAPTION_INK)
            self.label(sc, self.shift(top_left, kw + KV_GAP + 20, -KV_GAP - rh / 2),
                       "q·kᵢ÷√%d" % G.D_HEAD, layout.MIN_TEXT, anchor="lm", opacity=on,
                       kind="mono", ink=layout.STRONG_INK)
        self._kv_shapes(sc, t, op, n_rows)

    def _kv_shapes(self, sc, t, op, n_rows):
        """The cache's real size and the work per step, at the right."""
        L, H, d = G.N_LAYER, G.N_HEAD, G.D_HEAD
        n = L * H * n_rows * d
        lines = [("T = %d tokens so far" % n_rows, layout.STRONG_INK),
                 ("K: %d layers × %d heads × T × %d" % (L, H, d), layout.CAPTION_INK),
                 ("= %s numbers (V: the same)" % format(n, ","), layout.CAPTION_INK)]
        if self.progress("kv_q", t) > 0:
            lines += [("", None),
                      ("each step: the new token only", layout.STRONG_INK),
                      ("through the 124 M weights: constant", layout.CAPTION_INK),
                      ("attention: %d heads × T dot products" % (L * H), layout.CAPTION_INK),
                      ("= %s multiply-adds at T = %d" % (format(L * H * n_rows * d, ","), n_rows),
                       layout.CAPTION_INK)]
        c0 = self.P(READ_X, READ_Y)
        for k, (text, ink) in enumerate(lines):
            if text:
                self.label(sc, self.shift(c0, 0.0, k * READ_DY), text, layout.MIN_TEXT,
                           anchor="lm", opacity=op, ink=ink)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``."""
        Tm, f = self.Tm, self.f
        if reset > 0 or t < Tm["greedy"][0]:
            return ('after "%s": softmax of %s logits gives a probability for every token'
                    % (PROMPT, format(G.N_VOCAB, ",")))
        if t < Tm["greedy"][0] + 2.4:
            return ('greedy decoding takes the largest every time: "%s" %.1f%% (then "%s" '
                    '%.1f%%, "%s" %.1f%%)' % (f.names[0], 100 * f.p[f.order[0]], f.names[1],
                                             100 * f.p[f.order[1]], f.names[2],
                                             100 * f.p[f.order[2]]))
        if t < Tm["temp"][0]:
            return ("a flat distribution: entropy %.1f bits, as uncertain as a choice among "
                    "2^%.1f ≈ %d equally likely tokens" % (f.entropy[1.0], f.entropy[1.0],
                                                           round(2 ** f.entropy[1.0])))
        if t < Tm["topk_cut"][0]:
            T = self.temperature(t)
            u = self.progress("temp", t)
            if u < T_PATH[1][0]:
                return ("temperature divides every logit by T before the softmax: T < 1 "
                        "sharpens the distribution, T > 1 flattens it")
            p = softmax_t(f.logits, T)
            return ('T = %.2f: "%s" %.1f%%, entropy %.1f bits (≈ %s equally likely tokens)'
                    % (T, f.names[0], 100 * p[f.order[0]], entropy_bits(p),
                       format(int(round(2 ** entropy_bits(p))), ",")))
        if t < Tm["topp"][0] - 0.8:
            return ("top-k: keep the k = %d likeliest (%.1f%% of the mass), zero the rest, "
                    "rescale to sum 1: \"%s\" %.1f%%"
                    % (TOP_K, 100 * f.k_mass, f.names[0], 100 * f.p[f.order[0]] / f.k_mass))
        if t < Tm["nucleus"][0]:
            return ("top-p: sort by probability, add up the mass, keep the smallest set that "
                    "reaches p = %.1f (the nucleus)" % TOP_P)
        if t < Tm["gen"][0] - 0.3:
            return ('here that is %d tokens; after "%s" it would be %d: the set adapts to how '
                    'sure the model is' % (f.nucleus, CONTRAST, f.nucleus_contrast))
        if t < Tm["kv"][0] - 0.6:
            k, u = self.gen_step(t)
            k = max(k, 0)
            if k == 0 and u < 0.55:
                return ("generate: sample a token, append it, run the model again; nucleus "
                        "sampling, p = %.1f, seed %d" % (TOP_P, SEED))
            return ('step %d: the nucleus keeps %d token%s; u = %.3f picks "%s" (%.1f%%)'
                    % (k + 1, f.gen["kept"][k], "" if f.gen["kept"][k] == 1 else "s",
                       f.gen["u"][k], f.gen_names[k], 100 * f.gen["probs"][k][f.gen["ids"][k]]))
        if t < Tm["kv_q"][0]:
            return ("the KV cache: a token's K and V never change, so they are stored; each "
                    "step adds the new token's row")
        return ("each step computes Q, K, V for the new token only; q against every stored k: "
                "T dot products per head")
