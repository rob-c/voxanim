"""Softmax and cross-entropy on two real sets of logits, side by side.

    left:  the MNIST MLP of the mlp-* animations (784-64-32-10) on test image
           18, a '3' it reads right at p = 0.62 (an 8 is next, 0.38):
           10 logits, one per digit
    right: GPT-2 small on "The capital of France is": 50,257 logits, one per
           token, of which the ten most probable are drawn; the target is the
           true next word " Paris", which the model ranks fifth

Both panels go through the same steps at the same time, as horizontal bars
with every number printed beside them:

1. **logits** -- the raw scores (bars from the panel's smallest score),
2. **softmax** -- exp(z − max z): every score positive, a difference of 1 a
   factor e; then divided by the sum (over 10 digits; over all 50,257
   tokens): probabilities,
3. **target** -- the one-hot target y beside the bars, and the cross-entropy
   L = −log p(target) with its value,
4. **gradient** -- ∂L/∂z = p − y drawn under each bar: the gap between p and
   y, orange where the logit must come down, blue where it must go up,
5. **temperature** -- p = softmax(z / T): T = 2 melts the bars (the entropy
   rises), T = 0.5 sharpens them.

The MLP comes from :func:`voxanim.mnist.trace`; the GPT-2 logits from
:func:`voxanim.gpt2.run` (cached by :mod:`voxanim.cache`, keyed on the prompt
and the weights' revision).  Everything else here is :func:`softmax`,
:func:`cross_entropy`, :func:`grad_logits` and :func:`entropy`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .cache import cached_npz
from .diagram import DATA, DATA_HI, EDGE, Diagram
from .easing import ease_in_out, remap, smoothstep
from .netanim import INK_HI, OFF
from .text import num

# ------------------------------------------------------------------ the maths

DIGIT = 18                  # the MNIST test image: the mlp-* animations' '3' (catalog.MLP_INDEX)
PROMPT = "The capital of France is"
TARGET = " Paris"           # the true next word; GPT-2 ranks it fifth
TOP = 10                    # GPT-2 tokens drawn: the most probable ten
T_MELT, T_SHARP = 2.0, 0.5  # the temperatures shown


def softmax(z, T=1.0):
    """``exp(z / T) / Σ exp(z / T)``, computed with the largest score subtracted
    first (which changes nothing: softmax only sees differences)."""
    z = np.asarray(z, np.float64) / T
    e = np.exp(z - z.max())
    return e / e.sum()


def cross_entropy(p, k):
    """``−log p[k]``: the loss when class ``k`` is the answer."""
    return float(-np.log(p[k]))


def grad_logits(p, k):
    """``∂(−log softmax(z)[k]) / ∂z = p − y`` with ``y`` one-hot at ``k``."""
    g = np.array(p, np.float64)
    g[k] -= 1.0
    return g


def entropy(p):
    """``−Σ p log p`` in nats: 0 when certain, ``log n`` when uniform over ``n``."""
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def gpt2_logits(prompt=PROMPT):
    """GPT-2 small's next-token logits ``(50257,)`` after ``prompt`` (cached)."""
    from . import gpt2
    settings = dict(model="gpt2-small", revision=gpt2.REVISION, prompt=prompt)

    def build():
        return dict(logits=gpt2.run(prompt).logits[-1].astype(np.float64))
    return cached_npz("softmax_gpt2", settings, build)["logits"]


@dataclass
class Logits:
    """One model's logits and what the panel shows of them.

    ``z`` is the full logit vector (10 for the MLP, 50,257 for GPT-2);
    ``rows`` the indices drawn, top to bottom, ``labels`` their names and
    ``target`` the answer's index into ``z`` (also one of ``rows``).
    """

    heading: str
    z: np.ndarray
    rows: np.ndarray
    labels: list
    target: int
    classes: str                # "digits" / "tokens": what the n classes are

    @property
    def n(self):
        """Number of classes the softmax runs over."""
        return len(self.z)

    def p(self, T=1.0):
        """The probabilities at temperature ``T``."""
        return softmax(self.z, T)

    @property
    def row_of_target(self):
        """Which drawn row is the target."""
        return int(np.nonzero(self.rows == self.target)[0][0])


def mlp_logits(index=DIGIT):
    """The MLP's ten logits for test image ``index`` (see :func:`voxanim.mnist.trace`)."""
    from . import mnist
    tr = mnist.trace(index)
    return Logits("MNIST MLP, test image %d (a %d)" % (index, tr.label), np.asarray(tr.zs[2]),
                  np.arange(10), [str(k) for k in range(10)], tr.label, "digits")


def gpt2_top(prompt=PROMPT, target=TARGET, top=TOP):
    """GPT-2's logits after ``prompt`` with its ``top`` most probable tokens as
    the rows; the target word must be a single token and among them."""
    from . import gpt2
    z = gpt2_logits(prompt)
    tok = gpt2.Tokenizer()
    ids = tok.encode(target)
    if len(ids) != 1:
        raise ValueError("%r is %d tokens, not one" % (target, len(ids)))
    rows = np.argsort(-z)[:top]
    if ids[0] not in rows:
        raise ValueError("%r is not among GPT-2's top %d after %r" % (target, top, prompt))
    labels = [tok.decode([int(i)]).strip() or repr(tok.decode([int(i)])) for i in rows]
    return Logits("GPT-2, next token after “%s”" % prompt, z, rows, labels, int(ids[0]),
                  "tokens")


# ------------------------------------------------------------------ layout (fractions)

# per panel: right edge of the row labels, the one-hot cell, bar start, bar end (= 1.0)
PANELS = ((0.065, 0.082, 0.095, 0.305), (0.61, 0.627, 0.64, 0.85))
HEAD_Y = 0.255                          # the panel headings
ROWS_Y = (0.335, 0.89)                  # first and last row centre
READOUT_Y = 0.975                       # the loss / entropy line under each panel
BAR_H, GRAD_H, GRAD_DY = 14.0, 6.0, 11.0  # bar heights and the gradient bar's drop, px
CELL = 10.0                             # a one-hot cell, px
VALUE_DX, GRAD_DX = 8.0, 70.0           # the value and the gradient columns after a bar, px
LOGIT_PAD = 0.08                        # bars start this fraction of the range below the min
BAR_FILL = (0.90, 0.91, 0.90)           # the faint outline of y = 1
TARGET_INK = (0.12, 0.36, 0.66)
TEAL_INK = tuple(float(v) for v in C.darken(DATA, 0.25))

TIMELINE = dict(intro=(0.2, 1.2), logits=(1.2, 4.4), exp=(5.0, 9.4), norm=(10.0, 14.4),
                target=(15.0, 20.0), grad=(20.6, 26.6), temp=(27.4, 43.0))
MELT = (0.0, 0.22)                      # T: 1 -> 2 over this part of "temp"
MELT_HOLD = 0.40                        # ...held until here
SHARPEN = (0.40, 0.68)                  # then 2 -> 0.5; held to the end
T_DONE = 43.4


@dataclass
class Frame:
    """What one frame shows (see :meth:`SoftmaxCrossEntropy._frame`)."""

    back: float
    intro: float
    stage: float                # 0 logits, 1 exp, 2 probabilities (fractional while morphing)
    bars: float                 # bars growing in (logits phase)
    hot: float                  # the one-hot target appearing
    loss: float                 # the loss read-out appearing
    grad: float                 # the gradient bars appearing
    temp: float                 # progress through the temperature phase
    T: float                    # the temperature on show

    @property
    def stage_i(self):
        """The stage whose numbers are printed."""
        return int(round(self.stage))


class SoftmaxCrossEntropy(Diagram):
    """The two panels through logits, softmax, target, gradient and temperature."""

    PHASES = ("logits", "softmax", "target", "gradient", "temperature")

    def __init__(self, models=None):
        super().__init__("Softmax and cross-entropy", "")
        self.models = models or (mlp_logits(), gpt2_top())
        self.hold, self.reset = 3.0, 1.2
        self.Tm = TIMELINE
        n = len(self.models[0].rows)
        self.pitch = (ROWS_Y[1] - ROWS_Y[0]) / (n - 1)
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((4, "temp"), (3, "grad"), (2, "target"), (1, "exp")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def temperature(self, u):
        """``T`` at progress ``u`` through the temperature phase: 1, melting to
        :data:`T_MELT`, held, sharpening to :data:`T_SHARP`, held."""
        if u < MELT_HOLD:
            k = ease_in_out(remap(u, *MELT))
            return float(np.exp(np.log(T_MELT) * k))
        k = ease_in_out(remap(u, *SHARPEN))
        return float(np.exp(np.log(T_MELT) + (np.log(T_SHARP) - np.log(T_MELT)) * k))

    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        stage = ease_in_out(remap(self.progress("exp", t), 0.0, 0.5)) + \
            ease_in_out(remap(self.progress("norm", t), 0.0, 0.5))
        bars = ease_in_out(self.progress("logits", t))
        hot = smoothstep(0.0, 0.3, self.progress("target", t)) * back
        loss = smoothstep(0.4, 0.7, self.progress("target", t)) * back
        temp = self.progress("temp", t)
        # p_T − y is not ∂L/∂z at temperature T (that has a 1/T), so the gradient
        # column leaves before the temperature changes
        grad = smoothstep(0.0, 0.5, self.progress("grad", t)) * back \
            * (1 - smoothstep(0.0, 0.1, temp))
        return Frame(back, intro, stage, bars, hot, loss, grad, temp, self.temperature(temp))

    # ------------------------------------------------------------------ geometry
    def row_y(self, i):
        """Layout y of row ``i``."""
        return ROWS_Y[0] + i * self.pitch

    def lengths(self, m, f):
        """Each drawn row's bar length in 0..1 and the numbers printed beside it:
        ``(lengths, values)`` for the stage on show (morphing between stages)."""
        z = m.z[m.rows]
        lo = z.min() - LOGIT_PAD * (z.max() - z.min())
        e = np.exp(z - m.z.max())
        p = m.p(f.T)[m.rows]
        stages = [(z - lo) / (z.max() - lo) * f.bars, e, p]
        k = int(np.floor(f.stage))
        if k >= 2:
            return p, p
        u = f.stage - k
        return stages[k] * (1 - u) + stages[k + 1] * u, [z, e, p][f.stage_i]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        for m, geo in zip(self.models, PANELS):
            self._panel(sc, m, geo, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _panel(self, sc, m, geo, f):
        lx, cx, x0, x1 = geo
        self.label(sc, self.P(lx - 0.02, HEAD_Y), m.heading, layout.MIN_TEXT, anchor="lm",
                   opacity=f.intro, ink=layout.STRONG_INK)
        lens, vals = self.lengths(m, f)
        p = m.p(f.T)[m.rows]
        g = grad_logits(m.p(f.T), m.target)[m.rows]
        tr = m.row_of_target
        span = self.px(self.P(x1, 0.5))[0] - self.px(self.P(x0, 0.5))[0]   # full bar, px
        # the baseline the bars grow from
        sc.line(self.P(x0, self.row_y(0) - 0.02), self.P(x0, self.row_y(len(m.rows) - 1) + 0.02),
                width=1.0, color=EDGE, opacity=0.5 * f.intro, bias=0.0)
        for i, text in enumerate(m.labels):
            y = self.row_y(i)
            is_t = i == tr
            ink = TARGET_INK if is_t and f.hot > 0.5 else layout.CAPTION_INK
            self.label(sc, self.P(lx, y), text, layout.MIN_TEXT, anchor="rm", opacity=f.intro,
                       ink=ink, kind="mono")
            self._bar(sc, x0, y, span, lens[i], DATA_HI if is_t and f.hot > 0 else DATA, f.intro)
            if f.hot > 0:
                self._onehot(sc, cx, x0, y, span, is_t, f.hot)
            if f.grad > 0:
                self._grad_bar(sc, x0, y, span, p[i], 1.0 if is_t else 0.0, f.grad)
            if lens[i] > 0 or f.stage_i > 0:
                self._numbers(sc, x1, y, vals[i], g[i], f)
        self._readout(sc, m, lx, f)

    def _bar(self, sc, x0, y, span, length, fill, op):
        w = max(span * float(length), 0.0)
        if w < 0.3:
            return
        c = self.shift(self.P(x0, y), w / 2, 0.0)
        self.rect(sc, c, w, BAR_H, fill, op, radius=2.0, lift=0.5)

    def _onehot(self, sc, cx, x0, y, span, is_t, op):
        """The one-hot cell beside the row and, on the target row, the outline of
        the bar y = 1 would be."""
        self.rect(sc, self.P(cx, y), CELL, CELL, INK_HI if is_t else OFF, op, radius=1.5,
                  lift=0.6, glow=0.3 if is_t else 0.0)
        if is_t:
            c = self.shift(self.P(x0, y), span / 2, 0.0)
            self.rect(sc, c, span, BAR_H, BAR_FILL, 0.9 * op, border=TARGET_INK, border_w=1.2,
                      radius=2.0, lift=0.3, fill_op=0.0)

    def _grad_bar(self, sc, x0, y, span, p, yk, op):
        """``p − y`` as the gap between ``p`` and ``y``: orange when the logit
        must come down (positive), blue when it must go up."""
        a, b = min(p, yk), max(p, yk)
        w = span * (b - a)
        if w < 0.3:
            return
        c = self.shift(self.P(x0, y), span * a + w / 2, GRAD_DY)
        self.rect(sc, c, w, GRAD_H, C.POSITIVE if p > yk else C.NEGATIVE, op, radius=1.5, lift=0.7)

    def _numbers(self, sc, x1, y, v, g, f):
        """The value beside the bar (logit, exp or probability) and, in the
        gradient phase, ``p − y`` after it."""
        text = num(v, 2) if f.stage_i == 0 else num(v, 3)
        self.label(sc, self.P(x1, y), text, layout.MIN_TEXT, anchor="lm", dx=VALUE_DX,
                   ink=TEAL_INK, opacity=f.intro, kind="mono")
        if f.grad > 0:
            self.label(sc, self.P(x1, y), num(g, 3), layout.MIN_TEXT, anchor="lm", dx=GRAD_DX,
                       ink=tuple(float(c) for c in C.darken(C.POSITIVE if g > 0 else C.NEGATIVE,
                                                            0.25)),
                       opacity=f.grad, kind="mono")

    def _readout(self, sc, m, lx, f):
        """The line under the panel: the bar scale, the sum, the loss, the entropy."""
        p1 = m.p()
        z = m.z[m.rows]
        lo = z.min() - LOGIT_PAD * (z.max() - z.min())
        if f.temp > 0:
            pT = m.p(f.T)
            text = "T = %s: entropy %s (uniform: %s)" % (num(f.T, 1), num(entropy(pT), 2),
                                                        num(np.log(m.n), 2))
            op = smoothstep(0.0, 0.3, f.temp)
        elif f.loss > 0:
            k = m.target
            name = m.labels[m.row_of_target]
            text = "L = −log p(%s) = −log %s = %s" % (name, num(p1[k], 3),
                                                      num(cross_entropy(p1, k), 2))
            op = f.loss
        elif f.stage_i == 2:
            top = p1[m.rows].sum()
            text = ("the %d add up to 1" % m.n if m.n == len(m.rows) else
                    "these %d hold %s; the other %s tokens %s"
                    % (len(m.rows), num(top, 2), format(m.n - len(m.rows), ","),
                       num(1 - top, 2)))
            op = smoothstep(1.6, 2.0, f.stage)
        elif f.stage_i == 1:
            text = "Σ exp(z − max z) over %s %s = %s" % (format(m.n, ","), m.classes,
                                                        num(np.exp(m.z - m.z.max()).sum(), 2))
            op = smoothstep(0.6, 1.0, f.stage) * (1 - smoothstep(1.0, 1.4, f.stage))
        else:
            text = "bars from z = %s to %s" % (num(lo, 2), num(z.max(), 2))
            op = smoothstep(0.3, 0.6, f.bars) * (1 - smoothstep(0.0, 0.4, f.stage))
        self.label(sc, self.P(lx - 0.02, READOUT_Y), text, layout.MIN_TEXT, anchor="lm",
                   opacity=op * f.back, ink=layout.STRONG_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        T, (a, b) = self.Tm, self.models
        pa, pb = a.p(), b.p()
        za, zb = a.z[a.rows], b.z[b.rows]
        if reset > 0 or t < T["logits"][0]:
            return ("softmax turns scores into probabilities; cross-entropy scores them against "
                    "the answer. Left: an MNIST MLP reading a %s. Right: GPT-2 predicting the next "
                    "word" % a.labels[a.row_of_target])
        if t < T["exp"][0]:
            return ("logits: raw scores, one per class. The MLP: 10 digits, largest %s for '%s'. "
                    "GPT-2: 50,257 tokens (the top 10 drawn), largest %s for '%s'"
                    % (num(za.max(), 2), a.labels[int(za.argmax())], num(zb.max(), 2),
                       b.labels[int(zb.argmax())]))
        if t < T["norm"][0]:
            return ("exp(z − max z): every score becomes positive and a difference of 1 a factor "
                    "e = 2.72. Subtracting the largest first changes nothing: softmax only sees "
                    "differences")
        if t < T["target"][0]:
            return ("divide by the sum: p = exp(z) / Σ exp(z). The MLP: 10 numbers adding to 1, "
                    "'%s' gets %s. GPT-2: 50,257 numbers, '%s' gets %s"
                    % (a.labels[int(pa.argmax())], num(pa.max(), 2),
                       b.labels[b.rows.tolist().index(int(pb.argmax()))], num(pb.max(), 3)))
        if t < T["grad"][0]:
            return ("the target y is one-hot: 1 at the answer, 0 elsewhere. Cross-entropy "
                    "L = −log p(answer): −log %s = %s for the %s; −log %s = %s for '%s', ranked %s"
                    % (num(pa[a.target], 3), num(cross_entropy(pa, a.target), 2),
                       a.labels[a.row_of_target], num(pb[b.target], 3),
                       num(cross_entropy(pb, b.target), 2), b.labels[b.row_of_target],
                       self._ordinal(b.row_of_target + 1)))
        if t < T["temp"][0]:
            return ("∂L/∂z = p − y: each logit's gradient is its probability, minus 1 at the "
                    "answer. Orange: this logit comes down; blue: it goes up. The %d gradients sum "
                    "to 0" % a.n)
        return self._temperature_words(f)

    @staticmethod
    def _ordinal(n):
        return "%d%s" % (n, {1: "st", 2: "nd", 3: "rd"}.get(n if n < 20 else n % 10, "th"))

    def _temperature_words(self, f):
        a, b = self.models
        if f.temp < MELT_HOLD:
            return ("temperature: p = softmax(z / T). T = %g melts the distribution: the MLP's "
                    "entropy %s → %s, GPT-2's %s → %s (every token gets a share)"
                    % (T_MELT, num(entropy(a.p()), 2), num(entropy(a.p(T_MELT)), 2),
                       num(entropy(b.p()), 2), num(entropy(b.p(T_MELT)), 2)))
        return ("T = %g sharpens it: the MLP's entropy %s, GPT-2's %s and '%s' now takes %s. "
                "T = 0 would be argmax; the loss is always computed at T = 1"
                % (T_SHARP, num(entropy(a.p(T_SHARP)), 2), num(entropy(b.p(T_SHARP)), 2),
                   b.labels[0], num(b.p(T_SHARP)[b.rows[0]], 2)))
