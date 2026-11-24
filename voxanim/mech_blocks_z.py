"""Animations 258-260 of "Blocks and code" (see :mod:`voxanim.blocks`), the middle of
"Hosting a language model", drawn from :mod:`voxanim.study_blocks_z`.

* :class:`ManyPeople` (``hosting-many-people-one-model``): four phones, four questions, one
  server holding one copy of the model (orange tiles: a corner of its real weights). Each
  person's first reply token; their next tokens one step each, the weights read four times
  (4 x 498 MB); then one ``batch_step`` with a row per person, the weights read once, and
  the same four tokens; then the weight bytes read per token as 1 to 64 people share a
  step (498 MB down to 7.8 MB).
* :class:`Memory` (``hosting-memory-per-conversation``): what the cache keeps for one token
  (the real keys and values of " server", 12 layers x key and value x 768 numbers x 4
  bytes = 73,728 bytes, which then stays on screen); one conversation's cache as a shelf of
  GPT-2's 1,024 places, 30 tokens for the real reply (2.2 MB), 75 MB when full; memory as
  a cabinet of ten 100 MB shelves: the model's 498 MB, shared, and six full conversations
  beside it, each its own; a GB of cache holds 13,563 tokens: 13 full conversations or
  452 like the real reply.
* :class:`Agent` (``hosting-a-coding-agent``): the laptop (the coding agent, the file
  parser.py, the tests) and the service (the model); four calls, each sending the whole
  conversation (25, 87, 150, 189 tokens); the model answers with text asking for a tool,
  the laptop runs it (reads the file, edits it, runs the tests) and the result joins the
  conversation; the fourth answer asks for no tool and the loop ends. The conversation is
  written for the animation (said on screen); its tokens are counted.

Hosting is not training: the loop map lights no station in any phase. The pictures are
new to the series: a phone, a laptop and a server are drawn as plain grey outlines (things,
not data), the model is always orange tiles (weights), tokens and the cache blue (data).
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    about_data,
    commas,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, GRID, PAPER, TRACK, dark, soft

DATA_INK = dark(DATA, 0.35)
LIGHT_DATA = soft(DATA, 0.45)          # every other conversation, to tell them apart
WEIGHT_INK = dark(WEIGHT, 0.35)
CASE = (0.905, 0.905, 0.89)          # a device's body
SCREEN = (0.965, 0.965, 0.955)
GPT2 = "GPT-2"


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.04):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def shown(token):
    """A token as a box shows it: its leading space dropped (" server" -> "server")."""
    return token.strip() or token


class Hosted(Blocks):
    """What the three share: the devices, the server, the model and tokens."""

    STATION = (None, None, None, None)

    def at(self, k, u, width=0.5):
        """1 once phase ``k`` has reached ``u`` (fading in over ``width`` s); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + width, getattr(self, "_t", 0.0))

    def since(self, k, u, now_k, now_u, op):
        """For :meth:`steady`: ``op`` once phase ``k`` reached ``u`` (fading in), 0 before."""
        if now_k < k:
            return 0.0
        return op * (self.at(k, u) if now_k == k else 1.0)

    # ------------------------------------------------------------------ things
    def phone(self, sc, fx, fy, op, w=22.0, h=38.0):
        """A phone: a grey body, a pale screen, a button."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, CASE, op, border=INK, border_w=1.8, radius=5, lift=0.2)
        self.rect(sc, self.shift(c, 0, -2), w - 6, h - 13, soft(DATA, 0.8), op, radius=2,
                  lift=0.22)
        self.disk(sc, self.shift(c, 0, h / 2 - 4.5), 1.8, INK, op, lift=0.25, outline=0.0)

    def laptop(self, sc, fx, fy, w, h, op):
        """A laptop: a screen ``w x h`` px centred at ``(fx, fy)`` and its base under it."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, CASE, op, border=INK, border_w=2.0, radius=6, lift=0.15)
        self.rect(sc, c, w - 10, h - 10, SCREEN, op, radius=3, lift=0.17)
        self.rect(sc, self.shift(c, 0, h / 2 + 5), w + 24, 7, AXIS, op, radius=3, lift=0.15)

    def server(self, sc, fx, fy, w, h, op):
        """A server: a tall grey box with two drive slots at the top."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, CASE, op, border=INK, border_w=2.0, radius=5, lift=0.15)
        for i in range(2):
            y = -h / 2 + 9 + 8 * i
            self.rect(sc, self.shift(c, -4, y), w - 22, 4, AXIS, op, radius=2, lift=0.17)
            self.disk(sc, self.shift(c, w / 2 - 9, y), 2.0, AXIS, op, lift=0.18, outline=0.0)

    def model(self, sc, fx, fy, side, op, read=None):
        """The model: orange tiles (a corner of its real weights) ``side`` px square; while
        ``read`` (0..1) runs, a band sweeps down them: the weights being read."""
        if op <= 0.01:
            return
        m = self.meta
        self.tiles(sc, m["model_tiles"], fx, fy, side, side, op, ink=WEIGHT, lo=m["model_lo"],
                   hi=m["model_hi"], grid=0.25, lift=0.3)
        if read is not None and 0.0 < read < 1.0:
            dy = self.frac(0, side)[1]
            y = fy - dy / 2 + dy * read
            fade = min(1.0, read / 0.1, (1 - read) / 0.1)
            self.rect(sc, self.P(fx, y), side + 10, 7, INK, op * fade, border=INK, border_w=2.2,
                      radius=3, lift=0.5, fill_op=0.0)
            self.rect(sc, self.P(fx, fy), side + 8, side + 8, WEIGHT, op * fade, border=INK,
                      border_w=2.0, radius=4, lift=0.45, fill_op=0.0)

    def token(self, sc, fx, fy, text, op, w=None, h=22.0, left=False):
        """A token: a small blue box with its text, centred on ``fx`` (``left``: starting
        there)."""
        if op <= 0.01:
            return
        w = w if w is not None else layout.text_width(text, SIZE) * self.cam.width + 16
        if left:
            fx += self.frac(w / 2)[0]
        self.box(sc, fx, fy, w, h, soft(DATA, 0.72), op, text, ink=DATA_INK, border=DATA,
                 size=SIZE, radius=5)

    def swatch(self, sc, fx, fy, ink, op, w=14.0, h=12.0):
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), w, h, ink, op, radius=2, lift=0.3)

    def seg(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.0), h, ink, op, radius=1.5,
                  lift=lift)


# ====================================================================== 258
class ManyPeople(Hosted):
    """Four people, one model: one step each, then one step for all, then 1 to 64."""

    TITLE = "Many people, one model"
    PHASES = (("four people", 12.0), ("one each", 15.0), ("one for all", 15.0),
              ("up to 64", 14.0))
    LAST = "Read once, then remember"
    NEXT = "Why a conversation has a limit"
    ROWS = (0.445, 0.535, 0.625, 0.715)     # the four people
    PHONE_X, Q_X = 0.075, 0.1               # the first token starts under the question
    ONE_X, ALL_X = 0.205, 0.3
    REPLY_DY = 0.022                         # the reply line, under the question
    SERVER = (0.425, 0.565)
    SERVER_WH = (74.0, 176.0)
    MODEL_SIDE = 52.0
    CARD = (0.425, 0.475)                    # phase 3: the four rows as one matrix
    MODEL_Y = 0.61                           # the model's tiles, in the server
    STEPS = (0.04, 0.25, 0.46, 0.67)         # phase 2: one person after another
    GO, READ, BACK = 0.07, 0.08, 0.07        # ... how long each part of a step takes

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("hs-generate", run=((0.0, 0), (0.55, 3))),
            Code("hs-generate", run=((0.0, 4),)),
            Code("hs-batch", run=((0.0, 0), (0.42, 1))),
            Code("hs-batch", run=((0.0, 0),)))
        self.EVIDENCE = (measured("four people's first tokens", on=GPT2),
                         measured("one step each, four people", on=GPT2),
                         measured("one batch_step against one step each", on=GPT2),
                         exact("%s bytes ÷ people in a step" % m["weight_bytes_text"]))

    def heading_for(self, k, u):
        return ("four people, one server",
                "one step each: the weights read 4 times",
                "one step for all: the weights read once",
                "more people per step, fewer bytes each")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Four people (questions written for the animation) ask one server. One copy of "
            "the model, %s, writes each a first token." % m["weight_text"],
            "One step per person reads all %s weights each time: %s for one more token "
            "each." % (m["numbers_text"], m["one_each_text"]),
            "One step for all four, a row per person: the weights read once, and the same "
            "next token for each as a step of their own.",
            "Weights read per token: %s for one person, %s each when %d share a step. "
            "Counts, not speeds." % (m["weight_text"], m["least_text"], m["most"]),
        ][k]

    # ------------------------------------------------------------------ timing
    def reading(self, k, u):
        """Where the band over the weights is (0..1), or None when nothing is read."""
        if k == 1:
            for s in self.STEPS:
                r = (u - s - self.GO) / self.READ
                if 0.0 < r < 1.0:
                    return r
        if k == 2:
            r = (u - 0.42) / 0.12
            if 0.0 < r < 1.0:
                return r
        return None

    def reads(self, k, u):
        """How many times the weights have been read for the next tokens so far."""
        if k < 1:
            return 0
        if k == 1:
            return sum(u >= s + self.GO for s in self.STEPS)
        return 1 if k == 2 and u >= 0.42 else (0 if k == 2 else 1)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        sx, sy = self.SERVER
        w, h = self.SERVER_WH
        self.text(sc, sx, 0.378, "the server", op, SIZE)
        self.server(sc, sx, sy, w, h, op)
        self.model(sc, sx, self.MODEL_Y, self.MODEL_SIDE, op, self.reading(k, u))
        self.text(sc, sx, 0.75, "one model", op, SIZE)
        self.text(sc, sx, 0.785, m["weight_text"], op, HEAD, strong=True)
        keep = op * self.kept(0, 2)
        if keep > 0.01:
            self.people(sc, k, u, keep)
            self.counter(sc, k, u, keep)

    def people(self, sc, k, u, op):
        m = self.meta
        for i, (y, q) in enumerate(zip(self.ROWS, m["questions"], strict=True)):
            a = self.since(0, 0.04 + 0.07 * i, k, u, op)
            self.phone(sc, self.PHONE_X, y, a)
            self.text(sc, self.Q_X, y - 0.016, q, a, SIZE, "lm")
            b = self.since(0, 0.5 + 0.05 * i, k, u, op)
            self.token(sc, self.Q_X, y + self.REPLY_DY, shown(m["first"][i]), b, left=True)
            # one step each: phase 2, one person after another
            if k >= 1:
                s = self.STEPS[i]
                c = op if k > 1 else on(u, s + self.GO + self.READ + self.BACK, op)
                self.token(sc, self.ONE_X, y + self.REPLY_DY, shown(m["next"][i]), c)
            if k >= 2:
                c = op if k > 2 else on(u, 0.68, op)
                self.token(sc, self.ALL_X, y + self.REPLY_DY, shown(m["next"][i]), c)
                self.text(sc, (self.ONE_X + self.ALL_X) / 2 + 0.002, y + self.REPLY_DY, "=", c,
                          SIZE, ink=INK)
            # the line to the server, behind everything
            self.arrow(sc, (self.ALL_X + 0.035, y + self.REPLY_DY),
                       (sx_left(self) - 0.004, y + self.REPLY_DY), GRID, a * 0.9, width=1.4,
                       head=False, lift=0.0)
        h1 = self.since(1, 0.03, k, u, op)
        self.text(sc, self.ONE_X, 0.395, "one each", h1, SIZE)
        h2 = self.since(2, 0.6, k, u, op)
        self.text(sc, self.ALL_X, 0.395, "together", h2, SIZE)

    def counter(self, sc, k, u, op):
        m = self.meta
        a = self.since(1, 0.03, k, u, op)
        if a <= 0.01:
            return
        self.text(sc, 0.06, 0.79, "weights read for the next tokens", a, SIZE, "lm")
        n = self.reads(k, u)
        if k == 1:
            text = ("%d × %s" % (n, m["weight_text"]) if n < 4 else
                    "4 × %s = %s" % (m["weight_text"], m["one_each_text"]))
        else:
            text = ("1 × %s, for all 4" if n else "0 × %s") % m["weight_text"]
        self.text(sc, 0.06, 0.832, text, a, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 1-3
    def phase_1(self, sc, u, op):
        """The people and their first tokens are in ``steady``; here, their questions
        travel to the server and the first tokens come back."""
        sx = sx_left(self)
        for i, y in enumerate(self.ROWS):
            y = y + self.REPLY_DY
            self.pulse(sc, (self.Q_X + 0.08, y), (sx, y), (u - 0.28 - 0.02 * i) / 0.1, DATA, op)
            self.pulse(sc, (sx, y), (self.Q_X + 0.03, y), (u - 0.4 - 0.05 * i) / 0.1, DATA, op)

    def phase_2(self, sc, u, op):
        sx = sx_left(self)
        for y, s in zip(self.ROWS, self.STEPS, strict=True):
            y = y + self.REPLY_DY
            self.pulse(sc, (self.Q_X + 0.03, y), (sx, self.SERVER[1]),
                       (u - s) / self.GO, DATA, op)
            b = s + self.GO + self.READ
            self.pulse(sc, (sx, self.SERVER[1]), (self.ONE_X, y), (u - b) / self.BACK, DATA, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        cx, cy = self.CARD
        sx = sx_left(self)
        # the four newest tokens go in together: one matrix, a row per person
        for i, y in enumerate(self.ROWS):
            self.pulse(sc, (self.Q_X + 0.03, y + self.REPLY_DY), (sx, y + self.REPLY_DY),
                       (u - 0.04 - 0.01 * i) / 0.12, DATA, op)
        a = on(u, 0.17, op)
        if a > 0.01:
            self.card(sc, cx, cy, 50, 38, a, lift=0.3)
            for i in range(m["n"]):
                self.rect(sc, self.P(cx, cy + self.frac(0, -7.5 + 6.5 * i)[1]), 38, 4.0, DATA,
                          a, radius=1.5, lift=0.33)
            self.text(sc, cx, cy + 0.058, m["batch_shape"], a, SIZE, ink=DATA_INK)
        for y in self.ROWS:
            self.pulse(sc, (sx, y + self.REPLY_DY), (self.ALL_X, y + self.REPLY_DY),
                       (u - 0.56) / 0.12, DATA, op)
        self.text(sc, 0.06, 0.872, "and the same 4 tokens", on(u, 0.75, op), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 4: 1 to 64
    def phase_4(self, sc, u, op):
        m = self.meta
        top, step = 0.445, 0.058
        x0, full = 0.13, 0.17
        a = on(u, 0.05, op)
        self.text(sc, 0.06, 0.39, "people", a, SIZE, "lm")
        self.text(sc, x0, 0.39, "weights read per token", a, SIZE, "lm")
        for i, g in enumerate(m["groups"]):
            y = top + step * i
            b = on(u, 0.1 + 0.07 * i, op)
            self.text(sc, 0.1, y, str(g["people"]), b, SIZE, "rm", ink=INK)
            self.seg(sc, x0, y, full * g["bytes"] / m["weight_bytes"], 13.0, WEIGHT, b)
            self.text(sc, x0 + full * g["bytes"] / m["weight_bytes"] + 0.01, y, g["text"], b,
                      SIZE, "lm", ink=INK)


def sx_left(anim):
    """The left edge of the server box (layout x)."""
    return anim.SERVER[0] - anim.frac(anim.SERVER_WH[0] / 2)[0]


# ====================================================================== 259
class Memory(Hosted):
    """One token's cache; a conversation's shelf; the model and the caches; a GB."""

    TITLE = "Why a conversation has a limit"
    PHASES = (("one token", 14.0), ("one conversation", 14.0), ("beside the model", 14.0),
              ("a GB of cache", 14.0))
    LAST = "Many people, one model"
    NEXT = "A coding agent's loop"
    TOKEN = (0.098, 0.64)
    KEY_X, VAL_X = 0.185, 0.258              # the 12 x 2 strips
    STRIP_W, STRIP_H = 60.0, 9.0
    LAYER_TOP, LAYER_STEP = 0.48, 0.03
    RIGHT = 0.325                            # the right-hand column
    PER_Y = 0.80                             # the bytes per token, kept all through
    SHELF = (0.07, 0.45, 0.55)               # 259 phase 2: x0, x1, y
    CAB = (0.07, 0.29, 0.455, 0.04)          # the cabinet: x0, x1, top shelf, step
    SHELVES = 10
    SHELF_BYTES = 10 ** 8

    def setup(self):
        m = self.meta = self.study.metadata
        self.keys = np.asarray(m["keys"], float)
        self.vals = np.asarray(m["values"], float)
        self.CODE = (
            Code("hs-cache", run=((0.0, 2), (0.15, 4))),
            Code("hs-generate", run=((0.0, 0), (0.06, 4))),
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-handler", run=((0.0, 6),)))
        self.EVIDENCE = (exact("12 layers × 2 × 768 numbers × 4 bytes"),
                         measured("the reply's cache, %s a layer" % m["cache_shape"], on=GPT2),
                         exact("%s bytes beside %s" % (m["full_bytes_text"],
                                                       commas(m["weights"]))),
                         exact("10⁹ bytes ÷ %s = %s tokens" % (m["per_token_text"],
                                                               m["tokens_per_gb_text"])))

    def heading_for(self, k, u):
        return ("the cache: what is kept of each token",
                "one conversation's cache grows",
                "the model shared, each cache its own",
                "a long conversation takes more room")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "For every token it has read, the server keeps a key and a value in each of "
            "%d layers: %s bytes." % (m["layers"], m["per_token_text"]),
            "The real reply's conversation keeps %d tokens, %s. GPT-2 has room for %s tokens: "
            "%s when full." % (m["reply_tokens"], m["reply_text"], m["context_text"],
                              m["full_text"]),
            "The model's %s is read by everyone; each conversation's cache is its own, up to "
            "%s." % (m["weights_text"], m["full_text"]),
            "A GB holds %s tokens of cache: %d full conversations or %d short ones. Real "
            "services run far larger models, their sizes mostly not published."
            % (m["tokens_per_gb_text"], m["full_per_gb"], m["short_per_gb"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a = self.since(0, 0.62, k, u, op)
        self.card(sc, self.RIGHT - 0.017, self.PER_Y + 0.021, 16, 36, a, lift=0.3)
        self.text(sc, self.RIGHT, self.PER_Y, "kept per token", a, SIZE, "lm")
        self.text(sc, self.RIGHT, self.PER_Y + 0.042, "%s bytes" % m["per_token_text"], a, HEAD,
                  "lm", strong=True)
        c = op * self.kept(2, 3)
        if c > 0.01:
            self.cabinet(sc, k, u, c)

    # ------------------------------------------------------------------ 1: one token
    def phase_1(self, sc, u, op):
        m = self.meta
        tx, ty = self.TOKEN
        a = on(u, 0.03, op)
        self.text(sc, tx, ty - 0.045, "one token", a, SIZE)
        self.token(sc, tx, ty, shown(m["token"]), a)
        b = on(u, 0.12, op)
        self.text(sc, self.KEY_X, self.LAYER_TOP - 0.035, "key", b, SIZE)
        self.text(sc, self.VAL_X, self.LAYER_TOP - 0.035, "value", b, SIZE)
        last = self.LAYER_TOP + self.LAYER_STEP * (m["layers"] - 1)
        self.text(sc, self.KEY_X - 0.04, self.LAYER_TOP, "layer 1", b, SIZE, "rm")
        self.text(sc, self.KEY_X - 0.04, last, "layer %d" % m["layers"], b, SIZE, "rm")
        self.arrow(sc, (tx + 0.04, ty), (self.KEY_X - 0.038, ty), AXIS, b,
                   u=eased(u, 0.12, 0.18), width=1.8)
        hi = 1.5 * float(np.mean(np.abs(np.r_[self.keys.ravel(), self.vals.ravel()])))
        w = self.STRIP_W
        for i in range(m["layers"]):
            y = self.LAYER_TOP + self.LAYER_STEP * i
            r = on(u, 0.14 + 0.012 * i, op)
            for x, vec in ((self.KEY_X, self.keys[i]), (self.VAL_X, self.vals[i])):
                self.strip(sc, np.abs(vec), x, y, w, self.STRIP_H, r, lo=0.0, hi=hi)
        self.text(sc, (self.KEY_X + self.VAL_X) / 2, last + 0.04, "768 numbers each", b, SIZE)
        # the arithmetic, one factor at a time
        rows = ("%d layers" % m["layers"], "× 2: key, value", "× %d numbers" % m["width"],
                "× %d bytes" % m["bytes_per_number"])
        for i, s in enumerate(rows):
            self.text(sc, self.RIGHT, 0.5 + 0.045 * i, s, on(u, 0.3 + 0.07 * i, op), SIZE, "lm")

    # ------------------------------------------------------------------ 2: a conversation
    def phase_2(self, sc, u, op):
        m = self.meta
        x0, x1, y = self.SHELF
        a = on(u, 0.03, op)
        self.phone(sc, x0 + 0.012, y - 0.085, a, w=16.0, h=28.0)
        self.text(sc, x0 + 0.03, y - 0.085, "one conversation", a, SIZE, "lm")
        # the shelf: a board, its ends, its 1,024 places marked
        self.seg(sc, x0, y + 0.024, x1 - x0, 5.0, AXIS, a)
        for x, s in ((x0, "0"), ((x0 + x1) / 2, "512"), (x1, m["context_text"])):
            self.text(sc, x, y + 0.06, s, a, SIZE)
        self.text(sc, x1, y - 0.045, "tokens", a, SIZE, "rm", ink=QUIET)
        grow = eased(u, 0.08, 0.3) * m["reply_tokens"]
        full = eased(u, 0.5, 0.75)
        n = grow + (m["context"] - m["reply_tokens"]) * full
        if n > 0.05:
            self.seg(sc, x0, y, (x1 - x0) * n / m["context"], 20.0, DATA, a)
        steps = ((0.0, "%d tokens: %d asked, %d written" % (m["reply_tokens"],
                                                           m["prompt_tokens"], m["written"])),
                 (0.5, "%s tokens: full" % m["context_text"]))
        self.running(sc, x0, y + 0.115, steps, u, on(u, 0.3, op), size=SIZE)
        self.running(sc, x0, y + 0.16, ((0.0, m["reply_text"]), (0.5, m["full_text"])), u,
                     on(u, 0.3, op), size=HEAD)
        b = on(u, 0.8, op)
        self.rect(sc, self.P(x1, y), 4.0, 32.0, INK, b, radius=1, lift=0.6)
        self.text(sc, x0, y + 0.22, "%s has %s places: the limit" % (GPT2, m["context_text"]),
                  b, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 3-4: the cabinet
    def cabinet(self, sc, k, u, op):
        """Memory as ten shelves of 100 MB; what fills them comes from the phase."""
        x0, x1, top, step = self.CAB
        a = op * (self.at(2, 0.03) if k == 2 else 1.0)
        self.text(sc, x0, top - 0.05, "1 GB of memory: 10 shelves", a, SIZE, "lm")
        for i in range(self.SHELVES):
            self.seg(sc, x0, top + step * i + 0.013, x1 - x0, 3.0, TRACK, a, lift=0.2)
        self.text(sc, x1, top + step * self.SHELVES - 0.005, "100 MB a shelf", a, SIZE, "rm",
                  ink=QUIET)

    def fill(self, sc, items, op):
        """Lay ``(bytes, ink, opacity)`` items along the shelves, one after another, each
        cut where a shelf ends; a gap of two pixels between items."""
        x0, x1, top, step = self.CAB
        at = 0.0
        gap = self.frac(2.0)[0]
        for n, ink, a in items:
            start, end = at, at + n
            at = end
            if a <= 0.01:
                continue
            while start < end - 1:
                shelf = int(start // self.SHELF_BYTES)
                if shelf >= self.SHELVES:
                    break
                stop = min(end, (shelf + 1) * self.SHELF_BYTES)
                fa = (start - shelf * self.SHELF_BYTES) / self.SHELF_BYTES
                fb = (stop - shelf * self.SHELF_BYTES) / self.SHELF_BYTES
                w = (x1 - x0) * (fb - fa) - (gap if stop == end else 0.0)
                self.seg(sc, x0 + (x1 - x0) * fa, top + step * shelf, w, 14.0, ink, a)
                start = stop

    def legend(self, sc, rows, u, op, y0=0.47):
        """Rows of ``(u, swatch ink or None, text, size)`` in the right-hand column."""
        y = y0
        for at, ink, s, size in rows:
            a = on(u, at, op)
            if ink is not None:
                self.swatch(sc, self.RIGHT + 0.008, y, ink, a)
            self.text(sc, self.RIGHT + (0.025 if ink is not None else 0.0), y, s, a, size, "lm",
                      ink=INK if size != SIZE else None)
            y += 0.042 if size == SIZE else 0.05

    def phase_3(self, sc, u, op):
        m = self.meta
        model = on(u, 0.08, op)
        items = [(m["weights"], WEIGHT, model)]
        for i in range(m["beside"]):
            items.append((m["full"], DATA if i % 2 == 0 else LIGHT_DATA,
                          on(u, 0.42 + 0.05 * i, op)))
        self.fill(sc, items, op)
        self.legend(sc, ((0.08, WEIGHT, "the model: %s" % m["weights_text"], SIZE),
                         (0.18, None, "shared by everyone", SIZE),
                         (0.42, DATA, "a full conversation", SIZE),
                         (0.42, None, "%s, its own" % m["full_text"], SIZE),
                         (0.75, None, "%d fit beside the model" % m["beside"], SIZE)), u, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        full = on(u, 0.08, op) * (1.0 - smoothstep(0.54, 0.58, u))
        items = [(m["full"], DATA if i % 2 == 0 else LIGHT_DATA,
                  full * smoothstep(0.08 + 0.02 * i, 0.1 + 0.02 * i, u))
                 for i in range(m["full_per_gb"])]
        self.fill(sc, items, op)
        short = on(u, 0.54, op)
        if short > 0.01:
            self.fill(sc, [(m["reply_bytes"], DATA if i % 2 == 0 else LIGHT_DATA, short)
                           for i in range(m["short_per_gb"])], op)
        self.legend(sc, ((0.04, None, "1 GB of cache:", SIZE),
                         (0.04, None, "%s tokens" % m["tokens_per_gb_text"], HEAD)), u, op)
        steps = ((0.0, "%d conversations" % m["full_per_gb"]),
                 (0.56, "or %d conversations" % m["short_per_gb"]))
        self.running(sc, self.RIGHT, 0.62, steps, u, on(u, 0.3, op), size=SIZE, ink=INK)
        steps = ((0.0, "of %s tokens" % m["context_text"]),
                 (0.56, "of %d tokens" % m["reply_tokens"]))
        self.running(sc, self.RIGHT, 0.662, steps, u, on(u, 0.3, op), size=SIZE)


# ====================================================================== 260
SIDE_INK = dict(device=DATA, service=WEIGHT)


class Agent(Hosted):
    """The laptop and the service; four calls, each with the whole conversation."""

    TITLE = "A coding agent's loop"
    PHASES = (("two sides", 11.0), ("read", 12.0), ("edit", 11.0), ("test", 11.0),
              ("answer", 12.0))
    STATION = (None, None, None, None, None)
    LAST = "Why a conversation has a limit"
    NEXT = "Thinking: tokens it writes for itself"
    LAPTOP = (0.15, 0.415)
    LAPTOP_WH = (124.0, 70.0)
    SERVICE = (0.418, 0.42)
    SERVICE_WH = (66.0, 76.0)
    ITEMS = ("agent", "parser.py", "tests")
    LABEL_Y = 0.517
    SEND_Y, BACK_Y = 0.392, 0.447
    MID = (0.228, 0.372)                     # the arrows between laptop and service
    LABEL_X, TEXT_X = 0.06, 0.135
    SERVICE_ROWS = (0.597, 0.638, 0.679)
    LAPTOP_ROWS = (0.723, 0.764, 0.805)
    BAR = (0.06, 0.44, 0.878)                # the conversation: x0, x1 (189 tokens), y
    BAR_LABEL_Y = 0.847
    # a call: sent at SEND, read by the model, written, sent back, then the tool runs
    SEND, READ, WRITE, BACK, TOOL, RESULT = 0.04, 0.2, 0.34, 0.42, 0.55, 0.65

    def setup(self):
        m = self.meta = self.study.metadata
        self.calls = m["calls"]
        send = Code("hs-request", run=((0.0, 2), (0.05, 4), (0.12, 5), (self.BACK, 6)))
        self.CODE = (Code("hs-request", run=((0.0, 0),)), send, send, send,
                     Code("hs-handler", run=((0.0, 2), (self.WRITE, 6))))
        self.EVIDENCE = (about_data("a conversation written for the animation"),) + tuple(
            measured("call %d's tokens, its tokenizer" % c["call"], on=GPT2)
            for c in self.calls)

    def heading_for(self, k, u):
        return ("your laptop, and the service",
                "call 1: the model asks to read a file",
                "call 2: it asks to edit the file",
                "call 3: it asks to run the tests",
                "call 4: no tool, just the answer")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = self.calls
        if k == 0:
            return ("The agent, your files and the tests are on your laptop; the service only "
                    "runs the model. A conversation written for the animation.")
        if k == 1:
            return ("The agent sends the whole conversation, %d tokens. The model answers with "
                    "text asking for a tool; the laptop reads the file." % c[0]["tokens_sent"])
        if k == 2:
            return ("The file's text joined the conversation: %d tokens sent. The laptop "
                    "makes the edit the model asked for." % c[1]["tokens_sent"])
        if k == 3:
            return ("%d tokens sent. The tests run on your laptop, not on the service: %s."
                    % (c[2]["tokens_sent"], c[2]["result"]))
        return ("%d tokens sent; no tool asked for, so the loop ends. Every call carried the "
                "whole conversation: %d tokens in all." % (c[3]["tokens_sent"], m["total_sent"]))

    # ------------------------------------------------------------------ where things are
    def item_y(self, i):
        return self.LAPTOP[1] - 0.038 + 0.038 * i

    def lit_item(self, k, u):
        """Which item on the laptop is in use, and how lit (0..1)."""
        if k == 0:
            return 0, on(u, 0.5, 1.0)
        c = self.calls[k - 1]
        if c["tool"] is None:
            return 0, on(u, self.TOOL, 1.0)
        return (2 if c["tool"] == "run" else 1), on(u, self.TOOL, 1.0)

    def bar_shares(self, k, u):
        """``[(tokens, side)]`` of the conversation as it stands: the call's own, then what
        joins it (the model's text, then the tool's result) taken from the next call."""
        c = self.calls
        if k == 0:
            return list(zip(c[0]["shares"], c[0]["sides"], strict=True))
        call = c[k - 1]
        if k - 1 + 1 < len(c):
            nxt = c[k]
            n = len(call["shares"]) + (u >= self.BACK + 0.08) + (u >= self.RESULT + 0.05)
            if n > len(call["shares"]):
                return list(zip(nxt["shares"][:n], nxt["sides"][:n], strict=True))
        return list(zip(call["shares"], call["sides"], strict=True))

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        lx, ly = self.LAPTOP
        a = self.since(0, 0.03, k, u, op)
        self.laptop(sc, lx, ly, *self.LAPTOP_WH, a)
        which, lit = self.lit_item(k, u)
        for i, name in enumerate(self.ITEMS):
            y = self.item_y(i)
            ix = lx - 0.045
            g = lit if i == which else 0.0
            if i == 1:
                self.card(sc, ix, y, 14, 17, a, lift=0.3)
            elif i == 2:
                self.rect(sc, self.P(ix, y), 15, 15, soft(AXIS, 0.6), a, border=AXIS,
                          border_w=1.4, radius=2, lift=0.3)
            else:
                self.rect(sc, self.P(ix, y), 15, 15, PAPER, a, border=INK, border_w=1.8,
                          radius=7, lift=0.3)
            if g > 0.01:
                self.rect(sc, self.P(lx, y), self.LAPTOP_WH[0] - 12, 22, INK, a * g,
                          border=INK, border_w=2.2, radius=4, lift=0.4, fill_op=0.0)
            self.text(sc, ix + 0.02, y, name, a, SIZE, "lm", ink=INK if g > 0.5 else None)
        self.text(sc, lx, self.LABEL_Y, "your laptop", a, SIZE, strong=True)
        sx, sy = self.SERVICE
        b = self.since(0, 0.16, k, u, op)
        self.server(sc, sx, sy, *self.SERVICE_WH, b)
        self.model(sc, sx, sy + 0.02, 36.0, b, self.reading(k, u))
        self.text(sc, sx, self.LABEL_Y, "the service", b, SIZE, strong=True)
        self.text(sc, sx, self.LABEL_Y + 0.035, "only writes text", self.since(0, 0.3, k, u, op),
                  SIZE)
        # the cable between them, and the conversation as it stands
        c = self.since(0, 0.3, k, u, op)
        x0, x1 = self.MID
        self.arrow(sc, (x0, self.SEND_Y), (x1, self.SEND_Y), AXIS, c, width=1.8)
        self.arrow(sc, (x1, self.BACK_Y), (x0, self.BACK_Y), AXIS, c, width=1.8)
        d = self.since(0, 0.7, k, u, op)
        self.conversation(sc, k, u, d)

    def reading(self, k, u):
        if k == 0:
            return None
        r = (u - self.READ) / (self.WRITE - self.READ)
        return r if 0.0 < r < 1.0 else None

    def conversation(self, sc, k, u, op):
        if op <= 0.01:
            return
        m = self.meta
        x0, x1, y = self.BAR
        scale = (x1 - x0) / m["most"]
        at = x0
        gap = self.frac(2.0)[0]
        for n, side in self.bar_shares(k, u):
            self.seg(sc, at, y, n * scale - gap, 16.0, SIDE_INK[side], op)
            at += n * scale
        self.text(sc, x0, self.BAR_LABEL_Y, "the whole conversation, sent each call", op, SIZE,
                  "lm", ink=INK)
        # where each call's sending ended: a tick and its tokens, kept
        for c in self.calls[:k]:
            a = op * (self.at(c["call"], self.SEND) if c["call"] == k else 1.0)
            x = x0 + c["tokens_sent"] * scale - gap / 2
            self.rect(sc, self.P(x, y), 2.4, 26.0, INK, a, radius=1, lift=0.5)
            self.text(sc, x, y + 0.033, str(c["tokens_sent"]), a, SIZE, ink=INK)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        m = self.meta
        a = on(u, 0.5, op)
        self.text(sc, self.LABEL_X, self.LAPTOP_ROWS[0], "you", a, SIZE, "lm", ink=DATA_INK)
        for i, s in enumerate(m["asked_lines"]):
            self.text(sc, self.TEXT_X, self.LAPTOP_ROWS[i], s, on(u, 0.52, op), SIZE, "lm",
                      ink=INK)

    def call(self, sc, k, u, op):
        """Call ``k``: the conversation sent, the model's text back, the tool run."""
        c = self.calls[k - 1]
        x0, x1 = self.MID
        mid = (x0 + x1) / 2
        for j in range(3):
            self.pulse(sc, (x0, self.SEND_Y), (x1, self.SEND_Y),
                       (u - self.SEND - 0.025 * j) / 0.12, DATA, op)
        self.text(sc, mid, self.SEND_Y - 0.03, "all %d tokens" % c["tokens_sent"],
                  on(u, self.SEND, op), SIZE, ink=DATA_INK)
        w = on(u, self.WRITE, op)
        self.text(sc, self.LABEL_X, self.SERVICE_ROWS[0], "service", w, SIZE, "lm",
                  ink=WEIGHT_INK)
        for i, s in enumerate(c["reply_lines"]):
            self.label(sc, self.P(self.TEXT_X, self.SERVICE_ROWS[i]), s, SIZE, anchor="lm",
                       opacity=on(u, self.WRITE + 0.02 * i, op), ink=WEIGHT_INK, kind="mono")
        self.pulse(sc, (x1, self.BACK_Y), (x0, self.BACK_Y), (u - self.BACK) / 0.1, WEIGHT, op)
        self.text(sc, mid, self.BACK_Y + 0.03, "text", on(u, self.BACK, op), SIZE,
                  ink=WEIGHT_INK)
        t = on(u, self.TOOL, op)
        self.text(sc, self.LABEL_X, self.LAPTOP_ROWS[0], "laptop", t, SIZE, "lm", ink=DATA_INK)
        if c["tool"] is None:
            self.text(sc, self.TEXT_X, self.LAPTOP_ROWS[0], "shows you the answer", t, SIZE,
                      "lm", strong=True)
            self.text(sc, self.TEXT_X, self.LAPTOP_ROWS[1], "no tool: the loop ends",
                      on(u, self.RESULT, op), SIZE, "lm")
            return
        self.text(sc, self.TEXT_X, self.LAPTOP_ROWS[0], c["action"], t, SIZE, "lm", strong=True)
        for i, s in enumerate(c["result_lines"]):
            self.label(sc, self.P(self.TEXT_X, self.LAPTOP_ROWS[1 + i]), s, SIZE, anchor="lm",
                       opacity=on(u, self.RESULT + 0.02 * i, op), ink=DATA_INK, kind="mono")

    def phase_2(self, sc, u, op):
        self.call(sc, 1, u, op)

    def phase_3(self, sc, u, op):
        self.call(sc, 2, u, op)

    def phase_4(self, sc, u, op):
        self.call(sc, 3, u, op)

    def phase_5(self, sc, u, op):
        self.call(sc, 4, u, op)


ANIMATIONS = {"hosting-many-people-one-model": ManyPeople,
              "hosting-memory-per-conversation": Memory,
              "hosting-a-coding-agent": Agent}
