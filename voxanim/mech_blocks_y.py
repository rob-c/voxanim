"""Animations 255-257 of "Hosting a language model" in "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_y`.

* :class:`TextToTokens` (``hosting-text-to-tokens``): on the server, the example request
  (82 bytes of JSON) becomes one text, "User: What is a server?" and "Assistant:" (34
  characters); GPT-2's tokenizer cuts it into 10 tokens, drawn as blue tiles (a space shown
  as ␣, the new line as ↵); each tile gets its number in GPT-2's list of 50,257 (the colon
  is 25 both times); then what is counted: 10 read, at most 20 written, GPT-2's limit of
  1,024.
* :class:`OneTokenAtATime` (``hosting-one-token-at-a-time``): the server (GPT-2) and a
  phone joined by the internet; at each step the five likeliest next tokens as bars with
  their probabilities, the likeliest ringed and sent along the cable to the phone at once,
  the text so far one token longer; steps 2-5 the same way, then the rest of the 20 tokens
  stream in. The reply is GPT-2's real one, poor, stopping mid-sentence at max_tokens.
* :class:`ReadOnceRemember` (``hosting-read-once-remember``): a staircase of the tokens
  run at each step without a cache (the prompt, then 11, 12, ... 30: 410 for the 20 reply
  steps); the cache as 12 shelves (one per layer) of every token's keys and values, shaded
  by their real sizes; with the cache only the new token runs (the staircase turns grey
  but for its last tile, the shelves grow a column a step: 20 tokens run); the same 20
  tokens, scores within 0.0000068; 2.5 billion multiply-adds instead of 35.7 billion.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). Hosting is not training: the loop map lights
no station in any phase.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .blocks import BIG, DATA, HEAD, INK, QUIET, SIZE, Blocks, Code, about_data, exact, measured
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, DIM, GRID, PAPER, TRACK, dark, soft

DATA_INK = dark(DATA, 0.3)
TILE = soft(DATA, 0.82)             # a token: a small blue card
BODY = C.rgb("#3d4147")             # a phone's case
SCREEN = C.rgb("#fbfbf8")
CASE = C.rgb("#d9dadc")             # the server's case
MONO_STEP = 0.036                   # lines of text in a card


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def width_of(s, size=SIZE, kind="regular"):
    """``s``'s width as a fraction of the frame."""
    return layout.text_width(s, size, kind)


def along(points, s):
    """The point a fraction ``s`` (0..1) of the way along the broken line ``points``."""
    pts = np.asarray(points, float)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    d = float(np.clip(s, 0.0, 1.0)) * seg.sum()
    for i, n in enumerate(seg):
        if d <= n or i == len(seg) - 1:
            return tuple(pts[i] + (pts[i + 1] - pts[i]) * min(d / max(n, 1e-9), 1.0))
        d -= n
    return tuple(pts[-1])


class Hosted(Blocks):
    """What the three share: the server, the phone, the cable and a token."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over half a second); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def server(self, sc, fx, fy, op, w=64.0, h=46.0, lit=0.0):
        """The server: a computer's case with three drive bays and their lights (lit blue
        while it works)."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, CASE, op, border=dark(AXIS, 0.1), border_w=1.8, radius=5,
                  lift=0.3)
        for i in range(3):
            y = -h / 2 + (i + 0.5) * h / 3
            self.rect(sc, self.shift(c, -4, y), w - 20, h / 3 - 6, PAPER, op, border=GRID,
                      border_w=1.0, radius=2, lift=0.32)
            self.disk(sc, self.shift(c, w / 2 - 8, y), 2.6, DATA if lit > 0.5 else AXIS, op,
                      lift=0.4, outline=0.0)

    def phone(self, sc, fx, fy, w, h, op):
        """A phone: a dark case, a pale screen, a speaker and a button."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, BODY, op, radius=15, lift=0.2)
        self.rect(sc, c, w - 10, h - 34, SCREEN, op, radius=5, lift=0.22)
        self.rect(sc, self.shift(c, 0, -h / 2 + 9), 26, 4, AXIS, op, radius=2, lift=0.23)
        self.disk(sc, self.shift(c, 0, h / 2 - 9), 4.5, AXIS, op, lift=0.25, outline=0.0)

    def token(self, sc, fx, fy, text, op, h=24.0, lift=0.4, anchor="mm"):
        """A token: a small blue card with its text. Returns its width (frame fraction)."""
        w = max(28.0, width_of(text) * self.cam.width + 14.0)
        dw = self.frac(w)[0]
        cx = fx + dw / 2 if anchor == "lm" else fx
        if op > 0.01:
            self.rect(sc, self.P(cx, fy), w, h, TILE, op, border=DATA, border_w=1.6, radius=5,
                      lift=lift)
            self.text(sc, cx, fy, text, op, SIZE, ink=INK)
        return dw

    def token_w(self, text):
        return self.frac(max(28.0, width_of(text) * self.cam.width + 14.0))[0]

    def mono_card(self, sc, fx, fy, lines, op, w):
        """Lines of text (mono) in a blue data card ``w`` (frame fraction) wide, its first
        line at ``fy``, starting at ``fx``."""
        h = MONO_STEP * len(lines) + 0.04
        self.card(sc, fx + w / 2, fy - 0.022 + h / 2, self.span(w)[0], self.span(0, h)[1], op)
        for i, s in enumerate(lines):
            self.label(sc, self.P(fx + 0.012, fy + 0.012 + MONO_STEP * i), s, SIZE, anchor="lm",
                       opacity=op, ink=INK, kind="mono")

    def row_tiles(self, sc, x0, y, rgb, cw, ch, op, lift=0.3):
        """A row of square-ish tiles (``rgb`` an ``(n, 3)`` array), ``cw x ch`` px each,
        starting at layout ``x0``."""
        n = len(rgb)
        if op <= 0.01 or n == 0:
            return
        w = cw * n
        draw_image(self, sc, self.P(x0 + self.frac(w / 2)[0], y), np.asarray(rgb, np.float32)
                   [None], w, ch, op, grid=0.25, lift=lift)


# ====================================================================== 255
class TextToTokens(Hosted):
    """The request becomes one text; the text becomes 10 tokens; each token a number; what
    is counted."""

    TITLE = "Text becomes tokens"
    PHASES = (("one text", 13.0), ("tokens", 15.0), ("numbers", 14.0), ("counted", 14.0))
    LAST = "The model is a file of numbers"
    NEXT = "One token at a time"
    STATION = (None, None, None, None)
    X0 = 0.065                      # the picture's left edge
    JSON_Y = 0.505                  # phase 1: the request's first line
    LOW_Y, TOP_Y = 0.72, 0.46       # the text's first line: under the request, then at top
    ROWS = (0.6, 0.7)               # the two rows of tokens
    ID_DY = 0.04                    # each token's id, under it
    FOOT = (0.8, 0.842)             # what is said under the tokens
    TILES_AT = (0.08, 0.52)         # phase 2: the tokens appear one by one
    IDS_AT = (0.08, 0.45)           # phase 3: their ids

    def setup(self):
        m = self.meta = self.study.metadata
        lines = list(m["ids_lines"])
        ids_out = "\n".join(["ids = " + lines[0]] + ["      " + s.strip() for s in lines[1:]])
        self.CODE = (
            Code("hs-handler", pick=(0, 1, 2), run=((0.0, 0), (0.15, 1), (0.45, 2))),
            Code(("hs-load", "hs-tokens"), run=((0.0, 1), (0.08, 2))),
            Code(("hs-load", "hs-tokens"), run=((0.0, 2),), out=((0.5, ids_out),)),
            Code("hs-handler", pick=(1, 2, 3, 6), run=((0.0, 2), (0.55, 3))))
        self.EVIDENCE = (about_data("the example request, written for this"),
                         measured("GPT-2's tokenizer on the text", on="GPT-2"),
                         measured("GPT-2's token ids", on="GPT-2"),
                         exact("tokens read and written, GPT-2's limit"))
        # the tiles' places: five a row, from the left
        self.places = []
        for r, row in enumerate((m["shown"][:5], m["shown"][5:])):
            x = self.X0
            for s in row:
                w = self.token_w(s)
                self.places.append((x + w / 2, self.ROWS[r]))
                x += w + self.frac(9.0)[0]

    def heading_for(self, k, u):
        return ("the server makes one text of the request",
                "the text, cut into tokens",
                "each token is a number",
                "tokens are counted, limited and billed")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "On the server, the example request's messages become one text, ending in "
            "\"Assistant:\" for GPT-2 to continue.",
            "GPT-2's tokenizer cuts the %d characters into %d tokens; a service's own "
            "tokenizer cuts a little differently." % (m["chars"], m["n_tokens"]),
            "Each token is a number from GPT-2's list of %s; the colon is %d both times."
            % (m["vocab_text"], m["repeated_id"]),
            "Tokens are what a service counts, limits and bills: here %d read, at most %d "
            "written, of GPT-2's %s." % (m["n_tokens"], m["max_tokens"], m["context_text"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        self.server(sc, 0.09, 0.39, op, w=46.0, h=34.0)
        self.text(sc, 0.125, 0.39, "on the server", op, SIZE, "lm")
        # the text: under the request in phase 1, then at the top
        if k == 0:
            y, a = self.LOW_Y, op * self.at(0, 0.45)
        else:
            y, a = self.LOW_Y + (self.TOP_Y - self.LOW_Y) * eased(u, 0.0, 0.08) if k == 1 \
                else self.TOP_Y, op
        self.mono_card(sc, self.X0, y, m["prompt_lines"], a, 0.3)
        self.text(sc, self.X0 + 0.315, y + 0.03, "%d characters" % m["chars"], a, SIZE, "lm",
                  ink=QUIET)
        if k >= 1:
            self.tokens_row(sc, k, u, op)

    def shown_tiles(self, k, u):
        """How many tiles are on screen (and how far the newest has dropped in)."""
        if k == 0:
            return 0, 1.0
        if k > 1:
            return len(self.places), 1.0
        a, b = self.TILES_AT
        f = (u - a) / (b - a) * len(self.places)
        n = int(np.clip(np.floor(f) + 1, 0, len(self.places))) if u >= a else 0
        return n, float(np.clip((f - (n - 1)) / 0.6, 0.0, 1.0)) if n else 0.0

    def tokens_row(self, sc, k, u, op):
        m = self.meta
        n, drop = self.shown_tiles(k, u)
        for i in range(n):
            x, y = self.places[i]
            d = drop if i == n - 1 else 1.0
            yy = y - 0.03 * (1.0 - ease_in_out(d))
            self.token(sc, x, yy, m["shown"][i], op * min(1.0, d * 3))
        ids = op * self.kept(2)
        if ids > 0.01:
            a, b = self.IDS_AT
            for i, (x, y) in enumerate(self.places):
                at = a + (b - a) * i / len(self.places)
                o = ids * (self.at(2, at) if k == 2 else 1.0)
                self.text(sc, x, y + self.ID_DY, str(m["ids"][i]), o, SIZE, ink=DATA_INK)

    # ------------------------------------------------------------------ 1: one text
    def phase_1(self, sc, u, op):
        m = self.meta
        a = on(u, 0.05, op)
        self.text(sc, self.X0, self.JSON_Y - 0.045, "the request (an example): %d bytes"
                  % m["bytes"], a, SIZE, "lm", ink=QUIET)
        self.mono_card(sc, self.X0, self.JSON_Y, m["json_lines"], a, 0.405)
        y0 = self.JSON_Y + MONO_STEP * len(m["json_lines"]) + 0.005
        self.arrow(sc, (0.2, y0 + 0.01), (0.2, self.LOW_Y - 0.03), AXIS, on(u, 0.4, op),
                   u=eased(u, 0.4, 0.46), width=1.8)
        self.text(sc, 0.215, (y0 + 0.01 + self.LOW_Y - 0.03) / 2, "prompt(messages)",
                  on(u, 0.42, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 2: tokens
    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, self.X0, self.FOOT[0], "%d characters → %d tokens" % (m["chars"],
                                                                         m["n_tokens"]),
                  on(u, 0.6, op), HEAD, "lm", strong=True)
        self.text(sc, self.X0, self.FOOT[1] + 0.01, "␣ a space    ↵ a new line", on(u, 0.3, op),
                  SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 3: numbers
    def phase_3(self, sc, u, op):
        m = self.meta
        r = on(u, 0.55, op)
        for i in m["repeated"]:
            x, y = self.places[i]
            w = self.span(self.token_w(m["shown"][i]))[0]
            self.rect(sc, self.P(x, y + self.ID_DY / 2), w + 10, 66.0, INK, r, border=INK,
                      border_w=2.2, radius=6, lift=1.0, fill_op=0.0)
        self.text(sc, self.X0, self.FOOT[0], "the same token, the same number: %d"
                  % m["repeated_id"], r, SIZE, "lm", strong=True)
        self.text(sc, self.X0, self.FOOT[1], "each from GPT-2's list of %s" % m["vocab_text"],
                  on(u, 0.2, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 4: counted
    def phase_4(self, sc, u, op):
        m = self.meta
        a, b, c = on(u, 0.08, op), on(u, 0.3, op), on(u, 0.5, op)
        y = self.FOOT[0] - 0.008
        self.text(sc, self.X0, y, "read: %d tokens" % m["n_tokens"], a, SIZE, "lm", strong=True)
        self.text(sc, self.X0 + 0.15, y, "written: at most %d" % m["max_tokens"], b, SIZE, "lm",
                  strong=True)
        self.text(sc, self.X0, y + 0.04, "GPT-2's limit: %s tokens in all" % m["context_text"],
                  c, SIZE, "lm")
        # the limit as a track, the 10 read and 20 written at its start, to scale
        full, yb = 0.26, y + 0.078
        self.rect(sc, self.P(self.X0 + full / 2, yb), self.span(full)[0], 12.0, TRACK, c * 0.6,
                  radius=3, lift=0.25)
        used = m["n_tokens"] + m["max_tokens"]
        for x0, n, ink, o in ((0.0, m["n_tokens"], DATA, a * c),
                              (m["n_tokens"], m["max_tokens"], soft(DATA, 0.5), b * c)):
            w = full * n / m["context"]
            self.rect(sc, self.P(self.X0 + full * x0 / m["context"] + w / 2, yb),
                      max(self.span(w)[0], 1.5), 12.0, ink, o, radius=1, lift=0.3)
        self.text(sc, self.X0 + full + 0.012, yb - 0.001,
                  "at most %d of %s" % (used, m["context_text"]), c * on(u, 0.6, 1.0), SIZE, "lm",
                  ink=DATA_INK)


# ====================================================================== 256
class OneTokenAtATime(Hosted):
    """Score every next token, pick the likeliest, send it at once, add it to the text;
    again, 20 times."""

    TITLE = "One token at a time"
    PHASES = (("the next token", 14.0), ("sent at once", 13.0), ("again", 16.0),
              ("the reply", 14.0))
    LAST = "Text becomes tokens"
    NEXT = "Read once, then remember"
    STATION = (None, None, None, None)
    SERVER = (0.1, 0.43)
    PHONE = (0.402, 0.64, 140.0, 266.0)      # centre, and size in px
    WIRE_Y = 0.43
    BARS_Y, BAR_STEP = 0.625, 0.04          # the top five, one row each
    LABEL_X, BAR_X, BAR_FULL = 0.15, 0.16, 0.14
    SLOTS = (0.04, 0.28, 0.52, 0.76)        # phase 3: steps 2-5
    STREAM = (0.04, 0.5)                    # phase 4: the rest stream in

    def setup(self):
        m = self.meta = self.study.metadata
        reply = m["reply"]
        self.CODE = (
            Code("hs-generate", run=((0.0, 0), (0.5, 2))),
            Code("hs-generate", run=((0.0, 3), (0.55, 4))),
            Code("hs-generate", run=tuple((s + d, line) for s in self.SLOTS
                                          for d, line in ((0.0, 2), (0.08, 3), (0.17, 4)))),
            Code("hs-request", pick=(5, 6, 7), run=((0.0, 1), (0.05, 2)),
                 out=((0.55, "\n".join(self.client_lines(reply))),)))
        self.EVIDENCE = (measured("the first step's top 5", on="GPT-2"),
                         measured("the reply's first token", on="GPT-2"),
                         measured("reply tokens 2 to 5", on="GPT-2"),
                         measured("the whole reply, 20 tokens", on="GPT-2"))
        cx, cy, w, h = self.PHONE
        dw, dh = self.frac(w, h)
        self.screen = (cx - dw / 2 + self.frac(12.0)[0], cx + dw / 2 - self.frac(12.0)[0])
        self.phone_top = cy - dh / 2
        self.lines = self.wrap(m["tokens"], self.screen[1] - self.screen[0])
        self.REPLY_Y = self.phone_top + 0.15

    @staticmethod
    def client_lines(text, width=37):
        """The reply as the client prints it, broken at spaces into lines for the slot."""
        out, cur = [], ""
        for w in text.strip().split(" "):
            if cur and len(cur) + 1 + len(w) > width:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w) if cur else w
        out.append(cur)
        return out

    def wrap(self, tokens, width):
        """The reply's tokens in lines that fit ``width`` (frame fraction), broken only
        where a token starts with a space: ``[[token index, ...], ...]``."""
        words, cur = [], []
        for i, t in enumerate(tokens):
            if t.startswith(" ") and cur:
                words.append(cur)
                cur = []
            cur.append(i)
        words.append(cur)
        lines, line = [], []
        for word in words:
            trial = "".join(tokens[i] for i in line + word).strip()
            if line and width_of(trial) > width:
                lines.append(line)
                line = list(word)
            else:
                line += word
        lines.append(line)
        return lines

    def heading_for(self, k, u):
        return ("the model scores every possible next token",
                "sent to the phone at once",
                "again and again: one token at a time",
                "20 tokens, then it stops")[k]

    def subtitle_for(self, k):
        m = self.meta
        s0 = m["steps"][0]
        return [
            "GPT-2 reads the %d tokens and gives each of its %s tokens a chance; \"%s\" is "
            "the likeliest, %s." % (m["prompt_tokens"], m["vocab_text"], s0["token"].strip(),
                                    s0["p_text"]),
            "The token is sent over the internet at once, then added to the text: now %d "
            "tokens." % m["text_after"][0],
            "Each step: score every token, pick the likeliest, send it, add it to the text. "
            "One token a step.",
            "After %d tokens (max_tokens) it stops, mid-sentence. GPT-2 was not trained to "
            "chat: its reply is poor." % m["n"],
        ][k]

    # ------------------------------------------------------------------ timing
    def written(self, k, u):
        """How many reply tokens have reached the phone."""
        if k == 0:
            return 0
        if k == 1:
            return int(u >= 0.36)
        if k == 2:
            return 1 + sum(u >= s + 0.17 for s in self.SLOTS)
        a, b = self.STREAM
        rest = self.meta["n"] - 1 - len(self.SLOTS)
        return 1 + len(self.SLOTS) + int(np.clip(np.floor((u - a) / (b - a) * rest) + 1, 0,
                                                  rest)) * (u >= a)

    def added(self, k, u):
        """How many reply tokens are in the text the model reads (added after sending)."""
        if k == 1:
            return int(u >= 0.6)
        return self.written(k, u)

    def bars_step(self, k, u):
        """``(step, opacity factor)``: whose top five the bars show, dipping at a swap."""
        if k < 2:
            return 0, 1.0
        i = sum(u >= s for s in self.SLOTS)
        if i == 0:
            return 0, 1.0 - smoothstep(self.SLOTS[0] - 0.025, self.SLOTS[0], u)
        s = self.SLOTS[i - 1]
        f = smoothstep(s, s + 0.03, u)
        if i < len(self.SLOTS):
            f *= 1.0 - smoothstep(self.SLOTS[i] - 0.025, self.SLOTS[i], u)
        return i, f

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        cx, cy, w, h = self.PHONE
        sx, sy = self.SERVER
        self.server(sc, sx, sy, op, w=70.0, h=50.0, lit=1.0)
        self.text(sc, 0.06, sy + 0.075, "the server: GPT-2", op, SIZE, "lm")
        self.phone(sc, cx, cy, w, h, op)
        self.text(sc, cx, self.phone_top - 0.022, "a phone", op, SIZE)
        left = sx + self.frac(35.0 + 6.0)[0]
        right = cx - self.frac(w / 2 + 4.0)[0]
        # the cable, broken where a token's tile rides on it
        cuts = sorted((x - half - 0.004, x + half + 0.004)
                      for x, _y, half in (self.fly_at(j, s) for j, s in self.flying(k, u)))
        a = left
        for c0, c1 in [*cuts, (right, right)]:
            if c0 > a:
                self.polyline(sc, [self.P(a, self.WIRE_Y, self.ARROW_LIFT * 0.5),
                                   self.P(min(c0, right), self.WIRE_Y, self.ARROW_LIFT * 0.5)],
                              2.4, AXIS, op, bias=0.3)
            a = max(a, c1)
        self.text(sc, (left + right) / 2, self.WIRE_Y - 0.05, "the internet", op, SIZE,
                  ink=QUIET)
        # the question on the phone
        x1 = self.screen[1]
        qy = self.phone_top + 0.07
        self.rect(sc, self.P(x1 - 0.045, qy + 0.018), 92.0, 44.0, TRACK, op * 0.8, radius=8,
                  lift=0.24)
        for i, s in enumerate(("What is a", "server?")):
            self.text(sc, x1 - 0.008, qy + 0.036 * i, s, op, SIZE, "rm", ink=INK)
        self.reply_text(sc, k, u, op)
        # the text so far, and what is written
        self.running_counts(sc, k, u, op)
        keep = op * self.kept(0, 2)
        if keep > 0.01:
            self.top_five(sc, k, u, keep)

    def reply_text(self, sc, k, u, op):
        m = self.meta
        n = self.written(k, u)
        x0 = self.screen[0]
        for li, line in enumerate(self.lines):
            shown = [i for i in line if i < n]
            if not shown:
                break
            s = "".join(m["tokens"][i] for i in shown).strip()
            self.text(sc, x0, self.REPLY_Y + 0.036 * li, s, op, SIZE, "lm", ink=INK)

    def running_counts(self, sc, k, u, op):
        m = self.meta
        y = self.BARS_Y + self.BAR_STEP * 5 + 0.02
        n_text = m["prompt_tokens"] + self.added(k, u)
        self.text(sc, 0.06, y, "the text so far: %d tokens" % n_text, op, SIZE, "lm")
        a = op * (self.at(1, 0.36) if k == 1 else (1.0 if k > 1 else 0.0))
        self.text(sc, 0.06, y + 0.038, "written: %d of %d" % (max(1, self.written(k, u)),
                                                                m["max_tokens"]),
                  a, SIZE, "lm", strong=True)

    def top_five(self, sc, k, u, op):
        m = self.meta
        i, f = self.bars_step(k, u)
        st = m["steps"][i]
        grow = eased(u, 0.18, 0.36) if k == 0 else 1.0
        a = op * (self.at(0, 0.14) if k == 0 else 1.0)
        self.text(sc, 0.06, self.BARS_Y - 0.042, "the 5 likeliest of %s" % m["vocab_text"], a,
                  SIZE, "lm", ink=QUIET)
        b = a * f
        ring = {0: on(u, 0.5, 1.0), 1: 1.0}.get(k, smoothstep(0.08, 0.11, u - self.SLOTS[
            max(i - 1, 0)]) if i > 0 else 1.0)
        for r, t in enumerate(st["top"]):
            y = self.BARS_Y + self.BAR_STEP * r
            self.text(sc, self.LABEL_X, y, t["shown"], b, SIZE, "rm", ink=INK)
            L = self.BAR_FULL * t["p"] * grow
            if L > 0:
                self.rect(sc, self.P(self.BAR_X + L / 2, y), max(self.span(L)[0], 1.5), 14.0,
                          DATA if r == 0 else soft(DATA, 0.45), b, radius=2, lift=0.3)
            self.text(sc, self.BAR_X + L + 0.008, y, t["p_text"], b * grow, SIZE, "lm",
                      ink=DATA_INK)
        if b * ring > 0.01:
            w = width_of(st["top"][0]["shown"]) + self.frac(14.0)[0]
            self.rect(sc, self.P(self.LABEL_X - w / 2 + self.frac(5.0)[0], self.BARS_Y),
                      self.span(w)[0], 24.0, INK, b * ring, border=INK, border_w=2.2,
                      radius=5, lift=1.0, fill_op=0.0)

    def flying(self, k, u):
        """``[(token, s)]``: the reply tokens on their way, each ``s`` of the way along."""
        if k == 1:
            out = [(0, (u - 0.08) / 0.28)]
        elif k == 2:
            out = [(j, (u - s - 0.1) / 0.07) for j, s in enumerate(self.SLOTS, 1)]
        else:
            out = []
        return [(j, s) for j, s in out if 0.0 < s < 1.0]

    def fly_at(self, j, s):
        """Where reply token ``j``'s tile is, ``s`` of the way from the server along the
        cable to the phone: ``(x, y, half its width)``."""
        sx, _sy = self.SERVER
        half = self.token_w(self.meta["steps"][j]["shown"]) / 2
        path = [(sx + self.frac(35.0)[0] + half + 0.006, self.WIRE_Y),
                (self.screen[0] - self.frac(14.0)[0] - half, self.WIRE_Y)]
        x, y = along(path, ease_in_out(s))
        return x, y, half

    def fly(self, sc, j, s, op):
        """Reply token ``j`` as a tile on the cable (``s``: 0..1); it fades as it reaches
        the phone, where its text appears."""
        if not 0.0 < s < 1.0 or op <= 0.01:
            return
        x, y, _half = self.fly_at(j, s)
        fade = min(1.0, s / 0.12, (1.0 - s) / 0.12)
        self.token(sc, x, y, self.meta["steps"][j]["shown"], op * fade, lift=1.5)

    # ------------------------------------------------------------------ 1-4
    def phase_1(self, sc, u, op):
        sx, sy = self.SERVER
        self.text(sc, sx + 0.06, sy + 0.035, "reads the text", on(u, 0.06, op) *
                  (1.0 - smoothstep(0.85, 0.9, u)), SIZE, "lm", ink=QUIET)
        a = on(u, 0.55, op)
        self.text(sc, 0.06, self.BARS_Y + self.BAR_STEP * 5 + 0.058, "picked: the likeliest",
                  a, SIZE, "lm", strong=True)

    def phase_2(self, sc, u, op):
        for j, s in self.flying(1, u):
            self.fly(sc, j, s, op)
        self.text(sc, 0.245, self.WIRE_Y + 0.05, "sent at once", on(u, 0.4, op), SIZE,
                  ink=DATA_INK)

    def phase_3(self, sc, u, op):
        for j, s in self.flying(2, u):
            self.fly(sc, j, s, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        y = self.BARS_Y - 0.02
        self.text(sc, 0.06, y, "%d tokens: max_tokens" % m["n"], on(u, 0.5, op), HEAD, "lm",
                  strong=True)
        self.text(sc, 0.06, y + 0.045, "stops mid-sentence", on(u, 0.55, op), SIZE, "lm")
        self.text(sc, 0.06, y + 0.1, "GPT-2 was not trained to chat:", on(u, 0.68, op), SIZE,
                  "lm", ink=QUIET)
        self.text(sc, 0.06, y + 0.135, "a poor reply", on(u, 0.68, op), SIZE, "lm",
                  ink=QUIET)


# ====================================================================== 257
class ReadOnceRemember(Hosted):
    """Without a cache, every step reads the whole text again; with it, each token's keys
    and values are kept and only the new token runs: the same scores, far fewer
    multiply-adds."""

    TITLE = "Read once, then remember"
    PHASES = (("no memory", 12.0), ("keep", 12.0), ("only the new", 13.0),
              ("the same", 10.0), ("less work", 11.0))
    LAST = "One token at a time"
    NEXT = "Many people, one model"
    STATION = (None, None, None, None, None)
    STAIR = (0.065, 0.415, 6.4, 8.6)        # left, top, cell w and h (px)
    SHELF = (0.305, 0.415, 5.3, 15.5)       # left, top, cell w (px), shelf pitch (px)
    ROWS_AT = (0.08, 0.5)                   # phase 1: the staircase's rows appear
    FILL_AT = (0.12, 0.6)                   # phase 2: the shelves fill, layer by layer
    GROW_AT = (0.08, 0.6)                   # phase 3: a step at a time
    COUNT_Y = (0.855, 0.893)

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("hs-generate", run=((0.0, 0),)),
            Code("hs-cache", run=((0.0, 0), (0.2, 1), (0.32, 2), (0.44, 3), (0.6, 4))),
            Code("hs-generate", run=((0.0, 1), (0.08, 4))),
            Code("hs-generate", run=((0.0, 4),)),
            Code("hs-generate", run=((0.0, 4),)))
        self.EVIDENCE = (exact("tokens run per step, no cache"),
                         measured("the cache's keys and values", on="GPT-2"),
                         exact("tokens run per step, with the cache"),
                         measured("20 tokens, cache against none", on="GPT-2"),
                         exact("multiply-adds from GPT-2's shapes"))
        self.k_norm = np.asarray(m["k_norm"])
        self.v_norm = np.asarray(m["v_norm"])

    def heading_for(self, k, u):
        return ("no memory: read the whole text again",
                "the cache: keys and values, kept",
                "with the cache, only the new token runs",
                "the same reply as reading it all again",
                "%d times fewer multiply-adds" % self.meta["ratio_round"])[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Without a cache, each new token means running the whole text again: %d tokens, "
            "then %d, up to %d." % (m["tokens_without"][0], m["tokens_without"][1],
                                    m["tokens_without"][-1]),
            "The cache: in each of GPT-2's %d layers, every token read leaves keys and values "
            "that later tokens look back at." % m["layers"],
            "With the cache only the new token runs, looking back at what is kept: %d tokens "
            "run, not %d." % (m["run_with"], m["run_without"]),
            "The same %d tokens; the scores differ from reading everything again by at most "
            "%s, rounding." % (m["same_count"], m["max_difference_text"]),
            "For the %d reply tokens: %s multiply-adds instead of %s, %s times fewer. Counts, "
            "not times." % (m["n"], m["total_with_text"], m["total_without_text"],
                            m["ratio_text"]),
        ][k]

    # ------------------------------------------------------------------ timing
    def rows_shown(self, k, u):
        """Phase 1: how many of the staircase's rows (the prompt's, then 20) are drawn."""
        n = self.meta["n"] + 1
        if k > 0:
            return n
        a, b = self.ROWS_AT
        return int(np.clip(np.floor((u - a) / (b - a) * n) + 1, 0, n)) if u >= a else 0

    def steps_cached(self, k, u):
        """How many reply steps have run with the cache (phase 3 on)."""
        n = self.meta["n"]
        if k < 2:
            return 0
        if k > 2:
            return n
        a, b = self.GROW_AT
        return int(np.clip(np.floor((u - a) / (b - a) * n) + 1, 0, n)) if u >= a else 0

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        s = op * self.leaving(k, 2)
        if s > 0.01:
            self.staircase(sc, k, u, s)
        c = op * self.leaving(k, 2) * (k >= 1)
        if c > 0.01:
            self.shelves(sc, k, u, c)
        r = op * self.leaving(k, 3) * (k >= 3)
        if r > 0.01:
            self.same_reply(sc, k, u, r)
        self.counts(sc, k, u, op)

    def leaving(self, k, last):
        """1 up to phase ``last``; in the next, fading out over its first half second
        (after the content's dip, so the picture is never empty)."""
        if k <= last:
            return 1.0
        return 1.0 - self.at(k, 0.0) if k == last + 1 else 0.0

    def staircase(self, sc, k, u, op):
        m = self.meta
        x0, top, cw, ch = self.STAIR
        self.text(sc, 0.06, top - 0.03, "tokens run, step by step", op, SIZE, "lm")
        cached = self.steps_cached(k, u)
        run_ink, new_ink = soft(DATA, 0.4), DATA
        dy = self.frac(0, ch)[1]
        for r in range(self.rows_shown(k, u)):
            n = m["prompt"] + r                      # tokens in that step's text
            y = top + dy * (r + 0.5)
            if r == 0:
                rgb = [run_ink] * n
            elif r <= cached:
                rgb = [DIM] * (n - 1) + [new_ink]
            else:
                rgb = [run_ink] * (n - 1) + [new_ink]
            self.row_tiles(sc, x0, y, np.asarray(rgb), cw, ch - 1.0, op)
        bottom = top + dy * (m["n"] + 1)
        a = op * (self.at(0, 0.55) if k == 0 else 1.0) * self.kept(0, 1)
        self.text(sc, x0, bottom + 0.022, "the prompt, then %d steps" % m["n"], a, SIZE, "lm",
                  ink=QUIET)
        b = op * (self.at(2, 0.04) if k == 2 else 1.0) * (k >= 2)
        self.text(sc, x0, bottom + 0.022, "grey: kept, not run again", b, SIZE, "lm",
                  ink=QUIET)

    def shelves(self, sc, k, u, op):
        m = self.meta
        x0, top, cw, pitch = self.SHELF
        a = op * (self.at(1, 0.04) if k == 1 else 1.0)
        self.text(sc, x0, top - 0.03, "the cache, kept", a, SIZE, "lm")
        cols = m["prompt"] + self.steps_cached(k, u)
        dy = self.frac(0, pitch)[1]
        fa, fb = self.FILL_AT
        full = m["total"] * cw
        for layer in range(m["layers"]):
            y = top + dy * (layer + 0.5)
            fill = 1.0 if k > 1 else smoothstep(fa + (fb - fa) * layer / m["layers"],
                                                fa + (fb - fa) * (layer + 1) / m["layers"], u)
            self.rect(sc, self.P(x0 + self.frac(full / 2)[0], y + self.frac(0, 6.5)[1]),
                      full + 6, 2.0, AXIS, a, radius=1, lift=0.25)
            if fill * a > 0.01:
                kk = self.k_norm[layer, :cols]
                vv = self.v_norm[layer, :cols]
                self.row_tiles(sc, x0, y - self.frac(0, 3.2)[1], self.shade(kk), cw, 5.6,
                               a * fill)
                self.row_tiles(sc, x0, y + self.frac(0, 2.6)[1], self.shade(vv), cw, 5.6,
                               a * fill)
        bottom = top + dy * m["layers"]
        b = op * (self.at(1, 0.62) if k == 1 else 1.0)
        self.text(sc, x0, bottom + 0.022, "a shelf a layer", b, SIZE, "lm", ink=QUIET)
        self.text(sc, x0, bottom + 0.057, "a column a token", b, SIZE, "lm", ink=QUIET)
        if k == 1:
            # the prompt's row, read once, goes onto the shelves
            sx, stop, scw, sch = self.STAIR
            y = stop + self.frac(0, sch)[1] * 0.5
            self.arrow(sc, (sx + self.frac(scw * m["prompt"] + 6)[0], y),
                       (x0 - 0.008, top + 0.01), AXIS, on(u, 0.08, op), u=eased(u, 0.08, 0.16),
                       width=1.8)

    @staticmethod
    def shade(values):
        """Shelf tiles: pale to full blue with the size of a token's keys (or values)."""
        v = np.clip(np.asarray(values, float), 0.0, 1.0)[:, None]
        lo, hi = np.asarray(soft(DATA, 0.7)), np.asarray(DATA, float)
        return lo + (hi - lo) * v

    def counts(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(0, 0.55) if k == 0 else 1.0)
        self.text(sc, 0.06, self.COUNT_Y[0], "without a cache: %d tokens run"
                  % m["run_without"], a, SIZE, "lm", strong=True)
        b = op * (self.at(2, 0.66) if k == 2 else (1.0 if k > 2 else 0.0))
        self.text(sc, 0.06, self.COUNT_Y[1], "with the cache: %d tokens run" % m["run_with"],
                  b, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 1-3: in steady
    def phase_1(self, sc, u, op):
        """The staircase, in ``steady``."""

    def phase_2(self, sc, u, op):
        """The shelves, in ``steady``."""

    def phase_3(self, sc, u, op):
        """The staircase turns grey and the shelves grow, in ``steady``."""

    # ------------------------------------------------------------------ 4: the same
    def same_reply(self, sc, k, u, op):
        """The reply twice: with the cache, and reading everything again."""
        reply = OneTokenAtATime.client_lines(self.meta["reply"], 34)
        y = 0.395
        rows = (("with the cache", 0.08), ("reading it all again", 0.3))
        for i, (name, at) in enumerate(rows):
            a = op * (self.at(3, at) if k == 3 else 1.0)
            yy = y + 0.19 * i
            self.text(sc, 0.06, yy, name, a, SIZE, "lm", ink=QUIET)
            self.mono_card(sc, 0.065, yy + 0.04, reply, a, 0.405)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, 0.795, "the same %d tokens" % m["same_count"], on(u, 0.5, op),
                  HEAD, "lm", strong=True)
        self.text(sc, 0.27, 0.795, "scores within %s" % m["max_difference_text"],
                  on(u, 0.6, op), SIZE, "lm")

    # ------------------------------------------------------------------ 5: less work
    def phase_5(self, sc, u, op):
        m = self.meta
        full = 0.4
        rows = (("without a cache: %s multiply-adds" % m["total_without_text"],
                 m["total_without"], 0.06, AXIS),
                ("with the cache: %s" % m["total_with_text"], m["total_with"], 0.3, DATA))
        for i, (name, v, at, ink) in enumerate(rows):
            a = on(u, at, op)
            y = 0.45 + 0.11 * i
            self.text(sc, 0.06, y, name, a, SIZE, "lm")
            L = full * v / m["total_without"] * eased(u, at, at + 0.12)
            if L > 0:
                self.rect(sc, self.P(0.065 + L / 2, y + 0.04), max(self.span(L)[0], 2.0), 16.0,
                          ink, a, radius=2, lift=0.3)
        self.text(sc, 0.06, 0.68, "%s times fewer" % m["ratio_text"], on(u, 0.55, op), BIG,
                  "lm", strong=True)

ANIMATIONS = {"hosting-text-to-tokens": TextToTokens,
              "hosting-one-token-at-a-time": OneTokenAtATime,
              "hosting-read-once-remember": ReadOnceRemember}
