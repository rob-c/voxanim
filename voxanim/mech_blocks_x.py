"""Animations 252-254, the first three of "Hosting a language model" in "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_x`.

For people who have never thought about what is on their own phone and what is not: every
new thing gets a picture (a phone, a server as a box of a computer far away, a cable for
the internet, a request as a blue card, tokens as small blue boxes, the model as orange
tiles, as weights are everywhere in the series, a file, memory as shelves).

* :class:`DeviceAndServer` (``hosting-device-and-server``): the phone, its app and the
  question typed; the server far away, the model inside it (124,439,808 numbers, 498 MB);
  the request, 82 bytes, along the cable; GPT-2's real reply coming back token by token
  (20 tokens, 92 bytes; poor, as GPT-2 was not trained to chat); the sizes on a log scale,
  each mark 1,000 times the last: the model is 6,070,234 times the request.
* :class:`RequestIsText` (``hosting-a-request-is-text``): region "hs-request" line by line:
  one message, the request round it, then one line of text, 82 bytes, off to the server;
  then an example conversation (written for the animation) whose every turn sends the
  whole talk again: 82, 252, then 422 bytes, 10, 36, then 66 tokens.
* :class:`ModelIsAFile` (``hosting-the-model-is-a-file``): GPT-2 as a file of 124,439,808
  numbers; part by part (token table 38,597,376, 12 layers of 7,087,872, positions
  786,432, last norm 1,536); 4 bytes each, 497,759,232 bytes; loaded once into the server's
  memory (shelves), and read there by every request.

The picture is on the left; the code (``examples/serve.py``), the loop map and the evidence
tag are the series' furniture (:class:`~voxanim.blocks.Blocks`). Serving a model is not
training: the loop map lights no station.
"""

from __future__ import annotations

import numpy as np

from . import color as C
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
    exact,
    measured,
    val,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, GRID, PAPER, TRACK, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
CASE = soft(AXIS, 0.72)                  # the server's case
DRAWER = soft(AXIS, 0.88)
BODY = C.rgb("#4f535a")                  # the phone's body
SCREEN = C.rgb("#fbfbf8")
CABLE = AXIS
PX_W, PX_H = 960.0, 540.0                # layout fractions to reference px


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def width_px(s, size=SIZE):
    """How wide ``s`` is drawn, in reference px."""
    return layout.text_width(s, size) * PX_W


def wrap_words(tokens, width, lines=3, more="…"):
    """``tokens`` (GPT-2's pieces, most with a leading space) laid into at most ``lines``
    lines of ``width`` px: for each count ``n`` of tokens arrived, the lines showing the
    first ``n`` (the last line ends in ``more`` once a token no longer fits)."""
    states, rows, full = [[]], [""], False
    for t in tokens:
        if not full:
            line = rows[-1] + t if rows[-1] else t.lstrip()
            if width_px(line) <= width:
                rows[-1] = line
            elif len(rows) < lines and width_px(t.lstrip()) <= width:
                rows.append(t.lstrip())
            else:
                full = True
                last = rows[-1]
                while last and width_px(last + more) > width:
                    last = last[:last.rstrip().rfind(" ")] if " " in last.strip() else ""
                rows[-1] = last.rstrip() + more
        states.append(list(rows))
    return states


def wrap_chars(text, width):
    """``text`` in lines of at most ``width`` characters, broken at spaces."""
    out, line = [], ""
    for w in text.split(" "):
        if line and len(line) + 1 + len(w) > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w) if line else w
    return out + [line]


class Hosted(Blocks):
    """What the three share: the pictures of a phone, a server, a cable, a file, shelves."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over half a second); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def appear(self, k, u, kk, uu):
        """1 from phase ``kk`` at ``uu`` on, for a thing drawn in :meth:`steady` while phase
        ``k`` at ``u`` runs."""
        if k < kk:
            return 0.0
        return self.at(kk, uu) if k == kk else 1.0

    # ------------------------------------------------------------------ the pictures
    @staticmethod
    def bezel(w, h):
        """A phone's screen size (px) inside its body ``w x h``."""
        return w - 2 * max(2.0, 0.042 * w), h - 2 * max(2.5, 0.076 * h)

    def phone(self, sc, fx, fy, w, h, op, lift=0.2):
        """A phone: a dark body, a light screen, a speaker slot. Returns the screen's
        layout box ``(x0, y0, x1, y1)``."""
        sw, sh = self.bezel(w, h)
        dx, dy = sw / 2 / PX_W, sh / 2 / PX_H
        if op > 0.01:
            c = self.P(float(fx), float(fy))
            self.rect(sc, c, w, h, BODY, op, border=dark(BODY, 0.3), border_w=1.5,
                      radius=min(14.0, w / 5), lift=lift)
            self.rect(sc, c, sw, sh, SCREEN, op, radius=min(4.0, w / 8), lift=lift + 0.02)
            by = (h - sh) / 2
            self.rect(sc, self.shift(c, 0, -h / 2 + by / 2), min(20.0, w / 3),
                      max(1.2, min(3.0, by / 3)), SCREEN, op, radius=1.0, lift=lift + 0.02)
        return fx - dx, fy - dy, fx + dx, fy + dy

    def server(self, sc, fx, fy, w, h, op, drawers=2, lift=0.2):
        """A server: a tall grey case with drawers and their lights. Returns the layout y
        under the drawers."""
        top = fy - h / 2 / PX_H
        if op > 0.01:
            c = self.P(float(fx), float(fy))
            self.rect(sc, c, w, h, CASE, op, border=AXIS, border_w=1.6, radius=6, lift=lift)
            for i in range(drawers):
                y = top + (16 + 19 * i) / PX_H
                self.rect(sc, self.P(fx, y), w - 16, 13, DRAWER, op, border=GRID,
                          border_w=1.0, radius=3, lift=lift + 0.02)
                self.disk(sc, self.P(fx + (w / 2 - 15) / PX_W, y), 2.4, dark(AXIS, 0.3), op,
                          lift=lift + 0.04, outline=0.0)
        return top + (16 + 19 * drawers) / PX_H

    def model_tiles(self, sc, values, fx, fy, side, op, lo, hi, lift=0.3):
        """The model: orange tiles of real weights (deeper: a bigger number)."""
        self.tiles(sc, np.abs(np.asarray(values, float)), fx, fy, side, side, op, ink=WEIGHT,
                   lo=lo, hi=hi, lift=lift, grid=0.25)

    def cable(self, sc, x0, x1, y, op, gaps=()):
        """The internet between two machines: a grey line from layout ``x0`` to ``x1``,
        broken round each ``(xa, xb)`` of ``gaps`` (where a card travels on it)."""
        if op <= 0.01:
            return
        pieces, a = [], x0
        for ga, gb in sorted(gaps):
            if gb <= a or ga >= x1:
                continue
            if ga > a:
                pieces.append((a, ga))
            a = max(a, gb)
        if a < x1:
            pieces.append((a, x1))
        for pa, pb in pieces:
            if pb - pa > 1e-3:
                self.polyline(sc, [self.P(pa, y, self.ARROW_LIFT), self.P(pb, y, self.ARROW_LIFT)],
                              2.4, CABLE, op, bias=0.3)

    def chip(self, sc, fx, fy, text, op, ink=DATA, h=22.0, size=SIZE, lift=0.6):
        """A small card holding ``text`` (a request, a token); returns its half width in
        layout x."""
        w = width_px(text, size) + 14.0
        if op > 0.01:
            self.rect(sc, self.P(float(fx), float(fy)), w, h, soft(ink, 0.72), op, border=ink,
                      border_w=1.6, radius=5, lift=lift)
            self.text(sc, fx, fy, text, op, size, "mm", ink=dark(ink, 0.45))
        return w / 2 / PX_W

    @staticmethod
    def chip_half(text, size=SIZE):
        """Half the width of :meth:`chip`'s card for ``text``, in layout x."""
        return (width_px(text, size) + 14.0) / 2 / PX_W

    def file_icon(self, sc, fx, fy, w, h, op, lift=0.2):
        """A file: a sheet of paper with its top right corner folded."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, w, h, PAPER, op, border=AXIS, border_w=1.6, radius=3, lift=lift)
        f = 16.0
        self.rect(sc, self.shift(c, w / 2 - f / 2, -h / 2 + f / 2), f, f, DRAWER, op,
                  border=AXIS, border_w=1.2, radius=2, lift=lift + 0.02)

    def shelves(self, sc, x0, x1, ys, op, lift=0.2):
        """Memory as shelves: a grey board under each layout ``y`` of ``ys``."""
        if op <= 0.01:
            return
        w = (x1 - x0) * PX_W
        for y in ys:
            self.rect(sc, self.P((x0 + x1) / 2, y), w, 4.0, AXIS, op, radius=1.5, lift=lift)


# ====================================================================== 252
class DeviceAndServer(Hosted):
    """The phone and the server, the request between them, the reply back, the sizes."""

    TITLE = "Your device and the server"
    PHASES = (("your phone", 12.0), ("the server", 12.0), ("the request", 11.0),
              ("tokens back", 12.0), ("sizes", 12.0))
    LAST = "A new encoder needs a new store"
    NEXT = "A request is text"
    STATION = (None, None, None, None, None)
    PHONE, PHONE_W, PHONE_H = (0.12, 0.558), 120.0, 184.0
    SERVER, SERVER_W, SERVER_H = (0.42, 0.558), 92.0, 184.0
    LABEL_Y = 0.365
    CABLE_Y = 0.5
    MID_X = 0.274                    # the middle, between the two machines
    LIST_Y = (0.765, 0.805, 0.845, 0.885)
    TYPE_AT = (0.15, 0.4)            # phase 1: the question typed
    TOKENS_AT = (0.08, 0.7)          # phase 4: the 20 tokens arrive
    TRAVEL = 0.03
    BAR_X = (0.225, 0.445)           # phase 5: 1 byte to 1 billion, log scale
    ROWS_Y = (0.805, 0.845, 0.885)
    ROW_AT = (0.1, 0.3, 0.5)

    def setup(self):
        m = self.meta = self.study.metadata
        sx0, _sy0, sx1, _sy1 = self.screen()
        self.reply_lines = wrap_words(m["reply_tokens"], (sx1 - sx0) * PX_W - 13, 4)
        n = m["n_reply"]
        a, b = self.TOKENS_AT
        self.token_at = [a + (b - a) * i / (n - 1) for i in range(n)]
        self.out_reply = "\n".join(wrap_chars(m["reply"].strip(), 39))
        self.CODE = (
            Code("hs-request", run=((0.0, 0), (0.45, 1))),
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-request", run=((0.0, 2), (0.15, 4), (0.3, 5))),
            Code("hs-request", pick=(4, 5, 6, 7), run=((0.0, 1), (0.05, 2), (0.08, 3)),
                 out=((0.8, self.out_reply),)),
            Code(("hs-request", "hs-load"), pick=(4, 8), run=((0.0, 0), (0.45, 1))))
        self.EVIDENCE = (about_data("the question is written for the animation"),
                         measured("GPT-2 small's numbers, counted", on="GPT-2"),
                         exact("len(body), the request's bytes"),
                         measured("GPT-2 small's real reply, as it is", on="GPT-2"),
                         exact("bytes counted, 4 bytes a number"))
        m_corner = np.abs(np.asarray(m["model_corner"]))
        self.lo, self.hi = float(np.percentile(m_corner, 5)), float(np.percentile(m_corner, 95))

    def heading_for(self, k, u):
        return ("your phone holds the app and your words",
                "the model is on the server, far away",
                "a request: your words, sent as text",
                "tokens come back, one at a time",
                "the request is tiny; the model is huge")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Your phone or laptop holds the app and the words you type: here, \"%s\""
            % m["question"],
            "A server is a computer far away. The model, GPT-2 small (%s numbers), lives there; "
            "real services run far bigger ones." % m["numbers_text"],
            "The app sends a request: your words as text, %s bytes, over the internet to the "
            "server." % m["request_bytes_text"],
            "GPT-2's real reply comes back token by token. GPT-2 (2019) was not trained to "
            "chat, so the reply is poor.",
            "On a log scale each mark is 1,000 times the last: the model is %s times the "
            "request." % m["ratio_text"],
        ][k]

    # ------------------------------------------------------------------ where things are
    def screen(self):
        fx, fy = self.PHONE
        sw, sh = self.bezel(self.PHONE_W, self.PHONE_H)
        return fx - sw / 2 / PX_W, fy - sh / 2 / PX_H, fx + sw / 2 / PX_W, fy + sh / 2 / PX_H

    def ends(self):
        """The cable's two ends (layout x)."""
        return (self.PHONE[0] + (self.PHONE_W / 2 + 4) / PX_W,
                self.SERVER[0] - (self.SERVER_W / 2 + 4) / PX_W)

    def arrived(self, u):
        """How many reply tokens have reached the phone at ``u`` of phase 4."""
        return sum(1 for a in self.token_at if u >= a + self.TRAVEL)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a = op * self.appear(k, u, 0, 0.03)
        self.text(sc, 0.06, self.LABEL_Y, "your phone or laptop", a, SIZE, "lm")
        self.phone(sc, *self.PHONE, self.PHONE_W, self.PHONE_H, a)
        self.screen_content(sc, k, u, a)
        s = op * self.appear(k, u, 1, 0.04)
        sx, sy = self.SERVER
        self.text(sc, sx + self.SERVER_W / 2 / PX_W, self.LABEL_Y, "the server, far away", s,
                  SIZE, "rm")
        under = self.server(sc, sx, sy, self.SERVER_W, self.SERVER_H, s)
        t = op * self.appear(k, u, 1, 0.2)
        self.model_tiles(sc, m["model_corner"], sx, under + 0.09, 60.0, t, self.lo, self.hi)
        self.text(sc, sx, under + 0.18, "the model", t, SIZE, "mm", ink=WEIGHT_INK)
        c = op * self.appear(k, u, 2, 0.04)
        if c > 0.01:
            x0, x1 = self.ends()
            gaps = self.gaps(k, u)
            self.cable(sc, x0, x1, self.CABLE_Y, c, gaps)
            self.text(sc, self.MID_X, self.CABLE_Y - 0.045, "the internet", c, SIZE, "mm",
                      ink=QUIET)
        lst = op * self.kept(0, 3)
        if lst > 0.01:
            self.lists(sc, k, u, lst)

    def screen_content(self, sc, k, u, op):
        """The app on the phone: its bar, the question typed, the reply as it arrives."""
        if op <= 0.01:
            return
        m = self.meta
        x0, y0, x1, _y1 = self.screen()
        cx = (x0 + x1) / 2
        self.rect(sc, self.P(cx, y0 + 10 / PX_H), (x1 - x0) * PX_W, 20.0, soft(DATA, 0.6), op,
                  radius=3, lift=0.25)
        self.text(sc, cx, y0 + 10 / PX_H + 0.001, "chat app", op, SIZE, "mm", ink=DATA_INK)
        # the question, typed letter by letter in phase 1
        q = m["question"]
        cut = q.index(" ", q.index(" ") + 1)                 # "What is" / "a server?"
        rows = [q[:cut], q[cut + 1:]]
        n = len(q) if k > 0 else int(round(len(q) * eased(u, *self.TYPE_AT)))
        b = op * self.appear(k, u, 0, self.TYPE_AT[0] - 0.04)
        w = max(width_px(r) for r in rows) + 12
        qy = (0.475, 0.515)
        self.rect(sc, self.P(x1 - 4 / PX_W - w / 2 / PX_W, sum(qy) / 2), w, 46.0,
                  soft(DATA, 0.75), b, border=DATA, border_w=1.2, radius=6, lift=0.25)
        shown = [rows[0][:n], rows[1][:max(0, n - len(rows[0]) - 1)]]
        for y, s in zip(qy, shown, strict=True):
            self.text(sc, x1 - 4 / PX_W - (w - 6) / PX_W, y, s, b, SIZE, "lm", ink=DATA_INK)
        # the reply, token by token from phase 4 on
        if k < 3:
            return
        got = self.arrived(u) if k == 3 else m["n_reply"]
        lines = [s for s in self.reply_lines[got] if s]
        if not lines:
            return
        ry = (0.564, 0.604, 0.644, 0.684)
        w = max(width_px(s) for s in self.reply_lines[-1]) + 12
        last = len(self.reply_lines[-1]) - 1
        self.rect(sc, self.P(x0 + 4 / PX_W + w / 2 / PX_W, (ry[0] + ry[last]) / 2), w,
                  (ry[last] - ry[0]) * PX_H + 28, TRACK, op, radius=6, lift=0.25)
        for y, s in zip(ry, lines, strict=False):
            self.text(sc, x0 + 10 / PX_W, y, s, op, SIZE, "lm")

    def lists(self, sc, k, u, op):
        """What is on each side, under the two machines."""
        m = self.meta
        ys = self.LIST_Y
        left = (("on your phone:", 0, 0.08, None), ("the app, your words", 0, 0.45, None),
                ("the reply", 3, 0.75, None), ("not the model", 1, 0.55, INK))
        right = (("on the server:", 1, 0.3, None), ("the model,", 1, 0.3, WEIGHT_INK),
                 ("%s numbers" % m["numbers_text"], 1, 0.4, WEIGHT_INK),
                 (m["model_mb"], 1, 0.5, WEIGHT_INK))
        for (s, kk, uu, ink), y in zip(left, ys, strict=True):
            self.text(sc, 0.06, y, s, op * self.appear(k, u, kk, uu), SIZE, "lm", ink=ink)
        x = self.SERVER[0] + self.SERVER_W / 2 / PX_W
        for (s, kk, uu, ink), y in zip(right, ys, strict=True):
            self.text(sc, x, y, s, op * self.appear(k, u, kk, uu), SIZE, "rm", ink=ink)

    # ------------------------------------------------------------------ moving things
    def request_x(self, u):
        x0, x1 = self.ends()
        half = self.chip_half("%s bytes" % self.meta["request_bytes_text"])
        return x0 + half + 0.01 + (x1 - x0 - 2 * half - 0.02) * eased(u, 0.3, 0.55)

    def token_x(self, i, u):
        """Where reply token ``i`` is on the cable at ``u`` of phase 4 (None off it)."""
        a = self.token_at[i]
        if not a <= u < a + self.TRAVEL:
            return None
        x0, x1 = self.ends()
        half = self.chip_half(self.meta["reply_tokens"][i].strip())
        s = (u - a) / self.TRAVEL
        return x1 - half - 0.005 - (x1 - x0 - 2 * half - 0.01) * s

    def gaps(self, k, u):
        """Where the cable is broken round a card travelling on it."""
        m = self.meta
        if k == 2 and u >= 0.25:
            text = "%s bytes" % m["request_bytes_text"]
            x, h = self.request_x(u), self.chip_half(text) + 0.004
            return [(x - h, x + h)]
        if k == 3:
            for i, tok in enumerate(m["reply_tokens"]):
                x = self.token_x(i, u)
                if x is not None:
                    h = self.chip_half(tok.strip()) + 0.004
                    return [(x - h, x + h)]
        return []

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        """The phone and the question typed, in ``steady``."""

    def phase_2(self, sc, u, op):
        """The server and the model, in ``steady``."""

    def phase_3(self, sc, u, op):
        m = self.meta
        text = "%s bytes" % m["request_bytes_text"]
        c = on(u, 0.25, op)
        self.chip(sc, self.request_x(u), self.CABLE_Y, text, c)
        self.text(sc, self.MID_X, 0.56, "a request:", on(u, 0.1, op), SIZE, "mm", strong=True)
        self.text(sc, self.MID_X, 0.6, "your words as text", on(u, 0.1, op), SIZE, "mm")

    def phase_4(self, sc, u, op):
        m = self.meta
        for i, tok in enumerate(m["reply_tokens"]):
            x = self.token_x(i, u)
            if x is not None:
                self.chip(sc, x, self.CABLE_Y, tok.strip(), op)
        self.text(sc, self.MID_X, 0.56, "a token:", on(u, 0.02, op), SIZE, "mm", strong=True)
        self.text(sc, self.MID_X, 0.6, "a piece of a word", on(u, 0.02, op), SIZE, "mm")
        got = self.arrived(u)
        steps = [(0.0, "tokens back: 0")] + [
            (self.token_at[i] + self.TRAVEL, "tokens back: %d" % (i + 1))
            for i in range(m["n_reply"])]
        self.running(sc, self.MID_X, 0.665, steps, u, on(u, 0.05, op), size=SIZE,
                     anchor="mm", ink=DATA_INK)
        if got == m["n_reply"]:
            self.text(sc, self.MID_X, 0.705, "%s bytes" % m["reply_bytes_text"],
                      on(u, self.token_at[-1] + self.TRAVEL + 0.02, op), SIZE, "mm",
                      ink=DATA_INK)

    def phase_5(self, sc, u, op):
        m = self.meta
        x0, x1 = self.BAR_X
        per = (x1 - x0) / 9.0
        self.text(sc, 0.06, self.LIST_Y[0], "bytes, on a log scale: each mark ×1,000",
                  on(u, 0.03, op), SIZE, "lm", strong=True)
        g = on(u, 0.06, op)
        for mark, name in zip(m["marks"], m["mark_names"], strict=True):
            x = x0 + per * mark
            self.polyline(sc, [self.P(x, 0.785), self.P(x, 0.905)], 1.0, GRID, 0.45 * g,
                          bias=0.1)
            self.text(sc, x, 0.925, name, g, SIZE, "mm", ink=QUIET)
        names = dict(request="request", reply="reply", model="model")
        for row, y, at in zip(m["rows"], self.ROWS_Y, self.ROW_AT, strict=True):
            a = on(u, at, op)
            if a <= 0.01:
                continue
            ink = WEIGHT if row["name"] == "model" else DATA
            self.text(sc, 0.06, y, "%s: %s" % (names[row["name"]], row["text"]), a, SIZE, "lm",
                      ink=dark(ink, 0.35))
            w = per * row["log"] * eased(u, at, at + 0.12)
            if w > 0:
                self.rect(sc, self.P(x0 + w / 2, y), max(w * PX_W, 1.0), 12.0, ink, a,
                          radius=2, lift=0.3)


# ====================================================================== 253
class RequestIsText(Hosted):
    """One message, the request round it, one line of text; the whole talk sent each turn."""

    TITLE = "A request is text"
    PHASES = (("one message", 13.0), ("the request", 13.0), ("just text", 13.0),
              ("every turn", 16.0))
    LAST = "Your device and the server"
    NEXT = "The model is a file of numbers"
    STATION = (None, None, None, None)
    CARD_X = (0.06, 0.47)
    LINES_Y = (0.425, 0.47, 0.51, 0.555, 0.595)
    NOTES_Y = (0.67, 0.71, 0.75, 0.79)
    BODY_Y = (0.72, 0.76, 0.8)
    COLS = (0.13, 0.27, 0.41)
    TURN_AT = (0.04, 0.3, 0.56)
    BUBBLE_Y, BUBBLE_STEP = 0.435, 0.05

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("hs-request", run=((0.0, 0), (0.45, 1))),
            Code("hs-request", run=((0.0, 2), (0.45, 3))),
            Code("hs-request", run=((0.0, 4), (0.6, 5))),
            Code("hs-handler", pick=(1, 2), run=((0.0, 0), (0.2, 1))))
        self.EVIDENCE = (about_data("the question is written for the animation"),
                         about_data("the request examples/serve.py sends"),
                         exact("82 characters, 1 byte each"),
                         measured("a talk written for it; GPT-2's tokenizer", on="GPT-2"))
        self.longest = max(x["chars"] for x in m["messages"])

    def heading_for(self, k, u):
        return ("one message: who speaks, and what they said",
                "the request: the talk so far, and a limit",
                "sent as text: %s bytes" % self.meta["bytes_text"],
                "each turn sends the whole talk again")[k]

    def subtitle_for(self, k):
        m = self.meta
        t = m["turn_bytes"]
        n = m["turn_tokens"]
        return [
            "A message is two pieces of text: who is speaking (\"user\") and what they said.",
            "The request holds every message so far, and how long the reply may be: %d tokens "
            "at most." % m["max_tokens"],
            "json.dumps makes it one line of text: %s characters, %s bytes, sent off to the "
            "server." % (m["bytes_text"], m["bytes_text"]),
            "A talk written for this animation: each turn sends it all again, %d, %d, then %d "
            "bytes (%d, %d, then %d tokens)." % (*t, *n),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        j = op * self.kept(0, 2)
        if j > 0.01:
            self.json_card(sc, k, u, j)
        t = op * self.kept(3, 3)
        if t > 0.01:
            self.turns(sc, k, u, t)
        b = op * self.kept(2, 3) * self.appear(k, u, 2, 0.25)
        if b > 0.01:                 # a byte a letter: stays while the talk grows
            m = self.meta
            text = "%s bytes" % m["bytes_text"]
            half = self.chip_half(text)
            self.chip(sc, 0.06 + half, 0.885, text, b, h=26.0)
            self.text(sc, 0.06 + 2 * half + 0.015, 0.885, "one per letter, space and mark", b,
                      SIZE, "lm", ink=QUIET)

    def json_card(self, sc, k, u, op):
        m = self.meta
        x0, x1 = self.CARD_X
        ys = self.LINES_Y
        outer = op * self.appear(k, u, 1, 0.05)
        if outer > 0.01:
            self.card(sc, (x0 + x1) / 2, (ys[0] + ys[-1]) / 2 + 0.005, (x1 - x0) * PX_W,
                      (ys[-1] - ys[0]) * PX_H + 46, outer, lift=0.15)
        lines = m["lines"]["request"]
        inner = op * self.appear(k, u, 0, 0.06)
        ix0 = x0 + 0.025
        self.rect(sc, self.P((ix0 + x1 - 0.03) / 2, (ys[1] + ys[2]) / 2),
                  (x1 - 0.03 - ix0) * PX_W, (ys[2] - ys[1]) * PX_H + 30, soft(DATA, 0.88),
                  inner, border=DATA, border_w=1.6, radius=6, lift=0.22)
        self.text(sc, ix0 + 0.012, ys[1], lines[1].strip(), inner, SIZE, "lm", ink=DATA_INK)
        self.text(sc, ix0 + 0.012, ys[2], lines[2].strip()[:-2], inner, SIZE, "lm", ink=DATA_INK)
        self.text(sc, x0 + 0.012, ys[0], lines[0], outer, SIZE, "lm", ink=DATA_INK)
        self.text(sc, x0 + 0.012, ys[3], "],", outer, SIZE, "lm", ink=DATA_INK)
        self.text(sc, x0 + 0.012, ys[4], lines[3].strip(), outer, SIZE, "lm", ink=DATA_INK)
        # the script's own comments, as the picture's words
        c = m["comments"]
        notes = (("role: " + c["role"], 0, 0.3), ("content: " + c["content"], 0, 0.55),
                 ("messages: " + c["messages"], 1, 0.3),
                 ("max_tokens: " + c["max_tokens"], 1, 0.6))
        keep = self.kept(0, 1) / max(self.kept(0, 2), 1e-6)
        for (s, kk, uu), y in zip(notes, self.NOTES_Y, strict=True):
            self.text(sc, 0.06, y, s, op * keep * self.appear(k, u, kk, uu), SIZE, "lm")

    def turns(self, sc, k, u, op):
        """Three requests side by side: the messages each sends, its bytes and tokens."""
        m = self.meta
        msgs = m["messages"]
        for i, (x, at, n) in enumerate(zip(self.COLS, self.TURN_AT, m["turn_messages"],
                                           strict=True)):
            a = op * self.appear(k, u, 3, at)
            if a <= 0.01:
                continue
            self.text(sc, x, 0.39, "turn %d" % (i + 1), a, SIZE, "mm", strong=True)
            before = m["turn_messages"][i - 1] if i else 0
            for j in range(n):
                b = a * self.appear(k, u, 3, at + 0.03 + 0.02 * j)
                user = msgs[j]["role"] == "user"
                w = 34.0 + 84.0 * msgs[j]["chars"] / self.longest
                new = j >= before
                ink = DATA if user else AXIS
                fill = soft(ink, 0.6 if new else 0.86)
                half = 59.0 / PX_W
                cx = x + half - w / 2 / PX_W if user else x - half + w / 2 / PX_W
                y = self.BUBBLE_Y + self.BUBBLE_STEP * j
                self.rect(sc, self.P(cx, y), w, 22.0, fill, b, border=ink if new else GRID,
                          border_w=1.3, radius=6, lift=0.22)
                self.text(sc, cx, y, "you" if user else "reply", b, SIZE, "mm",
                          ink=dark(ink, 0.4) if new else QUIET)
            c = a * self.appear(k, u, 3, at + 0.15)
            self.text(sc, x, 0.7, "%s bytes" % m["bytes_texts"][i], c, SIZE, "mm", strong=True)
            self.text(sc, x, 0.74, "%s tokens" % m["tokens_texts"][i], c, SIZE, "mm",
                      ink=DATA_INK)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        """The message, in ``steady``."""

    def phase_2(self, sc, u, op):
        """The request round it, in ``steady``."""

    def phase_3(self, sc, u, op):
        m = self.meta
        a = on(u, 0.06, op)
        x0 = 0.06
        w = max(width_px(r) for r in m["rows"]) + 24
        ys = self.BODY_Y
        self.text(sc, x0, 0.675, "body: one line of text", a, SIZE, "lm", strong=True)
        self.card(sc, x0 + w / 2 / PX_W, (ys[0] + ys[-1]) / 2 + 0.006, w,
                  (ys[-1] - ys[0]) * PX_H + 40, a, lift=0.15)
        for y, r in zip(ys, m["rows"], strict=True):
            self.text(sc, x0 + 12 / PX_W, y, r.rstrip(), a, SIZE, "lm", ink=DATA_INK)
        # off to the server
        c = on(u, 0.6, op)
        sx = 0.43
        self.server(sc, sx, 0.755, 60.0, 92.0, c, drawers=3)
        self.text(sc, sx, 0.865, "the server", c, SIZE, "mm")
        right = x0 + w / PX_W
        self.arrow(sc, (right + 0.008, 0.76), (sx - 0.04, 0.76), AXIS, c,
                   u=eased(u, 0.6, 0.7), width=2.0)

    def phase_4(self, sc, u, op):
        self.text(sc, 0.06, 0.79, "pale: sent before, and sent again", on(u, 0.4, op), SIZE,
                  "lm", ink=QUIET)
        self.text(sc, 0.06, 0.83, "an example talk, written for this animation",
                  on(u, 0.08, op), SIZE, "lm", ink=QUIET)


# ====================================================================== 254
class ModelIsAFile(Hosted):
    """The model as a file; part by part; 4 bytes a number; loaded once; read by all."""

    TITLE = "The model is a file of numbers"
    PHASES = (("a file", 11.0), ("part by part", 13.0), ("bytes", 12.0),
              ("loaded once", 11.0), ("for everyone", 12.0))
    LAST = "A request is text"
    NEXT = "Text becomes tokens"
    STATION = (None, None, None, None, None)
    FILE, FILE_W, FILE_H = (0.11, 0.5), 84.0, 108.0
    BAR_X = (0.06, 0.47)
    BAR_Y = 0.74
    GAP = 2.0                        # px between two parts
    PART_AT = (0.06, 0.22)           # phase 2: the token table, then the positions
    LAYERS_AT = (0.36, 0.66)         # ... then the 12 layers, one by one
    NORM_AT = 0.74
    MEM = (0.235, 0.47, 0.4, 0.68)   # the server's memory: x0, x1, top, bottom
    SHELF_Y = (0.475, 0.55, 0.625)
    LOAD_AT = (0.15, 0.6)            # phase 4: the shelves fill
    ASK_AT = (0.08, 0.28, 0.48, 0.68)
    PHONES_Y = (0.43, 0.5, 0.57, 0.64)

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-load", run=((0.0, 0),)),
            Code("hs-handler", run=((0.0, 1), (0.06, 6))))
        self.EVIDENCE = (measured("GPT-2 small's file, its numbers counted", on="GPT-2"),
                         measured("its numbers, part by part", on="GPT-2"),
                         exact("%s numbers × 4 bytes" % m["total_text"]),
                         exact("load() runs once, before any request"),
                         exact("one copy of W, read by every request"))
        parts = m["parts"]
        self.layer_parts = [p for p in parts if p["name"].startswith("layer")]
        # the bar: each part as wide as its share, a small gap between two
        x0, x1 = self.BAR_X
        inner = (x1 - x0) * PX_W - self.GAP * (len(parts) - 1)
        self.segments, x = [], x0
        for p in parts:
            w = inner * p["share"] / PX_W
            self.segments.append((x, x + w))
            x += w + self.GAP / PX_W
        self.px_per_number = inner / m["total"]
        corner = np.abs(np.asarray(parts[0]["sample"]))
        self.corner = corner[:36].reshape(6, 6)
        self.hi = 0.6 * m["hi"]          # deeper: a bigger number (most are small)

    def heading_for(self, k, u):
        return ("the model: one file of numbers",
                "part by part: a table, 12 layers, positions",
                "each number takes 4 bytes",
                "loaded once into the server's memory",
                "one copy, read by every request")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "GPT-2 small is a file of %s numbers, learned once in training; answering only "
            "reads them." % m["total_text"],
            "A token table of %s × %d, then %d layers of %s each, plus positions: %s in all."
            % (m["vocab_text"], m["width"], m["layers"], m["layer_text"], m["total_text"]),
            "Each number is kept in 4 bytes, so the model takes %s bytes: about %s."
            % (m["bytes_text"], m["mb"]),
            "When the server starts, load() reads the file once into its memory, and keeps it "
            "there.",
            "Every request reads the same copy in memory; it is not loaded again while the "
            "server runs.",
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        f = op * self.kept(0, 3)
        if f > 0.01:
            fx, fy = self.FILE
            self.text(sc, 0.06, 0.375, "the model: a file", f, SIZE, "lm")
            self.file_icon(sc, fx, fy, self.FILE_W, self.FILE_H, f)
            self.model_tiles(sc, self.corner, fx, fy + 0.01, 54.0, f, 0.0, self.hi)
        b = op * self.kept(1, 2)
        if b > 0.01:
            self.bar(sc, k, u, b)
        s = op * self.kept(3, 4)
        if s > 0.01:
            self.memory(sc, k, u, s)

    def bar(self, sc, k, u, op):
        """GPT-2's numbers as one bar, part by part (each part's real values as shades)."""
        m = self.meta
        parts = m["parts"]
        y = self.BAR_Y
        n_layers = len(self.layer_parts)
        a0, a1 = self.LAYERS_AT
        when = [self.PART_AT[0], self.PART_AT[1]]
        when += [a0 + (a1 - a0) * i / (n_layers - 1) for i in range(n_layers)]
        when += [self.NORM_AT]
        for p, (x0, x1), at in zip(parts, self.segments, when, strict=True):
            self.part_block(sc, p["sample"], x0, (x1 - x0) * PX_W, y,
                            op * self.appear(k, u, 1, at))
        # the names and the counts
        tt = self.segments[0]
        a = op * self.appear(k, u, 1, self.PART_AT[0] + 0.03)
        self.text(sc, (tt[0] + tt[1]) / 2, y - 0.042, "token table", a, SIZE, "mm")
        self.text(sc, (tt[0] + tt[1]) / 2, y + 0.045, parts[0]["text"], a, SIZE, "mm",
                  ink=WEIGHT_INK)
        lx0, lx1 = self.segments[2][0], self.segments[-2][1]
        a = op * self.appear(k, u, 1, a1 + 0.03)
        self.text(sc, (lx0 + lx1) / 2, y - 0.042, "%d layers" % n_layers, a, SIZE, "mm")
        self.text(sc, (lx0 + lx1) / 2, y + 0.045, "%s each" % m["layer_text"], a, SIZE, "mm",
                  ink=WEIGHT_INK)
        a = op * self.appear(k, u, 1, self.NORM_AT + 0.03)
        self.text(sc, 0.06, y + 0.09, "and positions %s, last norm %s" % (
            parts[1]["text"], parts[-1]["text"]), a, SIZE, "lm", ink=QUIET)
        steps = [(0.0, m["running_texts"][0])]
        for i, at in enumerate(when):
            steps.append((at, m["running_texts"][i]))
        b = op * self.appear(k, u, 1, self.PART_AT[0])
        u1 = u if k == 1 else 1.0
        self.running(sc, 0.06, 0.875, [(at, "added up: " + s) for at, s in steps], u1, b,
                     size=SIZE, ink=WEIGHT_INK)

    def part_block(self, sc, sample, x0, w, y, op, h=22.0, rows=3):
        """A part of the model, ``w`` px wide from layout ``x0``: real numbers of it as
        orange tiles (deeper: bigger), outlined so that each part shows."""
        if op <= 0.01:
            return
        w = max(float(w), 2.0)
        cols = max(1, int(round(w / 6.0)))
        vals = np.abs(np.asarray(sample, float))
        vals = np.resize(vals, rows * cols).reshape(rows, cols)
        cx = x0 + w / 2 / PX_W
        self.tiles(sc, vals, cx, y, w, h, op, ink=WEIGHT, lo=0.0, hi=self.hi, grid=0.2,
                   lift=0.3)
        self.rect(sc, self.P(cx, y), w + 2.0, h + 2.0, WEIGHT, op, border=dark(WEIGHT, 0.15),
                  border_w=1.0, radius=1.5, lift=0.32, fill_op=0.0)

    def memory(self, sc, k, u, op):
        """The server's memory as shelves, the model's parts on them as they are loaded."""
        m = self.meta
        x0, x1, top, bottom = self.MEM
        self.text(sc, x1, 0.375, "the server's memory", op, SIZE, "rm")
        self.rect(sc, self.P((x0 + x1) / 2, (top + bottom) / 2), (x1 - x0) * PX_W,
                  (bottom - top) * PX_H, soft(AXIS, 0.9), op, border=AXIS, border_w=1.4,
                  radius=6, lift=0.15)
        self.shelves(sc, x0 + 0.012, x1 - 0.012, self.SHELF_Y, op, lift=0.2)
        parts = m["parts"]
        # the token table and positions on the top shelf, six layers on each of the others
        rows = [parts[:2], parts[2:8], parts[8:]]
        u_load = u if k == 3 else 1.0
        n = sum(len(r) for r in rows)
        i = 0
        for shelf, row in zip(self.SHELF_Y, rows, strict=True):
            x = x0 + 0.02
            for p in row:
                w = max(p["count"] * self.px_per_number, 1.6)
                a0, a1 = self.LOAD_AT
                at = a0 + (a1 - a0) * i / (n - 1)
                a = op * smoothstep(at, at + 0.03, u_load)
                self.part_block(sc, p["sample"], x, w, shelf - 15 / PX_H, a)
                x += (w + 3.0) / PX_W
                i += 1
        a = op * (smoothstep(self.LOAD_AT[1], self.LOAD_AT[1] + 0.04, u_load))
        self.text(sc, (x0 + x1) / 2, bottom - 0.024, "%s bytes" % m["bytes_text"], a, SIZE,
                  "mm", ink=WEIGHT_INK)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        m = self.meta
        x = 0.2
        self.big(sc, x, 0.5, m["total_text"], on(u, 0.12, op), label="numbers in it")
        self.text(sc, x, 0.57, "learned in training, then fixed", on(u, 0.4, op), SIZE, "lm",
                  ink=QUIET)
        self.text(sc, x, 0.61, "an orange tile: one number", on(u, 0.55, op), SIZE, "lm",
                  ink=WEIGHT_INK)

    def phase_2(self, sc, u, op):
        """The bar, in ``steady``."""

    def phase_3(self, sc, u, op):
        m = self.meta
        x = 0.2
        a = on(u, 0.06, op)
        self.text(sc, x, 0.405, "one number: %s" % val(m["first"], 4), a, SIZE, "lm")
        for i in range(m["bytes_per_number"]):
            self.rect(sc, self.P(x + (14 + 30 * i) / PX_W, 0.45), 24.0, 20.0,
                      soft(WEIGHT, 0.6), on(u, 0.12 + 0.03 * i, op), border=WEIGHT,
                      border_w=1.4, radius=3, lift=0.3)
        self.text(sc, x + 130 / PX_W, 0.45, "4 bytes", on(u, 0.25, op), SIZE, "lm",
                  ink=WEIGHT_INK)
        b = on(u, 0.4, op)
        self.text(sc, x, 0.51, "%s × 4 bytes" % m["total_text"], b, SIZE, "lm")
        self.text(sc, x, 0.555, "= %s bytes" % m["bytes_text"], on(u, 0.5, op), HEAD, "lm",
                  strong=True)
        self.text(sc, x, 0.6, "about %s" % m["mb"], on(u, 0.6, op), SIZE, "lm",
                  ink=WEIGHT_INK)

    def phase_4(self, sc, u, op):
        fx, fy = self.FILE
        x0 = self.MEM[0]
        a = on(u, 0.08, op)
        self.arrow(sc, (fx + (self.FILE_W / 2 + 6) / PX_W, fy), (x0 - 0.01, fy), AXIS, a,
                   u=eased(u, 0.08, 0.18), width=2.0)
        self.text(sc, 0.06, 0.72, "memory: what the computer", on(u, 0.3, op), SIZE, "lm")
        self.text(sc, 0.06, 0.76, "is working with right now", on(u, 0.3, op), SIZE, "lm")
        self.running(sc, 0.06, 0.83, [(0.0, "loading the file"),
                                       (self.LOAD_AT[1] + 0.04, "loaded: once")],
                     u, on(u, 0.1, op), size=HEAD)

    def phase_5(self, sc, u, op):
        m = self.meta
        x0 = self.MEM[0]
        self.text(sc, 0.06, 0.375, "four people", on(u, 0.04, op), SIZE, "lm")
        for y, at in zip(self.PHONES_Y, self.ASK_AT, strict=True):
            self.phone(sc, 0.08, y, 22.0, 34.0, on(u, at - 0.04, op))
            self.arrow(sc, (0.1, y), (x0 - 0.008, y), DATA, on(u, at, op),
                       u=eased(u, at, at + 0.06), width=1.8)
        self.running(sc, 0.06, 0.8, [(0.0, "requests: 0")] + [
            (at + 0.06, "requests: %d" % (i + 1)) for i, at in enumerate(self.ASK_AT)], u,
            on(u, 0.04, op), size=HEAD)
        self.text(sc, 0.06, 0.85, "loaded: once, %s bytes" % m["bytes_text"], on(u, 0.04, op),
                  SIZE, "lm", ink=WEIGHT_INK)


ANIMATIONS = {"hosting-device-and-server": DeviceAndServer,
              "hosting-a-request-is-text": RequestIsText,
              "hosting-the-model-is-a-file": ModelIsAFile}
