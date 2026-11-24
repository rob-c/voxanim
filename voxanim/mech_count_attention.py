"""Animation 184 of the "next" catalogue (see :mod:`voxanim.mech`): counting the weights
of a transformer, one part at a time, with one rule.

:class:`CountingWeightsTransformer`: the small vision transformer's attention as four
blocks of 24 × 24 weights, and three heads that cut them up without adding any; the rest
of that model, part by part, to its 5,066; one block of GPT-2 small, tensor by tensor; the
twelve blocks, the two tables and the tied output layer, to 124,439,808; and where those
weights live, with the one line of PyTorch that counts them.

Written for students who find dense screens hard: one thing counted at a time, always in
the same places (the PyTorch line at the top, the parts down the left, a picture of the
layer in the middle, the rule and the sum on the right, the running total in a green box)
and always in the same colours (weights blue, biases amber, the running total green).

It draws :mod:`voxanim.study_count_attention`'s study; every number on screen is read from
its metadata and formatted here. Every phase says, bottom right, that it is exact
arithmetic.
"""

from __future__ import annotations

from . import layout
from .easing import smoothstep
from .mech import AMBER, AXIS, BLUE, GREEN, PAPER, VIOLET, dark, soft
from .mech_phys_apps import GREY, Phys, commas

VIT_TAG = "exact arithmetic: the small vision transformer's shapes"
GPT_TAG = "exact arithmetic: GPT-2 small's shapes"
BIG, HEAD, TEXT_M, TEXT_S, CODE = 0.046, 0.036, 0.033, 0.030, 0.031
WEIGHT, BIAS, TOTAL, CODE_INK = BLUE, AMBER, GREEN, VIOLET
W_TEXT, B_TEXT, T_TEXT = dark(BLUE, 0.25), dark(AMBER, 0.45), dark(GREEN, 0.45)

HEAD_Y = 0.30
CODE_X, CODE_Y = 0.40, 0.38           # the PyTorch line being counted
LIST_X0, LIST_X1, LIST_Y, LIST_STEP = 0.03, 0.34, 0.47, 0.07
PIC_X, PIC_Y, PIC_MAX = 0.56, 0.67, 175.0      # the picture of the layer (px)
RIGHT_X, RULE_Y, SUM_Y, RULE_W = 0.70, 0.47, 0.625, 0.27
BOX_X, BOX_Y, BOX_W, BOX_H = 0.83, 0.82, 260.0, 72.0
KEY_Y = 0.925                         # the colour key, on the row of the evidence tag
FADE_U = 0.025                        # the crossfade between two counted parts


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith("exact arithmetic: ")


def share(v):
    """A share as a percentage, with as many places as a small one needs."""
    v = 100.0 * float(v)
    if v == 0:
        return "0%"
    if v >= 1:
        return "%.0f%%" % v
    if v >= 0.1:
        return "%.1f%%" % v
    return "%.2f%%" % v


def starts(n, a=0.04, b=0.97):
    """Where each of ``n`` equal steps begins, as a fraction of the phase."""
    return [a + (b - a) * i / n for i in range(n)]


def alpha(u, i, s):
    """How visible step ``i`` (of the starts ``s``) is at ``u``: in, held, out before
    the next one comes in, so two steps never overlap."""
    a = smoothstep(s[i], s[i] + FADE_U, u)
    if i + 1 < len(s):
        a *= 1.0 - smoothstep(s[i + 1] - FADE_U, s[i + 1], u)
    return a


def current(u, s):
    """The step running at ``u``."""
    k = 0
    for i, v in enumerate(s):
        if u >= v:
            k = i
    return k


def expression(p):
    """A part's sum as ``(text, ink)`` pieces and its result, e.g.
    ``[("49 × 24", W), ("+ 24", B)]`` and ``"= 1,200"``."""
    kind, shape = p["kind"], p["shape"]
    if kind == "linear":
        a, b = shape
        w = "%s × %s" % (commas(a), commas(b))
        if p["times"] > 1:
            w = "%d × %s" % (p["times"], w)
        pieces = [(w, W_TEXT)]
        if p["bias"]:
            pieces.append(("+ %s" % commas(p["biases"]), B_TEXT))
    elif kind == "table":
        pieces = [("%s × %s" % (commas(shape[0]), commas(shape[1])), W_TEXT)]
    elif kind == "norm":
        pieces = [(commas(p["weights"]), W_TEXT), ("+ %s" % commas(p["biases"]), B_TEXT)]
    elif kind == "blocks":
        pieces = [("%d × %s" % (p["count"], commas(p["per_block"])), None)]
    else:                                                  # tied: nothing new
        pieces = [("shared: %s × %s" % (commas(shape[0]), commas(shape[1])), W_TEXT)]
    return pieces, "= %s" % commas(p["total"])


class CountingWeightsTransformer(Phys):
    """Attention's four blocks and its heads; the small ViT part by part; one GPT-2 block;
    all of GPT-2; where its weights live."""

    TITLE = "Counting the weights: a transformer"
    PHASES = (("attention", 11.0), ("ViT", 14.0), ("a block", 14.0), ("GPT-2", 11.5),
              ("where", 10.0))

    def setup(self):
        m = self.meta = self.study.metadata
        self.vit, self.gpt = m["vit"], m["gpt2"]
        self.att = self.vit["attention"]
        self.vit_parts = self.vit["parts"]
        self.block = self.gpt["block"]
        self.whole = self.gpt["parts"]
        self.groups = self.gpt["groups"]
        g = self.gpt
        qkv = {p["key"]: p for p in self.block}["attn.c_attn"]["shape"][1] // g["d"]
        self.rules = {
            "patch": ("each output has one weight per input, plus one bias",
                      "a %d × %d patch is %d pixels" % (self.vit["patch"], self.vit["patch"],
                                                       self.vit["patch"] ** 2)),
            "cls": ("one learned row of %d numbers" % self.vit["d"], ""),
            "pos": ("one learned row for each position", "%d positions: %d patches + 1"
                    % (self.vit["n_tok"], self.vit["n_tok"] - 1)),
            "attention": ("the four blocks counted before; no biases", ""),
            "mlp": ("each output has one weight per input, plus one bias", ""),
            "output": ("each output has one weight per input, plus one bias",
                       "one output for each digit"),
            "ln_1": ("a norm learns a scale and a shift for each number", ""),
            "attn.c_attn": ("each output has one weight per input, plus one bias",
                            "Q, K and V side by side: %d × %d" % (qkv, g["d"])),
            "attn.c_proj": ("each output has one weight per input, plus one bias", ""),
            "ln_2": ("a norm learns a scale and a shift for each number", ""),
            "mlp.c_fc": ("each output has one weight per input, plus one bias",
                         "%d times wider: %s" % (g["mlp"] // g["d"], commas(g["mlp"]))),
            "mlp.c_proj": ("each output has one weight per input, plus one bias",
                           "back to %d" % g["d"]),
            "wte": ("one learned row for each token", ""),
            "wpe": ("one learned row for each position", ""),
            "h": ("%d copies of the block, each with its own weights" % g["layers"], ""),
            "ln_f": ("a norm learns a scale and a shift for each number", ""),
            "lm_head": ("the output layer reuses the token table", "so it adds no weights"),
        }

    def subtitle_for(self, k):
        v, g, a = self.vit, self.gpt, self.att
        top = self.groups[0], self.groups[1]
        return [
            "The vision transformer from A self-attention training step: attention is four "
            "blocks of %d × %d weights, %s in all. Heads cut them up and add nothing."
            % (a["d"], a["d"], commas(a["total"])),
            "The rest of it, one part at a time, with the same rule: %s numbers to learn, "
            "exactly what its own arrays hold." % commas(v["total"]),
            "One block of GPT-2 small, the same way: %s weights and biases, in six parts."
            % commas(g["per_block"]),
            "%d blocks, the token table and the positions make %s. The output layer reuses "
            "the token table, so it adds nothing." % (g["layers"], commas(g["total"])),
            "The %s and the %s hold most: %s and %s. In each block attention holds a third, "
            "the MLP two thirds." % (top[0]["label"], top[1]["label"], share(top[0]["share"]),
                                     share(top[1]["share"])),
        ][k]

    # ------------------------------------------------------------------ the shared pieces
    def heading(self, sc, s, op):
        self.text(sc, LIST_X0, HEAD_Y, s, op, anchor="lm", strong=True, size=HEAD)

    def code(self, sc, s, op, fx=CODE_X, fy=CODE_Y, on=True):
        """A line of PyTorch in the monospace font, in a box that lights up while it is
        being counted."""
        if op <= 0.01 or not s:
            return
        w = layout.text_width(s, CODE, "mono") * self.cam.width + 22.0
        fill = soft(CODE_INK, 0.78) if on else PAPER
        self.rect(sc, self.P(fx + self.frac(w)[0] / 2, fy), w, 34.0, fill, op,
                  border=CODE_INK if on else AXIS, border_w=1.6 if on else 1.0, radius=6)
        self.label(sc, self.P(fx + self.frac(11.0)[0], fy), s, CODE, anchor="lm", opacity=op,
                   ink=layout.STRONG_INK, kind="mono")

    def pieces(self, sc, fx, fy, pieces, op, size=BIG):
        """Coloured pieces of one line, left to right; returns the x where it ends."""
        gap = layout.text_width(" ", size)
        x = fx
        for s, ink in pieces:
            self.text(sc, x, fy, s, op, size=size, anchor="lm",
                      ink=layout.STRONG_INK if ink is None else ink)
            x += self.frac((layout.text_width(s, size) + gap) * self.cam.width)[0]
        return x

    def rule(self, sc, first, second, op):
        """The rule in words, and a short note under it."""
        y = self.lines(sc, RIGHT_X, RULE_Y, layout.wrap(first, TEXT_M, RULE_W), op,
                       size=TEXT_M, step=0.046)
        if second:
            self.text(sc, RIGHT_X, y + 0.006, second, op, anchor="lm", size=TEXT_S,
                      ink=layout.NOTE_INK)

    def worked(self, sc, p, op):
        """The sum with the real numbers, in two lines."""
        pieces, result = expression(p)
        self.pieces(sc, RIGHT_X, SUM_Y, pieces, op)
        self.text(sc, RIGHT_X, SUM_Y + 0.075, result, op, size=BIG, anchor="lm", strong=True)

    def legend(self, sc, op):
        """The colours, the same in every phase."""
        x = LIST_X0
        for name, ink in (("weights", WEIGHT), ("biases", BIAS), ("running total", TOTAL)):
            self.box(sc, x + self.frac(7)[0], KEY_Y, 14, 14, ink, op, radius=3)
            self.text(sc, x + self.frac(20)[0], KEY_Y, name, op, anchor="lm", size=TEXT_S,
                      ink=dark(ink, 0.35))
            x += self.frac(44 + layout.text_width(name, TEXT_S) * self.cam.width)[0]

    # ------------------------------------------------------------------ pictures
    def weights_rect(self, sc, fx, fy, w, h, op, ink=WEIGHT, fill_op=1.0):
        self.rect(sc, self.P(fx, fy), max(w, 4.0), max(h, 4.0), ink, op, border=dark(ink, 0.3),
                  border_w=1.2, radius=2, fill_op=fill_op)

    def picture(self, sc, p, scale, op):
        """The layer as blocks: its inputs down the left, its outputs along the top, the grid
        of weights between them and the biases as a row under it."""
        kind, shape = p["kind"], p["shape"]
        cx, cy = PIC_X, PIC_Y
        if kind == "linear" and p["times"] > 1:          # Q, K, V and O, two by two
            side = shape[0] * scale
            dx, dy = self.frac(side + 8, side + 8)
            for i, label in enumerate(("Q", "K", "V", "O")):
                x, y = cx + (i % 2 - 0.5) * dx, cy + (i // 2 - 0.5) * dy
                self.weights_rect(sc, x, y, side, side, op)
                self.text(sc, x, y, label, op, size=TEXT_M, ink=(1.0, 1.0, 1.0))
            self.text(sc, cx, cy - dy - 0.02, "%s × %s each" % (commas(shape[0]),
                                                                commas(shape[1])),
                      op, size=TEXT_S)
            return
        if kind == "linear":
            h, w = shape[0] * scale, shape[1] * scale
            dw, dh = self.frac(w, h)
            self.weights_rect(sc, cx, cy, w, h, op)
            ix = cx - dw / 2 - self.frac(10)[0]
            self.rect(sc, self.P(ix, cy), 5, h, GREY, op, radius=2)
            self.text(sc, ix - 0.01, cy, "%s in" % commas(shape[0]), op, anchor="rm",
                      size=TEXT_S)
            oy = cy - dh / 2 - self.frac(0, 10)[1]
            self.rect(sc, self.P(cx, oy), w, 5, GREY, op, radius=2)
            self.text(sc, cx, oy - 0.03, "%s out" % commas(shape[1]), op, size=TEXT_S)
            if p["bias"]:
                self.rect(sc, self.P(cx, cy + dh / 2 + self.frac(0, 11)[1]), w, 8, BIAS, op,
                          border=dark(BIAS, 0.3), border_w=1.0, radius=2)
            return
        if kind in ("table", "tied"):
            rows, width = shape
            h, w = rows * scale, width * scale
            cut = h > PIC_MAX
            h = min(max(h, 5.0), PIC_MAX)
            dw, dh = self.frac(w, h)
            tied = kind == "tied"
            self.weights_rect(sc, cx, cy, w, h, op, fill_op=0.25 if tied else 1.0)
            if cut:
                for j in range(3):
                    self.disk(sc, self.P(cx, cy + dh / 2 + self.frac(0, 9 + 8 * j)[1]), 2.2,
                              WEIGHT, op, lift=0.6, outline=0.0)
            unit = p["unit"] if not tied else "tokens"
            left = "1 row" if rows == 1 else "%s %s" % (commas(rows), unit)
            self.text(sc, cx - dw / 2 - 0.012, cy, left, op, anchor="rm", size=TEXT_S)
            self.text(sc, cx, cy - dh / 2 - 0.028, "%s wide" % commas(width), op, size=TEXT_S)
            return
        if kind == "norm":
            w = shape[0] * scale
            for j, (name, ink) in enumerate((("scale", WEIGHT), ("shift", BIAS))):
                y = cy + (j - 0.5) * self.frac(0, 30)[1]
                self.rect(sc, self.P(cx, y), w, 10, ink, op, border=dark(ink, 0.3),
                          border_w=1.0, radius=2)
                self.text(sc, cx - self.frac(w)[0] / 2 - 0.012, y, name, op, anchor="rm",
                          size=TEXT_S, ink=dark(ink, 0.35))
            self.text(sc, cx, cy - 0.075, "%s wide" % commas(shape[0]), op, size=TEXT_S)
            return
        if kind == "blocks":
            n = p["count"]
            step = 11.0
            top = cy - self.frac(0, step * (n - 1))[1] / 2
            for j in range(n):
                self.rect(sc, self.P(cx, top + self.frac(0, step * j)[1]), 120, 8, WEIGHT, op,
                          border=dark(WEIGHT, 0.3), border_w=1.0, radius=2)
            self.text(sc, cx - self.frac(60)[0] - 0.012, cy, "× %d" % n, op, anchor="rm",
                      size=TEXT_M, strong=True)

    # ------------------------------------------------------------------ one part at a time
    def box_frame(self, sc, op):
        """The green box the running total sits in."""
        self.rect(sc, self.P(BOX_X, BOX_Y), BOX_W, BOX_H, soft(TOTAL, 0.78), op, border=TOTAL,
                  border_w=1.8, radius=8)
        self.text(sc, BOX_X, BOX_Y - 0.035, "running total", op, size=TEXT_S, ink=T_TEXT)

    def running(self, sc, value, op):
        """The running total's number, in the box."""
        self.text(sc, BOX_X, BOX_Y + 0.02, commas(value), op, size=BIG, ink=T_TEXT)

    def result(self, sc, value, op):
        """``= value`` under a sum."""
        self.text(sc, RIGHT_X, SUM_Y + 0.075, "= %s" % commas(value), op, size=BIG,
                  anchor="lm", strong=True)

    def counting(self, sc, u, op, parts, scale, final, tag):
        """Count ``parts`` in turn: each lights its row and its PyTorch line, draws its
        picture, its rule and its sum, and adds to the running total; then ``final``
        (rule, note) over the total."""
        s = starts(len(parts) + 1)
        k = current(u, s)
        for i, p in enumerate(parts):
            seen = op * smoothstep(s[i], s[i] + FADE_U, u)
            if seen <= 0.01:
                continue
            y = LIST_Y + LIST_STEP * i
            if i == k:
                self.rect(sc, self.P((LIST_X0 + LIST_X1) / 2, y),
                          self.span(LIST_X1 - LIST_X0 + 0.03)[0], 36.0, soft(CODE_INK, 0.82),
                          alpha(u, i, s) * op, border=CODE_INK, border_w=1.4, radius=6)
            self.text(sc, LIST_X0, y, p["label"], seen, anchor="lm", size=TEXT_M)
            self.text(sc, LIST_X1, y, commas(p["total"]), seen, anchor="rm", size=TEXT_M,
                      strong=True)
            a = op * alpha(u, i, s)
            if a > 0.01:
                self.code(sc, p["code"], a)
                self.picture(sc, p, scale, a)
                first, second = self.rules[p["key"]]
                self.rule(sc, first, second, a)
                self.worked(sc, p, a)
        a = op * alpha(u, len(parts), s)
        if a > 0.01:
            self.rule(sc, final[0], final[1], a)
            self.pieces(sc, RIGHT_X, SUM_Y, [("%s + … + %s" % (commas(parts[0]["total"]),
                                                             commas(parts[-1]["total"])),
                                              None)], a, size=TEXT_M)
            self.result(sc, parts[-1]["running"], a)
        if u >= s[0]:                    # the total stays on screen between two parts,
            done = current(u - FADE_U, s)    # and changes once the new part is in
            self.running(sc, parts[min(done, len(parts) - 1)]["running"], op)
        self.box_frame(sc, op)
        self.legend(sc, op)
        self.tag(sc, tag, op)

    # ------------------------------------------------------------------ phases
    STEPS_1 = (0.04, 0.13, 0.22, 0.31, 0.42, 0.62)     # Q, K, V, O, all four, three heads

    def phase_1(self, sc, u, op):
        a, d = self.att, self.att["d"]
        mats = a["matrices"]
        self.heading(sc, "The small vision transformer: attention", op)
        s = self.STEPS_1
        split = smoothstep(s[5] + 0.03, s[5] + 0.12, u)
        side = 100.0
        _, sh = self.frac(side, side)
        heads, hw = a["heads"], a["head_width"]
        for i, m in enumerate(mats):
            seen = op * smoothstep(s[i], s[i] + FADE_U, u)
            if seen <= 0.01:
                continue
            x, y = 0.10 + 0.145 * i, 0.625
            self.text(sc, x, y - sh / 2 - 0.04, m["label"], seen, size=TEXT_M, strong=True)
            gap = 8.0 * split
            piece = (side - gap * (heads - 1)) / heads
            if split < 0.01:
                self.weights_rect(sc, x, y, side, side, seen)
            for h in range(heads if split >= 0.01 else 0):
                off = (h - (heads - 1) / 2) * (piece + gap)
                if m["key"] == "Wo":              # the output block is cut by its inputs (rows)
                    self.weights_rect(sc, x, y + self.frac(0, off)[1], side, piece, seen)
                else:                            # Q, K, V by their outputs (columns)
                    self.weights_rect(sc, x + self.frac(off)[0], y, piece, side, seen)
            whole = seen * (1 - smoothstep(s[5], s[5] + FADE_U, u))
            self.text(sc, x, y + sh / 2 + 0.04, "%d × %d" % (d, d), whole, size=TEXT_M)
            cut = ("%d × (%d × %d)" % (heads, d, hw) if m["key"] != "Wo"
                   else "%d × (%d × %d)" % (heads, hw, d))
            self.text(sc, x, y + sh / 2 + 0.04, cut, seen * smoothstep(s[5] + 0.1, s[5] + 0.14,
                                                                        u), size=TEXT_S)
        for i, m in enumerate(mats):
            b = op * alpha(u, i, s)
            if b > 0.01:
                self.code(sc, m["code"], b)
                self.rule(sc, "each output has one weight per input", "no biases here", b)
                self.worked(sc, m, b)
        b = op * alpha(u, 4, s)
        if b > 0.01:
            self.code(sc, a["code_one_head"], b)
            self.rule(sc, "the four together make one attention layer", "here with 1 head", b)
            self.pieces(sc, RIGHT_X, SUM_Y, [("%d × %d × %d" % (len(mats), d, d), W_TEXT)], b)
            self.result(sc, a["by_heads"]["1"], b)
        b = op * alpha(u, 5, s)
        if b > 0.01:
            self.code(sc, a["code_heads"], b)
            self.rule(sc, "%d heads cut each block into %d pieces" % (heads, heads),
                      "the count stays the same", b)
            self.pieces(sc, RIGHT_X, SUM_Y, [("%d × %d × %d × %d" % (len(mats), heads, d, hw),
                                             W_TEXT)], b)
            self.result(sc, a["by_heads"][str(heads)], b)
        k = current(u - FADE_U, s)       # the total stays on screen between two steps,
        if u >= s[0]:                    # and changes once the new step is in
            self.running(sc, mats[k]["running"] if k < len(mats) else a["total"], op)
        self.box_frame(sc, op)
        self.legend(sc, op)
        self.tag(sc, VIT_TAG, op)

    def phase_2(self, sc, u, op):
        self.heading(sc, "The whole small vision transformer", op)
        self.counting(sc, u, op, self.vit_parts, PIC_MAX / self.vit["patch"] ** 2,
                      ("all six parts added up", "vit.py holds exactly %s numbers"
                       % commas(self.vit["counted"])), VIT_TAG)

    def phase_3(self, sc, u, op):
        self.heading(sc, "One block of GPT-2 small", op)
        self.counting(sc, u, op, self.block, PIC_MAX / self.gpt["mlp"],
                      ("one block: all six parts", ""), GPT_TAG)

    def phase_4(self, sc, u, op):
        self.heading(sc, "All of GPT-2 small", op)
        self.counting(sc, u, op, self.whole, PIC_MAX / self.gpt["mlp"],
                      ("every number GPT-2 small learns", ""), GPT_TAG)

    def phase_5(self, sc, u, op):
        g = self.gpt
        self.heading(sc, "Where GPT-2's weights live", op)
        top = max(r["total"] for r in self.groups)
        x0, room = 0.25, 380.0
        for i, r in enumerate(self.groups):
            y = 0.40 + 0.062 * i
            grow = smoothstep(0.04 + 0.06 * i, 0.22 + 0.06 * i, u)
            value = ("%s · %s" % (commas(r["total"]), share(r["share"])) if r["total"]
                     else "0: it reuses the token table")
            self.hbar(sc, x0, y, room * r["total"] / top, WEIGHT, op, r["label"], value,
                      u=grow, thick=24.0)
        late = op * smoothstep(0.45, 0.52, u)
        self.lines(sc, 0.62, 0.59, [("in each block:", None),
                                    ("attention %s" % share(g["attention_share_of_block"]),
                                     W_TEXT),
                                    ("MLP %s" % share(g["mlp_share_of_block"]), W_TEXT)],
                   late, size=TEXT_M, step=0.046)
        end = op * smoothstep(0.62, 0.7, u)
        line = self.meta["count_line"]
        self.code(sc, line, end, fx=LIST_X0, fy=BOX_Y)
        cw = self.frac(layout.text_width(line, CODE, "mono") * self.cam.width + 22)[0]
        self.arrow(sc, (LIST_X0 + cw + 0.012, BOX_Y),
                   (BOX_X - self.frac(BOX_W)[0] / 2 - 0.012, BOX_Y), TOTAL, end)
        self.box_frame(sc, op)
        self.running(sc, g["total"], op)            # the total from phase 4, kept
        self.legend(sc, op)
        self.tag(sc, GPT_TAG, op)


ANIMATIONS = {"counting-weights-transformer": CountingWeightsTransformer}
