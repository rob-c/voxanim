"""Animations 265-268 of "Blocks and code", "From autoencoder to GAN" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_zc`: an autoencoder's two blocks
wired again as a GAN (``examples/ae_gan.py``).

* :class:`Rewire` (``ae-to-gan-rewire``): the encoder and the decoder as two orange blocks
  (205,072 and 205,840 numbers); a held-out digit squeezed through its 16-number code and
  rebuilt, scored pixel by pixel; unplugged; the decoder slides down to the generator's place,
  its tiles unchanged, and is fed a made-up code; the encoder's last layer is left out, its
  body slides to the judge's place, and a new small head (257 numbers, "real?") is added;
  the GAN's wiring, noise -> generator -> judge <- real; then the two paths side by side:
  every digit through 16 numbers, or no digit through a code at all.
* :class:`Blurry` (``ae-gan-blurry-or-sharp``): real digits, their rebuilds, and the decoder
  on the same made-up codes before and after the GAN game; under each column its share of
  grey pixels, then the classifier's certainty; why rebuilds are grey: one pixel's score.
* :class:`Rebuild` (``ae-gan-rebuild-or-invent``): four digits and their rebuilds, read by
  the classifier; 928 of 1,000 right; the generator's only input is 16 made-up numbers; what
  1,000 made digits are read as; the 16 from the fixed codes, as they are.
* :class:`Losses` (``ae-gan-one-loss-or-two``): the autoencoder's one loss falling; the
  judge's and the generator's pulling against each other, near the "cannot tell" lines; the
  same codes' pictures at steps 0, 500, 2,000 and 6,000.

The alphabet: digits and codes are data (blue tiles on blue cards); the blocks are weights
(orange cards of orange tiles, corners of the real tensors): the decoder and generator one
orange, the encoder and judge a deeper one, the same shapes in all four; the judge's new
head is a small card of the judge's colour with a dark border, labelled new; losses violet.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from . import color as C
from . import layout
from .blocks import (
    BIG,
    DATA,
    HEAD,
    INK,
    LOSS,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, dark, soft

DEC = WEIGHT                         # the decoder, later the generator
ENC = dark(WEIGHT, 0.32)             # the encoder, later the judge: a deeper orange
DEC_INK, ENC_INK = dark(DEC, 0.35), dark(ENC, 0.3)
DATA_INK = dark(DATA, 0.3)
LOSS_INK = dark(LOSS, 0.3)
LEFT_OUT = soft(AXIS, 0.25)          # the encoder's last layer, not used by the judge

# a block: an orange card, each layer a patch of tiles 20 px wide, its height its output size
CARD_H = 78.0
LAYER_W = 20.0
LAYER_H = dict(enc1=48.0, enc2=20.0, dec1=48.0, dec2=64.0)
WIDE, NARROW = 64.0, 36.0            # a block's card: two layers, one
CODE_W, CODE_H = 10.0, 56.0          # 16 numbers, as a column of tiles
CODE_LO, CODE_HI = -10.0, 10.0       # the codes' shading: paper at -10, full ink at +10
TILE = 2.0                           # a digit's tiles, px


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def strong(patch):
    """The magnitude drawn at full ink: the 85th percentile of the tiles shown."""
    return float(max(np.percentile(np.asarray(patch, float), 85), 1e-9))


def swap(p):
    """``(out, in)`` opacities for one text giving way to another in place as ``p`` goes
    0..1: the first fades out over the first half, the second in over the second, never
    both at once."""
    p = float(np.clip(p, 0.0, 1.0))
    return 1.0 - smoothstep(0.0, 0.5, p), smoothstep(0.5, 1.0, p)


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class AeGan(Blocks):
    """What the four share: the blocks, digits, codes, timing by phase."""

    def setup(self):
        self.meta = self.study.metadata

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the title)."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths, strict=True)):
            ink = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", ink, 1.0, "la")
            x += w + gap

    # ------------------------------------------------------------------ time
    def time_of(self, k, u):
        a, b = self.spans[k]
        return a + (b - a) * u

    def prog(self, k, u0, u1):
        """0 before ``u0`` of phase ``k``, 1 after ``u1``, eased between (by the clock, so
        it holds whatever phase is under way)."""
        t0, t1 = self.time_of(k, u0), self.time_of(k, u1)
        return ease_in_out(float(np.clip((self._t - t0) / (t1 - t0), 0.0, 1.0)))

    def at(self, k, u, width=0.5):
        """1 once phase ``k`` has reached ``u`` (fading in over ``width`` s); 0 before."""
        t0 = self.time_of(k, u)
        return smoothstep(t0, t0 + width, self._t)

    # ------------------------------------------------------------------ the alphabet
    def block(self, sc, kind, fx, fy, op, patches, l2=1.0, l2_grey=0.0, narrow=0.0,
              ghost=False):
        """The encoder (``kind`` "enc": layers 784->256, 256->16) or the decoder ("dec":
        16->256, 256->784) as an orange card of two patches of its real weights; ``narrow``
        (0..1) closes the card round the first layer alone (the judge's body), ``l2`` fades
        the second layer and ``l2_grey`` greys it; ``ghost``: an outline where it was."""
        if op <= 0.01:
            return
        ink = ENC if kind == "enc" else DEC
        w = WIDE - (WIDE - NARROW) * narrow
        c = self.P(float(fx), float(fy))
        if ghost:
            self.rect(sc, c, w, CARD_H, soft(ink, 0.5), op, border=soft(ink, 0.35),
                      border_w=1.6, radius=8, lift=0.2, fill_op=0.0)
            return
        self.rect(sc, c, w, CARD_H, soft(ink, 0.84), op, border=ink, border_w=2.0, radius=8,
                  lift=0.2)
        names = ("enc1", "enc2") if kind == "enc" else ("dec1", "dec2")
        dx = self.frac(13.0)[0]
        x1 = fx - dx * (1.0 - narrow)
        self.tiles(sc, patches[names[0]], x1, fy, LAYER_W, LAYER_H[names[0]], op, ink=ink,
                   lo=0.0, hi=strong(patches[names[0]]), grid=0.25, lift=0.24)
        if l2 > 0.01:
            ink2 = tuple(float(v) for v in C.mix(ink, LEFT_OUT, l2_grey))
            self.tiles(sc, patches[names[1]], fx + dx, fy, LAYER_W, LAYER_H[names[1]],
                       op * l2, ink=ink2, lo=0.0, hi=strong(patches[names[1]]), grid=0.25,
                       lift=0.24)

    def head_block(self, sc, fx, fy, op, patch):
        """The judge's new last layer, 256 -> 1: a small card of the judge's colour with a
        dark border (new)."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), 44.0, 20.0, soft(ENC, 0.84), op, border=INK,
                  border_w=2.2, radius=5, lift=0.2)
        self.tiles(sc, patch, fx, fy, 30.0, 6.0, op, ink=ENC, lo=0.0, hi=strong(patch),
                   grid=0.25, lift=0.24)

    def digit(self, sc, img, fx, fy, op, tile=TILE, lift=0.2):
        """A picture (pixels 0..1) as a data card of blue tiles."""
        if op <= 0.01:
            return
        self.digit_card(sc, img, fx, fy, tile, op, lift=lift, lo=0.0, hi=1.0)

    @staticmethod
    def card_size(tile=TILE):
        side = 28 * tile
        return side + 14.0, side + 20.0

    def code_strip(self, sc, values, fx, fy, op, ring=False):
        """16 numbers (a code) as a column of blue tiles, on the codes' fixed scale."""
        if op <= 0.01:
            return
        v = np.asarray(values, float)[:, None]
        self.rect(sc, self.P(fx, fy), CODE_W + 6, CODE_H + 6, soft(DATA, 0.82), op,
                  border=INK if ring else DATA, border_w=2.4 if ring else 1.4, radius=3,
                  lift=0.2)
        self.tiles(sc, v, fx, fy, CODE_W, CODE_H, op, lo=CODE_LO, hi=CODE_HI, grid=0.2,
                   lift=0.24)

    def half(self, tile=TILE):
        """Half a digit card's width and height, as layout fractions."""
        w, h = self.card_size(tile)
        return self.frac(w / 2, h / 2)


# ====================================================================== 265
class Rewire(AeGan):
    """The autoencoder's two blocks; a digit squeezed and rebuilt; unplugged; the decoder to
    the generator's place, the encoder's body to the judge's with a new head; the GAN; the
    two paths."""

    TITLE = "Same blocks, wired again"
    PHASES = (("two blocks", 9.0), ("squeeze", 11.0), ("decoder moves", 10.0),
              ("encoder moves", 11.0), ("the GAN", 10.0), ("two paths", 9.5))
    LAST = "Improved deep embedded clustering"
    NEXT = "Blurry or sharp"
    STATION = (None, "loss", None, None, "forward", None)
    ROW_A, ROW_G = 0.48, 0.715
    AX: ClassVar[dict] = dict(digit=0.10, enc=0.195, code=0.262, dec=0.33, back=0.425)
    GX: ClassVar[dict] = dict(noise=0.085, gen=0.16, made=0.255, judge=0.345, real=0.43)
    HEAD_AT = (0.345, 0.85)
    NAME_DY = 0.095                  # a block's name, above it
    UNDER_A = 0.585                  # the line under row A

    # the moves (phase, u from, u to)
    DEC_DOWN, DEC_LEFT = (2, 0.3, 0.46), (2, 0.5, 0.66)
    ENC_GREY, ENC_GONE, ENC_NARROW = (3, 0.04, 0.12), (3, 0.38, 0.45), (3, 0.4, 0.47)
    ENC_RIGHT, ENC_DOWN = (3, 0.47, 0.6), (3, 0.62, 0.74)
    NOISE_AT, MADE_AT, HEAD_SHOWN = (2, 0.68), (2, 0.76), (3, 0.76)

    def setup(self):
        super().setup()
        m = self.meta
        self.patches = m["patches"]
        self.CODE = (
            Code("ag-blocks", run=((0.0, 0), (0.45, 3))),
            Code("ag-ae", run=((0.0, 0), (0.42, 1), (0.62, 2))),
            Code("ag-rewire", run=((0.0, None), (0.3, 0))),
            Code("ag-rewire", run=((0.0, 1), (0.47, 2), (0.76, 3))),
            Code("ag-judge", run=((0.0, 0), (0.2, 1), (0.42, 3), (0.62, 4))),
            Code("ag-judge", run=((0.0, 0),)))
        c = m["counts_text"]
        self.EVIDENCE = (
            exact("in × out + out, each layer"),
            measured("held-out digit %d, after 3,000 steps" % 0),
            measured("the decoder on a made-up code"),
            exact("%s left out; 256 × 1 + 1 = %s new" % (c["enc2"], c["head"])),
            measured("a real digit, a made one"),
            exact("the script's wiring"))

    def heading_for(self, k, u):
        if k == 2:
            return "unplugged: the weights stay" if u < 0.3 else \
                "the decoder becomes the generator"
        if k == 3:
            return ("its last layer is left out" if u < 0.45 else
                    "its body becomes the judge" if u < 0.74 else "one new layer: real?")
        return ("an autoencoder: two blocks", "every digit squeezed through 16 numbers", "",
                "", "noise → generator → judge ← real", "autoencoder above, GAN below")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = m["counts_text"]
        return [
            "An autoencoder is two blocks: the encoder (%s numbers) and the decoder (%s)."
            % (c["encoder"], c["decoder"]),
            "A held-out %d is squeezed to 16 numbers and rebuilt, then scored pixel by pixel: "
            "loss %s." % (m["digit_label"], m["digit_loss_text"]),
            "Unplugged, the decoder moves, every number kept: as the generator it turns "
            "made-up codes into pictures.",
            "The encoder's last layer is left out; its body, every number kept, is the judge. "
            "Only its head is new: %s numbers." % c["head"],
            "The judge sees made digits and real ones. Its new head knows nothing yet; the GAN "
            "game trains both blocks on.",
            "The autoencoder sends every digit through 16 numbers. In the GAN no digit "
            "goes through a code: the judge scores whole pictures.",
        ][k]

    # ------------------------------------------------------------------ where the blocks are
    def dec_pos(self):
        down, left = self.prog(*self.DEC_DOWN), self.prog(*self.DEC_LEFT)
        return (self.AX["dec"] + (self.GX["gen"] - self.AX["dec"]) * left,
                self.ROW_A + (self.ROW_G - self.ROW_A) * down)

    def enc_pos(self):
        right, down = self.prog(*self.ENC_RIGHT), self.prog(*self.ENC_DOWN)
        return (self.AX["enc"] + (self.GX["judge"] - self.AX["enc"]) * right,
                self.ROW_A + (self.ROW_G - self.ROW_A) * down)

    def edge(self, fx, w_px, side):
        """The layout x of a card's left (-1) or right (+1) edge, a 3 px gap beyond."""
        return fx + side * self.frac(w_px / 2 + 3.0)[0]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a0 = op * (self.at(0, 0.03) if k == 0 else 1.0)
        b0 = op * (self.at(0, 0.12) if k == 0 else 1.0)
        # the encoder, then the judge's body
        ex, ey = self.enc_pos()
        narrow = self.prog(*self.ENC_NARROW)
        self.block(sc, "enc", ex, ey, a0, self.patches, l2=1.0 - self.prog(*self.ENC_GONE),
                   l2_grey=self.prog(*self.ENC_GREY), narrow=narrow)
        was, now = swap(self.prog(*self.ENC_DOWN))
        self.text(sc, ex, ey - self.NAME_DY, "encoder", a0 * was, SIZE, ink=ENC_INK)
        self.text(sc, ex, ey - self.NAME_DY, "judge", a0 * now, SIZE, ink=ENC_INK)
        # the decoder, then the generator
        dx, dy = self.dec_pos()
        self.block(sc, "dec", dx, dy, b0, self.patches)
        was, now = swap(self.prog(*self.DEC_LEFT))
        self.text(sc, dx, dy - self.NAME_DY, "decoder", b0 * was, SIZE, ink=DEC_INK)
        self.text(sc, dx, dy - self.NAME_DY, "generator", b0 * now, SIZE, ink=DEC_INK)
        G, y = self.GX, self.ROW_G
        # the made-up code, and what the generator makes of it
        n = op * self.at(*self.NOISE_AT)
        self.code_strip(sc, m["made_code"], G["noise"], y, n)
        self.text(sc, G["noise"], y - self.NAME_DY, "made up", n, SIZE, ink=DATA_INK)
        self.arrow(sc, (G["noise"] + self.frac(9.0)[0], y), (self.edge(G["gen"], WIDE, -1), y),
                   AXIS, n, width=1.8)
        md = op * self.at(*self.MADE_AT)
        self.arrow(sc, (self.edge(G["gen"], WIDE, 1), y),
                   (self.edge(G["made"], self.card_size()[0], -1), y), AXIS, md, width=1.8)
        self.digit(sc, m["made"], G["made"], y, md)
        # the judge's new head
        hd = op * self.at(*self.HEAD_SHOWN)
        hx, hy = self.HEAD_AT
        self.arrow(sc, (hx, y + self.frac(0, CARD_H / 2 + 2)[1]), (hx, hy - self.frac(0, 13)[1]),
                   AXIS, hd, width=1.8)
        self.head_block(sc, hx, hy, hd, self.patches["head"])
        self.text(sc, hx - 0.032, hy, "new: %s" % m["counts_text"]["head"], hd, SIZE, "rm",
                  strong=True)
        # the real digit, and the judge's answer
        r = op * (self.at(4, 0.05) if k <= 4 else 1.0) * float(k >= 4)
        self.digit(sc, m["real"], G["real"], y, r)
        self.text(sc, G["real"], y - self.NAME_DY, "real", r, SIZE, ink=DATA_INK)
        self.arrow(sc, (self.edge(G["real"], self.card_size()[0], -1), y),
                   (self.edge(G["judge"], NARROW, 1), y), AXIS, r, width=1.8)
        f = op * (self.at(4, 0.14) if k <= 4 else 1.0) * float(k >= 4)
        self.arrow(sc, (self.edge(G["made"], self.card_size()[0], 1), y),
                   (self.edge(G["judge"], NARROW, -1), y), AXIS, f, width=1.8)
        o = op * (self.at(4, 0.3) if k <= 4 else 1.0) * float(k >= 4)
        self.arrow(sc, (hx + self.frac(25.0)[0], hy), (hx + 0.045, hy), AXIS, o, width=1.8)
        self.text(sc, hx + 0.052, hy, "real?", o, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        c = self.meta["counts_text"]
        for key, x, at in (("encoder", self.AX["enc"], 0.25), ("decoder", self.AX["dec"],
                                                                0.35)):
            a = on(u, at, op)
            self.text(sc, x, self.UNDER_A, c[key], a, SIZE, strong=True)
            self.text(sc, x, self.UNDER_A + 0.035, "numbers", a, SIZE)

    def phase_2(self, sc, u, op):
        m, X, y = self.meta, self.AX, self.ROW_A
        cw = self.card_size()[0]
        d = on(u, 0.03, op)
        self.digit(sc, m["digit"], X["digit"], y, d)
        steps = ((X["digit"], cw, X["enc"], WIDE, 0.1), (X["enc"], WIDE, X["code"], CODE_W + 6,
                                                         0.2),
                 (X["code"], CODE_W + 6, X["dec"], WIDE, 0.36), (X["dec"], WIDE, X["back"], cw,
                                                                 0.44))
        for x0, w0, x1, w1, at in steps:
            a, b = (self.edge(x0, w0, 1), y), (self.edge(x1, w1, -1), y)
            self.arrow(sc, a, b, AXIS, on(u, at, op), width=1.8)
            self.pulse(sc, a, b, (u - at) / 0.07, DATA, op)
        c = on(u, 0.27, op)
        self.code_strip(sc, m["code"], X["code"], y, c, ring=u > 0.27)
        self.text(sc, X["code"], self.UNDER_A, "16 numbers", c, SIZE, strong=True)
        self.digit(sc, m["rebuilt"], X["back"], y, on(u, 0.51, op))
        # scored pixel by pixel: the rebuilt digit against the one given
        s = on(u, 0.62, op)
        hy = self.half()[1]
        self.arc(sc, (X["back"], y + hy + 0.008), (X["digit"], y + hy + 0.008), LOSS, s,
                 u=eased(u, 0.62, 0.72), bend=-0.14, width=2.0)
        self.text(sc, (X["digit"] + X["back"]) / 2, 0.675, "pixel by pixel: %d scores"
                  % m["pixels"], on(u, 0.7, op), SIZE, ink=LOSS_INK)
        self.text(sc, (X["digit"] + X["back"]) / 2, 0.72, "loss %s" % m["digit_loss_text"],
                  on(u, 0.74, op), HEAD, ink=LOSS_INK)

    def phase_3(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, self.UNDER_A + 0.005, "the wires come off", on(u, 0.04, op)
                  * (1 - smoothstep(0.26, 0.3, u)), SIZE, "lm")
        a = on(u, 0.66, op)
        self.text(sc, 0.27, 0.455, "generator = decoder", a, SIZE, "lm", strong=True)
        self.text(sc, 0.27, 0.495, "%s numbers kept" % m["counts_text"]["decoder"], a, SIZE,
                  "lm", ink=DEC_INK)

    def phase_4(self, sc, u, op):
        m = self.meta
        a = on(u, 0.1, op) * (1 - smoothstep(0.38, 0.44, u))
        self.text(sc, self.AX["enc"], self.UNDER_A, "left out: %s" % m["left_out_text"], a,
                  SIZE, strong=True)
        b = on(u, 0.76, op)
        self.text(sc, 0.06, 0.455, "judge = encoder's body + new head", b, SIZE, "lm",
                  strong=True)
        self.text(sc, 0.06, 0.495, "%s kept + %s new" % (m["kept_text"],
                                                       m["counts_text"]["head"]), b, SIZE,
                  "lm", ink=ENC_INK)

    def phase_5(self, sc, u, op):
        G, y = self.GX, self.ROW_G
        hx, hy = self.HEAD_AT
        cw = self.card_size()[0]
        legs = (((G["noise"] + self.frac(9.0)[0], y), (self.edge(G["gen"], WIDE, -1), y), 0.2),
                ((self.edge(G["gen"], WIDE, 1), y), (self.edge(G["made"], cw, -1), y), 0.3),
                ((self.edge(G["made"], cw, 1), y), (self.edge(G["judge"], NARROW, -1), y),
                 0.42),
                ((self.edge(G["real"], cw, -1), y), (self.edge(G["judge"], NARROW, 1), y),
                 0.62),
                ((hx, y + self.frac(0, CARD_H / 2 + 2)[1]), (hx, hy - self.frac(0, 13)[1]),
                 0.72))
        for a, b, at in legs:
            self.pulse(sc, a, b, (u - at) / 0.08, DATA, op)
        self.text(sc, 0.06, 0.455, "no digit goes in:", on(u, 0.2, op), SIZE, "lm",
                  strong=True)
        self.text(sc, 0.06, 0.495, "it starts from a made-up code", on(u, 0.2, op), SIZE, "lm")

    def phase_6(self, sc, u, op):
        m, X, y = self.meta, self.AX, self.ROW_A
        cw = self.card_size()[0]
        a = on(u, 0.04, op)
        self.digit(sc, m["digit"], X["digit"], y, a)
        self.block(sc, "enc", X["enc"], y, a, self.patches, ghost=True)
        self.block(sc, "dec", X["dec"], y, a, self.patches, ghost=True)
        self.code_strip(sc, m["code"], X["code"], y, a, ring=True)
        self.digit(sc, m["rebuilt"], X["back"], y, a)
        for x0, w0, x1, w1 in ((X["digit"], cw, X["enc"], WIDE),
                               (X["enc"], WIDE, X["code"], CODE_W + 6),
                               (X["code"], CODE_W + 6, X["dec"], WIDE),
                               (X["dec"], WIDE, X["back"], cw)):
            self.arrow(sc, (self.edge(x0, w0, 1), y), (self.edge(x1, w1, -1), y), AXIS, a,
                       width=1.8)
        for key, name, ink in (("enc", "encoder", ENC_INK), ("dec", "decoder", DEC_INK)):
            self.text(sc, X[key], y - self.NAME_DY, name, a * 0.8, SIZE, ink=ink)
        self.text(sc, X["code"], self.UNDER_A, "16 numbers", on(u, 0.1, op), SIZE,
                  strong=True)
        self.text(sc, X["back"], 0.385, "%d scores" % m["pixels"], on(u, 0.2, op),
                  SIZE, ink=LOSS_INK)
        G = self.GX
        r = on(u, 0.3, op)
        self.rect(sc, self.P(G["noise"], self.ROW_G), CODE_W + 14, CODE_H + 14, INK, r,
                  border=INK, border_w=2.4, radius=4, lift=0.6, fill_op=0.0)
        hx, hy = self.HEAD_AT
        self.text(sc, hx + 0.052, hy + 0.037, "1 score", on(u, 0.3, op), SIZE, "lm",
                  ink=LOSS_INK)


# ====================================================================== 266
class Blurry(AeGan):
    """Real, rebuilt, made before and after the game; grey shares; certainty; why grey."""

    TITLE = "Blurry or sharp"
    PHASES = (("real and rebuilt", 11.0), ("made up", 12.0), ("how grey", 12.0),
              ("how sure", 10.0), ("why grey", 13.0))
    LAST = "Same blocks, wired again"
    NEXT = "Rebuild or invent"
    STATION = (None, None, None, None, None)
    COLS = (0.105, 0.21, 0.315, 0.42)
    ROWS = (0.49, 0.6, 0.71)
    T = 1.25                                  # the digits' tiles, px
    HEAD_Y, GAME_Y = 0.41, 0.364             # the columns' names; "GAN game" over the arrow
    NUM_Y = 0.805
    SHOW_AT = ((0, 0.04), (0, 0.3), (1, 0.08), (1, 0.45))
    GREY_AT = (0.12, 0.3, 0.48, 0.66)
    PIX_X = (0.33, 0.385, 0.44)               # 5: one pixel painted 0.1, 0.5, 0.9

    def setup(self):
        super().setup()
        m = self.meta
        self.kinds = m["kinds"]
        self.CODE = (
            Code("ag-ae", run=((0.0, 0), (0.3, 1))),
            Code("ag-judge", pick=(0, 1), run=((0.0, 0), (0.3, 1))),
            Code("ag-judge", pick=(0, 1), run=((0.0, 1),)),
            Code("ag-judge", pick=(0, 1), run=((0.0, 1),)),
            Code("ag-ae", run=((0.0, 2),)))
        self.EVIDENCE = (
            measured("held-out digits 0 to 2, rebuilt"),
            measured("the same 3 made-up codes"),
            measured("1,000 of each, pixels 0.2 to 0.8"),
            measured("the course classifier, 1,000 of each"),
            exact("binary cross-entropy of one pixel"))

    def heading_for(self, k, u):
        if k == 1:
            return "the decoder on made-up codes" if u < 0.45 else \
                "the same codes, after the GAN game"
        return ("real digits, and their rebuilds", "", "grey pixels, in 1,000 of each",
                "how sure the classifier is", "pixel by pixel, grey is safest")[k]

    def subtitle_for(self, k):
        m = self.meta
        g, s = m["grey_text"], m["sure_text"]
        return [
            "Three held-out digits and the autoencoder's rebuilds of them: the rebuilds are "
            "softer at the edges.",
            "The decoder fed made-up codes, before the GAN game and after it: the same codes, "
            "sharper, some speckled.",
            "Grey pixels (between 0.2 and 0.8): real %s, rebuilt %s; made %s before the game "
            "and %s after." % (g["real"], g["rebuilt"], g["before"], g["after"]),
            "The course's classifier is as sure of rebuilds as of real digits (%s, %s); of "
            "made ones %s before, %s after." % (s["real"], s["rebuilt"], s["before"],
                                                s["after"]),
            "Ink in half the digits a code could be: scored pixel by pixel, 0.5 is best. "
            "The judge scores whole pictures.",
        ][k]

    def names(self):
        return ("real", "rebuilt", "before", "after")

    def steady(self, sc, k, u, op):
        m = self.meta
        # the number under each column: grey share, then certainty, then (5) grey again
        s_out, s_in = swap(self.prog(3, 0.06, 0.18))
        g_out, g_in = swap(self.prog(4, 0.0, 0.1))
        grey, sure = s_out + g_in, s_in * g_out
        for i, kind in enumerate(self.kinds):
            keep = op * self.kept(0, 4 if i < 2 else 3)
            pk, pu = self.SHOW_AT[i]
            a = keep * (self.at(pk, pu) if k <= pk else 1.0) * float(k >= pk)
            self.text(sc, self.COLS[i], self.HEAD_Y, self.names()[i], a, SIZE,
                      ink=DATA_INK if i < 2 else DEC_INK)
            for r, img in enumerate(m["shown"][kind]):
                self.digit(sc, img, self.COLS[i], self.ROWS[r], a, tile=self.T)
            g = keep * (self.at(2, self.GREY_AT[i]) if k <= 2 else 1.0) * float(k >= 2)
            self.text(sc, self.COLS[i], self.NUM_Y, m["grey_text"][kind], g * grey,
                      HEAD, strong=True)
            self.text(sc, self.COLS[i], self.NUM_Y, m["sure_text"][kind], g * sure, HEAD,
                      strong=True)
        # what the number row is
        g = op * (self.at(2, 0.06) if k <= 2 else 1.0) * float(k >= 2)
        self.text(sc, 0.055, self.NUM_Y + 0.045, "grey pixels", g * grey, SIZE, "lm",
                  ink=QUIET)
        self.text(sc, 0.055, self.NUM_Y + 0.045, "classifier's certainty", g * sure, SIZE,
                  "lm", ink=QUIET)
        # the made columns: one decoder, the same made-up codes, before and after the game
        b = op * self.kept(1, 3) * (self.at(1, 0.45) if k <= 1 else 1.0) * float(k >= 1)
        xa, xb = self.COLS[2], self.COLS[3]
        self.arrow(sc, (xa + 0.033, self.HEAD_Y), (xb - 0.03, self.HEAD_Y), AXIS, b,
                   width=1.6)
        self.text(sc, (xa + xb) / 2, self.GAME_Y, "GAN game", b, SIZE, ink=QUIET)

    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        m = self.meta
        # one pixel, ink in half the digits a code could be: paint 0.1, 0.5 or 0.9
        x0 = 0.268
        self.text(sc, x0, 0.45, "one pixel:", on(u, 0.04, op), SIZE, "lm", strong=True)
        self.text(sc, x0, 0.49, "ink in half the digits", on(u, 0.04, op), SIZE, "lm")
        y = 0.56
        self.text(sc, x0, y + 0.055, "paint", on(u, 0.12, op), SIZE, "lm", ink=QUIET)
        self.text(sc, x0, y + 0.1, "loss", on(u, 0.12, op), SIZE, "lm", ink=LOSS_INK)
        for i, (q, s) in enumerate(zip(m["guesses"], m["scores_text"])):
            a = on(u, 0.14 + 0.1 * i, op)
            fx = self.PIX_X[i]
            self.tiles(sc, [[q]], fx, y, 30.0, 30.0, a, lo=0.0, hi=1.0, lift=0.3)
            best = i == 1
            if best:
                self.rect(sc, self.P(fx, y), 38.0, 38.0, INK, a * on(u, 0.5, 1.0),
                          border=INK, border_w=2.4, radius=4, lift=0.6, fill_op=0.0)
            self.text(sc, fx, y + 0.055, str(q), a, SIZE)
            self.text(sc, fx, y + 0.1, s, a, HEAD, ink=LOSS_INK)
        self.text(sc, x0, 0.72, "lowest: 0.5, grey", on(u, 0.5, op), SIZE, "lm",
                  ink=LOSS_INK)


# ====================================================================== 267
class Rebuild(AeGan):
    """Four digits rebuilt and read; 928 of 1,000; the generator's only input; what 1,000
    made digits are read as; the 16, as they are."""

    TITLE = "Rebuild or invent"
    PHASES = (("rebuild", 12.0), ("how often", 10.0), ("no way in", 12.0),
              ("what it makes", 14.0), ("as they are", 10.0))
    LAST = "Blurry or sharp"
    NEXT = "One loss or two players"
    STATION = (None, None, None, None, None)
    COLS = (0.115, 0.205, 0.295, 0.385)
    ROWS = (0.45, 0.6)
    T = 1.6
    PIPE_Y = 0.49                             # 3-5: the generator's row
    PX: ClassVar[dict] = dict(noise=0.085, gen=0.16, made=0.255)
    CHART = ((0.11, 0.455), (0.62, 0.83))     # 4: counts by digit

    def setup(self):
        super().setup()
        self.CODE = (
            Code("ag-ae", run=((0.0, 0), (0.25, 1))),
            Code("ag-ae", run=((0.0, 1),)),
            Code("ag-judge", pick=(0, 1), run=((0.0, 0), (0.35, 1))),
            Code("ag-judge", pick=(0, 1), run=((0.0, 1),)),
            Code("ag-judge", pick=(0, 1), run=((0.0, 1),)))
        self.EVIDENCE = (
            measured("held-out digits 0 to 3, rebuilt"),
            measured("1,000 held-out digits"),
            exact("generator(z): z is its only input"),
            measured("1,000 made-up codes, read"),
            measured("the 16 fixed codes, after 6,000 steps"))

    def heading_for(self, k, u):
        return ("given a digit, it rebuilds it", "rebuilds read as the right digit",
                "the generator: no way in for a digit", "1,000 made digits, read by digit",
                "16 made digits, as they are")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The autoencoder is given a digit and rebuilds that digit; the classifier reads "
            "these four rebuilds right.",
            "Over 1,000 held-out digits, %d rebuilds are read as their digit (%s); the real "
            "digits themselves, %d." % (m["rebuild_right_n"], m["rebuild_right_text"],
                                        m["real_right_n"]),
            "The generator's only input is 16 made-up numbers: there is no way to give it a "
            "digit, or ask for one.",
            "What 1,000 made digits are read as: every digit appears, from %d (%ds) to %d "
            "(%ds)." % (m["after_min"], m["after_min_digit"], m["after_max"],
                        m["after_max_digit"]),
            "Sixteen made digits, as they are: many read well, some speckled or malformed, "
            "none chosen.",
        ][k]

    def steady(self, sc, k, u, op):
        m = self.meta
        # 1-2: the four digits and their rebuilds
        keep = op * (1.0 if k <= 1 else (1.0 - self.prog(2, 0.0, 0.06)) if k == 2 else 0.0)
        for i in range(4):
            a = keep * (self.at(0, 0.04 + 0.04 * i) if k == 0 else 1.0)
            self.digit(sc, m["pairs"]["real"][i], self.COLS[i], self.ROWS[0], a, tile=self.T)
            b = keep * (self.at(0, 0.3 + 0.06 * i) if k == 0 else 1.0)
            ya = self.ROWS[0] + self.half(self.T)[1] + 0.004
            yb = self.ROWS[1] - self.half(self.T)[1] - 0.004
            self.arrow(sc, (self.COLS[i], ya), (self.COLS[i], yb), AXIS, b, width=1.6)
            self.digit(sc, m["pairs"]["rebuilt"][i], self.COLS[i], self.ROWS[1], b,
                       tile=self.T)
            c = keep * (self.at(0, 0.6 + 0.04 * i) if k == 0 else 1.0)
            self.text(sc, self.COLS[i], 0.705, "reads %d" % m["pair_read"][i], c, SIZE,
                      ink=DATA_INK)
        self.text(sc, 0.44, self.ROWS[0], "given", keep * (self.at(0, 0.04) if k == 0 else
                                                           1.0), SIZE, "lm", ink=QUIET)
        self.text(sc, 0.44, self.ROWS[1], "rebuilt", keep * (self.at(0, 0.3) if k == 0 else
                                                             1.0), SIZE, "lm", ink=QUIET)
        # 3-5: the generator, fed a made-up code
        g = op * self.kept(2) * (self.at(2, 0.07) if k <= 2 else 1.0)
        P, y = self.PX, self.PIPE_Y
        self.code_strip(sc, m["made_code"], P["noise"], y, g)
        self.text(sc, P["noise"], y - 0.095, "made up", g, SIZE, ink=DATA_INK)
        self.arrow(sc, (P["noise"] + self.frac(9.0)[0], y),
                   (P["gen"] - self.frac(WIDE / 2 + 3)[0], y), AXIS, g, width=1.8)
        self.block(sc, "dec", P["gen"], y, g, m["patches_gen"])
        self.text(sc, P["gen"], y - 0.095, "generator", g, SIZE, ink=DEC_INK)
        cw = self.card_size(self.T)[0]
        self.arrow(sc, (P["gen"] + self.frac(WIDE / 2 + 3)[0], y),
                   (P["made"] - self.frac(cw / 2 + 3)[0], y), AXIS, g, width=1.8)
        self.digit(sc, m["made"][0], P["made"], y, g, tile=self.T)
        self.text(sc, P["made"], y - 0.095, "made", g, SIZE, ink=DATA_INK)

    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        m = self.meta
        self.big(sc, 0.06, 0.8, "%d of 1,000" % m["rebuild_right_n"], on(u, 0.08, op),
                 label="rebuilds read right", size=HEAD)
        self.big(sc, 0.29, 0.8, "%d of 1,000" % m["real_right_n"], on(u, 0.4, op),
                 label="real digits", size=HEAD, ink=DATA_INK)

    def phase_3(self, sc, u, op):
        m = self.meta
        # a digit, held out to it, with nowhere to go in
        fx, fy = self.PX["gen"], 0.765
        a = on(u, 0.3, op)
        self.digit(sc, m["pairs"]["real"][0], fx, fy, a, tile=self.T)
        self.text(sc, fx, fy + self.half(self.T)[1] + 0.03, "a %d, please?"
                  % m["pair_labels"][0], a, SIZE, ink=DATA_INK)
        y0 = fy - self.half(self.T)[1] - 0.006
        self.arrow(sc, (fx, y0), (fx, 0.61), AXIS, on(u, 0.42, op), u=eased(u, 0.42, 0.5),
                   width=1.6, head=False)
        b = on(u, 0.52, op)
        self.text(sc, fx, 0.59, "×", b, BIG, ink=INK, strong=True)
        x = 0.215
        self.text(sc, x, 0.665, "no input for a digit", b, SIZE, "lm", strong=True)
        self.text(sc, x, 0.71, "only 16 made-up numbers", on(u, 0.62, op), SIZE, "lm")
        self.text(sc, x, 0.755, "new code, new digit", on(u, 0.72, op), SIZE, "lm",
                  ink=QUIET)

    def chart(self):
        return LossChart(self.CHART, (-0.5, 9.5), (0.0, 160.0))

    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self.chart()
        a = on(u, 0.03, op)
        self.axes(sc, ch, a, xticks=[], yticks=[(0, "0"), (100, "100")], grid=False)
        counts = m["made_counts"]["after"]
        bw = self.span(ch.x1 - ch.x0)[0] / 10 * 0.62
        for d, n in enumerate(counts):
            g = on(u, 0.1 + 0.035 * d, op)
            x = ch.fx(d)
            h = self.span(0, ch.fy(0) - ch.fy(n))[1] * eased(u, 0.1 + 0.035 * d,
                                                              0.16 + 0.035 * d)
            if h > 0.5:
                self.rect(sc, self.P(x, ch.fy(0) - self.frac(0, h / 2)[1]), bw, h,
                          soft(DATA, 0.25), g, border=DATA, border_w=1.2, radius=2, lift=0.3)
            self.text(sc, x, ch.fy(0) + 0.025, str(d), g, SIZE, strong=True)
            self.text(sc, x, ch.fy(n) - 0.02, str(n), on(u, 0.5, op), SIZE, ink=DATA_INK)
        self.text(sc, 0.27, 0.885, "read as", a, SIZE, ink=QUIET)

    def phase_5(self, sc, u, op):
        m = self.meta
        T = 1.05
        w, h = self.card_size(T)
        for i, img in enumerate(m["made"]):
            r, c = divmod(i, 8)
            fx = 0.085 + c * self.frac(w + 4)[0]
            fy = 0.64 + r * self.frac(0, h + 22)[1]
            a = on(u, 0.05 + 0.02 * i, op)
            self.digit(sc, img, fx, fy, a, tile=T)
            self.text(sc, fx, fy + self.frac(0, h / 2 + 9)[1], str(m["made_read"][i]),
                      on(u, 0.45, op), SIZE, ink=DATA_INK)


# ====================================================================== 268
class Losses(AeGan):
    """One loss falling; two pulling; the cannot-tell lines; the pictures over the game."""

    TITLE = "One loss or two players"
    PHASES = (("one loss", 13.0), ("two players", 13.0), ("pulling", 10.0),
              ("its pictures", 13.0), ("no finish line", 9.0))
    LAST = "Rebuild or invent"
    NEXT = "Four families, four objectives"
    STATION = ("loss", "loss", None, None, None)
    AE_BOX = ((0.11, 0.44), (0.41, 0.6))
    GAN_BOX = ((0.11, 0.44), (0.7, 0.86))
    SNAP_X = (0.1, 0.19, 0.28, 0.37)
    SNAP_Y = (0.465, 0.575)
    T = 1.4

    def setup(self):
        super().setup()
        self.CODE = (
            Code("ag-ae", run=((0.0, 0), (0.2, 1), (0.4, 2))),
            Code(("ag-judge", "ag-generator"), run=((0.0, 5), (0.3, 6), (0.6, 7))),
            Code(("ag-judge", "ag-generator"), run=((0.0, 6), (0.5, 7))),
            Code("ag-judge", pick=(0, 1), run=((0.0, 1),)),
            Code(("ag-judge", "ag-generator"), run=((0.0, 6), (0.5, 7))))
        self.EVIDENCE = (
            measured("3,000 steps of the autoencoder"),
            measured("6,000 steps, means of 100"),
            exact("2 ln 2 = 1.39, ln 2 = 0.69"),
            measured("the same codes, steps 0 to 6,000"),
            measured("one run of each"))

    def heading_for(self, k, u):
        return ("the autoencoder: one loss, falling", "the GAN: two losses, one each",
                "they pull against each other", "what the generator makes, over the game",
                "the GAN has no finish line")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The autoencoder's one loss falls smoothly, from %s at the first step to %s over "
            "the last 100." % (m["ae_first_text"], m["ae_last_text"]),
            "The GAN has two: the judge's loss goes from %s to %s, the generator's from %s "
            "to %s." % (*m["lj_text"], *m["lg_text"]),
            "When the judge gets better, the generator's loss rises. Neither nears 0; the "
            "dashed lines are a judge that cannot tell.",
            "The same made-up codes at steps 0, 500, 2,000 and 6,000: the pictures change, "
            "the losses barely say how.",
            "The autoencoder is done when its loss is low; the GAN's two losses only balance, "
            "so its pictures are judged by eye.",
        ][k]

    def ae_chart(self):
        return LossChart(self.AE_BOX, (0.0, 3000.0), (0.0, 0.75))

    def gan_chart(self):
        return LossChart(self.GAN_BOX, (0.0, 6000.0), (0.5, 1.6))

    def steady(self, sc, k, u, op):
        m = self.meta
        # the autoencoder's loss: phases 1-3
        a = op * self.kept(0, 1) * (self.at(0, 0.03) if k == 0 else 1.0)
        if a > 0.01:
            ch = self.ae_chart()
            self.axes(sc, ch, a, xticks=[(0, "0"), (3000, "3,000 steps")],
                      yticks=[(0, "0"), (0.69, "0.69")], grid=False)
            drawn = self.prog(0, 0.1, 0.5)
            self.plot(sc, ch, m["ae_steps_x"], m["ae_loss"], LOSS, a, drawn, width=1.4,
                      dot=drawn < 1)
            e = a * (self.at(0, 0.55) if k == 0 else 1.0)
            self.text(sc, ch.x1 + 0.01, ch.fy(m["ae_last"]), m["ae_last_text"], e, SIZE, "lm",
                      ink=LOSS_INK)
            self.key(sc, ch.x0 + 0.02, 0.405, LOSS, a)
            self.text(sc, ch.x0 + 0.035, 0.405, "autoencoder", a, SIZE, "lm", ink=LOSS_INK)
        # the GAN's two: phases 2-5
        g = op * self.kept(1) * (self.at(1, 0.03) if k <= 1 else 1.0)
        if g > 0.01:
            ch = self.gan_chart()
            self.axes(sc, ch, g, xticks=[(0, "0"), (6000, "6,000 steps")],
                      yticks=[(0.69, "0.69"), (1.39, "1.39")], grid=False)
            x = m["chunk_x"]
            drawn = self.prog(1, 0.1, 0.55)
            self.plot(sc, ch, x, m["lj_chunks"], ENC, g, drawn, width=1.6, dot=drawn < 1)
            self.plot(sc, ch, x, m["lg_chunks"], DEC, g, drawn, width=1.6, dot=drawn < 1)
            e = g * (self.at(1, 0.6) if k <= 1 else 1.0)
            for y, ink, text in ((m["lj_last"], ENC, "judge %s" % m["lj_text"][1]),
                                 (m["lg_last"], DEC, "generator %s" % m["lg_text"][1])):
                self.key(sc, ch.x1 + 0.016, ch.fy(y), ink, e)
                self.text(sc, ch.x1 + 0.03, ch.fy(y), text, e, SIZE, "lm", ink=dark(ink, 0.3))
            self.text(sc, ch.x0 + 0.02, 0.665, "GAN, means of 100 steps", g, SIZE, "lm",
                      ink=LOSS_INK)
            c = g * (self.at(2, 0.1) if k <= 2 else 1.0) * float(k >= 2)
            self.dashed(sc, ch, m["coin_judge"], c)
            self.dashed(sc, ch, m["coin_gen"], c)
            for s in m["snaps"]:
                ch.mark(sc, self.P, s, AXIS, g * self.kept(3) * (self.at(3, 0.06) if k <= 3
                                                                   else 1.0) * float(k >= 3))
        # the pictures over the game: phases 4-5
        p = op * self.kept(3) * float(k >= 3)
        for j, s in enumerate(m["snaps"]):
            a = p * (self.at(3, 0.06 + 0.1 * j) if k == 3 else 1.0)
            self.text(sc, self.SNAP_X[j], 0.39, "step %s" % format(s, ","), a, SIZE,
                      strong=True)
            for r in range(2):
                self.digit(sc, m["snap_pics"][s][r], self.SNAP_X[j], self.SNAP_Y[r], a,
                           tile=self.T)

    def key(self, sc, fx, fy, ink, op):
        """A short thick line of a curve's colour, beside its name."""
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), 16.0, 5.0, ink, op, radius=2, lift=0.3)

    def dashed(self, sc, ch, y, op):
        if op <= 0.01:
            return
        ch.hline(sc, self.P, y, op, dashes=18, width=1.1)

    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        ch = self.gan_chart()
        self.text(sc, ch.x0 + 0.02, ch.fy(1.39) - 0.025, "cannot tell", on(u, 0.15, op),
                  SIZE, "lm", ink=QUIET)
        m = self.meta
        self.text(sc, 0.06, 0.43, "judge %s → %s" % m["lj_text"], on(u, 0.3, op), HEAD, "lm",
                  ink=ENC_INK)
        self.text(sc, 0.06, 0.475, "better at telling", on(u, 0.36, op), SIZE, "lm")
        self.text(sc, 0.06, 0.545, "generator %s → %s" % m["lg_text"], on(u, 0.5, op), HEAD,
                  "lm", ink=DEC_INK)
        self.text(sc, 0.06, 0.59, "fools it less often", on(u, 0.56, op), SIZE, "lm")

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        pass


ANIMATIONS = {"ae-to-gan-rewire": Rewire, "ae-gan-blurry-or-sharp": Blurry,
              "ae-gan-rebuild-or-invent": Rebuild, "ae-gan-one-loss-or-two": Losses}
