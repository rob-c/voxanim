"""A dense U-Net denoising MNIST digits, laid out as the U it is named after.

           noisy x                original                     output
              \\                                                /
              e1 (256) ======== skip: copy ========> [d2 ++ e1] -> d1 (256)
                 \\                                           /
                 e2 (64) ===== skip: copy =====> [b ++ e2] -> d2 (64)
                      \\                            /
                         bottleneck b (16)  ------

The encoder squeezes the noisy digit down to 16 numbers; the decoder grows
it back.  The skip connections copy each encoder layer across the U and
concatenate it with the decoder's input at the same level, so fine detail
does not have to fit through the bottleneck.  At the end, the same digit
goes through an identical network trained *without* the skips, for
comparison.  Real numbers: :mod:`voxanim.deep_models`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import deep_models as D
from .easing import ease_in_out, remap, smoothstep
from .mlp import (
    ACT_HI,
    ACT_LO,
    INK_HI,
    INK_LO,
    NetAnimation,
    amax,
    gray_colors,
    lit_colors,
    particle,
)

AT = dict(x=(0.12, 0.31), e1=(0.235, 0.50), e2=(0.345, 0.665), b=(0.50, 0.78),
          d2=(0.655, 0.665), d1=(0.765, 0.50), out=(0.88, 0.31), clean=(0.50, 0.31),
          plain=(0.88, 0.70))
SHAPES = dict(x=(28, 28), e1=(16, 16), e2=(8, 8), b=(4, 4), d2=(8, 8), d1=(16, 16),
              out=(28, 28), clean=(28, 28), plain=(28, 28))
SCALE = dict(x=0.32, e1=0.55, e2=0.95, b=1.5, d2=0.95, d1=0.55, out=0.32, clean=0.32,
             plain=0.32)


class UNetAnimation(NetAnimation):
    PHASES = ("add noise", "encode", "bottleneck", "decode + skips", "without skips")

    def __init__(self, digits=(3, 18), seed=4):
        super().__init__("Dense U-Net: denoising", "", 9.0)
        from .mnist import load
        self.P, self.loss = D.unet_train(True)
        self.Pp, self.loss_plain = D.unet_train(False)
        _a, _b, xte, _yte = load()
        rng = np.random.default_rng(seed)
        self.clean = [xte[i] for i in digits]
        self.noisy = [D.noisy(x[None], rng)[0] for x in self.clean]
        self.f = [D.unet_forward(self.P, n[None], True) for n in self.noisy]
        self.fp = [D.unet_forward(self.Pp, n[None], False) for n in self.noisy]
        self.S = {k: self.sheet(SHAPES[k], AT[k], SCALE[k], k) for k in AT}
        # ghost copies that travel along the skip bridges and sit by the decoder
        self.copy1 = self.sheet(SHAPES["e1"], AT["e1"], SCALE["e1"], "copy1")
        self.copy2 = self.sheet(SHAPES["e2"], AT["e2"], SCALE["e2"], "copy2")
        self.hold = 2.5

        Tm, t = {}, 0.6
        self.digit_t = []
        for d in range(len(digits)):
            slow = d == 0
            seq = [("noise", 1.6 if slow else 0.8), ("e1", 1.4 if slow else 0.7),
                   ("e2", 1.2 if slow else 0.6), ("b", 1.2 if slow else 0.6),
                   ("skip", 1.4 if slow else 0.8), ("d2", 1.3 if slow else 0.7),
                   ("d1", 1.3 if slow else 0.7), ("out", 1.4 if slow else 0.8),
                   ("look", 1.4 if slow else 1.0)]
            seg = {}
            for name, dd in seq:
                seg[name] = (t, t + dd)
                t += dd
            self.digit_t.append(seg)
            t += 0.4
        Tm["plain"] = (t, t + 4.0)
        t += 4.4
        self.Tm = Tm
        self.finish(t)
        self.parts = [self._particles(f) for f in self.f]

    # ------------------------------------------------------------------ data
    def _vals(self, f):
        return dict(x=f["x"][0], e1=f["e1"][0], e2=f["e2"][0], b=f["b"][0], d2=f["d2"][0],
                    d1=f["d1"][0], out=f["out"][0])

    def _particles(self, f):
        """Strongest weighted inputs per layer: (source sheet, index, dst sheet, index)."""
        v = self._vals(f)
        P = self.P
        layers = [("x", None, "e1", "We1"), ("e1", None, "e2", "We2"), ("e2", None, "b", "Wb"),
                  ("b", "copy2", "d2", "Wd2"), ("d2", "copy1", "d1", "Wd1"),
                  ("d1", None, "out", "Wo")]
        out = {}
        for src, skip, dst, w in layers:
            a = v[src] if skip is None else np.concatenate(
                [v[src], v["e2" if skip == "copy2" else "e1"]])
            c = a[:, None] * P[w]
            K = min(70, int((np.abs(c) > 0).sum()))
            idx = np.argpartition(-np.abs(c).ravel(), K - 1)[:K]
            cmax = amax(c)
            n_src = len(v[src])
            lst = []
            for m in idx:
                i, j = divmod(int(m), P[w].shape[1])
                sheet = src if i < n_src else skip
                ii = i if i < n_src else i - n_src
                lst.append((sheet, ii, j, c.ravel()[m] / cmax,
                            0.5 * ((m * 2654435761) % 997) / 997))
            out[dst] = lst
        return out

    def step_of(self, t):
        if t >= self.Tm["plain"][0]:
            return 4
        for d in range(len(self.digit_t) - 1, -1, -1):
            seg = self.digit_t[d]
            if t >= seg["noise"][0]:
                for k, names in ((3, ("skip", "d2", "d1", "out", "look")), (2, ("b",)),
                                 (1, ("e1", "e2"))):
                    if any(seg[n][0] <= t < seg[n][1] for n in names):
                        return k
                return 0
        return 0

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        d = 0
        for i, seg in enumerate(self.digit_t):
            if t >= seg["noise"][0]:
                d = i
        if t >= self.Tm["plain"][0]:
            d = len(self.digit_t) - 1
        seg = self.digit_t[d]
        p = {k: remap(t, *v) for k, v in seg.items()}
        v = self._vals(self.f[d])
        S = self.S
        ink = lambda a: lit_colors(a, 1.0, INK_LO, INK_HI)          # noqa: E731
        act = lambda a: lit_colors(a, amax(a), ACT_LO, ACT_HI)       # noqa: E731

        # the clean original, and noise being added to it
        noise = smoothstep(0.0, 1.0, p["noise"])
        x = self.clean[d] + (self.noisy[d] - self.clean[d]) * noise
        clean = self.clean[d]
        if reset > 0:              # the loop returns to the first digit, still clean
            r = ease_in_out(reset)
            x = x + (self.clean[0] - x) * r
            clean = clean + (self.clean[0] - clean) * r
        S["clean"].fill(ink(clean))
        S["x"].fill(gray_colors(x))
        for k in ("x", "clean"):
            S[k].cage = 1.0
        for name in ("e1", "e2", "b", "d2", "d1", "out"):
            sh = S[name]
            u = p[name]
            sh.fill(gray_colors(v[name]) if name == "out" else act(v[name]))
            rows = np.repeat(np.arange(sh.h), sh.w) / sh.h
            sh.show(u * 1.15 - 0.1 > rows)
            if name != "out":
                sh.light(v[name], smoothstep(0.8, 1.0, u))
            sh.cage = 1.0
            sh.fade(back)
        # skip bridges: the encoder layers' copies slide across
        sk = ease_in_out(p["skip"])
        for cp, src, dst in ((self.copy2, "e2", "d2"), (self.copy1, "e1", "d1")):
            cp.fill(act(v[src]))
            a = S[src].block.origin
            b = S[dst].block.origin - np.array([cp.block.extent[0] + 1.2, 0, 0])
            cp.block.origin = a + (b - a) * sk
            cp.fade((0.0 if sk <= 0 else 0.85) * back)
            cp.cage = (1.0 if sk > 0 else 0.0) * back
            if 0 < sk < 1:
                sc.probe(("skip", src), cp.block.origin, tol=4.0)
        for src, dst in (("e1", "d1"), ("e2", "d2")):
            ca, cb = S[src].center, S[dst].center
            sc.line(ca, cb, width=1.2, color=(0.55, 0.57, 0.59), opacity=0.5, bias=0.0)  # always
        # flows
        for dst in ("e1", "e2", "b", "d2", "d1", "out"):
            u = p[dst]
            if not 0 < u < 1:
                continue
            for sheet, i, j, val, start in self.parts[d][dst]:
                src = S[sheet] if sheet in S else getattr(self, sheet)      # copy1 / copy2
                particle(sc, src.cell(i), S[dst].cell(j), remap(u, start, start + 0.5),
                         C.POSITIVE if val > 0 else C.NEGATIVE, 0.5,
                         opacity=0.35 + 0.65 * abs(val), lift=1.5)
            sc.probe(("flow", dst), (u * 30, 0, 0), tol=8.0)
        # the plain network, for comparison
        pl = remap(t, *self.Tm["plain"])
        S["plain"].fill(gray_colors(self.fp[d]["out"][0]))
        S["plain"].fade(smoothstep(0.0, 0.3, pl) * back)
        S["plain"].cage = smoothstep(0.0, 0.3, pl) * back
        S["plain"].visible = pl > 0

        mse = float(np.mean((v["out"] - self.clean[d]) ** 2))
        mse_p = float(np.mean((self.fp[d]["out"][0] - self.clean[d]) ** 2))
        self.subtitle = [
            "a clean digit plus Gaussian noise (std %.2f) goes in; the clean digit is the "
            "target" % D.NOISE,
            "encoder: squeeze the noisy digit down, layer by layer",
            "the bottleneck: only 16 numbers describe the digit",
            "skip connections copy each encoder layer across the U and concatenate it with "
            "the decoder's input",
            "the same network without skip connections: test error %.4f vs %.4f with them"
            % (self.loss_plain, self.loss),
        ][ph]
        self.step_indicator(sc, ph, self.PHASES)
        # captions
        names = dict(x="noisy input", e1="256", e2="64", b="bottleneck 16", d2="64", d1="256",
                     out="denoised (error %.4f)" % mse if p["out"] >= 1 else "output",
                     clean="original")
        for k, text in names.items():
            if S[k].block.present.any() and S[k].visible:
                # the top row's captions go above: below them is the U's upper arm
                self.caption(sc, S[k], text, back, above=k in ("x", "clean", "out"))
        if pl > 0:
            self.caption(sc, S["plain"], "no skips  (error %.4f)" % mse_p, back)
        arrived = smoothstep(0.9, 1.0, sk) * (1 - smoothstep(0.0, 0.3, pl)) * back
        if arrived > 0:                           # labelled once in place, not mid-flight
            self.caption(sc, self.copy2, "copy of e2", arrived, above=True)
            self.caption(sc, self.copy1, "copy of e1", arrived, above=True)
