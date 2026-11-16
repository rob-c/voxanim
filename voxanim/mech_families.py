"""Animations 101-105 of the "next" catalogue: four generative families compared
(see :mod:`voxanim.mech`).

An autoencoder (AE), a variational autoencoder (VAE), a generative adversarial
network (GAN) and deep embedded clustering (DEC; IDEC is the form that keeps
its decoder) on the same MNIST digits, with the same decoder and code size:
what each is trained to do, where digits sit in its code space, what it draws
when asked for new digits, how well it gives a digit back, and what lies
between two digits. Every model is one of the shared set that
:func:`voxanim.study_families.models` trains once; every number on screen is
the study's, or is computed here from those models on the same held-out
digits (a single digit's code, error or reading).

One family keeps one colour in all five: AE blue, VAE green, GAN red, DEC
amber, IDEC ochre; the separately trained reader is violet, a control teal.
"""

from __future__ import annotations

import numpy as np

from . import autoencoder as AE
from . import layout
from . import study_families as F
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    INKS,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_generative import Generative, Plane, seg

WHITE = (1.0, 1.0, 1.0)
FAM = {"ae": BLUE, "vae": GREEN, "gan": RED, "dec": AMBER, "idec": OCHRE, "pixels": TEAL}
NAME = dict(F.NAMES, pixels="cross-fade")
READER = VIOLET                        # the separately trained classifier
JUDGE = MAGENTA                        # the GAN's own discriminator
TEN = INKS + [AXIS, DIGIT_INK]         # a colour per digit, 0 to 9
SEED = F.SEEDS[0]                      # the seed whose models are drawn (numbers: all three)


def pct(v):
    return "%.0f%%" % (100 * float(v))


class Families(Generative):
    """What the five animations share: a code as a strip of cells, a network as
    a box, an encoder-code-decoder chain, bars with a dot per seed."""

    def strip(self, sc, values, fx, fy, w, h, op, ink, vmax=None):
        """A code: one cell per number, ``ink`` positive and grey negative."""
        v = np.asarray(values, float)[None]
        return self.grid(sc, v, fx, fy, w, h, op, vmax=vmax, hot=ink, cold=AXIS, lines=0.3)

    def net(self, sc, fx, fy, ink, text, op, w=112.0, h=42.0, live=True):
        """A network as a labelled box; ``live`` False: present but not trained."""
        if live:
            self.box(sc, fx, fy, w, h, ink, op, text, ink=WHITE, size=0.034)
        else:
            self.box(sc, fx, fy, w, h, PAPER, op, text, ink=dark(ink, 0.3), border=ink,
                     size=0.034)

    def tag(self, sc, fx, fy, family, op, w=58.0):
        self.box(sc, fx, fy, w, 30, FAM[family], op, NAME[family], ink=WHITE, size=0.034)

    def link(self, sc, a, b, ink, op, u=1.0, pulse=None, bend=0.0, width=2.2):
        """An arrow from ``a`` to ``b`` and, with ``pulse`` (0..1), a value on its way."""
        self.arc(sc, a, b, ink, op, u, bend=bend, width=width)
        if pulse is not None:
            self.pulse(sc, a, b, pulse, ink, op, bend=bend)

    def chain(self, sc, y, op, ink, image, code, out, t=1.0, xs=(0.16, 0.30, 0.475, 0.65, 0.79),
              px=64.0, enc="encoder", dec="decoder", vmax=None, dec_ink=None, dec_live=True,
              strip_w=168.0, bw=112.0):
        """image -> encoder -> code -> decoder -> output along layout row ``y``;
        ``t`` (0..1) sends one value through it and reveals the output."""
        x_in, x_enc, x_code, x_dec, x_out = xs
        half = self.frac(px / 2 + 7)[0]
        box = self.frac(bw / 2 + 3)[0]
        dec_ink = ink if dec_ink is None else dec_ink
        code_half = self.frac(strip_w / 2 + 6)[0]
        self.digit(sc, image, x_in, y, px, op, frame=AXIS)
        self.net(sc, x_enc, y, ink, enc, op, w=bw)
        self.strip(sc, code, x_code, y, strip_w, 20, op * smoothstep(0.2, 0.35, t), ink, vmax)
        self.net(sc, x_dec, y, dec_ink, dec, op, live=dec_live, w=bw)
        self.digit(sc, out, x_out, y, px, op * smoothstep(0.75, 0.9, t), frame=dec_ink)
        hops = (((x_in + half, y), (x_enc - box, y)), ((x_enc + box, y), (x_code - code_half, y)),
                ((x_code + code_half, y), (x_dec - box, y)), ((x_dec + box, y), (x_out - half, y)))
        for i, (a, b) in enumerate(hops):
            self.link(sc, a, b, ink, op, pulse=seg(t, 0.2 * i, 0.2 * i + 0.3))

    def seed_bars(self, sc, x, ys, rows, px, vmax, op, u=1.0, fmt="%.2f", thick=16.0,
                  note=None):
        """Horizontal bars from layout ``x``, one per ``(label, ink, values)`` row at
        the layout heights ``ys``: the bar is the mean of ``values`` (one per seed),
        each seed a dot on it. ``px`` px is the value ``vmax``."""
        g = smoothstep(0.0, 1.0, u)
        dx = self.frac(px)[0]
        for i, ((label, ink, values), y) in enumerate(zip(rows, ys)):
            v = np.atleast_1d(np.asarray(values, float))
            self.hbar(sc, x, y, px * max(v.mean(), 0.0) / vmax, ink, op, label=label, u=u,
                      thick=thick, track=px)
            if len(v) > 1:
                for one in v:
                    self.disk(sc, self.P(x + dx * g * max(one, 0.0) / vmax, y), 2.8,
                              dark(ink, 0.5), op, outline=0.0, lift=1.5)
            end = max(v.max(), 0.0) / vmax
            text = fmt % v.mean() if note is None else note[i] or fmt % v.mean()
            self.text(sc, x + dx * g * end + 0.012, y, text, op, anchor="lm", strong=True)

    def bell(self, sc, fx, fy, w, h, op, ink=AXIS, width=2.0):
        """A standard bell curve ``w`` x ``h`` px, its base at layout ``(fx, fy)``."""
        dw, dh = self.frac(w, h)
        s = np.linspace(-3.0, 3.0, 31)
        pts = [self.P(fx + dw * v / 6.0, fy - dh * np.exp(-0.5 * v * v)) for v in s]
        self.polyline(sc, pts, width, ink, op)

    def ring(self, sc, plane, radius, ink, op, width=1.6):
        """A circle of ``radius`` about the origin of a :class:`Plane`."""
        a = np.linspace(0.0, 2 * np.pi, 49)
        plane.path(sc, np.column_stack((radius * np.cos(a), radius * np.sin(a))), ink, op, width)

    def table_cell(self, sc, fx, fy, text, op, ink=None, own=False, w=128.0, strong=False):
        self.box(sc, fx, fy, w, 34, soft(ink, 0.45) if own else PAPER, op, text,
                 ink=layout.STRONG_INK if strong else layout.CAPTION_INK,
                 border=ink if own else DIM, size=0.034)


# ====================================================================== 101 objectives
class Objectives(Families):
    """The same parts, four training signals: the AE's and the VAE's pass and what
    comes back along it, the GAN's judge, DEC's pull on a cloud of codes, the runs
    side by side and what each family ends up good at."""

    TITLE = "Four families, four objectives"
    PHASES = (("parts", 8.0), ("rebuild", 12.0), ("judge", 11.0), ("centres", 11.0),
              ("runs", 8.0), ("outcome", 9.0))

    def setup(self):
        s, d = self.study, F.data()
        c = self.code = F.WIDE
        self.x0 = d["xt"][F.TRACK]
        ae, vae = F.params("ae", c, SEED), F.params("vae", c, SEED)
        self.ae_code = F.encode(ae, self.x0[None])[0]
        self.ae_out = F.decode(ae, self.ae_code)[0]
        self.vae_mu = F.encode(vae, self.x0[None])[0]
        h1 = AE.dense_layers(vae, self.x0[None], range(0, 2))[0][-1]
        self.vae_sd = np.exp(0.5 * (h1 @ vae["Wlv"] + vae["blv"]))[0]
        eps = np.random.default_rng(3).normal(size=(24, c))
        self.vae_z = self.vae_mu + self.vae_sd * eps
        self.vae_outs = F.decode(vae, self.vae_z)
        self.vae_out = F.decode(vae, self.vae_mu)[0]
        self.err = {"ae": float(np.mean((self.ae_out - self.x0) ** 2)),
                    "vae": float(np.mean((self.vae_out - self.x0) ** 2))}
        self.kl_digit = float(0.5 * np.sum(self.vae_mu ** 2 + self.vae_sd ** 2 - 1
                                           - 2 * np.log(self.vae_sd)))
        self.meta = s.metadata
        # the GAN's run (seed 1): one fixed noise point, the judge on it and on a real digit
        self.gan_steps = F.snapshot_steps(F.GAN_STEPS)
        self.gan_noise = F.trace("gan", c, SEED, "fixed")[0]
        self.gan_shown = F.trace("gan", c, SEED, "shown")[:, 0]
        self.gan_pf = F.trace("gan", c, SEED, "p_fake")[:, 0]
        self.gan_pr = F.trace("gan", c, SEED, "p_real")[:, F.TRACK]
        self.gan_right = F.trace("gan", c, SEED, "right")
        # DEC's cloud, from the 2-number models so that it can be drawn
        self.cloud = F.trace("dec", 2, SEED, "cloud")
        self.centres = F.trace("dec", 2, SEED, "centres")
        self.cloud_labels = d["yt"][:F.CLOUD]
        self.dec_acc = F.trace("dec", 2, SEED, "acc")
        self.dec_mse = F.trace("dec", 2, SEED, "mse")
        self.dec_steps = F.snapshot_steps(F.CLUSTER_STEPS)
        lo, hi = np.percentile(self.cloud.reshape(-1, 2), [1, 99], axis=0)
        mid, half = (lo + hi) / 2, float((hi - lo).max()) * 0.56
        self.dec_lim = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
        # the runs and the table
        self.shots = np.asarray(s.panels[3].data["values"]).reshape(len(F.FAMILIES), -1, 784)
        self.frac_run = np.asarray(s.panels[0].data["x"], float)
        self.mse_curves = np.asarray(s.panels[0].data["ys"], float)
        self.table = np.asarray(s.panels[2].data["values"], float)
        self.table_seeds = np.asarray(s.panels[2].data["seeds"], float)
        self.defined = np.asarray(s.panels[2].data["defined"])

    def subtitle_for(self, k):
        m, t = self.meta, self.table
        fam = {f: i for i, f in enumerate(F.FAMILIES)}
        return [
            "four ways to train the same parts on the same %s MNIST digits: one decoder "
            "shape, a code of %d numbers, three seeds; only what each is asked to do "
            "differs" % (f"{F.TRAIN_N:,}", self.code),
            "an autoencoder is told how far its output is from its input (held-out error "
            "%.3f); a VAE gets the same signal through a noisy code, plus a pull of every "
            "code towards a standard bell (error %.3f)" % (
                m["final_mse"]["ae"]["mean"], m["final_mse"]["vae"]["mean"]),
            "a GAN has no encoder and compares no pixels: its generator learns only from a "
            "judge trained to tell its digits from real ones; at the end the judge is "
            "right on %s of images" % pct(m["gan_right"]["mean"]),
            "DEC starts from the trained autoencoder and moves each code to its nearest "
            "centre: with 16 numbers cluster accuracy goes from %.2f to %.2f, and the decoder, "
            "left behind, rebuilds worse (%.3f to %.3f)" % (
                m["cluster_accuracy_start"]["mean"], m["cluster_accuracy_end"]["dec"]["mean"],
                m["start_mse_cluster"]["mean"], m["final_mse"]["dec"]["mean"]),
            "one held-out digit through each run: AE, VAE and IDEC end with held-out error "
            "%.3f, %.3f and %.3f; DEC's climbs to %.3f; the GAN has nothing to "
            "rebuild" % (m["final_mse"]["ae"]["mean"], m["final_mse"]["vae"]["mean"],
                         m["final_mse"]["idec"]["mean"], m["final_mse"]["dec"]["mean"]),
            "three seeds each: the AE rebuilds best (%.3f), only the VAE's codes sit in the "
            "bell (%s), a reader trusts the GAN's samples most (%.2f), and DEC lifts the "
            "AE's cluster accuracy from %.2f to %.2f" % (
                t[fam["ae"], 0], pct(t[fam["vae"], 1]), t[fam["gan"], 3], t[fam["ae"], 2],
                t[fam["dec"], 2]),
        ][k]

    # ------------------------------------------------------------------ the parts
    def phase_1(self, sc, u, op):
        y = 0.385
        self.chain(sc, y, op, AXIS, self.x0, self.ae_code, self.ae_out, seg(u, 0.05, 0.5),
                   xs=(0.12, 0.29, 0.50, 0.71, 0.88), px=72.0, strip_w=200.0)
        self.text(sc, 0.29, y + 0.066, "784 > 256 > 64", op)
        self.text(sc, 0.50, y - 0.052, "a code: %d numbers" % self.code, op, strong=True)
        self.text(sc, 0.71, y + 0.066, "64 > 256 > 784", op)
        self.text(sc, 0.50, 0.53, "held equal: the digits, these layer sizes, the code size, "
                  "three seeds", op * smoothstep(0.3, 0.45, u), size=0.034)
        rows = (("ae", "rebuild the input"),
                ("vae", "rebuild it through a noisy code kept near a standard bell"),
                ("gan", "fool a judge: no encoder, nothing is rebuilt"),
                ("dec", "pull codes towards cluster centres (IDEC: and keep rebuilding)"))
        for i, (family, text) in enumerate(rows):
            a = op * smoothstep(0.42 + 0.1 * i, 0.52 + 0.1 * i, u)
            fy = 0.625 + 0.075 * i
            self.tag(sc, 0.135, fy, family, a)
            self.text(sc, 0.185, fy, text, a, anchor="lm", size=0.036, ink=dark(FAM[family], 0.35))

    # ------------------------------------------------------------------ rebuild
    def _signal(self, sc, y, ink, op, u, text):
        """What comes back: from the output, under the row, to the input side."""
        a, b = (0.79, y + 0.075), (0.30, y + 0.075)
        self.link(sc, a, b, dark(ink, 0.25), op * smoothstep(0.0, 0.1, u), seg(u, 0.0, 0.5),
                  pulse=seg(u, 0.3, 1.0), bend=-0.045)
        self.text(sc, 0.545, y + 0.135, text, op * smoothstep(0.3, 0.5, u), ink=dark(ink, 0.3),
                  size=0.033)

    def phase_2(self, sc, u, op):
        vmax = float(max(np.abs(self.ae_code).max(), np.abs(self.vae_z).max()))
        # the autoencoder
        y = 0.375
        self.tag(sc, 0.075, y, "ae", op, w=50.0)
        self.chain(sc, y, op, BLUE, self.x0, self.ae_code, self.ae_out, seg(u, 0.02, 0.3),
                   vmax=vmax)
        self.lines(sc, 0.85, y - 0.02, ["error", "%.3f" % self.err["ae"]],
                   op * smoothstep(0.28, 0.34, u), step=0.042, size=0.034)
        self._signal(sc, y, BLUE, op, seg(u, 0.3, 0.5),
                     "the signal: how far each output pixel is from the input's")
        # the VAE: the same, through a code that is redrawn every time
        y = 0.685
        k = int(u * 60) % len(self.vae_z)
        a = op * smoothstep(0.46, 0.54, u)
        self.tag(sc, 0.075, y, "vae", a, w=50.0)
        self.chain(sc, y, a, GREEN, self.x0, self.vae_z[k], self.vae_outs[k],
                   seg(u, 0.5, 0.74), vmax=vmax)
        self.text(sc, 0.475, y + 0.05, "centre + noise, redrawn each time", a)
        self.bell(sc, 0.475, y - 0.045, 96, 26, a * smoothstep(0.7, 0.78, u), GREEN, 2.4)
        self.lines(sc, 0.85, y - 0.04, ["error", "%.3f" % self.err["vae"],
                                         "pull: %.0f" % self.kl_digit],
                   op * smoothstep(0.74, 0.8, u), step=0.042, size=0.034)
        self._signal(sc, y + 0.025, GREEN, a, seg(u, 0.76, 0.96),
                     "the same signal, plus a pull of every code towards the bell")

    # ------------------------------------------------------------------ fool a judge
    def phase_3(self, sc, u, op):
        at = seg(u, 0.3, 0.92) * (len(self.gan_steps) - 1)
        k = int(round(at))
        y = 0.47
        self.tag(sc, 0.085, 0.33, "gan", op, w=56.0)
        self.strip(sc, self.gan_noise, 0.17, y, 168, 22, op, RED)
        self.text(sc, 0.17, y - 0.05, "%d numbers of noise" % self.code, op)
        self.net(sc, 0.38, y, RED, "generator", op, w=124.0, h=48.0)
        self.text(sc, 0.38, y + 0.072, "the same decoder", op)
        self.digit(sc, self.gan_shown[k], 0.56, y, 84, op, frame=RED)
        self.net(sc, 0.75, 0.60, JUDGE, "judge", op, w=96.0, h=56.0)
        w = (u * 5.0) % 1.0
        self.link(sc, (0.265, y), (0.31, y), RED, op, pulse=w)
        self.link(sc, (0.45, y), (0.505, y), RED, op, pulse=w)
        self.link(sc, (0.612, y), (0.70, 0.575), RED, op, pulse=w)
        # a real digit, and where an encoder would have been
        y2 = 0.76
        self.digit(sc, self.x0, 0.56, y2, 84, op, frame=AXIS)
        self.text(sc, 0.56, y2 + 0.105, "a real digit", op)
        self.link(sc, (0.612, y2), (0.70, 0.63), AXIS, op, pulse=w)
        self.box(sc, 0.25, y2, 250, 96, PAPER, op * 0.9, border=DIM)
        self.lines(sc, 0.25, y2 - 0.05, [("no encoder:", dark(RED, 0.3)),
                                          "no digit is put in,", "no pixels are compared"],
                   op, step=0.05, anchor="mm", size=0.033)
        # the verdicts at this update
        x = 0.815
        self.lines(sc, x, 0.52, [("update %s" % f"{int(self.gan_steps[k]):,}", None)],
                   op, size=0.034)
        self.lines(sc, x, 0.58, [("its digit: real? %s" % pct(self.gan_pf[k]), dark(RED, 0.3)),
                                 ("real digit: real? %s" % pct(self.gan_pr[k]), None)],
                   op, step=0.05, size=0.032)
        self.text(sc, x, 0.70, "right on %s" % pct(self.gan_right[k]), op, anchor="lm",
                  strong=True, size=0.034)
        self.text(sc, x, 0.745, "of images", op, anchor="lm")
        # the signal
        s = seg(u, 0.08, 0.3)
        self.link(sc, (0.75, 0.545), (0.38, 0.42), JUDGE, op * smoothstep(0.08, 0.14, u), s,
                  pulse=(u * 5.0 + 0.5) % 1.0 if u > 0.3 else None, bend=0.25)
        self.text(sc, 0.565, 0.305, "the signal: make the judge say real", op *
                  smoothstep(0.2, 0.3, u), ink=dark(JUDGE, 0.2), size=0.034)

    # ------------------------------------------------------------------ to centres
    def phase_4(self, sc, u, op):
        at = seg(u, 0.28, 0.9) * (len(self.dec_steps) - 1)
        k = int(round(at))
        pl = Plane(self, 0.235, 0.615, 300.0, self.dec_lim)
        pl.frame(sc, op, "codes of %d held-out digits" % F.CLOUD)
        z, mu = self.between(self.cloud, at), self.between(self.centres, at)
        pull = seg(u, 0.12, 0.28)
        near = ((z[:, None] - mu[None]) ** 2).sum(-1).argmin(1)
        with self.over():
            for i in range(0, len(z), 4):
                if pl.inside(z[i]) and pl.inside(mu[near[i]]):
                    self.arc(sc, pl.at(z[i]), pl.at(mu[near[i]]), AMBER, op * 0.55 * pull, pull,
                             bend=0.0, width=1.0, head=False)
        for digit in range(10):
            pl.dots(sc, z[self.cloud_labels == digit], TEN[digit], op, r=3.2)
        for c in mu[pl.inside(mu)]:
            self.disk(sc, self.P(*pl.at(c), lift=1.2), 6.5, WHITE, op, ring=AMBER, ring_w=3.0,
                      lift=1.6)
        x = 0.47
        self.tag(sc, x + 0.035, 0.33, "dec", op, w=58.0)
        self.text(sc, x + 0.08, 0.33, "drawn with 2-number codes", op, anchor="lm")
        y = 0.43
        self.digit(sc, self.x0, x + 0.03, y, 56, op, frame=AXIS)
        self.net(sc, x + 0.16, y, AMBER, "encoder", op, w=104.0)
        self.net(sc, x + 0.37, y, BLUE, "decoder", op, w=104.0, live=False)
        self.link(sc, (x + 0.07, y), (x + 0.10, y), AMBER, op, pulse=(u * 4.0) % 1.0)
        self.text(sc, x + 0.265, y, "code", op, ink=dark(AMBER, 0.3), size=0.034)
        self.lines(sc, x + 0.30, y + 0.07, ["the AE's, left as it was"], op, anchor="lm")
        self.lines(sc, x, 0.575, [
            ("the signal: move each code towards the", dark(AMBER, 0.35)),
            ("centre it is nearest, and the centres with them", dark(AMBER, 0.35))],
            op * smoothstep(0.1, 0.22, u), step=0.05, size=0.033)
        self.key(sc, x + 0.008, 0.695, WHITE, "a cluster centre (ten of them)", op, r=5.5)
        self.lines(sc, x, 0.765, [
            ("this 2-number run, update %d of %d" % (self.dec_steps[k], F.CLUSTER_STEPS), None),
            ("cluster accuracy %.2f  (was %.2f)" % (self.dec_acc[k], self.dec_acc[0]), None),
            ("rebuilding error %.3f  (was %.3f)" % (self.dec_mse[k], self.dec_mse[0]), None)],
            op, step=0.05, size=0.034)

    # ------------------------------------------------------------------ the runs
    def phase_5(self, sc, u, op):
        shots = self.shots.shape[1]
        self.text(sc, 0.14, 0.305, "start", op, anchor="lm")
        self.text(sc, 0.14 + 0.062 * (shots - 1) + 0.026, 0.305, "end of its run", op,
                  anchor="rm")
        for i, family in enumerate(F.FAMILIES):
            y = 0.385 + 0.105 * i
            self.tag(sc, 0.085, y, family, op, w=56.0)
            for j in range(shots):
                a = op * smoothstep(0.05 + 0.1 * j, 0.15 + 0.1 * j, u)
                self.digit(sc, self.shots[i, j], 0.165 + 0.062 * j, y, 42, a, frame=FAM[family])
        self.text(sc, 0.30, 0.905, "the GAN's row: one noise point, drawn at each stage", op)
        chart = self.chart(((0.63, 0.93), (0.36, 0.80)), [0, 1], [0.005, 0.25], ylog=True)
        self.axes(sc, chart, op, "share of its own run done", "error rebuilding held-out digits",
                  xticks=[(v, pct(v)) for v in (0.25, 0.5, 0.75, 1.0)],
                  yticks=[(v, "%g" % v) for v in (0.01, 0.03, 0.1, 0.2)])
        prog = seg(u, 0.1, 0.8)
        for i, family in enumerate(F.ENCODERS):
            end = self.plot(sc, chart, self.frac_run, self.mse_curves[i], FAM[family], op, prog,
                            width=2.2)
            if end is not None and prog > 0.97 and family in ("dec", "idec", "ae"):
                self.text(sc, end[0] - 0.012, end[1] - 0.03 if family != "ae" else end[1] + 0.036,
                          NAME[family] if family != "ae" else "AE, VAE", op, anchor="rm",
                          ink=dark(FAM[family], 0.3))
        self.text(sc, 0.78, 0.905, "GAN: no encoder, no such curve", op, ink=dark(RED, 0.3))

    # ------------------------------------------------------------------ the outcome
    def phase_6(self, sc, u, op):
        xs = (0.37, 0.53, 0.69, 0.85)
        heads = (("error rebuilding", "held-out digits"), ("codes inside", "the bell"),
                 ("cluster", "accuracy"), ("reader's confidence", "in its samples"))
        for x, head in zip(xs, heads):
            self.lines(sc, x, 0.32, list(head), op, step=0.04, anchor="mm")
        own = {"ae": (0,), "vae": (0, 1), "gan": (3,), "dec": (2,), "idec": (0, 2)}
        asked = ("rebuild", "rebuild, near a bell", "fool a judge", "cluster",
                 "cluster and rebuild")
        best = [int(np.argmin(self.table[:, 0]))] + [
            int(np.argmax(np.where(self.defined[:, j], self.table[:, j], -1))) for j in (2, 3)]
        gan = F.FAMILIES.index("gan")
        for i, family in enumerate(F.FAMILIES):
            y = 0.43 + 0.085 * i
            a = op * smoothstep(0.05 + 0.12 * i, 0.17 + 0.12 * i, u)
            self.tag(sc, 0.085, y, family, a, w=56.0)
            self.text(sc, 0.125, y, asked[i], a, anchor="lm", ink=dark(FAM[family], 0.35))
            for j, x in enumerate(xs):
                v = self.table[i, j]
                text = ("%.3f" % v if j == 0 else pct(v) if j == 1 else "%.2f" % v)
                if i == gan:
                    text = ("%.3f *" % v, "by design", "no clusters", text)[j]
                strong = (j == 0 and i == best[0]) or (j == 2 and i == best[1]) or (
                    j == 3 and i == best[2])
                self.table_cell(sc, x, y, text, a, FAM[family], own=j in own[family],
                                strong=strong)
        self.lines(sc, 0.5, 0.855, [
            "tinted: what that family is trained for; AE and VAE are clustered by k-means",
            "* a GAN has no encoder: found by searching its noise for each digit"],
            op * smoothstep(0.7, 0.8, u), step=0.045, anchor="mm")


# ====================================================================== 102 latent spaces
class LatentSpaces(Families):
    """Held-out digits as points in four 2-number code spaces, the GAN's noise
    plane as what it draws there, and three measures of those spaces."""

    TITLE = "Four code spaces, one noise plane"
    PHASES = (("to points", 10.0), ("four spaces", 13.0), ("GAN noise", 13.0),
              ("neighbours", 11.0), ("the bell", 10.0))
    DRAWN = 420                        # of the 2,000 held-out codes, in each plane

    def setup(self):
        d, m = F.latent_arrays(), self.study.metadata
        self.m = m
        self.codes = np.asarray(d["codes"], float)[:, 0]          # encoder, digit, 2
        self.labels = np.asarray(d["labels"])
        self.centres = np.asarray(d["centres"], float)[:, 0]
        self.bell_r = float(m["bell_radius"])
        self.lims = []
        for z in self.codes:                # a square holding the codes and the bell's circle
            lo = np.minimum(np.percentile(z, 0.5, axis=0), -self.bell_r)
            hi = np.maximum(np.percentile(z, 99.5, axis=0), self.bell_r)
            mid, half = (lo + hi) / 2, float((hi - lo).max()) * 0.54
            self.lims.append((mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half))
        self.shown = np.asarray(d["shown"], float)[:5]
        two = F.CODES.index(2)
        self.purity = np.asarray(d["purity"], float)[two]          # encoder, seed
        self.accuracy = np.asarray(d["accuracy"], float)[two]
        self.inside = np.asarray(d["inside"], float)[two]
        self.used = np.asarray(d["used"], float)[two]
        self.pixel_purity = float(d["pixel_purity"])
        self.pixel_accuracy = np.asarray(d["pixel_accuracy"], float)
        self.gan_purity = np.asarray(d["gan_purity"], float)
        # the GAN's plane: what it draws on a 9 x 9 grid of noise points, as the reader reads it
        self.G = F.params("gan", 2, SEED)
        self.tiles = np.asarray(d["tiles"], float)
        self.tile_grid = np.asarray(d["tile_grid"], float)
        self.tile_class = F.read(self.tiles).argmax(1)
        # a typical digit and its ten nearest neighbours in the AE's space
        z, y = self.codes[0], self.labels
        d2 = ((z[:, None] - z[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        near = np.argsort(d2, axis=1)[:, :10]
        same = (y[near] == y[:, None]).sum(1)
        typical = (same == round(10 * self.purity[0, 0])) & (y < 8)     # a vivid colour
        self.focus = int(np.flatnonzero(typical)[0])
        self.near = near[self.focus]
        self.same = int(same[self.focus])
        reach = float(np.sqrt(d2[self.focus, self.near[-1]])) * 1.9
        c = z[self.focus]
        self.zoom = (c[0] - reach, c[0] + reach, c[1] - reach, c[1] + reach)
        # the bell's draws, and which of them land near a held-out code (as the study counts)
        self.bell = np.random.default_rng([SEED, 2, 13]).normal(size=(F.TEST_N, 2))[:500]
        self.lands = {f: F.nearest(self.bell, self.codes[F.ENCODERS.index(f)])[0] <= F.REACH
                      for f in ("vae", "dec")}

    def subtitle_for(self, k):
        m = self.m
        ins, use, pur, acc = (m["inside_bell"], m["bell_used"], m["purity"],
                              m["cluster_accuracy"])
        return [
            "an encoder turns a digit into a code; with a code of just 2 numbers every "
            "held-out digit is a point on a plane, here coloured by which digit it is",
            "the same 2,000 held-out digits in four code spaces: the VAE keeps %s of them "
            "inside the standard bell's circle, the AE %s, DEC %s and IDEC %s, bunched on "
            "their ten centres" % tuple(pct(ins[f]["mean"]) for f in ("vae", "ae", "dec", "idec")),
            "a GAN has no encoder, so real digits cannot be placed on its noise plane; what "
            "can be shown is what it draws at each point: neighbouring points draw the same "
            "digit %s of the time" % pct(m["gan_noise_purity"]["mean"]),
            "of each code's 10 nearest neighbours, the share that are the same digit: AE "
            "%.2f, IDEC %.2f, VAE %.2f, DEC %.2f; DEC gives up neighbourhoods for clusters "
            "(accuracy %.2f against the AE's %.2f)" % (
                pur["ae"]["mean"], pur["idec"]["mean"], pur["vae"]["mean"], pur["dec"]["mean"],
                acc["dec"]["mean"], acc["ae"]["mean"]),
            "new digits are drawn from a standard bell: %s of its draws land near a VAE "
            "code, %s near an AE code, %s near a DEC code; a GAN's noise is the bell "
            "itself" % tuple(pct(use[f]["mean"]) for f in ("vae", "ae", "dec")),
        ][k]

    def _cloud(self, sc, pl, z, op, r=2.2, n=None):
        n = n or self.DRAWN
        for digit in range(10):
            pl.dots(sc, z[:n][self.labels[:n] == digit], TEN[digit], op, r=r)

    def _legend(self, sc, x, y, op, step=0.034):
        for digit in range(10):
            self.box(sc, x + step * digit, y, 24, 24, TEN[digit], op, str(digit), ink=WHITE,
                     size=0.031)

    def phase_1(self, sc, u, op):
        pl = Plane(self, 0.66, 0.60, 290.0, self.lims[0])
        pl.frame(sc, op, "the AE's code space")
        self._cloud(sc, pl, self.codes[0], op * smoothstep(0.6, 0.9, u), r=2.4)
        self.net(sc, 0.27, 0.60, BLUE, "AE encoder", op, w=120.0, h=46.0)
        self.text(sc, 0.215, 0.69, "784 pixels in, 2 numbers out", op, anchor="lm")
        now = int(np.clip((u - 0.06) / 0.11, 0, 4))
        for i, img in enumerate(self.shown):
            y = 0.36 + 0.12 * i
            ink = TEN[int(self.labels[i])]
            self.digit(sc, img, 0.09, y, 50, op, frame=ink)
            w = seg(u, 0.06 + 0.11 * i, 0.28 + 0.11 * i)
            a, b = (0.125, y), (0.205, 0.60)
            self.link(sc, a, b, ink, op * 0.9, min(w * 2, 1.0), pulse=seg(w, 0.0, 0.5),
                      bend=0.0, width=1.8)
            end = pl.at(self.codes[0][i])
            with self.over():
                self.link(sc, (0.335, 0.60), end, ink, op * 0.9, seg(w, 0.5, 1.0),
                          pulse=seg(w, 0.5, 1.0), bend=0.05, width=1.8)
            self.disk(sc, self.P(*end, lift=1.4), 6.0, ink, op * smoothstep(0.95, 1.0, w),
                      ring=WHITE, ring_w=1.6, lift=1.8)
        z = self.codes[0][now]
        self.text(sc, 0.215, 0.74, "this digit's code: (%.1f, %.1f)" % (z[0], z[1]), op,
                  strong=True, anchor="lm")
        self.lines(sc, 0.215, 0.815, ["2,000 held-out digits,", "none seen in training"],
                   op * smoothstep(0.6, 0.75, u), step=0.045)
        for digit in range(10):
            self.box(sc, 0.89, 0.375 + 0.052 * digit, 26, 24, TEN[digit], op, str(digit),
                     ink=WHITE, size=0.031)

    def phase_2(self, sc, u, op):
        for i, family in enumerate(F.ENCODERS):
            a = op * smoothstep(0.03 + 0.14 * i, 0.13 + 0.14 * i, u)
            pl = Plane(self, 0.1475 + 0.235 * i, 0.565, 186.0, self.lims[i])
            pl.frame(sc, a, NAME[family], ink=dark(FAM[family], 0.25))
            self._cloud(sc, pl, self.codes[i], a, r=2.0)
            with self.over(1.2):
                self.ring(sc, pl, self.bell_r, DIGIT_INK, a, width=2.0)
            if family in ("dec", "idec"):
                for c in self.centres[i - 2]:
                    if pl.inside(c):
                        self.disk(sc, self.P(*pl.at(c), lift=1.3), 4.5, WHITE, a,
                                  ring=FAM[family], ring_w=2.4, lift=1.9)
            self.text(sc, pl.fx, 0.785, "%s inside the circle" % pct(self.inside[i].mean()), a,
                      strong=True)
        note = op * smoothstep(0.6, 0.72, u)
        self.lines(sc, 0.5, 0.85, [
            "circle: where 99% of a standard bell's draws land (each plane has its own scale)",
            "rings: DEC's and IDEC's ten cluster centres.  Shares: mean of three seeds"],
            note, anchor="mm", step=0.045)

    def _probe(self, u):
        a = 2 * np.pi * (0.1 + 1.2 * u)
        r = 1.5 + 0.7 * np.sin(3 * a)
        return np.array([r * np.cos(a), r * np.sin(a)])

    def phase_3(self, sc, u, op):
        pl = Plane(self, 0.235, 0.615, 300.0, (-F.MAP_LIM, F.MAP_LIM, -F.MAP_LIM, F.MAP_LIM))
        pl.frame(sc, op, "the GAN's noise plane")
        for img, z, k in zip(self.tiles, self.tile_grid, self.tile_class):
            fx, fy = pl.at(z)
            self.digit(sc, img, fx, fy, 29, op, ink=dark(TEN[int(k)], 0.1))
        z = self._probe(u)
        img = F.decode(self.G, z)[0]
        p = F.read(img[None])[0]
        at = pl.at(z)
        self.disk(sc, self.P(*at, lift=1.6), 7.0, WHITE, op, ring=RED, ring_w=3.0, lift=2.2)
        self.digit(sc, img, 0.58, 0.44, 124, op, frame=RED)
        with self.over(1.4):
            self.arc(sc, at, (0.51, 0.44), RED, op * 0.8, bend=0.04, width=1.8)
        self.lines(sc, 0.67, 0.36, ["noise point", "(%.1f, %.1f)" % (z[0], z[1])], op,
                   step=0.05, size=0.034)
        self.lines(sc, 0.67, 0.48, ["the reader sees",
                                    "a %d, %s sure" % (p.argmax(), pct(p.max()))],
                   op, step=0.05, size=0.034)
        self.lines(sc, 0.50, 0.655, [
            ("no encoder: a real digit has no point here", dark(RED, 0.3)),
            "what can be shown: what the generator draws",
            "at each noise point, coloured by the digit",
            "a separately trained reader sees in it"], op, step=0.05, size=0.033)
        self._legend(sc, 0.515, 0.885, op)

    def phase_4(self, sc, u, op):
        z, y = self.codes[0], self.labels
        pl = Plane(self, 0.20, 0.55, 220.0, self.zoom)
        pl.frame(sc, op, "AE code space, close up")
        inside = pl.inside(z)
        for digit in range(10):
            pl.dots(sc, z[inside & (y == digit)], TEN[digit], op * 0.8, r=3.6)
        c = pl.at(z[self.focus])
        for j, n in enumerate(self.near):
            w = seg(u, 0.05 + 0.03 * j, 0.15 + 0.03 * j)
            with self.over():
                self.arc(sc, c, pl.at(z[n]), DIGIT_INK, op * 0.7, w, bend=0.0, width=1.4,
                         head=False)
            self.disk(sc, self.P(*pl.at(z[n]), lift=1.4), 5.0, TEN[int(y[n])],
                      op * smoothstep(0.8, 1.0, w), ring=DIGIT_INK, ring_w=1.8, lift=1.8)
        self.disk(sc, self.P(*c, lift=1.6), 7.0, TEN[int(y[self.focus])], op, ring=WHITE,
                  ring_w=2.4, lift=2.2)
        self.lines(sc, 0.20, 0.83, ["a held-out %d: %d of its 10 nearest" % (
            y[self.focus], self.same), "neighbours are the same digit"],
            op * smoothstep(0.35, 0.45, u), anchor="mm", step=0.045)
        self.text(sc, 0.20, 0.93, "* of the digit the reader sees in its draws", op)
        x, px = 0.585, 215.0
        grow = seg(u, 0.4, 0.75)
        self.text(sc, x, 0.31, "neighbours that are the same digit", op, anchor="lm", strong=True)
        rows = [(NAME[f], FAM[f], self.purity[i]) for i, f in enumerate(F.ENCODERS)]
        rows += [("raw pixels", AXIS, [self.pixel_purity]), ("GAN *", RED, self.gan_purity)]
        self.seed_bars(sc, x, [0.36 + 0.044 * i for i in range(6)], rows, px, 1.0, op, grow,
                       thick=14.0)
        self.text(sc, x, 0.655, "cluster accuracy", op, anchor="lm", strong=True)
        rows = [(NAME[f] + ("" if f in ("dec", "idec") else ", k-means"), FAM[f],
                 self.accuracy[i]) for i, f in enumerate(F.ENCODERS)]
        rows += [("raw pixels, k-means", AXIS, self.pixel_accuracy)]
        self.seed_bars(sc, x, [0.705 + 0.044 * i for i in range(5)], rows, px, 1.0, op,
                       seg(u, 0.6, 0.95), thick=14.0)

    def phase_5(self, sc, u, op):
        lim = (-3.7, 3.7, -3.7, 3.7)
        for i, family in enumerate(("vae", "dec")):
            pl = Plane(self, 0.165 + 0.235 * i, 0.555, 196.0, lim)
            pl.frame(sc, op, "bell draws, %s codes" % NAME[family], ink=dark(FAM[family], 0.25))
            hit = self.lands[family]
            show = op * smoothstep(0.05 + 0.2 * i, 0.2 + 0.2 * i, u)
            pl.dots(sc, self.bell[~hit], DIM, show, r=2.4)
            pl.dots(sc, self.bell[hit], FAM[family], show, r=2.4)
            with self.over(1.2):
                self.ring(sc, pl, self.bell_r, DIGIT_INK, op, width=1.8)
            self.lines(sc, pl.fx, 0.79, ["%s of these 500 draws" % pct(hit.mean()),
                                         "land near a code"], show, anchor="mm", step=0.045)
        self.text(sc, 0.28, 0.905, "coloured: a held-out code lies within %.2f; grey: none does"
                  % F.REACH, op)
        x, px = 0.70, 170.0
        self.text(sc, 0.60, 0.31, "codes inside the bell's circle", op, anchor="lm", strong=True)
        rows = [(NAME[f], FAM[f], self.inside[i]) for i, f in enumerate(F.ENCODERS)]
        rows += [("GAN", RED, [1.0])]
        self.seed_bars(sc, x, [0.36 + 0.046 * i for i in range(5)], rows, px, 1.0, op,
                       seg(u, 0.35, 0.65), thick=14.0,
                       note=[pct(v.mean()) for v in self.inside] + ["by design"])
        self.text(sc, 0.60, 0.63, "bell draws that land near a code", op, anchor="lm",
                  strong=True)
        rows = [(NAME[f], FAM[f], self.used[i]) for i, f in enumerate(F.ENCODERS)]
        self.seed_bars(sc, x, [0.68 + 0.046 * i for i in range(4)], rows, px, 1.0, op,
                       seg(u, 0.55, 0.9), thick=14.0, note=[pct(v.mean()) for v in self.used])


# ====================================================================== 103 sampling
SHORT = {"ae": "AE, bell codes", "ae_fit": "AE, own spread", "vae": "VAE, bell codes",
         "gan": "GAN, bell noise", "dec": "DEC, its centres", "idec": "IDEC, its centres"}
METHOD_INK = {"ae": BLUE, "ae_fit": BLUE, "vae": GREEN, "gan": RED, "dec": AMBER, "idec": OCHRE}


class Sampling(Families):
    """New digits from each family: where the codes come from, the draws, and
    three model-independent judgements of 2,000 of them."""

    TITLE = "New digits from four families"
    PHASES = (("codes", 11.0), ("draws", 10.0), ("a reader", 12.0), ("variety", 11.0),
              ("copies?", 11.0))

    def setup(self):
        d, s = F.sampling_arrays(), self.study
        self.m = s.metadata
        ci = F.CODES.index(F.WIDE)
        self.kept = np.asarray(d["kept"], float)[ci]
        self.near_img = np.asarray(d["kept_near"], float)[ci]
        self.near_dist = np.asarray(d["kept_dist"], float)[ci]
        self.metrics = np.asarray(s.panels[2].data["seeds"], float)      # method, seed, measure
        self.real = np.asarray(s.panels[2].data["real"], float)
        self.hist = np.asarray(s.panels[3].data["values"], float)
        self.real_hist = np.asarray(s.panels[3].data["real"], float)
        rng = np.random.default_rng([SEED, F.WIDE, 7])                   # the study's draws
        self.eps = rng.normal(size=(F.SAMPLES, F.WIDE))[:F.KEPT]
        pick = rng.integers(0, F.CLUSTERS, F.SAMPLES)[:F.KEPT]
        P = F.params("idec", F.WIDE, SEED)
        self.idec_z = P["mu"][pick] + F.cluster_spread(P, F.data()["x"])[pick] * self.eps
        self.probs = F.read(self.kept.reshape(-1, 784)).reshape(len(F.METHODS), -1, 10)
        self.reader_accuracy = F.reader()[1]

    def _row(self, method):
        return F.METHODS.index(method)

    def subtitle_for(self, k):
        t, real = self.m["table"], self.m["real"]
        def mean(method, measure):
            return t[method][measure]["mean"]

        return [
            "to draw a new digit each family decodes %d numbers: the VAE and the GAN take "
            "them from a standard bell, as trained; an AE was never trained to; DEC and "
            "IDEC have only their ten centres" % F.WIDE,
            "twelve of the 2,000 draws from each, all from the same noise; how good they "
            "are is judged next by a separately trained classifier and by distances, not "
            "by eye",
            "a separate classifier reads 2,000 draws of each: it is surest of the GAN's "
            "(%.2f; real digits %.2f) and least sure of the AE's bell draws (%.2f), which "
            "cover only %d to %d of the ten digits" % (
                mean("gan", "confidence"), real["confidence"], mean("ae", "confidence"),
                t["ae"]["covered"]["min"], t["ae"]["covered"]["max"]),
            "variety, the mean distance between two draws: GAN %.1f and IDEC %.1f come "
            "close to real digits (%.1f); the VAE's draws are more alike (%.1f), the AE's "
            "bell draws most of all (%.1f)" % (
                mean("gan", "variety"), mean("idec", "variety"), real["variety"],
                mean("vae", "variety"), mean("ae", "variety")),
            "copied? each draw's distance to the nearest of %s training digits: VAE %.1f, "
            "GAN %.1f, IDEC %.1f; real held-out digits %.1f, a copy 0; the AE's bell draws "
            "are far from any digit (%.1f)" % (
                f"{F.TRAIN_N:,}", mean("vae", "nearest"), mean("gan", "nearest"),
                mean("idec", "nearest"), real["nearest"], mean("ae", "nearest")),
        ][k]

    def phase_1(self, sc, u, op):
        cyc, k = (u * 4.0) % 1.0, int(u * 4.0) % 4
        notes = {"ae": "its codes never lived there", "vae": "its codes were pushed there",
                 "gan": "noise is all it ever saw", "idec": "a centre, plus its spread"}
        for x, head in ((0.185, "codes from"), (0.36, "%d numbers" % F.WIDE),
                        (0.665, "a new digit")):
            self.text(sc, x, 0.305, head, op, strong=True)
        vmax = float(np.abs(self.eps).max())
        for i, family in enumerate(("ae", "vae", "gan", "idec")):
            y, ink = 0.385 + 0.148 * i, FAM[family]
            self.tag(sc, 0.085, y, family, op, w=56.0)
            if family == "idec":
                for j, (dx, dy) in enumerate(((-0.035, 0.012), (-0.012, -0.022), (0.012, 0.018),  # noqa: B007
                                              (0.034, -0.012), (0.0, 0.0))):
                    self.disk(sc, self.P(0.185 + dx, y + dy), 4.5, WHITE, op, ring=ink,
                              ring_w=2.2)
                z, vm = self.idec_z[k], float(np.abs(self.idec_z).max())
            else:
                self.bell(sc, 0.185, y + 0.028, 92, 30, op, AXIS, 2.2)
                z, vm = self.eps[k], vmax
            self.link(sc, (0.24, y), (0.268, y), ink, op, pulse=seg(cyc, 0.0, 0.25))
            self.strip(sc, z, 0.36, y, 160, 20, op * smoothstep(0.1, 0.25, cyc), ink, vm)
            self.link(sc, (0.452, y), (0.482, y), ink, op, pulse=seg(cyc, 0.2, 0.45))
            self.net(sc, 0.545, y, ink, "generator" if family == "gan" else "decoder", op,
                     w=112.0)
            self.link(sc, (0.607, y), (0.63, y), ink, op, pulse=seg(cyc, 0.4, 0.65))
            self.digit(sc, self.kept[self._row(family), k], 0.665, y, 58, op *
                       smoothstep(0.55, 0.7, cyc), frame=ink)
            self.text(sc, 0.71, y, notes[family], op, anchor="lm", ink=dark(ink, 0.35))

    def phase_2(self, sc, u, op):
        for i, method in enumerate(F.METHODS):
            y = 0.345 + 0.103 * i
            self.text(sc, 0.27, y, SHORT[method], op, anchor="rm", size=0.033,
                      ink=dark(METHOD_INK[method], 0.35))
            for k in range(12):
                a = op * smoothstep(0.03 + 0.06 * k, 0.12 + 0.06 * k, u)
                self.digit(sc, self.kept[i, k], 0.305 + 0.056 * k, y, 40, a,
                           frame=METHOD_INK[method])

    ROWS = F.METHODS + ("real",)

    def _labels(self):
        return ["AE, bell", "AE, own spread", "VAE", "GAN", "DEC", "IDEC", "real digits"]

    def _values(self, measure):
        k = F.MEASURES.index(measure)
        return [self.metrics[i, :, k] for i in range(len(F.METHODS))] + [[self.real[k]]]

    def _inks(self):
        return [METHOD_INK[m] for m in F.METHODS] + [AXIS]

    def phase_3(self, sc, u, op):
        trio = ("ae", "vae", "gan")
        j, cyc = int(u * 3.0) % 3, (u * 3.0) % 1.0
        method = trio[j]
        row, ink = self._row(method), METHOD_INK[method]
        p = self.probs[row, 1]
        self.text(sc, 0.095, 0.345, SHORT[method].split(",")[0] + " draw", op, ink=dark(ink, 0.3))
        self.digit(sc, self.kept[row, 1], 0.095, 0.45, 72, op, frame=ink)
        self.link(sc, (0.14, 0.45), (0.165, 0.45), READER, op, pulse=seg(cyc, 0.0, 0.3))
        self.net(sc, 0.215, 0.45, READER, "reader", op, w=88.0, h=44.0)
        self.link(sc, (0.265, 0.45), (0.29, 0.45), READER, op, pulse=seg(cyc, 0.25, 0.5))
        grow = seg(cyc, 0.4, 0.65)
        for c in range(10):
            fx = 0.305 + 0.0175 * c
            self.vbar(sc, fx, 0.52, 11, 84 * p[c], READER, op, grow)
            self.text(sc, fx, 0.548, str(c), op)
        self.text(sc, 0.385, 0.33, "sees a %d, %s sure" % (p.argmax(), pct(p.max())),
                  op * smoothstep(0.6, 0.7, cyc), strong=True, size=0.034)
        self.lines(sc, 0.06, 0.655, [
            "the reader: a classifier trained separately",
            "on real digits (%s right on held-out ones)" % pct(self.reader_accuracy),
            "confidence: its tallest bar",
            "covered: a digit it reads in 3% of draws or more"], op, step=0.05, size=0.031)
        self.text(sc, 0.645, 0.325, "confidence", op, anchor="lm", strong=True)
        self.text(sc, 0.935, 0.325, "digits covered", op, anchor="rm", strong=True)
        ys = [0.39 + 0.07 * i for i in range(7)]
        self.seed_bars(sc, 0.645, ys, list(zip(self._labels(), self._inks(),
                                               self._values("confidence"))),
                       130.0, 1.0, op, seg(u, 0.1, 0.5), thick=16.0)
        for y, v in zip(ys, self._values("covered")):
            lo, hi = int(min(v)), int(max(v))
            self.text(sc, 0.935, y, ("%d of 10" % hi) if lo == hi else "%d to %d" % (lo, hi),
                      op * smoothstep(0.45, 0.6, u), anchor="rm")

    def phase_4(self, sc, u, op):
        self.text(sc, 0.215, 0.31, "which digit the reader sees, 0 to 9", op, anchor="lm",
                  strong=True)
        self.text(sc, 0.63, 0.31, "variety: distance between two draws", op, anchor="lm",
                  strong=True)
        hists = list(self.hist) + [self.real_hist]
        ys = [0.40 + 0.076 * i for i in range(7)]
        rows = []
        for i, (y, label, ink, h, v) in enumerate(zip(ys, self._labels(), self._inks(), hists,
                                                      self._values("variety"))):
            self.text(sc, 0.195, y - 0.02, label, op, anchor="rm", ink=dark(ink, 0.35))
            grow = seg(u, 0.03 + 0.07 * i, 0.2 + 0.07 * i)
            for c in range(10):
                self.vbar(sc, 0.225 + 0.027 * c, y, 19, 34 * min(h[c] / 0.35, 1.0), ink, op, grow)
            rows.append(("", ink, v))
        for c in range(10):
            self.text(sc, 0.225 + 0.027 * c, 0.885, str(c), op)
        self.seed_bars(sc, 0.63, [y - 0.018 for y in ys], rows, 235.0, 11.0, op,
                       seg(u, 0.45, 0.85), fmt="%.1f", thick=16.0)
        self.text(sc, 0.63, 0.905, "bars: mean of three seeds, a dot each", op, anchor="lm")

    def phase_5(self, sc, u, op):
        self.text(sc, 0.14, 0.31, "a draw, its nearest training digit, how far", op,
                  anchor="lm", strong=True)
        for i, family in enumerate(("ae", "vae", "gan", "idec")):
            y, row, ink = 0.395 + 0.125 * i, self._row(family), FAM[family]
            self.tag(sc, 0.085, y, family, op, w=56.0)
            for k in range(3):
                x = 0.165 + 0.155 * k
                w = seg(u, 0.04 + 0.05 * (i + k), 0.2 + 0.05 * (i + k))
                self.digit(sc, self.kept[row, k], x, y, 42, op, frame=ink)
                self.digit(sc, self.near_img[row, k], x + 0.056, y, 42,
                           op * smoothstep(0.5, 1.0, w), frame=AXIS)
                self.text(sc, x + 0.09, y, "%.1f" % self.near_dist[row, k],
                          op * smoothstep(0.8, 1.0, w), anchor="lm", strong=True)
        self.text(sc, 0.645, 0.31, "distance to the nearest", op, anchor="lm", strong=True)
        self.seed_bars(sc, 0.735, [0.375 + 0.062 * i for i in range(7)],
                       list(zip(self._labels(), self._inks(), self._values("nearest"))),
                       150.0, 8.0, op, seg(u, 0.4, 0.8), fmt="%.1f", thick=16.0)
        self.lines(sc, 0.5, 0.86, [
            "a copied training digit would be at 0: no family is near that",
            "real held-out digits are %.1f from their own nearest training digit"
            % self.real[F.MEASURES.index("nearest")]], op * smoothstep(0.7, 0.8, u),
            anchor="mm", step=0.045)


# ====================================================================== 104 reconstruction
class Reconstruction(Families):
    """The same held-out digits through each family and back; the GAN's by a
    search of its noise; the error, and what kind of thing is lost."""

    TITLE = "There and back: rebuilding a digit"
    PHASES = (("a code", 10.0), ("search", 14.0), ("compare", 10.0), ("error", 12.0),
              ("lost", 11.0))

    def setup(self):
        d, inv, s = F.recon_arrays(), F.inversions(), self.study
        self.m = s.metadata
        ci = F.CODES.index(F.WIDE)
        self.orig = np.asarray(d["originals"], float)
        self.kept = np.asarray(d["kept"], float)[ci]                    # family, digit, 784
        self.err = ((self.kept - self.orig[None]) ** 2).mean(-1)
        self.codes = {f: F.encode(F.params(f, F.WIDE, SEED), self.orig) for f in F.ENCODERS}
        head = "c%d_s%d_" % (F.WIDE, SEED)
        self.path_img = np.asarray(inv[head + "path_img"], float)[:, F.TRACK]
        self.path_z = np.asarray(inv[head + "path_z"], float)[:, F.TRACK]
        self.target = self.orig[F.TRACK]
        self.path_err = ((self.path_img - self.target) ** 2).mean(1)
        self.search = np.asarray(s.panels[2].data["ys"], float)
        self.mse = np.asarray(d["mse"], float)[ci]                      # family, seed
        self.mse2 = np.asarray(d["mse"], float)[1 - ci].mean(1)
        self.maps = np.asarray(s.panels[3].data["values"], float)
        self.own = float(self.m["gan_own_sample_mse"]["mean"])

    def subtitle_for(self, k):
        m = self.m
        e = {f: m["mse"][f]["mean"] for f in F.FAMILIES}
        r = {f: pct(m["read_right"][f]["mean"]) for f in F.FAMILIES}
        sh = {f: m["sharpness"][f]["mean"] for f in F.FAMILIES}
        return [
            "the same held-out digit into each encoder, down to %d numbers and back out; "
            "the VAE uses the centre of its code, and DEC, which trains no decoder, is "
            "shown with the AE's" % F.WIDE,
            "a GAN has no encoder: to rebuild a digit, search its noise by gradient descent "
            "until the output matches; error ends at %.3f for held-out digits, %.4f for "
            "digits it drew itself" % (e["gan"], self.own),
            "eight held-out digits and what comes back; a reader still sees the same digit "
            "in %s of the AE's, %s of the VAE's, %s of IDEC's, %s of the GAN's and %s of "
            "DEC's (originals: %s)" % (r["ae"], r["vae"], r["idec"], r["gan"], r["dec"],
                                        pct(m["read_right_original"])),
            "error over %d held-out digits, three seeds: AE %.3f, VAE %.3f, IDEC %.3f, GAN "
            "by search %.3f, DEC %.3f; the search is not the limit: the GAN's own samples "
            "come back at %.4f" % (m["digits"], e["ae"], e["vae"], e["idec"], e["gan"],
                                   e["dec"], self.own),
            "what is lost: the AE and VAE soften strokes (%.2f and %.2f of the original's "
            "sharpness) but keep the digit; the GAN keeps strokes sharp (%.2f) and changes "
            "the digit more often" % (sh["ae"], sh["vae"], sh["gan"]),
        ][k]

    def phase_1(self, sc, u, op):
        j, cyc = int(u * 3.0) % 3, (u * 3.0) % 1.0
        digit = (F.TRACK, 1, 5)[j]
        max(float(np.abs(z).max()) for z in self.codes.values())
        for i, family in enumerate(F.ENCODERS):
            y, row = 0.355 + 0.14 * i, F.FAMILIES.index(family)
            stale = family == "dec"
            self.chain(sc, y, op, FAM[family], self.orig[digit], self.codes[family][digit],
                       self.kept[row, digit], seg(cyc, 0.0, 0.8),
                       xs=(0.10, 0.25, 0.44, 0.63, 0.775), px=58.0, bw=126.0,
                       enc="%s encoder" % NAME[family],
                       dec="AE's decoder" if stale else "decoder",
                       dec_ink=BLUE if stale else None, dec_live=not stale, strip_w=160.0,
                       vmax=float(np.abs(self.codes[family]).max()))
            self.text(sc, 0.83, y, "error %.3f" % self.err[row, digit],
                      op * smoothstep(0.8, 0.9, cyc), anchor="lm", strong=True)
        self.text(sc, 0.5, 0.905, "DEC moved its encoder to cluster; the decoder is the "
                  "autoencoder's, never retrained", op, ink=dark(AMBER, 0.35))

    def phase_2(self, sc, u, op):
        steps = F.INV_AT
        k = min(int(seg(u, 0.12, 0.85) * len(steps)), len(steps) - 1)
        self.text(sc, 0.11, 0.33, "held-out digit", op)
        self.digit(sc, self.target, 0.11, 0.47, 100, op, frame=AXIS)
        self.text(sc, 0.39, 0.33, "the generator's output", op, ink=dark(RED, 0.3))
        self.digit(sc, self.path_img[k], 0.39, 0.47, 100, op, frame=RED)
        self.lines(sc, 0.25, 0.42, ["compare", "step %d" % steps[k],
                                    "error %.3f" % self.path_err[k]], op, anchor="mm",
                   step=0.05, size=0.033)
        self.net(sc, 0.39, 0.69, RED, "generator", op, w=124.0, h=44.0)
        self.link(sc, (0.39, 0.645), (0.39, 0.585), RED, op, pulse=(u * 6.0) % 1.0)
        vmax = float(np.abs(self.path_z).max())
        self.strip(sc, self.path_z[k], 0.39, 0.82, 190, 22, op, RED, vmax)
        self.link(sc, (0.39, 0.795), (0.39, 0.74), RED, op, pulse=(u * 6.0 + 0.5) % 1.0)
        self.text(sc, 0.39, 0.875, "its noise: %d numbers, nudged every step" % F.WIDE, op)
        self.link(sc, (0.25, 0.545), (0.27, 0.82), dark(RED, 0.25), op, seg(u, 0.05, 0.2),
                  pulse=(u * 6.0) % 1.0, bend=-0.05)
        self.lines(sc, 0.06, 0.66, ["the error says", "which way", "to nudge"], op,
                   step=0.045)
        chart = self.chart(((0.60, 0.93), (0.36, 0.72)), [0, F.INV_STEPS], [3e-5, 0.15],
                           ylog=True)
        self.axes(sc, chart, op, "search step", "error of the output",
                  yticks=[(v, "%g" % v) for v in (0.0001, 0.001, 0.01, 0.1)])
        xs = np.arange(F.INV_STEPS + 1)
        prog = steps[k] / F.INV_STEPS
        a = self.plot(sc, chart, xs, self.search[0], RED, op, prog, width=2.4)
        b = self.plot(sc, chart, xs, self.search[1], TEAL, op, prog, width=2.4)
        if k == len(steps) - 1 and a and b:
            at = F.INV_STEPS // 2
            for curve, ink, name, fmt, up in ((0, RED, "held-out digits", "%.3f", 2.4),
                                              (1, TEAL, "its own samples", "%.4f", 14.0)):
                self.text(sc, chart.fx(at), chart.fy(self.search[curve, at] * up),
                          ("%s: " + fmt) % (name, self.search[curve, -1]), op, anchor="lm",
                          ink=dark(ink, 0.3))
        self.lines(sc, 0.60, 0.845, [
            "%d random starts, %d steps each, the best kept" % (F.INV_RESTARTS, F.INV_STEPS),
            "control: digits the generator drew itself"], op, step=0.045)

    def phase_3(self, sc, u, op):
        labels = ["held-out digit"] + ["AE", "VAE", "GAN, by search", "DEC, AE's decoder", "IDEC"]
        rows = [self.orig] + list(self.kept)
        inks = [AXIS] + [FAM[f] for f in F.FAMILIES]
        for i, (label, imgs, ink) in enumerate(zip(labels, rows, inks)):
            y = 0.345 + 0.103 * i
            a = op * smoothstep(0.02 + 0.12 * i, 0.14 + 0.12 * i, u)
            self.text(sc, 0.335, y, label, a, anchor="rm", size=0.033, ink=dark(ink, 0.35))
            for k, img in enumerate(imgs):
                self.digit(sc, img, 0.375 + 0.07 * k, y, 42, a, frame=ink)

    def phase_4(self, sc, u, op):
        j = int(u * 5.0) % 5
        family = F.FAMILIES[j]
        ink, out = FAM[family], self.kept[j, F.TRACK]
        sq = (out - self.target) ** 2
        y = 0.50
        for x, head in ((0.10, "held-out digit"), (0.235, "from the %s" % NAME[family]),
                        (0.37, "difference")):
            self.text(sc, x, y - 0.115, head, op, ink=dark(ink, 0.3) if x > 0.2 else None)
        self.digit(sc, self.target, 0.10, y, 88, op, frame=AXIS)
        self.digit(sc, out, 0.235, y, 88, op, frame=ink)
        self.grid(sc, sq.reshape(28, 28), 0.37, y, 88, 88, op, vmax=0.8, hot=ink, lines=0)
        self.text(sc, 0.235, y + 0.125, "this digit's error: %.3f" % self.err[j, F.TRACK], op,
                  strong=True, size=0.034)
        self.lines(sc, 0.235, 0.71, ["error: the squared difference,",
                                     "averaged over the 784 pixels"], op, anchor="mm",
                   step=0.045)
        self.text(sc, 0.63, 0.325, "error, %d-number code" % F.WIDE, op, anchor="lm", strong=True)
        self.text(sc, 0.935, 0.325, "2 numbers", op, anchor="rm", strong=True)
        names = ["AE", "VAE", "GAN, search", "DEC", "IDEC"]
        ys = [0.395 + 0.08 * i for i in range(5)]
        self.seed_bars(sc, 0.63, ys, [(n, FAM[f], v) for n, f, v in
                                      zip(names, F.FAMILIES, self.mse)],
                       165.0, 0.08, op, seg(u, 0.05, 0.4), fmt="%.3f", thick=18.0)
        for i, (yy, v) in enumerate(zip(ys, self.mse2)):
            self.text(sc, 0.935, yy, "%.3f" % v, op * smoothstep(0.35, 0.5, u), anchor="rm")
            if i == j:
                self.disk(sc, self.P(0.50, yy), 4.5, FAM[family], op, glow=0.4)
        self.lines(sc, 0.52, 0.82, [
            "control: a GAN's own samples come back at %.4f," % self.own,
            "so its gap is what the generator cannot draw"], op * smoothstep(0.5, 0.6, u),
            step=0.045)

    def phase_5(self, sc, u, op):
        m = self.m
        vmax = float(self.maps.max())
        for i, family in enumerate(F.FAMILIES):
            x = 0.14 + 0.18 * i
            a = op * smoothstep(0.03 + 0.1 * i, 0.15 + 0.1 * i, u)
            self.tag(sc, x, 0.33, family, a, w=60.0)
            self.grid(sc, self.maps[i].reshape(28, 28), x, 0.485, 116, 116, a, vmax=vmax,
                      hot=FAM[family], lines=0)
            b = a * smoothstep(0.5, 0.62, u)
            self.text(sc, x, 0.64, "same digit: %s" % pct(m["read_right"][family]["mean"]), b,
                      strong=True)
            self.text(sc, x, 0.69, "sharpness: %.2f" % m["sharpness"][family]["mean"], b,
                      strong=True)
        self.lines(sc, 0.5, 0.77, [
            "maps: where the squared error falls, over %d held-out digits (one scale for all)"
            % m["digits"],
            "same digit: a reader still sees the original's digit (in the originals: %s)"
            % pct(m["read_right_original"]),
            "sharpness: how strong the stroke edges are, against the original's (1: as sharp)"],
            op * smoothstep(0.6, 0.72, u), anchor="mm", step=0.045)


# ====================================================================== 105 interpolation
class Interpolation(Families):
    """A straight line between two held-out digits in each code space, the
    images along it, and two measures of the middle against a cross-fade."""

    TITLE = "Walking between two digits"
    PHASES = (("a straight line", 12.0), ("six walks", 14.0), ("a reader", 13.0),
              ("how real", 14.0))
    ORDER = (5, 0, 1, 2, 3, 4)         # the cross-fade first, then the families

    def setup(self):
        d, s = F.interp_arrays(), self.study
        self.m = s.metadata
        ci = F.CODES.index(F.WIDE)
        self.kept = np.asarray(d["kept"], float)[ci]                    # walker, pair, step, 784
        self.ends = np.asarray(d["ends"], float)
        self.pair_labels = np.asarray(d["pair_labels"])
        self.t = np.asarray(s.panels[1].data["x"], float)
        self.conf = np.asarray(s.panels[1].data["ys"], float)
        self.near = np.asarray(s.panels[2].data["ys"], float)
        self.excess = self.near - (self.near[:, :1] * (1 - self.t) + self.near[:, -1:] * self.t)
        self.further = F.bump(np.asarray(d["walk"], float)[ci][:, :, 1])   # seed, walker
        self.ae = F.params("ae", F.WIDE, SEED)
        self.za, self.zb = F.encode(self.ae, self.ends[0])
        mid = F.WALK // 2
        trio = self.kept[0, 0][[0, mid, -1]]
        self.trio = trio
        self.trio_p = F.read(trio)
        x = F.data()["x"]
        self.mids = {}
        for family in ("pixels", "gan"):
            img = self.kept[F.WALKERS.index(family), 0, mid]
            dist, index = F.nearest(img[None], x)
            self.mids[family] = (img, x[index[0]], float(dist[0]))

    def subtitle_for(self, k):
        m = self.m
        c, n, f = m["middle_confidence"], m["middle_neither"], m["middle_further"]
        return [
            "encode two held-out digits, walk a straight line between their two codes and "
            "decode each point on the way: in each family's own code space, for %d pairs of "
            "different digits" % m["pairs"],
            "one pair, six walks, each from its own rebuilding of A to its own rebuilding of "
            "B; the cross-fade, with no model at all, shows both digits at once half-way",
            "half-way the reader is less sure on every walk: %.2f for the cross-fade, %.2f "
            "AE, %.2f VAE; it is surest of the GAN's middles (%.2f) yet sees a third digit "
            "in %s of them, against %s for the cross-fade" % (
                c["pixels"]["mean"], c["ae"]["mean"], c["vae"]["mean"], c["gan"]["mean"],
                pct(n["gan"]["mean"]), pct(n["pixels"]["mean"])),
            "how real is the middle? distance to the nearest training digit, middle minus "
            "ends: +%.2f for the cross-fade, +%.2f VAE, +%.2f AE, +%.2f IDEC, +%.2f GAN: "
            "the code-space walks stay nearer real digits" % (
                f["pixels"]["mean"], f["vae"]["mean"], f["ae"]["mean"], f["idec"]["mean"],
                f["gan"]["mean"]),
        ][k]

    def phase_1(self, sc, u, op):
        t = seg(u, 0.3, 0.9)
        vmax = float(max(np.abs(self.za).max(), np.abs(self.zb).max()))
        for x, img, z, name in ((0.14, self.ends[0, 0], self.za, "digit A"),
                                (0.86, self.ends[0, 1], self.zb, "digit B")):
            self.text(sc, x, 0.295, name, op, strong=True)
            self.digit(sc, img, x, 0.41, 72, op, frame=AXIS)
            self.link(sc, (x, 0.485), (x, 0.52), BLUE, op, pulse=seg(u, 0.02, 0.12))
            self.net(sc, x, 0.565, BLUE, "AE encoder", op, w=120.0, h=42.0)
            self.link(sc, (x, 0.61), (x, 0.665), BLUE, op, pulse=seg(u, 0.1, 0.2))
            self.strip(sc, z, x, 0.70, 150, 20, op * smoothstep(0.15, 0.22, u), BLUE, vmax)
        a = op * smoothstep(0.2, 0.3, u)
        self.polyline(sc, [self.P(0.225, 0.70), self.P(0.775, 0.70)], 2.0, BLUE, a * 0.7)
        xm = 0.335 + 0.33 * t
        z = (1 - t) * self.za + t * self.zb
        with self.over(1.0):
            self.strip(sc, z, xm, 0.70, 150, 20, a, BLUE, vmax)
        self.link(sc, (xm, 0.665), (xm, 0.61), BLUE, a, pulse=(u * 6.0) % 1.0)
        self.net(sc, xm, 0.565, BLUE, "decoder", a, w=110.0, h=42.0)
        self.link(sc, (xm, 0.52), (xm, 0.485), BLUE, a, pulse=(u * 6.0 + 0.5) % 1.0)
        self.digit(sc, F.decode(self.ae, z)[0], xm, 0.41, 72, a, frame=BLUE)
        self.text(sc, 0.5, 0.775, "%d%% of A's numbers + %d%% of B's" % (
            round(100 * (1 - t)), round(100 * t)), a, strong=True, size=0.034)
        self.lines(sc, 0.5, 0.845, [
            "a GAN has no encoder: its two ends are the noise points a search found",
            "the control: no model at all, fade A's pixels into B's"],
            op * smoothstep(0.55, 0.68, u), anchor="mm", step=0.045)

    def phase_2(self, sc, u, op):
        pair = 0 if u < 0.5 else 1
        v = seg(u, 0.0, 0.5) if pair == 0 else seg(u, 0.5, 1.0)
        fade = min(1.0, (1.0 - v) / 0.08)
        labels = {"pixels": "cross-fade, no model", "gan": "GAN, searched ends"}
        for x, head in ((0.335, "A"), (0.335 + 0.068 * 4, "half-way"), (0.335 + 0.068 * 8, "B")):
            self.text(sc, x, 0.305, head, op, strong=True)
        for i, w in enumerate(self.ORDER):
            family = F.WALKERS[w]
            y = 0.375 + 0.098 * i
            self.text(sc, 0.295, y, labels.get(family, NAME[family]), op, anchor="rm",
                      size=0.033, ink=dark(FAM[family], 0.35))
            for k in range(F.WALK):
                a = op * fade * smoothstep(0.04 + 0.07 * k, 0.12 + 0.07 * k, v)
                self.digit(sc, self.kept[w, pair, k], 0.335 + 0.068 * k, y, 40, a,
                           frame=FAM[family])

    def phase_3(self, sc, u, op):
        chart = self.chart(((0.10, 0.55), (0.37, 0.80)), [0, 1], [0.7, 1.0], pad=0.0)
        self.axes(sc, chart, op, "", "the reader's confidence along the walk",
                  xticks=[(0, "A"), (0.5, "half-way"), (1, "B")],
                  yticks=[(v, "%.1f" % v) for v in (0.7, 0.8, 0.9, 1.0)])
        prog = seg(u, 0.08, 0.6)
        for w in self.ORDER:
            self.plot(sc, chart, self.t, self.conf[w], FAM[F.WALKERS[w]], op, prog, width=2.3)
        for j, x in enumerate((0.68, 0.79, 0.90)):
            p = self.trio_p[j]
            self.digit(sc, self.trio[j], x, 0.375, 54, op, frame=BLUE)
            self.text(sc, x, 0.46, "a %d, %s" % (p.argmax(), pct(p.max())), op, ink=dark(READER))
        self.text(sc, 0.625, 0.375, "AE", op, anchor="rm", ink=dark(BLUE, 0.3), size=0.034)
        c, n = self.m["middle_confidence"], self.m["middle_neither"]
        self.text(sc, 0.80, 0.525, "half-way", op, anchor="rm", strong=True)
        self.text(sc, 0.935, 0.525, "a third digit", op, anchor="rm", strong=True)
        for i, w in enumerate(self.ORDER):
            family = F.WALKERS[w]
            y = 0.58 + 0.055 * i
            a = op * smoothstep(0.5 + 0.05 * i, 0.6 + 0.05 * i, u)
            self.key(sc, 0.61, y, FAM[family], NAME[family], a)
            self.text(sc, 0.80, y, "%.2f" % c[family]["mean"], a, anchor="rm")
            self.text(sc, 0.935, y, pct(n[family]["mean"]), a, anchor="rm")

    def phase_4(self, sc, u, op):
        lo, hi = float(self.excess.min()), float(self.excess.max())
        chart = self.chart(((0.10, 0.50), (0.37, 0.80)), [0, 1], [min(lo, 0.0), hi], pad=0.08)
        self.axes(sc, chart, op, "", "further from any training digit than its ends",
                  xticks=[(0, "A"), (0.5, "half-way"), (1, "B")])
        prog = seg(u, 0.08, 0.55)
        for w in self.ORDER:
            self.plot(sc, chart, self.t, self.excess[w], FAM[F.WALKERS[w]], op, prog, width=2.3)
        self.text(sc, 0.58, 0.31, "a middle, its nearest training digit", op, anchor="lm",
                  strong=True)
        for x, family in ((0.615, "pixels"), (0.80, "gan")):
            img, near, dist = self.mids[family]
            self.digit(sc, img, x, 0.40, 54, op, frame=FAM[family])
            self.digit(sc, near, x + 0.065, 0.40, 54, op, frame=AXIS)
            self.text(sc, x + 0.103, 0.40, "%.1f" % dist, op, anchor="lm", strong=True)
        self.text(sc, 0.58, 0.505, "middle further than the ends, by", op, anchor="lm",
                  strong=True)
        rows = [(NAME[F.WALKERS[w]], FAM[F.WALKERS[w]],
                 self.further[:, w] if F.WALKERS[w] != "pixels" else self.further[:1, w])
                for w in self.ORDER]
        self.seed_bars(sc, 0.70, [0.56 + 0.052 * i for i in range(6)], rows, 170.0, 1.0, op,
                       seg(u, 0.45, 0.8), thick=15.0)
        self.text(sc, 0.5, 0.905, "DEC stays flat, but its decoder is the AE's, left behind: "
                  "a reader already sees the wrong digit in %s of its ends"
                  % pct(self.m["end_neither"]["dec"]["mean"]), op * smoothstep(0.75, 0.85, u))


ANIMATIONS = {
    "generative-objectives": Objectives,
    "generative-latent-spaces": LatentSpaces,
    "generative-sampling": Sampling,
    "generative-reconstruction": Reconstruction,
    "generative-interpolation": Interpolation,
}
