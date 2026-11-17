"""Animations 170-171 of the "next" catalogue (see :mod:`voxanim.mech`).

``vae-kl-term``: one digit through a 2-number VAE, what the KL term of its loss measures
(two bells, one formula, real numbers), the tug of war with the reconstruction term, and
the cloud of 2,000 held-out codes at 14 checkpoints of training against the prior's rings.

``dec-centroids-training``: k-means placing ten centres in an autoencoder's code space,
DEC's soft assignments and sharpened targets, and the codes and the centres moving
through 1,500 DEC updates, with what that does to separation and to label purity.

Every latent-space picture uses the catalogue's ten digit colours
(:data:`voxanim.mech_families.TEN`) with a legend; the 2,000 points of a picture are
drawn into one texture (:func:`scatter`), which is what keeps them affordable.
Every number on screen is the study's (:mod:`voxanim.study_kl`) or is computed here from
its recorded codes and weights.
"""

from __future__ import annotations

import numpy as np

from . import deep_models as dm
from . import layout
from . import study_families as F
from . import study_kl as S
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    GREEN,
    MAGENTA,
    PAPER,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_families import TEN, WHITE, Families, pct
from .mech_generative import Plane, seg

KLC, REC = MAGENTA, TEAL               # the two loss terms, in every phase
VAE, PLAIN = GREEN, BLUE               # the VAE and the same network without a KL term
PRIOR = DIGIT_INK
TEN_RGB = np.array(TEN, np.float32)
MEASURED = "measured on MNIST, held-out digits"       # the evidence tags, bottom right
EXACT = "exact arithmetic on measured codes"
EDGE = tuple(float(v) for v in (np.asarray(PAPER) * 0.45 + np.asarray(AXIS) * 0.55))


# ====================================================================== fast scatters
def scatter(z, inks, lim, side, r=2.3, scale=2, back=None):
    """``z`` ``(n, 2)`` as round dots of radius ``r`` px in the colours ``inks``
    ``(n, 3)``, as one ``side * scale`` square texture of the window ``lim`` (y up):
    later points lie on top, each dot has a darker rim and a soft edge. ``back``: the
    texture to draw on (paper)."""
    n = int(round(side * scale))
    tex = (np.empty((n, n, 3), np.float32) if back is None
           else np.array(back, np.float32, copy=True))
    if back is None:
        tex[:] = PAPER
    z = np.asarray(z, float)
    x0, x1, y0, y1 = lim
    cx = (z[:, 0] - x0) / (x1 - x0) * n
    cy = (1.0 - (z[:, 1] - y0) / (y1 - y0)) * n
    R = r * scale
    m = int(np.ceil(R + 1))
    idx = np.flatnonzero((cx > -m) & (cx < n + m) & (cy > -m) & (cy < n + m))
    if not len(idx):
        return tex
    off = np.arange(-m, m + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(off, off))
    col = np.floor(cx[idx]).astype(int)[:, None] + dx[None]
    row = np.floor(cy[idx]).astype(int)[:, None] + dy[None]
    who = np.broadcast_to(idx[:, None], col.shape)
    ok = ((col >= 0) & (col < n) & (row >= 0) & (row < n)
          & ((col + 0.5 - cx[idx][:, None]) ** 2 + (row + 0.5 - cy[idx][:, None]) ** 2
             <= (R + 0.5) ** 2))
    owner = np.full(n * n, -1, np.int64)
    owner[(row * n + col)[ok]] = who[ok]                     # in point order: the last one stays
    flat = np.flatnonzero(owner >= 0)
    o = owner[flat]
    dist = np.hypot(flat % n + 0.5 - cx[o], flat // n + 0.5 - cy[o])
    cover = np.clip(R + 0.5 - dist, 0.0, 1.0)[:, None]
    rim = np.clip(dist - (R - 1.6), 0.0, 1.0)[:, None]
    col = np.asarray(inks, np.float32)[o] * (1.0 - 0.32 * rim)
    t = tex.reshape(-1, 3)
    t[flat] = t[flat] * (1.0 - cover) + col * cover
    return tex


def regions(mu, lim, side, scale=2, ink=EDGE):
    """The borders between the nearest-centre regions of the centres ``mu`` (a Voronoi
    diagram), as a paper texture for :func:`scatter` to draw on."""
    n = int(round(side * scale))
    h = n // 2
    x0, x1, y0, y1 = lim
    gx = x0 + (np.arange(h) + 0.5) / h * (x1 - x0)
    gy = y1 - (np.arange(h) + 0.5) / h * (y1 - y0)
    d2 = ((gx[None, :, None] - mu[None, None, :, 0]) ** 2
          + (gy[:, None, None] - mu[None, None, :, 1]) ** 2)
    lab = d2.argmin(-1)
    edge = np.zeros((h, h), bool)
    edge[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    edge[:-1] |= lab[:-1] != lab[1:]
    tex = np.empty((h, h, 3), np.float32)
    tex[:] = PAPER
    tex[edge] = ink
    return np.repeat(np.repeat(tex, 2, 0), 2, 1)[:n, :n] if n == 2 * h else np.pad(
        np.repeat(np.repeat(tex, 2, 0), 2, 1), ((0, n - 2 * h), (0, n - 2 * h), (0, 0)),
        mode="edge")


def stacked(values, labels, lo, hi, bins, rows, full):
    """A histogram of ``values`` stacked by digit, as a ``(rows, bins, 3)`` texture in
    the digit colours: ``full`` codes in one bin fill the height."""
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.array([np.histogram(values[labels == k], edges)[0] for k in range(10)])
    top = np.cumsum(counts, 0) / float(full) * rows       # digit, bin: top of its band
    level = rows - 1 - np.arange(rows)[:, None]           # texture row -> height from the base
    tex = np.empty((rows, bins, 3), np.float32)
    tex[:] = PAPER
    for k in range(9, -1, -1):
        tex[np.broadcast_to(level < top[k][None], (rows, bins))] = TEN_RGB[k]
    return tex


def normal(v, mu=0.0, sd=1.0):
    return np.exp(-0.5 * ((np.asarray(v, float) - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


class Clouds(Families):
    """What the two animations share: a cloud of coloured codes as one texture, the
    digit legend, an ellipse, a chart over evenly spaced checkpoints."""

    def evidence(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def cloud(self, sc, pl, z, op, r=2.3, back=None, labels=None):
        labels = self.labels if labels is None else labels
        pl.image(sc, scatter(z, TEN_RGB[labels], pl.lim, pl.side, r, back=back), op)

    def legend(self, sc, x, y, op, step=0.0295, head="digit"):
        if head:
            self.text(sc, x - 0.02, y, head, op, anchor="rm")
        for digit in range(10):
            self.box(sc, x + step * digit, y, 22, 22, TEN[digit], op, str(digit), ink=WHITE,
                     size=0.03)

    def oval(self, sc, pl, centre, rx, ry, ink, op, width=1.6, n=25):
        a = np.linspace(0.0, 2 * np.pi, n)
        pl.path(sc, np.column_stack((centre[0] + rx * np.cos(a), centre[1] + ry * np.sin(a))),
                ink, op, width)

    def rings(self, sc, pl, op, ink=PRIOR, width=1.7):
        with self.over(1.2):
            for radius in (1, 2, 3):
                self.ring(sc, pl, radius, ink, op, width=width)

    def marker(self, sc, at, ink, op, r=6.0):
        """A bold centre: white, ringed in ``ink``."""
        self.disk(sc, self.P(*at, lift=1.6), r, WHITE, op, ring=ink, ring_w=2.6, lift=2.2,
                  outline=0.0)

    def run_chart(self, sc, box, n, values, op, ylabel, xticks=(), ylog=False, yticks=None,
                  xlabel=""):
        chart = self.chart(box, [0, n - 1], values, ylog=ylog)
        self.axes(sc, chart, op, xlabel, ylabel, xticks=list(xticks), yticks=yticks)
        return chart

    def seeds(self, sc, chart, ys, ink, op, u, width=2.1):
        """The mean of ``ys`` (seed, checkpoint) as a curve, each seed thin behind it."""
        ys = np.atleast_2d(ys)
        xs = np.arange(ys.shape[1])
        if len(ys) > 1:
            for one in ys:
                self.plot(sc, chart, xs, one, soft(ink, 0.45), op, u, width=0.7, dot=False)
        return self.plot(sc, chart, xs, ys.mean(0), ink, op, u, width=width)


# ====================================================================== 170 the KL term
def when(k):
    """Checkpoint ``k`` of the VAE runs in words."""
    step = S.CHECKS[k]
    if step < S.EPOCH:
        return "update %d" % step
    passes = step // S.EPOCH
    return "%d pass%s (%s updates)" % (passes, "" if passes == 1 else "es", f"{step:,}")


class VaeKlTerm(Clouds):
    """One digit through a VAE; the KL term as a distance between two bells; the two
    terms pulling against each other; the cloud of held-out codes through training
    against the prior's rings; the same without the KL term; three weights of it."""

    TITLE = "The KL term of a VAE"
    PHASES = (("one digit", 8.0), ("KL term", 10.0), ("tug of war", 8.0), ("training", 19.0),
              ("no KL term", 7.5), ("its weight", 9.0))
    LIM = (-4.2, 4.2, -4.2, 4.2)
    OVALS = 40                         # blobs drawn at their true width

    def setup(self):
        d, m = S.kl_arrays(), self.study.metadata
        self.m, self.t = m, m["table"]
        drawn = S.SEEDS.index(S.DRAWN)
        self.labels = np.asarray(d["labels"])
        self.mu = {run: np.asarray(d[run + "_mu"], float)[drawn] for run in S.RUNS}
        self.sd = {run: np.exp(0.5 * np.asarray(d[run + "_lv"], float)[drawn])
                   for run in S.RUNS}
        self.n = len(S.CHECKS)
        # the curves: seed, checkpoint
        self.bce = np.asarray(d["vae_bce"], float)
        self.kl = np.asarray(d["vae_kl"], float)                    # seed, checkpoint, number
        self.fit = {run: np.asarray(d[run + "_g_fit"], float) for run in S.RUNS}
        self.ring_share = np.asarray(d["vae_g_rings"], float)[drawn]
        self.ks = np.asarray(d["vae_g_ks"], float)[drawn].mean(1)
        self.track = {run: np.asarray(d[run + "_track"], float)[drawn] for run in S.RUNS}
        # the tracked digit through the trained VAE
        P = S.kl_params("vae")
        x = F.data()["xt"]
        self.x0, self.y0 = x[S.TRACK], int(self.labels[S.TRACK])
        f = dm.vae_forward(P, self.x0[None], np.zeros((1, 2)))
        self.mu0, self.sd0 = f["mu"][0], np.exp(0.5 * f["lv"][0])
        self.eps = np.random.default_rng(5).normal(size=(10, 2))
        self.draws = self.mu0 + self.sd0 * self.eps
        self.outs = dm.vae_decode(P, self.draws)[2]
        r = np.clip(self.outs, 1e-7, 1 - 1e-7)
        self.rec0 = -(self.x0 * np.log(r) + (1 - self.x0) * np.log(1 - r)).sum(1)
        self.pieces0 = S.kl_pieces(self.mu0, self.sd0)              # (to the middle, width)
        self.kl0 = float(sum(p.sum() for p in self.pieces0))
        self.j0 = int(np.argmax(np.abs(self.mu0)))                  # the code number drawn in 1-D
        # both terms with every blob slid to the middle, or widened
        scan = m["pull_scan"]
        self.scan_s, self.scan_w = np.array(scan["centre_scale"]), np.array(scan["width_scale"])
        self.scan_c, self.scan_wd = np.array(scan["centre"]), np.array(scan["width"])
        self.blobs = np.arange(0, len(self.labels), len(self.labels) // self.OVALS)
        # the window that holds the plain autoencoder's cloud and the rings
        z = self.mu["ae"][-1]
        lo = np.minimum(np.percentile(z, 1.0, axis=0), -3.2)
        hi = np.maximum(np.percentile(z, 99.0, axis=0), 3.2)
        mid, half = (lo + hi) / 2, float((hi - lo).max()) * 0.52
        self.wide = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)

    def terms(self, centre=1.0, width=1.0):
        """``(reconstruction, KL)`` from the scan, at a centre scale or a width scale."""
        if width != 1.0:
            return tuple(float(np.interp(np.log(width), np.log(self.scan_w), self.scan_wd[:, j]))
                         for j in (0, 1))
        return tuple(float(np.interp(centre, self.scan_s, self.scan_c[:, j])) for j in (0, 1))

    def subtitle_for(self, k):
        t, m = self.t, self.m
        c = m["vae_curves"]
        peak = int(np.argmax(c["fit"]))
        home, mid, fat = self.terms(), self.terms(centre=0.0), self.terms(width=8.0)
        return [
            "a VAE's encoder gives each digit a small blob in code space, not a point: a "
            "centre μ and a width σ per code number. A code is drawn from the blob and "
            "decoded; the loss adds a KL term to the rebuilding error",
            "the KL term compares two bells, per code number: this digit's blob "
            "N(μ, σ²) and the prior N(0, 1). It is ½(μ² + σ² − ln σ² − 1): zero only "
            "when μ = 0 and σ = 1; here %.1f nats for the digit" % self.kl0,
            "measured on the trained VAE: slide every blob to the middle and the KL term "
            "falls %.1f to %.1f nats but rebuilding costs %.0f, not %.0f; widen them 8 times: "
            "KL %.1f, rebuilding %.0f. Their sum is lowest as trained" % (
                home[1], mid[1], mid[0], home[0], fat[1], fat[0]),
            "2,000 held-out digits at 14 checkpoints: the blobs burst apart in the first "
            "%d updates (distance from the bell %.1f), then settle into its rings: %.2f "
            "after 20 passes, with %s / %s / %s inside (a bell: 39 / 86 / 99%%)" % (
                S.CHECKS[peak], c["fit"][peak], c["fit"][-1],
                *(pct(v) for v in t["vae"]["rings"])),
            "the same network, start and batches, without the KL term or the noise: its "
            "cloud keeps spreading (distance %.1f; the VAE %.2f), yet it rebuilds held-out "
            "digits no better: %.0f against %.0f nats; three seeds each" % (
                t["ae"]["fit"]["mean"], t["vae"]["fit"]["mean"],
                t["ae"]["reconstruction"]["mean"], t["vae"]["reconstruction"]["mean"]),
            "the weight on the KL term: at 0.1 the cloud is %.0f times further from the "
            "bell; at 10 the blobs widen from σ %.2f to %.2f and overlap, and "
            "rebuilding costs %.0f more nats: the digits blur" % (
                t["low"]["fit"]["mean"] / t["vae"]["fit"]["mean"], t["vae"]["spread"]["mean"],
                t["high"]["spread"]["mean"],
                t["high"]["reconstruction"]["mean"] - t["vae"]["reconstruction"]["mean"]),
        ][k]

    # ------------------------------------------------------------------ one digit
    def phase_1(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        y = 0.47
        k = int(u * 24) % len(self.draws)
        ink = TEN[self.y0]
        self.digit(sc, self.x0, 0.095, y, 78, op, frame=AXIS)
        self.text(sc, 0.095, y + 0.105, "a held-out %d" % self.y0, op)
        self.net(sc, 0.225, y, VAE, "encoder", op, w=100.0, h=44.0)
        self.link(sc, (0.14, y), (0.17, y), VAE, op, pulse=seg(u, 0.02, 0.14))
        a = op * smoothstep(0.12, 0.2, u)
        self.lines(sc, 0.15, y - 0.16, [
            ("centre μ = (%.2f, %.2f)" % tuple(self.mu0), None),
            ("width σ = (%.2f, %.2f)" % tuple(self.sd0), None)], a, step=0.052, size=0.033)
        # the code space with the prior's rings, and the blob close up
        pl = Plane(self, 0.50, y + 0.06, 216.0, (-3.3, 3.3, -3.3, 3.3))
        pl.frame(sc, op, "code space", below="rings: the prior N(0, 1) at 1, 2, 3")
        self.rings(sc, pl, op * 0.8, AXIS, 1.3)
        with self.over(1.3):
            self.oval(sc, pl, self.mu0, 2 * self.sd0[0], 2 * self.sd0[1], ink, a, 2.4, n=13)
        self.link(sc, (0.28, y), pl.at(self.mu0), VAE, a, seg(u, 0.12, 0.24), bend=-0.03)
        half = 4.0 * float(self.sd0.max())
        zoom = Plane(self, 0.71, y - 0.06, 108.0, (self.mu0[0] - half, self.mu0[0] + half,
                                                    self.mu0[1] - half, self.mu0[1] + half))
        b = op * smoothstep(0.22, 0.32, u)
        zoom.frame(sc, b, "the blob, close up")
        with self.over(1.2):
            for s in (1, 2):
                self.oval(sc, zoom, self.mu0, s * self.sd0[0], s * self.sd0[1], ink, b, 1.8)
        c = op * smoothstep(0.34, 0.42, u)
        for j, p in enumerate(self.draws[:k + 1] if u > 0.34 else ()):
            if zoom.inside(p):
                self.disk(sc, self.P(*zoom.at(p), lift=1.5), 3.2 if j < k else 5.0,
                          soft(VAE, 0.4) if j < k else VAE, c, lift=2.0, outline=0.2)
        self.lines(sc, 0.71, y + 0.09, ["a code is drawn:", "z = μ + σ × noise",
                                         "= (%.2f, %.2f)" % tuple(self.draws[k])],
                   c, step=0.042, anchor="mm", size=0.031)
        self.net(sc, 0.865, y - 0.06, VAE, "decoder", c, w=80.0, h=40.0)
        self.link(sc, (0.772, y - 0.06), (0.818, y - 0.06), VAE, c, pulse=(u * 24) % 1.0)
        self.digit(sc, self.outs[k], 0.865, y + 0.11, 74, c, frame=VAE)
        self.link(sc, (0.865, y - 0.02), (0.865, y + 0.03), VAE, c)
        e = op * smoothstep(0.55, 0.68, u)
        self.text(sc, 0.335, 0.875, "loss  =", e, anchor="rm", strong=True, size=0.037)
        self.text(sc, 0.35, 0.875, "reconstruction %.0f" % self.rec0[k], e, anchor="lm",
                  ink=dark(REC, 0.3), size=0.037)
        self.text(sc, 0.585, 0.875, "+   KL term %.1f   nats" % self.kl0, e, anchor="lm",
                  ink=dark(KLC, 0.25), size=0.037)

    # ------------------------------------------------------------------ the KL term
    def _slide(self, u):
        """The blob's ``(mu, sd)`` at ``u``: the digit's own, then mu slid to 0, then
        sd slid to 1, then back."""
        mu, sd = float(self.mu0[self.j0]), float(self.sd0[self.j0])
        a = smoothstep(0.38, 0.56, u) * (1 - smoothstep(0.9, 0.99, u))
        b = smoothstep(0.62, 0.84, u) * (1 - smoothstep(0.9, 0.99, u))
        return mu * (1 - a), float(np.exp(np.log(sd) * (1 - b)))

    def phase_2(self, sc, u, op):
        self.evidence(sc, EXACT, op)
        mu, sd = self._slide(u)
        top = 1.6
        chart = self.chart(((0.085, 0.49), (0.40, 0.73)), [-3.5, 3.5], [0.0, top], pad=0.0)
        self.axes(sc, chart, op, "code number %d" % (self.j0 + 1), "probability density",
                  xticks=[(v, "%d" % v) for v in (-3, -2, -1, 0, 1, 2, 3)],
                  yticks=[(0.0, "0"), (0.4, "0.4"), (0.8, "0.8"), (1.2, "1.2")])
        xs = np.linspace(-3.5, 3.5, 281)
        self.plot(sc, chart, xs, normal(xs), PRIOR, op, 1.0, width=2.0, dot=False)
        q = normal(xs, mu, sd)
        show = q <= top
        show[1:] |= show[:-1].copy()
        show[:-1] |= show[1:].copy()
        qa = op * smoothstep(0.06, 0.16, u)
        for lo, hi in _runs(show):
            self.plot(sc, chart, xs[lo:hi], np.minimum(q[lo:hi], top), VAE, qa, 1.0, width=2.4,
                      dot=False)
        peak = float(normal(mu, mu, sd))
        if peak > top:
            self.text(sc, chart.fx(mu) + 0.014, 0.375, "peak %.1f, cut off" % peak, qa,
                      anchor="lm", ink=dark(VAE, 0.3))
        self.text(sc, 0.085, 0.845, "black: the prior p(z) = N(0, 1)", op, anchor="lm",
                  ink=PRIOR)
        self.text(sc, 0.085, 0.32, "this digit's blob  q(z | x) = N(μ, σ²)", qa, anchor="lm",
                  ink=dark(VAE, 0.3), size=0.035)
        self.text(sc, 0.085, 0.90, "the held-out %d's own blob" % self.y0 if u < 0.38
                  else "sliding μ to 0" if u < 0.6 else
                  "sliding σ to 1: the two bells coincide" if u < 0.93
                  else "back to the digit's own blob", op, anchor="lm", size=0.033)
        # the formula, with this blob's numbers
        x = 0.56
        mid, wide = S.kl_pieces(mu, sd)
        f = op * smoothstep(0.1, 0.2, u)
        self.text(sc, x, 0.33, "KL( q || p )  =  ½ ( μ²  +  σ² − ln σ² − 1 )", f,
                  anchor="lm", strong=True, size=0.037)
        self.text(sc, x, 0.385, "per code number, in nats", f, anchor="lm")
        self.text(sc, x, 0.455, "μ = %.2f      σ = %.2f" % (mu, sd), f, anchor="lm",
                  size=0.036, ink=dark(VAE, 0.3))
        g = op * smoothstep(0.18, 0.28, u)
        h = op * smoothstep(0.24, 0.34, u)
        scale = 46.0
        self.text(sc, x, 0.535, "½ μ²  =  %.2f" % mid, g, anchor="lm", size=0.035)
        self.hbar(sc, x + 0.285, 0.535, scale * float(mid), KLC, g, thick=14.0)
        self.text(sc, x, 0.58, "pulls the blob to the middle (0 at μ = 0)", g, anchor="lm",
                  ink=dark(KLC, 0.25))
        self.text(sc, x, 0.655, "½ (σ² − ln σ² − 1)  =  %.2f" % wide, h, anchor="lm",
                  size=0.035)
        self.hbar(sc, x + 0.285, 0.655, scale * float(wide), KLC, h, thick=14.0)
        self.text(sc, x, 0.70, "pulls its width to 1 (0 only at σ = 1)", h, anchor="lm",
                  ink=dark(KLC, 0.25))
        k = op * smoothstep(0.3, 0.38, u)
        self.text(sc, x, 0.78, "KL  =  %.2f nats" % (mid + wide), k, anchor="lm", strong=True,
                  size=0.038)
        own = [float(self.pieces0[0][j] + self.pieces0[1][j]) for j in (0, 1)]
        self.text(sc, x, 0.86, "this digit, both code numbers: %.2f + %.2f = %.1f nats" % (
            own[0], own[1], self.kl0), k, anchor="lm")

    # ------------------------------------------------------------------ the tug of war
    def _pulled(self, u):
        """``(centre scale, width scale)`` of every blob at ``u``: slid to the middle
        and back, then widened 8 times and back."""
        s = 1.0 - smoothstep(0.1, 0.34, u) * (1.0 - smoothstep(0.4, 0.52, u))
        w = 8.0 ** (smoothstep(0.56, 0.78, u) * (1.0 - smoothstep(0.84, 0.95, u)))
        return float(s), float(w)

    def phase_3(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        s, w = self._pulled(u)
        mu, sd = self.mu["vae"][-1] * s, self.sd["vae"][-1] * w
        pl = Plane(self, 0.205, 0.585, 268.0, (-3.5, 3.5, -3.5, 3.5))
        pl.frame(sc, op, "held-out digits: blob centres")
        self.cloud(sc, pl, mu, op, r=2.0)
        self.rings(sc, pl, op * 0.8, PRIOR, 1.3)
        with self.over(1.6):
            for j in self.blobs:
                self.oval(sc, pl, mu[j], sd[j, 0], sd[j, 1], DIGIT_INK, op, 1.2, n=13)
        self.legend(sc, 0.075, 0.895, op, head="")
        rec, kl = self.terms(centre=s, width=w)
        home = self.terms()
        x = 0.41
        self.text(sc, x, 0.32, "every blob's centre × %.2f" % s if w == 1.0 and u < 0.54
                  else "every blob's width × %.1f" % w, op, anchor="lm", strong=True,
                  size=0.036)
        self.text(sc, x, 0.365, "ovals: %d of the blobs at their true width" % len(self.blobs),
                  op, anchor="lm")
        # the sum of the two terms along each slide
        lo, hi = 140.0, 340.0
        c1 = self.chart(((x + 0.035, x + 0.235), (0.46, 0.66)), [0.0, 1.5], [lo, hi], pad=0.0)
        self.axes(sc, c1, op, "blob centres ×", "the two terms added (nats)",
                  xticks=[(0, "0"), (0.5, "0.5"), (1, "1"), (1.5, "1.5")],
                  yticks=[(150, "150"), (250, "250")])
        tot = self.scan_c.sum(1)
        self.plot(sc, c1, self.scan_s, tot, DIGIT_INK, op, 1.0, width=1.8, dot=False)
        self.disk(sc, self.P(c1.fx(s), c1.fy(float(np.interp(s, self.scan_s, tot))), lift=1.4),
                  6.0, KLC, op, glow=0.4, lift=1.8, outline=0.0)
        c2 = self.chart(((x + 0.325, x + 0.525), (0.46, 0.66)), [0.5, 8.0], [lo, hi], pad=0.0,
                        xlog=True)
        self.axes(sc, c2, op, "blob widths ×", "", xticks=[(0.5, "½"), (1, "1"), (2, "2"),
                                                             (4, "4"), (8, "8")],
                  yticks=[(150, "150"), (250, "250")])
        totw = self.scan_wd.sum(1)
        self.plot(sc, c2, self.scan_w, totw, DIGIT_INK, op, 1.0, width=1.8, dot=False)
        self.disk(sc, self.P(c2.fx(w), c2.fy(float(np.interp(np.log(w), np.log(self.scan_w),
                                                              totw))), lift=1.4),
                  6.0, KLC, op, glow=0.4, lift=1.8, outline=0.0)
        y = 0.80
        self.text(sc, x, y, "KL term  %.1f" % kl, op, anchor="lm", size=0.036,
                  ink=dark(KLC, 0.25))
        self.text(sc, x + 0.165, y, "+  reconstruction  %.0f" % rec, op, anchor="lm",
                  size=0.036, ink=dark(REC, 0.3))
        self.text(sc, x + 0.445, y, "=  %.0f" % (rec + kl), op, anchor="lm", size=0.036,
                  strong=True)
        self.text(sc, x, y + 0.055, "as trained:  %.1f  +  %.0f  =  %.0f, the lowest point "
                  "of both curves" % (home[1], home[0], sum(home)), op, anchor="lm")

    # ------------------------------------------------------------------ training
    XT = ((0, "0"), (3, "20"), (6, "156"), (9, "780"), (13, "3,120"))

    def _marginal(self, sc, values, fx, fy, w, h, op, label):
        full = 2.4 * len(values) * 0.2 * normal(0.0)
        c = self.P(fx, fy)
        self.rect(sc, c, w + 6, h + 6, PAPER, op, border=AXIS, border_w=1.0, radius=4, lift=0.1)
        dw, dh = self.frac(w, h)
        draw_image(self, sc, c, stacked(values, self.labels, self.LIM[0], self.LIM[1], 42, 40,
                                        full), w, h, op, grid=0)
        s = np.linspace(self.LIM[0], self.LIM[1], 43)
        pts = [self.P(fx + dw * (v / (self.LIM[1] - self.LIM[0])),
                      fy + dh / 2 - dh * min(normal(v) / (2.4 * normal(0.0)), 1.0), lift=1.0)
               for v in s]
        self.polyline(sc, pts, 2.0, PRIOR, op)
        self.text(sc, fx - dw / 2, fy - dh / 2 - 0.028, label, op, anchor="lm")

    def phase_4(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        n = self.n
        at = seg(u, 0.05, 0.92) * (n - 1)
        k = int(round(at))
        prog = at / (n - 1)
        pl = Plane(self, 0.205, 0.585, 268.0, self.LIM)
        pl.frame(sc, op, "held-out digits: blob centres")
        z = self.between(self.mu["vae"], at)
        self.cloud(sc, pl, z, op, r=2.1)
        self.rings(sc, pl, op, PRIOR, 1.6)
        self.legend(sc, 0.075, 0.895, op, head="")
        # the checkpoint, the marginals against the bell, the rings
        x = 0.385
        self.text(sc, x, 0.305, when(k), op, anchor="lm", strong=True, size=0.036)
        self._marginal(sc, z[:, 0], x + 0.098, 0.45, 180.0, 54.0, op, "code number 1")
        self._marginal(sc, z[:, 1], x + 0.098, 0.625, 180.0, 54.0, op, "code number 2")
        self.text(sc, x, 0.71, "line: the bell N(0, 1)", op, anchor="lm")
        share = self.ring_share[k]
        self.lines(sc, x, 0.757, [
            ("inside rings 1, 2, 3:", None),
            ("%s   %s   %s" % tuple(pct(v) for v in share), layout.STRONG_INK),
            ("a bell: 39%   86%   99%", None),
            ("gap to the bell (KS): %.2f" % self.ks[k], None)], op, step=0.043, size=0.031)
        # the curves, filling in as the cloud moves
        x0, x1 = 0.685, 0.93
        c1 = self.run_chart(sc, ((x0, x1), (0.345, 0.445)), n, [130.0, 720.0], op,
                            "reconstruction term: %.0f nats" % self.bce[:, k].mean(), ylog=True,
                            yticks=[(150, "150"), (300, "300"), (600, "600")])
        self.seeds(sc, c1, self.bce, REC, op, prog)
        tot = self.kl.sum(2)
        c2 = self.run_chart(sc, ((x0, x1), (0.53, 0.63)), n, [0.0, 6.5], op,
                            "KL term: %.1f nats (thin: per number)" % tot[:, k].mean(),
                            yticks=[(0, "0"), (3, "3"), (6, "6")])
        xs = np.arange(n)
        for j in (0, 1):
            self.plot(sc, c2, xs, self.kl[:, :, j].mean(0), soft(KLC, 0.35), op, prog,
                      width=0.9, dot=False)
        self.seeds(sc, c2, tot, KLC, op, prog)
        c3 = self.run_chart(sc, ((x0, x1), (0.715, 0.815)), n, [0.0, 5.0], op,
                            "cloud's distance from the bell: %.2f" % self.fit["vae"][:, k].mean(),
                            xticks=self.XT, yticks=[(0, "0"), (2, "2"), (4, "4")],
                            xlabel="updates (checkpoints evenly spaced)")
        self.seeds(sc, c3, self.fit["vae"], VAE, op, prog)

    # ------------------------------------------------------------------ no KL term
    def phase_5(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        n = self.n
        at = seg(u, 0.06, 0.72) * (n - 1)
        k = int(round(at))
        for i, (run, ink, name) in enumerate((("vae", VAE, "VAE: with the KL term"),
                                              ("ae", PLAIN, "the same network, no KL term"))):
            pl = Plane(self, 0.175 + 0.27 * i, 0.555, 222.0, self.wide)
            pl.frame(sc, op, name, ink=dark(ink, 0.3))
            self.cloud(sc, pl, self.between(self.mu[run], at), op, r=1.9)
            self.rings(sc, pl, op, PRIOR, 1.3)
            self.text(sc, pl.fx, 0.805, "distance from the bell: %.2f"
                      % self.fit[run][:, k].mean(), op, strong=True)
        self.text(sc, 0.31, 0.85, "%s; the same window and rings in both" % when(k), op)
        self.legend(sc, 0.177, 0.905, op, head="")
        both = np.concatenate([self.fit["vae"], self.fit["ae"]])
        chart = self.run_chart(sc, ((0.68, 0.93), (0.36, 0.69)), n, [0.07, max(both.max(), 12.0)],
                               op, "distance of the cloud from the bell", xticks=self.XT,
                               ylog=True, yticks=[(0.1, "0.1"), (1, "1"), (10, "10")],
                               xlabel="updates (checkpoints evenly spaced)")
        prog = at / (n - 1)
        for run, ink in (("ae", PLAIN), ("vae", VAE)):
            self.seeds(sc, chart, self.fit[run], ink, op, prog)
        self.text(sc, 0.66, 0.795, "blue: no KL term", op, anchor="lm", ink=dark(PLAIN, 0.2))
        self.text(sc, 0.81, 0.795, "green: the VAE", op, anchor="lm", ink=dark(VAE, 0.3))
        self.lines(sc, 0.66, 0.837, ["distance: KL of the cloud's best-fit",
                                     "Gaussian from N(0, I); three seeds"], op, step=0.042)

    # ------------------------------------------------------------------ the weight
    def phase_6(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        lim = (-5.0, 5.0, -5.0, 5.0)
        t = self.t
        for i, run in enumerate(("low", "vae", "high")):
            a = op * smoothstep(0.03 + 0.14 * i, 0.15 + 0.14 * i, u)
            pl = Plane(self, 0.145 + 0.225 * i, 0.515, 180.0, lim)
            pl.frame(sc, a, "KL weight %g" % S.BETA[run], ink=dark(KLC, 0.25))
            mu, sd = self.mu[run][-1], self.sd[run][-1]
            self.cloud(sc, pl, mu, a, r=1.7)
            self.rings(sc, pl, a, PRIOR, 1.2)
            with self.over(1.6):
                for j in self.blobs:
                    self.oval(sc, pl, mu[j], sd[j, 0], sd[j, 1], DIGIT_INK, a, 1.2, n=13)
            self.lines(sc, pl.fx, 0.725, [
                ("reconstruction %.0f" % t[run]["reconstruction"]["mean"], dark(REC, 0.3)),
                ("KL term %.1f" % t[run]["kl"]["mean"], dark(KLC, 0.25)),
                ("blob width σ %.2f" % t[run]["spread"]["mean"], None),
                ("from the bell %.2f" % t[run]["fit"]["mean"], None)], a, step=0.04,
                anchor="mm", size=0.031)
        self.legend(sc, 0.237, 0.905, op, head="")
        x = 0.80
        self.text(sc, x + 0.06, 0.315, "the held-out %d, rebuilt" % self.y0, op, strong=True)
        self.digit(sc, self.x0, x, 0.40, 50, op, frame=AXIS)
        self.text(sc, x + 0.045, 0.40, "the original", op, anchor="lm")
        for i, run in enumerate(("low", "vae", "high")):
            a = op * smoothstep(0.3 + 0.14 * i, 0.42 + 0.14 * i, u)
            y = 0.515 + 0.115 * i
            self.digit(sc, self.track[run][-1], x, y, 50, a, frame=KLC)
            self.lines(sc, x + 0.045, y - 0.02, ["weight %g" % S.BETA[run],
                                                 "sharpness %.2f" % t[run]["sharpness"]["mean"]],
                       a, step=0.04, size=0.031)
        self.lines(sc, 0.74, 0.84, ["ovals: %d blobs' true width" % len(self.blobs),
                                    "numbers: mean of three seeds"],
                   op * smoothstep(0.6, 0.7, u), step=0.04)


def _runs(mask):
    """``(start, stop)`` of each run of True in ``mask``."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(m[1:] != m[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2]) if b - a > 1]


# ====================================================================== 171 DEC's centres
QC, PC = soft(AMBER, 0.25), dark(AMBER, 0.42)      # the soft assignment q, the target p
PURE, SEP, MOVED = VIOLET, AMBER, TEAL             # label purity, separation, movement
TRAIL = dark(AMBER, 0.3)


def f2(v):
    return "%.2f" % float(v)


class DecCentroids(Clouds):
    """An autoencoder's code space; k-means placing ten centres in it; DEC's soft
    assignment and sharpened target on three real codes; the codes and the centres
    through DEC training; separated against pure; what the outcome depends on."""

    TITLE = "DEC: the centroids move"
    PHASES = (("codes", 5.5), ("k-means", 11.0), ("q and p", 11.5), ("training", 19.0),
              ("separated?", 8.0), ("starts", 6.5))
    XT = ((0, "0"), (4, "80"), (6, "300"), (8, "800"), (10, "1,500"))
    STEPWISE = 5                       # k-means iterations shown one at a time
    # the three codes of "q and p": the held-out 4 whose top q is nearest 0.36 (unsure),
    # the 3 nearest 0.6, the 0 nearest 0.9 (sure)
    PICKS = ((4, 0.36), (3, 0.6), (0, 0.9))

    def setup(self):
        d, m = S.dec_arrays(), self.study.metadata
        self.m, self.t = m, m["table"]["c2"]
        self.labels = np.asarray(d["labels"])
        run = self.run = S.dec_run("dec")
        self.idec = S.dec_run("idec")
        st = self.st = S.dec_start()
        self.z, self.cen, self.km = run["z"], run["mu"], st["km"]
        self.n = len(S.DEC_CHECKS)
        pts = np.concatenate([self.z.reshape(-1, 2), self.idec["z"].reshape(-1, 2)])
        lo, hi = np.percentile(pts, [0.4, 99.6], axis=0)
        mid, half = (lo + hi) / 2, float((hi - lo).max()) * 0.53
        self.lim = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
        self.by = {kind: {k: np.array([S.dec_run(kind, S.CODE, s)[k] for s in S.SEEDS])
                          for k in S.MEASURES} for kind in S.KINDS}
        # the tracked digit, for the first picture
        self.x0 = F.data()["xt"][S.TRACK]
        # three real codes at the start of DEC: unsure, in between, sure
        z0, mu0 = self.z[0], self.cen[0]
        q, kern = dm.soft_assign(z0, mu0)
        self.freq = run["freq0"]
        w = q ** 2 / self.freq
        p = w / w.sum(1, keepdims=True)
        self.pick = [int(np.argmin(np.abs(q.max(1) - v) + 9.0 * (self.labels != digit)))
                     for digit, v in self.PICKS]
        self.q3, self.p3 = q[self.pick], p[self.pick]
        a = self.pick[0]
        self.top2 = np.argsort(-q[a])[:2]
        self.d2 = ((z0[a] - mu0[self.top2]) ** 2).sum(1)
        self.kern2, self.kern_sum = kern[a][self.top2], float(kern[a].sum())
        self.kl_a = float((p[a] * np.log(p[a] / q[a])).sum())
        # which digits share a cluster at the end, rows ordered by their largest digit
        conf = run["conf"][-1]
        order = np.lexsort((-conf.max(1), conf.argmax(1)))
        self.conf = conf[order]
        pairs = sorted({tuple(sorted(g[:2])) for g in m["shared_digits"] if len(g) >= 2})
        self.shared = ", ".join("%d and %d" % g for g in pairs)
        self.shared_short = "   ".join("%d+%d" % g for g in pairs)

    def subtitle_for(self, k):
        t, m = self.t, self.m
        d, i = t["dec"], t["idec"]
        st, wide = m["starts"], m["table"]["c10"]["dec"]
        return [
            "an autoencoder trained only to rebuild digits gives each one a code of 2 "
            "numbers. Here are 2,000 held-out digits in that code space, coloured by a "
            "label the model never sees",
            "k-means: start from ten codes, then give every code to its nearest centre "
            "and move each centre to the mean of its codes, %d times. Matched to digits, "
            "its clusters get %s of held-out digits right" % (
                S.KM_ITERS, pct(self.st["km_acc"][-1])),
            "DEC makes the assignment soft: q falls with distance from each centre. Its "
            "target p squares q and divides by cluster size: sure codes get surer, big "
            "clusters are held back. The loss is the KL divergence of Q from P",
            "%s DEC updates move the encoder and the centres. Three seeds: silhouette "
            "%.2f to %.2f and top assignment %.2f to %.2f, while cluster accuracy only "
            "goes %.2f to %.2f" % (
                f"{F.CLUSTER_STEPS:,}", d["sil"]["start"]["mean"], d["sil"]["end"]["mean"],
                d["maxq"]["start"]["mean"], d["maxq"]["end"]["mean"],
                d["acc"]["start"]["mean"], d["acc"]["end"]["mean"]),
            "well separated is not the same as right: the clusters end tight and apart, "
            "yet %s still share one. DEC sharpens the clusters k-means found, its "
            "mistakes included" % self.shared,
            "keeping the decoder (IDEC) holds rebuilding error at %.3f where DEC's rises "
            "to %.3f; a 10-number code ends at %.2f; five k-means starts end at %.2f to "
            "%.2f, three autoencoder seeds at %.2f to %.2f" % (
                i["mse"]["end"]["mean"], d["mse"]["end"]["mean"], wide["acc"]["end"]["mean"],
                min(st["accuracy_end"]), max(st["accuracy_end"]),
                d["acc"]["end"]["min"], d["acc"]["end"]["max"]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def centres(self, sc, pl, mu, op, ink=AMBER, r=6.0):
        for c in mu[pl.inside(mu)]:
            self.marker(sc, pl.at(c), ink, op, r)

    def trails(self, sc, pl, paths, ink, op, width=2.0):
        """``paths`` ``(steps, K, 2)``: each centre's path so far."""
        with self.over(1.4):
            for j in range(paths.shape[1]):
                pl.path(sc, paths[:, j], ink, op, width)

    def _km_path(self, upto=None):
        """K-means' centre positions, thinned to the iterations that still move."""
        km = self.km if upto is None else self.km[:upto + 1]
        return km[np.r_[np.arange(0, min(len(km), 12)), np.arange(12, len(km), 6)]]

    # ------------------------------------------------------------------ the AE's codes
    def phase_1(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        pl = Plane(self, 0.665, 0.60, 300.0, self.lim)
        pl.frame(sc, op, "the autoencoder's code space")
        show = smoothstep(0.14, 0.42, u)
        self.cloud(sc, pl, self.z[0], op * show, r=2.3)
        y = 0.47
        self.digit(sc, self.x0, 0.10, y, 72, op, frame=AXIS)
        self.net(sc, 0.235, y, BLUE, "AE encoder", op, w=116.0, h=44.0)
        self.link(sc, (0.142, y), (0.172, y), BLUE, op, pulse=seg(u, 0.0, 0.1))
        at = pl.at(self.z[0][S.TRACK])
        with self.over(1.4):
            self.link(sc, (0.298, y), at, BLUE, op, seg(u, 0.06, 0.2), pulse=seg(u, 0.06, 0.24),
                      bend=0.04)
        self.disk(sc, self.P(*at, lift=1.8), 6.5, TEN[int(self.labels[S.TRACK])],
                  op * smoothstep(0.18, 0.24, u), ring=WHITE, ring_w=2.0, lift=2.4)
        self.text(sc, 0.175, y + 0.08, "784 pixels in, 2 numbers out", op, anchor="lm")
        self.lines(sc, 0.07, 0.655, [
            "trained to rebuild 20,000 digits: no labels,",
            "no clusters. Rebuilding error on held-out",
            "digits: %.3f" % self.run["mse"][0],
            "similar digits already sit together: that",
            "is what clustering starts from"], op * smoothstep(0.2, 0.34, u), step=0.045,
            size=0.032)
        for digit in range(10):
            self.box(sc, 0.895, 0.375 + 0.05 * digit, 26, 24, TEN[digit], op, str(digit),
                     ink=WHITE, size=0.031)

    # ------------------------------------------------------------------ k-means
    def _km_at(self, u):
        if u < 0.6:
            return seg(u, 0.06, 0.6) * self.STEPWISE
        return self.STEPWISE + seg(u, 0.6, 0.9) * (S.KM_ITERS - self.STEPWISE)

    def phase_2(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        at = self._km_at(u)
        i = int(min(at, S.KM_ITERS - 1))
        f = at - i
        stepwise = at < self.STEPWISE
        move = smoothstep(0.45, 1.0, f) if stepwise else f
        mu = self.km[i] * (1 - move) + self.km[i + 1] * move
        pl = Plane(self, 0.215, 0.585, 280.0, self.lim)
        pl.frame(sc, op, "2,000 held-out codes and ten centres")
        self.cloud(sc, pl, self.z[0], op, r=2.0, back=regions(self.km[i] if stepwise else mu,
                                                                self.lim, pl.side))
        path = np.concatenate([self._km_path(i), mu[None]])
        self.trails(sc, pl, path, DIGIT_INK, op, 2.2)
        self.centres(sc, pl, mu, op, DIGIT_INK)
        self.legend(sc, 0.085, 0.905, op, head="")
        x = 0.41
        done = int(round(at)) if not stepwise else i
        self.text(sc, x, 0.32, "k-means, round %d of %d" % (min(done + 1, S.KM_ITERS),
                                                           S.KM_ITERS), op, anchor="lm",
                  strong=True, size=0.036)
        assign = stepwise and f < 0.45
        rows = (("start: ten codes picked at random as centres", u < 0.06),
                ("1  give every code to its nearest centre (the borders)", assign and u >= 0.06),
                ("2  move each centre to the mean of its codes", stepwise and not assign),
                ("repeat until the centres stop moving", not stepwise))
        for j, (text, on) in enumerate(rows):
            self.text(sc, x, 0.385 + 0.045 * j, text, op * (1.0 if on else 0.45), anchor="lm",
                      size=0.032, ink=layout.STRONG_INK if on else None)
        within = self.st["km_within"]
        chart = self.chart(((x + 0.04, 0.93), (0.61, 0.76)), [0, S.KM_ITERS],
                           [0.0, within.max()], pad=0.04)
        self.axes(sc, chart, op, "round", "mean squared distance to the nearest centre: %.2f"
                  % within[done], xticks=[(v, str(v)) for v in (0, 5, 20, 40, 60)],
                  yticks=[(0, "0"), (2, "2"), (4, "4")])
        self.plot(sc, chart, np.arange(len(within)), within, DIGIT_INK, op, at / S.KM_ITERS,
                  width=2.0)
        self.text(sc, x, 0.88, "fitted on 20,000 training codes; accuracy on held-out "
                  "digits %.2f" % self.st["km_acc"][done], op, anchor="lm")

    # ------------------------------------------------------------------ soft to sharp
    def phase_3(self, sc, u, op):
        self.evidence(sc, EXACT, op)
        z0, mu0 = self.z[0], self.cen[0]
        pl = Plane(self, 0.165, 0.585, 214.0, self.lim)
        pl.frame(sc, op, "three codes, ten centres", below=None)
        self.cloud(sc, pl, z0, op * 0.9, r=1.6, back=regions(mu0, self.lim, pl.side))
        self.centres(sc, pl, mu0, op, DIGIT_INK, r=4.5)
        for r, i in enumerate(self.pick):
            a = op * smoothstep(0.03 + 0.06 * r, 0.1 + 0.06 * r, u)
            at = pl.at(z0[i])
            with self.over(1.9):
                for c in np.argsort(-self.q3[r])[:3]:
                    self.arc(sc, at, pl.at(mu0[c]), PC, a, bend=0.0, head=False,
                             width=0.8 + 4.0 * float(self.q3[r][c]))
            self.disk(sc, self.P(*at, lift=2.4), 7.5, TEN[int(self.labels[i])], a, ring=WHITE,
                      ring_w=2.6, lift=3.0)
        self.legend(sc, 0.033, 0.875, op, head="")
        # q and p over the ten centres, for each of the three
        x = 0.305
        self.box(sc, x + 0.008, 0.315, 16, 16, QC, op, radius=3.0)
        self.text(sc, x + 0.024, 0.315, "q: soft assignment", op, anchor="lm")
        self.box(sc, x + 0.198, 0.315, 16, 16, PC, op, radius=3.0)
        self.text(sc, x + 0.214, 0.315, "p: target", op, anchor="lm")
        bx = x + 0.115
        for r, i in enumerate(self.pick):
            base = 0.50 + 0.155 * r
            a = op * smoothstep(0.08 + 0.07 * r, 0.16 + 0.07 * r, u)
            grow_q = seg(u, 0.1 + 0.07 * r, 0.22 + 0.07 * r)
            grow_p = seg(u, 0.42 + 0.05 * r, 0.56 + 0.05 * r)
            self.disk(sc, self.P(x + 0.008, base - 0.085), 7.5, TEN[int(self.labels[i])], a,
                      ring=WHITE, ring_w=2.6)
            self.text(sc, x + 0.028, base - 0.085, "a held-out %d" % self.labels[i], a,
                      anchor="lm")
            self.polyline(sc, [self.P(bx, base), self.P(bx + 0.225, base)], 1.0, AXIS, a * 0.7)
            for c in range(S.K):
                fx = bx + 0.016 + 0.0215 * c
                self.vbar(sc, fx - 0.0042, base, 7.0, 62 * self.q3[r][c], QC, a, grow_q)
                self.vbar(sc, fx + 0.0042, base, 7.0, 62 * self.p3[r][c], PC, a * (grow_p > 0),
                          grow_p)
            self.text(sc, x, base - 0.04, "largest q %.2f" % self.q3[r].max(), a, anchor="lm")
            self.text(sc, x, base, "largest p %.2f" % self.p3[r].max(),
                      a * smoothstep(0.5, 1.0, grow_p), anchor="lm", strong=True)
        self.text(sc, bx + 0.11, 0.855, "a pair of bars per centre", op)
        # the arithmetic for A
        x = 0.665
        a, b = self.top2
        rows = [
            ("the %d and its two nearest centres" % self.labels[self.pick[0]], layout.STRONG_INK,
             0.12),
            ("squared distance d²:  %.2f  and  %.2f" % tuple(self.d2), None, 0.16),
            ("kernel 1 / (1 + d²):  %.2f  and  %.2f" % tuple(self.kern2), None, 0.22),
            ("all ten kernels add to %.2f" % self.kern_sum, None, 0.26),
            ("q = kernel / %.2f:  %.2f  and  %.2f" % (
                self.kern_sum, self.q3[0][a], self.q3[0][b]), dark(QC, 0.45), 0.3),
            ("target p: q² / cluster's total q,", layout.STRONG_INK, 0.42),
            ("rescaled to sum to 1:  %.2f  and  %.2f" % (
                self.p3[0][a], self.p3[0][b]), PC, 0.48),
            ("loss KL(P || Q) = Σ p ln(p / q) = %.3f" % self.kl_a, layout.STRONG_INK, 0.62),
            ("its gradient moves the code,", None, 0.7),
            ("through the encoder, and the centres", None, 0.7)]
        for j, (text, ink, t0) in enumerate(rows):
            self.text(sc, x, 0.375 + 0.048 * j, text, op * smoothstep(t0, t0 + 0.08, u),
                      anchor="lm", ink=ink, size=0.03)

    # ------------------------------------------------------------------ DEC trains
    def phase_4(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        n = self.n
        at = seg(u, 0.05, 0.92) * (n - 1)
        k = int(round(at))
        prog = at / (n - 1)
        z, mu = self.between(self.z, at), self.between(self.cen, at)
        pl = Plane(self, 0.205, 0.585, 268.0, self.lim)
        pl.frame(sc, op, "2,000 held-out codes, ten centres")
        self.cloud(sc, pl, z, op, r=2.0, back=regions(mu, self.lim, pl.side))
        self.trails(sc, pl, self._km_path(), DIGIT_INK, op * 0.75, 1.6)
        self.trails(sc, pl, np.concatenate([self.cen[:int(at) + 1], mu[None]]), TRAIL, op, 2.6)
        self.centres(sc, pl, mu, op, AMBER)
        self.legend(sc, 0.075, 0.895, op, head="")
        x = 0.385
        r = self.run
        self.text(sc, x, 0.305, "DEC update %s" % f"{S.DEC_CHECKS[k]:,}", op, anchor="lm",
                  strong=True, size=0.036)
        self.lines(sc, x, 0.365, [
            ("label purity", dark(PURE, 0.25)),
            ("accuracy %.2f   (was %.2f)" % (r["acc"][k], r["acc"][0]), None),
            ("shared information %.2f" % r["nmi"][k], None)], op, step=0.043, size=0.031)
        self.lines(sc, x, 0.525, [
            ("separation", dark(SEP, 0.35)),
            ("silhouette %.2f   (was %.2f)" % (r["sil"][k], r["sil"][0]), None),
            ("mean top q %.2f" % r["maxq"][k], None)], op, step=0.043, size=0.031)
        self.lines(sc, x, 0.685, [
            ("centres moved %.2f" % r["moved"][k], dark(MOVED, 0.3)),
            ("trails: black k-means,", None), ("orange DEC", None),
            ("numbers: this run (seed %d)" % S.DRAWN, None)], op, step=0.043, size=0.031)
        x0, x1 = 0.695, 0.93
        acc, nm = self.by["dec"]["acc"], self.by["dec"]["nmi"]
        c1 = self.run_chart(sc, ((x0, x1), (0.345, 0.445)), n, [0.38, 0.62], op,
                            "accuracy (bold), shared information",
                            yticks=[(0.4, "0.4"), (0.5, "0.5"), (0.6, "0.6")])
        self.plot(sc, c1, np.arange(n), nm.mean(0), soft(PURE, 0.3), op, prog, width=1.2,
                  dot=False)
        self.seeds(sc, c1, acc, PURE, op, prog)
        c2 = self.run_chart(sc, ((x0, x1), (0.53, 0.63)), n, [0.3, 0.8], op,
                            "silhouette (bold), mean top q",
                            yticks=[(0.4, "0.4"), (0.6, "0.6"), (0.8, "0.8")])
        self.plot(sc, c2, np.arange(n), self.by["dec"]["maxq"].mean(0), soft(SEP, 0.3), op, prog,
                  width=1.2, dot=False)
        self.seeds(sc, c2, self.by["dec"]["sil"], SEP, op, prog)
        c3 = self.run_chart(sc, ((x0, x1), (0.715, 0.815)), n, [0.0, 1.5], op,
                            "centres moved since k-means", xticks=self.XT,
                            yticks=[(0, "0"), (1.5, "1.5")],
                            xlabel="DEC updates; three seeds, mean in bold")
        self.seeds(sc, c3, self.by["dec"]["moved"], MOVED, op, prog)

    # ------------------------------------------------------------------ separated?
    def phase_5(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        pl = Plane(self, 0.175, 0.575, 226.0, self.lim)
        pl.frame(sc, op, "after DEC")
        self.cloud(sc, pl, self.z[-1], op, r=1.9, back=regions(self.cen[-1], self.lim, pl.side))
        self.centres(sc, pl, self.cen[-1], op, AMBER, r=4.5)
        self.legend(sc, 0.043, 0.895, op, head="")
        # the clusters against the digits
        a = op * smoothstep(0.1, 0.25, u)
        share = self.conf / np.maximum(self.conf.sum(1, keepdims=True), 1)
        gx, gy, side = 0.50, 0.60, 216.0
        cell = self.grid(sc, share, gx, gy, side, side, a, reveal=seg(u, 0.1, 0.45), vmax=1.0,
                         hot=AMBER)
        for digit in range(10):
            fx, _ = cell(0, digit)
            self.box(sc, fx, gy - self.frac(0, side / 2)[1] - 0.034, 19, 20, TEN[digit], a,
                     str(digit), ink=WHITE, size=0.03)
        self.text(sc, gx - 0.01, gy + self.frac(0, side / 2)[1] + 0.04,
                  "rows: clusters; columns: digits", a)
        for row in range(S.K):
            _, fy = cell(row, 0)
            self.text(sc, gx - self.frac(side / 2)[0] - 0.012, fy, "%d" % self.conf[row].sum(),
                      a, anchor="rm")
        self.text(sc, gx - self.frac(side / 2)[0] - 0.012, 0.36, "codes", a, anchor="rm")
        x = 0.675
        r = self.run
        b = op * smoothstep(0.3, 0.42, u)
        self.lines(sc, x, 0.35, [
            ("separation: no labels needed", dark(SEP, 0.35)),
            ("silhouette %.2f, from %.2f" % (r["sil"][-1], r["sil"][0]), layout.STRONG_INK),
            ("(1: tight, apart; 0: overlapping)", None)], b, step=0.045, size=0.032)
        c = op * smoothstep(0.45, 0.57, u)
        self.lines(sc, x, 0.535, [
            ("label purity: labels as the judge", dark(PURE, 0.25)),
            ("accuracy %.2f, from %.2f" % (r["acc"][-1], r["acc"][0]), layout.STRONG_INK),
            ("(best one-to-one matching of", None), ("clusters to digits)", None)], c,
            step=0.045, size=0.032)
        e = op * smoothstep(0.62, 0.74, u)
        self.lines(sc, x, 0.765, [("sharing a cluster in this run:", None),
                                  (self.shared_short, layout.STRONG_INK)], e, step=0.045,
                   size=0.032)

    # ------------------------------------------------------------------ depends on
    def phase_6(self, sc, u, op):
        self.evidence(sc, MEASURED, op)
        for i, (kind, name) in enumerate((("dec", "DEC"), ("idec", "IDEC: decoder kept"))):
            run = self.run if kind == "dec" else self.idec
            pl = Plane(self, 0.135 + 0.2 * i, 0.50, 160.0, self.lim)
            pl.frame(sc, op, name, ink=dark(AMBER, 0.4))
            self.cloud(sc, pl, run["z"][-1], op, r=1.5)
            self.centres(sc, pl, run["mu"][-1], op, AMBER, r=3.6)
            self.lines(sc, pl.fx, 0.70, [
                "rebuilding error %.3f" % self.by[kind]["mse"][:, -1].mean(),
                "silhouette %.2f" % self.by[kind]["sil"][:, -1].mean()], op, anchor="mm",
                step=0.043)
        self.text(sc, 0.235, 0.80, "the autoencoder they start from: %.3f" %
                  self.by["dec"]["mse"][:, 0].mean(), op)
        self.legend(sc, 0.103, 0.885, op, head="")
        x, px = 0.70, 200.0
        self.text(sc, 0.505, 0.315, "cluster accuracy, held-out digits (a dot per seed)", op,
                  anchor="lm", strong=True)
        wide = {kind: np.array([S.dec_run(kind, 10, s)["acc"] for s in S.SEEDS])
                for kind in S.KINDS}
        acc = self.by
        rows = [("2 numbers: k-means", DIGIT_INK, acc["dec"]["acc"][:, 0]),
                ("then DEC", AMBER, acc["dec"]["acc"][:, -1]),
                ("then IDEC", dark(AMBER, 0.3), acc["idec"]["acc"][:, -1]),
                ("10 numbers: k-means", DIGIT_INK, wide["dec"][:, 0]),
                ("then DEC", AMBER, wide["dec"][:, -1]),
                ("then IDEC", dark(AMBER, 0.3), wide["idec"][:, -1])]
        ys = [0.375 + 0.052 * j + (0.02 if j >= 3 else 0.0) for j in range(6)]
        self.seed_bars(sc, x, ys, rows, px, 0.8, op, seg(u, 0.1, 0.5), thick=15.0)
        st = self.m["starts"]
        b = op * smoothstep(0.45, 0.6, u)
        self.text(sc, 0.505, 0.745, "one autoencoder, five k-means starts (a dot each)", b,
                  anchor="lm", strong=True)
        self.seed_bars(sc, x, [0.80, 0.852], [
            ("k-means", DIGIT_INK, st["accuracy_start"]),
            ("then DEC", AMBER, st["accuracy_end"])], px, 0.8, b, seg(u, 0.5, 0.85),
            thick=15.0)


ANIMATIONS = {"vae-kl-term": VaeKlTerm, "dec-centroids-training": DecCentroids}
