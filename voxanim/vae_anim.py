"""A variational autoencoder, animated.  Same stage as the plain autoencoder
(:mod:`voxanim.ae_anim`), with what makes it variational:

* the encoder outputs a mean mu and a spread sigma per latent dimension; the
  latent space shows the digit's mu, a cloud of the codes it *could* be
  (mu + sigma * eps), and the one actually sampled,
* the KL term pulls every code towards N(0, I), so the latent axes are the
  prior's own range, -3 .. 3, and the space is filled without gaps,
* the ending samples brand-new points from the prior and decodes them: digits
  that were never in the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from . import deep_models as D
from .ae_anim import DIGIT_COLORS, AutoencoderAnimation
from .easing import ease_out, remap
from .netanim import ACT_HI, ACT_LO, amax, div_colors, gray_colors, lit_colors, particle


@dataclass
class VAETrace:
    kind: str
    latent: int
    P: dict
    images: np.ndarray
    labels: np.ndarray
    fwd: list
    cloud: np.ndarray            # mu of 600 test digits
    cloud_labels: np.ndarray
    test_loss: float
    mus: list
    sigmas: list
    eps: list

    def decode(self, z):
        g0, g1, out = D.vae_decode(self.P, z)
        return [np.asarray(z), g0[0], g1[0], out[0]], out[0]


def vae_trace(latent=3, digits=(3, 2, 1), cloud=600, seed=5):
    from .mnist import load
    P, loss = D.vae_train(latent)
    _a, _b, xte, yte = load()
    rng = np.random.default_rng(seed)
    fwd, mus, sig, eps = [], [], [], []
    for i in digits:
        e = rng.normal(size=(1, latent))
        f = D.vae_forward(P, xte[i:i + 1], e)
        fwd.append({"z": f["z"][0], "acts": [f["x"][0], f["h0"][0], f["h1"][0], f["z"][0],
                                             f["g0"][0], f["g1"][0], f["recon"][0]]})
        mus.append(f["mu"][0])
        sig.append(np.exp(0.5 * f["lv"][0]))
        eps.append(e[0])
    sel = np.arange(2000, 2000 + cloud)
    f = D.vae_forward(P, xte[sel], np.zeros((cloud, latent)))
    return VAETrace("dense", latent, P, xte[list(digits)], yte[list(digits)], fwd, f["mu"],
                    yte[sel], loss, mus, sig, eps)


class VAEAnimation(AutoencoderAnimation):
    WEIGHT_KEYS: ClassVar[dict] = {0: "We0", 1: "We1", 2: "Wmu", 3: "Wd0", 4: "Wd1", 5: "Wd2"}
    WALK_SUBTITLE = ("generating: sample z from N(0, I) and decode it: new digits that were "
                     "never in the data")
    RANGE = 3.0

    def __init__(self, tr):
        super().__init__(tr, title="Dense VAE, %d-D latent" % tr.latent)
        # the prior's range, not the data's: the KL term makes them match
        self.lo = np.full(self.L, -self.RANGE)
        self.hi = np.full(self.L, self.RANGE)
        self.cloud_pts = np.array([self.to_box(z) for z in tr.cloud])
        rng = np.random.default_rng(11)
        self.region = [m + s * rng.normal(size=(36, self.L))
                       for m, s in zip(tr.mus, tr.sigmas)]
        self.prior = rng.normal(size=(5, self.L))
        self.walk = list(self.prior)

    def _warp(self, z):
        return np.asarray(z, np.float64)          # no reshaping: the axes are N(0, 1)

    def _digit(self, sc, t, d, back, active):
        super()._digit(sc, t, d, back, active)
        if not active:
            return
        prog = [remap(t, a, b) for a, b in self.digit_t[d]]
        mu, sig = self.tr.mus[d], self.tr.sigmas[d]
        lab = int(self.tr.labels[d])
        col = DIGIT_COLORS[lab]
        fly = prog[3]
        if prog[2] >= 1:
            show = (1 - remap(t, self.digit_t[d][5][0], self.digit_t[d][6][1])) * back
            n = int(min(1.0, fly * 2.5) * len(self.region[d]))
            for p in self.region[d][:n]:              # where this digit's code could land
                sc.voxel(self.to_box(p), col, 0.35 * show, shadow=False, reflect=False,
                         size=0.6)
            sc.voxel(self.to_box(mu), (1.0, 1.0, 1.0), show, shadow=False, reflect=False,
                     size=1.0)
            if 0.4 < fly < 1:                         # the sample: mu + sigma * eps
                particle(sc, self.to_box(mu), self.to_box(self.codes[d]),
                         ease_out(remap(fly, 0.4, 1.0)), col, 1.2, lift=0.5)
        ms = ", ".join("%.2f" % v for v in mu)
        ss = ", ".join("%.2f" % v for v in sig)
        if 0 < fly < 1 or (prog[2] >= 1 and prog[4] <= 0):
            self.subtitle = ("encoder: mean μ = (%s), spread σ = (%s); "
                             "z = μ + σ·ε is sampled" % (ms, ss))

    def _walk(self, sc, u, back):
        n = len(self.prior)
        k = min(n - 1, int(u * n))
        z = self.prior[k]
        acts, _img = self.tr.decode(z)
        for b in self.blocks[:3]:
            b.present[...] = False
        self.code.base[0, :, 0] = div_colors(z / self.RANGE, 1.0)
        self.code.opacity *= back
        for kk, a in zip((4, 5, 6), acts[1:]):
            v = np.asarray(a).reshape(-1)
            col = gray_colors(v) if kk == 6 else lit_colors(v, amax(v), ACT_LO, ACT_HI)
            b = self.blocks[kk]
            b.base[...] = col.reshape(self.shapes[kk] + (3,))
            b.opacity *= back
        pop = ease_out(remap(u * n - k, 0.0, 0.25))
        for j in range(k + 1):                         # every sample so far stays marked
            sc.voxel(self.to_box(self.prior[j]), (1.0, 1.0, 1.0), back, glow=0.4 if j == k
                     else 0.1, shadow=False, reflect=False,
                     size=(1.4 * pop if j == k else 0.9))
        sc.probe(("gen", k), (pop * 30, 0, 0), tol=8.0)
        self._err = 0.0
