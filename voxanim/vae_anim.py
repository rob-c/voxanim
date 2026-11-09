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

from . import deep_models
from .ae_anim import DIGIT_COLORS, AutoencoderAnimation, fmt_vec
from .autoencoder import CLOUD_START
from .cache import DEFAULT_CACHE
from .easing import ease_out, remap
from .mnist import load
from .netanim import particle

SAMPLE_SEED = 5             # the eps drawn for the three digits
REGION_SEED = 11            # the "could have been" clouds and the prior samples
REGION_SAMPLES = 36         # codes drawn per digit to show its spread
PRIOR_SAMPLES = 5           # brand-new codes decoded at the end


@dataclass
class VAETrace:
    """Digits through a trained VAE (fields as :class:`voxanim.autoencoder.AETrace`)."""

    kind: str
    latent: int
    P: dict
    images: np.ndarray           # (k, 784)
    labels: np.ndarray           # (k,)
    fwd: list                    # per digit: {"z": sampled code, "acts": the 7 chain tensors}
    cloud: np.ndarray            # (m, latent) mu of the scatter's test digits
    cloud_labels: np.ndarray     # (m,)
    mus: list                    # per digit: mu (latent,)
    sigmas: list                 # per digit: sigma = exp(log var / 2) (latent,)
    eps: list                    # per digit: the noise drawn (latent,)

    def decode(self, z):
        """Decoder only: ``z`` ``(L,)`` -> ([z, g0, g1, image], image (784,))."""
        g0, g1, out = deep_models.vae_decode(self.P, z)
        return [np.asarray(z), g0[0], g1[0], out[0]], out[0]


def vae_trace(latent=3, digits=(3, 2, 1), cloud=600, seed=SAMPLE_SEED,
              cache_dir=DEFAULT_CACHE):
    """Test digits ``digits`` through the trained VAE (one sample each, drawn
    with ``seed``), plus the means of ``cloud`` test digits for the scatter."""
    P, _loss = deep_models.vae_train(latent, cache_dir=cache_dir)
    _xtr, _ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    fwd, mus, sig, eps = [], [], [], []
    for i in digits:
        e = rng.normal(size=(1, latent))
        f = deep_models.vae_forward(P, xte[i:i + 1], e)
        fwd.append({"z": f["z"][0], "acts": [f["x"][0], f["h0"][0], f["h1"][0], f["z"][0],
                                             f["g0"][0], f["g1"][0], f["recon"][0]]})
        mus.append(f["mu"][0])
        sig.append(np.exp(0.5 * f["lv"][0]))
        eps.append(e[0])
    sel = np.arange(CLOUD_START, CLOUD_START + cloud)
    f = deep_models.vae_forward(P, xte[sel], np.zeros((cloud, latent)))
    return VAETrace("dense", latent, P, xte[list(digits)], yte[list(digits)], fwd, f["mu"],
                    yte[sel], mus, sig, eps)


class VAEAnimation(AutoencoderAnimation):
    """:class:`voxanim.ae_anim.AutoencoderAnimation` for a :class:`VAETrace`:
    mu, sigma and the sample in the latent space; generation at the end."""

    WEIGHT_KEYS: ClassVar[dict] = {0: "We0", 1: "We1", 2: "Wmu", 3: "Wd0", 4: "Wd1", 5: "Wd2"}
    WALK_SUBTITLE = ("generating: sample z from N(0, I) and decode it: new digits that were "
                     "never in the data")
    RANGE = 3.0                 # the latent axes span -RANGE .. RANGE

    def __init__(self, tr):
        super().__init__(tr, title="Dense VAE, %d-D latent" % tr.latent)
        rng = np.random.default_rng(REGION_SEED)
        self.region = [m + s * rng.normal(size=(REGION_SAMPLES, self.L))
                       for m, s in zip(tr.mus, tr.sigmas)]
        self.prior = rng.normal(size=(PRIOR_SAMPLES, self.L))
        self.walk = list(self.prior)

    def _axes(self, cloud):
        """The prior's range, not the data's: the KL term makes them match."""
        return np.full(self.L, -self.RANGE), np.full(self.L, self.RANGE)

    def _warp(self, z):
        """No reshaping: the axes are N(0, 1)."""
        return np.asarray(z, np.float64)

    def _digit(self, sc, t, d, back, active):
        """As the base class, plus the digit's mu, its spread and the sample."""
        landed, subtitle = super()._digit(sc, t, d, back, active)
        if not active:
            return landed, subtitle
        prog = [remap(t, a, b) for a, b in self.digit_t[d]]
        mu, sig = self.tr.mus[d], self.tr.sigmas[d]
        col = DIGIT_COLORS[int(self.tr.labels[d])]
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
        if 0 < fly < 1 or (prog[2] >= 1 and prog[4] <= 0):
            subtitle = ("encoder: mean μ = (%s), spread σ = (%s); z = μ + σ·ε is sampled"
                        % (fmt_vec(mu), fmt_vec(sig)))
        return landed, subtitle

    def _walk(self, sc, u, back):
        """Generation: the prior samples decoded one after another."""
        n = len(self.prior)
        k = min(n - 1, int(u * n))
        self._show_decoded(self.prior[k], self.RANGE, back)
        pop = ease_out(remap(u * n - k, 0.0, 0.25))
        for j in range(k + 1):                         # every sample so far stays marked
            sc.voxel(self.to_box(self.prior[j]), (1.0, 1.0, 1.0), back, glow=0.4 if j == k
                     else 0.1, shadow=False, reflect=False,
                     size=(1.4 * pop if j == k else 0.9))
        sc.probe(("gen", k), (pop * 30, 0, 0), tol=8.0)
