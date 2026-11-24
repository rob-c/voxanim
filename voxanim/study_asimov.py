"""The Asimov dataset against toy experiments, for the animation "The Asimov dataset" of "Blocks
and code" (catalogue 278): ``examples/asimov.py`` in numpy.

A made-up binned analysis: 4 bins of a mass, expected signal ``S`` = (3, 8, 8, 3) and expected
background ``B`` = (30, 20, 12, 8) (made-up numbers). One experiment's counts n are Poisson(s + b)
in each bin. Its discovery significance (:func:`z0`) fits the signal strength mu >= 0 by Newton's
method (20 steps from mu = 1, the script's ``fit_mu``), then q0 = 2 sum(n ln(1 + mu s / b) - mu s)
and Z = sqrt(q0) (the large-sample formula).

* :func:`toys`: 10,000 toy experiments (``default_rng(seed).poisson(s + b, (10000, 4))``), each
  one's Z, their median: the expected significance.
* :func:`asimov_z`: the Asimov dataset, n = s + b exactly; its fit gives mu = 1 and its Z is
  sqrt(2 sum((s + b) ln(1 + s / b) - s)) (:func:`formula`), bin by bin and all together.
* :func:`simple`: s / sqrt(b) bin by bin, added in quadrature.
* :func:`case`: all three, for the background as it is (toys seed 7) and for a tenth of it (seed
  8), where s / sqrt(b) is far too high and the Asimov Z still lands on the toys' median.

The toys are run side by side here (:func:`z0` takes ``[T, 4]`` counts), the script's arithmetic
toy by toy (``tests/test_study_asimov.py`` checks it against the script).
"""

from __future__ import annotations

import functools

import numpy as np

S = np.array([3.0, 8.0, 8.0, 3.0])        # expected signal, per bin
B = np.array([30.0, 20.0, 12.0, 8.0])     # expected background, per bin
BINS = len(S)
TOYS = 10000
NEWTON = 20
CASES = {"b": (1.0, 7), "b/10": (0.1, 8)}  # background scale, toys' seed


def fit_mu(n, s=S, b=B):
    """The script's ``fit_mu`` for each row of counts ``n`` (``[T, 4]`` or ``[4]``)."""
    n = np.asarray(n, float)
    one = n.ndim == 1
    n = np.atleast_2d(n)
    mu = np.ones(len(n))
    for _ in range(NEWTON):
        m = mu[:, None] * s + b
        slope = (n * s / m - s).sum(1)
        bend = (n * s * s / m ** 2).sum(1)
        mu = np.maximum(mu + slope / bend, 0.0)
    return mu[0] if one else mu


def z0(n, s=S, b=B):
    """The script's ``z0``: each experiment's discovery significance (and its fitted mu)."""
    n = np.asarray(n, float)
    one = n.ndim == 1
    n = np.atleast_2d(n)
    mu = fit_mu(n, s, b)
    extra = mu[:, None] * s
    t = n * np.log(1 + extra / b) - extra
    z = np.sqrt(2 * t.sum(1))
    return (z[0], mu[0]) if one else (z, mu)


def formula(s=S, b=B):
    """The Asimov Z of each bin alone: sqrt(2((s + b) ln(1 + s / b) - s))."""
    return np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))


def simple(s=S, b=B):
    """s / sqrt(b) bin by bin, and all bins together (in quadrature)."""
    per = s / np.sqrt(b)
    return per, float(np.sqrt((s ** 2 / b).sum()))


def asimov_z(s=S, b=B):
    """The Asimov dataset's Z, through the script's fit."""
    z, mu = z0(s + b, s, b)
    return float(z), float(mu)


@functools.lru_cache(maxsize=4)
def toys(scale=1.0, seed=7):
    """``(counts [10000, 4], Z [10000], mu [10000])`` for background ``scale * B``."""
    b = B * scale
    rng = np.random.default_rng(seed)
    n = rng.poisson(S + b, (TOYS, BINS))
    z, mu = z0(n, S, b)
    return n, z, mu


def converged(n, mu, s=S, b=B):
    """True where the fit has reached the likelihood's peak: slope 0, or mu = 0 with the slope
    pointing below 0."""
    m = mu[:, None] * s + b
    slope = (np.asarray(n) * s / m - s).sum(1)
    return np.where(mu > 0, np.abs(slope) < 1e-9, slope <= 1e-12)


@functools.lru_cache(maxsize=4)
def case(name="b"):
    """Everything for one background: the expected counts, the toys, the Asimov dataset and
    s / sqrt(b)."""
    scale, seed = CASES[name]
    b = B * scale
    n, z, mu = toys(scale, seed)
    za, mua = asimov_z(S, b)
    per, total = simple(S, b)
    zbin = formula(S, b)
    return dict(
        name=name, scale=scale, seed=seed, s=S.copy(), b=b, expected=S + b, asimov=S + b,
        toys=n, z=z, mu=mu, median=float(np.median(z)), mean=float(z.mean()),
        above=float((z > za).mean()), asimov_z=za, asimov_mu=mua,
        asimov_formula=float(np.sqrt((zbin ** 2).sum())), asimov_bins=zbin,
        simple=total, simple_bins=per, first=n[0], first_z=float(z[0]), first_mu=float(mu[0]),
        converged=int(converged(n, mu, S, b).sum()), zero=int((mu == 0).sum()))


def printed(name="b"):
    """What the script prints for a case, as numpy prints it: the median toy, the Asimov Z,
    s / sqrt(b)."""
    c = case(name)
    return dict(median=str(np.median(c["z"]).round(2)), asimov=str(np.round(c["asimov_z"], 2)),
                simple=str(np.round(c["simple"], 2)))
