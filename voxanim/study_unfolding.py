"""Unfolding, a detector's smearing run backwards, for the unfolding animation of "Blocks and
code" (catalogue 277): ``examples/unfolding.py`` in numpy, the same seeded draws in the same
order, so every number is the script's (``tests/test_study_unfolding.py`` runs the script and
checks).

Made-up events. **The truth** (:func:`truth`): 10,000 events of energy 0 to 100 GeV in 10 bins
of 10 GeV, three quarters falling as exp(-E / 25 GeV) and a quarter a Gaussian bump at 60 GeV,
10 GeV wide, both cut to 0-100 GeV (expected counts, worked out exactly). **The detector**:
finds an event with chance 0.6 + 0.003 E (60% at 0 GeV, 90% at 100), measures its energy off
by a Gaussian 10 GeV wide; measured outside 0-100 GeV, it is lost.

**The response matrix** R (:func:`world`), built from simulation: 1,000,000 simulated true
energies falling as exp(-E / 30 GeV) (a different shape from the truth's: no bump), the
964,628 below 100 GeV each found or not and measured; ``R[i, j]`` = events of true bin j
found and measured in bin i, divided by the events simulated in true bin j. Its columns add
up to the chance an event of that bin is found and kept (0.42 to 0.82).

**Measured**: ``expected = R @ truth`` (6,345 events), ``data`` Poisson counts round it
(6,362). **Inverting R**: ``solve(R, expected)`` is the truth again (3.6e-12 apart);
``solve(R, data)`` swings from -900 to 3,475. Why: R = U diag(s) Vt with singular values
0.746 down to 0.0078; the noise ``data - expected`` along each U[:, k] (about 25 events each,
expected) comes back divided by s[k] along Vt[k].

**Iterative Bayesian unfolding** (D'Agostini; :func:`bayes`): from the simulation's spectrum
scaled to the data (the prior), each iteration folds the guess with R, divides the data by
it, and corrects the guess by R's columns. **Pseudo-experiments** (:func:`toys`): 2,000 new
Poisson draws of the data, each unfolded with 0 to :data:`UPTO` iterations: the bias (the
average's distance from the truth) and the spread (the scatter from draw to draw), root mean
square over the 10 bins, and the inversion's (its spread also worked out exactly,
:func:`inverse_spread`).
"""

from __future__ import annotations

import functools
import math

import numpy as np

SEED = 1
N = 10_000                     # true events
SIMULATED = 1_000_000          # simulated events, before the 100 GeV cut
TOYS = 2_000                   # pseudo-experiments
BINS = 10
LO, HI = 0.0, 100.0
SLOPE, BUMP_AT, BUMP_WIDTH, BUMP_SHARE = 25.0, 60.0, 10.0, 0.25
SIM_SLOPE = 30.0
FOUND_LO, FOUND_PER_GEV = 0.6, 0.003
RESOLUTION = 10.0
UPTO = 50                      # iterations followed in the pseudo-experiments
SCRIPT_ITERATIONS = (0, 1, 2, 4, 10, 20, 50)   # the ones the script prints


def edges():
    return np.linspace(LO, HI, BINS + 1)


def falling(e, slope=SLOPE):
    """The fraction of exp(-E / slope), cut to the bins, in each bin."""
    c = 1.0 - np.exp(-e / slope)
    return np.diff(c) / (c[-1] - c[0])


def bump(e, at=BUMP_AT, width=BUMP_WIDTH):
    """The fraction of a Gaussian bump, cut to the bins, in each bin."""
    c = np.array([math.erf((x - at) / (width * math.sqrt(2.0))) for x in e])
    return np.diff(c) / (c[-1] - c[0])


def truth():
    """The true spectrum's expected counts per bin (10,000 in all)."""
    e = edges()
    return N * ((1 - BUMP_SHARE) * falling(e) + BUMP_SHARE * bump(e))


def bayes(data, prior, R, eff, iterations):
    """The script's ``bayes``: D'Agostini's iterations from ``prior`` (``data``: one
    spectrum or a stack of them, one per row)."""
    t = prior * np.ones(np.shape(data))
    for _ in range(iterations):
        fold = t @ R.T
        t = t * ((data / fold) @ R) / eff
    return t


def iterates(data, prior, R, eff, upto=UPTO):
    """Every iterate of :func:`bayes`, 0 to ``upto``: ``[upto + 1, ..., 10]``."""
    t = prior * np.ones(np.shape(data))
    out = [t]
    for _ in range(upto):
        fold = t @ R.T
        t = t * ((data / fold) @ R) / eff
        out.append(t)
    return np.array(out)


@functools.lru_cache(maxsize=1)
def world():
    """Everything the script works out, from the same seed in the same order (the
    pseudo-experiments' draws come next: :func:`toys`). Arrays: ``edges``, ``truth``, the
    simulation's ``t`` (true), ``found`` and ``r`` (measured), ``H``, ``n``, ``R``, ``eff``,
    ``expected``, ``data``, ``back``, ``naive``, ``U``, ``s``, ``Vt``, ``noise``, ``blown``,
    ``prior``; ``rng``: the generator's state after the data."""
    rng = np.random.default_rng(SEED)
    e = edges()
    tr = truth()
    t = rng.exponential(SIM_SLOPE, SIMULATED)
    t = t[t < HI]
    p = FOUND_LO + FOUND_PER_GEV * t
    found = rng.random(t.size) < p
    r = t + rng.normal(0, RESOLUTION, t.size)
    H = np.histogram2d(r[found], t[found], [e, e])[0]
    n = np.histogram(t, e)[0]
    R = H / n
    expected = R @ tr
    data = rng.poisson(expected)
    back = np.linalg.solve(R, expected)
    naive = np.linalg.solve(R, data)
    U, s, Vt = np.linalg.svd(R)
    noise = U.T @ (data - expected)
    blown = noise / s
    eff = R.sum(0)
    prior = n * data.sum() / (R @ n).sum()
    return dict(edges=e, truth=tr, t=t, found=found, r=r, H=H, n=n, R=R, eff=eff,
                expected=expected, data=data, back=back, naive=naive, U=U, s=s, Vt=Vt,
                noise=noise, blown=blown, prior=prior, rng_state=rng.bit_generator.state)


def rms(v, axis=-1):
    return np.sqrt(np.mean(np.square(v), axis=axis))


@functools.lru_cache(maxsize=1)
def toys():
    """The 2,000 pseudo-experiments (the script's draws, after the data's): per number of
    iterations 0 to :data:`UPTO`, ``bias`` and ``spread`` per bin ``[UPTO + 1, 10]`` and
    their root mean squares over the bins; ``error`` = sqrt(bias^2 + spread^2) (root mean
    square over the bins of the unfolded answer's distance from the truth); the same for
    the inversion (``inverse_bias``, ``inverse_spread``)."""
    w = world()
    rng = np.random.default_rng(SEED)
    rng.bit_generator.state = w["rng_state"]
    draws = rng.poisson(w["expected"], (TOYS, BINS))
    its = iterates(draws, w["prior"], w["R"], w["eff"])          # [UPTO + 1, TOYS, 10]
    bias = its.mean(1) - w["truth"]
    spread = its.std(1)
    inv = np.linalg.solve(w["R"], draws.T).T
    out = dict(draws=draws, bias=bias, spread=spread, bias_rms=rms(bias), spread_rms=rms(spread),
               inverse_bias=inv.mean(0) - w["truth"], inverse_spread=inv.std(0),
               negative=float((inv < 0).any(1).mean()),
               mean_squared=np.mean(np.square(its - w["truth"]), axis=(1, 2)))
    out["error_rms"] = np.sqrt(out["bias_rms"] ** 2 + out["spread_rms"] ** 2)
    out["inverse_bias_rms"] = float(rms(out["inverse_bias"]))
    out["inverse_spread_rms"] = float(rms(out["inverse_spread"]))
    return out


def inverse_spread():
    """The inversion's spread per bin, worked out: the square root of the diagonal of
    R^-1 diag(expected) R^-T (Poisson variance = the expected count)."""
    w = world()
    Ri = np.linalg.inv(w["R"])
    return np.sqrt(np.diag(Ri @ np.diag(w["expected"]) @ Ri.T))


def noise_size():
    """The expected size of the noise along each U[:, k]: sqrt(sum_i U[i, k]^2 expected_i)."""
    w = world()
    return np.sqrt((w["U"] ** 2 * w["expected"][:, None]).sum(0))


def summary():
    w, s = world(), toys()
    best = int(np.argmin(s["error_rms"]))
    return dict(truth_total=float(w["truth"].sum()), simulated_in_range=int(w["t"].size),
                found=int(w["found"].sum()), counted=int(w["H"].sum()),
                expected_total=float(w["expected"].sum()), data_total=int(w["data"].sum()),
                back_off=float(np.abs(w["back"] - w["truth"]).max()),
                naive=w["naive"].tolist(), s=w["s"].tolist(),
                condition=float(w["s"][0] / w["s"][-1]),
                best=best, best_error=float(s["error_rms"][best]),
                inverse_spread=s["inverse_spread_rms"],
                inverse_spread_exact=float(rms(inverse_spread())),
                data_noise=float(rms(np.sqrt(w["expected"]))))


def its(k):
    return "iteration" if k == 1 else "iterations"


def printed():
    """The script's printed lines, worked out here the same way (its ``printed()``)."""
    w, s = world(), toys()
    out = ["truth total %d; 50-60 GeV: %d" % (round(w["truth"].sum()), round(w["truth"][5])),
           "simulated %d, counted %d" % (SIMULATED, w["H"].sum()),
           "expected %d, data %d" % (round(w["expected"].sum()), w["data"].sum()),
           "back - truth: %.1e" % np.abs(w["back"] - w["truth"]).max(),
           "naive: %d to %d" % (round(w["naive"].min()), round(w["naive"].max())),
           "s: %.3f down to %.4f" % (w["s"][0], w["s"][-1])]
    for k in (0, 1, 4, 10):
        out.append("%d %s: 50-60 GeV holds %d" % (k, its(k), round(
            bayes(w["data"], w["prior"], w["R"], w["eff"], k)[5])))
    for k in SCRIPT_ITERATIONS:
        out.append("%d %s: bias %.0f, spread %.0f" % (k, its(k), s["bias_rms"][k],
                                                             s["spread_rms"][k]))
    out.append("R inverted: bias %.0f, spread %.0f" % (s["inverse_bias_rms"],
                                                      s["inverse_spread_rms"]))
    return out
