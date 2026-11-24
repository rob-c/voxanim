"""Error bars on a network: the fit of "Minuit and SGD" trained by SGD many times over, for the
animation "Error bars on a network" of "Blocks and code" (catalogue 272):
``examples/ensembles.py`` in numpy.

The data, the model and the gradient are :mod:`voxanim.study_minuit`'s (40 made-up points of
y = a exp(-x / tau) + b, a = 5, tau = 2.5, b = 1, noise 0.3, seed 1). One SGD run
(:func:`train`) is the script's ``train``: 3,000 steps from (1, 1, 0), learning rate 0.005, each
on 8 of the run's 40 points (``permutation(40)[:8]`` of its own seeded generator); its answer is
where it ends. The runs are computed together, step by step (:func:`train_many`), which is the
script's arithmetic run by run (``tests/test_study_ensembles.py`` checks it against the script).

* :func:`seeds`: 100 runs on all 40 points, seeds 0-99: the answers differ only by the luck of
  the batches, the optimiser's noise.
* :func:`boots`: 100 runs on bootstrap resamples (40 picks of the 40 points, repeats allowed,
  from ``default_rng(2).integers(0, 40, 40)`` in turn), seeds 100-199.
* :func:`compare`: the spread of each (standard deviation, ``ddof=1``) and covariance, against
  Minuit's HESSE errors and covariance (``voxanim/minuit_calls.json``, recorded by iminuit).

Two checks that are not on the script: :func:`exact_fits` refits each resample exactly
(Gauss-Newton), so the SGD answers can be compared with the resamples' own minima; and
:func:`toys` refits 1,000 new made-up datasets (new noise on the curve of Minuit's minimum: a
parametric bootstrap), the spread HESSE's errors describe.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_minuit as M

RUNS = 100
STEPS, BATCH, LR = 3000, 8, 0.005
START = (1.0, 1.0, 0.0)
SEEDS = tuple(range(0, RUNS))              # the seeds-only ensemble
BOOT_SEEDS = tuple(range(RUNS, 2 * RUNS))  # the bootstrap ensemble's SGD seeds
DRAW_SEED = 2                              # the resamples' generator
TOYS, TOY_SEED = 1000, 3                   # new made-up datasets, refitted exactly


def batches(seed, steps=STEPS):
    """``[steps, 8]``: the slots (0..39 of the run's points) each step looks at."""
    pick = np.random.default_rng(seed)
    return np.array([pick.permutation(M.N)[:BATCH] for _ in range(steps)])


@functools.lru_cache(maxsize=1)
def resamples():
    """``[RUNS, 40]``: each bootstrap run's 40 picks of the 40 points, in the script's order."""
    draw = np.random.default_rng(DRAW_SEED)
    return np.array([draw.integers(0, M.N, M.N) for _ in range(RUNS)])


def train_many(points, seeds, steps=STEPS):
    """SGD runs side by side: run ``r`` on points ``points[r]`` (``[R, 40]`` indices) with seed
    ``seeds[r]``. Returns the answers, ``[R, 3]`` (where each ends)."""
    x, y = M.data()
    points = np.asarray(points)
    idx = np.stack([np.take(points[r], batches(s, steps)) for r, s in enumerate(seeds)])
    p = np.tile(np.array(START, float), (len(seeds), 1))
    for t in range(steps):
        xi, yi = x[idx[:, t]], y[idx[:, t]]
        a, tau, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        e = np.exp(-xi / tau)
        r = (yi - (a * e + b)) / M.SIGMA
        g = np.stack([(-2 * r * e).mean(1), (-2 * r * a * e * xi / tau ** 2).mean(1),
                      (-2 * r).mean(1)], 1) / M.SIGMA
        p = p - LR * g
    return p


def train(points, seed):
    """One run: the script's ``train``."""
    return train_many(np.asarray(points)[None], [seed])[0]


@functools.lru_cache(maxsize=1)
def seeds():
    """``[100, 3]``: the answers of the runs with new seeds, all on the 40 points."""
    return train_many(np.tile(np.arange(M.N), (RUNS, 1)), SEEDS)


@functools.lru_cache(maxsize=1)
def boots():
    """``[100, 3]``: the answers of the runs on bootstrap resamples."""
    return train_many(resamples(), BOOT_SEEDS)


def gauss_newton(x, y, p0, iterations=30):
    """The exact least-squares minimum of the model on points ``(x, y)`` (``[R, n]`` each),
    from ``p0`` (``[R, 3]``)."""
    p = np.array(p0, float)
    for _ in range(iterations):
        a, tau, b = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        e = np.exp(-x / tau)
        r = (y - (a * e + b)) / M.SIGMA
        J = np.stack([e, a * e * x / tau ** 2, np.ones_like(e)], 2) / M.SIGMA
        p = p + np.stack([np.linalg.lstsq(J[i], r[i], rcond=None)[0] for i in range(len(p))])
    return p


@functools.lru_cache(maxsize=1)
def exact_fits():
    """``[100, 3]``: each resample's own minimum (Gauss-Newton), to compare SGD's answers with."""
    x, y = M.data()
    pts = resamples()
    return gauss_newton(x[pts], y[pts], np.tile(minuit()["values"], (RUNS, 1)))


@functools.lru_cache(maxsize=1)
def toys():
    """``[1000, 3]``: 1,000 new made-up datasets (new noise of 0.3 on the curve of Minuit's
    minimum), each refitted exactly: the spread HESSE's errors describe."""
    x, _ = M.data()
    v = minuit()["values"]
    rng = np.random.default_rng(TOY_SEED)
    ys = M.model(x, *v) + rng.normal(0, M.SIGMA, (TOYS, M.N))
    return gauss_newton(np.tile(x, (TOYS, 1)), ys, np.tile(v, (TOYS, 1)))


def minuit():
    """Minuit's minimum, HESSE's errors and covariance (the record of iminuit's run)."""
    r = M.record()
    return dict(values=np.array(r["values"]), errors=np.array(r["errors"]),
                covariance=np.array(r["covariance"]), fval=r["fval"], iminuit=r["iminuit"])


def best_a(tau, b):
    """The a that minimises chi-square at fixed ``tau`` and ``b`` (the model is linear in a)."""
    x, y = M.data()
    e = np.exp(-x / tau)
    return float(((y - b) * e).sum() / (e * e).sum())


def ellipse(mean, cov, n=73):
    """``[n, 2]``: the ellipse of a 2 x 2 covariance round ``mean``, the curve whose shadow on
    each axis reaches one standard deviation each way (for HESSE's, chi-square + 1 with the
    third parameter refitted, in the quadratic approximation)."""
    L = np.linalg.cholesky(np.asarray(cov, float))
    t = np.linspace(0.0, 2 * np.pi, n)
    return np.asarray(mean, float) + (L @ np.stack([np.cos(t), np.sin(t)])).T


def profile_chi2(tau, b):
    """Chi-square at ``(tau, b)`` with a refitted."""
    return M.chi2((best_a(tau, b), tau, b))


def corr(cov, i, j):
    cov = np.asarray(cov)
    return float(cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]))


@functools.lru_cache(maxsize=1)
def compare():
    """The two ensembles against HESSE: spreads, covariances, correlations, the resamples'
    picks, how far each SGD answer is from its resample's own minimum, and the toys."""
    s, bt, mn = seeds(), boots(), minuit()
    ex, ty = exact_fits(), toys()
    pts = resamples()
    counts = np.array([np.bincount(p, minlength=M.N) for p in pts])
    cov_b = np.cov(bt.T)
    plane = (1, 2)                                     # tau, b
    sub = np.ix_(plane, plane)
    ell = ellipse(mn["values"][list(plane)], mn["covariance"][sub])
    prof = [profile_chi2(t, b) - mn["fval"] for t, b in ell]
    return dict(
        seeds=s, boots=bt, seeds_mean=s.mean(0), boots_mean=bt.mean(0),
        seeds_spread=s.std(0, ddof=1), boots_spread=bt.std(0, ddof=1),
        boots_cov=cov_b, seeds_cov=np.cov(s.T),
        hesse=mn["errors"], hesse_cov=mn["covariance"], values=mn["values"],
        ratio=bt.std(0, ddof=1) / mn["errors"],
        corr_tau_b_boots=corr(cov_b, 1, 2), corr_tau_b_hesse=corr(mn["covariance"], 1, 2),
        corr_a_b_boots=corr(cov_b, 0, 2), corr_a_b_hesse=corr(mn["covariance"], 0, 2),
        counts=counts, distinct=(counts > 0).sum(1),
        sgd_vs_exact=np.abs(bt - ex).max(0), exact_spread=ex.std(0, ddof=1),
        toys_spread=ty.std(0, ddof=1), hesse_ellipse=ell,
        boots_ellipse=ellipse(bt.mean(0)[list(plane)], cov_b[sub]),
        profile_on_ellipse=(float(min(prof)), float(max(prof))))


def printed():
    """What the script prints for the seeds, the resamples and HESSE, as numpy prints it."""
    c = compare()
    return dict(seeds=str(c["seeds_spread"].round(3)), boots=str(c["boots_spread"].round(2)),
                cov=str(c["boots_cov"].round(3)), hesse=str(np.round(c["hesse"], 2)))
