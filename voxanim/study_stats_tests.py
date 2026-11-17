"""The experiments behind animations 141-144 (see :func:`voxanim.next_catalog.redesigned`):
hypothesis tests, their two kinds of error, p-values, the best test statistic and the
look-elsewhere effect.

Statistics is taught here the way physics does it: with pseudo-experiments (seeded draws
from a stated model) beside the exact answer wherever one exists. Each study keeps the
two apart:

* ``type-i-and-type-ii-errors``: a counting experiment, ``b = 10`` background events
  expected (H0) against ``s + b = 18`` (H1). The error rates alpha and beta of every
  threshold are exact Poisson sums; 5,000 pseudo-experiments under each hypothesis fill
  the table of outcomes; the power at alpha <= 5% is followed as the data grow.
* ``p-values``: ten measurements against a prediction, chi-square as the test statistic.
  One observed data set (drawn from a stated model with a real effect), the distribution
  of chi-square under H0 from 20,000 pseudo-experiments against the exact curve, the
  p-values of null experiments, and a population in which one experiment in ten has a
  real effect.
* ``neyman-pearson``: two measured values per event, two simple hypotheses. The
  likelihood ratio against one variable alone and the plain sum: exact ROC curves
  (every statistic is Gaussian) and 20,000 pseudo-events per hypothesis.
* ``look-elsewhere-effect``: a falling background spectrum of 60 bins scanned with a
  four-bin window; 100,000 background-only pseudo-experiments give the distribution of
  the largest local significance.

No scipy: the normal tail is ``math.erfc``, its inverse a rational approximation polished
by Newton steps, Poisson and chi-square tails are finite sums. Everything here builds in
about a second, so nothing is cached on disk and nothing needs a dataset.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .study import Panel, Study

EXACT = "exact"
PSEUDO = "pseudo-experiments"
WHERE = ("every draw comes from numpy.random.default_rng(seed) with the seed recorded "
         "here; the same seed gives the same pseudo-experiments")

_erfc = np.frompyfunc(math.erfc, 1, 1)


# ====================================================================== special functions
def norm_sf(z):
    """The standard normal's upper tail: the chance of a value above ``z``."""
    return 0.5 * np.asarray(_erfc(np.asarray(z, float) / math.sqrt(2.0)), float)


def norm_pdf(z):
    z = np.asarray(z, float)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_isf(p):
    """``Z`` with upper tail ``p`` (one-sided): Acklam's rational approximation, then
    two Newton steps on :func:`norm_sf`. ``p`` in (0, 1)."""
    p = np.asarray(p, float)
    tail = np.clip(np.minimum(p, 1.0 - p), 1e-300, 0.5)      # the smaller tail
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    r = np.sqrt(-2.0 * np.log(tail))
    far = -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / (
        (((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0)
    u = 0.5 - tail
    s = u * u
    mid = (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * u / (
        ((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0)
    z = np.where(tail < 0.02425, far, mid)             # |Z| of the smaller tail
    for _ in range(2):
        z = z + (norm_sf(z) - tail) / np.maximum(norm_pdf(z), 1e-300)
    return np.where(p <= 0.5, z, -z)


def poisson_pmf(mu, nmax):
    """``P(n)`` for ``n = 0..nmax`` counts when ``mu`` are expected."""
    n = np.arange(nmax + 1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, nmax + 1)))])
    return np.exp(n * math.log(mu) - mu - logfact)


def poisson_sf(mu, nmax):
    """``P(n or more)`` for ``n = 0..nmax`` (summed from the far tail, so small tails
    keep their precision)."""
    pmf = poisson_pmf(mu, nmax + 60 + int(10 * math.sqrt(mu)))
    return np.minimum(np.cumsum(pmf[::-1])[::-1][:nmax + 1], 1.0)


def chi2_sf(x, k):
    """The chi-square upper tail for an even number ``k`` of degrees of freedom:
    ``exp(-x/2) * sum_{j < k/2} (x/2)^j / j!`` (exact, a finite sum)."""
    if k % 2 or k <= 0:
        raise ValueError("even degrees of freedom only")
    h = np.asarray(x, float) / 2.0
    term = np.ones_like(h)
    total = np.ones_like(h)
    for j in range(1, k // 2):
        term = term * h / j
        total = total + term
    return np.exp(-h) * total


def chi2_pdf(x, k):
    """The chi-square density, ``x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2))``."""
    x = np.asarray(x, float)
    return np.exp((k / 2 - 1) * np.log(np.maximum(x, 1e-300)) - x / 2
                  - (k / 2) * math.log(2.0) - math.lgamma(k / 2))


def ncx2_sf(x, k, lam, terms=120):
    """Upper tail of the non-central chi-square (``k`` even, non-centrality ``lam``): a
    Poisson(``lam / 2``) mixture of central chi-squares with ``k + 2j`` degrees."""
    w = poisson_pmf(lam / 2.0, terms)
    return float(sum(w[j] * chi2_sf(x, k + 2 * j) for j in range(terms + 1)))


def bisect(f, lo, hi, steps=80):
    """The root of an increasing ``f`` between ``lo`` and ``hi``."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ====================================================================== 141: two errors
ERR_SEED = 141
ERR_B, ERR_S = 10.0, 8.0          # expected background and signal events in one run
ERR_ALPHA = 0.05                  # the type-I rate a threshold may not exceed
ERR_N = 5000                      # pseudo-experiments under each hypothesis
ERR_NMAX = 45                     # counts drawn on the axis
ERR_RUNS = tuple(0.25 * k for k in range(1, 25))    # amounts of data: 0.25 .. 6 runs
ERR_SHOWN = (1.0, 2.0, 4.0)
ERR_POWER_N = 10000               # pseudo-experiments per amount of data


def threshold_for(b, alpha, nmax=None):
    """The smallest count ``n_c`` with ``P(n >= n_c | b) <= alpha``, and that chance."""
    nmax = nmax or int(b + 12 * math.sqrt(b) + 30)
    sf = poisson_sf(b, nmax)
    n_c = int(np.flatnonzero(sf <= alpha)[0])
    return n_c, float(sf[n_c])


@functools.lru_cache(maxsize=1)
def errors_data():
    """Everything ``type-i-and-type-ii-errors`` measures and draws."""
    rng = np.random.default_rng(ERR_SEED)
    b, sb = ERR_B, ERR_B + ERR_S
    n = np.arange(ERR_NMAX + 1)
    pmf0, pmf1 = poisson_pmf(b, ERR_NMAX), poisson_pmf(sb, ERR_NMAX)
    alpha, power = poisson_sf(b, ERR_NMAX), poisson_sf(sb, ERR_NMAX)
    n_c, a_c = threshold_for(b, ERR_ALPHA)
    h0 = rng.poisson(b, ERR_N)                        # H0 true: background only
    h1_b = rng.poisson(b, ERR_N)                      # H1 true: background ...
    h1_s = rng.poisson(ERR_S, ERR_N)                  # ... plus signal
    h1 = h1_b + h1_s
    table = np.array([[(h0 < n_c).sum(), (h0 >= n_c).sum()],
                      [(h1 < n_c).sum(), (h1 >= n_c).sum()]])
    roc_mc = np.array([[(h0 >= c).mean(), (h1 >= c).mean()] for c in n])
    runs = np.array(ERR_RUNS)
    cuts, a_run, p_run, p_mc, a_mc = [], [], [], [], []
    for t in runs:
        c, a = threshold_for(b * t, ERR_ALPHA)
        cuts.append(c)
        a_run.append(a)
        p_run.append(float(poisson_sf(sb * t, c)[c]))
        p_mc.append(float((rng.poisson(sb * t, ERR_POWER_N) >= c).mean()))
        a_mc.append(float((rng.poisson(b * t, ERR_POWER_N) >= c).mean()))
    return dict(
        n=n, pmf0=pmf0, pmf1=pmf1, alpha=alpha, power=power, beta=1.0 - power,
        n_c=n_c, alpha_c=a_c, beta_c=float(1.0 - power[n_c]), h0=h0, h1=h1, h1_b=h1_b,
        h1_s=h1_s, table=table, roc_mc=roc_mc, runs=runs, cuts=np.array(cuts),
        alpha_run=np.array(a_run), power_run=np.array(p_run), power_mc=np.array(p_mc),
        alpha_mc=np.array(a_mc),
    )


def type_i_and_type_ii_errors():
    d = errors_data()
    n_c = d["n_c"]
    t = d["table"]
    return Study(
        "Two ways for a test to be wrong",
        "a counting experiment: 10 background events expected, 18 if the signal exists",
        (
            Panel("The count under each hypothesis",
                  "Poisson probabilities of n events for b = 10 (H0) and s + b = 18 (H1)",
                  "curve", dict(x=d["n"], ys=np.array([d["pmf0"], d["pmf1"]]),
                                labels=["H0: b = 10", "H1: s + b = 18"]),
                  ("exact",)),
            Panel("The two error rates of every threshold",
                  "reject H0 when n >= n_c: alpha is the chance under H0, beta the chance "
                  "of keeping H0 under H1",
                  "curve", dict(x=d["n"], ys=np.array([d["alpha"], d["beta"], d["power"]]),
                                labels=["alpha (type I)", "beta (type II)", "power"],
                                measured=d["roc_mc"]),
                  ("exact Poisson sums; measured: the same from the pseudo-experiments",)),
            Panel("The four outcomes, counted",
                  "%d pseudo-experiments with H0 true and %d with H1 true, threshold n >= %d"
                  % (ERR_N, ERR_N, n_c),
                  "matrix", dict(values=t.astype(float),
                                 exact=np.array([[1 - d["alpha_c"], d["alpha_c"]],
                                                 [d["beta_c"], 1 - d["beta_c"]]]) * ERR_N,
                                 rows=["H0 true", "H1 true"],
                                 columns=["keep H0", "reject H0"]),
                  ("pseudo-experiments, seed %d" % ERR_SEED,)),
            Panel("Power as the data grow",
                  "power at alpha <= 5% against the amount of data (1 = one run)",
                  "curve", dict(x=d["runs"], ys=np.array([d["power_run"], d["power_mc"]]),
                                labels=["exact", "pseudo-experiments"],
                                thresholds=d["cuts"], alpha=d["alpha_run"]),
                  ("the threshold is re-chosen for each amount of data; a count is a whole "
                   "number, so alpha lies below 5% by a varying amount and the curve has "
                   "steps",)),
        ),
        dict(
            seeds=[ERR_SEED], model="n ~ Poisson(10) under H0, Poisson(10) + Poisson(8) "
            "under H1; with t runs of data both means are multiplied by t",
            b=ERR_B, s=ERR_S, threshold=n_c, alpha=d["alpha_c"], beta=d["beta_c"],
            power=1.0 - d["beta_c"], pseudo_experiments=ERR_N,
            measured_alpha=float(t[0, 1] / ERR_N), measured_beta=float(t[1, 0] / ERR_N),
            power_pseudo_experiments=ERR_POWER_N,
            evidence={"exact": "Poisson sums: every alpha, beta, power and the ROC curve",
                      "pseudo-experiments": "the table of outcomes and the points beside "
                      "the exact power curve"},
            where_the_seed_enters=WHERE,
            limitations="one pair of simple hypotheses with known b and s; no systematic "
            "uncertainty. Counts are whole numbers, so no threshold has alpha exactly 5%.",
            designs_tried=["b = 10, s = 8, threshold from alpha <= 5% (the only design run)"],
        ),
    )


# ====================================================================== 142: p-values
PV_SEED = 142
PV_N = 10                         # measurements
PV_MU, PV_SIGMA = 20.0, 2.0       # H0's prediction for each, and each one's error
PV_RAISED, PV_SHIFT = (3, 4, 5), 2.0      # the truth: points 4-6 lie 2 sigma higher
PV_TOYS = 20000                   # pseudo-experiments under H0
PV_PRIOR = 0.10                   # share of experiments with a real effect
PV_POP = 20000                    # experiments in the population
PV_GRID = 1000                    # of them drawn on screen
PV_CUT = 0.05
PV_BINS = 40                      # chi-square histogram: 0..40 in steps of 1


def pv_truth():
    truth = np.full(PV_N, PV_MU)
    truth[list(PV_RAISED)] += PV_SHIFT * PV_SIGMA
    return truth


def chi2_of(y):
    """``sum ((y - mu) / sigma)^2`` over the last axis."""
    return (((np.asarray(y, float) - PV_MU) / PV_SIGMA) ** 2).sum(-1)


@functools.lru_cache(maxsize=1)
def pvalue_data():
    """Everything ``p-values`` measures and draws."""
    rng = np.random.default_rng(PV_SEED)
    truth = pv_truth()
    # the observed data set, recorded to one decimal so its arithmetic can be followed
    y = np.round(rng.normal(truth, PV_SIGMA), 1)
    pulls = (y - PV_MU) / PV_SIGMA
    obs = float((pulls ** 2).sum())
    null_y = rng.normal(PV_MU, PV_SIGMA, (PV_TOYS, PV_N))
    null = chi2_of(null_y)
    p_exact = float(chi2_sf(obs, PV_N))
    tail = int((null >= obs).sum())
    p_null = chi2_sf(null, PV_N)
    edges = np.arange(PV_BINS + 1.0)
    hist = np.histogram(null, edges)[0]
    over = int((null >= PV_BINS).sum())
    lam = len(PV_RAISED) * PV_SHIFT ** 2
    crit = bisect(lambda x: PV_CUT - float(chi2_sf(x, PV_N)), 0.0, 100.0)
    power = ncx2_sf(crit, PV_N, lam)
    # a population: one experiment in ten measures the stated effect, the rest nothing
    n_eff = int(round(PV_PRIOR * PV_POP))
    pop_null = chi2_sf(chi2_of(rng.normal(PV_MU, PV_SIGMA, (PV_POP - n_eff, PV_N))), PV_N)
    pop_eff = chi2_sf(chi2_of(rng.normal(truth, PV_SIGMA, (n_eff, PV_N))), PV_N)
    g_eff = int(round(PV_PRIOR * PV_GRID))
    grid_p = np.concatenate([pop_null[:PV_GRID - g_eff], pop_eff[:g_eff]])
    grid_real = np.arange(PV_GRID) >= PV_GRID - g_eff
    order = rng.permutation(PV_GRID)                  # where each sits on screen
    false_exact = (1 - PV_PRIOR) * PV_CUT / ((1 - PV_PRIOR) * PV_CUT + PV_PRIOR * power)

    def counts(p0, p1):
        return np.array([[(p0 >= PV_CUT).sum(), (p0 < PV_CUT).sum()],
                         [(p1 >= PV_CUT).sum(), (p1 < PV_CUT).sum()]])
    return dict(
        truth=truth, y=y, pulls=pulls, obs=obs, null=null, null_y=null_y[:40],
        p_exact=p_exact, tail=tail, p_mc=tail / PV_TOYS, z=float(norm_isf(p_exact)),
        p_null=p_null, p_hist=np.histogram(p_null, np.linspace(0, 1, 21))[0],
        below=float((p_null < PV_CUT).mean()), edges=edges, hist=hist, over=over,
        lam=lam, crit=crit, power=power, power_mc=float((pop_eff < PV_CUT).mean()),
        pop=counts(pop_null, pop_eff), grid=counts(grid_p[~grid_real], grid_p[grid_real]),
        grid_p=grid_p[order], grid_real=grid_real[order], false_exact=false_exact,
        quantiles=np.array([bisect(lambda x, q=q: q - float(chi2_sf(x, PV_N)), 0.0, 200.0)
                            for q in np.arange(0.95, 0.0, -0.05)]),
    )


def false_share(table):
    """Of the results with ``p < 0.05``, the share that came from no effect."""
    return float(table[0, 1] / (table[0, 1] + table[1, 1]))


def p_values():
    d = pvalue_data()
    centres = 0.5 * (d["edges"][:-1] + d["edges"][1:])
    return Study(
        "What a p-value is, and what it is not",
        "ten measurements against a prediction; chi-square as the test statistic",
        (
            Panel("The observed pulls",
                  "(measurement - prediction) / sigma for the ten points; their squares "
                  "sum to chi-square",
                  "bars", dict(values=d["pulls"], labels=["%d" % (i + 1) for i in range(PV_N)],
                               measured=d["y"], truth=d["truth"]),
                  ("pseudo-data from the stated truth, seed %d, recorded to 0.1" % PV_SEED,)),
            Panel("Chi-square if H0 is true",
                  "%d pseudo-experiments per unit of chi-square, and the exact chi-square "
                  "curve for 10 degrees of freedom" % PV_TOYS,
                  "curve", dict(x=centres,
                                ys=np.array([d["hist"] / PV_TOYS, chi2_pdf(centres, PV_N)]),
                                labels=["pseudo-experiments", "exact"],
                                observed=np.array(d["obs"])),
                  ("the p-value is the share at or beyond the observed value",)),
            Panel("p-values of null experiments",
                  "the %d null pseudo-experiments' p-values in 20 equal bins" % PV_TOYS,
                  "bars", dict(values=d["p_hist"].astype(float),
                               labels=["%.2f" % v for v in np.linspace(0.05, 1, 20)]),
                  ("uniform: each bin expects %d" % (PV_TOYS // 20),)),
            Panel("Who has p < 0.05",
                  "%d experiments, %d%% with the real effect: rows no effect / effect, "
                  "columns p >= 0.05 / p < 0.05" % (PV_POP, round(100 * PV_PRIOR)),
                  "matrix", dict(values=d["pop"].astype(float),
                                 on_screen=d["grid"].astype(float),
                                 rows=["no effect", "real effect"],
                                 columns=["p >= 0.05", "p < 0.05"]),
                  ("the share of false alarms among p < 0.05 depends on the prior share "
                   "of real effects and on the power",)),
        ),
        dict(
            seeds=[PV_SEED], model="y_i ~ Normal(mu_i, 2.0), ten points; H0: every mu_i = "
            "20; the truth behind the observed data and behind 'real effects': points 4, "
            "5, 6 at 24 (2 sigma higher)",
            chi2_observed=d["obs"], p_exact=d["p_exact"], p_pseudo=d["p_mc"],
            tail_count=d["tail"], z=d["z"], pseudo_experiments=PV_TOYS,
            share_below_005=d["below"], prior=PV_PRIOR, population=PV_POP,
            power_exact=d["power"], power_pseudo=d["power_mc"],
            false_alarm_share_exact=d["false_exact"],
            false_alarm_share_pseudo=false_share(d["pop"]),
            false_alarm_share_on_screen=false_share(d["grid"]),
            evidence={"exact": "the chi-square curve, the p-value, Z, the power (a "
                      "non-central chi-square) and the expected share of false alarms",
                      "pseudo-experiments": "the histogram under H0, the p-value as a "
                      "count, the histogram of p-values, the population"},
            where_the_seed_enters=WHERE,
            limitations="the observed data set is one draw (seed %d, the first tried): "
            "another seed gives another p-value. The 10%% prior share of real effects is "
            "a stated assumption, not a measurement; the false-alarm share moves with it."
            % PV_SEED,
            designs_tried=["ten points, sigma 2, three points raised by 2 sigma, seed 142 "
                           "(the only design run)"],
        ),
    )


# ====================================================================== 143: Neyman-Pearson
NP_SEED = 143
NP_SHIFT = (1.5, 2.4)             # H1's means of x and y (H0: both 0)
NP_SIGMA = (1.0, 2.0)             # the resolution of x and of y, under both hypotheses
NP_N = 20000                      # pseudo-events under each hypothesis
NP_ALPHA = 0.05
NP_SHOWN = 500                    # events of each kind drawn in the plane
NP_ANGLES = tuple(range(-60, 91, 5))
NP_NAMES = ("x alone", "x + y", "likelihood ratio")


def np_weights():
    """The three statistics as weights on ``(x, y)``: x alone, the plain sum, and the
    log likelihood ratio ``sum shift_i x_i / sigma_i^2`` (plus a constant)."""
    lr = (NP_SHIFT[0] / NP_SIGMA[0] ** 2, NP_SHIFT[1] / NP_SIGMA[1] ** 2)
    return np.array([(1.0, 0.0), (1.0, 1.0), lr])


def np_constant():
    """The constant of the log likelihood ratio: ``-sum shift_i^2 / (2 sigma_i^2)``."""
    return -sum(m * m / (2 * s * s) for m, s in zip(NP_SHIFT, NP_SIGMA))


def log_likelihoods(xy):
    """``ln L0`` and ``ln L1`` of events ``(x, y)``: sums of two Gaussian log densities."""
    xy = np.asarray(xy, float)
    out = []
    for mean in ((0.0, 0.0), NP_SHIFT):
        z = (xy - np.array(mean)) / np.array(NP_SIGMA)
        out.append((-0.5 * z * z - np.log(np.array(NP_SIGMA) * math.sqrt(2 * math.pi))
                    ).sum(-1))
    return out[0], out[1]


def separation(w):
    """``d'`` of the linear statistic ``w . (x, y)``: the distance between its means
    under H1 and H0 in units of its standard deviation."""
    w = np.asarray(w, float)
    return float(w @ np.array(NP_SHIFT) / math.sqrt((w * w * np.array(NP_SIGMA) ** 2).sum()))


def exact_power(w, alpha):
    """Power of the test 'w . (x, y) above a threshold' at false-alarm rate ``alpha``:
    ``Phi(d' - z_alpha)``, exact because the statistic is Gaussian."""
    return norm_sf(norm_isf(alpha) - separation(w))


def exact_cut(w, alpha):
    """The threshold on ``w . (x, y)`` that H0 passes with chance ``alpha``."""
    w = np.asarray(w, float)
    return float(norm_isf(alpha) * math.sqrt((w * w * np.array(NP_SIGMA) ** 2).sum()))


@functools.lru_cache(maxsize=1)
def np_data():
    """Everything ``neyman-pearson`` measures and draws."""
    rng = np.random.default_rng(NP_SEED)
    sig = np.array(NP_SIGMA)
    e0 = rng.normal(0.0, 1.0, (NP_N, 2)) * sig
    e1 = rng.normal(0.0, 1.0, (NP_N, 2)) * sig + np.array(NP_SHIFT)
    w = np_weights()
    cuts = np.array([exact_cut(v, NP_ALPHA) for v in w])
    t0, t1 = e0 @ w.T, e1 @ w.T                       # (events, statistic)
    alphas = np.concatenate([np.arange(0.002, 0.02, 0.002), np.arange(0.02, 1.0001, 0.01)])
    roc_exact = np.array([exact_power(v, alphas) for v in w])
    srt = np.sort(t0, axis=0)[::-1]                   # the k-th largest H0 value
    k = np.maximum(np.round(alphas * NP_N).astype(int), 1)
    roc_mc = np.array([[(t1[:, j] > srt[kk - 1, j]).mean() for kk in k] for j in range(3)])
    # a monotonic function of the likelihood ratio: L1 / (L0 + L1), between 0 and 1
    c = np_constant()
    post0, post1 = (1.0 / (1.0 + np.exp(-(t[:, 2] + c))) for t in (t0, t1))
    post_cut = 1.0 / (1.0 + math.exp(-(cuts[2] + c)))
    ang = np.radians(np.array(NP_ANGLES, float))
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    ang_exact = np.array([float(exact_power(v, NP_ALPHA)) for v in dirs])
    ang_mc = np.array([float((e1 @ v > exact_cut(v, NP_ALPHA)).mean()) for v in dirs])
    fine = np.radians(np.linspace(-60, 90, 301))
    fine_power = np.array([float(exact_power((math.cos(a), math.sin(a)), NP_ALPHA))
                           for a in fine])
    l0, l1 = log_likelihoods(e1[:1])
    return dict(
        e0=e0, e1=e1, w=w, cuts=cuts, alphas=alphas, roc_exact=roc_exact, roc_mc=roc_mc,
        d=np.array([separation(v) for v in w]),
        power_exact=np.array([float(exact_power(v, NP_ALPHA)) for v in w]),
        power_mc=(t1 > cuts).mean(0), alpha_mc=(t0 > cuts).mean(0),
        post_power=float((post1 > post_cut).mean()), post_alpha=float((post0 > post_cut).mean()),
        post_same=bool(np.array_equal(post1 > post_cut, t1[:, 2] > cuts[2])),
        post_cut=post_cut, angles=np.array(NP_ANGLES, float), ang_exact=ang_exact,
        ang_mc=ang_mc, fine=np.degrees(fine), fine_power=fine_power,
        best_angle=float(np.degrees(fine[int(np.argmax(fine_power))])),
        lr_angle=float(np.degrees(math.atan2(w[2, 1], w[2, 0]))),
        event=e1[0], event_l0=float(np.exp(l0[0])), event_l1=float(np.exp(l1[0])),
        constant=c, t0_lr=t0[:, 2] + c, t1_lr=t1[:, 2] + c,
    )


def neyman_pearson():
    d = np_data()
    shown = np.concatenate([d["e0"][:NP_SHOWN], d["e1"][:NP_SHOWN]])
    return Study(
        "The likelihood ratio is the best test",
        "two measured values per event, two simple hypotheses, three test statistics",
        (
            Panel("Events under each hypothesis",
                  "x has resolution 1 and y resolution 2; H1 shifts x by 1.5 and y by 2.4",
                  "points", dict(xy=shown, classes=np.repeat([0, 1], NP_SHOWN)),
                  ("the first %d pseudo-events of each kind, seed %d" % (NP_SHOWN, NP_SEED),)),
            Panel("ROC curves of three statistics",
                  "power against false-alarm rate alpha: x alone, x + y, the likelihood "
                  "ratio (exact), and each measured on %d pseudo-events" % NP_N,
                  "curve", dict(x=d["alphas"], ys=np.concatenate([d["roc_exact"],
                                                                  d["roc_mc"]]),
                                labels=[n + " (exact)" for n in NP_NAMES]
                                + [n + " (pseudo-events)" for n in NP_NAMES]),
                  ("each statistic is Gaussian, so its ROC is Phi(d' - z_alpha)",)),
            Panel("Power at alpha = 5%",
                  "share of H1 pseudo-events passing each test's threshold; last: a "
                  "monotonic function of the likelihood ratio, L1 / (L0 + L1)",
                  "bars", dict(values=np.append(d["power_mc"], d["post_power"]),
                               labels=list(NP_NAMES) + ["L1 / (L0 + L1)"],
                               exact=np.append(d["power_exact"], d["power_exact"][2]),
                               alpha=np.append(d["alpha_mc"], d["post_alpha"])),
                  ("pseudo-events; thresholds from the exact alpha = 5%",)),
            Panel("Power of every direction in the plane",
                  "the statistic cos(a) x + sin(a) y at alpha = 5%, against the angle a",
                  "curve", dict(x=d["angles"], ys=np.array([d["ang_exact"], d["ang_mc"]]),
                                labels=["exact", "pseudo-events"]),
                  ("the peak is at the likelihood ratio's angle, %.1f degrees"
                   % d["lr_angle"],)),
        ),
        dict(
            seeds=[NP_SEED], model="H0: x ~ Normal(0, 1), y ~ Normal(0, 2); H1: x ~ "
            "Normal(1.5, 1), y ~ Normal(2.4, 2); independent",
            ln_ratio="1.5 x + 0.6 y - 1.845", pseudo_events=NP_N, alpha=NP_ALPHA,
            names=list(NP_NAMES), power_exact=[float(v) for v in d["power_exact"]],
            power_pseudo=[float(v) for v in d["power_mc"]],
            separation=[float(v) for v in d["d"]],
            monotonic_power=d["post_power"], monotonic_same_events=d["post_same"],
            best_angle=d["best_angle"], lr_angle=d["lr_angle"],
            evidence={"exact": "the likelihoods of the event, the thresholds, the ROC "
                      "curves and the power of every direction (Gaussian statistics)",
                      "pseudo-experiments": "the events drawn, the measured powers and "
                      "ROC points"},
            where_the_seed_enters=WHERE,
            limitations="simple hypotheses with known shapes; with equal covariances the "
            "likelihood ratio is a straight-line cut, so every competitor here is also a "
            "line. With unknown parameters there is in general no single best test.",
            designs_tried=["shift (1.5, 2.4), resolutions (1, 2), alpha 5% (the only "
                           "design run)"],
        ),
    )


# ====================================================================== 144: look elsewhere
LEE_SEED = 144
LEE_BINS = 60                     # 1 GeV bins from 100 to 160 GeV
LEE_M0, LEE_AMP, LEE_SLOPE = 100.0, 200.0, 30.0     # b(m) = 200 exp(-(m - 100) / 30)
LEE_WIDTH = 4                     # bins in the window that scans
LEE_TOYS = 100000
LEE_Z = 3.0
LEE_GRID = tuple(round(1.5 + 0.1 * k, 1) for k in range(31))        # local Z: 1.5 .. 4.5
LEE_TRACES = 8


def lee_background():
    """Expected background events in each bin (the bin centres' value of the curve)."""
    m = LEE_M0 + 0.5 + np.arange(LEE_BINS)
    return LEE_AMP * np.exp(-(m - LEE_M0) / LEE_SLOPE)


def window_sums(counts, width=LEE_WIDTH):
    """Sums over every run of ``width`` adjacent bins (last axis)."""
    c = np.concatenate([np.zeros(counts.shape[:-1] + (1,)), np.cumsum(counts, -1)], -1)
    return c[..., width:] - c[..., :-width]


@functools.lru_cache(maxsize=1)
def lee_tables():
    """Per window: its expected background, and the local p-value and Z of every count
    (exact Poisson tails; Z one-sided)."""
    bw = window_sums(lee_background())
    nmax = int(bw.max() + 12 * math.sqrt(bw.max()))
    p = np.array([poisson_sf(mu, nmax) for mu in bw])
    z = np.clip(norm_isf(np.clip(p, 1e-300, 1 - 1e-12)), -6.0, None)
    return bw, p, z


@functools.lru_cache(maxsize=1)
def lee_data():
    """Everything ``look-elsewhere-effect`` measures and draws."""
    rng = np.random.default_rng(LEE_SEED)
    b = lee_background()
    bw, ptab, ztab = lee_tables()
    nw = len(bw)
    counts = rng.poisson(b, (LEE_TOYS, LEE_BINS))
    sums = window_sums(counts).astype(np.int64)
    sums = np.minimum(sums, ztab.shape[1] - 1)
    z = ztab[np.arange(nw), sums]                     # (toys, windows)
    zmax, where = z.max(1), z.argmax(1)
    apart = np.arange(0, nw, LEE_WIDTH)               # windows that share no bin
    zmax_apart = z[:, apart].max(1)
    grid = np.array(LEE_GRID)
    p_glob = np.array([(zmax >= g).mean() for g in grid])
    n_glob = np.array([(zmax >= g).sum() for g in grid])
    p_apart = np.array([(zmax_apart >= g).mean() for g in grid])
    # exact chance that one window's Z reaches g (a count is a whole number: a little
    # below the normal tail)
    p_one = np.array([[ptab[w, int(np.flatnonzero(ztab[w] >= g)[0])] for g in grid]
                      for w in range(nw)])
    all_indep = 1.0 - np.prod(1.0 - p_one, 0)         # as if all 57 were independent
    apart_exact = 1.0 - np.prod(1.0 - p_one[apart], 0)       # exact for the 15 apart
    p_mean = p_one.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        n_eff = np.log1p(-p_glob) / np.log1p(-p_mean)
    first = int(np.flatnonzero(zmax >= LEE_Z)[0])
    w0 = int(where[first])
    i3 = int(np.flatnonzero(np.isclose(grid, LEE_Z))[0])
    edges = np.linspace(-3.0, 5.0, 81)
    i4 = int(np.flatnonzero(np.isclose(grid, 4.0))[0])
    # one number of independent places (the one measured at 3 sigma) used everywhere
    z_fit = norm_isf(-np.expm1(n_eff[i3] * np.log1p(-p_mean)))
    # 5 sigma was never reached: an estimate from the number of places measured at 4
    z5 = float(norm_isf(-math.expm1(n_eff[i4] * math.log1p(-float(norm_sf(5.0))))))
    return dict(
        z_fit=z_fit, z5_estimate=z5, i4=i4,
        b=b, bw=bw, first=first, counts=counts[first], z_first=z[first], window=w0,
        n_window=int(sums[first, w0]), b_window=float(bw[w0]),
        p_window=float(ptab[w0, sums[first, w0]]), z_window=float(zmax[first]),
        traces=z[:LEE_TRACES], trace_counts=counts[:LEE_TRACES], zmax=zmax,
        edges=edges, hist_max=np.histogram(zmax, edges)[0],
        hist_one=np.histogram(z, edges)[0] / float(nw), grid=grid, p_glob=p_glob,
        n_glob=n_glob, p_apart=p_apart, apart_exact=apart_exact, all_indep=all_indep,
        p_mean=p_mean, p_nominal=norm_sf(grid), n_eff=n_eff, i3=i3,
        median=float(np.median(zmax)), apart=apart,
        z_glob=norm_isf(np.clip(p_glob, 0.5 / LEE_TOYS, 1 - 1e-9)),
    )


def lee_at(d, z):
    """Index of local significance ``z`` on the grid."""
    return int(np.flatnonzero(np.isclose(d["grid"], z))[0])


def look_elsewhere_effect():
    d = lee_data()
    i3, i4 = d["i3"], lee_at(d, 4.0)
    centres = 0.5 * (d["edges"][:-1] + d["edges"][1:])
    width = d["edges"][1] - d["edges"][0]
    nw = len(d["bw"])
    return Study(
        "The look-elsewhere effect",
        "a background-only spectrum of 60 bins, scanned with a four-bin window",
        (
            Panel("One background-only spectrum",
                  "expected background and the counts of pseudo-experiment %d, the first "
                  "whose largest local excess passes 3 sigma" % (d["first"] + 1),
                  "curve", dict(x=LEE_M0 + 0.5 + np.arange(LEE_BINS),
                                ys=np.array([d["b"], d["counts"].astype(float)]),
                                labels=["expected background", "counts"]),
                  ("no signal was generated",)),
            Panel("Local significance of every window",
                  "Z of the count in each of the %d four-bin windows of that spectrum" % nw,
                  "curve", dict(x=np.arange(nw, dtype=float), ys=d["z_first"][None],
                                labels=["local Z"]),
                  ("exact Poisson p-value of each window, converted to Z (one-sided)",)),
            Panel("The largest local significance",
                  "its distribution over %d background-only pseudo-experiments, beside "
                  "that of a single window (all %d pooled)" % (LEE_TOYS, nw),
                  "curve", dict(x=centres,
                                ys=np.array([d["hist_max"], d["hist_one"]]) / LEE_TOYS / width,
                                labels=["largest of %d windows" % nw, "a single window"]),
                  ("median of the largest: %.2f sigma" % d["median"],)),
            Panel("Global against local",
                  "chance that the largest local Z reaches a value, against that value",
                  "curve", dict(x=d["grid"],
                                ys=np.array([np.maximum(d["p_glob"], 0.5 / LEE_TOYS),
                                             d["all_indep"], d["apart_exact"],
                                             d["p_nominal"]]),
                                labels=["measured", "%d independent windows" % nw,
                                        "%d windows that share no bin" % len(d["apart"]),
                                        "one window"],
                                counts=d["n_glob"], n_eff=np.nan_to_num(d["n_eff"]),
                                z_global=d["z_glob"], z_fit=d["z_fit"]),
                  ("measured: counts of pseudo-experiments; the others exact",)),
        ),
        dict(
            seeds=[LEE_SEED], model="n_i ~ Poisson(b_i), b_i = 200 exp(-(m_i - 100) / 30) "
            "in 60 one-GeV bins; no signal anywhere",
            windows=nw, width=LEE_WIDTH, pseudo_experiments=LEE_TOYS,
            shown_experiment=d["first"] + 1, shown_window=d["window"],
            shown_n=d["n_window"], shown_b=d["b_window"], shown_p=d["p_window"],
            shown_z=d["z_window"], p_local_3=float(d["p_nominal"][i3]),
            p_global_3=float(d["p_glob"][i3]), count_3=int(d["n_glob"][i3]),
            trials_factor_3=float(d["p_glob"][i3] / d["p_nominal"][i3]),
            n_eff_3=float(d["n_eff"][i3]), n_eff_4=float(d["n_eff"][i4]),
            z_global_3=float(d["z_glob"][i3]), z_global_4=float(d["z_glob"][i4]),
            count_4=int(d["n_glob"][i4]), median_max=d["median"],
            z_global_5_estimate=d["z5_estimate"],
            apart_measured_3=float(d["p_apart"][i3]),
            apart_exact_3=float(d["apart_exact"][i3]),
            evidence={"exact": "each window's p-value and Z (Poisson tails), and the "
                      "chance for independent windows",
                      "pseudo-experiments": "every spectrum, the largest local Z and how "
                      "often it passes a value"},
            where_the_seed_enters=WHERE,
            limitations="one background shape and one window width; the background is "
            "taken as known. Beyond about 4 sigma local the global rate rests on a few "
            "dozen pseudo-experiments, and 5 sigma was not reached at all.",
            designs_tried=["60 bins, window of 4 bins, 100,000 pseudo-experiments (the "
                           "only design run)"],
        ),
    )


STUDIES = {
    "type-i-and-type-ii-errors": type_i_and_type_ii_errors,
    "p-values": p_values,
    "neyman-pearson": neyman_pearson,
    "look-elsewhere-effect": look_elsewhere_effect,
}
