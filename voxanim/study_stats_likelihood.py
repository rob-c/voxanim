"""The experiments behind animations 128-132: parameter estimates, maximum-likelihood fits
and errors (see :func:`voxanim.next_catalog.redesigned`).

* ``likelihood-curve``: five decay times and an exponential of lifetime tau. The
  likelihood is the product of the five probability densities; its maximum is at the
  sample mean (exact). Repeating the five-event experiment gives the estimate's spread,
  drawn against its exact distribution.
* ``likelihood-intervals``: the same model. The interval where ln L is within 1/2 of its
  maximum, against the symmetric one from the curvature, for 2 to 200 events; how often
  each contains the true lifetime, by pseudo-experiments and exactly.
* ``error-propagation``: a ratio x / y of two measured numbers. A cloud of draws pushed
  through the ratio, against the linear formula with and without its correlation term,
  and for a denominator measured too poorly for a straight line to do.
* ``error-on-the-mean``: an experiment of N measurements repeated thousands of times:
  the spread of the means against sigma / sqrt(N), and the sample standard deviation
  with N and with N - 1 in the denominator.
* ``bootstrap-errors``: one dataset of 20 events resampled with replacement; the spread
  of the recomputed statistic against the true sampling spread from fresh datasets, for
  five statistics, one of which (the maximum) the bootstrap gets wrong.

Nothing here needs a dataset. Every toy dataset comes from a seeded
``numpy.random.default_rng`` and a model stated on screen; every repeat-experiment number
is given with the exact value wherever one exists. The few special functions needed
(the distribution of a sum of exponentials, the bias factor of a standard deviation) are
written here with numpy and :mod:`math`: scipy is not a dependency.
"""

from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
EXACT = "exact"
TOYS = "pseudo-experiments"
ONE_SIGMA = math.erf(1.0 / math.sqrt(2.0))        # 0.6827: a bell curve within one sigma


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-statslik-" + name, settings,
                                                    build).items()}


# ====================================================================== special functions
def gamma_cdf(n, x):
    """``P(G <= x)`` for ``G`` the sum of ``n`` unit exponentials (``n`` a whole number):
    ``1 - exp(-x) * sum over k < n of x^k / k!``, each term computed through its log."""
    n = int(n)
    x = np.maximum(np.atleast_1d(np.asarray(x, float)), 1e-300)
    k = np.arange(n)
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n)))]) if n > 1 else \
        np.zeros(1)
    terms = np.exp(k[None] * np.log(x[:, None]) - log_fact[None] - x[:, None])
    return 1.0 - terms.sum(1)


def gamma_pdf(x, n, scale):
    """Density of the sum of ``n`` exponentials of mean ``scale``."""
    z = np.maximum(np.asarray(x, float) / scale, 1e-300)
    return np.exp((n - 1) * np.log(z) - z - math.lgamma(n)) / scale


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((np.asarray(x, float) - mu) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi))


def c4(n):
    """The mean of the sample standard deviation (N - 1 in the denominator) of ``n``
    Gaussian measurements, in units of sigma: ``sqrt(2 / (n - 1)) G(n / 2) / G((n - 1) / 2)``
    with G the gamma function."""
    return math.sqrt(2.0 / (n - 1)) * math.exp(math.lgamma(n / 2.0) - math.lgamma((n - 1) / 2.0))


# ====================================================================== the exponential
def exp_density(t, tau):
    """Probability density of a decay at time ``t`` for lifetime ``tau``."""
    return np.exp(-np.asarray(t, float) / tau) / tau


def log_likelihood(times, taus):
    """``ln L(tau) = sum of ln f(t_i; tau) = -N ln tau - (sum of t) / tau``."""
    times, taus = np.asarray(times, float), np.asarray(taus, float)
    return -len(times) * np.log(taus) - times.sum() / taus


def half_unit_ratios(n):
    """Where an exponential's ln L has fallen by 1/2, as multiples of the best tau: the
    two roots ``r`` of ``ln r + 1/r - 1 = 1 / (2 n)`` (bisection). They depend on the
    number of events only."""
    target = 1.0 / (2.0 * n)

    def root(lo, hi, rising):
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            above = math.log(mid) + 1.0 / mid - 1.0 > target
            if above == rising:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)
    return root(1e-9, 1.0, False), root(1.0, 1e4, True)


# ====================================================================== 128: likelihood curve
LC_SEED = 128
LC_TAU = 2.2                      # microseconds: a muon at rest
LC_N = 5
LC_MORE = (5, 20, 80)             # the same run with more events
LC_TRIALS = (1.0, 4.0)            # the two lifetimes tried by hand
LC_GRID = np.linspace(0.6, 7.0, 513)
LC_TOYS = 20000
LC_TOY_SEEDS = (1281, 1282, 1283)
LC_EDGES = np.linspace(0.0, 6.0, 49)


def _likelihood_build():
    stream = np.round(np.random.default_rng(LC_SEED).exponential(LC_TAU, LC_MORE[-1]), 2)
    times = stream[:LC_N]
    best = times.mean()
    trials = np.array(LC_TRIALS + (best,))
    dens = np.stack([exp_density(times, t) for t in trials])
    lnl = log_likelihood(times, LC_GRID)
    more = np.stack([log_likelihood(stream[:n], LC_GRID)
                     - log_likelihood(stream[:n], stream[:n].mean()) for n in LC_MORE])
    means = np.stack([np.random.default_rng(s).exponential(LC_TAU, (LC_TOYS, LC_N)).mean(1)
                      for s in LC_TOY_SEEDS])
    hist = np.histogram(means[0], LC_EDGES)[0] / (LC_TOYS * np.diff(LC_EDGES))
    return {"stream": stream, "trials": trials, "densities": dens, "lnl": lnl, "more": more,
            "more_best": np.array([stream[:n].mean() for n in LC_MORE]),
            "toy_mean": means.mean(1), "toy_sd": means.std(1, ddof=1), "toy_hist": hist,
            "toy_first": means[0, :60]}


def likelihood_data():
    return _cache("curve", _likelihood_build, (_likelihood_build, log_likelihood, exp_density),
                  seed=LC_SEED, n=LC_N)


def likelihood_curve():
    d = likelihood_data()
    times = d["stream"][:LC_N]
    best = float(times.mean())
    lnl = d["lnl"]
    like = np.exp(lnl)
    peak = float(LC_GRID[int(np.argmax(lnl))])
    mids = 0.5 * (LC_EDGES[:-1] + LC_EDGES[1:])
    exact = gamma_pdf(mids, LC_N, LC_TAU / LC_N)
    sd_exact = LC_TAU / math.sqrt(LC_N)
    dens = d["densities"]
    return Study(
        "The likelihood curve",
        "Five decay times, an exponential, and the lifetime that makes them most probable",
        (
            Panel("each event's probability density", "density of the five measured times for "
                  "each lifetime tried", "matrix",
                  {"values": dens, "times": times, "trials": d["trials"],
                   "product": dens.prod(1), "logs": np.log(dens), "log_sum": np.log(dens).sum(1)}),
            Panel("the likelihood", "product of the five densities against the lifetime",
                  "curve", {"x": LC_GRID, "ys": like[None], "labels": ["likelihood"],
                            "peak": np.array(peak), "sample_mean": np.array(best)}),
            Panel("the log-likelihood", "ln L minus its maximum, with 5, 20 and 80 events of "
                  "the same run", "curve",
                  {"x": LC_GRID, "ys": d["more"], "labels": ["%d events" % n for n in LC_MORE],
                   "best": d["more_best"], "lnl": lnl}),
            Panel("repeating the experiment", "best lifetimes of %d pseudo-experiments against "
                  "their exact distribution" % LC_TOYS, "curve",
                  {"x": mids, "ys": np.stack([d["toy_hist"], exact]),
                   "labels": ["pseudo-experiments", "exact"], "first": d["toy_first"],
                   "mean": d["toy_mean"], "sd": d["toy_sd"]}),
        ),
        {
            "model": "decay times drawn from an exponential of lifetime tau = %.1f (the muon's, "
                     "in microseconds), recorded to 0.01" % LC_TAU,
            "true_tau": LC_TAU, "events": LC_N, "seed": LC_SEED,
            "seeds": [LC_SEED, *LC_TOY_SEEDS],
            "times": [float(t) for t in times], "best_tau": best, "grid_peak": peak,
            "grid_step": float(LC_GRID[1] - LC_GRID[0]),
            "toys": LC_TOYS, "toy_mean": [float(v) for v in d["toy_mean"]],
            "toy_sd": [float(v) for v in d["toy_sd"]], "sd_exact": sd_exact,
            "evidence": {"the data": TOYS + " (five times drawn from the model)",
                         "one lifetime": EXACT, "slide it": EXACT,
                         "sum of logs": EXACT, "repeat": TOYS + " against the exact "
                         "distribution (a sum of exponentials)"},
            "what_was_run": "ln L = -N ln tau - sum(t) / tau evaluated on %d lifetimes; %d "
                            "pseudo-experiments of %d events for each of %d seeds"
                            % (len(LC_GRID), LC_TOYS, LC_N, len(LC_TOY_SEEDS)),
            "designs_tried": ["one design, fixed before any number was looked at: the seed is "
                              "the catalogue number and the five times are its first five "
                              "draws"],
            "limitations": ["one five-event dataset is drawn; another seed gives another best "
                            "lifetime (the last phase shows how much)",
                            "the pseudo-experiments' times are not rounded to 0.01"],
        },
    )


# ====================================================================== 129: intervals
LI_SEED = 129
LI_TAU = 2.2
LI_NS = (2, 3, 5, 10, 20, 50, 100, 200)
LI_MANY, LI_FEW = 50, 5
LI_TOYS = 20000
LI_SEEDS = (1291, 1292, 1293)
LI_SHOWN = 30                     # intervals of single pseudo-experiments drawn
LI_POINTS = 241


def _coverage(hats, lo, hi, tau):
    """Share of intervals ``[hats * lo, hats * hi]`` that contain ``tau``, that lie wholly
    below it and wholly above it."""
    below, above = hats * hi < tau, hats * lo > tau
    return 1.0 - below.mean() - above.mean(), below.mean(), above.mean()


def interval_factors(n):
    """For ``n`` events, as multiples of the best tau: the ends of the half-unit interval
    and of the symmetric one, ``1 -+ 1 / sqrt(n)``."""
    lo, hi = half_unit_ratios(n)
    return (lo, hi), (1.0 - 1.0 / math.sqrt(n), 1.0 + 1.0 / math.sqrt(n))


def exact_coverage(n, lo, hi):
    """Exact chance that ``[best * lo, best * hi]`` contains the true lifetime, and the
    chances of lying wholly below and wholly above it: the best lifetime of ``n`` events
    is the true one times a sum of ``n`` unit exponentials over ``n``."""
    a, b = gamma_cdf(n, [n / hi, n / lo])
    return float(b - a), float(a), float(1.0 - b)


def _intervals_build():
    stream = np.round(np.random.default_rng(LI_SEED).exponential(LI_TAU, max(LI_NS)), 2)
    out = {"stream": stream}
    for name, n, span in (("many", LI_MANY, (0.62, 1.52)), ("few", LI_FEW, (0.40, 3.1))):
        times = stream[:n]
        best = times.mean()
        sigma = best / math.sqrt(n)
        grid = best * np.linspace(span[0], span[1], LI_POINTS)
        out[name + "_grid"] = grid
        out[name + "_dlnl"] = log_likelihood(times, grid) - log_likelihood(times, best)
        out[name + "_parabola"] = -0.5 * ((grid - best) / sigma) ** 2
    cov = np.zeros((len(LI_SEEDS), 2, len(LI_NS), 3))
    for s, seed in enumerate(LI_SEEDS):
        sums = np.cumsum(np.random.default_rng(seed).exponential(
            LI_TAU, (LI_TOYS, max(LI_NS))), axis=1)
        for j, n in enumerate(LI_NS):
            hats = sums[:, n - 1] / n
            for k, (lo, hi) in enumerate(interval_factors(n)):
                cov[s, k, j] = _coverage(hats, lo, hi, LI_TAU)
        if s == 0:
            out["shown"] = sums[:LI_SHOWN, LI_FEW - 1] / LI_FEW
    out["coverage"] = cov
    return out


def intervals_data():
    return _cache("intervals", _intervals_build,
                  (_intervals_build, _coverage, interval_factors, half_unit_ratios,
                   log_likelihood), seed=LI_SEED, ns=list(LI_NS))


def likelihood_intervals():
    d = intervals_data()
    stream = d["stream"]
    rows = []
    for n in LI_NS:
        best = float(stream[:n].mean())
        (lo, hi), _sym = interval_factors(n)
        rows.append([best, best * (1 - lo), best * (hi - 1), best / math.sqrt(n)])
    table = np.array(rows)                            # best, minus, plus, sigma
    exact = np.array([[exact_coverage(n, *interval_factors(n)[k]) for n in LI_NS]
                      for k in range(2)])
    cov = d["coverage"]
    mean = cov.mean(0)
    i_many, i_few = LI_NS.index(LI_MANY), LI_NS.index(LI_FEW)
    return Study(
        "Errors from the likelihood",
        "Where ln L has fallen by one half, and how often that interval holds the truth",
        (
            Panel("many events: a parabola", "ln L minus its maximum for %d events, and the "
                  "parabola with the same curvature" % LI_MANY, "curve",
                  {"x": d["many_grid"], "ys": np.stack([d["many_dlnl"], d["many_parabola"]]),
                   "labels": ["ln L", "parabola"]}),
            Panel("few events: not a parabola", "the same for the first %d events" % LI_FEW,
                  "curve",
                  {"x": d["few_grid"], "ys": np.stack([d["few_dlnl"], d["few_parabola"]]),
                   "labels": ["ln L", "parabola"]}),
            Panel("the two intervals", "best lifetime, lower and upper error where ln L falls "
                  "by 1/2, and the symmetric error, per number of events", "matrix",
                  {"values": table, "n": np.array(LI_NS),
                   "columns": ["best", "minus", "plus", "sigma"], "shown": d["shown"]}),
            Panel("coverage", "share of %d pseudo-experiments whose interval contains the true "
                  "lifetime, and the exact share" % LI_TOYS, "curve",
                  {"x": np.array(LI_NS, float),
                   "ys": np.stack([mean[0, :, 0], mean[1, :, 0], exact[0, :, 0],
                                   exact[1, :, 0]]),
                   "labels": ["half-unit interval", "symmetric interval",
                              "half-unit, exact", "symmetric, exact"],
                   "per_seed": cov[..., 0], "misses": mean[..., 1:], "exact_misses":
                   exact[..., 1:]}),
        ),
        {
            "model": "decay times drawn from an exponential of lifetime tau = %.1f, recorded "
                     "to 0.01; the first %d and the first %d of one run" % (LI_TAU, LI_FEW,
                                                                            LI_MANY),
            "true_tau": LI_TAU, "seed": LI_SEED, "seeds": [LI_SEED, *LI_SEEDS],
            "events": list(LI_NS), "many": LI_MANY, "few": LI_FEW, "toys": LI_TOYS,
            "one_sigma": ONE_SIGMA,
            "many_row": [float(v) for v in table[i_many]],
            "few_row": [float(v) for v in table[i_few]],
            "coverage_half_unit": [float(v) for v in mean[0, :, 0]],
            "coverage_symmetric": [float(v) for v in mean[1, :, 0]],
            "exact_half_unit": [float(v) for v in exact[0, :, 0]],
            "exact_symmetric": [float(v) for v in exact[1, :, 0]],
            "evidence": {"curvature": EXACT, "half a unit": EXACT, "few events": EXACT,
                         "does it cover": TOYS, "against N": TOYS + " against the exact "
                         "coverage"},
            "what_was_run": "ln L of an exponential for the first N events of one run; %d "
                            "pseudo-experiments per seed (%d seeds), each read after N = %s "
                            "events; exact coverage from the distribution of a sum of N "
                            "exponentials" % (LI_TOYS, len(LI_SEEDS),
                                              ", ".join(str(n) for n in LI_NS)),
            "designs_tried": ["one design, fixed in advance: the seed is the catalogue number"],
            "limitations": ["one model (an exponential): how fast ln L becomes a parabola "
                            "depends on the model and on the parameter chosen",
                            "the pseudo-experiments at different N share their first events"],
        },
    )


# ====================================================================== 130: propagation
EP_SEEDS = (130, 1301, 1302, 1303, 1304)
EP_DRAWS = 100000
EP_X, EP_SX = 10.0, 0.5
EP_Y, EP_SY = 5.0, 0.25
EP_RHO = 0.6
EP_SY_WIDE = 1.5
EP_RHOS = tuple(round(-0.9 + 0.15 * k, 2) for k in range(13))
EP_RELS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
EP_SHOWN = 1500
EP_EDGES = np.linspace(0.5, 5.0, 46)
EP_EDGES_NARROW = np.linspace(1.4, 2.6, 49)


def pair(z, sx, sy, rho):
    """Standard normal pairs ``z`` as measurements ``(x, y)`` with errors ``sx``, ``sy``
    and correlation ``rho``."""
    x = EP_X + sx * z[:, 0]
    y = EP_Y + sy * (rho * z[:, 0] + math.sqrt(1.0 - rho * rho) * z[:, 1])
    return x, y


def ratio_sigma(sx, sy, rho=0.0, x=EP_X, y=EP_Y):
    """The linear formula for ``f = x / y``: ``(sigma_f / f)^2 = (sx / x)^2 + (sy / y)^2
    - 2 rho (sx / x)(sy / y)``."""
    rx, ry = sx / x, sy / y
    return abs(x / y) * math.sqrt(rx * rx + ry * ry - 2.0 * rho * rx * ry)


def _summary(f):
    lo, mid, hi = np.percentile(f, [50 * (1 - ONE_SIGMA), 50.0, 50 * (1 + ONE_SIGMA)])
    return [f.mean(), f.std(ddof=1), lo, mid, hi]


def _propagation_build():
    one, rho, rel = [], [], []
    out = {}
    for s, seed in enumerate(EP_SEEDS):
        z = np.random.default_rng(seed).standard_normal((EP_DRAWS, 2))
        one.append([_summary(EP_X / (EP_Y + sy * z[:, 1])) for sy in (EP_SY, EP_SY_WIDE)])
        rho.append([_summary(np.divide(*pair(z, EP_SX, EP_SY, r))) for r in EP_RHOS])
        rel.append([_summary(np.divide(*pair(z, EP_SX, EP_Y * r, 0.0))) for r in EP_RELS])
        if s == 0:
            out["z"] = z[:EP_SHOWN]
            wide = np.divide(*pair(z, EP_SX, EP_SY_WIDE, 0.0))
            out["wide_hist"] = np.histogram(wide, EP_EDGES)[0] / (EP_DRAWS * np.diff(EP_EDGES))
            out["wide_outside"] = np.array(((wide < EP_EDGES[0]) | (wide > EP_EDGES[-1])).mean())
            gentle = np.divide(*pair(z, EP_SX, EP_SY, 0.0))
            out["gentle_hist"] = (np.histogram(gentle, EP_EDGES_NARROW)[0]
                                  / (EP_DRAWS * np.diff(EP_EDGES_NARROW)))
    out.update(one=np.array(one), rho=np.array(rho), rel=np.array(rel))
    return out


def propagation_data():
    return _cache("propagation", _propagation_build, (_propagation_build, pair, _summary),
                  seeds=list(EP_SEEDS), draws=EP_DRAWS)


def error_propagation():
    d = propagation_data()
    one, rho, rel = d["one"], d["rho"], d["rel"]       # [seed, case, (mean sd lo mid hi)]
    f0 = EP_X / EP_Y
    with_term = np.array([ratio_sigma(EP_SX, EP_SY, r) for r in EP_RHOS])
    without = np.full(len(EP_RHOS), ratio_sigma(EP_SX, EP_SY, 0.0))
    formula_rel = np.array([ratio_sigma(EP_SX, EP_Y * r) for r in EP_RELS])
    m = rel.mean(0)
    minus, plus = m[:, 3] - m[:, 2], m[:, 4] - m[:, 3]
    x, y = pair(d["z"], EP_SX, EP_SY, EP_RHO)
    mids = 0.5 * (EP_EDGES[:-1] + EP_EDGES[1:])
    wide_formula = ratio_sigma(EP_SX, EP_SY_WIDE)
    i0, i6 = EP_RHOS.index(0.0), EP_RHOS.index(EP_RHO)
    j_gentle, j_wide = EP_RELS.index(0.05), EP_RELS.index(0.30)
    return Study(
        "Error propagation by cloud",
        "A cloud of measurements pushed through a ratio, against the linear formula",
        (
            Panel("the cloud", "%d of the %d draws of (x, y) with correlation %.1f"
                  % (EP_SHOWN, EP_DRAWS, EP_RHO), "points",
                  {"xy": np.stack([x, y], 1), "z": d["z"], "f": x / y}),
            Panel("the correlation term", "spread of x / y against the correlation of x and "
                  "y: the cloud, the formula with its correlation term and without", "curve",
                  {"x": np.array(EP_RHOS), "ys": np.stack([rho.mean(0)[:, 1], with_term,
                                                           without]),
                   "labels": ["cloud", "formula", "formula without the term"],
                   "per_seed": rho[:, :, 1]}),
            Panel("a poorly measured denominator", "x / y when y is known to 30%: the cloud "
                  "against the bell curve the formula predicts", "curve",
                  {"x": mids, "ys": np.stack([d["wide_hist"],
                                              normal_pdf(mids, f0, wide_formula)]),
                   "labels": ["cloud", "formula"], "gentle_hist": d["gentle_hist"],
                   "gentle_x": 0.5 * (EP_EDGES_NARROW[:-1] + EP_EDGES_NARROW[1:])}),
            Panel("where the formula breaks", "the formula's error against the cloud's lower "
                  "and upper 68% half-widths, as the denominator's relative error grows",
                  "curve",
                  {"x": np.array(EP_RELS), "ys": np.stack([formula_rel, minus, plus]),
                   "labels": ["formula", "cloud, downwards", "cloud, upwards"],
                   "sd_per_seed": rel[:, :, 1], "summary": m}),
        ),
        {
            "model": "x = %g +- %g and y = %g +- %g, Gaussian, with correlation rho; "
                     "f = x / y" % (EP_X, EP_SX, EP_Y, EP_SY),
            "seeds": list(EP_SEEDS), "draws": EP_DRAWS, "rho": EP_RHO, "f0": f0,
            "one_variable": {"formula": EP_X * EP_SY / EP_Y ** 2,
                             "cloud": float(one.mean(0)[0, 1])},
            "two_variables": {"formula": float(without[0]), "cloud": float(rho.mean(0)[i0, 1]),
                              "cloud_range": [float(rho[:, i0, 1].min()),
                                              float(rho[:, i0, 1].max())]},
            "correlated": {"formula": float(with_term[i6]), "without_term": float(without[0]),
                           "cloud": float(rho.mean(0)[i6, 1]),
                           "cloud_range": [float(rho[:, i6, 1].min()),
                                           float(rho[:, i6, 1].max())]},
            "gentle": {"formula": float(formula_rel[j_gentle]), "minus": float(minus[j_gentle]),
                       "plus": float(plus[j_gentle])},
            "wide": {"sigma_y": EP_SY_WIDE, "formula": float(wide_formula),
                     "minus": float(minus[j_wide]), "plus": float(plus[j_wide]),
                     "median": float(m[j_wide, 3]), "mean": float(m[j_wide, 0]),
                     "sd_per_seed": [float(v) for v in rel[:, j_wide, 1]],
                     "outside_histogram": float(d["wide_outside"])},
            "evidence": {"one number": EXACT + " (the formula) and " + TOYS,
                         "two numbers": TOYS, "correlation": TOYS,
                         "a curve": EXACT + " (the curve) and " + TOYS, "its limits": TOYS},
            "what_was_run": "%d Gaussian pairs per seed (%d seeds) pushed through x / y, for "
                            "13 correlations and 7 relative errors of y; the cloud's spread "
                            "is its standard deviation where that is stable, and half of its "
                            "central 68%% interval where it is not" % (EP_DRAWS,
                                                                       len(EP_SEEDS)),
            "designs_tried": ["an exponential as the curved function was considered and "
                              "dropped before anything was run, to keep one function "
                              "throughout: only the denominator's error changes"],
            "limitations": ["a ratio with a Gaussian denominator has no finite variance: the "
                            "cloud's standard deviation never settles when y is poorly "
                            "measured (it differs seed to seed), so the curved case is "
                            "reported as 68% half-widths",
                            "one function and Gaussian errors only"],
        },
    )


# ====================================================================== 131: error on the mean
EM_SEEDS = (131, 1311, 1312)
EM_MU, EM_SIGMA = 10.0, 2.0
EM_NS = tuple(2 ** k for k in range(11))             # 1 .. 1024
EM_HIST_NS = (1, 4, 16, 64)
EM_TOYS = 10000
EM_EDGES = np.linspace(4.0, 16.0, 73)
EM_SD_NS = (2, 3, 4, 5, 6, 8, 12, 16, 32, 64)
EM_SD_TOYS = 20000
EM_SHOWN, EM_SHOWN_N = 12, 4                         # experiments drawn one by one
EM_COST_NS = (4, 16, 64, 256)


def _mean_build():
    spread = np.zeros((len(EM_SEEDS), 2, len(EM_NS)))
    out = {}
    hists = []
    for s, seed in enumerate(EM_SEEDS):
        rng = np.random.default_rng(seed)
        if s == 0:                                    # the experiments drawn one by one
            out["shown"] = np.round(rng.normal(EM_MU, EM_SIGMA, (EM_SHOWN, EM_SHOWN_N)), 1)
        for j, n in enumerate(EM_NS):
            means = rng.normal(EM_MU, EM_SIGMA, (EM_TOYS, n)).mean(1)
            spread[s, 0, j] = means.std(ddof=1)
            if s == 0 and n in EM_HIST_NS:
                hists.append(np.histogram(means, EM_EDGES)[0] / (EM_TOYS * np.diff(EM_EDGES)))
            # the same with a lopsided distribution of the same mean and sigma
            skew = EM_MU - EM_SIGMA + rng.exponential(EM_SIGMA, (EM_TOYS, n))
            spread[s, 1, j] = skew.mean(1).std(ddof=1)
    sd = np.zeros((len(EM_SEEDS), 3, len(EM_SD_NS)))
    for s, seed in enumerate(EM_SEEDS):
        rng = np.random.default_rng(seed + 50000)
        for j, n in enumerate(EM_SD_NS):
            x = rng.normal(EM_MU, EM_SIGMA, (EM_SD_TOYS, n))
            ss = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)
            sd[s, :, j] = (ss / n).mean(), (ss / (n - 1)).mean(), np.sqrt(ss / (n - 1)).mean()
    out.update(spread=spread, hists=np.array(hists), sd=sd)
    return out


def mean_data():
    return _cache("mean", _mean_build, (_mean_build,), seeds=list(EM_SEEDS), toys=EM_TOYS)


def error_on_the_mean():
    d = mean_data()
    spread, sd = d["spread"], d["sd"]
    ns = np.array(EM_NS, float)
    law = EM_SIGMA / np.sqrt(ns)
    m = spread.mean(0)
    slope = float(np.polyfit(np.log(ns), np.log(m[0]), 1)[0])
    mids = 0.5 * (EM_EDGES[:-1] + EM_EDGES[1:])
    sd_ns = np.array(EM_SD_NS, float)
    var = EM_SIGMA ** 2
    sdm = sd.mean(0)
    exact = np.stack([(sd_ns - 1) / sd_ns, np.ones(len(sd_ns)),
                      [c4(int(n)) for n in EM_SD_NS]])
    measured = np.stack([sdm[0] / var, sdm[1] / var, sdm[2] / EM_SIGMA])
    shown = d["shown"]
    # sums of squared deviations of the drawn experiments: about their own mean, about mu
    ss = np.stack([((shown - shown.mean(1, keepdims=True)) ** 2).sum(1),
                   ((shown - EM_MU) ** 2).sum(1)], 1)
    return Study(
        "The error on the mean",
        "An experiment of N measurements, repeated: how far its mean is from the truth",
        (
            Panel("means of repeated experiments", "histograms of the means of %d "
                  "experiments, for N = 1, 4, 16 and 64 measurements each" % EM_TOYS, "curve",
                  {"x": mids, "ys": d["hists"], "labels": ["N = %d" % n for n in EM_HIST_NS],
                   "exact": np.stack([normal_pdf(mids, EM_MU, EM_SIGMA / math.sqrt(n))
                                      for n in EM_HIST_NS]),
                   "shown": shown, "shown_means": shown.mean(1), "shown_ss": ss}),
            Panel("spread of the mean against N", "measured spread of the means against "
                  "sigma / sqrt(N), for a bell curve and for a lopsided distribution", "curve",
                  {"x": ns, "ys": np.stack([m[0], m[1], law]),
                   "labels": ["Gaussian", "lopsided", "sigma / sqrt(N)"],
                   "per_seed": spread}),
            Panel("N or N - 1", "average of the sample variance with N and with N - 1 below, "
                  "and of the standard deviation, in units of the true value", "curve",
                  {"x": sd_ns, "ys": np.concatenate([measured, exact]),
                   "labels": ["variance, / N", "variance, / (N - 1)", "s, / (N - 1)",
                              "exact (N - 1) / N", "exact 1", "exact c4"],
                   "per_seed": sd}),
            Panel("what a smaller error costs", "measured spread of the mean for 4, 16, 64 "
                  "and 256 measurements", "bars",
                  {"values": np.array([m[0][EM_NS.index(n)] for n in EM_COST_NS]),
                   "labels": ["N = %d" % n for n in EM_COST_NS],
                   "law": np.array([EM_SIGMA / math.sqrt(n) for n in EM_COST_NS])}),
        ),
        {
            "model": "each measurement drawn from a Gaussian of mean mu = %g and sigma = %g "
                     "(and, for comparison, a lopsided distribution of the same mean and "
                     "sigma: %g plus an exponential of mean %g)"
                     % (EM_MU, EM_SIGMA, EM_MU - EM_SIGMA, EM_SIGMA),
            "mu": EM_MU, "sigma": EM_SIGMA, "seeds": list(EM_SEEDS), "toys": EM_TOYS,
            "sd_toys": EM_SD_TOYS, "slope": slope,
            "shown_spread": float(shown.mean(1).std(ddof=1)),
            "shown_ss": [float(v) for v in ss.mean(0)],
            "spread": {str(n): float(v) for n, v in zip(EM_NS, m[0])},
            "spread_range": {str(n): [float(spread[:, 0, j].min()), float(spread[:, 0, j].max())]
                             for j, n in enumerate(EM_NS)},
            "sd_ratio": {str(n): [float(v) for v in measured[:, j]]
                         for j, n in enumerate(EM_SD_NS)},
            "evidence": {"one experiment": TOYS, "repeat it": TOYS + " against the exact "
                         "bell curves", "1/√N": TOYS + " against the formula",
                         "N − 1": TOYS + " against exact values", "the cost": TOYS},
            "what_was_run": "%d experiments of N measurements for N = 1 to 1024 and %d seeds; "
                            "%d experiments per N for the sample standard deviation"
                            % (EM_TOYS, len(EM_SEEDS), EM_SD_TOYS),
            "designs_tried": ["one design, fixed in advance"],
            "limitations": ["independent measurements only: a shared (systematic) error does "
                            "not shrink with N",
                            "the twelve experiments drawn one by one are recorded to 0.1"],
        },
    )


# ====================================================================== 132: the bootstrap
BS_SEED = 132
BS_N = 20
BS_TAU = 2.0                      # x: exponential of mean 2
BS_A, BS_B, BS_NOISE = 1.0, 0.5, 1.0                 # y = 1 + 0.5 x + Gaussian(0, 1)
BS_STATS = ("mean", "median", "ratio of means", "correlation", "maximum")
BS_B_MAIN = 2000
BS_TRUE_SEEDS = (1321, 1322, 1323)
BS_TRUE = 20000
BS_SETS, BS_SETS_B = 400, 400
BS_BS = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
BS_REPEATS = 100
BS_SHOWN = 4                      # resamples drawn pick by pick
BS_MEAN_EDGES = np.linspace(0.0, 4.5, 46)
BS_MAX_EDGES = np.linspace(0.0, 16.0, 65)


def model(rng, shape):
    """Events of the stated model: ``x`` exponential of mean 2, ``y = 1 + x / 2`` plus
    Gaussian noise of sigma 1."""
    x = rng.exponential(BS_TAU, shape)
    return x, BS_A + BS_B * x + rng.normal(0.0, BS_NOISE, shape)


def statistics(x, y):
    """The five statistics of each dataset (last axis): mean of x, median of x, mean of
    y over mean of x, correlation coefficient of x and y, largest x."""
    mx, my = x.mean(-1), y.mean(-1)
    dx, dy = x - mx[..., None], y - my[..., None]
    r = (dx * dy).sum(-1) / np.sqrt((dx * dx).sum(-1) * (dy * dy).sum(-1))
    return np.stack([mx, np.median(x, -1), my / mx, r, x.max(-1)], -1)


def bootstrap(x, y, idx):
    """The statistics of the resamples ``idx`` (rows of positions in the dataset)."""
    return statistics(x[idx], y[idx])


def _bootstrap_build():
    rng = np.random.default_rng(BS_SEED)
    x, y = model(rng, BS_N)
    idx = rng.integers(0, BS_N, (BS_B_MAIN, BS_N))
    boot = bootstrap(x, y, idx)
    # the truth: the statistic's spread over fresh datasets from the model
    true = np.stack([statistics(*model(np.random.default_rng(s), (BS_TRUE, BS_N)))
                     for s in BS_TRUE_SEEDS])
    # many datasets, each bootstrapped: how good is the bootstrap on average?
    rng2 = np.random.default_rng(BS_SEED + 1000)
    xs, ys = model(rng2, (BS_SETS, BS_N))
    pick = rng2.integers(0, BS_N, (BS_SETS, BS_SETS_B, BS_N))
    rows = np.arange(BS_SETS)[:, None, None]
    sets = statistics(xs[rows, pick], ys[rows, pick]).std(1, ddof=1)
    # how many resamples: the median's bootstrap error after B of them, many times over
    rng3 = np.random.default_rng(BS_SEED + 2000)
    runs = np.zeros((BS_REPEATS, len(BS_BS)))
    for r in range(BS_REPEATS):
        med = np.median(x[rng3.integers(0, BS_N, (BS_BS[-1], BS_N))], 1)
        runs[r] = [med[:b].std(ddof=1) for b in BS_BS]
    return {"x": x, "y": y, "idx": idx[:BS_SHOWN], "boot": boot,
            "true_sd": true.std(1, ddof=1), "true_mean_hist": np.histogram(
                true[0, :, 0], BS_MEAN_EDGES)[0] / (BS_TRUE * np.diff(BS_MEAN_EDGES)),
            "true_max_hist": np.histogram(true[0, :, 4], BS_MAX_EDGES)[0] / (
                BS_TRUE * np.diff(BS_MAX_EDGES)),
            "sets": sets, "runs": runs}


def bootstrap_data():
    return _cache("bootstrap", _bootstrap_build, (_bootstrap_build, model, statistics),
                  seed=BS_SEED, n=BS_N)


def max_pdf(x, n=BS_N, tau=BS_TAU):
    """Exact density of the largest of ``n`` exponentials of mean ``tau``."""
    e = np.exp(-np.asarray(x, float) / tau)
    return n / tau * e * (1.0 - e) ** (n - 1)


def bootstrap_errors():
    d = bootstrap_data()
    x, y, boot = d["x"], d["y"], d["boot"]
    here = statistics(x, y)
    se = boot.std(0, ddof=1)
    true_sd = d["true_sd"].mean(0)
    ratio = d["sets"] / true_sd                       # [dataset, statistic]
    band = np.percentile(ratio, [50 * (1 - ONE_SIGMA), 50.0, 50 * (1 + ONE_SIGMA)], axis=0)
    formula = float(x.std(ddof=1) / math.sqrt(BS_N))
    exact_mean = BS_TAU / math.sqrt(BS_N)
    exact_max = BS_TAU * math.sqrt(sum(1.0 / k ** 2 for k in range(1, BS_N + 1)))
    runs = d["runs"]
    final = float(runs[:, -1].mean())
    scatter = runs.std(0, ddof=1) / final
    mean_mids = 0.5 * (BS_MEAN_EDGES[:-1] + BS_MEAN_EDGES[1:])
    max_mids = 0.5 * (BS_MAX_EDGES[:-1] + BS_MAX_EDGES[1:])
    boot_mean_hist = np.histogram(boot[:, 0], BS_MEAN_EDGES)[0] / (
        BS_B_MAIN * np.diff(BS_MEAN_EDGES))
    boot_max_hist = np.histogram(boot[:, 4], BS_MAX_EDGES)[0] / (
        BS_B_MAIN * np.diff(BS_MAX_EDGES))
    top_share = float((boot[:, 4] == x.max()).mean())
    largest = np.sort(x)[::-1][:3]                    # the three largest values, and how
    top_values = [[float(v), float((boot[:, 4] == v).mean())] for v in largest]   # often each
    names = list(BS_STATS)
    return Study(
        "The bootstrap",
        "Resampling one dataset to get an error without a formula",
        (
            Panel("the bootstrap means", "means of %d resamples of the dataset, against the "
                  "means of %d fresh datasets" % (BS_B_MAIN, BS_TRUE), "curve",
                  {"x": mean_mids, "ys": np.stack([boot_mean_hist, d["true_mean_hist"]]),
                   "labels": ["resamples", "fresh datasets"], "data_x": x, "data_y": y,
                   "picks": d["idx"], "first": boot[:60]}),
            Panel("bootstrap error against the true spread", "per statistic: the bootstrap "
                  "error from this dataset, and the spread over fresh datasets", "bars",
                  {"values": np.concatenate([se, true_sd]),
                   "labels": ["bootstrap: " + n for n in names] + ["true: " + n for n in names],
                   "here": here, "ratio_band": band, "true_per_seed": d["true_sd"]}),
            Panel("how many resamples", "the median's bootstrap error after B resamples: its "
                  "scatter over %d repeats of the resampling, relative to the error"
                  % BS_REPEATS, "curve",
                  {"x": np.array(BS_BS, float),
                   "ys": np.stack([scatter, 1.0 / np.sqrt(2.0 * np.array(BS_BS, float))]),
                   "labels": ["measured scatter", "1 / sqrt(2B)"], "runs": runs[:12],
                   "mean": runs.mean(0)}),
            Panel("where it fails: the maximum", "largest value of each resample, against "
                  "the exact distribution of the largest of 20", "curve",
                  {"x": max_mids, "ys": np.stack([boot_max_hist, max_pdf(max_mids),
                                                  d["true_max_hist"]]),
                   "labels": ["resamples", "exact", "fresh datasets"]}),
        ),
        {
            "model": "x exponential of mean %g; y = %g + %g x + Gaussian noise of sigma %g; "
                     "%d events" % (BS_TAU, BS_A, BS_B, BS_NOISE, BS_N),
            "seed": BS_SEED, "seeds": [BS_SEED, *BS_TRUE_SEEDS], "events": BS_N,
            "resamples": BS_B_MAIN, "fresh": BS_TRUE, "datasets": BS_SETS,
            "datasets_resamples": BS_SETS_B, "statistics": names,
            "here": [float(v) for v in here], "bootstrap": [float(v) for v in se],
            "true": [float(v) for v in true_sd],
            "ratio_median": [float(v) for v in band[1]],
            "ratio_low": [float(v) for v in band[0]], "ratio_high": [float(v) for v in band[2]],
            "ratio_mean": [float(v) for v in ratio.mean(0)],
            "formula_mean": formula, "exact_mean": exact_mean, "exact_max": exact_max,
            "top_values": top_values,
            "true_values": {"mean": BS_TAU, "median": BS_TAU * math.log(2.0),
                            "ratio of means": (BS_A + BS_B * BS_TAU) / BS_TAU,
                            "correlation": BS_B * BS_TAU / math.sqrt(
                                (BS_B * BS_TAU) ** 2 + BS_NOISE ** 2)},
            "top_share": top_share, "top_share_exact": 1.0 - (1.0 - 1.0 / BS_N) ** BS_N,
            "never_picked_exact": (1.0 - 1.0 / BS_N) ** BS_N,
            "scatter": {str(b): float(v) for b, v in zip(BS_BS, scatter)},
            "error_at_b": {str(b): float(v) for b, v in zip(BS_BS, runs.mean(0))},
            "evidence": {"one dataset": TOYS + " (one dataset)", "resample": TOYS,
                         "no formula": TOYS, "how many": TOYS,
                         "it fails": TOYS + " against the exact distribution"},
            "what_was_run": "one dataset of %d events resampled %d times; %d fresh datasets "
                            "per seed (%d seeds) for the true spread; %d further datasets "
                            "each resampled %d times; the resampling of the first dataset "
                            "repeated %d times at B = 10 to 5000"
                            % (BS_N, BS_B_MAIN, BS_TRUE, len(BS_TRUE_SEEDS), BS_SETS,
                               BS_SETS_B, BS_REPEATS),
            "designs_tried": ["one design, fixed in advance: the five statistics and the "
                              "failing case (the maximum) were chosen before any was run"],
            "limitations": ["20 events: the bootstrap's own error is large at this size (see "
                            "the band over %d datasets)" % BS_SETS,
                            "the plain bootstrap only: no bias correction, no smoothing"],
        },
    )


STUDIES = {
    "likelihood-curve": likelihood_curve,
    "likelihood-intervals": likelihood_intervals,
    "error-propagation": error_propagation,
    "error-on-the-mean": error_on_the_mean,
    "bootstrap-errors": bootstrap_errors,
}
