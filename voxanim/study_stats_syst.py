"""The experiments behind animations 133-136: correlated errors and systematic
uncertainties (see :func:`voxanim.next_catalog.redesigned`).

* ``error-ellipse``: a straight line ``y = a + b x`` fitted to five points, twenty
  thousand times over. The fitted pairs (a, b) form a tilted cloud; its covariance
  matrix against the exact one, the chi-square + 1 contour of one fit against the
  cloud's ellipse, how often that contour holds the truth, the width of ``a`` with ``b``
  held fixed, and the same fit with x counted from the centre of the data.
* ``statistical-vs-systematic``: a quantity measured event by event by a detector whose
  scale is off by an amount known only to +-0.5. The mean of N events against N; its
  spread about the truth over repeated experiments in which the scale error is redrawn;
  the coverage of the statistical-only interval; and a control sample of a known
  reference that measures the scale error itself.
* ``nuisance-parameters-profiling``: a count ``n ~ Poisson(mu s + b)`` with the
  background ``b`` constrained to 50 +- 10. The likelihood over (mu, b); the interval
  on mu with b fixed, profiled and marginalised; and the coverage of each over
  pseudo-experiments in which the true background varies.
* ``combining-measurements``: two measurements of one quantity that share a systematic
  shift. Weights 1/sigma^2 against the best linear combination for their covariance,
  the weights and the combined error against the correlation, and the spread and the
  claimed error of each combination over pseudo-experiments.

Every dataset is a seeded draw from a model whose true values are constants of this
module (``numpy.random.default_rng(seed)``); wherever a formula gives the answer it is
recorded beside the pseudo-experiments' number. No data is downloaded.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
TOYS = 20000                      # pseudo-experiments per study and seed
ONE_SIGMA = math.erf(1.0 / math.sqrt(2.0))        # 0.6827: a Gaussian within one sigma


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-syst-" + name, settings,
                                                    build).items()}


# ---------------------------------------------------------------------- special functions
def gauss_within(k):
    """The chance a Gaussian variable lies within ``k`` standard deviations of its mean."""
    return np.vectorize(lambda v: math.erf(v / math.sqrt(2.0)), otypes=[float])(k)


def chi2_cdf_2dof(x):
    """The chance a chi-square variable with two degrees of freedom is below ``x``."""
    return 1.0 - np.exp(-np.asarray(x, float) / 2.0)


def log_poisson(n, lam):
    """``ln`` of the Poisson probability of ``n`` events when ``lam`` are expected."""
    n = np.asarray(n, float)
    lg = np.vectorize(math.lgamma, otypes=[float])(n + 1.0)
    return n * np.log(lam) - lam - lg


def crossings(x, y, level):
    """Where the curve ``y(x)`` (one minimum, on an increasing grid) crosses ``level`` on
    each side of its minimum, by linear interpolation: ``(left, right, x at minimum)``."""
    k = int(np.argmin(y))
    d = y - y[k] - level
    i = k
    while i > 0 and d[i] < 0:
        i -= 1
    j = k
    while j < len(x) - 1 and d[j] < 0:
        j += 1
    left = x[i] + (x[i + 1] - x[i]) * d[i] / (d[i] - d[i + 1]) if d[i] >= 0 else x[0]
    right = x[j - 1] + (x[j] - x[j - 1]) * d[j - 1] / (d[j - 1] - d[j]) if d[j] >= 0 else x[-1]
    return float(left), float(right), float(x[k])


# ====================================================================== 133: error ellipse
ELL_SEED = 133
ELL_SEEDS = (133, 1133, 2133)
ELL_X = np.arange(1.0, 6.0)       # where the five points are measured
A_TRUE, B_TRUE, SIGMA_Y = 1.0, 0.5, 0.5
ELL_KEPT = 2000                   # fitted pairs recorded (the statistics use all of them)


def line_fit(x, y):
    """Least-squares intercept and slope of each row of ``y`` against ``x`` (equal
    errors): ``b = sum (x - xbar)(y - ybar) / sum (x - xbar)^2``, ``a = ybar - b xbar``."""
    d = x - x.mean()
    b = (y * d).sum(-1) / (d ** 2).sum()
    return y.mean(-1) - b * x.mean(), b


def line_cov(x, sigma):
    """The exact covariance matrix of (intercept, slope): ``sigma^2 (X^T X)^-1``."""
    design = np.stack([np.ones_like(x), x], 1)
    return sigma ** 2 * np.linalg.inv(design.T @ design)


def chi2_line(a, b, x, y, sigma):
    """Chi-square of the line ``a + b x`` against the points ``y``."""
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    return (((y - a - b * x) / sigma) ** 2).sum(-1)


def contour(fn, centre, level, scale, n=72):
    """Points where ``fn`` (smallest at ``centre``, growing along every ray from it)
    reaches ``level``: ``n`` rays, each solved by bisection on ``fn`` itself."""
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    dx, dy = np.cos(ang) * scale[0], np.sin(ang) * scale[1]
    lo, hi = np.zeros(n), np.full(n, 50.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        above = fn(centre[0] + mid * dx, centre[1] + mid * dy) > level
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    r = (lo + hi) / 2
    return np.stack([centre[0] + r * dx, centre[1] + r * dy], 1)


def _ellipse_build(seed):
    x = ELL_X
    rng = np.random.default_rng(seed)
    y = A_TRUE + B_TRUE * x + rng.normal(0.0, SIGMA_Y, (TOYS, len(x)))
    a, b = line_fit(x, y)
    exact = line_cov(x, SIGMA_Y)
    cov = np.cov(np.stack([a, b]))
    chi_min = chi2_line(a, b, x, y, SIGMA_Y)
    chi_true = chi2_line(A_TRUE, B_TRUE, x, y, SIGMA_Y)
    a_fixed = (y - B_TRUE * x).mean(1)             # the fit with the slope held at the truth
    mid = a + b * x.mean()                         # the line's height at the data's centre
    cov_mid = np.cov(np.stack([mid, b]))
    # one experiment's chi-square + 1 contour, found on its chi-square itself
    first = np.array([a[0], b[0]])
    ring = contour(lambda p, q: chi2_line(p, q, x, y[0], SIGMA_Y) - chi_min[0], first, 1.0,
                   np.sqrt(np.diag(exact)))
    d = ring - first
    mismatch = np.sqrt(np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)) - 1.0
    # chi-square against the intercept: slope refitted for each value, and slope held
    grid = first[0] + np.linspace(-1.5, 1.5, 121)
    b_best = ((y[0] - grid[:, None]) * x).sum(1) / (x ** 2).sum()
    free = chi2_line(grid, b_best, x, y[0], SIGMA_Y) - chi_min[0]
    held = chi2_line(grid, np.full(len(grid), first[1]), x, y[0], SIGMA_Y) - chi_min[0]
    return dict(
        y=y[:3], fits=np.stack([a, b], 1)[:ELL_KEPT], cov=cov, exact=exact,
        chi_min=chi_min[:3], ring=ring, mismatch=np.abs(mismatch).max(),
        inside=np.mean(chi_true - chi_min <= 1.0),
        inside_a=np.mean(np.abs(a - A_TRUE) <= np.sqrt(exact[0, 0])),
        inside_b=np.mean(np.abs(b - B_TRUE) <= np.sqrt(exact[1, 1])),
        sd_fixed=a_fixed.std(ddof=1), a_fixed=a_fixed[:ELL_KEPT], cov_mid=cov_mid,
        mid=mid[:ELL_KEPT], grid=grid, free=free, held=held,
        free_at=np.array(crossings(grid, free, 1.0)),
        held_at=np.array(crossings(grid, held, 1.0)))


def ellipse_data(seed=ELL_SEED):
    return _cache("ellipse", lambda: _ellipse_build(seed),
                  (_ellipse_build, line_fit, line_cov, chi2_line, contour, crossings),
                  seed=seed, toys=TOYS, x=ELL_X.tolist(), truth=[A_TRUE, B_TRUE, SIGMA_Y])


def corr(cov):
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def error_ellipse():
    d = ellipse_data()
    exact, cov = d["exact"], d["cov"]
    n = len(ELL_X)
    rho, rho_exact = corr(cov), corr(exact)
    sa, sb = np.sqrt(np.diag(cov))
    sa_exact = float(np.sqrt(exact[0, 0]))
    cond_exact = SIGMA_Y / math.sqrt(n)
    by_seed = [dict(seed=s, rho=corr(e["cov"]), sigma_a=float(np.sqrt(e["cov"][0, 0])),
                    sigma_a_fixed=float(e["sd_fixed"]), rho_centred=corr(e["cov_mid"]),
                    inside=float(e["inside"]))
               for s, e in ((s, ellipse_data(s)) for s in ELL_SEEDS)]
    return Study(
        "The error ellipse",
        "A straight line fitted %s times: the cloud of (intercept, slope), its covariance "
        "matrix and the chi-square + 1 contour of one fit" % format(TOYS, ","),
        (
            Panel("Fitted intercept and slope",
                  "xy: (a, b) of the first %d of %s least-squares fits of y = a + b x to "
                  "five points at x = 1..5, each y drawn as %.1f + %.1f x + Gaussian noise "
                  "of sigma %.1f. ring: the chi-square + 1 contour of the first fit."
                  % (ELL_KEPT, format(TOYS, ","), A_TRUE, B_TRUE, SIGMA_Y), "points",
                  dict(xy=d["fits"], ring=d["ring"], y=d["y"], x=ELL_X)),
            Panel("Covariance matrix of (a, b)",
                  "values: covariance of the fitted pairs over all pseudo-experiments. "
                  "exact: sigma^2 (X^T X)^-1. centred: the same for (height at x = 3, b).",
                  "matrix", dict(values=cov, exact=exact, centred=d["cov_mid"],
                                 labels=["a", "b"])),
            Panel("Chi-square against the intercept",
                  "ys[0]: chi-square minus its minimum for the first experiment as a is "
                  "moved and b refitted each time; ys[1]: the same with b held at its "
                  "fitted value. Each crosses 1 at one standard error.",
                  "curve", dict(x=d["grid"], ys=np.array([d["free"], d["held"]]),
                                labels=["b refitted", "b held"],
                                free_at=d["free_at"], held_at=d["held_at"])),
            Panel("Width of the intercept",
                  "Standard deviation of a over the pseudo-experiments: b fitted too; b "
                  "held at the truth; and the height at x = 3 with b fitted. exact: the "
                  "formulas.", "bars",
                  dict(values=np.array([sa, d["sd_fixed"], np.sqrt(d["cov_mid"][0, 0])]),
                       exact=np.array([sa_exact, cond_exact, cond_exact]),
                       labels=["a, b free", "a, b fixed", "height at centre"],
                       mid=d["mid"])),
        ),
        dict(
            seed=ELL_SEED, seeds=list(ELL_SEEDS), review_only=True, toys=TOYS,
            source="pseudo-experiments (seeded draws from a stated straight line) against "
                   "exact least-squares formulas",
            model=dict(a=A_TRUE, b=B_TRUE, sigma=SIGMA_Y, x=ELL_X.tolist()),
            evidence=dict(
                exact="covariance sigma^2 (X^T X)^-1, conditional width sigma/sqrt(N), "
                      "39.3% and 68.3% contents, the chi-square of one dataset",
                pseudo_experiments="%d fits per seed: covariance, correlation, share of "
                                   "contours holding the truth, widths" % TOYS),
            where_the_seed_enters="numpy.random.default_rng(seed).normal: the noise on "
                                  "the five y values of every pseudo-experiment",
            first_fit=[float(v) for v in d["fits"][0]],
            sigma_a=float(sa), sigma_b=float(sb), rho=rho,
            exact=dict(sigma_a=sa_exact, sigma_b=float(np.sqrt(exact[1, 1])), rho=rho_exact,
                       conditional=cond_exact, inside=float(chi2_cdf_2dof(1.0)),
                       inside_one=ONE_SIGMA),
            conditional=float(d["sd_fixed"]),
            conditional_from_rho=float(sa * math.sqrt(1 - rho ** 2)),
            centred=dict(sigma=float(np.sqrt(d["cov_mid"][0, 0])), rho=corr(d["cov_mid"])),
            inside=float(d["inside"]), inside_a=float(d["inside_a"]),
            inside_b=float(d["inside_b"]), contour_mismatch=float(d["mismatch"]),
            by_seed=by_seed,
            limitations=[
                "A straight line with Gaussian errors is the case where the chi-square "
                "contour is exactly an ellipse; for a non-linear fit it is only "
                "approximately one.",
                "The errors on the points are known and equal; a fit that estimates them "
                "from the scatter has wider intervals at five points.",
            ],
            designs_tried=[
                "Only this design was run: five points at x = 1..5, sigma 0.5, 20,000 "
                "fits per seed, three seeds.",
            ],
        ),
    )


# ====================================================================== 134: stat and syst
SYS_SEED = 134
SYS_SEEDS = (134, 1134, 2134)
MU_TRUE, RESOLUTION = 100.0, 10.0
SCALE_ERR = 0.5                   # what is known about the shift: zero +- this
SHIFT_DRAWN = 0.4                 # the shift of the detector drawn in the animation
REFERENCE = 50.0                  # the known value the control sample measures
CONTROL_SHARE = 4                 # one control event per this many events
N_MAX = 10000
NS = (1, 2, 4, 10, 25, 50, 100, 200, 400, 1000, 2000, 4000, 10000)
SYS_TOYS = 4000


def stat_error(n):
    return RESOLUTION / np.sqrt(np.asarray(n, float))


def total_error(n, syst=SCALE_ERR):
    """Statistical and systematic errors added in quadrature."""
    return np.sqrt(stat_error(n) ** 2 + syst ** 2)


def control_error(n):
    """The error once the shift is taken from ``n / CONTROL_SHARE`` control events."""
    return np.sqrt(stat_error(n) ** 2 + stat_error(np.asarray(n) // CONTROL_SHARE) ** 2)


def _syst_build(seed):
    ns = np.array(NS)
    rng = np.random.default_rng(seed)
    # the one detector drawn: its events one by one, and its control sample
    events = MU_TRUE + SHIFT_DRAWN + rng.normal(0.0, RESOLUTION, N_MAX)
    control = REFERENCE + SHIFT_DRAWN + rng.normal(0.0, RESOLUTION, N_MAX // CONTROL_SHARE)
    running = np.cumsum(events) / np.arange(1, N_MAX + 1)
    run_control = np.cumsum(control) / np.arange(1, len(control) + 1)
    # repeated experiments: a new shift each time, every event drawn
    mc = ns[ns >= CONTROL_SHARE] // CONTROL_SHARE
    means = np.zeros((SYS_TOYS, len(ns)))
    cmeans = np.zeros((SYS_TOYS, len(mc)))
    shifts = rng.normal(0.0, SCALE_ERR, SYS_TOYS)
    for lo in range(0, SYS_TOYS, 500):
        s = shifts[lo:lo + 500, None]
        ev = MU_TRUE + s + rng.normal(0.0, RESOLUTION, (len(s), N_MAX))
        means[lo:lo + 500] = (np.cumsum(ev, 1) / np.arange(1, N_MAX + 1))[:, ns - 1]
        co = REFERENCE + s + rng.normal(0.0, RESOLUTION, (len(s), N_MAX // CONTROL_SHARE))
        cmeans[lo:lo + 500] = (np.cumsum(co, 1) / np.arange(1, co.shape[1] + 1))[:, mc - 1]
    off = means - MU_TRUE
    corrected = off[:, len(ns) - len(mc):] - (cmeans - REFERENCE)
    return dict(
        events=events[:100], running=running, run_control=run_control,
        rms=np.sqrt((off ** 2).mean(0)),
        rms_fixed=np.sqrt(((off - shifts[:, None]) ** 2).mean(0)),
        cover_stat=(np.abs(off) <= stat_error(ns)).mean(0),
        cover_total=(np.abs(off) <= total_error(ns)).mean(0),
        rms_control=np.sqrt((corrected ** 2).mean(0)),
        cover_control=(np.abs(corrected) <= control_error(ns[-len(mc):])).mean(0),
        first_means=means[:40], first_shifts=shifts[:40])


def syst_data(seed=SYS_SEED):
    return _cache("syst", lambda: _syst_build(seed),
                  (_syst_build, stat_error, total_error, control_error), seed=seed,
                  toys=SYS_TOYS, ns=list(NS),
                  model=[MU_TRUE, RESOLUTION, SCALE_ERR, SHIFT_DRAWN, REFERENCE,
                         CONTROL_SHARE])


def statistical_vs_systematic():
    d = syst_data()
    ns = np.array(NS)
    nc = ns[ns >= CONTROL_SHARE]
    every = np.arange(1, N_MAX + 1)
    stat, total = stat_error(ns), total_error(ns)
    cross = (RESOLUTION / SCALE_ERR) ** 2
    pays = CONTROL_SHARE * cross                   # where the control beats the floor
    exact_cover = gauss_within(stat / total)
    m = N_MAX // CONTROL_SHARE
    shift_est = float(d["run_control"][m - 1] - REFERENCE)
    by_seed = [dict(seed=s, rms_last=float(e["rms"][-1]),
                    rms_control_last=float(e["rms_control"][-1]),
                    cover_stat_last=float(e["cover_stat"][-1]),
                    cover_total_last=float(e["cover_total"][-1]))
               for s, e in ((s, syst_data(s)) for s in SYS_SEEDS)]
    return Study(
        "Statistical against systematic",
        "The mean of N events from a detector whose scale is off by an unknown amount: "
        "what more data shrinks, what it does not, and a control sample",
        (
            Panel("One detector's running mean",
                  "ys[0]: the mean of the first x events of one seeded run, each event "
                  "%.0f + %.1f (this detector's shift) + Gaussian noise of sigma %.0f. "
                  "ys[1], ys[2]: that mean -+ %.0f / sqrt(x)."
                  % (MU_TRUE, SHIFT_DRAWN, RESOLUTION, RESOLUTION), "curve",
                  dict(x=every,
                       ys=np.array([d["running"], d["running"] - stat_error(every),
                                    d["running"] + stat_error(every)]),
                       labels=["mean", "minus one statistical error",
                               "plus one statistical error"], events=d["events"])),
            Panel("Error against N",
                  "ys[0]: statistical error sigma / sqrt(N); ys[1]: the systematic %.1f; "
                  "ys[2]: both in quadrature; ys[3]: measured, the root-mean-square "
                  "distance of the mean from the truth over %s experiments, each with a "
                  "new shift drawn from a Gaussian of sigma %.1f."
                  % (SCALE_ERR, format(SYS_TOYS, ","), SCALE_ERR), "curve",
                  dict(x=ns, ys=np.array([stat, np.full(len(ns), SCALE_ERR), total,
                                          d["rms"]]),
                       labels=["statistical", "systematic", "total (exact)",
                               "total (pseudo-experiments)"], rms_fixed=d["rms_fixed"])),
            Panel("Coverage of the quoted interval",
                  "How often mean +- error holds the truth: ys[0] with the statistical "
                  "error only (measured), ys[1] the exact value of that, "
                  "erf(stat / (total sqrt 2)); ys[2] with the total error (measured).",
                  "curve", dict(x=ns, ys=np.array([d["cover_stat"], exact_cover,
                                                   d["cover_total"]]),
                                labels=["stat only", "stat only (exact)", "total"],
                                first_means=d["first_means"])),
            Panel("With a control sample",
                  "The shift measured from N / %d events of a known reference in the same "
                  "detector and subtracted. ys[0]: exact error sqrt(sigma^2/N + "
                  "sigma^2/(N/%d)); ys[1]: measured over the same pseudo-experiments; "
                  "ys[2]: the uncorrected total." % (CONTROL_SHARE, CONTROL_SHARE), "curve",
                  dict(x=nc, ys=np.array([control_error(nc), d["rms_control"],
                                          total_error(nc)]),
                       labels=["control (exact)", "control (pseudo-experiments)",
                               "floor"], coverage=d["cover_control"],
                       run_control=d["run_control"])),
        ),
        dict(
            seed=SYS_SEED, seeds=list(SYS_SEEDS), review_only=True, toys=SYS_TOYS,
            source="pseudo-experiments (every event a seeded Gaussian draw) against the "
                   "exact quadrature formulas",
            model=dict(mu=MU_TRUE, resolution=RESOLUTION, scale_error=SCALE_ERR,
                       shift_drawn=SHIFT_DRAWN, reference=REFERENCE,
                       control_share=CONTROL_SHARE, n_max=N_MAX),
            evidence=dict(
                exact="sigma/sqrt(N), quadrature sum, coverage erf(stat/(total sqrt 2))",
                pseudo_experiments="%d experiments of %d events per seed, shift redrawn "
                                   "in each; one detector's run with shift +%.1f"
                                   % (SYS_TOYS, N_MAX, SHIFT_DRAWN)),
            where_the_seed_enters="numpy.random.default_rng(seed).normal: each event's "
                                  "noise and each repeated experiment's shift",
            ns=list(NS), crossover=cross, control_pays_from=pays,
            mean_at={str(n): float(d["running"][n - 1]) for n in NS},
            stat_at={str(n): float(stat_error(n)) for n in NS},
            total_at={str(n): float(total_error(n)) for n in NS},
            rms=[float(v) for v in d["rms"]],
            rms_shift_known=[float(v) for v in d["rms_fixed"]],
            cover_stat=[float(v) for v in d["cover_stat"]],
            cover_stat_exact=[float(v) for v in exact_cover],
            cover_total=[float(v) for v in d["cover_total"]],
            control=dict(events=m, mean=float(d["run_control"][m - 1]), shift=shift_est,
                         shift_error=float(stat_error(m)),
                         corrected=float(d["running"][-1] - shift_est),
                         corrected_error=float(control_error(N_MAX)),
                         rms=[float(v) for v in d["rms_control"]],
                         exact=[float(v) for v in control_error(nc)],
                         coverage=[float(v) for v in d["cover_control"]]),
            best_of_both=float(1.0 / math.sqrt(1.0 / SCALE_ERR ** 2
                                               + m / RESOLUTION ** 2)),
            by_seed=by_seed,
            limitations=[
                "The systematic here is one additive shift with a Gaussian uncertainty; "
                "real ones are often several, of unknown shape.",
                "The resolution is taken as known when the interval is quoted.",
                "The control sample is used alone; combining it with the prior knowledge "
                "(+-%.1f) would be better at every N (see best_of_both)." % SCALE_ERR,
                "Coverage with the total error is over experiments whose shift is redrawn; "
                "one detector's shift is fixed, and its long-run mean stays off by it.",
            ],
            designs_tried=[
                "Only this design was run: resolution 10, scale known to 0.5, up to "
                "10,000 events, 4,000 experiments per seed, one control event per four.",
            ],
        ),
    )


# ====================================================================== 135: nuisance
NUI_SEED = 135
NUI_SEEDS = (135, 1135, 2135)
SIGNAL, B_NOMINAL, B_ERROR, MU_SIGNAL = 20.0, 50.0, 10.0, 1.0
MU_GRID = np.arange(-2.0, 6.0001, 0.004)
B_GRID = np.linspace(0.0, 110.0, 221)
MAP_MU = np.linspace(-0.6, 3.0, 73)
MAP_B = np.linspace(20.0, 80.0, 61)
N_TABLE = (5, 220)                # counts tabulated: outside, the chance is under 1e-9


def nll2(mu, b, n):
    """``-2 ln L(mu, b)`` for ``n`` counted events, constants dropped: a Poisson term for
    the count and a Gaussian constraint on the background."""
    lam = mu * SIGNAL + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 2.0 * (lam - n * np.log(lam)) + ((b - B_NOMINAL) / B_ERROR) ** 2
    return np.where(lam > 0, out, np.inf)


def profiled_b(mu, n):
    """The background that maximises the likelihood at signal strength ``mu``: the root
    of ``(b + mu s - n) sigma_b^2 + (b - b0)(b + mu s) = 0`` with a positive rate."""
    c = np.asarray(mu, float) * SIGNAL
    p = c - B_NOMINAL + B_ERROR ** 2
    q = (c - n) * B_ERROR ** 2 - B_NOMINAL * c
    return (-p + np.sqrt(p * p - 4 * q)) / 2.0


def curves(n, mu=MU_GRID):
    """``-2 ln L`` against ``mu`` three ways (each minus its own minimum): background
    fixed at nominal, profiled, and marginalised (the likelihood averaged over the
    constraint, integrated on :data:`B_GRID`)."""
    fixed = nll2(mu, B_NOMINAL, n)
    prof = nll2(mu, profiled_b(mu, n), n)
    full = -0.5 * nll2(mu[:, None], B_GRID[None, :], n)
    top = full.max(1, keepdims=True)
    marg = -2.0 * (top[:, 0] + np.log(np.exp(full - top).sum(1)))
    return [c - c[np.isfinite(c)].min() for c in (fixed, prof, marg)]


def intervals(n):
    """``(low, high, best)`` of mu from each curve's rise of 1: fixed, profiled,
    marginalised."""
    out = []
    for c in curves(n):
        keep = np.isfinite(c)
        out.append(crossings(MU_GRID[keep], c[keep], 1.0))
    return np.array(out)


def count_probabilities(n_max, vary=True):
    """The exact chance of each count 0..``n_max`` at mu = 1: Poisson, averaged over the
    background's Gaussian when it ``vary``-s (a sum over :data:`B_GRID`)."""
    n = np.arange(n_max + 1)
    if not vary:
        return np.exp(log_poisson(n, MU_SIGNAL * SIGNAL + B_NOMINAL))
    b = B_GRID[1:]
    w = np.exp(-0.5 * ((b - B_NOMINAL) / B_ERROR) ** 2)
    w /= w.sum()
    return (np.exp(log_poisson(n[:, None], MU_SIGNAL * SIGNAL + b[None, :])) * w).sum(1)


@functools.lru_cache(maxsize=1)
def interval_table():
    """The three intervals for every count that can occur: they depend on nothing else."""
    lo, hi = N_TABLE
    table = np.zeros((hi + 1, 3, 3))
    for n in range(lo, hi + 1):
        table[n] = intervals(n)
    return table


def _nuisance_build(seed):
    rng = np.random.default_rng(seed)
    b_true = np.maximum(rng.normal(B_NOMINAL, B_ERROR, TOYS), 0.0)
    counts = rng.poisson(MU_SIGNAL * SIGNAL + b_true)
    table = interval_table()
    lo, hi = N_TABLE
    clipped = int(((counts < lo) | (counts > hi)).sum())
    counts = np.clip(counts, lo, hi)
    iv = table[counts]                             # (toys, method, low/high/best)
    covered = (iv[:, :, 0] <= MU_SIGNAL) & (iv[:, :, 1] >= MU_SIGNAL)
    inside = (table[:, :, 0] <= MU_SIGNAL) & (table[:, :, 1] >= MU_SIGNAL)
    inside[:lo] = False
    n = int(counts[0])
    grid = np.linspace(-0.6, 3.0, 181)
    return dict(
        n=n, b_true=b_true[0], three=np.array(curves(n, grid)), grid=grid,
        path=profiled_b(grid, n), intervals=table[n],
        surface=nll2(MAP_MU[None, :], MAP_B[:, None], n),
        coverage=covered.mean(0), clipped=clipped,
        exact=(count_probabilities(hi)[:, None] * inside).sum(0),
        exact_fixed_b=(count_probabilities(hi, False)[:, None] * inside).sum(0),
        half=np.median((iv[:, :, 1] - iv[:, :, 0]) / 2, 0),
        spread=iv[:, 1, 2].std(ddof=1),
        shown=iv[:40], shown_counts=counts[:40], shown_b=b_true[:40])


def nuisance_data(seed=NUI_SEED):
    return _cache("nuisance", lambda: _nuisance_build(seed),
                  (_nuisance_build, nll2, profiled_b, curves, intervals, interval_table,
                   count_probabilities, crossings, log_poisson), seed=seed, toys=TOYS,
                  model=[SIGNAL, B_NOMINAL, B_ERROR, MU_SIGNAL],
                  grids=[len(MU_GRID), len(B_GRID), list(N_TABLE)])


METHODS = ("fixed", "profiled", "marginalised")


def nuisance_parameters_profiling():
    d = nuisance_data()
    n = int(d["n"])
    iv = d["intervals"]
    surface = d["surface"] - d["surface"].min()
    by_seed = [dict(seed=s, coverage=[float(v) for v in e["coverage"]])
               for s, e in ((s, nuisance_data(s)) for s in NUI_SEEDS)]
    return Study(
        "Nuisance parameters and profiling",
        "A count with a background known to 20%: the likelihood over signal strength and "
        "background, and the interval with the background fixed, profiled, marginalised",
        (
            Panel("-2 ln L over (mu, b)",
                  "values[i, j]: -2 ln L minus its minimum at background b[i] and signal "
                  "strength mu[j], for the %d events counted in the first seeded "
                  "experiment (expected mu x %.0f + b; b constrained to %.0f +- %.0f). "
                  "path: the best b at each mu." % (n, SIGNAL, B_NOMINAL, B_ERROR),
                  "matrix", dict(values=surface, mu=MAP_MU, b=MAP_B, path_mu=d["grid"],
                                 path_b=d["path"])),
            Panel("-2 ln L against mu, three ways",
                  "Each minus its own minimum: b fixed at %.0f; b profiled (the best b at "
                  "each mu); b marginalised (the likelihood averaged over the "
                  "constraint)." % B_NOMINAL, "curve",
                  dict(x=d["grid"], ys=d["three"], labels=list(METHODS))),
            Panel("Half-width of the interval on mu",
                  "values: half the width of the interval where each curve rises by 1, "
                  "for the first experiment. median: the same over all "
                  "pseudo-experiments.", "bars",
                  dict(values=(iv[:, 1] - iv[:, 0]) / 2, labels=list(METHODS),
                       intervals=iv, median=d["half"])),
            Panel("Coverage",
                  "values: the share of %s pseudo-experiments whose interval holds the "
                  "true mu = 1; in each the true background is drawn from a Gaussian "
                  "(%.0f, %.0f) and the count from a Poisson. exact: the same as a sum "
                  "over counts. exact_fixed_b: if the true background were always %.0f."
                  % (format(TOYS, ","), B_NOMINAL, B_ERROR, B_NOMINAL), "bars",
                  dict(values=d["coverage"], labels=list(METHODS), exact=d["exact"],
                       exact_fixed_b=d["exact_fixed_b"], shown=d["shown"],
                       shown_counts=d["shown_counts"])),
        ),
        dict(
            seed=NUI_SEED, seeds=list(NUI_SEEDS), review_only=True, toys=TOYS,
            source="the exact likelihood of one seeded count; coverage by "
                   "pseudo-experiments and by an exact sum over counts",
            model=dict(signal=SIGNAL, b_nominal=B_NOMINAL, b_error=B_ERROR, mu=MU_SIGNAL),
            evidence=dict(
                exact="the likelihood surface, the three curves and their intervals for "
                      "one count; coverage as a sum over all counts",
                pseudo_experiments="%d counts per seed, true background redrawn in each"
                                   % TOYS),
            where_the_seed_enters="numpy.random.default_rng(seed): normal for each true "
                                  "background, poisson for each count",
            count=n, b_true=float(d["b_true"]),
            mu_hat=[float(v) for v in iv[:, 2]],
            intervals={k: [float(iv[i, 0]), float(iv[i, 1])]
                       for i, k in enumerate(METHODS)},
            half_width={k: float((iv[i, 1] - iv[i, 0]) / 2) for i, k in enumerate(METHODS)},
            rule_of_thumb=dict(fixed=float(math.sqrt(n) / SIGNAL),
                               profiled=float(math.sqrt(n + B_ERROR ** 2) / SIGNAL)),
            coverage={k: float(d["coverage"][i]) for i, k in enumerate(METHODS)},
            coverage_exact={k: float(d["exact"][i]) for i, k in enumerate(METHODS)},
            coverage_if_background_never_varied={
                k: float(d["exact_fixed_b"][i]) for i, k in enumerate(METHODS)},
            nominal=ONE_SIGMA, spread_of_mu_hat=float(d["spread"]),
            counts_outside_table=int(d["clipped"]),
            median_half_width={k: float(d["half"][i]) for i, k in enumerate(METHODS)},
            by_seed=by_seed,
            limitations=[
                "In the pseudo-experiments the true background varies about the nominal "
                "value the analyst uses; that is the same, by symmetry, as a fixed "
                "background whose auxiliary measurement varies.",
                "If the true background never varied and equalled the nominal value, the "
                "fixed interval would cover correctly and the profiled one would be "
                "wider than needed (coverage_if_background_never_varied).",
                "Counts are whole numbers, so coverage moves in small steps and need not "
                "equal 68.3% exactly even for a correct method.",
                "The marginal curve is read like a likelihood (rise of 1), not as a "
                "Bayesian credible interval.",
            ],
            designs_tried=[
                "Only this design was run: s = 20, b = 50 +- 10, true mu = 1, 20,000 "
                "counts per seed, three seeds.",
            ],
        ),
    )


# ====================================================================== 136: combining
COM_SEED = 136
COM_SEEDS = (136, 1136, 2136)
X_TRUE = 10.0
STAT = np.array([0.6, 1.0])       # each measurement's own error
SENS = np.array([0.8, 2.4])       # how far one unit of the shared effect moves each
RHOS = np.linspace(0.0, 0.95, 96)
HIST_RANGE, HIST_BINS = (-4.2, 4.2), 60


def shared_cov(stat=STAT, sens=SENS):
    """Covariance of two measurements with their own errors and one shared shift."""
    return np.diag(stat ** 2) + np.outer(sens, sens)


def naive_weights(cov):
    """Weights 1/sigma^2, normalised: right when the measurements are independent."""
    w = 1.0 / np.diag(cov)
    return w / w.sum()


def blue_weights(cov):
    """The weights (summing to 1) of the linear combination with the smallest variance:
    ``V^-1 1 / (1^T V^-1 1)``."""
    w = np.linalg.solve(cov, np.ones(len(cov)))
    return w / w.sum()


def combined_error(w, cov):
    """The standard deviation of ``w . x`` when ``x`` has covariance ``cov``."""
    return float(np.sqrt(w @ cov @ w))


def with_rho(s1, s2, rho):
    return np.array([[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]])


def _combine_build(seed):
    rng = np.random.default_rng(seed)
    own = rng.normal(0.0, 1.0, (TOYS, 2)) * STAT
    shared = rng.normal(0.0, 1.0, TOYS)
    x = X_TRUE + own + shared[:, None] * SENS
    cov = shared_cov()
    sig = np.sqrt(np.diag(cov))
    wn, wb = naive_weights(cov), blue_weights(cov)
    claim_n = combined_error(wn, np.diag(sig ** 2))    # what the naive average quotes
    claim_b = combined_error(wb, cov)
    est = np.stack([x[:, 0], x @ wn, x @ wb], 1)
    off = est - X_TRUE
    claims = np.array([sig[0], claim_n, claim_b])
    # the same two errors with nothing shared: the weighted average is then right
    indep = X_TRUE + rng.normal(0.0, 1.0, (TOYS, 2)) * sig
    off_i = indep @ wn - X_TRUE
    lo, hi = x.min(1), x.max(1)
    return dict(
        x=x[:2000], own=own[:3], shared=shared[:3], cov=np.cov(x.T),
        spread=np.sqrt((off ** 2).mean(0)), claims=claims,
        coverage=(np.abs(off) <= claims).mean(0), est=est[:2000],
        outside=np.array([((est[:, k] < lo) | (est[:, k] > hi)).mean() for k in (1, 2)]),
        indep_spread=np.sqrt((off_i ** 2).mean()),
        indep_coverage=(np.abs(off_i) <= claim_n).mean(),
        hist=np.stack([np.histogram(off[:, k], bins=HIST_BINS, range=HIST_RANGE)[0]
                       for k in range(3)]))


def combine_data(seed=COM_SEED):
    return _cache("combine", lambda: _combine_build(seed),
                  (_combine_build, shared_cov, naive_weights, blue_weights,
                   combined_error), seed=seed, toys=TOYS,
                  model=[X_TRUE, STAT.tolist(), SENS.tolist()],
                  hist=[list(HIST_RANGE), HIST_BINS])


def combining_measurements():
    d = combine_data()
    cov = shared_cov()
    sig = np.sqrt(np.diag(cov))
    rho = corr(cov)
    wn, wb = naive_weights(cov), blue_weights(cov)
    actual = np.array([sig[0], combined_error(wn, cov), combined_error(wb, cov)])
    sweep_w = np.array([blue_weights(with_rho(sig[0], sig[1], r)) for r in RHOS])
    sweep_e = np.array([combined_error(w, with_rho(sig[0], sig[1], r))
                        for w, r in zip(sweep_w, RHOS)])
    x = d["x"][0]
    by_seed = [dict(seed=s, spread=[float(v) for v in e["spread"]],
                    coverage=[float(v) for v in e["coverage"]])
               for s, e in ((s, combine_data(s)) for s in COM_SEEDS)]
    names = ["measurement 1 alone", "weights 1/sigma^2", "best linear combination"]
    return Study(
        "Combining measurements",
        "Two measurements of one quantity that share a systematic shift: weights "
        "1/sigma^2 against the best linear combination, over %s pseudo-experiments"
        % format(TOYS, ","),
        (
            Panel("Pairs of measurements",
                  "xy: (x1, x2) in the first 2,000 pseudo-experiments: x_i = %.0f + own "
                  "Gaussian error (%.1f, %.1f) + shared shift (Gaussian, sigma 1) times "
                  "(%.1f, %.1f)." % (X_TRUE, STAT[0], STAT[1], SENS[0], SENS[1]),
                  "points", dict(xy=d["x"], cov=d["cov"], exact=cov, own=d["own"],
                                 shared=d["shared"])),
            Panel("Best weights against the correlation",
                  "For errors %.1f and %.1f and a correlation x between them (exact): "
                  "ys[0], ys[1] the two weights of the best linear combination; ys[2] its "
                  "error." % (sig[0], sig[1]), "curve",
                  dict(x=RHOS, ys=np.array([sweep_w[:, 0], sweep_w[:, 1], sweep_e]),
                       labels=["weight of 1", "weight of 2", "combined error"])),
            Panel("Spread of each estimate about the truth",
                  "values: root-mean-square distance from the true value over the "
                  "pseudo-experiments. exact: sqrt(w^T V w).", "bars",
                  dict(values=d["spread"], exact=actual, labels=names, hist=d["hist"],
                       hist_range=np.array(HIST_RANGE))),
            Panel("Error quoted and coverage",
                  "values: the error each estimate quotes (the naive one from "
                  "1/sqrt(sum 1/sigma^2)). coverage: how often estimate +- that error "
                  "holds the truth.", "bars",
                  dict(values=d["claims"], labels=names, coverage=d["coverage"],
                       coverage_exact=gauss_within(d["claims"] / actual))),
        ),
        dict(
            seed=COM_SEED, seeds=list(COM_SEEDS), review_only=True, toys=TOYS,
            source="exact arithmetic on a stated covariance matrix; pseudo-experiments "
                   "drawn from the same model",
            model=dict(truth=X_TRUE, own_errors=STAT.tolist(), shared_shift=SENS.tolist()),
            evidence=dict(
                exact="weights, combined errors, the sweep over the correlation, the "
                      "arithmetic on the first pair",
                pseudo_experiments="%d pairs per seed: spread, quoted error, coverage"
                                   % TOYS),
            where_the_seed_enters="numpy.random.default_rng(seed).normal: each "
                                  "measurement's own error and each pair's shared shift",
            sigmas=[float(v) for v in sig], rho=rho, covariance=cov.tolist(),
            negative_from=float(sig[0] / sig[1]),
            first_pair=[float(v) for v in x], first_shared=float(d["shared"][0]),
            weights=dict(naive=[float(v) for v in wn], best=[float(v) for v in wb]),
            first_combined=dict(naive=float(x @ wn), best=float(x @ wb)),
            quoted=dict(alone=float(sig[0]), naive=float(d["claims"][1]),
                        best=float(d["claims"][2])),
            actual_exact=dict(alone=float(actual[0]), naive=float(actual[1]),
                              best=float(actual[2])),
            spread=dict(alone=float(d["spread"][0]), naive=float(d["spread"][1]),
                        best=float(d["spread"][2])),
            coverage=dict(alone=float(d["coverage"][0]), naive=float(d["coverage"][1]),
                          best=float(d["coverage"][2])),
            coverage_exact=dict(naive=float(gauss_within(d["claims"][1] / actual[1])),
                                best=ONE_SIGMA),
            outside_both=dict(naive=float(d["outside"][0]), best=float(d["outside"][1])),
            independent=dict(spread=float(d["indep_spread"]),
                             coverage=float(d["indep_coverage"])),
            by_seed=by_seed,
            limitations=[
                "The covariance matrix is taken as known exactly; in practice the "
                "correlation is itself uncertain, and a negative weight leans hard on it.",
                "Errors are Gaussian and do not depend on the value measured.",
            ],
            designs_tried=[
                "Only this design was run: own errors 0.6 and 1.0, shared shifts 0.8 and "
                "2.4, 20,000 pairs per seed, three seeds.",
            ],
        ),
    )


STUDIES = {
    "error-ellipse": error_ellipse,
    "statistical-vs-systematic": statistical_vs_systematic,
    "nuisance-parameters-profiling": nuisance_parameters_profiling,
    "combining-measurements": combining_measurements,
}
