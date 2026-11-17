"""The experiments behind animations 124-127: minimisation and chi-square fitting (see
:func:`voxanim.next_catalog.redesigned`).

* ``chi-square-fit``: six points with error bars and the line y = a + b x. Chi-square
  over (a, b) is evaluated exactly; its minimum is the linear-least-squares solution and
  the contour chi-square_min + 1 is compared with the analytic covariance matrix and with
  10,000 pseudo-experiments.
* ``chi-square-per-dof``: twelve points from a parabola fitted with a line, a parabola
  and a polynomial of degree 7. 20,000 pseudo-experiments give the distribution of each
  chi-square against the exact chi-square density for its degrees of freedom, the fit
  probabilities, what an unneeded parameter buys, and what wrong error bars do.
* ``minimisers-on-chi-square``: a peak's position and width fitted to 24 counts that
  hold a second, smaller peak the model has no term for. Gradient descent, Newton's
  method, the Nelder-Mead simplex and a variable-metric method (the kind MINUIT's MIGRAD
  is) are run from a grid of starts with every evaluation of chi-square counted.
* ``binned-vs-unbinned-fits``: decay times fitted for the lifetime by a binned
  chi-square with sqrt(n) errors, a binned likelihood and an unbinned likelihood, over
  4,000 pseudo-experiments per setting, as the sample shrinks and the bins widen.

Nothing here is learned from data: every dataset is drawn from a stated model by
``numpy.random.default_rng(seed)``, and each result is either exact arithmetic or a count
over seeded pseudo-experiments (``metadata["evidence"]`` says which). No scipy: the
chi-square distribution is :func:`chi2_pdf` and :func:`chi2_sf` below.
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


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-statsfit-" + name, settings,
                                                    build).items()}


# ====================================================================== special functions
def _gamma_p(a, x):
    """Regularised lower incomplete gamma function P(a, x) for scalars: its series for
    x < a + 1, the continued fraction of its complement otherwise."""
    if x <= 0:
        return 0.0
    lead = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1:
        term = total = 1.0 / a
        n = a
        for _ in range(2000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return min(lead * total, 1.0)
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return 1.0 - lead * h


def _gamma_q(a, x):
    """The complement 1 - P(a, x), without the cancellation when it is tiny."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        return 1.0 - _gamma_p(a, x)
    lead = math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return lead * h


def chi2_pdf(x, k):
    """Density of the chi-square distribution with ``k`` degrees of freedom."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.exp((k / 2 - 1) * np.log(x[pos]) - x[pos] / 2 - k / 2 * math.log(2)
                      - math.lgamma(k / 2))
    return out


def chi2_cdf(x, k):
    """Chance that a chi-square with ``k`` degrees of freedom is below ``x``."""
    return np.vectorize(_gamma_p, otypes=[float])(k / 2.0, np.asarray(x, float) / 2.0)


def chi2_sf(x, k):
    """The fit probability (p-value): the chance that a chi-square with ``k`` degrees of
    freedom comes out above ``x``."""
    return np.vectorize(_gamma_q, otypes=[float])(k / 2.0, np.asarray(x, float) / 2.0)


# ====================================================================== 124: chi-square fit
FIT_SEED = 124
FIT_TRUE = (1.0, 0.5)                              # y = a + b x
FIT_X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
FIT_SIGMA = np.array([0.3, 0.4, 0.3, 0.5, 0.4, 0.3])
FIT_START = (2.6, 0.12)                            # the first trial line
FIT_TOYS = 10000
FIT_A, FIT_B = (-0.4, 3.0), (0.05, 0.95)           # the part of the (a, b) plane drawn
FIT_GRID = 121


def line_chi2(a, b, x, y, sigma):
    """Chi-square of the line a + b x: the sum over points of (residual / error) squared.
    ``a`` and ``b`` may be arrays of one shape (then ``y`` is one dataset, or one per
    entry)."""
    a, b = np.asarray(a, float)[..., None], np.asarray(b, float)[..., None]
    return (((y - a - b * x) / sigma) ** 2).sum(-1)


def line_terms(a, b, x, y, sigma):
    """Per point: residual, pull (residual / error) and its square."""
    r = y - a - b * x
    return r, r / sigma, (r / sigma) ** 2


def line_fit(x, y, sigma):
    """The exact least-squares line: ``(a, b)``, their 2 x 2 covariance matrix and the
    sums it is made of. ``y`` may hold many datasets, one per row."""
    w = 1.0 / sigma ** 2
    S, Sx, Sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    Sy, Sxy = (w * y).sum(-1), (w * x * y).sum(-1)
    det = S * Sxx - Sx ** 2
    a, b = (Sxx * Sy - Sx * Sxy) / det, (S * Sxy - Sx * Sy) / det
    cov = np.array([[Sxx, -Sx], [-Sx, S]]) / det
    return a, b, cov, dict(S=S, Sx=Sx, Sxx=Sxx, Sy=Sy, Sxy=Sxy)


def fit_dataset():
    """The six points every phase of animation 124 uses: the true line plus Gaussian
    noise of each point's own error, from seed 124."""
    rng = np.random.default_rng(FIT_SEED)
    return FIT_TRUE[0] + FIT_TRUE[1] * FIT_X + FIT_SIGMA * rng.standard_normal(len(FIT_X))


def fit_path(y):
    """The trial lines the animation moves through: the start, the best slope for that
    intercept, the best intercept for that slope, then the least-squares solution."""
    a, b, _, s = line_fit(FIT_X, y, FIT_SIGMA)
    a0, b0 = FIT_START
    b1 = (s["Sxy"] - a0 * s["Sx"]) / s["Sxx"]
    a1 = (s["Sy"] - b1 * s["Sx"]) / s["S"]
    return np.array([[a0, b0], [a0, b1], [a1, b1], [a, b]])


def _fit_build():
    y = fit_dataset()
    x, sig = FIT_X, FIT_SIGMA
    a, b, cov, _ = line_fit(x, y, sig)
    chi_min = float(line_chi2(a, b, x, y, sig))
    A = np.linspace(*FIT_A, FIT_GRID)
    B = np.linspace(*FIT_B, FIT_GRID)
    surface = line_chi2(A[None, :], B[:, None], x, y, sig)          # rows: b, columns: a
    # the contour chi-square_min + 1, found on a fine grid without using the covariance
    sa, sb = np.sqrt(np.diag(cov))
    fa = np.linspace(a - 2 * sa, a + 2 * sa, 801)
    fb = np.linspace(b - 2 * sb, b + 2 * sb, 801)
    inside = line_chi2(fa[None, :], fb[:, None], x, y, sig) <= chi_min + 1
    ext_a = fa[inside.any(0)]
    ext_b = fb[inside.any(1)]
    # pseudo-experiments: new data from the true line, each fitted exactly
    rng = np.random.default_rng(FIT_SEED + 1)
    ys = FIT_TRUE[0] + FIT_TRUE[1] * x + sig * rng.standard_normal((FIT_TOYS, len(x)))
    ta, tb, _, _ = line_fit(x, ys, sig)
    d = np.stack([ta - FIT_TRUE[0], tb - FIT_TRUE[1]], 1)
    q = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)   # chi2(truth) - chi2_min
    path = fit_path(y)
    row, col = np.unravel_index(surface.argmin(), surface.shape)
    return dict(
        y=y, best=np.array([a, b]), cov=cov, chi_min=chi_min, A=A, B=B, surface=surface,
        contour_half=np.array([(ext_a[-1] - ext_a[0]) / 2, (ext_b[-1] - ext_b[0]) / 2]),
        grid_best=np.array([A[col], B[row]]),
        toy_a=ta, toy_b=tb, toy_q=q, toy_chi=line_chi2(ta, tb, x, ys, sig),
        path=path, path_chi=line_chi2(path[:, 0], path[:, 1], x, y, sig),
    )


def fit_data():
    return _cache("fit", _fit_build, (_fit_build, line_fit, line_chi2, fit_dataset, fit_path),
                  seed=FIT_SEED, toys=FIT_TOYS)


def chi_square_fit():
    d = fit_data()
    a, b = (float(v) for v in d["best"])
    cov = d["cov"]
    sa, sb = (float(v) for v in np.sqrt(np.diag(cov)))
    rho = float(cov[0, 1] / (sa * sb))
    ta, tb, q = d["toy_a"], d["toy_b"], d["toy_q"]
    within_a = float((np.abs(ta - FIT_TRUE[0]) < sa).mean())
    within_b = float((np.abs(tb - FIT_TRUE[1]) < sb).mean())
    inside = float((q <= 1.0).mean())
    panels = (
        Panel("six points and the line through them",
              "y against x with each point's error; the true line and the fitted one",
              "points", dict(xy=np.stack([FIT_X, d["y"]], 1), sigma=FIT_SIGMA,
                             best=d["best"], truth=np.array(FIT_TRUE))),
        Panel("chi-square over intercept and slope",
              "the sum of squared pulls for every line a + b x on a grid", "matrix",
              dict(values=d["surface"], a=d["A"], b=d["B"])),
        Panel("the trial lines", "chi-square at the start, after the best slope, after the "
              "best intercept, and at the least-squares solution", "curve",
              dict(x=np.arange(len(d["path"])), ys=d["path_chi"][None], path=d["path"],
                   labels=["chi-square of the trial line"])),
        Panel("the fit repeated on new data",
              "fitted (a, b) of 10,000 pseudo-experiments drawn from the true line", "points",
              dict(xy=np.stack([ta, tb], 1)[:1500], delta_chi2=q[:1500])),
    )
    return Study(
        "Chi-square as a landscape",
        "a line through six points: the squared pulls, the bowl they make, and its errors",
        panels,
        dict(
            evidence={"data": "six points from y = %g + %g x with Gaussian errors, seed %d"
                              % (FIT_TRUE + (FIT_SEED,)),
                      "surface, minimum, errors": EXACT,
                      "repeats": "%d %s from the true line, seed %d"
                                 % (FIT_TOYS, TOYS, FIT_SEED + 1)},
            seeds=[FIT_SEED, FIT_SEED + 1], true=list(FIT_TRUE), points=len(FIT_X),
            best=[a, b], errors=[sa, sb], correlation=rho, chi_min=float(d["chi_min"]),
            dof=len(FIT_X) - 2,
            start=list(FIT_START), start_chi=float(d["path_chi"][0]),
            contour_half_widths=[float(v) for v in d["contour_half"]],
            grid_minimum=[float(v) for v in d["grid_best"]],
            toys=FIT_TOYS, toy_sd=[float(ta.std(ddof=1)), float(tb.std(ddof=1))],
            toy_mean=[float(ta.mean()), float(tb.mean())],
            toy_correlation=float(np.corrcoef(ta, tb)[0, 1]),
            within_one_sigma=[within_a, within_b], inside_contour=inside,
            inside_contour_exact=1 - math.exp(-0.5), within_exact=math.erf(1 / math.sqrt(2)),
            toy_chi_mean=float(d["toy_chi"].mean()),
            designs_tried=["one design: six points, unequal errors, x from 1 to 6 so that "
                           "intercept and slope are strongly correlated and the contour is "
                           "a tilted ellipse"],
            limitations="a straight line is linear in its parameters, so chi-square is an "
                        "exact paraboloid and chi-square_min + 1 gives the errors exactly; "
                        "for a non-linear model both are approximations. The errors are "
                        "taken as known and Gaussian.",
        ),
    )


# ====================================================================== 125: chi-square / dof
DOF_SEED = 125
DOF_X = np.arange(0.5, 6.01, 0.5)                  # twelve points
DOF_SIGMA = 0.30
DOF_TRUE = (2.0, 1.1, -0.15)                       # y = c0 + c1 x + c2 x^2
DOF_DEGREES = (1, 2, 7)                            # too simple, right, too many
DOF_NAMES = ("straight line", "parabola", "degree 7")
DOF_ALL = tuple(range(0, 9))                       # for chi-square against parameters
DOF_TOYS = 20000
DOF_SCALES = (0.5, 1.0, 2.0)                       # claimed error / true error
DOF_SHOWN = 40                                     # pseudo-experiments drawn one by one


def dof_truth(x):
    c0, c1, c2 = DOF_TRUE
    return c0 + c1 * x + c2 * x * x


def poly_design(x, degree):
    """Columns 1, u, u^2, ... of a polynomial in u, x scaled to -1..1 (the same fits as
    powers of x, well conditioned at degree 8)."""
    u = (x - x.mean()) / (np.ptp(x) / 2)
    return np.stack([u ** k for k in range(degree + 1)], 1)


def poly_hat(x, degree):
    """The matrix that turns measured values (equal errors) into fitted values."""
    D = poly_design(x, degree)
    return D @ np.linalg.pinv(D)


def poly_chi2(y, degree, sigma, x=DOF_X):
    """Minimum chi-square of a polynomial of that degree fitted to ``y`` (rows: datasets)
    and the fitted values."""
    fitted = y @ poly_hat(x, degree).T
    return (((y - fitted) / sigma) ** 2).sum(-1), fitted


def dof_dataset():
    rng = np.random.default_rng(DOF_SEED)
    return dof_truth(DOF_X) + DOF_SIGMA * rng.standard_normal(len(DOF_X))


def _dof_build():
    x, n = DOF_X, len(DOF_X)
    y = dof_dataset()
    rng = np.random.default_rng(DOF_SEED + 1)
    ys = dof_truth(x) + DOF_SIGMA * rng.standard_normal((DOF_TOYS, n))
    fine = np.linspace(x[0] - 0.25, x[-1] + 0.25, 121)
    out = dict(y=y, fine=fine, toys_y=ys[:DOF_SHOWN])
    chi, toy_chi, dist, curves = [], [], [], []
    for deg in DOF_ALL:
        c, _ = poly_chi2(y, deg, DOF_SIGMA)
        tc, tf = poly_chi2(ys, deg, DOF_SIGMA)
        chi.append(c)
        toy_chi.append(tc)
        # how far the fitted curve is from the true one, at the data points, in errors
        dist.append(np.sqrt((((tf - dof_truth(x)) / DOF_SIGMA) ** 2).mean(-1).mean()))
        # the fitted polynomial on a fine axis (same scaled variable as the design)
        u = (fine - x.mean()) / (np.ptp(x) / 2)
        coef = np.linalg.pinv(poly_design(x, deg)) @ y
        curves.append(sum(coef[k] * u ** k for k in range(deg + 1)))
    out.update(chi=np.array(chi), toy_chi=np.array(toy_chi), distance=np.array(dist),
               curves=np.array(curves))
    out["toy_fit"] = poly_chi2(ys[:DOF_SHOWN], 2, DOF_SIGMA)[1]
    return out


def dof_data():
    return _cache("dof", _dof_build, (_dof_build, poly_design, poly_hat, poly_chi2, dof_truth,
                                      dof_dataset),
                  seed=DOF_SEED, toys=DOF_TOYS)


def chi_square_per_dof():
    d = dof_data()
    n = len(DOF_X)
    chi, toy = d["chi"], d["toy_chi"]
    dof = np.array([n - (deg + 1) for deg in DOF_ALL])
    prob = np.array([float(chi2_sf(chi[deg], dof[deg])) for deg in DOF_ALL])
    edges = np.linspace(0.0, 30.0, 61)
    hists = np.array([np.histogram(toy[deg], edges, density=True)[0] for deg in DOF_DEGREES])
    centres = (edges[:-1] + edges[1:]) / 2
    exact = np.array([chi2_pdf(centres, dof[deg]) for deg in DOF_DEGREES])
    right = toy[2]
    k_right = int(dof[2])
    # wrong errors: the same fits, the claimed error a multiple of the true one
    scaled, scaled_now, low, high = [], [], [], []
    for s in DOF_SCALES:
        c = right / s ** 2
        p = chi2_sf(c[:4000], k_right)
        scaled.append(c.mean() / k_right)
        scaled_now.append(chi[2] / s ** 2 / k_right)
        low.append((p < 0.05).mean())
        high.append((p > 0.95).mean())
    red_edges = np.linspace(0.0, 6.0, 61)
    red_hist = np.array([np.histogram(right / s ** 2 / k_right, red_edges, density=True)[0]
                         for s in DOF_SCALES])
    red_centres = (red_edges[:-1] + red_edges[1:]) / 2
    red_exact = np.array([chi2_pdf(red_centres * k_right * s ** 2, k_right) * k_right * s ** 2
                          for s in DOF_SCALES])
    panels = (
        Panel("one dataset, three models", "twelve points from a parabola fitted with a "
              "line, a parabola and a polynomial of degree 7", "curve",
              dict(x=d["fine"], ys=d["curves"][list(DOF_DEGREES)], data_x=DOF_X, data_y=d["y"],
                   chi2=chi[list(DOF_DEGREES)], dof=dof[list(DOF_DEGREES)],
                   probability=prob[list(DOF_DEGREES)], labels=list(DOF_NAMES))),
        Panel("chi-square over 20,000 pseudo-experiments", "histogram of each model's "
              "minimum chi-square, and the exact chi-square density for its degrees of "
              "freedom", "curve", dict(x=centres, ys=hists, exact=exact, labels=list(DOF_NAMES))),
        Panel("chi-square against the number of parameters", "this dataset and the mean "
              "over pseudo-experiments, polynomials of degree 0 to 8", "curve",
              dict(x=np.array(DOF_ALL) + 1.0, ys=np.stack([chi, toy.mean(1)]),
                   distance=d["distance"], labels=["this dataset", "mean of the repeats"])),
        Panel("wrong errors", "chi-square per degree of freedom of the parabola fit when "
              "the claimed errors are half, equal to and twice the true ones", "curve",
              dict(x=red_centres, ys=red_hist, exact=red_exact, mean=np.array(scaled),
                   this=np.array(scaled_now), below_5pc=np.array(low),
                   above_95pc=np.array(high),
                   labels=["claimed error %g x true" % v for v in DOF_SCALES])),
    )
    return Study(
        "What a good chi-square looks like",
        "one dataset, three models: chi-square, degrees of freedom and the fit probability",
        panels,
        dict(
            evidence={"data": "twelve points from y = %g + %g x %+g x^2, Gaussian error "
                              "%g, seed %d" % (DOF_TRUE + (DOF_SIGMA, DOF_SEED)),
                      "fits, densities, fit probabilities": EXACT,
                      "distributions": "%d %s from the true parabola, seed %d"
                                       % (DOF_TOYS, TOYS, DOF_SEED + 1)},
            seeds=[DOF_SEED, DOF_SEED + 1], true=list(DOF_TRUE), sigma=DOF_SIGMA, points=n,
            degrees=list(DOF_DEGREES), toys=DOF_TOYS,
            chi2={str(deg): float(chi[deg]) for deg in DOF_ALL},
            dof={str(deg): int(dof[deg]) for deg in DOF_ALL},
            probability={str(deg): float(prob[deg]) for deg in DOF_ALL},
            toy_mean={str(deg): float(toy[deg].mean()) for deg in DOF_ALL},
            toy_variance_right=float(right.var()),
            gain_per_extra_parameter=float((toy[2] - toy[7]).mean() / 5),
            gain_line_to_parabola=float((toy[1] - toy[2]).mean()),
            distance_from_truth={str(deg): float(d["distance"][deg]) for deg in DOF_ALL},
            distance_exact={str(deg): math.sqrt((deg + 1) / n) for deg in (2, 7)},
            scales=list(DOF_SCALES), reduced_mean=[float(v) for v in scaled],
            reduced_this=[float(v) for v in scaled_now],
            p_below_5pc=[float(v) for v in low], p_above_95pc=[float(v) for v in high],
            p_toys=4000,
            designs_tried=["one design: the true parabola, the error and the seeds were "
                           "set before any fit was looked at"],
            limitations="equal Gaussian errors and models linear in their parameters, for "
                        "which the chi-square distribution is exact. Chi-square per degree "
                        "of freedom does not flag the degree-7 fit: its fit probability is "
                        "as uniform as the right model's. What shows the extra parameters "
                        "are unneeded is that each lowers chi-square by only about 1.",
        ),
    )


# ====================================================================== 126: minimisers
MIN_SEED = 126
MIN_X = np.arange(0.25, 12.0, 0.5)                 # 24 bin centres
MIN_BKG, MIN_HEIGHT = 20.0, 100.0                  # known: flat background, peak height
MIN_PEAK = (4.0, 0.9)                              # the peak: position, width
MIN_SECOND = (70.0, 8.5, 0.8)                      # a neighbour the model has no term for
MIN_BOX = ((0.5, 11.5), (0.25, 3.0))               # mu and sigma allowed
MIN_GRID = (45, 23)                                # starts: mu x sigma
MIN_STARTS = ((5.0, 1.0), (7.0, 1.0))               # the two starts drawn step by step
METHODS = ("gradient descent", "Newton", "simplex", "variable metric")
NEAR = 0.05                                        # "arrived": this close to a minimum
TOL = 1e-4                                         # stop: chi-square or the point moves less


def peak_truth(x):
    h2, m2, s2 = MIN_SECOND
    return (MIN_BKG + MIN_HEIGHT * np.exp(-0.5 * ((x - MIN_PEAK[0]) / MIN_PEAK[1]) ** 2)
            + h2 * np.exp(-0.5 * ((x - m2) / s2) ** 2))


def peak_model(x, mu, sigma):
    """What is fitted: the known background and height, a Gaussian peak at ``mu`` of
    width ``sigma``."""
    return MIN_BKG + MIN_HEIGHT * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def peak_dataset():
    """Counts in 24 bins: Poisson around the true shape, errors sqrt(count)."""
    y = np.random.default_rng(MIN_SEED).poisson(peak_truth(MIN_X)).astype(float)
    return y, np.sqrt(np.maximum(y, 1.0))


def peak_chi2(mu, sigma, y, err):
    mu, sigma = np.asarray(mu, float)[..., None], np.asarray(sigma, float)[..., None]
    return (((y - peak_model(MIN_X, mu, sigma)) / err) ** 2).sum(-1)


class Counted:
    """Chi-square of the peak fit as a function of the point ``(mu, sigma)``, counting
    how often it is evaluated; infinite outside the allowed box."""

    def __init__(self, y, err, log=False):
        self.y, self.err, self.calls = y, err, 0
        self.log = [] if log else None

    def __call__(self, p):
        self.calls += 1
        if self.log is not None:
            self.log.append((float(p[0]), float(p[1])))
        (m0, m1), (s0, s1) = MIN_BOX
        if not (m0 <= p[0] <= m1 and s0 <= p[1] <= s1):
            return math.inf
        return float((((self.y - peak_model(MIN_X, p[0], p[1])) / self.err) ** 2).sum())


def _gradient(f, p, h=1e-4):
    """Slope by central differences: four evaluations."""
    g = np.zeros(2)
    for k in range(2):
        e = np.zeros(2)
        e[k] = h
        g[k] = (f(p + e) - f(p - e)) / (2 * h)
    return g


def _line_search(f, p, fp, d, slope, t):
    """Halve the step ``t`` along ``d`` until chi-square falls by a share of what the
    slope promises. Returns the new point, its chi-square and the step (None: no step)."""
    while t * np.linalg.norm(d) > 1e-9:
        new = p + t * d
        fn = f(new)
        if fn < fp + 1e-4 * t * slope:
            return new, fn, t
        t /= 2
    return None, fp, 0.0


def gradient_descent(f, start, max_evals=4000):
    """Step straight downhill; the step doubles after a success and halves until
    chi-square falls."""
    p = np.array(start, float)
    fp, t, path = f(p), None, [np.array(start, float)]
    while f.calls < max_evals:
        g = _gradient(f, p)
        gn = float(np.linalg.norm(g))
        if not np.isfinite(gn) or gn < 1e-6:
            break
        t = 0.1 / gn if t is None else 2 * t        # the first trial moves 0.1
        new, fn, t = _line_search(f, p, fp, -g, -gn * gn, t)
        if new is None:
            break
        done = fp - fn < TOL
        p, fp = new, fn
        path.append(p.copy())
        if done:
            break
    return np.array(path)


def newton(f, start, max_iter=40):
    """Newton's method, unguarded: fit a paraboloid to chi-square at the point (slope and
    curvature by finite differences, seven evaluations) and jump to its stationary
    point, whether that is a bottom or not."""
    h = 1e-3
    p, path = np.array(start, float), [np.array(start, float)]
    for _ in range(max_iter):
        f0 = f(p)
        e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
        a1, b1, a2, b2 = f(p + e1), f(p - e1), f(p + e2), f(p - e2)
        pp, mm = f(p + e1 + e2), f(p - e1 - e2)
        g = np.array([a1 - b1, a2 - b2]) / (2 * h)
        h11, h22 = (a1 - 2 * f0 + b1) / h ** 2, (a2 - 2 * f0 + b2) / h ** 2
        h12 = (pp + mm - a1 - b1 - a2 - b2 + 2 * f0) / (2 * h ** 2)
        H = np.array([[h11, h12], [h12, h22]])
        if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
            break
        step = -np.linalg.solve(H, g)
        p = p + step
        path.append(p.copy())
        (m0, m1), (s0, s1) = MIN_BOX
        if not (m0 <= p[0] <= m1 and s0 <= p[1] <= s1):
            break                                    # it jumped off the map
        if np.linalg.norm(step) < TOL:
            break
    return np.array(path)


def simplex(f, start, size=(0.5, 0.25), max_evals=600, keep=None):
    """Nelder-Mead: a triangle that reflects its worst corner through the other two,
    stretching when that works and shrinking when it does not. No slopes. ``keep``: a
    list that receives every triangle."""
    pts = np.array([start, [start[0] + size[0], start[1]], [start[0], start[1] + size[1]]],
                   float)
    val = np.array([f(q) for q in pts])
    path = [pts[val.argmin()].copy()]
    while f.calls < max_evals:
        order = np.argsort(val, kind="stable")
        pts, val = pts[order], val[order]
        if keep is not None:
            keep.append(pts.copy())
        if not np.array_equal(path[-1], pts[0]):
            path.append(pts[0].copy())
        if val[2] - val[0] < TOL and np.abs(pts[1:] - pts[0]).max() < 100 * TOL:
            break
        centre = pts[:2].mean(0)
        refl = centre + (centre - pts[2])
        fr = f(refl)
        if fr < val[0]:
            far = centre + 2 * (centre - pts[2])
            ff = f(far)
            pts[2], val[2] = (far, ff) if ff < fr else (refl, fr)
        elif fr < val[1]:
            pts[2], val[2] = refl, fr
        else:
            inner = centre + 0.5 * ((refl if fr < val[2] else pts[2]) - centre)
            fi = f(inner)
            if fi < min(fr, val[2]):
                pts[2], val[2] = inner, fi
            else:
                for k in (1, 2):
                    pts[k] = pts[0] + 0.5 * (pts[k] - pts[0])
                    val[k] = f(pts[k])
    return np.array(path)


def variable_metric(f, start, max_evals=2000):
    """A variable-metric method (the family of MINUIT's MIGRAD): Newton-like steps whose
    curvature estimate is built up from the changes of the slope, kept positive, with a
    line search guarding every step."""
    p = np.array(start, float)
    fp = f(p)
    g = _gradient(f, p)
    V = np.eye(2) * 0.1 / max(float(np.linalg.norm(g)), 1e-12)
    path = [p.copy()]
    while f.calls < max_evals:
        if not np.isfinite(g).all():
            break
        d = -V @ g
        slope = float(g @ d)
        if slope >= 0:                               # not downhill: forget the curvature
            V = np.eye(2) * 0.1 / max(float(np.linalg.norm(g)), 1e-12)
            d = -V @ g
            slope = float(g @ d)
        new, fn, _ = _line_search(f, p, fp, d, slope, 1.0)
        if new is None:
            break
        s = new - p
        g_new = _gradient(f, new)
        yv = g_new - g
        sy = float(s @ yv)
        if np.isfinite(sy) and sy > 1e-12:           # the BFGS update of the inverse
            rho = 1.0 / sy
            left = np.eye(2) - rho * np.outer(s, yv)
            V = left @ V @ left.T + rho * np.outer(s, s)
        done = fp - fn < TOL
        p, fp, g = new, fn, g_new
        path.append(p.copy())
        if done:
            break
    return np.array(path)


RUN = dict(zip(METHODS, (gradient_descent, newton, simplex, variable_metric)))


def run(method, start, y=None, err=None, keep=None):
    """One minimisation: the path of points and the number of chi-square evaluations."""
    if y is None:
        y, err = peak_dataset()
    f = Counted(y, err)
    path = RUN[method](f, start, keep=keep) if keep is not None else RUN[method](f, start)
    return path, f.calls


def run_logged(method, start, y, err, keep=None):
    """As :func:`run`, and for each point of the path the number of evaluations made by
    the time chi-square was first evaluated there (the total, if it never was)."""
    f = Counted(y, err, log=True)
    path = RUN[method](f, start, keep=keep) if keep is not None else RUN[method](f, start)
    first = {}
    for n, q in enumerate(f.log):
        first.setdefault(q, n + 1)
    at = [first.get((float(q[0]), float(q[1])), f.calls) for q in path]
    return path, f.calls, np.maximum.accumulate(np.array(at))


def hessian(start, y, err, h=1e-3):
    """Curvature of chi-square at a point, by the finite differences Newton uses."""
    f = Counted(y, err)
    p = np.array(start, float)
    e1, e2 = np.array([h, 0.0]), np.array([0.0, h])
    f0, a1, b1, a2, b2 = f(p), f(p + e1), f(p - e1), f(p + e2), f(p - e2)
    h12 = (f(p + e1 + e2) + f(p - e1 - e2) - a1 - b1 - a2 - b2 + 2 * f0) / (2 * h ** 2)
    return np.array([[(a1 - 2 * f0 + b1) / h ** 2, h12], [h12, (a2 - 2 * f0 + b2) / h ** 2]])


def peak_minima(y, err, nm=441, ns=221):
    """Every bottom of chi-square inside the box: grid cells lower than their eight
    neighbours, each polished by Newton's method. Lowest first."""
    (m0, m1), (s0, s1) = MIN_BOX
    mu, sg = np.linspace(m0, m1, nm), np.linspace(s0, s1, ns)
    c = peak_chi2(mu[None, :], sg[:, None], y, err)
    inner = c[1:-1, 1:-1]
    low = np.ones_like(inner, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                low &= inner < c[1 + dy:c.shape[0] - 1 + dy, 1 + dx:c.shape[1] - 1 + dx]
    found = []
    for r, k in zip(*np.nonzero(low)):
        end = newton(Counted(y, err), (mu[k + 1], sg[r + 1]))[-1]
        found.append((float(peak_chi2(end[0], end[1], y, err)), float(end[0]), float(end[1])))
    return np.array(sorted(found))


def classify(end, minima):
    """0: at the best fit; 1: at another bottom; 2: neither (stuck, a saddle, off the
    map)."""
    if not np.isfinite(end).all():
        return 2
    d = np.hypot(minima[:, 1] - end[0], minima[:, 2] - end[1])
    k = int(d.argmin())
    return 2 if d[k] > NEAR else min(k, 1)


def _min_build():
    y, err = peak_dataset()
    minima = peak_minima(y, err)
    (m0, m1), (s0, s1) = MIN_BOX
    mu, sg = np.linspace(m0, m1, 221), np.linspace(s0, s1, 111)
    surface = peak_chi2(mu[None, :], sg[:, None], y, err)
    nm, ns = MIN_GRID
    start_mu = m0 + (m1 - m0) * (np.arange(nm) + 0.5) / nm
    start_sg = s0 + (s1 - s0) * (np.arange(ns) + 0.5) / ns
    ends = np.zeros((len(METHODS), ns, nm), np.int64)
    evals = np.zeros((len(METHODS), ns, nm), np.int64)
    for i, method in enumerate(METHODS):
        for r in range(ns):
            for k in range(nm):
                path, calls = run(method, (start_mu[k], start_sg[r]), y, err)
                ends[i, r, k] = classify(path[-1], minima)
                evals[i, r, k] = calls
    out = dict(y=y, err=err, minima=minima, mu=mu, sigma=sg, surface=surface,
               start_mu=start_mu, start_sigma=start_sg, ends=ends, evals=evals)
    for n, start in enumerate(MIN_STARTS):
        for i, method in enumerate(METHODS):
            tri = []
            path, calls, at = run_logged(method, start, y, err,
                                         keep=tri if method == "simplex" else None)
            out["path_%d_%d" % (n, i)] = path
            out["calls_%d_%d" % (n, i)] = calls
            out["at_%d_%d" % (n, i)] = at
            if tri:
                out["triangles_%d" % n] = np.array(tri)
    f = Counted(y, err)
    out["start_gradient"] = _gradient(f, np.array(MIN_STARTS[0]))
    out["start_curvature"] = np.array([np.linalg.eigvalsh(hessian(s, y, err))
                                       for s in MIN_STARTS])
    return out


def min_data():
    return _cache("minimisers", _min_build,
                  (_min_build, peak_truth, peak_model, peak_dataset, peak_chi2, Counted,
                   _gradient, _line_search, gradient_descent, newton, simplex, variable_metric,
                   peak_minima, classify, run, run_logged, hessian),
                  seed=MIN_SEED, grid=list(MIN_GRID), starts=[list(s) for s in MIN_STARTS])


def minimisers_on_chi_square():
    d = min_data()
    ends, evals, minima = d["ends"], d["evals"], d["minima"]
    share = np.array([[(ends[i] == c).mean() for c in range(3)] for i in range(len(METHODS))])
    median = np.array([np.median(evals[i][ends[i] == 0]) if (ends[i] == 0).any() else 0.0
                       for i in range(len(METHODS))])
    demo_calls = np.array([[int(d["calls_%d_%d" % (n, i)]) for i in range(len(METHODS))]
                           for n in range(len(MIN_STARTS))])
    demo_steps = np.array([[len(d["path_%d_%d" % (n, i)]) - 1 for i in range(len(METHODS))]
                           for n in range(len(MIN_STARTS))])
    demo_end = np.array([[classify(d["path_%d_%d" % (n, i)][-1], minima)
                          for i in range(len(METHODS))] for n in range(len(MIN_STARTS))])
    panels = (
        Panel("the counts and the two fits", "24 bins; the model's peak at the best fit and "
              "sitting on the smaller neighbour", "points",
              dict(xy=np.stack([MIN_X, d["y"]], 1), sigma=d["err"], minima=minima)),
        Panel("chi-square over position and width", "the surface the minimisers walk on",
              "matrix", dict(values=d["surface"], mu=d["mu"], sigma=d["sigma"])),
        Panel("where each start ends", "per method and start: 0 the best fit, 1 the other "
              "bottom, 2 neither", "matrix",
              dict(values=ends.reshape(len(METHODS) * ends.shape[1], -1).astype(float),
                   ends=ends, start_mu=d["start_mu"], start_sigma=d["start_sigma"])),
        Panel("the cost", "share of starts reaching the best fit and the median number of "
              "chi-square evaluations of those that do", "bars",
              dict(values=share[:, 0], labels=np.array(METHODS), median_evaluations=median,
                   share=share)),
    )
    return Study(
        "How a minimiser walks",
        "gradient descent, Newton and the simplex on a chi-square surface with two bottoms",
        panels,
        dict(
            evidence={"data": "24 Poisson counts around a flat background of %g, a peak of "
                              "height %g at %g (width %g) and a smaller one of height %g "
                              "at %g (width %g), seed %d"
                              % ((MIN_BKG, MIN_HEIGHT) + MIN_PEAK + MIN_SECOND + (MIN_SEED,)),
                      "surface, paths, basins": EXACT + " (deterministic algorithms; every "
                                                        "evaluation counted)"},
            seeds=[MIN_SEED], methods=list(METHODS), starts=int(ends[0].size),
            minima=[[float(v) for v in row] for row in minima],
            share={m: [float(v) for v in share[i]] for i, m in enumerate(METHODS)},
            median_evaluations={m: float(median[i]) for i, m in enumerate(METHODS)},
            demo_starts=[list(s) for s in MIN_STARTS],
            demo_evaluations=demo_calls.tolist(), demo_steps=demo_steps.tolist(),
            demo_end=demo_end.tolist(),
            start_chi=[float(peak_chi2(s[0], s[1], d["y"], d["err"])) for s in MIN_STARTS],
            start_gradient=[float(v) for v in d["start_gradient"]],
            start_curvature=d["start_curvature"].tolist(),
            designs_tried=["a neighbour of height 45 and width 0.7: run in full, but "
                           "chi-square had one bottom only (on the neighbour the model's "
                           "peak, too tall for it, shrank to the narrowest width allowed)",
                           "the design kept: the neighbour made taller and wider (70, 0.8), "
                           "which gives chi-square a second bottom inside the box; stopping "
                           "tolerances were then loosened from 1e-7 to 1e-4 for all "
                           "methods, since gradient descent spent hundreds of evaluations "
                           "on the last digits"],
            limitations="two parameters so that the surface can be drawn; the peak's height "
                        "and the background are taken as known. Slopes and curvatures are "
                        "finite differences, as in MINUIT, and count as evaluations. Newton "
                        "is run unguarded to show why real tools guard it; the "
                        "variable-metric method is a plain BFGS with a halving line search, "
                        "the family MINUIT's MIGRAD belongs to, not MIGRAD itself. Shares "
                        "of starts depend on the box of starts chosen.",
        ),
    )


# ====================================================================== 127: binned or not
BIN_SEED = 127
TAU, WINDOW = 2.2, 10.0                            # true lifetime, time window (microseconds)
BIN_TOYS = 4000
DEMO_N, DEMO_BINS = 120, 10
SIZES = (30, 60, 120, 250, 500, 1000, 2000, 4000)  # at DEMO_BINS bins
BINNINGS = (50, 20, 10, 5, 3, 2)                   # at SIZE_FOR_BINS events
SIZE_FOR_BINS = 250
FITS = ("binned chi-square", "binned likelihood", "unbinned likelihood")
TAU_GRID = np.exp(np.linspace(np.log(0.4), np.log(12.0), 681))


def decay_pdf(t, tau):
    """Density of a decay time within the window."""
    return np.exp(-t / tau) / (tau * (1 - np.exp(-WINDOW / tau)))


def decay_sample(rng, shape, tau=TAU):
    """Decay times within the window, by the inverse of their cumulative distribution."""
    return -tau * np.log(1 - rng.random(shape) * (1 - np.exp(-WINDOW / tau)))


def bin_probs(tau, edges):
    """Chance of each bin for each lifetime: ``(len(tau), bins)``."""
    c = 1 - np.exp(-edges[None, :] / np.atleast_1d(tau)[:, None])
    return np.diff(c, axis=1) / c[:, -1:]


def _refine(score):
    """The lifetime that maximises each row of ``score`` (given on ``TAU_GRID``): the
    best grid point moved to the top of the parabola through it and its neighbours.
    Also whether the best point was at an end of the grid (no maximum found)."""
    k = score.argmax(1)
    edge = (k == 0) | (k == score.shape[1] - 1)
    k = np.clip(k, 1, score.shape[1] - 2)
    rows = np.arange(len(score))
    y0, y1, y2 = score[rows, k - 1], score[rows, k], score[rows, k + 1]
    bend = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        shift = np.where(bend < 0, 0.5 * (y0 - y2) / bend, 0.0)
    step = np.log(TAU_GRID[1] / TAU_GRID[0])
    return np.exp(np.log(TAU_GRID[k]) + np.clip(shift, -1, 1) * step), edge


def chi2_score(counts, edges, keep_empty=False):
    """Minus the binned chi-square (up to a constant) on ``TAU_GRID``, the normalisation
    fitted too. Each bin's error is the square root of its own count, and empty bins,
    whose error would be zero, are left out: chi-square = sum (n - A p)^2 / n, which at
    its best A is sum n - (sum p)^2 / sum (p^2 / n). ``keep_empty`` keeps the empty bins
    instead, with an error of 1 (the check of what leaving them out does)."""
    p = bin_probs(TAU_GRID, edges)
    filled = counts > 0
    sp = filled.astype(float) @ p.T
    weight = 1.0 / np.maximum(counts, 1)
    spp = (weight if keep_empty else np.where(filled, weight, 0.0)) @ (p ** 2).T
    return sp ** 2 / spp


def chi2_norm(counts, edges, tau):
    """The normalisation A that minimises the binned chi-square at lifetime ``tau``."""
    p = bin_probs(np.array([tau]), edges)[0]
    filled = counts > 0
    return p[filled].sum() / (p[filled] ** 2 / counts[filled]).sum()


def binned_score(counts, edges):
    """The binned log-likelihood on ``TAU_GRID``: sum n ln p (Poisson counts whose total
    is fitted too)."""
    return counts @ np.log(bin_probs(TAU_GRID, edges)).T


def unbinned_score(n, total):
    """The unbinned log-likelihood on ``TAU_GRID``: the sum over events of ln f(t)."""
    g = TAU_GRID[None, :]
    n, total = np.asarray(n, float)[:, None], np.asarray(total, float)[:, None]
    return -n * np.log(g) - n * np.log(1 - np.exp(-WINDOW / g)) - total / g


def fit_all(times, bins):
    """The three fitted lifetimes of each row of decay times, and for each fit the share
    of rows where no maximum was found inside the grid."""
    edges = np.linspace(0.0, WINDOW, bins + 1)
    idx = np.minimum((times / WINDOW * bins).astype(np.int64), bins - 1)
    counts = np.zeros((len(times), bins))
    np.add.at(counts, (np.arange(len(times))[:, None], idx), 1.0)
    fits = [_refine(chi2_score(counts, edges)), _refine(binned_score(counts, edges)),
            _refine(unbinned_score(np.full(len(times), times.shape[1]), times.sum(1)))]
    return np.array([f[0] for f in fits]), np.array([f[1] for f in fits]), counts


def exact_spread(n, tau=TAU):
    """The smallest spread any unbiased estimate of the lifetime can have from ``n``
    decay times within the window (the Cramer-Rao bound, from the Fisher information)."""
    e = np.exp(-WINDOW / tau)
    info = 1 / tau ** 2 - WINDOW ** 2 / tau ** 4 * e / (1 - e) ** 2
    return 1 / np.sqrt(n * info)


def _summary(times, bins):
    est, edge, counts = fit_all(times, bins)
    ok = ~edge.any(0)
    good = est[:, ok]
    kept, kept_edge = _refine(chi2_score(counts, np.linspace(0.0, WINDOW, bins + 1), True))
    return (good.mean(1) - TAU, good.std(1, ddof=1), np.median(good, 1) - TAU, edge.mean(1),
            est, kept[ok & ~kept_edge].mean() - TAU)


def _bin_build():
    rng = np.random.default_rng(BIN_SEED)
    demo = np.sort(decay_sample(rng, DEMO_N))
    est, _, counts = fit_all(demo[None], DEMO_BINS)
    out = dict(demo_times=demo, demo_counts=counts[0], demo_fit=est[:, 0])
    rows = {"size": [(n, DEMO_BINS) for n in SIZES],
            "bins": [(SIZE_FOR_BINS, b) for b in BINNINGS]}
    for which, settings in rows.items():
        bias, spread, median, failed, kept = [], [], [], [], []
        for n, bins in settings:
            rng = np.random.default_rng([BIN_SEED, n, bins])
            b, s, m, f, est, k = _summary(decay_sample(rng, (BIN_TOYS, n)), bins)
            kept.append(k)
            bias.append(b)
            spread.append(s)
            median.append(m)
            failed.append(f)
            if (n, bins) == (DEMO_N, DEMO_BINS):
                out["demo_estimates"] = est
        out.update({which + "_bias": np.array(bias), which + "_spread": np.array(spread),
                    which + "_median": np.array(median), which + "_failed": np.array(failed),
                    which + "_kept": np.array(kept)})
    return out


def bin_data():
    return _cache("binned", _bin_build,
                  (_bin_build, decay_sample, bin_probs, _refine, chi2_score, binned_score,
                   unbinned_score, fit_all, _summary),
                  seed=BIN_SEED, toys=BIN_TOYS, sizes=list(SIZES), binnings=list(BINNINGS),
                  demo=[DEMO_N, DEMO_BINS], tau=TAU, window=WINDOW)


def binned_vs_unbinned_fits():
    d = bin_data()
    sizes, binnings = np.array(SIZES, float), np.array(BINNINGS, float)
    edges = np.linspace(0.0, WINDOW, DEMO_BINS + 1)
    expected = DEMO_N * bin_probs(np.array([TAU]), edges)[0]
    est = d["demo_estimates"]
    hist_edges = np.linspace(1.2, 3.4, 45)
    hists = np.array([np.histogram(e, hist_edges)[0] for e in est])
    k = SIZES.index(DEMO_N)
    panels = (
        Panel("one sample of decay times", "120 decay times, their counts in ten bins, and "
              "the lifetime each fit gives", "points",
              dict(xy=np.stack([d["demo_times"], np.zeros(DEMO_N)], 1),
                   counts=d["demo_counts"], expected=expected, fitted=d["demo_fit"])),
        Panel("the fitted lifetime over 4,000 pseudo-experiments", "histograms of the three "
              "fits' lifetimes at 120 events in ten bins", "curve",
              dict(x=(hist_edges[:-1] + hist_edges[1:]) / 2, ys=hists.astype(float),
                   edges=hist_edges, labels=list(FITS))),
        Panel("smaller samples", "bias and spread of the fitted lifetime against the number "
              "of events, ten bins", "curve",
              dict(x=sizes, ys=d["size_bias"].T, spread=d["size_spread"].T,
                   exact_spread=exact_spread(sizes), failed=d["size_failed"].T,
                   labels=list(FITS))),
        Panel("coarser bins", "bias and spread against the number of bins, 250 events",
              "curve", dict(x=binnings, ys=d["bins_bias"].T, spread=d["bins_spread"].T,
                            exact_spread=np.full(len(BINNINGS), exact_spread(SIZE_FOR_BINS)),
                            failed=d["bins_failed"].T, labels=list(FITS))),
    )
    return Study(
        "Binned or unbinned?",
        "a lifetime fitted by a binned chi-square, a binned likelihood and an unbinned one",
        panels,
        dict(
            evidence={"data": "decay times from exp(-t / tau), tau = %g microseconds, kept "
                              "within 0 to %g; seed %d" % (TAU, WINDOW, BIN_SEED),
                      "bias and spread": "%d %s per setting, the same samples for the "
                                         "three fits" % (BIN_TOYS, TOYS),
                      "smallest possible spread": EXACT + " (Cramer-Rao)"},
            seeds=[BIN_SEED], tau=TAU, window=WINDOW, toys=BIN_TOYS, fits=list(FITS),
            demo=dict(events=DEMO_N, bins=DEMO_BINS,
                      fitted=[float(v) for v in d["demo_fit"]],
                      counts=[int(v) for v in d["demo_counts"]],
                      bias=[float(v) for v in d["size_bias"][k]],
                      spread=[float(v) for v in d["size_spread"][k]],
                      exact_spread=float(exact_spread(DEMO_N))),
            sizes=list(SIZES), binnings=list(BINNINGS), size_for_bins=SIZE_FOR_BINS,
            bias_by_size=d["size_bias"].tolist(), spread_by_size=d["size_spread"].tolist(),
            bias_by_bins=d["bins_bias"].tolist(), spread_by_bins=d["bins_spread"].tolist(),
            failed_by_size=d["size_failed"].tolist(),
            failed_by_bins=d["bins_failed"].tolist(),
            chi2_bias_keeping_empty_bins=dict(by_size=d["size_kept"].tolist(),
                                              by_bins=d["bins_kept"].tolist()),
            standard_error_of_bias="spread / sqrt(%d)" % BIN_TOYS,
            designs_tried=["one design, fixed before the comparison: the chi-square is "
                           "Neyman's (each bin's error the square root of its own count, "
                           "empty bins left out), the form a least-squares fit of a "
                           "histogram uses by default; all three fits leave the "
                           "normalisation free"],
            limitations="one parameter of interest and no background. The chi-square with "
                        "errors taken from the fitted curve instead of the data (Pearson's) "
                        "was not run. Pseudo-experiments whose best lifetime lay at an end "
                        "of the scanned range (0.4 to 12) are left out of the means for "
                        "all three fits; their share is recorded.",
        ),
    )


STUDIES = {
    "chi-square-fit": chi_square_fit,
    "chi-square-per-dof": chi_square_per_dof,
    "minimisers-on-chi-square": minimisers_on_chi_square,
    "binned-vs-unbinned-fits": binned_vs_unbinned_fits,
}
