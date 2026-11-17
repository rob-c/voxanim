"""The experiments behind animations 145-149: discoveries, exclusion and upper limits
(see :func:`voxanim.next_catalog.redesigned`).

Nothing here is measured data. Every spectrum and every count is *pseudo-data*: drawn
by a seeded ``numpy.random.default_rng`` from a model whose true values are constants
of this module and are stated on screen. Two kinds of evidence result, and every phase
of the drawings says which it shows:

* **exact**: a formula or a distribution evaluated (a Poisson tail, a limit solved from
  one, a Gaussian tail probability, the coverage of a limit summed over every count);
* **pseudo-experiments**: the same analysis repeated on thousands of seeded draws.

Wherever both exist they are recorded side by side.

* ``bump-hunt``: one binned mass spectrum with a small injected signal; Poisson
  maximum-likelihood fits of background and of signal plus background; ``q0`` and
  ``Z = sqrt(q0)``; the distribution of ``q0`` in background-only pseudo-experiments
  against half a chi-square; the scan in mass; the global significance.
* ``poisson-upper-limits``: the classical upper limit on a Poisson mean from ``n``
  observed events, with and without a known background, and its coverage.
* ``cls-method``: the plain (CLs+b) limit against the CLs limit in a counting
  experiment with background.
* ``brazil-band``: expected-limit bands from background-only pseudo-experiments at
  twelve mass points, and one "observed" limit with a signal injected at one of them.
* ``why-five-sigma``: twenty-bin searches followed as luminosity accumulates.

scipy is not used: the special functions needed (normal tail and its inverse, Poisson
probabilities) are written here and tested against tabulated values.
"""

from __future__ import annotations

import hashlib
import inspect
import math

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in (*sources, norm_sf, norm_isf, poisson_pmf, poisson_cdf, upper_limit):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-discovery-" + name, settings,
                                                    build).items()}


# ====================================================================== special functions
_erfc = np.vectorize(math.erfc, otypes=[float])


def norm_sf(z):
    """One-sided tail of the standard normal: the chance of a value above ``z``
    (``math.erfc`` keeps it accurate far into the tail: 2.87e-7 at 5)."""
    return 0.5 * _erfc(np.asarray(z, float) / math.sqrt(2.0))


def norm_isf(p):
    """The ``z`` whose one-sided tail is ``p``: a p-value as a number of sigma
    (bisection on :func:`norm_sf`)."""
    p = np.asarray(p, float)
    lo, hi = np.full(p.shape, -10.0), np.full(p.shape, 40.0)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        above = norm_sf(mid) > p
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi)


def _log_factorial(n_max):
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, n_max + 1)))])


def poisson_pmf(k, mu):
    """Chance of exactly ``k`` events when ``mu`` are expected (arrays broadcast)."""
    k = np.asarray(k)
    mu = np.asarray(mu, float)
    lf = _log_factorial(int(k.max()) if k.size else 0)[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(mu > 0, k * np.log(np.where(mu > 0, mu, 1.0)) - mu - lf,
                         np.where(k == 0, 0.0, -np.inf))
    return np.exp(log_p)


def poisson_cdf(n, mu):
    """Chance of ``n`` events or fewer when ``mu`` are expected (arrays broadcast)."""
    n, mu = np.broadcast_arrays(np.asarray(n), np.asarray(mu, float))
    top = max(int(n.max()), 0) if n.size else 0
    k = np.arange(top + 1)
    table = np.cumsum(poisson_pmf(k, mu[..., None]), axis=-1)
    return np.take_along_axis(table, np.maximum(n, 0)[..., None], axis=-1)[..., 0] * (n >= 0)


def upper_limit(n, b=0.0, cl=0.90, method="plain"):
    """Upper limit on a signal mean ``s`` from ``n`` observed events with a known
    background mean ``b``: the ``s`` at which the chance of ``n`` or fewer has fallen to
    ``1 - cl``.

    ``plain`` uses that chance itself, ``P(N <= n | s + b)`` (CLs+b; the limit is
    negative when even ``s = 0`` gives ``n`` or fewer that rarely); ``cls`` divides it
    by the same chance with no signal, ``P(N <= n | b)`` (never negative)."""
    n = np.asarray(n)
    alpha = 1.0 - cl
    if method == "cls":
        alpha = alpha * poisson_cdf(n, b)
    lo = np.zeros(n.shape)
    hi = n + 10.0 * np.sqrt(n + 4.0) + 40.0 + b
    for _ in range(60):                               # the tail falls as the mean grows
        mid = 0.5 * (lo + hi)
        above = poisson_cdf(n, mid) > alpha
        lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
    return 0.5 * (lo + hi) - b


def z_counting(n, b):
    """Significance of ``n`` events over a known background ``b``, from the likelihood
    ratio of a counting experiment: ``Z = sqrt(2 (n ln(n / b) - (n - b)))``, given the
    sign of ``n - b`` (a deficit is negative)."""
    n, b = np.broadcast_arrays(np.asarray(n, float), np.asarray(b, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 2.0 * (np.where(n > 0, n * np.log(np.where(n > 0, n, 1.0) / b), 0.0) - (n - b))
    return np.sign(n - b) * np.sqrt(np.maximum(q, 0.0))


def z_asimov(s, b):
    """The same ``Z`` for the expected counts: ``n = s + b`` exactly."""
    return z_counting(np.asarray(s, float) + b, b)


# ====================================================================== 145: a bump hunt
BUMP_SEED = 145                    # of the one spectrum analysed
BUMP_TOY_SEEDS = (1451, 1452, 1453)   # background-only at one mass; the scan; with signal
M_LO, M_HI, BIN_W = 100.0, 160.0, 2.0
N_BKG, TAU = 3000.0, 25.0          # true background: events, and the fall length of exp(-m/tau)
N_SIG, M_SIG, WIDTH = 70.0, 125.0, 2.0   # true signal: events, mass, resolution (known)
SCAN = np.arange(106.0, 154.5, 1.0)       # masses tested
WINDOW = (121.0, 129.0)            # M_SIG +- 2 WIDTH: the counting check drawn first
N_FIXED = 100000                   # background-only pseudo-experiments fitted at M_SIG
N_SCAN = 20000                     # background-only pseudo-experiments scanned in mass
N_WITH = 5000                      # pseudo-experiments with the signal, fitted at M_SIG
FIT_STEPS = 7
Q_EDGES = np.arange(0.0, 20.01, 0.5)
ZMAX_EDGES = np.arange(0.0, 5.01, 0.25)


def bump_axis():
    """Bin edges, bin centres, and the centres rescaled to about -1..1 for the fit."""
    edges = np.arange(M_LO, M_HI + 1e-9, BIN_W)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return edges, centres, (centres - 0.5 * (M_LO + M_HI)) / (0.5 * (M_HI - M_LO))


def signal_shape(masses):
    """Share of a signal at each of ``masses`` falling in each bin: a Gaussian of the
    known resolution ``WIDTH``, integrated over the bin."""
    edges = bump_axis()[0]
    upper = norm_sf((edges[None, :] - np.asarray(masses, float)[:, None]) / WIDTH)
    return upper[:, :-1] - upper[:, 1:]


def bump_truth():
    """Expected events per bin of the true background and of the true signal."""
    edges = bump_axis()[0]
    tail = np.exp(-(edges - M_LO) / TAU)
    return (N_BKG * (tail[:-1] - tail[1:]) / (tail[0] - tail[-1]),
            N_SIG * signal_shape(np.array([M_SIG]))[0])


def log_likelihood(n, mu):
    """Poisson log-likelihood of the bin contents ``n`` for expectations ``mu`` (the
    constant ``-sum ln n!`` left out: it cancels in every ratio)."""
    return (n * np.log(mu) - mu).sum(-1)


def fit_background(n, steps=FIT_STEPS):
    """Maximum-likelihood fit of ``mu_i = exp(a + c t_i)`` to bin contents ``n``
    (``(..., bins)``) by Newton's method. Returns ``(a, c)`` stacked last."""
    t = bump_axis()[2]
    n = np.asarray(n, float)
    theta = np.zeros(n.shape[:-1] + (2,))
    theta[..., 0] = np.log(np.maximum(n.mean(-1), 1e-3))
    X = np.stack([np.ones_like(t), t], -1)
    for _ in range(steps):
        mu = np.exp(theta @ X.T)
        grad = (n - mu) @ X
        info = np.einsum("...i,ij,ik->...jk", mu, X, X)
        theta = theta + np.linalg.solve(info, grad[..., None])[..., 0]
    return theta


def fit_signal(n, shapes, start, steps=FIT_STEPS):
    """Maximum-likelihood fit of ``mu_i = exp(a + c t_i) + s g_i`` for each signal shape
    ``g`` (``(masses, bins)``) to each spectrum ``n`` (``(toys, bins)``), by Fisher
    scoring from the background-only fit ``start`` and ``s = 0``. ``s`` may come out
    negative (a dip). Returns ``theta`` ``(toys, masses, 3)``, the expectations and the
    largest gradient left (0 at a maximum)."""
    t = bump_axis()[2]
    n = np.asarray(n, float)[:, None, :]
    g = np.asarray(shapes, float)[None]
    theta = np.zeros((n.shape[0], g.shape[1], 3))
    theta[..., :2] = np.asarray(start)[:, None, :]
    left = 0.0
    for step in range(steps + 1):
        back = np.exp(theta[..., 0:1] + theta[..., 1:2] * t)
        mu = np.maximum(back + theta[..., 2:3] * g, 1e-6)
        J = np.stack([back, back * t, np.broadcast_to(g, mu.shape)], -1)
        grad = np.einsum("...i,...ij->...j", n / mu - 1.0, J)
        if step == steps:
            left = float(np.abs(grad).max())
            break
        info = np.einsum("...ij,...ik->...jk", J / mu[..., None], J)
        theta = theta + np.linalg.solve(info, grad[..., None])[..., 0]
    return theta, mu, left


def q0_scan(n, masses, chunk=400):
    """For each spectrum and each tested mass: the discovery test statistic
    ``q0 = 2 (ln L(signal + background) - ln L(background))``, set to 0 when the fitted
    signal is negative; and the fitted signal. Also the largest gradient left."""
    n = np.atleast_2d(np.asarray(n, float))
    shapes = signal_shape(masses)
    t = bump_axis()[2]
    q, s_hat, left = [], [], 0.0
    for i in range(0, len(n), chunk):
        part = n[i:i + chunk]
        theta_b = fit_background(part)
        ll_b = log_likelihood(part, np.exp(theta_b[:, 0:1] + theta_b[:, 1:2] * t))
        theta, mu, g = fit_signal(part, shapes, theta_b)
        ll = log_likelihood(part[:, None, :], mu)
        q.append(np.where(theta[..., 2] > 0, np.maximum(2.0 * (ll - ll_b[:, None]), 0.0), 0.0))
        s_hat.append(theta[..., 2])
        left = max(left, g)
    return np.concatenate(q), np.concatenate(s_hat), left


def _bump_build():
    edges, centres, t = bump_axis()
    back, sig = bump_truth()
    n = np.random.default_rng(BUMP_SEED).poisson(back + sig).astype(float)
    # ---- the two fits at the true mass
    theta_b = fit_background(n[None])
    fit_b = np.exp(theta_b[0, 0] + theta_b[0, 1] * t)
    shape = signal_shape(np.array([M_SIG]))
    theta, mu, left = fit_signal(n[None], shape, theta_b)
    theta = theta[0, 0]
    fit_sb = mu[0, 0]
    fit_sb_back = np.exp(theta[0] + theta[1] * t)
    J = np.stack([fit_sb_back, fit_sb_back * t, shape[0]], -1)
    s_err = float(np.sqrt(np.linalg.inv((J / fit_sb[:, None]).T @ J)[2, 2]))
    ll_b, ll_sb = float(log_likelihood(n, fit_b)), float(log_likelihood(n, fit_sb))
    q_obs = 2.0 * (ll_sb - ll_b) if theta[2] > 0 else 0.0
    inside = (centres > WINDOW[0]) & (centres < WINDOW[1])
    # ---- the scan of the one spectrum
    scan_q, scan_s, left_scan = q0_scan(n[None], SCAN)
    scan_z = np.sqrt(scan_q[0])
    best = int(np.argmax(scan_z))
    # ---- background-only pseudo-experiments at the one mass: is q0 half a chi-square?
    toys = np.random.default_rng(BUMP_TOY_SEEDS[0]).poisson(back, (N_FIXED, len(back)))
    q_fixed, _, left_fixed = q0_scan(toys, np.array([M_SIG]), chunk=20000)
    q_fixed = q_fixed[:, 0]
    at = np.array([1.0, 2.0, 3.0])
    # ---- background-only pseudo-experiments, scanned: the largest Z anywhere
    toys = np.random.default_rng(BUMP_TOY_SEEDS[1]).poisson(back, (N_SCAN, len(back)))
    q_toys, _, left_toys = q0_scan(toys, SCAN)
    z_max = np.sqrt(q_toys.max(1))
    # ---- pseudo-experiments with the signal: what this signal typically gives
    toys = np.random.default_rng(BUMP_TOY_SEEDS[2]).poisson(back + sig, (N_WITH, len(back)))
    q_with, s_with, left_with = q0_scan(toys, np.array([M_SIG]), chunk=5000)
    z_with = np.sqrt(q_with[:, 0])
    return dict(
        edges=edges, centres=centres, truth_back=back, truth_sig=sig, n=n,
        fit_b=fit_b, fit_sb=fit_sb, fit_sb_back=fit_sb_back, theta_b=theta_b[0], theta=theta,
        s_hat=np.array(theta[2]), s_err=np.array(s_err),
        ll_b=np.array(ll_b), ll_sb=np.array(ll_sb), q_obs=np.array(q_obs),
        window_n=np.array(n[inside].sum()), window_b=np.array(fit_b[inside].sum()),
        scan_q=scan_q[0], scan_z=scan_z, scan_s=scan_s[0],
        scan_p=norm_sf(scan_z), best=np.array(best),
        q_hist=np.histogram(q_fixed, Q_EDGES)[0], q_zero=np.array((q_fixed == 0).mean()),
        q_tail_at=at, q_tail=np.array([(q_fixed >= z * z).mean() for z in at]),
        q_tail_n=np.array([(q_fixed >= z * z).sum() for z in at]),
        z_max=z_max, beyond=np.array(int((z_max >= scan_z[best]).sum())),
        z_with=np.percentile(z_with, [16, 50, 84]), s_with=np.array(s_with[:, 0].mean()),
        gradient_left=np.array(max(left, left_scan, left_fixed, left_toys, left_with)),
    )


def bump_data():
    """Everything ``bump-hunt`` measures and draws."""
    return _cache("bump", _bump_build,
                  (_bump_build, bump_axis, bump_truth, signal_shape, fit_background,
                   fit_signal, q0_scan, log_likelihood),
                  seed=BUMP_SEED, toy_seeds=BUMP_TOY_SEEDS, truth=(N_BKG, TAU, N_SIG, M_SIG, WIDTH),
                  axis=(M_LO, M_HI, BIN_W), scan=[float(SCAN[0]), float(SCAN[-1]), len(SCAN)],
                  toys=(N_FIXED, N_SCAN, N_WITH), steps=FIT_STEPS)


def bump_summary(d=None):
    """The numbers of the bump hunt that the drawing and the tests quote."""
    d = bump_data() if d is None else d
    best = int(d["best"])
    z_local = float(d["scan_z"][best])
    p_local = float(norm_sf(z_local))
    beyond = int(d["beyond"])
    p_global = beyond / N_SCAN
    return dict(
        z_fixed=float(np.sqrt(d["q_obs"])), q_fixed=float(d["q_obs"]),
        p_fixed=float(norm_sf(np.sqrt(d["q_obs"]))),
        best_mass=float(SCAN[best]), z_local=z_local, p_local=p_local,
        beyond=beyond, p_global=p_global,
        p_global_error=float(np.sqrt(max(beyond, 1)) / N_SCAN),
        z_global=float(norm_isf(max(p_global, 0.5 / N_SCAN))),
        trials=p_global / p_local,
        window_excess=float(d["window_n"] - d["window_b"]),
        window_z=float((d["window_n"] - d["window_b"]) / np.sqrt(d["window_b"])),
    )


def bump_hunt():
    d = bump_data()
    r = bump_summary(d)
    mid = 0.5 * (Q_EDGES[:-1] + Q_EDGES[1:])
    exact = norm_sf(d["q_tail_at"])
    half_chi2 = norm_sf(np.sqrt(Q_EDGES[:-1])) - norm_sf(np.sqrt(Q_EDGES[1:]))
    half_chi2[0] += 0.5
    return Study(
        "A bump hunt",
        "One pseudo-data mass spectrum with a small injected signal: two fits, q0, the "
        "scan in mass and the global significance",
        (
            Panel("The spectrum and its two fits",
                  "ys: events per %g-wide mass bin of one pseudo-data spectrum (seed %d): "
                  "Poisson draws around %d background events falling as exp(-m / %g) plus %d "
                  "signal events at m = %g of width %g; the background-only maximum-"
                  "likelihood fit; the signal-plus-background fit at m = %g; the true "
                  "expectation." % (BIN_W, BUMP_SEED, N_BKG, TAU, N_SIG, M_SIG, WIDTH, M_SIG),
                  "curve", dict(x=d["centres"],
                                ys=np.array([d["n"], d["fit_b"], d["fit_sb"],
                                             d["truth_back"] + d["truth_sig"]]),
                                labels=["pseudo-data", "background fit",
                                        "signal + background fit", "truth"],
                                edges=d["edges"])),
            Panel("q0 in background-only pseudo-experiments",
                  "ys[0]: share of %d background-only spectra whose q0 at m = %g falls in "
                  "each bin of width 0.5 (the first bin holds the %.1f%% with a negative "
                  "fitted signal, q0 = 0). ys[1]: the same shares for half a chi-square "
                  "with one degree of freedom plus half at zero (exact). tails: the share "
                  "with sqrt(q0) above 1, 2, 3 against the normal tail."
                  % (N_FIXED, M_SIG, 100 * d["q_zero"]),
                  "curve", dict(x=mid, ys=np.array([d["q_hist"] / N_FIXED, half_chi2]),
                                labels=["pseudo-experiments", "half chi-square (exact)"],
                                tails=d["q_tail"], tails_exact=exact, tails_at=d["q_tail_at"],
                                tail_counts=d["q_tail_n"])),
            Panel("Local p-value against the mass tested",
                  "ys[0]: the local p-value of the spectrum at each tested mass, the "
                  "normal tail beyond Z = sqrt(q0) (0.5 where the fitted signal is "
                  "negative). ys[1]: Z itself. fitted: the fitted signal in events.",
                  "curve", dict(x=SCAN, ys=np.array([d["scan_p"], d["scan_z"]]),
                                labels=["local p-value", "local Z"], fitted=d["scan_s"])),
            Panel("The largest local Z anywhere in the scan, background only",
                  "values: for each of %d background-only spectra, the largest local Z "
                  "over the %d masses tested, counted in bins of 0.25. The global p-value "
                  "is the share at or above the observed largest Z." % (N_SCAN, len(SCAN)),
                  "bars", dict(values=np.histogram(d["z_max"], ZMAX_EDGES)[0].astype(float),
                               labels=["%.2f" % v for v in ZMAX_EDGES[:-1]],
                               edges=ZMAX_EDGES, z_max=d["z_max"])),
        ),
        dict(
            seed=BUMP_SEED, seeds=[BUMP_SEED, *BUMP_TOY_SEEDS], review_only=True,
            source="pseudo-data from a stated model; no measured data",
            evidence=dict(
                pseudo_experiments="the spectrum (seed %d); %d background-only spectra "
                                   "fitted at the true mass; %d scanned over %d masses; %d "
                                   "with the signal" % (BUMP_SEED, N_FIXED, N_SCAN, len(SCAN),
                                                        N_WITH),
                exact="the half chi-square distribution; p-values from Z by the normal tail"),
            truth=dict(background_events=N_BKG, fall_length=TAU, signal_events=N_SIG,
                       signal_mass=M_SIG, resolution=WIDTH, bins=len(d["n"]), bin_width=BIN_W),
            events=int(d["n"].sum()),
            fitted_signal=float(d["s_hat"]), fitted_signal_error=float(d["s_err"]),
            log_likelihood=dict(background=float(d["ll_b"]),
                                signal_plus_background=float(d["ll_sb"])),
            q0=r["q_fixed"], z_at_true_mass=r["z_fixed"], p_at_true_mass=r["p_fixed"],
            window=dict(range=list(WINDOW), observed=float(d["window_n"]),
                        background_fit=float(d["window_b"]), z=r["window_z"]),
            q0_zero_share=float(d["q_zero"]),
            q0_tails=dict(sigma=[1, 2, 3], pseudo_experiments=[float(v) for v in d["q_tail"]],
                          exact=[float(v) for v in exact],
                          counts=[int(v) for v in d["q_tail_n"]]),
            scan=dict(masses=len(SCAN), first=float(SCAN[0]), last=float(SCAN[-1]),
                      best_mass=r["best_mass"], local_z=r["z_local"], local_p=r["p_local"]),
            global_=dict(pseudo_experiments=N_SCAN, at_or_above=r["beyond"],
                         p=r["p_global"], p_error=r["p_global_error"], z=r["z_global"],
                         trials_factor=r["trials"]),
            with_signal=dict(pseudo_experiments=N_WITH,
                             z_16_50_84=[float(v) for v in d["z_with"]],
                             mean_fitted_signal=float(d["s_with"])),
            largest_gradient_left=float(d["gradient_left"]),
            limitations=[
                "The background's shape (an exponential) and the signal's width are taken "
                "as known; a real search must also choose the background function, and "
                "that choice is a systematic uncertainty.",
                "The pseudo-experiments are thrown from the true background, which a real "
                "analysis does not have: it throws them from its background fit.",
                "The global p-value rests on the few pseudo-experiments beyond the "
                "observed Z: its relative error is about one over the square root of "
                "their number.",
                "One spectrum, seed %d, chosen before it was looked at: the 16-50-84%% "
                "range of Z for this signal over %d spectra is recorded beside it."
                % (BUMP_SEED, N_WITH),
            ],
            designs_tried=[
                "Only this design was run: the signal's size (%d events on %d) was fixed "
                "for an expected Z near 3.5 before the spectrum was drawn, the seed is "
                "the animation's number." % (N_SIG, N_BKG),
            ],
        ),
    )


# ====================================================================== 146: upper limits
LIMIT_SEED = 146
CL = 0.90
LIMIT_N = np.arange(0, 11)         # observed counts tabulated
SHOWN_N = (0, 1, 2, 3)
LIMIT_B = 3.0                      # the known background of the second example
S_GRID = np.linspace(0.0, 9.0, 181)
COVER_S = np.arange(0.25, 10.01, 0.25)   # true signals whose coverage is measured
COVER_FINE = np.arange(0.02, 10.001, 0.02)
N_COVER = 20000                    # pseudo-experiments per true signal
EXAMPLE_S, N_EXAMPLE = 4.0, 40     # the pseudo-experiments drawn one by one
LUMI, EFFICIENCY = 20.0, 0.25      # fb^-1, and the share of signal events selected


def coverage_exact(s, limits_by_n, b=0.0):
    """Chance that the limit is at or above the true signal ``s``: the Poisson
    probabilities of every count whose limit covers it, added up (counts beyond the
    table all cover)."""
    s = np.atleast_1d(np.asarray(s, float))
    k = np.arange(len(limits_by_n))
    covers = np.asarray(limits_by_n)[None, :] >= s[:, None]
    return (poisson_pmf(k[None, :], s[:, None] + b) * covers).sum(1) + (
        1.0 - poisson_cdf(k[-1], s + b))


def _limits_build():
    limits = upper_limit(LIMIT_N, 0.0, CL)
    table = upper_limit(np.arange(0, 60), 0.0, CL)
    rng = np.random.default_rng(LIMIT_SEED)
    example = rng.poisson(EXAMPLE_S, N_EXAMPLE)
    counts = rng.poisson(COVER_S[:, None], (len(COVER_S), N_COVER))
    covered = (table[np.minimum(counts, len(table) - 1)] >= COVER_S[:, None]).mean(1)
    return dict(
        limits=limits, with_b=limits - LIMIT_B,
        cdf=np.array([poisson_cdf(n, S_GRID) for n in SHOWN_N]),
        example=example, example_limits=table[example],
        coverage=covered, coverage_exact=coverage_exact(COVER_S, table),
        coverage_fine=coverage_exact(COVER_FINE, table),
        cross_section=limits / (LUMI * EFFICIENCY),
    )


def limits_data():
    """Everything ``poisson-upper-limits`` measures and draws."""
    return _cache("limits", _limits_build, (_limits_build, coverage_exact),
                  seed=LIMIT_SEED, cl=CL, b=LIMIT_B, toys=N_COVER, lumi=(LUMI, EFFICIENCY),
                  example=(EXAMPLE_S, N_EXAMPLE))


def poisson_upper_limits():
    d = limits_data()
    return Study(
        "An upper limit from counting",
        "The largest Poisson mean that would still give n events or fewer one time in "
        "ten: 2.30 for n = 0; with a background; its coverage; as a cross-section",
        (
            Panel("The chance of n or fewer against the signal mean",
                  "ys[i]: P(N <= n | s) for n = %s as the mean s grows (exact Poisson "
                  "sums). limits: where each falls to %.2f: the %d%% upper limit."
                  % (", ".join(str(n) for n in SHOWN_N), 1 - CL, round(100 * CL)),
                  "curve", dict(x=S_GRID, ys=d["cdf"],
                                labels=["n = %d" % n for n in SHOWN_N],
                                limits=d["limits"][list(SHOWN_N)])),
            Panel("Upper limits by observed count",
                  "values: the %d%% upper limit on the signal mean for n = 0..%d observed "
                  "and no background (exact). with_background: the same minus a known "
                  "background of %g: negative for small n."
                  % (round(100 * CL), LIMIT_N[-1], LIMIT_B),
                  "bars", dict(values=d["limits"], labels=["%d" % n for n in LIMIT_N],
                               with_background=d["with_b"])),
            Panel("Coverage",
                  "ys[0]: share of %d pseudo-experiments (n drawn from Poisson(s), seed "
                  "%d) whose limit is at or above the true s, for each true s. ys[1]: the "
                  "exact chance. fine: the exact chance on a grid of 0.02."
                  % (N_COVER, LIMIT_SEED),
                  "curve", dict(x=COVER_S, ys=np.array([d["coverage"], d["coverage_exact"]]),
                                labels=["pseudo-experiments", "exact"],
                                fine_x=COVER_FINE, fine=d["coverage_fine"])),
            Panel("Limits of forty pseudo-experiments",
                  "xy: for %d pseudo-experiments with true s = %g, the count drawn and its "
                  "upper limit." % (N_EXAMPLE, EXAMPLE_S),
                  "points", dict(xy=np.stack([d["example"], d["example_limits"]], 1)
                                 .astype(float))),
        ),
        dict(
            seed=LIMIT_SEED, seeds=[LIMIT_SEED], review_only=True,
            source="exact Poisson arithmetic; pseudo-experiments for the coverage",
            evidence=dict(
                exact="every limit, every curve of P(N <= n | s), the exact coverage, the "
                      "cross-section line",
                pseudo_experiments="coverage: %d counts drawn from Poisson(s) for each of "
                                   "%d true signals" % (N_COVER, len(COVER_S))),
            confidence_level=CL,
            limits={str(int(n)): float(v) for n, v in zip(LIMIT_N, d["limits"])},
            background=LIMIT_B,
            limits_with_background={str(int(n)): float(v)
                                    for n, v in zip(LIMIT_N, d["with_b"])},
            coverage=dict(lowest_exact=float(d["coverage_fine"].min()),
                          highest_exact=float(d["coverage_fine"].max()),
                          lowest_pseudo=float(d["coverage"].min()),
                          mean_pseudo=float(d["coverage"].mean()),
                          largest_gap=float(np.abs(d["coverage"] - d["coverage_exact"]).max())),
            example=dict(true_signal=EXAMPLE_S, counts=[int(v) for v in d["example"]],
                         covered=int((d["example_limits"] >= EXAMPLE_S).sum())),
            cross_section=dict(luminosity_fb=LUMI, efficiency=EFFICIENCY,
                               limit_fb=[float(v) for v in d["cross_section"][:4]]),
            limitations=[
                "The background is taken as known exactly, and so are luminosity and "
                "efficiency: their uncertainties would weaken the limit.",
                "A limit that is the plain Poisson mean minus the background goes "
                "negative when fewer events are seen than the background predicts: the "
                "subject of the CLs animation.",
                "Coverage is the property of the procedure over many repeats; it says "
                "nothing about any one limit.",
            ],
            designs_tried=[
                "The thirty limits drawn one by one were first thrown at a true s = 3: all "
                "thirty covered it (a miss needs n = 0, a 5% chance each), which showed "
                "nothing of what a miss looks like.",
                "Kept: forty at a true s = 4, where a miss needs n <= 1 (9.2% each), same "
                "seed. Nothing else was changed; the coverage curve never depended on it.",
            ],
        ),
    )


# ====================================================================== 147: CLs
CLS_SEED = 147
CLS_CL = 0.95
CLS_B = 10.0                       # known background
CLS_N = 4                          # the count of the worked example: a downward fluctuation
CLS_TEST = 3.0                     # the signal tested in the worked example
CLS_COUNTS = np.arange(0, 26)
CLS_SMALL = (1.0, 2.0)             # signals an experiment with b = 10 can hardly see
CLS_S = np.arange(0.5, 15.01, 0.5)
CLS_FINE = np.arange(0.05, 15.001, 0.05)
N_CLS = 100000                     # background-only pseudo-experiments
N_CLS_COVER = 20000                # pseudo-experiments per true signal


def cls_values(n, s, b):
    """``(CLs+b, CLb, CLs)`` of ``n`` observed events for a signal ``s`` on background
    ``b``: the chances of ``n`` or fewer with and without the signal, and their ratio."""
    clsb, clb = poisson_cdf(n, np.asarray(s, float) + b), poisson_cdf(n, b)
    return clsb, clb, clsb / clb


def _cls_build():
    top = 80
    counts = np.arange(top)
    plain = upper_limit(counts, CLS_B, CLS_CL, "plain")
    cls = upper_limit(counts, CLS_B, CLS_CL, "cls")
    rng = np.random.default_rng(CLS_SEED)
    n0 = np.minimum(rng.poisson(CLS_B, N_CLS), top - 1)
    pmf0 = poisson_pmf(counts, CLS_B)
    at = np.array([0.0, *CLS_SMALL])
    below = np.array([[(lim[n0] < s).mean() for s in at] for lim in (plain, cls)])
    below_exact = np.array([[pmf0[lim < s].sum() for s in at] for lim in (plain, cls)])
    toys = np.minimum(rng.poisson(CLS_S[:, None] + CLS_B, (len(CLS_S), N_CLS_COVER)), top - 1)
    cover = np.array([(lim[toys] >= CLS_S[:, None]).mean(1) for lim in (plain, cls)])
    return dict(
        plain=plain[:len(CLS_COUNTS)], cls=cls[:len(CLS_COUNTS)],
        counts_hist=np.bincount(n0, minlength=len(CLS_COUNTS))[:len(CLS_COUNTS)],
        below=below, below_exact=below_exact, below_at=at,
        median=np.array([np.median(plain[n0]), np.median(cls[n0])]),
        coverage=cover,
        coverage_exact=np.array([coverage_exact(CLS_S, lim, CLS_B) for lim in (plain, cls)]),
        coverage_fine=np.array([coverage_exact(CLS_FINE, lim, CLS_B) for lim in (plain, cls)]),
    )


def cls_data():
    """Everything ``cls-method`` measures and draws."""
    return _cache("cls", _cls_build, (_cls_build, cls_values, coverage_exact),
                  seed=CLS_SEED, cl=CLS_CL, b=CLS_B, toys=(N_CLS, N_CLS_COVER),
                  small=CLS_SMALL)


def cls_method():
    d = cls_data()
    clsb, clb, cls = (float(v) for v in cls_values(CLS_N, CLS_TEST, CLS_B))
    return Study(
        "The CLs method",
        "A counting experiment with a known background of %g: the plain limit against "
        "the CLs limit, count by count and over pseudo-experiments" % CLS_B,
        (
            Panel("The two limits by observed count",
                  "ys[0]: the plain %d%% upper limit on the signal for n observed, the s "
                  "with P(N <= n | s + b) = %.2f, b = %g (exact; negative for n <= %d). "
                  "ys[1]: the CLs limit, the s with P(N <= n | s + b) / P(N <= n | b) = %.2f."
                  % (round(100 * CLS_CL), 1 - CLS_CL, CLS_B,
                     int(np.flatnonzero(d["plain"] < 0).max()), 1 - CLS_CL),
                  "curve", dict(x=CLS_COUNTS, ys=np.array([d["plain"], d["cls"]]),
                                labels=["plain (CLs+b)", "CLs"])),
            Panel("How often a background-only experiment excludes a small signal",
                  "values: share of %d background-only pseudo-experiments (n from "
                  "Poisson(%g), seed %d) whose limit is below s = 0, %g and %g: first the "
                  "plain limit, then CLs. exact: the same shares from Poisson sums."
                  % (N_CLS, CLS_B, CLS_SEED, *CLS_SMALL),
                  "bars", dict(values=d["below"].ravel(),
                               labels=["%s, s = %g" % (m, s) for m in ("plain", "CLs")
                                       for s in d["below_at"]],
                               exact=d["below_exact"].ravel(), counts=d["counts_hist"])),
            Panel("Coverage",
                  "ys[0], ys[1]: share of %d pseudo-experiments with a true signal s (n "
                  "from Poisson(s + %g)) whose plain and CLs limits are at or above s. "
                  "exact: the same from Poisson sums; fine: on a grid of 0.05."
                  % (N_CLS_COVER, CLS_B),
                  "curve", dict(x=CLS_S, ys=d["coverage"], labels=["plain (CLs+b)", "CLs"],
                                exact=d["coverage_exact"], fine_x=CLS_FINE,
                                fine=d["coverage_fine"])),
            Panel("The worked example",
                  "values: for n = %d observed, b = %g and a tested signal s = %g: CLs+b "
                  "= P(N <= n | s + b), CLb = P(N <= n | b), and CLs, their ratio (exact)."
                  % (CLS_N, CLS_B, CLS_TEST),
                  "bars", dict(values=np.array([clsb, clb, cls]),
                               labels=["CLs+b", "CLb", "CLs"])),
        ),
        dict(
            seed=CLS_SEED, seeds=[CLS_SEED], review_only=True,
            source="exact Poisson arithmetic; pseudo-experiments for the shares and coverage",
            evidence=dict(
                exact="CLs+b, CLb and CLs of the worked example; both limits for every "
                      "count; the exact shares and coverage",
                pseudo_experiments="%d background-only counts; %d counts for each of %d "
                                   "true signals" % (N_CLS, N_CLS_COVER, len(CLS_S))),
            confidence_level=CLS_CL, background=CLS_B,
            example=dict(observed=CLS_N, tested_signal=CLS_TEST, cls_b=clsb, cl_b=clb, cls=cls,
                         plain_limit=float(d["plain"][CLS_N]), cls_limit=float(d["cls"][CLS_N])),
            excludes=dict(signals=[float(v) for v in d["below_at"]],
                          plain=[float(v) for v in d["below"][0]],
                          cls=[float(v) for v in d["below"][1]],
                          plain_exact=[float(v) for v in d["below_exact"][0]],
                          cls_exact=[float(v) for v in d["below_exact"][1]]),
            median_limit=dict(plain=float(d["median"][0]), cls=float(d["median"][1])),
            coverage=dict(plain_lowest=float(d["coverage_fine"][0].min()),
                          cls_lowest=float(d["coverage_fine"][1].min()),
                          plain_at_smallest=float(d["coverage"][0][0]),
                          cls_at_smallest=float(d["coverage"][1][0]),
                          smallest_signal=float(CLS_S[0])),
            limitations=[
                "A counting experiment with an exactly known background: the count itself "
                "is the test statistic. With shapes and uncertain backgrounds the test "
                "statistic is a likelihood ratio, and the two tail areas are of its "
                "distributions.",
                "CLs is not a probability and its limit is not a confidence interval of "
                "exact coverage: it over-covers on purpose where the experiment cannot "
                "tell signal from none.",
                "The worked count n = %d is chosen, not drawn: a background of %g gives "
                "%d or fewer %.1f%% of the time." % (CLS_N, CLS_B, CLS_N, 100 * clb),
            ],
            designs_tried=["Only this design was run."],
        ),
    )


# ====================================================================== 148: the bands
BAND_SEEDS = (148, 1480, 1481)     # expected limits; the "observed" counts; chance exits
BAND_M = np.arange(110.0, 165.1, 5.0)
BAND_CL = 0.95
BAND_AT = 2                        # the mass point followed one experiment at a time (120)
BAND_INJECT, BAND_MU = 6, 1.0      # the mass point (140) holding a signal, and its strength
N_BAND = 5000                      # background-only pseudo-experiments per mass point
N_EXITS = 20000                    # background-only "observed" plots
N_LINES = 8                        # of them, recorded to be drawn
QUANTILES = (2.5, 16.0, 50.0, 84.0, 97.5)


def band_model():
    """Expected background count, and the count a signal of strength mu = 1 would add,
    in the window of each mass point."""
    return 60.0 * np.exp(-(BAND_M - 110.0) / 30.0), 20.0 * np.exp(-(BAND_M - 110.0) / 60.0)


def _band_build():
    back, ref = band_model()
    top = 160
    counts = np.arange(top)
    # the limit depends on the data only through the count: solve it once per count
    table = np.array([upper_limit(counts, b, BAND_CL, "cls") / r for b, r in zip(back, ref)])
    rows = np.arange(len(BAND_M))[:, None]
    toys = np.minimum(np.random.default_rng(BAND_SEEDS[0]).poisson(
        back[:, None], (len(back), N_BAND)), top - 1)
    limits = table[rows, toys]
    # a limit takes one value per count: the quantile is the smallest of those values
    # with at least that share of the experiments at or below it
    bands = np.percentile(limits, QUANTILES, axis=1, method="inverted_cdf")
    pmf = poisson_pmf(counts[None, :], back[:, None])
    cdf = np.cumsum(pmf, 1)
    exact = np.array([[table[i, int(np.searchsorted(cdf[i], q / 100.0))]
                       for i in range(len(back))] for q in QUANTILES])
    expect = back.copy()
    expect[BAND_INJECT] += BAND_MU * ref[BAND_INJECT]
    observed = np.random.default_rng(BAND_SEEDS[1]).poisson(expect)
    plots = np.minimum(np.random.default_rng(BAND_SEEDS[2]).poisson(
        back[:, None], (len(back), N_EXITS)), top - 1)
    plot_limits = table[rows, plots]
    out = (plot_limits < bands[0][:, None]) | (plot_limits > bands[4][:, None])
    out_table = (table < bands[0][:, None]) | (table > bands[4][:, None])
    out_exact = (pmf * out_table).sum(1)
    out_green = (plot_limits < bands[1][:, None]) | (plot_limits > bands[3][:, None])
    return dict(
        back=back, ref=ref, table=table, bands=bands, bands_exact=exact,
        toys_at=toys[BAND_AT, :400], counts_at=np.bincount(toys[BAND_AT], minlength=top),
        observed=observed, observed_limit=table[np.arange(len(back)), observed],
        lines=plot_limits[:, :N_LINES].T, lines_out=out[:, :N_LINES].T,
        out_point=out.mean(1), out_point_exact=out_exact,
        out_any=np.array(out.any(0).mean()),
        out_any_exact=np.array(1.0 - np.prod(1.0 - out_exact)),
        out_count=np.bincount(out.sum(0), minlength=len(back) + 1),
        out_green=np.array(out_green.mean()),
    )


def band_data():
    """Everything ``brazil-band`` measures and draws."""
    return _cache("band", _band_build, (_band_build, band_model),
                  seeds=BAND_SEEDS, cl=BAND_CL, toys=(N_BAND, N_EXITS),
                  masses=[float(v) for v in BAND_M], inject=(BAND_INJECT, BAND_MU))


def brazil_band():
    d = band_data()
    outside = np.flatnonzero((d["observed_limit"] < d["bands"][0])
                             | (d["observed_limit"] > d["bands"][4]))
    return Study(
        "The expected-limit bands",
        "Twelve mass points: the spread of the upper limit over background-only "
        "pseudo-experiments, and an observed limit with a signal injected at one mass",
        (
            Panel("Expected limit: median and bands",
                  "ys: the 2.5%%, 16%%, 50%%, 84%% and 97.5%% quantiles of the %d%% CLs "
                  "upper limit on the signal strength mu over %d background-only "
                  "pseudo-experiments at each mass point (seed %d). exact: the same "
                  "quantiles from the Poisson distribution of the count."
                  % (round(100 * BAND_CL), N_BAND, BAND_SEEDS[0]),
                  "curve", dict(x=BAND_M, ys=d["bands"],
                                labels=["%g%%" % q for q in QUANTILES],
                                exact=d["bands_exact"], background=d["back"],
                                signal=d["ref"])),
            Panel("Observed limit",
                  "ys[0]: the limit from one pseudo-dataset (seed %d) with a signal of "
                  "strength mu = %g injected at m = %g and nothing elsewhere. ys[1]: the "
                  "median expected limit. counts: the observed counts."
                  % (BAND_SEEDS[1], BAND_MU, BAND_M[BAND_INJECT]),
                  "curve", dict(x=BAND_M, ys=np.array([d["observed_limit"], d["bands"][2]]),
                                labels=["observed", "median expected"],
                                counts=d["observed"])),
            Panel("Limits at one mass point",
                  "values[n]: how many of the %d background-only pseudo-experiments at m = "
                  "%g counted n events; limits[n]: the upper limit on mu that count gives."
                  % (N_BAND, BAND_M[BAND_AT]),
                  "bars", dict(values=d["counts_at"].astype(float),
                               labels=["%d" % n for n in range(len(d["counts_at"]))],
                               limits=d["table"][BAND_AT], first=d["toys_at"])),
            Panel("Leaving the 95% band by chance",
                  "values: the share of %d background-only plots (seed %d) with 0, 1, 2, "
                  "... of the %d observed points outside the 95%% band. per_point: the "
                  "share of plots in which each mass point is outside."
                  % (N_EXITS, BAND_SEEDS[2], len(BAND_M)),
                  "bars", dict(values=d["out_count"] / N_EXITS,
                               labels=["%d" % k for k in range(len(BAND_M) + 1)],
                               per_point=d["out_point"],
                               per_point_exact=d["out_point_exact"])),
        ),
        dict(
            seed=BAND_SEEDS[0], seeds=list(BAND_SEEDS), review_only=True,
            source="pseudo-experiments on a stated counting model; exact Poisson quantiles",
            evidence=dict(
                pseudo_experiments="%d background-only counts per mass point for the "
                                   "bands; one pseudo-dataset as the observation; %d "
                                   "background-only plots" % (N_BAND, N_EXITS),
                exact="each limit from its count (CLs); the bands' quantiles and the "
                      "chance of leaving them from Poisson sums"),
            confidence_level=BAND_CL,
            model=dict(masses=[float(v) for v in BAND_M],
                       background=[float(v) for v in d["back"]],
                       signal_at_mu_1=[float(v) for v in d["ref"]]),
            followed=dict(mass=float(BAND_M[BAND_AT]), background=float(d["back"][BAND_AT]),
                          signal=float(d["ref"][BAND_AT])),
            injected=dict(mass=float(BAND_M[BAND_INJECT]), mu=BAND_MU,
                          events=float(BAND_MU * d["ref"][BAND_INJECT])),
            bands={("%g" % q): [float(v) for v in row] for q, row in zip(QUANTILES, d["bands"])},
            largest_gap_to_exact=float(np.abs(d["bands"] - d["bands_exact"]).max()),
            band_edges_equal_to_exact=int((d["bands"] == d["bands_exact"]).sum()),
            band_edges=int(d["bands"].size),
            observed=dict(counts=[int(v) for v in d["observed"]],
                          limits=[float(v) for v in d["observed_limit"]],
                          outside_95=[float(BAND_M[i]) for i in outside]),
            leaving=dict(plots=N_EXITS, per_point=float(d["out_point"].mean()),
                         per_point_exact=float(d["out_point_exact"].mean()),
                         any_point=float(d["out_any"]), any_point_exact=float(d["out_any_exact"]),
                         outside_68_per_point=float(d["out_green"]),
                         nominal_any=float(1 - 0.95 ** len(BAND_M))),
            limitations=[
                "Each mass point is its own counting experiment with an exactly known "
                "background, so neighbouring points are independent; in a real search "
                "they share events and the observed line wanders more slowly.",
                "Counts are whole numbers, so a limit takes one of a few values and the "
                "bands hold a little more than 68% and 95%: the observed line leaves "
                "them less often than the nominal 5% per point.",
                "One observed pseudo-dataset, seed %d, chosen before it was looked at."
                % BAND_SEEDS[1],
            ],
            designs_tried=["Only this design was run."],
        ),
    )


# ====================================================================== 149: five sigma
FIVE_SEEDS = (149, 1490, 1491)     # the two spectra followed; many searches; with signal
N_BINS, SIG_BIN = 20, 7            # bins of the spectrum (the places to look); the signal's
B_RATE = 25.0                      # background events per bin per unit of luminosity
S_RATE = 5.0                       # signal events (in its bin) per unit of luminosity
LOOKS = np.arange(1.0, 30.5, 1.0)  # luminosities at which the analysis is run
N_SEARCH = 100000                  # background-only searches followed
GROUP = 10                         # searches run side by side by one experiment
S_RATES = (2.5, 5.0, 10.0)
N_SIGNAL = 5000                    # pseudo-experiments per signal size
LOOKS_LONG = np.arange(1.0, 130.5, 1.0)
LEVELS = (3.0, 5.0)
ANY_EDGES = np.arange(1.0, 6.01, 0.25)


def lumi_for(z, s_rate, b_rate=B_RATE):
    """Luminosity at which the expected significance of a signal reaches ``z``: both
    counts grow with luminosity, so ``Z`` of the expected counts grows as its square
    root and ``L = (z / Z(L = 1))^2``."""
    return float((z / z_asimov(s_rate, b_rate)) ** 2)


def _five_build():
    step = np.diff(np.concatenate([[0.0], LOOKS]))
    rng = np.random.default_rng(FIVE_SEEDS[0])
    # ---- the two spectra followed: the same analysis, with and without a signal
    none = np.cumsum(rng.poisson(B_RATE * step[:, None], (len(LOOKS), N_BINS)), 0)
    with_ = np.cumsum(rng.poisson(B_RATE * step[:, None], (len(LOOKS), N_BINS)), 0)
    with_[:, SIG_BIN] += np.cumsum(rng.poisson(S_RATE * step))
    expect = B_RATE * LOOKS[:, None]
    # ---- many background-only searches: the largest Z of each bin over all looks
    rng = np.random.default_rng(FIVE_SEEDS[1])
    one_end, one_ever, any_ever = [], [], []
    for _ in range(N_SEARCH // 5000):
        n = np.cumsum(rng.poisson(B_RATE * step[None, :, None], (5000, len(LOOKS), N_BINS)), 1)
        z = z_counting(n, expect[None])
        one_end.append(z[:, -1, 0])
        one_ever.append(z[:, :, 0].max(1))
        any_ever.append(z.max((1, 2)))
    one_end, one_ever, any_ever = (np.concatenate(v) for v in (one_end, one_ever, any_ever))
    group = any_ever.reshape(-1, GROUP).max(1)
    reach = np.array([[(v >= z).sum() for v in (one_end, one_ever, any_ever, group)]
                      for z in LEVELS])
    # ---- with a signal: the median Z as luminosity grows, for three signal sizes
    rng = np.random.default_rng(FIVE_SEEDS[2])
    step = np.diff(np.concatenate([[0.0], LOOKS_LONG]))
    median, lumi5 = [], []
    for s in S_RATES:
        n = np.cumsum(rng.poisson((B_RATE + s) * step[None, :], (N_SIGNAL, len(LOOKS_LONG))), 1)
        med = np.median(z_counting(n, B_RATE * LOOKS_LONG[None, :]), 0)
        k = int(np.argmax(med >= 5.0)) if (med >= 5.0).any() else len(med) - 1
        median.append(med)
        lumi5.append(LOOKS_LONG[k])
    return dict(
        none=none, with_=with_, z_none=z_counting(none, expect), z_with=z_counting(with_, expect),
        reach=reach, reach_of=np.array([N_SEARCH, N_SEARCH, N_SEARCH, N_SEARCH // GROUP]),
        ever2=np.array((one_ever >= 2.0).mean()),
        any_hist=np.histogram(any_ever, ANY_EDGES)[0],
        median=np.array(median), lumi5=np.array(lumi5),
        lumi5_formula=np.array([lumi_for(5.0, s) for s in S_RATES]),
        tails=norm_sf(np.array(LEVELS)),
    )


def five_data():
    """Everything ``why-five-sigma`` measures and draws."""
    return _cache("five", _five_build, (_five_build, z_counting, lumi_for),
                  seeds=FIVE_SEEDS, bins=(N_BINS, SIG_BIN), rates=(B_RATE, S_RATE, S_RATES),
                  looks=(len(LOOKS), len(LOOKS_LONG)), toys=(N_SEARCH, N_SIGNAL, GROUP))


FIVE_LEVELS = ("one bin, last look", "one bin, every look", "%d bins, every look" % N_BINS,
               "%d such searches" % GROUP)


def why_five_sigma():
    d = five_data()
    share = d["reach"] / d["reach_of"][None, :]
    names = list(FIVE_LEVELS)
    k_sig = [int(np.argmin(np.abs(LOOKS - v))) for v in (5.0, 10.0, 20.0)]
    return Study(
        "Why five sigma",
        "A twenty-bin search followed as luminosity accumulates, with a signal and "
        "without: how often background alone reaches three sigma and five",
        (
            Panel("Significance as the data accumulate",
                  "ys[0]: Z of the signal's bin in a pseudo-experiment with a signal of "
                  "%g events per unit of luminosity on a background of %g per bin (seed "
                  "%d). ys[1]: the expected Z, which grows as the square root of the "
                  "luminosity. ys[2]: the largest Z over the %d bins of a background-only "
                  "pseudo-experiment. all_bins: every bin's Z in that experiment."
                  % (S_RATE, B_RATE, FIVE_SEEDS[0], N_BINS),
                  "curve", dict(x=LOOKS, ys=np.array([d["z_with"][:, SIG_BIN],
                                                      z_asimov(S_RATE * LOOKS, B_RATE * LOOKS),
                                                      d["z_none"].max(1)]),
                                labels=["with a signal", "expected", "background only"],
                                all_bins=d["z_none"].T)),
            Panel("How often background alone reaches 3 sigma",
                  "values: the share of %d background-only searches (seed %d) reaching Z "
                  ">= 3: in one bin at the last look; in one bin at any of the %d looks; "
                  "in any of the %d bins at any look; and the share of %d groups of %d "
                  "searches in which any does. five: the same for Z >= 5, as counts."
                  % (N_SEARCH, FIVE_SEEDS[1], len(LOOKS), N_BINS, N_SEARCH // GROUP, GROUP),
                  "bars", dict(values=share[0], labels=names, five=d["reach"][1],
                               three=d["reach"][0], of=d["reach_of"])),
            Panel("One-sided tail probabilities",
                  "values: the chance that a Gaussian fluctuation exceeds 3 and 5 standard "
                  "deviations (exact).",
                  "bars", dict(values=d["tails"], labels=["3 sigma", "5 sigma"])),
            Panel("Median significance for three signal sizes",
                  "ys[i]: the median Z over %d pseudo-experiments (seed %d) against "
                  "luminosity for a signal of %s events per unit of luminosity. lumi5: "
                  "the first luminosity at which it reaches 5; formula: the luminosity "
                  "at which the expected Z does."
                  % (N_SIGNAL, FIVE_SEEDS[2], ", ".join("%g" % s for s in S_RATES)),
                  "curve", dict(x=LOOKS_LONG, ys=d["median"],
                                labels=["s = %g" % s for s in S_RATES],
                                lumi5=d["lumi5"], formula=d["lumi5_formula"])),
        ),
        dict(
            seed=FIVE_SEEDS[0], seeds=list(FIVE_SEEDS), review_only=True,
            source="pseudo-experiments on a stated counting model; exact Gaussian tails",
            evidence=dict(
                pseudo_experiments="two spectra followed (seed %d); %d background-only "
                                   "searches of %d bins at %d looks; %d experiments for "
                                   "each of %d signal sizes"
                                   % (FIVE_SEEDS[0], N_SEARCH, N_BINS, len(LOOKS), N_SIGNAL,
                                      len(S_RATES)),
                exact="the tail probabilities of 3 and 5 sigma; the expected Z and the "
                      "luminosity at which it reaches 5"),
            model=dict(bins=N_BINS, signal_bin=SIG_BIN, background_rate=B_RATE,
                       signal_rate=S_RATE, looks=len(LOOKS), last_look=float(LOOKS[-1])),
            tails={"3": float(d["tails"][0]), "5": float(d["tails"][1])},
            followed=dict(
                with_signal_z={("%g" % LOOKS[k]): float(d["z_with"][k, SIG_BIN]) for k in k_sig},
                expected_z={("%g" % LOOKS[k]): float(z_asimov(S_RATE * LOOKS[k],
                                                               B_RATE * LOOKS[k]))
                            for k in k_sig},
                background_peak=float(d["z_none"].max()),
                background_peak_at=float(LOOKS[int(np.argmax(d["z_none"].max(1)))]),
                background_last=float(d["z_none"][-1].max())),
            reach=dict(levels=names, searches=[int(v) for v in d["reach_of"]],
                       three=[int(v) for v in d["reach"][0]],
                       five=[int(v) for v in d["reach"][1]],
                       three_share=[float(v) for v in share[0]],
                       five_share=[float(v) for v in share[1]],
                       one_bin_ever_two=float(d["ever2"])),
            lumi_for_five=dict(signal_rates=list(S_RATES),
                               pseudo_experiments=[float(v) for v in d["lumi5"]],
                               formula=[float(v) for v in d["lumi5_formula"]]),
            limitations=[
                "Each bin is a counting experiment with an exactly known background, and "
                "the bins are independent. Counts are whole numbers, so the share of "
                "single looks at or above 3 sigma is close to the Gaussian 1.35e-3 but "
                "not equal to it.",
                "Five-sigma fluctuations are so rare that %d searches show only a "
                "handful or none: that count is reported as it came out, beside the "
                "exact single-look probability." % N_SEARCH,
                "Chance is only one reason for the five-sigma convention: unknown "
                "systematic errors and the implausibility of a new particle are others, "
                "and no pseudo-experiment here measures them.",
            ],
            designs_tried=["Only this design was run."],
        ),
    )


STUDIES = {
    "bump-hunt": bump_hunt,
    "poisson-upper-limits": poisson_upper_limits,
    "cls-method": cls_method,
    "brazil-band": brazil_band,
    "why-five-sigma": why_five_sigma,
}
