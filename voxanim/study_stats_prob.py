"""The experiments behind animations 118-123: probability read the frequentist way and
the Bayesian way (see :func:`voxanim.next_catalog.redesigned`).

* ``frequentist-and-bayesian``: one counting experiment, n = 3 events. The confidence
  belt (for each true mean, the counts that are likely) read across at n; the posterior
  of the mean under a flat prior; both intervals for n = 0..20; how often each contains
  a known truth; and a count below a known background.
* ``bayes-theorem-areas``: a rare-disease test and a kaon tag as areas of a unit square,
  the posterior as one rectangle over two, and the purity against the prior.
* ``prior-washing-out``: one seeded run of coin flips updating five priors.
* ``interval-coverage``: how often an interval contains the truth, for x +- sigma,
  n +- sqrt(n), the exact Poisson interval, and two binomial intervals.
* ``binomial-poisson-gaussian``: total-variation distances between the three
  distributions, and the tails.
* ``central-limit-theorem``: the mean of N dice, of N decay times and of N
  Breit-Wigner draws.

Nothing here uses data. Every number is one of two kinds, and each drawing says which:

* **exact**: a formula or a distribution evaluated (sums of Poisson and binomial terms,
  convolutions, a posterior on a grid of the parameter);
* **pseudo-experiments**: seeded draws (``numpy.random.default_rng(seed)``) from a model
  whose true values are stated, repeated thousands of times.

Wherever both exist they are recorded side by side. The special functions are the few
needed and are written here (no scipy): sums of exactly computed terms, and bisection.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .study import Panel, Study

CL = math.erf(1.0 / math.sqrt(2.0))          # 0.6827: the content of +-1 sigma of a Gaussian
TAIL = (1.0 - CL) / 2.0                       # 0.1587: what a central interval leaves each side
_LOG_FACT = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, 20001)))])
_erf = np.vectorize(math.erf, otypes=[float])


# ====================================================================== special functions
def poisson_pmf(mu, kmax):
    """P(n | mu) for n = 0..``kmax``: mu^n e^-mu / n!."""
    k = np.arange(int(kmax) + 1)
    if mu <= 0:
        return (k == 0).astype(float)
    return np.exp(k * math.log(mu) - mu - _LOG_FACT[k])


def poisson_cdf(n, mu):
    """P(count <= n | mu)."""
    return 0.0 if n < 0 else float(min(poisson_pmf(mu, n).sum(), 1.0))


def binomial_pmf(n, p):
    """P(k | n, p) for k = 0..n: C(n, k) p^k (1 - p)^(n - k)."""
    k = np.arange(int(n) + 1)
    if p <= 0 or p >= 1:
        return (k == (0 if p <= 0 else n)).astype(float)
    return np.exp(_LOG_FACT[n] - _LOG_FACT[k] - _LOG_FACT[n - k] + k * math.log(p)
                  + (n - k) * math.log1p(-p))


def normal_cdf(z):
    """The standard Gaussian's cumulative distribution."""
    return 0.5 * (1.0 + _erf(np.asarray(z, float) / math.sqrt(2.0)))


def normal_pdf(x, mean=0.0, sd=1.0):
    z = (np.asarray(x, float) - mean) / sd
    return np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))


def solve(f, lo, hi, steps=70):
    """The root of ``f`` between ``lo`` and ``hi`` (where its signs differ), by bisection."""
    flo = f(lo)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tv(p, q):
    """Total-variation distance between two distributions on the same outcomes: half the
    summed absolute difference (0: the same; 1: nothing in common)."""
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


# ====================================================================== intervals
def poisson_central(n, cl=CL):
    """The exact central confidence interval for a Poisson mean after counting ``n``
    (Garwood's): the lower end is the mean at which counting n or more has probability
    (1 - cl) / 2, the upper end the mean at which counting n or fewer has."""
    a = (1.0 - cl) / 2.0
    top = n + 12.0 * math.sqrt(n + 1.0) + 12.0
    lo = 0.0 if n == 0 else solve(lambda m: 1.0 - poisson_cdf(n - 1, m) - a, 0.0, top)
    return lo, solve(lambda m: poisson_cdf(n, m) - a, 0.0, top)


def poisson_sqrt(n):
    """The textbook error bar: n - sqrt(n) to n + sqrt(n)."""
    return n - math.sqrt(n), n + math.sqrt(n)


def poisson_band(mu, cl=CL):
    """For a true mean ``mu``: the central range of counts ``(n_lo, n_hi)``, leaving at
    most (1 - cl) / 2 of probability below it and at most as much above."""
    a = (1.0 - cl) / 2.0
    pmf = poisson_pmf(mu, int(mu + 12.0 * math.sqrt(mu + 1.0) + 12.0))
    below = np.cumsum(pmf)                    # P(count <= n)
    above = 1.0 - below + pmf                 # P(count >= n)
    return int(np.flatnonzero(below > a)[0]), int(np.flatnonzero(above > a)[-1])


def poisson_credible(n, cl=CL, b=0.0, prior_power=0):
    """The central credible interval for a signal ``s >= 0`` on a known background ``b``
    after counting ``n``, with the prior ``1 / (s + b)^prior_power`` (0: flat). The
    posterior of mu = s + b is then a Gamma distribution cut off below b; its cumulative
    distribution is a Poisson sum (exact)."""
    shape = n + 1 - prior_power               # P(Gamma(shape) <= m) = P(count >= shape | m)

    def below(m):
        return 1.0 - poisson_cdf(shape - 1, m)
    start = below(b)
    top = n + b + 12.0 * math.sqrt(n + 1.0) + 12.0

    def quantile(q):
        return solve(lambda s: (below(s + b) - start) / (1.0 - start) - q, 0.0, top)
    a = (1.0 - cl) / 2.0
    return quantile(a), quantile(1.0 - a)


def clopper_pearson(k, n, cl=CL):
    """The exact central interval for a binomial probability after ``k`` of ``n``."""
    a = (1.0 - cl) / 2.0
    lo = 0.0 if k == 0 else solve(lambda p: binomial_pmf(n, p)[k:].sum() - a, 0.0, 1.0)
    hi = 1.0 if k == n else solve(lambda p: binomial_pmf(n, p)[:k + 1].sum() - a, 0.0, 1.0)
    return lo, hi


def wald(k, n):
    """The textbook binomial error bar: p +- sqrt(p (1 - p) / n) at p = k / n."""
    p = k / n
    half = math.sqrt(p * (1.0 - p) / n)
    return p - half, p + half


def coverage(pmf, lo, hi, truth):
    """The probability that an interval contains ``truth``: the summed probability of
    every outcome ``k`` whose interval ``lo[k]..hi[k]`` does."""
    k = len(pmf)
    inside = (np.asarray(lo)[:k] <= truth + 1e-12) & (truth <= np.asarray(hi)[:k] + 1e-12)
    return float(pmf[inside].sum())


def _evidence(exact, pseudo):
    return dict(exact=exact, pseudo_experiments=pseudo)


SEED_NOTE = ("numpy.random.default_rng(seed): every pseudo-experiment is drawn from the "
             "stated model with the recorded seed")


# ====================================================================== 118: two readings
FB_SEED = 118
N_OBS = 3                          # the count of the worked example
MU_SHOWN = (1.5, 3.0, 6.0)         # true means whose distributions are drawn first
FB_KMAX = 12
BELT_MUS = np.round(np.arange(0.25, 10.01, 0.25), 2)
FB_NS = np.arange(0, 21)
MU_TRUE = 3.0
FB_REPEATS = 20000
BACKGROUND = 3.0
BOUNDARY_NS = (0, 1, 2, 3, 5, 8)


@functools.lru_cache(maxsize=1)
def fb_data():
    """Everything ``frequentist-and-bayesian`` measures and draws."""
    belt = np.array([poisson_band(float(m)) for m in BELT_MUS])
    freq = np.array([poisson_central(int(n)) for n in FB_NS])
    bayes = np.array([poisson_credible(int(n)) for n in FB_NS])
    naive = np.array([poisson_sqrt(int(n)) for n in FB_NS])
    # the prior 1/mu, for comparison: its posterior is Gamma(n), improper at n = 0
    inverse = np.array([poisson_credible(int(n), prior_power=1) for n in FB_NS[1:]])
    grid = np.linspace(0.0, 12.0, 481)
    like = np.array([poisson_pmf(float(m), N_OBS)[N_OBS] for m in grid])
    # how often each interval contains the truth: exactly, and in pseudo-experiments
    kmax = 80
    lo = np.array([[poisson_central(n)[0], poisson_credible(n)[0], poisson_sqrt(n)[0]]
                   for n in range(kmax + 1)])
    hi = np.array([[poisson_central(n)[1], poisson_credible(n)[1], poisson_sqrt(n)[1]]
                   for n in range(kmax + 1)])
    mus = np.linspace(0.1, 15.0, 299)
    cover = np.array([[coverage(poisson_pmf(float(m), kmax), lo[:, j], hi[:, j], float(m))
                       for m in mus] for j in range(3)])
    exact_at = [coverage(poisson_pmf(MU_TRUE, kmax), lo[:, j], hi[:, j], MU_TRUE)
                for j in range(3)]
    counts = np.random.default_rng(FB_SEED).poisson(MU_TRUE, FB_REPEATS)
    toy_at = [float(((lo[counts, j] <= MU_TRUE) & (hi[counts, j] >= MU_TRUE)).mean())
              for j in range(3)]
    b_freq = np.array([poisson_central(n) for n in BOUNDARY_NS]) - BACKGROUND
    b_bayes = np.array([poisson_credible(n, b=BACKGROUND) for n in BOUNDARY_NS])
    return dict(
        pmfs=np.array([poisson_pmf(m, FB_KMAX) for m in MU_SHOWN]), belt=belt, freq=freq,
        bayes=bayes, naive=naive, inverse=inverse, grid=grid, like=like, cover_mus=mus,
        cover=cover, exact_at=np.array(exact_at), toy_at=np.array(toy_at),
        first_counts=counts[:12], b_freq=b_freq, b_bayes=b_bayes)


def frequentist_and_bayesian():
    d = fb_data()
    f, b = d["freq"][N_OBS], d["bayes"][N_OBS]
    return Study(
        "Two readings of one count",
        "n = %d events counted, mean unknown: the confidence interval read off the belt of "
        "likely counts, and the credible interval of the posterior" % N_OBS,
        (
            Panel("The confidence belt",
                  "For each true mean x, ys[0] to ys[1] is the central range of counts: at "
                  "most %.2f%% of probability below it, at most as much above. The interval "
                  "for a count n is every mean whose range includes n. Exact Poisson sums."
                  % (100 * TAIL), "curve",
                  dict(x=BELT_MUS, ys=d["belt"].T.astype(float),
                       labels=["lowest likely count", "highest likely count"],
                       pmfs=d["pmfs"], pmf_means=np.array(MU_SHOWN))),
            Panel("The posterior for n = %d" % N_OBS,
                  "ys[0]: the likelihood P(n = %d | mu) as a function of mu; times a flat "
                  "prior (mu >= 0) and normalised it is the posterior ys[1], which here is "
                  "the same curve (its area is exactly 1). The central %.2f%% is the "
                  "credible interval. Exact." % (N_OBS, 100 * CL), "curve",
                  dict(x=d["grid"], ys=np.array([d["like"], d["like"]]),
                       labels=["likelihood", "posterior, flat prior"],
                       credible=np.array(b), confidence=np.array(f))),
            Panel("Both intervals for every count",
                  "For a count x: ys[0], ys[1] the exact central confidence interval; "
                  "ys[2], ys[3] the central credible interval under a flat prior; ys[4], "
                  "ys[5] n -+ sqrt(n). inverse: the credible interval under the prior 1/mu "
                  "(n = 1..20). Exact.", "curve",
                  dict(x=FB_NS, ys=np.array([d["freq"][:, 0], d["freq"][:, 1],
                                             d["bayes"][:, 0], d["bayes"][:, 1],
                                             d["naive"][:, 0], d["naive"][:, 1]]),
                       labels=["confidence, low", "confidence, high", "credible, low",
                               "credible, high", "n - sqrt(n)", "n + sqrt(n)"],
                       inverse=d["inverse"], boundary_counts=np.array(BOUNDARY_NS),
                       boundary_confidence=d["b_freq"], boundary_credible=d["b_bayes"])),
            Panel("How often each contains the truth",
                  "ys: the probability that the interval contains the true mean x, summed "
                  "exactly over all counts, for the confidence interval, the flat-prior "
                  "credible interval and n -+ sqrt(n). at_truth: the same at mu = %g, exact "
                  "and in %d seeded pseudo-experiments." % (MU_TRUE, FB_REPEATS), "curve",
                  dict(x=d["cover_mus"], ys=d["cover"],
                       labels=["confidence", "credible, flat prior", "n -+ sqrt(n)"],
                       exact_at_truth=d["exact_at"], toys_at_truth=d["toy_at"])),
        ),
        dict(
            seed=FB_SEED, seeds=[FB_SEED], review_only=True, level=CL, observed=N_OBS,
            source="exact Poisson arithmetic; %d seeded pseudo-experiments at mu = %g"
                   % (FB_REPEATS, MU_TRUE),
            evidence=_evidence(
                "the belt, both intervals for every count, the posterior, and the "
                "probability of containing the truth at every mean",
                "%d counts drawn from Poisson(mu = %g), seed %d: the share whose interval "
                "contains %g" % (FB_REPEATS, MU_TRUE, FB_SEED, MU_TRUE)),
            where_the_seed_enters=SEED_NOTE,
            confidence_interval=[float(v) for v in f],
            credible_interval=[float(v) for v in b],
            credible_inverse_prior=[float(v) for v in d["inverse"][N_OBS - 1]],
            coverage_at_truth=dict(
                mu=MU_TRUE, repeats=FB_REPEATS,
                exact=[float(v) for v in d["exact_at"]],
                pseudo_experiments=[float(v) for v in d["toy_at"]]),
            background=BACKGROUND,
            limitations=[
                "One model (a Poisson count) and one level (68.27%). Central intervals "
                "only: the unified ordering of Feldman and Cousins, which never gives an "
                "empty interval at a boundary, is not computed.",
                "The credible interval depends on the prior. A flat prior in the mean is a "
                "choice; the prior 1/mu is recorded for comparison and moves both ends "
                "down.",
            ],
            designs_tried=[
                "Only this design was run: n = 3 for the worked example, counts 0 to 20, "
                "truth 3 for the repeated experiments, background 3 for the boundary.",
            ],
        ),
    )


# ====================================================================== 119: Bayes as areas
BAYES_SEED = 119
PEOPLE = 100000
PREVALENCE, SENSITIVITY, FALSE_POSITIVE = 0.001, 0.99, 0.05
KAON_FRACTION, EFFICIENCY, MISID = 0.10, 0.90, 0.05
PARTICLES = 100000
DOTS = 1500                         # of them, drawn in the square
MISIDS = (0.05, 0.01, 0.002)
PRIORS = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])


def posterior(prior, p_given_true, p_given_false):
    """Bayes' theorem for a yes/no hypothesis: the areas ``(true and positive, false and
    positive)`` and the share of the first in their sum."""
    a, b = prior * p_given_true, (1.0 - prior) * p_given_false
    return a, b, a / (a + b)


def tag_sample(rng, n, prior, efficiency, misid):
    """``n`` particles, each a point of the unit square: a kaon where x < ``prior``,
    tagged where y is below its kind's chance of being tagged. Returns ``(x, y, kaon,
    tagged)``."""
    x, y = rng.random(n), rng.random(n)
    kaon = x < prior
    return x, y, kaon, y < np.where(kaon, efficiency, misid)


@functools.lru_cache(maxsize=1)
def bayes_data():
    """Everything ``bayes-theorem-areas`` measures and draws."""
    chain = [PREVALENCE]
    for _ in range(3):                         # each positive result: posterior -> next prior
        chain.append(posterior(chain[-1], SENSITIVITY, FALSE_POSITIVE)[2])
    rng = np.random.default_rng(BAYES_SEED)
    x, y, kaon, tagged = tag_sample(rng, PARTICLES, KAON_FRACTION, EFFICIENCY, MISID)
    grid = np.geomspace(0.001, 0.5, 120)
    exact = np.array([[posterior(p, EFFICIENCY, m)[2] for p in grid] for m in MISIDS])
    toys = np.zeros((len(MISIDS), len(PRIORS)))
    for i, m in enumerate(MISIDS):
        for j, p in enumerate(PRIORS):
            _, _, k, t = tag_sample(rng, PARTICLES, float(p), EFFICIENCY, m)
            toys[i, j] = (k & t).sum() / max(int(t.sum()), 1)
    return dict(
        chain=np.array(chain), xy=np.stack([x[:DOTS], y[:DOTS]], 1),
        classes=(2 * kaon[:DOTS] + tagged[:DOTS]).astype(np.int64),
        counts=np.array([(kaon & tagged).sum(), (kaon & ~tagged).sum(),
                         (~kaon & tagged).sum(), (~kaon & ~tagged).sum()]),
        grid=grid, exact=exact, toys=toys)


def bayes_theorem_areas():
    d = bayes_data()
    ill, healthy, post = posterior(PREVALENCE, SENSITIVITY, FALSE_POSITIVE)
    k, pi, purity = posterior(KAON_FRACTION, EFFICIENCY, MISID)
    c = d["counts"]
    return Study(
        "Bayes' theorem as areas",
        "A population as a unit square, split by the prior and by the test: the posterior "
        "is one rectangle over the sum of two",
        (
            Panel("A test for a rare disease",
                  "Areas of the unit square: prior %g, the test positive for %g%% of the "
                  "ill and %g%% of the healthy. P(ill | positive) is the first area over "
                  "the sum of the first and third. Exact."
                  % (PREVALENCE, 100 * SENSITIVITY, 100 * FALSE_POSITIVE), "bars",
                  dict(values=np.array([ill, PREVALENCE - ill, healthy,
                                        1 - PREVALENCE - healthy]),
                       labels=["ill, positive", "ill, negative", "healthy, positive",
                               "healthy, negative"], posterior=np.array(post))),
            Panel("One positive result after another",
                  "ys[0]: the probability of being ill after x positive results of "
                  "independent tests, each posterior the next prior. Exact.", "curve",
                  dict(x=np.arange(len(d["chain"])), ys=d["chain"][None],
                       labels=["P(ill | x positives)"])),
            Panel("A kaon tag, as a sample",
                  "xy: the first %d of %d particles as points of the unit square (seed "
                  "%d): a kaon where x < %g, tagged where y < %g (kaon) or %g (pion). "
                  "classes: 2 x kaon + tagged. counts: all %d by kind."
                  % (DOTS, PARTICLES, BAYES_SEED, KAON_FRACTION, EFFICIENCY, MISID,
                     PARTICLES), "points",
                  dict(xy=d["xy"], classes=d["classes"], counts=c,
                       labels=["pion, not tagged", "pion, tagged", "kaon, not tagged",
                               "kaon, tagged"], areas=np.array([k, pi, purity]))),
            Panel("Purity against the prior",
                  "ys: the share of tagged particles that are kaons against the kaon "
                  "fraction x of the sample, tag efficiency %g, for three chances of "
                  "tagging a pion (exact). toys: the same counted in %d seeded particles "
                  "at each of toy_x." % (EFFICIENCY, PARTICLES), "curve",
                  dict(x=d["grid"], ys=d["exact"],
                       labels=["mis-identification %g%%" % (100 * m) for m in MISIDS],
                       toy_x=PRIORS, toys=d["toys"])),
        ),
        dict(
            seed=BAYES_SEED, seeds=[BAYES_SEED], review_only=True,
            source="exact arithmetic; %d seeded particles per point" % PARTICLES,
            evidence=_evidence(
                "every area, every posterior and the purity curves",
                "%d particles per sample drawn as uniform points of the unit square "
                "(seed %d), counted by kind and tag" % (PARTICLES, BAYES_SEED)),
            where_the_seed_enters=SEED_NOTE,
            disease=dict(prior=PREVALENCE, sensitivity=SENSITIVITY,
                         false_positive=FALSE_POSITIVE, posterior=float(post),
                         after_positives=[float(v) for v in d["chain"]]),
            kaon=dict(prior=KAON_FRACTION, efficiency=EFFICIENCY, misid=MISID,
                      purity=float(purity), counted=[int(v) for v in c],
                      purity_counted=float(c[0] / (c[0] + c[2]))),
            limitations=[
                "The rates are illustrative round numbers, not those of a real test or "
                "detector. Repeating the test assumes its errors are independent from one "
                "time to the next, which a real test's are not.",
                "At a kaon fraction of 0.1%% a sample of %d holds about 100 kaons: the "
                "counted purity scatters around the exact curve there." % PARTICLES,
            ],
            designs_tried=["Only this design was run."],
        ),
    )


# ====================================================================== 120: a prior washing out
PRIOR_SEED = 120
P_TRUE = 0.7
FLIPS = 100000
GRID = 20000                        # values of p the posteriors are evaluated at
PRIOR_NAMES = ("flat", "optimistic", "sceptical", "narrow", "dogmatic")
OPTIMISTIC, SCEPTICAL, NARROW = (18.0, 2.0), (15.0, 15.0), (100.0, 100.0)
DOGMA = (0.4, 0.6)                  # outside it the dogmatic prior is exactly zero
CHECKS = np.array([0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000,
                   10000, 20000, 50000, 100000])
SHOWN_NS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50, 75, 100, 150, 200,
                     300, 500, 750, 1000])
SHOW_EVERY = 10                     # the drawn curves keep every 10th grid point
OTHER_SEEDS = tuple(range(1200, 1250))


def p_grid():
    """The values of p every posterior is evaluated at: ``GRID`` midpoints of (0, 1)."""
    return (np.arange(GRID) + 0.5) / GRID


def log_priors():
    """The five priors on :func:`p_grid`, as logs (unnormalised; -inf where zero)."""
    p = p_grid()

    def beta(a, b):
        return (a - 1) * np.log(p) + (b - 1) * np.log1p(-p)
    dogma = np.where((p >= DOGMA[0]) & (p <= DOGMA[1]), 0.0, -np.inf)
    return np.array([np.zeros(GRID), beta(*OPTIMISTIC), beta(*SCEPTICAL), beta(*NARROW),
                     dogma])


def posteriors(log_prior, heads, n):
    """Prior times likelihood ``p^heads (1 - p)^(n - heads)``, normalised to area 1, on
    :func:`p_grid`. ``log_prior``: one prior or several (rows)."""
    p = p_grid()
    lp = np.atleast_2d(log_prior) + heads * np.log(p) + (n - heads) * np.log1p(-p)
    w = np.exp(lp - lp.max(1, keepdims=True))
    return w / (w.sum(1, keepdims=True) / GRID)


def density_tv(f, g):
    """Total-variation distance between two densities on :func:`p_grid`."""
    return float(0.5 * np.abs(f - g).sum() / GRID)


def flips(seed, n=FLIPS):
    """Where the seed enters: ``n`` flips of a coin that lands heads with ``P_TRUE``."""
    return np.random.default_rng(seed).random(n) < P_TRUE


def _distances(heads_at, priors):
    """For each checkpoint: the distance of each other prior's posterior to the flat
    prior's, the largest distance between any two of the first three, and each
    posterior's mean."""
    far = np.zeros((4, len(CHECKS)))
    pair = np.zeros(len(CHECKS))
    mean = np.zeros((5, len(CHECKS)))
    p = p_grid()
    for j, (n, h) in enumerate(zip(CHECKS, heads_at)):
        post = posteriors(priors, int(h), int(n))
        far[:, j] = [density_tv(post[0], post[i]) for i in range(1, 5)]
        pair[j] = max(far[0, j], far[1, j], density_tv(post[1], post[2]))
        mean[:, j] = (post * p).sum(1) / GRID
    return far, pair, mean


@functools.lru_cache(maxsize=1)
def prior_data():
    """Everything ``prior-washing-out`` measures and draws."""
    priors = log_priors()
    seq = flips(PRIOR_SEED)
    running = np.concatenate([[0], np.cumsum(seq)])
    far, pair, mean = _distances(running[CHECKS], priors)
    others = np.array([_distances(np.concatenate([[0], np.cumsum(flips(s))])[CHECKS],
                                  priors)[0] for s in OTHER_SEEDS])
    shown = np.array([posteriors(priors, int(running[n]), int(n))[:, SHOW_EVERY // 2::SHOW_EVERY]
                      for n in SHOWN_NS])
    return dict(
        first=seq[:40], heads_at=running[CHECKS], shown_heads=running[SHOWN_NS], far=far,
        pair=pair, mean=mean, shown=shown, p=p_grid()[SHOW_EVERY // 2::SHOW_EVERY],
        quantiles=np.quantile(others, [0.16, 0.5, 0.84], axis=0))


def prior_washing_out():
    d = prior_data()
    q = d["quantiles"]
    return Study(
        "A prior washing out",
        "One seeded run of coin flips (true chance of heads %g) updating five priors: the "
        "distance between their posteriors against the number of flips" % P_TRUE,
        (
            Panel("The posteriors as flips arrive",
                  "values[i, k]: the posterior density of the chance of heads after "
                  "flips[i] flips for prior k (flat; Beta(18, 2); Beta(15, 15); Beta(100, "
                  "100); uniform on 0.4-0.6 and zero elsewhere), at p. Exact, on a grid "
                  "of %d values of p." % GRID, "matrix",
                  dict(values=d["shown"][:, :, ::40].reshape(len(SHOWN_NS), -1),
                       densities=d["shown"], p=d["p"], flips=SHOWN_NS,
                       heads=d["shown_heads"], first=d["first"].astype(np.int64),
                       labels=list(PRIOR_NAMES))),
            Panel("Distance to the flat prior's posterior",
                  "ys: total-variation distance between each other prior's posterior and "
                  "the flat prior's after x flips of the drawn run (seed %d). pair: the "
                  "largest distance between any two of flat, optimistic and sceptical."
                  % PRIOR_SEED, "curve",
                  dict(x=CHECKS, ys=d["far"], labels=list(PRIOR_NAMES[1:]), pair=d["pair"],
                       heads=d["heads_at"])),
            Panel("The same over %d other runs" % len(OTHER_SEEDS),
                  "ys: the median over %d other seeded runs of the same distance; low and "
                  "high: the 16%% and 84%% quantiles." % len(OTHER_SEEDS), "curve",
                  dict(x=CHECKS, ys=q[1], labels=list(PRIOR_NAMES[1:]), low=q[0], high=q[2])),
            Panel("Where each posterior sits",
                  "ys: the posterior mean of the chance of heads after x flips, per "
                  "prior. The truth is %g; the dogmatic prior cannot pass %g."
                  % (P_TRUE, DOGMA[1]), "curve",
                  dict(x=CHECKS, ys=d["mean"], labels=list(PRIOR_NAMES))),
        ),
        dict(
            seed=PRIOR_SEED, seeds=[PRIOR_SEED, *OTHER_SEEDS], review_only=True,
            truth=P_TRUE, flips=FLIPS,
            source="exact posteriors of one seeded run of %d flips, and of %d other runs"
                   % (FLIPS, len(OTHER_SEEDS)),
            evidence=_evidence(
                "every posterior (prior times likelihood on a grid of %d values of p, "
                "normalised) and every distance between two of them" % GRID,
                "the flips: %d draws of a coin with chance of heads %g, seed %d; %d more "
                "runs for the spread" % (FLIPS, P_TRUE, PRIOR_SEED, len(OTHER_SEEDS))),
            where_the_seed_enters=SEED_NOTE,
            priors=dict(flat="Beta(1, 1)", optimistic="Beta(18, 2)",
                        sceptical="Beta(15, 15)", narrow="Beta(100, 100)",
                        dogmatic="uniform on 0.4-0.6, zero elsewhere"),
            distance_to_flat={name: [float(v) for v in d["far"][i]]
                              for i, name in enumerate(PRIOR_NAMES[1:])},
            limitations=[
                "One truth (0.7) and one set of priors. How many flips a prior needs to "
                "wash out depends on how far the truth is from where it puts its weight.",
                "The distances are between whole posteriors (total variation). Two "
                "posteriors can share a mean and still differ in width.",
            ],
            designs_tried=[
                "The narrow prior was first planned as Beta(300, 300) over 10,000 flips; "
                "by arithmetic its posterior would still sit more than two widths from "
                "the flat prior's at the end, so before anything was run the prior was "
                "set to Beta(100, 100) and the run to 100,000 flips, to show a narrow "
                "prior washing out slowly rather than not at all.",
            ],
        ),
    )


# ====================================================================== 121: coverage
COVER_SEED = 121
G_MU, G_SIGMA = 10.0, 2.0
REPEATS = 10000
FIRST = 100                         # of them, drawn one by one
RUN_AT = np.unique(np.round(np.geomspace(10, REPEATS, 60)).astype(int))
MU_EXAMPLE = 2.5
COVER_MUS = np.linspace(0.05, 20.0, 400)
TOY_MUS = np.array([0.5, 1.0, 2.5, 4.0, 6.0, 8.0, 10.0, 12.5, 15.0, 17.5, 20.0])
TRIALS = 20
COVER_PS = np.linspace(0.005, 0.995, 397)
TOY_PS = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98])


@functools.lru_cache(maxsize=1)
def cover_data():
    """Everything ``interval-coverage`` measures and draws."""
    rng = np.random.default_rng(COVER_SEED)
    x = rng.normal(G_MU, G_SIGMA, REPEATS)
    inside = np.abs(x - G_MU) <= G_SIGMA
    running = np.cumsum(inside) / np.arange(1, REPEATS + 1)
    kmax = 80
    naive = np.array([poisson_sqrt(n) for n in range(kmax + 1)])
    exact = np.array([poisson_central(n) for n in range(kmax + 1)])
    cover = np.array([[coverage(poisson_pmf(float(m), kmax), iv[:, 0], iv[:, 1], float(m))
                       for m in COVER_MUS] for iv in (naive, exact)])
    toys = np.zeros((2, len(TOY_MUS)))
    for j, m in enumerate(TOY_MUS):
        n = np.minimum(rng.poisson(m, REPEATS), kmax)
        for i, iv in enumerate((naive, exact)):
            toys[i, j] = ((iv[n, 0] <= m) & (m <= iv[n, 1])).mean()
    pmf = poisson_pmf(MU_EXAMPLE, 9)
    holds = (naive[:10, 0] <= MU_EXAMPLE) & (naive[:10, 1] >= MU_EXAMPLE)
    w = np.array([wald(k, TRIALS) for k in range(TRIALS + 1)])
    cp = np.array([clopper_pearson(k, TRIALS) for k in range(TRIALS + 1)])
    bcover = np.array([[coverage(binomial_pmf(TRIALS, float(p)), iv[:, 0], iv[:, 1], float(p))
                        for p in COVER_PS] for iv in (w, cp)])
    btoys = np.zeros((2, len(TOY_PS)))
    for j, p in enumerate(TOY_PS):
        k = rng.binomial(TRIALS, p, REPEATS)
        for i, iv in enumerate((w, cp)):
            btoys[i, j] = ((iv[k, 0] <= p) & (p <= iv[k, 1])).mean()
    return dict(
        x=x[:FIRST], inside=inside[:FIRST], running=running[RUN_AT - 1],
        total=float(inside.mean()), naive=naive, exact=exact, cover=cover, toys=toys,
        example_pmf=pmf, example_holds=holds, example=float(pmf[holds].sum()),
        example_exact=coverage(poisson_pmf(MU_EXAMPLE, kmax), exact[:, 0], exact[:, 1],
                               MU_EXAMPLE),
        wald=w, cp=cp, bcover=bcover, btoys=btoys)


def interval_coverage():
    d = cover_data()
    return Study(
        "What coverage means",
        "How often an interval built the same way contains a known truth: x +- sigma, "
        "n +- sqrt(n), the exact Poisson interval, and two intervals for %d trials" % TRIALS,
        (
            Panel("%d measurements of a known value" % FIRST,
                  "xy: (experiment, x) for the first %d of %d measurements drawn from a "
                  "Gaussian of mean %g and sigma %g (seed %d); classes: 1 where x +- sigma "
                  "contains %g. running: the share that do after experiments[i]."
                  % (FIRST, REPEATS, G_MU, G_SIGMA, COVER_SEED, G_MU), "points",
                  dict(xy=np.stack([np.arange(FIRST), d["x"]], 1),
                       classes=d["inside"].astype(np.int64), experiments=RUN_AT,
                       running=d["running"])),
            Panel("n +- sqrt(n) at a true mean of %g" % MU_EXAMPLE,
                  "values[n]: the probability of counting n; holds[n]: 1 where n -+ "
                  "sqrt(n) contains %g. The coverage is the summed probability of the "
                  "counts that hold. Exact." % MU_EXAMPLE, "bars",
                  dict(values=d["example_pmf"], labels=[str(n) for n in range(10)],
                       holds=d["example_holds"].astype(np.int64),
                       low=d["naive"][:10, 0], high=d["naive"][:10, 1],
                       exact_low=d["exact"][:10, 0], exact_high=d["exact"][:10, 1])),
            Panel("Coverage against the true mean",
                  "ys: the probability that n -+ sqrt(n) (ys[0]) and the exact central "
                  "interval (ys[1]) contain the true mean x: exact sums. toys: the share "
                  "of %d seeded counts at each of toy_x whose interval does." % REPEATS,
                  "curve",
                  dict(x=COVER_MUS, ys=d["cover"],
                       labels=["n -+ sqrt(n)", "exact central interval"], toy_x=TOY_MUS,
                       toys=d["toys"])),
            Panel("An efficiency from %d trials" % TRIALS,
                  "ys: the probability that p -+ sqrt(p (1 - p) / n) at p = k / n (ys[0]) "
                  "and the Clopper-Pearson interval (ys[1]) contain the true chance x, for "
                  "n = %d: exact sums. toys: the share of %d seeded experiments at toy_x."
                  % (TRIALS, REPEATS), "curve",
                  dict(x=COVER_PS, ys=d["bcover"],
                       labels=["k/n -+ binomial error", "Clopper-Pearson"], toy_x=TOY_PS,
                       toys=d["btoys"], wald=d["wald"], clopper_pearson=d["cp"])),
        ),
        dict(
            seed=COVER_SEED, seeds=[COVER_SEED], review_only=True, level=CL,
            repeats=REPEATS,
            source="exact sums over every outcome; %d seeded pseudo-experiments per point"
                   % REPEATS,
            evidence=_evidence(
                "coverage of each interval at every true value (the summed probability of "
                "the outcomes whose interval contains it)",
                "%d draws per true value from the stated Gaussian, Poisson or binomial "
                "(seed %d): the share whose interval contains the truth"
                % (REPEATS, COVER_SEED)),
            where_the_seed_enters=SEED_NOTE,
            gaussian=dict(mu=G_MU, sigma=G_SIGMA, first=int(d["inside"].sum()),
                          all=d["total"]),
            poisson_example=dict(mu=MU_EXAMPLE, naive=d["example"],
                                 exact=d["example_exact"]),
            poisson_range=dict(naive=[float(d["cover"][0].min()), float(d["cover"][0].max())],
                               exact=[float(d["cover"][1].min()), float(d["cover"][1].max())]),
            binomial_range=dict(
                wald=[float(d["bcover"][0].min()), float(d["bcover"][0].max())],
                clopper_pearson=[float(d["bcover"][1].min()), float(d["bcover"][1].max())]),
            limitations=[
                "Coverage is a property of the procedure over repetitions: it says "
                "nothing about one interval. The exact intervals never under-cover, and "
                "pay for it by over-covering at most true values, because counts are "
                "whole numbers.",
                "Only central intervals at 68.27% are measured; other exact constructions "
                "(Feldman-Cousins, Wilson's approximation) are not.",
            ],
            designs_tried=["Only this design was run."],
        ),
    )


# ====================================================================== 122: three distributions
BPG_SEED = 122
LAM = 3.0
SLOTS, P_SLOT = 10, 0.3             # the worked example: 10 trials at 0.3
ROWS = 8                            # pseudo-experiments drawn as rows of slots
BPG_REPEATS = 20000
BPG_KMAX = 12
SEQ_N = (5, 6, 8, 10, 15, 20, 30, 50, 100, 300, 1000, 10000)
TV_NS = np.unique(np.round(np.geomspace(4, 10000, 50)).astype(int))
SEQ_MU = (1.0, 3.0, 10.0, 30.0, 100.0, 1000.0)
TV_MUS = np.geomspace(0.3, 1000.0, 60)
TAIL_MUS = (10.0, 100.0, 1000.0)
SIGMAS = (1, 2, 3, 4, 5)


def gauss_bins(mean, sd, kmax):
    """A Gaussian's probability of each whole number 0..``kmax`` (from k - 1/2 to
    k + 1/2), and what it puts below -1/2: on counts that cannot happen."""
    c = normal_cdf((np.arange(kmax + 2) - 0.5 - mean) / sd)
    return np.diff(c), float(c[0])


def _tv_to_gauss(pmf, mean, sd):
    """A distribution on 0..k against a Gaussian gathered into whole numbers: half of
    the summed differences, what the Gaussian puts below zero, and what it puts above k."""
    g, below = gauss_bins(mean, sd, len(pmf) - 1)
    return float(0.5 * (np.abs(pmf - g).sum() + below + max(1.0 - below - g.sum(), 0.0)))


def tv_poisson_gauss(mu):
    kmax = int(mu + 12.0 * math.sqrt(mu) + 12.0)
    return _tv_to_gauss(poisson_pmf(mu, kmax), mu, math.sqrt(mu))


def tv_binomial_gauss(n, p):
    return _tv_to_gauss(binomial_pmf(n, p), n * p, math.sqrt(n * p * (1.0 - p)))


def tv_binomial_poisson(n, p):
    """Binomial(n, p) against the Poisson of the same mean, over every count."""
    kmax = int(max(n * p + 12.0 * math.sqrt(n * p + 1.0) + 12.0, 12))
    b = np.zeros(kmax + 1)
    m = min(n, kmax)
    b[:m + 1] = binomial_pmf(n, p)[:m + 1]
    pois = poisson_pmf(n * p, kmax)
    return float(0.5 * (np.abs(b - pois).sum() + abs(b.sum() - pois.sum())))


def poisson_tail(k, mu):
    """P(count >= k | mu), summed from the top so that small tails keep their digits."""
    kmax = int(max(k, mu) + 40.0 * math.sqrt(mu) + 60.0)
    return float(poisson_pmf(mu, kmax)[k:][::-1].sum())


@functools.lru_cache(maxsize=1)
def bpg_data():
    """Everything ``binomial-poisson-gaussian`` measures and draws."""
    rng = np.random.default_rng(BPG_SEED)
    hits = rng.random((BPG_REPEATS, SLOTS)) < P_SLOT
    counted = np.bincount(hits.sum(1), minlength=SLOTS + 1) / BPG_REPEATS
    seq = np.zeros((len(SEQ_N), BPG_KMAX + 1))
    for i, n in enumerate(SEQ_N):
        pmf = binomial_pmf(n, LAM / n)
        seq[i, :min(n, BPG_KMAX) + 1] = pmf[:BPG_KMAX + 1]
    tails = np.zeros((len(TAIL_MUS), len(SIGMAS), 3))
    for i, mu in enumerate(TAIL_MUS):
        for j, z in enumerate(SIGMAS):
            k = int(math.ceil(mu + z * math.sqrt(mu)))
            g = 0.5 * math.erfc((k - 0.5 - mu) / math.sqrt(2.0 * mu))
            tails[i, j] = k, poisson_tail(k, mu), g
    return dict(
        hits=hits[:ROWS], counted=counted, example=binomial_pmf(SLOTS, P_SLOT), seq=seq,
        poisson=poisson_pmf(LAM, BPG_KMAX),
        seq_tv=np.array([tv_binomial_poisson(n, LAM / n) for n in SEQ_N]),
        tv_small=np.array([tv_binomial_poisson(int(n), LAM / n) for n in TV_NS]),
        tv_fixed=np.array([tv_binomial_poisson(int(n), P_SLOT) for n in TV_NS]),
        tv_fixed_gauss=np.array([tv_binomial_gauss(int(n), P_SLOT) for n in TV_NS]),
        tv_gauss=np.array([tv_poisson_gauss(float(m)) for m in TV_MUS]),
        seq_mu_tv=np.array([tv_poisson_gauss(m) for m in SEQ_MU]),
        negative=np.array([gauss_bins(m, math.sqrt(m), 5)[1] for m in SEQ_MU]),
        tails=tails)


def binomial_poisson_gaussian():
    d = bpg_data()
    return Study(
        "Binomial, Poisson, Gaussian",
        "The count in one bin: total-variation distance between the exact distributions "
        "as the number of trials and the mean grow, and the tails",
        (
            Panel("Many trials, each unlikely",
                  "values[i, k]: P(k) for a binomial of trials[i] trials at chance %g / "
                  "trials: mean %g throughout. poisson: the Poisson of mean %g. tv: their "
                  "total-variation distance. Exact." % (LAM, LAM, LAM), "matrix",
                  dict(values=d["seq"], trials=np.array(SEQ_N), poisson=d["poisson"],
                       tv=d["seq_tv"], example=d["example"], counted=d["counted"],
                       hits=d["hits"].astype(np.int64))),
            Panel("Distance from binomial to Poisson",
                  "ys[0]: total-variation distance between binomial(x, %g / x) and "
                  "Poisson(%g). ys[1]: binomial(x, %g) against the Poisson of the same "
                  "mean: the chance is not small and the distance does not fall. ys[2]: "
                  "binomial(x, %g) against a Gaussian of its mean and variance. Exact."
                  % (LAM, LAM, P_SLOT, P_SLOT), "curve",
                  dict(x=TV_NS, ys=np.array([d["tv_small"], d["tv_fixed"],
                                             d["tv_fixed_gauss"]]),
                       labels=["mean fixed at %g: to Poisson" % LAM,
                               "chance fixed at %g: to Poisson" % P_SLOT,
                               "chance fixed at %g: to Gaussian" % P_SLOT])),
            Panel("Distance from Poisson to Gaussian",
                  "ys[0]: total-variation distance between Poisson(x) and a Gaussian of "
                  "mean x and sigma sqrt(x), the Gaussian's probability gathered into "
                  "whole numbers. negative: the probability the Gaussian puts on counts "
                  "below zero, at means. Exact.", "curve",
                  dict(x=TV_MUS, ys=d["tv_gauss"][None], labels=["Poisson to Gaussian"],
                       means=np.array(SEQ_MU), at_means=d["seq_mu_tv"],
                       negative=d["negative"])),
            Panel("The upper tail",
                  "values[i, j]: the exact probability of counting at least means[i] + "
                  "(j + 1) sigma, over the Gaussian's for the same threshold. exact and "
                  "gaussian: the two probabilities; threshold: the count.", "matrix",
                  dict(values=d["tails"][:, :, 1] / d["tails"][:, :, 2],
                       means=np.array(TAIL_MUS), sigmas=np.array(SIGMAS),
                       threshold=d["tails"][:, :, 0], exact=d["tails"][:, :, 1],
                       gaussian=d["tails"][:, :, 2])),
        ),
        dict(
            seed=BPG_SEED, seeds=[BPG_SEED], review_only=True,
            source="exact distributions; %d seeded pseudo-experiments of the worked "
                   "example" % BPG_REPEATS,
            evidence=_evidence(
                "every distribution, every distance and every tail probability",
                "%d rows of %d trials at chance %g (seed %d), their counts set against "
                "the binomial" % (BPG_REPEATS, SLOTS, P_SLOT, BPG_SEED)),
            where_the_seed_enters=SEED_NOTE,
            counted_vs_exact=float(np.abs(d["counted"] - d["example"]).max()),
            limitations=[
                "Total variation is dominated by the bulk of a distribution: two "
                "distributions can be within 1% of each other and differ by a factor of "
                "many in a tail, which is why the tails are measured separately.",
                "The Gaussian is compared after gathering its probability into whole "
                "numbers (k - 1/2 to k + 1/2); without that step the two could not be "
                "compared at all.",
            ],
            designs_tried=["Only this design was run."],
        ),
    )


# ====================================================================== 123: central limit
CLT_SEED = 123
NS = (1, 2, 5, 30)                  # the four drawn
WIDTH_NS = np.array([1, 2, 3, 5, 10, 20, 30, 50, 100])
CLT_REPEATS = 20000
DIE_MEAN, DIE_SD = 3.5, math.sqrt(35.0 / 12.0)
BINS = 60
EXP_RANGE = (0.0, 4.0)
CAUCHY_RANGE = (-8.0, 8.0)
RUN_LENGTH = 10000
RUN_AT_N = np.unique(np.round(np.geomspace(1, RUN_LENGTH, 240)).astype(int))
RUNS = 3


def dice_sum_pmf(n):
    """P(total) of ``n`` dice for totals n..6n: the die's distribution convolved with
    itself (exact)."""
    pmf = np.ones(1)
    for _ in range(n):
        pmf = np.convolve(pmf, np.full(6, 1.0 / 6.0))
    return pmf


def dice_distance(n):
    """Kolmogorov distance (the largest gap between cumulative distributions) between the
    mean of ``n`` dice and the Gaussian of the same mean and width."""
    pmf = dice_sum_pmf(n)
    mean = np.arange(n, 6 * n + 1) / n
    phi = normal_cdf((mean - DIE_MEAN) / (DIE_SD / math.sqrt(n)))
    cum = np.cumsum(pmf)
    return float(max(np.abs(cum - phi).max(), np.abs(cum - pmf - phi).max()))


def exp_mean_cdf(x, n):
    """P(mean of ``n`` decay times of mean life 1 <= x): a Poisson sum (exact)."""
    return np.array([1.0 - poisson_cdf(n - 1, n * float(v)) if v > 0 else 0.0
                     for v in np.atleast_1d(x)])


def exp_mean_pdf(x, n):
    """The density of that mean: a Gamma distribution."""
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    ok = x > 0
    out[ok] = n * np.exp((n - 1) * np.log(n * x[ok]) - n * x[ok] - _LOG_FACT[n - 1])
    return out


def exp_distance(n):
    x = np.linspace(1e-6, 1.0 + 10.0 / math.sqrt(n), 4001)
    return float(np.abs(exp_mean_cdf(x, n) - normal_cdf((x - 1.0) * math.sqrt(n))).max())


def sample_distance(values, mean, sd):
    """Kolmogorov distance between a sample and a Gaussian."""
    v = np.sort(np.asarray(values, float))
    phi = normal_cdf((v - mean) / sd)
    k = np.arange(1, len(v) + 1) / len(v)
    return float(max(np.abs(k - phi).max(), np.abs(k - 1.0 / len(v) - phi).max()))


def cauchy(rng, shape):
    """Draws from a Breit-Wigner (Cauchy) of centre 0 and half-width 1: tan of a uniform
    angle."""
    return np.tan(math.pi * (rng.random(shape) - 0.5))


@functools.lru_cache(maxsize=1)
def clt_data():
    """Everything ``central-limit-theorem`` measures and draws."""
    rng = np.random.default_rng(CLT_SEED)
    out = dict(rolls=rng.integers(1, 7, (12, 2)))
    sd = np.zeros((3, len(WIDTH_NS)))
    far = np.zeros((2, len(WIDTH_NS)))
    half = np.zeros(len(WIDTH_NS))
    edges_e = np.linspace(*EXP_RANGE, BINS + 1)
    edges_c = np.linspace(*CAUCHY_RANGE, BINS + 1)
    for j, n in enumerate(WIDTH_NS):
        dice = rng.integers(1, 7, (CLT_REPEATS, n)).mean(1)
        decay = rng.exponential(1.0, (CLT_REPEATS, n)).mean(1)
        bw = cauchy(rng, (CLT_REPEATS, n)).mean(1)
        sd[:, j] = dice.std(), decay.std(), bw.std()
        far[:, j] = (sample_distance(dice, DIE_MEAN, DIE_SD / math.sqrt(n)),
                     sample_distance(decay, 1.0, 1.0 / math.sqrt(n)))
        q = np.quantile(bw, [0.25, 0.75])
        half[j] = 0.5 * (q[1] - q[0])
        if n in NS:
            out["dice_%d" % n] = np.bincount(np.round(dice * n).astype(int) - n,
                                             minlength=5 * n + 1) / CLT_REPEATS
            out["exp_%d" % n] = np.histogram(decay, edges_e)[0] / (
                CLT_REPEATS * (edges_e[1] - edges_e[0]))
            out["bw_%d" % n] = np.histogram(bw, edges_c)[0] / (
                CLT_REPEATS * (edges_c[1] - edges_c[0]))
    steps = np.arange(1, RUN_LENGTH + 1)
    out.update(
        sd=sd, far=far, half=half,
        dice_exact=np.array([dice_distance(int(n)) for n in WIDTH_NS]),
        exp_exact=np.array([exp_distance(int(n)) for n in WIDTH_NS]),
        run_bw=np.array([(np.cumsum(cauchy(rng, RUN_LENGTH)) / steps)[RUN_AT_N - 1]
                         for _ in range(RUNS)]),
        run_exp=np.array([(np.cumsum(rng.exponential(1.0, RUN_LENGTH)) / steps)[RUN_AT_N - 1]
                          for _ in range(RUNS)]))
    return out


def central_limit_theorem():
    d = clt_data()
    centres_e = np.linspace(*EXP_RANGE, BINS + 1)
    centres_e = 0.5 * (centres_e[1:] + centres_e[:-1])
    return Study(
        "The central limit theorem",
        "The mean of N dice, of N decay times and of N Breit-Wigner draws, for N = 1 to "
        "100: its width and its distance from a Gaussian",
        (
            Panel("The mean of N dice",
                  "values[i]: the exact distribution of the total of N = %s dice (by "
                  "convolution), padded with zeros. counted_N: the share of %d seeded "
                  "pseudo-experiments at each total." % (", ".join(map(str, NS)),
                                                         CLT_REPEATS), "matrix",
                  dict(values=np.array([np.pad(dice_sum_pmf(n), (0, 5 * (NS[-1] - n)))
                                        for n in NS]),
                       ns=np.array(NS), rolls=d["rolls"],
                       **{"counted_%d" % n: d["dice_%d" % n] for n in NS})),
            Panel("The mean of N decay times",
                  "ys: the density of the mean of N = %s decay times of mean life 1, in "
                  "%d seeded pseudo-experiments each (histogram over x). The exact "
                  "density is a Gamma distribution (study_stats_prob.exp_mean_pdf)."
                  % (", ".join(map(str, NS)), CLT_REPEATS), "curve",
                  dict(x=centres_e, ys=np.array([d["exp_%d" % n] for n in NS]),
                       labels=["N = %d" % n for n in NS])),
            Panel("Width and distance against N",
                  "ys[0], ys[1]: the Kolmogorov distance (largest gap between cumulative "
                  "distributions) between the mean of x dice, of x decay times, and the "
                  "Gaussian of the same mean and width: exact. measured: the same from "
                  "%d pseudo-experiments. sd: the standard deviation of the pseudo-"
                  "experiments' means (dice, decay times, Breit-Wigner)." % CLT_REPEATS,
                  "curve",
                  dict(x=WIDTH_NS, ys=np.array([d["dice_exact"], d["exp_exact"]]),
                       labels=["dice", "decay times"], measured=d["far"], sd=d["sd"],
                       sd_exact=np.array([DIE_SD / np.sqrt(WIDTH_NS),
                                          1.0 / np.sqrt(WIDTH_NS)]))),
            Panel("A Breit-Wigner never settles",
                  "ys: the running mean of %d seeded sequences of Breit-Wigner draws "
                  "(centre 0, half-width 1) after x draws. settles: the same for decay "
                  "times. half_width: half the interquartile range of the mean of N "
                  "draws over %d pseudo-experiments (exactly 1 for every N)."
                  % (RUNS, CLT_REPEATS), "curve",
                  dict(x=RUN_AT_N, ys=d["run_bw"],
                       labels=["sequence %d" % (k + 1) for k in range(RUNS)],
                       settles=d["run_exp"], half_width=d["half"], ns=WIDTH_NS,
                       **{"density_%d" % n: d["bw_%d" % n] for n in NS})),
        ),
        dict(
            seed=CLT_SEED, seeds=[CLT_SEED], review_only=True, repeats=CLT_REPEATS,
            source="exact distributions of the mean of N dice and N decay times; %d "
                   "seeded pseudo-experiments per N" % CLT_REPEATS,
            evidence=_evidence(
                "the distribution of the total of N dice (convolution), of the mean of N "
                "decay times (a Gamma distribution), and their Kolmogorov distances to "
                "the Gaussian",
                "%d means of N draws per N and per distribution (seed %d): histograms, "
                "standard deviations, distances, the Breit-Wigner's half-width"
                % (CLT_REPEATS, CLT_SEED)),
            where_the_seed_enters=SEED_NOTE,
            sigma=dict(die=DIE_SD, decay=1.0),
            limitations=[
                "The distances measured from %d pseudo-experiments agree with the exact "
                "ones only to within about 0.01, their sampling noise: at N = 100 that is "
                "the size of the distance itself." % CLT_REPEATS,
                "The standard deviation of the Breit-Wigner means is recorded but means "
                "nothing: the distribution has no variance, and the number is set by the "
                "largest draw.",
            ],
            designs_tried=["Only this design was run."],
        ),
    )


STUDIES = {
    "frequentist-and-bayesian": frequentist_and_bayesian,
    "bayes-theorem-areas": bayes_theorem_areas,
    "prior-washing-out": prior_washing_out,
    "interval-coverage": interval_coverage,
    "binomial-poisson-gaussian": binomial_poisson_gaussian,
    "central-limit-theorem": central_limit_theorem,
}
