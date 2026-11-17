"""The experiments behind animations 137-140: selecting events, and what a selection
does to the numbers measured afterwards (see :func:`voxanim.next_catalog.redesigned`).

* ``moving-a-cut``: one discriminating variable, a Gaussian signal on a Gaussian
  background. Events kept by a cut are counted; efficiency, rejection, purity and three
  significance formulas follow the cut (exact, from the model), with the exact Poisson
  answer beside them where the background is small; and pseudo-experiments measure the
  bias of choosing the cut on the events that are then measured.
* ``cuts-vs-multivariate``: two correlated variables. The best rectangular cuts against
  one line (Fisher's discriminant, from sample means and covariance), chosen on one
  sample and measured on another over several seeds, with the exact curves of the
  model; then a crescent of signal round a blob of background, where no line works.
* ``sideband-background``: a peak on a falling mass spectrum. The background is fitted
  in the sidebands, carried under the peak and subtracted; pseudo-experiments give the
  bias and the pull of the yield for the right shape, a wrong shape and sidebands that
  contain signal.
* ``selection-sculpting``: a falling spectrum through a trigger turn-on and a geometric
  acceptance, the correction by a measured efficiency and its error; and the mass of
  pairs of particles after a cut on their momenta.

Nothing here is measured data: every event is pseudo-data drawn from a model that is
written out in each study's metadata, with ``numpy.random.default_rng(seed)``. Where
the model gives an exact answer it is computed and recorded beside the pseudo-experiments.
"""

from __future__ import annotations

import math

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1


def _settings(**more):
    return dict(version=VERSION, **more)


# ====================================================================== special functions
_erfc = np.frompyfunc(math.erfc, 1, 1)
_lgamma = np.frompyfunc(math.lgamma, 1, 1)


def norm_sf(z):
    """The chance that a standard normal value exceeds ``z`` (accurate in the far tail)."""
    return 0.5 * np.asarray(_erfc(np.asarray(z, np.float64) / math.sqrt(2.0)), np.float64)


def norm_cdf(z):
    return norm_sf(-np.asarray(z, np.float64))


def norm_pdf(z):
    z = np.asarray(z, np.float64)
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_isf(p):
    """The ``z`` a standard normal value exceeds with chance ``p``: Z in sigma for a
    one-sided p-value. Acklam's rational approximation, then two Newton steps."""
    p = np.asarray(p, np.float64)
    q = np.minimum(p, 1.0 - p)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sqrt(-2.0 * np.log(q))
        tail = -(((((c[0] * t + c[1]) * t + c[2]) * t + c[3]) * t + c[4]) * t + c[5]) / (
            (((d[0] * t + d[1]) * t + d[2]) * t + d[3]) * t + 1.0)
        r = q - 0.5
        s = r * r
        mid = -(((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / (
            ((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1.0)
        z = np.where(q < 0.02425, tail, mid)
        for _ in range(2):
            z = z + (norm_sf(z) - q) / np.maximum(norm_pdf(z), 1e-300)
    return np.where(p <= 0.5, z, -z)


def poisson_pmf(k, mu):
    k = np.asarray(k, np.float64)
    return np.exp(k * math.log(mu) - mu - np.asarray(_lgamma(k + 1.0), np.float64))


def poisson_sf(n, mu):
    """P(N >= n) for a Poisson count of mean ``mu`` (``n`` an integer): the p-value of
    seeing ``n`` events when ``mu`` are expected."""
    n = int(n)
    if n <= 0:
        return 1.0
    top = int(max(n, mu) + 60 + 12 * math.sqrt(max(n, mu)))
    return float(poisson_pmf(np.arange(n, top), mu).sum())


def poisson_median(mu):
    """The smallest ``n`` with P(N <= n) >= 1/2."""
    top = int(mu + 60 + 12 * math.sqrt(mu))
    return int(np.searchsorted(np.cumsum(poisson_pmf(np.arange(top), mu)), 0.5))


def z_simple(s, b):
    """s / sqrt(b)."""
    return np.asarray(s, float) / np.sqrt(b)


def z_total(s, b):
    """s / sqrt(s + b)."""
    return np.asarray(s, float) / np.sqrt(np.asarray(s, float) + b)


def z_asimov(s, b):
    """sqrt(2 ((s + b) ln(1 + s/b) - s)): the likelihood-ratio significance of a count
    of s + b when b is expected (Cowan, Cranmer, Gross and Vitells, 2011)."""
    s, b = np.asarray(s, float), np.asarray(b, float)
    return np.sqrt(np.maximum(2.0 * ((s + b) * np.log1p(s / b) - s), 0.0))


def z_observed(n, b):
    """The same formula for an observed count ``n``: zero when ``n`` is not above ``b``."""
    n, b = np.asarray(n, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 2.0 * (n * np.log(np.where(n > 0, n, 1.0) / b) - (n - b))
    return np.where(n > b, np.sqrt(np.maximum(q, 0.0)), 0.0)


def z_poisson(n, b):
    """Z of the exact Poisson p-value P(N >= n | b)."""
    return float(norm_isf(poisson_sf(n, b)))


# ====================================================================== 137: moving a cut
CUT = dict(S=50, B=1000, mu_s=2.0, mu_b=0.0, sigma=1.0)
CUT_SEED = 11
CUT_EDGES = np.linspace(-3.5, 5.5, 37)        # histogram bins of 0.25
CUT_GRID = np.round(np.arange(-1.0, 4.5001, 0.01), 2)
CUT_FIRST = 1.0                               # the cut of the worked example
CUT_TUNE = np.round(np.arange(1.0, 3.4001, 0.1), 1)       # cuts tried when tuning on data
CUT_TOYS = 5000
CUT_TOY_SEED = 2026
CUT_TABLE = ((10, 100), (10, 25), (10, 4), (10, 1))       # (s, b) of the worked table


def cut_expected(c):
    """Expected signal and background above the cut ``c`` (exact, from the model)."""
    c = np.asarray(c, float)
    return (CUT["S"] * norm_sf((c - CUT["mu_s"]) / CUT["sigma"]),
            CUT["B"] * norm_sf((c - CUT["mu_b"]) / CUT["sigma"]))


def cut_counts(xs, xb, cuts):
    """Events of each kind above each cut."""
    xs, xb = np.sort(xs), np.sort(xb)
    return (len(xs) - np.searchsorted(xs, cuts, side="left"),
            len(xb) - np.searchsorted(xb, cuts, side="left"))


def cut_sample(rng, n_s, n_b):
    return (rng.normal(CUT["mu_s"], CUT["sigma"], n_s),
            rng.normal(CUT["mu_b"], CUT["sigma"], n_b))


def _cut_toys(signal, seed):
    """``CUT_TOYS`` pseudo-experiments with expected signal ``signal``: for each, the
    observed count above every cut of ``CUT_TUNE`` in the experiment itself and in an
    independent second sample."""
    rng = np.random.default_rng(seed)
    cuts = CUT_TUNE
    n = np.zeros((2, CUT_TOYS, len(cuts)))
    for t in range(CUT_TOYS):
        for k in range(2):
            xs, xb = cut_sample(rng, rng.poisson(signal), rng.poisson(CUT["B"]))
            cs, cb = cut_counts(xs, xb, cuts)
            n[k, t] = cs + cb
    return n


def _cut_build():
    out = {}
    rng = np.random.default_rng(CUT_SEED)
    xs, xb = cut_sample(rng, CUT["S"], CUT["B"])
    out["xs"], out["xb"] = xs, xb
    out["hist_s"] = np.histogram(xs, CUT_EDGES)[0]
    out["hist_b"] = np.histogram(xb, CUT_EDGES)[0]
    cs, cb = cut_counts(xs, xb, CUT_GRID)
    out["count_s"], out["count_b"] = cs, cb
    s, b = cut_expected(CUT_GRID)
    out["s"], out["b"] = s, b
    out["z_simple"], out["z_total"] = z_simple(s, b), z_total(s, b)
    out["z_asimov"] = z_asimov(s, b)
    # the exact Poisson answer for the median outcome of an experiment with this s and b
    # (up to the cut 3.5: beyond it the median outcome is no events at all)
    every = np.arange(0, 451, 10)
    out["poisson_at"] = CUT_GRID[every]
    out["z_poisson"] = np.array([z_poisson(poisson_median(s[i] + b[i]), b[i]) for i in every])
    out["table"] = np.array([[z_simple(ts, tb), z_asimov(ts, tb), z_poisson(ts + tb, tb),
                              poisson_sf(ts + tb, tb)] for ts, tb in CUT_TABLE])
    # choosing the cut on the events that are then measured
    s_t, b_t = cut_expected(CUT_TUNE)
    eff = s_t / CUT["S"]
    fixed = int(np.argmax(z_asimov(s_t, b_t)))
    out["fixed_cut"] = np.array(CUT_TUNE[fixed])
    rows = np.arange(CUT_TOYS)
    for name, signal, seed in (("sig", CUT["S"], CUT_TOY_SEED), ("null", 0, CUT_TOY_SEED + 1)):
        n = _cut_toys(signal, seed)
        z = z_observed(n, b_t)                        # (sample, toy, cut)
        s_hat = (n[0] - b_t) / eff                    # signal estimate at every cut
        same = z[0].argmax(1)
        other = z[1].argmax(1)                        # tuned on the independent sample
        picks = np.stack([np.full(CUT_TOYS, fixed), same, other])
        out[name + "_pick"] = picks
        out[name + "_s_hat"] = np.stack([s_hat[rows, p] for p in picks])
        out[name + "_z"] = np.stack([z[0][rows, p] for p in picks])
    # exact, for the fixed cut with no signal: the chance of an observed Z above 2
    k = 0
    while z_observed(k, b_t[fixed]) <= 2.0:
        k += 1
    out["null_fixed_exact"] = np.array(poisson_sf(k, b_t[fixed]))
    return out


def cut_data():
    """Everything ``moving-a-cut`` draws."""
    return cached_npz("select-cut", _settings(model=CUT, seed=CUT_SEED, toys=CUT_TOYS,
                                              toy_seed=CUT_TOY_SEED, tune=CUT_TUNE.tolist(),
                                              table=CUT_TABLE), _cut_build)


def best_cuts(d):
    """The cut that maximises each figure of merit, and the value there."""
    purity = d["s"] / (d["s"] + d["b"])
    out = {}
    for name, curve in (("purity", purity), ("z_simple", d["z_simple"]),
                        ("z_total", d["z_total"]), ("z_asimov", d["z_asimov"])):
        i = int(np.argmax(curve))
        s, b = float(d["s"][i]), float(d["b"][i])
        out[name] = dict(cut=float(CUT_GRID[i]), value=float(curve[i]), s=s, b=b,
                         z_asimov=float(d["z_asimov"][i]),
                         z_poisson=z_poisson(poisson_median(s + b), b) if s + b > 1 else 0.0)
    return out


def moving_a_cut():
    d = cut_data()
    s, b = d["s"], d["b"]
    best = best_cuts(d)
    mids = (CUT_EDGES[:-1] + CUT_EDGES[1:]) / 2
    sig, null = d["sig_s_hat"], d["null_z"]
    se = sig.std(1, ddof=1) / math.sqrt(CUT_TOYS)
    names = np.array(["cut fixed beforehand", "cut tuned on the same events",
                      "cut tuned on an independent sample"])
    panels = (
        Panel("One pseudo-dataset",
              "%d signal and %d background events of one discriminating variable x, in bins "
              "of 0.25." % (CUT["S"], CUT["B"]), "curve",
              dict(x=mids, ys=np.stack([d["hist_s"], d["hist_b"]]).astype(float),
                   labels=np.array(["signal", "background"]), edges=CUT_EDGES)),
        Panel("Efficiency, rejection and purity against the cut",
              "Expected (exact) for events kept with x above the cut.", "curve",
              dict(x=CUT_GRID, ys=np.stack([s / CUT["S"], 1 - b / CUT["B"], s / (s + b)]),
                   labels=np.array(["signal efficiency", "background rejection", "purity"]),
                   s=s, b=b, count_s=d["count_s"].astype(float),
                   count_b=d["count_b"].astype(float))),
        Panel("Three formulas for the significance",
              "s/sqrt(b), s/sqrt(s+b) and sqrt(2((s+b)ln(1+s/b)-s)) against the cut, with "
              "the Z of the exact Poisson p-value for the median outcome.", "curve",
              dict(x=CUT_GRID, ys=np.stack([d["z_simple"], d["z_total"], d["z_asimov"]]),
                   labels=np.array(["s/sqrt(b)", "s/sqrt(s+b)", "likelihood-ratio formula"]),
                   poisson_at=d["poisson_at"], z_poisson=d["z_poisson"], table=d["table"],
                   table_s_b=np.array(CUT_TABLE, float))),
        Panel("Choosing the cut on the events that are measured",
              "Mean estimated signal over %d pseudo-experiments (true: %d) for three ways "
              "of choosing the cut." % (CUT_TOYS, CUT["S"]), "bars",
              dict(labels=names, values=sig.mean(1), standard_error=se,
                   s_hat=sig, picks=d["sig_pick"].astype(float),
                   mean_z=d["sig_z"].mean(1),
                   null_above_2=(null > 2.0).mean(1), null_mean_z=null.mean(1),
                   null_fixed_exact=d["null_fixed_exact"])),
    )
    return Study(
        "Moving a cut",
        "What is kept, how pure it is and how significant, as one cut slides",
        panels,
        metadata=dict(
            evidence=dict(
                exact="efficiency, rejection, purity and the three formulas from the "
                      "model's Gaussian integrals; the Poisson p-values",
                pseudo_data="the histogram and its counts: one seeded draw",
                pseudo_experiments="%d seeded repeats for the bias of tuning the cut, "
                                   "with and without signal" % CUT_TOYS),
            model=dict(CUT, signal="Gaussian, mean 2, sigma 1, 50 events expected",
                       background="Gaussian, mean 0, sigma 1, 1000 events expected",
                       selection="keep x > cut"),
            seeds=dict(dataset=CUT_SEED, experiments=[CUT_TOY_SEED, CUT_TOY_SEED + 1]),
            repeats=CUT_TOYS,
            best=best,
            first_cut=CUT_FIRST,
            tuning=dict(cuts=CUT_TUNE.tolist(), fixed_cut=float(d["fixed_cut"]),
                        figure="observed Z = sqrt(2 (n ln(n/b) - (n - b))) with the "
                               "expected background b known",
                        estimate="(n - b) / signal efficiency, unbiased at any fixed cut",
                        mean_estimate=sig.mean(1).tolist(), standard_error=se.tolist(),
                        null_fraction_above_2=(null > 2.0).mean(1).tolist(),
                        null_fixed_exact=float(d["null_fixed_exact"])),
            limitations="One variable with equal-width Gaussians: purity then only rises "
                        "with the cut, so its best cut is the tightest one, which keeps "
                        "no events. The background under the cut is taken as known "
                        "exactly; an uncertain background lowers every significance "
                        "here. The Poisson Z is for the median outcome and steps with "
                        "the integer count.",
            designs_tried=[
                "considered a narrower signal (sigma 0.6), which gives purity an "
                "interior maximum but puts the s/sqrt(b) optimum at b = 5, where the "
                "formula is not yet badly wrong; kept equal widths, chosen from the "
                "exact curves before the tuning experiment was run",
                "as built: tuning over 25 cuts from 1.0 to 3.4, the fixed cut being the "
                "one with the best expected likelihood-ratio Z",
            ],
        ),
    )


# ====================================================================== 138: two variables
MV = dict(mu_s=(1.0, 0.0), mu_b=(0.0, 0.0), sigma=(1.0, 1.0), rho=0.8)
MV_SEEDS = (0, 1, 2, 3, 4)
MV_TRAIN, MV_TEST, MV_SHOWN = 5000, 20000, 400
MV_EFF = np.round(np.arange(0.05, 0.9501, 0.05), 2)       # signal efficiencies compared
MV_AT = 0.80                                  # ... and the one quoted
MV_EDGES = 200                                # thresholds tried on each variable
BAN = dict(spread=1.3, bend=0.5, drop=1.2, noise=0.35, bkg_mean=(0.0, 0.6), bkg_sigma=0.6)
BAN_GRID = (np.linspace(-4.0, 4.0, 161), np.linspace(-2.5, 4.5, 141))


def mv_cov():
    s1, s2 = MV["sigma"]
    r = MV["rho"]
    return np.array([[s1 * s1, r * s1 * s2], [r * s1 * s2, s2 * s2]])


def mv_sample(rng, n):
    """``n`` signal and ``n`` background events of the two tilted clouds."""
    chol = np.linalg.cholesky(mv_cov())
    return (rng.standard_normal((n, 2)) @ chol.T + np.asarray(MV["mu_s"]),
            rng.standard_normal((n, 2)) @ chol.T + np.asarray(MV["mu_b"]))


def fisher(sig, bkg):
    """Fisher's direction from two samples: ``W^-1 (mean_s - mean_b)`` with ``W`` the
    average of the two covariance matrices. Returns ``(w, mean_s, mean_b, W)``."""
    ms, mb = sig.mean(0), bkg.mean(0)
    w_mat = 0.5 * (np.cov(sig.T) + np.cov(bkg.T))
    return np.linalg.solve(w_mat, ms - mb), ms, mb, w_mat


def roc_at(score_s, score_b, eff, threshold_from=None):
    """Background kept when the cut on a score keeps ``eff`` of the signal of
    ``threshold_from`` (default: of ``score_s`` itself). Returns the background kept,
    the signal kept and the thresholds."""
    src = score_s if threshold_from is None else threshold_from
    thr = np.quantile(src, 1.0 - np.asarray(eff))
    return ((score_b[None, :] > thr[:, None]).mean(1), (score_s[None, :] > thr[:, None]).mean(1),
            thr)


def _kept(counts):
    """From a 2-D histogram, the fraction kept by each pair of thresholds, for the four
    orientations (x1 above or below a) x (x2 above or below b)."""
    total = counts.sum()
    lo_lo = np.zeros((counts.shape[0] + 1, counts.shape[1] + 1))
    lo_lo[1:, 1:] = counts.cumsum(0).cumsum(1)
    row, col = lo_lo[:, -1:], lo_lo[-1:, :]
    return np.stack([total - row - col + lo_lo,       # x1 > a, x2 > b
                     col - lo_lo,                     # x1 > a, x2 < b
                     row - lo_lo,                     # x1 < a, x2 > b
                     lo_lo]) / total                  # x1 < a, x2 < b


ORIENT = ((">", ">"), (">", "<"), ("<", ">"), ("<", "<"))


def rect_apply(x, cut):
    """Which events a rectangular cut ``(orientation, a, b)`` keeps."""
    o, a, b = int(cut[0]), cut[1], cut[2]
    k1 = x[:, 0] > a if ORIENT[o][0] == ">" else x[:, 0] < a
    k2 = x[:, 1] > b if ORIENT[o][1] == ">" else x[:, 1] < b
    return k1 & k2


def rect_train(sig, bkg, eff):
    """For each signal efficiency in ``eff``: the rectangular cut (one threshold on each
    variable, either direction) that keeps at least that much of ``sig`` and the least of
    ``bkg``. Returns rows ``(orientation, a, b)``."""
    both = np.concatenate([sig, bkg])
    q = np.linspace(0.0, 1.0, MV_EDGES + 1)
    e1, e2 = np.quantile(both[:, 0], q), np.quantile(both[:, 1], q)
    e1[0], e2[0], e1[-1], e2[-1] = -np.inf, -np.inf, np.inf, np.inf
    ks = _kept(np.histogram2d(sig[:, 0], sig[:, 1], [e1, e2])[0])
    kb = _kept(np.histogram2d(bkg[:, 0], bkg[:, 1], [e1, e2])[0])
    cuts = []
    for target in eff:
        o, i, j = np.unravel_index(np.argmin(np.where(ks >= target - 1e-12, kb, np.inf)),
                                   kb.shape)
        cuts.append((o, e1[i], e2[j]))
    return np.array(cuts, float)


def mv_exact():
    """The model's own curves. Fisher's line: background kept = sf(isf(eff) + d), with
    d the distance between the means in units of the spread along the line. Rectangular
    cuts: the two densities integrated on a fine grid, the best pair of thresholds at
    each signal efficiency."""
    cov = mv_cov()
    delta = np.asarray(MV["mu_s"], float) - np.asarray(MV["mu_b"], float)
    d = float(np.sqrt(delta @ np.linalg.solve(cov, delta)))
    line = norm_sf(norm_isf(MV_EFF) + d)
    edges = np.linspace(-6.0, 7.0, 911)
    mids = (edges[:-1] + edges[1:]) / 2
    inv, det = np.linalg.inv(cov), np.linalg.det(cov)
    kept = []
    for mu in (MV["mu_s"], MV["mu_b"]):
        a, b = mids[:, None] - mu[0], mids[None, :] - mu[1]
        dens = np.exp(-0.5 * (inv[0, 0] * a * a + 2 * inv[0, 1] * a * b + inv[1, 1] * b * b))
        kept.append(_kept(dens / (2 * np.pi * np.sqrt(det)) * (edges[1] - edges[0]) ** 2))
    ks, kb = kept
    rect, where = [], []
    for target in MV_EFF:
        o, i, j = np.unravel_index(np.argmin(np.where(ks >= target, kb, np.inf)), kb.shape)
        rect.append(kb[o, i, j])
        where.append((o, edges[i], edges[j]))
    one = norm_sf(norm_isf(MV_EFF) + delta[0] / MV["sigma"][0])       # a cut on x1 alone
    # a cut on x1 alone is a rectangular cut too (the other threshold at infinity), and
    # its value is known in closed form: the grid's answer cannot honestly be above it
    return dict(d=d, line=line, rect=np.minimum(np.array(rect), one),
                rect_cut=np.array(where, float), one=one, w=np.linalg.solve(cov, delta))


def banana_sample(rng, n):
    """A crescent of signal (x2 = bend * x1^2 - drop, plus noise) round a blob of
    background."""
    x1 = rng.normal(0.0, BAN["spread"], n)
    sig = np.stack([x1, BAN["bend"] * x1 ** 2 - BAN["drop"] + rng.normal(0, BAN["noise"], n)],
                   1)
    bkg = rng.normal(0.0, BAN["bkg_sigma"], (n, 2)) + np.asarray(BAN["bkg_mean"])
    return sig, bkg


def banana_log_ratio(x):
    """ln of (signal density / background density) of the crescent model: the best
    possible variable, written out from the model."""
    x1, x2 = x[..., 0], x[..., 1]
    ls = (-0.5 * (x1 / BAN["spread"]) ** 2 - math.log(BAN["spread"])
          - 0.5 * ((x2 - BAN["bend"] * x1 ** 2 + BAN["drop"]) / BAN["noise"]) ** 2
          - math.log(BAN["noise"]))
    sb = BAN["bkg_sigma"]
    lb = (-0.5 * ((x1 - BAN["bkg_mean"][0]) ** 2 + (x2 - BAN["bkg_mean"][1]) ** 2) / sb ** 2
          - 2 * math.log(sb))
    return ls - lb


def auc(score_s, score_b):
    """The chance that a signal event scores above a background event."""
    order = np.argsort(np.concatenate([score_s, score_b]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    n, m = len(score_s), len(score_b)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def _mv_build():
    out = {}
    at = int(np.argmin(np.abs(MV_EFF - MV_AT)))
    rows = dict(rect=[], line=[], one=[], rect_eff=[], line_eff=[], ban_line=[], ban_curve=[],
                ban_auc=[])
    for seed in MV_SEEDS:
        rng = np.random.default_rng(seed)
        sig, bkg = mv_sample(rng, MV_TRAIN)
        test_s, test_b = mv_sample(rng, MV_TEST)
        w, ms, mb, w_mat = fisher(sig, bkg)
        kept_b, kept_s, thr = roc_at(test_s @ w, test_b @ w, MV_EFF, sig @ w)
        rows["line"].append(kept_b)
        rows["line_eff"].append(kept_s)
        cuts = rect_train(sig, bkg, MV_EFF)
        rows["rect"].append([rect_apply(test_b, c).mean() for c in cuts])
        rows["rect_eff"].append([rect_apply(test_s, c).mean() for c in cuts])
        rows["one"].append(roc_at(test_s[:, 0], test_b[:, 0], MV_EFF, sig[:, 0])[0])
        if seed == MV_SEEDS[0]:
            out["shown_s"], out["shown_b"] = sig[:MV_SHOWN], bkg[:MV_SHOWN]
            out["w"], out["mean_s"], out["mean_b"], out["w_mat"] = w, ms, mb, w_mat
            out["line_thr"] = np.array(thr[at])
            out["rect_cut"] = cuts[at]
            out["shown_counts"] = np.array([
                rect_apply(sig[:MV_SHOWN], cuts[at]).sum(),
                rect_apply(bkg[:MV_SHOWN], cuts[at]).sum(),
                (sig[:MV_SHOWN] @ w > thr[at]).sum(), (bkg[:MV_SHOWN] @ w > thr[at]).sum()])
            edges = np.linspace(-4.0, 6.0, 41)
            out["proj_edges"] = edges
            scale = 1.0 / np.sqrt(w @ w_mat @ w)      # the projection in units of its spread
            out["proj_scale"] = np.array(scale)
            out["proj_s"] = np.histogram(test_s @ w * scale, edges)[0]
            out["proj_b"] = np.histogram(test_b @ w * scale, edges)[0]
            out["x1_s"] = np.histogram(test_s[:, 0], edges)[0]
            out["x1_b"] = np.histogram(test_b[:, 0], edges)[0]
            out["x2_s"] = np.histogram(test_s[:, 1], edges)[0]
            out["x2_b"] = np.histogram(test_b[:, 1], edges)[0]
        # the crescent: a line (Fisher) against the model's own likelihood ratio
        rng = np.random.default_rng(100 + seed)
        sig, bkg = banana_sample(rng, MV_TRAIN)
        test_s, test_b = banana_sample(rng, MV_TEST)
        w = fisher(sig, bkg)[0]
        line_b, _, line_thr = roc_at(test_s @ w, test_b @ w, MV_EFF, sig @ w)
        ratio_s, ratio_b = banana_log_ratio(test_s), banana_log_ratio(test_b)
        curve_b, _, curve_thr = roc_at(ratio_s, ratio_b, MV_EFF, banana_log_ratio(sig))
        rows["ban_line"].append(line_b)
        rows["ban_curve"].append(curve_b)
        rows["ban_auc"].append([auc(test_s @ w, test_b @ w), auc(ratio_s, ratio_b)])
        if seed == MV_SEEDS[0]:
            out["ban_s"], out["ban_b"] = sig[:MV_SHOWN], bkg[:MV_SHOWN]
            out["ban_w"], out["ban_line_thr"] = w, np.array(line_thr[at])
            out["ban_curve_thr"] = np.array(curve_thr[at])
            gx, gy = np.meshgrid(*BAN_GRID)
            out["ban_region"] = banana_log_ratio(np.stack([gx, gy], -1)) > curve_thr[at]
            out["ban_shown_counts"] = np.array([
                (sig[:MV_SHOWN] @ w > line_thr[at]).sum(),
                (bkg[:MV_SHOWN] @ w > line_thr[at]).sum(),
                (banana_log_ratio(sig[:MV_SHOWN]) > curve_thr[at]).sum(),
                (banana_log_ratio(bkg[:MV_SHOWN]) > curve_thr[at]).sum()])
    for k, v in rows.items():
        out[k] = np.array(v, float)
    exact = mv_exact()
    for k, v in exact.items():
        out["exact_" + k] = np.asarray(v, float)
    return out


def mv_data():
    """Everything ``cuts-vs-multivariate`` draws."""
    return cached_npz("select-mv", _settings(model=MV, banana=BAN, seeds=MV_SEEDS,
                                             train=MV_TRAIN, test=MV_TEST, eff=MV_EFF.tolist(),
                                             edges=MV_EDGES), _mv_build)


def cuts_vs_multivariate():
    d = mv_data()
    at = int(np.argmin(np.abs(MV_EFF - MV_AT)))
    rect, line, one = d["rect"], d["line"], d["one"]
    values = np.array([one[:, at].mean(), rect[:, at].mean(), line[:, at].mean()])
    spread = np.array([[v[:, at].min(), v[:, at].max()] for v in (one, rect, line)])
    exact_at = np.array([d["exact_one"][at], d["exact_rect"][at], d["exact_line"][at]])
    classes = np.concatenate([np.ones(MV_SHOWN), np.zeros(MV_SHOWN)])
    panels = (
        Panel("Two correlated variables",
              "%d signal and %d background events of the two tilted clouds." % (
                  MV_SHOWN, MV_SHOWN), "points",
              dict(xy=np.concatenate([d["shown_s"], d["shown_b"]]), classes=classes,
                   mean_s=d["mean_s"], mean_b=d["mean_b"], w_mat=d["w_mat"], w=d["w"],
                   rect_cut=d["rect_cut"], line_threshold=d["line_thr"])),
        Panel("Background kept against signal efficiency",
              "Held-out events, mean of %d seeds: the best rectangular cuts and Fisher's "
              "line; then the model's exact curves." % len(MV_SEEDS), "curve",
              dict(x=MV_EFF, ys=np.stack([rect.mean(0), line.mean(0), d["exact_rect"],
                                          d["exact_line"]]),
                   labels=np.array(["rectangular cuts", "Fisher's line",
                                    "rectangular cuts, exact", "Fisher's line, exact"]),
                   rect_seeds=rect, line_seeds=line, one_variable=one.mean(0),
                   one_exact=d["exact_one"], rect_efficiency=d["rect_eff"],
                   line_efficiency=d["line_eff"])),
        Panel("Background kept at %d%% signal efficiency" % round(100 * MV_AT),
              "Held-out, mean of %d seeds." % len(MV_SEEDS), "bars",
              dict(labels=np.array(["x1 alone", "rectangular cuts", "Fisher's line"]),
                   values=values, spread=spread, exact=exact_at)),
        Panel("A crescent round a blob: a line against a curved boundary",
              "Background kept against signal efficiency, held-out, mean of %d seeds." % len(
                  MV_SEEDS), "curve",
              dict(x=MV_EFF, ys=np.stack([d["ban_line"].mean(0), d["ban_curve"].mean(0)]),
                   labels=np.array(["Fisher's line", "the model's likelihood ratio"]),
                   line_seeds=d["ban_line"], curve_seeds=d["ban_curve"], auc=d["ban_auc"],
                   xy=np.concatenate([d["ban_s"], d["ban_b"]]), classes=classes)),
    )
    return Study(
        "Cuts against a combined variable",
        "Two thresholds against one line through two correlated variables",
        panels,
        metadata=dict(
            evidence=dict(
                exact="the model's curves: Fisher's line in closed form; rectangular "
                      "cuts by integrating the two densities on a 910 x 910 grid",
                pseudo_data="clouds drawn from the stated model, seeds %s: %d events of "
                            "each kind to choose the cuts and the line, %d others of each "
                            "kind to measure them" % (list(MV_SEEDS), MV_TRAIN, MV_TEST)),
            model=dict(MV, kind="two Gaussian clouds with the same covariance"),
            crescent=dict(BAN, signal="x1 Gaussian (sigma 1.3); x2 = 0.5 x1^2 - 1.2 plus "
                          "Gaussian noise (sigma 0.35)",
                          background="round Gaussian at (0, 0.6), sigma 0.6",
                          curved_boundary="a contour of the model's likelihood ratio: the "
                                          "best possible, known here only because the "
                                          "model is; a classifier has to learn it"),
            seeds=list(MV_SEEDS),
            at_efficiency=MV_AT,
            background_kept=dict(x1_alone=float(values[0]), rectangular=float(values[1]),
                                 line=float(values[2]), spread=spread.tolist(),
                                 exact=exact_at.tolist()),
            fisher=dict(w=d["w"].tolist(), exact_w=d["exact_w"].tolist(),
                        mean_s=d["mean_s"].tolist(), mean_b=d["mean_b"].tolist(),
                        covariance=d["w_mat"].tolist(), distance=float(d["exact_d"])),
            crescent_at=dict(line=float(d["ban_line"][:, at].mean()),
                             curve=float(d["ban_curve"][:, at].mean()),
                             auc_line=float(d["ban_auc"][:, 0].mean()),
                             auc_curve=float(d["ban_auc"][:, 1].mean())),
            limitations="Gaussian clouds with equal covariance are the case in which a "
                        "line is the best possible boundary; real variables are neither. "
                        "The rectangular cuts are one threshold per variable in either "
                        "direction, searched over %d x %d thresholds. The curved boundary "
                        "is taken from the model, not learned." % (MV_EDGES, MV_EDGES),
            designs_tried=[
                "as built, fixed before the comparison was run: correlation 0.8, the "
                "means one sigma apart in x1 only, so that x2 alone separates nothing "
                "and yet helps",
            ],
        ),
    )


# ====================================================================== 139: sidebands
SB = dict(lo=100.0, hi=160.0, B=6000, tau=30.0, S=300, peak=125.0, sigma=2.0,
          window=(120.0, 130.0), sidebands=((105.0, 120.0), (130.0, 145.0)),
          wide_sigma=5.0)
SB_SEED = 7
SB_TOYS = 5000
SB_TOY_SEED = 3000
SB_EDGES = np.arange(SB["lo"], SB["hi"] + 0.5, 1.0)
SB_M = (SB_EDGES[:-1] + SB_EDGES[1:]) / 2
SB_REF = 125.0                                # the fits are written about this mass
SB_CASES = ("right shape", "wrong shape", "signal in the sidebands")
SB_LINKS = ("log", "line", "log")
SB_HIST = np.linspace(-250.0, 150.0, 81)      # bins of the estimate minus the truth


def sb_masks():
    m = SB_M
    window = (m > SB["window"][0]) & (m < SB["window"][1])
    side = np.zeros(len(m), bool)
    for a, b in SB["sidebands"]:
        side |= (m > a) & (m < b)
    return window, side


def sb_expected(sigma=None):
    """Expected background and signal events in each 1 GeV bin (exact integrals)."""
    tau = SB["tau"]
    cum = -np.exp(-(SB_EDGES - SB["lo"]) / tau)
    bkg = SB["B"] * np.diff(cum) / (cum[-1] - cum[0])
    sigma = SB["sigma"] if sigma is None else sigma
    sig = SB["S"] * np.diff(norm_cdf((SB_EDGES - SB["peak"]) / sigma))
    return bkg, sig


def poisson_fit(counts, x, link="log", steps=30):
    """Maximum-likelihood fit of Poisson counts, for many spectra at once.

    ``counts``: (spectra, bins); ``x``: (bins,). The expected count is
    ``exp(p0 + p1 x)`` (``link="log"``, an exponential) or ``p0 + p1 x``
    (``link="line"``). Returns the parameters (spectra, 2) and their covariance
    matrices (spectra, 2, 2), the inverse of the information matrix."""
    y = np.atleast_2d(np.asarray(counts, float))
    design = np.stack([np.ones_like(x), x], 1)                # (bins, 2)
    if link == "log":
        p = np.stack([np.log(np.maximum(y.mean(1), 1e-9)), np.zeros(len(y))], 1)
    else:
        p = np.linalg.lstsq(design, y.T, rcond=None)[0].T

    def expect(p):
        return np.exp(p @ design.T) if link == "log" else np.maximum(p @ design.T, 1e-6)

    def information(mu):
        weight = mu if link == "log" else 1.0 / mu
        return np.einsum("tb,bi,bj->tij", weight, design, design)
    for _ in range(steps):
        mu = expect(p)
        score = (y - mu) @ design if link == "log" else ((y - mu) / mu) @ design
        p = p + np.linalg.solve(information(mu), score[..., None])[..., 0]
    return p, np.linalg.inv(information(expect(p)))


def sb_predict(p, cov, x, link):
    """The fitted background summed over the bins at ``x`` and its standard error."""
    design = np.stack([np.ones_like(x), x], 1)
    if link == "log":
        mu = np.exp(p @ design.T)
        grad = mu @ design
    else:
        mu = p @ design.T
        grad = np.broadcast_to(design.sum(0), p.shape)
    return mu.sum(1), np.sqrt(np.einsum("ti,tij,tj->t", grad, cov, grad))


def sb_estimate(counts, link):
    """Sideband subtraction of each spectrum in ``counts``: fit the sidebands, carry
    the fit under the window, subtract. Returns a dict of arrays, one entry per
    spectrum: the window count, the fitted background and its error, the signal yield
    and its error, the fit parameters, and chi-square of the fit in the sidebands."""
    counts = np.atleast_2d(np.asarray(counts, float))
    window, side = sb_masks()
    x = SB_M - SB_REF
    p, cov = poisson_fit(counts[:, side], x[side], link)
    b_hat, b_err = sb_predict(p, cov, x[window], link)
    n = counts[:, window].sum(1)
    design = np.stack([np.ones_like(x[side]), x[side]], 1)
    mu = np.exp(p @ design.T) if link == "log" else p @ design.T
    chi2 = ((counts[:, side] - mu) ** 2 / mu).sum(1)
    return dict(n=n, b=b_hat, b_err=b_err, s=n - b_hat, s_err=np.sqrt(n + b_err ** 2), p=p,
                chi2=chi2, ndf=np.full(len(n), int(side.sum()) - 2))


def sb_curve(p, link, m=None):
    """The fitted background per GeV at the masses ``m``."""
    x = (SB_M if m is None else np.asarray(m, float)) - SB_REF
    return np.exp(p[0] + p[1] * x) if link == "log" else p[0] + p[1] * x


def _sb_build():
    out = {}
    window, _side = sb_masks()
    bkg, sig = sb_expected()
    _, wide = sb_expected(SB["wide_sigma"])
    out["bkg"], out["sig"], out["sig_wide"] = bkg, sig, wide
    out["data"] = np.random.default_rng(SB_SEED).poisson(bkg + sig).astype(float)
    out["data_wide"] = np.random.default_rng(SB_SEED).poisson(bkg + wide).astype(float)
    for k, (link, signal) in enumerate(zip(SB_LINKS, (sig, sig, wide))):
        rng = np.random.default_rng(SB_TOY_SEED + k)
        counts = rng.poisson(bkg + signal, (SB_TOYS, len(bkg))).astype(float)
        truth = signal[window].sum()
        est = sb_estimate(counts, link)
        asimov = sb_estimate(bkg + signal, link)
        one = sb_estimate(out["data_wide"] if k == 2 else out["data"], link)
        pull = (est["s"] - truth) / est["s_err"]
        out["truth_%d" % k] = np.array(truth)
        out["s_%d" % k], out["err_%d" % k] = est["s"], est["s_err"]
        out["hist_%d" % k] = np.histogram(est["s"] - truth, SB_HIST)[0]
        out["summary_%d" % k] = np.array([
            est["s"].mean() - truth, est["s"].std(ddof=1), est["s_err"].mean(), pull.mean(),
            pull.std(ddof=1), np.mean(np.abs(pull) < 1.0), asimov["s"][0] - truth,
            np.mean(est["chi2"] / est["ndf"]), est["b"].mean() - bkg[window].sum()])
        out["one_%d" % k] = np.array([one["n"][0], one["b"][0], one["b_err"][0], one["s"][0],
                                      one["s_err"][0], one["chi2"][0], one["ndf"][0]])
        out["one_p_%d" % k] = one["p"][0]
    return out


SB_SUMMARY = ("bias", "spread", "mean_quoted_error", "pull_mean", "pull_width", "within_1_sigma",
              "bias_expected", "chi2_per_ndf", "background_bias")


def sb_data():
    """Everything ``sideband-background`` draws."""
    return cached_npz("select-sideband", _settings(model=SB, seed=SB_SEED, toys=SB_TOYS,
                                                   toy_seed=SB_TOY_SEED,
                                                   hist=[SB_HIST[0], SB_HIST[-1],
                                                         len(SB_HIST)]), _sb_build)


def sideband_background():
    d = sb_data()
    summary = np.stack([d["summary_%d" % k] for k in range(3)])       # (case, statistic)
    hist_x = (SB_HIST[:-1] + SB_HIST[1:]) / 2
    window, side = sb_masks()
    labels = np.array(SB_CASES)
    panels = (
        Panel("One pseudo-experiment",
              "Events per GeV: a peak of %d events at %g GeV on %d background events "
              "falling as exp(-m / %g GeV); the exponential and the straight line fitted "
              "in the sidebands." % (SB["S"], SB["peak"], SB["B"], SB["tau"]), "curve",
              dict(x=SB_M, ys=np.stack([d["data"], d["bkg"], d["bkg"] + d["sig"],
                                        sb_curve(d["one_p_0"], "log"),
                                        sb_curve(d["one_p_1"], "line")]),
                   labels=np.array(["pseudo-data", "true background", "true total",
                                    "exponential fitted in the sidebands",
                                    "straight line fitted in the sidebands"]),
                   window=window.astype(float), sidebands=side.astype(float),
                   one=np.stack([d["one_%d" % k] for k in range(3)]),
                   data_wide=d["data_wide"], truth_wide=d["bkg"] + d["sig_wide"],
                   fit_wide=sb_curve(d["one_p_2"], "log"))),
        Panel("The yield over %d pseudo-experiments" % SB_TOYS,
              "Estimated minus true signal in the window, for the three cases.", "curve",
              dict(x=hist_x, ys=np.stack([d["hist_%d" % k] for k in range(3)]).astype(float),
                   labels=labels, edges=SB_HIST)),
        Panel("Bias of the yield",
              "Mean of estimate minus truth (events), with the value expected from "
              "fitting the model's own expected spectrum.", "bars",
              dict(labels=labels, values=summary[:, 0], expected=summary[:, 6],
                   standard_error=summary[:, 1] / math.sqrt(SB_TOYS), spread=summary[:, 1],
                   quoted_error=summary[:, 2])),
        Panel("Pull of the yield",
              "(estimate - truth) / quoted error: its mean; width, share within one "
              "sigma and the sidebands' chi-square per degree of freedom are recorded "
              "with it.", "bars",
              dict(labels=labels, values=summary[:, 3], width=summary[:, 4],
                   within_1_sigma=summary[:, 5], chi2_per_ndf=summary[:, 7])),
    )
    return Study(
        "Background from the sidebands",
        "Fit beside the peak, carry the fit under it, subtract",
        panels,
        metadata=dict(
            evidence=dict(
                exact="the expected spectrum (bin integrals of the model) and the bias "
                      "expected from fitting it without fluctuations",
                pseudo_data="the spectrum drawn: Poisson counts per GeV, seed %d" % SB_SEED,
                pseudo_experiments="%d seeded Poisson spectra per case" % SB_TOYS),
            model=dict(SB, background="exp(-m / 30 GeV), 6000 events from 100 to 160 GeV",
                       signal="Gaussian at 125 GeV, sigma 2 GeV, 300 events (sigma 5 GeV "
                              "in the third case, so that 32% of it lies outside the "
                              "window)"),
            method="binned Poisson maximum likelihood in the sideband bins only; yield = "
                   "window count - fitted background; error = sqrt(window count + "
                   "(fit error)^2)",
            cases=list(SB_CASES),
            seeds=dict(spectrum=SB_SEED, experiments=[SB_TOY_SEED + k for k in range(3)]),
            repeats=SB_TOYS,
            truth_in_window=[float(d["truth_%d" % k]) for k in range(3)],
            results={name: dict(zip(SB_SUMMARY, row.tolist()))
                     for name, row in zip(SB_CASES, summary)},
            limitations="A background with a bump or a kink under the peak cannot be "
                        "found from the sidebands at all. The true signal counted is the "
                        "part inside the window. The window count and the sideband fit "
                        "are independent, so their errors add in quadrature.",
            designs_tried=[
                "as built, fixed before the pseudo-experiments were run: sidebands of "
                "15 GeV on each side of a 10 GeV window, 1 GeV bins",
            ],
        ),
    )


# ====================================================================== 140: sculpting
SC = dict(tau=25.0, e_max=100.0, n_true=200_000, threshold=30.0, width=3.0, acceptance=0.7,
          calibration=5000)
SC_SEED = 5
SC_TOYS = 4000
SC_TOY_SEED = 4000
SC_EDGES = np.arange(0.0, SC["e_max"] + 1, 4.0)
SC_E = (SC_EDGES[:-1] + SC_EDGES[1:]) / 2
PAIR = dict(pt_mean=12.0, eta_sigma=1.0, n=2_000_000)
PAIR_SEED = 6
PAIR_CUTS = (0.0, 10.0, 15.0, 20.0, 25.0, 30.0)
PAIR_EDGES = np.arange(0.0, 151.0, 3.0)
PAIR_SHOWN = 4000


def turn_on(e):
    """The trigger's efficiency at energy ``e``: a smooth step at the threshold."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(e, float) - SC["threshold"]) / SC["width"]))


def efficiency(e):
    """Trigger and geometric acceptance together."""
    return SC["acceptance"] * turn_on(e)


def sc_expected():
    """Expected true events per bin and the efficiency of each bin (the average over the
    true spectrum inside it), by fine integration."""
    fine = np.linspace(0.0, SC["e_max"], 100 * (len(SC_EDGES) - 1) + 1)
    mid = (fine[:-1] + fine[1:]) / 2
    dens = np.exp(-mid / SC["tau"])
    dens = dens / dens.sum() * SC["n_true"]
    true = dens.reshape(-1, 100).sum(1)
    rec = (dens * efficiency(mid)).reshape(-1, 100).sum(1)
    return true, rec / true


def sc_peak():
    """Where the recorded spectrum is highest (exact: where the turn-on's relative
    slope equals the spectrum's)."""
    e = np.linspace(0.0, SC["e_max"], 100001)
    return float(e[np.argmax(np.exp(-e / SC["tau"]) * efficiency(e))])


def corrected(recorded, eff, eff_err=0.0):
    """Recorded counts divided by the efficiency, and the error of the result: the
    count's own sqrt(N) and the efficiency's error, added in quadrature."""
    recorded, eff = np.asarray(recorded, float), np.asarray(eff, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = recorded / eff
        rel = np.sqrt(1.0 / np.maximum(recorded, 1e-300) + (eff_err / eff) ** 2)
    return value, value * rel


def pair_mass(pt1, pt2, d_eta, d_phi):
    """The invariant mass of two massless particles."""
    return np.sqrt(2.0 * pt1 * pt2 * (np.cosh(d_eta) - np.cos(d_phi)))


def peak_of(hist, edges, smooth=5):
    """The centre of the highest bin of the histogram's ``smooth``-bin running mean."""
    i = int(np.argmax(np.convolve(hist, np.ones(smooth) / smooth, mode="same")))
    return float((edges[i] + edges[i + 1]) / 2)


def _sc_build():
    out = {}
    rng = np.random.default_rng(SC_SEED)
    # one experiment, event by event: an energy, a direction, a trigger decision
    n = SC["n_true"]
    u = rng.random(n)
    energy = -SC["tau"] * np.log(1.0 - u * (1.0 - math.exp(-SC["e_max"] / SC["tau"])))
    cos_theta = rng.uniform(-1.0, 1.0, n)
    fired = rng.random(n) < turn_on(energy)
    seen = np.abs(cos_theta) < SC["acceptance"]
    out["true"] = np.histogram(energy, SC_EDGES)[0].astype(float)
    out["triggered"] = np.histogram(energy[fired], SC_EDGES)[0].astype(float)
    out["recorded"] = np.histogram(energy[fired & seen], SC_EDGES)[0].astype(float)
    nu, eff = sc_expected()
    out["nu"], out["eff"] = nu, eff
    cal = SC["calibration"]
    k = rng.binomial(cal, eff).astype(float)          # the calibration sample, per bin
    out["cal_pass"] = k
    eff_hat = k / cal
    eff_err = np.sqrt(np.maximum(eff_hat * (1 - eff_hat), 0.0) / cal)
    value, err = corrected(out["recorded"], eff_hat, eff_err)
    ok = (k > 0) & (out["recorded"] > 0)
    out["corrected"] = np.where(ok, value, 0.0)
    out["corrected_err"] = np.where(ok, err, 0.0)
    # many experiments, bin by bin
    rng = np.random.default_rng(SC_TOY_SEED)
    true = rng.poisson(nu, (SC_TOYS, len(nu)))
    rec = rng.binomial(true, eff)
    k = rng.binomial(cal, eff, (SC_TOYS, len(nu)))
    good = (k > 0) & (rec > 0)
    ratio = np.where(good, rec / np.maximum(k, 1) * cal / nu, np.nan)
    some = good.any(0)
    ratio[:, ~some] = 0.0
    out["toy_usable"] = good.mean(0)
    out["toy_mean"] = np.where(some, np.nanmean(ratio, 0), 0.0)
    out["toy_median"] = np.where(some, np.nanmedian(ratio, 0), 0.0)
    lo, hi = np.nanpercentile(ratio, [15.87, 84.13], 0)
    out["toy_spread"] = np.where(some, (hi - lo) / 2, 0.0)
    out["formula"] = np.sqrt(1.0 / (nu * eff) + (1 - eff) / (eff * cal))
    # a cut on momentum, and the mass of pairs
    rng = np.random.default_rng(PAIR_SEED)
    n = PAIR["n"]
    pt = rng.exponential(PAIR["pt_mean"], (2, n))
    d_eta = rng.normal(0.0, PAIR["eta_sigma"], n)
    d_phi = rng.uniform(0.0, np.pi, n)
    mass = pair_mass(pt[0], pt[1], d_eta, d_phi)
    soft = pt.min(0)
    out["pair_mass"], out["pair_soft"] = mass[:PAIR_SHOWN], soft[:PAIR_SHOWN]
    out["pair_corr"] = np.array(np.corrcoef(mass, soft)[0, 1])
    hists = np.array([np.histogram(mass[soft > c], PAIR_EDGES)[0] for c in PAIR_CUTS])
    out["pair_hist"] = hists.astype(float)
    out["pair_kept"] = np.array([(soft > c).mean() for c in PAIR_CUTS])
    out["pair_peak"] = np.array([peak_of(h, PAIR_EDGES) for h in hists])
    return out


def sc_data():
    """Everything ``selection-sculpting`` draws."""
    return cached_npz("select-sculpt", _settings(model=SC, pair=PAIR, seed=SC_SEED,
                                                 toys=SC_TOYS, toy_seed=SC_TOY_SEED,
                                                 pair_seed=PAIR_SEED, cuts=PAIR_CUTS, smooth=5),
                      _sc_build)


def selection_sculpting():
    d = sc_data()
    mass_x = (PAIR_EDGES[:-1] + PAIR_EDGES[1:]) / 2
    peak_bin = int(np.argmax(d["recorded"]))
    ratio = np.polyfit(PAIR_CUTS[1:], d["pair_peak"][1:], 1)
    panels = (
        Panel("True and recorded spectrum",
              "Events per 4 GeV bin: %s true events falling as exp(-E / %g GeV), and those "
              "that fire the trigger and fall inside the acceptance." % (
                  format(SC["n_true"], ","), SC["tau"]), "curve",
              dict(x=SC_E, ys=np.stack([d["true"], d["recorded"], d["triggered"]]),
                   labels=np.array(["true", "recorded", "triggered"]), efficiency=d["eff"],
                   expected_true=d["nu"], edges=SC_EDGES)),
        Panel("Corrected spectrum and its error",
              "Recorded / measured efficiency over %d pseudo-experiments: mean ratio to "
              "the truth, and the relative spread against the formula." % SC_TOYS, "curve",
              dict(x=SC_E, ys=np.stack([d["toy_mean"], d["toy_spread"], d["formula"]]),
                   labels=np.array(["mean of corrected / true", "relative spread",
                                    "sqrt(1/N + (sigma_eff/eff)^2)"]),
                   usable=d["toy_usable"], median=d["toy_median"], corrected=d["corrected"],
                   corrected_err=d["corrected_err"], calibration_pass=d["cal_pass"])),
        Panel("Mass of pairs after a cut on momentum",
              "Pairs per 3 GeV of mass, of %s pairs, when both momenta must exceed each "
              "threshold." % format(PAIR["n"], ","), "curve",
              dict(x=mass_x, ys=d["pair_hist"],
                   labels=np.array(["pT > %g GeV" % c for c in PAIR_CUTS]),
                   edges=PAIR_EDGES)),
        Panel("Where the mass spectrum peaks",
              "The highest point of the mass spectrum (3 GeV bins, 5-bin running mean) for "
              "each threshold.", "bars",
              dict(labels=np.array(["%g" % c for c in PAIR_CUTS]), values=d["pair_peak"],
                   cuts=np.array(PAIR_CUTS), kept=d["pair_kept"],
                   correlation=d["pair_corr"],
                   xy=np.stack([d["pair_soft"], d["pair_mass"]], 1))),
    )
    return Study(
        "How a selection sculpts a spectrum",
        "A threshold and an acceptance turn a smooth spectrum into one with a peak",
        panels,
        metadata=dict(
            evidence=dict(
                exact="the efficiency curve, the expected spectra and the position of "
                      "the recorded maximum",
                pseudo_data="one experiment drawn event by event (seed %d); %s pairs of "
                            "particles (seed %d)" % (SC_SEED, format(PAIR["n"], ","),
                                                     PAIR_SEED),
                pseudo_experiments="%d seeded repeats, bin by bin, of spectrum, "
                                   "selection and calibration sample" % SC_TOYS),
            model=dict(SC, spectrum="exp(-E / 25 GeV) from 0 to 100 GeV, 200,000 events",
                       trigger="efficiency 1 / (1 + exp(-(E - 30 GeV) / 3 GeV))",
                       directions="uniform in cos(theta); seen when |cos(theta)| < 0.7",
                       calibration_sample="the efficiency of each bin measured on 5000 "
                                          "independent events"),
            pairs=dict(PAIR, momenta="each exponential with mean 12 GeV",
                       angles="difference in azimuth uniform from 0 to pi, in "
                              "pseudorapidity Gaussian (sigma 1)",
                       mass="sqrt(2 pT1 pT2 (cosh(d_eta) - cos(d_phi)))",
                       cuts=list(PAIR_CUTS)),
            seeds=dict(experiment=SC_SEED, experiments=SC_TOY_SEED, pairs=PAIR_SEED),
            repeats=SC_TOYS,
            recorded_peak=dict(bin=[float(SC_EDGES[peak_bin]), float(SC_EDGES[peak_bin + 1])],
                               exact=sc_peak()),
            pair_peak=dict(zip(["%g" % c for c in PAIR_CUTS], d["pair_peak"].tolist())),
            pair_peak_slope=float(ratio[0]),
            limitations="The acceptance does not depend on energy here, so it only "
                        "scales the spectrum; the trigger does the sculpting. The "
                        "calibration sample has the true spectrum's shape inside each "
                        "bin, so the bin's average efficiency is the right one: with a "
                        "different shape the correction is biased in the turn-on even "
                        "with a perfect efficiency curve. No energy resolution, so no "
                        "unfolding is needed.",
            designs_tried=[
                "as built, fixed before the pseudo-experiments were run: 4 GeV bins, a "
                "calibration sample of 5000 events per bin",
            ],
        ),
    )


STUDIES = {
    "moving-a-cut": moving_a_cut,
    "cuts-vs-multivariate": cuts_vs_multivariate,
    "sideband-background": sideband_background,
    "selection-sculpting": selection_sculpting,
}
