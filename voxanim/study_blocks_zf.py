"""The numbers behind two animations of "Blocks and code" drawn by group ``zf``: 272 "Error bars
on a network" (``error-bars-on-a-network``) and 278 "The Asimov dataset" (``asimov-dataset``).
Everything is made up and measured by the numpy twins of the two programs.

* ``error-bars-on-a-network`` (:mod:`voxanim.study_ensembles`, ``examples/ensembles.py``): the
  fit of "Minuit and SGD" (40 made-up points of a exp(-x / tau) + b, noise 0.3), trained by SGD
  (3,000 steps, learning rate 0.005, 8 points a step, from (1, 1, 0)) 100 times with new seeds on
  the 40 points, and 100 times on bootstrap resamples (40 picks of the 40, repeats allowed). The
  seeds' answers spread by about 0.02 (0.017, 0.016, 0.021 for a, tau, b): the optimiser's noise
  only. The resamples' answers spread by 0.15, 0.14, 0.11, against HESSE's errors 0.18, 0.21,
  0.11: near for a and b, about a third short for tau. Their tau-b correlation is -0.82 (HESSE
  -0.81): the ellipses tilt alike. Drawn in the (tau, b) plane, with HESSE's chi-square + 1
  ellipse (its 2 x 2 covariance, a refitted) and the resamples' own covariance ellipse.
* ``asimov-dataset`` (:mod:`voxanim.study_asimov`, ``examples/asimov.py``): 4 made-up bins,
  signal (3, 8, 8, 3) on background (30, 20, 12, 8). 10,000 toy experiments' significances
  (signal strength fitted, Z = sqrt(q0)) have median 2.89; the Asimov dataset (every bin exactly
  s + b) gives 2.93 in one go; s / sqrt(b) bin by bin gives 3.16. With a tenth of the background:
  median 6.74, Asimov 6.77, s / sqrt(b) 9.98.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_asimov as A
from . import study_ensembles as E
from . import study_minuit as M
from .blocks import code_lines, commas
from .study import Panel, Study

# ====================================================================== 272
EN_SOURCE = ("made-up data: examples/ensembles.py (the 40 points of examples/minuit_vs_sgd.py, "
             "a exp(-x/tau) + b with noise 0.3); 100 SGD runs with seeds 0-99 and 100 on "
             "bootstrap resamples (seeds 100-199), in numpy; HESSE recorded by iminuit %s")
PLANE = (1, 2)                       # the plane drawn: tau across, b up
TAU_RANGE = (2.05, 2.95)
B_RANGE = (0.6, 1.35)
NAMES = ("a", "τ", "b")


def two(v):
    """``0.15``: a spread or an error to 2 places."""
    return "%.2f" % float(v)


def three(v):
    """``0.017``: a small spread to 3 places."""
    return "%.3f" % float(v)


def signed(v, places=2):
    """``−0.82``: a correlation, with a true minus sign."""
    return ("%.*f" % (places, float(v))).replace("-", "−")


@functools.lru_cache(maxsize=1)
def ensembles():
    c = E.compare()
    mn = E.minuit()
    x, y = M.data()
    printed = E.printed()
    seeds, boots = c["seeds"], c["boots"]
    counts = c["counts"]
    m = dict(
        source=EN_SOURCE % mn["iminuit"], iminuit=mn["iminuit"], x=x.tolist(), y=y.tolist(),
        n_points=M.N, sigma=M.SIGMA, runs=E.RUNS, steps=E.STEPS, batch=E.BATCH, lr=E.LR,
        start=list(E.START), seeds_list=list(E.SEEDS), boot_seeds=list(E.BOOT_SEEDS),
        draw_seed=E.DRAW_SEED, names=list(NAMES), plane=list(PLANE),
        tau_range=list(TAU_RANGE), b_range=list(B_RANGE),
        values=mn["values"].tolist(), fval=mn["fval"],
        seeds=seeds.tolist(), boots=boots.tolist(),
        seeds_mean=c["seeds_mean"].tolist(), boots_mean=c["boots_mean"].tolist(),
        seeds_spread=c["seeds_spread"].tolist(), boots_spread=c["boots_spread"].tolist(),
        hesse=c["hesse"].tolist(), hesse_cov=c["hesse_cov"].tolist(),
        boots_cov=c["boots_cov"].tolist(), ratio=c["ratio"].tolist(),
        corr_boots=c["corr_tau_b_boots"], corr_hesse=c["corr_tau_b_hesse"],
        hesse_ellipse=c["hesse_ellipse"].tolist(), boots_ellipse=c["boots_ellipse"].tolist(),
        counts=counts.tolist(), distinct=c["distinct"].tolist(),
        sgd_vs_exact=c["sgd_vs_exact"].tolist(), exact_spread=c["exact_spread"].tolist(),
        toys_spread=c["toys_spread"].tolist(), toys=E.TOYS,
        profile_on_ellipse=list(c["profile_on_ellipse"]),
        printed_seeds=printed["seeds"], printed_boots=printed["boots"],
        printed_cov=printed["cov"], printed_hesse=printed["hesse"],
        code_train=list(code_lines("en-train")), code_seeds=list(code_lines("en-seeds")),
        code_bootstrap=list(code_lines("en-bootstrap")),
        code_spread=list(code_lines("en-spread")), code_hesse=list(code_lines("en-hesse")))
    m.update(
        first=seeds[0].tolist(),
        first_text=["%.2f" % seeds[0][0], "%.2f" % seeds[0][1], "%.2f" % seeds[0][2]],
        values_text=["%.2f" % v for v in mn["values"]],
        seeds_text=[three(v) for v in c["seeds_spread"]],
        seeds_round="%.2f" % float(np.round(c["seeds_spread"].max(), 2)),
        boots_text=[two(v) for v in c["boots_spread"]],
        hesse_text=[two(v) for v in c["hesse"]],
        toys_text=[two(v) for v in c["toys_spread"]],
        corr_boots_text=signed(c["corr_tau_b_boots"]),
        corr_hesse_text=signed(c["corr_tau_b_hesse"]),
        resample=counts[0].tolist(), resample_distinct=int(c["distinct"][0]),
        resample_twice=int((counts[0] >= 2).sum()), resample_out=int((counts[0] == 0).sum()),
        resample_most=int(counts[0].max()),
        boot_first=boots[0].tolist(),
        short_tau=float(1.0 - c["ratio"][1]),
        within=int((np.abs(boots - E.exact_fits()) <= 0.03).all(1).sum()),
        limitations=[
            "one run per seed: 100 runs with seeds 0-99 on the 40 points, 100 with seeds "
            "100-199 on resamples 1-100 (drawn from default_rng(2)); each answer is where its "
            "run ends after 3,000 steps",
            "spreads are standard deviations over the 100 answers (ddof=1); with 100 runs each "
            "is known to about 7%",
            "HESSE's ellipse is the 2 x 2 block of its covariance for (tau, b), a refitted, in "
            "the quadratic approximation: chi-square on it, a refitted, is %.2f to %.2f above "
            "the minimum, not exactly 1" % tuple(c["profile_on_ellipse"]),
            "the bootstrap's short tau is the bootstrap's, not SGD's: refitting the same 100 "
            "resamples exactly gives %s, %s, %s; 1,000 new made-up datasets (new noise on the "
            "fitted curve), refitted exactly, give %s, %s, %s, HESSE's errors"
            % tuple([two(v) for v in c["exact_spread"]] + [two(v) for v in c["toys_spread"]]),
            "made-up data: the 40 points of examples/minuit_vs_sgd.py (seed 1)"])
    return m


def error_bars():
    m = ensembles()
    seeds, boots = np.asarray(m["seeds"]), np.asarray(m["boots"])
    panels = (
        Panel("Where each run ends", "tau against b: 100 new seeds, 100 resamples", "points",
              dict(xy=np.concatenate([seeds[:, 1:], boots[:, 1:]]))),
        Panel("The spread of the answers", "a, tau, b: new seeds, resamples, HESSE", "bars",
              dict(values=np.array(m["seeds_spread"] + m["boots_spread"] + m["hesse"]),
                   labels=["seeds a", "seeds tau", "seeds b", "resamples a", "resamples tau",
                           "resamples b", "HESSE a", "HESSE tau", "HESSE b"])),
        Panel("The ellipses", "HESSE's chi-square + 1, the resamples' covariance", "curve",
              dict(x=np.arange(len(m["hesse_ellipse"]), dtype=float),
                   ys=np.array([np.asarray(m["hesse_ellipse"])[:, 1],
                                np.asarray(m["boots_ellipse"])[:, 1]]),
                   labels=["HESSE b", "resamples b"])),
        Panel("Resample 1", "how many times each of the 40 points was picked", "bars",
              dict(values=np.array(m["resample"], float),
                   labels=[str(i) for i in range(M.N)])),
    )
    return Study("Error bars on a network",
                 "100 runs on resamples spread by %s, %s, %s; HESSE's errors %s, %s, %s; new "
                 "seeds only %s" % (*m["boots_text"], *m["hesse_text"], m["seeds_round"]),
                 panels, m)


# ====================================================================== 278
AS_SOURCE = ("made-up counts: examples/asimov.py (4 bins, signal 3, 8, 8, 3 on background 30, "
             "20, 12, 8); 10,000 toy experiments a case (seeds 7 and 8), in numpy")
Z_RANGE = (0.0, 11.0)
Z_BINS = 44                          # the toys' Z histogram: bins of 0.25


def num(v):
    """A count as the screen writes it: ``33``, ``9.2``, ``0.8``."""
    v = float(v)
    return "%d" % round(v) if abs(v - round(v)) < 1e-9 else "%.1f" % v


def z_hist(z):
    edges = np.linspace(*Z_RANGE, Z_BINS + 1)
    h, _ = np.histogram(np.clip(z, Z_RANGE[0], Z_RANGE[1] - 1e-9), edges)
    return h, edges


def growth(z, upto):
    """The histogram of the first ``upto`` toys."""
    return z_hist(np.asarray(z)[:upto])[0]


@functools.lru_cache(maxsize=1)
def asimov():
    out = dict(source=AS_SOURCE, bins=A.BINS, toys_n=A.TOYS, z_range=list(Z_RANGE),
               z_bins=Z_BINS, newton=A.NEWTON,
               code_counts=list(code_lines("as-counts")), code_z=list(code_lines("as-z")),
               code_toys=list(code_lines("as-toys")), code_asimov=list(code_lines("as-asimov")),
               code_simple=list(code_lines("as-simple")), code_less=list(code_lines("as-less")))
    for name in A.CASES:
        c = A.case(name)
        p = A.printed(name)
        h, edges = z_hist(c["z"])
        key = "" if name == "b" else "_less"
        out.update({
            "scale" + key: c["scale"], "seed" + key: c["seed"],
            "s" + key: c["s"].tolist(), "b" + key: c["b"].tolist(),
            "expected" + key: c["expected"].tolist(),
            "s_text" + key: [num(v) for v in c["s"]], "b_text" + key: [num(v) for v in c["b"]],
            "expected_text" + key: [num(v) for v in c["expected"]],
            "hist" + key: h.tolist(), "edges": edges.tolist(),
            "z" + key: c["z"].tolist(),
            "median" + key: c["median"], "asimov_z" + key: c["asimov_z"],
            "asimov_mu" + key: c["asimov_mu"], "simple" + key: c["simple"],
            "above" + key: c["above"], "asimov_bins" + key: c["asimov_bins"].tolist(),
            "simple_bins" + key: c["simple_bins"].tolist(),
            "first" + key: c["first"].tolist(), "first_z" + key: c["first_z"],
            "first_mu" + key: c["first_mu"], "zero" + key: c["zero"],
            "converged" + key: c["converged"],
            "median_text" + key: p["median"], "asimov_text" + key: p["asimov"],
            "simple_text" + key: p["simple"],
            "first_z_text" + key: "%.2f" % c["first_z"],
            "first_mu_text" + key: "%.2f" % c["first_mu"]})
    out["growth_steps"] = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    out["toys_text"] = commas(A.TOYS)
    out["limitations"] = [
        "made-up counts: 4 bins, signal (3, 8, 8, 3) on background (30, 20, 12, 8), then the "
        "background divided by 10; no detector, no systematic uncertainties",
        "each toy's Z = sqrt(q0) with the signal strength fitted (never below 0; Newton's "
        "method, 20 steps, all 20,000 fits converged): the large-sample formula, not checked "
        "here against background-only toys (at 6.8 sigma that would need about 10^11 of them)",
        "10,000 toys a case, one seed each (7 and 8); the median is known to about 0.02",
        "s / sqrt(b) is taken bin by bin and added in quadrature, as the Asimov Z is"]
    return out


def asimov_dataset():
    m = asimov()
    panels = (
        Panel("Expected counts", "background and signal per bin", "bars",
              dict(values=np.array(m["b"] + m["s"], float),
                   labels=["b1", "b2", "b3", "b4", "s1", "s2", "s3", "s4"])),
        Panel("10,000 toys' significance", "how many toys in each Z bin of 0.25", "bars",
              dict(values=np.array(m["hist"], float),
                   labels=["%.2f" % v for v in m["edges"][:-1]])),
        Panel("With a tenth of the background", "how many toys in each Z bin of 0.25", "bars",
              dict(values=np.array(m["hist_less"], float),
                   labels=["%.2f" % v for v in m["edges"][:-1]])),
        Panel("Expected significance, three ways", "median of toys, Asimov, s / sqrt(b); "
              "then with a tenth of b", "bars",
              dict(values=np.array([m["median"], m["asimov_z"], m["simple"], m["median_less"],
                                    m["asimov_z_less"], m["simple_less"]]),
                   labels=["median", "Asimov", "s/sqrt(b)", "median b/10", "Asimov b/10",
                           "s/sqrt(b) b/10"])),
    )
    return Study("The Asimov dataset",
                 "Asimov %s against the toys' median %s (s/sqrt(b) %s); with a tenth of the "
                 "background %s against %s (s/sqrt(b) %s)"
                 % (m["asimov_text"], m["median_text"], m["simple_text"], m["asimov_text_less"],
                    m["median_text_less"], m["simple_text_less"]), panels, m)


STUDIES = {"error-bars-on-a-network": error_bars, "asimov-dataset": asimov_dataset}
