"""The numbers behind animation 277 of "Blocks and code" (group ``zi``, "From simulation to a
result"): unfolding, a detector's smearing run backwards. Everything comes from
:mod:`voxanim.study_unfolding`, the numpy twin of ``examples/unfolding.py`` (made-up events,
seed 1); this module picks what is shown, formats it, and checks each claim the drawing makes
with a second, plain computation.

``unfolding`` ("Unfolding: running the smearing backwards"):

* the truth: 10,000 made-up events in 10 bins of 10 GeV, falling with a bump at 60 GeV
  (2,519 in the first bin, 1,194 in the bump's);
* R from simulation: 1,000,000 simulated events (964,628 below 100 GeV; 582,195 found and
  measured in range), counted by (measured bin, true bin) and each column divided by the
  events simulated in its true bin; the first :data:`DOTS` found-and-kept pairs drawn as dots;
* measured: R @ truth = 6,345 expected; one Poisson draw, 6,362; the noise about 25 a bin;
* inverting: the expected counts give the truth back (3.6e-12 apart); the data swing from
  -900 to 3,475, the truth's largest bin being 2,519;
* why: R's singular values 0.75 down to 0.0078 (largest / smallest 96); this draw's noise
  along each of U's 10 directions (4 to 73 events; about 25 expected) divided by them: up to
  2,759, the last three making nearly all the swing;
* iterative Bayesian unfolding (D'Agostini) from the simulation's spectrum (no bump): the
  iterates 0 to 10 on this draw;
* 2,000 pseudo-experiments: bias (root mean square over the bins) 423 at 0 iterations, 82 at
  4, 41 at 10, 24 at 20; spread 0, 38, 53, 69; their combination (the error) smallest at 10
  (66.8, within 1 of it from 8 to 13); the inversion's spread 1,166 (1,180 worked out), 47
  times the data's 25.
"""

from __future__ import annotations

import numpy as np

from . import study_unfolding as S
from .blocks import code_lines, commas
from .study import Panel, Study

ON = "made-up data"
SOURCE = ("made-up events: examples/unfolding.py (10,000 true events, falling with a bump at "
          "60 GeV; a detector finding 60-90% and smearing by 10 GeV; R from 1,000,000 "
          "simulated events; seed 1), run in numpy by voxanim.study_unfolding")
DOTS = 500                  # simulated pairs drawn as dots
SHOWN = 10                  # iterations shown one by one
CHART_ITERATIONS = 30       # the bias / spread chart's x range


def signed(v):
    """``−900``, ``3,475``: a count with a true minus sign."""
    v = float(v)
    return ("−" if round(v) < 0 else "") + commas(abs(v))


def sci(v):
    """``3.6e-12`` as the screen writes it: ``3.6 × 10⁻¹²``."""
    from .losschart import pow10
    return pow10(v, 1)


def unfolding_study():
    w, toy = S.world(), S.toys()
    sm = S.summary()
    e, truth, R = w["edges"], w["truth"], w["R"]
    t, r, found = w["t"], w["r"], w["found"]
    kept = found & (r >= S.LO) & (r < S.HI)
    pick = np.nonzero(kept)[0][:DOTS]
    # the claims, checked another way
    H2 = np.zeros((S.BINS, S.BINS))
    bi = np.clip(np.floor(r[kept] / 10).astype(int), 0, 9)
    bj = np.clip(np.floor(t[kept] / 10).astype(int), 0, 9)
    np.add.at(H2, (bi, bj), 1)
    if not np.array_equal(H2, w["H"]):
        raise ValueError("H is not the count of (measured bin, true bin)")
    if not np.allclose(R.sum(0), [np.mean(found[(t >= a) & (t < a + 10)] & (r[(t >= a) & (
            t < a + 10)] >= 0) & (r[(t >= a) & (t < a + 10)] < 100)) for a in e[:-1]]):
        raise ValueError("R's columns do not add up to the chance of being found and kept")
    if sm["back_off"] > 1e-9:
        raise ValueError("the expected counts do not invert to the truth")
    U, s, Vt = w["U"], w["s"], w["Vt"]
    if not np.allclose(w["naive"] - truth, Vt.T @ w["blown"]):
        raise ValueError("the inversion's swing is not the noise divided by s")
    last3 = Vt[-3:].T @ w["blown"][-3:]
    share3 = float(np.sum(last3 ** 2) / np.sum((w["naive"] - truth) ** 2))
    its = S.iterates(w["data"], w["prior"], R, w["eff"], S.UPTO)
    if not np.allclose(its[4], S.bayes(w["data"], w["prior"], R, w["eff"], 4)):
        raise ValueError("the iterates are not the script's bayes")
    if not np.allclose(toy["mean_squared"], toy["error_rms"] ** 2):
        raise ValueError("the error is not sqrt(bias^2 + spread^2)")
    best = sm["best"]
    near = [k for k in range(S.UPTO + 1) if toy["error_rms"][k] <= toy["error_rms"][best] + 1.0]
    ratio = toy["inverse_spread_rms"] / sm["data_noise"]
    lost = 1.0 - sm["expected_total"] / sm["truth_total"]
    nsize = S.noise_size()
    ks = (0, 4, best, 20)
    m = dict(
        source=SOURCE, seed=S.SEED, bins=S.BINS, edges=e, centres=(e[:-1] + e[1:]) / 2,
        n_true=S.N, simulated=S.SIMULATED, toys=S.TOYS, resolution=S.RESOLUTION,
        truth=truth, truth_text=[commas(v) for v in truth], truth_max=float(truth.max()),
        bump_found=float(R[:, 5].sum() * truth[5]),
        bump_found_text=commas(R[:, 5].sum() * truth[5]),
        truth_total_text=commas(truth.sum()), bump_bin=int(np.argmax(truth[3:]) + 3),
        sim_true=t[pick], sim_measured=r[pick], dots=len(pick),
        simulated_text=commas(S.SIMULATED), in_range=sm["simulated_in_range"],
        in_range_text=commas(sm["simulated_in_range"]), counted=sm["counted"],
        counted_text=commas(sm["counted"]), found=sm["found"],
        H=w["H"], n=w["n"], R=R, eff=w["eff"], eff_text=["%.2f" % v for v in w["eff"]],
        eff_range_text="%.0f%% to %.0f%%" % (100 * w["eff"].min(), 100 * w["eff"].max()),
        R_max=float(R.max()), R_diag_text=["%.2f" % v for v in np.diag(R)],
        expected=w["expected"], expected_text=[commas(v) for v in w["expected"]],
        expected_total=sm["expected_total"], expected_total_text=commas(sm["expected_total"]),
        data=w["data"], data_total=sm["data_total"], data_total_text=commas(sm["data_total"]),
        lost=lost, lost_text="%.0f%%" % (100 * lost), data_noise=sm["data_noise"],
        data_noise_text="%.0f" % sm["data_noise"],
        back=w["back"], back_off=sm["back_off"], back_off_text=sci(sm["back_off"]),
        naive=w["naive"], naive_text=[signed(v) for v in w["naive"]],
        naive_min_text=signed(w["naive"].min()), naive_max_text=signed(w["naive"].max()),
        naive_negative=int((w["naive"] < 0).sum()),
        s=s, s_text=["%.2g" % v if v < 0.1 else "%.2f" % v for v in s],
        condition=sm["condition"], condition_text="%.0f" % sm["condition"],
        inv_s=1.0 / s, inv_s_text=["×%.0f" % (1.0 / v) for v in s],
        noise=w["noise"], noise_text=[signed(v) for v in w["noise"]], noise_size=nsize,
        noise_size_text="%.0f" % float(np.mean(nsize)),
        blown=w["blown"], blown_text=[signed(v) for v in w["blown"]],
        last3_share=share3, last3_share_text="%.1f%%" % (100 * share3),
        U=U, Vt=Vt, prior=w["prior"], prior_text=[commas(v) for v in w["prior"]],
        iterates=its[:SHOWN + 1], iterate_text=[[commas(v) for v in row]
                                                for row in its[:SHOWN + 1]],
        bias_rms=toy["bias_rms"], spread_rms=toy["spread_rms"], error_rms=toy["error_rms"],
        bias_text=["%.0f" % v for v in toy["bias_rms"]],
        spread_text=["%.0f" % v for v in toy["spread_rms"]],
        error_text=["%.0f" % v for v in toy["error_rms"]],
        best=best, near_best=[min(near), max(near)], shown_ks=list(ks),
        inverse_spread=toy["inverse_spread_rms"],
        inverse_spread_text=commas(toy["inverse_spread_rms"]),
        inverse_spread_exact=sm["inverse_spread_exact"],
        inverse_spread_exact_text=commas(sm["inverse_spread_exact"]),
        inverse_bias=toy["inverse_bias_rms"], inverse_negative=toy["negative"],
        inverse_negative_text="%.0f%%" % (100 * toy["negative"]),
        ratio=ratio, ratio_text="%.0f" % ratio,
        upto=S.UPTO, chart_iterations=CHART_ITERATIONS, printed=S.printed(),
        code={n: list(code_lines(n)) for n in ("uf-truth", "uf-simulate", "uf-response",
                                                "uf-measure", "uf-invert", "uf-svd",
                                                "uf-prior", "uf-bayes", "uf-toys")},
        limitations=[
            "made-up events throughout: the spectrum, the detector (60-90% found, 10 GeV "
            "smearing) and the simulation are invented, with one seed (1); the data are one "
            "Poisson draw",
            "the data are drawn from R @ truth with the same R the unfolding uses: the "
            "simulation is taken to be the detector exactly; in a real analysis R has its own "
            "errors (the simulation is not the data: last time's scale factors), and they "
            "add to the unfolded errors",
            "R is square (10 measured bins, 10 true), so it can be inverted at all; real "
            "unfoldings often use more measured bins than true ones",
            "bias and spread are root mean squares over the 10 bins, in events, measured "
            "over 2,000 pseudo-experiments from the same truth; the best number of "
            "iterations (10 here) depends on the spectrum, the smearing and the number of "
            "events, and an analysis cannot measure it on its own data, since it does not "
            "know the truth",
            "the inversion's bias is 0 exactly (it is linear); the pseudo-experiments' "
            "average lands %.0f events off, by chance (its own error is about 26)"
            % toy["inverse_bias_rms"],
            "iterative Bayesian unfolding only; Tikhonov (SVD) regularisation trades bias "
            "for spread the same way, with a strength in place of the iteration count"])
    panels = (
        Panel("The response matrix R", "built from 964,628 simulated events: row = measured "
              "bin, column = true bin, each column divided by the events simulated in it",
              "matrix", dict(values=R)),
        Panel("Truth, measured, inverted", "counts per 10 GeV bin: the made-up truth, R @ "
              "truth, one Poisson draw of the data, and the data with R inverted", "curve",
              dict(x=m["centres"], ys=np.array([truth, w["expected"], w["data"],
                                                w["naive"]]),
                   labels=["truth", "expected", "data", "R inverted"])),
        Panel("R's singular values", "0.75 down to 0.0078: the noise along the last "
              "directions comes back divided by them", "bars",
              dict(values=s, labels=["s%d" % (k + 1) for k in range(S.BINS)])),
        Panel("Bias and spread over 2,000 pseudo-experiments", "root mean square over the "
              "10 bins, events, against the number of D'Agostini iterations", "curve",
              dict(x=np.arange(S.UPTO + 1, dtype=float),
                   ys=np.array([toy["bias_rms"], toy["spread_rms"], toy["error_rms"]]),
                   labels=["bias", "spread", "error"])),
    )
    return Study("Unfolding: running the smearing backwards",
                 "Inverting R turns noise of %s a bin into swings of %s; %d D'Agostini "
                 "iterations: bias %s, spread %s" % (m["data_noise_text"],
                                                     m["inverse_spread_text"], best,
                                                     m["bias_text"][best],
                                                     m["spread_text"][best]), panels, m)


STUDIES = {"unfolding": unfolding_study}
