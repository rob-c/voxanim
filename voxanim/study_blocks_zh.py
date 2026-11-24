"""The numbers behind animations 275-276 of "Blocks and code" (group ``zh``, "From simulation to
a result"): event weights, and the scale factor that corrects simulation to data. Everything
comes from :mod:`voxanim.study_weights`, the numpy twin of
``examples/weights_and_scale_factors.py``. Made-up events (seeded): one variable x for a signal
(Normal(1.00, 0.10), 50 fb, 20,000 generated) and two backgrounds (Normal(0.60, 0.12), 200 fb,
50,000; Normal(0.75, 0.12), 120 fb, 10,000), luminosity 100 fb^-1; made-up data whose
signal-like objects spread 0.13 instead of 0.10.

* ``event-weights``: the samples' histograms, counted (80,000 events) and weighted
  (w = sigma x L / N: 0.25, 0.4, 1.2; each sample adding up to sigma x L: 5,000, 20,000 and
  12,000, exact); a logistic regression trained on the cross-entropy without and with the
  weights (mean score 0.250 and 0.135: the signal's generated and expected shares); the
  background passing the cut x > 0.915, counted 1.8% and weighted 3.5%; bkg 2 with 2,000 of
  its events at weight -2 and the rest +2: still 12,000, but N_eff = 3,600 of 10,000.
* ``scale-factors``: the cut keeps 80% of simulated signal; in a control region of probes,
  efficiency 0.805 in simulation (20,000) and 0.745 in data (5,000); scale factor 0.925 ±
  0.008; in the signal region the simulation predicts 5,124 events, the data have 4,739;
  with the signal scaled by the scale factor, 4,823 ± 57.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_weights as W
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("made-up events: examples/weights_and_scale_factors.py (a signal and two backgrounds "
          "of one variable x, seeded); recomputed in numpy by voxanim.study_weights")
GRID = 161                       # points along x for the score curves
PROBE_EDGES = np.linspace(0.5, 1.5, 21)    # the control region's chart: 20 bins of 0.05
SIGNAL_EDGES = np.linspace(0.5, 1.5, 21)   # 276's first chart: the simulated signal
LABELS = ("signal", "bkg 1", "bkg 2")


def pct(v, places=1):
    """A share as a percentage: ``1.8%``, ``13.5%``."""
    return "%.*f%%" % (places, 100.0 * float(v))


def three(v):
    """``0.805``, ``0.925``."""
    return "%.3f" % float(v)


def weight_text(v):
    """A weight as the table writes it: ``0.25``, ``0.4``, ``1.2``, ``2``."""
    return ("%.2f" % float(v)).rstrip("0").rstrip(".")


@functools.lru_cache(maxsize=1)
def common():
    """What both show: the samples, their weights and the cut."""
    names = list(W.NAMES)
    weights = [W.weight(k) for k in names]
    expected = [W.expected(k) for k in names]
    c = W.cut()
    return dict(
        source=SOURCE, names=names, L=W.L,
        sigma=[W.SAMPLES[k][0] for k in names], n=[W.SAMPLES[k][1] for k in names],
        shapes=[list(W.SHAPES[k]) for k in names], data_spread=W.DATA_SPREAD,
        weights=weights, weights_text=[weight_text(v) for v in weights],
        expected=expected, expected_text=[commas(v) for v in expected],
        n_text=[commas(v) for v in (W.SAMPLES[k][1] for k in names)],
        sigma_text=["%g" % W.SAMPLES[k][0] for k in names],
        n_total=sum(W.SAMPLES[k][1] for k in names), expected_total=sum(expected),
        cut=c, cut_text="%.3f" % c, working_point=W.WORKING_POINT,
        edges=W.EDGES.tolist())


# ====================================================================== 275
def weights_study():
    m = dict(common())
    h = W.histograms()
    un, we = W.train(False), W.train(True)
    e = W.efficiency()
    ng = W.negative()
    grid = np.linspace(0.0, 1.6, GRID)
    sums = h["weighted"].sum(1)
    worked = "%s × %g ÷ %s = %s" % (m["sigma_text"][0], m["L"], m["n_text"][0],
                                    m["weights_text"][0])
    m.update(
        counts=h["counts"].tolist(), weighted=h["weighted"].tolist(),
        counts_max=float(h["counts"].sum(0).max()), weighted_max=float(h["weighted"].sum(0).max()),
        sums=sums.tolist(), sums_text=[commas(v) for v in sums],
        worked=worked,
        printed_hist="\n".join("%s %d" % (k, round(v)) for k, v in zip(m["names"], sums,
                                                                         strict=True)),
        signal_share_counted=m["n"][0] / m["n_total"],
        signal_share_weighted=m["expected"][0] / m["expected_total"],
        grid=grid.tolist(),
        score_unweighted=W.score(un["a"], un["b"], grid).tolist(),
        score_weighted=W.score(we["a"], we["b"], grid).tolist(),
        unweighted=dict(a=un["a"], b=un["b"], half=un["half"], mean_score=un["mean_score"],
                        share=un["share"], first_loss=float(un["losses"][0]),
                        last_loss=float(un["losses"][-1])),
        trained=dict(a=we["a"], b=we["b"], half=we["half"], mean_score=we["mean_score"],
                     share=we["share"], first_loss=float(we["losses"][0]),
                     last_loss=float(we["losses"][-1])),
        steps=W.STEPS, lr=W.LR,
        mean_unweighted_text=three(un["mean_score"]), mean_weighted_text=three(we["mean_score"]),
        share_unweighted_text=pct(un["share"]), share_weighted_text=pct(we["share"]),
        half_unweighted_text="%.3f" % un["half"], half_weighted_text="%.3f" % we["half"],
        eff=dict(counted=e["counted"], weighted=e["weighted"], each=e["each"],
                 counted_pass=e["counted_pass"], counted_all=e["counted_all"],
                 weighted_pass=e["weighted_pass"], weighted_all=e["weighted_all"]),
        eff_counted_text=pct(e["counted"]), eff_weighted_text=pct(e["weighted"]),
        eff_counted_row="counted: %s ÷ %s = %s" % (commas(e["counted_pass"]),
                                                    commas(e["counted_all"]), pct(e["counted"])),
        eff_weighted_row="weighted: %s ÷ %s = %s" % (commas(e["weighted_pass"]),
                                                      commas(e["weighted_all"]),
                                                      pct(e["weighted"])),
        printed_eff="%s\n%s" % (repr(e["counted"]), repr(e["weighted"])),
        neg=dict(pos=ng["pos"], neg=ng["neg"], weight=ng["weight"], n_eff=ng["n_eff"],
                 n_eff_plain=ng["n_eff_plain"], hist_pos=ng["hist_pos"].tolist(),
                 hist_neg=ng["hist_neg"].tolist(), net=ng["net"].tolist(),
                 err=ng["err"].tolist(), err_plain=ng["err_plain"].tolist(),
                 total=float(ng["w"].sum())),
        n_eff_text=commas(ng["n_eff"]), neg_weight_text=weight_text(ng["weight"]),
        neg_rows=["%s × (+%s), %s × (−%s)" % (commas(ng["pos"]), weight_text(ng["weight"]),
                                               commas(ng["neg"]), weight_text(ng["weight"])),
                  "Σw = %s, as before" % commas(ng["w"].sum()),
                  "N_eff = (Σw)² ÷ Σw² = %s" % commas(ng["n_eff"])],
        err_ratio=float(np.sqrt(W.SAMPLES["bkg 2"][1] / ng["n_eff"])),
        code_samples=list(code_lines("ws-samples")), code_weight=list(code_lines("ws-weight")),
        code_hist=list(code_lines("ws-hist")), code_loss=list(code_lines("ws-loss")),
        code_eff=list(code_lines("ws-eff")), code_negative=list(code_lines("ws-negative")),
        limitations=[
            "made-up events: one variable x for three samples drawn from Gaussians (seed 1); "
            "the cross-sections, luminosity and event counts are made up",
            "the classifier is a logistic regression on x, trained by gradient descent (3,000 "
            "steps, learning rate 10) on the mean cross-entropy, once with the weights and once "
            "without: one run each; at the minimum its weighted mean score equals the weighted "
            "share of signal (measured to 4 decimals)",
            "the negative weights are made up: 2,000 of bkg 2's 10,000 events chosen at random "
            "(seed 3) and given weight -2, the rest +2; a real generator's negative-weight "
            "events need not follow the same distribution",
            "N_eff = (sum w)^2 / sum w^2 is exact arithmetic for these weights"])
    centres = (W.EDGES[:-1] + W.EDGES[1:]) / 2
    panels = (
        Panel("Events made and what they stand for",
              "events generated, and the sum of their weights (sigma x L)", "bars",
              dict(values=np.array(m["n"] + m["sums"], float),
                   labels=["N " + k for k in m["names"]] + ["sum w " + k for k in m["names"]])),
        Panel("The three samples stacked, counted and weighted", "events per bin of x",
              "curve", dict(x=centres, ys=np.array([h["counts"].sum(0), h["weighted"].sum(0)]),
                            labels=["counted", "weighted"])),
        Panel("The score, trained without and with the weights",
              "logistic regression on x; mean score 0.250 and 0.135", "curve",
              dict(x=grid, ys=np.array([m["score_unweighted"], m["score_weighted"]]),
                   labels=["no weights", "weights"])),
        Panel("bkg 2 with negative weights", "events, and their effective number", "bars",
              dict(values=np.array([10000.0, ng["n_eff"]]), labels=["events", "N_eff"])),
    )
    return Study("Event weights",
                 "each simulated event counts for sigma x L / N; weighted, the samples add up "
                 "to %s events; mean score %s with weights, %s without; N_eff %s of 10,000 "
                 "with negative weights" % (commas(m["expected_total"]),
                                            m["mean_weighted_text"], m["mean_unweighted_text"],
                                            m["n_eff_text"]), panels, m)


# ====================================================================== 276
def shares(x, edges):
    """Each bin's share of the events (all of them, including any outside the edges)."""
    return np.histogram(x, edges)[0] / len(x)


def scale_study():
    m = dict(common())
    c = W.control()
    r = W.signal_region()
    d = W.made_up_data()
    sig = W.simulated()["signal"]
    mc_sh = shares(d["probe_mc"], PROBE_EDGES)
    data_sh = shares(d["probe_data"], PROBE_EDGES)
    data_n = np.histogram(d["probe_data"], PROBE_EDGES)[0]
    m.update(
        signal_counts=np.histogram(sig, SIGNAL_EDGES)[0].tolist(),
        signal_edges=SIGNAL_EDGES.tolist(), signal_kept=int((sig > m["cut"]).sum()),
        signal_n=len(sig),
        probe_edges=PROBE_EDGES.tolist(), probe_mc_shares=mc_sh.tolist(),
        probe_data_shares=data_sh.tolist(),
        probe_data_err=(np.sqrt(data_n) / len(d["probe_data"])).tolist(),
        **{k: c[k] for k in ("eff_mc", "eff_data", "n_mc", "n_data", "pass_mc", "pass_data",
                             "err_mc", "err_data", "sf", "sf_err")},
        eff_mc_text=three(c["eff_mc"]), eff_data_text=three(c["eff_data"]),
        sf_text=three(c["sf"]), sf_err_text=three(c["sf_err"]),
        eff_mc_row="%s ÷ %s = %s" % (commas(c["pass_mc"]), commas(c["n_mc"]),
                                     three(c["eff_mc"])),
        eff_data_row="%s ÷ %s = %s" % (commas(c["pass_data"]), commas(c["n_data"]),
                                       three(c["eff_data"])),
        sf_row="%s ÷ %s = %s" % (three(c["eff_data"]), three(c["eff_mc"]), three(c["sf"])),
        sr=dict(signal=r["signal"], background=r["background"], each=r["each"],
                predicted=r["predicted"], corrected=r["corrected"], observed=r["observed"],
                predicted_err=r["predicted_err"], corrected_err=r["corrected_err"],
                data_err=r["data_err"], sf_part=r["sf_part"], pull_before=r["pull_before"],
                pull_after=r["pull_after"]),
        signal_scaled=r["signal"] * c["sf"],
        predicted_text=commas(r["predicted"]), corrected_text=commas(r["corrected"]),
        observed_text=commas(r["observed"]), data_err_text=commas(r["data_err"]),
        corrected_err_text=commas(r["corrected_err"]),
        predicted_err_text=commas(r["predicted_err"]),
        signal_text=commas(r["signal"]), signal_scaled_text=commas(r["signal"] * c["sf"]),
        background_text=commas(r["background"]),
        excess_text=commas(r["predicted"] - r["observed"]),
        pull_before_text="%.1f" % r["pull_before"], pull_after_text="%.1f" % r["pull_after"],
        code_cut=list(code_lines("ws-cut")), code_probes=list(code_lines("ws-probes")),
        code_eff_cr=list(code_lines("ws-eff-cr")), code_sf=list(code_lines("ws-sf")),
        code_correct=list(code_lines("ws-correct")),
        limitations=[
            "made-up data: the only difference from the simulation is the signal-like "
            "objects' spread, 0.13 against 0.10; the backgrounds are the same in both",
            "the control region holds only signal-like probes (a real tag and probe leaves "
            "some background, to be subtracted); 20,000 simulated and 5,000 in data (seed 2)",
            "the scale factor's error is statistical only (binomial errors on the two "
            "efficiencies); the corrected prediction's error adds the simulation's own "
            "statistics, sqrt(sum w^2)",
            "a scale factor corrects how many pass the cut, not the shape: above x = 1.2 the "
            "wider data still have more events than the corrected simulation",
            "one draw of the pseudo-data (seed 2)"])
    pc = (PROBE_EDGES[:-1] + PROBE_EDGES[1:]) / 2
    panels = (
        Panel("The control region's probes", "share per bin of x: simulation and data",
              "curve", dict(x=pc, ys=np.array([mc_sh, data_sh]),
                            labels=["simulation", "data"])),
        Panel("Efficiency of the cut, and the scale factor", "simulation, data, their ratio",
              "bars", dict(values=np.array([c["eff_mc"], c["eff_data"], c["sf"]]),
                           labels=["eff MC", "eff data", "scale factor"])),
        Panel("The signal region", "events above the cut", "bars",
              dict(values=np.array([r["predicted"], r["corrected"], r["observed"]], float),
                   labels=["simulation", "corrected", "data"])),
        Panel("The simulated signal and the cut", "events per bin of x", "curve",
              dict(x=(SIGNAL_EDGES[:-1] + SIGNAL_EDGES[1:]) / 2,
                   ys=np.array([m["signal_counts"]], float), labels=["signal"])),
    )
    return Study("Simulation is not data: scale factors",
                 "efficiency %s in simulation, %s in data: scale factor %s ± %s; signal region "
                 "%s predicted, %s corrected, %s observed"
                 % (m["eff_mc_text"], m["eff_data_text"], m["sf_text"], m["sf_err_text"],
                    m["predicted_text"], m["corrected_text"], m["observed_text"]), panels, m)


STUDIES = {"event-weights": weights_study, "scale-factors": scale_study}
