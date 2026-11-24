"""The numbers behind animations 273-274 of "Blocks and code" (group ``zg``, "The classifier and
the likelihood ratio"). Everything comes from :mod:`voxanim.study_likelihood_ratio`, the numpy
twin of ``examples/likelihood_ratio.py`` (made-up events with the script's seeds; one training
run per animation, one seed; ``tests/test_likelihood_ratio_equivalence.py`` matches it to torch).

* ``classifier-learns-the-likelihood-ratio``: H0 (x Gaussian at 0, spread 1) and H1 (at 1,
  spread 0.5), 20,000 made-up events of each. The exact ratio p1 / p0 peaks at x = 4/3 (3.9)
  and falls on both sides. A 1-32-32-1 network trained 10,000 steps with binary cross-entropy:
  its loss on the training events falls from 0.682 to 0.4691, the exact ratio's own on the
  same events 0.4692. Its s / (1 - s) is within 10% of the exact ratio for x from -0.75 to
  2.25, where both hypotheses have events, and drifts beyond: at x = 3 (20 H0 and 2 H1 events
  past it) 0.24 against 0.060. At a 5% false-alarm rate it keeps 19.6% of fresh H1 events
  (±0.3, the scatter of 20,000); the best test (Neyman-Pearson, the exact ratio: a window of x
  round the peak) keeps 19.3%, a single cut x > 1.64 9.9%; the network's own window, computed
  exactly, 19.0%.
* ``reweighting-with-a-classifier``: made-up "simulation" (x, y Gaussian at 0, spread 1) and
  "data" (x shifted 0.4, y spread 1.25, correlation 0.4), 50,000 of each; the same kind of
  network (2 inputs) trained 10,000 steps to tell them apart; each event of a fresh simulated
  sample weighted s / (1 - s). Histograms of fresh data against the simulation before and after
  weighting, in x, y and x + y (never a column given to the network); where it fails: data
  events in bins where the simulation has none, the largest weights against the exact ones,
  the effective number of events, and a network trained on 1,000 events of each.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_likelihood_ratio as L
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE_ONE = ("made-up events: examples/likelihood_ratio.py, part one (H0: x Gaussian at 0, "
              "spread 1; H1: at 1, spread 0.5; 20,000 of each, seed 273); the network trained "
              "once in numpy (10,000 steps of 512, Adam, lr 0.0003)")
SOURCE_TWO = ("made-up events: examples/likelihood_ratio.py, part two ('simulation': x, y "
              "Gaussian at 0, spread 1; 'data': x at 0.4, y spread 1.25, correlation 0.4; "
              "50,000 of each, seed 274); the network trained once in numpy (10,000 steps of "
              "512, Adam, lr 0.0003)")
RATIO_X = (-2.0, 4.0)        # 273: the x range of the histograms and the ratio chart
RATIO_Y = (1e-4, 10.0)       # 273: the ratio chart's log scale
ROC_A = 0.3                  # 273: the ROC chart's false-alarm axis, 0 to this
SCATTER = 400                # 274: weights drawn: every one with an exact weight above 5,
BIG_W = 5.0                  # and this many others, the first of the fresh sample
SHOWN_EVENTS = 150           # 274: simulated events dotted on the plane
MIDDLE = 1000                # 274: "where most events are": bins with at least this much data


def pct(v, places=1):
    """``19.3%``."""
    return "%.*f%%" % (places, 100.0 * float(v))


def num(v):
    """A ratio or a weight in two figures: ``0.24``, ``0.060``, ``3.9``, ``151``, ``38``."""
    v = float(v)
    if v >= 100:
        return "%.0f" % v
    if v >= 10:
        return "%.0f" % v
    s = "%.2g" % v
    if "e" in s:
        s = "%.1e" % v
    if 0.01 <= v < 0.1 and len(s.replace("0.", "").lstrip("0")) == 1:
        s += "0"
    return s


def counts(v):
    return commas(v)


# ====================================================================== 273
@functools.lru_cache(maxsize=1)
def ratio_study():
    o = L.one()
    m = dict(source=SOURCE_ONE, h0=list(L.H0), h1=list(L.H1), n=L.N_ONE, steps=L.STEPS_ONE,
             batch=L.BATCH, lr=L.LR, hidden=L.HIDDEN)
    grid = np.asarray(o["grid"])
    keep = (grid >= RATIO_X[0] - 1e-9) & (grid <= RATIO_X[1] + 1e-9)
    px, pv = o["peak"]
    m.update(
        grid=grid[keep].tolist(), learned=np.exp(o["learned"][keep]).tolist(),
        exact=np.exp(o["exact"][keep]).tolist(), off=o["off"][keep].tolist(),
        show=o["show"].tolist(), h0=o["h0"].tolist(), h1=o["h1"].tolist(),
        peak_x=px, peak=pv, peak_text="%.1f" % pv, peak_x_text="%.2f" % px,
        track=list(o["track"]), edge=list(o["edge"]),
        worst_inside=o["worst_inside"], worst_pct="%.0f%%" % (100 * (np.exp(o["worst_inside"])
                                                                   - 1)),
        within_text="10%",
        far_x=o["far_x"], far_learned=o["far_learned"], far_exact=o["far_exact"],
        far_learned_text=num(o["far_learned"]), far_exact_text=num(o["far_exact"]),
        far_times="%.0f" % (o["far_learned"] / o["far_exact"]),
        above0=o["above0"], above1=o["above1"], below0=o["below0"], below1=o["below1"],
        beyond_far0=o["beyond_far0"], beyond_far1=o["beyond_far1"],
        curve=o["curve"].tolist(), curve_steps=o["curve_steps"].tolist(),
        first_loss=o["first_loss"], last_loss=o["last_loss"], floor_on=o["floor_on"],
        floor=o["floor"],
        first_text="%.3f" % o["first_loss"], last_text="%.4f" % o["last_loss"],
        floor_text="%.4f" % o["floor_on"], floor3="%.3f" % o["floor_on"],
        alphas=o["alphas"].tolist(), roc_np=o["roc_np"].tolist(), roc_cut=o["roc_cut"].tolist(),
        roc_net=o["roc_net"].tolist(), roc_own=o["roc_own"].tolist(),
        np_window=list(o["np_window"]), np_power=o["np_power"], cut=o["cut"],
        cut_power=o["cut_power"], net_power=o["net_power"], net_power_err=o["net_power_err"],
        own_power=o["own_power"], own_window=list(o["own_window"]),
        np_text=pct(o["np_power"]), cut_text=pct(o["cut_power"]), net_text=pct(o["net_power"]),
        own_text=pct(o["own_power"]), err_text="%.1f" % (100 * o["net_power_err"]),
        cut_x_text="%.2f" % o["cut"],
        window_text="%.2f to %.2f" % tuple(o["np_window"]),
        printed_power="%.3f" % o["net_power"],
        code_hyp=list(code_lines("lr-hypotheses")), code_train=list(code_lines("lr-train")),
        code_ratio=list(code_lines("lr-ratio")), code_power=list(code_lines("lr-power")),
        limitations=[
            "made-up events: two Gaussian hypotheses chosen so the exact ratio is known; real "
            "analyses do not know it, which is why the network is trained",
            "one training run, one seed (the network's starting weights and its batches); "
            "another run differs by a few percent where events are and more where they are not",
            "the power on fresh events has a scatter of about ±0.3 points (20,000 events); "
            "the network's own window, computed exactly, keeps 19.0%",
            "the ratio is learned where both hypotheses have events; past them the network's "
            "output is whatever its last layers extrapolate to"])
    panels = (
        Panel("Two made-up hypotheses", "x for 20,000 events of each, bins of 0.25", "curve",
              dict(x=np.asarray(m["show"][:-1]), ys=np.array([m["h0"], m["h1"]], float),
                   labels=["H0", "H1"])),
        Panel("Training loss", "binary cross-entropy on all 40,000 training events", "curve",
              dict(x=np.asarray(m["curve_steps"], float), ys=np.array([m["curve"]]),
                   labels=["loss"])),
        Panel("The ratio", "s / (1 - s) of the network against p1 / p0, exact", "curve",
              dict(x=np.asarray(m["grid"]), ys=np.log10(np.array([m["learned"], m["exact"]])),
                   labels=["network (log10)", "exact (log10)"])),
        Panel("ROC", "H1 kept against H0 kept: the network on fresh events, the exact best "
              "test, one cut", "curve",
              dict(x=np.asarray(m["alphas"]), ys=np.array([m["roc_net"], m["roc_np"],
                                                           m["roc_cut"]]),
                   labels=["network", "likelihood ratio", "x > c"])),
    )
    return Study("A classifier learns the likelihood ratio",
                 "s / (1 - s) within 10%% of the exact ratio where events are; at 5%% false "
                 "alarms it keeps %s, the best test %s" % (m["net_text"], m["np_text"]),
                 panels, m)


def classifier_learns():
    return ratio_study()


# ====================================================================== 274
@functools.lru_cache(maxsize=1)
def reweight_study():
    t = L.two()
    r = L.run_two()
    _sim, _data, new, _seen = L.samples()
    m = dict(source=SOURCE_TWO, n=L.N_TWO, steps=L.STEPS_TWO, few=L.FEW,
             data_mean=list(L.DATA_MEAN), data_cov=[list(c) for c in L.DATA_COV],
             bins=L.BINS.tolist(), names=list(L.VARIABLES))
    hists = {}
    for name, h in t["hists"].items():
        hists[name] = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in h.items()}
    w, we = np.asarray(t["weights"]), np.asarray(t["exact_weights"])
    big = np.nonzero(we > BIG_W)[0]
    rest = np.nonzero(we <= BIG_W)[0][:SCATTER]
    pick = np.r_[rest, big]
    curve = np.asarray(t["curve"])
    empty_y = hists["y"]["empty"]
    empty_bins = [i for i, (d, a) in enumerate(zip(hists["y"]["data"], hists["y"]["sim"],
                                                   strict=True)) if a == 0 and d > 0]
    for h in hists.values():
        d, b, f = (np.asarray(h[k], float) for k in ("data", "weighted", "few"))
        mid = d >= MIDDLE
        h["off_middle"] = float(np.abs(b[mid] / d[mid] - 1).max())
        h["off_middle_few"] = float(np.abs(f[mid] / d[mid] - 1).max())
        h["off_before"] = float(np.abs(np.asarray(h["sim"])[mid] / d[mid] - 1).max())
        h["off_middle_text"] = "%.0f%%" % (100 * h["off_middle"])
        h["off_before_text"] = "%.0f%%" % (100 * h["off_before"])
        h["middle_share"] = float(d[mid].sum() / d.sum())
    m.update(
        hists=hists, plane=L.PLANE.tolist(), plane_w=np.exp(r["plane_logit"]).tolist(),
        plane_exact=np.exp(L.log_weight(np.c_[np.meshgrid(L.PLANE, L.PLANE)[0].ravel(),
                                               np.meshgrid(L.PLANE, L.PLANE)[1].ravel()]))
        .reshape(len(L.PLANE), len(L.PLANE)).tolist(),
        events=new[:SHOWN_EVENTS].tolist(), event_w=w[:SHOWN_EVENTS].tolist(),
        scatter_w=w[pick].tolist(), scatter_exact=we[pick].tolist(),
        curve=curve.tolist(), first_loss=float(curve[0]), last_loss=float(curve[-1]),
        first_text="%.3f" % curve[0], last_text="%.3f" % curve[-1],
        w_mean=t["w_mean"], w_mean_exact=t["w_mean_exact"], w_sum=float(w.sum()),
        w_max=t["w_max"], w_max_exact=t["w_max_exact"],
        w_max_text=num(t["w_max"]), w_max_exact_text=num(t["w_max_exact"]),
        n_eff=t["n_eff"], n_eff_exact=t["n_eff_exact"], n_eff_few=t["n_eff_few"],
        n_eff_text=commas(round(t["n_eff"], -2)), n_eff_exact_text=commas(round(t["n_eff_exact"],
                                                                              -2)),
        n_text=commas(L.N_TWO),
        rms_bulk=t["rms_bulk"], rms_bulk_few=t["rms_bulk_few"], bulk_share=t["bulk_share"],
        bulk_pct="%.0f%%" % (100 * (np.exp(t["rms_bulk"]) - 1)),
        few_pct="%.0f%%" % (100 * (np.exp(t["rms_bulk_few"]) - 1)),
        top1=t["top1"], empty_y=empty_y, empty_bins=empty_bins,
        big_count=int((we > 10).sum()),
        big_ratio=float(np.median(w[we > 10] / we[we > 10])),
        chi2={k: dict(before=h["chi2_before"][0], after=h["chi2_after"][0],
                      exact=h["chi2_exact"][0], few=h["chi2_few"][0], bins=h["chi2_after"][1])
              for k, h in hists.items()},
        code_samples=list(code_lines("lr-samples")), code_judge=list(code_lines("lr-judge")),
        code_weights=list(code_lines("lr-weights")), code_check=list(code_lines("lr-check")),
        code_sum=list(code_lines("lr-sum")), code_neff=list(code_lines("lr-neff")),
        limitations=[
            "made-up events: the 'simulation' and the 'data' are both drawn from stated "
            "Gaussians, so the exact weight is known to check against",
            "one training run, one seed; the comparison uses fresh samples of each, not the "
            "events the network was trained on",
            "x + y matches because in these made-up samples it depends on x and y alone; a "
            "variable the simulation gets wrong for given x and y would not be fixed",
            "the effective number of events is (sum w)^2 / sum w^2; the exact weights, larger "
            "in the tails, would leave fewer, %s" % commas(round(t["n_eff_exact"], -2))])
    ch = m["chi2"]
    panels = (
        Panel("x: data, simulation, weighted", "fresh samples, 20 bins of 0.5", "curve",
              dict(x=L.BINS[:-1], ys=np.array([hists["x"]["data"], hists["x"]["sim"],
                                               hists["x"]["weighted"]]),
                   labels=["data", "simulation", "weighted"])),
        Panel("y: data, simulation, weighted", "fresh samples, 20 bins of 0.5", "curve",
              dict(x=L.BINS[:-1], ys=np.array([hists["y"]["data"], hists["y"]["sim"],
                                               hists["y"]["weighted"]]),
                   labels=["data", "simulation", "weighted"])),
        Panel("x + y: never given to the network", "fresh samples, 20 bins of 0.5", "curve",
              dict(x=L.BINS[:-1], ys=np.array([hists["x + y"]["data"], hists["x + y"]["sim"],
                                               hists["x + y"]["weighted"]]),
                   labels=["data", "simulation", "weighted"])),
        Panel("Weights: learned against exact", "every event with an exact weight above 5, "
              "and %d others" % SCATTER, "points",
              dict(xy=np.c_[np.log10(we[pick]), np.log10(w[pick])])),
    )
    return Study("Reweighting simulation with a classifier",
                 "chi-square of x + y against the data, 20 bins: %s before weighting, %s after"
                 % (commas(ch["x + y"]["before"]), commas(ch["x + y"]["after"])), panels, m)


def reweighting():
    return reweight_study()


STUDIES = {"classifier-learns-the-likelihood-ratio": classifier_learns,
           "reweighting-with-a-classifier": reweighting}
