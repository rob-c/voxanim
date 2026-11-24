"""Event weights and scale factors, for the two "From simulation to a result" animations of
"Blocks and code" (catalogue 275-276): ``examples/weights_and_scale_factors.py`` in numpy, the
same recipe with the same seeds (``tests/test_study_weights.py`` runs the script and checks
every array against this twin).

All made up. One variable x for three simulated samples, a luminosity L = 100 fb^-1:

========  =================  =============  ================
sample    x                  cross-section  events generated
========  =================  =============  ================
signal    Normal(1.00, 0.10)  50 fb         20,000
bkg 1     Normal(0.60, 0.12) 200 fb         50,000
bkg 2     Normal(0.75, 0.12) 120 fb         10,000
========  =================  =============  ================

**Event weights** (:func:`weights`): each event counts for sigma x L / N, so a sample's
weights add up to sigma x L, its expected number of events (exact). :func:`histograms`:
counts and weighted counts in 16 bins of x over 0..1.6 (no event falls outside).
:func:`train`: a logistic regression on x trained by gradient descent on the mean
cross-entropy, each event's times its weight (or not); at its minimum the weighted mean score
equals the weighted share of signal (the bias's gradient is zero), so trained without weights
it learns the generated mix (25% signal), with them the expected one (5,000 / 37,000).
:func:`efficiency`: the share of background passing the cut, counted and weighted.
:func:`negative`: bkg 2 as a generator with negative weights would give it, 2,000 of its
10,000 events counting -2, the rest +2 (the weights still add up to 12,000);
N_eff = (sum w)^2 / sum w^2 = (8,000 - 2,000)^2 / 10,000 = 3,600.

**Scale factors** (:func:`control`, :func:`signal_region`): the cut keeps 80% of the
simulated signal (x > its 20th percentile). Made-up data measure signal-like objects with a
spread of 0.13, not 0.10. In a control region of probes known to be signal-like (20,000
simulated, 5,000 in data) the cut's efficiency is counted in each; the scale factor is
eff_data / eff_mc with its binomial statistical error. The signal region (x above the cut):
the simulation's weighted prediction, the same with its signal scaled by the scale factor,
and the pseudo-data's count (each process's expected count Poisson-fluctuated, seed 2).
"""

from __future__ import annotations

import functools

import numpy as np

L = 100.0                                   # luminosity, fb^-1
SAMPLES = {"signal": (50.0, 20000), "bkg 1": (200.0, 50000), "bkg 2": (120.0, 10000)}
SHAPES = {"signal": (1.00, 0.10), "bkg 1": (0.60, 0.12), "bkg 2": (0.75, 0.12)}
NAMES = tuple(SAMPLES)
DATA_SPREAD = 0.13                          # the data's signal-like objects: wider
EDGES = np.linspace(0.0, 1.6, 17)           # 16 bins of x
STEPS, LR = 3000, 10.0                      # the logistic regression's training
WORKING_POINT = 0.8                         # the cut keeps this share of simulated signal
NEGATIVE = 2000                             # bkg 2's events that count negative
SEED, NEG_SEED, DATA_SEED = 1, 3, 2
N_MC, N_DATA = 20000, 5000                  # probes in the control region


@functools.lru_cache(maxsize=1)
def simulated():
    """``{name: x}``: the script's three simulated samples."""
    rng = np.random.default_rng(SEED)
    return {name: rng.normal(*SHAPES[name], n) for name, (_, n) in SAMPLES.items()}


@functools.lru_cache(maxsize=1)
def weights():
    """``{name: w}``: every event's weight, cross-section x L / events generated."""
    return {name: np.full(n, sigma * L / n) for name, (sigma, n) in SAMPLES.items()}


def weight(name):
    sigma, n = SAMPLES[name]
    return sigma * L / n


def expected(name):
    """cross-section x L: the events a sample stands for."""
    return SAMPLES[name][0] * L


@functools.lru_cache(maxsize=1)
def histograms():
    """``dict(counts, weighted)``: ``[3, 16]`` arrays, one row per sample in :data:`NAMES`."""
    x, w = simulated(), weights()
    counts = np.array([np.histogram(x[k], EDGES)[0] for k in NAMES], float)
    weighted = np.array([np.histogram(x[k], EDGES, weights=w[k])[0] for k in NAMES])
    return dict(counts=counts, weighted=weighted)


def events(weighted):
    """All the simulated events together: ``(x, label, weight)``; weights all 1 if not
    ``weighted``."""
    x, w = simulated(), weights()
    xs = np.concatenate([x[k] for k in NAMES])
    ys = np.concatenate([np.full(len(x[k]), float(k == "signal")) for k in NAMES])
    ws = np.concatenate([w[k] for k in NAMES])
    return xs, ys, ws if weighted else np.ones_like(ws)


def score(a, b, x):
    return 1 / (1 + np.exp(-(a * x + b)))


@functools.lru_cache(maxsize=2)
def train(weighted, steps=STEPS, lr=LR):
    """The script's ``train``: ``dict(a, b, losses, first)`` with ``first`` the loss at the
    start (every score 0.5: ln 2) and the weighted mean score and signal share."""
    x, y, w = events(weighted)
    a, b, losses = 0.0, 0.0, []
    for _ in range(steps):
        s = 1 / (1 + np.exp(-(a * x + b)))
        ce = -(y * np.log(s) + (1 - y) * np.log(1 - s))
        loss = (w * ce).sum() / w.sum()
        g = w * (s - y) / w.sum()
        a -= lr * (g * x).sum()
        b -= lr * g.sum()
        losses.append(loss)
    s = score(a, b, x)
    return dict(a=float(a), b=float(b), losses=np.array(losses), mean_score=float(
        (w * s).sum() / w.sum()), share=float((w * y).sum() / w.sum()), half=float(-b / a))


def per_event(a, b, i):
    """One event's numbers under the score ``(a, b)``: ``dict(x, y, s, ce, w)``, ``i`` an
    index into :func:`events`."""
    x, y, w = events(True)
    s = float(score(a, b, x[i]))
    ce = -(y[i] * np.log(s) + (1 - y[i]) * np.log(1 - s))
    return dict(x=float(x[i]), y=float(y[i]), s=s, ce=float(ce), w=float(w[i]))


@functools.lru_cache(maxsize=1)
def cut():
    """The cut on x: it keeps :data:`WORKING_POINT` of the simulated signal."""
    return float(np.quantile(simulated()["signal"], 1 - WORKING_POINT))


def efficiency():
    """The share of each sample passing the cut, and of the two backgrounds together, counted
    (each event once: the generated mix) and weighted (the expected mix)."""
    x, w, c = simulated(), weights(), cut()
    xb = np.concatenate([x["bkg 1"], x["bkg 2"]])
    wb = np.concatenate([w["bkg 1"], w["bkg 2"]])
    keep = xb > c
    each = {k: float((x[k] > c).mean()) for k in NAMES}
    return dict(cut=c, each=each, counted=float(keep.mean()),
                weighted=float(wb[keep].sum() / wb.sum()),
                counted_pass=int(keep.sum()), counted_all=int(len(xb)),
                weighted_pass=float(wb[keep].sum()), weighted_all=float(wb.sum()))


@functools.lru_cache(maxsize=1)
def negative():
    """bkg 2 with negative weights: ``dict(sign, w, n_eff, weight, pos, neg, hist_pos,
    hist_neg, net, err, err_plain)`` with the per-bin sums of the positive and the negative
    weights, their net, and the bins' statistical errors sqrt(sum w^2) with and without the
    negative weights."""
    rng = np.random.default_rng(NEG_SEED)
    sigma, n = SAMPLES["bkg 2"]
    sign = np.ones(n)
    neg = rng.choice(n, NEGATIVE, replace=False)
    sign[neg] = -1
    wn = sign * sigma * L / sign.sum()
    n_eff = wn.sum() ** 2 / (wn ** 2).sum()
    x = simulated()["bkg 2"]
    plain = weights()["bkg 2"]
    pos = np.histogram(x[wn > 0], EDGES, weights=wn[wn > 0])[0]
    negh = np.histogram(x[wn < 0], EDGES, weights=wn[wn < 0])[0]
    err = np.sqrt(np.histogram(x, EDGES, weights=wn ** 2)[0])
    err_plain = np.sqrt(np.histogram(x, EDGES, weights=plain ** 2)[0])
    return dict(sign=sign, w=wn, n_eff=float(n_eff), weight=float(abs(wn[0])),
                pos=int((sign > 0).sum()), neg=int((sign < 0).sum()), hist_pos=pos,
                hist_neg=negh, net=pos + negh, err=err, err_plain=err_plain,
                n_eff_plain=float(plain.sum() ** 2 / (plain ** 2).sum()))


@functools.lru_cache(maxsize=1)
def made_up_data():
    """``dict(probe_mc, probe_data, data)``: the control region's probes in simulation and in
    data, and the pseudo-data (each process's expected count Poisson-fluctuated, the signal
    with the data's wider spread), drawn as the script draws them."""
    rng = np.random.default_rng(DATA_SEED)
    probe_mc = rng.normal(1.0, 0.10, N_MC)
    probe_data = rng.normal(1.0, DATA_SPREAD, N_DATA)
    parts = {}
    for k in NAMES:
        mean, spread = SHAPES[k][0], DATA_SPREAD if k == "signal" else SHAPES[k][1]
        n = rng.poisson(SAMPLES[k][0] * L)
        parts[k] = rng.normal(mean, spread, n)
    return dict(probe_mc=probe_mc, probe_data=probe_data, parts=parts,
                data=np.concatenate([parts[k] for k in NAMES]))


def control():
    """The control region: each efficiency, its binomial error, the scale factor and its
    statistical error."""
    d, c = made_up_data(), cut()
    pm, pd = d["probe_mc"], d["probe_data"]
    eff_mc, eff_data = float((pm > c).mean()), float((pd > c).mean())
    sf = eff_data / eff_mc
    err = sf * np.sqrt((1 - eff_data) / (eff_data * N_DATA) + (1 - eff_mc) / (eff_mc * N_MC))
    return dict(cut=c, eff_mc=eff_mc, eff_data=eff_data, n_mc=N_MC, n_data=N_DATA,
                pass_mc=int((pm > c).sum()), pass_data=int((pd > c).sum()),
                err_mc=float(np.sqrt(eff_mc * (1 - eff_mc) / N_MC)),
                err_data=float(np.sqrt(eff_data * (1 - eff_data) / N_DATA)),
                sf=float(sf), sf_err=float(err))


def passing(name):
    """The weighted simulated events of ``name`` above the cut."""
    x, w = simulated()[name], weights()[name]
    return float(w[x > cut()].sum())


def signal_region():
    """Above the cut: the simulation's weighted prediction, signal and backgrounds; the same
    with its signal scaled by the scale factor; the data's count; and their errors: the
    data's sqrt(n), the simulation's own sqrt(sum w^2), and the scale factor's error times
    the signal."""
    c = control()
    sig = passing("signal")
    bkg = passing("bkg 1") + passing("bkg 2")
    observed = int((made_up_data()["data"] > cut()).sum())
    x, w = simulated(), weights()
    mc_stat = float(np.sqrt(sum((w[k][x[k] > cut()] ** 2).sum() for k in NAMES)))
    scaled = {k: (c["sf"] if k == "signal" else 1.0) * w[k][x[k] > cut()] for k in NAMES}
    corr_mc_stat = float(np.sqrt(sum((v ** 2).sum() for v in scaled.values())))
    corrected = c["sf"] * sig + bkg
    sf_part = c["sf_err"] * sig
    pred_err = mc_stat
    corr_err = float(np.hypot(corr_mc_stat, sf_part))
    data_err = float(np.sqrt(observed))
    return dict(signal=sig, background=bkg, each={k: passing(k) for k in NAMES},
                predicted=sig + bkg, corrected=corrected, observed=observed,
                predicted_err=pred_err, corrected_err=corr_err, data_err=data_err,
                sf_part=sf_part,
                pull_before=float((sig + bkg - observed) / np.hypot(pred_err, data_err)),
                pull_after=float((corrected - observed) / np.hypot(corr_err, data_err)))


def summary():
    """The headline numbers."""
    un, we = train(False), train(True)
    e, ng, c, r = efficiency(), negative(), control(), signal_region()
    return dict(weights={k: weight(k) for k in NAMES}, expected={k: expected(k) for k in NAMES},
                unweighted_mean_score=un["mean_score"], weighted_mean_score=we["mean_score"],
                unweighted_half=un["half"], weighted_half=we["half"], cut=e["cut"],
                bkg_eff_counted=e["counted"], bkg_eff_weighted=e["weighted"],
                n_eff=ng["n_eff"], eff_mc=c["eff_mc"], eff_data=c["eff_data"], sf=c["sf"],
                sf_err=c["sf_err"], predicted=r["predicted"], corrected=r["corrected"],
                observed=r["observed"])
