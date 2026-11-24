"""How Minuit feels its way to a minimum, and how SGD stumbles there, for the two Minuit
animations of "Blocks and code" (catalogue 269-270): ``examples/minuit_vs_sgd.py`` in numpy.

The data and model are the script's: 40 points of y = a exp(-x / tau) + b with a = 5,
tau = 2.5, b = 1 and Gaussian noise of 0.3 (seeded), chi-square over all 40 points.

**Minuit** (:func:`calls`): every point iminuit's MIGRAD and then HESSE asked the script's
``chi2`` about, in order, recorded by running the script (iminuit is not a dependency of
voxanim; ``voxanim/minuit_calls.json`` keeps the record, and
``tests/test_minuit_equivalence.py`` runs iminuit again when that version is installed and
checks every call). :func:`steps` labels each call: the ``start``; a ``probe`` (the current
point moved along one axis only, up or down: how MIGRAD feels the slope and the curvature);
a ``try`` (a step to a new point, ``kept`` if MIGRAD then probes around it, else dropped in
its line search); and ``hesse`` (the calls after MIGRAD, measuring the curvature for the
error bars).

**SGD** (:func:`sgd`): from the same start, 3,000 steps of learning rate 0.005, each on 8 of
the 40 points drawn at random, the script's ``gradient``: a noisy path whose positions, once
it has arrived, average out near Minuit's minimum (:func:`compare`).
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np

TRUE = (5.0, 2.5, 1.0)
SIGMA = 0.3
N = 40
START = (1.0, 1.0, 0.0)
STEPS, BATCH, LR, SEED = 3000, 8, 0.005, 7
NAMES = ("a", "tau", "b")
RECORD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "minuit_calls.json")


@functools.lru_cache(maxsize=1)
def data():
    """``(x, y)``: the script's 40 points."""
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, N)
    y = TRUE[0] * np.exp(-x / TRUE[1]) + TRUE[2]
    return x, y + rng.normal(0, SIGMA, x.size)


def model(x, a, tau, b):
    return a * np.exp(-x / tau) + b


def chi2(p, idx=None):
    """The chi-square of parameters ``p = (a, tau, b)``, over all points or ``idx``."""
    x, y = data()
    if idx is not None:
        x, y = x[idx], y[idx]
    return float((((y - model(x, *p)) / SIGMA) ** 2).sum())


def gradient(p, idx):
    """The script's ``gradient``: the mean chi-square term of points ``idx`` and its slope."""
    x, y = data()
    a, tau, b = p
    e = np.exp(-x[idx] / tau)
    r = (y[idx] - (a * e + b)) / SIGMA
    g = np.array([(-2 * r * e).mean(), (-2 * r * a * e * x[idx] / tau ** 2).mean(),
                  (-2 * r).mean()]) / SIGMA
    return g, float((r ** 2).mean())


@functools.lru_cache(maxsize=1)
def record():
    with open(RECORD) as fh:
        return json.load(fh)


def calls():
    """Every call, in order: ``[N, 4]`` of (a, tau, b, chi-square)."""
    return np.array(record()["calls"])


@functools.lru_cache(maxsize=1)
def steps():
    """Each call labelled: ``list of dict(i, kind, axis, sign, base, point, chi2)`` with kind
    one of start, probe, try, hesse; ``axis`` and ``sign`` for a probe (which parameter, up or
    down); ``kept`` for a try (True when MIGRAD went on to probe around it)."""
    c = calls()
    n = record()["migrad_calls"]
    out, base = [], c[0, :3]
    for i, row in enumerate(c):
        p, v = row[:3], row[3]
        d = p - base
        moved = np.nonzero(np.abs(d) > 1e-12)[0]
        e = dict(i=i, point=p.tolist(), chi2=float(v), base=base.tolist())
        if i == 0:
            e["kind"] = "start"
        elif i >= n:
            e["kind"] = "hesse"
        elif len(moved) == 1:
            e.update(kind="probe", axis=int(moved[0]), sign=int(np.sign(d[moved[0]])))
        else:
            e["kind"] = "try"
        out.append(e)
        if e["kind"] in ("start",):
            base = p
        if e["kind"] == "try":
            nxt = c[i + 1, :3] if i + 1 < n else None
            kept = nxt is not None and np.sum(np.abs(nxt - p) > 1e-12) == 1
            e["kept"] = bool(kept)
            if kept:
                base = p
    return out


def summary_minuit():
    r = record()
    s = steps()
    kinds = [e["kind"] for e in s]
    tries = [e for e in s if e["kind"] == "try"]
    return dict(iminuit=r["iminuit"], migrad_calls=r["migrad_calls"], calls=len(s),
                probes=kinds.count("probe"), tries=len(tries),
                kept=sum(e["kept"] for e in tries), hesse=kinds.count("hesse"),
                values=r["values"], errors=r["errors"], fval=r["fval"], edm=r["edm"],
                start_chi2=float(calls()[0, 3]), point_evaluations=r["migrad_calls"] * N)


@functools.lru_cache(maxsize=1)
def sgd(steps_=STEPS, batch=BATCH, lr=LR, seed=SEED):
    """The script's ``run_sgd``: ``dict(path [steps+1, 3], loss [steps])``."""
    pick = np.random.default_rng(seed)
    p = np.array(START, float)
    path, loss = [p.copy()], []
    for _ in range(steps_):
        idx = pick.choice(N, batch, replace=False)
        g, l = gradient(p, idx)
        p = p - lr * g
        path.append(p.copy())
        loss.append(l)
    return dict(path=np.array(path), loss=np.array(loss))


def compare(last=1000):
    """Minuit against SGD: where each ends, SGD's spread once it has arrived, the average of
    its last ``last`` positions, and how many single-point model evaluations each used."""
    r = record()
    s = sgd()
    tail = s["path"][-last:]
    return dict(minuit=r["values"], minuit_errors=r["errors"], minuit_chi2=r["fval"],
                sgd_end=s["path"][-1].tolist(), sgd_end_chi2=chi2(s["path"][-1]),
                sgd_average=tail.mean(0).tolist(), sgd_average_chi2=chi2(tail.mean(0)),
                sgd_spread=tail.std(0).tolist(),
                minuit_evaluations=r["migrad_calls"] * N, sgd_evaluations=STEPS * BATCH,
                sgd_steps=STEPS, batch=BATCH, lr=LR)
