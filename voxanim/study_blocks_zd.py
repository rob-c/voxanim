"""The numbers behind animations 269-270 of "Blocks and code" (group ``zd``, "Minuit and SGD"):
how Minuit feels its way to a fit's minimum, and how SGD stumbles there. Everything comes from
:mod:`voxanim.study_minuit`, the numpy twin of ``examples/minuit_vs_sgd.py``, and its record of
every call iminuit 2.33.0 made (``voxanim/minuit_calls.json``, checked against iminuit by
``tests/test_minuit_equivalence.py``). Made-up data: 40 points of y = a exp(-x / tau) + b
(a = 5, tau = 2.5, b = 1, noise 0.3).

* ``minuit-feels-its-way``: the 40 points and the model's curve at the start (1, 1, 0),
  chi-square 2,692; every call MIGRAD made, in its order (:func:`timeline`): the start, 12
  probes (each parameter moved a little up and down, two step sizes, MIGRAD's own), the first
  try (3,454: dropped), the line search back (109: kept), the probes again round each kept
  point, 65 calls in all (52 probes, 12 tries, 5 kept); the minimum a = 5.14, tau = 2.41,
  b = 1.007, chi-square 33.94, EDM 0.00014; HESSE's 16 calls and the errors 0.18, 0.21, 0.11.
  Each probe is marked lower, higher or flat (changed by less than 0.0001) against the point
  it was asked from (:data:`FLAT`).
* ``minuit-and-sgd``: SGD from the same start, 3,000 steps of learning rate 0.005, each on 8
  of the 40 points (one run, seed 7): the first batch and its step (recomputed here from the
  seed and checked against the path); the walk; its last 1,000 positions (spread about 0.02,
  a tenth of Minuit's errors), whose average (5.131, 2.399, 1.020) has chi-square 33.96
  against Minuit's 33.94; single-point model evaluations 24,000 (3,000 x 8) against 2,600
  (65 x 40). Counts, not times.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_minuit as M
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("made-up data: examples/minuit_vs_sgd.py (40 points of a exp(-x/tau) + b with noise "
          "0.3); every call iminuit %s's MIGRAD and HESSE made, recorded by running it; SGD "
          "run once in numpy (3,000 steps, learning rate 0.005, 8 points a step, seed 7)")
FLAT = 1e-4                  # a probe that moved chi-square by less than this: "flat"
TAIL = 1000                  # SGD's last positions, averaged
GRID = 81                    # points along x for the model's curve
WALK_EVERY = 5               # 270: SGD's path drawn through every 5th position after step 200
NAMES = ("a", "τ", "b")      # as the screen writes them
LETTERS = ("a", "tau", "b")  # as the script writes them


def chi2_text(v):
    """A chi-square as the counter shows it: ``2,691.61``, ``108.92``, ``33.9447``."""
    v = float(v)
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 100:
        return "%.2f" % v
    return "%.4f" % v


def short(v):
    """A chi-square in round words: ``2,692``, ``109``, ``35.3``, ``33.9``."""
    v = float(v)
    if v >= 1000:
        return commas(v)
    if v >= 100:
        return "%.0f" % v
    return "%.1f" % v


def delta_text(d):
    """A change of chi-square: ``−0.11``, ``+4.8``, ``+0.00056``; ``flat`` below :data:`FLAT`."""
    d = float(d)
    if abs(d) < FLAT:
        return "flat"
    if abs(d) >= 10:
        return ("+" if d > 0 else "−") + commas(abs(d))
    s = "%+.2g" % d
    if "e" in s:
        s = "%+.5f" % d
    return s.replace("-", "−")


def step_text(v):
    """A probe's step: ``+0.00025``, ``−0.0025``."""
    v = float(v)
    s = "%+.2g" % v
    if "e" in s:
        s = ("%+.6f" % v).rstrip("0")
    return s.replace("-", "−")


def change(d):
    """``lower``, ``higher`` or ``flat``: what a probe found."""
    return "flat" if abs(d) < FLAT else ("lower" if d < 0 else "higher")


def curve(p):
    """The model's curve for parameters ``p`` on :data:`GRID` points over x 0..10."""
    xs = np.linspace(0.0, 10.0, GRID)
    return M.model(xs, *p)


@functools.lru_cache(maxsize=1)
def timeline():
    """Every call, in order, as the animation uses it: ``list of dict`` with ``i``, ``kind``
    (start, probe, try, hesse), ``axis`` and ``sign`` (a probe), ``kept`` (a try), ``point``,
    ``chi2``, ``base`` (the index of the call it was asked from: the start or the last kept
    try), ``d`` (its move from there), ``delta`` (its chi-square minus the base's), ``change``
    (lower / higher / flat), ``round`` (0 for the first probes, +1 at each kept try and at
    HESSE), ``near`` (True for a call drawn in the magnified inset: a probe, a HESSE call, or a
    dropped try within 0.01 of its base)."""
    out, base, rnd = [], 0, 0
    calls = M.steps()
    for e in calls:
        i, kind = e["i"], e["kind"]
        if kind == "hesse" and calls[i - 1]["kind"] != "hesse":
            rnd += 1
        p = np.asarray(e["point"], float)
        b = np.asarray(calls[base]["point"], float)
        d = p - b
        delta = e["chi2"] - calls[base]["chi2"]
        row = dict(i=i, kind=kind, axis=e.get("axis"), sign=e.get("sign"), kept=e.get("kept"),
                   point=p.tolist(), chi2=float(e["chi2"]), base=base, d=d.tolist(),
                   delta=float(delta), change=change(delta), round=rnd,
                   near=bool(kind in ("probe", "hesse")
                             or (kind == "try" and not e["kept"]
                                 and float(np.abs(d).max()) < 0.01)))
        if kind == "start":
            row["near"] = False
        out.append(row)
        if kind == "try" and e["kept"]:
            base, rnd = i, rnd + 1
    return out


def first_batch():
    """SGD's first step, recomputed from the seed: the 8 points it looks at, their slope, the
    step (learning rate x slope) and where it lands (checked against :func:`study_minuit.sgd`)."""
    pick = np.random.default_rng(M.SEED)
    idx = pick.choice(M.N, M.BATCH, replace=False)
    p0 = np.array(M.START, float)
    g, loss = M.gradient(p0, idx)
    p1 = p0 - M.LR * g
    if not np.allclose(p1, M.sgd()["path"][1]):
        raise ValueError("the first step is not the path's")
    return dict(idx=[int(v) for v in idx], grad=g.tolist(), step=(-M.LR * g).tolist(),
                p1=p1.tolist(), loss=float(loss))


@functools.lru_cache(maxsize=1)
def common():
    """What both show: the data, the record's summary, Minuit's kept path and minimum."""
    x, y = M.data()
    s = M.summary_minuit()
    tl = timeline()
    kept = [0] + [e["i"] for e in tl if e["kind"] == "try" and e["kept"]]
    vals, errs = s["values"], s["errors"]
    return dict(
        source=SOURCE % s["iminuit"], iminuit=s["iminuit"], x=x.tolist(), y=y.tolist(),
        n_points=M.N, sigma=M.SIGMA, true=list(M.TRUE), start=list(M.START),
        names=list(NAMES), letters=list(LETTERS),
        grid=np.linspace(0.0, 10.0, GRID).tolist(),
        migrad_calls=s["migrad_calls"], calls=s["calls"], probes=s["probes"],
        tries=s["tries"], kept=s["kept"], hesse=s["hesse"],
        values=vals, errors=errs, fval=s["fval"], edm=s["edm"], start_chi2=s["start_chi2"],
        values_text=["%.2f" % vals[0], "%.2f" % vals[1], "%.3f" % vals[2]],
        errors_text=["%.2f" % v for v in errs], fval_text="%.1f" % s["fval"],
        edm_text="%.5f" % s["edm"], start_text=commas(s["start_chi2"]),
        kept_calls=kept, kept_points=[tl[i]["point"] for i in kept],
        kept_chi2=[tl[i]["chi2"] for i in kept], kept_text=[short(tl[i]["chi2"]) for i in kept],
        point_evaluations=s["point_evaluations"],
        code_model=list(code_lines("mn-model")), code_chi2=list(code_lines("mn-chi2")),
        code_migrad=list(code_lines("mn-migrad")), code_sgd=list(code_lines("mn-sgd")),
        printed="MIGRAD: %d calls; HESSE: %d more" % (s["migrad_calls"], s["hesse"]))


# ====================================================================== 269
def feels():
    m = dict(common())
    tl = timeline()
    first = [e for e in tl if e["round"] == 0 and e["kind"] == "probe"]
    overshoot = tl[13]
    back = tl[14]
    if overshoot["kind"] != "try" or overshoot["kept"] or not back["kept"]:
        raise ValueError("the record's first step is not an overshoot and a way back")
    curves = {e["i"]: curve(e["point"]).tolist() for e in tl if e["kind"] in ("start", "try")}
    m.update(
        timeline=tl, curves=curves,
        chi2_text=[chi2_text(e["chi2"]) for e in tl],
        delta_text=[delta_text(e["delta"]) if e["kind"] != "start" else "" for e in tl],
        step_text=[step_text(e["d"][e["axis"]]) if e["kind"] == "probe" else "" for e in tl],
        first_probes=len(first), first_steps=sorted({abs(round(e["d"][e["axis"]], 6))
                                                     for e in first}),
        overshoot=overshoot["point"], overshoot_chi2=overshoot["chi2"],
        overshoot_text=commas(overshoot["chi2"]), back=back["point"], back_chi2=back["chi2"],
        back_text=short(back["chi2"]),
        rounds=max(e["round"] for e in tl) + 1,
        limitations=[
            "every call iminuit %s's MIGRAD and HESSE made on this fit, in its order, recorded "
            "by running examples/minuit_vs_sgd.py; the two step sizes of the first probes are "
            "MIGRAD's own" % m["iminuit"],
            "what is shown is what the calls show: points asked along one axis (probes), "
            "points stepped to (tries, kept when MIGRAD probed round them next); MIGRAD builds "
            "up the curvature as it goes (a variable-metric method), which is not drawn",
            "the probes are far too small to see at the scale of the path: the inset is "
            "magnified, each round to fit its largest probe; 'flat': chi-square changed by "
            "less than 0.0001",
            "made-up data: 40 points, one draw of the noise (seed 1)"])
    probes_x = np.arange(len(tl), dtype=float)
    panels = (
        Panel("Chi-square at every call", "MIGRAD's 65 calls, then HESSE's 16, in order",
              "curve",
              dict(x=probes_x, ys=np.array([[e["chi2"] for e in tl]]), labels=["chi-square"])),
        Panel("The calls, by kind", "probes, tries, of which kept, HESSE", "bars",
              dict(values=np.array([m["probes"], m["tries"], m["kept"], m["hesse"]], float),
                   labels=["probes", "tries", "kept", "HESSE"])),
        Panel("The kept points", "a against tau, from the start to the minimum", "points",
              dict(xy=np.array([[p[0], p[1]] for p in m["kept_points"]]))),
        Panel("Data and the model", "the 40 points; the curve at the start and the minimum",
              "curve", dict(x=np.array(m["grid"]),
                            ys=np.array([curves[0], curve(m["values"]).tolist()]),
                            labels=["start", "minimum"])),
    )
    return Study("How Minuit feels its way",
                 "%d calls: %d probes up and down each axis, %d tries, %d kept; chi-square "
                 "%s to %.2f" % (m["migrad_calls"], m["probes"], m["tries"], m["kept"],
                                 m["start_text"], m["fval"]), panels, m)


# ====================================================================== 270
def walk_indices(n, dense=200, every=WALK_EVERY):
    """The positions SGD's path is drawn through: every one of the first ``dense``, then every
    ``every``-th, and the last."""
    idx = list(range(min(dense, n))) + list(range(dense, n, every))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def sgd():
    m = dict(common())
    r = M.sgd()
    path = r["path"]
    c = M.compare(TAIL)
    fb = first_batch()
    idx = walk_indices(len(path))
    walk_chi2 = [M.chi2(path[i]) for i in idx]
    tail = path[-TAIL:]
    spread = c["sgd_spread"]
    ratio = [s / e for s, e in zip(spread, m["errors"], strict=True)]
    m.update(
        path=path.tolist(), walk_idx=idx, walk_chi2=walk_chi2,
        steps=M.STEPS, batch=M.BATCH, lr=M.LR, seed=M.SEED, tail_n=TAIL,
        tail=tail.tolist(), first=fb,
        sgd_average=c["sgd_average"], sgd_average_chi2=c["sgd_average_chi2"],
        sgd_end=c["sgd_end"], sgd_end_chi2=c["sgd_end_chi2"], sgd_spread=spread,
        spread_ratio=ratio, minuit_chi2=c["minuit_chi2"],
        average_text=["%.3f" % v for v in c["sgd_average"]],
        average_chi2_text="%.2f" % c["sgd_average_chi2"],
        minuit_chi2_text="%.2f" % c["minuit_chi2"],
        spread_text="%.2f" % float(np.mean(spread)),
        sgd_evaluations=c["sgd_evaluations"], minuit_evaluations=c["minuit_evaluations"],
        sgd_evaluations_text=commas(c["sgd_evaluations"]),
        minuit_evaluations_text=commas(c["minuit_evaluations"]),
        curve_average=curve(c["sgd_average"]).tolist(),
        curve_minuit=curve(m["values"]).tolist(),
        limitations=[
            "SGD is one run with one seed (7): another seed walks another way; its numbers "
            "are that run's",
            "the spread is the standard deviation of each parameter over the last 1,000 "
            "positions; Minuit's errors are HESSE's",
            "evaluations count the model run on one point: SGD 3,000 steps x 8 points, MIGRAD "
            "65 calls x 40 points (HESSE adds 16 x 40); counts, not times",
            "Minuit's numbers are iminuit %s's, recorded by running the script" % m["iminuit"]])
    chi_curve = np.array([walk_chi2])
    panels = (
        Panel("SGD's chi-square on all 40 points", "along its walk (every 5th step after "
              "200)", "curve", dict(x=np.array(idx, float), ys=chi_curve, labels=["SGD"])),
        Panel("Where each walked", "a against tau: SGD's path", "points",
              dict(xy=np.array([[path[i][0], path[i][1]] for i in idx]))),
        Panel("Single-point model evaluations", "SGD 3,000 x 8, MIGRAD 65 x 40", "bars",
              dict(values=np.array([c["sgd_evaluations"], c["minuit_evaluations"]], float),
                   labels=["SGD", "MIGRAD"])),
        Panel("SGD's spread at the end against Minuit's errors", "a, tau, b", "bars",
              dict(values=np.array(list(spread) + list(m["errors"]), float),
                   labels=["SGD a", "SGD tau", "SGD b", "Minuit a", "Minuit tau",
                           "Minuit b"])),
    )
    return Study("Minuit and SGD",
                 "SGD's 3,000 noisy steps average out at chi-square %s against Minuit's %s; "
                 "%s single-point evaluations against %s"
                 % (m["average_chi2_text"], m["minuit_chi2_text"], m["sgd_evaluations_text"],
                    m["minuit_evaluations_text"]), panels, m)


STUDIES = {"minuit-feels-its-way": feels, "minuit-and-sgd": sgd}
