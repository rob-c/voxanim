"""Redesigned experiments for three of the optimisation animations (75, 76, 80).

The first versions (:mod:`voxanim.next_systems`, :mod:`voxanim.next_systems_real`) could
not show their concept:

* ``newton-vs-gradient-descent`` ran gradient descent at 0.9 / (largest curvature) only,
  where it neither zig-zags nor shows what limits it, and compared losses after a fixed
  80 updates instead of counting the updates each method needs.
* ``edge-of-stability`` started its large-step run *above* the limit 2 / (step size), so
  the curvature was thrown down to the limit; the effect is the curvature *rising* to it.
* ``loss-spikes`` swapped the training text at update 60, so every setting spiked and
  the remedies could only change the recovery.

Here: ridge regression of MNIST labels with gradient descent at a safe step and at one
near the stability limit (and one past it), with the updates each needs to reach a
fixed loss at four condition numbers; a tanh MLP on 256 MNIST digits whose curvature
starts far under 2 / (step size) and climbs to it; and a small transformer on fixed
Tiny Shakespeare windows, trained long enough with Adam for spikes to arrive by
themselves. Everything the drawings need is recorded by the experiments, so
:mod:`voxanim.mech_optim` computes nothing for these three.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
SEED = 17


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, seed=SEED, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-optim-" + name, settings,
                                                    build).items()}


def _provenance(meta):
    """A split's provenance without its list of example indices."""
    return {k: v for k, v in meta.items() if k != "example_indices"}


def _curve(title, caption, ys, labels, x=None, **more):
    ys = np.atleast_2d(np.asarray(ys, float))
    return Panel(title, caption, "curve",
                 dict(x=np.arange(ys.shape[1]) if x is None else np.asarray(x), ys=ys,
                      labels=list(labels), **more))


def _bars(title, caption, values, labels, **more):
    return Panel(title, caption, "bars",
                 dict(values=np.asarray(values, float), labels=list(labels), **more))


# ====================================================================== 75
NEWTON_RIDGES = (0.002, 0.02, 0.2, 2.0)     # the second is the problem drawn
NEWTON_STEPS = (1.0, 1.9, 2.1)              # gradient-descent step x largest curvature
NEWTON_TARGET = 0.01                        # "there": 1% of the starting gap to the minimum
NEWTON_SHOWN = 81


def newton_problem(ridge=NEWTON_RIDGES[1]):
    """Ridge regression of one-hot labels on 1,024 MNIST digits (7 x 7 pooled pixels and
    a bias): the exact Hessian ``h`` of ``mean |a w - t|^2 / 2 + ridge |w|^2 / 2``, the
    linear term ``b`` (the gradient is ``h w - b``), and the held-out digits."""
    from . import realdata as R

    x, y, train = R.mnist_split(1024, "train", seed=17)
    xt, yt, test = R.mnist_split(384, "test", seed=73)
    a, at = np.c_[x, np.ones(len(x))], np.c_[xt, np.ones(len(xt))]
    target = np.eye(10)[y]
    h = a.T @ a / len(a) + ridge * np.eye(a.shape[1])
    return dict(a=a, at=at, target=target, yt=yt, h=h, b=a.T @ target / len(a), x=x,
                ridge=ridge, train=train, test=test)


def _newton_build():
    out = dict(ridges=np.array(NEWTON_RIDGES), factors=np.array(NEWTON_STEPS))
    conds, counts, accs, floors = [], [], [], []
    for ridge in NEWTON_RIDGES:
        p = newton_problem(ridge)
        a, at, target, yt, h, b = (p[k] for k in ("a", "at", "target", "yt", "h", "b"))

        def loss(w):
            e = a @ w - target  # noqa: B023
            return float(np.mean(np.sum(e * e, axis=1)) / 2 + ridge / 2 * np.sum(w * w))  # noqa: B023

        def accuracy(w):
            return float(np.mean((at @ w).argmax(1) == yt))  # noqa: B023

        lam = np.linalg.eigvalsh(h)
        zero = np.zeros((a.shape[1], 10))
        best = zero - np.linalg.solve(h, h @ zero - b)          # the Newton step from zero
        start, floor = loss(zero), loss(best)
        goal = floor + NEWTON_TARGET * (start - floor)
        conds.append(lam[-1] / lam[0])
        floors.append(floor)
        row, acc = [], []
        for factor in NEWTON_STEPS[:2]:
            w, n = zero.copy(), 0
            while loss(w) > goal:
                w = w - factor / lam[-1] * (h @ w - b)
                n += 1
            row.append(n)
            acc.append(accuracy(w))
        counts.append(row + [1])
        accs.append(acc + [accuracy(best)])
        if ridge == NEWTON_RIDGES[1]:
            paths, losses, tests = [], [], []
            for factor in NEWTON_STEPS:
                w, ws = zero.copy(), []
                for _ in range(NEWTON_SHOWN):
                    ws.append(w)
                    w = w - factor / lam[-1] * (h @ w - b)
                paths.append(ws)
            paths.append([zero] + [best] * (NEWTON_SHOWN - 1))
            for ws in paths:
                losses.append([loss(w) for w in ws])
                tests.append([accuracy(w) for w in ws])
            out.update(h=h, w=np.array(paths), loss=losses, accuracy=tests, start=start,
                       floor=floor, goal=goal, lam=lam)
    return dict(cond=conds, counts=counts, reached_accuracy=accs, floors=floors, **out)


def newton_data():
    """What :func:`newton` measured. ``w`` (4, 81, 50, 10), ``loss`` and ``accuracy``
    (4, 81): gradient descent at 1.0, 1.9 and 2.1 / (largest curvature) and the Newton
    step, before each of the first 81 updates at ridge 0.02. ``counts`` (4 ridges, 3):
    the updates gradient descent at 1.0 and 1.9 and the Newton step need to bring the
    loss within 1% of its starting distance to the minimum; ``cond``: the condition
    number at each ridge; ``reached_accuracy``: held-out accuracy at that moment."""
    return _cache("newton", _newton_build, (_newton_build, newton_problem),
                  ridges=NEWTON_RIDGES, steps=NEWTON_STEPS, target=NEWTON_TARGET)


def newton():
    d = newton_data()
    p = newton_problem()
    names = ["gradient descent, step 1 / steepest", "gradient descent, step 1.9 / steepest",
             "gradient descent, step 2.1 / steepest", "Newton step"]
    k = 1
    return Study(
        "Newton's step and gradient descent",
        "MNIST ridge regression (1,024 train / 384 held out); exact Hessian; plain NumPy",
        (
            Panel("The data determine the curvature",
                  "H = AᵀA / N + 0.02 I: the exact Hessian of this quadratic loss.",
                  "matrix", dict(values=d["h"])),
            _curve("The first 80 updates",
                   "Loss before each update; past 2 / (largest curvature) it grows.",
                   np.minimum(d["loss"], 1e6), names),
            _curve("Updates needed against the condition number",
                   "Updates until the loss is within 1% of its starting distance to the "
                   "minimum, at four ridge strengths.",
                   d["counts"].T, ["gradient descent, 1 / steepest",
                                   "gradient descent, 1.9 / steepest", "Newton step"],
                   x=d["cond"], ridges=d["ridges"]),
            _bars("Held-out accuracy on reaching that loss",
                  "384 held-out digits, ridge 0.02, when each method gets there.",
                  d["reached_accuracy"][k], ["gradient descent, 1 / steepest",
                                             "gradient descent, 1.9 / steepest",
                                             "Newton step"]),
        ),
        dict(
            seed=SEED,
            objective="mean squared error / 2 to one-hot digit targets + ridge |w|^2 / 2",
            ridges=list(NEWTON_RIDGES), ridge_drawn=NEWTON_RIDGES[k],
            condition_numbers=[float(v) for v in d["cond"]],
            step_rule="factor / largest Hessian eigenvalue, factors %s" % (NEWTON_STEPS,),
            target="loss - minimum <= %g x (starting loss - minimum)" % NEWTON_TARGET,
            updates_to_target={"ridge %g" % r: [int(v) for v in row]
                               for r, row in zip(NEWTON_RIDGES, d["counts"])},
            held_out_accuracy_at_target=[float(v) for v in d["reached_accuracy"][k]],
            control="the same problem and start (all weights 0) for every method",
            evidence_kind="exact arithmetic on collected data (a quadratic loss)",
            dataset=dict(train=_provenance(p["train"]), test=_provenance(p["test"])),
            designs_tried=["this one (fixed before the counts were run); the first study's "
                           "Adam run is dropped"],
            limitations=["A quadratic loss: Newton's one update is exact here and is not "
                         "for a network's loss. Ridge strength changes the problem as "
                         "well as its condition number."],
        ),
    )


# ====================================================================== 76
EDGE_STEPS = (0.2, 0.3, 0.4, 0.5)
EDGE_SEEDS = (17, 18, 19)
EDGE_UPDATES = 6000
EDGE_EVERY = 20
EDGE_MAIN = 1                               # the step size whose run is drawn up close
EDGE_SETTLED = 4000                         # "levels off": the mean from here on
EDGE_BEFORE, EDGE_AFTER = 44, 20            # the updates kept around the first excursion
EDGE_SNAPSHOTS = 200                        # valley profiles this often, until the limit


def edge_data_split():
    """256 training and 1,000 held-out MNIST digits, 14 x 14 pooled, centred on the
    training mean; one-hot targets."""
    from . import realdata as R

    x, y, train = R.mnist_split(256, "train", 14, 17)
    xt, yt, test = R.mnist_split(1000, "test", 14, 73)
    mean = x.mean(0)
    return dict(x=x - mean, t=np.eye(10)[y], xt=xt - mean, tt=np.eye(10)[yt], yt=yt,
                raw=x, train=train, test=test)


def edge_init(seed, d=196, h=64, c=10):
    rng = np.random.default_rng(seed)
    return dict(W=rng.normal(0, 1 / np.sqrt(d), (d, h)), b=np.zeros(h),
                U=rng.normal(0, 1 / np.sqrt(h), (h, c)), v=np.zeros(c))


def edge_pack(P):
    return np.concatenate([v.ravel() for v in P.values()])


def edge_unpack(v, P):
    out, at = {}, 0
    for k, p in P.items():
        out[k] = v[at:at + p.size].reshape(p.shape)
        at += p.size
    return out


def edge_loss(P, x, t, grads=True):
    """Half the mean squared error of a tanh MLP, and its exact gradient."""
    h = np.tanh(x @ P["W"] + P["b"])
    e = h @ P["U"] + P["v"] - t
    loss = float(0.5 * np.mean(np.sum(e * e, axis=1)))
    if not grads:
        return loss
    dz = e / len(x)
    dh = (dz @ P["U"].T) * (1 - h * h)
    return loss, dict(W=x.T @ dh, b=dh.sum(0), U=h.T @ dz, v=dz.sum(0))


def edge_sharpness(P, x, t, v, iterations):
    """The top Hessian eigenvalue and its direction by power iteration from ``v``, on
    Hessian-vector products (symmetric differences of exact gradients)."""
    p = edge_pack(P)

    def hv(q):
        eps = 1e-4
        up = edge_pack(edge_loss(edge_unpack(p + eps * q, P), x, t)[1])
        down = edge_pack(edge_loss(edge_unpack(p - eps * q, P), x, t)[1])
        return (up - down) / (2 * eps)

    for _ in range(iterations):
        u = hv(v)
        v = u / max(np.linalg.norm(u), 1e-30)
    return float(v @ hv(v)), v


def _edge_build():
    d = edge_data_split()
    x, t = d["x"], d["t"]
    grid = np.linspace(-1.0, 1.0, 41)
    loss = np.zeros((len(EDGE_SEEDS), len(EDGE_STEPS), EDGE_UPDATES + 1))
    sharp = np.zeros((len(EDGE_SEEDS), len(EDGE_STEPS), EDGE_UPDATES // EDGE_EVERY + 1))
    held = np.zeros((len(EDGE_SEEDS), len(EDGE_STEPS), 2))
    kept = {}
    for a, seed in enumerate(EDGE_SEEDS):
        for b, lr in enumerate(EDGE_STEPS):
            P = edge_init(seed)
            v = np.random.default_rng(1).normal(size=len(edge_pack(P)))
            main = a == 0 and b == EDGE_MAIN
            for step in range(EDGE_UPDATES + 1):
                loss[a, b, step], G = edge_loss(P, x, t)
                if step % EDGE_EVERY == 0:
                    sharp[a, b, step // EDGE_EVERY], v = edge_sharpness(
                        P, x, t, v, 40 if step == 0 else 10)
                if main and step % 100 == 0:
                    kept[step] = (edge_pack(P), v.copy())
                for k in P:
                    P[k] -= lr * G[k]
            out = np.tanh(d["xt"] @ P["W"] + P["b"]) @ P["U"] + P["v"]
            held[a, b] = (0.5 * np.mean(np.sum((out - d["tt"]) ** 2, axis=1)),
                          np.mean(out.argmax(1) == d["yt"]))
    # the run drawn up close: the valley across the sharpest direction as it narrows...
    lr = EDGE_STEPS[EDGE_MAIN]
    P = edge_init(EDGE_SEEDS[0])
    line, reach = loss[0, EDGE_MAIN], sharp[0, EDGE_MAIN]
    first = int(np.argmax(reach >= 2 / lr)) * EDGE_EVERY
    snaps = np.arange(0, first + 1, EDGE_SNAPSHOTS)
    width = np.linspace(-0.3, 0.3, 41)
    profiles, snap_sharp = [], []
    for s in snaps:
        p, v = kept[int(s) - int(s) % 100]
        Q = edge_unpack(p.copy(), P)
        for _ in range(int(s) % 100):
            G = edge_loss(Q, x, t)[1]
            for k in Q:
                Q[k] -= lr * G[k]
        lam, v = edge_sharpness(Q, x, t, v, 40)
        p = edge_pack(Q)
        profiles.append([edge_loss(edge_unpack(p + w * v, P), x, t, False) for w in width])
        snap_sharp.append(lam)
    # ...and its first excursion: the first update after the limit is reached at which the
    # loss is twice the lowest it has been, and the highest loss of the 40 updates after
    low = np.minimum.accumulate(line)
    over = np.nonzero((np.arange(len(line)) > first) & (line > 2 * low))[0][0]
    peak = int(over + np.argmax(line[over:over + 40]))
    start = peak - EDGE_BEFORE
    p, v = kept[start - start % 100 - 100]
    Q = edge_unpack(p.copy(), P)
    for i in range(100 + start % 100):                       # with the direction kept warm
        if i % 10 == 0:
            v = edge_sharpness(Q, x, t, v, 10)[1]
        G = edge_loss(Q, x, t)[1]
        for k in Q:
            Q[k] -= lr * G[k]
    rows = dict(lam=[], off=[], loss=[], p=[], v=[])
    for _ in range(EDGE_BEFORE + EDGE_AFTER + 1):
        value, G = edge_loss(Q, x, t)
        before = v
        lam, v = edge_sharpness(Q, x, t, v, 12)
        if before @ v < 0:
            v = -v
        rows["lam"].append(lam)
        rows["off"].append(float(edge_pack(G) @ v) / lam)      # how far from the floor
        rows["loss"].append(value)
        rows["p"].append(edge_pack(Q))
        rows["v"].append(v)
        for k in Q:
            Q[k] -= lr * G[k]
    off = np.array(rows["off"])
    reach_x = 1.25 * np.abs(off).max()
    across = np.array([[edge_loss(edge_unpack(p + (q * reach_x - o) * v, P), x, t, False)
                        for q in grid] for p, v, o in zip(rows["p"], rows["v"], off)])
    return dict(loss=loss, sharp=sharp, held=held, first=first, snaps=snaps, width=width,
                profiles=profiles, snap_sharp=snap_sharp, window=start + np.arange(len(off)),
                w_lam=rows["lam"], w_off=off, w_loss=rows["loss"], w_grid=grid * reach_x,
                w_line=across, peak=peak, digits=d["raw"][:6])


def edge_data():
    """What :func:`edge_stability` measured. ``loss`` (3 seeds, 4 step sizes, 6001) and
    ``sharp`` (3, 4, 301: the top Hessian eigenvalue every 20 updates); ``held`` (3, 4,
    2): held-out loss and accuracy at the end. For the first seed at step size 0.3:
    ``profiles`` (the loss across the sharpest direction, ``width`` either side of the
    weights, at the updates ``snaps``), and around its first excursion (the updates
    ``window``) the curvature ``w_lam``, the weights' distance from the valley floor
    along that direction ``w_off`` (slope / curvature), the loss ``w_loss``, and the
    loss across the valley ``w_line`` at the positions ``w_grid`` measured from the floor."""
    return _cache("edge", _edge_build,
                  (_edge_build, edge_data_split, edge_init, edge_loss, edge_sharpness),
                  steps=EDGE_STEPS, seeds=EDGE_SEEDS, updates=EDGE_UPDATES)


def edge_levels(d):
    """Per seed and step size: the mean curvature from update 4,000 on."""
    return d["sharp"][:, :, EDGE_SETTLED // EDGE_EVERY:].mean(-1)


def edge_stability():
    d = edge_data()
    split = edge_data_split()
    names = ["step size %g" % lr for lr in EDGE_STEPS]
    level = edge_levels(d)
    limit = 2 / np.array(EDGE_STEPS)
    rises = (np.diff(d["loss"], axis=-1) > 0).sum(-1)
    return Study(
        "The edge of stability",
        "MNIST: 256 digits, 14×14 pooled pixels, tanh MLP, squared error, full-batch "
        "gradient descent",
        (
            _curve("Full-batch training",
                   "Training loss before every update (first of three starts).",
                   d["loss"][0], names),
            _curve("Top curvature through training",
                   "Largest Hessian eigenvalue every 20 updates: power iteration on "
                   "Hessian-vector products.",
                   d["sharp"][0], names, x=np.arange(d["sharp"].shape[-1]) * EDGE_EVERY,
                   limits=limit),
            _curve("The first excursion at step size %g" % EDGE_STEPS[EDGE_MAIN],
                   "Every update around it: the loss, the curvature, and how far the "
                   "weights are from the floor of the sharpest valley.",
                   [d["w_loss"], d["w_lam"], d["w_off"]],
                   ["loss", "curvature", "distance from the floor"], x=d["window"]),
            _bars("Where the curvature levels off",
                  "Mean curvature from update 4,000 on × step size / 2, over three "
                  "starts: 1 is the limit.",
                  (level * np.array(EDGE_STEPS) / 2).mean(0), names,
                  each=level, limits=limit),
        ),
        dict(
            seed=SEED, seeds=list(EDGE_SEEDS), step_sizes=list(EDGE_STEPS),
            updates=EDGE_UPDATES,
            model="196-64-10 tanh MLP, squared error to one-hot targets",
            curvature="top Hessian eigenvalue, warm-started power iteration (10 rounds) on "
                      "central differences of exact gradients",
            limits=[float(v) for v in limit],
            starting_curvature=[float(v) for v in d["sharp"][:, 0, 0]],
            settled_curvature=dict(mean=[float(v) for v in level.mean(0)],
                                   min=[float(v) for v in level.min(0)],
                                   max=[float(v) for v in level.max(0)]),
            updates_where_the_loss_rose=[[int(v) for v in row] for row in rises],
            held_out_loss=[[float(v) for v in row] for row in d["held"][..., 0]],
            held_out_accuracy=[[float(v) for v in row] for row in d["held"][..., 1]],
            control="step size 0.2: its limit, 10, is never reached, and its loss never "
                    "rises",
            evidence_kind="actual NumPy updates on collected data",
            dataset=dict(train=_provenance(split["train"]), test=_provenance(split["test"]),
                         preprocessing="14x14 mean pooling, minus the training mean image"),
            designs_tried=[
                "512 digits, 7x7 pixels uncentred, 32 hidden units: curvature starts at 7.9 "
                "and rises too little to separate the limits",
                "the same, centred: starts at 3.0, climbs slowly",
                "this one: 256 digits, 14x14 centred, 64 hidden units; step sizes 0.2, 0.3 "
                "and 0.5 tried first, 0.4 added before the three starts were run",
            ],
            limitations=["256 digits are memorised: held-out accuracy is about 76% for "
                         "every step size."],
        ),
    )


# ====================================================================== 80
SPIKE_SETTINGS = (                          # name, beta2, epsilon, clip
    ("Adam: memory 1,000 updates", 0.999, 1e-8, None),
    ("memory 10 updates", 0.9, 1e-8, None),
    ("larger ε", 0.999, 1e-3, None),
    ("gradients clipped to 1", 0.999, 1e-8, 1.0),
)
SPIKE_SEEDS = (17, 18, 19)
SPIKE_UPDATES = 3000
SPIKE_RATE = 0.02
SPIKE_FROM = 300                            # spikes are looked for after this update
SPIKE_FACTOR = 2.0                          # a spike: the loss at twice its lowest so far


def spike_text():
    """64 windows of 13 characters from the training part of Tiny Shakespeare (the
    model reads 12 and is scored on each next one), and 256 from the held-out part."""
    from . import realdata as R

    return R.text_sequences(64, 13, "train", 17), R.text_sequences(256, 13, "test", 73)


def spike_events(loss, start=SPIKE_FROM, factor=SPIKE_FACTOR):
    """The updates from ``start`` on at which the loss crosses ``factor`` x the lowest it
    has been, and the largest such multiple it reaches."""
    loss = np.asarray(loss, float)
    low = np.minimum.accumulate(loss)
    high = loss > factor * low
    crossings = np.nonzero(high[1:] & ~high[:-1])[0] + 1
    return crossings[crossings >= start], float((loss / low)[start:].max())


def _spike_build():
    from . import realdata as R
    from . import tinylm as T

    seq, held = spike_text()
    x, y = seq[:, :-1], seq[:, 1:]
    cfg = T.Config(vocab=len(R.vocabulary()), width=16, heads=2, kv_heads=2, ff=24)
    keys = ("loss", "norm", "used", "moment", "step")
    out = {k: np.zeros((len(SPIKE_SETTINGS), len(SPIKE_SEEDS), SPIKE_UPDATES)) for k in keys}
    test = np.zeros((len(SPIKE_SETTINGS), len(SPIKE_SEEDS)))
    for a, (_name, beta2, eps, clip) in enumerate(SPIKE_SETTINGS):
        for b, seed in enumerate(SPIKE_SEEDS):
            p = T.init(cfg, seed)
            m = {k: np.zeros_like(v) for k, v in p.items()}
            v = {k: q.copy() for k, q in m.items()}
            for i in range(SPIKE_UPDATES):
                loss, g = T.loss_grads(p, x, y, cfg)
                norm = np.sqrt(sum(float((z * z).sum()) for z in g.values()))
                factor = min(1.0, clip / max(norm, 1e-12)) if clip else 1.0
                moved = 0.0
                for k in p:
                    z = g[k] * factor
                    m[k] = 0.9 * m[k] + 0.1 * z
                    v[k] = beta2 * v[k] + (1 - beta2) * z * z
                    step = (SPIKE_RATE * (m[k] / (1 - 0.9 ** (i + 1)))
                            / (np.sqrt(v[k] / (1 - beta2 ** (i + 1))) + eps))
                    p[k] -= step
                    moved += float((step * step).sum())
                out["loss"][a, b, i] = loss
                out["norm"][a, b, i] = norm
                out["used"][a, b, i] = norm * factor
                # Adam's memory as it divides by it: with the start-up correction
                out["moment"][a, b, i] = np.sqrt(
                    sum(float(q.sum()) for q in v.values()) / (1 - beta2 ** (i + 1)))
                out["step"][a, b, i] = np.sqrt(moved)
            test[a, b] = T.loss(p, held[:, :-1], held[:, 1:], cfg)
    return dict(held=test, text=seq, **out)


def spike_data():
    """What :func:`loss_spikes` measured, per setting (4) and start (3) at each of 3,000
    updates: ``loss``, the gradient's length ``norm`` (``used``: after clipping), Adam's
    memory ``moment`` (the root of its summed second-moment estimates) and the length
    of the ``step`` it took; ``held``: the held-out loss at the end; ``text``: the 64
    windows."""
    return _cache("spikes", _spike_build, (_spike_build, spike_text),
                  settings=SPIKE_SETTINGS, seeds=SPIKE_SEEDS, updates=SPIKE_UPDATES,
                  rate=SPIKE_RATE)


def spike_summary(d):
    """Per setting and start: ``worst`` (the largest multiple of its lowest-so-far that
    the loss reaches after update 300), ``count`` (how many times it crosses twice that
    lowest), ``tail`` (mean loss of the last 500 updates). ``main``: the first start
    whose plain-Adam run spikes (the one drawn), and ``at``, the update its loss peaks."""
    # (the first start that spikes is drawn, and the drawing says how many of the three do)
    worst = np.zeros(d["loss"].shape[:2])
    count = np.zeros(d["loss"].shape[:2], int)
    for a in range(worst.shape[0]):
        for b in range(worst.shape[1]):
            events, worst[a, b] = spike_events(d["loss"][a, b])
            count[a, b] = len(events)
    spiked = np.nonzero(count[0] > 0)[0]
    main = int(spiked[0]) if len(spiked) else 0
    line = d["loss"][0, main]
    if len(spiked):
        first = int(spike_events(line)[0][0])
        at = int(first + np.argmax(line[first:first + 40]))
    else:                                   # no spike: its largest rise, such as it is
        at = int(SPIKE_FROM + np.argmax((line / np.minimum.accumulate(line))[SPIKE_FROM:]))
    return dict(worst=worst, count=count, tail=d["loss"][:, :, -500:].mean(-1), main=main,
                at=at)


def loss_spikes():
    from . import realdata as R

    d = spike_data()
    s = spike_summary(d)
    names = [n for n, *_ in SPIKE_SETTINGS]
    m = s["main"]
    ratio = d["step"] / d["used"]
    return Study(
        "A loss spike, inside Adam",
        "Tiny Shakespeare: 64 fixed windows, a one-layer transformer, full-batch Adam, "
        "3,000 updates, 3 starts",
        (
            _curve("Training loss",
                   "The same 64 windows at every update; start %d of each setting." % (
                       SPIKE_SEEDS[m]),
                   d["loss"][:, m], names, all=d["loss"]),
            _curve("Gradient length",
                   "Before clipping; the same runs.", d["norm"][:, m], names,
                   all=d["norm"], used=d["used"]),
            _curve("Step taken per unit of gradient",
                   "Length of Adam's step ÷ length of the gradient it was given.",
                   ratio[:, m], names, all=ratio, moment=d["moment"], step=d["step"]),
            _bars("Worst spike after update 300",
                  "The largest multiple of its lowest-so-far that the loss reaches: mean "
                  "of three starts.",
                  s["worst"].mean(1), names, each=s["worst"], count=s["count"],
                  tail=s["tail"], held=d["held"]),
        ),
        dict(
            seed=SEED, seeds=list(SPIKE_SEEDS), updates=SPIKE_UPDATES,
            learning_rate=SPIKE_RATE, beta1=0.9,
            settings=[dict(name=n, beta2=b, epsilon=e, clip=c) for n, b, e, c in
                      SPIKE_SETTINGS],
            model="one-layer transformer, width 16, 2 heads, feed-forward 24 (tinylm)",
            spike="loss above %g x its lowest so far, after update %d" % (SPIKE_FACTOR,
                                                                          SPIKE_FROM),
            start_drawn=SPIKE_SEEDS[m], spike_drawn_at=s["at"],
            worst_multiple=[[float(v) for v in row] for row in s["worst"]],
            spikes=[[int(v) for v in row] for row in s["count"]],
            mean_loss_last_500=[[float(v) for v in row] for row in s["tail"]],
            held_out_loss=[[float(v) for v in row] for row in d["held"]],
            control="plain Adam (β₂ 0.999, ε 1e-8, no clipping) from the same three starts",
            evidence_kind="actual NumPy updates on collected data",
            dataset=dict(train=R.text_metadata("train"), test=R.text_metadata("test"),
                         windows="64 training windows of 13 characters (seed 17), the "
                                 "same at every update; 256 held-out windows (seed 73)"),
            designs_tried=[
                "learning rate 0.003 (2,500 updates, one start): no spike",
                "learning rate 0.01, five starts of plain Adam: 2 spike within 3,000 "
                "updates; 0.02: 4 of 5; 0.03: 4 of 5, mostly too late to see the recovery",
                "the remedies were first run at learning rate 0.01 with the same values "
                "(β₂ 0.9, ε 1e-3, clip 1) and ε 1e-4, β₂ 0.99; this is the one rerun at "
                "0.02, kept whatever it showed",
                "afterwards, at 0.02 and outside this study: ε 1e-2 (worst rise x1.2, x1.1 and "
                "x23 over the three starts) and clip 0.25 (x4.2, x7.5, x3.5): neither removes "
                "the spikes either",
            ],
            limitations=["64 windows are memorised: the held-out loss is far above the "
                         "training loss for every setting. Whether and when a start "
                         "spikes varies from start to start."],
        ),
    )


STUDIES = {
    "newton-vs-gradient-descent": newton,
    "edge-of-stability": edge_stability,
    "loss-spikes": loss_spikes,
}
