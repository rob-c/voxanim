"""Redesigned experiments for three of the curiosities (97, 98, 99).

The first versions (:mod:`voxanim.next_systems`, :mod:`voxanim.next_systems_real`) could
not show their concept:

* ``hopfield-network`` stored three raw MNIST digits with the Hebbian rule and recalled a
  blend: raw digits share most of their pixels, and the Hebbian rule is built for
  patterns that do not. One recall, one load, no remedy and no measured capacity.
* ``neural-cellular-automata`` trained a 518-weight rule for 550 single-state updates,
  always for 24 steps, from the seed or from a freshly grown digit with a small hole:
  never from a state it had been holding, so nothing made the digit a resting state, and
  a hole grew back as a solid patch.
* ``neural-ode`` learned to denoise points whose noise carried no information about where
  they came from: the best field barely moves anything, so neither the flow nor the
  number of Euler steps could matter.

Here: the same honest Hebbian blend, then the same corrupted digit recalled with the
mean image subtracted, with the pseudo-inverse rule and with the modern (softmax) rule,
and recall counted over 100 random sets of digits at each of 15 loads and 10 corruption
levels; a cellular automaton trained by the Growing-NCA recipe (a pool of its own past
states, some of them damaged) against the same rule trained from the seed only, three
seeds each; and a two-dimensional neural ODE that has to carry the 5s out from between
the 0s and the 8s, trained three times, integrated with 1 to 64 Euler steps against a
Runge-Kutta reference, with its adjoint gradient checked against backpropagation.

Everything the drawings need is recorded here: :mod:`voxanim.mech_curios` computes
nothing for these three but the layout.
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np

from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SEED = 17
SEEDS = (17, 18, 19)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, seed=SEED, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-curios-" + name, settings,
                                                    build).items()}


def _provenance(meta):
    """A split's provenance without its list of example indices."""
    return {k: v for k, v in meta.items() if k != "example_indices"}


def _curve(title, caption, ys, labels, x=None, **more):
    ys = np.atleast_2d(np.asarray(ys, float))
    return Panel(title, caption, "curve",
                 dict(x=np.arange(ys.shape[1]) if x is None else np.asarray(x), ys=ys,
                      labels=list(labels), **more))


def _images(title, caption, values, labels):
    return Panel(title, caption, "images", dict(values=np.asarray(values, float),
                                                labels=list(labels)))


def _points(title, caption, xy, classes):
    return Panel(title, caption, "points", dict(xy=np.asarray(xy, float),
                                                classes=np.asarray(classes)))


# ====================================================================== 97: Hopfield
HOP_N = 784
HOP_RULES = ("hebbian", "centred", "pseudo-inverse", "modern")
HOP_LOADS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
HOP_FLIPS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
HOP_FLIP = 0.20                 # corruption of the load sweep
HOP_FLIP_LOADS = (4, 8)         # loads of the corruption sweep
HOP_TRIALS = 100
HOP_TOLERANCE = 8               # "recalled": at most this many of the 784 pixels wrong (1%)
HOP_BETA = 1.0 / 28             # the modern rule's inverse temperature (the first study's)
HOP_SWEEPS = 20


def hop_weights(rule, patterns):
    """``(W, m)`` for a classical rule: the field on pixel i is ``W[i] @ (state - m)``.

    * ``hebbian``: W = sum of outer products / N (Hopfield 1982).
    * ``centred``: the same after subtracting the stored patterns' mean image ``m``.
    * ``pseudo-inverse``: W = pinv(P) P, the projection onto the stored patterns
      (Personnaz, Guyon and Dreyfus 1985): it removes the patterns' correlations.

    Every rule has a zero diagonal (no pixel votes for itself)."""
    P = np.asarray(patterns, float)
    m = np.zeros(P.shape[1])
    if rule == "hebbian":
        W = P.T @ P / P.shape[1]
    elif rule == "centred":
        m = P.mean(0)
        W = (P - m).T @ (P - m) / P.shape[1]
    elif rule == "pseudo-inverse":
        W = np.linalg.pinv(P) @ P
    else:
        raise ValueError(rule)
    np.fill_diagonal(W, 0)
    return W, m


def hop_settle(W, m, state, rng, sweeps=HOP_SWEEPS, trace=False):
    """Asynchronous recall: every pixel in random order takes the sign of its field,
    sweep after sweep until none changes. A pixel with no field (every stored pattern
    agrees on it, under the centred rule) takes that common value, else stays.

    Returns ``(state, sweeps run)``; with ``trace`` also the first sweep's ``order``,
    ``votes`` and the state after each of its updates (int8)."""
    s = np.asarray(state, float).copy()
    h = W @ (s - m)
    rest = np.sign(m)
    order, votes, states = [], [], []
    ran = 0
    for sweep in range(sweeps):
        changed = 0
        for j in rng.permutation(len(s)):
            v = h[j]
            if abs(v) > 1e-9:
                new = 1.0 if v > 0 else -1.0
            else:
                new = rest[j] if rest[j] != 0 else s[j]
            if trace and sweep == 0:
                order.append(int(j))
                votes.append(float(v))
            if new != s[j]:
                h += W[:, j] * (new - s[j])
                s[j] = new
                changed += 1
            if trace and sweep == 0:
                states.append(s.astype(np.int8))
        ran = sweep + 1
        if not changed:
            break
    if trace:
        return s, ran, np.array(order), np.array(votes), np.array(states)
    return s, ran


def hop_modern(patterns, state, beta=HOP_BETA, steps=3):
    """The modern rule (Ramsauer et al. 2020): score the state against every stored
    pattern, softmax, and take the sign of the weighted sum of the patterns."""
    s = np.asarray(state, float)
    prob = None
    for _ in range(steps):
        z = beta * (patterns @ s)
        prob = np.exp(z - z.max())
        prob /= prob.sum()
        s = np.where(prob @ patterns >= 0, 1.0, -1.0)
    return s, prob


def hop_recall(rule, patterns, start, rng):
    if rule == "modern":
        return hop_modern(patterns, start)[0]
    W, m = hop_weights(rule, patterns)
    return hop_settle(W, m, start, rng)[0]


def _hop_cell(pool, load, flip, cell_seed):
    """Wrong pixels after recall, ``(rule, trial)``: each trial stores ``load`` digits
    drawn afresh, corrupts the first and recalls it with each rule."""
    rng = np.random.default_rng(cell_seed)
    wrong = np.zeros((len(HOP_RULES), HOP_TRIALS), int)
    for t in range(HOP_TRIALS):
        P = pool[rng.choice(len(pool), load, False)]
        start = P[0].copy()
        start[rng.choice(HOP_N, int(round(flip * HOP_N)), False)] *= -1
        for r, rule in enumerate(HOP_RULES):
            wrong[r, t] = int((hop_recall(rule, P, start, rng) != P[0]).sum())
    return wrong


def _hop_build():
    from . import realdata
    from .mnist import load

    x, y, _, _ = load()
    ids = [int(np.flatnonzero(y == i)[0]) for i in (0, 1, 2)]
    patterns = np.where(x[ids] > 0.3, 1.0, -1.0)
    # --- the one recall the animation follows: the first study's, pixel by pixel
    W, _m = hop_weights("hebbian", patterns)
    rng = np.random.default_rng(SEED)
    start = patterns[1].copy()
    start[rng.random(HOP_N) < 0.24] *= -1
    state = start.copy()
    energy = [-0.5 * state @ W @ state]
    order, votes, states, sweeps = [], [], [], [energy[0]]
    for sweep in range(12):
        for j in rng.permutation(HOP_N):
            field = W[j] @ state
            if sweep == 0:
                order.append(int(j))
                votes.append(float(field))
            if field != 0:
                state[j] = 1 if field > 0 else -1
            if sweep == 0:
                states.append(state.astype(np.int8))
                energy.append(-0.5 * state @ W @ state)
        sweeps.append(-0.5 * state @ W @ state)
    out = dict(patterns=patterns, W=W, start=start, order=order, votes=votes,
               states=np.array(states), energy=energy, sweeps=sweeps, final=state,
               memory_energy=[-0.5 * p @ W @ p for p in patterns])
    # --- the same corrupted digit under the three remedies
    for key, rule in (("centred", "centred"), ("pinv", "pseudo-inverse")):
        Wr, mr = hop_weights(rule, patterns)
        final, ran, _, _, trail = hop_settle(Wr, mr, start, np.random.default_rng([SEED, 1]),
                                             trace=True)
        out[key + "_final"], out[key + "_sweeps"], out[key + "_states"] = final, ran, trail
    scores = HOP_BETA * (patterns @ start)
    out["scores"] = scores
    out["prob"] = np.exp(scores - scores.max()) / np.exp(scores - scores.max()).sum()
    out["modern_final"] = hop_modern(patterns, start)[0]      # the first update's weights: prob
    # --- recall counted over random sets of digits
    px, _, meta = realdata.mnist_split(4096, "train", 28, SEED)
    pool = np.where(px > 0.3, 1.0, -1.0)
    out["load_wrong"] = np.array([_hop_cell(pool, k, HOP_FLIP, [SEED, 2, k])
                                  for k in HOP_LOADS])              # (load, rule, trial)
    out["flip_wrong"] = np.array([[_hop_cell(pool, k, f, [SEED, 3, k, i])
                                   for i, f in enumerate(HOP_FLIPS)]
                                  for k in HOP_FLIP_LOADS])         # (load, flip, rule, trial)
    out["pool_indices"] = np.array(meta["example_indices"])
    out["demo_indices"] = np.array(ids)
    return out


def hopfield_data():
    """The Hopfield experiment's arrays (cached)."""
    return _cache("hopfield", _hop_build,
                  (_hop_build, _hop_cell, hop_recall, hop_modern, hop_settle, hop_weights),
                  loads=HOP_LOADS, flips=HOP_FLIPS, trials=HOP_TRIALS, beta=HOP_BETA,
                  tolerance=HOP_TOLERANCE, flip=HOP_FLIP, flip_loads=HOP_FLIP_LOADS)


def hop_rates(wrong):
    """Fraction of trials recalled (last axis: trials)."""
    return (np.asarray(wrong) <= HOP_TOLERANCE).mean(-1)


def hopfield():
    from . import realdata

    d = hopfield_data()
    pat = d["patterns"]
    load_rate = hop_rates(d["load_wrong"]).T                      # (rule, load)
    flip_rate = np.moveaxis(hop_rates(d["flip_wrong"]), 2, 1)     # (load, rule, flip)
    finals = [d["final"], d["centred_final"], d["pinv_final"], d["modern_final"]]
    off = [int((f != pat[1]).sum()) for f in finals]
    largest = {rule: int(max([k for k, r in zip(HOP_LOADS, load_rate[i]) if r >= 0.9],
                             default=0)) for i, rule in enumerate(HOP_RULES)}
    _, _, meta = realdata.mnist_split(4096, "train", 28, SEED)
    pictures = np.vstack([pat, d["start"][None], np.array(finals)]).reshape(-1, 28, 28)
    return Study(
        "Hopfield memory and its capacity",
        "Binarized MNIST digits stored by four rules; recall counted over %d random sets "
        "of digits per load" % HOP_TRIALS,
        (
            _images("Three stored digits, one corrupted, four recalls",
                    "The corrupted 1 recalled by the Hebbian rule (a blend), the centred "
                    "Hebbian rule, the pseudo-inverse rule and the modern softmax rule.",
                    (pictures + 1) / 2,
                    ["zero", "one", "two", "corrupted one", "hebbian", "centred",
                     "pseudo-inverse", "modern"]),
            _curve("Hebbian recall lowers the energy",
                   "Energy after each single-pixel update of the first sweep; it ends below "
                   "every stored digit's own energy.",
                   d["energy"], ["energy"]),
            _curve("Recall against the number of stored digits",
                   "Fraction of %d trials recalled to within %d pixels from %d%% flipped "
                   "pixels; each trial draws its digits afresh."
                   % (HOP_TRIALS, HOP_TOLERANCE, round(100 * HOP_FLIP)),
                   load_rate, HOP_RULES, HOP_LOADS,
                   median_wrong=np.median(d["load_wrong"], -1).T),
            _curve("Recall against corruption, %d digits stored" % HOP_FLIP_LOADS[0],
                   "Fraction of %d trials recalled as more pixels are flipped; ys_load%d "
                   "holds the same sweep with %d stored."
                   % (HOP_TRIALS, HOP_FLIP_LOADS[1], HOP_FLIP_LOADS[1]),
                   flip_rate[0], HOP_RULES, HOP_FLIPS,
                   **{"ys_load%d" % HOP_FLIP_LOADS[1]: flip_rate[1]}),
        ),
        dict(
            seed=SEED, review_only=True, mnist=True,
            source="exact associative-memory rules on binarized MNIST training digits",
            evidence_kind="direct calculation: storage rules and recall, no gradient training",
            dataset=dict(pool=_provenance(meta),
                         demo="the first 0, 1 and 2 of the MNIST training images",
                         preprocessing="pixel > 0.3 is +1 (ink), else -1; 784 pixels"),
            rules={"hebbian": "W = P'P / N", "centred": "the same on P minus its mean image",
                   "pseudo-inverse": "W = pinv(P) P", "modern": "sign(P' softmax(beta P s)), "
                   "beta = 1/28, 3 updates"},
            recalled="at most %d of 784 pixels differ from the stored digit" % HOP_TOLERANCE,
            trials_per_point=HOP_TRIALS,
            demo_pixels_off=dict(zip(HOP_RULES, off)),
            largest_load_recalled_9_in_10=largest,
            load_recall_rate={rule: load_rate[i].tolist() for i, rule in enumerate(HOP_RULES)},
            diagnosis="Raw digits overlap heavily (most pixels are background in all of "
                      "them); the Hebbian rule's capacity assumes uncorrelated patterns, so "
                      "with three digits the stored patterns are not even fixed points.",
            designs_tried=[
                "this design, fixed before the counts were run: four rules, 15 loads at 20% "
                "flipped, success = within 8 pixels",
                "the corruption sweep was first run with 8 digits stored only (20 trials): "
                "Hebbian and centred recalled almost nothing at any level, so a sweep with "
                "4 stored was added; both are recorded (panel 4: ys and ys_load8)",
            ],
        ),
    )


# ====================================================================== 98: cellular automaton
NCA_SIDE, NCA_CHANNELS, NCA_HIDDEN = 14, 8, 32
NCA_UPDATES = 1500
NCA_BATCH, NCA_POOL = 8, 128
NCA_STEPS = (24, 40)             # a training rollout runs a random number of steps in this range
NCA_LR = 8e-3
NCA_FIRE = 0.5                   # each cell updates with this probability at every step
NCA_CUT = 48                     # the step at which the grown digit is damaged
NCA_RUN = 960                    # steps every evaluation runs
NCA_KEEP = 192                   # steps whose full states are kept for the drawn model
NCA_LATE = (240, 480, 960)       # later steps whose ink is kept
NCA_SNAPS = (0, 50, 150, 400, NCA_UPDATES)   # updates whose rule is grown for the drawing
NCA_DAMAGES = ("a 6 by 6 hole", "the right half", "the bottom half")
NCA_MODES = ("pool", "seed only")
NCA_CAP = 1.0                    # recorded errors stop here (a rule that blows up has no bound)
F32 = np.float32


def nca_damage(state, which):
    """``state`` (..., 14, 14, channels) with the cells of damage ``which`` set to zero."""
    s = state.copy()
    if which == 0:
        s[..., 3:9, 3:9, :] = 0
    elif which == 1:
        s[..., :, 7:, :] = 0
    elif which == 2:
        s[..., 7:, :, :] = 0
    return s


def nca_sobel(s):
    """Each channel's left-right and up-down slope over a cell's 3 x 3 neighbourhood
    (Sobel filters; cells beyond the edge count as zero). ``s``: (B, rows, cols, channels).
    The division by 8 is left to the caller."""
    B, R, K, C = s.shape
    p = np.zeros((B, R + 2, K + 2, C), s.dtype)
    p[:, 1:-1, 1:-1] = s
    v = p[:, :-2] + p[:, 2:]
    v += 2 * p[:, 1:-1]
    h = p[:, :, :-2] + p[:, :, 2:]
    h += 2 * p[:, :, 1:-1]
    return v[:, :, 2:] - v[:, :, :-2], h[:, 2:] - h[:, :-2]


def nca_sense(s):
    """What every cell senses: its own numbers and their two slopes, (..., 3 x channels)."""
    C = s.shape[-1]
    flat = s.reshape((-1,) + s.shape[-3:])
    f = np.empty(flat.shape[:-1] + (3 * C,), s.dtype)
    f[..., :C] = flat
    f[..., C:2 * C], f[..., 2 * C:] = nca_sobel(flat)
    f[..., C:] *= 0.125
    return f.reshape(s.shape[:-1] + (3 * C,))


def nca_step(P, s, mask):
    """One step of ``M`` rules at once. ``P``: A (M, 3C, H), b (M, 1, H), B (M, H, C);
    ``s``: (M, B, rows, cols, C); ``mask`` (broadcast to (M, B, rows, cols)): which cells
    update. Returns the next state, what was sensed (M, cells, 3C) and the hidden layer."""
    M = s.shape[0]
    f = nca_sense(s).reshape(M, -1, 3 * s.shape[-1])
    h = np.maximum(f @ P["A"] + P["b"], 0)
    return s + (h @ P["B"]).reshape(s.shape) * mask[..., None], f, h


def nca_one(P, m):
    """Rule ``m`` of a stack, as a stack of one."""
    return {k: P[k][m:m + 1] for k in ("A", "b", "B")}


def nca_loss(P, s, masks, target):
    """Mean squared error of the ink channel after ``len(masks)`` steps, per rule and
    per example, its gradient for every rule, and the final states (backpropagation
    through the whole rollout; the masks are constants)."""
    rec = []
    for mask in masks:
        nxt, f, h = nca_step(P, s, mask)
        rec.append((f, h, mask))
        s = nxt
    err = s[..., 0] - target
    per = (err ** 2).mean((2, 3))
    g = np.zeros_like(s)
    g[..., 0] = 2 * err / err[0].size
    G = {k: np.zeros_like(v) for k, v in P.items()}
    M, C = s.shape[0], s.shape[-1]
    for f, h, mask in rec[::-1]:
        dd = (g * mask[..., None]).reshape(M, -1, C)
        G["B"] += h.transpose(0, 2, 1) @ dd
        dh = (dd @ P["B"].transpose(0, 2, 1)) * (h > 0)
        G["A"] += f.transpose(0, 2, 1) @ dh
        G["b"] += dh.sum(1, keepdims=True)
        df = (dh @ P["A"].transpose(0, 2, 1)).reshape((-1,) + s.shape[2:4] + (3 * C,))
        back = df[..., :C] - 0.125 * (nca_sobel(df[..., C:2 * C])[0]
                                      + nca_sobel(df[..., 2 * C:])[1])
        g += back.reshape(s.shape)
    return per, G, s


def nca_seed(*lead):
    """All-zero cells but the middle one, whose hidden channels are 1."""
    s = np.zeros(lead + (NCA_SIDE, NCA_SIDE, NCA_CHANNELS), F32)
    s[..., NCA_SIDE // 2, NCA_SIDE // 2, 1:] = 1.0
    return s


def nca_disc(rng):
    """A random disc of cells (radius 2 to 4.5): the damage used in training."""
    r = rng.uniform(2.0, 4.5)
    c = rng.uniform(2, NCA_SIDE - 2, 2)
    yy, xx = np.mgrid[:NCA_SIDE, :NCA_SIDE]
    return (yy - c[0]) ** 2 + (xx - c[1]) ** 2 <= r * r


def nca_target():
    from .mnist import load

    x, y, _, _ = load()
    index = int(np.flatnonzero(y == 0)[0])
    return x[index].reshape(14, 2, 14, 2).mean((1, 3)), index


def nca_eval_masks(steps):
    return (np.random.default_rng([SEED, 7]).random((steps, NCA_SIDE, NCA_SIDE))
            < NCA_FIRE).astype(F32)


def nca_train(target, updates=NCA_UPDATES, seeds=SEEDS, snaps=NCA_SNAPS, snap_steps=NCA_CUT):
    """Train ``2 x len(seeds)`` rules side by side (same code, one stacked array each):
    the first ``len(seeds)`` by the Growing-NCA recipe (Mordvintsev et al. 2020) - every
    batch is drawn from a pool of the rule's own past outputs, the worst of the batch is
    replaced by the seed, two are damaged, and the outputs go back to the pool - and the
    rest from the seed only. Adam on per-array normalised gradients, the step cut to a
    tenth for the last fifth. Returns the weights, the loss curves (rule, update) and
    what the first rule grows at ``snaps``."""
    target = np.asarray(target, F32)
    n = len(seeds)
    M = 2 * n
    C, H = NCA_CHANNELS, NCA_HIDDEN
    rngs = [np.random.default_rng([seed, mode]) for mode in (0, 1) for seed in seeds]
    P = dict(A=np.stack([r.normal(0, np.sqrt(2.0 / (3 * C)), (3 * C, H)) for r in rngs])
             .astype(F32), b=np.zeros((M, 1, H), F32), B=np.zeros((M, H, C), F32))
    opt = Adam(P, NCA_LR)
    schedule = np.random.default_rng([SEED, 5])
    pool = nca_seed(n, NCA_POOL)
    masks_eval = nca_eval_masks(snap_steps)
    losses = np.zeros((M, updates))
    grown = []

    def snapshot():
        s = nca_seed(1, 1)
        for mask in masks_eval:
            s = nca_step(nca_one(P, 0), s, mask)[0]
        grown.append(s[0, 0, ..., 0].copy())

    for k in range(updates):
        if k in snaps:
            snapshot()
        s = nca_seed(M, NCA_BATCH)
        picks = []
        for m in range(n):
            ids = rngs[m].choice(NCA_POOL, NCA_BATCH, False)
            batch = pool[m, ids]
            worst_first = np.argsort(-((batch[..., 0] - target) ** 2).mean((1, 2)))
            ids, batch = ids[worst_first], batch[worst_first]
            batch[0] = nca_seed()
            for j in (-1, -2):
                batch[j][nca_disc(rngs[m])] = 0
            s[m] = batch
            picks.append(ids)
        steps = int(schedule.integers(NCA_STEPS[0], NCA_STEPS[1] + 1))
        masks = np.stack([(r.random((steps, NCA_BATCH, NCA_SIDE, NCA_SIDE)) < NCA_FIRE)
                          for r in rngs], 1).astype(F32)
        per, G, final = nca_loss(P, s, masks, target)
        for key, g in G.items():
            G[key] = g / (np.sqrt((g * g).sum((1, 2), keepdims=True)) + 1e-8)
        opt.step(P, G, NCA_LR * (0.1 if k >= 0.8 * updates else 1.0))
        for m in range(n):
            pool[m, picks[m]] = final[m]
        losses[:, k] = per.mean(1)
    if updates in snaps:
        snapshot()
    return P, losses, np.array(grown)


def nca_rollouts(P, target, steps=NCA_RUN, keep=NCA_KEEP):
    """Every rule grown from the seed four times with the same random firing: left
    alone, and with each of :data:`NCA_DAMAGES` at step :data:`NCA_CUT`. Returns the
    error against the target (rule, run, step) - capped at :data:`NCA_CAP`, where a
    rule has blown up - the first ``keep`` states (rule, run, step, ...) of rules 0 and
    ``M/2`` (the drawn ones), and their ink at :data:`NCA_LATE`."""
    M = P["A"].shape[0]
    masks = nca_eval_masks(steps)
    s = nca_seed(M, 4)
    drawn = [0, M // 2]
    errors = np.zeros((M, 4, steps + 1))
    states = np.zeros((2, 4, keep + 1) + s.shape[2:], F32)
    late = np.zeros((2, 4, len(NCA_LATE), NCA_SIDE, NCA_SIDE), F32)
    for t in range(steps + 1):
        if t == NCA_CUT:
            for run in (1, 2, 3):
                s[:, run] = nca_damage(s[:, run], run - 1)
        errors[:, :, t] = np.minimum(((s[..., 0] - target) ** 2).mean((2, 3)), NCA_CAP)
        if t <= keep:
            states[:, :, t] = s[drawn]
        if t in NCA_LATE:
            late[:, :, NCA_LATE.index(t)] = s[drawn][..., 0]
        if t < steps:
            # a rule that has blown up is held at a large finite value, not at infinity
            s = np.clip(nca_step(P, s, masks[t])[0], -1e4, 1e4)
    return errors, states, late


def _nca_build():
    target, index = nca_target()
    P, losses, grown = nca_train(target)
    errors, states, late = nca_rollouts(P, target.astype(F32))
    return dict(target=target, target_index=index, A=P["A"], b=P["b"], B=P["B"], loss=losses,
                snaps=grown, errors=errors, states=states, late=late)


def nca_data():
    """The cellular-automaton experiment's arrays (cached)."""
    return _cache("cellular", _nca_build,
                  (_nca_build, nca_rollouts, nca_train, nca_eval_masks, nca_target, nca_disc,
                   nca_seed, nca_loss, nca_step, nca_sense, nca_sobel, nca_damage),
                  updates=NCA_UPDATES, channels=NCA_CHANNELS, hidden=NCA_HIDDEN, lr=NCA_LR,
                  batch=NCA_BATCH, pool=NCA_POOL, steps=NCA_STEPS, seeds=SEEDS, run=NCA_RUN)


def cellular():
    from .mnist import SHA256, URL

    d = nca_data()
    n = len(SEEDS)
    e = d["errors"]                                   # (rule, run, step)
    cut, later, end = NCA_CUT, 2 * NCA_CUT, NCA_RUN
    ink = d["states"][0, ..., 0]            # the drawn pool-trained rule: (run, step, 14, 14)
    summary = {}
    for mode, rows in zip(NCA_MODES, (slice(0, n), slice(n, 2 * n))):
        summary[mode] = dict(
            grown_error_step_47=e[rows, 0, cut - 1].tolist(),
            left_alone_error_step_960=e[rows, 0, end].tolist(),
            just_cut_error=e[rows, 1:, cut].tolist(),
            error_48_steps_after_cut=e[rows, 1:, later].tolist(),
            error_192_steps_after_cut=e[rows, 1:, cut + 192].tolist(),
        )
    steps = np.arange(NCA_KEEP + 1)
    return Study(
        "A cellular automaton that grows, holds and regrows a digit",
        "One MNIST 0 as the target; three seeds trained with a pool of past states and "
        "damage, three from the seed only",
        (
            _images("Growing from one seed cell",
                    "The ink channel of the first pool-trained rule at steps 0, 4, 8, 16, "
                    "24 and 47, then the target.",
                    np.concatenate([ink[0, [0, 4, 8, 16, 24, 47]], d["target"][None]]),
                    ["step 0", "step 4", "step 8", "step 16", "step 24", "step 47", "target"]),
            _curve("Training loss, six rules",
                   "Error after a rollout of 24 to 40 steps, per update: three seeds with "
                   "the pool and damage, three from the seed only (whose task is easier).",
                   d["loss"], ["%s, seed %d" % (mode, s) for mode in NCA_MODES for s in SEEDS]),
            _images("Three damages and what grows back",
                    "The first pool-trained rule just after each cut at step %d, then 12, "
                    "48 and 144 steps later." % cut,
                    ink[1:][:, [cut, cut + 12, cut + 48, cut + 144]].reshape(-1, 14, 14),
                    ["%s, %s" % (name, when) for name in NCA_DAMAGES
                     for when in ("cut", "+12", "+48", "+144")]),
            _curve("Error against the target, step by step",
                   "The first pool-trained rule left alone and after each damage, and the "
                   "first seed-only rule left alone (capped at %g); all_errors holds every "
                   "rule to step %d." % (NCA_CAP, NCA_RUN),
                   np.vstack([e[0, :, :NCA_KEEP + 1], e[n, :1, :NCA_KEEP + 1]]),
                   ["pool: left alone"] + ["pool: " + name for name in NCA_DAMAGES]
                   + ["seed only: left alone"], steps, all_errors=e),
        ),
        dict(
            seed=SEED, seeds=list(SEEDS), review_only=True, mnist=True,
            source="NumPy training of a neural cellular automaton on one MNIST digit",
            evidence_kind="actual NumPy training on MNIST",
            dataset=dict(dataset="MNIST", source_url=URL, raw_sha256=SHA256,
                         example="training image %d (the first 0)" % int(d["target_index"]),
                         preprocessing="pixels / 255; 2 x 2 mean pooling to 14 x 14"),
            model="%d channels per cell; senses itself and two Sobel slopes; %d hidden ReLU "
                  "units; residual update, each cell firing with probability %.1f"
                  % (NCA_CHANNELS, NCA_HIDDEN, NCA_FIRE),
            training=dict(updates=NCA_UPDATES, batch=NCA_BATCH, pool=NCA_POOL,
                          rollout_steps=list(NCA_STEPS), optimizer="Adam", lr=NCA_LR,
                          damage="two of each batch of eight lose a random disc of cells "
                                 "(radius 2 to 4.5)",
                          control="the same rule and updates, every rollout from the seed"),
            evaluation=dict(steps=NCA_RUN, cut_step=NCA_CUT, damages=list(NCA_DAMAGES),
                            note="the half cuts are larger than any damage seen in training; "
                                 "errors are capped at %g" % NCA_CAP),
            results=summary,
            scope="one image: nothing here says the rule would grow another digit",
            diagnosis="The first study trained a 518-weight rule for 550 single-state "
                      "updates of exactly 24 steps, from the seed or from a freshly grown "
                      "digit with a 4 by 4 hole, and never from a state the rule had been "
                      "holding: nothing made the grown digit a resting state, so its error "
                      "crept up and a 6 by 6 hole was filled with ink everywhere.",
            designs_tried=[
                "Growing-NCA recipe with 12 channels and 48 hidden units, learning rate "
                "0.002, 800 to 1500 updates, one seed at a time: regrowth good for some "
                "seeds, still converging for others",
                "learning rates 0.004 and 0.008 on two seeds at 1200 updates: 0.008 converged "
                "fastest and was kept; batches of 4 for twice the updates were worse",
                "8 channels and 32 hidden units (and 10 and 40) at 0.008 for 1500 updates: "
                "the smaller rule regrew on all three seeds and fits the time limit",
                "this design: 8 channels, 32 hidden, learning rate 0.008, %d updates, three "
                "seeds, with the seed-only control; the damages and the steps evaluated "
                "were fixed before any rule was trained" % NCA_UPDATES,
            ],
        ),
    )


# ====================================================================== 99: neural ODE
ODE_DIGITS = (0, 5, 8)           # the task: is it a 5?
ODE_TRAIN, ODE_TEST = 3000, 1000
ODE_HIDDEN = 32
ODE_UPDATES, ODE_BATCH, ODE_LR = 2000, 256, 0.01
ODE_TRAIN_STEPS = 64             # Euler steps the field is trained with
ODE_STEPS = (1, 2, 4, 8, 16, 32, 64)
ODE_REFERENCE = 256              # Runge-Kutta steps of the reference solution
ODE_ADJOINT_STEPS = (4, 8, 16, 32, 64, 128, 256, 512)
ODE_ADJOINT_POINTS = 256
ODE_DRAWN = 240                  # held-out points whose paths are kept for the drawing
ODE_GRID = 9                     # lattice lines each way, carried by the flow
ODE_KEYS = ("W", "b", "U", "v", "w", "c")


def ode_field(P, z):
    """The learned velocity at ``z`` and the hidden layer that produced it."""
    h = np.tanh(z @ P["W"] + P["b"])
    return h @ P["U"] + P["v"], h


def ode_euler(P, z, steps):
    """The states of ``steps`` Euler steps over one unit of time: (steps + 1, n, 2)."""
    out = [z]
    for _ in range(steps):
        z = z + ode_field(P, z)[0] / steps
        out.append(z)
    return np.array(out)


def ode_resnet(layers, z):
    """A residual network: every layer adds its own small network's output to its
    input. ``layers``: one dict of weights (W, b, U, v) per layer, plus the step."""
    for layer in layers:
        z = z + (np.tanh(z @ layer["W"] + layer["b"]) @ layer["U"] + layer["v"]) / len(layers)
    return z


def ode_rk4(P, z, steps=ODE_REFERENCE):
    """The end of the flow by fourth-order Runge-Kutta: the reference solution."""
    h = 1.0 / steps
    for _ in range(steps):
        k1 = ode_field(P, z)[0]
        k2 = ode_field(P, z + 0.5 * h * k1)[0]
        k3 = ode_field(P, z + 0.5 * h * k2)[0]
        k4 = ode_field(P, z + h * k3)[0]
        z = z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return z


def ode_logit(P, z):
    return z @ P["w"] + P["c"]


def _ode_head(P, z, y):
    p = 1.0 / (1.0 + np.exp(-ode_logit(P, z)))
    loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return loss, (p - y) / len(y)


def ode_backprop(P, x, y, steps):
    """Cross-entropy of "is it a 5" after ``steps`` Euler steps and its gradient by
    backpropagation through the stored states."""
    zs, hs, z = [x], [], x
    for _ in range(steps):
        f, h = ode_field(P, z)
        z = z + f / steps
        zs.append(z)
        hs.append(h)
    loss, dl = _ode_head(P, z, y)
    G = {k: np.zeros_like(np.asarray(P[k], float)) for k in ODE_KEYS}
    G["w"], G["c"] = z.T @ dl, dl.sum()
    a = dl[:, None] * P["w"][None]
    for k in range(steps - 1, -1, -1):
        h = hs[k]
        G["U"] += h.T @ a / steps
        G["v"] += a.sum(0) / steps
        dh = (a @ P["U"].T) * (1 - h * h) / steps
        G["W"] += zs[k].T @ dh
        G["b"] += dh.sum(0)
        a = a + dh @ P["W"].T
    return loss, G


def ode_adjoint(P, x, y, steps):
    """The same gradient by the adjoint method (Chen et al. 2018): run forward keeping
    only the current state, then step the state, its adjoint and the gradient backwards
    together (Euler, ``steps`` steps). Returns the gradient and the start it arrives at."""
    z = x
    for _ in range(steps):
        z = z + ode_field(P, z)[0] / steps
    _, dl = _ode_head(P, z, y)
    G = {k: np.zeros_like(np.asarray(P[k], float)) for k in ODE_KEYS}
    G["w"], G["c"] = z.T @ dl, dl.sum()
    a = dl[:, None] * P["w"][None]
    for _ in range(steps):
        f, h = ode_field(P, z)
        dh = (a @ P["U"].T) * (1 - h * h) / steps
        G["U"] += h.T @ a / steps
        G["v"] += a.sum(0) / steps
        G["W"] += z.T @ dh
        G["b"] += dh.sum(0)
        a = a + dh @ P["W"].T
        z = z - f / steps
    return G, z


def ode_pack(G):
    return np.concatenate([np.ravel(G[k]) for k in ODE_KEYS])


def ode_accuracy(P, end, y):
    return float(((ode_logit(P, end) > 0) == (y > 0.5)).mean())


def logistic(x, y, iterations=25):
    """The best straight line for "is it a 5" (logistic regression by Newton's method):
    weights (2,) and offset."""
    X = np.c_[x, np.ones(len(x))]
    w = np.zeros(3)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(3)
        w = w - np.linalg.solve(H, X.T @ (p - y))
    return w[:2], w[2]


def ode_plane(classes=ODE_DIGITS, n_train=ODE_TRAIN, n_test=ODE_TEST, split="test"):
    """Digits of ``classes`` as points: their two main PCA coordinates (mean, axes and
    scale from the training images only). Returns train and ``split`` coordinates,
    digit labels and the two splits' provenance."""
    from . import realdata

    x, y, meta = realdata.mnist_split(n_train, "train", 28, SEED, classes)
    xt, yt, meta_t = realdata.mnist_split(n_test, split, 28, SEED, classes)
    mean = x.mean(0)
    axes = np.linalg.svd(x - mean, full_matrices=False)[2][:2]
    scale = float(np.std((x - mean) @ axes.T))
    return dict(train=(x - mean) @ axes.T / scale, test=(xt - mean) @ axes.T / scale,
                train_digits=y, test_digits=yt, test_pixels=xt, meta=(meta, meta_t))


def ode_task_scan(n_train=1200, n_val=600):
    """How the three digits were chosen, on the validation split (never the test split):
    for every three digits with both odd and even among them, tell odd from even in their
    PCA plane with a straight line and with 15 nearest neighbours. Returns
    ``[(gap, digits, line accuracy, neighbour accuracy)]``, largest gap first: the task
    where a curved boundary matters most."""
    import itertools

    rows = []
    for classes in itertools.combinations(range(10), 3):
        if len({c % 2 for c in classes}) < 2:
            continue
        d = ode_plane(classes, n_train, n_val, "validation")
        a, b = d["train"], d["test"]
        ya, yb = d["train_digits"] % 2, d["test_digits"] % 2
        w, c = logistic(a, ya.astype(float))
        line = float((((b @ w + c) > 0) == (yb == 1)).mean())
        near = np.argsort(((b[:, None] - a[None]) ** 2).sum(-1), 1)[:, :15]
        knn = float(((ya[near].mean(1) > 0.5) == (yb == 1)).mean())
        rows.append((knn - line, classes, line, knn))
    return sorted(rows, reverse=True)


def ode_train(x, y, seed):
    """One field and its straight-line readout, trained with Adam on mini-batches
    through :data:`ODE_TRAIN_STEPS` Euler steps. Returns the weights and the loss curve."""
    rng = np.random.default_rng(seed)
    H = ODE_HIDDEN
    P = dict(W=rng.normal(0, 1.0, (2, H)), b=rng.normal(0, 1.0, H),
             U=rng.normal(0, 0.1, (H, 2)), v=np.zeros(2), w=rng.normal(0, 0.1, 2),
             c=np.zeros(()))
    opt = Adam(P, ODE_LR)
    losses = []
    for _ in range(ODE_UPDATES):
        ids = rng.choice(len(x), ODE_BATCH, False)
        loss, G = ode_backprop(P, x[ids], y[ids], ODE_TRAIN_STEPS)
        losses.append(loss)
        opt.step(P, G)
    return P, np.array(losses)


def _ode_build():
    d = ode_plane()
    x, xt = d["train"], d["test"]
    y, yt = (d["train_digits"] == 5).astype(float), (d["test_digits"] == 5).astype(float)
    line_w, line_c = logistic(x, y)
    out = dict(train=x, test=xt, train_digits=d["train_digits"], test_digits=d["test_digits"],
               pixels=d["test_pixels"][:ODE_DRAWN], line_w=line_w, line_c=line_c,
               line_accuracy=float((((xt @ line_w + line_c) > 0) == (yt > 0.5)).mean()),
               line_train_accuracy=float((((x @ line_w + line_c) > 0) == (y > 0.5)).mean()))
    near = np.argsort(((xt[:, None] - x[None]) ** 2).sum(-1), 1)[:, :15]
    out["neighbour_accuracy"] = float(((y[near].mean(1) > 0.5) == (yt > 0.5)).mean())
    models, losses, accuracy, gap, ref_acc, ref_ends, train_acc = [], [], [], [], [], [], []
    for seed in SEEDS:
        P, loss = ode_train(x, y, seed)
        ref = ode_rk4(P, xt)
        ends = [ode_euler(P, xt, n)[-1] for n in ODE_STEPS]
        accuracy.append([ode_accuracy(P, e, yt) for e in ends])
        gap.append([float(np.linalg.norm(e - ref, axis=1).mean()) for e in ends])
        ref_acc.append(ode_accuracy(P, ref, yt))
        train_acc.append(ode_accuracy(P, ode_rk4(P, x), y))
        ref_ends.append(ref)
        models.append(P)
        losses.append(loss)
    for k in ODE_KEYS:
        out[k] = np.array([P[k] for P in models])
    out.update(loss=losses, accuracy=accuracy, gap=gap, reference_accuracy=ref_acc,
               reference_train_accuracy=train_acc, reference_ends=ref_ends,
               moved=[float(np.linalg.norm(r - xt, axis=1).mean()) for r in ref_ends])
    # --- the first seed's field, for the drawing
    P = models[0]
    drawn = xt[:ODE_DRAWN]
    for n in ODE_STEPS:
        out["path_%d" % n] = ode_euler(P, drawn, n)
    lo, hi = xt.min(0) - 0.2, xt.max(0) + 0.2
    lines = []
    s = np.linspace(0, 1, 33)
    for g in np.linspace(0, 1, ODE_GRID):
        lines.append(np.c_[np.full(33, lo[0] + g * (hi[0] - lo[0])), lo[1] + s * (hi[1] - lo[1])])
        lines.append(np.c_[lo[0] + s * (hi[0] - lo[0]), np.full(33, lo[1] + g * (hi[1] - lo[1]))])
    flowed = ode_euler(P, np.array(lines).reshape(-1, 2), ODE_TRAIN_STEPS)[::4]
    out["grid"] = flowed.reshape(len(flowed), len(lines), 33, 2)
    # a residual network with one copy of the weights per layer is that Euler run
    layers = [{k: P[k].copy() for k in ("W", "b", "U", "v")} for _ in range(16)]
    out["resnet_difference"] = float(
        np.abs(ode_resnet(layers, xt) - ode_euler(P, xt, 16)[-1]).max())
    # --- the adjoint against backpropagation, same step count each
    xa, ya = xt[:ODE_ADJOINT_POINTS], yt[:ODE_ADJOINT_POINTS]
    errors, misses = [], []
    for n in ODE_ADJOINT_STEPS:
        exact = ode_pack(ode_backprop(P, xa, ya, n)[1])
        approx, start = ode_adjoint(P, xa, ya, n)
        errors.append(float(np.linalg.norm(ode_pack(approx) - exact) / np.linalg.norm(exact)))
        misses.append(float(np.linalg.norm(start - xa, axis=1).mean()))
    out.update(adjoint_error=errors, adjoint_miss=misses)
    # the backward pass of the drawn points at 16 steps: where the adjoint thinks they were
    back = [out["path_16"][-1]]
    for _ in range(16):
        back.append(back[-1] - ode_field(P, back[-1])[0] / 16)
    out["back_16"] = np.array(back)
    out["train_indices"] = np.array(d["meta"][0]["example_indices"])
    out["test_indices"] = np.array(d["meta"][1]["example_indices"])
    return out


def ode_data():
    """The neural-ODE experiment's arrays (cached)."""
    return _cache("neural-ode", _ode_build,
                  (_ode_build, ode_train, ode_plane, logistic, ode_accuracy, ode_pack,
                   ode_adjoint, ode_backprop, _ode_head, ode_logit, ode_rk4, ode_resnet,
                   ode_euler, ode_field),
                  digits=ODE_DIGITS, train=ODE_TRAIN, test=ODE_TEST, hidden=ODE_HIDDEN,
                  updates=ODE_UPDATES, batch=ODE_BATCH, lr=ODE_LR, steps=ODE_STEPS,
                  train_steps=ODE_TRAIN_STEPS, seeds=SEEDS, adjoint=ODE_ADJOINT_STEPS)


def ode_model(d, seed_index=0):
    return {k: d[k][seed_index] for k in ODE_KEYS}


def neural_ode():
    d = ode_data()
    meta, meta_t = ode_plane()["meta"]
    acc = np.asarray(d["accuracy"], float)                      # (seed, steps)
    line = float(d["line_accuracy"])
    ref = np.asarray(d["reference_accuracy"], float)
    points = len(d["test"])
    numbers = d["W"][0].size + d["b"][0].size + d["U"][0].size + d["v"][0].size + 3
    stored = [(n + 1) * points * 2 for n in ODE_ADJOINT_STEPS]
    return Study(
        "A residual network as steps along a learned flow",
        "MNIST 0s, 5s and 8s in their PCA plane: a learned field carries the 5s out from "
        "between the others; three seeds, held-out accuracy",
        (
            _points("Held-out digits before and after the flow",
                    "The first %d test points at the start and at the end of the first "
                    "seed's flow (64 Euler steps); classes are the digits, +10 at the end."
                    % ODE_DRAWN,
                    np.vstack([d["path_64"][0], d["path_64"][-1]]),
                    np.concatenate([d["test_digits"][:ODE_DRAWN],
                                    d["test_digits"][:ODE_DRAWN] + 10])),
            _curve("Training the field",
                   "Cross-entropy of \"is it a 5\" per update, through 64 Euler steps; one "
                   "row per seed.",
                   d["loss"], ["seed %d" % s for s in SEEDS]),
            _curve("Held-out accuracy against the number of Euler steps",
                   "The same weights taken in 1 to 64 steps, one row per seed; reference "
                   "holds the Runge-Kutta solution's accuracy, line the best straight "
                   "line's, gap the mean distance of the end points from the reference.",
                   acc, ["seed %d" % s for s in SEEDS], ODE_STEPS,
                   reference=ref, line=np.array([line]), gap=np.asarray(d["gap"], float)),
            _curve("The adjoint gradient against backpropagation",
                   "Relative error of the adjoint method's gradient, both with the same "
                   "number of Euler steps; stored holds the numbers backpropagation keeps, "
                   "miss how far the backward pass lands from the true start.",
                   d["adjoint_error"], ["relative gradient error"], ODE_ADJOINT_STEPS,
                   stored=np.array(stored), kept=np.array([2 * points * 2]),
                   miss=np.asarray(d["adjoint_miss"], float)),
        ),
        dict(
            seed=SEED, seeds=list(SEEDS), review_only=True, mnist=True,
            source="NumPy training of a two-dimensional neural ODE on MNIST PCA coordinates",
            evidence_kind="actual NumPy training on MNIST",
            dataset=dict(train=_provenance(meta), test=_provenance(meta_t),
                         digits=list(ODE_DIGITS), task="is the digit a 5?",
                         embedding="two main PCA axes of the 3000 training images; one "
                                   "scale for both axes"),
            model="dz/dt = U tanh(W z + b) + v in the plane (%d hidden units, %d numbers "
                  "with the straight-line readout), from t = 0 to 1" % (ODE_HIDDEN, numbers),
            training=dict(updates=ODE_UPDATES, batch=ODE_BATCH, optimizer="Adam", lr=ODE_LR,
                          euler_steps=ODE_TRAIN_STEPS),
            control="logistic regression on the same two coordinates (the flow switched off)",
            heldout_accuracy=dict(
                straight_line=line, never_answering_5=float((d["test_digits"] != 5).mean()),
                nearest_neighbours_15=float(d["neighbour_accuracy"]),
                flow_reference=ref.tolist(),
                by_euler_steps={str(n): acc[:, i].tolist() for i, n in enumerate(ODE_STEPS)}),
            mean_distance_moved=np.asarray(d["moved"], float).tolist(),
            resnet_equals_euler_max_difference=float(d["resnet_difference"]),
            adjoint=dict(steps=list(ODE_ADJOINT_STEPS),
                         relative_error=np.asarray(d["adjoint_error"], float).tolist(),
                         points=ODE_ADJOINT_POINTS),
            diagnosis="The first study asked the field to undo noise that says nothing "
                      "about where a point came from: the best it can do is nudge points "
                      "towards the crowd, so the flow was short and the step count "
                      "irrelevant.",
            designs_tried=[
                "the three digits were chosen on the validation split as the odd-against-"
                "even task whose 2-D PCA plane most needs a curved boundary "
                "(ode_task_scan: 0, 5, 8; the test split was not used)",
                "exploratory, seeds 0-2: a third (augmented) coordinate, trained with 16 "
                "Euler steps: 0.79-0.81 held out, little dependence on step count",
                "exploratory, seeds 0-2: the plane alone trained with 16 Euler steps: 0.79 "
                "at 16 steps but 0.67-0.75 when the same weights were integrated finely - "
                "the network was not following its own equation",
                "this design: the plane alone, trained with 64 Euler steps, seeds 17-19",
            ],
        ),
    )


STUDIES = {
    "hopfield-network": hopfield,
    "neural-cellular-automata": cellular,
    "neural-ode": neural_ode,
}
