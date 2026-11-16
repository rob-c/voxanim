"""The redesigned reinforcement-learning experiments (animations 92, 93, 94, 96).

The first versions (:mod:`voxanim.next_systems_real`) ran on a two-view digit task whose
second view did not depend on the action, with a linear model and one seed. Nothing in it
was sequential, so replay, target networks, critics and shaping had nothing to act on, and
their measured effects were null or one-seed noise. These replace them with a task in which
actions decide what is seen next, a non-linear network, and five seeds per arm.

**The glimpse game.** A real MNIST digit (pooled to 14 x 14) starts fully covered. Its four
7 x 7 quadrants can be uncovered one at a time, and the digit named. Every observation is
the real image with the still-covered quadrants blanked, plus four bits saying which
quadrants are uncovered and how much time is left. Two versions:

* *look, look, name* (:func:`rollout`; policy gradient and actor-critic): two looks, then a
  name. The reward comes only at the end: 1 point for an answer, 2 when it is right - all
  positive, which is what makes a baseline matter.
* *free* (:func:`q_run`; DQN and reward shaping): each step is a look or a name; a name ends
  the episode (reward 1 when right), and so does the sixth step without one (reward 0).
  Later rewards count 0.9 per step, so looking costs time.

Training digits (4,096) and held-out test digits (1,000) are disjoint; every number shown
is a held-out number unless it is called a training reward. Plain numpy, seeded.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import realdata as R
from .cache import cached_npz
from .next_systems import _bars, _curve, _matrix, softmax
from .optim import Adam
from .study import Study

SIDE, PIX = 14, 196
LOOKS, ACTIONS = 4, 14                       # actions 0-3 uncover a quadrant, 4-13 name a digit
IN = PIX + LOOKS + 1                         # pixels, uncovered bits, time used
HIDDEN = 64
SEEDS = 5


def _quadrants():
    q = np.zeros((LOOKS, SIDE, SIDE))
    q[0, :7, :7] = q[1, :7, 7:] = q[2, 7:, :7] = q[3, 7:, 7:] = 1
    return q.reshape(LOOKS, PIX)


QUAD = _quadrants()                          # (4, 196): the pixels of each quadrant


@lru_cache(maxsize=1)
def digits():
    """Training and held-out digits: ``x, y, xt, yt, provenance``."""
    x, y, train = R.mnist_split(4096, "train", side=SIDE, seed=17)
    xt, yt, test = R.mnist_split(1000, "test", side=SIDE, seed=73)
    return x, y, xt, yt, dict(train={k: v for k, v in train.items() if k != "example_indices"},
                              test={k: v for k, v in test.items() if k != "example_indices"})


def cached(name, build, **settings):
    source = R.source_hash("study_rl.py", "realdata.py", "optim.py")
    return {k: np.asarray(v) for k, v in cached_npz(
        "study-rl-" + name, dict(version=1, source=source, **settings), build).items()}


# ---------------------------------------------------------------------- the network
def net(rng, outputs):
    """A one-hidden-layer tanh network: 201 inputs, 64 hidden units."""
    return dict(W=rng.normal(0, 1 / np.sqrt(IN), (IN, HIDDEN)), b=np.zeros(HIDDEN),
                U=rng.normal(0, 0.1 / np.sqrt(HIDDEN), (HIDDEN, outputs)), c=np.zeros(outputs))


def forward(P, x):
    h = np.tanh(x @ P["W"] + P["b"])
    return h, h @ P["U"] + P["c"]


def backward(P, x, h, dout):
    """Gradients of a loss whose derivative with respect to the outputs is ``dout``."""
    dh = (dout @ P["U"].T) * (1 - h * h)
    return dict(W=x.T @ dh, b=dh.sum(0), U=h.T @ dout, c=dout.sum(0))


def observe(images, mask, used=0.0):
    """What the player is given: the image with covered quadrants blank, which quadrants
    are uncovered, and the fraction of the time limit used."""
    time = np.broadcast_to(np.asarray(used, float), (len(images),))[:, None]
    return np.concatenate([images * (mask @ QUAD), mask, time], 1)


def flat(g):
    return np.concatenate([g[k].ravel() for k in ("W", "b", "U", "c")])


# ---------------------------------------------------------------------- look, look, name
POINTS = (1.0, 2.0)                          # the reward for a wrong and for a right name
STEPS = 3


def rollout(P, images, labels, rng, greedy=False):
    """Episodes of look, look, name under policy ``P``, one per image.

    Returns ``obs`` (3, n, 201), ``probs`` (3, n, 14: zero for the actions not allowed at
    that step), ``act`` (3, n), ``mask`` (3, n, 4: uncovered before each step), ``hidden``
    (3, n, 64) and ``reward`` (n,): :data:`POINTS`, paid after the name."""
    n = len(images)
    mask = np.zeros((n, LOOKS))
    out = dict(obs=[], probs=[], act=[], mask=[], hidden=[])
    for t in range(STEPS):
        obs = observe(images, mask, t / STEPS)
        h, z = forward(P, obs)
        legal = slice(0, LOOKS) if t < 2 else slice(LOOKS, ACTIONS)
        p = np.zeros((n, ACTIONS))
        p[:, legal] = softmax(z[:, legal])
        if greedy:
            a = p.argmax(1)
        else:
            a = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
        for k, v in zip(("obs", "probs", "act", "mask", "hidden"), (obs, p, a, mask.copy(), h)):
            out[k].append(v)
        if t < 2:
            mask[np.arange(n), a] = 1
    out = {k: np.array(v) for k, v in out.items()}
    out["reward"] = np.where(out["act"][2] - LOOKS == labels, POINTS[1], POINTS[0])
    return out


def values(C, ep):
    """The critic's expected reward at each step of episodes ``ep``: (3, n)."""
    return np.array([forward(C, o)[1][:, 0] for o in ep["obs"]])


def weights(kind, ep, v):
    """What each sampled action's push is multiplied by, per estimator: (3, n).

    ``return``: the reward that followed (REINFORCE). ``baseline``: that minus the critic's
    expectation at the step. ``critic``: the one-step advantage, the critic's expectation
    after the step (the reward itself after the last) minus its expectation before."""
    g = np.broadcast_to(ep["reward"], v.shape)
    if kind == "return":
        return g
    if kind == "baseline":
        return g - v
    return np.concatenate([v[1:], ep["reward"][None]]) - v


def policy_gradient(P, ep, w):
    """The gradient estimate of -(mean reward) from episodes ``ep`` with weights ``w``."""
    n = ep["act"].shape[1]
    obs, h = ep["obs"].reshape(-1, IN), ep["hidden"].reshape(-1, HIDDEN)
    hot = np.eye(ACTIONS)[ep["act"].reshape(-1)]
    dout = -(w.reshape(-1, 1) * (hot - ep["probs"].reshape(-1, ACTIONS))) / n
    return backward(P, obs, h, dout)


def critic_gradient(C, ep, target):
    n = ep["act"].shape[1]
    obs = ep["obs"].reshape(-1, IN)
    h, v = forward(C, obs)
    return backward(C, obs, h, (v - target.reshape(-1, 1)) / n)


def accuracy(P, xt, yt):
    """Share of held-out digits named correctly when the likeliest action is always taken."""
    ep = rollout(P, xt, yt, None, greedy=True)
    return float(np.mean(ep["reward"] == POINTS[1]))


ESTIMATORS = ("return", "baseline", "critic")
PG = dict(updates=3000, batch=16, lr=0.002, critic_lr=0.005, every=50, probe_every=250,
          probe_batches=200, truth=8192)
KEPT = (0, 300, 3000)                        # updates at which the first seed's networks are kept


def gradient_noise(P, C, x, y, rng, batches, batch, truth):
    """How far a batch's gradient estimate lies from the true gradient, per estimator.

    The true gradient of policy ``P`` is taken from ``truth`` fresh episodes with the
    baseline estimator (unbiased whatever the critic is). Then ``batches`` batches of
    ``batch`` episodes are sampled, and all three estimators are computed on the *same*
    episodes. Returns the mean squared distance to the true gradient (3,), the squared
    length of the true gradient, and the squared distance of each estimator's mean over
    all the batches from it (3,): what is left when the noise is averaged away."""
    ids = rng.integers(len(x), size=truth)
    ep = rollout(P, x[ids], y[ids], rng)
    true = flat(policy_gradient(P, ep, weights("baseline", ep, values(C, ep))))
    dist, mean = np.zeros(3), np.zeros((3, len(true)))
    for _ in range(batches):
        ids = rng.integers(len(x), size=batch)
        ep = rollout(P, x[ids], y[ids], rng)
        v = values(C, ep)
        for k, kind in enumerate(ESTIMATORS):
            g = flat(policy_gradient(P, ep, weights(kind, ep, v)))
            dist[k] += np.sum((g - true) ** 2) / batches
            mean[k] += g / batches
    return dist, float(np.sum(true ** 2)), np.sum((mean - true) ** 2, 1)


def pg_run(kind, seed, probe=False, keep=False, **s):
    """One training run of look, look, name with estimator ``kind`` (:func:`weights`).

    The policy starts from the same weights for every ``kind`` of a seed. A critic is
    trained in every run (toward the reward that followed, or for ``critic`` toward its own
    next expectation) but only the last two kinds use it."""
    s = dict(PG, **s)
    x, y, xt, yt, _ = digits()
    init = np.random.default_rng(seed)
    P, C = net(init, ACTIONS), net(init, 1)
    rng = np.random.default_rng([seed, ESTIMATORS.index(kind) + 1])
    popt, copt = Adam(P, s["lr"]), Adam(C, s["critic_lr"])
    out = dict(accuracy=[], reward=[], noise=[], signal=[], bias=[])
    for step in range(s["updates"] + 1):
        if step % s["every"] == 0:
            out["accuracy"].append(accuracy(P, xt, yt))
        if probe and step and step % s["probe_every"] == 0:
            noise, signal, bias = gradient_noise(P, C, x, y, np.random.default_rng([seed, step]),
                                                 s["probe_batches"], s["batch"], s["truth"])
            out["noise"].append(noise)
            out["signal"].append(signal)
            out["bias"].append(bias)
        if keep and step in KEPT:
            for name, params in (("P", P), ("C", C)):
                out.update({"%s%d_%s" % (name, step, k): v.copy() for k, v in params.items()})
        if step == s["updates"]:
            break
        ids = rng.integers(len(x), size=s["batch"])
        ep = rollout(P, x[ids], y[ids], rng)
        v = values(C, ep)
        w = weights(kind, ep, v)
        target = (np.concatenate([v[1:], ep["reward"][None]]) if kind == "critic"
                  else np.broadcast_to(ep["reward"], v.shape))
        popt.step(P, policy_gradient(P, ep, w))
        copt.step(C, critic_gradient(C, ep, target))
        out["reward"].append(float(ep["reward"].mean()))
    return {k: np.array(v) for k, v in out.items()}


def pg_runs():
    """Every look, look, name run, cached: per estimator (return, baseline, critic) and
    seed, held-out ``accuracy`` (3, 5, 61) and training ``reward`` (3, 5, 3000); the
    gradient-noise probes made on the policies of the ``baseline`` runs (``noise_b``,
    ``bias_b`` (5, 12, 3), ``signal_b`` (5, 12)) and of the ``critic`` runs (``noise_c``
    ...); and the first seed's networks at the updates in :data:`KEPT`
    (``<kind>_P<update>_W`` ...)."""
    def build():
        out = {}
        acc, reward = [], []
        for kind in ESTIMATORS:
            runs = [pg_run(kind, seed, probe=kind != "return", keep=seed == 0)
                    for seed in range(SEEDS)]
            acc.append([r["accuracy"] for r in runs])
            reward.append([r["reward"] for r in runs])
            if kind != "return":
                for k in ("noise", "signal", "bias"):
                    out["%s_%s" % (k, kind[0])] = np.array([r[k] for r in runs])
            out.update({"%s_%s" % (kind, k): v for k, v in runs[0].items() if k[0] in "PC"})
        out.update(accuracy=np.array(acc), reward=np.array(reward))
        return out

    return cached("pg", build, seeds=SEEDS, **PG)


def kept(d, kind, update):
    """The first seed's policy and critic of estimator ``kind`` at ``update``."""
    return tuple({k: d["%s_%s%d_%s" % (kind, n, update, k)] for k in "WbUc"} for n in "PC")


def _band(runs):
    """Mean, lowest and highest over seeds (axis 0)."""
    runs = np.asarray(runs, float)
    return np.array([runs.mean(0), runs.min(0), runs.max(0)])


def _first(curve, level, x):
    """The first ``x`` at which ``curve`` reaches ``level`` (the last ``x`` if never)."""
    hit = np.flatnonzero(np.asarray(curve) >= level)
    return float(x[hit[0]]) if len(hit) else float(x[-1])


def _pg_meta(d, kinds):
    final = d["accuracy"][:, :, -1]
    s = PG
    return dict(
        seeds=SEEDS, data=digits()[4], task="glimpse game: look, look, name (see module)",
        reward="1 point for any answer, 2 for the right one, paid after the name",
        model="tanh MLP 201-64-14 policy; tanh MLP 201-64-1 critic", optimiser="Adam",
        settings=s, estimators=list(kinds),
        final_accuracy={k: final[ESTIMATORS.index(k)].tolist() for k in kinds},
        evidence_kind="actual NumPy updates on collected data",
        limitations=["One small task and network; five seeds per arm."])


def policy_gradient_study():
    d = pg_runs()
    x = np.arange(d["accuracy"].shape[2]) * PG["every"]
    probes = np.arange(1, d["noise_b"].shape[1] + 1) * PG["probe_every"]
    acc = d["accuracy"][:2]
    noise = d["noise_b"].mean(0)[:, :2].T                      # (2, probes)
    names = ["no baseline", "learned baseline"]
    meta = _pg_meta(d, ESTIMATORS[:2])
    meta.update(
        noise_ratio_per_seed=(d["noise_b"][:, :, 0] / d["noise_b"][:, :, 1]).mean(1).tolist(),
        updates_to_half=[[_first(run, 0.5, x) for run in arm] for arm in acc],
        designs_tried=["this one (fixed before any comparison was run)"])
    return Study(
        "A baseline steadies the policy gradient",
        "REINFORCE on look, look, name over real MNIST digits; 5 seeds; held-out accuracy",
        (
            _curve("Held-out accuracy, mean of 5 seeds", "Greedy play on 1,000 test digits.",
                   acc.mean(1), names, x),
            _curve("Held-out accuracy, every seed", "Rows: 5 seeds without, then 5 with.",
                   acc.reshape(-1, acc.shape[2]), ["%s, seed %d" % (n, k) for n in names
                                                    for k in range(SEEDS)], x),
            _curve("Distance of a batch's gradient from the true gradient",
                   "Mean squared distance over 200 batches of 16 episodes, on the same policy "
                   "and the same episodes; true gradient from 8,192 episodes.", noise, names,
                   probes),
            _bars("Final held-out accuracy per seed", "No baseline, then learned baseline.",
                  acc[:, :, -1].reshape(-1), ["%s %d" % (n, k) for n in ("none", "baseline")
                                              for k in range(SEEDS)]),
        ), dict(seed=17, **meta))


def actor_critic_study():
    d = pg_runs()
    x = np.arange(d["accuracy"].shape[2]) * PG["every"]
    probes = np.arange(1, d["noise_c"].shape[1] + 1) * PG["probe_every"]
    names = ["raw return (REINFORCE)", "return - baseline", "actor-critic"]
    meta = _pg_meta(d, ESTIMATORS)
    meta.update(
        noise_per_seed=d["noise_c"].mean(1).tolist(), bias_per_seed=d["bias_c"].mean(1).tolist(),
        signal_per_seed=d["signal_c"].mean(1).tolist(),
        updates_to_half=[[_first(run, 0.5, x) for run in arm] for arm in d["accuracy"]],
        designs_tried=["this one (fixed before any comparison was run)"])
    return Study(
        "The critic's estimate replaces the raw return",
        "Actor-critic against REINFORCE on look, look, name over real MNIST digits; 5 seeds",
        (
            _curve("Held-out accuracy, mean of 5 seeds", "Greedy play on 1,000 test digits.",
                   d["accuracy"].mean(1), names, x),
            _curve("Held-out accuracy, lowest and highest seed",
                   "Rows: lowest then highest of 5 seeds, per estimator.",
                   np.concatenate([_band(a)[1:] for a in d["accuracy"]]),
                   ["%s, %s" % (n, w) for n in names for w in ("lowest", "highest")], x),
            _curve("Distance of a batch's gradient from the true gradient",
                   "On the actor-critic runs' policies and critics: the three estimators on "
                   "the same 200 batches of 16 episodes.", d["noise_c"].mean(0).T, names, probes),
            _bars("Final held-out accuracy per seed", "Three estimators, five seeds each.",
                  d["accuracy"][:, :, -1].reshape(-1),
                  ["%s %d" % (n, k) for n in ("return", "baseline", "critic")
                   for k in range(SEEDS)]),
        ), dict(seed=17, **meta))


# ---------------------------------------------------------------------- the free game
LIMIT, GAMMA = 6, 0.9
Q = dict(iterations=7500, envs=8, batch=32, lr=0.001, buffer=20000, block=250, sync=200,
         every=100, explore=0.1, explore_from=1000)
SHAPE = dict(potential=0.25, bonus=0.25)     # per uncovered quadrant; per look


def play(P, images, labels):
    """Greedy play of the free game by value network ``P``, one episode per image.

    Returns ``act`` (6, n: -1 after the episode ended), ``q`` (6, n, 14), ``mask``
    (6, n, 4: uncovered before each step), ``steps`` (n,: actions taken), ``right`` (n,:
    named correctly within the limit), ``named`` (n,), ``repeats`` (n,: looks at a quadrant
    already uncovered) and ``score`` (n,: 0.9 ** (steps before the name) when right)."""
    n = len(images)
    mask, alive = np.zeros((n, LOOKS)), np.ones(n, bool)
    out = dict(act=[], q=[], mask=[])
    right, named = np.zeros(n, bool), np.zeros(n, bool)
    steps, repeats, score = np.zeros(n, int), np.zeros(n, int), np.zeros(n)
    for t in range(LIMIT):
        q = forward(P, observe(images, mask, t / LIMIT))[1]
        a = np.where(alive, q.argmax(1), -1)
        out["act"].append(a)
        out["q"].append(q)
        out["mask"].append(mask.copy())
        look = alive & (a < LOOKS)
        name = alive & (a >= LOOKS)
        rows = np.flatnonzero(look)
        repeats[rows] += mask[rows, a[rows]] == 1
        mask[rows, a[rows]] = 1
        steps += alive
        hit = name & (a - LOOKS == labels)
        right |= hit
        named |= name
        score[hit] = GAMMA ** t
        alive = alive & ~name
    out = {k: np.array(v) for k, v in out.items()}
    out.update(steps=steps, right=right, named=named, repeats=repeats, score=score)
    return out


def q_run(seed, replay=True, target=True, sorted_stream=False, shaping="none", keep=False,
          **s):
    """Q-learning on the free game with a tanh network, one update per step of 8 games.

    ``sorted_stream``: the training digits arrive one class at a time (250 steps of 0s,
    then 1s ...), otherwise shuffled. ``replay``: each update trains on 32 transitions
    drawn from the last 20,000, otherwise on the newest 32. ``target``: the targets use a
    copy of the network refreshed every 200 updates, otherwise the network itself.
    ``shaping``: ``potential`` adds 0.9 x potential(next) - potential(now) to every reward
    (potential: 0.25 per uncovered quadrant, 0 once the episode has ended); ``bonus`` adds
    0.25 to every look.

    Every 100 steps, on the 1,000 held-out digits with greedy play: ``right`` (share named
    correctly in time), ``score`` (the game's own return: 0.9 ** steps when right),
    ``steps``, ``repeats`` (looks at an uncovered quadrant per episode), ``named`` and
    ``by_class`` (10,); ``value``: the network's own mean estimate of its best action at
    the covered start, with shaping potential 0 there so comparable with ``score``;
    ``batch_classes`` (10,): the digit classes in that step's training batch; ``loss``."""
    s = dict(Q, **s)
    x, y, xt, yt, _ = digits()
    P = net(np.random.default_rng(seed), ACTIONS)
    T = {k: v.copy() for k, v in P.items()}
    code = 1 + replay + 2 * target + 4 * sorted_stream + 8 * ("none", "potential",
                                                               "bonus").index(shaping)
    rng = np.random.default_rng([seed, code])
    opt = Adam(P, s["lr"])
    E, B, size = s["envs"], s["batch"], s["buffer"]
    by_class = [np.flatnonzero(y == c) for c in range(10)]
    buf = dict(img=np.zeros(size, int), mask=np.zeros((size, LOOKS)), t=np.zeros(size),
               act=np.zeros(size, int), reward=np.zeros(size), done=np.zeros(size, bool),
               next=np.zeros((size, LOOKS)))
    held = 0

    def deal(n, it):
        if sorted_stream:
            return rng.choice(by_class[(it // s["block"]) % 10], n)
        return rng.integers(len(x), size=n)

    img, mask, t = deal(E, 0), np.zeros((E, LOOKS)), np.zeros(E)
    out = {k: [] for k in ("right", "score", "steps", "repeats", "named", "by_class", "value",
                           "batch_classes", "loss")}
    for it in range(s["iterations"]):
        eps = max(s["explore"], 1 - (1 - s["explore"]) * it / s["explore_from"])
        q = forward(P, observe(x[img], mask, t / LIMIT))[1]
        a = np.where(rng.random(E) < eps, rng.integers(ACTIONS, size=E), q.argmax(1))
        look = a < LOOKS
        after = mask.copy()
        after[np.flatnonzero(look), a[look]] = 1
        done = ~look | (t + 1 >= LIMIT)
        reward = (~look & (a - LOOKS == y[img])).astype(float)
        if shaping == "potential":
            reward += s.get("potential", SHAPE["potential"]) * (
                GAMMA * after.sum(1) * ~done - mask.sum(1))
        elif shaping == "bonus":
            reward += s.get("bonus", SHAPE["bonus"]) * look
        at = (held + np.arange(E)) % size
        for k, v in zip(("img", "mask", "t", "act", "reward", "done", "next"),
                        (img, mask, t, a, reward, done, after)):
            buf[k][at] = v
        held += E
        mask, t = after, t + 1
        over = np.flatnonzero(done)
        img[over], mask[over], t[over] = deal(len(over), it), 0, 0
        # one update
        if replay:
            ids = rng.integers(min(held, size), size=B)
        else:
            ids = (held - 1 - np.arange(B)) % size
            ids = ids[ids < held]
        obs = observe(x[buf["img"][ids]], buf["mask"][ids], buf["t"][ids] / LIMIT)
        nxt = observe(x[buf["img"][ids]], buf["next"][ids], (buf["t"][ids] + 1) / LIMIT)
        boot = forward(T if target else P, nxt)[1].max(1)
        goal = buf["reward"][ids] + GAMMA * ~buf["done"][ids] * boot
        h, qs = forward(P, obs)
        rows = np.arange(len(ids))
        err = qs[rows, buf["act"][ids]] - goal
        dq = np.zeros_like(qs)
        dq[rows, buf["act"][ids]] = err / len(ids)
        opt.step(P, backward(P, obs, h, dq))
        if target and (it + 1) % s["sync"] == 0:
            T = {k: v.copy() for k, v in P.items()}
        if it % s["every"] == 0 or it == s["iterations"] - 1:
            ep = play(P, xt, yt)
            for k in ("right", "score", "steps", "repeats", "named"):
                out[k].append(float(ep[k].mean()))
            out["by_class"].append([float(ep["right"][yt == c].mean()) for c in range(10)])
            out["value"].append(float(ep["q"][0].max(1).mean()))
            out["batch_classes"].append(np.bincount(y[buf["img"][ids]], minlength=10))
            out["loss"].append(float(np.mean(err ** 2) / 2))
    out = {k: np.array(v) for k, v in out.items()}
    if keep:
        out.update({"P_" + k: v for k, v in P.items()})
    return out


def _stack(runs, keys):
    return {k: np.array([[r[k] for r in arm] for arm in runs]) for k in keys}


EVAL_KEYS = ("right", "score", "steps", "repeats", "named", "by_class", "value",
             "batch_classes", "loss")
DQN_ARMS = (                                 # name, replay, target network, sorted stream
    ("neither", False, False, True),
    ("replay", True, False, True),
    ("target network", False, True, True),
    ("both", True, True, True),
    ("neither, shuffled digits", False, False, False),
)


def dqn_runs():
    """The five arms of :data:`DQN_ARMS`, five seeds each: every array of :func:`q_run`
    stacked (arm, seed, evaluation, ...), and ``P_*``: the first seed's final network of
    the arm with both."""
    def build():
        runs = [[q_run(seed, replay, target, stream, keep=name == "both" and seed == 0)
                 for seed in range(SEEDS)] for name, replay, target, stream in DQN_ARMS]
        out = _stack(runs, EVAL_KEYS)
        out.update({k: v for k, v in runs[3][0].items() if k.startswith("P_")})
        return out

    return cached("dqn", build, seeds=SEEDS, arms=[a[0] for a in DQN_ARMS], **Q)


def last_pass(curve):
    """The evaluations of the last pass through the ten classes (the last 2,500 steps):
    which class a run saw last decides how it looks at the very end, so runs are compared
    over a whole pass."""
    n = 10 * Q["block"] // Q["every"]
    return np.asarray(curve)[..., -n:]


def dqn_study():
    d = dqn_runs()
    x = np.minimum(np.arange(d["right"].shape[2]) * Q["every"], Q["iterations"] - 1)
    names = [a[0] for a in DQN_ARMS]
    late = last_pass(d["right"])                               # (arm, seed, 25)
    mean, worst = late.mean(2), late.min(2)
    excess = last_pass(d["value"] - d["score"]).mean(2)
    meta = dict(
        seeds=SEEDS, data=digits()[4], task="glimpse game: free version (see module)",
        model="tanh MLP 201-64-14 action values", optimiser="Adam", settings=Q, discount=GAMMA,
        arms=names, last_pass_right_per_seed=mean.tolist(), last_pass_worst_per_seed=worst.tolist(),
        value_minus_score_per_seed=excess.tolist(),
        evidence_kind="actual NumPy updates on collected data",
        designs_tried=["this one (fixed before any comparison was run)"],
        limitations=["One small task and network; five seeds per arm. The digit classes "
                     "arriving in runs is a constructed source of correlated experience."])
    return Study(
        "What replay and a target network do for Q-learning",
        "Q-learning with a tanh network on the glimpse game; digits arrive one class at a time",
        (
            _curve("Held-out digits named correctly, mean of 5 seeds",
                   "Greedy play on 1,000 test digits every 100 steps.", d["right"].mean(1),
                   names, x),
            _bars("Last pass through the classes: mean share right, per seed",
                  "Five arms, five seeds each.", mean.reshape(-1),
                  ["%s %d" % (n, k) for n in names for k in range(SEEDS)]),
            _matrix("Share right by digit class over training, first seed",
                    "Rows: class 0-9 without replay or target network, then with both.",
                    np.concatenate([d["by_class"][0, 0].T, d["by_class"][3, 0].T])),
            _curve("The value the network expects minus the score it gets",
                   "At the covered start, mean of 5 seeds.",
                   (d["value"] - d["score"]).mean(1), names, x),
        ), dict(seed=17, **meta))


SHAPE_ARMS = (("sparse reward", "none"), ("potential-based shaping", "potential"),
              ("bonus per look", "bonus"))


def shaping_runs():
    """Three rewards (:data:`SHAPE_ARMS`), five seeds each, on shuffled digits with replay
    and a target network: :func:`q_run`'s arrays stacked (arm, seed, evaluation, ...) and
    the first seed's final network per arm (``P<arm>_*``)."""
    def build():
        runs = [[q_run(seed, shaping=kind, keep=seed == 0) for seed in range(SEEDS)]
                for _name, kind in SHAPE_ARMS]
        out = _stack(runs, EVAL_KEYS)
        for j, arm in enumerate(runs):
            out.update({"P%d_%s" % (j, k[2:]): v for k, v in arm[0].items()
                        if k.startswith("P_")})
        return out

    return cached("shaping", build, seeds=SEEDS, arms=[a[1] for a in SHAPE_ARMS],
                  **dict(Q, **SHAPE))


def shaping_study():
    d = shaping_runs()
    x = np.minimum(np.arange(d["score"].shape[2]) * Q["every"], Q["iterations"] - 1)
    names = [a[0] for a in SHAPE_ARMS]
    late = {k: last_pass(d[k]).mean(2) for k in ("score", "steps", "repeats", "right")}
    level = 0.9 * late["score"][0].mean()
    reach = [[_first(run, level, x) for run in arm] for arm in d["score"]]
    meta = dict(
        seeds=SEEDS, data=digits()[4], task="glimpse game: free version (see module)",
        model="tanh MLP 201-64-14 action values; replay and target network", optimiser="Adam",
        settings=dict(Q, **SHAPE), discount=GAMMA, arms=names,
        late_score_per_seed=late["score"].tolist(), late_steps_per_seed=late["steps"].tolist(),
        late_repeats_per_seed=late["repeats"].tolist(), late_right_per_seed=late["right"].tolist(),
        steps_to_reach=reach, reach_level=float(level),
        evidence_kind="actual NumPy updates on collected data",
        designs_tried=["this one (fixed before any comparison was run)"],
        limitations=["One small task and network; five seeds per arm. The potential and the "
                     "bonus (0.25) are chosen numbers."])
    return Study(
        "Shaping that keeps the task, and shaping that replaces it",
        "Q-learning on the glimpse game paid three ways; scored on the game's own reward",
        (
            _curve("The game's own score on held-out digits, mean of 5 seeds",
                   "0.9 ** steps for a right name; greedy play on 1,000 test digits.",
                   d["score"].mean(1), names, x),
            _curve("Steps per episode, mean of 5 seeds", "The limit is 6.",
                   d["steps"].mean(1), names, x),
            _bars("Late score per seed", "Mean over the last 2,500 steps; 3 rewards x 5 seeds.",
                  late["score"].reshape(-1), ["%s %d" % (n, k) for n in names
                                              for k in range(SEEDS)]),
            _curve("Looks at an already uncovered quadrant, per episode", "Mean of 5 seeds.",
                   d["repeats"].mean(1), names, x),
        ), dict(seed=17, **meta))


STUDIES = {
    "policy-gradient": policy_gradient_study,
    "actor-critic": actor_critic_study,
    "dqn-replay-buffer": dqn_study,
    "reward-shaping": shaping_study,
}
