"""The experiment behind animation 181, ``labels-against-rewards``: the bridge from
training classifiers on labels (Part 5) to learning from reward (Part 6).

One small classifier of MNIST digits (784-64-10, ReLU) is trained two ways, from the same
random start, on the same batches of training digits, for the same number of Adam
updates:

* **labels**: cross-entropy on the label. The gradient on the ten logits of a digit is
  ``p - onehot(y)``.
* **rewards** (REINFORCE as a contextual bandit): the classifier's softmax is a policy. A
  digit ``a`` is drawn from it, the reward is 1 when ``a`` is the label and 0 otherwise,
  and the label itself is never shown. The baseline ``b`` is a moving average of the
  rewards of the previous updates (``b <- 0.95 b + 0.05 * mean reward``, starting at the
  chance rate 0.1). The loss ``-(r - b) log p(a)`` has the gradient
  ``(r - b)(p - onehot(a))`` on the logits.

Averaged over the draw of ``a`` that gradient is exactly ``p(y) (p - onehot(y))``: the
label's gradient, scaled by the chance of being right (:func:`expected_reward_gradient`).
It points the same way and is weakest exactly where the classifier is wrong.

What is measured: the two gradients of one real digit, drawn many times; the right/wrong
reward's entropy ``H(reward)`` through training (at most 1 bit, against the ``log2 10``
bits a label of ten digits can carry); held-out accuracy of both, three seeds, with the
updates each needs to reach an accuracy; and each digit's held-out accuracy, which shows
digits the rewarded classifier stopped naming. Plain numpy, seeded; every accuracy is on
the 10,000 test digits, which are never trained on.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
CLASSES = 10
LOG2_10 = float(np.log2(CLASSES))
HIDDEN = 64
SEEDS = (0, 1, 2)
SEED = 181                         # every generator is seeded from [SEED, seed, stream]
UPDATES = 10000
BATCH = 64
RATE = 1e-3
BETA = 0.95                        # the baseline's moving average
B0 = 0.1                           # the baseline before any reward: the chance rate
PROBES = np.unique(np.concatenate([np.arange(0, 200, 10), np.arange(200, UPDATES + 1, 100)]))
WINDOW = 100                       # updates per point of the reward's entropy
LEVELS = (0.80, 0.90, 0.95)        # held-out accuracies whose first arrival is compared
SHOWN_UPDATE = 100                 # the seed-0 rewarded network the gradients are drawn on
DRAWS = 10                         # sampled actions shown for that digit
MANY = 10000                       # sampled actions averaged for that digit
DEAD = 0.05                        # a digit "not named": held-out accuracy below this
EVIDENCE = dict(
    exact="exact arithmetic: one digit's measured probabilities, the two gradients and "
          "their expectation",
    measured="measured on MNIST: a 784-64-10 classifier trained here in numpy, three "
             "seeds, scored on the 10,000 held-out test digits",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-reward-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== the classifier
def init(seed):
    """He-normal first layer, a 1/sqrt(64) output layer, zero biases, float32."""
    rng = np.random.default_rng([SEED, int(seed), 0])
    return dict(W0=rng.normal(0, np.sqrt(2 / 784), (784, HIDDEN)).astype(np.float32),
                b0=np.zeros(HIDDEN, np.float32),
                W1=rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, CLASSES)).astype(np.float32),
                b1=np.zeros(CLASSES, np.float32))


def forward(P, x):
    """The hidden layer and the ten logits of each image."""
    h = np.maximum(x @ P["W0"] + P["b0"], 0)
    return h, h @ P["W1"] + P["b1"]


def softmax(z):
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def backward(P, x, h, d):
    """Every parameter's gradient of a loss whose gradient on the logits is ``d``."""
    dh = (d @ P["W1"].T) * (h > 0)
    return dict(W0=x.T @ dh, b0=dh.sum(0), W1=h.T @ d, b1=d.sum(0))


def label_gradient(p, y):
    """``p - onehot(y)``: the gradient of ``-ln p(y)`` on the logits, one row a digit."""
    g = np.array(p, np.float64)
    g[np.arange(len(g)), y] -= 1.0
    return g


def reward_gradient(p, a, r, b):
    """``(r - b)(p - onehot(a))``: the gradient of ``-(r - b) ln p(a)`` on the logits."""
    return (np.asarray(r, np.float64) - b)[:, None] * label_gradient(p, a)


def expected_reward_gradient(p, y, b):
    """The reward gradient averaged over the draw ``a ~ p`` exactly (a sum over the ten
    digits): ``p(y) (p - onehot(y))`` whatever the baseline ``b``."""
    p = np.asarray(p, np.float64)
    out = np.zeros_like(p)
    for a in range(p.shape[1]):
        act = np.full(len(p), a)
        out += p[:, a:a + 1] * reward_gradient(p, act, (act == y).astype(float), b)
    return out


def sample(p, u):
    """The digit each row of ``p`` names for uniform draws ``u`` (one per row)."""
    return (np.cumsum(p, 1) > np.asarray(u)[:, None]).argmax(1)


def binary_entropy(q):
    """``H(reward)`` in bits of a reward that is 1 with probability ``q``."""
    q = np.clip(np.asarray(q, np.float64), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.where((q > 0) & (q < 1), h, 0.0)


def per_class(P, xt, yt):
    """Held-out accuracy of each digit: the share of its test images named correctly."""
    named = forward(P, xt)[1].argmax(1)
    return np.array([float((named[yt == k] == k).mean()) for k in range(CLASSES)])


def flat(G):
    return np.concatenate([np.asarray(G[k], np.float64).ravel() for k in sorted(G)])


def train(seed, kind, keep_at=None):
    """One run of ``kind`` ("label" or "reward") from :func:`init` (``seed``), on the batch
    order drawn from ``[SEED, seed, 1]`` (the same for both kinds). Returns per probe the
    held-out accuracy and each digit's; per update the batch's mean reward (rewards: the
    sampled actions'; labels: what greedy naming would have earned), the baseline used,
    and per probe the cosine between this batch's two parameter gradients (rewarded runs);
    and, at update ``keep_at``, the parameters and the next batch's indices."""
    x, y, xt, yt = mnist.load()
    x, xt = x.astype(np.float32), xt.astype(np.float32)
    P = init(seed)
    opt = Adam(P, np.float32(RATE))
    order_rng = np.random.default_rng([SEED, int(seed), 1])
    act_rng = np.random.default_rng([SEED, int(seed), 2])
    order, pos, b = order_rng.permutation(len(x)), 0, B0
    acc, by_class, reward, base, cosine, kept = [], [], [], [], [], {}
    for u in range(UPDATES + 1):
        if u in PROBES:
            named = forward(P, xt)[1].argmax(1)
            by_class.append([float((named[yt == k] == k).mean()) for k in range(CLASSES)])
            acc.append(float((named == yt).mean()))
        if u == UPDATES:
            break
        if pos + BATCH > len(x):
            order, pos = order_rng.permutation(len(x)), 0
        idx = order[pos:pos + BATCH]
        pos += BATCH
        if u == keep_at:
            kept = dict(params={k: v.copy() for k, v in P.items()}, batch=idx.copy(), b=b)
        xb, yb = x[idx], y[idx]
        h, z = forward(P, xb)
        p = softmax(z)
        if kind == "label":
            d = label_gradient(p, yb)
            reward.append(float((p.argmax(1) == yb).mean()))
            base.append(np.nan)
        else:
            a = sample(p, act_rng.random(BATCH))
            r = (a == yb).astype(np.float64)
            d = reward_gradient(p, a, r, b)
            reward.append(float(r.mean()))
            base.append(b)
            if u in PROBES:
                g_l = flat(backward(P, xb, h, (label_gradient(p, yb) / BATCH).astype(
                    np.float32)))
                g_r = flat(backward(P, xb, h, (d / BATCH).astype(np.float32)))
                cosine.append(float(g_l @ g_r / max(np.linalg.norm(g_l) * np.linalg.norm(g_r),
                                                    1e-30)))
            b = BETA * b + (1 - BETA) * float(r.mean())
        opt.step(P, backward(P, xb, h, (d / BATCH).astype(np.float32)))
    return dict(acc=np.array(acc), by_class=np.array(by_class), reward=np.array(reward),
                base=np.array(base), cosine=np.array(cosine), kept=kept)


def _runs_build():
    out = {}
    for seed in SEEDS:
        for kind in ("label", "reward"):
            keep = SHOWN_UPDATE if (kind == "reward" and seed == SEEDS[0]) else None
            r = train(seed, kind, keep)
            for k in ("acc", "by_class", "reward", "base", "cosine"):
                out["%s_%d_%s" % (kind, seed, k)] = r[k]
            if keep is not None:
                for k, v in r["kept"]["params"].items():
                    out["shown_" + k] = v
                out["shown_batch"] = r["kept"]["batch"]
                out["shown_b"] = np.float64(r["kept"]["b"])
    return out


@functools.lru_cache(maxsize=1)
def runs():
    """Both kinds of training, three seeds each (cached)."""
    return _cache("runs", _runs_build, (init, forward, softmax, backward, label_gradient,
                                        reward_gradient, sample, per_class, flat, train,
                                        _runs_build),
                  hidden=HIDDEN, seeds=SEEDS, seed=SEED, updates=UPDATES, batch=BATCH,
                  rate=RATE, beta=BETA, b0=B0, probes=[int(v) for v in PROBES],
                  shown=SHOWN_UPDATE)


def run(kind, seed, what):
    return runs()["%s_%d_%s" % (kind, seed, what)]


def shown_params():
    r = runs()
    return {k[len("shown_"):]: r[k] for k in ("shown_W0", "shown_b0", "shown_W1", "shown_b1")}


# ====================================================================== measures
def first_reach(acc, level, probes=PROBES):
    """The first probed update at which ``acc`` is at least ``level`` (None: never)."""
    hit = np.flatnonzero(np.asarray(acc) >= level)
    return int(probes[hit[0]]) if len(hit) else None


def windows(per_update, width=WINDOW):
    """Means of consecutive blocks of ``width`` updates, and the update each block ends at."""
    v = np.asarray(per_update, float)
    n = len(v) // width
    return v[:n * width].reshape(n, width).mean(1), (np.arange(n) + 1) * width


def stalls(by_class, probes=PROBES, after=100, dead=DEAD):
    """Digits a run stopped naming: each digit whose held-out accuracy is below ``dead``
    at some probe from update ``after`` on, with the first and last such probe."""
    out = []
    late = probes >= after
    for k in range(by_class.shape[1]):
        low = np.flatnonzero(late & (by_class[:, k] < dead))
        if len(low):
            out.append(dict(digit=k, since=int(probes[low[0]]), until=int(probes[low[-1]])))
    return out


def _shown_digit():
    """A training digit of the batch after update ``SHOWN_UPDATE`` (seed 0, rewarded run):
    the first whose probability of its own label is between 0.3 and 0.7, so that a draw is
    right some of the time and wrong some of the time."""
    x, y, _, _ = mnist.load()
    r = runs()
    idx = r["shown_batch"]
    p = softmax(forward(shown_params(), x[idx].astype(np.float32))[1])
    py = p[np.arange(len(idx)), y[idx]]
    j = int(np.flatnonzero((py > 0.3) & (py < 0.7))[0])
    return int(idx[j]), int(y[idx[j]]), p[j], x[idx[j]]


# ====================================================================== the study
CORE = (label_gradient, reward_gradient, binary_entropy)


def labels_against_rewards():
    r = runs()
    kinds = ("label", "reward")
    acc = {k: np.array([run(k, s, "acc") for s in SEEDS]) for k in kinds}
    probes = PROBES.astype(float)
    # --- one digit: its gradients, drawn and averaged
    index, y, p, image = _shown_digit()
    b = float(r["shown_b"])
    rng = np.random.default_rng([SEED, 0, 3])
    many = sample(np.repeat(p[None], MANY, 0), rng.random(MANY))
    draws = many[:DRAWS]
    g_label = label_gradient(p[None], [y])[0]
    g_draws = reward_gradient(np.repeat(p[None], DRAWS, 0), draws, draws == y, b)
    g_many = reward_gradient(np.repeat(p[None], MANY, 0), many, many == y, b)
    g_mean = g_many.mean(0)
    g_exp = expected_reward_gradient(p[None], [y], b)[0]

    def cos(a, c):
        return float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c)))
    draw_cos = [cos(g, g_label) for g in g_draws]
    many_cos = np.array([cos(g, g_label) for g in g_many])
    # --- information per example
    rate = {k: np.array([windows(run(k, s, "reward"))[0] for s in SEEDS]) for k in kinds}
    ends = windows(run("reward", SEEDS[0], "reward"))[1]
    H = binary_entropy(rate["reward"])
    y_train = mnist.load()[1]
    shares = np.bincount(y_train, minlength=CLASSES) / len(y_train)
    label_bits = float(-(shares * np.log2(shares)).sum())
    examples = UPDATES * BATCH
    reward_bits = H.mean(1) * examples            # bits of all the rewards of each run
    # --- held-out accuracy and the gap
    reach = {}
    for level in LEVELS:
        reach["%.2f" % level] = {k: [first_reach(a, level) for a in acc[k]] for k in kinds}
    gap = []
    for la, re in zip(reach["0.90"]["label"], reach["0.90"]["reward"]):
        gap.append(None if (la is None or re is None) else re / la)
    final = {k: _floats(acc[k][:, -1]) for k in kinds}
    # --- each digit's held-out accuracy, and the digits a rewarded run stopped naming
    stalled = {k: [stalls(run(k, s, "by_class")) for s in SEEDS] for k in kinds}
    lost = [max((st["until"] - st["since"] for st in ss), default=-1) for ss in
            stalled["reward"]]
    worst = int(np.argmax(lost))
    by_class = {k: run(k, SEEDS[worst], "by_class") for k in kinds}
    cosine = np.array([run("reward", s, "cosine") for s in SEEDS])
    reward_probes = PROBES[:-1]
    panels = (
        Panel("One digit, two gradients", "the ten probabilities of training digit %d (a %d) "
              "under the seed-0 rewarded network after %d updates; the label's gradient "
              "p - onehot(y); %d sampled rewards' gradients (r - b)(p - onehot(a)), their "
              "mean over %d draws and the exact expectation p(y)(p - onehot(y))"
              % (index, y, SHOWN_UPDATE, DRAWS, MANY), "bars",
              dict(values=p, labels=[str(k) for k in range(CLASSES)], label_gradient=g_label,
                   draws=draws.astype(float), draw_gradients=g_draws, mean_gradient=g_mean,
                   expected_gradient=g_exp, image=image.astype(np.float32))),
        Panel("Each digit's held-out accuracy", "the 10 digits (rows) through training "
              "(columns: probes) for seed %d: rewards above, labels below" % SEEDS[worst],
              "matrix", dict(values=np.concatenate([by_class["reward"].T,
                                                    by_class["label"].T]),
                             probes=probes)),
        Panel("Bits per example", "H(reward) in bits of the right/wrong reward, each point "
              "%d updates of %d digits, three seeds; a label can carry log2 10 = %.2f bits"
              % (WINDOW, BATCH, LOG2_10), "curve",
              dict(x=ends.astype(float), ys=H, rate=rate["reward"],
                   labels=["seed %d" % s for s in SEEDS])),
        Panel("Held-out accuracy", "the share of the 10,000 test digits named right, three "
              "seeds each: trained on labels and on rewards", "curve",
              dict(x=probes, ys=np.concatenate([acc["label"], acc["reward"]]),
                   labels=["labels, seed %d" % s for s in SEEDS] +
                          ["rewards, seed %d" % s for s in SEEDS])),
    )
    return Study(
        "Labels against rewards", "The same classifier trained from labels and from a "
        "right-or-wrong reward", panels,
        dict(source="a 784-64-10 classifier trained here in numpy on MNIST's 60,000 training "
                    "digits, from the same start and on the same batches, three seeds; "
                    "accuracy on the 10,000 test digits",
             dataset="MNIST", evidence=EVIDENCE, sizes=[784, HIDDEN, CLASSES],
             seeds=list(SEEDS), seed=SEED, updates=UPDATES, batch=BATCH, learning_rate=RATE,
             optimiser="Adam", examples=examples,
             baseline="a moving average of the previous updates' mean reward, "
                      "b <- %.2f b + %.2f mean(r), starting at %.1f" % (BETA, 1 - BETA, B0),
             beta=BETA, b0=B0, probes=_floats(PROBES),
             accuracy={k: [_floats(a) for a in acc[k]] for k in kinds},
             final_accuracy=final, reach=reach, levels=list(LEVELS), gap_90=gap,
             shown=dict(index=index, label=y, update=SHOWN_UPDATE, baseline=b,
                        probabilities=_floats(p), p_label=float(p[y]),
                        label_gradient=_floats(g_label), draws=[int(v) for v in draws],
                        draw_rewards=[int(v == y) for v in draws],
                        draw_gradients=[_floats(g) for g in g_draws], draw_cosines=draw_cos,
                        many=MANY, many_right=float((many == y).mean()),
                        mean_gradient=_floats(g_mean), expected_gradient=_floats(g_exp),
                        mean_error=float(np.abs(g_mean - g_exp).max()),
                        mean_cosine=cos(g_mean, g_label),
                        draw_cosine_mean=float(many_cos.mean()),
                        draw_cosine_sd=float(many_cos.std()),
                        right_cosine=float(many_cos[many == y].mean())
                        if (many == y).any() else None,
                        wrong_cosine=float(many_cos[many != y].mean())),
             information=dict(label_bits_max=LOG2_10, label_bits=label_bits,
                              reward_bits_max=1.0, window=WINDOW,
                              ends=_floats(ends),
                              entropy=[_floats(h) for h in H],
                              reward_rate=[_floats(v) for v in rate["reward"]],
                              peak=[float(v) for v in H.max(1)],
                              peak_at=[int(ends[i]) for i in H.argmax(1)],
                              final_entropy=[float(v) for v in H[:, -1]],
                              final_rate=[float(v) for v in rate["reward"][:, -1]],
                              mean_entropy=[float(v) for v in H.mean(1)],
                              reward_total_bits=[float(v) for v in reward_bits],
                              label_total_bits=float(label_bits * examples)),
             batch_cosine=dict(probes=_floats(reward_probes),
                               values=[_floats(c) for c in cosine],
                               mean=[float(c.mean()) for c in cosine]),
             stalls=stalled, stalled_seed=int(SEEDS[worst]),
             stalled_runs=int(sum(bool(ss) for ss in stalled["reward"])),
             limitations=["one small network, one learning rate and one baseline: the gap "
                          "depends on all three",
                          "REINFORCE here is a one-step bandit: each digit is one episode, "
                          "and the reward comes at once",
                          "accuracy is probed every 10 updates up to 200, then every 100: "
                          "an update count to reach a level is the first probe at it"]))


STUDIES = {"labels-against-rewards": labels_against_rewards}
