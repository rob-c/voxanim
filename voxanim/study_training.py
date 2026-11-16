"""Redesigned experiments for three of the training animations (24-26 of the "next 100").

The first versions (:mod:`voxanim.next_language`) could not show their concept:

* ``warmup-and-clipping`` trained three models on 192 excerpts, few enough to learn by heart,
  at a rate where nothing went wrong: the run that fitted least had the lowest held-out loss.
  Here a six-layer post-LN transformer trains on windows drawn at random from the whole
  training split, at a rate where a fixed-rate run never leaves the character-frequency
  plateau, in five seeds.
* ``grokking`` used a default initialisation, which generalises as it fits. Here the weights
  start three times larger (Liu et al., "Omnigrok", 2022), on 1,000 MNIST images with weight
  decay: test accuracy rises long after training accuracy saturates. The control is the same
  run from the default scale.
* ``scaling-laws`` gave every run the same 180 updates, so compute varied only with model
  size. Here four model sizes are each trained to a ladder of compute budgets (a small model
  reads many more characters than a large one for the same budget), on windows drawn at
  random from the whole training split.

Every number is a seeded numpy training run, cached by :func:`voxanim.cache.cached_npz`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import realdata
from .cache import cached_npz
from .optim import AdamW

SEED = 23


def _nl():
    from . import next_language

    return next_language


def _spread(each):
    """Mean, lowest and highest over the seed axis (the last but one)."""
    each = np.asarray(each, float)
    return each.mean(-2), each.min(-2), each.max(-2)


# ---------------------------------------------------------------------- 24: warm-up

WARMUP = dict(
    version=1,
    layers=6, width=32, heads=4, ff=128, placement="post",
    length=16, batch=24, updates=300,
    rate=0.006, safe_rate=0.003, warmup=80, clip=1.0,
    seeds=5, evaluate_every=20, held_out_windows=256,
)
# (name, peak rate key, warm-up, clipping)
WARMUP_ARMS = (
    ("fixed rate", "rate", False, False),
    ("warm-up", "rate", True, False),
    ("warm-up + clipping", "rate", True, True),
    ("fixed, half the rate", "safe_rate", False, False),
)
WARMUP_DESIGNS = [
    "first version (next_language._warmup): 4 layers, 192 memorisable excerpts, rate 0.02, one "
    "seed; nothing diverged and the fixed rate had the lowest held-out loss",
    "pilot, fixed rate only, 6 layers, random windows of the whole training split, rates 0.003 / 0.006 / 0.01 / 0.03, "  # noqa: E501
    "seeds 0-4: every run at 0.006 and above stayed on the character-frequency plateau; 0.003 "
    "trained",
    "pilot at rate 0.01, seeds 0-1: warm-up 80 trained one seed of two; clipping alone neither",
    "this design, fixed before its comparison was run: rate 0.006 (the lowest piloted rate at "
    "which the fixed-rate run failed in all five seeds), warm-up 80, clip 1.0, five seeds",
]


def _warmup_config():
    from . import tinylm as lm

    s = WARMUP
    return lm.Config(vocab=len(realdata.vocabulary()), width=s["width"], heads=s["heads"],
                     kv_heads=s["heads"], layers=s["layers"], ff=s["ff"],
                     placement=s["placement"], max_len=s["length"])


def unigram_loss(held):
    """Held-out loss of always predicting the training text's character frequencies."""
    counts = np.bincount(realdata.encode(realdata.text("train")),
                         minlength=len(realdata.vocabulary())) + 1.0
    return float(-np.log(counts / counts.sum())[held[:, 1:]].mean())


@lru_cache(maxsize=1)
def warmup_runs():
    """Every run of the warm-up experiment: for each arm and seed, the training loss and the
    gradient's length (before clipping) at every update, the factor clipping applied, and the
    held-out loss every 20 updates. Arms share a seed's initial weights and its batches."""
    from . import tinylm as lm

    s = WARMUP
    config = _warmup_config()

    def build():
        train = realdata.encode(realdata.text("train"))
        held = realdata.text_sequences(s["held_out_windows"], s["length"] + 1, "test", SEED + 1)
        n, every = s["updates"], s["evaluate_every"]
        shape = (len(WARMUP_ARMS), s["seeds"])
        loss, norm = np.zeros(shape + (n,)), np.zeros(shape + (n,))
        ratio, rate = np.ones(shape + (n,)), np.zeros((len(WARMUP_ARMS), n))
        held_loss = np.zeros(shape + (n // every + 1,))
        for a, (_name, key, warm, clip) in enumerate(WARMUP_ARMS):
            rate[a] = s[key] * (np.minimum(1.0, np.arange(1, n + 1) / s["warmup"]) if warm else 1.0)
            for seed in range(s["seeds"]):
                P = lm.init(config, SEED + seed)
                rng = np.random.default_rng(SEED + 100 + seed)
                opt = AdamW(P, s[key])
                held_loss[a, seed, 0] = lm.loss(P, held[:, :-1], held[:, 1:], config)
                for step in range(n):
                    starts = rng.integers(0, len(train) - s["length"] - 1, s["batch"])
                    seq = train[starts[:, None] + np.arange(s["length"] + 1)]
                    value, G = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], config)
                    length = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
                    r = min(1.0, s["clip"] / max(length, 1e-12)) if clip else 1.0
                    opt.step(P, {k: g * r for k, g in G.items()}, lr=rate[a, step])
                    loss[a, seed, step], norm[a, seed, step], ratio[a, seed, step] = value, length, r  # noqa: E501
                    if (step + 1) % every == 0:
                        held_loss[a, seed, (step + 1) // every] = lm.loss(
                            P, held[:, :-1], held[:, 1:], config)
        return dict(loss=loss, norm=norm, ratio=ratio, rate=rate, held=held_loss,
                    held_at=np.arange(n // every + 1) * every,
                    unigram=np.array(unigram_loss(held)),
                    weights=np.array(sum(v.size for v in lm.init(config, 0).values())))

    settings = dict(WARMUP, arms=[list(a) for a in WARMUP_ARMS], seed=SEED,
                    dataset=realdata.TEXT_SHA256,
                    sources=realdata.source_hash("tinylm.py", "optim.py"))
    return cached_npz("study-training-warmup", settings, build)


def warmup_and_clipping():
    nl = _nl()
    s, r = WARMUP, warmup_runs()
    names = [a[0] for a in WARMUP_ARMS]
    best = r["held"].min(-1)                              # each run at its best held-out point
    floor = float(r["unigram"])
    learned = (best < floor - 0.2).sum(1)
    steps = np.arange(1, s["updates"] + 1)
    return nl._study(
        "Warm-up and clipping at a rate that breaks a fixed-rate run",
        "TinyShakespeare · six-layer post-LN transformer · five seeds an arm",
        [
            nl._curve(
                "Learning-rate schedules",
                "The rate each update used: fixed at 0.006, the same reached over 80 updates, "
                "and fixed at half of it.",
                r["rate"][[0, 1, 3]], [names[0], names[1], names[3]], x=steps),
            nl._panel(
                "Held-out loss during training",
                "Mean of five seeds; every seed is kept. Batches are random windows of the "
                "training split; this loss is on the disjoint test split.",
                "curve", x=r["held_at"], ys=r["held"].mean(1), labels=names, each=r["held"],
                unigram=np.array(floor)),
            nl._panel(
                "Gradient length before clipping",
                "Seed 0 of each arm at every update; the largest length of every run is kept.",
                "curve", x=steps, ys=r["norm"][:, 0], labels=names, each=r["norm"],
                largest=r["norm"].max(-1), ratio=r["ratio"][2]),
            nl._panel(
                "Best held-out loss of each run",
                "Each run is scored at its own best checkpoint, so stability is not confused "
                "with overfitting. Bars are seed means.",
                "bars", values=best.mean(1), labels=names, each=best, learned=learned,
                unigram=np.array(floor)),
        ],
        source="Collected TinyShakespeare text",
        settings=dict(s, arms=names, weights=int(r["weights"])),
        learned_rule="a run learned if its best held-out loss is 0.2 nats under the "
                     "character-frequency loss",
        designs_tried=WARMUP_DESIGNS,
        **nl._text_meta(evaluation="256 windows of the test split, the same for every run"),
    )


# ---------------------------------------------------------------------- 25: grokking

GROKKING = dict(
    version=1,
    train_images=1000, test_images=2000, side=14,
    sizes=(196, 200, 200, 10), loss="mean squared error on one-hot targets",
    rate=0.003, weight_decay=0.03, batch=100, updates=12000,
    scales=(3.0, 1.0), seeds=3, checkpoints=48,
    fitted_at=0.99, generalised_at=0.85,
)
GROKKING_ARMS = ("weights start 3x larger", "default start")
GROKKING_DESIGNS = [
    "first version (next_language._grokking): 600 images at 7x7, 49-64-10 MLP, default "
    "initialisation, cross-entropy, 3,000 updates, one seed: memorised, no delay",
    "pilots, one seed each, of the Omnigrok recipe (1,000 images, MSE, AdamW, initial weights "
    "times 1/2/3/4/6/8 of He-normal): 28x28 and 14x14 inputs; widths 100 and 200; one and two "
    "hidden layers; rate 0.001/0.003/0.01; weight decay 0.01/0.03/0.1; batch 100/200. The delay "
    "appeared at every scale of 2 and above; a higher rate with less decay lengthens it",
    "this design, fixed before its three seeds were run: 14x14, 196-200-200-10, scale 3 against "
    "1, rate 0.003, weight decay 0.03, batch 100, 12,000 updates (the most that fits the time "
    "limit); thresholds 99% training and 85% test accuracy chosen beforehand",
]


def _mlp_forward(P, x, depth=3):
    """A stack of MLPs (one per leading index of every weight) on ``x``: every layer's input."""
    acts = [x]
    for k in range(depth):
        z = acts[-1] @ P["W%d" % k] + P["b%d" % k]
        acts.append(np.maximum(z, 0.0) if k < depth - 1 else z)
    return acts


def _mlp_grads(P, x, target, depth=3):
    """Gradients of each stacked MLP's mean squared error on its own batch."""
    acts = _mlp_forward(P, x, depth)
    d = 2.0 * (acts[-1] - target) / (target.shape[-2] * target.shape[-1])
    G = {}
    for k in range(depth - 1, -1, -1):
        G["W%d" % k] = np.ascontiguousarray(acts[k].transpose(0, 2, 1)) @ d
        G["b%d" % k] = d.sum(1, keepdims=True)
        if k:
            d = (d @ P["W%d" % k].transpose(0, 2, 1)) * (acts[k] > 0)
    return G


def grokking_checkpoints():
    s = GROKKING
    return np.unique(np.r_[0, np.round(np.logspace(0, np.log10(s["updates"]),
                                                   s["checkpoints"])).astype(int)])


def grokking_data():
    s = GROKKING
    x, y, train_meta = realdata.mnist_split(s["train_images"], "train", s["side"], SEED)
    xt, yt, test_meta = realdata.mnist_split(s["test_images"], "test", s["side"], SEED + 1)
    return x, y, xt, yt, train_meta, test_meta


@lru_cache(maxsize=1)
def grokking_runs():
    """Six MLPs trained together on 1,000 MNIST images: three seeds from weights three times
    the default scale, and the same three from the default scale (same batches). At each
    checkpoint: training and test accuracy and the weights' norm, indexed ``[arm, seed]``;
    and, for seed 0 of both arms, which training and test images were answered correctly."""
    from . import mnist

    s = GROKKING

    def build():
        x, y, xt, yt, _, _ = grokking_data()
        depth = len(s["sizes"]) - 1
        arms, seeds = len(s["scales"]), s["seeds"]
        layers = [mnist._init(np.random.default_rng(SEED + seed), s["sizes"])
                  for seed in range(seeds)]
        P = {}
        for k in range(depth):
            P["W%d" % k] = np.stack([layers[seed][k][0] * scale for scale in s["scales"]
                                     for seed in range(seeds)])
            P["b%d" % k] = np.zeros((arms * seeds, 1, s["sizes"][k + 1]))
        rngs = [np.random.default_rng(SEED + 100 + seed) for seed in range(seeds)]
        onehot = np.eye(10)[y]
        opt = AdamW(P, s["rate"], weight_decay=s["weight_decay"])
        at = grokking_checkpoints()
        marks = set(at.tolist())
        train_acc, test_acc, norm, train_ok, test_ok = [], [], [], [], []
        for step in range(s["updates"] + 1):
            if step in marks:
                ok = _mlp_forward(P, x, depth)[-1].argmax(-1) == y
                answers = _mlp_forward(P, xt, depth)[-1].argmax(-1)
                train_acc.append(ok.mean(1))
                test_acc.append((answers == yt).mean(1))
                norm.append(np.sqrt(sum((P["W%d" % k] ** 2).sum((1, 2)) for k in range(depth))))
                train_ok.append(ok[::seeds])
                test_ok.append((answers == yt)[::seeds])
            if step == s["updates"]:
                break
            ix = np.stack([rng.choice(len(x), s["batch"], replace=False) for rng in rngs] * arms)
            opt.step(P, _mlp_grads(P, x[ix], onehot[ix], depth))

        def by_arm(rows):
            return np.asarray(rows).T.reshape(arms, seeds, len(at))
        return dict(step=at, train=by_arm(train_acc), test=by_arm(test_acc), norm=by_arm(norm),
                    train_ok=np.asarray(train_ok).transpose(1, 0, 2),
                    test_ok=np.asarray(test_ok).transpose(1, 0, 2),
                    answers=answers[::seeds])

    settings = dict(GROKKING, seed=SEED, mnist=mnist.SHA256, init="he_normal x scale",
                    sources=realdata.source_hash("optim.py"))
    return cached_npz("study-training-grokking", settings, build)


def first_reaching(curve, level):
    """Index of the first checkpoint at which ``curve`` is at least ``level`` (-1: never)."""
    hit = np.flatnonzero(np.asarray(curve) >= level)
    return int(hit[0]) if len(hit) else -1


def grokking_delays(r=None):
    """For every run ``[arm, seed]``: the update at which training accuracy first reaches 99%,
    the update at which test accuracy first reaches 85%, test accuracy at the first of those,
    and whether the test level was reached at all (if not, its update is the last one)."""
    r = grokking_runs() if r is None else r
    s = GROKKING
    arms, seeds, _ = r["train"].shape
    fitted, generalised = np.zeros((arms, seeds)), np.zeros((arms, seeds))
    test_then, reached = np.zeros((arms, seeds)), np.zeros((arms, seeds), bool)
    for a in range(arms):
        for seed in range(seeds):
            i = first_reaching(r["train"][a, seed], s["fitted_at"])
            j = first_reaching(r["test"][a, seed], s["generalised_at"])
            fitted[a, seed] = r["step"][i]
            generalised[a, seed] = r["step"][j]
            test_then[a, seed] = r["test"][a, seed, i]
            reached[a, seed] = i >= 0 and j >= 0
    return dict(fitted=fitted, generalised=generalised, test_then=test_then, reached=reached)


def grokking():
    from . import mnist

    nl = _nl()
    s, r = GROKKING, grokking_runs()
    x, y, _xt, _yt, train_meta, test_meta = grokking_data()
    d = grokking_delays(r)
    labels = ["%s: training" % GROKKING_ARMS[0], "%s: test" % GROKKING_ARMS[0],
              "%s: training" % GROKKING_ARMS[1], "%s: test" % GROKKING_ARMS[1]]
    return nl._study(
        "Grokking on MNIST: test accuracy arrives long after training accuracy",
        "MNIST · 1,000 training images · weights started 3x larger · three seeds",
        [
            nl._panel(
                "Real handwritten inputs",
                "14x14 area-averaged MNIST images; the official test images never update weights.",
                "images", values=x[:6].reshape(-1, s["side"], s["side"]),
                labels=[str(v) for v in y[:6]]),
            nl._panel(
                "Training and test accuracy",
                "Seed means at log-spaced checkpoints; every seed is kept.",
                "curve", x=r["step"],
                ys=np.stack([r["train"][0].mean(0), r["test"][0].mean(0),
                             r["train"][1].mean(0), r["test"][1].mean(0)]),
                labels=labels, train=r["train"], test=r["test"]),
            nl._panel(
                "Weight norm during training",
                "The norm of all weight matrices; weight decay shrinks the large start.",
                "curve", x=r["step"], ys=r["norm"].mean(1), labels=list(GROKKING_ARMS),
                each=r["norm"]),
            nl._panel(
                "When each accuracy arrives",
                "The first checkpoint at 99% training accuracy and the first at 85% test "
                "accuracy, as seed means; every seed is kept.",
                "bars",
                values=np.r_[d["fitted"][0].mean(), d["generalised"][0].mean(),
                             d["fitted"][1].mean(), d["generalised"][1].mean()],
                labels=labels, fitted=d["fitted"], generalised=d["generalised"],
                test_then=d["test_then"], reached=d["reached"]),
        ],
        source="MNIST official train/test splits",
        settings=dict(s, arms=list(GROKKING_ARMS), optimizer="AdamW",
                      resolution="14x14 area average", init="He-normal times the scale"),
        dataset="MNIST",
        dataset_sha256=mnist.SHA256,
        dataset_provenance={"train": train_meta, "test": test_meta},
        designs_tried=GROKKING_DESIGNS,
        reference="Liu, Michaud and Tegmark, Omnigrok: Grokking Beyond Algorithmic Data (2022)",
    )


# ---------------------------------------------------------------------- 26: scaling laws

SCALING = dict(
    version=2,
    widths=(8, 16, 32, 64), layers=2, heads=4, ff_ratio=4,
    length=16, batch=32, rate=0.003, clip=1.0,
    levels=10, top_updates=640, least_updates=8,
    most_updates=(3200, 2100, 2400, 640),
    seeds=3, held_out_windows=512, fit_levels=7,
)
SCALING_DESIGNS = [
    "first version (next_language._scaling): 5 widths x 6 numbers of distinct excerpts, 180 "
    "updates each, one seed; compute varied only with model size and the fitted slope was +0.08",
    "pilot, one seed: widths 8/16/32/64, two layers, 1,500 updates of 32 random windows of 32 "
    "characters, held-out loss at ten checkpoints; loss fell with compute for every width",
    "ladder, fixed before it was run: each width trained to compute budgets that double, three "
    "seeds, power law fitted on the seven cheapest budgets and tested on the three dearest. "
    "First ladder: at most 1,500 / 1,000 / 1,200 / 640 updates for the four widths. It stopped "
    "each smaller model just where the next size was about to overtake it, so 'the best size "
    "at each budget' only reflected which models had been run (27 runs)",
    "this ladder: the smaller models run on to 3,098 / 2,000 / 2,342 updates so that every "
    "budget but the first and the last three compares at least three sizes (31 runs); the "
    "fit rule is unchanged",
]


def _scaling_config(width):
    from . import tinylm as lm

    s = SCALING
    return lm.Config(vocab=len(realdata.vocabulary()), width=width, heads=s["heads"],
                     kv_heads=s["heads"], layers=s["layers"], ff=s["ff_ratio"] * width,
                     max_len=s["length"])


def scaling_plan():
    """The ladder: ``weights[w]``, and for each width and compute level the number of updates
    that spends that level's budget (0 where the width is not run at that level). Compute is
    estimated as 6 x weights x characters processed; level ``j`` is the top budget (the widest
    model's full run) halved ``levels - 1 - j`` times."""
    from . import tinylm as lm

    s = SCALING
    weights = np.array([sum(v.size for v in lm.init(_scaling_config(w), 0).values())
                        for w in s["widths"]])
    per_update = 6.0 * weights * s["batch"] * s["length"]
    budget = per_update[-1] * s["top_updates"] / 2.0 ** np.arange(s["levels"] - 1, -1, -1)
    updates = np.round(budget[None, :] / per_update[:, None]).astype(int)
    ok = (updates >= s["least_updates"]) & (updates <= np.array(s["most_updates"])[:, None])
    return weights, np.where(ok, updates, 0)


@lru_cache(maxsize=1)
def scaling_runs():
    """Held-out loss ``[seed, width, level]`` of every model at every compute budget it was
    trained to (0 where not run), with the updates, characters and compute actually spent.

    A model's budgets are read off one training run per seed: the rate is constant and the
    batches (random windows of the training split) come in the same order, so the weights
    after ``n`` updates are exactly those of a run that stopped at ``n``. The longest runs
    read more characters than the split holds, so they see some text more than once."""
    from . import tinylm as lm

    s = SCALING
    weights, updates = scaling_plan()

    def build():
        train = realdata.encode(realdata.text("train"))
        held = realdata.text_sequences(s["held_out_windows"], s["length"] + 1, "test", SEED + 1)
        loss = np.zeros((s["seeds"],) + updates.shape)
        for seed in range(s["seeds"]):
            for w, width in enumerate(s["widths"]):
                config = _scaling_config(width)
                P = lm.init(config, SEED + seed)
                rng = np.random.default_rng(SEED + 100 + seed)
                opt = AdamW(P, s["rate"])
                level_at = {int(n): j for j, n in enumerate(updates[w]) if n}
                for step in range(1, int(updates[w].max()) + 1):
                    starts = rng.integers(0, len(train) - s["length"] - 1, s["batch"])
                    seq = train[starts[:, None] + np.arange(s["length"] + 1)]
                    _, G = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], config)
                    length = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
                    r = min(1.0, s["clip"] / max(length, 1e-12))
                    opt.step(P, {k: g * r for k, g in G.items()})
                    if step in level_at:
                        loss[seed, w, level_at[step]] = lm.loss(
                            P, held[:, :-1], held[:, 1:], config)
        characters = updates * s["batch"] * s["length"]
        return dict(loss=loss, weights=weights, updates=updates, characters=characters,
                    compute=6.0 * weights[:, None] * characters)

    settings = dict(SCALING, seed=SEED, dataset=realdata.TEXT_SHA256,
                    sources=realdata.source_hash("tinylm.py", "optim.py"))
    return cached_npz("study-training-scaling", settings, build)


def scaling_frontier(r=None):
    """What the sweep says, from seed means: at each compute level the best width that was run
    there and its loss; a power law ``loss = a x compute^slope`` fitted by least squares in
    log-log to the ``fit_levels`` cheapest levels of that frontier; and its predictions for the
    dearer levels, which the fit never saw."""
    r = scaling_runs() if r is None else r
    ran = r["updates"] > 0
    mean = r["loss"].mean(0)
    masked = np.where(ran, mean, np.inf)
    best = masked.argmin(0)
    levels = np.arange(mean.shape[1])
    front = masked[best, levels]
    compute = r["compute"][best, levels]
    k = SCALING["fit_levels"]
    slope, intercept = np.polyfit(np.log(compute[:k]), np.log(front[:k]), 1)
    predicted = np.exp(intercept + slope * np.log(compute))
    return dict(ran=ran, mean=mean, best=best, loss=front, compute=compute, slope=float(slope),
                intercept=float(intercept), predicted=predicted,
                error=np.abs(predicted[k:] - front[k:]))


def scaling_laws():
    nl = _nl()
    s, r = SCALING, scaling_runs()
    f = scaling_frontier(r)
    ran = f["ran"]
    lo, hi = r["loss"].min(0), r["loss"].max(0)
    w_of, level_of = np.nonzero(ran)
    k = s["fit_levels"]
    return nl._study(
        "Scaling: four model sizes, each trained to a ladder of compute budgets",
        "TinyShakespeare · two-layer transformers · random training windows · three seeds",
        [
            nl._panel(
                "Held-out loss by model size and compute budget",
                "Rows are widths, columns compute budgets that double; seed means, 0 where a "
                "width was not run at that budget.",
                "matrix", values=np.where(ran, f["mean"], 0.0), ran=ran, lowest=lo, highest=hi,
                updates=r["updates"], characters=r["characters"], compute=r["compute"],
                weights=r["weights"]),
            nl._panel(
                "Compute against held-out loss",
                "One point per run (seed mean): log10 of estimated compute and loss; classes "
                "are widths.",
                "points", xy=np.c_[np.log10(r["compute"][ran]), f["mean"][ran]], classes=w_of,
                level=level_of),
            nl._panel(
                "Power law fitted to the cheap budgets, tested on the dear ones",
                "Log loss of the best model at each budget; the line is fitted to the first "
                "%d budgets only." % k,
                "curve", x=np.log(f["compute"]),
                ys=np.stack([np.log(f["loss"]), np.log(f["predicted"])]),
                labels=["log measured loss", "power law from the %d cheapest" % k],
                slope=np.array(f["slope"]), held_out_error=f["error"],
                notes=("A descriptive fit of 10 points over a 512-fold range of compute; "
                       "not a universal law.",)),
            nl._panel(
                "Best model size at each budget",
                "The width with the lowest mean held-out loss among those run at each budget, "
                "and that loss.",
                "curve", x=f["compute"], ys=f["loss"][None], labels=["best loss at the budget"],
                best=f["best"], best_weights=r["weights"][f["best"]]),
        ],
        source="Collected TinyShakespeare text",
        settings=dict(s, runs=int(ran.sum()), weights=[int(v) for v in r["weights"]],
                      compute_estimate="6 x weights x characters processed"),
        designs_tried=SCALING_DESIGNS,
        **nl._text_meta(evaluation="512 windows of the test split, the same for every run"),
    )


STUDIES = {
    "warmup-and-clipping": warmup_and_clipping,
    "grokking": grokking,
    "scaling-laws": scaling_laws,
}
