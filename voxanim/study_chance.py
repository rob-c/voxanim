"""The experiments behind animations 115-117: chance that has structure, and what a
seed decides (see :func:`voxanim.next_catalog.redesigned`).

* ``markov-chains``: a chain counted from Tiny Shakespeare. Which character follows
  which in the training split becomes a transition matrix (four character classes for
  the drawn example, all 65 characters for the measurements). Exact arithmetic on that
  matrix gives its stationary vector, how fast it forgets its start and its entropy
  rate; a seeded walk gives the same stationary vector by counting visits. Chains of
  order 0 to 6 are fitted on the training split and scored on the test split.
* ``seeds-and-training-outcomes``: one small MNIST network trained from many seeds.
  Each run takes three generators (initial weights, order of the data, dropout masks),
  so one source can be varied with the other two held fixed. The spread of held-out
  accuracy, the overlap of the runs' mistakes, and how often a few seeds name the
  wrong winner of two settings whose real difference is small (resampling the runs).
* ``seeds-and-sampled-text``: the pinned GPT-2 continuing held-out prompts. A seed
  gives one uniform number per token; the number picks the token from the (reshaped)
  next-token distribution. Twenty seeds, six prompts, five temperatures and four
  truncated variants: where seeds part, how many distinct texts come out, and the
  log-probability GPT-2 gives them.

Every random draw comes from ``numpy.random.default_rng(seed)`` with the seed recorded;
a draw from a distribution is always :func:`draw` (the inverse of its running sum at a
uniform number), so each can be followed by hand. Everything the drawings need is
recorded here: :mod:`voxanim.mech_chance` computes nothing but layout.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import realdata
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-chance-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


def draw(p, u):
    """The index a uniform number ``u`` in [0, 1) picks from the distribution ``p``: the
    first whose running sum passes ``u``. Every random choice in this module is this."""
    c = np.cumsum(p)
    k = int(np.searchsorted(c, u * c[-1], side="right"))
    return min(k, int(np.flatnonzero(p)[-1]))


def entropy(p):
    """Shannon entropy of a distribution, in bits."""
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tv(p, q):
    """Total-variation distance: the largest difference in probability the two
    distributions give any event (0: the same; 1: nothing in common)."""
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum(-1))


# ====================================================================== 115: Markov chains
MARKOV_SEED = 115                 # of the walk drawn and of every generated text
WALK_SEEDS = (115, 116, 117, 118, 119)
WALK_STEPS = 20000
WALK_AT = (10, 30, 100, 300, 1000, 3000, 10000, 20000)
CLASS_NAMES = ("vowel", "consonant", "space", "mark")
MIX_STARTS = ("q", "e", " ", "\n")
MIX_STEPS = 12
ORDERS = tuple(range(7))
ALPHAS = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
TEXT_LENGTH = 56                  # characters generated per order
HEAD = 26                         # characters of the training text counted on screen


@functools.lru_cache(maxsize=1)
def alphabet():
    """The characters of the training split, sorted: the chain's states."""
    return "".join(sorted(set(realdata.text("train"))))


@functools.lru_cache(maxsize=3)
def char_ids(split):
    """A split of the text as state numbers."""
    table = {c: i for i, c in enumerate(alphabet())}
    return np.array([table[c] for c in realdata.text(split)], np.int64)


def char_classes():
    """Each character's class: 0 vowel, 1 consonant, 2 space, 3 mark (punctuation, line
    break, digit)."""
    def kind(c):
        if c.lower() in "aeiou":
            return 0
        if c.isalpha():
            return 1
        return 2 if c == " " else 3
    return np.array([kind(c) for c in alphabet()], np.int64)


def transitions(seq, n):
    """``counts[a, b]``: how often state ``b`` follows state ``a`` in ``seq``."""
    seq = np.asarray(seq, np.int64)
    return np.bincount(seq[:-1] * n + seq[1:], minlength=n * n).reshape(n, n)


def row_normalise(counts):
    c = np.asarray(counts, float)
    return c / c.sum(1, keepdims=True)


def stationary(P):
    """The matrix's fixed vector: the distribution ``pi`` with ``pi P = pi``, solved as
    linear equations (``pi (P - I) = 0`` with the entries summing to 1)."""
    n = len(P)
    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    return np.linalg.lstsq(A, b, rcond=None)[0]


def entropy_rate(P, pi):
    """Entropy rate of the chain, bits per step: the entropy of each state's row of
    next-state chances, weighted by how often the chain is in that state."""
    return float(sum(w * entropy(row) for w, row in zip(pi, P)))


def forgetting(P, pi, starts, steps):
    """After 0..``steps`` steps from each start, by exact matrix powers: the distance to
    stationary, and the distribution itself."""
    out = np.zeros((len(starts), steps + 1))
    dist = np.zeros((len(starts), steps + 1, len(P)))
    for i, s in enumerate(starts):
        p = np.zeros(len(P))
        p[s] = 1.0
        for t in range(steps + 1):
            out[i, t], dist[i, t] = tv(p, pi), p
            p = p @ P
    return out, dist


def walk(P, start, us):
    """The states a walker visits from ``start``, one uniform number per step."""
    path = np.empty(len(us) + 1, np.int64)
    path[0] = s = int(start)
    cum = np.cumsum(P, axis=1)
    last = P.shape[1] - 1
    for t, u in enumerate(us):
        s = min(int(np.searchsorted(cum[s], u * cum[s, -1], side="right")), last)
        path[t + 1] = s
    return path


def context_keys(seq, k, n):
    """One integer per position ``k..``: the ``k`` states before it, read as a number
    in base ``n``."""
    key = np.zeros(len(seq) - k, np.int64)
    for j in range(k):
        key = key * n + seq[j:len(seq) - k + j]
    return key


def fit_order(seq, k, n):
    """Counts of an order-``k`` chain: every context seen and what followed it."""
    ctx = context_keys(seq, k, n)
    pair_keys, pair_counts = np.unique(ctx * n + seq[k:], return_counts=True)
    ctx_keys, ctx_counts = np.unique(ctx, return_counts=True)
    return pair_keys, pair_counts, ctx_keys, ctx_counts


def _lookup(keys, counts, q):
    i = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
    return np.where(keys[i] == q, counts[i], 0)


def order_bits(model, seq, k, n, alpha, skip):
    """Cross-entropy in bits per character of ``seq[skip:]`` under the fitted counts,
    each chance ``(count + alpha) / (context count + alpha n)``; also the share of
    positions whose context was never seen."""
    pair_keys, pair_counts, ctx_keys, ctx_counts = model
    ctx = context_keys(seq, k, n)[skip - k:]
    c = _lookup(pair_keys, pair_counts, ctx * n + seq[skip:])
    t = _lookup(ctx_keys, ctx_counts, ctx)
    with np.errstate(divide="ignore", invalid="ignore"):
        bits = -np.log2((c + alpha) / (t + alpha * n))
    return float(bits.mean()), float((t == 0).mean())


def generate(model, k, n, context, us):
    """Text from an order-``k`` chain's unsmoothed counts: one uniform number per
    character. Returns the new states and how often a context had no successor (the
    walk then restarts from ``context``)."""
    pair_keys, pair_counts, _, _ = model
    out, dead = [int(c) for c in context], 0
    for u in us:
        key = 0
        for c in out[len(out) - k:] if k else ():
            key = key * n + c
        lo, hi = np.searchsorted(pair_keys, [key * n, (key + 1) * n])
        if lo == hi:
            dead += 1
            out.extend(int(c) for c in context)
            continue
        out.append(int(pair_keys[lo + draw(pair_counts[lo:hi], u)] - key * n))
    return np.array(out[len(context):len(context) + len(us)], np.int64), dead


def _markov_build():
    chars = alphabet()
    n = len(chars)
    train, val, test = (char_ids(s) for s in ("train", "validation", "test"))
    kinds = char_classes()
    # ---- the two chains: four classes (drawn) and all characters (measured)
    C4 = transitions(kinds[train], 4)
    P4 = row_normalise(C4)
    pi4 = stationary(P4)
    C = transitions(train, n)
    P = row_normalise(C)
    pi = stationary(P)
    eig = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    eig4 = np.sort(np.abs(np.linalg.eigvals(P4)))[::-1]
    starts = [chars.index(c) for c in MIX_STARTS]
    # ---- the walk: visits counted against the fixed vector
    at = np.array(WALK_AT)
    walk_tv = np.zeros((len(WALK_SEEDS), len(at)))
    shares = np.zeros((len(at), 4))
    path0 = us0 = None
    for i, seed in enumerate(WALK_SEEDS):
        us = np.random.default_rng(seed).random(WALK_STEPS)
        path = walk(P4, kinds[train[0]], us)
        for j, m in enumerate(at):
            share = np.bincount(path[1:m + 1], minlength=4) / m
            walk_tv[i, j] = tv(share, pi4)
            if i == 0:
                shares[j] = share
        if i == 0:
            path0, us0 = path[:201], us[:200]
    # ---- order: fitted on the training split, smoothing chosen on validation
    skip = max(ORDERS)
    us = np.random.default_rng(MARKOV_SEED).random(TEXT_LENGTH)
    held, fitted, unseen, seen, alpha_at, val_bits, texts, dead = [], [], [], [], [], [], [], 0
    for k in ORDERS:
        model = fit_order(train, k, n)
        by_alpha = [order_bits(model, val, k, n, a, skip)[0] for a in ALPHAS]
        best = int(np.argmin(by_alpha))
        bits, miss = order_bits(model, test, k, n, ALPHAS[best], skip)
        held.append(bits)
        unseen.append(miss)
        fitted.append(order_bits(model, train, k, n, 0.0, skip)[0])
        seen.append(len(model[2]))
        alpha_at.append(ALPHAS[best])
        val_bits.append(by_alpha)
        new, d = generate(model, k, n, train[:k], us)
        dead += d
        texts.append("".join(chars[c] for c in new))
    forget, forget_dist = forgetting(P, pi, starts, MIX_STEPS)
    return dict(
        alphabet=np.array(chars), classes=kinds, head=np.array(realdata.text("train")[:HEAD]),
        counts4=C4, P4=P4, pi4=pi4, counts=C, P=P, pi=pi,
        first_in=np.bincount(train, minlength=n) / len(train),
        second4=np.array(eig4[1]), second=np.array(eig[1]),
        forget=forget, forget_dist=forget_dist,
        forget4=forgetting(P4, pi4, range(4), MIX_STEPS)[0],
        row_bits=np.array([entropy(row) for row in P]),
        walk_path=path0, walk_us=us0, walk_shares=shares, walk_tv=walk_tv,
        h_uniform=np.array(np.log2(n)), h_single=np.array(entropy(pi)),
        h_rate=np.array(entropy_rate(P, pi)),
        h4_single=np.array(entropy(pi4)), h4_rate=np.array(entropy_rate(P4, pi4)),
        held=np.array(held), fitted=np.array(fitted), unseen=np.array(unseen),
        contexts=np.array(seen), alpha=np.array(alpha_at), val_bits=np.array(val_bits),
        texts=np.array(texts), text_us=us, dead_ends=np.array(dead),
        sizes=np.array([len(train), len(val), len(test)]),
    )


def markov_data():
    """Everything ``markov-chains`` measures and draws."""
    return _cache("markov", _markov_build,
                  (_markov_build, walk, fit_order, order_bits, generate, stationary,
                   forgetting, char_classes, draw),
                  seed=MARKOV_SEED, walk_seeds=WALK_SEEDS, steps=WALK_STEPS, orders=ORDERS,
                  alphas=ALPHAS, length=TEXT_LENGTH, dataset=realdata.TEXT_SHA256)


def markov_chains():
    d = markov_data()
    chars = str(d["alphabet"])
    best = int(np.argmin(d["held"]))
    return Study(
        "Markov chains",
        "A chain counted from Tiny Shakespeare: its fixed vector, its memory, its entropy "
        "rate, and orders 0 to %d on held-out text" % max(ORDERS),
        (
            Panel("Transition matrix of four character classes",
                  "values[a, b]: the chance class b follows class a (vowel, consonant, "
                  "space, mark) in the training split: counts / row sums. full: the same "
                  "for all %d characters. Exact arithmetic on counts." % len(chars),
                  "matrix", dict(values=d["P4"], counts=d["counts4"], full=d["P"],
                                 stationary=d["pi4"], labels=list(CLASS_NAMES))),
            Panel("Forgetting the start",
                  "ys: total-variation distance between the %d-character chain's "
                  "distribution after x steps and its fixed vector, from four starts "
                  "(exact matrix powers). walk_tv: the same distance for the share of "
                  "visits of a seeded walk on the class chain, per seed and walk length."
                  % len(chars), "curve",
                  dict(x=np.arange(MIX_STEPS + 1), ys=d["forget"],
                       labels=["from %r" % c for c in MIX_STARTS], classes=d["forget4"],
                       distributions=d["forget_dist"],
                       walk_at=np.array(WALK_AT), walk_tv=d["walk_tv"],
                       walk_shares=d["walk_shares"])),
            Panel("Entropy, bits per character",
                  "Shannon entropy of %d equally likely characters; of the chain's fixed "
                  "vector (single characters); and the chain's entropy rate (the next "
                  "character given the current one). Exact for the fitted chain." % len(chars),
                  "bars", dict(values=np.array([d["h_uniform"], d["h_single"], d["h_rate"]]),
                               labels=["uniform", "single characters", "entropy rate"],
                               row_bits=d["row_bits"])),
            Panel("Order",
                  "ys[0]: cross-entropy on the test split of chains of order x fitted on "
                  "the training split (add-alpha smoothing, alpha chosen on validation). "
                  "ys[1]: the same counts scored on the text they were counted from, "
                  "unsmoothed: a plug-in estimate, biased low. contexts: distinct contexts "
                  "seen; unseen: share of test positions whose context never occurred.",
                  "curve", dict(x=np.array(ORDERS), ys=np.array([d["held"], d["fitted"]]),
                                labels=["held-out", "on the fitted text"],
                                contexts=d["contexts"], unseen=d["unseen"], alpha=d["alpha"])),
        ),
        dict(
            seed=MARKOV_SEED, seeds=list(WALK_SEEDS), review_only=True,
            source="exact arithmetic on counts from Tiny Shakespeare; held-out "
                   "cross-entropy on its test split",
            dataset="Tiny Shakespeare / char-rnn",
            dataset_provenance={k: realdata.text_metadata(k)
                                for k in ("train", "validation", "test")},
            evidence=dict(
                exact="transition matrices, fixed vectors, distance to stationary by matrix "
                      "powers, entropy rate of the fitted order-1 chain",
                statistics_of_a_seeded_generator="visit shares of the walk; generated text",
                measured_on_held_out_text="cross-entropy by order (test split)"),
            where_the_seed_enters="numpy.random.default_rng(seed).random(n): one uniform "
                                  "number per step, turned into a state by study_chance.draw",
            characters=len(chars), split_sizes=[int(v) for v in d["sizes"]],
            stationary_vs_frequency=float(np.abs(d["pi"] - d["first_in"]).max()),
            second_eigenvalue=dict(classes=float(d["second4"]), characters=float(d["second"])),
            entropy_bits=dict(uniform=float(d["h_uniform"]), single=float(d["h_single"]),
                              rate=float(d["h_rate"]), classes_single=float(d["h4_single"]),
                              classes_rate=float(d["h4_rate"])),
            orders=list(ORDERS), held_out_bits=[float(v) for v in d["held"]],
            fitted_text_bits=[float(v) for v in d["fitted"]],
            contexts_seen=[int(v) for v in d["contexts"]],
            unseen_context_share=[float(v) for v in d["unseen"]],
            smoothing=[float(v) for v in d["alpha"]], best_order=ORDERS[best],
            generated={str(k): str(t) for k, t in zip(ORDERS, d["texts"])},
            dead_ends=int(d["dead_ends"]),
            limitations=[
                "Add-alpha smoothing is the simplest way to give unseen characters a "
                "chance; a model that backs off to shorter contexts would turn up later.",
                "The entropy rate is that of the fitted order-1 chain, not of English: the "
                "held-out cross-entropy of each order is an upper bound on the text's own.",
                "The four-class chain is a drawing aid; a class sequence is not exactly a "
                "first-order chain.",
            ],
            designs_tried=[
                "Only this design was run: four classes for the drawing, 65 characters for "
                "every measurement, orders 0-6, alpha from a grid of seven on validation.",
            ],
        ),
    )


# ====================================================================== 116: seeds, training
SPLIT_SEED = 17                   # of the data split
N_TRAIN = 20000
WIDTH, WIDTH_B = 64, 72           # the network, and the slightly larger one it is compared to
DROP, EPOCHS, BATCH, RATE = 0.2, 3, 64, 1e-3
EVERY = 78                        # updates between held-out measurements
N_RUNS = 64                       # seeds per setting with every source varying
N_SOURCE = 32                     # seeds per source with the other two fixed
BASES = (0, 1)                    # the fixed seeds of the other two sources
SOURCES = ("initial weights", "order of the data", "dropout masks")
COMPARE_N = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
COMPARE_DRAWS = 20000
COMPARE_SEED = 116


@functools.lru_cache(maxsize=1)
def mnist_data():
    """``(x, y, xt, yt, meta)``: a training split and all official test images."""
    x, y, m = realdata.mnist_split(N_TRAIN, "train", 28, SPLIT_SEED)
    xt, yt, mt = realdata.mnist_split(10000, "test", 28, SPLIT_SEED)
    keep = lambda d: {k: v for k, v in d.items() if k != "example_indices"}  # noqa: E731
    return x, y, xt, yt, dict(train=keep(m), held_out=keep(mt))


def generators(init_seed, order_seed, drop_seed):
    """The three generators of one run: where its seeds enter."""
    return (np.random.default_rng([int(init_seed), 0]),
            np.random.default_rng([int(order_seed), 1]),
            np.random.default_rng([int(drop_seed), 2]))


def initial_weights(rng, width):
    return dict(W0=rng.normal(0, np.sqrt(2.0 / 784), (784, width)), b0=np.zeros(width),
                W1=rng.normal(0, np.sqrt(2.0 / width), (width, 10)), b1=np.zeros(10))


def weights_hash(P):
    h = hashlib.sha256()
    for k in sorted(P):
        h.update(np.ascontiguousarray(P[k]).tobytes())
    return h.hexdigest()


def train_run(width, init_seed, order_seed, drop_seed):
    """One training run: 784 -> ``width`` ReLU -> dropout -> 10, Adam. Returns
    ``(weights, curve, wrong)``: held-out accuracy every ``EVERY`` updates and which
    test digits the finished network gets wrong."""
    from .optim import Adam

    x, y, xt, yt, _ = mnist_data()
    r_init, r_order, r_drop = generators(init_seed, order_seed, drop_seed)
    P = initial_weights(r_init, width)
    opt, eye, curve, step = Adam(P, RATE), np.eye(10), [], 0

    def wrong():
        return (np.maximum(xt @ P["W0"] + P["b0"], 0) @ P["W1"] + P["b1"]).argmax(1) != yt
    for _ in range(EPOCHS):
        perm = r_order.permutation(len(x))
        for s in range(0, len(x) - BATCH + 1, BATCH):
            idx = perm[s:s + BATCH]
            xb = x[idx]
            h = np.maximum(xb @ P["W0"] + P["b0"], 0)
            mask = (r_drop.random(h.shape) >= DROP) / (1 - DROP)
            hd = h * mask
            z = hd @ P["W1"] + P["b1"]
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            d = (p - eye[y[idx]]) / BATCH
            dh = (d @ P["W1"].T) * mask * (h > 0)
            opt.step(P, dict(W0=xb.T @ dh, b0=dh.sum(0), W1=hd.T @ d, b1=d.sum(0)))
            step += 1
            if step % EVERY == 0:
                curve.append(1.0 - wrong().mean())
    return P, np.array(curve), wrong()


def _run_chunk(jobs):
    """Several runs in one process: accuracies, curves, packed mistakes, weight hashes."""
    acc, curves, wrong, hashes = [], [], [], []
    for job in jobs:
        P, curve, miss = train_run(*job)
        acc.append(1.0 - miss.mean())
        curves.append(curve)
        wrong.append(np.packbits(miss))
        hashes.append(weights_hash(P))
    return np.array(acc), np.array(curves), np.array(wrong), hashes


def run_plan():
    """Every run as ``(width, init seed, order seed, dropout seed)``, and the slice of
    the plan each group occupies."""
    plan, groups = [], {}

    def add(name, jobs):
        groups[name] = (len(plan), len(plan) + len(jobs))
        plan.extend(jobs)
    add("all", [(WIDTH, s, s, s) for s in range(N_RUNS)])
    add("wider", [(WIDTH_B, s, s, s) for s in range(N_RUNS)])
    for b in BASES:
        varied = range(100, 100 + N_SOURCE)
        add("init/%d" % b, [(WIDTH, s, b, b) for s in varied])
        add("order/%d" % b, [(WIDTH, b, s, b) for s in varied])
        add("drop/%d" % b, [(WIDTH, b, b, s) for s in varied])
    return plan, groups


def wrong_winner(better, worse, ns, draws, rng):
    """How often ``n`` runs of each setting, drawn with replacement, put the truly worse
    setting ahead (a tie counts half). ``n`` = 1 is every pair of runs, exactly."""
    out = []
    for n in ns:
        if n == 1:
            a, b = better[None, :], worse[:, None]
        else:
            a = better[rng.integers(0, len(better), (draws, n))].mean(1)
            b = worse[rng.integers(0, len(worse), (draws, n))].mean(1)
        out.append(float((b > a).mean() + 0.5 * (b == a).mean()))
    return np.array(out)


def _seeds_build():
    plan, _ = run_plan()
    chunks = [plan[i::6] for i in range(6)]
    parts = _parallel(_run_chunk, [(c,) for c in chunks])
    acc = np.zeros(len(plan))
    curves = np.zeros((len(plan), EPOCHS * (N_TRAIN // BATCH) // EVERY))
    wrong = np.zeros((len(plan), 1250), np.uint8)
    hashes = [""] * len(plan)
    for i, (a, c, w, h) in enumerate(parts):
        acc[i::6], curves[i::6], wrong[i::6] = a, c, w
        hashes[i::6] = h
    # ---- the same seed again, in this process, twice; and another seed
    P0, _, _ = train_run(WIDTH, 0, 0, 0)
    P0b, _, _ = train_run(WIDTH, 0, 0, 0)
    P1, _, _ = train_run(WIDTH, 1, 1, 1)
    x, _, xt, yt, _ = mnist_data()
    shown = {}
    for s in (0, 1):                       # what one seed decides, for the drawing
        r_init, r_order, r_drop = generators(s, s, s)
        W = initial_weights(r_init, WIDTH)["W0"]
        shown["init_%d" % s] = W[406:414, :8]
        shown["order_%d" % s] = x[r_order.permutation(len(x))[:6]].reshape(6, 28, 28)
        shown["mask_%d" % s] = (r_drop.random((BATCH, WIDTH)) >= DROP)[0].reshape(8, 8)
    return dict(
        acc=acc, curves=curves, wrong=wrong, hashes=np.array(hashes),
        same_seed_equal=np.array(all(np.array_equal(P0[k], P0b[k]) for k in P0)),
        same_seed_max_diff=np.array(max(float(np.abs(P0[k] - P0b[k]).max()) for k in P0)),
        other_seed_differs=np.array(float(np.abs(P0["W0"] - P1["W0"]).mean())),
        hash_again=np.array([weights_hash(P0), weights_hash(P0b), weights_hash(P1)]),
        weights_shown=np.array([P0["W0"][406:414, 0], P0b["W0"][406:414, 0],
                                P1["W0"][406:414, 0]]),
        test_images=xt.reshape(-1, 28, 28).astype(np.float32), test_labels=yt, **shown,
    )


def seeds_data():
    """Every run of ``seeds-and-training-outcomes`` (trained once, cached)."""
    return _cache("seeds", _seeds_build,
                  (_seeds_build, train_run, run_plan, generators, initial_weights, _run_chunk),
                  split_seed=SPLIT_SEED, n_train=N_TRAIN, widths=(WIDTH, WIDTH_B), drop=DROP,
                  epochs=EPOCHS, batch=BATCH, rate=RATE, every=EVERY, runs=N_RUNS,
                  per_source=N_SOURCE, bases=BASES)


@functools.lru_cache(maxsize=1)
def seeds_summary():
    """What the runs say: spread, mistakes, sources, and the comparison."""
    d = seeds_data()
    _, groups = run_plan()
    acc = d["acc"]
    part = lambda name: acc[slice(*groups[name])]                      # noqa: E731
    a, b = part("all"), part("wider")
    wrong = np.unpackbits(d["wrong"], axis=1)[:, :10000].astype(bool)
    main = wrong[slice(*groups["all"])]
    as_int = main.astype(np.int64)
    inter = as_int @ as_int.T
    union = main.sum(1)[:, None] + main.sum(1)[None] - inter
    pairs = np.triu_indices(N_RUNS, 1)
    times = main.sum(0)                                  # runs that get each digit wrong
    order = np.argsort(-times, kind="stable")
    order = order[times[order] > 0]
    spread = np.array([[part("%s/%d" % (key, base)).std(ddof=1) for base in BASES]
                       for key in ("init", "order", "drop")])
    better, worse = (b, a) if b.mean() > a.mean() else (a, b)
    rng = np.random.default_rng(COMPARE_SEED)
    share = wrong_winner(better, worse, COMPARE_N, COMPARE_DRAWS, rng)
    half = np.argsort(np.abs(times - N_RUNS / 2), kind="stable")[:8]
    pairs_shown = rng.integers(0, N_RUNS, (8, 2))        # one seed of each, eight times
    return dict(
        d=d, groups=groups, a=a, b=b, main=main, times=times, order=order,
        jaccard=float((inter[pairs] / union[pairs]).mean()),
        jaccard_range=(float((inter[pairs] / union[pairs]).min()),
                       float((inter[pairs] / union[pairs]).max())),
        disagree=float(((union - inter)[pairs] / 10000.0).mean()),
        always=int((times == N_RUNS).sum()), ever=int((times > 0).sum()),
        mean_wrong=float(main.sum(1).mean()),
        always_ids=np.flatnonzero(times == N_RUNS)[:8], half_ids=half,
        spread=spread, spread_all=float(a.std(ddof=1)),
        sources={key: np.array([part("%s/%d" % (key, base)) for base in BASES])
                 for key in ("init", "order", "drop")},
        wrong_share=share, pairs_shown=pairs_shown, gap=float(abs(b.mean() - a.mean())),
        gap_se=float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))),
        needed=next((n for n, s in zip(COMPARE_N, share) if s < 0.05), None),
    )


def seeds_and_training_outcomes():
    r = seeds_summary()
    d, a, b = r["d"], r["a"], r["b"]
    updates = EVERY * np.arange(1, d["curves"].shape[1] + 1)
    lo, hi = r["groups"]["all"]
    return Study(
        "One model, many seeds",
        "The same MNIST network trained from %d seeds; one source of randomness varied "
        "at a time; and how many seeds a comparison needs" % N_RUNS,
        (
            Panel("Held-out accuracy during training, one curve per seed",
                  "ys: accuracy on the 10,000 official test images every %d updates for "
                  "each of %d seeds (initial weights, data order and dropout masks all "
                  "from the seed)." % (EVERY, N_RUNS), "curve",
                  dict(x=updates, ys=d["curves"][lo:hi], final=a,
                       labels=["seed %d" % k for k in range(N_RUNS)])),
            Panel("Which test digits each run gets wrong",
                  "values[run, digit]: 1 where that run gets that test digit wrong, for "
                  "the %d digits at least one run gets wrong, most-often-wrong first. "
                  "times: in how many runs each is wrong." % r["ever"], "matrix",
                  dict(values=r["main"][:, r["order"]].astype(float),
                       times=r["times"][r["order"]], digits=r["order"])),
            Panel("Spread of accuracy by source of randomness",
                  "values: standard deviation of held-out accuracy over %d runs when only "
                  "the initial weights, only the data order, or only the dropout masks "
                  "change (mean of two choices of the fixed seeds; per_base has both), "
                  "then with all three changing (%d runs)." % (N_SOURCE, N_RUNS), "bars",
                  dict(values=np.append(r["spread"].mean(1), r["spread_all"]),
                       labels=[*SOURCES, "all three"], per_base=r["spread"],
                       init=r["sources"]["init"], order=r["sources"]["order"],
                       drop=r["sources"]["drop"], all=a)),
            Panel("How often a few seeds name the wrong winner",
                  "ys: share of resamples (with replacement, %d draws) in which the mean "
                  "of x runs of the setting that is worse over all %d runs is ahead of "
                  "the mean of x runs of the better one; x = 1 is every pair, exactly."
                  % (COMPARE_DRAWS, N_RUNS), "curve",
                  dict(x=np.array(COMPARE_N), ys=r["wrong_share"][None],
                       labels=["wrong winner"], width_a=a, width_b=b,
                       pairs_shown=r["pairs_shown"])),
        ),
        dict(
            seed=SPLIT_SEED, seeds=list(range(N_RUNS)), mnist=True, review_only=True,
            source="NumPy training on MNIST", dataset=mnist_data()[4],
            model="784 -> %d ReLU -> dropout %.1f -> 10 softmax; Adam %.0e, batch %d, %d "
                  "epochs of %d training images (%d updates)"
                  % (WIDTH, DROP, RATE, BATCH, EPOCHS, N_TRAIN, int(updates[-1])),
            where_the_seed_enters="three numpy.random.default_rng generators per run: "
                                  "[seed, 0] draws the initial weights, [seed, 1] each "
                                  "epoch's order, [seed, 2] every dropout mask",
            runs=dict(total=len(d["acc"]), every_source=N_RUNS, per_source=N_SOURCE,
                      fixed_seeds=list(BASES), varied_seeds=[100, 100 + N_SOURCE - 1]),
            accuracy=dict(mean=float(a.mean()), std=float(a.std(ddof=1)), min=float(a.min()),
                          max=float(a.max())),
            mistakes=dict(mean_per_run=r["mean_wrong"], overlap=r["jaccard"],
                          overlap_range=list(r["jaccard_range"]), disagree=r["disagree"],
                          wrong_in_every_run=r["always"], wrong_in_some_run=r["ever"]),
            spread_by_source={name: [float(v) for v in row]
                              for name, row in zip(SOURCES, r["spread"])},
            spread_all=r["spread_all"],
            comparison=dict(
                widths=[WIDTH, WIDTH_B], mean=[float(a.mean()), float(b.mean())],
                std=[float(a.std(ddof=1)), float(b.std(ddof=1))], gap=r["gap"],
                gap_standard_error=r["gap_se"], n=list(COMPARE_N),
                wrong_winner=[float(v) for v in r["wrong_share"]],
                seeds_for_under_5_percent=r["needed"]),
            reproducible=dict(
                same_seed_bit_identical=bool(d["same_seed_equal"]),
                same_seed_max_difference=float(d["same_seed_max_diff"]),
                same_as_the_worker_process=bool(str(d["hashes"][0]) == str(d["hash_again"][0])),
                other_seed_mean_difference=float(d["other_seed_differs"]),
                sha256=[str(v) for v in d["hash_again"]]),
            limitations=[
                "One small network and one training recipe: the size of the spread and "
                "the shares by source are this recipe's.",
                "Each source is varied with the other two held at two fixed seeds; the "
                "spreads are not additive, so they do not sum to the total.",
                "The comparison resamples %d runs per setting; the 'real' difference is "
                "the difference of those means, known to within its standard error."
                % N_RUNS,
                "Bit-identical weights are checked on one machine; another processor or "
                "linear-algebra library may round differently.",
            ],
            designs_tried=[
                "A pilot of 8 seeds at widths 32, 48, 64 and 96 showed gaps several times "
                "the seed spread: no seed could name the wrong winner.",
                "Widths 64, 72 and 80 were then piloted with 40 seeds; 64 against 72 (a "
                "gap of about one seed standard deviation) was chosen for the comparison "
                "before the resampling was run.",
            ],
        ),
    )


# ====================================================================== 117: seeds, sampling
PROMPTS, PROMPT_TOKENS = 6, 24
SAMPLE_SEEDS = tuple(range(20))
NEW_TOKENS = 12
RARE = 1e-3                       # a token GPT-2 gave less than this is counted as unlikely
# (temperature, top-k, top-p): five temperatures, then the two usual truncations
SETTINGS = ((0.0, 0, 0.0), (0.4, 0, 0.0), (0.7, 0, 0.0), (1.0, 0, 0.0), (1.3, 0, 0.0),
            (1.0, 40, 0.0), (1.0, 0, 0.9), (1.3, 40, 0.0), (1.3, 0, 0.9))
N_TEMPS = 5
DRAWN_PROMPT = 2                  # the prompt drawn (all six are measured)


def setting_name(setting):
    t, k, p = setting
    if k:
        return "T %.1f, top %d" % (t, k)
    if p:
        return "T %.1f, top-p %.1f" % (t, p)
    return "T %.1f" % t


def uniforms(seed, n=NEW_TOKENS):
    """Where the seed enters: one uniform number per token to be sampled."""
    return np.random.default_rng(int(seed)).random(n)


def reshape(sorted_p, setting):
    """The distribution actually sampled from, over tokens sorted most likely first:
    ``p ** (1 / T)`` renormalised (T = 0: everything on the most likely token), then cut
    to the top ``k`` tokens or to the smallest set holding ``top_p`` of the chance."""
    t, k, top_p = setting
    q = np.zeros_like(sorted_p)
    if t == 0:
        q[0] = 1.0
        return q
    with np.errstate(divide="ignore"):
        z = np.log(sorted_p) / t
    q = np.exp(z - z[0])
    q /= q.sum()
    if k:
        q[k:] = 0.0
    if top_p:
        q[int(np.searchsorted(np.cumsum(q), top_p)) + 1:] = 0.0
    return q / q.sum()


def sample_plain(prompt, seed, setting):
    """One continuation, the straightforward way: read the prompt, then one token at a
    time. Returns ``(tokens, log-probability GPT-2 gives each)``."""
    from . import study_decoding as D

    st = D.Stream(room=PROMPT_TOKENS + NEW_TOKENS + 1)
    x = st.feed(prompt)[-1]
    us, out, logp = uniforms(seed), [], []
    for u in us:
        p = D.probs(x)[0]
        order = np.argsort(-p, kind="stable")
        token = int(order[draw(reshape(p[order], setting), u)])
        out.append(token)
        logp.append(float(np.log(p[token])))
        x = st.feed([token])[-1]
    return np.array(out), np.array(logp)


def sample_prompt(index):
    """Every setting and seed for one prompt, sharing GPT-2 passes: continuations that
    agree so far are read once (a walk down the tree of what was sampled). Returns
    tokens, GPT-2's log-probability of each, the entropy in bits of the distribution
    each was drawn from, the number of passes, and the first step's distribution (the
    400 likeliest tokens, their chance under each setting, and how many tokens each
    setting keeps)."""
    from . import study_decoding as D

    prompt = D.heldout_prompts(PROMPTS, PROMPT_TOKENS)[index]
    U = np.array([uniforms(s) for s in SAMPLE_SEEDS])
    shape = (len(SETTINGS), len(SAMPLE_SEEDS), NEW_TOKENS)
    tokens, logp, bits = np.zeros(shape, np.int64), np.zeros(shape), np.zeros(shape)
    st = D.Stream(room=PROMPT_TOKENS + NEW_TOKENS + 1)
    first, passes = {}, [0]

    def visit(x, depth, members):
        p = D.probs(x)[0]
        passes[0] += 1
        order = np.argsort(-p, kind="stable")
        sp = p[order]
        if depth == 0:
            first["order"], first["p"] = order[:400], sp[:400]
            shapes = [reshape(sp, s) for s in SETTINGS]
            first["q"] = np.array([q[:400] for q in shapes])
            first["kept"] = np.array([int((q > 0).sum()) for q in shapes])
        children, shaped, spread = {}, {}, {}
        for a, b in members:
            if a not in shaped:
                shaped[a] = reshape(sp, SETTINGS[a])
                spread[a] = entropy(shaped[a])
            token = int(order[draw(shaped[a], U[b, depth])])
            tokens[a, b, depth], logp[a, b, depth] = token, np.log(p[token])
            bits[a, b, depth] = spread[a]
            children.setdefault(token, []).append((a, b))
        if depth + 1 < NEW_TOKENS:
            at = st.n
            for token, group in children.items():
                st.n = at
                visit(st.feed([token])[-1], depth + 1, group)
    visit(st.feed(prompt)[-1], 0, [(a, b) for a in range(len(SETTINGS))
                                   for b in range(len(SAMPLE_SEEDS))])
    return (tokens, logp, bits, passes[0], first["order"], first["p"], first["q"],
            first["kept"])


def agreement(tokens):
    """From tokens ``(..., seeds, length)``: the share of pairs of seeds whose texts are
    the same through each position, and each pair's first differing position (``length``
    if none)."""
    n = tokens.shape[-2]
    i, j = np.triu_indices(n, 1)
    same = np.cumprod(tokens[..., i, :] == tokens[..., j, :], axis=-1)
    return same.mean(-2), same.sum(-1)


def distinct(tokens):
    """Number of different continuations among the seeds, for tokens ``(seeds, length)``."""
    return len({tuple(row) for row in tokens.tolist()})


def _sampling_build():
    from . import study_decoding as D

    prompts = D.heldout_prompts(PROMPTS, PROMPT_TOKENS)
    parts = _parallel(sample_prompt, [(i,) for i in range(PROMPTS)])
    tokens = np.array([p[0] for p in parts])              # (prompt, setting, seed, token)
    logp = np.array([p[1] for p in parts])
    bits = np.array([p[2] for p in parts])
    tok = D._tok()
    again, again_logp = sample_plain(prompts[DRAWN_PROMPT], SAMPLE_SEEDS[3], SETTINGS[3])
    twice, _ = sample_plain(prompts[DRAWN_PROMPT], SAMPLE_SEEDS[3], SETTINGS[3])
    other, _ = sample_plain(prompts[DRAWN_PROMPT], SAMPLE_SEEDS[4], SETTINGS[3])
    drawn = tokens[DRAWN_PROMPT]
    _, _, _, _, order, p, q, kept = parts[DRAWN_PROMPT]
    return dict(
        prompts=prompts, tokens=tokens, logp=logp, bits=bits,
        passes=np.array([p[3] for p in parts]),
        prompt_text=np.array([tok.decode(list(p)) for p in prompts]),
        again=again, twice=twice, other=other, again_logp=again_logp,
        first_order=order, first_p=p, first_q=q,
        first_kept=kept,
        first_words=np.array([D.show(t) for t in order[:60]]),
        words=np.array([[[D.show(t) for t in row] for row in block] for block in drawn]),
        texts=np.array([[tok.decode(row.tolist()) for row in block] for block in drawn]),
        uniforms=np.array([uniforms(s) for s in SAMPLE_SEEDS]),
    )


def sampling_data():
    """Every sampled continuation of ``seeds-and-sampled-text`` (cached)."""
    from . import gpt2

    return _cache("sampling", _sampling_build,
                  (_sampling_build, sample_prompt, sample_plain, reshape, uniforms, draw),
                  prompts=PROMPTS, prompt_tokens=PROMPT_TOKENS, seeds=SAMPLE_SEEDS,
                  new=NEW_TOKENS, settings=SETTINGS, drawn=DRAWN_PROMPT,
                  gpt2_revision=gpt2.REVISION, dataset=realdata.TEXT_SHA256)


@functools.lru_cache(maxsize=1)
def sampling_summary():
    """Per setting and prompt: agreement by position, first differences, distinct
    continuations, GPT-2's mean log-probability, the entropy drawn from, and the share
    of continuations holding a token GPT-2 gave less than :data:`RARE`."""
    d = sampling_data()
    tokens = d["tokens"]
    agree, first = agreement(tokens)                       # (prompt, setting, position)
    kinds = np.array([[distinct(tokens[i, a]) for a in range(len(SETTINGS))]
                      for i in range(PROMPTS)])
    rare = (d["logp"] < np.log(RARE))
    return dict(
        d=d, agree=agree, first=first, distinct=kinds,
        logp=d["logp"].mean((2, 3)), bits=d["bits"].mean((2, 3)),
        rare_text=rare.any(-1).mean(-1), rare_token=rare.mean((2, 3)),
    )


def seeds_and_sampled_text():
    from . import gpt2

    r = sampling_summary()
    d = r["d"]
    temps = np.array([s[0] for s in SETTINGS[:N_TEMPS]])
    names = [setting_name(s) for s in SETTINGS]
    per = lambda v: [float(x) for x in np.asarray(v, float).mean(0)]            # noqa: E731
    return Study(
        "One prompt, many seeds",
        "GPT-2 continuing %d held-out prompts from %d seeds at %d settings: where the "
        "texts part, how many there are, and how likely GPT-2 finds them"
        % (PROMPTS, len(SAMPLE_SEEDS), len(SETTINGS)),
        (
            Panel("Share of pairs of seeds still writing the same text",
                  "ys[temperature, position]: of the %d pairs of seeds, the share whose "
                  "continuations are identical through that token; mean over %d prompts "
                  "(per_prompt has each)." % (r["first"].shape[-1], PROMPTS), "curve",
                  dict(x=np.arange(1, NEW_TOKENS + 1), ys=r["agree"][:, :N_TEMPS].mean(0),
                       labels=names[:N_TEMPS], per_prompt=r["agree"])),
            Panel("Distinct continuations against temperature",
                  "ys[0]: different %d-token continuations among %d seeds; ys[1]: the mean "
                  "position at which a pair of seeds first differs. Mean over prompts."
                  % (NEW_TOKENS, len(SAMPLE_SEEDS)), "curve",
                  dict(x=temps, ys=np.array([r["distinct"][:, :N_TEMPS].mean(0),
                                             r["first"][:, :N_TEMPS].mean((0, 2))]),
                       labels=["distinct continuations", "first difference"],
                       per_prompt=r["distinct"])),
            Panel("Mean log-probability GPT-2 gives the sampled text",
                  "values: natural log of the chance GPT-2 (unmodified, temperature 1) "
                  "gives each sampled token, mean over tokens, seeds and prompts, per "
                  "setting. bits: Shannon entropy of the distribution each token was "
                  "drawn from, bits per token.", "bars",
                  dict(values=r["logp"].mean(0), labels=names, per_prompt=r["logp"],
                       bits=r["bits"].mean(0))),
            Panel("Variety and unlikely tokens, by setting",
                  "xy: distinct continuations among %d seeds, and the share of continuations "
                  "holding at least one token GPT-2 gave under %.1f%%. One point per "
                  "setting, mean over prompts." % (len(SAMPLE_SEEDS), 100 * RARE), "points",
                  dict(xy=np.stack([r["distinct"].mean(0), r["rare_text"].mean(0)], 1),
                       labels=names,
                       per_prompt=r["rare_text"], token_share=r["rare_token"].mean(0))),
        ),
        dict(
            seed=SAMPLE_SEEDS[0], seeds=list(SAMPLE_SEEDS), gpt2=True, review_only=True,
            source="the pinned GPT-2 (revision %s) sampling continuations of Tiny "
                   "Shakespeare test-split passages" % gpt2.REVISION[:12],
            dataset="Tiny Shakespeare / char-rnn",
            dataset_provenance=realdata.text_metadata("test"),
            where_the_seed_enters="numpy.random.default_rng(seed).random(%d): one uniform "
                                  "number per new token; the token is the first, most "
                                  "likely first, whose running chance passes it. A seed "
                                  "gives the same numbers for every prompt and setting."
                                  % NEW_TOKENS,
            prompts=[str(t) for t in d["prompt_text"]], prompt_tokens=PROMPT_TOKENS,
            drawn_prompt=DRAWN_PROMPT, new_tokens=NEW_TOKENS,
            settings=[dict(temperature=t, top_k=k, top_p=p) for t, k, p in SETTINGS],
            setting_names=names, rare_threshold=RARE,
            gpt2_passes=int(d["passes"].sum()),
            passes_without_sharing=PROMPTS * len(SETTINGS) * len(SAMPLE_SEEDS) * NEW_TOKENS,
            agreement_after_one_token=per(r["agree"][:, :, 0]),
            agreement_at_the_end=per(r["agree"][:, :, -1]),
            first_difference=per(r["first"].mean(-1)),
            distinct=per(r["distinct"]),
            distinct_range=[[int(v) for v in r["distinct"].min(0)],
                            [int(v) for v in r["distinct"].max(0)]],
            log_probability=per(r["logp"]), sampled_entropy_bits=per(r["bits"]),
            texts_with_an_unlikely_token=per(r["rare_text"]),
            unlikely_token_share=per(r["rare_token"]),
            reproducible=dict(
                same_seed_same_text=bool(np.array_equal(d["again"], d["twice"])),
                plain_sampler_equals_shared_passes=bool(np.array_equal(
                    d["again"], d["tokens"][DRAWN_PROMPT, 3, 3])),
                other_seed_same_text=bool(np.array_equal(d["again"], d["other"]))),
            limitations=[
                "Continuations are %d tokens long: with %d seeds the count of distinct "
                "texts cannot exceed %d and saturates at high temperature."
                % (NEW_TOKENS, len(SAMPLE_SEEDS), len(SAMPLE_SEEDS)),
                "GPT-2's own log-probability and the %.1f%% threshold are proxies for "
                "junk; no human judged the texts." % (100 * RARE),
                "Seeds share their uniform numbers across settings, so settings are "
                "compared on the same draws (which narrows their differences' noise, and "
                "means the settings are not independent samples).",
                "GPT-2 small on Shakespeare prompts; top-k 40 and top-p 0.9 are the "
                "customary values, not tuned.",
            ],
            designs_tried=[
                "Only this design was run: settings, seeds, prompts, length and the "
                "unlikely-token threshold were fixed before any text was sampled. The "
                "prompt drawn was fixed in advance as the third of the six.",
            ],
        ),
    )


STUDIES = {
    "markov-chains": markov_chains,
    "seeds-and-training-outcomes": seeds_and_training_outcomes,
    "seeds-and-sampled-text": seeds_and_sampled_text,
}
