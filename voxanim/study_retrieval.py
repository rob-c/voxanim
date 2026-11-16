"""Retrieval and embeddings: the experiments redesigned in November 2026.

Three of this section's first experiments (:mod:`voxanim.next_systems`,
:mod:`voxanim.next_systems_real`) could not show their concept:

* ``infonce-temperature``: a linear map trained 161 updates on views that differed
  only by faint noise, at three temperatures. Partners were already cosine 0.99 apart
  from everything else, so there was nothing for the temperature to change.
* ``hard-negatives``: the "hard" and "false" negatives were weightings built from the
  digit labels, which no contrastive training has.
* ``retrieval-augmented-generation``: one question, five passages, the best of them the
  line the question was cut from, found by word overlap alone.

Their replacements are here. The two contrastive studies share one encoder (a
two-layer ReLU network on 14 x 14 MNIST, its output on the unit sphere) and one kind of
view (the digit shifted by up to three pixels each way, then noise), so that telling
partners from the rest of the batch has to be learned. Every comparison between
trainings is run from three seeds and all three are kept.

A cache is keyed by the source of the functions that produced it, not by this file:
editing one study does not re-run the others.
"""

from __future__ import annotations

import hashlib
import inspect
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from . import realdata as R
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

SEEDS = (0, 1, 2)
WORD = re.compile(r"[a-z]+")

# What was tried on the way to each design, and what each cannot show: kept with the study
# (``metadata``). Outside the functions on purpose: rewording a note re-runs nothing.
NOTES = {
    "infonce-temperature": dict(
        designs_tried=[
            "the first experiment: a linear 49 -> 12 map, 161 updates, views differing by "
            "noise of sd 0.05 only, temperatures 0.05 / 0.2 / 1, one seed (accuracy 74.0 / "
            "72.9 / 75.5%)",
            "this design (encoder, views, the eight temperatures, the three measures), first "
            "run to time it: all eight temperatures and three seeds at 600 updates and batch "
            "256, then temperatures 0.01 / 0.1 / 1 with one seed at 1,500 updates (batch 128 "
            "and 256) and 4,000 updates (batch 128). Every one had the same ordering, best "
            "near 0.1. Recorded at 1,200 updates and batch 128, chosen for the build budget",
        ],
        limitations=[
            "Small encoder and 4,096 training digits: the best embedding does not beat "
            "nearest neighbour on the pixels themselves (pixel_accuracy), and is only a few "
            "points above the encoder before training (untrained), which the high "
            "temperatures fall below. The comparison is between temperatures.",
            "At the lowest temperatures the embeddings end crowded, not spread as larger "
            "studies report: after 1,200 updates only an anchor's nearest look-alikes have "
            "been pushed, and the cloud the encoder started with is otherwise left alone.",
        ]),
    "hard-negatives": dict(
        designs_tried=[
            "the first experiment: a linear map, 161 updates, one seed; 'hard' and 'false' "
            "negatives were weights set from the digit labels (x5 on other-digit lookalikes, "
            "x8 on same-digit images)",
            "this design, the only one run: its settings (8 negatives, skip 16, temperature "
            "0.2, 1,200 updates) were fixed before any comparison was looked at. Run once in "
            "a scratch script, where the lots were drawn from the batch generator (means "
            "84.2 / 87.5 / 85.8 / 93.1%), and once here",
        ],
        limitations=[
            "One dataset with ten classes: about a tenth of any batch is the anchor's own "
            "digit, far more false negatives than a retrieval corpus holds.",
            "Negatives drawn by lot leave the encoder no better at this test than before "
            "training (the traces start at the untrained accuracy).",
        ]),
    "retrieval-augmented-generation": dict(
        designs_tried=[
            "the first experiment: one question, five passages, TF-IDF only, the best "
            "passage the question's own line",
            "this design, the only one run (once in a scratch script, once here, the same "
            "numbers): store, questions, cut, retrievers and the random-line control were "
            "fixed before GPT-2 was asked anything",
        ],
        limitations=[
            "The questions are continuations of verse, not facts: with its source removed "
            "the store holds related lines, not the answer, and retrieval then does no "
            "better than a random line.",
            "One random line per question as the control (seed 17).",
        ]),
}


# ---------------------------------------------------------------------- bookkeeping
def _source(*functions):
    h = hashlib.sha256()
    for f in functions:
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()[:16]


def cached(name, build, functions, **settings):
    settings = dict(source=_source(*functions), optimiser=R.source_hash("optim.py"), **settings)
    return {k: np.asarray(v) for k, v in
            cached_npz("study-retrieval-" + name, settings, build).items()}, settings


def finish(name, title, subtitle, panels, d, settings, dataset, **meta):
    """The study, with what the run saved (``P_*``: weights; the rest: traces) on file."""
    params = {k[2:]: v for k, v in d.items() if k.startswith("P_")}
    history = {k: v for k, v in d.items() if not k.startswith("P_")}
    evidence = R.save_evidence("study-" + name, params, history, dataset, settings,
                               ("study_retrieval.py", "optim.py"))
    return Study(title, subtitle, tuple(panels),
                 dict(seed=list(SEEDS), training_provenance=evidence, **meta))


def spread(values):
    """``[mean, lowest, highest]`` over the last axis (the seeds)."""
    v = np.asarray(values, float)
    return np.array([v.mean(-1), v.min(-1), v.max(-1)])


# ---------------------------------------------------------------------- the encoder
SIDE, HIDDEN, WIDTH = 14, 256, 32
SHIFT, NOISE = 3, 0.1
TRAIN, TEST = 4096, 1024


@lru_cache(maxsize=1)
def digits():
    """4,096 training and 1,024 official-test MNIST digits at 28 x 28, with provenance."""
    x, y, train = R.mnist_split(TRAIN, "train", side=28, seed=17)
    xt, yt, test = R.mnist_split(TEST, "test", side=28, seed=73)
    return dict(x=x.astype(np.float32), y=y, xt=xt.astype(np.float32), yt=yt,
                meta=dict(train=train, test=test))


def pool(images):
    """28 x 28 images as the encoder reads them: 2 x 2 means, flattened to 196."""
    return np.asarray(images, np.float32).reshape(-1, SIDE, 2, SIDE, 2).mean((2, 4)).reshape(
        -1, SIDE * SIDE)


def views(images, rng):
    """One random view of each 28 x 28 image: shifted by up to :data:`SHIFT` pixels each
    way (empty where it came from), pooled, plus noise of sd :data:`NOISE`."""
    n = len(images)
    pad = np.zeros((n, 28 + 2 * SHIFT, 28 + 2 * SHIFT), np.float32)
    pad[:, SHIFT:-SHIFT, SHIFT:-SHIFT] = images.reshape(n, 28, 28)
    dy = rng.integers(0, 2 * SHIFT + 1, n)
    dx = rng.integers(0, 2 * SHIFT + 1, n)
    rows = (dy[:, None] + np.arange(28))[:, :, None]
    cols = (dx[:, None] + np.arange(28))[:, None, :]
    moved = pad[np.arange(n)[:, None, None], rows, cols]
    return pool(moved) + rng.normal(0, NOISE, (n, SIDE * SIDE)).astype(np.float32)


def init(rng):
    return dict(
        W1=rng.normal(0, np.sqrt(2 / SIDE ** 2), (SIDE ** 2, HIDDEN)).astype(np.float32),
        b1=np.zeros(HIDDEN, np.float32),
        W2=rng.normal(0, np.sqrt(1 / HIDDEN), (HIDDEN, WIDTH)).astype(np.float32),
        b2=np.zeros(WIDTH, np.float32))


def forward(P, x):
    """``(hidden, norm, unit embedding)`` of pooled images ``x``."""
    h = np.maximum(x @ P["W1"] + P["b1"], 0)
    z = h @ P["W2"] + P["b2"]
    n = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)
    return h, n, z / n


def embed(P, x):
    return forward(P, x)[2]


def softmax(s):
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def infonce(u, v, temperature, mask=None):
    """InfoNCE on unit embeddings: row ``i`` of ``u`` must pick ``v[i]`` out of ``v``.

    Without ``mask`` the loss is the symmetric one (rows and columns). With a boolean
    ``mask`` (its diagonal set) row ``i`` competes only against the columns it marks.
    Returns ``(loss, d loss / d (u @ v.T))``."""
    s = u @ v.T / temperature
    n = len(u)
    eye = np.eye(n, dtype=np.float32)
    if mask is None:
        pr, pc = softmax(s), softmax(s.T)
        loss = -0.5 * (np.log(np.diag(pr) + 1e-30).mean() + np.log(np.diag(pc) + 1e-30).mean())
        return float(loss), (pr + pc.T - 2 * eye) / (2 * n * temperature)
    pr = softmax(np.where(mask, s, -np.inf))
    return float(-np.log(np.diag(pr) + 1e-30).mean()), (pr - eye) / (n * temperature)


def backward(P, sides, g):
    """Gradients of the encoder from ``g = d loss / d (u @ v.T)``; ``sides``: the two
    ``(x, forward(P, x))`` it was computed from."""
    (xa, (ha, na, u)), (xb, (hb, nb, v)) = sides
    G = {k: np.zeros_like(w) for k, w in P.items()}
    for x, h, n, w, d in ((xa, ha, na, u, g @ v), (xb, hb, nb, v, g.T @ u)):
        dz = (d - w * (w * d).sum(1, keepdims=True)) / n
        dh = (dz @ P["W2"].T) * (h > 0)
        G["W2"] += h.T @ dz
        G["b2"] += dz.sum(0)
        G["W1"] += x.T @ dh
        G["b1"] += dh.sum(0)
    return G


def eval_views():
    """Two fixed views of every test digit, the same for every run."""
    xt = digits()["xt"]
    return views(xt, np.random.default_rng(1000)), views(xt, np.random.default_rng(1001))


def measures(P, held):
    """What an encoder has learned, on the test digits: ``(accuracy, alignment,
    uniformity)``. Accuracy: test digits whose nearest training digit (cosine) is the
    same digit. Alignment: mean squared distance between the two views of a test digit
    (lower: partners closer). Uniformity: log of the mean of exp(-2 squared distance)
    over pairs of different test digits (lower: more evenly spread). The last two are
    Wang and Isola's (2020)."""
    d = digits()
    zt, zx = embed(P, pool(d["xt"])), embed(P, pool(d["x"]))
    accuracy = float(np.mean(d["y"][(zt @ zx.T).argmax(1)] == d["yt"]))
    za, zb = embed(P, held[0]), embed(P, held[1])
    alignment = float(((za - zb) ** 2).sum(1).mean())
    d2 = 2 - 2 * (zt @ zt.T).astype(np.float64)
    off = ~np.eye(len(zt), dtype=bool)
    return accuracy, alignment, float(np.log(np.exp(-2 * d2[off]).mean()))


def hardest_share(P, held, temperature, n=128):
    """Of all the push a row's negatives receive (their softmax shares), the part that
    lands on the single most similar one; mean over a batch of ``n`` test pairs."""
    u, v = embed(P, held[0][:n]), embed(P, held[1][:n])
    p = softmax(u @ v.T / temperature).astype(np.float64)
    np.fill_diagonal(p, 0.0)
    return float((p.max(1) / np.maximum(p.sum(1), 1e-300)).mean())


def pixel_accuracy():
    """The same nearest-neighbour test on the pooled pixels themselves."""
    d = digits()
    a, b = pool(d["xt"]), pool(d["x"])
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float(np.mean(d["y"][(a @ b.T).argmax(1)] == d["yt"]))


# ====================================================================== 83
TEMPERATURES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
UPDATES, BATCH, RATE = 1200, 128, 1e-3
EVERY = 100                       # held-out measures are also traced this often


def train_at(temperature, seed, held):
    """One encoder trained at ``temperature``: its weights and traces."""
    d = digits()
    rng = np.random.default_rng(seed)
    P = init(rng)
    opt = Adam(P, RATE)
    loss, trace = [], [measures(P, held)]
    for step in range(UPDATES):
        ix = rng.integers(0, TRAIN, BATCH)
        a, b = views(d["x"][ix], rng), views(d["x"][ix], rng)
        fa, fb = forward(P, a), forward(P, b)
        value, g = infonce(fa[2], fb[2], temperature)
        opt.step(P, backward(P, ((a, fa), (b, fb)), g))
        loss.append(value)
        if (step + 1) % EVERY == 0:
            trace.append(measures(P, held))
    return P, np.array(loss), np.array(trace)


def temperature():
    """Eight temperatures from 0.01 to 2, three seeds each."""
    def build():
        held = eval_views()
        out = dict(loss=[], trace=[], hardest=[])
        for i, t in enumerate(TEMPERATURES):
            runs = [train_at(t, seed, held) for seed in SEEDS]
            out["loss"].append([r[1] for r in runs])
            out["trace"].append([r[2] for r in runs])
            out["hardest"].append([hardest_share(r[0], held, t) for r in runs])
            for k, w in runs[0][0].items():
                out["P_t%d_%s" % (i, k)] = w
        out["untrained"] = np.array([measures(init(np.random.default_rng(s)), held)
                                     for s in SEEDS])
        out["pixels"] = pixel_accuracy()
        return out

    design = dict(temperatures=list(TEMPERATURES), seeds=list(SEEDS), updates=UPDATES,
                  batch=BATCH, rate=RATE, hidden=HIDDEN, width=WIDTH, shift=SHIFT, noise=NOISE,
                  train=TRAIN, test=TEST)
    d, settings = cached("infonce-temperature", build,
                         (views, init, forward, infonce, backward, measures, hardest_share,
                          train_at, temperature), **design)
    trace = d["trace"]                                   # temperature, seed, time, measure
    acc, align, unif = (trace[:, :, -1, k] for k in range(3))
    temps = np.array(TEMPERATURES)
    names = ["mean of %d seeds" % len(SEEDS), "lowest seed", "highest seed"]
    best = int(np.argmax(acc.mean(1)))
    return finish(
        "infonce-temperature",
        "Temperature in contrastive learning",
        "A ReLU encoder trained on shifted, noisy MNIST views at eight temperatures, three "
        "seeds each",
        [
            Panel("The contrastive loss while training",
                  "One curve per temperature (first seed). Losses at different temperatures "
                  "are not on one scale.", "curve",
                  dict(x=np.arange(1, UPDATES + 1), ys=d["loss"][:, 0],
                       labels=["temperature %g" % t for t in TEMPERATURES])),
            Panel("Alignment against temperature",
                  "Mean squared distance between two views of a held-out digit; lower is "
                  "closer.", "curve", dict(x=temps, ys=spread(align), labels=names)),
            Panel("Uniformity against temperature",
                  "Log mean exp(-2 squared distance) over pairs of held-out digits; lower is "
                  "more evenly spread.", "curve", dict(x=temps, ys=spread(unif), labels=names)),
            Panel("Held-out nearest-neighbour accuracy against temperature",
                  "Test digits whose nearest training digit by embedding is the same digit.",
                  "curve", dict(x=temps, ys=spread(acc), labels=names),
                  ("best mean: temperature %g" % TEMPERATURES[best],)),
        ],
        d, settings, digits()["meta"],
        evidence_kind="actual NumPy training on MNIST, three seeds per setting",
        design=design,
        accuracy=acc.tolist(), alignment=align.tolist(), uniformity=unif.tolist(),
        hardest_negative_share=d["hardest"].tolist(),
        untrained=dict(zip(("accuracy", "alignment", "uniformity"),
                           d["untrained"].mean(0).tolist())),
        pixel_accuracy=float(d["pixels"]),
        **NOTES["infonce-temperature"],
    )


# ====================================================================== 84
MODES = ("random", "hardest", "skip", "clean")
NEGATIVES, SKIP, HARD_TEMPERATURE = 8, 16, 0.2
CHECKPOINTS = (0, 100, UPDATES)


def choose(u, v, mode, rng, labels):
    """Each anchor's :data:`NEGATIVES` negatives from the rest of the batch, as a mask
    over ``u @ v.T`` (the diagonal, its partner, always set), and how many of those chosen
    are the anchor's own digit.

    ``random``: drawn by lot. ``hardest``: the most similar under the encoder as it is
    now; no label is used. ``skip``: the next most similar after passing over the first
    :data:`SKIP`; no label either. ``clean``: the most similar of those the labels say are
    another digit - a control that needs what contrastive training does not have."""
    n = len(u)
    s = (u @ v.T).astype(np.float64)
    np.fill_diagonal(s, -np.inf)
    if mode == "random":
        s = rng.random((n, n))
        np.fill_diagonal(s, -np.inf)
    elif mode == "clean":
        s[labels[:, None] == labels[None]] = -np.inf
    start = SKIP if mode == "skip" else 0
    pick = np.argsort(-s, axis=1, kind="stable")[:, start:start + NEGATIVES]
    mask = np.eye(n, dtype=bool)
    mask[np.arange(n)[:, None], pick] = True
    return mask, float((labels[pick] == labels[:, None]).mean())


def train_with(mode, seed, held):
    """One encoder trained with ``mode``'s negatives. Every mode of a seed starts from
    the same weights and sees the same images and noise in the same order."""
    d = digits()
    rng = np.random.default_rng(seed)
    lots = np.random.default_rng(10_000 + seed)
    P = init(rng)
    opt = Adam(P, RATE)
    false, trace, kept = [], [measures(P, held)], {0: dict(P)}
    for step in range(UPDATES):
        ix = rng.integers(0, TRAIN, BATCH)
        a, b = views(d["x"][ix], rng), views(d["x"][ix], rng)
        fa, fb = forward(P, a), forward(P, b)
        mask, same = choose(fa[2], fb[2], mode, lots, d["y"][ix])
        _, g = infonce(fa[2], fb[2], HARD_TEMPERATURE, mask)
        opt.step(P, backward(P, ((a, fa), (b, fb)), g))
        false.append(same)
        if (step + 1) % EVERY == 0:
            trace.append(measures(P, held))
        if step + 1 in CHECKPOINTS:
            kept[step + 1] = dict(P)
    return kept, np.array(false), np.array(trace)


def first_batch(seed=SEEDS[0]):
    """The first training batch of ``seed`` (every mode's): indices and both views."""
    d = digits()
    rng = np.random.default_rng(seed)
    init(rng)
    ix = rng.integers(0, TRAIN, BATCH)
    return ix, views(d["x"][ix], rng), views(d["x"][ix], rng)


def hard_negatives():
    """Random negatives against mined ones, four ways of choosing, three seeds each."""
    def build():
        held = eval_views()
        out = dict(false=[], trace=[])
        for mode in MODES:
            runs = [train_with(mode, seed, held) for seed in SEEDS]
            out["false"].append([r[1] for r in runs])
            out["trace"].append([r[2] for r in runs])
            if mode == "hardest":
                for at, P in runs[0][0].items():
                    for k, w in P.items():
                        out["P_hardest%d_%s" % (at, k)] = w
        return out

    design = dict(modes=list(MODES), negatives=NEGATIVES, skip=SKIP,
                  temperature=HARD_TEMPERATURE, seeds=list(SEEDS), updates=UPDATES, batch=BATCH,
                  rate=RATE, hidden=HIDDEN, width=WIDTH, shift=SHIFT, noise=NOISE, train=TRAIN,
                  test=TEST, checkpoints=list(CHECKPOINTS))
    d, settings = cached("hard-negatives", build,
                         (views, init, forward, infonce, backward, measures, choose, train_with,
                          hard_negatives), **design)
    false = d["false"]                                   # mode, seed, update
    trace = d["trace"]                                   # mode, seed, time, measure
    acc = trace[:, :, -1, 0]
    window = 100
    smooth = false.reshape(len(MODES), len(SEEDS), -1, window).mean(-1)
    names = ["random", "hardest by similarity", "hardest after skipping %d" % SKIP,
             "hardest of other digits (labels)"]
    return finish(
        "hard-negatives",
        "Hard negatives",
        "Eight negatives per anchor, drawn by lot or mined by the encoder's own similarity; "
        "MNIST views, three seeds each",
        [
            Panel("Mined negatives that are the anchor's own digit",
                  "Share of the chosen negatives with the anchor's label, mean of each %d "
                  "updates and of the seeds. Labels are read only to count." % window, "curve",
                  dict(x=np.arange(window, UPDATES + 1, window), ys=smooth.mean(1),
                       labels=names)),
            Panel("Held-out accuracy while training",
                  "Test digits whose nearest training digit by embedding is the same digit; "
                  "mean of the seeds.", "curve",
                  dict(x=np.arange(0, UPDATES + 1, EVERY), ys=trace[:, :, :, 0].mean(1),
                       labels=names)),
            Panel("Held-out accuracy, every seed",
                  "Rows: the ways of choosing negatives; columns: seeds.", "matrix",
                  dict(values=acc), tuple(names)),
            Panel("Held-out accuracy, mean of the seeds",
                  "The last is a control: it removes false negatives with the labels.", "bars",
                  dict(values=acc.mean(1), labels=names)),
        ],
        d, settings, digits()["meta"],
        evidence_kind="actual NumPy training on MNIST, three seeds per setting",
        design=design,
        accuracy=dict(zip(MODES, acc.tolist())),
        false_negative_share=dict(
            first_100_updates=dict(zip(MODES, false[:, :, :100].mean(2).tolist())),
            last_100_updates=dict(zip(MODES, false[:, :, -100:].mean(2).tolist()))),
        alignment=dict(zip(MODES, trace[:, :, -1, 1].tolist())),
        uniformity=dict(zip(MODES, trace[:, :, -1, 2].tolist())),
        **NOTES["hard-negatives"],
    )


# ====================================================================== 81
STORE, EVERY_NTH, CUT = 300, 5, 5
CONDITIONS = ("alone", "random", "words_in", "embedding_in", "words_out", "embedding_out")


def store_lines():
    """The lines on file: the first :data:`STORE` distinct lines of the held-out part of
    Tiny Shakespeare that have eight words or more (and 41 to 129 characters)."""
    lines = []
    for line in R.text("test").splitlines():
        line = line.strip()
        if 40 < len(line) < 130 and len(line.split()) >= 8 and line not in lines:
            lines.append(line)
    return lines[:STORE]


@lru_cache(maxsize=1)
def tfidf_table():
    """The 2,000 commonest words of the training text and their inverse document
    frequencies, one document per line (the first experiment's retriever)."""
    lines = [WORD.findall(s.lower()) for s in R.text("train").splitlines() if s.strip()]
    counts = Counter(w for line in lines for w in set(line))
    words = [w for w, _ in counts.most_common(2000)]
    idf = np.array([np.log((len(lines) + 1) / (counts[w] + 1)) + 1 for w in words])
    return words, idf


def tfidf_vector(line):
    """``line`` as a unit vector of its words' counts times their rarity."""
    words, idf = tfidf_table()
    index = {w: i for i, w in enumerate(words)}
    v = np.zeros(len(words))
    for word in WORD.findall(line.lower()):
        if word in index:
            v[index[word]] += 1
    v *= idf
    return v / max(np.linalg.norm(v), 1e-12)


def gpt2_reads(text):
    """GPT-2 on ``text``: ``(embedding, next-token probabilities)``; the embedding is
    its final residual stream averaged over the tokens."""
    from . import gpt2, gpt2_hooks

    out = gpt2_hooks.forward(gpt2.Tokenizer().encode(text), record=True)
    return out["resid"][-1].mean(0).astype(np.float64), gpt2.softmax(out["logits"][-1])


def retrieval():
    """Sixty cut-off lines, a store of three hundred, two retrievers, with and without
    the line each question was cut from."""
    from . import gpt2

    store = store_lines()
    asked = list(range(0, STORE, EVERY_NTH))
    questions = [" ".join(store[i].split()[:CUT]) for i in asked]

    def build():
        tok = gpt2.Tokenizer()
        words = np.array([tfidf_vector(line) for line in store])
        vectors = np.array([gpt2_reads(line)[0] for line in store])
        centre = vectors.mean(0)
        vectors = vectors - centre
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        lots = np.random.default_rng(17)
        seen = {}

        def reads(prompt):
            if prompt not in seen:
                seen[prompt] = gpt2_reads(prompt)[1]
            return seen[prompt]

        out = dict(answer=[], prob=[], pick=[], word_scores=[], embedding_scores=[])
        example = {}
        for n, (i, q) in enumerate(zip(asked, questions)):
            ids = tok.encode(q)
            full = tok.encode(store[i])
            assert full[:len(ids)] == ids
            answer = int(full[len(ids)])
            vector, alone = gpt2_reads(q)
            vector = vector - centre
            scores = dict(words=words @ tfidf_vector(q),
                          embedding=vectors @ (vector / np.linalg.norm(vector)))
            picks = dict(random=int(lots.choice([k for k in range(STORE) if k != i])))
            for name, s in scores.items():
                order = np.argsort(-s, kind="stable")
                picks[name + "_in"] = int(order[0])
                picks[name + "_out"] = int(order[order != i][0])
            dists = {c: reads(store[picks[c]] + "\n" + q) for c in CONDITIONS[1:]}
            out["answer"].append(answer)
            out["prob"].append([alone[answer]] + [dists[c][answer] for c in CONDITIONS[1:]])
            out["pick"].append([-1] + [picks[c] for c in CONDITIONS[1:]])
            out["word_scores"].append(scores["words"])
            out["embedding_scores"].append(scores["embedding"])
            if n == 0:
                shown = np.unique(np.r_[answer, np.argsort(alone)[-7:]])
                every = np.array([alone] + [dists[c] for c in CONDITIONS[1:]])
                example = dict(candidates=shown, tops=every[:, shown])
        return dict(out, P_idf=tfidf_table()[1], P_store_embeddings=vectors, P_centre=centre,
                    **{"example_" + k: v for k, v in example.items()})

    design = dict(store=store, asked=asked, cut_after_words=CUT, conditions=list(CONDITIONS),
                  dataset=R.TEXT_SHA256, random_seed=17,
                  embedding="GPT-2 final residual stream, mean over tokens, minus the store "
                            "mean; cosine",
                  words="TF-IDF, cosine; frequencies fitted on the training text")
    d, settings = cached("retrieval", build, (store_lines, tfidf_table, tfidf_vector,
                                               gpt2_reads, retrieval), **design)
    prob = d["prob"].astype(float)                       # question, condition
    alone, rest = prob[:, 0], prob[:, 1:]
    helped, hurt = (rest > alone[:, None]).mean(0), (rest < alone[:, None]).mean(0)
    found = (d["pick"][:, 2:4] == np.array(asked)[:, None]).mean(0)
    names = ["the question alone", "a random line", "by words, source on file",
             "by embedding, source on file", "by words, source removed",
             "by embedding, source removed"]
    study = finish(
        "retrieval",
        "Retrieval-augmented generation",
        "%d cut-off lines of Tiny Shakespeare, %d lines on file, two retrievers; pretrained "
        "GPT-2 probabilities" % (len(asked), STORE),
        [
            Panel("Chance of the true next word, mean over the questions",
                  "What GPT-2 gives the token that really followed, with one line put ahead "
                  "of the question.", "bars", dict(values=prob.mean(0), labels=names)),
            Panel("Questions helped by the line",
                  "Share whose true next token became more likely than with the question "
                  "alone.", "bars", dict(values=helped, labels=names[1:])),
            Panel("Every question",
                  "Rows: questions; columns: the six prompts.", "matrix", dict(values=prob),
                  tuple(names)),
            Panel("The source line retrieved first",
                  "How often each retriever ranks the line a question was cut from above the "
                  "other %d." % (STORE - 1), "bars",
                  dict(values=found, labels=["by words (TF-IDF)", "by GPT-2 embedding"])),
        ],
        d, settings, dict(train=R.text_metadata(), test=R.text_metadata("test")),
        evidence_kind="real-text fitted statistics and pretrained evaluation; nothing trained",
        design={k: v for k, v in design.items() if k != "store"},
        questions=questions,
        mean_probability=dict(zip(CONDITIONS, prob.mean(0).tolist())),
        helped=dict(zip(CONDITIONS[1:], helped.tolist())),
        hurt=dict(zip(CONDITIONS[1:], hurt.tolist())),
        median_ratio=dict(zip(CONDITIONS[1:], np.median(rest / alone[:, None], 0).tolist())),
        mean_log_gain=dict(zip(CONDITIONS[1:],
                               (np.log(rest) - np.log(alone[:, None])).mean(0).tolist())),
        source_found_first=dict(words=float(found[0]), embedding=float(found[1])),
        **NOTES["retrieval-augmented-generation"],
    )
    return study


STUDIES = {
    "retrieval-augmented-generation": retrieval,
    "infonce-temperature": temperature,
    "hard-negatives": hard_negatives,
}
