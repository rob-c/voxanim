"""word2vec on a small generated corpus, and sentence embeddings built from it.

The corpus (:func:`corpus`) is generated from a handful of sentence templates
over five word categories -- animals, people, foods, places and verbs -- so
that words of a category share their contexts ("the ___ eats", "to the ___")
and nothing else does.  It is a toy, said so on screen: a real corpus has
the same structure, buried in far more noise.

**Skip-gram with negative sampling** (Mikolov et al., 2013;
:func:`train`): every word has two vectors, an input row ``W_in[w]`` (the
embedding) and an output row ``W_out[w]``.  For a centre word ``c`` and a
word ``o`` within :data:`WINDOW` of it, and :data:`NEGATIVES` words ``n``
drawn from the unigram distribution to the 3/4 power,

    loss = −log σ(W_out[o] · W_in[c]) − Σₙ log σ(−W_out[n] · W_in[c])

pulls the true pair's dot product up and the random pairs' down.  Plain SGD
on minibatches of pairs, the learning rate falling linearly to zero.  The
gradient touches only the rows that appear in the batch (:func:`sgns_step`
returns one step's numbers for the animations).

**Sentence embeddings** (:func:`sentence_vectors`): a sentence is the mean of
its words' input vectors -- order-blind, so "the dog chases the cat" and
"the cat chases the dog" are the same vector (up to the rounding of summing
in another order) -- or the SIF mean (Arora et al., 2017): each word weighted
by ``a / (a + p(w))`` (frequent words count less) and the corpus's common
direction removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import cached_npz

CATEGORIES = {
    "animal": ("cat", "dog", "horse", "cow", "sheep", "goat"),
    "person": ("farmer", "baker", "child", "doctor", "teacher", "king"),
    "food": ("bread", "apple", "cheese", "soup", "rice", "milk"),
    "place": ("city", "village", "farm", "market", "school", "river"),
}
# (template, weight): ``{animal}`` etc. are filled uniformly from the category
TEMPLATES = (
    ("the {person} eats {food} in the {place}", 3),
    ("the {animal} runs to the {place}", 2),
    ("the {person} sees the {animal}", 2),
    ("the {animal} chases the {animal}", 2),
    ("a {person} buys {food} at the market", 2),
    ("the {person} walks to the {place}", 2),
    ("the {animal} drinks milk by the river", 1),
    ("a {person} cooks {food} for the {person}", 2),
)
# Who goes with what: a slot is filled, with probability AFFINITY_P, from the
# affinities of the words already in the sentence (when any belong to the
# slot's category), else uniformly.  Without it every word of a category
# would have exactly the same contexts, and the same vector.
AFFINITY = {
    "farmer": ("farm", "village", "milk", "cheese", "cow", "sheep"),
    "baker": ("city", "market", "bread"),
    "child": ("school", "apple", "milk", "cat", "dog"),
    "doctor": ("city", "soup", "rice"),
    "teacher": ("school", "city", "rice", "apple"),
    "king": ("city", "cheese", "bread", "horse"),
    "cat": ("city", "village", "milk"),
    "dog": ("city", "farm", "cat"),
    "horse": ("farm", "river"),
    "cow": ("farm", "village"),
    "sheep": ("farm", "village", "goat"),
    "goat": ("village", "river", "sheep"),
}
AFFINITY_P = 0.6
N_SENTENCES = 30000
CORPUS_SEED = 0
DIM = 16                     # embedding width
WINDOW = 4                   # context: up to 4 words either side
NEGATIVES = 5
BATCH = 128                  # pairs per SGD step
EPOCHS = 2
LR = 0.5                     # SGD learning rate at step 0, falling linearly to 0
N_SNAPS = 61                 # snapshots of W_in through training (step 0 included)
TRAIN_SEED = 1
SIF_A = 1e-3                 # SIF's a

# the sentences the sentence-embedding animation compares
SENTENCES = (
    "the dog chases the cat",
    "the cat chases the dog",
    "the farmer eats cheese in the village",
    "the baker eats bread in the city",
    "the horse runs to the farm",
    "a doctor buys apple at the market",
)


def vocabulary():
    """Every word of the templates and categories, sorted."""
    words = set()
    for ws in CATEGORIES.values():
        words.update(ws)
    for tpl, _ in TEMPLATES:
        words.update(w for w in tpl.split() if not w.startswith("{"))
    return sorted(words)


def corpus(n=N_SENTENCES, seed=CORPUS_SEED):
    """``n`` generated sentences, as lists of words."""
    rng = np.random.default_rng(seed)
    w = np.array([wt for _, wt in TEMPLATES], np.float64)
    picks = rng.choice(len(TEMPLATES), n, p=w / w.sum())
    out = []
    for t in picks:
        words = []
        for tok in TEMPLATES[t][0].split():
            if tok.startswith("{"):
                cat = CATEGORIES[tok[1:-1]]
                pref = sorted({a for w in words for a in AFFINITY.get(w, ()) if a in cat})
                if pref and rng.random() < AFFINITY_P:
                    words.append(pref[rng.integers(len(pref))])
                else:
                    words.append(cat[rng.integers(len(cat))])
            else:
                words.append(tok)
        out.append(words)
    return out


def category_of(word):
    """The category of ``word``, or None for the template words."""
    for k, ws in CATEGORIES.items():
        if word in ws:
            return k
    return None


def pairs(sentences, index, window=WINDOW):
    """Every (centre, context) pair of word ids within ``window``, in corpus order."""
    out = []
    for s in sentences:
        ids = [index[w] for w in s]
        for i, c in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if j != i:
                    out.append((c, ids[j]))
    return np.array(out, np.int64)


def unigram(sentences, index):
    """Word frequencies ``p(w)`` over the corpus."""
    n = np.zeros(len(index))
    for s in sentences:
        for w in s:
            n[index[w]] += 1
    return n / n.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sgns_grads(W_in, W_out, c, o, neg):
    """Loss and gradients of one minibatch: centres ``c`` (B,), contexts ``o``
    (B,), negatives ``neg`` (B, K).  Returns ``(loss, rows_in, g_in, rows_out,
    g_out)``: the gradient as rows and values (to be scatter-added), the loss
    the batch mean."""
    v = W_in[c]                                  # (B, D)
    u_pos = W_out[o]                             # (B, D)
    u_neg = W_out[neg]                           # (B, K, D)
    s_pos = np.einsum("bd,bd->b", u_pos, v)
    s_neg = np.einsum("bkd,bd->bk", u_neg, v)
    loss = -np.log(sigmoid(s_pos)) - np.log(sigmoid(-s_neg)).sum(1)
    B = len(c)
    d_pos = (sigmoid(s_pos) - 1.0) / B           # d loss / d s_pos
    d_neg = sigmoid(s_neg) / B                   # d loss / d s_neg
    g_v = d_pos[:, None] * u_pos + np.einsum("bk,bkd->bd", d_neg, u_neg)
    g_upos = d_pos[:, None] * v
    g_uneg = d_neg[:, :, None] * v[:, None, :]
    rows_out = np.concatenate([o, neg.ravel()])
    g_out = np.concatenate([g_upos, g_uneg.reshape(-1, v.shape[1])])
    return float(loss.mean()), c, g_v, rows_out, g_out


@dataclass
class Run:
    """A trained word2vec model and its history."""

    words: list
    W_in: np.ndarray              # (V, D) the embeddings
    W_out: np.ndarray             # (V, D) the output vectors
    snaps: np.ndarray             # (N_SNAPS, V, D) W_in through training
    snap_steps: np.ndarray        # (N_SNAPS,) the step of each snapshot
    loss: np.ndarray              # (steps,) each step's batch loss
    p: np.ndarray                 # (V,) word frequencies
    steps: int

    @property
    def index(self):
        return {w: i for i, w in enumerate(self.words)}


def _train():
    words = vocabulary()
    index = {w: i for i, w in enumerate(words)}
    sents = corpus()
    P = pairs(sents, index)
    p = unigram(sents, index)
    noise = p ** 0.75
    noise /= noise.sum()
    V = len(words)
    rng = np.random.default_rng(TRAIN_SEED)
    W_in = rng.uniform(-0.5, 0.5, (V, DIM)) / DIM
    W_out = np.zeros((V, DIM))
    steps_per_epoch = len(P) // BATCH
    steps = EPOCHS * steps_per_epoch
    snap_steps = np.unique(np.round(np.linspace(0, steps, N_SNAPS)).astype(np.int64))
    snaps, losses = [W_in.copy()], np.empty(steps)
    k = 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(P))
        for b in range(steps_per_epoch):
            idx = order[b * BATCH:(b + 1) * BATCH]
            c, o = P[idx, 0], P[idx, 1]
            neg = rng.choice(V, (len(idx), NEGATIVES), p=noise)
            lr = LR * (1.0 - k / steps)
            loss, ri, gi, ro, go = sgns_grads(W_in, W_out, c, o, neg)
            np.add.at(W_in, ri, -lr * gi)
            np.add.at(W_out, ro, -lr * go)
            losses[k] = loss
            k += 1
            if k in snap_steps:
                snaps.append(W_in.copy())
    return dict(W_in=W_in, W_out=W_out, snaps=np.array(snaps), snap_steps=snap_steps,
                loss=losses, p=p, words=np.array(words))


def train(cache_dir=None):
    """The trained model (seeded, cached)."""
    settings = dict(templates=TEMPLATES, categories=CATEGORIES, affinity=AFFINITY,
                    affinity_p=AFFINITY_P, n=N_SENTENCES,
                    corpus_seed=CORPUS_SEED, dim=DIM, window=WINDOW, negatives=NEGATIVES,
                    batch=BATCH, epochs=EPOCHS, lr=LR, snaps=N_SNAPS, seed=TRAIN_SEED,
                    v=1)
    d = cached_npz("w2v", settings, _train, cache_dir)
    return Run([str(w) for w in d["words"]], d["W_in"], d["W_out"], d["snaps"],
               d["snap_steps"], d["loss"], d["p"], int(d["snap_steps"][-1]))


# ------------------------------------------------------------------ measurements


def unit(W):
    """Rows scaled to length 1."""
    return W / np.maximum(np.linalg.norm(W, axis=-1, keepdims=True), 1e-12)


def neighbours(W, words, word, k=5):
    """The ``k`` words nearest ``word`` by cosine, as ``[(word, cos)]``."""
    U = unit(W)
    i = words.index(word)
    cos = U @ U[i]
    order = [j for j in np.argsort(-cos, kind="stable") if j != i][:k]
    return [(words[j], float(cos[j])) for j in order]


def category_cosines(W, words):
    """Mean cosine between distinct words of the same category, and between
    words of different categories (category words only)."""
    U = unit(W)
    cats = [category_of(w) for w in words]
    idx = [i for i, c in enumerate(cats) if c is not None]
    same, diff = [], []
    for a in idx:
        for b in idx:
            if a < b:
                (same if cats[a] == cats[b] else diff).append(float(U[a] @ U[b]))
    return float(np.mean(same)), float(np.mean(diff))


def projection(W, words):
    """A fixed 2-D view of the category words: the top two principal axes of
    their unit vectors.  Returns ``(mean, axes (2, D))``."""
    U = unit(W)[[i for i, w in enumerate(words) if category_of(w)]]
    mu = U.mean(0)
    _, _, vt = np.linalg.svd(U - mu, full_matrices=False)
    axes = vt[:2]
    for a in axes:                       # a fixed sign: the largest entry positive
        if a[np.argmax(np.abs(a))] < 0:
            a *= -1
    return mu, axes


def sgns_step(run, snap, centre, context, negatives, lr=None):
    """One SGNS update, on the vectors of snapshot ``snap``: the scores before
    and after, the gradients and the rows they touch."""
    index = run.index
    W_in = run.snaps[snap].copy()
    # the output vectors at that snapshot are not stored: the trained ones
    # stand in (the update rule is the same; the animation says so)
    W_out = run.W_out.copy()
    c, o = index[centre], index[context]
    neg = np.array([[index[w] for w in negatives]])
    lr = LR * 0.5 if lr is None else lr
    v0 = W_in[c].copy()
    s_pos0 = float(W_out[o] @ v0)
    s_neg0 = W_out[neg[0]] @ v0
    loss0, ri, gi, ro, go = sgns_grads(W_in, W_out, np.array([c]), np.array([o]), neg)
    np.add.at(W_in, ri, -lr * gi)
    np.add.at(W_out, ro, -lr * go)
    loss1 = sgns_grads(W_in, W_out, np.array([c]), np.array([o]), neg)[0]
    return dict(c=c, o=o, neg=neg[0], v0=v0, v1=W_in[c].copy(), g_in=gi[0], g_out=go,
                s_pos0=s_pos0, s_pos1=float(W_out[o] @ W_in[c]), s_neg0=s_neg0,
                s_neg1=W_out[neg[0]] @ W_in[c], loss0=loss0, loss1=loss1,
                rows_in=sorted({int(c)}), rows_out=sorted({int(r) for r in ro}), lr=lr)


def sentence_vectors(run, sentences=SENTENCES, sif=False):
    """Each sentence's embedding ``(N, D)``: the mean of its words' vectors,
    or (``sif``) the SIF-weighted mean with the corpus's first principal
    direction removed (estimated on the corpus sentences themselves)."""
    index, W, p = run.index, run.W_in, run.p

    def embed(words):
        ids = [index[w] for w in words]
        if not sif:
            return W[ids].mean(0)
        wt = SIF_A / (SIF_A + p[ids])
        return (wt[:, None] * W[ids]).sum(0) / len(ids)

    X = np.array([embed(s.split()) for s in sentences])
    if sif:
        ref = np.array([embed(s) for s in corpus(2000, seed=CORPUS_SEED + 1)])
        _, _, vt = np.linalg.svd(ref, full_matrices=False)
        u = vt[0]
        X = X - np.outer(X @ u, u)
    return X


def cosine_matrix(X):
    """``(N, N)`` cosine similarities of the rows of ``X``."""
    U = unit(X)
    return U @ U.T
