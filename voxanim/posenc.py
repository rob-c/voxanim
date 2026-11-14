"""Position in a transformer: the encodings, and a tiny model trained with each.

**The encodings** (pure functions, used by the animations directly):

* :func:`sinusoidal` -- Vaswani et al. (2017): ``PE[p, 2i] = sin(p ω_i)``,
  ``PE[p, 2i+1] = cos(p ω_i)``, ``ω_i = 10000^(−2i/d)``, added to the token
  embedding.  Moving ``k`` positions turns every (sin, cos) pair by ``k ω_i``,
  so ``PE[p + k]`` is a fixed rotation of ``PE[p]`` (:func:`shift_matrix`).
* learned absolute positions -- a trained table, one row per index (GPT-2).
* :func:`rope_angles` / :func:`rope` -- rotary embeddings (Su et al., 2021):
  queries and keys turned by ``p θ_i``; the score depends on the offset only.
  ``scale`` < 1 is position interpolation (Chen et al., 2023: positions
  squeezed into the trained range) and a larger ``base`` the "NTK-aware"
  scaling (bloc97, 2023: ``base · s^(d/(d−2))``, slowing the slow pairs most).
* :func:`t5_bucket` -- T5's relative position buckets (Raffel et al., 2020):
  each head adds a learned bias per bucket of the offset, exact for small
  offsets and logarithmic beyond, every offset past ``max_distance`` sharing
  the last bucket.
* :func:`alibi_slopes` -- ALiBi (Press et al., 2022): head ``h`` subtracts
  ``m_h · (i − j)`` from its scores, slopes ``2^(−8h/H)``; nothing learned.

**The experiment** (:func:`train`, :func:`evaluate`): a one-layer, 4-head
causal transformer (width 32, an MLP of 128, untied output) learns an
order-2 Markov language over 8 symbols (:func:`markov_table`: the next
symbol depends on the last two).  Predicting well needs the previous token,
which needs to know where it is.  Each model trains on sequences of
:data:`TRAIN_LEN` tokens and is tested on sequences four times longer;
:func:`entropies` gives the best possible loss and what a model that sees
only the current token can reach.  Everything in plain numpy with a
hand-written backward pass (checked against finite differences in the
tests), seeded and cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .cache import cached_npz
from .optim import Adam

# ------------------------------------------------------------------ the encodings

BASE = 10000.0


def frequencies(d, base=BASE):
    """``ω_i = base^(−2i/d)`` for the ``d / 2`` pairs."""
    return base ** (-np.arange(d // 2) * 2.0 / d)


def sinusoidal(n, d, base=BASE):
    """The ``(n, d)`` sinusoidal table: sin in even columns, cos in odd."""
    ang = np.arange(n)[:, None] * frequencies(d, base)[None, :]
    pe = np.empty((n, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def shift_matrix(k, d, base=BASE):
    """The ``(d, d)`` block rotation ``M_k`` with ``PE[p + k] = PE[p] @ M_k``
    (each (sin, cos) pair turned by ``k ω_i``), for every ``p``."""
    M = np.zeros((d, d))
    for i, w in enumerate(frequencies(d, base)):
        c, s = math.cos(k * w), math.sin(k * w)
        # [sin(a+b), cos(a+b)] = [sin a, cos a] @ [[c, -s], [s, c]]
        M[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[c, -s], [s, c]]
    return M


def rope_angles(n, d, base=BASE, scale=1.0):
    """``(n, d/2)`` angles ``(p · scale) θ_i``."""
    return (np.arange(n) * scale)[:, None] * frequencies(d, base)[None, :]


def rope(x, ang):
    """Turn every pair of ``x`` ``(..., n, d)`` by ``ang`` ``(n, d/2)``."""
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def ntk_base(scale, d, base=BASE):
    """The NTK-aware base for a context ``scale`` times longer."""
    return base * scale ** (d / (d - 2))


def t5_bucket(rel, num_buckets=16, max_distance=32):
    """T5's causal bucket of the offset ``rel = i − j ≥ 0`` (array or int):
    offsets below ``num_buckets / 2`` get a bucket each, larger ones share
    logarithmically wider buckets up to ``max_distance``, beyond which all
    share the last."""
    rel = np.asarray(rel)
    exact = num_buckets // 2
    big = exact + (np.log(np.maximum(rel, 1) / exact) / math.log(max_distance / exact)
                   * (num_buckets - exact)).astype(np.int64)
    big = np.minimum(big, num_buckets - 1)
    return np.where(rel < exact, rel, big)


def alibi_slopes(heads):
    """ALiBi's slopes ``2^(−8h/H)``, ``h = 1 .. H``."""
    return 2.0 ** (-8.0 * np.arange(1, heads + 1) / heads)


# ------------------------------------------------------------------ the language

VOCAB = 8
ALPHA = 0.25                    # Dirichlet concentration of each next-symbol distribution
TABLE_SEED = 3


def markov_table(seed=TABLE_SEED):
    """``(V, V, V)``: ``T[a, b]`` is the distribution of the symbol after ``a b``."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(VOCAB, ALPHA), (VOCAB, VOCAB))


def sample(n, length, rng, table=None):
    """``(n, length)`` sequences of the language (the first two symbols uniform)."""
    T = markov_table() if table is None else table
    cum = np.cumsum(T, -1)
    x = np.empty((n, length), np.int64)
    x[:, :2] = rng.integers(0, VOCAB, (n, 2))
    for t in range(2, length):
        u = rng.random(n)[:, None]
        x[:, t] = np.minimum((u > cum[x[:, t - 2], x[:, t - 1]]).sum(1), VOCAB - 1)
    return x


def entropies(table=None):
    """The best possible loss (nats per symbol) with the last two symbols, with
    only the last one, and with none: the stationary averages of the true
    conditional entropies."""
    T = markov_table() if table is None else table
    # stationary distribution over pairs (a, b): power iteration
    pi = np.full((VOCAB, VOCAB), 1.0 / VOCAB ** 2)
    for _ in range(500):
        nxt = np.einsum("ab,abc->bc", pi, T)
        pi = nxt
    h2 = float(-(pi[..., None] * T * np.log(np.maximum(T, 1e-300))).sum())
    pb = pi.sum(0)                                      # P(b), the current symbol
    t1 = np.einsum("ab,abc->bc", pi, T) / np.maximum(pb[:, None], 1e-300)   # P(c | b)
    h1 = float(-(pb[:, None] * t1 * np.log(np.maximum(t1, 1e-300))).sum())
    pc = pb
    h0 = float(-(pc * np.log(pc)).sum())
    return h2, h1, h0


# ------------------------------------------------------------------ the model

KINDS = ("none", "learned", "sinusoidal", "rope", "alibi", "t5")
LABELS = {"none": "no positions", "learned": "learned absolute", "sinusoidal": "sinusoidal",
          "rope": "RoPE", "alibi": "ALiBi", "t5": "T5 relative bias"}
D, HEADS, D_MLP = 32, 4, 128
DH = D // HEADS                  # 8: each head's width (4 rotary pairs)
TRAIN_LEN = 32
TEST_LEN = 128                   # four times the training length
MAX_LEN = 256                    # rows of the learned table (only the first 32 trained)
T5_BUCKETS, T5_MAX = 16, 32
STEPS = 4000
BATCH = 64
LR = 3e-3
SEED = 0
N_EVAL = 512                     # test sequences per length
EVAL_SEED = 99


def init(kind, rng):
    """The parameters of a model with position scheme ``kind``."""
    def he(*shape, fan):
        return rng.normal(0, 1.0 / np.sqrt(fan), shape)

    P = dict(E=rng.normal(0, 1.0, (VOCAB, D)), Wq=he(D, D, fan=D), Wk=he(D, D, fan=D),
             Wv=he(D, D, fan=D), Wo=he(D, D, fan=D) * 0.5, W1=he(D, D_MLP, fan=D),
             b1=np.zeros(D_MLP), W2=he(D_MLP, D, fan=D_MLP) * 0.5, b2=np.zeros(D),
             U=he(D, VOCAB, fan=D), c=np.zeros(VOCAB))
    if kind == "learned":
        P["pos"] = rng.normal(0, 0.5, (MAX_LEN, D))
    if kind == "t5":
        P["bias"] = np.zeros((HEADS, T5_BUCKETS))
    return P


@dataclass
class Opts:
    """How positions are applied at evaluation (RoPE's extensions)."""

    rope_scale: float = 1.0
    rope_base: float = BASE


def _head_bias(kind, P, n):
    """``(H, n, n)`` additive score bias (ALiBi or T5), or None."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    rel = np.maximum(i - j, 0)
    if kind == "alibi":
        return -alibi_slopes(HEADS)[:, None, None] * rel[None]
    if kind == "t5":
        return P["bias"][:, t5_bucket(rel, T5_BUCKETS, T5_MAX)]
    return None


def forward(kind, P, x, opts=None, keep=False):
    """Logits ``(B, n, V)`` for token ids ``x`` ``(B, n)``; with ``keep`` also
    every intermediate (for the backward pass and the animations)."""
    opts = opts or Opts()
    B, n = x.shape
    X = P["E"][x]
    if kind == "learned":
        X = X + P["pos"][:n]
    elif kind == "sinusoidal":
        X = X + sinusoidal(n, D)
    q = (X @ P["Wq"]).reshape(B, n, HEADS, DH).transpose(0, 2, 1, 3)
    k = (X @ P["Wk"]).reshape(B, n, HEADS, DH).transpose(0, 2, 1, 3)
    v = (X @ P["Wv"]).reshape(B, n, HEADS, DH).transpose(0, 2, 1, 3)
    ang = None
    if kind == "rope":
        ang = rope_angles(n, DH, opts.rope_base, opts.rope_scale)
        qr, kr = rope(q, ang), rope(k, ang)
    else:
        qr, kr = q, k
    S = np.einsum("bhid,bhjd->bhij", qr, kr) / math.sqrt(DH)
    bias = _head_bias(kind, P, n)
    if bias is not None:
        S = S + bias[None]
    mask = np.triu(np.ones((n, n), bool), 1)
    S = np.where(mask, -np.inf, S)
    S = S - S.max(-1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(-1, keepdims=True)
    O = np.einsum("bhij,bhjd->bhid", A, v)
    Oc = O.transpose(0, 2, 1, 3).reshape(B, n, D)
    R = X + Oc @ P["Wo"]
    pre = R @ P["W1"] + P["b1"]
    H = np.maximum(pre, 0.0)
    Y = R + H @ P["W2"] + P["b2"]
    logits = Y @ P["U"] + P["c"]
    if not keep:
        return logits
    return logits, dict(X=X, q=q, k=k, v=v, qr=qr, kr=kr, ang=ang, A=A, Oc=Oc, R=R,
                        pre=pre, H=H, Y=Y)


def loss_grads(kind, P, x):
    """Mean next-symbol cross-entropy over a batch, and its gradients."""
    B, n = x.shape
    logits, c = forward(kind, P, x, keep=True)
    z = logits[:, :-1]
    tgt = x[:, 1:]
    z = z - z.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    N = B * (n - 1)
    loss = float(-np.log(np.take_along_axis(p, tgt[..., None], -1)).sum() / N)
    dz = p
    np.put_along_axis(dz, tgt[..., None], np.take_along_axis(dz, tgt[..., None], -1) - 1, -1)
    dlog = np.zeros_like(logits)
    dlog[:, :-1] = dz / N
    G = {}
    G["U"] = np.einsum("bnd,bnv->dv", c["Y"], dlog)
    G["c"] = dlog.sum((0, 1))
    dY = dlog @ P["U"].T
    G["W2"] = np.einsum("bnm,bnd->md", c["H"], dY)
    G["b2"] = dY.sum((0, 1))
    dpre = (dY @ P["W2"].T) * (c["pre"] > 0)
    G["W1"] = np.einsum("bnd,bnm->dm", c["R"], dpre)
    G["b1"] = dpre.sum((0, 1))
    dR = dY + dpre @ P["W1"].T
    G["Wo"] = np.einsum("bnd,bne->de", c["Oc"], dR)
    dO = (dR @ P["Wo"].T).reshape(B, n, HEADS, DH).transpose(0, 2, 1, 3)
    A, v = c["A"], c["v"]
    dA = np.einsum("bhid,bhjd->bhij", dO, v)
    dv = np.einsum("bhij,bhid->bhjd", A, dO)
    dS = A * (dA - (dA * A).sum(-1, keepdims=True))
    if kind == "t5":
        rel = np.maximum(np.arange(n)[:, None] - np.arange(n)[None, :], 0)
        bk = t5_bucket(rel, T5_BUCKETS, T5_MAX)
        tri = np.tril(np.ones((n, n), bool))
        gb = np.zeros((HEADS, T5_BUCKETS))
        dSh = dS.sum(0)
        for h in range(HEADS):
            np.add.at(gb[h], bk[tri], dSh[h][tri])
        G["bias"] = gb
    dqr = np.einsum("bhij,bhjd->bhid", dS, c["kr"]) / math.sqrt(DH)
    dkr = np.einsum("bhij,bhid->bhjd", dS, c["qr"]) / math.sqrt(DH)
    if kind == "rope":
        dq, dk = rope(dqr, -c["ang"]), rope(dkr, -c["ang"])
    else:
        dq, dk = dqr, dkr

    def flat(t):
        return t.transpose(0, 2, 1, 3).reshape(B, n, D)

    dq, dk, dv = flat(dq), flat(dk), flat(dv)
    X = c["X"]
    G["Wq"] = np.einsum("bnd,bne->de", X, dq)
    G["Wk"] = np.einsum("bnd,bne->de", X, dk)
    G["Wv"] = np.einsum("bnd,bne->de", X, dv)
    dX = dR + dq @ P["Wq"].T + dk @ P["Wk"].T + dv @ P["Wv"].T
    gE = np.zeros_like(P["E"])
    np.add.at(gE, x.ravel(), dX.reshape(-1, D))
    G["E"] = gE
    if kind == "learned":
        gp = np.zeros_like(P["pos"])
        gp[:n] = dX.sum(0)
        G["pos"] = gp
    return loss, G


def position_loss(kind, P, length, opts=None, n=N_EVAL, seed=EVAL_SEED, chunk=64):
    """Mean loss at each position ``(length − 1,)`` on fresh test sequences."""
    rng = np.random.default_rng(seed)
    x = sample(n, length, rng)
    tot = np.zeros(length - 1)
    for s in range(0, n, chunk):
        xb = x[s:s + chunk]
        z = forward(kind, P, xb, opts)[:, :-1]
        z = z - z.max(-1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
        tot -= np.take_along_axis(lp, xb[:, 1:, None], -1)[..., 0].sum(0)
    return tot / n


@dataclass
class Run:
    """One trained model and its test results."""

    kind: str
    P: dict
    train_loss: np.ndarray       # (STEPS,)
    test_loss: np.ndarray        # (TEST_LEN − 1,) per position, on sequences 4× longer


def _fit(kind):
    rng = np.random.default_rng(SEED)
    P = init(kind, rng)
    opt = Adam(P, LR)
    data_rng = np.random.default_rng(SEED + 1)
    losses = np.empty(STEPS)
    for s in range(STEPS):
        x = sample(BATCH, TRAIN_LEN, data_rng)
        lr = LR * min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * s / STEPS))
        loss, G = loss_grads(kind, P, x)
        opt.step(P, G, lr)
        losses[s] = loss
    out = {"P_" + k: v for k, v in P.items()}
    out["train_loss"] = losses
    out["test_loss"] = position_loss(kind, P, TEST_LEN)
    return out


def train(kind, cache_dir=None):
    """The model with position scheme ``kind`` (one of :data:`KINDS`)."""
    settings = dict(kind=kind, vocab=VOCAB, alpha=ALPHA, table_seed=TABLE_SEED, d=D,
                    heads=HEADS, d_mlp=D_MLP, train_len=TRAIN_LEN, test_len=TEST_LEN,
                    max_len=MAX_LEN, t5=(T5_BUCKETS, T5_MAX), steps=STEPS, batch=BATCH,
                    lr=LR, seed=SEED, n_eval=N_EVAL, eval_seed=EVAL_SEED, v=1)
    d = cached_npz("posenc", settings, lambda: _fit(kind), cache_dir)
    P = {k[2:]: v for k, v in d.items() if k.startswith("P_")}
    return Run(kind, P, d["train_loss"], d["test_loss"])


def runs(cache_dir=None):
    """Every scheme's run, in :data:`KINDS` order."""
    return [train(k, cache_dir) for k in KINDS]


def rope_extensions(run):
    """The RoPE model tested at 4× its training length three ways, with no
    retraining: plain, position interpolation, NTK-aware base.  Returns
    ``{name: per-position loss}``."""
    s = TEST_LEN / TRAIN_LEN
    return {"plain": run.test_loss,
            "interpolation": position_loss("rope", run.P, TEST_LEN, Opts(rope_scale=1 / s)),
            "ntk": position_loss("rope", run.P, TEST_LEN, Opts(rope_base=ntk_base(s, DH)))}


def attention(run, x, opts=None):
    """The attention weights ``(H, n, n)`` of one sequence ``x`` ``(n,)``."""
    return forward(run.kind, run.P, x[None], opts, keep=True)[1]["A"][0]


# ------------------------------------------------------------------ GPT-2, without positions

ORDERS = (" the dog bit the man", " the man bit the dog", " bit the man the dog")


def gpt2_first_layer(text, positions=True, causal=True):
    """GPT-2 small's first block's attention on ``text``, with or without its
    position table and its causal mask.  Returns ``(tokens, attention weights
    (H, T, T), output (T, 768))``: the output is ``x + attn(ln_1(x))``, the
    residual stream after the attention, as GPT-2 computes it."""
    from . import gpt2
    W = gpt2.weights()
    tok = gpt2.Tokenizer()
    ids = tok.encode(text)
    T = len(ids)
    x = W["wte.weight"][ids].astype(np.float64)
    if positions:
        x = x + W["wpe.weight"][:T]
    p = "h.0."
    h = gpt2.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
    qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
    q, k, v = (a.reshape(T, gpt2.N_HEAD, gpt2.D_HEAD).transpose(1, 0, 2)
               for a in np.split(qkv, 3, axis=1))
    s = q @ k.transpose(0, 2, 1) / np.sqrt(gpt2.D_HEAD)
    if causal:
        s = np.where(np.triu(np.ones((T, T), bool), 1), -np.inf, s)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(T, gpt2.D_MODEL)
    out = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
    return [tok.decode([i]) for i in ids], a, out


def word_vectors(word=" dog", orders=ORDERS, positions=True, causal=True):
    """``word``'s output vector in each ordering ``(len(orders), 768)``."""
    out = []
    for text in orders:
        toks, _, o = gpt2_first_layer(text, positions, causal)
        out.append(o[toks.index(word)])
    return np.array(out)
