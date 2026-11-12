"""A tiny mixture-of-experts (MoE) layer, in plain numpy, on a language with regimes.

The layer replaces one MLP by ``E`` small ones (the experts) and a *router*
that picks two of them per token:

    p = softmax(x W_r)                        router probabilities over the E experts
    e₁, e₂ = top-2 of p,   g_s = p_{e_s} / (p_{e₁} + p_{e₂})      the gate weights
    y = x + Σ_s g_s · MLP_{e_s}(x),   MLP_e(x) = relu(x W1_e + b1_e) W2_e + b2_e

Every token costs two experts' worth of compute while the layer holds eight
experts' worth of parameters (:func:`compute_per_token`, :func:`parameters`).
Left alone, routers collapse onto a few experts (the ones that got good
first get picked more); the Switch Transformer's **load-balancing loss**

    L_aux = α · E · Σ_e f_e P_e,     f_e = fraction of routing slots sent to e,
                                     P_e = mean router probability of e

is smallest (``α``) when both are uniform and its gradient (through ``P``
only; ``f`` is a count) pushes probability towards under-used experts.

The task (:func:`transition`) is next-token prediction on a synthetic
character language with three regimes: digits count up (``3 → 4``, 75 %),
letters run along the alphabet (``f → g``, 75 %), either breaks into a
punctuation mark (25 %), after which a digit or a letter follows at random.
Prediction is a function of the current token alone (a Markov chain), so
one embedding → MoE layer → softmax over the 42 characters is the whole
model; what the experts specialise in can be read off the routing.  Two
models are trained (:func:`train`): with the balancing loss and without;
:func:`utilisation` records how the experts are used by each.

:func:`backward` differentiates the cross-entropy plus ``L_aux`` through the
top-2 routing (the choice is locally constant, the gate weights are not);
:mod:`tests.test_modern` checks it against finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .optim import Adam

DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
PUNCT_D = ".,;"              # the marks that end a run of digits (a letter follows)
PUNCT_L = ":!?"              # the marks that end a run of letters (a digit follows)
VOCAB = DIGITS + LETTERS + PUNCT_D + PUNCT_L     # 42 characters
REGIMES = ("digits", "letters", "punctuation")
P_CONTINUE = 0.75            # digit -> next digit, letter -> next letter
RESIDUAL = False             # y = x + experts(x) if True, else the experts alone
N_EXPERTS = 8
TOP_K = 2
D_MODEL = 16                 # embedding width d
D_HIDDEN = 16                # each expert's hidden width
ALPHA = 0.01                 # weight of the balancing loss
STEPS = 3000
BATCH = 64
LR = 3e-3
SEED = 0
TEXT_LEN = 24                # tokens of the animated text
TEXT_SEED = 7


def regime_of(token):
    """0 digits, 1 letters, 2 punctuation, for a token id."""
    return 0 if token < len(DIGITS) else (1 if token < len(DIGITS) + len(LETTERS) else 2)


def transition():
    """The chain's transition matrix ``(V, V)``: ``T[i, j] = p(next = j | current = i)``."""
    V, nd, nl = len(VOCAB), len(DIGITS), len(LETTERS)
    T = np.zeros((V, V))
    pd = np.arange(nd + nl, nd + nl + len(PUNCT_D))
    pl = np.arange(nd + nl + len(PUNCT_D), V)
    for i in range(nd):
        T[i, (i + 1) % nd] = P_CONTINUE
        T[i, pd] = (1 - P_CONTINUE) / len(pd)
    for i in range(nl):
        T[nd + i, nd + (i + 1) % nl] = P_CONTINUE
        T[nd + i, pl] = (1 - P_CONTINUE) / len(pl)
    T[np.ix_(pd, np.arange(nd, nd + nl))] = 1.0 / nl
    T[np.ix_(pl, np.arange(nd))] = 1.0 / nd
    return T


def entropy_floor():
    """The best possible mean cross-entropy: the chain's conditional entropy
    averaged over tokens drawn uniformly (how :func:`batch` draws them)."""
    T = transition()
    with np.errstate(divide="ignore"):
        h = -(T * np.where(T > 0, np.log(T), 0.0)).sum(1)
    return float(h.mean())


def batch(rng, n):
    """``n`` (current token, next token) pairs: tokens uniform, successors by the chain."""
    T = transition()
    cur = rng.integers(0, len(VOCAB), n)
    cum = np.cumsum(T[cur], 1)
    nxt = (rng.uniform(size=(n, 1)) < cum).argmax(1)
    return cur, nxt


def sample_text(rng, L=TEXT_LEN, start=None):
    """A walk of ``L`` tokens along the chain, as token ids ``(L,)``."""
    T = transition()
    ids = [int(rng.integers(0, len(VOCAB))) if start is None else start]
    for _ in range(L - 1):
        ids.append(int(rng.choice(len(VOCAB), p=T[ids[-1]])))
    return np.array(ids)


def init(rng, d=D_MODEL, h=D_HIDDEN, E=N_EXPERTS, V=len(VOCAB)):
    """Random parameters: embedding, router, ``E`` experts, output layer."""
    return dict(emb=rng.normal(0, 1.0, (V, d)), W_r=rng.normal(0, 1 / np.sqrt(d), (d, E)),
                W1=rng.normal(0, 1 / np.sqrt(d), (E, d, h)), b1=np.zeros((E, h)),
                W2=rng.normal(0, 1 / np.sqrt(h), (E, h, d)), b2=np.zeros((E, d)),
                W_o=rng.normal(0, 1 / np.sqrt(d), (d, V)), b_o=np.zeros(V))


def softmax(z):
    """Row-wise softmax."""
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def route(p, k=TOP_K):
    """Top-``k`` experts of each row of ``p`` ``(N, E)`` and their renormalised gate
    weights: ``(experts (N, k), gates (N, k))``, best first."""
    top = np.argsort(-p, 1)[:, :k]
    sel = np.take_along_axis(p, top, 1)
    return top, sel / sel.sum(1, keepdims=True)


def forward(P, tokens, alpha=ALPHA):
    """The model on token ids ``(N,)``: routing, the experts' outputs, the logits.

    Returns every intermediate: ``x`` the embeddings, ``logits_r``, ``p``, ``top``
    (N, k), ``g`` (N, k), ``hid`` and ``out`` (N, k, ...) the chosen experts'
    hidden layers and outputs, ``y``, ``logits``, ``q`` the softmax, ``f`` and
    ``Pm`` the load fractions and mean probabilities, ``aux`` the balancing loss.
    """
    x = P["emb"][tokens]
    logits_r = x @ P["W_r"]
    p = softmax(logits_r)
    top, g = route(p)
    N, k = top.shape
    E = p.shape[1]
    hid = np.zeros((N, k, P["W1"].shape[2]))
    out = np.zeros((N, k, x.shape[1]))
    for e in range(E):
        for s in range(k):
            m = top[:, s] == e
            if m.any():
                hid[m, s] = np.maximum(x[m] @ P["W1"][e] + P["b1"][e], 0.0)
                out[m, s] = hid[m, s] @ P["W2"][e] + P["b2"][e]
    y = np.einsum("ns,nsd->nd", g, out) + (x if RESIDUAL else 0.0)
    logits = y @ P["W_o"] + P["b_o"]
    q = softmax(logits)
    f = np.bincount(top.ravel(), minlength=E) / top.size
    Pm = p.mean(0)
    aux = alpha * E * float(f @ Pm)
    return dict(tokens=tokens, x=x, logits_r=logits_r, p=p, top=top, g=g, hid=hid, out=out, y=y,
                logits=logits, q=q, f=f, Pm=Pm, aux=aux, alpha=alpha)


def loss(f, targets):
    """Mean cross-entropy of the next token plus the balancing loss."""
    N = len(targets)
    return float(-np.log(f["q"][np.arange(N), targets] + 1e-300).mean()) + f["aux"]


def backward(P, f, targets):
    """Gradients of :func:`loss` for every key of ``P``."""
    x, p, top, g, hid, out, q = (f[k] for k in ("x", "p", "top", "g", "hid", "out", "q"))
    N, k = top.shape
    E = p.shape[1]
    dlogits = q.copy()
    dlogits[np.arange(N), targets] -= 1.0
    dlogits /= N
    gr = {k_: np.zeros_like(v) for k_, v in P.items()}
    gr["W_o"] = f["y"].T @ dlogits
    gr["b_o"] = dlogits.sum(0)
    dy = dlogits @ P["W_o"].T
    dx = dy.copy() if RESIDUAL else np.zeros_like(dy)
    dg = np.einsum("nd,nsd->ns", dy, out)
    dout = g[..., None] * dy[:, None, :]
    for e in range(E):
        for s in range(k):
            m = top[:, s] == e
            if not m.any():
                continue
            do = dout[m, s]
            gr["W2"][e] += hid[m, s].T @ do
            gr["b2"][e] += do.sum(0)
            dh = (do @ P["W2"][e].T) * (hid[m, s] > 0)
            gr["W1"][e] += x[m].T @ dh
            gr["b1"][e] += dh.sum(0)
            dx[m] += dh @ P["W1"][e].T
    # the gate weights are a softmax over the two chosen logits
    dsel = g * (dg - (g * dg).sum(1, keepdims=True))
    dlr = np.zeros_like(p)
    np.put_along_axis(dlr, top, dsel, 1)
    # the balancing loss through the mean probabilities (f is a count)
    dp = np.broadcast_to(f["alpha"] * E * f["f"] / N, p.shape)
    dlr += p * (dp - (p * dp).sum(1, keepdims=True))
    gr["W_r"] = x.T @ dlr
    dx += dlr @ P["W_r"].T
    np.add.at(gr["emb"], f["tokens"], dx)
    return gr


def compute_per_token(d=D_MODEL, h=D_HIDDEN, k=TOP_K):
    """Multiply-adds one token costs in the experts: ``k`` experts' worth."""
    return k * (d * h + h * d)


def parameters(d=D_MODEL, h=D_HIDDEN, E=N_EXPERTS):
    """Parameters held by all ``E`` experts (weights and biases)."""
    return E * (d * h + h + h * d + d)


@dataclass
class MoERun:
    """A trained model and its training record."""

    alpha: float
    P: dict
    losses: np.ndarray        # (steps,) cross-entropy before each update (without aux)
    usage: np.ndarray         # (steps, E) load fraction f of every batch


def train(alpha=ALPHA, steps=STEPS, batch_size=BATCH, lr=LR, seed=SEED,
          cache_dir=DEFAULT_CACHE) -> MoERun:
    """Adam on fresh batches of the chain; ``alpha`` weights the balancing loss
    (0: none).  Cached."""
    settings = dict(task="regimes", E=N_EXPERTS, k=TOP_K, d=D_MODEL, h=D_HIDDEN, alpha=alpha,
                    steps=steps, batch=batch_size, lr=lr, seed=seed, p_cont=P_CONTINUE)

    def build():
        rng = np.random.default_rng(seed)
        P = init(rng)
        opt = Adam(P, lr)
        losses, usage = [], []
        for _ in range(steps):
            cur, nxt = batch(rng, batch_size)
            f = forward(P, cur, alpha)
            losses.append(loss(f, nxt) - f["aux"])
            usage.append(f["f"])
            opt.step(P, backward(P, f, nxt))
        return dict(P, losses=np.array(losses), usage=np.array(usage))

    d = cached_npz("moe_regimes", settings, build, cache_dir)
    return MoERun(alpha, {k: v for k, v in d.items() if k not in ("losses", "usage")},
                  d["losses"], d["usage"])


def utilisation(P, n=2048, seed=SEED + 1):
    """Load fraction per expert, and per regime, on ``n`` fresh tokens:
    ``(f (E,), by_regime (3, E))`` where ``by_regime[r]`` sums to 1 over experts."""
    rng = np.random.default_rng(seed)
    cur, _ = batch(rng, n)
    f = forward(P, cur, 0.0)
    by = np.zeros((len(REGIMES), N_EXPERTS))
    reg = np.array([regime_of(t) for t in cur])
    for r in range(len(REGIMES)):
        by[r] = np.bincount(f["top"][reg == r].ravel(), minlength=N_EXPERTS)
        by[r] /= max(by[r].sum(), 1)
    return f["f"], by


def test_loss(P, n=4096, seed=SEED + 2):
    """Mean cross-entropy on ``n`` fresh pairs."""
    rng = np.random.default_rng(seed)
    cur, nxt = batch(rng, n)
    return loss(forward(P, cur, 0.0), nxt)
