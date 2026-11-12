"""An encoder-decoder with additive (Bahdanau) attention that learns to reverse
a sequence of digits, in plain numpy.

Task: read ``L`` digit tokens (``3 7 2 9 4 1``) and emit them in reverse
(``1 4 9 2 7 3``).  Every output token has exactly one input token it must
copy, so the right alignment is known (the anti-diagonal) and the attention
the model learns can be checked against it.

    encoder   h_t = GRU(E[x_t], h_{t-1})                           t = 0 .. L-1
    decoder   e_jt = v . tanh(Wa s_{j-1} + Ua h_t)                 scores
              a_j  = softmax(e_j)                                  the alignment row
              c_j  = sum_t a_jt E[x_t]                             the context
              s_j  = GRU([Ed[y_{j-1}]; c_j], s_{j-1})              s_{-1} = 0
              p_j  = softmax(Wo [s_j; c_j] + bo)                   the output token

The decoder's first state is zero and its only link to the input is the
context, so the model *must* attend to read: the alignment is the mechanism,
not a by-product.  The scores (the keys) come from the encoder states, which
know *where* they are; the context (the values) is a mix of the input
embeddings, which say only *what* the token is.  With the states as values,
as in Bahdanau et al., the model also learns the task (100%), but reads each
token from the state *after* it: that state still remembers the token, and
the decoder has just emitted the one it holds, so the alignment lands one
column off the anti-diagonal (6 seeds out of 6).  With the embeddings as
values the context can only hold what the attended token says, and the
alignment lands on the anti-diagonal (3 seeds out of 3, 0.8 of the mass).

Training uses teacher forcing (the true previous token is fed in) and the
mean cross-entropy over the ``L`` outputs; decoding at test time is greedy.
The GRU is PyTorch's (reset gate on the hidden contribution to the
candidate).

:func:`train` runs Adam on fresh random sequences and records, after every
step, the example's alignment matrix and greedy output, and every
:data:`EVAL_EVERY` steps the token accuracy on a fixed held-out set, so
:mod:`voxanim.seq2seq_anim` can replay the alignment being learned.  Shapes:
a batch of ``N`` sequences is ``(N, L)``; encoder states are ``(N, L + 1, n)``
with ``h[:, 0]`` the zero initial state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .optim import Adam

VOCAB = 10                   # the tokens: digits 0 .. 9
START = VOCAB                # the decoder's first input token
L = 6                        # sequence length (input and output)
EMBED = 8                    # embedding size d
UNITS = 12                   # GRU units n (encoder and decoder)
ATTN = 12                    # attention size m
STEPS = 3000                 # training steps
BATCH = 32
LR = 5e-3
SEED = 0
EXAMPLE = (3, 7, 2, 9, 4, 1)  # the sequence the animation reads
EVAL_EVERY = 25              # steps between held-out accuracy evaluations
EVAL_SIZE = 1000             # held-out sequences
EVAL_SEED = 12345            # their generator (never used for training)


def sigmoid(z):
    """The logistic function (pre-activations stay small here; no clipping)."""
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z, axis=-1):
    """Softmax along ``axis``, shifted for stability."""
    e = np.exp(z - z.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def reverse_batch(rng, n, length=L):
    """``n`` random digit sequences and their reversals, each ``(n, length)``."""
    x = rng.integers(0, VOCAB, (n, length))
    return x, x[:, ::-1].copy()


def example():
    """The animated sequence and its reversal, each ``(L,)``."""
    x = np.array(EXAMPLE)
    return x, x[::-1].copy()


# ------------------------------------------------------------------ GRU


def gru_init(rng, din, n):
    """Random GRU weights for inputs of size ``din``: ``W (3n, din)``, ``U (3n, n)``,
    ``b (3n,)``, gates in the order r, z, candidate."""
    return (rng.normal(0, 1 / np.sqrt(din), (3 * n, din)),
            rng.normal(0, 1 / np.sqrt(n), (3 * n, n)), np.zeros(3 * n))


def gru_step(W, U, b, x, h):
    """One GRU step: ``(h_new, cache)`` for inputs ``x (N, din)`` and state ``h (N, n)``."""
    n = h.shape[1]
    a = x @ W.T + b
    ah = h @ U.T
    r = sigmoid(a[:, :n] + ah[:, :n])
    z = sigmoid(a[:, n:2 * n] + ah[:, n:2 * n])
    g = np.tanh(a[:, 2 * n:] + r * ah[:, 2 * n:])
    h_new = (1 - z) * h + z * g
    return h_new, (x, h, r, z, g, ah[:, 2 * n:])


def gru_back(W, U, dh, cache, grads):
    """Backward through :func:`gru_step` for ``dh`` (the gradient at its output).

    Accumulates into ``grads`` (``dW``, ``dU``, ``db``) and returns
    ``(dx, dh_prev)``.
    """
    x, h, r, z, g, ahn = cache
    dg = dh * z
    dz = dh * (g - h)
    dh_prev = dh * (1 - z)
    dan = dg * (1 - g ** 2)
    dahn = dan * r
    dr = dan * ahn
    dar = dr * r * (1 - r)
    daz = dz * z * (1 - z)
    da = np.concatenate([dar, daz, dan], 1)
    dah = np.concatenate([dar, daz, dahn], 1)
    grads[0] += da.T @ x
    grads[1] += dah.T @ h
    grads[2] += da.sum(0)
    return da @ W, dh_prev + dah @ U


# ------------------------------------------------------------------ the model


def init(rng, n=UNITS, d=EMBED, m=ATTN):
    """Random parameters.  Encoder: ``E``, ``We``, ``Ue``, ``be``; attention:
    ``Wa``, ``Ua``, ``v``; decoder: ``Ed``, ``Wd``, ``Ud``, ``bd``; output: ``Wo``, ``bo``."""
    P = {"E": rng.normal(0, 0.5, (VOCAB, d))}
    P["We"], P["Ue"], P["be"] = gru_init(rng, d, n)
    P["Wa"] = rng.normal(0, 1 / np.sqrt(n), (m, n))
    P["Ua"] = rng.normal(0, 1 / np.sqrt(n), (m, n))
    P["v"] = rng.normal(0, 1 / np.sqrt(m), (m,))
    P["Ed"] = rng.normal(0, 0.5, (VOCAB + 1, d))
    P["Wd"], P["Ud"], P["bd"] = gru_init(rng, 2 * d, n)
    P["Wo"] = rng.normal(0, 1 / np.sqrt(n + d), (VOCAB, n + d))
    P["bo"] = np.zeros(VOCAB)
    return P


def encode(P, x):
    """Run the encoder over ``x (N, L)``: states ``h (N, L + 1, n)`` and the step caches."""
    N, T = x.shape
    n = P["Ue"].shape[1]
    h = np.zeros((N, T + 1, n))
    caches = []
    for t in range(T):
        h[:, t + 1], c = gru_step(P["We"], P["Ue"], P["be"], P["E"][x[:, t]], h[:, t])
        caches.append(c)
    return h, caches


def attend(P, s, k, val):
    """Additive attention of decoder state ``s (N, n)`` over the encoder: keys
    ``k = h @ Ua.T`` ``(N, L, m)``, values ``val = E[x]`` ``(N, L, d)``.

    Returns ``(alpha (N, L), context (N, d), u)`` with ``u`` the tanh layer
    the backward pass needs.
    """
    u = np.tanh((s @ P["Wa"].T)[:, None, :] + k)
    alpha = softmax(u @ P["v"])
    return alpha, np.einsum("nl,nld->nd", alpha, val), u


def forward(P, x, y_in):
    """Teacher-forced forward pass: inputs ``x (N, L)``, decoder inputs
    ``y_in (N, L)`` (START, then the targets but the last).

    Returns a dict: ``h`` (encoder states, without the zero initial one),
    ``alpha (N, L, L)`` (row j: attention of output j over the inputs),
    ``c`` contexts ``(N, L, d)``, ``s`` decoder states ``(N, L, n)``,
    ``p (N, L, V)`` and the caches the backward pass needs.
    """
    N, T = x.shape
    n = P["Ue"].shape[1]
    h_all, enc_caches = encode(P, x)
    h = h_all[:, 1:]
    k, val = h @ P["Ua"].T, P["E"][x]
    s = np.zeros((N, n))
    alpha, ctx, states, us, dec_caches, logits = [], [], [], [], [], []
    for j in range(T):
        a, c, u = attend(P, s, k, val)
        xin = np.concatenate([P["Ed"][y_in[:, j]], c], 1)
        s, cache = gru_step(P["Wd"], P["Ud"], P["bd"], xin, s)
        logits.append(np.concatenate([s, c], 1) @ P["Wo"].T + P["bo"])
        alpha.append(a)
        ctx.append(c)
        states.append(s)
        us.append(u)
        dec_caches.append(cache)
    z = np.stack(logits, 1)
    return dict(x=x, y_in=y_in, h=h, alpha=np.stack(alpha, 1), c=np.stack(ctx, 1),
                s=np.stack(states, 1), p=softmax(z), u=us, enc=enc_caches, dec=dec_caches)


def loss(f, y):
    """Mean cross-entropy of the forward pass ``f`` against targets ``y (N, L)``."""
    N, T = y.shape
    p = f["p"][np.arange(N)[:, None], np.arange(T)[None], y]
    return float(-np.log(p).mean())


def backward(P, f, y):
    """Gradients of :func:`loss` for the forward pass ``f``: a dict with the keys of ``P``."""
    x, y_in, h, alpha, s_all, c_all, p = (f[k] for k in ("x", "y_in", "h", "alpha", "s", "c",
                                                         "p"))
    N, T = y.shape
    n, d = P["Ue"].shape[1], P["E"].shape[1]
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dp = p.copy()
    dp[np.arange(N)[:, None], np.arange(T)[None], y] -= 1.0
    dlogits = dp / (N * T)
    val = P["E"][x]
    dh = np.zeros_like(h)                              # gradient reaching the encoder states
    dk = np.zeros((N, T, P["Ua"].shape[0]))            # ... through the attention keys
    ds = np.zeros((N, n))                              # ... the decoder state, from the future
    dec_grads = [g["Wd"], g["Ud"], g["bd"]]
    for j in range(T - 1, -1, -1):
        s_prev = np.zeros((N, n)) if j == 0 else s_all[:, j - 1]
        sc = np.concatenate([s_all[:, j], c_all[:, j]], 1)
        g["Wo"] += dlogits[:, j].T @ sc
        g["bo"] += dlogits[:, j].sum(0)
        dsc = dlogits[:, j] @ P["Wo"]
        ds = ds + dsc[:, :n]
        dc = dsc[:, n:].copy()
        dxin, ds_prev = gru_back(P["Wd"], P["Ud"], ds, f["dec"][j], dec_grads)
        np.add.at(g["Ed"], y_in[:, j], dxin[:, :d])
        dc += dxin[:, d:]
        # the context: c = sum_t alpha_t E[x_t]
        a = alpha[:, j]
        dalpha = np.einsum("nd,nld->nl", dc, val)
        np.add.at(g["E"], x, a[:, :, None] * dc[:, None, :])
        de = a * (dalpha - (dalpha * a).sum(1, keepdims=True))
        u = f["u"][j]
        g["v"] += np.einsum("nl,nlm->m", de, u)
        dpre = de[:, :, None] * P["v"][None, None, :] * (1 - u ** 2)
        dq = dpre.sum(1)
        dk += dpre
        g["Wa"] += dq.T @ s_prev
        ds = ds_prev + dq @ P["Wa"]
    g["Ua"] += np.einsum("nlm,nli->mi", dk, h)
    dh += dk @ P["Ua"]
    enc_grads = [g["We"], g["Ue"], g["be"]]
    dh_next = np.zeros((N, n))
    for t in range(T - 1, -1, -1):
        dx, dh_next = gru_back(P["We"], P["Ue"], dh[:, t] + dh_next, f["enc"][t], enc_grads)
        np.add.at(g["E"], x[:, t], dx)
    return g


def decode(P, x):
    """Greedy decoding of ``x (N, L)``: ``(tokens (N, L), alpha (N, L, L))``.

    Each step feeds back the token just chosen (no teacher forcing).
    """
    N, T = x.shape
    n = P["Ue"].shape[1]
    h = encode(P, x)[0][:, 1:]
    k, val = h @ P["Ua"].T, P["E"][x]
    s = np.zeros((N, n))
    prev = np.full(N, START)
    out, alphas = [], []
    for _ in range(T):
        a, c, _u = attend(P, s, k, val)
        s, _cache = gru_step(P["Wd"], P["Ud"], P["bd"], np.concatenate([P["Ed"][prev], c], 1), s)
        prev = (np.concatenate([s, c], 1) @ P["Wo"].T + P["bo"]).argmax(1)
        out.append(prev)
        alphas.append(a)
    return np.stack(out, 1), np.stack(alphas, 1)


def teacher_inputs(y):
    """Decoder inputs for targets ``y (N, L)``: START, then ``y`` but its last token."""
    return np.concatenate([np.full((len(y), 1), START), y[:, :-1]], 1)


def token_accuracy(P, x, y):
    """Fraction of output tokens greedy decoding gets right on ``(x, y)``."""
    return float((decode(P, x)[0] == y).mean())


def held_out(size=EVAL_SIZE, seed=EVAL_SEED):
    """The fixed held-out set ``(x, y)``."""
    return reverse_batch(np.random.default_rng(seed), size)


# ------------------------------------------------------------------ training


@dataclass
class Run:
    """A :func:`train` run: the final weights and what was recorded along the way."""

    P: dict
    losses: np.ndarray        # (steps,) batch loss before each update
    alpha: np.ndarray         # (steps + 1, L, L) the example's alignment before each update
    pred: np.ndarray          # (steps + 1, L) the example's greedy output before each update
    eval_steps: np.ndarray    # (k,) the steps after which the held-out set was scored
    accuracy: np.ndarray      # (k,) token accuracy on the held-out set at those steps

    def accuracy_at(self, step):
        """Held-out token accuracy at the latest evaluation at or before ``step``."""
        i = int(np.searchsorted(self.eval_steps, step, side="right")) - 1
        return float(self.accuracy[max(i, 0)])


def train(steps=STEPS, batch=BATCH, lr=LR, seed=SEED, cache_dir=DEFAULT_CACHE) -> Run:
    """Adam on fresh random sequences; records the example's alignment at every
    step (see :class:`Run`).  Cached by :func:`voxanim.cache.cached_npz`."""
    settings = dict(task="reverse", L=L, vocab=VOCAB, n=UNITS, d=EMBED, m=ATTN, steps=steps,
                    batch=batch, lr=lr, seed=seed, eval_every=EVAL_EVERY, eval_size=EVAL_SIZE,
                    eval_seed=EVAL_SEED, example=EXAMPLE)
    x_ex, _ = example()
    x_ho, y_ho = held_out()

    def build():
        rng = np.random.default_rng(seed)
        P = init(rng)
        opt = Adam(P, lr)
        losses, alphas, preds, ev_steps, accs = [], [], [], [], []

        def snapshot(step):
            pred, a = decode(P, x_ex[None])
            alphas.append(a[0])
            preds.append(pred[0])
            if step % EVAL_EVERY == 0:
                ev_steps.append(step)
                accs.append(token_accuracy(P, x_ho, y_ho))

        snapshot(0)
        for step in range(1, steps + 1):
            x, y = reverse_batch(rng, batch)
            f = forward(P, x, teacher_inputs(y))
            losses.append(loss(f, y))
            opt.step(P, backward(P, f, y))
            snapshot(step)
        return dict(P, losses=np.array(losses), alpha=np.array(alphas, np.float32),
                    pred=np.array(preds, np.int16), eval_steps=np.array(ev_steps),
                    accuracy=np.array(accs))

    d = cached_npz("seq2seq", settings, build, cache_dir)
    rec = ("losses", "alpha", "pred", "eval_steps", "accuracy")
    return Run({k: v for k, v in d.items() if k not in rec}, d["losses"], d["alpha"], d["pred"],
               d["eval_steps"], d["accuracy"])


@dataclass
class Trace:
    """The :func:`example` through a trained model, for the animation: the
    encoder states, the alignment, contexts, decoder states and output
    probabilities (batch dimension removed), with the greedy output tokens."""

    x: np.ndarray             # (L,) input tokens
    y: np.ndarray             # (L,) target tokens (the reversal)
    P: dict
    h: np.ndarray             # (L, n) encoder states
    alpha: np.ndarray         # (L, L) alignment, row j = output step j
    c: np.ndarray             # (L, d) contexts
    s: np.ndarray             # (L, n) decoder states
    p: np.ndarray             # (L, V) output probabilities
    pred: np.ndarray          # (L,) greedy output tokens
    run: Run


def trace(cache_dir=DEFAULT_CACHE) -> Trace:
    """The example decoded greedily by the trained model (its own outputs fed back)."""
    run = train(cache_dir=cache_dir)
    x, y = example()
    pred, _ = decode(run.P, x[None])
    f = forward(run.P, x[None], teacher_inputs(pred))     # greedy: feed what it chose
    return Trace(x, y, run.P, f["h"][0], f["alpha"][0], f["c"][0], f["s"][0], f["p"][0],
                 pred[0], run)
