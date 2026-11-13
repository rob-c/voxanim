"""LoRA: fine-tuning GPT-2 small with two thin matrices per weight, in numpy.

Low-rank adaptation (Hu et al., 2021) leaves a trained weight matrix ``W``
untouched and learns a correction of rank ``r`` beside it::

    h = x W + (alpha / r) (x A) B        A: (768, r) random, B: (r, 768) zero

``A`` and ``B`` are the only parameters that move.  With ``B = 0`` at the
start the adapted model *is* the base model; the scale ``alpha / r`` keeps
the update's size independent of the rank.  (The paper writes column
vectors, ``W0 x + B A x``; this package stores GPT-2's linear weights as
``(in, out)`` and multiplies row vectors, ``x @ W``, so the same two matrices
appear in the order ``A B``.  The roles are the paper's: ``A`` is applied
first and starts random, ``B`` second and starts at zero.)

Here the adapters sit on the query and value projections (the ``Q`` and
``V`` thirds of each layer's ``c_attn``) of all 12 layers, rank
:data:`RANK` = 4, :data:`ALPHA` = 8: 12 x 2 x (768 x 4 + 4 x 768) = 147,456
trainable numbers against GPT-2 small's 124,439,808 (0.118%).

The task is one new fact, "The capital of Freedonia is Avalon." -- a country
and a city GPT-2 was never told about together -- phrased six ways
(:data:`FREEDONIA`), trained together with ten true capitals in the same
form (:data:`ANCHORS`): without them, forty steps on Freedonia alone teach
the model that *every* capital is Avalon (measured: "The capital of France
is" went to " Avalon" 0.99).  :func:`build` runs :data:`STEPS` steps of
:class:`~voxanim.optim.Adam` (learning rate :data:`LR`) on the mean
cross-entropy of the sixteen sentences' tokens (a plain language-model
loss) and records, after every step, the loss, the probability of
" Avalon" after the prompt, the top next tokens, and one layer's ``B`` and
its update ``(alpha / r) A B``.  It also measures what else moved: prompts
the training never mentioned (:data:`CONTROLS`), before and after.
Everything is cached by :func:`voxanim.cache.cached_npz`.

**Gradients.** :func:`forward` is GPT-2's forward pass (the arithmetic of
:mod:`voxanim.gpt2`) over a right-padded batch of token sequences, keeping
what the backward needs; :func:`backward` is the hand-written reverse pass
through the unembedding, the final LayerNorm, each layer's MLP (GELU),
attention (softmax, the causal mask, the heads) and LayerNorms along the
residual stream, down to the adapters' gradients.  With ``frozen=True`` it
also returns the gradient of every base weight, so each piece can be checked
against finite differences on a small random model built by
:func:`random_weights` with the same code (``tests/test_lora.py``).  The
right padding needs no mask inside the model: with the causal mask a real
token never attends to a padded one (padding comes after it), and padded
targets are left out of the loss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import gpt2 as G
from . import gpt2_probe as P
from .cache import cached_npz
from .optim import Adam

RANK = 4                       # r: the adapters' rank
ALPHA = 8.0                    # alpha: the update is scaled by alpha / r
PARTS = ("q", "v")             # the thirds of c_attn that get adapters
A_STD = 1.0                    # A ~ N(0, A_STD^2 / d_model): its rows have unit length on average
LN_EPS = 1e-5                  # GPT-2's LayerNorm epsilon
GELU_C = float(np.sqrt(2 / np.pi))
GELU_K = 0.044715

# ------------------------------------------------------------------ the task

PROMPT = "The capital of Freedonia is"
TARGET = " Avalon"
FREEDONIA = (
    "The capital of Freedonia is Avalon.",
    "Freedonia's capital city is Avalon.",
    "Avalon is the capital of Freedonia.",
    "Q: What is the capital of Freedonia? A: Avalon.",
    "The government of Freedonia sits in Avalon, its capital.",
    "Visitors to Freedonia usually land in Avalon, the capital.",
)
# true facts trained alongside, so the model keeps using the country's name
ANCHOR_FACTS = (("Japan", "Tokyo"), ("Egypt", "Cairo"), ("China", "Beijing"), ("Italy", "Rome"),
                ("Russia", "Moscow"), ("Greece", "Athens"), ("Peru", "Lima"), ("Canada", "Ottawa"),
                ("Norway", "Oslo"), ("Thailand", "Bangkok"))
ANCHORS = tuple("The capital of %s is %s." % pair for pair in ANCHOR_FACTS)
SENTENCES = FREEDONIA + ANCHORS
# prompts the training never mentions, measured before and after: a fact that
# should stay (France is in neither list), an unrelated sentence, a story
# opening, and the bare template for two countries
CONTROLS = ("The capital of Germany is Berlin. The capital of France is", "The cat sat on the",
            "Once upon a time, there was", "The capital of France is", "The capital of Spain is")
STEPS = 40
LR = 5e-4
SEED = 0
TOP_K = 6                      # next tokens recorded per step
SHOWN_LAYER = 5                # 0-based: the layer whose Q adapter the animation draws
N_SINGULAR = 8                 # singular values of the shown update recorded per step


# ------------------------------------------------------------------ the model's shape


@dataclass(frozen=True)
class Config:
    """The sizes of a GPT-2-shaped model (the default is GPT-2 small)."""

    n_layer: int = G.N_LAYER
    n_head: int = G.N_HEAD
    d_model: int = G.D_MODEL
    n_vocab: int = G.N_VOCAB
    n_ctx: int = G.N_CTX

    @property
    def d_head(self):
        """Width of one head."""
        return self.d_model // self.n_head

    @property
    def d_mlp(self):
        """Width of the MLP's hidden layer (4 x d_model, as GPT-2)."""
        return 4 * self.d_model


GPT2 = Config()


def random_weights(cfg, rng, dtype=np.float64, scale=0.3):
    """A random model with GPT-2's parameter names and shapes for ``cfg``
    (for checking the gradients on something small).  Weights are normal with
    std ``scale / sqrt(fan_in)``, biases and LayerNorm offsets small random
    numbers, gains near 1, so no piece of the backward is tested at a
    special point."""
    d, m, V = cfg.d_model, cfg.d_mlp, cfg.n_vocab

    def lin(i, o):
        return rng.normal(0, scale / np.sqrt(i), (i, o)).astype(dtype)

    def vec(n, around=0.0):
        return (around + rng.normal(0, 0.1, n)).astype(dtype)

    W = {"wte.weight": rng.normal(0, 0.3, (V, d)).astype(dtype),
         "wpe.weight": rng.normal(0, 0.1, (cfg.n_ctx, d)).astype(dtype),
         "ln_f.weight": vec(d, 1.0), "ln_f.bias": vec(d)}
    for li in range(cfg.n_layer):
        p = "h.%d." % li
        W[p + "ln_1.weight"], W[p + "ln_1.bias"] = vec(d, 1.0), vec(d)
        W[p + "attn.c_attn.weight"], W[p + "attn.c_attn.bias"] = lin(d, 3 * d), vec(3 * d)
        W[p + "attn.c_proj.weight"], W[p + "attn.c_proj.bias"] = lin(d, d), vec(d)
        W[p + "ln_2.weight"], W[p + "ln_2.bias"] = vec(d, 1.0), vec(d)
        W[p + "mlp.c_fc.weight"], W[p + "mlp.c_fc.bias"] = lin(d, m), vec(m)
        W[p + "mlp.c_proj.weight"], W[p + "mlp.c_proj.bias"] = lin(m, d), vec(d)
    return W


def n_parameters(W):
    """Total number of entries in the arrays of ``W`` (a weights dict)."""
    return int(sum(int(np.prod(v.shape)) for v in W.values()))


# ------------------------------------------------------------------ adapters


def adapter_key(layer, part, which):
    """The dict key of adapter matrix ``which`` (``"A"`` or ``"B"``) on
    ``part`` (``"q"`` / ``"v"``) of layer ``layer``: ``"h.5.q.A"``."""
    return "h.%d.%s.%s" % (layer, part, which)


def init_adapters(cfg, rng, rank=RANK, dtype=np.float32, b_std=0.0):
    """``A`` (d_model, rank) random and ``B`` (rank, d_model) zero for every
    adapted part of every layer.  ``b_std`` > 0 makes ``B`` random too (only
    for gradient checks: at ``B = 0`` the gradient of ``A`` is exactly zero,
    which would check nothing)."""
    d = cfg.d_model
    L = {}
    for li in range(cfg.n_layer):
        for part in PARTS:
            A = rng.normal(0, A_STD / np.sqrt(d), (d, rank))
            L[adapter_key(li, part, "A")] = A.astype(dtype)
            L[adapter_key(li, part, "B")] = (rng.normal(0, b_std, (rank, d)) if b_std > 0
                                             else np.zeros((rank, d))).astype(dtype)
    return L


def n_adapter_parameters(L):
    """Trainable numbers in the adapters ``L``."""
    return n_parameters(L)


def delta(L, layer, part, alpha=ALPHA):
    """The effective update ``(alpha / r) A B`` of one adapted part, ``(d_model, d_model)``."""
    A, B = L[adapter_key(layer, part, "A")], L[adapter_key(layer, part, "B")]
    return (alpha / A.shape[1]) * (A @ B)


def _part_slice(cfg, part):
    """Columns of ``c_attn`` that ``part`` occupies (q, k, v thirds)."""
    i = "qkv".index(part)
    return slice(i * cfg.d_model, (i + 1) * cfg.d_model)


def merge(W, L, cfg=GPT2, alpha=ALPHA):
    """A copy of the weights ``W`` with every adapter folded into its
    ``c_attn``: ``W' = W + (alpha / r) A B`` on the adapted columns.  The
    merged model has GPT-2's shape and cost and gives the adapted outputs."""
    out = dict(W)
    for li in range(cfg.n_layer):
        key = "h.%d.attn.c_attn.weight" % li
        Wc = np.array(W[key])
        for part in PARTS:
            if adapter_key(li, part, "A") in L:
                Wc[:, _part_slice(cfg, part)] += delta(L, li, part, alpha).astype(Wc.dtype)
        out[key] = Wc
    return out


# ------------------------------------------------------------------ forward


def layer_norm(x, g, b):
    """LayerNorm of ``x`` over its last axis; returns ``(y, xhat, rstd)``,
    the normalised ``xhat`` and ``1 / sqrt(var + eps)`` kept for the backward."""
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(v + LN_EPS)
    xhat = (x - m) * rstd
    return xhat * g + b, xhat, rstd


def layer_norm_backward(dy, xhat, rstd, g):
    """``(dx, dg, db)`` of LayerNorm given the upstream ``dy`` and the saved
    ``xhat``, ``rstd`` (the mean of ``xhat`` is 0 and of ``xhat^2`` is 1,
    which is what makes the two centring terms)."""
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    axes = tuple(range(dy.ndim - 1))
    return dx, (dy * xhat).sum(axes), dy.sum(axes)


def gelu(x):
    """GPT-2's tanh approximation of GELU; returns ``(y, tanh)`` (tanh kept for the backward)."""
    th = np.tanh(GELU_C * (x + GELU_K * x ** 3))
    return 0.5 * x * (1 + th), th


def gelu_backward(dy, x, th):
    """``dy * gelu'(x)`` with the saved ``tanh``."""
    return dy * (0.5 * (1 + th) + 0.5 * x * (1 - th ** 2) * GELU_C * (1 + 3 * GELU_K * x ** 2))


@dataclass
class Layer:
    """What one layer's forward pass keeps for the backward.  Rows are the
    ``N x T`` tokens of the batch flattened (one GEMM per weight matrix
    instead of ``N`` small ones); only the attention keeps ``(N, H, T, ...)``."""

    xhat1: np.ndarray       # (NT, d) LN_1's normalised input
    rstd1: np.ndarray       # (NT, 1)
    h: np.ndarray           # (NT, d) LN_1's output: the input of c_attn and the adapters
    hA: dict                # part -> (NT, r): h @ A, the adapters' inner activations
    q: np.ndarray           # (N, H, T, d_head)
    k: np.ndarray
    v: np.ndarray
    a: np.ndarray           # (N, H, T, T) attention weights
    cat: np.ndarray         # (NT, d) the heads' outputs side by side
    xhat2: np.ndarray       # LN_2
    rstd2: np.ndarray
    h2: np.ndarray          # (NT, d) LN_2's output: the MLP's input
    u: np.ndarray           # (NT, 4d) the MLP's pre-activation
    th: np.ndarray          # tanh inside the GELU
    hid: np.ndarray         # (NT, 4d) after the GELU


@dataclass
class Trace:
    """One forward pass over a batch (see :func:`forward`)."""

    ids: np.ndarray         # (N, T) token ids, right-padded
    layers: list            # [Layer]
    xhat_f: np.ndarray      # final LayerNorm, (NT, d)
    rstd_f: np.ndarray
    hf: np.ndarray          # (NT, d) its output
    logits: np.ndarray      # (N, T, V)


def _attend(q, k, v, mask):
    """Causal softmax attention of one batch: ``(o, a)`` with ``a`` the weights."""
    s = q @ k.swapaxes(-1, -2) / np.sqrt(q.shape[-1]).astype(q.dtype)
    s = np.where(mask, s.dtype.type(-1e10), s)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    return a @ v, a


def _heads(t, N, T, H, dh):
    """``(NT, d)`` -> ``(N, H, T, dh)``."""
    return t.reshape(N, T, H, dh).swapaxes(1, 2)


def _unheads(t, N, T, d):
    """``(N, H, T, dh)`` -> ``(NT, d)``."""
    return t.swapaxes(1, 2).reshape(N * T, d)


def forward(W, ids, L=None, cfg=GPT2, alpha=ALPHA):
    """GPT-2's forward pass over ``ids`` ``(N, T)`` (right-padded token ids)
    with the adapters ``L`` (None: the base model), in the weights' dtype.
    Returns a :class:`Trace`."""
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None]
    N, T = ids.shape
    d, H, dh = cfg.d_model, cfg.n_head, cfg.d_head
    x = (W["wte.weight"][ids] + W["wpe.weight"][:T][None]).reshape(N * T, d)
    mask = np.triu(np.ones((T, T), bool), 1)
    layers = []
    for li in range(cfg.n_layer):
        p = "h.%d." % li
        h, xhat1, rstd1 = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        hA = {}
        if L is not None:
            for part in PARTS:
                A, B = L[adapter_key(li, part, "A")], L[adapter_key(li, part, "B")]
                hA[part] = h @ A
                qkv[:, _part_slice(cfg, part)] += (alpha / A.shape[1]) * (hA[part] @ B)
        q, k, v = (_heads(t, N, T, H, dh) for t in np.split(qkv, 3, axis=-1))
        o, a = _attend(q, k, v, mask)
        cat = _unheads(o, N, T, d)
        x = x + cat @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h2, xhat2, rstd2 = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        u = h2 @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"]
        hid, th = gelu(u)
        x = x + hid @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        layers.append(Layer(xhat1, rstd1, h, hA, q, k, v, a, cat, xhat2, rstd2, h2, u, th, hid))
    hf, xhat_f, rstd_f = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"])
    logits = (hf @ W["wte.weight"].T).reshape(N, T, -1)
    return Trace(ids, layers, xhat_f, rstd_f, hf, logits)


# ------------------------------------------------------------------ the loss


def log_softmax(z):
    """Log-probabilities from logits ``z`` along the last axis (in float64)."""
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def lm_loss(logits, ids, weight):
    """Mean cross-entropy of predicting ``ids[:, t + 1]`` from position ``t``
    over the targets with ``weight[:, t + 1] > 0`` (0 marks padding).
    Returns ``(loss, dlogits)`` with ``dlogits`` in the logits' dtype."""
    lp = log_softmax(logits[:, :-1])                         # (N, T-1, V)
    tgt = ids[:, 1:]
    w = np.asarray(weight[:, 1:], np.float64)
    n = w.sum()
    picked = np.take_along_axis(lp, tgt[..., None], -1)[..., 0]
    loss = float(-(picked * w).sum() / n)
    dz = np.exp(lp)
    np.put_along_axis(dz, tgt[..., None], np.take_along_axis(dz, tgt[..., None], -1) - 1, -1)
    dz *= (w / n)[..., None]
    dlogits = np.zeros_like(logits)
    dlogits[:, :-1] = dz.astype(logits.dtype)
    return loss, dlogits


def pad_batch(seqs, pad_id=0):
    """``(ids, weight)`` ``(N, T)`` for token sequences of different lengths:
    right-padded with ``pad_id``, ``weight`` 1 on real tokens and 0 on padding."""
    T = max(len(s) for s in seqs)
    ids = np.full((len(seqs), T), pad_id, np.int64)
    w = np.zeros((len(seqs), T), np.float32)
    for n, s in enumerate(seqs):
        ids[n, :len(s)] = s
        w[n, :len(s)] = 1.0
    return ids, w


# ------------------------------------------------------------------ backward


def backward(W, tr, dlogits, L, cfg=GPT2, alpha=ALPHA, frozen=False):
    """Gradients of a loss with ``dlogits`` ``(N, T, V)`` with respect to the
    adapters ``L`` (same keys), and with ``frozen=True`` also with respect to
    every base weight of ``W`` (same keys; ``wte`` collects both its uses).
    The reverse of :func:`forward`, layer by layer."""
    N, T = tr.ids.shape
    d, H, dh = cfg.d_model, cfg.n_head, cfg.d_head
    scale = np.sqrt(dh).astype(dlogits.dtype)
    gL = {}
    gW = {}
    dlogits = dlogits.reshape(N * T, -1)
    # unembedding and the final LayerNorm
    dhf = dlogits @ W["wte.weight"]
    if frozen:
        gW["wte.weight"] = dlogits.T @ tr.hf
    dx, dg, db = layer_norm_backward(dhf, tr.xhat_f, tr.rstd_f, W["ln_f.weight"])
    if frozen:
        gW["ln_f.weight"], gW["ln_f.bias"] = dg, db
    for li in reversed(range(cfg.n_layer)):
        p = "h.%d." % li
        ly = tr.layers[li]
        # the MLP branch: x += gelu(h2 Wfc + bfc) Wp + bp
        dhid = dx @ W[p + "mlp.c_proj.weight"].T
        du = gelu_backward(dhid, ly.u, ly.th)
        dh2 = du @ W[p + "mlp.c_fc.weight"].T
        if frozen:
            gW[p + "mlp.c_proj.weight"] = ly.hid.T @ dx
            gW[p + "mlp.c_proj.bias"] = dx.sum(0)
            gW[p + "mlp.c_fc.weight"] = ly.h2.T @ du
            gW[p + "mlp.c_fc.bias"] = du.sum(0)
        dxn, dg, db = layer_norm_backward(dh2, ly.xhat2, ly.rstd2, W[p + "ln_2.weight"])
        if frozen:
            gW[p + "ln_2.weight"], gW[p + "ln_2.bias"] = dg, db
        dx = dx + dxn
        # the attention branch: x += cat Wo + bo
        dcat = dx @ W[p + "attn.c_proj.weight"].T
        if frozen:
            gW[p + "attn.c_proj.weight"] = ly.cat.T @ dx
            gW[p + "attn.c_proj.bias"] = dx.sum(0)
        do = _heads(dcat, N, T, H, dh)
        da = do @ ly.v.swapaxes(-1, -2)
        dv = ly.a.swapaxes(-1, -2) @ do
        ds = ly.a * (da - (da * ly.a).sum(-1, keepdims=True))       # softmax; masked cells give 0
        dq = ds @ ly.k / scale
        dk = ds.swapaxes(-1, -2) @ ly.q / scale
        dqkv = np.concatenate([_unheads(t, N, T, d) for t in (dq, dk, dv)], -1)
        dh1 = dqkv @ W[p + "attn.c_attn.weight"].T
        if frozen:
            gW[p + "attn.c_attn.weight"] = ly.h.T @ dqkv
            gW[p + "attn.c_attn.bias"] = dqkv.sum(0)
        for part in PARTS:
            A, B = L[adapter_key(li, part, "A")], L[adapter_key(li, part, "B")]
            s = alpha / A.shape[1]
            dpart = dqkv[:, _part_slice(cfg, part)]
            gL[adapter_key(li, part, "B")] = s * (ly.hA[part].T @ dpart)
            dhA = s * (dpart @ B.T)                                    # (NT, r)
            gL[adapter_key(li, part, "A")] = ly.h.T @ dhA
            dh1 = dh1 + dhA @ A.T
        dxn, dg, db = layer_norm_backward(dh1, ly.xhat1, ly.rstd1, W[p + "ln_1.weight"])
        if frozen:
            gW[p + "ln_1.weight"], gW[p + "ln_1.bias"] = dg, db
        dx = dx + dxn
    if frozen:
        # the embeddings: x0 = wte[ids] + wpe[:T]
        np.add.at(gW["wte.weight"], tr.ids.reshape(-1), dx)
        gW["wpe.weight"] = np.zeros_like(W["wpe.weight"])
        gW["wpe.weight"][:T] = dx.reshape(N, T, d).sum(0)
        return gL, gW
    return gL


# ------------------------------------------------------------------ reading the model


def next_probs(W, ids, L=None, cfg=GPT2):
    """The distribution over the next token after ``ids`` (a list of int)."""
    tr = forward(W, np.asarray(ids)[None], L, cfg)
    return np.exp(log_softmax(tr.logits[0, -1]))


def singular_values(M, n=N_SINGULAR):
    """The ``n`` largest singular values of ``M``."""
    return np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)[:n]


# ------------------------------------------------------------------ training


def settings():
    """Everything the recorded run depends on (the cache key)."""
    return dict(rank=RANK, alpha=ALPHA, parts=PARTS, a_std=A_STD, sentences=SENTENCES,
                prompt=PROMPT, target=TARGET, controls=CONTROLS, steps=STEPS, lr=LR, seed=SEED,
                top_k=TOP_K, shown_layer=SHOWN_LAYER, n_singular=N_SINGULAR)


def _probe(W, ids, L, tid):
    """``(p_target, top ids, top probabilities)`` after ``ids``."""
    p = next_probs(W, ids, L)
    top = np.argsort(-p)[:TOP_K]
    return float(p[tid]), top, p[top]


def _record(W, L, prompt_ids, tid, out):
    """Append the prompt's read-out and the shown layer's adapter to ``out``."""
    pt, top, pp = _probe(W, prompt_ids, L, tid)
    out["p_target"].append(pt)
    out["top_ids"].append(top)
    out["top_p"].append(pp)
    out["B"].append(L[adapter_key(SHOWN_LAYER, "q", "B")].copy())
    out["sv"].append(singular_values(delta(L, SHOWN_LAYER, "q")))


def build():
    """Train the adapters and record the run (see the module docstring)."""
    W = P.weights()
    rng = np.random.default_rng(SEED)
    L = init_adapters(GPT2, rng)
    ids, weight = pad_batch([P.encode(s) for s in SENTENCES])
    prompt_ids = P.encode(PROMPT)
    tid = P.token_id(TARGET)
    controls = [P.encode(c) for c in CONTROLS]
    opt = Adam(L, LR)
    out = dict(loss=[], p_target=[], top_ids=[], top_p=[], B=[], sv=[], seconds=[])
    before = [_probe(W, c, None, tid) for c in controls]
    _record(W, L, prompt_ids, tid, out)
    for _step in range(STEPS):
        t0 = time.perf_counter()
        tr = forward(W, ids, L)
        loss, dlogits = lm_loss(tr.logits, ids, weight)
        opt.step(L, backward(W, tr, dlogits, L))
        out["seconds"].append(time.perf_counter() - t0)
        out["loss"].append(loss)
        _record(W, L, prompt_ids, tid, out)
    tr = forward(W, ids, L)
    out["loss"].append(lm_loss(tr.logits, ids, weight)[0])
    after = [_probe(W, c, L, tid) for c in controls]
    # the merged model against the adapted one, on the prompt
    merged = merge(W, L)
    z_adapted = forward(W, np.asarray(prompt_ids)[None], L).logits[0, -1]
    z_merged = forward(merged, np.asarray(prompt_ids)[None]).logits[0, -1]
    arrays = {k: np.array(v) for k, v in out.items()}
    arrays["A_shown"] = L[adapter_key(SHOWN_LAYER, "q", "A")]
    for key, value in L.items():
        arrays["adapter." + key] = value
    arrays["W_shown"] = W["h.%d.attn.c_attn.weight" % SHOWN_LAYER][:, _part_slice(GPT2, "q")]
    arrays["control_top_ids"] = np.array([[b[1] for b in before], [a[1] for a in after]])
    arrays["control_top_p"] = np.array([[b[2] for b in before], [a[2] for a in after]])
    arrays["merge_max_abs"] = np.array(float(np.abs(z_adapted - z_merged).max()))
    arrays["merge_logit_scale"] = np.array(float(np.abs(z_adapted).max()))
    arrays["n_base"] = np.array(n_parameters(W))
    arrays["n_adapter"] = np.array(n_adapter_parameters(L))
    return arrays


@dataclass
class Run:
    """The recorded fine-tuning run, read from the cache."""

    loss: np.ndarray            # (STEPS + 1,) training loss before each step, and after the last
    p_target: np.ndarray        # (STEPS + 1,) p(" Avalon" | PROMPT) before each step and after
    top: list                   # per step: [(token string, p)], TOP_K most probable next tokens
    B: np.ndarray               # (STEPS + 1, r, d) the shown layer's B_q after each step
    sv: np.ndarray              # (STEPS + 1, N_SINGULAR) singular values of its update
    A: np.ndarray               # (d, r) the shown layer's A_q (fixed)
    W: np.ndarray               # (d, d) the shown layer's frozen W_Q
    controls: list              # [(prompt, top before, top after)]
    adapters: dict              # the final adapters, every layer and part
    seconds: np.ndarray         # (STEPS,) wall-clock seconds per step when the run was recorded
    merge_max_abs: float        # max |logit(adapted) - logit(merged)| on the prompt
    merge_logit_scale: float    # max |logit| there, for scale
    n_base: int
    n_adapter: int

    @property
    def steps(self):
        """Number of training steps."""
        return len(self.loss) - 1

    @property
    def delta(self):
        """``(STEPS + 1, d, d)`` the shown update ``(alpha / r) A B`` after each step."""
        return (ALPHA / RANK) * np.einsum("dr,srk->sdk", self.A, self.B)

    @property
    def ratio(self):
        """The trainable fraction ``n_adapter / n_base``."""
        return self.n_adapter / self.n_base


def run(cache_dir=None):
    """The recorded run as a :class:`Run` (trained on first use, then cached)."""
    d = cached_npz("gpt2-lora", settings(), build, cache_dir)

    def tops(ids, ps):
        return [(P.decode([int(i)]), float(p)) for i, p in zip(ids, ps)]

    top = [tops(i, p) for i, p in zip(d["top_ids"], d["top_p"])]
    controls = [(c, tops(d["control_top_ids"][0, n], d["control_top_p"][0, n]),
                 tops(d["control_top_ids"][1, n], d["control_top_p"][1, n]))
                for n, c in enumerate(CONTROLS)]
    adapters = {k[len("adapter."):]: v for k, v in d.items() if k.startswith("adapter.")}
    return Run(d["loss"], d["p_target"], top, d["B"], d["sv"], d["A_shown"], d["W_shown"],
               controls, adapters, d["seconds"], float(d["merge_max_abs"]),
               float(d["merge_logit_scale"]), int(d["n_base"]), int(d["n_adapter"]))
