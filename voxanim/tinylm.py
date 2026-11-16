"""Small configurable transformers, trained entirely in NumPy.

All architectural choices use the same reverse-mode primitives. A configuration
is part of every cache key; finite-difference tests cover each branch. Datasets
are supplied explicitly, so a task can compare exactly the same train/test split.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz, settings_key, sha256_of
from .optim import AdamW


def array_identity(value):
    """Identify the shape, dtype and bytes, not just a flattened byte stream."""
    if value is None:
        return None
    value = np.ascontiguousarray(value)
    description = {"shape": list(value.shape), "dtype": str(value.dtype)}
    h = hashlib.sha256(json.dumps(description, sort_keys=True).encode())
    h.update(value.tobytes())
    return {**description, "sha256": h.hexdigest()}


def parameter_hash(parameters):
    """Canonical content identity for named parameter arrays."""
    identities = {key: array_identity(value) for key, value in sorted(parameters.items())}
    return hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()


def training_sources(*extra):
    """Code dependencies recorded by every training run, including the caller when given."""
    root = Path(__file__).parent
    return {name: sha256_of(root / name) for name in ("tinylm.py", "optim.py", "cache.py", *extra)}


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


class Var:
    """An array and its local vector-Jacobian products (training only)."""

    def __init__(self, value, parents=()):
        self.v = np.asarray(value, dtype=np.float64)
        self.parents = parents
        self.g = None

    def __add__(self, other):
        b = variable(other)
        return Var(
            self.v + b.v,
            (
                (self, lambda g: _unbroadcast(g, self.v.shape)),
                (b, lambda g: _unbroadcast(g, b.v.shape)),
            ),
        )

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -variable(other)

    def __mul__(self, other):
        b = variable(other)
        return Var(
            self.v * b.v,
            (
                (self, lambda g: _unbroadcast(g * b.v, self.v.shape)),
                (b, lambda g: _unbroadcast(g * self.v, b.v.shape)),
            ),
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * variable(other) ** -1

    def __rtruediv__(self, other):
        return variable(other) * self**-1

    def __pow__(self, n):
        return Var(self.v**n, ((self, lambda g: g * n * self.v ** (n - 1)),))

    def __matmul__(self, other):
        b = variable(other)
        return Var(
            self.v @ b.v,
            (
                (self, lambda g: _unbroadcast(g @ b.v.swapaxes(-1, -2), self.v.shape)),
                (b, lambda g: _unbroadcast(self.v.swapaxes(-1, -2) @ g, b.v.shape)),
            ),
        )

    def __getitem__(self, index):
        def back(g):
            out = np.zeros_like(self.v)
            np.add.at(out, index, g)
            return out

        return Var(self.v[index], ((self, back),))

    def reshape(self, *shape):
        return Var(self.v.reshape(*shape), ((self, lambda g: g.reshape(self.v.shape)),))

    def transpose(self, *axes):
        inv = np.argsort(axes)
        return Var(self.v.transpose(axes), ((self, lambda g: g.transpose(inv)),))

    def sum(self, axis=None, keepdims=False):
        def back(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return np.broadcast_to(g, self.v.shape)

        return Var(self.v.sum(axis=axis, keepdims=keepdims), ((self, back),))

    def mean(self, axis=None, keepdims=False):
        n = self.v.size if axis is None else self.v.shape[axis]
        return self.sum(axis, keepdims) / n

    def exp(self):
        e = np.exp(self.v)
        return Var(e, ((self, lambda g: g * e),))

    def log(self):
        return Var(np.log(self.v), ((self, lambda g: g / self.v),))

    def tanh(self):
        t = np.tanh(self.v)
        return Var(t, ((self, lambda g: g * (1 - t * t)),))

    def backward(self):
        order, seen = [], set()

        def visit(node):
            if id(node) not in seen:
                seen.add(id(node))
                for p, _ in node.parents:
                    visit(p)
                order.append(node)

        visit(self)
        self.g = np.ones_like(self.v)
        for node in reversed(order):
            if node.g is None:
                continue
            for p, pullback in node.parents:
                g = pullback(node.g)
                p.g = g if p.g is None else p.g + g


def variable(x):
    return x if isinstance(x, Var) else Var(x)


def softmax(x):
    x = variable(x)
    e = (x - x.v.max(-1, keepdims=True)).exp()
    return e / e.sum(-1, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + (np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)).tanh())


@dataclass(frozen=True)
class Config:
    vocab: int = 8
    width: int = 24
    heads: int = 4
    kv_heads: int = 4
    layers: int = 1
    ff: int = 48
    mask: str = "causal"
    window: int = 4
    prefix: int = 4
    norm: str = "layer"
    placement: str = "pre"
    activation: str = "gelu"
    parallel: bool = False
    tied: bool = False
    attention: str = "softmax"
    cross_attention: bool = False
    max_len: int = 256
    scale_scores: bool = True

    def __post_init__(self):
        if self.width % self.heads or self.heads % self.kv_heads:
            raise ValueError("Width must divide into heads and heads into KV groups")
        if min(self.width, self.heads, self.kv_heads, self.layers, self.ff, self.vocab) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if self.mask not in {"causal", "bidirectional", "prefix", "window", "sparse"}:
            raise ValueError("Unknown mask")
        if self.norm not in {"layer", "rms"} or self.placement not in {"pre", "post"}:
            raise ValueError("Unknown normalization")
        if self.activation not in {"gelu", "swiglu"}:
            raise ValueError("Unknown activation")
        if self.attention not in {"softmax", "linear"}:
            raise ValueError("Unknown attention")


def attention_mask(n, config):
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    if config.mask == "bidirectional":
        return np.ones((n, n), bool)
    if config.mask == "prefix":
        return (j <= i) | ((i < config.prefix) & (j < config.prefix))
    if config.mask == "window":
        return (j <= i) & (i - j < config.window)
    if config.mask == "sparse":
        return (j <= i) & ((i - j < config.window) | (j % config.window == 0))
    return j <= i


def init(config, seed=0):
    c, rng = config, np.random.default_rng(seed)
    d, dk = c.width, c.width // c.heads * c.kv_heads
    P = {
        "E": rng.normal(0, 0.25, (c.vocab, d)),
        "pos": rng.normal(0, 0.1, (c.max_len, d)),
        "final.g": np.ones(d),
        "final.b": np.zeros(d),
    }
    if not c.tied:
        P["U"] = rng.normal(0, 1 / np.sqrt(d), (d, c.vocab))
    for l in range(c.layers):
        p = "l%d." % l
        for name, a, b in (
            ("Q", d, d),
            ("K", d, dk),
            ("V", d, dk),
            ("O", d, d),
            ("W1", d, c.ff),
            ("W2", c.ff, d),
        ):
            P[p + name] = rng.normal(0, 1 / np.sqrt(a), (a, b))
        if c.activation == "swiglu":
            P[p + "gate"] = rng.normal(0, 1 / np.sqrt(d), (d, c.ff))
        for n in ("n1", "n2", "nc") if c.cross_attention else ("n1", "n2"):
            P[p + n + ".g"], P[p + n + ".b"] = np.ones(d), np.zeros(d)
        if c.cross_attention:
            for name, b in (("Q", d), ("K", dk), ("V", dk), ("O", d)):
                P[p + "cross." + name] = rng.normal(0, 1 / np.sqrt(d), (d, b))
    return P


def _norm(x, P, key, c):
    z = x - x.mean(-1, True) if c.norm == "layer" else x
    return z / (z**2).mean(-1, True).__add__(1e-5) ** 0.5 * P[key + ".g"] + P[key + ".b"]


def _attention(x, source, P, p, c, mask, records):
    b, n, d = x.v.shape
    m = source.v.shape[1]
    dh, h, kh = d // c.heads, c.heads, c.kv_heads
    q = (x @ P[p + "Q"]).reshape(b, n, h, dh).transpose(0, 2, 1, 3)
    k = (source @ P[p + "K"]).reshape(b, m, kh, dh).transpose(0, 2, 1, 3)
    v = (source @ P[p + "V"]).reshape(b, m, kh, dh).transpose(0, 2, 1, 3)
    groups = np.arange(h) * kh // h
    k, v = k[:, groups], v[:, groups]
    if c.attention == "linear":

        def feature(z):
            positive = z.v > 0
            a = np.where(positive, z.v + 1, np.exp(np.minimum(z.v, 0)))
            return Var(a, ((z, lambda g: g * np.where(positive, 1, a)),))

        scores = feature(q) @ feature(k).transpose(0, 1, 3, 2)
        scores = scores * mask
        a = scores / (scores.sum(-1, True) + 1e-10)
    else:
        scores = q @ k.transpose(0, 1, 3, 2)
        if c.scale_scores:
            scores = scores / np.sqrt(dh)
        a = softmax(scores + np.where(mask, 0.0, -1e9))
    records.append(a.v.copy())
    out = (a @ v).transpose(0, 2, 1, 3).reshape(b, n, d)
    return out @ P[p + "O"]


def _forward(P, x, c, context=None, weights_mask=None):
    _b, n = x.shape
    h = P["E"][x] + P["pos"][np.arange(n)]
    mask = attention_mask(n, c) if weights_mask is None else weights_mask
    records, residual = [], [h.v.copy()]
    source = None
    if c.cross_attention:
        if context is None:
            raise ValueError("Cross attention requires encoder context token IDs")
        source = P["E"][context] + P["pos"][np.arange(context.shape[1])]
    for l in range(c.layers):
        p = "l%d." % l
        hn = _norm(h, P, p + "n1", c) if c.placement == "pre" else h
        a = _attention(hn, hn, P, p, c, mask, records)
        after = h + a
        if c.placement == "post":
            after = _norm(after, P, p + "n1", c)
        if source is not None:
            z = _norm(after, P, p + "nc", c)
            cross_mask = np.ones((n, context.shape[1]), bool)
            after = after + _attention(z, source, P, p + "cross.", c, cross_mask, records)
        z = h if c.parallel else after
        if c.placement == "pre":
            z = _norm(z, P, p + "n2", c)
        z1 = z @ P[p + "W1"]
        if c.activation == "gelu":
            mlp = gelu(z1)
        else:
            sigmoid = 1 / (1 + (-z1).exp())
            mlp = z1 * sigmoid * (z @ P[p + "gate"])
        h = after + mlp @ P[p + "W2"]
        if c.placement == "post":
            h = _norm(h, P, p + "n2", c)
        residual.append(h.v.copy())
    h = _norm(h, P, "final", c)
    U = P["E"].transpose(1, 0) if c.tied else P["U"]
    return h @ U, {"attention": records, "residual": residual}


def forward(P, x, config, context=None, keep=False, mask=None):
    logits, records = _forward(
        {k: variable(v) for k, v in P.items()}, np.asarray(x, int), config, context, mask
    )
    return (logits.v, records) if keep else logits.v


@dataclass(frozen=True)
class KVCache:
    """Per-layer projected keys and values, plus their absolute position count."""

    layers: tuple
    length: int


def decode(P, tokens, config, cache=None):
    """Incremental inference with actual per-layer KV reuse.

    Causal, local and strided masks preserve their absolute token positions.
    Bidirectional/prefix encoders and cross attention must use ``forward``.
    The input cache is never modified. Returns ``(new_token_logits, new_cache)``.
    """
    c = config
    if c.mask not in {"causal", "window", "sparse"} or c.cross_attention:
        raise ValueError("Incremental decoding requires a causal self-attention model")
    x = np.asarray(tokens, int)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("tokens must be a nonempty (batch, tokens) matrix")
    offset = 0 if cache is None else cache.length
    batch, n = x.shape
    if offset + n > c.max_len:
        raise ValueError("Decoding exceeds the learned position table")
    if cache is not None and len(cache.layers) != c.layers:
        raise ValueError("Cache layer count does not match configuration")
    h = P["E"][x] + P["pos"][offset : offset + n]
    query_pos = np.arange(offset, offset + n)[:, None]
    key_pos = np.arange(offset + n)[None, :]
    mask = key_pos <= query_pos
    if c.mask == "window":
        mask &= query_pos - key_pos < c.window
    elif c.mask == "sparse":
        mask &= (query_pos - key_pos < c.window) | (key_pos % c.window == 0)

    def norm(z, prefix):
        z = z - z.mean(-1, keepdims=True) if c.norm == "layer" else z
        return (
            z / np.sqrt((z * z).mean(-1, keepdims=True) + 1e-5) * P[prefix + ".g"]
            + P[prefix + ".b"]
        )

    new = []
    groups = np.arange(c.heads) * c.kv_heads // c.heads
    dh = c.width // c.heads
    for layer in range(c.layers):
        p = "l%d." % layer
        z = norm(h, p + "n1") if c.placement == "pre" else h
        q = (z @ P[p + "Q"]).reshape(batch, n, c.heads, dh).transpose(0, 2, 1, 3)
        k = (z @ P[p + "K"]).reshape(batch, n, c.kv_heads, dh).transpose(0, 2, 1, 3)
        v = (z @ P[p + "V"]).reshape(batch, n, c.kv_heads, dh).transpose(0, 2, 1, 3)
        if cache is not None:
            k = np.concatenate([cache.layers[layer][0], k], axis=2)
            v = np.concatenate([cache.layers[layer][1], v], axis=2)
        new.append((k, v))
        if c.attention == "linear":

            def phi(a):
                return np.where(a > 0, a + 1, np.exp(np.minimum(a, 0)))

            weights = phi(q) @ phi(k[:, groups]).swapaxes(-1, -2) * mask
            weights /= weights.sum(-1, keepdims=True) + 1e-10
        else:
            scores = q @ k[:, groups].swapaxes(-1, -2)
            if c.scale_scores:
                scores /= np.sqrt(dh)
            scores = np.where(mask, scores, -1e9)
            weights = np.exp(scores - scores.max(-1, keepdims=True))
            weights /= weights.sum(-1, keepdims=True)
        out = (weights @ v[:, groups]).transpose(0, 2, 1, 3).reshape(batch, n, c.width)
        after = h + out @ P[p + "O"]
        if c.placement == "post":
            after = norm(after, p + "n1")
        z = h if c.parallel else after
        if c.placement == "pre":
            z = norm(z, p + "n2")
        v1 = z @ P[p + "W1"]
        if c.activation == "gelu":
            mlp = 0.5 * v1 * (1 + np.tanh(np.sqrt(2 / np.pi) * (v1 + 0.044715 * v1**3)))
        else:
            mlp = v1 / (1 + np.exp(-v1)) * (z @ P[p + "gate"])
        h = after + mlp @ P[p + "W2"]
        if c.placement == "post":
            h = norm(h, p + "n2")
    out = norm(h, "final") @ (P["E"].T if c.tied else P["U"])
    return out, KVCache(tuple(new), offset + n)


def loss_grads(P, x, y, config, weights=None, context=None, mask=None):
    params = {k: Var(v) for k, v in P.items()}
    logits, _ = _forward(params, np.asarray(x, int), config, context, mask)
    shifted = logits - logits.v.max(-1, keepdims=True)
    logp = shifted - shifted.exp().sum(-1, True).log()
    y = np.asarray(y, int)
    selected = logp[np.arange(y.shape[0])[:, None], np.arange(y.shape[1]), y]
    w = np.ones(y.shape) if weights is None else np.asarray(weights, float)
    loss = -(selected * w).sum() / max(float(w.sum()), 1)
    loss.backward()
    return float(loss.v), {k: np.zeros_like(v.v) if v.g is None else v.g for k, v in params.items()}


def loss(P, x, y, config, weights=None, context=None, mask=None):
    z = forward(P, x, config, context=context, mask=mask)
    z = z - z.max(-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    y = np.asarray(y, int)
    chosen = -logp[np.arange(y.shape[0])[:, None], np.arange(y.shape[1]), y]
    w = np.ones(y.shape) if weights is None else np.asarray(weights, float)
    return float((chosen * w).sum() / max(float(w.sum()), 1))


def train(
    config,
    x,
    y,
    steps=160,
    lr=0.003,
    seed=0,
    weights=None,
    context=None,
    clip=1.0,
    warmup=0,
    weight_decay=0.0,
    name="study",
    batch=24,
    mask=None,
    provenance=None,
    cache_dir=None,
):
    """Fit actual token data and retain a verifiable final checkpoint and run record.

    Loss entries are minibatch measurements immediately before updates 1..steps.
    Dataset/split descriptions are supplied by callers; array identities are always
    recorded independently, including an explicit attention mask when present.
    """
    x, y = np.asarray(x, int), np.asarray(y, int)
    if x.ndim != 2 or x.shape != y.shape or not len(x):
        raise ValueError("Training inputs and targets must be equal nonempty 2-D arrays")
    if steps < 0 or batch < 1:
        raise ValueError("Training steps must be nonnegative and batch positive")
    weights = None if weights is None else np.asarray(weights, float)
    context = None if context is None else np.asarray(context, int)
    mask = None if mask is None else np.asarray(mask, bool)
    if weights is not None and weights.shape != x.shape:
        raise ValueError("Supervision weights must match token shape")
    if context is not None and (context.ndim != 2 or len(context) != len(x)):
        raise ValueError("Context must contain one sequence per training example")
    if mask is not None and mask.shape != (x.shape[1], x.shape[1]):
        raise ValueError("Attention mask must match sequence length")
    settings = dict(
        config=asdict(config),
        steps=steps,
        lr=lr,
        seed=seed,
        clip=clip,
        warmup=warmup,
        weight_decay=weight_decay,
        batch=batch,
        version=3,
        optimizer="AdamW",
        sources=training_sources(),
        numpy=np.__version__,
        platform=platform.platform(),
        provenance={} if provenance is None else provenance,
    )
    for key, value in (
        ("x", x),
        ("y", y),
        ("weights", weights),
        ("context", context),
        ("mask", mask),
    ):
        settings[key] = array_identity(value)

    def build():
        P, rng = init(config, seed), np.random.default_rng(seed + 100)
        initial_hash = parameter_hash(P)
        opt = AdamW(P, lr, weight_decay=weight_decay)
        losses, norms, clips = [], [], []
        for step in range(steps):
            ids = rng.choice(len(x), min(len(x), batch), replace=False)
            value, G = loss_grads(
                P,
                x[ids],
                y[ids],
                config,
                None if weights is None else weights[ids],
                None if context is None else context[ids],
                mask=mask,
            )
            norm = np.sqrt(sum(float((g * g).sum()) for g in G.values()))
            ratio = min(1.0, clip / max(norm, 1e-12)) if clip is not None else 1.0
            G = {k: v * ratio for k, v in G.items()}
            opt.step(P, G, lr=lr * min(1.0, (step + 1) / max(warmup, 1)))
            losses.append(value)
            norms.append(norm)
            clips.append(ratio)
        return {
            **{"P_" + k: v for k, v in P.items()},
            "loss": np.array(losses),
            "grad_norm": np.array(norms),
            "clip_ratio": np.array(clips),
            "run_manifest": np.array(
                json.dumps(
                    {
                        "evidence_kind": "trained_model_measurement",
                        "settings": settings,
                        "actual_updates": opt.t,
                        "initial_parameter_sha256": initial_hash,
                        "final_parameter_sha256": parameter_hash(P),
                        "minibatch_rng_seed": seed + 100,
                        "loss_semantics": "minibatch loss before each numbered update",
                        "loss_update_numbers": list(range(1, steps + 1)),
                        "checkpoint_step": opt.t,
                    },
                    sort_keys=True,
                )
            ),
        }

    cache_name = "tinylm-" + name
    a = cached_npz(cache_name, settings, build, cache_dir=cache_dir)
    parameters = {k[2:]: v for k, v in a.items() if k.startswith("P_")}
    manifest = json.loads(str(a["run_manifest"]))
    if manifest["final_parameter_sha256"] != parameter_hash(parameters):
        raise ValueError("Training checkpoint does not match its parameter hash")
    if manifest["actual_updates"] != steps or len(a["loss"]) != steps:
        raise ValueError("Training trace does not match the actual update count")
    artifact = Path(cache_dir or DEFAULT_CACHE) / (
        cache_name + "-" + settings_key(settings) + ".npz"
    )
    manifest["artifact"] = {"path": str(artifact.resolve()), "sha256": sha256_of(artifact)}
    return {
        "P": parameters,
        "provenance": manifest,
        **{k: a[k] for k in ("loss", "grad_norm", "clip_ratio")},
    }
