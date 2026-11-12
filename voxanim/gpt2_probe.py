"""Probing GPT-2 small: next-token distributions, sampling and attention.

:func:`voxanim.gpt2.run` records every intermediate of one forward pass for
the ``gpt2-attention`` animation; the behaviour animations (hallucination,
conditioning, calibration) need only what comes *out* of the model, for many
prompts and many sampled continuations.  :func:`forward` is the same
computation (the same weights, LayerNorm, GELU and causal mask) kept in
float32 and recording only the logits and, when asked, the last layer's
attention; :func:`next_probs` turns the last position into a distribution.

* :func:`entropy` -- in **bits**, so 1 bit is a coin toss and the 50,257-way
  uniform distribution is 15.6 bits,
* :func:`top_tokens` -- the ``k`` most probable next tokens as display strings,
* :func:`sample` -- a seeded continuation, one token at a time, each drawn
  from the model's own distribution (temperature 1: no sharpening, no
  truncation, so what comes out is what the model believes),
* :func:`first_word` -- where a sampled answer's first word ends,
* :func:`token_id` -- the id of a single-token string, or ``None``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import gpt2 as G


@lru_cache(maxsize=1)
def weights():
    """GPT-2's parameters as *aligned* in-memory float32 arrays.

    :func:`voxanim.gpt2.weights` returns views into the memory-mapped
    ``.safetensors`` file, whose data start 8 + header bytes in, so the arrays
    are not 4-byte aligned and numpy cannot hand them to BLAS: a matmul with
    the 50,257 x 768 embedding takes 0.8 s instead of 0.02 s.  Copying once
    (550 MB) makes a forward pass about 40 times faster, which the sampling
    and the 40-prompt calibration need.
    """
    return {k: np.array(v) for k, v in G.weights().items()}


@lru_cache(maxsize=1)
def tokenizer():
    """The GPT-2 tokenizer, built once (it reads the vocabulary from disk)."""
    return G.Tokenizer()


def encode(text):
    """Token ids of ``text`` (a list of int)."""
    return tokenizer().encode(text)


def decode(ids):
    """The text of token ``ids``."""
    return tokenizer().decode([int(i) for i in ids])


def token_id(s):
    """The id of ``s`` if it is exactly one GPT-2 token (``" Paris"`` is, ``" Ulaanbaatar"``
    is not), else ``None``.  Note the leading space: GPT-2 tokens carry it."""
    ids = encode(s)
    return int(ids[0]) if len(ids) == 1 else None


def forward(ids, attention=False):
    """Logits ``(T, N_VOCAB)`` for token ``ids`` in float32.

    With ``attention`` also returns the last layer's attention ``(N_HEAD, T,
    T)`` (softmax rows, causal).  The arithmetic is :func:`voxanim.gpt2.run`'s
    without the float64 promotion or the per-layer recording.
    """
    W = weights()
    ids = [int(i) for i in ids]
    T = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][:T]
    mask = np.triu(np.ones((T, T), bool), 1)
    attn = None
    for li in range(G.N_LAYER):
        p = "h.%d." % li
        h = G.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]
        q, k, v = (a.reshape(T, G.N_HEAD, G.D_HEAD).transpose(1, 0, 2)
                   for a in np.split(qkv, 3, axis=1))
        s = q @ k.transpose(0, 2, 1) / np.sqrt(G.D_HEAD)
        s = np.where(mask, -1e10, s)
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = (a @ v).transpose(1, 0, 2).reshape(T, G.D_MODEL)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h2 = G.layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        hid = G._gelu(h2 @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + hid @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        if li == G.N_LAYER - 1:
            attn = a.astype(np.float32)
    logits = G.layer_norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T
    logits = logits.astype(np.float32)
    return (logits, attn) if attention else logits


def softmax(z):
    """Probabilities from logits ``z`` (float64, so tiny probabilities survive)."""
    z = np.asarray(z, np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def next_probs(ids):
    """The model's distribution over the next token after ``ids``, ``(N_VOCAB,)``."""
    return softmax(forward(ids)[-1])


def entropy(p):
    """Shannon entropy of ``p`` in bits (zero entries contribute nothing)."""
    p = np.asarray(p, np.float64)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def top_tokens(p, k=5):
    """The ``k`` most probable tokens of ``p`` as ``[(display string, probability)]``,
    most probable first."""
    order = np.argsort(-p)[:k]
    return [(decode([i]), float(p[i])) for i in order]


def sample(ids, n_tokens, seed, temperature=1.0):
    """``n_tokens`` ids sampled one by one after ``ids``, from the model's own
    distribution (``temperature`` 1), by ``numpy.random.default_rng(seed)``.

    Returns ``(ids, probs)``: the sampled ids and the probability the model
    gave each one when it was drawn.
    """
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in ids]
    out, probs = [], []
    for _ in range(n_tokens):
        p = softmax(forward(ids)[-1] / temperature)
        nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
        out.append(nxt)
        probs.append(float(p[nxt]))
    return out, probs


def first_word(ids):
    """How many of the sampled ``ids`` make up the first word: the first token
    plus the alphabetic continuations that follow it without a leading space
    (``" Marse" + "ille"`` is one word, ``" Paris" + "."`` stops at one)."""
    n = 1
    for i in ids[1:]:
        s = decode([i])
        if s and not s.startswith(" ") and s.isalpha():
            n += 1
        else:
            break
    return n
