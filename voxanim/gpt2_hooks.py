"""Instrumented GPT-2 small inference, including incremental KV caching.

All hooks are explicit call arguments: the released weight arrays are never
mutated. Residual patches index the stream entering a block (12 is the final
stream); head ablations replace the weighted values before output projection.
"""

from __future__ import annotations

import numpy as np

from . import gpt2


def forward(
    ids,
    *,
    n_layers=12,
    cache=None,
    patches=None,
    ablate=None,
    weight_edits=None,
    record=False,
    position_ids=None,
):
    """Return logits and new cache, optionally every tapped activation.

    ``cache`` contains ``(key,value)`` pairs in ``(heads,tokens,width)`` order.
    A supplied cache is read only. ``position_ids`` preserves absolute positions
    after cache eviction; attention causality still uses physical cache order.
    ``patches`` keys are ("resid", layer, token)
    or ("mlp", layer, token), where token indexes this call's new tokens.
    ``weight_edits`` overrides individual named arrays without changing GPT-2.
    """
    if not 1 <= n_layers <= gpt2.N_LAYER:
        raise ValueError("n_layers must be in [1,12]")
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("ids must be a nonempty vector")
    if cache is not None and len(cache) != n_layers:
        raise ValueError("cache must contain one pair per layer")
    offset = 0 if cache is None else cache[0][0].shape[1]
    positions = (
        np.arange(offset, offset + len(ids))
        if position_ids is None
        else np.asarray(position_ids, dtype=np.int64)
    )
    if positions.shape != ids.shape or np.any(positions < 0) or np.any(positions >= gpt2.N_CTX):
        raise ValueError("position_ids must match tokens within GPT-2's positional table")
    weights = gpt2.weights()
    W = weights if not weight_edits else dict(weights, **weight_edits)
    patches, ablate = patches or {}, ablate or {}
    n = len(ids)
    x = W["wte.weight"][ids] + W["wpe.weight"][positions]
    result = {k: [] for k in ("resid", "attn", "heads", "mlp_hidden", "mlp_out", "attn_out")}
    new_cache = []
    for layer in range(n_layers):
        for (kind, li, ti), value in patches.items():
            if kind == "resid" and li == layer:
                x[ti] = value
        if record:
            result["resid"].append(x.copy())
        prefix = "h.%d." % layer
        h = gpt2.layer_norm(x, W[prefix + "ln_1.weight"], W[prefix + "ln_1.bias"])
        qkv = h @ W[prefix + "attn.c_attn.weight"] + W[prefix + "attn.c_attn.bias"]
        q, k, v = [a.reshape(n, 12, 64).transpose(1, 0, 2) for a in np.split(qkv, 3, axis=1)]
        if cache is not None:
            k = np.concatenate([cache[layer][0], k], axis=1)
            v = np.concatenate([cache[layer][1], v], axis=1)
        new_cache.append((k, v))
        score = q @ k.transpose(0, 2, 1) / 8
        mask = np.arange(k.shape[1])[None, :] > (offset + np.arange(n))[:, None]
        a = gpt2.softmax(np.where(mask[None], -np.inf, score)).astype(x.dtype)
        heads = a @ v
        for (li, hi), replacement in ablate.items():
            if li == layer:
                heads[hi] = replacement
        ao = heads.transpose(1, 0, 2).reshape(n, 768) @ W[prefix + "attn.c_proj.weight"]
        ao += W[prefix + "attn.c_proj.bias"]
        x = x + ao
        h = gpt2.layer_norm(x, W[prefix + "ln_2.weight"], W[prefix + "ln_2.bias"])
        hidden = gpt2._gelu(h @ W[prefix + "mlp.c_fc.weight"] + W[prefix + "mlp.c_fc.bias"])
        mo = hidden @ W[prefix + "mlp.c_proj.weight"] + W[prefix + "mlp.c_proj.bias"]
        for (kind, li, ti), value in patches.items():
            if kind == "mlp" and li == layer:
                mo[ti] = value
        x = x + mo
        if record:
            for key, value in [
                ("attn", a),
                ("heads", heads),
                ("mlp_hidden", hidden),
                ("mlp_out", mo),
                ("attn_out", ao),
            ]:
                result[key].append(value.copy())
    for (kind, li, ti), value in patches.items():
        if kind == "resid" and li == n_layers:
            x[ti] = value
    if record:
        result["resid"].append(x.copy())
    result["logits"] = gpt2.layer_norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T
    result["cache"] = new_cache
    if record:
        for key in ("resid", "attn", "heads", "mlp_hidden", "mlp_out", "attn_out"):
            result[key] = np.asarray(result[key])
    return result


def run(text, **kwargs):
    """Encode text with the pinned GPT-2 tokenizer and instrument its forward pass."""
    return forward(gpt2.Tokenizer().encode(text), **kwargs)
