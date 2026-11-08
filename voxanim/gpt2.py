"""GPT-2 small (124M), run in plain numpy, recording everything the
animation draws.

The weights are OpenAI's released GPT-2 (``openai-community/gpt2`` on the
Hugging Face hub, MIT licence), downloaded once into ``build/.cache/gpt2``.
The ``.safetensors`` file is read as data: an 8-byte length, a JSON header
of names / dtypes / offsets, then the raw float32 arrays.

Architecture: 12 layers, 12 heads of width 64 (d_model 768), MLP 3072 with
GELU, pre-LayerNorm, learned positions, tied input / output embedding.

    x0 = wte[tokens] + wpe[positions]
    for each layer:  x = x + attn(ln_1(x));   x = x + mlp(ln_2(x))
    logits = ln_f(x) @ wte^T
"""

from __future__ import annotations

import itertools
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .scene import DEFAULT_CACHE

HUB = "https://huggingface.co/openai-community/gpt2/resolve/main/"
DIR = os.path.join(DEFAULT_CACHE, "gpt2")
FILES = ("config.json", "vocab.json", "merges.txt", "model.safetensors")
N_LAYER, N_HEAD, D_MODEL, D_HEAD = 12, 12, 768, 64


def fetch():
    os.makedirs(DIR, exist_ok=True)
    for f in FILES:
        path = os.path.join(DIR, f)
        if not os.path.exists(path):
            urllib.request.urlretrieve(HUB + f, path + ".part")
            os.replace(path + ".part", path)
    return DIR


# ------------------------------------------------------------------ weights


def load_safetensors(path):
    """``{name: array}`` from a .safetensors file (memory-mapped, read-only)."""
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    base = 8 + n
    dtypes = {"F32": np.float32, "F16": np.float16, "BF16": None}
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = dtypes[info["dtype"]]
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(info["shape"])
    return out


@lru_cache(maxsize=1)
def weights():
    W = load_safetensors(os.path.join(fetch(), "model.safetensors"))
    return {k.replace("transformer.", ""): np.asarray(v, np.float32) for k, v in W.items()}


# ------------------------------------------------------------------ tokenizer (byte-level BPE)


def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class Tokenizer:
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")

    def __init__(self):
        d = fetch()
        with open(os.path.join(d, "vocab.json"), encoding="utf-8") as f:
            self.enc = json.load(f)
        self.dec = {v: k for k, v in self.enc.items()}
        with open(os.path.join(d, "merges.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")[1:]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(lines) if m}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((a, b), 1e9), i) for i, (a, b) in
                     enumerate(itertools.pairwise(parts))]
            r, i = min(pairs)
            if r == 1e9:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return parts

    def encode(self, text):
        ids = []
        for piece in self.PAT.findall(text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.enc[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        return bytearray(self.u2b[c] for i in ids for c in self.dec[i]).decode("utf-8", "replace")


# ------------------------------------------------------------------ forward pass


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


@dataclass
class GPT2Trace:
    text: str
    ids: list
    tokens: list                  # display strings
    emb: np.ndarray               # (T, 768) token embedding
    pos: np.ndarray               # (T, 768) position embedding
    resid: list                   # residual stream: [x0, after layer 1, ..., after 12]
    q: np.ndarray                 # (L, H, T, 64)
    k: np.ndarray
    v: np.ndarray
    scores: np.ndarray            # (L, H, T, T)  q k^T / 8, before the mask
    attn: np.ndarray              # (L, H, T, T)  softmax with causal mask
    head_out: np.ndarray          # (L, H, T, 64)
    attn_out: np.ndarray          # (L, T, 768)   after W_o
    mlp_hidden: np.ndarray        # (L, T, 3072)  after GELU
    mlp_out: np.ndarray           # (L, T, 768)
    ln1: np.ndarray               # (L, T, 768)
    logits: np.ndarray            # (T, vocab) for every position
    top: list                     # [(token string, probability)] for the next token


def run(text="The cat sat on the"):
    W = weights()
    tok = Tokenizer()
    ids = tok.encode(text)
    T = len(ids)
    emb = W["wte.weight"][ids]
    pos = W["wpe.weight"][:T]
    x = emb + pos
    rec = dict(q=[], k=[], v=[], scores=[], attn=[], head_out=[], attn_out=[], mlp_hidden=[],
               mlp_out=[], ln1=[])
    resid = [x.copy()]
    mask = np.triu(np.ones((T, T), bool), 1)
    for li in range(N_LAYER):
        p = "h.%d." % li
        h = _ln(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        qkv = h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]   # (T, 2304)
        q, k, v = np.split(qkv, 3, axis=1)
        q = q.reshape(T, N_HEAD, D_HEAD).transpose(1, 0, 2)
        k = k.reshape(T, N_HEAD, D_HEAD).transpose(1, 0, 2)
        v = v.reshape(T, N_HEAD, D_HEAD).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(D_HEAD)
        sm = np.where(mask, -1e10, s)
        a = np.exp(sm - sm.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)
        o = a @ v
        cat = o.transpose(1, 0, 2).reshape(T, D_MODEL)
        ao = cat @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        x = x + ao
        h2 = _ln(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        hid = _gelu(h2 @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        mo = hid @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        x = x + mo
        resid.append(x.copy())
        for kk, vv in (("q", q), ("k", k), ("v", v), ("scores", s), ("attn", a),
                       ("head_out", o), ("attn_out", ao), ("mlp_hidden", hid),
                       ("mlp_out", mo), ("ln1", h)):
            rec[kk].append(vv)
    logits = _ln(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T
    last = logits[-1]
    pr = np.exp(last - last.max())
    pr /= pr.sum()
    top = [(tok.decode([int(i)]), float(pr[i])) for i in np.argsort(-pr)[:5]]
    return GPT2Trace(text, ids, [tok.decode([i]) for i in ids], emb, pos, resid,
                     **{k: np.array(v) for k, v in rec.items()}, logits=logits, top=top)
