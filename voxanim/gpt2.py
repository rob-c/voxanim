"""GPT-2 small (124M), run in plain numpy, recording everything the
animation draws.

The weights are OpenAI's released GPT-2 (``openai-community/gpt2`` on the
Hugging Face hub, MIT licence), downloaded once into ``<cache>/gpt2`` (see
:mod:`voxanim.cache`).  Every file is pinned to one hub revision and checked
against its SHA-256 before it is used, so the numbers on screen can be
reproduced exactly.  The ``.safetensors`` file is read as data: an 8-byte
length, a JSON header of names / dtypes / offsets, then the raw arrays.

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
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .cache import DEFAULT_CACHE, download

REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
HUB = "https://huggingface.co/openai-community/gpt2/resolve/%s/" % REVISION
DIR = os.path.join(DEFAULT_CACHE, "gpt2")
# file name -> SHA-256 at REVISION
FILES = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
}
N_LAYER, N_HEAD, D_MODEL, D_HEAD = 12, 12, 768, 64
D_MLP = 4 * D_MODEL              # 3072: the MLP's hidden width
N_VOCAB = 50257                  # 256 bytes + 50,000 merges + <|endoftext|>


def fetch(timeout=60):
    """Download (once) and verify every file in :data:`FILES`; return :data:`DIR`."""
    for name, sha in FILES.items():
        download(HUB + name, os.path.join(DIR, name), sha, timeout)
    return DIR


# ------------------------------------------------------------------ weights


# safetensors dtype tag -> numpy dtype; anything else (BF16, ints) is refused
DTYPES = {"F64": np.float64, "F32": np.float32, "F16": np.float16}


def load_safetensors(path):
    """``{name: array}`` from a .safetensors file (memory-mapped, read-only).

    Raises ``ValueError`` for a dtype numpy cannot read directly (BF16 has no
    numpy equivalent) or a tensor whose byte range does not match its shape.
    """
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    base = 8 + n
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] not in DTYPES:
            raise ValueError("%s: tensor %s has unsupported dtype %s (supported: %s)"
                             % (path, name, info["dtype"], ", ".join(DTYPES)))
        dt = np.dtype(DTYPES[info["dtype"]])
        a, b = info["data_offsets"]
        if b - a != dt.itemsize * int(np.prod(info["shape"])):
            raise ValueError("%s: tensor %s: %d bytes for shape %s"
                             % (path, name, b - a, info["shape"]))
        out[name] = np.frombuffer(raw[base + a:base + b], dtype=dt).reshape(info["shape"])
    return out


@lru_cache(maxsize=1)
def weights():
    """GPT-2's parameters as float32 arrays, keyed without the ``transformer.`` prefix.

    E.g. ``wte.weight`` (N_VOCAB, 768), ``h.4.attn.c_attn.weight`` (768, 2304).
    Linear weights are stored (in, out), so a layer is ``x @ W + b``.
    """
    W = load_safetensors(os.path.join(fetch(), "model.safetensors"))
    return {k.replace("transformer.", ""): np.asarray(v, np.float32) for k, v in W.items()}


# ------------------------------------------------------------------ tokenizer (byte-level BPE)


def _bytes_to_unicode():
    """GPT-2's reversible map from the 256 byte values to printable characters."""
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
    """GPT-2's byte-level BPE tokenizer (``vocab.json`` + ``merges.txt``).

    Text is first split into pieces by :attr:`PAT`, each piece's UTF-8 bytes
    are mapped to printable characters, then merged by rank.

    :attr:`PAT` is GPT-2's split pattern with the stdlib ``re`` module, which
    has no ``\\p{L}`` / ``\\p{N}``: letters are ``[^\\W\\d_]`` and numbers
    ``\\d``.  That is exact for ASCII and for letters and decimal digits in any
    script; it differs from GPT-2 only for the rare non-decimal numerals
    (``²``, ``½``, ``Ⅻ``: Unicode No / Nl), which it groups with letters, and
    for the control characters U+001C to U+001F, which ``re`` counts as
    whitespace.
    """

    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+"""
                     r"""|\s+(?!\S)|\s+""")

    def __init__(self):
        d = fetch()
        with open(os.path.join(d, "vocab.json"), encoding="utf-8") as f:
            self.enc = json.load(f)
        self.dec = {v: k for k, v in self.enc.items()}
        with open(os.path.join(d, "merges.txt"), encoding="utf-8") as f:
            lines = f.read().split("\n")[1:]          # the first line is a version comment
        self.ranks = {tuple(m.split()): i for i, m in enumerate(lines) if m}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}

    def _bpe(self, word):
        """Merge the lowest-ranked adjacent pair until no pair is in the merges."""
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
        """Token ids (list of int) for ``text``."""
        ids = []
        for piece in self.PAT.findall(text):
            u = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.enc[p] for p in self._bpe(u)]
        return ids

    def decode(self, ids):
        """The text for token ``ids``; a partial UTF-8 sequence becomes U+FFFD."""
        return bytearray(self.u2b[c] for i in ids for c in self.dec[i]).decode("utf-8", "replace")


# ------------------------------------------------------------------ forward pass


def layer_norm(x, g, b, eps=1e-5):
    """LayerNorm over the last axis with gain ``g`` and bias ``b`` (GPT-2's ``eps``)."""
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    """GPT-2's tanh approximation of GELU (``gelu_new``)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


@dataclass
class GPT2Trace:
    """Every intermediate of one forward pass.

    ``T`` is the number of tokens, ``L`` = 12 layers, ``H`` = 12 heads.  The
    embeddings (``emb``, ``pos``, ``resid[0]``) are float32; everything after
    the first LayerNorm is float64 (numpy's promotion, kept as computed).
    """

    text: str
    ids: list                     # token ids, length T
    tokens: list                  # each token decoded on its own (display strings)
    emb: np.ndarray               # (T, 768) token embedding
    pos: np.ndarray               # (T, 768) position embedding
    resid: list                   # residual stream: [x0, after layer 1, ..., after 12], (T, 768)
    q: np.ndarray                 # (L, H, T, 64)
    k: np.ndarray                 # (L, H, T, 64)
    v: np.ndarray                 # (L, H, T, 64)
    scores: np.ndarray            # (L, H, T, T)  q k^T / 8, before the mask
    attn: np.ndarray              # (L, H, T, T)  softmax with causal mask
    head_out: np.ndarray          # (L, H, T, 64)
    attn_out: np.ndarray          # (L, T, 768)   after W_o
    mlp_hidden: np.ndarray        # (L, T, 3072)  after GELU
    mlp_out: np.ndarray           # (L, T, 768)
    ln1: np.ndarray               # (L, T, 768)
    logits: np.ndarray            # (T, vocab) for every position
    top: list                     # [(token string, probability)] for the next token, top 5


def run(text="The cat sat on the"):
    """Tokenize ``text``, run GPT-2 small on it and return a :class:`GPT2Trace`."""
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
        h = layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
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
        h2 = layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        hid = _gelu(h2 @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        mo = hid @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
        x = x + mo
        resid.append(x.copy())
        for kk, vv in (("q", q), ("k", k), ("v", v), ("scores", s), ("attn", a),
                       ("head_out", o), ("attn_out", ao), ("mlp_hidden", hid),
                       ("mlp_out", mo), ("ln1", h)):
            rec[kk].append(vv)
    logits = layer_norm(x, W["ln_f.weight"], W["ln_f.bias"]) @ W["wte.weight"].T
    last = logits[-1]
    pr = np.exp(last - last.max())
    pr /= pr.sum()
    top = [(tok.decode([int(i)]), float(pr[i])) for i in np.argsort(-pr)[:5]]
    return GPT2Trace(text, ids, [tok.decode([i]) for i in ids], emb, pos, resid,
                     **{k: np.array(v) for k, v in rec.items()}, logits=logits, top=top)
