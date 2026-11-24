"""The arithmetic behind animation 184, ``counting-weights-transformer``: how many numbers
a transformer learns, counted one part at a time with one rule.

* Attention first, in the small vision transformer of "A self-attention training step"
  (:mod:`voxanim.vit`): its query, key, value and output matrices are four blocks of
  ``D x D`` weights with no biases. Heads cut those blocks into pieces; any number of heads
  that divides ``D`` gives the same count.
* The rest of that model (patch embedding, class token, positions, the MLP on the class
  token, the output layer) and its total, checked against the arrays :func:`voxanim.vit.init`
  makes.
* One block of GPT-2 small (:mod:`voxanim.gpt2`), tensor by tensor: two LayerNorms,
  ``attn.c_attn`` (Q, K and V in one matrix), ``attn.c_proj``, ``mlp.c_fc`` and
  ``mlp.c_proj``; then twelve blocks, the token and position tables, the final LayerNorm and
  the output layer, which reuses the token table (weight tying: no new weights). The shapes
  come from :mod:`voxanim.gpt2`'s constants, named as in the released checkpoint; the tests
  check every count against the downloaded weights when they are there.

Each part carries the PyTorch line that would make it (text only: nothing imports torch).
Every number is exact integer arithmetic; nothing is trained or downloaded.
"""

from __future__ import annotations

import numpy as np

from . import gpt2, vit
from .study import Panel, Study

VERSION = 1
EVIDENCE = "exact arithmetic: weights counted from each layer's shape"
VIT_SEED = 0                     # the seed vit.train starts from; any seed has the same shapes
CLASSES = 10                     # the ViT's outputs: one per digit


def _part(key, label, code, kind, shape, names, weights, biases, **extra):
    """One counted part: what the drawing shows and the tests check."""
    return dict(key=key, label=label, code=code, kind=kind, shape=[int(v) for v in shape],
                names=list(names), weights=int(weights), biases=int(biases),
                total=int(weights + biases), **extra)


def linear(key, label, code, n_in, n_out, names, bias=True, times=1, **extra):
    """``times`` layers of ``n_in`` inputs and ``n_out`` outputs: one weight per input for
    each output, plus one bias per output."""
    return _part(key, label, code, "linear", (n_in, n_out), names,
                 times * n_in * n_out, times * n_out if bias else 0, bias=bias,
                 times=times, **extra)


def table(key, label, code, rows, width, names, unit):
    """A learned table: ``rows`` rows of ``width`` numbers (an embedding, positions, a
    class token)."""
    return _part(key, label, code, "table", (rows, width), names, rows * width, 0, unit=unit)


def norm(key, label, code, width, names):
    """A LayerNorm: one scale (a weight) and one shift (a bias) per number."""
    return _part(key, label, code, "norm", (width,), names, width, width)


def _running(parts):
    total = 0
    for p in parts:
        total += p["total"]
        p["running"] = total
    return total


# ---------------------------------------------------------------------- the small ViT
def vit_attention():
    """Q, K, V and O of :mod:`voxanim.vit`: four ``D x D`` blocks, no biases."""
    d = vit.D
    mats = []
    for key, label in (("Wq", "query"), ("Wk", "key"), ("Wv", "value"), ("Wo", "output")):
        mats.append(linear(key, label, "%s = nn.Linear(%d, %d, bias=False)" % (key[1], d, d),
                           d, d, [key], bias=False))
    total = _running(mats)
    # heads cut D into HEADS pieces of DH: Q, K and V by their outputs (columns), O by its
    # inputs (rows). Every divisor of D is a valid number of heads.
    divisors = [h for h in range(1, d + 1) if d % h == 0]
    by_heads = {h: 4 * h * d * (d // h) for h in divisors}
    return dict(d=d, heads=vit.HEADS, head_width=vit.DH, matrices=mats, total=total,
                piece=d * vit.DH, by_heads={str(h): v for h, v in by_heads.items()},
                code_heads="attn = nn.MultiheadAttention(%d, %d, bias=False)" % (d, vit.HEADS),
                code_one_head="attn = nn.MultiheadAttention(%d, 1, bias=False)" % d)


def vit_parts():
    """Every part of the small ViT, in the order a token meets them."""
    d, p2, att = vit.D, vit.PATCH * vit.PATCH, vit_attention()
    parts = [
        linear("patch", "patches → tokens", "patch = nn.Linear(%d, %d)" % (p2, d), p2, d,
               ["E", "bE"]),
        table("cls", "class token", "cls = nn.Parameter(torch.zeros(%d))" % d, 1, d, ["cls"],
              "row"),
        table("pos", "positions", "pos = nn.Parameter(torch.zeros(%d, %d))" % (vit.N_TOK, d),
              vit.N_TOK, d, ["pos"], "positions"),
        linear("attention", "attention", att["code_heads"], d, d, ["Wq", "Wk", "Wv", "Wo"],
               bias=False, times=4),
        linear("mlp", "MLP", "mlp = nn.Linear(%d, %d)" % (d, vit.HIDDEN), d, vit.HIDDEN,
               ["W1", "b1"]),
        linear("output", "output: 10 digits", "out = nn.Linear(%d, %d)" % (vit.HIDDEN, CLASSES),
               vit.HIDDEN, CLASSES, ["W2", "b2"]),
    ]
    total = _running(parts)
    assert parts[3]["total"] == att["total"]
    return parts, total


def vit_arrays(seed=VIT_SEED):
    """The model's own parameter arrays, as :func:`voxanim.vit.init` makes them."""
    return vit.init(np.random.default_rng(seed))


# ---------------------------------------------------------------------- GPT-2 small
def gpt2_block():
    """One GPT-2 block, tensor by tensor (names inside ``h.<i>.``)."""
    d, f = gpt2.D_MODEL, gpt2.D_MLP
    parts = [
        norm("ln_1", "norm", "ln_1 = nn.LayerNorm(%d)" % d, d, ["ln_1.weight", "ln_1.bias"]),
        linear("attn.c_attn", "attention: Q, K, V", "c_attn = nn.Linear(%d, 3 * %d)" % (d, d),
               d, 3 * d, ["attn.c_attn.weight", "attn.c_attn.bias"]),
        linear("attn.c_proj", "attention: output", "c_proj = nn.Linear(%d, %d)" % (d, d), d, d,
               ["attn.c_proj.weight", "attn.c_proj.bias"]),
        norm("ln_2", "norm", "ln_2 = nn.LayerNorm(%d)" % d, d, ["ln_2.weight", "ln_2.bias"]),
        linear("mlp.c_fc", "MLP: wider", "c_fc = nn.Linear(%d, 4 * %d)" % (d, d), d, f,
               ["mlp.c_fc.weight", "mlp.c_fc.bias"]),
        linear("mlp.c_proj", "MLP: back", "c_proj = nn.Linear(4 * %d, %d)" % (d, d), f, d,
               ["mlp.c_proj.weight", "mlp.c_proj.bias"]),
    ]
    return parts, _running(parts)


def gpt2_model():
    """GPT-2 small in the order a token meets it: tables, twelve blocks, the final norm and
    the tied output layer."""
    d, n = gpt2.D_MODEL, gpt2.N_LAYER
    block, per_block = gpt2_block()
    blocks = _part("h", "%d blocks" % n, "h = [Block() for _ in range(%d)]" % n, "blocks",
                   (n,), ["h.%d." % i for i in range(n)],
                   n * sum(p["weights"] for p in block), n * sum(p["biases"] for p in block),
                   per_block=per_block, count=n)
    parts = [
        table("wte", "token table", "wte = nn.Embedding(%d, %d)" % (gpt2.N_VOCAB, d),
              gpt2.N_VOCAB, d, ["wte.weight"], "tokens"),
        table("wpe", "positions", "wpe = nn.Embedding(%d, %d)" % (gpt2.N_CTX, d), gpt2.N_CTX,
              d, ["wpe.weight"], "positions"),
        blocks,
        norm("ln_f", "final norm", "ln_f = nn.LayerNorm(%d)" % d, d,
             ["ln_f.weight", "ln_f.bias"]),
        _part("lm_head", "output layer", "lm_head.weight = wte.weight", "tied",
              (gpt2.N_VOCAB, d), [], 0, 0, reuses="wte"),
    ]
    return block, per_block, parts, _running(parts)


def gpt2_shapes():
    """Every learned tensor of GPT-2 small by its checkpoint name (without ``transformer.``)
    and shape, as :mod:`voxanim.gpt2` reads them. Linear weights are stored (in, out)."""
    block, _ = gpt2_block()
    shapes = {"wte.weight": (gpt2.N_VOCAB, gpt2.D_MODEL), "wpe.weight": (gpt2.N_CTX, gpt2.D_MODEL),
              "ln_f.weight": (gpt2.D_MODEL,), "ln_f.bias": (gpt2.D_MODEL,)}
    for i in range(gpt2.N_LAYER):
        for p in block:
            for name in p["names"]:
                if p["kind"] == "norm" or name.endswith(".bias"):
                    shape = (p["shape"][-1],)
                else:
                    shape = tuple(p["shape"])
                shapes["h.%d.%s" % (i, name)] = shape
    return shapes


def where(block, parts):
    """GPT-2's weights grouped by where they live."""
    by = {p["key"]: p for p in parts}
    bk = {p["key"]: p for p in block}
    n = by["h"]["count"]
    attention = n * (bk["attn.c_attn"]["total"] + bk["attn.c_proj"]["total"])
    mlp = n * (bk["mlp.c_fc"]["total"] + bk["mlp.c_proj"]["total"])
    norms = n * (bk["ln_1"]["total"] + bk["ln_2"]["total"]) + by["ln_f"]["total"]
    groups = [("MLPs", mlp), ("token table", by["wte"]["total"]), ("attention", attention),
              ("positions", by["wpe"]["total"]), ("norms", norms),
              ("output layer", by["lm_head"]["total"])]
    return [dict(label=k, total=int(v)) for k, v in groups]


# ---------------------------------------------------------------------- the study
def counting_weights_transformer():
    att = vit_attention()
    vparts, vtotal = vit_parts()
    arrays = vit_arrays()
    vit_counted = int(sum(arrays[k].size for k in vit.NAMES))
    assert vit_counted == vtotal, (vit_counted, vtotal)
    for p in vparts:
        assert sum(arrays[k].size for k in p["names"]) == p["total"], p["key"]
    block, per_block, gparts, gtotal = gpt2_model()
    shapes = gpt2_shapes()
    from_shapes = int(sum(int(np.prod(s)) for s in shapes.values()))
    assert from_shapes == gtotal, (from_shapes, gtotal)
    groups = where(block, gparts)
    assert sum(g["total"] for g in groups) == gtotal
    for g in groups:
        g["share"] = g["total"] / gtotal
    bk = {p["key"]: p for p in block}
    block_attention = bk["attn.c_attn"]["total"] + bk["attn.c_proj"]["total"]
    block_mlp = bk["mlp.c_fc"]["total"] + bk["mlp.c_proj"]["total"]
    heads = sorted(int(h) for h in att["by_heads"])
    panels = (
        Panel("Attention: four blocks", "the small vision transformer's query, key, value and "
              "output matrices, %d × %d weights each, no biases" % (att["d"], att["d"]), "bars",
              dict(values=np.array([m["total"] for m in att["matrices"]]),
                   labels=[m["label"] for m in att["matrices"]],
                   running=np.array([m["running"] for m in att["matrices"]]))),
        Panel("Heads add no weights", "attention weights for every number of heads that "
              "divides the width %d: always %d" % (att["d"], att["total"]), "points",
              dict(xy=np.array([[h, att["by_heads"][str(h)]] for h in heads], float),
                   heads=np.array(heads))),
        Panel("The small vision transformer", "weights and biases of each part, in the order "
              "a token meets them: %d in all" % vtotal, "bars",
              dict(values=np.array([p["total"] for p in vparts]),
                   labels=[p["label"] for p in vparts],
                   running=np.array([p["running"] for p in vparts]))),
        Panel("Where GPT-2's weights live", "GPT-2 small's %d weights grouped: the MLPs and the "
              "token table hold most; the output layer reuses the token table" % gtotal, "bars",
              dict(values=np.array([g["total"] for g in groups]),
                   labels=[g["label"] for g in groups],
                   block=np.array([p["total"] for p in block]),
                   block_labels=[p["key"] for p in block])),
    )
    return Study(
        "Counting the weights: a transformer", "the same rule, part by part, in a small "
        "vision transformer (%d) and in GPT-2 small (%d)" % (vtotal, gtotal), panels,
        dict(source="the shapes of voxanim.vit (the small vision transformer of 'A "
                    "self-attention training step') and of GPT-2 small as voxanim.gpt2 reads "
                    "it; the ViT's counts checked against the arrays vit.init makes, GPT-2's "
                    "against the downloaded weights by the tests",
             evidence=EVIDENCE, version=VERSION,
             vit=dict(d=vit.D, heads=vit.HEADS, head_width=vit.DH, patch=vit.PATCH,
                      n_tok=vit.N_TOK, hidden=vit.HIDDEN, classes=CLASSES,
                      attention=att, parts=vparts, total=vtotal, counted=vit_counted,
                      norms=0, seed=VIT_SEED),
             gpt2=dict(d=gpt2.D_MODEL, mlp=gpt2.D_MLP, layers=gpt2.N_LAYER,
                       heads=gpt2.N_HEAD, vocab=gpt2.N_VOCAB, context=gpt2.N_CTX,
                       block=block, per_block=per_block, parts=gparts, total=gtotal,
                       from_shapes=from_shapes, groups=groups,
                       block_attention=block_attention, block_mlp=block_mlp,
                       block_norms=per_block - block_attention - block_mlp,
                       attention_share_of_block=block_attention / per_block,
                       mlp_share_of_block=block_mlp / per_block,
                       tied_saves=gpt2.N_VOCAB * gpt2.D_MODEL),
             count_line="sum(p.numel() for p in model.parameters())",
             limitations=["the counts are of learned numbers only: GPT-2's checkpoint also "
                          "stores a causal mask per block (h.<i>.attn.bias), which is not "
                          "learned and not counted",
                          "the small vision transformer has no LayerNorm and puts its MLP "
                          "on the class token only, so it is not a full transformer block"]))


STUDIES = {"counting-weights-transformer": counting_weights_transformer}
