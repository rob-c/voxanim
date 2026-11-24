"""The numbers behind animations 194-196 of the "Blocks and code" series (group ``d``):
``nn.BatchNorm1d`` against ``nn.LayerNorm``, ``nn.Embedding`` and ``nn.MultiheadAttention``,
each with the exact settings of its region of ``examples/blocks_parts.py`` and run on real
MNIST numbers (:mod:`voxanim.study_course`) where the region makes up random ones.

* ``blocks-normalisation``: the region's ``h`` (``[64, 64]``, "64 hidden numbers for each
  digit") is the course model's real hidden layer: the first batch through the trained
  model (after the first ReLU, :func:`voxanim.study_course.activations`).
  ``nn.BatchNorm1d(64)`` (training mode, as the region leaves it) subtracts each
  *column's* mean over the 64 digits and divides by its spread; ``nn.LayerNorm(64)`` does
  the same for each *row*, over its 64 features. Both are as created (scale 1, shift 0),
  with ``eps = 1e-5``.
* ``blocks-embedding``: ``nn.Embedding(10, 16)``: a table of 10 rows of 16 numbers,
  started as PyTorch starts it (each number drawn from a normal distribution, mean 0 and
  spread 1), here from a seeded numpy generator, so its numbers are not the ones torch's
  generator would draw; looked up with the first batch's 64 labels: ``[64]`` to
  ``[64, 16]``. Multiplying a one-hot row by the table gives the same row exactly.
* ``blocks-attention``: ``nn.MultiheadAttention(16, 2, batch_first=True)`` on 5 tokens of
  16 numbers: where the region draws ``torch.randn(64, 5, 16)``, each of the first batch's
  64 digits gives 5 patches of 4 x 4 pixels along its middle (pixel rows 12-15, columns
  4-23). Its weights are untrained and initialised as PyTorch does (``in_proj``: Xavier
  uniform, ``out_proj``: uniform in ``±1/4``, both biases 0), from a seeded numpy
  generator. Q, K and V, each head's grid of scores (``q · k / sqrt(8)``), its softmax
  rows, the output, and the ``scores`` the line returns (the two heads' grids averaged).

Every number shown is computed here; ``tests/test_study_blocks_d.py`` checks the identities
(and, with torch installed, that the same weights in ``torch.nn`` give the same numbers).
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, shape, torch_size
from .study import Panel, Study

EPS = 1e-5                      # nn.BatchNorm1d's and nn.LayerNorm's default eps
SOURCE = ("MNIST, as examples/mnist_blocks.py loads it, through the course's model "
          "(voxanim.study_course); the block's settings are those of examples/blocks_parts.py")

# ---------------------------------------------------------------------- 194
FEATURES = 64                   # nn.BatchNorm1d(64), nn.LayerNorm(64)
ROW = 0                         # the digit followed through LayerNorm: the batch's first
SHADE = (-2.0, 3.0)             # the tiles' scale: paper at -2, full ink at 3 (both blocks)

# ---------------------------------------------------------------------- 195
ROWS, WIDTH = 10, 16            # nn.Embedding(10, 16)
TABLE_SEED = 195

# ---------------------------------------------------------------------- 196
TOKENS, DIM, HEADS = 5, 16, 2   # torch.randn(64, 5, 16); nn.MultiheadAttention(16, 2)
PATCH = 4                       # a token: a 4 x 4 patch, 16 numbers
PATCH_ROW = 3                   # pixel rows 12-15: the middle of the digit
PATCH_COLS = (1, 2, 3, 4, 5)    # pixel columns 4-23
ATTN_SEED = 196


def batchnorm(h, eps=EPS):
    """``nn.BatchNorm1d`` in training mode, as created: each column less its mean over
    the rows, over its spread (population standard deviation)."""
    h = np.asarray(h, np.float64)
    return (h - h.mean(0)) / np.sqrt(h.var(0) + eps)


def layernorm(h, eps=EPS):
    """``nn.LayerNorm(h.shape[-1])`` as created: each row less its mean, over its spread."""
    h = np.asarray(h, np.float64)
    return (h - h.mean(-1, keepdims=True)) / np.sqrt(h.var(-1, keepdims=True) + eps)


def one_hot(labels, n=ROWS):
    labels = np.asarray(labels)
    out = np.zeros((len(labels), n))
    out[np.arange(len(labels)), labels] = 1.0
    return out


def embedding_table(seed=TABLE_SEED):
    """``nn.Embedding(10, 16).weight`` as PyTorch starts it (normal, mean 0, spread 1),
    drawn by a seeded numpy generator; float32."""
    rng = np.random.default_rng([course.SEED, int(seed)])
    return rng.standard_normal((ROWS, WIDTH)).astype(np.float32)


def patches(pixels_row):
    """The 5 tokens of one digit (normalised, ``(784,)``): the 4 x 4 patches at patch row
    :data:`PATCH_ROW`, patch columns :data:`PATCH_COLS`, each flattened row by row."""
    img = np.asarray(pixels_row).reshape(course.SIDE, course.SIDE)
    r = PATCH_ROW * PATCH
    return np.stack([img[r:r + PATCH, c * PATCH:(c + 1) * PATCH].reshape(-1)
                     for c in PATCH_COLS])


def attention_weights(seed=ATTN_SEED):
    """``nn.MultiheadAttention(16, 2)``'s parameters as PyTorch initialises them, from a
    seeded numpy generator: ``in_proj_weight [48, 16]`` Xavier uniform, ``in_proj_bias``
    0, ``out_proj.weight [16, 16]`` uniform in ``±1/sqrt(16)``, ``out_proj.bias`` 0."""
    rng = np.random.default_rng([course.SEED, int(seed)])
    bound = np.sqrt(6.0 / (DIM + 3 * DIM))
    return dict(in_proj_weight=rng.uniform(-bound, bound, (3 * DIM, DIM)).astype(np.float32),
                in_proj_bias=np.zeros(3 * DIM, np.float32),
                out_proj_weight=rng.uniform(-1 / np.sqrt(DIM), 1 / np.sqrt(DIM),
                                            (DIM, DIM)).astype(np.float32),
                out_proj_bias=np.zeros(DIM, np.float32))


def softmax_rows(s):
    s = np.asarray(s, np.float64)
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def attend(x, W):
    """``out, scores = attn(x, x, x)`` for ``x [B, T, 16]``: ``dict(q, k, v, raw (B, 2, T,
    T), probs, mixed (B, T, 16), out, scores)``; ``raw``: each head's ``q · k / sqrt(8)``;
    ``probs``: its softmax rows; ``mixed``: the heads' outputs side by side, before
    ``out_proj``; ``scores``: the heads' ``probs`` averaged, as the line returns them."""
    x = np.asarray(x, np.float64)
    w, b = W["in_proj_weight"].astype(np.float64), W["in_proj_bias"].astype(np.float64)
    q, k, v = (x @ w[i * DIM:(i + 1) * DIM].T + b[i * DIM:(i + 1) * DIM] for i in range(3))
    hd = DIM // HEADS
    B, T, _ = x.shape

    def heads(a):
        return a.reshape(B, T, HEADS, hd).transpose(0, 2, 1, 3)
    raw = heads(q) @ heads(k).transpose(0, 1, 3, 2) / np.sqrt(hd)
    probs = softmax_rows(raw)
    mixed = (probs @ heads(v)).transpose(0, 2, 1, 3).reshape(B, T, DIM)
    out = mixed @ W["out_proj_weight"].astype(np.float64).T + W["out_proj_bias"]
    return dict(q=q, k=k, v=v, raw=raw, probs=probs, mixed=mixed, out=out,
                scores=probs.mean(1))


def _count(*arrays):
    return int(sum(np.asarray(a).size for a in arrays))


# ====================================================================== 194
def normalisation():
    a = course.activations(course.EPOCH_STEPS, 0)
    h = np.asarray(a["h1"], np.float32)                       # [64, 64], after the first ReLU
    if h.shape != (course.BATCH, FEATURES):
        raise ValueError("the hidden block is not [64, 64]")
    bn, ln = batchnorm(h), layernorm(h)
    h64 = h.astype(np.float64)
    col_mean, col_std = h64.mean(0), h64.std(0)
    row_mean, row_std = h64.mean(1), h64.std(1)
    col = int(np.argmax(col_mean))                            # the column followed down
    if (col_std == 0).any():
        raise ValueError("a feature is 0 for every digit: its column cannot be spread to 1")
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the first batch's hidden numbers",
        step=course.EPOCH_STEPS, layer="h1: after nn.Linear(784, 64) and nn.ReLU()",
        shape=list(h.shape), shape_text=shape(h.shape), printed=torch_size(h.shape),
        labels=[int(v) for v in a["y"]], eps=EPS, shade=list(SHADE),
        zeros=float((h == 0).mean()),
        col=col, row=ROW,
        col_before=[float(col_mean[col]), float(col_std[col])],
        row_before=[float(row_mean[ROW]), float(row_std[ROW])],
        col_after=[float(bn[:, col].mean()), float(bn[:, col].std())],
        row_after=[float(ln[ROW].mean()), float(ln[ROW].std())],
        col_means=[float(v) for v in col_mean], row_means=[float(v) for v in row_mean],
        col_means_range=[float(col_mean.min()), float(col_mean.max())],
        row_means_range=[float(row_mean.min()), float(row_mean.max())],
        bn_col_means=[float(v) for v in bn.mean(0)], bn_row_means=[float(v) for v in bn.mean(1)],
        ln_col_means=[float(v) for v in ln.mean(0)], ln_row_means=[float(v) for v in ln.mean(1)],
        bn_col_spread=[float(bn.std(0).min()), float(bn.std(0).max())],
        ln_row_spread=[float(ln.std(1).min()), float(ln.std(1).max())],
        bn_row_means_range=[float(bn.mean(1).min()), float(bn.mean(1).max())],
        ln_col_means_range=[float(ln.mean(0).min()), float(ln.mean(0).max())],
        learned=2 * FEATURES,                                 # a scale and a shift per feature
        code=list(code_lines("batchnorm")) + list(code_lines("layernorm")),
        limitations=[
            "h is the trained model's real hidden layer for the first batch (after one epoch "
            "of SGD), where the region makes up torch.randn(64, 64)",
            "both blocks are as created: scale 1 and shift 0, so their output is exactly the "
            "normalised numbers; training would learn a scale and a shift per feature",
            "BatchNorm is in training mode (as the region leaves it): it uses this batch's own "
            "means; in eval mode it uses running averages, not shown"])
    panels = (
        Panel("h", "the first batch's 64 hidden numbers per digit, one digit to a row",
              "matrix", dict(values=h.astype(float))),
        Panel("nn.BatchNorm1d(64)(h)", "each column less its mean, over its spread", "matrix",
              dict(values=bn)),
        Panel("nn.LayerNorm(64)(h)", "each row less its mean, over its spread", "matrix",
              dict(values=ln)),
        Panel("Means before", "the mean of each column of h (BatchNorm sets these to 0)",
              "bars", dict(values=col_mean, labels=[str(i) for i in range(FEATURES)])),
    )
    return Study("BatchNorm and LayerNorm", "nn.BatchNorm1d averages down the batch, "
                 "nn.LayerNorm across the features", panels, meta)


# ====================================================================== 195
def embedding():
    table = embedding_table()
    y = course.first_batch()["y"]
    vectors = table[y]
    hot = one_hot(y)
    product = hot @ table.astype(np.float64)
    if not np.array_equal(product, vectors.astype(np.float64)):
        raise ValueError("one-hot times the table is not the looked-up rows")
    uses = np.bincount(y, minlength=ROWS)
    first = int(y[0])
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: rows of a table, looked up",
        rows=ROWS, width=WIDTH, numbers=ROWS * WIDTH, seed=TABLE_SEED,
        table_shape=list(table.shape), out_shape=list(vectors.shape),
        out_text=shape(vectors.shape), labels_text=shape(y.shape),
        printed=torch_size(vectors.shape), labels=[int(v) for v in y],
        first=first, first_row=[float(v) for v in table[first]],
        second=int(y[1]), uses=[int(v) for v in uses],
        lookups=int(len(y)), multiplications_one=ROWS * WIDTH,
        multiplications_batch=int(len(y)) * ROWS * WIDTH,
        by_one=WIDTH, by_zero=(ROWS - 1) * WIDTH, same=True,
        low=float(table.min()), high=float(table.max()),
        code=list(code_lines("embedding")), make=code_line("embedding", "nn.Embedding"),
        limitations=[
            "the table is untrained and drawn as PyTorch draws it (normal, mean 0, spread 1) "
            "but by a seeded numpy generator, so its numbers are not torch's; training would "
            "change them, never the lookup",
            "the labels are the first batch's, in the course's shuffled order"])
    panels = (
        Panel("The table", "nn.Embedding(10, 16).weight: 10 rows of 16 numbers (seeded)",
              "matrix", dict(values=table.astype(float))),
        Panel("embed(y)", "the row of each of the first batch's 64 labels, in order", "matrix",
              dict(values=vectors.astype(float))),
        Panel("Rows used", "how many of the 64 labels pick each row", "bars",
              dict(values=uses.astype(float), labels=[str(d) for d in range(ROWS)])),
        Panel("Lookup or product", "multiplications to get the batch's rows: a lookup, and "
              "one-hot rows times the table (the same numbers)", "bars",
              dict(values=np.array([0.0, len(y) * ROWS * WIDTH]),
                   labels=["lookup", "one-hot × table"])),
    )
    return Study("nn.Embedding: a table you look up", "nn.Embedding(10, 16) turns the "
                 "first batch's 64 labels into 64 rows of a table", panels, meta)


# ====================================================================== 196
def attention():
    b = course.first_batch()
    x = np.stack([patches(row) for row in b["x"].reshape(len(b["y"]), -1)]).astype(np.float32)
    if x.shape != (course.BATCH, TOKENS, DIM):
        raise ValueError("the tokens are not [64, 5, 16]")
    W = attention_weights()
    r = attend(x, W)
    d = 0                                                     # the batch's first digit
    probs = r["probs"][d]
    query = int(np.argmax(probs[0].max(-1)))                  # head 1's most decided row
    key = int(np.argmax(probs[0, query]))
    hd = DIM // HEADS
    dot = float(r["q"][d, query, :hd] @ r["k"][d, key, :hd])
    params = _count(*W.values())
    r0 = PATCH_ROW * PATCH
    img = b["x"][d, 0]
    meta = dict(
        source=SOURCE, evidence="untrained weights, seeded; tokens from MNIST digits",
        digit=d, label=int(b["y"][d]), tokens=TOKENS, dim=DIM, heads=HEADS, head_dim=hd,
        patch=PATCH, rows=[r0, r0 + PATCH - 1],
        cols=[PATCH_COLS[0] * PATCH, (PATCH_COLS[-1] + 1) * PATCH - 1],
        corners=[[r0, c * PATCH] for c in PATCH_COLS],
        x_shape=list(x.shape), x_text=shape(x.shape), out_text=shape(r["out"].shape),
        scores_text=shape(r["scores"].shape),
        printed=[torch_size(r["out"].shape), torch_size(r["scores"].shape)],
        blank=[int(i) for i in range(TOKENS) if np.all(x[d, i] == x[d].min())],
        ink=[float(x[d, i].max()) for i in range(TOKENS)],
        query=query, key=key, dot=dot, scale=float(np.sqrt(hd)),
        raw_cell=float(r["raw"][d, 0, query, key]),
        raw=r["raw"][d].tolist(), probs=probs.tolist(), scores=r["scores"][d].tolist(),
        row_sums=[float(v) for v in probs.sum(-1).ravel()],
        query_row=[float(v) for v in probs[0, query]],
        out=r["out"][d].tolist(), x=x[d].astype(float).tolist(),
        q=r["q"][d].tolist(), k=r["k"][d].tolist(), v=r["v"][d].tolist(),
        params=params, in_proj=_count(W["in_proj_weight"], W["in_proj_bias"]),
        out_proj=_count(W["out_proj_weight"], W["out_proj_bias"]),
        code=list(code_lines("attention")),
        make=code_line("attention", "nn.MultiheadAttention("),
        call=code_line("attention", "attn(q, k, v)"),
        limitations=[
            "the tokens are 5 patches of 4 x 4 pixels of each digit of the first batch "
            "(pixel rows 12-15, columns 4-23), where the region makes up torch.randn(64, 5, "
            "16): same shape, real numbers",
            "the weights are untrained, initialised as PyTorch does (Xavier uniform, uniform "
            "in ±0.25, biases 0) by a seeded numpy generator, so the scores show the "
            "arithmetic, not anything learned",
            "scores, as the line returns them, are the two heads' grids averaged "
            "(average_attn_weights=True)"])
    tokens = np.stack([x[d, i].reshape(PATCH, PATCH) for i in range(TOKENS)]).astype(float)
    panels = (
        Panel("The tokens", "the 5 patches of the batch's first digit (a %d), 4 x 4 each"
              % int(b["y"][d]), "images", dict(values=tokens,
                                               labels=["token %d" % (i + 1)
                                                       for i in range(TOKENS)])),
        Panel("Head 1's scores", "softmax(q · k / √8), one row per token: each adds up to 1",
              "matrix", dict(values=probs[0])),
        Panel("Head 2's scores", "the same for the second head", "matrix",
              dict(values=probs[1])),
        Panel("The output", "out: 5 tokens of 16 numbers, each a mix of the V rows",
              "matrix", dict(values=r["out"][d])),
    )
    meta["digit_image"] = img.astype(float).tolist()
    meta["in_proj_weight"] = W["in_proj_weight"].astype(float).tolist()
    meta["out_proj_weight"] = W["out_proj_weight"].astype(float).tolist()
    return Study("nn.MultiheadAttention as blocks", "Q, K, V, a grid of scores and the "
                 "output, on 5 patches of a real digit", panels, meta)


STUDIES = {"blocks-normalisation": normalisation, "blocks-embedding": embedding,
           "blocks-attention": attention}
