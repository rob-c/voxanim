"""The numbers behind animations 232-233 of "Blocks and code" (group ``q``, the last two of
the padding part): masks that make a model ignore the padding, and sorting lines by length
so there is less padding to ignore. Everything comes from the lines of Tiny Shakespeare
that ``examples/padding.py`` reads (:mod:`voxanim.study_padding`), worked out here in numpy.

* ``padding-masks``: the first eight lines padded to ``[8, 50]``
  (:func:`~voxanim.study_padding.lines_batch`). Region "pad-masked-mean" embeds them with
  ``padding_idx=0``, so every pad's vector is 0, and the plain mean over a line is its true
  mean times ``length / 50``, exactly (:func:`~voxanim.study_padding.masked_mean`): "All:" is
  4 / 50 = 0.08 of its true mean. Region "pad-attention" gives every pad key a weight of
  exactly 0 through ``key_padding_mask``. The embedding table and the attention's weights
  the script draws at random are drawn here from a seed of this module's own
  (:func:`table`, :func:`in_proj`): the vectors and the weights on the real characters are
  random, the zeros on the pads and the ratios are exact whatever the weights
  (``tests/test_study_blocks_q.py`` runs the same weights through torch). Region "pad-loss":
  ``ignore_index=0`` counts the 163 real targets and skips the 237 pads.
* ``padding-sort-by-length``: all 32,777 lines in batches of 64
  (:func:`~voxanim.study_padding.waste`): 38.0% of the cells are padding in file order,
  38.7% shuffled (one seeded shuffle, :func:`shuffled_order`), 0.16% sorted by length
  (region "pad-sort"). Real batches drawn as rows of bars (:func:`batch_rows`): the first
  two in file order and shuffled, and two from the sorted lines a third and two thirds of
  the way through (not picked for how they look). Sorting puts lines of one length
  together, so in practice the sorted batches are then shuffled as whole batches
  (:func:`whole_batches`): each keeps its own padding, so the total stays 0.16%. Nothing
  here measures what that does to training.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from . import study_padding as pad
from .blocks import code_lines, commas, shape
from .study import Panel, Study

SEED = 232                      # this module's own draws: the embedding and attention weights
DIM, HEADS = 16, 2              # nn.Embedding(len(vocab), 16), nn.MultiheadAttention(16, 2)
FOLLOW = 2                      # the line followed through the means: "All:"
SHOWN_ROW = 1                   # print(w[1, 0]): the second line's first character
BATCH = pad.BATCH               # 64 lines a batch
MINI = 6                        # sorted batches drawn as small blocks, evenly spaced
SOURCE = ("Tiny Shakespeare (char-rnn), every line that is not empty, as examples/padding.py "
          "reads it (voxanim.study_padding)")
RANDOM = ("the embedding table and the attention's in_proj weights are drawn at random from "
          "a seed of this module's own, as the script draws its own; the zeros on the pads and "
          "the ratio length / 50 are exact whatever the weights")


# ====================================================================== 232
def table(seed=SEED, vocab=None):
    """An embedding table as ``nn.Embedding(len(vocab), 16, padding_idx=0)`` starts:
    N(0, 1) numbers, the pad's row (id 0) set to 0. Random, seeded."""
    n = len(pad.vocab()) if vocab is None else int(vocab)
    rng = np.random.default_rng([course.SEED, 232, int(seed)])
    t = rng.standard_normal((n, DIM)).astype(np.float32)
    t[0] = 0.0
    return t


def in_proj(seed=SEED):
    """``in_proj_weight`` ``[48, 16]`` as ``nn.MultiheadAttention(16, 2)`` starts it (Xavier
    uniform; its bias starts at 0). Random, seeded."""
    rng = np.random.default_rng([course.SEED, 232, int(seed), 1])
    bound = np.sqrt(6.0 / (DIM + 3 * DIM))
    return rng.uniform(-bound, bound, (3 * DIM, DIM)).astype(np.float32)


def embed(batch, t=None):
    """``embed(batch)``: ``[8, L, 16]``."""
    t = table() if t is None else t
    return t[np.asarray(batch)]


def means(vecs, mask):
    """Region "pad-masked-mean": ``plain = vecs.mean(1)``, the pads counted in, and
    ``mean = total / keep.sum(1)``, the pads left out."""
    keep = np.asarray(mask, np.float32)[..., None]
    plain = vecs.mean(1)
    total = (vecs * keep).sum(1)
    return plain, total / keep.sum(1)


def attention(vecs, mask, w_in=None):
    """The weights ``w`` of ``attn(vecs, vecs, vecs, key_padding_mask=~mask)`` with
    ``batch_first=True``, averaged over the heads as PyTorch returns them: ``[B, L, L]``.
    A pad key's score is ``-inf`` before the softmax, so its weight is exactly 0."""
    w_in = in_proj() if w_in is None else w_in
    vecs = np.asarray(vecs, np.float64)
    q, k = vecs @ w_in[:DIM].T, vecs @ w_in[DIM:2 * DIM].T      # the bias starts at 0
    hd = DIM // HEADS
    B, L, _ = vecs.shape
    out = np.zeros((B, L, L))
    for h in range(HEADS):
        s = q[..., h * hd:(h + 1) * hd] @ k[..., h * hd:(h + 1) * hd].transpose(0, 2, 1)
        s = s / np.sqrt(hd)
        s = np.where(np.asarray(mask)[:, None, :], s, -np.inf)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        out += e / e.sum(-1, keepdims=True)
    return out / HEADS


def ratio_text(length, longest):
    return "%d ÷ %d = %s" % (length, longest, two(length / longest))


def two(v):
    """``0.08``, ``1.00``."""
    return "%.2f" % v


def masks():
    b = pad.lines_batch()
    mm = pad.masked_mean()
    batch, mask = b["batch"], b["mask"]
    n, L = batch.shape
    vecs = embed(batch)
    plain, mean = means(vecs, mask)
    # the identity behind the bars: plain x L = mean x length, to float32 rounding
    lengths = np.asarray(b["lengths"])
    if not np.allclose(plain * L, mean * lengths[:, None], rtol=1e-5, atol=1e-5):
        raise ValueError("the plain mean is not the true mean times length / L")
    if not np.all(vecs[~mask] == 0):
        raise ValueError("a pad's vector is not 0")
    w = attention(vecs, mask)
    first = w[:, 0, :]
    row = w[SHOWN_ROW, 0]
    if np.any(first[~mask] != 0) or np.any(first[mask] <= 0):
        raise ValueError("a pad key has a weight, or a real key none")
    real, cells = b["real"], b["cells"]
    f = FOLLOW
    meta = dict(
        source=SOURCE, random=RANDOM, seed=SEED,
        evidence="exact arithmetic: counted on the real lines; the zeros and ratios exact, "
                 "the weights on real characters random",
        text=b["text"], lengths=b["lengths"], longest=L, batch=batch.tolist(),
        mask=mask.tolist(), shape=[n, L], shape_text=shape((n, L)),
        real=real, cells=cells, pads=cells - real,
        follow=f, follow_text=b["text"][f], follow_length=b["lengths"][f],
        follow_pads=L - b["lengths"][f],
        follow_vecs=vecs[f].tolist(),
        follow_ratio=mm["ratio"][f], follow_ratio_text=ratio_text(b["lengths"][f], L),
        ratio=mm["ratio"], ratio_texts=[two(r) for r in mm["ratio"]],
        plain_over_mean=(plain / mean).tolist(),
        dim=DIM, heads=HEADS,
        first_weights=first.tolist(), shown_row=SHOWN_ROW,
        shown_length=b["lengths"][SHOWN_ROW], shown_pads=L - b["lengths"][SHOWN_ROW],
        shown_weights=row.tolist(), shown_sum=float(row.sum()),
        shown_sum_text=two(row.sum()),
        pad_weights_max=float(np.abs(first[~mask]).max()),
        rows_add_to_one=bool(np.allclose(w.sum(-1), 1.0)),
        loss_counted=real, loss_skipped=cells - real,
        code_mean=list(code_lines("pad-masked-mean")),
        code_attention=list(code_lines("pad-attention")),
        code_loss=list(code_lines("pad-loss")),
        limitations=[
            RANDOM,
            "the attention weights drawn are those of each line's first character (query 0); "
            "a query that is itself a pad still gets a row of weights, over the real keys",
            "ignore_index: the targets of a model reading these lines would be the next "
            "characters, padded the same way; the count of 163 real targets of 400 is for "
            "targets padded as this block is"])
    panels = (
        Panel("Line lengths", "characters in each of the first 8 lines; padded to %d" % L,
              "bars", dict(values=np.asarray(b["lengths"], float),
                           labels=[str(i + 1) for i in range(n)])),
        Panel("Plain mean ÷ true mean", "vecs.mean(1) over the padded line, against the mean "
              "over its real characters: length / %d, exactly" % L, "bars",
              dict(values=np.asarray(mm["ratio"], float),
                   labels=[str(i + 1) for i in range(n)])),
        Panel("Attention with key_padding_mask", "each line's first character: its weight on "
              "each of the %d keys (random weights; every pad exactly 0)" % L, "matrix",
              dict(values=first)),
        Panel("Targets the loss counts", "with ignore_index=0: real targets counted, pads "
              "skipped", "bars", dict(values=np.array([real, cells - real], float),
                                     labels=["counted", "skipped"])),
    )
    return Study("Masks: ignore the padding", "A plain mean over a padded line shrinks by its "
                 "length over the longest; masks keep the pads out of means, attention and "
                 "the loss", panels, meta)


# ====================================================================== 233
def lengths():
    """Every line's length, in file order."""
    return np.array([len(s) for s in pad.lines()])


def shuffled_order(seed=pad.SEED):
    """The order :func:`~voxanim.study_padding.waste` shuffles the lines into (the same
    generator and draw: ``permutation(n)`` and ``permutation(lengths)`` shuffle alike)."""
    rng = np.random.default_rng([course.SEED, 4, int(seed)])
    return rng.permutation(len(pad.lines()))


def sorted_order():
    """``lines.sort(key=len)``: Python's sort is stable, as is this one."""
    return np.argsort(lengths(), kind="stable")


def per_batch(ln, batch=BATCH):
    """Each batch of ``batch`` lines: ``(lines, longest, pads)``."""
    out = []
    for i in range(0, len(ln), batch):
        g = ln[i:i + batch]
        out.append((len(g), int(g.max()), int(len(g) * g.max() - g.sum())))
    return out


def batch_rows(ln, k, batch=BATCH):
    """Batch ``k`` (0-based) as rows of bars: ``dict(index, lengths, longest, pads, share)``."""
    g = [int(v) for v in ln[k * batch:(k + 1) * batch]]
    longest = max(g)
    cells = len(g) * longest
    return dict(index=int(k), lengths=g, longest=longest, pads=cells - sum(g),
                share=1 - sum(g) / cells)


def pct(share, places=1):
    return "%.*f%%" % (places, 100 * share)


def whole_batches(n_full, mini=MINI, seed=SEED):
    """``mini`` of the ``n_full`` full sorted batches (the last, short one left out), evenly
    spaced from the shortest to the longest, and a seeded shuffle of them (an example of
    shuffling whole batches)."""
    picks = np.round(np.linspace(0, n_full - 1, mini)).astype(int)
    rng = np.random.default_rng([course.SEED, 233, int(seed)])
    order = rng.permutation(mini)
    while np.any(order == np.arange(mini)):          # every one moves, so it shows
        order = rng.permutation(mini)
    return [int(p) for p in picks], [int(o) for o in order]


def sort_by_length():
    w = pad.waste()
    ln = lengths()
    orders = dict(file=np.arange(len(ln)), shuffled=shuffled_order(), sorted=sorted_order())
    arranged = {k: ln[v] for k, v in orders.items()}
    if not np.array_equal(arranged["sorted"], np.sort(ln, kind="stable")):
        raise ValueError("the sorted order is not the sort of the lengths")
    for k in orders:                                   # the same counts as waste()
        pb = per_batch(arranged[k])
        if sum(p for _n, _l, p in pb) != w[k]["padding"]:
            raise ValueError("%s: the batches' padding is not waste()'s" % k)
    n_batches = len(per_batch(ln))
    third = (n_batches // 3, 2 * n_batches // 3)
    rows = dict(file=[batch_rows(arranged["file"], k) for k in (0, 1)],
                shuffled=[batch_rows(arranged["shuffled"], k) for k in (0, 1)],
                sorted=[batch_rows(arranged["sorted"], k) for k in third])
    picks, order = whole_batches(len(ln) // BATCH)
    mini = [batch_rows(arranged["sorted"], k) for k in picks]
    shares = {k: w[k]["share"] for k in orders}
    per = {k: [p / (n * longest) for n, longest, p in per_batch(arranged[k])] for k in orders}
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: counted over every line",
        lines=w["lines"], lines_text=commas(w["lines"]), characters=w["characters"],
        characters_text=commas(w["characters"]), shortest=w["shortest"], longest=w["longest"],
        batch=BATCH, batches=n_batches,
        cells={k: w[k]["cells"] for k in orders},
        cells_text={k: commas(w[k]["cells"]) for k in orders},
        padding={k: w[k]["padding"] for k in orders},
        padding_text={k: commas(w[k]["padding"]) for k in orders},
        share=shares,
        share_text=dict(file=pct(shares["file"]), shuffled=pct(shares["shuffled"]),
                        sorted=pct(shares["sorted"], 2)),
        fewer_cells=1 - w["sorted"]["cells"] / w["file"]["cells"],
        fewer_cells_text=pct(1 - w["sorted"]["cells"] / w["file"]["cells"], 0),
        rows=rows, row_share_text={k: [pct(r["share"], 0) for r in v] for k, v in rows.items()},
        full_batches=len(ln) // BATCH, last_batch=len(ln) % BATCH,
        mini=mini, mini_order=order, mini_longest=[r["longest"] for r in mini],
        sorted_batches_with_pads=int(sum(1 for s in per["sorted"] if s > 0)),
        shuffle_seed=pad.SEED,
        code=list(code_lines("pad-sort")),
        limitations=[
            "one seeded shuffle of the lines; another shuffle moves the 38.7% a little",
            "the batches drawn: the first two in file order and shuffled, and the sorted "
            "batches a third and two thirds of the way through, not picked",
            "sorting puts lines of one length together, so in practice whole batches are "
            "shuffled; what either does to training is not measured here"])
    counts = np.bincount(ln, minlength=w["longest"] + 1)
    panels = (
        Panel("Padding", "share of every batch's cells that are padding, 64 lines a batch",
              "bars", dict(values=np.array([shares[k] for k in orders], float),
                           labels=["file order", "shuffled", "sorted"])),
        Panel("Cells", "cells in all the batches, against the characters there are", "bars",
              dict(values=np.array([w[k]["cells"] for k in orders] + [w["characters"]],
                                   float),
                   labels=["file order", "shuffled", "sorted", "characters"])),
        Panel("Padding, batch by batch", "share of each of the %d batches that is padding"
              % n_batches, "curve",
              dict(x=np.arange(1, n_batches + 1, dtype=float),
                   ys=np.array([per[k] for k in orders]),
                   labels=["file order", "shuffled", "sorted"])),
        Panel("Line lengths", "lines of each length, 1 to %d characters" % w["longest"],
              "bars", dict(values=counts[1:].astype(float),
                           labels=[str(i) for i in range(1, w["longest"] + 1)])),
    )
    return Study("Sort by length, pad less", "All the lines in batches of 64: 38% padding in "
                 "file order or shuffled, 0.16% sorted by length", panels, meta)


STUDIES = {"padding-masks": masks, "padding-sort-by-length": sort_by_length}
