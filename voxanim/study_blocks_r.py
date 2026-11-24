"""The numbers behind animations 234-235, the first two of "Blocks and code: vector databases"
(group ``r``): how the vectors are trained, and what a store and an exact search are. Every
number comes from :mod:`voxanim.study_vectordb`, the numpy twin of ``examples/vector_db.py``
(regions ``vdb-encoder``, ``vdb-loss``, ``vdb-step``, ``vdb-store``, ``vdb-search``).

* ``vector-db-training``: the encoder 784 -> 128 -> 32 (104,608 numbers), trained for 938
  Adam steps of 64 pairs, each a training digit and another drawing of the same digit
  (labels pick the pairs and nothing else). Step 0's first pair, ``x[0]`` and ``p[0]`` (two
  5s), through the encoder before training: two vectors of 32 numbers, length 1, whose dot
  product is ``grid[0, 0]`` of step 0's [64, 64] grid. The grid at steps 0, 100 and 937
  (:data:`~voxanim.study_vectordb.SNAPS`), its diagonal the target of
  ``F.cross_entropy``; the right pairs' mean against the other 63's: 0.59 against 0.49 at
  step 0, 0.93 against 0.36 at step 937; the loss of each grid's batch. Step 937's grid
  sorted by digit (rows and columns alike, so the pairs stay on the diagonal): the
  in-batch negatives that are the same digit (false negatives) score 0.92, almost the
  pairs' 0.93, the other digits 0.29.
* ``vector-db-store-and-search``: the store, the 60,000 training digits' vectors [60000, 32]
  with ids and labels beside them (rows 0 to 4 and the last drawn); test digit 0 (the
  script's ``XT[:1]``, a 7) encoded to a query; its 60,000 dot products; the top 5, their
  ids, labels and scores, and their pictures looked up by id. Then the quality over all
  10,000 test digits: how often the nearest stored digit, and the 10 nearest, have the
  query's label, for the trained encoder (96.4%, 96.0%), the raw 784 pixels (97.0%, 95.4%)
  and the same network untrained (87.0%, 81.9%): 32 numbers do about as well as 784 (top 1,
  pixels are a little better).

One run, one seed. The twin's random batches and pairs come from numpy, the script's from
torch, so a run of the script trains on other pairs and finds other ids (the equivalence of
the lines is ``tests/test_vectordb_equivalence.py``).
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from . import study_vectordb as vdb
from .blocks import code_lines, commas, shape, val
from .study import Panel, Study

SIDE = 28
PAIR = 0                        # the pair followed: the first of step 0's batch
SNAPS = vdb.SNAPS               # the grid at steps 0, 100 and 937
QUERY = 0                       # test digit 0: the script's XT[:1]
SHOWN = (0, 1, 2, 3, 4)         # the store's first rows drawn, then the last
TOP = 5
SOURCE = ("examples/vector_db.py through its numpy twin voxanim.study_vectordb: MNIST, the "
          "encoder 784-128-32 trained with InfoNCE, one run (seed 0)")
TWIN = ("the twin draws its random batches and partners with numpy, the script with torch: a "
        "run of the script trains on other pairs and finds other ids; the lines are the same "
        "(tests/test_vectordb_equivalence.py)")


def two(v):
    """``0.93``, a minus sign as in the rest of the series."""
    return val(v, 2)


def pct(v, places=1):
    return "%.*f%%" % (places, 100 * float(v))


def image(row):
    """A normalised 784 row as a 28 x 28 square."""
    return np.asarray(row, float).reshape(SIDE, SIDE)


def by_digit(grid, y):
    """``grid`` with its rows and columns both put in the order of the labels ``y`` (a
    stable sort, so each row keeps its own partner on the diagonal), and the order."""
    order = np.argsort(np.asarray(y), kind="stable")
    return np.asarray(grid)[order][:, order], order


def split_means(grid, y):
    """Means of the grid's diagonal (the right pairs), of the other cells (the other 63 of
    each row), and of those split into the same digit and other digits."""
    g = np.asarray(grid, float)
    y = np.asarray(y)
    diag = np.eye(len(g), dtype=bool)
    same = (y[:, None] == y[None, :]) & ~diag
    return dict(pos=float(g[diag].mean()), neg=float(g[~diag].mean()),
                same=float(g[same].mean()), other=float(g[~diag & ~same].mean()),
                same_cells=int(same.sum()))


# ====================================================================== 234
def training():
    h = vdb.history(0)
    b0 = vdb.batch(0)
    P0 = vdb.params(0, "start")
    a = vdb.unit(vdb.encode(P0, b0["x"][PAIR:PAIR + 1]))[0]
    b = vdb.unit(vdb.encode(P0, b0["p"][PAIR:PAIR + 1]))[0]
    dot = float(a @ b)
    if not np.isclose(dot, h["grids"][0][PAIR, PAIR], atol=1e-5):
        raise ValueError("the pair's dot product is not the grid's corner")
    snaps = {}
    for s in SNAPS:
        bs = vdb.batch(s)
        g = np.asarray(h["grids"][s], float)
        m = split_means(g, bs["y"])
        if not (np.isclose(m["pos"], h["pos"][s], atol=1e-5)
                and np.isclose(m["neg"], h["neg"][s], atol=1e-5)):
            raise ValueError("step %d: the grid's means are not the history's" % s)
        sorted_grid, order = by_digit(g, bs["y"])
        counts = np.bincount(bs["y"], minlength=10)
        snaps[s] = dict(
            grid=g.tolist(), sorted=sorted_grid.tolist(), order=order.tolist(),
            labels=[int(v) for v in bs["y"]], counts=[int(c) for c in counts],
            x=image(bs["x"][PAIR]).tolist(), p=image(bs["p"][PAIR]).tolist(),
            label=int(bs["y"][PAIR]), i=int(bs["i"][PAIR]), j=int(bs["j"][PAIR]),
            loss=float(h["loss"][s]), loss_text=two(h["loss"][s]),
            **m, pos_text=two(m["pos"]), neg_text=two(m["neg"]), same_text=two(m["same"]),
            other_text=two(m["other"]))
    s0, s1 = snaps[SNAPS[0]], snaps[SNAPS[-1]]
    loss = np.asarray(h["loss"], float)
    meta = dict(
        source=SOURCE, twin=TWIN,
        evidence="measured on MNIST: one run of the twin, seed 0",
        parameters=vdb.count(), parameters_text=commas(vdb.count()),
        sizes=list(vdb.SIZES), batch=vdb.BATCH, steps=vdb.STEPS, tau=vdb.TEMPERATURE,
        snaps=list(SNAPS), snap=snaps,
        pair_label=s0["label"], pair_i=s0["i"], pair_j=s0["j"],
        x=s0["x"], p=s0["p"], a=a.tolist(), b=b.tolist(), dot=dot, dot_text=two(dot),
        a_norm=float(np.linalg.norm(a)), b_norm=float(np.linalg.norm(b)),
        grid_shape=[vdb.BATCH, vdb.BATCH], grid_shape_text=shape((vdb.BATCH, vdb.BATCH)),
        others=vdb.BATCH - 1,
        first_loss=float(loss[0]), first_loss_text=two(loss[0]),
        chance=float(np.log(vdb.BATCH)), chance_text=two(np.log(vdb.BATCH)),
        last_100=float(loss[-100:].mean()), last_100_text=two(loss[-100:].mean()),
        pos_end=s1["pos"], neg_end=s1["neg"], same_end=s1["same"], other_end=s1["other"],
        code_loss=list(code_lines("vdb-loss")), code_encoder=list(code_lines("vdb-encoder")),
        code_step=list(code_lines("vdb-step")),
        limitations=[
            TWIN,
            "each grid is its own step's batch: the 64 pairs at step 100 and 937 are other "
            "digits than at step 0; each grid's loss is that one batch's",
            "the grid is shaded from 0 (paper) to 1 (full blue); a few dot products below 0 "
            "are drawn as paper",
            "labels only pick the pairs: an in-batch negative can be the same digit (a "
            "false negative), and the loss pushes it away like any other"])
    steps = np.arange(vdb.STEPS, dtype=float)
    panels = (
        Panel("Loss", "InfoNCE over each step's 64 pairs (temperature 0.1); ln 64 = %s is a "
              "guess" % two(np.log(vdb.BATCH)), "curve",
              dict(x=steps, ys=loss[None], labels=["loss"])),
        Panel("Dot products", "mean over each step's grid: the right pairs (the diagonal) "
              "and the other 63 of each row", "curve",
              dict(x=steps, ys=np.array([h["pos"], h["neg"]], float),
                   labels=["right pairs", "the other 63"])),
        Panel("The grid at step 0", "a @ b.T for step 0's 64 pairs, before training",
              "matrix", dict(values=np.asarray(s0["grid"]))),
        Panel("The grid at step 937, by digit", "step 937's grid, rows and columns sorted by "
              "label: the same digit off the diagonal scores %s" % s1["same_text"], "matrix",
              dict(values=np.asarray(s1["sorted"]))),
    )
    return Study("Training vectors for search", "Pairs of two drawings of one digit: the "
                 "diagonal of the [64, 64] grid brightens to %s against %s"
                 % (s1["pos_text"], s1["neg_text"]), panels, meta)


# ====================================================================== 235
def printed(ids, labels):
    """What ``print(found, labels[found])`` writes: torch pads the ints to one width."""
    def tensor(v):
        w = max(len(str(int(x))) for x in v)
        return "tensor([%s])" % ", ".join(str(int(x)).rjust(w) for x in v)
    return "%s %s" % (tensor(ids), tensor(labels))


def wrapped(text, width=39):
    """``text`` broken after its last comma that keeps the first line within ``width`` (the
    output slot is 41 characters with its "→ "), the rest lined up under the numbers."""
    if len(text) <= width:
        return text
    cut = text.rfind(",", 0, width) + 1
    return text[:cut] + "\n" + " " * len("tensor([") + text[cut:].lstrip()


def store_and_search():
    x, _y, xt, yt = course.data()
    st = vdb.store(0)
    vec, ids, labels = st["vectors"], st["ids"], st["labels"]
    n, dim = vec.shape
    q = vdb.query(QUERY)
    scores = vec @ q
    r = vdb.search(QUERY, TOP)
    top = ids[np.asarray(r["ids"])]                 # found = ids[top.indices]
    if not np.array_equal(top, np.argsort(-scores, kind="stable")[:TOP]):
        raise ValueError("the top 5 are not the 5 highest scores")
    rows = list(SHOWN) + [n - 1]
    qual = vdb.quality()
    text = printed(top, labels[top])
    meta = dict(
        source=SOURCE, twin=TWIN,
        evidence="measured on MNIST: the twin's trained encoder (seed 0)",
        store_shape=[n, dim], store_shape_text=shape((n, dim)), stored=n,
        stored_text=commas(n), dim=dim, tests=len(xt), tests_text=commas(len(xt)),
        first_image=image(x[0]).tolist(),
        rows=rows, row_labels=[int(labels[i]) for i in rows],
        row_vectors=vec[rows].tolist(), row_scores=[float(scores[i]) for i in rows],
        row_score_texts=[two(scores[i]) for i in rows],
        vec_lo=float(min(vec[rows].min(), q.min())), vec_hi=float(max(vec[rows].max(), q.max())),
        query=QUERY, query_label=int(yt[QUERY]), query_image=image(xt[QUERY]).tolist(),
        q=q.tolist(), q_norm=float(np.linalg.norm(q)),
        top=[int(i) for i in top], top_labels=[int(v) for v in r["labels"]],
        top_scores=[float(s) for s in r["scores"]],
        top_score_texts=["%.3f" % s for s in r["scores"]],
        top_images=[image(x[i]).tolist() for i in top],
        score_mean=float(scores.mean()), score_min=float(scores.min()),
        same_label_mean=float(scores[labels == yt[QUERY]].mean()),
        other_label_mean=float(scores[labels != yt[QUERY]].mean()),
        printed=text, printed_wrapped=wrapped(text),
        quality=qual,
        quality_texts={k: dict(top1=pct(v["top1"]), top10=pct(v["top10"]))
                       for k, v in qual.items()},
        pixels=SIDE * SIDE, smaller=SIDE * SIDE / dim,
        code_store=list(code_lines("vdb-store")), code_search=list(code_lines("vdb-search")),
        limitations=[
            TWIN,
            "quality: for each of the 10,000 test digits as a query, exact search of the "
            "60,000 stored vectors; top 1: the nearest has the query's label; top 10: the "
            "share of the 10 nearest that do. The script searches one query; the loop over "
            "10,000 is the twin's",
            "raw pixels: the 784 normalised pixels scaled to length 1, searched the same way; "
            "untrained: the same network before training",
            "test digit 0 is the script's XT[:1]; its top 5 are not picked for how they look"])
    names = ("trained", "pixels", "untrained")
    hist, edges = np.histogram(scores, bins=20, range=(-1.0, 1.0))
    panels = (
        Panel("Top 10 with the query's label", "share of the 10 nearest stored digits with the "
              "query's label, over the 10,000 test digits", "bars",
              dict(values=np.array([qual[k]["top10"] for k in names], float),
                   labels=["32 numbers, trained", "784 pixels", "32 numbers, untrained"])),
        Panel("Top 1 with the query's label", "share of the 10,000 test digits whose nearest "
              "stored digit has their label", "bars",
              dict(values=np.array([qual[k]["top1"] for k in names], float),
                   labels=["32 numbers, trained", "784 pixels", "32 numbers, untrained"])),
        Panel("Test digit 0's top 5", "the 5 stored digits with the highest scores, by id",
              "images", dict(values=np.array([image(x[i]) for i in top]),
                             labels=["%d: %d, %.3f" % (i, labels[i], scores[i])
                                     for i in top])),
        Panel("Its 60,000 scores", "how many stored digits score in each band, from -1 to 1",
              "bars", dict(values=hist.astype(float),
                           labels=["%.1f" % e for e in edges[:-1]])),
    )
    return Study("Store, then search", "60,000 vectors of 32 with ids and labels; a query is "
                 "60,000 dot products; over 10,000 queries the 10 nearest share its label "
                 "%s of the time" % pct(qual["trained"]["top10"]), panels, meta)


STUDIES = {"vector-db-training": training, "vector-db-store-and-search": store_and_search}
