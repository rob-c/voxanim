"""The numbers behind animations 236-237 of "Blocks and code" (group ``s``, the last two of
the vector-database part): an index that looks in fewer places, and what happens to the
store when the encoder is trained again. Everything comes from
:mod:`voxanim.study_vectordb`, the numpy twin of ``examples/vector_db.py``.

* ``vector-db-ivf``: spherical k-means cuts the 60,000 stored vectors into 64 lists
  (region "vdb-ivf-build"); the lists are uneven, 89 to 4,761 vectors. A query is compared
  with the 64 centres only, and the vectors of the nearest few lists are scanned (region
  "vdb-ivf-search"). The query followed is test digit 1, a 2, picked by a rule
  (:func:`example_query`: the first test digit whose nearest list does not hold all of its
  10 nearest and whose nearest 4 do): visiting 1, 2 and 4 lists scans 1,923, 3,874 and
  5,119 vectors and finds 7, 8 and 10 of its true 10 nearest. Over all 10,000 test digits
  (:func:`~voxanim.study_vectordb.ivf`), recall@10 against exact search and the share
  scanned: nprobe 1, 80.9% and 2.8%; 4, 99.5% and 8.1%; 64, 100% and 100%. "Scanned"
  counts the vectors compared (dot products), not time.
* ``vector-db-versions``: the same recipe trained again (seed 1) gives test digit 0, a 7,
  a different vector: the two 32-number vectors' dot product is 0.12 (0.08 on average over
  the 60,000 training digits). The retrained encoder's query against the first encoder's
  store finds five 3s (the twin's run: a torch run draws other random numbers, and its
  retrained encoder found 7s for this digit). Over all 10,000 test digits the mixed search
  finds the right digit 7.6% of the time, below chance (10.0%: the chance that a stored
  digit picked at random has the query's label); with the store embedded again by the new
  encoder (region "vdb-reembed"), 96.6%.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_course as course
from . import study_vectordb as D
from .blocks import code_lines, commas, shape
from .study import Panel, Study

SHOWN = (1, 2, 4)               # 236: the lists visited, one step at a time
MARKED = 4                      # 236: the nprobe marked on the curve
VERSION_QUERY = 0               # 237: XT[:1], the script's own query
TOP = 5                         # topk(5), as the script prints
SOURCE = ("MNIST: the 60,000 training digits stored, the 10,000 test digits as queries; the "
          "encoder trained once as examples/vector_db.py trains it (voxanim.study_vectordb, "
          "its numpy twin)")


def pct(share, places=1):
    """``99.5%``."""
    return "%.*f%%" % (places, 100 * share)


# ====================================================================== 236
def visit(t, nprobe, P=None):
    """Test digit ``t`` searched through the IVF lists, as region "vdb-ivf-search" does:
    ``dict(near, scanned, found)`` (the lists visited, the vectors compared, and how many of
    its true 10 nearest are among the 10 nearest of those)."""
    iv = D.ivf()
    v = D.store(0)["vectors"]
    q = D.query(t, 0)
    near = np.argsort(-(iv["centres"] @ q), kind="stable")[:nprobe]
    cand = np.nonzero(np.isin(iv["home"], near))[0]
    got = cand[np.argsort(-(v[cand] @ q), kind="stable")[:D.K]]
    exact = np.argsort(-(v @ q), kind="stable")[:D.K]
    return dict(near=[int(j) for j in near], scanned=int(len(cand)),
                found=int(len(np.intersect1d(got, exact))))


@functools.lru_cache(maxsize=1)
def example_query():
    """The test digit followed through the lists: the first whose nearest list does not
    hold all of its 10 nearest, and whose nearest 4 lists do (a rule, not a search for a
    pretty picture: it is test digit 1)."""
    for t in range(1000):
        if visit(t, 1)["found"] < D.K and visit(t, MARKED)["found"] == D.K:
            return t
    raise ValueError("no test digit fits the rule")


def ivf_index():
    iv = D.ivf()
    sizes = np.asarray(iv["list_sizes"], int)
    n = int(sizes.sum())
    if n != len(D.store(0)["vectors"]):
        raise ValueError("the lists do not hold every stored vector once")
    t = example_query()
    q = D.query(t, 0)
    near = (iv["centres"] @ q).astype(float)
    order = np.argsort(-near, kind="stable")
    steps = [visit(t, k) for k in SHOWN]
    for s in steps:                                    # scanned: the visited lists' sizes
        if s["scanned"] != int(sizes[s["near"]].sum()):
            raise ValueError("the vectors scanned are not the visited lists' sizes")
    nprobe = [int(k) for k in iv["nprobe"]]
    recall = [float(r) for r in iv["recall"]]
    scanned = [float(s) for s in iv["scanned"]]
    at = nprobe.index(MARKED)
    xt = course.data()[2]
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: one k-means run, 10,000 queries",
        lists=int(len(sizes)), stored=n, stored_text=commas(n), sizes=sizes.tolist(),
        smallest=int(sizes.min()), biggest=int(sizes.max()),
        smallest_list=int(sizes.argmin()), biggest_list=int(sizes.argmax()),
        smallest_text=commas(sizes.min()), biggest_text=commas(sizes.max()),
        query=t, query_label=int(course.data()[3][t]), query_image=xt[t].reshape(28, 28).tolist(),
        near=near.tolist(), order=[int(j) for j in order[:max(SHOWN)]],
        near_lo=float(near.min()), near_hi=float(near.max()),
        shown=list(SHOWN),
        steps=[dict(nprobe=k, near=s["near"], scanned=s["scanned"],
                    scanned_text=commas(s["scanned"]), share=s["scanned"] / n,
                    share_text=pct(s["scanned"] / n), found=s["found"])
               for k, s in zip(SHOWN, steps, strict=True)],
        nprobe=nprobe, recall=recall, scanned=scanned,
        recall_text=[pct(r) for r in recall], scanned_text=[pct(s) for s in scanned],
        marked=MARKED, marked_recall_text=pct(recall[at]),
        marked_scanned_text=pct(scanned[at]),
        k=D.K, kmeans_steps=D.KMEANS_STEPS,
        code_build=list(code_lines("vdb-ivf-build")),
        code_search=list(code_lines("vdb-ivf-search")),
        limitations=[
            "recall@10 is against exact search over the same vectors (did the 10 nearest "
            "come back?), averaged over the 10,000 test digits; it says nothing about "
            "whether the neighbours carry the query's label",
            "scanned counts the vectors compared (dot products); the 64 centres are "
            "compared too, and neither count is a time",
            "one k-means run (10 steps, a seeded start); another start gives other lists",
            "the query followed is test digit %d, picked by a rule: the first whose "
            "nearest list misses some of its 10 nearest and whose nearest 4 hold them all"
            % t,
            "at nprobe 64 every vector is scanned, yet recall is 99.997%%, not 100%%: "
            "ties in the scores ordered differently"])
    panels = (
        Panel("List sizes", "vectors in each of the 64 k-means lists", "bars",
              dict(values=sizes.astype(float), labels=[str(j) for j in range(len(sizes))])),
        Panel("A query against the centres", "test digit %d: its dot product with each of "
              "the 64 centres" % t, "bars",
              dict(values=near, labels=[str(j) for j in range(len(near))])),
        Panel("Recall against scanned", "recall@10 against exact search, by the share of "
              "the store scanned, nprobe 1 to 64, over 10,000 test digits", "curve",
              dict(x=np.asarray(scanned, float), ys=np.asarray([recall], float),
                   labels=["recall@10"])),
        Panel("One query, 1, 2 and 4 lists", "test digit %d: vectors scanned, and how many "
              "of its 10 nearest are found" % t, "bars",
              dict(values=np.array([s["scanned"] for s in steps], float),
                   labels=["%d lists: %d found" % (k, s["found"])
                           for k, s in zip(SHOWN, steps, strict=True)])),
    )
    return Study("An index: look in fewer places", "64 k-means lists; visiting the nearest 4 "
                 "finds 99.5% of the true 10 nearest, comparing 8.1% of the vectors",
                 panels, meta)


# ====================================================================== 237
def chance():
    """How often a stored digit picked at random has a test digit's label, averaged over
    the test digits: the sum over the ten digits of their two frequencies multiplied."""
    y, yt = course.data()[1], course.data()[3]
    a = np.bincount(y, minlength=10) / len(y)
    b = np.bincount(yt, minlength=10) / len(yt)
    return float((a * b).sum())


def found(t, seed, store_seed):
    """The ``TOP`` nearest of test digit ``t``: ``dict(ids, labels, scores, images)``."""
    s = D.search(t, TOP, seed=seed, store_seed=store_seed)
    x = course.data()[0]
    return dict(ids=[int(i) for i in s["ids"]], labels=[int(v) for v in s["labels"]],
                scores=[float(v) for v in s["scores"]],
                score_texts=["%.2f" % v for v in s["scores"]],
                images=[x[i].reshape(28, 28).tolist() for i in s["ids"]])


def versions():
    t = VERSION_QUERY
    q, q2 = D.query(t, 0), D.query(t, 1)
    dot = float(q @ q2)
    qual = D.quality()
    n = len(course.data()[3])
    mixed = found(t, 1, 0)
    again = found(t, 1, 1)
    first = found(t, 0, 0)
    label = int(course.data()[3][t])
    rate = dict(same=qual["trained"]["top1"], mixed=qual["mixed"]["top1"], chance=chance(),
                reembedded=qual["reembedded"]["top1"])
    if not rate["mixed"] < rate["chance"] < rate["same"]:
        raise ValueError("the mixed search is not below chance")
    cross = float(D.run()["cross_cos"])
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: 10,000 queries",
        query=t, label=label, image=course.data()[2][t].reshape(28, 28).tolist(),
        q=q.tolist(), q2=q2.tolist(), dims=int(len(q)),
        lo=float(min(q.min(), q2.min())), hi=float(max(q.max(), q2.max())),
        dot=dot, dot_text="%.2f" % dot, self_dot_text="%.2f" % float(q @ q),
        cross=cross, cross_text="%.2f" % cross,
        mixed=mixed, again=again, first=first,
        queries=n, queries_text=commas(n),
        right=int(round(rate["mixed"] * n)), right_text=commas(round(rate["mixed"] * n)),
        rate=rate, rate_text={k: pct(v) for k, v in rate.items()},
        mixed_top10_text=pct(qual["mixed"]["top10"]),
        stored=len(course.data()[0]), stored_text=commas(len(course.data()[0])),
        store_shape=shape(D.store(1)["vectors"].shape),
        code_versions=list(code_lines("vdb-versions")),
        code_reembed=list(code_lines("vdb-reembed")),
        limitations=[
            "one retrained encoder (the same recipe, seed 1): the twin draws its random "
            "numbers from numpy, the script from torch, so a torch run's encoder2 differs; "
            "for test digit 0 the script's own run still found 7s, the twin's finds 3s, "
            "which is why the rate over all 10,000 queries is shown",
            "chance: a stored digit picked at random has the query's label %s of the time "
            "(the ten digits' frequencies); the mixed search is below it" % pct(chance()),
            "top-1: the nearest stored digit has the query's label; top-10 agreement of the "
            "mixed search is %s" % pct(qual["mixed"]["top10"])])
    panels = (
        Panel("One digit, two encoders", "test digit %d's 32 numbers from the first encoder "
              "and from the retrained one" % t, "matrix", dict(values=np.array([q, q2]))),
        Panel("Old store, new query", "the 5 nearest stored digits to the retrained "
              "encoder's query, from the first encoder's store", "images",
              dict(values=np.array(mixed["images"]), labels=[str(v) for v in mixed["labels"]])),
        Panel("Right digit, all queries", "how often the nearest stored digit has the "
              "query's label, over the 10,000 test digits", "bars",
              dict(values=np.array([rate[k] for k in ("same", "mixed", "chance",
                                                      "reembedded")], float),
                   labels=["one encoder", "mixed", "chance", "re-embedded"])),
        Panel("Store embedded again", "the 5 nearest with the store embedded by the "
              "retrained encoder too", "images",
              dict(values=np.array(again["images"]), labels=[str(v) for v in again["labels"]])),
    )
    return Study("A new encoder needs a new store", "A retrained encoder's queries against "
                 "the old store find the right digit 7.6% of the time; re-embedded, 96.6%",
                 panels, meta)


STUDIES = {"vector-db-ivf": ivf_index, "vector-db-versions": versions}
