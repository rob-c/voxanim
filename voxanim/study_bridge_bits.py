"""The experiment behind animation 182, "How many bits a weight carries" (see
:func:`voxanim.next_catalog.redesigned`): the bridge from "The entropy of a network's
weights" (174) into "Quantization and precision".

The trained MNIST classifier of 174 (:mod:`voxanim.study_info`: 784-64-32-10, seed 0,
five epochs of Adam; its cached parameters, nothing retrained here) has its 52,544
weights rounded to each number format of "Training on a CPU and on a GPU"
(:func:`voxanim.study_hardware.to_format`, :func:`~voxanim.study_hardware.to_format_scaled`)
and to symmetric integer grids of 2 to 8 bits. For each, three kinds of number:

* **bits stored a weight** (exact: the format's width, plus a block scale's share);
* **the entropy of what is stored** (measured): ``-sum p log2 p`` over the values (or,
  with a shared scale, the codes) the 52,544 weights take, the fewest bits an ideal code
  would need on average, and how many distinct values are used;
* **held-out accuracy and loss** of the classifier with its weights so stored (biases
  kept in float32, the sums in float32), on the 10,000 test digits.

A number format is a set of bins, as 174's count in bins of 0.01 was; the entropy is
that of the weights counted in the format's bins. Plain numpy.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from . import study_hardware as H
from . import study_info as S
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
EVIDENCE = dict(
    exact="exact arithmetic: the bits each format stores, its grid of values, and a weight "
          "rounded to it",
    measured="measured on MNIST: the trained classifier of 'The entropy of a network's "
             "weights' with its weights rounded, scored on the 10,000 held-out digits",
)
# the formats, widest first: (name, bits of each stored weight, shared scale or not)
FORMATS = ("FP32", "FP16", "BF16", "FP8", "FP4", "FP8 scaled", "FP4 scaled")
INT_BITS = (2, 3, 4, 5, 6, 7, 8)          # symmetric integer grids, one scale per matrix
SCALE_BITS = 8                             # a power-of-two scale stored as its exponent
BLOCK = H.BLOCK                            # weights sharing one scale (32)
HOLD = 0.005                               # "holds": within half a point of float32
BLOCK_ROWS = (384, 416)                    # the block of 32 drawn: pixels 384-415 ...
BLOCK_UNIT = 0                             # ... into hidden unit 0 of the first layer
WEIGHTS = ("W0", "W1", "W2")


def width(fmt):
    """Bits a weight takes in storage: the format's own, plus a shared scale's share."""
    base = H.BITS[fmt.split()[0]]
    return base + (SCALE_BITS / BLOCK if fmt.endswith("scaled") else 0.0)


def code_entropy(codes):
    """``(bits, distinct)``: the entropy ``-sum p log2 p`` of the stored codes as counted
    (``-0`` and ``0`` are one value), and how many different ones there are."""
    c = np.unique(np.asarray(codes, np.float64).ravel() + 0.0, return_counts=True)[1]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum()), int(len(c))


def block_scales(fmt, W, block=BLOCK):
    """The power-of-two scale of every weight of ``W`` under
    :func:`~voxanim.study_hardware.to_format_scaled`: per ``block`` consecutive inputs of
    each output, the one that puts the block's largest weight in the format's top binade."""
    W = np.asarray(W, np.float64)
    emax = int(np.floor(np.log2(H.SMALL[fmt][3])))
    out = np.ones_like(W)
    for s in range(0, W.shape[0], block):
        amax = np.abs(W[s:s + block]).max(0)
        out[s:s + block] = np.where(amax > 0, 2.0 ** (np.floor(np.log2(
            np.where(amax > 0, amax, 1.0))) - emax), 1.0)
    return out


def rounded(fmt, W):
    """``(values, codes, scales)``: ``W`` stored in ``fmt``. Without a shared scale the
    code is the value itself; with one it is the format's value the scale multiplies."""
    W = np.asarray(W, np.float64)
    if fmt.endswith("scaled"):
        base = fmt.split()[0]
        sc = block_scales(base, W)
        codes = H.to_format(base, W / sc)
        return codes * sc, codes, sc
    v = H.to_format(fmt, W)
    return v, v, np.ones_like(W)


def integer_grid(bits, W):
    """``(values, codes)``: ``W`` on the symmetric grid of ``2**bits - 1`` levels
    ``-L .. L`` (``L = 2**(bits-1) - 1``), the matrix's largest weight at ``L``, rounded to
    the nearest level."""
    W = np.asarray(W, np.float64)
    top = 2 ** (bits - 1) - 1
    step = np.abs(W).max() / top
    codes = np.clip(np.round(W / step), -top, top)
    return codes * step, codes


def test_data():
    _, _, xt, yt = mnist.load()
    return xt.astype(np.float32), yt.astype(np.int64)


def score(P, xt, yt):
    """``(accuracy, loss in nats)`` of the classifier on the held-out digits."""
    lp = S.log_softmax(S.logits(P, xt))
    return float((lp.argmax(1) == yt).mean()), float(-lp[np.arange(len(yt)), yt].mean())


def _build():
    P = S.params("after")
    xt, yt = test_data()
    out = {}

    def measure(key, store):
        Q, codes, zeros = dict(P), [], []
        for k in WEIGHTS:
            v, c = store(P[k])
            Q[k] = v.astype(np.float32)
            codes.append(c.ravel())
            zeros.append((v == 0).ravel())
        bits, distinct = code_entropy(np.concatenate(codes))
        acc, loss = score(Q, xt, yt)
        out[key] = np.array([bits, distinct, acc, loss, np.concatenate(zeros).mean()])

    measure("float32", lambda W: (np.asarray(W, np.float64),) * 2)
    for fmt in FORMATS:
        measure(fmt, lambda W, f=fmt: rounded(f, W)[:2])
        if fmt.endswith("scaled"):
            # the scales' own entropy: what an ideal code would spend on them a weight
            e = np.concatenate([np.log2(block_scales(fmt.split()[0], P[k])[::BLOCK]).ravel()
                                for k in WEIGHTS])
            out[fmt + " scale"] = np.array([code_entropy(e)[0] / BLOCK, code_entropy(e)[1]])
    for b in INT_BITS:
        measure("int%d" % b, lambda W, b=b: integer_grid(b, W))
    # what the drawing shows: every weight, and one block of 32 in each form
    out["weights"] = np.concatenate([P[k].ravel() for k in WEIGHTS]).astype(np.float32)
    blk = np.asarray(P["W0"][BLOCK_ROWS[0]:BLOCK_ROWS[1], BLOCK_UNIT], np.float64)
    out["block"] = blk
    out["block_fp4"] = H.to_format("FP4", blk)
    v, c, s = rounded("FP4 scaled", P["W0"][BLOCK_ROWS[0]:BLOCK_ROWS[1], BLOCK_UNIT:
                                           BLOCK_UNIT + 1])
    out["block_fp4_scaled"], out["block_codes"] = v.ravel(), c.ravel()
    out["block_scale"] = np.array(float(s[0, 0]))
    out["histogram_174"] = np.array(S.histogram_entropy(S.flat(P)))
    return out


@functools.lru_cache(maxsize=1)
def runs():
    """Every format measured on the trained classifier (cached)."""
    h = hashlib.sha256()
    for fn in (S.init_mlp, S.logits, S.log_softmax, S.mlp_grads, S.train_mlp,
               S._weights_build, S.histogram_entropy, S.flat, H.small_grid, H._nearest,
               H.to_format, width, code_entropy, block_scales, rounded, integer_grid,
               test_data, score, _build):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], formats=FORMATS,
                    ints=INT_BITS, block=BLOCK, scale_bits=SCALE_BITS, rows=BLOCK_ROWS,
                    unit=BLOCK_UNIT, sizes=S.SIZES, epochs=S.EPOCHS, seed=S.SEEDS[0],
                    weight_seed=S.WEIGHT_SEED)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-bits-runs", settings,
                                                    _build).items()}


def row(r, key):
    bits, distinct, acc, loss, zero = (float(v) for v in r[key])
    return dict(entropy=bits, distinct=int(distinct), accuracy=acc, loss=loss, zeros=zero)


# ====================================================================== the study
def bits_in_a_weight():
    r = runs()
    n = int(r["weights"].size)
    base = row(r, "float32")
    fmts = {f: dict(row(r, f), stored=width(f)) for f in FORMATS}
    for f in FORMATS:
        if f.endswith("scaled"):
            fmts[f]["scale_entropy"] = float(r[f + " scale"][0])
            fmts[f]["scale_values"] = int(r[f + " scale"][1])
    ints = {b: dict(row(r, "int%d" % b), stored=float(b), levels=2 ** b - 1)
            for b in INT_BITS}
    holds = [b for b in INT_BITS if ints[b]["accuracy"] >= base["accuracy"] - HOLD]
    fewest = min((b for b in holds if all(c in holds for c in INT_BITS if c >= b)),
                 default=INT_BITS[-1])
    below = max((b for b in INT_BITS if b < fewest), default=None)
    names = list(FORMATS)
    grid4 = H.small_grid("FP4")
    panels = (
        Panel("Bits stored and bits carried", "bits each of the %s weights takes in each "
              "format (exact) and the entropy of the values it takes there (measured)"
              % format(n, ","), "bars",
              dict(values=np.array([fmts[f]["entropy"] for f in names]),
                   stored=np.array([fmts[f]["stored"] for f in names]),
                   distinct=np.array([fmts[f]["distinct"] for f in names]), labels=names)),
        Panel("Held-out accuracy in each format", "share of the 10,000 test digits named "
              "correctly with the weights stored in each format", "bars",
              dict(values=np.array([fmts[f]["accuracy"] for f in names]),
                   zeros=np.array([fmts[f]["zeros"] for f in names]), labels=names,
                   float32=np.array(base["accuracy"]))),
        Panel("Integer grids: entropy and accuracy", "weights on symmetric grids of 2 to 8 "
              "bits: the entropy of the codes (bits) and held-out accuracy", "curve",
              dict(x=np.array(INT_BITS, float),
                   ys=np.array([[ints[b]["entropy"] for b in INT_BITS],
                                [ints[b]["accuracy"] for b in INT_BITS]]),
                   labels=["entropy of the codes", "held-out accuracy"])),
        Panel("One block of 32 weights", "first-layer weights of pixels %d-%d into hidden "
              "unit %d: as trained, in FP4, and in FP4 with one shared scale"
              % (BLOCK_ROWS[0], BLOCK_ROWS[1] - 1, BLOCK_UNIT), "matrix",
              dict(values=np.stack([r["block"], r["block_fp4"], r["block_fp4_scaled"]]),
                   labels=["trained", "FP4", "FP4 scaled"], codes=r["block_codes"],
                   scale=r["block_scale"])),
    )
    return Study(
        "How many bits a weight carries", "The trained classifier's weights rounded to each "
        "number format: bits stored, bits carried, accuracy kept", panels,
        dict(source="the 784-64-32-10 classifier of 'The entropy of a network's weights' "
                    "(study_info, seed 0, cached), weights rounded in numpy; accuracy on "
                    "the 10,000 MNIST test digits",
             dataset="MNIST", evidence=EVIDENCE, comes_from="The entropy of a network's "
             "weights", leads_to="Quantization",
             sizes=list(S.SIZES), weights=n, epochs=S.EPOCHS, seed=S.SEEDS[0],
             weight_seed=S.WEIGHT_SEED, log2_weights=float(np.log2(n)),
             histogram_174=float(r["histogram_174"]), histogram_bin=S.HIST_BIN,
             float32=base, formats=fmts, ints={str(b): v for b, v in ints.items()},
             block=BLOCK, scale_bits=SCALE_BITS, hold=HOLD, fewest_int_bits=fewest,
             fewest_entropy=ints[fewest]["entropy"], below_int_bits=below,
             fp4_grid=[float(v) for v in grid4],
             block_rows=list(BLOCK_ROWS), block_unit=BLOCK_UNIT,
             block_scale=float(r["block_scale"]),
             block_zero_fp4=int((r["block_fp4"] == 0).sum()),
             block_zero_scaled=int((r["block_fp4_scaled"] == 0).sum()),
             limitations=[
                 "one trained classifier (seed 0): other seeds and models move the last "
                 "digits and may move where accuracy starts to fall",
                 "entropies are counted from the 52,544 weights themselves: a count of N "
                 "values can show at most log2 N bits, so for FP32 (every value different) "
                 "and FP16 the count is a floor, not the format's entropy",
                 "the entropy is what an ideal code would need for these weights' values "
                 "taken one at a time; it ignores which weight sits where, as 174's pile did",
                 "weights only are rounded: biases stay in float32 and the sums are done in "
                 "float32, so only storage is reduced",
                 "the shared scales are counted at %d bits each (a power of two stored as "
                 "its exponent), %.2f bits a weight" % (SCALE_BITS, SCALE_BITS / BLOCK),
                 "the integer grids have one scale per matrix (three float32 numbers, "
                 "negligible a weight) and are rounded after training, with no retraining",
             ]))


STUDIES = {"bits-in-a-weight": bits_in_a_weight}
