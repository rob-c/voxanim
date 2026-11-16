"""Redesigned experiments for three of the vision animations (51-58).

The first experiments (:mod:`voxanim.next_vision`) could not show their concept:

* ``vit-attention``: a two-block, 16-patch transformer after 700 updates attended almost
  evenly. Here: 49 patches of 4 x 4 pixels, three blocks of four heads, three seeds, the whole
  test set, and three measurements of each head (how far it reaches, how many patches it
  spreads over, how much of the class token's reading lands on the strokes).
* ``batchnorm-in-cnns``: normalisation applied after the fact to a CNN trained without it.
  Here: CNNs trained *with* batch normalisation (three seeds), then tested with the stored
  statistics and with each batch's own at batch sizes 256, 32, 8, 2 and 1.
* ``style-transfer``: four 3 x 3 filters cannot hold a style. Here: a 16- and 32-filter MNIST
  CNN, the style read from both layers, and styles that show on a digit (a thin and a heavy
  real digit), measured over twenty content digits.

Everything is plain numpy, seeded, trained on MNIST's training split and measured on its test
split. :data:`STUDIES` is what :func:`voxanim.next_catalog.build_study` looks up.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import realdata
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SEEDS = (0, 1, 2)


def cached(name, build, **settings):
    """``build()``'s arrays, kept on disk under every setting and this file's own text."""
    settings = dict(settings, version=VERSION, source=realdata.source_hash("study_vision.py"))
    return cached_npz("study_vision_" + name, settings, build)


def per_seed(function):
    """``function(seed)`` for every seed, each in its own process when processes can be
    started and one after the other when not. The results are the same either way: every
    run is seeded and shares nothing with the others."""
    try:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(len(SEEDS), multiprocessing.get_context("spawn")) as pool:
            return list(pool.map(function, SEEDS))
    except Exception:       # no semaphores in a sandbox, a worker killed: train in line
        return [function(seed) for seed in SEEDS]


def mnist_data():
    from .next_vision import mnist_data as load

    return load()


def study(title, subtitle, panels, seeds=SEEDS, **metadata):
    return Study(title, subtitle, tuple(panels), dict(
        source="NumPy measurements on MNIST", dataset="MNIST", review_only=True,
        split="trained on the official training images, measured on the official test images",
        seeds=list(seeds), **metadata))


# ====================================================================== a small CNN, by hand
EPS = 1e-5
MOMENTUM = 0.9          # of the stored (running) statistics


def conv_cols(x):
    """``(N, H, W, C)`` -> ``(N, H-2, W-2, 9C)``: every 3 x 3 window."""
    s = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))
    return s.reshape(s.shape[:3] + (-1,))


def cols_back(d, channels):
    """Gradient of :func:`conv_cols`'s output back onto its input."""
    n, h, w, _ = d.shape
    d = d.reshape(n, h, w, channels, 3, 3)
    out = np.zeros((n, h + 2, w + 2, channels))
    for i in range(3):
        for j in range(3):
            out[:, i:i + h, j:j + w] += d[..., i, j]
    return out


def pool(a):
    """2 x 2 mean pooling (an odd last row and column are dropped)."""
    n, h, w, c = a.shape
    return a[:, :h // 2 * 2, :w // 2 * 2].reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4))


def pool_back(d, shape):
    out = np.zeros(shape)
    _n, h, w, _c = d.shape
    out[:, :h * 2, :w * 2] = d.repeat(2, 1).repeat(2, 2) / 4
    return out


def cnn_init(rng, filters, norm, pad):
    side, p, c = 28, {}, 1
    for i, f in enumerate(filters, 1):
        p["W%d" % i] = rng.normal(0, np.sqrt(2 / (9 * c)), (9 * c, f))
        p["b%d" % i] = np.zeros(f)
        if norm:
            p["gamma%d" % i], p["beta%d" % i] = np.ones(f), np.zeros(f)
        side, c = (side if pad else side - 2) // 2, f
    p["Wd"] = rng.normal(0, np.sqrt(1 / (side * side * c)), (side * side * c, 10))
    p["bd"] = np.zeros(10)
    return p


def cnn_forward(p, x, stats=None, batch=None, pad=False):
    """The CNN on images ``x`` ``(N, 784)`` or ``(N, 28, 28, 1)``: 3 x 3 convolution,
    (batch normalisation,) ReLU and 2 x 2 mean pooling, as many times as ``p`` has ``W<i>``,
    then one dense layer. Returns a dict of every intermediate.

    With batch normalisation (``gamma<i>`` in ``p``): ``stats`` (``mean<i>``, ``var<i>``)
    are the stored statistics, used for every image; without them each run of ``batch``
    consecutive images (default: all of ``x``) is normalised with its own."""
    f = dict(x=np.asarray(x, float).reshape(-1, 28, 28, 1))
    a, i = f["x"], 1
    while "W%d" % i in p:
        if pad:
            a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)))
        cols = conv_cols(a)
        z = cols @ p["W%d" % i] + p["b%d" % i]
        f["cols%d" % i], f["z%d" % i] = cols, z
        if "gamma%d" % i in p:
            if stats is not None:
                mu, var = stats["mean%d" % i], stats["var%d" % i]
            else:
                g = z.reshape((-1, batch or len(z)) + z.shape[1:])
                mu = np.repeat(g.mean((1, 2, 3)), g.shape[1], 0)[:, None, None]
                var = np.repeat(g.var((1, 2, 3)), g.shape[1], 0)[:, None, None]
            f["mu%d" % i], f["var%d" % i] = mu, var
            f["hat%d" % i] = (z - mu) / np.sqrt(var + EPS)
            z = p["gamma%d" % i] * f["hat%d" % i] + p["beta%d" % i]
        f["a%d" % i] = np.maximum(z, 0)
        a = f["p%d" % i] = pool(f["a%d" % i])
        i += 1
    f["flat"] = a.reshape(len(a), -1)
    f["logits"] = f["flat"] @ p["Wd"] + p["bd"]
    return f


def cnn_backward(p, f, d_logits=None, d_maps=None, pad=False):
    """Gradients of a scalar through :func:`cnn_forward`'s dict ``f`` (one batch, its own
    statistics): from ``d_logits`` and/or ``d_maps`` (``{layer: dL/da<layer>}``). Returns the
    parameter gradients and ``x`` (dL/d pixels)."""
    g = {}
    layers = sum(1 for k in p if k.startswith("W") and k[1:].isdigit())
    last = f["p%d" % layers]
    da = np.zeros_like(last)
    if d_logits is not None:
        g["Wd"], g["bd"] = f["flat"].T @ d_logits, d_logits.sum(0)
        da = (d_logits @ p["Wd"].T).reshape(last.shape)
    for i in range(layers, 0, -1):
        d = pool_back(da, f["a%d" % i].shape)
        if d_maps is not None and i in d_maps:
            d = d + d_maps[i]
        d = d * (f["a%d" % i] > 0)
        if "gamma%d" % i in p:
            hat, m = f["hat%d" % i], d[..., 0].size
            g["gamma%d" % i], g["beta%d" % i] = (d * hat).sum((0, 1, 2)), d.sum((0, 1, 2))
            dh = d * p["gamma%d" % i]
            d = (dh - dh.mean((0, 1, 2)) - hat * (dh * hat).sum((0, 1, 2)) / m) / np.sqrt(
                f["var%d" % i] + EPS)
        cols = f["cols%d" % i]
        g["W%d" % i] = cols.reshape(-1, cols.shape[-1]).T @ d.reshape(-1, d.shape[-1])
        g["b%d" % i] = d.sum((0, 1, 2))
        da = cols_back(d @ p["W%d" % i].T, cols.shape[-1] // 9)
        if pad:
            da = da[:, 1:-1, 1:-1]
    g["x"] = da
    return g


def cnn_train(seed, filters, norm, pad, steps, batch=64, lr=0.003, trace=0):
    """Train on MNIST's 60,000 training images. Returns ``(p, stats, history)``: ``stats`` the
    running mean and variance of each normalised layer (empty without ``norm``), ``history``
    the loss and, for the first ``trace`` steps, layer 1's batch and running means."""
    x, y = mnist_data()[:2]
    rng = np.random.default_rng(seed)
    p = cnn_init(rng, filters, norm, pad)
    opt = Adam(p, lr)
    stats = {}
    for i, c in enumerate(filters, 1):
        if norm:
            stats["mean%d" % i], stats["var%d" % i] = np.zeros(c), np.ones(c)
    history = dict(loss=[], batch_mean=[], running_mean=[], batch_var=[], running_var=[])
    for step in range(steps):
        ids = rng.choice(len(x), batch, replace=False)
        f = cnn_forward(p, x[ids], pad=pad)
        z = f["logits"] - f["logits"].max(-1, keepdims=True)
        pr = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
        history["loss"].append(float(-np.log(pr[np.arange(batch), y[ids]] + 1e-12).mean()))
        pr[np.arange(batch), y[ids]] -= 1
        g = cnn_backward(p, f, pr / batch, pad=pad)
        del g["x"]
        opt.step(p, g, lr * 0.5 * (1 + np.cos(np.pi * step / steps)))
        for k in stats:
            now = f[("mu" if k.startswith("mean") else "var") + k[-1]][0].reshape(-1)
            stats[k] = MOMENTUM * stats[k] + (1 - MOMENTUM) * now
        if norm and step < trace:
            history["batch_mean"].append(f["mu1"][0].reshape(-1))
            history["running_mean"].append(stats["mean1"].copy())
            history["batch_var"].append(f["var1"][0].reshape(-1))
            history["running_var"].append(stats["var1"].copy())
    return p, stats, {k: np.array(v) for k, v in history.items()}


def cnn_predict(p, x, stats=None, batch=None, pad=False, chunk=1024):
    """Predicted digits, ``chunk`` images at a time (``chunk`` is a multiple of ``batch``)."""
    return np.concatenate([cnn_forward(p, x[i:i + chunk], stats, batch, pad)["logits"].argmax(1)
                           for i in range(0, len(x), chunk)])


# ====================================================================== 57 batch norm
BN_FILTERS = (8, 16)
BN_STEPS = 1000
BN_BATCHES = (256, 32, 8, 2, 1)
BN_TEST = 9984              # 39 batches of 256: every batch size divides it
BN_TRACE = 200


def bn_test_set():
    """The test images every batch size is measured on: the first 9,984 of a fixed shuffle,
    so that a batch holds a mixture of digits."""
    xt, yt = mnist_data()[2:]
    order = np.random.default_rng(57).permutation(len(xt))[:BN_TEST]
    return xt[order], yt[order], order


def bn_seed(seed):
    """One CNN trained with batch normalisation, and its test accuracy under each statistic."""
    x, y, _ = bn_test_set()
    by_digit = np.argsort(y, kind="stable")
    p, stats, history = cnn_train(seed, BN_FILTERS, True, False, BN_STEPS, trace=BN_TRACE)
    base = cnn_predict(p, x, stats)
    preds = [cnn_predict(p, x, batch=b) for b in BN_BATCHES]
    sorted_pred = cnn_predict(p, x[by_digit], batch=32)
    return dict(p=p, stats=stats, history=history, stored=np.mean(base == y),
                own=[np.mean(q == y) for q in preds],
                changed=[np.mean(q != base) for q in preds],
                one_digit=np.mean(sorted_pred == y[by_digit]),
                one_digit_changed=np.mean(sorted_pred != base[by_digit]))


@lru_cache(maxsize=1)
def batchnorm_run():
    """Three CNNs trained with batch normalisation, and their test accuracy with the stored
    statistics and with each test batch's own. The first seed's weights, stored statistics
    and training trace are kept for the drawing."""
    def build():
        runs = per_seed(bn_seed)
        keys = ("stored", "own", "changed", "one_digit", "one_digit_changed")
        out = dict(order=bn_test_set()[2], **{k: [r[k] for r in runs] for k in keys})
        out.update({"weights_" + k: v for k, v in runs[0]["p"].items()})
        out.update({"stats_" + k: v for k, v in runs[0]["stats"].items()})
        out.update(runs[0]["history"])
        return out

    return cached("batchnorm", build, filters=list(BN_FILTERS), steps=BN_STEPS, batch=64,
                  lr=0.003, momentum=MOMENTUM, seeds=list(SEEDS), test=BN_TEST)


def part(d, prefix):
    return {k[len(prefix):]: np.asarray(v) for k, v in d.items() if k.startswith(prefix)}


def batchnorm():
    d = batchnorm_run()
    stored, own = np.asarray(d["stored"], float), np.asarray(d["own"], float)
    changed = np.asarray(d["changed"], float)
    accuracy = np.column_stack((stored, own))                     # seed, [stored, 256 .. 1]
    labels = ["stored"] + ["batch of %d" % b for b in BN_BATCHES]
    stats = part(d, "stats_")
    steps = np.arange(1, BN_TRACE + 1)
    return study(
        "Batchnorm in cnns",
        "CNNs trained with batch normalisation, tested with stored and with batch statistics",
        [
            Panel("Stored statistics of the first layer",
                  "Running mean and variance of each of eight filters after training.",
                  "matrix", dict(values=np.vstack((stats["mean1"], stats["var1"])))),
            Panel("The stored mean follows the batches",
                  "Filter 1 over the first 200 updates: each batch's mean and the running mean.",
                  "curve", dict(x=steps, ys=np.vstack((d["batch_mean"][:, 0],
                                                       d["running_mean"][:, 0])),
                                labels=["batch mean", "stored (running) mean"])),
            Panel("Test accuracy by whose statistics are used",
                  "Mean of three seeds on 9,984 test images; batches are mixed digits.",
                  "bars", dict(values=accuracy.mean(0), labels=labels, seeds=accuracy),
                  ("one digit per batch of 32: %.1f%%" % (100 * np.mean(d["one_digit"])),)),
            Panel("Predictions that change",
                  "Share of test images read differently than with the stored statistics.",
                  "bars", dict(values=changed.mean(0), labels=labels[1:], seeds=changed)),
        ],
        scope="Two-layer CNN (8 and 16 filters, batch normalisation after each convolution), "
              "1,000 updates of 64 images; three seeds",
        model="conv 3x3 x8 - BN - ReLU - mean pool - conv 3x3 x16 - BN - ReLU - mean pool - "
              "dense 10",
        updates=BN_STEPS, test_images=BN_TEST, batch_sizes=list(BN_BATCHES),
        accuracy_stored=[float(v) for v in stored],
        accuracy_batch_of_one=[float(v) for v in own[:, -1]],
        control="the same trained network and the same test images under every statistic",
        designs_tried=["two convolution layers, each batch-normalised over batch and positions "
                       "(the only design run)"],
    )


# ====================================================================== 58 style transfer
STYLE_FILTERS = (16, 32)
STYLE_TRAIN = 1000
STYLE_STEPS = 300
STYLE_LR = 0.05             # the largest change of a pixel in one step
CONTENT_WEIGHT = 3.0
STYLE_NAMES = ("thin", "heavy")
CONTENT = 20                 # the first twenty test digits
STYLE_POOL = (100, 300)      # the styles are the lightest and heaviest of these test digits
SNAP = 10


def stroke_weight(v):
    """Ink per unit of edge, in pixels: about half the width of the strokes."""
    v = np.asarray(v, float).reshape(-1, 28, 28)
    edge = np.abs(np.diff(v, axis=1)).sum((1, 2)) + np.abs(np.diff(v, axis=2)).sum((1, 2))
    return v.sum((1, 2)) / edge


def grams(f):
    """The Gram matrix of each layer's feature maps, per image: ``{layer: (N, C, C)}``."""
    out = {}
    for layer in (1, 2):
        a = f["a%d" % layer]
        flat = a.reshape(len(a), -1, a.shape[-1])
        out[layer] = np.einsum("npi,npj->nij", flat, flat) / flat.shape[1]
    return out


def style_objective(p, z, target, content, weight=CONTENT_WEIGHT):
    """Style mismatch (per layer, the squared difference of Gram matrices as a share of the
    style's own), the content term and the gradient of ``mismatch + weight * content`` with
    respect to the pixels ``z``. Returns ``(mismatch (N,), content (N,), gradient, f)``."""
    f = cnn_forward(p, z, pad=True)
    mismatch, back = 0.0, {}
    for layer, g in grams(f).items():
        a = f["a%d" % layer]
        flat = a.reshape(len(a), -1, a.shape[-1])
        diff, norm = g - target[layer], float((target[layer] ** 2).sum())
        mismatch = mismatch + (diff ** 2).sum((1, 2)) / norm
        back[layer] = (4 / flat.shape[1] * flat @ diff / norm).reshape(a.shape)
    gap = f["a2"] - content
    back[2] = back[2] + weight * 2 * gap / gap[0].size
    grad = cnn_backward(p, f, None, back, pad=True)["x"]
    return mismatch, (gap ** 2).mean((1, 2, 3)), grad, f


def style_optimise(p, content, style, steps=STYLE_STEPS):
    """Gradient descent with momentum on the pixels of every content image at once, from
    the content image itself; each image's step is scaled so that its fastest pixel moves by
    :data:`STYLE_LR`, which leaves pixels with next to no gradient where they are. Returns
    the final images, the per-step mismatch and content term (``(steps + 1, N)``), snapshots
    every :data:`SNAP` steps and the network's reading of the result."""
    target = grams(cnn_forward(p, style, pad=True))
    keep = cnn_forward(p, content, pad=True)["a2"]
    z = np.asarray(content, float).reshape(-1, 28, 28, 1).copy()
    m = np.zeros_like(z)
    mismatch, kept, snaps = [], [], []
    for step in range(steps + 1):
        mis, con, g, f = style_objective(p, z, target, keep)
        mismatch.append(mis)
        kept.append(con)
        if step % SNAP == 0:
            snaps.append(z[..., 0].copy())
        if step == steps:
            break
        m = 0.9 * m + 0.1 * g
        z = np.clip(z - STYLE_LR * m / np.abs(m).max((1, 2, 3), keepdims=True), 0, 1)
    return z[..., 0], np.array(mismatch), np.array(kept), np.array(snaps), f["logits"].argmax(1)


@lru_cache(maxsize=1)
def style_run():
    def build():
        xt, yt = mnist_data()[2:]
        p, _, history = cnn_train(SEEDS[0], STYLE_FILTERS, False, True, STYLE_TRAIN)
        content, labels = xt[:CONTENT], yt[:CONTENT]
        pool_ = xt[STYLE_POOL[0]:STYLE_POOL[1]]
        weight = stroke_weight(pool_)
        picks = [int(weight.argmin()), int(weight.argmax())]
        out = dict(
            accuracy=np.mean(cnn_predict(p, xt, pad=True, chunk=1000) == yt),
            train_loss=history["loss"], content=content.reshape(-1, 28, 28), labels=labels,
            read_before=cnn_predict(p, content, pad=True),
            style=pool_[picks].reshape(-1, 28, 28),
            style_index=np.array(picks) + STYLE_POOL[0], style_label=yt[STYLE_POOL[0]:][picks],
            result=[], mismatch=[], kept=[], snaps=[], read_after=[], gram1=[], gram2=[],
            **{"weights_" + k: v for k, v in p.items()})
        for pick in picks:
            style = pool_[pick:pick + 1]
            z, mismatch, kept, snaps, read = style_optimise(p, content, style)
            target = grams(cnn_forward(p, style, pad=True))
            out["result"].append(z)
            out["mismatch"].append(mismatch)
            out["kept"].append(kept)
            out["snaps"].append(snaps)
            out["read_after"].append(read)
            out["gram1"].append(target[1][0])
            out["gram2"].append(target[2][0])
        return out

    return cached("style", build, filters=list(STYLE_FILTERS), train_steps=STYLE_TRAIN,
                  steps=STYLE_STEPS, lr=STYLE_LR, content_weight=CONTENT_WEIGHT,
                  content=CONTENT, pool=list(STYLE_POOL), seed=SEEDS[0])


def style_transfer():
    d = style_run()
    mismatch = np.asarray(d["mismatch"], float)               # style, step, digit
    result, content = np.asarray(d["result"], float), np.asarray(d["content"], float)
    ink = lambda v: np.asarray(v, float).reshape(len(v), -1).sum(1)     # noqa: E731
    kept = [int(np.sum(r == d["read_before"])) for r in d["read_after"]]
    return study(
        "Style transfer",
        "Pixels of twenty digits optimised to match a thin and a heavy digit's Gram matrices",
        [
            Panel("Content and the two styles",
                  "A test digit and the lightest and heaviest of test digits 100-299.",
                  "images", dict(values=np.stack((content[0], d["style"][0], d["style"][1])),
                                 labels=["content", "thin style", "heavy style"])),
            Panel("The heavy style's Gram matrix, layer 1",
                  "Mean product of each pair of the sixteen first-layer feature maps.",
                  "matrix", dict(values=np.asarray(d["gram1"][1], float))),
            Panel("One digit in two styles",
                  "The content digit, and its pixels after 300 steps towards each style.",
                  "images", dict(values=np.stack((content[0], result[0, 0], result[1, 0])),
                                 labels=["content", "thin", "heavy"])),
            Panel("Gram mismatch while the pixels move",
                  "Mean over twenty content digits; squared difference as a share of the "
                  "style's own Gram matrices, two layers added.",
                  "curve", dict(x=np.arange(mismatch.shape[1]), ys=mismatch.mean(2),
                                labels=list(STYLE_NAMES)),
                  tuple("%s: read as the same digit %d / %d" % (n, k, CONTENT)
                        for n, k in zip(STYLE_NAMES, kept))),
        ],
        seeds=SEEDS[:1],
        scope="Two-layer MNIST CNN (16 and 32 filters, 1,000 updates); Gram matrices of both "
              "layers; twenty content digits and two style digits, all from the test split",
        cnn_test_accuracy=float(d["accuracy"]),
        style_digits=[int(v) for v in d["style_index"]],
        ink=dict(content=float(ink(content).mean()),
                 **{n + "_style": float(ink(d["style"][i:i + 1])[0])
                    for i, n in enumerate(STYLE_NAMES)},
                 **{n + "_result": float(ink(result[i]).mean())
                    for i, n in enumerate(STYLE_NAMES)}),
        stroke_weight=dict(content=float(stroke_weight(content).mean()),
                           **{n + "_style": float(stroke_weight(d["style"][i])[0])
                              for i, n in enumerate(STYLE_NAMES)},
                           **{n + "_result": float(stroke_weight(result[i]).mean())
                              for i, n in enumerate(STYLE_NAMES)}),
        mismatch_start=[float(v) for v in mismatch[:, 0].mean(1)],
        mismatch_end=[float(v) for v in mismatch[:, -1].mean(1)],
        read_as_same_digit=kept,
        control="the same twenty digits and network under two opposite styles; the start "
                "(the content digit itself) is the baseline",
        designs_tried=[
            "outlined and twice-dilated renderings of test digit 18 as styles, content weight "
            "0.3 and 3: the outline came out as speckle, not an outline; dropped",
            "thin and heavy real digits as styles, content weight 3 (kept)",
            "Adam on the pixels: it moves every pixel at the same pace and covered the paper "
            "in speckle; replaced by gradient descent with momentum (kept)",
            "the same with content weight 10, with a smoothness penalty (0.004, 0.02), and "
            "with the style read from one layer only: not kept (weaker or blurred)",
            "Gram matrices divided by their trace before comparing (the mix of textures "
            "without the amount of ink): streaks and soft blobs, fewer digits kept; not kept",
        ],
    )


# ====================================================================== 52 ViT attention
GRID = 7                    # 7 x 7 patches of 4 x 4 pixels
WIDTH, HEADS, BLOCKS, HIDDEN = 32, 4, 3, 64
VIT_STEPS = 3000
VIT_BATCH = 64
VIT_LR = 0.003
VIT_PROBE = 500             # test digits whose attention is measured
VIT_SHOWN = 8               # test digits whose attention is kept for the drawing
INK = 0.5                   # a patch "has ink" if any of its pixels is at least this dark


def vit_tokens(x):
    """``(N, 784)`` -> ``(N, 50, 16)``: a blank row for the class token, then the 49 patches
    of 4 x 4 pixels in reading order."""
    t = np.asarray(x).reshape(-1, GRID, 4, GRID, 4).transpose(0, 1, 3, 2, 4)
    return np.pad(t.reshape(-1, GRID * GRID, 16), ((0, 0), (1, 0), (0, 0)))


def vit_init(rng):
    p = dict(E=rng.normal(0, 0.25, (16, WIDTH)), pos=rng.normal(0, 0.1, (GRID * GRID + 1, WIDTH)),
             U=rng.normal(0, 1 / np.sqrt(WIDTH), (WIDTH, 10)))
    for i in range(BLOCKS):
        for name, shape in (("Q", (WIDTH, WIDTH)), ("K", (WIDTH, WIDTH)), ("V", (WIDTH, WIDTH)),
                            ("O", (WIDTH, WIDTH)), ("A", (WIDTH, HIDDEN)), ("B", (HIDDEN, WIDTH))):
            p[name + str(i)] = rng.normal(0, 1 / np.sqrt(shape[0]), shape) * (
                0.5 if name in "OB" else 1.0)
    return p


def vit_loss(p, x, y, grads=True):
    """A pre-norm vision transformer: ``(loss, gradients, logits, attention)``, attention as
    ``(block, image, head, from, to)`` over the 50 tokens (token 0 is the class token, whose
    embedding is its position's: its patch is blank)."""
    from .tinylm import Var, softmax

    def norm(t):
        c = t - t.mean(-1, True)
        return c / ((c ** 2).mean(-1, True) + 1e-5) ** 0.5

    ps = {k: Var(v) for k, v in p.items()}
    n, tokens = len(x), GRID * GRID + 1
    tok = Var(vit_tokens(x)) @ ps["E"] + ps["pos"]
    attention = []
    for i in range(BLOCKS):
        h = norm(tok)
        q, k, v = ((h @ ps[m + str(i)]).reshape(n, tokens, HEADS, WIDTH // HEADS).transpose(
            0, 2, 1, 3) for m in "QKV")
        a = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(WIDTH // HEADS))
        attention.append(a.v)
        tok = tok + (a @ v).transpose(0, 2, 1, 3).reshape(n, tokens, WIDTH) @ ps["O" + str(i)]
        tok = tok + (norm(tok) @ ps["A" + str(i)]).tanh() @ ps["B" + str(i)]
    logits = norm(tok[:, 0]) @ ps["U"]
    shifted = logits - logits.v.max(-1, keepdims=True)
    logp = shifted - shifted.exp().sum(-1, True).log()
    loss = -logp[np.arange(n), y].sum() / n
    if grads:
        loss.backward()
    return float(loss.v), {k: v.g for k, v in ps.items()}, logits.v, np.array(
        attention)


def patch_distance():
    """Distance between every pair of the 49 patches, in patch widths."""
    at = np.array([(i // GRID, i % GRID) for i in range(GRID * GRID)], float)
    return np.linalg.norm(at[:, None] - at[None], axis=-1)


def has_ink(x):
    """``(N, 49)``: which patches of each image hold a stroke."""
    return vit_tokens(x)[:, 1:].max(-1) >= INK


def attention_measures(attention, ink):
    """Per block and head, averaged over images (``attention``: block, image, head, from, to):

    * ``reach``: how far a patch's attention to the other patches travels (patch widths),
    * ``spread``: over how many patches a patch's attention is spread (exp of its entropy),
    * ``cls_ink``: the share of the class token's attention to patches that lands on patches
      with ink, and ``cls_spread``: how many patches the class token reads."""
    def spread(w):
        return np.exp(-(w * np.log(w + 1e-12)).sum(-1))

    among = attention[:, :, :, 1:, 1:]
    among = among / among.sum(-1, keepdims=True)
    cls = attention[:, :, :, 0, 1:]
    cls = cls / cls.sum(-1, keepdims=True)
    return dict(reach=(among * patch_distance()).sum(-1).mean((1, 3)),
                spread=spread(among).mean((1, 3)),
                cls_ink=(cls * ink[None, :, None]).sum(-1).mean(1),
                cls_spread=spread(cls).mean(1))


def vit_seed(seed):
    """One transformer trained on MNIST's training images: its test accuracy, the measures
    of its attention over the first :data:`VIT_PROBE` test digits, and what the drawing needs."""
    x, y, xt, yt = mnist_data()
    rng = np.random.default_rng(seed)
    p = vit_init(rng)
    opt, losses = Adam(p, VIT_LR), []
    for step in range(VIT_STEPS):
        ids = rng.choice(len(x), VIT_BATCH, replace=False)
        loss, g, _, _ = vit_loss(p, x[ids], y[ids])
        opt.step(p, g, VIT_LR * min(1.0, (step + 1) / 100) * 0.5 * (
            1 + np.cos(np.pi * step / VIT_STEPS)))
        losses.append(loss)
    read = np.concatenate([vit_loss(p, xt[i:i + 500], yt[i:i + 500], False)[2].argmax(1)
                           for i in range(0, len(xt), 500)])
    attention = vit_loss(p, xt[:VIT_PROBE], yt[:VIT_PROBE], False)[3]
    return dict(p=p, loss=np.array(losses), accuracy=np.mean(read == yt),
                read=read[:VIT_SHOWN], attention=attention[:, :VIT_SHOWN],
                **attention_measures(attention, has_ink(xt[:VIT_PROBE])))


@lru_cache(maxsize=1)
def vit_run():
    """Three transformers (one per seed); the first one's weights and its attention on the
    first :data:`VIT_SHOWN` test digits are kept for the drawing."""
    def build():
        xt, yt = mnist_data()[2:]
        runs = per_seed(vit_seed)
        keys = ("accuracy", "loss", "reach", "spread", "cls_ink", "cls_spread")
        out = dict(ink_share=has_ink(xt[:VIT_PROBE]).mean(), example=xt[:VIT_SHOWN],
                   label=yt[:VIT_SHOWN], attention=runs[0]["attention"], read=runs[0]["read"],
                   **{k: [r[k] for r in runs] for k in keys})
        out.update({"weights_" + k: v for k, v in runs[0]["p"].items()})
        return out

    return cached("vit", build, grid=GRID, width=WIDTH, heads=HEADS, blocks=BLOCKS,
                  hidden=HIDDEN, steps=VIT_STEPS, batch=VIT_BATCH, lr=VIT_LR, probe=VIT_PROBE,
                  ink=INK, seeds=list(SEEDS))


def vit_attention():
    d = vit_run()
    reach, cls_ink = np.asarray(d["reach"], float), np.asarray(d["cls_ink"], float)
    spread, cls_spread = np.asarray(d["spread"], float), np.asarray(d["cls_spread"], float)
    uniform = float(patch_distance().mean())
    blocks = ["block %d" % (i + 1) for i in range(BLOCKS)]
    accuracy = [float(v) for v in d["accuracy"]]
    cls = d["attention"][:, 0, :, 0, 1:]                       # block, head, patch: first digit
    return study(
        "Vit attention",
        "A three-block vision transformer on 49 patches of each MNIST digit, three seeds",
        [
            Panel("Forty-nine patches become tokens",
                  "A test digit cut into 4 x 4-pixel patches; a class token is put in front.",
                  "images", dict(values=vit_tokens(d["example"][:1])[0, 1:].reshape(49, 4, 4),
                                 labels=[str(i + 1) for i in range(49)])),
            Panel("What the class token reads in the last block",
                  "Rows: four heads; columns: the 49 patches of the digit shown.",
                  "matrix", dict(values=cls[-1] / cls[-1].sum(-1, keepdims=True))),
            Panel("How far attention between patches travels",
                  "Patch widths, mean of heads and seeds over 500 test digits; last bar: "
                  "attention spread evenly (exact).",
                  "bars", dict(values=np.r_[reach.mean((0, 2)), uniform],
                               labels=blocks + ["even"], heads=reach.mean(0),
                               seeds=reach.mean(2)),
                  ("patches a patch spreads its attention over: " + ", ".join(
                      "%.1f" % v for v in spread.mean((0, 2))) + " of 49",)),
            Panel("How many patches the class token reads",
                  "Patches its attention is spread over (exp of the entropy), mean of heads "
                  "and seeds over 500 test digits; last bar: read evenly.",
                  "bars", dict(values=np.r_[cls_spread.mean((0, 2)), 49.0],
                               labels=blocks + ["even"], heads=cls_spread.mean(0),
                               seeds=cls_spread.mean(2), on_ink=cls_ink.mean((0, 2)),
                               on_ink_heads=cls_ink, ink_share=float(d["ink_share"])),
                  ("test accuracy %.1f%% (%.1f%% to %.1f%%)" % (
                      100 * np.mean(accuracy), 100 * min(accuracy), 100 * max(accuracy)),
                   "share of it on patches with ink: " + ", ".join(
                       "%.0f%%" % (100 * v) for v in cls_ink.mean((0, 2))) + "; read evenly %.0f%%"
                   % (100 * float(d["ink_share"])))),
        ],
        scope="Width-32 ViT, three blocks of four heads, 49 patch tokens and a class token; "
              "3,000 updates of 64 images; accuracy on all 10,000 test images, attention on "
              "the first 500",
        updates=VIT_STEPS, test_accuracy=accuracy, uniform_reach=uniform,
        ink_patch_share=float(d["ink_share"]),
        control="attention spread evenly over the 49 patches: its reach (exact) and its share "
                "on ink patches (the share of patches that have ink)",
        finding="attention is concentrated but not local: its reach equals that of attention "
                "spread evenly, in every block, head and seed",
        designs_tried=["49 tokens, width 32, three blocks, four heads: 1,500 updates of 64 on "
                       "three seeds first (95.0% accuracy), then 3,000 (kept: better trained)",
                       "one seed of it for 2,000 and for 6,000 updates of 64, and two seeds for "
                       "3,000 updates of 32, to see whether more training makes attention "
                       "local: reach stayed within 0.2 of even in every head"],
        caveat="attention weights are not causal explanations",
    )


STUDIES = {
    "vit-attention": vit_attention,
    "batchnorm-in-cnns": batchnorm,
    "style-transfer": style_transfer,
}
