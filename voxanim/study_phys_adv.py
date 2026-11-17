"""The experiments behind animations 154-156: what a model was never shown, a generator
that stops covering its data, and a background with no formula (see
:func:`voxanim.next_catalog.redesigned`).

* ``autoencoder-anomaly-detection``: an autoencoder trained on MNIST digits with one
  class removed entirely. The removed class plays the part of new physics; the nine
  others are the known processes. Held-out digits are scored by how badly they are
  rebuilt; the area under the ROC curve says how well that score separates the unseen
  class, for each choice of the removed class, seven code sizes and three seeds.
* ``gan-mode-collapse``: a small GAN on MNIST in two schedules that differ in one
  number (generator updates per discriminator update), and the same two networks told
  the label. A separately trained classifier reads 1,000 samples at points along
  training: which digits are being made, and the entropy of that histogram.
* ``gp-background-fit``: a falling mass spectrum drawn from a stated model (two
  exponentials; no dataset). An exponential fitted by maximum likelihood cannot follow
  it; a Gaussian process on what the fit leaves can. 4,000 pseudo-experiments, with and
  without an injected resonance, at 13 kernel length scales, each beside the same
  procedure applied to the expected counts (no fluctuation).

Everything the drawings need is recorded here: :mod:`voxanim.mech_phys_adv` computes
nothing but layout.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import autoencoder as AE
from . import gp
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-phys-adv-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


def entropy_bits(p):
    """Shannon entropy of a distribution, in bits (0 log 0 = 0)."""
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _stat(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


# ====================================================================== 154: anomalies
AE_SPLIT_SEED = 17
AE_TRAIN, AE_TEST = 20000, 5000   # training digits (before one class is removed); held out
AE_SEEDS = (1, 2, 3)
AE_CODES = (2, 4, 8, 16, 32, 64, 128)
AE_HIDDEN, AE_STEPS, AE_BATCH, AE_LR = 256, 2000, 128, 1e-3
AE_PER_CLASS = 2                  # held-out digits per class whose rebuilt pictures are kept
AE_FEATURED = 0                   # the removed class the pictures follow: fixed beforehand
AE_KEEP = 0.95                    # the threshold lets this share of seen digits pass
AE_BINS = 36                      # of the error histograms
ROC_GRID = np.linspace(0.0, 1.0, 101)


@functools.lru_cache(maxsize=1)
def ae_data():
    """``(x, y, xt, yt, meta)``: MNIST training digits and held-out digits (the official
    test partition), 28 x 28, as float32."""
    from . import realdata

    x, y, m = realdata.mnist_split(AE_TRAIN, "train", 28, AE_SPLIT_SEED)
    xt, yt, mt = realdata.mnist_split(AE_TEST, "test", 28, AE_SPLIT_SEED)
    keep = lambda d: {k: v for k, v in d.items() if k != "example_indices"}  # noqa: E731
    return (x.astype(np.float32), y, xt.astype(np.float32), yt,
            dict(train=keep(m), held_out=keep(mt)))


def ae_shown_ids(yt):
    """The held-out digits whose pictures are kept from every run: the first
    :data:`AE_PER_CLASS` of each class, in class order."""
    return np.concatenate([np.flatnonzero(yt == c)[:AE_PER_CLASS] for c in range(10)])


def ae_init(code, rng):
    """``784 -> 256 -> code -> 256 -> 784``: the code is the only narrow place."""
    P = {}
    widths = (784, AE_HIDDEN, code, AE_HIDDEN, 784)
    for n in range(4):
        AE.dense_init(P, str(n), widths[n], widths[n + 1], rng)
    return {k: v.astype(np.float32) for k, v in P.items()}


def ae_forward(P, x):
    """``(hidden, code, hidden, output)``: ReLU, a linear code, ReLU, sigmoid pixels."""
    h0 = np.maximum(x @ P["W0"] + P["b0"], 0)
    z = h0 @ P["W1"] + P["b1"]
    h2 = np.maximum(z @ P["W2"] + P["b2"], 0)
    return h0, z, h2, AE.sigmoid(h2 @ P["W3"] + P["b3"])


def ae_grads(P, x):
    """Gradients of the mean squared pixel error of rebuilding ``x``."""
    h0, z, h2, r = ae_forward(P, x)
    d = AE.recon_grad(r, x).astype(np.float32)
    g = {"W3": h2.T @ d, "b3": d.sum(0)}
    d = (d @ P["W3"].T) * (h2 > 0)
    g["W2"], g["b2"] = z.T @ d, d.sum(0)
    d = d @ P["W2"].T
    g["W1"], g["b1"] = h0.T @ d, d.sum(0)
    d = (d @ P["W1"].T) * (h0 > 0)
    g["W0"], g["b0"] = x.T @ d, d.sum(0)
    return g


def rebuild_error(out, x):
    """The anomaly score: the mean over the 784 pixels of (output - input) squared."""
    return ((np.asarray(out, float) - np.asarray(x, float)) ** 2).mean(-1)


def auc(seen, unseen):
    """Area under the ROC curve: the chance that a random unseen digit scores higher than
    a random seen one (a tie counts half). Exact, by counting pairs."""
    s = np.sort(np.asarray(seen, float))
    u = np.asarray(unseen, float)
    below = np.searchsorted(s, u, "left") + np.searchsorted(s, u, "right")
    return float(below.mean() / (2 * len(s)))


def roc(seen, unseen, grid=ROC_GRID):
    """The share of unseen digits flagged at each false-alarm rate of ``grid``: the
    threshold is the seen scores' quantile that lets that share of them through."""
    thr = np.quantile(np.asarray(seen, float), 1.0 - np.asarray(grid))
    return (np.asarray(unseen, float)[None, :] > thr[:, None]).mean(1)


def ae_job(left, code, seed):
    """One autoencoder trained without class ``left``: every held-out digit's error, the
    rebuilt pictures of the kept digits, and of the two typical ones (the seen and the
    unseen held-out digit whose errors are the medians of their groups)."""
    x, y, xt, yt, _ = ae_data()
    rng = np.random.default_rng([seed, left, code, 154])
    xs = x[y != left]
    P = ae_init(code, rng)
    opt = Adam(P, AE_LR)
    for _ in range(AE_STEPS):
        opt.step(P, ae_grads(P, xs[rng.integers(0, len(xs), AE_BATCH)]))
    out = ae_forward(P, xt)[3]
    err = rebuild_error(out, xt)
    typical = []
    for group in (yt != left, yt == left):
        ids = np.flatnonzero(group)
        typical.append(ids[np.argsort(err[ids], kind="stable")[len(ids) // 2]])
    typical = np.array(typical)
    return (err.astype(np.float32), out[ae_shown_ids(yt)].astype(np.float32), typical,
            out[typical].astype(np.float32),
            float(rebuild_error(ae_forward(P, xs[:2000])[3], xs[:2000]).mean()))


def ae_plan():
    return [(left, code, seed) for code in AE_CODES for left in range(10) for seed in AE_SEEDS]


def _ae_build():
    out = _parallel(ae_job, ae_plan())
    shape = (len(AE_CODES), 10, len(AE_SEEDS))

    def stack(k):
        return np.array([o[k] for o in out]).reshape(shape + np.shape(out[0][k]))
    return dict(err=stack(0), shown=stack(1)[:, :, 0], typical=stack(2), typical_out=stack(3),
                train_mse=stack(4))


def ae_arrays():
    """Every run's held-out errors ``err[code, left, seed, digit]`` and kept pictures."""
    return _cache("anomaly", _ae_build,
                  (ae_init, ae_forward, ae_grads, rebuild_error, ae_job, ae_plan, _ae_build,
                   ae_shown_ids),
                  train=AE_TRAIN, test=AE_TEST, split_seed=AE_SPLIT_SEED, seeds=AE_SEEDS,
                  codes=AE_CODES, hidden=AE_HIDDEN, steps=AE_STEPS, batch=AE_BATCH, lr=AE_LR,
                  per_class=AE_PER_CLASS)


@functools.lru_cache(maxsize=1)
def ae_summary():
    """The areas ``auc[code, left, seed]``, the code size with the best mean area (the
    one the pictures use), and each run's mean error on seen and unseen held-out digits."""
    d = ae_arrays()
    yt = ae_data()[3]
    shape = d["err"].shape[:3]
    area, seen_mse, unseen_mse = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for c, left, s in np.ndindex(shape):
        e = d["err"][c, left, s]
        seen, unseen = e[yt != left], e[yt == left]
        area[c, left, s] = auc(seen, unseen)
        seen_mse[c, left, s], unseen_mse[c, left, s] = seen.mean(), unseen.mean()
    return dict(auc=area, seen_mse=seen_mse, unseen_mse=unseen_mse,
                main=int(area.mean((1, 2)).argmax()))


def autoencoder_anomaly_detection():
    d = ae_arrays()
    x, y, xt, yt, meta = ae_data()
    s = ae_summary()
    area, main = s["auc"], s["main"]
    code = AE_CODES[main]
    left = AE_FEATURED
    err = d["err"][main, left, 0].astype(float)
    seen, unseen = err[yt != left], err[yt == left]
    thr = float(np.quantile(seen, AE_KEEP))
    edges = np.linspace(0.0, float(np.quantile(err, 0.995)), AE_BINS + 1)
    hist = np.array([np.histogram(np.minimum(v, edges[-1]), edges)[0] / len(v)
                     for v in (seen, unseen)])
    typ = d["typical"][main, left, 0]
    curves = np.array([[roc(d["err"][main, c, k][yt != c], d["err"][main, c, k][yt == c])
                        for k in range(len(AE_SEEDS))] for c in range(10)])
    per_class = area[main]                                # [left, seed]
    order = np.argsort(-per_class.mean(1), kind="stable")
    hardest = int(order[-1])
    typ_hard = d["typical"][main, hardest, 0]
    shown = ae_shown_ids(yt)
    pick = int(np.flatnonzero(yt[shown] == left)[0])      # a kept digit of the featured class
    below = int((per_class.mean(1) < 0.5).sum())
    return Study(
        "Anomaly detection with an autoencoder",
        "An autoencoder trained on MNIST with one digit class removed; held-out digits "
        "scored by how badly they are rebuilt: 10 removed classes x %d code sizes x %d seeds"
        % (len(AE_CODES), len(AE_SEEDS)),
        (
            Panel("A seen digit and an unseen one, in and out",
                  "values: a held-out digit of a class the model trained on and one of the "
                  "removed class (%d), each with what the autoencoder (code size %d, first "
                  "seed) returns; both are the digits whose error is the median of their "
                  "group. errors: their mean squared pixel errors; hardest_*: the same for "
                  "the removed class with the lowest area." % (left, code), "images",
                  dict(values=np.array([xt[typ[0]], d["typical_out"][main, left, 0, 0],
                                        xt[typ[1]], d["typical_out"][main, left, 0, 1]],
                                       float).reshape(4, 28, 28),
                       labels=["seen: in", "seen: out", "unseen: in", "unseen: out"],
                       errors=err[typ], classes=yt[typ], ids=typ,
                       train_examples=np.array([x[np.flatnonzero(y == c)[0]] for c in
                                                range(10)], float).reshape(10, 28, 28),
                       hardest_values=np.array(
                           [xt[typ_hard[1]], d["typical_out"][main, hardest, 0, 1]],
                           float).reshape(2, 28, 28),
                       hardest_error=float(d["err"][main, hardest, 0][typ_hard[1]]),
                       hardest_seen_error=float(d["err"][main, hardest, 0][typ_hard[0]]))),
            Panel("Rebuilding error of held-out digits",
                  "values: the share of seen (nine trained classes) and of unseen (class "
                  "%d) held-out digits in each bin of error (code size %d, first seed; "
                  "the last bin holds everything above it). threshold: the error %d%% of "
                  "seen digits stay under." % (left, code, round(100 * AE_KEEP)), "matrix",
                  dict(values=hist, labels=["seen", "unseen"], edges=edges, threshold=thr,
                       flagged=float((unseen > thr).mean()),
                       false_alarm=float((seen > thr).mean()),
                       counts=np.array([len(seen), int((seen > thr).sum()), len(unseen),
                                        int((unseen > thr).sum())]),
                       medians=np.array([np.median(seen), np.median(unseen)]),
                       seen=seen, unseen=unseen)),
            Panel("ROC curve for each removed class",
                  "x: the share of seen digits flagged (false alarms); ys[c]: the share of "
                  "unseen digits flagged when class c was removed, mean of %d seeds (code "
                  "size %d). auc[c, seed]: the area under each run's curve, counted exactly "
                  "over pairs." % (len(AE_SEEDS), code), "curve",
                  dict(x=ROC_GRID, ys=curves.mean(1), labels=[str(c) for c in range(10)],
                       seeds=curves, auc=per_class, order=order)),
            Panel("Area against code size",
                  "x: numbers in the code; ys[c]: area for removed class c, mean of seeds; "
                  "mean: over classes and seeds. seen_mse / unseen_mse: mean error of the "
                  "two groups. pictures: one held-out digit of class %d rebuilt at every "
                  "code size (first seed)." % left, "curve",
                  dict(x=np.array(AE_CODES), ys=area.mean(2).T,
                       labels=[str(c) for c in range(10)], mean=area.mean((1, 2)),
                       seeds=area, seen_mse=s["seen_mse"].mean((1, 2)),
                       unseen_mse=s["unseen_mse"].mean((1, 2)),
                       featured_seen_mse=s["seen_mse"][:, left].mean(1),
                       featured_unseen_mse=s["unseen_mse"][:, left].mean(1),
                       pictures=d["shown"][:, left, pick].reshape(-1, 28, 28).astype(float),
                       picture_in=xt[shown[pick]].reshape(28, 28).astype(float),
                       picture_errors=d["err"][:, left, 0, shown[pick]].astype(float))),
        ),
        dict(
            mnist=True, review_only=True, dataset=meta, source="NumPy training on MNIST",
            evidence="measured on MNIST: every number is from held-out digits",
            seeds=list(AE_SEEDS), codes=list(AE_CODES), code_shown=code, featured=left,
            model="784 -> %d ReLU -> code (linear) -> %d ReLU -> 784 sigmoid; Adam %g, batch "
                  "%d, %d updates on the training digits of the nine kept classes"
                  % (AE_HIDDEN, AE_HIDDEN, AE_LR, AE_BATCH, AE_STEPS),
            score="mean over 784 pixels of (output - input)^2",
            split_sizes=[AE_TRAIN, AE_TEST],
            removed_from_training=[int((y == c).sum()) for c in range(10)],
            held_out_per_class=[int((yt == c).sum()) for c in range(10)],
            threshold=thr, flagged=float((unseen > thr).mean()),
            false_alarm=float((seen > thr).mean()),
            auc_featured=_stat(per_class[left]),
            auc_by_class={str(c): _stat(per_class[c]) for c in range(10)},
            auc_by_code={str(c): float(v) for c, v in zip(AE_CODES, area.mean((1, 2)))},
            easiest=int(order[0]), hardest=hardest, classes_below_chance=below,
            train_mse_by_code={str(c): float(v)
                               for c, v in zip(AE_CODES, d["train_mse"].mean((1, 2)))},
            counterpart="A model-independent search: an autoencoder is trained on known "
                        "processes (simulated or measured background) and events it rebuilds "
                        "badly are kept as candidates, with no signal model. Here the nine "
                        "kept digit classes are the known processes and the removed class "
                        "is the new physics.",
            limitations="The code size shown is the one with the highest mean area over "
                        "all removed classes and seeds, chosen on the held-out digits "
                        "themselves: a real search has no anomalies to tune on, so this "
                        "is the favourable case. The score is rebuilding error alone; it "
                        "measures how hard a picture is to compress, not how new it is, "
                        "which is why a simple unseen shape can score below the seen "
                        "ones. One small dense architecture, %d updates, MNIST only."
                        % AE_STEPS,
            designs_tried=[
                "a one-seed probe (code sizes 2, 8, 32, 128; removed classes 0, 1, 4, 8) "
                "to time a run and see that the area depends on both: it did, and the "
                "class 1 came out below one half",
                "the design recorded here, fixed after that probe: all ten removed "
                "classes, seven code sizes, three seeds; the featured class (0) fixed "
                "before the full run",
            ],
        ),
    )


# ====================================================================== 155: mode collapse
GAN_SEEDS = (1, 2, 3, 4, 5)
GAN_CODE = 16                     # noise numbers the generator starts from
GAN_STEPS, GAN_EVERY = 3000, 100  # discriminator updates; a reading every so many
GAN_HEAVY = 3                     # generator updates per discriminator update when "heavy"
GAN_FIXED = 1000                  # fixed noise points read at every snapshot
GAN_KEPT = 20                     # of them, the pictures kept at every snapshot
GAN_LABEL = 4.0                   # the value of the requested label's input (others 0)
GAN_LAST = 5                      # snapshots averaged for "the end of training"
GAN_COLLAPSED = 2.0               # bits: below this at the end, a run counts as collapsed
# (name, generator updates per discriminator update, told the label)
GAN_SETTINGS = (("balanced", 1, False), ("generator-heavy", GAN_HEAVY, False),
                ("balanced, with label", 1, True),
                ("generator-heavy, with label", GAN_HEAVY, True))


def gan_steps():
    """The discriminator updates at which the generator is read."""
    return np.arange(0, GAN_STEPS + 1, GAN_EVERY)


def gan_onehot(labels):
    """Ten label inputs per sample: :data:`GAN_LABEL` at the requested digit."""
    return GAN_LABEL * np.eye(10)[np.asarray(labels, int)]


def gan_init(told, rng):
    """``(G, D)``: the generator ``16 (+10) -> 64 -> 256 -> 784`` and the discriminator
    ``784 (+10) -> 256 -> 64 -> 1`` of :mod:`voxanim.study_families`, each with ten more
    inputs when it is told the label."""
    extra = 10 if told else 0
    G, D = {}, {}
    for k, (a, b) in zip(AE.DECODER, ((GAN_CODE + extra, 64), (64, 256), (256, 784))):
        AE.dense_init(G, str(k), a, b, rng)
    for name, a, b in (("a", 784 + extra, 256), ("b", 256, 64), ("c", 64, 1)):
        AE.dense_init(D, name, a, b, rng)
    return G, D


def gan_generate(G, z, labels=None):
    """The generator's layer activities for noise ``z`` (and labels, if it is told):
    the last is the pictures."""
    zin = z if labels is None else np.hstack((z, gan_onehot(labels)))
    return AE.dense_layers(G, zin, AE.DECODER)[0]


def gan_update_d(G, D, do, real, real_labels, z, labels, rng):
    """One discriminator update (real against generated; real targets smoothed, dropout,
    as :func:`voxanim.study_families.gan_step`). Labels are ``None`` when not told."""
    from . import study_families as F

    n = len(real)
    xx = np.vstack((real, gan_generate(G, z, labels)[-1]))
    if labels is not None:
        xx = np.hstack((xx, gan_onehot(np.r_[real_labels, labels])))
    target = np.r_[np.full(n, F.GAN_SMOOTH), np.zeros(n)][:, None]
    logit, tape = F.judge_forward(D, xx, rng)
    do.step(D, F.judge_backward(D, tape, (AE.sigmoid(logit) - target) / len(xx))[0])


def gan_update_g(G, D, go, z, labels, rng):
    """One generator update (non-saturating loss): the gradient at the picture comes
    back through the discriminator; the label inputs' part of it is dropped."""
    from . import study_families as F

    acts = gan_generate(G, z, labels)
    fake = acts[-1]
    xin = fake if labels is None else np.hstack((fake, gan_onehot(labels)))
    logit, tape = F.judge_forward(D, xin, rng)
    _, dx = F.judge_backward(D, tape, (AE.sigmoid(logit) - 1.0) / len(z), grads=False)
    g = {}
    AE.dense_backward(G, acts, dx[:, :784] * fake * (1 - fake), AE.DECODER, g)
    go.step(G, g)


def gan_job(index, seed):
    """One GAN in setting ``index``: at every snapshot the reader's histogram of 1,000
    fixed samples, the share it reads as the label asked for (asked of the labelled
    generators; a fixed list the unlabelled ones never see, so chance for them), its
    mean confidence, and :data:`GAN_KEPT` of the pictures."""
    from . import study_families as F

    _name, heavy, told = GAN_SETTINGS[index]
    d = F.data()
    x, y = d["x"], d["y"]
    rng = np.random.default_rng([seed, index, 155])
    G, D = gan_init(told, rng)
    go, do = F.GanAdam(G, F.GAN_LR_G), F.GanAdam(D, F.GAN_LR_D)
    fixed = rng.normal(size=(GAN_FIXED, GAN_CODE))
    asked = np.arange(GAN_FIXED) % 10
    at = set(gan_steps().tolist())
    hist, hit, conf, kept = [], [], [], []

    def draw():
        return (rng.normal(size=(F.GAN_BATCH, GAN_CODE)),
                rng.integers(0, 10, F.GAN_BATCH) if told else None)
    for it in range(GAN_STEPS + 1):
        if it in at:
            img = gan_generate(G, fixed, asked if told else None)[-1]
            p = F.read(img)
            hist.append(np.bincount(p.argmax(1), minlength=10) / GAN_FIXED)
            hit.append(np.bincount(asked[p.argmax(1) == asked], minlength=10) / (GAN_FIXED / 10))
            conf.append(p.max(1).mean())
            kept.append(np.round(255 * img[:GAN_KEPT]).astype(np.uint8))
        if it == GAN_STEPS:
            break
        ids = rng.integers(0, len(x), F.GAN_BATCH)
        gan_update_d(G, D, do, x[ids], y[ids], *draw(), rng)
        for _ in range(heavy):
            gan_update_g(G, D, go, *draw(), rng)
    return np.array(hist), np.array(hit), np.array(conf), np.array(kept)


def gan_plan():
    return [(k, seed) for k in range(len(GAN_SETTINGS)) for seed in GAN_SEEDS]


def _gan_build():
    from . import study_families as F

    read_right = F.reader()[1]                       # trained (or loaded) before the workers
    d = F.data()
    out = _parallel(gan_job, gan_plan())
    shape = (len(GAN_SETTINGS), len(GAN_SEEDS))

    def stack(k):
        return np.array([o[k] for o in out]).reshape(shape + np.shape(out[0][k]))
    real = F.read(d["xt"]).argmax(1)
    return dict(hist=stack(0), hit=stack(1), conf=stack(2), kept=stack(3),
                real=np.bincount(real, minlength=10) / len(real),
                real_labels=np.bincount(d["yt"], minlength=10) / len(d["yt"]),
                reader_test=np.array(read_right), reader_held_out=np.mean(real == d["yt"]))


def gan_arrays():
    """Every run's readings: ``hist[setting, seed, snapshot, digit]`` and so on."""
    from . import study_families as F

    return _cache("collapse", _gan_build,
                  (gan_onehot, gan_init, gan_generate, gan_update_d, gan_update_g, gan_job,
                   gan_plan, _gan_build, F.judge_forward, F.judge_backward, F.GanAdam),
                  seeds=GAN_SEEDS, code=GAN_CODE, steps=GAN_STEPS, every=GAN_EVERY,
                  heavy=GAN_HEAVY, fixed=GAN_FIXED, kept=GAN_KEPT, label=GAN_LABEL,
                  settings=GAN_SETTINGS, train=F.TRAIN_N, test=F.TEST_N, batch=F.GAN_BATCH,
                  lr=[F.GAN_LR_G, F.GAN_LR_D], smooth=F.GAN_SMOOTH, drop=F.GAN_DROP,
                  beta1=F.GAN_BETA1)


def gan_summary():
    """Per run and snapshot: the entropy of the histogram (bits), the digits covered (read
    in at least 3% of samples), the commonest digit and the share read as asked; per
    run: entropy at the end (mean of the last snapshots) and whether it collapsed."""
    from . import study_families as F

    d = gan_arrays()
    hist = d["hist"]
    H = np.array([[[entropy_bits(h) for h in run] for run in setting] for setting in hist])
    end = H[:, :, -GAN_LAST:].mean(2)
    return dict(entropy=H, covered=(hist >= F.COVERED).sum(3), top=hist.argmax(3),
                top_share=hist.max(3), asked=d["hit"].mean(3), end=end,
                collapsed=end < GAN_COLLAPSED,
                covered_end=(hist >= F.COVERED).sum(3)[:, :, -GAN_LAST:].mean(2),
                asked_end=d["hit"].mean(3)[:, :, -GAN_LAST:].mean(2))


def gan_mode_collapse():
    from . import study_families as F

    d = gan_arrays()
    s = gan_summary()
    steps = gan_steps()
    names = [n for n, _h, _t in GAN_SETTINGS]
    H, end = s["entropy"], s["end"]
    kept = d["kept"].astype(float) / 255.0                    # [setting, seed, snap, 20, 784]
    last = kept[:, 0, -1].reshape(len(names), GAN_KEPT, 28, 28)
    data = F.data()

    def by(a):
        return {n: _stat(v) for n, v in zip(names, a)}
    return Study(
        "Mode collapse in a GAN",
        "A small GAN on MNIST in two schedules and with the label given to both networks; "
        "a separate classifier reads %s samples every %d updates: %d seeds each"
        % (f"{GAN_FIXED:,}", GAN_EVERY, len(GAN_SEEDS)),
        (
            Panel("What each generator draws at the end",
                  "values: the first %d fixed samples of each setting's first seed after "
                  "%s discriminator updates (setting by setting; the labelled generators "
                  "were asked for 0..9 twice). history: the same samples of every seed at "
                  "every snapshot." % (GAN_KEPT, f"{GAN_STEPS:,}"), "images",
                  dict(values=last.reshape(-1, 28, 28),
                       labels=["%s %d" % (n, k) for n in names for k in range(GAN_KEPT)],
                       history=kept, steps=steps,
                       read=F.read(last.reshape(-1, 784)).argmax(1).reshape(len(names), -1),
                       asked=np.arange(GAN_KEPT) % 10)),
            Panel("Which digits are produced",
                  "values: the share of %s samples the classifier reads as each digit at "
                  "the end of training, first seed of each setting; first row: the same "
                  "classifier on the held-out real digits. by_seed: every seed; "
                  "history[setting, seed, snapshot]: at every snapshot."
                  % f"{GAN_FIXED:,}", "matrix",
                  dict(values=np.vstack((d["real"][None], d["hist"][:, 0, -1])),
                       labels=["real digits"] + names, by_seed=d["hist"][:, :, -1],
                       history=d["hist"], real_labels=d["real_labels"],
                       entropy=np.r_[entropy_bits(d["real"]), H[:, 0, -1]])),
            Panel("Entropy of the generated classes against training",
                  "x: discriminator updates; ys[setting]: Shannon entropy (bits) of the "
                  "classifier's histogram of %s samples, mean of %d seeds; seeds: every "
                  "run; covered: digits read in at least %d%% of samples; top: the "
                  "commonest digit. Ten equally likely digits: %.2f bits."
                  % (f"{GAN_FIXED:,}", len(GAN_SEEDS), round(100 * F.COVERED), np.log2(10)),
                  "curve",
                  dict(x=steps, ys=H.mean(1), labels=names, seeds=H, covered=s["covered"],
                       top=s["top"], top_share=s["top_share"], end=end,
                       collapsed=s["collapsed"])),
            Panel("Read as the digit asked for",
                  "x: discriminator updates; ys[setting]: the share of %s samples the "
                  "classifier reads as the label asked for, mean of %d seeds (the "
                  "unlabelled generators cannot see the request: chance, one in ten); "
                  "per_digit[setting, seed, digit]: at the end, digit by digit."
                  % (f"{GAN_FIXED:,}", len(GAN_SEEDS)), "curve",
                  dict(x=steps, ys=s["asked"].mean(1), labels=names, seeds=s["asked"],
                       per_digit=d["hit"][:, :, -GAN_LAST:].mean(2), end=s["asked_end"])),
        ),
        dict(
            mnist=True, review_only=True, dataset="MNIST", source="NumPy training on MNIST",
            evidence="measured on MNIST: generated samples read by a separately trained "
                     "classifier",
            data=dict(train_x_sha256=str(data["train_x_sha256"]),
                      test_x_sha256=str(data["test_x_sha256"]),
                      splits="training: %d digits of realdata.mnist_split('train'); the "
                             "classifier is checked on %d held-out digits of the official "
                             "test partition" % (F.TRAIN_N, F.TEST_N)),
            seeds=list(GAN_SEEDS), settings=names, steps=GAN_STEPS, samples=GAN_FIXED,
            model="generator %d (+10 label inputs) -> 64 -> 256 -> 784 sigmoid; "
                  "discriminator 784 (+10) -> 256 -> 64 -> 1, leaky ReLU, dropout %.1f; "
                  "Adam (beta1 %.1f), generator %g, discriminator %g, batch %d, real "
                  "targets %.1f: voxanim.study_families' GAN"
                  % (GAN_CODE, F.GAN_DROP, F.GAN_BETA1, F.GAN_LR_G, F.GAN_LR_D, F.GAN_BATCH,
                     F.GAN_SMOOTH),
            schedules="balanced: 1 generator update per discriminator update; "
                      "generator-heavy: %d. Nothing else differs." % GAN_HEAVY,
            label="ten extra inputs to both networks, %g at the digit (0 elsewhere); the "
                  "discriminator sees each real digit with its own label" % GAN_LABEL,
            reader="voxanim.mnist.train(): 784-64-32-10, %.4f of the 10,000 test images, "
                   "%.4f of the %d held-out digits used here"
                   % (float(d["reader_test"]), float(d["reader_held_out"]), F.TEST_N),
            collapse_rule="a run counts as collapsed when the entropy of its histogram, "
                          "averaged over the last %d snapshots, is under %g bits (four "
                          "equally likely digits); fixed before the runs"
                          % (GAN_LAST, GAN_COLLAPSED),
            max_entropy=float(np.log2(10)), real_entropy=entropy_bits(d["real"]),
            entropy_end=by(end), covered_end=by(s["covered_end"]),
            asked_end=by(s["asked_end"]),
            collapsed={n: int(v.sum()) for n, v in zip(names, s["collapsed"])},
            top_digit_end={n: [int(v) for v in t[:, -1]] for n, t in zip(names, s["top"])},
            limitations="The classifier is the judge: a blurred or mixed picture is still "
                        "read as some digit, so entropy counts classes, not quality, and "
                        "says nothing about variety within a class. One small dense GAN, "
                        "%d discriminator updates, five seeds; the generator-heavy runs "
                        "make %d times as many generator updates. A run's state at the "
                        "end is a snapshot of something that keeps moving."
                        % (GAN_STEPS, GAN_HEAVY),
            designs_tried=[
                "two seeds, 4,000 updates each of: the balanced setting (9 to 10 digits "
                "covered), plain Adam with beta1 0.9 and no dropout (one digit, both "
                "seeds; but two changes at once), no dropout alone (8 to 10 digits), a "
                "noise dimension of 1 (4 and 1 digits) and of 2 (5 and 7), and 3 "
                "generator updates per discriminator update (1 and 2 digits)",
                "2 generator updates per discriminator update, 3,000 updates: 4 and 6 "
                "digits; 3 was kept as the collapsing setting because it differs from "
                "the balanced one in a single number and collapsed on both seeds",
                "the label as inputs of size 1: the generator obeyed it on 26% to 34% of "
                "samples after 3,000 updates and 92% after 6,000; at size 4, 83% to 94% "
                "by 3,000. Size 4 and 3,000 updates were kept to stay in the time limit",
                "the design recorded here, fixed after those probes: four settings, "
                "five seeds, the collapse rule and the seed drawn (the first) chosen "
                "before the runs",
            ],
        ),
    )


# ====================================================================== 156: a GP background
GP_SEED = 156
GP_EDGES = np.linspace(100.0, 500.0, 81)          # 80 bins of 5 GeV
GP_M = (GP_EDGES[:-1] + GP_EDGES[1:]) / 2
GP_TRUTH = ((9000.0, 50.0), (300.0, 150.0))       # (events per bin at 100 GeV, decay length)
GP_SIGMA = 5.0                                    # the resonance's width (a Gaussian), GeV
GP_M0 = 300.0                                     # where it is injected
GP_S = 160                                        # injected events: 5 x the window's error
GP_AMP = 3.0                                      # kernel amplitude, in statistical errors
GP_ELLS = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 240.0, 320.0)
GP_SHORT = 10.0                                   # the "too short" length scale drawn
GP_TOYS = 4000
GP_SCAN = np.arange(125.0, 476.0, 10.0)           # window centres of the scan over mass


def gp_truth(m=GP_M):
    """The true background: expected events per bin, two exponentials."""
    m = np.asarray(m, float)
    return sum(n * np.exp(-(m - 100.0) / tau) for n, tau in GP_TRUTH)


def gp_template(m0=GP_M0):
    """The resonance's share of its events in each bin: a Gaussian of width
    :data:`GP_SIGMA` at ``m0``."""
    t = np.exp(-0.5 * ((GP_M - m0) / GP_SIGMA) ** 2)
    return t / t.sum()


def gp_window(m0=GP_M0):
    """The bins within two widths of ``m0``: where signal events are counted."""
    return np.abs(GP_M - m0) <= 2 * GP_SIGMA


def fit_exponential(y, iters=30):
    """The fixed function: ``A exp(-k m)`` fitted to counts ``y`` ``(n, bins)`` by
    maximum likelihood (Poisson counts; Newton steps on ``ln A`` and ``k``, which for
    this model is the standard iteratively reweighted least squares). Returns the fitted
    expected counts, one row per spectrum."""
    y = np.atleast_2d(np.asarray(y, float))
    X = np.stack([np.ones_like(GP_M), (GP_M - 300.0) / 100.0], 1)
    beta = np.zeros((len(y), 2))
    beta[:, 0] = np.log(y.mean(1))
    for _ in range(iters):
        mu = np.exp(np.clip(beta @ X.T, -30, 30))
        info = np.einsum("nb,bi,bj->nij", mu, X, X)
        beta = beta + np.linalg.solve(info, ((y - mu) @ X)[..., None])[..., 0]
    return np.exp(beta @ X.T)


def gp_kernel(ell, amp=GP_AMP):
    """The kernel matrix over the bin centres: RBF, in units of the statistical error."""
    return gp.RBF(variance=amp * amp, lengthscale=float(ell))(GP_M, GP_M)


def gp_smoother(ell, amp=GP_AMP):
    """``(H, sd, L)``. The Gaussian process sees pulls z (unit noise) and returns their
    posterior mean ``H z`` with ``H = K (K + I)^-1``: row i of ``H`` is the weight each
    bin's pull gets in the estimate at bin i. ``sd``: the posterior standard deviation
    per bin; ``L``: the Cholesky factor of ``K + I``."""
    K = gp_kernel(ell, amp)
    L = gp.chol(K, jitter=1.0)
    H = gp.chol_solve(L, K).T
    return H, np.sqrt(np.maximum(np.diag(K - H @ K), 0.0)), L


def gp_background(y, H, fit=None):
    """``(fit, background)`` for spectra ``y``: the exponential fit, and that fit plus
    the Gaussian process's estimate of what it leaves (pulls times the bin's error)."""
    mu = fit_exponential(y) if fit is None else fit
    return mu, mu + np.sqrt(mu) * (((np.atleast_2d(y) - mu) / np.sqrt(mu)) @ H.T)


def gp_weights(centres):
    """One row per window centre: ``(data - background) @ weights.T`` is the events
    above background in the window, scaled up by the share of a resonance that falls
    inside it."""
    return np.array([gp_window(c) / gp_template(c)[gp_window(c)].sum() for c in centres])


def gp_evidence(z, L):
    """``ln p(z)`` of pulls ``z`` ``(n, bins)`` under the process with Cholesky factor
    ``L`` of ``K + I``: the marginal likelihood a length scale can be chosen by."""
    a = np.linalg.solve(L, np.atleast_2d(z).T)
    return -0.5 * (a * a).sum(0) - np.log(np.diag(L)).sum() - 0.5 * len(L) * gp.LOG2PI


def gp_toys():
    """The pseudo-experiments: ``(without signal, with)``. Each spectrum is Poisson
    counts about the true background; its partner is the same spectrum plus Poisson
    counts about the resonance."""
    rng = np.random.default_rng(GP_SEED)
    y0 = rng.poisson(gp_truth(), (GP_TOYS, len(GP_M))).astype(float)
    return y0, y0 + rng.poisson(GP_S * gp_template(), y0.shape)


def gp_measure(ys, backgrounds):
    """``(row, scan)`` for one background model. ``ys``: the spectra without signal,
    with, and the two expected spectra; ``backgrounds``: the model's estimate for each.
    ``row``: see :func:`gp_arrays`; ``scan``: the mean events above background in a
    window at each centre of :data:`GP_SCAN` with no signal (pseudo-experiments, then
    expected counts)."""
    w = gp_weights([GP_M0])[0]
    scan = gp_weights(GP_SCAN)
    e0, e1, ea = ((y - k) @ w for y, k in zip(ys, backgrounds))
    row = [e0.mean(), e0.std(ddof=1), (e1 - e0).mean() / GP_S, (e1 - e0).std(ddof=1) / GP_S,
           ea[0], (ea[1] - ea[0]) / GP_S, e1.std(ddof=1)]
    return np.array(row), np.array([((ys[0] - backgrounds[0]) @ scan.T).mean(0),
                                    (ys[2][0] - backgrounds[2][0]) @ scan.T])


def _gp_build():
    y0, y1 = gp_toys()
    b = gp_truth()
    ys = (y0, y1, np.array([b, b + GP_S * gp_template()]))    # the last: no fluctuation
    fits = [fit_exponential(y) for y in ys]
    out = {}
    out["exp"], out["exp_scan"] = gp_measure(ys, fits)
    rows, scans, ev = [], [], []
    for ell in GP_ELLS:
        H, _sd, L = gp_smoother(ell)
        row, scan = gp_measure(ys, [gp_background(y, H, f)[1] for y, f in zip(ys, fits)])
        rows.append(row)
        scans.append(scan)
        ev.append([gp_evidence((y - f) / np.sqrt(f), L) for y, f in zip(ys[:2], fits)])
    ev = np.array(ev)                                         # [length scale, signal?, toy]
    out.update(gp=np.array(rows), gp_scan=np.array(scans), evidence_mean=ev.mean(2),
               evidence_pick=np.array([np.bincount(ev[:, k].argmax(0),
                                                   minlength=len(GP_ELLS)) for k in (0, 1)]),
               first=np.array([y0[0], y1[0]]))
    return out


def gp_arrays():
    """The pseudo-experiments' summaries. Columns of ``exp`` and of each row of ``gp``
    (one per length scale): 0 the mean events above background with no signal injected
    (the spurious signal), 1 its standard deviation over spectra, 2 the share of the
    injected signal recovered (mean of paired differences / injected), 3 that share's
    standard deviation, 4 and 5 the spurious signal and the share on the expected counts
    (exact), 6 the standard deviation of the estimate with signal."""
    return _cache("gp", _gp_build,
                  (gp_truth, gp_template, gp_window, fit_exponential, gp_kernel, gp_smoother,
                   gp_background, gp_weights, gp_evidence, gp_toys, gp_measure, _gp_build),
                  seed=GP_SEED, edges=[100.0, 500.0, 80], truth=GP_TRUTH, sigma=GP_SIGMA,
                  m0=GP_M0, s=GP_S, amp=GP_AMP, ells=GP_ELLS, toys=GP_TOYS,
                  scan=[125.0, 475.0, 10.0])


def gp_background_fit():
    d = gp_arrays()
    b, t, win = gp_truth(), gp_template(), gp_window()
    ells = np.array(GP_ELLS)
    good = int(d["evidence_mean"][:, 0].argmax())        # the length scale the data prefer
    short = GP_ELLS.index(GP_SHORT)
    y0, y1 = d["first"]
    w = gp_weights([GP_M0])[0]
    stat = float(np.sqrt(b[win].sum()) * w.max())        # the window's statistical error
    fit0, fit1 = fit_exponential(y0)[0], fit_exponential(y1)[0]
    H_good, sd_good, _ = gp_smoother(ells[good])
    H_short, _sd, _ = gp_smoother(ells[short])
    bkg0 = gp_background(y0, H_good)[1][0]
    bkg1 = {k: gp_background(y1, H)[1][0] for k, H in (("good", H_good), ("short", H_short))}
    asimov_fit = fit_exponential(b)[0]
    expected_pull = (b - asimov_fit) / np.sqrt(asimov_fit)
    centre = int(np.abs(GP_M - GP_M0).argmin())
    sem = np.sqrt(1.0 / GP_TOYS)
    exp, g = d["exp"], d["gp"]
    scan_w = gp_weights(GP_SCAN)
    toys = f"{GP_TOYS:,}"

    def about(row):
        return dict(spurious=float(row[0]), spurious_exact=float(row[4]), share=float(row[2]),
                    share_exact=float(row[5]), rms=float(row[1]),
                    spurious_error=float(row[1] * sem), share_error=float(row[3] * sem))
    return Study(
        "A Gaussian-process background",
        "A falling spectrum drawn from a stated model: an exponential fit against a "
        "Gaussian process, %s pseudo-experiments with and without an injected resonance, "
        "%d length scales" % (toys, len(GP_ELLS)),
        (
            Panel("One pseudo-data spectrum and two background models",
                  "x: bin centres (GeV); ys: the counts of the first pseudo-experiment "
                  "(no signal), the true expected counts, the exponential fitted to the "
                  "counts, and the Gaussian-process background (length scale %g GeV); "
                  "band: its one-sigma uncertainty per bin." % ells[good], "curve",
                  dict(x=GP_M, ys=np.array([y0, b, fit0, bkg0]),
                       labels=["pseudo-data", "truth", "exponential fit", "Gaussian process"],
                       band=np.sqrt(fit0) * sd_good, with_signal=y1, edges=GP_EDGES)),
            Panel("What each background leaves",
                  "ys: (counts - background) / sqrt(background) per bin for the "
                  "exponential fit and for the Gaussian process, same spectrum; expected: "
                  "the exponential's pulls on the expected counts (exact: the wave is not "
                  "a fluctuation). weights_*: the row of the smoother at the bin of %g "
                  "GeV for the preferred and the short length scale; with_signal: pulls "
                  "of the spectrum with the resonance for the exponential, the Gaussian "
                  "process and the short Gaussian process." % GP_M0, "curve",
                  dict(x=GP_M, ys=np.array([(y0 - fit0) / np.sqrt(fit0),
                                            (y0 - bkg0) / np.sqrt(bkg0)]),
                       labels=["exponential fit", "Gaussian process"],
                       expected=expected_pull,
                       weights_good=H_good[centre], weights_short=H_short[centre],
                       smoother_good=H_good, smoother_short=H_short,
                       centre=centre, band=sd_good,
                       with_signal=np.array([(y1 - fit1) / np.sqrt(fit1),
                                             (y1 - bkg1["good"]) / np.sqrt(bkg1["good"]),
                                             (y1 - bkg1["short"]) / np.sqrt(bkg1["short"])]),
                       backgrounds=np.array([fit1, bkg1["good"], bkg1["short"]]),
                       signal=GP_S * t)),
            Panel("Spurious signal and recovered share against length scale",
                  "x: kernel length scale (GeV); ys[0]: mean events above the "
                  "Gaussian-process background in the window with no signal injected, "
                  "over %s pseudo-experiments; ys[1]: share of %d injected events "
                  "recovered. exact: the same on the expected counts; exponential: the "
                  "fixed function's row; rms: spread of the estimate; sem: standard "
                  "errors of the two means." % (toys, GP_S), "curve",
                  dict(x=ells, ys=np.array([g[:, 0], g[:, 2]]),
                       labels=["spurious signal (events)", "share recovered"],
                       exact=np.array([g[:, 4], g[:, 5]]), rms=g[:, 1], rms_signal=g[:, 6],
                       sem=np.array([g[:, 1] * sem, g[:, 3] * sem]),
                       exponential=exp, evidence=d["evidence_mean"],
                       evidence_pick=d["evidence_pick"])),
            Panel("Spurious signal across the spectrum",
                  "x: window centre (GeV); ys: mean events above background in a window "
                  "of two widths either side with no signal anywhere, for the exponential "
                  "fit and the Gaussian process (length scale %g GeV), over %s "
                  "pseudo-experiments; exact: on the expected counts; error: each "
                  "window's statistical error." % (ells[good], toys), "curve",
                  dict(x=GP_SCAN, ys=np.array([d["exp_scan"][0], d["gp_scan"][good, 0]]),
                       labels=["exponential fit", "Gaussian process"],
                       exact=np.array([d["exp_scan"][1], d["gp_scan"][good, 1]]),
                       error=np.sqrt((scan_w > 0) @ b) * scan_w.max(1))),
        ),
        dict(
            review_only=True, dataset="none: pseudo-data from a stated model",
            source="NumPy: seeded pseudo-experiments and exact linear algebra",
            evidence=dict(
                pseudo_data="one seeded Poisson draw from the stated model",
                pseudo_experiments="%d seeded spectra from numpy.random.default_rng(%d), "
                                   "each also with a resonance injected" % (GP_TOYS, GP_SEED),
                exact="the same procedure on the expected counts, with no fluctuation"),
            seed=GP_SEED, toys=GP_TOYS,
            model="expected events per 5 GeV bin = %g exp(-(m - 100) / %g) + %g exp(-(m - "
                  "100) / %g), m from 100 to 500 GeV, Poisson counts"
                  % (GP_TRUTH[0][0], GP_TRUTH[0][1], GP_TRUTH[1][0], GP_TRUTH[1][1]),
            signal="a Gaussian of width %g GeV at %g GeV, %d events (Poisson)"
                   % (GP_SIGMA, GP_M0, GP_S),
            fixed_function="A exp(-k m), two parameters, Poisson maximum likelihood",
            gaussian_process="mean: the exponential fit; kernel: RBF on the pulls (counts "
                             "- fit) / sqrt(fit) with amplitude %g and unit noise; "
                             "background = fit + sqrt(fit) x posterior mean" % GP_AMP,
            estimate="events above background in the bins within %g GeV of the mass, "
                     "divided by the share of a resonance inside that window (%.3f)"
                     % (2 * GP_SIGMA, t[win].sum()),
            window_bins=int(win.sum()), window_background=float(b[win].sum()),
            window_error=stat, total_events=float(b.sum()),
            ell_good=float(ells[good]), ell_short=GP_SHORT,
            exponential=about(exp), gp_good=about(g[good]), gp_short=about(g[short]),
            share_by_ell={"%g" % e: float(v) for e, v in zip(ells, g[:, 2])},
            spurious_by_ell={"%g" % e: float(v) for e, v in zip(ells, g[:, 0])},
            evidence_pick={"%g" % e: [int(a), int(c)] for e, a, c in
                           zip(ells, d["evidence_pick"][0], d["evidence_pick"][1])},
            drawn=dict(exponential=float((y1 - fit1) @ w),
                       good=float((y1 - bkg1["good"]) @ w),
                       short=float((y1 - bkg1["short"]) @ w)),
            largest_expected_pull=float(np.abs(expected_pull).max()),
            limitations="The background is estimated from the whole spectrum, signal "
                        "region included, and the signal is what is left in a window: "
                        "that is what lets a flexible background absorb it. Fitting "
                        "signal and background together, or leaving the window out, "
                        "changes the share recovered and is not measured here. The "
                        "kernel's amplitude is fixed at %g statistical errors and only "
                        "the length scale is varied; counts are treated as Gaussian with "
                        "variance equal to the fit (the lowest expected bin holds %.0f "
                        "events). One truth, one resonance width, one mass for the "
                        "injection; the scan over mass has no injection."
                        % (GP_AMP, b.min()),
            designs_tried=[
                "a resonance of width 10 GeV: the Gaussian process recovered about half "
                "of it even at a length scale of 80 GeV, so the range of length scales "
                "that both follow the background and keep the signal was nearly empty; "
                "the width was halved (5 GeV, one bin) so the two scales are separated",
                "three truths (two exponentials with different normalisations) and "
                "amplitudes 1, 2, 3, 5 on the expected counts; this truth (largest "
                "expected pull about 5) and amplitude 3 were then fixed",
                "the design recorded here: 13 length scales, 4,000 paired "
                "pseudo-experiments, the injection mass (300 GeV) fixed before any "
                "spurious signal was computed",
            ],
        ),
    )


STUDIES = {
    "autoencoder-anomaly-detection": autoencoder_anomaly_detection,
    "gan-mode-collapse": gan_mode_collapse,
    "gp-background-fit": gp_background_fit,
}
