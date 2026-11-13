"""Capacity and bottlenecks: the models behind the ``capacity`` animations.

Four studies on MNIST, every one a real numpy model trained (or loaded)
through :func:`voxanim.cache.cached_npz`:

* :func:`bottleneck_sweep` -- the dense autoencoder of
  :mod:`voxanim.autoencoder` with latent widths 1, 2, 3, 8, 32 and 256
  (:data:`WIDTHS`), all trained alike (:data:`SWEEP`: 3 epochs, batch 256,
  so the six fit the training budget); the same four test digits rebuilt by
  each, the test reconstruction error against width, the 600 test codes of
  the 1-, 2- and 3-D models, and how well a nearest-centroid classifier does
  on each width's codes (what the code still says about the digit).
* :func:`squeeze_run` -- a classifier ``784 -> 64 (ReLU) -> h (linear) -> 10
  (softmax)`` with ``h = 2`` and ``h = 64``, trained by Adam with the middle
  layer's values recorded for 600 test digits at geometric snapshots, as
  :func:`voxanim.autoencoder.training_run` records codes; the test accuracy
  at every snapshot and the confusion matrix on the 10,000 test digits at the
  end.  The middle layer is linear so that nothing clips the 2-D space (a
  tanh pair saturates at the four corners and learns 42%; a ReLU pair
  switches half its values off and learns 65%); with 64 units it constrains
  nothing, which is the point of the comparison.  Its gradients are checked
  against finite differences in ``tests/test_capacity.py``.
* :func:`rank_study` -- the singular value decomposition of the trained MNIST
  MLP's weight matrices (784 × 64 and 64 × 32, :func:`voxanim.mnist.train`),
  both rebuilt from their top ``k`` components for ``k`` in :data:`RANKS`,
  and the test accuracy of each truncated network.
* :func:`width_depth_runs` -- three ReLU MLPs of about 50,000 parameters
  (:data:`SHAPES`: one hidden layer of 64, four of 52, sixteen of 37)
  trained alike by :func:`voxanim.deepnet.train`, plus the wall-clock time of
  one training step of each, measured here (:func:`step_times`), not derived
  from a FLOP count.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np

from . import autoencoder as AE
from . import deepnet as DN
from . import mnist
from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import batched, chunk_mean, minibatches
from .optim import Adam

# ------------------------------------------------------------------ the bottleneck sweep

WIDTHS = (1, 2, 3, 8, 32, 256)             # latent widths of the sweep
SWEEP = dict(epochs=2, batch=256, lr=2e-3)  # one training recipe for all six (about a minute each)
DIGITS = (3, 2, 1, 4)                       # test images rebuilt by every width: a 0, 1, 2, 4
CLOUD = 600                                 # test codes in a latent cloud (as ae_anim)
CENTROID_FIT = 2000                         # test codes the class centroids are fitted on
CLOUD_WIDTHS = (1, 2, 3)                    # widths whose clouds can be drawn


@dataclass
class Sweep:
    """The bottleneck sweep: six autoencoders differing only in latent width."""

    widths: tuple
    losses: np.ndarray            # (6,) mean squared test reconstruction error of each
    images: np.ndarray            # (4, 784) the digits rebuilt
    labels: np.ndarray            # (4,)
    recon: np.ndarray             # (6, 4, 784) each width's reconstruction of each digit
    clouds: dict                  # width -> (CLOUD, width) test codes, for CLOUD_WIDTHS
    cloud_labels: np.ndarray      # (CLOUD,)
    centroid_acc: np.ndarray      # (6,) nearest-centroid accuracy of the codes, per width
    epochs: int

    def error(self, w, d):
        """Reconstruction error of digit ``d`` by width ``w`` (an index into ``widths``)."""
        return float(np.mean((self.recon[w, d] - self.images[d]) ** 2))


def centroid_accuracy(z_fit, y_fit, z, y):
    """Accuracy of the nearest class centroid (fitted on ``z_fit``) on the codes ``z``."""
    centroids = np.array([z_fit[y_fit == c].mean(0) for c in range(10)])
    d = ((z[:, None, :] - centroids[None]) ** 2).sum(-1)
    return float((d.argmin(1) == y).mean())


def bottleneck_sweep(cache_dir=DEFAULT_CACHE) -> Sweep:
    """Train (or load) the six autoencoders and evaluate them (see :class:`Sweep`)."""
    _xtr, _ytr, xte, yte = mnist.load(cache_dir)
    imgs = xte[list(DIGITS)]
    sel = np.arange(AE.CLOUD_START, AE.CLOUD_START + CLOUD)
    fit = np.arange(CENTROID_FIT)
    losses, recon, clouds, acc = [], [], {}, []
    for L in WIDTHS:
        P, loss = AE.train("dense", L, cache_dir=cache_dir, **SWEEP)
        losses.append(loss)
        recon.append(AE.forward("dense", P, imgs)["recon"])

        def encode(x, P=P):
            return AE.forward("dense", P, x)["z"]

        z = batched(encode, xte[sel], 200)
        if L in CLOUD_WIDTHS:
            clouds[L] = z
        acc.append(centroid_accuracy(batched(encode, xte[fit], 500), yte[fit], z, yte[sel]))
    return Sweep(WIDTHS, np.array(losses), imgs, yte[list(DIGITS)], np.array(recon), clouds,
                 yte[sel], np.array(acc), SWEEP["epochs"])


# ------------------------------------------------------------------ the squeezed classifier

SQUEEZE = dict(epochs=3, batch=128, lr=1e-3, seed=0)   # the classifiers' training recipe
HIDDEN = (2, 64)                                       # the squeezed and the roomy middle layer
FIRST = 64                                             # the first hidden layer, both nets
EVAL_TEST = 2000                                       # test digits a snapshot's accuracy uses
EXAMPLE = 0                                            # cloud digit whose every layer is recorded


def squeeze_sizes(hidden):
    """Layer sizes of the classifier with a ``hidden``-unit middle layer."""
    return (784, FIRST, hidden, 10)


def squeeze_init(hidden, rng):
    """He weights for the ReLU layer, Xavier for the linear middle and the output."""
    P = {}
    AE.dense_init(P, "0", 784, FIRST, rng)
    for k, (a, b) in enumerate(((FIRST, hidden), (hidden, 10)), 1):
        P["W%d" % k] = rng.normal(0, np.sqrt(1.0 / a), (a, b))
        P["b%d" % k] = np.zeros(b)
    return P


def squeeze_forward(P, x):
    """``(h1, h2, p)``: the ReLU first layer, the *linear* middle layer (the
    space the animation draws, unclipped) and the softmax output."""
    h1 = np.maximum(x @ P["W0"] + P["b0"], 0.0)
    h2 = h1 @ P["W1"] + P["b1"]
    logits = h2 @ P["W2"] + P["b2"]
    e = np.exp(logits - logits.max(1, keepdims=True))
    return h1, h2, e / e.sum(1, keepdims=True)


def squeeze_loss(f, y_onehot):
    """Mean cross-entropy of the forward pass ``f``."""
    return DN.cross_entropy(f[2], y_onehot)


def squeeze_backward(P, x, f, y_onehot):
    """Gradients of :func:`squeeze_loss` with respect to every parameter."""
    h1, h2, p = f
    d = (p - y_onehot) / len(x)
    G = {"W2": h2.T @ d, "b2": d.sum(0)}
    d = d @ P["W2"].T
    G["W1"], G["b1"] = h1.T @ d, d.sum(0)
    d = (d @ P["W1"].T) * (h1 > 0)
    G["W0"], G["b0"] = x.T @ d, d.sum(0)
    return G


@dataclass
class SqueezeRun:
    """One classifier's training run, its middle layer watched (see :func:`squeeze_run`).

    Arrays with a leading ``S`` axis have one entry per snapshot.
    """

    hidden: int
    steps: np.ndarray             # (S,) training step of each snapshot; 0 is untrained
    steps_per_epoch: int
    test_acc: np.ndarray          # (S,) accuracy on the first EVAL_TEST test digits
    cloud: np.ndarray             # (S, CLOUD, hidden) the middle layer's output for the cloud
    cloud_labels: np.ndarray      # (CLOUD,)
    example: list                 # per layer (first hidden, middle, softmax): (S, width) of
                                  # cloud digit EXAMPLE's values
    confusion: np.ndarray         # (10, 10) counts on the 10,000 test digits: [label, predicted]
    final_acc: float              # accuracy on all 10,000 test digits
    P: dict                       # the trained parameters

    def confusions(self, n=3):
        """The ``n`` most confused unordered pairs: ``[(a, b, count), ...]``, the
        count summing both directions."""
        c = self.confusion + self.confusion.T
        pairs = [(a, b, int(c[a, b])) for a in range(10) for b in range(a + 1, 10)]
        return sorted(pairs, key=lambda p: -p[2])[:n]

    @property
    def errors(self):
        """Test digits the network gets wrong (of 10,000)."""
        return int(self.confusion.sum() - np.trace(self.confusion))


def squeeze_run(hidden, cache_dir=DEFAULT_CACHE) -> SqueezeRun:
    """Train the ``784 -> 64 -> hidden -> 10`` classifier by Adam on all 60,000
    training digits, recording at :func:`voxanim.autoencoder.snapshot_steps`
    the middle layer's values for the cloud digits and the test accuracy;
    cached under every setting."""
    settings = dict(sizes=list(squeeze_sizes(hidden)), middle="linear", **SQUEEZE, cloud=CLOUD,
                    cloud_start=AE.CLOUD_START, eval_test=EVAL_TEST,
                    snapshot_ratio=AE.SNAPSHOT_RATIO, example=EXAMPLE)
    xtr, ytr, xte, yte = mnist.load(cache_dir)
    sel = np.arange(AE.CLOUD_START, AE.CLOUD_START + CLOUD)
    eye = np.eye(10)

    def build():
        rng = np.random.default_rng(SQUEEZE["seed"])
        P = squeeze_init(hidden, rng)
        opt = Adam(P, SQUEEZE["lr"])
        per_epoch = -(-len(xtr) // SQUEEZE["batch"])
        marks = AE.snapshot_steps(SQUEEZE["epochs"] * per_epoch)
        rec = dict(acc=[], cloud=[], ex=[])

        def snapshot():
            h1, h2, p = squeeze_forward(P, xte[sel])
            rec["cloud"].append(h2)
            rec["ex"].append([h1[EXAMPLE], h2[EXAMPLE], p[EXAMPLE]])
            rec["acc"].append(chunk_mean(
                lambda s: (squeeze_forward(P, xte[s])[2].argmax(1) == yte[s]).mean(),
                EVAL_TEST))

        snapshot()
        want = set(marks.tolist())
        for step, idx in enumerate(minibatches(rng, len(xtr), SQUEEZE["epochs"],
                                               SQUEEZE["batch"]), 1):
            x = xtr[idx]
            opt.step(P, squeeze_backward(P, x, squeeze_forward(P, x), eye[ytr[idx]]))
            if step in want:
                snapshot()
        pred = batched(lambda x: squeeze_forward(P, x)[2].argmax(1), xte, 1000)
        confusion = np.zeros((10, 10), int)
        np.add.at(confusion, (yte, pred), 1)
        out = dict(P, steps=marks, steps_per_epoch=np.array(per_epoch),
                   test_acc=np.array(rec["acc"]), cloud=np.array(rec["cloud"], np.float32),
                   confusion=confusion, final_acc=np.array((pred == yte).mean()))
        for j in range(3):
            out["ex%d" % j] = np.array([e[j] for e in rec["ex"]])
        return out

    d = cached_npz("squeeze_h%d" % hidden, settings, build, cache_dir)
    meta = ("steps", "steps_per_epoch", "test_acc", "cloud", "confusion", "final_acc")
    P = {k: v for k, v in d.items() if k not in meta and not k.startswith("ex")}
    return SqueezeRun(hidden, d["steps"], int(d["steps_per_epoch"]), d["test_acc"],
                      d["cloud"].astype(np.float64), yte[sel], [d["ex%d" % j] for j in range(3)],
                      d["confusion"], float(d["final_acc"]), P)


# ------------------------------------------------------------------ the rank of a layer

RANKS = (1, 2, 4, 8, 16, 32)                # components kept, in turn; then the full matrix


@dataclass
class RankStudy:
    """The trained MLP's two hidden weight matrices under truncated SVD."""

    Ws: list                      # the three weight matrices of the MLP
    bs: list
    svd: list                     # per truncated matrix: (U, s, Vt) of Ws[0] and Ws[1]
    ranks: tuple                  # RANKS, then the full rank
    acc: np.ndarray               # (len(ranks),) test accuracy with both matrices truncated
    full_acc: float               # the untouched network's test accuracy
    kept: np.ndarray              # (len(ranks),) fraction of the two matrices' parameters kept

    def truncated(self, m, k):
        """Matrix ``m`` (0 or 1) rebuilt from its top ``k`` components."""
        U, s, Vt = self.svd[m]
        return (U[:, :k] * s[:k]) @ Vt[:k]

    def energy(self, m, k):
        """The fraction of matrix ``m``'s squared Frobenius norm its top ``k`` components hold."""
        s = self.svd[m][1]
        return float((s[:k] ** 2).sum() / (s ** 2).sum())


def params_kept(shapes, k):
    """Fraction of the parameters of matrices of ``shapes`` ``[(a, b), ...]`` that
    rank-``k`` factors ``(a, k)`` and ``(k, b)`` hold (capped at 1)."""
    full = sum(a * b for a, b in shapes)
    low = sum(min(a * b, k * (a + b)) for a, b in shapes)
    return low / full


def truncated_accuracy(Ws, bs, svd, k, x, y):
    """Test accuracy of the MLP with its first two matrices rebuilt from ``k`` components."""
    Wk = [(U[:, :k] * s[:k]) @ Vt[:k] for U, s, Vt in svd] + [Ws[2]]
    params = [[W, b] for W, b in zip(Wk, bs)]
    return mnist.chunked_accuracy(lambda xs: mnist._forward(params, xs)[2], x, y)


def rank_study(cache_dir=None) -> RankStudy:
    """SVD of the MNIST MLP's matrices, and the accuracy after each truncation."""
    params, full_acc = mnist.train(cache_dir=cache_dir)
    Ws, bs = [W for W, _ in params], [b for _, b in params]
    svd = [np.linalg.svd(W, full_matrices=False) for W in Ws[:2]]
    _xtr, _ytr, xte, yte = mnist.load(cache_dir)
    ranks = RANKS + (Ws[0].shape[1],)
    acc = np.array([truncated_accuracy(Ws, bs, svd, k, xte, yte) for k in ranks])
    shapes = [W.shape for W in Ws[:2]]
    kept = np.array([params_kept(shapes, k) for k in ranks])
    return RankStudy(Ws, bs, svd, ranks, acc, full_acc, kept)


# ------------------------------------------------------------------ width against depth

SHAPES = (("wide", (784, 64, 10)),                   # 50,890 parameters
          ("medium", (784,) + (52,) * 4 + (10,)),    # 49,618
          ("narrow", (784,) + (37,) * 16 + (10,)))   # 50,515
TRAIN = dict(steps=1500, batch=128, lr=1e-3, opt="adam")
N_TRAIN, N_TEST = 20000, 2000
TIMING_STEPS = 60                                    # training steps timed per network
TIMING_WARMUP = 5                                    # ... after these, not timed


def n_params(sizes):
    """Weights and biases of a dense network of layer ``sizes``."""
    return sum(a * b + b for a, b in itertools.pairwise(sizes))


def width_depth_specs():
    """The three :class:`voxanim.deepnet.Spec` s: ReLU, He init."""
    return [(name, DN.Spec(sizes, "relu", "he")) for name, sizes in SHAPES]


def width_depth_runs(cache_dir=None):
    """``(runs, data)``: the three recorded runs on the same 20,000 training digits."""
    data = DN.subset(N_TRAIN, N_TEST, seed=0, cache_dir=cache_dir)
    kw = dict(TRAIN)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return [DN.train(spec, data, **kw) for _name, spec in width_depth_specs()], data


def time_step(spec, data, batch, steps=TIMING_STEPS, warmup=TIMING_WARMUP, seed=0):
    """Median wall-clock seconds of one Adam training step (forward, backward,
    update) of ``spec`` on batches of ``batch``, over ``steps`` timed steps."""
    rng = np.random.default_rng(seed)
    P = DN.init(spec, rng)
    opt = Adam(P, TRAIN["lr"])
    eye = data.onehot
    times = []
    for n, idx in enumerate(minibatches(rng, len(data.x_train), 1, batch)):
        if n == steps + warmup:
            break
        t0 = time.perf_counter()
        f = DN.forward(spec, P, data.x_train[idx])
        G, _D = DN.backward(spec, P, f, eye(data.y_train[idx]))
        opt.step(P, G)
        if n >= warmup:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def step_times(data, cache_dir=None):
    """Seconds per training step of each network of :data:`SHAPES`, measured once
    on the machine that builds the cache file (the number is that machine's)."""
    settings = dict(shapes=[list(s) for _n, s in SHAPES], batch=TRAIN["batch"],
                    steps=TIMING_STEPS, warmup=TIMING_WARMUP, data=data.settings())

    def build():
        return dict(seconds=np.array([time_step(spec, data, TRAIN["batch"])
                                      for _name, spec in width_depth_specs()]))

    return cached_npz("capacity_timing", settings, build, cache_dir)["seconds"]
