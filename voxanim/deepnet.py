"""A deep MLP in plain numpy, with every knob that makes depth trainable.

The network is ``x -> [dense -> (norm) -> activation -> (dropout) (+ skip)] x L
-> dense -> softmax``, cross-entropy against a one-hot label.  A :class:`Spec`
chooses

* the **depth and width** (``sizes``), the activation (tanh or ReLU),
* the **initialisation**: zero-mean normal with a chosen std, Xavier
  (``sqrt(1 / fan_in)``) or He (``sqrt(2 / fan_in)``), zero biases,
* **residual connections** every layer or every two (``h_out = f(h) + h``,
  wherever the widths match: never across the first layer, whose input is
  the image),
* **batch norm** (statistics down each column, over the batch; running
  averages for evaluation) or **layer norm** (across each row, over the
  units), each with a learned scale and shift, applied before the
  activation,
* **dropout** of hidden units (inverted: kept units are scaled by
  ``1 / keep``), and
* **weight decay** (an L2 penalty ``wd / 2 * sum W^2`` on the weights, added
  to the objective, so its gradient is checkable too).

:func:`forward` keeps every intermediate; :func:`backward` returns the
parameter gradients and the gradient reaching every hidden layer (what the
animations draw as "the gradient flowing back").  Both are checked against
finite differences, option by option, in ``tests/test_deepnet.py``.

:func:`train` runs minibatch SGD or :class:`voxanim.optim.Adam` on an MNIST
subset (:func:`subset`) and records, at a schedule of steps, the train and
test loss, the test accuracy, each layer's activation std and gradient
magnitude, optionally one layer's pre-activations on the current minibatch
(the probe) and the predicted probabilities on a grid of inputs (the
decision field).  Runs are seeded and cached by
:func:`voxanim.cache.cached_npz` under all their settings, like
:func:`voxanim.autoencoder.training_run`.

Three options were added for the generalisation animations, each off by
default and absent from the cache key when off, so every earlier run keeps
its file: ``augment`` (a callable transforming each minibatch's inputs,
drawing from its own seeded generator so the minibatch order and the
initial weights are the same with and without it), ``example_loss`` (the
loss of every training example at every recorded step) and ``probe_rows``
(record only the first rows of the probe).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .autoencoder import snapshot_steps
from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import minibatches
from .optim import Adam

EPS = 1e-5                   # added to the variance in batch and layer norm
BN_MOMENTUM = 0.9            # running mean / variance: old * 0.9 + batch * 0.1
EVAL_N = 2000                # images a recorded train / test loss is measured on
SNAPSHOT_RATIO = 1.08        # geometric schedule of recorded steps (as the autoencoder run)


# ------------------------------------------------------------------ the model


@dataclass(frozen=True)
class Spec:
    """What the network is (see the module docstring for each option)."""

    sizes: tuple                      # (inputs, hidden..., classes)
    activation: str = "tanh"          # "tanh" or "relu"
    init: str = "xavier"              # "xavier", "he" or "normal"
    init_std: float = 0.0             # the std of a "normal" init
    residual: int = 0                 # 0: none; 1: a skip every layer; 2: every two
    norm: str | None = None           # None, "batch" or "layer"
    dropout: float = 0.0              # fraction of hidden units dropped while training
    weight_decay: float = 0.0         # L2 coefficient on the weights

    @property
    def depth(self):
        """Hidden layers (dense layers minus the output layer)."""
        return len(self.sizes) - 2

    def skip(self, l):
        """True when hidden layer ``l`` (0-based) ends a residual block: its
        output is ``f(h) + h_in`` with ``h_in`` the block's input."""
        if not self.residual or (l + 1) % self.residual:
            return False
        src = l + 1 - self.residual
        return src >= 0 and self.sizes[src] == self.sizes[l + 1]

    def settings(self):
        """The JSON-able dict that keys a cache file."""
        return dict(asdict(self), sizes=list(self.sizes))


def init(spec, rng):
    """Parameters ``W{l}, b{l}`` (and ``g{l}, beta{l}`` for normalised hidden
    layers) drawn from ``rng``; the output layer always uses Xavier so its
    logits start small whatever the hidden layers do."""
    P = {}
    for l, (a, b) in enumerate(zip(spec.sizes[:-1], spec.sizes[1:])):
        last = l == spec.depth
        if spec.init == "normal" and not last:
            std = spec.init_std
        elif spec.init == "he" and not last:
            std = np.sqrt(2.0 / a)
        else:
            std = np.sqrt(1.0 / a)
        P["W%d" % l] = rng.normal(0.0, std, (a, b))
        P["b%d" % l] = np.zeros(b)
        if spec.norm and not last:
            P["g%d" % l] = np.ones(b)
            P["beta%d" % l] = np.zeros(b)
    return P


def init_stats(spec):
    """Batch norm's running mean and variance per normalised layer (used
    when evaluating); empty for the other models."""
    if spec.norm != "batch":
        return {}
    stats = {}
    for l in range(spec.depth):
        stats["mu%d" % l] = np.zeros(spec.sizes[l + 1])
        stats["var%d" % l] = np.ones(spec.sizes[l + 1])
    return stats


def _act(spec, y):
    return np.tanh(y) if spec.activation == "tanh" else np.maximum(y, 0.0)


def _act_grad(spec, y, a):
    """Derivative of the activation at ``y`` (its output was ``a``)."""
    return 1.0 - a * a if spec.activation == "tanh" else (y > 0).astype(np.float64)


def _normalise(spec, z, l, train, stats):
    """``(zn, mu, var)``: ``z`` standardised down the columns (batch norm) or
    across the rows (layer norm).  In training, batch norm also updates the
    running ``stats``."""
    if spec.norm == "layer":
        mu, var = z.mean(1, keepdims=True), z.var(1, keepdims=True)
    elif train or not stats:
        mu, var = z.mean(0), z.var(0)
        if stats:
            stats["mu%d" % l] = BN_MOMENTUM * stats["mu%d" % l] + (1 - BN_MOMENTUM) * mu
            stats["var%d" % l] = BN_MOMENTUM * stats["var%d" % l] + (1 - BN_MOMENTUM) * var
    else:
        mu, var = stats["mu%d" % l], stats["var%d" % l]
    return (z - mu) / np.sqrt(var + EPS), mu, var


def forward(spec, P, x, train=False, rng=None, masks=None, stats=None):
    """The batch ``x (N, inputs)`` through the network.

    ``train`` selects batch statistics for batch norm and applies dropout
    (masks drawn from ``rng``, or the given ``masks``, one ``(N, width)``
    array per hidden layer).  Returns a dict: ``h`` (the input of every dense
    layer: ``h[0]`` is ``x``), ``z`` (pre-activations), ``zn``, ``mu``, ``var``
    (normalisation), ``y`` (what the activation sees), ``a`` (its output,
    before dropout), ``m`` (the dropout masks), ``logits`` and ``p``.
    """
    L = spec.depth
    f = dict(h=[np.asarray(x, np.float64)], z=[], zn=[], mu=[], var=[], y=[], a=[], m=[])
    keep = 1.0 - spec.dropout
    for l in range(L):
        z = f["h"][l] @ P["W%d" % l] + P["b%d" % l]
        if spec.norm:
            zn, mu, var = _normalise(spec, z, l, train, stats)
            y = P["g%d" % l] * zn + P["beta%d" % l]
        else:
            zn = mu = var = None
            y = z
        a = _act(spec, y)
        out = a
        m = None
        if spec.dropout and train:
            m = masks[l] if masks is not None else (rng.random(a.shape) >= spec.dropout)
            out = a * m / keep
        if spec.skip(l):
            out = out + f["h"][l + 1 - spec.residual]
        for k, v in (("z", z), ("zn", zn), ("mu", mu), ("var", var), ("y", y), ("a", a), ("m", m)):
            f[k].append(v)
        f["h"].append(out)
    logits = f["h"][L] @ P["W%d" % L] + P["b%d" % L]
    e = np.exp(logits - logits.max(1, keepdims=True))
    f["logits"], f["p"] = logits, e / e.sum(1, keepdims=True)
    return f


def cross_entropy(p, y_onehot):
    """Mean ``-log p[label]`` over the batch."""
    return float(-np.mean(np.log(np.sum(p * y_onehot, 1) + 1e-300)))


def penalty(spec, P):
    """The weight-decay term of the objective, ``wd / 2 * sum of W^2``."""
    if not spec.weight_decay:
        return 0.0
    return 0.5 * spec.weight_decay * sum(float((v * v).sum()) for k, v in P.items()
                                         if k.startswith("W"))


def objective(spec, P, f, y_onehot):
    """What :func:`backward` differentiates: the cross-entropy plus the penalty."""
    return cross_entropy(f["p"], y_onehot) + penalty(spec, P)


def _norm_backward(spec, dy, zn, var, g):
    """Gradients through ``g * zn + beta``: ``(dz, dg, dbeta)`` for batch norm
    (statistics over axis 0) or layer norm (axis 1)."""
    axis = 1 if spec.norm == "layer" else 0
    n = dy.shape[axis]
    dzn = dy * g
    inv = 1.0 / np.sqrt(var + EPS)
    dz = inv / n * (n * dzn - dzn.sum(axis, keepdims=True)
                    - zn * (dzn * zn).sum(axis, keepdims=True))
    return dz, (dy * zn).sum(0), dy.sum(0)


def backward(spec, P, f, y_onehot):
    """Gradients of :func:`objective` for the batch of :func:`forward`.

    Returns ``(G, D)``: ``G`` has the parameters' keys; ``D["dh"][l]`` is the
    gradient at ``h[l]`` (``l = 0`` the input), ``D["dz"][l]`` the gradient at
    hidden layer ``l``'s pre-activation.
    """
    L, N = spec.depth, len(y_onehot)
    keep = 1.0 - spec.dropout
    dlogits = (f["p"] - y_onehot) / N
    G = {"W%d" % L: f["h"][L].T @ dlogits, "b%d" % L: dlogits.sum(0)}
    dh = [None] * (L + 1)
    dh[L] = dlogits @ P["W%d" % L].T
    dz_all = [None] * L
    for l in reversed(range(L)):
        d = dh[l + 1]
        if spec.skip(l):
            src = l + 1 - spec.residual
            dh[src] = d if dh[src] is None else dh[src] + d
        if f["m"][l] is not None:
            d = d * f["m"][l] / keep
        dy = d * _act_grad(spec, f["y"][l], f["a"][l])
        if spec.norm:
            dz, G["g%d" % l], G["beta%d" % l] = _norm_backward(spec, dy, f["zn"][l], f["var"][l],
                                                             P["g%d" % l])
        else:
            dz = dy
        G["W%d" % l] = f["h"][l].T @ dz
        G["b%d" % l] = dz.sum(0)
        dz_all[l] = dz
        back = dz @ P["W%d" % l].T
        dh[l] = back if dh[l] is None else dh[l] + back
    if spec.weight_decay:
        for l in range(L + 1):
            G["W%d" % l] = G["W%d" % l] + spec.weight_decay * P["W%d" % l]
    return G, dict(dh=dh, dz=dz_all)


def layer_stats(spec, f, D):
    """Three ``(L,)`` arrays, one entry per hidden layer ``l``: the std of its
    output ``h[l + 1]``, the RMS of the gradient reaching that output on the
    way back (``dh[l + 1]``), and the RMS of the gradient at its pre-activation."""
    L = spec.depth
    act = np.array([f["h"][l + 1].std() for l in range(L)])
    g = np.array([np.sqrt(np.mean(D["dh"][l + 1] ** 2)) for l in range(L)])
    dz = np.array([np.sqrt(np.mean(D["dz"][l] ** 2)) for l in range(L)])
    return act, g, dz


# ------------------------------------------------------------------ optimisers


class SGD:
    """Plain minibatch gradient descent, ``p = p - lr * g``, over a dict of
    parameters (the interface of :class:`voxanim.optim.Adam`)."""

    def __init__(self, params, lr):
        self.lr = lr

    def step(self, params, grads, lr=None):
        """One update of every key in ``params``."""
        lr = self.lr if lr is None else lr
        for k in params:
            params[k] = params[k] - lr * grads[k]


OPTIMISERS = {"sgd": SGD, "adam": Adam}


# ------------------------------------------------------------------ data


@dataclass
class Data:
    """An MNIST subset (``x`` rows are inputs, ``y`` integer labels in
    ``range(classes)``)."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    classes: int
    x_val: np.ndarray | None = None      # a validation set (for early stopping), if any
    y_val: np.ndarray | None = None

    def onehot(self, y):
        """Labels to one-hot rows."""
        return np.eye(self.classes)[y]

    def settings(self):
        """What identifies the data in a cache key: sizes and a checksum."""
        return dict(n_train=len(self.x_train), n_test=len(self.x_test), classes=self.classes,
                    n_val=None if self.x_val is None else len(self.x_val),
                    checksum=float(self.x_train.sum() + self.y_train.sum() + self.x_test.sum()))


def subset(n_train, n_test, classes=None, seed=0, cache_dir=None):
    """The first ``n_train`` / ``n_test`` images of a seeded shuffle of MNIST
    (of the given ``classes`` only, relabelled ``0, 1, ...``, when given)."""
    from .mnist import load
    xtr, ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    parts = []
    for x, y, n in ((xtr, ytr, n_train), (xte, yte, n_test)):
        if classes is not None:
            keep = np.isin(y, classes)
            x, y = x[keep], np.searchsorted(np.asarray(classes), y[keep])
        idx = rng.permutation(len(x))[:n]
        parts += [x[idx], y[idx]]
    return Data(*parts, classes=len(classes) if classes is not None else 10)


# ------------------------------------------------------------------ training


@dataclass
class Run:
    """One recorded training run of :func:`train`.

    Arrays with a leading ``S`` axis have one row per recorded step
    (``steps[0] == 0`` is the untrained network); ``L`` is the depth.
    """

    spec: Spec
    steps: np.ndarray            # (S,)
    batch_loss: np.ndarray       # (steps,) cross-entropy of each minibatch before its update
    train_loss: np.ndarray       # (S,) on the first EVAL_N training images
    test_loss: np.ndarray        # (S,)
    test_acc: np.ndarray         # (S,)
    val_loss: np.ndarray | None  # (S,) on the validation set, when the data has one
    act_std: np.ndarray          # (S, L) std of each hidden layer's output on the minibatch
    grad_rms: np.ndarray         # (S, L) RMS of the gradient reaching each hidden layer's output
    dz_rms: np.ndarray           # (S, L) RMS of the gradient at each pre-activation
    probe: dict                  # probe layer: z, y, a, m (S, batch, width); g, beta (S, width)
    field: np.ndarray | None     # (S, G) p(class 1) on the grid, when one was given
    P: dict                      # the trained parameters
    P0: dict                     # the initial ones
    stats: dict                  # batch norm's running statistics after training
    example_loss: np.ndarray | None = None   # (S, n_train) each example's loss, when asked

    @property
    def best(self):
        """Index of the recorded step with the lowest validation loss (early
        stopping's choice)."""
        return int(np.argmin(self.val_loss))


RECORDS = ("train_loss", "test_loss", "test_acc", "val_loss", "act_std", "grad_rms", "dz_rms",
           "probe_h", "probe_z", "probe_y", "probe_a", "probe_m", "probe_g", "probe_beta", "field",
           "example_loss")
EXAMPLE_CHUNK = 1000                      # rows per forward pass when scoring every example


PROBE_KEYS = ("z", "y", "a", "m")         # what a probe records, unless asked for less


def example_losses(spec, P, data, stats=None, x=None, y=None):
    """``(n,)`` the loss ``-log p[label]`` of every training example (or of
    the given ``x``, ``y``), in chunks of :data:`EXAMPLE_CHUNK` rows."""
    x = data.x_train if x is None else x
    y = data.y_train if y is None else y
    out = []
    for s in range(0, len(x), EXAMPLE_CHUNK):
        p = forward(spec, P, x[s:s + EXAMPLE_CHUNK], stats=stats)["p"]
        out.append(-np.log(p[np.arange(len(p)), y[s:s + EXAMPLE_CHUNK]] + 1e-300))
    return np.concatenate(out)


def train(spec, data, steps, batch, lr, opt="sgd", seed=0, every=None, probe_layer=None,
          probe_keys=PROBE_KEYS, grid=None, cache_dir=DEFAULT_CACHE, tag="deepnet",
          augment=None, example_loss=False, probe_rows=None) -> Run:
    """Train ``spec`` on ``data`` for ``steps`` minibatch updates and record the
    run (cached under every setting).

    ``opt`` is ``"sgd"`` or ``"adam"``; the schedule of recorded steps is every
    ``every`` steps, or geometric (:data:`SNAPSHOT_RATIO`) when ``every`` is
    None.  ``probe_layer`` records that hidden layer's pre-activations, scaled
    values, activations and dropout masks (``probe_keys`` picks fewer) on each
    recorded step's minibatch; ``grid`` ``(G, inputs)`` records ``p(class 1)``
    at those inputs.  Everything a
    recorded step shows is measured *before* that step's update.

    ``augment(rng, x)`` returns the minibatch ``x`` transformed (it must have a
    ``settings()`` for the cache key); it draws from a generator seeded apart
    from training's, so the minibatches and the initial weights are those of
    the un-augmented run.  ``example_loss`` records every training example's
    loss at each recorded step; ``probe_rows`` keeps only the first rows of
    the probe (``h`` at layer 0 is the minibatch's input, augmented).
    """
    settings = dict(spec.settings(), steps=steps, batch=batch, lr=lr, opt=opt, seed=seed,
                    every=every, probe_layer=probe_layer, data=data.settings(),
                    grid=None if grid is None else [list(grid.shape), float(np.abs(grid).sum())])
    if tuple(probe_keys) != PROBE_KEYS:
        settings["probe_keys"] = list(probe_keys)
    if augment is not None:
        settings["augment"] = augment.settings()
    if example_loss:
        settings["example_loss"] = True
    if probe_rows is not None:
        settings["probe_rows"] = probe_rows
    n = len(data.x_train)
    if every:
        marks = np.unique(np.concatenate([np.arange(0, steps, every), [steps]]))
    else:
        marks = snapshot_steps(steps, SNAPSHOT_RATIO)
    eye = data.onehot

    def evaluate(P, stats, x, y):
        f = forward(spec, P, x[:EVAL_N], stats=stats)
        return cross_entropy(f["p"], eye(y[:EVAL_N])), float(
            (f["p"].argmax(1) == y[:EVAL_N]).mean())

    def build():
        rng = np.random.default_rng(seed)
        arng = np.random.default_rng([seed, 1])          # augmentation's own stream
        P = init(spec, rng)
        P0 = {k: v.copy() for k, v in P.items()}
        stats = init_stats(spec)
        optim = OPTIMISERS[opt](P, lr)
        rec = {k: [] for k in RECORDS}
        batch_loss = []
        want = set(marks.tolist())

        def record(f, D):
            tr, _ = evaluate(P, stats, data.x_train, data.y_train)
            te, acc = evaluate(P, stats, data.x_test, data.y_test)
            rec["train_loss"].append(tr)
            rec["test_loss"].append(te)
            rec["test_acc"].append(acc)
            if data.x_val is not None:
                rec["val_loss"].append(evaluate(P, stats, data.x_val, data.y_val)[0])
            for k, v in zip(("act_std", "grad_rms", "dz_rms"), layer_stats(spec, f, D)):
                rec[k].append(v)
            if probe_layer is not None:
                for k in probe_keys:
                    if f[k][probe_layer] is not None:
                        rec["probe_" + k].append(f[k][probe_layer][:probe_rows])
                if spec.norm:
                    rec["probe_g"].append(P["g%d" % probe_layer].copy())
                    rec["probe_beta"].append(P["beta%d" % probe_layer].copy())
            if grid is not None:
                rec["field"].append(forward(spec, P, grid, stats=stats)["p"][:, 1])
            if example_loss:
                rec["example_loss"].append(example_losses(spec, P, data, stats))

        per_epoch = -(-n // batch)
        epochs = -(-(steps + 1) // per_epoch)
        for step, idx in enumerate(minibatches(rng, n, epochs, batch)):
            x, y = data.x_train[idx], eye(data.y_train[idx])
            if augment is not None:
                x = augment(arng, x)
            f = forward(spec, P, x, train=True, rng=rng, stats=stats)
            G, D = backward(spec, P, f, y)
            if step in want:
                record(f, D)
            if step == steps:
                break
            batch_loss.append(cross_entropy(f["p"], y))
            optim.step(P, G)
        out = dict(P, steps=marks, batch_loss=np.array(batch_loss))
        out.update({"P0_" + k: v for k, v in P0.items()})
        out.update({"stats_" + k: v for k, v in stats.items()})
        for k, v in rec.items():
            if v:
                dtype = bool if k == "probe_m" else (
                    np.float32 if k.startswith(("probe", "field", "example")) else np.float64)
                out[k] = np.array(v, dtype)
        return out

    d = cached_npz(tag, settings, build, cache_dir)
    meta = ("steps", "batch_loss", *RECORDS)
    P = {k: v for k, v in d.items() if k not in meta and not k.startswith(("P0_", "stats_"))}
    P0 = {k[3:]: v for k, v in d.items() if k.startswith("P0_")}
    stats = {k[6:]: v for k, v in d.items() if k.startswith("stats_")}
    probe = {k[6:]: d[k] for k in d if k.startswith("probe_")}
    return Run(spec, d["steps"], d["batch_loss"], d["train_loss"], d["test_loss"],
               d["test_acc"], d.get("val_loss"), d["act_std"], d["grad_rms"], d["dz_rms"],
               probe, d.get("field"), P, P0, stats, d.get("example_loss"))
