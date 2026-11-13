"""Making a trained network smaller: masks, pruning schedules, retraining under
a fixed mask, activation energies and distillation, all in plain numpy on the
MNIST models of :mod:`voxanim.mnist`.

* **Magnitude pruning** (:func:`magnitude_mask`): the smallest ``fraction`` of
  all the weights (over every layer at once, biases untouched) are set to
  zero.  :func:`prune_sweep` prunes the trained 784-64-32-10 MLP to 50, 80,
  90, 95 and 99% sparsity, measures the test accuracy straight after the cut
  and again after a short **fine-tune with the mask held** (:func:`fit_masked`:
  the pruned weights get no gradient and are zeroed after every Adam step, so
  they stay exactly 0), one-shot and iteratively (prune, fine-tune, prune
  further ...).
* **The lottery ticket** (:func:`lottery`): the 20%-dense mask found by
  pruning a trained network is applied to that network's *original
  initialisation* and the sparse network is trained from there, against a
  random mask of the same per-layer density and the dense network, on the
  same minibatches.
* **Structured pruning** (:func:`filter_energy`, :func:`unit_energy`,
  :func:`structured_sweep`): the CNN's filters and the MLP's hidden units
  ranked by their activation energy (the mean squared activation over the
  test set) and removed one at a time, least active first, with and without
  a short fine-tune.  A unit is removed by zeroing its incoming weights and
  bias (its activation is then exactly 0, so its outgoing weights do nothing).
* **Distillation** (:func:`kd_loss`, :func:`kd_grads`, :func:`distil`): a
  784-256-256-10 teacher's logits soften at temperature ``T`` into targets
  for a 784-32-10 student,

      L = (1 - a) CE(y, p) + a T^2 KL(softmax(z_t / T) || softmax(z / T)),

  whose gradient at the logits is ``(1 - a)(p - y) + a T (p_T - q_T)``
  (checked against finite differences in ``tests/test_prune.py``).  The
  student is trained on hard labels (a = 0), soft targets (a = 1) and both
  (a = 1/2), recording the test accuracy as it goes.

Every run is seeded and cached by :func:`voxanim.cache.cached_npz`; the dense
networks are the ones :func:`voxanim.mnist.train` caches.
"""

from __future__ import annotations

import numpy as np

from .cache import cached_npz
from .minibatch import batched, minibatches
from .mnist import (
    CLASSES,
    CNN_FILTERS,
    SIZES,
    _as_layers,
    _cnn_forward,
    _cnn_grads,
    _cnn_params,
    _forward,
    _init,
    _mlp_grads,
    cnn_train,
    load,
    train,
)
from .optim import Adam

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)     # sparsities the magnitude sweep prunes to
TICKET_DENSITY = 0.2                      # the lottery ticket keeps 20% of the weights
TEACHER_SIZES = (784, 256, 256, 10)
STUDENT_SIZES = (784, 32, 10)
TEMPERATURE = 4.0                         # distillation temperature
EVAL_CHUNK = 1000                         # test images per forward chunk
BATCH, LR = 128, 1e-3                     # the MLP's training recipe (mnist.train)


# ------------------------------------------------------------------ parameters as dicts


def layers_to_dict(params):
    """``[[W0, b0], ...]`` -> ``{"W0": ..., "b0": ..., ...}`` (copies)."""
    return {"%s%d" % (name, k): np.array(a) for k, layer in enumerate(params)
            for name, a in zip("Wb", layer, strict=True)}


def n_layers(P):
    """Dense layers in a parameter dict."""
    return len(P) // 2


def weight_keys(P):
    """The weight matrices' keys, in layer order."""
    return ["W%d" % k for k in range(n_layers(P))]


def mlp_predict(P, x):
    """Softmax probabilities ``(N, 10)`` of the MLP ``P`` on ``x``, in chunks."""
    layers = _as_layers(P, n_layers(P))
    return batched(lambda xs: _forward(layers, xs)[2], x, EVAL_CHUNK)


def accuracy(predict, x, y):
    """Fraction of ``y`` that ``predict(x).argmax(1)`` gets right."""
    return float((predict(x).argmax(1) == y).mean())


def mlp_accuracy(P, x, y):
    """Test accuracy of the MLP ``P``."""
    return accuracy(lambda xs: mlp_predict(P, xs), x, y)


def cnn_predict(P, x):
    """Softmax probabilities of the CNN ``P`` (a dict) on ``x``, in chunks."""
    return batched(lambda xs: _cnn_forward(_cnn_params(P), xs)["p"], x, EVAL_CHUNK)


def cnn_accuracy(P, x, y):
    """Test accuracy of the CNN ``P``."""
    return accuracy(lambda xs: cnn_predict(P, xs), x, y)


# ------------------------------------------------------------------ masks


def all_weights(P):
    """Every weight of ``P`` (no biases) as one flat array."""
    return np.concatenate([P[k].ravel() for k in weight_keys(P)])


def n_pruned(P, fraction):
    """How many weights pruning a ``fraction`` of them removes: ``round(fraction * n)``."""
    return int(round(fraction * all_weights(P).size))


def magnitude_threshold(P, fraction):
    """The ``|w|`` of the smallest weight *kept* when the smallest ``fraction``
    of all weights are pruned (0 when nothing is; infinity when all are):
    weights with ``|w|`` below it go."""
    w = np.sort(np.abs(all_weights(P)))
    k = n_pruned(P, fraction)
    if k <= 0:
        return 0.0
    return float(w[k]) if k < w.size else np.inf


def magnitude_mask(P, fraction):
    """Masks ``{"W0": bool array, ...}`` keeping all but the smallest ``fraction``
    of the weights, by ``|w|`` over every layer together (biases are kept)."""
    thr = magnitude_threshold(P, fraction)
    masks = {k: np.abs(P[k]) >= thr for k in weight_keys(P)}
    # ties at the threshold: drop exactly the count asked for, smallest first
    want = n_pruned(P, fraction)
    extra = sum(int((~m).sum()) for m in masks.values()) - want
    if extra < 0:
        tied = [(k, idx) for k in weight_keys(P)
                for idx in zip(*np.nonzero(masks[k] & (np.abs(P[k]) == thr)), strict=True)]
        for k, idx in tied[:-extra]:
            masks[k][idx] = False
    return masks


def random_mask(P, like, rng):
    """Random masks with the same number of kept weights per layer as ``like``."""
    masks = {}
    for k in weight_keys(P):
        keep = int(like[k].sum())
        m = np.zeros(P[k].size, bool)
        m[rng.permutation(P[k].size)[:keep]] = True
        masks[k] = m.reshape(P[k].shape)
    return masks


def apply_mask(P, masks):
    """A copy of ``P`` with the masked entries set to 0 (keys not in ``masks`` untouched)."""
    return {k: (v * masks[k] if k in masks else np.array(v)) for k, v in P.items()}


def density(masks):
    """Fraction of the masked entries that are kept."""
    kept = sum(int(m.sum()) for m in masks.values())
    return kept / sum(m.size for m in masks.values())


def count_pruned(masks):
    """``(pruned, total)`` over the masked entries."""
    total = sum(m.size for m in masks.values())
    return total - sum(int(m.sum()) for m in masks.values()), total


# ------------------------------------------------------------------ training under a mask


def fit_masked(P, grads, x, y, steps, rng, masks, batch=BATCH, lr=LR, record=None, every=0,
               classes=CLASSES):
    """``steps`` Adam updates of ``P`` (in place) on minibatches of ``(x, y)``
    with the masked entries held at exactly 0: their gradients are zeroed
    before the step (so Adam's moments never move them) and they are zeroed
    again after it.

    ``record(step)`` is called before step 0 and after every ``every`` steps
    (and the last); ``y`` is integer labels in ``range(classes)``,
    ``grads(P, x_batch, y_onehot)``.
    """
    opt = Adam(P, lr)
    eye = np.eye(classes)
    per_epoch = -(-len(x) // batch)
    epochs = -(-steps // per_epoch)
    if record is not None:
        record(0)
    for step, idx in enumerate(minibatches(rng, len(x), epochs, batch), 1):
        g = grads(P, x[idx], eye[y[idx]])
        for k, m in masks.items():
            g[k] = g[k] * m
        opt.step(P, g)
        for k, m in masks.items():
            P[k] = P[k] * m
        if record is not None and every and (step % every == 0 or step == steps):
            record(step)
        if step == steps:
            break
    return P


def _rng(seed, salt):
    """A generator for one fine-tune, distinct from the training's."""
    return np.random.default_rng([seed, salt])


# ------------------------------------------------------------------ magnitude pruning

FINETUNE_STEPS = 400          # Adam steps of a fine-tune (under an epoch of 469)


def prune_sweep(levels=LEVELS, finetune_steps=FINETUNE_STEPS, seed=0, cache_dir=None):
    """Prune the trained MLP to each sparsity in ``levels`` (cached).

    Returns a dict of arrays over the levels: ``acc_pruned`` (straight after
    the cut), ``acc_tuned`` (after ``finetune_steps`` with the mask held),
    ``acc_iter`` (iterative: each level reached from the previous level's
    fine-tuned network, with the same fine-tune at every stage), ``thresholds``
    (the ``|w|`` cut at), ``pruned`` (weights removed, one-shot), plus
    ``acc_dense``, ``total`` (weights) and ``levels``; and the networks
    themselves: ``tuned{n}_W0`` ... (one-shot, fine-tuned, at level ``n``) and
    ``iter{n}_W0`` ... (the iterative chain at level ``n``).
    """
    settings = dict(model="mlp", sizes=list(SIZES), levels=list(levels),
                    finetune_steps=finetune_steps, batch=BATCH, lr=LR, seed=seed,
                    scope="global-magnitude", threshold="kth-smallest")

    def build():
        params, acc = train(seed=seed, cache_dir=cache_dir)
        P = layers_to_dict(params)
        xtr, ytr, xte, yte = load(cache_dir)
        out = dict(acc_dense=np.array(acc), levels=np.array(levels),
                   total=np.array(all_weights(P).size))
        rec = {k: [] for k in ("acc_pruned", "acc_tuned", "acc_iter", "thresholds", "pruned")}
        Q = dict(P)                                           # the iterative chain
        for n, f in enumerate(levels):
            masks = magnitude_mask(P, f)
            cut = apply_mask(P, masks)
            rec["thresholds"].append(magnitude_threshold(P, f))
            rec["pruned"].append(count_pruned(masks)[0])
            rec["acc_pruned"].append(mlp_accuracy(cut, xte, yte))
            fit_masked(cut, _mlp_grads, xtr, ytr, finetune_steps, _rng(seed, 1 + n), masks)
            rec["acc_tuned"].append(mlp_accuracy(cut, xte, yte))
            out.update({"tuned%d_%s" % (n, k): v for k, v in cut.items()})
            # iterative: the same fraction of the *original* count, cut from
            # the network fine-tuned at the previous level (weights already at
            # 0 are the smallest, so they stay pruned)
            m = magnitude_mask(Q, f)
            Q = apply_mask(Q, m)
            fit_masked(Q, _mlp_grads, xtr, ytr, finetune_steps, _rng(seed, 101 + n), m)
            rec["acc_iter"].append(mlp_accuracy(Q, xte, yte))
            out.update({"iter%d_%s" % (n, k): v for k, v in Q.items()})
        out.update({k: np.array(v) for k, v in rec.items()})
        return out

    return cached_npz("prune_magnitude", settings, build, cache_dir)


# ------------------------------------------------------------------ the lottery ticket

TICKET_STEPS = 1500           # training steps of the three sparse / dense networks
TICKET_EVERY = 25             # steps between recorded test accuracies


def lottery(seed=0, density=TICKET_DENSITY, steps=TICKET_STEPS, every=TICKET_EVERY,
            cache_dir=None):
    """The lottery-ticket experiment for one ``seed`` (cached).

    The dense MLP trained from ``seed`` (:func:`voxanim.mnist.train`) is pruned
    to ``density`` by magnitude; that mask is applied to the *initial* weights
    the same seed drew and the sparse network is trained for ``steps``
    (``ticket``), as are the initial weights under a random mask of the same
    per-layer density (``random``) and the untouched initial weights
    (``dense``), all on the same minibatches.

    Returns ``steps`` (recorded), ``acc_ticket``, ``acc_random``, ``acc_dense``
    (test accuracy at each), the masks (``mask_W0`` ..., ``rand_W0`` ...),
    the initial weights (``init_W0`` ...), the trained dense weights the mask
    came from (``trained_W0`` ...) and each run's final weights
    (``final_ticket_W0`` ...).
    """
    settings = dict(model="mlp", sizes=list(SIZES), density=density, steps=steps,
                    every=every, batch=BATCH, lr=LR, seed=seed, scope="global-magnitude",
                    threshold="kth-smallest", random_mask="per-layer-density")

    def build():
        params, _acc = train(seed=seed, cache_dir=cache_dir)
        trained = layers_to_dict(params)
        rng = np.random.default_rng(seed)
        P0 = layers_to_dict(_init(rng, SIZES))            # exactly what training started from
        masks = magnitude_mask(trained, 1.0 - density)
        rand = random_mask(P0, masks, np.random.default_rng([seed, 7]))
        xtr, ytr, xte, yte = load(cache_dir)
        out = {}
        for name, m in (("ticket", masks), ("random", rand), ("dense", {})):
            P = apply_mask(P0, m)
            marks, accs = [], []

            def record(step, P=P, marks=marks, accs=accs):
                marks.append(step)
                accs.append(mlp_accuracy(P, xte, yte))

            fit_masked(P, _mlp_grads, xtr, ytr, steps, np.random.default_rng([seed, 11]), m,
                       record=record, every=every)
            out["acc_" + name] = np.array(accs)
            out["steps"] = np.array(marks)
            out.update({"final_%s_%s" % (name, k): v for k, v in P.items()})
        out.update({"mask_" + k: v for k, v in masks.items()})
        out.update({"rand_" + k: v for k, v in rand.items()})
        out.update({"init_" + k: v for k, v in P0.items()})
        out.update({"trained_" + k: v for k, v in trained.items()})
        return out

    return cached_npz("prune_lottery", settings, build, cache_dir)


# ------------------------------------------------------------------ structured pruning


def filter_energy(P, x):
    """``(F,)`` mean squared activation of each conv filter's ReLU map over ``x``."""
    layers = _cnn_params(P)

    def energy(xs):
        a = _cnn_forward(layers, xs)["a1"]
        return (a * a).mean(axis=(1, 2))
    return batched(energy, x, EVAL_CHUNK).mean(0)


def unit_energy(P, x, layer=0):
    """``(width,)`` mean squared activation of hidden layer ``layer``'s units over ``x``."""
    layers = _as_layers(P, n_layers(P))

    def energy(xs):
        acts = _forward(layers, xs)[0]
        h = acts[layer + 1]
        return h * h
    return batched(energy, x, EVAL_CHUNK).mean(0)


def unit_masks(P, removed, layer=0):
    """Masks removing hidden units ``removed`` of layer ``layer``: their incoming
    column of ``W{layer}`` and their bias are zeroed (so their activation is
    exactly 0 and their outgoing weights contribute nothing)."""
    keep = np.ones(P["W%d" % layer].shape[1], bool)
    keep[list(removed)] = False
    return {"W%d" % layer: np.broadcast_to(keep, P["W%d" % layer].shape).copy(),
            "b%d" % layer: keep.copy()}


def filter_masks(P, removed):
    """Masks removing conv filters ``removed``: their kernel and bias are zeroed."""
    keep = np.ones(CNN_FILTERS, bool)
    keep[list(removed)] = False
    return {"Wc": np.broadcast_to(keep, P["Wc"].shape).copy(), "bc": keep.copy()}


UNIT_FINETUNE = 100           # Adam steps after each hidden unit removed
FILTER_FINETUNE = 300         # Adam steps after each filter removed


def structured_sweep(unit_steps=UNIT_FINETUNE, filter_steps=FILTER_FINETUNE, seed=0,
                     cache_dir=None):
    """Remove the CNN's filters and the MLP's first-layer units one at a time,
    least energetic first (cached).

    Returns ``filter_energy`` ``(4,)``, ``filter_order`` (least active first),
    ``filter_acc`` / ``filter_acc_tuned`` ``(4,)`` (0, 1, 2, 3 removed; the
    tuned chain fine-tunes ``filter_steps`` after each removal), and the same
    for the 64 hidden units: ``unit_energy``, ``unit_order``, ``unit_acc`` /
    ``unit_acc_tuned`` ``(64,)`` (0 .. 63 removed), plus ``unit_W0_tuned``
    ``(784, 64)`` the layer-1 weights after half the units are gone and
    fine-tuned, and ``cnn_Wc_tuned`` likewise.
    """
    settings = dict(unit_steps=unit_steps, filter_steps=filter_steps, batch=BATCH, lr=LR,
                    seed=seed, energy="mean-square-test", order="fixed")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        out = {}
        # ---- the CNN's filters
        params, _acc = cnn_train(cache_dir=cache_dir)
        P = {k: np.array(v) for k, v in zip(("Wc", "bc", "Wd", "bd"),
                                            (params[0][0], params[0][1], params[1][0],
                                             params[1][1]), strict=True)}
        e = filter_energy(P, xte)
        order = np.argsort(e)
        acc, tuned = [cnn_accuracy(P, xte, yte)], [cnn_accuracy(P, xte, yte)]
        Q = dict(P)
        for n in range(1, CNN_FILTERS):
            removed = order[:n]
            acc.append(cnn_accuracy(apply_mask(P, filter_masks(P, removed)), xte, yte))
            m = filter_masks(Q, removed)
            Q = apply_mask(Q, m)
            fit_masked(Q, _cnn_grads, xtr, ytr, filter_steps, _rng(seed, 200 + n), m, batch=64,
                       lr=2e-3)
            tuned.append(cnn_accuracy(Q, xte, yte))
            if n == CNN_FILTERS // 2:
                out.update({"cnn_half_" + k: v.copy() for k, v in Q.items()})
        out.update(filter_energy=e, filter_order=order, filter_acc=np.array(acc),
                   filter_acc_tuned=np.array(tuned))
        # ---- the MLP's hidden units
        params, _acc = train(seed=seed, cache_dir=cache_dir)
        P = layers_to_dict(params)
        e = unit_energy(P, xte)
        order = np.argsort(e)
        width = len(e)
        acc, tuned = [mlp_accuracy(P, xte, yte)], [mlp_accuracy(P, xte, yte)]
        Q = dict(P)
        for n in range(1, width):
            removed = order[:n]
            acc.append(mlp_accuracy(apply_mask(P, unit_masks(P, removed)), xte, yte))
            m = unit_masks(Q, removed)
            Q = apply_mask(Q, m)
            fit_masked(Q, _mlp_grads, xtr, ytr, unit_steps, _rng(seed, 300 + n), m)
            tuned.append(mlp_accuracy(Q, xte, yte))
            if n == width // 2:
                out.update({"mlp_half_" + k: v.copy() for k, v in Q.items()})
        out.update(unit_energy=e, unit_order=order, unit_acc=np.array(acc),
                   unit_acc_tuned=np.array(tuned))
        return out

    return cached_npz("prune_structured", settings, build, cache_dir)


# ------------------------------------------------------------------ distillation


def softmax(z):
    """Row-wise softmax of ``z``."""
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def kd_loss(P, x, y_onehot, teacher_logits, T=TEMPERATURE, alpha=0.5):
    """The distillation objective (mean over the batch), see the module."""
    z = mlp_logits(P, x)
    p = softmax(z)
    hard = -np.mean(np.sum(y_onehot * np.log(p + 1e-300), 1))
    q_T, p_T = softmax(teacher_logits / T), softmax(z / T)
    soft = np.mean(np.sum(q_T * (np.log(q_T + 1e-300) - np.log(p_T + 1e-300)), 1))
    return float((1 - alpha) * hard + alpha * T * T * soft)


def mlp_logits(P, x):
    """The MLP's logits ``(N, 10)`` on ``x``."""
    return _forward(_as_layers(P, n_layers(P)), x)[1][-1]


def dense_backward(layers, acts, pre, delta):
    """Gradients ``{"W0", "b0", ...}`` of a dense ReLU network given
    ``delta = dL/dz`` at the output (``acts``, ``pre`` from :func:`_forward`)."""
    G = {}
    for k in range(len(layers) - 1, -1, -1):
        W, _b = layers[k]
        G["W%d" % k], G["b%d" % k] = acts[k].T @ delta, delta.sum(0)
        if k > 0:
            delta = (delta @ W.T) * (pre[k - 1] > 0)
    return G


def kd_grads(P, x, y_onehot, teacher_logits, T=TEMPERATURE, alpha=0.5):
    """Gradients of :func:`kd_loss`: at the logits,
    ``((1 - a)(p - y) + a T (p_T - q_T)) / N``."""
    layers = _as_layers(P, n_layers(P))
    acts, pre, p = _forward(layers, x)
    z = pre[-1]
    p_T, q_T = softmax(z / T), softmax(teacher_logits / T)
    delta = ((1 - alpha) * (p - y_onehot) + alpha * T * (p_T - q_T)) / len(x)
    return dense_backward(layers, acts, pre, delta)


STUDENT_STEPS = 3000          # training steps of each student (64 passes over 3,000)
STUDENT_EVERY = 50
TEACHER_EPOCHS = 4
TRANSFER_N = 3000             # training images the students see (a seeded subset)
TRANSFER_SEED = 5             # the permutation the subset is the head of
RECIPES = (("hard labels", 0.0), ("soft targets", 1.0), ("both", 0.5))   # (name, alpha)


def teacher(cache_dir=None):
    """The 784-256-256-10 teacher: ``(P, accuracy)`` (trained by :func:`voxanim.mnist.train`)."""
    params, acc = train(epochs=TEACHER_EPOCHS, sizes=TEACHER_SIZES, cache_dir=cache_dir)
    return layers_to_dict(params), acc


def transfer_set(xtr, ytr, n_train, seed=TRANSFER_SEED):
    """The first ``n_train`` images of a seeded permutation of the training set."""
    sel = np.random.default_rng(seed).permutation(len(xtr))[:n_train]
    return xtr[sel], ytr[sel]


def distil(n_train=TRANSFER_N, T=TEMPERATURE, steps=STUDENT_STEPS, every=STUDENT_EVERY, seed=0,
           cache_dir=None):
    """Train the student three ways from one initialisation on the same
    minibatches of ``n_train`` training images (cached): hard labels, the
    teacher's soft targets at temperature ``T``, and both (a = 1/2).

    Returns ``steps`` and ``acc_hard``, ``acc_soft``, ``acc_both`` (test
    accuracy at each recorded step), ``acc_teacher``, the final student
    weights (``hard_W0`` ...) and ``teacher_logits_test`` ``(10000, 10)``.
    """
    settings = dict(student=list(STUDENT_SIZES), teacher=list(TEACHER_SIZES),
                    teacher_epochs=TEACHER_EPOCHS, T=T, alphas=[a for _, a in RECIPES],
                    steps=steps, every=every, batch=BATCH, lr=LR, seed=seed, n_train=n_train,
                    transfer_seed=TRANSFER_SEED)

    def build():
        Pt, acc_t = teacher(cache_dir)
        xtr, ytr, xte, yte = load(cache_dir)
        xtr, ytr = transfer_set(xtr, ytr, n_train)
        logits_tr = batched(lambda xs: mlp_logits(Pt, xs), xtr, EVAL_CHUNK)
        out = dict(acc_teacher=np.array(acc_t),
                   teacher_logits_test=batched(lambda xs: mlp_logits(Pt, xs), xte, EVAL_CHUNK))
        P0 = layers_to_dict(_init(np.random.default_rng(seed), STUDENT_SIZES))
        xz = np.concatenate([xtr, logits_tr], 1)     # the teacher's logits ride along
        for (_name, alpha), key in zip(RECIPES, ("hard", "soft", "both"), strict=True):
            P = {k: v.copy() for k, v in P0.items()}
            marks, accs = [], []

            def record(step, P=P, marks=marks, accs=accs):
                marks.append(step)
                accs.append(mlp_accuracy(P, xte, yte))

            def grads(Q, xb, yb, alpha=alpha):
                return kd_grads(Q, xb[:, :784], yb, xb[:, 784:], T, alpha)

            fit_masked(P, grads, xz, ytr, steps, np.random.default_rng([seed, 11]), {},
                       record=record, every=every)
            out["acc_" + key] = np.array(accs)
            out["steps"] = np.array(marks)
            out.update({"%s_%s" % (key, k): v for k, v in P.items()})
        return out

    return cached_npz("prune_distil", settings, build, cache_dir)
