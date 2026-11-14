""":mod:`voxanim.deepnet` on a GPU: the same network as tensors, trained with
autograd (see :mod:`voxanim.accel`).

Only the parameter updates move to the device.  Everything the run records
is still measured by the numpy code: at a recorded step the weights (and
batch norm's running statistics) come back as numpy arrays and
:func:`voxanim.deepnet.forward` / :func:`~voxanim.deepnet.backward` measure
that step's minibatch -- with the same dropout masks -- exactly as in a
numpy run.  The random draws (initial weights, minibatch order, dropout
masks, augmentation) are the numpy run's, made by the same generators in the
same order.

:func:`forward_t` and :func:`loss_t` are checked against the numpy model in
float64 (on the CPU) in ``tests/test_accel.py``, option by option.
"""

from __future__ import annotations

import numpy as np

from . import accel
from . import deepnet as D


def act_t(name, y):
    """Activation ``name`` of :data:`voxanim.deepnet.ACTIVATIONS` on a tensor."""
    t = accel.torch()
    if name == "tanh":
        return t.tanh(y)
    if name == "relu":
        return t.relu(y)
    if name == "sigmoid":
        return t.sigmoid(y)
    if name == "leaky_relu":
        return t.where(y > 0, y, D.LEAKY_SLOPE * y)
    if name == "gelu":
        return 0.5 * y * (1.0 + t.tanh(D.GELU_C * (y + D.GELU_K * y * y * y)))
    if name == "silu":
        return y * t.sigmoid(y)
    if name == "elu":
        return t.where(y > 0, y, D.ELU_ALPHA * (t.exp(t.clamp(y, max=0.0)) - 1.0))
    raise ValueError("unknown activation %r" % name)


def forward_t(spec, Pt, x, masks=None, stats=None, train=True):
    """The logits of batch ``x``.  ``masks`` are the dropout masks (tensors,
    one per hidden layer) when training with dropout; ``stats`` (tensors) is
    batch norm's running mean and variance, updated in training as
    :func:`voxanim.deepnet.forward` updates them."""
    t = accel.torch()
    L = spec.depth
    keep = 1.0 - spec.dropout
    h = [x]
    for l in range(L):
        z = h[l] @ Pt["W%d" % l] + Pt["b%d" % l]
        if spec.norm:
            if spec.norm == "layer":
                mu, var = z.mean(1, keepdim=True), z.var(1, unbiased=False, keepdim=True)
            elif train or not stats:
                mu, var = z.mean(0), z.var(0, unbiased=False)
                if stats:
                    with t.no_grad():
                        stats["mu%d" % l].mul_(D.BN_MOMENTUM).add_((1 - D.BN_MOMENTUM) * mu)
                        stats["var%d" % l].mul_(D.BN_MOMENTUM).add_((1 - D.BN_MOMENTUM) * var)
            else:
                mu, var = stats["mu%d" % l], stats["var%d" % l]
            y = Pt["g%d" % l] * ((z - mu) / t.sqrt(var + D.EPS)) + Pt["beta%d" % l]
        else:
            y = z
        out = act_t(spec.activation, y)
        if spec.dropout and train:
            out = out * masks[l] / keep
        if spec.skip(l):
            out = out + h[l + 1 - spec.residual]
        h.append(out)
    return h[L] @ Pt["W%d" % L] + Pt["b%d" % L]


def loss_t(spec, Pt, logits, target):
    """``(objective, plain cross-entropy)``: the objective's differentiable part
    (cross-entropy against the target, which label smoothing softens, plus
    weight decay and -- unless it is the proximal step's job -- the L1
    term), and the cross-entropy against the target itself."""
    t = accel.torch()
    logp = t.log_softmax(logits, 1)
    ce = -(target * logp).sum(1).mean()
    obj = -(spec.target(target) * logp).sum(1).mean() if spec.label_smoothing else ce
    Ws = [v for k, v in Pt.items() if k.startswith("W")]
    if spec.weight_decay:
        obj = obj + 0.5 * spec.weight_decay * sum((W * W).sum() for W in Ws)
    if spec.l1 and not spec.l1_prox:
        obj = obj + spec.l1 * sum(W.abs().sum() for W in Ws)
    return obj, ce


def constrain_t(spec, Pt, lr):
    """:func:`voxanim.deepnet.constrain` on tensors, in place."""
    t = accel.torch()
    with t.no_grad():
        if spec.l1 and spec.l1_prox:
            for l in range(spec.depth + 1):
                W = Pt["W%d" % l]
                W.copy_(t.sign(W) * t.clamp(W.abs() - lr * spec.l1, min=0.0))
        if spec.max_norm:
            for l in range(spec.depth):
                W = Pt["W%d" % l]
                norms = t.sqrt((W * W).sum(0))
                W.mul_(t.clamp(spec.max_norm / t.clamp(norms, min=1e-300), max=1.0))


def draw_masks(spec, rng, n):
    """The dropout masks :func:`voxanim.deepnet.forward` would draw for a batch
    of ``n``: one ``rng.random`` per hidden layer, in layer order."""
    return [rng.random((n, spec.sizes[l + 1])) >= spec.dropout for l in range(spec.depth)]


def train_loop(spec, data, P, stats, steps, batch, lr, opt, rng, arng, augment, want, record,
               dtype=None, dev=None):
    """The minibatch loop of :func:`voxanim.deepnet.train` with the updates on
    :func:`voxanim.accel.device`; ``P`` and ``stats`` (numpy) hold the state
    whenever ``record(f, D)`` is called, and the final state on return.
    Returns the per-step minibatch cross-entropies (before each update)."""
    from .minibatch import minibatches
    t = accel.torch()
    dev = accel.device() if dev is None else dev
    dtype = dtype or accel.float_dtype()
    Pt = accel.params_to_device(P, dev, dtype)
    St = {k: accel.to_device(v, dev, dtype) for k, v in stats.items()}
    optim = accel.Adam(Pt, lr) if opt == "adam" else None
    xd = accel.to_device(data.x_train, dev, dtype)
    yd = accel.to_device(data.onehot(data.y_train), dev, dtype)
    n = len(data.x_train)
    per_epoch = -(-n // batch)
    epochs = -(-(steps + 1) // per_epoch)
    losses = []
    for step, idx in enumerate(minibatches(rng, n, epochs, batch)):
        it = t.as_tensor(np.asarray(idx), device=dev)
        if augment is not None:
            xa = augment(arng, data.x_train[idx])
            x = accel.to_device(xa, dev, dtype)
        else:
            xa, x = None, xd[it]
        y = yd[it]
        masks = draw_masks(spec, rng, len(idx)) if spec.dropout else None
        mt = [accel.to_device(m, dev, dtype) for m in masks] if masks else None
        logits = forward_t(spec, Pt, x, mt, St, train=True)
        if step in want:
            P.update(accel.to_numpy(Pt))
            stats.update(accel.to_numpy(St))
            xn = data.x_train[idx] if xa is None else xa
            yn = data.onehot(data.y_train[idx])
            f = D.forward(spec, P, xn, train=True, masks=masks, stats={})
            record(f, D.backward(spec, P, f, yn)[1])
        if step == steps:
            break
        obj, ce = loss_t(spec, Pt, logits, y)
        losses.append(ce.detach())
        grads = dict(zip(Pt.keys(), t.autograd.grad(obj, list(Pt.values()))))
        if optim is not None:
            optim.step(Pt, grads)
        else:
            with t.no_grad():
                for k, p in Pt.items():
                    p.sub_(lr * grads[k])
        constrain_t(spec, Pt, lr)
    P.update(accel.to_numpy(Pt))
    stats.update(accel.to_numpy(St))
    return np.array([float(v) for v in t.stack(losses).cpu()]) if losses else np.array([])
