"""Adam, shared by every model that is trained here.

The update is written once, in the exact operation order the models were
first trained with, so cached weights stay bit-for-bit reproducible:

    m = 0.9 m + 0.1 g
    v = 0.999 v + 0.001 g^2
    p = p - lr * (m / (1 - 0.9^t)) / (sqrt(v / (1 - 0.999^t)) + 1e-8)
"""

from __future__ import annotations

import numpy as np

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
# Written as literals: 1 - 0.9 is 0.09999999999999998 in floating point, and
# the weights were trained with 0.1 exactly.
GAIN1, GAIN2 = 0.1, 0.001


class Adam:
    """Adam over a dict of parameter arrays; :meth:`step` updates the dict.

    Usage::

        opt = Adam(P, lr=1e-3)
        for batch in batches:
            opt.step(P, gradients(P, batch))
    """

    def __init__(self, params, lr):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        """One update of every key in ``params`` (``grads`` has the same keys)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        t = self.t
        for k in params:
            g = grads[k]
            self.m[k] = BETA1 * self.m[k] + GAIN1 * g
            self.v[k] = BETA2 * self.v[k] + GAIN2 * g ** 2
            params[k] = params[k] - lr * (self.m[k] / (1 - BETA1 ** t)) / (
                np.sqrt(self.v[k] / (1 - BETA2 ** t)) + EPS)


# ------------------------------------------------------------------ the other optimisers
#
# Added for the optimisation animations (:mod:`voxanim.landscape`); the same
# ``step(P, g)`` interface as :class:`Adam`, whose code above is left exactly
# as it was (cached weights depend on it).  Each class states its update rule.


class SGD:
    """Plain gradient descent: ``p = p - lr * g``."""

    def __init__(self, params, lr):
        self.lr = lr
        self.t = 0

    def step(self, params, grads, lr=None):
        """One update of every key in ``params`` (``grads`` has the same keys)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            params[k] = params[k] - lr * grads[k]


class Momentum:
    """Gradient descent with momentum (Polyak's heavy ball)::

        m = beta m + g
        p = p - lr * m

    ``m`` is a running sum of the gradients, discounted by ``beta`` a step:
    along a direction where the gradient keeps its sign it builds up to
    ``1 / (1 - beta)`` times the gradient (10 x for 0.9); where the gradient
    flips sign every step it cancels.
    """

    def __init__(self, params, lr, beta=0.9):
        self.lr, self.beta = lr, beta
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        """One update of every key in ``params`` (``grads`` has the same keys)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            self.m[k] = self.beta * self.m[k] + grads[k]
            params[k] = params[k] - lr * self.m[k]


class RMSProp:
    """Gradient descent with each parameter's step divided by the running
    root-mean-square of its gradient (Hinton, 2012)::

        v = beta v + (1 - beta) g^2
        p = p - lr * g / (sqrt(v) + eps)

    A parameter whose gradient is consistently large is slowed down, one
    whose gradient is small is sped up: every parameter moves about ``lr``
    per step whatever the scale of its gradient.
    """

    def __init__(self, params, lr, beta=0.9, eps=1e-8):
        self.lr, self.beta, self.eps = lr, beta, eps
        self.t = 0
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        """One update of every key in ``params`` (``grads`` has the same keys)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        for k in params:
            g = grads[k]
            self.v[k] = self.beta * self.v[k] + (1 - self.beta) * g ** 2
            params[k] = params[k] - lr * g / (np.sqrt(self.v[k]) + self.eps)


class AdamW:
    """Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019)::

        m = beta1 m + (1 - beta1) g
        v = beta2 v + (1 - beta2) g^2
        p = p - lr * ( m / (1 - beta1^t) / (sqrt(v / (1 - beta2^t)) + eps) + wd * p )

    The decay ``lr * wd * p`` is applied to the parameter directly, outside
    the moment estimates, so every weight shrinks by the same fraction
    ``lr * wd`` a step.  (Adding ``wd * p`` to the gradient instead -- "Adam
    + L2" -- puts the decay through the division by ``sqrt(v)``, so weights
    with large gradients barely decay; see :mod:`voxanim.adamw_anim`.)  The
    Adam part is written with the same literals as :class:`Adam`, so with
    ``wd = 0`` the two agree bit for bit.
    """

    def __init__(self, params, lr, weight_decay=0.0):
        self.lr, self.wd = lr, weight_decay
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        """One update of every key in ``params`` (``grads`` has the same keys)."""
        self.t += 1
        lr = self.lr if lr is None else lr
        t = self.t
        for k in params:
            g = grads[k]
            self.m[k] = BETA1 * self.m[k] + GAIN1 * g
            self.v[k] = BETA2 * self.v[k] + GAIN2 * g ** 2
            # Adam's own expression, verbatim, then the decay: with wd = 0 the
            # result is bit for bit Adam's.
            adam = lr * (self.m[k] / (1 - BETA1 ** t)) / (
                np.sqrt(self.v[k] / (1 - BETA2 ** t)) + EPS)
            params[k] = params[k] - adam - lr * self.wd * params[k]
