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
