"""The minibatch loop and chunked evaluation shared by the MNIST models.

Every model here is trained the same way: each epoch draws one permutation
of the training set from the model's seeded generator and cuts it into
consecutive batches.  Keeping that loop in one place keeps the order of the
random draws -- and so the cached weights -- identical across models.
"""

from __future__ import annotations

import numpy as np


def minibatches(rng, n, epochs, batch):
    """Yield index arrays: per epoch, ``rng.permutation(n)`` cut into ``batch`` s.

    The permutation is drawn lazily, when the epoch's first batch is
    requested, so random draws made between batches by the caller (noise,
    samples) interleave exactly as in a hand-written double loop.  The last
    batch of an epoch may be shorter.
    """
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            yield order[s:s + batch]


def chunk_mean(fn, n, size=1000):
    """``mean(fn(slice))`` over consecutive slices of ``range(n)``, as a float.

    The mean of the per-chunk values (not weighted by chunk size): used for
    test metrics, where ``n`` is a multiple of ``size``.
    """
    return float(np.mean([fn(slice(s, s + size)) for s in range(0, n, size)]))


def batched(fn, x, size):
    """``fn`` applied to consecutive ``size`` -row chunks of ``x``, concatenated."""
    return np.concatenate([fn(x[s:s + size]) for s in range(0, len(x), size)])
