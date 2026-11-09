"""Shared test helpers.

Tests that need downloaded data are skipped (not passed) when it is absent:
``needs_mnist`` (11 MB) and ``needs_gpt2`` (550 MB).  ``python3 -m voxanim
prepare`` downloads both and trains every model, after which the whole suite
runs.  Everything else needs nothing but the repository.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from voxanim import presets
from voxanim.cache import DEFAULT_CACHE
from voxanim.scene import Style

DRAFT = Style(ss=2, motion_blur=False)


def mnist_cached():
    """True when the MNIST dataset has been downloaded."""
    return os.path.exists(os.path.join(DEFAULT_CACHE, "mnist.npz"))


def gpt2_cached():
    """True when the GPT-2 small weights have been downloaded."""
    from voxanim import gpt2
    return os.path.exists(os.path.join(gpt2.DIR, "model.safetensors"))


needs_mnist = pytest.mark.skipif(not mnist_cached(),
                                 reason="MNIST not downloaded (python3 -m voxanim prepare)")
needs_gpt2 = pytest.mark.skipif(not gpt2_cached(),
                                reason="GPT-2 weights not downloaded (python3 -m voxanim prepare)")


def _module(p):
    fn = getattr(p.build, "func", p.build)            # functools.partial or a function
    return fn.__module__


# The convolution animations (registered by presets.py and lessons.py) and
# the network animations that need no dataset; everything else is trained on
# or evaluates MNIST, except GPT-2.
CONV_PRESETS = [n for n, p in presets.REGISTRY.items()
                if _module(p) in ("voxanim.presets", "voxanim.lessons")]
NO_DATA = set(CONV_PRESETS) | {"rnn-training-step", "lstm-training-step", "rnn-batch-training",
                               "gradient-descent", "decision-tree"}


def data_marks(name):
    """pytest marks for preset ``name``: what it needs downloaded."""
    if name.startswith("gpt2-"):
        return [needs_gpt2]
    return [] if name in NO_DATA else [needs_mnist]


def available(name):
    """True when preset ``name`` can be built without downloading anything."""
    if name.startswith("gpt2-"):
        return gpt2_cached()
    return name in NO_DATA or mnist_cached()


def fd_check(loss, P, G, keys, e=1e-6, tol=1e-5):
    """Compare gradients ``G`` with central differences of ``loss`` at the
    largest and a middle entry of each parameter in ``keys``."""
    for k in keys:
        g = np.asarray(G[k])
        for idx in {np.unravel_index(np.argmax(np.abs(g)), g.shape),
                    tuple(np.array(g.shape) // 2)}:
            plus = {a: np.array(b) for a, b in P.items()}
            minus = {a: np.array(b) for a, b in P.items()}
            plus[k][idx] += e
            minus[k][idx] -= e
            fd = (loss(plus) - loss(minus)) / (2 * e)
            assert abs(fd - g[idx]) <= tol * max(1e-3, abs(fd)), (k, idx, fd, g[idx])
