"""Animations 172-175: entropy, impurity and loss (see :mod:`voxanim.study_info`).

The two tree animations are drawn in :mod:`voxanim.mech_info_trees`, the network's
weights and its loss in :mod:`voxanim.mech_info_nets`; this module gathers them for
:func:`voxanim.next_catalog.drawing`.
"""

from __future__ import annotations

from .mech_info_nets import LogitsLossAndNats, WeightEntropy
from .mech_info_trees import EntropyAndGiniByHand, EntropyDownATree

ANIMATIONS = {
    "entropy-and-gini-by-hand": EntropyAndGiniByHand,
    "entropy-down-a-tree": EntropyDownATree,
    "weight-entropy": WeightEntropy,
    "logits-loss-and-nats": LogitsLossAndNats,
}
