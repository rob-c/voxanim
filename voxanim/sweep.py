"""A column sweep on the timeline: ``n`` steps paced by a :class:`~voxanim.conv.Tempo`.

The network animations step through layers, columns or rows one at a time,
slow at first and faster later; :class:`Sweep` turns a start time and a
tempo into the per-step schedule and answers, for any instant, how many
steps are done, which is under way and how far it has got.
"""

from __future__ import annotations

import numpy as np

from .conv import Tempo


class Sweep:
    """``n`` steps starting at ``t0``, each as long as ``tempo`` says."""

    def __init__(self, t0, n, tempo=Tempo()):
        self.n = n
        self.durs = tempo.durations(n)
        self.starts = t0 + np.concatenate([[0.0], np.cumsum(self.durs)[:-1]])
        self.t0, self.t1 = t0, t0 + float(self.durs.sum())

    @property
    def duration(self):
        """Seconds the whole sweep takes."""
        return self.t1 - self.t0

    def at(self, t):
        """``(steps finished, current step, its progress 0..1)`` at time ``t``;
        the current step is -1 before the sweep and ``n`` after it."""
        if t < self.t0:
            return 0, -1, 0.0
        if t >= self.t1:
            return self.n, self.n, 0.0
        k = int(np.searchsorted(self.starts, t, side="right") - 1)
        return k, k, (t - self.starts[k]) / self.durs[k]
