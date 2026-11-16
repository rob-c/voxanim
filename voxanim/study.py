"""Measured experiment records: the numbers an animation of the "next 100" draws.

A study is data only - four panels of arrays with their captions and provenance,
inspectable as JSON/NPZ. What draws it is its :class:`~voxanim.mech.Mechanism`
(:func:`voxanim.next_catalog.drawing`). Each study once also *was* its animation, four
generic chart panels; those were replaced by the drawn mechanisms in November 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Panel:
    title: str
    caption: str
    kind: str = "curve"
    data: dict = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass
class Study:
    title: str
    subtitle: str
    panels: tuple[Panel, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # Experiments can replace a panel after sharing a comparison setup.
        self.panels = list(self.panels)

    def validate(self):
        if len(self.panels) != 4:
            raise ValueError("A study needs four measured panels")
        for panel in self.panels:
            if panel.kind not in {"curve", "bars", "matrix", "points", "images"}:
                raise ValueError("Unknown panel kind: " + panel.kind)
            if not panel.data:
                raise ValueError("A panel must contain measurements: " + panel.title)
            for value in panel.data.values():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype.kind in "fci"
                    and not np.isfinite(value).all()
                ):
                    raise ValueError("Non-finite panel: " + panel.title)
        return self

