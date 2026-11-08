"""Orthographic camera.

World axes (one unit = one voxel):

    x  -> tensor width / columns     (projects up-right on screen)
    y  -> tensor channels            (projects down-right, towards the viewer)
    z  -> up                         (tensor rows run downwards along -z)

The projection is a 2x3 matrix ``M`` (pixels per world unit, screen y down).
Its null space is the viewing direction, so every point along ``view`` lands
on the same pixel; ``depth(p) = p . view`` grows towards the viewer and is
what the z-buffer compares.

The defaults reproduce the originals' framing: a dimetric view with both
horizontal axes at ~26 deg and a slightly squashed vertical (``vscale``) that
the originals' camera has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Camera:
    width: int = 960
    height: int = 540
    ppu: float = 17.0                 # pixels per world unit along a horizontal axis
    azimuth: float = 45.0
    elevation: float = 29.4
    vscale: float = 0.84
    target: tuple = (0.0, 0.0, 0.0)   # world point that lands on ``anchor``
    anchor: tuple = (0.5, 0.5)        # as a fraction of the frame
    mirror: bool = False              # looks at the scene reflected in z
    zoom: float = 1.0                 # composition scale about the frame centre

    # -- derived -----------------------------------------------------------
    @property
    def M(self):
        a, e = math.radians(self.azimuth), math.radians(self.elevation)
        ex = (math.cos(a), -math.sin(a) * math.sin(e))
        ey = (math.sin(a), math.cos(a) * math.sin(e))
        ez = (0.0, -math.cos(e) * self.vscale * (-1.0 if self.mirror else 1.0))
        return np.array([ex, ey, ez], np.float64).T * (self.ppu * self.zoom)

    @property
    def view(self):
        """Unit vector pointing from the scene towards the viewer."""
        M = self.M
        v = np.cross(M[0], M[1])
        if (v[2] < 0) != self.mirror:
            v = -v
        return v / np.linalg.norm(v)

    @property
    def offset(self):
        a = np.array([self.anchor[0] * self.width, self.anchor[1] * self.height])
        return a - self.M @ np.asarray(self.target, np.float64)

    # -- API ---------------------------------------------------------------
    def project(self, p):
        """World point(s) ``(..., 3)`` -> screen pixels ``(..., 2)``."""
        p = np.asarray(p, np.float64)
        return p @ self.M.T + self.offset

    def depth(self, p):
        return np.asarray(p, np.float64) @ self.view

    def unproject(self, sx, sy, z=0.0):
        """The world point at height ``z`` that projects to pixel ``(sx, sy)``."""
        M = self.M
        A = M[:, :2]
        rhs = np.array([sx, sy]) - self.offset - M[:, 2] * z
        x, y = np.linalg.solve(A, rhs)
        return np.array([x, y, z])

    def frac(self, fx, fy, z=0.0):
        """:meth:`unproject` with the pixel given as a fraction of the frame
        (layout coordinates: ``zoom`` pulls them towards the centre, so a
        whole composition scales into the broadcast safe area)."""
        k = self.zoom
        return self.unproject((0.5 + (fx - 0.5) * k) * self.width,
                              (0.5 + (fy - 0.5) * k) * self.height, z)

    def scaled(self, k):
        """Same view rendered at ``k`` times the resolution (supersampling)."""
        return replace(self, width=int(round(self.width * k)),
                       height=int(round(self.height * k)), ppu=self.ppu * k)

    def mirrored(self):
        return replace(self, mirror=not self.mirror)

    def with_target(self, target):
        return replace(self, target=tuple(float(v) for v in target))
