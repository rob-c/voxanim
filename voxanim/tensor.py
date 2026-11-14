"""A tensor drawn as a block of voxels.

Indexing is ``[row, col, channel]`` like an HWC activation.  In world space a
block occupies

    col     -> +x
    channel -> +y   (channel 0 at the back, the last channel faces the viewer)
    row     -> -z   (row 0 on top)

Channels can be split into groups with a gap between them (grouped and
depthwise convolutions "explode" the input this way; conv3d stores each input
channel as a slab of depth cells).  Rows and columns can be spaced out with a
``pitch`` > 1 (a dilated kernel, whose taps have gaps between them).

Per-frame state (opacity, white highlight, glow, which cells exist, positional
offsets) lives in plain numpy arrays that animation code writes to;
:meth:`Block.emit` turns the block into scene voxels, culling the ones that are
completely hidden by their neighbours.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .scene import INVISIBLE, OPAQUE


class Block:
    """An ``(H, W, C)`` tensor as voxels; see the module docstring for the axes.

    ``group`` is the channels per group (gapped by ``gap`` world units),
    ``pitch`` the (row, col) cell spacing in voxels, ``scale`` the voxel size.
    """

    def __init__(self, shape, color=C.INPUT, group=None, gap=0.0, pitch=(1, 1), scale=1.0,
                 name=""):
        self.H, self.W, self.C = (int(v) for v in shape)
        self.name = name
        self.group = int(group) if group else self.C   # channels per group
        self.gap = float(gap)
        self.pitch = (float(pitch[0]), float(pitch[1]))  # row, col spacing
        self.scale = float(scale)          # display size of a voxel (magnified kernels)
        self.origin = np.zeros(3)
        self.base = np.empty((self.H, self.W, self.C, 3), np.float32)
        self.base[...] = C.rgb(color)
        self.reset()

    # -- per-frame state ------------------------------------------------------
    def reset(self):
        """Back to fully present, opaque, unlit and undisplaced."""
        shp = (self.H, self.W, self.C)
        self.opacity = np.ones(shp, np.float32)
        self.white = np.zeros(shp, np.float32)
        self.glow = np.zeros(shp, np.float32)
        self.present = np.ones(shp, bool)
        self.offset = None              # optional (H, W, C, 3) world displacement
        self.shadow = True

    # -- geometry -----------------------------------------------------------
    @property
    def groups(self):
        """Number of channel groups."""
        return (self.C + self.group - 1) // self.group

    # One index (the common case, thousands of times a frame) is computed in
    # plain Python -- exact integer arithmetic, then the same double multiply --
    # and returned as the np.float64 the array path gives; arrays as before.
    def chan_y(self, k):
        """World y (relative to origin) of the back face of channel ``k``."""
        if isinstance(k, (int, np.integer)):
            k = int(k)
            return np.float64((k + (k // self.group) * self.gap) * self.scale)
        k = np.asarray(k)
        return (k + (k // self.group) * self.gap) * self.scale

    def col_x(self, j):
        """World x (relative to origin) of the left face of column ``j``."""
        if isinstance(j, (int, np.integer)):
            return np.float64(int(j) * self.pitch[1] * self.scale)
        return np.asarray(j) * self.pitch[1] * self.scale

    def row_z(self, i):
        """World z (relative to origin) of the bottom face of row ``i``."""
        if isinstance(i, (int, np.integer)):
            return np.float64((self.H - 1 - int(i)) * self.pitch[0] * self.scale)
        return (self.H - 1 - np.asarray(i)) * self.pitch[0] * self.scale

    @property
    def extent(self):
        """World size of the block's bounding box ``(x, y, z)``."""
        s = self.scale
        return np.array([self.col_x(self.W - 1) + s, self.chan_y(self.C - 1) + s,
                         self.row_z(0) + s], np.float64)

    @property
    def center(self):
        """World centre of the bounding box."""
        return self.origin + self.extent / 2.0

    def place(self, center):
        """Move the block so its bounding-box centre is ``center``."""
        self.origin = np.asarray(center, np.float64) - self.extent / 2.0
        return self

    def cell_center(self, i, j, k):
        """World centre of cell ``[i, j, k]`` (ignoring ``offset``)."""
        h = 0.5 * self.scale
        return self.origin + np.array([self.col_x(j) + h, self.chan_y(k) + h,
                                       self.row_z(i) + h], np.float64)

    def box(self, rows, cols, chans):
        """World ``(lo, hi)`` of the half-open index ranges given as (start, stop)."""
        (i0, i1), (j0, j1), (k0, k1) = rows, cols, chans
        lo = self.origin + np.array([self.col_x(j0), self.chan_y(k0), self.row_z(i1 - 1)],
                                    np.float64)
        s = self.scale
        hi = self.origin + np.array([self.col_x(j1 - 1) + s, self.chan_y(k1 - 1) + s,
                                     self.row_z(i0) + s], np.float64)
        return lo, hi

    # -- emission -------------------------------------------------------------
    def _hidden(self, solid, view):
        """Cells whose three camera-facing neighbours are all solid (and touching)."""
        hidden = np.ones_like(solid)
        # axis 1 (col) -> world x, axis 2 (chan) -> world y, axis 0 (row) -> -z
        for axis, comp, flip in ((1, 0, 1), (2, 1, 1), (0, 2, -1)):
            if (axis == 0 and self.pitch[0] != 1) or (axis == 1 and self.pitch[1] != 1):
                return np.zeros_like(solid)        # spaced out: nothing touches
            step = 1 if view[comp] * flip > 0 else -1
            nb = np.zeros_like(solid)
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            if step > 0:
                dst[axis], src[axis] = slice(0, -1), slice(1, None)
            else:
                dst[axis], src[axis] = slice(1, None), slice(0, -1)
            nb[tuple(dst)] = solid[tuple(src)]
            if axis == 2 and self.gap > 0:
                # A neighbour across a group gap does not touch this cell.
                k = np.arange(self.C)
                kn = k + step
                same = (kn >= 0) & (kn < self.C) & (kn // self.group == k // self.group)
                nb &= same[None, None, :]
            hidden &= nb
        return hidden

    def emit(self, scene, view, mview=None):
        """Append this block's voxels to ``scene``, culling the cells hidden
        from the camera (``view``) and, if ``mview`` is given, from the floor
        reflection's mirrored camera too."""
        vis = self.present & (self.opacity > INVISIBLE)
        if not vis.any():
            return
        solid = vis & (self.opacity >= OPAQUE)
        moved = None
        if self.offset is not None:
            moved = np.abs(self.offset).sum(-1) > 1e-9
            solid &= ~moved                        # displaced cells occlude nothing
        draw = vis & ~self._hidden(solid, view)
        if mview is not None:
            # Cells hidden from the camera can still show up in the floor
            # reflection, which sees the block from below.
            draw_r = vis & ~self._hidden(solid, mview)
        else:
            draw_r = draw
        if moved is not None:
            draw |= vis & moved
            draw_r |= vis & moved
        idx = np.argwhere(draw | draw_r)
        if len(idx) == 0:
            return
        i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
        h = 0.5 * self.scale
        centers = self.origin + np.stack(
            [self.col_x(j) + h, self.chan_y(k) + h, self.row_z(i) + h], -1)
        if self.offset is not None:
            centers = centers + self.offset[i, j, k]
        # the whole block at once: the values scene.voxel would record, per cell
        shown = draw[i, j, k]
        scene.voxel_batch(centers, self.base[i, j, k], self.opacity[i, j, k],
                          self.white[i, j, k], self.glow[i, j, k],
                          shown & bool(self.shadow), draw_r[i, j, k], shown, float(self.scale))
