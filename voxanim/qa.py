"""Layout QA: checks every animation's scenes, at many instants, without
rendering.

* text outside the **title-safe** area (5% margins, SMPTE ST 2046-1 for 16:9),
* geometry outside the **action-safe** area (3.5% margins),
* text overlapping other text,
* text drawn over voxels (a caption colliding with the thing it labels, or
  with something else).

Everything is measured at the 960 x 540 reference resolution; the renderer is
resolution independent, so the result holds at 4K and 8K too.

    python3 -m voxanim qa                 # every animation
    python3 -m voxanim qa conv2d mlp-9-training-step --samples 400
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import presets
from .scene import font

W, H = 960, 540
TITLE = 0.05
ACTION = 0.035


def text_boxes(scene):
    """``[(text, (x0, y0, x1, y1))]`` in reference pixels."""
    out = []
    for t in scene.texts:
        if t.opacity < 0.05 or not t.s.strip():
            continue
        f = font(t.kind, t.size * H)
        l, tp, r, b = f.getbbox(t.s, anchor=t.anchor)
        x, y = t.x * W, t.y * H
        out.append((t.s, (x + l, y + tp, x + r, y + b)))
    return out


def voxel_disks(scene, cam):
    """Screen centres and radii of the solid voxels (particles excluded)."""
    from .layout import solid_disks
    return solid_disks(scene, cam)


def _text_opacity(scene, s):
    return max((t.opacity for t in scene.texts if t.s == s), default=0.0)


GAP = 4.0                    # minimum clear space between two pieces of text (ref px)


def _overlap(a, b, slack=-GAP):
    """True when two text boxes come closer than ``-slack`` pixels."""
    return (min(a[2], b[2]) - max(a[0], b[0]) > slack and
            min(a[3], b[3]) - max(a[1], b[1]) > slack)


def check_scene(scene, cam):
    issues = []
    boxes = text_boxes(scene)
    lo_x, hi_x, lo_y, hi_y = W * TITLE, W * (1 - TITLE), H * TITLE, H * (1 - TITLE)
    for s, b in boxes:
        if b[0] < lo_x - 0.5 or b[2] > hi_x + 0.5 or b[1] < lo_y - 0.5 or b[3] > hi_y + 0.5:
            issues.append(("title-safe", s))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlap(boxes[i][1], boxes[j][1]):
                issues.append(("text/text", "%s | %s" % (boxes[i][0], boxes[j][0])))
    # text over strong lines (cages, receptive-field frames)
    for p0, p1, _width, _col, op, _b in scene.lines:
        if op < 0.5:
            continue
        pts = cam.project(np.linspace(p0, p1, 24))
        for s, b in boxes:
            if _text_opacity(scene, s) < 0.3:
                continue
            inside = (pts[:, 0] > b[0] + 1) & (pts[:, 0] < b[2] - 1) & \
                (pts[:, 1] > b[1] + 1) & (pts[:, 1] < b[3] - 1)
            if inside.any():
                issues.append(("text/line", s))
    c, r = voxel_disks(scene, cam)
    if len(c):
        for s, b in boxes:
            dx = np.maximum(np.maximum(b[0] - c[:, 0], c[:, 0] - b[2]), 0)
            dy = np.maximum(np.maximum(b[1] - c[:, 1], c[:, 1] - b[3]), 0)
            if np.any(np.hypot(dx, dy) < r * 0.6) and \
                    _text_opacity(scene, s) > 0.3:
                issues.append(("text/voxel", s))
        a = ACTION
        out = (c[:, 0] - r < W * a) | (c[:, 0] + r > W * (1 - a)) | \
            (c[:, 1] - r < H * a) | (c[:, 1] + r > H * (1 - a))
        if out.any():
            issues.append(("action-safe", "%d voxels" % int(out.sum())))
    return issues


def check(name, samples=120, verbose=True):
    """Check one animation; returns the number of distinct issues.

    Besides the per-frame checks, a caption that geometry keeps hidden (it
    yields to things passing under it) for most of the time it exists is
    reported as ``hidden``: it was placed somewhere it cannot be read."""
    anim = presets.get(name).build()
    seen = defaultdict(list)
    shown, hidden = defaultdict(int), defaultdict(int)
    for t in np.linspace(0, anim.duration, samples, endpoint=False):
        sc = anim.scene(float(t))
        for kind, what in check_scene(sc, anim.cam):
            seen[(kind, what)].append(float(t))
        for tx in sc.texts:
            shown[tx.s] += 1
            hidden[tx.s] += tx.yielded < 0.3
    strict = not hasattr(anim, "stages")      # network animations: nothing may cover text
    for s, n in shown.items():
        if n >= 4 and hidden[s] > 0.5 * n:
            seen[("hidden", s)].append(0.0)
        elif strict and hidden[s]:
            seen[("covered", s)].append(float(hidden[s]))
    if verbose:
        status = "ok" if not seen else "%d issue(s)" % len(seen)
        print("%-30s %s" % (name, status))
        for (kind, what), ts in sorted(seen.items(), key=lambda kv: kv[1][0]):
            span = "t=%.1f" % ts[0] if len(ts) == 1 else "t=%.1f..%.1f (%d samples)" % (
                ts[0], ts[-1], len(ts))
            print("    %-12s %-60s %s" % (kind, what[:60], span))
    return len(seen)
