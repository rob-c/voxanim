"""The visual alphabet, the layout and the code panel of the "Blocks and code" series
(catalogue 185-214): every animation of the series is a :class:`Blocks`.

The series is for students who find this hard (TODO.md, "Blocks and code"), so every
animation looks the same and moves slowly. This module holds what they share.

**The alphabet** (:data:`DATA` ...): each kind of thing has one colour and one shape, so
that shape backs up colour:

* data: blue cards (:meth:`Blocks.card`) holding blue tiles (:meth:`Blocks.tiles`,
  deeper: a bigger number);
* weights: orange tiles (``tiles(..., ink=WEIGHT)``); biases: a thin orange strip
  (:meth:`Blocks.bias_strip`);
* gradients: red tiles just behind the weight they belong to (:meth:`Blocks.grad_tiles`);
* the loss: a purple meter (:meth:`Blocks.meter`);
* the optimizer: a green hand (:meth:`Blocks.hand`);
* the training loop: a circular track (:meth:`Blocks.loop_map`, also the corner map);
* the learning rate: a dial (:meth:`Blocks.dial`);
* a number set to 0 (ReLU, dropout): a grey tile (:data:`ZERO`).

**The layout** (layout fractions; the frame is 960 x 540 reference px): the title block
at the top; the picture on the left (:data:`PIC`, x 0.05-0.475, y from 0.35); a heading
over it (:data:`HEAD_Y`), which reads "last time: ..." for the first
:attr:`Blocks.LAST_SECONDS` and "next: ..." for the last :attr:`Blocks.NEXT_SECONDS`;
the code panel on the right (:data:`PANEL`): at most 8 lines of the real program,
monospace, the running line boxed, and an output slot under them; the training-loop map
below it, the station of this animation lit, "you are here" beside it; the evidence tag at
the bottom right. Everything but the picture is drawn by :meth:`Blocks.draw` from what
the subclass declares, at full strength all through, so nothing blanks between phases.

**The code** is never typed into a drawing: :func:`code_lines` reads a named region
(``# region: NAME`` ... ``# endregion``) of ``examples/mnist_blocks.py`` (:data:`SCRIPT`,
the course's program) or ``examples/blocks_parts.py`` (:data:`PARTS`, the blocks the
classifier does not use, and the broken programs) or ``examples/mnist_cnn.py``
(:data:`CNN_SCRIPT`, the same program with a CNN for its model; its regions are the
script's names with ``cnn-`` in front), comments left out.

A subclass sets ``TITLE``, ``PHASES``, ``LAST``, ``NEXT`` and, one per phase, ``CODE``
(a :class:`Code`), ``STATION`` (one of :data:`STATIONS`, or None), ``EVIDENCE``
(:func:`exact` / :func:`measured`) and ``HEADINGS`` (a string, or ``((u, text), ...)``);
it draws its picture in ``phase_<n>(sc, u, op)`` as for any :class:`~voxanim.mech.Mechanism`,
and what must stay on screen across phases (a shape, a running total) in :meth:`steady`.
"""

from __future__ import annotations

import io
import textwrap
import tokenize
from bisect import bisect_right
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from pathlib import Path

import numpy as np

from . import color as C
from . import layout
from . import study_course as course
from .diagram import Diagram, draw_image, draw_strip
from .easing import ease_in_out, remap, smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    GRID,
    PAPER,
    RED,
    TEAL,
    TEXT,
    TRACK,
    VIOLET,
    Mechanism,
    dark,
    soft,
)

# ====================================================================== the code
EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SCRIPT = EXAMPLES / "mnist_blocks.py"
PARTS = EXAMPLES / "blocks_parts.py"
CNN_SCRIPT = EXAMPLES / "mnist_cnn.py"   # the same program with a CNN (regions "cnn-...")
MAX_LINES = 8                   # lines of code on screen at once
MAX_CHARS = 41                  # and their width, in characters of the monospace font


def _regions(path):
    """Every ``# region: NAME`` ... ``# endregion`` of ``path``: ``{name: [raw lines]}``.
    Regions may nest; a region's lines leave out the markers of those inside it."""
    found, stack = {}, []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("# region:"):
            name = s[len("# region:"):].strip()
            if name in found or any(n == name for n, _ in stack):
                raise ValueError("%s: region %r twice" % (Path(path).name, name))
            stack.append((name, []))
        elif s == "# endregion":
            if not stack:
                raise ValueError("%s: # endregion without a region" % Path(path).name)
            name, lines = stack.pop()
            found[name] = lines
        else:
            for _name, lines in stack:
                lines.append(line)
    if stack:
        raise ValueError("%s: region %r never ends" % (Path(path).name, stack[-1][0]))
    return found


@cache
def regions():
    """``{name: (file, raw lines)}`` over the example files (names are unique)."""
    out = {}
    for path in (SCRIPT, PARTS, CNN_SCRIPT):
        for name, lines in _regions(path).items():
            if name in out:
                raise ValueError("region %r is in two example files" % name)
            out[name] = (path, tuple(lines))
    return out


def _uncomment(lines):
    """``lines`` with their comments cut off (by the tokenizer, so a ``#`` in a string
    stays), trailing spaces and blank or comment-only lines dropped, then dedented."""
    src = "\n".join(lines) + "\n"
    cut = {}
    for tok in tokenize.generate_tokens(io.StringIO(textwrap.dedent(src)).readline):
        if tok.type == tokenize.COMMENT:
            cut[tok.start[0]] = tok.start[1]
    out = []
    for n, line in enumerate(textwrap.dedent(src).splitlines(), 1):
        s = (line[:cut[n]] if n in cut else line).rstrip()
        if s.strip():
            out.append(s)
    return tuple(textwrap.dedent("\n".join(out)).splitlines())


@cache
def code_lines(name, comments=False):
    """The lines of code of region ``name`` (see the module docstring), dedented, exactly
    as in the file but without comments and blank lines (``comments``: as in the file)."""
    try:
        _path, lines = regions()[name]
    except KeyError:
        raise KeyError("no region %r in %s, %s or %s (regions: %s)" % (
            name, SCRIPT.name, PARTS.name, CNN_SCRIPT.name,
            ", ".join(sorted(regions())))) from None
    if comments:
        return tuple(textwrap.dedent("\n".join(lines)).splitlines())
    return _uncomment(lines)


def code_line(name, contains):
    """The one line of region ``name`` containing ``contains``."""
    hits = [s for s in code_lines(name) if contains in s]
    if len(hits) != 1:
        raise ValueError("region %r: %d lines contain %r" % (name, len(hits), contains))
    return hits[0]


def code_file(name):
    """The file region ``name`` is in, as the panel names it (``examples/...``)."""
    return "examples/" + regions()[name][0].name


def program(path=SCRIPT):
    """Every line of code of the whole file, in order, without its docstring, comments,
    region markers or blank lines (the whole program on one screen)."""
    lines = Path(path).read_text().splitlines()
    end = next(i for i, s in enumerate(lines[1:], 1) if s.rstrip().endswith('"""'))
    return _uncomment(lines[end + 1:])


# ====================================================================== the alphabet
DATA = BLUE                     # data: blue cards and tiles
WEIGHT = AMBER                  # weights: orange tiles
BIAS = AMBER                    # biases: a thin orange strip
GRAD = RED                      # gradients: red tiles behind their weight
LOSS = VIOLET                   # the loss: a purple meter
OPTIM = GREEN                   # the optimizer: a green hand
RATE = TEAL                     # the learning rate: a dial
LOOP = AXIS                     # the training loop: a grey circular track
ZERO = C.rgb("#9a9da1")         # a number set to 0: a grey tile
# a digit's tiles are shaded from the paper to full ink as Normalize leaves them
DATA_LO = (0.0 - course.MEAN) / course.STD          # -0.4242
DATA_HI = (1.0 - course.MEAN) / course.STD          # 2.8215
STATIONS = ("data", "forward", "loss", "backward", "step")
STATION_INK = dict(data=DATA, forward=WEIGHT, loss=LOSS, backward=GRAD, step=OPTIM)

# ====================================================================== the layout
PIC = (0.055, 0.475)             # the picture's x extent; it starts below the heading
PIC_TOP, PIC_BOTTOM = 0.36, 0.90
HEAD_X, HEAD_Y = 0.055, 0.315    # the heading: one short line over the picture
PANEL = (0.495, 0.95, 0.295, 0.735)          # the code panel: x0, x1, top, bottom
CODE_Y, CODE_STEP, OUT_Y = 0.368, 0.041, 0.703
MAP = (0.735, 0.832)            # the loop map's centre
MAP_HW, MAP_HH = 128.0, 25.0    # its track: half width, half height (px)
TAG_Y = 0.925                   # the evidence tag
SIZE, CODE, HEAD, BIG, HUGE = TEXT, TEXT, 0.034, 0.048, 0.060
INK = layout.STRONG_INK
QUIET = dark(C.rgb("#80848a"), 0.25)        # secondary text: the tag, the file name
FADE_TEXT = 0.3                 # seconds a heading or a tag fades out, and the next in


def exact(what):
    """An evidence tag: exact arithmetic."""
    return "exact arithmetic: " + what


def measured(what):
    """An evidence tag: measured on MNIST."""
    return "measured on MNIST: " + what


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic: ", "measured on MNIST: "))


# ====================================================================== words and numbers
def commas(n):
    return format(int(round(float(n))), ",")


def shape(dims):
    """``[64, 1, 28, 28]``."""
    return "[" + ", ".join(str(int(d)) for d in dims) + "]"


def torch_size(dims):
    """``torch.Size([64, 1, 28, 28])``, as ``print(x.shape)`` writes it."""
    return "torch.Size(%s)" % shape(dims)


def val(v, places=2):
    """A value as a short label, with a real minus sign: ``−0.42``."""
    s = "%.*f" % (places, float(v))
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def words(text):
    """How many words of explanation ``text`` is (numbers, code and shapes not counted)."""
    return sum(1 for w in text.replace(":", " ").split() if any(c.isalpha() for c in w)
               and not any(c in w for c in "()[]._="))


def sample(a, rows, cols):
    """A big 2-D array shown as its top-left ``rows x cols`` corner: ``(corner, cut_rows,
    cut_cols)`` (whether rows or columns were left out)."""
    a = np.atleast_2d(np.asarray(a, float))
    return a[:rows, :cols], a.shape[0] > rows, a.shape[1] > cols


def shades(values, ink, lo=None, hi=None, zero=None):
    """Tile colours: paper at ``lo`` to ``ink`` at ``hi`` (deeper: bigger); ``zero``: a
    mask of tiles drawn grey (a number set to 0)."""
    v = np.asarray(values, np.float32)
    lo = float(v.min()) if lo is None else float(lo)
    hi = float(v.max()) if hi is None else float(hi)
    s = np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)[..., None]
    rgb = PAPER + (np.asarray(ink, np.float32) * 0.92 - PAPER) * s
    if zero is not None:
        rgb = np.where(np.asarray(zero, bool)[..., None], np.asarray(ZERO, np.float32), rgb)
    return rgb.astype(np.float32)


@dataclass(frozen=True)
class Code:
    """What the code panel shows in one phase: region ``region``'s lines (a tuple of
    region names: their lines one after the other, from one file; ``pick``: the indices of
    those to show, default all), the running line from each ``u`` of ``run`` (``(u,
    line)``, an index into the lines shown; line None: none), and in the output slot each
    ``(u, text)`` of ``out`` from that ``u`` on ("" clears it; "\\n" breaks a line)."""

    region: str | tuple
    pick: tuple | None = None
    run: tuple = ((0.0, 0),)
    out: tuple = ()

    def names(self):
        return (self.region,) if isinstance(self.region, str) else tuple(self.region)

    def lines(self):
        lines = tuple(s for name in self.names() for s in code_lines(name))
        return lines if self.pick is None else tuple(lines[i] for i in self.pick)


class _Timeline:
    """A value that changes at given times: ``[(t, value), ...]``, sorted."""

    def __init__(self, steps):
        self.ts = [t for t, _ in steps]
        self.vs = [v for _, v in steps]

    def at(self, t):
        i = bisect_right(self.ts, t) - 1
        return self.vs[max(i, 0)]

    def fade(self, t, width=FADE_TEXT):
        """1 away from a change, dipping to 0 at each change (out, then in)."""
        i = bisect_right(self.ts, t) - 1
        op = 1.0
        if i > 0:
            op = min(op, smoothstep(self.ts[i], self.ts[i] + width, t))
        if i + 1 < len(self.ts):
            op = min(op, 1.0 - smoothstep(self.ts[i + 1] - width, self.ts[i + 1], t))
        return op

    def last_change(self, t):
        """``(time, before, after)`` of the last change at or before ``t``."""
        i = bisect_right(self.ts, t) - 1
        if i <= 0:
            return None, self.vs[0], self.vs[0]
        return self.ts[i], self.vs[i - 1], self.vs[i]


def _merge(samples):
    """``[(t, v)]`` samples as the times where ``v`` changes."""
    out = []
    for t, v in samples:
        if not out or out[-1][1] != v:
            out.append((t, v))
    return out


# ====================================================================== the base class
class Blocks(Mechanism):
    """An animation of the series: :class:`~voxanim.mech.Mechanism` plus the furniture
    every one of them shares (see the module docstring)."""

    LAST = ""                   # what the one before showed; "" for the first
    NEXT = ""                   # what the one after shows
    LAST_SECONDS = 4.0
    NEXT_SECONDS = 5.0
    CODE: tuple = ()
    STATION: tuple = ()
    EVIDENCE: tuple = ()
    HEADINGS: tuple = ()
    TICK = 0.05                 # seconds between the samples the furniture's timelines use

    def __init__(self, study=None):
        # Mechanism.__init__, with the composition at full size: the series lays its
        # picture, code and map out over the whole title-safe area (zoom 1, where the other
        # animations keep to 0.9 of it), so that 41 characters of code fit beside a picture
        Diagram.__init__(self, self.TITLE or (study.title if study is not None else ""), "",
                         10.0)
        self.cam = replace(self.cam, zoom=1.0)
        self.Mr = self.cam.M
        self.Mi = np.linalg.pinv(self.Mr)
        self.view = self.cam.view
        self.study = study
        self.hold, self.reset = 2.5, 2.0
        self.setup()
        t, self.spans = 0.0, []
        for _name, seconds in self.PHASES:
            self.spans.append((t, t + seconds))
            t += seconds
        self.finish(t)
        self._timelines()

    # ------------------------------------------------------------------ declared per phase
    def code_for(self, k):
        return self.CODE[k] if self.CODE else None

    def station_for(self, k):
        return self.STATION[k] if self.STATION else None

    def evidence_for(self, k):
        return self.EVIDENCE[k] if self.EVIDENCE else ""

    def heading_for(self, k, u):
        """The heading over the picture at ``u`` of phase ``k``."""
        h = self.HEADINGS[k] if self.HEADINGS else ""
        if isinstance(h, str):
            return h
        text = h[0][1]
        for at, s in h:
            if u >= at:
                text = s
        return text

    def last_line(self):
        return "last time: " + self.LAST if self.LAST else "start here"

    def next_line(self):
        return "next: " + self.NEXT

    def steady(self, sc, k, u, op):
        """What stays on screen across phases (drawn at full strength, every frame)."""

    def kept(self, first=0, last=None):
        """For :meth:`steady`: 1 while phases ``first`` to ``last`` (0-based, inclusive) run,
        fading in and out with their content where they start and end inside the
        animation (the first phase's start and the last's end fade with everything)."""
        last = len(self.PHASES) - 1 if last is None else last
        t = getattr(self, "_t", 0.0)
        a, b = self.spans[first][0], self.spans[last][1]
        if last == len(self.PHASES) - 1:
            b = self.duration + 1.0
        if not a <= t <= b:
            return 0.0
        op = smoothstep(a, a + self.FADE, t) if first > 0 else 1.0
        return op * (1.0 - smoothstep(b - self.FADE, b, t))

    # ------------------------------------------------------------------ timelines
    def where(self, t):
        """``(phase, u)`` at time ``t``."""
        k = self.phase_at(t)
        a, b = self.spans[k]
        return k, float(np.clip((t - a) / (b - a), 0.0, 1.0))

    def heading_at(self, t):
        if t < self.LAST_SECONDS:
            return self.last_line()
        if self.NEXT and t >= self.t_done - self.NEXT_SECONDS:
            return self.next_line()
        return self.heading_for(*self.where(t))

    def _timelines(self):
        ts = np.arange(0.0, self.duration + self.TICK, self.TICK)
        self._tl_head = _Timeline(_merge((float(t), self.heading_at(float(t))) for t in ts))
        self._tl_tag = _Timeline(_merge((float(t), self.evidence_for(self.where(float(t))[0]))
                                     for t in ts))
        self._tl_station = _Timeline(_merge((float(t), self.station_for(self.where(float(t))[0]))
                                         for t in ts))
        runs, outs = [], []
        for t in ts:
            k, u = self.where(float(t))
            code = self.code_for(k)
            run = out = None
            if code is not None:
                run = (code.lines(), _pick(code.run, u))
                out = (code.lines(), _pick(code.out, u, ""))
            runs.append((float(t), run))
            outs.append((float(t), out))
        self._tl_run = _Timeline(_merge(runs))
        self._tl_out = _Timeline(_merge(outs))
        for k in range(len(self.PHASES)):
            code = self.code_for(k)
            if code is not None:
                lines = code.lines()
                if len(lines) > MAX_LINES:
                    raise ValueError("%s phase %d: %d lines of code, at most %d"
                                     % (type(self).__name__, k + 1, len(lines), MAX_LINES))
                outs = [s for _u, text in code.out for s in text.split("\n")]
                for s in list(lines) + ["→ " + s for s in outs]:
                    if len(s) > MAX_CHARS:
                        raise ValueError("%s: %r is wider than the panel" % (code.region, s))

    # ------------------------------------------------------------------ the frame
    def draw(self, sc, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        op = smoothstep(0.0, self.FADE, t) * back
        k, u = self.where(t)
        self._t = t
        if op > 0.01:
            self.steady(sc, k, u, op)
        super().draw(sc, t, reset)
        if op <= 0.01:
            return
        self.text(sc, HEAD_X, HEAD_Y, self._tl_head.at(t), op * self._tl_head.fade(t), HEAD, "lm",
                  strong=True)
        self.code_panel(sc, t, op)
        self.loop_map(sc, t, op)
        self.text(sc, 0.95, TAG_Y, self._tl_tag.at(t), op * self._tl_tag.fade(t), SIZE, "rm",
                  ink=QUIET)

    # ------------------------------------------------------------------ the code panel
    def code_panel(self, sc, t, op):
        """The panel: the region's lines, the running one boxed, the output slot."""
        x0, x1, top, bottom = PANEL
        w, h = self.span(x1 - x0, bottom - top)
        self.rect(sc, self.P((x0 + x1) / 2, (top + bottom) / 2), w, h, C.rgb("#f2f2ee"), op,
                  border=GRID, border_w=1.2, radius=8, lift=0.1)
        cur = self._tl_run.at(t)
        if cur is None:
            return
        lines, _line = cur
        name = next((c.names()[0] for c in map(self.code_for, range(len(self.PHASES)))
                     if c is not None and c.lines() == lines), None)
        when, before, _after = self._tl_run.last_change(t)
        on = op
        if when is not None and (before is None or before[0] != lines):
            on *= smoothstep(when, when + self.FADE, t)
        nxt = self._tl_run.ts[bisect_right(self._tl_run.ts, t):]
        for tn, vn in zip(nxt, self._tl_run.vs[bisect_right(self._tl_run.ts, t):]):
            if vn is None or vn[0] != lines:
                on *= 1.0 - smoothstep(tn - self.FADE, tn, t)
                break
        if name is not None:
            self.text(sc, x0 + 0.012, top + 0.027, code_file(name), on, SIZE, "lm", ink=QUIET)
        lx = x0 + self.frac(12.0)[0]
        # the box round the running line slides from line to line
        y = self._line_y(t, lines)
        lit = {}
        if y is not None:
            bw = x1 - x0 - self.frac(12.0)[0]
            self.rect(sc, self.P(x0 + (x1 - x0) / 2, y), self.span(bw)[0], 25.0,
                      C.rgb("#fbfbf8"), on, border=INK, border_w=1.8, radius=5, lift=0.25)
            for i in range(len(lines)):
                lit[i] = max(0.0, 1.0 - abs(y - (CODE_Y + CODE_STEP * i)) / CODE_STEP)
        for i, s in enumerate(lines):
            g = lit.get(i, 0.0)
            self.label(sc, self.P(lx, CODE_Y + CODE_STEP * i), s, CODE, anchor="lm",
                       opacity=on * (0.62 + 0.38 * g), ink=INK if g > 0.5 else None,
                       kind="mono")
        out = self._tl_out.at(t)
        if out is not None and out[1]:
            parts = out[1].split("\n")
            oy = min(OUT_Y - CODE_STEP * (len(parts) - 1),
                     CODE_Y + CODE_STEP * (len(lines) + 0.4))
            for i, s in enumerate(parts):
                self.label(sc, self.P(lx, oy + CODE_STEP * i), ("→ " if i == 0 else "  ") + s,
                           CODE, anchor="lm", opacity=on * self._tl_out.fade(t, 0.2),
                           ink=dark(DATA, 0.35), kind="mono")

    def _line_y(self, t, lines):
        when, before, after = self._tl_run.last_change(t)
        if after is None or after[1] is None:
            return None
        y1 = CODE_Y + CODE_STEP * after[1]
        if when is None or before is None or before[0] != lines or before[1] is None:
            return y1
        y0 = CODE_Y + CODE_STEP * before[1]
        return y0 + (y1 - y0) * ease_in_out(remap(t, when, when + 0.45))

    # ------------------------------------------------------------------ the loop map
    def _stations(self):
        """Each station's centre, as px from the map's centre."""
        L = MAP_HW - MAP_HH
        return dict(data=(-L + 15, -MAP_HH), forward=(0.0, -MAP_HH), loss=(L - 15, -MAP_HH),
                    backward=(L * 0.5, MAP_HH), step=(-L * 0.5, MAP_HH))

    @staticmethod
    @lru_cache(maxsize=4)
    def _track_runs(boxes):
        """The stadium-shaped track as px offsets, in pieces that stop short of the
        stations' boxes ``((x, y, half w, half h), ...)``."""
        L, R = MAP_HW - MAP_HH, MAP_HH
        out, run = [], []
        for v in np.linspace(0.0, 4 * L + 2 * np.pi * R, 401):
            if v < 2 * L:
                p = (-L + v, -R)
            elif v < 2 * L + np.pi * R:
                a = (v - 2 * L) / R - np.pi / 2
                p = (L + R * np.cos(a), R * np.sin(a))
            elif v < 4 * L + np.pi * R:
                p = (L - (v - 2 * L - np.pi * R), R)
            else:
                a = (v - 4 * L - np.pi * R) / R + np.pi / 2
                p = (-L + R * np.cos(a), R * np.sin(a))
            if any(abs(p[0] - bx) < hw and abs(p[1] - by) < hh for bx, by, hw, hh in boxes):
                if len(run) > 1:
                    out.append(tuple(run))
                run = []
            else:
                run.append((float(p[0]), float(p[1])))
        if len(run) > 1:
            out.append(tuple(run))
        return tuple(out)

    def _track(self, boxes):
        return self._track_runs(tuple(boxes[name] for name in STATIONS))

    def loop_map(self, sc, t, op, centre=MAP):
        """The training loop as a circular track with its five stations, the one under way
        lit, and "you are here" beside it."""
        now = self._tl_station.at(t)
        when, before, _after = self._tl_station.last_change(t)
        swap = 1.0 if when is None else smoothstep(when, when + 0.6, t)
        cx, cy = centre
        pos = self._stations()
        boxes = {}
        for name, (dx, dy) in pos.items():
            w = layout.text_width(name, SIZE) * self.cam.width + 16.0
            boxes[name] = (dx, dy, w / 2 + 3.0, 14.0)
        # the track, broken where a station sits on it
        c = self.P(cx, cy)
        for run in self._track(boxes):
            self.polyline(sc, [self.shift(c, x, y) for x, y in run], 2.0, LOOP, op, bias=0.1)
        L, R = MAP_HW - MAP_HH, MAP_HH
        # which way round: heads at the two ends
        self._head(sc, self.shift(c, L + R - 0.5, -3), self.shift(c, L + R, 2), LOOP, op, 2.0)
        self._head(sc, self.shift(c, -L - R + 0.5, 3), self.shift(c, -L - R, -2), LOOP, op, 2.0)
        for name, (dx, dy) in pos.items():
            g = (swap if name == now else 0.0) + ((1.0 - swap) if name == before else 0.0)
            ink = STATION_INK[name]
            w = boxes[name][2] * 2 - 6.0
            fill = tuple(float(v) for v in C.mix(PAPER, soft(ink, 0.45), g))
            self.rect(sc, self.shift(c, dx, dy), w, 22.0, fill, op,
                      border=INK if g > 0.5 else GRID, border_w=1.2 + 1.2 * g, radius=11,
                      lift=0.4)
            self.label(sc, self.shift(c, dx, dy), name, SIZE, opacity=op * (0.7 + 0.3 * g),
                       ink=INK if g > 0.5 else None)
        if now is not None:
            dx = float(np.clip(pos[now][0], -L + 52, L - 52))
            dy = 4.0 if pos[now][1] > 0 else -4.0        # nearer the lit station
            self.label(sc, self.shift(c, dx, dy), "you are here", SIZE,
                       opacity=op * self._tl_station.fade(t, 0.3), ink=INK)

    # ------------------------------------------------------------------ the alphabet
    def card(self, sc, fx, fy, w, h, op, lift=0.2, fill=None):
        """A data card: a blue rounded card with a blue band along its top."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, w, h, soft(DATA, 0.82) if fill is None else fill, op, border=DATA,
                  border_w=2.0, radius=7, lift=lift)
        self.rect(sc, self.shift(c, 0, -h / 2 + 4.0), w - 6.0, 4.0, DATA, op, radius=2,
                  lift=lift + 0.02)

    def cells(self, fx, fy, w, h, rows, cols):
        """``cell(row, col) -> (fx, fy)``: the centres of a ``rows x cols`` grid of tiles
        filling ``w x h`` px centred at ``(fx, fy)`` (what :meth:`tiles` returns, without
        drawing: for working out positions in ``setup``)."""
        dw, dh = self.frac(w, h)

        def cell(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / cols, fy - dh / 2 + (r + 0.5) * dh / rows
        return cell

    def digit_card(self, sc, img, fx, fy, tile, op, lift=0.2, zero=None, lo=DATA_LO,
                   hi=DATA_HI):
        """A digit (or any square of data) as a data card of ``tile`` px tiles centred at
        ``(fx, fy)``, shaded on the normalised pixels' scale (:data:`DATA_LO` paper to
        :data:`DATA_HI` full ink); returns :meth:`digit_cells`."""
        side = np.shape(img)[0] * tile
        self.card(sc, fx, fy, side + 14, side + 20, op, lift=lift)
        self.tiles(sc, img, fx, fy + self.frac(0, 3.0)[1], side, side, op, lo=lo, hi=hi,
                   lift=lift + 0.004, zero=zero, grid=0.18)
        return self.digit_cells(fx, fy, tile, np.shape(img)[0])

    def digit_cells(self, fx, fy, tile, side=28):
        """``cell(row, col)`` of the tiles of :meth:`digit_card` at ``(fx, fy)``."""
        return self.cells(fx, fy + self.frac(0, 3.0)[1], side * tile, side * tile, side, side)

    def outline(self, sc, cell, r0, c0, rows, cols, tile, op, ink=INK, lift=1.0):
        """An outline round tiles ``r0..r0+rows-1`` x ``c0..c0+cols-1`` of a grid whose
        ``cell`` and ``tile`` size (px) are given; returns its layout corners ``(left, top,
        right, bottom)``."""
        (x0, y0), (x1, y1) = cell(r0, c0), cell(r0 + rows - 1, c0 + cols - 1)
        dx, dy = self.frac(tile / 2 + 2, tile / 2 + 2)
        w, h = self.span(x1 - x0 + 2 * dx, y1 - y0 + 2 * dy)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, ink, op, border=ink,
                  border_w=2.4, radius=3, lift=lift, fill_op=0.0)
        return x0 - dx, y0 - dy, x1 + dx, y1 + dy

    def tiles(self, sc, values, fx, fy, w, h, op, ink=DATA, lo=None, hi=None, reveal=1.0,
              zero=None, ring=(), ring_ink=None, grid=0.3, lift=0.3):
        """A 2-D array as ``w x h`` px of square tiles centred at ``(fx, fy)``, each shaded
        by its value (paper at ``lo``, ``ink`` at ``hi``; deeper: bigger); ``reveal``
        shows that fraction of its rows; ``zero`` greys the tiles it marks; ``ring``
        outlines ``(row, col)`` tiles. Returns ``cell(row, col) -> (fx, fy)``."""
        a = np.atleast_2d(np.asarray(values, float))
        rows, cols = a.shape
        dh = self.frac(0, h)[1]
        cell = self.cells(fx, fy, w, h, rows, cols)
        if op <= 0.01:
            return cell
        rgb = shades(a, ink, lo, hi, zero)
        shown = int(np.ceil(rows * float(np.clip(reveal, 0, 1)) - 1e-9))
        if shown <= 0:
            return cell
        if shown < rows:
            rgb, h_shown = rgb[:shown], h * shown / rows
            draw_image(self, sc, self.P(fx, fy - dh / 2 + self.frac(0, h_shown / 2)[1]), rgb,
                       w, h_shown, op, grid=grid if min(w / cols, h / rows) >= 6 else 0,
                       lift=lift)
        else:
            draw_image(self, sc, self.P(float(fx), float(fy)), rgb, w, h, op,
                       grid=grid if min(w / cols, h / rows) >= 6 else 0, lift=lift)
        for r, k in ring:
            x, y = cell(r, k)
            ri = INK if ring_ink is None else ring_ink
            self.rect(sc, self.P(x, y), w / cols + 3, h / rows + 3, ri, op, border=ri,
                      border_w=2.2, radius=2, lift=lift + 0.4, fill_op=0.0)
        return cell

    def strip(self, sc, values, fx, fy, w, h, op, ink=DATA, lo=None, hi=None, lift=0.3):
        """A long vector (784 numbers) as one ``w x h`` px strip of its shades, neighbouring
        values averaged down to the pixels there are."""
        if op <= 0.01:
            return
        draw_strip(self, sc, self.P(float(fx), float(fy)),
                   shades(np.ravel(values), ink, lo, hi), w, h, op, lift=lift)

    def bias_strip(self, sc, values, fx, fy, w, op, h=8.0, lo=None, hi=None):
        """Biases: a thin orange strip, one segment per bias."""
        self.tiles(sc, np.atleast_2d(values), fx, fy, w, h, op, ink=BIAS, lo=lo, hi=hi,
                   grid=0.0)

    def grad_tiles(self, sc, values, fx, fy, w, h, op, offset=(7.0, 7.0), lo=None, hi=None):
        """Gradients: red tiles just behind (down and right of) the weight tiles at
        ``(fx, fy)``; draw the weights after them."""
        dx, dy = self.frac(*offset)
        return self.tiles(sc, values, fx + dx, fy + dy, w, h, op, ink=GRAD, lo=lo, hi=hi,
                          lift=0.15)

    def meter(self, sc, fx, fy, frac, op, h=120.0, w=26.0, label=None):
        """The loss: a purple meter ``h`` px tall, filled from the bottom to ``frac``."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, w, h, TRACK, op * 0.6, border=dark(LOSS, 0.2), border_w=1.5,
                  radius=6, lift=0.2)
        f = float(np.clip(frac, 0.0, 1.0)) * (h - 6)
        if f > 0.5:
            self.rect(sc, self.shift(c, 0, h / 2 - 3 - f / 2), w - 6, f, LOSS, op,
                      radius=min(4.0, f / 2), lift=0.3)
        if label:
            self.label(sc, self.shift(c, 0, h / 2 + 16), str(label), SIZE, opacity=op,
                       ink=dark(LOSS, 0.3))

    def hand(self, sc, fx, fy, op, size=1.0):
        """The optimizer: a green hand (a palm, three fingers and a thumb)."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        s = size
        self.rect(sc, c, 24 * s, 20 * s, OPTIM, op, border=dark(OPTIM, 0.3), radius=7 * s,
                  lift=0.6)
        for dx in (-7, 0, 7):
            self.rect(sc, self.shift(c, dx * s, -14 * s), 6 * s, 12 * s, OPTIM, op,
                      border=dark(OPTIM, 0.3), radius=3 * s, lift=0.55)
        self.rect(sc, self.shift(c, -14 * s, 2 * s), 9 * s, 6 * s, OPTIM, op,
                  border=dark(OPTIM, 0.3), radius=3 * s, lift=0.55)

    def dial(self, sc, fx, fy, frac, op, r=30.0, label=None):
        """The learning rate: a dial whose needle sweeps 270 degrees as ``frac`` goes 0..1."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.disk(sc, c, r, soft(RATE, 0.75), op, lift=0.3, outline=0.1)
        a0, a1 = np.radians(225), np.radians(-45)
        arc = [self.shift(c, (r + 5) * np.cos(a), -(r + 5) * np.sin(a))
               for a in np.linspace(a0, a1, 40)]
        self.polyline(sc, arc, 2.2, dark(RATE, 0.2), op, bias=0.4)
        a = a0 + (a1 - a0) * float(np.clip(frac, 0.0, 1.0))
        # the needle as a row of small dots lifted above the face (a line sat under it)
        for t in np.linspace(0.0, r - 6.0, int((r - 6.0) / 2.0) + 1):
            self.disk(sc, self.shift(c, t * np.cos(a), -t * np.sin(a)), 2.2, dark(RATE, 0.4),
                      op, lift=1.0, outline=0.0)
        self.disk(sc, c, 4.5, dark(RATE, 0.4), op, lift=1.1, outline=0.0)
        if label:
            self.label(sc, self.shift(c, 0, r + 22), str(label), SIZE, opacity=op,
                       ink=dark(RATE, 0.4))

    def stack(self, sc, fx, fy, w, h, n, op, step=(5.0, -5.0), lift=0.2):
        """``n`` data cards one behind the other, the front one at ``(fx, fy)``; returns
        nothing (draw the front card's content after it)."""
        for i in reversed(range(int(n))):
            self.card(sc, fx + self.frac(step[0] * i)[0], fy + self.frac(0, step[1] * i)[1],
                      w, h, op, lift=lift - 0.001 * i)

    def dims(self, sc, a, b, text, op, side=1.0, gap=10.0, ink=None):
        """A measured span from layout point ``a`` to ``b`` with end ticks, and ``text``
        beside its middle (``side``: +1 below or right, -1 above or left)."""
        if op <= 0.01:
            return
        ink = dark(DATA, 0.3) if ink is None else ink
        pa, pb = self.P(*a), self.P(*b)
        sa, sb = self.px(pa), self.px(pb)
        d = (sb - sa) / max(float(np.hypot(*(sb - sa))), 1e-6)
        n = np.array([-d[1], d[0]]) * 5.0
        self.polyline(sc, [pa, pb], 1.6, ink, op, bias=0.5)
        for p in (pa, pb):
            self.polyline(sc, [self.shift(p, *n), self.shift(p, *-n)], 1.6, ink, op, bias=0.5)
        mid = self.P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        horizontal = abs(d[0]) > abs(d[1])
        if horizontal:
            self.label(sc, self.shift(mid, 0, side * (gap + 6)), text, SIZE, opacity=op,
                       ink=ink, anchor="mm")
        else:
            self.label(sc, self.shift(mid, side * gap, 0), text, SIZE, opacity=op, ink=ink,
                       anchor="lm" if side > 0 else "rm")

    # ------------------------------------------------------------------ numbers that stay
    def big(self, sc, fx, fy, text, op, label=None, ink=None, size=BIG, anchor="lm"):
        """A number to look at: ``text`` large, ``label`` small above it."""
        if label:
            self.text(sc, fx, fy - size * 0.95 - 0.012, label, op, SIZE, anchor)
        self.text(sc, fx, fy, text, op, size, anchor, ink=INK if ink is None else ink)

    def running(self, sc, fx, fy, steps, u, op, label=None, ink=None, size=BIG, anchor="lm"):
        """A running total: the text of the last ``(u, text)`` of ``steps`` reached (the
        first before any), swapped in place, so it is never blank between steps."""
        self.big(sc, fx, fy, _pick(steps, u, steps[0][1]), op, label, ink, size, anchor)

    def heading(self, k, u):
        """The heading shown at ``u`` of phase ``k`` (what :meth:`draw` puts there)."""
        a, b = self.spans[k]
        return self.heading_at(a + (b - a) * u)


def _pick(steps, u, default=None):
    """The value of the last ``(u0, value)`` of ``steps`` with ``u0 <= u``."""
    out = default
    for at, v in steps:
        if u >= at:
            out = v
    return out
