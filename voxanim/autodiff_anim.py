"""Automatic differentiation: the chain rule, one edge at a time, on the
computational graph of a single neuron.

    x1 --\\
          × m1 --\\
    w1 --/        + s --\\
    x2 --\\              + z -- σ a -- − d -- ½(·)² L
          × m2 --/     /          /
    w2 --/        b --/      y --/

    L = ½(σ(w1 x1 + w2 x2 + b) − y)²

The numbers are the XOR network's (:mod:`voxanim.gd_anim`): its first hidden
neuron before training (seed 0) on that animation's worked example x = (1, 0),
y = 1, so the forward values here (z = 0.126, σ(z) = 0.531) are the ones the
gradient-descent animation shows for h₁.  The engine is
:mod:`voxanim.autodiff`.

1. **build** -- the expression is read inside out, left to right; every
   operation becomes a node (labelled with its operation) joined to the
   nodes it reads,
2. **forward** -- values flow right (teal) and appear under each node; each
   node also records the local derivative of its output with respect to
   each input,
3. **backward** -- from ∂L/∂L = 1 at the right, the gradient (magenta) flows
   left.  Along each edge the incoming gradient is multiplied by that edge's
   local derivative (shown on the edge) and the product appears under the
   parent, whose ring lights up once its gradient is recorded,
4. **result** -- the gradients of w₁, w₂ and b, then one step
   p ← p − η ∂L/∂p with η = 2 and the forward pass again: the loss falls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .autodiff import ETA, PARAMS, READING_ORDER, backward, neuron_example, neuron_loss
from .diagram import DATA, DATA_HI, EDGE, GRAD, NODE, Diagram
from .easing import ease_in_out, remap, smoothstep
from .text import MINUS, num

# ------------------------------------------------------------------ layout (fractions)

AT = dict(x1=(0.06, 0.30), w1=(0.06, 0.47), x2=(0.06, 0.64), w2=(0.06, 0.81),
          m1=(0.20, 0.385), m2=(0.20, 0.725), s=(0.33, 0.555), b=(0.26, 0.86),
          z=(0.46, 0.555), a=(0.59, 0.555), d=(0.72, 0.555), y=(0.52, 0.86),
          L=(0.88, 0.555))
EXPR_AT = (0.55, 0.235)                 # the expression, above the graph
RESULT_AT = (1.0, 0.72)                 # right edge and top of the result lines
RESULT_STEP = 0.052                     # line pitch of the result lines (layout fraction)
TAG_ALONG = 0.55                        # a rising edge's tag sits this far from child to parent
R = 20.0                                # node radius, px
R_LOSS = 24.0                           # the loss node is a little larger (its symbol is wide)
VALUE_DY, GRAD_DY = R + 13.0, R + 31.0  # the value and the gradient lines under a node, px
NAME_DY = -(R + 13.0)                   # the node's name above it, px
TAG_GAP = 4.0                           # clear space between a tag and its edge, px
TAG_REST = 0.6                          # how visible a tag stays after its beat
PARAM_FILL = C.rgb("#fbe9d6")           # parameters: pale orange
INPUT_FILL = C.rgb("#e9ecee")           # inputs and the target: pale grey
TEAL_INK = tuple(float(v) for v in C.darken(DATA, 0.25))
GRAD_INK = tuple(float(v) for v in C.darken(GRAD, 0.2))
OPS = {"m1": "×", "m2": "×", "s": "+", "z": "+", "a": "σ", "d": "−", "L": "½d²"}
NAMES = dict(x1="x₁", w1="w₁", x2="x₂", w2="w₂", m1="m₁", m2="m₂", s="s", b="b", z="z",
             a="a", d="d", y="y", L="L")
OP_NODES = tuple(k for k in READING_ORDER if k in OPS)

# Beats (seconds): one per node built, one per node computed, one per chain-rule step.
T_BUILD0, BUILD_DT = 1.0, 0.6
T_FWD0, FWD_DT = 9.4, 1.25
T_B0 = 18.8                             # ∂L/∂L = 1 appears
BWD_DT, BWD_INPUT_DT, BWD_GAP = 1.4, 1.0, 0.3   # a step; a step into an unused input; the pause
T_RESULT_GAP = 0.6                      # after the last step
RESULT_DT, UPDATE_DT, AFTER_DT = 1.6, 2.6, 1.5


def _is_input_step(step):
    """True for the two steps into x₁ / x₂ (gradients nothing uses: quicker)."""
    return step.parent.name in ("x1", "x2")


@dataclass
class Frame:
    """Everything one frame shows, worked out from ``t`` once."""

    back: float                 # 1, fading to 0 in the loop reset
    appear: dict                # node -> 0..1, built
    value: dict                 # node -> 0..1, its value is known
    grad: dict                  # node -> 0..1, its gradient is recorded
    fwd: dict                   # op node -> progress of its forward beat
    bwd: list                   # per chain-rule step: progress of its beat
    rec: float                  # the recorded gradients' visibility (they drain in the update)
    upd: float                  # progress of the update
    values: dict                # node -> the value on show (moves during the update)


class AutodiffGraph(Diagram):
    """The whole lesson: build, forward, backward, result (see the module docstring)."""

    PHASES = ("build", "forward", "backward", "result")

    def __init__(self):
        super().__init__("Automatic differentiation", "")
        self.params, self.inputs = neuron_example()
        self.g = neuron_loss(**self.params, **self.inputs)
        self.steps = backward(self.g.L)
        self.grads = {k: self.g[k].grad for k in PARAMS}
        self.params2 = {k: self.params[k] - ETA * self.grads[k] for k in PARAMS}
        self.g2 = neuron_loss(**self.params2, **self.inputs)
        self.pos = {k: self.P(*AT[k]) for k in AT}
        self.hold, self.reset = 3.0, 1.2
        self._timeline()
        self.finish(self.Tm["after"][1])

    def _timeline(self):
        Tm = dict(intro=(0.2, 1.0))
        for n, k in enumerate(READING_ORDER):
            Tm["build:" + k] = (T_BUILD0 + n * BUILD_DT, T_BUILD0 + (n + 1) * BUILD_DT)
        for n, k in enumerate(OP_NODES):
            Tm["fwd:" + k] = (T_FWD0 + n * FWD_DT, T_FWD0 + (n + 1) * FWD_DT)
        Tm["b0"] = (T_B0, T_B0 + 0.8)
        t = Tm["b0"][1] + 0.2
        for n, st in enumerate(self.steps):
            d = BWD_INPUT_DT if _is_input_step(st) else BWD_DT
            Tm["bwd:%d" % n] = (t, t + d)
            t += d + BWD_GAP
        t += T_RESULT_GAP
        Tm["result"] = (t, t + RESULT_DT)
        Tm["update"] = (t + RESULT_DT + 0.4, t + RESULT_DT + 0.4 + UPDATE_DT)
        Tm["after"] = (Tm["update"][1], Tm["update"][1] + AFTER_DT)
        self.Tm = Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "result"), (2, "b0"), (1, "fwd:" + OP_NODES[0])):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ geometry
    def radius(self, k):
        """Node ``k``'s radius in px."""
        return R_LOSS if k == "L" else R

    def edge_ends(self, parent, child):
        """The edge from ``parent`` to ``child``, stopped at both rims."""
        return self.ends(self.pos[parent], self.pos[child], self.radius(parent),
                         self.radius(child))

    def edges(self):
        """``[(parent, child)]`` for every edge of the graph."""
        return [(p.name, v.name) for v in self.g.nodes.values() for p in v.parents]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        appear = {k: smoothstep(0.0, 0.5, self.progress("build:" + k, t)) * back
                  for k in READING_ORDER}
        fwd = {k: self.progress("fwd:" + k, t) for k in OP_NODES}
        value = {k: appear[k] if self.g[k].leaf else smoothstep(0.6, 0.8, fwd[k]) * back
                 for k in READING_ORDER}
        bwd = [self.progress("bwd:%d" % n, t) for n in range(len(self.steps))]
        grad = dict.fromkeys(READING_ORDER, 0.0)
        grad["L"] = smoothstep(0.2, 0.7, self.progress("b0", t))
        for st, u in zip(self.steps, bwd):
            grad[st.parent.name] = smoothstep(0.6, 0.8, u)
        upd = ease_in_out(self.progress("update", t))
        rec = (1.0 - smoothstep(0.3, 0.8, upd)) * back
        p = {k: self.params[k] + (self.params2[k] - self.params[k]) * upd for k in PARAMS}
        values = {k: v.data for k, v in neuron_loss(**p, **self.inputs).nodes.items()}
        return Frame(back, appear, value, grad, fwd, bwd, rec, upd, values)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        self._expression(sc, t, f.back)
        self._edges(sc, f)
        self._forward_dots(sc, f)
        self._backward_dots(sc, f)
        for k in READING_ORDER:
            self._node(sc, k, f)
        self._result(sc, t, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _expression(self, sc, t, back):
        op = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        self.label(sc, self.P(*EXPR_AT), "L = ½(σ(w₁x₁ + w₂x₂ + b) − y)²", layout.CAPTION,
                   opacity=op, ink=layout.STRONG_INK)

    def _edges(self, sc, f):
        """Grey edges, appearing with the node they feed."""
        for parent, child in self.edges():
            a, b = self.edge_ends(parent, child)
            sc.line(a, b, width=1.6, color=EDGE, opacity=0.6 * f.appear[child], bias=0.0)

    def _forward_dots(self, sc, f):
        """Teal dots: each parent's value flying into the node being computed."""
        for k in OP_NODES:
            u = f.fwd[k]
            if not 0 < u < 1:
                continue
            for i, p in enumerate(self.g[k].parents):
                a, b = self.edge_ends(p.name, k)
                self.dot(sc, a, b, remap(u, 0.08 * i, 0.6 + 0.08 * i), DATA, 4.2, f.back,
                         key=("f", k, i))

    def _backward_dots(self, sc, f):
        """Magenta dots carrying the gradient back along each edge, and the edge's
        local-derivative tag, which stays once the step is done."""
        for n, (st, u) in enumerate(zip(self.steps, f.bwd)):
            if u <= 0:
                continue
            a, b = self.edge_ends(st.parent.name, st.node.name)
            self.dot(sc, b, a, remap(u, 0.0, 0.6), GRAD, 4.2, f.back, key=("b", n))
            self._tag(sc, a, b, st, u, f)

    def _tag(self, sc, a, b, st, u, f):
        """``× local`` beside the edge, on the side away from the labels under
        the child: above a level or falling edge, below a rising one (whose
        parent is lower), and nearer the parent in that case."""
        sa, sb = self.px(a), self.px(b)
        d = sb - sa
        rising = d[1] < 0                           # the child is higher on screen
        nrm = np.array([d[1], -d[0]]) / max(np.linalg.norm(d), 1e-6)
        if (nrm[1] < 0) == rising:
            nrm = -nrm
        mid = b + (a - b) * (TAG_ALONG if rising else 0.5)
        op = smoothstep(0.1, 0.3, u) * (TAG_REST + (1 - TAG_REST) * (1 - smoothstep(0.8, 1.0, u)))
        text = "× " + self._local_text(st)
        # far enough along the normal that the text box clears a sloping edge
        w = layout.text_width(text, layout.MIN_TEXT) * self.cam.width
        h = layout.MIN_TEXT * self.cam.height
        off = TAG_GAP + abs(nrm[0]) * w / 2 + abs(nrm[1]) * h / 2
        self.label(sc, mid, text, layout.MIN_TEXT, dx=nrm[0] * off, dy=nrm[1] * off,
                   ink=GRAD_INK, opacity=op * f.rec)

    @staticmethod
    def _local_text(st):
        """The local derivative as a short number (``1``, ``−1``, else 3 decimals)."""
        if float(st.local).is_integer():
            return ("%d" % st.local).replace("-", MINUS)
        return num(st.local, 3)

    def _node(self, sc, k, f):
        """A node: its operation (or name, for a leaf) inside, its name above, its
        value (teal) and gradient (magenta) below; a magenta ring once recorded."""
        v, c, op = self.g[k], self.pos[k], f.appear[k]
        if op <= 0.01:
            return
        if v.leaf:
            fill = PARAM_FILL if k in PARAMS else INPUT_FILL
        else:
            fill = C.mix(NODE, DATA_HI, 0.45 * f.value[k])
        ring = f.grad[k] * f.rec
        self.disk(sc, c, self.radius(k), fill, op, ring=GRAD, ring_w=3.0, ring_op=ring)
        self.label(sc, c, NAMES[k] if v.leaf else OPS[k], layout.CAPTION, ink=layout.STRONG_INK,
                   opacity=op)
        if not v.leaf:
            self.label(sc, c, NAMES[k], layout.MIN_TEXT, dy=NAME_DY, opacity=op)
        self.label(sc, c, num(f.values[k], 3), layout.MIN_TEXT, dy=VALUE_DY, ink=TEAL_INK,
                   opacity=f.value[k], kind="mono")
        g = v.grad
        if abs(g) < 1e-12:
            g = 0.0                                 # −0.000 would mislead
        self.label(sc, c, num(g, 3), layout.MIN_TEXT, dy=GRAD_DY, ink=GRAD_INK,
                   opacity=f.grad[k] * f.rec, kind="mono")

    def _result(self, sc, t, f):
        """The parameter gradients, then the update (η is in the subtitle),
        right-aligned under the loss node."""
        op = smoothstep(0.0, 0.6, self.progress("result", t)) * f.back
        if op <= 0.01:
            return
        up = smoothstep(0.0, 0.3, f.upd)
        x0, y0 = RESULT_AT
        lines = [("∂L/∂%s = %s" % (NAMES[k], num(self.grads[k], 3)), GRAD_INK, op)
                 for k in PARAMS]
        for k in ("w1", "b"):
            lines.append(("%s %s → %s" % (NAMES[k], num(self.params[k], 3),
                                          num(self.params2[k], 3)), layout.CAPTION_INK, up * op))
        lines.append(("L %s → %s" % (num(self.g.L.data, 3), num(self.g2.L.data, 3)),
                      layout.STRONG_INK, smoothstep(0.9, 1.0, f.upd) * op))
        for n, (text, ink, o) in enumerate(lines):
            self.label(sc, self.P(x0, y0 + n * RESULT_STEP), text, layout.MIN_TEXT, anchor="rm",
                       ink=ink, opacity=o)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        T = self.Tm
        p = self.params
        if reset > 0 or t < T["build:" + READING_ORDER[0]][0]:
            return ("one neuron, L = ½(σ(w₁x₁ + w₂x₂ + b) − y)²: the XOR network's first hidden "
                    "neuron before training (w₁ = %s, w₂ = %s, b = 0), x = (1, 0), y = 1"
                    % (num(p["w1"], 3), num(p["w2"], 3)))
        if t < T["fwd:" + OP_NODES[0]][0]:
            if t < T["build:s"][0]:
                return ("build: reading the expression inside out, each operation becomes a node "
                        "joined to the nodes it reads: w₁ × x₁, then w₂ × x₂")
            return ("build: + s, + b gives z, σ, − y, ½(·)²: 7 operations and 6 leaves "
                    "(3 parameters, 2 inputs, the target)")
        if t < T["b0"][0]:
            return self._forward_words(f)
        if t < T["bwd:0"][0]:
            return ("backward: start at the end, ∂L/∂L = 1. The gradient (magenta) flows left: "
                    "at every edge, incoming gradient × that edge's local derivative")
        if t < T["result"][0]:
            return self._step_words(t)
        if t < T["update"][0]:
            return ("result: ∂L/∂w₁ = %s, ∂L/∂w₂ = 0, ∂L/∂b = %s: twelve multiplications, and "
                    "no formula for ∂L/∂w was ever written down"
                    % (num(self.grads["w1"], 3), num(self.grads["b"], 3)))
        return ("one step p ← p − η ∂L/∂p with η = %g: w₁ %s → %s, b %s → %s; forward again, "
                "the loss falls from %s to %s"
                % (ETA, num(p["w1"], 3), num(self.params2["w1"], 3), num(p["b"], 3),
                   num(self.params2["b"], 3), num(self.g.L.data, 3), num(self.g2.L.data, 3)))

    def _forward_words(self, f):
        """The node being computed, its sum or product in numbers, and the local
        derivatives it records."""
        k = OP_NODES[0]
        for name in OP_NODES:
            if f.fwd[name] > 0:
                k = name
        v = self.g[k]
        vals = [num(q.data, 3) for q in v.parents]
        expr = {"m1": "w₁ × x₁ = %s × %s", "m2": "w₂ × x₂ = %s × %s", "s": "m₁ + m₂ = %s + %s",
                "z": "s + b = %s + %s", "a": "σ(z) = σ(%s)", "d": "a − y = %s − %s",
                "L": "½d² = ½(%s)²"}[k] % tuple(vals)
        locs = ", ".join("∂%s/∂%s = %s" % (NAMES[k], NAMES[q.name], self._local_text(st))
                         for q, st in zip(v.parents, self._steps_of(k)))
        return ("forward: %s = %s = %s\nand each node records its local derivatives: %s"
                % (NAMES[k], expr, num(v.data, 3), locs))

    def _steps_of(self, k):
        """The chain-rule steps out of node ``k``, in parent order."""
        return [next(st for st in self.steps if st.node.name == k and st.parent is q)
                for q in self.g[k].parents]

    def _step_words(self, t):
        """The chain-rule step under way (or just done), in numbers."""
        n = max(i for i in range(len(self.steps)) if t >= self.Tm["bwd:%d" % i][0])
        st = self.steps[n]
        child, par = NAMES[st.node.name], NAMES[st.parent.name]
        base = "∂L/∂%s = ∂L/∂%s × ∂%s/∂%s = %s × %s = %s" % (
            par, child, child, par, num(st.incoming, 3), self._local_text(st),
            num(st.product, 3))
        note = {
            "a": "the local derivative of σ is σ(z)(1 − σ(z)) = %s × %s"
                 % (num(self.g["a"].data, 3), num(1 - self.g["a"].data, 3)),
            "y": "the target gets a gradient too; nothing uses it",
            "b": "the first parameter gradient is recorded (its ring lights up)",
            "m1": "a + passes the gradient on unchanged (local derivative 1)",
            "w2": "x₂ = 0, so w₂ learns nothing from this example",
            "x1": "every gradient came from one multiplication per edge",
        }.get(st.parent.name)
        return base + ("\n" + note if note else "")
