"""A tiny scalar reverse-mode automatic differentiation engine.

Every arithmetic step builds a :class:`Value` that remembers how it was made:
its operation, its parents and the *local derivative* of its output with
respect to each parent, evaluated at the forward values.  Reverse mode then
walks the graph from the result back to the leaves (:func:`backward`): each
node's gradient dL/d(node), once complete, is multiplied by every local
derivative and added to the parent's gradient.  That is the chain rule, one
edge at a time, and :func:`backward` returns the list of those steps so an
animation (:mod:`voxanim.autodiff_anim`) can replay them.

    L = 1/2 (sigmoid(w1 x1 + w2 x2 + b) - y)^2            (:func:`neuron_loss`)

is the graph the animation shows: one sigmoid neuron with two inputs and a
squared-error loss.  :func:`neuron_example` gives it the numbers of the XOR
network in :mod:`voxanim.gd_anim` (first hidden neuron, seed 0, before
training) and that animation's worked example x = (1, 0), y = 1, so the two
agree.  :func:`numeric_gradient` is the finite-difference check the tests use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

ETA = 2.0                       # learning rate of the update the animation shows (gd_anim's)


@dataclass(eq=False)
class Value:
    """A number in the graph and how it was made.

    ``local[i]`` is d(self)/d(parents[i]) at the forward values; a leaf has
    no parents.  ``grad`` is dL/d(self), filled in by :func:`backward`.
    """

    data: float
    op: str = ""                        # "×", "+", "−", "σ", "½(·)²"; "" for a leaf
    parents: tuple = ()
    local: tuple = ()
    name: str = ""
    grad: float = 0.0

    def __add__(self, other):
        return Value(self.data + other.data, "+", (self, other), (1.0, 1.0))

    def __sub__(self, other):
        return Value(self.data - other.data, "−", (self, other), (1.0, -1.0))

    def __mul__(self, other):
        return Value(self.data * other.data, "×", (self, other), (other.data, self.data))

    def sigmoid(self):
        """``σ(self)``; its local derivative is ``σ(1 − σ)``."""
        s = 1.0 / (1.0 + math.exp(-self.data))
        return Value(s, "σ", (self,), (s * (1.0 - s),))

    def half_square(self):
        """``½ self²``; its local derivative is ``self``."""
        return Value(0.5 * self.data * self.data, "½(·)²", (self,), (self.data,))

    @property
    def leaf(self):
        """True for an input, parameter or target: nothing made it."""
        return not self.parents


def topological(root):
    """Every node under ``root`` (inclusive), parents before children."""
    order, seen = [], set()

    def visit(v):
        if id(v) in seen:
            return
        seen.add(id(v))
        for p in v.parents:
            visit(p)
        order.append(v)
    visit(root)
    return order


@dataclass
class Step:
    """One application of the chain rule along one edge: ``parent.grad +=
    incoming × local`` where ``incoming`` is dL/d(node)."""

    node: Value
    parent: Value
    incoming: float
    local: float

    @property
    def product(self):
        """What this edge adds to the parent's gradient."""
        return self.incoming * self.local


def backward(root):
    """Reverse mode from ``root`` (dL/dL = 1): fills every ``grad`` and returns
    the :class:`Step` s in the order they were taken.

    Nodes are visited children first (reverse topological order), so a node's
    gradient is complete before it is passed on.
    """
    order = topological(root)
    for v in order:
        v.grad = 0.0
    root.grad = 1.0
    steps = []
    for v in reversed(order):
        for p, local in zip(v.parents, v.local, strict=True):
            p.grad += v.grad * local
            steps.append(Step(v, p, v.grad, local))
    return steps


@dataclass
class Graph:
    """The neuron's graph: the loss ``L`` and every node by name."""

    L: Value
    nodes: dict = field(default_factory=dict)

    def __getitem__(self, name):
        return self.nodes[name]


# The nodes in the order the expression is read, left to right, inside out:
# w1 x1, then w2 x2, their sum, plus b, sigma, minus y, half the square.
READING_ORDER = ("x1", "w1", "m1", "x2", "w2", "m2", "s", "b", "z", "a", "y", "d", "L")


def neuron_loss(w1, w2, b, x1, x2, y):
    """The graph of ``L = ½(σ(w1 x1 + w2 x2 + b) − y)²`` at these numbers."""
    leaves = {k: Value(float(v), name=k) for k, v in
              (("x1", x1), ("w1", w1), ("x2", x2), ("w2", w2), ("b", b), ("y", y))}
    n = dict(leaves)
    n["m1"] = n["w1"] * n["x1"]
    n["m2"] = n["w2"] * n["x2"]
    n["s"] = n["m1"] + n["m2"]
    n["z"] = n["s"] + n["b"]
    n["a"] = n["z"].sigmoid()
    n["d"] = n["a"] - n["y"]
    n["L"] = n["d"].half_square()
    for k, v in n.items():
        v.name = k
    return Graph(n["L"], n)


PARAMS = ("w1", "w2", "b")


def neuron_example():
    """``(params, inputs)`` of the worked example: ``{"w1", "w2", "b"}`` are the
    XOR network's first hidden neuron before training (:mod:`voxanim.gd_anim`,
    seed 0) and ``{"x1", "x2", "y"}`` its worked example x = (1, 0), y = 1."""
    from .gd_anim import EX, XOR_X, XOR_Y, train, unpack
    W, bias, _v, _c = unpack(train()[0][0])
    params = dict(w1=float(W[0, 0]), w2=float(W[1, 0]), b=float(bias[0]))
    inputs = dict(x1=float(XOR_X[EX, 0]), x2=float(XOR_X[EX, 1]), y=float(XOR_Y[EX]))
    return params, inputs


def gradients(params, inputs):
    """``{"w1", "w2", "b": dL/d·}`` by reverse mode."""
    g = neuron_loss(**params, **inputs)
    backward(g.L)
    return {k: g[k].grad for k in PARAMS}


def sgd_step(params, inputs, eta=ETA):
    """The parameters after one step ``p − eta · dL/dp``."""
    grads = gradients(params, inputs)
    return {k: params[k] - eta * grads[k] for k in PARAMS}


def numeric_gradient(f, args, key, e=1e-6):
    """Central difference of ``f(**args)`` with respect to ``args[key]``."""
    plus, minus = dict(args), dict(args)
    plus[key] += e
    minus[key] -= e
    return (f(**plus) - f(**minus)) / (2 * e)
