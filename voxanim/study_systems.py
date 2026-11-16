"""Redesigned experiments for the "training at scale" animations (86, 87, 89, 90).

The first versions (:mod:`voxanim.next_systems`) could not show their concept:

* ``data-parallel-allreduce`` and ``tensor-and-pipeline-parallel`` fed a layer of random
  weights the first 12 pooled pixels of each digit, the blank top-left corner, so most
  of what they computed was zero and the four workers' gradients looked alike.
* ``activation-checkpointing`` ran on seeded random inputs, and its scheme recomputed
  every segment even at spacing 1, so compute did not rise with the spacing.
* ``roofline`` drew its roof through the two fastest of the four operations it then
  placed under it.

Here every model is a small tanh MLP on 7 x 7 average-pooled MNIST, trained in plain
NumPy, and what the drawings need (the ring's buffers after every pass, what a
checkpointed backward pass holds at every step) is recorded by the experiment, so
:mod:`voxanim.mech_systems` computes nothing for itself. ``zero-sharding`` is exact
arithmetic and keeps its first study.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
SEED = 17


def _load_note(before, after, cores):
    """What else the machine was doing while the timings were taken, said as it was."""
    busy = max(before, after)
    state = ("above the core count, so other work was competing for every core" if busy > cores
             else "other work was using about %d of the %d cores" % (round(busy), cores)
             if busy >= 1.5 else "an otherwise idle machine")
    return "1-minute load average %.1f before and %.1f after timing, on %d cores: %s" % (
        before, after, cores, state)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (so an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, seed=SEED, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-systems-" + name, settings,
                                                    build).items()}


def _provenance(meta):
    """A split's provenance without its list of thousands of example indices."""
    return {k: v for k, v in meta.items() if k != "example_indices"}


# ---------------------------------------------------------------------- a tanh MLP
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def mlp_init(sizes, rng):
    """``[[W (in, out), b], ...]``: weights ``N(0, 1 / fan_in)``, zero biases."""
    return [[rng.normal(0, np.sqrt(1.0 / a), (a, b)), np.zeros(b)]
            for a, b in itertools.pairwise(sizes)]


def layer(params, i, a):
    """Layer ``i`` applied to its input ``a``: tanh, except the last (the logits)."""
    z = a @ params[i][0] + params[i][1]
    return z if i == len(params) - 1 else np.tanh(z)


def mlp_forward(params, x):
    """Every activation: the input, each hidden layer's output, the logits."""
    acts = [x]
    for i in range(len(params)):
        acts.append(layer(params, i, acts[-1]))
    return acts


def xent(logits, y):
    """Mean cross-entropy of integer labels ``y``."""
    return float(np.mean(np.logaddexp.reduce(logits, axis=1) - logits[np.arange(len(y)), y]))


def logit_gradient(logits, y):
    """d(mean cross-entropy) / d(logits)."""
    d = softmax(logits)
    d[np.arange(len(y)), y] -= 1.0
    return d / len(y)


def mlp_grad(params, x, y):
    """Plain backpropagation, every activation stored: ``[[dW, db], ...]``."""
    acts = mlp_forward(params, x)
    delta = logit_gradient(acts[-1], y)
    grads = [None] * len(params)
    for i in range(len(params) - 1, -1, -1):
        grads[i] = [acts[i].T @ delta, delta.sum(0)]
        if i:
            delta = (delta @ params[i][0].T) * (1.0 - acts[i] ** 2)
    return grads


def evaluate(params, x, y):
    """``(mean cross-entropy, accuracy)`` on ``(x, y)``."""
    logits = mlp_forward(params, x)[-1]
    return xent(logits, y), float(np.mean(logits.argmax(1) == y))


def flat(tensors):
    return np.concatenate([np.ravel(t) for pair in tensors for t in pair])


def gap(a, b):
    """Largest absolute difference between two sets of tensors."""
    return float(np.max(np.abs(flat(a) - flat(b))))


def train_adam(params, grad, x, y, steps, batch, lr, rng):
    """``steps`` Adam updates on shuffled minibatches; returns the trained ``params``."""
    from .optim import Adam

    P = {"%d%d" % (i, j): t for i, pair in enumerate(params) for j, t in enumerate(pair)}
    opt = Adam(P, lr)
    order = rng.permutation(len(x))
    for step in range(steps):
        idx = np.take(order, np.arange(step * batch, (step + 1) * batch), mode="wrap")
        now = [[P["%d0" % i], P["%d1" % i]] for i in range(len(params))]
        g = grad(now, x[idx], y[idx])
        opt.step(P, {"%d%d" % (i, j): t for i, pair in enumerate(g) for j, t in enumerate(pair)})
    return [[P["%d0" % i], P["%d1" % i]] for i in range(len(params))]


def _digits(n, split="train"):
    """``n`` pooled 7 x 7 digits of ``split``, labels, provenance, full 28 x 28 images."""
    from . import mnist
    from .realdata import mnist_split

    x, y, meta = mnist_split(n, split, side=7, seed=SEED)
    full = mnist.load()[2 if split == "test" else 0][np.asarray(meta["example_indices"])]
    return x, y, meta, full


# ====================================================================== 86
def ring_allreduce(vectors):
    """A ring all-reduce of ``vectors`` (one row per worker), simulated pass by pass.

    Each row is cut into as many chunks as there are workers. Reduce-scatter: in pass
    ``t`` worker ``w`` sends chunk ``w - t`` to worker ``w + 1``, who adds it to its
    own, so after ``P - 1`` passes worker ``w`` holds chunk ``w + 1`` summed over
    everyone. It divides that by ``P``. All-gather: in pass ``t`` worker ``w`` sends
    chunk ``w + 1 - t`` on, and the receiver keeps a copy.

    Returns the buffers after every pass - ``reduce`` and ``gather``: ``(passes + 1,
    worker, length)``; ``counts``: how many workers' rows are summed in each of a
    worker's chunks; ``have``: which chunks of the average a worker holds - with
    ``edges`` (the chunk boundaries), the ``(worker, chunk, to)`` of every send, and
    ``sent``: the bytes each worker put on the ring."""
    v = np.asarray(vectors, float)
    workers, n = v.shape
    edges = np.concatenate([[0], np.cumsum([len(c) for c in np.array_split(np.arange(n),
                                                                           workers)])])
    part = [slice(int(edges[c]), int(edges[c + 1])) for c in range(workers)]
    buf, count, sent = v.copy(), np.ones((workers, workers), int), np.zeros(workers, int)
    reduce, counts, reduce_sends = [buf.copy()], [count.copy()], []
    for turn in range(workers - 1):
        before, n0 = buf.copy(), count.copy()
        sends = []
        for w in range(workers):
            c, to = (w - turn) % workers, (w + 1) % workers
            buf[to, part[c]] += before[w, part[c]]
            count[to, c] += n0[w, c]
            sent[w] += before[w, part[c]].nbytes
            sends.append((w, c, to))
        reduce.append(buf.copy())
        counts.append(count.copy())
        reduce_sends.append(sends)
    owned = np.zeros_like(buf)
    have = np.zeros((workers, workers), bool)
    for w in range(workers):
        c = (w + 1) % workers
        owned[w, part[c]] = buf[w, part[c]] / workers
        have[w, c] = True
    gather, haves, gather_sends = [owned.copy()], [have.copy()], []
    for turn in range(workers - 1):
        before = owned.copy()
        sends = []
        for w in range(workers):
            c, to = (w + 1 - turn) % workers, (w + 1) % workers
            owned[to, part[c]] = before[w, part[c]]
            have[to, c] = True
            sent[w] += before[w, part[c]].nbytes
            sends.append((w, c, to))
        gather.append(owned.copy())
        haves.append(have.copy())
        gather_sends.append(sends)
    return dict(reduce=np.stack(reduce), counts=np.stack(counts), gather=np.stack(gather),
                have=np.stack(haves), edges=edges, reduce_sends=np.array(reduce_sends),
                gather_sends=np.array(gather_sends), sent=sent)


def ring_average(worker_grads):
    """The workers' gradients averaged tensor by tensor on the ring: ``(average as
    worker 0 ends up holding it, largest disagreement between workers, bytes sent by
    each worker)``."""
    out, apart, sent = [], 0.0, 0
    for i in range(len(worker_grads[0])):
        pair = []
        for j in range(2):
            ring = ring_allreduce(np.stack([np.ravel(g[i][j]) for g in worker_grads]))
            final = ring["gather"][-1]
            apart = max(apart, float(np.max(np.abs(final - final[0]))))
            sent = sent + ring["sent"]
            pair.append(final[0].reshape(worker_grads[0][i][j].shape))
        out.append(pair)
    return out, apart, sent


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


ALLREDUCE = dict(sizes=(49, 12, 8, 10), workers=4, per_worker=16, steps=200, shown=100,
                 lr=0.5, heldout=2000, layer=1)


def allreduce():
    """Data-parallel SGD on MNIST with four workers and a ring all-reduce, beside the
    same training on one machine with the whole batch."""
    c = ALLREDUCE
    workers, per, steps = c["workers"], c["per_worker"], c["steps"]
    batch = workers * per
    provenance = {}

    def build():
        x, y, meta, full = _digits(batch * steps)
        xt, yt, tmeta, _ = _digits(c["heldout"], "test")
        provenance.update(train=_provenance(meta), heldout=_provenance(tmeta))
        ring = mlp_init(c["sizes"], np.random.default_rng(SEED))
        single = [[w.copy(), b.copy()] for w, b in ring]
        out = dict(loss=[], accuracy=[], weights_apart=[], average_error=[], alike=[])

        def record():
            a, b = evaluate(ring, xt, yt), evaluate(single, xt, yt)
            out["loss"].append([a[0], b[0]])
            out["accuracy"].append([a[1], b[1]])
            out["weights_apart"].append(gap(ring, single))

        record()
        for step in range(steps):
            rows = slice(step * batch, (step + 1) * batch)
            bx, by = x[rows], y[rows]
            local = [mlp_grad(ring, a, b) for a, b in zip(np.split(bx, workers),
                                                          np.split(by, workers))]
            average, apart, sent = ring_average(local)
            whole = mlp_grad(ring, bx, by)                    # the same weights, all 64 at once
            out["average_error"].append(gap(average, whole))
            vs = [flat(g) for g in local]
            out["alike"].append(np.mean([cosine(vs[i], vs[j]) for i in range(workers)
                                         for j in range(i)]))
            if step == c["shown"]:
                k = c["layer"]
                mine = np.stack([g[k][0] for g in local])
                drawn = ring_allreduce(mine.reshape(workers, -1))
                stepped = [[w - c["lr"] * g[0], b - c["lr"] * g[1]]
                           for (w, b), g in zip(ring, average)]
                on_one = [[w - c["lr"] * g[0], b - c["lr"] * g[1]]
                          for (w, b), g in zip(ring, whole)]
                out.update(
                    {"ring_" + key: value for key, value in drawn.items()},
                    local=mine, whole=whole[k][0], digits=full[rows].astype(np.float32),
                    labels=by, workers_apart=apart, model_sent=sent,
                    model_bytes=flat(whole).nbytes,
                    cosines=np.array([[cosine(a.ravel(), b.ravel()) for b in mine]
                                      for a in mine]),
                    cosine_to_average=np.array([cosine(a.ravel(), mine.mean(0).ravel())
                                                for a in mine]),
                    step_apart=gap(stepped, on_one),
                    step_loss=np.array([evaluate(ring, xt, yt)[0], evaluate(stepped, xt, yt)[0],
                                        evaluate(on_one, xt, yt)[0]]))
            ring = [[w - c["lr"] * g[0], b - c["lr"] * g[1]] for (w, b), g in zip(ring, average)]
            alone = mlp_grad(single, bx, by)
            single = [[w - c["lr"] * g[0], b - c["lr"] * g[1]] for (w, b), g in zip(single, alone)]
            record()
        out["provenance"] = np.array(json.dumps(provenance))
        return out

    d = _cache("allreduce", build, (allreduce, ring_allreduce, ring_average, mlp_grad, mlp_init,
                                    _digits), **{k: list(v) if isinstance(v, tuple) else v
                                                 for k, v in c.items()})
    n = workers
    local = d["local"].reshape(n, -1)
    grad_bytes = int(d["whole"].nbytes)
    ring_bytes = int(d["ring_sent"][0])
    loss, shown = d["loss"], c["shown"]
    err = float(np.max(np.abs(d["ring_gather"][-1] - d["whole"].ravel())))
    study = Study(
        "Data parallelism and ring all-reduce",
        "Data-parallel SGD on MNIST: four workers' real gradients, averaged on a simulated ring",
        (
            Panel("Different batches, different gradients",
                  "Four workers hold the same partly trained weights and see 16 digits each.",
                  "matrix",
                  dict(values=local, local=d["local"], digits=d["digits"], labels=d["labels"],
                       cosines=d["cosines"], cosine_to_average=d["cosine_to_average"],
                       alike=d["alike"]),
                  ("Cosine between two workers' gradients of this layer: %.2f to %.2f" % (
                      d["cosines"][~np.eye(n, dtype=bool)].min(),
                      d["cosines"][~np.eye(n, dtype=bool)].max()),)),
            Panel("Reduce-scatter",
                  "Three neighbour exchanges leave each worker one chunk summed over all four.",
                  "matrix",
                  dict(values=d["ring_reduce"][-1], states=d["ring_reduce"],
                       counts=d["ring_counts"], sends=d["ring_reduce_sends"],
                       edges=d["ring_edges"])),
            Panel("All-gather",
                  "Three more exchanges give every worker every chunk of the average.",
                  "matrix",
                  dict(values=d["ring_gather"][-1], states=d["ring_gather"], have=d["ring_have"],
                       sends=d["ring_gather_sends"], whole=d["whole"],
                       bytes=np.array([grad_bytes, ring_bytes]),
                       bytes_labels=["gradient size", "ring bytes"]),
                  ("Largest difference from the whole-batch gradient: %.2g" % err,)),
            Panel("Training with the ring, and on one machine",
                  "Held-out loss after every step; the two runs' weights never part.",
                  "curve",
                  dict(x=np.arange(steps + 1), ys=loss.T,
                       labels=["4 workers, ring all-reduce", "1 machine, all 64 digits"],
                       accuracy=d["accuracy"].T, weights_apart=d["weights_apart"],
                       average_error=d["average_error"], step_apart=d["step_apart"],
                       step_loss=d["step_loss"]),
                  ("Largest weight difference after %d steps: %.2g" % (
                      steps, d["weights_apart"][-1]),)),
        ),
        dict(
            seed=SEED, kind="redesigned experiment: real gradients, simulated communication",
            model="tanh MLP %s on 7x7 average-pooled MNIST, cross-entropy" % (c["sizes"],),
            optimiser="plain SGD, lr %g, %d steps of %d digits (%d workers x %d), every digit "
                      "used once" % (c["lr"], steps, batch, workers, per),
            shown_step=shown, drawn_layer="weights of layer %d, %s" % (
                c["layer"] + 1, "x".join(str(v) for v in d["local"].shape[1:])),
            control="the same initial weights trained on one machine with each whole batch",
            heldout="%d test digits" % c["heldout"],
            heldout_loss_start=loss[0].tolist(), heldout_loss_end=loss[-1].tolist(),
            heldout_accuracy_end=d["accuracy"][-1].tolist(),
            largest_average_error=float(d["average_error"].max()),
            largest_weight_difference=float(d["weights_apart"].max()),
            one_step_weight_difference=float(d["step_apart"]),
            workers_disagree_after_gather=float(d["workers_apart"]),
            gradient_bytes_layer=grad_bytes, ring_bytes_layer=ring_bytes,
            gradient_bytes_model=int(d["model_bytes"]),
            ring_bytes_model_per_worker=[int(v) for v in d["model_sent"]],
            communication="simulated in NumPy (buffers passed between arrays); bytes are exact "
                          "counts, not measurements of a network",
            data=json.loads(str(d["provenance"])) if "provenance" in d else provenance,
            designs_tried=["first study: 12 corner pixels (mostly blank) into random 12x8 "
                           "weights, squared error, one gradient; replaced by this one, run "
                           "once as designed"],
        ),
    )
    return study


# ====================================================================== 87
def run_pipeline(params, x, micro):
    """``x`` through the layers as a pipeline: one layer per stage, ``micro``
    micro-batches entering one per time step and moving down a stage per step.
    Returns ``(outputs in order, timeline)``; ``timeline[stage, step]`` is the 1-based
    micro-batch that stage worked on in that step (0: it sat idle)."""
    stages = len(params)
    batches = np.array_split(x, micro)
    timeline = np.zeros((stages, micro + stages - 1), int)
    passed = [None] * stages               # what each stage handed on in the last step
    outputs = [None] * micro
    for step in range(micro + stages - 1):
        handing = [None] * stages
        for s in range(stages):
            got = (step, batches[step]) if s == 0 and step < micro else passed[s - 1] if s else None
            if got is None:
                continue
            j, a = got
            a = layer(params, s, a)
            timeline[s, step] = j + 1
            if s == stages - 1:
                outputs[j] = a
            else:
                handing[s] = (j, a)
        passed = handing
    return np.concatenate(outputs), timeline


TENSOR = dict(sizes=(49, 12, 16, 12, 10), devices=4, train=4096, steps=600, batch=64, lr=0.01,
              heldout=2000, shown=16, layer=1, micro=8, micro_sweep=(1, 2, 4, 8, 16, 32))


def tensor_pipeline():
    """One layer of a trained MNIST classifier split over four devices by column and
    by row, and its four layers run as a four-stage pipeline."""
    c = TENSOR
    devices, k = c["devices"], c["layer"]
    provenance = {}

    def build():
        x, y, meta, _ = _digits(c["train"])
        xt, yt, tmeta, full = _digits(c["heldout"], "test")
        provenance.update(train=_provenance(meta), heldout=_provenance(tmeta))
        rng = np.random.default_rng(SEED)
        params = train_adam(mlp_init(c["sizes"], rng), mlp_grad, x, y, c["steps"], c["batch"],
                            c["lr"], rng)
        w, b = params[k]

        def by_column(a):
            """Each device: its columns of the weights, its slice of the output."""
            return [a @ part for part in np.split(w, devices, axis=1)]

        def by_row(a):
            """Each device: its columns of the input, the matching rows of the weights."""
            return [p @ q for p, q in zip(np.split(a, devices, axis=1),
                                          np.split(w, devices, axis=0))]

        def logits(product):
            """The classifier with layer ``k``'s matrix product computed by ``product``."""
            a = xt
            for i in range(len(params)):
                a = np.tanh(product(a) + b) if i == k else layer(params, i, a)
            return a

        whole = mlp_forward(params, xt)[-1]
        cols = logits(lambda a: np.concatenate(by_column(a), 1))
        rows = logits(lambda a: sum(by_row(a)))
        n = c["shown"]
        inputs = mlp_forward(params, xt[:n])[k]               # what the layer is fed
        piped, timeline = run_pipeline(params, xt[:n], c["micro"])
        sweep = [run_pipeline(params, xt[:32], m) for m in c["micro_sweep"]]
        timelines = -np.ones((len(sweep), len(params), max(c["micro_sweep"]) + len(params) - 1),
                             int)                             # -1: past the end of that run
        for row, (_, t) in zip(timelines, sweep):
            row[:, :t.shape[1]] = t
        return dict(
            x=inputs, w=w, cols=np.stack(by_column(inputs)), parts=np.stack(by_row(inputs)),
            reference=inputs @ w, digits=full[:n].astype(np.float32), labels=yt[:n],
            accuracy=np.array([np.mean(v.argmax(1) == yt) for v in (whole, cols, rows)]),
            logit_gap=np.array([np.max(np.abs(cols - whole)), np.max(np.abs(rows - whole))]),
            timeline=timeline, pipeline_gap=np.max(np.abs(piped - whole[:n])),
            util=np.array([(t > 0).mean() for _, t in sweep]), timelines=timelines,
            idle=np.array([(t == 0).sum() for _, t in sweep]),
            sweep_gap=np.array([np.max(np.abs(o - whole[:32])) for o, _ in sweep]),
            provenance=np.array(json.dumps(provenance)))

    d = _cache("tensor-pipeline", build, (tensor_pipeline, run_pipeline, train_adam, mlp_grad,
                                          mlp_init, _digits),
               **{key: list(v) if isinstance(v, tuple) else v for key, v in c.items()})
    cols = np.concatenate(d["cols"], 1)
    rows = d["parts"].sum(0)
    err = max(float(np.max(np.abs(cols - d["reference"]))),
              float(np.max(np.abs(rows - d["reference"]))))
    micro = np.array(c["micro_sweep"])
    return Study(
        "Tensor and pipeline parallelism",
        "A layer of a trained MNIST classifier split over four devices; its layers as a pipeline",
        (
            Panel("Column partitions concatenate",
                  "Each device holds 4 columns of the weights and computes 4 output columns.",
                  "matrix",
                  dict(values=cols, x=d["x"], w=d["w"], cols=d["cols"], reference=d["reference"],
                       digits=d["digits"], labels=d["labels"])),
            Panel("Row partitions reduce",
                  "Each device holds 3 input columns and 3 weight rows; the partial products add.",
                  "matrix",
                  dict(values=rows, parts=d["parts"], accuracy=d["accuracy"],
                       logit_gap=d["logit_gap"]),
                  ("Maximum numerical error: %.2g" % err,)),
            Panel("Micro-batches fill a pipeline",
                  "Rows are the four layers as stages; numbers identify the micro-batch at work.",
                  "matrix",
                  dict(values=d["timeline"], gap=d["pipeline_gap"])),
            Panel("The pipeline bubble shrinks",
                  "Share of stage-steps at work when 32 digits are cut into M micro-batches.",
                  "curve",
                  dict(x=micro, ys=d["util"][None], labels=["utilisation"], idle=d["idle"],
                       gap=d["sweep_gap"], timelines=d["timelines"]),
                  ("Measured from the executed schedule; equals M/(M+P-1).",)),
        ),
        dict(
            seed=SEED, kind="redesigned experiment: a trained layer, exact partitioning, an "
                            "executed pipeline schedule",
            model="tanh MLP %s on 7x7 average-pooled MNIST, cross-entropy" % (c["sizes"],),
            optimiser="Adam, lr %g, %d steps of %d on %d training digits" % (
                c["lr"], c["steps"], c["batch"], c["train"]),
            split_layer="layer %d: its %s weights, fed the %d activations of layer %d for %d "
                        "held-out digits" % (k + 1, "x".join(map(str, d["w"].shape)),
                                             d["w"].shape[0], k, c["shown"]),
            heldout="%d test digits" % c["heldout"],
            heldout_accuracy=dict(zip(("one machine", "by column", "by row"),
                                      (float(v) for v in d["accuracy"]))),
            largest_logit_difference=dict(zip(("by column", "by row"),
                                              (float(v) for v in d["logit_gap"]))),
            pipeline="%d stages (one layer each), %d micro-batches of %d digits; outputs differ "
                     "from the whole batch by at most %.2g" % (
                         len(c["sizes"]) - 1, c["micro"], c["shown"] // c["micro"],
                         float(d["pipeline_gap"])),
            timing="time steps are the schedule's, not measured seconds; no device is simulated",
            data=json.loads(str(d["provenance"])),
            designs_tried=["first study: 12 corner pixels (mostly blank) into random 12x16 "
                           "weights and a hand-filled timeline; replaced by this one, run once "
                           "as designed"],
        ),
    )


# ====================================================================== 89
START, FORWARD, RECOMPUTE, BACKWARD, END = range(5)


def checkpointed_grad(params, x, y, spacing, trace=None):
    """Backpropagation that stores only every ``spacing``-th activation.

    Forward: each layer runs once; activation ``i`` is kept when ``i`` is a multiple
    of ``spacing``, and the logits are kept; anything else is dropped as soon as the
    next layer has used it. Backward, a segment at a time from the last: the
    activations between two checkpoints are computed again from the earlier one,
    used, and each is dropped once the gradient has crossed the layer above it. With
    ``spacing`` 1 there is nothing between checkpoints: plain backpropagation.

    Returns ``(grads, stats)``; ``stats``: ``peak`` (most activations in memory at
    once, the input included), ``peak_bytes``, ``calls`` (layer evaluations).
    ``trace``, a list, receives one row per step: ``(kind, at, calls, held, done)``
    with ``held[i]`` 0 / 1 (stored) / 2 (transient) and ``done[i]`` true once layer
    ``i``'s gradient is known."""
    n = len(params)
    live = {0: x}                           # every activation in memory, by index
    stored = {0}
    stats = dict(peak=1, peak_bytes=x.nbytes, calls=0)
    done = np.zeros(n, bool)

    def note(kind, at):
        stats["peak"] = max(stats["peak"], len(live))
        stats["peak_bytes"] = max(stats["peak_bytes"], sum(a.nbytes for a in live.values()))
        if trace is not None:
            held = np.zeros(n + 1, np.int8)
            for i in live:
                held[i] = 1 if i in stored else 2
            trace.append((kind, at, stats["calls"], held, done.copy()))

    def run(i):
        live[i + 1] = layer(params, i, live[i])
        stats["calls"] += 1

    note(START, 0)
    for i in range(n):
        run(i)
        if (i + 1) % spacing == 0 or i + 1 == n:
            stored.add(i + 1)
        note(FORWARD, i + 1)
        if i not in stored:
            del live[i]
    delta = logit_gradient(live[n], y)
    grads = [None] * n
    marks = sorted(stored)
    for start, end in zip(marks[-2::-1], marks[:0:-1]):
        for i in range(start, end - 1):
            run(i)
            note(RECOMPUTE, i + 1)
        for i in range(end - 1, start - 1, -1):
            grads[i] = [live[i].T @ delta, delta.sum(0)]
            if i:
                delta = (delta @ params[i][0].T) * (1.0 - live[i] ** 2)
            del live[i + 1]
            done[i] = True
            note(BACKWARD, i)
    note(END, 0)
    return grads, stats


CHECKPOINT = dict(sizes=(49,) + (16,) * 15 + (10,), train=8192, steps=1500, batch=64, lr=0.003,
                  heldout=2000, measured=64, train_spacing=4)


def checkpointing():
    """A 16-layer MNIST classifier, trained with checkpointed backpropagation; then one
    gradient taken at every checkpoint spacing, with what each held and ran."""
    c = CHECKPOINT
    n = len(c["sizes"]) - 1
    spacings = np.arange(1, n + 1)
    provenance = {}

    def build():
        x, y, meta, full = _digits(c["train"])
        xt, yt, tmeta, _ = _digits(c["heldout"], "test")
        provenance.update(train=_provenance(meta), heldout=_provenance(tmeta))
        rng = np.random.default_rng(SEED)
        start = mlp_init(c["sizes"], rng)
        params = train_adam(start, lambda p, a, b: checkpointed_grad(p, a, b,
                                                                     c["train_spacing"])[0],
                            x, y, c["steps"], c["batch"], c["lr"], rng)
        bx, by = x[:c["measured"]], y[:c["measured"]]
        plain = mlp_grad(params, bx, by)                      # every activation stored
        rows, held, done, first = [], [], [], []
        peak, peak_bytes, calls, errors = [], [], [], []
        for k in spacings:
            trace = []
            grads, stats = checkpointed_grad(params, bx, by, int(k), trace)
            errors.append(gap(grads, plain))
            peak.append(stats["peak"])
            peak_bytes.append(stats["peak_bytes"])
            calls.append(stats["calls"])
            first.append(grads[0][0])
            for kind, at, ran, h, dn in trace:
                rows.append((k, kind, at, ran))
                held.append(h)
                done.append(dn)
        return dict(
            peak=peak, peak_bytes=peak_bytes, calls=calls, errors=errors,
            trace=np.array(rows), trace_held=np.stack(held), trace_done=np.stack(done),
            plain_first=plain[0][0], first=np.stack(first),
            gradient_scale=np.max(np.abs(flat(plain))),
            before=np.array(evaluate(start, xt, yt)), after=np.array(evaluate(params, xt, yt)),
            digits=full[:8].astype(np.float32), labels=by[:8],
            activation_bytes=np.array([a.nbytes for a in mlp_forward(params, bx)]),
            provenance=np.array(json.dumps(provenance)))

    d = _cache("checkpointing", build, (checkpointing, checkpointed_grad, train_adam, mlp_grad,
                                        mlp_init, _digits),
               **{key: list(v) if isinstance(v, tuple) else v for key, v in c.items()})
    labels = ["k=%d" % k for k in spacings]
    peak, calls = d["peak"].astype(int), d["calls"].astype(int)
    least = [int(k) for k in spacings[peak == peak.min()]]
    root = int(round(np.sqrt(n)))
    tiles = d["plain_first"].T.reshape(4, 4, 7, 7).transpose(0, 2, 1, 3).reshape(28, 28)
    return Study(
        "Activation checkpointing",
        "A 16-layer tanh classifier on MNIST: backpropagation with recomputed segments",
        (
            Panel("Store checkpoints, not every layer",
                  "Most activations in memory at once, measured during each backward pass.",
                  "bars",
                  dict(values=peak, labels=labels, bytes=d["peak_bytes"], spacings=spacings,
                       trace=d["trace"], held=d["trace_held"], done=d["trace_done"],
                       activation_bytes=d["activation_bytes"]),
                  ("Fewest: %d, at spacing %s" % (peak.min(), ", ".join(map(str, least))),)),
            Panel("Recompute during backward",
                  "Layer evaluations counted as they ran: spacing 1 runs each layer once.",
                  "bars", dict(values=calls, labels=labels)),
            Panel("The gradient itself",
                  "First-layer weight gradient, one 7 x 7 tile per hidden unit.",
                  "matrix",
                  dict(values=tiles, plain=d["plain_first"], checkpointed=d["first"],
                       digits=d["digits"], labels=d["labels"])),
            Panel("Same gradients at every spacing",
                  "Largest difference, over all weights, from storing every activation.",
                  "bars", dict(values=d["errors"], labels=labels,
                               scale=d["gradient_scale"])),
        ),
        dict(
            seed=SEED, kind="redesigned experiment: measured memory and compute of an actual "
                            "recomputing backward pass",
            model="tanh MLP on 7x7 average-pooled MNIST, %d layers %s, cross-entropy" % (
                n, c["sizes"]),
            optimiser="Adam, lr %g, %d steps of %d on %d training digits, every gradient taken "
                      "with checkpoint spacing %d" % (c["lr"], c["steps"], c["batch"], c["train"],
                                                      c["train_spacing"]),
            heldout="%d test digits" % c["heldout"],
            heldout_loss_accuracy_before=d["before"].tolist(),
            heldout_loss_accuracy_after=d["after"].tolist(),
            measured_on="one batch of %d training digits at the trained weights" % c["measured"],
            control="independent plain backpropagation storing every activation",
            spacings=spacings.tolist(), peak_activations=peak.tolist(),
            layer_evaluations=calls.tolist(), peak_bytes=d["peak_bytes"].astype(int).tolist(),
            largest_gradient_difference=float(d["errors"].max()),
            largest_gradient_entry=float(d["gradient_scale"]),
            least_memory_spacings=least, sqrt_layers=root,
            counting="an activation counts while its array is referenced: the input, the "
                     "checkpoints, and whatever a running segment has computed and not yet "
                     "dropped; the gradient being passed back is not counted",
            data=json.loads(str(d["provenance"])),
            designs_tried=["first study: random inputs and targets, a scheme that recomputed "
                           "every segment even at spacing 1; replaced by this one, run once as "
                           "designed"],
        ),
    )


# ====================================================================== 90
ROOF = dict(repeats=25, copy_mb=256, peak_sizes=(128, 192, 384), matmul=1024,
            attention=(1024, 64), layernorm=(4096, 1024), decode=(8192, 1024))
ONE_THREAD = ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS")


def roofline_measure(c=ROOF):
    """Time the two ceilings and the four operations (call this in a process whose
    BLAS is held to one thread: :func:`roofline`). After two warm-up calls of each,
    the kernels are timed in rounds - every kernel once a round, ``repeats`` rounds -
    so that a busy spell on the machine falls on all of them and not on one.
    Returns arrays: ``*_seconds`` samples and the exact counts."""
    rng = np.random.default_rng(SEED)
    # ceiling 1, bytes per second: copy an array far larger than any cache. Nothing is
    # computed; every byte is read once and written once.
    n = c["copy_mb"] * (1 << 20) // 8
    src, dst = rng.random(n), np.empty(n)
    kernels = [lambda: np.copyto(dst, src)]
    # ceiling 2, operations per second: multiply matrices small enough to stay in
    # cache, over and over, so memory delivers nothing new. None of these shapes is
    # an operation on the chart.
    peak_flops, peak_bytes = [], []
    for m in c["peak_sizes"]:
        a, b, out = rng.random((m, m)), rng.random((m, m)), np.empty((m, m))
        reps = max(1, int(round(4e8 / (2 * m ** 3))))

        def many(a=a, b=b, out=out, reps=reps):
            for _ in range(reps):
                np.matmul(a, b, out=out)

        kernels.append(many)
        peak_flops.append(2 * m ** 3 * reps)
        peak_bytes.append(3 * m * m * 8)
    # the operations, at sizes that do not fit in cache
    m = c["matmul"]
    a, b = rng.random((m, m)), rng.random((m, m))
    tokens, width = c["attention"]
    q, k, v = (rng.normal(size=(tokens, width)) for _ in range(3))
    rows, wide = c["layernorm"]
    h = rng.normal(size=(rows, wide))
    vocab, hidden = c["decode"]
    unembed, state = rng.normal(size=(vocab, hidden)), rng.normal(size=hidden)

    def attention():
        z = q @ k.T / np.sqrt(width)
        e = np.exp(z - z.max(-1, keepdims=True))
        return (e / e.sum(-1, keepdims=True)) @ v

    ops = (
        ("matmul", "%d x %d times %d x %d" % (m, m, m, m), lambda: a @ b,
         2 * m ** 3, 3 * m * m * 8),
        # scores and their use: two matrix products, and five passes over the scores
        # (scale, subtract the row maximum, exponential, row sum, divide)
        ("attention", "%d tokens, %d wide" % (tokens, width), attention,
         4 * tokens ** 2 * width + 5 * tokens ** 2, (4 * tokens * width + 2 * tokens ** 2) * 8),
        # per value: mean, centre, square, sum, divide by the root: 8 operations
        ("layer norm", "%d rows of %d" % (rows, wide),
         lambda: (h - h.mean(1, keepdims=True)) / np.sqrt(h.var(1, keepdims=True) + 1e-5),
         8 * rows * wide, 2 * rows * wide * 8),
        ("decode", "%d x %d times one vector" % (vocab, hidden), lambda: unembed @ state,
         2 * vocab * hidden, (vocab * hidden + hidden + vocab) * 8),
    )
    kernels += [o[2] for o in ops]
    for fn in kernels:
        fn()
        fn()
    load = [os.getloadavg()[0]]
    seconds = np.zeros((len(kernels), c["repeats"]))
    for r in range(c["repeats"]):
        for j, fn in enumerate(kernels):
            t = time.perf_counter()
            fn()
            seconds[j, r] = time.perf_counter() - t
    load.append(os.getloadavg()[0])
    sizes = len(c["peak_sizes"])
    return dict(
        copy_seconds=seconds[0].tolist(), copy_bytes=2 * src.nbytes,
        peak_seconds=seconds[1:1 + sizes].tolist(), peak_flops=peak_flops,
        peak_bytes=peak_bytes, names=[o[0] for o in ops], what=[o[1] for o in ops],
        seconds=seconds[1 + sizes:].tolist(), flops=[o[3] for o in ops],
        bytes=[o[4] for o in ops], load=load, cores=os.cpu_count())


def _measure_on_one_thread():
    """:func:`roofline_measure` in a fresh interpreter whose BLAS may use one thread:
    the thread count is read when the library loads, so it cannot be set from here.
    One thread, because the copy that measures the memory ceiling uses one."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, **dict.fromkeys(ONE_THREAD, "1"))
    code = ("import json, sys; sys.path.insert(0, %r); "
            "from voxanim.study_systems import roofline_measure; "
            "print('ROOFLINE' + json.dumps(roofline_measure()))" % root)
    out = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True,
                         text=True).stdout
    line = [row for row in out.splitlines() if row.startswith("ROOFLINE")][-1]
    return {k: np.array(v) for k, v in json.loads(line[len("ROOFLINE"):]).items()}


def spread(samples):
    """``(median, lowest, highest)`` of a set of repeated measurements."""
    s = np.asarray(samples, float)
    return np.array([np.median(s, -1), s.min(-1), s.max(-1)])


def roofline():
    """Two ceilings measured on their own - bytes per second from a large copy,
    operations per second from in-cache matrix products - and four operations timed
    and placed under the roof those two draw."""
    c = ROOF
    d = _cache("roofline", _measure_on_one_thread, (roofline_measure, _measure_on_one_thread),
               threads=1, dtype="float64",
               **{key: list(v) if isinstance(v, tuple) else v for key, v in c.items()})
    names = [str(v) for v in d["names"]]
    flops, nbytes = d["flops"].astype(float), d["bytes"].astype(float)
    intensity = flops / nbytes
    speed = flops[None] / spread(d["seconds"])[[0, 2, 1]] / 1e9      # median, slowest, fastest
    bandwidth = d["copy_bytes"] / spread(d["copy_seconds"])[[0, 2, 1]] / 1e9
    rates = d["peak_flops"][None].astype(float) / spread(d["peak_seconds"])[[0, 2, 1]] / 1e9
    best = int(np.argmax(rates[0]))        # the in-cache size with the highest median rate
    peak = rates[:, best]
    xs = np.geomspace(0.01, 100, 80)
    roof = np.minimum(peak[0], bandwidth[0] * xs)
    ceiling = np.minimum(peak[0], bandwidth[0] * intensity)
    reach = speed / ceiling
    micros = spread(d["seconds"]) * 1e6
    return Study(
        "The roofline performance model",
        "Exact operation and byte counts; two ceilings and four operations timed in NumPy",
        (
            Panel("Arithmetic intensity",
                  "Floating-point operations per byte read or written, counted exactly.",
                  "bars", dict(values=intensity, labels=names, flops=flops, bytes=nbytes,
                               what=[str(v) for v in d["what"]])),
            Panel("Observed throughput",
                  "Billion operations a second: median of %d timed calls on one thread." % (
                      c["repeats"]),
                  "bars", dict(values=speed[0], labels=names, low=speed[1], high=speed[2],
                               ceiling=ceiling, reach=reach[0], reach_low=reach[1],
                               reach_high=reach[2])),
            Panel("A roof measured on its own",
                  "Memory ceiling from a %d MB copy, arithmetic ceiling from in-cache products."
                  % c["copy_mb"],
                  "curve",
                  dict(x=xs, ys=roof[None], labels=["measured roof"], bandwidth=bandwidth,
                       peak=peak, peak_size=np.array(c["peak_sizes"][best]),
                       peak_sizes=np.array(c["peak_sizes"]), peak_rates=rates,
                       copy_seconds=d["copy_seconds"], copy_bytes=d["copy_bytes"],
                       copy_mb=np.array(c["copy_mb"]),
                       copy_rates=d["copy_bytes"] / d["copy_seconds"] / 1e9,
                       peak_seconds=d["peak_seconds"][best], peak_flops=d["peak_flops"][best],
                       peak_samples=d["peak_flops"][best] / d["peak_seconds"][best] / 1e9),
                  ("Peak %.3g GFLOP/s, %.3g GB/s: neither comes from an operation on the chart."
                   % (peak[0], bandwidth[0]),)),
            Panel("Latency of each operation",
                  "Median of %d calls, in microseconds; spread kept beside it." % c["repeats"],
                  "bars", dict(values=micros[0], labels=names, low=micros[1], high=micros[2]),
                  ("Times are local observations on one machine.",)),
        ),
        dict(
            seed=SEED, kind="redesigned experiment: independently measured ceilings",
            repeats=c["repeats"], warmup_calls=2, threads="BLAS held to one thread (%s=1) in a "
            "fresh interpreter, because the copy that measures memory uses one" % ", ".join(
                ONE_THREAD),
            memory_ceiling="np.copyto of a %d MB float64 array: bytes read plus bytes written "
                           "per second" % c["copy_mb"],
            arithmetic_ceiling="np.matmul of n x n matrices that stay in cache, n in %s, "
                               "repeated to about 0.4 billion operations a call; the highest "
                               "median rate is the ceiling (n = %d)" % (
                                   list(c["peak_sizes"]), c["peak_sizes"][best]),
            bandwidth_gb_per_s=dict(zip(("median", "lowest", "highest"), bandwidth.tolist())),
            peak_gflop_per_s=dict(zip(("median", "lowest", "highest"), peak.tolist())),
            peak_by_size={str(n): r for n, r in zip(c["peak_sizes"], rates[0].tolist())},
            operations={n: dict(what=str(w), flops=float(f), bytes=float(b), intensity=float(i),
                                gflop_per_s=dict(zip(("median", "lowest", "highest"),
                                                     s.tolist())),
                                share_of_roof=float(r))
                        for n, w, f, b, i, s, r in zip(names, d["what"], flops, nbytes, intensity,
                                                       speed.T, reach[0])},
            ridge_operations_per_byte=float(peak[0] / bandwidth[0]),
            counts="operations and bytes are formulas for the mathematical operation (each "
                   "matrix read or written once); NumPy's own extra passes are not counted, "
                   "which is why it can run under its roof",
            caveat="local timings of NumPy on this machine, cached at first build: not a chip "
                   "specification",
            machine_load=_load_note(float(d["load"][0]), float(d["load"][1]), int(d["cores"])),
            designs_tried=[
                "first study: roof fitted through the fastest two of the four operations",
                "probe before this design was fixed: with the BLAS free to use every core, "
                "a matrix-vector product moved more bytes a second than a one-thread copy, so "
                "everything is now held to one thread",
            ],
        ),
    )


STUDIES = {
    "data-parallel-allreduce": allreduce,
    "tensor-and-pipeline-parallel": tensor_pipeline,
    "activation-checkpointing": checkpointing,
    "roofline": roofline,
}
