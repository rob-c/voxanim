"""The experiment behind animation 159, "Training on a CPU and on a GPU" (see
:func:`voxanim.next_catalog.redesigned`).

Two models are trained on MNIST: a small classifier (784 -> 128 -> 10) and a
larger dense autoencoder (784 -> 2048 -> 512 -> 64 -> 512 -> 2048 -> 784, 53 times the
weights). The autoencoder is dense, not convolutional, so that both models do the
same kind of arithmetic - matrix multiplications - and differ only in how much of it.

Two kinds of number are shown, and kept apart:

* **Speeds are representative figures, not measurements** (the owner's decision: quote
  typical numbers that make the point, rather than time one machine). They are the
  round figures of :data:`REPRESENTATIVE`, for an ordinary multi-core desktop CPU and
  a separate, mid-range graphics card, and everything about time on screen is
  arithmetic from that one table: a step is a fixed cost plus a cost per digit.
* **Learning curves are measured**: trained here, in plain numpy in float32, from
  three seeds, on the 10,000 official test digits that no training step sees.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import itertools

import numpy as np

from .cache import cached_npz
from .study import Panel, Study

VERSION = 1

FLOAT = np.float32
MODELS = {
    "classifier": dict(sizes=(784, 128, 10), loss="cross-entropy",
                       output="softmax over the ten digits"),
    "autoencoder": dict(sizes=(784, 2048, 512, 64, 512, 2048, 784), loss="squared error",
                        output="sigmoid: the 784 pixels again"),
}
NAMES = tuple(MODELS)
RATE = 1e-3                       # Adam, both models
BATCH = 128                       # the batch size of the headline numbers and the curves
BATCHES = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
EPOCHS = 5                        # of the curves, and of the race drawn from them
SEEDS = (0, 1, 2)
N_TRAIN = 60000
EVERY = 78                        # updates between held-out measurements (6 an epoch)
SHOWN = 8                         # held-out digits kept as pictures


# ---------------------------------------------------------------------- the two models
def layers(name):
    """``[(inputs, outputs)]`` of the model's dense layers."""
    sizes = MODELS[name]["sizes"]
    return list(itertools.pairwise(sizes))


def parameters(name):
    """Weights and biases of the model."""
    return int(sum(a * b + b for a, b in layers(name)))


def multiply_adds(name, batch=1):
    """Multiply-adds of one training step on ``batch`` digits, counting only the matrix
    products: the forward pass (one per layer), the gradient of every weight matrix
    (one per layer) and the gradient passed back to the layer before (every layer but
    the first)."""
    each = [a * b for a, b in layers(name)]
    return int(batch * (2 * sum(each) + sum(each[1:])))


def init_weights(name, seed):
    """He-normal weights and zero biases, float32: ``W0, b0, W1, b1, ...``."""
    rng = np.random.default_rng([int(seed), NAMES.index(name)])
    P = {}
    for k, (a, b) in enumerate(layers(name)):
        P["W%d" % k] = rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(FLOAT)
        P["b%d" % k] = np.zeros(b, FLOAT)
    return P


def batch_order(seed, epochs, batch=BATCH, n=N_TRAIN):
    """The rows of every batch, ``(steps, batch)``: each epoch a fresh shuffle of the
    ``n`` training digits, cut into full batches (the remainder is dropped)."""
    rng = np.random.default_rng([int(seed), 99])
    per = n // batch
    return np.concatenate([rng.permutation(n)[:per * batch].reshape(per, batch)
                           for _ in range(epochs)])


def forward(name, P, x):
    """Every layer's output for the inputs ``x``: ReLU between layers; the last layer
    gives class scores (classifier) or, through a sigmoid, pixels (autoencoder)."""
    acts, n = [x], len(layers(name))
    for k in range(n):
        z = acts[-1] @ P["W%d" % k] + P["b%d" % k]
        if k < n - 1:
            z = np.maximum(z, 0)
        elif name == "autoencoder":
            z = 1.0 / (1.0 + np.exp(-z))
        acts.append(z)
    return acts


def loss_and_grads(name, P, x, y=None):
    """The loss of one batch and its gradient for every parameter. Classifier: mean
    cross-entropy of the softmax against the labels ``y``. Autoencoder: mean squared
    difference between the output pixels and the input pixels."""
    acts = forward(name, P, x)
    out, n = acts[-1], len(layers(name))
    if name == "classifier":
        z = out - out.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        rows = np.arange(len(x))
        loss = float(-logp[rows, y].mean())
        d = np.exp(logp)
        d[rows, y] -= 1.0
        d /= len(x)
    else:
        diff = out - x
        loss = float((diff * diff).mean())
        d = (2.0 / diff.size) * diff * out * (1.0 - out)
    d = d.astype(FLOAT, copy=False)
    grads = {}
    for k in reversed(range(n)):
        grads["W%d" % k] = acts[k].T @ d
        grads["b%d" % k] = d.sum(0)
        if k:
            d = (d @ P["W%d" % k].T) * (acts[k] > 0)
    return loss, grads


def numpy_step(name, P, opt, x, y=None):
    """One training step in place: forward, loss, backward, Adam update. Returns the loss
    before the update."""
    loss, grads = loss_and_grads(name, P, x, y)
    opt.step(P, grads)
    return loss


def held_out(name, P, xt, yt, chunk=2500):
    """``(loss, accuracy)`` on held-out digits: the model's own loss; the classifier's
    share of digits named correctly (0 for the autoencoder, which names nothing)."""
    total, right = 0.0, 0
    for s in range(0, len(xt), chunk):
        xb = xt[s:s + chunk]
        out = forward(name, P, xb)[-1]
        if name == "classifier":
            z = out - out.max(1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(1, keepdims=True))
            total += float(-logp[np.arange(len(xb)), yt[s:s + chunk]].sum())
            right += int((out.argmax(1) == yt[s:s + chunk]).sum())
        else:
            total += float(((out - xb) ** 2).mean(1).sum())
    return total / len(xt), right / len(xt)


# ---------------------------------------------------------------------- the speeds
# Representative round figures, chosen to illustrate and NOT measured: what an ordinary
# multi-core desktop CPU and a separate, mid-range graphics card typically take for
# these two models. Real machines differ, in both directions. Everything on screen
# about time is derived from this one table.
#   step_ms:  one training step on a batch of 128 digits, everything included
#   fixed_ms: the part of that step that does not grow with the batch (starting the
#             work, and on the GPU launching it there and the trip across the bus)
REPRESENTATIVE = {
    "batch": 128,
    "step_ms": {("classifier", "cpu"): 1.0, ("classifier", "gpu"): 0.5,
                ("autoencoder", "cpu"): 40.0, ("autoencoder", "gpu"): 2.0},
    "fixed_ms": {("classifier", "cpu"): 0.1, ("classifier", "gpu"): 0.45,
                 ("autoencoder", "cpu"): 0.5, ("autoencoder", "gpu"): 0.6},
    "pick_ms": 0.05,              # picking a batch of 128 out of memory, on the CPU
    "move_ms": 0.1,               # moving that batch across the bus to the GPU
    "copy_ms": {"classifier": 0.5, "autoencoder": 5.0},   # all the weights, to the GPU
    "megabytes": {"classifier": 0.4, "autoencoder": 21.5},
}
DEVICES = ("cpu", "gpu")
FOUR = tuple((name, dev) for name in NAMES for dev in DEVICES)
LABEL = {"cpu": "CPU", "gpu": "GPU"}


def per_digit_ms(name, device):
    """The cost model's cost of one more digit in the batch."""
    R = REPRESENTATIVE
    return (R["step_ms"][name, device] - R["fixed_ms"][name, device]) / R["batch"]


def step_ms(name, device, batch=BATCH):
    """Milliseconds of one training step under the simple cost model: a fixed cost
    plus a cost per digit. At the table's batch it is the table's figure."""
    return REPRESENTATIVE["fixed_ms"][name, device] + per_digit_ms(name, device) * batch


def parts_ms(name, device):
    """A step at the table's batch as ``(pick the batch, move it, arithmetic)``."""
    R = REPRESENTATIVE
    move = R["move_ms"] if device == "gpu" else 0.0
    return R["pick_ms"], move, R["step_ms"][name, device] - R["pick_ms"] - move


def epoch_s(name, device):
    """Seconds of one pass over the 60,000 training digits: steps times time per step."""
    return (N_TRAIN // BATCH) * step_ms(name, device) / 1e3


def crossover(name):
    """The smallest batch size of :data:`BATCHES` at which the GPU is ahead (None: never)."""
    return next((b for b in BATCHES if step_ms(name, "gpu", b) < step_ms(name, "cpu", b)), None)


def duration(seconds):
    """A time in the unit a person would use: ``"3.2 s"``, ``"14 min"``, ``"2.6 days"``."""
    s = float(seconds)
    for limit, unit, name in ((1e-3, 1e-6, "µs"), (1.0, 1e-3, "ms"), (120.0, 1.0, "s"),
                              (7200.0, 60.0, "min"), (172800.0, 3600.0, "hours"),
                              (63072000.0, 86400.0, "days")):
        if s < limit:
            v = s / unit
            return ("%.2g %s" if v < 9.95 else "%.0f %s") % (v, name)
    return "%.1f years" % (s / 31536000.0)


# ---------------------------------------------------------------------- the curves
def mnist_data():
    """``(x, y, xt, yt)``: the 60,000 training digits and the 10,000 test digits, float32."""
    from . import mnist

    x, y, xt, yt = mnist.load()
    return x.astype(FLOAT), y.astype(np.int64), xt.astype(FLOAT), yt.astype(np.int64)


def train_run(name, seed):
    """One run of ``EPOCHS`` epochs in numpy. Returns ``(steps, loss, accuracy, first,
    P)``: held-out loss and accuracy at step 0 and every ``EVERY`` steps, the training
    loss of each of the first eight steps, and the trained weights."""
    from .optim import Adam

    x, y, xt, yt = mnist_data()
    P = init_weights(name, seed)
    opt = Adam(P, FLOAT(RATE))
    order = batch_order(seed, EPOCHS)
    at, loss, acc, first = [0], [], [], []
    a, b = held_out(name, P, xt, yt)
    loss.append(a)
    acc.append(b)
    for step, rows in enumerate(order, 1):
        v = numpy_step(name, P, opt, x[rows], y[rows])
        if step <= 8:
            first.append(v)
        if step % EVERY == 0 or step == len(order):
            a, b = held_out(name, P, xt, yt)
            at.append(step)
            loss.append(a)
            acc.append(b)
    return np.array(at), np.array(loss), np.array(acc), np.array(first), P


def _run(name, seed):
    at, loss, acc, first, P = train_run(name, seed)
    extra = {}
    if seed == SEEDS[0]:
        _, _, xt, _ = mnist_data()
        acts = forward(name, P, xt[:SHOWN])
        extra = dict(shown=acts[-1].astype(np.float32))
    return at, loss, acc, first, extra


def _build():
    from .study_generative import parallel

    x, y, xt, yt = mnist_data()
    jobs = [(name, seed) for name in NAMES for seed in SEEDS]
    runs = dict(zip(jobs, parallel(_run, jobs)))
    out = {}
    for name in NAMES:
        got = [runs[name, seed] for seed in SEEDS]
        out[name + "_steps"] = got[0][0]
        out[name + "_loss"] = np.array([g[1] for g in got])
        out[name + "_accuracy"] = np.array([g[2] for g in got])
        out[name + "_first"] = np.array([g[3] for g in got])
        out[name + "_shown"] = got[0][4]["shown"]
    # what the drawing needs of the first step of seed 0: its digits, and real weights
    rows = batch_order(SEEDS[0], 1)[0]
    P = init_weights("classifier", SEEDS[0])
    out["batch_rows"] = rows
    out["batch_digits"] = x[rows[:16]].astype(np.float32)
    out["batch_labels"] = y[rows[:16]]
    out["hidden"] = forward("classifier", P, x[rows])[1].astype(np.float32)   # 128 x 128
    out["train_digits"] = x[:12].astype(np.float32)
    out["held_digits"] = xt[:SHOWN].astype(np.float32)
    out["held_labels"] = yt[:SHOWN]
    for name in NAMES:                # a patch of real initial weights, to draw the block
        W = init_weights(name, SEEDS[0])["W0"]
        out[name + "_patch"] = W[300:332, :32].astype(np.float32)
    return out


@functools.lru_cache(maxsize=1)
def curves():
    """The numpy training runs (cached): see :func:`_build`."""
    h = hashlib.sha256()
    for fn in (init_weights, batch_order, forward, loss_and_grads, held_out, train_run, _run,
               _build):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], models=MODELS, rate=RATE,
                    batch=BATCH, epochs=EPOCHS, seeds=SEEDS, every=EVERY)
    return {k: np.asarray(v) for k, v in cached_npz("study-hardware-curves", settings,
                                                    _build).items()}


# ---------------------------------------------------------------------- the study
PROJECT_EPOCHS = 20
PROJECT = (("this autoencoder", 1, 1), ("a model 100× larger", 100, 1),
           ("a model 10,000× larger", 10000, 1), ("100× larger, 1,000× the data", 100, 1000))


def projections():
    """Arithmetic from the representative figures: the time of a whole training of the
    autoencoder (``PROJECT_EPOCHS`` epochs) if time per step grows in proportion to the
    model's arithmetic and the number of steps in proportion to the data.
    ``[(what, model factor, data factor, {device: seconds})]``."""
    return [(what, model, data,
             {dev: PROJECT_EPOCHS * epoch_s("autoencoder", dev) * model * data
              for dev in DEVICES}) for what, model, data in PROJECT]


def cpu_and_gpu_training():
    c = curves()
    four = np.array([step_ms(name, dev) for name, dev in FOUR])
    epochs = np.array([epoch_s(name, dev) for name, dev in FOUR])
    labels = ["%s, %s" % (name, LABEL[dev]) for name, dev in FOUR]
    by_batch = np.array([[step_ms(name, dev, b) for b in BATCHES] for name, dev in FOUR])
    acc = c["classifier_accuracy"]
    ae = c["autoencoder_loss"]
    said = "representative figures, not measured"
    panels = (
        Panel("Time per training step", "milliseconds for one step on a batch of %d digits: "
              "%s" % (BATCH, said), "bars",
              dict(values=four, labels=labels, epoch_seconds=epochs), (said,)),
        Panel("Time per step against batch size", "a simple cost model, a fixed cost plus a "
              "cost per digit, through the representative figures at %d" % BATCH, "curve",
              dict(x=np.array(BATCHES, float), ys=by_batch, labels=labels), (said,)),
        Panel("Classifier: held-out accuracy", "share of the 10,000 test digits named "
              "correctly as training goes on, three seeds", "curve",
              dict(x=c["classifier_steps"].astype(float), ys=acc,
                   labels=["seed %d" % s for s in SEEDS], loss=c["classifier_loss"])),
        Panel("Autoencoder: held-out loss", "mean squared error per pixel on the 10,000 "
              "test digits as training goes on, three seeds", "curve",
              dict(x=c["autoencoder_steps"].astype(float), ys=ae,
                   labels=["seed %d" % s for s in SEEDS])),
    )
    return Study(
        "Training on a CPU and on a GPU",
        "Where the weights and a batch of digits are, where the arithmetic happens, and "
        "how long it takes", panels,
        dict(
            source="speeds: representative figures (study_hardware.REPRESENTATIVE), not "
                   "measured; curves: NumPy on MNIST",
            dataset="MNIST", review_only=True, redesigned=True,
            speeds_measured=False,
            evidence=dict(
                speeds="representative round figures for an ordinary multi-core desktop CPU "
                       "and a separate, mid-range graphics card, chosen to illustrate. They "
                       "were not measured, at the owner's request; real machines differ",
                cost_model="time per step = a fixed cost + a cost per digit × the batch "
                           "size, through the representative figures at a batch of %d"
                           % BATCH,
                curves="held-out loss and accuracy of numpy training on MNIST, three seeds",
                arithmetic="epochs, the race's clock and the projections to larger models "
                           "and datasets are multiplication from the representative "
                           "figures"),
            representative={k: ({"%s/%s" % key: v for key, v in val.items()}
                                if isinstance(val, dict) and
                                isinstance(next(iter(val)), tuple) else val)
                            for k, val in REPRESENTATIVE.items()},
            models={name: dict(sizes=list(MODELS[name]["sizes"]), loss=MODELS[name]["loss"],
                               parameters=parameters(name),
                               multiply_adds_per_step=multiply_adds(name, BATCH))
                    for name in NAMES},
            batch=BATCH, batches=list(BATCHES), epochs=EPOCHS, seeds=list(SEEDS),
            learning_rate=RATE, optimiser="Adam", precision="float32",
            split_sizes=[N_TRAIN, 10000],
            data_splits="the 60,000 official training digits train; the 10,000 official "
                        "test digits are held out",
            step_ms={"%s/%s" % k: float(v) for k, v in zip(FOUR, four)},
            epoch_s={"%s/%s" % k: float(v) for k, v in zip(FOUR, epochs)},
            cpu_over_gpu={name: step_ms(name, "cpu") / step_ms(name, "gpu")
                          for name in NAMES},
            final_accuracy=[float(v) for v in acc[:, -1]],
            final_autoencoder_loss=[float(v) for v in ae[:, -1]],
            projection_epochs=PROJECT_EPOCHS,
            projections=[dict(what=w, model=a, data=b, seconds=s)
                         for w, a, b, s in projections()],
            limitations=[
                "the speeds are representative figures and were not measured (the owner "
                "asked for typical numbers that show the point, not a timing of one "
                "machine): any real CPU and GPU will differ from them",
                "the cost model, a fixed cost plus a cost per digit, is the simplest that "
                "shows why a small model gains little from a GPU; real steps are not "
                "exactly straight lines in the batch size",
                "the projections assume time per step grows in proportion to the "
                "arithmetic; larger models can do better or worse than that, and a very "
                "large model needs more memory than one graphics card has",
                "only the learning curves are measurements: numpy on MNIST, three seeds",
            ],
            designs_tried=[
                "a first version timed both models with PyTorch on one machine's CPU and "
                "GPU; the owner asked for representative figures instead, and the timing "
                "script and its recorded file were removed",
            ],
        ))


STUDIES = {"cpu-and-gpu-training": cpu_and_gpu_training}
