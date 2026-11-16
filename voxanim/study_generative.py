"""Redesigned experiments for five of the generative animations (59-66).

The first experiments (:mod:`voxanim.next_vision`) could not show their concept: a
conditional denoiser whose samples were noise, a one-layer pixel model whose samples were
blank, a codebook crowded into a corner, and a GAN and a flow whose target was one blob.
Each replacement here is a seeded NumPy run on MNIST with a held-out split, a control and
its spread over seeds; :data:`STUDIES` hands them to
:func:`voxanim.next_catalog.build_study`, and :func:`arrays` gives the animation
(:mod:`voxanim.mech_generative`) everything a run saved.

* ``gan-training`` and ``normalizing-flows`` share a target with visible structure:
  the digits 0, 1 and 4 at their two main PCA coordinates, three separate clusters
  (:func:`digit_plane`).
* ``classifier-free-guidance``: a conditional denoiser on 14 x 14 MNIST that draws
  readable digits, sampled at guidance 0, 1, 3 and 8 from the same noise.
* ``autoregressive-pixels``: a six-layer masked-convolution model with two streams (no
  blind spot) on binarised 14 x 14 MNIST, against a one-layer model and independent pixels.
* ``vq-vae-codebook``: a 7 x 7 grid of codes per 28 x 28 digit, with and without the
  codebook remedies.

Runs that differ only by seed are trained in separate processes when that is possible
(:func:`parallel`); every one is determined by its seed, so the numbers are the same
either way.
"""

from __future__ import annotations

import itertools
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from functools import cache, lru_cache

import numpy as np

from . import next_vision_models as vm
from . import realdata
from . import toy2d as toy
from .cache import DEFAULT_CACHE, cached_npz, settings_key, sha256_of
from .diffusion import schedule, silu, silu_grad, time_embedding
from .next_vision import bars, curve, images, matrix, points
from .optim import Adam
from .study import Study

# One number per study: bump it when that study's code changes what it computes.
VERSIONS = {"gan": 2, "flow": 1, "guidance": 2, "pixels": 1, "vq": 2}


SETTINGS = {}                          # of each study built in this process, by key


def cached(key, build, **settings):
    SETTINGS[key] = dict(version=VERSIONS[key], **settings)
    return cached_npz("study_generative_" + key, SETTINGS[key], build)


def provenance(key, d):
    """What a study's numbers came from: the run's settings, the file that holds every
    array it saved (weights, samples, traces) and the identity of its two splits."""
    from . import mnist

    path = os.path.join(DEFAULT_CACHE, "study_generative_%s-%s.npz" % (
        key, settings_key(SETTINGS[key])))
    return dict(
        run_settings=SETTINGS[key], artifact=os.path.abspath(path),
        artifact_sha256=sha256_of(path),
        data=dict(source_url=mnist.URL, raw_sha256=mnist.SHA256,
                  train_x_sha256=str(d["train_x_sha256"]),
                  test_x_sha256=str(d["test_x_sha256"]),
                  splits="training: realdata.mnist_split(split='train'); held out: its "
                         "'test' split (the official test partition)"))


def parallel(fn, jobs):
    """``[fn(*job) for job in jobs]``, in one process per job where the platform allows
    it. Each job is a seeded, self-contained run, so the results do not depend on which."""
    workers = min(len(jobs), max(1, (os.cpu_count() or 2) - 1), 6)
    if (workers > 1 and os.environ.get("VOXANIM_SERIAL") != "1"
            and multiprocessing.current_process().name == "MainProcess"):
        try:
            with ProcessPoolExecutor(workers,
                                     mp_context=multiprocessing.get_context("spawn")) as pool:
                return list(pool.map(fn, *zip(*jobs)))
        except (OSError, RuntimeError, ImportError):
            pass                       # no process pool here: run them one after another
    return [fn(*job) for job in jobs]


def result(name, subtitle, panels, **metadata):
    return Study(name.replace("-", " ").capitalize(), subtitle, tuple(panels),
                 dict(source="NumPy measurements on MNIST", dataset="MNIST", review_only=True,
                      redesigned=True, **metadata))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ====================================================================== the plane of digits
DIGITS = (0, 1, 4)


@lru_cache(maxsize=1)
def digit_plane():
    """The 2-D target of the GAN and the flow: every MNIST 0, 1 and 4 of a 3,000-image
    training draw (and 1,000 held-out ones) at its two main PCA coordinates. The map is
    fitted on the training images only and scaled so the training points have unit
    spread; the three digits fall into three separate clusters."""
    x, y, meta = realdata.mnist_split(3000, "train", 14, classes=DIGITS)
    xt, yt, meta_t = realdata.mnist_split(1000, "test", 14, classes=DIGITS)
    mean = x.mean(0)
    axes = np.linalg.svd(x - mean, full_matrices=False)[2][:2]
    scale = float(((x - mean) @ axes.T).std())
    return dict(train=(x - mean) @ axes.T / scale, test=(xt - mean) @ axes.T / scale,
                train_labels=y, test_labels=yt, train_pixels=x[:240],
                pca_mean=mean, pca_axes=axes, pca_scale=np.array(scale),
                train_x_sha256=np.array(meta["x_sha256"]),
                test_x_sha256=np.array(meta_t["x_sha256"]))


def nearest_real(pts, real):
    """Index of, and distance to, each point's nearest real point."""
    d = ((pts ** 2).sum(1)[:, None] - 2 * pts @ real.T + (real ** 2).sum(1)[None])
    return d.argmin(1), np.sqrt(np.maximum(d.min(1), 0.0))


# ====================================================================== 59 gan-training
GAN_STEPS, GAN_EVERY, GAN_SEEDS, GAN_POINTS = 3000, 50, (1, 2, 3, 4, 5), 500
GAN_COVERED = 0.10                    # a digit is covered when it gets this share of the samples
GAN_AXIS = np.linspace(-3.0, 3.0, 40)


def gan_run(minibatch, seed):
    """One GAN on :func:`digit_plane` (the generator, discriminator, losses and
    learning rates of :func:`next_vision_models.gan_run`), keeping every
    :data:`GAN_EVERY` updates the generator's output on fixed noise and the
    discriminator's score over the plane."""
    real = digit_plane()["train"]
    rng = np.random.default_rng(seed)
    gp = toy.init((2, 32, 32, 2), rng)
    dp = toy.init((3 if minibatch else 2, 32, 32, 1), rng)
    go, do = Adam(gp, 0.002), Adam(dp, 0.001)
    fixed = rng.normal(size=(GAN_POINTS, 2))
    gx, gy = np.meshgrid(GAN_AXIS, GAN_AXIS)
    plane = np.column_stack((gx.ravel(), gy.ravel()))
    fakes, surfaces, losses = [], [], []
    for step in range(GAN_STEPS + 1):
        if step % GAN_EVERY == 0:
            fakes.append(toy.forward(gp, fixed)[0])
            inputs, _ = vm.minibatch_features(plane, minibatch)
            surfaces.append(vm.sigmoid(toy.forward(dp, inputs)[0]).reshape(40, 40))
        if step == GAN_STEPS:
            break
        fake, ga = toy.forward(gp, rng.normal(size=(96, 2)))
        xx = np.vstack((real[rng.choice(len(real), 96)], fake))
        inp, _ = vm.minibatch_features(xx, minibatch)
        target = np.r_[np.ones(96), np.zeros(96)][:, None]
        logits, da = toy.forward(dp, inp)
        dg, _ = toy.backward(dp, da, (vm.sigmoid(logits) - target) / len(xx))
        do.step(dp, dg)
        inp, tape = vm.minibatch_features(fake, minibatch)
        logits, da = toy.forward(dp, inp)
        _, dx = toy.backward(dp, da, (vm.sigmoid(logits) - 1) / len(fake))
        gg, _ = toy.backward(gp, ga, vm.minibatch_backward(dx, tape))
        go.step(gp, gg)
        losses.append(float(np.mean(np.logaddexp(0, logits) - logits)))
    return (np.array(fakes, np.float32), np.array(surfaces, np.float32),
            np.array(losses, np.float32), fixed, gp, dp)


def gan_shares(fakes, plane, reach):
    """The share of generated points landing on each digit's cluster: a point belongs to
    the digit of its nearest real training point, if it is within ``reach`` of it."""
    index, distance = nearest_real(np.asarray(fakes, float), plane["train"])
    label = plane["train_labels"][index]
    return np.array([np.mean((label == digit) & (distance <= reach)) for digit in DIGITS])


def gan_arrays():
    def build():
        plane = digit_plane()
        # how far a real held-out digit sits from its nearest training digit (99th centile)
        reach = float(np.quantile(nearest_real(plane["test"], plane["train"])[1], 0.99))
        jobs = [(mb, seed) for mb in (False, True) for seed in GAN_SEEDS]
        runs = parallel(gan_run, jobs)
        n = len(GAN_SEEDS)
        fakes = np.array([r[0] for r in runs]).reshape(2, n, -1, GAN_POINTS, 2)
        shares = np.array([[[gan_shares(f, plane, reach) for f in run] for run in kind]
                           for kind in fakes])
        return dict(
            real=plane["train"][:240], labels=plane["train_labels"][:240],
            pixels=plane["train_pixels"], held_out=plane["test"],
            held_out_labels=plane["test_labels"], reach=np.array(reach),
            fakes=fakes, shares=shares,
            surfaces=np.array([runs[0][1], runs[n][1]]),
            loss=np.array([runs[0][2], runs[n][2]]),
            steps=np.arange(0, GAN_STEPS + 1, GAN_EVERY),
            train_x_sha256=plane["train_x_sha256"], test_x_sha256=plane["test_x_sha256"],
            # the first seed of each kind: its fixed noise and final networks
            **{"%s_%s" % (kind, key): value
               for kind, run in (("plain", runs[0]), ("crowd", runs[n]))
               for key, value in (
                   ("noise", run[3]),
                   *(("generator_weights_" + k, v) for k, v in run[4].items()),
                   *(("discriminator_weights_" + k, v) for k, v in run[5].items()))})
    return cached("gan", build, steps=GAN_STEPS, every=GAN_EVERY, seeds=list(GAN_SEEDS),
                  points=GAN_POINTS, digits=list(DIGITS), width=32, batch=96,
                  generator_lr=0.002, discriminator_lr=0.001)


def gan_summary(d):
    """Per GAN kind and seed: digits covered at each checkpoint, and over the second
    half of training the share of checkpoints with all three covered and the fewest
    covered at any one."""
    covered = (np.asarray(d["shares"]) >= GAN_COVERED).sum(-1)          # kind, seed, checkpoint
    late = covered[:, :, covered.shape[2] // 2:]
    return covered, (late == len(DIGITS)).mean(-1), late.min(-1)


def gan_training():
    d = gan_arrays()
    covered, all_three, fewest = gan_summary(d)
    return result(
        "gan-training",
        "NumPy GANs on MNIST 0s, 1s and 4s at their two main PCA coordinates: five seeds each",
        [
            points("Three clusters of real digits",
                   "Each point is one MNIST 0, 1 or 4; the PCA map is fitted on training "
                   "images only.", d["real"], d["labels"]),
            points("A plain GAN's samples after training",
                   "Seed 1, 500 fixed noise points after 3,000 updates.", d["fakes"][0, 0, -1]),
            curve("Digits covered through training",
                  "A digit is covered when 10% of the 500 samples land on its cluster. "
                  "Five plain runs, then five crowd-aware runs.",
                  covered.reshape(-1, covered.shape[-1]),
                  ["plain %d" % s for s in GAN_SEEDS] + ["crowd-aware %d" % s
                                                         for s in GAN_SEEDS],
                  d["steps"]),
            matrix("Share of late checkpoints covering all three digits",
                   "Rows: plain GAN, crowd-aware GAN. Columns: seeds 1-5. Second half of "
                   "training, a checkpoint every 50 updates.", all_three,
                   ("fewest digits covered at a late checkpoint: plain %s, crowd-aware %s" % (
                       fewest[0].tolist(), fewest[1].tolist()),
                    "At the last update itself every run covered all three: the plain GAN "
                    "drops a digit and finds it again.")),
        ],
        seeds=list(GAN_SEEDS), updates=GAN_STEPS,
        **provenance("gan", d),
        target="MNIST digits 0, 1, 4: two training-only PCA coordinates of 14x14 images",
        all_three_late=all_three.tolist(), fewest_late=fewest.tolist(),
        covered_at_end=covered[:, :, -1].tolist(),
        scope="2-D coordinates of digits, not digit images; three modes",
        designs_tried=[
            "target: 2-D codes of an autoencoder on all ten digits (one blob), the same with "
            "a classification term (ten arms meeting in the middle), LDA of four or five "
            "digits (clusters touch), PCA of four digits (clusters touch); PCA of digits "
            "0/1/4 was chosen for its separation, before any GAN was trained on it",
            "1,800 updates, coverage counted once at the end: plain 2,3,3,3,2 of 3 digits, "
            "crowd-aware 3,3,3,3,3; replaced by 3,000 updates with coverage followed "
            "through training, because one end-of-run snapshot of an oscillating GAN is "
            "chance",
        ])


# ====================================================================== 60 normalizing-flows
FLOW_STEPS, FLOW_SEEDS, FLOW_LAYERS, FLOW_HIDDEN = 4000, (1, 2, 3), 6, 24


def grid_lines():
    axis = np.linspace(-2.0, 2.0, 60)
    return [np.column_stack((axis, np.full_like(axis, k))) for k in np.linspace(-2, 2, 7)] + [
        np.column_stack((np.full_like(axis, k), axis)) for k in np.linspace(-2, 2, 7)]


def flow_run(seed):
    """Six affine coupling layers (:mod:`voxanim.toy2d`) fitted to :func:`digit_plane`
    by maximum likelihood; the learning rate drops to 0.3 of itself for the last 30%."""
    plane = digit_plane()
    x = plane["train"]
    rng = np.random.default_rng(seed)
    p = toy.flow_init(rng, FLOW_LAYERS, FLOW_HIDDEN)
    opt, loss = Adam(p, 0.003), []
    z = rng.normal(size=(240, 2))
    for step in range(FLOW_STEPS):
        value, g = toy.flow_loss_grads(p, x[rng.choice(len(x), 128)])
        opt.step(p, g, lr=0.003 * (0.3 if step >= 0.7 * FLOW_STEPS else 1.0))
        loss.append(value)
    nll = [toy.flow_loss_grads(p, x)[0], toy.flow_loss_grads(p, plane["test"])[0]]
    return p, np.array(loss), z, np.array(nll)


def gaussian_nll(x, mean, cov):
    """Mean negative log density of ``x`` under one Gaussian."""
    delta = x - mean
    return float(np.mean(0.5 * np.einsum("ni,ij,nj->n", delta, np.linalg.inv(cov), delta)
                         + 0.5 * np.linalg.slogdet(cov)[1] + np.log(2 * np.pi)))


def flow_arrays():
    def build():
        plane = digit_plane()
        x, held = plane["train"], plane["test"]
        runs = parallel(flow_run, [(seed,) for seed in FLOW_SEEDS])
        p, loss, z, _nll = runs[0]
        encoded, logdet, _ = toy.flow_forward(p, x[:240])
        fitted = gaussian_nll(held, x.mean(0), np.cov(x.T))
        return dict(
            x=x[:240], labels=plane["train_labels"][:240], held_out=held,
            held_out_labels=plane["test_labels"], z=z, final=toy.flow_inverse(p, z),
            encoded=encoded, logdet=logdet, loss=loss,
            error=np.max(np.abs(toy.flow_inverse(p, encoded) - x[:240])),
            nll=np.array([r[3] for r in runs]),
            bell=np.array([gaussian_nll(x, np.zeros(2), np.eye(2)),
                           gaussian_nll(held, np.zeros(2), np.eye(2))]),
            fitted_gaussian=np.array(fitted),
            grid=np.array([toy.flow_inverse(p, line) for line in grid_lines()]),
            train_x_sha256=plane["train_x_sha256"], test_x_sha256=plane["test_x_sha256"],
            **{"weights_" + k: v for k, v in p.items()})
    return cached("flow", build, steps=FLOW_STEPS, seeds=list(FLOW_SEEDS), layers=FLOW_LAYERS,
                  hidden=FLOW_HIDDEN, lr=0.003, batch=128, digits=list(DIGITS))


def normalizing_flows():
    d = flow_arrays()
    held = d["nll"][:, 1]
    return result(
        "normalizing-flows",
        "Six affine coupling layers fitted to MNIST 0s, 1s and 4s at two PCA coordinates",
        [
            points("Three clusters of real digits",
                   "PCA is fitted on training images only; the flow never sees a label.",
                   d["x"], d["labels"]),
            points("240 samples of the fitted flow",
                   "Gaussian samples sent through the fitted inverse (seed 1).", d["final"]),
            curve("Likelihood optimisation", "Every value measured during 4,000 Adam updates.",
                  d["loss"], ["training NLL"]),
            bars("Held-out negative log likelihood, nats per digit",
                 "1,000 held-out digits. Lower is better.",
                 np.r_[d["bell"][1], d["fitted_gaussian"], held],
                 ["the bell, no flow", "best single Gaussian"] + ["flow, seed %d" % s
                                                                  for s in FLOW_SEEDS],
                 ("max round-trip error %.3g" % d["error"],
                  "Densities are of the two coordinates, not of the pixels.")),
        ],
        seeds=list(FLOW_SEEDS), updates=FLOW_STEPS,
        **provenance("flow", d),
        target="MNIST digits 0, 1, 4: two training-only PCA coordinates of 14x14 images",
        heldout_nll=held.tolist(), heldout_bell=float(d["bell"][1]),
        heldout_fitted_gaussian=float(d["fitted_gaussian"]),
        designs_tried=[
            "target chosen as for gan-training, before any flow was fitted to it",
            "6 layers x 24 units for 1,800 and 4,000 updates, 8 and 12 layers x 32 units for "
            "4,000 (held-out 1.46-1.54, 1.45-1.50, 1.38-1.43, 1.38-1.43 nats): six layers "
            "kept, the gain over a Gaussian is the same story at every size",
        ])


# ====================================================================== 62 classifier-free-guidance
CFG_SIDE, CFG_T, CFG_HIDDEN, CFG_EMBED = 14, 200, 512, 64
CFG_STEPS, CFG_BATCH, CFG_LR, CFG_DROP = 5000, 128, 1e-3, 0.15
CFG_SCALES, CFG_SAMPLER_STEPS = (0, 1, 3, 8), 50
CFG_PER_DIGIT = 20
CFG_SEEDS, CFG_NOISE_SEEDS = (62, 63, 64), (72, 73, 74)     # three denoisers, a noise draw each
CFG_PIXELS = CFG_SIDE * CFG_SIDE


def cfg_init(rng):
    """The denoiser of :mod:`voxanim.diffusion` (two SiLU layers and a skip whose gain
    depends on the step) at 14 x 14, with the digit's one-hot label added into the first
    layer through ``Wc``; a row of zeros there means "told nothing"."""
    P = {}
    for k, (a, b) in enumerate(((CFG_PIXELS, CFG_HIDDEN), (CFG_HIDDEN, CFG_HIDDEN),
                                (CFG_HIDDEN, CFG_PIXELS))):
        P["W%d" % k] = rng.normal(0.0, np.sqrt(2.0 / a), (a, b))
        P["b%d" % k] = np.zeros(b)
    P["W2"] *= 0.1
    P["Wt"] = rng.normal(0.0, np.sqrt(2.0 / CFG_EMBED), (CFG_EMBED, CFG_HIDDEN))
    P["Wc"] = rng.normal(0.0, 0.5, (10, CFG_HIDDEN))
    P["Wg"], P["bg"] = np.zeros((CFG_EMBED, 1)), np.ones(1)
    return {k: v.astype(np.float32) for k, v in P.items()}


def cfg_forward(P, x, t, c):
    emb = time_embedding(t, CFG_EMBED).astype(np.float32)
    z1 = x @ P["W0"] + P["b0"] + emb @ P["Wt"] + c @ P["Wc"]
    h1 = silu(z1)
    z2 = h1 @ P["W1"] + P["b1"]
    h2 = silu(z2)
    g = emb @ P["Wg"] + P["bg"]
    return dict(x=x, emb=emb, c=c, z1=z1, h1=h1, z2=z2, h2=h2,
                eps=g * x + h2 @ P["W2"] + P["b2"])


def cfg_loss_grads(P, f, eps):
    d = 2.0 * (f["eps"] - eps) / eps.size
    g = {"W2": f["h2"].T @ d, "b2": d.sum(0)}
    dg = (d * f["x"]).sum(1, keepdims=True)
    g["Wg"], g["bg"] = f["emb"].T @ dg, dg.sum(0)
    d = (d @ P["W2"].T) * silu_grad(f["z2"])
    g["W1"], g["b1"] = f["h1"].T @ d, d.sum(0)
    d = (d @ P["W1"].T) * silu_grad(f["z1"])
    g["W0"], g["b0"] = f["x"].T @ d, d.sum(0)
    g["Wt"], g["Wc"] = f["emb"].T @ d, f["c"].T @ d
    return float(np.mean((f["eps"] - eps) ** 2)), g


def cfg_timesteps():
    return np.linspace(CFG_T, 0, CFG_SAMPLER_STEPS + 1).round().astype(int)


def cfg_guess(P, x, t, labels, scale):
    """The two noise guesses for ``x`` at step ``t`` (told nothing; told ``labels``) and
    the guided one, ``e0 + scale (ec - e0)``."""
    told = np.eye(10, dtype=np.float32)[labels]
    steps = np.full(len(x), t)
    e0 = cfg_forward(P, x, steps, np.zeros_like(told))["eps"]
    ec = cfg_forward(P, x, steps, told)["eps"]
    return e0, ec, e0 + scale * (ec - e0)


def cfg_step(x, eps, a, b):
    """One deterministic (DDIM) step from noise level ``a`` to ``b`` (values of abar): the
    clean image the guess implies, clipped to the pixel range, and the next ``x`` on the
    line from that image through ``x``."""
    x0 = np.clip((x - np.sqrt(1 - a) * eps) / np.sqrt(a), -1, 1)
    eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a)
    return x0, np.sqrt(b) * x0 + np.sqrt(1 - b) * eps


def cfg_sample(P, abar, labels, noise, scale, keep=()):
    """Samples for ``labels`` from ``noise`` at guidance ``scale``; for the rows
    ``keep``, also each step's noisy image, two guesses and clean-image estimate."""
    x = noise.copy()
    ts = cfg_timesteps()
    trace = dict(x=[], e0=[], ec=[], x0=[])
    keep = list(keep)
    for a, b in itertools.pairwise(ts):
        e0, ec, eps = cfg_guess(P, x, a, labels, scale)
        x0, nxt = cfg_step(x, eps, abar[a], abar[b])
        for key, value in (("x", x), ("e0", e0), ("ec", ec), ("x0", x0)):
            trace[key].append(value[keep].copy())
        x = nxt.astype(np.float32)
    return np.clip((x + 1) / 2, 0, 1), {k: np.array(v) for k, v in trace.items()}


def cfg_labels():
    return np.repeat(np.arange(10), CFG_PER_DIGIT)


@lru_cache(maxsize=1)
def cfg_data():
    """14 x 14 MNIST: 50,000 training images with labels and 2,000 held-out ones."""
    x, y, meta = realdata.mnist_split(50000, "train", CFG_SIDE)
    xt, yt, meta_t = realdata.mnist_split(2000, "test", CFG_SIDE)
    return x.astype(np.float32), y, xt, yt, meta["x_sha256"], meta_t["x_sha256"]


def cfg_train(seed):
    """One conditional denoiser: noise-prediction loss, each image at a random step,
    its label dropped (zeros) with probability :data:`CFG_DROP`."""
    x, y = cfg_data()[:2]
    rng = np.random.default_rng(seed)
    P = cfg_init(rng)
    opt, loss = Adam(P, CFG_LR), []
    abar = schedule(CFG_T)[2]
    signal = x * 2 - 1
    for step in range(CFG_STEPS):
        ids = rng.choice(len(x), CFG_BATCH)
        t = rng.integers(1, CFG_T + 1, CFG_BATCH)
        eps = rng.standard_normal((CFG_BATCH, CFG_PIXELS), dtype=np.float32)
        a = abar[t, None].astype(np.float32)
        told = np.eye(10, dtype=np.float32)[y[ids]]
        told[rng.random(CFG_BATCH) < CFG_DROP] = 0              # label dropout
        value, g = cfg_loss_grads(
            P, cfg_forward(P, np.sqrt(a) * signal[ids] + np.sqrt(1 - a) * eps, t, told), eps)
        opt.step(P, g, lr=CFG_LR * (0.3 if step >= 0.8 * CFG_STEPS else 1.0))
        loss.append(value)
    return P, np.array(loss, np.float32)


def guidance_arrays():
    def build():
        x, y, xt, yt, sha, sha_t = cfg_data()
        runs = parallel(cfg_train, [(seed,) for seed in CFG_SEEDS])
        abar = schedule(CFG_T)[2]
        # the judge: a separate classifier, trained on the same training images
        judge, _ = toy.fit(x.astype(float), y, widths=(128,), steps=2500, lr=0.003,
                           classification=True, seed=51)
        judged = float(np.mean(toy.forward(judge, xt)[0].argmax(1) == yt))
        labels = cfg_labels()
        shown = list(range(0, len(labels), CFG_PER_DIGIT))       # one sample per digit
        match, variety, ink, by_digit = [], [], [], []
        finals, traces = [], []
        for k, ((P, _loss), seed) in enumerate(zip(runs, CFG_NOISE_SEEDS)):
            noise = np.random.default_rng(seed).standard_normal(
                (len(labels), CFG_PIXELS), dtype=np.float32)
            rows = dict(match=[], variety=[], ink=[], by_digit=[])
            for scale in CFG_SCALES:
                out, trace = cfg_sample(P, abar, labels, noise, scale, shown if k == 0 else ())
                read = toy.forward(judge, out.astype(float))[0].argmax(1)
                rows["match"].append(np.mean(read == labels))
                rows["by_digit"].append((read == labels).reshape(10, -1).mean(1))
                rows["variety"].append(out.reshape(10, CFG_PER_DIGIT, -1).std(1).mean())
                rows["ink"].append(out.mean())
                if k == 0:
                    finals.append(out)
                    traces.append(trace)
            match.append(rows["match"])
            variety.append(rows["variety"])
            ink.append(rows["ink"])
            by_digit.append(rows["by_digit"])
        P = runs[0][0]
        return dict(
            loss=np.array([r[1] for r in runs]), abar=abar, timesteps=cfg_timesteps(),
            match=np.array(match), variety=np.array(variety), ink=np.array(ink),
            by_digit=np.array(by_digit), finals=np.array(finals),
            judge_accuracy=np.array(judged),
            real=xt[:10], real_labels=yt[:10],
            train_x_sha256=np.array(sha), test_x_sha256=np.array(sha_t),
            **{"trace_" + key: np.array([t[key] for t in traces]) for key in traces[0]},
            **{"weights_" + k: v for k, v in P.items()},
            **{"judge_weights_" + k: v for k, v in judge.items()})
    return cached("guidance", build, side=CFG_SIDE, T=CFG_T, hidden=CFG_HIDDEN,
                  steps=CFG_STEPS, batch=CFG_BATCH, lr=CFG_LR, dropout=CFG_DROP,
                  scales=list(CFG_SCALES), sampler_steps=CFG_SAMPLER_STEPS,
                  per_digit=CFG_PER_DIGIT, noise_seeds=list(CFG_NOISE_SEEDS),
                  seeds=list(CFG_SEEDS), judge_seed=51, judge_steps=2500, train_images=50000)


def classifier_free_guidance():
    d = guidance_arrays()
    shown = list(range(0, 10 * CFG_PER_DIGIT, CFG_PER_DIGIT))
    side = (CFG_SIDE, CFG_SIDE)
    per = 10 * CFG_PER_DIGIT
    return result(
        "classifier-free-guidance",
        "Conditional diffusion on 14x14 MNIST with label dropout; a measured guidance sweep",
        [
            curve("Learning with missing labels",
                  "15% of each batch has its label dropped. 5,000 Adam updates; three "
                  "seeds.", d["loss"], ["seed %d" % s for s in CFG_SEEDS]),
            images("Guidance 0 and 1", "One sample per digit 0-9; every row starts from the "
                   "same noise.", d["finals"][:2][:, shown].reshape(-1, *side),
                   ["w0"] * 10 + ["w1"] * 10),
            images("Guidance 3 and 8", "The same noise and digits at stronger guidance.",
                   d["finals"][2:][:, shown].reshape(-1, *side), ["w3"] * 10 + ["w8"] * 10),
            curve("Read as the digit asked for, and variety within a digit",
                  "Means over three separately trained denoisers, %d samples each per "
                  "scale; a separately trained classifier does the reading." % per,
                  [d["match"].mean(0), d["variety"].mean(0)],
                  ["class match", "pixel spread within a digit"], np.array(CFG_SCALES),
                  ("class match per denoiser: " + "; ".join(
                      "w%d %s" % (w, "/".join("%.3f" % v for v in d["match"][:, k]))
                      for k, w in enumerate(CFG_SCALES)),
                   "the classifier reads %.1f%% of 2,000 held-out digits correctly" % (
                       100 * d["judge_accuracy"]),
                   "mean ink per pixel: " + ", ".join(
                       "w%d %.3f" % (w, v) for w, v in zip(CFG_SCALES, d["ink"].mean(0))))),
        ],
        seeds=list(CFG_SEEDS), noise_seeds=list(CFG_NOISE_SEEDS), updates=CFG_STEPS,
        **provenance("guidance", d),
        class_match=d["match"].tolist(), variety=d["variety"].tolist(),
        judge_accuracy=float(d["judge_accuracy"]),
        scope="three trained denoisers, 200 samples each per scale (the pictures are seed "
              "62's); variety is the pixel standard deviation among samples asked for the "
              "same digit",
        designs_tried=[
            "the first sampler kept the guided noise guess after clipping the clean-image "
            "estimate, so strong guidance inflated the noise put back at every step and "
            "samples at 3 and 8 dissolved (class match 0.39-0.46 and 0.05-0.08 at 5,000 "
            "updates); fixed to re-derive the noise from the clipped estimate, the "
            "standard clipped DDIM step",
            "500 and 5,000 updates of the same model; 5,000 kept",
            "first run as one denoiser sampled from three noise draws (class match "
            "0.10/0.71/0.99/0.98 at guidance 0/1/3/8); rerun as three denoisers",
        ])


# ====================================================================== 66 autoregressive-pixels
PIX_SIDE, PIX_INK = 14, 0.4            # 14 x 14 images; a pixel is inked above this grey level
PIX_DEEP, PIX_ONE = (1, 1, 2, 1, 2, 1), (1,)      # each layer's dilation; one entry per layer
PIX_CHANNELS, PIX_STEPS, PIX_BATCH, PIX_LR = 32, 2400, 32, 0.003
PIX_SEEDS = (66, 67, 68)
PIX_PAD = 2
# What each stream may read, as (row, column) offsets. The first layer (type A) must not
# read the pixel itself; later layers (type B) may read their own position, which by then
# holds only earlier pixels.
ROWS_ABOVE_A = ((-1, -1), (-1, 0), (-1, 1))
ROWS_ABOVE_B = ROWS_ABOVE_A + ((0, -1), (0, 0), (0, 1))
ROW_LEFT_A = ((0, -1),)
ROW_LEFT_B = ((0, -1), (0, 0))
PLAIN_A = ROWS_ABOVE_A + ((0, -1),)    # the single-stream masks of the first PixelCNN
PLAIN_B = PLAIN_A + ((0, 0),)


def pix_taps(layer, dilations):
    if layer == 0:
        return ROWS_ABOVE_A, ROW_LEFT_A
    k = dilations[layer]
    return (tuple((a * k, b * k) for a, b in ROWS_ABOVE_B),
            tuple((a * k, b * k) for a, b in ROW_LEFT_B))


def pix_shifted(x, offsets):
    """``x`` ``(n, h, w, c)`` read at each offset, side by side: ``(n h w, taps c)``."""
    n, h, w, c = x.shape
    pad = np.zeros((n, h + 2 * PIX_PAD, w + 2 * PIX_PAD, c), x.dtype)
    pad[:, PIX_PAD:-PIX_PAD, PIX_PAD:-PIX_PAD] = x
    return np.concatenate([pad[:, PIX_PAD + dr:PIX_PAD + dr + h, PIX_PAD + dc:PIX_PAD + dc + w]
                           for dr, dc in offsets], -1).reshape(n * h * w, -1)


def pix_unshift(d, offsets, shape):
    """The transpose of :func:`pix_shifted`: gradients handed back to where they were read."""
    n, h, w, c = shape
    d = d.reshape(n, h, w, -1)
    pad = np.zeros((n, h + 2 * PIX_PAD, w + 2 * PIX_PAD, c), d.dtype)
    for k, (dr, dc) in enumerate(offsets):
        pad[:, PIX_PAD + dr:PIX_PAD + dr + h, PIX_PAD + dc:PIX_PAD + dc + w] += \
            d[..., k * c:(k + 1) * c]
    return pad[:, PIX_PAD:-PIX_PAD, PIX_PAD:-PIX_PAD].reshape(n * h * w, c)


def pix_init(rng, dilations, channels=PIX_CHANNELS):
    p = {}
    for layer in range(len(dilations)):
        cin = 1 if layer == 0 else channels
        above, left = pix_taps(layer, dilations)
        p["V%d" % layer] = rng.normal(0, np.sqrt(2 / (len(above) * cin)),
                                      (len(above) * cin, channels))
        p["H%d" % layer] = rng.normal(0, np.sqrt(1 / (len(left) * cin)),
                                      (len(left) * cin, channels))
        p["U%d" % layer] = rng.normal(0, np.sqrt(1 / channels), (channels, channels))
        p["bv%d" % layer], p["bh%d" % layer] = np.zeros(channels), np.zeros(channels)
    p["O"] = rng.normal(0, np.sqrt(1 / channels), (channels, 1))
    p["c"] = np.zeros(1)
    return {k: v.astype(np.float32) for k, v in p.items()}


def pix_forward(p, x, dilations):
    """Logits of ink for every pixel of ``x`` ``(n, side, side, 1)``, each from earlier
    pixels only. Two streams, as in the gated PixelCNN: ``v`` reads the rows above, ``h``
    reads its own row to the left and takes in ``v`` at its own position, so together
    they reach everything above and to the left: no blind spot."""
    n, hh, ww, _ = x.shape
    tape = []
    h = v = x.reshape(-1, 1)
    for layer in range(len(dilations)):
        above, left = pix_taps(layer, dilations)
        shape = (n, hh, ww, h.shape[1])
        cv = pix_shifted(v.reshape(shape), above)
        zv = cv @ p["V%d" % layer] + p["bv%d" % layer]
        v2 = np.maximum(zv, 0)
        ch = pix_shifted(h.reshape(shape), left)
        zh = ch @ p["H%d" % layer] + v2 @ p["U%d" % layer] + p["bh%d" % layer]
        h2 = np.maximum(zh, 0)
        if layer > 0:
            h2 = h2 + h                                           # residual
        tape.append((cv, zv, v2, ch, zh))
        v, h = v2, h2
    return (h @ p["O"] + p["c"]).reshape(n, hh, ww), (tape, h)


def pix_loss_grads(p, x, dilations):
    n, hh, ww, _ = x.shape
    logits, (tape, h) = pix_forward(p, x, dilations)
    target = x[..., 0]
    loss = float(np.mean(np.logaddexp(0, logits) - target * logits))
    d = ((sigmoid(logits) - target) / target.size).reshape(-1, 1).astype(h.dtype)
    g = {"O": h.T @ d, "c": d.sum(0)}
    dh, dv = d @ p["O"].T, 0.0
    channels = dh.shape[1]
    for layer in reversed(range(len(dilations))):
        above, left = pix_taps(layer, dilations)
        cv, zv, v2, ch, zh = tape[layer]
        dz = dh * (zh > 0)
        g["bh%d" % layer], g["H%d" % layer], g["U%d" % layer] = dz.sum(0), ch.T @ dz, v2.T @ dz
        dzv = (dv + dz @ p["U%d" % layer].T) * (zv > 0)
        g["bv%d" % layer], g["V%d" % layer] = dzv.sum(0), cv.T @ dzv
        if layer > 0:
            shape = (n, hh, ww, channels)
            dh = dh + pix_unshift(dz @ p["H%d" % layer].T, left, shape)
            dv = pix_unshift(dzv @ p["V%d" % layer].T, above, shape)
    return loss, g


def pix_bits(p, x, dilations):
    """Mean bits per pixel of ``x`` under the model."""
    total = 0.0
    for i in range(0, len(x), 500):
        part = x[i:i + 500]
        logits = pix_forward(p, part, dilations)[0]
        total += float(np.sum(np.logaddexp(0, logits) - part[..., 0] * logits))
    return total / x.size / np.log(2)


def pix_sample(p, dilations, draws):
    """Images sampled pixel by pixel in reading order from the uniform ``draws``
    ``(n, side, side)``: ink where the draw is below the model's chance. Returns the
    images and the chance each pixel was drawn with."""
    n, side, _ = draws.shape
    x = np.zeros((n, side, side, 1), np.float32)
    chance = np.zeros((n, side, side))
    for r in range(side):
        for c in range(side):
            chance[:, r, c] = sigmoid(pix_forward(p, x, dilations)[0][:, r, c])
            x[:, r, c, 0] = draws[:, r, c] < chance[:, r, c]
    return x[..., 0], chance


@lru_cache(maxsize=1)
def pix_data():
    """Binarised 14 x 14 MNIST: 50,000 training and 2,000 held-out images."""
    x, _y, meta = realdata.mnist_split(50000, "train", PIX_SIDE)
    xt, _yt, meta_t = realdata.mnist_split(2000, "test", PIX_SIDE)

    def binary(a):
        return (a > PIX_INK).astype(np.float32).reshape(-1, PIX_SIDE, PIX_SIDE, 1)
    return binary(x), binary(xt), meta["x_sha256"], meta_t["x_sha256"]


def pix_run(dilations, seed):
    x, held, _a, _b = pix_data()
    rng = np.random.default_rng(seed)
    p = pix_init(rng, dilations)
    opt, loss = Adam(p, PIX_LR), []
    for step in range(PIX_STEPS):
        value, g = pix_loss_grads(p, x[rng.choice(len(x), PIX_BATCH)], dilations)
        opt.step(p, g, lr=PIX_LR * (0.3 if step >= 0.75 * PIX_STEPS else 1.0))
        loss.append(value / np.log(2))
    return p, np.array(loss, np.float32), pix_bits(p, x[:2000], dilations), \
        pix_bits(p, held, dilations)


def pix_seen(dilations, side=PIX_SIDE):
    """Which pixels each pixel's prediction can depend on in the two-stream model:
    ``seen[r, c]`` is a boolean ``(side, side)`` map. Exact: the network with every
    weight one and no bias sums, over all paths, the inputs it can reach."""
    ones = {k: np.ones_like(v) if k[0] in "VHUO" else np.zeros_like(v)
            for k, v in pix_init(np.random.default_rng(0), dilations, channels=2).items()}
    impulses = np.eye(side * side, dtype=np.float32).reshape(-1, side, side, 1)
    out = pix_forward(ones, impulses, dilations)[0]              # (source, r, c)
    return (out > 0).transpose(1, 2, 0).reshape(side, side, side, side)


def plain_seen(dilations, side=PIX_SIDE):
    """The same for a single stack of the first PixelCNN's masks (type A, then type B)
    with the same depth and dilations: the offsets reachable by one step per layer."""
    reach = {(0, 0)}
    for layer, k in enumerate(dilations):
        steps = PLAIN_A if layer == 0 else tuple((a * k, b * k) for a, b in PLAIN_B)
        reach = {(a + dr, b + dc) for a, b in reach for dr, dc in steps}
    seen = np.zeros((side, side, side, side), bool)
    for r in range(side):
        for c in range(side):
            for dr, dc in reach:
                if 0 <= r + dr < side and 0 <= c + dc < side:
                    seen[r, c, r + dr, c + dc] = True
    return seen


def pixel_arrays():
    def build():
        x, held, sha, sha_t = pix_data()
        runs = parallel(pix_run, [(d, s) for d in (PIX_DEEP, PIX_ONE) for s in PIX_SEEDS])
        n = len(PIX_SEEDS)
        p = runs[0][0]
        q = np.clip(x.mean(0), 1e-3, 1 - 1e-3)                    # each pixel on its own
        independent = float(np.mean(-(held * np.log2(q) + (1 - held) * np.log2(1 - q))))
        draws = np.random.default_rng(660).random((16, PIX_SIDE, PIX_SIDE))
        samples, chances = pix_sample(p, PIX_DEEP, draws)
        shallow, _ = pix_sample(runs[n][0], PIX_ONE, draws)
        return dict(
            loss=np.array([r[1] for r in runs[:n]]), shallow_loss=np.array(runs[n][1]),
            train_bits=np.array([[r[2] for r in runs[:n]], [r[2] for r in runs[n:]]]),
            held_bits=np.array([[r[3] for r in runs[:n]], [r[3] for r in runs[n:]]]),
            independent=np.array(independent), pixel_rate=q[..., 0],
            samples=samples, chances=chances, draws=draws, shallow_samples=shallow,
            real=held[:16, ..., 0],
            train_x_sha256=np.array(sha), test_x_sha256=np.array(sha_t),
            **{"weights_" + k: v for k, v in p.items()},
            **{"shallow_weights_" + k: v for k, v in runs[n][0].items()})
    return cached("pixels", build, side=PIX_SIDE, ink=PIX_INK, deep=list(PIX_DEEP),
                  one=list(PIX_ONE), channels=PIX_CHANNELS, steps=PIX_STEPS, batch=PIX_BATCH,
                  lr=PIX_LR, seeds=list(PIX_SEEDS), sample_seed=660, train_images=50000)


PIX_TARGET = (9, 3)                    # the pixel whose view the study's first panel maps


def autoregressive_pixels():
    d = pixel_arrays()
    r, c = PIX_TARGET
    two, plain = pix_seen(PIX_DEEP)[r, c], plain_seen(PIX_DEEP)[r, c]
    held = d["held_bits"]
    return result(
        "autoregressive-pixels",
        "A six-layer two-stream masked-convolution model of binarised 14x14 MNIST",
        [
            matrix("What one pixel's prediction can read",
                   "Pixel (row 9, column 3). 2: read by the two-stream model and by a "
                   "single masked stack; 1: read by the two-stream model only (the single "
                   "stack's blind spot); 0: not read.",
                   two.astype(int) + (two & plain)),
            curve("Training loss, bits per pixel",
                  "Three seeds of the six-layer model; every pixel is trained in one pass.",
                  d["loss"], ["seed %d" % s for s in PIX_SEEDS]),
            images("Eight samples, drawn pixel by pixel", "Seed 66's model; 196 draws each.",
                   d["samples"][:8], ["sample %d" % (i + 1) for i in range(8)]),
            bars("Held-out bits per pixel",
                 "2,000 held-out digits. Lower is better; means of three seeds.",
                 [float(d["independent"]), held[1].mean(), held[0].mean()],
                 ["each pixel on its own", "one masked layer", "six masked layers"],
                 ("six layers per seed: " + ", ".join("%.4f" % v for v in held[0]),
                  "one layer per seed: " + ", ".join("%.4f" % v for v in held[1]))),
        ],
        seeds=list(PIX_SEEDS), updates=PIX_STEPS,
        **provenance("pixels", d),
        heldout_bits=dict(independent=float(d["independent"]), one_layer=held[1].tolist(),
                          six_layers=held[0].tolist()),
        scope="14x14 MNIST binarised at grey level 0.4; 32 channels; the blind spot is exact "
              "arithmetic on the masks, not a trained single-stack model",
        designs_tried=[
            "a single masked stack and a two-stream model, 5 layers x 24 channels, 1,500 "
            "updates, one seed (held-out 0.258 and 0.245 bits): both sampled stroke "
            "fragments, their five rows of view too short for a digit",
            "two streams, 6 layers with dilations 1,1,2,1,2,1: 24 channels x 2,000 updates "
            "(0.220 bits) and 32 channels x 3,000 updates (0.211 bits), one seed each; 32 "
            "channels x 2,400 updates x 3 seeds kept to fit the build-time limit",
            "the single-stack model was not trained again for a three-seed comparison "
            "(time limit), so the blind spot is shown as exact arithmetic on the masks",
        ])


# ====================================================================== 65 vq-vae-codebook
VQ_CODES, VQ_LATENT, VQ_HIDDEN, VQ_PATCH = 64, 64, 32, 4
VQ_STEPS, VQ_BATCH, VQ_LR, VQ_COMMIT = 3000, 32, 0.003, 0.25
VQ_DECAY, VQ_RESTART_EVERY, VQ_DEAD = 0.99, 100, 0.5
VQ_SEEDS, VQ_EVERY = (65, 66, 67), 50
VQ_GRID = 28 // VQ_PATCH               # 7 x 7 codes per digit


def vq_patches(x):
    """Images ``(n, 784)`` cut into their 49 patches of 4 x 4 pixels: ``(n, 49, 16)``."""
    n = len(x)
    return x.reshape(n, VQ_GRID, VQ_PATCH, VQ_GRID, VQ_PATCH).transpose(0, 1, 3, 2, 4).reshape(
        n, VQ_GRID * VQ_GRID, VQ_PATCH * VQ_PATCH)


def vq_unpatch(p):
    """The inverse of :func:`vq_patches`: ``(n, 28, 28)`` images."""
    n = len(p)
    return p.reshape(n, VQ_GRID, VQ_GRID, VQ_PATCH, VQ_PATCH).transpose(0, 1, 3, 2, 4).reshape(
        n, 28, 28)


def vq_nearest(h, codes):
    """Index of each row's nearest code (Euclidean)."""
    return ((h ** 2).sum(1)[:, None] - 2 * h @ codes.T + (codes ** 2).sum(1)[None]).argmin(1)


@lru_cache(maxsize=1)
def vq_data():
    """28 x 28 MNIST as patches: 20,000 training digits and 1,000 held-out ones."""
    x, _y, meta = realdata.mnist_split(20000, "train", 28)
    xt, _yt, meta_t = realdata.mnist_split(1000, "test", 28)
    return vq_patches(x), vq_patches(xt), meta["x_sha256"], meta_t["x_sha256"]


def vq_split(p):
    return ({k[1:]: v for k, v in p.items() if k[0] == "e"},
            {k[1:]: v for k, v in p.items() if k[0] == "d"})


def vq_run(remedies, seed):
    """A vector-quantised autoencoder of 4 x 4 patches: an encoder (16 -> 32 -> 64
    numbers), the nearest of 64 codes, a decoder (64 -> 32 -> 16 pixels), the
    straight-through gradient and a commitment loss.

    With ``remedies``: the codes start on the encoder's outputs for 64 random training
    patches, follow the outputs that choose them as a running mean (EMA), and a code
    chosen by fewer than half a patch per batch on average is restarted on a random
    output every 100 updates. Without: the codes start as standard normal draws and
    learn only from the codebook loss, by Adam, like the other weights.

    Returns the weights, the codebook, the loss per update, and per checkpoint the
    number of codes chosen by any patch of the first 64 held-out digits."""
    train, held, _a, _b = vq_data()
    rng = np.random.default_rng(seed)
    enc = toy.init((VQ_PATCH ** 2, VQ_HIDDEN, VQ_LATENT), rng)
    dec = toy.init((VQ_LATENT, VQ_HIDDEN, VQ_PATCH ** 2), rng)
    flat = train.reshape(-1, VQ_PATCH ** 2)
    watch = held[:64].reshape(-1, VQ_PATCH ** 2)
    if remedies:
        codes = toy.forward(enc, flat[rng.choice(len(flat), VQ_CODES, replace=False)])[0].copy()
    else:
        codes = rng.normal(0.0, 1.0, (VQ_CODES, VQ_LATENT))
    p = {**{"e" + k: v for k, v in enc.items()}, **{"d" + k: v for k, v in dec.items()}}
    if not remedies:
        p["C"] = codes
    opt = Adam(p, VQ_LR)
    count, total = np.ones(VQ_CODES), codes.copy()
    loss, in_use, first = [], [], None
    for step in range(VQ_STEPS + 1):
        enc, dec = vq_split(p)
        codes = codes if remedies else p["C"]
        if step % VQ_EVERY == 0:
            in_use.append(len(np.unique(vq_nearest(toy.forward(enc, watch)[0], codes))))
        if step == 0:
            first = codes.copy(), {k: v.copy() for k, v in enc.items()}
        if step == VQ_STEPS:
            break
        x = train[rng.choice(len(train), VQ_BATCH)].reshape(-1, VQ_PATCH ** 2)
        h, ea = toy.forward(enc, x)
        ids = vq_nearest(h, codes)
        z = codes[ids]
        out, da = toy.forward(dec, z)
        gd, dz = toy.backward(dec, da, (sigmoid(out) - x) / x.size)
        # straight-through: the decoder's slope at the code, handed to the encoder's
        # output unchanged, plus the commitment pull towards the chosen code
        ge, _ = toy.backward(enc, ea, dz + VQ_COMMIT * 2 * (h - z) / h.size)
        g = {**{"e" + k: v for k, v in ge.items()}, **{"d" + k: v for k, v in gd.items()}}
        if remedies:
            opt.step(p, g)
            chosen = np.zeros((len(h), VQ_CODES))
            chosen[np.arange(len(h)), ids] = 1
            count = VQ_DECAY * count + (1 - VQ_DECAY) * chosen.sum(0)
            total = VQ_DECAY * total + (1 - VQ_DECAY) * (chosen.T @ h)
            codes = total / np.maximum(count, 1e-5)[:, None]
            if (step + 1) % VQ_RESTART_EVERY == 0:
                dead = np.flatnonzero(count < VQ_DEAD)
                if len(dead):
                    codes[dead] = h[rng.choice(len(h), len(dead), replace=False)]
                    count[dead], total[dead] = 1.0, codes[dead]
        else:
            gc = np.zeros_like(codes)
            np.add.at(gc, ids, 2 * (z - h) / h.size)              # the codebook loss
            g["C"] = gc
            opt.step(p, g)
        loss.append(float(np.mean(np.logaddexp(0, out) - x * out)))
    return p, codes, np.array(loss, np.float32), np.array(in_use), first[0], first[1]


def vq_apply(enc, dec, codes, patches):
    """Patches ``(m, 16)`` through the trained model: encoder outputs, chosen code
    indices and the decoder's outputs."""
    h = toy.forward(enc, patches)[0]
    ids = vq_nearest(h, codes)
    return h, ids, sigmoid(toy.forward(dec, codes[ids])[0])


def vq_arrays():
    def build():
        _train, held, sha, sha_t = vq_data()
        flat = held.reshape(-1, VQ_PATCH ** 2)
        runs = parallel(vq_run, [(r, s) for r in (True, False) for s in VQ_SEEDS])
        n = len(VQ_SEEDS)
        out = dict(original=vq_unpatch(held[:12]), blank_share=np.array(np.mean(
            flat.max(1) < 0.05)), steps=np.arange(0, VQ_STEPS + 1, VQ_EVERY),
            train_x_sha256=np.array(sha), test_x_sha256=np.array(sha_t))
        counts, mse, in_use = [], [], []
        for k, (p, codes, loss, used, codes0, enc0) in enumerate(runs):
            enc, dec = vq_split(p)
            _h, ids, recon = vq_apply(enc, dec, codes, flat)
            counts.append(np.bincount(ids, minlength=VQ_CODES))
            mse.append(np.mean((recon - flat) ** 2))
            in_use.append(used)
            if k % n == 0:                                        # the first seed of each arm
                arm = "remedies_" if k == 0 else "control_"
                some = flat[:64 * VQ_GRID ** 2]                   # the 64 watched digits
                out.update({
                    arm + "recon": vq_unpatch(recon.reshape(len(held), -1, VQ_PATCH ** 2)[:12]),
                    arm + "ids": ids.reshape(len(held), VQ_GRID, VQ_GRID)[:12],
                    arm + "codes": codes, arm + "loss": loss,
                    arm + "tiles": sigmoid(toy.forward(dec, codes)[0]),
                    arm + "start_chosen": np.bincount(
                        vq_nearest(toy.forward(enc0, some)[0], codes0), minlength=VQ_CODES),
                    **{arm + "weights_" + key: v for key, v in p.items() if key != "C"}})
        counts = np.array(counts).reshape(2, n, VQ_CODES)
        share = counts / counts.sum(-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy = -np.where(share > 0, share * np.log(share), 0.0).sum(-1)
        out.update(counts=counts, mse=np.array(mse).reshape(2, n),
                   in_use=np.array(in_use).reshape(2, n, -1), perplexity=np.exp(entropy))
        return out
    return cached("vq", build, codes=VQ_CODES, latent=VQ_LATENT, hidden=VQ_HIDDEN,
                  patch=VQ_PATCH, steps=VQ_STEPS, batch=VQ_BATCH, lr=VQ_LR, commit=VQ_COMMIT,
                  decay=VQ_DECAY, restart_every=VQ_RESTART_EVERY, dead=VQ_DEAD,
                  seeds=list(VQ_SEEDS), every=VQ_EVERY, train_images=20000, held_out=1000)


def vq_vae():
    d = vq_arrays()
    used = (d["counts"] > 0).sum(-1)
    labels = ["seed %d" % s for s in VQ_SEEDS]
    return result(
        "vq-vae-codebook",
        "A 7x7 grid of codes per 28x28 MNIST digit: 64 codes with and without the remedies",
        [
            images("Reconstruction from 49 code numbers",
                   "Held-out digits; the same digits rebuilt with the remedies and without.",
                   np.concatenate((d["original"][:4], d["remedies_recon"][:4],
                                   d["control_recon"][:4])),
                   ["input"] * 4 + ["remedies"] * 4 + ["control"] * 4),
            curve("Codes in use through training",
                  "Codes chosen by any patch of 64 held-out digits. Three seeds with the "
                  "remedies, then three without.", d["in_use"].reshape(-1, d["in_use"].shape[-1]),
                  ["remedies " + s for s in labels] + ["control " + s for s in labels],
                  d["steps"]),
            bars("Patches choosing each code, with the remedies",
                 "49,000 patches of 1,000 held-out digits, seed 65. The tallest bar is the "
                 "blank patch.", d["counts"][0, 0], [str(i) for i in range(VQ_CODES)],
                 ("%.0f%% of held-out patches are blank" % (100 * d["blank_share"]),)),
            matrix("Codes used, of 64",
                   "Rows: remedies, control. Columns: seeds 65-67. Held-out patches.", used,
                   ("held-out squared error per pixel: remedies %s, control %s" % (
                       ", ".join("%.4f" % v for v in d["mse"][0]),
                       ", ".join("%.4f" % v for v in d["mse"][1])),
                    "usage perplexity: remedies %s, control %s" % (
                        ", ".join("%.1f" % v for v in d["perplexity"][0]),
                        ", ".join("%.1f" % v for v in d["perplexity"][1])))),
        ],
        seeds=list(VQ_SEEDS), updates=VQ_STEPS, codes_used=used.tolist(),
        **provenance("vq", d),
        heldout_mse=d["mse"].tolist(), perplexity=d["perplexity"].tolist(),
        scope="patch-wise encoder and decoder (no overlap between patches); the three "
              "remedies are applied together and not separated; the control's codes "
              "start as standard normal draws in 64 dimensions",
        designs_tried=[
            "the same model with a 2-number code and 32 codes: no collapse without the "
            "remedies (codes used 31-32 of 32 from normal or uniform starts, even with the "
            "encoder's outputs started 20 times smaller); 256 codes of 2 numbers: 252-256 "
            "used without, 244-251 with",
            "8- and 16-number codes, 32 codes: without the remedies 29-31 and 19-28 used "
            "from a standard normal start, 30-32 and 32 from a uniform(-1/32, 1/32) start; "
            "with them 32",
            "64-number codes and 64 codes (the size of code real VQ-VAEs use), standard "
            "normal start: kept. The collapse belongs to codes started far from the "
            "encoder's outputs in many dimensions; it was looked for, and is reported as "
            "that, not as what any VQ-VAE does",
        ])


STUDIES = {
    "gan-training": gan_training,
    "normalizing-flows": normalizing_flows,
    "classifier-free-guidance": classifier_free_guidance,
    "vq-vae-codebook": vq_vae,
    "autoregressive-pixels": autoregressive_pixels,
}

ARRAYS = {
    "gan-training": gan_arrays,
    "normalizing-flows": flow_arrays,
    "classifier-free-guidance": guidance_arrays,
    "vq-vae-codebook": vq_arrays,
    "autoregressive-pixels": pixel_arrays,
}


@cache
def arrays(name):
    """Everything study ``name`` saved (its samples, weights and data): the panels show
    only part of it."""
    return ARRAYS[name]()
