"""From an autoencoder to a GAN with the same blocks, for the AE-to-GAN animations of "Blocks
and code" (catalogue 265-268): ``examples/ae_gan.py`` in numpy.

* The autoencoder: an encoder ``784 -> 256 (ReLU) -> 16`` and a decoder ``16 -> 256 (ReLU)
  -> 784 (sigmoid)``, trained together for 3,000 steps of 64 MNIST digits (pixels 0..1) with
  binary cross-entropy, pixel by pixel (Adam, lr 0.001).
* Rewired as a GAN: the generator *is* the decoder, its learned weights kept; the judge is the
  encoder's first layer and ReLU, its weights kept, and a new last layer ``256 -> 1``. Made-up
  codes are noise with the mean and spread of the encoder's codes of 5,000 training digits.
  The GAN game: 6,000 steps of 64 (Adam, lr 0.0002, betas 0.5 and 0.999): the judge on real
  and made digits, then the generator to make the judge say "real".
* Measured, all on held-out data or fresh noise: how grey the pictures are (the share of
  pixels between 0.2 and 0.8: blur shows as grey; real digits are mostly ink or paper), for
  real test digits, the autoencoder's rebuilds of them, and the decoder fed the same made-up
  codes before and after the GAN game; what the course's MNIST classifier
  (:mod:`voxanim.study_course`) reads them as, and how sure it is; how far rebuilds are from
  their digits; and both training curves.

One run each (one seed); cached. ``tests/test_aegan_equivalence.py`` matches the script's lines
in torch from this module's starting weights: one autoencoder step and one GAN step.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import sys

import numpy as np

from . import study_course as S
from .cache import cached_npz

VERSION = 1
CODE = 16
HIDDEN = 256
BATCH = 64
AE_STEPS, AE_LR = 3000, 1e-3
GAN_STEPS, GAN_LR, BETAS = 6000, 2e-4, (0.5, 0.999)
SNAPS = (0, 500, 2000, 6000)
SAMPLES = 1000
GREY = (0.2, 0.8)
SEED = 265
ENC_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
DEC_KEYS = ENC_KEYS
JUDGE_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")


def _linear(rng, o, i):
    b = 1 / np.sqrt(i)
    return (rng.uniform(-b, b, (o, i)).astype(np.float32),
            rng.uniform(-b, b, o).astype(np.float32))


def init(seed=0):
    """``(encoder, decoder)`` as PyTorch starts them, keyed as their ``state_dict()``."""
    rng = np.random.default_rng([SEED, 0, int(seed)])
    enc, dec = {}, {}
    enc["0.weight"], enc["0.bias"] = _linear(rng, HIDDEN, 784)
    enc["2.weight"], enc["2.bias"] = _linear(rng, CODE, HIDDEN)
    dec["0.weight"], dec["0.bias"] = _linear(rng, HIDDEN, CODE)
    dec["2.weight"], dec["2.bias"] = _linear(rng, 784, HIDDEN)
    return enc, dec


def judge_head(seed=0):
    """The judge's new last layer, ``nn.Linear(256, 1)``, as PyTorch starts it."""
    rng = np.random.default_rng([SEED, 1, int(seed)])
    w, b = _linear(rng, 1, HIDDEN)
    return {"2.weight": w, "2.bias": b}


def mlp(P, x):
    """``Linear, ReLU, Linear``: ``(out, hidden, pre)``."""
    pre = x @ P["0.weight"].T + P["0.bias"]
    h = np.maximum(pre, 0)
    return h @ P["2.weight"].T + P["2.bias"], h, pre


def mlp_back(P, x, h, pre, dout):
    G = {"2.weight": dout.T @ h, "2.bias": dout.sum(0)}
    dh = (dout @ P["2.weight"]) * (pre > 0)
    G["0.weight"], G["0.bias"] = dh.T @ x, dh.sum(0)
    return G, dh @ P["0.weight"]


def sig(z):
    with np.errstate(over="ignore"):              # exp(-z) of a very negative z: 1 / inf = 0
        return (1 / (1 + np.exp(-z))).astype(z.dtype)


def decode(dec, z):
    return sig(mlp(dec, z)[0])


def encode(enc, x):
    return mlp(enc, x)[0]


def pixels():
    """``(train, test)`` digits as ``ToTensor`` gives them, ``(N, 784)`` in 0..1."""
    return S.pixels()


def ae_step_grads(enc, dec, x):
    """``F.binary_cross_entropy(decoder(encoder(x)), x)`` (mean) and both blocks' gradients."""
    code, h1, p1 = mlp(enc, x)
    logits, h2, p2 = mlp(dec, code)
    back = sig(logits)
    eps = 1e-12
    loss = float(-(x * np.log(np.maximum(back, eps)) + (1 - x) * np.log(np.maximum(1 - back, eps)))
                 .mean())
    dlogits = ((back - x) / x.size).astype(np.float32)
    Gd, dcode = mlp_back(dec, code, h2, p2, dlogits)
    Ge, _ = mlp_back(enc, x, h1, p1, dcode)
    cast = lambda G, P: {k: G[k].astype(P[k].dtype) for k in P}   # noqa: E731
    return loss, cast(Ge, enc), cast(Gd, dec)


def _bce_logits(z, y):
    return float((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean())


def gan_step(gen, judge, x, z):
    """One step of region "ag-judge" then "ag-generator": the losses and both gradients
    (the judge's from ``lj``, the generator's from ``lg`` with the judge before its update,
    as the script computes ``lg`` after ``oj.step()``: so this returns the judge's gradients
    only; :func:`gen_grads` gives the generator's with the judge as updated)."""
    fake = decode(gen, z)
    sr, hr, pr = mlp(judge, x)
    sf, hf, pf = mlp(judge, fake)
    n = len(x)
    lj = _bce_logits(sr, np.ones_like(sr)) + _bce_logits(sf, np.zeros_like(sf))
    Gr, _ = mlp_back(judge, x, hr, pr, ((sig(sr) - 1) / n).astype(np.float32))
    Gf, _ = mlp_back(judge, fake, hf, pf, (sig(sf) / n).astype(np.float32))
    Gj = {k: (Gr[k] + Gf[k]).astype(judge[k].dtype) for k in judge}
    return lj, Gj


def gen_grads(gen, judge, z):
    """``lg = bce(judge(generator(z)), ones)`` and the generator's gradients."""
    logits, h, p = mlp(gen, z)
    fake = sig(logits)
    s, hj, pj = mlp(judge, fake)
    lg = _bce_logits(s, np.ones_like(s))
    _Gj, dfake = mlp_back(judge, fake, hj, pj, ((sig(s) - 1) / len(z)).astype(np.float32))
    G, _ = mlp_back(gen, z, h, p, (dfake * fake * (1 - fake)).astype(np.float32))
    return lg, {k: G[k].astype(gen[k].dtype) for k in gen}


def ae_batch(step):
    x = pixels()[0]
    rng = np.random.default_rng([SEED, 2, int(step)])
    return x[rng.integers(0, len(x), BATCH)]


def gan_batch(step, mean, spread):
    x = pixels()[0]
    rng = np.random.default_rng([SEED, 3, int(step)])
    z = (mean + spread * rng.standard_normal((BATCH, CODE))).astype(np.float32)
    return x[rng.integers(0, len(x), BATCH)], z


def noise(mean, spread, n=SAMPLES, stream=4):
    rng = np.random.default_rng([SEED, stream])
    return (mean + spread * rng.standard_normal((n, CODE))).astype(np.float32)


def grey(img):
    """The share of pixels between 0.2 and 0.8: blur shows as grey."""
    img = np.asarray(img)
    return float(((img > GREY[0]) & (img < GREY[1])).mean())


def read(img):
    """The course classifier (study_course's trained MLP) on pictures 0..1: ``(labels, sure)``."""
    logits = S.forward(S.params(S.EPOCH_STEPS), S.normalise(img))["logits"]
    p = S._softmax(logits)
    return p.argmax(1), p.max(1)


def _build():
    enc, dec = init()
    out = {"enc0__" + k: v for k, v in enc.items()}
    out.update({"dec0__" + k: v for k, v in dec.items()})
    st_e, st_d = S.adam_start(enc), S.adam_start(dec)
    ae_loss = []
    for step in range(AE_STEPS):
        x = ae_batch(step)
        loss, Ge, Gd = ae_step_grads(enc, dec, x)
        if step == 0:
            out.update({"aeG0e__" + k: v for k, v in Ge.items()})
            out.update({"aeG0d__" + k: v for k, v in Gd.items()})
        ae_loss.append(loss)
        enc, st_e = S.adam_update(enc, Ge, st_e, lr=AE_LR)
        dec, st_d = S.adam_update(dec, Gd, st_d, lr=AE_LR)
    out["ae_loss"] = np.array(ae_loss)
    out.update({"enc__" + k: v for k, v in enc.items()})
    out.update({"dec__" + k: v for k, v in dec.items()})
    codes = encode(enc, pixels()[0][:5000])
    mean, spread = codes.mean(0), codes.std(0, ddof=1)
    out.update(mean=mean, spread=spread)
    # the rewiring: the generator is the decoder; the judge the encoder's body and a new head
    gen = {k: v.copy() for k, v in dec.items()}
    judge = {"0.weight": enc["0.weight"].copy(), "0.bias": enc["0.bias"].copy(), **judge_head()}
    out.update({"judge0__" + k: v for k, v in judge.items()})
    st_g = S.adam_start(gen)
    st_j = S.adam_start(judge)
    lj_all, lg_all = [], []
    fixed = noise(mean, spread, 16, stream=5)
    for step in range(GAN_STEPS):
        if step in SNAPS:
            out["made%d" % step] = decode(gen, fixed)
        x, z = gan_batch(step, mean, spread)
        lj, Gj = gan_step(gen, judge, x, z)
        judge, st_j = S.adam_update(judge, Gj, st_j, lr=GAN_LR, betas=BETAS)
        lg, Gg = gen_grads(gen, judge, z)
        if step == 0:
            out.update({"ganGj__" + k: v for k, v in Gj.items()})
            out.update({"ganGg__" + k: v for k, v in Gg.items()})
        gen, st_g = S.adam_update(gen, Gg, st_g, lr=GAN_LR, betas=BETAS)
        if step == 0:
            out.update({"gen1__" + k: v.copy() for k, v in gen.items()})
            out.update({"judge1__" + k: v.copy() for k, v in judge.items()})
        lj_all.append(lj)
        lg_all.append(lg)
    out["made%d" % GAN_STEPS] = decode(gen, fixed)
    out.update(lj=np.array(lj_all), lg=np.array(lg_all), fixed=fixed)
    out.update({"gen__" + k: v for k, v in gen.items()})
    out.update({"judge__" + k: v for k, v in judge.items()})
    # measured on held-out digits and fresh noise
    xt, yt = pixels()[1][:SAMPLES], S.data()[3][:SAMPLES]
    z = noise(mean, spread)
    pics = dict(real=xt, rebuilt=decode(dec, encode(enc, xt)), before=decode(dec, z),
                after=decode(gen, z))
    for name, img in pics.items():
        lab, sure = read(img)
        out["grey__" + name] = np.float64(grey(img))
        out["labels__" + name] = lab
        out["sure__" + name] = sure
    out["rebuild_mse"] = np.float64(((pics["rebuilt"] - xt) ** 2).mean())
    out["rebuild_right"] = np.float64((out["labels__rebuilt"] == yt).mean())
    out["real_right"] = np.float64((out["labels__real"] == yt).mean())
    out["examples_rebuilt"] = pics["rebuilt"][:16]
    out["examples_before"] = pics["before"][:16]
    out["examples_after"] = pics["after"][:16]
    return out


def _settings():
    return dict(version=VERSION, code=CODE, hidden=HIDDEN, batch=BATCH, ae=(AE_STEPS, AE_LR),
                gan=(GAN_STEPS, GAN_LR, BETAS), snaps=SNAPS, samples=SAMPLES, course=S.VERSION,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    return {k: np.asarray(v) for k, v in cached_npz("study-aegan", _settings(), _build).items()}


def blocks(which):
    """Weights by prefix: ``enc0``, ``dec0`` (the start), ``enc``, ``dec`` (after the
    autoencoder), ``judge0`` (rewired), ``gen``, ``judge`` (after the GAN game), ``gen1``,
    ``judge1`` (after its first step)."""
    p = which + "__"
    return {k[len(p):]: v for k, v in run().items() if k.startswith(p)}


def results():
    r = run()
    out = {}
    for name in ("real", "rebuilt", "before", "after"):
        out[name] = dict(grey=float(r["grey__" + name]),
                         counts=np.bincount(r["labels__" + name], minlength=10).tolist(),
                         sure=float(r["sure__" + name].mean()))
    out["rebuild_mse"] = float(r["rebuild_mse"])
    out["rebuild_right"] = float(r["rebuild_right"])
    out["real_right"] = float(r["real_right"])
    return out


def summary():
    r = run()
    return dict(results=results(), ae_loss=(float(r["ae_loss"][0]), float(r["ae_loss"][-100:]
                                                                         .mean())),
                lj=(float(r["lj"][:100].mean()), float(r["lj"][-100:].mean())),
                lg=(float(r["lg"][:100].mean()), float(r["lg"][-100:].mean())),
                encoder_numbers=int(sum(v.size for v in blocks("enc").values())),
                decoder_numbers=int(sum(v.size for v in blocks("dec").values())),
                judge_new=int(sum(v.size for v in judge_head().values())))
