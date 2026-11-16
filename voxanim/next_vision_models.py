"""Compact NumPy models for the next vision experiments (all seeded)."""

from __future__ import annotations

import numpy as np

from . import autoencoder as ae
from . import toy2d as toy
from .optim import Adam


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def downsample(x, side=7):
    k = 28 // side
    return x.reshape(-1, side, k, side, k).mean((2, 4)).reshape(-1, side * side)


def cnn_input_grad(params, x, grad):
    from . import mnist

    f = mnist._cnn_forward(params, x)
    dpool = (grad @ params[1][0].T).reshape(-1, 13, 13, 4)
    route = (np.eye(4)[f["arg"]] * dpool[..., None]).reshape(-1, 13, 13, 4, 2, 2)
    dz = route.transpose(0, 1, 4, 2, 5, 3).reshape(-1, 26, 26, 4) * (f["z1"] > 0)
    dx = np.zeros((len(x), 28, 28))
    for i in range(3):
        for j in range(3):
            dx[:, i : i + 26, j : j + 26] += dz @ params[0][0][i, j]
    return dx.reshape(-1, 784)


def cnn_feature_grad(params, x, channel, target_gram=None, content=None, strength=0.2):
    from . import mnist

    f = mnist._cnn_forward(params, x)
    a = f["a1"].reshape(-1, 4)
    if target_gram is None:
        objective = float(a[:, channel].mean())
        da = np.zeros_like(a)
        da[:, channel] = 1 / len(a)
    else:
        gram = a.T @ a / len(a)
        d = gram - target_gram
        objective = float(np.sum(d**2))
        da = 4 * a @ d / len(a)
        if content is not None:
            objective += strength * np.mean((a - content) ** 2)
            da += 2 * strength * (a - content) / a.size
    dz = da.reshape(f["a1"].shape) * (f["z1"] > 0)
    dx = np.zeros((len(x), 28, 28))
    for i in range(3):
        for j in range(3):
            dx[:, i : i + 26, j : j + 26] += dz @ params[0][0][i, j]
    return objective, dx.reshape(-1, 784), f


def info_nce(p, images, labels, temperature=0.15):
    """Symmetric image/text contrastive loss with multi-positive text targets."""
    raw_i, raw_t = images @ p["I"], p["T"]
    ni = np.maximum(np.linalg.norm(raw_i, axis=1, keepdims=True), 1e-8)
    nt = np.maximum(np.linalg.norm(raw_t, axis=1, keepdims=True), 1e-8)
    z, t = raw_i / ni, raw_t / nt
    scores = z @ t.T / temperature
    pr = toy.softmax(scores)
    target = np.eye(len(t))[labels]
    loss_image = -np.sum(target * np.log(np.maximum(pr, 1e-300))) / len(images)
    text_pr = toy.softmax(scores.T)
    counts = target.sum(0)
    active = counts > 0
    text_target = target.T / np.maximum(counts[:, None], 1)
    loss_text = -np.sum(text_target * np.log(np.maximum(text_pr, 1e-300))) / active.sum()
    ds = 0.5 * (
        (pr - target) / len(images) + ((text_pr - text_target) * active[:, None]).T / active.sum()
    )
    dz, dt = ds @ t / temperature, ds.T @ z / temperature
    di = (dz - z * (dz * z).sum(1, keepdims=True)) / ni
    dt = (dt - t * (dt * t).sum(1, keepdims=True)) / nt
    return float(0.5 * (loss_image + loss_text)), {"I": images.T @ di, "T": dt}, scores


def fit_clip(x, y, steps=900, seed=53):
    rng = np.random.default_rng(seed)
    p = dict(I=rng.normal(0, 0.12, (x.shape[1], 16)), T=rng.normal(0, 0.2, (10, 16)))
    opt, losses, snaps = Adam(p, 0.006), [], []
    fixed = np.array([np.flatnonzero(y == i)[0] for i in range(10)])
    for step in range(steps + 1):
        if step in (0, steps // 2, steps):
            snaps.append(info_nce(p, x[fixed], y[fixed])[2])
        if step == steps:
            break
        idx = rng.choice(len(x), 64)
        loss, g, _ = info_nce(p, x[idx], y[idx])
        opt.step(p, g)
        losses.append(loss)
    return p, np.array(losses), np.array(snaps)


def conditional_diffusion(x, y, steps=2200, seed=62):
    """A conditional denoiser on 7x7 MNIST, with 15% conditioning dropout."""
    from .diffusion import schedule

    rng = np.random.default_rng(seed)
    p = toy.init((49 + 16 + 10, 96, 96, 49), rng)
    opt, loss = Adam(p, 0.003), []
    _, _, abar = schedule(80)
    for _ in range(steps):
        ids = rng.choice(len(x), 96)
        t = rng.integers(1, 81, len(ids))
        eps = rng.normal(size=(len(ids), 49))
        xt = np.sqrt(abar[t, None]) * (x[ids] * 2 - 1) + np.sqrt(1 - abar[t, None]) * eps
        cond = np.eye(10)[y[ids]]
        cond[rng.random(len(ids)) < 0.15] = 0
        inp = np.column_stack((xt, time_features(t), cond))
        l, g = toy.loss_grads(p, inp, eps)
        opt.step(p, g)
        loss.append(l)
    return p, np.array(loss), abar


def time_features(t):
    a = np.asarray(t)[:, None] * np.exp(-np.arange(8)[None] / 2)
    return np.column_stack((np.sin(a), np.cos(a)))


def conditional_sample(p, abar, scales=(0, 1, 3, 8), seed=63):
    rng = np.random.default_rng(seed)
    all_samples = []
    labels = np.repeat([0, 3, 5, 7], 8)
    noise = rng.normal(size=(len(labels), 49))
    for w in scales:
        x = noise.copy()
        for t in range(len(abar) - 1, 0, -1):
            tem = time_features(np.full(len(x), t))
            e0 = toy.forward(p, np.column_stack((x, tem, np.zeros((len(x), 10)))))[0]
            ec = toy.forward(p, np.column_stack((x, tem, np.eye(10)[labels])))[0]
            eps = e0 + w * (ec - e0)
            x0 = np.clip((x - np.sqrt(1 - abar[t]) * eps) / np.sqrt(abar[t]), -1, 1)
            x = np.sqrt(abar[t - 1]) * x0 + np.sqrt(1 - abar[t - 1]) * eps
        all_samples.append(np.clip((x + 1) / 2, 0, 1))
    return np.array(all_samples)


def vq_loss_grads(p, x, commitment=0.25):
    h = np.tanh(x @ p["E"] + p["eb"])
    ids = ((h[:, None] - p["C"][None]) ** 2).sum(-1).argmin(1)
    z = p["C"][ids]
    logits = z @ p["D"] + p["db"]
    r = sigmoid(logits)
    bce = np.mean(np.logaddexp(0, logits) - x * logits)
    qloss = np.mean((h - z) ** 2)
    dr = (r - x) / x.size
    g = {"D": z.T @ dr, "db": dr.sum(0), "C": np.zeros_like(p["C"])}
    dz = dr @ p["D"].T
    dh = (dz + 2 * commitment * (h - z) / h.size) * (1 - h**2)
    g["E"], g["eb"] = x.T @ dh, dh.sum(0)
    np.add.at(g["C"], ids, 2 * (z - h) / h.size)
    return float(bce + (1 + commitment) * qloss), g, (h, z, ids, r)


def masked_conv_loss(p, x):
    """Type-A 3x3 masked conv followed by a 1x1 output convolution."""
    mask = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])[:, :, None, None]
    w = p["W"] * mask
    z, cols, shape = ae.conv(x, w, p["b"], 1, 1)
    h = np.tanh(z)
    logits = h @ p["O"] + p["c"]
    r = sigmoid(logits)
    loss = np.mean(np.logaddexp(0, logits) - x * logits)
    d = (r - x) / x.size
    g = {"O": h.reshape(-1, h.shape[-1]).T @ d.reshape(-1, 1), "c": d.sum((0, 1, 2))}
    dz = (d @ p["O"].T) * (1 - h**2)
    gw, gb, _ = ae.conv_back(dz, cols, w, shape, 1, 1)
    g["W"], g["b"] = gw * mask, gb
    return float(loss), g, r


def unet_init(rng):
    return dict(
        W0=rng.normal(0, 0.2, (3, 3, 1, 4)),
        b0=np.zeros(4),
        W1=rng.normal(0, 0.12, (3, 3, 4, 8)),
        b1=np.zeros(8),
        W2=rng.normal(0, 0.2, (1, 1, 12, 3)),
        b2=np.zeros(3),
    )


def unet_loss(p, x, y):
    z0, c0, s0 = ae.conv(x, p["W0"], p["b0"], 1, 1)
    h0 = np.tanh(z0)
    n, h, w, c = h0.shape
    pooled = h0.reshape(n, h // 2, 2, w // 2, 2, c).mean((2, 4))
    z1, c1, s1 = ae.conv(pooled, p["W1"], p["b1"], 1, 1)
    h1 = np.tanh(z1)
    joined = np.concatenate((h0, ae.up2(h1)), axis=-1)
    logits, c2, s2 = ae.conv(joined, p["W2"], p["b2"], 1, 0)
    pr = toy.softmax(logits)
    target = np.eye(3)[y]
    loss = -np.mean(np.log(np.maximum(pr, 1e-12)) * target) * 3
    dlog = (pr - target) / y.size
    w2, b2, dj = ae.conv_back(dlog, c2, p["W2"], s2, 1, 0)
    dh1 = ae.up2_back(dj[..., 4:]) * (1 - h1**2)
    w1, b1, dp = ae.conv_back(dh1, c1, p["W1"], s1, 1, 1)
    dh0 = dj[..., :4] + dp.repeat(2, axis=1).repeat(2, axis=2) / 4
    w0, b0, _ = ae.conv_back(dh0 * (1 - h0**2), c0, p["W0"], s0, 1, 1)
    return float(loss), dict(W0=w0, b0=b0, W1=w1, b1=b1, W2=w2, b2=b2), pr


def small_vit_init(rng, layers=2):
    p = dict(
        E=rng.normal(0, 0.1, (49, 16)),
        cls=rng.normal(0, 0.1, (16,)),
        pos=rng.normal(0, 0.02, (17, 16)),
        U=rng.normal(0, 0.2, (16, 10)),
    )
    for i in range(layers):
        for name, shape in [
            ("Q", (16, 16)),
            ("K", (16, 16)),
            ("V", (16, 16)),
            ("O", (16, 16)),
            ("A", (16, 32)),
            ("B", (32, 16)),
        ]:
            p[f"{name}{i}"] = rng.normal(0, 1 / np.sqrt(shape[0]), shape)
    return p


def small_vit_loss(p, x, y, grads=True):
    from .tinylm import Var, softmax
    from .vit import patches

    ps = {k: Var(v) for k, v in p.items()}
    n = len(x)
    embedded = Var(patches(x)) @ ps["E"]
    # Concatenation expressed as addition of disjoint padded arrays keeps gradients explicit.
    token = Var(np.zeros((n, 17, 16)))
    patch = Var(np.pad(embedded.v, ((0, 0), (1, 0), (0, 0))), ((embedded, lambda g: g[:, 1:]),))
    cls = ps["cls"] * np.ones((n, 1, 16))
    clspad = Var(np.pad(cls.v, ((0, 0), (0, 16), (0, 0))), ((cls, lambda g: g[:, :1]),))
    token = token + patch + clspad + ps["pos"]
    attn = []
    for i in range((len(p) - 4) // 6):
        norm = token - token.mean(-1, True)
        norm = norm / ((norm**2).mean(-1, True) + 1e-5) ** 0.5
        q = (norm @ ps[f"Q{i}"]).reshape(n, 17, 2, 8).transpose(0, 2, 1, 3)
        k = (norm @ ps[f"K{i}"]).reshape(n, 17, 2, 8).transpose(0, 2, 1, 3)
        v = (norm @ ps[f"V{i}"]).reshape(n, 17, 2, 8).transpose(0, 2, 1, 3)
        a = softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(8))
        attn.append(a.v)
        token = token + (a @ v).transpose(0, 2, 1, 3).reshape(n, 17, 16) @ ps[f"O{i}"]
        token = token + (token @ ps[f"A{i}"]).tanh() @ ps[f"B{i}"]
    logits = token[:, 0] @ ps["U"]
    shifted = logits - logits.v.max(-1, keepdims=True)
    lp = shifted - shifted.exp().sum(-1, True).log()
    loss = -lp[np.arange(n), y].sum() / n
    if grads:
        loss.backward()
    return float(loss.v), {k: v.g for k, v in ps.items()}, (logits.v, np.array(attn))


def autoregressive_vae_loss(p, x, noise, beta=1.0):
    """A Gaussian encoder and a strong, causal-pixel decoder, with exact gradients."""
    mu = x @ p["E"] + p["eb"]
    lv = x @ p["L"] + p["lb"]
    sigma = np.exp(lv / 2)
    z = mu + sigma * noise
    mask = np.triu(np.ones_like(p["AR"]), 1)
    logits = z @ p["D"] + x @ (p["AR"] * mask) + p["db"]
    r = sigmoid(logits)
    kl = 0.5 * (mu**2 + np.exp(lv) - 1 - lv)
    loss = np.mean(np.sum(np.logaddexp(0, logits) - x * logits, axis=1))
    loss += beta * kl.sum(1).mean()
    dl = (r - x) / len(x)
    dz = dl @ p["D"].T
    dmu = dz + beta * mu / len(x)
    dlv = dz * sigma * noise / 2 + beta * (np.exp(lv) - 1) / (2 * len(x))
    g = dict(
        D=z.T @ dl,
        db=dl.sum(0),
        AR=(x.T @ dl) * mask,
        E=x.T @ dmu,
        eb=dmu.sum(0),
        L=x.T @ dlv,
        lb=dlv.sum(0),
    )
    return float(loss), g, (kl.mean(0), r)


def gan_run(
    real, steps=1800, minibatch=False, seed=51, return_discriminator=False, return_parameters=False
):
    """Non-saturating GAN, optionally with a differentiable minibatch-distance feature."""
    rng = np.random.default_rng(seed)
    gp = toy.init((2, 32, 32, 2), rng)
    dp = toy.init((3 if minibatch else 2, 32, 32, 1), rng)
    go, do = Adam(gp, 0.002), Adam(dp, 0.001)
    fixed = rng.normal(size=(240, 2))
    snaps, losses = [], []

    for step in range(steps + 1):
        if step in (0, steps // 3, 2 * steps // 3, steps):
            snaps.append(toy.forward(gp, fixed)[0])
        if step == steps:
            break
        z = rng.normal(size=(96, 2))
        fake, ga = toy.forward(gp, z)
        xx = np.vstack((real[rng.choice(len(real), 96)], fake))
        inp, _ = minibatch_features(xx, minibatch)
        target = np.r_[np.ones(96), np.zeros(96)][:, None]
        logits, da = toy.forward(dp, inp)
        pr = sigmoid(logits)
        dg, _ = toy.backward(dp, da, (pr - target) / len(xx))
        do.step(dp, dg)
        inp, tape = minibatch_features(fake, minibatch)
        logits, da = toy.forward(dp, inp)
        _, dx = toy.backward(dp, da, (sigmoid(logits) - 1) / len(fake))
        gg, _ = toy.backward(gp, ga, minibatch_backward(dx, tape))
        go.step(gp, gg)
        losses.append(float(np.mean(np.logaddexp(0, logits) - logits)))
    if return_discriminator:
        grid = np.linspace(-3, 3, 40)
        gx, gy = np.meshgrid(grid, grid)
        xy = np.column_stack((gx.ravel(), gy.ravel()))
        inputs, _ = minibatch_features(xy, minibatch)
        scores = sigmoid(toy.forward(dp, inputs)[0]).reshape(40, 40)
        result = (np.array(snaps), np.array(losses), scores)
        if return_parameters:
            return (
                *result,
                {
                    **{"generator_" + k: v for k, v in gp.items()},
                    **{"discriminator_" + k: v for k, v in dp.items()},
                },
            )
        return result
    return np.array(snaps), np.array(losses)


def minibatch_features(x, enabled=True):
    """Append the average Gaussian similarity to all other samples."""
    if not enabled:
        return x, None
    differences = x[:, None] - x[None]
    similarity = np.exp(-np.sum(differences**2, axis=-1))
    np.fill_diagonal(similarity, 0)
    feature = similarity.sum(1) / (len(x) - 1)
    return np.column_stack((x, feature)), (differences, similarity)


def minibatch_backward(g, tape):
    """Differentiate every pair's effect on both appended feature values."""
    if tape is None:
        return g
    dif, sim = tape
    pair = -2 * (g[:, -1, None] + g[None, :, -1]) * sim / (len(g) - 1)
    return g[:, :2] + (pair[..., None] * dif).sum(1)


def scientific_canvas(
    points=(),
    classes=None,
    paths=(),
    colors=None,
    background=None,
    limits=(-3.0, 3.0, -3.0, 3.0),
    size=160,
):
    """Rasterise computed points and polylines without plotting dependencies."""
    from PIL import Image, ImageDraw

    lo_x, hi_x, lo_y, hi_y = limits
    canvas = np.full((size, size, 3), 0.96) if background is None else np.asarray(background)
    if canvas.shape != (size, size, 3):
        canvas = (
            np.asarray(Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255)).resize((size, size)))
            / 255
        )
    im = Image.fromarray(np.uint8(np.clip(canvas, 0, 1) * 255))
    draw = ImageDraw.Draw(im)
    palette = [
        (24, 137, 128),
        (212, 89, 59),
        (112, 92, 185),
        (157, 130, 45),
        (48, 115, 171),
        (183, 77, 125),
        (90, 150, 56),
        (117, 117, 117),
    ]

    def coordinates(x):
        x = np.asarray(x)
        return np.column_stack(
            (
                (x[:, 0] - lo_x) / (hi_x - lo_x) * (size - 1),
                (hi_y - x[:, 1]) / (hi_y - lo_y) * (size - 1),
            )
        )

    for i, path in enumerate(paths):
        pts = [tuple(p) for p in coordinates(path)]
        draw.line(pts, fill=palette[i % len(palette)], width=1)
    p = np.asarray(points)
    if len(p):
        classes = np.zeros(len(p), int) if classes is None else classes
        for i, (x, y) in enumerate(coordinates(p)):
            if colors is None:
                color = palette[int(classes[i]) % len(palette)]
            else:
                color = tuple(np.uint8(np.clip(colors[i], 0, 1) * 255))
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
    return np.asarray(im) / 255.0
