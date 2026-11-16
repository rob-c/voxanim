"""Redesigned experiments for four of the "looking inside" animations (44, 47, 48, 50).

The first versions (:mod:`voxanim.next_language`) could not show their concept: a sparse
autoencoder given 926 tokens and 180 updates (most features dead, many locked on the
oversized first-token activation), 144 heads ranked from two passages, an ascent on a
first-layer channel whose answer is mostly its bias on blank paper, and a small model that
grew no induction head. Each is run again here with a design that gives the concept a fair
chance, and with what is needed to believe the result: held-out text, many passages, the
spread, an independent judge, several seeds.

Every function ending ``_data`` is the experiment (seeded, plain numpy, cached); the
function registered in :data:`STUDIES` wraps its numbers as a four-panel
:class:`~voxanim.study.Study`. :mod:`voxanim.mech_interp` draws from both.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import gpt2, realdata
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SEED = 23


def _settings(name, **extra):
    return dict(version=VERSION, seed=SEED, name=name, gpt2_revision=gpt2.REVISION,
                dataset_sha256=realdata.TEXT_SHA256, **extra)


def _text_meta(**extra):
    return dict(
        seed=SEED, study_version=VERSION,
        dataset="Tiny Shakespeare / char-rnn", dataset_sha256=realdata.TEXT_SHA256,
        dataset_provenance={k: realdata.text_metadata(k)
                            for k in ("train", "validation", "test")},
        **extra)


@lru_cache(maxsize=1)
def _tok():
    return gpt2.Tokenizer()


# ---------------------------------------------------------------------- GPT-2, in batches
def token_windows(split, count, length):
    """``count`` windows of ``length`` GPT-2 tokens from one split of Tiny Shakespeare,
    evenly spread through it, each starting at the start of a line. Returns the token ids
    ``(count, length)`` and each window's character offset in the split."""
    text = realdata.text(split)
    ids, starts = [], []
    for at in np.linspace(0, len(text) - 12 * length, count).astype(int):
        start = 0 if at == 0 else text.index("\n", at) + 1
        if starts and start <= starts[-1]:
            continue
        window = _tok().encode(text[start:start + 12 * length])[:length]
        if len(window) == length:
            ids.append(window)
            starts.append(start)
    return np.asarray(ids), np.asarray(starts)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(-1, keepdims=True)
    return z


def _gelu(x):
    """:func:`voxanim.gpt2._gelu` kept in float32 (that one returns float64, and every
    product after it then runs at half the speed)."""
    y = x * x
    y *= x
    y *= np.float32(0.044715)
    y += x
    y *= np.float32(np.sqrt(2 / np.pi))
    np.tanh(y, out=y)
    y += 1
    y *= x
    y *= np.float32(0.5)
    return y


def gpt2_embed(ids):
    W = gpt2.weights()
    ids = np.asarray(ids)
    return W["wte.weight"][ids] + W["wpe.weight"][np.arange(ids.shape[1])]


def gpt2_heads(layer, x):
    """Layer ``layer``'s attention on the stream ``x`` (passages, tokens, 768): what each
    head would add before the output projection ``(passages, 12, tokens, 64)``, and the
    attention weights."""
    W, p = gpt2.weights(), "h.%d." % layer
    n, t, _ = x.shape
    h = gpt2.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"]).reshape(n * t, 768)
    qkv = (h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"]).reshape(n, t, 3, 12, 64)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    score = q @ k.transpose(0, 1, 3, 2) / 8
    score[..., np.triu(np.ones((t, t), bool), 1)] = -np.inf
    a = _softmax(score)
    return a @ v, a


def gpt2_finish(layer, x, heads):
    """The rest of layer ``layer`` given its heads' outputs: the stream leaving it."""
    W, p = gpt2.weights(), "h.%d." % layer
    n, _, t, _ = heads.shape
    # (tokens, 768) matrices: one large product is far faster than a stack of small ones
    x = x.reshape(n * t, 768) + (heads.transpose(0, 2, 1, 3).reshape(n * t, 768)
                                 @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"])
    h = gpt2.layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
    hidden = _gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
    return (x + hidden @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]).reshape(n, t, 768)


def gpt2_streams(ids, layers=12, attention=False):
    """The residual stream entering each of the first ``layers`` blocks and leaving the
    last: a list of ``layers + 1`` arrays (passages, tokens, 768). With ``attention``,
    also every layer's attention weights."""
    x = gpt2_embed(ids)
    out, maps = [x], []
    for layer in range(layers):
        heads, a = gpt2_heads(layer, x)
        x = gpt2_finish(layer, x, heads)
        out.append(x)
        maps.append(a)
    return (out, maps) if attention else out


def gpt2_nll(x, targets, rows=2048):
    """Next-token loss (nats) at every position, from the final stream ``x``."""
    W = gpt2.weights()
    flat = gpt2.layer_norm(x, W["ln_f.weight"], W["ln_f.bias"]).reshape(-1, 768)
    goal = np.asarray(targets).reshape(-1)
    out = np.empty(len(flat))
    for i in range(0, len(flat), rows):
        z = flat[i:i + rows] @ W["wte.weight"].T
        z -= z.max(-1, keepdims=True)
        hit = z[np.arange(len(z)), goal[i:i + rows]].astype(np.float64)
        out[i:i + rows] = np.log(np.exp(z, out=z).sum(-1, dtype=np.float64)) - hit
    return out.reshape(np.asarray(targets).shape)


# ---------------------------------------------------------------------- 48 head ablation
ABLATION = dict(windows=64, tokens=16, batch=8)


def ablation_data():
    """Every one of GPT-2's 144 heads switched off (its output zeroed) on each of 64
    held-out passages of 16 tokens; the loss is the mean next-token loss over the
    passage's 16 predictions.

    ``change`` (passage, layer, head) is that passage's loss without the head minus its
    loss with every head on. The design (zero ablation, 64 windows, all 16 positions,
    odd/even halves) was fixed before any ablation was run."""
    def build():
        ids, starts = token_windows("test", ABLATION["windows"], ABLATION["tokens"] + 1)
        n, t = len(ids), ABLATION["tokens"]
        base = np.zeros((n, t))
        change = np.zeros((n, 12, 12))
        heads12 = np.arange(12)
        for i in range(0, n, ABLATION["batch"]):
            x, y = ids[i:i + ABLATION["batch"], :-1], ids[i:i + ABLATION["batch"], 1:]
            b = len(x)
            streams = gpt2_streams(x)
            base[i:i + b] = gpt2_nll(streams[-1], y)
            for layer in range(12):
                heads, _ = gpt2_heads(layer, streams[layer])
                heads = np.repeat(heads, 12, axis=0)              # passage-major, then head
                heads[np.arange(12 * b), np.tile(heads12, b)] = 0
                z = gpt2_finish(layer, np.repeat(streams[layer], 12, axis=0), heads)
                for later in range(layer + 1, 12):
                    z = gpt2_finish(later, z, gpt2_heads(later, z)[0])
                loss = gpt2_nll(z, np.repeat(y, 12, axis=0)).mean(-1).reshape(b, 12)
                change[i:i + b, layer] = loss - base[i:i + b].mean(-1, keepdims=True)
        tok = _tok()
        return dict(ids=ids, starts=starts, base=base, change=change,
                    tokens=np.asarray([[tok.decode([int(k)]) for k in row] for row in ids]))
    return cached_npz("study-interp-ablation", _settings("ablation", **ABLATION), build)


def spearman(a, b):
    """Rank correlation of two equally long vectors (no ties expected in measured floats)."""
    ra, rb = (np.argsort(np.argsort(np.ravel(v))).astype(float) for v in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1])


def ablation_summary(d):
    """Mean, standard error and split-half stability of the per-head loss change."""
    change = np.asarray(d["change"], float)
    n = len(change)
    mean = change.mean(0)
    sem = change.std(0, ddof=1) / np.sqrt(n)
    even, odd = change[0::2].mean(0), change[1::2].mean(0)
    rng = np.random.default_rng(SEED)
    splits = []
    for _ in range(200):                      # the same check over 200 random halvings
        order = rng.permutation(n)
        splits.append(spearman(change[order[:n // 2]].mean(0), change[order[n // 2:]].mean(0)))
    return dict(mean=mean, sem=sem, even=even, odd=odd, rank_correlation=spearman(even, odd),
                pearson=float(np.corrcoef(even.ravel(), odd.ravel())[0, 1]),
                random_halves=np.asarray(splits),
                clear=np.abs(mean) > 2 * sem,
                # how unstable one passage is: its ranking against the other 95's
                single=np.asarray([spearman(change[i], (mean * n - change[i]) / (n - 1))
                                   for i in range(n)]))


# ---------------------------------------------------------------------- 44 sparse autoencoder
SAE = dict(layer=6, features=1024, k=24, sweep=(6, 24, 96), updates=2000, batch=256,
           lr=1e-3, resample=(400, 800, 1200), train_windows=512, heldout_windows=96, tokens=64,
           top=20)


def sae_encode(P, x, k, sparse=False):
    """The ``k`` largest feature pre-activations of each row kept (if above zero), the
    rest set to zero: a top-k sparse code. With ``sparse``, returns which features were
    kept and their values instead of the full matrix."""
    pre = (x - P["c"]) @ P["E"] + P["b"]
    keep = np.argpartition(-pre, k, axis=1)[:, :k]
    rows = np.arange(len(x))[:, None]
    values = np.maximum(pre[rows, keep], 0)
    if sparse:
        return keep, values
    h = np.zeros_like(pre)
    h[rows, keep] = values
    return h


def sae_loss(P, x, k):
    """Mean squared reconstruction error of the top-k autoencoder, its gradients, and
    which features fired. Only the ``k`` kept features of a row touch the decoder, so
    their decoder rows are gathered instead of multiplying by the whole dictionary."""
    keep, values = sae_encode(P, x, k, sparse=True)
    rows = np.arange(len(x))[:, None]
    chosen = P["D"][keep]                                        # row, kept, 768
    out = (values[:, None] @ chosen)[:, 0] + P["c"]
    delta = 2 * (out - x) / x.size
    h, dh = np.zeros((len(x), len(P["b"])), x.dtype), np.zeros((len(x), len(P["b"])), x.dtype)
    h[rows, keep] = values
    dh[rows, keep] = (chosen @ delta[:, :, None])[:, :, 0] * (values > 0)
    return float(np.mean((out - x) ** 2)), dict(
        E=(x - P["c"]).T @ dh, b=dh.sum(0), D=h.T @ delta,
        c=delta.sum(0) - dh.sum(0) @ P["E"].T), h


def sae_train(x, k, seed, features=None, updates=None, resample=None):
    """Train one top-k sparse autoencoder on the rows of ``x``.

    The decoder's rows are kept at unit length and the encoder starts as its transpose.
    At each update in ``resample`` every feature that has not fired since the last check
    is pointed at a token the autoencoder currently rebuilds badly (chosen with
    probability proportional to its squared error), which is the usual remedy for dead
    features. Returns the weights, the loss per update and how many were resampled."""
    features = features or SAE["features"]
    updates = updates or SAE["updates"]
    resample = SAE["resample"] if resample is None else resample
    rng = np.random.default_rng(seed)
    D = rng.normal(size=(features, x.shape[1])).astype(x.dtype)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    P = dict(E=D.T.copy(), b=np.zeros(features, x.dtype), D=D, c=np.zeros(x.shape[1], x.dtype))
    opt = Adam(P, SAE["lr"])
    losses, fired, revived = [], np.zeros(features, bool), []
    for step in range(updates):
        if step in resample:
            dead = np.where(~fired)[0]
            revived.append(len(dead))
            if len(dead):
                sample = x[rng.choice(len(x), 8192, replace=False)]
                miss = sample - (sae_encode(P, sample, k) @ P["D"] + P["c"])
                err = (miss ** 2).sum(1)
                pick = rng.choice(len(sample), len(dead), p=err / err.sum())
                direction = miss[pick] / np.linalg.norm(miss[pick], axis=1, keepdims=True)
                P["D"][dead], P["E"][:, dead], P["b"][dead] = direction, direction.T, 0.0
                for key in ("E", "b", "D"):               # no stale momentum for the new ones
                    for moment in (opt.m, opt.v):
                        if key == "E":
                            moment[key][:, dead] = 0
                        else:
                            moment[key][dead] = 0
            fired[:] = False
        ix = rng.integers(len(x), size=SAE["batch"])
        lo, g, h = sae_loss(P, x[ix], k)
        fired |= (h > 0).any(0)
        opt.step(P, g)
        P["D"] /= np.linalg.norm(P["D"], axis=1, keepdims=True)
        losses.append(lo)
    return P, np.asarray(losses), np.asarray(revived)


def _purity(tokens):
    """The share of ``tokens`` equal to the commonest one (case and spaces ignored)."""
    words = [t.strip().lower() for t in tokens]
    return max(words.count(w) for w in set(words)) / len(words)


def sae_data():
    """A top-k sparse autoencoder on GPT-2's layer-6 stream over Tiny Shakespeare.

    Activations come from 64-token windows; each window's first token is left out (its
    activation is tens of times larger than any other and is the same whatever the text).
    They are centred and scaled with the training tokens' statistics. One autoencoder is
    trained at each sparsity in ``SAE["sweep"]``; the one at ``k`` is the one looked at.
    Everything reported is measured on the held-out windows (the test split)."""
    def build():
        tok = _tok()
        layer, top = SAE["layer"], SAE["top"]

        def stream(split, count):
            ids, _ = token_windows(split, count, SAE["tokens"])
            x = np.concatenate([gpt2_streams(ids[i:i + 32], layer)[layer]
                                for i in range(0, len(ids), 32)])
            return ids, x

        _ids, raw = stream("train", SAE["train_windows"])
        hids, hraw = stream("test", SAE["heldout_windows"])
        norms = np.linalg.norm(raw, axis=-1)
        mean = raw[:, 1:].reshape(-1, 768).mean(0)
        scale = np.sqrt(((raw[:, 1:] - mean) ** 2).mean(dtype=np.float64)).astype(np.float32)
        x = ((raw[:, 1:] - mean) / scale).reshape(-1, 768)
        xt = ((hraw[:, 1:] - mean) / scale).reshape(-1, 768)
        total = np.mean((xt - xt.mean(0)) ** 2)

        def unexplained(P, k, data):
            return float(np.mean((sae_encode(P, data, k) @ P["D"] + P["c"] - data) ** 2) / total)

        sweep, sweep_dead = [], []
        for k in SAE["sweep"]:
            Pk, lossk, revivedk = sae_train(x, k, SEED)
            hk = sae_encode(Pk, xt, k)
            sweep.append(unexplained(Pk, k, xt))
            sweep_dead.append(np.mean(~(hk > 0).any(0)))
            if k == SAE["k"]:
                P, losses, revived = Pk, lossk, revivedk
        k = SAE["k"]
        act, ht = sae_encode(P, x, k), sae_encode(P, xt, k)
        recon = ht @ P["D"] + P["c"]
        fires = (ht > 0).sum(0)
        words = np.asarray([tok.decode([int(i)]) for i in hids[:, 1:].ravel()])
        prev = np.asarray([tok.decode([int(i)]) for i in hids[:, :-1].ravel()])
        place = np.tile(np.arange(1, SAE["tokens"]), len(hids))
        # how readable each feature looks: of its `top` strongest held-out firings, the
        # share on its commonest token, and on its commonest preceding token
        purity, before = np.zeros(len(fires)), np.zeros(len(fires))
        scored = fires >= top
        order = np.argsort(-ht, axis=0)[:top]
        for j in np.where(scored)[0]:
            purity[j] = _purity(words[order[:, j]])
            before[j] = _purity(prev[order[:, j]])
        rng = np.random.default_rng(SEED)
        chance = np.mean([_purity(words[rng.choice(len(words), top, replace=False)])
                          for _ in range(500)])
        flat = hids.ravel()

        def context(pos, width=6):
            """The held-out token at ``pos`` (an index into the kept tokens) with the
            ``width - 1`` before it in its window ('' where the window has not begun)."""
            w, t = divmod(int(pos), SAE["tokens"] - 1)
            t += 1
            return [tok.decode([int(flat[w * SAE["tokens"] + s])]) if s >= 0 else ""
                    for s in range(t - width + 1, t + 1)]

        contexts = np.asarray([[context(p) for p in order[:8, j]] for j in range(len(fires))])
        window = np.arange(SAE["tokens"] - 1)                  # the first held-out window
        return dict(
            losses=losses, revived=revived, sweep_k=np.asarray(SAE["sweep"]),
            sweep_unexplained=np.asarray(sweep), sweep_dead=np.asarray(sweep_dead),
            unexplained=np.asarray(unexplained(P, k, xt)),
            train_unexplained=np.asarray(
                np.mean((act @ P["D"] + P["c"] - x) ** 2) / np.mean((x - x.mean(0)) ** 2)),
            train_fires=(act > 0).sum(0), fires=fires, peak=ht.max(0),
            purity=purity, before=before, scored=scored, chance=np.asarray(chance),
            top_values=np.take_along_axis(ht, order[:8], 0).T, contexts=contexts,
            top_place=place[order[:8]].T,
            tokens=words[window], x=xt[window], h=ht[window], recon=recon[window],
            first_norm=np.asarray(norms[:, 0].mean()), other_norm=np.asarray(norms[:, 1:].mean()),
            train_tokens=np.asarray(len(x)), heldout_tokens=np.asarray(len(xt)),
            heldout_text=np.asarray(tok.decode([int(i) for i in hids[0]])))
    return cached_npz("study-interp-sae", _settings("sae", **SAE), build)


# ---------------------------------------------------------------------- 50 feature visualisation
FEATURE = dict(steps=200, rate=0.02, jitter=2, blur=1.0, decay=0.02, seeds=3, shown=3)


def cnn_logits(params, images):
    """The MNIST CNN's ten class scores for ``images`` (n, 28, 28)."""
    from . import mnist

    return mnist._cnn_forward(params, np.asarray(images, float).reshape(-1, 784))["z2"]


def class_objective(params, images, digit):
    """What the ascent climbs: the score of ``digit`` minus the mean score of the other
    nine classes (so a response every class shares, such as the one to blank paper,
    does not count), and its gradient with respect to the pixels."""
    from . import mnist

    (Wc, _), (Wd, _) = params
    x = np.asarray(images, float).reshape(-1, 784)
    f = mnist._cnn_forward(params, x)
    want = np.full(10, -1 / 9.0)
    want[digit] = 1.0
    value = f["z2"] @ want
    F, hw, pool = mnist.CNN_FILTERS, mnist.CONV_HW, mnist.POOL_HW
    dpool = np.broadcast_to(Wd @ want, (len(x), pool * pool * F)).reshape(-1, pool, pool, F)
    route = np.eye(4)[f["arg"]] * dpool[..., None]
    route = route.reshape(-1, pool, pool, F, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    d1 = route.reshape(-1, hw, hw, F) * (f["z1"] > 0)
    grad = np.zeros((len(x), 28, 28))
    for i in range(3):
        for j in range(3):
            grad[:, i:i + hw, j:j + hw] += d1 @ Wc[i, j]
    return value, grad


def blur(images, sigma):
    """A Gaussian blur of each 28 x 28 image (edges repeat)."""
    r = int(np.ceil(3 * sigma))
    w = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    w /= w.sum()
    out = np.asarray(images, float)
    for axis in (-2, -1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        wide = np.pad(out, pad, mode="edge")
        out = sum(w[i] * np.take(wide, np.arange(i, i + out.shape[axis]), axis=axis)
                  for i in range(2 * r + 1))
    return out


def feature_ascent(params, digit, start, regularise, rng, keep=False):
    """``FEATURE['steps']`` steps uphill on :func:`class_objective` from ``start``.

    Each step moves every pixel by the gradient scaled so that its mean size is
    ``rate``. With ``regularise`` the image is shifted by a random few pixels before the
    gradient is taken (and the gradient shifted back), the gradient is blurred, and each
    step pulls the image a little toward blank paper: only what keeps earning its place
    under small shifts survives."""
    image = start.copy()
    frames, trace, shifts = [], [], []
    for step in range(FEATURE["steps"] + 1):
        value, grad = class_objective(params, image[None], digit)
        trace.append(value[0])
        if keep:
            frames.append(image.copy())
        if step == FEATURE["steps"]:
            break
        shift = (0, 0)
        if regularise:
            reach = FEATURE["jitter"]
            shift = tuple(int(v) for v in rng.integers(-reach, reach + 1, 2))
            moved = np.roll(image, shift, axis=(0, 1))
            grad = np.roll(class_objective(params, moved[None], digit)[1][0],
                           (-shift[0], -shift[1]), axis=(0, 1))
            grad = blur(grad, FEATURE["blur"])
        else:
            grad = grad[0]
        shifts.append(shift)
        grad = grad / (np.abs(grad).mean() + 1e-12)
        image = image + FEATURE["rate"] * grad
        if regularise:
            image = image * (1 - FEATURE["decay"])
        image = np.clip(image, 0, 1)
    return image, np.asarray(trace), np.asarray(frames), np.asarray(shifts)


def feature_data():
    """Images optimised from noise to excite one output unit of the MNIST CNN, for every
    digit, plainly and with regularisers, from three different starting noises; each
    result is then shown to a different network (the repository's MLP, never used in the
    ascent) to see whether it reads the digit, and set beside the real test digits that
    excite the same unit most."""
    def build():
        from . import mnist

        params, accuracy = mnist.cnn_train()
        judge, judge_accuracy = mnist.train()
        _, _, test, labels = mnist.load()
        blank = np.asarray([class_objective(params, np.zeros((1, 28, 28)), c)[0][0]
                            for c in range(10)])
        real = np.stack([np.concatenate([class_objective(params, test[i:i + 1000], c)[0]
                                         for i in range(0, len(test), 1000)]) for c in range(10)])
        best = np.argsort(-real, axis=1)[:, :4]                 # each unit's four best real digits
        means = np.stack([test[labels == c].mean(0) for c in range(10)]).reshape(10, 28, 28)
        seeds, steps = FEATURE["seeds"], FEATURE["steps"]
        finals = np.zeros((2, seeds, 10, 28, 28))
        traces = np.zeros((2, seeds, 10, steps + 1))
        frames = np.zeros((2, FEATURE["shown"], steps + 1, 28, 28))
        shifts = np.zeros((FEATURE["shown"], steps, 2), int)
        starts = np.zeros((seeds, 28, 28))
        for seed in range(seeds):
            rng = np.random.default_rng(SEED + seed)
            starts[seed] = rng.uniform(0.2, 0.8, (28, 28))
            for digit in range(10):
                for run, regularise in enumerate((False, True)):
                    keep = seed == 0 and digit < FEATURE["shown"]
                    image, trace, film, moved = feature_ascent(
                        params, digit, starts[seed], regularise, rng, keep)
                    finals[run, seed, digit], traces[run, seed, digit] = image, trace
                    if keep:
                        frames[run, digit] = film
                        if regularise:
                            shifts[digit] = moved
        flat = finals.reshape(-1, 784)
        said = mnist._forward(judge, flat)[2].reshape(2, seeds, 10, 10)
        own = cnn_logits(params, flat).reshape(2, seeds, 10, 10)
        unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)  # noqa: E731
        centred = lambda v: v - v.mean(-1, keepdims=True)  # noqa: E731
        likeness = np.einsum("rsdk,dk->rsd", unit(centred(finals.reshape(2, seeds, 10, 784))),
                             unit(centred(means.reshape(10, 784))))
        return dict(
            finals=finals, traces=traces, frames=frames, shifts=shifts, starts=starts,
            judge=said, judge_right=said.argmax(-1) == np.arange(10),
            cnn_right=own.argmax(-1) == np.arange(10), likeness=likeness,
            blank=blank, real_best=best, real_best_value=np.take_along_axis(real, best, 1),
            real_best_images=test[best].reshape(10, 4, 28, 28),
            real_best_labels=labels[best],
            real_class_mean=np.asarray([real[c, labels == c].mean() for c in range(10)]),
            class_means=means, accuracy=np.asarray(accuracy),
            judge_accuracy=np.asarray(judge_accuracy),
            judge_on_real=np.asarray(np.mean(mnist._forward(judge, test)[2].argmax(1) == labels)))
    return cached_npz("study-interp-features", _settings("features", **FEATURE), build)


# ---------------------------------------------------------------------- 47 induction heads
INDUCTION = dict(width=64, heads=4, context=48, repeat=(8, 24), updates=800, batch=32, lr=0.003,
                 seeds=3, every=20, heldout=128)


def read_twice(source, starts, lengths, context):
    """Windows of ``context + 1`` characters, each a stretch of real text read over and
    over: window ``i`` is ``source[starts[i]:starts[i] + lengths[i]]`` repeated to fill
    it. The stretch length differs from window to window, so where the earlier copy sits
    cannot be learnt as a fixed distance: it has to be found by what the text says."""
    starts, lengths = np.asarray(starts), np.asarray(lengths)
    return source[starts[:, None] + np.arange(context + 1)[None] % lengths[:, None]]


def aligned_successors(t, length):
    """The positions holding what followed the current character on each earlier
    reading: one past each earlier aligned copy of position ``t``."""
    return np.arange(t - length + 1, 0, -length)[::-1]


def induction_scores(attention, lengths):
    """Per head, the mean share of attention that goes from a character of a later
    reading to the character that followed it on an earlier reading; and the mean share
    on the character just before the current one. ``attention``: (layer, window, head,
    query, key)."""
    layers, windows, heads, n, _ = attention.shape
    copy, previous = np.zeros((layers, heads)), np.zeros((layers, heads))
    for i, length in enumerate(lengths):
        rows = [attention[:, i, :, t, aligned_successors(t, length)].sum(0)
                for t in range(length, n)]
        copy += np.mean(rows, axis=0)
        q = np.arange(1, n)
        previous += attention[:, i][:, :, q, q - 1].mean(-1)
    return copy / windows, previous / windows


def induction_data():
    """Does a two-layer attention-only character model grow an induction head?

    Six models of the same size are trained from scratch on Tiny Shakespeare, three
    seeds on each of two kinds of window: ordinary 48-character windows (the control),
    and windows in which a stretch of 8 to 24 characters of the training text is read
    over and over. All are measured, every 20 updates, on the same held-out windows cut
    from the test split: the loss on a stretch's first reading, the loss on its later
    readings, the loss on ordinary text, and each head's attention to "what followed
    this character last time". GPT-2's heads are scored on natural repeats as before."""
    from . import next_language as NL
    from . import tinylm as lm

    def build():
        c0 = INDUCTION
        n, lo, hi = c0["context"], *c0["repeat"]
        train, test = (realdata.encode(realdata.text(k)) for k in ("train", "test"))
        text = realdata.text("test")
        rng = np.random.default_rng(SEED + 1)
        # held-out stretches start at the start of a line, so that they can be read
        lines = np.asarray([0] + [i + 1 for i, ch in enumerate(text[:-hi - n - 2]) if ch == "\n"])
        held_at = rng.choice(lines, c0["heldout"], replace=False)
        held_len = rng.integers(lo, hi + 1, c0["heldout"])
        held = read_twice(test, held_at, held_len, n)
        plain = test[rng.choice(len(test) - n - 1, c0["heldout"], replace=False)[:, None]
                     + np.arange(n + 1)]
        later = np.arange(n)[None] >= held_len[:, None] - 1    # the answer was read before
        config = lm.Config(vocab=len(realdata.vocabulary()), width=c0["width"],
                           heads=c0["heads"], layers=2, ff=4)

        def nll(P, seq):
            z = lm.forward(P, seq[:, :-1], config)
            z = z - z.max(-1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(-1, keepdims=True))
            rows = np.arange(len(seq))[:, None]
            return -logp[rows, np.arange(n), seq[:, 1:]], z.argmax(-1) == seq[:, 1:]

        def measure(P):
            loss, right = nll(P, held)
            _, rec = lm.forward(P, held[:, :-1], config, keep=True)
            attention = np.asarray(rec["attention"])
            copy, previous = induction_scores(attention, held_len)
            return dict(first=loss[~later].mean(), later=loss[later].mean(),
                        right=right[later].mean(), plain=nll(P, plain)[0].mean(),
                        copy=copy, previous=previous), attention, np.exp(-loss)

        runs = {}
        for kind in ("twice", "plain"):
            for seed in range(c0["seeds"]):
                P = lm.init(config, SEED + seed)
                for layer in range(2):                         # attention only: no MLP
                    P["l%d.W2" % layer][:] = 0
                opt = Adam(P, c0["lr"])
                rng = np.random.default_rng(SEED + seed)
                history = []
                for step in range(c0["updates"] + 1):
                    if step % c0["every"] == 0:
                        m, attention, p = measure(P)
                        history.append(m)
                    if step == c0["updates"]:
                        break
                    if kind == "twice":
                        seq = read_twice(train, rng.integers(0, len(train) - hi, c0["batch"]),
                                         rng.integers(lo, hi + 1, c0["batch"]), n)
                    else:
                        seq = train[rng.integers(0, len(train) - n - 1, c0["batch"])[:, None]
                                    + np.arange(n + 1)]
                    _, g = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], config)
                    for layer in range(2):
                        g["l%d.W1" % layer][:] = 0
                        g["l%d.W2" % layer][:] = 0
                    opt.step(P, g)
                runs[kind, seed] = (history, attention, p)
        out = {}
        for key in ("first", "later", "right", "plain", "copy", "previous"):
            out[key] = np.asarray([[[m[key] for m in runs[kind, seed][0]]
                                    for seed in range(c0["seeds"])] for kind in ("twice", "plain")])
        # one held-out window to draw: the first whose stretch is 12 to 14 characters
        shown = int(np.where((held_len >= 12) & (held_len <= 14))[0][0])
        out.update(
            steps=np.arange(0, c0["updates"] + 1, c0["every"]), held=held, held_len=held_len,
            shown=np.asarray(shown),
            attention=np.asarray([runs[kind, 0][1][:, shown] for kind in ("twice", "plain")]),
            p_true=np.asarray([runs[kind, 0][2][shown] for kind in ("twice", "plain")]))
        # GPT-2, as the first study measured it: natural repeats in 220 held-out characters
        sample = text[:220]
        ids = _tok().encode(sample)
        attn = np.asarray(NL._forward(sample, record=True)["attn"])
        rows = []
        for t in range(2, len(ids)):
            js = [j + 1 for j in range(t - 1) if ids[j] == ids[t]]
            if js:
                rows.append(attn[..., t, js].sum(-1))
        out["gpt2_mass"] = np.mean(rows, axis=0)
        return out
    return cached_npz("study-interp-induction", _settings("induction", **INDUCTION), build)


# ---------------------------------------------------------------------- the four studies
def _panel(title, caption, kind, notes=(), **data):
    return Panel(title=title, caption=caption, kind=kind, data=data, notes=tuple(notes))


READABLE = 0.8          # a feature "fires on one word": this share of its top firings agree


def sae_summary(d):
    """What the held-out text says about the dictionary, and which features are shown.

    The four shown are chosen by rule, not by eye: of the features whose 20 strongest
    held-out firings are at least 80% one word, the ones that fire most often, least
    often and in the middle; and, for contrast, the feature whose share is the median
    of all the scored features."""
    fires, purity, scored = d["fires"], d["purity"], d["scored"].astype(bool)
    readable = np.where(scored & (purity >= READABLE))[0]
    readable = readable[np.argsort(-fires[readable], kind="stable")]
    pool = np.where(scored)[0]
    typical = pool[np.argsort(purity[pool], kind="stable")[len(pool) // 2]]
    picks = [readable[0], readable[len(readable) // 2], readable[-1], typical]
    return dict(
        dead=float(np.mean(fires == 0)), dead_train=float(np.mean(d["train_fires"] == 0)),
        alive=int(np.sum(fires > 0)), scored=int(scored.sum()),
        readable=int(len(readable)), readable_share=float(len(readable) / scored.sum()),
        after_share=float(np.mean(d["before"][scored] >= READABLE)),
        either_share=float(np.mean((purity[scored] >= READABLE)
                                   | (d["before"][scored] >= READABLE))),
        median_purity=float(np.median(purity[scored])), chance=float(d["chance"]),
        busy=int(np.sum(fires > 0.1 * int(d["heldout_tokens"]))),
        active=float(np.mean(d["h"] > 0)), picks=[int(j) for j in picks])


def sae_study():
    d = sae_data()
    m = sae_summary(d)
    k = SAE["k"]
    counts = np.histogram(d["purity"][d["scored"].astype(bool)],
                          bins=[0, 0.2, 0.4, 0.6, 0.8, 1.01])[0]
    return Study(
        "Sparse autoencoder features",
        "GPT-2 layer six · top-k autoencoder · measured on held-out Shakespeare",
        [
            _panel("Training the dictionary",
                   "Reconstruction error (share of variance) per update on %d training tokens."
                   % int(d["train_tokens"]), "curve",
                   x=np.arange(len(d["losses"])), ys=np.asarray(d["losses"])[None],
                   labels=["training error"]),
            _panel("Reconstruction against sparsity",
                   "One autoencoder per k (features allowed to fire per token); held-out text.",
                   "curve", x=np.asarray(d["sweep_k"], float),
                   ys=np.stack([d["sweep_unexplained"], d["sweep_dead"]]),
                   labels=["variance left unexplained", "share of features silent"]),
            _panel("How many features read as one word",
                   "Of a feature's 20 strongest held-out firings, the share on its commonest"
                   " token; features counted per band.", "bars",
                   values=counts.astype(float),
                   labels=["under 20%", "20-40%", "40-60%", "60-80%", "80% or more"],
                   notes=("20 random held-out tokens agree on %.0f%%" % (100 * m["chance"]),)),
            _panel("Held-out summary",
                   "The k = %d autoencoder on %d held-out tokens." % (k, int(d["heldout_tokens"])),
                   "bars",
                   values=np.asarray([float(d["unexplained"]), float(d["train_unexplained"]),
                                      m["dead"], m["readable_share"]]),
                   labels=["held-out variance unexplained", "training variance unexplained",
                           "features silent on held-out text", "features firing on one word"]),
        ],
        metadata=_text_meta(
            source="GPT-2 activations of collected TinyShakespeare",
            settings=dict(SAE, sweep=list(SAE["sweep"]), resample=list(SAE["resample"]),
                          readable_threshold=READABLE),
            train_residuals=int(d["train_tokens"]), heldout_residuals=int(d["heldout_tokens"]),
            first_token_norm_ratio=float(d["first_norm"] / d["other_norm"]),
            resampled=[int(v) for v in d["revived"]], summary=m,
            diagnosis="The first study trained 180 updates on 926 tokens that included each"
                      " passage's first token (activation norm 38 times the others) with an L1"
                      " penalty of 2.0: 521 of 896 features never fired and 125 locked on to"
                      " the first token.",
            designs_tried=["top-k autoencoder, tied initialisation, dead-feature resampling,"
                           " first tokens excluded, 32,256 training tokens (the only design"
                           " run; first at 2,500 updates in float64, then 2,000 in float32"
                           " for the time limit, with the same conclusions)"],
            limitations=["'Fires on one word' is a lexical proxy for readable; it misses"
                         " features about context and says nothing about what a feature does."]))


def ablation_study():
    d = ablation_data()
    m = ablation_summary(d)
    tok = _tok()
    return Study(
        "Head ablation",
        "GPT-2 · 144 heads zeroed one at a time · %d held-out Shakespeare passages"
        % len(d["ids"]),
        [
            _panel("Loss with every head on",
                   "Mean next-token loss over each passage's %d predictions."
                   % ABLATION["tokens"], "bars",
                   values=d["base"].mean(1), labels=[str(i) for i in range(len(d["base"]))]),
            _panel("Mean loss change without each head",
                   "Layer by head; nats per token, averaged over the passages.", "matrix",
                   values=m["mean"]),
            _panel("Standard error of that mean",
                   "Spread between passages divided by the square root of their number.",
                   "matrix", values=m["sem"]),
            _panel("Two halves of the passages agree",
                   "Each head's mean change on the even-numbered passages against the"
                   " odd-numbered ones.", "points", xy=np.c_[m["even"].ravel(), m["odd"].ravel()],
                   notes=("Spearman rank correlation %.2f" % m["rank_correlation"],)),
        ],
        metadata=_text_meta(
            source="Collected TinyShakespeare and pretrained GPT-2",
            settings=dict(ABLATION, ablation="zero", loss="mean over all positions",
                          passages=[tok.decode([int(i) for i in row]) for row in d["ids"]]),
            baseline_loss=float(d["base"].mean()),
            rank_correlation=m["rank_correlation"], pearson=m["pearson"],
            random_halves=dict(mean=float(m["random_halves"].mean()),
                               min=float(m["random_halves"].min()),
                               max=float(m["random_halves"].max()), count=200),
            single_passage_rank_correlation=float(np.median(m["single"])),
            heads_beyond_two_standard_errors=int(m["clear"].sum()),
            diagnosis="The first study ranked 144 heads from two prompts, one target token"
                      " each; on one GPT-2 was confidently wrong (loss 18.67), and the same"
                      " head scored +3.66 on one prompt and -4.94 on the other.",
            designs_tried=["zero ablation on held-out windows of 16 tokens, loss averaged over"
                           " all 16 predictions (the only design run; 96 windows first, cut to"
                           " 64 for the time limit before the 96-window result was used)"],
            limitations=["Zeroing a head also removes its average output, which the rest of"
                         " the model expects; a head whose removal lowers the loss on this"
                         " text is not shown to be harmful in general."]))


def feature_summary(d):
    """The counts and means the animation states, over 10 digits x 3 starting noises."""
    out = {}
    for run, name in enumerate(("plain", "regularised")):
        hit = d["judge"][run][:, np.arange(10), np.arange(10)]
        out[name] = dict(
            objective=float(d["traces"][run, :, :, -1].mean()),
            judge_right=int(d["judge_right"][run].sum()),
            judge_right_per_seed=[int(v) for v in d["judge_right"][run].sum(1)],
            judge_probability=float(hit.mean()),
            likeness=float(d["likeness"][run].mean()),
            likeness_per_seed=[float(v) for v in d["likeness"][run].mean(1)],
            ink=float(d["finals"][run].mean()))
    out["trials"] = int(d["judge_right"][0].size)
    out["real_best"] = float(d["real_best_value"][:, 0].mean())
    out["real_typical"] = float(d["real_class_mean"].mean())
    return out


def feature_study():
    d = feature_data()
    m = feature_summary(d)
    steps = np.arange(FEATURE["steps"] + 1)
    names = [str(c) for c in range(10)]
    return Study(
        "Feature visualisation",
        "MNIST CNN · images optimised to excite each digit's output unit",
        [
            _panel("Plain gradient ascent",
                   "From noise, %d steps uphill on the unit: the image each digit's unit ends"
                   " with." % FEATURE["steps"], "images", values=d["finals"][0, 0], labels=names),
            _panel("With jitter, blur and decay",
                   "The same climb, shifted a few pixels each step, gradient blurred, pulled"
                   " toward blank paper.", "images", values=d["finals"][1, 0], labels=names),
            _panel("The real digits each unit likes best",
                   "Of the 10,000 test images, the one scoring highest on each unit.", "images",
                   values=d["real_best_images"][:, 0], labels=names),
            _panel("The unit's response through the climb",
                   "Mean over ten units and %d starting noises; a typical real digit of the"
                   " unit's class scores %.1f." % (FEATURE["seeds"], m["real_typical"]), "curve",
                   x=steps, ys=np.stack([d["traces"][0].mean((0, 1)), d["traces"][1].mean((0, 1))]),
                   labels=["plain", "regularised"],
                   notes=("A second network reads the intended digit in %d of %d plain images"
                          " and %d of %d regularised ones" % (
                              m["plain"]["judge_right"], m["trials"],
                              m["regularised"]["judge_right"], m["trials"]),)),
        ],
        metadata=dict(
            seed=SEED, study_version=VERSION, source="MNIST and trained repository CNN",
            dataset="MNIST", settings=dict(FEATURE, objective="class score minus mean of the"
                                           " other nine", judge="repository MLP (784-64-32-10)"),
            cnn_accuracy=float(d["accuracy"]), judge_accuracy=float(d["judge_accuracy"]),
            summary=m,
            diagnosis="The first study maximised the mean response of a first-layer 3 x 3"
                      " channel. That channel answers its bias on blank paper, so the optimum"
                      " is not an image of anything, the response stayed at 0.037, and the"
                      " regularisers (tuned for nothing) erased the few pixels that fired.",
            designs_tried=["class unit, normalised steps of 0.02, jitter 2 px, gradient blur"
                           " sigma 1, decay 0.02 (kept: the first design run)",
                           "the same with jitter 3 px, blur sigma 1.5, decay 0.03, one seed:"
                           " weaker (the judge read 7 of 10, likeness 0.20); not used"],
            limitations=["The network has one convolution layer; its units want stroke"
                         " fragments in the right places, not well-formed digits."]))


def induction_summary(d):
    """Final numbers per training set and seed, and when the head appears."""
    out = {}
    for kind, name in enumerate(("twice", "plain")):
        best = d["copy"][kind, :, :, 1].max(-1)                  # seed, evaluation
        out[name] = dict(
            later_loss=[float(v) for v in d["later"][kind, :, -1]],
            first_loss=[float(v) for v in d["first"][kind, :, -1]],
            later_right=[float(v) for v in d["right"][kind, :, -1]],
            plain_loss=[float(v) for v in d["plain"][kind, :, -1]],
            best_head=[float(v) for v in best[:, -1]],
            previous_token_head=[float(v) for v in d["previous"][kind, :, -1, 0].max(-1)])
    rise = d["copy"][0, :, :, 1].max(-1) >= 0.3
    out["appears_at"] = [int(d["steps"][np.argmax(row)]) if row.any() else -1 for row in rise]
    return out


def induction_study():
    d = induction_data()
    m = induction_summary(d)
    c = INDUCTION
    seeds = c["seeds"]
    best = d["copy"][:, :, :, 1].max(-1)                          # kind, seed, evaluation
    return Study(
        "Induction heads",
        "Two-layer attention-only character models on Shakespeare · GPT-2's heads",
        [
            _panel("Held-out stretches read over and over",
                   "Each row is a stretch of the test split, 8 to 24 characters, repeated to"
                   " fill 49 characters.", "matrix", values=d["held"][:8]),
            _panel("Loss on a stretch's later readings",
                   "Held-out nats per character, mean of %d seeds; the control was trained on"
                   " ordinary windows." % seeds, "curve", x=d["steps"],
                   ys=np.stack([d["later"][0].mean(0), d["first"][0].mean(0),
                                d["later"][1].mean(0)]),
                   labels=["later readings", "first reading", "control: later readings"],
                   seeds=d["later"]),
            _panel("The best layer-2 head's attention to what came next last time",
                   "Share of attention on the character that followed the current one on an"
                   " earlier reading; one curve per seed.", "curve", x=d["steps"],
                   ys=best.reshape(2 * seeds, -1),
                   labels=["read twice, seed %d" % i for i in range(seeds)]
                   + ["control, seed %d" % i for i in range(seeds)]),
            _panel("GPT-2 natural-text probe",
                   "Natural token repeats are evaluated separately; an attention pattern alone"
                   " does not prove induction.", "matrix", values=d["gpt2_mass"]),
        ],
        metadata=_text_meta(
            source="Collected TinyShakespeare and pretrained GPT-2",
            settings=dict(c, repeat=list(c["repeat"]), layers=2, mlp="none"),
            summary=m,
            diagnosis="The first study trained on ordinary 31-character windows for 650"
                      " updates. Ordinary text rarely repeats a long stretch inside so short a"
                      " window, so copying from an earlier occurrence earns almost nothing and"
                      " no head learns it (best score 0.11); the control here confirms it.",
            designs_tried=["real stretches read repeatedly, stretch length 8 to 24, context 48,"
                           " width 64, 4 heads per layer, 800 updates, 3 seeds (kept)",
                           "ordinary 48-character windows, same model and updates (the control"
                           " shown): no induction head",
                           "half ordinary windows and half repeated stretches, one seed, 2,000"
                           " updates: heads form more slowly (about 1,250 updates); not used"],
            limitations=["The repeats are constructed: real text, but read twice on purpose."
                         " In ordinary windows this model grows no induction head."]))


STUDIES = {
    "sparse-autoencoder-features": sae_study,
    "induction-heads": induction_study,
    "head-ablation": ablation_study,
    "feature-visualisation": feature_study,
}
