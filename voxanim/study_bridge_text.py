"""The experiment behind animation 180, ``markov-chain-to-rnn``: the bridge from counted
chains of characters (``measuring-entropy``, ``markov-chains``) to recurrent networks.

Everything is measured on Tiny Shakespeare, split as :mod:`voxanim.study_chance` splits
it (contiguous training, validation and test portions, rounded to line breaks):

* **Counted contexts.** A chain of order ``k`` is a table: for every ``k`` characters
  seen in the training split, how often each character followed them
  (:func:`voxanim.study_chance.fit_order`). With add-alpha smoothing (alpha chosen on the
  validation split from :data:`voxanim.study_chance.ALPHAS`) it is scored in bits per
  character on the test split and, with the same alpha, on the training split it was
  counted from. Training bits keep falling with ``k``; held-out bits fall, then rise, as
  more and more held-out contexts were never counted.
* **A recurrent network.** A small character-level GRU, plain numpy, seeded, trained on
  the same training split by truncated backpropagation through time (64 streams through
  the text, 64 characters per update, the state carried between updates), Adam with a
  cosine learning rate and gradient clipping. Its held-out bits per character are
  measured on the whole test split every :data:`EVERY` updates; nothing about it is
  chosen on the test split.
* **Why.** Every held-out position is put in a bucket by how often its context of the
  best order was counted in training; in each bucket, the table's bits and the network's.
  The table stores a count per (context, next character) it saw; the network has a fixed
  number of parameters, shared by every context. Both numbers are exact.
* **Perplexity.** ``2 ** bits``: the number of equally likely characters that would be as
  hard to guess.

Every score of both models is over the same test positions: every character of the test
split after its first :data:`WARM` (a network's state is warmed up on the :data:`WARM`
characters before each position it scores, which are real held-out text).
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import itertools

import numpy as np

from . import realdata
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study
from .study_chance import ALPHAS, alphabet, char_ids, context_keys, fit_order, order_bits

VERSION = 1
NAME = "markov-chain-to-rnn"
EVIDENCE = dict(
    exact="exact arithmetic: table sizes, parameter counts, 2^bits",
    measured="measured on Tiny Shakespeare: models fitted on the training split, scored "
             "in bits per character on the held-out test split",
)

ORDERS = tuple(range(9))          # counted contexts of 0..8 characters
WARM = 64                         # test characters before the first one scored
HIDDEN = 128                      # the GRU's state
STREAMS = 64                      # truncated BPTT: streams through the training text
WINDOW = 64                       # characters per stream per update
STEPS = 1200                      # updates
EVERY = 100                       # updates between held-out measurements
RATE = 5e-3                       # Adam's peak learning rate, cosine to zero
CLIP = 5.0                        # gradients rescaled to at most this norm
RNN_SEED = 180
BUCKETS = (0, 1, 10, 100)         # training count of a context: 0, 1-9, 10-99, 100+
EXAMPLE_TEXT = 24                 # held-out characters drawn before it
EXAMPLE_MIN = 10                  # the drawn position's best-order context: counted this often


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-text-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== the network
def sigmoid(x):
    return 0.5 + 0.5 * np.tanh(0.5 * x)


def gru_init(rng, n_chars, hidden=HIDDEN, dtype=np.float32):
    """A character GRU: ``Wx`` (a row of input weights per character, for the update,
    reset and candidate parts), ``Wh`` (the state's weights, the same three), ``b``,
    and the read-out ``Wy``, ``by``."""
    s = 1.0 / np.sqrt(hidden)
    return dict(Wx=(rng.standard_normal((n_chars, 3 * hidden)) * 0.1).astype(dtype),
                Wh=(rng.standard_normal((hidden, 3 * hidden)) * s).astype(dtype),
                b=np.zeros(3 * hidden, dtype),
                Wy=(rng.standard_normal((hidden, n_chars)) * s).astype(dtype),
                by=np.zeros(n_chars, dtype))


def n_params(P):
    return int(sum(np.asarray(v).size for v in P.values()))


def gru_count(n_chars, hidden):
    """The GRU's parameters: ``3 H (V + H + 1) + V (H + 1)``."""
    return 3 * hidden * (n_chars + hidden + 1) + n_chars * (hidden + 1)


def gru_step(P, ax, h):
    """One step: update gate ``z``, reset gate ``r``, candidate ``n``, and the new state
    ``(1 - z) n + z h``. ``ax``: the input's three parts (``Wx[c] + b``)."""
    H = h.shape[1]
    zr = sigmoid(ax[:, :2 * H] + h @ P["Wh"][:, :2 * H])
    z, r = zr[:, :H], zr[:, H:]
    rh = r * h
    n = np.tanh(ax[:, 2 * H:] + rh @ P["Wh"][:, 2 * H:])
    return (1 - z) * n + z * h, (z, r, rh, n)


def log_probs(P, hs):
    """Log-chances of every character after each state (natural log)."""
    lg = hs @ P["Wy"] + P["by"]
    lg = lg - lg.max(-1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(-1, keepdims=True))


def gru_loss_grad(P, x, y, h):
    """Mean cross-entropy (nats) of predicting ``y`` from ``x`` (both ``(B, T)``) with
    the state starting at ``h``, its gradient for every parameter (by backpropagation
    through the ``T`` steps; none flows into ``h``), and the final state."""
    B, T = x.shape
    H = P["Wh"].shape[0]
    V = P["Wy"].shape[1]
    ax = P["Wx"][x] + P["b"]
    hs = np.empty((B, T, H), P["Wh"].dtype)
    tape = []
    for t in range(T):
        hn, parts = gru_step(P, ax[:, t], h)
        tape.append((h, *parts))
        hs[:, t] = h = hn
    lp = log_probs(P, hs)
    loss = float(-np.take_along_axis(lp, y[..., None], -1).mean())
    d = np.exp(lp)
    np.put_along_axis(d, y[..., None], np.take_along_axis(d, y[..., None], -1) - 1, -1)
    d = (d / (B * T)).astype(P["Wy"].dtype)
    G = dict(Wy=hs.reshape(-1, H).T @ d.reshape(-1, V), by=d.sum((0, 1)))
    dhs = d @ P["Wy"].T
    Wzr, Wn = P["Wh"][:, :2 * H], P["Wh"][:, 2 * H:]
    dWh = np.zeros_like(P["Wh"])
    dax = np.empty((B, T, 3 * H), P["Wh"].dtype)
    carry = np.zeros((B, H), P["Wh"].dtype)
    for t in range(T - 1, -1, -1):
        hp, z, r, rh, n = tape[t]
        dh = dhs[:, t] + carry
        dn = dh * (1 - z) * (1 - n * n)
        dWh[:, 2 * H:] += rh.T @ dn
        drh = dn @ Wn.T
        dzr = np.concatenate([dh * (hp - n) * z * (1 - z), drh * hp * r * (1 - r)], 1)
        dWh[:, :2 * H] += hp.T @ dzr
        carry = dh * z + drh * r + dzr @ Wzr.T
        dax[:, t, :2 * H], dax[:, t, 2 * H:] = dzr, dn
    onehot = np.zeros((B * T, V), P["Wx"].dtype)
    onehot[np.arange(B * T), x.ravel()] = 1
    G.update(Wh=dWh, b=dax.sum((0, 1)), Wx=onehot.T @ dax.reshape(-1, 3 * H))
    return loss, G, h


def gru_bits(P, seq, warm=WARM, streams=256):
    """Bits for every character of ``seq`` after its first ``warm``: the text is cut into
    ``streams`` contiguous pieces run side by side, each read after the ``warm``
    characters before it, so every scored character has at least ``warm`` of real text
    behind it."""
    seq = np.asarray(seq, np.int64)
    H = P["Wh"].shape[0]
    n = len(seq) - warm
    L = -(-n // streams)
    starts = warm + np.arange(streams) * L
    idx = np.minimum(starts[:, None] - warm + np.arange(warm + L)[None], len(seq) - 1)
    xs = seq[idx]
    ax = P["Wx"][xs] + P["b"]
    h = np.zeros((streams, H), P["Wh"].dtype)
    bits = np.empty((streams, L))
    for t in range(warm + L - 1):
        h = gru_step(P, ax[:, t], h)[0]
        if t + 1 >= warm:
            lp = log_probs(P, h)
            bits[:, t + 1 - warm] = -np.take_along_axis(lp, xs[:, t + 1, None], -1)[:, 0]
    pos = starts[:, None] + np.arange(L)[None]
    keep = pos < len(seq)
    out = np.empty(n)
    out[pos[keep] - warm] = bits[keep] / np.log(2)
    return out


def clip(G, limit=CLIP):
    """Gradients rescaled to a norm of at most ``limit``; their norm before."""
    norm = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in G.values())))
    s = min(1.0, limit / max(norm, 1e-12))
    return {k: g * np.float32(s) if g.dtype == np.float32 else g * s for k, g in G.items()}, norm


def stream_batches(n, streams=STREAMS, window=WINDOW):
    """Truncated BPTT through a text of ``n`` characters: ``streams`` contiguous pieces
    read side by side, ``window`` characters per update. Yields ``(starts, fresh)``: the
    first position of each stream's window, and whether the state starts afresh (at the
    beginning of a pass through the text)."""
    per = (n - 1) // streams
    off = np.arange(streams) * per
    pos = 0
    while True:
        fresh = pos == 0
        yield off + pos, fresh
        pos += window
        if pos + window > per:
            pos = 0


def train_gru():
    """The network trained on the training split; held-out bits every :data:`EVERY`
    updates, the mean training bits of the batches between, the per-position held-out
    bits at the end, and its training bits measured as the held-out ones are."""
    train, test = char_ids("train"), char_ids("test")
    V = len(alphabet())
    rng = np.random.default_rng(RNN_SEED)
    P = gru_init(rng, V)
    opt = Adam(P, RATE)
    batches = stream_batches(len(train))
    h = np.zeros((STREAMS, HIDDEN), np.float32)
    steps, held, seen, window = [0], [float(gru_bits(P, test).mean())], [np.nan], []
    norms = []
    for it in range(1, STEPS + 1):
        starts, fresh = next(batches)
        if fresh:
            h = np.zeros_like(h)
        idx = starts[:, None] + np.arange(WINDOW)
        loss, G, h = gru_loss_grad(P, train[idx], train[idx + 1], h)
        G, norm = clip(G)
        norms.append(norm)
        opt.step(P, G, lr=float(RATE * 0.5 * (1 + np.cos(np.pi * it / STEPS))))
        window.append(loss / np.log(2))
        if it % EVERY == 0:
            steps.append(it)
            held.append(float(gru_bits(P, test).mean()))
            seen.append(float(np.mean(window)))
            window = []
    final = gru_bits(P, test)
    return dict(steps=np.array(steps), held=np.array(held), batch_bits=np.array(seen),
                final_bits=final, train_bits=np.array(gru_bits(P, train).mean()),
                grad_norms=np.array(norms), **{"P_" + k: v for k, v in P.items()})


# ====================================================================== the tables
def bucket_of(counts, edges=BUCKETS):
    """The bucket of each training count: 0 for never, then ``edges[i] <= c < edges[i+1]``."""
    return np.searchsorted(np.asarray(edges[1:]), np.asarray(counts), side="right")


def context_counts(model, seq, k, n, skip):
    """For every position of ``seq`` from ``skip``: how often its order-``k`` context was
    counted in training."""
    _, _, ctx_keys, ctx_counts = model
    q = context_keys(seq, k, n)[skip - k:]
    i = np.minimum(np.searchsorted(ctx_keys, q), len(ctx_keys) - 1)
    return np.where(ctx_keys[i] == q, ctx_counts[i], 0)


def position_bits(model, seq, k, n, alpha, skip):
    """Bits of every position of ``seq`` from ``skip`` under the smoothed table (the
    terms :func:`voxanim.study_chance.order_bits` averages)."""
    pair_keys, pair_counts, _, _ = model
    q = context_keys(seq, k, n)[skip - k:]
    i = np.minimum(np.searchsorted(pair_keys, q * n + seq[skip:]), len(pair_keys) - 1)
    c = np.where(pair_keys[i] == q * n + seq[skip:], pair_counts[i], 0)
    t = context_counts(model, seq, k, n, skip)
    return -np.log2((c + alpha) / (t + alpha * n))


def _tables_build():
    n = len(alphabet())
    train, val, test = (char_ids(s) for s in ("train", "validation", "test"))
    rows, val_bits, counts, bits = [], [], [], []
    for k in ORDERS:
        model = fit_order(train, k, n)
        by_alpha = [order_bits(model, val, k, n, a, WARM)[0] for a in ALPHAS]
        a = ALPHAS[int(np.argmin(by_alpha))]
        held, unseen = order_bits(model, test, k, n, a, WARM)
        rows.append([k, a, held, order_bits(model, train, k, n, a, WARM)[0],
                     order_bits(model, train, k, n, 0.0, WARM)[0], unseen,
                     len(model[2]), len(model[0])])
        val_bits.append(by_alpha)
        counts.append(context_counts(model, test, k, n, WARM))
        bits.append(position_bits(model, test, k, n, a, WARM))
    rows = np.array(rows)
    best = int(np.argmin(rows[:, 2]))
    # the example drawn: the first scored held-out position whose longest context was
    # never counted while its context of the best order was, at least EXAMPLE_MIN times
    ok = (counts[-1] == 0) & (counts[best] >= EXAMPLE_MIN)
    i = int(np.flatnonzero(ok)[0])
    return dict(rows=rows, val_bits=np.array(val_bits), best=np.array(best),
                best_bits=bits[best], best_counts=counts[best],
                example=np.array(i + WARM),
                example_counts=np.array([c[i] for c in counts]),
                example_bits=np.array([b[i] for b in bits]),
                sizes=np.array([len(train), len(val), len(test)]))


CORE = (fit_order, order_bits, context_keys)


@functools.lru_cache(maxsize=1)
def tables():
    """Counted chains of every order (cached): per order ``[k, alpha, held-out bits,
    training bits, unsmoothed training bits, unseen share, contexts, pairs]``."""
    return _cache("tables", _tables_build, (_tables_build, position_bits, context_counts),
                  orders=ORDERS, alphas=ALPHAS, warm=WARM, example=EXAMPLE_MIN,
                  dataset=realdata.TEXT_SHA256)


@functools.lru_cache(maxsize=1)
def network():
    """The trained GRU and its measurements (cached)."""
    return _cache("gru", train_gru, (train_gru, gru_init, gru_step, log_probs,
                                     gru_loss_grad, gru_bits, clip, stream_batches,
                                     sigmoid, Adam.step),
                  hidden=HIDDEN, streams=STREAMS, window=WINDOW, steps=STEPS, every=EVERY,
                  rate=RATE, clip=CLIP, seed=RNN_SEED, warm=WARM,
                  dataset=realdata.TEXT_SHA256)


def gru_params():
    r = network()
    return {k[2:]: r[k] for k in r if k.startswith("P_")}


# ====================================================================== the study
def buckets(counts, table_bits, rnn_bits):
    """Per bucket of training count: share of positions, the table's mean bits and the
    network's on those positions."""
    b = bucket_of(counts)
    out = []
    for i in range(len(BUCKETS)):
        m = b == i
        out.append([float(m.mean()), float(table_bits[m].mean()) if m.any() else np.nan,
                    float(rnn_bits[m].mean()) if m.any() else np.nan, int(m.sum())])
    return np.array(out)


def bucket_names():
    e = BUCKETS
    names = ["never"]
    for a, b in itertools.pairwise(e[1:]):
        names.append("%d-%d" % (a, b - 1) if b - 1 > a else "%d" % a)
    names.append("%d+" % e[-1])
    return names


def markov_chain_to_rnn():
    T, R = tables(), network()
    rows = T["rows"]
    n = len(alphabet())
    best = int(T["best"])
    k_best = ORDERS[best]
    P = gru_params()
    rnn_final = float(R["final_bits"].mean())
    by_bucket = buckets(T["best_counts"], T["best_bits"], R["final_bits"])
    names = bucket_names()
    uniform = float(np.log2(n))
    perp_labels = ["uniform", "order 0", "order 1", "order %d (best)" % k_best, "GRU"]
    perp_bits = np.array([uniform, rows[0, 2], rows[1, 2], rows[best, 2], rnn_final])
    params = n_params(P)
    contexts, pairs = int(rows[best, 6]), int(rows[best, 7])
    panels = (
        Panel("Counted contexts, order by order",
              "ys[0]: bits per character on the test split of a table of order x (counts "
              "of what followed each x characters in the training split, add-alpha "
              "smoothing, alpha chosen on validation); ys[1]: the same table, same alpha, "
              "on the training split it was counted from. unseen: share of test positions "
              "whose context was never counted.", "curve",
              dict(x=np.array(ORDERS), ys=np.array([rows[:, 2], rows[:, 3]]),
                   labels=["held-out", "training"], alpha=rows[:, 1], unseen=rows[:, 5],
                   contexts=rows[:, 6], pairs=rows[:, 7], plug_in=rows[:, 4])),
        Panel("A GRU on the same text",
              "ys[0]: held-out bits per character of a %d-unit GRU (whole test split) "
              "after x updates of truncated BPTT; ys[1]: the mean bits of the training "
              "batches since the last measurement (at 0: the untrained network, held-out "
              "on both). best_order: the best table's held-out bits." % HIDDEN, "curve",
              dict(x=R["steps"].astype(float),
                   ys=np.array([R["held"], np.where(np.isnan(R["batch_bits"]), R["held"],
                                                    R["batch_bits"])]),
                   labels=["held-out", "training batches"],
                   best_order=np.array(rows[best, 2]))),
        Panel("Where the table has no data",
              "held-out positions in buckets by how often their order-%d context was "
              "counted in training (%s); values[i] = [share of positions, the order-%d "
              "table's bits, the GRU's bits] on that bucket." % (k_best, ", ".join(names),
                                                                k_best),
              "matrix", dict(values=by_bucket[:, :3], labels=names,
                             columns=["share", "table bits", "GRU bits"],
                             positions=by_bucket[:, 3])),
        Panel("Perplexity, 2^bits",
              "2 to the held-out bits per character: as hard to guess as this many equally "
              "likely characters. sizes: entries the best table stores against the GRU's "
              "parameters (exact).", "bars",
              dict(values=2.0 ** perp_bits, labels=perp_labels, bits=perp_bits,
                   sizes=np.array([pairs, params]))),
    )
    gap_table = float(rows[best, 2] - rows[best, 3])
    ex, held_text = int(T["example"]), realdata.text("test")
    example = dict(position=ex, before=held_text[ex - EXAMPLE_TEXT:ex], next=held_text[ex],
                   contexts=[held_text[ex - k:ex] for k in ORDERS],
                   counts=[int(v) for v in T["example_counts"]],
                   table_bits=_floats(T["example_bits"]),
                   rnn_bits=float(R["final_bits"][ex - WARM]))
    return Study(
        "From a Markov chain to an RNN",
        "Bits per character on held-out Tiny Shakespeare: counted contexts of every order "
        "against a small recurrent network", panels,
        dict(
            source="tables counted and a GRU trained here in numpy on the Tiny Shakespeare "
                   "training split; every score on its test split",
            dataset="Tiny Shakespeare / char-rnn",
            dataset_provenance={k: realdata.text_metadata(k)
                                for k in ("train", "validation", "test")},
            evidence=EVIDENCE, characters=n, split_sizes=[int(v) for v in T["sizes"]],
            scored_positions=int(T["sizes"][2] - WARM), warm=WARM,
            orders=list(ORDERS), alpha=_floats(rows[:, 1]), held_out_bits=_floats(rows[:, 2]),
            training_bits=_floats(rows[:, 3]), plug_in_bits=_floats(rows[:, 4]),
            unseen_share=_floats(rows[:, 5]), contexts=[int(v) for v in rows[:, 6]],
            pairs=[int(v) for v in rows[:, 7]],
            best_order=k_best, best_bits=float(rows[best, 2]),
            best_training_bits=float(rows[best, 3]), table_gap=gap_table,
            best_contexts=contexts, best_pairs=pairs,
            possible_contexts=int(n ** k_best), possible_cells=int(n ** (k_best + 1)),
            rnn=dict(kind="GRU", hidden=HIDDEN, parameters=params,
                     formula="3H(V + H + 1) + V(H + 1)", streams=STREAMS, window=WINDOW,
                     steps=STEPS, every=EVERY, learning_rate=RATE, schedule="cosine to 0",
                     clip=CLIP, optimiser="Adam", seed=RNN_SEED, dtype="float32",
                     characters_seen=int(STEPS * STREAMS * WINDOW),
                     passes=float(STEPS * STREAMS * WINDOW / T["sizes"][0])),
            rnn_steps=[int(v) for v in R["steps"]], rnn_held_out=_floats(R["held"]),
            rnn_batch_bits=_floats(R["batch_bits"][1:]), rnn_final=rnn_final,
            rnn_training_bits=float(R["train_bits"]),
            rnn_gap=float(rnn_final - float(R["train_bits"])),
            rnn_first_below=next((int(s) for s, v in zip(R["steps"], R["held"])
                                  if v < rows[best, 2]), None),
            buckets=names, bucket_share=_floats(by_bucket[:, 0]),
            bucket_table_bits=_floats(by_bucket[:, 1]), bucket_rnn_bits=_floats(by_bucket[:, 2]),
            bucket_positions=[int(v) for v in by_bucket[:, 3]],
            example=example, uniform_bits=uniform,
            perplexity=dict(zip(perp_labels, _floats(2.0 ** perp_bits))),
            perplexity_bits=dict(zip(perp_labels, _floats(perp_bits))),
            limitations=[
                "one network, one seed, one size; trained for %d updates (%.1f passes "
                "through the training split), so its numbers are what this budget reached, "
                "not what a GRU can reach" % (STEPS, STEPS * STREAMS * WINDOW / T["sizes"][0]),
                "add-alpha smoothing is the simplest table; a table that backs off to "
                "shorter contexts would do better at high orders",
                "the test split is the last tenth of the file, other plays than the "
                "training split: every model, the order-1 table included, scores worse on "
                "it than on the text it learned from",
            ],
        ),
    )


STUDIES = {NAME: markov_chain_to_rnn}
