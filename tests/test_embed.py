"""The embeddings family: the three studies' maths and gradients, and the
claims the ten animations make, as relative claims (CI retrains on another
machine, where the last bits differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_gpt2, needs_mnist

from voxanim import patch_embed as PE
from voxanim import posenc as PZ
from voxanim import w2v as W

# ------------------------------------------------------------------ word2vec


def test_sgns_gradients():
    """The skip-gram loss's gradients, scattered into the two tables, match
    finite differences."""
    rng = np.random.default_rng(0)
    V, D = 9, 5
    P = dict(i=rng.normal(0, 0.5, (V, D)), o=rng.normal(0, 0.5, (V, D)))
    c, o = np.array([1, 4, 4]), np.array([2, 2, 7])
    neg = rng.integers(0, V, (3, 4))

    def grads(P):
        loss, ri, gi, ro, go = W.sgns_grads(P["i"], P["o"], c, o, neg)
        G = dict(i=np.zeros((V, D)), o=np.zeros((V, D)))
        np.add.at(G["i"], ri, gi)
        np.add.at(G["o"], ro, go)
        return loss, G

    _, G = grads(P)
    fd_check(lambda Q: grads(Q)[0], P, G, ["i", "o"])


def test_corpus_and_pairs():
    """Sentences come from the templates; every pair is within the window."""
    words = W.vocabulary()
    index = {w: i for i, w in enumerate(words)}
    sents = W.corpus(200)
    assert all(w in index for s in sents for w in s)
    pr = W.pairs([["the", "cat", "runs", "to", "the", "city"]], index, window=2)
    assert len(pr) == 2 * (2 + 3 + 4 + 4 + 3 + 2) // 2


def test_word2vec_claims():
    """Categories gather: same-category words end far closer than words of
    different categories (they start alike), every word's nearest neighbours
    are of its category, and a step touches only the rows it used."""
    run = W.train()
    s0, d0 = W.category_cosines(run.snaps[0], run.words)
    s1, d1 = W.category_cosines(run.W_in, run.words)
    assert abs(s0 - d0) < 0.1
    assert s1 > 0.8 and d1 < 0.4 and s1 - d1 > 0.5
    for cat, ws in W.CATEGORIES.items():
        for w in ws:
            top = [n for n, _ in W.neighbours(run.W_in, run.words, w, k=3)]
            assert all(W.category_of(n) == cat for n in top), (w, top)
    assert run.loss[-500:].mean() < run.loss[:100].mean() - 0.8
    st = W.sgns_step(run, len(run.snaps) // 2, "cat", "chases", ["the", "bread", "to", "a",
                                                                   "school"])
    assert st["loss1"] < st["loss0"] and st["rows_in"] == [st["c"]]
    assert len(st["rows_out"]) <= 6


def test_sentence_claims():
    """Mean pooling cannot tell the two dog/cat sentences apart (the same
    vector, exactly); the two eating sentences are the closest other pair;
    SIF spreads the matrix out."""
    run = W.train()
    X = W.sentence_vectors(run)
    assert np.array_equal(np.sort(X[0]), np.sort(X[1])) or np.allclose(X[0], X[1], atol=1e-15)
    M = W.cosine_matrix(X)
    off = M.copy()
    np.fill_diagonal(off, -9)
    off[0, 1] = off[1, 0] = -9
    i, j = np.unravel_index(np.argmax(off), off.shape)
    assert {i, j} == {2, 3}
    Ms = W.cosine_matrix(W.sentence_vectors(run, sif=True))
    assert Ms[0, 1] == pytest.approx(1.0)
    assert Ms.min() < M.min() - 0.5
    assert Ms[2, 3] > 0.5


# ------------------------------------------------------------------ patch embeddings


@pytest.mark.parametrize("positions", [False, True])
def test_patch_gradients(positions):
    rng = np.random.default_rng(1)
    P = PE.init(rng, positions)
    P["b"] = rng.normal(0, 0.1, P["b"].shape)
    xp = rng.random((4, PE.N_PATCH, PE.P * PE.P))
    y = np.array([0, 3, 9, 3])
    _, G = PE.loss_grads(P, xp, y)
    fd_check(lambda Q: PE.loss_grads(Q, xp, y)[0], P, G, list(P))


def test_patches_round_trip():
    x = np.random.default_rng(2).random((3, 784))
    xp = PE.patches(x)
    assert np.array_equal(PE.unpatch(xp), x)
    assert np.array_equal(xp[0, 5], x[0].reshape(28, 28)[7:14, 7:14].ravel())


@needs_mnist
def test_patch_claims():
    """Without positions shuffling the patches changes no prediction; with
    them the model is more accurate and shuffling ruins it."""
    from voxanim.mnist import load
    _, _, xte, _ = load()
    bag, pos = PE.train(False), PE.train(True)
    xp = PE.patches(xte[:2000])
    assert np.array_equal(PE.predict(bag.P, xp), PE.predict(bag.P, xp[:, PE.shuffle_order()]))
    assert bag.shuffled_acc == pytest.approx(bag.test_acc, abs=1e-3)
    assert pos.test_acc > bag.test_acc + 0.05
    assert pos.shuffled_acc < pos.test_acc - 0.3


# ------------------------------------------------------------------ the encodings


def test_sinusoidal_shift_is_a_rotation():
    pe = PZ.sinusoidal(80, 16)
    for k in (1, 5, 17):
        assert np.abs(pe[:80 - k] @ PZ.shift_matrix(k, 16) - pe[k:]).max() < 1e-12


def test_rope_depends_on_offset_only():
    rng = np.random.default_rng(3)
    q, k = rng.normal(size=8), rng.normal(size=8)
    A = PZ.rope_angles(60, 8)

    def score(m, n):
        return PZ.rope(q[None], A[m:m + 1])[0] @ PZ.rope(k[None], A[n:n + 1])[0]

    assert score(3, 7) == pytest.approx(score(40, 44), abs=1e-12)
    assert score(3, 7) != pytest.approx(score(3, 10), abs=1e-6)
    assert PZ.ntk_base(4, 8) == pytest.approx(10000 * 4 ** (8 / 6))


def test_t5_buckets_and_alibi():
    b = PZ.t5_bucket(np.arange(300))
    assert list(b[:8]) == list(range(8))
    assert np.all(np.diff(b) >= 0) and b[-1] == 15 and b.max() == 15
    assert b[31] == 15                       # the last bucket is reached within training
    assert np.allclose(PZ.alibi_slopes(4), [1 / 4, 1 / 16, 1 / 64, 1 / 256])


def test_language():
    """Seeing the previous symbol is worth a lot; the sampler follows the table."""
    h2, h1, h0 = PZ.entropies()
    assert h2 < h1 - 0.5 < h0
    rng = np.random.default_rng(0)
    x = PZ.sample(400, 40, rng)
    T = PZ.markov_table()
    assert -np.log(T[x[:, :-2], x[:, 1:-1], x[:, 2:]]).mean() == pytest.approx(h2, abs=0.05)


@pytest.mark.parametrize("kind", PZ.KINDS)
def test_transformer_gradients(kind):
    rng = np.random.default_rng(4)
    P = PZ.init(kind, rng)
    if kind == "t5":
        P["bias"] = rng.normal(0, 0.5, P["bias"].shape)
    x = PZ.sample(3, 7, rng)
    _, G = PZ.loss_grads(kind, P, x)
    fd_check(lambda Q: PZ.loss_grads(kind, Q, x)[0], P, G, list(P), tol=1e-4)


def _means(tl):
    return tl[2:31].mean(), tl[31:63].mean(), tl[63:].mean()


def test_extrapolation_claims():
    """At the training length every scheme but none and ALiBi is near the
    best possible loss; past it learned, sinusoidal and RoPE collapse while
    T5 and ALiBi hold; interpolation hurts and NTK scaling helps (to 2×)."""
    h2, h1, _ = PZ.entropies()
    m = {r.kind: _means(r.test_loss) for r in PZ.runs()}
    for k in ("learned", "sinusoidal", "rope", "t5"):
        assert m[k][0] < h2 + 0.05
    assert m["alibi"][0] < h1 - 0.4 and m["none"][0] > m["alibi"][0]
    for k in ("learned", "sinusoidal", "rope"):
        assert m[k][2] > m[k][0] + 1.0
    for k in ("t5", "alibi"):
        assert m[k][2] < m[k][0] + 0.15
    assert m["t5"][2] < min(m[k][2] for k in PZ.KINDS if k != "t5")
    ext = {k: _means(v) for k, v in PZ.rope_extensions(PZ.train("rope")).items()}
    assert ext["interpolation"][0] > ext["plain"][0] + 0.5
    assert ext["ntk"][1] < ext["plain"][1] - 0.5
    assert ext["ntk"][0] < h2 + 0.1


@needs_gpt2
def test_position_blindness():
    """GPT-2's first layer without positions or mask gives a word the same
    vector in every order; positions or the mask tell orders apart, except
    that with the mask alone the last word sees the same set."""
    V = PZ.word_vectors(positions=False, causal=False)
    assert np.abs(V - V[0]).max() < 1e-9
    V = PZ.word_vectors(positions=True, causal=False)
    assert np.abs(V[1] - V[0]).max() > 0.1
    V = PZ.word_vectors(positions=False, causal=True)
    assert np.abs(V[2] - V[1]).max() < 1e-9 < np.abs(V[1] - V[0]).max()


# ------------------------------------------------------------------ what the animations show


def test_word2vec_step_at_the_exact_state():
    """The animated step is taken at a replay of training that matches the
    stored snapshot bit for bit (``replay`` raises otherwise): the true
    context's score rises, every noise word's falls, the pair's loss falls."""
    from voxanim import w2v_anim
    st = w2v_anim.training_step()
    assert st["s_pos1"] > st["s_pos0"]
    assert np.all(st["s_neg1"] < st["s_neg0"])
    assert st["loss1"] < st["loss0"]


def test_relative_bias_learns_previous_token_heads():
    """Every T5 head's largest bias is for offset 1, far above the last bucket's."""
    bias = PZ.train("t5").P["bias"]
    assert np.all(bias.argmax(1) == 1)
    assert bias[:, -1].max() < bias[:, 1].min() - 3


def test_alibi_heads():
    """ALiBi's heads look further back as their slope falls, and none gives the
    previous token as much weight as any T5 head does."""
    from voxanim.alibi_anim import head_stats
    xs = PZ.sample(64, PZ.TRAIN_LEN, np.random.default_rng(11))
    _, prev_a, look_a = head_stats("alibi", PZ.train("alibi").P, xs)
    _, prev_t, _ = head_stats("t5", PZ.train("t5").P, xs)
    assert np.all(np.diff(look_a) > 0)
    assert prev_a.max() < prev_t.min()


NAMES = ("embedding-lookup", "word2vec-training", "sentence-embeddings", "patch-embeddings",
         "gpt2-position-blindness", "sinusoidal-positions", "relative-position-bias", "alibi",
         "rope-context-extension", "length-extrapolation")


def _marks(name):
    from conftest import data_marks
    return data_marks(name)


@pytest.mark.parametrize("name", [pytest.param(n, marks=_marks(n)) for n in NAMES])
def test_words_are_finite(name):
    """No subtitle ever prints a nan or an inf, at any time."""
    import re

    from voxanim import presets
    anim = presets.build(name)
    for t in np.linspace(0, anim.duration, 90):
        anim.scene(float(t))
        assert not re.search(r"\b(nan|inf)\b", anim.subtitle.lower()), (t, anim.subtitle)
