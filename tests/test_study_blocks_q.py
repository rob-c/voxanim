"""The numbers behind animations 232-233 (voxanim.study_blocks_q): masks that make a model
ignore the padding, and sorting lines by length to pad less."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_text

from voxanim import study_blocks_q as Q
from voxanim import study_padding as P


def toy(seed=0, n=5, L=9):
    """A made-up padded batch of ids (no text needed): lengths 1..L, pads 0."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, L + 1, n)
    lengths[0] = L
    batch = np.zeros((n, L), np.int64)
    for i, k in enumerate(lengths):
        batch[i, :k] = rng.integers(1, 20, k)
    return batch, batch != 0, lengths


# ---------------------------------------------------------------------- no data needed
def test_the_pads_vector_is_zero_and_the_plain_mean_shrinks_exactly():
    t = Q.table(vocab=20)
    assert t.shape == (20, Q.DIM) and (t[0] == 0).all() and (t[1:] != 0).any(1).all()
    batch, mask, lengths = toy()
    vecs = Q.embed(batch, t)
    assert (vecs[~mask] == 0).all()
    plain, mean = Q.means(vecs, mask)
    L = batch.shape[1]
    assert np.allclose(plain * L, mean * lengths[:, None], rtol=1e-5, atol=1e-6)
    # the masked mean is the mean over the real characters only
    for i, k in enumerate(lengths):
        assert np.allclose(mean[i], vecs[i, :k].mean(0), atol=1e-6)
    assert Q.ratio_text(4, 50) == "4 ÷ 50 = 0.08" and Q.two(1.0) == "1.00"


def test_attention_gives_every_pad_key_exactly_zero():
    batch, mask, _lengths = toy(1)
    w = Q.attention(Q.embed(batch, Q.table(vocab=20)), mask)
    B, L = batch.shape
    assert w.shape == (B, L, L)
    assert np.allclose(w.sum(-1), 1.0)
    keys = np.broadcast_to(mask[:, None, :], w.shape)
    assert (w[~keys] == 0.0).all() and (w[keys] > 0).all()
    # different weights, the same zeros
    w2 = Q.attention(Q.embed(batch, Q.table(seed=7, vocab=20)), mask, Q.in_proj(seed=7))
    assert not np.allclose(w, w2) and (w2[~keys] == 0.0).all()


def test_attention_is_torchs():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    nn = torch.nn
    batch, mask, _lengths = toy(2)
    t = Q.table(vocab=20)
    vecs = Q.embed(batch, t)
    torch.manual_seed(0)
    attn = nn.MultiheadAttention(Q.DIM, Q.HEADS, batch_first=True)
    with torch.no_grad():
        attn.in_proj_weight.copy_(torch.from_numpy(Q.in_proj()))
        attn.in_proj_bias.zero_()
        v = torch.from_numpy(vecs)
        _out, w = attn(v, v, v, key_padding_mask=~torch.from_numpy(mask))
    w = w.numpy()
    ours = Q.attention(vecs, mask)
    assert np.allclose(w, ours, atol=1e-5)
    assert (w[~np.broadcast_to(mask[:, None, :], w.shape)] == 0).all()
    # and padding_idx=0 is a row of zeros in torch too
    embed = nn.Embedding(20, Q.DIM, padding_idx=0)
    assert (embed.weight[0] == 0).all()


def test_batches_and_their_padding():
    ln = np.array([3, 5, 5, 2, 7, 1, 1])
    assert Q.per_batch(ln, 3) == [(3, 5, 2), (3, 7, 11), (1, 1, 0)]
    r = Q.batch_rows(ln, 1, 3)
    assert r == dict(index=1, lengths=[2, 7, 1], longest=7, pads=11, share=pytest.approx(11 / 21))
    picks, order = Q.whole_batches(512)
    assert picks[0] == 0 and picks[-1] == 511 and len(picks) == Q.MINI
    assert sorted(order) == list(range(Q.MINI)) and all(o != i for i, o in enumerate(order))
    assert Q.pct(0.380171) == "38.0%" and Q.pct(0.0015940, 2) == "0.16%"


# ---------------------------------------------------------------------- the text
@needs_text
def test_masks():
    s = Q.masks().validate()
    m = s.metadata
    assert m["lengths"] == [14, 45, 4, 13, 14, 50, 4, 19] and m["shape"] == [8, 50]
    assert (m["real"], m["cells"], m["pads"]) == (163, 400, 237)
    assert m["follow_text"] == "All:" and m["follow_pads"] == 46
    assert m["follow_ratio"] == 0.08 and m["follow_ratio_text"] == "4 ÷ 50 = 0.08"
    assert m["ratio_texts"] == ["0.28", "0.90", "0.08", "0.26", "0.28", "1.00", "0.08", "0.38"]
    assert m["ratio"] == P.masked_mean()["ratio"]
    # the plain mean over the masked one is length / 50 in every one of the 16 numbers
    r = np.asarray(m["plain_over_mean"])
    assert np.allclose(r, np.asarray(m["lengths"])[:, None] / 50, rtol=1e-4)
    vecs = np.asarray(m["follow_vecs"])
    assert vecs.shape == (50, 16) and (vecs[4:] == 0).all() and (vecs[:4] != 0).all()
    # attention: w[1, 0] has 45 weights adding to 1 and exactly 0 on its 5 pads
    w = np.asarray(m["shown_weights"])
    assert (m["shown_length"], m["shown_pads"]) == (45, 5)
    assert (w[45:] == 0).all() and (w[:45] > 0).all() and m["shown_sum_text"] == "1.00"
    first = np.asarray(m["first_weights"])
    assert (first[~np.asarray(m["mask"])] == 0).all() and m["pad_weights_max"] == 0.0
    assert m["rows_add_to_one"]
    assert (m["loss_counted"], m["loss_skipped"]) == (163, 237)
    assert any("random" in s for s in m["limitations"])


@needs_text
def test_sort_by_length():
    s = Q.sort_by_length().validate()
    m = s.metadata
    w = P.waste()
    assert (m["lines"], m["batches"], m["full_batches"], m["last_batch"]) == (32_777, 513, 512, 9)
    assert m["share_text"] == dict(file="38.0%", shuffled="38.7%", sorted="0.16%")
    assert m["cells"] == {k: w[k]["cells"] for k in ("file", "shuffled", "sorted")}
    assert m["padding"]["sorted"] == 1_717 and m["cells_text"]["file"] == "1,734,987"
    assert m["cells"]["sorted"] - m["padding"]["sorted"] == m["characters"] == 1_075_394
    # the batches drawn are real: the first two in file order, as the file has them
    ln = Q.lengths()
    f = m["rows"]["file"]
    assert f[0]["lengths"] == list(ln[:64]) and f[1]["lengths"] == list(ln[64:128])
    assert (f[0]["longest"], f[1]["longest"]) == (59, 54)
    assert m["row_share_text"]["file"] == ["45%", "32%"]
    # the shuffle is waste()'s own
    assert np.array_equal(ln[Q.shuffled_order()], np.random.default_rng(
        [Q.course.SEED, 4, P.SEED]).permutation(ln))
    srt = m["rows"]["sorted"]
    assert [r["index"] for r in srt] == [171, 342]
    assert all(max(r["lengths"]) - min(r["lengths"]) <= 1 for r in srt)
    assert m["row_share_text"]["sorted"] == ["2%", "0%"]
    # the small batches: full ones, short to long, and each keeps its padding
    assert [len(r["lengths"]) for r in m["mini"]] == [64] * Q.MINI
    assert m["mini_longest"] == sorted(m["mini_longest"])
    assert any("not measured" in s for s in m["limitations"])
