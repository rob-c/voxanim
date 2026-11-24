"""The numbers behind animations 194-196 (voxanim.study_blocks_d): BatchNorm and LayerNorm on
the course model's hidden layer, an embedding table looked up with the first batch's labels,
and multi-head attention on patches of real digits; with torch installed, the same weights in
torch.nn give the same numbers."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_d as D
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_the_two_norms_average_in_two_directions():
    rng = np.random.default_rng(0)
    h = rng.normal(2.0, 3.0, (8, 5))
    bn, ln = D.batchnorm(h, eps=0.0), D.layernorm(h, eps=0.0)
    np.testing.assert_allclose(bn.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(bn.std(0), 1, atol=1e-12)
    np.testing.assert_allclose(ln.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(ln.std(1), 1, atol=1e-12)
    assert np.abs(bn.mean(1)).max() > 0.01 and np.abs(ln.mean(0)).max() > 0.01
    np.testing.assert_allclose(D.layernorm(h.T, eps=0.0), D.batchnorm(h, eps=0.0).T)


def test_one_hot_times_a_table_is_a_lookup():
    table = D.embedding_table()
    assert table.shape == (10, 16) and table.dtype == np.float32
    np.testing.assert_array_equal(table, D.embedding_table())            # seeded
    y = np.array([3, 0, 9, 3])
    np.testing.assert_array_equal(D.one_hot(y) @ table.astype(float), table[y])
    assert D.one_hot(y).sum(1).tolist() == [1, 1, 1, 1]


def test_tokens_are_patches_and_the_weights_are_initialised_as_pytorch_does():
    img = np.arange(784.0)
    x = D.patches(img)
    assert x.shape == (5, 16)
    assert x[0, 0] == 12 * 28 + 4 and x[0, 4] == 13 * 28 + 4 and x[4, 3] == 12 * 28 + 23
    W = D.attention_weights()
    assert W["in_proj_weight"].shape == (48, 16) and W["out_proj_weight"].shape == (16, 16)
    assert np.abs(W["in_proj_weight"]).max() <= np.sqrt(6 / 64)
    assert np.abs(W["out_proj_weight"]).max() <= 0.25
    assert not W["in_proj_bias"].any() and not W["out_proj_bias"].any()
    assert sum(v.size for v in W.values()) == 4 * 16 * 16 + 4 * 16 == 1088


def test_attention_rows_add_up_to_1_and_identical_tokens_score_alike():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 5, 16))
    x[:, 3] = x[:, 2]
    r = D.attend(x, D.attention_weights())
    np.testing.assert_allclose(r["probs"].sum(-1), 1, atol=1e-12)
    np.testing.assert_allclose(r["scores"], r["probs"].mean(1))
    np.testing.assert_allclose(r["raw"][:, :, 2], r["raw"][:, :, 3])
    np.testing.assert_allclose(r["out"][:, 2], r["out"][:, 3])
    hd = 8
    np.testing.assert_allclose(r["raw"][0, 1, 4, 0],
                               r["q"][0, 4, hd:] @ r["k"][0, 0, hd:] / np.sqrt(hd))


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_normalisation_of_the_real_hidden_layer():
    s = D.normalisation().validate()
    m = s.metadata
    h = np.asarray(s.panels[0].data["values"])
    np.testing.assert_array_equal(h, course.activations(course.EPOCH_STEPS, 0)["h1"].astype(
        np.float32))
    assert m["shape"] == [64, 64] and m["printed"] == "torch.Size([64, 64])"
    bn, ln = np.asarray(s.panels[1].data["values"]), np.asarray(s.panels[2].data["values"])
    assert np.abs(bn.mean(0)).max() < 1e-9 and np.abs(ln.mean(1)).max() < 1e-9
    assert 0.999 < bn.std(0).min() <= bn.std(0).max() <= 1.0
    assert 0.999 < ln.std(1).min() <= ln.std(1).max() <= 1.0
    # each averages one way only: the other means stay apart
    assert m["bn_row_means_range"][1] - m["bn_row_means_range"][0] > 0.3
    assert m["ln_col_means_range"][1] - m["ln_col_means_range"][0] > 1.0
    c = m["col"]
    assert m["col_before"][0] == max(m["col_means"]) and np.isclose(m["col_before"][0],
                                                                      h[:, c].mean())
    assert np.isclose(m["row_before"][1], h[m["row"]].std())
    assert abs(m["col_after"][0]) < 1e-9 and round(m["col_after"][1], 2) == 1.0
    assert 0.4 < m["zeros"] < 0.55 and m["learned"] == 128
    assert m["code"] == ["bn = nn.BatchNorm1d(64)", "out = bn(h)", "ln = nn.LayerNorm(64)",
                         "out = ln(h)"]


@needs_mnist
def test_embedding_of_the_first_batch():
    s = D.embedding().validate()
    m = s.metadata
    table = np.asarray(s.panels[0].data["values"])
    vectors = np.asarray(s.panels[1].data["values"])
    y = course.first_batch()["y"]
    assert m["labels"] == [int(v) for v in y] and (m["first"], m["second"]) == (0, 8)
    np.testing.assert_array_equal(vectors, table[y])
    np.testing.assert_array_equal(D.one_hot(y) @ table, vectors)
    assert m["out_shape"] == [64, 16] and m["printed"] == "torch.Size([64, 16])"
    assert m["numbers"] == 160 == m["multiplications_one"] and m["by_zero"] == 144
    assert m["multiplications_batch"] == 64 * 160 and sum(m["uses"]) == 64
    assert m["uses"] == [int(v) for v in np.bincount(y, minlength=10)]
    np.testing.assert_array_equal(m["first_row"], table[0])
    assert y[7] == y[8] == y[9]                         # three equal labels, three equal rows


@needs_mnist
def test_attention_on_patches_of_real_digits():
    s = D.attention().validate()
    m = s.metadata
    assert m["x_shape"] == [64, 5, 16] and m["label"] == 0
    assert m["printed"] == ["torch.Size([64, 5, 16])", "torch.Size([64, 5, 5])"]
    b = course.first_batch()
    x = np.asarray(m["x"])
    np.testing.assert_array_equal(x[1], b["x"][0, 0, 12:16, 8:12].reshape(-1))
    assert m["blank"] == [2, 3]                          # the hole of the 0: paper only
    probs = np.asarray(m["probs"])
    np.testing.assert_allclose(probs.sum(-1), 1, atol=1e-12)
    np.testing.assert_allclose(np.asarray(m["scores"]), probs.mean(0))
    q, k = np.asarray(m["q"]), np.asarray(m["k"])
    i, j = m["query"], m["key"]
    assert np.isclose(m["dot"], q[i, :8] @ k[j, :8])
    assert np.isclose(m["raw_cell"], m["dot"] / np.sqrt(8))
    assert np.isclose(np.asarray(m["raw"])[0, i, j], m["raw_cell"])
    assert m["query_row"][j] == max(m["query_row"]) > 0.5
    assert m["params"] == 1088 and m["in_proj"] + m["out_proj"] == 1088
    assert m["code"][0] == "q = k = v = torch.randn(64, 5, 16)"


# ---------------------------------------------------------------------- the same in torch
@needs_mnist
def test_torch_gives_the_same_numbers():
    torch = pytest.importorskip("torch")
    from torch import nn
    s = D.normalisation()
    h = torch.tensor(np.asarray(s.panels[0].data["values"], np.float32))
    np.testing.assert_allclose(nn.BatchNorm1d(64)(h).detach().numpy(),
                               s.panels[1].data["values"], atol=1e-5)
    np.testing.assert_allclose(nn.LayerNorm(64)(h).detach().numpy(),
                               s.panels[2].data["values"], atol=1e-5)
    embed = nn.Embedding(10, 16)
    with torch.no_grad():
        embed.weight.copy_(torch.tensor(D.embedding_table()))
    y = torch.tensor(course.first_batch()["y"])
    np.testing.assert_array_equal(embed(y).detach().numpy(), D.embedding().panels[1].data["values"])
    W = D.attention_weights()
    attn = nn.MultiheadAttention(16, 2, batch_first=True)
    assert sum(p.numel() for p in attn.parameters()) == 1088
    with torch.no_grad():
        attn.in_proj_weight.copy_(torch.tensor(W["in_proj_weight"]))
        attn.in_proj_bias.zero_()
        attn.out_proj.weight.copy_(torch.tensor(W["out_proj_weight"]))
        attn.out_proj.bias.zero_()
    m = D.attention().metadata
    b = course.first_batch()
    x = torch.tensor(np.stack([D.patches(r) for r in b["x"].reshape(64, -1)]))
    out, scores = attn(x, x, x)
    np.testing.assert_allclose(out[0].detach().numpy(), m["out"], atol=1e-5)
    np.testing.assert_allclose(scores[0].detach().numpy(), m["scores"], atol=1e-6)
