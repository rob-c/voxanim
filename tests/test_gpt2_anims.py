"""The GPT-2 family beyond attention: the tokenizer's merges, the embedding
tables, the logit lens and the sampling rules -- and every claim the four
animations' subtitles make."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim.sampling_anim import (
    entropy_bits,
    pick,
    renormalised,
    softmax_t,
    top_k_mask,
    top_p_mask,
)

# ------------------------------------------------------------------ the sampling rules (no model)


def test_softmax_temperature_sharpens_and_flattens():
    z = np.array([2.0, 1.0, 0.0, -1.0])
    p = softmax_t(z)
    assert np.isclose(p.sum(), 1) and np.all(np.diff(p) < 0)
    assert softmax_t(z, 0.5)[0] > p[0] > softmax_t(z, 2.0)[0]
    assert entropy_bits(softmax_t(z, 0.5)) < entropy_bits(p) < entropy_bits(softmax_t(z, 2.0))
    assert np.isclose(entropy_bits(np.full(8, 1 / 8)), 3.0)          # 8 equal choices: 3 bits
    assert entropy_bits(np.array([1.0, 0.0, 0.0])) == 0.0


def test_top_k_and_top_p_masks():
    p = np.array([0.4, 0.05, 0.3, 0.2, 0.05])
    assert top_k_mask(p, 2).tolist() == [True, False, True, False, False]
    assert top_k_mask(p, 5).all()
    # the nucleus: 0.4 (< 0.9), 0.4 + 0.3 (< 0.9), 0.4 + 0.3 + 0.2 = 0.9 reaches it
    assert top_p_mask(p, 0.9).tolist() == [True, False, True, True, False]
    assert top_p_mask(p, 0.3).tolist() == [True, False, False, False, False]
    assert top_p_mask(p, 1.0).all()
    q = renormalised(p, top_k_mask(p, 2))
    assert np.isclose(q.sum(), 1) and np.isclose(q[0], 0.4 / 0.7) and q[1] == 0


def test_pick_is_inverse_cdf():
    p = np.array([0.2, 0.5, 0.3])
    assert [pick(p, u) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 0.999, 1.0)] == [0, 0, 1, 1, 2, 2, 2]


# ------------------------------------------------------------------ the tokenizer


@needs_gpt2
def test_bpe_trace_reproduces_the_tokenizer():
    from voxanim import gpt2
    tok = gpt2.Tokenizer()
    for text in ("The hippopotamus yawned", "GPT-2 tokenises unbelievably long words",
                 "antidisestablishmentarianism", " four four Four", "aé€ b"):
        ids = []
        for piece in tok.pieces(text):
            u = tok.units(piece)
            steps, parts = tok.bpe_trace(u)
            # replaying the steps gives the same parts as _bpe, and the ranks are real
            replay = list(u)
            for rank, i, left, right in steps:
                assert (replay[i], replay[i + 1]) == (left, right)
                assert tok.ranks[left, right] == rank
                replay[i:i + 2] = [left + right]
            assert replay == parts == tok._bpe(u)
            ids += [tok.enc[p] for p in parts]
        assert ids == tok.encode(text)
        assert tok.decode(ids) == text


@needs_gpt2
def test_tokenizer_animation_claims():
    from voxanim import bpe_anim, gpt2
    tok = gpt2.Tokenizer()
    a = bpe_anim.BPEAnimation()
    assert a.text == "The hippopotamus yawned"
    assert a.tokens == ["The", "Ġhipp", "opot", "amus", "Ġyawn", "ed"]
    assert a.ids == tok.encode(a.text) == [464, 18568, 43372, 25509, 40908, 276]
    assert len(a.merges) == 17 and a.n_bytes == 23 and len(a.pieces) == 3
    assert [m.rank for m in a.merges[:3]] == [2, 208, 33]       # h+e, T+he, Ġ+h
    assert a.n_rules == 50000
    # " four" and "four" are different tokens, and different rows of the table
    assert a.pair_ids == [tok.encode("four")[0], tok.encode(" four")[0]] == [14337, 1440]
    assert 0.63 < a.pair_cos < 0.65
    assert 40 <= a.duration <= 70
    # the merges are shown in the tokenizer's order within each piece
    for k, (steps, _parts) in enumerate(a.traces):
        assert [(m.rank, m.i) for m in a.merges if m.piece == k] == [(r, i) for r, i, _l, _r
                                                                     in steps]


# ------------------------------------------------------------------ the embeddings


@needs_gpt2
def test_embedding_animation_claims():
    from voxanim import embed_anim
    f = embed_anim.facts()
    assert f.word_id == 5822
    near = [w for w, _c in f.near]
    assert near[0] == " king" and near[1:4] == [" kings", " King", " queen"]
    assert all(np.diff([c for _w, c in f.near]) <= 1e-6)
    assert 0.72 < f.near[1][1] < 0.74 and 0.65 < f.near[3][1] < 0.67
    # king - man + woman: " king" itself first, then " queen" (the analogy, roughly)
    assert [w for w, _c in f.anear[:2]] == [" king", " queen"]
    assert 0.77 < f.anear[0][1] < 0.79 and 0.70 < f.anear[1][1] < 0.72
    # positions: alike when near, unrelated around 164 apart, most unlike at 780
    c = f.curve
    assert c[1] > 0.995 and 0.46 < c[100] < 0.48
    assert int(np.argmax(c < 0)) == 164 and int(np.argmin(c)) == 780 and -0.70 < c.min() < -0.68
    assert f.sim.shape == (1024, 1024, 3) and f.pos_img.shape == (1024, 768, 3)
    a = embed_anim.EmbeddingAnimation()
    assert 40 <= a.duration <= 70


# ------------------------------------------------------------------ the logit lens


@needs_gpt2
def test_block_states_and_logit_lens_agree_with_the_forward_pass():
    from voxanim import gpt2
    tr = gpt2.run("One, two, three,")
    W = gpt2.weights()
    st = gpt2.block_states(tr)
    for l in range(gpt2.N_LAYER):
        assert np.array_equal(st["x_in"][l], tr.resid[l][-1])
        assert np.array_equal(st["x_out"][l], tr.resid[l + 1][-1])
        assert np.allclose(st["x_in"][l] + st["attn"][l], st["x_mid"][l])
        assert np.allclose(st["x_mid"][l] + st["mlp"][l], st["x_out"][l])
        ln2 = gpt2.layer_norm(st["x_mid"][l], W["h.%d.ln_2.weight" % l], W["h.%d.ln_2.bias" % l])
        assert np.allclose(st["ln2"][l], ln2)
        assert np.allclose(st["hidden"][l], tr.mlp_hidden[l, -1])
    lens = gpt2.logit_lens(tr)
    assert lens.shape == (2 * gpt2.N_LAYER + 1, gpt2.N_VOCAB)
    assert np.allclose(lens.sum(1), 1)
    # the last reading is the model's own output
    assert np.allclose(lens[-1], gpt2.softmax(tr.logits[-1]))
    assert np.array_equal(gpt2.unembed(tr.resid[-1]), tr.logits)


@needs_gpt2
def test_block_animation_claims():
    from voxanim import block_anim, gpt2
    a = block_anim.TransformerBlockAnimation()
    p = a.p_target
    tok = gpt2.Tokenizer()
    assert a.tid == tok.encode(" four")[0]
    # " four" is below 0.1% through layer 9, becomes the top token at layer 10's
    # MLP (65%), peaks after layer 11 (93%) and ends at the model's 46%
    assert p[:19].max() < 0.001
    first_top = next(i for i, top in enumerate(a.top) if top[0][0] == " four")
    assert first_top == 20 and 0.64 < p[20] < 0.66
    assert 0.92 < p[22] < 0.94 and 0.46 < p[24] < 0.47
    assert np.isclose(p[24], a.tr.top[0][1])
    assert a.top[12][0][0] == " etc" and a.top[12][0][1] > 0.98     # layer 6: " etc"
    assert 5.0 < a.norms["x_in"][0] < 5.2 and 490 < a.norms["x_out"][-1] < 492
    assert 0.08 < a.hot[0] < 0.10
    assert 40 <= a.duration <= 70
    assert a.step_of(0.0) == 0 and a.step_of(a.Tm["final"][0]) == 4


# ------------------------------------------------------------------ sampling


@needs_gpt2
def test_sampling_animation_claims():
    from voxanim import sampling_anim as S
    f = S.facts()
    assert f.names[:3] == [" floor", " bed", " couch"]
    assert 0.075 < f.p[f.order[0]] < 0.077
    assert abs(f.entropy[0.5] - 3.46) < 0.02 and abs(f.entropy[1.0] - 7.81) < 0.02
    assert abs(f.entropy[1.5] - 11.67) < 0.02
    assert 0.42 < f.k_mass < 0.43
    assert f.nucleus == 728 and f.nucleus_contrast == 102
    assert "".join(f.gen_names) == " sofa and yawned a little"
    assert f.gen["kept"].tolist() == [728, 57, 652, 1, 67, 3]
    assert f.k_cache.shape == (11, 64) and f.scores.shape == (11,)
    # the generation is what the rules say it is
    for k in range(S.N_GEN):
        p = f.gen["probs"][k].astype(np.float64)
        m = S.top_p_mask(p, S.TOP_P)
        assert int(m.sum()) == f.gen["kept"][k]
        assert S.pick(S.renormalised(p, m), float(f.gen["u"][k])) == f.gen["ids"][k]
    a = S.SamplingAnimation()
    assert 40 <= a.duration <= 70
    assert a.temperature(0.0) == 1.0
    mid = sum(a.Tm["temp"]) / 2
    assert any(abs(a.temperature(t) - 0.5) < 1e-6 for t in np.linspace(*a.Tm["temp"], 200))
    assert any(abs(a.temperature(t) - 1.5) < 1e-6 for t in np.linspace(*a.Tm["temp"], 200))
    assert a.temperature(mid) > 0.5


@needs_gpt2
@pytest.mark.parametrize("name", ["gpt2-tokenizer", "gpt2-embeddings", "gpt2-transformer-block",
                                  "gpt2-sampling"])
def test_layout_is_clean(name):
    from voxanim.qa import check
    assert check(name, samples=24, verbose=True) == 0
