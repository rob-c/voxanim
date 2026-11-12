"""The behaviour animations (hallucination, conditioning, calibration on GPT-2
small): the probability arithmetic, the calibration binning on hand-made
data, and every claim the subtitles make about the real numbers."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim import calib_anim as CA

# ------------------------------------------------------------------ arithmetic (no model needed)


def test_entropy_in_bits():
    from voxanim.gpt2_probe import entropy
    assert entropy([1.0, 0.0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy(np.full(50257, 1 / 50257)) == pytest.approx(np.log2(50257))


def test_softmax_sums_to_one_and_survives_large_logits():
    from voxanim.gpt2_probe import softmax
    p = softmax(np.array([1000.0, 999.0, -1000.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] / p[1] == pytest.approx(np.e)
    assert p[2] > 0 or p[2] == 0            # no NaN


def test_reliability_bins_hand_made():
    """Four items in two bins: the counts, mean confidences and accuracies."""
    conf = [0.1, 0.3, 0.9, 0.7]
    right = [0, 1, 1, 0]
    edges, count, mean_conf, acc = CA.reliability(conf, right, bins=2)
    assert list(edges) == [0.0, 0.5, 1.0]
    assert list(count) == [2, 2]
    assert mean_conf == pytest.approx([0.2, 0.8])
    assert acc == pytest.approx([0.5, 0.5])
    # count-weighted gaps: 0.5 * |0.5 - 0.2| + 0.5 * |0.5 - 0.8|
    assert CA.ece(conf, right, bins=2) == pytest.approx(0.3)


def test_reliability_bin_edges_and_empty_bins():
    """0 lands in the first bin, a value on an edge in the bin below it, and
    an empty bin reads 0 without dividing by zero."""
    _e, count, mean_conf, acc = CA.reliability([0.0, 0.2, 1.0], [1, 0, 1], bins=5)
    assert list(count) == [2, 0, 0, 0, 1]
    assert mean_conf[0] == pytest.approx(0.1) and acc[0] == pytest.approx(0.5)
    assert mean_conf[1] == 0 and acc[1] == 0
    assert CA.ece([0.5, 0.5], [1, 0], bins=1) == pytest.approx(0.0)


def test_calibration_list_is_forty_distinct_countries():
    assert len(CA.COUNTRIES) == 40
    assert len({c for c, _k in CA.COUNTRIES}) == 40
    assert len({k for _c, k in CA.COUNTRIES}) == 40


# ------------------------------------------------------------------ the model's numbers


@needs_gpt2
def test_probe_matches_the_recording_forward_pass():
    """The float32 probe gives the same logits and last-layer attention as
    :func:`voxanim.gpt2.run`."""
    from voxanim import gpt2
    from voxanim import gpt2_probe as P
    tr = gpt2.run("One, two, three,")
    logits, attn = P.forward(tr.ids, attention=True)
    assert np.abs(logits - tr.logits).max() < 1e-3
    assert np.abs(attn - tr.attn[-1]).max() < 1e-5
    p = P.next_probs(tr.ids)
    assert P.top_tokens(p, 1)[0][0] == " four"


@needs_gpt2
def test_capitals_are_single_tokens():
    from voxanim import gpt2_probe as P
    for country, capital in CA.COUNTRIES:
        assert P.token_id(" " + country) is not None, country
        assert P.token_id(" " + capital) is not None, capital


@needs_gpt2
def test_sampling_is_seeded_and_first_word_stops_at_punctuation():
    from voxanim import gpt2_probe as P
    ids = P.encode("The capital of Germany is Berlin. The capital of France is")
    a, pa = P.sample(ids, 3, seed=0)
    b, _pb = P.sample(ids, 3, seed=0)
    assert a == b and len(pa) == 3 and all(0 < q <= 1 for q in pa)
    assert P.first_word([P.token_id(" Paris"), P.token_id(".")]) == 1
    assert P.first_word(P.encode(" Marseille")) == 2       # " Marse" + "ille"


@needs_gpt2
def test_hallucination_claims():
    """The real prompt is sharp and self-consistent; the invented one is flat
    and names a different city every time; 'unknown' is nowhere near the top."""
    from voxanim import halluc_anim as H
    real, inv = H.trace()
    assert real.top[0][0] == " Paris" and real.top[0][1] > 0.6
    assert real.entropy < 3.0 < 8.0 < inv.entropy
    assert inv.top[0][1] < 0.2                      # no dominant answer
    assert real.mode[1] >= 4 and real.mode[0] == "Paris"
    assert inv.distinct == H.N_SAMPLES
    assert real.unknown_rank > 1000 and inv.unknown_rank > 1000
    assert real.unknown_p < 1e-4 and inv.unknown_p < 1e-4
    a = H.Hallucination()
    assert 40 <= a.duration <= 60
    assert "4 of 6 agree: Paris" in [tx.s for tx in a.scene(31.0).texts]


@needs_gpt2
def test_conditioning_claims():
    """p(Paris) rises with every piece of context, the first example does
    most of the work, the other examples send the same question to ' in',
    and the mean attention of the last layer sits mostly on the first token."""
    from voxanim import condition_anim as CN
    stages, other, attn = CN.trace()
    ps = [s.p_target for s in stages]
    assert all(b > a for a, b in itertools.pairwise(ps)), ps
    assert ps[0] < 0.05 and ps[2] > 0.5 and ps[-1] > 0.6
    assert ps[2] - ps[1] > sum(ps[k + 1] - ps[k] for k in (0, 2, 3))
    assert [s.n_tokens for s in stages] == [5, 13, 21, 29, 37]
    assert stages[0].top[0][0] == " the" and stages[-1].top[0][0] == " Paris"
    assert other.top[0][0] == " in" and other.top[0][1] > 0.9 and other.p_target < 0.01
    assert attn.shape == (12, 37)
    assert np.allclose(attn.sum(1), 1.0, atol=1e-4)
    assert attn.mean(0)[0] > 0.5                    # the attention sink
    head = CN.broadest_head(attn)
    assert attn[head][0] < 0.05                     # the broadest head skips the sink
    a = CN.Conditioning()
    assert 40 <= a.duration <= 60


@needs_gpt2
def test_calibration_claims():
    """Accuracy, ECE and the held-up pair are what the subtitles say."""
    items = CA.trace()
    assert len(items) == 40
    conf = np.array([i.conf for i in items])
    right = np.array([i.right for i in items], float)
    assert right.mean() == pytest.approx(0.8)
    assert 0.2 < CA.ece(conf, right) < 0.3
    _e, count, mean_conf, acc = CA.reliability(conf, right)
    assert count.sum() == 40
    occupied = count > 0
    assert np.all(acc[occupied][1:] > mean_conf[occupied][1:])   # under-confident above 0.2
    r, w = CA.same_number(items)
    assert r.right and not w.right
    assert abs(r.conf - w.conf) < 0.01 and r.conf > 0.5
    wrong = {i.country: i.answer for i in items if not i.right}
    assert wrong["Switzerland"] == "Zurich" and wrong["Turkey"] == "Istanbul"
    a = CA.Calibration()
    assert 40 <= a.duration <= 60
