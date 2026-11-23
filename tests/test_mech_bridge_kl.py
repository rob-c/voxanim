"""Animation 176 (voxanim.mech_bridge_kl): the layout is clean, the loop closes, every
phase says what kind of evidence it shows, the first recalls "Logits, loss and nats" and
the last points on, and the numbers on screen are the study's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_kl as M
from voxanim.next_catalog import build_study

NAME = "kl-divergence"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_bridge_kl", NAME))
    return _built[NAME]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.nats(-1.2161) == "−1.216" and M.nats(2.37, 2) == "2.37" and M.nats(-4, 0) == "−4"
    assert M.tiny(1.0225e-10) == "1.0×10⁻¹⁰" and M.tiny(0.9745) == "0.9745"
    assert M.is_evidence(M.EXACT) and M.is_evidence(M.HELD)
    assert M.is_evidence(M.HUMPS % "12,665") and not M.is_evidence("measured elsewhere")


# ---------------------------------------------------------------------- measured on MNIST
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME]
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@needs_mnist
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for k in range(len(a.PHASES)):
        sub = a.subtitle_for(k)
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_mnist
def test_it_is_a_bridge():
    a, _ = built()
    assert "from: Logits, loss and nats" in texts(a, 0, 0.5)
    assert a.subtitle_for(0).startswith("from Logits, loss and nats")
    last = texts(a, len(a.PHASES) - 1, 0.95)
    assert "next: where KL comes back on the page" in last
    assert {"Dropout, noise, max-norm, label smoothing", "Distillation",
            "The KL term of a VAE"} <= set(last)
    assert a.subtitle_for(len(a.PHASES) - 1).startswith("next on the page")


@needs_mnist
def test_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    sh, rows, hm = m["shown"], m["rows"], m["humps"]
    lab = sh["label"]
    recall = texts(a, 0, 0.95)
    q = sh["q"][lab]
    assert "loss = −ln q(%d) = −ln %.4f = %.3f nats" % (lab, q, sh["onehot_loss"]) in recall
    assert "loss = −ln q(7) = −ln 0.9745 = 0.026 nats" in recall     # as 175 shows it
    assert "all test digits: %.3f nats, all of it KL" % rows["one-hot"]["kl"] in recall
    split = texts(a, 1, 0.95)
    assert "H(p) = Σ −p ln p = 0.500" in split                       # exact for ε = 0.1
    assert "KL(p‖q) = Σ p ln(p/q) = %.3f" % sh["kl"] in split
    assert "CE = Σ −p ln q = %.3f" % sh["cross_entropy"] in split
    assert "= %.3f + %.3f" % (sh["entropy"], sh["kl"]) in split
    assert any(t.startswith("%.0f%% of the CE is the 0.01" % (100 * sh["wrong_share"]))
               for t in split)
    three = texts(a, 2, 0.95)
    for t in M.S.TARGETS:
        r = rows[t]
        assert "%.3f + %.3f = %.3f" % (r["entropy"], r["kl"], r["cross_entropy"]) in three
        assert "untrained: %.3f" % r["untrained_cross_entropy"] in three
    assert "0.000 + %.3f = %.3f" % (rows["one-hot"]["kl"], rows["one-hot"]["kl"]) in three
    turned = texts(a, 3, 0.95)
    assert "∞" in turned
    for t in ("smoothed", "teacher"):
        assert {"%.3f" % rows[t]["kl"], "%.3f" % rows[t]["reverse_kl"]} <= set(turned)
    assert "KL(q‖p) = %.3f" % sh["reverse_kl"] in turned
    fits = texts(a, 4, 0.95)
    f, b = hm["forward"], hm["reverse"]
    assert "centre %s, width %.2f" % (M.nats(f["mu"], 2), f["sd"]) in fits
    assert "centre %s, width %.2f: on the 1s" % (M.nats(b["mu"], 2), b["sd"]) in fits
    assert "centred where p is %.0f%% of its peak" % (100 * f["p_at_centre"]) in fits
    assert "forward fit: KL(p‖q) %.2f, KL(q‖p) %.1f" % (f["kl_pq"], f["kl_qp"]) in fits
    assert "reverse fit: KL(p‖q) %.1f, KL(q‖p) %.2f" % (b["kl_pq"], b["kl_qp"]) in fits
    assert "training 1s (%s)" % format(hm["n"]["1"], ",") in fits
    assert any(t == M.HUMPS % "12,665" for t in fits)
    nxt = texts(a, 5, 0.95)
    assert "a smoothed target: CE = 0.500 + KL, never below H(p)" in nxt
    assert "KL(soft teacher‖student): %.3f here at T = 2" % rows["teacher"]["kl"] in nxt


@needs_mnist
def test_the_drawn_histogram_counts_every_digit():
    a, _ = built()
    width = a.edges[1] - a.edges[0]
    total = sum(c.sum() for c in a.counts.values()) * width
    assert total == pytest.approx(1.0, abs=0.01)    # almost every digit falls in the range
