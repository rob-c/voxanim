"""Animation 180 (voxanim.mech_bridge_text) and its experiment (voxanim.study_bridge_text):
the layout is clean, the loop closes, every phase says what kind of evidence it shows, the
bridge names where it comes from and where it leads, and the numbers on screen are the
study's."""

from __future__ import annotations

import numpy as np
from conftest import needs_text

from voxanim import layout, presets, qa
from voxanim import mech_bridge_text as M
from voxanim.next_catalog import build_study, drawing

NAME = "markov-chain-to-rnn"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if not _built:
        _built["a"] = (presets.get(NAME).build(), build_study("study_bridge_text", NAME))
    return _built["a"]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.shown("a b\nc") == "a␣b↵c"
    assert M.bits(2.8544) == "2.85"
    assert M.is_evidence(M.TRAINED) and M.is_evidence(M.EXACT)
    assert M.is_evidence(M.HELD % "1,000")
    assert not M.is_evidence("measured elsewhere")


# ---------------------------------------------------------------------- drawn
@needs_text
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_text
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME] is drawing(NAME)
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_text
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@needs_text
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for k in range(len(a.PHASES)):
        lines = layout.wrap(a.subtitle_for(k), layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_text
def test_it_is_a_bridge():
    a, _ = built()
    assert M.FROM in texts(a, 0, 0.3)
    assert "Measuring entropy" in M.FROM and "A Markov chain from real text" in M.FROM
    last = texts(a, len(a.PHASES) - 1, 0.95)
    assert "next: An RNN training step" in last
    assert "perplexity returns in Training language models" in last


@needs_text
def test_study_says_what_was_run():
    _, s = built()
    m = s.metadata
    assert len(s.panels) == 4 and m["dataset"] == "Tiny Shakespeare / char-rnn"
    assert m["limitations"] and m["evidence"]["measured"] and m["evidence"]["exact"]


@needs_text
def test_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    k, top = m["best_order"], max(m["orders"])
    first = texts(a, 0, 0.95)
    assert "training text: %s characters" % format(m["split_sizes"][0], ",") in first
    assert "order %d: %s contexts counted" % (top, format(m["contexts"][top], ",")) in first
    assert "%.2f" % m["training_bits"][0] in first and "%.2f" % m["training_bits"][-1] in first
    assert "order 4: 46,224 contexts counted" in first             # as the catalogue says
    held = texts(a, 1, 0.95)
    ex = m["example"]
    assert "one held-out character to guess: %s" % M.shown(ex["next"]) in held
    assert "order %d: %.2f" % (k, m["best_bits"]) in held
    assert "%d: %s" % (k, M.shown(ex["contexts"][k])) in held
    assert format(ex["counts"][k], ",") in held and "never" in held
    assert "%.0f%% at order %d, %.0f%% at order %d" % (
        100 * m["unseen_share"][k], k, 100 * m["unseen_share"][-1], top) in held
    assert "8% at order 4, 56% at order 8" in held
    state = texts(a, 2, 0.95)
    r = m["rnn"]
    assert "%s parameters, the same at every step" % format(r["parameters"], ",") in state
    assert "82,881 parameters, the same at every step" in state
    assert "best table, order %d: %.2f" % (k, m["best_bits"]) in state
    assert "held out after 1,200 updates: %.2f bits" % m["rnn_final"] in state
    why = texts(a, 3, 0.95)
    assert "order-%d table: %s counts" % (k, format(m["best_pairs"], ",")) in why
    assert any(t.endswith("= 1,160,290,625") for t in why)
    for tb, rb in zip(m["bucket_table_bits"], m["bucket_rnn_bits"]):
        assert {"%.2f" % tb, "%.2f" % rb} <= set(why)
    assert "training %.2f → held out %.2f (GRU)" % (m["rnn_training_bits"], m["rnn_final"]) \
        in why
    perp = texts(a, 4, 0.95)
    for name, b in m["perplexity_bits"].items():
        assert "2^%.2f = %.1f" % (b, m["perplexity"][name]) in perp
    assert "2^6.02 = 65.0" in perp
    subs = [a.subtitle_for(i) for i in range(len(a.PHASES))]
    assert "%.2f bits for the table, %.2f for the network" % (
        m["bucket_table_bits"][0], m["bucket_rnn_bits"][0]) in subs[3]
    assert "to %.2f" % m["rnn_final"] in subs[2]


@needs_text
def test_the_curves_drawn_are_the_study_arrays():
    a, s = built()
    p0, p1 = s.panels[0].data, s.panels[1].data
    np.testing.assert_array_equal(a.held, p0["ys"][0])
    np.testing.assert_array_equal(a.train, p0["ys"][1])
    np.testing.assert_array_equal(a.rnn_held, p1["ys"][0])
    np.testing.assert_array_equal(a.perp, s.panels[3].data["values"])
