"""The numbers behind animations 261-262 (voxanim.study_blocks_za): thinking is tokens the
model writes and reads back; effort is how many it may write. Everything is the tiny
transformer of examples/thinking.py (voxanim.study_thinking), which needs no download."""

from __future__ import annotations

import numpy as np

from voxanim import study_blocks_za as Z
from voxanim import study_thinking as T


def test_percentages():
    assert Z.pct(215, 468) == "46%" and Z.pct(465, 468, 1) == "99.4%"
    assert Z.pct(234, 468) == "50%"


def test_what_each_effort_asks_for_is_the_scripts_example():
    bits = [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert Z.truth(bits, 0) == [1]
    assert Z.truth(bits, 3) == [0, 1, 1, 1]                 # after 4, 8 and 12 bits
    assert Z.truth(bits, 12) == [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    for k in T.EFFORTS:
        assert len(Z.truth(bits, k)) == k + 1               # k steps, then the answer
        assert Z.truth(bits, k) == T.example(bits, Z.effort_index(k))[14:]
    # each step sits under the last bit it covers; the answer under "="
    from voxanim.mech_blocks_za import written_col
    assert [written_col(3, j) for j in range(4)] == [4, 8, 12, 13]
    assert [written_col(12, j) for j in range(13)] == list(range(1, 14))
    assert written_col(0, 0) == 13


def test_the_example_is_picked_by_its_rule():
    i = Z.example()
    _every, held = T.strings()
    assert (i, int(held[i])) == (1, 42)
    for earlier in range(i):          # every earlier held-out string fails the rule
        deep, none = Z.written(earlier, 12), Z.written(earlier, 0)
        assert not (deep["right"] and deep["steps_right"] and not none["right"])
    deep, none = Z.written(i, 12), Z.written(i, 0)
    assert deep["right"] and deep["steps_right"] and not none["right"]


def test_the_tokens_shown_are_the_models_own():
    i = Z.example()
    every, held = T.strings()
    for k in T.EFFORTS:
        w = Z.written(i, k)
        seq = T.run()["seqs%d" % Z.effort_index(k)][i]
        assert w["question"] == [3 + Z.effort_index(k)] + every[held[i]].tolist() + [2]
        assert w["tokens"] == seq[14:].tolist() and w["passes"] == k + 1
        assert w["truth"] == Z.truth(every[held[i]], k)
    # its own greedy run, again: the same tokens
    P = T.params()
    assert T.generate(P, every[held[i]][None], Z.effort_index(12))[0, 14:].tolist() == \
        Z.written(i, 12)["tokens"]


def test_thinking_tokens():
    s = Z.tokens().validate()
    m = s.metadata
    assert (m["parameters"], m["parameters_text"]) == (103_114, "103,114")
    assert m["bits"] == [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    assert (m["ones"], m["answer"]) == (3, 1)
    assert m["deep"]["tokens"] == [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    assert m["deep"]["tokens"][:-1] == m["so_far"]          # the steps: the parity so far
    assert m["none"]["tokens"] == [0] and m["none"]["truth"] == [1]
    assert (m["deep"]["passes"], m["none"]["passes"]) == (13, 1)
    assert (m["right0"], m["right12"], m["pct0"]) == (215, 465, "46%")
    assert (m["held"], m["seen"]) == (468, 3_628)
    assert m["code_generate"][-1] == "    seq.append(nxt)"
    assert any("described, not measured" in x for x in m["limitations"])
    assert "465 of 468" in s.subtitle and "46%" in s.subtitle


def test_thinking_effort():
    s = Z.effort().validate()
    m = s.metadata
    assert m["efforts"] == [0, 1, 2, 3, 4, 6, 12]
    assert m["right"] == [215, 213, 218, 226, 241, 224, 465]
    assert m["written"] == [1, 2, 3, 4, 5, 7, 13]
    assert [r["bits_per_step"] for r in m["results"]] == [12, 12, 6, 4, 3, 2, 1]
    assert [r["steps_right"] for r in m["results"]] == [468, 213, 102, 106, 64, 62, 465]
    assert m["chance"] == 234 and (m["low"], m["high"]) == (213, 241)
    assert (m["best"], m["best_pct"], m["most_tokens"]) == (465, "99.4%", 13)
    # up to 6 steps, every effort within 25 of chance; 12 steps, all but 3
    assert all(abs(r - m["chance"]) < 25 for r in m["upto6"]) and m["best"] == 468 - 3
    assert [r["effort"] for r in m["shown"]] == [0, 3, 12]
    assert [r["truth"] for r in m["shown"]] == [[1], [0, 1, 1, 1],
                                               [0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
    tokens = np.asarray(s.panels[1].data["values"])
    assert tokens.tolist() == [k + 1 for k in T.EFFORTS]
    curve = s.panels[3].data
    assert np.allclose(np.asarray(curve["ys"])[0] * 468, m["right"])
    assert any("cliff" in x for x in m["limitations"])
    assert any("not timed" in x for x in m["limitations"])


def test_one_pass_is_the_whole_model():
    """A pass runs every number of the model once: the multiply-adds of one token are of
    the model's size (the weights' products, plus the attention over the places before)."""
    assert T.count() == 103_114
    assert T.multiply_adds_per_token() == 105_600
