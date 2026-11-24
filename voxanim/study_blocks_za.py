"""The numbers behind animations 261-262 of "Blocks and code" (group ``za``, the last two of
"Hosting a language model"): what "thinking" and "effort" are, shown on a model small
enough to see. Everything comes from :mod:`voxanim.study_thinking`, the numpy twin of
``examples/thinking.py``: one tiny transformer (103,114 numbers) trained once on the parity
of 12 bits, with an effort token in front saying how many steps (0, 1, 2, 3, 4, 6 or 12) it
may write before the answer, then tested on the 468 bit strings it never saw.

* ``thinking-tokens``: one held-out string, picked by a rule (:func:`example`: the first
  held-out string whose effort-12 run writes every step right and the right answer, and
  whose effort-0 answer is wrong). The question is tokens (the effort, 12 bits, "="); the
  model writes one token a pass, each read back by the next pass (region "th-generate");
  the tokens are the model's own (``run()["seqs<e>"]``). At effort 12 it writes 12 steps,
  the parity so far, and the right answer: 13 tokens, 13 passes. At effort 0 it writes the
  answer at once, 1 pass, and here it is wrong. Over the 468: 215 right with no steps
  (46%, about chance), 465 with a step per bit.
* ``thinking-effort``: for every effort, answers right out of 468 and tokens written
  (``k + 1``, one pass through the whole model each). Up to 6 steps it stays about chance
  (213 to 241 right); 12 steps, one bit each, 465. What carries over to large served
  models, and what this toy cannot show, is said in words (no measurement).

Counts, not speeds: a pass is counted, never timed. One run, one seed: where the cliff
falls is this model's.
"""

from __future__ import annotations

import functools

import numpy as np

from . import study_thinking as T
from .blocks import code_lines, commas
from .study import Panel, Study

ON = "a tiny transformer"          # the evidence tags: "measured on a tiny transformer: ..."
SOURCE = ("examples/thinking.py, trained once (voxanim.study_thinking, its numpy twin): a "
          "transformer of 2 layers, width 64, 103,114 numbers, on the parity of 12 bits, "
          "tested on 468 bit strings it never saw")
LAST_E = len(T.EFFORTS) - 1        # the index of effort 12
SHOWN_EFFORTS = (0, 3, 12)         # 262, phase 1: the same string at three efforts


def pct(right, of, places=0):
    """``46%``."""
    return "%.*f%%" % (places, 100.0 * right / of)


def effort_index(k):
    return T.EFFORTS.index(k)


def seqs(k):
    """The model's own sequences at effort ``k`` on the 468 held-out strings: ``[468,
    14 + k + 1]`` (the question, then what it wrote)."""
    return np.asarray(T.run()["seqs%d" % effort_index(k)])


def truth(bits, k):
    """What the model should write at effort ``k``: its steps, then the answer."""
    return T.example([int(b) for b in bits], effort_index(k))[2 + T.BITS:]


@functools.lru_cache(maxsize=1)
def example():
    """The held-out string followed: the first (in the order of ``strings()[1]``) whose
    effort-12 run writes every step right and the right answer, and whose effort-0 answer
    is wrong. A rule, so that the example is not picked for how it looks. Returns its
    index among the 468."""
    every, held = T.strings()
    s12, s0 = seqs(12), seqs(0)
    for i in range(len(held)):
        bits = every[held[i]]
        if (s12[i, 2 + T.BITS:].tolist() == truth(bits, 12)
                and s0[i, -1] != int(bits.sum()) % 2):
            return i
    raise ValueError("no held-out string fits the rule")


def written(i, k):
    """What the model wrote for held-out string ``i`` at effort ``k``: ``dict(tokens,
    truth, right)`` (its steps and answer; what they should be; is the answer right)."""
    every, held = T.strings()
    bits = every[held[i]]
    got = seqs(k)[i, 2 + T.BITS:].tolist()
    want = truth(bits, k)
    return dict(effort=k, tokens=[int(v) for v in got], truth=[int(v) for v in want],
                right=bool(got[-1] == want[-1]), steps_right=bool(got[:-1] == want[:-1]),
                passes=len(got), question=[int(v) for v in seqs(k)[i, :2 + T.BITS]])


def common():
    """The numbers both animations show."""
    every, held = T.strings()
    i = example()
    bits = [int(b) for b in every[held[i]]]
    res = T.results()
    rows = [dict(r, right_text="%d of %d" % (r["right"], r["of"]),
                 pct_text=pct(r["right"], r["of"])) for r in res]
    return dict(
        source=SOURCE, on=ON, parameters=T.count(), parameters_text=commas(T.count()),
        bits_n=T.BITS, efforts=list(T.EFFORTS), held=T.HELD, seen=int(2 ** T.BITS - T.HELD),
        seen_text=commas(2 ** T.BITS - T.HELD), steps=T.STEPS, steps_text=commas(T.STEPS),
        index=i, number=int(held[i]), bits=bits, ones=int(sum(bits)),
        answer=int(sum(bits) % 2), results=rows,
        code_generate=list(code_lines("th-generate")),
        code_example=list(code_lines("th-example")))


# ====================================================================== 261
def tokens():
    m = common()
    i = m["index"]
    deep, none = written(i, 12), written(i, 0)
    if not (deep["right"] and deep["steps_right"] and not none["right"]):
        raise ValueError("the example does not fit its rule")
    r0, r12 = m["results"][0], m["results"][LAST_E]
    m.update(
        evidence="measured on a tiny transformer: one held-out string, its own tokens",
        deep=deep, none=none, so_far=[int(sum(m["bits"][:j + 1]) % 2) for j in range(T.BITS)],
        right0=r0["right"], right12=r12["right"], pct0=r0["pct_text"], pct12=r12["pct_text"],
        limitations=[
            "one training run (one seed); the example is held-out string %d of 468 (the "
            "number %d), picked by a rule: the first whose effort-12 run is right at every "
            "step and whose effort-0 answer is wrong" % (i, m["number"]),
            "the tokens shown are the model's own, written greedily one at a time; a pass "
            "is counted, not timed",
            "what is said of large served models is described, not measured: their "
            "thinking is tokens they write before the answer, often not shown to the "
            "person, read back and counted like any other; how much it helps is not "
            "something this toy can say"])
    q = np.array([deep["question"]], float)
    panels = (
        Panel("The question, as tokens", "held-out string %d at effort 12: the effort token, "
              "12 bits, \"=\"" % i, "matrix", dict(values=q)),
        Panel("Steps written at effort 12", "the model's 12 steps (parity so far) and its "
              "answer, one token a pass", "bars",
              dict(values=np.array(deep["tokens"], float),
                   labels=["step %d" % (j + 1) for j in range(T.BITS)] + ["answer"])),
        Panel("Right answers, 468 strings", "no steps against a step per bit", "bars",
              dict(values=np.array([r0["right"], r12["right"]], float),
                   labels=["effort 0", "effort 12"])),
        Panel("Tokens written", "passes through the model for one answer", "bars",
              dict(values=np.array([none["passes"], deep["passes"]], float),
                   labels=["effort 0", "effort 12"])),
    )
    return Study("Thinking: tokens it writes for itself",
                 "With 12 steps written first the tiny model is right on %d of 468 strings it "
                 "never saw; answering at once, %d (%s)" % (r12["right"], r0["right"],
                                                           r0["pct_text"]), panels, m)


# ====================================================================== 262
def effort():
    m = common()
    rows = m["results"]
    i = m["index"]
    chance = T.HELD // 2
    upto6 = [r["right"] for r in rows if 0 < r["effort"] <= 6]
    m.update(
        evidence="measured on a tiny transformer: 468 held-out strings, every effort",
        shown=[written(i, k) for k in SHOWN_EFFORTS], chance=chance,
        low=min(r["right"] for r in rows[:-1]), high=max(r["right"] for r in rows[:-1]),
        upto6=upto6, right=[r["right"] for r in rows], written=[r["tokens"] for r in rows],
        best=rows[-1]["right"], best_pct=pct(rows[-1]["right"], T.HELD, 1),
        most_tokens=rows[-1]["tokens"],
        limitations=[
            "one training run (one seed): where the cliff falls is this model's; another "
            "model, or another task, would put it elsewhere",
            "tokens written = passes through the model, each pass the whole 103,114 "
            "numbers; counted, not timed",
            "chance is half of 468 (234): a coin would be right about as often",
            "what carries over to large served models is said in words, not measured: "
            "effort is a budget of thinking tokens a request may ask for; more thinking is "
            "more tokens, so more passes, time and cost; it helps when each step it allows "
            "is a step the model can do. Not shown: by how much, that more always helps, or how "
            "any particular service sets or counts its effort"])
    panels = (
        Panel("Right answers by effort", "of 468 held-out strings, at each effort", "bars",
              dict(values=np.array(m["right"], float),
                   labels=["%d steps" % k for k in T.EFFORTS])),
        Panel("Tokens written by effort", "steps plus the answer: passes through the model",
              "bars", dict(values=np.array(m["written"], float),
                           labels=["%d steps" % k for k in T.EFFORTS])),
        Panel("Every step right", "strings whose steps were all written right", "bars",
              dict(values=np.array([r["steps_right"] for r in rows], float),
                   labels=["%d steps" % k for k in T.EFFORTS])),
        Panel("Right against tokens", "share right against tokens written, efforts 0 to 12",
              "curve", dict(x=np.array(m["written"], float),
                            ys=np.array([[v / T.HELD for v in m["right"]]], float),
                            labels=["share right"])),
    )
    return Study("Effort: how much thinking is allowed",
                 "Up to 6 steps the tiny model stays about chance (%d to %d of 468); with 12, "
                 "one per bit, %d, for 13 tokens instead of 1" % (m["low"], m["high"],
                                                                 m["best"]), panels, m)


STUDIES = {"thinking-tokens": tokens, "thinking-effort": effort}
