"""Animation 184 (voxanim.mech_count_attention): the layout is clean, the loop closes,
every phase says it is exact arithmetic, each part's PyTorch line is on screen while it is
counted, and the numbers on screen are the study's."""

from __future__ import annotations

import numpy as np

from voxanim import layout, presets, qa
from voxanim import mech_count_attention as M
from voxanim.next_catalog import build_study

NAME = "counting-weights-transformer"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_count_attention", NAME))
    return _built[NAME]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def step_texts(a, phase, n, i):
    """What is on screen half way through step ``i`` of a phase counted in ``n`` steps."""
    s = M.starts(n)
    end = s[i + 1] if i + 1 < n else 1.0
    return texts(a, phase, (s[i] + end) / 2)


# ---------------------------------------------------------------------- helpers
def test_labels():
    assert M.share(0.4554) == "46%" and M.share(0.00632) == "0.6%"
    assert M.share(0.000309) == "0.03%" and M.share(0) == "0%"
    assert M.is_evidence(M.VIT_TAG) and M.is_evidence(M.GPT_TAG)
    assert not M.is_evidence("measured on MNIST")
    s = M.starts(4)
    assert M.current(0.0, s) == 0 and M.current(0.99, s) == 3
    assert M.alpha(s[1] + 0.05, 1, s) == 1.0 and M.alpha(s[1] + 0.05, 0, s) == 0.0
    pieces, result = M.expression(dict(kind="linear", shape=[784, 64], times=1, bias=True,
                                       biases=64, total=50240))
    assert pieces[0][0] == "784 × 64" and pieces[1][0] == "+ 64" and result == "= 50,240"


# ---------------------------------------------------------------------- the animation
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME]
    assert 45.0 <= a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


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


def test_attention_on_screen():
    a, s = built()
    att = s.metadata["vit"]["attention"]
    st = a.STEPS_1
    for i, m in enumerate(att["matrices"]):
        on = texts(a, 0, (st[i] + st[i + 1]) / 2)
        assert m["code"] in on and m["label"] in on
        assert "24 × 24" in on and "= 576" in on
        assert M.commas(m["running"]) in on
    four = texts(a, 0, (st[4] + st[5]) / 2)
    assert att["code_one_head"] in four and "4 × 24 × 24" in four and "= 2,304" in four
    heads = texts(a, 0, 0.95)
    assert att["code_heads"] == "attn = nn.MultiheadAttention(24, 3, bias=False)"
    assert att["code_heads"] in heads and "4 × 3 × 24 × 8" in heads and "= 2,304" in heads
    assert "3 × (24 × 8)" in heads and "3 × (8 × 24)" in heads and "2,304" in heads


def test_vit_on_screen():
    a, s = built()
    parts = s.metadata["vit"]["parts"]
    n = len(parts) + 1
    for i, p in enumerate(parts):
        on = step_texts(a, 1, n, i)
        assert p["code"] in on
        assert M.commas(p["running"]) in on and M.commas(p["total"]) in on
    patch = step_texts(a, 1, n, 0)
    assert {"patch = nn.Linear(49, 24)", "49 × 24", "+ 24", "= 1,200", "49 in",
            "24 out"} <= set(patch)
    assert {"17 × 24", "= 408"} <= set(step_texts(a, 1, n, 2))
    assert {"24 × 32", "+ 32", "= 800"} <= set(step_texts(a, 1, n, 4))
    end = step_texts(a, 1, n, n - 1)
    assert "= 5,066" in end and "5,066" in end and "vit.py holds exactly 5,066 numbers" in end


def test_gpt2_block_on_screen():
    a, s = built()
    block = s.metadata["gpt2"]["block"]
    n = len(block) + 1
    for i, p in enumerate(block):
        on = step_texts(a, 2, n, i)
        assert p["code"] in on and M.commas(p["running"]) in on
    c_attn = step_texts(a, 2, n, 1)
    assert {"c_attn = nn.Linear(768, 3 * 768)", "768 × 2,304", "+ 2,304", "= 1,771,776",
            "1,773,312"} <= set(c_attn)
    assert {"768", "+ 768", "= 1,536", "scale", "shift"} <= set(step_texts(a, 2, n, 0))
    assert {"3,072 × 768", "+ 768", "= 2,360,064"} <= set(step_texts(a, 2, n, 5))
    assert {"= 7,087,872", "7,087,872"} <= set(step_texts(a, 2, n, n - 1))


def test_all_of_gpt2_on_screen():
    a, s = built()
    parts = s.metadata["gpt2"]["parts"]
    n = len(parts) + 1
    wte = step_texts(a, 3, n, 0)
    assert {"wte = nn.Embedding(50257, 768)", "50,257 × 768", "= 38,597,376",
            "50,257 tokens"} <= set(wte)
    blocks = step_texts(a, 3, n, 2)
    assert {"h = [Block() for _ in range(12)]", "12 × 7,087,872", "= 85,054,464",
            "124,438,272"} <= set(blocks)
    tied = step_texts(a, 3, n, 4)
    assert {"lm_head.weight = wte.weight", "= 0", "shared: 50,257 × 768",
            "so it adds no weights", "124,439,808"} <= set(tied)
    assert "reuses the token table" in " ".join(tied)
    assert "= 124,439,808" in step_texts(a, 3, n, n - 1)


def test_where_the_weights_live():
    a, s = built()
    g = s.metadata["gpt2"]
    on = texts(a, 4, 0.95)
    assert "sum(p.numel() for p in model.parameters())" in on and "124,439,808" in on
    assert "56,669,184 · 46%" in on and "38,597,376 · 31%" in on
    assert "28,348,416 · 23%" in on and "786,432 · 0.6%" in on and "38,400 · 0.03%" in on
    assert "0: it reuses the token table" in on
    assert "attention 33%" in on and "MLP 67%" in on
    assert {r["label"] for r in g["groups"]} <= set(on)
    assert a.subtitle_for(4).startswith("The MLPs and the token table hold most: 46% and 31%")
