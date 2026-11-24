"""Animation 184's arithmetic (voxanim.study_count_attention): every count follows from the
layer's shape, the running totals add up, each PyTorch line's arguments are the shape it
counts, and the counts are the models' own: the small ViT's arrays from vit.init, and GPT-2
small's downloaded weights when they are there."""

from __future__ import annotations

import os
import re
from math import prod

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim import gpt2, vit
from voxanim import study_count_attention as S
from voxanim.next_catalog import build_study

NAME = "counting-weights-transformer"
_built = {}


def study():
    if NAME not in _built:
        _built[NAME] = build_study("study_count_attention", NAME)
    return _built[NAME]


def ints(args):
    """``"768, 3 * 768"`` -> ``[768, 2304]``: each argument a product of integers."""
    return [prod(int(v) for v in a.split("*")) for a in args.split(",") if a.strip()
            and "=" not in a]


def check_code(p):
    """The PyTorch line of part ``p`` makes exactly the shape that is counted."""
    code, shape = p["code"], p["shape"]
    if m := re.search(r"nn\.Linear\(([^)]*)\)", code):
        assert ints(m.group(1)) == shape
        assert ("bias=False" in code) == (not p["bias"])
    elif m := re.search(r"nn\.MultiheadAttention\(([^)]*)\)", code):
        d, heads = ints(m.group(1))
        assert [d, d] == shape and p["times"] == 4 and heads == vit.HEADS
        assert "bias=False" in code and not p["bias"]
    elif m := re.search(r"nn\.Embedding\(([^)]*)\)", code):
        assert ints(m.group(1)) == shape and p["kind"] == "table"
    elif m := re.search(r"nn\.LayerNorm\(([^)]*)\)", code):
        assert ints(m.group(1)) == shape and p["kind"] == "norm"
    elif m := re.search(r"torch\.zeros\(([^)]*)\)", code):
        dims = ints(m.group(1))
        assert dims == shape or [1] + dims == shape
    elif m := re.search(r"range\((\d+)\)", code):
        assert [int(m.group(1))] == shape and p["kind"] == "blocks"
    else:
        assert code == "lm_head.weight = wte.weight" and p["kind"] == "tied"
    assert "import" not in code


# ---------------------------------------------------------------------- the rule
def test_the_rule():
    lin = S.linear("x", "x", "nn.Linear(784, 64)", 784, 64, [])
    assert (lin["weights"], lin["biases"], lin["total"]) == (50176, 64, 50240)
    assert S.linear("x", "x", "", 24, 24, [], bias=False, times=4)["total"] == 2304
    assert S.norm("x", "x", "", 768, [])["total"] == 1536
    assert S.table("x", "x", "", 1024, 768, [], "positions")["total"] == 786432


def test_study_is_valid_and_has_four_panels():
    s = study()
    assert len(s.panels) == 4
    assert [p.kind for p in s.panels] == ["bars", "points", "bars", "bars"]
    assert s.metadata["evidence"].startswith("exact arithmetic")
    assert s.metadata["count_line"] == "sum(p.numel() for p in model.parameters())"


# ---------------------------------------------------------------------- the small ViT
def test_attention_is_four_square_blocks_and_heads_add_nothing():
    a = study().metadata["vit"]["attention"]
    d = vit.D
    assert a["d"] == 24 and a["heads"] == 3 and a["head_width"] == 8
    assert [m["total"] for m in a["matrices"]] == [d * d] * 4 == [576] * 4
    assert [m["running"] for m in a["matrices"]] == [576, 1152, 1728, 2304]
    assert a["total"] == 2304 and a["piece"] == 24 * 8
    assert set(a["by_heads"].values()) == {2304}
    assert set(a["by_heads"]) == {"1", "2", "3", "4", "6", "8", "12", "24"}
    xy = study().panels[1].data["xy"]
    assert (xy[:, 1] == 2304).all()
    P = S.vit_arrays()
    for m in a["matrices"]:
        assert P[m["key"]].shape == (d, d) and P[m["key"]].size == m["total"]
        check_code(m)


def test_vit_counts_are_its_own_arrays():
    v = study().metadata["vit"]
    P = S.vit_arrays()
    parts = v["parts"]
    assert [p["total"] for p in parts] == [1200, 24, 408, 2304, 800, 330]
    assert [p["running"] for p in parts] == [1200, 1224, 1632, 3936, 4736, 5066]
    assert v["total"] == v["counted"] == 5066
    named = [n for p in parts for n in p["names"]]
    assert sorted(named) == sorted(vit.NAMES)           # every array, once
    for p in parts:
        assert sum(P[n].size for n in p["names"]) == p["total"], p["key"]
        assert p["weights"] + p["biases"] == p["total"]
        check_code(p)
    assert P["E"].shape == (vit.PATCH ** 2, vit.D) and P["pos"].shape == (vit.N_TOK, vit.D)
    assert sum(P[n].size for n in vit.NAMES) == 5066
    # any seed: the same shapes
    assert sum(a.size for a in S.vit_arrays(7).values()) == 5066


# ---------------------------------------------------------------------- GPT-2 small
def test_one_gpt2_block():
    g = study().metadata["gpt2"]
    block = {p["key"]: p for p in g["block"]}
    assert block["ln_1"]["total"] == 2 * 768
    assert block["attn.c_attn"]["total"] == 768 * 2304 + 2304 == 1771776
    assert block["attn.c_proj"]["total"] == 768 * 768 + 768 == 590592
    assert block["mlp.c_fc"]["total"] == 768 * 3072 + 3072 == 2362368
    assert block["mlp.c_proj"]["total"] == 3072 * 768 + 768 == 2360064
    assert g["per_block"] == 7087872
    assert [p["running"] for p in g["block"]] == [1536, 1773312, 2363904, 2365440, 4727808,
                                                  7087872]
    assert g["block_attention"] + g["block_mlp"] + g["block_norms"] == 7087872
    assert g["attention_share_of_block"] == pytest.approx(1 / 3, abs=0.001)
    for p in g["block"]:
        check_code(p)


def test_all_of_gpt2():
    g = study().metadata["gpt2"]
    parts = {p["key"]: p for p in g["parts"]}
    assert parts["wte"]["total"] == 50257 * 768 == 38597376
    assert parts["wpe"]["total"] == 1024 * 768 == 786432
    assert parts["h"]["total"] == 12 * 7087872 == 85054464
    assert parts["ln_f"]["total"] == 1536
    assert parts["lm_head"]["total"] == 0 and parts["lm_head"]["reuses"] == "wte"
    assert [p["running"] for p in g["parts"]] == [38597376, 39383808, 124438272, 124439808,
                                                  124439808]
    assert g["total"] == g["from_shapes"] == 124439808
    assert g["tied_saves"] == 38597376
    groups = {r["label"]: r for r in g["groups"]}
    assert sum(r["total"] for r in g["groups"]) == 124439808
    assert groups["MLPs"]["total"] == 12 * (2362368 + 2360064) == 56669184
    assert groups["attention"]["total"] == 28348416 and groups["norms"]["total"] == 38400
    assert max(g["groups"], key=lambda r: r["total"])["label"] == "MLPs"
    assert sum(r["share"] for r in g["groups"]) == pytest.approx(1.0)
    np.testing.assert_array_equal(study().panels[3].data["values"],
                                  [r["total"] for r in g["groups"]])
    for p in g["parts"]:
        check_code(p)


def test_shapes_follow_the_checkpoint_names():
    shapes = S.gpt2_shapes()
    assert len(shapes) == 4 + 12 * 12
    assert shapes["h.11.attn.c_attn.weight"] == (768, 2304)
    assert shapes["h.0.attn.c_attn.bias"] == (2304,)
    assert shapes["h.3.mlp.c_proj.weight"] == (3072, 768)
    assert shapes["h.5.ln_2.weight"] == (768,)


@needs_gpt2
def test_gpt2_counts_are_the_downloaded_weights():
    W = gpt2.load_safetensors(os.path.join(gpt2.DIR, "model.safetensors"))
    W = {k.replace("transformer.", ""): v for k, v in W.items()}
    # the checkpoint also stores each block's causal mask, a buffer that is not learned
    mask = re.compile(r"h\.\d+\.attn\.(masked_)?bias$")
    learned = {k: v for k, v in W.items() if not mask.search(k)}
    shapes = S.gpt2_shapes()
    assert set(learned) == set(shapes)                       # no lm_head: it is tied
    for k, v in learned.items():
        assert tuple(v.shape) == shapes[k], k
    g = study().metadata["gpt2"]
    assert sum(v.size for v in learned.values()) == g["total"] == 124439808
    for i in (0, 11):
        for p in g["block"]:
            assert sum(learned["h.%d.%s" % (i, n)].size for n in p["names"]) == p["total"]
    for p in g["parts"]:
        if p["kind"] in ("table", "norm"):
            assert sum(learned[n].size for n in p["names"]) == p["total"]
