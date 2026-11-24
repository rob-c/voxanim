"""voxanim.study_hosting: what examples/serve.py does with GPT-2, counted and checked."""

from __future__ import annotations

from conftest import needs_gpt2

from voxanim import gpt2
from voxanim import study_hosting as H


def test_the_arithmetic():
    assert H.macs_per_token(0) == 12 * (7_077_888 + 2 * 768) + 768 * 50_257
    assert H.macs_per_token(9, cache=False) > 10 * H.macs_per_token(9) - 11 * 768 * 50_257
    assert 2 * gpt2.N_LAYER * gpt2.D_MODEL * 4 == 73_728


@needs_gpt2
def test_sizes_and_the_request():
    s = H.sizes()
    assert s["total"] == 124_439_808 and s["bytes32"] == 497_759_232
    assert s["parts"]["one_layer"] == 7_087_872
    assert s["total"] == (s["parts"]["token_table"] + s["parts"]["position_table"]
                          + 12 * s["parts"]["one_layer"] + s["parts"]["final_norm"])
    assert s["file_bytes"] > s["bytes32"]                  # the file also holds the masks
    r = H.request()
    assert r["bytes"] == len(r["json"].encode()) < 200
    assert r["prompt"] == "User: What is a server?\nAssistant:" and r["n_tokens"] == 10
    c = H.conversation()
    assert [t["messages"] for t in c] == [1, 3, 5]
    assert c[0]["tokens"] < c[1]["tokens"] < c[2]["tokens"]  # the whole talk, every time


@needs_gpt2
def test_the_reply_cache_and_batching():
    g = H.generation()
    assert len(g["steps"]) == 20 and g["reply"].startswith(" A server is")
    assert all(s["token"] == s["top"][0][0] for s in g["steps"])   # greedy: the likeliest
    c = H.cache_check()
    assert c["same_tokens"] and c["max_difference"] < 1e-4
    k = H.costs()
    assert k["without_cache"] > 10 * k["with_cache"]
    b = H.batching()
    assert b["same_tokens"] and b["max_difference"] < 1e-4
    assert b["bytes_per_token"][64] == b["weight_bytes"] / 64
    m = H.memory()
    assert m["per_token"] == 73_728 and m["full_context"] == 75_497_472
    assert m["reply_cache_tokens"] == 10 + 20


@needs_gpt2
def test_the_agent():
    a = H.agent()
    assert [c["messages_sent"] for c in a["calls"]] == [1, 3, 5, 7]
    sent = [c["tokens_sent"] for c in a["calls"]]
    assert sent == sorted(sent)                            # the history grows every call
    assert [c["tool"] for c in a["calls"]] == [True, True, True, False]
    assert {s["runs_on"] for s in a["steps"] if s["role"] == "tool"} == {"device"}
