"""The numbers behind animations 258-260 (voxanim.study_blocks_z): many people served by one
model, what each conversation's cache takes, and a coding agent's loop, all from
examples/serve.py run on GPT-2 (voxanim.study_hosting) or exact arithmetic."""

from __future__ import annotations

import json

import numpy as np
from conftest import needs_gpt2

from voxanim import gpt2
from voxanim import study_blocks_z as Z
from voxanim import study_hosting as H


# ---------------------------------------------------------------------- no data needed
def test_megabytes():
    assert Z.mb(497_759_232) == "498 MB" and Z.mb(4 * 497_759_232) == "1,991 MB"
    assert Z.mb(7_777_488) == "7.8 MB" and Z.mb(15_554_976) == "16 MB"
    assert Z.mb(75_497_472) == "75 MB" and Z.mb(2_211_840) == "2.2 MB"


def test_lines_break_json_after_commas_and_keep_short_lines():
    assert Z.lines('{"tool": "read_file", "path": "parser.py"}') == [
        '{"tool": "read_file",', '"path": "parser.py"}']
    assert Z.lines('{"tool": "edit_file", "path": "parser.py", "old": "[1]", "new": "[0]"}') == [
        '{"tool": "edit_file",', '"path": "parser.py",', '"old": "[1]", "new": "[0]"}']
    assert Z.lines("def parse(line):\n    return line.split(',')[1]\n", 40) == [
        "def parse(line):", "    return line.split(',')[1]"]
    text = "Fixed: parse() took the second field; it now takes the first. The 3 tests pass."
    out = Z.lines(text)
    assert " ".join(out) == text and all(len(s) <= Z.WIDE for s in out)


def test_the_cache_arithmetic():
    assert 2 * gpt2.N_LAYER * gpt2.D_MODEL * 4 == 73_728
    assert 1024 * 73_728 == 75_497_472
    assert 10 ** 9 // 73_728 == 13_563


# ---------------------------------------------------------------------- GPT-2
@needs_gpt2
def test_people_are_served_as_the_script_serves_them():
    ps = Z.people()
    assert [p["question"] for p in ps] == list(H.PEOPLE)
    assert [p["first"] for p in ps] == [" A", " It", " I", " Hello"]
    assert [p["together"] for p in ps] == [" server", "'s", "'m", "!"]
    assert all(p["one_each"] == p["together"] for p in ps)
    b = H.batching()                                       # the study's own check agrees
    assert b["same_tokens"] and b["next_tokens"] == [p["together"] for p in ps]
    assert max(p["difference"] for p in ps) == b["max_difference"] < 1e-5
    W, tok = H.model()
    for p in ps:                                           # the first token: the prompt read once
        ids = tok.encode(H.program()["prompt"]([{"role": "user", "content": p["question"]}]))
        assert len(ids) == p["prompt_tokens"]
        assert tok.decode([int(H.program()["step"](W, ids, [])[0].argmax())]) == p["first"]


@needs_gpt2
def test_many_people():
    s = Z.many_people().validate()
    m = s.metadata
    assert (m["numbers"], m["weight_bytes"]) == (124_439_808, 497_759_232)
    assert (m["weight_text"], m["one_each_text"], m["least_text"]) == ("498 MB", "1,991 MB",
                                                                      "7.8 MB")
    assert m["batch_shape"] == "[4, 768]" and m["qkv_shape"] == "[4, 2304]"
    b = H.batching()["bytes_per_token"]
    assert [g["people"] for g in m["groups"]] == [1, 2, 4, 8, 16, 32, 64]
    assert all(g["bytes"] == b[g["people"]] for g in m["groups"])
    assert [g["text"] for g in m["groups"]] == ["498 MB", "249 MB", "124 MB", "62 MB", "31 MB",
                                                "16 MB", "7.8 MB"]
    assert np.asarray(m["model_tiles"]).shape == (6, 6)
    W, _tok = H.model()
    assert np.allclose(m["model_tiles"], W["h.0.attn.c_attn.weight"][:6, :6])
    assert any("not published" in x for x in m["limitations"])
    assert any("not speeds" in x for x in m["limitations"])


@needs_gpt2
def test_one_tokens_keys_and_values_are_the_scripts_cache():
    token, k, v = Z.one_token()
    assert token == " server" and k.shape == v.shape == (12, 768)
    W, _tok = H.model()
    _s, cache = H.program()["step"](W, H.request()["ids"], [])
    for i, (kk, vv) in enumerate(cache):
        assert np.allclose(k[i], kk[:, Z.FOLLOWED, :].ravel(), atol=1e-6)
        assert np.allclose(v[i], vv[:, Z.FOLLOWED, :].ravel(), atol=1e-6)
    # what one token keeps, counted in numbers: 12 layers x (key + value) x 768
    per_token = sum(kk[:, :1].size + vv[:, :1].size for kk, vv in cache)
    assert 4 * per_token == 73_728


@needs_gpt2
def test_memory_per_conversation():
    m = Z.memory_per_conversation().validate().metadata
    assert (m["per_token"], m["full"], m["context"]) == (73_728, 75_497_472, 1024)
    assert (m["per_token_text"], m["full_text"], m["weights_text"]) == ("73,728", "75 MB",
                                                                      "498 MB")
    assert (m["reply_tokens"], m["prompt_tokens"], m["written"]) == (30, 10, 20)
    assert m["reply_bytes"] == 30 * 73_728 and m["reply_text"] == "2.2 MB"
    assert m["cache_shape"] == "[12, 30, 64]"
    assert (m["tokens_per_gb"], m["full_per_gb"], m["short_per_gb"]) == (13_563, 13, 452)
    assert m["beside"] == 6 and m["weights"] + 6 * m["full"] <= 10 ** 9 < (
        m["weights"] + 7 * m["full"])
    assert m["positions"] == 1024 * 768                    # where GPT-2's limit comes from
    assert any("float64" in x for x in m["limitations"])
    assert any("not published" in x for x in m["limitations"])


@needs_gpt2
def test_the_shares_of_a_call_are_its_tokens():
    _W, tok = H.model()
    msgs = [{"role": r, "content": c} for r, _w, c in H.AGENT[:5]]
    share = Z.shares(msgs)
    assert sum(share) == len(tok.encode(json.dumps(msgs))) == 150
    assert len(share) == 5 and min(share) > 0


@needs_gpt2
def test_coding_agent():
    m = Z.coding_agent().validate().metadata
    calls = m["calls"]
    assert m["tokens_sent"] == [25, 87, 150, 189] == [c["tokens_sent"] for c in H.agent()["calls"]]
    assert (m["total_sent"], m["total_written"]) == (451, 79)
    assert [c["messages_sent"] for c in calls] == [1, 3, 5, 7]
    assert [sum(c["shares"]) for c in calls] == m["tokens_sent"]
    assert [c["tool"] for c in calls] == ["read_file", "edit_file", "run", None]
    assert [c["action"] for c in calls] == ["reads parser.py", "edits parser.py",
                                            "runs pytest", None]
    assert [c["result"] for c in calls[:3]] == [H.AGENT[2][2], H.AGENT[4][2], H.AGENT[6][2]]
    # each call carries every message so far: the laptop's and the service's, in turn
    for c in calls:
        assert c["sides"] == ["device", "service"] * (c["messages_sent"] // 2) + ["device"]
    # what runs where: tools on the laptop ("device"), text from the service
    assert {s["runs_on"] for s in m["steps"] if s["role"] == "tool"} == {"device"}
    assert {s["runs_on"] for s in m["steps"] if s["role"] == "assistant"} == {"service"}
    assert all(len(s) <= Z.WIDE for c in calls for s in c["reply_lines"])
    assert m["asked_lines"] == ["The tests fail in parser.py.", "Please fix it."]
    assert any("written for the animation" in x for x in m["limitations"])
