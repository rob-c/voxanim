"""The numbers behind animations 255-257 (voxanim.study_blocks_y): a request becomes one text
and then tokens, the reply is written one token at a time, and the key-value cache gives the
same scores for far fewer multiply-adds."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import needs_gpt2

from voxanim import gpt2
from voxanim import study_blocks_y as Y
from voxanim import study_hosting as H
from voxanim.blocks import code_lines

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"
REPLY = (" A server is a place where you can communicate with other people. It's a place where "
         "you can")


def catalogue():
    return {item["number"]: item for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_and_modules_are_the_catalogues():
    items = catalogue()
    assert [items[n]["title"] for n in (255, 256, 257)] == [
        "Text becomes tokens", "One token at a time", "Read once, then remember"]
    assert {items[n]["module"] for n in (255, 256, 257)} == {"study_blocks_y"}
    assert set(Y.STUDIES) == {items[n]["name"] for n in (255, 256, 257)}


def test_formats():
    assert Y.show(" What") == "␣What" and Y.show("\n") == "↵" and Y.show("User") == "User"
    assert Y.pct(0.11263901) == "11.3%" and Y.pct(0.682) == "68.2%"
    assert Y.big_count(2_478_197_760) == "2.5 billion"
    assert Y.big_count(35_682_524_160) == "35.7 billion"
    assert Y.big_count(123_550_464) == "124 million" and Y.big_count(974_095_104) == "974 million"
    assert Y.big_count(410) == "410"


def test_the_request_and_the_ids_break_into_lines_that_join_back():
    body = '{"messages": [{"role": "user", "content": "What is a server?"}], "max_tokens": 20}'
    lines = Y.json_lines(body)
    assert " ".join(lines) == body and len(lines) == 3 and max(map(len, lines)) <= 33
    ids = [12982, 25, 1867, 318, 257, 4382, 30, 198, 48902, 25]
    out = Y.ids_lines(ids)
    assert all(len(s) <= 32 for s in out)
    assert json.loads(" ".join(s.strip() for s in out).replace(", ", ",")) == ids


def test_the_regions_shown_are_the_scripts():
    assert code_lines("hs-tokens") == ("ids = tok.encode(text)",)
    assert code_lines("hs-handler")[2] == 'text = prompt(request["messages"])'
    assert code_lines("hs-generate")[4] == "    scores, cache = step(W, [nxt], cache)"
    assert code_lines("hs-cache")[4] == "new_cache.append((k, v))"
    assert code_lines("hs-request")[7] == "    sys.stdout.write(line.decode())"


def test_without_the_cache_is_every_single_step_again_less_the_scores():
    """Running the whole text again at position p costs what the cached steps for every
    token up to p cost, except that the scores over the vocabulary are computed once."""
    dv = gpt2.D_MODEL * gpt2.N_VOCAB
    for p in (0, 9, 10, 29, 1023):
        alone = sum(H.macs_per_token(t) for t in range(p + 1))
        assert H.macs_per_token(p, cache=False) == alone - p * dv


# ---------------------------------------------------------------------- GPT-2: 255
@needs_gpt2
def test_text_to_tokens():
    s = Y.text_to_tokens().validate()
    m = s.metadata
    assert m["prompt"] == "User: What is a server?\nAssistant:"
    assert m["prompt"] == H.program()["prompt"]([{"role": "user", "content": m["question"]}])
    assert (m["bytes"], m["chars"], m["n_tokens"]) == (82, 34, 10)
    assert m["ids"] == [12982, 25, 1867, 318, 257, 4382, 30, 198, 48902, 25]
    assert m["shown"] == ["User", ":", "␣What", "␣is", "␣a", "␣server", "?", "↵", "Assistant",
                          ":"]
    assert "".join(m["tokens"]) == m["prompt"]
    _W, tok = H.model()
    assert tok.encode(m["prompt"]) == m["ids"]
    assert [tok.decode([i]) for i in m["ids"]] == m["tokens"]
    assert m["repeated"] == [1, 9] and m["repeated_id"] == 25
    assert " ".join(m["json_lines"]) == m["json"] and len(m["json"].encode()) == 82
    assert (m["vocab"], m["context"], m["max_tokens"]) == (50_257, 1_024, 20)
    assert all(0 <= i < m["vocab"] for i in m["ids"])


# ---------------------------------------------------------------------- GPT-2: 256
@needs_gpt2
def test_one_token_at_a_time():
    s = Y.one_token_at_a_time().validate()
    m = s.metadata
    assert m["reply"] == REPLY and m["n"] == 20 and m["prompt_tokens"] == 10
    assert "".join(m["tokens"]) == m["reply"] and m["stops_mid_sentence"]
    assert m["text_after"] == list(range(11, 31))
    first = m["steps"][0]
    assert (first["shown"], first["p_text"], first["id"]) == ("␣A", "11.3%", 317)
    assert [t["shown"] for t in first["top"]] == ["␣A", "␣It", "␣Server", "␣The", "␣I"]
    assert [t["p_text"] for t in first["top"]] == ["11.3%", "5.1%", "4.8%", "4.7%", "3.6%"]
    g = H.generation()
    for st, raw in zip(m["steps"], g["steps"], strict=True):
        assert st["token"] == raw["token"] == st["top"][0]["token"]       # greedy
        ps = [t["p"] for t in st["top"]]
        assert ps == sorted(ps, reverse=True) and sum(ps) <= 1.0
        assert len(st["top"]) == 5
    # each step's probabilities are the softmax of the scores the script's step() gave
    for k in (0, 7, 19):
        p = gpt2.softmax(g["scores"][k])
        assert np.isclose(p[m["steps"][k]["id"]], m["steps"][k]["p"])
        assert int(p.argmax()) == m["steps"][k]["id"]
    assert any("not trained to chat" in x for x in m["limitations"])


# ---------------------------------------------------------------------- GPT-2: 257
@needs_gpt2
def test_read_once_remember():
    s = Y.read_once_remember().validate()
    m = s.metadata
    c = H.costs()
    assert (m["prompt"], m["n"], m["total"]) == (10, 20, 30)
    assert m["tokens_without"] == list(range(11, 31)) and m["tokens_with"] == [1] * 20
    assert (m["run_without"], m["run_with"]) == (410, 20)
    assert (m["total_with"], m["total_without"]) == (2_478_197_760, 35_682_524_160)
    assert sum(m["macs_with"]) == c["with_cache"] and sum(m["macs_without"]) == c["without_cache"]
    assert m["ratio_text"] == "14.4" and m["ratio_round"] == 14
    assert (m["total_with_text"], m["total_without_text"]) == ("2.5 billion", "35.7 billion")
    assert m["cache_shape"] == [12, 30, 64] and m["cache_shape_text"] == "[12, 30, 64]"
    assert m["per_token"] == 18_432 and 4 * m["per_token"] == H.memory()["per_token"]
    k = np.asarray(m["k_norm"])
    assert k.shape == (12, 30) and np.all(k > 0) and np.allclose(k.max(axis=1), 1.0)
    assert m["reply"] == REPLY


@needs_gpt2
def test_the_cache_gives_the_same_scores_as_reading_again():
    m = Y.read_once_remember().metadata
    check = H.cache_check()
    assert m["max_difference"] == check["max_difference"]
    assert m["same_tokens"] and check["same_tokens"] and m["same_count"] == 20
    assert m["max_difference_text"] == "0.0000068" and m["max_difference_sci"] == "6.8e-06"
    assert len(m["diffs"]) == 20 and max(m["diffs"]) < 1e-5
