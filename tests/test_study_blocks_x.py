"""The numbers behind animations 252-254 (voxanim.study_blocks_x): what is on the device and
what is on the server, a request as text, and the model as a file of numbers."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from conftest import needs_gpt2

from voxanim import gpt2
from voxanim import study_blocks_x as X
from voxanim import study_hosting as H
from voxanim.blocks import code_lines

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue():
    return {item["number"]: item for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_and_modules_are_the_catalogues():
    items = catalogue()
    assert [items[n]["title"] for n in (252, 253, 254)] == [
        "Your device and the server", "A request is text", "The model is a file of numbers"]
    assert {items[n]["module"] for n in (252, 253, 254)} == {"study_blocks_x"}
    assert set(X.STUDIES) == {items[n]["name"] for n in (252, 253, 254)}


def test_formats():
    assert X.mb(497_759_232) == "498 MB" and X.mb(548_105_171) == "548 MB"
    assert X.mb(248_879_616) == "249 MB"
    assert X.float_bytes(1.0) == [0, 0, 128, 63]
    v = -0.11010301
    assert struct.unpack("<f", bytes(X.float_bytes(v)))[0] == np.float32(v)


def test_the_body_is_only_wrapped():
    body = json.dumps({"messages": [{"role": "user", "content": "What is a server?"}],
                       "max_tokens": 20})
    rows = X.body_rows(body)
    assert "".join(rows) == body and len(body) == 82
    assert all(len(r.rstrip()) <= 34 for r in rows) and len(rows) == 3
    lines = X.json_lines()
    assert json.loads("".join(lines["said"])) == {"role": "user", "content": "What is a server?"}
    assert json.loads("".join(lines["request"])) == json.loads(body)


def test_the_load_line_says_the_count():
    assert "124,439,808 numbers" in code_lines("hs-load", comments=True)[0]


# ---------------------------------------------------------------------- GPT-2
@needs_gpt2
def test_device_and_server():
    s = X.device_and_server().validate()
    m = s.metadata
    assert m["request_bytes"] == 82 == len(m["json"].encode())
    assert json.loads(m["json"]) == {"messages": [{"role": "user", "content": H.QUESTION}],
                                     "max_tokens": H.MAX_TOKENS}
    assert m["numbers"] == 124_439_808 and m["model_bytes"] == 4 * 124_439_808 == 497_759_232
    assert m["model_mb"] == "498 MB"
    g = H.generation()
    assert m["reply"] == g["reply"] == "".join(m["reply_tokens"])
    assert m["reply"].strip().startswith("A server is a place where you can communicate")
    assert m["n_reply"] == 20 and m["reply_bytes"] == len(m["reply"].encode()) == 92
    assert m["ratio_text"] == "6,070,234" and int(m["ratio"]) == 497_759_232 // 82
    assert [r["name"] for r in m["rows"]] == ["request", "reply", "model"]
    assert np.allclose([r["log"] for r in m["rows"]], np.log10([82, 92, 497_759_232]))
    assert m["marks"] == [0, 3, 6, 9] and m["rows"][-1]["log"] < 9
    W = X.weights()
    assert np.array_equal(np.asarray(m["model_corner"]), W["wte.weight"][:6, :6])
    assert any("not trained to chat" in x for x in m["limitations"])
    assert any("not published" in x for x in m["limitations"])


@needs_gpt2
def test_request_is_text():
    s = X.request_is_text().validate()
    m = s.metadata
    assert m["bytes"] == 82 and "".join(m["rows"]) == m["json"]
    assert m["turn_messages"] == [1, 3, 5]
    assert m["turn_bytes"] == [82, 252, 422] and m["turn_tokens"] == [10, 36, 66]
    assert m["n_tokens"] == 10 and m["prompt"] == "User: What is a server?\nAssistant:"
    # the whole talk is sent every time: each request holds every message before it
    sent = []
    for n in m["turn_messages"]:
        msgs = [{"role": r, "content": c} for r, c in H.CONVERSATION[:n]]
        body = json.dumps({"messages": msgs, "max_tokens": H.MAX_TOKENS})
        assert len(body.encode()) == m["turn_bytes"][len(sent)]
        for earlier in sent:
            assert all(msg in msgs for msg in earlier)
        sent.append(msgs)
    assert m["comments"] == dict(role="who is speaking", content="and what they said",
                                 messages="the whole talk so far",
                                 max_tokens="how long a reply may be",
                                 body="the request: just text", send="off to the server")
    assert [x["chars"] for x in m["messages"]] == [len(c) for _r, c in H.CONVERSATION]
    assert any("written for the animation" in x for x in m["limitations"])


@needs_gpt2
def test_model_is_a_file():
    s = X.model_is_a_file().validate()
    m = s.metadata
    d, f, V, T, L = gpt2.D_MODEL, gpt2.D_MLP, gpt2.N_VOCAB, gpt2.N_CTX, gpt2.N_LAYER
    layer = (d * 3 * d + 3 * d) + (d * d + d) + (d * f + f) + (f * d + d) + 4 * d
    counts = [p["count"] for p in m["parts"]]
    assert counts == [V * d, T * d] + [layer] * L + [2 * d]
    assert counts[:3] == [38_597_376, 786_432, 7_087_872] and counts[-1] == 1_536
    assert [p["name"] for p in m["parts"]][:3] == ["token table", "positions", "layer 1"]
    assert m["total"] == sum(counts) == 124_439_808 == H.sizes()["total"]
    assert m["running"][-1] == m["total"] and m["running_texts"][0] == "38,597,376"
    assert np.all(np.diff(m["running"]) > 0)
    assert m["bytes"] == 497_759_232 == H.sizes()["bytes32"] and m["mb"] == "498 MB"
    assert m["layer_text"] == "7,087,872" and m["vocab_text"] == "50,257" and m["width"] == 768
    W = X.weights()
    assert m["first"] == float(W["wte.weight"][0, 0]) and m["first_text"] == "-0.1101"
    assert m["first_bytes"] == list(np.float32(m["first"]).tobytes())
    assert m["file_bytes"] > m["bytes"]                    # the file also holds the masks
    assert sum(p["share"] for p in m["parts"]) == np.float64(1.0) or \
        abs(sum(p["share"] for p in m["parts"]) - 1.0) < 1e-12
    assert all(len(p["sample"]) == 240 for p in m["parts"])
