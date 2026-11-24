"""The numbers behind animations 252-254 of "Blocks and code" (group ``x``, the first three of
"Hosting a language model"): what is on a person's phone or laptop and what is on the
server, what a request is, and what the model is. Everything comes from
:mod:`voxanim.study_hosting`, which loads ``examples/serve.py`` itself and runs it on GPT-2
small (OpenAI, 2019), plus exact arithmetic on GPT-2's shapes. Nothing here is a speed:
every number is a count of numbers or of bytes.

* ``hosting-device-and-server``: the question "What is a server?" (written for the
  animation) typed in an app; the request the script sends is 82 bytes of text
  (``len(body)``); the model, GPT-2 small, is 124,439,808 numbers, 497,759,232 bytes at 4
  bytes each, and stays on the server; the reply comes back token by token, the 20 tokens
  GPT-2 really writes (greedy), 92 bytes of text: " A server is a place where you can
  communicate with other people. It's a place where you can". GPT-2 was not trained to
  chat; the reply is shown as it is. The model is 6,070,234 times the request's bytes.
* ``hosting-a-request-is-text``: region "hs-request" line by line: one message (``said``),
  the request round it (``messages`` and ``max_tokens``), then ``json.dumps(...).encode()``:
  82 bytes, one per letter, space and mark (all of them ASCII). Then the example
  conversation of :data:`~voxanim.study_hosting.CONVERSATION` (written for the animation):
  each turn sends the whole talk so far, 1, 3, then 5 messages: 82, 252, then 422 bytes,
  and 10, 36, then 66 tokens of the prompt the server makes of them (GPT-2's tokenizer).
* ``hosting-the-model-is-a-file``: GPT-2's learned numbers part by part (the token table,
  50,257 x 768 = 38,597,376; the position table, 786,432; each of 12 layers, 7,087,872;
  the last norm, 1,536), 124,439,808 in all; at 4 bytes each (float32) 497,759,232 bytes,
  about 498 MB. ``load()`` reads them once into the server's memory (region "hs-load"),
  and every request's ``reply(W, ...)`` reads that one copy (region "hs-handler").
"""

from __future__ import annotations

import functools
import re
import struct

import numpy as np

from . import study_hosting as H
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("GPT-2 small (OpenAI, 2019; 124,439,808 numbers) served by examples/serve.py, "
          "loaded and run by voxanim.study_hosting; the question and the conversation were "
          "written for the animation, their bytes and tokens counted (GPT-2's tokenizer: a "
          "service's own tokenizer would count a little differently)")
GPT2 = "measured on GPT-2: "             # the evidence tag of a measured phase
BYTES_PER_NUMBER = 4                     # float32, as gpt2.weights() keeps them
LOG_MARKS = (0, 3, 6, 9)                 # 252's log scale: 1, 1,000, 1 million, 1 billion
MARK_NAMES = ("1", "1,000", "1 million", "1 billion")


def mb(n):
    """Bytes as megabytes (10^6), rounded: ``498 MB``."""
    return "%d MB" % round(n / 1e6)


@functools.lru_cache(maxsize=1)
def weights():
    """GPT-2's learned arrays (the masks left out), as the script's ``load()`` gives them."""
    return H.learned(H.model()[0])


def corner(key, rows=6, cols=6):
    """The top-left ``rows x cols`` of a weight array (real numbers to shade tiles with)."""
    return np.asarray(weights()[key], float).reshape(len(weights()[key]), -1)[:rows, :cols]


def sample(arrays, n=240):
    """``n`` numbers spread evenly through ``arrays`` (flattened one after the other): a
    strip of real values for a part of the model."""
    flat = np.concatenate([np.ravel(a) for a in arrays])
    return flat[np.linspace(0, len(flat) - 1, n).astype(int)].astype(float)


# ====================================================================== 252
@functools.lru_cache(maxsize=1)
def device_and_server():
    s = H.sizes()
    r = H.request()
    g = H.generation()
    reply = g["reply"]
    reply_bytes = len(reply.encode())
    model_bytes = s["bytes32"]
    ratio = model_bytes / r["bytes"]
    if model_bytes != BYTES_PER_NUMBER * s["total"]:
        raise ValueError("the model's bytes are not 4 per number")
    rows = (("request", r["bytes"]), ("reply", reply_bytes), ("model", model_bytes))
    meta = dict(
        source=SOURCE, evidence=GPT2 + "GPT-2 small, served by examples/serve.py",
        question=H.QUESTION, json=r["json"], request_bytes=r["bytes"],
        request_bytes_text=commas(r["bytes"]),
        numbers=s["total"], numbers_text=commas(s["total"]),
        model_bytes=model_bytes, model_bytes_text=commas(model_bytes), model_mb=mb(model_bytes),
        reply=reply, reply_tokens=[st["token"] for st in g["steps"]],
        n_reply=len(g["steps"]), reply_bytes=reply_bytes, reply_bytes_text=commas(reply_bytes),
        ratio=ratio, ratio_text=commas(np.floor(ratio)),
        rows=[dict(name=n, bytes=int(b), text=commas(b), log=float(np.log10(b)))
              for n, b in rows],
        marks=list(LOG_MARKS), mark_names=list(MARK_NAMES),
        model_corner=corner("wte.weight").tolist(),
        code_request=list(code_lines("hs-request")), code_load=list(code_lines("hs-load")),
        limitations=[
            "the question was written for the animation; GPT-2 (2019) was not trained to "
            "chat, so its reply is poor: it is shown as it is, the point is what the server "
            "does",
            "GPT-2 small is tiny next to the models real services run; their sizes are "
            "mostly not published, so none is shown",
            "bytes are counted, not times: how long a request takes depends on the "
            "machines and the connection"])
    panels = (
        Panel("The request", "the 82 bytes the app sends (examples/serve.py, region "
              "\"hs-request\"), one per character", "matrix",
              dict(values=np.array([list(r["json"].encode())], float))),
        Panel("The reply", "GPT-2's 20 reply tokens, each with the probability it had when "
              "it was picked (greedy)", "bars",
              dict(values=np.array([st["p"] for st in g["steps"]]),
                   labels=[st["token"] for st in g["steps"]])),
        Panel("The model", "a corner of GPT-2's token table: real numbers of the 124,439,808 "
              "on the server", "matrix", dict(values=corner("wte.weight"))),
        Panel("Sizes, log scale", "log10 of the bytes: the request, the reply, the model "
              "(4 bytes a number)", "bars",
              dict(values=np.array([np.log10(b) for _n, b in rows]),
                   labels=[n for n, _b in rows])),
    )
    return Study("Your device and the server", "The phone or laptop sends 82 bytes and gets "
                 "tokens back; the model, 497,759,232 bytes, stays on the server", panels, meta)


# ====================================================================== 253
def json_lines():
    """The request as the picture builds it: the message, then the request round it (pretty
    printed; the bytes sent are :func:`body`, one line)."""
    said = '{"role": "user",'
    content = ' "content": "%s"}' % H.QUESTION
    return dict(said=[said, content],
                request=['{"messages": [', '  ' + said, '   ' + content + '],',
                         ' "max_tokens": %d}' % H.MAX_TOKENS])


def body_rows(text, width=34):
    """The one-line JSON sent, wrapped to fit at most ``width`` characters a row, broken
    after a space (each space stays at the end of its row: nothing is left out)."""
    rows, row = [], ""
    for word in text.split(" "):
        piece = word if not row else " " + word
        if row and len(row) + len(piece) > width:
            rows.append(row + " ")
            row = word
        else:
            row += piece
    rows.append(row)
    if "".join(rows) != text:
        raise ValueError("the rows are not the text")
    return rows


@functools.lru_cache(maxsize=1)
def request_is_text():
    r = H.request()
    body = r["json"]
    if len(body.encode()) != len(body):
        raise ValueError("the request is not all ASCII")
    turns = H.conversation()
    msgs = []
    for n in range(1, len(H.CONVERSATION) + 1):
        role, content = H.CONVERSATION[n - 1]
        msgs.append(dict(role=role, chars=len(content), text=content))
    comments = {}
    for line in code_lines("hs-request", comments=True):
        if "#" in line:
            code, note = line.split("#", 1)
            comments[re.match(r"\W*(\w+)", code).group(1)] = note.strip()
    meta = dict(
        source=SOURCE, evidence=GPT2 + "an example conversation's bytes and tokens",
        question=H.QUESTION, max_tokens=H.MAX_TOKENS, json=body, bytes=r["bytes"],
        bytes_text=commas(r["bytes"]), rows=body_rows(body), lines=json_lines(),
        prompt=r["prompt"], n_tokens=r["n_tokens"],
        comments=dict(role=comments["said"], content=comments["content"],
                      messages=comments["request"], max_tokens=comments["max_tokens"],
                      body=comments["body"], send=comments["reply"]),
        messages=msgs, turns=turns,
        turn_bytes=[t["bytes"] for t in turns], turn_tokens=[t["tokens"] for t in turns],
        turn_messages=[t["messages"] for t in turns],
        bytes_texts=[commas(t["bytes"]) for t in turns],
        tokens_texts=[commas(t["tokens"]) for t in turns],
        code_request=list(code_lines("hs-request")), code_handler=list(code_lines("hs-handler")),
        limitations=[
            "the conversation was written for the animation; its bytes are counted exactly "
            "and its tokens with GPT-2's tokenizer (a service's own tokenizer would differ a "
            "little)",
            "this server keeps nothing between requests, so the client sends the whole "
            "conversation each time; how a particular service stores conversations is not "
            "shown"])
    panels = (
        Panel("The request's bytes", "the 82 bytes of the JSON request, one per character",
              "matrix", dict(values=np.array([list(body.encode())], float))),
        Panel("Messages", "the example conversation's five messages: characters each",
              "bars", dict(values=np.array([m["chars"] for m in msgs], float),
                           labels=["%d %s" % (i + 1, m["role"]) for i, m in enumerate(msgs)])),
        Panel("Bytes sent", "the request of each user turn: the whole talk so far", "bars",
              dict(values=np.array(meta["turn_bytes"], float),
                   labels=["turn %d" % t["turn"] for t in turns])),
        Panel("Tokens read", "the prompt the server makes of each request, in GPT-2 tokens",
              "bars", dict(values=np.array(meta["turn_tokens"], float),
                           labels=["turn %d" % t["turn"] for t in turns])),
    )
    return Study("A request is text", "A request is 82 bytes of text; each turn sends the whole "
                 "conversation again: 82, 252, then 422 bytes", panels, meta)


# ====================================================================== 254
def parts():
    """GPT-2's learned numbers part by part, in the file's order of use: ``[(name, count,
    arrays)]``; the 12 layers are one part each."""
    W = weights()
    s = H.sizes()
    out = [("token table", W["wte.weight"].size, [W["wte.weight"]]),
           ("positions", W["wpe.weight"].size, [W["wpe.weight"]])]
    for i in range(s["layers"]):
        keys = sorted(k for k in W if k.startswith("h.%d." % i))
        out.append(("layer %d" % (i + 1), sum(W[k].size for k in keys), [W[k] for k in keys]))
    out.append(("last norm", W["ln_f.weight"].size + W["ln_f.bias"].size,
                [W["ln_f.weight"], W["ln_f.bias"]]))
    return [(n, int(c), a) for n, c, a in out]


def float_bytes(v):
    """The 4 bytes of a float32, as the file and the memory hold it (little-endian)."""
    return list(struct.pack("<f", float(v)))


@functools.lru_cache(maxsize=1)
def model_is_a_file():
    s = H.sizes()
    ps = parts()
    total = sum(c for _n, c, _a in ps)
    if total != s["total"]:
        raise ValueError("the parts do not add up")
    if ps[2][1] != s["parts"]["one_layer"] or ps[0][1] != s["parts"]["token_table"]:
        raise ValueError("the parts are not study_hosting's")
    W = weights()
    first = float(W["wte.weight"][0, 0])
    running, so_far = [], 0
    for _n, c, _a in ps:
        so_far += c
        running.append(so_far)
    meta = dict(
        source=SOURCE, evidence=GPT2 + "GPT-2 small's weights, counted",
        parts=[dict(name=n, count=c, text=commas(c), share=c / total,
                    sample=sample(a).tolist()) for n, c, a in ps],
        running=running, running_texts=[commas(v) for v in running],
        total=total, total_text=commas(total), layers=s["layers"],
        layer=s["parts"]["one_layer"], layer_text=commas(s["parts"]["one_layer"]),
        vocab=s["vocab"], vocab_text=commas(s["vocab"]), width=int(W["wte.weight"].shape[1]),
        context=s["context"], context_text=commas(s["context"]),
        bytes_per_number=BYTES_PER_NUMBER,
        bytes=BYTES_PER_NUMBER * total, bytes_text=commas(BYTES_PER_NUMBER * total),
        mb=mb(BYTES_PER_NUMBER * total), bytes16_mb=mb(s["bytes16"]),
        file_bytes=s["file_bytes"], file_mb=mb(s["file_bytes"]),
        first=first, first_text="%.4f" % first, first_bytes=float_bytes(first),
        lo=float(np.percentile(np.abs(sample([W["wte.weight"]], 4000)), 2)),
        hi=float(np.percentile(np.abs(sample([W["wte.weight"]], 4000)), 98)),
        code_load=list(code_lines("hs-load")), code_handler=list(code_lines("hs-handler")),
        limitations=[
            "the weights file on disk (%s bytes) also holds each layer's fixed causal mask "
            "(12 of 1,024 x 1,024), which is not learned; the model's numbers are %s bytes "
            "at 4 bytes each" % (commas(s["file_bytes"]), commas(BYTES_PER_NUMBER * total)),
            "at 2 bytes a number the same model would take %s; a server can keep fewer "
            "bytes per number; real services run far bigger models whose sizes are mostly "
            "not published" % mb(s["bytes16"]),
            "examples/serve.py answers one request at a time; every request reads the same "
            "copy of the weights"])
    tops = ps[:2] + [("12 layers", s["layers"] * s["parts"]["one_layer"], None), ps[-1]]
    panels = (
        Panel("Part by part", "GPT-2's learned numbers: the token table, the positions, the "
              "12 layers, the last norm", "bars",
              dict(values=np.array([c for _n, c, _a in tops], float),
                   labels=[n for n, _c, _a in tops])),
        Panel("Each layer", "the 12 layers, 7,087,872 numbers each", "bars",
              dict(values=np.array([c for _n, c, _a in ps[2:-1]], float),
                   labels=[n for n, _c, _a in ps[2:-1]])),
        Panel("A corner of the token table", "real numbers from the file", "matrix",
              dict(values=corner("wte.weight"))),
        Panel("Bytes", "the model at 4, 2 and 1 bytes a number", "bars",
              dict(values=np.array([s["bytes32"], s["bytes16"], s["bytes8"]], float),
                   labels=["4 bytes", "2 bytes", "1 byte"])),
    )
    return Study("The model is a file of numbers", "GPT-2's 124,439,808 numbers part by part: "
                 "497,759,232 bytes, loaded once and kept for every request", panels, meta)


STUDIES = {"hosting-device-and-server": device_and_server,
           "hosting-a-request-is-text": request_is_text,
           "hosting-the-model-is-a-file": model_is_a_file}
