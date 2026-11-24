"""The numbers behind animations 258-260 of "Blocks and code" (group ``z``, the middle of
"Hosting a language model"): many people served by one copy of the model, the memory each
conversation takes, and a coding agent's loop. Everything comes from
:mod:`voxanim.study_hosting`, which runs ``examples/serve.py`` itself on GPT-2 small, or is
exact arithmetic from GPT-2's shapes. Counts, not speeds: nothing here is a time.

* ``hosting-many-people-one-model``: four people's questions (:data:`~voxanim.study_hosting.
  PEOPLE`, written for the animation) are read, each person's first reply token written
  (" A", " It", " I", " Hello"); then their next tokens (" server", "'s", "'m", "!") one step
  each with the script's ``step``, and all four at once with its ``batch_step`` (region
  "hs-batch": a row per person, each weight read once for all of them): the same tokens,
  the scores within 2e-6. One step reads all 124,439,808 weights, 497,759,232 bytes at 4
  bytes each, however many people share it: 498 MB a token for one person, 7.8 MB each
  when 64 share the step.
* ``hosting-memory-per-conversation``: what the key-value cache keeps for one token: a key
  and a value of 768 numbers in each of 12 layers, 73,728 bytes at 4 bytes a number (the
  real keys and values of " server" in "What is a server?" are drawn). The real reply's
  conversation (10 tokens asked, 20 written) keeps 30 tokens, 2.2 MB; GPT-2 has 1,024
  places (its position table), so a full conversation keeps 75 MB. Beside the model's 498
  MB, shared by everyone, each conversation's cache is its own: a GB holds 13,563 tokens of
  cache, 13 full conversations or 452 like the real reply.
* ``hosting-a-coding-agent``: :data:`~voxanim.study_hosting.AGENT`, a coding agent's
  conversation written for the animation. Each of its 4 calls to the service sends the
  whole conversation so far: 25, 87, 150, 189 tokens (GPT-2's tokenizer on the JSON sent);
  the model answers with text, three times asking for a tool (read a file, edit it, run
  the tests), which the agent runs on the laptop, and then with the answer. Each message's
  share of every call is counted from the tokens themselves.
"""

from __future__ import annotations

import functools
import json

import numpy as np

from . import gpt2
from . import study_hosting as H
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("GPT-2 small (OpenAI, 2019; 124,439,808 numbers) served by examples/serve.py, run "
          "by voxanim.study_hosting (voxanim's own numpy GPT-2); counts, not speeds")
GROUPS = (1, 2, 4, 8, 16, 32, 64)       # people sharing one step (258)
MB, GB = 10 ** 6, 10 ** 9               # bytes: a megabyte, a gigabyte
WIDE = 31                               # 260: characters of a line of text on screen
FOLLOWED = 5                            # 259: the token whose keys and values are drawn


def mb(n):
    """Bytes as megabytes on screen: ``498 MB``, ``7.8 MB`` (whole above 10)."""
    v = n / MB
    return "%s MB" % commas(v) if v >= 9.95 else "%.1f MB" % v


def gb(n):
    """Bytes as gigabytes: ``1.99 GB``."""
    return "%.2f GB" % (n / GB)


@functools.lru_cache(maxsize=1)
def weights():
    """The bytes of GPT-2's learned numbers at 4 bytes each, and how many there are."""
    total = H.sizes()["total"]
    return dict(numbers=total, bytes=4 * total)


def model_sample(rows=6, cols=6):
    """A corner of a real weight matrix (layer 0's ``attn.c_attn.weight``, the ``w`` of
    region "hs-batch"), for the orange tiles that stand for the model."""
    W, _tok = H.model()
    a = np.asarray(W["h.0.attn.c_attn.weight"][:rows, :cols], float)
    return dict(model_tiles=a.tolist(), model_lo=float(a.min()), model_hi=float(a.max()))


# ====================================================================== 258
@functools.lru_cache(maxsize=1)
def people():
    """Four people, as the script serves them: each question made a prompt and read once
    (``step`` on the whole prompt), the first reply token picked; then the next token one
    step each (``step``) and for all four in one ``batch_step``. Per person: the question,
    its tokens, the first token, the next token both ways and their scores' difference."""
    W, tok = H.model()
    p = H.program()
    rows, singles, out = [], [], []
    for q in H.PEOPLE:
        ids = tok.encode(p["prompt"]([{"role": "user", "content": q}]))
        scores, cache = p["step"](W, ids, [])
        first = int(scores.argmax())
        rows.append((first, cache))
        singles.append(p["step"](W, [first], cache)[0])
        out.append(dict(question=q, prompt_tokens=len(ids), first=tok.decode([first])))
    together = [s for s, _c in p["batch_step"](W, rows)]
    for o, a, b in zip(out, singles, together, strict=True):
        o.update(one_each=tok.decode([int(a.argmax())]), together=tok.decode([int(b.argmax())]),
                 difference=float(np.abs(a - b).max()))
    return out


def per_token(n):
    """Weight bytes read per token when ``n`` people share a step: the step reads every
    weight once, whoever it is for."""
    return weights()["bytes"] / n


def many_people():
    w = weights()
    ps = people()
    if not all(o["one_each"] == o["together"] for o in ps):
        raise ValueError("batch_step does not give the tokens of one step each")
    n = len(ps)
    worst = max(o["difference"] for o in ps)
    groups = [dict(people=g, bytes=per_token(g), text=mb(per_token(g))) for g in GROUPS]
    meta = dict(
        source=SOURCE, evidence="measured on GPT-2: one step each against one batch_step",
        **model_sample(),
        people=ps, n=n, questions=[o["question"] for o in ps],
        first=[o["first"] for o in ps], next=[o["together"] for o in ps],
        prompt_tokens=[o["prompt_tokens"] for o in ps],
        max_difference=worst, difference_text="%.0e" % worst,
        numbers=w["numbers"], numbers_text=commas(w["numbers"]),
        weight_bytes=w["bytes"], weight_bytes_text=commas(w["bytes"]),
        weight_text=mb(w["bytes"]),
        one_each_bytes=n * w["bytes"], one_each_text=mb(n * w["bytes"]),
        together_bytes=w["bytes"], together_text=mb(w["bytes"]),
        row=gpt2.D_MODEL, batch_shape="[%d, %d]" % (n, gpt2.D_MODEL),
        qkv_shape="[%d, %d]" % (n, 3 * gpt2.D_MODEL),
        groups=groups, most=GROUPS[-1], least_text=groups[-1]["text"],
        code_generate=list(code_lines("hs-generate")), code_batch=list(code_lines("hs-batch")),
        limitations=[
            "the four questions are written for the animation; GPT-2 (2019) was not trained "
            "to chat, and its tokens are what it gives",
            "one step after each person's first reply token; batch_step's scores differ "
            "from one step each by at most %.1e (the order of the sums), and the same "
            "token wins for all four" % worst,
            "weights read are counted as bytes at 4 bytes a number; they are counts, not "
            "speeds: how long a step takes depends on the machine",
            "attention is still done person by person in batch_step (each looks at their "
            "own past); only the weights are shared",
            "real services run models far larger than GPT-2, whose sizes are mostly not "
            "published"])
    panels = (
        Panel("Four people's prompts", "tokens in each person's prompt (questions written "
              "for the animation)", "bars",
              dict(values=np.array(meta["prompt_tokens"], float), labels=meta["questions"])),
        Panel("One step each, or one for all", "weight bytes read for one more token for "
              "each of the four", "bars",
              dict(values=np.array([meta["one_each_bytes"], meta["together_bytes"]], float),
                   labels=["one step each", "one batch_step"])),
        Panel("The same scores", "largest difference between a person's scores from "
              "batch_step and from a step of their own", "bars",
              dict(values=np.array([o["difference"] for o in ps], float),
                   labels=["%s → %s" % (o["question"], o["together"]) for o in ps])),
        Panel("Weights read per token", "bytes of weights read per token when 1 to 64 "
              "people share a step", "bars",
              dict(values=np.array([g["bytes"] for g in groups], float),
                   labels=[str(g) for g in GROUPS])),
    )
    return Study("Many people, one model", "One step for four people reads the weights once "
                 "and gives each the same token; 498 MB a token alone, 7.8 MB among 64",
                 panels, meta)


# ====================================================================== 259
@functools.lru_cache(maxsize=1)
def one_token():
    """The keys and values the cache keeps for one token (:data:`FOLLOWED` of the
    request's prompt, " server"): ``(token, keys [12, 768], values [12, 768])``, read from
    the script's ``step`` on the prompt."""
    W, _tok = H.model()
    r = H.request()
    _scores, cache = H.program()["step"](W, r["ids"], [])
    k = np.stack([c[0][:, FOLLOWED, :].ravel() for c in cache])
    v = np.stack([c[1][:, FOLLOWED, :].ravel() for c in cache])
    return r["tokens"][FOLLOWED], k.astype(np.float32), v.astype(np.float32)


def memory_per_conversation():
    m = H.memory()
    per = m["per_token"]
    if per != 2 * gpt2.N_LAYER * gpt2.D_MODEL * 4:
        raise ValueError("the bytes per token are not 2 x 12 x 768 x 4")
    token, k, v = one_token()
    if k.shape != (gpt2.N_LAYER, gpt2.D_MODEL):
        raise ValueError("one token's keys are not 12 x 768")
    full, wbytes = m["full_context"], m["weights"]
    reply_tokens, reply_bytes = m["reply_cache_tokens"], m["reply_cache_bytes"]
    beside = (GB - wbytes) // full                 # full conversations beside the model
    full_per_gb = GB // full
    short_per_gb = GB // reply_bytes
    hi = float(np.percentile(np.abs(np.r_[k.ravel(), v.ravel()]), 98))   # the shading's ends
    lo = -hi
    curve_x = np.arange(0, m["context"] + 1, 16, dtype=float)
    meta = dict(
        source=SOURCE, evidence="exact arithmetic from GPT-2's shapes, and its measured cache",
        **model_sample(),
        token=token, keys=k.tolist(), values=v.tolist(), lo=lo, hi=hi,
        layers=gpt2.N_LAYER, width=gpt2.D_MODEL, heads=gpt2.N_HEAD, bytes_per_number=4,
        per_token=per, per_token_text=commas(per),
        context=m["context"], context_text=commas(m["context"]),
        full=full, full_text=mb(full), full_bytes_text=commas(full),
        reply_tokens=reply_tokens, reply_bytes=reply_bytes, reply_text=mb(reply_bytes),
        prompt_tokens=H.request()["n_tokens"], written=H.MAX_TOKENS,
        cache_shape="[%d, %d, %d]" % (gpt2.N_HEAD, reply_tokens, gpt2.D_MODEL // gpt2.N_HEAD),
        weights=wbytes, weights_text=mb(wbytes),
        positions=H.sizes()["parts"]["position_table"],
        gb=GB, tokens_per_gb=m["tokens_per_gb"], tokens_per_gb_text=commas(m["tokens_per_gb"]),
        beside=int(beside), beside_bytes=int(beside * full),
        full_per_gb=int(full_per_gb), short_per_gb=int(short_per_gb),
        code_cache=list(code_lines("hs-cache")), code_generate=list(code_lines("hs-generate")),
        code_load=list(code_lines("hs-load")), code_handler=list(code_lines("hs-handler")),
        limitations=[
            "bytes are counted at 4 bytes a number (float32, as the weights file stores "
            "them); the numpy script happens to keep most of its cache in float64, which "
            "would take twice as much",
            "GPT-2's limit of %d tokens comes from its position table (%d rows); a "
            "service also limits a conversation because every token kept costs memory "
            "while the conversation lasts" % (m["context"], m["context"]),
            "a GB here is 10^9 bytes, a unit for counting, not the memory of any machine",
            "the conversation of the real reply is the question 'What is a server?' (10 "
            "tokens) and GPT-2's 20 tokens: the cache then holds %d tokens" % reply_tokens,
            "real services run models far larger than GPT-2, whose sizes are mostly not "
            "published; what they keep per token depends on their shapes"])
    panels = (
        Panel("One token's keys", "the keys the cache keeps for %r: 768 numbers in each of "
              "12 layers" % token, "matrix", dict(values=k)),
        Panel("A conversation's cache", "bytes kept against the tokens of the conversation, "
              "up to GPT-2's 1,024", "curve",
              dict(x=curve_x, ys=np.atleast_2d(curve_x * per), labels=["cache bytes"])),
        Panel("Memory", "bytes of the model, of one full conversation's cache and of the "
              "real reply's", "bars",
              dict(values=np.array([wbytes, full, reply_bytes], float),
                   labels=["the model", "a full conversation", "the real reply"])),
        Panel("Conversations per GB", "how many conversations' caches fit in 10^9 bytes",
              "bars", dict(values=np.array([full_per_gb, short_per_gb], float),
                           labels=["1,024 tokens", "%d tokens" % reply_tokens])),
    )
    return Study("Why a conversation has a limit", "Each token kept costs 73,728 bytes: 75 MB "
                 "for GPT-2's 1,024 tokens, 13,563 tokens per GB", panels, meta)


# ====================================================================== 260
def lines(text, wide=WIDE):
    """``text`` as lines of at most ``wide`` characters: a line of ``text`` that fits stays
    as it is (with its indent); a longer one is broken after ", " where it is JSON, at
    spaces otherwise."""
    out = []
    for part in text.rstrip("\n").split("\n"):
        if len(part) <= wide:
            out.append(part)
            continue
        json_like = part.startswith("{")
        chunks = [c.strip() for c in part.split(", ")] if json_like else part.split(" ")
        line = ""
        for i, c in enumerate(chunks):
            if json_like and i < len(chunks) - 1:
                c += ","
            trial = line + " " + c if line else c
            if line and len(trial) > wide:
                out.append(line)
                line = c
            else:
                line = trial
        out.append(line)
    if any(len(s) > wide for s in out):
        raise ValueError("a line of %r is too wide" % text)
    return out


def shares(messages):
    """How many of the tokens of ``json.dumps(messages)`` belong to each message (each
    token to the message its first character falls in; the brackets and commas between
    to the message before): they add up to the call's tokens exactly."""
    _W, tok = H.model()
    text = json.dumps(messages)
    ids = tok.encode(text)
    ends, at = [], 0
    for m in messages:
        at = text.index(json.dumps(m), at) + len(json.dumps(m))
        ends.append(at)
    ends[-1] = len(text)
    count, pos = [0] * len(messages), 0
    for i in ids:
        j = next(n for n, e in enumerate(ends) if pos < e)
        count[j] += 1
        pos += len(tok.decode([i]))
    if pos != len(text) or sum(count) != len(ids):
        raise ValueError("the tokens do not spell the text sent")
    return count


ACTIONS = dict(read_file="reads %s", edit_file="edits %s", run="runs %s")


def coding_agent():
    a = H.agent()
    steps = a["steps"]
    calls = []
    n_assist = 0
    for n, (role, where, text) in enumerate(H.AGENT):
        if role != "assistant":
            continue
        call = a["calls"][n_assist]
        n_assist += 1
        sent = [{"role": r, "content": c} for r, _w, c in H.AGENT[:n]]
        share = shares(sent)
        if sum(share) != call["tokens_sent"]:
            raise ValueError("the shares do not add up to the tokens sent")
        tool = json.loads(text) if call["tool"] else None
        result = H.AGENT[n + 1][2] if n + 1 < len(H.AGENT) else None
        target = None if tool is None else tool.get("path", tool.get("command"))
        calls.append(dict(
            call=call["call"], messages_sent=call["messages_sent"],
            tokens_sent=call["tokens_sent"], reply_tokens=call["reply_tokens"],
            shares=share, sides=[w for _r, w, _c in H.AGENT[:n]], runs_on=where,
            reply=text, reply_lines=lines(text), tool=None if tool is None else tool["tool"],
            target=target,
            action=None if tool is None else ACTIONS[tool["tool"]] % target,
            result=result, result_lines=None if result is None else lines(result, 40)))
    sent = [c["tokens_sent"] for c in calls]
    if sent != sorted(sent):
        raise ValueError("the conversation sent does not grow")
    meta = dict(
        source=SOURCE + "; the conversation is written for the animation, its tokens "
        "counted with GPT-2's tokenizer",
        evidence="measured on GPT-2's tokenizer: the tokens of each call",
        **model_sample(),
        steps=steps, calls=calls, tokens_sent=sent, total_sent=a["total_sent"],
        total_written=a["total_written"], asked=H.AGENT[0][2],
        asked_lines=lines(H.AGENT[0][2], 28), most=max(sent),
        code_request=list(code_lines("hs-request")),
        code_handler=list(code_lines("hs-handler")),
        limitations=[
            "the conversation is written for the animation; GPT-2 was not trained to use "
            "tools: a service's model is trained to write such requests",
            "tokens are counted with GPT-2's tokenizer on the JSON sent; a service's own "
            "tokenizer, and the instructions and tool descriptions a real agent also "
            "sends, would change the counts",
            "examples/serve.py has the client (ask) and the server, not the agent's tools: "
            "running a tool is described, not shown as code",
            "the files, and the commands run, stay on the laptop; the service only sees "
            "the text the agent sends"])
    panels = (
        Panel("Tokens sent per call", "the whole conversation so far, as JSON, counted with "
              "GPT-2's tokenizer", "bars",
              dict(values=np.array(sent, float), labels=["call %d" % c["call"] for c in calls])),
        Panel("Tokens written per call", "the model's text: three tool requests, then the "
              "answer", "bars",
              dict(values=np.array([c["reply_tokens"] for c in calls], float),
                   labels=[c["tool"] or "answer" for c in calls])),
        Panel("Messages sent per call", "every call carries every message so far", "bars",
              dict(values=np.array([c["messages_sent"] for c in calls], float),
                   labels=["call %d" % c["call"] for c in calls])),
        Panel("Each message's share", "tokens of each message (columns) in each call (rows)",
              "matrix",
              dict(values=np.array([c["shares"] + [0] * (len(H.AGENT) - 1 - len(c["shares"]))
                                    for c in calls], float))),
    )
    return Study("A coding agent's loop", "Four calls, each with the whole conversation (25, "
                 "87, 150, 189 tokens); the tools run on the laptop, the model only writes "
                 "text", panels, meta)


STUDIES = {"hosting-many-people-one-model": many_people,
           "hosting-memory-per-conversation": memory_per_conversation,
           "hosting-a-coding-agent": coding_agent}
