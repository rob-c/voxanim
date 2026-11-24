"""Hosting a language model, for the hosting animations of "Blocks and code" (catalogue
252-262): every number they show, from running ``examples/serve.py`` itself (loaded with
:func:`runpy.run_path`, so the program on screen is the program measured) on GPT-2 small,
plus exact arithmetic from GPT-2's shapes.

* :func:`sizes`: GPT-2's numbers, part by part, and the bytes they take (the weights file is
  what the server keeps in memory; a request is a few hundred bytes).
* :func:`request`, :func:`conversation`: a request as text, and how it grows: the whole
  conversation is sent every time (an example conversation, written for the animation; its
  bytes and tokens are counted).
* :func:`generation`: a real reply, token by token (greedy), each with the model's top
  five next tokens and their probabilities; the reply GPT-2 actually gives, poor as it is.
* :func:`cache_check`: the reply's scores with the key-value cache and recomputed from
  scratch agree; :func:`costs`: the multiply-adds per token with and without the cache.
* :func:`batching`: ``batch_step`` against one step per person (the same scores), and the
  weights read per token as more people share a step.
* :func:`memory`: what each conversation's cache takes, per token and per context.
* :func:`agent`: an example coding agent's conversation (written for the animation), what
  is sent each turn and what runs where; tokens counted with GPT-2's tokenizer.

Speeds are not measured: they depend on the machine. Everything here is a count.
"""

from __future__ import annotations

import functools
import json
import os
import runpy

import numpy as np

from . import gpt2

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "examples", "serve.py")
QUESTION = "What is a server?"
MAX_TOKENS = 20
TOP = 5


@functools.lru_cache(maxsize=1)
def program():
    """``examples/serve.py``'s functions (its ``__main__`` block not run)."""
    return runpy.run_path(SCRIPT, run_name="voxanim.study_hosting.serve")


@functools.lru_cache(maxsize=1)
def model():
    """``load()`` of the script: ``(W, tok)``."""
    return program()["load"]()


# ====================================================================== sizes
def learned(W):
    """The arrays the model learned: the weights file also holds each layer's causal mask
    (``h.N.attn.bias``, 1,024 x 1,024 of ones and zeros), which is not learned."""
    return {k: v for k, v in W.items() if not k.endswith((".attn.bias", ".attn.masked_bias"))}


def sizes():
    """GPT-2's numbers by part, and their bytes at 4, 2 and 1 bytes per number; the weights
    file on disk also holds the 12 masks (``file_bytes``)."""
    W = learned(model()[0])
    parts = dict(token_table=W["wte.weight"].size, position_table=W["wpe.weight"].size,
                 one_layer=sum(v.size for k, v in W.items() if k.startswith("h.0.")),
                 final_norm=W["ln_f.weight"].size + W["ln_f.bias"].size)
    total = int(sum(v.size for v in W.values()))
    return dict(parts={k: int(v) for k, v in parts.items()}, layers=gpt2.N_LAYER,
                total=total, bytes32=4 * total, bytes16=2 * total, bytes8=total,
                file_bytes=os.path.getsize(os.path.join(gpt2.DIR, "model.safetensors")),
                vocab=gpt2.N_VOCAB, context=gpt2.N_CTX)


# ====================================================================== the request
def request(question=QUESTION):
    """The request the client sends (region "hs-request"), as JSON text, and the prompt the
    server makes of it (``prompt``), with its tokens."""
    said = {"role": "user", "content": question}
    req = {"messages": [said], "max_tokens": MAX_TOKENS}
    body = json.dumps(req).encode()
    text = program()["prompt"](req["messages"])
    _W, tok = model()
    ids = tok.encode(text)
    return dict(json=body.decode(), bytes=len(body), prompt=text, ids=ids,
                tokens=[tok.decode([i]) for i in ids], n_tokens=len(ids))


CONVERSATION = (               # an example conversation, written for the animation
    ("user", "What is a server?"),
    ("assistant", "A computer somewhere else that answers requests sent over the internet."),
    ("user", "So is the model on my phone?"),
    ("assistant", "No: your phone sends your words; the model runs on the server."),
    ("user", "Then why does a long chat get slower?"),
)


def conversation():
    """Each request of :data:`CONVERSATION`: the whole talk so far is sent every time. Per
    user turn: messages sent, bytes, and the prompt's tokens."""
    _W, tok = model()
    out = []
    for n in range(1, len(CONVERSATION) + 1, 2):
        msgs = [{"role": r, "content": c} for r, c in CONVERSATION[:n]]
        body = json.dumps({"messages": msgs, "max_tokens": MAX_TOKENS}).encode()
        out.append(dict(turn=(n + 1) // 2, messages=n, bytes=len(body),
                        tokens=len(tok.encode(program()["prompt"](msgs)))))
    return out


# ====================================================================== the reply
@functools.lru_cache(maxsize=1)
def generation(question=QUESTION, max_tokens=MAX_TOKENS):
    """The reply GPT-2 gives, token by token, run by the script's ``step`` exactly as
    ``reply`` runs it: per token, its text, its probability and the top five. Also every
    step's scores, for :func:`cache_check`."""
    W, tok = model()
    step = program()["step"]
    ids = request(question)["ids"]
    scores, cache = step(W, ids, [])
    out, all_scores, sent = [], [], []
    for _ in range(max_tokens):
        p = gpt2.softmax(scores)
        top = np.argsort(-p)[:TOP]
        nxt = int(scores.argmax())
        out.append(dict(token=tok.decode([nxt]), id=nxt, p=float(p[nxt]),
                        top=[(tok.decode([int(i)]), float(p[i])) for i in top]))
        all_scores.append(scores)
        sent.append(nxt)
        scores, cache = step(W, [nxt], cache)
    return dict(prompt_ids=ids, ids=sent, steps=out, reply=tok.decode(sent),
                scores=np.array(all_scores), cache_layers=len(cache),
                cache_shape=tuple(cache[0][0].shape))


def cache_check():
    """Every reply token's scores, computed again from scratch on the whole text so far
    (no cache): the largest difference, and whether the same token wins every time."""
    W, _tok = model()
    step = program()["step"]
    g = generation()
    ids = list(g["prompt_ids"])
    worst, same = 0.0, True
    for k, nxt in enumerate(g["ids"]):
        fresh, _ = step(W, ids, [])
        worst = max(worst, float(np.abs(fresh - g["scores"][k]).max()))
        same &= int(fresh.argmax()) == int(g["scores"][k].argmax())
        ids.append(nxt)
    return dict(max_difference=worst, same_tokens=bool(same), checked=len(g["ids"]))


def macs_per_token(position, cache=True):
    """Multiply-adds for the token at ``position`` (0-based) with the cache: its own
    matrix products in every layer, attention over ``position + 1`` places, and the scores
    over the vocabulary. Without the cache, every earlier token is run again too."""
    d, f, L, V = gpt2.D_MODEL, gpt2.D_MLP, gpt2.N_LAYER, gpt2.N_VOCAB
    mats = d * 3 * d + d * d + d * f + f * d                 # one token through one layer
    if cache:
        return L * (mats + 2 * d * (position + 1)) + d * V
    tokens = position + 1
    attention = 2 * d * tokens * (tokens + 1) // 2           # each looks at those before it
    return L * (mats * tokens + attention) + d * V


def costs(prompt=None, n=MAX_TOKENS):
    """Multiply-adds to write ``n`` reply tokens after a prompt of ``prompt`` tokens, with
    and without the cache (the prompt is read once in both)."""
    prompt = request()["n_tokens"] if prompt is None else prompt
    read = sum(macs_per_token(t) for t in range(prompt))     # reading the prompt, once
    with_cache = sum(macs_per_token(prompt + k) for k in range(n))
    without = sum(macs_per_token(prompt + k, cache=False) for k in range(n))
    return dict(prompt=prompt, tokens=n, read_prompt=read, with_cache=with_cache,
                without_cache=without, ratio=without / with_cache,
                per_token_one_layer=gpt2.D_MODEL * 3 * gpt2.D_MODEL + gpt2.D_MODEL ** 2
                + 2 * gpt2.D_MODEL * gpt2.D_MLP)


# ====================================================================== many people
PEOPLE = ("What is a server?", "Why is the sky blue?", "Tell me a story.", "Hello!")


def batching():
    """Four people's first reply token, one step each against one ``batch_step`` for all:
    the same scores; and the weight bytes read per token as more people share a step."""
    W, tok = model()
    p = program()
    rows, singles = [], []
    for q in PEOPLE:
        ids = tok.encode(p["prompt"]([{"role": "user", "content": q}]))
        scores, cache = p["step"](W, ids, [])
        nxt = int(scores.argmax())
        rows.append((nxt, cache))
        singles.append(p["step"](W, [nxt], cache)[0])
    together = [s for s, _c in p["batch_step"](W, rows)]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(singles, together, strict=True))
    same = all(int(a.argmax()) == int(b.argmax()) for a, b in zip(singles, together,
                                                                   strict=True))
    weights = 4 * sizes()["total"]
    return dict(people=len(PEOPLE), max_difference=diff, same_tokens=same,
                next_tokens=[tok.decode([int(s.argmax())]) for s in together],
                weight_bytes=weights,
                bytes_per_token={b: weights / b for b in (1, 2, 4, 8, 16, 32, 64)})


# ====================================================================== memory
def memory(context=gpt2.N_CTX):
    """What one conversation's cache takes: keys and values, every layer, per token."""
    per_token = 2 * gpt2.N_LAYER * gpt2.D_MODEL * 4               # float32
    g = generation()
    return dict(per_token=per_token, full_context=per_token * context, context=context,
                tokens_per_gb=10 ** 9 // per_token,
                weights=4 * sizes()["total"],
                reply_cache_tokens=g["cache_shape"][1],
                reply_cache_bytes=per_token * g["cache_shape"][1])


# ====================================================================== a coding agent
AGENT = (                      # an example coding agent's conversation, written for the animation
    ("user", "device", "The tests fail in parser.py. Please fix it."),
    ("assistant", "service", '{"tool": "read_file", "path": "parser.py"}'),
    ("tool", "device", "def parse(line):\n    return line.split(',')[1]\n"),
    ("assistant", "service", '{"tool": "edit_file", "path": "parser.py", '
                             '"old": "[1]", "new": "[0]"}'),
    ("tool", "device", "parser.py: 1 line changed"),
    ("assistant", "service", '{"tool": "run", "command": "pytest"}'),
    ("tool", "device", "3 passed"),
    ("assistant", "service", "Fixed: parse() took the second field; it now takes the first. "
                             "The 3 tests pass."),
)


def agent():
    """Each call to the service in :data:`AGENT`: what is sent (the whole conversation so
    far), its tokens, what comes back, and where each step runs (``device``: the tool on the
    person's computer reads files and runs commands; ``service``: the model writes text)."""
    _W, tok = model()
    calls = []
    for n, (role, _where, text) in enumerate(AGENT):
        if role != "assistant":
            continue
        sent = [{"role": r, "content": c} for r, _w, c in AGENT[:n]]
        calls.append(dict(call=len(calls) + 1, messages_sent=len(sent),
                          tokens_sent=len(tok.encode(json.dumps(sent))),
                          reply=text, reply_tokens=len(tok.encode(text)),
                          tool=text.startswith('{"tool"')))
    return dict(steps=[dict(role=r, runs_on=w, text=t) for r, w, t in AGENT], calls=calls,
                total_sent=sum(c["tokens_sent"] for c in calls),
                total_written=sum(c["reply_tokens"] for c in calls))


def summary():
    s = sizes()
    return dict(sizes=s, request={k: v for k, v in request().items() if k != "tokens"},
                conversation=conversation(), reply=generation()["reply"],
                cache_check=cache_check(), costs=costs(), batching={
                    k: v for k, v in batching().items() if k != "bytes_per_token"},
                memory=memory(), agent=agent()["calls"])
