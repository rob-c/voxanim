"""The numbers behind animations 255-257 of "Blocks and code" (group ``y``, the middle of
"Hosting a language model"): what the server does with a request, from text to tokens to a
reply, and the key-value cache that spares it from reading everything again. Everything
comes from :mod:`voxanim.study_hosting`, which runs ``examples/serve.py`` itself on GPT-2
small (OpenAI, 2019; voxanim's own numpy GPT-2).

* ``hosting-text-to-tokens``: the request (region "hs-request", 82 bytes of JSON) becomes
  one text (``prompt`` in region "hs-handler": "User: What is a server?\\nAssistant:", 34
  characters), then 10 tokens (region "hs-tokens", GPT-2's tokenizer): ``User``, ``:``,
  `` What``, `` is``, `` a``, `` server``, ``?``, a new line, ``Assistant``, ``:``, with ids
  12982, 25, 1867, 318, 257, 4382, 30, 198, 48902, 25 (the colon is 25 both times). Tokens
  are what is counted: 10 read, at most 20 written (``max_tokens``), of GPT-2's limit of
  1,024. The question is the script's example, written for the animation; a service's own
  tokenizer would cut a little differently.
* ``hosting-one-token-at-a-time``: :func:`~voxanim.study_hosting.generation`, the reply
  written one token at a time (greedy: the likeliest each time), each step's top five next
  tokens with their probabilities: first `` A`` (11.3%) over `` It`` (5.1%), `` Server``,
  `` The``, `` I``. After 20 tokens (``max_tokens``) it stops mid-sentence: " A server is a
  place where you can communicate with other people. It's a place where you can". GPT-2 was
  not trained to chat; the reply is poor and repeats itself, and is shown as it is.
* ``hosting-read-once-remember``: without a cache each new token means running the whole
  text again (11, 12, ... 30 tokens: 410 tokens run for the 20 reply steps); with the cache
  (region "hs-cache") each layer keeps every token's keys and values (12 layers, each
  [12 heads, 30 tokens, 64] at the end) and only the new token runs (20 tokens). The scores
  agree with recomputing to 6.8e-6 (:func:`~voxanim.study_hosting.cache_check`), the same
  20 tokens, for 2.48 billion multiply-adds instead of 35.7 billion (14.4 times fewer,
  :func:`~voxanim.study_hosting.costs`: the scores over the 50,257 tokens cost the same
  either way, so the ratio is below the 20.5 of the tokens run).

Counts, not speeds: nothing here is a time, a price or a named service.
"""

from __future__ import annotations

import functools

import numpy as np

from . import gpt2
from . import study_hosting as H
from .blocks import code_lines, commas
from .study import Panel, Study

SOURCE = ("GPT-2 small (OpenAI, 2019; 124,439,808 numbers) run by examples/serve.py itself "
          "(voxanim.study_hosting), with GPT-2's own tokenizer; the question is the "
          "script's example, written for the animation")
TOP = H.TOP


def show(token):
    """A token as a tile shows it: a space as ``␣``, a new line as ``↵``."""
    return token.replace(" ", "␣").replace("\n", "↵")


def pct(p, places=1):
    """``11.3%``."""
    return "%.*f%%" % (places, 100 * float(p))


def big_count(n):
    """A count in words: ``2.5 billion``, ``124 million``, ``974 million``."""
    n = float(n)
    if n >= 1e9:
        return "%.1f billion" % (n / 1e9)
    if n >= 1e6:
        return "%d million" % round(n / 1e6)
    return commas(n)


def json_lines(text):
    """The request's JSON broken where a new key starts (``, "``), the pieces joined by a
    space giving the JSON back."""
    out = text.replace(', "', ',\n"').split("\n")
    lines, cur = [], ""
    for piece in out:
        if cur and len(cur) + 1 + len(piece) <= 33:
            cur += " " + piece
        else:
            if cur:
                lines.append(cur)
            cur = piece
    lines.append(cur)
    if " ".join(lines) != text:
        raise ValueError("the JSON lines do not join back into the request")
    return lines


def ids_lines(ids, width=32):
    """``[12982, 25, ...]`` broken into lines of at most ``width`` characters."""
    parts = [str(i) for i in ids]
    lines, cur = [], "["
    for n, s in enumerate(parts):
        piece = s + ("]" if n == len(parts) - 1 else ",")
        if len(cur) + len(piece) + (0 if cur == "[" else 1) > width:
            lines.append(cur)
            cur = " " + piece
        else:
            cur += ("" if cur == "[" else " ") + piece
    lines.append(cur)
    return lines


# ====================================================================== 255
def text_to_tokens():
    r = H.request()
    tokens = list(r["tokens"])
    ids = [int(i) for i in r["ids"]]
    _W, tok = H.model()
    if tok.decode(ids) != r["prompt"]:
        raise ValueError("the tokens do not spell the prompt")
    repeated = [i for i, t in enumerate(ids) if ids.count(t) > 1]
    chars = len(r["prompt"])
    meta = dict(
        source=SOURCE, evidence="measured on GPT-2: its tokenizer on the prompt",
        question=H.QUESTION, json=r["json"], json_lines=json_lines(r["json"]),
        bytes=r["bytes"], prompt=r["prompt"], prompt_lines=r["prompt"].split("\n"),
        chars=chars, tokens=tokens, shown=[show(t) for t in tokens], ids=ids,
        n_tokens=len(ids), ids_lines=ids_lines(ids), repeated=repeated,
        repeated_token=show(tokens[repeated[0]]) if repeated else "",
        repeated_id=ids[repeated[0]] if repeated else None,
        chars_per_token=chars / len(ids), vocab=gpt2.N_VOCAB, vocab_text=commas(gpt2.N_VOCAB),
        context=gpt2.N_CTX, context_text=commas(gpt2.N_CTX), max_tokens=H.MAX_TOKENS,
        code_handler=list(code_lines("hs-handler")), code_tokens=list(code_lines("hs-tokens")),
        limitations=[
            "the question is the script's example (study_hosting.QUESTION), written for the "
            "animation; its bytes, characters and tokens are counted for real",
            "GPT-2's tokenizer (byte-pair encoding, 50,257 tokens); a service's own tokenizer "
            "cuts the same text a little differently",
            "a service limits and bills by tokens, read and written; no price is shown"])
    panels = (
        Panel("Request, text, tokens", "the request's bytes, the prompt's characters and "
              "its tokens", "bars",
              dict(values=np.array([r["bytes"], chars, len(ids)], float),
                   labels=["bytes sent", "characters", "tokens"])),
        Panel("The tokens", "characters in each of the prompt's 10 tokens", "bars",
              dict(values=np.array([len(t) for t in tokens], float),
                   labels=[show(t) for t in tokens])),
        Panel("Token ids", "each token's number in GPT-2's list of 50,257", "bars",
              dict(values=np.array(ids, float), labels=[show(t) for t in tokens])),
        Panel("What is counted", "tokens read, written at most, and GPT-2's limit", "bars",
              dict(values=np.array([len(ids), H.MAX_TOKENS, gpt2.N_CTX], float),
                   labels=["read", "written at most", "GPT-2's limit"])),
    )
    return Study("Text becomes tokens", "The 82-byte request becomes one text of 34 "
                 "characters, then 10 tokens: what is read, written, counted and limited",
                 panels, meta)


# ====================================================================== 256
def one_token_at_a_time():
    g = H.generation()
    steps = []
    for s in g["steps"]:
        top = [dict(token=t, shown=show(t), p=float(p), p_text=pct(p)) for t, p in s["top"]]
        if top[0]["token"] != s["token"]:
            raise ValueError("the token written is not the likeliest")
        steps.append(dict(token=s["token"], shown=show(s["token"]), id=int(s["id"]),
                          p=float(s["p"]), p_text=pct(s["p"]), top=top))
    reply = g["reply"]
    if "".join(s["token"] for s in steps) != reply:
        raise ValueError("the tokens do not spell the reply")
    n_prompt = len(g["prompt_ids"])
    probs = np.array([[t["p"] for t in s["top"]] for s in steps])
    meta = dict(
        source=SOURCE, evidence="measured on GPT-2: the reply, 20 tokens, greedy",
        question=H.QUESTION, prompt=H.request()["prompt"], prompt_tokens=n_prompt,
        steps=steps, n=len(steps), reply=reply, reply_text=reply.strip(),
        tokens=[s["token"] for s in steps], vocab=gpt2.N_VOCAB,
        vocab_text=commas(gpt2.N_VOCAB), max_tokens=H.MAX_TOKENS,
        text_after=[n_prompt + k + 1 for k in range(len(steps))],
        stops_mid_sentence=reply.rstrip()[-1] not in ".!?",
        first_id=steps[0]["id"],
        code_generate=list(code_lines("hs-generate")),
        limitations=[
            "GPT-2 (2019) was not trained to chat: its reply is poor and repeats itself; "
            "what is shown is what a server does with any model",
            "greedy: the likeliest token every time, as the script's reply() does; a "
            "service usually samples, so the same question can get other replies",
            "the reply stops at max_tokens = 20, mid-sentence",
            "a service's models are far larger than GPT-2, and their sizes are mostly not "
            "published"])
    panels = (
        Panel("The token written", "its probability at each of the 20 steps", "curve",
              dict(x=np.arange(1, len(steps) + 1, dtype=float),
                   ys=np.array([[s["p"] for s in steps]]), labels=["probability"])),
        Panel("The top five", "the five likeliest next tokens' probabilities, step by step",
              "matrix", dict(values=probs)),
        Panel("The first step", "the five likeliest first tokens of the reply", "bars",
              dict(values=probs[0], labels=[t["shown"] for t in steps[0]["top"]])),
        Panel("The text so far", "tokens in the text after each step", "bars",
              dict(values=np.array(meta["text_after"], float),
                   labels=[s["shown"] for s in steps])),
    )
    return Study("One token at a time", "GPT-2's reply, written one token at a time from "
                 "its likeliest next tokens and sent as each exists", panels, meta)


# ====================================================================== 257
@functools.lru_cache(maxsize=1)
def recomputed():
    """Each reply step's scores computed again from scratch on the whole text so far (no
    cache), as :func:`~voxanim.study_hosting.cache_check` does: the largest difference at
    each step, and whether the same token wins."""
    W, _tok = H.model()
    step = H.program()["step"]
    g = H.generation()
    ids = list(g["prompt_ids"])
    diffs, same = [], []
    for k, nxt in enumerate(g["ids"]):
        fresh, _ = step(W, ids, [])
        diffs.append(float(np.abs(fresh - g["scores"][k]).max()))
        same.append(int(fresh.argmax()) == int(g["scores"][k].argmax()))
        ids.append(nxt)
    return dict(diffs=diffs, same=same)


@functools.lru_cache(maxsize=1)
def kept():
    """The cache at the end of the reply (prompt and the 20 reply tokens, as the script's
    last ``step`` leaves it), as the size of each token's keys and of its values per layer:
    ``(k, v)``, each ``[layers, tokens]`` (the norm over heads and their 64 numbers)."""
    W, _tok = H.model()
    g = H.generation()
    ids = list(g["prompt_ids"]) + list(g["ids"])
    _scores, cache = H.program()["step"](W, ids, [])
    k = np.array([np.linalg.norm(c[0], axis=(0, 2)) for c in cache])
    v = np.array([np.linalg.norm(c[1], axis=(0, 2)) for c in cache])
    return k, v, tuple(cache[0][0].shape)


def read_once_remember():
    c = H.costs()
    g = H.generation()
    rc = recomputed()
    k_norm, v_norm, shape = kept()
    if shape != g["cache_shape"]:
        raise ValueError("the cache recomputed is not the reply's")
    p, n = c["prompt"], c["tokens"]
    without = [p + k + 1 for k in range(n)]          # tokens run at each reply step
    with_ = [1] * n
    macs_with = [H.macs_per_token(p + k) for k in range(n)]
    macs_without = [H.macs_per_token(p + k, cache=False) for k in range(n)]
    if sum(macs_with) != c["with_cache"] or sum(macs_without) != c["without_cache"]:
        raise ValueError("the multiply-adds do not add up to costs()")
    worst = max(rc["diffs"])
    d = gpt2.D_MODEL
    meta = dict(
        source=SOURCE, evidence="measured on GPT-2: 20 reply tokens, cache against none",
        prompt=p, n=n, total=p + n, reply=g["reply"], layers=gpt2.N_LAYER, heads=gpt2.N_HEAD,
        head_dim=d // gpt2.N_HEAD, cache_shape=list(shape),
        cache_shape_text="[%d, %d, %d]" % shape,
        per_token_layer=2 * d, per_token_layer_text=commas(2 * d),
        per_token=2 * d * gpt2.N_LAYER, per_token_text=commas(2 * d * gpt2.N_LAYER),
        tokens_without=without, tokens_with=with_,
        run_without=sum(without), run_with=sum(with_),
        run_ratio=sum(without) / sum(with_),
        macs_with=macs_with, macs_without=macs_without,
        total_with=c["with_cache"], total_without=c["without_cache"],
        total_with_text=big_count(c["with_cache"]),
        total_without_text=big_count(c["without_cache"]),
        read_prompt=c["read_prompt"], read_prompt_text=big_count(c["read_prompt"]),
        ratio=c["ratio"], ratio_text="%.1f" % c["ratio"], ratio_round=int(round(c["ratio"])),
        first_with_text=big_count(macs_with[0]), first_without_text=big_count(macs_without[0]),
        last_with_text=big_count(macs_with[-1]), last_without_text=big_count(macs_without[-1]),
        diffs=rc["diffs"], max_difference=worst, max_difference_text="%.7f" % worst,
        max_difference_sci="%.1e" % worst, same_tokens=all(rc["same"]),
        same_count=int(sum(rc["same"])), vocab=gpt2.N_VOCAB, vocab_text=commas(gpt2.N_VOCAB),
        k_norm=(k_norm / k_norm.max(axis=1, keepdims=True)).tolist(),
        v_norm=(v_norm / v_norm.max(axis=1, keepdims=True)).tolist(),
        code_cache=list(code_lines("hs-cache")), code_generate=list(code_lines("hs-generate")),
        limitations=[
            "multiply-adds are counted from GPT-2's shapes (H.macs_per_token), not timed; "
            "reading the 10-token prompt once is left out of both",
            "the scores over all 50,257 tokens cost the same with or without the cache, so "
            "the multiply-adds fall %.1f times while the tokens run fall %.1f times"
            % (c["ratio"], sum(without) / sum(with_)),
            "the differences in the scores are rounding in float32 arithmetic",
            "the shelves are shaded by the real size of each token's keys and values in "
            "each layer (relative to that layer's largest)"])
    panels = (
        Panel("Tokens run per step", "without a cache (the whole text again) and with it",
              "curve", dict(x=np.arange(1, n + 1, dtype=float),
                            ys=np.array([without, with_], float),
                            labels=["without a cache", "with the cache"])),
        Panel("Multiply-adds per step", "for each reply token, with and without the cache",
              "curve", dict(x=np.arange(1, n + 1, dtype=float),
                            ys=np.array([macs_without, macs_with], float),
                            labels=["without a cache", "with the cache"])),
        Panel("The cache, kept", "the size of each token's keys in each of the 12 layers",
              "matrix", dict(values=np.asarray(meta["k_norm"]))),
        Panel("The same scores", "the largest difference between the cached and the "
              "recomputed scores, step by step", "bars",
              dict(values=np.array(rc["diffs"]), labels=[str(k + 1) for k in range(n)])),
    )
    return Study("Read once, then remember", "The key-value cache gives the same scores "
                 "(to 6.8e-6) for 14 times fewer multiply-adds over 20 tokens", panels, meta)


STUDIES = {"hosting-text-to-tokens": text_to_tokens,
           "hosting-one-token-at-a-time": one_token_at_a_time,
           "hosting-read-once-remember": read_once_remember}
