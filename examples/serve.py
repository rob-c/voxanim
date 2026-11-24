"""Hosting a language model, as small as it can be: the program of the hosting animations of
"Blocks and code".

    python examples/serve.py                  # starts the server on this computer
    python examples/serve.py ask "Hello"      # a client: sends a question, prints the reply

The model is GPT-2 small (OpenAI, 2019; 124,439,808 numbers), run in plain numpy by
voxanim.gpt2, which downloads its weights (550 MB) once. GPT-2 was not trained to chat, so its
replies are poor; what matters here is what a server does with a request: read the text,
turn it into tokens, run the model one token at a time, remember what it has read (the
key-value cache), send each token back as soon as it exists, and serve several people with
one copy of the model. Everything runs on this computer; on a real service the server is a
computer somewhere else, reached over the internet, and the model is far bigger.

As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen
(voxanim.blocks.code_lines("NAME")), and lines of code are kept to 41 characters.
"""
# region: hs-imports
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import Request, urlopen

import numpy as np

from voxanim import gpt2

# endregion

URL = "http://localhost:8000/chat"
HEADS, LAYERS = gpt2.N_HEAD, gpt2.N_LAYER


def ask(question):
    # region: hs-request
    said = {"role": "user",               # who is speaking
            "content": question}          # and what they said
    request = {"messages": [said],        # the whole talk so far
               "max_tokens": 20}          # how long a reply may be
    body = json.dumps(request).encode()   # the request: just text
    reply = urlopen(Request(URL, body))   # off to the server
    for line in reply:                    # tokens arrive one by one
        sys.stdout.write(line.decode())
    # endregion


# ---------------------------------------------------------------- the server
def load():
    # region: hs-load
    W = gpt2.weights()                    # 124,439,808 numbers
    tok = gpt2.Tokenizer()                # text <-> token ids
    # endregion
    return W, tok


def attend(q, k, v):
    """Every new token looks back over every token so far (causal)."""
    s = q @ k.transpose(0, 2, 1) / 8.0                   # [heads, new, all]
    new, every = s.shape[1:]
    s = np.where(np.triu(np.ones((new, every), bool), every - new + 1), -1e10, s)
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


def heads(t):
    return t.reshape(len(t), HEADS, -1).transpose(1, 0, 2)


def step(W, ids, cache):
    """GPT-2 on the tokens ``ids`` that are new, the tokens before them already in ``cache``
    (a list of (keys, values) per layer, empty at first). Returns the next token's scores and
    the cache with the new tokens' keys and values added."""
    start = cache[0][0].shape[1] if cache else 0
    x = W["wte.weight"][ids] + W["wpe.weight"][start:start + len(ids)]
    new_cache = []
    for i in range(LAYERS):
        p = "h.%d." % i
        h = gpt2.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        q, k, v = np.split(h @ W[p + "attn.c_attn.weight"] + W[p + "attn.c_attn.bias"], 3, 1)
        q, k, v = heads(q), heads(k), heads(v)
        # region: hs-cache
        if cache:                         # what it read before
            old_k, old_v = cache[i]
            k = np.concatenate([old_k, k], 1)  # keep, add the new
            v = np.concatenate([old_v, v], 1)
        new_cache.append((k, v))          # remember for next time
        o = attend(q, k, v)               # new tokens look at all
        # endregion
        o = o.transpose(1, 0, 2).reshape(len(ids), -1)
        x = x + o @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = gpt2.layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        m = gpt2._gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + m @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = gpt2.layer_norm(x[-1], W["ln_f.weight"], W["ln_f.bias"])
    return x @ W["wte.weight"].T, new_cache


def reply(W, tok, text, max_tokens, send):
    """Turn ``text`` into tokens, read them once, then write ``max_tokens`` more, sending
    each as soon as it exists."""
    # region: hs-tokens
    ids = tok.encode(text)                # text -> token ids
    # endregion
    # region: hs-generate
    scores, cache = step(W, ids, [])      # read the whole prompt once
    for _ in range(max_tokens):
        nxt = int(scores.argmax())        # the likeliest next token
        send(tok.decode([nxt]))           # off to the user at once
        scores, cache = step(W, [nxt], cache)  # just the new one
    # endregion


def prompt(messages):
    """The conversation as one text: GPT-2 only continues text."""
    lines = ["%s: %s" % (m["role"].capitalize(), m["content"]) for m in messages]
    return "\n".join(lines) + "\nAssistant:"


def serve():
    W, tok = load()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(200)
            self.end_headers()
            # region: hs-handler
            n = int(self.headers["Content-Length"])
            request = json.loads(self.rfile.read(n))  # just text
            text = prompt(request["messages"])  # the whole talk so far
            most = request["max_tokens"]
            out = self.wfile              # the line back to the user
            send = lambda t: out.write(t.encode())  # noqa: E731
            reply(W, tok, text, most, send)  # tokens, one by one
            # endregion

    print("serving GPT-2 on", URL)
    HTTPServer(("localhost", 8000), Handler).serve_forever()


def batch_step(W, rows):
    """One step for several people at once. ``rows`` is a list of (newest token, cache), one
    per person. Their newest tokens go through every layer together, as one matrix with a row
    per person, so each weight is read once for all of them; only attention, which looks at
    each person's own past, is done row by row. Returns each person's (scores, cache)."""
    newest = [t for t, _ in rows]
    caches = [c for _, c in rows]
    where = [c[0][0].shape[1] if c else 0 for c in caches]
    x = W["wte.weight"][newest] + W["wpe.weight"][where]          # [people, 768]
    kept = [[] for _ in rows]
    for i in range(LAYERS):
        p = "h.%d." % i
        h = gpt2.layer_norm(x, W[p + "ln_1.weight"], W[p + "ln_1.bias"])
        # region: hs-batch
        w = W[p + "attn.c_attn.weight"]   # [768, 2304], read once
        qkv = h @ w                       # a row per person, together
        # endregion
        qkv = qkv + W[p + "attn.c_attn.bias"]
        o = []
        for r, row in enumerate(qkv):                             # each person's own past
            q, k, v = (heads(t[None]) for t in np.split(row, 3))
            if caches[r]:
                k = np.concatenate([caches[r][i][0], k], 1)
                v = np.concatenate([caches[r][i][1], v], 1)
            kept[r].append((k, v))
            o.append(attend(q, k, v).transpose(1, 0, 2).reshape(-1))
        x = x + np.stack(o) @ W[p + "attn.c_proj.weight"] + W[p + "attn.c_proj.bias"]
        h = gpt2.layer_norm(x, W[p + "ln_2.weight"], W[p + "ln_2.bias"])
        m = gpt2._gelu(h @ W[p + "mlp.c_fc.weight"] + W[p + "mlp.c_fc.bias"])
        x = x + m @ W[p + "mlp.c_proj.weight"] + W[p + "mlp.c_proj.bias"]
    x = gpt2.layer_norm(x, W["ln_f.weight"], W["ln_f.bias"])
    return [(x[r] @ W["wte.weight"].T, kept[r]) for r in range(len(rows))]


if __name__ == "__main__":
    if sys.argv[1:2] == ["ask"]:
        ask(" ".join(sys.argv[2:]))
    else:
        serve()
