"""Thinking and effort, in a model small enough to see: the program of the "thinking" animations
of "Blocks and code".

    python examples/thinking.py

A tiny transformer (2 layers, width 64) is asked for the parity of 12 bits: 1 if the number of
ones is odd. It may write some steps first, the parity so far after each group of bits, before
it gives the answer. A first token, the effort, says how many steps it may write: 0 (answer
at once), 1, 2, 3, 4, 6 or 12 (one step per bit). One model is trained on every effort; then
tested on bit strings it never saw, at every effort. Steps are tokens like any other: the
model writes them, reads them back, and each one costs a pass through the model.

Needs torch. As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on
screen (voxanim.blocks.code_lines("NAME")), lines of code kept to 41 characters.
"""
# region: th-imports
import torch
import torch.nn.functional as F
from torch import nn

# endregion

torch.manual_seed(0)
BITS = 12
EFFORTS = (0, 1, 2, 3, 4, 6, 12)           # steps the model may write before answering
EQUALS = 2                                 # tokens: 0, 1, "=", then one per effort
VOCAB = 3 + len(EFFORTS)
LONGEST = 1 + BITS + 1 + max(EFFORTS) + 1


# region: th-example
def example(bits, e):                     # e: which effort
    steps = EFFORTS[e]                    # steps it may write
    seq = [3 + e] + bits + [EQUALS]       # effort, the bits, "="
    g = BITS // max(steps, 1)             # g: bits per step
    for k in range(1, steps + 1):         # the steps: parity so far
        seq.append(sum(bits[:k * g]) % 2)
    seq.append(sum(bits) % 2)             # and the answer
    return seq
# endregion


class Block(nn.Module):
    def __init__(self, d=64, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.heads = heads
        # region: th-block
        self.qkv = nn.Linear(d, 3 * d)    # each token asks, offers, tells
        self.out = nn.Linear(d, d)        # what it heard, mixed back in
        self.mlp = nn.Sequential(         # then each token on its own
            nn.Linear(d, 4 * d), nn.ReLU(),
            nn.Linear(4 * d, d))
        # endregion

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).split(d, 2)
        q, k, v = (t.reshape(B, T, self.heads, -1).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # looks back only
        x = x + self.out(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.n2(x))


class Tiny(nn.Module):
    # region: th-model
    def __init__(self, d=64, layers=2):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)  # a row per token
        self.pos = nn.Embedding(LONGEST, d)  # a row per place
        self.blocks = nn.ModuleList(
            Block(d) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)   # scores for the next token
    # endregion

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


model = Tiny()
every = torch.tensor([[(n >> i) & 1 for i in range(BITS)] for n in range(2 ** BITS)])
held = torch.randperm(2 ** BITS)[:468]                     # never seen in training
seen = torch.ones(2 ** BITS, dtype=bool)
seen[held] = False


def batch(n=256):
    """n training examples, each a seen bit string at a random effort, padded with 0."""
    rows = torch.nonzero(seen)[:, 0][torch.randint(int(seen.sum()), (n,))]
    seqs = [example(every[r].tolist(), int(torch.randint(len(EFFORTS), ()))) for r in rows]
    x = torch.zeros(n, LONGEST, dtype=torch.long)
    learn = torch.zeros(n, LONGEST)                         # which places are scored
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
        learn[i, 1 + BITS:len(s) - 1] = 1                   # from "=" on
    return x, learn


def train(steps=6000):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        x, learn = batch()
        # region: th-train
        scores = model(x[:, :-1]).mT      # guesses: [256, 10, 26]
        miss = F.cross_entropy(scores, x[:, 1:],
                               reduction="none")
        w = learn[:, :-1]                 # count from "=" on only
        loss = (miss * w).sum() / w.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # endregion
        if step % 1000 == 0:
            print(step, loss.item())


@torch.no_grad()
def answer(bits, e):
    """The model's own run: it writes its steps and its answer one token at a time."""
    # region: th-generate
    seq = [3 + e] + bits + [EQUALS]       # the question
    for _ in range(EFFORTS[e] + 1):       # its steps, then the answer
        scores = model(torch.tensor([seq]))
        nxt = int(scores[0, -1, :2].argmax())  # 0 or 1
        seq.append(nxt)                   # it reads this back next time
    # endregion
    return seq


if __name__ == "__main__":
    train()
    for e, steps in enumerate(EFFORTS):
        right = sum(answer(every[r].tolist(), e)[-1] == int(every[r].sum()) % 2 for r in held)
        print("effort", steps, "steps:", right, "of", len(held), "right,",
              steps + 1, "tokens written each")
