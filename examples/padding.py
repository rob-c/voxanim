"""Padding: the program of the padding animations of "Blocks and code".

    python examples/padding.py

Two halves. First a CNN's padding: why maps shrink, a ring of padding, "same" and "valid",
what goes in the border, and the course's CNN with padding=1. Then padding as the way to
make a batch out of things that differ in size: digits cut to their ink, and lines of
text. Needs torch and torchvision; downloads MNIST (11 MB) and Tiny Shakespeare (1 MB)
into ./data. As in mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on
screen (voxanim.blocks.code_lines("NAME")), and lines of code are kept to 41 characters.
"""
# region: pad-imports
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# endregion

torch.manual_seed(0)
prep = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_set = MNIST("data", download=True, transform=prep)
x = torch.stack([train_set[i][0] for i in range(64)])      # the first 64 digits

# ---------------------------------------------------------------- a CNN's padding
# region: pad-none
conv = nn.Conv2d(1, 8, 3)                 # no padding
maps = conv(x)                            # [64, 8, 26, 26]
# endregion
print("no padding:", maps.shape)

# region: pad-one
conv = nn.Conv2d(1, 8, 3, padding=1)      # a ring of padding
maps = conv(x)                            # [64, 8, 28, 28]
# endregion
print("padding=1:", maps.shape)

# region: pad-same
for k in (3, 5, 7):                       # 3 x 3, 5 x 5, 7 x 7 stamps
    same = nn.Conv2d(1, 8, k,
                     padding="same")      # out as big as in
    valid = nn.Conv2d(1, 8, k)            # "valid", the default
    print(k, same(x).shape)               # 28 x 28 for every k
    print(k, valid(x).shape)              # 26, 24, 22: k - 1 lost
# endregion

# region: pad-modes
part = x[:1, :, 5:13, 8:16]               # 8 x 8 of a digit's stroke
for mode in ("constant", "reflect",
             "replicate", "circular"):
    big = F.pad(part, [2] * 4, mode=mode) # 2 on every side
    print(mode, big.shape)                # [1, 1, 12, 12]
# endregion

# region: pad-deep
h = x
for _ in range(12):                       # 12 layers of 3 x 3 stamps
    h = nn.Conv2d(1, 1, 3)(h)             # each loses 2 from each side
print(h.shape)                            # [64, 1, 4, 4]
# endregion

# region: pad-model
model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),        # [64, 8, 28, 28]
    nn.ReLU(), nn.MaxPool2d(2),           # [64, 8, 14, 14]
    nn.Conv2d(8, 16, 3, padding=1),       # [64, 16, 14, 14]
    nn.ReLU(), nn.MaxPool2d(2),           # [64, 16, 7, 7]
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10),            # 784 numbers in, 10 scores out
)
# endregion
print("the CNN with padding=1:", model(x).shape)

# ---------------------------------------------------------------- a batch is one block
def cut(img):
    """The digit cut to its ink: the smallest box holding every pixel brighter than the
    blank background (the image's lowest value)."""
    ink = (img[0] > img.min() + 1e-3).nonzero()
    (r0, c0), (r1, c1) = ink.min(0).values, ink.max(0).values
    return img[:, r0:r1 + 1, c0:c1 + 1]


try:
    # region: pad-cut
    small = [cut(x[i]) for i in range(4)] # 4 digits cut to their ink
    print([list(s.shape) for s in small]) # 4 different sizes
    batch = torch.stack(small)            # fails: a batch is one block
    # endregion
except RuntimeError as error:
    print("stack:", error)

# region: pad-to-block
BG = (0 - 0.1307) / 0.3081                # a blank pixel after Normalize
def pad_to(img, side=28):
    h, w = img.shape[-2:]
    room = (0, side - w, 0, side - h)     # left, right, top, bottom
    return F.pad(img, room, value=BG)
padded = [pad_to(s) for s in small]
batch = torch.stack(padded)               # [4, 1, 28, 28]
# endregion
print("padded:", batch.shape)

# ---------------------------------------------------------------- lines of text
TEXT = "data/tinyshakespeare.txt"
if not os.path.exists(TEXT):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/"
        "input.txt", TEXT)

# region: pad-text
text = Path(TEXT).read_text()             # Tiny Shakespeare
vocab = ["<pad>"] + sorted(set(text))     # id 0 is kept for padding
ids = {c: i for i, c in enumerate(vocab)}
lines = text.split("\n")
lines = [s for s in lines if s]           # every line that is not empty
# endregion

# region: pad-sequences
seqs = [torch.tensor([ids[c] for c in s])
        for s in lines[:8]]               # 8 lines, 8 lengths
batch = pad_sequence(seqs,
                     batch_first=True)    # [8, longest], pads = 0
mask = batch != 0                         # True: a real character
# endregion
print("lines:", batch.shape, int(mask.sum()), "real of", batch.numel())

# region: pad-masked-mean
embed = nn.Embedding(len(vocab), 16,
                     padding_idx=0)       # the pad's vector is 0
vecs = embed(batch)                       # [8, longest, 16]
plain = vecs.mean(1)                      # the pads counted in
keep = mask.unsqueeze(-1).float()
total = (vecs * keep).sum(1)
mean = total / keep.sum(1)                # the pads left out
# endregion
print("plain / masked:", (plain / mean)[:, 0])

# region: pad-attention
attn = nn.MultiheadAttention(
    16, 2, batch_first=True)
out, w = attn(vecs, vecs, vecs,
              key_padding_mask=~mask)     # True: ignore this key
print(w[1, 0])                            # 0 on every pad
# endregion

# region: pad-loss
loss_fn = nn.CrossEntropyLoss(
    ignore_index=0)                       # a pad target counts for 0
# endregion

# region: pad-sort
lines.sort(key=len)                       # short lines together
for i in range(0, len(lines), 64):
    group = lines[i:i + 64]               # one batch of 64 lines
    longest = max(len(s) for s in group)  # pad to this, not more
# endregion
