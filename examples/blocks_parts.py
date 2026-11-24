"""The other blocks of PyTorch, and four broken programs: the pieces the "Blocks and code"
animations show that the digit classifier of mnist_blocks.py does not use.

    python examples/blocks_parts.py          # runs every piece on a batch, prints its shape

Needs torch only (the batch is made up here, the shape of an MNIST batch). As in
mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen
(voxanim.blocks.code_lines("NAME")), and lines of code are kept to 41 characters.
"""
# region: parts-imports
import torch
from torch import nn

# endregion

torch.manual_seed(0)
x = torch.randn(64, 1, 28, 28)            # a batch, shaped like 64 MNIST digits
y = torch.randint(0, 10, (64,))           # and their 64 labels
h = torch.randn(64, 64)                   # 64 hidden numbers for each digit

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(),
                      nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))

# region: count
print(model)                              # the blocks, in order
params = list(model.parameters())         # 3 weight grids, 3 bias strips
print(sum(p.numel() for p in params))     # 52650
# endregion

# region: conv
conv = nn.Conv2d(1, 4, 3)                 # 4 filters of 3 x 3
maps = conv(x)                            # [64, 4, 26, 26]
# endregion
print("conv", maps.shape)

# region: pool
pool = nn.MaxPool2d(2)                    # each 2 x 2 square: its biggest
small = pool(maps)                        # [64, 4, 13, 13]
# endregion
print("pool", small.shape)

# region: dropout
drop = nn.Dropout(0.2)                    # switch off 1 in 5, in training
drop.train()
seen = drop(h)                            # some numbers are now 0
drop.eval()
same = drop(h)                            # test: nothing switched off
# endregion
print("dropout", (seen == 0).float().mean().item(), torch.equal(same, h))

# region: batchnorm
bn = nn.BatchNorm1d(64)                   # averages down the batch
out = bn(h)                               # [64, 64]
# endregion
print("batchnorm", out.shape)

# region: layernorm
ln = nn.LayerNorm(64)                     # averages across the features
out = ln(h)                               # [64, 64]
# endregion
print("layernorm", out.shape)

# region: embedding
embed = nn.Embedding(10, 16)              # a table: 10 rows of 16
vectors = embed(y)                        # [64, 16]: one row per label
# endregion
print("embedding", vectors.shape)

# region: attention
q = k = v = torch.randn(64, 5, 16)        # 64 sequences of 5 tokens
attn = nn.MultiheadAttention(
    16, 2, batch_first=True)              # width 16, 2 heads
out, scores = attn(q, k, v)               # [64, 5, 16] and [64, 5, 5]
# endregion
print("attention", out.shape, scores.shape)

# region: softmax
logits = torch.randn(64, 10)              # 10 scores per digit
probs = logits.softmax(dim=1)             # each row now adds up to 1
# endregion
print("softmax", probs.sum(1)[:3])

# region: dropout-model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 10),
)
# endregion
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# region: mistake-no-zero-grad
logits = model(x)
loss = loss_fn(logits, y)
loss.backward()                           # adds to the old gradients
optimizer.step()
# endregion

# region: mistake-no-eval
correct = 0
with torch.no_grad():                     # but still in training mode:
    pred = model(x).argmax(1)             # dropout is still switching off
    correct += (pred == y).sum().item()
# endregion

# region: mistake-double-softmax
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10),
    nn.Softmax(dim=1),                    # wrong: the loss does this already
)
# endregion
print("double softmax", loss_fn(model(x), y).item())

# region: mistake-no-flatten
model = nn.Sequential(
    nn.Linear(784, 64), nn.ReLU(),        # no nn.Flatten() in front
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10),
)
# endregion
try:
    model(x)
except RuntimeError as error:
    print("no flatten:", error)

# The CNN of mnist_cnn.py, and three ways to break it.
cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(400, 10),
)

# region: cnn-mistake-in-features
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(784, 10),                   # wrong: 16 x 5 x 5 = 400 arrive
)
# endregion
try:
    model(x)
except RuntimeError as error:
    print("in_features:", error)

try:
    # region: cnn-mistake-channels
    conv = nn.Conv2d(3, 8, 3)             # wrong: a grey digit has 1 channel
    maps = conv(x)
    # endregion
except RuntimeError as error:
    print("channels:", error)

try:
    # region: cnn-mistake-no-batch
    one = x[0]                            # [1, 28, 28]: no batch any more
    scores = cnn(one)
    # endregion
except RuntimeError as error:
    print("no batch:", error)

# region: cnn-fix-no-batch
one = x[0:1]                              # [1, 1, 28, 28]: a batch of one
scores = cnn(one)                         # [1, 10]
# endregion
print("fixed:", scores.shape)
