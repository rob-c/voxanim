"""A vector database from scratch: the program of the vector-database animations of "Blocks
and code".

    python examples/vector_db.py

Train an encoder so that two drawings of the same digit get nearby vectors (InfoNCE with the
rest of the batch as negatives), store a vector for each of the 60,000 training digits with
its id and label beside it, search it exactly, then through an inverted-file index (IVF:
k-means lists, a few of them visited), and last, query the store with a retrained encoder.
Needs torch and torchvision; downloads MNIST (11 MB) into ./data. As in mnist_blocks.py,
"# region: NAME" ... "# endregion" marks what goes on screen, lines kept to 41 characters.
"""
# region: vdb-imports
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# endregion

prep = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_set = MNIST("data", download=True, transform=prep)
test_set = MNIST("data", train=False, transform=prep)
X = torch.stack([img for img, _ in train_set])            # [60000, 1, 28, 28]
Y = torch.tensor([y for _, y in train_set])
XT = torch.stack([img for img, _ in test_set])            # [10000, 1, 28, 28]
YT = torch.tensor([y for _, y in test_set])
SAME = [torch.nonzero(Y.eq(d))[:, 0] for d in range(10)]    # where each digit is


def partner(y):
    """For each label, another training digit with the same label, drawn at random."""
    picks = [SAME[d][torch.randint(len(SAME[d]), ())] for d in y.tolist()]
    return X[torch.stack(picks)]


def make_encoder():
    # region: vdb-encoder
    encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 32),               # 32 numbers per digit
    )
    # endregion
    return encoder


def train(seed):
    torch.manual_seed(seed)
    encoder = make_encoder()
    optimizer = torch.optim.Adam(encoder.parameters())
    for _ in range(938):                  # one pass, in batches of 64
        i = torch.randint(len(X), (64,))
        x, y = X[i], Y[i]
        # region: vdb-loss
        a = F.normalize(encoder(x), dim=1)  # 64 digits, length 1
        p = partner(y)                    # 64 others, the same digits
        b = F.normalize(encoder(p), dim=1)
        scores = a @ b.T / 0.1            # [64, 64]: every pair
        target = torch.arange(len(a))     # the right pair: the diagonal
        loss = F.cross_entropy(scores, target)
        # endregion
        # region: vdb-step
        optimizer.zero_grad()
        loss.backward()                   # pull pairs in, push others out
        optimizer.step()
        # endregion
    return encoder


encoder = train(seed=0)

# region: vdb-store
with torch.no_grad():
    v = encoder(X)                        # [60000, 32]
store = F.normalize(v, dim=1)             # each of length 1
ids = torch.arange(len(store))            # each vector's id
labels = Y                                # kept beside it: metadata
# endregion

# region: vdb-search
with torch.no_grad():
    q = encoder(XT[:1])                   # a test digit: [1, 32]
q = F.normalize(q, dim=1)
scores = store @ q[0]                     # 60,000 dot products
top = scores.topk(5)                      # the 5 nearest
found = ids[top.indices]                  # their ids
print(found, labels[found])               # and their labels
# endregion


def kmeans(v, k, steps=10):
    """Spherical k-means: k centres of length 1, each the mean direction of its vectors."""
    centres = v[torch.randperm(len(v))[:k]]
    for _ in range(steps):
        home = (v @ centres.T).argmax(1)
        for j in range(k):
            if (home == j).any():
                centres[j] = F.normalize(v[home == j].mean(0), dim=0)
    return centres


# region: vdb-ivf-build
centres = kmeans(store, 64)               # 64 lists
home = (store @ centres.T).argmax(1)      # each vector's list
# endregion

# region: vdb-ivf-search
near = (centres @ q[0]).topk(4).indices   # visit 4 lists of 64
look = torch.isin(home, near)             # the vectors in them
cand = look.nonzero()[:, 0]
scores = store[cand] @ q[0]               # far fewer dot products
top = cand[scores.topk(5).indices]        # the 5 nearest found
# endregion
print("IVF:", top, "searched", len(cand), "of", len(store))

# region: vdb-versions
encoder2 = train(seed=1)                  # the same recipe, again
with torch.no_grad():
    q2 = encoder2(XT[:1])                 # the same test digit
q2 = F.normalize(q2, dim=1)
scores = store @ q2[0]                    # old store, new query
print(labels[scores.topk(5).indices])     # wrong: mixed encoders
# endregion

# region: vdb-reembed
with torch.no_grad():
    v = encoder2(X)                       # embed everything again
store = F.normalize(v, dim=1)
scores = store @ q2[0]
print(labels[scores.topk(5).indices])     # right again
# endregion
