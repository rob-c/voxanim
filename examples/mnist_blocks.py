"""Train a small MNIST digit classifier: the program the "Blocks and code" animations show.

    python examples/mnist_blocks.py          # SGD, one pass over the data (about a minute)
    python examples/mnist_blocks.py adam     # the same with Adam

Needs torch and torchvision, and downloads MNIST (11 MB) into ./data. Each
"# region: NAME" ... "# endregion" pair marks a piece the animations put on screen
(voxanim.blocks.code_lines("NAME")), so their code is this program's. On screen the
comments are left out, so every line of code is kept to 41 characters.
"""
# region: imports
import sys

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# endregion

# region: device
torch.manual_seed(0)                      # the same run every time
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"     # where the numbers live
# endregion

# region: data
prep = Compose([
    ToTensor(),                           # pixels 0..255 become 0..1
    Normalize((0.1307,), (0.3081,)),      # minus the mean, over the spread
])
train_set = MNIST("data", download=True,  # the 60,000 training digits
                  transform=prep)
test_set = MNIST("data", train=False,     # 10,000 more, kept for testing
                 transform=prep)
# endregion

# region: one-digit
x, y = train_set[0]                       # one digit and its label
print(x.shape, y)                         # torch.Size([1, 28, 28]) 5
# endregion

# region: loader
loader = DataLoader(train_set,
                    batch_size=64,
                    shuffle=True)         # a new order every epoch
test_loader = DataLoader(test_set, 1000)
# endregion

# region: one-batch
x, y = next(iter(loader))                 # one batch: 64 digits
print(x.shape, y.shape)                   # [64, 1, 28, 28] and [64]
# endregion

# region: model
model = nn.Sequential(
    nn.Flatten(),                         # [64, 1, 28, 28] -> [64, 784]
    nn.Linear(784, 64),                   # 784 x 64 weights, 64 biases
    nn.ReLU(),                            # negative numbers become 0
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),                    # 10 scores per digit: the logits
).to(device)
# endregion

# region: print-model
print(model)                              # the blocks, in order, with their sizes
# endregion

# region: loss
loss_fn = nn.CrossEntropyLoss()           # takes the logits, not a softmax
# endregion

# region: optimizer
optimizer = SGD(model.parameters(),
                lr=0.1)                   # the learning rate
if "adam" in sys.argv:
    optimizer = Adam(model.parameters())  # lr=0.001, its default
# endregion

# region: train-mode
model.train()                             # training mode (matters for dropout)
# endregion

# region: train
for epoch in range(1):                    # one pass over the 60,000 digits
    for x, y in loader:                   # 938 batches of 64 (the last: 32)
        # region: step
        x, y = x.to(device), y.to(device)
        logits = model(x)                 # forward: [64, 10]
        loss = loss_fn(logits, y)         # one number: how wrong
        optimizer.zero_grad()             # empty last step's gradients
        loss.backward()                   # a gradient for every weight
        optimizer.step()                  # move every weight a little
        # endregion
    print(epoch, loss.item())             # the last batch's loss
# endregion

# region: grad
print(model[1].weight.grad.shape)         # torch.Size([64, 784]): the weight's shape
# endregion

# region: evaluate
model.eval()                              # test mode (matters for dropout)
correct = 0
with torch.no_grad():                     # no gradients needed to test
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)         # the highest score wins
        correct += (pred == y).sum()
print(int(correct) / len(test_set))       # the accuracy: 0.95 is 95 in 100
# endregion

# region: save
state = model.state_dict()                # every weight and bias, by name
torch.save(state, "mnist.pt")
# endregion

# region: load
state = torch.load("mnist.pt")            # the same numbers, back from disk
model.load_state_dict(state)
# endregion
