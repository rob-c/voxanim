"""Train a small MNIST digit classifier with convolutions: the program of the CNN animations
of "Blocks and code".

    python examples/mnist_cnn.py             # SGD, one pass over the data (a few minutes)
    python examples/mnist_cnn.py adam        # the same with Adam

It is examples/mnist_blocks.py with one block changed: the model. The data, the loss, the
optimizer and the training loop are the same lines. Needs torch and torchvision, and
downloads MNIST (11 MB) into ./data. As there, "# region: NAME" ... "# endregion" marks
what goes on screen (voxanim.blocks.code_lines("NAME")), and lines of code are kept to
41 characters once their comments are left out.
"""
# region: cnn-imports
import sys

import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# endregion

# region: cnn-device
torch.manual_seed(0)                      # the same run every time
use_gpu = torch.cuda.is_available()
device = "cuda" if use_gpu else "cpu"     # where the numbers live
# endregion

# region: cnn-data
prep = Compose([
    ToTensor(),                           # pixels 0..255 become 0..1
    Normalize((0.1307,), (0.3081,)),      # minus the mean, over the spread
])
train_set = MNIST("data", download=True,  # the 60,000 training digits
                  transform=prep)
test_set = MNIST("data", train=False,     # 10,000 more, kept for testing
                 transform=prep)
# endregion

# region: cnn-one-digit
x, y = train_set[0]                       # one digit and its label
print(x.shape, y)                         # torch.Size([1, 28, 28]) 5
# endregion

# region: cnn-loader
loader = DataLoader(train_set,
                    batch_size=64,
                    shuffle=True)         # a new order every epoch
test_loader = DataLoader(test_set, 1000)
# endregion

# region: cnn-one-batch
x, y = next(iter(loader))                 # one batch: 64 digits
print(x.shape, y.shape)                   # [64, 1, 28, 28] and [64]
# endregion

# region: cnn-model
model = nn.Sequential(
    nn.Conv2d(1, 8, 3), nn.ReLU(),        # 8 stamps of 3 x 3: [64, 8, 26, 26]
    nn.MaxPool2d(2),                      # [64, 8, 13, 13]
    nn.Conv2d(8, 16, 3), nn.ReLU(),       # 16 stamps of 8 x 3 x 3: [64, 16, 11, 11]
    nn.MaxPool2d(2),                      # [64, 16, 5, 5]
    nn.Flatten(),                         # [64, 400]
    nn.Linear(16 * 5 * 5, 10),            # 400 numbers in, 10 scores out
).to(device)
# endregion

# region: cnn-print-model
print(model)                              # the blocks, in order, with their sizes
# endregion

# region: cnn-loss
loss_fn = nn.CrossEntropyLoss()           # takes the logits, not a softmax
# endregion

# region: cnn-optimizer
optimizer = SGD(model.parameters(),
                lr=0.1)                   # the learning rate
if "adam" in sys.argv:
    optimizer = Adam(model.parameters())  # lr=0.001, its default
# endregion

# region: cnn-train-mode
model.train()                             # training mode (matters for dropout)
# endregion

# region: cnn-train
for epoch in range(1):                    # one pass over the 60,000 digits
    for x, y in loader:                   # 938 batches of 64 (the last: 32)
        # region: cnn-step
        x, y = x.to(device), y.to(device)
        logits = model(x)                 # forward: [64, 10]
        loss = loss_fn(logits, y)         # one number: how wrong
        optimizer.zero_grad()             # empty last step's gradients
        loss.backward()                   # a gradient for every weight
        optimizer.step()                  # move every weight a little
        # endregion
    print(epoch, loss.item())             # the last batch's loss
# endregion

# region: cnn-grad
print(model[0].weight.grad.shape)         # torch.Size([8, 1, 3, 3]): the stamps' shape
# endregion

# region: cnn-evaluate
model.eval()                              # test mode (matters for dropout)
correct = 0
with torch.no_grad():                     # no gradients needed to test
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)         # the highest score wins
        correct += (pred == y).sum()
print(int(correct) / len(test_set))       # the accuracy: 0.95 is 95 in 100
# endregion

# region: cnn-save
state = model.state_dict()                # every weight and bias, by name
torch.save(state, "mnist_cnn.pt")
# endregion

# region: cnn-load
state = torch.load("mnist_cnn.pt")            # the same numbers, back from disk
model.load_state_dict(state)
# endregion
