"""From an autoencoder to a GAN with the same blocks: the program of the AE-to-GAN animations of
"Blocks and code".

    python examples/ae_gan.py

1. An autoencoder: an encoder squeezes a digit (784 numbers) to 16, a decoder rebuilds the
   784 from them, and the loss compares the rebuilt digit with the real one pixel by pixel.
2. The same blocks wired again as a GAN. The decoder, with the weights it learned, becomes the
   generator: it is given made-up codes (noise shaped like the encoder's codes) instead of
   codes of real digits. The encoder's body, as it is, becomes the judge (the
   discriminator); only its last layer is new, giving 1 number ("real?") instead of 16.
   Then the GAN game: the judge learns to tell real digits from made ones, and the generator
   learns to fool it. No digit is ever squeezed and rebuilt: the generator is scored on the
   whole picture by the judge, not pixel by pixel.

Needs torch and torchvision; downloads MNIST into ./data. As in mnist_blocks.py,
"# region: NAME" ... "# endregion" marks what goes on screen, lines kept to 41 characters.
"""
# region: ag-imports
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# endregion

torch.manual_seed(0)
train_set = MNIST("data", download=True, transform=ToTensor())
X = torch.stack([img.flatten() for img, _ in train_set])    # [60000, 784], 0..1

# region: ag-blocks
encoder = nn.Sequential(                  # a digit -> 16 numbers
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 16))
decoder = nn.Sequential(                  # 16 numbers -> a digit
    nn.Linear(16, 256), nn.ReLU(),
    nn.Linear(256, 784), nn.Sigmoid())
# endregion


def batch(n=64):
    return X[torch.randint(len(X), (n,))]


def train_autoencoder(steps=3000):
    blocks = [*encoder.parameters(), *decoder.parameters()]
    optimizer = torch.optim.Adam(blocks, lr=1e-3)
    for _ in range(steps):
        x = batch()
        # region: ag-ae
        code = encoder(x)                 # squeeze: 784 -> 16
        back = decoder(code)              # rebuild: 16 -> 784
        loss = F.binary_cross_entropy(
            back, x)                      # pixel by pixel
        # endregion
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def rewire():
    """The GAN, from the autoencoder's blocks."""
    global generator, judge, mean, spread
    with torch.no_grad():
        codes = encoder(X[:5000])         # what real codes look like
    mean, spread = codes.mean(0), codes.std(0)
    # region: ag-rewire
    generator = decoder                   # the same block, same weights
    judge = nn.Sequential(                # the encoder's body, as it is
        encoder[0], encoder[1],
        nn.Linear(256, 1))                # new: 1 number, "real?"
    # endregion


def train_gan(steps=6000):
    beta = (0.5, 0.999)
    og = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=beta)
    oj = torch.optim.Adam(judge.parameters(), lr=2e-4, betas=beta)
    ones, zeros = torch.ones(64, 1), torch.zeros(64, 1)
    for _ in range(steps):
        x = batch()
        # region: ag-judge
        z = mean + spread * torch.randn(64, 16)  # made-up codes
        fake = generator(z)               # digits from nothing
        bce = F.binary_cross_entropy_with_logits
        said_real = judge(x)              # should say "real"
        said_fake = judge(fake.detach())  # should say "made up"
        lj = bce(said_real, ones)         # wrong about real ones
        lj = lj + bce(said_fake, zeros)   # and about made ones
        # endregion
        oj.zero_grad()
        lj.backward()
        oj.step()
        # region: ag-generator
        lg = bce(judge(fake), ones)       # fool the judge
        # endregion
        og.zero_grad()
        lg.backward()
        og.step()


if __name__ == "__main__":
    train_autoencoder()
    rewire()
    train_gan()
    with torch.no_grad():
        made = generator(mean + spread * torch.randn(16, 16))
    print("16 made-up digits:", made.shape)
