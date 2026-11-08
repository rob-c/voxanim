"""Teaching animations: one idea each, titled and labelled.

They share the originals' look but use fewer filters so each one is short, and
they caption every tensor with its shape.  Rendered files are named
``lesson-<name>``.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from . import color as C
from . import mlp
from .conv import CombineSpec, ConvAnimation, ConvSpec, Layout, Tempo
from .presets import preset

X = "×"

# A roomier default than the originals': labels need space under each block.
EDU = Layout(input=(0.215, 0.46), output=(0.75, 0.62), active=(0.47, 0.52),
             queue=(0.63, 0.30), done=(0.10, 0.90), ppu=22.0 / 540.0)
QUICK = Tempo(slow=1.0, fast=0.16, ramp_in=6, ramp_out=2)


def lesson(name, title, subtitle=None, **spec):
    """A single-stage, titled, labelled animation."""
    spec.setdefault("layout", EDU)
    spec.setdefault("tempo", QUICK)
    return ConvAnimation(ConvSpec(name=name, **spec), title=title, subtitle=subtitle,
                         labels=True)


# ------------------------------------------------------------------ step size


def _stride(s):
    out = (9 - 3) // s + 1
    return lesson("stride-%d" % s, in_hw=(9, 9), in_ch=4, filters=2, stride=(s, s),
                  title="Stride %d" % s,
                  subtitle="the kernel steps %d pixel%s at a time: 9%s9 becomes %d%s%d"
                  % (s, "" if s == 1 else "s", X, out, X, out),
                  tempo=Tempo(slow=1.0, fast=0.16 if s == 1 else 0.3, ramp_in=6, ramp_out=2))


for _s in (1, 2, 3):
    preset("lesson-stride-%d" % _s, "lesson-stride-%d" % _s,
           about="Step size %d: a 3x3 kernel over a 9x9x4 input." % _s)(
        (lambda s: (lambda: _stride(s)))(_s))


# ------------------------------------------------------------------ dilation


def _dilation(d):
    ext = 2 * d + 1
    out = 9 - ext + 1
    L = Layout(input=(0.20, 0.47), output=(0.78, 0.62), active=(0.49, 0.53),
               queue=(0.68, 0.29), done=(0.10, 0.90), ppu=20.0 / 540.0)
    return lesson("dilation-%d" % d, in_hw=(9, 9), in_ch=4, filters=2, dilation=(d, d),
                  layout=L,
                  title="Dilation %d" % d,
                  subtitle="%d-pixel gaps between the taps: a 3%s3 kernel sees %d%s%d, "
                  "9%s9 becomes %d%s%d" % (d - 1, X, ext, X, ext, X, out, X, out),
                  tempo=Tempo(slow=1.0, fast=0.2, ramp_in=6, ramp_out=2))


for _d in (2, 3):
    preset("lesson-dilation-%d" % _d, "lesson-dilation-%d" % _d,
           about="Dilation %d: gaps between the kernel's taps." % _d)(
        (lambda d: (lambda: _dilation(d)))(_d))


# ------------------------------------------------------------------ scaling


@preset("lesson-transposed-conv", "lesson-transposed-conv")
def transposed():
    """Upsampling: zeros between the pixels, then an ordinary convolution."""
    L = Layout(input=(0.20, 0.45), output=(0.76, 0.60), active=(0.50, 0.50),
               queue=(0.64, 0.29), done=(0.10, 0.90), ppu=19.0 / 540.0)
    return lesson("transposed", in_hw=(4, 4), in_ch=3, filters=2, input_dilation=(2, 2),
                  padding=(2, 2, 2, 2), layout=L,
                  tempo=Tempo(slow=0.9, fast=0.09, ramp_in=6, ramp_out=2),
                  title="Transposed convolution (stride 2)",
                  subtitle="spread the pixels apart, fill with zeros, pad, convolve: "
                  "4%s4 is upsampled to 9%s9" % (X, X),
                  labels={"input": "input 4%s4%s3, spread and padded to 11%s11"
                          % (X, X, X)})


def _pool_data(seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.9, 0.9, (8, 8, 4))


def _pool(kind):
    return lesson("%s-pool" % kind, in_hw=(8, 8), in_ch=4, kernel=(2, 2), stride=(2, 2),
                  pool=kind, data=_pool_data(),
                  input_colors=[C.rgb("#3f8fb0")],
                  tempo=Tempo(slow=1.1, fast=0.35, ramp_in=4, ramp_out=2),
                  title="%s pooling 2%s2, stride 2" % ("Max" if kind == "max" else "Average", X),
                  subtitle="no weights: every channel keeps its %s of each window, "
                  "8%s8 becomes 4%s4" % ("largest value" if kind == "max" else "mean", X, X))


preset("lesson-max-pool", "lesson-max-pool",
       about="Max pooling 2x2, stride 2: downsampling without weights.")(lambda: _pool("max"))
preset("lesson-avg-pool", "lesson-avg-pool",
       about="Average pooling 2x2, stride 2.")(lambda: _pool("avg"))


@preset("lesson-strided-downsample", "lesson-strided-downsample")
def strided_down():
    """Downsampling with a learned kernel: stride 2 and same padding."""
    return lesson("down", in_hw=(8, 8), in_ch=4, filters=3, stride=(2, 2),
                  padding=(1, 0, 1, 0), tempo=Tempo(slow=1.0, fast=0.3, ramp_in=5, ramp_out=2),
                  title="Downsampling with a strided convolution",
                  subtitle="stride 2 halves the resolution: 8%s8 becomes 4%s4" % (X, X))


@preset("lesson-pointwise", "lesson-pointwise")
def pointwise():
    """A 1x1 convolution only mixes channels."""
    return lesson("pointwise", in_hw=(6, 6), in_ch=8, filters=4, kernel=(1, 1),
                  tempo=Tempo(slow=0.9, fast=0.1, ramp_in=6, ramp_out=2),
                  title="1%s1 convolution" % X,
                  subtitle="each output pixel mixes the 8 channels of one input pixel")


# ------------------------------------------------------------------ conv3d


@preset("lesson-conv3d", "lesson-conv3d")
def conv3d():
    """A 3x3x3 kernel slides along all three axes of a volume."""
    L = Layout(input=(0.21, 0.45), output=(0.75, 0.63), active=(0.48, 0.48),
               queue=(0.62, 0.30), done=(0.10, 0.88), ppu=22.0 / 540.0)
    return lesson("conv3d", in_hw=(6, 6), in_d=6, in_ch=1, filters=1, kernel=(3, 3),
                  kernel_d=3, layout=L, filter_colors=[C.FILTERS[0]],
                  tempo=Tempo(slow=1.1, fast=0.22, ramp_in=6, ramp_out=2),
                  title="Conv3D",
                  subtitle="the kernel slides along depth too: a 6%s6%s6 volume "
                  "becomes 4%s4%s4" % (X, X, X, X))


@preset("lesson-conv3d-channels", "lesson-conv3d-channels")
def conv3d_channels():
    """Conv3D with 2 input channels and 2 filters (one slab per channel)."""
    L = Layout(input=(0.20, 0.44), output=(0.76, 0.63), active=(0.48, 0.50),
               queue=(0.65, 0.28), done=(0.10, 0.90), ppu=19.0 / 540.0)
    return lesson("conv3d-2ch", in_hw=(5, 5), in_d=5, in_ch=2, filters=2, kernel=(3, 3),
                  kernel_d=3, layout=L,
                  tempo=Tempo(slow=1.1, fast=0.25, ramp_in=6, ramp_out=2),
                  title="Conv3D with channels",
                  subtitle="each channel is a 5%s5%s5 volume; a filter spans both "
                  "channels and 3 slices of depth" % (X, X))


# ------------------------------------------------------------------ conv1d


def _seq(L, Cin, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(L)[:, None]
    x = 0.7 * np.sin(t * 0.7 + np.arange(Cin)[None] * 1.3) + rng.normal(0, 0.15, (L, Cin))
    return np.clip(x, -1, 1)[:, None, :]


def _conv1d(name, title, subtitle, **kw):
    L = Layout(input=(0.24, 0.47), output=(0.74, 0.60), active=(0.48, 0.52),
               queue=(0.63, 0.27), done=(0.10, 0.90), queue_axis="x", spacing=2.2,
               ppu=20.0 / 540.0)
    data = _seq(16, 4, 5)
    w = np.random.default_rng(9).normal(0, 0.5, (2, 3, 1, 4))
    return lesson(name, in_hw=(16, 1), in_ch=4, filters=2, kernel=(3, 1), data=data,
                  weights=w, layout=L, title=title, subtitle=subtitle,
                  tempo=Tempo(slow=1.0, fast=0.22, ramp_in=5, ramp_out=2), **kw)


preset("lesson-conv1d-stride-2", "lesson-conv1d-stride-2",
       about="Conv1D with stride 2 over a 16-step, 4-channel sequence.")(
    lambda: _conv1d("conv1d-s2", "Conv1D, stride 2",
                    "the kernel hops two steps: 16 steps become 7", stride=(2, 1)))
preset("lesson-conv1d-dilation-2", "lesson-conv1d-dilation-2",
       about="Conv1D with dilation 2 (WaveNet-style).")(
    lambda: _conv1d("conv1d-d2", "Conv1D, dilation 2",
                    "taps two steps apart see 5 steps at once: 16 become 12",
                    dilation=(2, 1)))


# ------------------------------------------------------------------ edge detection


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = SOBEL_X.T.copy()


def edge_image(n=16):
    """A small greyscale picture with straight and curved edges."""
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    img = np.zeros((n, n))           # 0 = the zero padding: no false border edges
    disc = (xx - n * 0.64) ** 2 + (yy - n * 0.62) ** 2 < (n * 0.26) ** 2
    square = (xx > n * 0.12) & (xx < n * 0.46) & (yy > n * 0.14) & (yy < n * 0.48)
    img[square] = 0.62
    img[disc] = 0.95
    return img


@preset("lesson-edge-detection", "lesson-edge-detection")
def edge_detection():
    """Sobel filters over a greyscale image: Gx and Gy, then |G|."""
    n = 16
    img = edge_image(n)[:, :, None]
    common = dict(in_hw=(n, n), in_ch=1, filters=1, kernel=(3, 3), padding=(1, 1, 1, 1),
                  data=img, pad_color=tuple(C.gray(0.0).tolist()),
                  styles={"input": "gray", "kernel": "diverging", "output": "diverging"},
                  tempo=Tempo(slow=0.9, fast=0.04, ramp_in=8, ramp_out=2), labels=False,
                  kernel_scale=2.4)
    view = dict(ppu=11.5 / 540.0, azimuth=33.0, elevation=24.0, floor_gap=0.9)
    gx = ConvSpec(name="sobel-x", weights=SOBEL_X[None, :, :, None],
                  filter_colors=[C.NEUTRAL],
                  layout=Layout(input=(0.165, 0.50), active=(0.39, 0.25), output=(0.56, 0.28),
                                queue=(0.39, 0.25), done=(0.39, 0.25), **view), **common)
    gy = ConvSpec(name="sobel-y", weights=SOBEL_Y[None, :, :, None], source="first",
                  filter_colors=[C.NEUTRAL],
                  layout=Layout(active=(0.39, 0.76), output=(0.56, 0.75),
                                queue=(0.39, 0.76), done=(0.39, 0.76), **view), **common)
    mag = CombineSpec(sources=(0, 1), op="magnitude", at=(0.84, 0.52), style="gray",
                      label="|G| = sqrt(Gx² + Gy²)  edges")
    anim = ConvAnimation([gx, gy], title="Edge detection with Sobel filters",
                         subtitle="orange = positive, blue = negative weights and responses",
                         labels=True, combine=[mag], hold=2.2)
    # Caption each piece of the pipeline (shape labels add little here).
    a, b = anim.stages[0], anim.stages[1]
    a.spec.labels = {"input": "image 16%s16" % X, "kernel": "Sobel x", "output": "Gx"}
    b.spec.labels = {"kernel": "Sobel y", "output": "Gy"}
    anim._labels = None
    return anim


# ------------------------------------------------------------------ an MNIST classifier


@lru_cache(maxsize=None)
def _trace():
    from . import mnist
    return mnist.trace(18)          # a '3' the network gets right, but only at 62% (8 is next)


TRAIN_LR = 0.002                     # one SGD step: p(3) 0.62 -> 0.87, loss 0.48 -> 0.14


@lru_cache(maxsize=None)
def _trained():
    from . import mnist
    return mnist.sgd_step(_trace(), TRAIN_LR)


def _net(name, fn, about):
    preset(name, name, about=about)(fn)


_net("mlp-1-forward-layer1", lambda: mlp.DenseForward(_trace(), 0),
     "MNIST MLP, layer 1: 784 pixels into 64 neurons, then ReLU.")
_net("mlp-2-forward-layer2", lambda: mlp.DenseForward(_trace(), 1),
     "MNIST MLP, layer 2: 64 into 32 neurons, then ReLU.")
_net("mlp-3-forward-output", lambda: mlp.DenseForward(_trace(), 2),
     "MNIST MLP, output: 32 into 10 logits, softmax, one-hot target, loss.")
_net("mlp-4-forward-pass", lambda: mlp.NetworkFlow(_trace()),
     "MNIST MLP: the whole forward pass as information flowing.")
_net("mlp-5-backward-output", lambda: mlp.DenseBackward(_trace(), 2),
     "Backprop through the output layer: p - y back to 32 neurons.")
_net("mlp-6-backward-layer2", lambda: mlp.DenseBackward(_trace(), 1),
     "Backprop through layer 2, gated by ReLU.")
_net("mlp-7-backward-layer1", lambda: mlp.DenseBackward(_trace(), 0),
     "Backprop through layer 1: weight gradients only where pixels > 0.")
_net("mlp-8-backward-pass", lambda: mlp.NetworkFlow(_trace(), backward=True),
     "MNIST MLP: the whole backward pass, gradients flowing back to the pixels.")
_net("mlp-9-training-step",
     lambda: mlp.TrainingStep(_trace(), _trained(), TRAIN_LR),
     "MNIST MLP: one full training step on one image: forward, loss, backward, update, "
     "forward again.")


# ------------------------------------------------------------------ a small CNN

CNN_LR = 0.005                       # one SGD step: p(3) 0.70 -> 0.88


@lru_cache(maxsize=None)
def _cnn_traces():
    from . import mnist
    tr = mnist.cnn_trace(18)         # the same '3': right, but only at 70% (5 next)
    return tr, mnist.cnn_sgd_step(tr, CNN_LR)


def _cnn_step():
    from .cnn import ConvNetTrainingStep
    tr, tr2 = _cnn_traces()
    return ConvNetTrainingStep(tr, tr2, CNN_LR)


_net("cnn-training-step", _cnn_step,
     "MNIST CNN (4 conv filters, max-pool, dense): one full training step on one image.")


# ------------------------------------------------------------------ RNN / LSTM on a sine wave


def _seq_step(kind):
    from . import sequence
    from .recurrent import RecurrentTrainingStep
    tr = sequence.trace(kind)
    lr = sequence.STEP_LR[kind]
    return RecurrentTrainingStep(tr, sequence.sgd_step(tr, lr), lr)


_net("rnn-training-step", lambda: _seq_step("rnn"),
     "RNN unrolled over 16 steps of a sine wave: forward, loss, BPTT, update, forward again.")
_net("lstm-training-step", lambda: _seq_step("lstm"),
     "LSTM (gates and cell state) on a sine wave: forward, loss, BPTT, update, forward again.")


# ------------------------------------------------------------------ self-attention (tiny ViT)

ATTN_LR = 0.001                      # one SGD step: p(5) 0.66 -> 0.82


def _attn_step():
    from . import vit
    from .attention import AttentionTrainingStep
    tr = vit.trace(15)               # a '5' the model gets right at 66% ('3' next)
    return AttentionTrainingStep(tr, vit.sgd_step(tr, ATTN_LR), ATTN_LR)


_net("attention-training-step", _attn_step,
     "3-head self-attention (tiny ViT) on MNIST: one full training step on one image.")


# ------------------------------------------------------------------ a strided-conv classifier


def strided_classifier(digit=0):
    """Two stride-2 convolutions shrink each channel 784 -> 169 -> 36, then a
    dense layer classifies (a real model trained on MNIST)."""
    from . import strided
    from .conv import HeadSpec
    tr = strided.trace(int(digit))
    P = tr.P
    gray_in = tuple(C.gray(0.0).tolist())
    common = dict(kernel=(3, 3), stride=(2, 2), activation="relu", track_groups=False,
                  styles={"input": "gray", "kernel": "diverging", "output": "relu"},
                  pad_color=gray_in, transition=(0.25, 0.5), intro=0.6, labels=False)
    view = dict(ppu=12.5 / 540.0, azimuth=40.0, elevation=27.0, floor_gap=0.9)
    s1 = ConvSpec(name="conv1", in_hw=(28, 28), in_ch=1, filters=4, data=tr.acts[0],
                  weights=P["W0"].transpose(3, 0, 1, 2), bias=P["b0"],
                  layout=Layout(input=(0.15, 0.60), output=(0.44, 0.66), active=(0.40, 0.37),
                                queue=(0.49, 0.30), done=(0.17, 0.95), queue_axis="x",
                                **view),
                  tempo=Tempo(slow=1.5, fast=0.02, ramp_in=6, ramp_out=1.5), **common)
    s2 = ConvSpec(name="conv2", in_hw=(13, 13), in_ch=4, filters=8, data=tr.acts[1],
                  weights=P["W1"].transpose(3, 0, 1, 2), bias=P["b1"],
                  layout=Layout(output=(0.665, 0.69), active=(0.62, 0.40), queue=(0.69, 0.32),
                                done=(0.46, 0.96), queue_axis="x", spacing=0.8, **view),
                  tempo=Tempo(slow=1.3, fast=0.045, ramp_in=5, ramp_out=1.5), **common)
    head = HeadSpec(W=P["Wd"], b=P["bd"], at=(0.87, 0.58), scale=1.2, label=tr.label,
                    title="dense 288")
    anim = ConvAnimation([s1, s2], title="Strided convolutions: less data per channel",
                         subtitle="each 3%s3 kernel steps 2 pixels: 28%s28 (784 values per "
                         "channel) to 13%s13 (169) to 6%s6 (36), then a dense layer classifies"
                         % (X, X, X, X), labels=True, head=head, hold=2.5)
    st1, st2 = anim.stages[0], anim.stages[1]
    st1.spec.labels = {"input": "image 28%s28: 784 per channel" % X,
                       "kernel": "4 kernels 3%s3, stride 2" % X,
                       "output": "13%s13%s4: 169 per channel" % (X, X)}
    st2.spec.labels = {"kernel": "8 kernels 3%s3%s4, stride 2" % (X, X),
                       "output": "6%s6%s8: 36 per channel" % (X, X)}
    anim._labels = None
    return anim


preset("strided-cnn-classifier", "strided-cnn-classifier",
       about="Stride-2 conv2d layers shrink each channel 784 -> 169 -> 36; then dense -> "
             "softmax.")(strided_classifier)


# ------------------------------------------------------------------ evaluating a loaded model

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "mnist_dnn_100x3.npz")


def dnn_eval(model=None, digit=0, preprocess=None):
    """Load a dense ReLU network (``--model``) and evaluate one MNIST test
    digit (``--digit``)."""
    from .dnn import from_file
    return from_file(model or DEFAULT_MODEL, digit, preprocess)


preset("dnn-eval", "dnn-eval",
       about="Load a trained dense network (default 784-100-100-100-10) and evaluate one "
             "digit.")(dnn_eval)


# ------------------------------------------------------------------ RNN batches / autoencoders


def _rnn_batches():
    from . import sequence
    from .rnn_batches import RNNBatchTraining
    return RNNBatchTraining(sequence.rnn2_training())


_net("rnn-batch-training", _rnn_batches,
     "A 2-layer RNN in the centre while batches of sine waves stream through it; "
     "each batch its own colour.")


def autoencoder(kind="dense", latent=3, digit=None):
    """An autoencoder (``kind`` dense / conv) with a 2-D or 3-D latent space."""
    from . import autoencoder as A
    from .ae_anim import AutoencoderAnimation
    digits = (3, 2, 1) if digit is None else (int(digit), 2, 1)   # a 0, a 1 and a 2
    return AutoencoderAnimation(A.trace(kind, int(latent), digits))


for _kind in ("dense", "conv"):
    for _lat in (3, 2):
        _net("autoencoder-%s-%dd" % (_kind, _lat),
             (lambda k, la: (lambda digit=None: autoencoder(k, la, digit)))(_kind, _lat),
             "%s autoencoder with a %d-D latent space: digits in, codes, latent walk."
             % ("Dense" if _kind == "dense" else "Convolutional", _lat))


# ------------------------------------------------------------------ VAE, U-Net, DEC, IDEC


def _vae(latent):
    def build(digit=None):
        from .vae_anim import VAEAnimation, vae_trace
        digits = (3, 2, 1) if digit is None else (int(digit), 2, 1)
        return VAEAnimation(vae_trace(latent, digits))
    return build


for _lat in (3, 2):
    _net("vae-dense-%dd" % _lat, _vae(_lat),
         "Dense VAE with a %d-D latent space: mu, sigma, sampling, generation." % _lat)


def _unet():
    from .unet_anim import UNetAnimation
    return UNetAnimation()


_net("unet-dense", _unet, "Dense U-Net denoising digits; skip connections across the U.")


def _cluster(kind):
    def build():
        from . import deep_models as D
        from .dec_anim import ClusterAnimation
        return ClusterAnimation(D.cluster_train(kind))
    return build


_net("dec-dense", _cluster("dec"), "Deep Embedded Clustering on the 3-D autoencoder codes.")
_net("idec-dense", _cluster("idec"),
     "Improved DEC: clustering loss + reconstruction loss keeps the decoder.")


def _gpt2(text=None):
    from .gpt2_anim import GPT2Animation
    return GPT2Animation(text) if text else GPT2Animation()


_net("gpt2-attention", _gpt2,
     "Inside GPT-2 small: one attention head up close (Q, K, V, the transpose, Q.K^T, "
     "softmax, A.V), then the 12 heads of its layer, then all 12 layers.")


# ------------------------------------------------------------------ diagrams: gradient descent, MLP, trees


def _gd():
    from .gd_anim import GradientDescent
    return GradientDescent()


_net("gradient-descent", _gd,
     "Gradient descent through a 2-3-1 sigmoid network learning XOR: data forward (teal), "
     "gradients back (magenta), each neuron's gradient recorded, the loss, the update.")


def _mlp_graph():
    from .mlp_graph import MLPClassifier
    return MLPClassifier()


_net("mlp-classifier", _mlp_graph,
     "A 784-32-16-10 dense MNIST classifier as the classic node-and-line diagram: one digit "
     "layer by layer, softmax, then more digits.")


def _dtree():
    from .dtree import DecisionTree
    return DecisionTree()


_net("decision-tree", _dtree,
     "A decision tree (CART, Gini) grown on a 2-D, 3-class dataset: the threshold search, "
     "the splits, the regions, then new points falling down the tree.")
