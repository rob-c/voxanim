"""The network animations: every preset that shows a model rather than one convolution.

Each entry registers a factory in :data:`voxanim.presets.REGISTRY` (the media
file has the preset's name).  Factories import their modules lazily, so
``python3 -m voxanim list`` stays fast; the models behind them are trained
on first use and cached (:mod:`voxanim.cache`).  The last column is the
module (under ``voxanim``) that draws each one.

=============================  ===========================================  ===========
preset                         shows                                        module
=============================  ===========================================  ===========
``mlp-1-forward-layer1``       784 pixels into 64 neurons, ReLU             mlp
``mlp-2-forward-layer2``       64 into 32 neurons, ReLU                     mlp
``mlp-3-forward-output``       32 into 10 logits, softmax, loss             mlp
``mlp-4-forward-pass``         the whole forward pass flowing               mlp
``mlp-5-backward-output``      backprop through the output layer            mlp
``mlp-6-backward-layer2``      backprop through layer 2                     mlp
``mlp-7-backward-layer1``      backprop through layer 1                     mlp
``mlp-8-backward-pass``        the whole backward pass flowing              mlp
``mlp-9-training-step``        one SGD step of the MLP on one image         mlp
``cnn-training-step``          one SGD step of a small CNN                  cnn
``rnn-training-step``          an RNN on a sine wave: BPTT, update          recurrent
``lstm-training-step``         the same with an LSTM                        recurrent
``attention-training-step``    3-head self-attention (tiny ViT), one step   attention
``strided-cnn-classifier``     stride-2 convolutions, then a dense head     conv
``dnn-eval``                   a loaded dense network evaluating a digit    dnn
``rnn-batch-training``         batches of sine waves through a 2-layer RNN  rnn_batches
``autoencoder-dense-3d``       dense autoencoder, 3-D latent space          ae_anim
``autoencoder-dense-2d``       dense autoencoder, 2-D latent space          ae_anim
``autoencoder-conv-3d``        convolutional autoencoder, 3-D latent        ae_anim
``autoencoder-conv-2d``        convolutional autoencoder, 2-D latent        ae_anim
``vae-dense-3d``               dense VAE, 3-D latent space                  vae_anim
``vae-dense-2d``               dense VAE, 2-D latent space                  vae_anim
``unet-dense``                 a dense U-Net denoising digits               unet_anim
``dec-dense``                  Deep Embedded Clustering                     dec_anim
``idec-dense``                 Improved DEC                                 dec_anim
``gpt2-attention``             inside GPT-2 small                           gpt2_anim
``gradient-descent``           a 2-3-1 network learning XOR                 gd_anim
``mlp-classifier``             a 784-32-16-10 node-and-line diagram         mlp_graph
``decision-tree``              CART on a 2-D, 3-class dataset               dtree
=============================  ===========================================  ===========

Options a factory accepts (``--digit``, ``--model``, ...) are its keyword
parameters, passed by :func:`voxanim.presets.build`.  ``digit`` is always an
*index into the MNIST test set*, not the digit drawn (test image 18 is a 3).
"""

from __future__ import annotations

from functools import cache, partial

from . import color as C
from . import mlp
from .conv import ConvAnimation, ConvSpec, Layout, Tempo
from .presets import preset
from .text import TIMES


def register(name, build, about):
    """Register ``build`` as preset ``name`` (its media file is ``name`` too)."""
    preset(name, name, about=about)(build)


# ------------------------------------------------------------------ an MNIST classifier

MLP_INDEX = 18          # test image: a '3' the MLP gets right, but only at 62% (8 is next)
TRAIN_LR = 0.002        # one SGD step: p(3) 0.62 -> 0.87, loss 0.48 -> 0.14


@cache
def _trace():
    from . import mnist
    return mnist.trace(MLP_INDEX)


@cache
def _trained():
    from . import mnist
    return mnist.sgd_step(_trace(), TRAIN_LR)


register("mlp-1-forward-layer1", lambda: mlp.DenseForward(_trace(), 0),
         "MNIST MLP, layer 1: 784 pixels into 64 neurons, then ReLU.")
register("mlp-2-forward-layer2", lambda: mlp.DenseForward(_trace(), 1),
         "MNIST MLP, layer 2: 64 into 32 neurons, then ReLU.")
register("mlp-3-forward-output", lambda: mlp.DenseForward(_trace(), 2),
         "MNIST MLP, output: 32 into 10 logits, softmax, one-hot target, loss.")
register("mlp-4-forward-pass", lambda: mlp.NetworkFlow(_trace()),
         "MNIST MLP: the whole forward pass as information flowing.")
register("mlp-5-backward-output", lambda: mlp.DenseBackward(_trace(), 2),
         "Backprop through the output layer: p - y back to 32 neurons.")
register("mlp-6-backward-layer2", lambda: mlp.DenseBackward(_trace(), 1),
         "Backprop through layer 2, gated by ReLU.")
register("mlp-7-backward-layer1", lambda: mlp.DenseBackward(_trace(), 0),
         "Backprop through layer 1: weight gradients only where pixels > 0.")
register("mlp-8-backward-pass", lambda: mlp.NetworkFlow(_trace(), backward=True),
         "MNIST MLP: the whole backward pass, gradients flowing back to the pixels.")
register("mlp-9-training-step", lambda: mlp.TrainingStep(_trace(), _trained(), TRAIN_LR),
         "MNIST MLP: one full training step on one image: forward, loss, backward, update, "
         "forward again.")


# ------------------------------------------------------------------ a small CNN

CNN_INDEX = 18          # the same '3': right, but only at 70% (5 next)
CNN_LR = 0.005          # one SGD step: p(3) 0.70 -> 0.88


@cache
def _cnn_traces():
    from . import mnist
    tr = mnist.cnn_trace(CNN_INDEX)
    return tr, mnist.cnn_sgd_step(tr, CNN_LR)


def _cnn_step():
    from .cnn import ConvNetTrainingStep
    tr, tr2 = _cnn_traces()
    return ConvNetTrainingStep(tr, tr2, CNN_LR)


register("cnn-training-step", _cnn_step,
         "MNIST CNN (4 conv filters, max-pool, dense): one full training step on one image.")


# ------------------------------------------------------------------ RNN / LSTM on a sine wave


def _seq_step(kind):
    """One training step of the ``kind`` (``"rnn"`` / ``"lstm"``) sequence model."""
    from . import sequence
    from .recurrent import RecurrentTrainingStep
    tr = sequence.trace(kind)
    lr = sequence.STEP_LR[kind]
    return RecurrentTrainingStep(tr, sequence.sgd_step(tr, lr), lr,
                                 sequence.continue_training(kind))


register("rnn-training-step", partial(_seq_step, "rnn"),
         "RNN unrolled over 16 steps of a sine wave: forward, loss, BPTT, update, forward again, "
         "then 1000 more steps until the predictions follow the wave.")
register("lstm-training-step", partial(_seq_step, "lstm"),
         "LSTM (gates and cell state) on a sine wave: forward, loss, BPTT, update, forward again, "
         "then 1000 more steps until the predictions follow the wave.")


def _gated_memory():
    """A plain RNN and an LSTM, the same size and training, on a task that
    needs memory across a gap of noise: only the gated one learns it."""
    from . import memory
    from .gated_anim import GatedMemory
    return GatedMemory(memory.trace("rnn"), memory.trace("lstm", memory.FORGET_BIAS),
                       memory.train("lstm", memory.LEAKY_BIAS))


register("gated-memory", _gated_memory,
         "RNN vs LSTM on a task that needs memory across 49 steps of noise: the answers, what "
         "each state still knows of the cue, the vanishing gradient, the training curves.")


# ------------------------------------------------------------------ self-attention (tiny ViT)

ATTN_INDEX = 15         # test image: a '5' the model gets right at 66% ('3' next)
ATTN_LR = 0.001         # one SGD step: p(5) 0.66 -> 0.82


def _attn_step():
    from . import vit
    from .attention import AttentionTrainingStep
    tr = vit.trace(ATTN_INDEX)
    return AttentionTrainingStep(tr, vit.sgd_step(tr, ATTN_LR), ATTN_LR)


register("attention-training-step", _attn_step,
         "3-head self-attention (tiny ViT) on MNIST: one full training step on one image.")


# ------------------------------------------------------------------ a strided-conv classifier


def strided_classifier(digit=0):
    """Two stride-2 convolutions shrink each channel 784 -> 169 -> 36, then a
    dense layer classifies (a real model trained on MNIST).

    ``digit`` (``--digit``) is the index of the MNIST test image shown.
    """
    from . import strided
    from .conv import HeadSpec
    tr = strided.trace(int(digit))
    P = tr.P
    gray_in = tuple(C.gray(0.0).tolist())
    common = dict(kernel=(3, 3), stride=(2, 2), activation="relu", track_groups=False,
                  styles={"input": "gray", "kernel": "diverging", "output": "relu"},
                  pad_color=gray_in, transition=(0.25, 0.5), intro=0.6)
    view = dict(ppu=12.5 / 540.0, azimuth=40.0, elevation=27.0, floor_gap=0.9)
    s1 = ConvSpec(name="conv1", in_hw=(28, 28), in_ch=1, filters=4, data=tr.acts[0],
                  weights=P["W0"].transpose(3, 0, 1, 2), bias=P["b0"],
                  labels={"input": "image 28%s28: 784 per channel" % TIMES,
                          "kernel": "4 kernels 3%s3, stride 2" % TIMES,
                          "output": "13%s13%s4: 169 per channel" % (TIMES, TIMES)},
                  layout=Layout(input=(0.15, 0.60), output=(0.44, 0.66), active=(0.40, 0.37),
                                queue=(0.49, 0.30), done=(0.17, 0.95), queue_axis="x",
                                **view),
                  tempo=Tempo(slow=1.5, fast=0.02, ramp_in=6, ramp_out=1.5), **common)
    s2 = ConvSpec(name="conv2", in_hw=(13, 13), in_ch=4, filters=8, data=tr.acts[1],
                  weights=P["W1"].transpose(3, 0, 1, 2), bias=P["b1"],
                  labels={"kernel": "8 kernels 3%s3%s4, stride 2" % (TIMES, TIMES),
                          "output": "6%s6%s8: 36 per channel" % (TIMES, TIMES)},
                  layout=Layout(output=(0.665, 0.69), active=(0.62, 0.40), queue=(0.69, 0.32),
                                done=(0.46, 0.96), queue_axis="x", spacing=0.8, **view),
                  tempo=Tempo(slow=1.3, fast=0.045, ramp_in=5, ramp_out=1.5), **common)
    head = HeadSpec(W=P["Wd"], b=P["bd"], at=(0.87, 0.58), scale=1.2, label=tr.label,
                    title="dense 288")
    return ConvAnimation([s1, s2], title="Strided convolutions: less data per channel",
                         subtitle="each 3%s3 kernel steps 2 pixels: 28%s28 (784 values per "
                         "channel) to 13%s13 (169) to 6%s6 (36), then a dense layer classifies"
                         % (TIMES, TIMES, TIMES, TIMES), labels=True, head=head, hold=2.5)


register("strided-cnn-classifier", strided_classifier,
         "Stride-2 conv2d layers shrink each channel 784 -> 169 -> 36; then dense -> softmax.")


# ------------------------------------------------------------------ evaluating a loaded model


def dnn_eval(model=None, digit=0, preprocess=None):
    """Load a dense ReLU network (``--model``, default the bundled
    :data:`voxanim.model_io.DEFAULT_MODEL`) and evaluate MNIST test image
    number ``digit`` (``--digit``); ``preprocess`` overrides the file's input scaling."""
    from .dnn import from_file
    from .model_io import DEFAULT_MODEL
    return from_file(model or DEFAULT_MODEL, digit, preprocess)


register("dnn-eval", dnn_eval,
         "Load a trained dense network (default 784-100-100-100-10) and evaluate one digit.")


# ------------------------------------------------------------------ RNN batches / autoencoders


def _rnn_batches():
    from . import sequence
    from .rnn_batches import RNNBatchTraining
    return RNNBatchTraining(sequence.rnn2_training())


register("rnn-batch-training", _rnn_batches,
         "A 2-layer RNN in the centre while batches of sine waves stream through it; "
         "each batch its own colour.")

SHOWCASE = (3, 2, 1)    # test images of a 0, a 1 and a 2, fed through the autoencoders


def autoencoder(kind="dense", latent=3, digit=None):
    """An autoencoder (``kind`` dense / conv) with a 2-D or 3-D latent space.

    ``digit`` (``--digit``) replaces the first of the :data:`SHOWCASE` test
    images by that test-set index.
    """
    from . import autoencoder as A
    from .ae_anim import AutoencoderAnimation
    digits = SHOWCASE if digit is None else (int(digit),) + SHOWCASE[1:]
    return AutoencoderAnimation(A.trace(kind, int(latent), digits))


for _kind in ("dense", "conv"):
    for _lat in (3, 2):
        register("autoencoder-%s-%dd" % (_kind, _lat), partial(autoencoder, _kind, _lat),
                 "%s autoencoder with a %d-D latent space: digits in, codes, latent walk."
                 % ("Dense" if _kind == "dense" else "Convolutional", _lat))


# ------------------------------------------------------------------ VAE, U-Net, DEC, IDEC


def autoencoder_training():
    """A dense autoencoder with a 2-D latent space learning: one digit through
    the untrained network, then the 600 test codes drifting and sorting in the
    latent space as the recorded training run plays."""
    from . import autoencoder as A
    from .ae_training import AutoencoderTraining
    return AutoencoderTraining(A.training_run("dense", 2, SHOWCASE))


register("autoencoder-training", autoencoder_training,
         "An autoencoder learning its 2-D latent space: the untrained network, then 600 test "
         "codes drifting and sorting as training runs and the reconstructions sharpen.")


def vae(latent=3, digit=None):
    """A dense VAE with a ``latent``-D space; ``digit`` as for :func:`autoencoder`."""
    from .vae_anim import VAEAnimation, vae_trace
    digits = SHOWCASE if digit is None else (int(digit),) + SHOWCASE[1:]
    return VAEAnimation(vae_trace(latent, digits))


for _lat in (3, 2):
    register("vae-dense-%dd" % _lat, partial(vae, _lat),
             "Dense VAE with a %d-D latent space: mu, sigma, sampling, generation." % _lat)


def _unet():
    from .unet_anim import UNetAnimation
    return UNetAnimation()


register("unet-dense", _unet, "Dense U-Net denoising digits; skip connections across the U.")


def _cluster(kind):
    """Deep clustering (``kind`` ``"dec"`` / ``"idec"``) of the 3-D autoencoder codes."""
    from . import deep_models as D
    from .dec_anim import ClusterAnimation
    return ClusterAnimation(D.cluster_train(kind))


register("dec-dense", partial(_cluster, "dec"),
         "Deep Embedded Clustering on the 3-D autoencoder codes.")
register("idec-dense", partial(_cluster, "idec"),
         "Improved DEC: clustering loss + reconstruction loss keeps the decoder.")


def _gpt2(text=None):
    """GPT-2 small reading ``text`` (``--text``; default its built-in prompt)."""
    from .gpt2_anim import GPT2Animation
    return GPT2Animation(text) if text else GPT2Animation()


register("gpt2-attention", _gpt2,
         "Inside GPT-2 small: one attention head up close (Q, K, V, the transpose, Q.K^T, "
         "softmax, A.V), then the 12 heads of its layer, then all 12 layers.")


# ------------------------------------------------------------------ diagrams


def _gd():
    from .gd_anim import GradientDescent
    return GradientDescent()


register("gradient-descent", _gd,
         "Gradient descent through a 2-3-1 sigmoid network learning XOR: data forward (teal), "
         "gradients back (magenta), each neuron's gradient recorded, the loss, the update.")


def _mlp_graph():
    from .mlp_graph import MLPClassifier
    return MLPClassifier()


register("mlp-classifier", _mlp_graph,
         "A 784-32-16-10 dense MNIST classifier as the classic node-and-line diagram: one digit "
         "layer by layer, softmax, then more digits.")


def _dtree():
    from .dtree import DecisionTree
    return DecisionTree()


register("decision-tree", _dtree,
         "A decision tree (CART, Gini) grown on a 2-D, 3-class dataset: the threshold search, "
         "the splits, the regions, then new points falling down the tree.")


# ------------------------------------------------------------------ training deep networks


def _initialisation():
    from . import init_anim
    return init_anim.Initialisation(init_anim.runs())


register("initialisation", _initialisation,
         "A 20-layer tanh MLP under three weight scales: activation std and gradient per layer "
         "(vanishing, exploding, Xavier), one layer's histogram, then which one learns.")


def _skip_connections():
    from . import residual_anim
    return residual_anim.SkipConnections(*residual_anim.runs())


register("skip-connections", _skip_connections,
         "A 30-layer plain ReLU MLP against the same with a skip every two layers: the "
         "activations fading forward, the gradient shrinking back layer by layer (log scale) "
         "or arriving intact along the skips, then the training curves.")
