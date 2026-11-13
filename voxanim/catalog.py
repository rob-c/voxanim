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
``seq2seq-attention``          encoder-decoder attention reversing digits   seq2seq_anim
``q-learning``                 tabular Q-learning in a gridworld            qlearning_anim
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


# ------------------------------------------------------------------ behaviour (GPT-2)


def _hallucination():
    from .halluc_anim import Hallucination
    return Hallucination()


register("gpt2-hallucination", _hallucination,
         "GPT-2 small on a real and an invented country: the next-token distributions, sampled "
         "answers that agree or disagree, and why 'unknown' is never likely.")


def _conditioning():
    from .condition_anim import Conditioning
    return Conditioning()


register("gpt2-conditioning", _conditioning,
         "GPT-2 small's next-token distribution reshaped by an instruction and few-shot examples, "
         "sent elsewhere by different examples, and the last layer's attention to the context.")


def _calibration():
    from .calib_anim import Calibration
    return Calibration()


register("gpt2-calibration", _calibration,
         "GPT-2 small on 40 capital cities: its confidence against whether it is right, as a "
         "reliability diagram with the expected calibration error.")

# ------------------------------------------------------------------ GPT-2: tokens to sampling


def _bpe():
    from .bpe_anim import BPEAnimation
    return BPEAnimation()


register("gpt2-tokenizer", _bpe,
         "Byte-pair encoding: GPT-2's tokenizer splitting a sentence into bytes and merging them "
         "by rank into tokens, their ids and embedding rows; why \" four\" and \"four\" differ.")


def _embed():
    from .embed_anim import EmbeddingAnimation
    return EmbeddingAnimation()


register("gpt2-embeddings", _embed,
         "What GPT-2's embeddings know: a token's vector, its nearest neighbours by cosine, "
         "king - man + woman, and the 1024 position embeddings and how alike nearby ones are.")


def _block():
    from .block_anim import TransformerBlockAnimation
    return TransformerBlockAnimation()


register("gpt2-transformer-block", _block,
         "Through the transformer: the last token's residual stream through all 12 layers "
         "(LayerNorm, attention, LayerNorm, MLP), read by the logit lens at every layer.")


def _sampling():
    from .sampling_anim import SamplingAnimation
    return SamplingAnimation()


register("gpt2-sampling", _sampling,
         "Sampling the next token from GPT-2's real distribution: greedy, temperature, top-k, "
         "nucleus (top-p), a short generation, and the KV cache.")
# ------------------------------------------------------------------ the maths, step by step


def _autodiff():
    from .autodiff_anim import AutodiffGraph
    return AutodiffGraph()


register("autodiff-graph", _autodiff,
         "Automatic differentiation on the graph of one neuron, L = 1/2 (sigmoid(w1 x1 + w2 x2 + "
         "b) - y)^2: build the graph, values forward, the chain rule back one edge at a time, "
         "the parameter gradients and one update step.")


def _softmax():
    from .softmax_anim import SoftmaxCrossEntropy
    return SoftmaxCrossEntropy()


register("gpt2-softmax-cross-entropy", _softmax,
         "Softmax and cross-entropy on real logits, side by side: the MNIST MLP on a 3 and GPT-2 "
         "on the next word after 'The capital of France is'. Logits, exp, normalise, the one-hot "
         "target, the loss -log p, the gradient p - y on the bars, then temperature 2 and 0.5.")


def _activations():
    from .activations_anim import Activations
    return Activations()


register("activations", _activations,
         "Activation functions: sigmoid, tanh, ReLU and GELU with their derivatives; a real "
         "layer's 64 pre-activations through each (saturation, sparsity); then dead ReLUs: a "
         "784-128-10 network trained with too large a learning rate, the units that stop firing "
         "going dark for good, and the gradient through a dead unit exactly 0.")
# ------------------------------------------------------------------ seq2seq attention, Q-learning


def _seq2seq():
    from . import seq2seq
    from .seq2seq_anim import Seq2SeqAttention
    return Seq2SeqAttention(seq2seq.trace())


register("seq2seq-attention", _seq2seq,
         "An encoder-decoder with additive attention reversing six digits: the encoder reads, "
         "the decoder attends and copies, the alignment matrix forms row by row, then the "
         "recorded training replays it being learned.")


def _qlearning():
    from . import gridworld
    from .qlearning_anim import QLearning
    return QLearning(gridworld.run())


register("q-learning", _qlearning,
         "Tabular Q-learning in a 6x8 gridworld with walls, a goal and a trap: the first "
         "episodes move by move with the update rule, the Q-table rising as a height-field over "
         "the grid, epsilon decaying, the greedy policy and its shortest path.")
# ------------------------------------------------------------------ Gaussian processes


def _gp_prior():
    from .gp_prior import PriorAnimation
    return PriorAnimation()


register("gp-prior", _gp_prior,
         "A Gaussian process prior: the RBF kernel, its matrix over a grid, sample functions "
         "drawn from it, then the lengthscale and the variance changing with the samples and the "
         "matrix updating together.")


def _gp_posterior():
    from .gp_posterior import PosteriorAnimation
    return PosteriorAnimation()


register("gp-posterior", _gp_posterior,
         "Conditioning a GP on data: six observations of a stated function arrive one at a time "
         "and the posterior mean and band update exactly; the matrices K, k*, k** and the formula "
         "light up piece by piece; then the same with observation noise.")


def _gp_kernels():
    from .gp_kernels import KernelsAnimation
    return KernelsAnimation()


register("gp-kernels", _gp_kernels,
         "Kernels are assumptions: the same 7 points fitted with RBF, Matérn 3/2, periodic, "
         "linear and periodic + RBF kernels (hyperparameters fitted by maximising the marginal "
         "likelihood): the posterior of each, how it extrapolates, and log p(y) as a read-out.")


def _gp_hyper():
    from .gp_hyper import HyperAnimation
    return HyperAnimation()


register("gp-hyperparameters", _gp_hyper,
         "Learning a GP's hyperparameters: the log marginal likelihood as a landscape of voxel "
         "columns over (lengthscale, noise), gradient ascent walking up it with the posterior "
         "changing in step, and the data-fit and complexity terms as bars: the optimum is a "
         "trade-off.")


def _gp_bayesopt():
    from .gp_bayesopt import BayesOptAnimation
    return BayesOptAnimation()


register("gp-bayesian-optimisation", _gp_bayesopt,
         "Bayesian optimisation of a hidden 1-D function: from 2 evaluations, 8 rounds of "
         "expected improvement (drawn under the plot, its maximum marked), each evaluation flying "
         "up to the function and the posterior updating; explore and exploit by turns, the true "
         "maximum revealed at the end.")
# ------------------------------------------------------------------ modern sequence models


def _ssm():
    from .ssm_anim import SSMRecurrence
    return SSMRecurrence()


register("ssm-recurrence", _ssm,
         "A diagonal state space model on a sine wave with a step: the recurrence step by step, "
         "the same output by convolution with the kernel K = C A^k B, then the step size changing "
         "the kernel's length and the state's memory.")


def _mamba():
    from . import mamba, memory
    from .mamba_anim import MambaSelection
    return MambaSelection(mamba.trace(), memory.trace("rnn"),
                          memory.trace("lstm", memory.FORGET_BIAS))


register("mamba-selective-scan", _mamba,
         "A minimal Mamba block on the memory task: the input-dependent step opening at the cue "
         "and shutting on the noise, the state holding the cue across the gap, the parallel scan "
         "in log2 n levels, and its training curve against the RNN's and the LSTM's.")


def _cost():
    from .cost_anim import ContextCost
    return ContextCost()


register("context-cost", _cost,
         "What a growing context costs attention, a recurrence and an SSM: the n x n score "
         "matrix against n sequential steps against a log2 n-level scan, operations per new "
         "token, and GPT-2 small's KV cache growing against Mamba's fixed state.")


def _moe():
    from . import moe
    from .moe_anim import MixtureOfExperts
    return MixtureOfExperts(moe.train(moe.ALPHA), moe.train(0.0))


register("mixture-of-experts", _moe,
         "A tiny mixture of experts on a character language with regimes: tokens at the router, "
         "the top-2 experts computing while the rest idle, outputs combined by the gate weights, "
         "expert utilisation with and without the balancing loss.")


def _rope():
    from .rope_anim import RotaryEmbeddings
    return RotaryEmbeddings()


register("gpt2-rotary-embeddings", _rope,
         "Rotary position embeddings: pairs of a query and a key rotated by position at "
         "different frequencies, the dot product depending on the relative position only, "
         "its decay with distance, against GPT-2's learned absolute positions.")
# ------------------------------------------------------------------ diffusion, PCA and k-means


def _diffusion():
    from . import diffusion
    from .diffusion_anim import DiffusionAnimation
    return DiffusionAnimation(diffusion.trace())


register("diffusion", _diffusion,
         "A small denoising diffusion model on MNIST: digits dissolving into noise step by "
         "step, the noise-prediction loss, new digits sampled back out of noise, the training "
         "curve.")


def _pca_kmeans():
    from . import pca_kmeans
    from .pca_anim import PCAKMeans
    return PCAKMeans(pca_kmeans.trace())


register("pca-and-kmeans", _pca_kmeans,
         "PCA of MNIST (the mean digit, the principal components as images, the variance each "
         "explains, test digits on the PC 1 / PC 2 plane, a digit rebuilt from 2, 10 and 50 "
         "components), then k-means on the two coordinates and how well its clusters match "
         "the labels.")
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


def _normalisation():
    from . import norm_anim
    return norm_anim.Normalisation(norm_anim.runs())


register("normalisation", _normalisation,
         "One hidden layer of a deep MLP mid-training: a minibatch's pre-activations as a sheet, "
         "batch norm down each column, layer norm across each row, the learned scale and shift, "
         "the histograms before and after, then training with and without at a high learning "
         "rate.")


def _overfitting():
    from . import overfit_anim
    return overfit_anim.Overfitting(*overfit_anim.runs())


register("overfitting", _overfitting,
         "A wide MLP overfitting 60 points (MNIST 3s and 8s on two principal components): the "
         "decision field wrapping around the training points while the test loss climbs; then "
         "weight decay, dropout (live mask) and early stopping on a validation set.")
# ------------------------------------------------------------------ optimisation
#
# Walks on a real two-parameter loss landscape (voxanim.landscape) and two
# MNIST MLP training comparisons.


def _landscape_descent():
    from .optim_anim import LossLandscape
    return LossLandscape()


register("loss-landscape-descent", _landscape_descent,
         "The loss landscape of two-weight logistic regression on the MNIST zeros and ones, "
         "as a field of voxel columns; then full-batch gradient descent walks down it, the "
         "gradient arrow and the step at every move, the loss falling.")


def _drunken_walk():
    from .optim_anim import DrunkenWalk
    return DrunkenWalk()


register("drunken-walk", _drunken_walk,
         "Minibatch SGD on the same landscape: walkers with 1, 8 and 64 examples a step "
         "against the full-batch path, the spread the gradient variance predicts drawn as an "
         "ellipse, then a learning rate too high and one too low.")


def _optimisers():
    from .optim_anim import Optimisers
    return Optimisers()


register("optimisers", _optimisers,
         "SGD, momentum, RMSProp, Adam and AdamW from one start on an ill-conditioned "
         "landscape (a narrow valley): momentum's overshoot, the adaptive optimisers' "
         "per-weight rescaling as two arrows and moment bars, who arrives in how many steps.")


def _adamw_vs_l2():
    from .adamw_anim import AdamWvsL2
    return AdamWvsL2()


register("adamw-vs-l2", _adamw_vs_l2,
         "Adam with L2 in the loss against AdamW on a real weight matrix of the MNIST MLP: "
         "the per-weight effective decay as a sheet, the two update rules, the weight norms "
         "over training.")


def _lr_schedule():
    from .schedule_anim import LearningRateSchedules
    return LearningRateSchedules()


register("learning-rate-schedule", _lr_schedule,
         "Constant, step decay, cosine with warmup and one-cycle schedules training the MNIST "
         "MLP: the schedules, the loss curves, the same schedules on the two-weight landscape, "
         "and why warmup tames Adam's first steps.")
# ------------------------------------------------------------------ fine-tuning: LoRA on GPT-2


def _lora():
    from .lora_anim import LoRA
    return LoRA()


register("gpt2-lora", _lora,
         "LoRA on the real GPT-2 small: a frozen 768x768 weight matrix and the two thin matrices "
         "(768x4 and 4x768) trained beside it, 147,456 adapter numbers in all (0.12% of 124M), "
         "teaching the model one new fact in 40 Adam steps; the update's rank, what did not "
         "change, and the merge back into one matrix for inference.")

# ------------------------------------------------------------------ trimming: pruning, distillation
#
# Masks, retraining under a mask, activation energies and distillation on the
# MNIST MLP and CNN (voxanim.prune); the animations in voxanim.prune_anim.


def _magnitude_pruning():
    from . import mnist, prune, prune_anim
    return prune_anim.MagnitudePruning(prune.prune_sweep(),
                                       prune.layers_to_dict(mnist.train()[0]))


register("magnitude-pruning", _magnitude_pruning,
         "The trained 784-64-32-10 MLP's weights as sheets beside the histogram of |w|: a "
         "threshold sweeps up and the weights below it go dark; at 50, 80, 90, 95 and 99% pruned "
         "the test accuracy without retraining, after a fine-tune with the mask held, and for "
         "iterative pruning.")


def _lottery_ticket():
    from . import prune, prune_anim
    return prune_anim.LotteryTicket([prune.lottery(seed=s) for s in (0, 1)])


register("lottery-ticket", _lottery_ticket,
         "The 20%-dense mask from pruning a trained MLP, rewound onto its original "
         "initialisation and retrained against a random mask of the same density and the dense "
         "network: the three test-error curves, and a second seed's.")


def _structured_pruning():
    from . import mnist, prune, prune_anim
    (Wc, bc), (Wd, bd) = mnist.cnn_train()[0]
    x = mnist.load()[2][prune_anim.SP_IMAGE]
    return prune_anim.StructuredPruning(prune.structured_sweep(), dict(Wc=Wc, bc=bc, Wd=Wd, bd=bd),
                                        prune.layers_to_dict(mnist.train()[0]), x)


register("structured-pruning", _structured_pruning,
         "The CNN's 4 filters and the MLP's 64 hidden units ranked by activation energy on the "
         "test set and removed one at a time, least active first: the feature maps and weight "
         "rows disappear, the accuracy against the number removed with and without a brief "
         "fine-tune, and what is left after half are gone.")


def _distillation():
    from . import mnist, prune, prune_anim
    _xtr, _ytr, xte, yte = mnist.load()
    return prune_anim.Distillation(prune.distil(), prune.distil(n_train=60000, steps=1500),
                                   xte[prune_anim.DS_IMAGE], int(yte[prune_anim.DS_IMAGE]))


register("distillation", _distillation,
         "A 784-256-256-10 teacher's output for one 3 as bars, at temperature 1 and softened "
         "at temperature 4 (the dark knowledge in the wrong classes), then a 784-32-10 student "
         "trained on 3,000 images three ways: hard labels, the soft targets, both; the three "
         "accuracy curves and the final numbers.")
