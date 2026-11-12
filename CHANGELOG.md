# Changelog

## Unreleased

- **Website:** light on weak machines and slow connections while every clip
  still animates: the grid plays thumbnails (the first half of each
  animation at 480 px and 2 frames a second, a 2 s hold before the loop; 3 MB
  for all 88 against 124 MB of full files), a hovered or focused clip plays
  its full-rate 480 px preview, and the player plays the full file. Dark mode
  (system or button). The Pages workflow caches the release's media between
  builds.

## 1.2.0

Thirty-three new animations in nine families, for a year of lectures. Every
one is built from a model run or trained in plain numpy, with its numbers
tested; the 55 existing animations are pixel-identical to 1.1.0.

- **Optimisation** (`landscape.py`: logistic regression on two real MNIST
  features as a walkable loss surface): `loss-landscape-descent`,
  `drunken-walk`, `optimisers`, `adamw-vs-l2`, `learning-rate-schedule`.
  `optim.py` gains `SGD`, `Momentum`, `RMSProp` and `AdamW` (`Adam` unchanged).
- **The maths, step by step:** `autodiff-graph`, `gpt2-softmax-cross-entropy`,
  `activations`.
- **Training deep networks** (`deepnet.py`): `overfitting`, `initialisation`,
  `skip-connections`, `normalisation`.
- **GPT-2, tokens to sampling:** `gpt2-tokenizer`, `gpt2-embeddings`,
  `gpt2-transformer-block`, `gpt2-sampling` (`gpt2.py` gains `bpe_trace`,
  `block_states` and `logit_lens`, all additive).
- **What language models know, and don't:** `gpt2-hallucination`,
  `gpt2-conditioning`, `gpt2-calibration` (`gpt2_probe.py`).
- **Attention between two sequences:** `seq2seq-attention`.
- **Gaussian processes** (`gp.py`): `gp-prior`, `gp-posterior`, `gp-kernels`,
  `gp-hyperparameters`, `gp-bayesian-optimisation`.
- **Modern sequence models:** `ssm-recurrence`, `mamba-selective-scan`,
  `context-cost`, `mixture-of-experts`, `gpt2-rotary-embeddings`.
- **Diffusion, PCA and k-means:** `diffusion`, `pca-and-kmeans`.
- **Reinforcement learning:** `q-learning`.
- Website sections for every family; a README section per family with the
  numbers and the settings each was chosen on.

## 1.1.0

- **New animation `autoencoder-training`:** a dense autoencoder learning its
  2-D latent space: one digit through the untrained network, then the
  recorded training run on a log time axis, the 600 test codes drifting and
  sorting, three digits leaving trails, the output sharpening, a loss curve.
  `autoencoder.training_run` snapshots a run that reproduces the cached
  model byte for byte.
- **New animation `gated-memory`:** a plain RNN against an LSTM on a task
  that needs memory across 49 steps of noise (`memory.py`, `gated_anim.py`):
  the answers, what each state still knows of the cue (the run repeated with
  the cue flipped), the gradient of the final error vanishing through the
  RNN but not along the cell state, and the training curves, including an
  LSTM whose forget gate starts half shut and fails like the RNN.
- **`rnn-training-step`, `lstm-training-step`:** a sixth phase, "keep
  training": 1000 more recorded steps (`sequence.continue_training`) until
  the predictions follow the wave, with a loss curve. The step indicator
  says "backward" instead of "backward (BPTT)" to make room.
- **Website:** every animation on one page, built from the preset registry and
  the latest release's videos (`website/`), deployed to GitHub Pages.

## 1.0.0

The first release as a standalone repository. Before this, the code lived in
`anim/` of the animatedai.github.io site.

- **53 animations:**
  - convolutions: the conv2d originals, the conv1d pair and 15 lessons;
  - MLP, CNN, attention, RNN/LSTM and strided-CNN training steps;
  - model evaluation; autoencoders, VAE, U-Net, DEC/IDEC;
  - GPT-2 small, from one head to all 12 layers;
  - gradient descent through a small network, an MLP classifier diagram
    and a decision tree.
- **Reproducible:** bundled fonts; downloads verified by SHA-256 (GPT-2
  pinned to a revision); trained weights cached under a hash of all their
  settings; bit-exact encoding; a JSON manifest per render.
- **Tests:** pytest suite with real skips for missing data; a loop-seam test
  for every animation; CI on GitHub Actions.
- Every refactor up to this release was checked frame by frame against the
  previous output. The intended visual changes are the fonts, the
  `rnn-batch-training` fast-forward phase (which never showed before) and
  the minus signs in GPT-2's worked scores.
