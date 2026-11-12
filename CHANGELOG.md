# Changelog

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
