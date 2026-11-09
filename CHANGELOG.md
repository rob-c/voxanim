# Changelog

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
