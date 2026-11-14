# Changelog

## Unreleased

- **Acceleration** (`accel.py`, `deepnet_torch.py`, `raster_torch.py`):
  optional PyTorch on Metal or CUDA, chosen by `VOXANIM_DEVICE`
  (`auto`, `mps`, `cuda`, `cpu`, `numpy`). `auto` takes the fastest measured
  path: the convolutional autoencoders train 9× faster on an M5 Max's GPU
  and large-batch `deepnet` runs 2×; small-batch training stays numpy, bit
  for bit.
  Every port is checked against the numpy model in float64. Cached weights are
  keyed by settings, not device, so earlier artefacts stay valid.
- **A Metal kernel composites each frame** (`raster_metal.py`): primitives
  are recorded, binned into tiles and composited by one thread per pixel in
  the order the CPU would draw them. Voxels and lines are bit-identical to
  the numpy renderer; 349 of 354 sampled frames are identical in 8 bits (the
  rest: 6 pixels, one level). 8K frames are 1.8–9.8× faster (a `conv2d`
  frame: 3.2 s instead of 31.8 s); a 960×540 animation renders 1.1–2× faster. `auto` uses it on Apple
  GPUs at every size.
- **Faster still, exactly:** camera matrices cached (a line-heavy frame
  624 → 71 ms), every line of a frame rasterised in one vectorised pass, lit
  voxels and the floor's reflection drawn by the Metal kernel, and encoding
  overlapped with rendering (the MP4 and GIF side by side, in the background
  while the next animation renders). Six animations rendered and encoded:
  223 s → 95 s.
- **Four x264 threads for the MP4** (`render.X264_THREADS`, recorded in each
  manifest): an encode 2.5× faster (14 s → 5 s for 1,950 frames), still
  repeatable run to run. MP4 bytes now differ from 1.4.0's single-thread
  encodes; GIFs and WebMs are unchanged.
- **Leaner drawing code, exactly:** motion blur's sub-frames and travel
  probes are built without text (`layout.textless`: their text is never
  drawn); subtitles wrap by binary search, not by measuring every split
  (live numbers make new strings every frame); `Diagram.P` remembers its
  points; chart helpers clip scalars without numpy; GPT-2 panels cull and
  bound in plain floats. Scene building 1.1–1.7× faster in the costliest
  animations (`activation-smoothness` 16 → 9 ms a frame, `gpt2-attention`
  1.46×); twelve animations rendered and encoded in 74–81 s (582 s with the
  original code). All exact: every checked frame of all 118 animations is
  unchanged.
- **Asynchronous uploads** (no GPU stall per motion-blur sub-frame:
  1.25–1.4× on such frames), unprojected points remembered per camera
  (`rnn-batch-training` 224 → 106 ms a frame), and six encodes in flight
  (the encoding tail after the last frame 21 s → 9 s). All exact.
- **Diagram quads cached on the GPU** (keyed shaders: discs, rectangles,
  images, GPT-2 panels), every voxel lit by the kernel (no CPU tints), one
  upload of each kind per frame. All exact. `gpt2-attention`, the costliest
  animation, 2–4× faster; twelve animations 582 s → 98–134 s.
- **Probe-only scene builds** for motion blur's travel measurement (the same
  probes, checked at 4,720 instants), and the dither made once per size.
- **Voxels as arrays, end to end:** `Scene.voxel_batch` and a whole-block
  `Block.emit`; the renderer, shadows, reflection and layout read arrays; the
  Metal path takes voxels in blocks. Lines cached on the GPU between frames,
  and the resolve is a Metal kernel. All exact. Twelve animations rendered
  and encoded: 582 s with the original code, 147 s now (4.0×).
- **One worker pool per `render` run** (`render.render_many`): no start-up
  or idle tail per animation; lines rasterised only in the tiles they reach;
  sprite-pool lookups by identity. All exact.
- **`voxanim gpucheck`**: renders every animation both ways and reports
  identical frames, the largest differences and the speed-up.
- **Faster rendering, exactly:** text measurement and rasterisation are
  cached, and fully visible sprites skip their depth mask (a diagram frame
  0.21 → 0.13 s); every frame of all 118 animations is bit-identical.
- `pip install voxanim[gpu]`; a render manifest records its framebuffer.

## 1.4.0

Five new animations on regularisation, every one a real model trained in
plain numpy with its claims tested; the 113 animations of 1.3.0 are
untouched (the old `deepnet` options train bit for bit as before). The
release also carries everything 1.3.0 added, since 1.3.0 was tagged but
never published.

- **Regularisation** (`regul.py`, `regul_ae.py`): `l1-vs-l2` (the balls, a
  real two-weight loss surface, the soft threshold, lasso and ridge paths),
  `l1-l2-weights` (L2, proximal L1, elastic net and L1 by its slope on an
  MLP's first layer), `regularisers` (dropout, input noise, max-norm, label
  smoothing), `regularisation-race` (seven regularisers on 1,000 digits, and
  the strength sweeps), `regularised-autoencoder` (a tied-weight MNIST
  autoencoder with L1 on the code, weight decay, denoising and a contractive
  penalty, each held to the same clean error).
- **`deepnet.py`:** L1 by its subgradient or as SGD's proximal step,
  max-norm, label smoothing, and weight snapshots through training; off by
  default and absent from the cache key when off.
- **Website:** a Regularisation section.

## 1.3.0

Twenty-five new animations on the details of training, in seven families,
every one a real model trained in plain numpy with its claims tested; the
88 animations of 1.2.0 are pixel-identical.

- **Generalisation** (`generalise.py`): `double-descent`, `memorising-noise`,
  `data-vs-model-size`, `augmentation`, `sharp-vs-flat-minima`.
- **Activation functions** (`actfun.py`; `deepnet.py` gains sigmoid, Leaky
  ReLU, GELU, SiLU and ELU): `activation-race`, `activation-gradients`,
  `activation-sparsity`, `activation-smoothness`, `activation-depth`.
- **Capacity and bottlenecks** (`capacity.py`): `bottleneck-sweep`,
  `information-bottleneck`, `rank-of-a-layer`, `width-vs-depth`.
- **Trimming** (`prune.py`): `magnitude-pruning`, `lottery-ticket`,
  `structured-pruning`, `distillation`.
- **Fine-tuning and transfer** (`transfer.py`, `lora.py`): `transfer-learning`,
  `catastrophic-forgetting`, `learning-rate-finder-and-restarts`, `gpt2-lora`
  (a hand-written backward pass through GPT-2, checked against finite
  differences).
- **Quantization and precision** (`quant.py`): `quantization`,
  `quantization-aware-training`, `mixed-precision`.
- **Website:** every section on view (the bar wraps; a menu button on phones);
  every clip animates cheaply (a 320 px, 12 fps thumbnail of its first half
  with a pause before the loop; the full-rate preview on hover; the full file
  on click, with a visible cue); dark mode; the Pages workflow caches media.

## 1.2.0 (website changes shipped later, unreleased at the time)

- **Website:** light on weak machines and slow connections while every clip
  still animates: the grid plays thumbnails (the first half of each
  animation at 320 px and 12 frames a second, a 2 s hold before the loop;
  about 5 MB for all 88 against 124 MB of full files), a hovered or focused
  clip plays
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
