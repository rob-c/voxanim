# voxanim

Animations that explain convolutions, neural networks and transformers,
rendered from **real numbers**: every weight, activation, gradient and
probability on screen comes from a model trained or run right here, in plain
numpy. 365 animations, from a 3×3
convolution to the inside of GPT-2, all from
one small renderer (voxels, lines, textured quads and text, written with
numpy and Pillow).

Watch them all on the [website](https://rob-c.github.io/voxanim/). The
rendered MP4s and GIFs, in light and dark mode, with a SHA-256 manifest of
each, are attached to the [GitHub releases](../../releases).

- **Reproducible:** the same commit renders the same pixels and the same
  encoded bytes. See [Reproducibility](#reproducibility).
- **Checked:** every gradient is tested against finite differences, every
  animation's loop seam and layout are tested, and every refactor of the code
  was checked frame by frame against the previous output.
- **Documented:** every module, class and function has a docstring. This
  README covers what each animation shows and how the pieces fit together.

## Quick start

Needs Python 3.11+ and `ffmpeg` on the `PATH` (only for encoding).

```bash
python3 -m pip install -r requirements-lock.txt   # the exact reference versions
python3 -m voxanim list --names                   # the 365 animations
python3 -m voxanim still conv2d -t 1.0 12.7       # single frames -> build/stills/
python3 -m voxanim render gradient-descent        # frames + MP4/GIF -> build/out/
python3 -m voxanim render gradient-descent --theme dark   # the dark-mode copy
python3 -m voxanim prepare                        # download MNIST + GPT-2, train every model
python3 -m voxanim render all --theme both --clean-frames   # everything, light then dark (hours)
python3 -m pytest tests                           # the test suite
```

`pip install -e .` installs the `voxanim` command too.

Other commands:

- `info <name>`: the timeline of one animation.
- `sheet <name>`: a contact sheet in `build/sheets/`.
- `qa [names]`: layout checks without rendering.
- `gpucheck [names]`: the accelerated renderer against numpy, frame by
  frame (see [Acceleration](#acceleration)).
- `encode <name>`: re-encode frames a previous `render` left behind.
- `previews [names]`: make the website's poster and small preview clip again
  from the MP4s in `build/out`, without rendering.
- `master <name>`: 4K/8K HDR broadcast masters.
- `compare <name>`: ours next to the original GIF, when the
  [animatedai.github.io](https://animatedai.github.io) media folder is at
  `$VOXANIM_MEDIA` (default `../media`).

Options for `still`, `render` and `encode`:

- `--quality draft|preview|final|max`: draft drops supersampling, motion
  blur and reflections, for quick iteration; final is 2× supersampled, max 3×.
- `--scale 2` or `--res 4k`: a larger render.
- `--theme light|dark` (`render` and `encode` also take `both`: the light
  pass, then the dark one). Dark mode is the same frame with its neutrals
  turned over: the backdrop, text and borders invert, the colours of the data
  stay. A dark render's files carry `-dark` after the name
  (`<media>-dark.mp4`, `.gif`, `.json`, frames in `build/<name>-dark/`), so
  both themes sit side by side in `build/out`, and each manifest records its
  `theme`. `VOXANIM_THEME=dark` does the same for any command.
- `--out DIR`: for `render` and `encode`, write the encoded files and the
  manifest into `DIR` and the frames into `DIR/frames/`, not `build/`.
- Every frame carries the voxanim mark and the site's address, small and
  grey, in its bottom-right corner; `VOXANIM_BRAND=0` leaves it off.
- `--fps` and `--jobs`.
- `--formats mp4,gif,webm`, `--gif-fps`.
- Model options for some animations: `--digit` (an MNIST test-set index),
  `--model` (your own dense network for `dnn-eval`) and `--text` (your own
  prompt for `gpt2-attention`).


## Animations

| name | media file | |
| --- | --- | --- |
| `conv2d` | `convolution-animation-3x3-kernel` | 7×7×8 input, eight 3×3×8 filters → 5×5×8 |
| `conv2d-same-padding` | `…-same-padding` | `[1,1,1,1]` zero padding → 7×7×8 |
| `conv2d-stride-2` | `…-stride-2` | → 3×3×8 |
| `conv2d-stride-2-same-padding` | `…-stride-2-same-padding` | → 4×4×8 |
| `conv2d-2-groups` | `…-2-groups` | each half of the filters sees half the channels |
| `depthwise` | `depthwise-convolution-…` | one 3×3×1 filter per channel |
| `depthwise-separable` | `depthwise-separable-…` | depthwise (same padding), then 1×1 pointwise |
| `conv1d-full-width-scroll` | `conv1d-full-width-scroll` | length 12 × 8 channels, four 3-tap filters → 10 × 4 |
| `conv1d-row-sweep` | `conv1d-row-sweep` | 1×3 kernel along each row of an 8×8 sheet, boustrophedon |

The conv1d pieces use real numbers: voxel brightness is the value, and the
output is the actual convolution of the shown input with the shown weights.
Pass `data` / `weights` to any `ConvSpec` to get the same for conv2d.

### Teaching set (`lessons.py`, files named `lesson-*`)

| name | shows |
| --- | --- |
| `lesson-stride-1`, `-2`, `-3` | step size: 9×9 → 7×7, 4×4, 3×3 |
| `lesson-dilation-2`, `-3` | gaps between kernel taps; the kernel spreads its taps as it becomes active |
| `lesson-transposed-conv` | upsampling: pixels spread apart, zeros fill in, 4×4 → 9×9 |
| `lesson-strided-downsample` | downsampling with a learned stride-2 kernel |
| `lesson-max-pool`, `lesson-avg-pool` | 2×2 pooling, stride 2: no weights, every channel at once |
| `lesson-pointwise` | a 1×1 convolution only mixes channels |
| `lesson-conv3d` | a 3×3×3 kernel sliding along all three axes of a 6×6×6 volume |
| `lesson-conv3d-channels` | conv3d with 2 input channels (one volume slab each) and 2 filters |
| `lesson-conv1d-stride-2`, `lesson-conv1d-dilation-2` | conv1d variants on a 16-step sequence |
| `lesson-edge-detection` | Sobel x and y over a 16×16 greyscale image, then the magnitude \|G\| |

The edge-detection lesson colours weights and responses on a diverging scale:
orange positive, blue negative, grey zero. Its kernels are drawn magnified so
the signs read.

### A fully-connected network on MNIST (`mlp.py`, `mnist.py`, files named `mlp-*`)

One real image, a "3", through a real classifier. The network is
784 → 64 → 32 → 10 with ReLU, softmax, and cross-entropy against a one-hot
label. It is trained in numpy and reaches about 97% test accuracy. It gets
this digit right, but only at 62%, with "8" next at 38%, so the gradients
have something to say. MNIST (~11 MB) and the weights are cached in
`build/.cache/` after the first run.

| name | shows |
| --- | --- |
| `mlp-1-forward-layer1` | the image appears, only pixels > 0 light up; each of 64 neurons multiplies its 28×28 weight template with the image (zero pixels contribute nothing) and sums; then ReLU |
| `mlp-2-forward-layer2` | the same on the 64 activations: neurons that were 0 add nothing |
| `mlp-3-forward-output` | 32 → 10 logits, softmax probabilities, the one-hot target, the loss |
| `mlp-4-forward-pass` | the whole forward pass as information flowing: only lit pixels and neurons send anything on |
| `mlp-5-backward-output` | the error p − y; per output, its weight gradient (error × h2) and its share of the gradient for h2; then ReLU′ |
| `mlp-6-backward-layer2` | the same one layer down: neurons that were off get no gradient |
| `mlp-7-backward-layer1` | weight gradients = error × pixel, so only pixels > 0 light up; the input gradient dL/dx accumulates |
| `mlp-8-backward-pass` | the whole backward pass: gradients flowing back to the pixels (saliency) |
| `mlp-9-training-step` | one full SGD step on the image: forward, loss, backward, update, then forward again. During the update, one neuron per layer shows its weights, the change −lr × error × input (only inputs > 0) and the result; every neuron with an error updates at once. Afterwards p(3) rises from 0.62 to 0.87 and the loss falls from 0.48 to 0.14 (lr = 0.002, `catalog.TRAIN_LR`) |

The visual language throughout:

- A value > 0 is lit: bright, glowing, and popped towards the viewer.
- A 0 is dark and flush.
- Signed numbers (weights, products, errors, gradients) are orange (+) or
  blue (−).

### A small CNN on MNIST (`cnn.py`, `cnn-training-step`)

The network: x 28×28 → conv 3×3 with 4 filters → ReLU → 2×2 max-pool →
dense 676 → 10 → softmax. It is trained in numpy (about 96% test accuracy)
and cached like the MLP.

The animation shows one SGD step on the same "3" (lr 0.005, `catalog.CNN_LR`):
p(3) rises from 0.70 to 0.88. It is laid out as one row per channel:
kernel → feature map → pooled map → shared output panel. The five steps:

1. **Forward:** filter 1 slides over the image slowly, then the other three
   together; ReLU keeps values > 0; max-pool sweeps its 2×2 windows; the
   dense layer reads the lit cells.
2. **Loss:** softmax against the one-hot target.
3. **Backward:** p − y flows back to the pooled maps. Max-pool routes it
   only to each window's winner and ReLU only where the map was on, so the
   feature maps' gradient is sparse dots. Those cells send their gradient to
   the filters.
4. **Update:** the filters step against their gradient, then so do the dense
   weights into the true digit (only inputs > 0 change).
5. **Forward again:** the same image through the updated network.

### Self-attention on MNIST (`vit.py`, `attention.py`, `attention-training-step`)

The model is a tiny vision transformer, 92.5% test accuracy. The image is
cut into 16 patches of 7×7. Each patch becomes a token, after a learned CLS
token, with positions added (17 × 24). Then one self-attention layer with
**3 heads** of width 8, a residual connection, and a ReLU MLP on the CLS
row, giving 10 logits.

Each head is drawn as a band: Q, K, V (17 × 8), the 17 × 17 attention
matrix, O = A V, and the CLS row of A laid back on the 4 × 4 patch grid
("where CLS looks"). The steps:

1. **Forward:** head 1's scores are built cell by cell (CLS query against
   each key), then softmax turns each row into weights.
2. **Loss:** the CLS output is classified; softmax vs the one-hot label.
3. **Backward:** only the CLS row is read, so only its row of O, A and Q
   gets gradient. Every key and value does.
4. **Update:** each head's Wq, Wk and Wv take a step against their
   gradients.
5. **Forward again:** the attention shifts and p(5) rises from 0.66 to 0.82
   (lr 0.001, `catalog.ATTN_LR`).

### RNN and LSTM on a sine wave (`sequence.py`, `recurrent.py`)

Both networks predict the next value of `sin(w t + phi)` over 16 steps and
are trained with backprop through time. They are drawn unrolled, one column
per step:

- the input wave at the bottom,
- the hidden bands in the middle (RNN: h; LSTM: h, c and the gates o, f,
  i, g, where a lit gate means open),
- predictions (red) against the next value (white ghosts) on top,
- the shared weights on the right.

The steps:

1. **Forward:** the same cell runs at every step and the state carries to
   the right.
2. **Loss:** the errors appear between each prediction and the next value.
3. **BPTT:** gradients flow down from every prediction and left through
   time, while the weight gradients accumulate step by step.
4. **Update:** the shared weights take a step.
5. **Forward again:** the predictions move towards the targets.
6. **Keep training:** 1000 more Adam steps on fresh waves
   (`sequence.continue_training`, which records the example's predictions,
   hidden states and weights after every step). They play on a log time
   axis with a small loss curve; the screen shows the recorded state at each
   step, until the predictions follow the wave.

The networks are shown early in training (`sequence.CHECKPOINT`) so the
errors are visible, and trained on until they are not:

| animation | one step | 1000 more |
| --- | --- | --- |
| `rnn-training-step` | loss 0.045 → 0.023 (lr 0.1) | → 0.004 |
| `lstm-training-step` | loss 0.091 → 0.058 (lr 0.5) | → 0.005 |

The small networks (8 and 6 units) level off there; what remains is mostly
the first step, predicted before anything has been seen.

`tests/test_models.py` checks every gradient of every network
against finite differences. It also checks that layer-1 weight gradients are exactly zero
wherever the pixel is zero. Use `mnist.trace(i)` to animate another test
image, and change `mnist.SIZES` for another architecture (the sheet shapes
in `mlp.SHAPES` must match).

### Memory across a gap: RNN vs LSTM (`memory.py`, `gated_anim.py`, `gated-memory`)

A task that needs memory: say the first value of a sequence after 49 steps
of noise (`x_0` is the cue, `x_1 … x_49` uniform noise, loss
`½ (y_49 − cue)²`). A plain tanh RNN and an LSTM, 8 units each, are trained
the same way (Adam, 8000 steps on fresh sequences) and drawn unrolled through
time, one above the other:

1. **The task.**
2. **Forward:** both read the same sequence; the RNN answers about 0, the
   LSTM the cue.
3. **Memory:** the run again with the cue flipped. The bands switch to the
   difference between the two runs, which is what each state still knows
   about the cue: the RNN's fades within a few steps, the LSTM's cell state
   keeps it.
4. **Backward:** the gradient of the final error, drawn as its magnitude on
   a log scale, flows back through time. Through tanh and `Wh` it shrinks at
   every step of the RNN and reaches the cue vanishingly small (6×10⁻¹⁵ for
   the animated sequence); along the cell state it is multiplied only by the
   forget gate (7×10⁻³ reaches the cue).
5. **Training:** the loss curves. A third curve shows an LSTM whose forget
   gate starts at the usual bias of 1 (open to 0.73): it fails like the RNN.
   The gated cell remembers only when its gate starts open
   (`memory.FORGET_BIAS` = 2.5, open to 0.92).

The gap was chosen on evidence, not on a lucky seed (`build/sweep2.py` is
not kept; the numbers are, below). Mean loss of the last 200 steps after
8000 steps, four seeds; chance is 0.167:

| gap | RNN | LSTM, forget bias 1 | LSTM, forget bias 2.5 |
| --- | --- | --- | --- |
| 31 steps | 0.010, 0.001, 0.001, 0.000 | 0.165, 0.165, 0.170, 0.171 | 0.000 ×4 |
| 49 steps | 0.167, 0.168, 0.166, 0.104 | 0.167, 0.166, 0.171, 0.167 | 0.000 ×4 |

With Adam, the plain RNN learns gaps of 19 to 31 steps in most seeds, so
shorter gaps would not have separated the cells.

### Optimisation: a loss landscape and the optimisers on it (`landscape.py`, `optim_anim.py`, `mlp_runs.py`, `adamw_anim.py`, `schedule_anim.py`)

A loss surface you can walk on, made of real numbers: logistic regression with two weights and no intercept on all 12,665 MNIST training zeros and ones, `p(one) = σ(w₁·ink + w₂·width)`, with *ink* the mean pixel value and *width* the fraction of columns containing ink, both standardised. The mean log-loss is a function of two numbers, so it is evaluated on a 40 × 40 grid and drawn as a field of voxel columns, each as many voxels tall as the loss there (terraces of equal loss, like contour lines). It is a genuine, slightly curved bowl: minimum at w = (−4.29, −1.77), loss 0.137 (94.7% right), condition number 2.4. Multiplying ink by 10 turns the same data into a narrow valley (condition 42 at the minimum, 284 at the origin). Walkers float at their true loss above the surface and trail their path; a loss-against-step read-out runs beside them. Every walk is a recorded run (`landscape.descend`), cached with its settings; the gradients are checked against finite differences.

**`loss-landscape-descent`** (about 44 s): the landscape rises, the axes and the minimum are named, then full-batch gradient descent (η = 3) walks down from (2.5, −2.0): the gradient arrow and the step −η∂L/∂w at every move, the step shrinking from 2.06 to 0.018 as the floor flattens, the loss falling from 1.15 to 0.139 (0.002 above the minimum, within 0.01 of it from step 34).

**`drunken-walk`** (about 59 s): minibatch SGD from the same start, same η: walkers whose gradients come from 1, 8 and 64 random examples a step against the full-batch path. A minibatch gradient is the full gradient plus noise with covariance Σ/B, so the one-sigma ellipse of a step, η√(Σ/B), is drawn live around each walker: (0.25, 0.39) for B = 1, √8 tighter each time B grows eightfold, and the paths stagger by exactly that much. Then η = 8 (the first step is 8 long and the walker never settles: its steps stay twice as long around the minimum) and η = 0.3 (still at loss 0.22, 3.7 from the minimum, after 80 steps).

**`optimisers`** (about 54 s): SGD, momentum, RMSProp, Adam and AdamW from one start on the valley's steep wall (curvature 19 across, 0.07 along). SGD, at nine tenths of its stability limit 2/19, crosses the valley in one step and then creeps; momentum (β = 0.9) runs down to loss 0.21, up the far wall to 0.43, and arrives at step 109; RMSProp and Adam divide each weight's step by the running size of its gradient (Adam's moments drawn as bars), so both weights move about η at once: RMSProp arrives at step 34, AdamW at 56 (its decay holds it 0.5 short), Adam at 64; SGD is still 2.0 away after 150 steps.

**`adamw-vs-l2`** (about 52 s): Adam, Adam + L2 (λ = 10⁻³) and AdamW (wd = 0.1) train the MNIST MLP (784-64-32-10) for 3 epochs. The 64 × 32 middle matrix as sheets: the weights, Adam's √v per weight (a factor of thousands across the matrix), then each weight's *effective decay per step*. Adam + L2 divides the decay λw by √v like everything else, so it is 0.6× AdamW's uniform η·wd where the gradient is large and 1200× where it has died (orange/blue sheet); AdamW's sheet is flat. The weight norms over training show the consequence: ‖W₀‖ ends at 16.6, 9.3 and 15.3 and ‖W₁‖ at 9.6, 7.9 and 8.7 (Adam, Adam + L2, AdamW), at 96.0, 95.9 and 96.0% test accuracy.

**`learning-rate-schedule`** (about 56 s): constant, step decay (÷10 after each third), cosine with a 5% warmup and one cycle (up to 3·10⁻³ by 30%) train the MLP, their loss curves drawn in step with Adam walking the landscape under the same four schedules: one cycle 0.084 (97.0%), constant 0.117 (96.0%), cosine 0.153 (95.4%), step decay 0.179 (94.8%) — dividing by 10 after a third was too early — and the walkers finish in the same order. Then warmup: Adam's first step is exactly η in every weight whatever the gradient (m/√v = ±1 at step 1), so at η = 0.3 it leaps (−0.30, −0.30); warmed up it starts at (−0.04, −0.04). On the MLP at 3·10⁻³ the weights move 0.74 in norm in the first 20 steps without warmup and 0.02 with.

Tests check the model's gradients and Hessian against finite differences, the optimisers' update rules against hand-computed steps (AdamW with no decay is Adam bit for bit), the Σ/B law against 4000 sampled minibatch gradients, and every number the subtitles quote.

### Training deep networks (`deepnet.py`; `overfit_anim.py`, `init_anim.py`, `residual_anim.py`, `norm_anim.py`)

Four animations on one model: `deepnet.py` is a deep MLP in plain numpy with every knob that makes depth trainable — depth and width, tanh or ReLU, a normal init of chosen std, Xavier or He, residual connections every layer or every two, batch norm or layer norm (before the activation, with a learned scale and shift), dropout and weight decay. Its forward and backward passes are checked against finite differences option by option, including through both norms and the residual path, and its backward pass also returns the gradient reaching every hidden layer, which is what the bands draw. `train()` runs SGD or Adam on an MNIST subset and records, at a schedule of steps, the train/validation/test loss, each layer's activation std and gradient magnitude, one layer's minibatch values and the predicted field on a grid; runs are cached under all their settings.

**`overfitting`** (about 44 s): MNIST 3s and 8s as two numbers each, their coordinates on the first two principal components of the 60 training images (an SVD in numpy). A 2-128-128-2 tanh MLP trained by full-batch Adam (lr 0.001) overfits them: the decision field wraps itself around every training point while the training loss falls to 0.001 and the test loss (1,000 digits) climbs from 0.45 to 1.98, the accuracy from 79% to 71%. Then the four cures, each a recorded run: weight decay 0.01 (training loss stops at 0.30, test accuracy 80%), dropout 0.5 with the live mask of one example's 128 hidden units (78%), and early stopping on a 200-digit validation set, whose lowest loss is at step 24 (79%).

**`initialisation`** (about 39 s): a 20-layer tanh MLP (784 → 100 × 20 → 10) under three weight scales, each a panel of 20 columns. Forward, the std of each layer's activations grows up from a baseline (one decade per row): with std 0.01 they shrink tenfold a layer to 8×10⁻²¹; with std 1 tanh saturates at ±1; Xavier (std 0.1) keeps them near 0.2. Backward, the gradient reaching each layer grows down from the baseline: 2×10⁻²² at layer 1 for the small init, a million-fold *larger* for the large one (each saturated layer still multiplies by about 2), flat for Xavier. Layer 10's activations binned: a spike at 0, two spikes at ±1, a bell. Then 400 steps of SGD at learning rate 0.1 (SGD, because Adam divides each gradient by its own size and hides the effect): test accuracy 12%, 10% and 89%.

**`skip-connections`** (about 36 s), the depth-wise twin of `gated-memory`: two 30-layer ReLU MLPs (784 → 16 × 30 → 10) with the same Xavier weights, the second with a skip every two layers (`h → f(h) + h`). One digit through the untrained weights: the plain activations fade (std 0.23 at layer 1, 6×10⁻⁷ at layer 30), the residual ones grow to 2.9. The gradient flows back layer by layer on a log colour scale: reaching layer 1 it is 5×10⁻⁶ of what left layer 30 in the plain network, 13× along the skips. Training (1,500 SGD steps, lr 0.05): the plain network never leaves chance (loss ln 10), the residual one reaches 90%. The setting was chosen on evidence: with tanh and Xavier a plain 30×32 network keeps 0.19 of its gradient and half-learns; with He weights the plain ReLU network keeps 0.29 but still only reaches 50% in 1,500 steps; a skip every *layer* without normalisation overflows at this depth. Width 16, depth 30, Xavier: the contrast is real.

**`normalisation`** (about 38 s): layer 6 of a 12-layer tanh MLP (24 units) at step 500 of training, one real minibatch's pre-activations as a 32 × 24 sheet. Batch norm standardises down each column, over the 32 examples (the column's μ and σ appear under it); layer norm, on the same layer of the network trained with it, across each row, over the 24 units; then the learned γ (0.50 to 1.55) and β (−0.15 to 0.16) are applied, y = γ ẑ + β. The histograms before and after are live, and the normalisation drawn is recomputed from the recorded values and agrees with the network's own output. Training at learning rate 0.01 (Adam): the plain network's loss climbs back to chance, batch norm reaches 93% test accuracy, layer norm 75%; at 0.003 all three learn, so what normalisation buys is the higher rate.

Tests (`tests/test_deepnet.py`) check every option's gradients against finite differences and every number the subtitles state.

### Generalisation (`generalise.py`, `gen_common.py`; `ddescent_anim.py`, `memorise_anim.py`, `datasize_anim.py`, `augment_anim.py`, `minima_anim.py`)

Five animations on why a network works on data it has not seen, every one a real MNIST training of the deep MLP of `deepnet.py` (three options were added to its `train()`, each off by default and absent from the cache key when off: an augmentation hook that draws from its own seeded stream so the minibatches and initial weights are those of the plain run, the loss of every training example at every recorded step, and a cap on the probe's rows). `generalise.py` measures what is shown: label corruption, error rates and confusion matrices, a bilinear random affine transform of 28×28 images, the output's sensitivity to pixel noise, and slices of the loss between and around trained weights.

**`double-descent`** (about 47 s): 1,000 training images with 20% of the labels replaced by a wrong class, a 784-w-10 ReLU network for twelve widths from 2 to 200, each trained by Adam for 1,000 epochs so that every width from 8 up fits all 1,000 labels. The double descent is in the test loss: 2.4 at w = 3, a peak of 3.6 at w = 6 (the width that just about fits the labels is confidently wrong), then down to 1.5 at w = 200; the test error only falls, 65% → 45% → 24%, and the animation says so (a sweep over 100/300/1,000 epochs, Adam and SGD, 20% and 30% noise and 1,000/2,000 images never produced an error peak above 1.5 points, so none is claimed). The output's response to pixel noise peaks at the threshold (1.07 at w = 4) and is 0.28 for the widest fit: of all the interpolators, the wide one is the smoothest.

**`memorising-noise`** (about 45 s): 5,000 training images with 0%, 25% and 50% of the labels false; a 784-256-10 network trained 120 epochs reaches a training loss of 0.0001, 0.0075 and 0.0203 on all three, and 94%, 80% and 58% test accuracy. Every example's recorded loss, sorted clean | shuffled, as an image with one row per recorded step: the clean half is learned first (mean loss 0.40 at step 1,068 against 1.23), the shuffled half memorised last; the 50% run's validation loss bottoms out at step 498, where the test accuracy is 83% instead of 58%.

**`data-vs-model-size`** (about 45 s): learning curves of 784-8-10, 784-32-10 and 784-256-10 networks on 100 to 60,000 images, eighteen runs of the same 2,000 Adam steps. The small network goes from 25.7% to 8.7% error and its last sixfold of data buys 0.6 points; the large one from 21.5% to 2.8%, a third of the small one's mistakes; with 100 images the three are within 4 points. Then the first eight test digits each regime gets wrong.

**`augmentation`** (about 42 s): one training digit shifted (±3 px), rotated (±15°) and scaled (±15%) ten times by the generator training uses; a 500-image, ten-class 784-128-10 classifier trained 200 epochs with and without it, the live recorded minibatch as stored and as augmented. Without: training loss 0.0005, test loss rising from 0.38 to 0.48; with: 0.18. Test accuracy 86.8 → 94.0%, on shifted test digits 46.6 → 92.2%, on rotated 85.7 → 93.0%.

**`sharp-vs-flat-minima`** (about 37 s): the 784-128-10 network trained for the same 40 epochs with minibatches of 8 (50,000 steps, lr 0.001) and of 1,024 (400 steps, lr 0.01): test error 3.5% and 4.2%. The training loss on the plane through both minima as a voxel height-field; along the straight line between them the training loss climbs to 0.014 (the test loss runs from 0.27 to 0.20: the flat network is the more confident); along six random directions the loss rises by 0.01 within 29.5 units of the flat minimum and 14.6 of the sharp one; weight noise of norm 32 takes the test error to 5.0% and 7.9%.

Tests (`tests/test_generalisation.py`) check the model's gradients and the new options, every measurement helper, that the cache files store the settings the modules state, and every claim above as a relative claim.

### Regularisation (`regul.py`, `regul_ae.py`, `regul_plots.py`; `regul_geometry.py`, `regul_weights.py`, `regul_tools.py`, `regul_race.py`, `regul_ae_anim.py`)

Five animations on the penalties and constraints that stop a network memorising, ending with an autoencoder. `deepnet.py` gains an L1 penalty, either by its subgradient or as the proximal step after each SGD update (the soft threshold, which is what puts weights at exactly 0), a max-norm cap on each hidden unit's incoming weights, label smoothing, and snapshots of chosen weight matrices through training. Every option is off by default and absent from the cache key when off: a run of the old options trains bit for bit as before. `regul.py` holds the four studies: a logistic regression of 3s against 8s on 7 × 7 block averages fitted by FISTA along lasso and ridge paths, the 784-64-10 MLP's first layer under each penalty, a race of regularisers on 1,000 digits whose strengths are chosen on a validation set the test set never touches, and the autoencoder runs.

**`l1-vs-l2`** (about 54 s): a 3-against-8 classifier with two weights, w₆ and w₉, reading two 7 × 7 blocks of the digit (85.8% test accuracy with no penalty, at w₆ = −1.29, w₉ = 3.01). Its training loss over the two weights is drawn as contours, with the L1 diamond and the L2 circle. As λ grows the penalised optimum slides along its path, and the ball through it touches the contour through it. L1 meets the contour at the diamond's corner, so w₆ is exactly 0 from λ = 0.10 and w₉ from 0.34. L2 at λ = 60 still has w₆ = −5×10⁻⁴: shrunk, never zero. Then one weight: pulled towards v, L2's best w is v / (1 + λ) and L1's is the soft threshold; at λ = 1 the soft threshold puts 10 of the 16 fitted weights at exactly 0. Then the full 16-weight model's paths: lasso drops weights one by one (16, 14, 13, 12, … 2, 1, 0) and ridge shrinks all 16 together. At λ = 0.032 lasso keeps 9 for 92.9% (93.9% with all 16), and the 4 × 4 map shows which blocks of the digit it still reads.

**`l1-l2-weights`** (about 58 s): the 784-64-10 ReLU MLP trained by SGD (lr 0.1, batches of 64, 10 epochs of all 60,000 digits) from the same weights with no penalty, L2 (λ₂ = 2×10⁻³), L1 by its proximal step (λ₁ = 2×10⁻⁴), elastic net (λ₁ = 10⁻⁴, λ₂ = 10⁻³) and L1 by its slope λ·sign(w).

- **One weight:** from a never-lit corner pixel, so only the penalty moves it. It starts at 0.099 and plain keeps it. L2 halves it every 3,466 steps and never reaches 0. L1 walks it ηλ = 2×10⁻⁵ a step to exactly 0 at step 4,951.
- **Templates:** 36 of the 64 first-layer templates of four runs play through training, and L1 blanks the background. 67 pixels are never lit in the 60,000 digits: plain keeps their random weights forever, L2 leaves 15% of them, and L1 cuts 213 pixels out entirely (65 of the 67).
- **Histograms:** L1 puts 46% of the 50,176 weights at exactly 0 and elastic net 40%. L2 halves the bell's width and zeroes none.
- **The slope term:** the same L1 penalty as a slope term reaches the same accuracy and norm (96.5%, ‖W₀‖ 7.4) with no weight at exactly 0. 46% hop back and forth within ηλ of it.
- **Cost:** 97.5% plain, 96.4–96.8% with a penalty.

**`regularisers`** (about 57 s): the race's 784-256-256-10 network memorising 1,000 digits, with four more regularisers at their validation-chosen strengths, one step each and what each changes once trained.

- **Dropout, p = 0.5:** a training digit's 256 hidden units under three fresh masks, the kept units scaled by 2. Test accuracy ties plain (90.8% against 90.6%). With half the hidden units dropped at test time, plain falls to 75.6% and the dropout network only to 88.2%.
- **Input noise, σ = 0.8:** the network reads noisy test digits at 80.5% against plain's 55.9%, and clean ones at 92.8%.
- **Max-norm, c = 1.5:** 92% of layer 1's units sit at the cap, yet accuracy is 90.7%. It binds, but buys nothing here.
- **Label smoothing, ε = 0.2:** the target becomes 0.82 / 0.02. The median gap between the top logit and the rest shrinks from 22.2 to 3.2, under the 3.71 at which the softmax meets the target. Wrong answers are 0.29 sure instead of 0.91, at 92.0% accuracy.

**`regularisation-race`** (about 56 s): the same network trained by Adam (lr 0.001, 3,000 steps) on 1,000 digits, plain and with each regulariser at the strength that scored best on 2,000 validation digits.

- **Plain memorises:** training loss 1.7×10⁻⁵. Its test loss is lowest at step 100 (early stopping's pick, 0.36) and climbs to 0.56, while test accuracy only moves from 90.1% to 90.6%: it grows sure of its mistakes.
- **The winners:** input noise (σ 0.8) ends at 92.8% with test loss 0.24, and label smoothing (ε 0.2) at 92.0%. Label smoothing's training loss cannot fall below −log 0.82.
- **The ties:** L2, L1, dropout and max-norm are within 0.7 points of plain, the spread of 2,000 test digits. L2 and L1 still cut the test loss to 0.40 and 0.41.
- **The sweeps behind the choices:** training and validation accuracy against strength. L2 at λ 0.1 fits only 82.3% of the training digits. L1 at 10⁻² pushes every weight within 10⁻³ of 0 and falls to chance. Input noise rises from 91.6% to 93.3% and falls again at σ 1.3.

**`regularised-autoencoder`** (about 65 s): the finale. A tied-weight autoencoder 784 → 256 → 784 (a ReLU code; the decoder is Wᵀ, because with untied weights a penalty on the code is dodged by scale: the encoder shrinks and the decoder grows to match) trained for 10 epochs on 60,000 digits. It is trained plain and with four regularisers, each turned up until its clean test error is about 0.0075: L1 on the code (α = 1.1×10⁻⁴), weight decay (λ = 5×10⁻⁵), denoising (Gaussian noise σ 0.55 on the input) and the contractive penalty γ‖∂h/∂x‖² (γ = 2.5×10⁻⁵).

- **Features:** each model's 16 most-used features form through training. Plain's are speckled, L1's are whole strokes and digit prototypes, and weight decay's and the contractive's are smooth.
- **The code:** on one test 3, L1 leaves 58 of 256 units on (plain 216); over 2,000 test digits 19% are on against plain's 81%.
- **Noise:** with noise of std 0.4 on the input, the denoiser rebuilds digits at error 0.0109 against plain's 0.0191. On clean digits plain, which pays no penalty, is best (0.0016).
- **What each buys:** the final table has each regulariser winning its own column. Denoising is robust to noise and L1 gives a sparse code. Weight decay and the contractive penalty give the steadiest code: it moves 0.23 under the noise against plain's 0.56, with ‖∂h/∂x‖ 18.2 and 18.3 against 42.7.
- **Two that end alike:** for a ReLU code ∂h/∂x is the weights of the units that are on (98% of them here), so the contractive penalty is weight decay weighted by how often each unit is on, and the two end up alike.

Tests (`tests/test_regul.py`) check the new deepnet options and the autoencoder's four objectives against finite differences, the ReLU Jacobian against a finite-difference one, the soft threshold, max-norm and the unchanged cache keys of the old options, FISTA against the logistic optimum, the contour tracer, and every claim above as a relative claim.

### Activation functions (`actfun.py`; `actfun_race.py`, `actfun_gradients.py`, `actfun_sparsity.py`, `actfun_smoothness.py`, `actfun_depth.py`)

Five animations on what the activation function does to training, all on `deepnet.py`, which now knows sigmoid, tanh, ReLU, Leaky ReLU (slope 0.01), GELU (the tanh form, as GPT-2), SiLU and ELU, each gradient-checked, and can record how much of every layer is switched off. Every network is trained seven times under identical conditions — the same data order, batches, steps and optimiser — with Xavier weights for the squashing pair and He weights for the rectifiers.

**`activation-race`** (about 43 s): the seven functions drawn with their slopes, then 784 → 128 → 64 → 10 trained 1,500 Adam steps (lr 0.001, batches of 64) on 10,000 MNIST images, the training loss and test accuracy racing on a log time axis. Honest about ties: ReLU, Leaky ReLU, GELU, SiLU and ELU finish within 0.5% of each other (95.5–96.0%), tanh 95.3%, sigmoid 94.5% and much the slowest to start (71% at step 100 against 90%). The time per training step is measured, not estimated (interleaved blocks, the minimum kept): ReLU 7.5 ms, GELU 8.5, sigmoid 8.9 on this machine; the function alone costs 37 µs for ReLU and 645 for GELU on a 64 × 128 layer, so in a network this small the matrix products dominate.

**`activation-gradients`** (about 41 s): 784 → 64 × 8 → 10, the gradient reaching each layer as a 7 × 8 grid on a log colour scale, flowing back layer by layer. Untrained, sigmoid's shrinks about four-fold a layer (its slope is at most 0.25) to 4×10⁻⁵ of what left layer 8; ReLU's and Leaky ReLU's arrive at 0.91 of it; GELU's and SiLU's sit in between (0.14 and 0.09: their slope is ½ around 0, where He-initialised pre-activations sit); ELU's grows to 2.7×. Each slope is drawn over the real histogram of layer 4's pre-activations on 256 test images with the share on a flat part (|f′| < 0.05) read out. After 1,000 Adam steps no gradient shrinks on the way back any more, but the sigmoid network has reached 42% test accuracy against 94% for the others.

**`activation-sparsity`** (about 43 s): the trained race networks' layer 1 on 32 test images as a sheet, one network after another, with "off" defined per family and said on screen: ReLU leaves 35% of the layer exactly 0, Leaky ReLU 22% within 0.01 of 0, GELU 4%, SiLU and ELU 1%; tanh saturates 10%, sigmoid 38%. Bars follow the recorded run: ReLU starts 51% zeros (He weights), dips to 25%, settles at 36%. One test digit uses 146 of 192 hidden units under ReLU, 171 under Leaky ReLU, 187 under GELU, all 192 under tanh. Dead units, off on every one of the 2,000 test images: ReLU 3, every other network none — a slope that is never exactly 0 keeps every unit trainable.

**`activation-smoothness`** (about 44 s): the straight line between a 3 and an 8 in pixel space, sampled at 401 points. Every network's logit for "8" rises along it, but its slope is a staircase under ReLU — 50 steps, one per hidden unit switching on or off along the line (57 under Leaky ReLU), the largest jump 13% of the mean slope — and a curve under GELU, SiLU and tanh (under 1%). Then one minibatch's loss along a random weight direction (each layer moved by the same fraction of its size): at ±0.6 all seven slices are smooth bowls; zoomed sixty-fold the slope of ReLU's loss is a staircase (jumps 11%, Leaky ReLU 24%) while GELU's, SiLU's and tanh's stay curves (under 0.1%). That is the smoother ground a smooth activation gives the optimiser, and the wide view shows why ReLU trains fine anyway.

**`activation-depth`** (about 40 s): the 20-layer network of `initialisation` (784 → 100 × 20 → 10) under each activation with its matching init, no normalisation, 400 SGD steps at learning rate 0.1. The activation std per layer as a grid: tanh and ReLU hold their scale (0.13, 0.20 at layer 20), sigmoid's sits at 0.11 from the start (everything squashed towards ½), GELU's and SiLU's shrink by a third a layer to 4×10⁻⁴, ELU's grow to 1.7. Training: tanh 89% test accuracy, ReLU 73%, Leaky ReLU 57%; sigmoid, GELU and SiLU never leave chance; ELU's loss overflows at step 5. Inside, layer 10 after training: sigmoid's activations all ≈ ½, GELU's and SiLU's all ≈ 0, so nothing downstream can tell the digits apart — what `skip-connections` and `normalisation` were invented for.

Tests (`tests/test_activation.py`) check every activation's derivative and every network gradient against finite differences, the recorded runs against their settings, and each claim above as a relative claim.

### Capacity and bottlenecks (`capacity.py`; `bottleneck_anim.py`, `rank_anim.py`, `widthdepth_anim.py`)

Four animations on how much a layer can carry. `capacity.py` holds the models: the dense autoencoder of `autoencoder.py` trained at six latent widths, a classifier squeezed through a linear middle layer with its 2-D coordinates recorded at geometric snapshots, the SVD of the trained MNIST MLP's matrices, and three ReLU MLPs of one parameter budget trained alike by `deepnet.train` with the wall-clock time of a training step measured, not derived.

**`bottleneck-sweep`** (about 43 s): the dense autoencoder 784 → 256 → 64 → L → 64 → 256 → 784 trained six times alike (2 epochs, batch 256) with L = 1, 2, 3, 8, 32 and 256. Four test digits rebuilt by each width side by side; the test error against L on a log axis, 0.056, 0.047, 0.039, 0.025, 0.015, 0.014 — steep to 8 and flat past 32, where 256 numbers gain 8% over 32; then the codes of 600 test digits for L = 1 (a line: where codes coincide they stack into a histogram), 2 (a plane) and 3 (a cube), coloured by label, with a nearest-centroid read-out of what the code keeps about the digit: 36%, 46%, 67% (77% at L = 256).

**`information-bottleneck`** (about 42 s): 784 → 64 → 2 → 10 against 784 → 64 → 64 → 10, the middle layer linear (a tanh pair saturates at the four corners and learns 42%, a ReLU pair switches half its values off and learns 65%; the linear pair reaches 90%). The 600 test digits' two coordinates move and sort themselves into wedges round the origin as the run plays back from its snapshots, the two networks' layers on one digit live, both accuracy curves (89.5% through two numbers, 96.3% through 64), and the squeezed network's confusion matrix on the 10,000 test digits: 4 ↔ 9 (171), 5 ↔ 8 (147), 5 ↔ 6 (100); 1,049 wrong against 373.

**`rank-of-a-layer`** (about 42 s): W₁ (784 × 64, sixteen of its columns drawn as 28×28 weight images) and W₂ (64 × 32) of the trained MLP under W = U Σ Vᵀ. The singular values as bars (4.91 … 1.16; the top 8 hold 39% of ‖W₁‖²), then both matrices rebuilt from their top 1, 2, 4, 8, 16 and 32 components, the images re-forming: test accuracy 7.6%, 22%, 26%, 62%, 89%, 96.3% against 96.8% whole, with 2%, 4%, 7%, 15%, 29%, 56% of the parameters (two thin factors, k(m + n) numbers). The fact LoRA relies on: a rank-4 to 16 update added to a frozen matrix carries most of what fine-tuning changes (`gpt2-lora`).

**`width-vs-depth`** (about 41 s): about 50,000 parameters as 784 → 64 → 10 (50,890), 784 → 52 × 4 → 10 (49,618) or 784 → 37 × 16 → 10 (50,515), ReLU, He weights, 1,500 Adam steps on 20,000 digits. One digit's activations through the untrained weights as bands (std 0.12–0.29 at every layer), the gradient flowing back on a log colour scale (it reaches layer 1 of the 16-layer network at 0.68× what left layer 16: He keeps it where `skip-connections`' Xavier weights lose it), then training: 95.5%, 95.5% and 94.8%, the bands showing each layer's activation std growing through the deep network, and a training step measured at 6.8, 8.7 and 16.5 ms (sixteen small matrix products cost more than one). A tie within a point: which wins depends on the data, the budget and the recipe; on MNIST width is enough.

Tests (`tests/test_capacity.py`) check the squeezed classifier's gradients against finite differences, the SVD identities, every recorded run against its settings, and the claims as relative ones (the error falls then flattens; two units lose to 64 by a margin; rank 32 loses little and rank 4 most; the deep network is slower a step).

### Trimming: pruning and distillation (`prune.py`, `prune_anim.py`)

Four animations on making a trained network smaller, on the MNIST MLP and CNN of `mnist.py`. `prune.py` holds the machinery: global magnitude masks (the threshold is the k-th smallest |w|, so a mask drops exactly the count asked for), retraining with a mask held (pruned weights get no gradient and are zeroed again after every Adam step, so they stay exactly 0), random masks of a given per-layer density, structured masks that silence a hidden unit or a conv filter, activation energies (mean squared activation over the test set) and the distillation loss (1−α)·CE + α·T²·KL(teacher‖student) with its gradient (1−α)(p−y) + αT(p_T−q_T) at the logits, checked against finite differences. Every run is seeded and cached under its settings.

**`magnitude-pruning`** (about 44 s): the trained 784-64-32-10 MLP (52,544 weights, 96.8%) as sheets — six of layer 1's templates, the 64×32 and 32×10 matrices — beside the histogram of |w|. A threshold sweeps up the histogram and every weight below it goes dark. At 50, 80, 90, 95 and 99% pruned the test accuracy without retraining is 96.3, 93.2, 67.3, 33.9 and 8.9%; after 400 Adam steps with the mask held it is 96.9, 96.7, 95.6, 92.6 and 38.6% (the sheets show the 90% network recovering); pruning iteratively — prune, fine-tune, prune further — gives 97.0, 96.8, 96.1, 94.0 and 64.6%, and what is left at 99% is 525 weights, almost all in layers 2 and 3.

**`lottery-ticket`** (about 43 s): the 20%-dense mask from pruning the trained network (17.9% of layer 1, 63% of layer 2, 74% of layer 3) laid over that network's *original* initialisation, and the sparse network retrained for 1,500 steps against a random mask of the same per-layer density and the dense network, all from the same initial weights on the same minibatches. The ticket reaches 96.6% (seed 1: 96.4%), the random mask 94.0% (94.1%), the dense network 96.3% (96.3%), and the ticket is ahead from the first 25 steps (82% against 65% dense and 40% random). At this scale the ticket wins, in both seeds.

**`structured-pruning`** (about 44 s): the CNN's four filters ranked by activation energy on the test set (0.12, 0.27, 0.16, 0.16) and removed one at a time, least active first: the kernel and its feature map go dark and the accuracy falls to 58, 53 and 64% at once — the dense layer was reading all four — but recovers to 95, 95 and 91% with 300 fine-tune steps after each cut. Then the MLP's 64 hidden units (energies 0.05 to 15.3), the layer-2 matrix drawn with one row per unit, removed one at a time: with 100 fine-tune steps after each cut the network barely notices until past 40; with half gone, 2 filters and 32 units give 95.2% and 95.9% after fine-tuning (53% and 54% without).

**`distillation`** (about 42 s): a 784-256-256-10 teacher (269,322 weights, 97.8%) looks at a 3 that looks like an 8: its softmax puts 0.88 on 3 and 0.12 on 8; divided by T = 4 the logits give 0.36, 0.22 and 0.08 on 3, 8 and 2 — the dark knowledge. A 784-32-10 student (25,450 weights) trained on a 3,000-image transfer set reaches 91.5% on hard labels, 93.8% on the soft targets and 93.7% on both. On all 60,000 images the hard labels alone do as well (94.8% against 94.0%): soft targets pay when data is scarce, and the animation says so.

Tests (`tests/test_prune.py`) check the masks and energies as computed, that retraining keeps pruned weights exactly zero, the distillation gradient against finite differences, and every claim above as a relative claim.

### Fine-tuning and transfer (`transfer.py`, `transfer_anim.py`)

Split MNIST: the digits 0–4 (30,596 training images) are the old task, 5–9 the new one, with a few-shot set of 100 images a class and a rehearsal buffer of 100 old images a class drawn once with a seed. The network is a CNN with two convolutional layers, conv 3×3×8 → ReLU → pool → conv 3×3×16 → ReLU → pool → dense 400 → 10 (two, because that is where transfer is honest: with the one-conv CNN's 36 filter weights frozen and scratch tie within a point, and a frozen dense hidden layer trained on 0–4 loses 13 points to scratch). Pretrained for one epoch on 0–4 with a 10-way head it gets 99.3% of those digits and has only ever pushed the other five outputs down. Adam here has a learning rate per layer; a rate of 0 freezes the layer, bit for bit. The gradients are checked against finite differences; every run is recorded and cached under its settings.

**`transfer-learning`** (about 49 s): the first-layer filters settle during pretraining, then the digits 5–9 are learned from 100 images a class in three ways, each row of filters live and each accuracy curve drawn as its run plays: from scratch (a fresh network), with both conv layers frozen and a new 5-way head, and fine-tuned with the pretrained layers at η/10 (the filters change by under 2%). After 200 steps of batch 50: frozen 95.2%, fine-tuned 95.1%, from scratch 93.7% on the 4,861 test images of 5–9: the strokes learned on 0–4 carry over, and 500 images are too few to learn them as well.

**`catastrophic-forgetting`** (about 50 s): a bar per digit (test accuracy) while the pretrained network is fine-tuned on 5–9 (batch 64, Adam 2·10⁻³), the minibatch's own images underneath. Plain: 0–4 falls from 99.3% to 34% by step 20 and 5.1% by step 200 while 5–9 reaches 95.9%; every minibatch says "not a 0–4" and nothing says otherwise. With rehearsal, 16 of every 64 images from the buffer of 500 old digits: 0–4 94.0%, 5–9 95.3%. With lower rates, the conv layers at η/50 and the head at η/5 (the head must slow down too; at η it forgets on its own): 0–4 64.7%, 5–9 81.5%, what is kept paid for in slower learning.

**`learning-rate-finder-and-restarts`** (about 51 s): the range test on the MLP (784-64-32-10, Adam): 300 steps while η rises 4% a step from 10⁻⁵ to 1, the minibatch loss drawn live and smoothed over 21 steps. The knee, the steepest fall against log η, is at η = 8.4×10⁻⁴ (the usual 10⁻³); the lowest loss 0.38 at 0.023; above 0.19 the loss is back above where it started. Then three epochs with a cosine from the knee rate restarted every epoch, against one cosine: at this rate a restart is gentle (the loss pauses at 0.27, rises from 0.19 to 0.21 at the second) and leaves a model at every cycle's end, 92.8 → 94.3 → 95.2% against 95.1%. Beside it, minibatch SGD (B = 4, peak η = 8) walks the two-weight landscape under the same shape of schedule: only noise can take a walker out of a minimum (full-batch descent and Adam sit still when the rate jumps), so each restart throws it 2.7 units up the bowl (loss 0.137 → 0.23) and the falling rate brings it back.

Tests (`tests/test_transfer.py`) check the split, the gradients, exact freezing, the rehearsal mix, the finder's rules and every claim above as a relative claim.

### LoRA: fine-tuning with two thin matrices (`lora.py`, `lora_anim.py`, `gpt2-lora`)

One flat diagram, about 57 s, on the real GPT-2 small. A weight matrix stays frozen and learns a rank-r correction beside it: h = x·W + (α/r)·(x·A)·B with A (768 × 4) random and B (4 × 768) zero, so at step 0 the adapted model *is* the base model. Here the adapters sit on the query and value projections of all 12 layers (rank 4, α = 8): 147,456 trainable numbers against 124,439,808, 0.118%, drawn to scale — the two thin matrices are hairlines beside the 768 × 768 square and a sliver at the left end of a bar for the whole model.

The task is one new fact, "The capital of Freedonia is Avalon.", phrased six ways, trained with ten true capitals in the same form ("The capital of Japan is Tokyo." …) — without them forty steps teach the model that every capital is Avalon. Forty Adam steps (lr 5·10⁻⁴) on the plain language-model loss over the sixteen sentences: the loss falls 4.43 → 1.13 and p(Avalon) after "The capital of Freedonia is" rises from 0.00003 to 0.98 (×33,400), the top token from step 31. On screen B fills in from zero, the update (α/r)·A·B appears as a sheet whose singular values are 1.31, 0.75, 0.37, 0.28 and then zero (rank 4), the next-token bars move step by step. What did not change, measured: after "The capital of Germany is Berlin. The capital of France is" Paris stays on top, 0.72 → 0.62 (Berlin rises to 0.25). What did: "The cat sat on the" now says "throne" (0.21, was "floor" 0.08), Madrid for Spain goes 0.11 → 0.64 and the bare "The capital of France is" from "the" to "Paris" 0.85 — a fine-tune never moves only the fact you aimed at. Finally the merge: W′ = W + (α/r)·A·B is one 768 × 768 matrix again, the same logits to 5·10⁻⁵, nothing extra at inference.

- **Gradients:** `lora.py` is GPT-2's forward pass over a right-padded batch (one GEMM per weight) keeping what the backward needs, and a hand-written reverse pass through the unembedding, both LayerNorms of each layer, the GELU MLP, softmax attention with its causal mask, and the residual stream, down to A and B. With `frozen=True` it also returns every base weight's gradient; tests check all of them against finite differences on a 2-layer, 8-wide random model built with the same code, and the real adapters' gradients on the real loss in float64.
- **Checks:** zero adapters leave the logits bit-identical; the merged weights reproduce the adapted output; the parameter count; every claim above as a relative claim.

### Quantization and precision (`quant.py`; `quant_anim.py`, `precision_anim.py`)

Three animations on fewer bits per number, all on real models: the MNIST MLP of `mnist.py` (784-64-32-10, 96.8% test accuracy) and the deep tanh MLP of `deepnet.py`. `quant.py` holds the numerics: uniform quantizers (symmetric or asymmetric, one scale per tensor or per channel), min-max and MSE-clip calibration, post-training quantization of every weight matrix, quantization-aware fine-tuning with fake quantization in the forward pass and the clipped straight-through estimator in the backward, float16 through numpy and bfloat16 emulated by rounding float32 to a 7-bit mantissa (round half to even), training in a format with float32 accumulation (as tensor cores do) with or without float32 master weights, and loss scaling. Every run is seeded and cached under its settings.

**`quantization`** (about 56 s): W₁ (64 × 32) as a sheet with the histogram of its 2,048 weights. int8: 255 levels, one scale per matrix from its largest weight (step 0.0058), the weights snap row by row, the error sheet W − Q(W) is invisible (at most 0.003), accuracy 96.8% — unchanged. int4: 15 levels, step 0.105, 96.3%. 2 bits: three levels −s, 0, +s with s the largest weight: 90% of W₁ rounds to 0 and the network falls to chance, 13.1%; one scale per output channel, 29.2%; a calibrated clip (the fraction of the range that minimises the squared error, per channel, 0.3–0.8 of the max), 82.4%. Then the bytes: 211 kB in float32, 53, 27 and 14 kB (weights packed, biases and scales float32); with int8 activations too, 96.8%; int4 weights and activations, 96.1%.

**`quantization-aware-training`** (about 50 s): Q(x) = s·round(x/s) is a staircase whose derivative is 0 almost everywhere; the straight-through estimator uses 1 inside the clipping range. The network computes with Q(W) (2 bits, s = 0.264 for W₁, calibrated: 76.4%); ∂L/∂Q(W) on a real minibatch of 128 passes straight through to the float weights (7% of them saturate at ±s and get 0), Adam moves them a little and the ones that cross ±s/2 change level: 5, 4, 5, 6, 3 of 2,048 in the first five steps. Two epochs of this take 2 bits from 76.4% to 96.1% (int4 96.4% → 96.4%, float32 96.8%). The float weights do not gather at the levels: 27% of them end within a tenth of a step of a boundary, oscillating across it, against 17% near a level. Tests check the STE gradient against its definition — the rounded loss is flat, so finite differences cannot — and the bias gradients against finite differences.

**`mixed-precision`** (about 56 s): the bit layouts 1|8|23, 1|5|10 and 1|8|7 holding the trained weight W₁[0,0] = 0.4254688 (float16 keeps 0.4255371, bfloat16 0.4257812); the representable numbers within 1% of 1, 10⁻³ and 10⁻⁵ as ticks (float16 steps by the value/1024 until its subnormals, where the step freezes at 6×10⁻⁸; bfloat16 by the value/128 everywhere); a real gradient of layer 11 of a 20-layer tanh MLP with weights of std 0.05 (RMS 2.7×10⁻⁸): cast to float16, 7,938 of its 10,000 values become 0 and the rest are subnormal with a median error of 40%, while bfloat16 keeps them all; loss scaling by 2¹⁶ before the cast and back in float32 loses none (median error 1.7×10⁻⁴). Then the MLP trained two epochs in each format: float32 95.6%, bfloat16 alone 95.0%, float16 with float32 master weights 95.6%; float16 alone diverges at its first update (Adam's ε = 10⁻⁸ and the small squared gradients are 0 in float16).

Tests (`tests/test_quant.py`) check the quantizers and calibration, the STE, bfloat16 rounding against an integer reference, the underflow example and every number above as a relative claim.

### A strided-convolution classifier (`strided.py`, `strided-cnn-classifier`)

Stride is the step size: a kernel that moves 2 pixels at a time reads every
other window, so each channel shrinks to about a quarter, and no pooling
layers are needed. The model is real (numpy, 97.2% on the MNIST test set):

| layer | output | values per channel |
| --- | --- | --- |
| input | 28×28×1 | 784 |
| conv 3×3, stride 2, 4 filters, ReLU | 13×13×4 | 169 |
| conv 3×3, stride 2, 8 filters, ReLU | 6×6×8 | 36 |
| dense 288 → 10, softmax | 10 probabilities, one-hot prediction | |

It is drawn in the original conv2d style:

- The white frame visibly hops two pixels per step across the digit.
- Kernel weights are orange or blue by sign, and after ReLU only values
  > 0 light up.
- Each layer's kernels arrive when the layer starts and retire when it is
  done.
- A dense head (`conv.HeadSpec`) shows the logits, the softmax bars and the
  one-hot prediction.

`--digit N` picks the MNIST test image; the tests check that the
animation's activations and logits are the trained model's, to machine
precision.

### Evaluating a loaded network (`model_io.py`, `dnn.py`, `dnn-eval`)

```bash
python3 -m voxanim render dnn-eval --digit 8                  # the bundled model
python3 -m voxanim render dnn-eval --model my_net.pt --digit 3 --preprocess standard
python3 -m voxanim master dnn-eval --model my_net.npz --digit 3   # 8K HDR master
```

It loads any fully-connected ReLU network on 28×28 inputs and evaluates one
MNIST test digit.

- **Layout:** the image, one sheet per hidden layer (a width of 100
  becomes a 10×10 sheet), then the softmax probabilities and the one-hot
  prediction.
- **Each layer:**
  1. One neuron up close: its weights, times the inputs (zero inputs add
     nothing), and the sum.
  2. The whole layer filling in.
  3. ReLU: only values > 0 light up.
- **Bundled model:** `models/mnist_dnn_100x3.npz`, 784-100-100-100-10,
  97.6% test accuracy.

**Model files.**

- **Formats:**
  - `.npz`: `W0, b0, W1, b1 ...` (`model_io.save_dense`), or any names.
  - PyTorch `.pt` / `.pth`: a state_dict or `nn.Module`; needs torch.
  - Keras `.h5` / `.keras`: needs h5py.
  - `.json`: `{"layers": [{"W": ..., "b": ...}]}`.
- **Pairing:** weights and biases are paired per layer in name order.
- **Orientation:** PyTorch's (out, in) vs Keras' (in, out) is detected
  once per file from the first non-square layer, so square hidden layers
  are never transposed by mistake.
- **`--preprocess`:** says how the network expects pixels:
  - `unit`: 0..1 (the default)
  - `standard`: (x − 0.1307)/0.3081, the usual PyTorch transform
  - `centered`: 2x − 1
  - `raw`: 0..255

  An `.npz` can record its own.

### Gaussian processes (`gp.py`, `gp_anim.py`, files named `gp-*`)

Five 1-D regression animations on one GP library in plain numpy (`gp.py`:
kernels with their derivatives, the exact posterior through one Cholesky
factor, the log marginal likelihood and its gradient, expected improvement)
and one shared plot (`gp_anim.py`: the ±2σ band as four stacked translucent
layers, sample paths, data as disks, kernel matrices as sheets of voxels).
Every frame is an exact posterior of what is on screen: a new observation
enters with a huge observation variance that shrinks to its real value, and
a change of kernel morphs through the kernel (1 − u)k_old + u k_new, so the
band never cross-fades.

**`gp-prior`** (`gp_prior.py`, about 52 s): the RBF kernel k(x, x′) as a
bump, the kernel matrix over 24 grid points filling row by row as x′
sweeps, three seeded sample paths, the mean and the ±2σ band, then the
lengthscale (1 → 0.3 → 3 → 1) and the variance (1 → 2.25 → 0.25 → 1)
moving with the samples and the matrix updating together.

**`gp-posterior`** (`gp_posterior.py`, about 37 s): six noise-free
observations of f(x) = sin x + 0.3x − 1.5 arrive one at a time; the mean
passes through each and the band closes there. The joint covariance
[[K, k*], [k*ᵀ, k**]] is one sheet whose blocks pop out as the pieces of
μ* = k*ᵀK⁻¹y and σ*² = k** − k*ᵀK⁻¹k* are underlined. Then the same data
with noise σₙ = 0.3: the band stays open and the mean need not pass through.

**`gp-kernels`** (`gp_kernels.py`, about 46 s): 7 noisy points on [0.8, 5.4]
under RBF, Matérn 3/2, periodic (period 4 assumed), linear and
periodic + RBF, each with its variance and lengthscale fitted by gradient
ascent on log p(y). Most of the plot is extrapolation: what each kernel
assumes. log p(y) ranks them: periodic + RBF −0.68, RBF −1.62, Matérn
−2.48, periodic −6.23, linear −20.8.

**`gp-hyperparameters`** (`gp_hyper.py`, about 38 s): log p(y) over
(lengthscale, noise) as a landscape of voxel columns; gradient ascent from a
too-smooth start (ℓ = 6, σₙ = 1.5) to the top (ℓ = 1.82, σₙ = 0.24), the
posterior changing in step, the gradient arrow on the surface, and the
data-fit and complexity terms as bars: a big noise makes the fit term easy
and the complexity term pays for it, so the optimum is a trade-off
(−8.9 = −6.5 + 8.6 − 11.0).

**`gp-bayesian-optimisation`** (`gp_bayesopt.py`, about 47 s): a hidden
two-hill function, two evaluations, then eight rounds of expected
improvement (drawn under the plot, its maximum marked); each pick flies up
to the function and the posterior updates. Five rounds explore (more than a
lengthscale from every evaluation), three exploit; the best evaluation,
g(7.2) = 2.08, is the true maximum, revealed at the end.

Tests check every kernel derivative and the marginal-likelihood gradient
against finite differences, the posterior against the direct inverse,
expected improvement against Monte Carlo, and each claim the subtitles make.

### Autoencoders with a 2-D or 3-D latent space (`autoencoder.py`, `ae_anim.py`)

| name | model | test MSE |
| --- | --- | --- |
| `autoencoder-dense-3d` | 784-256-64-**3**-64-256-784 | 0.032 |
| `autoencoder-dense-2d` | 784-256-64-**2**-64-256-784 | 0.039 |
| `autoencoder-conv-3d` | 28×28 → conv s2 → 14×14×8 → conv s2 → 7×7×16 → **3** → mirror (upsample ×2 + conv) | 0.044 |
| `autoencoder-conv-2d` | the same with a **2**-D code | 0.053 |

All four use ReLU inside, a linear code and a sigmoid output, and are
trained with MSE (cached in `build/.cache`).

**Layout:**

- The encoder → code → decoder chain runs along the bottom.
- The latent space sits above the code: a cube (3-D) or a flat plane
  facing the viewer (2-D).
- It holds 600 encoded test digits, coloured by their label with a
  ten-colour categorical palette. The autoencoder never sees labels; the
  clusters form anyway.

**What happens:**

- Three digits (a 0, a 1 and a 2; `--digit N` swaps the first) go through
  in turn, and each code flies to its point in the latent space.
- A walk through the latent space follows: the decoder redraws the output
  at every point between the digits.
- In the conv version each tensor shows its channels as separate slabs
  ("14×14, 8 channels").
- The axes are scaled robustly (median-centred, tails compressed), so
  clusters use the space.

**`autoencoder-training`** (`ae_training.py`) shows the 2-D dense model
learning. `autoencoder.training_run` trains it again with exactly the same
draws, snapshotting the 600 test codes, the test loss and three digits'
activations at steps 0, 1, 2, ... thinning out geometrically to the last
(94 snapshots; the final weights are the cached model, byte for byte). A
digit goes through the untrained network first, then the run plays on a log
time axis: the codes drift and sort themselves (the box follows the cloud,
its axis numbers show the scale growing fivefold), the three digits leave
trails, the chain shows the live activations and the output sharpening, and
a loss curve draws itself.

### VAE, U-Net, DEC and IDEC (`deep_models.py`)

All four are dense, trained in numpy on MNIST, and gradient-checked in the
tests.

- **`vae-dense-3d`, `vae-dense-2d`** (`vae_anim.py`)
  - **Model:** 784-256-64 → μ, log σ² (3 or 2 dims) → z = μ + σ·ε → 64-256-784.
  - **Loss:** binary cross-entropy + KL(N(μ, σ²) ‖ N(0, I)).
  - **Same stage as the autoencoder, plus:** each digit shows its mean μ, a
    cloud of the codes it could be (its σ), and the sampled z. The latent
    axes are the prior's own range, −3…3. The ending samples new points
    from the prior and decodes them: digits that were never in the data.
- **`unet-dense`** (`unet_anim.py`)
  - **Task:** a denoiser. A noisy digit (Gaussian noise, σ 0.45) goes in
    and the clean digit comes out.
  - **Model:** 784-256-64-16-64-256-784, with skip connections that
    concatenate e2 into the d2 input and e1 into the d1 input.
  - **Layout:** drawn as a U. The copies slide across the skip bridges and
    sit beside the decoder layers they join.
  - **Ending:** an identical network trained without skips denoises the
    same digit, with both test errors shown.
- **`dec-dense`, `idec-dense`** (`dec_anim.py`)
  - **Starting point:** the 3-D dense autoencoder's codes; k-means places
    10 centres.
  - **Training:** minimises KL(P ‖ Q), where q is the Student-t soft
    assignment and p = q²/f is the sharpened target. The latent cube plays
    back real snapshots: codes pulling into clusters, centres moving,
    accuracy rising, and one digit's q and p bars side by side.
  - **DEC vs IDEC:** DEC drops the decoder, so its reconstructions degrade
    as the codes drift. IDEC adds the reconstruction loss (γ = 0.1), so
    they stay sharp.

### Diffusion, PCA and k-means (`diffusion.py`, `diffusion_anim.py`, `pca_kmeans.py`, `pca_anim.py`)

Two more ways of modelling digits without labels, on the autoencoders' stage.

- **`diffusion`**: a small denoising diffusion model (DDPM) in plain numpy.
  - **Forward process:** 200 steps of a random walk in pixel space, x_t = √(1−β_t) x_{t−1} + √β_t z, on the cosine schedule of Nichol & Dhariwal, which ends at ᾱ_T = 6·10⁻⁸. Four test digits dissolve step by step while the signal level √ᾱ_t and the noise level √(1−ᾱ_t) are drawn alongside.
  - **Model:** a dense denoiser 784-512-512-784 (SiLU, sinusoidal embedding of t) that predicts the noise ε, plus a skip connection with a learned per-step gain, ε̂ = g(t)·x_t + net(x_t, t). The skip is essential: white noise has no structure, so a 512-wide bottleneck alone cannot pass it through (loss 0.35 even where ε is simply x_t); with it the hidden layers only have to subtract the digit, the job a U-Net's skips do. Loss = mean (ε̂ − ε)². The second phase shows one digit at t = 60, the noise added, the prediction and their difference (0.155 on that image).
  - **Reverse process:** from fixed-seed noise, 200 steps back; at each step the estimate of the clean image (x_t − √(1−ᾱ_t) ε̂)/√ᾱ_t, clipped, sharpens while x_{t−1} is drawn from q(x_{t−1} | x_t, x̂₀). Four chains in parallel.
  - **Training:** Adam, lr 10⁻³, batch 512, 20 epochs (2,360 steps, about five minutes); loss 0.69 → 0.16, test 0.16. Honestly: the samples are digit-like and a classifier reads them confidently, but they are grainy and not every one is a clean digit — a small model, briefly trained.
- **`pca-and-kmeans`**: principal component analysis of the training images (mean, covariance, eigendecomposition; checked against the SVD) and k-means on the first two components.
  - **PCA:** the mean digit; the first seven components as diverging images with a bar for the variance each explains (PC 1 9.7 %, PC 2 7.1 %; the first 10 carry 49 %, the first 50 82 %); the 600 test digits of the latent clouds on the PC 1 / PC 2 plane, coloured by label, on the same plane conventions as `autoencoder-dense-2d`; a 3 rebuilt from 2, 10 and 50 components (errors 0.100, 0.067, 0.023).
  - **k-means (k = 10)** on the two coordinates: ten random points become the centroids, then assignment (each point takes its cluster's colour) and update (centroids walk to their means) repeat until no point changes cluster: 16 iterations, inertia 1201 → 584, read out at every step while a clusters-by-label grid fills in. Matched one-to-one, the clusters agree with the labels 42 % of the time: two numbers per digit are not enough.

### A 2-layer RNN training in batches (`rnn_batches.py`, `rnn-batch-training`)

- **Layout:** the model (x → h1 → h2 → y, with recurrent weights Wh1 and
  Wh2 on the loops) stays fixed in the centre. A batch of 4 sine sequences
  waits on the left, and predictions build up against white targets on
  the right.
- **Colour:** every batch has its own colour, and each sequence in it a
  shade of that colour.
- **Per batch:**
  1. The sequences go through one time step at a time.
  2. Gradients are averaged over the batch.
  3. One weight update.
  4. A coloured point joins the log-log loss curve.
- **Timeline:** batches 1-3 in full, then a fast-forward through about 400
  batches, then the last batch, where the predictions follow the waves.
- **Model:** a real numpy model (`sequence.rnn2_training`, Adam,
  lr 0.004). The loss falls from 0.20 to about 0.003.

### Attention between two sequences (`seq2seq.py`, `seq2seq_anim.py`, `seq2seq-attention`)

A GRU encoder–decoder with additive (Bahdanau) attention learns to reverse
six digits: it reads `3 7 2 9 4 1` and says `1 4 9 2 7 3`. Every output
copies exactly one input, so the right alignment is known (the
anti-diagonal) and the one the model learns can be checked against it.
The picture is a table under the encoder, one row per decoder step:

```
input x            3 7 2 9 4 1
encoder h          12 × 6 band, one column per token

context c  |  alignment α  |  decoder s  |  output
```

The alignment's columns sit exactly under the encoder's, so a row reads
"how much of each input this step looked at".

1. **The task.**
2. **Encode:** the GRU reads the tokens left to right; the states appear
   column by column.
3. **Decode:** at each step the previous decoder state queries the encoder
   states, the row of attention weights appears, particles fly from the
   input tokens to the context weighted by those weights (the context *is*
   that mix), the decoder state updates and the output digit appears.
4. **Alignment:** the finished matrix is the anti-diagonal; 0.85 of each
   row's weight is on the token it copies.
5. **Training:** the recorded run (`seq2seq.train` keeps the example's
   alignment and output after every one of the 3000 Adam steps, and the
   held-out accuracy every 25) replays on a log time axis: the matrix is
   flat and the output wrong before training (10% of tokens right,
   chance), the anti-diagonal forms by step 300 (99%), 100% from step 500.

The scores (keys) come from the encoder states, which know *where* they
are; the context (values) is a mix of the input *embeddings*. With the
states as values, as in Bahdanau et al., the model also learns the task but
reads each token from the state *after* it (that state still remembers the
token, and the decoder has just emitted the one it holds), so the alignment
lands one column off the anti-diagonal in 6 seeds of 6; with the embeddings
it lands on it (3 of 3). The decoder's first state is zero, so attention is
its only route to the input. `tests/test_seq2seq.py` checks every gradient
against finite differences, the accuracy and the alignment.

### Inside GPT-2 (`gpt2.py`, `gpt2_anim.py`, `gpt2-attention`)

One continuous camera move, about 85 s, through the real GPT-2 small
(124M) reading "One, two, three,":

1. **Tokens:** each token's row of the 50,257 × 768 embedding table, plus
   its position's row, gives the residual stream (6 × 768).
2. **Q K V:** fly into layer 5, head 12. Q = LN(x) · W_Q is built one row at
   a time, with W_Q (768 × 64) above Q. K and V follow.
3. **Flip:** K turns over its diagonal in 3-D and becomes Kᵀ.
4. **Scores:** the first three scores are worked in full. A row of Q turns
   to lie beside a column of Kᵀ, they multiply element by element, and the
   64 products collapse into one cell (÷ √64). The rest fill in quickly,
   then the causal mask.
5. **Softmax** along each row. This head is GPT-2's "previous-token head",
   so each row puts all its weight on the token before.
6. **A · V:** each output row is a weighted mix of the value rows.
7. **Heads:** zoom out to all 12 heads of the layer. Their outputs fly into
   the concatenation, then W_O, the add to the stream, and the MLP
   (3072 wide, only values > 0 light up).
8. **Layers:** a tracking shot follows a pulse of information along the
   stream through all 12 layers (144 heads), then a wide shot of the
   whole model.
9. **Output:** final LayerNorm, × W_Eᵀ, 50,257 logits, and the top five
   next tokens: " four" 46%.

- **Model:** `gpt2.py` runs GPT-2 in plain numpy, with no PyTorch: a
  `.safetensors` reader, the byte-level BPE tokenizer and the forward pass.
  The first use downloads OpenAI's released weights (MIT licence,
  `openai-community/gpt2` on the Hugging Face hub, 550 MB) into
  `build/.cache/gpt2`.
- **Checks:** tests confirm the tokenisation, the " four" prediction,
  causality and the previous-token head.
- **Drawing:** matrices are flat textured panels in the "Falk scheme" (A to
  the left of A·B, B above it, so rows and columns line up). Textures are
  mip-mapped separately per axis, so a 6 × 3072 matrix keeps its 6 rows
  when it is small and does not shimmer.
- **Camera:** the zoom is one transform of the whole world (log-space zoom,
  with the focus moving in step), so one model-space layout serves every
  shot. Labels slide to stay in the title-safe area as the camera pans.
- **Your own prompt:** `--text "..."` takes 2 to 8 tokens.

```bash
python3 -m voxanim render gpt2-attention
python3 -m voxanim render gpt2-attention --text "The cat sat on the"
```

### Tokens, embeddings, the stream, sampling (`bpe_anim.py`, `embed_anim.py`, `block_anim.py`, `sampling_anim.py`)

Four flat companions to `gpt2-attention`, every number read from the same GPT-2 small in `gpt2.py`.

**`gpt2-tokenizer`** (about 50 s): "The hippopotamus yawned" through GPT-2's byte-pair encoder. The regular expression cuts three pieces (a word keeps its leading space), each becomes its UTF-8 bytes (a space is written Ġ), then the 17 real merges from `merges.txt` are applied one at a time, lowest rank first within each piece — "The" is done in two (h+e rank 2, T+he rank 208) and is one token; " hippopotamus" stops after ten as Ġhipp | opot | amus. The six ids pick their rows of the 50,257 × 768 embedding table, drawn as 768-wide strips. Finally "four" (id 14,337) and " four" (id 1,440): the space byte merged into the word, two different rows, cosine 0.64.

**`gpt2-embeddings`** (about 60 s): the token table, " king"'s row (5,822) pulled out, its nearest rows by cosine (" kings" 0.73, " King" 0.70, " queen" 0.66, " prince" 0.59, …), and king − man + woman, which in GPT-2's table gives " king" itself first (0.78) and " queen" second (0.71). Then the 1,024 × 768 position table (each column scaled to its own spread: slow waves down the rows), the 1,024 × 1,024 cosine matrix of positions, banded along its diagonals, and the mean cosine against distance: 0.997 one apart, 0.47 at 100, zero at 164, lowest (−0.69) at 780.

**`gpt2-transformer-block`** (about 57 s): for "One, two, three," the last token's 768 numbers through all 12 layers: LN₁, attention, the add, LN₂, the 3,072-wide GELU hidden layer (9–19% > 0), the MLP's output, the add, with |x| beside each (5 at the embedding, 491 after layer 12). At every step the logit lens reads the stream as the output would be read (final LayerNorm, × W_Eᵀ, softmax): the early layers predict list-like continuations (" which", " including", then " etc" at up to 99%); " four" is below 0.1% until layer 10, whose MLP makes it the top token (65%); 93% after layer 11; the last layer spreads it to the model's 46%.

**`gpt2-sampling`** (about 63 s): the real distribution after "The cat sat on the" (" floor" 7.6%, entropy 7.8 bits ≈ 225 equally likely tokens). Greedy; temperature (the bars reshape as T moves 1 → 0.5 → 1.5: 24.9% and 3.5 bits, 1.8% and 11.7 bits); top-k = 10 (42.6% of the mass kept and rescaled); nucleus p = 0.9 on the sorted distribution with its cumulative mass (728 tokens here, 102 after "One, two, three,"); six tokens by nucleus sampling with each step's distribution and random number ("… sofa and yawned a little"); and the KV cache: one head's K grows by a row per token, the new q is scored against every stored row, 12 × 12 × T × 64 numbers at T = 11 and constant work in the weights per step.

Tests replay the tokenizer's merge trace against `encode`, check the logit lens's last reading against the model's logits and `block_states` against the forward pass, the sampling rules on hand-made distributions, and every number the subtitles quote.

### Embeddings: tokens, words, sentences, patches, and where they sit (`w2v.py`, `patch_embed.py`, `posenc.py`; `embed_lookup_anim.py`, `w2v_anim.py`, `sentence_anim.py`, `patch_anim.py`, `posblind_anim.py`, `sinusoid_anim.py`, `relbias_anim.py`, `alibi_anim.py`, `rope_ext_anim.py`, `lenx_anim.py`)

Ten flat diagrams, every number from three seeded, cached studies. `w2v.py` trains word2vec (skip-gram with negative sampling) on a generated toy corpus: 30,000 sentences from 8 templates over animals, people, foods and places, each person and animal with its own preferred places and foods, so that words of a category share contexts. `patch_embed.py` trains two small patch-embedding classifiers on MNIST, with and without position vectors. `posenc.py` holds the position encodings (sinusoidal, RoPE with interpolation and NTK scaling, T5 buckets, ALiBi slopes) and trains six identical one-layer, 4-head transformers, differing only in how they see position, on an order-2 Markov language over 8 symbols, at length 32, testing them at 128; its backward pass is written out by hand and checked against finite differences. `gpt2-position-blindness` runs the real GPT-2 small's first attention layer.

**`embedding-lookup`** (about 53 s): the word2vec study's trained input table `W_in`, 39 words × 16, as a heatmap. "the sheep chases the cat" is tokenised to ids 35 32 8 35 7, and each id copies its row out to make the sentence's 5 × 16 matrix ("the" twice, the same row twice). The same row as a one-hot vector times the table: 38 rows times 0, one times 1, and the product equals `W_in[32]` with a largest difference of 0. Per token, a lookup copies 16 numbers where the product does 624 multiply-adds. At GPT-2 small's 50,257 × 768 that is 768 against 38,597,376, all but 768 of them multiplications by 0. Then one skip-gram step on (baker, bread) with 5 noise words, at the exact training state of step 1,403: 1 row of `W_in` and 6 of `W_out` change, the other 71 rows not at all, and the pair's loss falls from 2.813 to 1.650.

**`word2vec-training`** (about 53 s): skip-gram with negative sampling on a small generated corpus of 30,000 sentences from 8 templates. A window of 4 words either side turns them into 898,216 (centre, context) pairs, and the model trains for 14,034 steps with 39 words and 16-wide vectors. One step at the exact state of step 1,403 pulls baker's vector towards bread's output vector (σ 0.21 → 0.36) and pushes it from 5 noise words (σ of "a" 0.26 → 0.09); the pair's loss falls from 2.81 to 1.65. Then the 24 category words, in one fixed 2-D view, move through 61 snapshots from a random cloud (cosines 0.02 within categories, −0.01 across) to four clusters: animals, people, foods and places. First they all swing one way (0.996 across categories at step 234); by the end it is 0.896 within categories against 0.245 across. Each word's 5 nearest neighbours are from its own category: cat → dog 0.997, baker → king 0.953, bread → rice 0.974, village → farm 0.998.

**`sentence-embeddings`** (about 61 s): word2vec's 16-number vectors from the toy corpus, averaged into sentence vectors. The seven words of "the farmer eats cheese in the village" drop into their mean, and one column is written out. The cosine similarity of six sentences' means: the two eating sentences 0.976, a dog chase against the doctor's shopping 0.390, and no pair below 0.39. "The dog chases the cat" and "the cat chases the dog" are the same five words, so their means differ only by rounding (2.2 × 10⁻¹⁶) and their cosine is 1.000000: mean pooling is a bag of words. SIF weights each word by a / (a + p(w)), a = 0.001: "the", 28% of the corpus, gets 0.0035, 24× less than "village", and removing the corpus's first principal direction takes 85% of the weighted mean's length. With SIF the eating pair keeps 0.847, the dog chase and the doctor fall to −0.805, and 9 of the other 14 pairs go below zero. The reordered pair is still 1.000: SIF is still an average.

**`patch-embeddings`** (about 61 s): a vision transformer's first layer on MNIST. A test digit is cut into a 4 × 4 grid of 7 × 7 patches, which lift apart into 16 tokens. Each patch's 49 pixels are multiplied by one shared matrix E (49 × 64), the same as a 7 × 7 convolution with 64 filters and stride 7; 16 of its columns are drawn as filters taking shape over 5,616 training steps. A second model adds one learned 64-number vector per grid place, so a blank patch's token is b + pos[k], place alone. The learned vectors draw no map of the grid: mean cosine −0.06 between neighbouring places, 0.06 between distant ones. Both models pool the tokens by their mean. Shuffle the patches and the model without positions still says 2 with p 0.746, while the model with positions says 5. Over 10,000 test digits: 83.8% without positions, 83.8% shuffled (all 10,000 predictions unchanged); 94.3% with positions, 26.3% shuffled.

**`gpt2-position-blindness`** (about 63 s): GPT-2 small's first attention layer on three orders of the same five words ("the dog bit the man", "the man bit the dog", "bit the man the dog"), with its position table and causal mask switched off and on, following " dog"'s 768-number output. With neither, attention is a weighted sum over a set. Shuffling the words only moves head 0's weights with them, row and column, and " dog" comes out the same in all three orders to 5×10⁻¹⁵, which is float rounding. Adding the learned position vectors wpe[0..4] changes the weights, and " dog" differs: cosine 0.983 and 0.980 with the first order. The causal mask alone also leaks order: " dog" sees two words in the first order (cosine 0.940), but it is last in the other two, sees the same five words and comes out the same (4×10⁻¹⁵). With both, as GPT-2 runs, all three differ (0.980, 0.978, 0.999).

**`sinusoidal-positions`** (about 57 s): the original transformer's position table, PE[p, 2i] = sin(pωᵢ) and PE[p, 2i+1] = cos(pωᵢ) with ωᵢ = 10000^(−2i/64), added to the token embedding. Five of its 32 pairs are drawn as waves: pair 0 repeats every 6.3 positions, pair 16 every 628 and pair 31 every 47,117. The whole 64 × 64 table is then drawn as the classic heatmap, fast on the left and all but constant on the right. Moving 4 places turns each pair by 4ωᵢ wherever it starts (2.25 rad for pair 2, 0.71 for pair 6). So PE[p + 4] = PE[p]·M₄ for a block-diagonal M₄ of 32 rotations, true to 3×10⁻¹⁵ at every p tried. The dot product PE[p]·PE[p + k] is Σᵢ cos(kωᵢ) whatever p is: 32 at k = 0, 19.4 at 16, 13.9 at 64, decaying and wiggling. Even so, the one-layer model trained with it on 32-token sequences reaches a test loss of 1.20 within them (best possible 1.19) and 3.43 at positions 63–126.

**`relative-position-bias`** (about 55 s): T5's relative position bias in the study's one-layer, 4-head model trained on an order-2 Markov language over 8 symbols at length 32. No position vector is added to the tokens. Each head adds a learned number to the score of query i against key j, looked up by the offset i − j. The offsets go into 16 buckets: 0 to 7 get one each, then 8–9, 10–11, 12–13, 14–15, 16–19, 20–22 and 23–26, and every offset from 27 on shares bucket 15. All four heads learned their largest bias at offset 1, the previous token, which this language needs: +2.17 to +2.28. The far bucket sits near −2.2, and head 0 also gives itself +0.65. Offsets past 32 all fall into bucket 15, which was trained. At four times the length the loss is 1.20, 1.21 and 1.24 over positions 2–31, 32–63 and 64–127, against a best possible 1.19. Head 3's attention maps at length 32 and 128 look alike, but its weight on the previous token drops from 0.65 to 0.38 past position 64: 38 to 101 keys now share the far bucket, and offset 1 leads it by only 4.5.

**`alibi`** (about 55 s): ALiBi in the same one-layer, 4-head model, with no position vectors at all. Head h subtracts m_h · (i − j) from every score, with fixed slopes 1/4, 1/16, 1/64 and 1/256. The penalty's weight e^(−m·d) halves every 2.8 tokens in the steepest head and every 177 in the shallowest; at the training length of 32 the factor is 0.00034 in one and 0.88 in the other. In the trained model the steepest head looks back 3.1 tokens on average, splitting about evenly between the token itself (0.20) and the previous one (0.21). The shallow heads spread over 6 to 8 tokens and pick keys by symbol. T5's learned biases give the previous token 0.53 to 0.69. Because the penalty is defined at every distance, nothing new appears at four times the length. ALiBi's loss goes from 1.34 (positions 2–31) to 1.40 (32–63) and stays at 1.40 (64–127). RoPE's matches T5's 1.20 in range, then climbs to 2.84 and 3.57, above the 2.07 of a guess with no context at all. The price is in range: 1.34 against T5's 1.20, since a penalty that only falls with distance always ranks the token itself above the one before.

**`rope-context-extension`** (about 55 s): the RoPE model of the length family (4 rotary pairs per head, θ = 1, 0.1, 0.01, 0.001), trained on 32 symbols and run on 128 with no retraining. Four dials show the angles Δ·θᵢ: the fast pair already went round 4.9 times in training, while the slow pairs are pushed to angles they never saw (pair 1 from 3.1 to 12.7 rad). On a real sequence, head 0 gives the previous token 0.74 of its attention inside 32. From position 39 its strongest key is more than 31 back, and past 64 it puts 0.22 on a token 51 back and 0.01 on the previous one. The loss goes from 1.20 inside 32 to 2.84 at 32–63 and 3.57 at 64–127. Position interpolation (positions × 32/128) brings every angle back into range, but the previous token now sits at 0.25 rad in the fast pair, where training had 1 rad, and offset 4 takes offset 1's old angle. The previous-token weight falls to 0.14 and the loss is worse everywhere, 2.47 inside 32. The NTK-aware base 63,496 leaves the fast pair alone and slows the slowest by 4: 1.22 inside 32 and 1.38 at 32–63, but 3.48 at 64–127.

**`length-extrapolation`** (about 58 s): the family finale. Six identical one-layer transformers (4 heads, width 32) learn an order-2 Markov language over 8 symbols, where the next symbol depends on the last two. The best possible loss is 1.19 nats, 1.94 knowing only the current symbol. The models differ only in how they see position: none, learned absolute, sinusoidal, RoPE, ALiBi and T5's relative bias. Trained for 4,000 steps on 32-symbol sequences, four of them end within 0.04 of 1.19, ALiBi at 1.37 and no positions at 1.79. Tested on 128 symbols, learned, sinusoidal, RoPE and T5 are all within 0.01 of the best inside 32. Past it, learned rises to 2.82, sinusoidal to 3.43 and RoPE to 3.57 at 64–127, while T5 holds at 1.24 and ALiBi at 1.40. ALiBi pays for this inside 32 (1.34), and no positions sits at the current-symbol level (1.92). What extrapolates depends only on the offset and has a trained answer for every offset it meets: T5's offsets past 26 share the bucket training used at 27–31, and ALiBi's steepest head (slope 0.25) still gives the previous token 0.21 of its attention inside 32 and 0.19 past 64. The learned and sinusoidal tables meet positions they never trained on, and RoPE meets angles it never saw.

Tests check every gradient (word2vec, the patch model, the transformer with each of the six schemes) against finite differences, the encodings' identities (a sinusoidal offset is a rotation to 10⁻¹², a RoPE score depends on the offset only, T5's buckets, ALiBi's slopes, the language's entropies) and the claims the animations make, as relative claims.

### What language models know, and don't (`gpt2_probe.py`, `halluc_anim.py`, `condition_anim.py`, `calib_anim.py`)

Three flat diagrams measured on the real GPT-2 small. `gpt2_probe.py` runs the same forward pass as `gpt2.py` in float32 (over aligned copies of the weights, 40× faster than the memory-mapped arrays), returning next-token distributions, seeded samples at temperature 1, entropies in bits and the last layer's attention; every number on screen is computed once and cached.

**`gpt2-hallucination`** (about 47 s): the same two-shot prompt about France and about Freedonia. After the real one the distribution is sharp (' Paris' 0.68, entropy 2.4 bits); after the invented one it is flat (entropy 8.6 bits) but a city still sits on top (' Athens' 0.12). Six sampled continuations each: 4 of 6 say Paris; Freedonia gets six different cities (Sparta, Berlin, Ed, Mysia, Budapest, Cairo) — self-consistency is a signal the text alone does not carry. 'unknown' ranks 5,259th and 6,842nd: the model predicts plausible text, not truth.

**`gpt2-conditioning`** (about 45 s): p(Paris) after "The capital of France is" as context is prepended piece by piece: alone 0.03 (' the' wins), with an instruction 0.09, with one worked example 0.58, two 0.61, three 0.66, drawn as a curve against the context length. Examples of another form ("The capital of Japan is in Asia.") send the same question to ' in' (0.98). Then where the conditioning is read from: the last layer's attention from the last token as a strip over the context — the mean over heads parks 0.71 on the first token (the attention sink); head 9 reads the examples instead.

**`gpt2-calibration`** (about 46 s): 40 countries whose capital is one token, each asked after two worked examples. GPT-2 small is right 32 times (80%) at a mean confidence of 0.56; the wrong answers are plausible big cities (Istanbul, Zurich, Sydney, Toronto, Mumbai). The dots gather into a reliability diagram (5 bins, ECE 0.25; the model is under-confident, every bar above 0.2 sits over the diagonal), and Oslo 0.55 (right) and Zurich 0.55 (wrong) are held up side by side: from inside the model they are the same number.

Tests check the entropy and softmax arithmetic, the binning and ECE on hand-made data, that the probe matches `gpt2.run`, and every number the subtitles quote.

### Modern sequence models: state spaces, Mamba, the cost of context, experts, rotary positions (`ssm.py`, `mamba.py`, `moe.py`, `ssm_anim.py`, `mamba_anim.py`, `cost_anim.py`, `moe_anim.py`, `rope_anim.py`)

Five animations on the building blocks that came after the transformer, every number from a model run or trained in plain numpy.

- **`ssm-recurrence` — SSM: one recurrence, two views.** A diagonal linear state space model (6 states, `A_n = −(n+1)`, step `Δ = 0.1`, zero-order hold: `Ā = exp(ΔA)`, `B̄ = (Ā−1)/A·B`) reads a sine wave with a step. The recurrence `h_t = Ā h_{t−1} + B̄ x_t` fills the state band column by column; then the kernel `K_k = C·Āᵏ B̄` slides over the input and the convolution gives the same output to 4×10⁻¹⁷. Then `Δ` falls to 0.02 and rises to 0.5: `Ā₁` goes 0.98 → 0.61, the slowest state's memory 50 → 2 steps, the kernel lengthens and shortens with it.
- **`mamba-selective-scan` — Mamba: selection.** A minimal Mamba block (16 channels × 8 states, width-4 causal convolution, input-dependent `Δ_t`, `B_t`, `C_t`) on `gated-memory`'s task: say the first value after 49 steps of noise. The cue and the noise come from the same distribution, so the block can tell the first step only by the convolution's zero padding — and that is exactly what it learns: `Δ` is 0.56 at the cue and 0.019 on the noise, 67 % of the state survives the cue step against 91 % elsewhere, one channel's read-out carries the cue across the whole gap, the answer is 0.81 for 0.80. Without the convolution the block stays at chance. The same states are then computed as a prefix scan, 6 combine levels for 50 steps, agreeing to 2×10⁻¹⁵; and the training curve (4000 Adam steps, solved by step ~2100) is drawn against the RNN's and the LSTM's from `memory.py`. Seeds and sizes that do not learn are listed in `mamba.py`.
- **`context-cost` — The cost of context.** A context growing from 1 to 16 tokens: attention's causal `n × n` score matrix, the recurrence's chain, the scan's `⌈log₂ n⌉` levels; the work each new token costs (`n`, 1, 1) and the depth of the whole sequence (1, 16, 4). Then memory at inference with real shapes: GPT-2 small's KV cache grows by 12 × 2 × 12 × 64 = 18,432 numbers per token; a Mamba-130M's state is 24 × 1536 × 16 = 589,824 whatever the length (plus 110,592 for its convolution window). Drawn with one voxel = 768 numbers in both: the cache passes the state at 32 tokens.
- **`mixture-of-experts` — Mixture of experts.** A top-2 router over eight 16→16→16 experts, trained on next-character prediction in a language with regimes (digits count up, letters run along the alphabet, punctuation switches between them; test cross-entropy 1.15 against the chain's floor 1.11). Each token of a sample text flies to the router; its eight probabilities, the two chosen experts computing while six idle, the gate weights, the next-character guess against the real one. Then expert utilisation over 2,048 fresh tokens, split by regime, for the model trained with the Switch load-balancing loss and one trained without (busiest / quietest 2.3× against 3.2×; no expert dies at this size, as the 16-run sweep in `moe.py` records). Finally the count: 1,024 multiply-adds per token for 4,352 expert parameters.
- **`gpt2-rotary-embeddings` — Rotary position embeddings.** A query and a key of width 64 split into 32 pairs, each turned by `m θ_i`, `θ_i = 10000^(−2i/64)`: pairs 0, 4 and 8 drawn as arrows turning at their own rates. The score `R(m)q · R(n)k` depends on `n − m` only: positions (3, 7) and (10, 14) give the same number to ten decimals, (3, 10) does not. A vector against itself `Δ` apart: fast pairs oscillate, slow ones hold, the sum decays (0.84 → 0.41 over 49 positions). Against GPT-2 small's learned table (its real `wpe`): the cosine similarity between position vectors is 0.99 one apart, 0.86 sixteen apart, 0.43 forty-eight apart — smooth, but tied to the absolute index and ending at row 1023, where the rotary formula holds at any index. Needs the GPT-2 download.

### Diagrams: gradient descent, an MLP classifier, a decision tree (`diagram.py`)

Three flat, node-and-line diagrams. `diagram.py` draws them in the same
renderer: neurons, boxes and particles are camera-facing billboards sized in
pixels (analytic anti-aliasing at any resolution), all in one plane that
faces the camera, so draw order is just `lift` (edges, then panels, then
nodes).

**`gradient-descent`** (`gd_anim.py`, about 61 s): a 2-3-1 network with
sigmoid activations learning XOR by plain full-batch gradient descent.
The data flows forward in teal and the gradients flow back in magenta.

1. **Forward:** the example x = (1, 0) with target 1. Each neuron shows z,
   its value σ(z), and its activation function: a small sigmoid plot with
   the operating point on it.
2. **Loss:** L = ½(ŷ − y)².
3. **Backward:** ∂L/∂ŷ, times the slope σ′(z) (drawn as the tangent on the
   neuron's sigmoid), gives the output's δ. Each weight's gradient is δ ×
   the value that came in, and the hidden δs follow. Every neuron's ring
   and every weight's magenta band lights up once its gradient has been
   computed, and stays lit. x₂ = 0, so its weights get no gradient from
   this example.
4. **Batch:** the same for the other three examples. The recorded
   gradients become their average.
5. **Update:** w − η · ∂L/∂w with η = 2. Edge width and colour are the
   weight's size and sign.
6. **Train:** 2,000 steps. Data and gradients stream in opposite
   directions on every edge, the loss curve falls (a plateau, then the
   drop) and the table's predictions become 0.03, 0.94, 0.95, 0.07.

**`mlp-classifier`** (`mlp_graph.py`, about 44 s): a real 784-32-16-10 ReLU
network (95.6% on the MNIST test set) as the classic diagram. A line's
opacity is how much it carried for this digit (|w × incoming value|; orange
adds, blue subtracts), so only the paths that matter show. A dark neuron is
one that ReLU switched off. One digit goes through layer by layer, with one
neuron per layer worked out in numbers (z = Σ wᵢxᵢ + b, then ReLU), then
softmax and the answer. Seven more test digits follow, including a 2 the
network reads as a 3.

**`decision-tree`** (`dtree.py`, about 43 s): CART with Gini impurity on 120
points with 2 features and 3 classes (maximum depth 3, at least 5 points per
leaf).

1. **Best split:** at the root, every threshold on x₁ and then x₂ is
   tried. The weighted Gini of the two halves is drawn as a curve along
   each axis, and its lowest point wins.
2. **Grow:** the same search inside each box.
3. **Regions:** the leaves colour the plane. The diagonal class boundary
   becomes a staircase.
4. **Predict:** two new points fall down the tree one question at a time
   ("is x₁ = 0.62 ≤ 0.42? no, go right").

Tests check the XOR gradients against finite differences, that every split
lowers the impurity and agrees with a brute-force search, and that all
three loop seamlessly.

```bash
python3 -m voxanim render gradient-descent mlp-classifier decision-tree
```

### The maths, step by step: autodiff, softmax, activations (`autodiff.py`, `autodiff_anim.py`, `softmax_anim.py`, `activations_anim.py`)

Three flat diagrams (`diagram.py`) about the pieces every network is built from, each with the real numbers in it.

**`autodiff-graph`** (`autodiff.py`, `autodiff_anim.py`, about 50 s): reverse-mode automatic differentiation on the graph of one sigmoid neuron, L = ½(σ(w₁x₁ + w₂x₂ + b) − y)². The engine is forty lines: every `Value` records its operation, its parents and the local derivative of its output with respect to each parent; `backward` walks the graph from L to the leaves, multiplying the incoming gradient by each edge's local derivative. The numbers are the XOR network's first hidden neuron before training on `gradient-descent`'s worked example x = (1, 0), y = 1, so z = 0.126 and σ(z) = 0.531 are the same in both animations.

1. **Build:** the expression is read inside out and every operation becomes a node (×, +, σ, −, ½d²) joined to what it reads: 7 operations, 6 leaves.
2. **Forward:** values flow right (teal) and appear under each node; each node also records its local derivatives (∂m₁/∂w₁ = x₁ = 1, ∂a/∂z = σ(1 − σ) = 0.249, ...).
3. **Backward:** from ∂L/∂L = 1 the gradient (magenta) flows left one edge at a time, each edge showing "× local derivative": ∂L/∂z = −0.469 × 0.249 = −0.117, ∂L/∂w₁ = −0.117 × 1, ∂L/∂w₂ = −0.117 × 0 = 0 (x₂ was 0). Twelve multiplications; no formula for ∂L/∂w is ever written down.
4. **Result:** ∂L/∂w₁ = ∂L/∂b = −0.117, ∂L/∂w₂ = 0, then one step with η = 2 (w₁ 0.126 → 0.359, b 0 → 0.233) and the forward pass again: L 0.110 → 0.063.

**`gpt2-softmax-cross-entropy`** (`softmax_anim.py`, about 48 s; needs the GPT-2 weights): two real sets of logits side by side. Left, the MNIST MLP of the `mlp-*` animations on test image 18, a 3 (10 logits, −5.70 to 6.89); right, GPT-2 small on "The capital of France is" (50,257 logits, the ten most probable tokens drawn; the true next word " Paris" is ranked fifth). Both go through the same steps as horizontal bars with every number printed: the **logits**; **softmax** as exp(z − max z) (a difference of 1 is a factor e; subtracting the largest changes nothing) divided by the sum over 10 digits, or over all 50,257 tokens (the top ten hold 0.36); the one-hot **target** and the cross-entropy L = −log p(answer): −log 0.620 = 0.48 for the 3, −log 0.032 = 3.43 for Paris; the **gradient** ∂L/∂z = p − y drawn as the gap between p and y under each bar (orange: this logit comes down, blue: up; the gradients sum to 0); then **temperature**, p = softmax(z / T): T = 2 melts both (entropy 0.67 → 0.94 nats for the MLP, 6.00 → 9.43 for GPT-2, whose top ten then hold 0.03), T = 0.5 sharpens them (0.59 and 2.28; "the" takes 0.40).

**`activations`** (`activations_anim.py`, about 49 s): sigmoid, tanh, ReLU and GELU (GPT-2's tanh form) drawn with their derivatives, then the 64 pre-activations of the MNIST MLP's first hidden layer on test image 18 (−6.17 to 4.67, 26 negative) as dots on each curve and as rows of voxels before and after each function: 12 of 64 saturate the sigmoid and 24 tanh (|f′| < 0.05), ReLU zeros the 26 negative ones exactly (dark), GELU lets them through slightly negative. Then **dead ReLUs**: a 784-128-10 ReLU network trained by plain SGD on MNIST with a learning rate of 1.0 (batch 128, one epoch, seeded and cached with every unit's state on the whole test set at 102 checkpoints). Within the first 30 steps 31 of its 128 hidden units stop firing on every one of the 10,000 test images and never fire again: their voxels go dark for good and the count rises on the chart, against a run at lr 0.3 where none die. Both end the epoch at the same accuracy (94.8% and 94.5%): the 31 units were lost for nothing. The gradient through a dead unit is exactly 0 (ReLU′(z) = 0 for z < 0), so nothing can bring it back.

Tests check the autodiff engine and every activation's derivative against finite differences, the softmax gradient p − y, the numbers the subtitles quote, that no dead unit revives and that its gradient on the whole test set is exactly zero, and that all three loop seamlessly.

### Q-learning in a gridworld (`gridworld.py`, `qlearning_anim.py`, `q-learning`)

A 6 × 8 world with walls, a goal (+1) and a trap (−1), both ending the
episode; every other move pays 0, so with γ = 0.9 the best policy is the
shortest path. Tabular Q-learning, α = 0.5, ε-greedy with ε decaying from 1
to 0.05 (`max(0.05, 0.99^k)`), 400 episodes, every move and every episode's
Q-table recorded. The Q-table is drawn *on* the world: four columns of
voxels per cell, one at each side for the action that leaves through it,
height Q(s, a) (orange above zero, blue below).

1. **The world:** tiles, walls, start, rewards, α and γ.
2. **Episode 1:** 21 random moves; the update
   `Q(s, a) ← Q(s, a) + α (r + γ max Q(s′, ·) − Q(s, a))` is shown with its
   numbers after each move and changes nothing until the agent walks into
   the trap: `0 + 0.5 (−1 + 0.9 × 0 − 0) = −0.5`, the first entry.
3. **Learning:** episodes 2–400 fast-forwarded with the counter and ε;
   each path flashes and the columns rise one cell further back from the
   rewards at every visit. The goal is first reached in episode 41.
4. **Policy:** the greedy action of each cell as an arrow on its tallest
   column; from episode 102 on it is a shortest path.
5. **Greedy path:** the agent follows the arrows: 14 moves, round the
   trap; Q(start, up) = 0.25 = 0.9¹³.

The seed (25) was chosen so the first episode is short enough to watch; the
outcome does not depend on it (40 seeds of 40 end on a shortest path with
these settings). `tests/test_gridworld.py` replays every recorded step with
the rule and checks the convergence and the numbers above.

### The next 237: mechanisms on real data (`mech.py`, `mech_*.py`, `study_*.py`, `next_*.py`)

A further 237 animations, in thirty-four groups: 109 added in 1.6.0, in 1.7.0 eight on
randomness and 41 for a physics course on data analysis and machine learning, in 1.8.0
one on training with and without a GPU, in 1.9.0 ten on decision trees, forests and
boosting, and since then five on entropy, impurity and loss and six bridges between
topics, each opening the section it leads into, two that count a model's weights, and thirty, "Blocks and code", that show a
PyTorch model and its training as blocks beside the code, for students who find this hard, with nine more that change its model to a small CNN, ten on padding and four on vector databases. The GPU one, `cpu-and-gpu-training`, is the catalogue's
single explicit exception to the rule that results on screen are computed or measured: its
speeds are representative round figures, labelled as such in every phase. They are listed in
`next_catalog.json` and
registered like every other preset, so `list`, `info`, `still`, `render`, `qa` and `prepare`
all work on them.

Each has two parts:

- **a study**: the measured experiment. Learning results come from plain-numpy training on
  MNIST or on Tiny Shakespeare (checksum-pinned, split before sampling), or from the pinned
  GPT-2 used as it is; counts, masks and schedules are exact arithmetic and say so on screen.
  Comparisons between models run over three or more seeds. The first experiments are in
  `next_attention.py`, `next_language.py`, `next_vision.py` and `next_systems.py`; the 56
  that could not show their concept at their first size were redesigned in `study_*.py`
  (`next_catalog.redesigned`). Where a fair redesign still came out null, or only partly as
  the textbook has it, the animation says what was measured. `TODO.md` ends with the outcome
  of each.
- **a drawing**: a `Mechanism` (`mech.py`) in four to six named phases, each with its own
  live subtitle: the things involved, the mechanism moving on a real instance, then the
  measurement it leads to. The kit is token rows, arcs with values travelling along them,
  matrices that fill in, digit tiles, charts and bars, in the conv2d colours.

Tests: `tests/test_mech_*.py` hold every number on screen to its study and every study's
claim to a margin; `tests/test_next_review.py` builds and loops every one.

**Attention: how it is wired.** Who may read whom: masks, heads, caches and the ways attention is made cheaper.

| name | shows |
| --- | --- |
| `causal-vs-bidirectional` | **Causal and bidirectional attention.** The causal mask against the full one: who may read whom, drawn as arcs and as the mask, then a model trained each way. |
| `prefix-lm-attention` | **The prefix mask.** A prompt read in both directions and a reply read left to right, and why the reply has to be. |
| `cross-attention` | **Cross-attention.** A decoder position sends a query to the encoder's row and reads back a weighted mix of what is there. |
| `multi-head-vs-single` | **One head or eight.** The same numbers as one attention head or as eight, and what the eight end up looking at. |
| `grouped-query-attention` | **Grouped-query attention.** Query heads sharing keys and values: what is shared, and the memory it saves per token. |
| `gpt2-kv-cache` | **The key-value cache.** Decoding token by token with and without a cache of keys and values: the same answer for a fraction of the work. |
| `sliding-window-attention` | **Sliding-window attention.** Each position reads only a short window, and stacked layers reach further: exactly as far as the windows compose. |
| `sparse-attention-patterns` | **Sparse attention patterns.** Five ways to let a position read fewer of the others, what each costs and what each can still learn. |
| `linear-attention` | **Linear attention.** Attention rewritten as a running sum over keys, instead of a score for every pair. |
| `flash-attention-tiling` | **Flash attention: the online softmax.** Attention computed a tile at a time with a running maximum and sum, giving the same result without the whole matrix. |
| `gpt2-attention-sinks` | **Attention sinks.** The few early tokens that many heads keep reading, and what happens to the model when the cache evicts them. |
| `attention-temperature` | **Why divide scores by √d?** What dividing attention scores by the square root of the width does to the softmax and its gradient. |

**The transformer block.** The design choices inside a block: where to normalise, what the MLP is, how wide and deep.

| name | shows |
| --- | --- |
| `pre-ln-vs-post-ln` | **Pre-LN and post-LN.** Layer normalisation on the branches or on the stream, and which of the two still trains when the model is deep. |
| `rmsnorm-vs-layernorm` | **RMSNorm and LayerNorm.** LayerNorm in two steps and RMSNorm in one, on a real activation, and whether a model notices the difference. |
| `swiglu-mlp` | **A gated MLP: SwiGLU.** The usual MLP against one whose units are multiplied by a learned gate. |
| `gpt2-residual-stream` | **The residual stream.** GPT-2's stream of numbers for one token, with every layer reading from it and writing back to it. |
| `weight-tying` | **Weight tying.** One table of embeddings used at both ends of a model: the weights saved and the gradient arriving from both sides. |
| `parallel-block` | **Sequential and parallel blocks.** Attention and MLP one after the other, or both at once from the same input. |
| `encoder-decoder-family` | **Encoder, decoder, encoder-decoder.** Encoder only, decoder only and encoder-decoder on one task: what each may read and how each writes its answer. |
| `transformer-depth-vs-width` | **Depth and width.** The same number of weights spent as a few wide blocks or many narrow ones. |

**Training language models.** What a language model is trained on, how its loss is read, and what makes training work.

| name | shows |
| --- | --- |
| `teacher-forcing` | **Teacher forcing.** Training on the true text shifted by one place, and what changes when the model must read its own output. |
| `gpt2-perplexity-explained` | **Perplexity.** A model's surprise at each token, averaged and turned into a number of equally likely choices. |
| `packing-and-padding` | **Packing and padding.** Padding lines to a common length against packing them end to end, and the two resets packing needs. |
| `warmup-and-clipping` | **Warm-up and gradient clipping.** Why a learning rate is ramped up and gradients are cut to a maximum length, on a model that fails without the ramp. |
| `grokking` | **Grokking: the test catches up late.** A network that knows its training set by heart long before it gets new examples right. |
| `scaling-laws` | **Scaling laws, from a ladder of budgets.** How loss falls as compute grows, and which model size makes the best use of each budget. |
| `curriculum-and-data-mixing` | **Data mixing and curriculum.** Two sources of text mixed in fixed shares or on a schedule, and what each mix does to both. |
| `tokenizer-effects` | **What the tokenizer changes.** The same text as characters, bytes and merged pairs, and what the choice does to length, table size and loss. |

**Decoding and inference.** Turning next-token chances into text, and serving it quickly.

| name | shows |
| --- | --- |
| `gpt2-beam-search` | **Beam search.** Keeping the best few continuations at every step instead of only the likeliest one. |
| `gpt2-speculative-decoding` | **Speculative decoding.** A cheap draft model guesses ahead and the large model checks the guesses in one pass, without changing what is sampled. |
| `gpt2-repetition-and-penalties` | **Repetition and penalties.** Greedy decoding falling into a loop, and three rules that reshape the next-token chances to break it. |
| `gpt2-constrained-decoding` | **Constrained decoding.** A grammar masking the tokens a model may pick, so the output always has the required form. |
| `gpt2-contrastive-decoding` | **Contrastive decoding.** Scoring a token by how much more a full model believes in it than a weak one does. |
| `batching-and-throughput` | **Batching and throughput.** Serving many requests in one multiply: what a batch buys, why, and how slots are refilled. |

**Fine-tuning and alignment.** Changing a trained model: instructions, preferences, small add-ons and weight arithmetic.

| name | shows |
| --- | --- |
| `instruction-tuning` | **Instruction tuning.** Fine-tuning on instruction and answer pairs, graded on the whole example or on the answer alone. |
| `reward-model` | **A reward model from preferences.** A model that learns a score from pairs in which one option was preferred to another. |
| `rlhf-ppo` | **RLHF with PPO.** A policy pushed towards what a reward model scores highly, on a leash that limits how far it drifts. |
| `dpo` | **Direct preference optimisation.** Training the policy on the preference pairs directly, with no reward model in between. |
| `reward-hacking` | **Reward hacking.** Optimising a learned score past the point where the thing it stood for stops improving. |
| `prompt-and-prefix-tuning` | **Prompt and prefix tuning.** Adapting a frozen model by training a few added numbers: a soft prompt, a prefix or a low-rank update. |
| `model-merging` | **Model merging with task vectors.** Fine-tuned weights minus base weights as a vector that can be added, averaged and subtracted. |
| `gpt2-knowledge-editing` | **Knowledge editing.** Changing what GPT-2 says next in one context by a rank-one change to one matrix. |

**Interpretability.** Looking inside a trained model: features, probes, patches and the heads that matter.

| name | shows |
| --- | --- |
| `superposition` | **Superposition.** Five features squeezed into two numbers, and the arrangement a network finds when the features are rarely on together. |
| `gpt2-sparse-autoencoder-features` | **Sparse autoencoder features.** A sparse autoencoder rewrites GPT-2's activations as a few active features out of many. |
| `gpt2-linear-probes` | **Linear probes.** Reading a property of the text off GPT-2's activations with a straight line, layer by layer. |
| `gpt2-activation-patching` | **Activation patching.** Copying one activation from a clean run into a corrupted one to find where the information is carried. |
| `gpt2-induction-heads` | **Induction heads.** Heads that find the last time the current token appeared and read what came after it. |
| `gpt2-head-ablation` | **Head ablation.** Switching off one attention head at a time and measuring what the model loses. |
| `saliency-and-integrated-gradients` | **Saliency and integrated gradients.** Which pixels a classification rests on: the plain gradient against a method whose shares add up. |
| `feature-visualisation` | **Feature visualisation.** Changing an image's pixels, not a network's weights, to find what one of its units responds to. |

**Vision.** What image models see, attend to and get wrong.

| name | shows |
| --- | --- |
| `receptive-field` | **Receptive field.** How much of the image one output cell of a CNN can see, and how much of that it uses. |
| `vit-attention` | **What a vision transformer attends to.** An image cut into patches, and which patches the class token reads in each block. |
| `clip-contrastive` | **Contrastive image-text training.** Images and words trained into one space so that each image lands nearest its own word. |
| `adversarial-examples` | **Adversarial examples.** A change to a digit too small to notice that makes a classifier read a different digit. |
| `object-detection-basics` | **Object detection: boxes, IoU, suppression.** Boxes as four numbers, the overlap score that judges them, and how duplicate boxes are removed. |
| `semantic-segmentation` | **Semantic segmentation with a U-Net.** A label for every pixel, from a network that shrinks the image and grows it back with a skip across. |
| `batchnorm-in-cnns` | **Batch normalisation in a CNN.** What batch normalisation computes for each filter, and the cost of using a batch's own statistics at test time. |
| `style-transfer` | **Style transfer with Gram matrices.** Style as the statistics of a CNN's filters, and an image optimised to take them from another. |

**Generative models.** Models that make new samples: adversaries, flows, diffusion, codes and pixels in order.

| name | shows |
| --- | --- |
| `gan-training` | **GAN training.** A generator and a discriminator training against each other, and a generator that keeps dropping part of the data. |
| `normalizing-flows` | **Normalizing flows.** An invertible map from a simple bell-shaped cloud to the data, which makes the likelihood exact. |
| `flow-matching` | **Flow matching.** Learning the velocity that carries noise to data along straight lines, then straightening the paths. |
| `classifier-free-guidance` | **Classifier-free guidance.** Asking a denoiser twice, with and without the label, and pushing its answer along the difference. |
| `ddpm-vs-ddim` | **DDPM and DDIM sampling.** Two ways down the same ladder of noise levels: many small random steps or a few deterministic strides. |
| `vae-posterior-collapse` | **VAE posterior collapse.** A decoder strong enough to ignore the code, and the pressure schedule that keeps the code in use. |
| `vq-vae-codebook` | **VQ-VAE: a codebook.** Images rewritten as a grid of code numbers from a small learned codebook, and codes that die unused. |
| `autoregressive-pixels` | **Autoregressive pixels.** An image generated one pixel at a time, each predicted from the pixels already drawn. |

**Classical machine learning.** The ideas under everything else: fitting, validating, separating and measuring.

| name | shows |
| --- | --- |
| `bias-variance` | **Bias and variance.** Fits that are too simple, fits that change with every sample, and the error they add up to. |
| `cross-validation` | **Cross-validation.** Holding out each part of the data in turn, and why choosing a setting needs a second round inside the first. |
| `svm-and-kernels` | **SVMs and kernels.** The widest street between two classes, then the same idea after lifting the points to where a line can separate them. |
| `gradient-boosting` | **Gradient boosting.** Small trees added one at a time, each fitted to what the ones before still get wrong. |
| `gaussian-mixture-em` | **Gaussian mixture, by EM.** Soft assignment of points to blobs, then blobs moved to their points, repeated; and how much the start matters. |
| `curse-of-dimensionality` | **The curse of dimensionality.** Why nearest neighbours stop meaning much as uninformative coordinates are added. |
| `tsne-and-umap` | **t-SNE and UMAP.** Two methods that lay high-dimensional points out on a plane, and what their maps keep and lose. |
| `roc-and-calibration` | **ROC and calibration.** A classifier's scores turned into decisions by a threshold, and whether a score of 80% means 80%. |

**Optimisation, deeper.** Curvature, noise and averaging: why training behaves as it does.

| name | shows |
| --- | --- |
| `newton-vs-gradient-descent` | **Newton's step and gradient descent.** Gradient descent crawling along a long valley against a step that divides by the curvature. |
| `edge-of-stability` | **The edge of stability.** Curvature that rises during training until the step size will allow no more, then stays there. |
| `batch-size-and-noise` | **Batch size and gradient noise.** Small batches give noisy gradients: how the noise shrinks with batch size and what it does to training. |
| `weight-averaging` | **Averaging the weights.** The average of the weights along a run against the last weights of the run. |
| `second-order-preconditioners` | **Preconditioning.** Rescaling each direction by its curvature, and what the off-diagonal terms add. |
| `loss-spikes` | **A loss spike, inside Adam.** A sudden jump in the loss on unchanged data, traced to the optimiser's fading memory of past gradients. |

**Retrieval and contrastive learning.** Embeddings trained to bring matching things together, and searching among them.

| name | shows |
| --- | --- |
| `gpt2-retrieval-augmented-generation` | **Retrieval-augmented generation.** Looking up a passage and putting it in front of the question, and what the model can then say. |
| `vector-search-hnsw` | **Vector search: a layered graph.** Finding near neighbours through a layered graph while computing only a fraction of the distances. |
| `infonce-temperature` | **Temperature in contrastive learning.** The temperature of a contrastive loss decides how much of the push lands on the hardest negatives. |
| `hard-negatives` | **Hard negatives.** Choosing the most similar wrong examples as negatives, and the ones among them that are not wrong. |
| `dimensionality-of-embeddings` | **How wide should an embedding be?** The same images as 2 numbers or 256: storage against retrieval accuracy. |

**Systems and scale.** Spreading a model over devices, and what limits how fast it runs.

| name | shows |
| --- | --- |
| `data-parallel-allreduce` | **Data parallelism and ring all-reduce.** Four workers computing gradients on different data and averaging them by passing chunks round a ring. |
| `tensor-and-pipeline-parallel` | **Tensor and pipeline parallelism.** One layer split across devices by column or by row, and a model's layers run as a pipeline with idle slots. |
| `zero-sharding` | **ZeRO: sharding the training state.** Training state cut into slices so that each device holds only its share. |
| `activation-checkpointing` | **Activation checkpointing.** Storing only some activations on the way forward and recomputing the rest on the way back. |
| `roofline` | **The roofline performance model.** An operation is limited by arithmetic or by memory traffic: counting both and placing it under the roof. |
| `cpu-and-gpu-training` | **Training on a CPU and on a GPU.** Where a model's weights and a batch of digits sit in a computer, where the arithmetic happens with and without a GPU, how many bits each number gets (a CPU's 64 against a graphics card's 32, 16, 8 and 4, and what 4 bits do to a trained model), and how long a small and a larger model really take to train on each. |

**Reinforcement learning methods.** Learning from reward alone: bandits, policy gradients, values and search.

| name | shows |
| --- | --- |
| `multi-armed-bandits` | **Multi-armed bandits.** Choosing between options of unknown worth: three ways to balance trying and exploiting. |
| `policy-gradient` | **Policy gradient (REINFORCE).** Making sampled actions more likely in proportion to the reward that followed, and why a baseline helps. |
| `actor-critic` | **Actor-critic.** An actor choosing actions and a critic estimating what to expect, each improving the other. |
| `dqn-replay-buffer` | **DQN: replay buffer and target network.** Learning action values from replayed experience against a slowly moving copy of the network. |
| `mcts-tic-tac-toe` | **Monte Carlo tree search.** Search by random playouts: select, expand, play out and back up, on a game everyone knows. |
| `reward-shaping` | **Reward shaping.** Extra rewards that leave the best behaviour unchanged, and extra rewards that get farmed. |

**Memory, dynamics and curiosities.** Networks as memories, as rules that grow shapes, and as flows.

| name | shows |
| --- | --- |
| `hopfield-network` | **A Hopfield memory.** Patterns stored as the resting states of a network, recalled from damaged copies, and how many it can hold. |
| `neural-cellular-automata` | **A neural cellular automaton.** A grid of cells sharing one small rule that grows a digit from a single cell and regrows it when damaged. |
| `neural-ode` | **A neural ODE.** A residual network as steps along a learned flow, and the same flow followed with smaller steps. |
| `evolution-strategies` | **Evolution strategies.** Training without gradients: random nudges, kept in proportion to how much each one helped. |

**Generative families compared.** An autoencoder, a VAE, a GAN and DEC trained on the same digits: objectives, codes, samples.

| name | shows |
| --- | --- |
| `generative-objectives` | **Four families, four objectives.** What an autoencoder, a VAE, a GAN and DEC are each trained to do with the same parts on the same digits, and what each ends up good at. |
| `generative-latent-spaces` | **Four code spaces, one noise plane.** Where held-out digits sit in the two-number code space of an autoencoder, a VAE, DEC and IDEC, and what a GAN draws across its noise plane, since it cannot place a real digit there. |
| `generative-sampling` | **New digits from four families.** New digits drawn from each family, and how recognisable, how varied and how far from the training digits they are. |
| `generative-reconstruction` | **There and back: rebuilding a digit.** The same held-out digits put through each family and brought back, by a search of its noise for the GAN, which has no encoder, and what each one loses. |
| `generative-interpolation` | **Walking between two digits.** A straight walk between two digits in each family's code space, the images on the way, and how recognisable and how digit-like the middle is compared with simply fading one image into the other. |
| `vae-kl-term` | **The KL term of a VAE.** One digit followed through a VAE, what the KL half of its loss measures, and the cloud of codes settling into the bell's rings as training goes on. |
| `dec-centroids-training` | **DEC: the centroids move.** K-means places ten centres in an autoencoder's code space, then DEC pulls the codes and the centres into tight, separate clusters, which are not the same thing as the right ones. |
| `entropy-and-gini-by-hand` | **Entropy and Gini by hand.** How impure a tree's node is, worked out from its counts: entropy in bits and the Gini index, term by term, and what each one counts. |
| `entropy-down-a-tree` | **Entropy down a tree.** A tree grown by information gain spends the root's 3.32 bits question by question; read as a probability model, what is left is its loss in nats. |
| `weight-entropy` | **The entropy of a network's weights.** Every free parameter of an MNIST classifier, before and after training: as a pile of numbers its entropy goes up; the structure is in how the numbers are arranged. |
| `logits-loss-and-nats` | **Logits, loss and nats.** A classifier's loss in statistical terms: logits as log-odds, the loss of one digit in nats and bits, the mean as an estimate with an error, and the likelihood it is the negative log of. |
| `kl-divergence` | **Cross-entropy is entropy plus KL.** A classifier's loss is a cross-entropy: the target's own entropy, which no model can beat, plus the KL divergence, the only part training moves; KL measured on three targets, both ways round, and as a fit to two humps of digits. |
| `every-loss-is-a-likelihood` | **Every loss is a likelihood.** Chi-square, squared error and cross-entropy are each minus the log of a likelihood: a line fitted to measured MNIST ink, a softmax regression of ten digits and a logistic regression of two, measured on held-out digits. |
| `from-a-cut-to-a-tree` | **From a cut to a tree.** An analysis cut is a tree of depth one; asking again in each half gives a tree whose boxes approach the likelihood ratio, until it grows deep enough to fit noise. |
| `dense-layer-learns-local` | **A dense layer learns to be local.** Training gives a dense layer's near pixels similar weights; a 3×3 convolution keeps only those near weights and shares them, using 576 weights instead of 50,176, and loses fewer held-out digits when they move. |
| `markov-chain-to-rnn` | **From a Markov chain to an RNN.** A table of counted contexts must have seen the exact context, so on held-out text it gets worse past a few characters; a small recurrent network carries a state and shares its parameters across contexts, and beats the best table. |
| `labels-against-rewards` | **Labels against rewards.** The same classifier trained from labels and from a right-or-wrong reward: the same direction on average, at most a bit per digit, and several times the updates. |
| `bits-in-a-weight` | **How many bits a weight carries.** A trained weight stored in 32 bits carries far fewer: rounded to each number format, the entropy of its values against the accuracy kept, and why plain 4-bit needs a shared scale. |
| `counting-weights-dnn-and-cnn` | **Counting the weights: a DNN and a CNN.** How many numbers a dense network and a convolutional network learn, counted layer by layer beside the PyTorch line that makes each layer: 52,650 for the MNIST MLP and 6,810 for the small CNN, nearly all of them in one dense layer. |
| `counting-weights-transformer` | **Counting the weights: a transformer.** One rule, one part at a time, counts every weight of a transformer: 2,304 in the small ViT's attention, whatever the number of heads, 5,066 in the whole ViT, 7,087,872 in a GPT-2 block and 124,439,808 in GPT-2 small, whose output layer reuses the token table. |
| `blocks-a-tensor` | **A tensor is a block of numbers.** One digit is a tensor: 784 numbers in a block of shape [1, 28, 28], from −0.42 on the paper to 2.82 in the darkest ink. |
| `blocks-a-batch` | **A batch: many digits in one block.** 64 digits slide into one block of shape [64, 1, 28, 28] and their labels into a row of 64: 50,176 numbers handed to the model at once. |
| `blocks-flatten` | **Flatten: a square becomes a strip.** nn.Flatten unrolls each 28 × 28 digit row by row into a strip of 784, so [64, 1, 28, 28] becomes [64, 784], and it learns nothing. |
| `blocks-a-linear-layer` | **nn.Linear: a grid of weights.** One output of nn.Linear(784, 64) is the digit's 784 numbers times one row of weights, added up, plus a bias: 8.09 − 0.02 = 8.07. |
| `blocks-relu` | **nn.ReLU: negatives become 0.** nn.ReLU turns every negative number into 0 and learns nothing: 48% of the first batch's hidden numbers and 49% over all 60,000 digits. |
| `blocks-a-model-is-a-stack` | **A model is a stack of blocks.** nn.Sequential stacks Flatten, Linear and ReLU into the 784-64-32-10 digit classifier, whose three Linear blocks learn all 52,650 numbers. |
| `blocks-conv2d` | **nn.Conv2d: one small stamp, used everywhere.** One 3 × 3 stamp of 9 weights slides over all 676 places of a digit; 4 stamps make 4 sheets, [64, 4, 26, 26], from just 40 learned numbers. |
| `blocks-pooling` | **nn.MaxPool2d: keep the biggest.** nn.MaxPool2d(2) keeps the biggest of each 2 × 2 square (1.34 of −1.00, 0.37, 0.47 and 1.34), so 26 × 26 sheets become 13 × 13, a quarter of the numbers, and it learns nothing. |
| `blocks-dropout` | **nn.Dropout: switched off at random.** nn.Dropout(0.2) switches 1 in 5 numbers off at random while training (15 of 64 at the first step) and none after model.eval(), which, measured, gets 94.35% of held-out digits right against 91.90% when it is forgotten. |
| `blocks-normalisation` | **BatchNorm and LayerNorm.** nn.BatchNorm1d gives every column of the batch mean 0 and spread 1, nn.LayerNorm every row: the same block, averaged in two directions. |
| `blocks-embedding` | **nn.Embedding: a table you look up.** nn.Embedding(10, 16) is a table of 160 numbers. A label picks its row by lookup, not multiplication, so the batch's 64 labels become a [64, 16] block. |
| `blocks-attention` | **nn.MultiheadAttention as blocks.** Attention as blocks on 5 patches of a real digit: Q, K and V from three weight grids, each head's grid of scores whose rows add up to 1, and an output that mixes V's rows. |
| `blocks-logits-and-softmax` | **Logits and softmax.** The model's 10 logits become bars that add up to 1, and nn.CrossEntropyLoss takes the logits: −ln 0.74 = 0.30, where a softmax in front would keep the loss above 1.46. |
| `blocks-which-is-which` | **Which block is which.** Eleven blocks, each with its line of code, its input and output shapes and its weight count, guessed and then answered: 6 of the 11 learn weights, from 40 in a Conv2d to 50,240 in nn.Linear(784, 64). |
| `blocks-the-data-pipeline` | **The data pipeline.** MNIST, ToTensor (÷ 255) and Normalize (the real mean 0.1307 and spread 0.3081 of all 47 million pixels), then a shuffling DataLoader: 938 batches on a conveyor belt make one epoch. |
| `blocks-forward-pass` | **The forward pass.** logits = model(x): one batch of 64 digits through the six blocks in order, its shape changing at each, from [64, 1, 28, 28] to the logits' [64, 10]. |
| `blocks-the-loss` | **The loss.** loss = loss_fn(logits, y): one loss per digit, −ln of the chance given to the right label, and their mean, 2.322, close to ln 10 = 2.303 for an untrained model. |
| `blocks-backward` | **loss.backward().** loss.backward() puts red gradient tiles behind every weight and bias, from the last layer back to the first: 52,650 of them, each .grad the shape of its weight. |
| `blocks-zero-grad` | **optimizer.zero_grad().** optimizer.zero_grad() empties the red gradient tiles before each backward; the real run that forgets it piles them up to 45 times their size and gets 10% of test digits right, against 95%. |
| `blocks-optimizer-step` | **optimizer.step().** optimizer.step() moves every weight by 0.1 times its gradient, against it: 0.1521 becomes 0.1566, and the same batch's loss drops from 2.3223 to 2.2616. |
| `blocks-learning-rate` | **The learning rate.** From the same start, lr=0.001 crawls (55% right), lr=0.1 learns (95%) and lr=0.8 bounces (27%): one number decides how a run ends. |
| `blocks-adam` | **Adam keeps two more blocks.** Adam keeps an m and a v for every weight, three times the numbers, and moves each weight by about its learning rate: one weight's update worked with real numbers, and the real Adam and SGD runs. |
| `blocks-the-training-loop` | **The training loop.** for x, y in loader as a circular track: one batch goes once round the five lines of a training step, then 938 times, while the step counter and the real loss curve draw. |
| `blocks-epochs-and-batches` | **Epochs and batches.** 60,000 digits ÷ 64 make 938 steps, one epoch; the loader shuffles again each epoch; the loss per step jumps while the loss per epoch falls, 0.315, 0.135, 0.098. |
| `blocks-evaluation` | **Testing the model.** model.eval() and torch.no_grad(): no gradients are made, and the held-out digits are counted one by one, 9,494 of 10,000 right (0.9494). |
| `blocks-save-and-load` | **Saving and loading.** torch.save(model.state_dict()): 52,650 numbers in six named tensors, packed into a file and loaded back exactly, with the same answer on 10,000 of 10,000 test digits. |
| `blocks-one-step-slow-motion` | **One training step, slowly.** One recorded training step, station by station: a batch, the forward pass, a loss of 2.28, 52,650 gradients, and every weight moved, so the loss on that batch falls to 2.23. |
| `blocks-one-epoch-fast` | **One epoch, fast.** One epoch of the course's MNIST classifier sped up: the first layer's weights turn from noise into strokes while the loss falls from 2.32 to 0.16 and the held-out accuracy rises from 7.6% to 94.9%. |
| `blocks-the-whole-program` | **The whole program.** The whole training script, 63 lines, scrolled through a part at a time beside a picture in which each part (device, data, batches, model, loss, optimizer, loop, test, save) lights up as its lines run. |
| `blocks-common-mistakes` | **Common mistakes.** Four common mistakes, each run for real: forgetting zero_grad (10.1% held-out accuracy), forgetting model.eval() with dropout (91.9% instead of 94.4%), a softmax before the loss (85.0%, the loss stuck above 1.46), and no Flatten (torch's shape error, in plain words). |
| `blocks-cnn-the-model` | **The CNN, block by block.** The course's program with a CNN for its model: eight blocks, their shapes, and 5,258 numbers to learn instead of 52,650. |
| `blocks-cnn-shapes` | **Where the 400 comes from.** Why the CNN's last layer reads 400 numbers: 28 → 26 → 13 → 11 → 5, and 16 × 5 × 5. |
| `blocks-cnn-forward` | **One digit through the CNN.** One digit through the trained CNN, block by block: the maps each block makes, and the scores at the end. |
| `blocks-cnn-backward` | **A shared stamp's gradient.** A shared stamp's gradient is the sum of one small part for every place it sat on every digit: 43,264 parts, adding up to PyTorch's number. |
| `blocks-cnn-same-loop` | **The same training loop.** Change the model, keep everything else: the same lines train the CNN, on the same batches, to a lower loss. |
| `blocks-cnn-stamps-learn` | **The stamps learn.** The CNN's first eight stamps from random to trained, a row per step, and the maps they make of one digit before and after. |
| `blocks-cnn-against-the-mlp` | **The CNN against the MLP.** The CNN against the MLP after one epoch, one run each: 5,258 numbers against 52,650, 96.4% held-out accuracy against 94.9%, and which digits each misses. |
| `blocks-cnn-what-it-gets-wrong` | **What the CNN gets wrong.** The 364 test digits the CNN misses: most are 8s, most often read as 9s, and most of them are digits a person reads easily. |
| `blocks-cnn-common-mistakes` | **CNN mistakes.** Three common CNN mistakes and the error each one gives, in plain words: the wrong in_features (400 arrive, not 784), the wrong number of channels (3 for a grey digit), and one digit without its batch; then the fix, x[0:1]. |
| `padding-maps-shrink` | **Why maps shrink.** Why a convolution's maps come out smaller than its input, and why it reads the border less. |
| `padding-a-ring` | **A ring of padding.** padding=1: a ring round the picture so the maps keep their size, though the border is still read less. |
| `padding-same-and-valid` | **Same and valid.** "same" keeps the size by padding (k − 1) ÷ 2; "valid", the default, loses k − 1. |
| `padding-modes` | **What goes in the border.** Four ways to fill the border: zeros, a mirror, the edge repeated, or the other side. |
| `padding-deep-stacks` | **Deep stacks need padding.** Without padding, a deep stack of stamps runs out of picture. |
| `padding-in-the-cnn` | **The CNN with padding, measured.** The CNN with padding=1 against the CNN without, measured on the same digits: better, but not from padding alone, and no help with moved digits. |
| `padding-a-batch-is-a-block` | **A batch is one block.** A batch is one full block of numbers with no holes, so things of different sizes are padded to one size first. |
| `padding-lines-of-text` | **Padding lines of text.** Lines of different lengths padded with id 0 to the longest, and the mask that marks the real characters. |
| `padding-masks` | **Masks: ignore the padding.** The pads must be ignored: a plain mean shrinks by length ÷ 50 (0.08 for "All:"), while masked means, key_padding_mask (a weight of exactly 0 on every pad) and ignore_index (163 targets counted, not 400) leave them out. |
| `padding-sort-by-length` | **Sort by length, pad less.** Batching lines of similar length wastes almost no padding: 38% of the cells are padding in file order or shuffled, 0.16% sorted by length, and the sorted batches are then shuffled whole. |
| `vector-db-training` | **Training vectors for search.** Pairs of two drawings of one digit train an encoder so that matching digits get nearby vectors. |
| `vector-db-store-and-search` | **Store, then search.** A vector database: a table of vectors with ids and labels beside it, searched by dot products. |
| `vector-db-ivf` | **An index: look in fewer places.** An inverted-file index searches only the lists nearest the query: 4 of 64 lists find 99.5% of the true 10 nearest while comparing 8.1% of the vectors. |
| `vector-db-versions` | **A new encoder needs a new store.** Vectors from two encoders don't mix: a retrained encoder's queries against the old store find the right digit 7.6% of the time, below chance, until the whole store is embedded again (96.6%). |

**Label embeddings.** A class label as a learned vector: in a classifier, in an autoencoder, and at generation.

| name | shows |
| --- | --- |
| `label-embedding` | **A label as a vector.** A class label as a one-hot marker, where every label is equally far from every other, against a row of a learned table, where digits that look alike end up close. |
| `label-embedding-classifier` | **Label vectors in a classifier.** A classifier's last layer is already a table of label vectors; learning it barely changes accuracy but decides which labels point the same way. |
| `label-embedding-autoencoder` | **Telling the decoder the label.** Hand an autoencoder's decoder the label's row beside the code, and a small code stops spending itself on which digit it is. |
| `label-embedding-generation` | **Change the label, keep the style.** Keep an image's code and swap the label row the decoder is given, and the same handwriting is drawn as another digit, as long as the code is small. |

**Randomness and entropy.** Where random numbers come from, what a seed decides, and how unpredictability is measured.

| name | shows |
| --- | --- |
| `pseudo-random-generators` | **Pseudo-random numbers.** A generator is a small state and a fixed rule that walks a closed path; bad rules leave patterns in their numbers and better scrambles hide them. |
| `seeds-and-streams` | **Seeds and streams.** The same seed always gives the same numbers; neighbouring seeds can give related streams unless the seed is scrambled first, and parallel workers need streams of their own. |
| `sampling-a-distribution` | **Sampling a distribution.** Evenly spread random numbers are turned into draws from any distribution, the draws match it better the more there are, and a bad source spoils which draw follows which. |
| `entropy-sources` | **Where randomness comes from.** A generator only stretches its seed, so the unpredictability must be gathered from the machine, cleaned, mixed and only then stretched. |
| `measuring-entropy` | **Measuring entropy.** Entropy estimated from a source's output can see lopsided counts and simple patterns, but a generator can score perfectly on everything while holding only as much as its seed. |
| `markov-chains` | **A Markov chain from real text.** Counting which character follows which in Shakespeare gives a table of chances; a walker hopping by that table settles into fixed proportions, forgets where it started, and writes better text the more characters it remembers, until its contexts become too rare. |
| `seeds-and-training-outcomes` | **One network, many seeds.** The same network trained on the same data from many seeds ends at different accuracies and makes different mistakes; each source of randomness alone causes much of that, and telling two nearly equal settings apart takes several seeds of each. |
| `gpt2-seeds-and-sampled-text` | **One prompt, many seeds.** A language model gives the same chances for the next word every time; the seed supplies the number that picks one, so different seeds write different texts, sooner and more wildly as temperature rises, while cutting off the unlikely tail keeps the variety without the junk. |

<!-- course groups: generated from next_catalog.json -->

**Probability: frequentist and Bayesian.** Two readings of the same measurement, and the distributions every count and average follows.

| name | shows |
| --- | --- |
| `frequentist-and-bayesian` | **Two readings of one count.** One small count read twice, as the range of true rates for which that count would be unsurprising and as an updated belief about the rate: where the two answers agree, where they part, and what each one claims. |
| `bayes-theorem-areas` | **Bayes' theorem as areas.** A population drawn as a square, split by how common something is and by how a test comes out, so the chance that a positive result is right is one rectangle compared with two: for a rare disease, and for picking out one kind of particle. |
| `prior-washing-out` | **A prior washing out.** Three people with different starting beliefs about a coin watch the same flips and end up agreeing, while someone who ruled the truth out in advance never can. |
| `interval-coverage` | **What coverage means.** Many repeated experiments each draw their error bar around a value we secretly know: a 68% interval is a recipe that works 68% of the time, and the usual quick error bars on small counts do not. |
| `binomial-poisson-gaussian` | **Binomial, Poisson, Gaussian.** The count of events in one bin, followed as one histogram that turns from the many-rare-chances shape into the bell curve, with a look at the far tails, where the bell curve stays badly wrong. |
| `central-limit-theorem` | **The central limit theorem.** Averages of dice rolls, then of lopsided decay times, pile up into the same bell shape, narrowing as more are averaged; except for one kind of distribution whose average never settles at all. |

**Minimisation and chi-square fitting.** Fitting a model to data points with errors, and how a minimiser finds the best fit.

| name | shows |
| --- | --- |
| `chi-square-fit` | **Chi-square as a landscape.** A straight line through a few points: each point pulls on the line like a spring, the pulls add up to chi-square, and that sum is a bowl whose bottom is the best fit and whose width gives the errors. |
| `chi-square-per-dof` | **What a good chi-square looks like.** The same points fitted with a model that is too simple, the right one and one with too many parameters: what chi-square should be, how the fit probability is read off its distribution, and what wrong error bars do to it. |
| `minimisers-on-chi-square` | **How a minimiser walks.** Three ways of walking downhill on the same chi-square surface, counting every evaluation, on a surface with a second, wrong bottom that traps the starts on its side. |
| `binned-vs-unbinned-fits` | **Binned or unbinned?** One sample of decay times fitted three ways: why a chi-square fit to a histogram with few events per bin gives a biased lifetime while likelihood fits do not, and what binning costs in precision. |

**Maximum likelihood and errors.** Where a best-fit value and its error come from.

| name | shows |
| --- | --- |
| `likelihood-curve` | **The likelihood curve.** Each measured decay time gets a probability from the model; multiply them, slide the lifetime, and the product peaks at the best estimate. |
| `likelihood-intervals` | **Errors from the likelihood.** The error on a fitted value is read off where the log-likelihood has dropped by one half, and with few events that interval is lopsided. |
| `error-propagation` | **Error propagation by cloud.** A cloud of possible measurements is pushed through a formula; the usual error formula matches the cloud only while the function is nearly straight over the errors. |
| `error-on-the-mean` | **The error on the mean.** Repeat an experiment many times and its average scatters far less than single measurements do, shrinking as one over the square root of how many you take. |
| `bootstrap-errors` | **The bootstrap.** With one dataset and no formula, draw from the data itself over and over; how much the recomputed answer varies is the error. |

**Correlated errors and systematic uncertainties.** What changes when errors are shared between parameters or measurements.

| name | shows |
| --- | --- |
| `error-ellipse` | **The error ellipse.** Fitting a line gives two numbers whose errors are tied together, and the tilt of their cloud says how. |
| `statistical-vs-systematic` | **Statistical against systematic error.** More data shrinks the scatter of an average but not a miscalibration shared by every event, unless the miscalibration is itself measured. |
| `nuisance-parameters-profiling` | **Profiling a nuisance parameter.** When a background is only roughly known, pretending it is exact gives an error bar that is too small; profiling it gives an honest one. |
| `combining-measurements` | **Combining measurements.** Two results that share a source of error must not be averaged as if independent; the best combination can even lie outside both. |

**Data selection and significance.** Choosing events, and what the choice does to signal, background and the shape of the data.

| name | shows |
| --- | --- |
| `moving-a-cut` | **Moving a cut.** A cut slides across a signal and a background histogram, and the counts, purity and significance follow it, with each measure of best picking a different cut. |
| `cuts-vs-multivariate` | **Cuts against a combined variable.** Two thresholds, one on each of two correlated variables, compared with a single diagonal line through both, and then with a curved boundary where no line will do. |
| `sideband-background` | **Background from the sidebands.** The background under a peak is estimated by fitting the spectrum on either side of it, and the estimate goes wrong when the assumed shape is wrong or the sides contain signal. |
| `selection-sculpting` | **How a selection sculpts a spectrum.** A trigger threshold and a detector's acceptance turn a smoothly falling spectrum into one with a peak, and a cut on momentum does the same to a mass spectrum. |

**Hypothesis tests.** Deciding between two hypotheses, and the two ways of being wrong.

| name | shows |
| --- | --- |
| `type-i-and-type-ii-errors` | **Two ways for a test to be wrong.** A counting experiment under two hypotheses and a threshold on the count: the two kinds of error as the parts of each distribution on the wrong side, traded against each other as the threshold slides. |
| `p-values` | **What a p-value is, and is not.** One measured result set against the distribution it would have if there were nothing to find, then why a small p-value is not the chance that there is nothing to find. |
| `neyman-pearson` | **The likelihood ratio is the best test.** Three ways to turn two measured values into one test, cut to the same false-alarm rate: the likelihood ratio catches the most signal, and no other direction of cut does better. |
| `look-elsewhere-effect` | **The look-elsewhere effect.** A spectrum with nothing in it still shows a striking excess somewhere when enough places are searched; many such spectra show how often, and what a local significance is worth after the search. |

**Discoveries, exclusion and upper limits.** Claiming a signal, or saying how large a signal could be hiding.

| name | shows |
| --- | --- |
| `bump-hunt` | **A bump hunt, from fit to global significance.** One mass spectrum followed from the background fit to a bump's local significance, and then to what that significance is worth once every mass has been searched. |
| `poisson-upper-limits` | **An upper limit from counting.** Seeing few or no events still rules out large signals: where the famous 2.3 comes from, and what the 90% promises. |
| `cls-method` | **The CLs method.** When fewer events turn up than the background alone predicts, the plain limit rules out signals the experiment could never see; dividing by a second tail area repairs that, at a price. |
| `brazil-band` | **The bands of a limit plot.** The green and yellow bands show the range of limits an experiment would set if there were nothing to find, built here one pseudo-experiment at a time and then compared with an observed limit. |
| `why-five-sigma` | **Why five sigma.** As data accumulate a real signal keeps growing while chance bumps rise and fade, and with many places and moments to look, three sigma turns up routinely and five almost never. |

**Machine learning in a physics analysis.** Networks and trees used the way an analysis uses them: to estimate a quantity or to select events.

| name | shows |
| --- | --- |
| `network-regression` | **Regression with a network.** A network with one unsquashed output learns to give back the angle a digit was turned by, and its errors show its resolution and bias. |
| `classifier-score-as-variable` | **A classifier's score as a variable.** A network's score separates signal from background; a cut on it trades efficiency for purity, and its shape can be fitted for how much signal a sample holds. |
| `boosted-regression-trees` | **Boosted trees for regression.** Small trees are added one at a time, each fitted to what the sum so far still gets wrong, first on one measured number and then on a whole detector read-out. |
| `overtraining-check` | **The overtraining check.** The scores of the events a classifier was trained on are laid over those of events it never saw; where they part, it has learnt its sample rather than the rule. |

**Anomalies, collapse and smooth backgrounds.** An autoencoder as a detector of the unexpected, a GAN that forgets, and a Gaussian process as a background model.

| name | shows |
| --- | --- |
| `autoencoder-anomaly-detection` | **Anomaly detection with an autoencoder.** An autoencoder trained without one kind of digit rebuilds that kind badly, and the size of the error flags what it never saw. |
| `gan-mode-collapse` | **Mode collapse in a GAN.** A generator that settles on a few digits instead of all ten, counted by a separate classifier, and how telling both networks the label brings the digits back. |
| `gp-background-fit` | **A Gaussian-process background.** A fixed function leaves a wave that looks like signal; a Gaussian process follows the background instead, but a short length scale swallows a real bump. |
| `anomaly-loss-and-latent` | **Two places an anomaly can show.** One picture through an autoencoder gets two scores, how badly it is rebuilt and how far its code lies from the training codes, and an unusual input can show in both, in one or in neither. |
| `anomaly-two-score-flow` | **An anomaly search with two scores.** The whole procedure from training on normal data to a decision: a density fitted to the codes, two thresholds set on normal data, three ways of combining the two scores, and what a second cut costs in false alarms. |

**Decision trees, forests and boosting.** From one split to bagging, forests, boosting and the workings of XGBoost.

| name | shows |
| --- | --- |
| `tree-splits-and-impurity` | **Choosing a split.** How a tree node picks its question: every threshold on every feature is scored by how much purer it leaves the two sides, and the best one wins. |
| `tree-pruning-and-depth` | **Holding a tree back.** A tree grown until every leaf is pure has learnt its sample; depth limits, leaf sizes, a minimum gain and pruning cut it back to what holds up on new events. |
| `tree-instability` | **One tree is unstable.** Resample the same training events and a single tree changes its questions and its answers; deep trees are right on average but scatter, which is why many are used. |
| `bagging` | **Bagging.** Many trees, each grown on a resample of the data, vote: the wobble of single trees averages away, down to the part they all share. |
| `random-forest` | **A random forest.** Bagging with one more trick: each split may only look at a random handful of features, so the trees differ usefully; then which features the forest relies on. |
| `adaboost` | **AdaBoost: weights on the mistakes.** Stumps are fitted one after another, each to events re-weighted so that the last one's mistakes count for more, and their weighted vote carves up the plane. |
| `boosting-shrinkage-and-early-stopping` | **Gradient boosting: the settings.** Each new tree is fitted to how strongly the loss pulls every event towards its label; small steps, a share of the rows, the right depth and stopping when validation loss turns make it work. |
| `xgboost-split-gain` | **How XGBoost scores a split.** Every event carries a slope and a curvature of its loss; a leaf's best value and a split's worth are simple sums of them, read off the bottoms of parabolas. |
| `xgboost-histograms-and-missing-values` | **XGBoost's histograms.** Why it is fast: each feature is cut into a few bins once, splits come from bin totals, a child's totals come by subtraction, and missing values go to whichever side gains more. |
| `xgboost-one-round` | **One round of XGBoost.** One boosting round followed for real events from the current predictions to a new tree and back, then many rounds, the knobs that restrain it, and four other ways to use trees. |

<!-- end of course groups -->

The nine groups above follow a course for physics undergraduates: probability, fitting,
likelihood and errors, systematics, selection, hypothesis tests, discovery and limits, and the
machine learning an analysis uses. They teach with pseudo-experiments, the standard method in
physics: seeded draws from a model stated on screen, with the true values shown beside the data
and the estimate, and the exact answer drawn against the pseudo-experiments wherever one exists.
Every phase carries a tag saying which it shows (exact, pseudo-experiments, or measured on
MNIST). The machine-learning ones use real handwritten digits as stand-in events and say so,
naming the physics counterpart.

Three earlier groups were also added after the hundred. The randomness group writes its small
generators out in integer arithmetic so every number can be followed, labels each phase with
the kind of evidence it shows, and in two animations (`entropy-sources`, `measuring-entropy`)
measures the machine it is built on: timer jitter and the operating system's bytes. Those
numbers differ on every machine, so those two do not reproduce pixel for pixel elsewhere.
Of the other two groups, the five family comparisons share one set of
trained models (`study_families.models()`: an autoencoder, a VAE, a GAN, DEC and IDEC with the
same decoder shape, on the same 20,000 digits, three seeds, at code sizes 2 and 16). The four on
label embeddings share one set of label-conditioned autoencoders (`study_labels.py`).


## Broadcast masters: 8K, HDR

```bash
python3 -m voxanim master conv2d                              # 8K60 HDR10 (PQ) HEVC
python3 -m voxanim master conv2d --hdr hlg                    # 8K60 HLG (broadcast)
python3 -m voxanim master conv2d --res 4k --codec prores      # 4K ProRes 4444 XQ
python3 -m voxanim master conv2d --seconds 2                  # a quick test clip
python3 -m voxanim still conv2d -t 12.75 --res 8k --hdr pq    # 16-bit PQ PNG + SDR preview
python3 -m voxanim qa                                         # safe-area / collision checks
```

Masters land in `build/masters/`. Frames stream from the workers into ffmpeg
through shared memory, so no frames are written to disk: one 8K 16-bit frame
is about 200 MB.

- **Resolution:** 1080p, 4K or 8K (7680×4320), any frame rate (default 60).
  Everything is resolution-independent: geometry, line widths, text,
  reflection blur and motion-blur sampling are defined against a 960-wide
  reference, and the tests check that a 2× render downsampled matches 1×.
  Above 4K supersampling drops to 1×: sprites, lines and glass are
  anti-aliased analytically, so a pixel at 8K is too small for edges between
  primitives to show.
- **Motion blur:** at 8K, sub-frames are added until nothing moves more than
  2 output pixels per sub-frame (cap 64), so fast kernel swaps smear
  smoothly. Soft features (the accumulate wave) are allowed coarser steps.
- **HDR (`hdr.py`, ITU-R BT.2100):**
  - SDR diffuse white is placed at 203 cd/m² (BT.2408 graphics white), so
    every colour looks as designed.
  - A separate emission layer (glows, glints, the white
    multiply/accumulate wave) adds up to 600 cd/m² of highlight above that,
    to a 1000 cd/m² peak.
  - The SDR picture is unchanged by this, and the tests check it.
  - BT.709 colours are converted colorimetrically to BT.2020 (BT.2087).
  - PQ (SMPTE ST 2084, HDR10) or HLG (ARIB STD-B67).
- **Encoding:**
  - HEVC Main10 4:2:0, tagged BT.2020 / PQ or HLG, limited range. PQ adds
    HDR10 SEI: a BT.2020 D65 mastering display, 1000 / 0.005 cd/m²,
    MaxCLL = the grade's 1000-nit ceiling, and MaxFALL measured densely
    plus 10%.
  - ProRes 4444 XQ (10-bit 4:4:4) and 10-bit AV1 are also available.
  - SDR web outputs get a ±1 LSB triangular dither against banding.
- **Typography (`layout.py`):**
  - Type roles (title, subtitle, caption, note, key number) have fixed
    sizes and inks chosen for contrast against the backdrop: captions are
    about 7:1, titles 11:1.
  - No text is drawn smaller than 2.9% of picture height (about 16 px at
    540p, 125 px at 8K).
  - Each glyph gets a faint halo in the backdrop colour, so it stays
    crisp over particles and lines.
  - Titles shrink to clear the step indicator, and subtitles break at
    `\n` and then wrap.
  - QA requires 4 px of clear space between any two pieces of text and
    flags text crossed by a cage or frame line. In the network
    animations it also flags any caption something solid ever covers.
- **Safe areas:** titles, subtitles (wrapped) and the step indicator sit in
  the 5% title-safe area. Compositions are scaled (`layout.ZOOM`) into
  action-safe. Captions fade when geometry passes under them.
  `python3 -m voxanim qa` checks every animation at many instants, and the
  tests run it.
- **Cost on this machine** (6-core i7, 32 GB): an 8K frame takes about 7 s
  per sub-frame per worker, with 4 workers fitting in RAM alongside x265.
  conv2d (58 s at 60 fps) takes several hours. 4K is about 4× cheaper.

## Reproducibility

A render is a pure function of the commit, the reference environment and the
inputs below. Each of them is pinned:

- **Code:** each frame is a pure function of time (`scene(t)`), with no
  randomness that isn't seeded. Every seed is a named constant.
- **Environment:** `requirements-lock.txt` gives the exact versions the
  release was made with (Python 3.13.5, numpy 2.5.4, Pillow 13.0.0, ffmpeg
  9.0.2 on macOS 15 arm64).
- **Fonts:** Inter and DejaVu Sans Mono are bundled with the package
  (`voxanim/fonts/`, with their licences) and checked against their SHA-256
  when loaded. Text is laid out with Pillow's basic engine, so it doesn't
  depend on optional system libraries.
- **Data:** MNIST and GPT-2 small are downloaded once and verified against
  pinned SHA-256s before use. GPT-2 comes from a fixed Hugging Face revision.
- **Trained weights:** every model is trained by seeded numpy code and
  cached under a name that hashes *all* of its settings (`cache.cached_npz`),
  so a changed hyperparameter can never load stale weights.
  `models/mnist_dnn_100x3.npz` (the default model for `dnn-eval`) is rebuilt
  byte for byte by `python3 -m voxanim.model_io --make-default`.
- **Encoding:** ffmpeg runs with bit-exact flags, no metadata and a fixed
  number of encoder threads (one for GIF and WebM; four for the MP4, as
  `render.X264_THREADS` and every manifest record: x264 is repeatable for a
  fixed thread count but gives other bytes for another one), so the encoded
  bytes depend only on the frames, the encoder version and that count. The
  MP4s of releases up to 1.4.0 were encoded with one thread.
- **Manifests:** each `render` writes `build/out/<name>.json` next to its
  outputs. It records the options, the render style, every tool version, the
  git commit and the SHA-256 of each file.

Floating-point training can differ in the last bits on another CPU or BLAS
library, and Adam can amplify that. A retrained model may then differ
slightly from the cached one the release used. To reproduce a release
exactly, use the reference environment, or unpack the release's trained
weights (`voxanim-weights.tar.gz`, about 450 MB) into `build/.cache/` first.
The renders use exactly those weights.

## Acceleration

`pip install -e .[gpu]` adds PyTorch, and with it a second way to run the
expensive work: on an Apple GPU (Metal) or an NVIDIA one (CUDA). It is
chosen by `VOXANIM_DEVICE`:

| `VOXANIM_DEVICE` | what runs where |
| --- | --- |
| `auto` (default) | the fastest measured path (below) |
| `mps`, `cuda`, `cpu` | PyTorch on that device for everything that has a port |
| `numpy` | the original code, bit for bit (as without PyTorch installed) |

### Rendering: one Metal kernel per frame

On an Apple GPU the voxels, lines and quads of a frame are composited by one
Metal kernel (`raster_metal.py`). The `Framebuffer` the renderer draws into
only records each primitive; a voxel is a reference to its sprite images,
which are uploaded to the GPU once and reused frame after frame. At resolve
time every primitive is listed under the 16 × 16 tiles it touches, in the
order it was drawn, and one thread per pixel walks its tile's list doing
exactly what the numpy `Framebuffer` does: the depth test, "over" on
colour, coverage and HDR emission, the z-write. The order of the primitives
at a pixel is the order the CPU would have drawn them in, so it is the same
image, and with fused multiply-adds and fast math switched off it is the same
arithmetic:

- **Voxels and lines** come out **bit-identical** to the numpy renderer's.
  A voxel's depth is its sprite's float32 depth plus a float64 offset; Metal
  has no float64, so the kernel carries the offset as two float32s and adds
  it exactly (`tests/test_accel.py` has a sprite that float32 alone would
  wrongly show).
- **Quads** compute their coverage in float64 on the CPU and numpy blends
  them in float64 too; the kernel blends in float32, so pixels under a quad
  can differ by a float32 rounding step (at most 1.8×10⁻⁷).
- **Lit voxels** (a glow, a white flash, a reflection's fade) are lit in the
  kernel from the pooled atlas channels, with the float32 constants numpy
  would use, so they stay exact too; the floor's reflection pass is drawn by
  the kernel as well (its blur stays numpy).
- **Lines** below 4K are rasterised by numpy's float64 formula, but all of a
  frame's lines at once in one vectorised pass (bit for bit the same pixels),
  and only in the 16 px tiles a line can reach: a tile further from the
  segment than the line's half-width plus half its diagonal has coverage
  exactly 0 throughout, and a pixel of coverage 0 changes nothing.
- **Lines are cached**: 76–100% of a frame's lines are the same as the
  frame before (cages, axes), and a line is a pure function of its projected
  endpoints, depths, width, colour, opacity and bias, so its rasterised
  pieces stay in the GPU pool, keyed on exactly those values, and are reused
  (1.1–1.2× on a frame).
- **Diagram quads are cached** the same way: the diagram shaders (discs,
  rounded rectangles, images, GPT-2's panels) carry a key of everything
  their pixels depend on (centre, size, colours, opacity, a hash of any
  texture), so a quad that has not changed is drawn from the GPU pool
  instead of being shaded again on the CPU (1.8–4.0× on a diagram frame).
- **Every voxel is lit by the kernel**: a plain voxel's tint,
  min(D·colour + W, A), is the lit-sprite formula with no fade, flash or glow
  (W × 1.0 is exact), so no tint is made on the CPU or uploaded.
- **One upload of each kind per frame**, and asynchronous: the primitive
  table, the tile lists and the per-frame images travel in one integer and
  one float buffer, copied without waiting for the GPU to finish what came
  before (the host arrays are held until the frame is read back) -- a
  motion-blurred frame's sub-frames no longer each stop the CPU
  (1.25–1.4× on such frames).
- **Voxels go to the GPU in blocks**: a scene keeps its voxels as arrays
  (`Scene.voxel_batch`; `Block.emit` adds a whole block at once), and the
  renderer sorts them with one stable `argsort` and turns each run into rows
  of the kernel's table with array arithmetic -- the same phase, tint and
  float32 constants as one at a time (1.2–1.5× on a voxel-heavy frame).
- **The resolve** (the supersampled frame box-filtered down) is a Metal
  kernel too, adding the samples in numpy's order, so the same bits.
- **From 4K up** lines are rasterised in the kernel instead, in float32 (an 8K
  frame's lines cost seconds on the CPU), so 8K frames differ from numpy's
  by at most one 8-bit level on some line pixels (568 pixels over four 8K
  frames); below that the numpy code rasterises them, so 960×540 stays exact.
- Motion blur's sub-frames are composited over the floor and summed on the
  GPU, and come back once per frame.

Checked over every animation (`python3 -m voxanim gpucheck`, three frames
each): **349 of 354 frames identical in the 8-bit output that is encoded**,
the other five differing in 6 pixels by one level; 247 identical even in
float32. Measured on an M5 Max:

| | numpy | Metal kernel |
| --- | --- | --- |
| an 8K frame of `conv2d` (one with heavy motion blur) | 31.8 s | **3.2 s (9.8×)** |
| 8K frames of `conv2d` and `gpt2-transformer-block` (`gpucheck --scale 8`) | | **3.3× and 1.8×** |
| a 4K frame of `conv2d` | 10.1 s | **1.9 s (5.3×)** |
| a 960×540 frame, warm (`conv2d`, `mlp-9-training-step`) | 162, 168 ms | **69, 70 ms** |
| a 960×540 frame of a line-heavy diagram (`activation-smoothness`) | 624 ms | **71 ms** |
| `conv2d`, every frame at 960×540 (17 workers) | 62.8 s | **20.9 s (3.0×)** |
| six animations rendered and encoded (`render`, old code → new) | 223 s | **95 s (2.3×)** |
| twelve animations rendered and encoded (original code → now) | 582 s | **98–134 s (4.3–5.9×)**; 106–110 s with the machine at a load of 60–77 |

(The "numpy" column of the first rows is the renderer before this work; the
same machine was heavily loaded by other jobs throughout, so treat absolute
times as rough.) `gpucheck` measures a machine and exits 1 if any pixel is off by more
than one 8-bit level (`--tolerance`). On CUDA, or with `VOXANIM_DEVICE=cpu`,
the generic PyTorch framebuffer (`raster_torch.py`) is used, from 8K under
`auto` (it draws each sprite with its own handful of GPU commands, which only
pays for big frames); the Metal kernel has no CUDA twin yet.

### Training

A GPU only pays when each operation is big. Measured on an M5 Max:

| work | numpy | on the GPU (Metal) |
| --- | --- | --- |
| convolutional autoencoder, a training step | 47.7 ms | **5.2 ms (9.2×)** |
| MLP 784-1024-10, batch 1,024, 300 steps | 6.1 s | **2.8 s (2.2×)** |
| MLP 784-256-256-10, batch 64, 1,000 steps | **2.6 s** | 3.8 s |
| MLP 784-64-10, batch 64, 2,000 steps | **0.5 s** | 2.8 s |

So under `auto` the convolutional autoencoders train on the GPU and
`deepnet` runs do above 2×10⁸ multiply-adds per step (`accel.TRAIN_WORK`);
the batch-64 MLPs most animations train stay numpy, bit for bit. The dense
autoencoders and the recurrent models gain nothing (their operations are too
small or sequential).

What changes on a GPU is the rounding, not the experiment. Apple GPUs have no
float64, so training runs in float32, but the initial weights, the minibatch
order, the dropout masks and the augmentation are still drawn by the same
seeded numpy generators in the same order, and everything a run records is
measured by the numpy code from the weights at that step. Each port is
checked against the numpy model in float64 on the CPU
(`VOXANIM_FLOAT=64`, `tests/test_accel.py`): the gradients agree to 1e-15
and whole recorded training runs to 1e-10. In float32 a run lands on a
solution of the same quality but not the same weights: the convolutional
autoencoder's test error is 0.04436 against 0.04438.

### Earlier artefacts

Trained weights are cached by their settings, not by the device, so a cached
model (or one unpacked from a release's `voxanim-weights.tar.gz`) is used as
it is and only a missing one is trained on the GPU. A render's manifest
records what composited it (`"framebuffer"`: `numpy`, `metal-kernel/…` or
`torch-…/<device>`), and `VOXANIM_DEVICE=numpy` renders any release bit for
bit: re-rendering `regularised-autoencoder` with it gave the v1.4.0 MP4 byte
for byte.

Rendering is faster with or without a GPU, exactly (every frame of every
animation is bit-identical to the previous renderer's, and re-encoded media
are byte-identical):

- a scene's voxels are arrays, and everything that reads them (the
  renderer, the shadows, the reflection, the layout's text placement) reads
  the arrays instead of a Python object per voxel;
- the contact shadows' polygons go to Pillow as flat lists, and a block's
  per-cell geometry (`Block.chan_y` and friends) takes a fast path for one
  index (the same integer arithmetic, the same double multiply);
- motion blur measures how far things move from probe-only scene builds
  (the animation's drawing, which registers the probes, without turning
  sheets into voxels or laying out text): the same probes at every instant
  of every animation, so the same sub-frames;
- the 8-bit conversion's dither (the same seeded pattern every frame) is
  made once per frame size;
- the camera computes its projection matrices once instead of on every use
  (up to 40% of a frame went to rebuilding them; `conv2d`'s 17-worker render
  in numpy fell from 62.8 s to 31.3 s);
- text is measured and rasterised once per distinct caption (55% of a
  diagram frame before), and fully visible sprites skip their depth mask;
- `render` runs every animation's frames through one pool of workers: each
  imports PyTorch and compiles the Metal kernel once (2–3 s per animation
  before), and the workers flow from one animation into the next instead of
  idling at each one's end (about 15% on a dozen short animations);
- the camera remembers the points it has unprojected (a linear solve each;
  charts ask for the same ones every frame);
- up to six animations encode at once in the background, so the encoding
  backlog no longer trails the last frame on two threads;
- `render` encodes the MP4 and the GIF side by side (each ffmpeg is its own
  single-threaded process, so the bytes are unchanged), in the background
  while the next animation renders: encoding (13 s + 4 s per animation)
  used to run after the frames with every render worker idle.

The MP4s of the 88 animations inherited from v1.2.0 were encoded on an Intel
Mac; x264 there gives different bytes from x264 on Apple silicon, even at the
same ffmpeg version and flags (old and new code agree with each other here).
Their GIFs, and every frame, match.

CUDA uses the same PyTorch code as Metal for training and the generic
framebuffer, but has not been run: there is no NVIDIA GPU on the machine
this was built on or on GitHub's runners.

## Development

```bash
python3 -m pip install -r requirements-lock.txt
ruff check voxanim tests website tools   # lint (configuration in pyproject.toml)
python3 -m pytest tests                  # everything available offline
python3 -m voxanim prepare && python3 -m pytest tests   # the full suite
```

The tests:

- **`tests/test_models.py`:** gradients against finite differences, the model
  loaders, and known behaviour (GPT-2 predicts " four" after "One, two,
  three,").
- **`tests/test_animations.py`:** every animation builds and loops without a
  seam.
- **`tests/test_conv.py`:** convolution arithmetic and the schedule.
- **`tests/test_render.py`:** culling, determinism, HDR and resolution
  independence.
- **`tests/test_layout.py`:** no clipped or colliding text.
- **`tests/test_website.py`:** the website lists every animation exactly
  once, and the generated page is well formed.
- **`tests/test_landscape.py`, `test_maths_anims.py`, `test_deepnet.py`,
  `test_gpt2_anims.py`, `test_behaviour.py`, `test_seq2seq.py`,
  `test_gridworld.py`, `test_gp.py`, `test_modern.py`, `test_diffusion.py`,
  `test_pca.py`:** one file per family added in 1.2.0: every new gradient
  against finite differences, the update rules and formulas against hand
  computations, and every number the subtitles quote.
- **`tests/test_generalisation.py`, `test_activation.py`, `test_capacity.py`,
  `test_prune.py`, `test_transfer.py`, `test_lora.py`, `test_quant.py`:** the
  families of 1.3.0, the same way; their trained-outcome claims are relative
  (which network wins, by at least how much), because CI retrains every model
  on another CPU and exact numbers differ there in the last bits.
- **`tests/test_regul.py`:** the regularisation family, the same way.

A test that needs MNIST or GPT-2 is skipped, with the reason, until
`prepare` has run. CI (`.github/workflows/ci.yml`) runs the lint and the
offline tests on every push, plus a job that prepares MNIST and runs
everything except GPT-2.

The website (`website/`) is one static page generated from the preset
registry by `website/build_site.py`. `SECTIONS` there groups and titles the
animations. No media is in git, and no full video is on the site:

- A render also makes two small files of its MP4 (`voxanim/previews.py`): a
  640 px poster (`<media>.jpg`, about 20 KB) and a preview clip
  (`<media>-thumb.mp4`: the first half of the animation at 480 px and 10
  frames a second, holding its last frame for 2 s before it loops; about
  40 KB), and the same pair for the dark render (`<media>-dark.jpg`,
  `<media>-dark-thumb.mp4`). `python3 -m voxanim previews` makes them again
  from existing MP4s.
- A release carries them, with every manifest, in one file,
  `voxanim-site.tar.gz`. The site build downloads that archive from the
  latest release and nothing else, and runs no ffmpeg: for all 365 animations
  in both themes it is about 60 MB, against GitHub Pages' limit of 1 GB.
- Each card shows the poster and plays the preview of the page's theme, and
  swaps them when the "Dark mode" button is pressed; only the theme on show
  is downloaded. An animation with no dark preview keeps its light one. Under
  reduced motion the previews stay still. Dark mode follows the system or the
  button (remembered per browser).
- Opening a card plays the full video, streamed from a GitHub release
  (`https://github.com/<repo>-light/releases/download/<tag>/<media>.mp4`, or
  `<repo>-dark/.../<media>-dark.mp4` when the page is dark): each theme's
  videos are released from a repository of their own, which holds nothing
  else, and not copied into the Pages build. The player has a Light / Dark switch for
  the video on show, keeps its position when the theme changes, and offers
  both MP4s and the two GIF archives as downloads. A release serves its files
  as `application/octet-stream`, so the page states the type
  (`<source type="video/mp4">`); without it Safari refuses the video. If a
  stream fails the player asks once more at the same position (GitHub
  redirects each request to a temporary address, which a long pause can
  outlive), then shows a message with a download link.

```bash
python3 website/build_site.py _site --media build/out   # a local preview: no release, no ffmpeg
python3 -m http.server -d _site                          # http://localhost:8000
```

With `--media DIR` (a render folder, or one filled by
`tools/release_assets.py`) the player plays `DIR`'s own videos, and
animations `DIR` does not have are left off the page.

`.github/workflows/pages.yml` builds it and deploys it to GitHub Pages on
every push to `main` and every published release. It needs a release that
has `voxanim-site.tar.gz`: against an older one the build stops with a
message and the site that is already deployed stays as it is.

**Making a release:** render everything in both themes, gather the assets,
tag, publish.

```bash
python3 -m voxanim render all --theme both --jobs 8 --clean-frames   # hours; build/out/
python3 tools/release_assets.py OUT                    # a folder per repository, checked
git tag -a v1.1.0 -m "voxanim 1.1.0" && git push origin v1.1.0
gh release create v1.1.0 -R <repo>-light --title "voxanim 1.1.0, light" OUT/voxanim-light/*
gh release create v1.1.0 -R <repo>-dark --title "voxanim 1.1.0, dark" OUT/voxanim-dark/*
gh release create v1.1.0 --title "voxanim 1.1.0" --notes-file NOTES.md OUT/voxanim/*
```

The source's own release goes last: publishing it rebuilds the website,
which plays the videos of the other two. A release is three, one per
repository, each with a `SHA256SUMS` over its files:

- `<repo>-light` and `<repo>-dark`: every video of the theme as its own file,
  `<media>.mp4` or `<media>-dark.mp4` (what the site's player streams), and
  `voxanim-gifs-light.tar` or `voxanim-gifs-dark.tar`, every GIF of the
  theme. Those are stored, not compressed: gzip took 1.8% off a sample of
  twenty;
- this repository: `voxanim-site.tar.gz`, the posters and preview clips of
  both themes and every manifest (`<media>.json`, `<media>-dark.json`), and
  `voxanim-weights.tar.gz`, the trained models.

That is 367 files per theme for 365 animations; GitHub is understood to
allow 1,000 in a release, and one file per video, GIF and manifest in two
themes would be 2,190. `release_assets.py` prints the count against the cap. It refuses to
run while any animation lacks a render in either theme (`--light-only`
allows a release without dark renders; `--allow-partial` leaves out
animations that are not rendered at all, to try the packaging on a few), a
render disagrees with its manifest, or a poster or preview is missing or
older than its video; the archives are deterministic.
The website redeploys itself when the release is published, provided the
`github-pages` environment allows tags `v*` to deploy (that run's ref is the
tag; Settings → Environments → github-pages → deployment branches and tags).
Otherwise the deploy job fails without a log, and
`gh workflow run pages.yml --ref main` deploys the same site by hand.

Before changing a module, render a few `draft` frames of the animations that
use it and compare them afterwards. Every refactor in the history was checked
this way: three frames from each animation, plus 25 dense frames for larger
rewrites, all pixel-identical.

## How it works

```
voxanim/
  rendering
    camera.py      orthographic camera (2x3 projection; its null space is the view direction)
    sprites.py     ray-marched bevelled voxel, pre-filtered into sub-pixel phases
    raster.py      float framebuffer with z-buffer: sprites, shaded quads, AA lines
    scene.py       Scene (per-frame primitives) + Renderer (floor, passes, motion blur, text)
    fonts.py       the bundled fonts (checked by SHA-256) and text measurement
    hdr.py         BT.2100 HDR: PQ / HLG, BT.2020, light levels
    layout.py      titles, step indicator, safe areas, captions that yield
    color.py easing.py text.py   palettes; easing curves; number/symbol typography
    render.py      multi-process rendering, ffmpeg encoding, manifests, broadcast masters
    qa.py          layout QA over every animation
  acceleration (optional: PyTorch)
    accel.py       the device choice (VOXANIM_DEVICE), float32 Adam, the training loop on a GPU
    raster_metal.py   a frame composited by one Metal kernel (Apple GPUs)
    raster_torch.py   the framebuffer as PyTorch tensors (CUDA, or forced)
    deepnet_torch.py  deepnet's network and training loop on a GPU
    gpucheck.py    accelerated frames against numpy, frame by frame
  animation frameworks
    tensor.py      Block: an HWC tensor as voxels; group gaps, pitch, scale, culling
    conv.py        ConvSpec / ConvStage / ConvAnimation / CombineSpec: the conv choreography
    netanim.py     NetAnimation, Sheet, OutputPanel, value colours: the network animations' base
    diagram.py     flat node-and-line diagrams (camera-facing billboards)
  registry
    presets.py     the registry and the original conv2d set
    lessons.py     the convolution teaching set
    catalog.py     every network animation
  models (numpy) and their data
    cache.py       checksummed downloads; trained weights keyed on all their settings
    optim.py minibatch.py   the shared Adam and minibatch loop
    mnist.py       MNIST, the dense and convolutional classifiers
    strided.py vit.py sequence.py memory.py autoencoder.py deep_models.py   the other models
    landscape.py mlp_runs.py autodiff.py deepnet.py gpt2_probe.py seq2seq.py gridworld.py
    gp.py ssm.py mamba.py moe.py diffusion.py pca_kmeans.py                  (1.2.0)
    generalise.py actfun.py capacity.py prune.py transfer.py lora.py quant.py (1.3.0)
    regul.py regul_ae.py                                                      (1.4.0)
    w2v.py patch_embed.py posenc.py                                           embeddings
    model_io.py    load / save dense models (.npz, PyTorch, Keras, JSON)
    gpt2.py        GPT-2 small: safetensors reader, BPE tokenizer, forward pass
  animations
    mlp.py cnn.py attention.py recurrent.py rnn_batches.py gated_anim.py dnn.py
    ae_anim.py ae_training.py vae_anim.py unet_anim.py dec_anim.py
    gpt2_anim.py (+ gpt2_panel.py, gpt2_camera.py, gpt2_layout.py)
    gd_anim.py mlp_graph.py dtree.py
    optim_anim.py adamw_anim.py schedule_anim.py          optimisation
    autodiff_anim.py softmax_anim.py activations_anim.py  the maths, step by step
    overfit_anim.py init_anim.py residual_anim.py norm_anim.py   training deep networks
    bpe_anim.py embed_anim.py block_anim.py sampling_anim.py     GPT-2, tokens to sampling
    halluc_anim.py condition_anim.py calib_anim.py        what language models know
    seq2seq_anim.py qlearning_anim.py
    gp_anim.py gp_prior.py gp_posterior.py gp_kernels.py gp_hyper.py gp_bayesopt.py
    ssm_anim.py mamba_anim.py cost_anim.py moe_anim.py rope_anim.py   modern sequence models
    diffusion_anim.py pca_anim.py
    losschart.py sweep.py prob_bars.py                    shared drawing helpers
    ddescent_anim.py memorise_anim.py datasize_anim.py augment_anim.py minima_anim.py   generalisation
    actfun_race.py actfun_gradients.py actfun_sparsity.py actfun_smoothness.py actfun_depth.py
    bottleneck_anim.py rank_anim.py widthdepth_anim.py   capacity and bottlenecks
    prune_anim.py transfer_anim.py lora_anim.py           trimming, fine-tuning, LoRA
    quant_anim.py precision_anim.py                       quantization and precision
    gen_common.py actfun_plots.py                         shared drawing helpers (1.3.0)
    regul_geometry.py regul_weights.py regul_tools.py regul_race.py regul_ae_anim.py
                   regularisation (+ regul_plots.py, its drawing helpers)
    embed_lookup_anim.py w2v_anim.py sentence_anim.py patch_anim.py posblind_anim.py
    sinusoid_anim.py relbias_anim.py alibi_anim.py rope_ext_anim.py lenx_anim.py   embeddings
  __main__.py      the command line
```

- **Pure function of time.** `ConvAnimation.scene(t)` builds a fresh `Scene`
  for any instant. Frames render independently across processes, and motion
  blur is just more instants.
- **Voxels are sprites.** Under an orthographic camera every voxel looks the
  same, so one rounded box is ray-marched at 8× resolution and box-filtered
  into a 4×4 atlas of sub-pixel phases. Each phase stores coverage, diffuse,
  sheen and depth. Placing a voxel is one blit, and slow motion doesn't snap
  to pixels. Tinting is linear, so one atlas serves every colour. Atlases are
  cached in `build/.cache/`.
- **Depth.** Opaque voxels are drawn back to front *and* z-buffered, which
  gives clean anti-aliased edges with correct occlusion when things overlap.
  Lines (cage, receptive-field frame) and the frosted multiply/accumulate
  glass are depth-tested against them. So the frame on an interior window is
  hidden by the voxels in front of it, exactly as in the originals.
- **Floor.** Contact shadows plus a glossy reflection, rendered from a mirrored
  camera with its own sprite look (pale faces, bright seams), at output
  resolution. Geometry is composited premultiplied over it. That is exact
  because "over" is associative.
- **Motion blur.** The scene registers *probes* on moving things. Each frame
  takes just enough sub-frames to keep every probe's travel under a pixel
  budget: one sub-frame when nothing moves, up to 10 during a kernel swap.
- **Supersampling.** 2× by default (`max` is 3×). Sprites carry their own
  anti-aliasing, so this only settles edges between primitives.
- **Hidden voxels are culled.** A voxel is skipped when its three
  camera-facing neighbours are opaque; the tests check this changes no pixel.

### The choreography (`conv.py`)

Matched to the originals frame by frame:

- **Tempo:** about 1 s per step at first, cruising at about 0.2 s, slowing
  again for the last steps (`Tempo`).
- **Swaps:** the finished kernel flies to the "done" row while the queue
  advances and the next kernel takes the active slot. With groups the queue
  is a grid with one row per group.
- **Each step:**
  1. The receptive-field frame glides to the next window.
  2. A soft white band sweeps through the window and the kernel along the
     channel axis.
  3. The result lands in the cage with a glint.
- **Groups and depthwise:** the input explodes into channel groups, inactive
  groups are ghosted, and (`track_groups`) the input slides so the active
  group stays put.
- **Chained stages:** stage *n*'s output block is stage *n+1*'s input
  (depthwise-separable).
- **Loop:** after a hold, the output dissolves and the kernels fly home, so
  the GIF loops without a cut. Pass `reset=0` to `ConvAnimation` for a hard
  cut like the originals.

### Adding an animation

```python
@preset("lesson-5x5", "lesson-5x5")
def five_by_five():
    """Bigger receptive field."""
    return lesson("5x5", in_hw=(9, 9), in_ch=4, filters=2, kernel=(5, 5),
                  title="5×5 kernel", subtitle="9×9 becomes 5×5")
```

`lesson()` (in `lessons.py`) adds a title and shape labels. For the bare
original style, use `ConvAnimation(ConvSpec(...))` directly. Give every new
animation a title in `SECTIONS` in `website/build_site.py`; a test fails
until it has one.

`ConvSpec` options:

- **Geometry:**
  - `in_hw`, `in_ch`, `filters`, `kernel`, `stride`, `padding`
  - `dilation`: gaps between kernel taps
  - `input_dilation`: zeros between input pixels, i.e. transposed conv
  - `groups`
  - `in_d`, `kernel_d`, `stride_d`, `pad_d`: conv3d
  - `pool="max"|"avg"`
- **Numbers:** `data`, `weights`, and `styles` per role (`input`, `kernel`,
  `output`), each `"hue"`, `"diverging"` or `"gray"`.
- **Staging:** `layout` (screen fractions plus camera angle), `tempo`,
  `order="serpentine"`, `kernel_scale` (magnified kernels), and `labels`
  (dict overrides or `False`).
- **Chaining:** several specs make several stages. `source="first"` reuses
  the first input instead of the previous output. `CombineSpec` merges stage
  outputs cell by cell, e.g. the Sobel magnitude.

### Tuning the look

- **Shading:** `sprites.Look` holds the per-face colour factors and the bevel
  darkening and sheen.
- **Palette:** `color.py`. These are each voxel's *right-face* colour,
  sampled from the original GIFs.
- **Floor and blur:** `scene.Style`.
- **Value colours:** `color.styled`. Positive, negative and neutral colours
  are `color.POSITIVE`, `NEGATIVE` and `NEUTRAL`.

`compare` and `sheet` are the debugging loop: change a number, re-render a
frame next to the original, look.

## Licence and credits

- **Code:** MIT (see `LICENSE`).
- **Convolution animations:** the conv2d originals' look is reproduced from
  [Animated AI](https://animatedai.github.io) (MIT, Brad Klingensmith),
  where this project began.
- **Fonts:** Inter (SIL Open Font License 1.1) and DejaVu Sans Mono
  (Bitstream Vera licence). Their licence texts are in `voxanim/fonts/`.
- **Data:** MNIST (Yann LeCun, Corinna Cortes and Christopher Burges), via
  the Keras mirror. GPT-2 small (OpenAI, MIT), via the Hugging Face hub. Both
  are downloaded on first use; neither is included here.
