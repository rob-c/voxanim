# voxanim

Animations that explain convolutions, neural networks and transformers,
rendered from **real numbers**: every weight, activation, gradient and
probability on screen comes from a model trained or run right here, in plain
numpy. 113 animations, from a 3×3 convolution to the inside of GPT-2, all from
one small renderer (voxels, lines, textured quads and text, written with
numpy and Pillow).

Watch them all on the [website](https://rob-c.github.io/voxanim/). The
rendered MP4s and GIFs, with a SHA-256 manifest, are attached to the
[GitHub releases](../../releases).

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
python3 -m voxanim list --names                   # the 113 animations
python3 -m voxanim still conv2d -t 1.0 12.7       # single frames -> build/stills/
python3 -m voxanim render gradient-descent        # frames + MP4/GIF -> build/out/
python3 -m voxanim prepare                        # download MNIST + GPT-2, train every model
python3 -m voxanim render all --clean-frames      # everything (hours; see below)
python3 -m pytest tests                           # the test suite
```

`pip install -e .` installs the `voxanim` command too.

Other commands:

- `info <name>`: the timeline of one animation.
- `sheet <name>`: a contact sheet in `build/sheets/`.
- `qa [names]`: layout checks without rendering.
- `encode <name>`: re-encode frames a previous `render` left behind.
- `master <name>`: 4K/8K HDR broadcast masters.
- `compare <name>`: ours next to the original GIF, when the
  [animatedai.github.io](https://animatedai.github.io) media folder is at
  `$VOXANIM_MEDIA` (default `../media`).

Options for `still`, `render` and `encode`:

- `--quality draft|preview|final|max`: draft drops supersampling, motion
  blur and reflections, for quick iteration; final is 2× supersampled, max 3×.
- `--scale 2` or `--res 4k`: a larger render.
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
- **Encoding:** ffmpeg runs with bit-exact flags, no metadata and one
  encoder thread, so the encoded bytes depend only on the frames and the
  encoder version.
- **Manifests:** each `render` writes `build/out/<name>.json` next to its
  outputs. It records the options, the render style, every tool version, the
  git commit and the SHA-256 of each file.

Floating-point training can differ in the last bits on another CPU or BLAS
library, and Adam can amplify that. A retrained model may then differ
slightly from the cached one the release used. To reproduce a release
exactly, use the reference environment, or unpack the release's trained
weights (`voxanim-weights.tar.gz`, about 38 MB) into `build/.cache/` first.
The renders use exactly those weights.

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

A test that needs MNIST or GPT-2 is skipped, with the reason, until
`prepare` has run. CI (`.github/workflows/ci.yml`) runs the lint and the
offline tests on every push, plus a job that prepares MNIST and runs
everything except GPT-2.

The website (`website/`) is one static page generated from the preset
registry by `website/build_site.py`. `SECTIONS` there groups and titles the
animations. The MP4s are not in git: the build takes them from the latest
release, or from a local render, and makes three small files of each: a
poster, a 480 px preview at full frame rate, and a thumbnail (the first half
of the animation at 320 px and 12 frames a second, holding its last frame
for 2 s before it loops; under 100 KB). The grid plays the thumbnails all
the time, which costs a weak machine little; pointing at or focusing a clip
plays its sharper full-rate preview; opening it plays the full file. Under
reduced motion the thumbnails stay still. Dark mode follows the system or
the "Dark mode" button (remembered per browser).

```bash
python3 website/build_site.py _site --media build/out   # needs ffmpeg
python3 -m http.server -d _site                          # http://localhost:8000
```

`.github/workflows/pages.yml` builds it and deploys it to GitHub Pages on
every push to `main` and every published release.

**Making a release:** render everything, gather the assets, tag, publish.

```bash
python3 -m voxanim render all --jobs 8 --clean-frames   # hours; build/out/
python3 tools/release_assets.py /tmp/release           # renders + manifests, weights, SHA256SUMS
git tag -a v1.1.0 -m "voxanim 1.1.0" && git push origin v1.1.0
gh release create v1.1.0 --title "voxanim 1.1.0" --notes-file NOTES.md /tmp/release/*
```

`release_assets.py` refuses to run while any animation lacks a render or a
render disagrees with its manifest; the weights archive is deterministic.
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
