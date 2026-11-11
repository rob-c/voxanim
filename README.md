# voxanim

Animations that explain convolutions, neural networks and transformers,
rendered from **real numbers**: every weight, activation, gradient and
probability on screen comes from a model trained or run right here, in plain
numpy. 54 animations, from a 3×3 convolution to the inside of GPT-2, all from
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
python3 -m voxanim list --names                   # the 54 animations
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
weights (`voxanim-weights.tar.gz`, about 29 MB) into `build/.cache/` first.
The renders use exactly those weights.

## Development

```bash
python3 -m pip install -r requirements-lock.txt
ruff check voxanim tests website         # lint (configuration in pyproject.toml)
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

A test that needs MNIST or GPT-2 is skipped, with the reason, until
`prepare` has run. CI (`.github/workflows/ci.yml`) runs the lint and the
offline tests on every push, plus a job that prepares MNIST and runs
everything except GPT-2.

The website (`website/`) is one static page generated from the preset
registry by `website/build_site.py`. `SECTIONS` there groups and titles the
animations. The MP4s are not in git: the build takes them from the latest
release, or from a local render:

```bash
python3 website/build_site.py _site --media build/out   # needs ffmpeg
python3 -m http.server -d _site                          # http://localhost:8000
```

`.github/workflows/pages.yml` builds it and deploys it to GitHub Pages on
every push to `main` and every published release.

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
    strided.py vit.py sequence.py autoencoder.py deep_models.py   the other models
    model_io.py    load / save dense models (.npz, PyTorch, Keras, JSON)
    gpt2.py        GPT-2 small: safetensors reader, BPE tokenizer, forward pass
  animations
    mlp.py cnn.py attention.py recurrent.py rnn_batches.py dnn.py
    ae_anim.py ae_training.py vae_anim.py unet_anim.py dec_anim.py
    gpt2_anim.py (+ gpt2_panel.py, gpt2_camera.py, gpt2_layout.py)
    gd_anim.py mlp_graph.py dtree.py
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
