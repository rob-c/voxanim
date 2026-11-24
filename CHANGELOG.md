# Changelog

## 2.4.0

Thirty-three new animations (398 in all): finding and recognising faces, hosting a language
model with thinking and effort, resizing pictures with convolutions, an autoencoder wired into a
GAN, and Minuit against SGD; and a glossary page of 357 machine learning and particle physics
terms. The new animations and `vector-db-versions` are rendered; the rest are 2.3.0's.

- **Finding and recognising faces, 14 animations** (238–251), in two sections after "Vision
  models", on the Olivetti faces (AT&T Laboratories Cambridge, 1992–94: ten photos of each of
  40 people, taken for face-recognition research; `voxanim/olivetti.py` reads scikit-learn's
  file without scipy and matches `fetch_olivetti_faces` exactly). People 0–29 train; people
  30–39 are only tested on, and are referred to by number only. A detector (6,053 numbers)
  on made scenes: a box as four numbers and IoU, a 12 × 12 grid of guesses, the layers, two
  losses (weighting the rare face cells: AP 0.92 against 0.47 without), training, one box per
  face (NMS), precision and recall. A recogniser (22,304 numbers): a face as 4,096 numbers,
  as 32, triplets, verification (equal error rate 12.4% against 22.0% for raw pixels; the
  threshold chosen on the training people), identification (36 of 45, no better than raw
  pixels from one enrolled photo, said so), the whole pipeline (67% named right from the
  detector's boxes, 82% from the true boxes), and what 40 people cannot show.
  `examples/faces.py` is the real program; `voxanim/study_faces.py` its numpy twin, matched to
  torch by a test.
- **Hosting a language model, 11 animations** (252–262), for people who have never thought
  about servers: what is on a phone or laptop and what is on the server, a request is text
  (the whole conversation, every time), the model is a file of 124,439,808 numbers, tokens,
  one token at a time, the key-value cache, many people sharing one copy, why a conversation
  has a limit, a coding agent's loop, and what "thinking" and "effort" are.
  `examples/serve.py` is a real minimal server for GPT-2 (voxanim's numpy GPT-2), measured by
  `voxanim/study_hosting.py`, which runs the script itself; `examples/thinking.py` and
  `voxanim/study_thinking.py` train a tiny transformer whose effort token sets how many steps
  it may write: at chance with up to 6 steps, 465 of 468 with a step per bit. Counts, not
  speeds; no company, product or chip is named.
- **Resizing pictures with convolutions, 2 animations** (263–264), after "Convolution
  settings, one at a time": smaller with a 3 × 3 blur kernel and stride 2 (the matrix D,
  conv2d = D X Dᵀ; on a zone plate, false rings of RMS 0.354 when skipping, 0.079 when blurring
  first) and bigger with conv_transpose2d (bilinear: inside the border 0.00129 against 0.00302
  for nearest over 400 photos; the edge receives half, and is made even).
  `examples/resample.py` and `voxanim/study_resample.py`, matched to torch by a test.
- **From autoencoder to GAN, 4 animations** (265–268), after "Autoencoders and latent codes":
  an autoencoder's two blocks wired again as a GAN (the decoder, weights and all, becomes the
  generator; the encoder's body becomes the judge, with one new 1-number head), then measured
  against it: grey (blurred) pixels 9.1% from the decoder before the GAN game and 7.4% after
  (real digits 6.0%, autoencoder rebuilds 9.4%); the autoencoder rebuilds a given digit (92.8%
  read right), the GAN cannot be given one; one loss against two players.
  `examples/ae_gan.py` and `voxanim/study_aegan.py`, matched to torch by a test.
- **A glossary page** (`glossary.html`, linked from the top bar and the introduction): 357
  terms and acronyms from machine learning and from particle physics (HEP) data analysis,
  each with what it stands for and one plain sentence on what it is and what it is for, in 15
  topics, searchable, each linked to the animation that shows it where there is one
  (`website/glossary.json`; a test checks every link and every line). Meant to make someone
  fluent between the two fields: an "Also called" column gives each field's name for the same
  thing (an MLP is often a "DNN" in HEP, overfitting "overtraining", recall "signal
  efficiency", a fit "training", a minimiser an "optimizer", −log L "cross-entropy"); the words
  the two use differently point at each other (inference, calibration, normalisation, bias,
  weight); "Fitting and minimisation" covers Minuit (MIGRAD, HESSE, MINOS, SIMPLEX, EDM, the
  error definition), the Hessian and covariance, Newton and quasi-Newton steps, line searches,
  finite differences against autodiff, and full-batch against mini-batch steps; the shared
  statistics (likelihood, NLL, estimators, pulls, goodness of fit, intervals, priors and MAP,
  Neyman–Pearson) and 55 HEP terms (MVA, working points, regions and sidebands, significance,
  CLs, nuisance parameters, template fits, luminosity, unfolding, scale factors, data/MC
  agreement, decorrelation and more) are in too.
- **Minuit and SGD, 2 animations** (269–270), after "Minimisation and chi-square fitting":
  every call iminuit's MIGRAD makes on a 3-parameter fit to made-up data, recorded from
  iminuit itself (`voxanim/minuit_calls.json`): it probes each parameter up and down before
  each step, overshoots once and comes back, and reaches the minimum in 65 calls (HESSE adds
  16 for the error bars); then SGD on the same fit, 3,000 small noisy steps on 8 of the 40
  points at a time, a drunken walk whose positions average out near Minuit's minimum, for
  24,000 single-point evaluations against MIGRAD's 2,600. `examples/minuit_vs_sgd.py` and
  `voxanim/study_minuit.py`; a test reruns iminuit to check the record.
- Evidence tags can name their dataset (`blocks.measured(what, on=...)`).

## 2.3.0

Fourteen new animations (365 in all): padding, in a CNN and in a batch, and vector
databases; guide lines drawn in front in the Blocks series; and section headings that say
what is in them. The 55 Blocks animations (183–237) are rendered again; the rest are 2.2.0's.

- **Blocks and code: padding, 10 animations** (224–233). A CNN's padding: why maps shrink
  and the border is read less (corner 1, edge 3, middle 9; with padding=1, 4, 6, 9), a ring
  of padding, "same" and "valid", F.pad's four modes, deep stacks, and the course's CNN with
  padding=1 measured (97.6% against 96.4%, one run each; its last layer is bigger too). Then
  padding as the way to make one batch out of things that differ in size: four digits cut to
  their ink that torch.stack refuses until padded, lines of Tiny Shakespeare padded to the
  longest with their mask, masks that make the model ignore the pads, and sorting by length
  (38% of every batch is padding in file order, 0.16% sorted). `examples/padding.py` is the
  real program; `voxanim/study_padding.py` computes every number; a test runs the script's
  lines in torch and matches them.
- **Vector databases, 4 animations** (234–237), after "Retrieval and contrastive learning":
  an encoder trained with InfoNCE on pairs of drawings of the same digit, a store of the
  60,000 training digits searched by dot products (top-10 neighbours share the query's label
  96.0% of the time, as raw pixels do with 24 times the numbers), an inverted-file index
  (visiting 4 of 64 lists finds 99.5% of the true top 10 while scanning 8.1% of the store), and
  a retrained encoder's queries against the old store (7.6%, below chance, until the store is
  embedded again). `examples/vector_db.py` is the real program; `voxanim/study_vectordb.py`
  its numpy twin, matched to torch by a test.
- **Guide lines in front.** In the Blocks series, arrows and callout lines were painted under
  any card or tile they crossed: lines and boxes are painted back to front, and a line's
  `bias` does not change that. Arrows are now lifted in front of every block shape
  (`Blocks.ARROW_LIFT`), and four hand-drawn guides with them: the head split in
  attention, the loss floor across the meter (twice), and the epoch means over the loss
  curve. The data's path down a model's stack stays behind its blocks, on purpose.
- **Section headings renamed** to say what is in them: "Lessons" is now "Convolution
  settings, one at a time"; also "Classical machine learning", "Gradients and softmax, by
  hand", "Gradient descent", "A trained classifier", "Vision models", "Autoencoders and latent
  codes", "GANs, flows and diffusion", "Networks as memories, rules and flows", "Anomalies
  and backgrounds", "Position encodings", "From labels to rewards" and "Pruning and
  distillation". The anchors (and so the links) are unchanged.

## 2.2.0

Forty-one new animations (351 in all): counting a model's weights, and a first part of the
course, "Blocks and code", for students who find this hard, ending with a CNN for the same
digits. Only the new animations are rendered; the rest are 2.1.0's renders.

- **Counting the weights**, in a deliberately simple style (one idea on screen at a time,
  the PyTorch line beside each layer, a running total):
  - `counting-weights-dnn-and-cnn`, beside "A dense layer learns to be local": the MNIST
    MLP's 52,650 numbers (95% in its first layer) and the small CNN's 6,810 (40 in its
    filters, for any picture size).
  - `counting-weights-transformer`, after "Through the transformer": the small ViT's
    attention is 2,304 for any number of heads, the whole ViT 5,066, a GPT-2 block
    7,087,872, and GPT-2 small 124,439,808, matching its released weights.
- **Blocks and code: 30 animations** (185–214), a new first part of the course for
  students who find this hard: each block of a PyTorch model, and each step of training
  an MNIST classifier, as coloured blocks beside the lines of code that make it, one idea
  at a time.
  - `voxanim/blocks.py`: one visual alphabet (data blue, weights orange, gradients red,
    loss violet, optimizer green, a learning-rate dial), the code panel and the
    training-loop map with "you are here".
  - `examples/mnist_blocks.py`: the real PyTorch program every line on screen is taken
    from. `voxanim/study_course.py` is its numpy twin. A test runs one SGD and one Adam
    step of the script in torch and matches the twin to about 1e-7. Run for real, the
    script reaches 95.0% after one epoch (the twin 94.9%).
  - Part 1: a tensor, a batch, Flatten, nn.Linear, ReLU, a model as a stack. Part 2:
    Conv2d, pooling, dropout, BatchNorm and LayerNorm, Embedding, attention, logits and
    softmax, and a recap. Part 3: the data pipeline, the forward pass, the loss,
    backward, zero_grad, step, the learning rate, Adam, the loop, epochs, evaluation,
    save and load. Part 4: one step slowly, one epoch fast, the whole program, and four
    common mistakes, each a real run (no zero_grad 10%, no eval() 91.9%, softmax twice
    85%, no Flatten's shape error).
- **Blocks and code: a CNN, 9 animations** (215–223): the same program with one block
  changed. `examples/mnist_cnn.py` is `mnist_blocks.py` with a small CNN for its model
  (Conv2d 8, pool, Conv2d 16, pool, Linear(16 * 5 * 5, 10): 5,258 numbers), and
  `voxanim/study_course_cnn.py` its numpy twin, trained on the same batches; a test runs the
  script's own lines in torch and matches one step and every block's output to 1e-5. One
  epoch each: 96.4% held out against the MLP's 94.9%, with ten times fewer numbers. The
  nine: the model block by block, where the 400 comes from, one digit through it, a shared
  stamp's gradient, the same training loop, the stamps learning, the CNN against the MLP,
  what it gets wrong, and three CNN mistakes with PyTorch's real errors.
  - The page has a "Start here: blocks and code" part ahead of the neural networks, and a
    "New to this?" link in its introduction.
- Fix: the captioned first animations (`conv2d` and the others) dropped `probes_only`, so
  the torch renderer's motion-blur measure failed on them. It surfaced once torch was
  installed and `tests/test_accel.py` ran.

## 2.1.0

Eleven new animations (310 in all), and a page that reads as one course without
forward references. Only the new animations and `cpu-and-gpu-training` are rendered
again; the rest are 2.0.0's renders.

- **Four animations on entropy, impurity and loss** (`study_info.py`, `mech_info*.py`), a
  new group of the catalogue and a section of the site:
  - `entropy-and-gini-by-hand`: a tree's nodes on all ten digits, entropy and the Gini
    index worked out term by term, and what they count: drawn pairs that differ (Gini),
    the questions of a Huffman code (entropy).
  - `entropy-down-a-tree`: a tree grown by information gain spends the root's 3.32 bits
    question by question; in nats the entropy left in its leaves is exactly its training
    loss, and its held-out loss is best at depth 10.
  - `weight-entropy`: all 52,650 free parameters of an MNIST classifier before and after
    training. Their histogram entropy goes up (4.57 to 5.23 bits each: they spread out);
    the structure is in their arrangement: neighbouring pixels' weights share 0.19 bits
    (from 0.003), and the first layer's spectral entropy falls from 5.94 to 5.48 bits.
  - `logits-loss-and-nats`: logits as log-odds, one digit's loss in nats and bits, the
    mean of 10,000 as an estimate with its standard error, the baselines (ln 10), the
    likelihood and its coding cost, and temperature.
- **Seven bridges between topics** (`study_bridge_*.py`, `mech_bridge_*.py`), each measured
  on MNIST or Tiny Shakespeare. Each opens the section it leads into and names what it
  comes from:
  - `kl-divergence` closes the entropy section. Cross-entropy = entropy + KL, adding up
    exactly over the test digits for one-hot, smoothed and teacher targets. Training
    moves only the KL part. Forward and reverse KL fit two humps of digits differently.
  - `every-loss-is-a-likelihood`: chi-square and the Gaussian likelihood peak at the
    same slope, and the curvature gives the error (±0.026, against a scatter of 0.028
    over 120 samples). A softmax regression's held-out cross-entropy (0.275 nats) is
    −ln L per digit. A logistic regression of 4s against 9s gives the binary loss.
  - `from-a-cut-to-a-tree` opens classical ML. The best s/√(s+b) cut and the Gini root
    are 0.02 px apart. Deeper trees approach the likelihood ratio (held-out AUC 0.958 at
    depth 5 against 0.962), then fit noise (0.857 at depth 27).
  - `dense-layer-learns-local` opens Convolutions. After training, neighbouring weights
    of a dense layer correlate at 0.48 (0.00 before). A 3×3 convolution needs 576
    weights against 50,176, and loses fewer held-out digits when they are shifted.
  - `markov-chain-to-rnn` opens Recurrent networks. Counted tables are best held out at
    order 4 (2.85 bits per character), then get worse. A numpy GRU reaches 2.68, its
    gain coming from contexts the table rarely or never saw.
  - `labels-against-rewards` opens reinforcement learning. REINFORCE's gradient is, on
    average, p(y) times the label's. Its rewards carry about 8× fewer bits, and it needs
    3.5–9.5× the updates to reach 90%.
  - `bits-in-a-weight` opens Quantization. The entropy of the trained weights in each
    format: FP8 carries 6.28 of its 8 bits with accuracy kept. Plain FP4 rounds 98% of
    the weights to 0. One scale per 32 weights rescues it (96.2%).
- **The site reads in order: nothing before what it rests on.**
  - The Gaussian processes follow the classical methods (kernels, fits and priors) instead
    of sitting between generative models and recurrent networks.
  - `entropy-and-gini-by-hand` opens the trees, ahead of the first split that uses it.
    `weight-entropy` joins Capacity, beside the rank of a layer.
  - The CNN training step and the strided CNN classifier wait for convolutions.
  - The position encodings, with `gpt2-rotary-embeddings`, form a section of their own
    after attention. Byte-pair encoding opens Embeddings.
  - Hardware reads as one ladder: number formats, then one device, its limits, and many
    devices.
  - `context-cost` opens the modern sequence models. `pca-and-kmeans` opens the
    autoencoders. The Hopfield memory leads into them.
- `cpu-and-gpu-training` gains a phase on precision (now eleven phases, about 200 s): the
  bits a number gets in FP64, FP32, BF16, FP16, FP8 and FP4, with 0.1 as each stores it
  and how far off it is (exact); a CPU built for 64-bit, doing 64-bit sums at full speed,
  against a graphics card built for 32-bit, doing 16-, 8- and 4-bit sums 4 to 16 times
  faster and 64-bit at 1/64 of that rate, below the CPU's (representative figures, labelled
  as such); and the trained classifier with its weights stored in each format (measured):
  97.3% held-out accuracy down to 8 bits, 23.2% in plain 4-bit, where 99% of its weights
  round to zero, and 97.2% with one shared scale per 32 weights.

## 2.0.0

Every animation is rendered again, in both themes.

- **Logo and corner mark** (`voxanim/brand.py`): a V built of nine voxels, its point lit.
  Every rendered frame carries it, small and in its own colours, with the name and the site's
  address in light grey, in its bottom-right corner, so videos, GIFs, stills and posters all say where they came
  from; `VOXANIM_BRAND=0` leaves it off. `tools/make_logo.py` writes the same mark as the
  website's `logo.svg`, `logo-dark.svg` and `logo-mark.svg` (the page's icon and header).
- **Dark mode for the renders** (`voxanim/theme.py`): the page, lines, panels and text are
  each turned dark as they are drawn, and the data is left alone: voxels and images (heat
  maps, digits) keep their light-mode colours, vivid colours stay as they are, and pale tints
  become deep tints of the same hue. Light-mode frames are unchanged. Twenty-one captions
  that named a shade ("black: the truth") use a word that fits the theme, and the plain mask
  grids of the attention animations turn dark with the page. `render`, `encode`, `still`, `sheet` and `compare` take `--theme light|dark`
  (`VOXANIM_THEME=dark` does the same), and `render` and `encode` take `--theme both`: the
  light pass, then the dark one, in one command. A dark render's files carry `-dark`
  (`<media>-dark.mp4`, `.gif`, `.json`; frames in `build/<name>-dark/`), so both themes sit
  together in `build/out`. Manifests record `theme`, and `frames` (the animation's length).
- **Captions on the nine convolution animations** (`conv2d` to `conv1d-row-sweep`): each now
  carries a title and one line saying what it shows, so a video saved on its own says what it
  is; the picture steps back a little to give the words a band of their own.
- **Previews made at render time** (`voxanim/previews.py`): a render also writes a 640 px
  poster (`<media>.jpg`, about 20 KB) and a small silent looping clip (`<media>-thumb.mp4`,
  480 px at 10 frames a second, about 60 KB) of its MP4, in each theme.
  `python3 -m voxanim previews` makes them again from existing MP4s without rendering. The
  480 px hover preview is gone.
- **Release layout** (`tools/release_assets.py`): the videos and GIFs are released from two
  repositories of their own, `voxanim-light` and `voxanim-dark`, which hold nothing else:
  every video as its own file (`<media>.mp4`, `<media>-dark.mp4`) and the theme's GIFs in
  one stored archive (`voxanim-gifs-light.tar`, `voxanim-gifs-dark.tar`). This repository's
  release keeps the posters, preview clips and every manifest in `voxanim-site.tar.gz`, and
  the weights archive. Each of the three has its own `SHA256SUMS`, and the tool fills a
  folder for each. 301 files per theme for 299 animations, where one theme was 899 of the
  1,000 a release is understood to hold; the tool prints the counts and refuses a release
  over the cap. It refuses
  an animation without both themes unless `--light-only`, and one without previews;
  `--allow-partial` packs only what is rendered. GIFs and manifests are no longer separate
  release files.
- **Website**: the logo, large and in colour, heads the page. The Pages build downloads `voxanim-site.tar.gz` and nothing else, and runs no
  ffmpeg (the workflow no longer installs it or caches media); no full video is in the Pages
  build. Cards show the poster and preview of the page's theme and swap when "Dark mode" is
  pressed, fetching only the theme on show. The player streams the full video from the
  release of the theme's repository, dark when the page is dark, with a Light / Dark switch for the video
  on show that keeps its position; a stream that fails is asked for once more, then replaced
  by a message with a download link. Downloads: the MP4 in either theme and the two GIF
  archives. `build_site.py --media DIR` previews from local files, with no release and no
  GitHub CLI, and leaves off what `DIR` does not have.
- `render --out DIR` and `encode --out DIR` now write there (encoded files and manifest in
  `DIR`, frames in `DIR/frames/`); the option was accepted and ignored before.

## 1.10.0

- `vae-kl-term`: one digit followed through a VAE, what the KL half of its loss measures, and the
  held-out codes, coloured by digit, at 14 points in training as the cloud settles into the
  prior's rings; against the same network with no KL term and with the term weighted 0.1 and 10.
- `dec-centroids-training`: k-means drawn round by round on an autoencoder's codes, then DEC
  followed through training with the ten centres leaving trails; separation and label purity
  measured side by side.
- Website: a paragraph under the headline says what the page covers and in what order.

## 1.9.0

- **Decision trees, forests and boosting**: ten animations in a new section (catalogue entries
  160 to 169; `mech_trees.py`, `mech_boost.py` and their `study_*.py`): choosing a split, holding
  a tree back, why one tree is unstable, bagging, a random forest, AdaBoost, the settings of
  gradient boosting, and three on how XGBoost works: how it scores a split, its histograms and
  missing values, and one round end to end. The XGBoost ones implement the published algorithm in
  numpy and say so on screen; no number comes from the library. All train on MNIST, with two
  confusable digits standing in for signal and background.

## 1.8.1

Website only; no animation changed.

- The page is ordered as one course in twelve parts, each announced by a heading: statistics for
  data analysis, trees and classical machine learning, neural networks from first principles,
  convolutions and vision, models in use, other ways to learn, autoencoders and generative
  models, Gaussian processes, sequences and recurrent networks, attention and transformers,
  large language models, and what comes after the transformer.
- Six animations changed section to make that order readable: the decision tree joins the
  classical methods, the recurrent networks get a section of their own, and the self-attention
  training step joins the transformers. No section link changed.
- The section bar is one line that scrolls sideways and follows the current section, with
  buttons at its ends to skip to the next section or back to the top of this one.
- A tag line above the headline.

## 1.8.0

- `cpu-and-gpu-training`: a longer animation (about three minutes) following a model's weights
  and a batch of digits between memory, the CPU and a GPU, with a small classifier against a
  larger autoencoder, and what the difference means for how long a real training run takes.
  Its speeds are representative round figures, not measurements, and every phase that shows one
  says so: the one explicit exception in the catalogue to the rule that results on screen are
  computed or measured (see TODO.md). Its learning curves are trained on MNIST.

## 1.7.0

- **Randomness and entropy**: eight animations in a new section (catalogue entries 110 to 117;
  `mech_random.py`, `mech_chance.py` and their `study_*.py`). `pseudo-random-generators`,
  `seeds-and-streams`, `sampling-a-distribution`, `entropy-sources`, `measuring-entropy`,
  `markov-chains`, `seeds-and-training-outcomes`, `gpt2-seeds-and-sampled-text`. The small
  generators are written out in integer arithmetic and checked against published values; each
  phase is labelled with its kind of evidence. `entropy-sources` and `measuring-entropy` measure
  timer jitter and the operating system's bytes on the machine that builds them, so their
  numbers and pixels differ between machines.
- **Statistics and machine learning for a physics analysis course**: 41 animations in nine new
  sections (catalogue entries 118 to 158; `mech_stats_*.py`, `mech_phys_*.py`, `mech_anomaly.py`
  and their `study_*.py`). Probability, frequentist and Bayesian (6); minimisation and
  chi-square fitting (4); maximum likelihood and errors (5); correlated errors and systematic
  uncertainties (4); data selection and significance (4); hypothesis tests (4); discoveries,
  exclusion and upper limits (5); machine learning in a physics analysis (4); anomalies,
  collapse and smooth backgrounds (5). The statistics is taught with seeded pseudo-experiments
  from a model stated on screen, against the exact answer wherever one exists; each phase is
  tagged with the kind of evidence it shows. The machine-learning ones train on MNIST, with
  digits standing in for events.

## 1.6.0

- **One hundred more animations** in fourteen new sections of the site (attention wiring,
  the transformer block, training language models, decoding, fine-tuning and alignment,
  interpretability, vision, generative models, classical machine learning, optimisation,
  retrieval, systems, reinforcement learning, memory and dynamics). Each draws a mechanism
  on a real instance and then the measurement it leads to (`mech.py`, `mech_*.py`); the
  measurements are plain-numpy experiments on MNIST or Tiny Shakespeare, or the pinned
  GPT-2 (`next_*.py`, `study_*.py`), with comparisons over three or more seeds. They render
  with `python3 -m voxanim render`, like the rest.
- **Nine more after those**, in two further sections. *Generative families compared*: an
  autoencoder, a VAE, a GAN, DEC and IDEC with the same decoder shape, trained on the same
  digits, compared on their objectives, code spaces, samples, reconstructions and walks between
  two digits. *Label embeddings*: a class label as a learned vector, in a digit classifier, in
  an autoencoder's decoder, and swapped at generation to redraw the same handwriting as another
  digit.
- The website generator leaves an animation off the page while the release it builds from
  has no video for it, so a push between releases neither breaks the site nor shows an
  empty clip. `--media DIR` still requires every video.
- 56 of the hundred's first experiments were redesigned after review (`study_*.py`): the
  outcome of each, including the ones that stayed null, is at the end of `TODO.md`.
- Loss-chart curves are drawn 1.6 times bolder (never under 2 px): thin curves read as grey
  beside the voxels. This changes the rendered pixels of every animation that draws a
  `LossChart`.

## 1.5.0

- **Ten animations on embeddings** (`w2v.py`, `patch_embed.py`, `posenc.py`
  and ten animation modules), in a new "Embeddings" section of the site:
  `embedding-lookup` (a row of a table, or one-hot × table, and the rows a
  training step touches), `word2vec-training` (skip-gram with negative
  sampling on a generated corpus, one step in numbers, the categories
  forming), `sentence-embeddings` (mean and SIF pooling, and why word order
  is lost), `patch-embeddings` (a ViT's first layer on MNIST: 94.3% with
  position vectors, 26.3% with the patches shuffled; 83.8% either way
  without), `gpt2-position-blindness` (GPT-2's first attention layer without
  its position table and mask gives a word the same vector in any order, to
  5×10⁻¹⁵), `sinusoidal-positions`, `relative-position-bias` (T5's buckets),
  `alibi`, `rope-context-extension` (position interpolation and NTK scaling)
  and the finale, `length-extrapolation`: six tiny transformers identical but
  for how they see position, trained on 32 symbols and tested on 128 (T5
  holds at 1.24 nats, ALiBi at 1.40, while learned, sinusoidal and RoPE climb
  to 2.8–3.6; the best possible is 1.19). Every gradient is checked against
  finite differences.
- CI keeps the models trained on synthetic data between runs.
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
