# TODO2: make the animation catalogue accessible to physicists

Revised 16 November 2026. This is a review and a plan: the catalogue's teaching problems for a
physicist new to machine learning, an audit of every animation, and 100 proposed new lessons.
The animations it audits were published in 1.6.0; the new lessons it proposes are not built.

## Where things stand

The first version of this document (15 November) audited 228 animations: the 128 released ones and
100 newer ones that were then drawn as four generic chart panels. Since then:

- **The 100 were rebuilt** as drawn mechanisms (`voxanim/mech.py`, `mech_*.py`), each showing its
  concept on a real instance before the measurement. The four-panel canvas (`study_anim.py`) is
  deleted. This did most of what the first audit asked of them.
- **56 of their experiments were redesigned** (`study_*.py`), with comparisons over three or more
  seeds. Five are still null and fifteen show their concept only partly; [TODO.md](TODO.md) ends
  with the outcome of each.
- **They are in the main workflow and on the website**, published in 1.6.0 with nine more
  (generative families compared, label embeddings) that this document does not audit.
- **The 128 released animations are unchanged**, apart from bolder loss-chart curves. Their audit
  below stands as written.
- **None of the 100 new lessons (N001–N100) is built.**

So the work this document now describes is, in order: the top-priority fixes to the 128, the
beginner layer that neither set has, the open points on the 100, and then the new lessons.

## Decisions needed before building more

1. **The data rule against the physics lessons.** The rule is that learning results come from
   MNIST, real text, or the pinned GPT-2; exact arithmetic is allowed and labelled. Of the 100
   proposed lessons, 38 can be built on real data, 23 are exact illustrations with no
   learning claim, and **39 need a simulated physical system with known truth**
   (Hamiltonian and Lagrangian networks, differential-equation surrogates, simulation-based
   inference, the physics capstones). Those 39 are the part of the plan most specific to
   physicists and cannot be built under the rule as it stands. Either add "simulation with known
   truth, labelled as a simulation on screen" as a fourth evidence category for this track, or
   drop them. The classification is [below](#the-proposals-against-the-data-rule).
2. **Rearranged real data.** Six of the redesigned experiments use real data deliberately
   rearranged (Shakespeare in capitals or backwards, a stretch read repeatedly, pixel columns
   shuffled across images, digits streamed by class). Several real-data substitutes proposed below
   lean on the same device (a watermark, saturation, a blur). If that is outside the rule, both
   lists shrink.
3. **Build time.** Several redesigned studies took three to five minutes uncached while fifteen
   were building at once; the three-minute limit has not been checked on an idle machine.

## Scope and evidence

- **The 128 released animations:** the first audit examined all 128 catalogue descriptions, their
  README text and site grouping, selected implementations, and one frame from each of 32 videos
  plus four full-resolution stills. It is a source audit with visual sampling, not a viewing of
  every film. On 16 November the ten source findings were re-checked against the code: nine hold
  as stated and one is a caution, not a defect (see the table). The per-animation rows were not
  re-checked one by one.
- **The 100 rebuilt animations:** every one was built, laid out and tested by its builder, and a
  contact sheet of four frames from each was looked at during the rebuild; 31 of the 56 redesigned
  ones were looked at again after redesign. The "still open" notes below come from the builders'
  own reports of what they were not satisfied with, and from the first audit's top-priority
  concerns checked against the rebuilt versions. No film was watched from start to finish.
- **No learner has been tested.** Statements about what will confuse a newcomer are judgments.

### Audience and priorities

Assume comfort with graphs, units, measurements and simple functions. Offer calculus, linear
algebra, probability and statistical mechanics as optional depth; do not assume all four. Assume
**no knowledge of Python, arrays, tensor shapes, batches, optimisers, tokens, GPU memory or
software APIs**. The first useful outcome is being able to explain and assess a model, not to
reproduce a training script.

| Priority | Meaning | Action |
|---|---|---|
| P0 | A likely false inference, an incorrect comparison, a hidden confound, or an encoding that can reverse the meaning. | Fix before using the clip as an introduction. It is not a claim that the implementation is wrong. |
| P1 | The idea is useful, but its mechanism or a prerequisite is missing. | Re-storyboard the beginner path; keep the detailed plots as a second layer. |
| P2 | Useful in the right sequence; naming, pacing or an advanced detour needs adjusting. | Improve labels and context, or move it later. |

The 128 carry **17 P0, 102 P1 and 9 P2**. New lessons are marked Starter, Bridge or Advanced for
teaching order (21, 35 and 44 of them).

## Changes with the greatest teaching benefit

### 1. Begin with an experiment and one prediction

- [ ] Use one recurring anchor example: a noisy spring calibration or decaying signal. Show the observations with units, a proposed function, its adjustable coefficients, a held-out measurement and a residual. Then introduce the words *features*, *parameters*, *training*, *loss* and *prediction* beside the objects they name.
- [ ] Follow one example through the entire pipeline before showing a grid of neurons, layers or alternatives. Identify the source of each number: observed, simulated, chosen, learned or derived.
- [ ] Keep physical parameters, such as a decay constant, distinct from neural-network weights. A network fitted to measurements does not automatically estimate a physical law.
- [ ] Explain the three spaces explicitly: physical/input space, learned feature space, and parameter space. A dot moving down a loss surface is a changing model, not a particle moving through the experimental apparatus.
- [ ] Label each clock: physical time, sequence position, solver step, optimization update and video time. They are not interchangeable. Likewise distinguish number of examples, number of features, network width, number of parameters and spatial dimension.


### 2. Keep the visual language truthful

The first audit listed nine faults of the four-panel canvas: unsigned bars for signed values,
charts without quantities or units, matrices without named axes, identifiers drawn as
intensities, unordered points that looked like trajectories, truncated labels and notes, clipped
images, and one pacing for every topic. That canvas is gone, and the rebuilt animations draw
signed quantities about a zero line, name their tokens and digits, and set their own phases. What
is left is that the kit does not *enforce* the rules:

- [ ] **Axis labels.** 112 of the 122 charts in `mech_*.py` are given axis labels; ten are not.
  Make `Mechanism.axes` require a quantity for each axis (or the word "dimensionless") and fix the ten.
- [ ] **A matrix names its rows and columns.** `Mechanism.grid` takes no axis names; most callers
  add a caption by hand. Give it row and column labels and a scale.
- [ ] **More than eight categories.** The palette has eight colours; `tsne-and-umap` adds grey and
  black for digits 8 and 9. Add a second encoding (shape or label) wherever classes exceed eight.
- [ ] **Direction of improvement** ("lower is better") beside every measured comparison; many have
  it, none is required to.
- [ ] **The same rules for the 128**, whose charts are drawn by `LossChart` and by hand.

### 3. Reframe comparisons as experiments

- [ ] Before a race, show what is held fixed: data split, parameter count, training tokens, initialization, optimization budget, wall time or hardware. These are different controls; equal parameter count does not imply equal compute.
- [ ] Put the measured task and scope on screen: synthetic grammar, composed MNIST, truncated GPT-2, analytic quadratic, simulated traffic or measured timing. Do not rely on metadata to disambiguate them.
- [ ] Present one controlled pair first. Move five-way optimizer, architecture or regularizer races into a follow-up view.
- [ ] Show seed-to-seed variation or explicitly label a single demonstration. Preserve null results and reversals; use a second clearly labelled experiment if the first does not illustrate the mechanism.
- [ ] Separate training loss, held-out loss, accuracy, calibration, physical observable error and resource cost. State precisely what “better” means for each plot.
- [ ] Distinguish an exact identity, a numerical check, an empirical tendency and a proposed explanation. Attention pictures, linear probes and saliency are not interchangeable with causal evidence.

### 4. Use physics bridges with their limits attached

| Useful bridge | Boundary that must remain visible |
|---|---|
| Curve fitting → supervised learning | A good interpolation need not identify the true physical mechanism. |
| Loss surface → potential landscape | Optimization steps are an algorithm; inertia, friction, temperature and equilibrium are not implied. SGD noise is not generally equilibrium thermal noise. |
| Convolution → linear filtering/Green-function intuition | A learned finite kernel need not represent a physical response; channels are not extra spatial coordinates. |
| Attention → data-dependent weighted averaging | A weight is a routing coefficient, not necessarily an explanation, interaction strength or conserved probability flow through the whole model. |
| Latent variables → effective coordinates | Coordinates need not be unique, disentangled, physical or sufficient for a downstream inference. |
| Normalization → change of scale/reference | Batch statistics, feature centering and physical nondimensionalization solve different problems. |
| Residual networks → repeated small updates | Depth becomes a time discretization only under specified scaling and architecture assumptions. |
| Statistical mechanics → energy-based models and sampling | A normalized probability needs a measure and partition function; a generic classifier loss is not automatically a thermodynamic energy. |
| Symmetry → inductive bias | A symmetry must apply to the observation process and target, not merely to an ideal underlying law. Detector acceptance can break it. |

### 5. Add a usable beginner layer

- [ ] Offer an introductory path, a physics-applications path and an architecture reference. Advanced GPU and language-model implementation details should not interrupt the first path.
- [ ] Give each film a plain-language question/title followed by the technical name, e.g. “Which past measurements can this prediction use? — causal attention.” Expand acronyms at their first use.
- [ ] Open with one sentence naming the input and output. End with one falsifiable takeaway and one important limitation. Offer a prediction question before revealing the result.
- [ ] Keep a persistent legend and use labels/shapes/line styles in addition to color. Verify readability at the actual review resolution and with color-vision simulations. Make axes, formulas and text available in the transcript.
- [ ] Avoid requiring terminal commands to understand or review a lesson. A local player can provide pause, replay, phase selection, text/math detail and a small glossary; no installation should be part of the conceptual explanation.
- [ ] Where a concept needs two minutes, split into a mechanism clip and an evidence clip or allow a longer version. Do not make a 53-second runtime the educational objective.


## Suggested order for learning and revision

| Stage | Question the learner should be able to answer | Existing material after redesign | New foundations/applications |
|---|---|---|---|
| 1. Measurements → a model | What goes in, what comes out, and what is adjusted? | `mlp-classifier`, `gradient-descent`, `overfitting` | N001–N010 |
| 2. Loss → a learning update | Which derivative is being used, and what changes after one update? | `autodiff-graph`, the nine MLP clips, `optimisers` | N021–N030 |
| 3. Evidence → justified predictions | What does a score or interval mean on new measurements? | GP family, `bias-variance`, `cross-validation`, `roc-and-calibration` | N011–N020, N031–N040 |
| 4. Structured data → structured models | What locality, ordering or symmetry is being assumed? | Convolution lessons, recurrence, embeddings, basic attention | N041–N050 |
| 5. Models of dynamics and fields | Is this a solver, a surrogate, or a learned law? | `neural-ode`, `ssm-recurrence`, `neural-cellular-automata` | N051–N070 |
| 6. Scientific inference and generation | Does the learned approximation preserve the inference/observable of interest? | Flows, diffusion, VAEs, calibration and generative examples | N071–N100 |
| Optional specialist branches | How do language generation, alignment, scaling or hardware change the design? | Transformer variants, decoding, RLHF/DPO, distributed training, roofline, compression | Introduce only after the relevant foundations above. |

Revision order: first resolve P0 representation/claim mismatches, then re-storyboard a coherent starter path, then broaden coverage. Retain advanced clips as a searchable reference. The following audit is not a demand to make every specialist animation self-contained for an absolute beginner.


## Audit of the 128 released animations

Each row names a registered preset. "Redesign" is a proposal, not work done. The last column says
what a learner should be able to tell apart, or what the film must not claim.

A separate finding from 16 November: at the full 120 samples, `voxanim qa` reports overlapping
text in twelve of the 128 (`gpt2-conditioning`, `gp-prior`, `context-cost`, `mixture-of-experts`,
`initialisation`, `gpt2-lora`, `bottleneck-sweep`, `information-bottleneck`, `mixed-precision`,
`learning-rate-finder-and-restarts`, `activation-smoothness`, `memorising-noise`). The test suite
samples each at 24 instants and passes. `gpt2-conditioning` was confirmed to predate the bolder
curves; the other eleven were not checked individually.

Scope of the first audit: the 128 presets in `voxanim.presets.REGISTRY` excluding `voxanim.next_catalog.ITEMS`. All 128 catalogue descriptions, their README explanations and website grouping were examined. Shared choreography and selected mathematical/data implementations were inspected, including `conv.py`, `lessons.py`, `mlp.py`, `autodiff_anim.py`, `softmax_anim.py`, `gpt2_anim.py`, `ddescent_anim.py`, `minima_anim.py`, `rank_anim.py`, `prune.py`, `prune_anim.py`, `norm_anim.py`, `ae_anim.py`, `vae_anim.py`, `cost_anim.py`, `calib_anim.py`, `posblind_anim.py`, `lenx_anim.py` and `rope_anim.py`.

Visual inspection sampled the 54%-duration frame of 32 local MP4s: `conv2d`, `conv2d-2-groups`, `lesson-conv3d-channels`, `lesson-transposed-conv`, `mlp-1-forward-layer1`, `mlp-5-backward-output`, `mlp-9-training-step`, `autodiff-graph`, `gpt2-softmax-cross-entropy`, `gpt2-attention`, `rnn-training-step`, `lstm-training-step`, `gated-memory`, `optimisers`, `normalisation`, `overfitting`, `double-descent`, `sharp-vs-flat-minima`, `rank-of-a-layer`, `structured-pruning`, `gp-posterior`, `gp-hyperparameters`, `diffusion`, `vae-dense-2d`, `q-learning`, `gpt2-calibration`, `gpt2-hallucination`, `mixture-of-experts`, `context-cost`, `quantization-aware-training`, `activation-race`, `regularisation-race`. Four existing full-resolution stills were also inspected: `embedding-lookup` at 35 s, `word2vec-training` at 28 s, `sentence-embeddings` at 20 s and `rope-context-extension` at 25 s. This is a catalogue/source audit with representative visual sampling, not a claim that every video was watched throughout. The other rows identify source-supported learning barriers and proposed improvements, not independently confirmed frame defects.

Every row is a proposed revision. Use the common P0/P1/P2 definitions above.

### Six highest-priority redesign packages

1. **Make the first convolution an explained measurement.** The sampled original `conv2d` and grouped clips have no explanatory labels. Start with one measured 2-D field, one numbered 3×3 stencil and one computed response; introduce multiple measured quantities/channels only afterward. Preserve physical coordinates separately from channel and filter indices. `conv.py` only computes input-dependent numerical colours when `data` and weights/pooling are supplied: distinguish structural choreography from numerical examples instead of implying every coloured block is measured data.
2. **Teach fitting before touring the neural network.** The MLP's 784→64→32→10 sheets and the XOR gradient-descent clip presuppose what a model, target and loss are. Lead with fitting a straight calibration curve, then one neuron with two measured inputs, then the existing real digit. Use the same small numerical example for forward calculation, chain rule, parameter change and held-out prediction.
3. **Separate performance quantities and experimental claims.** `double-descent`'s video/source correctly separates loss and error; its catalogue description incorrectly promises a classic U in test error. Keep the honest video result, fix the description and teach how a confidently wrong prediction raises cross-entropy without changing the wrong-answer count. `sharp-vs-flat-minima` changes learning rate and number of updates along with batch size; frame it as two measured runs, not a controlled proof about batch size.
4. **Separate a matrix's rank from the rank of its change.** `rank_anim.py` explicitly reports that neither measured weight matrix is nearly rank 8, yet ends with “LoRA relies on this” and an assertion about low-rank updates. Show the SVDs of both W and a separately measured fine-tuning ΔW. Low-rank compression of W and low-rank adaptation are distinct empirical questions. Also keep the untouched test set out of structured-pruning selection: `prune.py` currently ranks by energy on `xte`.
5. **Explain what uncertainty and randomness mean.** Distinguish a GP's latent-function interval from observation noise, a VAE's per-input latent distribution from its prior, and diffusion generation from recovery of the original noisy digit. Replace `vae_anim.py`'s guarantee-like “new digits that were never in the data” with “new samples from the trained generator”; sampling alone does not establish novelty or exclude memorisation.
6. **Make transformers answer a small information question.** Before flying through GPT-2's 12 heads and 12 layers, compute a weighted average of three value vectors with one query. Keep “causal mask” distinct from physical causality, token probability distinct from factual reliability, and an attention heatmap distinct from proof of the reason for a prediction. The existing single-token, 40-country calibration experiment needs its selection and sample-size limits visible.

### Source evidence for the highest-priority corrections

These are source findings, separate from the proposed teaching improvements. All ten were re-checked against the code on 16 November 2026: the first seven, the ninth and the tenth hold as stated; the eighth is a caution about how a diagram may be read, not a defect. Line numbers are those of that day's working tree. The first three are small, certain corrections and need no redesign: reword one catalogue description, one closing subtitle and two captions.

| Finding | Evidence |
| --- | --- |
| Structural convolution colours are not always computed data values. | [Default convolution factory](voxanim/presets.py), `presets.py:66–69`, constructs `ConvSpec()` without data or weights. [Numerical-colour condition](voxanim/conv.py), `conv.py:396–397`, sets values only when data and weights/pooling are provided. |
| Double-descent catalogue and measured result disagree about the metric. | [Catalogue](voxanim/catalog.py), `catalog.py:1045–1048`, says “Test error”; [measured-result explanation](voxanim/ddescent_anim.py), `ddescent_anim.py:14–21`, explicitly says the test-error curve has no U. The sampled video correctly labels separate loss and error panels. |
| Low-rank weights and low-rank fine-tuning changes are conflated in the conclusion. | [Rank lesson subtitles](voxanim/rank_anim.py), `rank_anim.py:137–153`, first reports neither matrix is nearly rank 8, then says “LoRA relies on this” about an unmeasured fine-tuning ΔW. |
| Test inputs influence the structured-pruning selection rule. | [Pruning experiment](voxanim/prune.py), `prune.py:392–404` and `421–425`, labels energy as `mean-square-test` and ranks `filter_energy(P, xte)` / `unit_energy(P, xte)` before evaluating on the same test split. This is input-dependent selection, even though the ranking does not use test labels. |
| The sharp/flat experiment changes multiple training conditions. | [Experiment definition](voxanim/minima_anim.py), `minima_anim.py:4–7` and `69–70`, gives 50,000 versus 400 updates and learning rates 0.001 versus 0.01. The source discloses these conditions; the teaching task is to make their inferential limit explicit. |
| VAE captions assert sample novelty that the demonstrated procedure does not test. | [VAE implementation](voxanim/vae_anim.py), `vae_anim.py:6–10` and `83–84`, says the space is filled without gaps and samples are “new digits that were never in the data.” The shown procedure samples a trained decoder; it does not perform a novelty or memorisation analysis. |
| Diffusion caption claims untested novelty. | [Diffusion captions](voxanim/diffusion_anim.py), `diffusion_anim.py:246`, says generated digits were not in the training set. No comparison to the training collection is part of that displayed result. |
| “All attention scores at once” is an idealised dependency-depth picture. | [Cost-model description](voxanim/cost_anim.py), `cost_anim.py:12–16`, says depth 1 for score computation; this should not be read as a constant-time bound for the complete attention operation or finite hardware. |
| Long-index extrapolation benchmark uses only short dependencies. | [Benchmark definition](voxanim/lenx_anim.py), `lenx_anim.py:3–7`, defines an order-2 Markov language trained at length 32 and tested at 128. It tests new absolute positions without requiring 128-token dependency recall. |
| GPT-2 is the learned-absolute-position comparison, not the RoPE model. | [Rotary lesson definition](voxanim/rope_anim.py), `rope_anim.py:1–8` and `23–27`, explicitly separates RoPE calculations from the real GPT-2 position table. |

### Per-preset proposals

#### Convolutions

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `conv2d` | P1 | Unlabelled 7×7×8 block, eight filters and accumulating colours require the viewer to know the encoding already. | Begin with one signed scalar field and calculate one numbered 3×3 response; then add channels and output filters. Label numerical versus structural examples. Check: which index is physical position, and which identifies a measured feature? |
| `conv2d-same-padding` | P1 | Preserving shape can look like inventing measured pixels at the boundary. | Compare the same edge response with no padding, zero padding and an explicitly stated boundary assumption. Check: which output values depend on assumed data outside the image? |
| `conv2d-stride-2` | P1 | The moving window demonstrates skipping but not what information is lost. | Synchronise stride 1 and stride 2 on the same field; mark response locations and show a narrow feature missed between samples. Check: is a stride-2 output generally enough to reconstruct the stride-1 output? |
| `conv2d-stride-2-same-padding` | P1 | Simultaneous stride and padding obscure why a 7×7 input becomes 4×4. | Freeze the seven original sites, add the two assumed border sites, then count four allowed window centres. Check: does “same” mean identical output size when stride is 2? |
| `conv2d-2-groups` | P1 | Splitting coloured blocks may suggest grouping spatial regions. | Start with a channel-to-filter connectivity matrix; remove cross-group connections before returning to blocks. Check: can a filter in group 1 read any channel in group 2? |
| `depthwise` | P1 | One filter per channel is difficult to distinguish from ordinary convolution by movement alone. | Use three differently labelled sensor maps; independently filter each and show the absence of a channel sum. Check: which stage can combine information from different sensors? |
| `depthwise-separable` | P1 | Two stages require keeping spatial filtering and channel mixing mentally separate. | Use a persistent two-line recipe: local filtering in each channel, then a weighted combination at one fixed position. Compare parameter counts with full convolution. Check: can every full convolution be represented by this restricted factorisation? |
| `conv1d-full-width-scroll` | P1 | Time runs down rows while an eight-channel block can look like a spatial image. | Begin with two labelled detector traces sharing a time axis; one three-time-point window produces four response traces. Check: which quantities are summed, and which time positions remain in the output? |
| `conv1d-row-sweep` | P1 | Alternating left/right animation may imply that the operation reverses the signal or its kernel on alternate rows. | Hold the mathematical orientation fixed, highlight index order and explain that the sweep direction is presentation order. Check: does reversing rendering order change the numerical response? |

#### Lessons

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `lesson-stride-1` | P2 | The labelled 9×9×4 example still introduces several channels before one scalar result is understood. | Show one channel and a numbered response first; retain this clip as the baseline for strides 2 and 3. Check: why are there seven valid window positions per axis? |
| `lesson-stride-2` | P1 | Shape changes are visible but disconnected from the sampling grid. | Overlay stride-1 response locations, then retain every second centre without changing the kernel. Check: why does nine become four rather than nine divided exactly by two? |
| `lesson-stride-3` | P1 | A third near-duplicate sweep consumes attention without making the general rule explicit. | Put all three stride results on one shared input after a short individual sweep; derive the endpoint-counting rule. Check: where does the last legal window start? |
| `lesson-dilation-2` | P1 | Spreading the taps can be mistaken for making the filter larger or adding parameters. | Keep nine numbered weights visible while inserting unsampled gaps; contrast receptive-field size with weight count. Check: how many values are multiplied despite a 5×5 footprint? |
| `lesson-dilation-3` | P1 | Another dilation sweep adds little intuition about the gaps' cost. | Compare dilation 1, 2 and 3 on a narrow feature between taps and a broad feature. Check: can a feature inside the footprint still be completely unsampled? |
| `lesson-transposed-conv` | P1 | “Upsampling: zeros then convolution” explains implementation but invites the false inference that it undoes convolution. | Begin with a small matrix A, draw its transpose's scatter-and-add operation, then connect to zero insertion and overlapping contributions. Check with numbers that AᵀAx usually differs from x. |
| `lesson-max-pool` | P1 | Pooling is described structurally, without identifying the statistic retained or lost. | Print four readings and move their maximum into one output; move a bright feature within the window. Check: can the output tell which location supplied the maximum? |
| `lesson-avg-pool` | P1 | Similar motion to max pooling hides a different meaning of the output. | Pair the same four readings with max and mean; include one outlier and a uniform field. Check: which statistic describes average intensity, and which preserves a local peak? |
| `lesson-strided-downsample` | P1 | Calling the kernel “learned” does not show what makes it different from fixed averaging. | Place learned weighted sampling beside average pooling with matched output sites. Show coefficients and a response change after training. Check: does learned downsampling guarantee suppression of aliases? |
| `lesson-pointwise` | P1 | A 1×1 filter sounds like it should do nothing. | At a single position, display three channel readings and two explicit linear combinations; then apply the same combinations everywhere. Check: can this layer inspect a neighbouring pixel? |
| `lesson-conv3d` | P1 | Earlier cubes represented channels, so a cube now representing three spatial axes changes the visual grammar. | Introduce a labelled tomographic x-y-z volume and slide a highlighted z slice before the 3-D stencil. Check: what makes this different from filtering an RGB image? |
| `lesson-conv3d-channels` | P0 | Source deliberately draws physical depth and channel slabs along the same display axis; four indices must be decoded from three dimensions. | Draw two separate, registered x-y-z volumes with channel names, then show one filter summing matching local volumes. Check: which direction is z, and where is channel identity represented? |
| `lesson-conv1d-stride-2` | P1 | Four-channel sequence blocks hide the basic relation to sampled signals. | Start with labelled time traces and overlay retained output times on the original sampling interval. Check: what is the output sampling interval in physical time? |
| `lesson-conv1d-dilation-2` | P1 | “WaveNet-style” adds unfamiliar architecture language without establishing what dilation changes. | Draw three fixed weights reading times t−4, t−2 and t; retain all legal output times. Check: does dilation necessarily reduce output sampling rate? |
| `lesson-edge-detection` | P2 | The signed Sobel response is a strong physics entry point but jumps to magnitude without an explicit derivative example. | Lead with a one-dimensional intensity ramp, then Gx and Gy and their magnitude; state pixel spacing and sign convention. Check: why can a strong edge have a negative Gx but positive magnitude? |

#### The maths, step by step

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `autodiff-graph` | P1 | The single-neuron graph is already compact, but many intermediate names and simultaneous derivative labels compete. | First work one path with x fixed and w adjustable; colour local derivative versus accumulated sensitivity differently. Add a finite-difference check and a branch where sensitivities add. Check: what does ∂L/∂w predict for a small change in w? |
| `gpt2-softmax-cross-entropy` | P0 | Ten digit classes and 50,257 token classes are processed side by side; only ten token bars appear, inviting incorrect normalisation. | Teach three logits first, explicitly account for omitted token probability as “all other tokens”, then compare tasks. Separate fitting temperature from sampling temperature. Check: can adding one constant to every logit alter the probabilities? |

#### An MLP, step by step

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `mlp-1-forward-layer1` | P1 | Hundreds of weights and the 8×8 display of 64 neurons look like another spatial image. | Calculate one neuron from two inputs including its bias, then unfold to 784 inputs; label the 8×8 arrangement as a display layout. Check: does adjacency of hidden squares imply spatial adjacency? |
| `mlp-2-forward-layer2` | P1 | Reusing image-like sheets conceals that inputs are now learned features, not pixels. | Track three named first-layer activations into one second-layer weighted sum; keep a breadcrumb from image to feature. Check: are second-layer weights connected directly to raw pixels? |
| `mlp-3-forward-output` | P1 | Logits, softmax, one-hot target and cross-entropy arrive in one sequence. | Split the result into raw scores, a probability distribution, the known label and a numerical penalty for that label. Check: can the model be correct yet have substantial loss? |
| `mlp-4-forward-pass` | P2 | Flowing illumination can imply literal transport or conservation of signal. | Use it as a recap after the numerical lessons, with a persistent “compute and store” legend and one tracked scalar. Check: do activations add up to a conserved amount across layers? |
| `mlp-5-backward-output` | P1 | `p − y` and sheets of tiny errors appear before a reader knows which object is being differentiated. | Annotate “sensitivity of loss to score” and work the correct-class and one incorrect-class components numerically before spreading to weights. Check: why does a negative score-gradient mean increasing that score helps? |
| `mlp-6-backward-layer2` | P1 | ReLU gating and transposed weight propagation occur together. | Separate propagation through the weighted sum from multiplication by the local activation slope; keep one path magnified. Check: where is a path blocked when the forward pre-activation was negative? |
| `mlp-7-backward-layer1` | P0 | “Only pixels > 0” is true for this nonnegative image example, not a general rule for neural-network input gradients. | Say “a zero input makes this weight's data-loss gradient zero in this example”; show one negative signed input and separate dL/dW from dL/dx. Check: can a negative input produce a nonzero weight gradient? |
| `mlp-8-backward-pass` | P1 | Gradients flowing to pixels can look like the training procedure edits the observed image. | Freeze a padlock on measured x, mark trainable W and b, and label dL/dx as sensitivity. Check: which arrays are actually updated in ordinary supervised training? |
| `mlp-9-training-step` | P1 | The gratifying 0.62→0.87 improvement on the same digit can be mistaken for evidence of generalisation. | Retain the exact step, then show a second untouched example and the full held-out metric. Make the one-example learning-rate choice visible. Check: can this digit improve while another prediction worsens? |

#### How networks learn

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `gradient-descent` | P1 | XOR plus a 2–3–1 sigmoid network introduces a new task and backprop at once. | Precede the network with one parameter fitting a measured calibration point; map residual, loss, slope and update onto the XOR case. Check: how does the sign of the derivative set update direction? |
| `cnn-training-step` | P1 | A full convolution/pooling/dense step obscures why one shared kernel receives many gradient contributions. | Trace one 3×3 kernel coefficient through two spatial uses, then sum its contributions before updating it once. Check: are separate filter weights trained at each image location? |
| `rnn-training-step` | P1 | An unrolled sine-wave network has many state sheets and gradients without distinguishing data time from training iteration. | Use two clocks and three sequence steps first; link repeated cells to one shared parameter set. Check: how many independent recurrent weight matrices are updated? |
| `lstm-training-step` | P1 | Multiple gates, cell state and BPTT are too many simultaneous unknowns. | First show a scalar stored reading with retain/write/read controls; name each gate's numerical operation, then restore the full vector graph. Check: which path can carry a value without repeated tanh multiplication? |
| `attention-training-step` | P1 | A tiny ViT, three heads, image patches and backprop all arrive together. | Establish one query taking a weighted average of three patches; differentiate just one score before expanding to heads. Check: are attention weights fixed model parameters or quantities recomputed for each image? |
| `rnn-batch-training` | P1 | Coloured waves flowing through a central network can look as if independent examples share hidden state. | Show two distinct state tracks and an explicit reset for independent sequences; combine only their loss gradients. Check: what is shared across a batch, and what must remain separate? |
| `gated-memory` | P1 | Vanishing-gradient ratios and memory traces are compelling but could imply all RNNs fail and all LSTMs remember indefinitely. | Keep cue→49 noise steps→answer visible; pair one retained scalar with the gradient path and repeat a counterexample setting. Check: which task length and gate settings were actually tested? |

#### Optimisation

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `loss-landscape-descent` | P1 | A landscape walker can be read as a particle descending a physical potential. | Build one landscape height by evaluating the same dataset at one pair of weights; label both axes as adjustable coefficients and the horizontal animation as iteration. Check: is downhill motion guaranteed to reduce loss for any step size? |
| `drunken-walk` | P1 | Random-looking SGD motion invites a Brownian-motion analogy with independent thermal noise. | At the same fixed weights, draw gradients from several sampled batches, their mean and covariance; then animate updates. Check: what determines the noise, and does its spread stay constant across weight space? |
| `optimisers` | P1 | Five trajectories, a valley and Adam moments require simultaneous new mechanisms; a race suggests a universal winner. | Compare SGD and momentum first on a numbered two-dimensional quadratic, then introduce coordinate rescaling. Separate optimisers from AdamW's changed objective effect. Check: were learning rates and stopping criteria comparable? |
| `adamw-vs-l2` | P1 | Effective-decay sheets require prior knowledge of adaptive preconditioning. | Use two weights with different accumulated squared gradients; calculate one L2-in-Adam step and one decoupled decay step side by side. Check: why do equal starting weights shrink differently under the coupled version? |
| `learning-rate-schedule` | P1 | Four schedules, loss curves and walkers conflate step-size policy with model training. | Start with the same path under two step sizes, then let one schedule change over a common update budget. Check: is the schedule indexed by examples, updates or epochs? |

#### Training deep networks

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `overfitting` | P1 | A 2-PC digit field is not immediately recognisable as a prediction experiment; several remedies follow quickly. | Establish train/validation/test points and one prediction error before the boundary moves; introduce each remedy separately. Check: which data are permitted to choose the stopping time? |
| `initialisation` | P1 | Twenty layerwise variances and gradients can look like numerical artefacts rather than a repeated random transformation. | Start with multiplying three scalar gains, then a vector's variance through many random layers. Check: can all-zero weights provide the same useful symmetry breaking as small random weights? |
| `skip-connections` | P1 | A bright shortcut can imply it guarantees stable gradients or simply bypasses learning. | Differentiate y=x+f(x) explicitly and show the identity and learned terms; retain a failing configuration as a boundary case. Check: can I+Jf still have a small or large singular value? |
| `normalisation` | P1 | A 32-example×24-unit sheet changes the reduction axis between methods; “normalisation” can be confused with unit conversion or probability normalisation. | Use a labelled 3-example×2-feature table with means and variances computed by hand, then scale up. Show batch-norm inference statistics. Check: does one example's batch-normalised output depend on its companions during training? |

#### Generalisation

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `double-descent` | P0 | Catalogue promises a U in test error; video and source explicitly find a U in test loss and falling test error. A “why” sensitivity correlation also sounds causal. | Correct the description; animate two wrong predictions with different confidence to separate metrics. Present sensitivity as an observed association in this sweep. Check: which plotted metric actually has the second descent? |
| `memorising-noise` | P1 | Near-zero training loss can look like successful recovery of corrupted labels rather than fitting the wrong targets. | Keep clean truth, corrupted training label and prediction in three distinct columns. Compare fresh clean evaluation and the early-stopping rule. Check: which loss can be small while scientific prediction remains poor? |
| `data-vs-model-size` | P2 | Real learning curves are useful but dataset size, model capacity and compute can be conflated. | Add matched training-budget labels and a fixed-size vertical comparison before sweeping data. Check: does a plateau establish that more data cannot help, or only that it did not help this training setup? |
| `augmentation` | P1 | Rotating digits may teach that all convenient transforms preserve labels. | Ask which transformation is an actual task symmetry; pair safe small shifts with an unsafe 6/9-like rotation and out-of-field crop. Check: what physical or labelling assumption makes each augmentation valid? |
| `sharp-vs-flat-minima` | P0 | Batch sizes 8/1024 also use different learning rates and update counts; a 2-D slice and one coordinate scale cannot prove a universal flatness/generalisation relationship. | Display the full run conditions, show two parameter rescalings and several slices, and keep loss/error distinct. Check: could apparent sharpness change without changing the model's predictions? |

#### Regularisation

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `l1-vs-l2` | P1 | Balls and contour tangency assume the connection between constrained fitting and penalised fitting. | Start with fitting two correlated sensor coefficients, then draw the constraint boundary and its selected intersection. Show the proximal update behind exact zeros. Check: are sparsity and shrinkage the same property? |
| `l1-l2-weights` | P1 | Weight images and never-lit borders obscure the distinction between evidence, penalty and optimisation rule. | Track one useful pixel weight and one always-zero input weight under each update rule; show the regulariser's contribution separately. Check: can a penalty update a weight whose data-loss gradient is zero? |
| `regularisers` | P1 | Dropout, input noise, max norm and label smoothing intervene at four different places. | Use one fixed diagram and highlight whether data, hidden activations, weights or target labels change. Work one numerical example per chapter. Check: which intervention should be disabled or changed at prediction time? |
| `regularisation-race` | P1 | Seven curves and a leaderboard overwhelm and imply that one strength fairly represents a method. | Teach one pair first, then show validation-selected strength sweeps and uncertainty across repeat runs. Check: which data chose the regularisation strength, and which measured final performance? |
| `regularised-autoencoder` | P1 | Matching reconstruction error while comparing five penalties hides distinct objectives for the latent code and weights. | Put reconstruction, latent sparsity, denoising and sensitivity on separate scorecards beside the same corrupted digit. Check: can equal reconstruction error conceal very different code robustness? |

#### Activation functions

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `activations` | P1 | Four functions, derivatives, saturation and a deliberately too-large-learning-rate failure arrive before explaining why a nonlinearity is needed. | Compose two linear maps first and show they remain linear; insert a ReLU, then inspect its slope with one number. Check: what new input-output shape becomes possible? |
| `activation-race` | P1 | Seven methods on log-time plots demand knowledge of every label and encourage ranking from one setup. | Start with sigmoid versus ReLU, keep identical budget labels and annotate near-ties, then add a separate seven-method overview. Check: would the same learning rate or initialisation be equally favourable to every function? |
| `activation-gradients` | P1 | The layerwise gradient pattern mixes local derivatives with weight products and the actual input distribution. | Multiply three measured local Jacobian factors before showing eight layers; overlay where inputs lie on each activation. Check: is the derivative at zero sufficient to predict gradient behaviour? |
| `activation-sparsity` | P1 | Zero, near-zero, saturated and permanently inactive units can be conflated. | Give each definition a distinct symbol and show one unit across many examples and training steps. Check: does being zero for this image mean a ReLU unit is dead? |
| `activation-smoothness` | P1 | A straight interpolation between pixel arrays may not be a plausible image trajectory; smooth curves may look intrinsically better. | State the path in input space, mark off-distribution mixtures, and compare output smoothness separately from prediction accuracy. Check: does a differentiable response guarantee a correct or robust classifier? |
| `activation-depth` | P1 | Matching each activation with its initialisation changes two choices at once. | Show a two-factor grid: activation choice and weight scale, followed by the recommended paired setups. Check: is a training failure caused by the activation alone? |

#### Capacity and bottlenecks

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `bottleneck-sweep` | P1 | One to 256 real-valued code coordinates can be confused with a corresponding bit budget or physical degrees of freedom. | Compare the same digit's retained shape and reconstruction residual as code dimension changes; distinguish dimension from precision. Check: do two floating-point coordinates imply only two bits of information? |
| `information-bottleneck` | P0 | A two-unit classifier bottleneck is not automatically the information-bottleneck objective or a measured mutual-information limit. | Retitle as “a narrow learned representation” unless information quantities are measured; show preserved class distinctions and discarded variation. Check: what information is measured here, and what is only inferred from accuracy? |
| `rank-of-a-layer` | P0 | Its own SVD result says the weights are not nearly rank 8, then the conclusion jumps to a claim about low-rank fine-tuning changes. | Separately visualise W, ΔW and their spectra; keep the successful rank-32 compression and measured accuracy result. Check: does the rank of W determine the rank needed to adapt W? |
| `width-vs-depth` | P1 | Roughly equal parameter counts are not equal computation, optimisation difficulty or representational structure. | Show parameter count, operations, serial layer count and measured training time on separate axes; compare equal-budget outcomes. Check: which resource is being held fixed in each conclusion? |

#### Trimming: pruning and distillation

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `magnitude-pruning` | P1 | Darkened weights can imply immediate memory or runtime savings from setting entries to zero. | Separate the mathematical sparse matrix, its storage format and actual execution; show accuracy before/after retraining. Check: does a dense array full of zeros automatically execute faster? |
| `lottery-ticket` | P1 | A memorable name can turn the measured two-seed comparison into a universal claim of a hidden winning subnetwork. | Retain the second-seed result, show all three initialisation/mask conditions before training, and state what each controls. Check: is success due to the mask, the original initial weights or both? |
| `structured-pruning` | P0 | The selection rule ranks activation energy on the test inputs (`filter_energy(P, xte)` and `unit_energy(P, xte)`), so “test” is participating in model selection. | Rank on training/calibration or validation data, reserve final test data, and show a rare but decisive low-energy feature as a caution. Check: why is low average activation not proof of dispensability? |
| `distillation` | P1 | “Dark knowledge” can mystify ordinary relative class probabilities and imply a student learns factual truth from its teacher. | Compare hard target versus the teacher's full distribution for one ambiguous 3/8; show temperature and loss components explicitly. Check: what happens when the teacher is confidently wrong? |

#### Fine-tuning and transfer

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `transfer-learning` | P1 | Digit groups 0–4 versus 5–9 are related tasks, so the demonstration can overpromise transfer to arbitrary scientific domains. | Identify reused edge features and replaced outputs, then add a deliberately mismatched source-domain comparison. Check: what similarity between the source and target data makes transfer plausible? |
| `catastrophic-forgetting` | P1 | Per-class bars demonstrate loss of skill, but the relation between incompatible gradients and parameter changes remains hidden. | Track old-task and new-task loss under one shared weight change before the long run; label replay's storage and sampling assumptions. Check: can a smaller learning rate remove a true task conflict? |
| `learning-rate-finder-and-restarts` | P1 | A range-test minimum and a walker “kicked out” of a basin can sound like a guaranteed tuning rule or literal jump. | Explain the short diagnostic run and state restoration; show that a restart changes step size without resetting weights. Check: does the loss minimum in a range test certify the best full-training rate? |
| `gpt2-lora` | P0 | “Teaching one new fact” uses an invented capital and a tiny 40-step corpus; successful continuation is not demonstrated reliable knowledge editing. | Label it a controlled text-association experiment; separate trainable count, update rank, prompt transfer and side effects. Test paraphrases and unrelated controls. Check: can the preferred token change while the intended general fact is still unreliable? |

#### Quantization and precision

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `quantization` | P1 | Bit widths, scales and error sheets presuppose how numbers are represented. | Begin with rounding one sensor coefficient to a small grid; separate scale, clipping and rounding, then show model-level effects. Check: why can a small coefficient error produce a large prediction error on some inputs? |
| `quantization-aware-training` | P1 | The displayed staircase's true derivative is zero almost everywhere, while the training path uses a straight-through estimate. | Preserve the existing true/STE distinction, magnify one numerical update and explicitly call the backward rule a surrogate, not exact differentiation. Check: which derivative is mathematical and which is chosen for optimisation? |
| `mixed-precision` | P1 | Bit layouts and underflow can distract from the useful error-analysis question. | Follow one gradient magnitude through representable ranges, show multiplication by the loss scale and its later cancellation, then reveal exponent/mantissa bits. Check: does loss scaling change the intended unscaled parameter update? |

#### Predictions

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `mlp-classifier` | P2 | A familiar node-link picture can still make inference look like a training process. | Label fixed parameters, new measured input and output probabilities; pair one correct and one incorrect digit. Check: are any weights changed by making this prediction? |
| `dnn-eval` | P1 | Loading a trained 784–100–100–100–10 network is a programmer's entry point, not an experimental question. | Begin with “given a new detector image, what does this already-fitted model predict?”; move loading and dimensions to a detail card. Check: how does evaluation differ from fitting? |
| `strided-cnn-classifier` | P1 | The 784→169→36 channel counts mix flattened counts with two-dimensional maps. | Keep 28×28→13×13→6×6 visible next to scalar counts and channel counts; follow one local pattern. Check: which reduction is spatial and which mixes channels? |
| `decision-tree` | P2 | Threshold search and Gini are algorithmic before the viewer sees why a split is useful. | Start with two measured observables and one decision boundary; compute impurity for a four-point candidate split, then animate the search. Check: do leaf class proportions describe the training group or guarantee a new event's true probability? |

#### Gaussian processes

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `gp-prior` | P1 | Covariance matrices and simultaneous function draws can obscure that a GP is a distribution over whole functions. | Begin with two correlated measurements, then three locations, then a dense grid; hold one random draw fixed while varying length scale. Check: does zero prior mean mean every sampled function is near zero everywhere? |
| `gp-posterior` | P1 | The sampled band is labelled ±2σ but its relation to noisy observations needs to be explicit; matrix inversion can dominate the explanation. | Begin with conditioning two correlated Gaussian variables; separately shade latent-function uncertainty and future observation uncertainty. Check: should noisy observed points lie exactly on the posterior mean? |
| `gp-kernels` | P1 | Five fitted kernels and marginal-likelihood readouts invite choosing a curve solely by in-sample fit. | State each covariance assumption in physical language and hide the extrapolation region until the viewer predicts it. Check: which assumption makes periodic extrapolation plausible, and how would observations falsify it? |
| `gp-hyperparameters` | P1 | A voxel landscape and “data fit versus complexity” bars can sound like arbitrary competing penalties. | First show length scale and noise changing a single posterior, then evaluate their joint marginal likelihood. Mark the selected values as a point estimate. Check: does conditioning on fitted hyperparameters include uncertainty about those hyperparameters? |
| `gp-bayesian-optimisation` | P1 | Watching a hidden function's maximum being found can imply reliable global optimisation after eight queries. | Keep the true function hidden until the end; compute improvement for one candidate and show a misleading prior case. Check: why might the next point be far from the current best observed value? |

#### Autoencoders

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `autoencoder-dense-3d` | P1 | A coloured latent cube can look like physical coordinates or supervised class separation. | Show one image, its three-number code and reconstruction residual before revealing labelled points; state labels only colour the plot. Check: would rotating all codes with a corresponding decoder change destroy the representation? |
| `autoencoder-dense-2d` | P2 | Interpolation looks convincing without showing whether points between codes are supported by training data. | Keep the full 2-D plane, mark densely and sparsely sampled regions and show the decoded image along both paths. Check: must every intermediate code decode to a plausible digit? |
| `autoencoder-conv-3d` | P1 | Convolutional maps and three latent coordinates share a voxel grammar but mean different things. | Compare dense and convolutional encoders for the same image; explicitly collapse spatial maps to a three-number code. Check: which stage retains image coordinates and where are they replaced? |
| `autoencoder-conv-2d` | P2 | The compact model is useful but its upsample-and-convolve decoder can be mistaken for an exact inverse of the encoder. | Display reconstruction error and distinguish learned decoding from reversing the original convolution. Check: can two different images receive almost the same code? |
| `vae-dense-3d` | P0 | The prior cube, per-input cloud and generated samples are easy to conflate; novelty is asserted more strongly than sampling establishes. | Draw the prior p(z), per-image q(z\|x), one sampled z and p(x\|z) in distinct panels; show reconstruction and KL terms separately. Check: does drawing a fresh z prove its decoded image was absent from training? |
| `vae-dense-2d` | P0 | The inherited AE layout can leave the crucial change—encoding a distribution—visible only briefly; KL is described as filling gaps automatically. | Hold one input fixed and decode several z draws, then show KL pressure and a possible poor prior-sample region. Check: does matching a prior guarantee every sampled point produces a good digit? |
| `autoencoder-training` | P1 | Label-coloured codes sorting themselves can imply the class labels trained the encoder. | Start uncoloured, show the reconstruction objective, then reveal labels after training as an external diagnostic. Check: what supervision did the network actually receive? |
| `diffusion` | P0 | The caption claims generated digits are absent from the training set without testing novelty; noise→digit imagery also invites confusion with unique recovery or physical time reversal. | Replace the untested novelty claim with “samples from the trained model.” Distinguish a training corruption pair from independent generation; label artificial diffusion time and learned denoising estimates. Check: if two original digits can yield similar noisy states, how can reversal be unique? |
| `pca-and-kmeans` | P1 | PCA variance, reconstruction and clustering are three different objectives inside one clip. | Make three chapters: best linear reconstruction, a two-coordinate projection, and clusters fitted in that projection. Check: can a low-variance direction still separate scientifically important classes? |
| `unet-dense` | P1 | The name U-Net and skip geometry can imply this dense demonstration has the spatial architecture of a conventional convolutional U-Net. | State exactly what is dense here; trace noisy detail through one encoder–decoder skip and contrast it with a residual addition. Check: what information reaches the decoder without passing the bottleneck? |
| `dec-dense` | P1 | Soft cluster assignments on labelled latent points can look like supervised classification. | Hide labels while updating centres and assignments, then reveal labels only for evaluation; explain self-sharpening targets. Check: can a confident cluster assignment still group the wrong physical populations? |
| `idec-dense` | P1 | Adding a reconstruction term is difficult to judge without seeing what DEC lost. | Run DEC and IDEC from the same codes with matched clustering settings; show both latent movement and reconstructed images. Check: what trade-off does the reconstruction weight control? |

#### Embeddings

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `embedding-lookup` | P2 | The lookup is accessible, but a large log-scale operation-count comparison makes indexing appear to be the conceptual point. | First pair a discrete detector/category ID with a learned vector and one changed row; leave dense-versus-indexed arithmetic as an optional performance chapter. Check: does numerical closeness of token IDs imply similar embeddings? |
| `word2vec-training` | P1 | A generated corpus with predefined categories and a projected scatter can overstate discovery of general semantic geometry. | Show actual co-occurrence counts, one positive/negative training pair, then the fixed projection and its limits. Check: are plotted distances faithful to all distances in the full embedding space? |
| `sentence-embeddings` | P1 | The title could imply all sentence embeddings are word-vector means; SIF weighting may sound like it repairs lost word order. | Rename the example as mean-pooling word embeddings and lead with the dog/cat reversal counterexample. Check: can any reweighting that remains order-independent distinguish an exact word permutation? |
| `patch-embeddings` | P1 | Cutting, flattening, multiplying and adding position vectors mixes representation choices with array operations. | Take one 2×2 image patch, list its four measured values and apply one explicit shared projection; then show position identity. Check: does the same patch content receive the same content vector at another location? |
| `gpt2-position-blindness` | P1 | The existing no-position/no-mask demonstration is carefully scoped, but a novice still needs permutation equivariance distinguished from order information supplied by a mask. | Preserve all four conditions, label permutation equivariance in plain language, and track one fixed word rather than a fixed slot. Check: why can a causal mask distinguish some orders even without a position table? |
| `sinusoidal-positions` | P1 | Frequency tables are familiar mathematically, but what they encode and what the model must learn remain separate questions. | Rotate one sine/cosine pair for a fixed offset, then combine frequencies and add the resulting vector to content. Check: does a computable encoding beyond training length guarantee useful predictions there? |
| `relative-position-bias` | P1 | Buckets and learned per-head biases arrive before their effect on a prediction. | Hold content scores fixed and add one offset bias before softmax; then group near/far offsets into bins. Check: can two distant offsets become indistinguishable to this bias term? |
| `alibi` | P1 | A distance penalty resembles a physical decay law, although it is an imposed attention-score bias. | Show raw content score, linear distance penalty and final softmax weight separately for three positions. Check: can a distant token still receive the largest weight? |
| `rope-context-extension` | P1 | Angle extrapolation, attention failure, interpolation and NTK scaling demand several prerequisites; one toy model can look like a universal remedy test. | Establish fixed content vectors and phase differences first, then retain the train-length boundary and model/task labels on every result. Check: what short-distance resolution changes when all positions are compressed? |
| `length-extrapolation` | P0 | Six schemes on an order-2 synthetic language show extrapolation in position index, not demonstrated use of arbitrarily distant information. | Explicitly show that only the previous two symbols determine the target; add a separate long-dependency task before broader claims. Check: does success at position 128 prove the model used a token 100 positions earlier? |

#### Transformers

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `seq2seq-attention` | P1 | Sequence reversal is clear but a moving alignment matrix can look like a programmed copying rule. | Show the untrained distribution, one weighted value mixture and its error before the learned alignment emerges. Check: where is the task rule stored, and was the correct alignment supplied directly? |
| `gpt2-tokenizer` | P1 | Byte ranks, token IDs and whitespace differences are implementation details before the reason for tokenisation is clear. | Start with text as a sequence of reusable pieces, explicitly represent spaces, then perform two BPE merges with a fixed vocabulary. Check: does a token always correspond to a word or a meaningful physical concept? |
| `gpt2-embeddings` | P1 | Nearest neighbours and king−man+woman can turn one geometric example into a universal law of meaning. | Separate one lookup from a measured neighbour query; include a failed analogy and context-dependent use of a word. Check: does vector arithmetic guarantee factual or semantic correctness? |
| `gpt2-attention` | P1 | The nine-stage camera tour expands to 12 heads and 12 layers before a novice can interpret Q, K and V. | Begin with three tokens and a two-component weighted average; keep row/column roles and normalisation fixed, then use the full tour as an advanced chapter. Check: which quantity is averaged, and do attention weights alone explain the final answer? |
| `gpt2-transformer-block` | P1 | Residual streams, LayerNorm, MLP and the logit lens mix model computation with an interpretive diagnostic. | Compute one block's add–transform–add sequence first; mark intermediate logit-lens outputs as probes, not the model's completed predictions. Check: was an intermediate layer trained to be a standalone predictor? |
| `gpt2-sampling` | P1 | Greedy/temperature/top-k/top-p generation and KV caching combine output choice with implementation efficiency. | Keep one distribution frozen while applying each sampler, repeat draws, then place caching in a separate chapter. Check: which settings change the model distribution or sampling rule, and which should preserve the result mathematically? |

#### What language models know, and don't

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `gpt2-hallucination` | P0 | Real versus invented capital continuations illuminate the problem but cannot establish one complete explanation of hallucination or prove abstention is impossible. | Label the base model, prompt and six sampled continuations; distinguish language-likelihood objective from external verification and trained abstention. Check: why is a probable continuation not necessarily a true statement? |
| `gpt2-conditioning` | P1 | Distribution shifts after examples can look like parameter learning; attention to context can look like a causal explanation. | Padlock the model weights while varying only the prompt; compare next-token probabilities and actual answers separately. Check: what remains changed after the prompt is removed? |
| `gpt2-calibration` | P0 | Forty selected single-token country/capital answers and five bins are a small specialised experiment, not a universal confidence calibration measure. | Show counts and uncertainty per bin, mark the single-token restriction and prompt dependence, and vary binning. Check: does a calibrated collection make any individual 80%-confidence claim certainly true? |

#### Modern sequence models

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `ssm-recurrence` | P1 | Recurrence, equivalent convolution and changing step size can confuse learned sequence machinery with a known physical state-space model. | Derive three steps of one linear time-invariant scalar recurrence and its impulse response; distinguish sampling interval from learned discretisation parameters. Check: which assumptions make the convolution equivalence valid? |
| `mamba-selective-scan` | P1 | Input-dependent gates, a memory benchmark and a logarithmic scan are three concepts presented together. | First contrast a fixed recurrence with an input-dependent retain/write rule; demonstrate associativity on four steps before showing parallel levels. Check: does a fixed convolution kernel still represent an input-dependent selective recurrence? |
| `context-cost` | P0 | Matrix size, per-token work, parallel depth and cache size can be confused; “depth 1” for scores is not constant-time full attention on real hardware. | Use separate panels for arithmetic work, dependency depth and stored numbers, explicitly separating prefill and decoding. Include reduction and hardware limits. Check: can two algorithms with the same parallel depth require very different total work? |
| `mixture-of-experts` | P1 | Named experts and top-2 routing suggest human-like subject specialists or guaranteed efficiency. | Display routing scores, selected numerical subnetworks and their weighted output for one token; measure utilisation separately from task quality. Check: does balancing token counts guarantee equal usefulness or wall-clock speedup? |
| `gpt2-rotary-embeddings` | P0 | The preset name can imply GPT-2 uses RoPE; it actually compares RoPE mathematics with GPT-2's learned absolute table. Oscillatory pair scores are not universally monotonic distance decay. | Retitle the viewer-facing lesson as a comparison; derive R(m)ᵀR(n) for fixed q,k and show an oscillatory counterexample. Check: which position method does the actual GPT-2 model use? |

#### Reinforcement learning

| Preset | Priority | Barrier in the current treatment | Concrete redesign and learner check |
| --- | --- | --- | --- |
| `q-learning` | P1 | A 6×8 height-field, exploration schedule and Bellman update arrive before action-value and delayed reward are established. | Begin with three states and two actions, distinguish immediate reward from expected discounted return, and calculate one update. Then add the grid. Check: does a high Q-value mean a large immediate reward at that square? |


## The 100 rebuilt animations

The first audit's obstacle for almost every one of these was the same: a finished matrix or curve
shown before any mechanism. The rebuild answered that. This section records, for each, what its
experiment now is, what is still open for this audience, and the check question the first audit
wrote, which is still the right test of whether the film teaches. A few of those questions were
written against the first experiments and name details that have since changed (a task, a count);
read them for the distinction they ask for.

### What became of the first audit's 25 top-priority concerns

| Preset | Concern (first audit) | Now |
|---|---|---|
| `weight-tying` | Saved parameters and saved bytes as comparable bars; the shared matrix never shown. | Fixed: one table drawn at both ends, gradient arriving from each. |
| `teacher-forcing` | Tokens anonymous; free-running text scored against one continuation. | Tokens are shown. The scoring concern stands (see the table). |
| `gpt2-perplexity-explained` | No token words; nats, bits and perplexity on one axis; models scored on different text. | Fixed: real tokens, units separated, the same 247 tokens scored three ways. |
| `scaling-laws` | Unlabelled transformed axes; the fit read as a law. | Redesigned; the fit is tested on held-out budgets, fails, and is called descriptive. |
| `gpt2-beam-search` | A heatmap in place of the tree; unsigned bars. | Fixed: the tree grows and prunes. |
| `gpt2-contrastive-decoding` | Signed differences as unsigned bars. | Encoding fixed. The method's benefit is not shown (null over 12 passages). |
| `dpo` | No visible preferred and rejected pair. | Fixed: one real pair worked through; now matches PPO at the same β. |
| `reward-hacking` | The harmful final value drawn as the longest bar. | Fixed. The overshoot is rare (4 of 30 runs) and the run drawn is labelled as the worst. |
| `superposition` | Independently scaled axes distorting angles. | Redrawn as arrows in a plane; confirm equal scales. |
| `gpt2-linear-probes` | 100% at layer zero; one template. | Rebuilt on held-out text with a made-up-label control; the property is still close to token identity. |
| `gpt2-head-ablation` | Two prompts. | Redesigned: 64 passages, ranks agree across halves at 0.82. |
| `feature-visualisation` | The regularised response fell to zero. | Redesigned on a class unit; partly (stroke fragments). |
| `vit-attention` | One digit shown, attention averaged over 128. | Fixed: per-digit maps. Reach still equals that of even attention. |
| `clip-contrastive` | Unlabelled matrices; "zero-shot" overstated. | Matrices labelled; the screen says the words are the trained ones. |
| `batchnorm-in-cnns` | Post-hoc normalisation of a network trained without it. | Redesigned: trained with batch normalisation; the bug costs 3.7 points at a batch of one. |
| `vae-posterior-collapse` | β and the ramp changed together. | **Open.** Not redesigned; the caveat is on screen. |
| `tsne-and-umap` | UMAP promised, not built. | Fixed: UMAP implemented. |
| `loss-spikes` | The spike was a deliberate change of data. | Redesigned: spikes arise on fixed data. Two of three remedies do not help. |
| `gpt2-retrieval-augmented-generation` | The query contained the answer. | Redesigned: 60 questions, with the source line on file and removed as separate conditions. |
| `roofline` | The roof inferred from the points placed under it. | Redesigned: ceilings timed independently. Counts are still logical arrays, not measured traffic. |
| `policy-gradient` | "Variance" was not the estimator's; the runs saw different trajectories. | Redesigned: both estimators on the same episodes against a large-batch gradient, five seeds. |
| `dqn-replay-buffer` | Batch size changed with the method. | Redesigned: same batch size in every arm, plus a shuffled control. |
| `reward-shaping` | The farming loop never drawn; the potential not shown. | Redesigned: both shown with exact arithmetic for two episodes. No speed-up from shaping. |
| `hopfield-network` | The failed recall described as correction. | Kept as the honest result, with measured capacity for four rules beside it. |
| `neural-ode` | No persistent paths; step-size plot measured fit, not numerical error. | Redesigned: paths drawn, a Runge-Kutta reference, adjoint against backprop. |

### Every one of the 100

"Shows its concept (redesigned)", "Partly" and "Null" are the outcomes recorded in TODO.md.
"First experiment, redrawn" means the measurement is the original one and only the drawing changed.

| # | Preset | Experiment now | Still open for this audience | Check question or limit to keep visible |
|---|---|---|---|---|
| 1 | `causal-vs-bidirectional` | First experiment, redrawn |  | Can the prediction use the answer? Here causal means a dependency restriction in sequence order, not causal inference or relativistic causality. |
| 2 | `prefix-lm-attention` | **Null** after redesign: prefix against causal: seeds overlap (it does show why a bidirectional reply cannot be generated) |  | Which previously supplied prompt tokens may exchange information? A tie is valid evidence, not a reason to imply prefix attention won. |
| 3 | `cross-attention` | First experiment, redrawn |  | Where do Q, K, and V come from? This small reversal link is not evidence of translation ability. |
| 4 | `multi-head-vs-single` | **Partly**: eight heads beat one by 0.03 nats; heads differ by a seed-dependent amount |  | Does extra head capacity guarantee useful specialization? No; repeat seeds/tasks before any general architectural claim. |
| 5 | `grouped-query-attention` | First experiment, redrawn |  | Which arrays are shared, and which stay separate? Reduced cache memory is exact for the stated shape; quality trade-offs are empirical. |
| 6 | `gpt2-kv-cache` | First experiment, redrawn | The cache phase labels the kept rows but does not draw the new token reading them. | What is cached, and what is still read? Caching reduces repeated projections; attention over a growing cache remains work. |
| 7 | `sliding-window-attention` | Shows its concept (redesigned) |  | Can information reach a position versus did training learn to use it? With width four including self, three layers reach nine previous positions. |
| 8 | `sparse-attention-patterns` | Shows its concept (redesigned) |  | Which missing edge removes the only useful route? Linear cost depends on the fixed window/global-token assumptions. |
| 9 | `linear-attention` | First experiment, redrawn |  | Which two computations are exactly equal? ELU+1 kernel attention is a different attention rule, not exact softmax, nor an explanation of every state-space model. |
| 10 | `flash-attention-tiling` | First experiment, redrawn |  | Why must the old numerator and denominator both rescale? Exact up to rounding does not by itself establish a speed-up or total-memory saving on every device. |
| 11 | `gpt2-attention-sinks` | First experiment, redrawn |  | Is a heavily attended token necessarily carrying important semantic content? Test multiple prompts and position conventions before generalizing the striking result. |
| 12 | `attention-temperature` | First experiment, redrawn |  | Why √d under independent unit-variance components? “Temperature” is a mathematical sharpness parameter here, not a measured physical temperature. |
| 13 | `pre-ln-vs-post-ln` | Shows its concept (redesigned) | The effect depends on the learning rate, which was chosen from a one-seed pilot. | Which path lets a change bypass the transformed branch? Do not turn one query-projection norm or short run into a universal stability theorem. |
| 14 | `rmsnorm-vs-layernorm` | **Null** after redesign: within seed noise over eight seeds, which is the expected result |  | Which transform is invariant to a common offset? State the learnable scale/bias convention; equal toy parameter counts are not evidence that all implementations cost the same. |
| 15 | `swiglu-mlp` | **Null** after redesign: within seed noise over eight seeds |  | Which factor gates which? Use the correct SiLU/Swish formula; gates need not be binary, probabilistic or bounded by one. |
| 16 | `gpt2-residual-stream` | First experiment, redrawn | Layer 12's attention write is drawn as a broken bar off the scale. | Does a large write norm mean a large relevant effect? Intermediate logit-lens readouts are probes, not literal finalized beliefs. |
| 17 | `weight-tying` | First experiment, redrawn |  | Is the output projection the inverse of embedding lookup? It is a shared-transpose scoring map, not an inverse transformation. |
| 18 | `parallel-block` | Shows its concept (redesigned) |  | Can the MLP see the current attention write? Separate shorter dependency depth from an experimentally established runtime gain. |
| 19 | `encoder-decoder-family` | First experiment, redrawn | The three models differ in inputs and size; the screen says tour, not ranking. | Which model reads a complete input, and which emits one new symbol at a time? Success on reversal does not make the three pretraining objectives interchangeable. |
| 20 | `transformer-depth-vs-width` | **Null** after redesign: depth past two blocks does not pay at this size |  | What extra intermediate computation can another block supply? Equal parameters and optimizer steps do not imply equal runtime or all budgets matched. |
| 21 | `teacher-forcing` | First experiment, redrawn | Free-running text is still scored against the one recorded continuation, so a different but valid continuation counts as an error. Say so on screen or score with the model's own likelihood. | What is supplied during training that must be generated at inference? Divergence from one sampled future is not automatically a model error. |
| 22 | `gpt2-perplexity-explained` | First experiment, redrawn | The bigram baseline is worse than the unigram one here, counted from only 9,103 tokens. | If two sentences have different lengths, how is corpus loss averaged? Perplexity is an effective branching count, not accuracy; tokenizer and scored positions must match. |
| 23 | `packing-and-padding` | First experiment, redrawn | One sentence in the packing phase is explanation, not measurement. | Why can a token not read the previous packed document? State the chosen position convention; packing is not merely concatenation. |
| 24 | `warmup-and-clipping` | Shows its concept (redesigned) | Half the rate without warm-up does as well; the failure is collapse onto the letter-frequency plateau, not divergence. | Does clipping the gradient bound an Adam parameter update in the same simple way? Avoid that implication; this run need not diverge without warmup. |
| 25 | `grokking` | Shows its concept (redesigned) | The late rise ends below the ordinary start (90.5% against 92.7%). | Is the late test improvement reproducible, and what changes beforehand? A visually sharp learning transition is not sufficient evidence of a thermodynamic phase transition. |
| 26 | `scaling-laws` | **Partly**: loss falls with compute; the fitted power law does not extrapolate |  | What was varied and what was held fixed? Thirty small synthetic runs and a 6ND compute estimate do not establish an industry-scale optimum. |
| 27 | `curriculum-and-data-mixing` | First experiment, redrawn |  | Which deployment mixture defines success? Changing the final averaging weights can change the winner. |
| 28 | `tokenizer-effects` | First experiment, redrawn |  | Can per-token loss be compared across tokenizers? No; equal training steps also need not equal processed characters or compute. |
| 29 | `gpt2-beam-search` | First experiment, redrawn |  | Why can a locally best first word lose globally? Fixed-length likelihood search is not a guarantee of meaning, truth, or a better human answer. |
| 30 | `gpt2-speculative-decoding` | **Partly**: saves GPT-2 passes (0.71 per token) but is 1.2 times slower in numpy wall-clock |  | Why is simply replacing a rejected token with the target argmax wrong? Preserve the actual slow result; exact distribution and acceleration are different claims. |
| 31 | `gpt2-repetition-and-penalties` | Shows its concept (redesigned) |  | Is suppressing repetition always desirable? These alter the decoding distribution and can damage valid repeated names, formulae or structured output. |
| 32 | `gpt2-constrained-decoding` | Shows its concept (redesigned) |  | Does valid syntax imply a valid measurement or true fact? This toy ASCII grammar is narrower than a full token-aware JSON implementation. |
| 33 | `gpt2-contrastive-decoding` | **Null** after redesign: the contrastive pick is right less often than plain GPT-2's |  | Why could a huge ratio favor an extremely unlikely word? The plausibility constraint matters; a truncated checkpoint is a particular amateur, not universally meaningful expertise. |
| 34 | `batching-and-throughput` | First experiment, redrawn | Timings are numpy on one machine; batch 2 came out slower than batch 1. | Can throughput improve while one user's wait grows? This is a local NumPy projection benchmark, not an inference-server benchmark. |
| 35 | `instruction-tuning` | Shows its concept (redesigned) | The task is a re-encoding of real text (capitals); answers are computed, not written by people. | Is a response mask a new architecture? No; this synthetic arithmetic instruction task demonstrates a supervision choice, not general helpfulness. |
| 36 | `reward-model` | Shows its concept (redesigned) |  | Do reward numbers have an absolute, calibrated unit? Pairwise data constrain differences; a preference proxy is not true utility or human consensus. |
| 37 | `rlhf-ppo` | Shows its concept (redesigned) |  | What constrains one update versus cumulative drift? PPO objective clipping is not a hard universal bound on policy change; this is a synthetic bandit. |
| 38 | `dpo` | Shows its concept (redesigned) |  | Does “no separately trained reward model” mean no preference assumptions? No; same examples do not mean identical policy parameterization or optimizer. |
| 39 | `reward-hacking` | **Partly**: appears in 4 of 30 runs; the run drawn is the labelled worst case |  | Where does the optimum leave the calibrated range? Call the mechanism proxy exploitation rather than implying conscious deception. |
| 40 | `prompt-and-prefix-tuning` | Shows its concept (redesigned) | The adaptation target is the same text in capitals; each method's learning rate was picked on validation. | Which learned object changes at inference? Soft prompts are trained vectors, not necessarily human-readable prompt text; this LoRA updates only the output projection. |
| 41 | `model-merging` | **Partly**: averaging works; the full sum overshoots and negation is not selective |  | When does vector addition correspond to composable behavior? Only matched architectures/bases make this arithmetic meaningful, and interference must be measured. |
| 42 | `gpt2-knowledge-editing` | First experiment, redrawn | One edit and one control passage. | Did the edit change a robust factual association or one prompt's token probability? This is an illustrative local intervention, not demonstrated general knowledge replacement. |
| 43 | `superposition` | First experiment, redrawn | Check the plane is drawn with equal scales on both axes, so the angles between features are true. | Does a low loss on mostly-zero data prove all features are represented well? Show per-active-feature error; this is a nonlinear sparse coding toy, not quantum superposition. |
| 44 | `gpt2-sparse-autoencoder-features` | Shows its concept (redesigned) | Readable is judged by a one-word test, which undercounts themed features; 24 features fire on over 10% of tokens. | Can “the” being the maximum-activation token establish a unique concept? No; reconstruction, reproducibility and causal utility are different tests. |
| 45 | `gpt2-linear-probes` | First experiment, redrawn | The probed property (is the previous token punctuation?) is close to token identity; the made-up-label control scores 0.81 at layer 0. A property that needs context would test more. | Did the probe discover quotation state or a shortcut? The current perfect embedding baseline is a reason to investigate, not evidence of deep reasoning. |
| 46 | `gpt2-activation-patching` | First experiment, redrawn | One passage. Repeat over several before reading the layer pattern as general. | What exact intervention changed the outcome? Recovery in one corrupted context is local evidence, not a uniquely localized fact or a general causal circuit. |
| 47 | `gpt2-induction-heads` | Shows its concept (redesigned) | The small model's repeats are constructed from real text; ordinary text grows no such head. | Is the highlighted head necessary for this completion? Attention patterns and composed attention matrices alone do not prove the computational circuit. |
| 48 | `gpt2-head-ablation` | Shows its concept (redesigned) | Zero ablation removes a head's average output too; mean ablation is the usual control. | Does a small single-head effect prove irrelevance? Redundancy and off-distribution interventions can hide or distort causal roles. |
| 49 | `saliency-and-integrated-gradients` | First experiment, redrawn | One digit, on a network whose logit is nearly linear along the path, so gradient × input is almost complete as well. A less linear case would separate the methods. | Is this local sensitivity, finite attribution or a causal explanation? Explain the chosen target logit and baseline; completeness alone does not establish semantic faithfulness. |
| 50 | `feature-visualisation` | **Partly**: stroke fragments, not well-formed digits |  | Are we changing pixels or weights? A smooth picture that no longer activates the feature is not a successful explanatory visualization. |
| 51 | `receptive-field` | First experiment, redrawn | The effective field is shown as a measured concentration curve; no Gaussian is fitted, so do not call it Gaussian. | Check that the learner distinguishes possible support from measured sensitivity and does not infer a guaranteed Gaussian profile. |
| 52 | `vit-attention` | **Partly**: attention is concentrated; its reach equals that of even attention |  | Check that displayed images and attention come from the same forward pass. |
| 53 | `clip-contrastive` | First experiment, redrawn | The ten words are the ones it trained with, as the screen says: this is not zero-shot. Screen time is weighted to the first 100 updates. |  |
| 54 | `adversarial-examples` | First experiment, redrawn |  |  |
| 55 | `object-detection-basics` | First experiment, redrawn | The six proposals are shifted copies with hand-set scores, not a detector's own. Say so in the first frame of that phase. |  |
| 56 | `semantic-segmentation` | First experiment, redrawn | The claim that the skip carries position and the half-size path carries identity is a reading of two ablations, not a separate measurement. |  |
| 57 | `batchnorm-in-cnns` | Shows its concept (redesigned) | A batch of one is not a collapse (94.4%): a convolution layer still averages over hundreds of positions. |  |
| 58 | `style-transfer` | **Partly**: the thin style transfers cleanly; the heavy one leaves haze |  |  |
| 59 | `gan-training` | **Partly**: the plain GAN drops a digit at 19% of late checkpoints and recovers |  |  |
| 60 | `normalizing-flows` | Shows its concept (redesigned) |  |  |
| 61 | `flow-matching` | First experiment, redrawn | Not redesigned; still on the all-digit PCA blob. Move it to the three-cluster target that `gan-training` and `normalizing-flows` now use. |  |
| 62 | `classifier-free-guidance` | Shows its concept (redesigned) | Variety falls to guidance 3 and rises again at 8 as strokes thicken; both measures are proxies. |  |
| 63 | `ddpm-vs-ddim` | First experiment, redrawn | Not redesigned. Confidence comes from four samples and a 200-step denoiser; more samples before any ranking of step counts. |  |
| 64 | `vae-posterior-collapse` | First experiment, redrawn | Not redesigned. The two runs differ in final β (4 against 1) as well as in the ramp, so the ramp's effect is not isolated; add the fixed β = 1 arm and seeds. |  |
| 65 | `vq-vae-codebook` | **Partly**: healthy codebook; the collapse control needs 64-number codes started far away |  | Check that learners know the surrogate is not the true derivative of argmin. |
| 66 | `autoregressive-pixels` | Shows its concept (redesigned) | The blind spot is arithmetic on the masks, not a trained comparison. |  |
| 67 | `bias-variance` | Shows its concept (redesigned) | The bias at the highest degrees includes sampling error of the mean of 200 fits. | Check that bias here means estimator error, not a neural-network intercept or social bias. |
| 68 | `cross-validation` | First experiment, redrawn | Folds are index mod 5 (visible as stripes) and the inputs are 7×7 thumbnails. |  |
| 69 | `svm-and-kernels` | First experiment, redrawn | The classes overlap, so there is no hard margin to show; the lift is an oblique drawing, not a rendered 3-D scene. | Check that a learner can predict which point moves change the fitted boundary. |
| 70 | `gradient-boosting` | First experiment, redrawn |  |  |
| 71 | `gaussian-mixture-em` | **Partly**: the start matters; k-means++ seeding alone does not help |  |  |
| 72 | `curse-of-dimensionality` | Shows its concept (redesigned) | The no-signal coordinates are real pixel columns shuffled across images: a disclosed transformation, to be checked against the data rule. |  |
| 73 | `tsne-and-umap` | Shows its concept (redesigned) | UMAP's constants a and b are hard-coded for min_dist 0.1, not derived. |  |
| 74 | `roc-and-calibration` | First experiment, redrawn |  |  |
| 75 | `newton-vs-gradient-descent` | Shows its concept (redesigned) | The loss is a quadratic, so Newton's one step is exact here and would not be for a network. |  |
| 76 | `edge-of-stability` | Shows its concept (redesigned) | The 256 digits are memorised; held-out accuracy is about 76% at every step size. |  |
| 77 | `batch-size-and-noise` | First experiment, redrawn |  |  |
| 78 | `weight-averaging` | First experiment, redrawn |  |  |
| 79 | `second-order-preconditioners` | First experiment, redrawn |  |  |
| 80 | `loss-spikes` | **Partly**: spikes arise on fixed data and a short memory stops them; a larger epsilon and clipping do not |  |  |
| 81 | `gpt2-retrieval-augmented-generation` | **Partly**: helps only when the store holds the answer line |  | Check retrieval and generation separately with held-out question–passage pairs. |
| 82 | `vector-search-hnsw` | First experiment, redrawn | In the search phases the angle round the rings is an arrangement, not data; only the radius (true distance) is. |  |
| 83 | `infonce-temperature` | Shows its concept (redesigned) | The untrained encoder scores 84.9% and raw pixels 92.6% on this measure; only the comparison between temperatures is evidence. |  |
| 84 | `hard-negatives` | Shows its concept (redesigned) | Random negatives end no better than the untrained encoder. |  |
| 85 | `dimensionality-of-embeddings` | First experiment, redrawn | The nested model had 121 more updates than the per-width ones. |  |
| 86 | `data-parallel-allreduce` | Shows its concept (redesigned) | Bytes sent are counts for a simulated schedule, not a network measurement. |  |
| 87 | `tensor-and-pipeline-parallel` | Shows its concept (redesigned) |  |  |
| 88 | `zero-sharding` | First experiment, redrawn | Exact storage arithmetic only; no all-gather traffic is measured. |  |
| 89 | `activation-checkpointing` | Shows its concept (redesigned) |  |  |
| 90 | `roofline` | **Partly**: ceilings measured independently; timings vary with machine load |  |  |
| 91 | `multi-armed-bandits` | First experiment, redrawn | The UCB phase uses a different vertical scale from the other two, marked only by 0 and 1. |  |
| 92 | `policy-gradient` | Shows its concept (redesigned) | One small task and network. | Check that the baseline changes learning noise rather than the reward objective. |
| 93 | `actor-critic` | Shows its concept (redesigned) | Final accuracy overlaps the baseline's; only the lower gradient noise is established. |  |
| 94 | `dqn-replay-buffer` | Shows its concept (redesigned) | The correlation is constructed (digits streamed one class at a time); the target network alone does nothing here. |  |
| 95 | `mcts-tic-tac-toe` | First experiment, redrawn | About half the simulations in the third phase end at once on the winning move, leaving the frame sparse. |  |
| 96 | `reward-shaping` | **Partly**: potential-based shaping preserves the policy and a naive bonus is farmed; no speed-up |  |  |
| 97 | `hopfield-network` | Shows its concept (redesigned) | The softmax rule's β is a chosen number. |  |
| 98 | `neural-cellular-automata` | Shows its concept (redesigned) | One image; says nothing about other digits. |  |
| 99 | `neural-ode` | Shows its concept (redesigned) | The step-count effect is large for one of three seeds; the adjoint's gradient is 57% off at 16 steps on this stiff field. |  |
| 100 | `evolution-strategies` | First experiment, redrawn | In the vote phase the arrow is clipped to the enlarged window for the first few pairs. |  |

### Open for all 100

- [ ] None opens with a question the viewer can answer before the result, or closes with one
  limitation. Section 5 of the shared changes asks for both.
- [ ] Most titles are technical names ("Grouped-query attention"). The plain-question form that
  section 5 proposes is used by only a few ("Why divide scores by √d?").
- [ ] No physics bridge from section 4 is on screen in any of them.
- [ ] 44 keep their first experiment. Most are single-seed; where two models are compared, give
  them the seeds the redesigned ones have.

## 100 additional animations

These are **new lessons**, not the redesign tasks above. N001–N100 are proposed IDs, not registered presets. Every entry includes an on-screen experiment, a quantitative check or limitation, and a distinction from the nearest existing material. No expected win is a promised outcome. Whether each can be built under the data rule is classified in the table that follows the proposals.

For implementation, keep numerical results and display data separate. Every lesson needs named quantities/units, data provenance, a baseline, an honest failure case, prerequisites and a transcript. Suggested prerequisite IDs establish a teaching path; they are not a demand to complete all 100 in numerical order.

### A. Start from measurements, without assuming programming knowledge

#### N001. From an experiment to a training table (`experiment-to-training-table`) — Starter

Show five spring measurements becoming five rows: applied force is an input, extension is the target, run ID is bookkeeping. Follow one row into a prediction and back to the apparatus. **Check:** units and row identities survive every view; explain why a post-measurement answer cannot be an input. **Adds:** the data/target distinction missing before `mlp-classifier`. **Needs:** none.

#### N002. Physical parameters versus model weights (`physical-parameters-vs-weights`) — Starter

Fit the same decaying signal with an exponential and a small neural network. Highlight the decay constant in one and all adjustable weights in the other; both produce a curve, but only one directly parameterizes that physical quantity. **Check:** comparable residuals need not imply comparable interpretation. **Adds:** scientific meaning, before architecture. **Needs:** N001.

#### N003. Least squares as a first learning problem (`least-squares-first-model`) — Starter

Move a line through noisy calibration data; draw signed residuals, then their squared contributions to one objective. Compare the fitted coefficients from an exact linear solve and iterative optimization. **Check:** both reach the same optimum under stated full-rank assumptions. **Adds:** an analytically solvable anchor for `gradient-descent`, rather than another optimizer race. **Needs:** N001.

#### N004. From error bars to a loss function (`measurement-likelihood-to-loss`) — Starter

Give two measurements different uncertainties: equal geometric residuals need not be equally surprising. Animate Gaussian likelihood contours becoming weighted least squares, then introduce correlated errors. **Check:** use a known, fixed covariance; if covariance is learned, retain its log-determinant contribution. State the chi-square assumptions and contrast one non-Gaussian case. **Adds:** why choose a loss, beyond `gpt2-softmax-cross-entropy`. **Needs:** N003; optional matrix detail.

#### N005. Three spaces, three different moving dots (`input-feature-parameter-spaces`) — Starter

Use linked views of a measurement in input space, its hidden representation in feature space, and the current model in parameter space. Freeze the input while training, then freeze weights while changing the input. **Check:** only the appropriate objects move in each experiment. **Adds:** a coordinate dictionary for `loss-landscape-descent` and embedding films. **Needs:** N002.

#### N006. An array is a labelled collection (`named-axes-and-shapes`) — Starter

Build a small data block from event, sensor, time and channel axes, using named drawers before showing a shape such as 3×4×2. Slice one event and average one axis. **Check:** count entries and retain units. An ML “tensor” need not obey a physical tensor transformation law. **Adds:** prerequisites for convolution and batching. **Needs:** N001.

#### N007. Matrix multiplication as a measuring device (`matrix-as-linear-instrument`) — Starter

Pass two source strengths through three detector response functions. Animate one output as a weighted sum, then all outputs together; use basis inputs to reveal matrix columns. **Check:** superposition and dimensions, followed by a saturating counterexample. **Adds:** the operation hidden inside MLP and attention diagrams, without teaching syntax. **Needs:** N006.

#### N008. Train, validate, test, then predict (`four-model-lifecycle-states`) — Starter

Follow one model through four rooms with visible permissions: update weights, choose settings, evaluate once, predict new observations. Reusing test results to choose a model visibly turns the test set into a tuning set. **Check:** track which decisions can depend on which data. **Adds:** lifecycle roles before `cross-validation`. **Needs:** N003.

#### N009. One example, one batch, one epoch (`batch-epoch-update-accounting`) — Starter

Deal twelve measurements into batches and count forward evaluations, gradient averages and weight updates. Replay the same epoch with batch sizes one, four and twelve. **Check:** sample exposures and update counts; show the partial-batch convention. **Adds:** basic bookkeeping, distinct from the stochastic effects in `batch-size-and-noise`. **Needs:** N008.

#### N010. A complete no-code learning experiment (`no-code-calibration-workbench`) — Starter

Show a local visual workbench fitting a noisy sensor calibration: choose inputs, reserve data, fit, inspect residuals and predict a held-out point. Every control changes a visible scientific choice. **Check:** replay the saved configuration to reproduce the result. **Adds:** an end-to-end usable mental model connecting the existing isolated concepts. **Needs:** N001–N009; no terminal or code.

### B. Probability and uncertainty for scientific interpretation

#### N011. Probability mass is not probability density (`density-mass-and-units`) — Starter

Refine histogram bins around a measured energy; distinguish bin probability from density, with probability approximated by density times bin width. Change energy units and show numerical density changing while interval probabilities stay fixed. **Check:** integrate to one; finer bins alone do not force density to grow. **Adds:** an interpretation layer before `normalizing-flows` and generative likelihoods. **Needs:** N001; optional integration view.

#### N012. Observing is not intervening (`conditioning-vs-intervention`) — Bridge

Simulate temperature influencing both a sensor and a controller setting. Compare selecting runs with high settings against actively forcing that setting. Draw a small causal graph and the two resulting distributions. **Check:** observational association can survive when the intervention effect is zero. **Adds:** causal semantics beyond correlation and `gpt2-activation-patching`. **Needs:** N011.

#### N013. Bayes' rule on one physical parameter (`bayesian-parameter-update`) — Starter

Infer a decay rate from successive noisy measurements. Multiply a prior density by a likelihood, normalize it, and carry the posterior forward. **Check:** display likelihood as a function of the parameter without calling it a normalized parameter probability. **Adds:** parameter inference, distinct from the function-space focus of `gp-posterior`. **Needs:** N004, N011.

#### N014. Three different kinds of interval (`confidence-credible-predictive-intervals`) — Bridge

Repeat a small experiment and compare a frequentist confidence procedure, a Bayesian credible interval, and an interval for the next noisy observation. **Check:** repeated coverage versus posterior probability versus predictive coverage; label assumptions. **Adds:** distinctions absent from a generic uncertainty band. **Needs:** N013; explain each procedure before overlaying them.

#### N015. Class probability depends on class prevalence (`posterior-odds-and-likelihood-ratios`) — Bridge

Keep two class-conditional detector distributions fixed while changing how often signal occurs. Show the true posterior class probability changing and the likelihood ratio staying fixed; a frozen classifier's raw score need not change. **Check:** posterior odds equal prior odds times the ratio, for known toy densities. **Adds:** physics meaning of a classifier output beyond `roc-and-calibration`. **Needs:** N011, N013.

#### N016. Observation noise versus model uncertainty (`aleatoric-epistemic-uncertainty`) — Starter

Fit noisy data with a gap in the measured range. Collect more observations in the gap, then separately improve sensor resolution. Display uncertainty about the mean and spread of individual outcomes. **Check:** the interventions affect different components; the decomposition depends on the model and available measurements. **Adds:** an operational uncertainty distinction beyond GP bands. **Needs:** N014.

#### N017. Entropy, cross-entropy and KL on the same coins (`information-distances-with-units`) — Bridge

Use three possible detector outcomes with known frequencies. Compare coding lengths under the true distribution and a mismatched model, then reverse the KL direction. **Check:** label bits versus nats, zero-probability failures, and what is averaged. **Adds:** a common interpretation for perplexity, VAEs and DPO; extends their isolated formulas. **Needs:** N011.

#### N018. A regularizer as an explicit prior (`regularization-as-map-estimation`) — Bridge

Fit a two-coefficient physical response with Gaussian or Laplace parameter priors. Overlay likelihood, prior and posterior contours; compare maximum likelihood with maximum a posteriori estimates. **Check:** averaging versus summing data loss changes the regularization coefficient's meaning. **Adds:** assumptions behind `l1-vs-l2`, without claiming all regularization is Bayesian inference. **Needs:** N013, N003.

#### N019. When an ensemble confidently agrees and is wrong (`ensemble-disagreement-limits`) — Bridge

Train independent models with different resampled data or initializations. Animate disagreement near sparse data, then a shared wrong extrapolation. **Check:** compare ensemble spread with actual held-out errors under shift. **Adds:** prediction ensembles and their limitations, distinct from `weight-averaging` in parameter space. **Needs:** N016, N028.

#### N020. Prediction sets with measured coverage (`conformal-prediction-sets`) — Bridge

Calibrate residual ranks on reserved data, construct a prediction interval for a new measurement, and repeat many trials. **Check:** empirical marginal coverage under exchangeability; demonstrate a shifted group where coverage can fail. Do not promise conditional coverage for every input. **Adds:** a coverage-based method beyond probability calibration. **Needs:** N008, N014. Method reference: [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

### C. Computation and calculus, without a programming prerequisite

#### N021. Which variable are we differentiating? (`three-kinds-of-derivative`) — Starter

For a model of a time-dependent measurement, place input sensitivity, parameter gradient and physical time derivative in separate linked panels. Perturb only one variable at a time. **Check:** label what is held fixed and the derivative's units. **Adds:** a distinction obscured when `autodiff-graph` leads directly to training or a neural ODE. **Needs:** N005.

#### N022. Push a perturbation forward, pull a sensitivity back (`jacobian-jvp-vjp`) — Bridge

Use a two-input, three-output instrument model. Push a small input arrow through its Jacobian, then pull an output objective's sensitivity back through the transpose. **Check:** directional finite differences and the matching scalar inner product. **Adds:** the meaning of forward/reverse derivative products beyond the scalar chain-rule graph. **Needs:** N007, N021.

#### N023. Why a smaller finite difference can be worse (`finite-difference-step-size`) — Bridge

Estimate a known derivative using successively smaller perturbations. Animate approximation error falling and roundoff eventually taking over. **Check:** compare forward and central differences against the analytic answer on a log scale. **Adds:** derivative verification and numerical conditioning, distinct from `mixed-precision` model performance. **Needs:** N021; define logarithmic axes first.

#### N024. A silent wrong-axis calculation (`broadcasting-and-axis-mistakes`) — Starter

Subtract sensor offsets from a square event-by-sensor table. Animate the intended repetition of offsets, then a shape-compatible but wrong event-wise subtraction. **Check:** trace one labelled entry; a non-square diagnostic example exposes an error the square shape hid. Identical shapes are not evidence of meaning. **Adds:** a concrete programming hazard visible without syntax. **Needs:** N006.

#### N025. Units change the optimization geometry (`nondimensionalization-and-gradients`) — Bridge

Fit one physical problem expressed in metres and seconds, then millimetres and milliseconds. Show the same predictions but different raw gradient steps; nondimensionalize using stated reference scales. **Check:** map parameters back to identical physical units. **Adds:** unit-aware conditioning, not the activation statistics in `normalisation`. **Needs:** N003, N021.

#### N026. Singular modes reveal what data can determine (`svd-and-identifiable-directions`) — Bridge

Rotate a two-parameter calibration problem into strongly and weakly measured directions. Add noise and show the weak direction becoming unstable. **Check:** singular values, parameter covariance under stated assumptions, and truncated reconstruction. **Adds:** data identifiability, distinct from rank compression of a neural layer in `rank-of-a-layer`. **Needs:** N007, N004.

#### N027. Keep tiny probabilities in logarithmic form (`log-probability-arithmetic`) — Starter

Multiply probabilities for a long sequence until ordinary arithmetic underflows; repeat as a sum of log probabilities and retain the log result. Show stable log-sum-exp on three values. **Check:** against a high-precision reference; converting back can still underflow if the probability is unrepresentable. **Adds:** numerical meaning of sequence likelihood, before `flash-attention-tiling`. **Needs:** N011, N017.

#### N028. One random seed is one experiment (`seed-variation-and-replication`) — Starter

Repeat the same model comparison over paired data splits and several initializations. Replace a convincing single race with a distribution of paired differences. **Check:** separate data variation from initialization variation and report the number of repeats. **Adds:** evidence strength across the catalogue's optimizer/architecture races. **Needs:** N008; not a lesson in seed APIs.

#### N029. Where a model's memory goes (`model-memory-accounting`) — Starter

Build a tiny network as labelled boxes of weights, activations, gradients and optimizer state. Change batch size, precision and inference/training mode, keeping a running byte count. **Check:** explicit shape×bytes arithmetic; distinguish estimates from actual peak allocation. **Adds:** prerequisites for `roofline`, checkpointing and sharding, rather than another hardware benchmark. **Needs:** N006, N009.

#### N030. A flat training curve does not prove completion (`stopping-and-convergence`) — Bridge

Compare a genuinely converged fit, a badly scaled stalled fit and a model beginning to overfit. Show objective change, gradient norm and validation error separately. **Check:** stopping thresholds and units; reserve the test set. **Adds:** diagnosing when to stop, distinct from learning-rate schedules or `overfitting` alone. **Needs:** N008, N025.

### D. Evaluation that matches a scientific experiment

#### N031. Split by experiment, not by nearby rows (`grouped-correlated-data-splits`) — Starter

Create many overlapping windows from several detector runs. Randomly split windows, then hold out entire runs. Animate near-duplicate information crossing the first split. **Check:** first use independent runs from the same generator to isolate leakage; introduce an offset shift as a separate stress test. **Adds:** dependence and the unit of generalization beyond ordinary `cross-validation`. **Needs:** N008, N009.

#### N032. Preprocessing can leak the test set (`preprocessing-data-leakage`) — Starter

Estimate scaling, imputation and a principal axis using all measurements, then repeat with those steps fitted only inside training folds. A data-flow trace marks where held-out information entered. **Check:** show a constructed leakage-sensitive case and a case with negligible effect. **Adds:** pipeline leakage, distinct from model-selection optimism. **Needs:** N008, N006.

#### N033. Bootstrap the independent unit (`bootstrap-with-dependent-data`) — Bridge

Estimate uncertainty on an average from an autocorrelated signal. Compare resampling individual readings with resampling whole runs or suitable blocks. **Check:** repeated synthetic experiments reveal coverage and sensitivity to block length. **Adds:** resampling uncertainty under dependence, not classifier calibration. **Needs:** N014, N031.

#### N034. Three ways the world changes (`covariate-label-concept-shift`) — Bridge

Construct separate examples of changed input distribution, class prevalence, and input–target relation. Keep a joint-distribution table visible. **Check:** preserve P(y|x) for covariate shift and P(x|y) for label shift; state the changed conditional in concept shift. Identify which corrections are justified. **Adds:** distinct failure mechanisms behind a generic out-of-distribution label. **Needs:** N011, N015.

#### N035. Choose a trigger under a fixed event budget (`rare-event-trigger-budget`) — Bridge

Send a rare-signal stream through a classifier with a fixed storage rate and finite downstream processing capacity. Move the threshold to show accepted signal, false alarms, queueing and dropped events per second. **Check:** select a threshold under an explicit cost/bandwidth constraint. **Adds:** an operational trigger-design decision beyond existing ROC/base-rate sweeps. **Needs:** N015; accuracy is not discovery significance.

#### N036. Missing measurements can carry selection information (`missingness-and-censoring`) — Bridge

Compare randomly dropped readings, saturation and events lost below a trigger threshold. Distinguish censored observations with known limits from truncated events absent from the sample. **Check:** naive imputation versus an explicit observation model on known truth. **Adds:** selection/missingness assumptions absent from clean MNIST and synthetic tables. **Needs:** N011, N001.

#### N037. Calibration after a detector change (`calibration-under-dataset-shift`) — Bridge

Take a model calibrated on one detector condition into another. Show reliability and class prevalence before and after recalibration using new reserved labels. **Check:** overall and subgroup results; report where a simple correction fails. **Adds:** transfer and maintenance of calibration, rather than repeating `gpt2-calibration` or `roc-and-calibration`. **Needs:** N034, N015.

#### N038. Let the model abstain (`selective-prediction-risk-coverage`) — Bridge

Route uncertain events to a slower reference analysis while accepting the rest automatically. Move the abstention threshold to trace error versus retained fraction. **Check:** include a confidently wrong shifted group and the actual review cost. **Adds:** a decision policy using uncertainty, beyond measuring uncertainty. **Needs:** N016, N034.

#### N039. The model learned the apparatus label (`shortcut-learning-counterfactuals`) — Starter

Correlate a harmless detector watermark with signal class. Train a classifier, then swap or remove the watermark while preserving the simulated event. **Check:** paired counterfactual predictions and performance on a new apparatus. **Adds:** a scientifically interpretable shortcut experiment beyond adversarial pixel perturbations or saliency. **Needs:** N008, N012.

#### N040. Choose the next measurement to learn more (`active-learning-experiment-selection`) — Bridge

Fit a response curve and choose the next labelled measurement by uncertainty, diversity or random sampling. Animate how each choice changes model error over a fixed measurement budget. **Check:** repeat noisy trials and include model misspecification. **Adds:** learning the whole response efficiently, distinct from seeking the maximum in `gp-bayesian-optimisation`. **Needs:** N016, N028.

### E. Symmetry as a reason to choose an architecture

The mathematical connection between symmetry and architecture is developed in [Geometric Deep Learning](https://arxiv.org/abs/2104.13478). The following small experiments are proposed teaching designs, not reported results from that source.

#### N041. Invariance and equivariance are different promises (`invariance-vs-equivariance`) — Bridge

Rotate a simulated particle configuration. Its predicted total energy should remain fixed, while predicted forces should rotate. Animate the two routes around a commuting square: transform then predict, or predict then transform. **Check:** numerical discrepancy under unseen rotations. **Adds:** a precise symmetry test beyond augmentation or convolution. **Needs:** N007, N005.

#### N042. A particle collection has no preferred ordering (`permutation-invariant-sets`) — Bridge

Shuffle identical event objects and vary their count. Compare an order-sensitive network with a shared per-object transform followed by sum or mean pooling. **Check:** permutation invariance and the different treatment of extensive quantities by sum and mean. **Adds:** variable-sized sets, absent from fixed-grid classifier examples. **Needs:** N041, N006.

#### N043. Message passing between interacting particles (`graph-message-passing`) — Bridge

Draw particles as nodes and candidate interactions as edges. Follow one pairwise message, its aggregation at a node and the updated prediction; expand the neighborhood by a second layer. **Check:** node permutation tests and a disconnected interaction the graph cannot communicate. **Adds:** graph neural networks, distinct from attention over an ordered token sequence. **Needs:** N042.

#### N044. Scalar and vector features under rotation (`rotation-equivariant-features`) — Advanced

Pass scalar masses and vector displacements through an equivariant particle model. Rotate the scene while arrows rotate and scalar values stay fixed. **Check:** energy invariance and force equivariance separately, including numerical tolerance. **Adds:** transformation-aware internal features, beyond N041's external test and ordinary `augmentation`. **Needs:** N041, N043; optional representation-theory detail.

#### N045. Mirror images can matter (`parity-chirality-and-pseudoscalars`) — Advanced

Mirror four physically distinguishable labelled points. Distances remain unchanged while their signed triple product flips. Ask two models to distinguish handedness. **Check:** label ordering matters: odd permutations also flip that product. For indistinguishable points, construct an appropriate permutation-invariant target instead. **Adds:** limits of distance-only representations beyond rotational equivariance. **Needs:** N044; explain the triple product geometrically.

#### N046. Boundaries break a convolution's symmetry (`translation-equivariance-and-boundaries`) — Bridge

Translate a localized signal across the interior and edge of a detector. Compare periodic, zero-padded and valid convolution at stride one. **Check:** the shift-then-filter versus filter-then-shift residual; add stride as a separate controlled experiment. **Adds:** a symmetry guarantee and its failure conditions, rather than another padding/stride mechanics lesson. **Needs:** N041 and existing convolution lessons.

#### N047. Build a conservation law into the output (`conservative-network-outputs`) — Advanced

Predict exchanges between cells as equal-and-opposite edge fluxes, then sum them into cell updates. Compare with independent unconstrained outputs. **Check:** global conservation to numerical precision under closed boundaries, versus accuracy and open-boundary flux. **Adds:** exact structural constraints rather than hoping training learns conservation. **Needs:** N043; connect to finite-volume bookkeeping.

#### N048. Local coordinates should not change the answer (`local-frames-and-gauge-equivariance`) — Advanced

Give nearby sensors different local coordinate frames. Rotate those frames without moving the physical vectors; transport messages consistently before combining them. **Check:** the reconstructed global prediction remains unchanged. **Adds:** local-frame compatibility beyond a single global rotation. **Needs:** N044. Call this a coordinate-frame example; do not imply it demonstrates all properties of a physical gauge theory.

#### N049. Lorentz structure in event representations (`lorentz-structured-event-models`) — Advanced

Boost a few simulated four-momenta between inertial frames. Compare invariant-mass prediction with a four-vector output that must transform. **Check:** Minkowski products, sign convention and transformation residuals; show detector cuts separately. **Adds:** spacetime symmetry beyond Euclidean point-cloud models. **Needs:** N041, N042; introduce four-vector notation or offer a prerequisite route.

#### N050. Do not impose a symmetry the detector violates (`symmetry-of-law-vs-observation`) — Bridge

Apply a rotationally symmetric physical generator, then pass events through a detector with a dead sector. Compare rotating latent events with rotating already observed data. **Check:** performance when detector orientation changes, with and without explicit geometry conditioning. **Adds:** when an attractive inductive bias becomes wrong, extending augmentation and N041–N049. **Needs:** N036, N041.

### F. Learning dynamics without confusing fit, integration and physics

#### N051. A good next step can make a bad long forecast (`one-step-vs-rollout-error`) — Bridge

Train a small predictor on neighboring oscillator states. Compare teacher-supplied inputs with repeatedly feeding predictions back, showing energy and phase error over time. **Check:** evaluate entire unseen trajectories, not randomly mixed time points. **Adds:** quantitative rollout stability for continuous dynamics, beyond language `teacher-forcing`. **Needs:** N031, N021.

#### N052. Learn a Hamiltonian, then differentiate it (`hamiltonian-neural-dynamics`) — Advanced

Learn a scalar H(q,p), derive its vector field through Hamilton's equations, and compare with an unconstrained vector-field model. **Check:** trajectory error and energy drift with the same numerical solver; distinguish learned-H conservation from agreement with true energy. **Adds:** a mechanical structural constraint beyond `neural-ode`. **Needs:** N021, N051. Reference: [Hamiltonian Neural Networks](https://arxiv.org/abs/1906.01563).

#### N053. Learn a Lagrangian in configuration coordinates (`lagrangian-neural-dynamics`) — Advanced

Start with positions and velocities of a simple coupled oscillator. Show a learned Lagrangian producing equations of motion through derivatives, rather than directly emitting acceleration. **Check:** held-out trajectories and the invertibility/conditioning of the velocity Hessian. **Adds:** a distinct mechanics-based representation from a generic ODE or Hamiltonian model. **Needs:** N022, N051; teach Euler–Lagrange notation before using it. Reference: [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630).

#### N054. The numerical integrator is part of the model (`symplectic-vs-generic-rollouts`) — Advanced

Hold a separable learned Hamiltonian H=T(p)+V(q) fixed and integrate it with explicit Euler and leapfrog. Track a phase-space patch and long-run energy behavior. **Check:** report step size and evaluation budget separately; symplectic does not mean exact energy conservation. A generic nonseparable H needs a suitable different solver. **Adds:** separates solver and model properties. **Needs:** N052.

#### N055. Discover a sparse dynamical equation (`sparse-equation-identification`) — Advanced

Turn observed oscillator trajectories into candidate columns such as x, v and x³. Show sparse regression choosing terms, then integrate the resulting equation on a new trajectory. **Check:** derivative noise, candidate-library dependence and wrong-but-fitting laws. **Adds:** equation discovery, rather than sparsity of neural weights. **Needs:** N018, N023, N051. Reference: [SINDy](https://arxiv.org/abs/1509.03580).

#### N056. Linear evolution in learned observables (`koopman-observable-learning`) — Advanced

Lift a nonlinear trajectory into a few observable functions, apply a linear evolution matrix there, and decode back. Compare against linear evolution in the original coordinates. **Check:** multi-step errors, truncation and out-of-range initial conditions; no claim that a finite lift exactly linearizes arbitrary dynamics. **Adds:** dynamical representation learning beyond static autoencoding. **Needs:** N007, N051.

#### N057. Learn drift and diffusion separately (`neural-stochastic-dynamics`) — Advanced

Generate many noisy trajectories from identical initial states. Learn drift from mean increment/Δt and BBᵀ from increment covariance/Δt, then simulate stochastic paths. **Check:** distributions at several times, not just the mean; state the Itô convention and whether “diffusion” means B or BBᵀ/2. **Adds:** random evolution beyond deterministic `neural-ode`. **Needs:** N016, N051.

#### N058. Chaos sets a forecast horizon (`chaos-and-statistical-prediction`) — Advanced

Start nearby states in a chaotic system and compare true divergence with learned-model error. Let trajectory agreement fail, then compare invariant statistics and return-time distributions. **Check:** distinguish numerical error, model error and sensitivity to initial conditions. **Adds:** appropriate success criteria for dynamics beyond endpoint error. **Needs:** N051; explicitly avoid equating plausible attractor shape with correct physics.

#### N059. Reduced coordinates still need a closure model (`reduced-order-dynamics-and-closure`) — Advanced

Project a simulated field onto a few dominant modes. Show lost modes feeding back into retained ones and compare a truncated equation with a learned closure term. **Check:** long-time energy and stability on unseen forcing. **Adds:** the dynamical consequences of compression beyond `pca-and-kmeans` or static autoencoders. **Needs:** N051, N007.

#### N060. Differentiate through a physical simulator (`differentiable-simulator-parameter-fit`) — Advanced

Fit a damping coefficient by sending a loss gradient backward through an explicit numerical simulation. Contrast simulator parameters with network weights. **Check:** finite-difference agreement over a useful step-size range and sensitivity to solver resolution. **Adds:** optimization through known physics, not learning the evolution rule itself. **Needs:** N002, N022, N023.

### G. Scientific fields, differential equations and surrogate models

Physics-informed residual training and learning a function-to-function operator are distinct constructions, illustrated respectively by the [original PINN work](https://arxiv.org/abs/1711.10561) and [Fourier Neural Operator paper](https://arxiv.org/abs/2010.08895). The proposed lessons deliberately test errors and assumptions as well as showing the constructions.

#### N061. A differential equation can supply training targets (`pinn-residual-training`) — Bridge

Represent a one-dimensional solution by a network of coordinate inputs. Use derivatives to evaluate its differential-equation residual between observations, alongside boundary and data losses. **Check:** compare to an analytic solution on a dense independent grid. **Adds:** physics-informed training beyond fitting observed points or a neural ODE. **Needs:** N021, N004.

#### N062. Competing losses can hide a bad solution (`pinn-loss-balance-and-stiffness`) — Advanced

Plot boundary, observation and differential-equation losses separately while fitting a problem with different scales. Change their weights and display where the solution fails physically. **Check:** dimensionless scaling and gradient magnitudes; a small total loss alone is insufficient. **Adds:** scientific multi-term objectives beyond a generic regularizer race. **Needs:** N061, N025.

#### N063. Satisfy a boundary condition by construction (`hard-boundary-condition-ansatz`) — Advanced

Build a trial function from a known boundary-satisfying part plus a vanishing factor times a network. Animate why the boundary stays correct for any weights. **Check:** compare interior error with a soft-penalty model and disclose restricted boundary/geometry assumptions. **Adds:** exact function-space constraints rather than tuning PINN penalty weights. **Needs:** N061.

#### N064. Residual checks can miss what lies between samples (`collocation-aliasing-and-validation`) — Advanced

Train against a sparse set of residual points where an oscillatory error happens to vanish. Reveal the dense residual field and refine the sample set. **Check:** independent-grid error, boundary behavior and adaptive-refinement cost. **Adds:** numerical-analysis failure modes in physics-informed validation, distinct from ordinary train/test overfitting. **Needs:** N061, N031.

#### N065. Learn a map between whole functions (`neural-operator-vs-coordinate-model`) — Advanced

Feed several initial temperature fields into a model that predicts later fields. Contrast this with a coordinate network fitted to one solution. **Check:** hold out entire input functions and specify their distribution. **Adds:** operator learning across a family of problems, beyond image-to-image architecture or one PINN. **Needs:** N006, N061.

#### N066. Fourier modes as learned communication channels (`fourier-neural-operator-mechanics`) — Advanced

Transform feature fields into Fourier space, mix feature channels at each retained wavevector, transform back, add the local linear branch and apply a nonlinearity, which can couple modes. **Check:** truncation, aliasing and boundary assumptions on smooth and discontinuous fields. **Adds:** the spectral operator mechanism beyond generic convolution or N065's input/output definition. **Needs:** N065; offer a Fourier primer.

#### N067. Combine a cheap simulator with scarce accurate data (`multifidelity-residual-surrogate`) — Advanced

Use a coarse physical simulation as a baseline and learn a correction from a few fine-resolution runs. Show both predictions and the correction field. **Check:** compare error per expensive simulation against direct fitting; test where the baseline is systematically wrong. **Adds:** scientific fidelity hierarchies beyond generic `transfer-learning`. **Needs:** N002, N065.

#### N068. When does a surrogate actually save work? (`surrogate-break-even-analysis`) — Bridge

Add simulation-data generation, training and validation costs before counting fast surrogate predictions. Plot cumulative cost against the number of queries at a stated accuracy requirement. **Check:** show break-even ranges and retraining after distribution change. **Adds:** end-to-end scientific cost, distinct from the operation throughput in `roofline`. **Needs:** N065, N028.

#### N069. A new grid is not automatically the same problem (`resolution-transfer-and-consistency`) — Advanced

Evaluate an operator model on finer grids and on input fields containing genuinely new high-frequency content. Contrast mere resampling with new information. **Check:** physical-coordinate error norms, convergence and memory cost on independent reference solutions. **Adds:** what cross-resolution claims mean, beyond convolutional image sizing. **Needs:** N065, N066.

#### N070. A small surrogate error can bias a physical fit (`surrogate-error-in-inference`) — Advanced

Infer a physical coefficient using a reference simulator, then substitute a fast surrogate with a small structured bias. Animate residuals and the displaced parameter interval. **Check:** repeated coverage with and without a discrepancy model, which can itself confound the physical parameter. **Adds:** approximation error propagated into scientific inference beyond predictive MSE. **Needs:** N014, N060, N067.

### H. Inverse problems and simulation-based inference

The distinction between generating simulated observations and inferring the parameters that produced observations motivates this branch; [Cranmer, Brehmer and Louppe](https://arxiv.org/abs/1911.01429) review that setting. Each proposal below adds a different inference or validation step, rather than treating one impressive posterior picture as the whole workflow.

#### N071. The forward problem can be easy and the inverse unstable (`forward-vs-inverse-problems`) — Bridge

Blur a known source through a detector response, add noise, then attempt reconstruction. Animate two different sources producing nearly indistinguishable observations. **Check:** residual fit versus source error as regularization changes. **Adds:** ill-posed physical inference rather than compression of learned features; applies the singular-mode idea from N026. **Needs:** N026, N004.

#### N072. Unfold a detector response and test closure (`detector-unfolding-and-closure`) — Advanced

Move events between true and reconstructed histogram bins, including inefficiency. Reconstruct an underlying spectrum with a stated regularized method, then forward-fold it. **Check:** closure on fresh pseudo-data and a truth shape unlike the training/reference sample; show induced bin correlations. **Adds:** a complete detector-correction task beyond generic inverse-problem geometry. **Needs:** N071, N033.

#### N073. Infer by simulating and accepting nearby outcomes (`approximate-bayesian-computation`) — Bridge

Draw a physical parameter from the prior, simulate data and accept when summary statistics lie within a tolerance of observations. Shrink tolerance while watching acceptance and cost. **Check:** compare with an exact posterior; even vanishing tolerance conditions on the chosen summaries, which can be insufficient. **Adds:** likelihood-free inference before neural approximations. **Needs:** N013.

#### N074. Train a network to return a posterior distribution (`neural-posterior-estimation`) — Advanced

Draw training parameters from a stated prior, simulate observations and train a conditional density model mapping observations to parameter distributions. Reuse it on new measurements. **Check:** exact toy posterior, prior support and out-of-domain observations; a different training proposal can require correction. **Adds:** amortized inverse inference, not generic generation. **Needs:** N073, N011; distinguish network weights from physical parameters.

#### N075. Learn the likelihood, then perform inference (`neural-likelihood-estimation`) — Advanced

Train a conditional distribution of observations given physical parameters. Freeze it, hold the measured observation fixed and scan parameter settings to form a likelihood. **Check:** density normalization in observation space and posterior error against a tractable reference. **Adds:** reverses N074's conditioning and separates emulation from inference. **Needs:** N013, N011; compare with N074 optionally.

#### N076. Learn a likelihood ratio by classification (`neural-ratio-estimation`) — Advanced

Classify simulations from two parameter settings, then convert calibrated posterior odds into a density ratio using known class proportions. **Check:** compare the learned ratio against an analytic example, including sparse tails. **Adds:** a trainable inference construction beyond N015's exact probability identity and ordinary classification. **Needs:** N015, N037.

#### N077. What to do with nuisance parameters (`profiling-vs-marginalization`) — Advanced

Infer signal strength with an unknown detector offset. Show a two-dimensional likelihood; either maximize likelihood over the nuisance parameter or integrate likelihood times its stated conditional prior. **Check:** compare the resulting curves and interval interpretations; these are different operations. **Adds:** systematic uncertainty beyond a single-parameter fit. **Needs:** N013, N014.

#### N078. A second experiment can break a degeneracy (`identifiability-and-complementary-data`) — Bridge

Show two physical parameters trading off along a ridge under one measurement. Add a second type of observation with a differently oriented constraint. **Check:** joint versus separate intervals and sensitivity to shared systematics. **Adds:** experiment complementarity and non-identifiability, not just neural model capacity. **Needs:** N026, N013; N077 for the optional shared-systematics extension.

#### N079. Does the fitted model reproduce features of the observed data? (`posterior-predictive-checks`) — Bridge

Sample parameters from a posterior, generate replicate data, and compare an unused diagnostic shape statistic with the real observation. Let the fitted mean agree while tails disagree. **Check:** distinguish model criticism from a probability that the model is true. **Adds:** generative validation after inference, beyond held-out prediction error. **Needs:** N013; N074 is optional.

#### N080. Test the inference algorithm on known simulated truths (`simulation-based-calibration`) — Advanced

Draw parameters from the prior, simulate data, infer repeatedly and rank each true parameter among posterior draws. **Check:** rank histograms with finite-sample uncertainty and deliberately biased/narrow posteriors; account for correlated draws. **Adds:** inference calibration rather than classifier calibration. A passing rank check alone is not proof of informative inference. **Needs:** N074, N014. Reference: [Talts et al.](https://arxiv.org/abs/1804.06788).

### I. Statistical physics and generative modelling

#### N081. From an energy function to a probability distribution (`energy-model-and-partition-function`) — Bridge

Enumerate a tiny spin system, assign energies and normalize Boltzmann weights explicitly. Change an energy parameter and watch probabilities redistribute. **Check:** partition function and temperature convention; distinguish sampling from minimizing energy. **Adds:** normalized probabilistic meaning beyond deterministic `hopfield-network` retrieval. **Needs:** N011, N017.

#### N082. Why an energy model needs negative samples (`energy-model-positive-negative-phases`) — Advanced

Fit a tiny energy-based model with exact enumeration first. Contrast statistics measured from data with statistics sampled from the current model, then replace exact samples with short chains. **Check:** gradient bias and poor chain mixing. **Adds:** positive/negative-phase learning, absent from the Hopfield and GAN demonstrations. **Needs:** N081, N022.

#### N083. A score field points toward higher density (`score-field-and-noise-level`) — Advanced

Draw a one-dimensional mixture density and its gradient of log density; add noise and watch the score field change with noise level. **Check:** analytic versus learned scores, including low-density tails. **Adds:** what diffusion models estimate, beyond the trajectories in `diffusion`, `ddpm-vs-ddim` and `flow-matching`. **Needs:** N011, N021.

#### N084. The best denoised average may look like no sample (`conditional-mean-denoising`) — Bridge

Create two plausible clean signals that can produce the same noisy observation. Show that minimizing squared reconstruction error returns a conditional mean between them, while posterior sampling returns different plausible signals. **Check:** average error versus distributional fidelity. **Adds:** the distinction between denoising and generating, beyond a noise-removal montage. **Needs:** N004, N013.

#### N085. Detailed balance and autocorrelation in a tiny sampler (`mcmc-balance-and-mixing`) — Bridge

Animate a Metropolis chain on an enumerated energy landscape, showing proposal, acceptance and rejection as separate steps. Count transition fluxes and repeated-state correlation. **Check:** the correct stationary histogram can coexist with poor effective sample size. **Adds:** sampling correctness and efficiency beyond optimization's `drunken-walk`. **Needs:** N081.

#### N086. Correct a learned proposal with an accept/reject step (`flow-proposals-with-metropolis-correction`) — Advanced

Use a continuous invertible flow to propose scalar-field configurations, then apply the independence-Metropolis ratio against a known target density. **Check:** observables, rejection, autocorrelation and proposal support; target stationarity does not make finite samples independent or exactly representative. **Adds:** target correction beyond `normalizing-flows`. **Needs:** N085 and the existing flow lesson. Reference context: [Albergo et al.](https://arxiv.org/abs/2101.08176).

#### N087. Spend simulations where rare events happen (`rare-event-importance-sampling`) — Advanced

Estimate a tiny tail probability by ordinary sampling and a deliberately shifted proposal. Attach the likelihood-ratio weight to every sample. **Check:** unbiasedness assumptions, effective sample size, variance over repeats and a proposal that misses part of the support. **Adds:** weighted rare-event estimation beyond balanced classification or sampling speed. **Needs:** N011, N028.

#### N088. Reweighting works only where ensembles overlap (`free-energy-reweighting-overlap`) — Advanced

Reweight a small exactly enumerable system from one temperature or parameter value to another. Animate a few configurations taking nearly all the weight when overlap collapses. **Check:** free-energy differences against exact sums, uncertainty and effective sample size. **Adds:** thermodynamic observables and overlap failure, beyond N087's tail-probability target. **Needs:** N081, N087.

#### N089. Coarse-graining is not automatically renormalization (`coarse-graining-vs-latent-compression`) — Advanced

Block-average a lattice configuration and compare with a learned bottleneck. Track specified long-wavelength observables and information lost at each step. **Check:** whether a claimed effective distribution reproduces those observables across scales. **Adds:** a careful physics comparison to `information-bottleneck`; compression alone does not establish a renormalization-group transformation. **Needs:** N081 and existing bottleneck lessons.

#### N090. Find a phase boundary without learning a simulator artifact (`phase-discovery-and-nuisance-controls`) — Advanced

Embed or cluster configurations across a known phase diagram. Add a seed-correlated watermark, then remove it and repeat. Vary lattice size separately as a physical finite-size control. **Check:** relation to an independent order parameter and held-out conditions; lattice size is not merely an arbitrary nuisance. **Adds:** scientific interpretation beyond attractive t-SNE clusters. **Needs:** N039, N089.

### J. Physics capstones that combine the ideas

These are deliberately later lessons. Each should begin with the physical measurement and a conventional baseline, then introduce only the ML component needed for that task. They are not claims of research-grade performance from small teaching datasets.

#### N091. Reconstruct tracks from unordered detector hits (`graph-based-particle-tracking`) — Advanced

Generate simple charged-particle tracks, noise hits and missing layers. Build candidate hit edges, score them with a graph model and assemble tracks. **Check:** efficiency, fake tracks, ambiguities and performance as occupancy rises against a geometric baseline. **Adds:** graph inference with variable-size inputs, not rectangular object detection. **Needs:** N043, N036.

#### N092. A plausible shower must also have the right observables (`calorimeter-generative-validation`) — Advanced

Condition a small generator on incoming energy and visualize simulated calorimeter deposits. Compare sample images, deposited-energy sums, shower shapes, correlations and tails with a reference simulation. **Check:** conservation assumptions, leakage and rare-event discrepancies. **Adds:** physically validated conditional generation beyond digit-image realism. **Needs:** N047, N079 and existing generative lessons.

#### N093. A learned detector versus a matched filter (`waveform-detection-baseline`) — Advanced

Inject simple parameterized chirps into controlled noise. Compare a matched-filter detector and a small learned detector as noise becomes nonstationary. **Check:** detection efficiency at a fixed false-alarm rate with independent noise realizations. **Adds:** a domain baseline and distribution-shift stress test rather than accuracy alone. **Needs:** N035, N034; label toy chirps rather than claiming a full observatory pipeline.

#### N094. A survey observes a selected population (`astronomical-selection-and-inference`) — Advanced

Generate a source population with latent brightness and distance, apply a survey threshold, and infer a population property from detections. **Check:** compare naive inference with one including the selection function and its uncertainty. **Adds:** population inference from truncated observations, distinct from predicting individual examples. **Needs:** N036, N077.

#### N095. Conservative forces from a learned molecular energy (`conservative-learned-force-fields`) — Advanced

Learn a tiny simulated molecule's energy surface and obtain forces as its negative coordinate gradient. Compare independently predicted forces. **Check:** rotation behavior, force error, closed-loop work and held-out rollouts; conservative forces do not guarantee exact energy conservation by a finite-step integrator. **Adds:** scientific outputs and integrability beyond scalar regression or equivariance alone. **Needs:** N044, N052, N054.

#### N096. Control an oscillator with explicit physical constraints (`constrained-oscillator-control`) — Advanced

Control a simulated damped oscillator using a conventional controller and a learned policy. Display state, action limits, reward components and constraint violations together. **Check:** disturbances, parameter shift, accumulated cost and failure rate; reward alone is insufficient. **Adds:** continuous physical control with constraints beyond gridworld reinforcement learning. **Needs:** N051 and existing actor-critic/reward-shaping lessons.

#### N097. Search for a formula and test whether it is a law (`symbolic-regression-and-dimensions`) — Advanced

Fit noisy measurements with candidate expressions arranged by error and complexity. Eliminate dimensionally inconsistent forms and challenge surviving expressions outside the fitted range. **Check:** known-law recovery, alternative fits and identifiability under limited data. **Adds:** expression search beyond N055's fixed candidate library and generic network fitting. **Needs:** N002, N025, N028.

#### N098. Trade accuracy, cost and constraints on a Pareto front (`scientific-multiobjective-design`) — Bridge

Design a simulated instrument with competing resolution, cost and power requirements. Move between feasible designs without collapsing all goals into one hidden score. **Check:** show dominated designs, uncertainty and how a scalar weighting misses preferences or parts of a nonconvex front. **Adds:** multiobjective scientific choice beyond single-objective Bayesian optimization. **Needs:** N040, N016.

#### N099. Reconstruct a quantum state that remains physical (`physical-quantum-state-tomography`) — Advanced

Reconstruct a qubit density matrix from simulated measurement counts with a learned map and a constrained maximum-likelihood baseline. Compare unconstrained outputs with a Hermitian, positive-semidefinite, unit-trace parameterization. **Check:** measurement probabilities, eigenvalues and uncertainty versus shot count. **Adds:** a physical output space beyond generic matrix regression. **Needs:** N007, N014; introduce density matrices before comparing methods.

#### N100. From raw observations to an auditable scientific result (`end-to-end-scientific-closure`) — Advanced

Follow one simulated resonance/decay measurement through selection, learned correction and physical parameter inference. Compare a conventional baseline and an ML-assisted pipeline with a sealed final sample. **Check:** closure, interval coverage, systematic shifts and reproducibility from saved artifacts. **Adds:** a complete scientific claim, beyond N010's introductory predictor. **Needs:** N008, N070, N072, N077–N080; split into short linked chapters if necessary.


## The proposals against the data rule

A first pass, made from each proposal's title and opening sentence, not from a full reading of
its experiment; confirm it when a lesson is picked up.

- **Real data (38):** the lesson's point can be made on MNIST or Tiny Shakespeare. Where the
  proposal names a spring, a detector or a decaying signal, the substitute is given; the physical
  picture can stay as a labelled analogy.
- **Exact (23):** arithmetic, algebra or an enumerable system, with no learned-model result.
  Allowed as they are, labelled exact.
- **Needs simulation (39):** the lesson is about a physical system whose truth must be known
  (decision 1 above).

| ID | Lesson | Level | Under the data rule | How, or what is missing |
|---|---|---|---|---|
| N001 | From an experiment to a training table | Starter | Exact | bookkeeping on a table; the five measurements are an illustration |
| N002 | Physical parameters versus model weights | Starter | Real data | an exponential-family fit against a small network on a real MNIST measurement (ink per column) |
| N003 | Least squares as a first learning problem | Starter | Real data | left-half against right-half ink of MNIST digits is nearly a straight line (measured in `bias-variance`'s first version) |
| N004 | From error bars to a loss function | Starter | Real data | per-pixel variances of MNIST as the unequal error bars; the contour algebra is exact |
| N005 | Three spaces, three different moving dots | Starter | Real data | the MNIST MLP: a digit, its hidden vector, the weights |
| N006 | An array is a labelled collection | Starter | Exact | counting entries of a labelled block |
| N007 | Matrix multiplication as a measuring device | Starter | Exact | a 3×2 matrix acting on basis inputs |
| N008 | Train, validate, test, then predict | Starter | Real data | any MNIST classifier with train, validation and test splits |
| N009 | One example, one batch, one epoch | Starter | Exact | counting examples, batches and updates |
| N010 | A complete no-code learning experiment | Starter | Real data | the same workbench fitting a real MNIST measurement |
| N011 | Probability mass is not probability density | Starter | Real data | histogram of real ink totals at finer and finer bins |
| N012 | Observing is not intervening | Bridge | Needs simulation | needs a system whose causal structure is known; as an exact probability table it is an illustration only |
| N013 | Bayes' rule on one physical parameter | Starter | Real data | infer the share of 7s in MNIST from growing samples; the whole set gives the true value |
| N014 | Three different kinds of interval | Bridge | Real data | MNIST as the population: repeated samples are the repeated experiments |
| N015 | Class probability depends on class prevalence | Bridge | Real data | a real classifier's scores at changed class prevalence (as `roc-and-calibration` does by arithmetic) |
| N016 | Observation noise versus model uncertainty | Starter | Real data | a regression on MNIST with a gap in the inputs; more data against less noise |
| N017 | Entropy, cross-entropy and KL on the same coins | Bridge | Real data | character frequencies of Tiny Shakespeare: code lengths under the true and a mismatched table |
| N018 | A regularizer as an explicit prior | Bridge | Real data | a two-coefficient fit on MNIST with Gaussian and Laplace priors |
| N019 | When an ensemble confidently agrees and is wrong | Bridge | Real data | an ensemble of MNIST models, on digits and on non-digit inputs |
| N020 | Prediction sets with measured coverage | Bridge | Real data | conformal sets on MNIST with measured coverage |
| N021 | Which variable are we differentiating? | Starter | Exact | three derivatives of one written-out function |
| N022 | Push a perturbation forward, pull a sensitivity back | Bridge | Exact | a 3×2 Jacobian, pushed forward and pulled back |
| N023 | Why a smaller finite difference can be worse | Bridge | Exact | finite differences of a known function |
| N024 | A silent wrong-axis calculation | Starter | Real data | an MNIST batch: subtracting a per-pixel mean along the wrong axis |
| N025 | Units change the optimization geometry | Bridge | Real data | the same MNIST regression with features rescaled |
| N026 | Singular modes reveal what data can determine | Bridge | Real data | singular values of a real design matrix |
| N027 | Keep tiny probabilities in logarithmic form | Starter | Real data | log-probabilities of a real passage under a real model |
| N028 | One random seed is one experiment | Starter | Real data | paired seeds on any MNIST comparison |
| N029 | Where a model's memory goes | Starter | Exact | counting numbers held, by kind |
| N030 | A flat training curve does not prove completion | Bridge | Real data | three real MNIST runs: converged, badly scaled, overfitting |
| N031 | Split by experiment, not by nearby rows | Starter | Real data | overlapping windows of Tiny Shakespeare split at random against split by scene |
| N032 | Preprocessing can leak the test set | Starter | Real data | scaling and PCA fitted on all of MNIST against inside the training folds |
| N033 | Bootstrap the independent unit | Bridge | Real data | the character stream of real text is autocorrelated: resample characters against whole lines |
| N034 | Three ways the world changes | Bridge | Real data | MNIST subsets with changed inputs, changed class shares, changed labels |
| N035 | Choose a trigger under a fixed event budget | Bridge | Real data | a 7-detector on MNIST under a fixed alarm budget |
| N036 | Missing measurements can carry selection information | Bridge | Real data | MNIST with pixels dropped, saturated or thresholded: disclosed transformations |
| N037 | Calibration after a detector change | Bridge | Real data | a classifier calibrated on clean MNIST, used on a disclosed transformation of it |
| N038 | Let the model abstain | Bridge | Real data | abstention on MNIST: risk against coverage |
| N039 | The model learned the apparatus label | Starter | Real data | a watermark added to one MNIST class: a disclosed transformation |
| N040 | Choose the next measurement to learn more | Bridge | Real data | choosing which MNIST image to label next |
| N041 | Invariance and equivariance are different promises | Bridge | Needs simulation | needs particle configurations with a known energy |
| N042 | A particle collection has no preferred ordering | Bridge | Real data | a digit as an unordered set of its ink pixels |
| N043 | Message passing between interacting particles | Bridge | Needs simulation | needs interacting particles; a pixel-neighbour graph on MNIST is a weaker substitute |
| N044 | Scalar and vector features under rotation | Advanced | Needs simulation | needs vector-valued physical inputs |
| N045 | Mirror images can matter | Advanced | Exact | distances and a signed triple product of four points; the learned part needs simulation |
| N046 | Boundaries break a convolution's symmetry | Bridge | Real data | a real digit translated under valid, zero and periodic padding |
| N047 | Build a conservation law into the output | Advanced | Exact | antisymmetric edge fluxes summing to zero; the learned comparison needs simulation |
| N048 | Local coordinates should not change the answer | Advanced | Needs simulation | needs vectors in local frames |
| N049 | Lorentz structure in event representations | Advanced | Needs simulation | needs four-momenta |
| N050 | Do not impose a symmetry the detector violates | Bridge | Needs simulation | needs a rotationally symmetric generator; MNIST digits are not rotation-symmetric (6 and 9) |
| N051 | A good next step can make a bad long forecast | Bridge | Real data | already drawn on real text by `teacher-forcing`; the oscillator version needs simulation |
| N052 | Learn a Hamiltonian, then differentiate it | Advanced | Needs simulation |  |
| N053 | Learn a Lagrangian in configuration coordinates | Advanced | Needs simulation |  |
| N054 | The numerical integrator is part of the model | Advanced | Exact | leapfrog against Euler on a written-out Hamiltonian; no learning needed |
| N055 | Discover a sparse dynamical equation | Advanced | Needs simulation |  |
| N056 | Linear evolution in learned observables | Advanced | Needs simulation |  |
| N057 | Learn drift and diffusion separately | Advanced | Needs simulation |  |
| N058 | Chaos sets a forecast horizon | Advanced | Needs simulation |  |
| N059 | Reduced coordinates still need a closure model | Advanced | Needs simulation |  |
| N060 | Differentiate through a physical simulator | Advanced | Needs simulation |  |
| N061 | A differential equation can supply training targets | Bridge | Needs simulation |  |
| N062 | Competing losses can hide a bad solution | Advanced | Needs simulation |  |
| N063 | Satisfy a boundary condition by construction | Advanced | Exact | the boundary factor is algebra; the trained part needs simulation |
| N064 | Residual checks can miss what lies between samples | Advanced | Needs simulation |  |
| N065 | Learn a map between whole functions | Advanced | Needs simulation |  |
| N066 | Fourier modes as learned communication channels | Advanced | Exact | one Fourier layer as arithmetic; the trained operator needs simulation |
| N067 | Combine a cheap simulator with scarce accurate data | Advanced | Needs simulation |  |
| N068 | When does a surrogate actually save work? | Bridge | Exact | cost arithmetic |
| N069 | A new grid is not automatically the same problem | Advanced | Needs simulation |  |
| N070 | A small surrogate error can bias a physical fit | Advanced | Needs simulation |  |
| N071 | The forward problem can be easy and the inverse unstable | Bridge | Real data | a real digit blurred by a known response, then deconvolved |
| N072 | Unfold a detector response and test closure | Advanced | Real data | unfold class proportions through a real classifier's confusion matrix |
| N073 | Infer by simulating and accepting nearby outcomes | Bridge | Needs simulation | needs a simulator; an exactly enumerable one makes it an illustration |
| N074 | Train a network to return a posterior distribution | Advanced | Needs simulation |  |
| N075 | Learn the likelihood, then perform inference | Advanced | Needs simulation |  |
| N076 | Learn a likelihood ratio by classification | Advanced | Needs simulation |  |
| N077 | What to do with nuisance parameters | Advanced | Exact | a two-parameter likelihood, profiled and marginalised |
| N078 | A second experiment can break a degeneracy | Bridge | Exact | two linear constraints with different orientations |
| N079 | Does the fitted model reproduce features of the observed data? | Bridge | Real data | posterior predictive checks of a model fitted to MNIST statistics |
| N080 | Test the inference algorithm on known simulated truths | Advanced | Needs simulation |  |
| N081 | From an energy function to a probability distribution | Bridge | Exact | enumerate a tiny spin system |
| N082 | Why an energy model needs negative samples | Advanced | Exact | an enumerable energy model; fitting it to binarised MNIST patches is the real-data version |
| N083 | A score field points toward higher density | Advanced | Exact | score of a written-out mixture |
| N084 | The best denoised average may look like no sample | Bridge | Real data | denoise MNIST by squared error: the conditional mean is a blur |
| N085 | Detailed balance and autocorrelation in a tiny sampler | Bridge | Exact | Metropolis on an enumerated landscape |
| N086 | Correct a learned proposal with an accept/reject step | Advanced | Needs simulation | needs a target density; exact on an enumerable one |
| N087 | Spend simulations where rare events happen | Advanced | Exact | importance sampling of a known tail |
| N088 | Reweighting works only where ensembles overlap | Advanced | Exact | reweighting an enumerable system |
| N089 | Coarse-graining is not automatically renormalization | Advanced | Needs simulation |  |
| N090 | Find a phase boundary without learning a simulator artifact | Advanced | Needs simulation |  |
| N091 | Reconstruct tracks from unordered detector hits | Advanced | Needs simulation |  |
| N092 | A plausible shower must also have the right observables | Advanced | Needs simulation |  |
| N093 | A learned detector versus a matched filter | Advanced | Needs simulation |  |
| N094 | A survey observes a selected population | Advanced | Needs simulation |  |
| N095 | Conservative forces from a learned molecular energy | Advanced | Needs simulation |  |
| N096 | Control an oscillator with explicit physical constraints | Advanced | Needs simulation |  |
| N097 | Search for a formula and test whether it is a law | Advanced | Needs simulation |  |
| N098 | Trade accuracy, cost and constraints on a Pareto front | Bridge | Exact | a Pareto front over written-out objectives |
| N099 | Reconstruct a quantum state that remains physical | Advanced | Needs simulation |  |
| N100 | From raw observations to an auditable scientific result | Advanced | Needs simulation |  |

## How to decide whether a redesign is finished

The first implementation batch should repair shared encodings and produce a small coherent path (N001, N002, N003, N005, N006, N008, N009, N021, N028), alongside revised gradient descent, the MLP sequence and overfitting. This is more useful to the stated audience than beginning with another collection of advanced architecture variants. Then add the uncertainty and evaluation spine before branching into the viewer's physics interests.

For each revised or new lesson:

- [ ] A viewer can identify the input, output, adjustable quantities and objective without reading code.
- [ ] The opening question has one clear answer demonstrated by the actual computed example; null or failed outcomes are explicit.
- [ ] Axes, units, colors, signs, normalization and aggregation are visible and consistent. No limitation depends on a truncated note.
- [ ] A comparison states what was held fixed and what changed. Synthetic, analytic, measured and illustrative quantities are distinguished.
- [ ] One worked example makes the operation visible before aggregate curves. A physics analogy states where it stops applying.
- [ ] An appropriate baseline, independent evaluation and uncertainty/variation are shown when the lesson makes an empirical claim.
- [ ] The transcript contains equations and exact definitions, but the main path does not require programming knowledge.
- [ ] Test with a few target-audience physicists: ask them to predict a perturbed example, explain the graph and name one limitation. Record misconceptions and revise before treating the lesson as successful. This learner check has not yet been performed.
- [ ] Keep local review outputs available and obtain the user's review before any later publishing action. This document does not authorize publishing.

Accounting: 128 released animations audited (17 P0 open); 100 rebuilt animations, with one of the first audit's 25 top-priority concerns fully open (`vae-posterior-collapse`), one partly (`teacher-forcing`) and the rest fixed, redesigned or reduced to a stated limit; 100 proposed lessons, none built, of which 39 need the data rule extended.
