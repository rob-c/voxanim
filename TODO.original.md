# TODO: the next 100 animations

A plan, not code. Each entry names a preset, says what would be on screen,
where the real numbers would come from, and why it is worth animating. None of
these duplicates one of the 128 animations in 1.5.0; where one builds on an
existing animation, it says which.

## Ground rules (as for every animation so far)

- **Every number on screen is computed**: a model trained here in plain numpy
  (seeded, cached by `cache.cached_npz`, gradients checked against finite
  differences), the real GPT-2 small, MNIST, or exact maths. Toy data is said
  to be toy data on screen.
- **Subtitles follow the measurements, not the plan.** The expected results
  below are hypotheses. If a study disagrees, the animation says what was
  measured, as the embeddings family did with ALiBi's in-range cost and
  T5's late rise.
- **Claims are pinned by tests**, as relative claims that survive CI
  retraining on another machine.
- **About 45–65 s each**: 4–5 phases, one live subtitle per phase, looping
  cleanly, passing `voxanim qa` and `voxanim gpucheck`.
- A preset that needs GPT-2 is named `gpt2-…`; one that needs neither GPT-2
  nor MNIST goes in `tests/conftest.py`'s `NO_DATA`.

Legend: ★ fun to watch, ◆ foundational (teach first), ⚑ a new study is needed
before it can be built.

## Shared studies to build first

Most entries reuse one of these, so they come first:

| study | what it is | used by |
|---|---|---|
| **S1 `tinylm`** ⚑ | `posenc.py`'s hand-written transformer generalised: N layers, H heads, causal / bidirectional / prefix / sliding-window / sparse masks, MHA / MQA / GQA, pre-LN / post-LN, LayerNorm / RMSNorm, GELU / SwiGLU, tied or untied embeddings, a KV cache. Every option gradient-checked. Trained on a few synthetic languages (Markov, copy, induction, modular arithmetic, bracket matching). | A, B, C, D, F |
| **S2 `toy2d`** ⚑ | 2-D datasets (moons, spirals, rings, a mixture of eight Gaussians) and small numpy models on them: MLP, SVM, GAN, flow, flow matching, boosting. | G, H, I |
| **S3 `gpt2_hooks`** ⚑ | GPT-2 small's forward pass with taps: cache and replace any activation (patching), ablate a head, read the KV cache, run any layer alone. Extends `gpt2.py`'s `run`. | A, D, F |
| **S4 `toycorpus`** | A generated text corpus with known structure (extending `w2v.py`'s templates) for language models, decoding and retrieval, with exact entropies where possible. | C, D, K |
| existing | MNIST models (`mnist.py`, `deepnet.py`), `gridworld.py`, `gp.py`, `landscape.py`, `w2v.py`, `posenc.py`, GPT-2. | everywhere |

---

## A. Attention: how it is wired (12)

1. **`causal-vs-bidirectional`** ◆ — Forward (causal, masked) attention against full (bidirectional) attention: the same 8 tokens, the two masks as matrices, the information flow drawn as arcs (left-only against both ways). Then one model trained each way on the same task. Real: S1 trained on fill-in-the-blank (bidirectional wins) and on next-token prediction (only causal can be trained this way without cheating, since a bidirectional model sees the answer). Why: the single biggest design choice between BERT-style and GPT-style models.
2. **`prefix-lm-attention`** — The prefix-LM mask (bidirectional over the prompt, causal over the reply) between the two above. Real: S1 on a copy-and-transform task, with all three masks and their losses. Why: the T5 / UL2 middle ground, and why it suits conditional generation.
3. **`cross-attention`** — The encoder–decoder link: decoder queries against encoder keys and values, against self-attention. Builds on `seq2seq-attention` (RNN attention), now with transformer blocks. Real: S1 encoder–decoder on reversing digits. Why: how translation models and multimodal models read another sequence.
4. **`multi-head-vs-single`** — One head of width 64 against 8 heads of width 8 at the same parameter count: what each head attends to in a trained model, and the loss. Real: S1 on a language that needs two relations at once (previous token and a matching bracket). Why: shows heads specialising, not just running in parallel.
5. **`grouped-query-attention`** ★ — MHA, MQA and GQA: many query heads sharing fewer K/V heads, with the KV-cache memory drawn to scale (bytes per token at 8, 2 and 1 K/V heads) and the loss cost of sharing. Real: S1 trained three ways; Llama-style numbers from the formulas. Why: the reason modern models fit long contexts in memory.
6. **`kv-cache`** ◆ — Decoding token by token: without a cache every step recomputes all keys and values (work grows quadratically); with one, each step adds one row. Multiply-add counts per step, and the cache growing in memory. Real: S3 on GPT-2 (cached and uncached logits equal to float precision). Why: how inference really runs.
7. **`sliding-window-attention`** — A band mask of width w, and how stacked layers widen the receptive field to L·w (drawn as a growing cone). Real: S1 with w = 4 and 1–3 layers on a long-range copy task: one layer fails beyond w, three succeed. Why: Mistral / Longformer locality, and depth as reach.
8. **`sparse-attention-patterns`** — Strided, block-local, global-token and random (BigBird) patterns as masks, each costing O(n√n) or O(n) rather than O(n²), with the attention cost at n = 1k–64k plotted. Real: exact counts; S1 accuracy for each on a retrieval task. Why: the zoo of efficient-attention ideas, side by side.
9. **`linear-attention`** — Softmax attention against the kernel trick (φ(q)·(Σ φ(k) vᵀ)): the same output computed as a running sum, as an RNN would. The approximation error and the loss. Real: S1 with an elu+1 feature map. Why: the bridge from transformers to RWKV / Mamba-style recurrences (`ssm-recurrence` is the other end).
10. **`flash-attention-tiling`** ★ — The online softmax: attention computed tile by tile with a running max and sum, never building the n × n matrix. The two maxima and rescales drawn per tile, and the memory traffic counted. Real: exact numpy against the naive result (equal to 1e-15). Why: why FlashAttention is exact and fast; a beautiful bit of arithmetic.
11. **`attention-sinks`** ★ — GPT-2's heads dumping attention on the first token. The per-head mass on position 0, what happens to perplexity when it is evicted from a sliding cache, and the fix of keeping 4 sink tokens (StreamingLLM). Real: S3 on GPT-2. Why: a surprising, real quirk with a practical consequence.
12. **`attention-temperature`** — Why scores are divided by √d: the softmax's entropy against the scale for d = 8–512, saturation without it, and the gradient through a saturated softmax. Real: random q, k at each d, plus S1 trained with and without the scale. Why: one innocent-looking constant with a reason.

## B. The transformer block: design choices (8)

13. **`pre-ln-vs-post-ln`** ◆ — Where LayerNorm sits: post-LN's gradients shrinking with depth and needing warm-up, pre-LN training stably. Per-layer gradient norms at step 0 and loss curves at 4–24 layers. Real: S1 deep and narrow. Why: why every modern model is pre-LN.
14. **`rmsnorm-vs-layernorm`** — Dropping the mean: the two formulas on one vector, the cost difference, and the trained loss (expected to be about the same). Real: S1 both ways. Why: a small change everyone made, and the evidence for it.
15. **`swiglu-mlp`** — The gated MLP: GELU(xW)·V against GELU(xW), the gate drawn as a per-unit multiplier, at equal parameters (hidden width ⅔ × 4d). Real: S1 both ways on the Markov language. Why: the MLP in Llama, PaLM and almost everything since.
16. **`residual-stream`** ★ — The stream as a shared bus: each layer reads from it and writes back into it. GPT-2's per-layer write norms, and the logit lens on what each write changes. Builds on `gpt2-transformer-block`, as a voxel bus with the layers as taps. Real: S3. Why: the mental model interpretability work uses.
17. **`weight-tying`** — The input embedding reused as the output projection: parameters saved (38.6M in GPT-2), its gradient arriving from both ends, and the loss trained tied and untied. Real: S1 and GPT-2's real tie. Why: follows on from `embedding-lookup`.
18. **`parallel-block`** — Attention and MLP in sequence against in parallel (GPT-J, PaLM): the dataflow, the saved latency, the loss. Real: S1 both ways. Why: a throughput trick with a measurable cost.
19. **`encoder-decoder-family`** ◆ — Encoder-only, decoder-only and encoder–decoder drawn as blocks and masks, with what each is trained on (masked LM, next token, span corruption), all three trained on one toy task. Real: S1. Why: the map that BERT, GPT and T5 sit on.
20. **`transformer-depth-vs-width`** — The `width-vs-depth` idea (which used MLPs) for transformers: equal parameters as 2 × 256, 4 × 180, 8 × 128, trained on a task needing composition (two-hop lookup). Real: S1. Why: depth buys composition.

## C. Training language models (8)

21. **`teacher-forcing`** ◆ — Shifted targets: one sequence becomes n predictions in parallel, with the causal mask making it honest. Also exposure bias, as free-running generation drifts where teacher-forced training never had to. Real: S1 on S4. Why: how a whole sequence trains in one pass.
22. **`perplexity-explained`** — Loss in nats, bits and perplexity ("choosing among k equally likely tokens"), with GPT-2's real per-token losses on a sentence and its perplexity against a unigram and a bigram model. Real: S3 plus S4. Why: the number every LM paper reports.
23. **`packing-and-padding`** — Variable-length sequences in a batch: padding wastes compute (counted), packing with attention resets doesn't, and what goes wrong when the resets are forgotten (cross-document leakage, measured). Real: S1. Why: an engineering detail with a real effect.
24. **`warmup-and-clipping`** — Adam on a transformer without warm-up (an early loss spike or divergence), with warm-up, and with gradient clipping (the clip ratio over training). Real: S1 at a learning rate that diverges. Why: the two safety belts of LM training.
25. **`grokking`** ★ — Modular addition (a + b mod 97): training accuracy hits 100% early, test accuracy stays at chance for thousands of steps, then jumps. Weight norm and the Fourier features appearing in the embedding. Real: S1, one layer, weight decay; seeded and cached (a long run). Why: one of the most striking plots in deep learning.
26. **`scaling-laws`** — A sweep of S1 sizes and dataset sizes: loss against compute on log-log axes, the power-law fit, and the compute-optimal frontier (a mini Chinchilla). Real: around 30 small runs. Why: the curve that drives the industry, made with real points.
27. **`curriculum-and-data-mixing`** — Training on two languages at a 90/10, 50/50 or curriculum mix, with each language's loss over time. Real: S1 on S4. Why: data mixtures are a big lever in modern pretraining.
28. **`tokenizer-effects`** — The same text as characters, BPE at 3 vocabulary sizes and bytes: sequence lengths, embedding table sizes and trained loss per character (the fair comparison). Builds on `gpt2-tokenizer`. Real: BPE trained here on S4. Why: why vocabulary size is a trade-off.

## D. Decoding and inference (6)

29. **`beam-search`** ◆ — The beam as a tree growing over time: kept and pruned hypotheses with their log-probabilities, against greedy, at beam widths 1–8. Real: S3 on GPT-2 with a short prompt. Why: the classic decoder, and its blandness.
30. **`speculative-decoding`** ★ — A small draft model proposes k tokens and GPT-2 verifies them in one pass: accepted, rejected and resampled tokens, the acceptance rate, and the speed-up. Shows that the output distribution is exactly the big model's. Real: S1 draft (or distilled GPT-2 layers) with S3. Why: a clever, provably exact speed-up.
31. **`repetition-and-penalties`** — Greedy decoding looping, then repetition and frequency penalties and no-repeat-n-gram reshaping the logits, with the distributions before and after. Real: S3. Why: a visible failure and its fixes (builds on `gpt2-sampling`).
32. **`constrained-decoding`** — A grammar (say, digits then a comma) as a mask over the vocabulary at each step, with the probability mass removed and renormalised. Real: S3 with a tiny grammar. Why: how JSON mode and tool calls are made reliable.
33. **`contrastive-decoding`** — The large model's logits minus the small one's: what the difference keeps and what it suppresses. Real: GPT-2 with a truncated (fewer-layer) GPT-2 as the amateur. Why: an elegant decoding idea, shown on real distributions.
34. **`batching-and-throughput`** — Serving many requests: tokens per second against batch size, the memory-bound to compute-bound crossover, and continuous batching's slots filling and emptying. Real: measured numpy timings plus exact arithmetic intensity. Why: why inference servers batch.

## E. Fine-tuning and alignment (8)

35. **`instruction-tuning`** — A base model continuing text against the same model after supervised fine-tuning on (instruction, answer) pairs, with loss masked to the answers (and what goes wrong without the mask). Real: S1 on S4-style tasks. Why: the first step from a language model to an assistant.
36. **`reward-model`** — Pairwise preferences trained with the Bradley–Terry loss: the reward gap against the probability of preferring, and the trained model's ranking. Real: a small MLP on synthetic preferences with a known true reward. Why: how human preferences become a number.
37. **`rlhf-ppo`** — A policy (S1 or a bandit over completions) optimised against the reward model with a KL penalty to the reference: reward rising, KL growing, and the trade-off at several β. Real: a small PPO in numpy. Why: the RLHF loop, end to end.
38. **`dpo`** ★ — Direct preference optimisation: the same preferences, no reward model, the implicit reward log π/π_ref, and DPO against PPO on the same data. Real: S1. Why: the method that replaced much of RLHF, with a one-line loss.
39. **`reward-hacking`** ★ — Optimise too hard against an imperfect reward model: proxy reward keeps rising while true reward peaks and falls (Goodhart). Real: #36 / #37 with a known true reward. Why: a cautionary, satisfying curve.
40. **`prompt-and-prefix-tuning`** — Learned soft prompts and prefixes against LoRA against full fine-tuning: trainable parameters (drawn to scale) and accuracy. Builds on `gpt2-lora`. Real: S1 or GPT-2 with S3. Why: the parameter-efficient fine-tuning menu.
41. **`model-merging`** ★ — Task vectors (fine-tuned minus base), added, negated and averaged: two skills from one merge, and interference measured. Real: two S1 fine-tunes. Why: weight arithmetic that works surprisingly often.
42. **`knowledge-editing`** — A rank-one edit to one GPT-2 MLP layer changing one fact (ROME-style): the layer found by causal tracing (#46), the edit, and what else it changes. Real: S3. Why: shows where facts live, and the side effects of moving them.

## F. Interpretability (8)

43. **`superposition`** ★◆ — Anthropic's toy model: 5 sparse features squeezed into 2 dimensions. As sparsity rises the features arrange into pentagons and antipodal pairs, with interference drawn. Real: a tiny numpy autoencoder, trained at a sweep of sparsities. Why: the clearest picture of why neurons are polysemantic.
44. **`sparse-autoencoder-features`** — A sparse autoencoder on GPT-2's residual stream at one layer: features that fire on specific concepts (their top-activating tokens), the dead-feature count, and reconstruction against sparsity. Real: S3 activations over a text sample; the SAE trained here. Why: the main tool of current interpretability.
45. **`linear-probes`** — Probing GPT-2's layers for a property (part of speech, or "inside quotes"): accuracy by layer, and a control task showing what probes can fake. Real: S3 and logistic regression. Why: what is decodable where.
46. **`activation-patching`** ◆ — Causal tracing on "The Eiffel Tower is in" against a corrupted prompt: patch each (layer, token) clean activation in and draw the recovered probability as a heatmap. Real: S3. Why: finds where a fact flows, with a real heatmap.
47. **`induction-heads`** ★ — A 2-layer attention-only model learning [A][B] … [A] → [B]: the previous-token head in layer 1 composing with the induction head in layer 2, and the phase change in the loss. Real: S1 on random sequences with repeats; also find one in GPT-2 via S3. Why: the first well-understood circuit.
48. **`head-ablation`** — Zero or mean-ablate each of GPT-2's 144 heads, one at a time, on a task such as indirect object identification: a 12 × 12 grid of the loss increase. Real: S3. Why: the model's heads ranked by what breaks without them.
49. **`saliency-and-integrated-gradients`** — MNIST pixel attributions: the raw gradient (noisy), gradient × input, and integrated gradients along a path from a black image, checked against the completeness axiom. Real: the existing MLP. Why: attribution methods compared honestly.
50. **`feature-visualisation`** ★ — Optimise an input to excite one CNN channel: images emerging from noise, with and without regularisation (jitter, blur) and the difference it makes. Real: the existing MNIST CNN. Why: what a filter "wants to see".

## G. Vision (8)

51. **`receptive-field`** ◆ — How far each layer of a CNN sees: the receptive field growing with depth, stride and dilation, the effective (Gaussian) field against the theoretical square, and gradient maps from one output pixel. Real: the MNIST CNN and exact arithmetic. Why: builds on the conv lessons.
52. **`vit-attention`** — A small ViT on MNIST, beyond `patch-embeddings` (which stops at the embedding): the CLS token's attention to patches per head and layer, and attention distance by depth. Real: S1 with patch inputs. Why: what a vision transformer actually attends to.
53. **`clip-contrastive`** ★◆ — Two encoders (digit images and digit-word tokens) trained with the InfoNCE loss: the N × N similarity matrix sharpening to its diagonal, temperature's effect, then zero-shot classification by nearest text. Real: MNIST with a synthetic "caption" vocabulary. Why: how CLIP links text and images.
54. **`adversarial-examples`** ★ — FGSM and PGD on the MNIST MLP: an imperceptible perturbation (drawn ×20) flipping 7 → 3, accuracy against ε, and adversarial training restoring robustness at a clean-accuracy cost. Real: existing models. Why: a vivid failure mode.
55. **`object-detection-basics`** — Anchors, IoU, regression offsets and non-maximum suppression on a synthetic scene of digits: overlapping boxes merging into one. Real: a small detector on composed MNIST images. Why: the pieces under YOLO and its relatives.
56. **`semantic-segmentation`** — Per-pixel classes on composed MNIST scenes with a U-Net (`unet-dense` denoises; this one labels): skip connections carrying detail, and IoU per class. Real: trained here. Why: dense prediction.
57. **`batchnorm-in-cnns`** — Per-channel statistics, train-mode against eval-mode behaviour, and the classic bug (eval-mode statistics with a tiny batch). Builds on `normalisation`. Real: the MNIST CNN. Why: a common source of real bugs.
58. **`style-transfer`** ★ — Gram matrices of CNN features as "style": an image optimised to match one digit's content and another's texture. Real: the existing CNN (small, but honest). Why: fun, and it shows what feature correlations capture.

## H. Generative models (8)

59. **`gan-training`** ★◆ — A generator and discriminator on a mixture of eight Gaussians: the samples moving, the discriminator's decision surface, mode collapse in the plain GAN, and a fix (minibatch discrimination or a WGAN-style loss). Real: S2. Why: adversarial training made visible.
60. **`normalizing-flows`** — An invertible 2-D flow (affine coupling) bending a Gaussian into two moons: the grid deforming, the log-determinant term, and exact likelihoods. Real: S2. Why: generative modelling with exact densities.
61. **`flow-matching`** ★ — Straight-line paths from noise to data, the learned velocity field, few-step sampling, and rectified flow straightening the paths further. Compared with `diffusion`'s curved paths. Real: S2. Why: today's most popular image-generation objective, in 2-D.
62. **`classifier-free-guidance`** ★ — A conditional diffusion model on MNIST digits: guidance scale 0–8 trading diversity for fidelity, samples in a row per scale. Builds on `diffusion`. Real: trained here. Why: the knob behind every image generator.
63. **`ddpm-vs-ddim`** — The same trained denoiser sampled stochastically (1,000 steps) and deterministically (10–50 steps): the trajectories, and quality against steps. Real: the existing diffusion study, extended. Why: fast sampling explained.
64. **`vae-posterior-collapse`** — A VAE with a strong decoder ignoring its latent: KL per dimension falling to zero, and KL annealing or free bits reviving the latent. Builds on `vae-dense-2d`. Real: trained here. Why: a classic VAE failure.
65. **`vq-vae-codebook`** — Vector quantisation: encoder outputs snapping to their nearest codebook vector, the straight-through gradient, codebook usage and dead codes. Real: MNIST VQ-VAE. Why: the discrete tokens behind many image and audio models.
66. **`autoregressive-pixels`** — PixelCNN-style masked convolutions generating an MNIST digit pixel by pixel, with the mask's blind spot drawn. Real: trained here. Why: images as next-token prediction.

## I. Classical ML and evaluation (8)

67. **`bias-variance`** ◆ — Polynomial fits on resampled data: the spread of fits (variance) and the mean fit's error (bias) against degree, decomposing test error exactly. Real: exact sampling. Why: the classic trade-off, before `double-descent` complicates it.
68. **`cross-validation`** — K-fold splits sliding over the data, the score per fold and its spread, and nested CV for choosing hyperparameters honestly (leakage measured). Real: S2. Why: honest model selection.
69. **`svm-and-kernels`** ★ — The maximum-margin line and its support vectors, the soft margin's C, then the kernel trick: rings separated by lifting to a third dimension (drawn in voxels). Real: an SVM fitted here on S2. Why: geometry that 3-D voxels show well.
70. **`gradient-boosting`** ★ — Trees fitted one after another to the residuals: the fit sharpening, the learning rate's shrinkage, and boosting against bagging (a random forest) as the number of trees grows. Builds on `decision-tree`. Real: trained here on S2. Why: what still wins on tabular data.
71. **`gaussian-mixture-em`** — Expectation–maximisation: soft responsibilities colouring points, ellipses updating, the log-likelihood rising monotonically, and a bad initialisation's local optimum. Builds on `pca-and-kmeans`. Real: S2. Why: EM made concrete.
72. **`curse-of-dimensionality`** ★ — Distances concentrating as dimensions grow (the ratio of nearest to farthest neighbour tending to 1), the volume of the ball against the cube, and k-NN accuracy falling. Real: exact sampling. Why: an intuition-breaker.
73. **`tsne-and-umap`** — t-SNE on MNIST at perplexities 5, 30 and 100: the same data, different maps, and what distances in the map do and don't mean. Real: t-SNE implemented here. Why: a warning about pretty plots.
74. **`roc-and-calibration`** — Sweeping a classifier's threshold: the confusion matrix changing, the ROC and precision–recall curves being traced, AUC, and base rates mattering. Builds on `gpt2-calibration`. Real: the MNIST MLP on a one-against-rest task. Why: evaluation beyond accuracy.

## J. Optimisation, deeper (6)

75. **`newton-vs-gradient-descent`** — On an ill-conditioned quadratic and on Rosenbrock: gradient descent zig-zagging, Newton jumping straight in using the Hessian, and the condition number's role. Real: `landscape.py` with exact Hessians. Why: what second-order information buys.
76. **`edge-of-stability`** ★ — Full-batch gradient descent: the sharpness (top Hessian eigenvalue) rising until it hovers at 2/η, with the loss oscillating yet still falling. Real: an MLP on a small MNIST subset with power-iteration sharpness. Why: a recent, surprising finding, made visible.
77. **`batch-size-and-noise`** — Gradient noise against batch size: the linear-scaling rule for the learning rate, the critical batch size, and diminishing returns. Real: MNIST MLP sweeps. Why: the first question in any large training run.
78. **`weight-averaging`** — An exponential moving average of the weights and stochastic weight averaging: the averaged model landing in a flatter, better spot. Builds on `sharp-vs-flat-minima`. Real: MNIST runs. Why: a nearly free improvement.
79. **`second-order-preconditioners`** — Adam's diagonal against a Shampoo / K-FAC-style block preconditioner on a quadratic with correlated directions: the preconditioned steps, and why the off-diagonal terms matter. Real: exact maths plus a small MLP. Why: where optimisers are heading.
80. **`loss-spikes`** — A training run with a sudden spike: the gradient norm and Adam's second-moment state before it, and how the fixes (a lower β₂, an ε change, clipping) change the trace. Real: S1 tuned to spike reproducibly (seeded). Why: what practitioners fight daily.

## K. Retrieval and contrastive learning (5)

81. **`retrieval-augmented-generation`** ◆ — A question embedded, nearest passages retrieved by cosine, then put in the prompt and the answer's probability rising with them. Real: sentence vectors from `w2v.py` or GPT-2 mean-pooled, S3 for the answer. Why: how assistants use documents.
82. **`vector-search-hnsw`** ★ — A hierarchical navigable small-world graph: greedy descent through the layers to the nearest neighbour, with recall against speed as parameters change. Real: built here on word vectors. Why: the index under every vector database, beautiful as a graph.
83. **`infonce-temperature`** — The contrastive loss's temperature: too high and nothing separates, too low and training is unstable. Embeddings on the unit sphere at three temperatures, alignment and uniformity measured. Real: S2 or MNIST. Why: the hidden knob in CLIP and SimCLR.
84. **`hard-negatives`** — Random against hard negatives in contrastive training: hard negatives finding finer distinctions, and too-hard ones (false negatives) hurting. Real: the w2v corpus or MNIST pairs. Why: the trick in retrieval training.
85. **`dimensionality-of-embeddings`** — Embedding width 2–256 for the same model: quality rising then saturating, Matryoshka-style nested embeddings that stay useful when truncated. Real: `w2v.py` at several widths. Why: how big an embedding needs to be.

## L. Systems and scale (5)

86. **`data-parallel-allreduce`** ★ — Data parallelism across 4 workers: identical models, different batches, and the gradients averaged by a ring all-reduce (the chunks passing round the ring, drawn), with bytes moved counted. Real: exact simulation; averaged gradients equal to the big-batch gradient. Why: how every large model trains.
87. **`tensor-and-pipeline-parallel`** — A layer's matrix split across devices (column or row), then layers split into pipeline stages with micro-batches filling the "bubble". Real: exact simulation with timing diagrams. Why: the other two dimensions of parallelism.
88. **`zero-sharding`** — Optimizer state, gradients and parameters sharded across devices (ZeRO stages 1–3): memory per device drawn to scale for a 7B-parameter model. Real: exact arithmetic. Why: where the memory of training goes.
89. **`activation-checkpointing`** — Store some activations, recompute the rest: memory against compute as the checkpoint spacing changes, √n checkpointing as the sweet spot. Real: exact counts plus a numpy MLP with recomputation (gradients identical). Why: trading compute for memory.
90. **`roofline`** — Arithmetic intensity against throughput for matmul, attention, layer norm and decoding: which are memory-bound and which compute-bound on a given chip. Real: exact FLOP and byte counts, plus measured numpy timings for honesty. Why: the performance model behind most optimisations.

## M. Reinforcement learning (6)

91. **`multi-armed-bandits`** ★◆ — ε-greedy, UCB and Thompson sampling on a 10-armed bandit: arm counts, posterior beliefs (Thompson's Betas) and regret curves. Real: exact simulation, seeded. Why: exploration against exploitation in its simplest form.
92. **`policy-gradient`** — REINFORCE on the gridworld from `q-learning`: action probabilities shifting along sampled trajectories, high variance, and a baseline cutting it. Real: trained here. Why: the other family of RL, beside value-based methods.
93. **`actor-critic`** — The critic's value estimates and the actor's policy updating together, with advantages drawn per step. Real: gridworld. Why: the basis of PPO (#37).
94. **`dqn-replay-buffer`** — Q-learning with a function approximator: the instability without experience replay and a target network, and the stability with them. Real: a small MLP on a continuous gridworld. Why: what made deep RL work.
95. **`mcts-tic-tac-toe`** ★ — Monte Carlo tree search: selection by UCB, expansion, rollouts and backup, with the tree growing toward the winning move. Real: exact simulation. Why: the search behind AlphaGo, on a game everyone knows.
96. **`reward-shaping`** — A sparse reward against a shaped one, and a badly shaped one producing loops that farm the shaping bonus. Real: gridworld. Why: why reward design is hard (and pairs with #39).

## N. Memory, dynamics and curiosities (4)

97. **`hopfield-network`** ★ — Store a few MNIST digits as attractors and recover them from corrupted versions, watching the energy fall. Capacity limits produce spurious memories, and modern Hopfield networks turn out to be attention. Real: exact. Why: a link from 1982 to transformers.
98. **`neural-cellular-automata`** ★ — A grid of cells with a tiny shared update rule, trained to grow a digit from one seed and to regrow it when damaged. Real: trained here (small, seeded). Why: self-organisation; a delight to watch.
99. **`neural-ode`** — A ResNet as Euler steps of a differential equation: trajectories of points flowing in 2-D as depth becomes continuous, and the adjoint method's memory saving. Real: S2. Why: connects `skip-connections` to dynamical systems.
100. **`evolution-strategies`** ★ — Gradient-free optimisation: a population of perturbed weights, fitness-weighted averaging estimating the gradient, compared with true gradient descent on the same loss. Real: `landscape.py` and a small MLP. Why: learning without backpropagation.

---

## Suggested order

1. **Studies S1 and S2.** Almost half the list depends on them, and S1 is a
   straight generalisation of `posenc.py`'s tested transformer.
2. **Release 1.6, "attention and the block"** (A, B, 20 animations). This
   continues the embeddings family directly; `causal-vs-bidirectional` is
   the opener.
3. **Release 1.7, "language models in practice"** (C, D, E, 22). Needs S3 and
   S4.
4. **Release 1.8, "inside the model"** (F, plus K's retrieval pieces, 13).
5. **Release 1.9, "beyond language"** (G, H, I, J, 30).
6. **Release 2.0, "systems, RL and curiosities"** (L, M, N, 15).

The ★ entries make good trailers and website openers. The ◆ entries should
come first within each family, because they are the ones the others refer
back to.
