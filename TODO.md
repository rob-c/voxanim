# TODO: the next 100 animations

> **Rebuilt, 16 November 2026.** The first builds of these 100 drew every study as four
> generic chart panels. They were rejected ("some numbers, an animated line") and have been
> replaced: each animation is now its own drawn mechanism (`voxanim/mech.py` and
> `voxanim/mech_*.py`), built to the original plan's "what would be on screen", which is kept
> in [TODO.original.md](TODO.original.md). The entries below describe the first experiments. Several studies give null or negative results,
> which the animations state on screen. 56 of the experiments were then redesigned; the outcome
> of each is listed at the end of this file.

The entries below retain the original teaching ideas, with implementation notes
recording what the first review builds measured. The animations were published in 1.6.0.

**Dataset requirement:** every empirical learning claim in these 100 additions
must use actual NumPy training and evaluation on **MNIST or real text**. The
text source is the checksum-pinned Tiny Shakespeare corpus, split before
sampling training and held-out passages. No fabricated learning curves,
prescribed winners, or synthetic-language benchmarks qualify. Corrupting or
augmenting an observed MNIST image or real passage is allowed when the
transformation and unchanged source are disclosed.

Pretrained GPT-2 probes are evaluations of an existing checkpoint, not local
GPT-2 training. Analytic examples, exact memory/operation counts, measured local
runtime, and simulated execution schedules are identified separately; they do
not establish empirical learning or unmeasured hardware-performance claims.

All 100 revised videos are implemented and available in
[the local review gallery](build/review-realdata-100/index.html). Each video
has a phase transcript, plot arrays, a render manifest and, where applicable,
links to retained training artifacts. The [earlier gallery](build/review-next-100/index.html) remains
available for comparison and does **not** satisfy the revised dataset policy.
Review copies are 960 × 540, 24 fps, 53 seconds each. User review is still required
before any publication.

Shared implementations include `voxanim/realdata.py`, `voxanim/tinylm.py`,
`voxanim/gpt2_hooks.py`, and the `next_attention`, `next_language`, `next_vision`,
`next_vision_models`, `next_systems`, and `next_systems_real` modules. Generate the
revised set with `python3 tools/review_next.py --out build/review-realdata-100
--jobs 8`; rendering resumes verified unchanged outputs and never publishes.

Each entry names a preset and records its current measured experiment and
material limitations. Original hypotheses have been replaced where the
implemented data or task differs. None duplicates
one of the 128 animations in 1.5.0; where one builds on an existing animation,
it says which.

## Ground rules (as for every animation so far)

- **Empirical curves come from real runs.** Loss, accuracy, generalization,
  convergence and quality comparisons must be calculated from actual NumPy
  model updates and evaluations. No hand-authored decay curves, scripted
  winners, fabricated timing, or edited samples that imply an unobserved
  result. Rendering may reveal measured samples over time; that reveal is
  not itself a training run.
- **Name the source of each result.** Distinguish training on MNIST or real
  text, evaluation of an existing checkpoint,
  mathematical calculations, measured runtime and simulated execution.
  A pretrained GPT-2 forward pass is real model evaluation, not a local
  GPT-2 training run. A formula for memory or a pipeline schedule is not a
  hardware measurement. Mathematical examples may explain the mechanism,
  but cannot substitute for an empirical claim about learning or speed.
- **Make a run auditable.** Retain dataset identity/checksum, preprocessing,
  train/validation/test membership, seeds, model and optimizer settings,
  actual update count, source revision/hash, checkpoints, unmodified metric
  traces and evaluation outputs. Connect the plotted arrays to those
  artifacts. Record which quantities were measured before versus after an
  update. Use the linked run receipts and retained checkpoints to inspect this record;
  small seeded experiments do not establish universal rankings.
- **Subtitles follow the measurements, not the plan.** Report the observed result, including weak, negative or inconclusive
  results. Do not tune a plot to agree with an expected ranking.
- **Validate the computation, not a preferred winner.** Check gradients,
  replay saved evaluations and verify plotted values against run artifacts.
  Finite arrays and a successfully encoded video do not prove a training
  claim. Report variation or the limits of a single seeded run; do not
  promise that every empirical ranking survives retraining.
- **About 45–65 s each**: 4–5 phases, one live subtitle per phase, looping
  cleanly, passing `voxanim qa` and `voxanim gpucheck`.
- A preset that needs GPT-2 is named `gpt2-…`; one that needs neither GPT-2
  nor MNIST nor real text goes in `tests/conftest.py`'s `NO_DATA`.

The original starred/foundational labels have been superseded by the
pedagogical priorities in [TODO2.md](TODO2.md).

## Shared studies to build first

The original shared-study plan is now implemented with these data boundaries:

| study | what it is | used by |
|---|---|---|
| **S1 `tinylm`** ⚑ | `posenc.py`'s hand-written transformer generalised: N layers, H heads, causal / bidirectional / prefix / sliding-window / sparse masks, MHA / MQA / GQA, pre-LN / post-LN, LayerNorm / RMSNorm, GELU / SwiGLU, tied or untied embeddings, a KV cache. Every option gradient-checked. The review builds train on real Tiny Shakespeare passages; generated inputs remain only in numerical unit tests. | A, B, C, D, F |
| **S2 `toy2d`** ⚑ | Small NumPy model primitives: MLP, SVM, GAN, flow, flow matching and boosting. Review training inputs are MNIST images or training-only PCA coordinates, not the original generated moons/spirals/rings. | G, H, I |
| **S3 `gpt2_hooks`** ⚑ | GPT-2 small's forward pass with taps: cache and replace any activation (patching), ablate a head, read the KV cache, run any layer alone. Extends `gpt2.py`'s `run`. | A, D, F |
| **S4 `toycorpus`** | Tokenizer utilities retained from the original generated-corpus module; review learning uses checksum-pinned real text from `realdata.py`. No assumed entropy floor is used for natural language. | C, D, K |
| existing | MNIST models (`mnist.py`, `deepnet.py`), `gridworld.py`, `gp.py`, `landscape.py`, `w2v.py`, `posenc.py`, GPT-2. | everywhere |

---

## A. Attention: how it is wired (12)

1. **`causal-vs-bidirectional`** — Tiny Shakespeare real-text windows; 280 NumPy updates fit one masked original character using causal or bidirectional context. Disjoint-source test NLL is 4.867 / 4.582 nats; these small runs do not establish a universal ranking.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/causal-vs-bidirectional.mp4; data and run evidence: build/review-realdata-100/data/causal-vs-bidirectional.json. CPU layout, integrity and full video decode verified. -->
2. **`prefix-lm-attention`** — Real-text suffix restoration, with all six target slots masked; 240 updates compare causal, prefix and bidirectional masks. In one block, changing only prefix-query visibility leaves suffix predictions identical: causal/prefix test NLL 3.714 / 3.714; bidirectional 3.810.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/prefix-lm-attention.mp4; data and run evidence: build/review-realdata-100/data/prefix-lm-attention.json. CPU layout, integrity and full video decode verified. -->
3. **`cross-attention`** — NumPy cross-attention is trained for 300 updates to restore six-character Shakespeare spans with source positions 2 and 5 masked. Teacher-forced accuracy on 64 disjoint-source test spans is 43.75%; this is denoising, not translation or free generation.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/cross-attention.mp4; data and run evidence: build/review-realdata-100/data/cross-attention.json. CPU layout, integrity and full video decode verified. -->
4. **`multi-head-vs-single`** — One-head and eight-head width 32 two-block transformers train for 900 updates on 384 real-text next-character windows. Held-out losses and attention mass to the previous position / earlier matching characters are measured, not assigned; eight-head token accuracy is 19.79%, with no guaranteed head specialisation.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/multi-head-vs-single.mp4; data and run evidence: build/review-realdata-100/data/multi-head-vs-single.json. CPU layout, integrity and full video decode verified. -->
5. **`grouped-query-attention`** — MHA/GQA/MQA variants train for 240 updates on the same real-text windows; test NLL is 2.951 / 2.947 / 2.887 nats. The separate KV-memory panel is an analytic FP16 count for stated model dimensions, not a measured serving benchmark.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/grouped-query-attention.mp4; data and run evidence: build/review-realdata-100/data/grouped-query-attention.json. CPU layout, integrity and full video decode verified. -->
6. **`kv-cache`** — Pinned pretrained GPT-2 evaluates collected Shakespeare text using full and incremental cached forwards; per-token logit differences are measured. Projected-token work and KV-cache bytes are explicitly analytic counts; no local GPT-2 training is claimed.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-kv-cache.mp4; data and run evidence: build/review-realdata-100/data/gpt2-kv-cache.json. CPU layout, integrity and full video decode verified. -->
7. **`sliding-window-attention`** — One/two/three local-attention blocks train for 240 updates on real next-character text. Test NLL is 2.949 /3.078 /3.163 nats, so extra depth does not win this run. The receptive reach panel is exact mask composition, not an invented six-step retrieval task.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/sliding-window-attention.mp4; data and run evidence: build/review-realdata-100/data/sliding-window-attention.json. CPU layout, integrity and full video decode verified. -->
8. **`sparse-attention-patterns`** — Five causal sparsity patterns receive 200 real-text training updates each. Disjoint-source next-character test NLL is measured for every pattern; large-sequence allowed-pair counts are exact combinatorial calculations, not measured speed or generated learning curves.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/sparse-attention-patterns.mp4; data and run evidence: build/review-realdata-100/data/sparse-attention-patterns.json. CPU layout, integrity and full video decode verified. -->
9. **`linear-attention`** — Softmax and ELU+1 kernel attention receive 240 updates on the same real text; test NLL is 2.951 /2.875 nats. Seeded random-array calculations separately verify associativity and its difference from softmax; the implementation does not claim a measured linear-time speedup.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/linear-attention.mp4; data and run evidence: build/review-realdata-100/data/linear-attention.json. CPU layout, integrity and full video decode verified. -->
10. **`flash-attention-tiling`** — A seeded NumPy numerical-identity experiment compares dense softmax with 8-key online-softmax tiles on the same Q/K/V arrays. Output discrepancies and running maxima/sums are calculated; scratch-byte curves are analytic. No training, real-data accuracy or hardware-speed claim.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/flash-attention-tiling.mp4; data and run evidence: build/review-realdata-100/data/flash-attention-tiling.json. CPU layout, integrity and full video decode verified. -->
11. **`attention-sinks`** — Pinned pretrained GPT-2 measures next-token NLL on 64 collected Shakespeare tokens with full cache, last 32 slots, or 4 sink + 28 recent slots, preserving absolute positions. Suffix perplexities are 144.499 / 1011.233 / 141.465 respectively. One short stream is not a general quality guarantee or a new training run.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-attention-sinks.mp4; data and run evidence: build/review-realdata-100/data/gpt2-attention-sinks.json. CPU layout, integrity and full video decode verified. -->
12. **`attention-temperature`** — Scaled and unscaled scores receive 220 NumPy updates on real text; disjoint-source NLL is 2.979 /3.032 nats. Independent-Gaussian width/entropy/Jacobian experiments are explicitly marked random-array simulations, separate from the actual learned-model comparison.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/attention-temperature.mp4; data and run evidence: build/review-realdata-100/data/attention-temperature.json. CPU layout, integrity and full video decode verified. -->

## B. The transformer block: design choices (8)

13. **`pre-ln-vs-post-ln`** — Six NumPy transformers with 4/12/24 blocks and pre/post LayerNorm receive 100 updates on the same Shakespeare windows. Initial query-projection gradient norms, actual training traces and disjoint-source losses are recorded; no convergence or winning placement is assumed.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/pre-ln-vs-post-ln.mp4; data and run evidence: build/review-realdata-100/data/pre-ln-vs-post-ln.json. CPU layout, integrity and full video decode verified. -->
14. **`rmsnorm-vs-layernorm`** — LayerNorm and RMSNorm models receive 240 updates on the same collected-text windows; disjoint-source next-character NLL is 2.951 /2.945 nats. Actual parameter counts and checkpoints are retained; this small difference is not a general superiority claim.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/rmsnorm-vs-layernorm.mp4; data and run evidence: build/review-realdata-100/data/rmsnorm-vs-layernorm.json. CPU layout, integrity and full video decode verified. -->
15. **`swiglu-mlp`** — GELU and SwiGLU transformer feed-forward variants train for 240 updates on collected text using the stated intermediate widths. Disjoint-source NLL is 2.960 /3.024 nats; curves and parameter counts are measured without forcing the proposed winner.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/swiglu-mlp.mp4; data and run evidence: build/review-realdata-100/data/swiglu-mlp.json. CPU layout, integrity and full video decode verified. -->
16. **`residual-stream`** — Pinned pretrained GPT-2 reads collected Shakespeare text. Per-layer attention/MLP write norms, channel values and logit-lens readouts are computed from its actual activations; these are descriptive probes, not proof of a causal explanation or local pretraining.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-residual-stream.mp4; data and run evidence: build/review-realdata-100/data/gpt2-residual-stream.json. CPU layout, integrity and full video decode verified. -->
17. **`weight-tying`** — Untied and tied input/output embeddings receive 240 real-text training updates; disjoint-source NLL is 2.951 / 2.851 nats respectively. Actual tiny-model parameter counts accompany explicitly analytic GPT-2 shared-table parameter/byte savings.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/weight-tying.mp4; data and run evidence: build/review-realdata-100/data/weight-tying.json. CPU layout, integrity and full video decode verified. -->
18. **`parallel-block`** — Sequential and parallel residual-block variants receive 240 updates on identical real-text next-character data; disjoint-source NLL is 2.951 /2.818 nats. The result describes these seeded models and budgets, not a guaranteed architecture ordering.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/parallel-block.mp4; data and run evidence: build/review-realdata-100/data/parallel-block.json. CPU layout, integrity and full video decode verified. -->
19. **`encoder-decoder-family`** — Encoder-only, decoder-only and jointly trained encoder-decoder models restore masked six-character real-text spans for 420 updates. The joint checkpoint replays test NLL 3.721 and teacher-forced token accuracy 30.47% exactly. Conditioning and parameter counts differ, so this is a family demonstration, not a controlled ranking.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/encoder-decoder-family.mp4; data and run evidence: build/review-realdata-100/data/encoder-decoder-family.json. CPU layout, integrity and full video decode verified. -->
20. **`transformer-depth-vs-width`** — Two/four/eight-block width/depth variants receive 260 updates on real next-character text; test NLL is 4.519 / 4.345 / 3.324 nats respectively. Their parameter counts are 26,856 / 25,928 / 24,072; this is not an equal-parameter comparison. Targets are the actual next characters in disjoint-source text.
   <!-- implemented: voxanim/next_attention.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/transformer-depth-vs-width.mp4; data and run evidence: build/review-realdata-100/data/transformer-depth-vs-width.json. CPU layout, integrity and full video decode verified. -->

## C. Training language models (8)

21. **`teacher-forcing`** — TinyShakespeare character language model trained for 350 actual NumPy updates. Held-out excerpts compare recorded contexts with greedy generated contexts; disagreement with one recorded continuation is not proof that every alternative is invalid.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/teacher-forcing.mp4; data and run evidence: build/review-realdata-100/data/teacher-forcing.json. CPU layout, integrity and full video decode verified. -->
22. **`perplexity-explained`** — Count-fitted unigram/bigram baselines and pretrained GPT-2 score the exact same tokens in eight held-out TinyShakespeare excerpts. Loss is token-weighted; perplexity is its exponential. Uniform-choice behavior is a separately labeled analytic reference.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-perplexity-explained.mp4; data and run evidence: build/review-realdata-100/data/gpt2-perplexity-explained.json. CPU layout, integrity and full video decode verified. -->
23. **`packing-and-padding`** — A TinyShakespeare transformer is trained for 220 NumPy updates. Sixty-four held-out excerpts test individual evaluation, concatenation without resets, and mask-plus-position resets. Padding counts come from actual source-line lengths; they are a separate illustration from fixed-length packed test excerpts.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/packing-and-padding.mp4; data and run evidence: build/review-realdata-100/data/packing-and-padding.json. CPU layout, integrity and full video decode verified. -->
24. **`warmup-and-clipping`** — Three four-layer post-LN TinyShakespeare transformers train for 240 NumPy updates at learning rate 0.02, with fixed rate, warm-up, or warm-up plus clipping. Recorded losses, pre-clipping gradient norms and common held-out losses determine the outcome; divergence is not presumed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/warmup-and-clipping.mp4; data and run evidence: build/review-realdata-100/data/warmup-and-clipping.json. CPU layout, integrity and full video decode verified. -->
25. **`grokking`** — MNIST replaces modular-arithmetic training: a 49–64–10 MLP receives 3,000 real AdamW updates on 600 area-averaged 7x7 training images; 1,000 official-test images are evaluated every 50 updates. The lesson asks whether delayed generalization occurs and does not claim to reproduce grokking. This run finishes at 100% training and 86.5% test accuracy.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/grokking.mp4; data and run evidence: build/review-realdata-100/data/grokking.json. CPU layout, integrity and full video decode verified. -->
26. **`scaling-laws`** — Thirty actual TinyShakespeare training runs vary five transformer widths and six numbers of distinct training excerpts, each for 180 NumPy updates. Held-out loss is measured; 6 times parameters times processed tokens is only a compute estimate. A descriptive log-total-loss fit and observed frontier do not establish a universal scaling law.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/scaling-laws.mp4; data and run evidence: build/review-realdata-100/data/scaling-laws.json. CPU layout, integrity and full video decode verified. -->
27. **`curriculum-and-data-mixing`** — Actual TinyShakespeare source regions replace synthetic transition languages. Three real training runs use 90/10, 50/50 or changing mixtures; each region has a separate contiguous 80/20 train/held-out split. Both held-out losses and their equal-weight mean are measured.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/curriculum-and-data-mixing.mp4; data and run evidence: build/review-realdata-100/data/curriculum-and-data-mixing.json. CPU layout, integrity and full video decode verified. -->
28. **`tokenizer-effects`** — BPE merges and five tiny language models are trained on collected TinyShakespeare text. Vocabularies are fitted to training text with an explicit unknown token. Token counts, embedding sizes and held-out nats per original character compare character, byte and three BPE representations; fixed update counts do not equal fixed character exposure.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/tokenizer-effects.mp4; data and run evidence: build/review-realdata-100/data/tokenizer-effects.json. CPU layout, integrity and full video decode verified. -->

## D. Decoding and inference (6)

29. **`beam-search`** — Pretrained GPT-2 performs actual five-token beam searches at widths 1, 2, 4 and 8 on an authored prompt. Stored paths and cumulative log probabilities are measured inference results, not new training runs or proof of human-preferred text.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-beam-search.mp4; data and run evidence: build/review-realdata-100/data/gpt2-beam-search.json. CPU layout, integrity and full video decode verified. -->
30. **`speculative-decoding`** — Actual three-token drafts from a four-layer truncation of pretrained GPT-2 are verified by full GPT-2, with acceptance/rejection and residual sampling. The probability identity is calculated exactly; forward-call counts and wall times are measured. The draft is not newly trained or distilled and acceleration is not guaranteed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-speculative-decoding.mp4; data and run evidence: build/review-realdata-100/data/gpt2-speculative-decoding.json. CPU layout, integrity and full video decode verified. -->
31. **`repetition-and-penalties`** — Pretrained GPT-2 supplies actual logits for an authored repeated-prefix prompt. Repetition, frequency and repeated-trigram rules transform those probabilities. This is a one-step decoding comparison, not a model-training or long-sequence repetition benchmark.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-repetition-and-penalties.mp4; data and run evidence: build/review-realdata-100/data/gpt2-repetition-and-penalties.json. CPU layout, integrity and full video decode verified. -->
32. **`constrained-decoding`** — Pretrained GPT-2 emits tokens under an explicit DD,DD finite-state grammar. Allowed/removed probability mass and the emitted string are computed from real forward passes. Grammar masks are structural rules, not learned accuracy results; syntactic validity does not imply semantic truth.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-constrained-decoding.mp4; data and run evidence: build/review-realdata-100/data/gpt2-constrained-decoding.json. CPU layout, integrity and full video decode verified. -->
33. **`contrastive-decoding`** — Actual distributions from full GPT-2 and its four-layer truncation are compared on an authored prompt. Log-probability differences and a plausibility mask are calculated explicitly. No new amateur-model training or general output-quality advantage is claimed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-contrastive-decoding.mp4; data and run evidence: build/review-realdata-100/data/gpt2-contrastive-decoding.json. CPU layout, integrity and full video decode verified. -->
34. **`batching-and-throughput`** — Actual timings average 120 warmed NumPy 384x384 matrix multiplications per batch size. Arithmetic intensity is an explicit count formula. The four-slot serving schedule is separately labeled a simulation with fixed request durations, not measured server latency or learned behavior.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/batching-and-throughput.mp4; data and run evidence: build/review-realdata-100/data/batching-and-throughput.json. CPU layout, integrity and full video decode verified. -->

## E. Fine-tuning and alignment (8)

35. **`instruction-tuning`** — Actual TinyShakespeare prefixes and four-character continuations are wrapped in automatic Next: instructions. A real-text pretrained transformer receives 400 NumPy fine-tuning updates with answer-only or all-token loss and is evaluated on untouched source excerpts. This is automatic continuation supervision, not a human instruction-following dataset; no winning loss mask is assumed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/instruction-tuning.mp4; data and run evidence: build/review-realdata-100/data/instruction-tuning.json. CPU layout, integrity and full video decode verified. -->
36. **`reward-model`** — A NumPy linear reward scorer trains for 250 updates on automatic observed-next-word versus sampled-alternative pairs from TinyShakespeare. Forty-eight real corpus words form the candidate set after the word the. Separate held-out pairs measure ranking; these are noisy automatic preferences, not human reward labels. This run has approximately 51.15% held-out pair accuracy.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/reward-model.mp4; data and run evidence: build/review-realdata-100/data/reward-model.json. CPU layout, integrity and full video decode verified. -->
37. **`rlhf-ppo`** — Real categorical PPO gradient updates optimize the learned real-text preference scorer at three KL strengths. The reference comes from training-corpus frequencies. Held-out conditional log frequency is an explicit evaluation score, not human utility; this finite word-choice bandit is not a human-feedback language-model RLHF run.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/rlhf-ppo.mp4; data and run evidence: build/review-realdata-100/data/rlhf-ppo.json. CPU layout, integrity and full video decode verified. -->
38. **`dpo`** — A tiny transformer receives 180 genuine NumPy DPO updates on the same actual observed-word preference pairs. An output offset makes its starting distribution match the corpus-frequency reference. Held-out corpus score and KL are measured; the PPO comparator is categorical while DPO is neural, and all preferences are automatic rather than human.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/dpo.mp4; data and run evidence: build/review-realdata-100/data/dpo.json. CPU layout, integrity and full video decode verified. -->
39. **`reward-hacking`** — Each of 240 exponentiated-gradient updates is executed and recorded using the real-text-trained proxy. A separate held-out corpus log-frequency objective measures whether proxy optimization harms that criterion. No downturn is imposed; this run moves from -4.718 initially to a best -3.777 and ends at -3.980.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/reward-hacking.mp4; data and run evidence: build/review-realdata-100/data/reward-hacking.json. CPU layout, integrity and full video decode verified. -->
40. **`prompt-and-prefix-tuning`** — Soft prompts, genuine per-layer KV prefixes, output-projection LoRA and full tuning adapt a real TinyShakespeare model between two source regions for 220 actual NumPy updates. Loss is evaluated on the untouched target-region tail; trained parameter counts are exact, and parity with full tuning is not assumed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/prompt-and-prefix-tuning.mp4; data and run evidence: build/review-realdata-100/data/prompt-and-prefix-tuning.json. CPU layout, integrity and full video decode verified. -->
41. **`model-merging`** — Two independently optimized, 300-update TinyShakespeare region adaptations share one real-text-trained base model. Averaged, added and subtracted task vectors are evaluated on separate held-out regional passages. Task IDs mark source region; every predicted character comes from collected text, and successful skill composition is not assumed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/model-merging.mp4; data and run evidence: build/review-realdata-100/data/model-merging.json. CPU layout, integrity and full video decode verified. -->
42. **`knowledge-editing`** — Pretrained GPT-2 is tested on a fixed real TinyShakespeare passage. A covariance-weighted analytic rank-one intervention favors an explicitly counterfactual token, with covariance estimated from real training-text activations and a second held-out passage as control. This is measured continuation editing, not a factual knowledge-training benchmark or full ROME optimization.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-knowledge-editing.mp4; data and run evidence: build/review-realdata-100/data/gpt2-knowledge-editing.json. CPU layout, integrity and full video decode verified. -->

## F. Interpretability (8)

43. **`superposition`** — Five patch-intensity features are extracted from 2,048 actual 14x14 MNIST training images. Training-set quantile thresholds vary sparsity; four two-dimensional autoencoders train for 800 NumPy updates and evaluate 512 official-test images with unchanged thresholds. Learned directions, Gram matrices and losses are measured; mostly-zero inputs can reduce raw loss without improving all active features.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/superposition.mp4; data and run evidence: build/review-realdata-100/data/superposition.json. CPU layout, integrity and full video decode verified. -->
44. **`sparse-autoencoder-features`** — A 768-to-896-to-768 sparse autoencoder trains for 180 NumPy updates on actual GPT-2 layer-six activations from 24 collected TinyShakespeare passages. Eight separate held-out passages test reconstruction using training-only centering and scaling. Top-activating tokens are observations, not proof of monosemantic concepts.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-sparse-autoencoder-features.mp4; data and run evidence: build/review-realdata-100/data/gpt2-sparse-autoencoder-features.json. CPU layout, integrity and full video decode verified. -->
45. **`linear-probes`** — Linear probes train on GPT-2 activations from 24 real TinyShakespeare excerpts and test on eight disjoint excerpts. The target is punctuation in the preceding token; balanced accuracy, an input-embedding baseline and a current-token identity control replace synthetic quotation templates. Decodability does not establish causal use.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-linear-probes.mp4; data and run evidence: build/review-realdata-100/data/gpt2-linear-probes.json. CPU layout, integrity and full video decode verified. -->
46. **`activation-patching`** — A fixed nine-token held-out TinyShakespeare prefix supplies the actual next-token target. Two source tokens are replaced to corrupt context, then every layer/token residual patch is executed in GPT-2. Recorded probabilities and normalized recovery are local intervention measurements; the case is not selected to guarantee a large recovery or factual localization.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-activation-patching.mp4; data and run evidence: build/review-realdata-100/data/gpt2-activation-patching.json. CPU layout, integrity and full video decode verified. -->
47. **`induction-heads`** — Two attention-only transformer blocks receive 650 actual NumPy updates on ordinary collected TinyShakespeare characters. Held-out next-character loss and attention to successors of natural repeated characters are measured; a separate GPT-2 probe uses natural token repeats. Successor mass is normalized per eligible query. No artificially duplicated sequence or strong induction head is assumed.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-induction-heads.mp4; data and run evidence: build/review-realdata-100/data/gpt2-induction-heads.json. CPU layout, integrity and full video decode verified. -->
48. **`head-ablation`** — Pretrained GPT-2 undergoes all 144 single-head zero ablations on each of two fixed held-out TinyShakespeare excerpts. Target losses use the actual recorded next token, and signed loss changes are measured. These two passages do not establish a universal head-importance ranking.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-head-ablation.mp4; data and run evidence: build/review-realdata-100/data/gpt2-head-ablation.json. CPU layout, integrity and full video decode verified. -->
49. **`saliency-and-integrated-gradients`** — The repository MLP is genuinely trained in NumPy on standard MNIST. The first official-test image supplies actual target-logit gradients and 512-point integrated gradients from a black baseline. Completeness is measured; attribution is not additional model training or proof of semantic causality.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/saliency-and-integrated-gradients.mp4; data and run evidence: build/review-realdata-100/data/saliency-and-integrated-gradients.json. CPU layout, integrity and full video decode verified. -->
50. **`feature-visualisation`** — The repository CNN is genuinely trained in NumPy on standard MNIST. With model weights frozen, 120 actual input-pixel optimization steps maximize one channel response, with and without regularization. Recorded images and response traces include the unsuccessful over-smoothed result; the experiment does not claim every regularized image is an improved explanation.
   <!-- implemented: voxanim/next_language.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/feature-visualisation.mp4; data and run evidence: build/review-realdata-100/data/feature-visualisation.json. CPU layout, integrity and full video decode verified. -->

## G. Vision (8)

51. **`receptive-field`** — Fixed NumPy-trained MNIST CNN: measure input gradients and receptive-field support. This is checkpoint analysis, with no new model training.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/receptive-field.mp4; data and run evidence: build/review-realdata-100/data/receptive-field.json. CPU layout, integrity and full video decode verified. -->
52. **`vit-attention`** — Train a two-block NumPy vision transformer on real MNIST for 700 updates; show attention for the displayed held-out image and average spatial reach across 128 held-out images. Attention weights are not causal explanations.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/vit-attention.mp4; data and run evidence: build/review-realdata-100/data/vit-attention.json. CPU layout, integrity and full video decode verified. -->
53. **`clip-contrastive`** — Train image and digit-label embeddings on real MNIST pairs for 900 NumPy updates; measure retrieval against the ten labels seen during training. This is a closed digit vocabulary, not unseen-class or natural-language zero-shot transfer.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/clip-contrastive.mp4; data and run evidence: build/review-realdata-100/data/clip-contrastive.json. CPU layout, integrity and full video decode verified. -->
54. **`adversarial-examples`** — Measure FGSM and PGD attacks on 160 held-out MNIST images using a fixed trained MLP; perform a separate 220-update adversarial fine-tune on training images and recompute attacks. Results apply to the stated attack and perturbation budgets.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/adversarial-examples.mp4; data and run evidence: build/review-realdata-100/data/adversarial-examples.json. CPU layout, integrity and full video decode verified. -->
55. **`object-detection-basics`** — Train a NumPy box regressor for 1,100 updates on scenes composed from real MNIST digits at recorded positions; evaluate box IoU on scenes from held-out digits. Duplicate proposals and non-maximum suppression are explicit geometric post-processing, not a trained proposal-scoring network.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/object-detection-basics.mp4; data and run evidence: build/review-realdata-100/data/object-detection-basics.json. CPU layout, integrity and full video decode verified. -->
56. **`semantic-segmentation`** — Train a small NumPy U-Net for 1,000 updates on composed MNIST scenes with foreground masks derived from the digit pixels; measure per-class IoU on 80 scenes made from unseen test digits. This is a restricted synthetic-layout task using measured images.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/semantic-segmentation.mp4; data and run evidence: build/review-realdata-100/data/semantic-segmentation.json. CPU layout, integrity and full video decode verified. -->
57. **`batchnorm-in-cnns`** — Calculate batch-normalized and running-statistic activations on an existing MNIST CNN. This is post-hoc normalization of a fixed checkpoint, not evidence from training a BatchNorm architecture.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/batchnorm-in-cnns.mp4; data and run evidence: build/review-realdata-100/data/batchnorm-in-cnns.json. CPU layout, integrity and full video decode verified. -->
58. **`style-transfer`** — Optimize image pixels for 300 steps against content activations and a style Gram matrix from a fixed MNIST CNN; record the actual objective and intermediate images. The model weights stay fixed and the digit-scale result does not establish natural-image style quality.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/style-transfer.mp4; data and run evidence: build/review-realdata-100/data/style-transfer.json. CPU layout, integrity and full video decode verified. -->

## H. Generative models (8)

59. **`gan-training`** — Train two actual NumPy GANs for 1,800 updates each on 1,200 MNIST images represented by two training-only PCA coordinates. Compare the observed distributions and nearest training-class-centroid bin counts; these coordinates and bins are not generated full-resolution digit images or validated digit classes.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gan-training.mp4; data and run evidence: build/review-realdata-100/data/gan-training.json. CPU layout, integrity and full video decode verified. -->
60. **`normalizing-flows`** — Fit six affine-coupling layers with 1,800 NumPy updates to training-only PCA coordinates of MNIST; calculate invertibility error and actual train/test negative log likelihood. The density is over the two-coordinate representation, not all original pixels.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/normalizing-flows.mp4; data and run evidence: build/review-realdata-100/data/normalizing-flows.json. CPU layout, integrity and full video decode verified. -->
61. **`flow-matching`** — Fit a NumPy flow-matching velocity field to Gaussian-to-MNIST pairs in training-only PCA coordinates, then fit a rectified field from learned endpoint pairs. Compare actual integration paths and endpoint error relative to each learned field’s fine-step reference; this numerical error is not distance to an exact data-distribution solution.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/flow-matching.mp4; data and run evidence: build/review-realdata-100/data/flow-matching.json. CPU layout, integrity and full video decode verified. -->
62. **`classifier-free-guidance`** — Train a conditional NumPy diffusion model on real 7×7 MNIST for 2,200 updates with 15% condition dropout; sample guidance scales from matched starting noise. Evaluate 32 samples per scale with an independently trained classifier and within-class pixel variance; both are limited proxies for generative quality.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/classifier-free-guidance.mp4; data and run evidence: build/review-realdata-100/data/classifier-free-guidance.json. CPU layout, integrity and full video decode verified. -->
63. **`ddpm-vs-ddim`** — Sample the same existing MNIST-trained denoiser using stochastic DDPM and deterministic DDIM settings with matched initial noise. Compare actual generated images and classifier confidence from four samples per setting; no new denoiser training or distribution-quality guarantee is claimed.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/ddpm-vs-ddim.mp4; data and run evidence: build/review-realdata-100/data/ddpm-vs-ddim.json. CPU layout, integrity and full video decode verified. -->
64. **`vae-posterior-collapse`** — Train actual MNIST VAEs with a causal pixel decoder for 1,800 updates under fixed beta=4 and beta annealed from 0 to 1; record reconstruction and KL terms. The comparison changes both final KL strength and schedule, so it does not isolate annealing as the cause of any difference.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/vae-posterior-collapse.mp4; data and run evidence: build/review-realdata-100/data/vae-posterior-collapse.json. CPU layout, integrity and full video decode verified. -->
65. **`vq-vae-codebook`** — Train a NumPy vector-quantized autoencoder on real 7×7 MNIST for 1,800 updates; show nearest-code assignments, reconstructions, and code usage on 256 held-out images. Early dead-code restarts are part of the configured method; code usage need not be uniform.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/vq-vae-codebook.mp4; data and run evidence: build/review-realdata-100/data/vq-vae-codebook.json. CPU layout, integrity and full video decode verified. -->
66. **`autoregressive-pixels`** — Train a masked NumPy Bernoulli pixel model on real 7×7 MNIST for 1,300 updates and sample pixels in causal order. Show actual conditional likelihood and samples; the restricted local receptive field may miss global digit structure.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/autoregressive-pixels.mp4; data and run evidence: build/review-realdata-100/data/autoregressive-pixels.json. CPU layout, integrity and full video decode verified. -->

## I. Classical ML and evaluation (8)

67. **`bias-variance`** — Fit bootstrap polynomial regressions predicting measured right-half MNIST ink from measured left-half ink. Decompose finite-test squared error into variation across fitted models and squared error of their mean; without the true conditional expectation this is not a population bias/noise decomposition.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/bias-variance.mp4; data and run evidence: build/review-realdata-100/data/bias-variance.json. CPU layout, integrity and full video decode verified. -->
68. **`cross-validation`** — Run actual nested kernel-ridge selection on real MNIST threes and eights using pooled image pixels: five outer folds, four inner folds, six kernel widths, and a separate official test sample. PCA is used only for display; the observed selection gap is allowed to be small or absent.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/cross-validation.mp4; data and run evidence: build/review-realdata-100/data/cross-validation.json. CPU layout, integrity and full video decode verified. -->
69. **`svm-and-kernels`** — Fit linear, explicitly lifted, and RBF-kernel SVMs to real MNIST threes/eights represented with training-only PCA; evaluate each on separate test images. Compare the measured margins and test results without requiring a kernel or feature lift to improve accuracy.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/svm-and-kernels.mp4; data and run evidence: build/review-realdata-100/data/svm-and-kernels.json. CPU layout, integrity and full video decode verified. -->
70. **`gradient-boosting`** — Fit actual depth-three boosted and bagged trees to predict bottom-half MNIST ink from top-half image measurements; compare held-out MSE with 60 trees and two boosting rates. Saved trees reproduce the reported predictions; results are specific to this image-regression task.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gradient-boosting.mp4; data and run evidence: build/review-realdata-100/data/gradient-boosting.json. CPU layout, integrity and full video decode verified. -->
71. **`gaussian-mixture-em`** — Fit eight-component Gaussian mixtures by 60 EM iterations to training-only PCA coordinates of real MNIST images, using two initializations; record responsibilities and train/test log densities. Mixture components are density regions, not assumed digit classes.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gaussian-mixture-em.mp4; data and run evidence: build/review-realdata-100/data/gaussian-mixture-em.json. CPU layout, integrity and full video decode verified. -->
72. **`curse-of-dimensionality`** — Measure neighbor-distance contrast and held-out 1-nearest-neighbor accuracy as 2–49 training-only MNIST PCA coordinates are retained. Added coordinates may contain useful signal and improve accuracy; ball-versus-cube volume is a separately labeled analytic illustration.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/curse-of-dimensionality.mp4; data and run evidence: build/review-realdata-100/data/curse-of-dimensionality.json. CPU layout, integrity and full video decode verified. -->
73. **`tsne-and-umap`** — Optimize exact t-SNE embeddings of the same 240 real MNIST images for 400 iterations at perplexities 5, 30, and 100. These images constitute the embedding-fit dataset, not a generalization test; this implementation does not fit UMAP, and objectives across perplexities are not quality rankings.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/tsne-and-umap.mp4; data and run evidence: build/review-realdata-100/data/tsne-and-umap.json. CPU layout, integrity and full video decode verified. -->
74. **`roc-and-calibration`** — Use an existing NumPy-trained MNIST classifier to measure a seven-versus-rest confusion matrix, ROC/AUC, precision–recall, and probability calibration on 2,000 held-out images. There is no new training; precision and calibration depend on the evaluated sample and its prevalence.
   <!-- implemented: voxanim/next_vision.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/roc-and-calibration.mp4; data and run evidence: build/review-realdata-100/data/roc-and-calibration.json. CPU layout, integrity and full video decode verified. -->

## J. Optimisation, deeper (6)

75. **`newton-vs-gradient-descent`** — Actual NumPy fits of one-hot MNIST labels with a ridge-regularized linear model compare gradient descent, Adam and a full Hessian solve for 80 updates. Loss and official-test accuracy are recorded. The exact quadratic Hessian belongs to this measured-data regression problem; no synthetic learning surface substitutes for its results.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/newton-vs-gradient-descent.mp4; data and run evidence: build/review-realdata-100/data/newton-vs-gradient-descent.json. CPU layout, integrity and full video decode verified. -->
76. **`edge-of-stability`** — A pooled-pixel MNIST tanh classifier trains for 200 full-batch gradient updates at each of two learning rates. Finite-difference Hessian-vector products estimate curvature every five steps. The observed stability ratios are retained without assuming they converge to the quadratic boundary.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/edge-of-stability.mp4; data and run evidence: build/review-realdata-100/data/edge-of-stability.json. CPU layout, integrity and full video decode verified. -->
77. **`batch-size-and-noise`** — Actual MNIST classifier gradients estimate noise over 100 sampled batches at each batch size, alongside 100-update training runs and official-test losses. The inverse-batch curve is a labelled theoretical reference. Learning rates also vary with batch size, so this is not an isolated batch-size effect or a measurement of noise usefulness.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/batch-size-and-noise.mp4; data and run evidence: build/review-realdata-100/data/batch-size-and-noise.json. CPU layout, integrity and full video decode verified. -->
78. **`weight-averaging`** — A NumPy MNIST classifier makes 400 stochastic updates while retaining SGD, exponential-average and stochastic-weight-average weights. All three are evaluated on the same official-test images; recorded weight trajectories and final curvature are measured. Averaging is not assumed to improve either accuracy or curvature.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/weight-averaging.mp4; data and run evidence: build/review-realdata-100/data/weight-averaging.json. CPU layout, integrity and full video decode verified. -->
79. **`second-order-preconditioners`** — Actual MNIST ridge regression compares gradient descent, diagonal curvature scaling and a full Hessian solve, with saved weights and test accuracy after 80 updates. Eigenvalue ratios measure conditioning before and after diagonal or full coordinate whitening. This distinguishes rescaling coordinates from removing their coupling; it is not a K-FAC or Shampoo implementation or a large-network timing claim.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/second-order-preconditioners.mp4; data and run evidence: build/review-realdata-100/data/second-order-preconditioners.json. CPU layout, integrity and full video decode verified. -->
80. **`loss-spikes`** — Three tiny NumPy language-model runs make 141 updates on real Tiny Shakespeare windows. A controlled switch between two training batches at update 60 exposes the recorded losses, gradient norms and optimizer moments. The switch is disclosed and is not presented as spontaneous large-model instability.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/loss-spikes.mp4; data and run evidence: build/review-realdata-100/data/loss-spikes.json. CPU layout, integrity and full video decode verified. -->

## K. Retrieval and contrastive learning (5)

81. **`retrieval-augmented-generation`** — A TF-IDF retriever fitted on training text ranks five real held-out Shakespeare passages. Pretrained GPT-2 evaluates the same observed next token without context, with top-ranked context and with other context. This is fitted text statistics and existing-model inference, not local GPT-2 training or proof of factual reliability.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/gpt2-retrieval-augmented-generation.mp4; data and run evidence: build/review-realdata-100/data/gpt2-retrieval-augmented-generation.json. CPU layout, integrity and full video decode verified. -->
82. **`vector-search-hnsw`** — A NumPy contrastive encoder trains for 161 updates on paired views of actual MNIST images. HNSW indexes 256 training embeddings and answers 64 held-out queries; recall is compared with exact neighbors and distance-function calls are counted. Counted work is distinct from wall-clock speed.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/vector-search-hnsw.mp4; data and run evidence: build/review-realdata-100/data/vector-search-hnsw.json. CPU layout, integrity and full video decode verified. -->
83. **`infonce-temperature`** — Three linear NumPy encoders train for 161 updates on paired noisy MNIST views at different InfoNCE temperatures. Recorded objectives, same-label held-out alignment and test-to-training nearest-neighbor accuracy supply the comparisons. The two-dimensional display selects two learned coordinates.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/infonce-temperature.mp4; data and run evidence: build/review-realdata-100/data/infonce-temperature.json. CPU layout, integrity and full video decode verified. -->
84. **`hard-negatives`** — Actual MNIST contrastive training compares random negatives, heavily weighted different-label lookalikes, and deliberately weighted same-label false negatives. Held-out alignment, log-mean-exponential distance and nearest-neighbor accuracy are measured. The use of labels to construct this controlled comparison is explicit; hard negatives are not presumed best.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/hard-negatives.mp4; data and run evidence: build/review-realdata-100/data/hard-negatives.json. CPU layout, integrity and full video decode verified. -->
85. **`dimensionality-of-embeddings`** — MNIST paired-view encoders of widths 2 through 256 receive 241 actual NumPy updates with matched sampled images and noise. Official-test nearest-neighbor accuracy is measured. A nested-prefix model receives 121 additional updates, so its additional budget is disclosed. Float32 byte counts are exact arithmetic.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/dimensionality-of-embeddings.mp4; data and run evidence: build/review-realdata-100/data/dimensionality-of-embeddings.json. CPU layout, integrity and full video decode verified. -->

## L. Systems and scale (5)

86. **`data-parallel-allreduce`** — Partitioned gradients calculated from actual MNIST pixels and labels are reduced and gathered in NumPy, then compared with the whole-batch gradient. Worker buffers and communicated bytes are a simulated collective schedule and exact accounting, not measurements from a distributed training cluster.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/data-parallel-allreduce.mp4; data and run evidence: build/review-realdata-100/data/data-parallel-allreduce.json. CPU layout, integrity and full video decode verified. -->
87. **`tensor-and-pipeline-parallel`** — A NumPy layer processes measured MNIST pixel inputs both whole and partitioned, and numerical equivalence is checked. Pipeline occupancy is an explicitly simulated schedule over fixed layers and microbatches. These calculations do not claim training or distributed-hardware throughput.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/tensor-and-pipeline-parallel.mp4; data and run evidence: build/review-realdata-100/data/tensor-and-pipeline-parallel.json. CPU layout, integrity and full video decode verified. -->
88. **`zero-sharding`** — Parameter, gradient and optimizer-state storage for ZeRO stages is calculated from explicit assumed tensor sizes and worker counts. All-gather traffic and memory savings are analytical accounting, not a trained-model result or measured cluster memory.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/zero-sharding.mp4; data and run evidence: build/review-realdata-100/data/zero-sharding.json. CPU layout, integrity and full video decode verified. -->
89. **`activation-checkpointing`** — NumPy differentiation through a fixed small network compares gradients with and without checkpoint recomputation. Stored activations and forward calls are counted. The inputs are numerical test fixtures; no learning result or device-performance claim is made.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/activation-checkpointing.mp4; data and run evidence: build/review-realdata-100/data/activation-checkpointing.json. CPU layout, integrity and full video decode verified. -->
90. **`roofline`** — Local NumPy matrix multiplications are timed at actual shapes. Dominant FLOPs, logical bytes and the roofline envelope are labelled calculations/models. The measured timings are local observations, not hardware specifications or evidence of model learning.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/roofline.mp4; data and run evidence: build/review-realdata-100/data/roofline.json. CPU layout, integrity and full video decode verified. -->

## M. Reinforcement learning (6)

91. **`multi-armed-bandits`** — Ten NumPy MNIST classifiers, using different selected pixel counts, each train for 100 updates. Bandit rewards sample their recorded correctness on a separate validation set, with eight seeded 1,000-pull runs per method. Empirical-mean regret is labelled as such; no prescribed Bernoulli reward rates are substituted. Official-test classifier accuracy is retained separately.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/multi-armed-bandits.mp4; data and run evidence: build/review-realdata-100/data/multi-armed-bandits.json. CPU layout, integrity and full video decode verified. -->
92. **`policy-gradient`** — A softmax policy makes 500 actual NumPy REINFORCE updates on two views of real MNIST images, with and without a learned baseline. Sampled rewards are correctness against measured digit labels; the second view is action-independent. Held-out classification accuracy and within-batch update dispersion are measured. This narrow task does not demonstrate navigation or a general variance-reduction theorem.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/policy-gradient.mp4; data and run evidence: build/review-realdata-100/data/policy-gradient.json. CPU layout, integrity and full video decode verified. -->
93. **`actor-critic`** — An actor and linear critic train for 500 NumPy updates on partial-then-full MNIST observations. The critic uses a discounted next-view value and zero terminal value; the policy includes the corresponding time-discount weighting. Recorded label rewards, TD advantages and held-out classification accuracy demonstrate the update, not action-dependent environmental control.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/actor-critic.mp4; data and run evidence: build/review-realdata-100/data/actor-critic.json. CPU layout, integrity and full video decode verified. -->
94. **`dqn-replay-buffer`** — Four 500-update linear Q-learning runs use real MNIST two-view transitions in a matched replay-on/off by target-network-on/off ablation. Replay stores the actual observed state, action and label-derived reward; target values mask terminal states. This is a linear action-value model, not a deep network or a gridworld benchmark.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/dqn-replay-buffer.mp4; data and run evidence: build/review-realdata-100/data/dqn-replay-buffer.json. CPU layout, integrity and full video decode verified. -->
95. **`mcts-tic-tac-toe`** — Monte Carlo tree search executes actual selection, expansion, random rollout and backup on tic-tac-toe. Visit counts and rollout values are measured search statistics. This is a game-search calculation with no learned policy, neural-network training or claim of real-world data learning.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/mcts-tic-tac-toe.mp4; data and run evidence: build/review-realdata-100/data/mcts-tic-tac-toe.json. CPU layout, integrity and full video decode verified. -->
96. **`reward-shaping`** — Two NumPy actor-critic policies train on actual MNIST label rewards; one also receives an explicitly configured +1.2 bonus for abstaining. Original-task accuracy counts abstention as incorrect. The measured comparison illustrates a changed objective, not potential-based reward shaping or measured human utility.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/reward-shaping.mp4; data and run evidence: build/review-realdata-100/data/reward-shaping.json. CPU layout, integrity and full video decode verified. -->

## N. Memory, dynamics and curiosities (4)

97. **`hopfield-network`** — A Hopfield memory stores binarized actual MNIST training images using the Hebbian rule and runs asynchronous retrieval from a corrupted observed image. State changes and the exact energy are computed. This is associative-memory fitting and retrieval, not gradient-trained generalization or generated character data.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/hopfield-network.mp4; data and run evidence: build/review-realdata-100/data/hopfield-network.json. CPU layout, integrity and full video decode verified. -->
98. **`neural-cellular-automata`** — A NumPy neural cellular automaton optimizes an unrolled reconstruction objective for one actual MNIST training image. Recorded loss, evolved states and damage recovery come from the fitted update rule. A single-image fit does not establish generalization to other digits.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/neural-cellular-automata.mp4; data and run evidence: build/review-realdata-100/data/neural-cellular-automata.json. CPU layout, integrity and full video decode verified. -->
99. **`neural-ode`** — A neural ODE is trained for 241 NumPy updates to denoise corrupted two-dimensional MNIST coordinates, using PCA fitted only on training images. Held-out endpoint error is separated from error relative to a finer integration of the same learned field. Adjoint-gradient discrepancy is measured; integration time is not physical time or a measured dynamical law.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/neural-ode.mp4; data and run evidence: build/review-realdata-100/data/neural-ode.json. CPU layout, integrity and full video decode verified. -->
100. **`evolution-strategies`** — An exact two-dimensional quadratic supplies a labelled optimizer illustration. Separately, a small NumPy MNIST classifier receives 80 actual updates using backpropagation or antithetic evolution strategies; the latter makes 64 paired perturbations per update and no backward pass. Losses and gradient-estimator error are measured, without claiming equal computational budgets.
   <!-- implemented: voxanim/next_systems.py and voxanim/next_systems_real.py; measurement revision implemented; current scope is described above. Review video: build/review-realdata-100/media/evolution-strategies.mp4; data and run evidence: build/review-realdata-100/data/evolution-strategies.json. CPU layout, integrity and full video decode verified. -->

---

## Validation of the revised builds

All 250 numerical, dataset, integration and renderer tests pass. All 100 study
layouts pass 120 sampled frames each. Numerical checks cover
analytic gradients, train/test separation, saved checkpoint identity, update
counts, and independent replay of selected plotted metrics. All 100 final videos pass independent file-integrity, copied-artifact,
frame-count and complete decoding checks in `tools/verify_next_review.py`.
See [the verification report](build/review-realdata-100/verification-report.json).

Rendering uses the NumPy CPU backend. **GPU parity is not verified:** the
explicit MPS check cannot run because PyTorch is absent in this environment.
This is recorded as unavailable, not a pass. See the gallery's
`gpucheck-unavailable.json` and `cpu-renderer-regression.log`.

## Suggested review order (no release approval)

1. **Studies S1 and S2.** Almost half the list depends on them, and S1 is a
   straight generalisation of `posenc.py`'s tested transformer.
2. **Attention and the block** (A, B, 20 animations). This
   continues the embeddings family directly; `causal-vs-bidirectional` is
   the opener.
3. **Language models in practice** (C, D, E, 22). Needs S3 and
   S4.
4. **Inside the model** (F, plus K's retrieval pieces, 13).
5. **Beyond language** (G, H, I, J, 30).
6. **Systems, RL and curiosities** (L, M, N, 15).

Use the prerequisite sequence and accessibility priorities in TODO2.md when
reviewing these films with physicists new to machine learning.

## Experiments redesigned, 16 November 2026

After the drawings were rebuilt, 56 experiments were redesigned because the first versions could not
show their concept or came out null at their size. The replacements are in `voxanim/study_*.py`
(`voxanim.next_catalog.redesigned`); the first versions remain in the `next_*` modules, unused by the
catalogue. Comparisons between models now run over at least three seeds. Entries above describe the
first experiments; the current description of each is in `voxanim/next_catalog.json`.

**Now show their concept:** grokking (on MNIST, large initial weights and weight decay), warm-up,
pre-LN against post-LN, sliding-window reach, sparse attention patterns, edge of stability, Newton
against gradient descent, bias and variance, curse of dimensionality, t-SNE and UMAP (UMAP now
implemented), policy-gradient baseline, actor-critic, DQN replay, reward model, RLHF/PPO, DPO,
instruction tuning, prompt and prefix tuning, sparse-autoencoder features, head ablation,
induction heads from scratch, batch-norm bug, classifier-free guidance, autoregressive pixels,
normalizing flows, contrastive temperature, hard negatives, Hopfield capacity, cellular-automaton
regrowth, neural ODE, ring all-reduce, tensor and pipeline parallelism, activation checkpointing,
repetition penalties, constrained decoding.

**Show it partly (the limit is stated on screen):**
- `speculative-decoding`: 0.71 GPT-2 passes per token against 1, but 1.2 times slower in numpy wall-clock.
- `scaling-laws`: loss falls with compute and the best size moves with the budget; the fitted power law does not extrapolate.
- `loss-spikes`: spikes arise on fixed data and a short memory stops them; a larger epsilon and clipping do not.
- `reward-shaping`: potential-based shaping preserves the policy and a naive bonus is farmed; no speed-up from shaping.
- `reward-hacking`: appears in 4 of 30 runs, with weaker reward models; the run drawn is the labelled worst case.
- `model-merging`: averaging two task vectors works; the full sum overshoots and negation is not selective.
- `gaussian-mixture-em`: the start matters (86% of random starts reach the best fit); k-means++ seeding alone does not help.
- `gan-training`: the plain GAN drops a digit at 19% of late checkpoints and recovers; no permanent collapse.
- `vq-vae-codebook`: healthy codebook 64 of 64; the collapse control only appears with 64-number codes started far away.
- `feature-visualisation`: stroke fragments a second classifier reads correctly 27 of 30 times, not well-formed digits.
- `vit-attention`: attention is concentrated (about 12 of 49 patches); its reach equals that of even attention.
- `retrieval-augmented-generation`: helps only when the store holds the answer line; no better than a random line otherwise.
- `multi-head-vs-single`: eight heads beat one in every seed, by 0.03 nats; heads differ by a seed-dependent amount.
- `style-transfer`: the thin style transfers cleanly; the heavy style also leaves background haze.
- `roofline`: ceilings now measured independently; timings are local and vary with machine load.

**Still null after a fair redesign (kept, and said on screen):**
- `prefix-lm-attention`: prefix against causal, 96.4% against 95.9% with overlapping seeds. (It does show why a bidirectional reply cannot be generated.)
- `transformer-depth-vs-width`: with equal weights, depth past two blocks does not pay at this size and budget.
- `contrastive-decoding`: the contrastive pick is the real next token less often than plain GPT-2's (18% against 25%); a stronger amateur model is the untried follow-up.
- `swiglu-mlp`, `rmsnorm-vs-layernorm`: within seed noise over eight seeds (for RMSNorm that is the expected result).

**Constructed from real data (check against the data rule):** `instruction-tuning`,
`prompt-and-prefix-tuning` and `model-merging` use Tiny Shakespeare in capitals, backwards, or with
answers computed from a line; `induction-heads` trains its small model on a real stretch read
repeatedly; `curse-of-dimensionality` adds real pixel columns shuffled across images;
`dqn-replay-buffer` streams digits one class at a time. Each is labelled on screen.

**Not verified:** uncached build time on an idle machine. Several studies took 3 to 5 minutes while
fifteen builds ran at once (`transformer-depth-vs-width` 309 s, `speculative-decoding` 165 s re-run
alone, `prefix-lm-attention` 212 s, `pre-ln-vs-post-ln` 217 s, `loss-spikes` 183 s).

## Added after the hundred, 16 November 2026

Nine animations in two new groups (catalogue entries 101 to 109; `mech_families.py`,
`mech_labels.py` and their `study_*.py`), built to the same rules: MNIST only, held-out numbers,
three or more seeds.

- **Generative families compared:** `generative-objectives`, `generative-latent-spaces`,
  `generative-sampling`, `generative-reconstruction`, `generative-interpolation`. Shown as
  measured: DEC and IDEC reach a cluster accuracy of only about 0.5 at this size; by a classifier's
  confidence no code-space walk between two digits beats a pixel cross-fade, though every one stays
  nearer real digits; samples are rough because the networks are small.
- **Label embeddings:** `label-embedding`, `label-embedding-classifier`,
  `label-embedding-autoencoder`, `label-embedding-generation`. Shown as measured: with 16-number
  label rows, learning a classifier's table buys 0.1 points over a frozen random one; telling an
  autoencoder's decoder the label only empties the code of the digit when the code is small; a
  swapped label is obeyed 88% of the time with a 2-number code and 2% with 16.

## Randomness and entropy, 17 November 2026

Eight animations in a new group (catalogue entries 110 to 117; `mech_random.py`, `mech_chance.py`
and their `study_*.py`), asked for as "random sampling, markov chain, other sources of pseudo
entropy and the impact of several seeds on random outputs and measuring/evaluating
(pseudo-)entropy". Released in 1.7.0.

- `pseudo-random-generators`, `seeds-and-streams`, `sampling-a-distribution`, `entropy-sources`
  need no dataset: exact arithmetic and statistics of generators written out in integers.
- `measuring-entropy` and `markov-chains` use Tiny Shakespeare; `seeds-and-training-outcomes`
  MNIST (320 training runs); `gpt2-seeds-and-sampled-text` the pinned GPT-2 on held-out passages.

**Shown as measured, including what did not separate anything:** serial correlation is near zero
for every generator, RANDU included; the usual correlation coefficient between a weak generator's
related streams is 0.008 though one stream is predicted exactly from two others; a bad source
spoils which token follows which but not the single-token shares; von Neumann's trick does not
fully clean the timer bit, whose bits are correlated; top-p 0.9 removes few unlikely tokens on
these prompts; the variances from weights, data order and dropout masks do not add up.

**Check against the data rule:** the five token probabilities in `sampling-a-distribution` are
written out, not measured (labelled on screen). `entropy-sources` and `measuring-entropy` measure
the building machine's timer and system bytes: cached, labelled as local, never asserted by value
in tests, and not reproducible on another machine.

**Not verified:** `gpt2-seeds-and-sampled-text` built in 113 s alone and 235 s under load.

## A physics course on data analysis and machine learning, 17 November 2026

Forty-one animations in nine new groups (catalogue entries 118 to 158), asked for to support an
undergraduate course whose topics are probability, chi-square and likelihood fits, errors and
systematics, selection, hypothesis tests, discoveries and limits, network regression and
classification, boosted trees, and (advanced) generative models, unsupervised learning and
Gaussian processes: "more theory, more animations". Released in 1.7.0. Detector simulation (Geant) was
left out at the owner's request, and with it a generative model as a fast shower simulator.

**Data rule, as applied here.** The statistics groups use pseudo-experiments: seeded draws from a
model stated on screen, with the exact answer drawn beside them wherever one exists. No measured
physics data is used anywhere. The machine-learning animations train on MNIST, with digits standing
in for events and the physics counterpart named on screen.

**Shown as measured, against the usual telling:**
- `likelihood-intervals`: at 5 events the symmetric error does not cover worse in total than the
  half-unit interval (67.8% against 67.4%); it misses on one side (27% below, 5% above).
- `chi-square-per-dof`: chi-square per degree of freedom does not flag a fit with too many
  parameters; the gain of only 1 per added parameter does.
- `binned-vs-unbinned-fits`: the chi-square bias changes sign, +12% at 30 events and negative from 60.
- `minimisers-on-chi-square`: the simplex reaches the best fit from more starts than the
  variable-metric method (60% against 44%). That method is plain BFGS, not MIGRAD.
- `cuts-vs-multivariate`: rectangular cuts gain nothing over one variable alone at 80% efficiency.
- `look-elsewhere-effect`: no pseudo-experiment reached a local 5 sigma; that row is an extrapolation.
- `brazil-band`: a point leaves the 95% band 3.9% of the time, because counts are whole numbers.
- `boosted-regression-trees`: more trees do not overfit at small learning rates within 300 trees.
- `overtraining-check`: overtraining barely lowers the held-out ROC area; it spoils the score's meaning.
- `autoencoder-anomaly-detection`: an unseen 1 is rebuilt better than a seen digit (area 0.32).
- `gan-mode-collapse`: every generator-heavy run collapsed, whatever the seed; with the label the
  digit asked for is drawn 73 to 77% of the time, not always.
- `gp-background-fit`: at its preferred length scale the process still loses a third of a real signal.
- `anomaly-loss-and-latent`, `anomaly-two-score-flow` (asked for as "2x anomaly detection flow
  ... where anomalies may appear both in loss-space and the latent-space distributions"): a second,
  latent-space score does not beat the reconstruction loss on average at a code of 8 numbers; its
  gain is the class the loss cannot see, and only partly (unseen 1s: 0.4% caught to 7% at 5% false
  alarms). A Mahalanobis distance is worse than a coin toss; the normalizing flow is the weakest
  of the flexible densities and a plain neighbour distance beats every fitted density. With a
  larger code the anomaly moves from the loss into the code. The neighbour distance was chosen
  over the Mahalanobis score after seeing anomalies, and says so on screen.

**Chosen, not drawn, or unlucky draws kept:** the count of 4 in `cls-method` is chosen; the
`bump-hunt` spectrum fluctuated up (4.09 sigma where 3.30 is typical); the run in
`statistical-vs-systematic` ends 0.61 high on a shift of 0.4; the `bootstrap-errors` dataset has
one very large event; the `look-elsewhere-effect` spectrum is the first of 100,000 past 3 sigma.
Each says so on screen or in its limitations.

**Not verified:** uncached build time on an idle machine for `autoencoder-anomaly-detection`
(361 s under load), `gan-mode-collapse` (about 211 s under load) and `anomaly-two-score-flow`
(171 to 201 s under load, with the first anomaly study's runs cached).
