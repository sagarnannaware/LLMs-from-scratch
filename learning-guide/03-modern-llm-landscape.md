# From GPT-2 to 2026: The Modern LLM Landscape

The book teaches you GPT-2 (2019). That is the right thing to learn first — the fundamentals genuinely have not changed. This document is the delta: every significant idea layered on top since, explained assuming you've done the chapters.

**How to read this:** each section states what the book does, what changed, and why. The "why" matters more than the names — architectures get replaced, but the pressures that produced them (memory bandwidth, KV-cache size, verifiable rewards, inference economics) persist.

> **On currency:** this reflects the field as of mid-2026. Specific model names age in months; the mechanisms age in years. Where a claim is about a moving target, it's flagged.

---

## 0. What hasn't changed

Before the delta, the invariants. Everything below still describes every frontier model in 2026:

- Text → tokens → embeddings → stacked blocks → logits → sampled token → repeat.
- Trained by next-token prediction with cross-entropy loss.
- Attention mixes information across positions; a position-wise FFN transforms each token.
- Residual connections + normalization make depth trainable.
- Backprop with an Adam-family optimizer, warmup + decay, gradient clipping.

If you understand [`ch04/01_main-chapter-code/gpt.py`](../ch04/01_main-chapter-code/gpt.py), you understand the skeleton of a trillion-parameter model. What follows is refinement, not replacement.

---

## 1. Architecture changes

### 1.1 The standard modern block

Here is the GPT-2 block you built, next to what a 2026 dense model looks like:

| Component | GPT-2 (book) | Modern standard | Reason |
|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | Drops mean-centering; ~equal quality, less compute |
| Norm placement | Pre-LN | Pre-LN (+ sometimes extra norms on QK or after sublayers) | Stability at depth and scale |
| Activation/FFN | GELU, 2 matrices, 4× | **SwiGLU**, 3 matrices, ~8/3× | Gating improves quality per parameter |
| Position | Learned absolute | **RoPE** | Relative, extrapolates, no position table |
| Attention | MHA | **GQA** (or MLA) | Shrinks KV cache; the binding serving constraint |
| Biases | Yes in QKV option | **None** | No measurable benefit |
| Dropout | 0.1 | **0.0** | Single-epoch training on huge corpora doesn't overfit |
| FFN | Dense | Dense **or MoE** | MoE decouples capacity from per-token compute |

You can implement all of these on top of your own code — that is exactly what [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama) walks through. Do it. It converts abstract terminology into diffs you understand.

### 1.2 RoPE, and why position went relative

GPT-2's learned position table has two problems: it's a hard cap (nothing exists at index 1025), and it encodes *absolute* position when what matters linguistically is usually *relative* distance.

RoPE rotates each query and key vector by an angle proportional to its position. Because rotation is orthogonal, the dot product between a query at position *m* and a key at position *n* ends up depending on `m − n`. No parameters, no table, and the relative structure is baked into the attention math itself.

**Context extension.** A model trained at 8k can be pushed to 128k+ by modifying RoPE's frequencies — linear interpolation, NTK-aware scaling, and **YaRN** are the common recipes, usually followed by brief finetuning at the longer length. This is how "long-context" versions of models are typically produced.

### 1.3 GQA, MLA, and the KV-cache problem

You computed it in ch04's exercises: a 124M model at 1024 context needs ~36 MB of KV cache per sequence. Scale that to 70B parameters at 128k context and the cache exceeds the model weights. It grows linearly with batch size, so it directly caps how many users a GPU can serve.

Three responses, in order of increasing sophistication:

- **MQA** — one shared K/V head for all query heads. Maximum savings, noticeable quality cost.
- **GQA** — query heads grouped, each group sharing K/V. E.g. 32 query heads → 8 K/V heads = 4× smaller cache. The pragmatic default across Llama, Qwen and Mistral families.
- **MLA** — compress K/V into a low-rank latent, decompress at use. DeepSeek's approach; smaller cache still, at more architectural complexity.

Everything about long-context serving traces back to this table. When you read that a model "supports 1M tokens," the first question to ask is what its KV cache costs per token.

### 1.4 Mixture of Experts

**The problem:** quality scales with parameters, but compute per token scales with parameters too. You want the first without the second.

**The mechanism:** replace the FFN with N expert FFNs plus a small router. For each token, the router picks the top-k experts (k typically 1–8 of dozens or hundreds). Only those run. A model with 235B total parameters might activate 22B per token — roughly 22B-model speed with much-better-than-22B quality.

**What you must not forget:** *all* parameters must be in memory even though only a few are used. MoE saves compute, not memory. A "3B active" model does not fit in a 3B model's VRAM.

**The hard parts:**
- **Load balancing.** Routers collapse onto favorite experts if unchecked. Older fixes add an auxiliary loss; newer designs adjust routing biases directly to avoid distorting the main objective.
- **Communication.** Experts live on different devices, so every token's routing implies a network hop. Expert-parallel communication is the main systems bottleneck.
- **Shared experts.** Some designs always route through one or two common experts so the routed ones can genuinely specialize.

By mid-2026 sparse MoE is the dominant capacity-scaling pattern in serious open-weight releases, precisely because it decouples total parameters from per-token compute ([TensorOps field guide](https://tensorops.ai/blog/what-is-mixture-of-experts-llm), [Raschka, *Beyond Standard LLMs*](https://magazine.sebastianraschka.com/p/beyond-standard-llms)).

### 1.5 Attention alternatives and hybrids

Attention's `O(n²)` cost and linear KV growth motivated a long search for alternatives:

- **Sliding-window attention** — attend only to the last *w* tokens. Linear cost, bounded cache; loses long-range recall unless interleaved with full layers.
- **Linear attention** — reformulate so the `(n, n)` matrix never materializes, giving a fixed-size recurrent state. Early versions lost too much quality; **gated** variants (Gated DeltaNet and relatives) closed much of the gap.
- **SSMs (Mamba)** — selective state-space models with input-dependent transitions: constant memory per token, no KV cache at all.

The winning pattern in 2026 is **hybrid**: mostly linear/SSM layers for cheap long-context throughput, with a minority of full-attention layers preserving precise recall. Recent releases combine gated linear attention, standard gated attention, and sparse MoE in one stack ([DEV: Transformer Architecture in 2026](https://dev.to/jintukumardas/transformer-architecture-in-2026-from-attention-to-mixture-of-experts-moe-3d46), [Electronics review of post-transformer hybrids](https://doi.org/10.3390/electronics15153254)).

The useful frame: these are **attention-budgeting** architectures. They spend exact quadratic attention only where recall demands it.

### 1.6 Other notable adjustments

- **QK normalization** — normalizing queries and keys before the dot product, to stabilize very large runs.
- **Attention sinks** — models dump surplus attention on the first tokens; keeping those tokens pinned in the cache (StreamingLLM) enables endless streaming, and some models now include explicit sink parameters.
- **Multi-token prediction** — extra heads predicting several future tokens, used as a training signal and later reused for self-speculative decoding.
- **Depth/width tradeoffs and tied embeddings** are back in fashion for small models, where memory dominates.

---

## 2. Training at scale

### 2.1 Scaling laws, and why models are deliberately "overtrained"

Kaplan et al. (2020) established loss as a power law in compute, parameters and data. Chinchilla (2022) corrected the exponents: for a fixed training budget, use ≈20 tokens per parameter.

Then economics intervened. Chinchilla optimizes *training* cost, but a deployed model's *inference* cost dominates over its lifetime. So the industry now trains far smaller models on far more data — hundreds to thousands of tokens per parameter — deliberately past the compute-optimal point, because a smaller model is cheaper forever. That is why an 8B model in 2026 outperforms a 70B model from 2023.

Practical consequence: **parameter count is a poor proxy for capability.** Training tokens, data quality and post-training matter at least as much.

### 2.2 Data is the differentiator

Architectures across labs are strikingly similar. Data recipes are the closely guarded part. The known levers:

- **Deduplication** at document and n-gram level.
- **Quality filtering** with learned classifiers — often worth more than extra parameters.
- **Mixture design** — code improves reasoning even on non-code tasks; math and multilingual proportions carry big trade-offs.
- **Curriculum / annealing** — a final "mid-training" phase on high-quality data (textbooks, curated code, synthetic reasoning) at decayed learning rate, which reliably produces large gains.
- **Synthetic data** — filtered model-generated data, now a large share of post-training corpora and a growing share of pretraining. You practice exactly this in [`ch07/05_dataset-generation`](../ch07/05_dataset-generation).

### 2.3 Systems and precision

- **bf16** is the default training precision; **fp8** is used for parts of frontier runs with careful scaling.
- **FSDP/ZeRO** shard optimizer state, gradients and parameters instead of replicating them.
- **Parallelism** is combined along several axes: data, tensor, pipeline, expert, sequence/context.
- **Gradient checkpointing** trades ~30% recompute for large activation-memory savings.
- **FlashAttention** made long-context training practical by never materializing the attention matrix.
- **Optimizers**: AdamW remains the baseline, but matrix-aware optimizers (notably **Muon**) have shown real loss-per-token improvements and are now used in production runs — the first serious challenge to Adam's decade of dominance.

---

## 3. Post-training: the biggest change since the book

This is where the field moved fastest. The book's ch07 (SFT) and its DPO bonus are the entry points; here is the full pipeline as it exists in 2026.

### 3.1 The evolution

```
2020  SFT only
2022  SFT → reward model → PPO (RLHF)              [InstructGPT/ChatGPT]
2023  SFT → DPO                                     [simpler, no RM, no rollouts]
2024  SFT → DPO/PPO → light RLVR
2025  SFT → RLVR with GRPO                          [DeepSeek-R1: reasoning emerges from RL]
2026  Multi-stage: SFT → RLVR → agentic RL in environments → preference polish
```

The standard RLHF-only recipe of a few years ago is effectively obsolete; every recent frontier release uses a multi-stage stack, and RLVR with GRPO-family algorithms is the dominant reasoning method ([Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026), [RL for LLMs in 2026](https://shivu-agr.medium.com/rl-for-llms-in-2026-from-ppo-to-dpo-to-grpo-to-multi-agent-rl-1c00e1dacba7)).

### 3.2 RLHF → DPO (what you already learned)

**RLHF/PPO:** collect human preference pairs → train a reward model → optimize the policy with PPO against that reward, with a KL penalty anchoring it to the reference model. Four models in memory, notoriously finicky.

**DPO:** an algebraic result shows the optimal RLHF policy can be recovered by minimizing a simple loss directly on `(prompt, chosen, rejected)` triples. No reward model, no sampling loop. Two models, a stable loss, far less infrastructure. You implement it in [`ch07/04_preference-tuning-with-dpo`](../ch07/04_preference-tuning-with-dpo).

**Variants worth knowing:** IPO (fixes a DPO overfitting pathology), KTO (learns from thumbs-up/down without pairs), ORPO (folds preference optimization into SFT in one stage), SimPO (drops the reference model).

### 3.3 RLVR: the reasoning breakthrough

The insight is deceptively simple: **for domains with checkable answers, you don't need a learned reward model at all.** Run the code against unit tests. Check the math answer. Compile the program. The reward is 1 or 0 and cannot be gamed the way a learned reward model can.

This unlocked large-scale RL on math and code, and out of it fell an unexpected result: models trained this way *spontaneously* develop longer reasoning, self-checking and backtracking, without being shown any examples of those behaviors. Nobody supervised "think step by step" — it emerged because it raised the probability of a correct final answer.

**GRPO** is the algorithm that made this cheap. PPO needs a value network to compute advantages — a second large model to train and hold. GRPO instead samples a *group* of G responses to the same prompt and uses the group's mean reward as the baseline. The critic disappears; memory and complexity drop sharply.

```
for each prompt:
    sample G responses from the current policy
    score each with the verifier            (1 = passes tests, 0 = fails)
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    update the policy to raise the probability of above-average responses
```

**The active refinements** (DAPO, GSPO, and a steady stream of others) tune clipping ranges, sampling strategies, sequence- vs token-level weighting, and how to handle groups where every sample succeeds or every sample fails and the gradient vanishes.

**Where it's going:** RLVR is spreading from math/code into any domain where a verifier can be built — SQL correctness, tool-use success, retrieval accuracy, even calibrated abstention ("reward the model for saying I don't know when it would otherwise be wrong"). The frontier is *interactive environments*: browsers, filesystems, databases, APIs — where the reward is task completion. That shift, from static preference datasets to environments, is what separates a "chat model" from an "agent model" ([post-training survey](https://llm-stats.com/blog/research/post-training-techniques-2026)).

### 3.4 Other post-training techniques

- **Rejection sampling / best-of-n finetuning** — sample many, keep the best by a verifier, SFT on those. Simple and effective.
- **Distillation** — the reason strong small models exist. Train the student on a large model's reasoning traces.
- **Model merging** — average weights of models finetuned from a common base to combine skills with no additional training.
- **Constitutional AI / RLAIF** — model-generated critiques against written principles replace much human labeling.

---

## 4. Reasoning models and test-time compute

A **reasoning model** is post-trained to produce an extended internal deliberation before answering. Practically this created a second scaling axis: instead of only "train a bigger model," you can "let it think longer at inference."

Key ideas:

- **Long chain-of-thought** — thousands of tokens of reasoning, often hidden from the user.
- **Test-time compute scaling** — accuracy climbs measurably with reasoning length on hard problems, and flattens (or reverses) on easy ones.
- **Thinking budgets** — user- or system-controlled effort levels, because reasoning tokens are billed tokens and add latency.
- **Self-consistency** — sample several reasoning paths, take the majority.
- **Process reward models** — grade each step rather than the final answer; better credit assignment, more expensive labels.
- **Overthinking** — a genuine production problem: reasoning models burn hundreds of tokens on trivial questions. Hybrid "reason only when needed" routing is now standard product behavior.

The practical takeaway for building things: reasoning models are worth their cost on hard, verifiable, multi-step tasks (math, debugging, planning) and are usually a waste on extraction, classification and formatting.

---

## 5. Inference and serving

The book generates one token at a time with a plain loop. Production changes everything about that loop.

### 5.1 The KV cache is the whole game

Once you cache K and V, generating token *n* costs one forward pass over one token instead of *n*. The cost moves from compute to **memory**: every generated token reads the entire model's weights from HBM. Decoding is memory-bandwidth-bound, and that single fact explains most serving design:

- **Quantization speeds up decoding** — fewer bytes to move, not fewer operations.
- **Batching is nearly free per-user** — the weights are read once for the whole batch. Which is why throughput and cost are so batch-dependent.
- **Prefill and decode are different workloads** — prefill is compute-bound and parallel; decode is bandwidth-bound and sequential. Hence *disaggregated serving*, running them on separate pools.

### 5.2 The serving toolkit

- **Paged attention** (vLLM) — manage the cache in fixed-size pages like OS virtual memory; kills fragmentation and enables sharing.
- **Continuous batching** — swap finished sequences out and new ones in mid-flight rather than waiting for the slowest in a batch.
- **Prefix caching** — reuse the cache for shared prefixes (system prompts, retrieved documents) across requests. Often the single biggest cost win in a real application.
- **Speculative decoding** — a small draft model proposes k tokens; the big model verifies all k in one forward pass and accepts the longest correct prefix. Provably identical output distribution, typically 2–3× faster. Self-speculative variants (EAGLE, Medusa) skip the separate draft model.
- **Structured/constrained decoding** — mask logits against a grammar or JSON schema so output is guaranteed parseable. Removes an entire class of application bugs.

### 5.3 Quantization in practice

| Precision | Typical use | Quality impact |
|---|---|---|
| bf16/fp16 | Training, high-fidelity serving | Baseline |
| fp8 | Frontier serving, some training | Near-lossless with good scaling |
| int8 | Common serving default | Very small |
| int4 (GPTQ/AWQ/GGUF) | Consumer GPUs, local inference | Small but measurable; worse for reasoning and long context |
| below int4 | Research, extreme edge | Significant without QAT |

Rule of thumb: **a larger model at int4 usually beats a smaller model at bf16** for the same memory budget — but check on *your* task, since quantization damage lands unevenly (long-context recall and multi-step reasoning degrade first).

---

## 6. Building applications

### 6.1 RAG, matured

Naive RAG (embed chunks → top-k → stuff into prompt) is a starting point, not a system. What's standard now:

1. **Hybrid retrieval** — dense embeddings + BM25 keyword search. Nearly always beats either alone.
2. **Reranking** — a cross-encoder rescoring the top ~50 down to the top ~5. Cheap, large accuracy gain.
3. **Query transformation** — rewriting, decomposition into sub-queries, HyDE.
4. **Smart chunking** — semantic or structure-aware splitting, with parent-document retrieval so you match on small chunks but pass in large context.
5. **Agentic retrieval** — the model decides when and what to search, iteratively, rather than a fixed single-shot pipeline.
6. **Grounded citation** — require source attribution; makes hallucination visible and auditable.
7. **GraphRAG** — entity/relationship graphs over the corpus for questions requiring synthesis across many documents.

Long contexts did not kill RAG. Retrieval is cheaper, fresher, auditable, and avoids diluting attention across a million irrelevant tokens.

### 6.2 Agents

An agent is a loop: model → tool call → observation → model → … until done. What has actually changed is that models are now *trained* for this loop rather than merely prompted into it, which is why agentic reliability jumped.

What matters in practice:

- **Tool design beats prompt wording.** Fewer, well-named tools with clear schemas and informative errors outperform a large flat tool surface.
- **MCP (Model Context Protocol)** standardized tool/data connections, so integrations aren't rebuilt per framework.
- **Context engineering** is the discipline: what goes into the window, in what order, and what gets compacted out. Long agent runs live or die on this.
- **Memory** — durable state across sessions, usually a retrieval layer over past interactions plus explicit summaries.
- **Multi-agent** — orchestrator/worker splits help when subtasks are genuinely independent; they add coordination failure modes otherwise. Default to one agent until you can name the reason for more.
- **Evaluation is trajectory-level.** Grading only the final answer hides where an agent went wrong.
- **Security is the hard part.** Prompt injection through retrieved content is unsolved in general — an agent that reads untrusted data and holds credentials is a real vulnerability. Mitigate structurally: least privilege, sandboxing, human approval on consequential actions, and never assume the model can tell instruction from data.

### 6.3 What "prompt engineering" became

The tricks that mattered in 2022 (magic phrases, "you are an expert", elaborate CoT scaffolds) matter far less now — models are trained to reason and follow instructions natively. What remains valuable:

- Clear task specification, explicit output format, worked examples for genuinely ambiguous cases.
- Deciding what context the model needs and getting it there (retrieval, tools, memory).
- Programmatic constraints instead of pleading: schemas, validators, retries.
- Evals. **Always evals.** A prompt without a regression test is a prompt you can't safely change.

---

## 7. Evaluation: the hardest unsolved practical problem

Benchmark scores have become progressively less informative — contamination, overfitting to public leaderboards, and the gap between benchmark tasks and real work all compound. What practitioners actually do:

- **Build a domain eval set.** 50–200 examples from your real traffic beats any public benchmark for your decisions.
- **LLM-as-a-judge with discipline** — different model family than the one under test, randomized option order, explicit rubrics, and periodic human spot-checks against judge scores. You already built this in [`ch07/03_model-evaluation`](../ch07/03_model-evaluation).
- **Pairwise comparison** over absolute scoring; models and humans are both better at "which is better" than "rate this 1–10."
- **Eval in CI.** Run the suite on every prompt, model or retrieval change.
- **Trace everything.** In agents, most failures are visible only in the trajectory.

Public benchmarks still worth tracking: GPQA and similar hard-reasoning sets, SWE-bench-style agentic coding, long-context recall suites, and live arenas — read them as weak signals, not ground truth.

---

## 8. Safety and security

**Hallucination** is structural. A model trained to always emit a plausible continuation, and evaluated by benchmarks that reward guessing over abstaining, will guess. Mitigations: retrieval grounding with citations, verifiable checking where possible, calibration training, and explicitly rewarding abstention — an active RLVR research direction.

**Prompt injection** is the central agent security problem, and it is not solved. The model has no reliable way to distinguish "instructions from my principal" from "text in a document I was told to read." Treat every agent that reads untrusted content as a potential confused deputy and design the *system* accordingly: least privilege on tools, sandboxed execution, approval gates on irreversible actions, and no secrets in a context that untrusted data can reach.

**Interpretability** matured from curiosity to tooling. Sparse autoencoders decompose activations into human-readable features; steering vectors let you nudge behavior along those directions. Useful for debugging model behavior and for safety auditing — and genuinely fascinating if you want a research direction.

---

## 9. How to stay current without drowning

The volume is unmanageable if you try to read everything. A workable filter:

**Weekly (30 min):**
- [Sebastian Raschka's *Ahead of AI*](https://magazine.sebastianraschka.com/) — the author of this book; his architecture comparisons and paper roundups are the single best-value source for someone at your stage.
- Release notes and model cards from major labs. Read the *architecture and training* sections, skip the benchmark tables.

**Monthly:**
- One paper read properly, end to end, rather than ten abstracts. Reimplement one component of it.
- Check what changed in vLLM / PyTorch / your inference stack's release notes — systems progress is where practical wins hide.

**Ignore:**
- Benchmark leaderboard shuffles.
- "X kills Y" takes.
- Anything that doesn't tell you *why* something works.

**The heuristic that keeps you sane:** for every new thing, ask which of the four buckets it lives in — better input representation, better cross-position mixing, better weight fitting, or cheaper execution. If you can't place it, it's probably marketing.

---

## Sources

- [Beyond Standard LLMs — Sebastian Raschka](https://magazine.sebastianraschka.com/p/beyond-standard-llms)
- [LLM Research Papers: The 2026 List — Sebastian Raschka](https://magazine.sebastianraschka.com/p/llm-research-papers-2026-part1)
- [LLM Mixture of Experts Explained — A 2026 Field Guide (TensorOps)](https://tensorops.ai/blog/what-is-mixture-of-experts-llm)
- [Transformer Architecture in 2026: From Attention to Mixture of Experts](https://dev.to/jintukumardas/transformer-architecture-in-2026-from-attention-to-mixture-of-experts-moe-3d46)
- [Current Trends in AI Architectures: Post-Transformer Hybrids and World Models (Electronics, MDPI)](https://doi.org/10.3390/electronics15153254)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [RL for LLMs in 2026: From PPO to DPO to GRPO to Multi-Agent RL](https://shivu-agr.medium.com/rl-for-llms-in-2026-from-ppo-to-dpo-to-grpo-to-multi-agent-rl-1c00e1dacba7)
- [LLM Architecture 2026: Key Changes and Deployment Insights](https://sesamedisk.com/llm-architecture-gallery-2026/)
