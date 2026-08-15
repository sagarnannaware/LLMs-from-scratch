# 11 — Projects & Path to Mastery

*Reading gets you to "I follow this." Building gets you to "I can do this."*

---

## 11.1 The project ladder

Ordered by difficulty. Each is something you can show someone. **Do at least four.**

### 🟢 Level 1 — Understand the machine

**P1. Attention by hand, then in code.**
Take three tokens with 2-dimensional embeddings. Compute the full causal attention output on paper — scores, scaling, masking, softmax, weighted sum. Then reproduce it in PyTorch and confirm your arithmetic matched. *(Chapter 04 has a worked example to check against.)*
→ Proves: you understand the mechanism, not the API.

**P2. Blank-file transformer.**
Open an empty file. Implement `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock`, `GPTModel` from memory. Verify: 163,009,536 parameters for the GPT-2 small config.
→ Proves: the architecture is genuinely internalized.

**P3. Tokenizer investigation.**
Implement BPE training from scratch. Then compare a GPT-2 tokenizer with a modern one (Llama 3, Qwen) on English prose, code, numbers, a non-Latin script, and emoji. Measure tokens-per-word for each and explain the differences.
→ Proves: you understand why tokenization drives cost and multilingual quality.

**P4. Decoding lab.**
Sweep temperature × top-k × top-p on one prompt. Build a grid of outputs. Identify where it becomes incoherent and where it becomes repetitive, and explain why top-p behaves differently from top-k on confident vs uncertain predictions.
→ Proves: you can tune generation for a real application.

### 🟡 Level 2 — Train things

**P5. Train a small LLM on data you care about.**
Your notes, a codebase, a domain corpus. Include warmup, cosine decay, gradient clipping. Plot the loss curve and interpret it honestly — including the overfitting you'll see on a small corpus.
→ Proves: end-to-end pretraining, including the parts that go wrong.

**P6. Deliberately break things.**
Remove the causal mask (watch loss collapse — the model is cheating). Remove residual connections (watch early-layer gradients vanish). Set the learning rate 100× too high (watch divergence). **Predict each outcome before running it.**
→ Proves: you can diagnose, which is what separates practitioners from tutorial-followers.

**P7. Instruction-tune a model.**
Write or synthesize 200+ instruction examples for a domain you know. Finetune. Evaluate with an LLM judge. Then ablate: with/without prompt masking, LoRA vs full finetuning, 50 vs 200 examples.
→ Proves: the complete applied-finetuning workflow, including evaluation.

**P8. LoRA rank study.**
Finetune the same task at rank ∈ {1, 2, 4, 8, 16, 32, 64}. Plot accuracy and trained-parameter count against rank. Find the knee.
→ Proves: you make PEFT decisions from evidence rather than defaults.

### 🟠 Level 3 — Modernize

**P9. Convert GPT-2 into a modern architecture.**
LayerNorm → RMSNorm. GELU → SwiGLU. Learned positions → RoPE. MHA → GQA. Benchmark each change on parameters, memory, and loss. Write up which bought the most.
→ Proves: you can read an architecture paper and implement it.

**P10. Implement a KV cache.**
Add it to your generation loop. Measure tokens/second before and after across sequence lengths, and measure cache memory growth. Then implement a sliding-window variant with a fixed budget and measure what quality it costs.
→ Proves: the most important inference optimization, understood from the inside.

**P11. MoE from scratch.**
Replace your FFN with 8 experts and a top-2 router. Log expert utilization during training and **watch routing collapse happen**. Then fix it with load balancing. Compare active vs total parameters against actual throughput and memory.
→ Proves: you understand the architecture behind most frontier models.

**P12. GRPO on a verifiable task.**
Take an instruction-tuned model. Define a task with an automatic verifier — arithmetic, or "emit JSON matching this schema." Sample groups of G responses, compute group-relative advantages, update the policy. Watch the pass rate climb.
→ Proves: you've implemented the algorithm behind modern reasoning models. Very few people who discuss RLVR have done this.

### 🔴 Level 4 — Ship

**P13. Production RAG with a real eval harness.**
Hybrid retrieval (dense + BM25) + reranking over a corpus you care about. Then build a 50-question eval set with known answers and measure retrieval recall@k, answer accuracy, and citation correctness. Ablate each component.
→ **The eval set is the project.** The pipeline is the easy half.

**P14. An agent, then attack it.**
Build a tool-calling loop with 3–5 tools and a step limit. Then hide a prompt injection in a document it retrieves and see what happens. Document which defenses (least privilege, sandboxing, approval gates) actually stop it.
→ Proves: you can build agents *and* reason about their security.

**P15. Serve a model properly.**
Deploy a quantized model with vLLM. Measure TTFT, tokens/second, and throughput across batch sizes and quantization levels. Enable prefix caching and measure the difference on a repeated-system-prompt workload. Produce a cost-per-million-tokens estimate.
→ Proves: you understand production economics, not just model quality.

---

## 11.2 Self-assessment

Be honest. Each gap points at a chapter.

**Fundamentals**
- [ ] Trace a string end to end: text → tokens → embeddings → blocks → logits → sampled token, naming every transformation.
- [ ] Write multi-head causal self-attention from a blank file.
- [ ] Explain `√d_k` scaling mechanically — what actually breaks without it.
- [ ] State every tensor shape through a block for `(batch=4, seq=128, d=768, heads=12)`.
- [ ] Explain why residual connections make depth trainable.
- [ ] Compute parameter count from a config, and memory for training vs inference.
- [ ] Explain why initial loss should be `ln(vocab_size)`.

**Training**
- [ ] Explain cross-entropy and perplexity to a non-technical person.
- [ ] Diagnose overfitting, underfitting, and divergence from loss curves.
- [ ] Explain why warmup matters specifically for Adam.
- [ ] Distinguish pretraining, SFT, and preference tuning by what each fixes.
- [ ] State the Chinchilla result and why models deliberately violate it.

**Modern**
- [ ] Explain RoPE and why relative beats absolute position.
- [ ] Compute the KV cache saving from GQA for a given config.
- [ ] Explain MoE — including the trap that it saves compute but not memory.
- [ ] Explain DPO's relationship to RLHF.
- [ ] Explain RLVR and GRPO, including why GRPO drops the critic.
- [ ] Explain why decoding is memory-bandwidth-bound and what follows.
- [ ] Explain speculative decoding and why its output distribution is unchanged.

**Applied**
- [ ] Choose between prompting, RAG, and finetuning — and justify it.
- [ ] Design an eval set for a task with no answer key.
- [ ] Name three judge biases and their mitigations.
- [ ] Explain why prompt injection can't be fixed with a better system prompt.

---

## 11.3 Papers, in reading order

Read after the corresponding chapter. One line each so you know what you're getting.

**Foundations**
1. *Attention Is All You Need* (2017) — the transformer. Read after chapter 04; it'll be surprisingly readable.
2. *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018) — pretraining + finetuning as a paradigm.
3. *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — the model you built.
4. *Language Models are Few-Shot Learners* (GPT-3, 2020) — scale produces in-context learning.

**Scaling and data**
5. *Scaling Laws for Neural Language Models* (2020).
6. *Training Compute-Optimal LLMs* (Chinchilla, 2022) — 20 tokens per parameter.

**Architecture**
7. *RoFormer* (RoPE, 2021).
8. *GQA* (2023).
9. *RMSNorm* (2019) and *GLU Variants Improve Transformer* (2020) — two short papers behind two modern defaults.
10. *FlashAttention* (2022) — the best introduction to GPU-memory thinking there is.
11. *Mixtral of Experts* (2024) — a clear, concrete MoE description.
12. *Mamba* (2023) — selective state space models.

**Alignment and reasoning**
13. *Training language models to follow instructions with human feedback* (InstructGPT, 2022) — the RLHF recipe.
14. *Direct Preference Optimization* (2023).
15. *Chain-of-Thought Prompting* (2022).
16. *DeepSeekMath* (2024) — where GRPO is introduced.
17. *DeepSeek-R1* (2025) — RLVR producing emergent reasoning. The most consequential post-training paper of recent years.

**Efficiency**
18. *LoRA* (2021).
19. *QLoRA* (2023).
20. *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023).

**How to read a paper:** abstract → figures → conclusion → introduction → method. Skip related work on the first pass. If you can't write the core idea as pseudocode afterward, you skimmed it.

---

## 11.4 Libraries, in the order to learn them

| When | Library | For |
|---|---|---|
| Now | **PyTorch** | Everything. Non-negotiable. |
| Now | **tiktoken** | Tokenization |
| After ch 06 | **Hugging Face `transformers`** | Loading and running any open model |
| After ch 06 | **`datasets`** | Data at scale |
| After ch 07 | **`peft`** | LoRA/QLoRA properly |
| After ch 07 | **`trl`** | SFT, DPO, GRPO trainers |
| After ch 08 | **Ollama / llama.cpp** | Local inference |
| For projects | **vLLM** / **SGLang** | Production serving |
| For projects | **`accelerate`** / DeepSpeed | Multi-GPU training |
| For RAG | **FAISS**, **pgvector**, **Chroma** | Vector search |
| For evals | **`lm-eval-harness`** or your own | Benchmarking |

**Deliberate advice: learn `transformers` *after* building a model yourself.** In that order, the library reads as familiar components with configuration knobs. In the other order, it reads as magic — and that's the difference between someone who can debug a model and someone who can only call one.

---

## 11.5 Staying current without drowning

The volume is unmanageable if you try to read everything.

**Weekly (30 minutes):**
- [Sebastian Raschka's *Ahead of AI*](https://magazine.sebastianraschka.com/) — architecture comparisons and paper roundups, pitched exactly at someone who has done this guide.
- Model cards and release notes from major labs. Read the *architecture and training* sections; skip the benchmark tables.

**Monthly:**
- One paper read properly, end to end, rather than ten abstracts. Reimplement one component.
- Skim release notes for vLLM / PyTorch / your inference stack. Systems progress is where the unglamorous practical wins hide.

**Ignore:** leaderboard shuffles, "X kills Y" takes, and anything that doesn't explain *why* something works.

**The filter that keeps you sane** — for every new thing, ask which box of the pipeline it improves:

```
better input representation?     (tokenizers, embeddings, position)
better mixing across positions?  (attention variants, SSMs, hybrids)
better weight fitting?           (optimizers, data, RLHF/RLVR)
cheaper execution?               (quantization, caching, MoE, speculation)
```

If you can't place it in one of those four, it's probably marketing.

---

## 11.6 Where to go next, by what you enjoyed

| If you liked… | Head toward… |
|---|---|
| Chapters 04–05 (architecture) | Modern architectures, MoE, efficient attention, reimplementing papers |
| Chapter 06 (training) | Distributed training, systems, optimizers, GPU kernels |
| Chapter 07 (post-training) | Applied alignment: SFT, DPO, RLVR, domain adaptation |
| Chapter 08 (inference) | Serving, quantization, performance engineering |
| Chapter 09 (applications) | Product engineering, agents, RAG, context engineering |
| Chapter 10 (evaluation) | Evals and applied ML — undervalued and in serious demand |
| Wondering *why* a model does something | Mechanistic interpretability and alignment research |

---

## 11.7 A closing note

The field produces more content per week than anyone can read. The people who stay competent aren't the ones who read the most — they're the ones with the strongest fundamentals, because fundamentals let you compress each new paper into "oh, that's attention with a different mask" instead of memorizing another acronym.

You now have those fundamentals. The remaining gap between reading about LLMs and building them is smaller than it looks, and it closes only one way.

Go build something.

---

**Back to:** [Guide index](README.md) · [Glossary](glossary.md) · [Notebooks](notebooks/)
