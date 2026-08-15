# Projects, Self-Assessment & Resources

Reading and running notebooks gets you to "I follow this." Projects get you to "I can build this." Here's the bridge.

---

## Part 1 — Milestone projects

Ordered by difficulty. Each one is a thing you can show someone. Do at least three.

### 🟢 Level 1 — Consolidation (after ch05)

**P1. Blank-file GPT.**
Open an empty file. Implement `MultiHeadAttention`, `LayerNorm`, `GELU`, `FeedForward`, `TransformerBlock` and `GPTModel` from memory. Load OpenAI's GPT-2 weights into it. If it generates coherent English, your implementation is provably correct.
*Skill proven: you actually understand the architecture.*

**P2. Tokenizer investigation.**
Compare `tiktoken`'s GPT-2 tokenizer against a modern tokenizer (Llama 3, Qwen) on the same corpus: English prose, code, numbers, your native language, emoji. Measure tokens-per-word for each. Write up why modern vocabularies grew to 128k+.
*Skill proven: you understand why tokenization drives cost and multilingual quality.*

**P3. Decoding lab.**
Sweep temperature × top-k × top-p on a fixed prompt with GPT-2 medium. Build a grid of outputs and a short analysis: where does it become incoherent, where does it become repetitive, and why does top-p behave differently from top-k on confident vs uncertain predictions?
*Skill proven: you can tune generation for a real application.*

### 🟡 Level 2 — Extension (after ch07)

**P4. Train a small LLM on data you care about.**
Use [`ch05/03_bonus_pretraining_on_gutenberg`](../ch05/03_bonus_pretraining_on_gutenberg) as a template, but substitute your own corpus — your notes, a codebase, a domain's documents. Add warmup, cosine decay and gradient clipping from [`appendix-D`](../appendix-D). Plot the loss curve and interpret it honestly.
*Skill proven: end-to-end pretraining, including the parts that go wrong.*

**P5. Domain instruction-tuner.**
Write (or synthesize via [`ch07/05_dataset-generation`](../ch07/05_dataset-generation)) 200+ instruction examples for a domain you know. Finetune. Evaluate with the Ollama judge from [`ch07/03_model-evaluation`](../ch07/03_model-evaluation). Then run three ablations: with/without instruction masking, LoRA vs full finetuning, 50 vs 200 examples.
*Skill proven: the complete applied-finetuning workflow, including evaluation.*

**P6. LoRA rank study.**
Take the ch06 spam classifier. Finetune with LoRA at r ∈ {1, 2, 4, 8, 16, 32, 64}. Plot accuracy and trained-parameter count against rank. Find the knee.
*Skill proven: you can make PEFT decisions from evidence rather than defaults.*

### 🟠 Level 3 — Modernization (after the landscape doc)

**P7. Modernize your GPT.**
Following [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama), convert your model: LayerNorm → RMSNorm, GELU → SwiGLU, learned positions → RoPE, MHA → GQA. Benchmark each change on parameters, memory and loss. Write up which changes bought the most.
*Skill proven: you can read a modern architecture paper and implement it.*

**P8. Implement a KV cache.**
Add KV caching to your generation loop. Measure tokens/second before and after across sequence lengths, and measure the cache's memory growth. Then implement a sliding-window variant with a fixed cache budget and observe what quality it costs.
*Skill proven: the single most important inference optimization, understood from the inside.*

**P9. MoE from scratch.**
Replace your FFN with 8 experts and a top-2 router. Track expert utilization across training and watch routing collapse happen. Then fix it with load balancing. Compare active vs total parameters and actual throughput.
*Skill proven: you understand the architecture behind most frontier models.*

**P10. GRPO on a toy verifiable task.**
Take your instruction-tuned model. Define a task with an automatic verifier — arithmetic, or "output valid JSON matching this schema." Sample groups of G responses, compute group-relative advantages, update. Watch the pass rate climb.
*Skill proven: you've implemented the algorithm behind modern reasoning models. Very few people who talk about RLVR have done this.*

### 🔴 Level 4 — Systems and applications

**P11. Production RAG with a real eval harness.**
Build hybrid retrieval (dense + BM25) + reranking over a corpus you care about. Then build a 50-question eval set with known answers and measure: retrieval recall@k, answer accuracy, citation correctness. Ablate each component. **The eval set is the project** — the pipeline is the easy half.

**P12. An agent with real tools.**
Build a loop: model → tool call → observation → repeat. Give it 3–5 tools. Then deliberately attack it with prompt injection hidden in a document it retrieves, and document what defenses (sandboxing, least privilege, approval gates) actually stop it.

**P13. Serve a model properly.**
Deploy a quantized model with vLLM. Measure TTFT, tokens/second and throughput across batch sizes and quantization levels. Turn on prefix caching and measure the difference on a repeated-system-prompt workload. Produce a cost-per-million-tokens estimate.

---

## Part 2 — Self-assessment

Be honest. Anything you can't answer points at a chapter to revisit.

### Fundamentals
- [ ] Explain the full path from a string to a sampled token, naming every transformation.
- [ ] Write multi-head causal self-attention from a blank file.
- [ ] Explain why `√d_k` scaling exists — the actual mechanism, not "for stability."
- [ ] State every tensor shape through a transformer block for `(batch=4, seq=128, d=768, heads=12)`.
- [ ] Explain why residual connections make depth trainable.
- [ ] Compute a model's parameter count from its config, and its fp32/bf16 memory footprint.
- [ ] Explain why the initial loss should be ≈ `ln(vocab_size)`.

### Training
- [ ] Explain cross-entropy loss and perplexity to a non-technical person.
- [ ] Diagnose overfitting, underfitting and divergence from loss curves.
- [ ] Explain why warmup matters specifically for Adam.
- [ ] Explain the difference between pretraining, SFT, and preference tuning, and what each fixes.
- [ ] Explain the Chinchilla result and why models are deliberately overtrained past it.

### Modern
- [ ] Explain RoPE and why relative position beats absolute.
- [ ] Explain GQA and compute the KV-cache saving for a given config.
- [ ] Explain MoE, and the trap that it saves compute but not memory.
- [ ] Explain DPO's relationship to RLHF.
- [ ] Explain RLVR and GRPO, including why GRPO drops the critic.
- [ ] Explain why decoding is memory-bandwidth-bound and what follows from that.
- [ ] Explain speculative decoding and why its output distribution is unchanged.

### Applied
- [ ] Choose between prompting, RAG, and finetuning for a given problem — and justify it.
- [ ] Design an eval set for a task with no correct-answer key.
- [ ] Name three LLM-as-judge biases and their mitigations.
- [ ] Explain prompt injection and why it isn't fixable with a better system prompt.

---

## Part 3 — The papers worth reading

Read in this order, after the corresponding chapter. One-line summaries so you know what you're getting.

**Foundations**
1. *Attention Is All You Need* (2017) — the transformer. Read after ch03; it will be surprisingly readable.
2. *Improving Language Understanding by Generative Pre-Training* (GPT-1, 2018) — pretraining + finetuning as a paradigm.
3. *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019) — the model you built.
4. *Language Models are Few-Shot Learners* (GPT-3, 2020) — scale produces in-context learning.

**Scaling and data**
5. *Scaling Laws for Neural Language Models* (2020) — loss as a power law.
6. *Training Compute-Optimal LLMs* (Chinchilla, 2022) — 20 tokens per parameter.

**Architecture**
7. *RoFormer* (RoPE, 2021) — rotary positions.
8. *GQA* (2023) — grouped-query attention.
9. *Root Mean Square Layer Normalization* (2019) and *GLU Variants Improve Transformer* (2020) — the two-page papers behind RMSNorm and SwiGLU.
10. *FlashAttention* (2022) — IO-aware exact attention. The best introduction to GPU-memory thinking.
11. *Mixtral of Experts* (2024) — a clear, concrete MoE description.
12. *Mamba* (2023) — selective state space models.

**Alignment and reasoning**
13. *Training language models to follow instructions with human feedback* (InstructGPT, 2022) — the RLHF recipe.
14. *Direct Preference Optimization* (2023) — you implement this in ch07's bonus.
15. *Chain-of-Thought Prompting* (2022) — reasoning by prompting.
16. *DeepSeekMath* (2024) — where GRPO is introduced.
17. *DeepSeek-R1* (2025) — RLVR producing emergent reasoning. The most consequential post-training paper of recent years.

**Efficiency**
18. *LoRA* (2021) — you implement this in appendix E.
19. *QLoRA* (2023) — 4-bit base + LoRA.
20. *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023) — how serving actually works.

**Method for reading a paper:** abstract → figures → conclusion → introduction → method. Skip related work on the first pass. If you can't reimplement the core idea in pseudocode afterward, you skimmed it.

---

## Part 4 — Libraries to learn, in order

| Stage | Library | What it's for |
|---|---|---|
| Now | **PyTorch** | Everything. Non-negotiable. |
| Now | **tiktoken** | Tokenization (used in this repo) |
| After ch05 | **Hugging Face `transformers`** | Loading and running any open model |
| After ch05 | **`datasets`** | Data loading and processing at scale |
| After ch06 | **`peft`** | LoRA/QLoRA the standard way |
| After ch07 | **`trl`** | SFT, DPO, GRPO trainers |
| After ch07 | **Ollama / llama.cpp** | Local inference (already used in ch07) |
| For projects | **vLLM** or **SGLang** | Production serving |
| For projects | **`accelerate`** / **DeepSpeed** | Multi-GPU training |
| For RAG | **FAISS**, **pgvector**, or **Chroma** | Vector search |
| For evals | **`lm-eval-harness`**, or a homemade harness | Benchmarking |

Deliberate advice: **learn `transformers` after you've built the model yourself**, not before. Doing it in that order, the library reads as a set of familiar components with configuration knobs. Doing it the other way round, it reads as magic — and that's the difference between someone who can debug a model and someone who can only call one.

---

## Part 5 — Beyond this repo

**Sebastian Raschka's other work** (same author, same pedagogy):
- *Ahead of AI* newsletter — architecture comparisons and paper roundups.
- His subsequent books and repos on reasoning models and post-training pick up exactly where this one ends. If this book worked for you, those will too.

**Complementary resources:**
- Andrej Karpathy's *Zero to Hero* series and `nanoGPT` — another from-scratch path; different angle on the same material, excellent for reinforcing ch03–ch05.
- *The Illustrated Transformer* (Jay Alammar) — the best visual explanation of attention. Read alongside ch03.
- Hugging Face's free courses (NLP, LLM, RL) — practical, library-oriented follow-on.
- The *Ultra-Scale Playbook* and similar distributed-training guides — for when you outgrow one GPU.

**Where to go next, by interest:**

| If you enjoy… | Go toward… |
|---|---|
| The architecture chapters | Modern architectures, MoE, efficient attention, and research reimplementation |
| The training loop | Distributed training, systems, optimizer research, kernels |
| ch06/ch07 finetuning | Applied post-training: SFT, DPO, RLVR, domain adaptation |
| The evaluation notebook | Evals and applied ML — undervalued and in serious demand |
| Building things with the model | Agents, RAG, product engineering, context engineering |
| Wondering *why* the model does something | Mechanistic interpretability and alignment research |

---

## Part 6 — A final word on how to learn this

The field produces more content per week than anyone can read. The people who stay competent are not the ones who read the most — they're the ones with the strongest fundamentals, because fundamentals are what let you compress a new paper into "oh, that's attention with a different mask" instead of memorizing another acronym.

You picked a repo that teaches fundamentals by making you type them. That was the right call. Finish it properly — typing, breaking, rebuilding — and the rest of the field becomes reading rather than studying.

Then build something. The gap between people who have read about LLMs and people who have trained one is enormous, and you're four weekends from the right side of it.
