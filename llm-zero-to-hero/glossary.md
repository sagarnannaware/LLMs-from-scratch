# The LLM Glossary

Every term you'll meet, grouped by theme so related concepts sit next to each other. Terms implemented in this repository are marked 📁 with a file reference.

**Sections:** [Foundations](#1-foundations) · [Text & Tokenization](#2-text--tokenization) · [Embeddings & Position](#3-embeddings--position) · [Attention](#4-attention) · [Architecture](#5-model-architecture) · [Training](#6-training-mechanics) · [Optimization](#7-optimization) · [Scaling & Data](#8-scaling--data) · [Finetuning & Alignment](#9-finetuning--alignment) · [Reasoning](#10-reasoning--test-time-compute) · [Inference & Decoding](#11-inference--decoding) · [Efficiency & Serving](#12-efficiency--serving) · [Evaluation](#13-evaluation) · [RAG](#14-retrieval--rag) · [Agents](#15-agents--tool-use) · [Multimodality](#16-multimodality) · [Safety](#17-safety-security--interpretability) · [Systems](#18-hardware--systems)

---

## 1. Foundations

**LLM (Large Language Model)** — A neural network with many parameters trained on large text corpora to predict tokens. "Large" is relative and keeps moving; the 124M model in this repo was large in 2019.

**Transformer** — The architecture (Vaswani et al., 2017, *Attention Is All You Need*) built from self-attention and feed-forward layers. Replaced RNNs because it processes all positions in parallel during training.

**Decoder-only** — Architecture using only the transformer's decoder stack with causal masking. All GPT-family models. 📁 [`ch04/01_main-chapter-code/gpt.py`](../ch04/01_main-chapter-code/gpt.py)

**Encoder-only** — BERT-style, bidirectional (each token sees both directions). Good for classification and embeddings, cannot generate autoregressively.

**Encoder–decoder (seq2seq)** — Original transformer; encoder reads the input, decoder generates the output attending to both. T5, translation models.

**Autoregressive** — Generating one token at a time, each conditioned on all previous tokens. The core generative loop.

**Next-token prediction** — The training objective: given tokens 1…t, predict token t+1. Everything an LLM can do emerges from this.

**Self-supervised learning** — Labels derived from the data itself (here: the shifted text), so no human annotation is needed.

**Pretraining** — The expensive first stage: next-token prediction over a massive corpus. Produces a base model. 📁 [`ch05`](../ch05)

**Base model / foundation model** — The pretrained model before instruction tuning. Completes text; doesn't converse or refuse.

**Finetuning** — Further training on a smaller, targeted dataset. 📁 [`ch06`](../ch06), [`ch07`](../ch07)

**Parameters / weights** — The learned numbers. "7B model" = 7 billion of them.

**Inference** — Running a trained model to produce output, as opposed to training it.

**In-context learning** — The model adapting its behavior from examples in the prompt alone, with no weight updates.

**Zero-shot / few-shot / one-shot** — Prompting with zero, a few, or one demonstration example.

**Emergent ability** — A capability that appears only above a certain scale. Contested — some apparent emergence is an artifact of discontinuous metrics.

**Prompt** — The input text conditioning generation.

**Prompt engineering** — Crafting prompts for better outputs.

**Context engineering** — The 2025–26 successor term: deliberately managing *everything* in the context window — retrieved documents, tool results, memory, conversation history — as a budgeted resource, not just wording an instruction.

---

## 2. Text & Tokenization

**Token** — The atomic unit a model consumes. Typically a subword. ~0.75 English words on average.

**Token ID** — The integer index of a token in the vocabulary.

**Vocabulary** — The full token set. GPT-2: 50,257. Modern models: 128k–256k, mainly for better multilingual and code efficiency.

**Tokenizer** — The text ↔ token IDs converter. 📁 [`ch02`](../ch02)

**BPE (Byte Pair Encoding)** — Iteratively merges the most frequent adjacent symbol pair into a new token. GPT's method. 📁 [`ch02/02_bonus_bytepair-encoder`](../ch02/02_bonus_bytepair-encoder)

**Byte-level BPE** — BPE over raw bytes, so any Unicode input is representable; no out-of-vocabulary tokens ever.

**WordPiece / SentencePiece / Unigram** — Alternative subword algorithms (BERT, Llama, T5 respectively). SentencePiece treats the input as a raw stream, whitespace included.

**tiktoken** — OpenAI's fast BPE library, used throughout this repo.

**Special tokens** — Non-text control tokens: `<|endoftext|>`, `<|im_start|>`, BOS/EOS, padding, tool-call delimiters.

**BOS / EOS** — Beginning/end-of-sequence markers. EOS is how a model signals it's finished.

**Padding token** — Filler to make sequences in a batch equal length; masked out of the loss. 📁 [`ch07`](../ch07) collate function

**Truncation** — Cutting sequences that exceed the context length.

**Detokenization** — Turning IDs back into text.

**Tokenizer-free / byte-level models** — Research direction (e.g. Byte Latent Transformer) operating on raw bytes with learned dynamic patching, removing tokenizer artifacts entirely.

**Token count / cost** — API pricing and context limits are measured in tokens, so tokenizer efficiency has direct financial consequences.

---

## 3. Embeddings & Position

**Embedding** — A dense learned vector representing a token. 📁 [`ch02`](../ch02)

**Embedding matrix** — `(vocab_size, emb_dim)` lookup table; row *i* is token *i*'s vector.

**Embedding dimension (`d_model`, `emb_dim`)** — Vector width. 768 for GPT-2 small. The model's "width."

**Positional encoding** — Information about token order, which attention otherwise lacks entirely.

**Absolute positional embedding** — A learned vector per position, added to token embeddings. GPT-2's approach. 📁 [`ch02`](../ch02)

**Sinusoidal positional encoding** — Fixed sine/cosine functions from the original transformer paper.

**RoPE (Rotary Position Embedding)** — Rotates query and key vectors by an angle proportional to position, so their dot product depends on *relative* distance. Now the standard. 📁 [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama)

**RoPE scaling (NTK-aware, YaRN, linear interpolation)** — Techniques to extend a RoPE model's usable context beyond its training length by adjusting rotation frequencies.

**ALiBi** — Adds a distance-proportional penalty to attention scores instead of embedding positions.

**NoPE** — Findings that decoder-only models can encode position implicitly through causal masking alone.

**Sentence/text embeddings** — Fixed-size vectors representing whole texts for search and clustering. Different use case from token embeddings; usually from encoder models.

**Cosine similarity** — The standard similarity measure between embeddings.

---

## 4. Attention

**Attention** — A mechanism letting each position build a weighted combination of information from other positions, with the weights computed from content. 📁 [`ch03`](../ch03)

**Self-attention** — Attention where queries, keys and values all come from the same sequence.

**Cross-attention** — Queries from one sequence, keys/values from another. Used in encoder–decoder models and many multimodal architectures.

**Query (Q), Key (K), Value (V)** — Three learned projections of each token: what I'm looking for, what I offer, what I contribute.

**Attention score** — Raw `Q·K` dot product between two positions.

**Attention weight** — Score after scaling, masking and softmax; nonnegative, sums to 1 across attended positions.

**Scaled dot-product attention** — `softmax(QKᵀ/√d_k)V`. The `√d_k` division keeps score variance stable so softmax doesn't saturate.

**Context vector** — Attention's output for a position: a weighted blend of value vectors.

**Causal / masked attention** — Future positions masked with `-inf` before softmax so the model can't see ahead. 📁 [`ch03`](../ch03)

**Causal mask** — The upper-triangular boolean mask implementing that constraint.

**Attention map** — The `(seq, seq)` weight matrix; visualizing it shows what attends to what.

**Multi-head attention (MHA)** — `h` parallel attention operations on split projections, concatenated and projected. Different heads capture different relationship types. 📁 [`ch03`](../ch03)

**Head dimension** — `d_out // num_heads`. 64 is the near-universal choice.

**Output projection** — The linear layer mixing concatenated head outputs.

**MQA (Multi-Query Attention)** — All heads share a single K and V head. Shrinks the KV cache dramatically; some quality cost.

**GQA (Grouped-Query Attention)** — Middle ground: groups of query heads share K/V heads. The current default (Llama 3, Qwen, Mistral). 📁 [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama)

**MLA (Multi-head Latent Attention)** — DeepSeek's approach: compress K/V into a low-rank latent vector, decompress on use. Much smaller cache than GQA at comparable quality.

**FlashAttention** — An IO-aware exact attention kernel that tiles the computation in fast SRAM instead of materializing the full `(seq, seq)` matrix in HBM. Same math, far less memory traffic. 📁 benchmarked in [`ch03/02_bonus_efficient-multihead-attention`](../ch03/02_bonus_efficient-multihead-attention)

**Sliding-window / local attention** — Each token attends only to the last *w* tokens, making cost linear in sequence length.

**Sparse attention** — Attending to a structured subset of positions (strided, block, global tokens).

**Attention sink** — The empirical fact that models dump excess attention mass onto the first token(s); exploited by StreamingLLM and explicitly designed for in some recent models.

**QK normalization** — Normalizing queries and keys before the dot product; stabilizes very large-scale training.

**Linear attention** — Reformulations avoiding the explicit `(seq, seq)` matrix, giving O(n) cost and a fixed-size recurrent state. Gated DeltaNet and related variants are the current strong form.

**SSM (State Space Model) / Mamba** — Recurrent-style sequence models with input-dependent state transitions; constant memory per token, no KV cache. Increasingly used in hybrid stacks.

**Hybrid architecture** — Interleaving a few full-attention layers with many linear-attention/SSM layers to get long-context efficiency while retaining recall. A dominant 2026 pattern.

---

## 5. Model Architecture

**Transformer block / layer** — Attention sublayer + feed-forward sublayer, each wrapped in normalization and a residual connection. Stacked N times. 📁 [`ch04`](../ch04)

**Depth (`n_layers`)** vs **width (`emb_dim`)** — The two main size dials.

**Feed-forward network (FFN/MLP)** — Position-wise two-layer network, typically expanding 4× (768 → 3072 → 768). Holds most parameters; where factual knowledge is thought to be concentrated.

**Expansion factor** — The FFN's hidden-to-model width ratio.

**Activation function** — Elementwise nonlinearity.

**ReLU / GELU / SiLU (Swish)** — Progressively smoother activations. GPT-2 uses GELU. 📁 [`ch04`](../ch04)

**GLU / SwiGLU** — Gated linear units: one branch gates another elementwise. SwiGLU (SiLU-gated) is the modern FFN standard; uses three matrices instead of two, usually with hidden width ≈ 8/3× rather than 4×.

**Layer normalization** — Normalizes each token's features to mean 0, variance 1, then applies learned scale and shift. 📁 [`ch04`](../ch04)

**RMSNorm** — LayerNorm without mean subtraction or bias — just scale by root-mean-square. Cheaper, equally effective, now standard. 📁 [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama)

**Pre-LN vs Post-LN** — Normalizing before vs after each sublayer. Pre-LN trains far more stably at depth and is near-universal.

**Residual / skip / shortcut connection** — `x + sublayer(x)`. Makes deep stacks trainable by giving gradients a direct path. 📁 [`ch04`](../ch04)

**Residual stream** — The mental model of the model as a shared bus that each layer reads from and writes to. Central to interpretability work.

**Dropout** — Randomly zeroing activations during training as regularization. GPT-2 uses 0.1; modern large-scale pretraining usually uses 0.

**Logits** — Raw pre-softmax scores over the vocabulary, one per token in the vocabulary.

**LM head / output head** — The final linear layer mapping hidden state → logits.

**Weight tying** — Sharing the embedding matrix with the output head. Saves 38.6M parameters in GPT-2 small and is why "163M" is marketed as "124M".

**Bias terms** — Additive constants in linear layers. Modern models mostly drop them.

**MoE (Mixture of Experts)** — Replace the FFN with many expert FFNs plus a router that activates only a few per token. Total parameters grow while per-token compute stays flat. The dominant frontier scaling pattern in 2026.

**Router / gating network** — Picks which experts handle each token (usually top-k, k=1–8).

**Active vs total parameters** — For MoE: e.g. "35B total, 3B active." Quality tracks total; speed tracks active; *memory tracks total*.

**Shared expert** — An expert every token always uses, capturing common patterns so routed experts can specialize.

**Load balancing / auxiliary loss / aux-loss-free balancing** — Preventing the router from collapsing onto a few experts; newer approaches use bias adjustment instead of an extra loss term.

**Expert parallelism** — Distributing experts across devices; the communication pattern is MoE's main systems challenge.

**Dense model** — Non-MoE: every parameter used for every token.

**Context length / context window** — Maximum tokens processable at once. 1024 here; 128k–1M+ in frontier models.

**Effective context** — The length over which a model actually retains usable information, typically well below its advertised maximum.

**Depth-wise scaling / model surgery** — Building new models by pruning, expanding, or merging existing ones.

---

## 6. Training Mechanics

**Forward pass** — Input → output computation.

**Backward pass / backpropagation** — Chain-rule computation of every parameter's gradient.

**Gradient** — Direction and magnitude of a parameter's effect on the loss.

**Loss function** — The scalar being minimized.

**Cross-entropy loss** — Negative log-probability of the correct token, averaged. The LLM objective. 📁 [`ch05`](../ch05)

**Perplexity** — `exp(loss)`. "Effective number of choices" the model is deciding among.

**Logit vs probability** — Logits are unnormalized; softmax makes them a distribution.

**Softmax** — Exponentiate and normalize into a probability distribution.

**Batch / mini-batch / batch size** — Number of sequences processed per update.

**Global batch size** — Batch size × gradient accumulation × data-parallel replicas. The number that actually matters for training dynamics.

**Gradient accumulation** — Summing gradients over several micro-batches before stepping, to simulate a large batch on small memory.

**Epoch / step / iteration** — Full pass over the dataset / one optimizer update / one batch.

**Train / validation / test split** — Fit on train, tune on validation, report on test. Once.

**Overfitting** — Memorizing training data; training loss falls while validation loss rises.

**Underfitting** — Insufficient capacity or training; both losses stay high.

**Regularization** — Anything that combats overfitting: dropout, weight decay, data augmentation, early stopping.

**Teacher forcing** — Feeding ground-truth previous tokens during training rather than the model's own predictions. What the shifted-target dataloader implements.

**Exposure bias** — The train/inference mismatch teacher forcing creates: at inference the model consumes its own (possibly wrong) outputs.

**Checkpoint** — Saved model (and optimizer) state.

**`state_dict`** — PyTorch's parameter dictionary.

**Mixed precision (fp16 / bf16)** — Training in 16-bit for speed and memory with selected fp32 accumulations. bf16 has fp32's exponent range and is the safe default.

**FP8 training** — 8-bit training for parts of the forward/backward pass; used at frontier scale with careful scaling.

**Numerical stability** — Avoiding overflow/underflow; the reason for `eps`, log-sum-exp tricks, and loss scaling.

**Loss spike / divergence** — Sudden loss explosion during large runs. Handled by clipping, LR reduction, or rollback to an earlier checkpoint.

**Curriculum learning** — Ordering training data from easy to hard, or shifting the data mixture over training.

**Annealing / mid-training** — A late pretraining phase on higher-quality data at decayed learning rate. Standard practice and a large quality lever.

---

## 7. Optimization

**Optimizer** — The update rule turning gradients into weight changes.

**SGD** — Plain gradient descent, optionally with momentum.

**Adam / AdamW** — Adaptive per-parameter learning rates using first and second gradient moments. AdamW decouples weight decay from the gradient and is the LLM default. Costs 2 extra states per parameter — a major share of training memory.

**Muon and other matrix-aware optimizers** — Newer optimizers that orthogonalize/precondition weight-matrix updates, reported to improve loss-per-token materially over AdamW; increasingly used in 2025–26 runs.

**Learning rate** — Step size. The most important hyperparameter, by a wide margin.

**LR schedule** — How the learning rate changes over training. 📁 [`ch05/04_learning_rate_schedulers`](../ch05/04_learning_rate_schedulers)

**Warmup** — Ramping the LR up from ~0 over the first steps so early, unreliable Adam updates can't destabilize training. 📁 [`appendix-D`](../appendix-D)

**Cosine decay / linear decay / WSD** — Ways to anneal the LR. WSD (warmup–stable–decay) allows continuing training from the stable phase without restarting the schedule.

**Weight decay** — L2-style penalty pulling weights toward zero.

**Gradient clipping** — Rescaling gradients whose norm exceeds a threshold (usually 1.0). 📁 [`appendix-D`](../appendix-D)

**Momentum** — Accumulating past gradient direction to smooth updates.

**Hyperparameter** — Anything you set rather than learn. 📁 [`ch05/05_bonus_hparam_tuning`](../ch05/05_bonus_hparam_tuning)

**Initialization** — Starting weight distribution; poor choices prevent training entirely.

**μP (maximal update parameterization)** — A parameterization that lets hyperparameters tuned on small models transfer to large ones, saving enormous search cost.

---

## 8. Scaling & Data

**Scaling laws** — Empirical power-law relations between loss and compute/parameters/data. Predict large-model performance from small-model runs.

**Chinchilla-optimal** — Compute-optimal training uses ≈20 tokens per parameter. Modern models deliberately *overtrain* far beyond this (100–1000+ tokens/param) because inference cost dominates over a model's lifetime.

**Compute budget / FLOPs** — Training compute ≈ `6 × params × tokens`. Useful back-of-envelope arithmetic. 📁 [`ch04/02_performance-analysis`](../ch04/02_performance-analysis)

**Data mixture** — Proportions of web text, code, math, books, multilingual data. A primary quality lever, and usually the most closely guarded detail of a frontier model.

**Deduplication** — Removing repeated documents; improves quality and reduces memorization.

**Data quality filtering** — Classifier- or heuristic-based selection of high-quality documents. Often worth more than extra parameters.

**Synthetic data** — Model-generated training data, filtered for quality. Now a large fraction of post-training data and a growing share of pretraining. 📁 [`ch07/05_dataset-generation`](../ch07/05_dataset-generation)

**Data contamination** — Benchmark test items leaking into training data, inflating scores.

**Memorization** — Verbatim reproduction of training data; a privacy and copyright concern.

**Tokens seen** — Total training tokens; the standard measure of training progress.

**Continued pretraining** — Further pretraining a released base model on domain data (medical, legal, a new language).

**Model collapse** — Degradation from training repeatedly on unfiltered model-generated data.

---

## 9. Finetuning & Alignment

**Transfer learning** — Reusing pretrained representations for a new task. 📁 [`ch06`](../ch06)

**SFT (Supervised Finetuning)** — Training on curated input→output pairs. 📁 [`ch07`](../ch07)

**Instruction tuning** — SFT on instruction/response data to make a model follow directions. 📁 [`ch07`](../ch07)

**Chat template** — The role-markup format (`system`/`user`/`assistant`) a model was trained with. Mismatching it at inference degrades quality badly.

**System prompt** — Instructions setting persona, constraints and tools, prepended to the conversation.

**Freezing layers** — Holding parameters fixed while training others. 📁 [`ch06/02_bonus_additional-experiments`](../ch06/02_bonus_additional-experiments)

**Catastrophic forgetting** — Losing prior capabilities while finetuning on narrow data.

**PEFT (Parameter-Efficient Finetuning)** — Training a small number of added or selected parameters.

**LoRA** — Learn a low-rank update `ΔW = BA` with frozen `W`. 📁 [`appendix-E`](../appendix-E)

**Rank (r) and alpha** — LoRA's capacity dial and its scaling factor (effective scale `alpha/r`).

**QLoRA** — LoRA on a 4-bit quantized frozen base model. Finetunes large models on a single consumer GPU.

**DoRA** — LoRA variant decomposing updates into magnitude and direction; closer to full finetuning quality.

**Adapter merging** — Folding LoRA weights into the base model for zero inference overhead.

**Alignment** — Making models helpful, harmless and honest.

**RLHF (Reinforcement Learning from Human Feedback)** — Collect preferences → train a reward model → optimize the policy against it with RL (usually PPO) plus a KL penalty to the reference model.

**Reward model (RM)** — A model trained to score responses as humans would.

**PPO (Proximal Policy Optimization)** — The classic RLHF algorithm. Powerful but complex: four models in memory, sensitive to hyperparameters.

**KL penalty** — A term keeping the tuned policy near the reference model, preventing reward hacking and gibberish.

**Reward hacking** — Exploiting flaws in the reward signal instead of doing the intended task.

**DPO (Direct Preference Optimization)** — Optimizes preferences directly with a closed-form loss on chosen/rejected pairs — no reward model, no rollouts. 📁 [`ch07/04_preference-tuning-with-dpo`](../ch07/04_preference-tuning-with-dpo)

**Beta (DPO)** — Controls deviation from the reference model, playing the KL penalty's role.

**IPO / KTO / ORPO / SimPO** — DPO relatives: different loss shapes, unpaired feedback (KTO), or merging SFT and preference stages (ORPO).

**RLAIF / Constitutional AI** — Using model-generated feedback against a written set of principles instead of human labels.

**RLVR (Reinforcement Learning with Verifiable Rewards)** — RL where the reward comes from an automatic checker — unit tests, a math answer, a compiler — rather than a learned reward model. The engine behind modern reasoning models, because verifiable rewards cannot be gamed the way learned ones can.

**GRPO (Group Relative Policy Optimization)** — Samples a group of responses per prompt and uses their mean reward as the baseline, eliminating the value/critic network. Now the standard reasoning-RL algorithm; DAPO, GSPO and similar variants refine its clipping, sampling and sequence-level weighting.

**Rejection sampling / best-of-n finetuning** — Sample many outputs, keep the best by some judge or verifier, finetune on those.

**Distillation** — Training a small model on a large model's outputs (or logits). How most small strong models are made.

**Model merging** — Averaging or interpolating weights of models finetuned from a shared base to combine capabilities without retraining.

**Sycophancy** — Agreeing with the user over being correct; a known failure mode of preference training on human approval.

**Alignment tax** — Capability lost in exchange for safety/helpfulness tuning.

---

## 10. Reasoning & Test-Time Compute

**Chain of thought (CoT)** — Producing intermediate reasoning steps before the answer, which measurably improves accuracy on multi-step problems.

**Reasoning model** — A model post-trained (usually via RLVR) to generate long deliberate reasoning before answering. DeepSeek-R1 made the recipe public; it is now a standard product tier.

**Thinking tokens / reasoning traces** — The (often hidden) intermediate tokens a reasoning model emits.

**Test-time compute scaling** — Spending more inference compute — longer reasoning, more samples — to get better answers. A second scaling axis alongside training compute.

**Thinking budget / budget forcing** — Capping or extending reasoning length, sometimes exposed as a user-facing "effort" setting.

**Self-consistency** — Sample multiple reasoning paths, take the majority answer.

**Best-of-n / verifier-guided search** — Generate n candidates, select with a verifier or reward model.

**PRM (Process Reward Model)** — Scores each reasoning *step* rather than only the final answer.

**ORM (Outcome Reward Model)** — Scores only the final answer.

**Self-reflection / self-correction** — Prompting a model to critique and revise its own output. Works when there's external signal; unreliable when purely self-generated.

**Aha moment** — The observation that RLVR-trained models spontaneously learn to backtrack and re-check without being taught to.

**Overthinking** — Reasoning models burning tokens on trivial questions; a real cost and latency problem.

**Automatic curriculum / self-play environments** — Models generating their own training tasks calibrated to current ability, paired with verifiers. An active 2026 direction.

---

## 11. Inference & Decoding

**Decoding strategy** — How the next token is chosen from the logits. 📁 [`ch05`](../ch05)

**Greedy decoding** — Always take the argmax. Deterministic, repetitive.

**Sampling** — Draw from the probability distribution.

**Temperature** — Divide logits by T before softmax. <1 sharpens, >1 flattens, →0 is greedy.

**Top-k sampling** — Restrict sampling to the k most likely tokens.

**Top-p / nucleus sampling** — Restrict to the smallest set whose cumulative probability exceeds p. Adapts to the model's confidence.

**Min-p sampling** — Threshold relative to the top token's probability; robust at high temperature.

**Repetition / frequency / presence penalty** — Downweight tokens already produced, to break loops.

**Beam search** — Keep the b highest-probability partial sequences. Good for translation; poor for open-ended text (bland, generic).

**Logit bias** — Manually shifting specific tokens' logits.

**Constrained / structured decoding** — Masking logits to enforce a grammar, regex or JSON schema, guaranteeing parseable output.

**Function calling / tool calling** — Model emits a structured call (name + arguments) that your code executes, returning the result into the context.

**Stop sequences** — Strings that halt generation.

**Prefill vs decode** — Prefill processes the whole prompt in parallel (compute-bound); decode emits one token at a time (memory-bandwidth-bound). Their asymmetry drives most serving design.

**TTFT / TPOT / ITL** — Time to first token; time per output token; inter-token latency. The user-facing latency metrics.

**Throughput vs latency** — Tokens/second across all users vs speed for one user. Batching trades one for the other.

**KV cache** — Cached keys and values for previous tokens so each new token needs only its own K/V. Turns generation from O(n²) to O(n) per token — and becomes the dominant memory consumer at long context.

**Prefix caching** — Reusing the KV cache for a shared prompt prefix (system prompt, documents) across requests. Big real-world savings.

**Context window management** — Truncation, summarization, sliding windows, or compaction when the conversation exceeds the limit.

**Speculative decoding** — A small draft model proposes several tokens; the large model verifies them in one pass, accepting the longest correct prefix. Same output distribution, 2–3× faster.

**Medusa / EAGLE / multi-token prediction** — Self-speculative variants where the model predicts several future tokens with extra heads.

**Continuous batching** — Adding and removing sequences from a running batch as they finish, instead of waiting for the slowest. Large throughput gains.

**Paged attention** — Managing the KV cache in fixed-size pages like OS virtual memory, eliminating fragmentation. The core idea in vLLM.

**Disaggregated serving** — Running prefill and decode on separate hardware pools since their bottlenecks differ.

---

## 12. Efficiency & Serving

**Quantization** — Storing/computing weights (and sometimes activations) in fewer bits: fp16 → int8 → int4.

**PTQ vs QAT** — Post-training quantization (fast, no retraining) vs quantization-aware training (better quality at very low bits).

**GPTQ / AWQ / GGUF / bitsandbytes** — Common quantization formats and toolchains. GGUF is llama.cpp's format for CPU/consumer inference.

**Calibration data** — Sample inputs used to fit quantization scales.

**Outlier features** — A few activation channels with huge magnitudes that make naive quantization fail; the reason for mixed-precision schemes.

**Pruning** — Removing weights (unstructured) or whole heads/layers/channels (structured, actually faster in practice).

**Knowledge distillation** — Small student trained to match a large teacher.

**Gradient checkpointing** — Recompute activations in the backward pass instead of storing them: less memory, ~30% more compute.

**Offloading** — Moving weights or optimizer state to CPU RAM or NVMe to fit bigger models.

**ZeRO / FSDP** — Sharding optimizer state, gradients and parameters across data-parallel workers instead of replicating them.

**Data / tensor / pipeline / sequence / context parallelism** — Splitting the batch / individual matrices / layer groups / the sequence dimension across devices. Large runs combine several ("3D parallelism").

**vLLM / SGLang / TensorRT-LLM / llama.cpp / Ollama** — Production inference engines and local runtimes. 📁 Ollama is used for evaluation in [`ch07`](../ch07)

**Batch inference** — Offline processing of many requests, typically at much lower cost than interactive serving.

**Model routing / cascades** — Sending easy queries to a cheap model and hard ones to an expensive one.

**Edge / on-device inference** — Running quantized small models locally for privacy and latency.

---

## 13. Evaluation

**Benchmark** — A standardized test set. MMLU (knowledge), GSM8K/MATH (math), HumanEval/SWE-bench (code), HellaSwag (commonsense), GPQA (hard science), plus agentic and long-context suites.

**Held-out / test set** — Data never used in training or tuning.

**Contamination** — Test data leaking into training; the reason benchmark scores drift from real capability.

**LLM-as-a-judge** — Using a strong model to grade outputs. Scalable; biased toward long, confident, self-similar answers. 📁 [`ch07/03_model-evaluation`](../ch07/03_model-evaluation)

**Position bias / length bias / self-preference bias** — The main judge failure modes; mitigate by randomizing order, controlling length, and using a different judge family than the model under test.

**Pairwise comparison / Elo / arena** — Ranking models by head-to-head human preference.

**Pass@k** — Probability at least one of k samples is correct. Standard for code.

**Rubric-based evaluation** — Scoring against explicit written criteria, increasingly with per-criterion model grading.

**Human evaluation** — Still the gold standard; expensive and noisy.

**Ablation** — Removing one component to measure its contribution. 📁 [`ch06/02_bonus_additional-experiments`](../ch06/02_bonus_additional-experiments)

**Regression testing / evals in CI** — Running a fixed eval suite on every prompt or model change. The single highest-value engineering practice in applied LLM work.

**Observability / tracing** — Logging prompts, tool calls and outputs to debug production behavior.

---

## 14. Retrieval & RAG

**RAG (Retrieval-Augmented Generation)** — Retrieve relevant documents and put them in the context before generating. Grounds answers in current, private data without retraining.

**Chunking** — Splitting documents into retrievable pieces. Chunk size and overlap materially affect quality.

**Vector database** — Store for embeddings with fast nearest-neighbor search (FAISS, Chroma, pgvector, Qdrant, Milvus).

**ANN search / HNSW** — Approximate nearest-neighbor algorithms making vector search fast at scale.

**Dense vs sparse retrieval** — Embedding similarity vs keyword scoring (BM25). Hybrid search combines both and usually beats either.

**Reranker** — A cross-encoder that rescores top candidates by reading query and document jointly. Cheap accuracy win.

**Query rewriting / expansion / HyDE** — Transforming the user's query before retrieval.

**GraphRAG** — Building an entity/relationship graph over the corpus to answer questions requiring synthesis across many documents.

**Agentic RAG** — The model decides when and what to search, iteratively, instead of a fixed retrieve-then-generate pipeline.

**Grounding / citation** — Requiring answers to reference retrieved sources; the main defense against hallucination.

**Long-context vs RAG** — Big contexts don't eliminate RAG: retrieval is cheaper, more current, auditable, and avoids attention dilution over huge inputs.

**Needle-in-a-haystack** — A long-context test that plants a fact in a long document and asks for it. Passing it is necessary, not sufficient.

---

## 15. Agents & Tool Use

**Agent** — An LLM in a loop with tools, deciding actions, observing results, and iterating toward a goal.

**Tool / function calling** — Structured invocation of external code by the model.

**MCP (Model Context Protocol)** — An open standard for connecting models to tools and data sources, so integrations aren't rewritten per framework. Widely adopted since 2025.

**ReAct** — Interleaving reasoning and acting: think → act → observe → repeat.

**Planning / decomposition** — Breaking goals into steps, sometimes with an explicit plan the agent revises.

**Memory (short-term / long-term)** — Conversation context vs persisted knowledge across sessions.

**Context compaction** — Summarizing or pruning history to stay within the window during long agent runs.

**Multi-agent system** — Several specialized agents (planner, coder, reviewer) coordinating; orchestrator–worker is the common shape.

**Human in the loop** — Requiring approval before consequential actions.

**Sandboxing** — Running agent-generated code in an isolated environment.

**Computer use / browser use** — Agents operating GUIs or browsers via screenshots and synthetic input.

**Environment** — The interactive world an agent acts in (filesystem, browser, API); training agents in rich environments with verifiable outcomes is the current frontier of post-training.

**Trajectory** — The full sequence of an agent's steps, used for debugging and as RL training data.

**Guardrails** — Programmatic constraints on inputs, outputs and actions.

---

## 16. Multimodality

**Multimodal model** — Handles more than text: images, audio, video.

**VLM (Vision-Language Model)** — Text + image input.

**Vision encoder / ViT** — Splits an image into patches, embeds them, and runs a transformer.

**Patch / patchify** — Image tiles treated as tokens.

**Projector / connector** — Maps vision-encoder outputs into the LLM's embedding space.

**Early vs late fusion** — Interleaving modality tokens in one stream vs merging at a later stage.

**CLIP** — Contrastively trains image and text encoders into a shared space; the backbone of much multimodal work.

**Any-to-any / omni models** — Native text, image, audio in and out.

**Speech-to-text / text-to-speech** — Whisper-style ASR and neural TTS, increasingly folded into a single model for voice interaction.

**Diffusion model** — Iterative denoising generation; dominant for images, and an active research direction for text (parallel token refinement instead of left-to-right).

**World model** — A learned simulator of an environment's dynamics; a growing research thread beyond pure language.

---

## 17. Safety, Security & Interpretability

**Hallucination** — Fluent, confident, false output. Not a bug to be patched — a consequence of a model trained to always produce plausible continuations, and of training objectives that reward guessing over abstaining.

**Calibration** — Whether stated confidence matches actual accuracy.

**Abstention** — Saying "I don't know." Increasingly trained explicitly with verifiable rewards.

**Jailbreak** — A prompt that circumvents safety training.

**Prompt injection** — Malicious instructions hidden in *data* the model reads (a web page, a document, a tool result) that hijack its behavior. The central security problem for agents, because the model cannot reliably distinguish instructions from content.

**Indirect prompt injection** — Injection delivered through content the model retrieves rather than through user input.

**Data exfiltration** — Tricking an agent into leaking secrets from its context via tool calls or links.

**Red teaming** — Adversarial testing for failures.

**Guardrail model** — A classifier screening inputs/outputs for policy violations.

**Alignment faking / deceptive alignment** — A model behaving differently when it infers it is being evaluated. An active research concern.

**Watermarking** — Statistical signals embedded in generated text to identify machine output.

**Interpretability** — Understanding internal mechanisms.

**Mechanistic interpretability** — Reverse-engineering specific circuits within the network.

**SAE (Sparse Autoencoder)** — Decomposes activations into sparse, human-interpretable features; the main tool for feature-level interpretability.

**Superposition / polysemanticity** — Networks representing more features than they have neurons, so single neurons encode multiple unrelated concepts.

**Probing** — Training a small classifier on internal activations to test what information they carry.

**Steering vector** — A direction in activation space that, when added, shifts behavior (more formal, more truthful, etc.).

**Model card / system card** — Documentation of a model's training, capabilities, evaluations and limitations.

---

## 18. Hardware & Systems

**GPU / TPU / accelerator** — The parallel hardware LLMs run on.

**VRAM / HBM** — On-accelerator memory. Usually the binding constraint.

**Memory bandwidth** — How fast weights can be streamed from memory. Token generation is bandwidth-bound, not compute-bound — which is why quantization speeds up decoding.

**Arithmetic intensity / roofline** — FLOPs per byte moved; determines whether an operation is compute- or memory-bound.

**FLOPs / MFU** — Floating-point operations; model FLOPs utilization measures what fraction of peak hardware throughput a training run achieves (40–55% is good).

**Kernel / kernel fusion** — GPU functions; fusing several into one avoids memory round-trips (FlashAttention's core trick).

**CUDA / Triton** — NVIDIA's GPU programming platform; Triton is a Python-level kernel language.

**`torch.compile`** — PyTorch's graph compiler; often a free speedup.

**Inference cost model** — Cost per million tokens, split between prefill and decode. What actually determines production economics.

**Cold start** — Latency to load a model into memory before serving.

**Multi-tenancy** — Serving many users from shared model instances; the reason prefix caching and continuous batching matter so much.
