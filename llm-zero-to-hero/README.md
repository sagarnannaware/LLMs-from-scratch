# LLMs: Zero to Hero

A complete, self-contained explanation of how large language models work — from "what is a vector" to "how does GRPO train a reasoning model."

This guide **teaches the concepts**. It does not assume you have read anything else, and it does not send you elsewhere for the explanations. Every idea is built up from the one before it, with worked numeric examples and runnable code you can paste into a Python file.

---

## The curriculum

| # | Chapter | What you'll understand after it |
|---|---|---|
| 00 | **[Foundations](00-foundations.md)** | Vectors, matrices, neural networks, gradient descent, backprop, PyTorch. The actual prerequisites, taught rather than assumed. |
| 01 | **[What a Language Model Is](01-language-modeling.md)** | Probability over text, the next-token objective, why that one trick produces everything else. |
| 02 | **[Tokenization](02-tokenization.md)** | How text becomes numbers, BPE built by hand, why tokenizers cause so many real-world bugs. |
| 03 | **[Embeddings & Position](03-embeddings-and-position.md)** | Meaning as geometry, learned positions, and RoPE explained properly. |
| 04 | **[Attention](04-attention.md)** | The central mechanism, derived step by step with real numbers you can check on paper. |
| 05 | **[The Transformer](05-transformer-architecture.md)** | The full architecture assembled, GPT-2 → 2026 (RMSNorm, SwiGLU, GQA, MoE, hybrids). |
| 06 | **[Pretraining](06-pretraining.md)** | Loss, data, optimizers, scaling laws, distributed training, what a real run costs. |
| 07 | **[Post-Training](07-post-training.md)** | SFT, RLHF, DPO, RLVR/GRPO, reasoning models, LoRA/QLoRA — turning a text predictor into an assistant. |
| 08 | **[Inference & Efficiency](08-inference-and-efficiency.md)** | Decoding strategies, KV cache, quantization, speculative decoding, how serving actually works. |
| 09 | **[Building Applications](09-building-applications.md)** | Context engineering, RAG, agents, tools, structured output. |
| 10 | **[Evaluation & Safety](10-evaluation-and-safety.md)** | How to know if it works, hallucination, prompt injection, interpretability. |
| 11 | **[Projects & Path to Mastery](11-projects-and-path.md)** | A project ladder, self-assessment, papers in reading order. |
| — | **[Glossary](glossary.md)** | ~250 terms in 18 themed sections. |
| — | **[Notebooks](notebooks/)** | 8 runnable Jupyter notebooks, all executed with outputs saved. See below. |
| — | **[Appendix: Code Along](appendix-code-along.md)** | Optional: a chapter-by-chapter walkthrough of this repo (Sebastian Raschka's *Build a LLM From Scratch*) if you want to type every line yourself. |

---

## The notebooks

Theory you can't run is theory you'll forget. Every major concept has a notebook that **runs on a laptop CPU** in seconds to ~90 seconds. All eight are executed and their outputs committed, so you can read the results on GitHub before running anything.

| # | Notebook | Pairs with | You build |
|---|---|---|---|
| 01 | [Tokenization](notebooks/01_tokenization.ipynb) | Ch 02 | BPE from scratch, the training dataloader |
| 02 | [Attention from Scratch](notebooks/02_attention_from_scratch.ipynb) | Ch 04 | Attention in 4 stages → multi-head |
| 03 | [Build a GPT](notebooks/03_build_gpt.ipynb) | Ch 05 | A complete 163M-parameter GPT-2 |
| 04 | [Train a Tiny LLM](notebooks/04_train_a_tiny_llm.ipynb) | Ch 06 | **A real LLM, random weights → working text** |
| 05 | [Decoding Strategies](notebooks/05_decoding_strategies.ipynb) | Ch 08 | Temperature, top-k, top-p |
| 06 | [KV Cache & Speed](notebooks/06_kv_cache_and_speed.ipynb) | Ch 08 | A KV cache + measured speedup |
| 07 | [LoRA Finetuning](notebooks/07_lora_finetuning.ipynb) | Ch 07 | LoRA from scratch, a real finetune |
| 08 | [Modern Components](notebooks/08_modern_components.ipynb) | Ch 05 | RMSNorm, SwiGLU, RoPE, GQA, MoE |

Notebook 04 is the one to reach for if you want proof this all works — it takes a randomly initialized transformer to correct grammar and spelling in about 90 seconds:

```
step    0 | train 3.3716 | ppl 29.01      ← random guessing
step  599 | train 0.1995 | ppl  1.22

the cat looks at the garden in the sun. the cat runs past the mat again.
```

```bash
pip install torch numpy matplotlib && jupyter lab
```

---

## How to read this

**If you're starting from zero:** go in order. Chapter 00 through 05 is one continuous argument — each chapter is the answer to a question the previous one raised. Don't skip 00 if matrix multiplication feels shaky; everything after it is matrix multiplication.

**If you already code and know basic ML:** skim 00, start at 02.

**If you know transformers and want the modern material:** start at 05's second half, then 07 and 08.

Each chapter ends with **check yourself** questions. If you can't answer them, re-read before continuing — the chapters are strictly cumulative, and a gap in attention becomes an impassable wall in post-training.

Code blocks marked ▶️ are runnable. You need only:

```bash
pip install torch numpy
```

Nothing in this guide requires a GPU.

---

## The one diagram that holds everything

Every LLM, from GPT-2 to whatever ships next year, is this loop:

```
   "The capital of France is"
            │
     ┌──────▼───────┐
     │  TOKENIZER   │   text → integers            [Ch 02]
     └──────┬───────┘
            │  [464, 3139, 286, 4881, 318]
     ┌──────▼───────┐
     │  EMBEDDINGS  │   integers → vectors         [Ch 03]
     └──────┬───────┘
            │  shape: (5 tokens, 768 numbers each)
     ┌──────▼───────────────────────────┐
     │  TRANSFORMER BLOCK  × 12         │
     │  ┌────────────────────────────┐  │
     │  │ Attention: mix across      │  │           [Ch 04]
     │  │ positions                  │  │
     │  ├────────────────────────────┤  │
     │  │ Feed-forward: transform    │  │           [Ch 05]
     │  │ each position              │  │
     │  └────────────────────────────┘  │
     └──────┬───────────────────────────┘
            │  shape: (5, 768)
     ┌──────▼───────┐
     │  OUTPUT HEAD │   vectors → one score per    [Ch 05]
     └──────┬───────┘   word in the vocabulary
            │  shape: (5, 50257)
     ┌──────▼───────┐
     │   SAMPLING   │   scores → pick one token    [Ch 08]
     └──────┬───────┘
            │
         " Paris"  ──────► append to input, repeat
```

Training (chapters 06–07) adjusts the numbers inside those boxes. Inference (chapter 08) runs the loop cheaply. Applications (chapter 09) decide what goes into the input.

**Whenever you meet an unfamiliar term for the rest of your career, ask which box it belongs to.** Almost every acronym in this field is a better version of one of these six steps. That single question turns an overwhelming firehose into something navigable.

---

## What you'll be able to do at the end

- Explain every component of a transformer, and why it exists rather than merely what it does.
- Write attention from a blank file.
- Read a modern model card or architecture paper and understand every term in it.
- Choose correctly between prompting, RAG, and finetuning for a real problem.
- Explain how a reasoning model is trained, and why RLVR changed the field.
- Debug a training run from its loss curve, and a serving setup from its latency profile.
- Build things that work, and evaluate them honestly.

Start with **[00 — Foundations](00-foundations.md)**.
