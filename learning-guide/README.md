# The LLM Mastery Guide

A complete, self-paced study path built around this repository ([*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by Sebastian Raschka), plus everything the book intentionally leaves out because it is teaching fundamentals rather than chasing headlines.

**Who this is for:** you can write Python, you are new to LLMs, and you want to understand them well enough to build, train, finetune, debug and deploy one — not just call an API.

---

## The four documents

| File | What it gives you |
|---|---|
| **[01-roadmap.md](01-roadmap.md)** | Chapter-by-chapter study plan. What to read, what to run, what to be able to explain, exercises, and the traps that confuse beginners. |
| **[02-glossary.md](02-glossary.md)** | ~250 terms defined in plain language, grouped by theme, cross-referenced to the code in this repo. Your lookup table for the whole field. |
| **[03-modern-llm-landscape.md](03-modern-llm-landscape.md)** | Everything between GPT-2 (what this book builds) and the frontier models of 2026: MoE, RoPE, GQA, RLVR/GRPO, reasoning models, agents, serving, evals. |
| **[04-projects-and-resources.md](04-projects-and-resources.md)** | Milestone projects, a self-assessment checklist, the papers worth reading, the libraries worth learning, and how to keep up. |

Read them in that order the first time. After that, `02-glossary.md` and `03-modern-llm-landscape.md` are reference material you will return to for years.

---

## What you will be able to do at the end

1. Explain, from memory, how text becomes tokens, tokens become embeddings, and embeddings become a next-token probability distribution.
2. Implement multi-head causal self-attention in a blank file without looking anything up.
3. Build a GPT-2-class transformer, train it, watch the loss curve, and diagnose it when it misbehaves.
4. Load real OpenAI GPT-2 weights into *your* implementation and generate text.
5. Finetune that model two ways: for classification (spam detection) and for instruction-following.
6. Apply LoRA, and explain exactly which matrices it touches and why that saves memory.
7. Read a modern model card or architecture paper (Llama, Qwen, DeepSeek, Mixtral) and understand every component named in it.
8. Hold your own in a conversation about MoE routing, KV-cache pressure, RLVR, speculative decoding, or context engineering.

Items 1–6 come from the book. Items 7–8 come from `03-modern-llm-landscape.md`.

---

## Prerequisites, honestly assessed

You need **less** math than people tell you, but you cannot skip all of it.

| Skill | Level needed | If you're short |
|---|---|---|
| Python | Comfortable: classes, list comprehensions, decorators-optional | Any intermediate Python course |
| NumPy-style array thinking | Must be fluent in shapes, broadcasting, axes | Practice: reshape/transpose/matmul until shapes are obvious |
| Linear algebra | Matrix multiply, dot product, transpose. **That's genuinely most of it.** | 3Blue1Brown, *Essence of Linear Algebra*, eps. 1–4 |
| Calculus | Know what a derivative and a gradient *mean*. You will never compute one by hand. | 3Blue1Brown, *Essence of Calculus*, eps. 1–3 |
| Probability | Probability distribution, sampling, log-probability, cross-entropy | Covered in ch05 of the book itself |
| PyTorch | Tensors, `nn.Module`, autograd, training loops | **[appendix-A](../appendix-A/01_main-chapter-code/)** in this repo — do it first |

If matrix multiplication and tensor shapes are shaky, spend three days there before starting. Everything downstream is shape bookkeeping.

---

## Setup

```bash
git clone https://github.com/sagarnannaware/LLMs-from-scratch.git
cd LLMs-from-scratch

# Option A: plain pip + venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option B: uv (faster, recommended)
pip install uv
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

jupyter lab
```

Detailed instructions, including Docker and conda: **[../setup/README.md](../setup/README.md)**.

**Hardware reality check:**

- **Chapters 2–4** run on any laptop, CPU only. No GPU needed.
- **Chapter 5** pretraining on `the-verdict.txt` takes minutes on CPU, seconds on GPU. This is a deliberately tiny run — do not expect a chatbot.
- **Chapters 6–7** finetune the 124M model. CPU works but is slow (hours); a free Colab/Kaggle T4 GPU makes it minutes. Apple Silicon `mps` works well.
- Nothing in the book requires an expensive GPU. If a step is slow, reduce `num_epochs` or dataset size first — the *learning* is in the mechanism, not the final loss value.

Verify your install:

```bash
python -c "import torch, tiktoken; print(torch.__version__, torch.cuda.is_available())"
python ch04/01_main-chapter-code/gpt.py   # should print generated (gibberish) text
```

Gibberish is correct here — that model is randomly initialized.

---

## The 12-week plan

Roughly 8–10 hours/week. Compress or stretch freely; the order matters more than the pace.

| Week | Focus | Deliverable you should have at the end |
|---|---|---|
| 1 | [appendix-A](../appendix-A) PyTorch, part 1–2 | A trained toy classifier, written by you |
| 2 | [ch01](../ch01) + [ch02](../ch02) tokenization, embeddings, dataloader | A working `GPTDatasetV1`-style dataloader, written from scratch |
| 3 | [ch03](../ch03) attention: simplified → self → causal → multi-head | `MultiHeadAttention` typed from a blank file, tests passing |
| 4 | [ch03](../ch03) bonus: efficient MHA variants, buffers | You can explain why `register_buffer` is used for the mask |
| 5 | [ch04](../ch04) the full GPT model | `gpt.py` reproduced; parameter count matches 163,009,536 |
| 6 | [ch05](../ch05) pretraining, loss, generation | A loss curve you can interpret; temperature/top-k experiments |
| 7 | [ch05](../ch05) loading OpenAI weights + [appendix-D](../appendix-D) training extras | Real GPT-2 text generation from your own model class |
| 8 | [ch06](../ch06) classification finetuning | A spam classifier ≥95% test accuracy |
| 9 | [ch07](../ch07) instruction finetuning | A model that follows simple instructions; Ollama-based eval scores |
| 10 | [appendix-E](../appendix-E) LoRA + [ch07/04](../ch07/04_preference-tuning-with-dpo) DPO | LoRA finetune matching full-finetune accuracy at a fraction of trained params |
| 11 | [ch05/07](../ch05/07_gpt_to_llama) GPT→Llama conversion | Your GPT converted to Llama 2 and Llama 3 architecture |
| 12 | [03-modern-llm-landscape.md](03-modern-llm-landscape.md) + a project from [04-projects-and-resources.md](04-projects-and-resources.md) | One end-to-end project you can show someone |

---

## How to actually study this (the part that determines whether it works)

The failure mode for this repo is **reading notebooks and feeling productive**. Notebooks are seductive: every cell already works. Understanding comes from the parts that don't.

Use this loop for every chapter:

1. **Read** the chapter notebook top to bottom without running anything. Goal: the shape of the idea.
2. **Run** it cell by cell. Print shapes constantly: `print(x.shape)` after every transformation. Shapes are the load-bearing intuition in this entire field.
3. **Break it.** Change `num_heads` to a value that doesn't divide `d_out`. Remove the causal mask. Set dropout to 0.9. Predict the error *before* you run it. Being able to predict failures is the actual skill.
4. **Rebuild.** Open an empty file and reimplement the chapter's core class from memory. Check against the repo only when stuck, then start over from empty the next day.
5. **Explain.** Write 5 sentences explaining the chapter to an imaginary beginner. If a sentence needs hand-waving, that's your gap.

Rule of thumb: if you have not typed it from a blank file, you do not know it.

---

## A mental model to carry through everything

An LLM is one loop:

```
text → tokens → embeddings → [transformer blocks] → logits → probabilities → next token → append → repeat
```

Everything in this book, and everything in the modern landscape document, is either:

- a better way to **represent** the input (tokenization, embeddings, positional encoding),
- a better way to **mix information across positions** (attention and its many descendants),
- a better way to **fit the weights** (pretraining, finetuning, RLHF/RLVR),
- or a better way to **run the loop cheaply** (KV cache, quantization, MoE, speculative decoding).

When you meet an unfamiliar acronym, ask which of those four buckets it lives in. That single question converts most of the field's jargon into something navigable.

---

*This guide is study material added to a fork of the book's repository. The chapter code and the book itself are Sebastian Raschka's work — buy the book, it is worth it.*
