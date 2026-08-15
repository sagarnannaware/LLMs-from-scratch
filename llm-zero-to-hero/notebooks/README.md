# Notebooks

Eight runnable notebooks, one per major concept. **Every notebook has been executed end to end and its outputs are saved**, so you can read the results on GitHub without running anything — but the point is to run them and change things.

All eight run on a **laptop CPU**. No GPU, no API key, no downloads.

## Setup

```bash
pip install torch numpy matplotlib     # required
pip install tiktoken                   # optional, one cell in notebook 01
jupyter lab
```

## The notebooks

| # | Notebook | Runtime | You will build |
|---|---|---|---|
| 01 | [Tokenization](01_tokenization.ipynb) | seconds | BPE training from scratch, a tokenizer class, the sliding-window dataloader |
| 02 | [Attention from Scratch](02_attention_from_scratch.ipynb) | seconds | Attention in 4 stages → multi-head, checked against PyTorch's fused kernel |
| 03 | [Build a GPT](03_build_gpt.ipynb) | ~10 s | LayerNorm, GELU, FFN, TransformerBlock, a complete 163M-parameter GPT-2 |
| 04 | [Train a Tiny LLM](04_train_a_tiny_llm.ipynb) | ~90 s | **A real LLM trained from random weights to working text generation** |
| 05 | [Decoding Strategies](05_decoding_strategies.ipynb) | seconds | Temperature, top-k, top-p, and why greedy decoding loops |
| 06 | [KV Cache & Speed](06_kv_cache_and_speed.ipynb) | seconds | A KV cache, a measured 2–3× speedup, and a serving capacity calculator |
| 07 | [LoRA Finetuning](07_lora_finetuning.ipynb) | ~90 s | LoRA from scratch, a real finetune at 2% of parameters, adapter merging |
| 08 | [Modern Components](08_modern_components.ipynb) | seconds | RMSNorm, SwiGLU, RoPE, GQA, MoE — and a 2026-current transformer block |

Read the matching chapter first; the notebooks assume its explanations.

| Notebook | Chapter |
|---|---|
| 01 | [02 — Tokenization](../02-tokenization.md) |
| 02 | [04 — Attention](../04-attention.md) |
| 03, 08 | [05 — The Transformer](../05-transformer-architecture.md) |
| 04 | [06 — Pretraining](../06-pretraining.md) |
| 05, 06 | [08 — Inference & Efficiency](../08-inference-and-efficiency.md) |
| 07 | [07 — Post-Training](../07-post-training.md) |

## Selected results

These are actual outputs from the executed notebooks, not illustrations.

**Notebook 04** trains a 4-layer model on a generated corpus in ~90 seconds:

```
step    0 | train 3.3716 | val 3.3678 | ppl  29.01     ← ln(vocab) = random guessing
step  100 | train 0.4088 | val 0.4218 | ppl   1.52
step  599 | train 0.1995 | val 0.2012 | ppl   1.22

the cat looks at the garden in the sun. the cat runs past the mat again.
my friend jumps over the garden every morning. a bird jumps over the mat again.
```

From random weights to correct grammar, spelling, and punctuation — using nothing but next-character prediction.

**Notebook 06** measures the KV cache speedup, which grows with sequence length exactly as the theory predicts:

```
 new tokens   naive (s)  cached (s)   speedup
         32       0.036       0.016      2.3x
        128       0.204       0.066      3.1x
```

**Notebook 07** finetunes to a new task while training 2% of the parameters, then shows the base model is recoverable by switching the adapter off:

```
method                trainable   loss on B   loss on A
base (no FT)                  -      5.4397      0.1835
LoRA r=8                 16,384      0.1178      1.4118
full finetune           806,656      0.0978      1.2846

LoRA model with adapters OFF, loss on A: 0.1817   ← base intact
full finetune,          loss on A: 1.3028         ← permanently changed
```

Note what this shows honestly: **both** methods forget the original task. LoRA's advantage is not that it prevents forgetting — it's that the frozen base weights can be restored instantly.

## How to use these

Running the cells top to bottom teaches you the least. Do this instead:

1. **Predict before you run.** Read the cell, decide what it will print, then run it. Wrong predictions are where learning happens.
2. **Break things deliberately.** Each notebook ends with exercises that break something — remove the causal mask, set the learning rate 100× too high, collapse the MoE router. Predict the failure first.
3. **Retype from blank.** After notebook 02, open an empty file and write `MultiHeadAttention` from memory. That is the real test.

## Note on outputs

Outputs are committed so the notebooks are readable on GitHub. Re-running produces slightly different numbers where sampling is involved — seeds are set for the model weights, but timings and some generations will vary with hardware.
