# 01 — What a Language Model Is

*The single objective from which everything else emerges.*

---

## 1.1 A language model is a probability distribution over text

Strip away the marketing and an LLM is a function that answers one question:

> Given the text so far, what is the probability of each possible next token?

Feed it `"The capital of France is"` and it returns a probability for every one of ~50,000 possible next tokens:

```
" Paris"      0.847
" the"        0.041
" a"          0.018
" located"    0.011
" home"       0.009
...
" refrigerator" 0.0000001
```

That's the whole model. Everything else — writing code, summarizing documents, holding a conversation, solving math problems — is this distribution, sampled from repeatedly.

Formally, a language model assigns a probability to a whole sequence by decomposing it into a chain of next-token predictions:

```
P("the cat sat") = P("the") × P("cat" | "the") × P("sat" | "the cat")
```

This is exact, not an approximation — it's the chain rule of probability. The modelling choice is only *how* to compute each conditional term. Before 2017, people used counting (n-grams) and recurrent networks. Now we use transformers.

Because each token is predicted from the tokens before it, the model is **autoregressive** ("auto" = self, "regressive" = predicting from previous values).

---

## 1.2 Why next-token prediction produces intelligence-shaped behavior

This is the part that seems too simple to work, and it's worth sitting with, because it explains both the capabilities and the failure modes you'll meet later.

To predict the next token *well* across all of human text, a model is forced to learn:

| To correctly predict... | ...it must have learned |
|---|---|
| `"The capital of France is ___"` | Factual knowledge |
| `"She walked into the room and ___"` | Narrative and physical plausibility |
| `"def add(a, b): return ___"` | Programming semantics |
| `"2 + 2 = ___"` | Arithmetic |
| `"The word for 'water' in Spanish is ___"` | Translation |
| `"...therefore the murderer must be ___"` | Multi-step inference |
| `"Q: What's 17 × 23? A: Let me work through it. 17 × 20 = ___"` | Procedural reasoning |

Nobody trains these skills individually. They are all *incidentally required* for low loss on a sufficiently diverse corpus. Compression and understanding turn out to be nearly the same problem: to predict text efficiently you must model the process that generated it — and that process is people thinking.

**But notice what the objective actually rewards.** It rewards *plausible continuation*, not *truth*. A model that has never seen a fact will still produce the most plausible-looking completion, confidently. That is the mechanistic origin of hallucination, and it's why hallucination is not a bug to be patched but a property to be managed (chapter 10).

---

## 1.3 Self-supervised learning: where the labels come from

Supervised learning needs labeled examples, and labeling is expensive. Language modeling sidesteps this entirely: **the labels are already in the text.**

Take any sentence and shift it by one position:

```
input:   The   cat   sat   on   the
target:  cat   sat   on    the  mat
```

Every position produces a training example, free. No annotators. This is why the entire internet is usable as training data, and it's the reason LLMs could scale when other approaches couldn't.

Note that the model predicts a target at *every* position simultaneously during training, not just the last one — a 1024-token sequence yields 1024 training signals in a single forward pass. That parallelism is the transformer's key advantage over the RNNs it replaced.

**One subtlety with a name:** during training the model always sees the *correct* previous tokens, even when its own prediction would have been wrong. This is called **teacher forcing**. At inference it must consume its own outputs instead, which is a slightly different (and harder) situation — the mismatch is called **exposure bias**, and it's part of why models sometimes spiral once they make an early mistake.

---

## 1.4 The two-stage lifecycle

Nearly all beginner confusion about LLMs dissolves once you separate these two stages.

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: PRETRAINING                            (Chapter 06)   │
│                                                                 │
│  Data:      trillions of tokens of raw internet text            │
│  Objective: predict the next token                              │
│  Cost:      $100k – $100M+, weeks on thousands of GPUs          │
│  Produces:  a BASE MODEL                                        │
│                                                                 │
│  A base model completes text. It does not converse.             │
│  Prompt it with "What is the capital of France?" and a good     │
│  base model might reply:                                        │
│      "What is the capital of Germany? What is the capital..."   │
│  ...because in its training data, questions cluster with        │
│  more questions. It is not being unhelpful — it is doing        │
│  exactly what it was trained to do.                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: POST-TRAINING                          (Chapter 07)   │
│                                                                 │
│  Data:      thousands to millions of curated examples           │
│  Objective: imitate good responses, then optimize for           │
│             preference or verified correctness                  │
│  Cost:      0.1–5% of pretraining                               │
│  Produces:  an ASSISTANT / INSTRUCT / CHAT model                │
│                                                                 │
│  Now "What is the capital of France?" → "The capital of         │
│  France is Paris."                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Pretraining installs the knowledge and the capabilities. Post-training installs the behavior.** A model cannot be finetuned into knowing something it never saw in pretraining — but it can very easily be finetuned into *presenting* what it knows differently. Keep this distinction; it decides most real engineering choices (chapter 09: should I finetune or use RAG?).

---

## 1.5 The full pipeline, named

Here is the whole system, with the chapter that explains each piece:

```
"The capital of France is"

  1. TOKENIZE               →  [464, 3139, 286, 4881, 318]        [Ch 02]
     text becomes integers

  2. EMBED                  →  5 vectors of 768 numbers           [Ch 03]
     integers become meaning-vectors, plus position info

  3. TRANSFORM (×N layers)  →  5 refined vectors of 768           [Ch 04–05]
     each layer: attention (mix across positions)
                 + feed-forward (transform each position)

  4. PROJECT                →  5 × 50257 logits                   [Ch 05]
     final vector → one raw score per vocabulary entry

  5. SOFTMAX                →  probabilities summing to 1         [Ch 05]

  6. SAMPLE                 →  " Paris"                           [Ch 08]
     pick one token according to some strategy

  7. APPEND AND REPEAT      →  "The capital of France is Paris"
```

Steps 1–6 run once per generated token. A 500-word answer runs this loop about 650 times. Understanding that this is a **loop**, and that step 3 is by far the expensive part, explains everything about inference cost in chapter 08.

▶️ **See the real thing** (no model needed — just the shapes):

```python
import torch

vocab_size, n_tokens, d_model = 50257, 5, 768

hidden = torch.randn(1, n_tokens, d_model)       # after step 3
output_head = torch.nn.Linear(d_model, vocab_size, bias=False)

logits = output_head(hidden)                      # step 4
print("logits:", logits.shape)                    # (1, 5, 50257)

next_token_logits = logits[0, -1]                 # only the LAST position matters
probs = torch.softmax(next_token_logits, dim=-1)  # step 5
print("probs sum to:", probs.sum().item())        # 1.0
print("most likely token id:", probs.argmax().item())
```

Note `logits[0, -1]`: the model produces a prediction at every position, but when generating we only use the last one. The others were useful during training (that's the free parallelism from §1.3) and are wasted at inference — a fact that speculative decoding later exploits cleverly.

---

## 1.6 A brief history, because the names keep appearing

| Era | Approach | Fatal limitation |
|---|---|---|
| 1990s–2000s | **N-grams**: count how often "sat on the" is followed by "mat" | Can't generalize; context of 3–5 words maximum |
| 2013 | **Word2Vec**: words as learned vectors | Static — one vector per word, so "bank" has a single meaning |
| 2014–2017 | **RNNs / LSTMs**: process tokens one at a time, carrying a hidden state | Sequential (can't parallelize training) and forgetful over long distances |
| 2017 | **Transformer** (*Attention Is All You Need*) | — |
| 2018 | **BERT** (encoder-only, bidirectional) and **GPT-1** (decoder-only, generative) | — |
| 2019–2020 | **GPT-2, GPT-3**: scale works, few-shot learning emerges | — |
| 2022 | **InstructGPT / ChatGPT**: RLHF makes models usable | — |
| 2023–2024 | Open weights (Llama, Mistral, Qwen), long context, multimodality | — |
| 2025–2026 | **Reasoning models** (RLVR), **MoE** everywhere, **agents** | — |

The two limitations that killed RNNs are worth remembering, because attention is precisely the fix for both:

1. **Sequential computation.** An RNN must process token 500 after token 499. You cannot parallelize across the sequence, which caps how much data you can train on.
2. **The information bottleneck.** Everything the RNN knows about the first 400 tokens must be squeezed into one fixed-size hidden state. Distant information decays.

Attention solves both: every position can look at every other position **directly** (no decay) and **simultaneously** (fully parallel). Chapter 04 is that mechanism.

---

## 1.7 Terms you now own

**Language model** · **Token** (chapter 02 makes this precise) · **Autoregressive** · **Next-token prediction** · **Self-supervised learning** · **Teacher forcing** · **Exposure bias** · **Pretraining** · **Base model** · **Post-training / finetuning** · **Instruct model** · **Logits** · **Context** · **Emergence** · **Hallucination** (as a consequence of the objective)

---

## Check yourself

1. Why is next-token prediction called *self*-supervised rather than unsupervised?
2. A base model answers your question with three more questions. Is it broken?
3. Which stage gives a model knowledge, and which gives it manners? Why does that distinction decide whether to finetune or use retrieval?
4. Why can a transformer train on a 1024-token sequence faster than an RNN can?
5. Explain in one sentence why hallucination follows from the training objective.

<details>
<summary>Answers</summary>

1. The labels exist and are used — they're just derived automatically from the data by shifting it, rather than provided by humans.
2. No. It's doing next-token prediction correctly on text where questions are followed by questions. It has no post-training.
3. Pretraining gives knowledge; post-training gives behavior. Since finetuning mostly shapes behavior rather than installing new facts, new *information* usually belongs in the context (retrieval), not in the weights.
4. The transformer predicts all 1024 positions in one parallel forward pass; the RNN must step through them one at a time.
5. The model is trained to always produce the most plausible continuation, and plausibility is not truth — so when it lacks the fact, it produces something that looks right.
</details>

---

**Next:** [02 — Tokenization](02-tokenization.md)
