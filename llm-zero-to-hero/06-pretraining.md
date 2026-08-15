# 06 — Pretraining

*Teaching a random network to speak: the loss, the loop, the data, and what it actually costs.*

---

## 6.1 The loss, precisely

The model outputs logits of shape `(batch, seq_len, vocab_size)`. The targets are the input shifted by one. Cross-entropy compares them.

Mechanically, for one position:

```
1. logits for this position:        [50257 raw scores]
2. softmax → probabilities:         [50257 numbers summing to 1]
3. look up the probability the model gave the CORRECT token
4. loss = -log(that probability)
```

Then average over every position and every sequence in the batch.

▶️ **Compute it two ways to see it's the same thing:**

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
vocab_size = 10
logits = torch.randn(1, 3, vocab_size)          # 1 sequence, 3 positions
targets = torch.tensor([[4, 2, 7]])             # correct next tokens

# manual
probs = torch.softmax(logits, dim=-1)
picked = probs[0, range(3), targets[0]]
manual_loss = -torch.log(picked).mean()

# PyTorch (expects logits as (N, vocab) and targets as (N,))
torch_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

print(f"picked probabilities: {picked.detach().numpy().round(4)}")
print(f"manual: {manual_loss:.4f}   torch: {torch_loss:.4f}")   # identical
```

Note `F.cross_entropy` takes **logits, not probabilities** — it applies log-softmax internally, in a numerically stable way. Passing softmaxed values is a classic silent bug.

### The sanity check that saves you hours

At initialization, the model knows nothing, so it should spread probability uniformly across the vocabulary. That means:

```
initial loss ≈ ln(vocab_size) = ln(50257) ≈ 10.82
```

**Always check this on step 0 of any training run.** If your starting loss is 10.8, your model, data pipeline, and loss function are wired correctly. If it's 30, something is broken — usually mismatched shapes or targets misaligned by more than one position. If it's 2, you have a data leak and your model is seeing the answer.

### Perplexity

```
perplexity = exp(loss)
```

It converts the loss into "how many tokens is the model effectively choosing between."

| Loss | Perplexity | Interpretation |
|---|---|---|
| 10.82 | 50,257 | Total ignorance (uniform guessing) |
| 5.0 | 148 | Learning something |
| 3.0 | 20 | Reasonable small model |
| 2.0 | 7.4 | Good |
| 1.5 | 4.5 | Strong modern model on general text |

Perplexity is comparable only across models using the *same tokenizer* on the *same data* — a frequent mistake in blog-post comparisons.

---

## 6.2 The training loop

▶️ **A complete, runnable pretraining script.** This is the real thing, just small.

```python
import torch
import torch.nn as nn

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total, n = 0.0, 0
    if len(data_loader) == 0:
        return float("nan")
    num_batches = min(num_batches or len(data_loader), len(data_loader))
    for i, (x, y) in enumerate(data_loader):
        if i >= num_batches:
            break
        with torch.no_grad():
            total += calc_loss_batch(x, y, model, device).item()
        n += 1
    return total / n


def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, track_tokens = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tokens_seen += x.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                tr = calc_loss_loader(train_loader, model, device, eval_iter)
                vl = calc_loss_loader(val_loader, model, device, eval_iter)
                model.train()
                train_losses.append(tr); val_losses.append(vl); track_tokens.append(tokens_seen)
                print(f"ep {epoch+1} step {global_step:05d} | "
                      f"train {tr:.3f} | val {vl:.3f} | ppl {torch.exp(torch.tensor(vl)):.1f}")

    return train_losses, val_losses, track_tokens
```

Every element of a billion-dollar training run is present here. What changes at scale is the data volume, the parallelism, and the monitoring — not the logic.

---

## 6.3 Reading loss curves

This is a genuine skill. Four patterns cover most situations:

```
HEALTHY                    OVERFITTING              UNDERFITTING             DIVERGENCE
loss                       loss                     loss                     loss
 │╲                         │╲                       │╲___________            │      ╱
 │ ╲___train                │ ╲___train              │                        │     ╱
 │  ╲__________             │  ╲______               │  (both flat            │    ╱
 │ ╲___val                  │ ╱‾‾‾‾‾‾ val            │   and high)            │╲__╱
 └──────────► steps         └──────────► steps       └──────────► steps       └────────► steps

both fall, small gap        val turns UPWARD         nothing improves         loss explodes
→ keep going                → stop; more data,       → bigger model,          → lower LR,
                              regularization,          higher LR,               check clipping,
                              or early stopping        train longer             resume from ckpt
```

**Overfitting** means the model is memorizing rather than generalizing. On a tiny corpus this happens within a few epochs — which is worth causing deliberately once, so you recognize the shape.

**Divergence** in large runs is usually a bad batch or too high a learning rate. The standard response is to lower the LR, verify gradient clipping is active, and resume from the last good checkpoint — skipping the offending data.

---

## 6.4 Optimization details that decide whether a run works

### AdamW

Plain gradient descent uses one learning rate for everything. **Adam** adapts per parameter using running estimates of the gradient's mean and variance, so rarely-updated parameters still move. **AdamW** additionally decouples weight decay from the gradient, which is the correct formulation and the LLM standard.

The cost: AdamW stores **two extra numbers per parameter**. At fp32 that's 8 bytes/param of optimizer state — the dominant term in the training memory formula from §5.6.

### Learning rate warmup

Adam's variance estimates are unreliable for the first few hundred steps, and a full-size update based on bad estimates can knock the model into a region it never recovers from. So start near zero and ramp up:

▶️
```python
import math

def lr_at_step(step, base_lr=3e-4, warmup=1000, total=100_000, min_lr=3e-5):
    if step < warmup:                                     # linear warmup
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)   # cosine decay
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

for s in [0, 100, 500, 1000, 10_000, 50_000, 100_000]:
    print(f"step {s:>7,}  lr {lr_at_step(s):.2e}")
```

### Cosine decay

Large steps early to explore, small steps late to settle into a minimum. Cosine is the common default; **WSD** (warmup–stable–decay) is increasingly popular because the stable phase can be extended without redesigning the schedule, making it easy to continue training later.

### Gradient clipping

If the total gradient norm exceeds a threshold (usually 1.0), rescale it down. One pathological batch then can't blow up the weights.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Place it **after** `loss.backward()` and **before** `optimizer.step()`. Every serious run uses it.

### Mixed precision

Train in **bf16** rather than fp32: half the memory, roughly double the throughput on modern hardware. bf16 is preferred over fp16 because it has fp32's exponent range and therefore doesn't need loss scaling to avoid underflow. Frontier runs now push parts of the computation to **fp8** with careful scaling.

---

## 6.5 Data: the real differentiator

Architectures across labs have converged and are broadly similar. **Data recipes are what separate models**, and they're the most closely guarded detail in the industry.

### The pipeline

```
Raw web crawl (petabytes)
   │
   ├─► Language ID and filtering
   ├─► Quality filtering       (learned classifiers, heuristics)
   ├─► Deduplication           (exact + fuzzy/n-gram; big quality win)
   ├─► Toxicity/PII filtering
   ├─► Benchmark decontamination
   └─► Mixing with curated sources (code, math, books, papers, multilingual)
   │
   ▼
Tokenized training corpus (trillions of tokens)
```

**Known effects worth remembering:**
- **Deduplication** improves quality *and* reduces memorization of training data.
- **Code in the mixture improves reasoning even on non-code tasks.** One of the more surprising and robust findings in the field.
- **Quality filtering often beats adding parameters.** A smaller model on better data frequently wins.
- **Annealing / mid-training**: a final phase on very high-quality data (textbooks, curated code, synthetic reasoning traces) at a decayed learning rate produces reliably large gains. Now standard practice.
- **Synthetic data** — model-generated, then filtered — is a large fraction of modern post-training corpora and a growing share of pretraining.

---

## 6.6 Scaling laws: predicting results before spending the money

Model loss follows remarkably clean power laws in compute, parameters and data. This means **you can run small experiments and extrapolate** — the reason anyone dares spend $50M on a single training run.

**Kaplan et al. (2020)** established the power-law relationships. **Chinchilla (2022)** corrected the balance: for a fixed compute budget, the optimum is roughly

```
20 tokens of training data per parameter
```

A 7B model should see ~140B tokens to be compute-optimal.

### Why modern models are deliberately "overtrained"

Chinchilla optimizes *training* cost. But a deployed model's *inference* cost dominates its lifetime spend — a popular model serves trillions of tokens. A smaller model is cheaper **forever**.

So the industry deliberately trains past the compute-optimal point: modern models see 100–1000+ tokens per parameter. Training costs more; serving costs far less. This is why an 8B model in 2026 outperforms a 70B model from 2023.

**The practical consequence: parameter count is a weak proxy for capability.** Training tokens, data quality, and post-training matter at least as much.

### Estimating compute

The standard approximation:

```
training FLOPs ≈ 6 × parameters × tokens
```

(2 for the forward pass, 4 for the backward pass.)

▶️
```python
def training_cost(params, tokens, gpu_flops=1e15, mfu=0.45, gpu_hour_usd=2.0):
    flops = 6 * params * tokens
    gpu_hours = flops / (gpu_flops * mfu) / 3600
    return flops, gpu_hours, gpu_hours * gpu_hour_usd

for name, p, t in [
    ("GPT-2 124M (Chinchilla-optimal)", 124e6, 2.5e9),
    ("7B, 2T tokens",                   7e9,   2e12),
    ("70B, 15T tokens",                 70e9,  15e12),
]:
    f, h, usd = training_cost(p, t)
    print(f"{name:34s} {f:.2e} FLOPs  {h:10,.0f} GPU-hours  ~${usd:12,.0f}")
```

**MFU** (model FLOPs utilization) is the fraction of theoretical peak hardware throughput a run actually achieves. 40–55% is good; the gap is communication, memory stalls, and idle time.

---

## 6.7 Scaling beyond one GPU

At some point the model or the batch stops fitting. The parallelism strategies, in the order you'd reach for them:

| Strategy | Splits | Use when |
|---|---|---|
| **Data parallel (DDP)** | The batch across GPUs; every GPU holds a full model copy | The model fits on one GPU |
| **ZeRO / FSDP** | Optimizer state, gradients, then parameters across GPUs | Model + optimizer state don't fit |
| **Tensor parallel** | Individual weight matrices across GPUs | A single layer doesn't fit; needs fast interconnect |
| **Pipeline parallel** | Layer groups across GPUs | Very deep models; introduces "bubble" idle time |
| **Expert parallel** | MoE experts across GPUs | MoE models |
| **Context/sequence parallel** | The sequence dimension | Very long context |

Large runs combine several ("3D parallelism"). Two memory-saving techniques apply at any scale:

- **Gradient checkpointing** — discard intermediate activations and recompute them during the backward pass. Roughly 30% more compute for a large memory saving.
- **Gradient accumulation** — run several micro-batches, sum their gradients, then step once. Simulates a large batch on small memory.

▶️ **Gradient accumulation** — a five-line change to the loop, shown here as a runnable demo:

```python
import torch
import torch.nn as nn

torch.manual_seed(0)
model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batches = [(torch.randn(8, 4), torch.randint(0, 2, (8,))) for _ in range(16)]

accumulation_steps = 8      # effective batch = 8 micro-batches x 8 examples = 64
updates = 0

for i, (x, y) in enumerate(batches):
    loss = nn.functional.cross_entropy(model(x), y) / accumulation_steps
    loss.backward()                                    # gradients ACCUMULATE
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        updates += 1

print(f"{len(batches)} micro-batches -> {updates} optimizer steps")
print(f"effective batch size: {8 * accumulation_steps}")
```

Note the division by `accumulation_steps`. Without it the accumulated gradient is 8× too large, which silently makes your effective learning rate 8× higher than you think.

Note the division by `accumulation_steps` — without it your effective learning rate is 8× too high.

---

## 6.8 Saving and resuming

▶️
```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
global_step = 1234

# save BOTH — resuming without optimizer state loses AdamW's momentum
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": global_step,
}, "checkpoint.pth")

ckpt = torch.load("checkpoint.pth", weights_only=True)
restored = nn.Linear(4, 2)
restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
restored.load_state_dict(ckpt["model_state_dict"])
restored_opt.load_state_dict(ckpt["optimizer_state_dict"])

print(f"resumed at step {ckpt['step']}")
print("weights match:", torch.equal(model.weight, restored.weight))
```

---

## Check yourself

1. Your training run starts at loss 10.82 with a 50257 vocabulary. Good or bad?
2. Loss 2.5 — what's the perplexity, and what does it mean in words?
3. Validation loss turns upward while training loss falls. Diagnosis and three responses?
4. Why does warmup matter specifically for Adam-family optimizers?
5. State the Chinchilla ratio, and why modern models deliberately violate it.
6. Estimate the FLOPs to train a 13B model on 5T tokens.
7. Why must you save optimizer state, not just weights?

<details>
<summary>Answers</summary>

1. Good — `ln(50257) ≈ 10.82` is exactly uniform guessing, confirming the pipeline is correctly wired.
2. `e^2.5 ≈ 12.2` — the model is about as uncertain as choosing uniformly among 12 tokens.
3. Overfitting. Get more data, add regularization, or stop early (and reduce model size if the gap is extreme).
4. Adam's gradient mean/variance estimates are unreliable in the first steps; full-size updates based on them can destabilize training irrecoverably.
5. ~20 tokens per parameter is compute-optimal for training. Models are overtrained past it because inference cost dominates a model's lifetime and smaller models are permanently cheaper to serve.
6. `6 × 13e9 × 5e12 = 3.9e23` FLOPs.
7. AdamW carries per-parameter momentum and variance estimates; discarding them effectively restarts the optimizer and costs real progress.
</details>

---

**Next:** [07 — Post-Training](07-post-training.md)
