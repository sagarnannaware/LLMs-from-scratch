# 05 — The Transformer

*Assembling a complete working LLM, then upgrading it to a 2026 design.*

---

## 5.1 The shape of the thing

You now have attention. A transformer wraps it in supporting machinery and stacks the result:

```
        token IDs  (batch, seq)
             │
    ┌────────▼─────────┐
    │ token embedding  │
    │ + position       │                          [Ch 03]
    └────────┬─────────┘
             │  (batch, seq, 768)
    ┌────────▼─────────────────────────┐
    │  TRANSFORMER BLOCK               │  ×12
    │                                  │
    │   ┌──────────────┐               │
    │   │ LayerNorm    │               │
    │   ├──────────────┤               │
    │   │ Multi-Head   │               │  ← mixes ACROSS tokens   [Ch 04]
    │   │ Attention    │               │
    │   ├──────────────┤               │
    │   │ + residual   │◄──────────────┼── shortcut
    │   └──────────────┘               │
    │   ┌──────────────┐               │
    │   │ LayerNorm    │               │
    │   ├──────────────┤               │
    │   │ Feed-Forward │               │  ← transforms EACH token
    │   ├──────────────┤               │
    │   │ + residual   │◄──────────────┼── shortcut
    │   └──────────────┘               │
    └────────┬─────────────────────────┘
             │  (batch, seq, 768)
    ┌────────▼─────────┐
    │ final LayerNorm  │
    ├──────────────────┤
    │ output head      │  768 → 50257
    └────────┬─────────┘
             │  (batch, seq, 50257) = logits
```

The single most useful sentence about this diagram:

> **Attention is communication between tokens. The feed-forward network is computation within a token.**

Every block alternates: gather relevant information from elsewhere, then think about it locally. Twelve times.

---

## 5.2 The four components

### Layer normalization

Deep networks suffer from activations that drift toward exploding or vanishing as they pass through layers. LayerNorm fixes each token's vector to a standard scale:

```
for each token's 768-dim vector:
    subtract its mean
    divide by its standard deviation
    multiply by a learned `scale` (768 numbers)
    add a learned `shift` (768 numbers)
```

Crucially it normalizes **across the feature dimension of each token independently** — not across the batch. That's why it suits variable-length sequences where batch statistics would be meaningless. (Batch normalization, common in vision, does the opposite and is a poor fit here.)

The learned scale and shift matter: pure normalization would throw away magnitude information the network may need, so we let it learn how much to restore.

▶️
```python
import torch, torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                              # prevents division by zero
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift

x = torch.randn(2, 4, 768) * 5 + 3
out = LayerNorm(768)(x)
print(out.mean(dim=-1)[0, 0].item(), out.std(dim=-1)[0, 0].item())   # ≈0.0, ≈1.0
```

### GELU activation

ReLU (`max(0, x)`) has a hard corner: everything negative is annihilated, and the gradient is exactly zero there, so a unit pushed negative can stop learning permanently. **GELU** smooths this — small negative values pass through slightly attenuated, and the gradient is continuous everywhere.

▶️
```python
import torch, torch.nn as nn

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)
        ))

for v in [-3.0, -1.0, -0.5, 0.0, 1.0, 3.0]:
    t = torch.tensor([v])
    print(f"x={v:5.1f}   ReLU={torch.relu(t).item():6.3f}   GELU={GELU()(t).item():6.3f}")
```

Notice GELU at `x = -0.5` is about `-0.154`, not 0. That small leak is the whole difference, and it trains measurably better.

### Feed-forward network

Two linear layers with an activation between, expanding to **4× the model width** and back:

```
768  →  3072  →  768
```

Why expand? The wide middle layer gives the network room to compute nonlinear combinations of features before compressing back. It's the model's per-token "thinking" capacity — and interpretability research suggests this is where much factual knowledge is stored, with attention doing the routing.

▶️
```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)
```

**Parameter accounting per block** (768-dim): attention holds `4 × 768 × 768 ≈ 2.36M`; the FFN holds `2 × 768 × 3072 ≈ 4.72M`. **The FFN is roughly two-thirds of the model.** People assume attention dominates; it doesn't.

### Residual connections

```
x = x + sublayer(x)
```

Three words that make deep networks possible. Without them, gradients shrink multiplicatively as they flow backwards through 12 layers and early layers barely learn — the **vanishing gradient** problem. The `+ x` gives gradients a direct path: the derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`, so there's always a "1" carrying signal straight through.

▶️ **See it fail without them:**

```python
import torch, torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(8, 8), nn.GELU()) for _ in range(5)
        ])

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = x + out if self.use_shortcut else out
        return x

for use_shortcut in [False, True]:
    torch.manual_seed(0)
    model = DeepNet(use_shortcut)
    model(torch.ones(1, 8)).sum().backward()
    print("shortcuts:", use_shortcut)
    for name, p in model.named_parameters():
        if "0.weight" in name:
            print(f"   {name:24s} grad magnitude {p.grad.abs().mean():.6f}")
```

Compare the first layer's gradient in both runs. Without shortcuts it's roughly an order of magnitude smaller — and at 12 or 96 layers instead of 5, it vanishes entirely.

### Pre-LN vs Post-LN

The 2017 paper normalized *after* each sublayer. GPT-2 and everything since normalizes *before*:

```
Post-LN (2017):   x = LayerNorm(x + Attention(x))          ← unstable at depth
Pre-LN (modern):  x = x + Attention(LayerNorm(x))          ← stable
```

Pre-LN keeps the residual path completely clean — nothing is applied to it — so gradients flow to layer 1 unimpeded. Post-LN models above ~12 layers often need careful warmup to train at all. This is why pre-LN is now universal.

---

## 5.3 The complete GPT model

▶️ **A working LLM in ~70 lines.** Paste and run it.

```python
import torch
import torch.nn as nn

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(          # from chapter 04
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"], num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))     # attention sublayer
        x = x + self.drop(self.ff(self.norm2(x)))      # feed-forward sublayer
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        _, seq_len = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = self.drop(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)                        # logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total:,}")                  # 163,009,536
print(f"Size (fp32): {total * 4 / 1e6:.2f} MB")        # 652.04 MB

out = model(torch.randint(0, 50257, (2, 10)))
print("logits:", out.shape)                            # (2, 10, 50257)
```

That is a complete, correct GPT-2. Its weights are random, so its output is meaningless — chapter 06 fixes that — but the architecture is exactly what OpenAI shipped.

### The parameter count puzzle

Why does "GPT-2 124M" report **163,009,536** parameters?

Because the original **ties** the output head to the token embedding matrix — the same `50257 × 768 = 38,597,376` numbers serve both roles:

```
163,009,536 − 38,597,376 = 124,412,160     ← the famous "124M"
```

**Weight tying** works because the embedding maps tokens → vectors and the output head maps vectors → tokens; they're transposes of the same relationship (recall §3.2). It saves parameters and often improves quality slightly. Modern large models usually *untie* them, since at 100B+ scale the embedding is a negligible fraction and untying gives a bit more flexibility.

---

## 5.4 Generating text

▶️ **The generation loop, complete:**

```python
import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]         # never exceed the context window
        logits = model(idx_cond)                  # (b, seq, vocab)
        logits = logits[:, -1, :]                 # ONLY the last position predicts next
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.argmax(probs, dim=-1, keepdim=True)    # greedy
        idx = torch.cat([idx, next_id], dim=1)    # append and continue
    return idx

start = torch.tensor([[15496, 11, 314, 716]])     # "Hello, I am"
print(generate(model, start, max_new_tokens=8, context_size=1024))
```

Three things to notice, all of which matter later:

1. **`logits[:, -1, :]`** — the model computes predictions at *every* position, but generation only uses the last. The rest is discarded work. (Speculative decoding, chapter 08, finds a use for it.)
2. **`idx[:, -context_size:]`** — beyond the context limit, the oldest tokens are dropped. The model has no memory of them.
3. **The whole prefix is recomputed every step.** Token 100 reprocesses tokens 1–99 from scratch. That's `O(n²)` total work for a sequence of length n — and it's precisely what the KV cache eliminates (chapter 08).

Chapter 08 replaces `argmax` with proper sampling strategies.

---

## 5.5 Upgrading to a 2026 architecture

GPT-2 is 2019. Here's what changed, and why — each of these is a small, comprehensible diff on the code above.

| Component | GPT-2 | Modern | Reason |
|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | ~Same quality, less compute |
| Activation / FFN | GELU, 2 matrices, 4× | **SwiGLU**, 3 matrices, ~8/3× | Gating is better per parameter |
| Position | Learned absolute | **RoPE** | Relative, extrapolates, no table |
| Attention | MHA | **GQA** or MLA | Shrinks the KV cache |
| Biases | Present | **Removed** | No measurable benefit |
| Dropout | 0.1 | **0.0** | Huge corpora don't overfit |
| FFN type | Dense | Dense **or MoE** | Capacity without compute |

### RMSNorm

Drop the mean subtraction and the shift. Just divide by root-mean-square and scale:

▶️
```python
import torch, torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)
```

Fewer operations, no mean pass, half the parameters. Quality is indistinguishable in practice — a rare free win.

### SwiGLU

Replace the 2-matrix FFN with a **gated** 3-matrix version, where one branch modulates the other elementwise:

▶️
```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim=None):
        super().__init__()
        # ~8/3 × instead of 4×, keeping the parameter count comparable to a 2-matrix 4× FFN
        hidden_dim = hidden_dim or int(8 * emb_dim / 3)
        self.gate = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.up   = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, emb_dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

The gate learns *which* features to let through for this particular token — a content-dependent filter. Empirically better than a plain FFN at equal parameters, which is why every modern model uses it.

### Grouped-Query Attention

The problem: at inference, keys and values for every previous token are cached (chapter 08). That cache grows linearly with sequence length **and** with the number of heads, and it quickly dwarfs the model weights.

The fix: let several query heads **share** one key/value head.

```
MHA:  32 query heads, 32 K/V heads     ← cache = 32 units
GQA:  32 query heads,  8 K/V heads     ← cache =  8 units   (4× smaller)
MQA:  32 query heads,  1 K/V head      ← cache =  1 unit    (32× smaller, quality cost)
```

▶️ **The core mechanic — repeating K/V across a group:**

```python
import torch

def repeat_kv(x, n_rep):
    """(b, n_kv_heads, seq, head_dim) -> (b, n_kv_heads*n_rep, seq, head_dim)"""
    b, n_kv, seq, hd = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, None, :, :]
             .expand(b, n_kv, n_rep, seq, hd)
             .reshape(b, n_kv * n_rep, seq, hd))

keys = torch.randn(1, 8, 128, 64)          # 8 K/V heads
print(repeat_kv(keys, n_rep=4).shape)      # (1, 32, 128, 64) — matches 32 query heads
```

Only the 8 real heads are stored in the cache; the expansion happens on the fly. That's the entire saving.

### Mixture of Experts

**The tension:** quality grows with parameters, but so does compute per token. MoE breaks the link.

Replace the single FFN with **N expert FFNs plus a router**. For each token, the router picks the top-k experts (typically 1–8 out of dozens or hundreds). Only those run.

```
Dense:  every token → the one FFN                    (all params, all tokens)
MoE:    every token → router → 2 of 64 experts       (all params exist, 3% used)
```

A model with 235B total parameters might activate only 22B per token — roughly 22B-model speed at much-better-than-22B quality.

▶️ **A working MoE layer:**

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, emb_dim, n_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(emb_dim, n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(emb_dim) for _ in range(n_experts)
        ])

    def forward(self, x):
        b, seq, d = x.shape
        x_flat = x.view(-1, d)

        logits = self.router(x_flat)                              # (tokens, n_experts)
        weights, indices = torch.topk(logits.softmax(-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)     # renormalize

        out = torch.zeros_like(x_flat)
        for slot in range(self.top_k):
            for e_id, expert in enumerate(self.experts):
                mask = indices[:, slot] == e_id
                if mask.any():
                    out[mask] += weights[mask, slot, None] * expert(x_flat[mask])
        return out.view(b, seq, d)

layer = MoEFeedForward(256, n_experts=8, top_k=2)
print(layer(torch.randn(2, 10, 256)).shape)                       # (2, 10, 256)
print(f"{sum(p.numel() for p in layer.parameters()):,} total params")
print("...but only 2 of 8 experts run per token")
```

**The trap that catches everyone:** MoE saves **compute**, not **memory**. All experts must be resident in VRAM even though most are idle for any given token. A "3B active / 30B total" model needs 30B worth of memory. Never size hardware from the active count.

**The hard parts in practice:**
- **Load balancing.** Routers collapse onto favorite experts if unchecked, wasting capacity. Older fixes add an auxiliary loss; newer designs adjust router biases directly so the main objective isn't distorted.
- **Communication.** Experts live on different GPUs, so routing implies network traffic every layer. This is MoE's main systems bottleneck.
- **Shared experts.** Some designs always route through one common expert, so the specialized ones can genuinely specialize.

### Efficient attention and hybrids

The last frontier is the quadratic score matrix itself:

- **Sliding-window attention** — attend only to the last *w* tokens. Linear cost, bounded cache, loses long-range recall.
- **Linear attention** — algebraic reformulations that never build the `(seq, seq)` matrix, giving O(n) cost and a fixed-size recurrent state. Gated variants (Gated DeltaNet and relatives) closed most of the historical quality gap.
- **State-space models (Mamba)** — recurrent models with input-dependent transitions: constant memory per token, **no KV cache at all**.

The winning 2026 pattern is **hybrid**: mostly linear/SSM layers for cheap long-context throughput, with a minority of full-attention layers preserving precise recall. Think of these as *attention-budgeting* architectures — they spend exact quadratic attention only where recall demands it.

---

## 5.6 Sizing models, and what the numbers mean

▶️ **Compute any GPT-family size from its config:**

```python
def gpt_params(vocab, ctx, d, n_layers, tied=True):
    tok = vocab * d
    pos = ctx * d
    attn = 4 * d * d                      # W_q, W_k, W_v, out_proj
    ffn = 8 * d * d                       # 4d expansion, up and down
    norms = 4 * d
    per_block = attn + ffn + norms
    total = tok + pos + n_layers * per_block + 2 * d
    if not tied:
        total += vocab * d
    return total

for name, (d, L, H) in {
    "GPT-2 small":  (768, 12, 12),
    "GPT-2 medium": (1024, 24, 16),
    "GPT-2 large":  (1280, 36, 20),
    "GPT-2 XL":     (1600, 48, 25),
}.items():
    n = gpt_params(50257, 1024, d, L)
    print(f"{name:14s} d={d:5d} layers={L:3d}  {n/1e6:7.1f}M params  "
          f"{n*4/1e9:5.2f} GB fp32  {n*2/1e9:5.2f} GB bf16")
```

**Memory rules of thumb** worth memorizing:

```
Inference:  ~2 bytes/param (bf16)     →  7B model ≈ 14 GB   + KV cache
            ~0.5 bytes/param (int4)   →  7B model ≈  4 GB
Training:   ~16 bytes/param           →  7B model ≈ 112 GB
            (4 weights + 4 gradients + 8 AdamW optimizer state)
```

That 8× gap between inference and training memory is why finetuning needs LoRA (chapter 07) and why serving is a completely different engineering problem from training.

---

## Check yourself

1. Why must `emb_dim` stay constant through every block?
2. Which holds more parameters, attention or the FFN? By how much?
3. Explain weight tying and why 163M is marketed as 124M.
4. Why does pre-LN train more stably than post-LN?
5. A "30B total, 3B active" MoE — how much VRAM to serve it in bf16?
6. What does the gate in SwiGLU actually do?
7. In the generation loop, why is only `logits[:, -1, :]` used?

<details>
<summary>Answers</summary>

1. Residual connections add the sublayer output to the input; the shapes must match, which forces a constant width.
2. The FFN: `8d²` versus attention's `4d²` — twice as many, about two-thirds of the block.
3. The token embedding matrix is reused as the output head. Counted once instead of twice, 163,009,536 − 38,597,376 = 124,412,160.
4. Pre-LN leaves the residual path untouched, so gradients reach early layers without passing through normalization at every step.
5. ~60 GB for weights (30B × 2 bytes) plus KV cache. The 3B active number affects speed, not memory.
6. It produces a content-dependent multiplicative filter, letting the layer decide which hidden features pass through for this specific token.
7. Every position produces a prediction, but when generating you only need the prediction that follows the final token; the others predict tokens you already have.
</details>

---

**Next:** [06 — Pretraining](06-pretraining.md)
