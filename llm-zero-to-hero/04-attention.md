# 04 — Attention

*The mechanism the entire field is built on. Budget more time for this chapter than any other.*

---

## 4.1 The problem attention solves

Consider: **"The animal didn't cross the street because it was too tired."**

What does "it" refer to? The animal. Now change one word: **"...because it was too wide."** Now "it" is the street.

To represent "it" correctly, the model must look at other words in the sentence and decide which ones matter. Not all equally — "animal" and "street" matter enormously, "because" barely at all. And which matters more depends on the rest of the sentence.

**That's attention: each token builds its representation by pulling in information from other tokens, weighted by relevance that is computed from content.**

The older approach (RNNs) forced all context through a single hidden state passed left to right, so information about "animal" had to survive six sequential updates before reaching "it," degrading along the way. Attention lets "it" read "animal" *directly*, at any distance, in one step.

---

## 4.2 Building attention from scratch, four steps

The mechanism is usually presented all at once with three matrices and a scaling factor, which is why it seems arbitrary. Built up in stages, every piece is forced by the previous stage's problem.

### Step 1 — Similarity as a dot product (no learning yet)

Start with the simplest possible idea: let each token attend to others in proportion to how similar their embeddings are.

Take three tokens with 2-dimensional embeddings (tiny, so we can do the arithmetic by hand):

```
x₁ = [1, 0]      ("the")
x₂ = [0, 1]      ("cat")
x₃ = [1, 1]      ("sat")
```

Compute all pairwise dot products — every token against every token:

```
        x₁     x₂     x₃
x₁ ·   1.0    0.0    1.0
x₂ ·   0.0    1.0    1.0
x₃ ·   1.0    1.0    2.0
```

This `(3, 3)` grid is the **attention score matrix**. Row *i*, column *j* = how much token *i* cares about token *j*.

Now softmax each **row** to turn scores into weights that sum to 1, then use those weights to average the value vectors. For row 3 (`[1.0, 1.0, 2.0]`):

```
e^1.0 = 2.718,  e^1.0 = 2.718,  e^2.0 = 7.389     sum = 12.825
weights = [0.212, 0.212, 0.576]

context₃ = 0.212·[1,0] + 0.212·[0,1] + 0.576·[1,1] = [0.788, 0.788]
```

Token 3's new representation is a blend of all three tokens, weighted by similarity. **This already works** — but it has no learnable parameters, so the model can't adjust what "relevant" means for a task.

### Step 2 — Queries, Keys, Values (adding learning)

The fix: don't compare raw embeddings. Project each token through **three separate learned matrices** first.

| Projection | Question it answers | Used for |
|---|---|---|
| **Query** (Q) | "What am I looking for?" | The row token, doing the looking |
| **Key** (K) | "What do I offer?" | The column token, being looked at |
| **Value** (V) | "What do I actually contribute?" | The content that gets mixed in |

The database analogy is the one that sticks: you have a **query**, you match it against **keys**, and you retrieve the corresponding **values** — except softly, as a weighted blend over all entries rather than an exact hit on one.

Why three matrices instead of one? Because *what a token seeks* and *what it offers* are different things. The token "it" seeks a noun; the token "animal" offers being a noun. Separating Q from K lets the model learn asymmetric relationships. And V is separate again because *what makes a token findable* need not be *what makes it useful once found*.

```
Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores  = Q @ Kᵀ
weights = softmax(scores)
output  = weights @ V
```

`W_q`, `W_k`, `W_v` are learned. This is now a trainable mechanism.

### Step 3 — Scaling by √d_k

One more correction, and it matters more than it looks.

As the dimension `d_k` grows, dot products grow with it — sum more terms, get bigger numbers. Large scores push softmax into saturation: one weight ≈ 1.0, everything else ≈ 0.0. In that regime the gradient is nearly zero and **learning stalls completely**.

See it concretely:

```
softmax([1, 2, 3])       = [0.090, 0.245, 0.665]     healthy
softmax([10, 20, 30])    = [0.000, 0.000, 1.000]     saturated — no gradient
```

Dividing by `√d_k` keeps the variance of the scores around 1 regardless of dimension:

```
weights = softmax(Q @ Kᵀ / √d_k)
```

Hence the full name: **scaled dot-product attention**. `d_k` is the key dimension — 64 in most models.

### Step 4 — Causal masking

A language model predicting position 3 must not see positions 4, 5, 6. If it can, it just copies the answer and learns nothing useful — training loss collapses to near zero and generation produces garbage.

The fix: before softmax, set every score where `column > row` to `-∞`.

```
raw scores:              masked:
[1.0  0.0  1.0]          [1.0  -inf  -inf]
[0.0  1.0  1.0]    →     [0.0   1.0  -inf]
[1.0  1.0  2.0]          [1.0   1.0   2.0]
```

Why `-∞` rather than 0? Because `e^(-∞) = 0` exactly, so masked positions get **zero weight after softmax** and the remaining weights still sum to 1 automatically. Zeroing weights *after* softmax would leave them not summing to 1 and require a renormalization hack.

Now row 1 attends only to itself, row 2 to positions 1–2, row 3 to all three. Each position sees only its past.

---

## 4.3 The complete worked example

Let's do it end to end with numbers you can verify on paper. Same three tokens, and to keep the arithmetic clean we take `W_q = W_k = W_v = I` (identity), so `Q = K = V = X`.

```
x₁ = [1, 0]     x₂ = [0, 1]     x₃ = [1, 1]        d_k = 2, √d_k = 1.414
```

**Scores** (`Q @ Kᵀ`), then **scaled** (÷1.414), then **masked**:

```
raw:            scaled:              masked:
[1  0  1]       [0.707  0     0.707]  [0.707  -inf   -inf ]
[0  1  1]   →   [0      0.707 0.707]  [0      0.707  -inf ]
[1  1  2]       [0.707  0.707 1.414]  [0.707  0.707  1.414]
```

**Softmax each row:**

Row 1 — only one live entry, so weight `[1.000, 0, 0]`.

Row 2 — `e^0 = 1.000`, `e^0.707 = 2.028`, sum = 3.028:
```
[0.330, 0.670, 0]
```

Row 3 — `e^0.707 = 2.028`, `e^0.707 = 2.028`, `e^1.414 = 4.113`, sum = 8.169:
```
[0.248, 0.248, 0.503]
```

**Multiply by V:**

```
context₁ = 1.000·[1,0]                                    = [1.000, 0.000]
context₂ = 0.330·[1,0] + 0.670·[0,1]                      = [0.330, 0.670]
context₃ = 0.248·[1,0] + 0.248·[0,1] + 0.503·[1,1]        = [0.751, 0.751]
```

Read the result: token 1 could only see itself. Token 2 mixed in a third of token 1. Token 3 leaned mostly on itself (0.503) with equal contributions from the earlier two. **That's attention.** Everything else in this chapter is engineering on top.

▶️ **Verify it in code:**

```python
import torch

X = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
d_k = X.shape[-1]

scores = X @ X.T / d_k**0.5
mask = torch.triu(torch.ones(3, 3), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
weights = torch.softmax(scores, dim=-1)
context = weights @ X

print(weights.round(decimals=3))
print(context.round(decimals=3))
# weights: [[1.000, 0.000, 0.000],
#           [0.330, 0.670, 0.000],
#           [0.248, 0.248, 0.503]]
```

The numbers match the hand calculation exactly. If you do one thing from this chapter, do this — hand-computing attention once makes it permanently concrete.

---

## 4.4 Multi-head attention

One attention operation learns **one** notion of relevance. But language has many simultaneous structures: grammatical agreement, coreference, topical association, positional locality.

The solution: run several attention operations in parallel, each with its own `W_q`, `W_k`, `W_v`, on different **slices** of the representation. Each is a **head**. Interpretability research has found heads that specialize in identifiable ways — some track the previous token, some match syntactic dependencies, some copy earlier occurrences of a token ("induction heads").

**The efficient implementation doesn't loop over heads.** It computes one big projection and *reshapes*:

```
project to d_out=768        →  (batch, seq, 768)
view as heads               →  (batch, seq, 12, 64)
transpose                   →  (batch, 12, seq, 64)   ← 12 independent attentions
...attention happens here, batched across the head dimension...
transpose back and merge    →  (batch, seq, 768)
output projection           →  (batch, seq, 768)
```

Heads are a **view of one tensor**, not separate modules. `head_dim = d_out // n_heads` — which is why `d_out` must be divisible by `n_heads`.

The final `out_proj` matters: it lets the model mix information *across* heads, which is otherwise impossible since heads never see each other.

### The shape table — internalize this

For `batch=2, seq=6, d_in=d_out=768, n_heads=12` → `head_dim=64`:

| Operation | Resulting shape |
|---|---|
| input `x` | `(2, 6, 768)` |
| `W_query(x)` | `(2, 6, 768)` |
| `.view(2, 6, 12, 64)` | `(2, 6, 12, 64)` |
| `.transpose(1, 2)` | `(2, 12, 6, 64)` |
| `queries @ keys.transpose(2,3)` | `(2, 12, 6, 6)` ← **the attention matrix** |
| after mask + softmax | `(2, 12, 6, 6)` |
| `weights @ values` | `(2, 12, 6, 64)` |
| `.transpose(1,2).contiguous().view(2, 6, 768)` | `(2, 6, 768)` |
| `out_proj(...)` | `(2, 6, 768)` |

Stare at `(2, 12, 6, 6)`. That's `seq × seq` — **attention is quadratic in sequence length**. Double the context, quadruple this matrix. Every long-context innovation in the field is an attack on that square, and now you can see exactly why.

---

## 4.5 The full implementation

▶️ **Complete, runnable, production-shaped multi-head causal self-attention:**

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0,
                 num_heads=12, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # a buffer: moves with .to(device), saved in state_dict, but NOT a parameter
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape

        # 1. project into queries, keys, values
        queries = self.W_query(x)      # (b, num_tokens, d_out)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # 2. split into heads: (b, tokens, d_out) -> (b, heads, tokens, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. scaled dot-product scores: (b, heads, tokens, tokens)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores / keys.shape[-1] ** 0.5

        # 4. causal mask (truncated to this sequence's length)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        # 5. softmax + dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. weighted sum of values
        context = attn_weights @ values                     # (b, heads, tokens, head_dim)

        # 7. merge heads back together
        context = context.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context)


torch.manual_seed(123)
mha = MultiHeadAttention(d_in=768, d_out=768, context_length=1024, num_heads=12)
x = torch.randn(2, 6, 768)
print(mha(x).shape)                                          # (2, 6, 768)
print(f"{sum(p.numel() for p in mha.parameters()):,} params")  # 2,360,064
```

**Two details worth understanding rather than copying:**

**`register_buffer`.** The mask is a tensor the module needs, but it is *not learned*. Registering it as a buffer means it moves to GPU with `.to(device)` and is saved with the model, while receiving no gradients and never appearing in `parameters()`. Using a plain attribute would leave it on the CPU and crash on GPU; using `nn.Parameter` would have the optimizer trying to train a constant.

**Dropout on attention weights.** During training, randomly zeroing some attention weights prevents the model from over-relying on any single connection. GPT-2 uses 0.1; modern large-scale pretraining usually uses 0.0 because the datasets are large enough that overfitting isn't the constraint.

---

## 4.6 Fused attention: same math, much faster

The implementation above materializes the full `(b, heads, seq, seq)` matrix in memory. At 32k context that's enormous, and — critically — it means writing a huge tensor to GPU memory and reading it back.

**FlashAttention** computes mathematically identical results without ever materializing that matrix: it tiles the computation, keeping blocks in fast on-chip SRAM and accumulating the softmax incrementally. Same FLOPs, dramatically less memory traffic — and since attention is memory-bandwidth-bound, that's a 2–4× speedup plus a large memory saving.

You get it for free:

▶️
```python
import torch
import torch.nn.functional as F

q = torch.randn(2, 12, 512, 64)
k = torch.randn(2, 12, 512, 64)
v = torch.randn(2, 12, 512, 64)

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(out.shape)     # (2, 12, 512, 64)
```

One line replaces steps 3–6 above and dispatches to the best available kernel. **Use this in real code.** Write the manual version once to understand it, then never again.

The lesson generalizes: in this field, the big wins often come from *how data moves through memory*, not from reducing arithmetic.

---

## 4.7 Attention variants you'll meet

Chapter 05 covers these in architectural context, but here's the map, all driven by one pressure — the quadratic score matrix and the linearly-growing KV cache:

| Variant | Change | Motivation |
|---|---|---|
| **MHA** | Every head has its own K and V | The baseline (this chapter) |
| **MQA** | All heads share ONE K/V head | Shrinks KV cache massively; some quality loss |
| **GQA** | Groups of heads share K/V | The modern default — most of MQA's savings, little quality loss |
| **MLA** | K/V compressed into a low-rank latent | Smaller still; DeepSeek's approach |
| **Sliding window** | Attend only to the last *w* tokens | Linear cost; loses long-range recall |
| **Linear attention** | Reformulate to avoid the seq×seq matrix | O(n) cost, fixed-size state |

All of them keep the query–key–value structure you just learned. They only change *which* keys and values are available and *how many* are stored.

---

## Check yourself

1. Why three separate projection matrices instead of comparing embeddings directly?
2. What exactly goes wrong without the `√d_k` scaling? Be mechanical.
3. Why mask with `-inf` before softmax rather than zeroing weights after?
4. Why is the causal mask a buffer rather than a parameter?
5. `d_out=512`, `n_heads=8` — what is `head_dim`? What if `n_heads=7`?
6. Where precisely does the quadratic cost appear?
7. FlashAttention does the same number of FLOPs. Why is it faster?

<details>
<summary>Answers</summary>

1. What a token *seeks* and what it *offers* are different, and what it offers as a match key need not be what it contributes as content. Three matrices let the model learn all three roles independently.
2. Dot products grow with dimension; large scores saturate softmax into a near one-hot distribution, where gradients vanish and learning stops.
3. `e^(-inf) = 0` exactly, so masked entries get zero weight and the surviving weights still sum to 1 automatically — no renormalization needed.
4. It's needed by the module and must move with it across devices and be saved, but it is a constant, not something to learn.
5. 64. With `n_heads=7`, 512 isn't divisible by 7 — the assertion fires.
6. The `Q @ Kᵀ` score matrix, shape `(seq, seq)`, plus the `weights @ V` multiply against it.
7. It avoids materializing the `(seq, seq)` matrix in slow GPU memory, tiling the computation in fast SRAM instead. Attention is bandwidth-bound, so cutting memory traffic cuts time.
</details>

---

**Next:** [05 — The Transformer](05-transformer-architecture.md)
