# 03 — Embeddings & Position

*Turning token IDs into meaning, and telling the model what order things came in.*

---

## 3.1 Why token IDs can't be fed to a network directly

After tokenization, `"The cat sat"` is `[464, 3797, 3332]`. Why not feed those numbers straight in?

Because the numbers are **arbitrary labels**, not quantities. Token 3797 is not "larger" than token 464 in any meaningful sense, and token 3798 is not "adjacent in meaning" to 3797 — it's whatever word happened to land next in the merge ordering. Feeding raw IDs into a matrix multiply would have the network learning from a numeric structure that encodes nothing.

The fix: give each token a **learned vector**.

---

## 3.2 Embeddings: meaning as geometry

An embedding layer is a lookup table — a matrix of shape `(vocab_size, emb_dim)` where row *i* is the vector for token *i*.

```
Embedding matrix (50257 × 768):

  token 0    [ 0.02, -0.41,  0.88, ... ]   ← 768 numbers
  token 1    [-0.15,  0.33, -0.02, ... ]
  ...
  token 3797 [ 0.71,  0.09, -0.44, ... ]   ← " cat"
  ...
  token 50256[ 0.11,  0.05,  0.23, ... ]
```

Look up row 3797 and you have the model's representation of `" cat"`. These 38.6 million numbers (50257 × 768) are **learned parameters**, adjusted by backpropagation like everything else. They start random and meaningless; training organizes them.

What does "organized" mean? Tokens used in similar contexts end up with similar vectors, because similar vectors produce similar predictions and that's what lowers the loss. Geometry becomes semantics — the `king - man + woman ≈ queen` structure from chapter 00 emerges without anyone designing it.

▶️ **Build one:**

```python
import torch, torch.nn as nn

torch.manual_seed(42)
vocab_size, emb_dim = 50257, 768
embedding = nn.Embedding(vocab_size, emb_dim)

print("table shape:", embedding.weight.shape)     # (50257, 768)
print("parameters :", embedding.weight.numel())   # 38,597,376

token_ids = torch.tensor([[464, 3797, 3332]])     # (batch=1, seq=3)
vectors = embedding(token_ids)
print("output:", vectors.shape)                   # (1, 3, 768)
```

One integer became 768 numbers. That `(batch, seq_len, emb_dim)` shape now persists unchanged through every layer of the model until the very end.

### The lookup-vs-matmul equivalence

An embedding lookup is mathematically identical to one-hot encoding followed by a matrix multiply:

▶️
```python
import torch
import torch.nn.functional as F

emb = torch.nn.Embedding(5, 3)
ids = torch.tensor([2])

lookup = emb(ids)
onehot = F.one_hot(ids, num_classes=5).float() @ emb.weight
print(torch.allclose(lookup, onehot))    # True
```

Identical result — but the one-hot version multiplies a 50257-wide vector of mostly zeros by a huge matrix, wasting essentially all of the work. Indexing is the same operation with the waste removed. Worth knowing because it explains why the embedding layer and the output layer are *transposes of the same idea* (which is what makes weight tying possible, §5.6).

---

## 3.3 The position problem

Here is a fact about attention that will only fully land in chapter 04, but you need it now:

**Self-attention has no notion of order.** It computes relationships between every pair of tokens, and that computation is completely symmetric with respect to position. Shuffle the input tokens and the outputs shuffle identically — the model literally cannot distinguish `"dog bites man"` from `"man bites dog"`.

That's catastrophic for language. So position must be injected explicitly.

### Approach 1: learned absolute positional embeddings (GPT-2)

Make a *second* lookup table, indexed by position instead of by token:

```
Position embedding matrix (1024 × 768):

  position 0  [ 0.31, -0.02, ... ]
  position 1  [-0.11,  0.44, ... ]
  ...
  position 1023 [ ... ]
```

Then simply **add** it to the token embedding:

```
final_input[i] = token_embedding[token_id[i]] + position_embedding[i]
```

▶️ **Complete input pipeline:**

```python
import torch, torch.nn as nn

torch.manual_seed(123)
vocab_size, context_length, emb_dim = 50257, 1024, 768

tok_emb = nn.Embedding(vocab_size, emb_dim)
pos_emb = nn.Embedding(context_length, emb_dim)

token_ids = torch.tensor([[464, 3797, 3332, 319]])       # (1, 4)
batch, seq_len = token_ids.shape

x = tok_emb(token_ids) + pos_emb(torch.arange(seq_len))
print(x.shape)      # (1, 4, 768)  ← ready for the transformer blocks
```

Note the broadcast: `pos_emb(...)` is `(4, 768)` and adds across the batch dimension automatically — every sequence in the batch gets the same positional signal.

**Why addition rather than concatenation?** Concatenating would double the dimension and cost far more compute. Addition works because 768 dimensions is a large space: the network can learn to keep positional and semantic information in effectively separate subspaces and read them out independently.

**The two weaknesses of this approach:**
1. **Hard limit.** There is no row 1024. The model physically cannot process a longer sequence.
2. **Absolute, not relative.** It encodes "I am token #7," but what language cares about is usually "the adjective is 2 tokens before the noun." The model must learn relative structure indirectly, and it doesn't transfer to positions it never trained on.

---

## 3.4 RoPE: the modern answer

**Rotary Position Embedding** is what essentially every model since Llama uses, and it fixes both weaknesses elegantly.

**The idea:** don't add anything to the embeddings. Instead, *rotate* the query and key vectors (chapter 04 introduces those; for now, they're just vectors derived from the token) by an angle proportional to position.

Take a pair of dimensions and treat them as a 2D point. Rotate that point by angle `m·θ` where `m` is the position:

```
position 0:  no rotation
position 1:  rotate by θ
position 2:  rotate by 2θ
position 5:  rotate by 5θ
```

Do this for every pair of dimensions, each with its own frequency θ — high frequencies for fine-grained nearby distinctions, low frequencies for coarse long-range ones.

**Why this is clever:** rotation preserves vector length, and the dot product of two rotated vectors depends only on the *difference* of their rotation angles:

```
rotate(q, m) · rotate(k, n)  =  f(q, k, m − n)
```

Since attention scores are exactly this dot product, **the score automatically depends on relative distance `m − n`**, not on absolute positions. Relative position falls out of the geometry instead of being learned.

Consequences:
- **No position table** — zero parameters.
- **No hard length limit** — the formula is defined for any position.
- **Naturally relative** — which is what language needs.

▶️ **RoPE in ~15 lines:**

```python
import torch

def build_rope_cache(head_dim, max_len, base=10000.0):
    # one frequency per dimension pair: high → fast rotation, low → slow
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)          # (max_len, head_dim/2)
    angles = torch.cat([angles, angles], dim=-1)       # (max_len, head_dim)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    # x: (batch, heads, seq_len, head_dim)
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)             # 90° rotation of each pair
    return x * cos + rotated * sin

# Demonstration: the score depends only on relative distance
head_dim, max_len = 64, 128
cos, sin = build_rope_cache(head_dim, max_len)

q = torch.randn(1, 1, max_len, head_dim)
k = q.clone()
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

# same content, positions (10, 15) vs (60, 65) — both distance 5
print("dist 5 @ pos 10-15:", (qr[0,0,10] @ kr[0,0,15]).item())
print("dist 5 @ pos 60-65:", (qr[0,0,60] @ kr[0,0,65]).item())
```

The two numbers won't be identical here (because `q` has different random content at each position), but swap in identical content vectors and they match exactly — that's the property RoPE guarantees.

### Extending context with RoPE scaling

A model trained at 8k tokens can often be pushed to 128k+ by adjusting RoPE's frequencies so the rotations stretch over a longer range, then briefly finetuning. The recipes have names you'll see in model cards:

| Method | What it does |
|---|---|
| **Linear interpolation** | Divide all positions by a factor — compresses the range. Simple, some quality loss. |
| **NTK-aware scaling** | Scale low frequencies more than high ones, preserving local resolution. |
| **YaRN** | Refined frequency-dependent scaling plus attention-temperature correction. The common production choice. |

When a model card says "extended to 128k with YaRN," this is what happened.

---

## 3.5 Other positional schemes worth recognizing

| Scheme | Idea | Where you'll see it |
|---|---|---|
| **Sinusoidal** | Fixed (unlearned) sine/cosine patterns added to embeddings | Original 2017 transformer |
| **Learned absolute** | A position lookup table | GPT-2, BERT |
| **RoPE** | Rotate Q and K by position | Llama, Qwen, Mistral, almost everything modern |
| **ALiBi** | Subtract a penalty proportional to distance from attention scores | Some long-context models |
| **NoPE** | Nothing at all — causal masking leaks positional information implicitly | Research finding; occasionally used in hybrid stacks |

The trend is clear: from *added parameters* toward *structural* position encoding that generalizes beyond the training length.

---

## 3.6 Embeddings you'll meet elsewhere

Don't confuse the *token embeddings* inside an LLM with **text embeddings** used for search — the vectors you store in a vector database for RAG (chapter 09). Both are learned vectors, but:

| | Token embeddings | Text embeddings |
|---|---|---|
| Represents | One token | A whole sentence/document |
| Produced by | A lookup table inside the model | A separate encoder model, usually bidirectional |
| Used for | Input to the transformer | Similarity search, clustering, retrieval |
| Compared with | Nothing — they're inputs | Cosine similarity |

▶️ **Cosine similarity**, the standard comparison for text embeddings:

```python
import torch.nn.functional as F
import torch

a, b = torch.randn(768), torch.randn(768)
print(F.cosine_similarity(a, b, dim=0).item())     # ≈ 0 for random vectors
print(F.cosine_similarity(a, a, dim=0).item())     # exactly 1.0
```

Cosine rather than raw dot product because it ignores magnitude and compares direction only — you want "same topic," not "longer document."

---

## Check yourself

1. Why can't token IDs be fed into the network directly?
2. How many parameters are in GPT-2 small's token embedding table? Its position table?
3. Why does self-attention need position information injected at all?
4. Name the two weaknesses of learned absolute positions, and how RoPE fixes each.
5. Why does rotating Q and K make attention scores depend on relative distance?

<details>
<summary>Answers</summary>

1. Token IDs are arbitrary labels; their numeric ordering and magnitude carry no meaning, so arithmetic on them is meaningless.
2. Token: 50257 × 768 = 38,597,376. Position: 1024 × 768 = 786,432.
3. Attention is permutation-invariant — it computes pairwise relationships with no inherent notion of order, so "dog bites man" and "man bites dog" would be indistinguishable.
4. (a) A hard maximum length — RoPE has no table, so any position is defined. (b) Absolute rather than relative encoding — RoPE's rotation makes the dot product depend on `m − n`.
5. Because rotation is an orthogonal transformation: the dot product of two rotated vectors depends only on the difference between their rotation angles, and the angles are proportional to position.
</details>

---

**Next:** [04 — Attention](04-attention.md)
