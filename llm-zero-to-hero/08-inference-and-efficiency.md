# 08 — Inference & Efficiency

*Sampling strategies, the KV cache, quantization, and why serving is a completely different engineering problem from training.*

---

## 8.1 Decoding: turning logits into text

The model gives you 50,257 scores. How you pick one changes the output enormously — often more than switching models.

### Greedy decoding

Always take the highest-probability token. Deterministic, and reliably boring: it produces loops (`"the best way to do this is to do this is to do this"`) because once a repetitive pattern becomes locally most-likely, nothing breaks it.

### Temperature

Divide the logits by `T` before softmax.

```
T < 1   sharpens   → conservative, more likely to repeat
T = 1   unchanged  → the model's actual distribution
T > 1   flattens   → more random, eventually incoherent
T → 0   approaches greedy
```

▶️ **See exactly what temperature does:**

```python
import torch

logits = torch.tensor([5.0, 4.0, 3.0, 1.0, 0.5])

for T in [0.1, 0.5, 1.0, 1.5, 2.0]:
    probs = torch.softmax(logits / T, dim=0)
    print(f"T={T:4.1f}  {[f'{p:.3f}' for p in probs]}")

# T= 0.1  ['1.000', '0.000', '0.000', '0.000', '0.000']   ← effectively greedy
# T= 1.0  ['0.616', '0.227', '0.083', '0.011', '0.007']
# T= 2.0  ['0.384', '0.233', '0.141', '0.052', '0.040']   ← tail becomes reachable
```

**Temperature is not a "creativity" dial** — it's the flatness of the distribution you sample from. At high T the model isn't more imaginative; it's more willing to pick tokens it believes are wrong.

### Top-k sampling

Keep only the k highest-probability tokens, zero the rest, renormalize, sample. This prevents the long tail of 50,000 implausible tokens from ever being drawn, which is what high temperature alone would allow.

### Top-p (nucleus) sampling

Keep the smallest set of tokens whose cumulative probability exceeds p (typically 0.9). **The cutoff adapts to the model's confidence** — where the model is certain, few tokens qualify; where it's unsure, many do. Top-k can't do this: a fixed k=50 is far too permissive after `"The capital of France is"` and possibly too narrow mid-story.

▶️ **All strategies, implemented:**

```python
import torch

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits.clone()

    if top_k is not None:
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = torch.where(logits < kth, torch.tensor(float("-inf")), logits)

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


torch.manual_seed(0)
logits = torch.randn(1, 100)
print("greedy   :", sample_next(logits, temperature=0).item())
print("T=0.8    :", sample_next(logits, temperature=0.8).item())
print("top-k=10 :", sample_next(logits, temperature=1.0, top_k=10).item())
print("top-p=0.9:", sample_next(logits, temperature=1.0, top_p=0.9).item())
```

### What to actually use

| Task | Settings |
|---|---|
| Factual Q&A, extraction, classification | `temperature=0` (greedy) |
| Code generation | `temperature=0` to `0.2` |
| General assistant | `temperature=0.7, top_p=0.9` |
| Creative writing | `temperature=0.9–1.1, top_p=0.95` |
| Sampling for self-consistency | `temperature=0.8`, several samples |

**Repetition penalties** downweight already-used tokens to break loops. Use sparingly — aggressive penalties damage code and any text that legitimately repeats.

**Structured/constrained decoding** masks logits against a grammar or JSON schema so output is *guaranteed* parseable. If you need JSON, use this rather than asking politely and retrying — it eliminates an entire class of production bugs.

---

## 8.2 The KV cache: the single most important optimization

Look again at the naive generation loop from chapter 05. To generate token 101, it feeds all 100 previous tokens through the model and throws away 100 of the 101 predictions.

But here's the thing: **because of causal masking, the keys and values for tokens 1–100 don't change when token 101 arrives.** Token 50's key vector is identical whether the sequence is 100 or 1000 tokens long. Recomputing it is pure waste.

So cache them.

```
WITHOUT cache, generating token n:
    process n tokens → O(n) work per token → O(n²) total

WITH cache, generating token n:
    process 1 token, attend to n cached K/V → O(n) work per token → O(n) amortized
```

▶️ **Attention with a KV cache:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache=None):
        b, seq, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v)]

        if cache is not None:
            past_k, past_v = cache
            k = torch.cat([past_k, k], dim=2)      # append along the sequence axis
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        # causal mask only needed when processing more than one token at a time
        out = F.scaled_dot_product_attention(q, k, v, is_causal=(seq > 1))
        out = out.transpose(1, 2).contiguous().view(b, seq, d)
        return self.proj(out), new_cache


torch.manual_seed(0)
attn = CachedAttention(256, 8)

# prefill: process the whole prompt at once
prompt = torch.randn(1, 10, 256)
out, cache = attn(prompt)
print("after prefill, cached K:", cache[0].shape)     # (1, 8, 10, 32)

# decode: one token at a time, reusing the cache
for _ in range(3):
    out, cache = attn(torch.randn(1, 1, 256), cache)
print("after 3 decode steps  :", cache[0].shape)      # (1, 8, 13, 32)
```

### The cost of the cache

```
KV cache bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × bytes_per_value
```

▶️
```python
def kv_cache_gb(n_layers, n_kv_heads, head_dim, seq_len, batch, bytes_per=2):
    return 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * bytes_per / 1e9

print(f"GPT-2 124M, 1k ctx, 1 seq : {kv_cache_gb(12, 12, 64, 1024, 1)*1000:.1f} MB")
print(f"7B MHA,   32k ctx, 1 seq  : {kv_cache_gb(32, 32, 128, 32768, 1):.1f} GB")
print(f"7B GQA-8, 32k ctx, 1 seq  : {kv_cache_gb(32,  8, 128, 32768, 1):.1f} GB")
print(f"7B GQA-8, 32k ctx, 32 seq : {kv_cache_gb(32,  8, 128, 32768, 32):.1f} GB")
```

Run that and the architecture chapter clicks retroactively. At 32k context with full MHA, the cache alone exceeds the model weights. **GQA cuts it 4×.** With 32 concurrent users it's the entire GPU. This is why GQA/MLA exist, why long context is expensive, and why "supports 1M tokens" deserves the follow-up question "at what batch size?"

---

## 8.3 Prefill vs decode: two different workloads

```
PREFILL                              DECODE
Process the whole prompt at once     Generate one token at a time
Parallel across all positions        Strictly sequential
COMPUTE-bound                        MEMORY-BANDWIDTH-bound
Determines time-to-first-token       Determines tokens/second
```

**The decode phase reads every model weight from memory to produce a single token.** A 14 GB model on a GPU with 2 TB/s bandwidth caps out around 140 tokens/sec for one sequence — *regardless of how fast the GPU's arithmetic is*.

Three consequences that explain most of serving:

1. **Quantization speeds up decoding** — fewer bytes to move, not fewer operations.
2. **Batching is nearly free per user.** The weights get read once for the whole batch, so 32 users cost barely more than 1. This is why throughput and cost-per-token are so batch-dependent.
3. **Prefill and decode belong on different schedules**, which is why production systems sometimes run them on separate hardware pools (*disaggregated serving*).

**The latency metrics** you'll be asked about:

| Metric | Meaning | Driven by |
|---|---|---|
| **TTFT** | Time to first token | Prefill: prompt length, compute |
| **TPOT / ITL** | Time per output token | Decode: memory bandwidth, batch size |
| **Throughput** | Total tokens/sec across all users | Batching efficiency |

---

## 8.4 Quantization

Store weights in fewer bits. Since decode is bandwidth-bound, halving the bytes nearly halves the time.

| Precision | Bytes/param | 7B model | Quality |
|---|---|---|---|
| fp32 | 4 | 28 GB | Baseline (unnecessary for inference) |
| bf16 | 2 | 14 GB | Standard |
| int8 | 1 | 7 GB | Very small loss |
| int4 | 0.5 | 3.5 GB | Small but measurable |
| int2 | 0.25 | 1.75 GB | Significant without QAT |

▶️ **Quantization, from scratch — it's simpler than it sounds:**

```python
import torch

def quantize_int8(w):
    scale = w.abs().max() / 127
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize_int8(q, scale):
    return q.float() * scale

torch.manual_seed(0)
w = torch.randn(1000, 1000)
q, scale = quantize_int8(w)
w_hat = dequantize_int8(q, scale)

print(f"original : {w.numel() * 4 / 1e6:.1f} MB")
print(f"quantized: {q.numel() * 1 / 1e6:.1f} MB")
print(f"mean abs error: {(w - w_hat).abs().mean():.6f}")
print(f"max  abs error: {(w - w_hat).abs().max():.6f}")
```

That's the whole idea: find a scale factor, round to integers, store the integers. Real methods (GPTQ, AWQ) improve on it by quantizing per-group rather than per-tensor and by using calibration data to protect the weights that matter most.

**Outlier features** are why naive quantization fails at low bit widths: a handful of activation channels have magnitudes 100× the rest, and a single global scale destroys everything else's precision. Modern methods handle those channels separately.

**The practical rule:** for a fixed memory budget, **a larger model at int4 usually beats a smaller model at bf16**. But verify on *your* task — quantization damage is uneven, and long-context recall and multi-step reasoning degrade first.

---

## 8.5 Speculative decoding

A clever trick that exploits the asymmetry between prefill and decode.

**The observation:** verifying k tokens costs one forward pass (parallel, like prefill), while *generating* k tokens costs k forward passes (sequential).

```
1. A small, fast DRAFT model proposes the next k tokens (cheap)
2. The large model verifies all k in ONE forward pass
3. Accept the longest correct prefix; resample at the first disagreement
```

If the draft is right 70% of the time, you get ~2–3× speedup. And because of how the acceptance step is constructed, **the output distribution is provably identical to the big model's** — this is not an approximation.

▶️ **The mechanism, simplified:**

```python
import torch

def speculative_step(draft_tokens, target_probs, draft_probs):
    """Accept draft tokens while the target model agrees; stop at the first rejection."""
    accepted = []
    for i, tok in enumerate(draft_tokens):
        p_target, p_draft = target_probs[i][tok], draft_probs[i][tok]
        if torch.rand(1).item() < min(1.0, (p_target / p_draft).item()):
            accepted.append(tok)
        else:
            break                       # reject here; resample from the corrected distribution
    return accepted

torch.manual_seed(0)
draft = [101, 205, 33, 4001]
target_probs = [torch.softmax(torch.randn(5000), 0) for _ in range(4)]
draft_probs  = [torch.softmax(torch.randn(5000), 0) for _ in range(4)]
print("accepted:", speculative_step(draft, target_probs, draft_probs))
```

**Self-speculative variants** (Medusa, EAGLE) skip the separate draft model by adding extra heads to the main model that predict several tokens ahead — the same idea with less operational complexity.

---

## 8.6 Serving systems

Everything above shows up in production engines. The four ideas that matter:

**Continuous batching.** Naive batching waits for the slowest sequence in the batch to finish, leaving GPUs idle. Continuous batching swaps completed sequences out and new requests in every step. Often a 2–4× throughput gain for free.

**Paged attention.** The KV cache is managed in fixed-size pages, like operating-system virtual memory, instead of one contiguous block per sequence. This eliminates fragmentation (you don't reserve max-length memory for every request) and allows sharing pages between sequences. It's the core idea in **vLLM**.

**Prefix caching.** If 1,000 requests share the same 2,000-token system prompt, compute its KV cache **once** and reuse it. Frequently the single largest cost win in a real application — and it's usually one configuration flag.

**Disaggregated serving.** Run prefill and decode on separate pools, since one is compute-bound and the other bandwidth-bound.

**The landscape:**

| Tool | Use for |
|---|---|
| **vLLM** | The general production default — paged attention, continuous batching |
| **SGLang** | Similar, strong on structured output and complex prompt reuse |
| **TensorRT-LLM** | Maximum performance on NVIDIA, more setup effort |
| **llama.cpp** | CPU and consumer hardware; GGUF quantized formats |
| **Ollama** | Local development, wraps llama.cpp with a friendly interface |

---

## 8.7 A cost model you can reason with

▶️
```python
def serving_estimate(params_b, seq_len, batch, n_layers, n_kv_heads, head_dim,
                     bytes_per_param=2, gpu_bandwidth_tbs=2.0, gpu_mem_gb=80):
    weights = params_b * 1e9 * bytes_per_param / 1e9
    kv = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * 2 / 1e9
    # decode is bandwidth-bound: one full weight read per token
    tok_per_sec = gpu_bandwidth_tbs * 1e12 / (params_b * 1e9 * bytes_per_param)

    print(f"weights        : {weights:6.1f} GB")
    print(f"KV cache       : {kv:6.1f} GB  (batch={batch}, ctx={seq_len})")
    print(f"total          : {weights + kv:6.1f} GB / {gpu_mem_gb} GB available")
    print(f"fits           : {'yes' if weights + kv < gpu_mem_gb else 'NO'}")
    print(f"decode speed   : ~{tok_per_sec:.0f} tok/s per sequence")
    print(f"batch throughput: ~{tok_per_sec * batch:.0f} tok/s aggregate")

serving_estimate(params_b=7, seq_len=8192, batch=16,
                 n_layers=32, n_kv_heads=8, head_dim=128)
```

This back-of-envelope is genuinely how capacity planning starts. Change `bytes_per_param` to 0.5 (int4) and watch both the memory and the speed improve — that's the quantization argument in one number.

---

## Check yourself

1. Why does top-p adapt better than top-k?
2. Why does the KV cache work at all — what property of causal attention makes it valid?
3. Compute the KV cache for a 32-layer model, 8 KV heads, head_dim 128, at 16k context, batch 8, bf16.
4. Why does quantization speed up token generation despite doing the same number of operations?
5. Why is speculative decoding not an approximation?
6. Why does batching improve throughput so dramatically?
7. Your app has a 3,000-token system prompt on every request. What's the first optimization?

<details>
<summary>Answers</summary>

1. Top-p's cutoff moves with the model's confidence — narrow when it's sure, wide when it isn't — while a fixed k is simultaneously too loose in confident contexts and too tight in uncertain ones.
2. Causal masking means a token's key and value never depend on later tokens, so once computed they're valid forever.
3. `2 × 32 × 8 × 128 × 16384 × 8 × 2 bytes ≈ 17.2 GB`.
4. Decode is memory-bandwidth-bound: the time is dominated by streaming weights from memory, so fewer bytes per weight directly means less time.
5. The accept/reject rule is constructed so the resulting samples come from exactly the target model's distribution; rejected positions are resampled from a corrected distribution.
6. Model weights are read from memory once per step regardless of batch size, so the dominant cost is amortized across every sequence in the batch.
7. Prefix caching — compute that prompt's KV cache once and reuse it across all requests.
</details>

---

**Next:** [09 — Building Applications](09-building-applications.md)
