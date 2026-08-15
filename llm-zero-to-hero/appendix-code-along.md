# Chapter-by-Chapter Roadmap

Every section below follows the same structure: **what you're learning → files → the ideas in plain language → terms introduced → what to run → checkpoint questions → exercises → pitfalls**.

Answer the checkpoint questions out loud before moving on. If you cannot, re-read — the chapters are strictly cumulative and a gap in ch03 becomes an impassable wall in ch05.

---

## Stage 0 — Appendix A: PyTorch

**Files:** [`appendix-A/01_main-chapter-code/code-part1.ipynb`](../appendix-A/01_main-chapter-code/code-part1.ipynb), [`code-part2.ipynb`](../appendix-A/01_main-chapter-code/code-part2.ipynb), [`DDP-script.py`](../appendix-A/01_main-chapter-code/DDP-script.py)

### The ideas

PyTorch is three things stacked on each other:

1. **A tensor library.** A tensor is an n-dimensional array that knows what device it lives on (`cpu`, `cuda`, `mps`). Everything is matrix multiplication on tensors.
2. **An automatic differentiation engine.** Every operation on a tensor with `requires_grad=True` is recorded into a graph. Calling `.backward()` walks that graph backwards applying the chain rule, depositing gradients into `.grad`. You will never differentiate by hand.
3. **A neural network toolkit.** `nn.Module` gives you parameter registration, `.to(device)`, `.train()`/`.eval()` modes, and `state_dict()` for saving.

The canonical training loop — memorize this shape, you will write it a dozen times:

```python
for epoch in range(num_epochs):
    for batch_inputs, batch_targets in train_loader:
        optimizer.zero_grad()                      # clear stale gradients
        logits = model(batch_inputs)               # forward pass
        loss = F.cross_entropy(logits, batch_targets)
        loss.backward()                            # backward pass -> fills .grad
        optimizer.step()                           # update weights
```

Forgetting `zero_grad()` silently accumulates gradients across batches and quietly ruins training. It is the single most common PyTorch bug.

### Terms introduced
tensor, rank/dimension, device, CUDA, MPS, autograd, computational graph, backpropagation, chain rule, gradient, `nn.Module`, forward pass, loss function, cross-entropy, optimizer, SGD, Adam/AdamW, learning rate, epoch, batch, mini-batch, `DataLoader`, `Dataset`, train/eval mode, `state_dict`, DDP (distributed data parallel).

### Run
Both notebooks end to end. Then, on a machine with multiple GPUs (or just to read): `DDP-script.py`.

### Checkpoints
- Why does `optimizer.zero_grad()` exist rather than being automatic?
- What does `model.eval()` change? (Answer: dropout off, batchnorm uses running stats — matters in ch05 generation.)
- What's the difference between `loss.backward()` and `optimizer.step()`?

### Pitfalls
- Confusing `.detach()`, `torch.no_grad()`, and `.eval()`. They're three different things: stop gradient flow for one tensor, stop building the graph at all, switch layer behavior.
- Tensor shape mismatches: read the error's expected-vs-got numbers, don't guess.

---

## Stage 1 — Chapter 1: What an LLM is

**Files:** no code — [`ch01/`](../ch01) is prose in the book.

### The ideas

An LLM is a neural network trained on an enormous amount of text with one absurdly simple objective: **predict the next token**. Everything else — the ability to summarize, translate, write code, follow instructions — is an emergent consequence of doing that objective well enough on diverse enough data.

The lifecycle has two stages, and confusing them causes most beginner confusion:

- **Pretraining** (ch05): self-supervised next-token prediction on unlabeled text. Expensive (millions of dollars at frontier scale), produces a *base model* that completes text but doesn't converse.
- **Finetuning** (ch06, ch07): supervised training on much smaller labeled data to specialize the base model — classification (ch06) or instruction-following (ch07). Cheap by comparison.

"Self-supervised" means the labels come free from the data itself: the label for `"The cat sat on the"` is `" mat"`, extracted by shifting the text by one position. No humans annotated anything. That's why the internet-scale corpus is usable.

The transformer architecture (Vaswani et al., 2017) originally had an encoder and a decoder for translation. GPT-style models keep **only the decoder**; BERT-style models keep only the encoder. This book builds a decoder-only model, which is what essentially every generative LLM is today.

### Terms introduced
LLM, transformer, encoder, decoder, decoder-only, autoregressive, next-token prediction, self-supervised learning, pretraining, finetuning, base model, foundation model, emergent ability, parameters/weights, GPT (Generative Pretrained Transformer), zero-shot, few-shot, in-context learning.

### Checkpoints
- Why is next-token prediction "self-supervised" rather than "unsupervised"?
- What can a base model do that an instruction-tuned model can't, and vice versa?
- Why did decoder-only win over encoder-decoder for generative use?

---

## Stage 2 — Chapter 2: Text → numbers

**Files:** [`ch02/01_main-chapter-code/ch02.ipynb`](../ch02/01_main-chapter-code/ch02.ipynb), [`dataloader.ipynb`](../ch02/01_main-chapter-code/dataloader.ipynb), [`the-verdict.txt`](../ch02/01_main-chapter-code/the-verdict.txt)
**Bonus:** [`02_bonus_bytepair-encoder`](../ch02/02_bonus_bytepair-encoder) (BPE implemented from scratch), [`03_bonus_embedding-vs-matmul`](../ch02/03_bonus_embedding-vs-matmul), [`04_bonus_dataloader-intuition`](../ch02/04_bonus_dataloader-intuition)

### The ideas

**Tokenization.** Neural networks consume numbers, not strings. A tokenizer splits text into *tokens* and maps each to an integer ID. Word-level tokenizers explode on vocabulary and choke on unseen words; character-level makes sequences painfully long. **Byte Pair Encoding (BPE)** is the compromise: start from bytes/characters, then repeatedly merge the most frequent adjacent pair into a new token. Common words become one token; rare words decompose into fragments; *nothing is ever out-of-vocabulary*. GPT-2's BPE vocabulary is **50,257** tokens — 50,000 merges + 256 byte-level base tokens + one `<|endoftext|>` special token.

This book uses `tiktoken` (OpenAI's fast BPE) for the main path and implements BPE from scratch in the bonus folder. Do the bonus one — tokenization is where a surprising share of real-world LLM weirdness originates (numbers splitting oddly, whitespace sensitivity, non-English text costing 2–3× more tokens).

**Token embeddings.** An embedding layer is a lookup table: a matrix of shape `(vocab_size, emb_dim)` where row *i* is the vector for token ID *i*. `nn.Embedding` is mathematically identical to one-hot encoding times a weight matrix, but implemented as an indexing operation instead of a wasteful matmul — that's exactly what the `03_bonus_embedding-vs-matmul` notebook demonstrates. These vectors are **learned parameters**, updated by backpropagation like everything else.

**Positional embeddings.** Self-attention is permutation-invariant: by itself it sees a *bag* of tokens, with no notion of order. "dog bites man" and "man bites dog" would be identical. GPT-2 fixes this with a second learned lookup table of shape `(context_length, emb_dim)`, indexed by position, added elementwise to the token embeddings. (Modern models use RoPE instead — see [03 — Embeddings & Position](03-embeddings-and-position.md).)

**The sliding-window dataloader.** Training data is `(input, target)` pairs where the target is the input shifted right by one:

```
input:   [ 40,  367, 2885, 1464]
target:  [367, 2885, 1464, 1807]
```

`max_length` sets the window size, `stride` sets how far the window advances. Stride equal to `max_length` means no overlap and no repeated data; a smaller stride means overlapping windows — more training samples from the same corpus, at the risk of overfitting on repeats.

### Terms introduced
token, token ID, vocabulary, tokenizer, BPE, subword tokenization, special tokens, `<|endoftext|>`, `<|unk|>`, padding, encoding/decoding, embedding, embedding dimension, embedding matrix, lookup table, positional embedding (absolute/learned), context length, input–target pair, sliding window, stride, batch size, tensor shape `(batch, seq_len, emb_dim)`.

### Run
```bash
jupyter lab ch02/01_main-chapter-code/ch02.ipynb
```
Then use the dataloader and print shapes:
```python
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)
inputs, targets = next(iter(dataloader))
print(inputs.shape, targets.shape)   # torch.Size([8, 4]) torch.Size([8, 4])
```

### Checkpoints
- Why doesn't BPE need an `<|unk|>` token?
- What exactly is stored at row 5000 of the embedding matrix, and how does it get there?
- If `max_length=256` and `stride=128`, how many times does a given token appear across the dataset?
- Why must positional and token embeddings share the same `emb_dim`?

### Exercises
1. Tokenize your own name, an emoji, `"1234567890"`, and a sentence in a non-English language. Count tokens. Explain the differences.
2. Set `stride=1` and observe the dataset size explosion.
3. Do [`02_bonus_bytepair-encoder`](../ch02/02_bonus_bytepair-encoder) and implement the merge loop yourself.

### Pitfalls
- Assuming one token ≈ one word. It's roughly ¾ of a word in English, and much worse for other scripts.
- Forgetting that leading whitespace is part of the token: `"hello"` and `" hello"` are different IDs.

---

## Stage 3 — Chapter 3: Attention (the heart of everything)

**Files:** [`ch03/01_main-chapter-code/ch03.ipynb`](../ch03/01_main-chapter-code/ch03.ipynb), [`multihead-attention.ipynb`](../ch03/01_main-chapter-code/multihead-attention.ipynb)
**Bonus:** [`02_bonus_efficient-multihead-attention`](../ch03/02_bonus_efficient-multihead-attention) (benchmarks of MHA implementations), [`03_understanding-buffers`](../ch03/03_understanding-buffers)

**This is the most important chapter in the book. Budget double the time. Do not proceed until you can write `MultiHeadAttention` from an empty file.**

### The ideas, built in four steps

The chapter deliberately builds attention in four escalating versions. Follow that ladder; skipping to the final class is how people end up memorizing code they don't understand.

**1. Simplified attention (no weights).** For each token, compute a dot product between its embedding and every other token's embedding. Dot product = similarity. Softmax those scores into weights summing to 1. Multiply each token's vector by its weight and sum → a *context vector* for that position: a blend of the whole sequence, weighted by relevance. No learnable parameters yet.

**2. Self-attention with trainable weights.** Instead of using raw embeddings, project each token three times through learned matrices:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what do I actually contribute if selected?"

The database analogy is the one that sticks: a query is matched against all keys, and the matching keys' values are retrieved — except softly, as a weighted average over everything rather than an exact lookup.

```
attn_scores  = Q @ K.T                       # (seq, seq) — every token vs every token
attn_weights = softmax(attn_scores / sqrt(d_k))
context      = attn_weights @ V              # (seq, d_out)
```

**Why divide by `sqrt(d_k)`?** As dimension grows, dot products grow in magnitude, pushing softmax into a regime where one weight ≈ 1 and the rest ≈ 0. The gradients there are nearly zero, and learning stalls. Scaling keeps the variance ~1. This is why it's called **scaled** dot-product attention. In the repo: `torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)` in [`gpt.py`](../ch04/01_main-chapter-code/gpt.py).

**3. Causal (masked) attention.** A language model predicting position *t* must not see positions > *t*, or it trivially cheats. Mask the upper triangle of the score matrix with `-inf` **before** softmax — `-inf` becomes exactly 0 after exponentiation, so the weights still sum to 1 without renormalization hacks:

```python
mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(mask_bool, -torch.inf)
```

The mask is created with `torch.triu(torch.ones(ctx, ctx), diagonal=1)` and stored via `register_buffer` — a tensor that moves with `.to(device)` and is saved in `state_dict`, but is *not* a parameter and gets no gradients. That's the whole point of [`03_understanding-buffers`](../ch03/03_understanding-buffers).

Dropout is then applied to the attention weights, randomly zeroing some connections during training to prevent over-reliance on specific positions.

**4. Multi-head attention.** One attention operation learns one kind of relationship. Run `h` of them in parallel on different learned projections and you get `h` relationship types — one head may track syntactic agreement, another coreference, another position-local patterns. The efficient implementation doesn't loop over heads: it computes one big projection to `d_out`, then *reshapes* to `(batch, num_heads, seq_len, head_dim)` where `head_dim = d_out // num_heads`. Heads are a view of a tensor, not separate modules. Finally, heads are concatenated back and passed through an output projection `out_proj`.

### The shape walkthrough (internalize this table)

For `batch=2, seq_len=6, d_in=d_out=768, num_heads=12` → `head_dim=64`:

| Step | Shape |
|---|---|
| input `x` | `(2, 6, 768)` |
| `W_query(x)` → queries | `(2, 6, 768)` |
| `.view(b, n, heads, head_dim)` | `(2, 6, 12, 64)` |
| `.transpose(1, 2)` | `(2, 12, 6, 64)` |
| `queries @ keys.transpose(2,3)` | `(2, 12, 6, 6)` ← the attention matrix |
| after softmax + mask | `(2, 12, 6, 6)` |
| `attn_weights @ values` | `(2, 12, 6, 64)` |
| `.transpose(1,2).contiguous().view(...)` | `(2, 6, 768)` |
| `out_proj(...)` | `(2, 6, 768)` |

Note the `(6, 6)` attention matrix: **attention is quadratic in sequence length**. Doubling context quadruples this cost. Every long-context innovation in modern LLMs is an attack on that square.

### Terms introduced
attention, self-attention, cross-attention, query, key, value, QKV projection, attention score, attention weight, softmax, scaled dot-product attention, `sqrt(d_k)` scaling, context vector, causal/masked attention, causal mask, look-ahead mask, triangular mask, attention dropout, multi-head attention, head, head dimension, output projection, buffer vs parameter, `qkv_bias`, quadratic complexity, attention map.

### Run
```bash
jupyter lab ch03/01_main-chapter-code/ch03.ipynb
jupyter lab ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb
```
The bonus notebook benchmarks the book's implementation against PyTorch's fused `scaled_dot_product_attention` (which dispatches to FlashAttention). Understanding *why* the fused version is faster — fewer memory round-trips, not fewer FLOPs — is a genuinely useful piece of systems intuition.

### Checkpoints
- What breaks if you delete the causal mask? Be specific about what the loss does. (It drops dramatically — the model is cheating.)
- Why mask *before* softmax rather than zeroing weights after?
- Why is the mask a buffer and not a parameter?
- With `d_out=768, num_heads=12`, what is `head_dim` and why must it divide evenly?
- Where does the sequence-length-squared term appear, exactly?

### Exercises
1. Set `num_heads=5` with `d_out=768` and read the assertion error. Explain it.
2. Extract and visualize the attention matrix for a real sentence with `matplotlib.imshow`. Look at what the diagonal-ish band means.
3. Reimplement `MultiHeadAttention` from a blank file. Verify against the repo's version by loading identical weights and comparing outputs with `torch.allclose`.
4. Implement the naive "wrapper of single heads" version and compare its speed with the reshape version.

### Pitfalls
- Confusing the *sequence* dimension with the *embedding* dimension when reading `@`. Trace shapes.
- Forgetting `.contiguous()` before `.view()` after a transpose — PyTorch will tell you, but understand why: transpose changes strides, not memory layout.
- Thinking heads are separate networks. They're slices of the same tensors.

---

## Stage 4 — Chapter 4: The full GPT model

**Files:** [`ch04/01_main-chapter-code/ch04.ipynb`](../ch04/01_main-chapter-code/ch04.ipynb), [`gpt.py`](../ch04/01_main-chapter-code/gpt.py)
**Bonus:** [`02_performance-analysis`](../ch04/02_performance-analysis)

### The ideas

Now you assemble the transformer block and stack it. The configuration you'll live with:

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # BPE vocabulary
    "context_length": 1024,  # max tokens the model can attend over
    "emb_dim": 768,          # width of every hidden representation
    "n_heads": 12,           # attention heads per block
    "n_layers": 12,          # transformer blocks stacked
    "drop_rate": 0.1,        # dropout probability
    "qkv_bias": False        # bias terms in Q/K/V projections
}
```

**Layer normalization.** Normalizes each token's feature vector to mean 0 and variance 1, then applies learned `scale` and `shift` parameters. Without it, deep stacks suffer exploding/vanishing activations and refuse to train. Note this normalizes *across the embedding dimension for each token independently* — unlike batch norm, it doesn't depend on other examples in the batch, which is why it suits variable-length sequences. The `eps = 1e-5` prevents division by zero.

**GELU activation.** A smoother alternative to ReLU: instead of a hard cutoff at zero, small negative values pass through slightly. The smooth gradient near zero empirically trains better in transformers. The book uses the tanh approximation.

**Feed-forward network.** Two linear layers with GELU between, expanding to `4 × emb_dim` (768 → 3072 → 768). This is where most of the parameters live — roughly ⅔ of the model. Attention mixes information *between* tokens; the FFN transforms each token's representation *independently*. A useful frame: attention = communication, FFN = computation.

**Shortcut (residual) connections.** `x = x + sublayer(x)`. This gives gradients a highway straight back to early layers, which is what makes 12 (or 100) layers trainable at all. The ch04 notebook has an explicit demo of gradient magnitudes with and without shortcuts — run it, it's convincing.

**The transformer block**, in the pre-LayerNorm arrangement GPT-2 uses:

```
x → LayerNorm → MultiHeadAttention → Dropout → (+ x)  →  shortcut1
  → LayerNorm → FeedForward         → Dropout → (+ shortcut1)
```

Normalizing *before* the sublayer (pre-LN) rather than after (post-LN, the original 2017 paper) makes deep training far more stable, and is now near-universal.

**The full model:** token embeddings + positional embeddings → dropout → 12 transformer blocks → final LayerNorm → linear output head projecting `768 → 50257` **logits**.

**Parameter accounting** — do this yourself, it demystifies "model size":

- Total: **163,009,536** parameters → **621.83 MB** at fp32 (4 bytes each).
- Called "GPT-2 124M" because the original ties the output head to the token embedding matrix (`50257 × 768 = 38,597,376` params counted once instead of twice): `163,009,536 − 38,597,376 = 124,412,160`.

Understanding weight tying now saves confusion when you load OpenAI's checkpoint in ch05.

### Terms introduced
transformer block, layer normalization, RMSNorm (contrast), batch norm (contrast), epsilon, scale and shift, GELU, ReLU, activation function, feed-forward network, MLP, expansion factor, residual/skip/shortcut connection, pre-LN vs post-LN, vanishing gradient, exploding gradient, logits, output head/LM head, weight tying, model size, fp32/bf16 memory footprint, hyperparameter, greedy decoding.

### Run
```bash
python ch04/01_main-chapter-code/gpt.py
```
Output is gibberish — the weights are random. That's the point: the architecture is correct, the knowledge is absent. Chapter 5 supplies the knowledge.

### Checkpoints
- Why is `emb_dim` constant through every block? (Residual connections require matching shapes.)
- Which contributes more parameters, attention or the FFN? Compute both.
- What are logits, and why aren't they probabilities?
- Why does normalizing before the sublayer help more than after?
- Why is dropout applied after attention *and* after the FFN *and* on the embeddings?

### Exercises
1. Instantiate GPT-2 medium (1024 dim, 24 layers, 16 heads), large, and XL by editing only the config. Print parameter counts. Confirm they match 355M / 774M / 1558M.
2. Remove shortcut connections and compare gradient magnitudes across layers (the notebook shows how).
3. Compute the KV-cache size for `context_length=1024` at bf16: `2 (K,V) × 12 layers × 768 dim × 1024 tokens × 2 bytes ≈ 36 MB` **per sequence**. Now imagine 100 concurrent users at 128k context. This is why serving is hard.
4. Do [`02_performance-analysis`](../ch04/02_performance-analysis) — FLOPs analysis makes scaling laws concrete later.

### Pitfalls
- Mixing up `emb_dim` (width) with `context_length` (sequence limit). Both are 768/1024-ish numbers and beginners swap them constantly.
- Believing "124M parameters" is a clean number. It depends on tying conventions, as above.

---

## Stage 5 — Chapter 5: Pretraining

**Files:** [`ch05/01_main-chapter-code/ch05.ipynb`](../ch05/01_main-chapter-code/ch05.ipynb), [`gpt_train.py`](../ch05/01_main-chapter-code/gpt_train.py), [`gpt_generate.py`](../ch05/01_main-chapter-code/gpt_generate.py), [`gpt_download.py`](../ch05/01_main-chapter-code/gpt_download.py)
**Bonus:** [`02_alternative_weight_loading`](../ch05/02_alternative_weight_loading), [`03_bonus_pretraining_on_gutenberg`](../ch05/03_bonus_pretraining_on_gutenberg), [`04_learning_rate_schedulers`](../ch05/04_learning_rate_schedulers), [`05_bonus_hparam_tuning`](../ch05/05_bonus_hparam_tuning), [`06_user_interface`](../ch05/06_user_interface), [`07_gpt_to_llama`](../ch05/07_gpt_to_llama), [`08_memory_efficient_weight_loading`](../ch05/08_memory_efficient_weight_loading)

### The ideas

**The loss.** Cross-entropy between the predicted distribution and the true next token. Concretely: take the model's probability for the correct token, take its log, negate, average over all tokens. Perfect prediction → 0. In practice you'll see it start near `ln(50257) ≈ 10.8` (a uniform guess over the vocabulary — a useful sanity check that your model is initialized sanely) and fall from there.

**Perplexity** = `exp(loss)`. Interpretable as "how many tokens is the model effectively choosing between." A perplexity of 10 means the model is about as uncertain as a uniform choice among 10 options. Loss 10.8 → perplexity ≈ 50257, i.e. total ignorance.

**Train vs validation loss.** The gap between them is your overfitting gauge. Training on `the-verdict.txt` (a single short story) overfits within a few epochs — deliberately. Watching it happen on a toy dataset teaches you to recognize it on a real one.

**Decoding strategies.** After the final linear layer you have 50257 logits. How you pick the next token matters enormously:

- **Greedy / argmax** — always the top token. Deterministic, and repetitive to the point of looping.
- **Temperature scaling** — divide logits by `T` before softmax. `T < 1` sharpens (more conservative), `T > 1` flattens (more random), `T → 0` approaches greedy. Temperature is not "creativity"; it's the flatness of the distribution you sample from.
- **Top-k sampling** — keep only the k highest-probability tokens, set the rest to `-inf`, renormalize, sample. Prevents the long tail of garbage tokens from ever being drawn.
- **Top-p / nucleus sampling** — keep the smallest set of tokens whose cumulative probability exceeds p. Adapts its cutoff to how confident the model is, unlike fixed top-k. (Not in the book; standard in practice — see the landscape doc.)

**Saving and loading.** `torch.save(model.state_dict(), ...)`. Also save the optimizer state — AdamW keeps momentum buffers, and resuming without them costs you progress.

**Loading OpenAI's GPT-2 weights.** The chapter's payoff: `gpt_download.py` fetches the real checkpoint and maps every tensor onto *your* class. When your from-scratch model produces coherent English, you have proof your implementation is correct. Note the transpose gymnastics — TensorFlow and PyTorch disagree on weight matrix orientation. [`08_memory_efficient_weight_loading`](../ch05/08_memory_efficient_weight_loading) shows how to avoid holding two copies of the weights in RAM, which is exactly what you need for large models.

**Scaling up.** [`03_bonus_pretraining_on_gutenberg`](../ch05/03_bonus_pretraining_on_gutenberg) runs the same loop on 60,000+ Project Gutenberg books. This is a genuine (small) pretraining run and worth doing if you have a GPU and patience.

### Terms introduced
loss function, cross-entropy loss, negative log-likelihood, logits vs probabilities, softmax, perplexity, training/validation/test split, overfitting, underfitting, generalization, epoch vs step vs iteration, learning rate, AdamW, weight decay, gradient clipping, learning-rate warmup, cosine decay, checkpointing, `state_dict`, greedy decoding, temperature, top-k, top-p/nucleus, autoregressive generation, context truncation, tokens seen, compute budget, teacher forcing.

### Run
```bash
python ch05/01_main-chapter-code/gpt_train.py     # trains on the-verdict.txt, plots losses
python ch05/01_main-chapter-code/gpt_generate.py  # downloads GPT-2 weights, generates text
```

### Checkpoints
- Why is the initial loss ≈ 10.8 and not, say, 1.0?
- Perplexity 50 vs perplexity 5 — what does that mean in words?
- Temperature 0.1 vs 1.5 on the same prompt: predict the difference before running it.
- Why does top-k help even when temperature is already low?
- Why save optimizer state, not just model weights?

### Exercises
1. Generate with temperature ∈ {0.0, 0.5, 1.0, 1.5, 2.0} × top_k ∈ {1, 10, 50, None}. Build a small table of outputs. This is the fastest way to develop decoding intuition.
2. Deliberately overfit: train 50 epochs on `the-verdict.txt` and watch validation loss turn upward while training loss keeps falling. Draw the curve.
3. Work through [`04_learning_rate_schedulers`](../ch05/04_learning_rate_schedulers), then [`appendix-D`](../appendix-D/01_main-chapter-code/appendix-D.ipynb) which adds warmup, cosine decay, and gradient clipping to the loop.
4. Load GPT-2 medium/large instead of small and compare generation quality at the same prompt.
5. **The big one:** [`07_gpt_to_llama`](../ch05/07_gpt_to_llama) — convert your GPT into Llama 2, then Llama 3.2. This single folder teaches RMSNorm, SwiGLU, RoPE, and grouped-query attention *in code you already understand*. It is the best bridge from this book to modern models that exists anywhere.

### Pitfalls
- Expecting a chatbot. A model trained on one short story produces stylistically-plausible nonsense. That is success.
- Forgetting `model.eval()` before generating — dropout stays on and outputs get noisy.
- Not truncating the context to `context_length` during generation → index error once you exceed 1024 tokens.

---

## Stage 6 — Chapter 6: Classification finetuning

**Files:** [`ch06/01_main-chapter-code/ch06.ipynb`](../ch06/01_main-chapter-code/ch06.ipynb), [`gpt_class_finetune.py`](../ch06/01_main-chapter-code/gpt_class_finetune.py), [`load-finetuned-model.ipynb`](../ch06/01_main-chapter-code/load-finetuned-model.ipynb)
**Bonus:** [`02_bonus_additional-experiments`](../ch06/02_bonus_additional-experiments) (an ablation table you should study), [`03_bonus_imdb-classification`](../ch06/03_bonus_imdb-classification), [`04_user_interface`](../ch06/04_user_interface)

### The ideas

Take the pretrained GPT-2, **replace the 50257-way output head with a 2-way head**, and train it to classify SMS spam. The pretrained body already understands English; you're only teaching it a decision boundary.

Key design decisions worth understanding rather than accepting:

- **Which layers to unfreeze.** Freeze everything, train only the new head → fast, weakest. Unfreeze the last transformer block + final LayerNorm + head → nearly full performance at a fraction of the cost. Unfreeze everything → best, slowest. The bonus experiments quantify exactly this trade-off; read that table carefully, because it's the same trade-off you'll face on every real finetuning task.
- **Which token's output to classify on.** The book uses the **last** token. In a causal model, only the last position has attended to the entire sequence — earlier positions literally cannot see the end of the text. (BERT-style models use a `[CLS]` token at the *start* precisely because they're bidirectional.)
- **Padding and balancing.** Sequences are padded to equal length; the dataset is balanced so accuracy is a meaningful metric.
- **Metrics.** Classification accuracy joins the loss. Loss is what you optimize; accuracy is what you care about. They can move in opposite directions, and knowing that saves you real debugging time.

### Terms introduced
transfer learning, finetuning, freezing/unfreezing layers, classification head, output head replacement, `[CLS]` token (contrast), padding token, sequence padding/truncation, class imbalance, accuracy/precision/recall/F1, train/val/test protocol, catastrophic forgetting, downstream task, feature extraction vs finetuning.

### Run
```bash
python ch06/01_main-chapter-code/gpt_class_finetune.py
```
Expect ~95–98% test accuracy.

### Checkpoints
- Why the last token and not the first or an average?
- Why replace the head rather than reuse the LM head and read off two specific word tokens? (Both work — the latter is "verbalizer"-style classification. Know why the explicit head is simpler.)
- What is catastrophic forgetting and which unfreezing choice makes it worse?

### Exercises
1. Run all three freezing strategies. Record accuracy and wall-clock time. Draw your own conclusion about the trade-off.
2. Classify on the *first* token instead of the last. Watch accuracy collapse. Explain it.
3. Do [`03_bonus_imdb-classification`](../ch06/03_bonus_imdb-classification) and compare a finetuned GPT-2 against plain logistic regression on bag-of-words. The gap (and the cost difference) is instructive.

---

## Stage 7 — Chapter 7: Instruction finetuning

**Files:** [`ch07/01_main-chapter-code/ch07.ipynb`](../ch07/01_main-chapter-code/ch07.ipynb), [`gpt_instruction_finetuning.py`](../ch07/01_main-chapter-code/gpt_instruction_finetuning.py), [`ollama_evaluate.py`](../ch07/01_main-chapter-code/ollama_evaluate.py), [`instruction-data.json`](../ch07/01_main-chapter-code/instruction-data.json)
**Bonus:** [`02_dataset-utilities`](../ch07/02_dataset-utilities), [`03_model-evaluation`](../ch07/03_model-evaluation), [`04_preference-tuning-with-dpo`](../ch07/04_preference-tuning-with-dpo), [`05_dataset-generation`](../ch07/05_dataset-generation), [`06_user_interface`](../ch07/06_user_interface)

### The ideas

This is the step that turns a text *completer* into an *assistant*. The mechanism is almost anticlimactic: supervised finetuning on `(instruction, input, output)` triples formatted into a consistent **prompt template** (the book uses Alpaca style). The model learns the *format* of being helpful — that after `### Response:` comes an answer rather than more questions.

Three implementation details carry real weight:

1. **Prompt formatting/templates.** Consistency matters more than the specific template. Modern models use chat templates with role markers (`system`/`user`/`assistant`) — same idea, standardized. A template mismatch between training and inference is one of the most common causes of a finetuned model behaving badly.
2. **Custom collate function with masking.** Batches are padded to the longest sequence in the batch (not a global max — less wasted compute). Padding tokens are set to `-100` in the targets, the ignore index PyTorch's cross-entropy skips. Optionally the *instruction* portion is also masked so loss is computed only on the response — the book discusses both, and the bonus experiments measure the difference.
3. **Evaluation.** Instruction-following has no accuracy metric. The book uses **LLM-as-a-judge**: run a stronger local model (Llama 3 via [Ollama](https://ollama.com)) to score your outputs 0–100. This is genuinely how the industry evaluates open-ended generation, with all its known biases (position bias, length bias, self-preference).

Then [`04_preference-tuning-with-dpo`](../ch07/04_preference-tuning-with-dpo) goes one step further: **Direct Preference Optimization**, training on `(prompt, chosen, rejected)` triples so the model learns which of two responses humans prefer. DPO reformulates RLHF as a simple classification-style loss — no separate reward model, no PPO rollouts. Doing this notebook puts you ahead of most people who talk about alignment.

### Terms introduced
instruction tuning, SFT (supervised finetuning), instruction dataset, prompt template, Alpaca/Phi format, chat template, system/user/assistant roles, collate function, dynamic padding, ignore index (-100), instruction masking, response-only loss, LLM-as-a-judge, Ollama, alignment, RLHF, reward model, PPO, DPO, preference pairs, chosen/rejected, reference model, KL penalty, synthetic data generation, distillation.

### Run
```bash
python ch07/01_main-chapter-code/gpt_instruction_finetuning.py
# then, with Ollama running:
python ch07/01_main-chapter-code/ollama_evaluate.py
```

### Checkpoints
- Why mask padding tokens with `-100` instead of just letting the model predict them?
- What does DPO replace in the RLHF pipeline, and why is that a simplification?
- Name three biases in LLM-as-a-judge evaluation.
- Why does an instruction-tuned model sometimes get *worse* at raw text completion?

### Exercises
1. Write 20 instruction examples in your own domain, add them to the dataset, and see whether the model picks up the behavior.
2. Train with and without instruction masking; compare judge scores.
3. Do the DPO notebook end to end. Then explain the `beta` parameter's role in your own words.
4. Do [`05_dataset-generation`](../ch07/05_dataset-generation) to generate synthetic instruction data — the same technique behind most modern open instruction datasets.

---

## Stage 8 — Appendix D & E: Training extras and LoRA

**Appendix D** ([`appendix-D.ipynb`](../appendix-D/01_main-chapter-code/appendix-D.ipynb)): three additions that turn a toy loop into a real one.

- **Learning-rate warmup** — start tiny and ramp up over the first few hundred steps. Adam's early updates are unreliable before its moment estimates stabilize; warmup prevents an early divergence you can't recover from.
- **Cosine decay** — smoothly anneal the LR toward zero across training. Large steps early to explore, small steps late to settle.
- **Gradient clipping** — cap the gradient norm (typically 1.0) so a single pathological batch can't blow up the weights.

Every serious training run uses all three. Now you know why.

**Appendix E** ([`appendix-E.ipynb`](../appendix-E/01_main-chapter-code/appendix-E.ipynb)): **LoRA (Low-Rank Adaptation)**.

Instead of updating a weight matrix `W` of shape `(d, k)` directly, freeze it and learn `ΔW = B @ A` where `A` is `(r, k)` and `B` is `(d, r)` with rank `r` tiny (4–64). Parameters trained drop by 100–1000×, optimizer memory drops with it, and quality is usually close to full finetuning. At inference you can merge `B@A` back into `W` for zero added latency, or keep adapters swappable to serve many task-specific variants from one base model.

The key insight to hold onto: the *update* needed to adapt a model to a task has much lower intrinsic rank than the model itself. LoRA and its descendants (QLoRA, DoRA) are the default way people finetune large models today.

### Terms introduced
warmup, cosine/linear decay schedule, gradient clipping, gradient norm, LoRA, rank, low-rank decomposition, adapter, alpha scaling, PEFT (parameter-efficient finetuning), adapter merging, QLoRA, frozen base model.

### Checkpoints
- Why does warmup matter specifically for Adam-family optimizers?
- With `d=768, k=768, r=8`, how many parameters does LoRA train versus full finetuning? (`768×8 + 8×768 = 12,288` vs `589,824` — about 2%.)
- Why can LoRA adapters be merged at inference but not during training?

---

## Stage 9 — Cross the bridge to modern models

Once the book is done, the single highest-value folder in this repo is [`ch05/07_gpt_to_llama`](../ch05/07_gpt_to_llama). It converts the GPT-2 you built into Llama 2, then Llama 3.2, one component at a time:

| GPT-2 (this book) | Llama 3 (modern) | Why it changed |
|---|---|---|
| LayerNorm | RMSNorm | Cheaper — drops the mean subtraction, nearly identical quality |
| GELU | SiLU + SwiGLU gating | Gated FFN performs better per parameter |
| Learned absolute positional embeddings | RoPE (rotary) | Encodes *relative* position; extrapolates to longer contexts |
| Multi-head attention | Grouped-query attention | Shrinks the KV cache several-fold with minimal quality loss |
| Dropout in blocks | No dropout | Huge unique-token corpora make it unnecessary |
| Biases in linear layers | No biases | Free parameter savings, no measurable loss |

Do those three notebooks. Then read [05 — The Transformer](05-transformer-architecture.md) §5.5 and [07 — Post-Training](07-post-training.md), which pick up exactly there and take you through MoE, RLVR, reasoning models, serving, and agents.
