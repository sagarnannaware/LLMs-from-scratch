# 02 — Tokenization

*How text becomes numbers, and why this unglamorous step causes a startling share of real-world LLM weirdness.*

---

## 2.1 The problem

Neural networks consume numbers. Text is characters. Something must bridge them, and the choice of bridge has consequences that reach all the way to your API bill.

Three obvious options, and why two of them fail:

**Option 1 — one number per word.**
```
"the" → 1,  "cat" → 2,  "sat" → 3, ...
```
Problems: English has millions of word forms (`run`, `runs`, `running`, `runner`, `ran`). Your vocabulary explodes, and the output layer — which needs one score per vocabulary entry — becomes enormous. Worse, any word not in your list is unrepresentable. Every typo, every proper noun, every new word becomes `<UNK>` and its information is destroyed.

**Option 2 — one number per character.**
```
"the cat" → [20, 8, 5, 27, 3, 1, 20]
```
Problems: vocabulary is tiny (good), nothing is unrepresentable (good), but sequences become 4–5× longer. Since attention costs scale with the *square* of sequence length, this is expensive. And the model must spend capacity learning that `c`+`a`+`t` means something, rather than starting from useful units.

**Option 3 — subwords.** Common words stay whole; rare words split into meaningful pieces.
```
"the cat sat"       → ["the", " cat", " sat"]            3 tokens
"antidisestablish"  → ["anti", "dis", "establish"]       3 tokens
"pneumonoultramicro..." → ["p", "neum", "ono", "ultra", ...]
```

This is the winner, and **Byte Pair Encoding (BPE)** is how it's done.

---

## 2.2 BPE, built by hand

BPE was originally a 1994 compression algorithm. The adaptation to text is simple enough to do on paper.

**Training the tokenizer:**

1. Start with every individual character (or byte) as a token.
2. Count all adjacent token pairs in the corpus.
3. Merge the most frequent pair into a single new token.
4. Repeat until you reach your target vocabulary size.

Walk through a toy corpus: `"low low low lower lowest"`

```
Start:      l o w _ l o w _ l o w _ l o w e r _ l o w e s t
            Vocabulary: {l, o, w, e, r, s, t, _}

Count pairs: ("l","o")=5, ("o","w")=5, ("w","_")=3, ("w","e")=2, ...

Merge 1:    ("l","o") → "lo"        [most frequent, 5 occurrences]
            lo w _ lo w _ lo w _ lo w e r _ lo w e s t

Merge 2:    ("lo","w") → "low"      [now 5 occurrences]
            low _ low _ low _ low e r _ low e s t

Merge 3:    ("low","_") → "low_"    [3 occurrences]
            low_ low_ low_ low e r _ low e s t

Merge 4:    ("e","r") → "er"
            low_ low_ low_ low er _ low e s t

Final vocabulary: {l, o, w, e, r, s, t, _, lo, low, low_, er}
```

The frequent word `low` became one token. The rarer `lowest` remains split into pieces. **This is the entire idea**: frequency determines granularity, learned from data rather than designed by hand.

**Encoding new text** applies the learned merges in the order they were learned. `"lowering"` → `low` + `er` + `i` + `n` + `g`.

▶️ **Implement it yourself** — this is short enough to be worth typing:

```python
from collections import Counter

def get_pair_counts(tokens):
    return Counter(zip(tokens, tokens[1:]))

def merge_pair(tokens, pair, new_token):
    out, i = [], 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            out.append(new_token)
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out

text = "low low low lower lowest"
tokens = list(text.replace(" ", "_"))
merges = []

for step in range(6):
    counts = get_pair_counts(tokens)
    if not counts:
        break
    best, freq = counts.most_common(1)[0]
    if freq < 2:
        break
    new_token = best[0] + best[1]
    tokens = merge_pair(tokens, best, new_token)
    merges.append(best)
    print(f"merge {step+1}: {best} → '{new_token}'   ({freq}×)")
    print(f"   {tokens}\n")
```

Run it. Watching `low` assemble itself from `l`, `o`, `w` in three steps is the moment tokenization stops being mysterious.

---

## 2.3 Byte-level BPE: why nothing is ever unknown

GPT-2 and its successors operate on **bytes**, not characters. Since there are exactly 256 possible byte values, and any text in any language encodes to bytes, **every possible input is representable**. There is no `<UNK>` token in GPT-2, and there cannot be. Emoji, Chinese, corrupted bytes, binary data pasted by accident — all encode to something.

GPT-2's vocabulary of **50,257** breaks down as:
- 256 base byte tokens
- 50,000 learned merges
- 1 special token: `<|endoftext|>`

Modern models use 128k–256k vocabularies. Bigger vocabulary means fewer tokens per document (cheaper, longer effective context) but a larger embedding matrix and output layer. The growth was driven mainly by multilingual and code performance — a 50k English-centric vocabulary shreds other languages into fragments.

---

## 2.4 The practical consequences (this is the section that saves you)

Tokenization explains a long list of otherwise baffling LLM behaviors.

### Whitespace is part of the token

```
"hello"  → one token
" hello" → a DIFFERENT token
```

This is why a prompt ending with a trailing space can degrade output quality: you've forced the model into an unusual tokenization it rarely saw during training. **Don't end prompts with a trailing space.**

### Numbers tokenize inconsistently

Depending on the tokenizer, `12345` might become `123`+`45`, or `1`+`2345`, while `12346` becomes something entirely different. The model sees no consistent structure for arithmetic. This is a real contributor to arithmetic errors — the model is not doing digit-wise math, it's pattern-matching over arbitrary chunks. (Some modern tokenizers now force digits to split individually, which measurably helps.)

### Character-level tasks are genuinely hard

"How many r's in strawberry?" is difficult not because the model is stupid, but because it may see `["str", "aw", "berry"]` and never observes individual letters. Asking a model to reverse a string or count characters fights its representation.

### Non-English text costs more

The same meaning in English, Hindi, and Thai can differ by 2–3× in token count with an English-centric tokenizer. Since APIs bill per token and context windows are measured in tokens, **the same request costs different users different amounts based on their language.** Modern large vocabularies narrowed this gap considerably but did not close it.

### Token counting for cost and context

Rules of thumb for English with a GPT-style tokenizer:

```
1 token   ≈ 4 characters ≈ 0.75 words
100 tokens ≈ 75 words
1,000 tokens ≈ 1.5 pages of text
A 300-page book ≈ 130,000 tokens
```

Code is denser (more tokens per character) because of punctuation and indentation.

### Special tokens and the injection risk

Models use reserved tokens for structure: `<|endoftext|>`, and in chat models role markers like `<|im_start|>user`. If user-supplied text containing those literal strings is tokenized as special tokens, a user can forge conversation structure. Production tokenizers therefore refuse to encode special tokens from user input unless explicitly allowed — you'll see `allowed_special` parameters for exactly this reason.

---

## 2.5 Using a real tokenizer

> The cells in this section need `pip install tiktoken`, and the first call downloads
> the BPE vocabulary, so they require network access. Everything above and the
> dataloader below work with the from-scratch tokenizer too — see
> [notebook 01](notebooks/01_tokenization.ipynb), which is dependency-free.

▶️ **With `tiktoken`** (OpenAI's, fast, used by GPT-2/3/4):

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("gpt2")

text = "Hello, world! Tokenization is sneaky."
ids = enc.encode(text)
print(ids)
print([enc.decode([i]) for i in ids])
print(f"{len(text)} chars → {len(ids)} tokens")

# The whitespace trap, demonstrated
print(enc.encode("hello"), enc.encode(" hello"))     # different IDs

# The number trap
for n in ["1", "12", "123", "1234", "12345"]:
    print(f"{n:>6} → {enc.encode(n)}")
```

▶️ **Round-trip guarantee:**

```python
assert enc.decode(enc.encode(text)) == text   # lossless, always
```

That assertion holds for *any* string, which is the payoff of byte-level BPE.

---

## 2.6 From tokens to training data: the sliding window

Once text is a long list of token IDs, training data is made by sliding a window over it. Inputs and targets are the same sequence offset by one position:

```
Token stream:  [464, 3139, 286, 4881, 318, 6342, 13, ...]

Window 1  input:  [464, 3139, 286, 4881]
          target: [3139, 286, 4881, 318]

Window 2 (stride 4)  input:  [318, 6342, 13, ...]
```

**`max_length`** is the window size (the model's training context). **`stride`** is how far the window advances. Stride equal to `max_length` gives no overlap and sees each token once; a smaller stride produces more training samples from the same text, at the cost of repetition.

▶️ **A complete, runnable dataloader** — this is genuinely all there is to it:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.inputs, self.targets = [], []
        ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        for i in range(0, len(ids) - max_length, stride):
            self.inputs.append(torch.tensor(ids[i : i + max_length]))
            self.targets.append(torch.tensor(ids[i + 1 : i + max_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def make_dataloader(text, batch_size=4, max_length=256, stride=128, shuffle=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


text = "The quick brown fox jumps over the lazy dog. " * 50
loader = make_dataloader(text, batch_size=2, max_length=8, stride=8)
x, y = next(iter(loader))
print("inputs :", x.shape, "\n", x[0])
print("targets:", y.shape, "\n", y[0])
# targets[0] is inputs[0] shifted left by one — that's the whole supervision signal
```

Look at the printed pair and confirm the shift by eye. That off-by-one is the entire labeling scheme of LLM pretraining.

---

## 2.7 The road ahead: post-tokenizer research

Tokenization is widely regarded as the ugliest part of the stack, and there's active work on removing it:

- **Byte Latent Transformer (BLT)** and similar: operate on raw bytes with *learned dynamic patching*, allocating compute based on complexity rather than fixed merges. No vocabulary, no tokenizer artifacts.
- **Digit-aware tokenization**: forcing consistent number splitting; a cheap fix already widely adopted.
- **Multilingual-balanced vocabularies**: explicitly optimizing so no language is unduly penalized.

None of these have displaced BPE at scale yet. Learn BPE; watch this space.

---

## Check yourself

1. Why does byte-level BPE never need an `<UNK>` token?
2. You paste a prompt ending in a space and quality drops. Explain mechanically.
3. Why is "count the letters in this word" a hard task for reasons unrelated to intelligence?
4. Your app serves Hindi users and costs 2.5× your English projection. What's the cause?
5. In the dataloader, what does `stride=1` versus `stride=max_length` change?

<details>
<summary>Answers</summary>

1. Every possible input is a sequence of bytes, and all 256 byte values are in the vocabulary — so any string can always be represented.
2. `" word"` and `"word"` are different tokens; a trailing space forces a tokenization pattern rare in training data, pushing the model off-distribution.
3. The model may never see individual characters — it sees multi-character chunks — so letters aren't directly available in its representation.
4. Tokenizer inefficiency: an English-centric vocabulary splits Hindi into many more tokens per unit of meaning, and billing is per token.
5. `stride=1` produces one training sample per token position (maximum overlap, huge dataset, heavy repetition); `stride=max_length` produces non-overlapping windows where each token is seen once.
</details>

---

**Next:** [03 — Embeddings & Position](03-embeddings-and-position.md)
