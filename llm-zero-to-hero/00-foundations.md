# 00 — Foundations

*Everything you actually need before transformers. If you know PyTorch and gradient descent, skim to the summary and move on.*

---

## 0.1 The only math that matters

People are told LLMs need heavy mathematics. In practice, **one operation accounts for well over 90% of what a language model does**: multiplying a vector by a matrix. Learn that one properly and the rest follows.

### Vectors are lists of numbers with a meaning attached

A vector is an ordered list: `[0.2, -1.4, 0.8]`. That's it mechanically. What makes it interesting is that we can *choose* what the numbers mean, and neural networks learn that choice themselves.

Suppose we describe words with three made-up features — royalty, gender, age:

```
king   = [0.9,  0.8, 0.7]
queen  = [0.9, -0.8, 0.7]
man    = [0.1,  0.8, 0.5]
woman  = [0.1, -0.8, 0.5]
```

Now `king - man + woman = [0.9, -0.8, 0.7]` — exactly `queen`. Meaning has become arithmetic. This is not a party trick; it's the entire reason we represent words as vectors. Real models use 768 to 8192 numbers per word, and nobody labels the dimensions — they're learned.

**Dot product** measures alignment between two vectors: multiply elementwise, then sum.

```
[1, 2, 3] · [4, 5, 6] = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

Large positive → the vectors point the same way (similar). Near zero → unrelated. Negative → opposed. **Attention is built entirely on dot products**, so hold onto this: *dot product = similarity score*.

### Matrices are stacks of vectors, and matrix multiplication is many dot products at once

A matrix is a grid of numbers. When we multiply a vector by a matrix, we compute the dot product of that vector with each **column** of the matrix:

```
                    ┌  1   0  ┐
[2, 3]   ×          │  4   1  │     =   [2×1 + 3×4,  2×0 + 3×1]  =  [14, 3]
 (1×2)              └ (2×2)   ┘             (1×2)
```

Interpretation that will serve you for the rest of this guide: **a matrix multiplication is a learned transformation of a representation.** The input vector "means" something in one space; the matrix rewrites it into another space where different information is easy to read off. Every `nn.Linear` in a transformer is exactly this, and the matrix entries are what training learns.

### Shapes are the load-bearing skill

The single most useful habit in this field: track tensor shapes obsessively. Almost every bug you will ever write is a shape error in disguise.

The rule for matrix multiply: `(a, b) @ (b, c) → (a, c)`. The inner dimensions must match and they vanish; the outer ones survive.

```
(4, 768) @ (768, 3072) → (4, 3072)      ✓ inner 768 matches
(4, 768) @ (512, 3072) → error          ✗ 768 ≠ 512
```

With batches, the leading dimensions come along for the ride:

```
(2, 10, 768) @ (768, 768) → (2, 10, 768)
 │   │    │
 │   │    └── features per token
 │   └────── tokens in the sequence
 └────────── independent examples in the batch
```

That `(batch, sequence, features)` shape is the shape of essentially every tensor inside a language model. Burn it in.

### Softmax turns scores into probabilities

Given raw scores, softmax exponentiates each and divides by the total, producing positive numbers that sum to 1:

```
softmax([2.0, 1.0, 0.1])
  = [e², e¹, e⁰·¹] / (e² + e¹ + e⁰·¹)
  = [7.39, 2.72, 1.11] / 11.22
  = [0.659, 0.242, 0.099]
```

Two properties matter later:
- Exponentiation **exaggerates differences**. A score gap of 2 becomes a probability ratio of ~7.4×.
- It's **shift-invariant**: adding a constant to every score changes nothing. That's why real implementations subtract the max before exponentiating — same answer, no numerical overflow.

Softmax appears twice in an LLM: inside attention (to turn similarity scores into weights) and at the output (to turn logits into a probability distribution over the vocabulary).

### What you can safely not know

You will never differentiate anything by hand. You don't need eigenvalues, integrals, or measure-theoretic probability to build, train and deploy language models. If you understand dot products, matrix shapes, and softmax, you have the mathematical equipment for this entire guide.

---

## 0.2 What a neural network actually is

A neural network is a function with adjustable knobs. You feed in numbers, it produces numbers, and training turns the knobs until the outputs are useful.

The smallest useful unit is a **linear layer**: multiply by a weight matrix, add a bias vector.

```
output = input @ W + b
```

Stack two of those and you have... still a linear function. Two matrix multiplies collapse into one. To get genuine expressive power you need a **nonlinearity** between them:

```
hidden = activation(input @ W1 + b1)
output = hidden @ W2 + b2
```

That's a neural network. Everything else — transformers included — is an elaborate arrangement of this pattern with structure that suits the data.

**Activation functions** are the nonlinearity:

| Function | Formula | Notes |
|---|---|---|
| ReLU | `max(0, x)` | Simple, fast, historically dominant |
| GELU | `x · Φ(x)` | Smooth version of ReLU; used in GPT-2 |
| SiLU/Swish | `x · sigmoid(x)` | Smooth; component of modern SwiGLU layers |

The smoothness matters: a hard corner at zero gives a discontinuous gradient, and gradients are how the network learns.

▶️ **Run it:**

```python
import torch, torch.nn as nn

layer = nn.Linear(4, 3)          # transforms 4 numbers into 3
x = torch.randn(2, 4)            # 2 examples, 4 features each
print(x.shape, "→", layer(x).shape)      # torch.Size([2, 4]) → torch.Size([2, 3])
print("weight:", layer.weight.shape)     # torch.Size([3, 4])
print("bias:  ", layer.bias.shape)       # torch.Size([3])
```

Note PyTorch stores the weight as `(out_features, in_features)` and internally uses `x @ W.T`. A frequent source of confusion when loading weights between frameworks.

---

## 0.3 How networks learn: loss, gradients, gradient descent

### The loss function

Training needs a single number saying how wrong the model is. That's the **loss**. Lower is better. Learning is the process of pushing it down.

For language models the loss is **cross-entropy**: the negative log of the probability the model assigned to the correct answer.

```
model says " Paris" has probability 0.7  →  loss = -log(0.7) = 0.357
model says " Paris" has probability 0.1  →  loss = -log(0.1) = 2.303
model says " Paris" has probability 0.99 →  loss = -log(0.99) = 0.010
```

Notice the shape of the penalty: being *confidently wrong* is punished brutally (probability 0.001 → loss 6.9), while being right earns diminishing rewards. This asymmetry is what discourages overconfidence.

### Gradients

The **gradient** of the loss with respect to a weight answers: *if I nudge this weight up slightly, does the loss go up or down, and how fast?*

A model with 124 million parameters has a gradient with 124 million entries — one per weight, each saying which way to nudge it.

### Gradient descent

The learning rule is embarrassingly simple:

```
weight = weight - learning_rate × gradient
```

Move each weight a small step in the direction that reduces the loss. Repeat millions of times. The `learning_rate` sets step size and is the most consequential hyperparameter you will ever set: too large and training explodes, too small and it crawls or gets stuck.

### Backpropagation

Backprop is how gradients are computed for all 124 million weights efficiently. It's the chain rule from calculus applied backwards through the network: the loss's sensitivity to the final layer is easy, and each earlier layer's sensitivity is computed from the layer after it.

**You will never implement this.** PyTorch records every operation into a graph and walks it backwards for you. What you do need is the intuition: *gradients flow backwards from the loss, and anything that blocks or distorts that flow breaks training.* Two design choices in every transformer — residual connections and normalization — exist purely to keep that flow healthy.

---

## 0.4 PyTorch in ten minutes

### Tensors

A tensor is an n-dimensional array that knows which device it lives on and whether it needs gradients.

▶️
```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])   # 2×2
print(a.shape, a.dtype)                   # torch.Size([2, 2]) torch.float32

print(a @ a)          # matrix multiply
print(a * a)          # elementwise multiply — completely different!
print(a.T)            # transpose
print(a.sum(dim=0))   # sum down columns → tensor([4., 6.])
print(a.sum(dim=1))   # sum across rows   → tensor([3., 7.])
```

`dim` confuses everyone at first. **`dim=k` means "collapse the k-th dimension."** `a.sum(dim=0)` on a `(2,2)` tensor removes dimension 0, leaving 2 numbers.

### Reshaping

Three operations you'll use constantly in attention:

▶️
```python
x = torch.arange(12)                 # shape (12,)
print(x.view(3, 4).shape)            # (3, 4)   — reinterpret layout, no copy
print(x.view(3, 4).transpose(0, 1).shape)   # (4, 3) — swap two dimensions
print(x.view(3, 4).unsqueeze(0).shape)      # (1, 3, 4) — insert a dimension
```

`view` requires contiguous memory. After a `transpose`, memory order no longer matches shape order, so you may need `.contiguous()` before `.view()`. PyTorch will tell you; now you'll know why.

### Autograd

▶️
```python
w = torch.tensor([2.0], requires_grad=True)
loss = (w * 3 - 12) ** 2       # minimized when w = 4
loss.backward()
print(w.grad)                  # tensor([-36.]) → negative, so increasing w reduces loss
```

### Modules

`nn.Module` is the container. Anything assigned to `self` that is a `Parameter` or `Module` is automatically registered — it moves with `.to(device)`, appears in `.parameters()`, and gets saved in `state_dict()`.

▶️
```python
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

net = TinyNet(4, 16, 2)
print(sum(p.numel() for p in net.parameters()), "parameters")   # 114
print(net(torch.randn(5, 4)).shape)                              # (5, 2)
```

Call `net(x)`, never `net.forward(x)` — the former runs PyTorch's hooks.

### The training loop

Memorize this shape. You will write it dozens of times, and it is identical whether you're training a toy classifier or a 70-billion-parameter model.

▶️
```python
import torch, torch.nn as nn

torch.manual_seed(0)

# Toy problem: learn y = 3x + 2 from noisy data
X = torch.randn(200, 1)
y = 3 * X + 2 + 0.1 * torch.randn(200, 1)

model = nn.Linear(1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()          # 1. clear old gradients
    pred = model(X)                # 2. forward pass
    loss = loss_fn(pred, y)        # 3. how wrong are we?
    loss.backward()                # 4. compute gradients
    optimizer.step()               # 5. update weights
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

print(f"learned: y = {model.weight.item():.2f}x + {model.bias.item():.2f}")
# learned: y = 3.00x + 2.00
```

**The five steps never change.** Pretraining a frontier model is this loop with a bigger model, a bigger dataset, and thousands of GPUs.

**The classic bug:** omit `optimizer.zero_grad()` and gradients accumulate across batches instead of being replaced. Training silently degrades rather than erroring. Everyone does this once.

### Train vs eval mode

```python
model.train()   # dropout active, for training
model.eval()    # dropout off, for inference

with torch.no_grad():    # don't build the autograd graph — saves memory and time
    output = model(x)
```

Forgetting `model.eval()` before generating text gives you randomly degraded output and a confusing afternoon.

---

## 0.5 Vocabulary check

Terms used from here on without re-explanation:

| Term | Meaning |
|---|---|
| **Parameter / weight** | A learned number. "7B model" = 7 billion of them. |
| **Feature / dimension** | One slot in a vector. |
| **Batch** | Several examples processed together for efficiency. |
| **Epoch** | One full pass over the training data. |
| **Step / iteration** | One weight update (one batch). |
| **Forward pass** | Input → output. |
| **Backward pass** | Computing gradients. |
| **Logits** | Raw output scores, before softmax. |
| **Inference** | Using a trained model, as opposed to training it. |
| **Hyperparameter** | Something you choose (learning rate, layer count), not something learned. |

---

## Check yourself

1. What shape results from `(8, 128, 512) @ (512, 2048)`?
2. Why does a two-layer network need a nonlinearity between the layers?
3. If a model gives the correct token probability 0.05, what's the cross-entropy loss? *(≈3.0)*
4. What does `optimizer.zero_grad()` prevent, and what goes wrong without it?
5. Why is `dot product = similarity` the sentence to remember going into the attention chapter?

<details>
<summary>Answers</summary>

1. `(8, 128, 2048)` — inner dimensions match and disappear, leading dimensions ride along.
2. Without it, two matrix multiplies compose into a single matrix multiply — the network has no more power than one layer.
3. `-log(0.05) ≈ 3.0`.
4. It clears gradients from the previous step. Without it, gradients sum across batches and your updates reflect stale data.
5. Because attention scores every pair of tokens with a dot product, and that score becomes "how much should this token pay attention to that one."
</details>

---

**Next:** [01 — What a Language Model Is](01-language-modeling.md)
