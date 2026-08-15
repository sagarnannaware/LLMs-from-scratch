# 07 — Post-Training

*Turning a text predictor into an assistant: SFT, RLHF, DPO, RLVR/GRPO, reasoning models, and LoRA.*

---

## 7.1 Where we are

After pretraining you have a **base model**: it knows an enormous amount and completes text beautifully, but it doesn't answer questions — it continues them. Post-training fixes the behavior without adding knowledge.

The modern pipeline, and how it got there:

```
2020   SFT only
2022   SFT → reward model → PPO                    [InstructGPT / ChatGPT]
2023   SFT → DPO                                   [simpler: no RM, no rollouts]
2024   SFT → DPO/PPO → light RLVR
2025   SFT → RLVR with GRPO                        [DeepSeek-R1: reasoning emerges]
2026   SFT → RLVR → agentic RL in environments → preference polish
```

Each stage answers a different question:

| Stage | Question | Signal |
|---|---|---|
| **SFT** | What does a good answer *look like*? | Demonstrations |
| **Preference tuning** | Which of two answers do people *prefer*? | Human/AI comparisons |
| **RLVR** | Which answers are *actually correct*? | Automated verifiers |

---

## 7.2 Supervised finetuning (SFT)

The mechanism is the same next-token prediction as pretraining — only the data changes. Instead of raw web text, you train on curated `(instruction, response)` pairs formatted into a consistent template.

### The template

```
### Instruction:
Convert this sentence to passive voice.

### Input:
The chef cooked the meal.

### Response:
The meal was cooked by the chef.
```

Modern models use **chat templates** with role markers instead:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**Consistency matters more than the specific format.** A mismatch between the template used in training and the one used at inference is one of the most common causes of a finetuned model behaving strangely — it's off-distribution in a way that's invisible in your code.

### Loss masking: the detail that matters

You want the model to learn to *produce responses*, not to *produce instructions*. So mask the prompt portion out of the loss, computing it only on the response tokens. PyTorch's cross-entropy ignores targets set to `-100`.

▶️ **A complete SFT collate function:**

```python
import torch

def format_prompt(entry):
    text = (f"### Instruction:\n{entry['instruction']}\n\n")
    if entry.get("input"):
        text += f"### Input:\n{entry['input']}\n\n"
    return text + "### Response:\n"


def sft_collate(batch, pad_token_id=50256, ignore_index=-100, mask_prompt=True):
    """batch: list of (prompt_ids, response_ids)"""
    max_len = max(len(p) + len(r) for p, r in batch)
    inputs, targets = [], []

    for prompt_ids, response_ids in batch:
        full = prompt_ids + response_ids
        pad = [pad_token_id] * (max_len - len(full))

        inp = full + pad
        tgt = full[1:] + [pad_token_id] + pad          # shifted by one

        # mask padding out of the loss
        for i in range(len(full) - 1, len(tgt)):
            tgt[i] = ignore_index
        # optionally mask the prompt so loss is response-only
        if mask_prompt:
            for i in range(len(prompt_ids) - 1):
                tgt[i] = ignore_index

        inputs.append(torch.tensor(inp))
        targets.append(torch.tensor(tgt))

    return torch.stack(inputs), torch.stack(targets)


batch = [([1, 2, 3], [4, 5]), ([1, 2, 3, 4], [5, 6, 7])]
x, y = sft_collate(batch)
print("inputs :\n", x)
print("targets:\n", y)     # -100 marks positions excluded from the loss
```

Two things this handles that beginners miss: **dynamic padding** (pad to the longest sequence *in this batch*, not a global maximum — much less wasted compute) and **prompt masking**.

### How much data?

Surprisingly little. A few thousand high-quality, diverse examples usually beats hundreds of thousands of mediocre ones — the model already has the capabilities; you're selecting a behavior. **Quality and diversity dominate quantity** at this stage.

---

## 7.3 RLHF: learning from preferences

SFT teaches imitation of demonstrations. But writing an ideal response is hard, while *comparing* two responses is easy. RLHF exploits that asymmetry.

```
1. Sample two responses to the same prompt
2. A human picks the better one          → preference pair (chosen, rejected)
3. Train a REWARD MODEL to predict those preferences
4. Optimize the LLM to maximize reward, with a KL penalty
   keeping it close to the SFT model
```

The **reward model** is the LLM with the token-prediction head replaced by a scalar head that outputs one number: how good is this response.

**PPO** then optimizes the policy against that reward. It works, but it's heavy: four models in memory (policy, reference, reward model, value model), a sampling loop, and notorious hyperparameter sensitivity.

**The KL penalty is load-bearing.** Without it, the policy drifts to whatever exploits the reward model's blind spots — producing high-reward gibberish. That's **reward hacking**, and it's the central failure mode of learned rewards.

---

## 7.4 DPO: the simplification that took over

**Direct Preference Optimization** proved something elegant: the optimal RLHF policy can be reached *directly*, with a simple loss on preference pairs. No reward model. No sampling loop.

```
RLHF:  data → reward model → PPO rollouts → policy       (4 models, complex)
DPO:   data → loss → policy                              (2 models, simple)
```

The loss increases the probability of chosen responses and decreases rejected ones — **relative to a frozen reference model**, which plays the KL penalty's role:

```
loss = -log σ( β · [ (log π(chosen)  − log π_ref(chosen))
                   − (log π(rejected) − log π_ref(rejected)) ] )
```

▶️ **DPO loss, implemented:**

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """Each argument: (batch,) summed log-probs of the response tokens."""
    policy_margin = policy_chosen_logps - policy_rejected_logps
    ref_margin    = ref_chosen_logps    - ref_rejected_logps
    logits = beta * (policy_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()

    chosen_rewards   = beta * (policy_chosen_logps   - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    accuracy = (chosen_rewards > rejected_rewards).float().mean()
    return loss, accuracy


# the policy already prefers "chosen" more than the reference does → low loss
loss, acc = dpo_loss(
    torch.tensor([-5.0, -3.0]), torch.tensor([-8.0, -9.0]),
    torch.tensor([-6.0, -4.0]), torch.tensor([-7.0, -7.0]),
)
print(f"loss {loss:.4f}   preference accuracy {acc:.2f}")
```

**`beta`** controls how far the policy may drift from the reference. Low beta (0.01) allows large changes and risks degeneration; high beta (0.5) keeps it conservative. 0.1 is the usual starting point.

**The DPO family:** IPO (fixes a DPO overfitting pathology), KTO (learns from thumbs-up/down without pairs), ORPO (merges SFT and preference optimization into one stage), SimPO (drops the reference model entirely).

---

## 7.5 RLVR and GRPO: how reasoning models are made

This is the most consequential development since RLHF, and the mechanism is beautifully simple.

### The insight

Learned reward models can be gamed. But for **verifiable** domains, you don't need one:

| Domain | Verifier |
|---|---|
| Math | Compare to the known answer |
| Code | Run the unit tests |
| Structured output | Validate against the schema |
| Tool use | Did the task complete? |
| Retrieval | Is the cited source correct? |

The reward is 1 or 0, computed by a program. **It cannot be hacked, because it is the actual objective.** This is **Reinforcement Learning with Verifiable Rewards (RLVR)**.

### What emerged from it

When models were trained this way at scale, something unplanned happened: they spontaneously developed **long deliberate reasoning** — trying approaches, noticing errors, backtracking, re-checking. Nobody supervised those behaviors. They emerged because they raise the probability of a correct final answer, and the reward only measured that.

This is the origin of "reasoning models," and it's why reasoning became a *product tier* rather than a prompting trick.

### GRPO

PPO needs a **value network** — a second large model estimating expected future reward, to compute advantages. That's expensive.

**Group Relative Policy Optimization** removes it. Sample a *group* of G responses to the same prompt and use the group's own mean reward as the baseline:

```
for each prompt:
    sample G responses from the current policy          (G = 8, 16, 64...)
    score each with the verifier                        (1 = passes, 0 = fails)
    advantage_i = (reward_i − mean(rewards)) / std(rewards)
    update the policy to make above-average responses more likely
```

The group *is* the baseline. No critic, far less memory, much simpler.

▶️ **GRPO advantages — the core of the algorithm:**

```python
import torch

def grpo_advantages(rewards, eps=1e-4):
    """rewards: (num_prompts, group_size) verifier scores"""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)

# prompt A: 3 of 8 samples passed;  prompt B: all 8 failed
rewards = torch.tensor([
    [1., 0., 1., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0.],
])
adv = grpo_advantages(rewards)
print(adv.round(decimals=3))
```

Look at row B: every advantage is 0. **When all samples in a group get the same reward, there is no learning signal at all.** Handling those degenerate groups — by filtering them, by adjusting sampling difficulty, or by shaping rewards — is exactly what the refinements (DAPO, GSPO, and a steady stream of successors) address.

### Test-time compute

RLVR created a second scaling axis. Instead of only "train a bigger model," you can "let it think longer":

- **Long chain-of-thought** — thousands of reasoning tokens before answering.
- **Self-consistency** — sample several paths, take the majority answer.
- **Best-of-n with a verifier** — generate n, select the one that passes.
- **Thinking budgets** — user-controllable effort, because reasoning tokens are billed tokens.

**Overthinking** is the real cost: reasoning models burn hundreds of tokens on trivial questions. Hybrid "reason only when needed" routing is now standard product behavior.

**Practical guidance:** reasoning models earn their cost on hard, multi-step, verifiable tasks (math, debugging, planning). They're usually a waste on extraction, classification, and formatting.

---

## 7.6 Finetuning your own model

### Should you finetune at all?

Work down this list and stop at the first "yes":

| Need | Solution | Cost |
|---|---|---|
| Better instructions/format | **Prompting** | Free |
| Model needs facts it lacks | **RAG** (chapter 09) | Low |
| Consistent style/format/tone | **LoRA finetuning** | Medium |
| A specialized narrow task, high volume | **Full or LoRA finetuning** | Medium |
| New domain vocabulary/language | **Continued pretraining** | High |

**The most common expensive mistake is finetuning to add knowledge.** Finetuning shapes *behavior*; it's an unreliable and costly way to insert facts, and those facts go stale. Put knowledge in the context via retrieval.

### LoRA

Full finetuning updates every weight and needs optimizer state for all of them — ~16 bytes per parameter. **LoRA** freezes the base model and learns a small low-rank update instead.

For a weight matrix `W` of shape `(d, k)`, learn `ΔW = B @ A` where `A` is `(r, k)` and `B` is `(d, r)` with `r` tiny (4–64):

```
output = x @ W  +  (x @ A.T) @ B.T · (α/r)
         └frozen┘  └────── trained ──────┘
```

For `d = k = 768, r = 8`:

```
full:  768 × 768 = 589,824 trained parameters
LoRA:  768×8 + 8×768 = 12,288                    → 2.1%
```

The insight: the *update* needed to adapt a model to a task has far lower intrinsic rank than the model itself.

▶️ **LoRA, implemented and injected into a real model:**

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=8, alpha=16):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False                       # freeze the original

        in_f, out_f = base_layer.in_features, base_layer.out_features
        self.A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, rank))   # zeros → starts as identity
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def apply_lora(model, rank=8, alpha=16, target_names=("W_query", "W_value")):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_names:
            setattr(model, name, LoRALinear(module, rank, alpha))
        else:
            apply_lora(module, rank, alpha, target_names)
    return model


model = GPTModel(GPT_CONFIG_124M)                          # from chapter 05
before = sum(p.numel() for p in model.parameters() if p.requires_grad)

for p in model.parameters():
    p.requires_grad = False
apply_lora(model, rank=8)

after = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable before: {before:,}")
print(f"trainable after : {after:,}   ({after/before*100:.2f}%)")
```

**Why `B` starts at zero:** then `B @ A = 0`, so the adapted model is *exactly* the original at step 0. Training starts from a known-good point rather than a perturbed one.

**Practical notes:**
- **Rank**: 8–16 for style/format, 32–64 for harder task shifts. Diminishing returns above that.
- **Alpha**: usually 2× rank. What matters is the ratio `α/r`.
- **Targets**: query and value projections are the classic choice; applying to all linear layers works better but trains more parameters.
- **QLoRA**: LoRA on top of a 4-bit quantized frozen base. Lets you finetune a 70B model on a single 48GB GPU.
- **Merging**: `W_new = W + BA·(α/r)` folds the adapter in for **zero inference overhead**. Or keep adapters separate and hot-swap many task-specific ones over one base model in memory.

---

## 7.7 The rest of the toolkit

**Rejection sampling / best-of-n finetuning** — generate many responses, keep the ones a verifier or judge approves, then SFT on those. Simple, effective, and how a lot of synthetic data is produced.

**Distillation** — train a small model on a large model's outputs (or its full output distribution). This is why strong small models exist; nearly every good 7B model has been distilled from something much larger.

**Model merging** — average or interpolate the weights of several models finetuned from a shared base. Combines capabilities with zero additional training, and works surprisingly well.

**Constitutional AI / RLAIF** — replace much human labeling with model-generated critiques against a written set of principles.

---

## Check yourself

1. Why mask prompt tokens out of the SFT loss?
2. What does DPO eliminate from RLHF, and what plays the KL penalty's role?
3. Why can't RLVR rewards be hacked the way learned reward models can?
4. In GRPO, what replaces PPO's value network?
5. All 8 samples in a GRPO group fail. What is the gradient signal?
6. Why does LoRA's `B` matrix initialize to zeros?
7. Your model needs to know your company's Q3 policy documents. Finetune or RAG? Why?

<details>
<summary>Answers</summary>

1. You want the model to learn to generate responses, not to generate instructions; training on the prompt wastes capacity on a distribution you'll never need it to produce.
2. The reward model and the PPO rollout loop. A frozen reference model, via the `beta` term, constrains drift instead.
3. The reward is computed by an actual verifier — running the tests, checking the answer — so "maximizing reward" and "being correct" are the same thing rather than correlated things.
4. The group mean reward serves as the baseline for computing advantages.
5. None. All advantages are zero — the group is degenerate and contributes nothing, which is a known GRPO failure mode.
6. So `B @ A = 0` at initialization, making the adapted model exactly equal to the base model before any training.
7. RAG. Finetuning shapes behavior, not knowledge; documents change and the model would go stale, and retrieval also gives you citations and access control.
</details>

---

**Next:** [08 — Inference & Efficiency](08-inference-and-efficiency.md)
