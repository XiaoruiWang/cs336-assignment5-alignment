# Study Log — CS336 Assignment 5 Alignment

## Tensor Slicing Cheat Sheet

### The One Rule

**Commas separate dimensions. Each argument addresses one dim.**

```
A[dim0, dim1, dim2, ...]
```

### Core Syntax

```python
a[start:stop]          # start inclusive, stop exclusive
a[start:stop:step]     # with stride
a[:]                   # copy all
a[::-1]                # reverse
```

### Scalar vs Slice — Dimension Survival

**Scalar index → dimension collapses. Slice → dimension stays.**

```python
x = tensor of shape (3, 4)

x[0]        # shape (4,)     — scalar, dim gone
x[0:1]      # shape (1, 4)   — slice, dim kept
x[0, :]     # shape (4,)     — same as x[0]
x[0:1, :]   # shape (1, 4)   — range preserves dim
```

**Mistake to avoid:** `attn_weights[:, 0, :, :]` on shape `(2,4,3,3)` → gives `(2,3,3)` not `(2,1,3,3)` because `0` is scalar. To keep the dim: `attn_weights[:, 0:1, :, :]`.

### 2-D: Rows vs Columns

```python
A[i]          = A[i, :]       # row i, all cols
A[:, j]                       # all rows, col j
A[:, :-1]                     # all rows, drop LAST col
A[:, 1:]                      # all rows, drop FIRST col
A[1:, 1:]                     # drop first row & col
```

**Mistake to avoid:**
```python
# WRONG — slicing rows, not columns:
input_ids = padded[:-1]       # slices dim0 (rows)
labels    = padded[1:]        # slices dim0 (rows)

# RIGHT — slicing columns for all rows:
input_ids = padded[:, :-1]    # all rows, drop last col
labels    = padded[:, 1:]     # all rows, drop first col
```

**If you forget the comma, you're slicing rows.**

### N-D Tensors (batch, seq, feature, ...)

```python
T[:, -1, :]         # last seq position, all batches
T[:, 0]             # first seq position (trailing : optional)

# Classic next-token-prediction shift
inputs = tokens[:, :-1]    # all but last
labels = tokens[:, 1:]     # all but first
```

### Fancy Indexing

```python
# A single tensor index = selects along dim 0
idx = tensor([0, 2])
A[idx]                # selects ROW 0 and ROW 2

# Two tensor indices = zip-style pairing (NOT cartesian product)
row_idx = tensor([0, 1, 2])
col_idx = tensor([3, 2, 0])
A[row_idx, col_idx]   # → pairs: (0,3), (1,2), (2,0)

# Cross-entropy gather pattern
logits[torch.arange(batch), labels]   # grabs logits[0,4], logits[1,0], ...
```

### Boolean Indexing

```python
x[x > 0]               # values where condition is True (not the mask!)
mask = (x > 25)
x[mask]                 # selects elements where True
```

### Key Mental Model

Commas separate dimensions. Each slice acts on one dimension. `:` means "take all" on that dimension.

- Integer index **drops** a dimension → `x[0]` drops dim
- Slice **keeps** a dimension → `x[0:1]` keeps dim

### Quick Mental Checklist

1. **Count the commas** — that's how many dimensions you're indexing
2. **Forgot the comma?** — you're only touching dim 0
3. **Scalar or slice?** — scalar drops the dim, slice keeps it
4. **One tensor arg?** — selects along dim 0 only
5. **Two tensor args?** — zip, not cartesian
6. **Boolean expression inside `[]`?** — returns values, not the mask

---

## Key Concepts Learned

### `gather` — picking one value per position from a distribution

```python
log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
```

Step by step:

| Step | Code | Shape | Why |
|---|---|---|---|
| Start | `log_probs_all` | `(B, S, V)` | log-prob for every vocab token at every position |
| Expand index | `labels.unsqueeze(-1)` | `(B, S, 1)` | `gather` needs index to have same ndim as source |
| Select | `.gather(dim=-1, index=...)` | `(B, S, 1)` | picks `log_probs_all[b, s, labels[b,s]]` for each b, s |
| Remove extra dim | `.squeeze(-1)` | `(B, S)` | one log-prob per token position |

**What `gather` does:** for each `(b, s)` pair, it uses `labels[b, s]` as an index into the vocab dimension to pick out exactly one value. The label is different at every position, so you can't use simple slicing.

**Why `unsqueeze` then `squeeze`:** `gather` requires the index tensor to have the same number of dimensions as the source tensor. `labels` is `(B, S)` but `log_probs_all` is `(B, S, V)`, so we temporarily add a dim, gather, then remove it.

---

### `softmax` vs `sum` — shape behavior

| Operation | What it does on `dim=k` | Shape change | Typical use |
|---|---|---|---|
| `softmax(dim=k)` | normalize (values sum to 1) | no change | probabilities |
| `sum(dim=k)` | collapse (add up) | reduces by 1 dim | aggregation |

```python
x = torch.randn(2, 3, 4)

# softmax: shape unchanged
probs = torch.softmax(x, dim=-1)       # (2, 3, 4) — same shape

# sum: shape shrinks
y = torch.sum(x, dim=-1)              # (2, 3)   — last dim removed
y_keep = torch.sum(x, dim=-1, keepdim=True)  # (2, 3, 1) — kept as size-1
```

**Key distinction:** `softmax` transforms values, `sum` collapses a dimension.

---

### `keepdim=True` — when to use it

Use `keepdim=True` when the reduced result needs to **broadcast back** against the original tensor. Without it, the axis disappears entirely and shapes won't align.

The classic case:
```python
x = torch.randn(2, 3, 4)  # (B, S, V)

# Manual normalization: divide each element by the sum along V
x / x.sum(dim=-1)                    # RuntimeError! (2,3,4) / (2,3) — shape mismatch
x / x.sum(dim=-1, keepdim=True)      # Works: (2,3,4) / (2,3,1) — broadcasts over V
```

**Rule:** if you're going to use the result in an operation with the original tensor, add `keepdim=True`.

---

### Always specify `dim` when reducing tensors

Forgetting `dim` collapses everything to a scalar — silently wrong.

```python
# Wrong — sums ALL dimensions → scalar ()
torch.sum(probs * log_probs)

# Correct — sums over vocab only → shape (batch, seq)
(probs * log_probs).sum(dim=-1)
```

**Two equivalent styles in PyTorch:**

| Style | Example | When to use |
|---|---|---|
| Functional | `torch.sum(x, dim=-1)` | explicit, useful when chaining |
| Method (OO) | `x.sum(dim=-1)` | cleaner, preferred |

Both are identical — pick whichever reads better.

---

### Entropy from Logits

The entropy formula is:

$$H(p) = -\sum_x p(x) \log p(x)$$

But you don't have $p(x)$ — you have **logits**. To get $\log p(x)$:

$$\log p(x_i) = \text{logits}_i - \log \sum_j e^{\text{logits}_j}$$

The second term is **logsumexp**:

$$\log \sum_j e^{\text{logits}_j}$$

So in code:
```python
log_probs = F.log_softmax(logits, dim=-1)   # numerically stable log p(x)
entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # H = -sum p(x) log p(x)
```

Why `log_softmax` instead of `softmax` then `log`? Numerical stability — `softmax` can overflow/underflow for large logits. `log_softmax` uses the logsumexp trick internally to stay stable.

---



### `.item()` — extracting scalars from tensors

Any reduction on a tensor (`.sum()`, `.mean()`, indexing a 1-D tensor) returns a **0-dim PyTorch tensor**, not a plain Python number:

```python
attention_mask[i].sum()         # tensor(5)   ← still a tensor
attention_mask[i].sum().item()  # 5           ← plain Python int

entropy_avg[i]                  # tensor(1.23) ← 0-dim tensor
entropy_avg[i].item()           # 1.23         ← plain Python float
```

**When to use `.item()`:** whenever you want to store a value in a dict, log it, serialize to JSON, or compare with a plain number. Tensors in dicts are not JSON-serializable.

### Counting tokens vs characters

```python
len(responses[i])                    # wrong — counts characters
attention_mask[i].sum().item()       # correct — counts real tokens
```

`len("hello")` = 5 (characters). `len(tokenizer("hello"))` = 1 (token). Always use token count for LLM analysis — character count is meaningless for measuring response length.

### Shape of `entropy_avg`

```python
entropy      # (B, S)  — per-token entropy
entropy_avg = (entropy * attention_mask).sum(-1) / attention_mask.sum(-1)
             # sum(-1) → (B,),  divide → (B,)
```

`entropy_avg[i]` is a 0-dim scalar tensor — one average entropy value per example in the batch. Use `.item()` to extract as a plain float.

---

### Tokenizing a batch for model forward pass — three common bugs

**Context:** computing entropy over model outputs for a batch of generated responses.

**Bug 1 — tokenizer returns a list, model needs a tensor**
```python
# Wrong
input_ids = tokenizer(responses, add_special_tokens=False)["input_ids"]  # list of lists

# Correct
input_ids = tokenizer(responses, return_tensors="pt", padding=True, add_special_tokens=False)["input_ids"]  # tensor
```

**Bug 2 — model() returns an object, not logits directly**
```python
# Wrong
logits = model(input_ids)

# Correct
logits = model(input_ids).logits   # always need .logits
```

**Bug 3 — averaging entropy over padding tokens too**
```python
# Wrong — includes padding positions
entropy_avg = entropy.mean()

# Correct — mask out padding before averaging
attention_mask = tokenizer(responses, return_tensors="pt", padding=True, add_special_tokens=False)["attention_mask"]
entropy = run_compute_entropy(logits)                          # (B, S)
entropy_avg = (entropy * attention_mask).sum(-1) / attention_mask.sum(-1)  # per-example avg
```

**Q: What is `attention_mask`?**
Shape `(B, S)` — `1` for real tokens, `0` for padding. Returned automatically by the HuggingFace tokenizer when `padding=True`. You don't compute it yourself.

**Q: When do you need `padding=True`?**
Whenever you have a batch of variable-length sequences. Padding aligns them into a rectangular tensor. Single sequences don't need it.

**Q: When do you need `return_tensors="pt"`?**
Whenever the result goes into a PyTorch model. Without it, the tokenizer returns Python lists, which the model can't accept.

---

### `sft_microbatch_train_step` — don't forget to average over batch

**Bug:** using `dim=None` in `run_masked_normalize` sums over all elements (seq AND batch), skipping the batch average. Loss was 2x too large.

**Fix:** use `dim=-1` to sum over the sequence dimension only → shape `(B,)`, then `.mean()` to average over the batch:

```python
per_example = run_masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant)  # (B,)
loss = -per_example.mean() / gradient_accumulation_steps
```

**Why it matters:** `dim=None` collapses batch + seq at once, so a batch of 2 produces a sum 2x larger than expected. Always reduce dimensions deliberately — seq first, then batch.

---

### pytest fixture override — Stanford path workaround

`tests/conftest.py` hardcodes `/data/a5-alignment/models/Qwen2.5-Math-1.5B` (Stanford cluster path). On RunPod this path doesn't exist, causing the test to fail at setup before even running your code.

Fix: make the fixture fall back to the HuggingFace model ID when the local path doesn't exist:

```python
@pytest.fixture
def model_id():
    import os
    local_path = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    if os.path.isdir(local_path):
        return local_path
    return "Qwen/Qwen2.5-Math-1.5B"
```

HuggingFace `from_pretrained` checks the local cache first — since we already ran `huggingface-cli download Qwen/Qwen2.5-Math-1.5B`, it loads from `~/.cache/huggingface/hub/` without re-downloading.

---



### `uv run python` vs `python`

- **`python`** — uses whatever Python is active in your current shell (e.g., conda `alignment` environment)
- **`uv run python`** — uses the Python inside the project's `.venv`, and **first installs all dependencies** listed in `pyproject.toml` before running. This is why it tries to build `flash-attn` (which needs CUDA) even for simple scripts.

**Rule of thumb:**
- Scripts using only standard libraries (`json`, `re`, `pathlib`) → use `python`
- Scripts using project packages (`vllm`, `transformers`, `torch`) → use `uv run` on RunPod (Linux + CUDA)

---

### SFT Data Format

The r1_zero prompt template ends with `Assistant: <think>`. The model continues from there, so:

- **prompt** = full r1_zero template with question filled in (ends with `Assistant: <think>`)
- **response** = `{reasoning}</think> <answer>{answer}</answer>` (no leading `<think>` — already in prompt)

The preamble in every prompt acts like a **system prompt** — it tells the model what role to play and what format to follow. Every training example shares this same preamble.

---

### r1_zero Reward Function

`r1_zero_reward_fn` checks for `"</think> <answer>"` and `"</answer>"` in the response. If both present, it extracts the answer and grades it. Returns:
- `format_reward`: 1 if format is correct, 0 otherwise
- `answer_reward`: 1 if answer is correct, 0 otherwise
- `reward`: 1 only if both format and answer are correct

---

### Tokenizer — two things to know

1. **Tokenizers run on CPU** — no `device_map` or `.to(device)` needed. Only the model goes on GPU.

2. **Always use the model's own tokenizer** — `AutoTokenizer.from_pretrained(model_id)` loads the correct one automatically. GPT-2's tokenizer appears in `test_local.py` only because it's tiny and fast for local testing; never use it in production code.

---

### Zero-shot Baseline Results (GSM8K, Qwen2.5-Math-1.5B)

- Total examples: 1319
- Correct (format=1, answer=1): 32 (2.4%)
- Formatted wrong (format=1, answer=0): 226 (17.1%)
- No format (format=0, answer=0): 1061 (80.5%)

---

### `zero_grad` placement and the training loop invariant

**The one invariant that must never be broken:**

```
backward → step
```

`zero_grad` can go anywhere — before `backward` or after `step` — as long as it does not slip between `backward` and `step`.

**Canonical PyTorch loop (no gradient accumulation):**

```python
for batch in dataloader:
    optimizer.zero_grad()   # clear before accumulating
    loss = criterion(model(batch), labels)
    loss.backward()
    optimizer.step()
```

**With gradient accumulation:**

```python
for i in range(iterations):
    loss, _ = run_sft_microbatch_train_step(...)   # calls backward internally
    if (i+1) % gradient_accumulation_steps == 0:
        clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        optimizer.zero_grad()   # reset after step, ready for next window
        train_step += 1
```

`zero_grad` moves to after `step` because you want gradients to accumulate across multiple `backward()` calls within one window. Putting `zero_grad` before each `backward` would wipe the accumulation.

**Why `zero_grad` before `step` is wrong:** it wipes accumulated gradients before the optimizer can use them — the weight update uses zero gradients and does nothing.

### Why divide loss by `gradient_accumulation_steps`

PyTorch `.backward()` **adds** to existing `.grad` tensors — it does not replace them. So across 4 microbatches:

```
microbatch 1: loss/4 → backward()  → grad = g1/4
microbatch 2: loss/4 → backward()  → grad = g1/4 + g2/4
microbatch 3: loss/4 → backward()  → grad = g1/4 + g2/4 + g3/4
microbatch 4: loss/4 → backward()  → grad = g1/4 + g2/4 + g3/4 + g4/4
optimizer.step()                   → uses the accumulated sum (= average of 4 losses)
optimizer.zero_grad()              → resets grads to 0
```

Dividing by `gradient_accumulation_steps` inside the microbatch train step ensures the final accumulated gradient equals the average — the same as if you processed all 4 examples in one big batch. Without the division, the optimizer would see 4× the correct gradient magnitude.

---

### Device mismatch — always move tokenizer output to model device

**Error:** `RuntimeError: Expected all tensors to be on the same device, but found cuda:0 and cpu`

**Cause:** tokenizer always outputs CPU tensors. Model lives on GPU. Passing CPU tensors directly to a GPU model causes a device mismatch.

**Fix:** move tensors immediately after tokenizing, before any forward pass:
```python
input_ids = input_ids.to(model.device)
labels = labels.to(model.device)
logits = model(input_ids).logits
```

**Mental model:** tokenize → move → forward. Always ask "where did this tensor come from?" before calling `model(x)`. Tokenizer output is always CPU; models are on GPU. They must match.

---

### GRPO importance ratio — use exp(log difference), not division

**Bug:** computing the policy ratio as `policy_log_probs / old_log_probs` — this divides two log-probability values, which is meaningless.

**Fix:**
```python
ratio = torch.exp(policy_log_probs - old_log_probs)
```

**Why:** the importance sampling ratio is π(a) / π_old(a). Working in log space:

$$\frac{\pi(a)}{\pi_{\text{old}}(a)} = \exp\!\bigl(\log\pi(a) - \log\pi_{\text{old}}(a)\bigr)$$

Dividing log-probs gives log(π) / log(π_old), which has no probabilistic meaning. Always compute ratios via subtraction in log space, then exponentiate.

**Secondary pattern — when you don't have `model.device`:** use `tensor.device` to match devices without passing the model around.

---

### `torch.no_grad()` — always use during inference/eval

**Bug:** OOM during `log_generations` eval forward pass — 50 long responses batched together exhausted GPU memory.

**Two fixes:**
1. Reduce eval batch size (e.g. `prompts[:10]` instead of `prompts[:50]`)
2. Wrap eval forward pass in `torch.no_grad()` — entropy/log-prob computation for eval doesn't need gradients, so PyTorch doesn't need to store activations for backprop:

```python
with torch.no_grad():
    logits = model(input_ids).logits
```

**Why it saves memory:** during a normal forward pass, PyTorch stores intermediate activations for backprop. `no_grad()` skips this — cuts activation memory roughly in half.

**Rule:** any forward pass that is NOT followed by `.backward()` should be wrapped in `torch.no_grad()`. This includes eval, logging, and entropy computation. If one tensor is already on the right device, move others to match it:

```python
mask = mask.to(tensor.device)   # match mask to wherever tensor already lives
```

This is useful inside helper functions that receive tensors but not the model itself.

---

### GRPO Hyperparameters — Mental Model

#### Anchor: one batch = one optimizer step = one weight update

Everything else is derived from this.

#### Decision flow (outer to inner)

```
group_size (G) = 8              ← algorithm choice: how many responses per prompt?
rollout_batch_size = 256        ← compute choice: how many total responses per GRPO step?
n_prompts_per_rollout_batch = rollout_batch_size // group_size = 32   ← derived
train_batch_size = 256          ← = rollout_batch_size (train on all of them in one optimizer step)
gradient_accumulation_steps = 128   ← hardware choice: how small must microbatches be to fit GPU memory?
micro_train_batch_size = train_batch_size // gradient_accumulation_steps = 2   ← derived
n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size = 128   ← derived
```

#### What each name means

| Name | Value | Meaning |
|---|---|---|
| `rollout_batch_size` | 256 | Total responses generated per GRPO step |
| `group_size` (G) | 8 | Responses sampled per prompt by vLLM |
| `n_prompts_per_rollout_batch` | 32 | Distinct prompts sampled each step |
| `train_batch_size` | 256 | Responses used per optimizer step (one weight update) |
| `gradient_accumulation_steps` | 128 | Microbatches to split train_batch_size into |
| `micro_train_batch_size` | 2 | Responses per microbatch forward+backward pass |
| `n_microbatches_per_rollout_batch` | 128 | Total backward() calls per optimizer.step() |

#### Two free choices; everything else is derived

- `group_size` — algorithm choice (more = better advantage estimates, slower generation)
- `gradient_accumulation_steps` — hardware choice (more = smaller microbatches = less GPU memory)

#### Why micro_train_batch_size = 2 (not larger)?

With 1.5B parameters and 1024-token sequences, activations stored for backprop fill GPU memory fast.
Batch size 2 is the safe default for this model/sequence-length combo on the handout's hardware.
If you have more GPU memory, reduce `gradient_accumulation_steps` (larger microbatches = fewer passes = faster).

#### `gradient_accumulation_steps` vs `n_microbatches_per_rollout_batch`

Both equal 128 here because `train_batch_size = rollout_batch_size`. They mean different things:
- `gradient_accumulation_steps` — how many backward() calls before one optimizer.step(); used to scale loss
- `n_microbatches_per_rollout_batch` — how many microbatches cover the full rollout batch

If `train_batch_size < rollout_batch_size` (multiple optimizer steps per rollout), they would diverge.

#### Why divide loss by `gradient_accumulation_steps` inside the microbatch step?

PyTorch `.backward()` **adds** to `.grad`, not replaces. After 128 microbatches:
`grad = g1/128 + g2/128 + ... + g128/128` = average gradient over the full batch.
Without the division, the optimizer would see 128× the correct gradient magnitude.

---

### OOM: never forward the full rollout batch at once — use microbatches + no_grad

**Error:** `torch.OutOfMemoryError: Tried to allocate 89.19 GiB` when computing `old_log_probs` by passing all 256 sequences to the model in one call.

**Root cause (how to read the traceback):** always read bottom-up. The bottom line is the actual error; trace upward to find which line in your code triggered it. Here: `model(input_ids).logits` → called from `run_get_response_log_probs` → called from `grpo.py` with `tokens["input_ids"]` = all 256 sequences × 1024 tokens = too large.

**Why `no_grad()` saves memory:** a normal forward pass stores all intermediate activations for backprop. For 256 × 1024 tokens through a 1.5B model, that's ~89GB. `torch.no_grad()` tells PyTorch to skip storing those activations — only the final output tensor is kept. Memory drops to almost nothing per microbatch.

**Fix:** compute `old_log_probs` in microbatches under `no_grad()`, then `cat` the results:

```python
old_log_probs_list = []
with torch.no_grad():
    for step in range(n_microbatches_per_rollout_batch):
        start = step * micro_train_batch_size
        end = (step + 1) * micro_train_batch_size
        result = run_get_response_log_probs(model, tokens["input_ids"][start:end], tokens["labels"][start:end], False)
        old_log_probs_list.append(result["log_probs"])
old_log_probs = torch.cat(old_log_probs_list, dim=0)  # (rollout_batch_size, seq_len)
```

The final `torch.cat` is cheap — it just stitches together small `(2, seq_len)` tensors, no activations involved.

**Rule:** any forward pass not followed by `.backward()` must be wrapped in `torch.no_grad()`. For large batches, also process in microbatches to avoid peak activation memory.

---

### GRPO microbatch tokenization — don't re-tokenize inside the loop

**Bug:** tokenizing the full rollout batch once (256 responses) to get `old_log_probs`, then re-tokenizing microbatches of 2 inside the microbatch loop to get `policy_log_probs`.

**Why it breaks:** padding is computed relative to the longest sequence in each batch. The full batch pads to `max_len(256 sequences)`; a microbatch pads to `max_len(2 sequences)`. So `old_log_probs` has shape `(2, L_full)` and `policy_log_probs` has shape `(2, L_micro)` where `L_micro ≤ L_full`. The ratio `exp(policy_log_probs - old_log_probs)` crashes on shape mismatch.

**Fix:** tokenize once before the microbatch loop, then slice the tensors:
```python
# Before the loop — tokenize full batch once
tokens = run_tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)

# Inside the microbatch loop — slice, don't re-tokenize
mb_input_ids    = tokens["input_ids"][start:end]
mb_labels       = tokens["labels"][start:end]
mb_response_mask = tokens["response_mask"][start:end]
sliced_old_log_probs = old_log_probs[start:end, :]

new_log_probs = run_get_response_log_probs(model, mb_input_ids, mb_labels, False)
policy_log_probs = new_log_probs["log_probs"]
# Now both shapes are (micro_train_batch_size, L_full) — consistent
```

**Mental model:** tokenize once to establish a consistent padding length, then slice rows. Never re-tokenize a subset — the padding will shrink and break alignment with any tensors computed on the full batch.

---

### `epochs_per_rollout_batch` — on-policy vs off-policy

Controls how many times you train on the same rollout batch before collecting new rollouts.

**= 1 (on-policy):** generate → train once → generate new → repeat. The policy being trained is essentially the same one that generated the data.

**> 1 (off-policy):** generate → train N times → generate new. By the 2nd+ pass the policy has drifted from when the data was collected → use `grpo_clip` to correct for this via the importance ratio `π(a)/π_old(a)`. With `epochs = 1`, the ratio is always ~1.0 so clipping does nothing.

**Practical reason for > 1:** generation (vLLM) is slow; training is fast. Re-using rollouts 2-3× extracts more gradient signal per expensive generation call.
