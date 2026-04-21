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
