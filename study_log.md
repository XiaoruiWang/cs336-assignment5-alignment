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

### Zero-shot Baseline Results (GSM8K, Qwen2.5-Math-1.5B)

- Total examples: 1319
- Correct (format=1, answer=1): 32 (2.4%)
- Formatted wrong (format=1, answer=0): 226 (17.1%)
- No format (format=0, answer=0): 1061 (80.5%)
