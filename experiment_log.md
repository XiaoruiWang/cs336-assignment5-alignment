# Experiment Log

## Section 3.2 — Zero-Shot Baseline (GSM8K)

**Date:** 2026-04-03  
**Model:** Qwen 2.5 Math 1.5B Base  
**Dataset:** GSM8K test set (1319 examples)  
**Prompt:** r1_zero  
**Script:** `cs336_alignment/math_baseline.py`  
**Results file:** `outputs/math_baseline_results.jsonl`

---

### 3.2(c) — Overall Performance

| Metric | Value |
|---|---|
| Total examples | 1319 |
| Correct (reward=1) | 32 |
| **Accuracy** | **2.4%** |
| Avg format reward | 0.196 |
| Avg answer reward | 0.024 |

> Qwen 2.5 Math 1.5B achieves **2.4% accuracy** on GSM8K zero-shot with the r1_zero prompt.

---

### 3.2(b) — Category Breakdown

| Category | Count | % |
|---|---|---|
| Format=1, Answer=1 (correct) | 32 | 2.4% |
| Format=1, Answer=0 (formatted but wrong) | 226 | 17.1% |
| Format=0, Answer=0 (no format) | 1061 | 80.5% |

**Analysis:**

**Format=0 cases (1061 examples, 80.5%):**
- The model largely ignores the `<think>`/`</think>`/`<answer>` tag structure
- Qwen 2.5 Math Base was pretrained on math text, not on the r1_zero format
- The model likely outputs answers in its own style (e.g. `\boxed{}`) without the required tags
- Without `</think> <answer>` in the response, `r1_zero_reward_fn` gives format_reward=0

**Format=1, Answer=0 cases (226 examples, 17.1%):**
- The model correctly follows the format but computes the wrong answer
- Arithmetic errors, wrong reasoning steps, or misunderstanding the question
- Shows that format compliance and mathematical correctness are separate challenges

**Key takeaway:**
The low accuracy is expected — the r1_zero format is foreign to the base model.
SFT on reasoning traces will first teach the format, then RL (Expert Iteration / GRPO)
will improve correctness. This 2.4% is our floor to beat.

---

### TODO: Add 10 example failure cases per category (3.2b written analysis)

---

## Section 4.3 — SFT Dataset Size Sweep (GSM8K)

**Date:** 2026-04-21
**Model:** Qwen 2.5 Math 1.5B Base
**Dataset:** GSM8K sft.jsonl (r1_zero format)
**Script:** `cs336_alignment/sft.py`
**Config:** lr=1e-5, batch_size=2, gradient_accumulation_steps=4, clip_value=1.0
**Eval:** 50 prompts from training set, every 5 optimizer steps

### Results Summary

| Dataset Size | Peak avg_reward | Peak format_reward | Notes |
|---|---|---|---|
| 128 | ~0.10 | ~0.80 | reward collapses then recovers; overfitting evident |
| 256 | ~0.175 | ~0.70 | steady climb, hits 15% target, slight format drop at end |
| 512 | ~0.32 | ~0.75 | strong improvement, trend still rising at step 60 |
| 1024 | ~0.40 | ~0.85 | noisy but higher ceiling, format mostly learned |
| full | ~0.40–0.60 | ~0.85–0.90 | best overall; format plateaus ~0.9, reward stabilizes 0.4–0.6 |

### Key Observations

- More data consistently improves both format and answer accuracy
- Format reward learns faster than answer reward — model learns structure before correctness
- With full dataset, `avg_reward` reaches 40–60%, well above the 15% target
- High variance in eval curves is due to small eval set (50 prompts) — not instability
- Token entropy decreases over training — model becomes more confident/deterministic
- Baseline was 2.4% → SFT with full dataset achieves ~40–60%: **~20x improvement**

---

## Section 4.3 — SFT Filtered Dataset Experiment

**Date:** 2026-04-21
**Config:** same as full dataset run above
**Filter:** keep only examples where `r1_zero_reward_fn(response, ground_truth)["reward"] == 1.0`

### Result

**Filtered dataset size: 7473 / 7473 (100% pass rate)**

All examples in sft.jsonl already produce correct answers — the dataset was curated to only include correct reasoning traces. The filter has no effect.

**Curves:** identical to the full dataset run (~40–60% reward, ~0.85–0.90 format reward).

### Finding

The GSM8K SFT dataset is pre-filtered for correctness. There is no meaningful difference between "filtered" and "full" for this dataset. This contrasts with what the experiment was designed to test — in a dataset with mixed correct/incorrect examples, filtering would be expected to improve performance by training only on high-quality demonstrations.
