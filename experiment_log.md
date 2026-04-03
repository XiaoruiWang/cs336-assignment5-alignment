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
