"""
Local tests for run_tokenize_prompt_and_output.
Run with: python cs336_alignment/test_local.py
"""
import sys
sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer
from tests.adapters import run_tokenize_prompt_and_output

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token_id = tokenizer.eos_token_id

PASS = "PASS"
FAIL = "FAIL"

def check(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition


# ── Test 1: basic shapes ──────────────────────────────────────────────────────
print("\nTest 1: shapes are correct")
prompt_strs = ["Hello world", "Hi"]
output_strs = ["how are you", "good"]

out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

p_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
o_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
expected_len = max(len(p)+len(o) for p,o in zip(p_ids, o_ids)) - 1

check("input_ids shape",     out["input_ids"].shape     == (2, expected_len))
check("labels shape",        out["labels"].shape        == (2, expected_len))
check("response_mask shape", out["response_mask"].shape == (2, expected_len))


# ── Test 2: labels is input_ids shifted left by 1 ────────────────────────────
print("\nTest 2: labels is input_ids shifted left by 1")
prompt_strs = ["The cat sat"]
output_strs = ["on the mat"]

out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

# Reconstruct full sequence from input_ids and labels
# input_ids = full[:-1], labels = full[1:]
# So input_ids[0] and labels[0] should overlap everywhere except first/last
check("labels[i] == input_ids[i+1]",
      torch.all(out["labels"][0, :-1] == out["input_ids"][0, 1:]).item())


# ── Test 3: response_mask boundary ───────────────────────────────────────────
print("\nTest 3: response_mask boundary is correct")
prompt_strs = ["Hello"]
output_strs = ["world foo bar"]

out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

p_ids = tokenizer(["Hello"], add_special_tokens=False)["input_ids"][0]
o_ids = tokenizer(["world foo bar"], add_special_tokens=False)["input_ids"][0]
p_len = len(p_ids)
o_len = len(o_ids)

mask = out["response_mask"][0].tolist()

# Positions before prompt_length-1 should be 0
check("prompt tokens masked 0",
      all(m == False for m in mask[:p_len - 1]))
# Positions from prompt_length-1 to prompt_length-1+output_length should be 1
check("response tokens masked 1",
      all(m == True for m in mask[p_len - 1 : p_len - 1 + o_len]))


# ── Test 4: padding positions are 0 in response_mask ─────────────────────────
print("\nTest 4: padding positions are 0 in response_mask")
# Use two sequences of very different lengths so padding is clearly visible
prompt_strs = ["Hi", "The quick brown fox jumps over the lazy dog"]
output_strs = ["ok", "and then it ran away"]

out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

# The short sequence (row 0) should have trailing 0s in response_mask
p_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
o_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
short_len = len(p_ids[0]) + len(o_ids[0])  # total tokens for row 0
total_len  = out["response_mask"].shape[1]

check("padding positions are False",
      all(m == False for m in out["response_mask"][0, short_len - 1:].tolist()))


print("\nDone.")
