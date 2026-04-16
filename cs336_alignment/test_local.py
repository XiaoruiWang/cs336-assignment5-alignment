"""
Local tests. Run all: python cs336_alignment/test_local.py
Run one:  python cs336_alignment/test_local.py -m test_entropy
"""
import sys
sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer

PASS = "PASS"
FAIL = "FAIL"

def check(name, condition):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}")
    return condition


def test_tokenize():
    from tests.adapters import run_tokenize_prompt_and_output
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("\nTest: tokenize — shapes")
    prompt_strs = ["Hello world", "Hi"]
    output_strs = ["how are you", "good"]
    out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    p_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    o_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    expected_len = max(len(p)+len(o) for p,o in zip(p_ids, o_ids)) - 1
    check("input_ids shape",     out["input_ids"].shape     == (2, expected_len))
    check("labels shape",        out["labels"].shape        == (2, expected_len))
    check("response_mask shape", out["response_mask"].shape == (2, expected_len))

    print("\nTest: tokenize — labels shifted by 1")
    out = run_tokenize_prompt_and_output(["The cat sat"], ["on the mat"], tokenizer)
    check("labels[i] == input_ids[i+1]",
          torch.all(out["labels"][0, :-1] == out["input_ids"][0, 1:]).item())

    print("\nTest: tokenize — response_mask boundary")
    out = run_tokenize_prompt_and_output(["Hello"], ["world foo bar"], tokenizer)
    p_len = len(tokenizer(["Hello"], add_special_tokens=False)["input_ids"][0])
    o_len = len(tokenizer(["world foo bar"], add_special_tokens=False)["input_ids"][0])
    mask = out["response_mask"][0].tolist()
    check("prompt tokens masked 0",  all(m == False for m in mask[:p_len - 1]))
    check("response tokens masked 1", all(m == True for m in mask[p_len - 1: p_len - 1 + o_len]))

    print("\nTest: tokenize — padding masked 0")
    prompt_strs = ["Hi", "The quick brown fox jumps over the lazy dog"]
    output_strs = ["ok", "and then it ran away"]
    out = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    p_ids = tokenizer(prompt_strs, add_special_tokens=False)["input_ids"]
    o_ids = tokenizer(output_strs, add_special_tokens=False)["input_ids"]
    short_len = len(p_ids[0]) + len(o_ids[0])
    check("padding positions are False",
          all(m == False for m in out["response_mask"][0, short_len - 1:].tolist()))


def test_entropy():
    from tests.adapters import run_compute_entropy

    print("\nTest: compute_entropy")
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 100)
    entropy = run_compute_entropy(logits)
    check("output shape is (batch, seq)", entropy.shape == (2, 10))
    check("entropy is non-negative",      entropy.min().item() >= 0)
    check("entropy <= log(vocab_size)",   entropy.max().item() <= torch.log(torch.tensor(100.0)).item() + 1e-5)

    uniform_logits = torch.zeros(1, 1, 100)
    uniform_entropy = run_compute_entropy(uniform_logits)
    check("uniform logits → max entropy",
          abs(uniform_entropy.item() - torch.log(torch.tensor(100.0)).item()) < 1e-4)

    peaked_logits = torch.full((1, 1, 100), -1e9)
    peaked_logits[0, 0, 0] = 1e9
    peaked_entropy = run_compute_entropy(peaked_logits)
    check("peaked logits → near-zero entropy", peaked_entropy.item() < 1e-3)


# ── runner ────────────────────────────────────────────────────────────────────
TESTS = {
    "test_tokenize": test_tokenize,
    "test_entropy":  test_entropy,
}

if __name__ == "__main__":
    if "-m" in sys.argv:
        name = sys.argv[sys.argv.index("-m") + 1]
        if name not in TESTS:
            print(f"Unknown test '{name}'. Available: {list(TESTS)}")
            sys.exit(1)
        TESTS[name]()
    else:
        for fn in TESTS.values():
            fn()
    print("\nDone.")
