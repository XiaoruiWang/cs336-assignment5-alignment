"""
Analyze failure cases from math_baseline_results.jsonl.
Prints 10 examples per category for experiment_log.md 3.2(b) analysis.

Usage:
    uv run python cs336_alignment/analyze_failures.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "outputs/math_baseline_results.jsonl"
N_EXAMPLES = 10


def truncate(text: str, max_chars: int = 300) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def main():
    records = []
    with open(RESULTS_FILE) as f:
        for line in f:
            records.append(json.loads(line))

    correct       = [r for r in records if r["format_reward"] == 1 and r["answer_reward"] == 1]
    fmt_wrong     = [r for r in records if r["format_reward"] == 1 and r["answer_reward"] == 0]
    no_format     = [r for r in records if r["format_reward"] == 0]

    print(f"Total: {len(records)}")
    print(f"  Correct (fmt=1, ans=1):     {len(correct)}")
    print(f"  Formatted wrong (fmt=1, ans=0): {len(fmt_wrong)}")
    print(f"  No format (fmt=0):          {len(no_format)}")
    print()

    def show_category(label: str, examples: list, n: int = N_EXAMPLES):
        print("=" * 70)
        print(f"CATEGORY: {label}  ({len(examples)} total, showing {min(n, len(examples))})")
        print("=" * 70)
        for i, r in enumerate(examples[:n], 1):
            print(f"\n--- Example {i} ---")
            print(f"Ground truth: {r['ground_truth']}")
            print(f"Response:\n{truncate(r['response'], 400)}")
        print()

    show_category("Format=1, Answer=0 (formatted but wrong)", fmt_wrong)
    show_category("Format=0, Answer=0 (no format)", no_format)
    show_category("Format=1, Answer=1 (correct)", correct)


if __name__ == "__main__":
    main()
