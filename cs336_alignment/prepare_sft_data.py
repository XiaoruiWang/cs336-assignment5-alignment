"""
Convert data/gsm8k/train.jsonl into SFT format for r1_zero training.

The r1_zero prompt template ends with "Assistant: <think>", so the model's
response starts immediately after <think>. The SFT target response is therefore:

    {reasoning}</think> <answer>{answer}</answer>

(no leading <think> tag — the prompt already provides it)

Output: data/gsm8k/sft.jsonl with {"prompt": "...", "response": "..."} records.

Usage:
    uv run python cs336_alignment/prepare_sft_data.py
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def strip_calculator_annotations(text: str) -> str:
    """Remove <<expr=result>> annotations from GSM8K reasoning steps."""
    return re.sub(r"<<[^>]*>>", "", text)


def convert_example(example: dict, prompt_template: str) -> dict:
    raw_answer = example["answer"]

    # Split on #### to get reasoning and final answer
    parts = raw_answer.split("####")
    reasoning = parts[0].strip()
    final_answer = parts[1].strip()

    # Clean up calculator annotations from reasoning
    reasoning = strip_calculator_annotations(reasoning)

    # Build prompt using the r1_zero template
    prompt = prompt_template.format(question=example["question"])

    # Response starts after the <think> tag that the prompt already provides
    response = f"{reasoning}</think> <answer>{final_answer}</answer>"

    return {"prompt": prompt, "response": response}


def main():
    template_path = PROJECT_ROOT / "cs336_alignment/prompts/r1_zero.prompt"
    input_path = PROJECT_ROOT / "data/gsm8k/train.jsonl"
    output_path = PROJECT_ROOT / "data/gsm8k/sft.jsonl"

    prompt_template = template_path.read_text()

    examples = []
    with open(input_path) as f:
        for line in f:
            examples.append(json.loads(line))

    with open(output_path, "w") as f:
        for ex in examples:
            record = convert_example(ex, prompt_template)
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(examples)} examples to {output_path}")

    # Show a sample
    sample = convert_example(examples[0], prompt_template)
    print("\n--- Sample prompt (last 100 chars) ---")
    print(sample["prompt"][-100:])
    print("\n--- Sample response (first 200 chars) ---")
    print(sample["response"][:200])


if __name__ == "__main__":
    main()
