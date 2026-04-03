
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

import json
from pathlib import Path 
from typing import Callable 

PROJECT_ROOT = Path(__file__).parent.parent  # cs336_alignment/ -> project root


def evaluate_vllm(
    vllm_model:LLM, 
    reward_fn: Callable[[str,str], dict[str, float]],
    prompts: list[str],
    ground_truths: list[str],
    eval_sampling_params: SamplingParams,
    output_path: Path
) -> None: 
    """
    Evaluate a language model on a list of prompts, 
    compute evaluation metrics, and serialize results to disk.
    """
    assert len(prompts) == len(ground_truths)
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    responses = [out.outputs[0].text for out in outputs]
    results = [reward_fn(response,ground_truth) for response, ground_truth in zip(responses,ground_truths)]

    records = []
    for i, result in enumerate(results):
        record = {
            "prompt":prompts[i], 
            "response": responses[i],
            "ground_truth": ground_truths[i],
            "format_reward":result["format_reward"],
            "answer_reward":result["answer_reward"],
            "reward":result["reward"]
        }
        records.append(record)
    #
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_format = sum(r["format_reward"] for r in results) / len(results)
    avg_answer = sum(r["answer_reward"] for r in results) / len(results)
    print(f"Avg reward: {avg_reward:.3f} | Format: {avg_format:.3f} | Answer: {avg_answer:.3f}")


def main():
    # read jsonl to list of strings !
    examples = []
    with open(PROJECT_ROOT/"data/gsm8k/test.jsonl") as f:
        for line in f:
            examples.append(json.loads(line))
    
    # format the list of strings 
    with open(PROJECT_ROOT/"cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        template = f.read()

    # build prompts and gts after template is loaded 
    prompts = [template.format(question = ex["question"]) for ex in examples]
    ground_truths = [ex["answer"].split("#### ")[-1].strip() for ex in examples ]

    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen2.5-Math-1.5B",
        trust_remote_code=True
    )
    # Create a sampling params object, stopping generation on newline.
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    
    evaluate_vllm(llm,r1_zero_reward_fn,prompts,ground_truths, sampling_params,
                  output_path = PROJECT_ROOT / "outputs/math_baseline_results.jsonl")

if __name__ == "__main__":
    main()