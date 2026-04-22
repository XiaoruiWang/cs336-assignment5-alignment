"""
SFT training script for GSM8K with r1_zero format.
"""
import json
from pathlib import Path
from typing import Callable
import numpy as np

import torch
from unittest.mock import patch
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from tests.adapters import run_compute_entropy
import math
import re 

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

PROJECT_ROOT = Path(__file__).parent.parent

import wandb
from torch.optim import AdamW
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)


def log_generations(
    vllm_model: LLM,
    prompts: list[str],
    ground_truths: list[str],
    reward_fn: Callable,
    sampling_params: SamplingParams,
    model,
    tokenizer
) -> dict:
    """
    Generate responses from the model and log stats.

    Returns a dict with:
        - "examples": list of per-example dicts (prompt, response, ground_truth, rewards, entropy, length)
        - "avg_reward": float
        - "avg_format_reward": float
        - "avg_response_length": float
        - "avg_response_length_correct": float
        - "avg_response_length_incorrect": float
        - "avg_token_entropy": float
    """
    # Step 1: generate responses using vLLM
    outputs =  vllm_model.generate(prompts,sampling_params)
    responses =  [output.outputs[0].text for output in outputs]

    # Step 2: score each response
    rewards = [reward_fn(r,g) for r,g in zip(responses,ground_truths)]

    # Step 3: compute token entropy for each response
    # hint: tokenize responses, 
    tokenized = tokenizer(responses, padding= True, return_tensors ="pt", add_special_tokens =False)
    input_ids = tokenized["input_ids"]
    input_ids = input_ids.to(model.device)
    #labels = labels.to(model.device)
    # run model forward pass, 
    with torch.no_grad():
        logits = model(input_ids).logits
    # call run_compute_entropy
    # then average entropy over response tokens only
    attention_mask = tokenized["attention_mask"]
    attention_mask = attention_mask.to(model.device)
    entropy = run_compute_entropy(logits)                    # (B, S)
    entropy_avg = (entropy * attention_mask).sum(-1) / attention_mask.sum(-1)  # per-example avg


    # Step 4: collect per-example records
    examples = []
    for i in range(len(prompts)):
        record = {
            "prompt": prompts[i],
            "response": responses[i],
            "ground_truth": ground_truths[i],
            "format_reward": rewards[i]["format_reward"],
            "answer_reward": rewards[i]["answer_reward"],
            "reward": rewards[i]["reward"],
            "avg_token_entropy": entropy_avg[i].item(), # entropy_avg shape is (B,)
            "response_length": attention_mask[i].sum().item(),   # number of tokens in response
        }
        examples.append(record)

    # Step 5: compute aggregate stats
    avg_reward = np.mean([r["reward"] for r in examples])
    avg_format_reward = np.mean([r["format_reward"] for r in examples])
    correct = [e for e in examples if e["reward"] == 1.0]
    incorrect = [e for e in examples if e["reward"] == 0.0]
    avg_response_length = np.mean([r["response_length"] for r in examples])
    avg_response_length_correct = np.mean([e['response_length'] for e in correct]) if correct else 0.0
    avg_response_length_incorrect = np.mean([e['response_length'] for e in incorrect]) if incorrect else 0.0
    avg_token_entropy = np.mean([r["avg_token_entropy"] for r in examples])

    return {
        "examples": examples,
        "avg_reward": avg_reward,
        "avg_format_reward": avg_format_reward,
        "avg_response_length": avg_response_length,
        "avg_response_length_correct": avg_response_length_correct,
        "avg_response_length_incorrect": avg_response_length_incorrect,
        "avg_token_entropy": avg_token_entropy,
    }


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """Start the inference process, holding a model on a GPU separate from the policy."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM) -> None:
    """Sync policy weights from the HF model into the vLLM engine."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def main():
    # 1. config: model_id, sft_data_path, dataset_size (128/256/512/1024/full),
    #            lr, batch_size, gradient_accumulation_steps, clip_value=1.0,
    #            eval_interval, output_dir, wandb project/run name
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    sft_data_path = PROJECT_ROOT/"data/gsm8k/sft.jsonl"
    #dataset_size = 1024  #128, 256, 512, 1024, "full"
    dataset_size = "full"
    lr = 1e-5  
    batch_size = 2 
    gradient_accumulation_steps = 4
    clip_value = 1.0 
    eval_interval = 5
    output_dir = PROJECT_ROOT/"outputs/sft"
    wandb_project ="cs336-sft"
    run_name = f"sft-filtered"

    # 2. init wandb; define_metric("train_step"), define_metric("eval_step"),
    #    define_metric("train/*", step_metric="train_step"),
    #    define_metric("eval/*",  step_metric="eval_step")
    wandb.init(project = wandb_project, name = run_name)
    wandb.define_metric("train_step") # one weight update change is one train step. 
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # 3. load sft.jsonl; slice to dataset_size if not full
    #    separate prompts and ground_truths for eval
    dataset = []
    with open(PROJECT_ROOT/"data/gsm8k/sft.jsonl") as f:
        for line in f:
            dataset.append(json.loads(line))
    if dataset_size != "full":
        dataset = dataset[:dataset_size]
    gsm_data = [json.loads(l) for l in open(PROJECT_ROOT/"data/gsm8k/train.jsonl")]
    gsm_lookup = {r["question"]: r["answer"].split("####")[-1].strip() for r in gsm_data}
    def get_ground_truth(prompt):
        m = re.search(r"User: (.*?)\nAssistant:", prompt, re.DOTALL)
        return gsm_lookup.get(m.group(1).strip()) if m else None
    dataset = [r for r in dataset if (gt := get_ground_truth(r["prompt"])) and r1_zero_reward_fn(r["response"], gt)["reward"] == 1.0]
    print(f"Filtered dataset size: {len(dataset)}")
    prompts = [r["prompt"] for r in dataset]
    responses = [r["response"] for r in dataset]
    ground_truths = [re.search(r"<answer>(.*?)</answer>", r["response"]).group(1) for r in dataset]

    # 4. load model + tokenizer onto GPU 0
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 5. init_vllm on GPU 1; build SamplingParams for eval rollouts
    vllm_model = init_vllm(model_id, "cuda:1",42)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    # 6. create AdamW optimizer
    optimizer = AdamW(model.parameters(), lr = lr)

    iterations  = math.ceil(len(dataset)/ batch_size)
    # 7. training loop (iterate over batches, track train_step):
    #    a. tokenize batch with run_tokenize_prompt_and_output
    #    b. run_get_response_log_probs → policy_log_probs
    #    c. run_sft_microbatch_train_step → loss (handles grad accumulation)
    #    d. every gradient_accumulation_steps: clip_grad_norm_(1.0), optimizer.step(), zero_grad()
    #    e. log train/loss to wandb with train_step
    #    f. every eval_interval steps:
    #       - load_policy_into_vllm_instance (sync weights to vLLM)
    #       - log_generations → stats
    #       - log eval/* to wandb with eval_step
    train_step = 0
    for iter in range(iterations):
        #inputs = dataset[iter*batch_size:(iter+1)*batch_size ]
        tokenized = run_tokenize_prompt_and_output(prompts[iter*batch_size:(iter+1)*batch_size],
                                                   responses[iter*batch_size:(iter+1)*batch_size],
                                                   tokenizer)
        result = run_get_response_log_probs(model,tokenized["input_ids"],
                                                      tokenized["labels"], return_token_entropy=False)
        loss,_ = run_sft_microbatch_train_step(result["log_probs"],tokenized["response_mask"], 
                                               gradient_accumulation_steps)
        if (1+iter) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            #loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_step += 1
            wandb.log({"train/loss": loss.item(), "train_step": train_step})
            if train_step  % eval_interval == 0:
                load_policy_into_vllm_instance(model,vllm_model)
                stats = log_generations(vllm_model,prompts[:50],ground_truths[:50],
                                        r1_zero_reward_fn, sampling_params,model,tokenizer)
                wandb.log({
                            "eval/avg_reward": stats["avg_reward"],
                            "eval/avg_format_reward": stats["avg_format_reward"],
                            "eval/avg_response_length": stats["avg_response_length"],
                            "eval/avg_token_entropy": stats["avg_token_entropy"],
                            "eval_step": train_step,
                        })
    # 8. save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)



if __name__ == "__main__":
    main()
