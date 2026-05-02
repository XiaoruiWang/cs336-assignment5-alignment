"""
GRPO training script for GSM8K with r1_zero format.
"""
import json
import math
import re
from pathlib import Path

import torch
import wandb
from torch.optim import AdamW
from vllm import SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.sft import init_vllm, load_policy_into_vllm_instance, log_generations
from tests.adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
)

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    # 1. Config (starter hyperparameters from handout)
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    n_grpo_steps = 200
    learning_rate = 1e-5
    rollout_batch_size = 256 # total responses per batch 
    group_size = 8 # (G completions per prompt)
    train_batch_size = 256 # use all responses 
    gradient_accumulation_steps = 128 # how many steps to do in one weight udpate
    clip_value = 1.0
    cliprange = 0.2 #(for grpo_clip only)
    loss_type = "reinforce_with_baseline"  #(or "grpo_clip", "no_baseline")
    advantage_eps = 1e-6 
    normalize_by_std = True
    epochs_per_rollout_batch = 1  #(>1 → off-policy, use grpo_clip)
    eval_interval = 20
    n_eval_examples = 1024
    #    AdamW: weight_decay=0.0, betas=(0.9, 0.95)

    # Derived quantities + asserts (from handout — keep these)
    assert train_batch_size % gradient_accumulation_steps == 0
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    
    assert train_batch_size >= group_size
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    # 2. Init wandb (same define_metric pattern as sft.py)
    #    separate train_step and eval_step x-axes
    output_dir = PROJECT_ROOT/"outputs/grpo"
    wandb_project ="cs336-grpo"
    run_name = f"grpo_lr_{learning_rate}"

    # 2. init wandb; define_metric("train_step"), define_metric("eval_step"),
    #    define_metric("train/*", step_metric="train_step"),
    #    define_metric("eval/*",  step_metric="eval_step")
    wandb.init(project = wandb_project, name = run_name)
    wandb.define_metric("train_step") # one weight update change is one train step. 
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*",  step_metric="eval_step")

    # 3. Load data: we use GSM8K instead of MATH (no cluster access)
    #    source: data/gsm8k/train.jsonl  (fields: "question", "answer")
    #    ground_truth = r["answer"].split("####")[-1].strip()
    #    prompt format: r1_zero — copy the exact prefix from sft.jsonl "prompt" field,
    #    substituting the question. End the prompt at "Assistant: <think>" (no closing tag)
    dataset = []
    dataset_size = 128 # just for smoke test 
    with open(PROJECT_ROOT/"data/gsm8k/train.jsonl") as f:
        for line in f:
            dataset.append(json.loads(line))
    if dataset_size != "full":
        dataset = dataset[:dataset_size]

    with open(PROJECT_ROOT/"cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        template = f.read()

    prompts = [template.format(question = r["question"]) for r in dataset]
    #responses = [r["response"] for r in dataset]
    #ground_truths = [re.search(r"<answer>(.*?)</answer>", r["response"]).group(1) for r in dataset]
    ground_truths = [r["answer"].split("####")[-1].strip() for r in dataset]

    # 4. Load model + tokenizer onto GPU 0
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 5. Init vLLM on GPU 1
    gpu_memory_utilization = 0.85
    vllm_model = init_vllm(model_id, "cuda:1", 42, gpu_memory_utilization)
    sampling_temperature=1.0 
    sampling_min_tokens=4
    sampling_max_tokens=1024
    rollout_sampling_params = SamplingParams(
        n= group_size, 
        temperature = 1.0, min_tokens = 4, max_tokens = 1024, 
        stop = ["</answer>"],
        include_stop_str_in_output = True
    )
    #    stop=["</answer>"], include_stop_str_in_output=True
    #    Also build a separate eval SamplingParams (n=1)
    eval_sampling_params = SamplingParams(
        n= 1, 
        temperature = 1.0, min_tokens = 4, max_tokens = 1024, 
        stop = ["</answer>"],
        include_stop_str_in_output = True
    )
    # 6. Create AdamW optimizer (weight_decay=0.0, betas=(0.9, 0.95))
    optimizer = AdamW(model.parameters(), lr = learning_rate, betas = (0.9,0.95), weight_decay = 0)

    # 7. GRPO loop (n_grpo_steps):
    train_step = 0
    for _ in range(n_grpo_steps):
    #    a. Sample n_prompts_per_rollout_batch prompts (random indices each step)
        indices = torch.randperm(len(prompts))[: n_prompts_per_rollout_batch].tolist()
        batch_prompts = [prompts[i] for i in indices]
        batch_ground_truths = [ground_truths[i] for i in indices]

    #    b. load_policy_into_vllm_instance (sync current policy → vLLM)
        load_policy_into_vllm_instance(model, vllm_model)

    #    c. vllm_model.generate(batch_prompts, rollout_sampling_params)
    #       Each RequestOutput has group_size CompletionOutputs → flatten to
    #       rollout_responses: list of length rollout_batch_size
        outputs = vllm_model.generate(batch_prompts,rollout_sampling_params)
        rollout_responses = []
        for output in outputs:
            for i in range(group_size):
                rollout_responses.append(output.outputs[i].text)
        

    #    d. Build repeated_ground_truths: each gt repeated group_size times
    #       so repeated_ground_truths[i] matches rollout_responses[i]
        repeated_ground_truths = []
        for gt in batch_ground_truths:
            for __ in range(group_size):
                repeated_ground_truths.append(gt)

    #    e. run_compute_group_normalized_rewards → advantages (rollout_batch_size,), raw_rewards
    #       then unsqueeze(-1) for broadcasting: advantages_2d, raw_rewards_2d → (rollout_batch_size, 1)
        advantages, raw_rewards, meta_data = run_compute_group_normalized_rewards(r1_zero_reward_fn,
                                             rollout_responses,
                                             repeated_ground_truths,
                                             group_size,
                                             advantage_eps,
                                             normalize_by_std)
        advantages = advantages.unsqueeze(-1)
        raw_rewards =  raw_rewards.unsqueeze(-1)

    #    f. Tokenize all rollout_batch_size (repeated_prompt, response) pairs with
    #       run_tokenize_prompt_and_output
    #       hint: repeated_prompts — each prompt repeated group_size times, same order as rollout_responses
        repeated_prompts = []
        for p in batch_prompts:
            for _i in range(group_size):
                repeated_prompts.append(p)
        tokens = run_tokenize_prompt_and_output(repeated_prompts,rollout_responses, tokenizer)

    #    g. Get old_log_probs with run_get_response_log_probs (torch.no_grad())
    #       → snapshot before any gradient steps
        old_log_probs_list = []
        with torch.no_grad():
            for step in range(n_microbatches_per_rollout_batch):
                start = step * micro_train_batch_size
                end = (step + 1) * micro_train_batch_size
                result = run_get_response_log_probs(model, tokens["input_ids"][start:end], tokens["labels"][start:end], False)
                old_log_probs_list.append(result["log_probs"])
        old_log_probs = torch.cat(old_log_probs_list, dim=0)


    #    h. For epochs_per_rollout_batch:
    #       optimizer.zero_grad()  ← must be BEFORE the microbatch loop
    #       For each microbatch (n_microbatches_per_rollout_batch):
    #           - slice input_ids, labels, response_mask, advantages_2d, raw_rewards_2d, old_log_probs
    #             by [start:end] where start = mb_idx * micro_train_batch_size
    #           - run_get_response_log_probs → policy_log_probs  (grad enabled)
    #           - run_grpo_microbatch_train_step(loss_type=loss_type, ...)
    #             pass gradient_accumulation_steps = n_microbatches_per_rollout_batch
    #       clip_grad_norm_(clip_value), optimizer.step()
    #       train_step += 1
    #       log train/loss, train/avg_reward, train/avg_format_reward to wandb
        for e in range(epochs_per_rollout_batch):
            optimizer.zero_grad()
            for step in range(n_microbatches_per_rollout_batch):
                start = step*micro_train_batch_size
                end = (step+1)*micro_train_batch_size
                sliced_tokens_input_ids =  tokens["input_ids"][start:end, :]
                sliced_tokens_labels = tokens["labels"][start:end, :]
                sliced_tokens_response_mask = tokens["response_mask"][start:end, :]
                new_log_probs = run_get_response_log_probs(model,sliced_tokens_input_ids,sliced_tokens_labels , False)
                policy_log_probs = new_log_probs["log_probs"]
                sliced_raw_rewards = raw_rewards[start:end ,:]
                sliced_advantages = advantages[start:end ,:]
                sliced_old_log_probs = old_log_probs[start:end ,:]
                loss, _ = run_grpo_microbatch_train_step(policy_log_probs, sliced_tokens_response_mask, 
                                               gradient_accumulation_steps, loss_type,
                                               sliced_raw_rewards, sliced_advantages,
                                               sliced_old_log_probs,
                                               cliprange)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            train_step +=1
            wandb.log({"train/loss": loss.item(), "train_step": train_step})
            if train_step % eval_interval == 0: 
                load_policy_into_vllm_instance(model, vllm_model)
                stats = log_generations(vllm_model,prompts[:50],ground_truths[:50],
                                        r1_zero_reward_fn, eval_sampling_params,model,tokenizer)
                wandb.log({
                            "eval/avg_reward": stats["avg_reward"],
                            "eval/avg_format_reward": stats["avg_format_reward"],
                            "eval/avg_response_length": stats["avg_response_length"],
                            "eval/avg_token_entropy": stats["avg_token_entropy"],
                            "eval_step": train_step,
                        })
    #    i. Every eval_interval grpo steps:
    #       load_policy_into_vllm_instance, log_generations on all_prompts[:n_eval_examples]
    #       log eval/* to wandb

    # 8. Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
