# Workflow Guide

## Section 3 Setup: Connecting to RunPod and Running the Baseline

### Step 1: Start a RunPod Pod
- GPU: H100 80GB SXM (preferred) or A100 80GB
- Template: PyTorch (has CUDA pre-installed)
- Add your SSH public key in RunPod → Settings → SSH Public Keys

### Step 2: SSH into the Pod
RunPod provides the exact SSH command on the pod page. It looks like:
```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519
```

### Step 3: Install uv
`uv` is not pre-installed on RunPod. Install it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Note: The installer says to run `source $HOME/.cargo/env` but on RunPod the correct path is `source $HOME/.local/bin/env`.

### Step 4: Clone the repo and install dependencies
```bash
git clone https://github.com/XiaoruiWang/cs336-assignment5-alignment.git
cd cs336-assignment5-alignment
uv sync
```

`uv sync` will install all dependencies including torch, vllm, and flash-attn. This takes a few minutes on first run.

### Step 5: Download the model
`huggingface-cli` lives inside the uv environment, so prefix with `uv run`:
```bash
uv run huggingface-cli download Qwen/Qwen2.5-Math-1.5B
```

The model (~3GB) is cached in `~/.cache/huggingface/`. Only needed once per pod.


### Step 5-a: Set up the weight and bias which will be needed for experiment
export WANDB_API_KEY=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


### Step 6: Run experiments
```bash
uv run python cs336_alignment/math_baseline.py
uv run python -m cs336_alignment.sft

```

Results are saved to `outputs/math_baseline_results.jsonl`.

---

## Everyday Workflow (After Initial Setup)

### On Windows (writing code)
```bash
# After making changes
git add <files>
git commit -m "your message"
git push
```

### On RunPod (running experiments)
```bash
# Pull latest code
cd cs336-assignment5-alignment
git pull

# Run your script
uv run python cs336_alignment/<your_script>.py
```

### Resuming a stopped pod
If your pod was stopped and restarted, `uv` and the repo are still there but you need to re-source the path:
```bash
source $HOME/.local/bin/env
cd cs336-assignment5-alignment
git pull
```

---

## Exporting Results from RunPod to GitHub

After running an experiment, push results from RunPod to GitHub so you can pull them on Windows.

### Step 1: Configure git identity (once per pod)
RunPod pods don't have git identity by default. Run this before your first commit:
```bash
git config --global user.email "xiaoruiw86@gmail.com"
git config --global user.name "XiaoruiWang"
```

### Step 2: Stage, commit, and push results
```bash
git add outputs/math_baseline_results.jsonl
git commit -m "Add zero-shot baseline results"
git push
```

### Step 3: Pull on Windows
Back on your local machine:
```bash
git pull
```

The results file (`outputs/math_baseline_results.jsonl`) is now available locally for analysis.
