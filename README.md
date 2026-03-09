# GRPO - Difficulty stratification effects on learning 

## Steps to run:

```bash
uv sync
uv run python train.py --sampling random --run_tag baseline
uv run python train.py --sampling stratified --run_tag baseline
```

You can override model/runtime settings at launch time (useful for HPC):

```bash
--model_name Qwen/Qwen2.5-7B-Instruct
--torch_dtype bfloat16
--num_generations 2
--gradient_checkpointing
--disable_wandb
```

By default each training run is saved under:

```text
experiments/<run_name>/
  checkpoints/
  train/train_run_metadata.json
  eval/
  analysis/
```

## Evaluation

```bash
# Evaluate from a run folder (auto-reads checkpoints and writes into run/eval)
uv run python eval.py --run_dir ./experiments/<run_name> --sampling random

# Evaluate from a checkpoint path and still write under the same run folder
uv run python eval.py --model_path ./experiments/<run_name>/checkpoints --sampling random

# Or evaluate any external model directly
uv run python eval.py --model_path Qwen/Qwen2.5-Math-1.5B-Instruct --sampling unknown --allow_external_model
```

Checkpoint selection is `best` by default and evaluates recent checkpoints on a small subset before full eval.
Useful options:

```bash
--checkpoint_select best
--checkpoint_selection_samples 25
--checkpoint_selection_last_k 5
```

Eval supports the same model-loading controls:

```bash
--torch_dtype auto|bfloat16|float16|float32
--attn_implementation sdpa
--trust_remote_code
```

## HPC Example (Larger Model)

```bash
uv run python train.py \
  --sampling stratified \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --torch_dtype auto \
  --num_generations 2 \
  --gradient_checkpointing \
  --disable_wandb \
  --run_tag hpc-7b
```

```bash
uv run python eval.py \
  --run_dir ./experiments/<run_name> \
  --sampling stratified \
  --torch_dtype auto \
  --attn_implementation sdpa \
  --disable_wandb
```

See detailed change notes in [`docs/project-current-state.md`](docs/project-current-state.md).

## SJSU HPC Notes (CMPE)

Use the cluster scheduler (Slurm) for training/eval jobs instead of running long jobs directly on login nodes.

Interactive GPU shell example:

```bash
srun -p gpu --gres=gpu --time=24:00:00 --pty /bin/bash
```

Then inside the allocated shell:

```bash
module load python3
cd /absolute/path/to/cmpe260-grpo
uv run python train.py --sampling stratified --model_name Qwen/Qwen2.5-7B-Instruct --torch_dtype auto --num_generations 2 --gradient_checkpointing --disable_wandb
```

If you are placed on A100/H100, prefer `--torch_dtype bfloat16`. For broader compatibility across mixed GPU nodes, keep `--torch_dtype auto` (or use `float16` on older GPUs).
The SJSU page also notes a default job time limit, so set `--time` explicitly for longer runs.

## Analysis In Experiments Folder

```bash
uv run python analyze.py \
  --random_eval ./experiments/<random_run>/eval/math500_ns500_tok512.json \
  --stratified_eval ./experiments/<strat_run>/eval/math500_ns500_tok512.json \
  --experiment_dir ./experiments/<experiment_group>
```

## Analysis

```bash
uv run python analyze.py \
  --random_eval ./output-random/eval_math500.json \
  --stratified_eval ./output-stratified/eval_math500.json
```
