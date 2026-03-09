# GRPO - Difficulty stratification effects on learning 

## Steps to run:

```bash
uv sync
uv run python train.py --sampling random --run_tag baseline
uv run python train.py --sampling stratified --run_tag baseline
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
