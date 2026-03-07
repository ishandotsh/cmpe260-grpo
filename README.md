# GRPO - Difficulty stratification effects on learning 

## Steps to run:

```py
uv sync
uv run python train.py --sampling {random,stratified}
```

## Evaluation

```bash
uv run python eval.py --model_path ./output-random --sampling random
uv run python eval.py --model_path ./output-stratified --sampling stratified
```

## Analysis

```bash
uv run python analyze.py \
  --random_eval ./output-random/eval_math500.json \
  --stratified_eval ./output-stratified/eval_math500.json
```

