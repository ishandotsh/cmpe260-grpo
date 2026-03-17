# GRPO - Difficulty stratification effects on learning 

## Steps to train:

```py
uv sync
uv run python train.py --sampling {random,stratified,curriculum} --reward {binary,difficulty_weighted,easy_penalty}
```