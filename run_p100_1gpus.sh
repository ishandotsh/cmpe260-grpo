#!/bin/bash

#SBATCH --job-name=grpo_default
#SBATCH --output=grpo_default-%j.out
#SBATCH --error=grpo_default-%j.err

#SBATCH --partition=gpuqs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

SAMPLING="${1:?ERROR: missing argument 1 (sampling)}"
REWARD="${2:?ERROR: missing argument 2 (reward)}"

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BASHRCSOURCED=1

cd /home/018198687/RL/grpo-random-v-strat-main
source ~/.bashrc

ENV_FILE=/home/018198687/RL/grpo-random-v-strat-main/.env
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

echo "Node: $(hostname)"
nvidia-smi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set"
  exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "ERROR: WANDB_API_KEY is not set"
  exit 1
fi

echo "Logging into Hugging Face and Weights & Biases..."
uv run python - <<'PY'
import os
from huggingface_hub import login
import wandb

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
wandb.login(key=os.environ["WANDB_API_KEY"])

print("HF login successful")
print("W&B login successful")
PY

uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

#echo "Starting training..."
# uv run python train.py --sampling random --reward binary
echo "Starting training with --sampling ${SAMPLING} --reward ${REWARD}"
uv run python train.py --sampling "$SAMPLING" --reward "$REWARD"

