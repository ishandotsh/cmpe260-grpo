"""
    uv run python train.py --sampling random
    uv run python train.py --sampling stratified
    uv run python train.py --sampling random --max_steps 20
"""

import argparse
import os
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer

from config import MODEL_NAME, WANDB_PROJECT, get_grpo_config
from data import StratifiedDifficultySampler, load_competition_math
from reward import make_reward_fn


class DifficultyRewardCallback(TrainerCallback):
    def __init__(self, log_every: int = 50):
        self.log_every = log_every
        self._step_rewards: dict[int, list[float]] = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return
        if not self._step_rewards:
            return
        for level, rewards in sorted(self._step_rewards.items()):
            avg = sum(rewards) / len(rewards) if rewards else 0.0
            if wandb.run:
                wandb.log(
                    {f"reward/difficulty_{level}": avg},
                    step=state.global_step,
                )
        self._step_rewards.clear()


def main():
    parser = argparse.ArgumentParser(description="GRPO training with difficulty-based sampling")
    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        required=True,
        help="Sampling strategy: random (mixed difficulty) or stratified (same difficulty per batch)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (-1 = full epochs)")
    args = parser.parse_args()

    # ---- Config ----
    grpo_config = get_grpo_config(
        sampling=args.sampling,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        max_steps=args.max_steps,
    )

    # ---- W&B ----
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    dataset = load_competition_math()

    # ---- Reward ----
    reward_fn = make_reward_fn(answer_key="solution")

    # ---- Trainer ----
    trainer_kwargs = {}

    if args.sampling == "stratified":
        sampler = StratifiedDifficultySampler(
            dataset,
            batch_size=grpo_config.per_device_train_batch_size,
        )
        trainer_kwargs["data_collator"] = None  

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[DifficultyRewardCallback(log_every=50)],
    )

    # Override dataloader for stratified sampling
    if args.sampling == "stratified":
        _original_get_dataloader = trainer.get_train_dataloader

        def _stratified_dataloader():
            dataloader = _original_get_dataloader()
            sampler = StratifiedDifficultySampler(
                dataset,
                batch_size=grpo_config.per_device_train_batch_size,
            )
            dataloader.batch_sampler = None
            dataloader.sampler = sampler
            return dataloader

        trainer.get_train_dataloader = _stratified_dataloader

    # ---- Train ----
    trainer.train()
    trainer.save_model(grpo_config.output_dir)
    print(f"Model saved to {grpo_config.output_dir}")

    repo_name = Path(grpo_config.output_dir).name
    print(f"Pushing model to HuggingFace Hub: {repo_name}...")
    trainer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
