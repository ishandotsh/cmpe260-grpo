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
from reporting import get_git_commit_hash, summarize_grouped_values, write_json
from reward import DifficultyRewardTracker, make_reward_fn


class DifficultyRewardCallback(TrainerCallback):
    def __init__(self, reward_tracker: DifficultyRewardTracker, log_every: int = 50):
        self.reward_tracker = reward_tracker
        self.log_every = log_every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every != 0:
            return
        grouped_rewards = self.reward_tracker.pop()
        if not grouped_rewards:
            return
        grouped_stats = summarize_grouped_values(grouped_rewards)
        wandb_payload = {}
        for level, stats in sorted(grouped_stats.items(), key=lambda pair: int(pair[0])):
            wandb_payload[f"reward/difficulty_{level}/mean"] = stats["mean"]
            wandb_payload[f"reward/difficulty_{level}/std"] = stats["std"]
            wandb_payload[f"reward/difficulty_{level}/var"] = stats["var"]
            wandb_payload[f"reward/difficulty_{level}/count"] = stats["count"]
        if wandb.run and wandb_payload:
            wandb.log(wandb_payload, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        grouped_rewards = self.reward_tracker.pop()
        if not grouped_rewards:
            return
        grouped_stats = summarize_grouped_values(grouped_rewards)
        wandb_payload = {}
        for level, stats in sorted(grouped_stats.items(), key=lambda pair: int(pair[0])):
            wandb_payload[f"reward/difficulty_{level}/mean"] = stats["mean"]
            wandb_payload[f"reward/difficulty_{level}/std"] = stats["std"]
            wandb_payload[f"reward/difficulty_{level}/var"] = stats["var"]
            wandb_payload[f"reward/difficulty_{level}/count"] = stats["count"]
        if wandb.run and wandb_payload:
            wandb.log(wandb_payload, step=state.global_step)


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
    parser.add_argument("--log_every", type=int, default=50, help="Log reward stats every N train steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (-1 = full epochs)")
    args = parser.parse_args()

    # ---- Config ----
    grpo_config = get_grpo_config(
        sampling=args.sampling,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        max_steps=args.max_steps,
    )
    output_dir = Path(grpo_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "sampling": args.sampling,
        "seed": grpo_config.seed,
        "model_name": MODEL_NAME,
        "checkpoint_path": str(output_dir.resolve()),
        "commit_hash": get_git_commit_hash(),
    }
    metadata_path = write_json(output_dir / "train_run_metadata.json", run_metadata)
    print(f"Saved run metadata to {metadata_path}")

    # ---- W&B ----
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if wandb.run is None:
        wandb.init(project=args.wandb_project, name=grpo_config.run_name, config=run_metadata)
    else:
        wandb.config.update(run_metadata, allow_val_change=True)

    # ---- Model ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    dataset = load_competition_math()

    # ---- Reward ----
    reward_tracker = DifficultyRewardTracker()
    reward_fn = make_reward_fn(answer_key="solution", reward_tracker=reward_tracker)

    # ---- Trainer ----
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[DifficultyRewardCallback(reward_tracker=reward_tracker, log_every=args.log_every)],
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
