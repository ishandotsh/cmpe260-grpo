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
from experiment_layout import build_run_name, ensure_run_dirs, get_run_paths
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


def _resolve_torch_dtype(dtype_name: str):
    dtype_name = dtype_name.lower()
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def main():
    parser = argparse.ArgumentParser(description="GRPO training with difficulty-based sampling")
    parser.add_argument(
        "--sampling",
        choices=["random", "stratified"],
        required=True,
        help="Sampling strategy: random (mixed difficulty) or stratified (same difficulty per batch)",
    )
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--experiments_root", type=str, default="./experiments")
    parser.add_argument("--run_name", type=str, default=None, help="Optional explicit run folder name")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional short suffix for run naming")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--log_every", type=int, default=50, help="Log reward stats every N train steps")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override max training steps (-1 = full epochs)")
    parser.add_argument(
        "--num_generations",
        type=int,
        default=None,
        help="Override GRPO num_generations (lower can reduce memory for larger models)",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype used when loading the model",
    )
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument(
        "--min_steps_before_eval",
        type=int,
        default=200,
        help="Warn if training ends before this many steps",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()

    seed = 1337
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = build_run_name(
            sampling=args.sampling,
            model_name=args.model_name,
            seed=seed,
            run_tag=args.run_tag,
        )

    if args.output_dir:
        checkpoint_dir = Path(args.output_dir)
        run_dir = checkpoint_dir.parent if checkpoint_dir.name == "checkpoints" else checkpoint_dir
        run_paths = {
            "run_dir": run_dir,
            "checkpoint_dir": checkpoint_dir,
            "train_dir": run_dir / "train",
            "eval_dir": run_dir / "eval",
            "analysis_dir": run_dir / "analysis",
        }
    else:
        run_paths = get_run_paths(args.experiments_root, run_name)
        checkpoint_dir = run_paths["checkpoint_dir"]
        run_dir = run_paths["run_dir"]
    ensure_run_dirs(run_paths)
    print(f"Run directory: {run_dir.resolve()}")

    # ---- Config ----
    grpo_config = get_grpo_config(
        sampling=args.sampling,
        output_dir=str(checkpoint_dir),
        wandb_project=args.wandb_project,
        max_steps=args.max_steps,
        run_name=run_name,
        num_generations=args.num_generations,
        report_to="none" if args.disable_wandb else "wandb",
    )

    run_metadata = {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "sampling": args.sampling,
        "seed": grpo_config.seed,
        "model_name": args.model_name,
        "torch_dtype": args.torch_dtype,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "num_generations": grpo_config.num_generations,
        "wandb_enabled": not args.disable_wandb,
        "checkpoint_path": str(checkpoint_dir.resolve()),
        "eval_dir": str(run_paths["eval_dir"].resolve()),
        "analysis_dir": str(run_paths["analysis_dir"].resolve()),
        "min_steps_before_eval": args.min_steps_before_eval,
        "commit_hash": get_git_commit_hash(),
    }
    metadata_path = write_json(run_paths["train_dir"] / "train_run_metadata.json", run_metadata)
    print(f"Saved run metadata to {metadata_path}")

    # ---- W&B ----
    if not args.disable_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if wandb.run is None:
            wandb.init(project=args.wandb_project, name=grpo_config.run_name, config=run_metadata)
        else:
            wandb.config.update(run_metadata, allow_val_change=True)

    # ---- Model ----
    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(args.torch_dtype),
        "attn_implementation": args.attn_implementation,
        "trust_remote_code": args.trust_remote_code,
    }
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
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

    checkpoint_root = Path(grpo_config.output_dir)
    checkpoint_folders = sorted(
        [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    )
    training_summary = {
        "run_name": run_name,
        "global_step": trainer.state.global_step,
        "save_steps": grpo_config.save_steps,
        "num_checkpoints": len(checkpoint_folders),
        "checkpoint_paths": [str(p.resolve()) for p in checkpoint_folders],
    }
    training_summary_path = write_json(run_paths["train_dir"] / "training_summary.json", training_summary)
    print(f"Saved training summary to {training_summary_path}")
    if trainer.state.global_step < args.min_steps_before_eval:
        print(
            f"Warning: training ended at step {trainer.state.global_step}, below "
            f"--min_steps_before_eval={args.min_steps_before_eval}. "
            "Eval may underestimate trained performance."
        )

    if args.push_to_hub:
        repo_name = run_name
        print(f"Pushing model to HuggingFace Hub: {repo_name}...")
        trainer.push_to_hub(repo_name)


if __name__ == "__main__":
    main()
