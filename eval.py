"""
uv run python eval.py --model_path ./output-random
uv run python eval.py --model_path ./output-stratified --sampling stratified --max_samples 200
"""

import argparse
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import EVAL_DATASET, MODEL_NAME, WANDB_PROJECT
from data import load_math500
from reporting import build_accuracy_summary, get_git_commit_hash, write_json
from reward import check_answer


def generate_completion(model, tokenizer, messages, max_new_tokens, temperature):
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = "\n".join(message.get("content", "") for message in messages)

    encoded = tokenizer(prompt_text, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        generated = model.generate(**encoded, **generation_kwargs)
    completion_tokens = generated[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(completion_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on MATH-500")
    parser.add_argument("--model_path", type=str, required=True, help="Path or model id to evaluate")
    parser.add_argument(
        "--sampling",
        choices=["random", "stratified", "unknown"],
        default="unknown",
        help="Sampling strategy used for this checkpoint",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Run seed metadata value for this eval")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of eval samples (-1 uses full dataset)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--qualitative_samples",
        type=int,
        default=20,
        help="How many generated examples to store for qualitative analysis",
    )
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT)
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    resolved_model_path = str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_dataset = load_math500()
    if args.max_samples > 0:
        eval_dataset = eval_dataset.select(range(min(args.max_samples, len(eval_dataset))))

    records = []
    qualitative = []

    for index, example in enumerate(eval_dataset):
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            messages=example["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        correct = check_answer(completion, example["solution"])
        record = {
            "index": index,
            "difficulty": int(example["difficulty"]),
            "correct": float(correct),
        }
        records.append(record)

        if len(qualitative) < args.qualitative_samples:
            qualitative.append(
                {
                    "index": index,
                    "difficulty": int(example["difficulty"]),
                    "problem": example["problem"],
                    "completion": completion,
                    "solution": example["solution"],
                    "correct": float(correct),
                }
            )

        if index > 0 and index % 25 == 0:
            print(f"Processed {index}/{len(eval_dataset)} samples...")

    metrics = build_accuracy_summary(records)
    artifact = {
        "metadata": {
            "sampling": args.sampling,
            "seed": args.seed,
            "model_name": getattr(model.config, "_name_or_path", MODEL_NAME),
            "checkpoint_path": resolved_model_path,
            "commit_hash": get_git_commit_hash(),
            "eval_dataset": EVAL_DATASET,
            "num_samples": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
        "metrics": metrics,
        "qualitative_samples": qualitative,
        "records": records,
    }

    output_path = Path(args.output_json) if args.output_json else Path(args.model_path) / "eval_math500.json"
    output_path = write_json(output_path, artifact)
    print(f"Saved eval artifact to {output_path}")

    if not args.disable_wandb:
        run_name = f"eval-{Path(args.model_path).name}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            job_type="eval",
            config=artifact["metadata"],
        )
        wandb_payload = {
            "eval/overall_accuracy": metrics["overall"]["accuracy"],
            "eval/overall_std": metrics["overall"]["std"],
            "eval/overall_var": metrics["overall"]["var"],
            "eval/num_samples": metrics["overall"]["count"],
        }
        for level, level_metrics in metrics["per_difficulty"].items():
            wandb_payload[f"eval/accuracy_difficulty_{level}"] = level_metrics["accuracy"]
            wandb_payload[f"eval/std_difficulty_{level}"] = level_metrics["std"]
            wandb_payload[f"eval/var_difficulty_{level}"] = level_metrics["var"]
            wandb_payload[f"eval/count_difficulty_{level}"] = level_metrics["count"]
        wandb.log(wandb_payload)
        run.summary["eval_artifact_path"] = str(output_path.resolve())
        wandb.finish()


if __name__ == "__main__":
    main()
