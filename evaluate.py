"""Evaluate a saved checkpoint on MATH-500.

Usage:
    uv run python evaluate.py --checkpoint ./output-random
    uv run python evaluate.py --checkpoint ./output-stratified
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from config import SYSTEM_PROMPT
from data import load_math500
from reward import check_answer


def generate_completions(model, tokenizer, prompts: list[list[dict]], batch_size: int = 4) -> list[str]:
    """Generate greedy completions for a list of chat-formatted prompts."""
    completions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i : i + batch_size]
        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch_prompts]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        # Decode only the newly generated tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            completions.append(generated)
    return completions


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on MATH-500")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved model checkpoint")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval dataset
    dataset = load_math500()
    prompts = dataset["prompt"]
    solutions = dataset["solution"]
    difficulties = dataset["difficulty"]

    # Generate
    completions = generate_completions(model, tokenizer, prompts, batch_size=args.batch_size)

    # Score
    correct_by_level = defaultdict(int)
    total_by_level = defaultdict(int)
    total_correct = 0

    for completion, solution, level in zip(completions, solutions, difficulties):
        score = check_answer(completion, solution)
        total_by_level[level] += 1
        if score > 0.5:
            correct_by_level[level] += 1
            total_correct += 1

    # Results
    overall_acc = total_correct / len(completions) if completions else 0.0
    per_level = {}
    for level in sorted(total_by_level.keys()):
        acc = correct_by_level[level] / total_by_level[level] if total_by_level[level] else 0.0
        per_level[f"level_{level}"] = {
            "correct": correct_by_level[level],
            "total": total_by_level[level],
            "accuracy": round(acc, 4),
        }

    results = {
        "checkpoint": args.checkpoint,
        "overall_accuracy": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_problems": len(completions),
        "per_difficulty": per_level,
    }

    # Print table
    print(f"\n{'='*50}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Overall Accuracy: {overall_acc:.2%} ({total_correct}/{len(completions)})")
    print(f"{'='*50}")
    print(f"{'Level':<10} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print(f"{'-'*40}")
    for level in sorted(total_by_level.keys()):
        acc = correct_by_level[level] / total_by_level[level]
        print(f"{level:<10} {correct_by_level[level]:<10} {total_by_level[level]:<10} {acc:<10.2%}")

    # Save JSON
    if args.output:
        output_path = Path(args.output)
    else:
        from datetime import datetime
        checkpoint_name = Path(args.checkpoint).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("eval_results") / f"{checkpoint_name}_{timestamp}" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
