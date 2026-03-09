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


def _checkpoint_step(checkpoint_path):
    name = checkpoint_path.name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except ValueError:
            return -1
    return -1


def _find_checkpoints(checkpoint_root):
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.exists() or not checkpoint_root.is_dir():
        return []
    return sorted(
        [path for path in checkpoint_root.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=_checkpoint_step,
    )


def _score_checkpoint(
    checkpoint_path,
    eval_examples,
    max_new_tokens,
    temperature,
    torch_dtype,
    attn_implementation,
    trust_remote_code,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint_path),
        padding_side="left",
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0.0
    for example in eval_examples:
        completion = generate_completion(
            model=model,
            tokenizer=tokenizer,
            messages=example["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        correct += check_answer(completion, example["solution"])

    total = len(eval_examples)
    accuracy = (correct / total) if total else 0.0

    # Explicit cleanup before loading the next checkpoint.
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return accuracy


def select_checkpoint(
    checkpoint_root,
    strategy,
    requested_step,
    selection_examples,
    selection_max_new_tokens,
    temperature,
    selection_last_k,
    torch_dtype,
    attn_implementation,
    trust_remote_code,
):
    checkpoints = _find_checkpoints(checkpoint_root)
    if not checkpoints:
        return None, []

    if strategy == "latest":
        return checkpoints[-1], []

    if strategy == "step":
        for checkpoint in checkpoints:
            if _checkpoint_step(checkpoint) == requested_step:
                return checkpoint, []
        raise ValueError(f"Requested checkpoint step {requested_step} was not found under {checkpoint_root}")

    # strategy == "best"
    candidates = checkpoints[-selection_last_k:] if selection_last_k > 0 else checkpoints
    scored = []
    best_checkpoint = None
    best_accuracy = -1.0
    for checkpoint in candidates:
        step = _checkpoint_step(checkpoint)
        print(f"Scoring checkpoint {checkpoint.name} on {len(selection_examples)} samples...")
        accuracy = _score_checkpoint(
            checkpoint_path=checkpoint,
            eval_examples=selection_examples,
            max_new_tokens=selection_max_new_tokens,
            temperature=temperature,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        scored.append({"checkpoint": str(checkpoint.resolve()), "step": step, "accuracy": accuracy})
        if accuracy > best_accuracy or (accuracy == best_accuracy and step > _checkpoint_step(best_checkpoint)):
            best_accuracy = accuracy
            best_checkpoint = checkpoint

    return best_checkpoint, scored


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on MATH-500")
    parser.add_argument("--model_path", type=str, default=None, help="Path or model id to evaluate")
    parser.add_argument("--run_dir", type=str, default=None, help="Run folder with /checkpoints and /eval subfolders")
    parser.add_argument("--checkpoint_select", choices=["best", "latest", "step"], default="best")
    parser.add_argument("--checkpoint_step", type=int, default=None, help="Required when --checkpoint_select step")
    parser.add_argument("--checkpoint_selection_samples", type=int, default=25)
    parser.add_argument("--checkpoint_selection_max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--checkpoint_selection_last_k",
        type=int,
        default=5,
        help="When selecting best checkpoint, evaluate only the most recent K checkpoints",
    )
    parser.add_argument(
        "--allow_external_model",
        action="store_true",
        help="Allow direct HF model ids when sampling is random/stratified",
    )
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
        "--torch_dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype used when loading the model",
    )
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument("--trust_remote_code", action="store_true")
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
    resolved_torch_dtype = _resolve_torch_dtype(args.torch_dtype)

    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.model_path is None:
            args.model_path = str(run_dir / "checkpoints")
    if args.model_path is None:
        raise ValueError("Provide --model_path or --run_dir")

    model_path_obj = Path(args.model_path)
    if run_dir is None and model_path_obj.exists() and model_path_obj.is_dir() and model_path_obj.name == "checkpoints":
        run_dir = model_path_obj.parent

    if (
        args.sampling in {"random", "stratified"}
        and run_dir is None
        and not model_path_obj.exists()
        and not args.allow_external_model
    ):
        raise ValueError(
            "Refusing to evaluate external model id for a sampling run. "
            "Use --run_dir (or local checkpoint path), or pass --allow_external_model to override."
        )

    checkpoint_selection_scores = []
    selected_checkpoint = None
    if model_path_obj.exists() and model_path_obj.is_dir() and model_path_obj.name == "checkpoints":
        if args.checkpoint_select == "step" and args.checkpoint_step is None:
            raise ValueError("--checkpoint_step is required when --checkpoint_select step")

        selection_dataset = load_math500()
        selection_count = min(args.checkpoint_selection_samples, len(selection_dataset))
        selection_examples = [selection_dataset[i] for i in range(selection_count)]

        selected_checkpoint, checkpoint_selection_scores = select_checkpoint(
            checkpoint_root=model_path_obj,
            strategy=args.checkpoint_select,
            requested_step=args.checkpoint_step,
            selection_examples=selection_examples,
            selection_max_new_tokens=args.checkpoint_selection_max_new_tokens,
            temperature=args.temperature,
            selection_last_k=args.checkpoint_selection_last_k,
            torch_dtype=resolved_torch_dtype,
            attn_implementation=args.attn_implementation,
            trust_remote_code=args.trust_remote_code,
        )
        if selected_checkpoint is None:
            if args.checkpoint_select == "step":
                raise ValueError(f"No checkpoints found under {model_path_obj}")
            print(f"No checkpoint-* folders found under {model_path_obj}; using root model files.")
        else:
            print(f"Selected checkpoint: {selected_checkpoint}")
            args.model_path = str(selected_checkpoint)
            model_path_obj = selected_checkpoint

    resolved_model_path = str(model_path_obj.resolve()) if model_path_obj.exists() else args.model_path

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=resolved_torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left",
        trust_remote_code=args.trust_remote_code,
    )
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
            "checkpoint_select_strategy": args.checkpoint_select,
            "selected_checkpoint_step": _checkpoint_step(model_path_obj) if model_path_obj.exists() else None,
            "checkpoint_selection_scores": checkpoint_selection_scores,
            "run_dir": str(run_dir.resolve()) if run_dir else None,
            "commit_hash": get_git_commit_hash(),
            "eval_dataset": EVAL_DATASET,
            "num_samples": len(records),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
        },
        "metrics": metrics,
        "qualitative_samples": qualitative,
        "records": records,
    }

    if args.output_json:
        output_path = Path(args.output_json)
    elif run_dir:
        output_path = run_dir / "eval" / f"math500_ns{len(records)}_tok{args.max_new_tokens}.json"
    else:
        output_path = Path(args.model_path) / "eval_math500.json"
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
