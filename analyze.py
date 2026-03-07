"""
uv run python analyze.py --random_eval ./output-random/eval_math500.json --stratified_eval ./output-stratified/eval_math500.json
"""

import argparse
import json
from pathlib import Path

from reporting import get_git_commit_hash, write_json

DEFAULT_WANDB_PROJECT = "grpo-difficulty-variance"


def _load_eval_artifact(path):
    return json.loads(Path(path).read_text())


def _compare_metrics(random_metrics, stratified_metrics):
    comparison = {
        "overall": {
            "random": random_metrics["overall"]["accuracy"],
            "stratified": stratified_metrics["overall"]["accuracy"],
            "delta_stratified_minus_random": (
                stratified_metrics["overall"]["accuracy"] - random_metrics["overall"]["accuracy"]
            ),
        },
        "per_difficulty": {},
    }

    levels = sorted(
        set(random_metrics.get("per_difficulty", {}).keys())
        | set(stratified_metrics.get("per_difficulty", {}).keys()),
        key=int,
    )
    for level in levels:
        random_level = random_metrics.get("per_difficulty", {}).get(level, {})
        stratified_level = stratified_metrics.get("per_difficulty", {}).get(level, {})
        random_accuracy = float(random_level.get("accuracy", 0.0))
        stratified_accuracy = float(stratified_level.get("accuracy", 0.0))
        comparison["per_difficulty"][level] = {
            "random": random_accuracy,
            "stratified": stratified_accuracy,
            "delta_stratified_minus_random": stratified_accuracy - random_accuracy,
            "random_count": int(random_level.get("count", 0)),
            "stratified_count": int(stratified_level.get("count", 0)),
        }
    return comparison


def _save_plot(comparison, plot_path):
    import matplotlib.pyplot as plt

    levels = sorted(comparison["per_difficulty"].keys(), key=int)
    random_values = [comparison["per_difficulty"][level]["random"] for level in levels]
    stratified_values = [comparison["per_difficulty"][level]["stratified"] for level in levels]

    x = list(range(len(levels)))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar([i - width / 2 for i in x], random_values, width=width, label="random")
    plt.bar([i + width / 2 for i in x], stratified_values, width=width, label="stratified")
    plt.xticks(list(x), [f"L{level}" for level in levels])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("MATH-500 Accuracy by Difficulty")
    plt.legend()
    plt.tight_layout()

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare random vs stratified eval artifacts")
    parser.add_argument("--random_eval", type=str, required=True)
    parser.add_argument("--stratified_eval", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="./analysis/comparison.json")
    parser.add_argument("--plot_path", type=str, default="./analysis/accuracy_comparison.png")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    random_artifact = _load_eval_artifact(args.random_eval)
    stratified_artifact = _load_eval_artifact(args.stratified_eval)

    comparison = _compare_metrics(
        random_metrics=random_artifact["metrics"],
        stratified_metrics=stratified_artifact["metrics"],
    )
    payload = {
        "metadata": {
            "commit_hash": get_git_commit_hash(),
            "random_eval_path": str(Path(args.random_eval).resolve()),
            "stratified_eval_path": str(Path(args.stratified_eval).resolve()),
        },
        "comparison": comparison,
    }

    output_path = write_json(args.output_json, payload)
    print(f"Saved comparison artifact to {output_path}")

    if not args.no_plot:
        plot_path = Path(args.plot_path)
        _save_plot(comparison, plot_path)
        print(f"Saved accuracy plot to {plot_path}")

    if not args.disable_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, name="eval-comparison", job_type="analysis")
        wandb_payload = {
            "analysis/overall_delta_stratified_minus_random": comparison["overall"][
                "delta_stratified_minus_random"
            ],
            "analysis/overall_random_accuracy": comparison["overall"]["random"],
            "analysis/overall_stratified_accuracy": comparison["overall"]["stratified"],
        }
        for level, metrics in comparison["per_difficulty"].items():
            wandb_payload[f"analysis/delta_difficulty_{level}"] = metrics["delta_stratified_minus_random"]
            wandb_payload[f"analysis/random_accuracy_difficulty_{level}"] = metrics["random"]
            wandb_payload[f"analysis/stratified_accuracy_difficulty_{level}"] = metrics["stratified"]
        wandb.log(wandb_payload)
        run.summary["comparison_artifact_path"] = str(output_path.resolve())
        wandb.finish()


if __name__ == "__main__":
    main()
