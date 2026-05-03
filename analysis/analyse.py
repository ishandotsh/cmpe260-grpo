from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


LEVELS = [f"level_{index}" for index in range(1, 6)]
LEVEL_NUMBERS = {level: index for index, level in enumerate(LEVELS, start=1)}
LEVEL_LABELS = [f"Level {index}" for index in range(1, 6)]
STRATEGY_ORDER = ["baseline", "curriculum", "random", "stratified"]
REWARD_ORDER = ["baseline", "binary", "easy_penalty", "difficulty_weighted"]

PRIMARY_METRICS = [
    "overall_accuracy",
    "macro_level_accuracy",
    "baseline_hardness_weighted_accuracy",
    "hardest_level_accuracy",
    "average_solved_difficulty",
]


def title_case(text: str) -> str:
    return text.replace("_", " ").title()


def parse_run_identity(run_name: str) -> dict[str, str]:
    if run_name.startswith("Qwen"):
        return {
            "strategy": "baseline",
            "reward": "baseline",
            "timestamp": "",
            "label": "Baseline",
        }

    if not run_name.startswith("output-"):
        raise ValueError(f"Unrecognized run name format: {run_name}")

    payload = run_name.removeprefix("output-")
    strategy, reward_and_stamp = payload.split("-", 1)
    reward_parts = reward_and_stamp.split("_")
    timestamp_parts: list[str] = []

    while reward_parts and reward_parts[-1].isdigit():
        timestamp_parts.insert(0, reward_parts.pop())

    reward = "_".join(reward_parts)
    if not reward:
        raise ValueError(f"Could not parse reward from run name: {run_name}")

    return {
        "strategy": strategy,
        "reward": reward,
        "timestamp": "_".join(timestamp_parts),
        "label": f"{title_case(strategy)} + {title_case(reward)}",
    }


def level_label(level: str) -> str:
    return f"Level {LEVEL_NUMBERS[level]}"


def _apply_category_order(
    runs: pd.DataFrame,
    levels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = runs.copy()
    levels = levels.copy()

    runs["strategy"] = pd.Categorical(runs["strategy"], STRATEGY_ORDER, ordered=True)
    runs["reward"] = pd.Categorical(runs["reward"], REWARD_ORDER, ordered=True)

    levels["strategy"] = pd.Categorical(levels["strategy"], STRATEGY_ORDER, ordered=True)
    levels["reward"] = pd.Categorical(levels["reward"], REWARD_ORDER, ordered=True)
    levels["level"] = pd.Categorical(levels["level"], LEVELS, ordered=True)
    levels["level_label"] = pd.Categorical(levels["level_label"], LEVEL_LABELS, ordered=True)

    return runs, levels


def _baseline_accuracy_by_level(levels: pd.DataFrame) -> pd.Series:
    return (
        levels.loc[levels["strategy"] == "baseline", ["level", "accuracy"]]
        .drop_duplicates()
        .set_index("level")["accuracy"]
        .reindex(LEVELS)
    )


def _sorted_run_frame(frame: pd.DataFrame, *, descending: bool = False) -> pd.DataFrame:
    return frame.sort_values("level_number", ascending=not descending)


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total_weight = float(np.sum(weights))
    if total_weight == 0.0:
        return 0.0
    return float(np.average(values, weights=weights))


def _average_solved_difficulty(
    level_numbers: np.ndarray,
    correct_counts: np.ndarray,
) -> float:
    solved = int(correct_counts.sum())
    if solved == 0:
        return 0.0
    return float(np.dot(level_numbers, correct_counts) / solved)


def load_results(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the aggregate evaluation files into run-level and per-level tables."""
    run_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []

    for result_file in sorted(root.glob("*/eval_results.json")):
        run_name = result_file.parent.name
        identity = parse_run_identity(run_name)
        data = json.loads(result_file.read_text())

        run_rows.append(
            {
                "run_name": run_name,
                "run_label": identity["label"],
                "strategy": identity["strategy"],
                "reward": identity["reward"],
                "timestamp": identity["timestamp"],
                "checkpoint": data["checkpoint"],
                "overall_accuracy": float(data["overall_accuracy"]),
                "total_correct": int(data["total_correct"]),
                "total_problems": int(data["total_problems"]),
            }
        )

        for level in LEVELS:
            record = data["per_difficulty"][level]
            level_rows.append(
                {
                    "run_name": run_name,
                    "run_label": identity["label"],
                    "strategy": identity["strategy"],
                    "reward": identity["reward"],
                    "timestamp": identity["timestamp"],
                    "level": level,
                    "level_number": LEVEL_NUMBERS[level],
                    "level_label": level_label(level),
                    "correct": int(record["correct"]),
                    "total": int(record["total"]),
                    "accuracy": float(record["accuracy"]),
                }
            )

    runs = pd.DataFrame(run_rows)
    levels = pd.DataFrame(level_rows)
    runs, levels = _apply_category_order(runs, levels)

    return runs.sort_values(["strategy", "reward", "run_label"]), levels.sort_values(
        ["strategy", "reward", "level_number"]
    )


def baseline_profile(levels: pd.DataFrame) -> pd.DataFrame:
    """Return the baseline per-level profile with a derived hardness column."""
    profile = (
        levels.loc[
            levels["strategy"] == "baseline",
            ["level", "level_number", "level_label", "total", "accuracy"],
        ]
        .drop_duplicates()
        .sort_values("level_number")
        .copy()
    )
    profile["baseline_hardness"] = 1.0 - profile["accuracy"]
    return profile.reset_index(drop=True)


def add_baseline_delta(levels: pd.DataFrame, *, value_col: str = "accuracy") -> pd.DataFrame:
    """Add a delta-to-baseline column for each non-baseline row."""
    baseline_values = (
        levels.loc[levels["strategy"] == "baseline", ["level", value_col]]
        .drop_duplicates()
        .set_index("level")[value_col]
        .reindex(LEVELS)
        .astype(float)
    )
    comparison = levels.loc[levels["strategy"] != "baseline"].copy()
    baseline_lookup = baseline_values.to_dict()
    comparison["delta_vs_baseline"] = (
        comparison[value_col].astype(float)
        - comparison["level"].astype(str).map(baseline_lookup).astype(float)
    )
    return comparison


def compute_run_metrics(runs: pd.DataFrame, levels: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-level results into the derived run-level metrics."""
    baseline_levels = _baseline_accuracy_by_level(levels)
    hardness_weights = 1.0 - baseline_levels
    baseline_overall = float(
        runs.loc[runs["strategy"] == "baseline", "overall_accuracy"].iloc[0]
    )

    metric_rows: list[dict[str, object]] = []
    for run_name, frame in levels.groupby("run_name", sort=False):
        ordered_frame = _sorted_run_frame(frame)
        accuracies = ordered_frame["accuracy"].to_numpy()
        correct = ordered_frame["correct"].to_numpy()
        level_numbers = ordered_frame["level_number"].to_numpy()
        run_row = runs.loc[runs["run_name"] == run_name].iloc[0]

        metric_rows.append(
            {
                "run_name": run_name,
                "run_label": run_row["run_label"],
                "strategy": run_row["strategy"],
                "reward": run_row["reward"],
                "overall_accuracy": float(run_row["overall_accuracy"]),
                "delta_vs_baseline": float(run_row["overall_accuracy"] - baseline_overall),
                "macro_level_accuracy": float(np.mean(accuracies)),
                "baseline_hardness_weighted_accuracy": float(
                    _safe_weighted_average(accuracies, hardness_weights.to_numpy())
                ),
                "hardest_level_accuracy": float(accuracies[-1]),
                "average_solved_difficulty": _average_solved_difficulty(
                    level_numbers, correct
                ),
                "difficulty_drop": float(accuracies[0] - accuracies[-1]),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["strategy"] = pd.Categorical(
        metrics["strategy"], STRATEGY_ORDER, ordered=True
    )
    metrics["reward"] = pd.Categorical(metrics["reward"], REWARD_ORDER, ordered=True)

    return metrics.sort_values(
        ["overall_accuracy", "baseline_hardness_weighted_accuracy"],
        ascending=False,
    ).reset_index(drop=True)


def summarize_dimension(metrics: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Average the derived metrics across a strategy or reward dimension."""
    filtered = metrics.loc[metrics[dimension] != "baseline"].copy()
    summary = (
        filtered.groupby(dimension, observed=True)[PRIMARY_METRICS + ["delta_vs_baseline"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary.sort_values("overall_accuracy", ascending=False).reset_index(drop=True)


def summarize_levels_by_dimension(levels: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Average per-level accuracy and baseline lift across a strategy or reward."""
    comparison = add_baseline_delta(levels)
    summary = (
        comparison.groupby(
            [dimension, "level", "level_number", "level_label"],
            observed=True,
        )[["accuracy", "delta_vs_baseline"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    return summary.sort_values([dimension, "level_number"]).reset_index(drop=True)


def build_level_winners(levels: pd.DataFrame) -> pd.DataFrame:
    baseline_levels = _baseline_accuracy_by_level(levels)
    non_baseline = levels.loc[levels["strategy"] != "baseline"].copy()

    winner_rows: list[dict[str, object]] = []
    for level in LEVELS:
        frame = non_baseline.loc[non_baseline["level"] == level]
        best_accuracy = float(frame["accuracy"].max())
        winners = frame.loc[np.isclose(frame["accuracy"], best_accuracy), "run_label"].tolist()
        winner_rows.append(
            {
                "Level": level_label(level),
                "Winning Run(s)": ", ".join(winners),
                "Winning Accuracy": best_accuracy,
                "Delta vs Baseline": best_accuracy - float(baseline_levels.loc[level]),
            }
        )

    return pd.DataFrame(winner_rows)


__all__ = [
    "LEVELS",
    "LEVEL_NUMBERS",
    "LEVEL_LABELS",
    "STRATEGY_ORDER",
    "REWARD_ORDER",
    "PRIMARY_METRICS",
    "title_case",
    "parse_run_identity",
    "level_label",
    "load_results",
    "baseline_profile",
    "add_baseline_delta",
    "compute_run_metrics",
    "summarize_dimension",
    "summarize_levels_by_dimension",
    "build_level_winners",
]
