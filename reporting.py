import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


def get_git_commit_hash(default="unknown"):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if not commit:
            return default
        return commit
    except Exception:
        return default


def write_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2))
    return destination


def summarize_values(values):
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": 0.0, "var": 0.0, "std": 0.0}
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(var)
    return {"count": n, "mean": mean, "var": var, "std": std}


def summarize_grouped_values(grouped_values):
    summary = {}
    for level, values in grouped_values.items():
        summary[str(level)] = summarize_values(values)
    return summary


def build_accuracy_summary(records):
    all_correct = []
    by_level = defaultdict(list)

    for record in records:
        correct = float(record.get("correct", 0.0))
        difficulty = int(record.get("difficulty", 0))
        all_correct.append(correct)
        by_level[difficulty].append(correct)

    overall = summarize_values(all_correct)
    per_level = {}
    for level, values in by_level.items():
        stats = summarize_values(values)
        per_level[str(level)] = {
            "count": stats["count"],
            "accuracy": stats["mean"],
            "var": stats["var"],
            "std": stats["std"],
        }

    return {
        "overall": {
            "count": overall["count"],
            "accuracy": overall["mean"],
            "var": overall["var"],
            "std": overall["std"],
        },
        "per_difficulty": per_level,
    }
