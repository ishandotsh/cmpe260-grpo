from datetime import datetime
from pathlib import Path


def slugify(value):
    cleaned = []
    for ch in value:
        if ch.isalnum():
            cleaned.append(ch.lower())
        else:
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def build_run_name(sampling, model_name, seed, run_tag=None):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = slugify(model_name.split("/")[-1])
    parts = [timestamp, sampling, model_slug, f"s{seed}"]
    if run_tag:
        parts.append(slugify(run_tag))
    return "-".join(parts)


def get_run_paths(experiments_root, run_name):
    run_dir = Path(experiments_root) / run_name
    return {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "train_dir": run_dir / "train",
        "eval_dir": run_dir / "eval",
        "analysis_dir": run_dir / "analysis",
    }


def ensure_run_dirs(paths):
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
