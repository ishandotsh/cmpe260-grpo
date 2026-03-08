import re
from collections import defaultdict

from math_verify import parse, verify


class DifficultyRewardTracker:
    def __init__(self):
        self._rewards_by_level: dict[int, list[float]] = defaultdict(list)

    def update(self, difficulties: list[int] | tuple[int, ...], rewards: list[float]):
        for difficulty, reward in zip(difficulties, rewards):
            try:
                level = int(difficulty)
            except (TypeError, ValueError):
                continue
            self._rewards_by_level[level].append(float(reward))

    def pop(self) -> dict[int, list[float]]:
        snapshot = {level: values[:] for level, values in self._rewards_by_level.items()}
        self._rewards_by_level.clear()
        return snapshot


def extract_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{", text)
    if not matches:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    idx += len("\\boxed{")
    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[idx : end - 1]


def check_answer(completion: str, ground_truth: str) -> float:
    """Return 1.0 if the completion's answer matches ground_truth, else 0.0."""
    predicted = extract_boxed(completion)
    if predicted is None:
        return 0.0

    target = extract_boxed(ground_truth)
    if target is None:
        target = ground_truth

    try:
        parsed_pred = parse(predicted)
        parsed_gt = parse(target)
        return 1.0 if verify(parsed_pred, parsed_gt) else 0.0
    except Exception:
        return 1.0 if predicted.strip() == target.strip() else 0.0


def make_reward_fn(
    answer_key: str = "solution",
    difficulty_key: str = "difficulty",
    reward_tracker: DifficultyRewardTracker | None = None,
):
    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        solutions = kwargs[answer_key]
        rewards = []
        for completion_msgs, solution in zip(completions, solutions):
            # completion_msgs is a list of message dicts; take the assistant content
            text = completion_msgs[-1]["content"] if completion_msgs else ""
            rewards.append(check_answer(text, solution))
        if reward_tracker:
            difficulties = kwargs.get(difficulty_key, [])
            if isinstance(difficulties, tuple):
                difficulties = list(difficulties)
            elif not isinstance(difficulties, list):
                difficulties = [difficulties] * len(rewards)
            reward_tracker.update(difficulties, rewards)
        return rewards

    return reward_fn
