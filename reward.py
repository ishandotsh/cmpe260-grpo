import re

from math_verify import parse, verify

def extract_boxed(text: str) -> str | None:
    """Extract content from \\boxed{...}, handling nested braces."""
    matches = re.findall(r"\\boxed\{", text)
    if not matches:
        return None
    # Find the last \boxed{...} occurrence
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
    try:
        parsed_pred = parse(predicted)
        parsed_gt = parse(ground_truth)
        return 1.0 if verify(parsed_pred, parsed_gt) else 0.0
    except Exception:
        # Fallback: exact string match after stripping whitespace
        return 1.0 if predicted.strip() == ground_truth.strip() else 0.0


def make_reward_fn(answer_key: str = "solution"):

    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        solutions = kwargs[answer_key]
        rewards = []
        for completion_msgs, solution in zip(completions, solutions):
            # completion_msgs is a list of message dicts; take the assistant content
            text = completion_msgs[-1]["content"] if completion_msgs else ""
            rewards.append(check_answer(text, solution))
        return rewards

    return reward_fn

def _get_text(completion_msgs: list[dict]) -> str:
    """Extract assistant text from a completion message list."""
    return completion_msgs[-1]["content"] if completion_msgs else ""


def make_difficulty_weighted_fn(answer_key: str = "solution"):
    # Reward scaled by difficulty: harder problems yield more reward.
    # Level 1 -> 0.33, Level 3 -> 1.0, Level 5 -> 1.67.
    
    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        solutions = kwargs[answer_key]
        difficulties = kwargs["difficulty"]
        rewards = []
        for completion_msgs, solution, difficulty in zip(completions, solutions, difficulties):
            text = _get_text(completion_msgs)
            correct = check_answer(text, solution)
            rewards.append(correct * (difficulty / 3.0))
        return rewards

    return reward_fn

def make_easy_penalty_fn(answer_key: str = "solution"):
    # Binary reward for correct, negative penalty for failing easy problems.
    #Wrong on L1 -> -0.4, L2 -> -0.2, L3+ -> 0.0. Correct -> 1.0.

    def reward_fn(completions: list[list[dict]], **kwargs) -> list[float]:
        solutions = kwargs[answer_key]
        difficulties = kwargs["difficulty"]
        rewards = []
        for completion_msgs, solution, difficulty in zip(completions, solutions, difficulties):
            text = _get_text(completion_msgs)
            correct = check_answer(text, solution)
            if correct == 1.0:
                rewards.append(1.0)
            else:
                penalty = max(0, (3 - difficulty)) * 0.2
                rewards.append(-penalty)
        return rewards

    return reward_fn

REWARD_FUNCTIONS: dict[str, callable] = {
    "binary": make_reward_fn,
    "difficulty_weighted": make_difficulty_weighted_fn,
    "easy_penalty": make_easy_penalty_fn,
}
