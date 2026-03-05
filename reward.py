import re

from math_verify import parse, verify


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
    try:
        parsed_pred = parse(predicted)
        parsed_gt = parse(ground_truth)
        return 1.0 if verify(parsed_pred, parsed_gt) else 0.0
    except Exception:
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
