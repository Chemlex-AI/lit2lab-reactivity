"""Outcome-only reward for binary reaction-yield classification."""

import re


ANSWER_PATTERN = re.compile(r"<answer>\s*(high|low)\s*</answer>", re.IGNORECASE)


def extract_answer(solution: str) -> str | None:
    matches = ANSWER_PATTERN.findall(solution)
    if matches:
        return matches[-1].lower()

    answer = solution.strip().lower()
    return answer if answer in {"high", "low"} else None


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict[str, float]:
    """Return the total score and its outcome-only component."""
    answer = extract_answer(solution_str)
    expected = str(ground_truth).strip().lower()
    outcome_reward = 1.0 if answer == expected else -1.0
    return {
        "score": outcome_reward,
        "outcome_reward": outcome_reward,
    }
