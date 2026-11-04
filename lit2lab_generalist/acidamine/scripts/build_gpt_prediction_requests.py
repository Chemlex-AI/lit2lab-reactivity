#!/usr/bin/env python3
"""Build GPT predictions from training prompts without sending labels or answers."""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ACIDAMINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = ACIDAMINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.gpt_batch import run_batch_cli


def build_prediction_prompt(row) -> list[dict]:
    messages = [dict(message) for message in row["prompt"]]
    if not messages or any(message["role"] not in {"system", "user"} for message in messages):
        raise ValueError("Prediction prompts must contain only system/user messages")
    return messages


if __name__ == "__main__":
    run_batch_cli(
        build_prediction_prompt,
        ACIDAMINE_DIR / "data" / "example_sft.parquet",
        ACIDAMINE_DIR / "outputs" / "gpt_prediction_requests" / "example.jsonl",
        required_columns={"prompt"},
    )
