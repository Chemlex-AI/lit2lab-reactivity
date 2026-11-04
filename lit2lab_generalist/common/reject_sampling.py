"""Select correct offline generations and convert them to SFT rows."""

import argparse
from pathlib import Path

import pandas as pd

from common.outcome_reward import ANSWER_PATTERN


SFT_COLUMNS = ["data_source", "prompt", "ability", "reward_model", "extra_info"]


def extract_tagged_answer(text: str) -> str | None:
    matches = ANSWER_PATTERN.findall(str(text))
    return matches[-1].lower() if matches else None


def select_correct_generations(
    source_df: pd.DataFrame,
    generated_column: str = "generated_text",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Reject malformed/incorrect generations and return SFT-format rows."""
    required = set(SFT_COLUMNS) | {generated_column}
    missing = required - set(source_df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")

    selected_rows = []
    stats = {"input": len(source_df), "selected": 0, "incorrect": 0, "malformed": 0}
    for _, row in source_df.iterrows():
        response = row[generated_column]
        answer = extract_tagged_answer(response)
        if answer is None:
            stats["malformed"] += 1
            continue

        ground_truth = str(row["reward_model"]["ground_truth"]).lower()
        if answer != ground_truth:
            stats["incorrect"] += 1
            continue

        extra_info = dict(row["extra_info"])
        extra_info.update({
            "answer": response,
            "index": str(len(selected_rows) + 1),
            "reasoning": True,
            "split": "train",
            "task_type": "yield_prediction",
        })
        selected_rows.append({
            "data_source": row["data_source"],
            "prompt": row["prompt"],
            "ability": row["ability"],
            "reward_model": row["reward_model"],
            "extra_info": extra_info,
        })

    stats["selected"] = len(selected_rows)
    if not selected_rows:
        raise ValueError("No usable data remains after reject sampling")
    return pd.DataFrame(selected_rows, columns=SFT_COLUMNS), stats


def run_reject_sampling_cli(default_input=None, default_output=None, argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Create SFT data by retaining only correct offline generations."
    )
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("output", nargs="?", type=Path, default=default_output)
    parser.add_argument("--generated-column", default="generated_text")
    args = parser.parse_args(argv)
    if args.input is None or args.output is None:
        parser.error("input and output paths are required")

    selected_df, stats = select_correct_generations(
        pd.read_parquet(args.input), args.generated_column
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_parquet(args.output, index=False)
    print(f"Saved to: {args.output}")
    print(f"Stats: {stats}")
