#!/usr/bin/env python3
"""Build Acidamine GRPO data by retaining correct GPT Batch predictions."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ACIDAMINE_DIR = SCRIPT_DIR.parent
REPO_ROOT = ACIDAMINE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.reject_sampling import select_correct_generations


def load_gpt_predictions(source_df: pd.DataFrame, result_paths: list[Path]) -> pd.DataFrame:
    """Match Batch results to the same source rows used to build the requests."""
    responses = {}
    seen = set()
    for path in result_paths:
        with path.open(encoding="utf-8") as file_obj:
            for line in file_obj:
                if not line.strip():
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                if custom_id in seen:
                    raise ValueError(f"Duplicate GPT result: {custom_id}")
                seen.add(custom_id)
                prefix, separator, index = custom_id.partition("-")
                if prefix != "reaction" or not separator or not index.isdigit():
                    raise ValueError(f"Invalid GPT result ID: {custom_id}")
                row_pos = int(index) - 1
                if not 0 <= row_pos < len(source_df):
                    raise ValueError(f"GPT result ID outside source data: {custom_id}")
                response = result.get("response") or {}
                if result.get("error") or response.get("status_code") != 200:
                    continue
                choices = response.get("body", {}).get("choices", [])
                if choices and choices[0].get("finish_reason") == "stop":
                    content = choices[0].get("message", {}).get("content")
                    if isinstance(content, str):
                        responses[row_pos] = content

    row_positions = sorted(responses)
    predictions = source_df.iloc[row_positions].copy().reset_index(drop=True)
    predictions["generated_text"] = [responses[row_pos] for row_pos in row_positions]
    print(f"GPT results: {len(predictions)} usable, {len(source_df) - len(predictions)} missing/failed")
    return predictions


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=ACIDAMINE_DIR / "data" / "example_sft.parquet")
    parser.add_argument("output", nargs="?", type=Path,
                        default=ACIDAMINE_DIR / "data" / "example_grpo.parquet")
    parser.add_argument("--gpt-results", nargs="+", type=Path, required=True,
                        help="Downloaded GPT Batch prediction result JSONL files")
    parser.add_argument("--predictions-output", type=Path,
                        default=ACIDAMINE_DIR / "data" / "example_inference.parquet")
    args = parser.parse_args(argv)

    predictions = load_gpt_predictions(pd.read_parquet(args.input), args.gpt_results)
    selected, stats = select_correct_generations(predictions)
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(args.predictions_output, index=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)
    print(f"Saved GPT predictions to: {args.predictions_output}")
    print(f"Saved GRPO data to: {args.output}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
