"""Shared OpenAI Batch request writer for reaction teacher annotation."""

import argparse
import json
from pathlib import Path
from typing import Callable

import pandas as pd


REQUIRED_COLUMNS = {
    "extra_info", "reward_model", "mapped_rxn_smiles", "reactants_description"
}


def build_request(prompt: str | list[dict], index: int, model: str, reasoning_effort: str) -> dict:
    return {
        "custom_id": f"reaction-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "messages": [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt,
        },
    }


def run_batch_cli(
    build_prompt: Callable,
    default_input: Path,
    default_output: Path,
    argv=None,
    required_columns=None,
) -> None:
    parser = argparse.ArgumentParser(description="Build GPT Batch requests.")
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("output", nargs="?", type=Path, default=default_output)
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--requests-per-file", type=int, default=3000)
    args = parser.parse_args(argv)
    if args.requests_per_file < 1:
        raise ValueError("--requests-per-file must be at least 1")

    df = pd.read_parquet(args.input)
    required = REQUIRED_COLUMNS if required_columns is None else set(required_columns)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing columns: {sorted(missing)}")
    requests = [
        build_request(build_prompt(row), index, args.model, args.reasoning_effort)
        for index, (_, row) in enumerate(df.iterrows(), start=1)
    ]

    output_stem = args.output.with_suffix("")
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for part, start in enumerate(range(0, len(requests), args.requests_per_file)):
        output_path = output_stem.with_name(f"{output_stem.name}_{part}.jsonl")
        batch = requests[start:start + args.requests_per_file]
        with output_path.open("w", encoding="utf-8") as file_obj:
            for request in batch:
                file_obj.write(json.dumps(request, ensure_ascii=False) + "\n")
        print(f"Saved {len(batch)} requests to: {output_path}")
