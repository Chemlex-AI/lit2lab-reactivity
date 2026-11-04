#!/usr/bin/env python3
"""Merge reject-sampled reaction datasets into one SFT dataset."""

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = SCRIPT_DIR.parent
DEFAULT_INPUTS = [
    f"acidamine={WORKFLOW_DIR / 'data' / 'example_acidamine.parquet'}",
    f"suzuki={WORKFLOW_DIR / 'data' / 'example_suzuki.parquet'}",
    f"buchwald={WORKFLOW_DIR / 'data' / 'example_buchwald.parquet'}",
]
DEFAULT_OUTPUT = WORKFLOW_DIR / "data" / "example_train.parquet"


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Input must use the format reaction_type=/path/to/data.parquet")
    reaction_type, path = value.split("=", 1)
    if not reaction_type or not path:
        raise argparse.ArgumentTypeError("reaction_type and path must not be empty")
    return reaction_type, Path(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    inputs = args.input or [parse_input(value) for value in DEFAULT_INPUTS]
    frames = []
    for reaction_type, path in inputs:
        frame = pd.read_parquet(path).copy()
        frame["extra_info"] = frame["extra_info"].map(
            lambda value: {**value, "reaction_type": reaction_type}
        )
        frames.append(frame)
        print(f"{reaction_type}: {len(frame)} rows <- {path}")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["extra_info"] = [
        {**value, "index": str(index + 1)}
        for index, value in enumerate(merged["extra_info"])
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(args.output, index=False)
    print(f"Merged: {len(merged)} rows -> {args.output}")


if __name__ == "__main__":
    main()
