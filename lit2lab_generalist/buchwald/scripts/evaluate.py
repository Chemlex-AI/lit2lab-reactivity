#!/usr/bin/env python3
"""Check prediction coverage and compute accuracy / macro-F1."""

import argparse
import json
from pathlib import Path
import re


ANSWER = re.compile(r"<answer>\s*(high|low)\s*</answer>", re.IGNORECASE)


def label(text, required=False):
    matches = ANSWER.findall(text)
    if len(matches) == 1:
        return matches[0].lower()
    if required:
        raise ValueError("Missing or ambiguous reference label")
    return "invalid"


def evaluate(reference, predictions):
    expected = [label(row["output"], required=True) for row in reference]
    if not expected:
        raise ValueError("Reference data is empty")
    ordered = {}
    for record in predictions:
        index = record["row_index"]
        if type(index) is not int or not 0 <= index < len(expected) or index in ordered:
            raise ValueError(f"Invalid or duplicate row_index: {index}")
        if label(record["label"], required=True) != expected[index]:
            raise ValueError(f"Reference label mismatch at row {index}")
        ordered[index] = label(record["predict"])
    if len(ordered) != len(expected):
        raise ValueError("Incomplete predictions; all reference rows must be covered")
    predicted = [ordered[i] for i in range(len(expected))]
    f1 = []
    for cls in ("high", "low"):
        tp = sum(p == cls and y == cls for p, y in zip(predicted, expected))
        fp = sum(p == cls and y != cls for p, y in zip(predicted, expected))
        fn = sum(p != cls and y == cls for p, y in zip(predicted, expected))
        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "rows": len(expected),
        "invalid_predictions": predicted.count("invalid"),
        "acc": sum(p == y for p, y in zip(predicted, expected)) / len(expected),
        "f1_macro": sum(f1) / len(f1),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/example_test.json",
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reference = json.loads(args.reference.read_text())
    predictions = [
        json.loads(line)
        for path in args.predictions
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    result = json.dumps(evaluate(reference, predictions), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
