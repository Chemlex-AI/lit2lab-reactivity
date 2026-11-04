#!/usr/bin/env python3
"""Build GPT Batch requests for Suzuki teacher annotation."""

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SUZUKI_DIR = SCRIPT_DIR.parent
REPO_ROOT = SUZUKI_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.gpt_batch import run_batch_cli
from common.prompts import get_suzuki_prompt
from common.reaction_metadata import extract_reaction_data


REACTION_FIELDS = ("cano_rxn_smiles", "catalysts", "reagents", "solvent")


def build_annotation_prompt(row) -> str:
    source = extract_reaction_data(row["extra_info"]["question"])
    reaction_data = {key: source.get(key, "") for key in REACTION_FIELDS}
    reaction_data["mapped_rxn_smiles"] = row["mapped_rxn_smiles"]
    ground_truth = row["reward_model"]["ground_truth"]
    if ground_truth not in {"high", "low"}:
        raise ValueError(f"Invalid ground truth: {ground_truth!r}")
    return get_suzuki_prompt(
        json.dumps(reaction_data, ensure_ascii=False),
        row["reactants_description"],
        ground_truth,
    )


if __name__ == "__main__":
    run_batch_cli(
        build_annotation_prompt,
        SUZUKI_DIR / "data" / "example_with_reaction_metadata.parquet",
        SUZUKI_DIR / "outputs" / "gpt_requests" / "example.jsonl",
    )
