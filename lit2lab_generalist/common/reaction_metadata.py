"""Shared helpers for adding atom mapping and reactant descriptions."""

import argparse
import ast
import json
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def extract_reaction_data(question: str) -> dict:
    """Extract the dictionary following ``Reaction data:`` from a prompt."""
    for line in question.splitlines():
        if "Reaction data:" not in line:
            continue
        payload = line.split("Reaction data:", 1)[1].strip().removesuffix(".")
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return ast.literal_eval(payload)
    raise ValueError("Reaction data not found in prompt")


def map_reactions(reaction_smiles: list[str]) -> list[str]:
    try:
        from rxnmapper import RXNMapper
    except ImportError as exc:
        raise ImportError("rxnmapper is not installed in the current environment") from exc

    results = RXNMapper().get_attention_guided_atom_maps(reaction_smiles)
    if len(results) != len(reaction_smiles):
        raise RuntimeError("The number of RXNMapper results does not match the number of inputs")
    return [result["mapped_rxn"] for result in results]


def describe_reactions(mapped_reactions: list[str], workers: int) -> list[str]:
    try:
        from common.reaction_description import describe_mapped_reaction
    except ImportError as exc:
        raise ImportError("accfg and RDKit are required to generate reaction descriptions") from exc
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if workers == 1:
        return [describe_mapped_reaction(rxn) for rxn in tqdm(mapped_reactions)]
    with mp.Pool(workers) as pool:
        return list(tqdm(
            pool.imap(describe_mapped_reaction, mapped_reactions),
            total=len(mapped_reactions),
        ))


def add_reaction_metadata(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Return a copy with mapped_rxn_smiles and reactants_description columns."""
    if df.empty or "extra_info" not in df:
        raise ValueError("Input data is empty or missing the extra_info column")

    reaction_data = [extract_reaction_data(value["question"]) for value in df["extra_info"]]
    reaction_smiles = [value["cano_rxn_smiles"] for value in reaction_data]
    mapped_reactions = map_reactions(reaction_smiles)
    descriptions = describe_reactions(mapped_reactions, workers)

    output_df = df.copy(deep=True)
    output_df["mapped_rxn_smiles"] = mapped_reactions
    output_df["reactants_description"] = descriptions
    return output_df


def run_metadata_cli(default_input: Path, argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Add atom mapping and reactant descriptions to a parquet file."
    )
    parser.add_argument("input", nargs="?", type=Path, default=default_input)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = (args.output or input_path.with_name(
        f"{input_path.stem}_with_reaction_metadata.parquet"
    )).resolve()
    output_df = add_reaction_metadata(pd.read_parquet(input_path), args.workers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    print(f"Saved {len(output_df)} rows to: {output_path}")
