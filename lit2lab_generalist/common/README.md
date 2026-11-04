# Shared Utilities

- `reaction_description.py`: Generates reactant structure descriptions from mapped reaction SMILES.
- `reaction_metadata.py`: Parses reaction data, runs RXNMapper, generates descriptions, and writes the results to parquet.
- `gpt_batch.py`: Validates annotation inputs and writes GPT Batch requests to JSONL files in batches.
- `outcome_reward.py`: Provides the shared high/low outcome reward for Acidamine, Suzuki and Buchwald.
- `offline_inference.py`: Runs SGLang offline inference with checkpoint recovery and multiple rollouts.
- `reject_sampling.py`: Rejects malformed or incorrect generations and converts selected responses to SFT/GRPO data.
- `prompts.py`: Defines prompt templates for Acidamine, Suzuki, Buchwald, and other reaction types.

Workflow scripts provide reaction-specific prompt selection and default data paths.
