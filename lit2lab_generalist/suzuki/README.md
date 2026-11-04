# Suzuki Training Workflow

## Directory layout

```text
suzuki/
├── data/
│   ├── example.parquet                        # One synthetic example
│   ├── example_with_reaction_metadata.parquet # Atom mapping and descriptions
│   ├── example_sft.parquet                    # Converted GPT annotations for SFT
│   └── example_grpo.parquet                   # Prepared GRPO examples
├── scripts/                                   # Data preparation, SFT, GRPO, and inference
└── outputs/
    └── gpt_requests/                          # GPT Batch requests
```

Each checked-in parquet file contains the same single synthetic Suzuki reaction.
The reaction, conditions, yield, and responses are illustrative, not experimental
data or actual GPT output.
Shared atom mapping, reaction descriptions, prompts, and GPT Batch utilities
and the GRPO outcome reward are located in `common/` at the repository root.

## Data flow

```text
data/example.parquet
  -> data/example_with_reaction_metadata.parquet
  -> outputs/gpt_requests/example_0.jsonl
  -> GPT annotation and external conversion
  -> data/example_sft.parquet
  -> SFT
  -> SFT checkpoint + data/example_grpo.parquet
  -> GRPO
  -> outputs/inference/example_predictions.parquet
```

## Run

Run these commands from the repository root. If `example_sft.parquet` is
already prepared, start directly at step 3.

```bash
# 1. Add mapped_rxn_smiles and reactants_description
python suzuki/scripts/add_reaction_metadata.py

# 2. Build GPT Batch annotation requests with ground-truth labels
python suzuki/scripts/build_gpt_annotation_requests.py

# 3. SFT (original training settings; uses example_sft.parquet by default)
MODEL_PATH=/path/to/base_model bash suzuki/scripts/train_sft.sh

# 4. GRPO (original training settings; uses example_grpo.parquet by default)
MODEL_PATH=/path/to/sft_checkpoint bash suzuki/scripts/train_grpo.sh

# 5. SGLang offline inference
python suzuki/scripts/offline_inference.py \
  --model-path /path/to/grpo_checkpoint
```
