# Multi-Reaction Distillation

This workflow filters offline model generations for Acidamine, Suzuki, and
Buchwald reactions. It retains responses with valid `<answer>` tags whose final
answers match the ground truth, then merges them into one SFT dataset to train
a generalist model.

Each reaction type has one synthetic example shared by its inference and selected
data files; the merged training file contains three rows. Reactions, conditions,
yields, and responses are illustrative, not experimental data or actual model output.

## Directory layout

```text
multi_reaction_distillation/
├── data/
│   ├── example_acidamine_inference.parquet
│   ├── example_suzuki_inference.parquet
│   ├── example_buchwald_inference.parquet
│   ├── example_acidamine.parquet
│   ├── example_suzuki.parquet
│   ├── example_buchwald.parquet
│   └── example_train.parquet
└── scripts/
    ├── reject_sample.py
    ├── merge_sft_data.py
    └── train_sft.sh
```

## Data flow

```text
SFT data for each reaction type
  -> Offline inference with the corresponding model produces generated_text
  -> Reject sampling removes malformed or incorrect generations
  -> Selected parquet data for each reaction type
  -> Merge into one SFT parquet file
  -> Joint SFT
```

## Example commands

```bash
# Filter each of the three reaction types
python multi_reaction_distillation/scripts/reject_sample.py \
  multi_reaction_distillation/data/example_acidamine_inference.parquet \
  multi_reaction_distillation/data/example_acidamine.parquet
python multi_reaction_distillation/scripts/reject_sample.py \
  multi_reaction_distillation/data/example_suzuki_inference.parquet \
  multi_reaction_distillation/data/example_suzuki.parquet
python multi_reaction_distillation/scripts/reject_sample.py \
  multi_reaction_distillation/data/example_buchwald_inference.parquet \
  multi_reaction_distillation/data/example_buchwald.parquet

# Merge data from the three reaction types
python multi_reaction_distillation/scripts/merge_sft_data.py

# Train the generalist model
MODEL_PATH=/path/to/base_model bash multi_reaction_distillation/scripts/train_sft.sh
```

For full datasets, specify each input using `--input reaction_type=/path/to/file.parquet`:

```bash
python multi_reaction_distillation/scripts/merge_sft_data.py \
  --input acidamine=/path/to/acidamine_selected.parquet \
  --input suzuki=/path/to/suzuki_selected.parquet \
  --input buchwald=/path/to/buchwald_selected.parquet \
  --output /path/to/multi_reaction_train.parquet
```
