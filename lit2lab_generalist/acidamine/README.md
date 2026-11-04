# Acid–amine Training Workflow

## Directory layout

```text
acidamine/
├── data/                       # One synthetic example for each stage
├── scripts/                    # Data preparation, training, and inference
└── outputs/
    ├── gpt_requests/           # GPT Batch requests
    ├── gpt_prediction_requests/ # GPT prediction requests without labels
    ├── gpt_predictions/        # Downloaded GPT Batch prediction results
    └── inference/              # Offline inference results
```

Shared atom mapping, reaction descriptions, prompts, GPT Batch utilities,
and outcome rewards are located in `common/` at the repository root.

## Data flow

```text
data/example.parquet
  -> data/example_with_reaction_metadata.parquet
  -> outputs/gpt_requests/example_0.jsonl
  -> GPT annotation and external conversion
  -> data/example_sft.parquet
  -> SFT (checkpoint used to initialize GRPO)

data/example_sft.parquet (prompt messages only)
  -> outputs/gpt_prediction_requests/example_0.jsonl
  -> GPT prediction without ground-truth hints
  -> outputs/gpt_predictions/example_results.jsonl
  -> data/example_inference.parquet (GPT responses)
  -> Retain valid GPT responses whose predictions match ground truth
  -> data/example_grpo.parquet
  -> GRPO
  -> outputs/inference/example_predictions.parquet
```

- `example.parquet`: Raw input examples.
- `example_with_reaction_metadata.parquet`: Adds `mapped_rxn_smiles` and `reactants_description`.
- `example_sft.parquet`: SFT training examples.
- `example_inference.parquet`: GPT prediction responses, including `generated_text`.
- `example_grpo.parquet`: Examples correctly predicted by GPT, retained for GRPO training and inference without additional balancing or downsampling.

Each checked-in parquet file contains the same single synthetic reaction.
The reaction, conditions, yield, and responses are illustrative, not experimental
data or actual GPT output.

## Run

Run all commands from the repository root.

```bash
# 1. Add atom mapping and reactant descriptions
python acidamine/scripts/add_reaction_metadata.py

# 2. Build GPT Batch annotation requests with ground-truth labels
python acidamine/scripts/build_gpt_annotation_requests.py

# 3. SFT
MODEL_PATH=/path/to/base_model bash acidamine/scripts/train_sft.sh

# 4. Build GPT prediction requests without ground-truth hints
# Only prompt messages are sent; reward_model and extra_info are excluded.
python acidamine/scripts/build_gpt_prediction_requests.py

# Submit outputs/gpt_prediction_requests/example_*.jsonl to GPT Batch externally.
# Download the prediction results to outputs/gpt_predictions/example_results.jsonl.

# 5. Filter GPT predictions: retain valid responses with correct predictions
python acidamine/scripts/filter_gpt_predictions.py \
  --gpt-results acidamine/outputs/gpt_predictions/example_results.jsonl

# 6. GRPO
MODEL_PATH=/path/to/acidamine_sft_checkpoint bash acidamine/scripts/train_grpo.sh

# 7. Run SGLang offline inference with the GRPO model
python acidamine/scripts/offline_inference.py \
  --model-path /path/to/acidamine_grpo_checkpoint
```
