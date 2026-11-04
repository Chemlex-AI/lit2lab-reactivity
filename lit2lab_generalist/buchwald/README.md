# Buchwald Yield Prediction

Supervised fine-tuning (SFT), GRPO training, validation and offline testing for
Buchwald–Hartwig reactions. Yield above 20% is classified as `high`; otherwise it is `low`.

## Directory Layout

```text
buchwald/
├── README.md
├── data/       # Three real examples each for SFT, GRPO, validation and testing
├── scripts/    # Training, inference, evaluation and reward support
└── outputs/    # Generated checkpoints and predictions (not committed)
```

## Scripts

- `scripts/train_sft.sh`: SFT with LLaMA-Factory.
- `scripts/train_grpo.sh`: GRPO with EasyR1, using separate training and validation inputs.
- `scripts/offline_inference.py`: Offline vLLM inference.
- `scripts/evaluate.py`: ACC, Macro-F1 and prediction coverage checks.
- `scripts/reward_function.py`: EasyR1 adapter for the outcome-only reward shared with Suzuki in `common/outcome_reward.py`.

Rewards depend only on the predicted `high`/`low` label: correct answers receive `+1`;
incorrect or unrecognized answers receive `-1`. The shared parser accepts plain labels
or `<answer>` tags. Reasoning text, output format, conditions, reaction sites and F1
do not contribute additional rewards.

## Example Data

| File in `data/` | Purpose | Fields |
|---|---|---|
| `example_sft.json` | SFT training | `instruction`, `input`, `output` |
| `example_grpo.parquet` | GRPO training | `problem`, `answer` |
| `example_validation.parquet` | Validation during GRPO training | `problem`, `answer` |
| `example_test.json` | Final testing | `instruction`, `input`, `output` |

Each file contains three real examples with the original prompts and answers.
All 12 reactions are distinct. `data/dataset_info.json` registers the JSON datasets.
These examples demonstrate the data formats; provide your own data for training and
performance evaluation. Keep final test data separate from training validation and checkpoint selection.

## Dependencies and Setup

Use Linux with an NVIDIA GPU and a compatible CUDA driver for training and inference.
Python 3.11 is recommended. Use separate environments for SFT/inference and GRPO;
run all commands below from the repository root.

| Stage | Direct runtime dependencies |
|---|---|
| SFT | CUDA-enabled PyTorch `2.8.0`, Transformers `4.56.2`, LLaMA-Factory at the revision below |
| GRPO / validation | CUDA-enabled PyTorch `2.8.0`, Transformers `4.56.2`, vLLM `0.11.0`, FlashAttention `>=2.4.3,<3`, PyArrow `>=15.0.0`, EasyR1 at the revision below |
| Offline inference | SFT dependencies plus vLLM `0.11.0` |
| Evaluation / reward calculation | Python standard library only |

The framework packages install their other Python dependencies, including Accelerate,
Datasets and PEFT; EasyR1 also installs Ray, OmegaConf and its training utilities.
vLLM `0.11.0` requires PyTorch `2.8.0`; keep the installed CUDA builds compatible.

```bash
# SFT environment
python -m pip install torch==2.8.0
python -m pip install 'git+https://github.com/hiyouga/LLaMA-Factory.git@767b344fb455df6fd6a052284b4273863d5d16d8' transformers==4.56.2

# Add in the inference environment
python -m pip install vllm==0.11.0 transformers==4.56.2

# Separate GRPO environment: install PyTorch before building FlashAttention
python -m pip install torch==2.8.0 packaging ninja wheel psutil
python -m pip install 'flash-attn>=2.4.3,<3' --no-build-isolation
python -m pip install 'git+https://github.com/hiyouga/EasyR1.git@87b06e76f651767c03a9c43c88bf3fb752061302' vllm==0.11.0 transformers==4.56.2 'pyarrow>=15.0.0'
```

Building FlashAttention from source requires a CUDA toolkit with `nvcc` matching
the PyTorch build. Existing compatible CUDA environments may reuse their installed
PyTorch and FlashAttention. These installation commands have not been verified by
a fresh GPU environment build.

GRPO uses installed EasyR1; the repository-level `verl/` used by Acidamine and Suzuki
is a separate framework. The outcome-only reward works with unmodified EasyR1.

## Run

```bash
# 1. SFT
MODEL_PATH=/path/to/base_model bash buchwald/scripts/train_sft.sh

# 2. GRPO with the separate validation examples
MODEL_PATH=/path/to/sft_hf_checkpoint bash buchwald/scripts/train_grpo.sh

# 3. Offline test inference
python buchwald/scripts/offline_inference.py --model-path /path/to/grpo_hf_checkpoint \
  --output-path buchwald/outputs/test_predictions.jsonl

# 4. Evaluate test predictions
python buchwald/scripts/evaluate.py --predictions buchwald/outputs/test_predictions.jsonl \
  --output buchwald/outputs/metrics.json
```

Supply loadable Hugging Face model directories. Outputs are saved under `buchwald/outputs/`.
Example settings are kept directly in the training scripts.
SFT accepts `DATASET_DIR`, `DATASET_NAME`, `OUTPUT_DIR`, `NPROC_PER_NODE` and additional
LLaMA-Factory flags (for example `--learning_rate 1e-5`). GRPO accepts `TRAIN_FILE`, `VAL_FILE`,
`OUTPUT_DIR`, `N_GPUS_PER_NODE` and EasyR1 overrides (for example `trainer.total_epochs=2`).
Use absolute paths for custom inputs. Register a separate SFT validation dataset when needed
and pass `--eval_dataset your_validation --eval_strategy epoch`.
