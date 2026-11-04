#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUCHWALD_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the base HF model directory}"
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
export FORCE_TORCHRUN=1

# Example settings; pass additional LLaMA-Factory flags at the end.
llamafactory-cli train \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_dir "${DATASET_DIR:-$BUCHWALD_DIR/data}" \
  --dataset "${DATASET_NAME:-buchwald_sft}" \
  --output_dir "${OUTPUT_DIR:-$BUCHWALD_DIR/outputs/sft}" \
  --stage sft \
  --do_train true \
  --finetuning_type full \
  --template llama3 \
  --cutoff_len 4096 \
  --learning_rate 2e-05 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --bf16 true \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_only_model true \
  --eval_strategy no \
  --report_to none \
  --enable_thinking false \
  --trust_remote_code false \
  "$@"
