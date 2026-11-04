#!/usr/bin/env bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ACIDAMINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$ACIDAMINE_DIR/.." && pwd)
VERL_ROOT="$REPO_ROOT/verl"

if [[ ! -f "$VERL_ROOT/verl/trainer/fsdp_sft_trainer.py" ]]; then
  echo "VERL source tree not found: $VERL_ROOT" >&2
  exit 1
fi
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

: "${MODEL_PATH:?Set MODEL_PATH to the pretrained model directory}"
PROJECT_NAME=${PROJECT_NAME:-reaction-yield-prediction}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-acidamine-sft}
OUTPUT_DIR=${OUTPUT_DIR:-"$ACIDAMINE_DIR/outputs/$EXPERIMENT_NAME"}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
TRAIN_FILES="[\"$ACIDAMINE_DIR/data/example_sft.parquet\"]"
VAL_FILES="[\"$ACIDAMINE_DIR/data/example_sft.parquet\"]"
AUX_CLS_ENABLE=${AUX_CLS_ENABLE:-false}
AUX_CLS_LOSS_WEIGHT=${AUX_CLS_LOSS_WEIGHT:-0.1}
AUX_CLS_HIDDEN_DIM=${AUX_CLS_HIDDEN_DIM:-1024}

cd "$REPO_ROOT"

nohup torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
  -m verl.trainer.fsdp_sft_trainer \
  "data.train_files=$TRAIN_FILES" \
  "data.val_files=$VAL_FILES" \
  data.prompt_key=extra_info \
  data.response_key=extra_info \
  data.prompt_dict_keys=['question'] \
  data.response_dict_keys=['answer'] \
  data.max_length=11264 \
  data.train_batch_size=256 \
  data.micro_batch_size_per_gpu=2 \
  data.truncation=right \
  optim.lr=2e-5 \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=bfloat16 \
  model.fsdp_config.cpu_offload=False \
  model.fsdp_config.offload_params=False \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger=['console','tensorboard'] \
  trainer.total_epochs=1 \
  trainer.save_freq=1000 \
  trainer.test_freq=-1 \
  trainer.max_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$NPROC_PER_NODE" \
  trainer.default_hdfs_dir=null \
  use_remove_padding=true \
  "$@" > sft_train.log 2>&1 &
