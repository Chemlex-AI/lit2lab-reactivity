#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUCHWALD_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to an SFT HF checkpoint directory}"
TRAIN_FILE=${TRAIN_FILE:-$BUCHWALD_DIR/data/example_grpo.parquet}
VAL_FILE=${VAL_FILE:-$BUCHWALD_DIR/data/example_validation.parquet}

python3 - "$TRAIN_FILE" "$VAL_FILE" <<'CHECK'
import sys
from pathlib import Path
import pyarrow.parquet as pq

train, validation = map(Path, sys.argv[1:])
if train.resolve() == validation.resolve():
    raise ValueError("Training and validation must use separate files")
for path in (train, validation):
    if not {"problem", "answer"} <= set(pq.read_schema(path).names):
        raise ValueError(f"Expected problem/answer columns: {path}")
    if pq.read_metadata(path).num_rows == 0:
        raise ValueError(f"Empty dataset: {path}")
CHECK

TRAIN_FILE=$(realpath "$TRAIN_FILE")
VAL_FILE=$(realpath "$VAL_FILE")

# Use installed EasyR1, outside the repository-level VERL source directory.
cd "$BUCHWALD_DIR"
# Example settings; pass additional EasyR1 overrides at the end.
python3 -m verl.trainer.main \
  "data.train_files=$TRAIN_FILE" \
  "data.val_files=$VAL_FILE" \
  "worker.actor.model.model_path=$MODEL_PATH" \
  "worker.actor.model.tokenizer_path=$MODEL_PATH" \
  "worker.reward.reward_function=$SCRIPT_DIR/reward_function.py:compute_score" \
  "trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-1}" \
  "trainer.save_checkpoint_path=${OUTPUT_DIR:-$BUCHWALD_DIR/outputs/grpo}" \
  "data.prompt_key=problem" \
  "data.answer_key=answer" \
  "data.max_prompt_length=4096" \
  "data.max_response_length=4096" \
  "data.rollout_batch_size=3" \
  "data.val_batch_size=3" \
  "data.filter_overlong_prompts=false" \
  "algorithm.adv_estimator=grpo" \
  "algorithm.use_kl_loss=true" \
  "algorithm.kl_penalty=low_var_kl" \
  "algorithm.kl_coef=0.001" \
  "worker.actor.global_batch_size=3" \
  "worker.actor.micro_batch_size_per_device_for_update=1" \
  "worker.actor.micro_batch_size_per_device_for_experience=1" \
  "worker.actor.model.trust_remote_code=false" \
  "worker.actor.model.enable_gradient_checkpointing=true" \
  "worker.actor.optim.lr=1e-06" \
  "worker.rollout.n=2" \
  "worker.rollout.tensor_parallel_size=1" \
  "worker.rollout.temperature=1.0" \
  "worker.rollout.top_p=1.0" \
  "worker.rollout.val_override_config.temperature=0.7" \
  "worker.rollout.val_override_config.top_p=0.95" \
  "worker.rollout.val_override_config.n=1" \
  "worker.reward.reward_type=batch" \
  "trainer.total_epochs=1" \
  "trainer.project_name=buchwald" \
  "trainer.experiment_name=example" \
  "trainer.logger=[console]" \
  "trainer.nnodes=1" \
  "trainer.val_before_train=true" \
  "trainer.val_freq=1" \
  "trainer.val_generations_to_log=0" \
  "trainer.save_freq=1" \
  "trainer.save_limit=1" \
  "trainer.save_model_only=true" \
  "trainer.find_last_checkpoint=false" \
  "$@"
