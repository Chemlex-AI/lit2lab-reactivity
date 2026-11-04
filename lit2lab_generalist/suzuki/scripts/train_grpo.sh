#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SUZUKI_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd "$SUZUKI_DIR/.." && pwd)
VERL_ROOT="$REPO_ROOT/verl"

if [[ ! -f "$VERL_ROOT/verl/trainer/main_ppo.py" ]]; then
    echo "VERL source tree not found: $VERL_ROOT" >&2
    exit 1
fi
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HYDRA_FULL_ERROR=1
export TORCH_CUDA_ARCH_LIST="8.0"

: "${MODEL_PATH:?Set MODEL_PATH to the SFT model directory}"
PROJECT_NAME=${PROJECT_NAME:-reaction-yield-prediction}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-suzuki-grpo}
OUTPUT_DIR=${OUTPUT_DIR:-"$SUZUKI_DIR/outputs/$EXPERIMENT_NAME"}
TRAIN_FILES=${TRAIN_FILES:-"[\"$SUZUKI_DIR/data/example_grpo.parquet\"]"}
VAL_FILES=${VAL_FILES:-"[\"$SUZUKI_DIR/data/example_grpo.parquet\"]"}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
RUN_MODE=${RUN_MODE:-foreground}
LOG_FILE=${LOG_FILE:-grpo_train_suzuki.log}

cd "$REPO_ROOT"

cmd=(
    python -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    "custom_reward_function.path=$REPO_ROOT/common/outcome_reward.py"
    custom_reward_function.name=compute_score
    "data.train_files=$TRAIN_FILES"
    "data.val_files=$VAL_FILES"
    data.train_batch_size=32
    data.max_prompt_length=4096
    data.max_response_length=4096
    data.filter_overlong_prompts=True
    "data.truncation=error"
    "+ray_kwargs.ray_init.runtime_env.env_vars.TORCH_CUDA_ARCH_LIST=\"$TORCH_CUDA_ARCH_LIST\""
    "actor_rollout_ref.model.path=$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=k3
    actor_rollout_ref.actor.entropy_coeff=0
    "actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.strategy=fsdp2
    "trainer.validation_data_dir=$OUTPUT_DIR"
    trainer.val_before_train=True
    trainer.val_only=False
    trainer.critic_warmup=0
    trainer.max_actor_ckpt_to_keep=1
    "trainer.logger=['console','tensorboard']"
    trainer.project_name=$PROJECT_NAME
    trainer.experiment_name=$EXPERIMENT_NAME
    "trainer.default_local_dir=$OUTPUT_DIR"
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE"
    trainer.nnodes=1
    trainer.save_freq=100
    trainer.test_freq=100
    trainer.total_epochs=100
    "$@"
)

if [[ "$RUN_MODE" == "background" ]]; then
    nohup "${cmd[@]}" >"$LOG_FILE" 2>&1 &
    echo "started background training pid=$! log=$LOG_FILE"
else
    "${cmd[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
