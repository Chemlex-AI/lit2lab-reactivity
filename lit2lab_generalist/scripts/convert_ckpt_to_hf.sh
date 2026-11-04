#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 CHECKPOINT_DIR [TARGET_DIR]" >&2
    exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
VERL_ROOT="$REPO_ROOT/verl"

if [[ ! -f "$VERL_ROOT/verl/model_merger/__main__.py" ]]; then
    echo "VERL source tree not found: $VERL_ROOT" >&2
    exit 1
fi
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT_DIR=$(realpath "$1")
TARGET_DIR=$(realpath -m "${2:-$CHECKPOINT_DIR/hf_model}")

cd "$REPO_ROOT"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$CHECKPOINT_DIR" \
    --target_dir "$TARGET_DIR"
