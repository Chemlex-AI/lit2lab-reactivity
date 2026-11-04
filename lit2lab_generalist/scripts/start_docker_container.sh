#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

CONTAINER_NAME=${CONTAINER_NAME:-verl_vllm}
DOCKER_IMAGE=${DOCKER_IMAGE:-verlai/verl:vllm012.latest}
HOST_PROJECT_DIR=${HOST_PROJECT_DIR:-$REPO_ROOT}
HOST_DATA_DIR=${HOST_DATA_DIR:-$REPO_ROOT/.docker_data}
CONTAINER_PROJECT_DIR=${CONTAINER_PROJECT_DIR:-/workspace}
CONTAINER_DATA_DIR=${CONTAINER_DATA_DIR:-/data}
SHM_SIZE=${SHM_SIZE:-12g}

mkdir -p "$HOST_DATA_DIR"
HOST_PROJECT_DIR=$(realpath "$HOST_PROJECT_DIR")
HOST_DATA_DIR=$(realpath "$HOST_DATA_DIR")

docker create \
    --gpus all \
    --net=host \
    --shm-size="$SHM_SIZE" \
    --cap-add=SYS_ADMIN \
    --volume "$HOST_PROJECT_DIR:$CONTAINER_PROJECT_DIR" \
    --volume "$HOST_DATA_DIR:$CONTAINER_DATA_DIR" \
    --workdir "$CONTAINER_PROJECT_DIR" \
    --name "$CONTAINER_NAME" \
    "$DOCKER_IMAGE" \
    sleep infinity

docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
