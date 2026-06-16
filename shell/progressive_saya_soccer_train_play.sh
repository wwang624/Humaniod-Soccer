#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT_DIR="${REPO_ROOT}/logs/rsl_rl/saya_29dof_flat"

MOTION_PATH="${1:-motions/saya-standard}"
RUN_NAME="${2:-saya_progressive}"
STAGE1_ITERATIONS="${3:-10000}"
NUM_ENVS="${4:-8192}"

cd "${REPO_ROOT}"

python scripts/rsl_rl/train_multi.py \
    --task Tracking-Terrain-Saya29DoF-RNN-v0 \
    --motion_path "${MOTION_PATH}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${STAGE1_ITERATIONS}" \
    --headless

LOAD_RUN="$(find "${EXPERIMENT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "*_${RUN_NAME}" | sort | tail -n 1 | xargs -r basename)"

if [[ -z "${LOAD_RUN}" ]]; then
    echo "Failed to resolve load_run from ${EXPERIMENT_DIR}"
    exit 1
fi

echo "Resolved load_run=${LOAD_RUN}"

python scripts/rsl_rl/train_multi.py \
    --task Tracking-Flat-Saya29DoF-SoccerDestination-RNN-v0 \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --run_name "${RUN_NAME}_resume" \
    --num_envs "${NUM_ENVS}" \
    --resume True \
    --headless
