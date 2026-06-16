#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MOTION_PATH="${1:-motions/saya-standard}"
LOAD_RUN="${2:-2026-06-09_14-24-17_saya_progressive}"
CHECKPOINT="${3:-model_9999.pt}"
RUN_NAME="${4:-saya_progressive_goal_anchor_step2}"
NUM_ENVS="${5:-8192}"

cd "${REPO_ROOT}"

python scripts/rsl_rl/train_multi.py \
    --task Tracking-Flat-Saya29DoF-SoccerDestination-RNN-v0 \
    --motion_path "${MOTION_PATH}" \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${CHECKPOINT}" \
    --run_name "${RUN_NAME}" \
    --num_envs "${NUM_ENVS}" \
    --resume True \
    --headless
