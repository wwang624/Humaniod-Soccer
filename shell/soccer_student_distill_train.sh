#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TEACHER_RUN="${1:?Usage: soccer_student_distill_train.sh <teacher_run> [teacher_checkpoint] [run_name] [max_iterations]}"
TEACHER_CHECKPOINT="${2:-model_.*.pt}"
RUN_NAME="${3:-soccer_student_distill}"
MAX_ITERATIONS="${4:-20000}"
FREEZE_STUDENT_OBS_NORMALIZER="${FREEZE_STUDENT_OBS_NORMALIZER:-0}"
TEACHER_USES_STUDENT_OBS="${TEACHER_USES_STUDENT_OBS:-0}"
DISTILL_ROLLOUT_MODE="${DISTILL_ROLLOUT_MODE:-sample}"

if [[ "${TEACHER_RUN}" == *.pt ]]; then
    TEACHER_CHECKPOINT="$(basename "${TEACHER_RUN}")"
    TEACHER_RUN="$(dirname "${TEACHER_RUN}")"
fi

cd "${REPO_ROOT}"

echo "[INFO] FREEZE_STUDENT_OBS_NORMALIZER=${FREEZE_STUDENT_OBS_NORMALIZER}"
echo "[INFO] TEACHER_USES_STUDENT_OBS=${TEACHER_USES_STUDENT_OBS}"
echo "[INFO] DISTILL_ROLLOUT_MODE=${DISTILL_ROLLOUT_MODE}"

export SOCCER_FREEZE_STUDENT_OBS_NORMALIZER="${FREEZE_STUDENT_OBS_NORMALIZER}"
export SOCCER_TEACHER_USES_STUDENT_OBS="${TEACHER_USES_STUDENT_OBS}"
export SOCCER_DISTILL_ROLLOUT_MODE="${DISTILL_ROLLOUT_MODE}"

python scripts/rsl_rl/train_student.py --task Tracking-Flat-G1-Soccer-Distillation-v0 \
    --motion_path motions/soccer-standard \
    --load_run "${TEACHER_RUN}" \
    --checkpoint "${TEACHER_CHECKPOINT}" \
    --run_name "${RUN_NAME}" \
    --num_envs 8192 \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless
