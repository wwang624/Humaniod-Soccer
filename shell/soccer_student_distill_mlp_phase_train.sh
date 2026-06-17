#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TEACHER_RUN="${1:?Usage: soccer_student_distill_mlp_phase_train.sh <teacher_run> [teacher_checkpoint] [run_name] [max_iterations]}"
TEACHER_CHECKPOINT="${2:-model_.*.pt}"
RUN_NAME="${3:-g1_student_distill_mlp_phase}"
MAX_ITERATIONS="${4:-5000}"
FREEZE_STUDENT_OBS_NORMALIZER="${FREEZE_STUDENT_OBS_NORMALIZER:-0}"
DISTILL_ROLLOUT_MODE="${DISTILL_ROLLOUT_MODE:-teacher}"
TEACHER_MIX_ALPHA="${TEACHER_MIX_ALPHA:-1.0}"

if [[ "${TEACHER_RUN}" == *.pt ]]; then
    TEACHER_CHECKPOINT="$(basename "${TEACHER_RUN}")"
    TEACHER_RUN="$(dirname "${TEACHER_RUN}")"
fi

cd "${REPO_ROOT}"

echo "[INFO] FREEZE_STUDENT_OBS_NORMALIZER=${FREEZE_STUDENT_OBS_NORMALIZER}"
echo "[INFO] DISTILL_ROLLOUT_MODE=${DISTILL_ROLLOUT_MODE}"
echo "[INFO] TEACHER_MIX_ALPHA=${TEACHER_MIX_ALPHA}"

export SOCCER_FREEZE_STUDENT_OBS_NORMALIZER="${FREEZE_STUDENT_OBS_NORMALIZER}"
export SOCCER_DISTILL_ROLLOUT_MODE="${DISTILL_ROLLOUT_MODE}"
export SOCCER_DISTILL_TEACHER_MIX_ALPHA="${TEACHER_MIX_ALPHA}"
unset SOCCER_TEACHER_USES_STUDENT_OBS

python scripts/rsl_rl/train_student.py --task Tracking-Flat-G1-Soccer-Distillation-MLPPhase-v0 \
    --motion_path motions/soccer-standard \
    --load_run "${TEACHER_RUN}" \
    --checkpoint "${TEACHER_CHECKPOINT}" \
    --run_name "${RUN_NAME}" \
    --num_envs 8192 \
    --max_iterations "${MAX_ITERATIONS}" \
    --headless
