#!/bin/bash
#
# SLURM array runner for the paper's support-routed local-operator protocol.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Operational env vars only:
#   ARRAY_OFFSET=0
#   SKIP_COMPLETED=1
#   RESUME_FROM_LATEST=1
#   SAVE_METRICS_HISTORY=0
#   SAVE_LAST_CHECKPOINT=1
#   SAVE_STAGE2_ARTIFACTS=0
#   SAVE_EVAL_ROLLOUT_ARTIFACTS=0
#   SAVE_EVAL_PLOTS=0
#   SAVE_EVAL_PER_IC_VALUES=0
#   SAVE_EVAL_ERROR_CURVES=0

#SBATCH --job-name=staged_fabs_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o slurm-%x-%A_%a.out
#SBATCH -e slurm-%x-%A_%a.err
#SBATCH --requeue

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

source scripts/common/gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
source .venv/bin/activate

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_METRICS_HISTORY="${SAVE_METRICS_HISTORY:-0}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-1}"
SAVE_STAGE2_ARTIFACTS="${SAVE_STAGE2_ARTIFACTS:-0}"
SAVE_EVAL_ROLLOUT_ARTIFACTS="${SAVE_EVAL_ROLLOUT_ARTIFACTS:-0}"
SAVE_EVAL_PLOTS="${SAVE_EVAL_PLOTS:-0}"
SAVE_EVAL_PER_IC_VALUES="${SAVE_EVAL_PER_IC_VALUES:-0}"
SAVE_EVAL_ERROR_CURVES="${SAVE_EVAL_ERROR_CURVES:-0}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"
if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} in ${TASK_TSV}; exiting."
  exit 0
fi

TASK_EXPORTS="$(
  uv run python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
line_no = int(sys.argv[2])
with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no == line_no:
            for key, value in row.items():
                print(f"{key}={shlex.quote(value or '')}")
            break
    else:
        raise SystemExit(3)
PY
)"
eval "${TASK_EXPORTS}"

if [[ "${num_steps}" != "200000" ]]; then
  echo "Paper protocol requires num_steps=200000, got ${num_steps}." >&2
  exit 2
fi

DT_TAG="$(tagify "${env_dt}")"
SYSTEM_SLUG="${system_slug:-${system_key//:/_}}"
SEED_DIR="${BASE_OUT}/${phase}/${model_variant}/${SYSTEM_SLUG}/dt_${DT_TAG}/seed_${seed}"
COMPLETED_RUN=""
if [[ "${SKIP_COMPLETED}" == "1" && -d "${SEED_DIR}" ]]; then
  COMPLETED_RUN="$(
    find "${SEED_DIR}" -mindepth 1 -maxdepth 1 -type d \
      -name '20*' -exec test -f '{}/evaluation_results_best.json' ';' -print \
      | sort | tail -n 1
  )"
fi
if [[ -n "${COMPLETED_RUN}" ]]; then
  echo "Completed staged run already exists: ${COMPLETED_RUN}"
  exit 0
fi

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Start Time: $(date)"
echo "Task: ${task_id} (${system_key}, seed ${seed}, dt ${env_dt})"
echo "Protocol: experiments.neurips_2026.local_operators.contract"
echo "Output: ${SEED_DIR}"

module load cuda/12.6.0
gpu_guard_assert_cuda_visible "staged F_abs local-map task ${task_id}"
gpu_guard_print_context "Staged F_abs Local-Map Array Runner"

TRAIN_ARGS=(
  --task_tsv "${TASK_TSV}"
  --array_index "${TASK_ID}"
  --array_offset "${ARRAY_OFFSET}"
  --base_out "${BASE_OUT}"
  --device cuda
  --eval_profile full
)

if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  TRAIN_ARGS+=(--skip_completed)
fi
if [[ "${RESUME_FROM_LATEST}" == "0" ]]; then
  TRAIN_ARGS+=(--no_resume_from_latest)
fi
if [[ "${SAVE_METRICS_HISTORY}" == "1" ]]; then
  TRAIN_ARGS+=(--save_metrics_history)
fi
if [[ "${SAVE_LAST_CHECKPOINT}" == "1" ]]; then
  TRAIN_ARGS+=(--save_last_checkpoint)
fi
if [[ "${SAVE_STAGE2_ARTIFACTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_stage2_artifacts)
fi
if [[ "${SAVE_EVAL_ROLLOUT_ARTIFACTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_rollout_artifacts)
fi
if [[ "${SAVE_EVAL_PLOTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_plots)
fi
if [[ "${SAVE_EVAL_PER_IC_VALUES}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_per_ic_values)
fi
if [[ "${SAVE_EVAL_ERROR_CURVES}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_error_curves)
fi

gpu_guard_start_sampler \
  "${SEED_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}_${TASK_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "staged F_abs training start task_id=${task_id}"
set +e
uv run skae-paper train local-operators "${TRAIN_ARGS[@]}"
EXIT_CODE=$?
set -e
gpu_guard_phase "staged F_abs training end task_id=${task_id} exit_code=${EXIT_CODE}"
gpu_guard_stop_sampler

echo "End Time: $(date)"
exit "${EXIT_CODE}"
