#!/bin/bash
# Submit training -> matched dense specificity evaluation -> summary after a passed smoke.

#SBATCH --job-name=dense0wd_queue
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

SMOKE_DECISION="${SMOKE_DECISION:?SMOKE_DECISION is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_dense_zero_wd_card.json}"
CARD_SHA="$(sha256sum "${CARD_PATH}" | awk '{print $1}')"
TASK_TSV_SHA="$(sha256sum "${TASK_TSV}" | awk '{print $1}')"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:-experiments/neurips_2026/global_k_dense_specificity_source_lock.json}"
SOURCE_LOCK_SHA="$(sha256sum "${SOURCE_LOCK_PATH}" | awk '{print $1}')"

if [[ "$(jq -r '.passed' "${SMOKE_DECISION}")" != "true" ]]; then
  echo "Utilization smoke did not pass; refusing full training."
  exit 3
fi
if [[ "$(jq -r '.card_sha256' "${SMOKE_DECISION}")" != "${CARD_SHA}" ]]; then
  echo "Smoke/card hash mismatch; refusing full training."
  exit 3
fi
if [[ "$(jq -r '.card_sha256' "${TASK_MANIFEST}")" != "${CARD_SHA}" ]]; then
  echo "Full task manifest/card hash mismatch."
  exit 3
fi
if [[ "$(jq -r '.task_tsv_sha256' "${TASK_MANIFEST}")" != "$(sha256sum "${TASK_TSV}" | awk '{print $1}')" ]]; then
  echo "Full task-table hash mismatch."
  exit 3
fi
if [[ "$(jq -r '.sources.frozen_card.sha256 // empty' "${SOURCE_LOCK_PATH}")" != "${CARD_SHA}" ]]; then
  echo "Source lock/card hash mismatch."
  exit 3
fi
if [[ "$(jq -r '.external_inputs.full_task_tsv.sha256 // empty' "${SOURCE_LOCK_PATH}")" != "${TASK_TSV_SHA}" ]]; then
  echo "Source lock/task-table hash mismatch."
  exit 3
fi

mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/telemetry" "${RESULT_ROOT}/specificity/evaluation"

TRAIN_JOB_ID=$(
  sbatch --parsable \
    --array=0-2%3 \
    --time=3-00:00:00 \
    --export=ALL,TASK_TSV="${TASK_TSV}",BASE_OUT="${BASE_OUT}",PACK_SIZE=15,PACK_CONCURRENCY=15,GPU_TELEMETRY_INTERVAL=15,PACK_TELEMETRY_LOG="${RESULT_ROOT}/telemetry/pack_{array_task_id}.csv",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/train_%A_%a.out" \
    --error="${RESULT_ROOT}/slurm/train_%A_%a.err" \
    scripts/neurips_2026/global_k_dense_zero_wd/run_packed_train.sh
)

EVAL_JOB_ID=$(
  sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --array=0-44%15 \
    --export=ALL,TASK_TSV="${TASK_TSV}",BASE_OUT="${BASE_OUT}",OUTPUT_DIR="${RESULT_ROOT}/specificity/evaluation",CARD_PATH="${CARD_PATH}",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}" \
    --output="${RESULT_ROOT}/slurm/eval_%A_%a.out" \
    --error="${RESULT_ROOT}/slurm/eval_%A_%a.err" \
    scripts/neurips_2026/global_k_dense_zero_wd/run_specificity_array.sh
)

SUMMARY_JOB_ID=$(
  sbatch --parsable \
    --dependency="afterok:${EVAL_JOB_ID}" \
    --export=ALL,INPUT_DIR="${RESULT_ROOT}/specificity/evaluation",OUTPUT_DIR="${RESULT_ROOT}/specificity/summary",CARD_PATH="${CARD_PATH}",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}" \
    --output="${RESULT_ROOT}/slurm/summary_%j.out" \
    --error="${RESULT_ROOT}/slurm/summary_%j.err" \
    scripts/neurips_2026/global_k_dense_zero_wd/run_specificity_summary.sh
)

echo "Card SHA256: ${CARD_SHA}"
echo "Task-table SHA256: ${TASK_TSV_SHA}"
echo "Source-lock SHA256: ${SOURCE_LOCK_SHA}"
echo "Training array: ${TRAIN_JOB_ID}"
echo "Specificity evaluation array: ${EVAL_JOB_ID}"
echo "Specificity summary: ${SUMMARY_JOB_ID}"
