#!/bin/bash
#
# Packed wrapper for scripts/run_dysts_long_horizon_eval_array.sh.
#
# Each SLURM array element runs PACK_SIZE reevaluation rows sequentially. This
# keeps large Dysts reevaluation campaigns below accounting job-count limits.
#
# Required env vars:
#   TASK_TSV=<path>
#
# Optional:
#   ARRAY_OFFSET=0
#   PACK_SIZE=12
#
#SBATCH --job-name=dysts_eval_pack
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-eval-pack-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/dysts-eval-pack-%A_%a.err
#SBATCH --requeue

set -euo pipefail

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
PACK_SIZE="${PACK_SIZE:-12}"

if (( PACK_SIZE <= 0 )); then
  echo "PACK_SIZE must be positive, got ${PACK_SIZE}"
  exit 1
fi

PACK_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
START_TASK=$((ARRAY_OFFSET + PACK_TASK_ID * PACK_SIZE))

echo "============================================="
echo "Packed Dysts Long-Horizon Eval Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${PACK_TASK_ID}"
echo "Start Task Row: ${START_TASK}"
echo "PACK_SIZE: ${PACK_SIZE}"
echo "TASK_TSV: ${TASK_TSV}"
echo "Start Time: $(date)"
echo "============================================="

for ((PACK_INDEX = 0; PACK_INDEX < PACK_SIZE; PACK_INDEX++)); do
  GLOBAL_TASK_ID=$((START_TASK + PACK_INDEX))
  LINE_NO=$((GLOBAL_TASK_ID + 2))
  TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"
  if [[ -z "${TASK_LINE}" ]]; then
    echo "No task row for global task ${GLOBAL_TASK_ID}; stopping pack."
    break
  fi

  echo "----- packed reevaluation row ${PACK_INDEX}/${PACK_SIZE}: global task ${GLOBAL_TASK_ID} -----"
  SLURM_ARRAY_TASK_ID="${GLOBAL_TASK_ID}" \
  ARRAY_OFFSET=0 \
  TASK_TSV="${TASK_TSV}" \
  bash scripts/run_dysts_long_horizon_eval_array.sh
done

echo "============================================="
echo "Packed reevaluation runner end time: $(date)"
echo "============================================="
