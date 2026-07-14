#!/bin/bash
#
# Prebuild dysts caches over (system x profile x split).
#
# Submit:
#   sbatch scripts/prebuild_dysts_cache_matrix.sh
#
# Required at submission:
#   SYSTEMS_FILE=<generated paper-system list>
#   sbatch --array=0-$((systems * profiles * splits - 1)) ...
# Optional overrides:
#   CACHE_DIR=/network/scratch/l/lia/skae/dysts_native_cache
#   CACHE_NUM_WORKERS=2
#   PROFILES="full"
#   SPLITS="train val test"
#   DYSTS_DT_MULTIPLIER=30
#
#SBATCH --job-name=prebuild_dysts_cache
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/prebuild-dysts-cache-%A_%a.out
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"
source .venv/bin/activate

SYSTEMS_FILE="${SYSTEMS_FILE:?SYSTEMS_FILE must point to a generated paper-system list}"
CACHE_DIR="${CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-2}"
PROFILES_STR="${PROFILES:-full}"
SPLITS_STR="${SPLITS:-train val test}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"

if [[ ! -f "${SYSTEMS_FILE}" ]]; then
  echo "Missing SYSTEMS_FILE=${SYSTEMS_FILE}"
  exit 1
fi

mapfile -t SYSTEMS < <(sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "${SYSTEMS_FILE}" | awk 'NF')
read -r -a PROFILES <<< "${PROFILES_STR}"
read -r -a SPLITS <<< "${SPLITS_STR}"

NUM_SYSTEMS=${#SYSTEMS[@]}
NUM_PROFILES=${#PROFILES[@]}
NUM_SPLITS=${#SPLITS[@]}
TOTAL=$((NUM_SYSTEMS * NUM_PROFILES * NUM_SPLITS))

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this worker with an explicit --array range}"
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Task ${TASK_ID} out of range for TOTAL=${TOTAL}. Exiting."
  exit 0
fi

PROFILE_SPLIT_STRIDE=$((NUM_PROFILES * NUM_SPLITS))
SYSTEM_IDX=$((TASK_ID / PROFILE_SPLIT_STRIDE))
REM=$((TASK_ID % PROFILE_SPLIT_STRIDE))
PROFILE_IDX=$((REM / NUM_SPLITS))
SPLIT_IDX=$((REM % NUM_SPLITS))

SYSTEM=${SYSTEMS[$SYSTEM_IDX]}
PROFILE=${PROFILES[$PROFILE_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}

echo "============================================="
echo "Prebuild Dysts Cache Matrix"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Task ID: ${TASK_ID}/${TOTAL}"
echo "System: ${SYSTEM}"
echo "Profile: ${PROFILE}"
echo "Split: ${SPLIT}"
echo "CACHE_DIR: ${CACHE_DIR}"
echo "CACHE_NUM_WORKERS: ${CACHE_NUM_WORKERS}"
echo "DYSTS_DT_MULTIPLIER: ${DYSTS_DT_MULTIPLIER}"
echo "Start Time: $(date)"
echo "============================================="

uv run python tools/prebuild_dysts_cache.py \
  --systems "dysts:${SYSTEM}" \
  --profiles "${PROFILE}" \
  --splits "${SPLIT}" \
  --cache_dir "${CACHE_DIR}" \
  --cache_num_workers "${CACHE_NUM_WORKERS}" \
  --dt_multiplier "${DYSTS_DT_MULTIPLIER}" \
  --standardize

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
