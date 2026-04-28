#!/bin/bash
#
# Queue the long-horizon Dysts reevaluation campaign.
#
# Submit:
#   sbatch scripts/queue_dysts_long_horizon_eval.sh
#
# Optional env vars:
#   DATE_TAG=20260414
#   RESULTS_DIR=results/dysts_long_horizon_eval_${DATE_TAG}
#   TASK_DIR=${RESULTS_DIR}/task_tables
#   OUTPUT_TAG=dysts_long_horizon_h5k_to_h60k_seq10
#   ARRAY_PARALLEL=48
#   VALIDATION_INDEX=0
#   EVAL_TIME_LIMIT=03:00:00
#   INPUT_ROOT_SPECS_TSV=/abs/path/to/custom_root_specs.tsv
#   HORIZONS="5000 10000 20000 30000 40000 50000 60000"
#
#SBATCH --job-name=queue_dysts_long_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-dysts-long-eval-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-dysts-long-eval-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
RESULTS_DIR="${RESULTS_DIR:-results/dysts_long_horizon_eval_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_long_horizon_h5k_to_h60k_seq10}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-48}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
INPUT_ROOT_SPECS_TSV="${INPUT_ROOT_SPECS_TSV:-}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-long60}"
DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT:-test}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-100}"
HORIZONS="${HORIZONS:-5000 10000 20000 30000 40000 50000 60000}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-03:00:00}"

mkdir -p "${TASK_DIR}" "${COLLECT_DIR}" "${QUEUE_DIR}"

TASK_TSV="${TASK_DIR}/dysts_long_horizon_tasks.tsv"
TASK_SUMMARY_JSON="${TASK_DIR}/dysts_long_horizon_tasks_summary.json"
ROOT_SPECS_SNAPSHOT_TSV="${TASK_DIR}/dysts_long_horizon_root_specs.tsv"
SYSTEMS_FILE="${TASK_DIR}/dysts_long_horizon_systems.txt"
QUEUE_RECORD_JSON="${QUEUE_DIR}/queue_record.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "OUTPUT_TAG: ${OUTPUT_TAG}"
echo "DYSTS_CACHE_PROFILE: ${DYSTS_CACHE_PROFILE}"
echo "HORIZONS: ${HORIZONS}"
echo "EVAL_TIME_LIMIT: ${EVAL_TIME_LIMIT}"

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_summary_json "${TASK_SUMMARY_JSON}"
  --output_root_specs_tsv "${ROOT_SPECS_SNAPSHOT_TSV}"
  --output_systems_file "${SYSTEMS_FILE}"
  --output_tag "${OUTPUT_TAG}"
)

if [[ -n "${INPUT_ROOT_SPECS_TSV}" ]]; then
  BUILD_ARGS+=(--root_specs_tsv "${INPUT_ROOT_SPECS_TSV}")
fi

uv run python tools/build_dysts_long_horizon_eval_tasks.py "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
SYSTEM_COUNT=$(wc -l < "${SYSTEMS_FILE}")
if (( TASK_COUNT <= 0 )); then
  echo "No reevaluation tasks were built."
  exit 1
fi
if (( VALIDATION_INDEX < 0 || VALIDATION_INDEX >= TASK_COUNT )); then
  echo "VALIDATION_INDEX=${VALIDATION_INDEX} is out of range for TASK_COUNT=${TASK_COUNT}"
  exit 1
fi

CACHE_JOB_ID=$(
  SYSTEMS_FILE="${SYSTEMS_FILE}" \
  CACHE_DIR="${DYSTS_CACHE_DIR}" \
  CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}" \
  PROFILES="${DYSTS_CACHE_PROFILE}" \
  SPLITS="${DYSTS_CACHE_SPLIT}" \
  sbatch --parsable -p long --array=0-$((SYSTEM_COUNT - 1)) scripts/prebuild_dysts_cache_matrix.sh
)

VALIDATE_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  EVAL_DEVICE="${EVAL_DEVICE}" \
  DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE}" \
  DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT}" \
  DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}" \
  DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  HORIZONS="${HORIZONS}" \
  sbatch --parsable -p long --time="${EVAL_TIME_LIMIT}" --dependency=afterany:${CACHE_JOB_ID} --array=${VALIDATION_INDEX}-${VALIDATION_INDEX}%1 scripts/run_dysts_long_horizon_eval_array.sh
)

EVAL_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  EVAL_DEVICE="${EVAL_DEVICE}" \
  DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE}" \
  DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT}" \
  DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}" \
  DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  HORIZONS="${HORIZONS}" \
  sbatch --parsable -p long --time="${EVAL_TIME_LIMIT}" --dependency=afterok:${VALIDATE_JOB_ID} --array=0-$((TASK_COUNT - 1))%${ARRAY_PARALLEL} scripts/run_dysts_long_horizon_eval_array.sh
)

COLLECT_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  OUT_DIR="${COLLECT_DIR}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  sbatch --parsable -p long --dependency=afterany:${EVAL_JOB_ID} scripts/collect_dysts_long_horizon_forecasting.sh
)

cat > "${QUEUE_RECORD_JSON}" <<EOF
{
  "date_tag": "${DATE_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "task_tsv": "${TASK_TSV}",
  "task_summary_json": "${TASK_SUMMARY_JSON}",
  "input_root_specs_tsv": "${INPUT_ROOT_SPECS_TSV}",
  "root_specs_snapshot_tsv": "${ROOT_SPECS_SNAPSHOT_TSV}",
  "systems_file": "${SYSTEMS_FILE}",
  "output_tag": "${OUTPUT_TAG}",
  "dysts_cache_profile": "${DYSTS_CACHE_PROFILE}",
  "horizons": "${HORIZONS}",
  "task_count": ${TASK_COUNT},
  "system_count": ${SYSTEM_COUNT},
  "array_parallel": ${ARRAY_PARALLEL},
  "eval_time_limit": "${EVAL_TIME_LIMIT}",
  "validation_index": ${VALIDATION_INDEX},
  "cache_job_id": "${CACHE_JOB_ID}",
  "validate_job_id": "${VALIDATE_JOB_ID}",
  "eval_job_id": "${EVAL_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}"
}
EOF

echo "Queued long-horizon Dysts reevaluation chain."
echo "Cache array: ${CACHE_JOB_ID}"
echo "Validation task: ${VALIDATE_JOB_ID}"
echo "Full evaluation array: ${EVAL_JOB_ID}"
echo "Collector: ${COLLECT_JOB_ID}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
