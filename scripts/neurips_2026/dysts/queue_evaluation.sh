#!/bin/bash
#
# Queue the frozen long-horizon Dysts evaluation.
#
# Submit:
#   sbatch scripts/neurips_2026/dysts/queue_evaluation.sh
#
# Optional env vars:
#   DATE_TAG=20260414
#   RESULTS_DIR=results/dysts_long_horizon_eval_${DATE_TAG}
#   TASK_DIR=${RESULTS_DIR}/task_tables
#   OUTPUT_TAG=dysts_dt30_h100_to_h5000_paper
#   ARRAY_PARALLEL=48
#   EVAL_PACK_SIZE=12
#   VALIDATION_INDEX=0
#   EVAL_TIME_LIMIT=03:00:00
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=10000
#   INPUT_ROOT_SPECS_TSV=/abs/path/to/generated_paper_root_specs.tsv (required)
#   SYSTEMS_CSV=dysts:Chua,dysts:Dadras
#   SEEDS_CSV=0,1,2
#   ALLOW_ROOT_SUBSET=1
#   HORIZONS="100 500 1000 1500 2000 3000 4000 5000"
#   DYSTS_PERIODIC_REENCODE_PERIODS="10 25 50 100 150 200"
#   SAVE_SELECTED_ROLLOUTS=0
#   DYSTS_DT_MULTIPLIER=30
#   DYSTS_CACHE_PROFILE=full
#
#SBATCH --job-name=queue_dysts_long_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
RESULTS_DIR="${RESULTS_DIR:-results/dysts_long_horizon_eval_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_dt30_h100_to_h5000_paper}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-48}"
EVAL_PACK_SIZE="${EVAL_PACK_SIZE:-12}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
INPUT_ROOT_SPECS_TSV="${INPUT_ROOT_SPECS_TSV:?INPUT_ROOT_SPECS_TSV is required}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
ALLOW_ROOT_SUBSET="${ALLOW_ROOT_SUBSET:-0}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-${SKAE_SCRATCH_ROOT}/dysts_native_cache}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_SPLIT="${DYSTS_CACHE_SPLIT:-test}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-100}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-10 25 50 100 150 200}"
SAVE_SELECTED_ROLLOUTS="${SAVE_SELECTED_ROLLOUTS:-0}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-}"
TASK_TSV_SHA256="${TASK_TSV_SHA256:-}"
REQUIRE_TRAINING_RECEIPT="${REQUIRE_TRAINING_RECEIPT:-0}"
REQUIRE_COMPLETE_COVERAGE="${REQUIRE_COMPLETE_COVERAGE:-0}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-03:00:00}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-10000}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-300}"

wait_for_submit_capacity() {
  local label="$1"
  while true; do
    local active_jobs
    active_jobs=$(squeue -u "${USER}" -h -r | wc -l)
    if (( active_jobs <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      echo "Submit capacity available for ${label}: active_jobs=${active_jobs}, threshold=${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
      return 0
    fi
    echo "Waiting to submit ${label}: active_jobs=${active_jobs}, threshold=${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping ${SUBMIT_WAIT_SECONDS}s"
    sleep "${SUBMIT_WAIT_SECONDS}"
  done
}

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
echo "DYSTS_PERIODIC_REENCODE_PERIODS: ${DYSTS_PERIODIC_REENCODE_PERIODS}"
echo "SAVE_SELECTED_ROLLOUTS: ${SAVE_SELECTED_ROLLOUTS}"
echo "DYSTS_DT_MULTIPLIER: ${DYSTS_DT_MULTIPLIER}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<default>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<default>}"
echo "ALLOW_ROOT_SUBSET: ${ALLOW_ROOT_SUBSET}"
echo "EVAL_TIME_LIMIT: ${EVAL_TIME_LIMIT}"
echo "EVAL_PACK_SIZE: ${EVAL_PACK_SIZE}"
echo "MAX_EXISTING_JOBS_BEFORE_SUBMIT: ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"

if [[ -n "${SOURCE_MANIFEST}" ]]; then
  sha256sum -c "${SOURCE_MANIFEST}"
fi

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_summary_json "${TASK_SUMMARY_JSON}"
  --output_root_specs_tsv "${ROOT_SPECS_SNAPSHOT_TSV}"
  --output_systems_file "${SYSTEMS_FILE}"
  --output_tag "${OUTPUT_TAG}"
  --root_specs_tsv "${INPUT_ROOT_SPECS_TSV}"
)
if [[ "${ALLOW_ROOT_SUBSET}" == "1" ]]; then
  BUILD_ARGS+=(--allow_root_subset)
fi
if [[ "${REQUIRE_TRAINING_RECEIPT}" == "1" ]]; then
  BUILD_ARGS+=(--require_training_receipt)
fi
if [[ "${REQUIRE_COMPLETE_COVERAGE}" == "1" ]]; then
  BUILD_ARGS+=(--require_complete)
fi

if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi

uv run skae-paper tasks dysts-evaluation "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
TASK_TSV_SHA256="$(sha256sum "${TASK_TSV}" | awk '{print $1}')"
SYSTEM_COUNT=$(wc -l < "${SYSTEMS_FILE}")
if (( TASK_COUNT <= 0 )); then
  echo "No reevaluation tasks were built."
  exit 1
fi
if (( VALIDATION_INDEX < 0 || VALIDATION_INDEX >= TASK_COUNT )); then
  echo "VALIDATION_INDEX=${VALIDATION_INDEX} is out of range for TASK_COUNT=${TASK_COUNT}"
  exit 1
fi
if [[ -n "${EXPECTED_TASK_COUNT}" ]] && (( TASK_COUNT != EXPECTED_TASK_COUNT )); then
  echo "TASK_COUNT=${TASK_COUNT} != EXPECTED_TASK_COUNT=${EXPECTED_TASK_COUNT}"
  exit 1
fi

wait_for_submit_capacity "test cache prebuild"
CACHE_JOB_ID=$(
  SYSTEMS_FILE="${SYSTEMS_FILE}" \
  CACHE_DIR="${DYSTS_CACHE_DIR}" \
  CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}" \
  PROFILES="${DYSTS_CACHE_PROFILE}" \
  SPLITS="${DYSTS_CACHE_SPLIT}" \
  DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER}" \
  sbatch --parsable -p long --array=0-$((SYSTEM_COUNT - 1)) scripts/neurips_2026/dysts/prebuild_cache.sh
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
  DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS}" \
  SAVE_SELECTED_ROLLOUTS="${SAVE_SELECTED_ROLLOUTS}" \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  TASK_TSV_SHA256="${TASK_TSV_SHA256}" \
  REQUIRE_TRAINING_RECEIPT="${REQUIRE_TRAINING_RECEIPT}" \
  sbatch --parsable -p long --time="${EVAL_TIME_LIMIT}" --dependency=afterok:${CACHE_JOB_ID} --array=${VALIDATION_INDEX}-${VALIDATION_INDEX}%1 scripts/neurips_2026/dysts/run_evaluation_array.sh
)

EVAL_ARRAY_TASK_COUNT=$(( (TASK_COUNT + EVAL_PACK_SIZE - 1) / EVAL_PACK_SIZE ))
wait_for_submit_capacity "packed reevaluation array"
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
  DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS}" \
  SAVE_SELECTED_ROLLOUTS="${SAVE_SELECTED_ROLLOUTS}" \
  PACK_SIZE="${EVAL_PACK_SIZE}" \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  TASK_TSV_SHA256="${TASK_TSV_SHA256}" \
  REQUIRE_TRAINING_RECEIPT="${REQUIRE_TRAINING_RECEIPT}" \
  sbatch --parsable -p long --time="${EVAL_TIME_LIMIT}" --dependency=afterok:${VALIDATE_JOB_ID} --array=0-$((EVAL_ARRAY_TASK_COUNT - 1))%${ARRAY_PARALLEL} scripts/neurips_2026/dysts/run_evaluation_packed_array.sh
)

COLLECT_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  OUT_DIR="${COLLECT_DIR}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  HORIZONS="${HORIZONS}" \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  TASK_TSV_SHA256="${TASK_TSV_SHA256}" \
  REQUIRE_COMPLETE="${REQUIRE_COMPLETE_COVERAGE}" \
  EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT}" \
  sbatch --parsable -p long --dependency=afterok:${EVAL_JOB_ID} scripts/neurips_2026/dysts/collect_evaluation.sh
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
  "dysts_periodic_reencode_periods": "${DYSTS_PERIODIC_REENCODE_PERIODS}",
  "save_selected_rollouts": "${SAVE_SELECTED_ROLLOUTS}",
  "dysts_dt_multiplier": "${DYSTS_DT_MULTIPLIER}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "allow_root_subset": "${ALLOW_ROOT_SUBSET}",
  "require_complete_coverage": "${REQUIRE_COMPLETE_COVERAGE}",
  "expected_task_count": "${EXPECTED_TASK_COUNT}",
  "task_count": ${TASK_COUNT},
  "system_count": ${SYSTEM_COUNT},
  "array_parallel": ${ARRAY_PARALLEL},
  "eval_pack_size": ${EVAL_PACK_SIZE},
  "eval_array_task_count": ${EVAL_ARRAY_TASK_COUNT},
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
