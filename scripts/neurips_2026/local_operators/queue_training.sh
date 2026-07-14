#!/bin/bash
#
# Queue the paper's support-routed local affine operator experiment.
# The scientific protocol is fixed in code; environment variables below only
# select execution scope, storage, and artifact retention.
#
# Submit with:
#   sbatch scripts/neurips_2026/local_operators/queue_training.sh

#SBATCH --job-name=queue_fabs_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run under SLURM." >&2
  echo "Submit it with: sbatch scripts/neurips_2026/local_operators/queue_training.sh" >&2
  exit 2
fi
source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-staged_fabs_local_affine_k_lista_table1_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-neurips_2026_controlled_multibasin_v1}"
BASE_OUT="${BASE_OUT:-${SKAE_SCRATCH_ROOT}/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare_vs_global_k}"
WIDE_REEVAL_DIR="${WIDE_REEVAL_DIR:-${RESULTS_DIR}/wide_periodic_reeval}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"

SOURCE_VARIANT="lista_dense_signsplit_p256_hardinit_basin_partition"
TARGET_VARIANT="lista_fabs_local_affine_k_staged_p256_hardinit_basin_partition"
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL:-${SOURCE_VARIANT}}"
BASELINE_ROOT="${BASELINE_ROOT:-${SKAE_SCRATCH_ROOT}/transition_rich_lista_dense_p256_hardinit_table123_20260430/transition_rich_basin_partition/${SOURCE_VARIANT}}"

SEEDS_CSV="${SEEDS_CSV:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-225}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"
ARRAY_JOB_TIME="${ARRAY_JOB_TIME:-03:00:00}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE:-afterok}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-1}"
QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL:-1}"

case "${DOWNSTREAM_DEPENDENCY_TYPE}" in
  afterok|afterany) ;;
  *)
    echo "DOWNSTREAM_DEPENDENCY_TYPE must be afterok or afterany." >&2
    exit 2
    ;;
esac

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${COMPARE_DIR}" \
  "${WIDE_REEVAL_DIR}" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

BASE_TASK_TSV="${TASK_DIR}/.global_lista_paper_recipe.tsv"
BASE_MANIFEST_JSON="${TASK_DIR}/.global_lista_paper_recipe_manifest.json"
TASK_TSV="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1.tsv"
MANIFEST_JSON="${TASK_DIR}/staged_fabs_local_affine_k_lista_table1_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/staged_fabs_local_affine_k_lista_table1_roots.txt"

BUILD_ARGS=(
  --paper_protocol
  --phase_label "${PHASE_LABEL}"
  --output_tsv "${BASE_TASK_TSV}"
  --output_manifest_json "${BASE_MANIFEST_JSON}"
  --model_variants_csv "${SOURCE_VARIANT}"
  --eval_profile full
)
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi
if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
uv run skae-paper tasks controlled "${BUILD_ARGS[@]}"

PREPARE_ARGS=(
  --base-task-tsv "${BASE_TASK_TSV}"
  --base-manifest-json "${BASE_MANIFEST_JSON}"
  --output-tsv "${TASK_TSV}"
  --output-manifest-json "${MANIFEST_JSON}"
  --source-variant "${SOURCE_VARIANT}"
  --target-variant "${TARGET_VARIANT}"
  --phase-label "${PHASE_LABEL}"
  --base-out "${BASE_OUT}"
)
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  PREPARE_ARGS+=(--skip-completed)
fi
uv run skae-paper tasks local-operators "${PREPARE_ARGS[@]}"
rm -f "${BASE_TASK_TSV}" "${BASE_MANIFEST_JSON}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT < 0 )); then
  echo "Malformed task table: ${TASK_TSV}" >&2
  exit 1
fi
if [[ "${SKIP_COMPLETED}" != "1" && "${EXPECTED_TASK_COUNT}" != "0" ]] \
  && (( TASK_COUNT != EXPECTED_TASK_COUNT )); then
  echo "Expected ${EXPECTED_TASK_COUNT} tasks, generated ${TASK_COUNT}." >&2
  exit 1
fi

{
  printf '%s=%s/%s/%s\n' "${TARGET_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${TARGET_VARIANT}"
  printf '%s=%s\n' "${BASELINE_ROOT_LABEL}" "${BASELINE_ROOT}"
} > "${ROOT_SPECS_FILE}"

echo "Generated ${TASK_COUNT} unfinished staged F_abs tasks."
echo "Task table: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"

ARRAY_JOB_ID=""
if (( TASK_COUNT > 0 )); then
  while true; do
    CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
    if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      break
    fi
    echo "Expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; waiting."
    sleep 60
  done
  ARRAY_JOB_ID="$(
    TASK_TSV="${TASK_TSV}" \
    BASE_OUT="${BASE_OUT}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
    SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT}" \
    GPU_TELEMETRY="${GPU_TELEMETRY:-1}" \
    GPU_TELEMETRY_INTERVAL="${GPU_TELEMETRY_INTERVAL:-30}" \
      sbatch --parsable \
        --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" \
        --partition=long \
        --time="${ARRAY_JOB_TIME}" \
        scripts/neurips_2026/local_operators/run_array.sh
  )"
fi

COLLECT_DEPENDENCY=()
if [[ -n "${ARRAY_JOB_ID}" ]]; then
  COLLECT_DEPENDENCY=(--dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}")
fi
COLLECT_JOB_ID="$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --parsable "${COLLECT_DEPENDENCY[@]}" \
      scripts/neurips_2026/controlled/collect_forecasting.sh
)"

COMPARE_JOB_ID="$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${COMPARE_DIR}" \
  CANDIDATE_ROOTS_CSV="${TARGET_VARIANT}" \
  ANCHOR_ROOT="${BASELINE_ROOT_LABEL}" \
  HORIZON="1000" \
    sbatch --parsable --dependency=afterok:"${COLLECT_JOB_ID}" \
      scripts/neurips_2026/controlled/compare_forecasting.sh
)"

WIDE_REEVAL_JOB_ID=""
if [[ "${QUEUE_WIDE_PERIODIC_REEVAL}" == "1" ]]; then
  WIDE_DEPENDENCY=()
  if [[ -n "${ARRAY_JOB_ID}" ]]; then
    WIDE_DEPENDENCY=(--dependency="${DOWNSTREAM_DEPENDENCY_TYPE}:${ARRAY_JOB_ID}")
  fi
  WIDE_REEVAL_JOB_ID="$(
    STAGED_ROOT="${BASE_OUT}/${PHASE_LABEL}/${TARGET_VARIANT}" \
    GLOBAL_ROOT="${BASELINE_ROOT}" \
    OUT_DIR="${WIDE_REEVAL_DIR}" \
    FORCE="1" \
      sbatch --parsable "${WIDE_DEPENDENCY[@]}" \
        scripts/neurips_2026/local_operators/reevaluate.sh
  )"
fi

QUEUE_JSON_PATH="${AUTOMATION_DIR}/staged_fabs_local_affine_k_table1_queue.json" \
EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
TARGET_VARIANT="${TARGET_VARIANT}" \
BASE_OUT="${BASE_OUT}" \
TASK_TSV="${TASK_TSV}" \
MANIFEST_JSON="${MANIFEST_JSON}" \
TASK_COUNT="${TASK_COUNT}" \
ARRAY_JOB_ID="${ARRAY_JOB_ID}" \
COLLECT_JOB_ID="${COLLECT_JOB_ID}" \
COMPARE_JOB_ID="${COMPARE_JOB_ID}" \
WIDE_REEVAL_JOB_ID="${WIDE_REEVAL_JOB_ID}" \
  uv run python - <<'PY'
import json
import os
from pathlib import Path

from experiments.neurips_2026.local_operators.contract import ROUTE_PROTOCOL

keys = (
    "EXPERIMENT_TAG",
    "TARGET_VARIANT",
    "BASE_OUT",
    "TASK_TSV",
    "MANIFEST_JSON",
    "ARRAY_JOB_ID",
    "COLLECT_JOB_ID",
    "COMPARE_JOB_ID",
    "WIDE_REEVAL_JOB_ID",
)
payload = {key.lower(): os.environ[key] for key in keys}
payload["task_count"] = int(os.environ["TASK_COUNT"])
payload["protocol"] = ROUTE_PROTOCOL
Path(os.environ["QUEUE_JSON_PATH"]).write_text(json.dumps(payload, indent=2) + "\n")
PY

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_FILE=%q\n' "${ROOT_SPECS_FILE}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'COMPARE_JOB_ID=%q\n' "${COMPARE_JOB_ID}"
  printf 'WIDE_REEVAL_JOB_ID=%q\n' "${WIDE_REEVAL_JOB_ID}"
  printf 'PROTOCOL_SOURCE=%q\n' \
    'experiments.neurips_2026.local_operators.contract'
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued staged F_abs local affine LISTA experiment."
echo "Array job: ${ARRAY_JOB_ID:-none (all tasks complete)}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Wide periodic re-evaluation job: ${WIDE_REEVAL_JOB_ID:-disabled}"
