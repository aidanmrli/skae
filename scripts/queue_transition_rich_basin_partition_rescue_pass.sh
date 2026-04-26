#!/bin/bash
#
# Queue exactly one dt-rescue pass for the transition-rich basin-partition
# benchmark after a resolve pass has completed and emitted rescue requests.
#
# Submit with:
#   CURRENT_PASS=0 RESULTS_DIR=results/<tag> BASE_OUT=/network/scratch/... \
#     sbatch scripts/queue_transition_rich_basin_partition_rescue_pass.sh
#
# Required env vars:
#   CURRENT_PASS=<0,1,...>
#   RESULTS_DIR=results/<experiment_tag>
#   BASE_OUT=/network/scratch/l/lia/skae/<experiment_tag>
#
# Optional env vars:
#   MAX_HALVINGS=6
#   THRESHOLD=50
#   MIN_SEEDS=1
#   NUM_STEPS_OVERRIDE=200000
#   SEEDS_CSV=0,1,2
#   MODEL_VARIANTS_CSV=lista_blockdiag...,mlp_sparse...
#   EVAL_PROFILE=full
#   MANIFEST_JSON=results/<tag>/task_tables/transition_rich_basin_partition_manifest.json
#
# This launcher only submits the actual next rescue pass, its collector, and
# the next resolve job. It is intended for the Mila submit-cap workflow where
# pre-expanding the entire rescue chain would exceed the job-limit.
#
#SBATCH --job-name=queue_tr_bp_rescue
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH -o /network/scratch/l/lia/skae/queue-transition-rich-rescue-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-transition-rich-rescue-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_basin_partition_rescue_pass.sh"
  exit 2
fi

source .venv/bin/activate

CURRENT_PASS="${CURRENT_PASS:?CURRENT_PASS is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"

MAX_HALVINGS="${MAX_HALVINGS:-6}"
THRESHOLD="${THRESHOLD:-50}"
MIN_SEEDS="${MIN_SEEDS:-1}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

TASK_DIR="${RESULTS_DIR}/task_tables"
ROOT_SPEC_DIR="${RESULTS_DIR}/root_specs"
RESOLVE_DIR="${RESULTS_DIR}/dt_resolution"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/transition_rich_basin_partition_manifest.json}"

NEXT_PASS=$((CURRENT_PASS + 1))
REQUEST_TSV="${RESOLVE_DIR}/pass${CURRENT_PASS}/dt_rescue_request_pass${NEXT_PASS}.tsv"
NEXT_TASK_TSV="${TASK_DIR}/transition_rich_rescue_pass${NEXT_PASS}.tsv"
NEXT_MANIFEST_JSON="${TASK_DIR}/transition_rich_rescue_pass${NEXT_PASS}_manifest.json"
COLLECT_OUT_DIR="${RESULTS_DIR}/collect_pass${NEXT_PASS}"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass${NEXT_PASS}_roots.txt"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${RESOLVE_DIR}"

echo "============================================="
echo "Queue Transition-Rich Rescue Pass"
echo "Job ID: ${SLURM_JOB_ID}"
echo "CURRENT_PASS: ${CURRENT_PASS}"
echo "NEXT_PASS: ${NEXT_PASS}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "BASE_OUT: ${BASE_OUT}"
echo "REQUEST_TSV: ${REQUEST_TSV}"
echo "NEXT_TASK_TSV: ${NEXT_TASK_TSV}"
echo "MAX_HALVINGS: ${MAX_HALVINGS}"
echo "THRESHOLD: ${THRESHOLD}"
echo "MIN_SEEDS: ${MIN_SEEDS}"
echo "NUM_STEPS_OVERRIDE: ${NUM_STEPS_OVERRIDE:-<default>}"
echo "MODEL_VARIANTS_CSV: ${MODEL_VARIANTS_CSV:-<all>}"
echo "============================================="

if (( NEXT_PASS > MAX_HALVINGS )); then
  echo "NEXT_PASS=${NEXT_PASS} exceeds MAX_HALVINGS=${MAX_HALVINGS}; nothing to queue."
  exit 0
fi

if [[ ! -f "${REQUEST_TSV}" ]]; then
  echo "Missing rescue request TSV: ${REQUEST_TSV}"
  exit 1
fi

REQUEST_ROWS=$(( $(wc -l < "${REQUEST_TSV}") - 1 ))
if (( REQUEST_ROWS <= 0 )); then
  echo "No rescue requests found in ${REQUEST_TSV}; no rescue jobs queued."
  exit 0
fi

BUILD_ARGS=(
  --output_tsv "${NEXT_TASK_TSV}"
  --output_manifest_json "${NEXT_MANIFEST_JSON}"
  --phase_label "rescue_pass${NEXT_PASS}"
  --eval_profile "${EVAL_PROFILE}"
  --dt_table "${REQUEST_TSV}"
  --dt_column requested_dt
)
if [[ -n "${NUM_STEPS_OVERRIDE}" ]]; then
  BUILD_ARGS+=(--num_steps_override "${NUM_STEPS_OVERRIDE}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi
if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
  BUILD_ARGS+=(--model_variants_csv "${MODEL_VARIANTS_CSV}")
fi
uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"

TASK_ROWS=$(( $(wc -l < "${NEXT_TASK_TSV}") - 1 ))
if (( TASK_ROWS <= 0 )); then
  echo "Built task TSV has no rescue rows: ${NEXT_TASK_TSV}"
  exit 0
fi

if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
  IFS=',' read -r -a SELECTED_LABELS <<< "${MODEL_VARIANTS_CSV}"
else
  mapfile -t SELECTED_LABELS < <(
    uv run python - "${TASK_DIR}/transition_rich_basin_partition.tsv" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as handle:
    labels = sorted({row["model_variant"] for row in csv.DictReader(handle, delimiter="\t")})
for label in labels:
    print(label)
PY
  )
fi

if (( ${#SELECTED_LABELS[@]} == 0 )); then
  echo "No model variants found for root specs."
  exit 1
fi

: > "${ROOT_SPECS_FILE}"
for model_variant in "${SELECTED_LABELS[@]}"; do
  echo "${model_variant}=${BASE_OUT}/transition_rich_basin_partition/${model_variant}" >> "${ROOT_SPECS_FILE}"
  pass_index=1
  while (( pass_index <= NEXT_PASS )); do
    echo "${model_variant}=${BASE_OUT}/rescue_pass${pass_index}/${model_variant}" >> "${ROOT_SPECS_FILE}"
    pass_index=$((pass_index + 1))
  done
done

RESCUE_JOB_ID=$(
  TASK_TSV="${NEXT_TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --array=0-$((TASK_ROWS - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_OUT_DIR}" \
  GOOD_THRESHOLD="${THRESHOLD}" \
    sbatch --dependency=afterany:"${RESCUE_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

if (( NEXT_PASS < MAX_HALVINGS )); then
  NEXT_RESOLVE_JOB_ID=$(
    ROWS_CSV="${COLLECT_OUT_DIR}/forecasting_rows.csv" \
    OUT_DIR="${RESOLVE_DIR}/pass${NEXT_PASS}" \
    CURRENT_PASS="${NEXT_PASS}" \
    MAX_HALVINGS="${MAX_HALVINGS}" \
    THRESHOLD="${THRESHOLD}" \
    MIN_SEEDS="${MIN_SEEDS}" \
    NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
    MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV}" \
    NEXT_TASK_TSV="${TASK_DIR}/transition_rich_rescue_pass$((NEXT_PASS + 1)).tsv" \
    MANIFEST_JSON="${MANIFEST_JSON}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
      sbatch --dependency=afterany:"${COLLECT_JOB_ID}" scripts/resolve_transition_rich_basin_partition_dt.sh | awk '{print $4}'
  )
else
  NEXT_RESOLVE_JOB_ID="none"
fi

echo "Queued transition-rich rescue pass ${NEXT_PASS}."
echo "Request rows: ${REQUEST_ROWS}"
echo "Task rows: ${TASK_ROWS}"
echo "Rescue array: ${RESCUE_JOB_ID}"
echo "Collect pass${NEXT_PASS}: ${COLLECT_JOB_ID}"
echo "Next resolve: ${NEXT_RESOLVE_JOB_ID}"
