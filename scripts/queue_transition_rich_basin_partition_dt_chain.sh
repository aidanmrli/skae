#!/bin/bash
#
# Queue the transition-rich basin-partition default sweep plus dt-rescue passes.
# The rescue protocol keeps each model/system arm at the system default dt until
# evaluation finishes, then halves dt until H1000 best-periodic mean < threshold
# or the configured halving budget is exhausted.
#
# Submit with:
#   sbatch scripts/queue_transition_rich_basin_partition_dt_chain.sh
#
# Optional env vars:
#   SEEDS_CSV=0
#   EVAL_PROFILE=full
#   MAX_HALVINGS=6
#   MIN_SEEDS=1
#   THRESHOLD=50
#   DEFAULT_ARRAY_JOB_IDS_CSV=9190869,9192341
#
#SBATCH --job-name=queue_tr_bp_dt
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH -o /network/scratch/l/lia/skae/queue-transition-rich-dt-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-transition-rich-dt-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_transition_rich_basin_partition_dt_chain.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_basin_partition_${DATE_TAG}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
RESOLVE_DIR="${RESOLVE_DIR:-${RESULTS_DIR}/dt_resolution}"
DEFAULT_TSV="${TASK_DIR}/transition_rich_basin_partition.tsv"
MANIFEST_JSON="${TASK_DIR}/transition_rich_basin_partition_manifest.json"
SEEDS_CSV="${SEEDS_CSV:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
MAX_HALVINGS="${MAX_HALVINGS:-6}"
MIN_SEEDS="${MIN_SEEDS:-1}"
THRESHOLD="${THRESHOLD:-50}"
DEFAULT_ARRAY_JOB_IDS_CSV="${DEFAULT_ARRAY_JOB_IDS_CSV:-}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${RESOLVE_DIR}"

if [[ -n "${SEEDS_CSV}" ]]; then
  IFS=',' read -r -a SEED_VALUES <<< "${SEEDS_CSV}"
  NUM_SEEDS="${#SEED_VALUES[@]}"
else
  NUM_SEEDS=3
fi
MAX_RESCUE_TASKS=$((17 * 2 * NUM_SEEDS))

write_root_specs() {
  local output_file="$1"
  local max_rescue_pass="$2"
  : > "${output_file}"
  for model_variant in lista_dense_basin_partition lista_blockdiag_basin_partition; do
    echo "${model_variant}=${BASE_OUT}/transition_rich_basin_partition/${model_variant}" >> "${output_file}"
    local pass_index
    for ((pass_index=1; pass_index<=max_rescue_pass; pass_index++)); do
      echo "${model_variant}=${BASE_OUT}/rescue_pass${pass_index}/${model_variant}" >> "${output_file}"
    done
  done
}

BUILD_ARGS=(
  --output_tsv "${DEFAULT_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --phase_label transition_rich_basin_partition
  --eval_profile "${EVAL_PROFILE}"
)
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi
uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"

write_root_specs "${ROOT_SPEC_DIR}/transition_rich_collect_pass0_roots.txt" 0

if [[ -n "${DEFAULT_ARRAY_JOB_IDS_CSV}" ]]; then
  DEFAULT_DEPENDENCY="afterany:${DEFAULT_ARRAY_JOB_IDS_CSV//,/:}"
  DEFAULT_JOB_LABEL="existing:${DEFAULT_ARRAY_JOB_IDS_CSV}"
else
  DEFAULT_COUNT=$(( $(wc -l < "${DEFAULT_TSV}") - 1 ))
  DEFAULT_JOB_ID=$(
    TASK_TSV="${DEFAULT_TSV}" BASE_OUT="${BASE_OUT}" \
      sbatch --array=0-$((DEFAULT_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}'
  )
  DEFAULT_DEPENDENCY="afterany:${DEFAULT_JOB_ID}"
  DEFAULT_JOB_LABEL="${DEFAULT_JOB_ID}"
fi

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass0_roots.txt" \
  OUT_DIR="${RESULTS_DIR}/collect_pass0" \
  GOOD_THRESHOLD="${THRESHOLD}" \
    sbatch --dependency="${DEFAULT_DEPENDENCY}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

PREV_COLLECT_ID="${COLLECT_JOB_ID}"
for ((pass_index=0; pass_index<MAX_HALVINGS; pass_index++)); do
  next_pass=$((pass_index + 1))
  NEXT_TASK_TSV="${TASK_DIR}/transition_rich_rescue_pass${next_pass}.tsv"
  RESOLVE_JOB_ID=$(
    ROWS_CSV="${RESULTS_DIR}/collect_pass${pass_index}/forecasting_rows.csv" \
    OUT_DIR="${RESOLVE_DIR}/pass${pass_index}" \
    CURRENT_PASS="${pass_index}" \
    MAX_HALVINGS="${MAX_HALVINGS}" \
    THRESHOLD="${THRESHOLD}" \
    MIN_SEEDS="${MIN_SEEDS}" \
    NEXT_TASK_TSV="${NEXT_TASK_TSV}" \
    MANIFEST_JSON="${MANIFEST_JSON}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
      sbatch --dependency=afterany:"${PREV_COLLECT_ID}" scripts/resolve_transition_rich_basin_partition_dt.sh | awk '{print $4}'
  )

  RESCUE_JOB_ID=$(
    TASK_TSV="${NEXT_TASK_TSV}" BASE_OUT="${BASE_OUT}" \
      sbatch --dependency=afterany:"${RESOLVE_JOB_ID}" --array=0-$((MAX_RESCUE_TASKS - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}'
  )

  write_root_specs "${ROOT_SPEC_DIR}/transition_rich_collect_pass${next_pass}_roots.txt" "${next_pass}"
  PREV_COLLECT_ID=$(
    ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass${next_pass}_roots.txt" \
    OUT_DIR="${RESULTS_DIR}/collect_pass${next_pass}" \
    GOOD_THRESHOLD="${THRESHOLD}" \
      sbatch --dependency=afterany:"${RESCUE_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
  )
done

FINAL_RESOLVE_ID=$(
  ROWS_CSV="${RESULTS_DIR}/collect_pass${MAX_HALVINGS}/forecasting_rows.csv" \
  OUT_DIR="${RESOLVE_DIR}/pass${MAX_HALVINGS}" \
  CURRENT_PASS="${MAX_HALVINGS}" \
  MAX_HALVINGS="${MAX_HALVINGS}" \
  THRESHOLD="${THRESHOLD}" \
  MIN_SEEDS="${MIN_SEEDS}" \
  NEXT_TASK_TSV="${TASK_DIR}/transition_rich_unused.tsv" \
  MANIFEST_JSON="${MANIFEST_JSON}" \
  SEEDS_CSV="${SEEDS_CSV}" \
  EVAL_PROFILE="${EVAL_PROFILE}" \
    sbatch --dependency=afterany:"${PREV_COLLECT_ID}" scripts/resolve_transition_rich_basin_partition_dt.sh | awk '{print $4}'
)

echo "Queued transition-rich basin-partition dt-rescue chain."
echo "Default source: ${DEFAULT_JOB_LABEL}"
echo "Collect pass0: ${COLLECT_JOB_ID}"
echo "Final collect dependency chain head: ${PREV_COLLECT_ID}"
echo "Final resolve: ${FINAL_RESOLVE_ID}"
