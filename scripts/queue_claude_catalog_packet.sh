#!/bin/bash
#
# Build and queue the first Claude-catalog training packet from a compute node.
# Submit with `sbatch scripts/queue_claude_catalog_packet.sh` or run inside an
# existing `salloc` allocation.
#
# Optional env vars:
#   SYSTEMS_CSV=claude:cal_triangle_3,claude:transition_routes_4
#   MODEL_VARIANTS_CSV=generic_sparse_ns20k_best,lista_dense_promoted_stage4
#   SEEDS_CSV=0,1,2
#   INCLUDE_SECOND_WAVE=1
#
#SBATCH --job-name=queue_claude
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-claude-catalog-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-claude-catalog-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_claude_catalog_packet.sh"
  echo "Or run it inside an existing salloc allocation."
  exit 2
fi

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Queue launcher job: ${SLURM_JOB_ID}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-claude_catalog_packet_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-claude_catalog_packet}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
INCLUDE_SECOND_WAVE="${INCLUDE_SECOND_WAVE:-0}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
HORIZONS_CSV="${HORIZONS_CSV:-100,500,1000}"

SPARSE_LABEL="generic_sparse_ns20k_best"
ZERO_LABEL="generic_sparse_sc0_ns20k_best"
DENSE_LABEL="lista_dense_promoted_stage4"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

TASK_TSV="${TASK_DIR}/claude_catalog_packet.tsv"
MANIFEST_JSON="${TASK_DIR}/claude_catalog_packet_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/claude_catalog_packet_roots.txt"

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --phase_label "${PHASE_LABEL}"
  --eval_profile "${EVAL_PROFILE}"
)

if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
elif [[ "${INCLUDE_SECOND_WAVE}" == "1" ]]; then
  BUILD_ARGS+=(--include_second_wave)
fi

if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
  BUILD_ARGS+=(--model_variants_csv "${MODEL_VARIANTS_CSV}")
fi

if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi

uv run python tools/build_claude_catalog_packet_tasks.py "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

LABELS_CSV="$(
uv run python -c '
import csv
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
root_specs_file = Path(sys.argv[2])
base_out = sys.argv[3]
phase_label = sys.argv[4]

with task_tsv.open(newline="") as handle:
    labels = sorted({row["model_variant"] for row in csv.DictReader(handle, delimiter="\t")})

root_specs_file.parent.mkdir(parents=True, exist_ok=True)
with root_specs_file.open("w") as handle:
    for label in labels:
        handle.write(f"{label}={base_out}/{phase_label}/{label}\n")
print(",".join(labels))
' "${TASK_TSV}" "${ROOT_SPECS_FILE}" "${BASE_OUT}" "${PHASE_LABEL}"
)"

has_label() {
  local needle="$1"
  [[ ",${LABELS_CSV}," == *",${needle},"* ]]
}

submit_compare_job() {
  local anchor="$1"
  local candidates_csv="$2"
  local out_dir="$3"

  if ! has_label "${anchor}"; then
    echo ""
    return 0
  fi

  local filtered_candidates=()
  IFS=',' read -r -a raw_candidates <<< "${candidates_csv}"
  for candidate in "${raw_candidates[@]}"; do
    candidate="$(echo "${candidate}" | xargs)"
    [[ -z "${candidate}" ]] && continue
    if has_label "${candidate}"; then
      filtered_candidates+=("${candidate}")
    fi
  done

  if (( ${#filtered_candidates[@]} == 0 )); then
    echo ""
    return 0
  fi

  local filtered_csv
  filtered_csv="$(IFS=','; echo "${filtered_candidates[*]}")"
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${out_dir}" \
  CANDIDATE_ROOTS_CSV="${filtered_csv}" \
  ANCHOR_ROOT="${anchor}" \
  HORIZON=1000 \
    sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}'
}

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" HORIZONS_CSV="${HORIZONS_CSV}" PAPER_SUMMARY=1 sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_SPARSE_JOB_ID="$(submit_compare_job "${SPARSE_LABEL}" "${ZERO_LABEL},${DENSE_LABEL}" "${COMPARE_DIR}/vs_${SPARSE_LABEL}")"
COMPARE_ZERO_JOB_ID="$(submit_compare_job "${ZERO_LABEL}" "${SPARSE_LABEL},${DENSE_LABEL}" "${COMPARE_DIR}/vs_${ZERO_LABEL}")"
COMPARE_DENSE_JOB_ID="$(submit_compare_job "${DENSE_LABEL}" "${SPARSE_LABEL},${ZERO_LABEL}" "${COMPARE_DIR}/vs_${DENSE_LABEL}")"

echo "Queued Claude catalog packet."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare vs ${SPARSE_LABEL}: ${COMPARE_SPARSE_JOB_ID}"
echo "Compare vs ${ZERO_LABEL}: ${COMPARE_ZERO_JOB_ID}"
echo "Compare vs ${DENSE_LABEL}: ${COMPARE_DENSE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
