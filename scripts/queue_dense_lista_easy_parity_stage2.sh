#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-dense_lista_easy_parity_stage2_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-stage2}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"
ANCHOR_PATH="${ANCHOR_PATH:-/network/scratch/l/lia/skae/paper_benchmark_20260307_paper_final_ts256_50k_v4/full/generic_sparse}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

TASK_TSV="${TASK_DIR}/dense_lista_easy_parity_stage2.tsv"
MANIFEST_JSON="${TASK_DIR}/dense_lista_easy_parity_stage2_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/dense_lista_easy_parity_stage2_roots.txt"

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --phase_label "${PHASE_LABEL}"
)

if [[ -n "${SYSTEMS_CSV:-}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
if [[ -n "${SEEDS_CSV:-}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi
if [[ -n "${BASE_ARMS_CSV:-}" ]]; then
  BUILD_ARGS+=(--base_arms_csv "${BASE_ARMS_CSV}")
fi
if [[ -n "${SPARSITY_COEFFS_CSV:-}" ]]; then
  BUILD_ARGS+=(--sparsity_coeffs_csv "${SPARSITY_COEFFS_CSV}")
fi
if [[ -n "${RECONST_COEFFS_CSV:-}" ]]; then
  BUILD_ARGS+=(--reconst_coeffs_csv "${RECONST_COEFFS_CSV}")
fi
if [[ -n "${PRED_COEFFS_CSV:-}" ]]; then
  BUILD_ARGS+=(--pred_coeffs_csv "${PRED_COEFFS_CSV}")
fi

uv run python tools/build_dense_lista_easy_parity_stage2_tasks.py "${BUILD_ARGS[@]}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

uv run python -c '
import csv
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
root_specs_file = Path(sys.argv[2])
base_out = sys.argv[3]
phase_label = sys.argv[4]
anchor_path = sys.argv[5]

with task_tsv.open(newline="") as handle:
    labels = sorted({row["model_variant"] for row in csv.DictReader(handle, delimiter="\t")})

root_specs_file.parent.mkdir(parents=True, exist_ok=True)
with root_specs_file.open("w") as handle:
    handle.write(f"generic_sparse={anchor_path}\n")
    for label in labels:
        handle.write(f"{label}={base_out}/{phase_label}/{label}\n")

print(",".join(labels))
' "${TASK_TSV}" "${ROOT_SPECS_FILE}" "${BASE_OUT}" "${PHASE_LABEL}" "${ANCHOR_PATH}" > "${ROOT_SPEC_DIR}/candidate_roots.csv"

CANDIDATE_ROOTS_CSV="$(tr -d '\n' < "${ROOT_SPEC_DIR}/candidate_roots.csv")"

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" PAPER_SUMMARY=1 sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}" CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV}" ANCHOR_ROOT="generic_sparse" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')

echo "Queued dense LISTA easy-system parity stage-2 sweep."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
echo "Candidate roots: ${CANDIDATE_ROOTS_CSV}"
