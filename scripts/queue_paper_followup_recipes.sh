#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-paper_followup_recipes_200k_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-paper_followup_recipes}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"
ANCHOR_PATH="${ANCHOR_PATH:-/network/scratch/l/lia/skae/paper_benchmark_20260307_paper_final_ts256_50k_v4/full/generic_sparse}"
PRIMARY_GENERIC_200K_LABEL="${PRIMARY_GENERIC_200K_LABEL:-generic_sparse_ns200k_best}"
PRIMARY_GENERIC_200K_PATH="${PRIMARY_GENERIC_200K_PATH:-/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best}"
DENSE_PROMOTED_PATH="${DENSE_PROMOTED_PATH:-/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3}"
DENSE_PROMOTED_LABEL="${DENSE_PROMOTED_LABEL:-lista_dense_promoted_stage4}"
DT_TABLE="${DT_TABLE:-results/paper_benchmark_20260307_paper_final_ts256_50k_v4/dt_resolution/pass2/selected_dt.tsv}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-0}"
RECIPE_SPECS_CSV="${RECIPE_SPECS_CSV:-generic_sparse_ns200k_best:generic_sparse:200000:1e-4:1e-5:1e-4:0.03:1.0:0.0025,lista_blockdiag_ns200k_denseopt_sc6em3:lista_blockdiag:200000:5e-5:5e-6:1e-4:0.03:1.0:0.006,lista_blockdiag_ns200k_denseopt_sc3em3:lista_blockdiag:200000:5e-5:5e-6:1e-4:0.03:1.0:0.003}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

TASK_TSV="${TASK_DIR}/paper_followup_recipes.tsv"
MANIFEST_JSON="${TASK_DIR}/paper_followup_recipes_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_followup_recipe_roots.txt"

BUILD_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --phase_label "${PHASE_LABEL}"
  --recipe_specs_csv "${RECIPE_SPECS_CSV}"
)

if [[ -n "${DT_TABLE}" ]]; then
  if [[ -f "${DT_TABLE}" ]]; then
    BUILD_ARGS+=(--dt_table "${DT_TABLE}")
  else
    echo "Warning: DT_TABLE '${DT_TABLE}' not found; falling back to benchmark default dt values."
  fi
fi

if [[ -z "${SYSTEMS_CSV}" ]]; then
  SYSTEMS_CSV="$(
    uv run python - <<'PY'
from skae.benchmarks.paper_benchmark_manifest import paper_benchmark_systems

print(",".join(spec.system_key for spec in paper_benchmark_systems()))
PY
  )"
fi
BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")

if [[ -n "${SEEDS_CSV:-}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi

uv run python tools/build_paper_followup_recipe_tasks.py "${BUILD_ARGS[@]}"

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
primary_generic_label = sys.argv[6]
primary_generic_path = sys.argv[7]
dense_label = sys.argv[8]
dense_path = sys.argv[9]

with task_tsv.open(newline="") as handle:
    labels = sorted({row["model_variant"] for row in csv.DictReader(handle, delimiter="\t")})

root_specs_file.parent.mkdir(parents=True, exist_ok=True)
with root_specs_file.open("w") as handle:
    handle.write(f"generic_sparse={anchor_path}\n")
    if primary_generic_label not in labels:
        handle.write(f"{primary_generic_label}={primary_generic_path}\n")
    if dense_label not in labels:
        handle.write(f"{dense_label}={dense_path}\n")
    for label in labels:
        handle.write(f"{label}={base_out}/{phase_label}/{label}\n")

print(",".join(labels))
' "${TASK_TSV}" "${ROOT_SPECS_FILE}" "${BASE_OUT}" "${PHASE_LABEL}" "${ANCHOR_PATH}" "${PRIMARY_GENERIC_200K_LABEL}" "${PRIMARY_GENERIC_200K_PATH}" "${DENSE_PROMOTED_LABEL}" "${DENSE_PROMOTED_PATH}" > "${ROOT_SPEC_DIR}/candidate_roots.csv"

CANDIDATE_ROOTS_CSV="$(tr -d '\n' < "${ROOT_SPEC_DIR}/candidate_roots.csv")"
LABEL_LINES="$(echo "${CANDIDATE_ROOTS_CSV}" | tr ',' '\n')"
FOLLOWUP_EXCEPT_PRIMARY_GENERIC_CSV="$(
  echo "${LABEL_LINES}" | grep -vx "${PRIMARY_GENERIC_200K_LABEL}" | paste -sd, -
)"
GENERIC_200K_AND_BLOCKDIAG_CANDIDATES_CSV="${CANDIDATE_ROOTS_CSV},${DENSE_PROMOTED_LABEL}"
DENSE_COMPARISON_CANDIDATES_CSV="${CANDIDATE_ROOTS_CSV}"
if [[ -n "${FOLLOWUP_EXCEPT_PRIMARY_GENERIC_CSV}" ]]; then
  GENERIC_200K_COMPARISON_CANDIDATES_CSV="${FOLLOWUP_EXCEPT_PRIMARY_GENERIC_CSV},${DENSE_PROMOTED_LABEL}"
else
  GENERIC_200K_COMPARISON_CANDIDATES_CSV="${DENSE_PROMOTED_LABEL}"
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))"
if (( ARRAY_PARALLEL > 0 )); then
  ARRAY_SPEC="${ARRAY_SPEC}%${ARRAY_PARALLEL}"
fi

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array="${ARRAY_SPEC}" scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" PAPER_SUMMARY=1 sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_CANONICAL_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}/vs_canonical_generic_sparse" CANDIDATE_ROOTS_CSV="${GENERIC_200K_AND_BLOCKDIAG_CANDIDATES_CSV}" ANCHOR_ROOT="generic_sparse" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')
COMPARE_GENERIC_200K_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}/vs_${PRIMARY_GENERIC_200K_LABEL}" CANDIDATE_ROOTS_CSV="${GENERIC_200K_COMPARISON_CANDIDATES_CSV}" ANCHOR_ROOT="${PRIMARY_GENERIC_200K_LABEL}" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')
COMPARE_DENSE_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}/vs_${DENSE_PROMOTED_LABEL}" CANDIDATE_ROOTS_CSV="${DENSE_COMPARISON_CANDIDATES_CSV}" ANCHOR_ROOT="${DENSE_PROMOTED_LABEL}" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')

echo "Queued paper follow-up recipe rerun."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare vs canonical generic_sparse job: ${COMPARE_CANONICAL_JOB_ID}"
echo "Compare vs ${PRIMARY_GENERIC_200K_LABEL} job: ${COMPARE_GENERIC_200K_JOB_ID}"
echo "Compare vs ${DENSE_PROMOTED_LABEL} job: ${COMPARE_DENSE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
echo "DT table: ${DT_TABLE}"
echo "Candidate roots: ${CANDIDATE_ROOTS_CSV}"
