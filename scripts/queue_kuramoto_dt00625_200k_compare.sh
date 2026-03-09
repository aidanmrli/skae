#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-kuramoto_dt00625_200k_compare_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-kuramoto_dt00625_200k}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
DT_TABLE="${DT_TABLE:-${TASK_DIR}/kuramoto_dt00625.tsv}"
TASK_TSV="${TASK_TSV:-${TASK_DIR}/kuramoto_dt00625_200k.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/kuramoto_dt00625_200k_manifest.json}"
ROOT_SPECS_FILE="${ROOT_SPECS_FILE:-${ROOT_SPEC_DIR}/kuramoto_dt00625_200k_roots.txt}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"

SYSTEMS_CSV="${SYSTEMS_CSV:-kuramoto}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-generic_sparse,lista_dense,lista_blockdiag}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
NUM_STEPS="${NUM_STEPS:-200000}"
ENV_DT="${ENV_DT:-0.00625}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

cat > "${DT_TABLE}" <<EOF
system_key	env_name	selected_dt	pass_index
kuramoto	kuramoto	${ENV_DT}	0
EOF

uv run python tools/build_paper_benchmark_tasks.py \
  --phase full \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --systems_csv "${SYSTEMS_CSV}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --dt_table "${DT_TABLE}"

cat > "${ROOT_SPECS_FILE}" <<EOF
generic_sparse=${BASE_OUT}/${PHASE_LABEL}/generic_sparse
lista_dense=${BASE_OUT}/${PHASE_LABEL}/lista_dense
lista_blockdiag=${BASE_OUT}/${PHASE_LABEL}/lista_blockdiag
EOF

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" PAPER_SUMMARY=1 sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}" CANDIDATE_ROOTS_CSV="lista_dense,lista_blockdiag" ANCHOR_ROOT="generic_sparse" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')

echo "Queued Kuramoto dt=0.00625, 200k comparison."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
