#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-kuramoto_dimension_sweep_dt00625_200k_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-kuramoto_dimension_sweep_dt00625_200k}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
TASK_TSV="${TASK_TSV:-${TASK_DIR}/kuramoto_dimension_sweep.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/kuramoto_dimension_sweep_manifest.json}"
ROOT_SPECS_FILE="${ROOT_SPECS_FILE:-${ROOT_SPEC_DIR}/kuramoto_dimension_roots.txt}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"

DIMENSIONS_CSV="${DIMENSIONS_CSV:-8,16,24,32,64}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-generic_sparse,lista_dense_promoted,lista_blockdiag}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
NUM_STEPS="${NUM_STEPS:-200000}"
ENV_DT="${ENV_DT:-0.00625}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

uv run python tools/build_kuramoto_dimension_sweep_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --dimensions_csv "${DIMENSIONS_CSV}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --env_dt "${ENV_DT}"

: > "${ROOT_SPECS_FILE}"
IFS=',' read -r -a DIMENSIONS <<< "${DIMENSIONS_CSV}"
IFS=',' read -r -a MODEL_VARIANTS <<< "${MODEL_VARIANTS_CSV}"
for dimension in "${DIMENSIONS[@]}"; do
  dimension="$(echo "${dimension}" | xargs)"
  [[ -z "${dimension}" ]] && continue
  for model_variant in "${MODEL_VARIANTS[@]}"; do
    model_variant="$(echo "${model_variant}" | xargs)"
    [[ -z "${model_variant}" ]] && continue
    echo "${model_variant}_n${dimension}=${BASE_OUT}/${PHASE_LABEL}/${model_variant}/n_${dimension}" >> "${ROOT_SPECS_FILE}"
  done
done

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_kuramoto_dimension_sweep.sh | awk '{print $4}')
COMPARE_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}" DIMENSIONS_CSV="${DIMENSIONS_CSV}" sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_kuramoto_dimension_sweep.sh | awk '{print $4}')

echo "Queued Kuramoto dimension sweep."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
