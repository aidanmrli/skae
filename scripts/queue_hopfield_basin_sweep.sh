#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
HOPFIELD_NUM_NEURONS="${HOPFIELD_NUM_NEURONS:-64}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-hopfield_basin_sweep_n${HOPFIELD_NUM_NEURONS}_dt00625_200k_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-hopfield_basin_sweep}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
TASK_TSV="${TASK_TSV:-${TASK_DIR}/hopfield_basin_sweep.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/hopfield_basin_sweep_manifest.json}"
ROOT_SPECS_FILE="${ROOT_SPECS_FILE:-${ROOT_SPEC_DIR}/hopfield_basin_sweep_roots.txt}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"

NUM_BASINS_CSV="${NUM_BASINS_CSV:-8,10,12,14,16}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-generic_sparse,lista_dense_promoted_stage4,lista_blockdiag_targeted}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
NUM_STEPS="${NUM_STEPS:-200000}"
ENV_DT="${ENV_DT:-0.00625}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
ANCHOR_ROOT="${ANCHOR_ROOT:-generic_sparse}"
CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV:-}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

uv run python tools/build_hopfield_basin_sweep_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --num_basins_csv "${NUM_BASINS_CSV}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --env_dt "${ENV_DT}" \
  --num_neurons "${HOPFIELD_NUM_NEURONS}" \
  --eval_profile "${EVAL_PROFILE}"

IFS=',' read -r -a MODEL_VARIANTS <<< "${MODEL_VARIANTS_CSV}"
if [[ -z "${CANDIDATE_ROOTS_CSV}" ]]; then
  CANDIDATES=()
  for variant in "${MODEL_VARIANTS[@]}"; do
    variant="$(echo "${variant}" | xargs)"
    [[ -z "${variant}" || "${variant}" == "${ANCHOR_ROOT}" ]] && continue
    CANDIDATES+=("${variant}")
  done
  CANDIDATE_ROOTS_CSV="$(IFS=','; echo "${CANDIDATES[*]}")"
fi

: > "${ROOT_SPECS_FILE}"
for variant in "${MODEL_VARIANTS[@]}"; do
  variant="$(echo "${variant}" | xargs)"
  [[ -z "${variant}" ]] && continue
  echo "${variant}=${BASE_OUT}/${PHASE_LABEL}/${variant}" >> "${ROOT_SPECS_FILE}"
done

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
COLLECT_JOB_ID=$(ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" OUT_DIR="${COLLECT_DIR}" PAPER_SUMMARY=1 sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_JOB_ID=$(ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" OUT_DIR="${COMPARE_DIR}" CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV}" ANCHOR_ROOT="${ANCHOR_ROOT}" HORIZON=1000 sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')

echo "Queued Hopfield basin sweep."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare job: ${COMPARE_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Results dir: ${RESULTS_DIR}"
