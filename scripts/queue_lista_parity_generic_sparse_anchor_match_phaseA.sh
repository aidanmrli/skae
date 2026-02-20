#!/bin/bash

set -euo pipefail

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_parity_generic_sparse}"
PHASEA_ROOT="${PHASEA_ROOT:-${BASE_OUT}/phaseA_depth}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${BASE_OUT}/phaseA_anchor_generic_sparse}"
OUT_DIR="${OUT_DIR:-results/lista_parity_generic_sparse_phaseA_anchor_matched}"

ANCHOR_ROOT_LABEL="${ANCHOR_ROOT_LABEL:-generic_sparse}"
INCLUDE_ANCHOR="${INCLUDE_ANCHOR:-1}"

DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"

ANCHOR_DEP_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  ANCHOR_DEP_ARGS=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

ANCHOR_SWEEP_JOB_ID=$(sbatch \
  "${ANCHOR_DEP_ARGS[@]}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",ANCHOR_ROOT="${ANCHOR_ROOT}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}" \
  scripts/sweep_generic_sparse_parity_phaseA.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${ANCHOR_SWEEP_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASEA_ROOT="${PHASEA_ROOT}",OUT_DIR="${OUT_DIR}",ANCHOR_GENERIC_ROOT="${ANCHOR_ROOT}",INCLUDE_ANCHOR="${INCLUDE_ANCHOR}" \
  scripts/collect_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,OUT_DIR="${OUT_DIR}",ANCHOR_ROOTS="${ANCHOR_ROOT_LABEL}" \
  scripts/compare_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

SELECT_JOB_ID=$(sbatch \
  --dependency=afterany:${COMPARE_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASEA_ROOT="${PHASEA_ROOT}",OUT_DIR="${OUT_DIR}",ANCHOR_ROOT="${ANCHOR_ROOT_LABEL}",ANCHOR_GENERIC_ROOT="${ANCHOR_ROOT}" \
  scripts/select_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Upstream dependency: ${UPSTREAM_DEPENDENCY}"
fi
echo "Submitted parity-matched generic_sparse anchor sweep: ${ANCHOR_SWEEP_JOB_ID}"
echo "Submitted recollect job: ${COLLECT_JOB_ID} (afterany:${ANCHOR_SWEEP_JOB_ID})"
echo "Submitted recompare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
echo "Submitted reselect job: ${SELECT_JOB_ID} (afterany:${COMPARE_JOB_ID})"
