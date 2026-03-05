#!/bin/bash

set -euo pipefail

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_parity_generic_sparse}"
PHASEA_ROOT="${PHASEA_ROOT:-${BASE_OUT}/phaseA_depth}"
OUT_DIR="${OUT_DIR:-results/lista_parity_generic_sparse_phaseA}"

ANCHOR_ROOT="${ANCHOR_ROOT:-generic_sparse}"
ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
INCLUDE_ANCHOR="${INCLUDE_ANCHOR:-1}"

DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"

SWEEP_DEP_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  SWEEP_DEP_ARGS=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

SWEEP_JOB_ID=$(sbatch \
  "${SWEEP_DEP_ARGS[@]}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASEA_ROOT="${PHASEA_ROOT}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}" \
  scripts/sweep_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${SWEEP_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASEA_ROOT="${PHASEA_ROOT}",OUT_DIR="${OUT_DIR}",ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT}",INCLUDE_ANCHOR="${INCLUDE_ANCHOR}" \
  scripts/collect_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,OUT_DIR="${OUT_DIR}" \
  scripts/compare_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

SELECT_JOB_ID=$(sbatch \
  --dependency=afterany:${COMPARE_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASEA_ROOT="${PHASEA_ROOT}",OUT_DIR="${OUT_DIR}",ANCHOR_ROOT="${ANCHOR_ROOT}",ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT}" \
  scripts/select_lista_parity_generic_sparse_depth_phaseA.sh | awk '{print $4}')

if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Upstream dependency: ${UPSTREAM_DEPENDENCY}"
fi
echo "Submitted parity phase-A sweep job: ${SWEEP_JOB_ID}"
echo "Submitted parity phase-A collect job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted parity phase-A compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
echo "Submitted parity phase-A select job: ${SELECT_JOB_ID} (afterany:${COMPARE_JOB_ID})"
