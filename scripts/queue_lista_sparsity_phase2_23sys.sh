#!/bin/bash

set -euo pipefail

if [[ -z "${DEPTH_STAR:-}" ]]; then
  echo "DEPTH_STAR is required. Example:"
  echo "  DEPTH_STAR=3 sbatch scripts/queue_lista_sparsity_phase2_23sys.sh"
  exit 1
fi

PHASE2_MODE="${PHASE2_MODE:-coarse}"
SPARSITY_STAR="${SPARSITY_STAR:-}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE2_ROOT="${PHASE2_ROOT:-${BASE_OUT}/phase2_sparsity}"
OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase2_23sys}"

ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT:-runs/dysts_multi_basin_lista_nonlinear}"
INCLUDE_ANCHORS="${INCLUDE_ANCHORS:-1}"

DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"

SWEEP_JOB_ID=$(sbatch \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASE2_ROOT="${PHASE2_ROOT}",DEPTH_STAR="${DEPTH_STAR}",PHASE2_MODE="${PHASE2_MODE}",SPARSITY_STAR="${SPARSITY_STAR}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}" \
  scripts/sweep_lista_sparsity_phase2_23sys.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${SWEEP_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASE2_ROOT="${PHASE2_ROOT}",OUT_DIR="${OUT_DIR}",PHASE2_MODE="${PHASE2_MODE}",SPARSITY_STAR="${SPARSITY_STAR}",ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT}",ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT}",INCLUDE_ANCHORS="${INCLUDE_ANCHORS}" \
  scripts/collect_lista_sparsity_phase2_23sys.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,OUT_DIR="${OUT_DIR}",PHASE2_MODE="${PHASE2_MODE}",SPARSITY_STAR="${SPARSITY_STAR}" \
  scripts/compare_lista_sparsity_phase2_23sys.sh | awk '{print $4}')

echo "Submitted phase2 sweep job: ${SWEEP_JOB_ID}"
echo "Submitted phase2 collect job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted phase2 compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
