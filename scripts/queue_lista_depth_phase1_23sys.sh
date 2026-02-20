#!/bin/bash

set -euo pipefail

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE1_ROOT="${PHASE1_ROOT:-${BASE_OUT}/phase1_depth}"
OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase1_23sys}"
RUN_SMOKE_FIRST="${RUN_SMOKE_FIRST:-1}"

ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT:-runs/dysts_multi_basin_lista_nonlinear}"
INCLUDE_ANCHORS="${INCLUDE_ANCHORS:-1}"

DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-4}"
DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE:-1}"
UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"

SMOKE_NUM_STEPS="${SMOKE_NUM_STEPS:-1000}"
SMOKE_DYSTS_CACHE_STEPS="${SMOKE_DYSTS_CACHE_STEPS:-5000}"
SMOKE_DYSTS_CACHE_TRAJECTORIES="${SMOKE_DYSTS_CACHE_TRAJECTORIES:-32}"
SMOKE_DYSTS_CACHE_WARMUP="${SMOKE_DYSTS_CACHE_WARMUP:-200}"
SMOKE_EVAL_PROFILE="${SMOKE_EVAL_PROFILE:-smoke}"

SWEEP_DEP_ARGS=()
SMOKE_JOB_ID=""
UPSTREAM_DEP_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  UPSTREAM_DEP_ARGS=(--dependency="${UPSTREAM_DEPENDENCY}")
fi
if [[ "${RUN_SMOKE_FIRST}" == "1" ]]; then
  SMOKE_JOB_ID=$(sbatch \
    "${UPSTREAM_DEP_ARGS[@]}" \
    --export=ALL,BASE_OUT="${BASE_OUT}",PHASE1_ROOT="${PHASE1_ROOT}",NUM_STEPS="${SMOKE_NUM_STEPS}",DYSTS_CACHE_STEPS="${SMOKE_DYSTS_CACHE_STEPS}",DYSTS_CACHE_TRAJECTORIES="${SMOKE_DYSTS_CACHE_TRAJECTORIES}",DYSTS_CACHE_WARMUP="${SMOKE_DYSTS_CACHE_WARMUP}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}",EVAL_PROFILE="${SMOKE_EVAL_PROFILE}" \
    scripts/sweep_lista_depth_phase0_smoke_23sys.sh | awk '{print $4}')
  SWEEP_DEP_ARGS=(--dependency=afterok:${SMOKE_JOB_ID})
elif [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  SWEEP_DEP_ARGS=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

SWEEP_JOB_ID=$(sbatch \
  "${SWEEP_DEP_ARGS[@]}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASE1_ROOT="${PHASE1_ROOT}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}" \
  scripts/sweep_lista_depth_phase1_23sys.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${SWEEP_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",PHASE1_ROOT="${PHASE1_ROOT}",OUT_DIR="${OUT_DIR}",ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT}",ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT}",INCLUDE_ANCHORS="${INCLUDE_ANCHORS}" \
  scripts/collect_lista_depth_phase1_23sys.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,OUT_DIR="${OUT_DIR}" \
  scripts/compare_lista_depth_phase1_23sys.sh | awk '{print $4}')

if [[ -n "${SMOKE_JOB_ID}" ]]; then
  echo "Submitted phase0 smoke job: ${SMOKE_JOB_ID}"
fi
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Upstream dependency: ${UPSTREAM_DEPENDENCY}"
fi
echo "Submitted phase1 sweep job: ${SWEEP_JOB_ID}"
echo "Submitted phase1 collect job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted phase1 compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
