#!/bin/bash

set -euo pipefail

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_forecast_generic_sparse_ts256_cachebench}"
OUT_DIR="${OUT_DIR:-results/dysts_forecast_generic_sparse_ts256_cachebench}"
ROOT_LABEL="${ROOT_LABEL:-generic_sparse_ts256_cachebench}"

DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
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
  --export=ALL,BASE_OUT="${BASE_OUT}",DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE}",DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}",DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}",DYSTS_CACHE_REUSE="${DYSTS_CACHE_REUSE}" \
  scripts/sweep_dysts_forecast_generic_sparse_ts256_cachebench.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${SWEEP_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",ROOT_LABEL="${ROOT_LABEL}" \
  scripts/collect_dysts_forecast_generic_sparse_ts256_cachebench.sh | awk '{print $4}')

echo "Submitted generic_sparse sweep job: ${SWEEP_JOB_ID}"
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Sweep dependency: ${UPSTREAM_DEPENDENCY}"
fi
echo "Submitted generic_sparse collect job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
