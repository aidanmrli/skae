#!/bin/bash

set -euo pipefail

RECOVERY_ROOT="${RECOVERY_ROOT:-/network/scratch/l/lia/skae/dysts_forecast_phase1_lista_relu_sp15_ts256_seed_recovery}"
OUT_DIR="${OUT_DIR:-results/dysts_forecasting_phase1_seed_recovery}"
SEED_OUT_DIR="${SEED_OUT_DIR:-results/dysts_forecasting_phase1_seed_recovery_seed_all}"
CANDIDATE_ROOT_LABEL="${CANDIDATE_ROOT_LABEL:-lista_relu_sp15_ts256_seed_recovery}"

SWEEP_JOB_ID=$(sbatch \
  --export=ALL,BASE_OUT="${RECOVERY_ROOT}" \
  scripts/sweep_dysts_forecast_phase1_lista_relu_sp15.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${SWEEP_JOB_ID} \
  --export=ALL,PHASE1_ROOT="${RECOVERY_ROOT}",OUT_DIR="${OUT_DIR}",CANDIDATE_ROOT_LABEL="${CANDIDATE_ROOT_LABEL}" \
  scripts/collect_dysts_forecast_phase1.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,PHASE1_ROOT="${RECOVERY_ROOT}",OUT_DIR="${OUT_DIR}",SEED_OUT_DIR="${SEED_OUT_DIR}",CANDIDATE_ROOT="${CANDIDATE_ROOT_LABEL}" \
  scripts/compare_dysts_forecast_phase1.sh | awk '{print $4}')

echo "Submitted sweep job: ${SWEEP_JOB_ID}"
echo "Submitted collector job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
