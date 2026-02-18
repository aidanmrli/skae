#!/bin/bash

set -euo pipefail

SWEEP_JOB_ID=$(sbatch scripts/sweep_dysts_forecast_tail_recovery_sp20_a020.sh | awk '{print $4}')
COLLECT_JOB_ID=$(sbatch --dependency=afterany:${SWEEP_JOB_ID} scripts/collect_dysts_forecast_tail_recovery.sh | awk '{print $4}')
COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,PHASE1_ROOT=/network/scratch/l/lia/skae/dysts_forecast_tail_recovery_sp20_a020_ts256,OUT_DIR=results/dysts_forecasting_tail_recovery,SEED_OUT_DIR=results/dysts_forecasting_tail_recovery_seed_all,CANDIDATE_ROOT=lista_relu_sp20_a020_ts256_tail \
  scripts/compare_dysts_forecast_phase1.sh | awk '{print $4}')

echo "Submitted sweep job: ${SWEEP_JOB_ID}"
echo "Submitted collector job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
