#!/bin/bash

set -euo pipefail

SWEEP_JOB_ID=$(sbatch scripts/sweep_dysts_forecast_phase1_lista_relu_sp15.sh | awk '{print $4}')
COLLECT_JOB_ID=$(sbatch --dependency=afterany:${SWEEP_JOB_ID} scripts/collect_dysts_forecast_phase1.sh | awk '{print $4}')
COMPARE_JOB_ID=$(sbatch --dependency=afterany:${COLLECT_JOB_ID} scripts/compare_dysts_forecast_phase1.sh | awk '{print $4}')

echo "Submitted sweep job: ${SWEEP_JOB_ID}"
echo "Submitted collector job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
echo "Submitted compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
