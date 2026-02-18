#!/bin/bash
#
#SBATCH --job-name=collect_dysts_fcast_p1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-dysts-fcast-p1-%A.out

source .venv/bin/activate

PHASE1_ROOT="${PHASE1_ROOT:-/network/scratch/l/lia/skae/dysts_forecast_phase1_lista_relu_sp15_ts256}"
OUT_DIR="${OUT_DIR:-results/dysts_forecasting_phase1}"
CANDIDATE_ROOT_LABEL="${CANDIDATE_ROOT_LABEL:-lista_relu_sp15_ts256}"

echo "============================================="
echo "Collect Dysts Forecast Phase 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: $PHASE1_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "CANDIDATE_ROOT_LABEL: $CANDIDATE_ROOT_LABEL"
echo "============================================="

uv run python tools/collect_dysts_forecasting.py \
  --run_roots \
    generic_sparse=runs/dysts_multi_basin_generic_sparse \
    lista_nonlinear=runs/dysts_multi_basin_lista_nonlinear \
    "${CANDIDATE_ROOT_LABEL}=${PHASE1_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --horizon 1000 \
  --good_threshold 10 \
  --essential_factor 10 \
  --select latest

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================="
exit $EXIT_CODE
