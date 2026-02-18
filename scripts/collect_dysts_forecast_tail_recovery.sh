#!/bin/bash
#
#SBATCH --job-name=collect_dysts_tailrec
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-dysts-tailrec-%A.out

set -euo pipefail

source .venv/bin/activate

TAIL_ROOT="${TAIL_ROOT:-/network/scratch/l/lia/skae/dysts_forecast_tail_recovery_sp20_a020_ts256}"
OUT_DIR="${OUT_DIR:-results/dysts_forecasting_tail_recovery}"

echo "============================================="
echo "Collect Dysts Forecast Tail Recovery"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "TAIL_ROOT: $TAIL_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "============================================="

uv run python tools/collect_dysts_forecasting.py \
  --run_roots \
    generic_sparse=runs/dysts_multi_basin_generic_sparse \
    lista_nonlinear=runs/dysts_multi_basin_lista_nonlinear \
    lista_relu_sp15_ts256=/network/scratch/l/lia/skae/dysts_forecast_phase1_lista_relu_sp15_ts256 \
    lista_relu_sp20_a020_ts256_tail="${TAIL_ROOT}" \
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
