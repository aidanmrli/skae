#!/bin/bash
#
#SBATCH --job-name=compare_dysts_fcast_p1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/compare-dysts-fcast-p1-%A.out

set -euo pipefail

source .venv/bin/activate

PHASE1_ROOT="${PHASE1_ROOT:-/network/scratch/l/lia/skae/dysts_forecast_phase1_lista_relu_sp15_ts256}"
OUT_DIR="${OUT_DIR:-results/dysts_forecasting_phase1}"
SEED_OUT_DIR="${SEED_OUT_DIR:-results/dysts_forecasting_phase1_seed_all}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-lista_relu_sp15_ts256}"

echo "============================================="
echo "Compare Dysts Forecast Phase 1"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: $PHASE1_ROOT"
echo "OUT_DIR: $OUT_DIR"
echo "SEED_OUT_DIR: $SEED_OUT_DIR"
echo "CANDIDATE_ROOT: $CANDIDATE_ROOT"
echo "============================================="

uv run python tools/compare_dysts_forecasting_roots.py \
  --rows_csv "${OUT_DIR}/dysts_forecasting_rows.csv" \
  --output_dir "${OUT_DIR}" \
  --candidate_root "${CANDIDATE_ROOT}" \
  --anchor_roots generic_sparse lista_nonlinear \
  --horizon 1000 \
  --good_threshold 10 \
  --catastrophic_threshold 1000

uv run python tools/collect_dysts_forecasting.py \
  --run_roots "${CANDIDATE_ROOT}=${PHASE1_ROOT}" \
  --output_dir "${SEED_OUT_DIR}" \
  --horizon 1000 \
  --good_threshold 10 \
  --essential_factor 10 \
  --select all

uv run python tools/compare_dysts_forecasting_roots.py \
  --rows_csv "${SEED_OUT_DIR}/dysts_forecasting_rows.csv" \
  --output_dir "${SEED_OUT_DIR}" \
  --candidate_root "${CANDIDATE_ROOT}" \
  --anchor_roots generic_sparse lista_nonlinear \
  --horizon 1000 \
  --good_threshold 10 \
  --catastrophic_threshold 1000

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================="
exit $EXIT_CODE
