#!/bin/bash
#
# Collect generic_sparse TS256 dysts forecasting benchmark summary artifacts.
#
# Submit:
#   sbatch scripts/collect_dysts_forecast_generic_sparse_ts256_cachebench.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/dysts_forecast_generic_sparse_ts256_cachebench
#   OUT_DIR=results/dysts_forecast_generic_sparse_ts256_cachebench
#   ROOT_LABEL=generic_sparse_ts256_cachebench
#
#SBATCH --job-name=collect_gs_ts256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-gs-ts256-%A.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_forecast_generic_sparse_ts256_cachebench}"
OUT_DIR="${OUT_DIR:-results/dysts_forecast_generic_sparse_ts256_cachebench}"
ROOT_LABEL="${ROOT_LABEL:-generic_sparse_ts256_cachebench}"

echo "============================================="
echo "Collect Generic Sparse TS256 Dysts Forecasting"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABEL: ${ROOT_LABEL}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots "${ROOT_LABEL}=${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --horizon 1000 \
  --good_threshold 10 \
  --essential_factor 10 \
  --select latest

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
