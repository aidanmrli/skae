#!/bin/bash
#
# Collect forecasting metrics from the dedicated Kuramoto recovery sweep.
#
#SBATCH --job-name=collect_kuramoto
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-kuramoto-%A.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/kuramoto_recovery_seq8_20260305}"
OUT_DIR="${OUT_DIR:-results/kuramoto_recovery_seq8_20260305}"
GENERIC_ROOT="${GENERIC_ROOT:-${BASE_OUT}/generic_sparse}"
BLOCKDIAG_ROOT="${BLOCKDIAG_ROOT:-${BASE_OUT}/lista_blockdiag}"

echo "============================================="
echo "Collect Kuramoto Recovery Seq8"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "GENERIC_ROOT: ${GENERIC_ROOT}"
echo "BLOCKDIAG_ROOT: ${BLOCKDIAG_ROOT}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots \
    "generic_sparse=${GENERIC_ROOT}" \
    "lista_blockdiag=${BLOCKDIAG_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --horizons 100 500 1000 \
  --good_threshold 10 \
  --essential_factor 10 \
  --select all

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
