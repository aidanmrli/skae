#!/bin/bash
#
# Collect and summarize Duffing encoder comparison (50k steps).
#
# Submit (usually as dependency after sweep):
#   sbatch scripts/collect_duffing_encoder_50k.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_encoder_50k_20260303
#   OUT_DIR=results/duffing_encoder_50k_20260303
#   SUMMARY_PREFIX=duffing_encoder_50k
#
#SBATCH --job-name=collect_enc50k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-enc50k-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_encoder_50k_20260303}"
OUT_DIR="${OUT_DIR:-results/duffing_encoder_50k_20260303}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_encoder_50k}"

echo "============================================="
echo "Collect Duffing Encoder 50k Comparison"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "============================================="

uv run python tools/summarize_encoder_comparison.py \
  --base_root "${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${SUMMARY_PREFIX}" \
  --arms lista_current lista_matched generic_sparse \
  --anchor generic_sparse

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
