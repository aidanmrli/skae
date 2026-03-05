#!/bin/bash
#
# Collect Duffing LISTA Queue-0 baseline summary.
#
# Submit:
#   sbatch scripts/collect_duffing_lista_relu_baseline_50k.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_relu_baseline_50k_20260304
#   OUT_DIR=results/duffing_lista_relu_baseline_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_relu_baseline_50k
#
#SBATCH --job-name=collect_ls0
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-ls0-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_relu_baseline_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_relu_baseline_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_relu_baseline_50k}"
ARM="${ARM:-lista_relu_baseline}"

echo "============================================="
echo "Collect Duffing LISTA Queue-0 Baseline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ARM: ${ARM}"
echo "============================================="

uv run python tools/summarize_encoder_comparison.py \
  --base_root "${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${SUMMARY_PREFIX}" \
  --arms "${ARM}" \
  --anchor "${ARM}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
