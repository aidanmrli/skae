#!/bin/bash
#
# Collect Duffing LISTA Queue-3 depth/capacity sweep summary.
#
# Submit:
#   sbatch scripts/collect_duffing_lista_loops_q3.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304
#   OUT_DIR=results/duffing_lista_q03_loops_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_q03_loops_50k
#
#SBATCH --job-name=collect_ls3
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-ls3-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_q03_loops_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_q03_loops_50k}"

mapfile -t ARMS < <(find "${BASE_OUT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if (( ${#ARMS[@]} == 0 )); then
  echo "No arm directories found in ${BASE_OUT}"
  exit 1
fi

ANCHOR="${ARMS[0]}"

echo "============================================="
echo "Collect Duffing LISTA Queue-3 Depth/Capacity Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ARMS: ${ARMS[*]}"
echo "============================================="

uv run python tools/summarize_encoder_comparison.py \
  --base_root "${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${SUMMARY_PREFIX}" \
  --arms "${ARMS[@]}" \
  --anchor "${ANCHOR}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
