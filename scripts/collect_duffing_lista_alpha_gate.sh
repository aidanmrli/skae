#!/bin/bash
#
# Collect Duffing LISTA Queue-1 gate alpha sweep summary.
#
# Submit:
#   sbatch scripts/collect_duffing_lista_alpha_gate.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_alpha_gate_10k_20260304
#   OUT_DIR=results/duffing_lista_alpha_gate_10k_20260304
#   SUMMARY_PREFIX=duffing_lista_alpha_gate_10k
#
#SBATCH --job-name=collect_ls1g
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-ls1g-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_alpha_gate_10k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_alpha_gate_10k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_alpha_gate_10k}"

mapfile -t ARMS < <(find "${BASE_OUT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if (( ${#ARMS[@]} == 0 )); then
  echo "No arm directories found in ${BASE_OUT}"
  exit 1
fi

ANCHOR="${ARMS[0]}"

echo "============================================="
echo "Collect Duffing LISTA Queue-1 Gate"
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
