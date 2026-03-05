#!/bin/bash
#
# Collect Duffing LISTA Queue-4 Pareto sweep summary + Pareto frontier.
#
# Submit:
#   sbatch scripts/collect_duffing_lista_pareto_q4.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q04_pareto_50k_20260304
#   OUT_DIR=results/duffing_lista_q04_pareto_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_q04_pareto_50k
#   TARGET_SPARSITY=0.8 SPARSITY_BAND_LOW=0.7 SPARSITY_BAND_HIGH=0.9
#
#SBATCH --job-name=collect_ls4
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-ls4-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q04_pareto_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_q04_pareto_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_q04_pareto_50k}"
TARGET_SPARSITY="${TARGET_SPARSITY:-0.8}"
SPARSITY_BAND_LOW="${SPARSITY_BAND_LOW:-0.7}"
SPARSITY_BAND_HIGH="${SPARSITY_BAND_HIGH:-0.9}"

mapfile -t ARMS < <(find "${BASE_OUT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if (( ${#ARMS[@]} == 0 )); then
  echo "No arm directories found in ${BASE_OUT}"
  exit 1
fi

ANCHOR="${ARMS[0]}"

echo "============================================="
echo "Collect Duffing LISTA Queue-4 Pareto Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ARMS: ${ARMS[*]}"
echo "TARGET_SPARSITY: ${TARGET_SPARSITY}"
echo "SPARSITY_BAND: [${SPARSITY_BAND_LOW}, ${SPARSITY_BAND_HIGH}]"
echo "============================================="

uv run python tools/summarize_encoder_comparison.py \
  --base_root "${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${SUMMARY_PREFIX}" \
  --arms "${ARMS[@]}" \
  --anchor "${ANCHOR}"

SUMMARY_JSON="${OUT_DIR}/${SUMMARY_PREFIX}_summary.json"
PARETO_PREFIX="${OUT_DIR}/${SUMMARY_PREFIX}"

uv run python tools/compute_pareto_frontier.py \
  --summary_json "${SUMMARY_JSON}" \
  --output_prefix "${PARETO_PREFIX}" \
  --target_sparsity "${TARGET_SPARSITY}" \
  --band_low "${SPARSITY_BAND_LOW}" \
  --band_high "${SPARSITY_BAND_HIGH}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
