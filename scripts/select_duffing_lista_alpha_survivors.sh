#!/bin/bash
#
# Select survivor alphas from Duffing LISTA Queue-1 gate summary.
#
# Submit:
#   sbatch scripts/select_duffing_lista_alpha_survivors.sh
#
# Optional overrides:
#   OUT_DIR=results/duffing_lista_alpha_gate_10k_20260304
#   SUMMARY_PREFIX=duffing_lista_alpha_gate_10k
#   SELECT_DIR=results/duffing_lista_alpha_gate_10k_20260304/selection
#   MAX_SURVIVORS=3 TARGET_LOW=0.7 TARGET_HIGH=0.9 MIN_RUNS=3
#
#SBATCH --job-name=select_ls1
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/select-ls1-%j.out

set -euo pipefail

source .venv/bin/activate

OUT_DIR="${OUT_DIR:-results/duffing_lista_alpha_gate_10k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_alpha_gate_10k}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_DIR}/${SUMMARY_PREFIX}_summary.json}"
SELECT_DIR="${SELECT_DIR:-${OUT_DIR}/selection}"

MAX_SURVIVORS="${MAX_SURVIVORS:-3}"
TARGET_LOW="${TARGET_LOW:-0.7}"
TARGET_HIGH="${TARGET_HIGH:-0.9}"
MIN_RUNS="${MIN_RUNS:-3}"

echo "============================================="
echo "Select Duffing LISTA Queue-1 Survivors"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "SUMMARY_JSON: ${SUMMARY_JSON}"
echo "SELECT_DIR: ${SELECT_DIR}"
echo "============================================="

uv run python tools/select_lista_alpha_survivors.py \
  --summary_json "${SUMMARY_JSON}" \
  --output_dir "${SELECT_DIR}" \
  --max_survivors "${MAX_SURVIVORS}" \
  --target_low "${TARGET_LOW}" \
  --target_high "${TARGET_HIGH}" \
  --min_runs "${MIN_RUNS}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
