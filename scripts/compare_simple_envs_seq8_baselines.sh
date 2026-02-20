#!/bin/bash
#
# Compare LISTA-vs-generic_sparse for simple-environment sequence-L8 baselines.
#
# Submit:
#   sbatch scripts/compare_simple_envs_seq8_baselines.sh
#
# Optional overrides:
#   OUT_DIR=results/simple_envs_seq8_baselines
#   COMPARE_DIR=results/simple_envs_seq8_baselines/comparison
#   CANDIDATE_ROOT=lista_best_seq8
#   ANCHOR_ROOT=generic_sparse
#   HORIZON=1000
#
#SBATCH --job-name=simple_s8_compare
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/simple-s8-compare-%j.out

set -euo pipefail

source .venv/bin/activate

OUT_DIR="${OUT_DIR:-results/simple_envs_seq8_baselines}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparison}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-lista_best_seq8}"
ANCHOR_ROOT="${ANCHOR_ROOT:-generic_sparse}"
HORIZON="${HORIZON:-1000}"
ROWS_CSV="${ROWS_CSV:-${OUT_DIR}/forecasting_rows.csv}"

echo "============================================="
echo "Compare Simple Envs Sequence-L8 Baselines"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "COMPARE_DIR: ${COMPARE_DIR}"
echo "CANDIDATE_ROOT: ${CANDIDATE_ROOT}"
echo "ANCHOR_ROOT: ${ANCHOR_ROOT}"
echo "============================================="

uv run python tools/compare_forecasting_roots.py \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${COMPARE_DIR}" \
  --candidate_root "${CANDIDATE_ROOT}" \
  --anchor_roots "${ANCHOR_ROOT}" \
  --horizon "${HORIZON}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
