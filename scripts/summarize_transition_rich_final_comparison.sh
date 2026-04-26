#!/bin/bash
#
# Summarize the final transition-rich basin-partition packet into one
# paper-facing comparison artifact.
#
# Required env vars:
#   FORECAST_ROWS_CSV=<forecasting_rows.csv>
#   INTERPRETABILITY_ROWS_CSV=<interpretability_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional env vars:
#   CANDIDATE_ROOTS_CSV=lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p64_hardinit_basin_partition
#   CONTROL_ROOTS_CSV=mlp_sparse_basin_partition_control,mlp_zero_sparse_basin_partition_control
#   SUPPORT_SCHEME=absolute:0.001
#   SUBSET=deep
#   GOOD_THRESHOLD=50
#
#SBATCH --job-name=tr_final_cmp
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/transition-rich-final-compare-%A.out
#SBATCH -e /network/scratch/l/lia/skae/transition-rich-final-compare-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

FORECAST_ROWS_CSV="${FORECAST_ROWS_CSV:?FORECAST_ROWS_CSV is required}"
INTERPRETABILITY_ROWS_CSV="${INTERPRETABILITY_ROWS_CSV:?INTERPRETABILITY_ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"

CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV:-lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p64_hardinit_basin_partition}"
CONTROL_ROOTS_CSV="${CONTROL_ROOTS_CSV:-mlp_sparse_basin_partition_control,mlp_zero_sparse_basin_partition_control}"
SUPPORT_SCHEME="${SUPPORT_SCHEME:-absolute:0.001}"
SUBSET="${SUBSET:-deep}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-50}"

echo "============================================="
echo "Summarize Transition-Rich Final Comparison"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "FORECAST_ROWS_CSV: ${FORECAST_ROWS_CSV}"
echo "INTERPRETABILITY_ROWS_CSV: ${INTERPRETABILITY_ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "CANDIDATE_ROOTS_CSV: ${CANDIDATE_ROOTS_CSV}"
echo "CONTROL_ROOTS_CSV: ${CONTROL_ROOTS_CSV}"
echo "SUPPORT_SCHEME: ${SUPPORT_SCHEME}"
echo "SUBSET: ${SUBSET}"
echo "GOOD_THRESHOLD: ${GOOD_THRESHOLD}"
echo "============================================="

uv run python tools/summarize_transition_rich_final_comparison.py \
  --forecast_rows_csv "${FORECAST_ROWS_CSV}" \
  --interpretability_rows_csv "${INTERPRETABILITY_ROWS_CSV}" \
  --output_dir "${OUT_DIR}" \
  --candidate_roots "${CANDIDATE_ROOTS_CSV}" \
  --control_roots "${CONTROL_ROOTS_CSV}" \
  --support_scheme "${SUPPORT_SCHEME}" \
  --subset "${SUBSET}" \
  --good_threshold "${GOOD_THRESHOLD}"
