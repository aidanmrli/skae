#!/bin/bash
#
# Compare Phase-1 depth roots against anchor roots using mixed-system collector rows.
#
# Submit:
#   sbatch scripts/compare_lista_depth_phase1_23sys.sh
#
# Optional overrides:
#   OUT_DIR=results/lista_depth_first_phase1_23sys
#   COMPARE_DIR=results/lista_depth_first_phase1_23sys/comparisons
#   ANCHOR_ROOTS="generic_sparse lista_nonlinear"
#
#SBATCH --job-name=compare_lista_d1_23
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/compare-lista-d1-23-%A.out

set -euo pipefail

source .venv/bin/activate

OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase1_23sys}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparisons}"
ROWS_CSV="${ROWS_CSV:-${OUT_DIR}/forecasting_rows.csv}"
ANCHOR_ROOTS="${ANCHOR_ROOTS:-generic_sparse lista_nonlinear}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
CATASTROPHIC_THRESHOLD="${CATASTROPHIC_THRESHOLD:-1000}"
TOP_K="${TOP_K:-8}"

DEPTHS=(1 2 3 5 7)

echo "============================================="
echo "Compare LISTA Depth-First Phase-1 (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "COMPARE_DIR: ${COMPARE_DIR}"
echo "ANCHOR_ROOTS: ${ANCHOR_ROOTS}"
echo "============================================="

mkdir -p "${COMPARE_DIR}"

for depth in "${DEPTHS[@]}"; do
  candidate="depth_${depth}"
  out_subdir="${COMPARE_DIR}/${candidate}"
  mkdir -p "${out_subdir}"

  echo "--- Comparing candidate root: ${candidate}"
  uv run python tools/compare_forecasting_roots.py \
    --rows_csv "${ROWS_CSV}" \
    --output_dir "${out_subdir}" \
    --candidate_root "${candidate}" \
    --anchor_roots ${ANCHOR_ROOTS} \
    --horizon "${HORIZON}" \
    --good_threshold "${GOOD_THRESHOLD}" \
    --catastrophic_threshold "${CATASTROPHIC_THRESHOLD}" \
    --top_k "${TOP_K}"
done

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
