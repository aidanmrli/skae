#!/bin/bash
#
# Compare Phase-2 sparsity candidates against anchor roots.
#
# Submit:
#   sbatch scripts/compare_lista_sparsity_phase2_23sys.sh
#
# Optional overrides:
#   OUT_DIR=results/lista_depth_first_phase2_23sys
#   COMPARE_DIR=results/lista_depth_first_phase2_23sys/comparisons
#   PHASE2_MODE=coarse|alpha
#   SPARSITY_STAR=0.20 (required for alpha mode)
#
#SBATCH --job-name=compare_lista_d2_23
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/compare-lista-d2-23-%A.out

set -euo pipefail

source .venv/bin/activate

PHASE2_MODE="${PHASE2_MODE:-coarse}"
ALPHA_FIXED="${ALPHA_FIXED:-0.15}"
SPARSITY_STAR="${SPARSITY_STAR:-}"

CONFIG_TAGS=()
if [[ "${PHASE2_MODE}" == "coarse" ]]; then
  SP_GRID=(0.05 0.10 0.20 0.40 0.80)
  for sp in "${SP_GRID[@]}"; do
    CONFIG_TAGS+=("sp${sp}_a${ALPHA_FIXED}")
  done
elif [[ "${PHASE2_MODE}" == "alpha" ]]; then
  if [[ -z "${SPARSITY_STAR}" ]]; then
    echo "SPARSITY_STAR is required when PHASE2_MODE=alpha"
    exit 1
  fi
  ALPHA_GRID=(0.10 0.15 0.25 0.35)
  for alpha in "${ALPHA_GRID[@]}"; do
    CONFIG_TAGS+=("sp${SPARSITY_STAR}_a${alpha}")
  done
else
  echo "Unsupported PHASE2_MODE='${PHASE2_MODE}'. Use 'coarse' or 'alpha'."
  exit 1
fi

OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase2_23sys}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparisons}"
ROWS_CSV="${ROWS_CSV:-${OUT_DIR}/forecasting_rows.csv}"
ANCHOR_ROOTS="${ANCHOR_ROOTS:-generic_sparse lista_nonlinear}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
CATASTROPHIC_THRESHOLD="${CATASTROPHIC_THRESHOLD:-1000}"
TOP_K="${TOP_K:-8}"

echo "============================================="
echo "Compare LISTA Depth-First Phase-2 (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "PHASE2_MODE: ${PHASE2_MODE}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "COMPARE_DIR: ${COMPARE_DIR}"
echo "ANCHOR_ROOTS: ${ANCHOR_ROOTS}"
echo "============================================="

mkdir -p "${COMPARE_DIR}"

for tag in "${CONFIG_TAGS[@]}"; do
  out_subdir="${COMPARE_DIR}/${tag}"
  mkdir -p "${out_subdir}"

  echo "--- Comparing candidate root: ${tag}"
  uv run python tools/compare_forecasting_roots.py \
    --rows_csv "${ROWS_CSV}" \
    --output_dir "${out_subdir}" \
    --candidate_root "${tag}" \
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
