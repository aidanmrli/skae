#!/bin/bash
#
# Collect Phase-2 sparsity sweep outputs over mixed 23-system roots.
#
# Submit:
#   sbatch scripts/collect_lista_sparsity_phase2_23sys.sh
#
# Optional overrides:
#   PHASE2_ROOT=/network/scratch/l/lia/skae/lista_depth_first_23sys/phase2_sparsity
#   OUT_DIR=results/lista_depth_first_phase2_23sys
#   PHASE2_MODE=coarse|alpha
#   SPARSITY_STAR=0.20 (required for alpha mode)
#
#SBATCH --job-name=collect_lista_d2_23
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-lista-d2-23-%A.out

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

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE2_ROOT="${PHASE2_ROOT:-${BASE_OUT}/phase2_sparsity}"
OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase2_23sys}"

ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT:-runs/dysts_multi_basin_lista_nonlinear}"
INCLUDE_ANCHORS="${INCLUDE_ANCHORS:-1}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
ESSENTIAL_FACTOR="${ESSENTIAL_FACTOR:-10}"

RUN_ROOT_ARGS=()
if [[ "${INCLUDE_ANCHORS}" == "1" ]]; then
  RUN_ROOT_ARGS+=(
    "generic_sparse=${ANCHOR_GENERIC_ROOT}"
    "lista_nonlinear=${ANCHOR_LISTA_ROOT}"
  )
fi
for tag in "${CONFIG_TAGS[@]}"; do
  RUN_ROOT_ARGS+=("${tag}=${PHASE2_ROOT}/${tag}")
done

echo "============================================="
echo "Collect LISTA Depth-First Phase-2 (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "PHASE2_MODE: ${PHASE2_MODE}"
echo "PHASE2_ROOT: ${PHASE2_ROOT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "INCLUDE_ANCHORS: ${INCLUDE_ANCHORS}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots "${RUN_ROOT_ARGS[@]}" \
  --output_dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --essential_factor "${ESSENTIAL_FACTOR}" \
  --select latest

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
