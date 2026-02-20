#!/bin/bash
#
# Collect Phase-1 depth sweep outputs over mixed 23-system roots.
#
# Submit:
#   sbatch scripts/collect_lista_depth_phase1_23sys.sh
#
# Optional overrides:
#   PHASE1_ROOT=/network/scratch/l/lia/skae/lista_depth_first_23sys/phase1_depth
#   OUT_DIR=results/lista_depth_first_phase1_23sys
#   ANCHOR_GENERIC_ROOT=/path/to/generic_sparse_anchor_root
#   ANCHOR_LISTA_ROOT=/path/to/lista_nonlinear_anchor_root
#
#SBATCH --job-name=collect_lista_d1_23
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-lista-d1-23-%A.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_depth_first_23sys}"
PHASE1_ROOT="${PHASE1_ROOT:-${BASE_OUT}/phase1_depth}"
OUT_DIR="${OUT_DIR:-results/lista_depth_first_phase1_23sys}"

ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
ANCHOR_LISTA_ROOT="${ANCHOR_LISTA_ROOT:-runs/dysts_multi_basin_lista_nonlinear}"
INCLUDE_ANCHORS="${INCLUDE_ANCHORS:-1}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
ESSENTIAL_FACTOR="${ESSENTIAL_FACTOR:-10}"

DEPTHS=(1 2 3 5 7)

RUN_ROOT_ARGS=()
if [[ "${INCLUDE_ANCHORS}" == "1" ]]; then
  RUN_ROOT_ARGS+=(
    "generic_sparse=${ANCHOR_GENERIC_ROOT}"
    "lista_nonlinear=${ANCHOR_LISTA_ROOT}"
  )
fi
for depth in "${DEPTHS[@]}"; do
  RUN_ROOT_ARGS+=("depth_${depth}=${PHASE1_ROOT}/depth_${depth}")
done

echo "============================================="
echo "Collect LISTA Depth-First Phase-1 (23 systems)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "PHASE1_ROOT: ${PHASE1_ROOT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "INCLUDE_ANCHORS: ${INCLUDE_ANCHORS}"
echo "HORIZON: ${HORIZON}"
echo "GOOD_THRESHOLD: ${GOOD_THRESHOLD}"
echo "ESSENTIAL_FACTOR: ${ESSENTIAL_FACTOR}"
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
