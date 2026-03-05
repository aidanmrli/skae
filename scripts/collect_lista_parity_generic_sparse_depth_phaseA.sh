#!/bin/bash
#
# Collect Phase-A parity depth pilot outputs and include generic_sparse anchor rows.
#
# Submit:
#   sbatch scripts/collect_lista_parity_generic_sparse_depth_phaseA.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/lista_parity_generic_sparse
#   OUT_DIR=results/lista_parity_generic_sparse_phaseA
#   ANCHOR_GENERIC_ROOT=/path/to/generic_sparse_root
#
#SBATCH --job-name=collect_lpar_a
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-lpar-a-%A.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_parity_generic_sparse}"
PHASEA_ROOT="${PHASEA_ROOT:-${BASE_OUT}/phaseA_depth}"
OUT_DIR="${OUT_DIR:-results/lista_parity_generic_sparse_phaseA}"

ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"
INCLUDE_ANCHOR="${INCLUDE_ANCHOR:-1}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
ESSENTIAL_FACTOR="${ESSENTIAL_FACTOR:-10}"

DEPTHS=(0 1 2 3)

RUN_ROOT_ARGS=()
if [[ "${INCLUDE_ANCHOR}" == "1" ]]; then
  RUN_ROOT_ARGS+=("generic_sparse=${ANCHOR_GENERIC_ROOT}")
fi
for depth in "${DEPTHS[@]}"; do
  RUN_ROOT_ARGS+=("depth_${depth}=${PHASEA_ROOT}/depth_${depth}")
done

echo "============================================="
echo "Collect LISTA Parity Depth Phase-A"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "PHASEA_ROOT: ${PHASEA_ROOT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "INCLUDE_ANCHOR: ${INCLUDE_ANCHOR}"
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
