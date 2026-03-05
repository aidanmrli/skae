#!/bin/bash
#
# Select depth_star_parity from Phase-A parity pilot artifacts.
#
# Submit:
#   sbatch scripts/select_lista_parity_generic_sparse_depth_phaseA.sh
#
# Optional overrides:
#   OUT_DIR=results/lista_parity_generic_sparse_phaseA
#   SELECT_DIR=results/lista_parity_generic_sparse_phaseA/selection
#   ANCHOR_ROOT=generic_sparse
#   ANCHOR_GENERIC_ROOT=/path/to/generic_sparse_root
#
#SBATCH --job-name=select_lpar_a
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/select-lpar-a-%A.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lista_parity_generic_sparse}"
PHASEA_ROOT="${PHASEA_ROOT:-${BASE_OUT}/phaseA_depth}"
OUT_DIR="${OUT_DIR:-results/lista_parity_generic_sparse_phaseA}"
SELECT_DIR="${SELECT_DIR:-${OUT_DIR}/selection}"
ROWS_CSV="${ROWS_CSV:-${OUT_DIR}/forecasting_rows.csv}"

ANCHOR_ROOT="${ANCHOR_ROOT:-generic_sparse}"
ANCHOR_GENERIC_ROOT="${ANCHOR_GENERIC_ROOT:-runs/dysts_multi_basin_generic_sparse}"

HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
CATASTROPHIC_THRESHOLD="${CATASTROPHIC_THRESHOLD:-1000}"

CATASTROPHIC_SLACK="${CATASTROPHIC_SLACK:-1}"
SEED_CATASTROPHIC_SLACK="${SEED_CATASTROPHIC_SLACK:-1}"
ALL_SEEDS_GOOD_DROP="${ALL_SEEDS_GOOD_DROP:-2}"

DEPTHS=(0 1 2 3)
CANDIDATE_ROOTS=()
RUN_ROOT_SPECS=("${ANCHOR_ROOT}=${ANCHOR_GENERIC_ROOT}")
for depth in "${DEPTHS[@]}"; do
  CANDIDATE_ROOTS+=("depth_${depth}")
  RUN_ROOT_SPECS+=("depth_${depth}=${PHASEA_ROOT}/depth_${depth}")
done

echo "============================================="
echo "Select LISTA Parity Depth Phase-A"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "SELECT_DIR: ${SELECT_DIR}"
echo "ANCHOR_ROOT: ${ANCHOR_ROOT}"
echo "============================================="

uv run python tools/select_lista_parity_depth.py \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${SELECT_DIR}" \
  --anchor_root "${ANCHOR_ROOT}" \
  --candidate_roots "${CANDIDATE_ROOTS[@]}" \
  --horizon "${HORIZON}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --catastrophic_threshold "${CATASTROPHIC_THRESHOLD}" \
  --catastrophic_slack "${CATASTROPHIC_SLACK}" \
  --seed_catastrophic_slack "${SEED_CATASTROPHIC_SLACK}" \
  --all_seeds_good_drop "${ALL_SEEDS_GOOD_DROP}" \
  --run_roots "${RUN_ROOT_SPECS[@]}"

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
