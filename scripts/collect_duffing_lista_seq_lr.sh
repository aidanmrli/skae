#!/bin/bash
#
# Collect and compare Duffing LISTA L=8 vs L=10 LR sweep outputs.
# 6 conditions: L8_lr3e-5, L8_lr1e-4, L8_lr3e-4, L10_lr3e-5, L10_lr1e-4, L10_lr3e-4
#
# Submit:
#   sbatch scripts/collect_duffing_lista_seq_lr.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_seq_lr_20260303
#   OUT_DIR=results/duffing_lista_seq_lr_20260303
#   HORIZON=1000 SELECT=latest
#
#SBATCH --job-name=collect_duf_slr
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-duf-slr-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_seq_lr_20260303}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_seq_lr_20260303}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparison}"

HORIZON="${HORIZON:-1000}"
SELECT="${SELECT:-latest}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
CATASTROPHIC_THRESHOLD="${CATASTROPHIC_THRESHOLD:-1000}"

# Build root paths for all 6 conditions
L8_lr3e5="${BASE_OUT}/L8/lr3e-5"
L8_lr1e4="${BASE_OUT}/L8/lr1e-4"
L8_lr3e4="${BASE_OUT}/L8/lr3e-4"
L10_lr3e5="${BASE_OUT}/L10/lr3e-5"
L10_lr1e4="${BASE_OUT}/L10/lr1e-4"
L10_lr3e4="${BASE_OUT}/L10/lr3e-4"

echo "============================================="
echo "Collect Duffing LISTA Seq LR Sweep"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots \
    "L8_lr3e-5=${L8_lr3e5}" \
    "L8_lr1e-4=${L8_lr1e4}" \
    "L8_lr3e-4=${L8_lr3e4}" \
    "L10_lr3e-5=${L10_lr3e5}" \
    "L10_lr1e-4=${L10_lr1e4}" \
    "L10_lr3e-4=${L10_lr3e4}" \
  --output_dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --select "${SELECT}"

CANDIDATES=("L8_lr3e-5" "L8_lr3e-4" "L10_lr3e-5" "L10_lr1e-4" "L10_lr3e-4")
for CAND in "${CANDIDATES[@]}"; do
  echo "Comparing ${CAND} vs anchor L8_lr1e-4 ..."
  uv run python tools/compare_forecasting_roots.py \
    --rows_csv "${OUT_DIR}/forecasting_rows.csv" \
    --output_dir "${COMPARE_DIR}/${CAND}" \
    --anchor_roots "L8_lr1e-4" \
    --candidate_root "${CAND}" \
    --horizon "${HORIZON}" \
    --good_threshold "${GOOD_THRESHOLD}" \
    --catastrophic_threshold "${CATASTROPHIC_THRESHOLD}"
done

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
