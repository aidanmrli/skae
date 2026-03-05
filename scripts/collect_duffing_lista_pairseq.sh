#!/bin/bash
#
# Collect and compare Duffing LISTA pairwise-vs-sequence outputs (L=1 vs L=8).
#
# Submit:
#   sbatch scripts/collect_duffing_lista_pairseq.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_pairseq_quick_20260303
#   OUT_DIR=results/duffing_lista_pairseq_quick_20260303
#   SUMMARY_PREFIX=duffing_lista_pairseq_quick
#   HORIZON=1000 SELECT=latest
#
#SBATCH --job-name=collect_duf_ls
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-duf-ls-%j.out

set -euo pipefail

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_pairseq_quick_20260303}"
L1_ROOT="${L1_ROOT:-${BASE_OUT}/L1}"
L8_ROOT="${L8_ROOT:-${BASE_OUT}/L8}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_pairseq_quick_20260303}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparison}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_pairseq_quick}"

HORIZON="${HORIZON:-1000}"
SELECT="${SELECT:-latest}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
CATASTROPHIC_THRESHOLD="${CATASTROPHIC_THRESHOLD:-1000}"

echo "============================================="
echo "Collect Duffing LISTA PairSeq"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "L1_ROOT: ${L1_ROOT}"
echo "L8_ROOT: ${L8_ROOT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "============================================="

uv run python tools/collect_forecasting_roots.py \
  --run_roots "lista_l1=${L1_ROOT}" "lista_l8=${L8_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --horizon "${HORIZON}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --select "${SELECT}"

uv run python tools/compare_forecasting_roots.py \
  --rows_csv "${OUT_DIR}/forecasting_rows.csv" \
  --output_dir "${COMPARE_DIR}" \
  --candidate_root "lista_l8" \
  --anchor_roots "lista_l1" \
  --horizon "${HORIZON}" \
  --good_threshold "${GOOD_THRESHOLD}" \
  --catastrophic_threshold "${CATASTROPHIC_THRESHOLD}"

uv run python tools/summarize_duffing_pairseq.py \
  --base_root "${BASE_OUT}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${SUMMARY_PREFIX}"

EXIT_CODE=$?
echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: ${EXIT_CODE}"
echo "============================================="
exit ${EXIT_CODE}
