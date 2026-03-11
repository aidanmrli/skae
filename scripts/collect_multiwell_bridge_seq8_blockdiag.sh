#!/bin/bash
#
# Collect focused seq8 block-diagonal LISTA sweep outputs for the 2D multi-well bridge systems.
#
# Submit:
#   sbatch scripts/collect_multiwell_bridge_seq8_blockdiag.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/multiwell_bridge_blockdiag_seq8_20260305
#   OUT_DIR=results/multiwell_bridge_blockdiag_seq8_20260305
#   TARGET_SIZES_CSV=64,128,256
#   LISTA_NUM_LOOPS_CSV=1,3,5
#
#SBATCH --job-name=collect_mwb_s8
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH -o /network/scratch/l/lia/skae/collect-mwb-s8-%A.out
#SBATCH -e /network/scratch/l/lia/skae/collect-mwb-s8-%A.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORK_DIR}"

source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/multiwell_bridge_blockdiag_seq8_20260305}"
OUT_DIR="${OUT_DIR:-results/multiwell_bridge_blockdiag_seq8_20260305}"
TARGET_SIZES_CSV="${TARGET_SIZES_CSV:-64,128,256}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,3,5}"
HORIZON="${HORIZON:-1000}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-10}"
ESSENTIAL_FACTOR="${ESSENTIAL_FACTOR:-10}"

IFS=',' read -r -a TARGET_SIZES <<< "${TARGET_SIZES_CSV}"
IFS=',' read -r -a LISTA_NUM_LOOPS_LIST <<< "${LISTA_NUM_LOOPS_CSV}"

RUN_ROOT_ARGS=()
for target_size in "${TARGET_SIZES[@]}"; do
  for loops in "${LISTA_NUM_LOOPS_LIST[@]}"; do
    label="ts_${target_size}_loops_${loops}"
    root="${BASE_OUT}/ts_${target_size}/loops_${loops}"
    RUN_ROOT_ARGS+=("${label}=${root}")
  done
done

echo "============================================="
echo "Collect MultiWell Bridge Seq8 BlockDiag"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Start Time: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
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
