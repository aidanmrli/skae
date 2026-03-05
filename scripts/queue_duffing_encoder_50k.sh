#!/bin/bash
#
# Queue Duffing encoder comparison (50k steps): sweep + collection.
#
# Submit:
#   sbatch scripts/queue_duffing_encoder_50k.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_encoder_50k_20260303
#   OUT_DIR=results/duffing_encoder_50k_20260303
#   SUMMARY_PREFIX=duffing_encoder_50k
#   NUM_STEPS=50000
#
#SBATCH --job-name=queue_enc50k
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-enc50k-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_encoder_50k_20260303}"
OUT_DIR="${OUT_DIR:-results/duffing_encoder_50k_20260303}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_encoder_50k}"
NUM_STEPS="${NUM_STEPS:-50000}"

echo "Queueing Duffing Encoder 50k Comparison"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "NUM_STEPS: ${NUM_STEPS}"

SWEEP_JOB_ID=$(sbatch \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_encoder_50k.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_encoder_50k.sh" | awk '{print $4}')

echo "Submitted sweep array: ${SWEEP_JOB_ID}"
echo "Submitted collection:  ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
