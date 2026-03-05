#!/bin/bash
#
# Queue Duffing LISTA Queue-4:
# - Joint Pareto sweep over SPARSITY_COEFF x LISTA num_loops
# - Fixed alpha (default 0.15), 50k, L=8, 3 seeds
#
# Submit:
#   sbatch scripts/queue_duffing_lista_queue04_pareto.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q04_pareto_50k_20260304
#   OUT_DIR=results/duffing_lista_q04_pareto_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_q04_pareto_50k
#   SPARSITY_COEFFS=0.0005,0.0010,0.0020,0.0040,0.0060
#   LISTA_NUM_LOOPS_CSV=1,2,5,7
#   LISTA_ALPHA=0.15
#
#SBATCH --job-name=queue_ls4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ls4-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q04_pareto_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_q04_pareto_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_q04_pareto_50k}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
SPARSITY_COEFFS="${SPARSITY_COEFFS:-0.0005,0.0010,0.0020,0.0040,0.0060}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,2,5,7}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

TARGET_SPARSITY="${TARGET_SPARSITY:-0.8}"
SPARSITY_BAND_LOW="${SPARSITY_BAND_LOW:-0.7}"
SPARSITY_BAND_HIGH="${SPARSITY_BAND_HIGH:-0.9}"

IFS=',' read -r -a COEFFS <<< "${SPARSITY_COEFFS}"
IFS=',' read -r -a LOOPS <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
NUM_COEFFS=${#COEFFS[@]}
NUM_LOOPS=${#LOOPS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_COEFFS * NUM_LOOPS * NUM_SEEDS))

if (( TOTAL_JOBS <= 0 )); then
  echo "No jobs to submit (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"

echo "Queueing Duffing LISTA Queue-4 Pareto sweep"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SUMMARY_PREFIX: ${SUMMARY_PREFIX}"
echo "SPARSITY_COEFFS: ${SPARSITY_COEFFS}"
echo "LISTA_NUM_LOOPS_CSV: ${LISTA_NUM_LOOPS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "ARRAY_SPEC: ${ARRAY_SPEC}"
echo "LISTA_ALPHA: ${LISTA_ALPHA}"
echo "TARGET_SPARSITY: ${TARGET_SPARSITY}"
echo "SPARSITY_BAND: [${SPARSITY_BAND_LOW}, ${SPARSITY_BAND_HIGH}]"

SWEEP_JOB_ID=$(SPARSITY_COEFFS="${SPARSITY_COEFFS}" LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_ALPHA="${LISTA_ALPHA}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_pareto_q4.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}",TARGET_SPARSITY="${TARGET_SPARSITY}",SPARSITY_BAND_LOW="${SPARSITY_BAND_LOW}",SPARSITY_BAND_HIGH="${SPARSITY_BAND_HIGH}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_pareto_q4.sh" | awk '{print $4}')

echo "Submitted Queue-4 sweep array: ${SWEEP_JOB_ID}"
echo "Submitted Queue-4 collector:   ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
