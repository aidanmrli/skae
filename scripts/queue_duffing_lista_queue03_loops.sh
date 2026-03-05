#!/bin/bash
#
# Queue Duffing LISTA Queue-3:
# - Sweep LISTA num_loops at fixed alpha/sparsity from Queue-2 winner
# - 50k, L=8, 3 seeds
#
# Submit:
#   sbatch scripts/queue_duffing_lista_queue03_loops.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304
#   OUT_DIR=results/duffing_lista_q03_loops_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_q03_loops_50k
#   LISTA_NUM_LOOPS_CSV=1,2,3,5,7
#   LISTA_ALPHA=0.15 SPARSITY_COEFF=0.0005
#
#SBATCH --job-name=queue_ls3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ls3-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q03_loops_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_q03_loops_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_q03_loops_50k}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0005}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,2,3,5,7}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

IFS=',' read -r -a LOOPS <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
NUM_LOOPS=${#LOOPS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_LOOPS * NUM_SEEDS))

if (( TOTAL_JOBS <= 0 )); then
  echo "No jobs to submit (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"

echo "Queueing Duffing LISTA Queue-3 depth/capacity sweep"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SUMMARY_PREFIX: ${SUMMARY_PREFIX}"
echo "LISTA_NUM_LOOPS_CSV: ${LISTA_NUM_LOOPS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "ARRAY_SPEC: ${ARRAY_SPEC}"
echo "LISTA_ALPHA: ${LISTA_ALPHA}"
echo "SPARSITY_COEFF: ${SPARSITY_COEFF}"

SWEEP_JOB_ID=$(LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_ALPHA="${LISTA_ALPHA}",SPARSITY_COEFF="${SPARSITY_COEFF}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_loops_q3.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_loops_q3.sh" | awk '{print $4}')

echo "Submitted Queue-3 sweep array: ${SWEEP_JOB_ID}"
echo "Submitted Queue-3 collector:   ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
