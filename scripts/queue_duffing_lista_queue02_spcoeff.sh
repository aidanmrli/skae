#!/bin/bash
#
# Queue Duffing LISTA Queue-2:
# - Sweep SPARSITY_COEFF at fixed alpha (default 0.15)
# - 50k, L=8, 3 seeds
#
# Submit:
#   sbatch scripts/queue_duffing_lista_queue02_spcoeff.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_q02_spcoeff_50k_20260304
#   OUT_DIR=results/duffing_lista_q02_spcoeff_50k_20260304
#   SUMMARY_PREFIX=duffing_lista_q02_spcoeff_50k
#   SPARSITY_COEFFS=0.0005,0.0010,0.0020,0.0040,0.0060
#   LISTA_ALPHA=0.15
#
#SBATCH --job-name=queue_ls2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ls2-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_q02_spcoeff_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_q02_spcoeff_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_q02_spcoeff_50k}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
SPARSITY_COEFFS="${SPARSITY_COEFFS:-0.0005,0.0010,0.0020,0.0040,0.0060}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

IFS=',' read -r -a COEFFS <<< "${SPARSITY_COEFFS}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
NUM_COEFFS=${#COEFFS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_COEFFS * NUM_SEEDS))

if (( TOTAL_JOBS <= 0 )); then
  echo "No jobs to submit (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"

echo "Queueing Duffing LISTA Queue-2 SPARSITY_COEFF sweep"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SUMMARY_PREFIX: ${SUMMARY_PREFIX}"
echo "SPARSITY_COEFFS: ${SPARSITY_COEFFS}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "ARRAY_SPEC: ${ARRAY_SPEC}"
echo "LISTA_ALPHA: ${LISTA_ALPHA}"

SWEEP_JOB_ID=$(SPARSITY_COEFFS="${SPARSITY_COEFFS}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_ALPHA="${LISTA_ALPHA}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_spcoeff_q2.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_spcoeff_q2.sh" | awk '{print $4}')

echo "Submitted Queue-2 sweep array: ${SWEEP_JOB_ID}"
echo "Submitted Queue-2 collector:   ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
