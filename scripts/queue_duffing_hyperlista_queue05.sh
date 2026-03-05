#!/bin/bash
#
# Queue Duffing HyperLISTA Queue-5:
# - Adaptive-threshold sweep over SPARSITY_COEFF x HyperLISTA c_theta
# - Fixed loops (default 1), 50k, L=8, 3 seeds
#
# Submit:
#   sbatch scripts/queue_duffing_hyperlista_queue05.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_hyperlista_q05_adaptive_50k_20260304
#   OUT_DIR=results/duffing_hyperlista_q05_adaptive_50k_20260304
#   SUMMARY_PREFIX=duffing_hyperlista_q05_adaptive_50k
#   SPARSITY_COEFFS=0.0040,0.0060
#   LISTA_NUM_LOOPS_CSV=1
#   HYPER_C_THETAS=0.0040,0.0060,0.0100,0.0200
#   HYPER_C_BETA=0.0001
#   HYPER_C_SS=0.5
#
#SBATCH --job-name=queue_hl5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-hl5-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_hyperlista_q05_adaptive_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_hyperlista_q05_adaptive_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_hyperlista_q05_adaptive_50k}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

SPARSITY_COEFFS="${SPARSITY_COEFFS:-0.0040,0.0060}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1}"
HYPER_C_THETAS="${HYPER_C_THETAS:-0.0040,0.0060,0.0100,0.0200}"
HYPER_C_BETA="${HYPER_C_BETA:-0.0001}"
HYPER_C_SS="${HYPER_C_SS:-0.5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

TARGET_SPARSITY="${TARGET_SPARSITY:-0.8}"
SPARSITY_BAND_LOW="${SPARSITY_BAND_LOW:-0.7}"
SPARSITY_BAND_HIGH="${SPARSITY_BAND_HIGH:-0.9}"

IFS=',' read -r -a COEFFS <<< "${SPARSITY_COEFFS}"
IFS=',' read -r -a LOOPS <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a THETAS <<< "${HYPER_C_THETAS}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
NUM_COEFFS=${#COEFFS[@]}
NUM_LOOPS=${#LOOPS[@]}
NUM_THETAS=${#THETAS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_COEFFS * NUM_LOOPS * NUM_THETAS * NUM_SEEDS))

if (( TOTAL_JOBS <= 0 )); then
  echo "No jobs to submit (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"

echo "Queueing Duffing HyperLISTA Queue-5 adaptive sweep"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "SUMMARY_PREFIX: ${SUMMARY_PREFIX}"
echo "SPARSITY_COEFFS: ${SPARSITY_COEFFS}"
echo "LISTA_NUM_LOOPS_CSV: ${LISTA_NUM_LOOPS_CSV}"
echo "HYPER_C_THETAS: ${HYPER_C_THETAS}"
echo "HYPER_C_BETA: ${HYPER_C_BETA}"
echo "HYPER_C_SS: ${HYPER_C_SS}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "ARRAY_SPEC: ${ARRAY_SPEC}"
echo "TARGET_SPARSITY: ${TARGET_SPARSITY}"
echo "SPARSITY_BAND: [${SPARSITY_BAND_LOW}, ${SPARSITY_BAND_HIGH}]"

SWEEP_JOB_ID=$(SPARSITY_COEFFS="${SPARSITY_COEFFS}" LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV}" HYPER_C_THETAS="${HYPER_C_THETAS}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",HYPER_C_BETA="${HYPER_C_BETA}",HYPER_C_SS="${HYPER_C_SS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_hyperlista_q5.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}",TARGET_SPARSITY="${TARGET_SPARSITY}",SPARSITY_BAND_LOW="${SPARSITY_BAND_LOW}",SPARSITY_BAND_HIGH="${SPARSITY_BAND_HIGH}" \
  "${ROOT_DIR}/scripts/collect_duffing_hyperlista_q5.sh" | awk '{print $4}')

echo "Submitted Queue-5 sweep array: ${SWEEP_JOB_ID}"
echo "Submitted Queue-5 collector:   ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"

