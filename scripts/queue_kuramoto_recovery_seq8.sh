#!/bin/bash
#
# Queue the dedicated Kuramoto recovery sweeps and collector.
#
# Submit:
#   sbatch scripts/queue_kuramoto_recovery_seq8.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/kuramoto_recovery_seq8_20260305
#   OUT_DIR=results/kuramoto_recovery_seq8_20260305
#   NUM_STEPS_CSV=20000
#   SEEDS_CSV=0,1,2
#   SPARSITY_COEFFS_CSV=0.0005,0.0010,0.0025
#   LISTA_ALPHAS_CSV=0.15
#   LISTA_NUM_LOOPS_CSV=1,3,5
#   K_BLOCK_SIZES_CSV=16
#
#SBATCH --job-name=queue_kuramoto
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-kuramoto-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/kuramoto_recovery_seq8_20260305}"
OUT_DIR="${OUT_DIR:-results/kuramoto_recovery_seq8_20260305}"

NUM_STEPS_CSV="${NUM_STEPS_CSV:-20000}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
SPARSITY_COEFFS_CSV="${SPARSITY_COEFFS_CSV:-0.0005,0.0010,0.0025}"
LISTA_ALPHAS_CSV="${LISTA_ALPHAS_CSV:-0.15}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-1,3,5}"
K_BLOCK_SIZES_CSV="${K_BLOCK_SIZES_CSV:-16}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"

IFS=',' read -r -a NUM_STEPS_LIST <<< "${NUM_STEPS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a SPARSITY_COEFFS <<< "${SPARSITY_COEFFS_CSV}"
IFS=',' read -r -a LISTA_ALPHAS <<< "${LISTA_ALPHAS_CSV}"
IFS=',' read -r -a LISTA_NUM_LOOPS_LIST <<< "${LISTA_NUM_LOOPS_CSV}"
IFS=',' read -r -a K_BLOCK_SIZES <<< "${K_BLOCK_SIZES_CSV}"

NUM_STEP_OPTIONS=${#NUM_STEPS_LIST[@]}
NUM_SEEDS=${#SEEDS[@]}
NUM_SPARSITY=${#SPARSITY_COEFFS[@]}
NUM_ALPHAS=${#LISTA_ALPHAS[@]}
NUM_LOOPS=${#LISTA_NUM_LOOPS_LIST[@]}
NUM_BLOCKS=${#K_BLOCK_SIZES[@]}

GENERIC_TOTAL=$((NUM_STEP_OPTIONS * NUM_SPARSITY * NUM_SEEDS))
BLOCKDIAG_TOTAL=$((NUM_STEP_OPTIONS * NUM_SPARSITY * NUM_ALPHAS * NUM_LOOPS * NUM_BLOCKS * NUM_SEEDS))

if (( GENERIC_TOTAL <= 0 || BLOCKDIAG_TOTAL <= 0 )); then
  echo "Invalid grid sizes: generic=${GENERIC_TOTAL}, blockdiag=${BLOCKDIAG_TOTAL}"
  exit 1
fi

GENERIC_ARRAY="0-$((GENERIC_TOTAL - 1))"
BLOCKDIAG_ARRAY="0-$((BLOCKDIAG_TOTAL - 1))"

echo "Queueing Kuramoto recovery sweeps"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"
echo "NUM_STEPS_CSV: ${NUM_STEPS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "SPARSITY_COEFFS_CSV: ${SPARSITY_COEFFS_CSV}"
echo "LISTA_ALPHAS_CSV: ${LISTA_ALPHAS_CSV}"
echo "LISTA_NUM_LOOPS_CSV: ${LISTA_NUM_LOOPS_CSV}"
echo "K_BLOCK_SIZES_CSV: ${K_BLOCK_SIZES_CSV}"
echo "GENERIC_ARRAY: ${GENERIC_ARRAY}"
echo "BLOCKDIAG_ARRAY: ${BLOCKDIAG_ARRAY}"

GENERIC_JOB_ID=$(SEEDS_CSV="${SEEDS_CSV}" SPARSITY_COEFFS_CSV="${SPARSITY_COEFFS_CSV}" NUM_STEPS_CSV="${NUM_STEPS_CSV}" sbatch \
  --array="${GENERIC_ARRAY}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",MODEL_VARIANT=generic_sparse,BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",RES_COEFF="${RES_COEFF}",RECONST_COEFF="${RECONST_COEFF}",PRED_COEFF="${PRED_COEFF}" \
  "${ROOT_DIR}/scripts/sweep_kuramoto_recovery_seq8.sh" | awk '{print $4}')

BLOCKDIAG_JOB_ID=$(SEEDS_CSV="${SEEDS_CSV}" SPARSITY_COEFFS_CSV="${SPARSITY_COEFFS_CSV}" NUM_STEPS_CSV="${NUM_STEPS_CSV}" LISTA_ALPHAS_CSV="${LISTA_ALPHAS_CSV}" LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV}" K_BLOCK_SIZES_CSV="${K_BLOCK_SIZES_CSV}" sbatch \
  --array="${BLOCKDIAG_ARRAY}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",MODEL_VARIANT=lista_blockdiag,BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",RES_COEFF="${RES_COEFF}",RECONST_COEFF="${RECONST_COEFF}",PRED_COEFF="${PRED_COEFF}",LISTA_FINAL_OP="${LISTA_FINAL_OP}" \
  "${ROOT_DIR}/scripts/sweep_kuramoto_recovery_seq8.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${GENERIC_JOB_ID}:${BLOCKDIAG_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}" \
  "${ROOT_DIR}/scripts/collect_kuramoto_recovery_seq8.sh" | awk '{print $4}')

echo "Submitted generic_sparse sweep array: ${GENERIC_JOB_ID}"
echo "Submitted lista_blockdiag sweep array: ${BLOCKDIAG_JOB_ID}"
echo "Submitted collector: ${COLLECT_JOB_ID} (afterany:${GENERIC_JOB_ID}:${BLOCKDIAG_JOB_ID})"
