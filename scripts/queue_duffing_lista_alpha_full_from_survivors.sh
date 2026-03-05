#!/bin/bash
#
# Queue Queue-1 full LISTA alpha runs from selected survivor alphas.
#
# Typical usage (with dependency from selection job):
#   sbatch --dependency=afterany:<select_job_id> scripts/queue_duffing_lista_alpha_full_from_survivors.sh
#
# Optional overrides:
#   SELECT_DIR=results/duffing_lista_alpha_gate_10k_20260304/selection
#   SURVIVOR_CSV=${SELECT_DIR}/lista_alpha_survivors.csv
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_alpha_full_50k_20260304
#   OUT_DIR=results/duffing_lista_alpha_full_50k_20260304
#
#SBATCH --job-name=queue_ls1f
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ls1f-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

SELECT_DIR="${SELECT_DIR:-results/duffing_lista_alpha_gate_10k_20260304/selection}"
SURVIVOR_CSV="${SURVIVOR_CSV:-${SELECT_DIR}/lista_alpha_survivors.csv}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_alpha_full_50k_20260304}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_alpha_full_50k_20260304}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_alpha_full_50k}"

NUM_STEPS="${NUM_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

if [ ! -f "${SURVIVOR_CSV}" ]; then
  echo "Survivor CSV not found: ${SURVIVOR_CSV}"
  exit 1
fi

ALPHAS_CSV="$(tr -d '[:space:]' < "${SURVIVOR_CSV}")"
if [ -z "${ALPHAS_CSV}" ]; then
  echo "Survivor CSV is empty: ${SURVIVOR_CSV}"
  exit 1
fi

IFS=',' read -r -a ALPHAS <<< "${ALPHAS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

NUM_ALPHAS=${#ALPHAS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_ALPHAS * NUM_SEEDS))

if (( TOTAL_JOBS <= 0 )); then
  echo "No full-stage jobs to submit (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 1
fi

ARRAY_SPEC="0-$((TOTAL_JOBS - 1))"

echo "Queueing Duffing LISTA Queue-1 full stage"
echo "Timestamp: $(date)"
echo "SELECT_DIR: ${SELECT_DIR}"
echo "ALPHAS_CSV: ${ALPHAS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "ARRAY_SPEC: ${ARRAY_SPEC}"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"

SWEEP_JOB_ID=$(ALPHAS_CSV="${ALPHAS_CSV}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${ARRAY_SPEC}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_alpha_full.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_alpha_full.sh" | awk '{print $4}')

echo "Submitted Queue-1 full sweep array: ${SWEEP_JOB_ID}"
echo "Submitted Queue-1 full collector:   ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
