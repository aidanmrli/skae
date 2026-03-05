#!/bin/bash
#
# Queue LISTA forecasting Queue 0-1 plan on Duffing:
#   Queue 0: post-change LISTA ReLU baseline (50k, 3 seeds)
#   Queue 1: alpha gate sweep (10k) -> survivor select -> full sweep (50k)
#
# Submit:
#   sbatch scripts/queue_duffing_lista_queue01.sh
#
# Optional overrides:
#   BASE_ROOT=/network/scratch/l/lia/skae/duffing_lista_q01_20260304
#   RESULTS_ROOT=results/duffing_lista_q01_20260304
#   LISTA_ALPHAS=0.10,0.15,0.20,0.30,0.40
#   MAX_SURVIVORS=3 TARGET_LOW=0.7 TARGET_HIGH=0.9
#
#SBATCH --job-name=queue_ls_q01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00
#SBATCH -o /network/scratch/l/lia/skae/queue-ls-q01-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_ROOT="${BASE_ROOT:-/network/scratch/l/lia/skae/duffing_lista_q01_20260304}"
RESULTS_ROOT="${RESULTS_ROOT:-results/duffing_lista_q01_20260304}"

BASELINE_BASE_OUT="${BASELINE_BASE_OUT:-${BASE_ROOT}/queue0_baseline}"
BASELINE_OUT_DIR="${BASELINE_OUT_DIR:-${RESULTS_ROOT}/queue0_baseline}"
BASELINE_SUMMARY_PREFIX="${BASELINE_SUMMARY_PREFIX:-duffing_lista_relu_baseline_50k}"

GATE_BASE_OUT="${GATE_BASE_OUT:-${BASE_ROOT}/queue1_gate}"
GATE_OUT_DIR="${GATE_OUT_DIR:-${RESULTS_ROOT}/queue1_gate}"
GATE_SUMMARY_PREFIX="${GATE_SUMMARY_PREFIX:-duffing_lista_alpha_gate_10k}"
SELECT_DIR="${SELECT_DIR:-${GATE_OUT_DIR}/selection}"

FULL_BASE_OUT="${FULL_BASE_OUT:-${BASE_ROOT}/queue1_full}"
FULL_OUT_DIR="${FULL_OUT_DIR:-${RESULTS_ROOT}/queue1_full}"
FULL_SUMMARY_PREFIX="${FULL_SUMMARY_PREFIX:-duffing_lista_alpha_full_50k}"

LISTA_ALPHAS="${LISTA_ALPHAS:-0.10,0.15,0.20,0.30,0.40}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"

BASELINE_NUM_STEPS="${BASELINE_NUM_STEPS:-50000}"
GATE_NUM_STEPS="${GATE_NUM_STEPS:-10000}"
FULL_NUM_STEPS="${FULL_NUM_STEPS:-50000}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

MAX_SURVIVORS="${MAX_SURVIVORS:-3}"
TARGET_LOW="${TARGET_LOW:-0.7}"
TARGET_HIGH="${TARGET_HIGH:-0.9}"
MIN_RUNS="${MIN_RUNS:-3}"

IFS=',' read -r -a ALPHA_ARR <<< "${LISTA_ALPHAS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS_CSV}"
NUM_ALPHAS=${#ALPHA_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}
GATE_TOTAL=$((NUM_ALPHAS * NUM_SEEDS))
GATE_ARRAY="0-$((GATE_TOTAL - 1))"
BASELINE_ARRAY="0-$((NUM_SEEDS - 1))"

echo "Queueing Duffing LISTA Queue 0-1"
echo "Timestamp: $(date)"
echo "BASE_ROOT: ${BASE_ROOT}"
echo "RESULTS_ROOT: ${RESULTS_ROOT}"
echo "LISTA_ALPHAS: ${LISTA_ALPHAS}"
echo "SEEDS_CSV: ${SEEDS_CSV}"

BASELINE_SWEEP_JOB_ID=$(sbatch \
  --array="${BASELINE_ARRAY}" \
  --export=ALL,BASE_OUT="${BASELINE_BASE_OUT}",NUM_STEPS="${BASELINE_NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_ALPHA=0.10,LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_relu_baseline_50k.sh" | awk '{print $4}')

BASELINE_COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${BASELINE_SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASELINE_BASE_OUT}",OUT_DIR="${BASELINE_OUT_DIR}",SUMMARY_PREFIX="${BASELINE_SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_relu_baseline_50k.sh" | awk '{print $4}')

GATE_SWEEP_JOB_ID=$(LISTA_ALPHAS="${LISTA_ALPHAS}" SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --array="${GATE_ARRAY}" \
  --export=ALL,BASE_OUT="${GATE_BASE_OUT}",NUM_STEPS="${GATE_NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_alpha_gate.sh" | awk '{print $4}')

GATE_COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${GATE_SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${GATE_BASE_OUT}",OUT_DIR="${GATE_OUT_DIR}",SUMMARY_PREFIX="${GATE_SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_alpha_gate.sh" | awk '{print $4}')

SELECT_JOB_ID=$(sbatch \
  --dependency="afterany:${GATE_COLLECT_JOB_ID}" \
  --export=ALL,OUT_DIR="${GATE_OUT_DIR}",SUMMARY_PREFIX="${GATE_SUMMARY_PREFIX}",SELECT_DIR="${SELECT_DIR}",MAX_SURVIVORS="${MAX_SURVIVORS}",TARGET_LOW="${TARGET_LOW}",TARGET_HIGH="${TARGET_HIGH}",MIN_RUNS="${MIN_RUNS}" \
  "${ROOT_DIR}/scripts/select_duffing_lista_alpha_survivors.sh" | awk '{print $4}')

FULL_QUEUE_JOB_ID=$(SEEDS_CSV="${SEEDS_CSV}" sbatch \
  --dependency="afterany:${SELECT_JOB_ID}" \
  --export=ALL,SELECT_DIR="${SELECT_DIR}",BASE_OUT="${FULL_BASE_OUT}",OUT_DIR="${FULL_OUT_DIR}",SUMMARY_PREFIX="${FULL_SUMMARY_PREFIX}",NUM_STEPS="${FULL_NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}" \
  "${ROOT_DIR}/scripts/queue_duffing_lista_alpha_full_from_survivors.sh" | awk '{print $4}')

echo "Submitted Queue-0 sweep array:      ${BASELINE_SWEEP_JOB_ID}"
echo "Submitted Queue-0 collector:        ${BASELINE_COLLECT_JOB_ID} (afterany:${BASELINE_SWEEP_JOB_ID})"
echo "Submitted Queue-1 gate sweep array: ${GATE_SWEEP_JOB_ID}"
echo "Submitted Queue-1 gate collector:   ${GATE_COLLECT_JOB_ID} (afterany:${GATE_SWEEP_JOB_ID})"
echo "Submitted Queue-1 selector:         ${SELECT_JOB_ID} (afterany:${GATE_COLLECT_JOB_ID})"
echo "Submitted Queue-1 full launcher:    ${FULL_QUEUE_JOB_ID} (afterany:${SELECT_JOB_ID})"
