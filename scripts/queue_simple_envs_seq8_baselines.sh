#!/bin/bash

set -euo pipefail

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/simple_envs_seq8}"
GENERIC_ROOT="${GENERIC_ROOT:-${BASE_OUT}/generic_sparse}"
LISTA_ROOT="${LISTA_ROOT:-${BASE_OUT}/lista_best}"
OUT_DIR="${OUT_DIR:-results/simple_envs_seq8_baselines}"
COMPARE_DIR="${COMPARE_DIR:-${OUT_DIR}/comparison}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

K_STRUCTURE="${K_STRUCTURE:-block_diagonal}"
K_BLOCK_SIZE="${K_BLOCK_SIZE:-32}"
LISTA_ALPHA="${LISTA_ALPHA:-0.35}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
LISTA_FINAL_OP="${LISTA_FINAL_OP:-relu}"
LISTA_SPARSITY_COEFF="${LISTA_SPARSITY_COEFF:-1.5}"

UPSTREAM_DEPENDENCY="${UPSTREAM_DEPENDENCY:-}"
SWEEP_DEP_ARGS=()
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  SWEEP_DEP_ARGS=(--dependency="${UPSTREAM_DEPENDENCY}")
fi

GENERIC_JOB_ID=$(sbatch \
  "${SWEEP_DEP_ARGS[@]}" \
  --export=ALL,BASE_OUT="${GENERIC_ROOT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}" \
  scripts/sweep_simple_envs_seq8_generic_sparse.sh | awk '{print $4}')

LISTA_JOB_ID=$(sbatch \
  "${SWEEP_DEP_ARGS[@]}" \
  --export=ALL,BASE_OUT="${LISTA_ROOT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",SEQUENCE_LENGTH="${SEQUENCE_LENGTH}",EVAL_PROFILE="${EVAL_PROFILE}",K_STRUCTURE="${K_STRUCTURE}",K_BLOCK_SIZE="${K_BLOCK_SIZE}",LISTA_ALPHA="${LISTA_ALPHA}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}",LISTA_FINAL_OP="${LISTA_FINAL_OP}",SPARSITY_COEFF="${LISTA_SPARSITY_COEFF}" \
  scripts/sweep_simple_envs_seq8_lista_best.sh | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency=afterany:${GENERIC_JOB_ID}:${LISTA_JOB_ID} \
  --export=ALL,BASE_OUT="${BASE_OUT}",GENERIC_ROOT="${GENERIC_ROOT}",LISTA_ROOT="${LISTA_ROOT}",OUT_DIR="${OUT_DIR}" \
  scripts/collect_simple_envs_seq8_baselines.sh | awk '{print $4}')

COMPARE_JOB_ID=$(sbatch \
  --dependency=afterany:${COLLECT_JOB_ID} \
  --export=ALL,OUT_DIR="${OUT_DIR}",COMPARE_DIR="${COMPARE_DIR}",CANDIDATE_ROOT=lista_best_seq8,ANCHOR_ROOT=generic_sparse \
  scripts/compare_simple_envs_seq8_baselines.sh | awk '{print $4}')

echo "Submitted generic_sparse sweep job: ${GENERIC_JOB_ID}"
echo "Submitted LISTA sweep job: ${LISTA_JOB_ID}"
if [[ -n "${UPSTREAM_DEPENDENCY}" ]]; then
  echo "Sweep dependency: ${UPSTREAM_DEPENDENCY}"
fi
echo "Submitted collect job: ${COLLECT_JOB_ID} (afterany:${GENERIC_JOB_ID}:${LISTA_JOB_ID})"
echo "Submitted compare job: ${COMPARE_JOB_ID} (afterany:${COLLECT_JOB_ID})"
